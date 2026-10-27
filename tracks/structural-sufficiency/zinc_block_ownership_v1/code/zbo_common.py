"""Shared plumbing for ``zinc_block_ownership_v1``.

Bookkeeping only: paths, hashing, the frozen official split caches, the
label-free relation/block caches, the train-only global-33 standardisation,
and the evaluation loop.  No model code lives here.

Data sources (all frozen, all official PyG ZINC ``subset=True``):

* ``zinc_arch_discovery_v1/cache/ad_train.pt``  — official train, 10,000 rows.
* ``zinc_arch_discovery_v1/cache/ad_valid.pt``  — official valid, 1,000 rows.
  Used **only** for per-epoch evaluation and best-checkpoint selection; never
  for features, standardisation or pretraining.
* ``zinc_arch_discovery_v1/analysis/norm.json`` — global-33 mean/scale computed
  on the official train 10k only.
* ``bjoc_full10k_pretraining_transfer_v1/pretrain_encoder_s{0,1}.pt`` — the
  train-only masked-SSL atom/bond encoders.

The official test cache is never constructed, opened, predicted or scored by
any module in this round.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
ROUND = HERE.parent
SUFF = ROUND.parent
REPO = SUFF.parents[1]

V1 = SUFF / "zinc_arch_discovery_v1" / "code"
BOSI_V0 = SUFF / "zinc_bosi_architecture_v0" / "code"
BJM = SUFF / "zinc_bosi_joint_moment_v1"
for _p in (str(HERE), str(BOSI_V0), str(V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402

ROUND_ID = "zinc-block-ownership-v1"
CONFIGS = ROUND / "configs"
CONFIG = CONFIGS / "zbo_v1.json"
RUNS = ROUND / "runs"
RUNS_SMOKE = ROUND / "runs_smoke"
CACHE = ROUND / "cache"
ANALYSIS = Path(os.environ.get("ZBO_ANALYSIS_DIR", str(ROUND / "analysis")))

DATA_ROOT = SUFF / "zinc_arch_discovery_v1"
DATA_TRAIN = DATA_ROOT / "cache" / "ad_train.pt"
DATA_VALID = DATA_ROOT / "cache" / "ad_valid.pt"
NORM_JSON = DATA_ROOT / "analysis" / "norm.json"
SSL_ROOT = REPO / "tracks/structure-contribution/results/bjoc_full10k_pretraining_transfer_v1"
BJM_RUNS = BJM / "runs"
DCC_VALID_PRED = (SUFF / "zinc_performance_search_v1" / "analysis"
                  / "valid_pred_B-R0-s0_s0.npz")

WIDTH = 96
RANK = 32
N_REL = 4
CLOSURE_DIM = 4
GLOBAL_DIM = 33
N_ATOM_FEAT = 108
BOND_FEAT = 7
BLOCK_IN = 2 * WIDTH + 4                 # 196
BLOCK_HIDDEN = 64
BLOCK_OUT = 32
READER_OLD_IN = 4 * WIDTH + GLOBAL_DIM   # 417
READER_NEW_IN = READER_OLD_IN + BLOCK_OUT  # 449
PARAM_CAP = 500_000
EXPECTED_PARAMETERS = 497_161
ARMS = ("R", "N", "S")

RECIPE = dict(
    optimizer="adamw",
    lr=1.0e-3,
    min_lr=1.0e-5,
    weight_decay=1.0e-5,
    warmup_steps=500,
    dropout=0.0,
    grad_clip=5.0,
    batch_size=128,
    eval_batch_size=256,
    loss="l1_raw_y",
    precision="fp32",
    epochs=600,
    steps_per_epoch=79,       # ceil(10000 / 128)
    total_updates=47_400,     # 79 * 600
    selection="strict minimum official-valid MAE, earlier epoch wins ties",
)
GPU_SOFT_BUDGET_H = 4.0
GPU_HARD_BUDGET_H = 6.0


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256_file(path: Path | str) -> str:
    return C.sha256_file(path)


def write_json(path: Path | str, payload: Any) -> None:
    C.write_json(path, payload)


def read_json(path: Path | str) -> Any:
    return C.read_json(path)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def state_hash(state: Mapping[str, Any]) -> str:
    """Canonical state hash, matching the historical ``ad_common.state_hash``.

    Keys are sorted and each tensor is hashed as contiguous float32 bytes.  The
    shape is implied by the key; omitting it keeps this hash directly comparable
    with the frozen BOSI/DCC-L initial-state hashes.
    """
    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if hasattr(value, "detach"):
            arr = value.detach().cpu().numpy()
        else:
            arr = np.asarray(value)
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        h.update(key.encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def rows_hash(rows: Sequence[int]) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(rows, dtype=np.int64)).tobytes()).hexdigest()


def rel_or_abs(path: Path | str) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO.resolve()))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def load_cache(split: str) -> dict:
    import torch

    path = {"train": DATA_TRAIN, "valid": DATA_VALID}[split]
    if not path.exists():
        raise FileNotFoundError(f"missing frozen official-{split} cache {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def load_norm() -> dict:
    d = read_json(NORM_JSON)
    if d.get("valid_or_test_used", False) is not False:
        raise RuntimeError(f"refusing norm.json whose provenance is not train-only: {d}")
    return {
        "source": d["source"],
        "global_mean": np.asarray(d["global_mean"], dtype=np.float32),
        "global_scale": np.asarray(d["global_scale"], dtype=np.float32),
        "fit_split": d.get("fit_split"),
        "valid_or_test_used": False,
        "path": rel_or_abs(NORM_JSON),
        "sha256": sha256_file(NORM_JSON),
    }


def ssl_path(seed: int) -> Path:
    return SSL_ROOT / f"pretrain_encoder_s{int(seed)}.pt"


def rel_path(split: str) -> Path:
    return CACHE / f"zbo_rel_{split}.pt"


def block_path(split: str) -> Path:
    return CACHE / f"zbo_blocks_{split}.pt"


def load_rel(split: str) -> dict:
    import torch

    path = rel_path(split)
    if not path.exists():
        raise FileNotFoundError(f"missing relation cache {path}; run zbo_prep.py")
    return torch.load(path, map_location="cpu", weights_only=False)


def load_blocks(split: str) -> dict:
    import torch

    path = block_path(split)
    if not path.exists():
        raise FileNotFoundError(f"missing block cache {path}; run zbo_prep.py")
    return torch.load(path, map_location="cpu", weights_only=False)


def official_valid_labels() -> np.ndarray:
    return np.asarray(load_cache("valid")["labels"], dtype=np.float64)


def official_train_labels() -> np.ndarray:
    return np.asarray(load_cache("train")["labels"], dtype=np.float64)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def predict(model, cache, rel, blocks, rows: np.ndarray, *, device,
            batch_size: int = 256, arm: str | None = None) -> np.ndarray:
    """Forward-only prediction over a fixed row list (no labels)."""
    import torch

    import zbo_blocks as ZB

    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(rows), int(batch_size)):
            chunk = np.asarray(rows[start:start + int(batch_size)], dtype=np.int64)
            batch = ZB.collate(cache, rel, blocks, chunk)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            if arm is not None:
                old = model.block_arm
                model.block_arm = arm
                pred = model(batch)
                model.block_arm = old
            else:
                pred = model(batch)
            out.append(pred.detach().float().cpu().numpy().reshape(-1))
    return np.concatenate(out, 0) if out else np.zeros(0, dtype=np.float32)


def mae(pred, y) -> float:
    return float(np.mean(np.abs(np.asarray(pred, np.float64)
                               - np.asarray(y, np.float64))))


def curve_metrics(curve: Sequence[Mapping[str, Any]], window: int = 20) -> dict:
    """W20 = min over 20-consecutive-epoch windows of the window median valid MAE.

    T100 = median valid MAE over the last 100 epochs.  Definitions are copied
    verbatim from the frozen ``zinc_composition_transfer_v1`` analysis.
    """
    valid = np.asarray([e["valid_mae"] for e in curve], dtype=np.float64)
    epochs = [int(e["epoch"]) for e in curve]
    n = len(valid)
    w20, w20_start = float("inf"), None
    if n >= window:
        for s in range(0, n - window + 1):
            med = float(np.median(valid[s:s + window]))
            if med < w20:
                w20, w20_start = med, epochs[s]
    return {
        "W20": w20 if math.isfinite(w20) else float(np.median(valid)),
        "W20_window_start_epoch": w20_start,
        "T100": float(np.median(valid[-100:])) if n >= 100 else float(np.median(valid)),
        "valid_min": float(valid.min()),
        "valid_last": float(valid[-1]),
        "n_epochs": int(n),
    }


# ---------------------------------------------------------------------------
# label-free structural grouping
# ---------------------------------------------------------------------------


def molecules_from_cache(cache: Mapping[str, Any]) -> list:
    """[(atom_types, [(u, v, bond_type), ...]), ...] from a frozen cache."""
    at = np.asarray(cache["atom_type"], dtype=np.int64)
    bt = np.asarray(cache["bond_type"], dtype=np.int64)
    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    bi = np.asarray(cache["bond_index"], dtype=np.int64)
    out = []
    for i in range(len(ap) - 1):
        a0, a1 = int(ap[i]), int(ap[i + 1])
        b0, b1 = int(bp[i]), int(bp[i + 1])
        out.append((at[a0:a1], [(int(bi[e, 0]), int(bi[e, 1]), int(bt[e]))
                                for e in range(b0, b1)]))
    return out


def radius2_support(train_graphs, query_graphs) -> dict:
    """Exact typed radius-2 support of the query graphs, counted on train only.

    Identical construction to ``zinc_dccl_bottleneck_v1`` /
    ``zinc_bosi_joint_moment_v1``.
    """

    def signature_sets(graphs):
        per_atom = []
        for atoms, bonds in graphs:
            nbrs: list[list[tuple[int, int]]] = [[] for _ in range(len(atoms))]
            for u, v, t in bonds:
                nbrs[u].append((t, v))
                nbrs[v].append((t, u))
            one = [f"r1({int(atoms[i])};"
                   + ",".join(sorted(f"{t}-{int(atoms[j])}" for t, j in nbrs[i])) + ")"
                   for i in range(len(atoms))]
            two = [f"r2({one[i]};"
                   + ",".join(sorted(f"{t}-{one[j]}" for t, j in nbrs[i])) + ")"
                   for i in range(len(atoms))]
            per_atom.append(two)
        return per_atom

    freq: dict[str, int] = {}
    for sigs in signature_sets(train_graphs):
        for s in sigs:
            freq[s] = freq.get(s, 0) + 1
    unseen, low4, supported = [], [], []
    for sigs in signature_sets(query_graphs):
        if not sigs:
            unseen.append(0.0)
            low4.append(0.0)
            supported.append(0.0)
            continue
        c = [freq.get(s, 0) for s in sigs]
        unseen.append(float(np.mean([x == 0 for x in c])))
        low4.append(float(np.mean([1 <= x <= 4 for x in c])))
        supported.append(float(np.mean([x >= 5 for x in c])))
    return {
        "unseen_frac": np.asarray(unseen),
        "low4_frac": np.asarray(low4),
        "supported_frac": np.asarray(supported),
        "train_vocab_size": int(len(freq)),
    }


def support_tier(unseen_frac: np.ndarray, low4_frac: np.ndarray) -> np.ndarray:
    tier = np.empty(len(unseen_frac), dtype=object)
    for i in range(len(unseen_frac)):
        if unseen_frac[i] > 0:
            tier[i] = "any_unseen"
        elif low4_frac[i] > 0:
            tier[i] = "low_support"
        else:
            tier[i] = "all_supported"
    return tier


def cycle_rank(graph) -> int:
    atoms, bonds = graph
    return len(bonds) - len(atoms) + 1


def paton_max_ring(graph) -> int:
    """Largest cycle length in the NetworkX Paton fundamental basis.

    Phase 1e identified this statistic (in the generator's canonical atom
    order) as the ZINC ``-cycles`` label mechanism.  We use it only as a
    structure-only proxy for the penalised subgroup, never as a model input.
    """
    import networkx as nx

    atoms, bonds = graph
    g = nx.Graph()
    g.add_nodes_from(range(len(atoms)))
    g.add_edges_from((int(u), int(v)) for u, v, _ in bonds)
    cycles = nx.cycle_basis(g)
    return int(max((len(c) for c in cycles), default=0))


def fixed_worst10_rows() -> list[int]:
    """Ten official-valid rows with the largest AE under a frozen historical model.

    The anchor is the seed-0 BOSI-B best-valid prediction from
    ``zinc_bosi_joint_moment_v1`` (frozen before this round).  If that file is
    absent we fall back to the frozen DCC-L seed-0 valid prediction.  The row
    list is independent of R/N/S and is written into the config before GPU.
    """
    labels = official_valid_labels()
    path = BJM_RUNS / "B_s0" / "valid_best.npz"
    if not path.exists():
        path = DCC_VALID_PRED
    z = np.load(path)
    rows = np.asarray(z["row"], dtype=np.int64)
    pred = np.asarray(z["pred"], dtype=np.float64)
    if len(rows) != len(labels):
        raise RuntimeError(f"anchor prediction length {len(rows)} != {len(labels)}")
    ae = np.abs(pred - labels[rows])
    order = np.argsort(-ae, kind="stable")[:10]
    return sorted(int(rows[i]) for i in order)


def block_row_features(blocks: Mapping[str, Any]) -> dict:
    """Per-molecule block-count / block-combination features (label-free)."""
    import collections

    bptr = np.asarray(blocks["block_ptr"], dtype=np.int64)
    n_edges = np.asarray(blocks["block_n_edges"], dtype=np.int64)
    sig = np.asarray(blocks["block_sig"], dtype=np.int64)
    n_mol = len(bptr) - 1
    n_blocks = np.zeros(n_mol, dtype=np.int64)
    n_multi = np.zeros(n_mol, dtype=np.int64)
    max_edges = np.zeros(n_mol, dtype=np.int64)
    size_combo: list[tuple] = []
    sig_combo: list[tuple] = []
    for i in range(n_mol):
        g0, g1 = int(bptr[i]), int(bptr[i + 1])
        sizes = [int(x) for x in n_edges[g0:g1]]
        n_blocks[i] = len(sizes)
        n_multi[i] = sum(1 for x in sizes if x >= 2)
        max_edges[i] = max(sizes) if sizes else 0
        size_combo.append(tuple(sorted(sizes)))
        sig_combo.append(tuple(sorted(tuple(int(v) for v in row)
                                      for row in sig[g0:g1])))
    return {
        "n_blocks": n_blocks,
        "n_multi_blocks": n_multi,
        "max_block_edges": max_edges,
        "block_size_combo": size_combo,
        "block_sig_combo": sig_combo,
    }
