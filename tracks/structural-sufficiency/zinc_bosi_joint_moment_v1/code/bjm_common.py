"""Shared plumbing for ``zinc_bosi_joint_moment_v1``.

Bookkeeping only: paths, hashing, the frozen official split caches, the
label-free relation caches, the train-only global-33 standardisation, and the
evaluation loop.  No model code lives here.

Data sources (all frozen, all official PyG ZINC ``subset=True``):

* ``zinc_arch_discovery_v1/cache/ad_train.pt``  — official train, 10,000 rows.
* ``zinc_arch_discovery_v1/cache/ad_valid.pt``  — official valid, 1,000 rows.
  Used **only** for per-epoch evaluation and best-checkpoint selection; never
  for features, standardisation or pretraining.
* ``zinc_arch_discovery_v1/analysis/norm.json`` — global-33 mean/scale computed
  on the official train 10k only (``fit_only=true, n_fit=10000,
  valid_or_test_used=false``); the very same arrays used by P3B and DCC-L.
* ``bjoc_full10k_pretraining_transfer_v1/pretrain_encoder_s{0,1}.pt`` — the
  train-only masked-SSL atom/bond encoders.

The official test cache is never constructed, opened, predicted or scored by
any module in this round.
"""

from __future__ import annotations

import hashlib
import json
import math
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
for _p in (str(HERE), str(BOSI_V0), str(V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402

ROUND_ID = "zinc-bosi-joint-moment-v1"
CONFIGS = ROUND / "configs"
CONFIG = CONFIGS / "bjm_v1.json"
RUNS = ROUND / "runs"
RUNS_SMOKE = ROUND / "runs_smoke"
CACHE = ROUND / "cache"
ANALYSIS = ROUND / "analysis"

DATA_ROOT = SUFF / "zinc_arch_discovery_v1"
DATA_TRAIN = DATA_ROOT / "cache" / "ad_train.pt"
DATA_VALID = DATA_ROOT / "cache" / "ad_valid.pt"
NORM_JSON = DATA_ROOT / "analysis" / "norm.json"
SSL_ROOT = REPO / "tracks/structure-contribution/results/bjoc_full10k_pretraining_transfer_v1"
DCCL_RUNS = SUFF / "zinc_dccl_bottleneck_v1" / "runs"
P3_RUNS = REPO / "tracks/structural-sufficiency/zinc_recipe_trunk_v2/runs"

N_ATOM_FEAT = 108
BOND_FEAT = 7
RANK = 32
N_REL = 4
CLOSURE_DIM = 4
GLOBAL_DIM = 33
WIDTH = 96
SIDE_IN = 2 * WIDTH + RANK + 3 * RANK * N_REL + N_REL + CLOSURE_DIM + 1  # 617
M_DIM = 3 * RANK + 2 * 12 * 12                                        # 384
PARAM_CAP = 500_000

EXPECTED_PARAMETERS = {"B": 473_889, "JM": 475_425, "CM": 475_425}
ARMS = ("B", "JM", "CM")

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
    steps_per_epoch=79,          # ceil(10000 / 128)
    total_updates=47_400,        # 79 * 600
    selection="strict minimum official-valid MAE, earlier epoch wins ties",
)
GLOBAL_BUDGET_A100_H = 5.0


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
    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        arr = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
        h.update(key.encode())
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def rows_hash(rows: Sequence[int]) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(rows, dtype=np.int64)).tobytes()).hexdigest()


def sha256_bytes(data) -> str:
    h = hashlib.sha256()
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    arr = np.ascontiguousarray(np.asarray(data))
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def sha256_array(a, dtype="<f8") -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(a), dtype=dtype).tobytes()).hexdigest()


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
    """The frozen train-10k-only global-33 standardisation."""
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
    return CACHE / f"bjm_rel_{split}.pt"


def load_rel(split: str) -> dict:
    import torch

    path = rel_path(split)
    if not path.exists():
        raise FileNotFoundError(f"missing relation cache {path}; run bjm_prep.py")
    return torch.load(path, map_location="cpu", weights_only=False)


def resolve_run(tag: str) -> Path:
    """Run directory for a frozen tag such as ``B_s0``."""
    return RUNS / tag


def rel_or_abs(path: Path | str) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO.resolve()))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def predict(model, cache, rel, rows: np.ndarray, *, device,
            batch_size: int = 256) -> np.ndarray:
    """Forward-only prediction over a fixed row list."""
    import torch

    import bosi_data as BD

    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(rows), int(batch_size)):
            chunk = np.asarray(rows[start:start + int(batch_size)], dtype=np.int64)
            batch = BD.collate(cache, rel, chunk)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            out.append(model(batch).detach().float().cpu().numpy().reshape(-1))
    return np.concatenate(out, 0) if out else np.zeros(0, dtype=np.float32)


def mae(pred, y) -> float:
    return float(np.mean(np.abs(np.asarray(pred, np.float64)
                               - np.asarray(y, np.float64))))


def curve_metrics(curve: Sequence[Mapping[str, Any]], window: int = 20) -> dict:
    """W20 = min over 20-consecutive-epoch windows of the window median valid MAE.

    T100 = median valid MAE over the last 100 epochs.  Both definitions are
    copied verbatim from the frozen ``zinc_composition_transfer_v1`` analysis.
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
# label-free structural grouping of the official-valid rows
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
    ``zinc_bosi_architecture_v0``: a per-atom string signature of the typed
    radius-1 and radius-2 environment, frequencies counted over the training
    molecules, then unseen / low (1-4) / supported (>=5) fractions.
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
