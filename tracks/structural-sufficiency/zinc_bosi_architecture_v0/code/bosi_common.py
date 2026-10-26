"""Shared paths/utilities for zinc_bosi_architecture_v0.

Everything here is bookkeeping: paths, hashing, the frozen split rows, the
Fit8k-only standardisation and the evaluation loop.  No model code.

Data sources are historical and frozen:

* ``zinc_arch_discovery_v1/cache/ad_train.pt`` — official-train cache (10k rows)
  used by DCC-L.  Fit8k / Monitor / Locked are row subsets of it.
* ``bjoc_predictive_bottleneck_v1/split_a.json`` — frozen row split.
* ``bjoc_full10k_pretraining_transfer_v1/pretrain_encoder_s{0,1}.pt`` — the
  train-only masked-SSL atom/bond encoders used by DCC-L.

Official valid and official test are never loaded by any module in this round.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROUND = _HERE.parent
_SUFF = _ROUND.parent
_REPO = _SUFF.parents[1]
_V1 = _SUFF / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402

ROUND_ID = "zinc-bosi-architecture-v0"
ROUND = _ROUND
# Remote jobs write into a gitignored result directory so the compute checkout
# never becomes dirty; the curated local copy still lives in analysis/.
ANALYSIS = Path(os.environ.get("BOSI_ANALYSIS_DIR", str(_ROUND / "analysis")))
CONFIGS = _ROUND / "configs"
RUNS = _ROUND / "runs"
RUNS_SMOKE = _ROUND / "runs_smoke"
CACHE = _ROUND / "cache"

DATA_CACHE = _SUFF / "zinc_arch_discovery_v1" / "cache" / "ad_train.pt"
BJOC = _REPO / "tracks/structure-contribution/results/bjoc_predictive_bottleneck_v1"
SSL_ROOT = _REPO / "tracks/structure-contribution/results/bjoc_full10k_pretraining_transfer_v1"
DCC_RUNS = _SUFF / "zinc_dccl_bottleneck_v1" / "runs"

PARAM_CAP = 500_000
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
    total_updates=47_400,
)
EXPECTED_PARAMETERS = 473_889


def sha256_file(path: Path | str) -> str:
    return C.sha256_file(path)


def write_json(path: Path | str, payload: Any) -> None:
    C.write_json(path, payload)


def log(msg: str) -> None:
    C.log(msg)


def split_rows() -> dict:
    """The frozen BJOC split A rows (fit8k / fit4k / monitor / locked)."""
    s = json.loads((BJOC / "split_a.json").read_text())
    f4 = json.loads((BJOC / "fit4k_subset.json").read_text())
    return {
        "fit4k": np.asarray(f4["rows"], dtype=np.int64),
        "fit8k": np.asarray(s["fit_rows"], dtype=np.int64),
        "monitor": np.asarray(s["monitor_rows"], dtype=np.int64),
        "locked": np.asarray(s["locked_rows"], dtype=np.int64),
        "hashes": {
            "fit4k": f4["fit4k_sha256_rows"],
            "fit8k": s["fit8k_sha256_rows"],
            "monitor": s["monitor_sha256_rows"],
            "locked": s["locked_sha256_rows"],
        },
    }


def load_train_cache() -> dict:
    import torch

    if not DATA_CACHE.exists():
        raise FileNotFoundError(f"missing frozen official-train cache {DATA_CACHE}")
    return torch.load(DATA_CACHE, map_location="cpu", weights_only=False)


def fit_only_norm(train: Mapping[str, Any], rows: np.ndarray) -> dict:
    """The global-33 standardisation of the frozen protocol (Fit rows only)."""
    g = np.asarray(train["global_feat"], dtype=np.float64)[rows]
    gmean = g.mean(0).astype(np.float32)
    gscale = np.maximum(g.std(0).astype(np.float32), 1e-6)
    return {
        "source": f"computed_from_fit_rows_n={len(rows)}",
        "global_mean": gmean,
        "global_scale": gscale,
        "fit_split": "bjoc-internal-fit",
        "valid_or_test_used": False,
        "n_fit_rows": int(len(rows)),
    }


def ssl_path(seed: int) -> Path:
    return SSL_ROOT / f"pretrain_encoder_s{int(seed)}.pt"


def predict(model, cache, rel, rows: np.ndarray, *, device, batch_size: int = 256,
            arm: str | None = None) -> np.ndarray:
    """Forward-only prediction over a fixed row list (no labels read)."""
    import torch

    import bosi_data as BD

    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(rows), int(batch_size)):
            chunk = np.asarray(rows[start:start + int(batch_size)], dtype=np.int64)
            batch = BD.collate(cache, rel, chunk)
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            if arm is not None:
                old = model.arm
                model.arm = arm
                pred = model(batch)
                model.arm = old
            else:
                pred = model(batch)
            out.append(pred.detach().float().cpu().numpy().reshape(-1))
    return np.concatenate(out, 0) if out else np.zeros(0, dtype=np.float32)


def mae(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred, np.float64) - np.asarray(y, np.float64))))


def labels(cache: Mapping[str, Any], rows: np.ndarray) -> np.ndarray:
    import torch  # noqa: F401  (cache may be a torch-loaded dict)

    return np.asarray(cache["labels"], dtype=np.float64)[np.asarray(rows, dtype=np.int64)]


def molecules_from_cache(cache: Mapping[str, Any]) -> list[tuple[np.ndarray, list]]:
    """[(atom_types, [(u, v, bond_type), ...]), ...] from the frozen cache."""
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


def radius2_support(fit_graphs, query_graphs) -> dict:
    """Fit8k-only support of exact typed radius-2 environments.

    Identical construction to ``zinc_dccl_bottleneck_v1`` (``dca_common.
    radius2_signature_sets``), re-implemented here so the round carries no
    dependency on the audit module's side effects.
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
    for sigs in signature_sets(fit_graphs):
        for s in sigs:
            freq[s] = freq.get(s, 0) + 1
    q = signature_sets(query_graphs)
    unseen, low4, supported, all_support = [], [], [], []
    for sigs in q:
        if not sigs:
            unseen.append(0.0); low4.append(0.0); supported.append(0.0); all_support.append([])
            continue
        c = [freq.get(s, 0) for s in sigs]
        unseen.append(float(np.mean([x == 0 for x in c])))
        low4.append(float(np.mean([1 <= x <= 4 for x in c])))
        supported.append(float(np.mean([x >= 5 for x in c])))
        all_support.append(c)
    return {
        "unseen_frac": np.asarray(unseen),
        "low4_frac": np.asarray(low4),
        "supported_frac": np.asarray(supported),
        "min_support": np.asarray([min(c) if c else 0 for c in all_support]),
        "median_support": np.asarray([float(np.median(c)) if c else 0 for c in all_support]),
        "fit_vocab_size": int(len(freq)),
    }


def support_tier(unseen_frac: np.ndarray, low4_frac: np.ndarray) -> np.ndarray:
    """all_supported / low_support / any_unseen (frozen definition)."""
    tier = np.empty(len(unseen_frac), dtype=object)
    for i in range(len(unseen_frac)):
        if unseen_frac[i] > 0:
            tier[i] = "any_unseen"
        elif low4_frac[i] > 0:
            tier[i] = "low_support"
        else:
            tier[i] = "all_supported"
    return tier


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
