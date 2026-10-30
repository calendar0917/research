"""Shared plumbing for LCRC-HC3-v1 (hierarchical three-bond composition).

This module holds paths, frozen constants, hashing, the frozen split, the
frozen recipe and the data boundary only.  No model and no structure
enumeration lives here.

Data boundary
-------------
* official PyG ZINC ``subset=True`` **train** cache: 10k labelled molecules.
* official **valid** cache: 1k, used only for Stage C checkpoint selection and
  analysis (and never for Stage A/B selection, which uses the internal Monitor
  rows).
* official **test** is never resolved: :func:`load_cache` refuses it and every
  path helper below only knows ``train`` / ``valid``.

Internal rows are the frozen ``bjoc_predictive_bottleneck_v1`` split A rows
(Fit8k 8000 / Monitor 1000 / Locked-dev 1000), re-verified against
``split_a.json`` on every load.

Frozen baseline code is imported read-only from
``zinc_local_relation_composition_v1/code``.  The new round never edits it.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

HERE = Path(__file__).resolve().parent          # .../zinc_lcrc_hierarchical_composition_v1/code
ROUND = HERE.parent                             # .../zinc_lcrc_hierarchical_composition_v1
SUFF = ROUND.parent                             # .../structural-sufficiency
REPO = SUFF.parents[1]

LCRC = SUFF / "zinc_local_relation_composition_v1"
LCRC_CODE = LCRC / "code"
LCRC_NORM = LCRC / "analysis" / "norm.json"
BJOC_RES = REPO / "tracks/structure-contribution/results/bjoc_predictive_bottleneck_v1"
SPLIT_FILE = BJOC_RES / "split_a.json"
DCCL_PERMOL = SUFF / "zinc_global_operator_search_v1/analysis/paired_A_vs_dccl_per_molecule.npz"
CYCLE_AUDIT_VALID = REPO / "tracks/ksvd/results/zinc_long_cycle_audit/valid_cycle_audit_label.csv"

CACHE = ROUND / "cache"
RUNS = ROUND / "runs"
ANALYSIS = ROUND / "analysis"
GATES = ANALYSIS / "gates"
CONFIGS = ROUND / "configs"

ROUND_ID = "lcrc-hc3-v1"

#: frozen supervised recipe (identical to the historical DCC-L / LCRC-A recipe)
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

PARAM_CAP = 500_000
PENALTY_Y = -3.0

#: implementation version of the structural cache
STRUCTURE_VERSION = "lcrc-hc3-v1-b"

#: dedicated RNG offset for the new composition modules.  Base construction uses
#: ``torch.manual_seed(seed)`` exactly as the frozen builder; the new modules are
#: drawn afterwards under ``seed + HC_SEED_OFFSET``.
HC_SEED_OFFSET = 771_003


def _bootstrap() -> None:
    for p in (LCRC_CODE, HERE):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


_bootstrap()

import lc_common as LC  # noqa: E402
import lc_data as LD  # noqa: E402
import lc_models as LM  # noqa: E402
import ad_common as AC  # noqa: E402


# ---------------------------------------------------------------------------
# io / provenance
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    print(msg, flush=True)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path | str) -> str:
    return AC.sha256_file(path)


def sha256_array(a: np.ndarray, dtype="<i8") -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a), dtype=dtype).tobytes()).hexdigest()


def state_hash(model) -> str:
    return AC.state_hash(model)


def sha256_state(state: Mapping[str, Any]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode())
        arr = state[key].detach().cpu().numpy() if hasattr(state[key], "detach") \
            else np.asarray(state[key])
        h.update(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
    return h.hexdigest()


def cpu_state(model) -> dict:
    return AC.cpu_state(model)


def load_state(model, state, device) -> None:
    AC.load_state(model, state, device)


def parameter_count(model) -> int:
    return AC.parameter_count(model)


def cosine_with_warmup_lr(step: int, total: int, warmup: int, peak: float, floor: float) -> float:
    return AC.cosine_with_warmup_lr(step, total, warmup, peak, floor)


def write_json(path: Path | str, payload: Any) -> None:
    AC.write_json(path, payload)


def read_json(path: Path | str) -> Any:
    return AC.read_json(path)


def atomic_torch_save(payload: Any, path: Path | str) -> None:
    AC.atomic_torch_save(payload, path)


def atomic_torch_load(path: Path | str):
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def git_commit() -> str:
    return AC.git_commit()


def git_dirty() -> bool:
    return AC.git_dirty()


def environment() -> dict:
    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available() else [],
        "host": socket.gethostname(),
        "git_commit": git_commit(),
        "git_dirty_tracked": git_dirty(),
        "round_id": ROUND_ID,
        "timestamp": now(),
    }


# ---------------------------------------------------------------------------
# data boundary
# ---------------------------------------------------------------------------

_ALLOWED_SPLITS = ("train", "valid")


def load_cache(split: str):
    """Load the frozen LCRC cache for an allowed split.

    ``test`` is refused before touching any file.  The LCRC loader itself also
    refuses unknown split names but would happily accept a path; this wrapper
    makes the boundary explicit.
    """
    if split not in _ALLOWED_SPLITS:
        raise PermissionError(f"split {split!r} is outside the LCRC-HC3 data boundary")
    return LD.load_cache(split)


def cache_path(split: str) -> Path:
    if split not in _ALLOWED_SPLITS:
        raise PermissionError(f"split {split!r} is outside the LCRC-HC3 data boundary")
    return LD.cache_path(split)


def split_rows() -> dict:
    s = read_json(SPLIT_FILE)
    rows = {
        "fit8k": np.asarray(s["fit_rows"], dtype=np.int64),
        "monitor": np.asarray(s["monitor_rows"], dtype=np.int64),
        "locked": np.asarray(s["locked_rows"], dtype=np.int64),
    }
    hashes = {
        "fit8k": s["fit8k_sha256_rows"],
        "monitor": s["monitor_sha256_rows"],
        "locked": s["locked_sha256_rows"],
    }
    return {"rows": rows, "hashes": hashes, "frozen": s}


def verify_split_hashes(rows: Mapping[str, np.ndarray] | None = None) -> dict:
    got = split_rows()
    rows = got["rows"] if rows is None else rows
    out = {}
    for key, value in rows.items():
        digest = sha256_array(value)
        if digest != got["hashes"][key]:
            raise RuntimeError(f"split row hash mismatch for {key}: {digest}")
        out[key] = {"sha256": digest, "n": int(len(value)), "matches_frozen": True}
    return out


def label_hashes(cache: Mapping[str, Any], rows: Mapping[str, np.ndarray]) -> dict:
    y = np.asarray(cache["labels"], dtype=np.float32)
    return {
        "full_labels_sha256": hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest(),
        "cache_meta_labels_sha256": (cache.get("meta") or {}).get("labels_sha256"),
        "per_split": {k: hashlib.sha256(np.ascontiguousarray(y[v]).tobytes()).hexdigest()
                      for k, v in rows.items()},
    }


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------


def fit8k_norm(train_cache: Mapping[str, Any], fit_rows: np.ndarray) -> dict:
    """Fit8k-only normalisation, same rules as the frozen LCRC prep.

    * atom scalars: mean/std over the atoms of the Fit8k molecules.
    * global33: mean/std over the Fit8k molecules.
    No Monitor / Locked / valid / test row contributes.
    """
    fit_rows = np.asarray(fit_rows, dtype=np.int64)
    ap = np.asarray(train_cache["atom_ptr"], dtype=np.int64)
    atom_idx = np.concatenate([np.arange(ap[r], ap[r + 1], dtype=np.int64) for r in fit_rows])
    scalars = np.asarray(train_cache["atom_scalar"], dtype=np.float64)[atom_idx]
    g = np.asarray(train_cache["global"], dtype=np.float64)[fit_rows]
    return {
        "source": f"computed_from_fit8k_rows_n={len(fit_rows)}",
        "fit_split": "fit8k",
        "valid_or_test_used": False,
        "n_fit_rows": int(len(fit_rows)),
        "global_mean": g.mean(0).astype(np.float32),
        "global_scale": np.maximum(g.std(0).astype(np.float32), 1e-6),
        "scalar_mean": scalars.mean(0).astype(np.float32),
        "scalar_scale": np.maximum(scalars.std(0).astype(np.float32), 1e-6),
    }


def load_full_norm() -> dict:
    """The historical full-train10k norm used by the frozen LCRC-A baselines."""
    d = read_json(LCRC_NORM)
    for k in ("global_mean", "global_scale", "scalar_mean", "scalar_scale"):
        d[k] = np.asarray(d[k], dtype=np.float32)
    return d


def norm_fingerprint(norm: Mapping[str, Any]) -> str:
    parts = []
    for k in ("global_mean", "global_scale", "scalar_mean", "scalar_scale"):
        parts.append(np.asarray(norm[k], dtype=np.float32).tobytes())
    return hashlib.sha256(b"".join(parts) + str(norm["source"]).encode()).hexdigest()


# ---------------------------------------------------------------------------
# batch helpers
# ---------------------------------------------------------------------------


def ranges(ptr: np.ndarray, rows: np.ndarray):
    """(flat index, owner slot, per-slot start, counts) for ragged splits."""
    rows = np.asarray(rows, dtype=np.int64)
    starts = np.asarray(ptr, dtype=np.int64)[rows]
    ends = np.asarray(ptr, dtype=np.int64)[rows + 1]
    counts = ends - starts
    idx = np.concatenate([np.arange(s, e, dtype=np.int64) for s, e in zip(starts, ends)]) \
        if len(rows) else np.zeros(0, dtype=np.int64)
    owner = np.repeat(np.arange(len(rows), dtype=np.int64), counts) if len(rows) \
        else np.zeros(0, dtype=np.int64)
    local = np.concatenate([np.arange(c, dtype=np.int64) for c in counts]) if len(rows) \
        else np.zeros(0, dtype=np.int64)
    start = np.cumsum(counts, dtype=np.int64) - counts
    return idx, owner, start, counts


def to_device(batch: Mapping[str, Any], device) -> dict:
    import torch

    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


__all__ = [
    "HERE", "ROUND", "SUFF", "REPO", "LCRC", "LCRC_CODE", "LCRC_NORM", "BJOC_RES",
    "SPLIT_FILE", "DCCL_PERMOL", "CYCLE_AUDIT_VALID", "CACHE", "RUNS", "ANALYSIS",
    "GATES", "CONFIGS", "ROUND_ID", "RECIPE", "PARAM_CAP", "PENALTY_Y",
    "STRUCTURE_VERSION", "HC_SEED_OFFSET", "LC", "LD", "LM", "AC",
    "log", "now", "ensure_dir", "sha256_file", "sha256_array", "state_hash",
    "sha256_state",
    "cpu_state", "load_state", "parameter_count", "cosine_with_warmup_lr",
    "write_json", "read_json", "atomic_torch_save", "atomic_torch_load",
    "git_commit", "git_dirty", "environment", "load_cache", "cache_path",
    "split_rows", "verify_split_hashes", "label_hashes", "fit8k_norm",
    "load_full_norm", "norm_fingerprint", "ranges", "to_device",
]
