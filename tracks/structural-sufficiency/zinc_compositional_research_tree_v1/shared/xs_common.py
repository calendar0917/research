"""Shared plumbing for the SSRC compositional-research tree v1 (CMA / SPC).

Everything in this module is bookkeeping, hashing, the frozen split, the
frozen recipe and the label-blind feature contract.  No model and no structure
enumeration lives here.

Data boundary
-------------
* ``ad_train.pt``  -- official PyG ZINC ``subset=True`` **train** cache (10k).
* ``ad_valid.pt``  -- official **valid** cache (1k), opened only by Stage IV.
* ``ad_test.pt``   -- official test.  **This file is never opened.**  Any call
  that would resolve to it raises.

Internal rows are the frozen ``bjoc_predictive_bottleneck_v1`` split A rows
(Fit8k 8000 / Monitor 1000 / Locked-dev 1000).  Row hashes are re-verified
against ``split_a.json`` on every load.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent          # .../zinc_compositional_research_tree_v1/shared
ROUND = HERE.parent                             # .../zinc_compositional_research_tree_v1
SUFF = ROUND.parent                             # .../structural-sufficiency
REPO = SUFF.parents[1]

CMA_DIR = ROUND / "cma"
SPC_DIR = ROUND / "spc"
ANALYSIS = ROUND / "analysis"
GATES = ANALYSIS / "gates"
RUNS = ROUND / "runs"
CACHE = ROUND / "cache"

AD_CODE = SUFF / "zinc_arch_discovery_v1" / "code"
BJOC_RES = REPO / "tracks/structure-contribution/results/bjoc_predictive_bottleneck_v1"
SSL_ROOT = REPO / "tracks/structure-contribution/results/bjoc_full10k_pretraining_transfer_v1"
DCCL_RUNS = SUFF / "zinc_dccl_bottleneck_v1" / "runs"
DATA_CACHE_DIR = SUFF / "zinc_arch_discovery_v1" / "cache"

ROUND_ID = "zinc-compositional-research-tree-v1"

TRAIN_CACHE = DATA_CACHE_DIR / "ad_train.pt"
VALID_CACHE = DATA_CACHE_DIR / "ad_valid.pt"
FORBIDDEN_TEST_CACHE = DATA_CACHE_DIR / "ad_test.pt"

#: frozen supervised recipe (identical to the historical DCC-L / BOSI R0 recipe)
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

#: frozen historical same-split reference (DCC-L, Fit8k, best Monitor).
#: ``zinc_dccl_bottleneck_v1/runs/fit8k_s*/train.json :: best_monitor.monitor_mae``
DCCL_FIT8K_MONITOR = {0: 0.12315087235253304, 1: 0.12777156261634082}

#: frozen label-penalty threshold used by every historical SSRC analysis
PENALTY_Y = -3.0

#: implementation version; bump to invalidate structure caches
STRUCTURE_VERSION = "ssrc-v1-a"


# ---------------------------------------------------------------------------
# io helpers
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
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a: np.ndarray, dtype="<i8") -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a), dtype=dtype).tobytes()).hexdigest()


def sha256_state(state: Mapping[str, Any]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode())
        arr = state[key].detach().cpu().numpy() if hasattr(state[key], "detach") else np.asarray(state[key])
        h.update(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
    return h.hexdigest()


def state_hash(model) -> str:
    import torch  # noqa: F401

    return sha256_state({k: v for k, v in model.state_dict().items()})


def _json_default(obj: Any):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def write_json(path: Path | str, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=False, default=_json_default)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_torch_save(payload: Any, path: Path | str) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
            text=True, stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except Exception:
        return True


# ---------------------------------------------------------------------------
# data boundary
# ---------------------------------------------------------------------------


def _refuse_test(path: Path) -> None:
    p = Path(path).resolve()
    if p == FORBIDDEN_TEST_CACHE.resolve() or p.name == "ad_test.pt":
        raise PermissionError("official ZINC test cache must never be opened in this round")


def load_cache(split: str) -> dict:
    """Load one frozen cache.  ``split`` in {train, valid}; test is refused."""
    import torch

    if split == "train":
        path = TRAIN_CACHE
    elif split == "valid":
        path = VALID_CACHE
    elif split == "test":
        _refuse_test(FORBIDDEN_TEST_CACHE)
        raise PermissionError("official ZINC test cache must never be opened in this round")
    else:
        raise ValueError(f"refusing to load cache split {split!r}")
    _refuse_test(path)
    if not path.exists():
        raise FileNotFoundError(f"missing frozen cache {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def split_rows() -> dict:
    """The frozen BJOC split A rows plus their historical hashes."""
    s = read_json(BJOC_RES / "split_a.json")
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
        ok = digest == got["hashes"][key]
        if not ok:
            raise RuntimeError(f"split row hash mismatch for {key}: {digest}")
        out[key] = {"sha256": digest, "n": int(len(value)), "matches_frozen": True}
    return out


def label_hashes(cache: Mapping[str, Any], rows: Mapping[str, np.ndarray]) -> dict:
    y = np.asarray(cache["labels"], dtype=np.float32)
    full = hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest()
    meta = cache.get("meta", {}) or {}
    out = {
        "cache": str(TRAIN_CACHE.relative_to(REPO)) if cache is not None else None,
        "full_labels_sha256": full,
        "cache_meta_labels_sha256": meta.get("labels_sha256"),
        "per_split": {k: hashlib.sha256(np.ascontiguousarray(y[v]).tobytes()).hexdigest()
                      for k, v in rows.items()},
    }
    return out


def ssl_path(seed: int) -> Path:
    return SSL_ROOT / f"pretrain_encoder_s{int(seed)}.pt"


def fit_only_norm(cache: Mapping[str, Any], rows: np.ndarray) -> dict:
    """Fit-only standardisation of the 33-dim global vector (frozen protocol)."""
    g = np.asarray(cache["global_feat"], dtype=np.float64)[np.asarray(rows, dtype=np.int64)]
    return {
        "source": f"computed_from_fit_rows_n={len(rows)}",
        "global_mean": g.mean(0).astype(np.float32),
        "global_scale": np.maximum(g.std(0).astype(np.float32), 1e-6),
        "valid_or_test_used": False,
        "n_fit_rows": int(len(rows)),
    }


# ---------------------------------------------------------------------------
# frozen input contract (gate 9 support)
# ---------------------------------------------------------------------------

ATOM_FEAT_DIM = 108
BOND_FEAT_DIM = 7
GLOBAL_DIM = 33

#: the only cache keys a model forward may consume
FORWARD_INPUT_WHITELIST = frozenset({
    "atom_feat", "bond_feat", "bond_index", "global_feat",
    "atom_batch", "bond_batch", "n_graphs", "n_atoms", "n_bonds",
    "cma_z_atom", "cma_z_bond", "cma_z_chord",
    "cma_path_z", "cma_path_bond", "cma_path_chord",
    "cma_path_batch", "cma_event_bond", "cma_event_n",
    "spc_cv_atom", "spc_cv_batch", "spc_cv_npairs",
    "spc_br_cv", "spc_br_atom_ptr", "spc_br_atom_idx", "spc_br_atom_w",
    "spc_br_bond_ptr", "spc_br_bond_idx", "spc_br_bond_w",
    "spc_br_natoms", "spc_br_nbonds", "spc_br_cyc",
    "spc_pair_br", "spc_pair_cv",
})

#: label-derived quantities that must never reach a forward pass
FORBIDDEN_INPUTS = frozenset({"labels", "y", "logp", "sa", "cycle", "penalty", "label"})


def assert_forward_inputs(batch: Mapping[str, Any]) -> None:
    bad = [k for k in batch if k not in FORWARD_INPUT_WHITELIST]
    if bad:
        raise RuntimeError(f"non-whitelisted forward input(s): {sorted(bad)}")


# ---------------------------------------------------------------------------
# misc numeric helpers
# ---------------------------------------------------------------------------


def cosine_with_warmup_lr(step: int, total: int, warmup: int, peak: float, floor: float) -> float:
    step = int(step)
    if warmup > 0 and step < warmup:
        return peak * float(step + 1) / float(warmup)
    if total <= warmup:
        return floor
    prog = min(1.0, max(0.0, (step - warmup) / float(max(1, total - warmup))))
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * prog))


def parameter_count(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def cpu_state(model) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_state(model, state, device) -> None:
    model.load_state_dict({k: v.to(device) for k, v in state.items()})


# ---------------------------------------------------------------------------
# python path bootstrap for the split source tree
# ---------------------------------------------------------------------------


def add_paths() -> None:
    for p in (HERE, CMA_DIR, SPC_DIR, AD_CODE):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


add_paths()
