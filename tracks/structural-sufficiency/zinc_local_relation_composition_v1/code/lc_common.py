"""Shared plumbing for LCRC-v1 (local contribution + relation-aware composition).

This module only provides paths, provenance helpers and the frozen feature
constants.  It contains no model or data logic.  The historical
``zinc_arch_discovery_v1`` helpers are reused read-only for checkpoint I/O,
hashing, the cosine schedule and the official P3 object cache.

The round only touches the official PyG ZINC subset train/valid splits during
preparation and training.  The official test split is not constructed here.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROUND_ID = "zinc-local-relation-composition-v1"

CODE_DIR = Path(__file__).resolve().parent
EXP_DIR = CODE_DIR.parent
REPO = EXP_DIR.parents[2]
CACHE = EXP_DIR / "cache"
RUNS = EXP_DIR / "runs"
ANALYSIS = EXP_DIR / "analysis"
CONFIGS = EXP_DIR / "configs"
PREDICTIONS = EXP_DIR / "predictions"

# Read-only reuse of the frozen architecture-discovery plumbing.
_ARCH = REPO / "tracks/structural-sufficiency/zinc_arch_discovery_v1/code"
if str(_ARCH) not in sys.path:
    sys.path.insert(0, str(_ARCH))

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402,F401

# Make the frozen P3 object-builder modules importable (used for the atom108
# block layout).  This does not load data or labels.
C.ensure_p3_modules()

# Frozen P3 inputs (label-free object cache + train-only masked-SSL encoders).
P3_ROOT = C.P3_ROOT
P3_OBJECTS = {split: getattr(C, f"P3_OBJECTS_{split.upper()}") for split in ("train", "valid")}
P3_PREP = C.P3_PREP
P3_SSL = {0: C.P3_SSL_S0, 1: C.P3_SSL_S1}

ATOM_FEAT_DIM = 108
BOND_FEAT_DIM = 7
GLOBAL_DIM = 33
N_ATOM_TYPES = 21
N_BOND_TYPES = 3
N_NBR_CODES = N_ATOM_TYPES * N_BOND_TYPES          # 63  (neighbour type x bond type)
N_PATH_CODES = N_ATOM_TYPES * N_BOND_TYPES * N_BOND_TYPES  # 189
N_SCALARS = 9
N_REL_CATEGORIES = 6
FEATURE_VERSION = "lcrc-v1-atom108-s1joint-s2typedpath"

#: The six mutually exclusive relation classes.  Distance bucket 0 = exactly 3,
#: 1 = 4..5, 2 = >= 6.  Ring flag = the two atoms share a non-trivial
#: biconnected block (a block with at least three atoms, i.e. a cycle block).
REL_CATEGORY_NAMES = (
    "d3_no_ring",
    "d3_ring",
    "d4_5_no_ring",
    "d4_5_ring",
    "d6plus_no_ring",
    "d6plus_ring",
)


def track_rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except Exception:
        return str(path)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path | str, obj) -> None:
    C.write_json(path, obj)


def read_json(path: Path | str):
    return C.read_json(path)


def now() -> str:
    return C.utcnow()


def log(msg: str) -> None:
    C.log(msg)


def require_cuda(device: str) -> None:
    import torch

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device} but torch.cuda.is_available() is False")


def env_summary() -> dict:
    return C.environment()


def atomic_save(obj, path: Path | str) -> None:
    C.atomic_torch_save(obj, path)


def read_json_or_none(path: Path | str):
    p = Path(path)
    if not p.exists():
        return None
    return C.read_json(p)


def safe_unlink(path: Path | str) -> None:
    p = Path(path)
    if p.exists():
        p.unlink()


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


__all__ = [
    "ROUND_ID", "CODE_DIR", "EXP_DIR", "REPO", "CACHE", "RUNS", "ANALYSIS",
    "CONFIGS", "PREDICTIONS", "P3_ROOT", "P3_OBJECTS", "P3_PREP", "P3_SSL",
    "ATOM_FEAT_DIM", "BOND_FEAT_DIM", "GLOBAL_DIM", "N_ATOM_TYPES",
    "N_BOND_TYPES", "N_NBR_CODES", "N_PATH_CODES", "N_SCALARS",
    "N_REL_CATEGORIES", "REL_CATEGORY_NAMES", "FEATURE_VERSION",
    "track_rel", "write_json", "read_json", "now", "log", "require_cuda",
    "env_summary", "atomic_save", "read_json_or_none", "ensure_dir",
]
