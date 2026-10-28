"""Shared plumbing for the second-hop composition pilot.

The pilot is *read-only* on LCRC-v1: it imports ``lc_common``/``lc_data`` from
``zinc_local_relation_composition_v1/code`` and never writes there.  The only
new on-disk object is a per-path supplement (middle atom + second bond id),
which is regenerated from the frozen cache's raw adjacency and verified against
the cache before use.

Official test is never constructed or loaded anywhere in this round.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROUND_ID = "zinc-secondhop-composition-pilot-v1"

CODE_DIR = Path(__file__).resolve().parent
EXP_DIR = CODE_DIR.parent
REPO = EXP_DIR.parents[2]

LCRC = REPO / "tracks/structural-sufficiency/zinc_local_relation_composition_v1"
LCRC_CODE = LCRC / "code"
if str(LCRC_CODE) not in sys.path:
    sys.path.insert(0, str(LCRC_CODE))

import lc_common as L  # noqa: E402
import lc_data as D  # noqa: E402,F401
import torch  # noqa: E402

CACHE = EXP_DIR / "cache"
RUNS = EXP_DIR / "runs"
ANALYSIS = EXP_DIR / "analysis"
CONFIGS = EXP_DIR / "configs"
CONFIG_PATH = CONFIGS / "shc_v1.json"

# Frozen feature layout of the reused LCRC cache.
BASE_X_DIM = 529
ATOM_OUT = 96
BOND_OUT = 96
WIDTH = 64
FEATURE_VERSION = "shc-v1-A529+B96+C32"
SUPPLEMENT_VERSION = "shc-v1-path-mid-bond2"

SECOND_HOP_DIM = 96
PROJ_DIM = 32
EXTRA_DIM = SECOND_HOP_DIM + PROJ_DIM  # 128
PROJ_SEED = 20261105
PARAM_CAP = 500_000
PARAMS_A = 237_519
PARAMS_NEW = PARAMS_A + EXTRA_DIM * 192  # +24,576 = 262,095

INVARIANCE_TOL = 1e-5
SCALE_RATIO_BAND = (0.5, 2.0)

GROUPS = {
    "ordinary": "y > -3",
    "tail": "y <= -3",
    "row172": 172,
    "fixed_worst10": [917, 753, 172, 659, 260, 678, 31, 224, 447, 229],
    "ring_rich": "cycle_rank >= 4",
}
ANCHOR_CSV = REPO / "tracks/structural-sufficiency/zinc_performance_search_v1/analysis/per_molecule_valid.csv"
WORST10_JSON = REPO / "tracks/structural-sufficiency/zinc_generalization_audit_v1/analysis/fixed_worst10.json"

HISTORICAL = {
    0: {
        "run": "lcrcA_s0",
        "best_valid_mae": 0.07658163924043765,
        "best_epoch": 476,
        "initial_state_sha256": "11c698ae669dbee1bca93ccc13048163e725b29659d54fe6498fa6440a0cfb1e",
        "checkpoint_sha256_prefix": "06340bcd9e45af03",
    },
    1: {
        "run": "lcrcA_s1",
        "best_valid_mae": 0.08050378037145128,
        "best_epoch": 454,
        "initial_state_sha256": "d9422859dc47bbe22496c7bb778f4ca69bd7afb49d704895db40bff7ec269a8e",
        "checkpoint_sha256_prefix": "7225fe11fce0a24d",
    },
}


def historical_dir(seed: int) -> Path:
    return LCRC / "runs" / HISTORICAL[int(seed)]["run"]


def read_json_or_none(path):
    p = Path(path)
    if not p.exists():
        return None
    return L.read_json(p)


def load_config() -> dict:
    return L.read_json(CONFIG_PATH)


def track_rel(path) -> str:
    return L.track_rel(path)


def supplement_path(split: str) -> Path:
    return CACHE / f"sh_{split}.pt"


def supplement_content_hash(sup) -> str:
    """sha256 over the supplement arrays (excludes creation metadata)."""
    import hashlib

    h = hashlib.sha256()
    for key in sorted(sup):
        v = sup[key]
        if torch.is_tensor(v):
            h.update(key.encode())
            h.update(str(v.dtype).encode())
            h.update(str(tuple(v.shape)).encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def projection_hashes(model) -> dict:
    return {
        "P_b_sha256": L.C.sha256_array(model.sh.P_b.detach().cpu().numpy()),
        "P_a_sha256": L.C.sha256_array(model.sh.P_a.detach().cpu().numpy()),
        "P_b_rms": float(model.sh.P_b.pow(2).mean().sqrt().item()),
        "P_a_rms": float(model.sh.P_a.pow(2).mean().sqrt().item()),
    }


__all__ = [
    "ROUND_ID", "CODE_DIR", "EXP_DIR", "REPO", "LCRC", "LCRC_CODE", "L", "D",
    "CACHE", "RUNS", "ANALYSIS", "CONFIGS", "CONFIG_PATH", "BASE_X_DIM",
    "ATOM_OUT", "BOND_OUT", "WIDTH", "FEATURE_VERSION", "SUPPLEMENT_VERSION",
    "SECOND_HOP_DIM", "PROJ_DIM", "EXTRA_DIM", "PROJ_SEED", "PARAM_CAP",
    "PARAMS_A", "PARAMS_NEW", "INVARIANCE_TOL", "SCALE_RATIO_BAND", "GROUPS",
    "ANCHOR_CSV", "WORST10_JSON", "HISTORICAL", "historical_dir",
    "load_config", "read_json_or_none", "track_rel", "supplement_path", "supplement_content_hash",
    "projection_hashes",
]
