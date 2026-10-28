"""Shared paths / imports for the non-local architecture tree v1.

The frozen LCRC-v1 code, feature cache, normalisation and SSL encoders are
reused read-only.  Nothing here builds images or reads labels other than the
official train/valid splits.
"""

from __future__ import annotations

import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
EXP_DIR = CODE_DIR.parent
REPO = EXP_DIR.parents[2]
ANALYSIS = EXP_DIR / "analysis"
CONFIGS = EXP_DIR / "configs"
RUNS = EXP_DIR / "runs"
CACHE = EXP_DIR / "cache"

LCRC_DIR = REPO / "tracks/structural-sufficiency/zinc_local_relation_composition_v1"
LCRC_CODE = LCRC_DIR / "code"
if str(LCRC_CODE) not in sys.path:
    sys.path.insert(0, str(LCRC_CODE))

import lc_common as L  # noqa: E402
import lc_data as D  # noqa: E402
import lc_models as M  # noqa: E402

ROUND_ID = "zinc-nonlocal-architecture-tree-v1"
SEEDS = (0, 1)
LAMBDA_GEF = 0.1           # fixed by analysis/nl_lambda.json, before training
PAD_MAX_ATOMS = 64         # ZINC subset max is 38; padded comfort bound


def ensure_dir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_path(name: str) -> Path:
    return CACHE / f"{name}.pt"


def require_cuda(device: str) -> None:
    import torch

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device} but torch.cuda.is_available() is False")


def train_cache():
    return D.load_cache("train")


def valid_cache():
    return D.load_cache("valid")


def load_norm():
    return D.load_norm()


__all__ = ["CODE_DIR", "EXP_DIR", "REPO", "ANALYSIS", "CONFIGS", "RUNS", "CACHE",
           "LCRC_DIR", "LCRC_CODE", "L", "D", "M", "ROUND_ID", "SEEDS",
           "LAMBDA_GEF", "PAD_MAX_ATOMS", "ensure_dir", "cache_path",
           "require_cuda", "train_cache", "valid_cache", "load_norm"]
