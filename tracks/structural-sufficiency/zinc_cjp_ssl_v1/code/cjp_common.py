"""Shared plumbing for CJP-SSL v1 (conditional joint-path self-supervision).

This module contains paths, frozen constants and thin provenance helpers only.
All model/data/training logic lives in ``cjp_data`` / ``cjp_models`` /
``cjp_pretrain`` / ``cjp_sup``.

The frozen LCRC-A reference implementation is imported read-only from
``tracks/structural-sufficiency/zinc_local_relation_composition_v1/code``.
Nothing in that directory is modified by this round.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROUND_ID = "zinc-conditional-joint-path-ssl-v1"

CODE_DIR = Path(__file__).resolve().parent
EXP_DIR = CODE_DIR.parent
REPO = EXP_DIR.parents[2]

CACHE = EXP_DIR / "cache"
RUNS = EXP_DIR / "runs"
ANALYSIS = EXP_DIR / "analysis"
CONFIGS = EXP_DIR / "configs"
PREDICTIONS = EXP_DIR / "predictions"

LC_DIR = REPO / "tracks/structural-sufficiency/zinc_local_relation_composition_v1"
LC_CODE = LC_DIR / "code"
if str(LC_CODE) not in sys.path:
    sys.path.insert(0, str(LC_CODE))

import lc_common as L  # noqa: E402
import lc_data as D  # noqa: E402,F401
import lc_models as M  # noqa: E402
import lc_train as T  # noqa: E402,F401

# ---- frozen vocabulary of the deployed LCRC-A -------------------------------
ATOM_FEAT_DIM = L.ATOM_FEAT_DIM            # 108
BOND_FEAT_DIM = L.BOND_FEAT_DIM            # 7
N_ATOM_TYPES = L.N_ATOM_TYPES              # 21 real categories
N_BOND_TYPES = L.N_BOND_TYPES              # 3 real categories
N_NBR_CODES = L.N_NBR_CODES                # 63
N_PATH_CODES = L.N_PATH_CODES              # 189
N_SCALARS = L.N_SCALARS                    # 9
N_ATOM_SLOTS = 28                          # P3 raw slot count (27 = mask token)
N_BOND_SLOTS = 4                           # P3 raw slot count (0 = mask token)

# ---- masking / target definition (frozen by the task book) -----------------
ATOM_MASK_TOKEN = 27
BOND_MASK_TOKEN = 0
N_TARGET_CLASSES = N_ATOM_TYPES * N_BOND_TYPES          # 63
TARGET_ATOM_CLASSES = N_ATOM_TYPES                      # 21
TARGET_BOND_CLASSES = N_BOND_TYPES                      # 3

# independent RNG streams (mask vs batch/data order) -------------------------
MASK_SEED_TAG = 20261029
BATCH_SEED_TAG = 20261030
MAX_PATHS_PER_MOLECULE = 3

# ---- pretraining recipe (frozen) -------------------------------------------
PRETRAIN = {
    "optimizer": "adamw",
    "lr": 1.0e-3,
    "weight_decay": 1.0e-5,
    "batch_size_molecules": 128,
    "grad_clip": 5.0,
    "updates": 7560,
    "dropout": 0.0,
    "dtype": "fp32",
    "path_loss": "per_molecule_mean_then_batch_mean_of_molecules_with_paths",
    "head": [192, 128, 63],
    "trainable": ["local", "pretext_head"],
    "frozen": ["atom_encoder", "bond_encoder"],
    "excluded": ["f", "rho"],
    "n_fit": 9500,
    "n_diag": 500,
    "head_init_seed_offset": 7717,
}

# ---- supervised recipe: exactly the frozen LCRC-A recipe -------------------
SUPERVISED_RECIPE = dict(T.RECIPE)

DEPLOY_PARAMETERS = 237_519


def now() -> str:
    return L.now()


def log(msg: str) -> None:
    L.log(msg)


def write_json(path, obj) -> None:
    L.write_json(path, obj)


def read_json(path):
    return L.read_json(path)


def track_rel(path) -> str:
    return L.track_rel(path)


def ensure_dir(path) -> Path:
    return L.ensure_dir(path)


def sha256_file(path) -> str:
    return L.C.sha256_file(path)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_array(a: np.ndarray) -> str:
    return L.C.sha256_array(np.asarray(a, dtype=np.float32))


def git_commit() -> str:
    return L.C.git_commit()


def git_dirty() -> bool:
    return L.C.git_dirty()


def state_hash(module: torch.nn.Module) -> str:
    """SHA-256 over the module state dict (sorted keys, float32 bytes)."""
    h = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        h.update(key.encode())
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def module_hashes(model: torch.nn.Module) -> dict[str, str]:
    return {name: state_hash(mod) for name, mod in sorted(model.named_children())}


def tensor_sha(*tensors: torch.Tensor) -> str:
    h = hashlib.sha256()
    for t in tensors:
        a = t.detach().cpu().contiguous().numpy()
        h.update(str(a.dtype).encode())
        h.update(str(tuple(a.shape)).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def gpu_hours(seconds: float) -> float:
    return float(seconds) / 3600.0


__all__ = [
    "ROUND_ID", "CODE_DIR", "EXP_DIR", "REPO", "CACHE", "RUNS", "ANALYSIS",
    "CONFIGS", "PREDICTIONS", "LC_DIR", "LC_CODE", "L", "D", "M", "T",
    "ATOM_FEAT_DIM", "BOND_FEAT_DIM", "N_ATOM_TYPES", "N_BOND_TYPES",
    "N_NBR_CODES", "N_PATH_CODES", "N_SCALARS", "N_ATOM_SLOTS",
    "N_BOND_SLOTS", "ATOM_MASK_TOKEN", "BOND_MASK_TOKEN", "N_TARGET_CLASSES",
    "TARGET_ATOM_CLASSES", "TARGET_BOND_CLASSES", "MASK_SEED_TAG",
    "BATCH_SEED_TAG", "MAX_PATHS_PER_MOLECULE", "PRETRAIN", "SUPERVISED_RECIPE",
    "DEPLOY_PARAMETERS", "now", "log", "write_json", "read_json", "track_rel",
    "ensure_dir", "sha256_file", "sha256_bytes", "sha256_array", "git_commit",
    "git_dirty", "state_hash", "module_hashes", "tensor_sha", "gpu_hours",
]
