"""Shared plumbing for the structural-binding CPU feasibility gate.

Round: zinc-structural-binding-feasibility-v1.

Only official PyG ZINC subset=True **train** raw graphs are read.  No property
label, no official valid/test, no CUDA.  The frozen LCRC encoding rules are
imported read-only from zinc_local_relation_composition_v1.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROUND_ID = "zinc-structural-binding-feasibility-v1"

CODE_DIR = Path(__file__).resolve().parent
EXP_DIR = CODE_DIR.parent
REPO = EXP_DIR.parents[2]
CACHE = EXP_DIR / "cache"
ANALYSIS = EXP_DIR / "analysis"
LCRC_CODE = REPO / "tracks/structural-sufficiency/zinc_local_relation_composition_v1/code"
ARCH_CODE = REPO / "tracks/structural-sufficiency/zinc_arch_discovery_v1/code"
CONTRIB_CODE = REPO / "tracks/structure-contribution/code"

for _p in (str(LCRC_CODE), str(ARCH_CODE), str(CONTRIB_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import lc_common as L  # noqa: E402
import lc_data as LD  # noqa: E402
import ad_data as AD  # noqa: E402

# Frozen LCRC feature constants.
N_ATOM_TYPES = L.N_ATOM_TYPES                  # 21
N_BOND_TYPES = L.N_BOND_TYPES                  # 3
N_MID_CODES = L.N_PATH_CODES                   # 189 = (mid type, bt1, bt2)
N_END_TYPES = N_ATOM_TYPES                     # 21
MAX_PATHS = 10                                 # observed maximum is 9
N_FOLDS = 5
TRAIN_FOLDS_PER_SPLIT = 3
DEV_FOLDS_PER_SPLIT = 1
TEST_FOLDS_PER_SPLIT = 1

#: deterministic salts; never selected on a model result
SALT_SPLIT = "sbf-v1-split-20261110"
SALT_STANDARD = "sbf-v1-standard-fake-20261110"
SALT_HARD = "sbf-v1-hard-fake-20261110"
SALT_MODEL = "sbf-v1-model-20261110"
SALT_BOOTSTRAP = "sbf-v1-bootstrap-20261110"
SALT_RENUMBER = "sbf-v1-renumber-20261110"
MODEL_SEED_BASE = int(hashlib.sha256(SALT_MODEL.encode()).hexdigest()[:8], 16)

#: frozen LCRC train cache (used only for the independent-rebuild comparison)
FROZEN_LC_TRAIN = L.CACHE / "lc_train.pt"
FROZEN_LC_TRAIN_SHA = "8777494f7745ff5a9a8e3d2913cae8102ad84d267e54fb65235d7eff4c3a6cb2"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_array(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path | str, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=False, ensure_ascii=False, default=_json_default)
    tmp.replace(p)


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


def read_json(path: Path | str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def log(msg: str) -> None:
    print(msg, flush=True)


def git_state() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=str(REPO), capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:
            return None
    return {
        "commit": run("rev-parse", "HEAD"),
        "short": run("rev-parse", "--short", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
    }


def env_state() -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "cuda_available": bool(torch.cuda.is_available()),
        "device": "cpu",
    }


class _Poison:
    """Any access to a property label raises."""

    def _boom(self, *a, **k):
        raise RuntimeError("LABEL_ACCESS: this gate is label-free")

    __getitem__ = __iter__ = __len__ = __array__ = __bool__ = _boom
    shape = property(lambda self: self._boom())
    dtype = property(lambda self: self._boom())


def load_frozen_lc_train() -> dict:
    """Load the frozen LCRC train cache for structural comparison only.

    The cache file contains labels; they are immediately replaced by ``_Poison``.
    No code path in this round reads or evaluates them.
    """
    if not FROZEN_LC_TRAIN.exists():
        raise FileNotFoundError(FROZEN_LC_TRAIN)
    sha = sha256_file(FROZEN_LC_TRAIN)
    if sha != FROZEN_LC_TRAIN_SHA:
        raise RuntimeError(f"frozen LCRC cache sha {sha} != {FROZEN_LC_TRAIN_SHA}")
    data = torch.load(FROZEN_LC_TRAIN, map_location="cpu", weights_only=False)
    if data["meta"].get("feature_version") != L.FEATURE_VERSION:
        raise RuntimeError("unexpected LCRC feature version")
    if "labels" in data:
        data["labels"] = _Poison()
    data["_sha256"] = sha
    return data


__all__ = [
    "ROUND_ID", "CODE_DIR", "EXP_DIR", "REPO", "CACHE", "ANALYSIS",
    "LCRC_CODE", "ARCH_CODE", "CONTRIB_CODE", "L", "LD", "AD",
    "N_ATOM_TYPES", "N_BOND_TYPES", "N_MID_CODES", "N_END_TYPES", "MAX_PATHS",
    "N_FOLDS", "TRAIN_FOLDS_PER_SPLIT", "DEV_FOLDS_PER_SPLIT", "TEST_FOLDS_PER_SPLIT",
    "SALT_SPLIT", "SALT_STANDARD", "SALT_HARD", "SALT_MODEL", "SALT_BOOTSTRAP",
    "SALT_RENUMBER", "MODEL_SEED_BASE", "FROZEN_LC_TRAIN", "FROZEN_LC_TRAIN_SHA",
    "sha256_bytes", "sha256_array", "sha256_file", "write_json", "read_json",
    "now", "log", "git_state", "env_state", "_Poison", "load_frozen_lc_train",
]
