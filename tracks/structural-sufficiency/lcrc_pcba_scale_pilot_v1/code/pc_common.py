"""Frozen constants, paths and IO for the PCBA scale pilot.

Round: ``lcrc-pcba-scale-pilot-v1``
Result label: ``SCALE_PILOT``

Everything here was fixed by the execution instruction before any PCBA graph
was built or any model step was taken.  The model definition is *not* copied:
``MODEL_CONFIG`` is imported from the historical round and asserted equal, and
the C1 module itself is built by ``bp_model.build_c1`` (see ``pc_model``).

The dataset is OGB's official ``ogbg-molpcba``.  Nothing in this module (or in
the rest of the round) opens the official Test targets: the label artifact is
written with every Test row set to NaN and the training code never indexes a
Test row.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROUND_ID = "lcrc-pcba-scale-pilot-v1"
PROTOCOL_ID = ROUND_ID
RESULT_LABEL = "SCALE_PILOT"

CODE_DIR = Path(__file__).resolve().parent
ROUND = CODE_DIR.parent
TRACK = ROUND.parent
REPO = TRACK.parents[1]

# Historical round this pilot inherits from (code is read, never modified).
BP = REPO / "tracks/structural-sufficiency/lcrc_benchmark_positioning_fast_v1"
BP_CODE = BP / "code"
BP_CACHE = BP / "cache"
BP_ANALYSIS = BP / "analysis"
if str(BP_CODE) not in sys.path:
    sys.path.insert(0, str(BP_CODE))

import bp_common as BP_COMMON  # noqa: E402  (sets sys.path for prep/tuq/mb2 code)

DATA = REPO / "data"
OGB_ROOT = DATA / "ogb"

ANALYSIS = ROUND / "analysis"
CACHE = ROUND / "cache"
RUNS = ROUND / "runs"
TESTS = ROUND / "tests"
CONFIGS = ROUND / "configs"

#: Official dataset identity.
DATASET = "ogbg-molpcba"
N_TASKS = 128
N_MOLECULES = 437_929
SPLIT_SIZES = {"train": 350_343, "valid": 43_793, "test": 43_793}
SEED = 0

# ---------------------------------------------------------------------------
# frozen model (historical LCRC-C1 / NO_BOND_MEAN)
# ---------------------------------------------------------------------------

MODEL_VARIANT = "C1"
MODEL_CONFIG: dict[str, Any] = dict(BP_COMMON.MODEL_CONFIG)
MODEL_VARIANT_DEFINITION = (
    "LCRC-A minus the 96-d b_mean = b_sum/deg local-input block; explicit S1/S2 "
    "and the global rho readout are unchanged"
)
FROZEN_WIDTHS = {
    "width": 64,
    "stat_rank": 32,
    "local_hidden": 192,
    "atom_hidden": 128,
    "atom_out": 96,
    "bond_hidden": 160,
    "bond_out": 96,
    "rho_hidden": 64,
}

# ---------------------------------------------------------------------------
# frozen recipe (historical OGB recipe; only the stopping rule is new)
# ---------------------------------------------------------------------------

RECIPE: dict[str, Any] = {
    "optimizer": "AdamW",
    "peak_lr": 3.0e-4,
    "weight_decay": 1.0e-3,
    "adam_betas": (0.9, 0.999),
    "adam_eps": 1.0e-8,
    "min_lr": 1.0e-5,
    "dropout": 0.10,
    "batch_size": 128,
    "gradient_clip": 5.0,
    "scheduler": "per_step_cosine",
    "scheduler_horizon_epochs": 360,
    "max_epochs": 40,
    "min_epochs": 12,
    "patience": 10,
    "loss": "masked BCEWithLogitsLoss over observed entries only",
    "selection": "max Validation AP (OGB official Evaluator); ties -> earlier epoch",
    "early_stop": "after min_epochs, stop when 10 consecutive epochs do not strictly improve best Validation AP",
    "test_policy": "the official Test targets are never opened in this round",
}

#: Exact historical OGB recipe values, asserted at import time so a silent edit
#: of the inherited module cannot change this pilot.
FROZEN_OGB_RECIPE = {
    "optimizer": "AdamW",
    "peak_lr": 3.0e-4,
    "weight_decay": 1.0e-3,
    "adam_betas": (0.9, 0.999),
    "adam_eps": 1.0e-8,
    "min_lr": 1.0e-5,
    "dropout": 0.10,
    "batch_size": 128,
    "gradient_clip": 5.0,
    "scheduler": "per_step_cosine",
    "scheduler_horizon_epochs": 360,
}

# ---------------------------------------------------------------------------
# budgets (GPU wall clock of the process holding the device)
# ---------------------------------------------------------------------------

BUDGET: dict[str, Any] = {
    "gpu_hours_total": 4.0,
    "smoke_seconds": 1200.0,          # <= 0.33 h
    "train_soft_seconds": 11880.0,    # 3.30 h -> graceful stop + save
    "train_hard_seconds": 12960.0,    # 3.60 h -> graceful stop + save (watchdog)
    "outer_kill_seconds": 13860.0,    # 3.85 h -> last-resort outer `timeout`
    "cache_build_worker_cap": 16,
    "smoke_train_rows": 8192,
    "smoke_valid_rows": 4096,
    "smoke_epochs": 2,
}

# ---------------------------------------------------------------------------
# correctness-check parameters (fixed, not tuned)
# ---------------------------------------------------------------------------

S2_AUDIT_SAMPLE = 400
HISTORICAL_CROSSCHECK_MOLECULES = 8
INFERENCE_TOL = 1e-5
RELOAD_TOL = 1e-5
GRAD_TOL = 1e-6
AP_AGREEMENT_TOL = 1e-9

LABEL_KEYS = {"label", "labels", "y", "target", "targets", "exp"}


# ---------------------------------------------------------------------------
# hashing / IO
# ---------------------------------------------------------------------------


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_tree(path: Path | str) -> str:
    """Hash a cache directory: sorted ``name<TAB>size<TAB>sha256`` lines."""
    root = Path(path)
    h = hashlib.sha256()
    for f in sorted(root.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(root).as_posix()
        h.update(f"{rel}\t{f.stat().st_size}\t{sha256_file(f)}\n".encode())
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def index_hash(rows: Iterable[int]) -> str:
    arr = np.asarray(list(rows), dtype=np.int64)
    return sha256_bytes(np.ascontiguousarray(arr).tobytes())


def _json_default(obj: Any):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def write_json(path: Path | str, obj: Any, *, compact: bool = False) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    if compact:
        text = json.dumps(obj, separators=(",", ":"), sort_keys=True, default=_json_default)
    else:
        text = json.dumps(obj, indent=2, sort_keys=True, default=_json_default)
    tmp.write_text(text + "\n", encoding="utf-8")
    tmp.replace(p)
    return p


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def track_rel(path: Path | str) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def log(msg: str) -> None:
    print(f"[{utcnow()}] {msg}", flush=True)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # pragma: no cover
        return ""


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except Exception:  # pragma: no cover
        return True


def environment() -> dict[str, Any]:
    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": np.__version__,
        "hostname": platform.node(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
    }


def seed_everything(seed: int) -> dict[str, Any]:
    import random

    import torch

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    return {"seed": int(seed), "deterministic_algorithms": False}


def assert_frozen_inheritance() -> dict[str, Any]:
    """Prove the imported historical constants still equal the frozen values."""
    if MODEL_CONFIG != FROZEN_WIDTHS:
        raise RuntimeError(f"MODEL_CONFIG drifted from the frozen widths: {MODEL_CONFIG}")
    for key, value in FROZEN_OGB_RECIPE.items():
        if BP_COMMON.OGB_RECIPE.get(key) != value:
            raise RuntimeError(
                f"historical OGB_RECIPE[{key!r}]={BP_COMMON.OGB_RECIPE.get(key)!r} != frozen {value!r}"
            )
    if MODEL_VARIANT != "C1":
        raise RuntimeError("this round only runs LCRC-C1")
    return {
        "inherited_module": track_rel(BP_CODE / "bp_common.py"),
        "model_config_equal": True,
        "ogb_recipe_subset_equal": True,
        "model_variant": MODEL_VARIANT,
    }


__all__ = [
    "ANALYSIS",
    "BP",
    "BP_CACHE",
    "BUDGET",
    "CACHE",
    "CODE_DIR",
    "DATASET",
    "MODEL_CONFIG",
    "MODEL_VARIANT",
    "N_MOLECULES",
    "N_TASKS",
    "OGB_ROOT",
    "PROTOCOL_ID",
    "RECIPE",
    "REPO",
    "RESULT_LABEL",
    "ROUND",
    "ROUND_ID",
    "RUNS",
    "SEED",
    "SPLIT_SIZES",
    "assert_frozen_inheritance",
    "environment",
    "git_commit",
    "index_hash",
    "log",
    "read_json",
    "seed_everything",
    "sha256_bytes",
    "sha256_tree",
    "sha256_file",
    "track_rel",
    "utcnow",
    "write_json",
]
