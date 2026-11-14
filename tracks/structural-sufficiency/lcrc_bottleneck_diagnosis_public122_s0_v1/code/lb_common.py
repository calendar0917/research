"""Frozen constants for ``lcrc_bottleneck_diagnosis_public122_s0_v1``.

One pre-registered diagnostic round.  Nothing in this module is chosen after
seeing a score: datasets, splits, update budgets, checkpoint grids, selection
rules, the nested data subsets and the readout probes are all written here and
mirrored in ``configs/diagnosis.yaml`` + ``PROTOCOL.md`` before the first formal
run.

Read-only upstream sources (hashed into every run record):

* ``lcrc_performance_route_8h_public122_s0_v1/code`` -- the corrected C1 / K /
  chem8 modules, the corrected D-MPNN recipe (ERRATA E4) and the frozen
  structure/target caches.
* ``lcrc_lipo_model_dissection_v1/code`` -- the frozen C1 -> FULL state remap.
* ``lcrc_molnet_benchmark_pilot_v2/code`` -- split / cache / norm / collate.
* ``lcrc_cross_task_preparation_v1/code`` -- graph build + ``lcrc_data.collate``.

The round measures *fixed-learning-process* sample efficiency, update-budget
response and frozen-readout headroom -- not an architecture search and not a
new SOTA.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROUND_ID = "lcrc-bottleneck-diagnosis-public122-s0-v1"
PROTOCOL_ID = ROUND_ID

CODE_DIR = Path(__file__).resolve().parent
TRACK = CODE_DIR.parent
REPO = TRACK.parents[2]

PERF = REPO / "tracks/structural-sufficiency/lcrc_performance_route_8h_public122_s0_v1"
PERF_CODE = PERF / "code"
DISSECT_CODE = REPO / "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code"
MOLNET_CODE = REPO / "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v2/code"

for _p in (str(CODE_DIR), str(PERF_CODE), str(DISSECT_CODE), str(MOLNET_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import lcpr_common as PC  # noqa: E402  (frozen correction-round constants)
import lcpr_data as PD  # noqa: E402
import lcpr_model as PM  # noqa: E402
import lcpr_train as PT  # noqa: E402
import dissect_common as DC  # noqa: E402
import dissect_model as DM  # noqa: E402

ANALYSIS = TRACK / "analysis"
CONFIGS = TRACK / "configs"
RUNS = TRACK / "runs"
CACHE = TRACK / "cache"
TESTS = TRACK / "tests"
SYNTHESIS = ANALYSIS / "synthesis"

UPSTREAM_CACHE = PERF / "cache"          # dmpnn_*.npz, perms_*, train_norm_*
STRUCT_CACHE = REPO / "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v2/cache"
CHEM8_DIR = REPO / "tracks/structural-sufficiency/lcrc_performance_longrun_v1/analysis/source/chem8"

# ---------------------------------------------------------------------------
# datasets (published splits, verbatim; official Valid/Test never re-split)
# ---------------------------------------------------------------------------

DATASETS: dict[str, dict[str, Any]] = {
    "Lipophilicity": {
        "task": "regression", "split_method": "random122",
        "sizes": (3360, 420, 420), "n_total": 4200,
        "primary": "rmse", "secondary": "mae", "units": "logD",
    },
    "BACE": {
        "task": "binary", "split_method": "scaffold122",
        "sizes": (1210, 151, 152), "n_total": 1513,
        "primary": "roc_auc", "secondary": "pr_auc", "units": "binary class (BACE-1 inhibitor)",
    },
}
PRIMARY_DATASET = "Lipophilicity"
SPLITS = ("train", "valid", "test")
EVAL_BATCH = 4096

TRAIN_SEED = 0

# ---------------------------------------------------------------------------
# recipes
# ---------------------------------------------------------------------------

#: C1 recipe (frozen R1 / R3 of the correction round; identical to lcpr_common)
C1_R1 = {
    "name": "R1", "optimizer": "AdamW", "adam_betas": (0.9, 0.999), "adam_eps": 1e-8,
    "peak_lr": 1.0e-3, "min_lr": 1.0e-5, "weight_decay": 1.0e-3, "batch_size": 128,
    "dropout": 0.10, "gradient_clip": 5.0, "scheduler": "per_step_cosine_fixed_horizon",
}
C1_R3 = {**C1_R1, "name": "R3", "weight_decay": 1.0e-2}

#: D-MPNN recipe = the ERRATA-E4-corrected D2F (LambdaLR base lr == init_lr).
DMPNN_D2F = {
    "name": "D2F", "optimizer": "Adam", "batch_size": 64, "epochs_declared": 100,
    "init_lr": 1.0e-4, "peak_lr": 1.0e-3, "min_lr": 1.0e-4, "weight_decay": 0.0,
    "gradient_clip": 5.0, "warmup_epochs": 2, "decay": "exponential",
    "optimizer_base_lr": 1.0e-4,   # == init_lr: the E4 fix
}

#: C1 update budget: U0 = 360 * ceil(N_official_Train / 128)
#: D-MPNN update budget: U0 = 100 * ceil(N_official_Train / 64)
#: D-MPNN warmup = 2 * ceil(N_official_Train / 64)  (NOT 2 fit epochs)
C1_EPOCHS_PER_HORIZON = 360
DMPNN_EPOCHS_PER_HORIZON = 100
DMPNN_WARMUP_EPOCHS = 2

#: 2U0 is an *observation budget* of the same path as U0, never a second
#: independent training run.
PHASES = ("E1", "E2")          # E1 = [1, U0], E2 = (U0, 2U0]
GRID_POINTS = 50               # 50 candidate updates per phase
WINDOW = 5                     # W_SEL window length
WINDOW_MIN_INDEX = WINDOW - 1  # first 4 points cannot host a full window

# ---------------------------------------------------------------------------
# nested supervision subsets (label-blind, published global row ids)
# ---------------------------------------------------------------------------

SUBSET_SALT = "lcrc-bottleneck-v1"
#: fit fraction -> absolute n on the *official Train* split
FIT_N = {
    "Lipophilicity": {"25": 840, "50": 1680, "100": 3360},
    "BACE": {"50": 605, "100": 1210},
}

#: structure norm (label-free) fitted once on the full official Train structures
#: and shared by every N; regression target norm fitted once on the smallest fit
#: subset (Lipo-25 = 840 rows) and shared by every N / model / phase.
TARGET_NORM_SOURCE = {"Lipophilicity": "25", "BACE": None}   # BACE: raw 0/1

FAMILIES = ("C1_R1", "DMPNN_D2F", "C1_R3", "CHEM8_R1", "K_R1")

#: --- the frozen Lipo main matrix (9 arms) and BACE matrix (4 arms) ---------
def lipo_matrix() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for frac in ("25", "50", "100"):
        out.append({"model": "C1_R1", "fit": frac, "dataset": PRIMARY_DATASET, "phase2": True})
        out.append({"model": "DMPNN_D2F", "fit": frac, "dataset": PRIMARY_DATASET, "phase2": True})
    for model in ("C1_R3", "CHEM8_R1", "K_R1"):
        out.append({"model": model, "fit": "100", "dataset": PRIMARY_DATASET, "phase2": True})
    return out


def bace_matrix() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for frac in ("50", "100"):
        out.append({"model": "C1_R1", "fit": frac, "dataset": "BACE", "phase2": False})
        out.append({"model": "DMPNN_D2F", "fit": frac, "dataset": "BACE", "phase2": False})
    return out


def arm_id(model: str, fit: str, dataset: str) -> str:
    tag = "LIP" if dataset == PRIMARY_DATASET else "BACE"
    return f"{tag}_{model}_F{fit}"


def all_arms() -> list[dict[str, Any]]:
    arms = lipo_matrix() + bace_matrix()
    for a in arms:
        a["arm_id"] = arm_id(a["model"], a["fit"], a["dataset"])
    return arms


#: optional restart arm (only if the pre-registered trigger fires on LIP_C1_R1_F100)
RESTART_ARM = "LIP_C1_R1_F100_RESTART"
RESTART_TRIGGER = {"train_rmse_drop_min": 0.05, "test_gain_max": 0.01}

#: readout probes
PROBE_ARMS = ("P_WARM", "P_POST", "P_PRE")
PROBE_TRAIN_UPDATES = 1800
PROBE_LR = 3.0e-4
PROBE_WEIGHT_DECAY = 1.0e-3
PROBE_BATCH = 128
PROBE_PSI_SEED = 1101
PROBE_NEW_PARAMS = 14705      # psi(64->32->16) + delta(187->64->1); exact, asserted
PROBE_PURCHASE_RULE = ">=1% Lipo Valid AND Test RMSE gain vs P_WARM (both splits)"

# ---------------------------------------------------------------------------
# checkpoint selection rules (all frozen; primary first)
# ---------------------------------------------------------------------------

R_SEL = "R_SEL"   # regression: min Valid MSE; classification: max Valid ROC-AUC
A_SEL = "A_SEL"   # regression: min Valid MAE; classification: min Valid BCE
W_SEL = "W_SEL"   # 5-point rolling window on the primary criterion; take window end
SELECTION_RULES = (R_SEL, A_SEL, W_SEL)

#: purchase / practical-magnitude rulers (not significance laws)
RMSE_GAIN_RULER = 0.01
AUC_GAIN_RULER = 0.005
#: short aliases used by lb_gates / lb_analysis
RMSE_RULER = RMSE_GAIN_RULER
AUC_RULER = AUC_GAIN_RULER
PROBE_MIN_RELATIVE_GAIN = 0.01
RESTART_TRAIN_DROP = RESTART_TRIGGER["train_rmse_drop_min"]
RESTART_TEST_GAIN = RESTART_TRIGGER["test_gain_max"]

BOOTSTRAP_SEED = 20261114
BOOTSTRAP_REPLICATES = 2000

#: budget
WALL_TARGET_HOURS = 5.0
WALL_HARD_HOURS = 6.0
GPU_SOFT_CAP_HOURS = 4.0
GPU_HARD_CAP_HOURS = 6.0
MAX_TRAJECTORIES = 20
MAX_RESTARTS = 1
SMOKE_MAX_EPOCHS = 2

#: fixed reservoirs so a formal run is reproducible in float32
RESERVOIR = ("python", "numpy", "torch_cpu", "torch_cuda")

# ---------------------------------------------------------------------------
# GPU pools (res only, two A100; one process per pool, res-2 and others untouched)
# ---------------------------------------------------------------------------

GPU_POOLS = ("res-gpu0", "res-gpu1")


# ---------------------------------------------------------------------------
# hashing / IO helpers (same conventions as upstream)
# ---------------------------------------------------------------------------


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def row_sha256(rows: Iterable[int]) -> str:
    arr = np.asarray(list(rows), dtype=np.int64)
    return sha256_bytes(np.ascontiguousarray(arr).tobytes())


def subset_key(dataset: str, row: int) -> str:
    """Label-blind nested-subset key: ``sha256('lcrc-bottleneck-v1|<ds>|<row>')``."""
    return sha256_text(f"{SUBSET_SALT}|{dataset}|{int(row)}")


def nested_subset(dataset: str, frac: str, rows: Sequence[int]) -> list[int]:
    """Ascending-by-hash prefix of the official Train rows (returns positions).

    Ties (hash collisions are impossible in practice, but order must be total)
    are broken by the published global row id.  No label, error, group or
    scaffold information enters the selection.
    """
    n_target = FIT_N[dataset][frac]
    rows = [int(r) for r in rows]
    if n_target > len(rows):
        raise ValueError(f"{dataset}: fit {frac} needs {n_target} > {len(rows)} Train rows")
    order = sorted(range(len(rows)), key=lambda i: (subset_key(dataset, rows[i]), rows[i]))
    return sorted(order[:n_target])


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


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)


def write_json(path: Path | str, obj: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return p


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def track_rel(path: Path | str) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO.resolve()))
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


CODE_PATHS = ("code", "configs", "PROTOCOL.md")


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no", "--", *CODE_PATHS],
            cwd=TRACK, text=True, stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except Exception:  # pragma: no cover
        return True


def env_info() -> dict[str, Any]:
    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": np.__version__,
        "hostname": platform.node(),
        "cuda_available": bool(torch.cuda.is_available()),
    }


# ---------------------------------------------------------------------------
# update-budget arithmetic
# ---------------------------------------------------------------------------


def official_train_n(dataset: str) -> int:
    return int(DATASETS[dataset]["sizes"][0])


def steps_per_epoch(model: str, dataset: str) -> int:
    bs = C1_R1["batch_size"] if model.startswith(("C1", "CHEM8", "K")) else DMPNN_D2F["batch_size"]
    return math.ceil(official_train_n(dataset) / bs)


def u0(model: str, dataset: str) -> int:
    """Frozen U0: the update count of the full official-Train horizon."""
    n = official_train_n(dataset)
    if model.startswith(("C1", "CHEM8", "K")):
        return C1_EPOCHS_PER_HORIZON * math.ceil(n / C1_R1["batch_size"])
    return DMPNN_EPOCHS_PER_HORIZON * math.ceil(n / DMPNN_D2F["batch_size"])


def dmpnn_warmup_steps(dataset: str) -> int:
    return DMPNN_WARMUP_EPOCHS * math.ceil(official_train_n(dataset) / DMPNN_D2F["batch_size"])


def grid_updates(model: str, dataset: str) -> dict[str, list[int]]:
    """The common checkpoint grid: 50 points per phase, plus step 0 (measured once).

    E1 = {round(j*U0/50), j=1..50}; Lipo E2 = {U0 + round(j*U0/50), j=1..50}.
    Rounding may map two j to the same integer; duplicates are evaluated once but
    the *phase candidate count* is exactly 50 as pre-registered.
    """
    U0 = u0(model, dataset)
    e1 = [int(round(j * U0 / GRID_POINTS)) for j in range(1, GRID_POINTS + 1)]
    e2: list[int] = []
    if _runs_phase2(dataset):
        e2 = [U0 + int(round(j * U0 / GRID_POINTS)) for j in range(1, GRID_POINTS + 1)]
    return {"E1": e1, "E2": e2}


def max_updates(model: str, dataset: str) -> int:
    return 2 * u0(model, dataset) if _runs_phase2(dataset) else u0(model, dataset)


def _runs_phase2(dataset: str) -> bool:
    return dataset == PRIMARY_DATASET


__all__ = [n for n in dir() if not n.startswith("_")]
