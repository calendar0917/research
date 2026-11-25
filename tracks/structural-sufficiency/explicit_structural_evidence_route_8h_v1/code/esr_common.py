"""Frozen constants, paths and registries for round ``explicit-structural-evidence-route-8h-v1``.

The round studies whether a *non-recursive*, explicit structural prediction
function can be competitive.  Everything here is declared before the first
formal GPU run:

* the four datasets and their frozen splits (Lipo/ESOL = official OGB scaffold,
  Tox21 = the frozen ChemBench split of ``lcrc_predictive_trace_benchmark_v1``,
  PCBA = official OGB scaffold);
* the seven pre-registered arms (BASE, UNARY, SREN-ADD, SREN-EVID, SREN-BANK,
  SREN-COMPOUND, HIDDEN) plus the conditional arms of Phase D;
* the two recipes (R3 for the small datasets, the frozen 40-epoch PCBA recipe);
* the shared structure encoder widths (1x for small datasets, 2x / W2 for PCBA);
* the evidence-aggregation constants (K, tau=1.0, initial pi = [.5,.25,.25]).

Read-only inherited assets (never modified by this round):

* ``lcrc_pcba_backbone_scaling_v1``: ``BSModel`` (the frozen LCRC-A graph at
  explicit widths), ``config_for_scale``, the FULL->C1 remap and the historical
  40-epoch recipe through ``sg_common``;
* ``lcrc_pcba_structural_gain_sprint_v1``: PCBA data/order/normalisation and the
  per-step LR schedule;
* ``lcrc_lipo_context_small_s0_v1``: official-scaffold Lipo caches + labels;
* ``lcrc_explicit_semantic_learning_8h_s0_v1``: official-scaffold ESOL view;
* ``lcrc_molnet_benchmark_pilot_v2``: frozen ChemBench Tox21 caches/targets and
  the multitask metric.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROUND_ID = "explicit-structural-evidence-route-8h-v1"
SCHEMA_RUN = "esr-run-v1"
SCHEMA_CHECKPOINT = "esr-checkpoint-v1"
T0_UTC = "2026-11-25T16:04:18Z"
T0_EPOCH = 1795622658
WALL_HARD_HOURS = 8.0
WALL_TRAIN_STOP_HOURS = 7.25
GPU_SOFT_HOURS = 14.0
GPU_HARD_HOURS = 16.0

CODE_DIR = Path(__file__).resolve().parent
ROUND = CODE_DIR.parent
TRACK = ROUND.parent
REPO = ROUND.parents[2]
ANALYSIS = ROUND / "analysis"
CONFIGS = ROUND / "configs"
RUNS = ROUND / "runs"
CACHE = ROUND / "cache"
FIGURES = ROUND / "figures"
TESTS = ROUND / "tests"
for p in (CODE_DIR,):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# ---------------------------------------------------------------------------
# inherited code directories (read-only)
# ---------------------------------------------------------------------------

SCALE_ROUND = TRACK / "lcrc_pcba_backbone_scaling_v1"
SG_SPRINT = TRACK / "lcrc_pcba_structural_gain_sprint_v1"
NMP_SPRINT = TRACK / "lcrc_pcba_no_mp_sprint_v1"
TU_SCREEN = TRACK / "lcrc_tu_quick_screen_v1"
PREP = TRACK / "lcrc_cross_task_preparation_v1"
LIPO_SMALL = TRACK / "lcrc_lipo_context_small_s0_v1"
MOLNET_V2 = TRACK / "lcrc_molnet_benchmark_pilot_v2"
SE_ROUND = TRACK / "lcrc_explicit_semantic_learning_8h_s0_v1"
PCBA_PILOT = TRACK / "lcrc_pcba_scale_pilot_v1"
LR_PROBE = TRACK / "lcrc_pcba_lr_probe_v1"

for _p in (
    CODE_DIR,
    SCALE_ROUND / "code",
    SG_SPRINT / "code",
    NMP_SPRINT / "code",
    TU_SCREEN / "code",
    PREP / "code",
    LIPO_SMALL / "code",
    MOLNET_V2 / "code",
    SE_ROUND / "code",
    PCBA_PILOT / "code",
    LR_PROBE / "code",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import bs_common as BSC  # noqa: E402
import bs_model as BSM  # noqa: E402
import nmp_model as NM  # noqa: E402  (AttentionPoolReadout of BASE)
import feature_spec as FS  # noqa: E402
import lcrc_data as PREP_DATA  # noqa: E402

# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------

BASE = "BASE"
UNARY = "UNARY"
ADD = "SREN-ADD"
EVID = "SREN-EVID"
BANK = "SREN-BANK"
COMPOUND = "SREN-COMPOUND"
HIDDEN = "HIDDEN"
FLEX = "SREN-BANK-FLEX"
ARMS = (BASE, UNARY, ADD, EVID, BANK, COMPOUND, HIDDEN)
EXPLICIT_ARMS = (ADD, EVID, BANK, COMPOUND)
CONDITIONAL_ARMS = (FLEX,)
ALL_ARMS = ARMS + CONDITIONAL_ARMS

ARM_NOTES = {
    BASE: "original BSModel: local h0 + f + rho + 4-head attention readout (unchanged)",
    UNARY: "real-pair token stream replaced by one SELF token per root (same relation MLP), BANK readout",
    ADD: "task-space scores already projected; sqrt(m) * mean aggregation",
    EVID: "task-space scores; sqrt(m) * (pi_mean*Mean + pi_pos*LME - pi_neg*LME(-)) aggregation",
    BANK: "channel-wise evidence aggregation then task-linear projection",
    COMPOUND: "per-root composite of incident relation tokens, then channel evidence + task-linear projection",
    HIDDEN: "BANK channels + hidden-space nonlinear task combination (capacity control, not a proposed model)",
    FLEX: "BANK channels decomposed into mean / positive-gap / negative-gap with six task read heads",
}

# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------

LIPO = "LIPO"
ESOL = "ESOL"
TOX21 = "TOX21"
PCBA = "PCBA"
SMALL_DATASETS = (LIPO, TOX21, ESOL)
DATASETS = {
    LIPO: {
        "task": "regression",
        "n_tasks": 1,
        "primary_metric": "rmse",
        "direction": "lower",
        "scale": 1,
        "K": 48,
        "epochs": 360,
        "n_train": 3360,
        "n_valid": 420,
        "n_test": 420,
        "split": "ogbg-mollipo official scaffold split (row ids of mol.csv.gz)",
        "units": "logD (raw OGB target)",
    },
    ESOL: {
        "task": "regression",
        "n_tasks": 1,
        "primary_metric": "rmse",
        "direction": "lower",
        "scale": 1,
        "K": 48,
        "epochs": 360,
        "n_train": 902,
        "n_valid": 113,
        "n_test": 113,
        "split": "ogbg-molesol official scaffold split",
        "units": "logS (log10 mol/L)",
    },
    TOX21: {
        "task": "multitask",
        "n_tasks": 12,
        "primary_metric": "macro_roc_auc",
        "direction": "higher",
        "scale": 1,
        "K": 48,
        "epochs": 360,
        "n_train": 6264,
        "n_valid": 783,
        "split": "frozen ChemBench Tox21 split (lcrc_predictive_trace_benchmark_v1 / molnet v2)",
        "units": "observed-label macro ROC-AUC over 12 endpoints",
    },
    PCBA: {
        "task": "multitask",
        "n_tasks": 128,
        "primary_metric": "ap",
        "direction": "higher",
        "scale": 2,
        "K": 64,
        "epochs": 25,
        "epochs_full": 40,
        "n_train": 350343,
        "n_valid": 43793,
        "split": "ogbg-molpcba official scaffold split",
        "units": "official OGB macro AP",
    },
}
DATASET_ORDER = (LIPO, TOX21, ESOL, PCBA)

#: historical PCBA performance coordinates (recorded, never retrained).
PCBA_HISTORY = {
    "W2": {"params": 994_324, "valid_ap": 0.26527331610028143, "test_ap": 0.2536156, "horizon": 40, "best_epoch": 24},
    "FIXED1": {"params": 1_128_468, "valid_ap": 0.2758464, "test_ap": 0.2570020, "horizon": 25, "best_epoch": 25},
    "DYNAMIC": {"params": 1_128_468, "valid_ap": 0.275067, "test_ap": 0.262226, "horizon": 40},
    "W2_prefix25_final": 0.2630573,
    "FIXED1_prefix25_final": 0.2758455,
}
SMALL_HISTORY = {
    "LIPO": {
        "C1_valid_rmse": 0.708624,
        "C1_test_rmse": 0.725392,
        "C1_ADD_valid_rmse": 0.709184,
        "C1_ADD_test_rmse": 0.703645,
        "C1_JOINT_valid_rmse": 0.702272,
        "C1_JOINT_test_rmse": 0.725703,
        "GINE_valid_rmse": 0.6091,
        "GINE_test_rmse": 0.6456,
    },
}

# ---------------------------------------------------------------------------
# model / aggregation constants
# ---------------------------------------------------------------------------

REL_DIM = 19  # 5 distance one-hots (SELF,1,2,3,4) + log1p(common) + 13 raw bond features
DIST_BUCKETS = 5
GLOBAL_EXTRA = 2  # log1p(n_atoms), log1p(n_relation_pairs)
TAU = 1.0
PI_INIT = (0.50, 0.25, 0.25)
DROPOUT = 0.10
PARAM_TOLERANCE = 0.05

#: Phase-D forced pairings of the paired 40-epoch extension.
EXTENSION_PAIRS = {EVID: ADD, BANK: HIDDEN, COMPOUND: BANK, ADD: HIDDEN}

#: Performance levels of the protocol (exploratory, not significance tests).
LEVELS = {
    "level1_delta_ap": 0.005,
    "level2_gap_to_fixed1_dynamic": 0.003,
    "level3_ap": 0.280,
    "level4_ap": 0.290,
}
SMALL_SIGNAL = {"LIPO_rel_rmse": 0.02, "TOX21_macro_auc": 0.005, "ESOL_rel_rmse": 0.02}

# ---------------------------------------------------------------------------
# recipes
# ---------------------------------------------------------------------------

RECIPE_SMALL = {
    "epochs": 360,
    "batch_size": 128,
    "peak_lr": 3.0e-4,
    "min_lr": 1.0e-5,
    "weight_decay": 1.0e-3,
    "adam_betas": (0.9, 0.999),
    "adam_eps": 1e-8,
    "gradient_clip": 5.0,
    "dropout": DROPOUT,
    "scheduler": "per_step_cosine_no_warmup",
    "precision": "fp32",
    "seed": 0,
    "shuffle_offset": 917_000,
}
RECIPE_PCBA = dict(BSC.RECIPE)
PCBA_STEPS_PER_EPOCH = int(BSC.STEPS_PER_EPOCH)
PCBA_TOTAL_STEPS_40 = int(BSC.TOTAL_STEPS_40)
scheduled_lr_pcba = BSC.scheduled_lr


# ---------------------------------------------------------------------------
# IO / hashing helpers
# ---------------------------------------------------------------------------


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def elapsed_hours() -> float:
    return (time.time() - T0_EPOCH) / 3600.0


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True, default=_json_default)
        fh.write("\n")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def state_fingerprint(state: Mapping[str, Any]) -> str:
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode("utf-8"))
        import torch

        arr = np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32)
        h.update(arr.tobytes())
    return h.hexdigest()


def git_commit() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        status = subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True)
        dirty = [ln for ln in status.splitlines() if ln.strip()]
        return {"commit": commit, "dirty": bool(dirty), "dirty_files": dirty[:40]}
    except Exception as exc:  # pragma: no cover
        return {"commit": None, "error": repr(exc)}


def environment() -> dict[str, Any]:
    import torch

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cpu_count": os.cpu_count(),
        "git": git_commit(),
        "captured_utc": utcnow(),
    }


def track_rel(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def log(msg: str) -> None:
    print(f"[{utcnow()}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# run registry / GPU ledger
# ---------------------------------------------------------------------------


def registry_path() -> Path:
    return ANALYSIS / "run_registry.json"


def load_registry() -> dict[str, Any]:
    if registry_path().is_file():
        return read_json(registry_path())
    return {"round_id": ROUND_ID, "runs": {}, "updated_utc": utcnow()}


def register_run(record: Mapping[str, Any]) -> dict[str, Any]:
    reg = load_registry()
    run_id = str(record["run_id"])
    entry = dict(reg["runs"].get(run_id, {}))
    entry.update(dict(record))
    entry["updated_utc"] = utcnow()
    reg["runs"][run_id] = entry
    reg["updated_utc"] = utcnow()
    write_json(registry_path(), reg)
    return entry


def budget_state() -> dict[str, Any]:
    reg = load_registry()
    used = 0.0
    for rec in reg["runs"].values():
        used += float(rec.get("actual_gpu_hours") or 0.0)
    return {
        "wall_elapsed_hours": elapsed_hours(),
        "wall_remaining_hours": WALL_HARD_HOURS - elapsed_hours(),
        "gpu_hours_used": used,
        "gpu_hours_remaining_soft": GPU_SOFT_HOURS - used,
        "gpu_hours_remaining_hard": GPU_HARD_HOURS - used,
        "analysis_reserve_hours": 1.0,
    }


__all__ = [name for name in dir() if not name.startswith("_")]
