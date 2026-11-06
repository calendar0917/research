"""Frozen definitions for the LCRC PCBA continuation-recipe round v1.

This module only names the new round, its two learning-rate continuation
recipes, the frozen source checkpoints and the pre-registered decision rules.
All data, feature, normalization, split, loss and metric definitions are
inherited read-only from the earlier PCBA rounds.

Official Test targets are unreachable from this module: the frozen training
label artifact has every Test row set to NaN and the loader refuses finite Test
labels.  Test labels are read only by ``cc_test.py`` after a prediction freeze.
"""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# paths / inherited code (read-only)
# ---------------------------------------------------------------------------

CODE_DIR = Path(__file__).resolve().parent
ROUND = CODE_DIR.parent
TRACK = ROUND.parent
REPO = TRACK.parents[1]

PILOT = REPO / "tracks/structural-sufficiency/lcrc_pcba_scale_pilot_v1"
PERF = REPO / "tracks/structural-sufficiency/lcrc_pcba_pipeline_perf_v1"
LR_PROBE = REPO / "tracks/structural-sufficiency/lcrc_pcba_lr_probe_v1"
SG_SPRINT = REPO / "tracks/structural-sufficiency/lcrc_pcba_structural_gain_sprint_v1"
NMP_SPRINT = REPO / "tracks/structural-sufficiency/lcrc_pcba_no_mp_sprint_v1"

PILOT_CODE = PILOT / "code"
PERF_CODE = PERF / "code"
LR_PROBE_CODE = LR_PROBE / "code"
SG_CODE = SG_SPRINT / "code"
NMP_CODE = NMP_SPRINT / "code"

for _p in (
    str(PILOT_CODE),
    str(PERF_CODE),
    str(LR_PROBE_CODE),
    str(SG_CODE),
    str(NMP_CODE),
    str(CODE_DIR),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sg_common as SC  # noqa: E402  (frozen inherited definitions)
import nmp_common as NC  # noqa: E402  (frozen no-MP definitions)

ANALYSIS = ROUND / "analysis"
RUNS = ROUND / "runs"
TESTS = ROUND / "tests"
TEST_PREDICTIONS = ANALYSIS / "test_predictions"

ROUND_ID = "lcrc-pcba-continuation-recipe-v1"
PROTOCOL_ID = ROUND_ID
SCHEMA_RUN = "lcrc-pcba-continuation-run-v1"
SCHEMA_CHECKPOINT = "lcrc-pcba-continuation-checkpoint-v1"
SCHEMA_ORDER_PLAN = "lcrc-pcba-continuation-order-plan-v1"
SCHEMA_TEST_FREEZE = "lcrc-pcba-test-prediction-freeze-v1"

DATASET = SC.DATASET
N_TASKS = SC.N_TASKS
N_MOLECULES = SC.N_MOLECULES
N_TRAIN = SC.N_TRAIN
N_VALID = SC.N_VALID
N_TEST = SC.N_TEST
BATCH_SIZE = SC.BATCH_SIZE
STEPS_PER_EPOCH = SC.STEPS_PER_EPOCH  # 2738
SEED = 0

EPOCH40 = 40
EPOCH50 = 50
EPOCH60 = 60
EPOCH70 = 70
EPOCH80 = 80
STEP40 = EPOCH40 * STEPS_PER_EPOCH       # 109,520
STEP50 = EPOCH50 * STEPS_PER_EPOCH       # 136,900
STEP60 = EPOCH60 * STEPS_PER_EPOCH       # 164,280
STEP70 = EPOCH70 * STEPS_PER_EPOCH       # 191,660
STEP80 = EPOCH80 * STEPS_PER_EPOCH       # 219,040

#: Recipe L keeps the 40-epoch cosine endpoint.
LR_FLOOR = 1.0e-5
#: Recipe R warm-restarts to this peak during epoch 41 and cosines back.
LR_PEAK = 5.0e-5
#: First epoch-42 step (exclusive end of the epoch-41 linear warm-up).
RESTART_WARMUP_END = STEP40 + STEPS_PER_EPOCH  # 112,258

#: Shared fixed training configuration (must not change after the freeze).
COMMON_RECIPE: dict[str, Any] = {
    "optimizer": "AdamW",
    "adam_betas": (0.9, 0.999),
    "adam_eps": 1.0e-8,
    "weight_decay": 1.0e-3,
    "dropout": 0.10,
    "batch_size": BATCH_SIZE,
    "gradient_clip": 5.0,
    "precision": "fp32",
    "loss": "masked BCEWithLogitsLoss over observed entries only",
    "early_stopping": False,
    "architecture": "unchanged from the source epoch-40 checkpoint",
    "optimizer_state": "restored AdamW moments, no re-initialisation",
    "batch_size_fixed": True,
    "seed": SEED,
}

# ---------------------------------------------------------------------------
# frozen source checkpoints (read-only)
# ---------------------------------------------------------------------------

SOURCE: dict[str, dict[str, Any]] = {
    "A": {
        "round": "lcrc-pcba-no-mp-sprint-v1",
        "kind": "NMPModel",
        "arm": "A",
        "arm_name": "A_attention_pooling",
        "mode": "attention_pool",
        "path": NMP_SPRINT / "runs/phaseC_seed0_A/checkpoint_epoch40.pt",
        "run_json": NMP_SPRINT / "runs/phaseC_seed0_A/run.json",
        "sha256": "f110935a6055f3c087f9d0b40580506a512cdeb09f6af67a56bde787cc93a437",
        "valid_ap": 0.24635609609821457,
        "state_tensors": 57,
        "optimizer_params": 51,
        "parameters": 287_644,
        "source_commit": "b1b917dc4f72d9815b8f00813a11d95db3868298",
    },
    "G": {
        "round": "lcrc-pcba-no-mp-sprint-v1",
        "kind": "NMPModel",
        "arm": "G",
        "arm_name": "G_graph_context",
        "mode": "graph_context",
        "path": NMP_SPRINT / "runs/phaseC_seed0_G/checkpoint_epoch40.pt",
        "run_json": NMP_SPRINT / "runs/phaseC_seed0_G/run.json",
        "sha256": "e363e3729629f5799f3e2e74ae091ec21e273565e542d939a6bef9bab16849bb",
        "valid_ap": 0.2508481135161762,
        "state_tensors": 53,
        "optimizer_params": 47,
        "parameters": 286_912,
        "source_commit": "b1b917dc4f72d9815b8f00813a11d95db3868298",
    },
    "V": {
        "round": "lcrc-pcba-structural-gain-sprint-v1",
        "kind": "SGModel",
        "arm": "V",
        "arm_name": "V_mp2_graph_context",
        "mode": "mp_context",
        "path": SG_SPRINT / "runs/phase3_seed0_V/checkpoint_epoch40.pt",
        "run_json": SG_SPRINT / "runs/phase3_seed0_V/run.json",
        "sha256": "2d75e4449cc1be669c779fa11b6d9842923965654c880290c430c64e698939a0",
        "valid_ap": 0.25777232477872586,
        "state_tensors": 55,
        "optimizer_params": 49,
        "parameters": 287_808,
        "source_commit": "c39616c3287537f82c70bca85a32c5098e37f56e",
    },
    "C": {
        "round": "lcrc-pcba-structural-gain-sprint-v1",
        "kind": "C1Model",
        "arm": "C",
        "arm_name": "C_control",
        "mode": "none",
        "path": SG_SPRINT / "runs/phase3_seed0_C/checkpoint_epoch40.pt",
        "run_json": SG_SPRINT / "runs/phase3_seed0_C/run.json",
        "sha256": "43ff839505eebcae58cf0a3789f1ee519a958fb1456d779a813c8ca487ea0f4e",
        "valid_ap": 0.23194477365845528,
        "state_tensors": 37,
        "optimizer_params": 31,
        "parameters": 245_184,
        "source_commit": "c39616c3287537f82c70bca85a32c5098e37f56e",
    },
}

MODEL_ORDER = ("A", "G", "V")
TEST_MODEL_ORDER = ("C", "A", "G", "V")
RECIPE_ORDER = ("L", "R")
RECIPES = RECIPE_ORDER
RECIPE_NAMES = {"L": "low_fixed_lr", "R": "lr_restart"}

#: Frozen data identities inherited from the structural-gain sprint.
FEATURE_SPEC_HASH = SC.SOURCE["feature_spec_hash"]
NORM_FINGERPRINT = SC.SOURCE["norm_fingerprint"]
SPLIT_PARTITION_SHA256 = SC.SOURCE["split_partition_sha256"]
LABEL_ARTIFACT_SHA256 = SC.SOURCE["label_artifact_sha256"]
CACHE_SHA256 = SC.SOURCE["cache_sha256"]
HISTORICAL_PLAN_SHA256 = NC.HISTORICAL_PARITY[0]["global_plan_sha256"]

#: Reference AP values used only for audit/replay, never for selection.
HISTORICAL_AP40 = {k: float(v["valid_ap"]) for k, v in SOURCE.items()}

#: Pre-registered decision thresholds.
DECISION = {
    "stage1": {
        "min_gain_over_ap40": 0.002,
        "late_stability_slack": 0.001,
        "task_win_threshold": 70,
        "task_win_min": 0.003,
        "tie_break_ap": 1e-4,
    },
    "stage2": {
        "late_gain_min": 0.003,
    },
    "selection": {
        "tie_break_ap": 1e-4,
    },
    "test": {
        "official_evaluator": "ogb.graphproppred.Evaluator('ogbg-molpcba')",
        "independent": "sklearn average_precision_score per task with NaN masking",
    },
}

BUDGET = {
    "gpu_target_hours": 7.0,
    "gpu_hard_hours": 8.0,
    "wall_target_hours": 5.0,
    "wall_hard_hours": 6.0,
    "max_concurrent_gpus": 2,
    "no_new_gpu_stage_below_minutes": 30,
}

#: Fixed evaluation epochs.
EVAL_EPOCHS_STAGE1 = (EPOCH50, EPOCH60)
EVAL_EPOCHS_STAGE2 = (EPOCH70, EPOCH80)

#: Fixed candidate epochs for final Validation selection.
SELECTION_EPOCHS = (EPOCH40, EPOCH60, EPOCH80)


# ---------------------------------------------------------------------------
# schedules (pure functions of the global optimizer step)
# ---------------------------------------------------------------------------


def scheduled_lr(recipe: str, global_step: int) -> float:
    """Deterministic continuation LR for the step *before* ``optimizer.step``.

    Recipe L
        ``1e-5`` for every step from epoch 41 onward.

    Recipe R
        Step 109,520..112,257 (epoch 41): linear ``1e-5 -> 5e-5``.
        Step 112,258..219,039 (epochs 42..80): per-step cosine from ``5e-5``
        to ``1e-5`` over the epoch-40 -> epoch-80 horizon.  The schedule is a
        function of the global optimizer step only; it is never modified after
        seeing Validation AP.
    """
    s = int(global_step)
    if recipe == "L":
        if s < STEP40:
            raise ValueError(f"Recipe L is only defined for steps >= {STEP40}, got {s}")
        return float(LR_FLOOR)
    if recipe == "R":
        if s < STEP40 or s > STEP80:
            raise ValueError(f"Recipe R is only defined on [{STEP40}, {STEP80}], got {s}")
        if s < RESTART_WARMUP_END:
            frac = (s - STEP40) / float(STEPS_PER_EPOCH - 1)
            return float(LR_FLOOR + (LR_PEAK - LR_FLOOR) * frac)
        frac = min(1.0, (s - RESTART_WARMUP_END) / float(STEP80 - RESTART_WARMUP_END))
        return float(LR_FLOOR + (LR_PEAK - LR_FLOOR) * 0.5 * (1.0 + math.cos(math.pi * frac)))
    raise KeyError(f"unknown recipe {recipe!r}")


def schedule_values(recipe: str, *, first_step: int = STEP40, last_step: int = STEP80) -> np.ndarray:
    steps = np.arange(int(first_step), int(last_step), dtype=np.int64)
    return np.asarray([scheduled_lr(recipe, int(s)) for s in steps], dtype=np.float64)


def schedule_sha256(recipe: str) -> str:
    """SHA256 over every continuation-step LR in ``[STEP40, STEP80)``."""
    values = schedule_values(recipe)
    return sha256_bytes(values.tobytes())


def epoch_lr(recipe: str, epoch: int) -> dict[str, Any]:
    start = (int(epoch) - 1) * STEPS_PER_EPOCH
    end = start + STEPS_PER_EPOCH
    return {
        "epoch": int(epoch),
        "first_step": int(start),
        "last_step": int(end - 1),
        "lr_first": scheduled_lr(recipe, start),
        "lr_last": scheduled_lr(recipe, end - 1),
        "lr_mean": float(np.mean(schedule_values(recipe, first_step=start, last_step=end))),
    }


def schedule_audit() -> dict[str, Any]:
    epochs = list(range(EPOCH40 + 1, EPOCH80 + 1))
    return {
        "schema": "lcrc-pcba-continuation-lr-audit-v1",
        "round_id": ROUND_ID,
        "definitions": {
            "L": "LR = 1e-5 constant for every step from 109520 onward",
            "R": (
                "steps 109520..112257 linear 1e-5->5e-5 (epoch 41); "
                "steps 112258..219039 per-step cosine 5e-5->1e-5 over the "
                "epoch40->epoch80 horizon"
            ),
            "global_step_zero_based": True,
            "function_of_global_step_only": True,
            "optimizer_moments_reset": False,
            "model_parameters_reset": False,
        },
        "constants": {
            "lr_floor": LR_FLOOR,
            "lr_peak": LR_PEAK,
            "step40": STEP40,
            "restart_warmup_end": RESTART_WARMUP_END,
            "step60": STEP60,
            "step80": STEP80,
            "steps_per_epoch": STEPS_PER_EPOCH,
        },
        "endpoints": {
            "L": {
                "step40": scheduled_lr("L", STEP40),
                "step50": scheduled_lr("L", STEP50),
                "step60": scheduled_lr("L", STEP60),
                "step70": scheduled_lr("L", STEP70),
                "step80_minus_1": scheduled_lr("L", STEP80 - 1),
            },
            "R": {
                "step40": scheduled_lr("R", STEP40),
                "epoch41_last": scheduled_lr("R", RESTART_WARMUP_END - 1),
                "epoch42_first": scheduled_lr("R", RESTART_WARMUP_END),
                "step50": scheduled_lr("R", STEP50),
                "step60": scheduled_lr("R", STEP60),
                "step70": scheduled_lr("R", STEP70),
                "step80_minus_1": scheduled_lr("R", STEP80 - 1),
            },
        },
        "per_epoch": {r: [epoch_lr(r, e) for e in epochs] for r in RECIPE_ORDER},
        "schedule_sha256": {r: schedule_sha256(r) for r in RECIPE_ORDER},
        "note": (
            "Recipe R is a learning-rate restart from an already trained epoch-40 "
            "checkpoint; it is not an 80-epoch from-scratch recipe."
        ),
    }


# ---------------------------------------------------------------------------
# helpers (thin wrappers over the inherited, already-audited helpers)
# ---------------------------------------------------------------------------

utcnow = SC.utcnow
write_json = SC.write_json
read_json = SC.read_json
sha256_file = SC.sha256_file
sha256_bytes = SC.sha256_bytes
sha256_int64 = SC.sha256_int64
sha256_batches = SC.sha256_batches
track_rel = SC.track_rel
log = SC.log
environment = SC.environment
capture_rng_state = SC.capture_rng_state
rng_state_hashes = SC.rng_state_hashes
git_commit = SC.git_commit
_json_default = SC._json_default
state_fingerprint = NC.state_fingerprint


def source_for(model: str) -> dict[str, Any]:
    if model not in SOURCE:
        raise KeyError(model)
    return SOURCE[model]


def recipe_config(recipe: str) -> dict[str, Any]:
    if recipe not in RECIPE_ORDER:
        raise KeyError(recipe)
    cfg = dict(COMMON_RECIPE)
    cfg.update(
        {
            "recipe": recipe,
            "recipe_name": RECIPE_NAMES[recipe],
            "lr_start": LR_FLOOR,
            "lr_peak": LR_PEAK if recipe == "R" else LR_FLOOR,
            "lr_end": LR_FLOOR,
            "lr_schedule_sha256": schedule_sha256(recipe),
        }
    )
    return cfg


__all__ = [
    "ANALYSIS",
    "BATCH_SIZE",
    "BUDGET",
    "CACHE_SHA256",
    "COMMON_RECIPE",
    "CODE_DIR",
    "DATASET",
    "DECISION",
    "EPOCH40",
    "EPOCH50",
    "EPOCH60",
    "EPOCH70",
    "EPOCH80",
    "EVAL_EPOCHS_STAGE1",
    "EVAL_EPOCHS_STAGE2",
    "FEATURE_SPEC_HASH",
    "HISTORICAL_AP40",
    "HISTORICAL_PLAN_SHA256",
    "LABEL_ARTIFACT_SHA256",
    "LR_FLOOR",
    "LR_PEAK",
    "MODEL_ORDER",
    "NMP_CODE",
    "NMP_SPRINT",
    "NORM_FINGERPRINT",
    "N_MOLECULES",
    "N_TASKS",
    "N_TRAIN",
    "N_VALID",
    "N_TEST",
    "PERF_CODE",
    "PILOT",
    "PILOT_CODE",
    "PROTOCOL_ID",
    "RECIPES",
    "RECIPE_NAMES",
    "RECIPE_ORDER",
    "REPO",
    "RESTART_WARMUP_END",
    "ROUND",
    "ROUND_ID",
    "RUNS",
    "SCHEMA_CHECKPOINT",
    "SCHEMA_ORDER_PLAN",
    "SCHEMA_RUN",
    "SCHEMA_TEST_FREEZE",
    "SEED",
    "SELECTION_EPOCHS",
    "SG_CODE",
    "SG_SPRINT",
    "SOURCE",
    "SPLIT_PARTITION_SHA256",
    "STEP40",
    "STEP50",
    "STEP60",
    "STEP70",
    "STEP80",
    "STEPS_PER_EPOCH",
    "TEST_MODEL_ORDER",
    "TEST_PREDICTIONS",
    "TESTS",
    "TRACK",
    "capture_rng_state",
    "environment",
    "epoch_lr",
    "git_commit",
    "log",
    "read_json",
    "recipe_config",
    "rng_state_hashes",
    "schedule_audit",
    "schedule_sha256",
    "schedule_values",
    "scheduled_lr",
    "sha256_batches",
    "sha256_bytes",
    "sha256_file",
    "sha256_int64",
    "source_for",
    "state_fingerprint",
    "track_rel",
    "utcnow",
    "write_json",
]
