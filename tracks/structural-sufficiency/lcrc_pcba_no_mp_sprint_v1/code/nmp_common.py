"""Frozen definitions for the LCRC PCBA no-edge-MP sprint v1.

All data, normalisation, order, loss, optimiser, schedule and metric definitions
are inherited from the structural-gain sprint; this module only names the new
round, its three arms and its pre-registered decision rules.

Official Test targets are unreachable: ``sg_data.load_task`` refuses finite Test
labels and no code path in this round indexes the Test split.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# paths and inherited code
# ---------------------------------------------------------------------------

CODE_DIR = Path(__file__).resolve().parent
ROUND = CODE_DIR.parent
TRACK = ROUND.parent
REPO = TRACK.parents[1]

PILOT = REPO / "tracks/structural-sufficiency/lcrc_pcba_scale_pilot_v1"
PERF = REPO / "tracks/structural-sufficiency/lcrc_pcba_pipeline_perf_v1"
LR_PROBE = REPO / "tracks/structural-sufficiency/lcrc_pcba_lr_probe_v1"
SG_SPRINT = REPO / "tracks/structural-sufficiency/lcrc_pcba_structural_gain_sprint_v1"

PILOT_CODE = PILOT / "code"
PERF_CODE = PERF / "code"
LR_PROBE_CODE = LR_PROBE / "code"
SG_CODE = SG_SPRINT / "code"

import sys

for _p in (str(PILOT_CODE), str(PERF_CODE), str(LR_PROBE_CODE), str(SG_CODE), str(CODE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sg_common as SC  # noqa: E402  (inherited frozen definitions)
import pc_model as PC  # noqa: E402

ANALYSIS = ROUND / "analysis"
RUNS = ROUND / "runs"
TESTS = ROUND / "tests"

ROUND_ID = "lcrc-pcba-no-mp-sprint-v1"
PROTOCOL_ID = ROUND_ID
SCHEMA_RUN = "lcrc-pcba-no-mp-run-v1"
SCHEMA_CHECKPOINT = "lcrc-pcba-no-mp-checkpoint-v1"

DATASET = SC.DATASET
N_TASKS = SC.N_TASKS
N_MOLECULES = SC.N_MOLECULES
N_TRAIN = SC.N_TRAIN
N_VALID = SC.N_VALID
N_TEST = SC.N_TEST
BATCH_SIZE = SC.BATCH_SIZE
TOTAL_EPOCHS = SC.TOTAL_EPOCHS
STEPS_PER_EPOCH = SC.STEPS_PER_EPOCH
TOTAL_STEPS_40 = SC.TOTAL_STEPS_40

SOURCE = SC.SOURCE  # frozen data/spec/normalisation identities

# ---------------------------------------------------------------------------
# historical references (never retrained in this round)
# ---------------------------------------------------------------------------

HISTORICAL = {
    "C": {
        "seed0_ap40": 0.23194477365845528,
        "seed1_ap40": 0.2269055079836495,
        "mean": 0.2294251408210524,
        "seed0_ap40_10_20_30": [0.20145150856692834, 0.2200291406127242, 0.22957291910653752],
        "seed1_ap40_10_20_30": [0.2006133392775197, 0.2192082817485945, 0.22555328709750214],
        "parameters": 245184,
        "new_parameters": 0,
    },
    "V": {
        "seed0_ap40": 0.25777232477872586,
        "seed1_ap40": 0.25895245725452765,
        "mean": 0.2583623910166268,
        "seed0_ap40_10_20_30": [0.2209033021478183, 0.24694282016950783, 0.25735474145276754],
        "seed1_ap40_10_20_30": [0.2187008824670966, 0.24676577186303533, 0.2565453157665737],
        "parameters": 287808,
        "new_parameters": 42624,
        "role": "historical bond-MP performance reference; NOT a no-MP arm",
    },
}
AP_V_MEAN = HISTORICAL["V"]["mean"]

#: Base-initialisation and order-plan identities recorded by the previous
#: Phase-3 from-scratch runs.  Our new runs must reproduce them bitwise.
HISTORICAL_PARITY = {
    0: {
        "base_init_fingerprint": "1b1f1ad35ab33f4107bc600166fce838d5e16ff5def2348743c73c0211dcf7c3",
        "global_plan_sha256": "06f437299fa451e449202a990d28f3b8d101e7aec709bcd72a878c8de864b6f6",
        "consumed40_prefix_sha256": "1a24dfca54898dd2b0e7c58d2b2f1956df67b233fdceaa7a376f81a2377a5f58",
    },
    1: {
        "base_init_fingerprint": "3af93edb835ea7763cf59479c3484e5def6814f8616ace2b9037a52d2f893d52",
        "global_plan_sha256": "0af6ec6a5dffe9a7af831855e2a15a7b049d97acebda2bc6cdeb96eff56936e0",
        "consumed40_prefix_sha256": "f86b59661de44a51dce6bed4173d219cad5a391f8ee285c16dfbb6c0736c9df6",
    },
}

# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------

ARMS: dict[str, dict[str, Any]] = {
    "S": {
        "name": "S_static_ffn2",
        "role": "strict_no_mp_static",
        "mode": "static_ffn",
        "assigned_pool": "res-gpu0",
        "new_parameters": 41664,
        "total_parameters": 286848,
    },
    "A": {
        "name": "A_attention_pooling",
        "role": "strict_no_mp_attention_readout",
        "mode": "attention_pool",
        "assigned_pool": "res-gpu1",
        "new_parameters": 42460,
        "total_parameters": 287644,
    },
    "G": {
        "name": "G_graph_context",
        "role": "no_bond_mp_graph_context",
        "mode": "graph_context",
        "assigned_pool": "res-gpu0",
        "new_parameters": 41728,
        "total_parameters": 286912,
    },
}
ARM_ORDER = ("S", "A", "G")
BASE_PARAMETERS = 245184
NEW_PARAMETER_TOLERANCE = 0.03
V_NEW_PARAMETERS = HISTORICAL["V"]["new_parameters"]
EXPECTED_PARAMETERS = {
    "C": 245184,
    "S": 286848,
    "A": 287644,
    "G": 286912,
    "V": 287808,
}

#: The original C1 residual width and hidden width.
WIDTH = 64

#: Fixed phases: (first_epoch, final_epoch, eval_epochs).
PHASES: dict[str, dict[str, Any]] = {
    "B": {"first_epoch": 1, "final_epoch": 20, "eval_epochs": (10, 20)},
    "C": {"first_epoch": 21, "final_epoch": 40, "eval_epochs": (40,)},
    "D": {"first_epoch": 1, "final_epoch": 40, "eval_epochs": (10, 20, 30, 40)},
}
RESUME_FROM = {"C": "B"}

#: Inherited frozen scratch recipe (identical object values).
SCRATCH_RECIPE = SC.SCRATCH_RECIPE
RECIPE = SCRATCH_RECIPE

BUDGET = {
    "wall_target_hours": 7.5,
    "wall_hard_hours": 8.0,
    "gpu_target_hours": 15.0,
    "gpu_hard_hours": 16.0,
    "no_new_gpu_stage_below_minutes": 30,
    "max_concurrent_gpus": 2,
}

DECISION = {
    "phase_b": {
        "ap20_floor": 0.220,
        "ap20_minus_ap10_floor": 0.002,
        "label_if_all_below": "EARLY_NO_SIGNAL",
    },
    "phase_c": {
        "ap40_floor": 0.242,
        "label_if_all_below": "NO_STRONG_SEED0_SIGNAL",
    },
    "static_close": {
        "per_seed_ap_min": 0.250,
        "mean_gap_to_v_max": 0.008,
        "per_seed_gain_over_c_min": 0.005,
        "label": "STATIC_CLOSE_TO_V",
    },
    "static_positive": {"label": "STATIC_POSITIVE_BUT_BELOW_V"},
    "no_static_gain": {"label": "NO_STATIC_GAIN"},
    "incomplete": {"label": "INCOMPLETE"},
    "global_communication": {"label": "GLOBAL_COMMUNICATION_PROMISING"},
    "v_retains": {"label": "EDGE_MP_BUNDLE_RETAINS_ADVANTAGE"},
    "second_seed_skip": "SECOND_SEED_NOT_RUN_RESOURCE_LIMIT",
}

#: Fixed smoke sizes.
PREFLIGHT_SMOKE_STEPS = 30
PREFLIGHT_GRAD_STEPS = 3
PREFLIGHT_IDENTITY_MOLECULES = 8
ATTENTION_DIAGNOSTIC_MOLECULES = 64


# ---------------------------------------------------------------------------
# schedule (inherited phase-3 rule)
# ---------------------------------------------------------------------------


def scheduled_lr(global_step: int) -> float:
    return float(SC.scheduled_lr(int(global_step), phase=3, kind="all"))


def lr_audit() -> dict[str, Any]:
    steps = STEPS_PER_EPOCH
    return {
        "steps_per_epoch": steps,
        "total_steps_40_epochs": TOTAL_STEPS_40,
        "step0": scheduled_lr(0),
        "warmup_last": scheduled_lr(steps - 1),
        "post_warmup_first": scheduled_lr(steps),
        "step10e": scheduled_lr(10 * steps),
        "step20e": scheduled_lr(20 * steps),
        "step40e": scheduled_lr(40 * steps),
        "formula": (
            "steps 0..2737 linear 3e-5->3e-4; then "
            "1e-5+(3e-4-1e-5)*0.5*(1+cos(pi*(s-2738)/(109520-2738)))"
        ),
    }


def recipe_for_arm(arm: str, phase: str) -> dict[str, Any]:
    if arm not in ARMS:
        raise KeyError(arm)
    p = PHASES[phase]
    recipe = dict(SCRATCH_RECIPE)
    recipe.update(
        {
            "round": ROUND_ID,
            "arm": arm,
            "arm_name": ARMS[arm]["name"],
            "role": ARMS[arm]["role"],
            "mode": ARMS[arm]["mode"],
            "phase": phase,
            "first_epoch": p["first_epoch"],
            "final_epoch": p["final_epoch"],
            "eval_epochs": list(p["eval_epochs"]),
            "lr_schedule": lr_audit(),
        }
    )
    return recipe


# ---------------------------------------------------------------------------
# helpers
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
seed_everything = SC.seed_everything
capture_rng_state = SC.capture_rng_state
rng_state_hashes = SC.rng_state_hashes
git_commit = SC.git_commit
_json_default = SC._json_default
state_fingerprint = PC.state_fingerprint


def parameter_count(model: Any) -> int:
    return int(sum(p.numel() for p in model.parameters()))


__all__ = [
    "ANALYSIS",
    "AP_V_MEAN",
    "ARMS",
    "ARM_ORDER",
    "ATTENTION_DIAGNOSTIC_MOLECULES",
    "BASE_PARAMETERS",
    "BATCH_SIZE",
    "BUDGET",
    "CODE_DIR",
    "DATASET",
    "DECISION",
    "EXPECTED_PARAMETERS",
    "HISTORICAL",
    "HISTORICAL_PARITY",
    "LR_PROBE",
    "LR_PROBE_CODE",
    "N_MOLECULES",
    "N_TASKS",
    "N_TRAIN",
    "N_VALID",
    "NEW_PARAMETER_TOLERANCE",
    "PERF",
    "PERF_CODE",
    "PHASES",
    "PILOT",
    "PILOT_CODE",
    "PREFLIGHT_GRAD_STEPS",
    "PREFLIGHT_IDENTITY_MOLECULES",
    "PREFLIGHT_SMOKE_STEPS",
    "PROTOCOL_ID",
    "RECIPE",
    "REPO",
    "ROUND",
    "ROUND_ID",
    "RUNS",
    "SCHEMA_CHECKPOINT",
    "SCHEMA_RUN",
    "SCRATCH_RECIPE",
    "SG_CODE",
    "SG_SPRINT",
    "SOURCE",
    "STEPS_PER_EPOCH",
    "TESTS",
    "TOTAL_EPOCHS",
    "TOTAL_STEPS_40",
    "TRACK",
    "V_NEW_PARAMETERS",
    "WIDTH",
    "capture_rng_state",
    "environment",
    "git_commit",
    "log",
    "lr_audit",
    "parameter_count",
    "read_json",
    "recipe_for_arm",
    "rng_state_hashes",
    "scheduled_lr",
    "seed_everything",
    "sha256_batches",
    "sha256_bytes",
    "sha256_file",
    "sha256_int64",
    "state_fingerprint",
    "track_rel",
    "utcnow",
    "write_json",
]
