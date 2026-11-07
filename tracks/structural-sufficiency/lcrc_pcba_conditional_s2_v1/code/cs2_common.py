"""Frozen definitions for the Conditional S2 round v1.

Round: ``lcrc-pcba-conditional-s2-v1``

The question is a *mechanism* question on top of the historical W2 (the 2x
LCRC-A backbone, 994,324 parameters, epoch-40 official Validation AP
0.2618669181):

    does replacing S2's scalar path gate with a low-rank vector gate that is
    conditioned on (path type, path endpoint type) produce a Validation AP gain
    beyond a near-parameter-matched plain FFN?

Two and only two new models are built from the *same untrained W2 seed-0
state*:

``C2`` = W2 + Conditional S2   (new parameters 10,752 -> total 1,005,076)
``F``  = W2 + LocalResidualFFN  (new parameters 10,664 -> total 1,004,988)

Nothing historical is modified; ``bs_common``/``bs_model``/``bs_train`` and the
older rounds are imported read-only.  No official Test row is ever indexed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

CODE_DIR = Path(__file__).resolve().parent
ROUND = CODE_DIR.parent
TRACK = ROUND.parent
REPO = TRACK.parents[1]

BS_ROUND = REPO / "tracks/structural-sufficiency/lcrc_pcba_backbone_scaling_v1"
BS_CODE = BS_ROUND / "code"
if str(BS_CODE) not in sys.path:
    sys.path.insert(0, str(BS_CODE))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import bs_common as BC  # noqa: E402  (frozen recipe, W2 identity, IO helpers)

ANALYSIS = ROUND / "analysis"
RUNS = ROUND / "runs"
TESTS = ROUND / "tests"

ROUND_ID = "lcrc-pcba-conditional-s2-v1"
PROTOCOL_ID = ROUND_ID
SCHEMA_RUN = "lcrc-pcba-conditional-s2-run-v1"
SCHEMA_CHECKPOINT = "lcrc-pcba-conditional-s2-checkpoint-v1"
SCHEMA_AUDIT = "lcrc-pcba-conditional-s2-audit-v1"

# ---------------------------------------------------------------------------
# inherited, read-only identities
# ---------------------------------------------------------------------------

DATASET = BC.DATASET
N_TASKS = BC.N_TASKS
N_MOLECULES = BC.N_MOLECULES
N_TRAIN = BC.N_TRAIN
N_VALID = BC.N_VALID
N_TEST = BC.N_TEST
BATCH_SIZE = BC.BATCH_SIZE
TOTAL_EPOCHS = BC.TOTAL_EPOCHS
STEPS_PER_EPOCH = BC.STEPS_PER_EPOCH
TOTAL_STEPS_40 = BC.TOTAL_STEPS_40

SOURCE = BC.SOURCE
RECIPE = BC.RECIPE
scheduled_lr = BC.scheduled_lr
lr_audit = BC.lr_audit

# ---------------------------------------------------------------------------
# baseline: historical W2, never retrained in this round
# ---------------------------------------------------------------------------

BASE_MODEL_KEY = "W2"
BASE_PARAMETERS = 994_324
BASE_AP40 = 0.26186691813350904  # protocol rounds this to 0.2618669181
BASE_TRAIN_AP40 = 0.5818745570283522
BASE_TRAIN_BCE40 = 0.02628940533386738
BASE_VALID_BCE40 = 0.04653792303564479
BASE_BEST_EPOCH = 24
BASE_BEST_AP = 0.26527331610028143
BASE_GLOBAL_PLAN_SHA256 = "06f437299fa451e449202a990d28f3b8d101e7aec709bcd72a878c8de864b6f6"
BASE_CONSUMED40_PREFIX_SHA256 = "1a24dfca54898dd2b0e7c58d2b2f1956df67b233fdceaa7a376f81a2377a5f58"
BASE_TRUNK_INIT_FINGERPRINT = "5f7101a8f378e7c9283caaade8d61eee733c0c9f3cfb65b5584c8a1e495ef63d"
BASE_FULL_TRUNK_INIT_FINGERPRINT = "109756bd24b2ab8589012031e3cf0b0389b0d275227b35044e9381a45c00a5db"
BASE_INIT_STATE_SHA256 = ""  # filled by analysis/initialization_audit.json at audit time

W2_RUN_DIR = BS_ROUND / "runs" / BASE_MODEL_KEY
W2_RUN_JSON = W2_RUN_DIR / "run.json"
W2_CURVE = W2_RUN_DIR / "train_curve.csv"
W2_PER_TASK = W2_RUN_DIR / "per_task_ap.csv"
W2_TRAIN_EVAL = W2_RUN_DIR / "train_eval_epoch40.json"
W2_CHECKPOINT = W2_RUN_DIR / "checkpoint_epoch40.pt"

#: The frozen W2 width dict (scale 2), passed explicitly to every constructor.
WIDTHS = BC.config_for_scale(2)

# ---------------------------------------------------------------------------
# the two new arms
# ---------------------------------------------------------------------------

COND_RANK = 16
FFN_HIDDEN = 40

NEW_PARAMETERS = {"C2": 10_752, "F": 10_664}
EXPECTED_PARAMETERS = {
    "C2": BASE_PARAMETERS + NEW_PARAMETERS["C2"],
    "F": BASE_PARAMETERS + NEW_PARAMETERS["F"],
}
EXPECTED_TOTALS = {"C2": 1_005_076, "F": 1_004_988}
PARAMETER_GAP = NEW_PARAMETERS["C2"] - NEW_PARAMETERS["F"]

#: Formal parameter accounting of the C2 branch (hard-coded for the audit).
C2_BRANCH = {
    "path_cond_ec": {"vocab": 400, "rank": COND_RANK, "parameters": 400 * COND_RANK},
    "path_cond_ee": {"vocab": 80, "rank": COND_RANK, "parameters": 80 * COND_RANK},
    "path_cond_v": {"rank": COND_RANK, "width": WIDTHS["atom_out"], "parameters": COND_RANK * WIDTHS["atom_out"]},
}
C2_ADDED = sum(int(v["parameters"]) for v in C2_BRANCH.values())

FFN_BRANCH = {
    "norm": {"width": WIDTHS["width"], "parameters": 2 * WIDTHS["width"]},
    "fc1": {
        "in": WIDTHS["width"],
        "out": FFN_HIDDEN,
        "parameters": WIDTHS["width"] * FFN_HIDDEN + FFN_HIDDEN,
    },
    "fc2": {
        "in": FFN_HIDDEN,
        "out": WIDTHS["width"],
        "parameters": FFN_HIDDEN * WIDTHS["width"] + WIDTHS["width"],
    },
}
FFN_ADDED = sum(int(v["parameters"]) for v in FFN_BRANCH.values())

MODELS: dict[str, dict[str, Any]] = {
    "C2": {
        "name": "LCRC-W2-ConditionalS2",
        "role": "conditional_s2_vector_gate",
        "new_parameters": int(NEW_PARAMETERS["C2"]),
        "parameters": int(EXPECTED_PARAMETERS["C2"]),
        "assigned_pool": "res-gpu1",
    },
    "F": {
        "name": "LCRC-W2-LocalResidualFFN",
        "role": "near_parameter_matched_capacity_control",
        "new_parameters": int(NEW_PARAMETERS["F"]),
        "parameters": int(EXPECTED_PARAMETERS["F"]),
        "assigned_pool": "res-gpu0",
    },
}
MODEL_ORDER = ("C2", "F")

SEED = 0
ORDER_SEED = 0
INIT_RULE_C2 = (
    "build the historical W2 seed-0 initial state with bs_model.build_model('W2', seed=0); "
    "copy every original W2 tensor bitwise; E_c/E_e from a private generator "
    "Normal(0, 1/sqrt(16)) (seed*1000003+29); V = 0. Global torch RNG unchanged."
)
INIT_RULE_F = (
    "build the historical W2 seed-0 initial state with bs_model.build_model('W2', seed=0); "
    "copy every original W2 tensor bitwise; LayerNorm default; fc1 standard Linear "
    "initialisation from a private generator (seed*1000003+31); fc2 weight/bias = 0. "
    "Global torch RNG unchanged."
)
PRIVATE_GENERATOR_OFFSETS = {"C2": 29, "F": 31}

#: Validation cadence is not re-derived in this round: the previous round
#: measured a full pass at ~7.2 s against a ~125 s epoch (5.8 %), so the brief
#: fixes every epoch for both arms.
EVAL_CADENCE = {
    "every": 1,
    "resolved": True,
    "source": "frozen by the round brief on the backbone-scaling measurement "
    "(W2 valid 7.15 s / epoch 121.7 s = 5.9 %); not re-derived",
    "every_epoch_1_to_40": True,
    "primary_epoch": TOTAL_EPOCHS,
    "diagnostic_epoch0": True,
}


def calc_eval_epochs(every: int = 1) -> list[int]:
    return [e for e in range(1, TOTAL_EPOCHS + 1) if e % int(every) == 0]


BUDGET = {"gpu_target_hours": 4.0, "gpu_hard_hours": 6.0, "max_concurrent_gpus": 2}

DECISION_RULES = {
    "promising": {
        "label": "PROMISING_CONDITIONAL_S2",
        "delta_c2_min": 0.005,
        "delta_cf_min": 0.003,
        "mean_delta_support_ge20_min": 0.003,
        "improved_support_ge20_min": 60,
        "support_ge20_total": 107,
    },
    "capacity_or_mixed": {"label": "CAPACITY_ONLY_OR_MIXED"},
    "no_clear": {"label": "NO_CLEAR_C2_GAIN", "delta_c2_max_exclusive": 0.002},
    "weak": {"label": "WEAK_OR_MIXED_C2", "delta_c2_min": 0.002, "delta_c2_max": 0.005},
    "regression": {"label": "C2_REGRESSION", "delta_c2_max": -0.002},
    "support_strata": {"ge20": 20, "ge5_lt20": 5},
}

#: Fixed probe sizes for the gradient-reachability audit.
PROBE_WARMUP_STEPS = 1
PROBE_TRAIN_STEPS = 3

# ---------------------------------------------------------------------------
# thin, inherited IO helpers
# ---------------------------------------------------------------------------

utcnow = BC.utcnow
write_json = BC.write_json
read_json = BC.read_json
sha256_file = BC.sha256_file
sha256_bytes = BC.sha256_bytes
log = BC.log
environment = BC.environment
seed_everything = BC.seed_everything
capture_rng_state = BC.capture_rng_state
rng_state_hashes = BC.rng_state_hashes
git_commit = BC.git_commit
git_state = BC.git_state
track_rel = BC.track_rel
host_facts = BC.host_facts
gpu_occupancy = BC.gpu_occupancy
gpu_peak = BC.gpu_peak
state_fingerprint = BC.state_fingerprint
parameter_count = BC.parameter_count
_json_default = BC._json_default

MET = BC.MET
PC = BC.PC
SD = BC.SD
TU = BC.TU
NM = BC.NM


def recipe_for_model(model_key: str) -> dict[str, Any]:
    """The frozen recipe, annotated with this round's arm identity."""
    if model_key not in MODELS:
        raise KeyError(model_key)
    m = MODELS[model_key]
    recipe = dict(RECIPE)
    recipe.update(
        {
            "round": ROUND_ID,
            "model": model_key,
            "model_name": m["name"],
            "role": m["role"],
            "base_model": BASE_MODEL_KEY,
            "seed": int(SEED),
            "epochs": int(TOTAL_EPOCHS),
            "eval_cadence": int(EVAL_CADENCE["every"]),
            "eval_epochs": list(calc_eval_epochs(int(EVAL_CADENCE["every"]))),
            "widths": dict(WIDTHS),
            "base_parameters": int(BASE_PARAMETERS),
            "new_parameters": int(m["new_parameters"]),
            "parameters": int(m["parameters"]),
            "new_modules": {
                "C2": {"E_c": [400, COND_RANK], "E_e": [80, COND_RANK], "V": [COND_RANK, WIDTHS["atom_out"]]},
                "F": {"ffn": [WIDTHS["width"], FFN_HIDDEN, WIDTHS["width"]], "norm": "LayerNorm(128)"},
            }[model_key],
            "init_rule": INIT_RULE_C2 if model_key == "C2" else INIT_RULE_F,
            "lr_schedule": lr_audit(),
            "test_policy": "official Test targets are NaN in the label artifact and are never indexed",
        }
    )
    return recipe


__all__ = [name for name in dir() if not name.startswith("_")]
