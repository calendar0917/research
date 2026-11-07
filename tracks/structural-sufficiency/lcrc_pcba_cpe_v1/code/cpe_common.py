"""Frozen definitions for the Cross-atom Predictive Evidence round (CPE v1).

Round: ``lcrc-pcba-cpe-v1``

Question: on the historical W2 backbone (2x LCRC-A, 994,324 parameters,
epoch-40 official Validation AP 0.2618669181), does an **explicit second-order
predictive-evidence statistic between different atoms** beat the
*identical-parameter, identical-init* control that uses the *same-atom*
product?

Two and only two arms, both built from the same untrained W2 seed-0 state:

``P`` = W2 + cross-atom evidence (``pair_cross``)   33,408 new -> 1,027,732
``S`` = W2 + same-atom  evidence (``pair_self``)    33,408 new -> 1,027,732

Nothing historical is modified: ``bs_common`` / ``bs_model`` / ``bs_train``,
``tu_model``, ``nmp_model``, ``sg_common``, ``sg_data``, ``pc_metrics`` and
``pc_model`` are imported read-only.  No official Test row is ever indexed.
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

ROUND_ID = "lcrc-pcba-cpe-v1"
PROTOCOL_ID = ROUND_ID
SCHEMA_RUN = "lcrc-pcba-cpe-run-v1"
SCHEMA_CHECKPOINT = "lcrc-pcba-cpe-checkpoint-v1"
SCHEMA_AUDIT = "lcrc-pcba-cpe-audit-v1"

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
BASE_BASE_PARAMETERS = 848_096
BASE_READOUT_PARAMETERS = 146_228
BASE_AP40 = 0.26186691813350904  # protocol writes 0.2618669181
BASE_AP40_PROTOCOL = 0.2618669181
BASE_TRAIN_AP40 = 0.5818745570283522
BASE_TRAIN_BCE40 = 0.025215364422066445
BASE_VALID_BCE40 = 0.04653792303564479
BASE_BEST_EPOCH = 24
BASE_BEST_AP = 0.26527331610028143
BASE_GLOBAL_PLAN_SHA256 = "06f437299fa451e449202a990d28f3b8d101e7aec709bcd72a878c8de864b6f6"
BASE_CONSUMED40_PREFIX_SHA256 = "1a24dfca54898dd2b0e7c58d2b2f1956df67b233fdceaa7a376f81a2377a5f58"
BASE_TRUNK_INIT_FINGERPRINT = "5f7101a8f378e7c9283caaade8d61eee733c0c9f3cfb65b5584c8a1e495ef63d"
BASE_FULL_TRUNK_INIT_FINGERPRINT = "109756bd24b2ab8589012031e3cf0b0389b0d275227b35044e9381a45c00a5db"
BASE_CHECKPOINT_SHA256 = "7fbdbd16f7b2077e77cdb85488bb2506f2e4ee98333d40fbcf7558074e26598a"

W2_RUN_DIR = BS_ROUND / "runs" / BASE_MODEL_KEY
W2_RUN_JSON = W2_RUN_DIR / "run.json"
W2_CURVE = W2_RUN_DIR / "train_curve.csv"
W2_PER_TASK = W2_RUN_DIR / "per_task_ap.csv"
W2_TRAIN_EVAL = W2_RUN_DIR / "train_eval_epoch40.json"
W2_CHECKPOINT = W2_RUN_DIR / "checkpoint_epoch40.pt"

#: The frozen W2 width dict (scale 2), passed explicitly to every constructor.
WIDTHS = BC.config_for_scale(2)

# ---------------------------------------------------------------------------
# the new module
# ---------------------------------------------------------------------------

EVIDENCE_WIDTH = 128  # == WIDTHS["width"] == the local h width == n_tasks
INIT_GENERATOR_SEED = 37  # private CPU generator for linear_A / linear_B

#: Exact accounting of the new module (see PROTOCOL.md section 6).
NEW_BRANCH = {
    "linear_A": {"in": EVIDENCE_WIDTH, "out": EVIDENCE_WIDTH, "parameters": EVIDENCE_WIDTH * EVIDENCE_WIDTH + EVIDENCE_WIDTH},
    "linear_B": {"in": EVIDENCE_WIDTH, "out": EVIDENCE_WIDTH, "parameters": EVIDENCE_WIDTH * EVIDENCE_WIDTH + EVIDENCE_WIDTH},
    "alpha_A": {"out": N_TASKS, "parameters": N_TASKS},
    "alpha_B": {"out": N_TASKS, "parameters": N_TASKS},
    "alpha_pair": {"out": N_TASKS, "parameters": N_TASKS},
}
NEW_PARAMETERS = int(sum(int(v["parameters"]) for v in NEW_BRANCH.values()))
EXPECTED_TOTALS = {"P": BASE_PARAMETERS + NEW_PARAMETERS, "S": BASE_PARAMETERS + NEW_PARAMETERS}

PAIR_MODES = {"P": "cross", "S": "self"}
MODELS: dict[str, dict[str, Any]] = {
    "P": {
        "name": "LCRC-W2-CrossAtomPredictiveEvidence",
        "role": "cross_atom_second_order_evidence",
        "pair_mode": "cross",
        "new_parameters": int(NEW_PARAMETERS),
        "parameters": int(EXPECTED_TOTALS["P"]),
        "assigned_pool": "res-gpu1",
    },
    "S": {
        "name": "LCRC-W2-SameAtomPredictiveEvidence",
        "role": "same_atom_second_order_control",
        "pair_mode": "self",
        "new_parameters": int(NEW_PARAMETERS),
        "parameters": int(EXPECTED_TOTALS["S"]),
        "assigned_pool": "res-gpu0",
    },
}
MODEL_ORDER = ("P", "S")

SEED = 0
ORDER_SEED = 0
INIT_RULE = (
    "build the historical W2 seed-0 initial state with "
    "bs_model.build_model('W2', seed=0); copy every original W2 tensor bitwise; "
    "linear_A/linear_B use the standard PyTorch Linear uniform initialisation "
    "from a private CPU generator with seed 37 (identical call order in P and S); "
    "alpha_A = alpha_B = alpha_pair = 0; global torch RNG unchanged"
)

#: ``|alpha_pair[t]|`` above this is called a *non-negligible* pair coefficient
#: (absolute logit units).  Frozen before any run.
PAIR_COEFFICIENT_NEGLIGIBLE_THRESHOLD = 1.0e-2

#: Fixed, label-free, result-free diagnostic molecule count (first rows of the
#: official Validation split in its own row order).
DIAGNOSTIC_N_VALID = 256

#: Fixed 64-molecule diagnostic batch used inside the training loop (first
#: batch of epoch 1's frozen plan, truncated) -- identical to the previous
#: rounds so the per-epoch diagnostics are comparable.
TRAIN_DIAGNOSTIC_MOLECULES = 64

EVAL_CADENCE = {
    "every": 1,
    "resolved": True,
    "source": "frozen by the round brief on the backbone-scaling measurement "
    "(W2 valid 7.21 s / epoch ~125 s = 5.8 %); not re-derived in this round",
    "every_epoch_1_to_40": True,
    "primary_epoch": TOTAL_EPOCHS,
    "diagnostic_epoch0": True,
}


def calc_eval_epochs(every: int = 1) -> list[int]:
    return [e for e in range(1, TOTAL_EPOCHS + 1) if e % int(every) == 0]


BUDGET = {"gpu_target_hours": 4.0, "gpu_hard_hours": 6.0, "max_concurrent_gpus": 2}

#: Pre-registered resource decision (not a significance test).
DECISION_RULES = {
    "promising": {
        "label": "PROMISING_CROSS_ATOM_EVIDENCE",
        "delta_base_min": 0.005,
        "delta_mech_min": 0.003,
        "mean_delta_support_ge20_min": 0.003,
        "improved_support_ge20_min": 60,
        "support_ge20_total": 107,
    },
    "capacity_without_advantage": {"label": "CAPACITY_WITHOUT_CROSS_ATOM_ADVANTAGE"},
    "weak_or_mixed": {"label": "WEAK_OR_MIXED"},
    "no_clear": {"label": "NO_CLEAR_GAIN", "delta_base_max_exclusive": 0.002},
    "regression": {"label": "REGRESSION", "delta_base_max": -0.002},
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


def diagnostic_valid_rows(task: Any) -> Any:
    """The fixed 256 diagnostic Validation rows: ``valid_rows[0:256]``.

    Selection depends only on the frozen official split order -- no label, no
    model output, no metric.
    """
    import numpy as np

    rows = np.asarray(task.valid_rows, dtype=np.int64)
    if rows.size < DIAGNOSTIC_N_VALID:
        raise RuntimeError(f"Validation split has {rows.size} rows < {DIAGNOSTIC_N_VALID}")
    return rows[:DIAGNOSTIC_N_VALID]


def diagnostic_valid_rows_record(task: Any) -> dict[str, Any]:
    rows = diagnostic_valid_rows(task)
    import numpy as np

    return {
        "selection": "first 256 rows of task.valid_rows in the official split's own row order",
        "depends_on_labels": False,
        "depends_on_model_outputs": False,
        "n_rows": int(rows.size),
        "first_row": int(rows[0]),
        "last_row": int(rows[-1]),
        "row_ids_sha256": sha256_bytes(",".join(str(int(r)) for r in rows).encode()),
        "row_ids": [int(r) for r in rows],
        "frozen_in": "tracks/structural-sufficiency/lcrc_pcba_cpe_v1/PROTOCOL.md section 12.1",
    }


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
            "pair_mode": m["pair_mode"],
            "seed": int(SEED),
            "epochs": int(TOTAL_EPOCHS),
            "eval_cadence": int(EVAL_CADENCE["every"]),
            "eval_epochs": list(calc_eval_epochs(int(EVAL_CADENCE["every"]))),
            "widths": dict(WIDTHS),
            "evidence_width": int(EVIDENCE_WIDTH),
            "base_parameters": int(BASE_PARAMETERS),
            "new_parameters": int(m["new_parameters"]),
            "parameters": int(m["parameters"]),
            "new_module": {
                "linear_A": [EVIDENCE_WIDTH, EVIDENCE_WIDTH],
                "linear_B": [EVIDENCE_WIDTH, EVIDENCE_WIDTH],
                "alpha_A": [N_TASKS],
                "alpha_B": [N_TASKS],
                "alpha_pair": [N_TASKS],
                "statistic": (
                    "pair_cross = (sum_A*sum_B - sum_AB) / (n*(n-1))"
                    if m["pair_mode"] == "cross"
                    else "pair_self = sum_AB / n"
                ),
            },
            "init_rule": INIT_RULE,
            "lr_schedule": lr_audit(),
            "test_policy": "official Test targets are NaN in the label artifact and are never indexed",
        }
    )
    return recipe


__all__ = [name for name in dir() if not name.startswith("_")]
