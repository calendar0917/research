"""Frozen definitions for the LCRC-W3 regularization diagnostic v1.

Round: ``lcrc-pcba-w3-regularization-diagnostic-v1``

Two and only two runs of the historical LCRC-A-W3 (2,120,844 parameters) differ
in exactly one regularization hyper-parameter each:

===========  =========  =============
run          dropout    AdamW ``wd``
===========  =========  =============
``W3_WD10``  0.10       0.010
``W3_DO20``  0.20       0.001
===========  =========  =============

The model, the data, the batch order and the rest of the from-scratch recipe
are the frozen historical W3 ones.  Everything here is inherited read-only from
``lcrc_pcba_backbone_scaling_v1`` (which itself inherits ``sg_common`` /
``sg_data`` / ``pc_metrics`` / ``pc_model`` / ``tu_model`` / ``nmp_model``).
No inherited module is edited.

Test targets are read from the official OGB raw label file, never from the
frozen training artifact, and never reach training.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Mapping

CODE_DIR = Path(__file__).resolve().parent
ROUND = CODE_DIR.parent
TRACK = ROUND.parent
REPO = TRACK.parents[1]

BS_ROUND = REPO / "tracks/structural-sufficiency/lcrc_pcba_backbone_scaling_v1"
for _p in (str(CODE_DIR), str(BS_ROUND / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bs_common as BC  # noqa: E402  (frozen width/recipe/data identities)
import bs_model as BM  # noqa: E402  (frozen W3 constructor + width audit)
import bs_train as BT  # noqa: E402  (the frozen training step)

ANALYSIS = ROUND / "analysis"
RUNS = ROUND / "runs"
TESTS = ROUND / "tests"
FIGURES = ANALYSIS / "figures"

ROUND_ID = "lcrc-pcba-w3-regularization-diagnostic-v1"
PROTOCOL_ID = ROUND_ID
SCHEMA_RUN = "lcrc-pcba-w3-regularization-run-v1"
SCHEMA_CHECKPOINT = "lcrc-pcba-w3-regularization-checkpoint-v1"
SCHEMA_EPOCH_METRICS = "lcrc-pcba-w3-regularization-epoch-metrics-v1"

DATASET = BC.DATASET
N_TASKS = BC.N_TASKS
N_MOLECULES = BC.N_MOLECULES
N_TRAIN = BC.N_TRAIN
N_VALID = BC.N_VALID
N_TEST = BC.N_TEST
SEED = BC.SEED
BATCH_SIZE = BC.BATCH_SIZE
TOTAL_EPOCHS = BC.TOTAL_EPOCHS
STEPS_PER_EPOCH = BC.STEPS_PER_EPOCH
TOTAL_STEPS_40 = BC.TOTAL_STEPS_40
SPLITS = ("train", "valid", "test")
ROW_COUNTS = {"train": N_TRAIN, "valid": N_VALID, "test": N_TEST}

SOURCE = BC.SOURCE
BASE_RECIPE = BC.RECIPE
scheduled_lr = BC.scheduled_lr

# ---------------------------------------------------------------------------
# the two runs
# ---------------------------------------------------------------------------


def _run(dropout: float, weight_decay: float, pool: str, role: str) -> dict[str, Any]:
    return {
        "dropout": float(dropout),
        "weight_decay": float(weight_decay),
        "assigned_pool": str(pool),
        "role": str(role),
        "model_key": "W3",
        "scale": 3,
        "parameters": 2_120_844,
    }


RUNS: dict[str, dict[str, Any]] = {
    "W3_WD10": _run(0.10, 1.0e-2, "res-gpu1", "stronger_weight_decay"),
    "W3_DO20": _run(0.20, 1.0e-3, "res-gpu0", "higher_dropout"),
}
RUN_ORDER = ("W3_WD10", "W3_DO20")

#: The only hyper-parameters that may differ between the two runs.
INTERVENTION_KEYS = ("dropout", "weight_decay")

EXPECTED_PARAMETERS = 2_120_844
EXPECTED_BASE_PARAMETERS = 1_809_408
EXPECTED_READOUT_PARAMETERS = 311_436
W3_CFG = BC.config_for_scale(3)

#: Everything that must be *identical* in the two runs (audited, not assumed).
SHARED_RECIPE: dict[str, Any] = {
    "seed": int(SEED),
    "epochs": int(TOTAL_EPOCHS),
    "batch_size": int(BATCH_SIZE),
    "optimizer": "AdamW",
    "adam_betas": tuple(BASE_RECIPE["adam_betas"]),
    "adam_eps": float(BASE_RECIPE["adam_eps"]),
    "learning_rate": float(BASE_RECIPE["learning_rate"]),
    "warmup_epochs": float(BASE_RECIPE["warmup_epochs"]),
    "warmup_start_lr": float(BASE_RECIPE["warmup_start_lr"]),
    "end_lr": float(BASE_RECIPE["end_lr"]),
    "scheduler": BASE_RECIPE["scheduler"],
    "gradient_clip": float(BASE_RECIPE["gradient_clip"]),
    "precision": "fp32",
    "loss": BASE_RECIPE["loss"],
    "early_stopping": False,
    "steps_per_epoch": int(STEPS_PER_EPOCH),
    "total_steps": int(TOTAL_STEPS_40),
    "batch_plan_sha256": BC.A0["global_plan_sha256"],
    "init_rule": BC.INIT_SEED_RULE,
    "eval_cadence": "every epoch, three splits (train, valid, test)",
    "checkpoint_selection": "max Validation AP, ties -> earlier epoch",
    "test_policy": "EXPOSED_TEST_EXPLORATORY: official Test scored every epoch, never used for selection",
}

EVAL_BATCH_SIZE = {"train": 512, "valid": 512, "test": 512}

# ---------------------------------------------------------------------------
# historical, read-only comparison values (no retraining)
# ---------------------------------------------------------------------------

BS_RUNS = BS_ROUND / "runs"

HISTORICAL: dict[str, dict[str, Any]] = {
    "W3": {
        "round": "lcrc-pcba-backbone-scaling-v1",
        "parameters": 2_120_844,
        "dropout": 0.10,
        "weight_decay": 1.0e-3,
        "train_ap40": 0.7054983915679482,
        "train_bce40": 0.021156915900669225,
        "valid_ap40": 0.259003625836404,
        "valid_bce40": 0.04907872415702596,
        "best_valid_ap": 0.26367025868637306,
        "best_epoch": 25,
        "best_checkpoint": str(BS_RUNS / "W3" / "checkpoint_best.pt"),
        "best_checkpoint_sha256": "b402c76c555cf98028961244e42c5df8c089826a623b3b68144b04a5e13ba9e9",
        "epoch40_checkpoint": str(BS_RUNS / "W3" / "checkpoint_epoch40.pt"),
        "epoch40_checkpoint_sha256": "4eb6a42f562a4fa8099d0a40053ce45f3bfd08645ce62999dee961e764cb18cf",
        "valid_curve": {
            20: 0.2627256007963041,
            25: 0.26367025868637306,
            30: 0.2628839360491647,
            35: 0.260241522584337,
            40: 0.259003625836404,
        },
        "valid_bce_curve": {
            20: 0.046715106375866525,
            25: 0.047680061967651816,
            30: 0.04828744276782277,
            35: 0.048790516166519034,
            40: 0.04907872415702596,
        },
        "test_ap_best": None,  # filled once, read-only, from the historical checkpoint
        "test_ap40": None,
        "source": "lcrc_pcba_backbone_scaling_v1/REPORT.md + runs/W3/run.json",
    },
    "W2": {
        "round": "lcrc-pcba-backbone-scaling-v1",
        "parameters": 994_324,
        "dropout": 0.10,
        "weight_decay": 1.0e-3,
        "train_ap40": 0.5818745570283522,
        "valid_ap40": 0.26186691813350904,
        "best_valid_ap": 0.26527331610028143,
        "best_epoch": 24,
        "test_ap_best": 0.2536168632,
        "test_ap40": 0.2509858,
        "source": "lcrc_pcba_backbone_scaling_v1/REPORT.md (Test from lcrc_pcba_rhic_v1/REPORT.md)",
    },
}

AP_W3_BEST = float(HISTORICAL["W3"]["best_valid_ap"])
AP_W3_40 = float(HISTORICAL["W3"]["valid_ap40"])
AP_W2_BEST = float(HISTORICAL["W2"]["best_valid_ap"])
AP_W2_40 = float(HISTORICAL["W2"]["valid_ap40"])
TRAIN_AP_W3_40 = float(HISTORICAL["W3"]["train_ap40"])

# ---------------------------------------------------------------------------
# decision rules (resource / direction thresholds, fixed in advance)
# ---------------------------------------------------------------------------

DECISION = {
    "best_delta_min": 0.005,
    "ap40_delta_min": 0.003,
    "support_ge20_min_positive": 20,
    "support_5_19_min_positive": 5,
    "train_ap_suppression_delta": -0.05,
    "peak_narrow_epochs": 5,
    "labels": (
        "PROMISING_SINGLE_SEED_SIGNAL",
        "FIT_SUPPRESSION_WITHOUT_GENERALIZATION_GAIN",
        "PEAK_ONLY_OR_UNSTABLE_SIGNAL",
        "NO_CLEAR_GAIN",
        "MIXED_SIGNAL",
    ),
}

#: Report focuses on these epochs.
REPORT_EPOCHS = (20, 25, 30, 35, 40)

BUDGET = {"gpu_target_hours": 10.0, "gpu_hard_hours": 16.0, "max_concurrent_gpus": 2}

# ---------------------------------------------------------------------------
# the official raw label file (Test labels; read-only, isolated)
# ---------------------------------------------------------------------------

OGB_ROOT = REPO / "data/ogb/ogbg_molpcba"
OGB_GRAPH_LABEL = OGB_ROOT / "raw/graph-label.csv.gz"
LABEL_READ_CHUNK_ROWS = 20_000


def run_dir(run_key: str) -> Path:
    if run_key not in RUNS:
        raise KeyError(run_key)
    return RUNS / run_key


def recipe_for_run(run_key: str) -> dict[str, Any]:
    """The exact recipe handed to the trainer and recorded in ``run.json``."""
    if run_key not in RUNS:
        raise KeyError(run_key)
    r = RUNS[run_key]
    recipe = dict(SHARED_RECIPE)
    recipe.update(
        {
            "round": ROUND_ID,
            "run": run_key,
            "role": r["role"],
            "model_key": r["model_key"],
            "scale": int(r["scale"]),
            "parameters": int(r["parameters"]),
            "dropout": float(r["dropout"]),
            "weight_decay": float(r["weight_decay"]),
            "widths": dict(W3_CFG),
            "lr_schedule": lr_audit(),
            "eval_batch_size": dict(EVAL_BATCH_SIZE),
            "intervention": {"dropout": float(r["dropout"]), "weight_decay": float(r["weight_decay"])},
        }
    )
    return recipe


def lr_audit() -> dict[str, Any]:
    return {
        "steps": int(STEPS_PER_EPOCH),
        "total_steps": int(TOTAL_STEPS_40),
        "step0": scheduled_lr(0, phase=3, kind="all"),
        "warmup_last": scheduled_lr(STEPS_PER_EPOCH - 1, phase=3, kind="all"),
        "post_warmup_first": scheduled_lr(STEPS_PER_EPOCH, phase=3, kind="all"),
        "step40e": scheduled_lr(TOTAL_STEPS_40, phase=3, kind="all"),
        "formula": (
            "steps 0..2737 linear 3e-5->3e-4; then "
            "1e-5+(3e-4-1e-5)*0.5*(1+cos(pi*(s-2738)/(109520-2738)))"
        ),
    }


def assert_shared_recipe() -> dict[str, Any]:
    """Both runs must agree on everything except the two intervention values."""
    a, b = (dict(recipe_for_run(k)) for k in RUN_ORDER)
    diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    expected = sorted(INTERVENTION_KEYS + ("run", "role", "intervention"))
    if diff != expected:
        raise RuntimeError(f"recipes differ in {diff}, expected exactly {expected}")
    return {"differing_keys": diff, "expected": expected, "ok": diff == expected}


# ---------------------------------------------------------------------------
# IO / hashing helpers (inherited from the audited chain)
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
state_fingerprint = BC.state_fingerprint
parameter_count = BC.parameter_count
gpu_occupancy = BC.gpu_occupancy
gpu_peak = BC.gpu_peak
host_facts = BC.host_facts
MET = BC.MET
SD = BC.SD
PC = BC.PC
_repo = REPO


def module_inventory() -> dict[str, Any]:
    """Which frozen implementation files this round actually executed."""
    files = {
        "this_round": ["code/rd_common.py", "code/rd_data.py", "code/rd_model.py", "code/rd_train.py"],
        "inherited_read_only": [
            "lcrc_pcba_backbone_scaling_v1/code/bs_common.py",
            "lcrc_pcba_backbone_scaling_v1/code/bs_model.py",
            "lcrc_pcba_backbone_scaling_v1/code/bs_train.py",
            "lcrc_pcba_structural_gain_sprint_v1/code/sg_common.py",
            "lcrc_pcba_structural_gain_sprint_v1/code/sg_data.py",
            "lcrc_pcba_lr_probe_v1/code/lr_data.py",
            "lcrc_pcba_scale_pilot_v1/code/pc_metrics.py",
            "lcrc_pcba_scale_pilot_v1/code/pc_model.py",
            "lcrc_pcba_pipeline_perf_v1/code/pf_pipeline.py",
            "lcrc_tu_quick_screen_v1/code/tu_model.py",
            "lcrc_pcba_no_mp_sprint_v1/code/nmp_model.py",
        ],
    }
    digests = {}
    for rel in files["this_round"] + files["inherited_read_only"]:
        path = REPO / "tracks/structural-sufficiency" / rel
        digests[rel] = sha256_file(path) if path.is_file() else f"missing:{path}"
    return {**files, "sha256": digests}


def _json_default(value: Any) -> Any:  # pragma: no cover - defensive
    return BC._json_default(value)


__all__ = [name for name in dir() if not name.startswith("_")]
