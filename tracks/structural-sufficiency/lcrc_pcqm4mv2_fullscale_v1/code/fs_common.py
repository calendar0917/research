"""Frozen identities for the PCQM4Mv2 full-Train scale round.

Round: ``lcrc-pcqm4mv2-fullscale-v1``

The science is inherited read-only from ``lcrc_pcqm4mv2_transfer_pilot_v1``
(cache format, ``lipo_v1`` features, model, recipe, normalisation, evaluator)
and the adopted E1 vectorised collation from
``lcrc_pcqm4mv2_efficiency_width_v1``.  This module only states the full-scale
identities: 3,378,606 Train rows, the 40-epoch 1,055,840-step horizon, the two
width arms and the budget.

Nothing here is chosen after seeing a score; the module fails loudly if the
inherited recipe or the pre-registered dimensions drift.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROUND_ID = "lcrc-pcqm4mv2-fullscale-v1"
PROTOCOL_ID = ROUND_ID
SCHEMA_RUN = "lcrc-pcqm4mv2-fullscale-run-v1"
SCHEMA_CHECKPOINT = "lcrc-pcqm4mv2-fullscale-checkpoint-v1"
SCHEMA_BUILD = "lcrc-pcqm4mv2-fullscale-cache-build-v1"
SCHEMA_VERIFY = "lcrc-pcqm4mv2-fullscale-verify-v1"

CODE_DIR = Path(__file__).resolve().parent
ROUND = CODE_DIR.parent
TRACK = ROUND.parent
REPO = TRACK.parents[1]

#: Read-only parent round and the adopted-optimisation round.
PARENT = TRACK / "lcrc_pcqm4mv2_transfer_pilot_v1"
PARENT_CODE = PARENT / "code"
PARENT_CACHE = PARENT / "cache"
PARENT_ANALYSIS = PARENT / "analysis"
PARENT_RUNS = PARENT / "runs"

EFFICIENCY = TRACK / "lcrc_pcqm4mv2_efficiency_width_v1"
EFFICIENCY_CODE = EFFICIENCY / "code"
EFFICIENCY_RUNS = EFFICIENCY / "runs"

for _p in (str(PARENT_CODE), str(EFFICIENCY_CODE), str(CODE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ew_collate as EW_COLLATE  # noqa: E402  (E1 vectorised collation, adopted)
import ew_common as EW  # noqa: E402  (width table / param accounting)
import ew_model as EW_MODEL  # noqa: E402  (scaled LCRC builder, frozen semantics)
import pq_common as PQ  # noqa: E402  (recipe / IO helpers)
import pq_data as PQ_DATA  # noqa: E402  (frozen cache / task / evaluator helpers)
import pq_model as PQ_MODEL  # noqa: E402  (historical W2 builder)
import pq_train as PQ_TRAIN  # noqa: E402  (training step / evaluation)

ANALYSIS = ROUND / "analysis"
CACHE = ROUND / "cache"
RUNS = ROUND / "runs"
TESTS = ROUND / "tests"
for _d in (ANALYSIS, CACHE, RUNS):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# dataset / run identity
# ---------------------------------------------------------------------------

DATASET = PQ.DATASET
TASK = PQ.TASK
N_OUT = int(PQ.N_OUT)
BATCH_SIZE = int(PQ.BATCH_SIZE)  # 128, fixed by the protocol
TOTAL_EPOCHS = 40
MODEL_SEED = 0
ORDER_SEED = 0

TRAIN_ROWS = 3_378_606
FULL_TRAIN_ROWS = TRAIN_ROWS
VALID_ROWS = 73_545
TEST_DEV_ROWS = 147_037
TEST_CHALLENGE_ROWS = 147_432

STEPS_PER_EPOCH = (TRAIN_ROWS + BATCH_SIZE - 1) // BATCH_SIZE  # 26,396
TOTAL_STEPS = STEPS_PER_EPOCH * TOTAL_EPOCHS  # 1,055,840

RECIPE: dict[str, Any] = dict(PQ.RECIPE)
FORMAL_IMPLEMENTATION = "E1"

WIDTHS: dict[int, dict[str, Any]] = {
    s: {**PQ.BS.config_for_scale(s), "n_out": N_OUT} for s in (2, 3)
}
EXPECTED_PARAMETERS: dict[int, int] = {2: 939_079, 3: 2_038_167}
MODEL_NAMES: dict[int, str] = {2: "LCRC-W2", 3: "LCRC-W3"}

RUNS_TABLE: dict[str, dict[str, Any]] = {
    "PCQM_W2_FULL": {
        "train_size": TRAIN_ROWS,
        "epochs": TOTAL_EPOCHS,
        "seed": MODEL_SEED,
        "scale": 2,
        "model": "W2",
        "model_name": MODEL_NAMES[2],
        "expected_parameters": EXPECTED_PARAMETERS[2],
        "label": "full_train_width_2x",
    },
    "PCQM_W3_FULL": {
        "train_size": TRAIN_ROWS,
        "epochs": TOTAL_EPOCHS,
        "seed": MODEL_SEED,
        "scale": 3,
        "model": "W3",
        "model_name": MODEL_NAMES[3],
        "expected_parameters": EXPECTED_PARAMETERS[3],
        "label": "full_train_width_3x",
    },
}
RUN_ORDER = ("PCQM_W2_FULL", "PCQM_W3_FULL")

#: Historical 1M controls (read-only; the numbers are the published results of
#: ``lcrc_pcqm4mv2_efficiency_width_v1`` and the transfer pilot).
HISTORICAL: dict[str, dict[str, Any]] = {
    "PCQM_W2_1M": {
        "train_rows": 1_000_000,
        "scale": 2,
        "parameters": EXPECTED_PARAMETERS[2],
        "best_epoch": 40,
        "best_valid_mae_eV": 0.132310904,
        "valid_mae_eV_at_40": 0.132310904,
        "monitor_mae_eV_at_40": 0.105891,
        "gpu_hours": 5.162,
        "source_round": "lcrc_pcqm4mv2_transfer_pilot_v1",
        "code_commit": "93ae701f",
        "checkpoint_best_sha256": (
            "e45316c29d9152adf79e702ebd701bf62e6237065112f8b000b419d1de3ed0b1"
        ),
    },
    "PCQM_W3_1M": {
        "train_rows": 1_000_000,
        "scale": 3,
        "parameters": EXPECTED_PARAMETERS[3],
        "best_epoch": 39,
        "best_valid_mae_eV": 0.127809167,
        "valid_mae_eV_at_40": 0.127874764,
        "monitor_mae_eV_at_40": 0.097213247,
        "gpu_hours": 3.447,
        "source_round": "lcrc_pcqm4mv2_efficiency_width_v1",
        "code_commit": "6e9d58c",
        "checkpoint_best_sha256": (
            "f4175f141702b9cc10b23404c6689936fed0692aac32086489e0a4bf2492b5ac"
        ),
    },
}
HISTORICAL_W2_CHECKPOINT = (
    PARENT_RUNS / "PCQM_W2_1M" / "checkpoint_best_valid.pt"
)

#: GPU-hour ledger of the whole round.
BUDGET = {
    "target_gpu_hours": 26.0,
    "hard_gpu_hours": 38.0,
    "reserve_gpu_hours_inference": 0.5,
    "note": (
        "target and hard ceilings include preflight, both formal 40-epoch "
        "runs, final Test-dev inference and a failure reserve"
    ),
}

#: Absolute tolerances used by the correctness gates (declared in advance).
GATES: dict[str, Any] = {
    "input_tensors": "bitwise equal (value, dtype, shape) for every field",
    "forward_abs_tol": 1e-6,
    "checkpoint_resume_bitwise": True,
    "official_vs_numpy_mae_abs_tol": 1e-9,
    "atom_renumbering_abs_tol": 1e-4,
}


def steps_per_epoch(n_rows: int = TRAIN_ROWS) -> int:
    return int((int(n_rows) + BATCH_SIZE - 1) // BATCH_SIZE)


def total_steps(n_rows: int = TRAIN_ROWS, epochs: int = TOTAL_EPOCHS) -> int:
    return int(steps_per_epoch(n_rows)) * int(epochs)


scheduled_lr = PQ.scheduled_lr
utcnow = PQ.utcnow
write_json = PQ.write_json
read_json = PQ.read_json
sha256_file = PQ.sha256_file
sha256_tree = PQ.sha256_tree
sha256_bytes = PQ.sha256_bytes
index_hash = PQ.index_hash
log = PQ.log
git_commit = PQ.git_commit
git_dirty = PQ.git_dirty
environment = PQ.environment
seed_everything = PQ.seed_everything
capture_rng_state = PQ.capture_rng_state
rng_state_hashes = PQ.rng_state_hashes
gpu_occupancy = PQ.gpu_occupancy
host_facts = PQ.host_facts
peak_rss_mb = PQ.peak_rss_mb
apply_deterministic_settings = PQ.apply_deterministic_settings


def track_rel(path: Path | str) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def param_table() -> dict[str, Any]:
    """Analytic parameter counts for the two arms (computed, not copied)."""
    return EW.param_table()


def recipe_for_run(run: str) -> dict[str, Any]:
    if run not in RUNS_TABLE:
        raise KeyError(run)
    spec = RUNS_TABLE[run]
    return {
        **{k: v for k, v in RECIPE.items() if not isinstance(v, tuple)},
        "adam_betas": list(RECIPE["adam_betas"]),
        "round": ROUND_ID,
        "run": run,
        "scale": int(spec["scale"]),
        "model": spec["model"],
        "model_name": spec["model_name"],
        "train_size": int(spec["train_size"]),
        "seed": int(spec["seed"]),
        "epochs": int(spec["epochs"]),
        "widths": dict(WIDTHS[int(spec["scale"])]),
        "lr_schedule": PQ.lr_audit(int(spec["train_size"])),
        "init_rule": PQ.INIT_SEED_RULE,
        "feature_schema": "lipo_v1",
        "n_outputs": N_OUT,
        "implementation": FORMAL_IMPLEMENTATION,
        "historical_controls": {k: v["best_valid_mae_eV"] for k, v in HISTORICAL.items()},
    }


def assert_config() -> None:
    """Pre-registered dimensions and recipe, checked before use."""
    if TRAIN_ROWS != 3_378_606:
        raise RuntimeError(f"official Train size {TRAIN_ROWS} != 3,378,606")
    if STEPS_PER_EPOCH != 26_396 or TOTAL_STEPS != 1_055_840:
        raise RuntimeError(f"horizon {STEPS_PER_EPOCH}/{TOTAL_STEPS} != 26396/1055840")
    if BATCH_SIZE != 128 or TOTAL_EPOCHS != 40 or MODEL_SEED != 0 or ORDER_SEED != 0:
        raise RuntimeError("formal recipe must be batch 128 / 40 epochs / seed 0")
    for key, want in (
        ("learning_rate", 3e-4),
        ("warmup_start_lr", 3e-5),
        ("end_lr", 1e-5),
        ("weight_decay", 1e-3),
        ("dropout", 0.10),
        ("gradient_clip", 5.0),
        ("adam_eps", 1e-8),
    ):
        if float(RECIPE[key]) != float(want):
            raise RuntimeError(f"recipe {key}={RECIPE[key]} != {want}")
    if tuple(RECIPE["adam_betas"]) != (0.9, 0.999):
        raise RuntimeError("AdamW betas changed")
    if set(RUNS_TABLE) != {"PCQM_W2_FULL", "PCQM_W3_FULL"}:
        raise RuntimeError("this round runs exactly W2 and W3 on the full Train split")
    table = param_table()
    for scale, expected in EXPECTED_PARAMETERS.items():
        if int(table[str(scale)]["total_parameters"]) != int(expected):
            raise RuntimeError(f"scale {scale}: analytic != pre-registered parameter count")
    if str(RECIPE["scheduler"]).find("40-epoch") < 0:
        raise RuntimeError("scheduler description drifted")


assert_config()


def json_safe(value: Any) -> Any:
    """Recursively convert numpy / torch scalars for JSON output."""
    import numpy as np
    import torch

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


__all__ = [
    "ANALYSIS",
    "BATCH_SIZE",
    "BUDGET",
    "CACHE",
    "DATASET",
    "EFFICIENCY",
    "EFFICIENCY_CODE",
    "EFFICIENCY_RUNS",
    "EW",
    "EW_COLLATE",
    "EW_MODEL",
    "EXPECTED_PARAMETERS",
    "FULL_TRAIN_ROWS",
    "FORMAL_IMPLEMENTATION",
    "GATES",
    "HISTORICAL",
    "HISTORICAL_W2_CHECKPOINT",
    "MODEL_NAMES",
    "MODEL_SEED",
    "N_OUT",
    "ORDER_SEED",
    "PARENT",
    "PARENT_ANALYSIS",
    "PARENT_CACHE",
    "PARENT_CODE",
    "PARENT_RUNS",
    "PQ",
    "PQ_DATA",
    "PQ_MODEL",
    "PQ_TRAIN",
    "PROTOCOL_ID",
    "RECIPE",
    "REPO",
    "ROUND",
    "ROUND_ID",
    "RUNS",
    "RUNS_TABLE",
    "RUN_ORDER",
    "SCHEMA_BUILD",
    "SCHEMA_CHECKPOINT",
    "SCHEMA_RUN",
    "SCHEMA_VERIFY",
    "STEPS_PER_EPOCH",
    "TASK",
    "TESTS",
    "TEST_CHALLENGE_ROWS",
    "TEST_DEV_ROWS",
    "TOTAL_EPOCHS",
    "TOTAL_STEPS",
    "TRAIN_ROWS",
    "VALID_ROWS",
    "WIDTHS",
    "apply_deterministic_settings",
    "assert_config",
    "capture_rng_state",
    "environment",
    "git_commit",
    "git_dirty",
    "gpu_occupancy",
    "host_facts",
    "index_hash",
    "json_safe",
    "log",
    "param_table",
    "peak_rss_mb",
    "read_json",
    "recipe_for_run",
    "rng_state_hashes",
    "scheduled_lr",
    "seed_everything",
    "sha256_bytes",
    "sha256_file",
    "sha256_tree",
    "steps_per_epoch",
    "total_steps",
    "track_rel",
    "utcnow",
    "write_json",
]
