"""Frozen definitions for the PCQM4Mv2 efficiency + width diagnostic round v1.

Round: ``lcrc-pcqm4mv2-efficiency-width-v1``

Two independent questions, one code base:

* **Q1 (engineering)** -- can the *real* end-to-end batch-128 training
  throughput of the frozen W2 recipe be improved while every input tensor, the
  forward result, the loss, the gradient, the optimizer state and the batch
  order stay identical?
* **Q2 (capacity)** -- at the same 1M PCQM4Mv2 Train molecules and the same
  40-epoch recipe, does the LCRC parameter count (W1 / W3 around the read-only
  historical W2) change Validation MAE?

Everything that defines the *science* is inherited read-only from
``lcrc_pcqm4mv2_transfer_pilot_v1``: the cache, the row permutation, the
monitor rows, the feature/label normalisation, the official evaluator, the
model builder and the recipe.  This module only states the new round's
identities and the two new runs.

The historical W2 run (commit ``93ae701f``, 939,079 parameters, 40 epochs,
Validation MAE 0.132310904 eV, 5.162 GPU-hours) is used as a **read-only
control**; it is never retrained here.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROUND_ID = "lcrc-pcqm4mv2-efficiency-width-v1"
PROTOCOL_ID = ROUND_ID
SCHEMA_RUN = "lcrc-pcqm4mv2-efficiency-width-run-v1"
SCHEMA_CHECKPOINT = "lcrc-pcqm4mv2-efficiency-width-checkpoint-v1"
SCHEMA_PROFILE = "lcrc-pcqm4mv2-efficiency-profile-v1"
SCHEMA_BENCH = "lcrc-pcqm4mv2-efficiency-bench-v1"
SCHEMA_VERIFY = "lcrc-pcqm4mv2-efficiency-verify-v1"

CODE_DIR = Path(__file__).resolve().parent
ROUND = CODE_DIR.parent
TRACK = ROUND.parent
REPO = TRACK.parents[1]

#: Read-only parent round: cache, subsets, normalisation, evaluator, model,
#: recipe and the historical W2 run.
OLD_ROUND = TRACK / "lcrc_pcqm4mv2_transfer_pilot_v1"
OLD_CODE = OLD_ROUND / "code"
OLD_CACHE = OLD_ROUND / "cache"
OLD_ANALYSIS = OLD_ROUND / "analysis"
OLD_RUNS = OLD_ROUND / "runs"

for _p in (str(OLD_CODE), str(CODE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pq_common as PQ  # noqa: E402  (frozen IO helpers / recipe / widths)
import pq_data as PQ_DATA  # noqa: E402  (frozen cache loader + task assembly)
import pq_model as PQ_MODEL  # noqa: E402  (frozen W2 builder)
import pq_train as PQ_TRAIN  # noqa: E402  (frozen training step / evaluation)

ANALYSIS = ROUND / "analysis"
CACHE = ROUND / "cache"
RUNS = ROUND / "runs"
TESTS = ROUND / "tests"
CONFIGS = ROUND / "configs"
for _d in (ANALYSIS, RUNS):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# inherited identities (asserted, never re-defaulted)
# ---------------------------------------------------------------------------

DATASET = PQ.DATASET
TASK = PQ.TASK
N_OUT = int(PQ.N_OUT)  # 1 regression value
BATCH_SIZE = int(PQ.BATCH_SIZE)  # 128 -- formal runs are fixed at this
TOTAL_EPOCHS = int(PQ.TOTAL_EPOCHS)  # 40
MODEL_SEED = int(PQ.MODEL_SEED)  # 0
ORDER_SEED = int(PQ.ORDER_SEED)  # 0
SAMPLE_SEED = int(PQ.SAMPLE_SEED)  # 42
MONITOR_SEED = int(PQ.MONITOR_SEED)  # 43
N_MONITOR = int(PQ.N_MONITOR)  # 20,000
RECIPE: dict[str, Any] = dict(PQ.RECIPE)
TRAIN_ROWS = 1_000_000
STEPS_PER_EPOCH = int(PQ.steps_per_epoch(TRAIN_ROWS))
TOTAL_STEPS = int(PQ.total_steps(TRAIN_ROWS))

#: Historical W2 control (read-only; never retrained in this round).
HISTORICAL = {
    "run": "PCQM_W2_1M",
    "scale": 2,
    "model": "W2",
    "parameters": 939_079,
    "train_rows": 1_000_000,
    "seed": 0,
    "epochs": 40,
    "best_epoch": 40,
    "best_valid_mae_eV": 0.132310904,
    "valid_mae_eV_at_40": 0.132310904,
    "monitor_mae_eV_at_40": 0.105891,
    "gpu_hours": 5.162,
    "code_commit": "93ae701f",
    "run_dir": "tracks/structural-sufficiency/lcrc_pcqm4mv2_transfer_pilot_v1/runs/PCQM_W2_1M",
    "checkpoint_best_sha256": "e45316c29d9152adf79e702ebd701bf62e6237065112f8b000b419d1de3ed0b1",
}
OLD_RUN_CURVE = OLD_RUNS / "PCQM_W2_1M" / "train_curve.csv"
OLD_RUN_JSON = OLD_RUNS / "PCQM_W2_1M" / "run.json"
OLD_RUN_BEST = OLD_RUNS / "PCQM_W2_1M" / "checkpoint_best_valid.pt"

#: Width scales.  Scale 1 / 2 / 3 are the inherited ``bs_common`` table with
#: ``n_out = 1``; only ``width scale`` differs between the three arms.
WIDTHS: dict[int, dict[str, Any]] = {
    s: {**PQ.BS.config_for_scale(s), "n_out": N_OUT} for s in (1, 2, 3)
}
EXPECTED_PARAMETERS: dict[int, int] = {1: 259_831, 2: 939_079, 3: 2_038_167}
MODEL_NAMES: dict[int, str] = {1: "LCRC-W1", 2: "LCRC-W2", 3: "LCRC-W3"}

#: The two new formal runs.  Identical data, recipe, seed, batch size, epochs
#: and batch order; only the width scale changes.
RUNS_TABLE: dict[str, dict[str, Any]] = {
    "PCQM_W1_1M": {
        "train_size": TRAIN_ROWS,
        "cache_rows": TRAIN_ROWS,
        "epochs": TOTAL_EPOCHS,
        "seed": MODEL_SEED,
        "scale": 1,
        "model": "W1",
        "model_name": MODEL_NAMES[1],
        "expected_parameters": EXPECTED_PARAMETERS[1],
        "label": "capacity_scan_1x",
    },
    "PCQM_W3_1M": {
        "train_size": TRAIN_ROWS,
        "cache_rows": TRAIN_ROWS,
        "epochs": TOTAL_EPOCHS,
        "seed": MODEL_SEED,
        "scale": 3,
        "model": "W3",
        "model_name": MODEL_NAMES[3],
        "expected_parameters": EXPECTED_PARAMETERS[3],
        "label": "capacity_scan_3x",
    },
}
RUN_ORDER = ("PCQM_W3_1M", "PCQM_W1_1M")  # W3 first (the more expensive arm)

#: Absolute error tolerances for the correctness gates (declared before any
#: measurement).  The input-tensor gate is exact; the model-level gate allows
#: only float32 reassociation on the pre-declared scale.
GATES: dict[str, Any] = {
    "input_tensors": "bitwise equal (value, dtype, shape) for every field",
    "forward_abs_tol": 1e-6,
    "forward_rel_tol": 1e-6,
    "loss_abs_tol": 1e-7,
    "grad_abs_tol": 1e-7,
    "parameter_bitwise_after_200_steps": True,
    "parameter_abs_tol_after_200_steps": 1e-7,
    "fixed_sample_prediction_abs_tol": 1e-6,
    "official_vs_numpy_mae_abs_tol": 1e-9,
    "equivalent_steps": 200,
    "atom_renumbering_abs_tol": 1e-4,
    "min_speedup_to_adopt": 1.15,
}

#: Budget (A100 GPU-hours) of the whole new round, and of the profiling stage.
BUDGET = {
    "round_hard_gpu_hours": 12.0,
    "profile_stage_gpu_hours": 0.75,
    "formal_runs_reserve_gpu_hours": 11.0,
    "note": "profiling + preflight + both formal 40-epoch runs share this ledger",
}

#: Throughput-investigation-only batch sizes (never used for a formal run).
PROBE_BATCH_SIZES = (128, 256, 512)


def steps_per_epoch(n_rows: int = TRAIN_ROWS) -> int:
    return int(PQ.steps_per_epoch(int(n_rows), BATCH_SIZE))


def total_steps(n_rows: int = TRAIN_ROWS, epochs: int = TOTAL_EPOCHS) -> int:
    return int(PQ.total_steps(int(n_rows), int(epochs), BATCH_SIZE))


scheduled_lr = PQ.scheduled_lr


def recipe_for_run(run: str) -> dict[str, Any]:
    """The frozen recipe, annotated with this round's run identity."""
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
        "cache_rows": int(spec["cache_rows"]),
        "seed": int(spec["seed"]),
        "epochs": int(spec["epochs"]),
        "widths": dict(WIDTHS[int(spec["scale"])]),
        "lr_schedule": PQ.lr_audit(int(spec["train_size"])),
        "init_rule": PQ.INIT_SEED_RULE,
        "feature_schema": "lipo_v1",
        "n_outputs": N_OUT,
        "historical_control": HISTORICAL["run"],
    }


def param_table() -> dict[str, Any]:
    """Analytic parameter count per scale (computed, not copied)."""
    import feature_spec as FS

    spec = FS.lipo_v1_spec()
    out: dict[str, Any] = {}
    for scale, cfg in WIDTHS.items():
        analytic = PQ.expected_parameter_count(
            cfg,
            atom_feat_dim=spec.atom_feat_dim,
            bond_feat_dim=spec.bond_feat_dim,
            n_scalars=spec.n_scalars,
            global_dim=spec.global_dim,
            n_nbr_codes=spec.n_nbr_codes,
            n_path_codes=spec.n_path_codes,
        )
        out[str(scale)] = {
            "scale": int(scale),
            "model": f"W{scale}",
            "widths": {k: int(v) for k, v in cfg.items() if k != "n_out"},
            "base_parameters": int(analytic["base_parameters"]),
            "readout_parameters": int(analytic["readout_parameters"]),
            "total_parameters": int(analytic["total_parameters"]),
            "expected_parameters": int(EXPECTED_PARAMETERS[scale]),
            "matches_expected": int(analytic["total_parameters"]) == int(EXPECTED_PARAMETERS[scale]),
        }
    return out


# ---------------------------------------------------------------------------
# IO / provenance (inherited helpers)
# ---------------------------------------------------------------------------

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


def cpu_seconds() -> float:
    import resource

    return float(resource.getrusage(resource.RUSAGE_SELF).ru_utime + resource.getrusage(resource.RUSAGE_SELF).ru_stime)


def assert_recipe() -> None:
    """The pre-registered recipe of this round, checked at import-use time."""
    if BATCH_SIZE != 128 or TOTAL_EPOCHS != 40:
        raise RuntimeError("formal recipe must be batch 128 / 40 epochs")
    if MODEL_SEED != 0 or ORDER_SEED != 0:
        raise RuntimeError("formal seed/order must be 0")
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
    if STEPS_PER_EPOCH != 7_813 or TOTAL_STEPS != 312_520:
        raise RuntimeError(f"steps/epoch {STEPS_PER_EPOCH} total {TOTAL_STEPS} != 7813/312520")
    for scale, expected in EXPECTED_PARAMETERS.items():
        got = param_table()[str(scale)]["total_parameters"]
        if int(got) != int(expected):
            raise RuntimeError(f"scale {scale}: analytic {got} != expected {expected}")
    if set(RUNS_TABLE) != {"PCQM_W1_1M", "PCQM_W3_1M"}:
        raise RuntimeError("this round runs exactly W1 and W3 on 1M rows")


assert_recipe()


def rng_state_hash_summary() -> dict[str, Any]:
    return rng_state_hashes(capture_rng_state())


def json_safe(value: Any) -> Any:
    """Recursively convert numpy / torch scalars so ``write_json`` never fails."""
    import torch

    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float):
        return value
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return str(value)


__all__ = [
    "ANALYSIS",
    "BATCH_SIZE",
    "BUDGET",
    "CACHE",
    "CONFIGS",
    "DATASET",
    "EXPECTED_PARAMETERS",
    "GATES",
    "HISTORICAL",
    "MODEL_NAMES",
    "MODEL_SEED",
    "N_OUT",
    "OLD_ANALYSIS",
    "OLD_CACHE",
    "OLD_CODE",
    "OLD_ROUND",
    "OLD_RUNS",
    "OLD_RUN_BEST",
    "OLD_RUN_CURVE",
    "OLD_RUN_JSON",
    "ORDER_SEED",
    "PQ",
    "PQ_DATA",
    "PQ_MODEL",
    "PQ_TRAIN",
    "PROBE_BATCH_SIZES",
    "PROTOCOL_ID",
    "RECIPE",
    "REPO",
    "ROUND",
    "ROUND_ID",
    "RUNS",
    "RUNS_TABLE",
    "RUN_ORDER",
    "SCHEMA_BENCH",
    "SCHEMA_CHECKPOINT",
    "SCHEMA_PROFILE",
    "SCHEMA_RUN",
    "SCHEMA_VERIFY",
    "STEPS_PER_EPOCH",
    "TOTAL_EPOCHS",
    "TOTAL_STEPS",
    "TESTS",
    "TRAIN_ROWS",
    "WIDTHS",
    "apply_deterministic_settings",
    "assert_recipe",
    "capture_rng_state",
    "cpu_seconds",
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
