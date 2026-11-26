"""Identity, paths, frozen constants and IO for `fixed-source-scaffold-balanced-learning-v0`.

The scientific constants are *not* decided here at runtime: the recipe, the model
and the data layer are the frozen CSC v0 objects, re-imported read-only.  This
module only adds the round identity, the read-only source paths and the weight
rule (section 5 of ``PROTOCOL.md``).
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Any

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROUND_DIR = CODE_DIR.parent
TRACK_DIR = ROUND_DIR.parent
REPO_ROOT = TRACK_DIR.parents[1]

ANALYSIS_DIR = ROUND_DIR / "analysis"
CONFIG_DIR = ROUND_DIR / "configs"
RUNS_DIR = ROUND_DIR / "runs"
RESULTS_DIR = ROUND_DIR / "results"

# ---------------------------------------------------------------------------
# read-only reuse roots
# ---------------------------------------------------------------------------

CSC_DIR = TRACK_DIR / "context_conditioned_structural_computation_v0"
CSC_CODE = CSC_DIR / "code"
CSC_RUN = CSC_DIR / "runs/mollipo_A_seed0"
SFL_DIR = TRACK_DIR / "lcrc_structural_failure_localization_v0"
SFL_CODE = SFL_DIR / "code"
SFL_CASES = SFL_DIR / "analysis/cases.csv"
EBR_DIR = TRACK_DIR / "lcrc_encoder_boundary_reference_v0"
EBR_PAIRED = EBR_DIR / "analysis/paired_errors.csv"
C1_DIR = TRACK_DIR / "lcrc_readout_context_pilot_small_v1"
C1_MAIN_TABLE = C1_DIR / "analysis/tables/main_table.csv"
OGB_DIR = REPO_ROOT / "data/ogb_reader/ogbg_mollipo"
MOL_CSV = OGB_DIR / "mapping/mol.csv.gz"
SPLIT_DIR = OGB_DIR / "split/scaffold"

assert (CSC_CODE / "cs_model.py").is_file(), CSC_CODE
assert (SFL_CODE / "sfl_common.py").is_file(), SFL_CODE

for _path in (str(CSC_CODE), str(SFL_CODE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import cs_common as C  # noqa: E402  (frozen CSC v0 constants / IO)

# ---------------------------------------------------------------------------
# round identity
# ---------------------------------------------------------------------------

ROUND_ID = "fixed-source-scaffold-balanced-learning-v0"
PROTOCOL_ID = "fixed-source-scaffold-balanced-learning-v0"
SHORT_ID = "sbl-v0"
SCHEMA_RUN = "sbl-run-v1"
SCHEMA_CHECKPOINT = "sbl-checkpoint-v1"
SCHEMA_CHECKS = "sbl-checks-v1"

DATASET = "ogbg-mollipo"
TRAIN_SPLIT = "train"
VALID_SPLIT = "valid"
#: the evaluator refuses anything else (Test is never read, never scored)
ALLOWED_SPLITS: tuple[str, ...] = ("train", "valid")

N_TRAIN = 3360
N_VALID = 420
SEED = 0
EPOCHS = 360

#: historical FIXED1 (CSC v0 arm A, seed 0) — read-only reference numbers
HIST = {
    "run_dir": "tracks/structural-sufficiency/context_conditioned_structural_computation_v0/runs/mollipo_A_seed0",
    "parameters": 294135,
    "initial_state_sha256": "8b2df74e8a18a4cc0c4e6933340fbe1829f240d210f0c02d0626cee29086fdb1",
    "base_state_sha256": "539e0dbec7cb80a78b39b94884b9265c111b085136a20e77511f8313bf16eb62",
    "best_epoch": 140,
    "best_valid_rmse_raw": 0.7266780069873353,
    "best_valid_mae_raw": 0.49844932380229395,
    "best_train_rmse_raw": 0.3079864751738112,
    "best_train_mae_raw": 0.23350284132462634,
    "endpoint_valid_rmse_raw": 0.7415200258174056,
    "endpoint_valid_mae_raw": 0.5060955929875536,
    "last20_valid_rmse_raw": 0.7406963081735681,
    "last20_valid_mae_raw": 0.5061564662748497,
    "train_plan_sha256": "f9c1398c5705ca5a31d2cee418fa1fda6d5bc944ad17451d3a6a98524ae31276",
}
C1_REF = {"valid_rmse_raw": 0.7086243164299489, "valid_mae_raw": 0.5123326828400689, "best_epoch": 275}
GINE_REF = {"valid_rmse_raw": 0.6780750298150204, "valid_mae_raw": 0.487331749229366}

#: verdict thresholds (section 8, frozen before training)
VERDICT = {
    "rmse_delta_gain": -0.03,
    "mae_no_degradation": 0.005,
    "c1_reference": C1_REF["valid_rmse_raw"],
}

#: weight-contrast gate (section 4, frozen before the formal measurement)
CONTRAST_GATE = {"min_weight_ratio": 1.5, "max_ess_fraction": 0.9}

#: support diagnostic (section 7, frozen): Morgan r=2, 2048 bits
MORGAN_RADIUS = 2
MORGAN_BITS = 2048
#: trimmed sensitivity: drop the molecules with the largest *common* squared error
TRIM_FRACTION = 0.05

BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261126

#: frozen recipe: exactly CSC v0's dict (imported, not re-typed)
RECIPE = dict(C.RECIPE)
RECIPE["epochs"] = EPOCHS
WEIGHT_RULE = "weight_i = N_train / (K_train * n_scaffold_i); loss = mean_batch(weight_i * per_molecule_mse)"
BATCH_SIZE = int(C.RECIPE["batch_size"])

# ---------------------------------------------------------------------------
# re-exported helpers (single implementation, no copy)
# ---------------------------------------------------------------------------

utcnow = C.utcnow
sha256_file = C.sha256_file
sha256_bytes = C.sha256_bytes
sha256_array = C.sha256_array
state_fingerprint = C.state_fingerprint
write_json = C.write_json
read_json = C.read_json
write_csv = C.write_csv
atomic_torch_save = C.atomic_torch_save
track_rel = C.track_rel
jsonable = C.jsonable
log = C.log
rmse = C.rmse
mae = C.mae
lr_at = C.lr_at
steps_per_epoch = C.steps_per_epoch
set_seeds = C.set_seeds
state_cpu = C.state_cpu
git_commit = C.git_commit
git_dirty = C.git_dirty
environment = C.environment


def rel(path: Path | str) -> str:
    return C.track_rel(path)


def total_steps(epochs: int = EPOCHS) -> int:
    return int(steps_per_epoch(DATASET)) * int(epochs)


def percentile_summary(values: np.ndarray) -> dict[str, Any]:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.size == 0:
        return {"n": 0}
    out = {
        "n": int(v.size),
        "mean": float(v.mean()),
        "std": float(v.std(ddof=0)),
        "min": float(v.min()),
        "max": float(v.max()),
        "max_over_min": float(v.max() / v.min()) if v.min() > 0 else None,
    }
    for q in (1, 5, 25, 50, 75, 95, 99):
        out[f"p{q:02d}"] = float(np.percentile(v, q))
    out["n_distinct"] = int(np.unique(v).size)
    return out


def effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64)
    return float(w.sum() ** 2 / np.sum(w * w))


def dump_json_text(payload: Any) -> str:
    return json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"


def recipe_of_record(*, k_train: int, weight_sha256: str) -> dict[str, Any]:
    """Frozen recipe of the single training arm (PROTOCOL section 5)."""
    out = {k: (list(v) if isinstance(v, tuple) else v) for k, v in RECIPE.items()}
    out.update(
        {
            "round_id": ROUND_ID,
            "protocol_id": PROTOCOL_ID,
            "dataset": DATASET,
            "arm": "SCAFFOLD_BALANCED",
            "seed": SEED,
            "steps_per_epoch": int(steps_per_epoch(DATASET)),
            "total_steps": int(total_steps(EPOCHS)),
            "loss": "weighted MSE on standardised targets",
            "weight_rule": WEIGHT_RULE,
            "k_train": int(k_train),
            "n_train": int(N_TRAIN),
            "weight_sha256": str(weight_sha256),
            "lr_step0": lr_at(0, total_steps(EPOCHS)),
            "lr_final_step": lr_at(total_steps(EPOCHS), total_steps(EPOCHS)),
            "test_role": "NOT_READ_NOT_SCORED",
        }
    )
    return out
