"""Identity, frozen constants, paths and IO for the role-preserving C1 pilot.

Nothing scientific is computed here.  Every frozen scientific constant (the
role split rule, the control hash, the gamma form, the C1 readout, the Rf
recipe, the decision scales) is declared in this module or in a sibling module
and hashed into ``analysis/protocol_freeze.json`` by :mod:`rp_prepare`.

Read-only reuse
---------------
The whole C1 pipeline (atom/bond encoders, S1, typed S2, LocalMLP, ``sum_v f``
and ``rho([g, mean h, mean h^2, log1p n])``) and the frozen topology partition +
S2 ownership cache are *consumed*, never redefined:

* ``lcrc_readout_context_pilot_small_v1`` — the canonical C1 initial state, the
  C1 ``LocalTrunk`` and the original ``ArmA`` readout;
* ``lcrc_relation_interface_routing_pilot_v1`` — the frozen per-split topology
  routing cache (S2 ``path_cross`` / ``path_owner1`` / ``path_owner2``) and the
  exact partition / routing implementation used to verify it;
* ``lcrc_cross_task_preparation_v1`` / ``structural-query-learning`` /
  ``lcrc_lipo_model_dissection_v1`` — the frozen data, feature spec and modules.

Path discipline: every path is resolved from this file, so the module behaves
identically in the local checkout and in the remote compute checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------
ROUND_ID = "lcrc-relation-role-preserving-pilot-v1"
PROTOCOL_ID = ROUND_ID
STUDY_ID = ROUND_ID

CODE_DIR = Path(__file__).resolve().parent
ROUND_DIR = CODE_DIR.parent
TRACK_DIR = ROUND_DIR.parent                       # tracks/structural-sufficiency
REPO_ROOT = TRACK_DIR.parents[1]                   # repository root

RESULTS_DIR = ROUND_DIR / "results"
ANALYSIS_DIR = ROUND_DIR / "analysis"
CACHE_DIR = ROUND_DIR / "cache"
CONFIG_DIR = ROUND_DIR / "configs"
FROZEN_CONFIG = CONFIG_DIR / "frozen.yaml"

# ---------------------------------------------------------------------------
# read-only reuse roots (never written by this round)
# ---------------------------------------------------------------------------
SQL_DIR = REPO_ROOT / "tracks/structural-query-learning"
SQL_CODE = SQL_DIR / "code"
SQL_DATA = SQL_DIR / "data"

RC_DIR = REPO_ROOT / "tracks/structural-sufficiency/lcrc_readout_context_pilot_small_v1"
RC_CODE = RC_DIR / "code"
RC_RESULTS = RC_DIR / "results"

PREP_DIR = REPO_ROOT / "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1"
PREP_CODE = PREP_DIR / "code"
DISSECT_DIR = REPO_ROOT / "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1"
DISSECT_CODE = DISSECT_DIR / "code"

#: The frozen topology partition / S2 routing cache of the previous round.
PRED_DIR = REPO_ROOT / "tracks/structural-sufficiency/lcrc_relation_interface_routing_pilot_v1"
PRED_CODE = PRED_DIR / "code"
PRED_CACHE = PRED_DIR / "cache"
PRED_ANALYSIS = PRED_DIR / "analysis"
PRED_MANIFEST = PRED_CACHE / "manifest.json"
PRED_SUPPORT = PRED_ANALYSIS / "partition_support.json"

ROUTE_DIR = REPO_ROOT / "tracks/structural-sufficiency/gnn_route_clarification_8h_v1"
ROUTE_ANALYSIS = ROUTE_DIR / "analysis"
ROUTE_STAGE1 = ROUTE_ANALYSIS / "tables/stage1.csv"

ORACLE_CONTRIB_NPZ = ROUTE_ANALYSIS / "oracle_contrib/ogbg-mollipo.npz"
ORACLE_TASK_DIR = REPO_ROOT / "tracks/structural-sufficiency/lcrc_lipo_oracle_task_relevance_public122_s0_v1"
WC_LOG_PRED_CSV = ORACLE_TASK_DIR / "analysis/wc_logp_oracle.csv"
WC_LOG_TRAIN_JSON = ORACLE_TASK_DIR / "runs/wc_logp/train.json"

#: Read-only historical C1 baselines (numbers only; never retrained here).
C1_LIPO_TRAJECTORY = RC_RESULTS / "ogbg-mollipo/A/trajectory.json"

# ---------------------------------------------------------------------------
# data / tasks / models
# ---------------------------------------------------------------------------
DATASET = "ogbg-mollipo"
SPLITS: tuple[str, ...] = ("train", "valid", "test")
EXPECTED_SPLIT_SIZES = {"train": 3360, "valid": 420, "test": 420}
EXPECTED_TOTAL_ROWS = 4200
TASKS: tuple[str, ...] = ("lipo", "wc_logp")
TASK_TARGET_NAME = {
    "lipo": "official ogbg-mollipo experimental label (exp)",
    "wc_logp": "RDKit Wildman-Crippen logP (computed property)",
}
MODELS: tuple[str, ...] = ("RR_REAL", "RR_CONTROL")
MODEL_ROLE_KEY = {"RR_REAL": "path_role_real", "RR_CONTROL": "path_role_control"}
MODEL_ROLE_SOURCE = {
    "RR_REAL": "frozen topology block ownership: owner(e1)==owner(e2) -> group 0 (internal), else group 1 (cross)",
    "RR_CONTROL": "label-free S2 type hash of (path_cat, path_end63, path_closure) -> group 0/1",
}

#: SHA-256 of the frozen routing cache reused from the previous round.  The
#: prepare stage re-verifies both the file bytes and a full rebuild of every
#: array, so a silently edited cache cannot enter this round.
EXPECTED_ROUTING_CACHE_SHA256 = {
    "train": "8fc85c7fba7c8bd11e230f0f278f9be53df70d181eae568eeedb29fbbc2500dc",
    "valid": "5e43da2eb3bd99e36ae761c2bdef22833d6f263821172129fad5ec28967d1ec3",
    "test": "890407976d3e9b9c136b83eb5390c099af39358cfe91433914d6aec073fca6cd",
}
#: Frozen fingerprints of the *values* in the routing cache (from the previous
#: round's committed partition support record).
EXPECTED_ROUTING_FINGERPRINT = {
    "train": "31238683192b53554fd68fcf70c47c88ddd70ff0cbfffb5f0e848902bc39e160",
    "valid": "12dc6f35dc2c53dd21df9ce954f8d184f9c0f5ff34d7dfd057756e34153a41e5",
    "test": "8cdaae5a092ad5fe5bfd1eb24c09872bac8fee3c100698cde2cbeb5f7eef54a8",
}

# ---------------------------------------------------------------------------
# frozen seeds
# ---------------------------------------------------------------------------
TRAIN_SEED = 0
INIT_SEED_CANONICAL = 0          # the C1 canonical initial state (read-only reuse)
BOOTSTRAP_SEED = 20261117
ORACLE_CONTROL_SEED = 20261117

# ---------------------------------------------------------------------------
# frozen architecture constants (hashed into the protocol freeze)
# ---------------------------------------------------------------------------
WIDTH = 64
STAT_RANK = 32
ATOM_OUT = 96
S2_TYPED_DIM = 3 * STAT_RANK + 8        # type 32 + end 32 + joint 32 + closure 8
S2_CONTENT_DIM = ATOM_OUT               # a_end * path_gate
S2_ROLE_DIM = S2_TYPED_DIM + S2_CONTENT_DIM   # 200: the S2 columns of the C1 input
LOCAL_INPUT_DIM = 433                   # frozen C1 local column count
RHO_HIDDEN = 64
DROPOUT = 0.10
N_ROLE_CLASSES = 2

#: Fixed, label-free salt of the CONTROL grouping.  Chosen *before* any outcome
#: was seen; never re-drawn, never tuned on labels or on the REAL group sizes.
CONTROL_SALT = "lcrc-relation-role-preserving-pilot-v1/s2-type-hash/control"
CONTROL_TABLE_FILE = "s2_control_type_hash_v1.npz"
CONTROL_MANIFEST = "s2_control_type_hash_v1.json"
ROLE_NAMES = {"RR_REAL": ("internal", "cross"), "RR_CONTROL": ("hash0", "hash1")}

#: Training recipe Rf, copied verbatim from ``gnn_route_clarification_8h_v1``.
RECIPE: dict[str, Any] = {
    "optimizer": "AdamW",
    "adam_betas": (0.9, 0.999),
    "adam_eps": 1e-8,
    "peak_lr": 1.0e-3,
    "min_lr": 1.0e-5,
    "warmup_epochs": 5,
    "weight_decay": 1.0e-5,
    "batch_size": 128,
    "eval_batch_size": 256,
    "dropout": DROPOUT,
    "gradient_clip": 5.0,
    "epochs": 360,
    "scheduler": "per_step_linear_warmup_then_cosine",
    "amp": False,
    "ema": False,
    "swa": False,
    "soup": False,
    "early_stopping": False,
    "selection": "minimum Val RMSE; strictly smaller replaces, an exact tie keeps the earlier epoch",
    "test_policy": "always (DEVELOPMENT_TEST; never used for selection or tuning)",
    "shuffle_generator": "np.random.default_rng(0).permutation over Train split-local rows, one per epoch",
}

# ---------------------------------------------------------------------------
# pre-declared decision scales (pilot-local, never universal thresholds)
# ---------------------------------------------------------------------------
RETENTION_FACTOR = 1.10              # Lipo Valid RMSE <= 1.10 x C1(Rf)
BOOTSTRAP_DRAWS = 2000
MIN_ORACLE_CONTROLS = 8
MAX_ORACLE_CONTROLS = 16
ORACLE_POSITIVE_FRACTION = 0.60      # >= 60% matched molecules with positive enrichment
INCREMENT_PROB = 0.95                # P(REAL better) required for an increment claim
INCREMENT_ALPHA = 0.05               # 95% paired bootstrap CI must exclude 0

#: Attribution / readout tolerances (frozen).
TRACE_TOL_BASE = 1e-5
TRACE_TOL_REL = 1e-6


def trace_tolerance(value, base: float = TRACE_TOL_BASE, rel: float = TRACE_TOL_REL):
    arr = np.asarray(value, dtype=np.float64)
    return base + rel * np.maximum(1.0, np.abs(arr))


# ---------------------------------------------------------------------------
# imports of the read-only reuse stack
# ---------------------------------------------------------------------------
_PATHS_READY = False
_PRED_READY = False


def prepare_imports() -> None:
    global _PATHS_READY
    if _PATHS_READY:
        return
    for path in (str(RC_CODE), str(SQL_CODE), str(PREP_CODE), str(DISSECT_CODE), str(PRED_CODE)):
        if path not in sys.path:
            sys.path.insert(0, path)
    _PATHS_READY = True


def prepare_pred_imports() -> None:
    """Put the previous round's partition / routing implementation on ``sys.path``."""
    prepare_imports()
    global _PRED_READY
    if _PRED_READY:
        return
    for path in (str(PRED_CODE),):
        if path not in sys.path:
            sys.path.insert(0, path)
    _PRED_READY = True


def modules() -> dict[str, Any]:
    prepare_imports()
    import adapters
    import cr_common
    import cr_data
    import cr_models
    import dissect_common
    import dissect_model
    import feature_spec
    import lcrc_data
    import molecular_graph
    import rr_data
    import sq_data
    import sq_lcrc

    return {
        "adapters": adapters,
        "cr_common": cr_common,
        "cr_data": cr_data,
        "cr_models": cr_models,
        "dissect_common": dissect_common,
        "dissect_model": dissect_model,
        "feature_spec": feature_spec,
        "lcrc_data": lcrc_data,
        "molecular_graph": molecular_graph,
        "rr_data": rr_data,
        "sq_data": sq_data,
        "sq_lcrc": sq_lcrc,
    }


def pred_modules() -> dict[str, Any]:
    prepare_pred_imports()
    import rr_partition
    import rr_routing

    return {"rr_partition": rr_partition, "rr_routing": rr_routing}


# ---------------------------------------------------------------------------
# IO / hashing (delegated to the frozen read-only helper module)
# ---------------------------------------------------------------------------


def _c():
    return modules()["cr_common"]


def sha256_bytes(payload: bytes) -> str:
    return _c().sha256_bytes(payload)


def sha256_file(path) -> str:
    return _c().sha256_file(path)


def sha256_text(text: str) -> str:
    return _c().sha256_text(text)


def sha256_array(array: np.ndarray) -> str:
    return _c().sha256_array(array)


def sha256_json(payload: Any) -> str:
    return _c().sha256_json(payload)


def jsonable(value: Any) -> Any:
    return _c().jsonable(value)


def read_json(path) -> Any:
    return _c().read_json(path)


def atomic_write_json(path, payload: Any):
    return _c().atomic_write_json(path, payload)


def atomic_write_text(path, text: str):
    return _c().atomic_write_text(path, text)


def atomic_write_npz(path, arrays: Mapping[str, Any]):
    return _c().atomic_write_npz(path, arrays)


def atomic_torch_save(path, payload: Any):
    return _c().atomic_torch_save(path, payload)


def write_csv(path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None):
    return _c().write_csv(path, rows, columns)


def read_csv(path) -> list[dict[str, str]]:
    return _c().read_csv(path)


def ensure_dir(path) -> Path:
    return _c().ensure_dir(path)


def sha256_state(state: Mapping[str, Any], *, keys: Any = None) -> str:
    return _c().sha256_state(state, keys=keys)


def set_global_seeds(seed: int = TRAIN_SEED) -> dict[str, Any]:
    return _c().set_global_seeds(seed)


def capture_rng_state() -> dict[str, Any]:
    return _c().capture_rng_state()


def restore_rng_state(payload: Mapping[str, Any]) -> None:
    return _c().restore_rng_state(payload)


def rng_state_hash(**kwargs) -> str:
    return _c().rng_state_hash(**kwargs)


def environment_snapshot() -> dict[str, Any]:
    return _c().environment_snapshot()


def git_commit() -> str | None:
    return _c().git_commit()


def git_branch() -> str | None:
    return _c().git_branch()


def git_dirty_tracked() -> bool | None:
    return _c().git_dirty_tracked()


def lockfile_sha256() -> str | None:
    return _c().lockfile_sha256()


def rmse(pred, target) -> float:
    return _c().rmse(pred, target)


def mae(pred, target) -> float:
    return _c().mae(pred, target)


def target_scaler(labels) -> dict[str, float]:
    return _c().target_scaler(labels)


def Ledger(path, **kwargs):
    return _c().Ledger(path, **kwargs)


def stage_timer(ledger, name: str, **kwargs):
    return _c().stage_timer(ledger, name, **kwargs)


# ---------------------------------------------------------------------------
# cache paths of this round
# ---------------------------------------------------------------------------


def control_table_path() -> Path:
    return CACHE_DIR / CONTROL_TABLE_FILE


def control_manifest_path() -> Path:
    return CACHE_DIR / CONTROL_MANIFEST


def task_results_dir(task: str, model: str) -> Path:
    return RESULTS_DIR / task / model


def prediction_paths(task: str, model: str) -> dict[str, Path]:
    out = {}
    for split in SPLITS:
        out[f"best_{split}"] = task_results_dir(task, model) / f"best_predictions_{split}.npz"
        out[f"final_{split}"] = task_results_dir(task, model) / f"final_predictions_{split}.npz"
    out["trajectory"] = task_results_dir(task, model) / "trajectory.json"
    out["best_checkpoint"] = task_results_dir(task, model) / "checkpoint_best_val.pt"
    out["final_checkpoint"] = task_results_dir(task, model) / "checkpoint_final.pt"
    return out


__all__ = [name for name in dir() if not name.startswith("_")]
