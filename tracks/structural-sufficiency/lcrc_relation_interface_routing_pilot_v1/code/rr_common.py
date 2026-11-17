"""Identity, paths and frozen constants for the relation--interface routing pilot.

Nothing scientific is computed here.  Every frozen scientific constant (the
topology-block rule, the routing rule, the token/readout forms, the Rf recipe,
the selection score, the oracle-control protocol) lives in the sibling modules
and is hashed into ``analysis/protocol_freeze.json`` at prepare time.

Path discipline: every path is resolved from this file, so the module behaves
identically in the local checkout and in the read-only compute checkout on
``res``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROUND_ID = "lcrc-relation-interface-routing-pilot-v1"
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
SQL_RESULTS = SQL_DIR / "results/sq-pilot-small-v1"

RC_DIR = REPO_ROOT / "tracks/structural-sufficiency/lcrc_readout_context_pilot_small_v1"
RC_CODE = RC_DIR / "code"
RC_RESULTS = RC_DIR / "results"

PREP_DIR = REPO_ROOT / "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1"
PREP_CODE = PREP_DIR / "code"
DISSECT_DIR = REPO_ROOT / "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1"
DISSECT_CODE = DISSECT_DIR / "code"
ROUTE_DIR = REPO_ROOT / "tracks/structural-sufficiency/gnn_route_clarification_8h_v1"
ROUTE_ANALYSIS = ROUTE_DIR / "analysis"
ORACLE_CONTRIB_NPZ = ROUTE_ANALYSIS / "oracle_contrib/ogbg-mollipo.npz"
ORACLE_CONTRIB_SUMMARY = ROUTE_ANALYSIS / "oracle_contrib/summary.json"
ORACLE_TASK_DIR = REPO_ROOT / "tracks/structural-sufficiency/lcrc_lipo_oracle_task_relevance_public122_s0_v1"
WC_LOG_PRED_CSV = ORACLE_TASK_DIR / "analysis/wc_logp_oracle.csv"
WC_LOG_TRAIN_JSON = ORACLE_TASK_DIR / "runs/wc_logp/train.json"

#: Read-only historical baselines (numbers only; never retrained here).
C1_LIPO_TRAJECTORY = RC_RESULTS / "ogbg-mollipo/A/trajectory.json"
C1_LIPO_SELECTED_ARM = RC_RESULTS / "ogbg-mollipo/A/trajectory.json"
GINE_ROUTE_STAGE23 = ROUTE_ANALYSIS / "tables/stage23.json"
GINE_ROUTE_PROTOCOL = ROUTE_ANALYSIS / "protocol_freeze.json"
EXPLAIN_ROUTE_LEDGERS = ROUTE_ANALYSIS / "ledgers"

# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
DATASET = "ogbg-mollipo"
SPLITS: tuple[str, ...] = ("train", "valid", "test")
EXPECTED_SPLIT_SIZES = {"train": 3360, "valid": 420, "test": 420}
EXPECTED_TOTAL_ROWS = 4200
EXPECTED_CACHE_SHA256 = {
    "ogbg_mollipo_sq-pilot-cache-v1.npz": "fe002287fa9c581278bc5430f5b4a5b26b35d9d4d5207d138f15956d93123c7c",
    "ogbg_mollipo_sq-pilot-lipo-raw-v1.npz": "3005aa55e766d83a237b1659ad2cabb51b8702eef6d82c203fdecf9cb3c42cc3",
}
SPLIT_FILE = {"train": "train_idx", "valid": "valid_idx", "test": "test_idx"}

#: The two authorised targets of this round (one checkpoint pool each).
TASKS: tuple[str, ...] = ("lipo", "wc_logp")
TASK_DATASET = {"lipo": DATASET, "wc_logp": DATASET}
TASK_TARGET_NAME = {"lipo": "official ogbg-mollipo experimental label (exp)", "wc_logp": "RDKit Wildman-Crippen logP (computed property)"}
MODELS: tuple[str, ...] = ("R1", "R2")

# ---------------------------------------------------------------------------
# frozen seeds / init streams
# ---------------------------------------------------------------------------
TRAIN_SEED = 0
INIT_SEED_CANONICAL = 0          # the C1 canonical initial state (read-only reuse)
INIT_STREAM_BLOCK = 2001         # f_block
INIT_STREAM_IFACE = 2002         # interface evidence projector + interface readout
INIT_STREAM_RHO = 2003           # rho_g
INIT_STREAM_CONTEXT = 2004       # interface context MLP (R2 only)
BOOTSTRAP_SEED = 20261117
ORACLE_CONTROL_SEED = 20261117

# ---------------------------------------------------------------------------
# frozen architecture constants (hashed into the protocol freeze)
# ---------------------------------------------------------------------------
WIDTH = 64
IFACE_EVIDENCE_DIM = 200         # type(32)+end(32)+joint(32)+closure(8)+content(96)
IFACE_WIDTH = 64
CONTEXT_WIDTH = 64
RHO_HIDDEN = 64
DROPOUT = 0.10
BLOCK_SIZE_FEATURES = 2          # log1p(n_bonds), log1p(n_atoms)
BLOCK_TOKEN_DIM = WIDTH + BLOCK_SIZE_FEATURES

#: Training recipe Rf, copied from ``gnn_route_clarification_8h_v1``.
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

#: Decision scales frozen before training (declared, not tuned).
RETENTION_FACTOR = 1.10          # Val RMSE within 1.10x of the frozen C1 baseline
CAPACITY_CONFOUNDED_RATIO = 1.10  # R2/R1 parameter ratio above which capacity confounds
MIN_ORACLE_CONTROLS = 8
MAX_ORACLE_CONTROLS = 16
BOOTSTRAP_DRAWS = 2000
PARTITION_GATE_FRACTION = 0.50   # Valid molecules needing >=2 blocks and >=1 cross S2

#: Oracle/attribution tolerances (frozen).
TRACE_TOL_BASE = 1e-5
TRACE_TOL_REL = 1e-6


def trace_tolerance(value, base: float = TRACE_TOL_BASE, rel: float = TRACE_TOL_REL):
    arr = np.asarray(value, dtype=np.float64)
    return base + rel * np.maximum(1.0, np.abs(arr))


# ---------------------------------------------------------------------------
# imports of the read-only reuse stack
# ---------------------------------------------------------------------------
_PATHS_READY = False


def prepare_imports() -> None:
    global _PATHS_READY
    if _PATHS_READY:
        return
    for path in (str(RC_CODE), str(SQL_CODE), str(PREP_CODE), str(DISSECT_CODE)):
        if path not in sys.path:
            sys.path.insert(0, path)
    _PATHS_READY = True


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
        "sq_data": sq_data,
        "sq_lcrc": sq_lcrc,
    }


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
# routing cache identity
# ---------------------------------------------------------------------------


def routing_cache_path(split: str) -> Path:
    return CACHE_DIR / f"ogbg_mollipo_{split}_topology-routing-cache-v1.npz"


def task_results_dir(task: str, model: str) -> Path:
    return RESULTS_DIR / task / model


def expected_task_cache_sha256() -> dict[str, str]:
    """SHA-256 of the frozen routing caches, read from the prepare manifest."""
    path = ANALYSIS_DIR / "partition_support.json"
    if not path.is_file():
        return {}
    payload = read_json(path)
    return {str(k): str(v) for k, v in payload.get("routing_cache_sha256", {}).items()}


__all__ = [name for name in dir() if not name.startswith("_")]
