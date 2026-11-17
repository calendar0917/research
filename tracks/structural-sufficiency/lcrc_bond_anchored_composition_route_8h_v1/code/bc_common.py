"""Identity, frozen constants, read-only reuse paths and IO for the BAJC round.

Round: ``lcrc-bond-anchored-composition-route-8h-v1``.

Nothing scientific is computed here.  Every frozen scientific constant (branch
form, institution recipes, seeds, decision scales) is declared in this module or
in a sibling module and hashed into ``analysis/protocol_freeze.json`` by
:mod:`bc_prepare` *before* the first formal training unit.

Read-only reuse (never redefined, never retrained)
-------------------------------------------------
* ``lcrc_explanation_task_relevance_sprint_v1`` -> ``xr_common`` installs the
  whole frozen stack on ``sys.path`` and owns the verified checkpoint identity
  table (the two C1 teachers: experimental logD and WC_LOGP).
* ``gnn_route_clarification_8h_v1`` -> ``gr_data`` (verified official split
  caches, Train-only target scaler, the ``TargetOverride`` WC_LOGP bundle) and
  ``gr_models`` (the frozen ``arm A`` C1 model and its canonical seed-0 state).
* ``lcrc_readout_context_pilot_small_v1`` -> ``cr_models.ArmA`` / ``LocalTrunk``
  (the frozen C1 atom/bond/S1/S2/LocalMLP/``sum_v f``/``rho`` function) and its
  canonical initial state builder.
* ``lcrc_auto_subgraph_faithfulness_pilot_v1`` -> ``as_data`` (the verified
  S1/S2 occurrence -> physical-bond map and the gated collate) and ``as_gate``
  (``CONDITIONAL_S1_S2_EVIDENCE_MASK``, the already validated physical-bond
  evidence gate that section C4 of the task book reuses).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------
ROUND_ID = "lcrc-bond-anchored-composition-route-8h-v1"
PROTOCOL_ID = ROUND_ID
STUDY_ID = ROUND_ID

CODE_DIR = Path(__file__).resolve().parent
ROUND_DIR = CODE_DIR.parent
TRACK_DIR = ROUND_DIR.parent
REPO_ROOT = TRACK_DIR.parents[1]

RESULTS_DIR = ROUND_DIR / "results"
ANALYSIS_DIR = ROUND_DIR / "analysis"
CACHE_DIR = ROUND_DIR / "cache"
CONFIG_DIR = ROUND_DIR / "configs"
FROZEN_CONFIG = CONFIG_DIR / "frozen.yaml"

# ---------------------------------------------------------------------------
# read-only source rounds
# ---------------------------------------------------------------------------
SS = TRACK_DIR
SPRINT_DIR = SS / "lcrc_explanation_task_relevance_sprint_v1"
SPRINT_CODE = SPRINT_DIR / "code"
ROUTE_DIR = SS / "gnn_route_clarification_8h_v1"
ROUTE_CODE = ROUTE_DIR / "code"
RC_DIR = SS / "lcrc_readout_context_pilot_small_v1"
RC_CODE = RC_DIR / "code"
PILOT_DIR = SS / "lcrc_auto_subgraph_faithfulness_pilot_v1"
PILOT_CODE = PILOT_DIR / "code"
PREP_DIR = SS / "lcrc_cross_task_preparation_v1"
PREP_CODE = PREP_DIR / "code"
DISSECT_DIR = SS / "lcrc_lipo_model_dissection_v1"
DISSECT_CODE = DISSECT_DIR / "code"
SQL_DIR = REPO_ROOT / "tracks/structural-query-learning"
SQL_CODE = SQL_DIR / "code"

#: The two frozen C1 teachers this round loads (never retrains).
TEACHER_LIPO = "L_lipophilicity"
TEACHER_WC = "L_WC_LOGP"
GINE_REFERENCE = "G0_lipophilicity"

# ---------------------------------------------------------------------------
# task / data identity
# ---------------------------------------------------------------------------
DATASET = "ogbg-mollipo"
SPLITS: tuple[str, ...] = ("train", "valid", "test")
EXPECTED_SPLIT_SIZES = {"train": 3360, "valid": 420, "test": 420}
EXPECTED_TOTAL_ROWS = 4200
TASKS: tuple[str, ...] = ("lipo", "wc_logp")
TASK_TARGET_NAME = {
    "lipo": "official ogbg-mollipo experimental label (exp, experimental logD)",
    "wc_logp": "RDKit Wildman-Crippen logP (computed property; not experimental logD)",
}
TASK_ORACLE_PROPERTY = {"wc_logp": "WC_LOGP"}
TASK_TEACHER = {"lipo": TEACHER_LIPO, "wc_logp": TEACHER_WC}

# --- Stage C only: the conditional ESOL transfer ---------------------------
#: ``esol`` is *not* part of the round's main comparison.  It is the pre-registered
#: conditional transfer task of PROTOCOL.md section 4 and is only ever trained in
#: the E2E institution (there is no ESOL teacher and no historical ESOL baseline is
#: retrained or compared against).
DATASET_ESOL = "ogbg-molesol"
ESOL_SPLIT_SIZES = {"train": 902, "valid": 113, "test": 113}
TASK_DATASET = {"lipo": DATASET, "wc_logp": DATASET, "esol": DATASET_ESOL}
TASK_TARGET_NAME_EXTRA = {
    "esol": "official ogbg-molesol label (ESOL aqueous solubility), its own Train-only scaler",
}

# ---------------------------------------------------------------------------
# the two research models and the two training institutions
# ---------------------------------------------------------------------------
VARIANTS: tuple[str, ...] = ("JOINT", "MARGINAL")
INSTITUTIONS: tuple[str, ...] = ("FROZEN", "E2E")
SEEDS: dict[str, tuple[int, ...]] = {"FROZEN": (0,), "E2E": (0, 1)}
VARIANT_NOTE = {
    "JOINT": "each three-bond path event uses its real second S2 occurrence (p,q) and (p_rev,q_rev)",
    "MARGINAL": (
        "identical events/anchors/first path/closure4/count; the second S2 occurrence is replaced by "
        "the molecule-level mean of the second-path representation per direction"
    ),
}

# ---------------------------------------------------------------------------
# frozen architecture constants (hashed into the protocol freeze)
# ---------------------------------------------------------------------------
WIDTH = 64               # C1 local representation width h_v
STAT_RANK = 32
ATOM_OUT = 96
S2_TYPED_DIM = 3 * STAT_RANK + 8          # type 32 + end 32 + joint 32 + closure 8
S2_CONTENT_DIM = ATOM_OUT                 # a_end * path_gate(code)
S2_EVIDENCE_DIM = S2_TYPED_DIM + S2_CONTENT_DIM   # 200: the frozen typed-S2 record r_p
BOND_EVIDENCE_DIM = 96                    # C1 bond encoder output b_e
PAIR_HIDDEN = 32                          # u(r) width
BOND_HIDDEN = 16                          # v(b) width
PAIR_INPUT_DIM = PAIR_HIDDEN + PAIR_HIDDEN + PAIR_HIDDEN + BOND_HIDDEN + 1   # 113
PSI_HIDDEN = 64
PSI_OUT = 32
C1_PARAMETERS = 228674
C1_TRUNK_PARAMETERS = 213376
C1_READOUT_PARAMETERS = 15298
BRANCH_MAX_FRACTION = 0.15

DROPOUT = 0.10
BRANCH_DROPOUT = 0.10
BRANCH_PENALTY = 0.02

# ---------------------------------------------------------------------------
# frozen recipes
# ---------------------------------------------------------------------------
#: Institution E: verbatim Rf from ``gnn_route_clarification_8h_v1``.
RECIPE_E2E: dict[str, Any] = {
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
    "shuffle_generator": "np.random.default_rng(0).permutation over Train split-local rows, one per epoch",
    "target": "Train-only mean/std normalisation of the task label",
}

#: Institution F: only the new composition branch is trained; the C1 teacher is frozen.
RECIPE_FROZEN: dict[str, Any] = {
    "optimizer": "AdamW",
    "adam_betas": (0.9, 0.999),
    "adam_eps": 1e-8,
    "lr": 1.0e-3,
    "weight_decay": 1.0e-5,
    "batch_size": 128,
    "eval_batch_size": 256,
    "gradient_clip": 5.0,
    "epochs": 180,
    "branch_dropout": BRANCH_DROPOUT,
    "branch_penalty": BRANCH_PENALTY,
    "teacher_mode": "eval(); every parameter and buffer frozen; epoch 0 == the frozen C1 prediction",
    "selection": "minimum Val RMSE over epochs 0..180; epoch 0 is a legal candidate",
    "target": "full standardised prediction MSE (y_base frozen, only the residual is learnable)",
}

# ---------------------------------------------------------------------------
# frozen seeds / statistics
# ---------------------------------------------------------------------------
TRAIN_SEED_DEFAULT = 0
INIT_SEED_CANONICAL = 0
BRANCH_INIT_SEED = 20261118          # one isolated stream, identical for JOINT and MARGINAL
BRANCH_DROPOUT_SEED = 20261119       # isolated generator so the C1 stream is untouched
BOOTSTRAP_SEED = 20261118
BOOTSTRAP_DRAWS = 2000
ORACLE_CONTROL_SEED = 20261118
MAX_ORACLE_CONTROLS = 16

# ---------------------------------------------------------------------------
# pre-declared decision scales (task book section 8 / 10; never universal)
# ---------------------------------------------------------------------------
STAGE_C_JOINT_VS_MARGINAL = 0.010
STAGE_C_JOINT_VS_C1 = 0.015
STAGE_C_BOOTSTRAP_PROB = 0.80
PRACTICAL_JOINT_VS_C1 = 0.020
PRACTICAL_JOINT_VS_MARGINAL = 0.010
MIN_PAIRING_SUPPORT = 0.30
MIN_ORACLE_ELIGIBLE_FRACTION = 0.70
#: A Train gain that is paired with a Valid loss of at least this size counts as
#: a *reversed* gain for the fourth (anti-degeneracy) Stage-C trigger.
REVERSAL_TOLERANCE = 0.005


# ---------------------------------------------------------------------------
# import path discipline / frozen module access
# ---------------------------------------------------------------------------
_PATHS_READY = False
_MODULES: dict[str, Any] | None = None


def install_paths() -> None:
    global _PATHS_READY
    if _PATHS_READY:
        return
    for path in (str(SPRINT_CODE), str(ROUTE_CODE), str(RC_CODE), str(PILOT_CODE), str(PREP_CODE), str(DISSECT_CODE), str(SQL_CODE)):
        if path not in sys.path:
            sys.path.insert(0, path)
    _PATHS_READY = True


def modules() -> dict[str, Any]:
    """The frozen read-only stack (imported once)."""
    global _MODULES
    if _MODULES is not None:
        return _MODULES
    install_paths()
    import as_common
    import as_data
    import as_gate
    import cr_common
    import cr_data
    import cr_models
    import feature_spec
    import gr_common
    import gr_data
    import gr_models
    import lcrc_data
    import molecular_graph
    import sq_data
    import sq_lcrc
    import xr_common

    _MODULES = {
        "as_common": as_common,
        "as_data": as_data,
        "as_gate": as_gate,
        "cr_common": cr_common,
        "cr_data": cr_data,
        "cr_models": cr_models,
        "feature_spec": feature_spec,
        "gr_common": gr_common,
        "gr_data": gr_data,
        "gr_models": gr_models,
        "lcrc_data": lcrc_data,
        "molecular_graph": molecular_graph,
        "sq_data": sq_data,
        "sq_lcrc": sq_lcrc,
        "xr_common": xr_common,
    }
    return _MODULES


# ---------------------------------------------------------------------------
# IO / hashing
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


def sha256_state(state: Mapping[str, Any], *, keys: Any = None) -> str:
    return _c().sha256_state(state, keys=keys)


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


def rmse(pred, target) -> float:
    return _c().rmse(pred, target)


def mae(pred, target) -> float:
    return _c().mae(pred, target)


def target_scaler(labels) -> dict[str, float]:
    return _c().target_scaler(labels)


def set_global_seeds(seed: int = TRAIN_SEED_DEFAULT) -> dict[str, Any]:
    return _c().set_global_seeds(seed)


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


class Ledger:
    """Thin re-export of the frozen ledger (stage timings / GPU seconds)."""

    def __new__(cls, path, **kwargs):
        return _c().Ledger(path, **kwargs)


def stage_timer(ledger, name: str, **kwargs):
    return _c().stage_timer(ledger, name, **kwargs)


# ---------------------------------------------------------------------------
# result paths
# ---------------------------------------------------------------------------


def unit_dir(institution: str, task: str, variant: str, seed: int) -> Path:
    return RESULTS_DIR / institution.lower() / task / f"{variant}_s{seed}"


def unit_paths(institution: str, task: str, variant: str, seed: int) -> dict[str, Path]:
    base = unit_dir(institution, task, variant, seed)
    return {
        "dir": base,
        "checkpoint_best": base / "checkpoint_best_val.pt",
        "checkpoint_final": base / "checkpoint_final.pt",
        "curve": base / "curve.csv",
        "status": base / "status.json",
        "predictions": base / "predictions.npz",
        "epoch0": base / "epoch0_predictions.npz",
        "ledger": base / "compute_ledger.json",
        "trajectory": base / "trajectory.json",
    }


def relpath(path: Path | str) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


__all__ = [name for name in dir() if not name.startswith("_")]
