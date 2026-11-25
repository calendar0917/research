"""Shared constants, frozen choices, reuse ledger and hashing.

Round: ``lcrc-support-joint-function-v0`` (``lsj-v0``).

Everything scientific (tasks, arms, seed, epochs, recipe, the Train-only
support construction, the fallback rule, the candidate-selection rule for the
interaction contrasts, the bootstrap, the budgets) is declared here **before**
the first formal training run.  Nothing is chosen from a score.

Reused read-only
----------------
* ``lcrc_predictive_trace_benchmark_v1``: the ``ptb_model.PTBLCRC`` A0 graph,
  the ``ptb_train_mol`` training unit (recipe order, shuffle generator, loss),
  the MolHIV data view and ``mb2_metrics``.
* ``lcrc_lipo_model_dissection_v1``: the ``dissect_model.DissectLCRC`` C1 graph,
  ``dissect_data`` (public122 split, structure caches, target normaliser),
  ``dissect_train`` (360-epoch R3 loop) and its reused ``r122_C1_s0`` baseline.
* ``lcrc_sparse_identity_v1``: the reused MolHIV ``A0`` s0 baseline
  (``mol_ogbg-molhiv_s0_A0``) plus its Valid/Test predictions.
* ``lcrc_molnet_benchmark_pilot_v2`` / ``lcrc_cross_task_preparation_v1``:
  structure caches, ``lipo_v1`` FeatureSpec, collate, metrics.

No upstream file is modified by this round.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROUND_ID = "lcrc-support-joint-function-v0"
PROTOCOL_ID = ROUND_ID

CODE_DIR = Path(__file__).resolve().parent
TRACK = CODE_DIR.parent
REPO = TRACK.parents[2]

PTB = REPO / "tracks/structural-sufficiency/lcrc_predictive_trace_benchmark_v1"
PTB_CODE = PTB / "code"
DISSECT = REPO / "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1"
DISSECT_CODE = DISSECT / "code"
SPARSE = REPO / "tracks/structural-sufficiency/lcrc_sparse_identity_v1"
SJ_V0 = REPO / "tracks/structural-sufficiency/lcrc_support_joint_function_v0"

if str(PTB_CODE) not in sys.path:
    sys.path.insert(0, str(PTB_CODE))

import ptb_common as PC  # noqa: E402  (inserts dissect / molnet_v2 / prep paths)

import dissect_common as DC  # noqa: E402,F401
import dissect_data as DD  # noqa: E402,F401
import dissect_model as DM  # noqa: E402,F401
import lcrc_data as PREP_DATA  # noqa: E402,F401
import mb2_metrics as MET  # noqa: E402,F401
import ptb_data as PD  # noqa: E402,F401
import ptb_model as PM  # noqa: E402,F401

ANALYSIS = TRACK / "analysis"
CONFIGS = TRACK / "configs"
RUNS = TRACK / "runs"
TESTS = TRACK / "tests"

#: frozen upstream files consumed read-only (hashed into every run record)
REUSED_MODULES: dict[str, Path] = dict(PC.REUSED_MODULES)
REUSED_MODULES.update(
    {
        "ptb/ptb_common.py": PTB_CODE / "ptb_common.py",
        "ptb/ptb_data.py": PTB_CODE / "ptb_data.py",
        "ptb/ptb_model.py": PTB_CODE / "ptb_model.py",
        "ptb/ptb_train_mol.py": PTB_CODE / "ptb_train_mol.py",
    }
)

# ---------------------------------------------------------------------------
# tasks (official splits only, never re-split)
# ---------------------------------------------------------------------------

SEED = 0
N_SEEDS = (SEED,)

TASKS: dict[str, dict[str, Any]] = {
    "hiv": {
        "key": "hiv",
        "dataset": "ogbg-molhiv",
        "family": "ptb",
        "variant": "A0",
        "task": "binary",
        "epochs": 100,
        "primary": "roc_auc",
        "secondary": "pr_auc",
        "direction": "higher",
        "units": "binary activity",
        "split_sizes": {"train": 32901, "valid": 4113, "test": 4113},
        "test_role": "DEVELOPMENT_TEST",
        "test_source": "official OGB scaffold test split",
    },
    "lipo": {
        "key": "lipo",
        "dataset": "Lipophilicity",
        "family": "dissect",
        "variant": "C1",
        "task": "regression",
        "epochs": 360,
        "primary": "rmse",
        "secondary": "mae",
        "direction": "lower",
        "units": "logD",
        "split_sizes": {"train": 3360, "valid": 420, "test": 420},
        "test_role": "DEVELOPMENT_TEST",
        "test_source": "published ChemBench repeat 122 test split (already exposed in earlier rounds)",
        "repeat": 122,
    },
}
TASK_ORDER = ("hiv", "lipo")

#: frozen R3 recipe; the task epochs are per-task (100 / 360)
RECIPE: dict[str, Any] = dict(DC.RECIPE)
RECIPE["epochs_by_task"] = {k: int(v["epochs"]) for k, v in TASKS.items()}

#: seeds are set at the point the reused baseline loop sets them
#: (ptb_train_mol: before construction; dissect_train: after construction).
SEED_TIMING = {"hiv": "before_build", "lipo": "after_build"}

# ---------------------------------------------------------------------------
# frozen full-model layouts (from dissect_common; the joint block is derived
# from the model's own column map at runtime, never hardcoded)
# ---------------------------------------------------------------------------

LOCAL_LAYOUT = dict(DC.LOCAL_LAYOUT)
JOINT_FULL_COLS = tuple(range(*DC.LOCAL_LAYOUT["s2_joint"]))
STAT_RANK = 32  # dissect_model.DEFAULT_CONFIG["stat_rank"]
WIDTH = 64  # dissect_model.DEFAULT_CONFIG["width"]

# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------

ARM_A = "A"
ARM_B = "B"
ARM_C = "C"
ARMS = (ARM_A, ARM_B, ARM_C)
TRAINED_ARMS = (ARM_B, ARM_C)
ARM_LABEL = {
    ARM_A: "ORIGINAL LCRC (reused frozen baseline, no retraining)",
    ARM_B: "JOINT table over the Train-only Omega (support-indexed free matrix)",
    ARM_C: "ADDITIVE table J = U Theta (orthogonal marginal projection only)",
}

# ---------------------------------------------------------------------------
# Train-only support construction (frozen definitions)
# ---------------------------------------------------------------------------


def key_field(p: int, n_nbr_codes: int) -> int:  # pragma: no cover - documentation only
    raise NotImplementedError("see sj_support.py")


SUPPORT = {
    "grouping": "p = path_cat, e = path_end63 (production S2 integer codes, no new fields)",
    "omega": "the set of distinct (p, e) pairs that actually occur in the Train split",
    "weights": "type-equal: every Omega element has weight 1 (no occurrence / molecule weighting)",
    "design": (
        "X has one row per Omega element and exactly two unit entries per row: the column of its p "
        "and the column of its e (bipartite incidence matrix, |P|+|E| columns)"
    ),
    "svd": "float64 reduced SVD of X; U = left singular vectors with s > 1e-10 * s_max",
    "projection": "P = U U^T; J_add = P J; J_int = J - P J",
    "fallback": {
        "same_component": "least-squares marginal fit Jhat = argmin ||X Theta - J_table||_F; J(p,e) = Theta[p] + Theta[e]",
        "otherwise": "type-equal mean over the Omega rows of J_table",
    },
    "legality": "bond_type_2 is encoded in both p and e; only p % n_bond == e % n_bond is legal",
    "unseen_policy": "no new learnable parameter; the fallback is a deterministic function of J_table",
}

# ---------------------------------------------------------------------------
# reused frozen baselines (arm A).  Never retrained here.
# ---------------------------------------------------------------------------

REUSE: dict[str, dict[str, Any]] = {
    "hiv": {
        "arm": ARM_A,
        "round": "lcrc-sparse-identity-v1",
        "run_dir": SPARSE / "runs/mol_ogbg-molhiv_s0_A0",
        "run_json": SPARSE / "runs/mol_ogbg-molhiv_s0_A0/run.json",
        "curve": SPARSE / "runs/mol_ogbg-molhiv_s0_A0/curve.csv",
        "checkpoint": SPARSE / "runs/mol_ogbg-molhiv_s0_A0/checkpoint.pt",
        "valid_predictions": SPARSE / "analysis/predictions/A0_s0_valid.npz",
        "test_predictions": SPARSE / "analysis/predictions/A0_s0_test.npz",
        "test_scores": SPARSE / "analysis/test_scores.json",
        "validation_metrics": SPARSE / "analysis/validation_metrics.json",
        "expected": {
            "checkpoint_sha256": "2340e38e85ed8c0f2ee2cfeab1bbb8667e8383d2e62cc81d4e8feb5aa40599ac",
            "parameters": 247106,
            "best_epoch": 53,
            "split_sizes": {"train": 32901, "valid": 4113, "test": 4113},
            "seed": 0,
        },
    },
    "lipo": {
        "arm": ARM_A,
        "round": "lcrc-lipo-model-dissection-v1",
        "run_dir": DISSECT / "runs/r122_C1_s0",
        "run_json": DISSECT / "runs/r122_C1_s0/train.json",
        "curve": DISSECT / "runs/r122_C1_s0/curve.csv",
        "checkpoint": DISSECT / "runs/r122_C1_s0/checkpoint.pt",
        "valid_predictions": DISSECT / "runs/r122_C1_s0/valid_predictions.npz",
        "test_predictions": DISSECT / "runs/r122_C1_s0/test_predictions.npz",
        "test_scores": DISSECT / "runs/r122_C1_s0/test_eval.json",
        "expected": {
            "checkpoint_sha256": "414e6302679768715b7f9846eedc5c606513251f2eb45b77ab41b4e949803bbb",
            "parameters": 228674,
            "best_epoch": 132,
            "split_sizes": {"train": 3360, "valid": 420, "test": 420},
            "seed": 0,
        },
    },
}

#: frozen decision references (route screening only, never significance thresholds)
REFERENCE_DELTA = {"hiv": 0.005, "lipo": 0.03}

# ---------------------------------------------------------------------------
# analysis freeze
# ---------------------------------------------------------------------------

BOOTSTRAP_SEED = 20261126
BOOTSTRAP_REPLICATES = 2000
SCAFFOLD_SOURCE = "RDKit MurckoScaffold.MurckoScaffoldSmiles(includeChirality=False)"

#: frozen candidate rule for the interaction contrasts (section 10)
CONTRAST_MAX_CANDIDATES = 5
CONTRAST_MIN_PAIR_DOCFREQ = 20  # each of the four pairs must occur in >= 20 distinct Train molecules

TOL = {
    "param_bitwise": 0.0,
    "joint_replay_abs": 1e-5,
    "forward_abs": 1e-5,
    "invariance_abs": 1e-5,
    "orthogonality_abs": 1e-5,
    "interaction_abs": 1e-8,
    "metric_recheck": 1e-5,
}

# ---------------------------------------------------------------------------
# budget (this round)
# ---------------------------------------------------------------------------

GPU_SOFT_CAP_HOURS = 2.0
GPU_HARD_CAP_HOURS = 3.0
WALL_CLOCK_HARD_CAP_HOURS = 2.0
PREFERRED_GPU = 1  # res GPU1 (GPU0 is occupied by another round's job)
EXECUTION_REGIME = "res / a100-2 / process / res-gpu1 (torch 2.5.1+cu124)"


# ---------------------------------------------------------------------------
# hashing / json / logging
# ---------------------------------------------------------------------------

sha256_file = PC.sha256_file
sha256_bytes = PC.sha256_bytes
sha256_text = PC.sha256_text
stable_json = PC.stable_json
write_json = PC.write_json
read_json = PC.read_json
write_csv = PC.write_csv
track_rel = PC.track_rel
index_sha256 = PC.index_sha256
utcnow = PC.utcnow
env_info = PC.env_info
git_commit = PC.git_commit
git_dirty = PC.git_dirty


def log(msg: str) -> None:
    print(f"[{ROUND_ID}] {PC.utcnow()} {msg}", flush=True)


def code_hashes() -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted(CODE_DIR.glob("*.py")):
        out[f"code/{p.name}"] = sha256_file(p)
    for p in sorted(TESTS.glob("*.py")):
        out[f"tests/{p.name}"] = sha256_file(p)
    for p in (sorted(TRACK.glob("*.md")) + sorted(CONFIGS.glob("*.json"))):
        out[p.name] = sha256_file(p)
    for name, path in sorted(REUSED_MODULES.items()):
        out[f"reused/{name}"] = sha256_file(path)
    return out


def run_name(task_key: str, arm: str, seed: int = SEED) -> str:
    return f"{task_key}_{arm}_s{int(seed)}"


def run_dir(task_key: str, arm: str, seed: int = SEED) -> Path:
    return RUNS / run_name(task_key, arm, seed)


def num(value: Any) -> Any:
    """JSON-safe scalar."""
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return None if not np.isfinite(f) else f
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def config_payload() -> dict[str, Any]:
    """The frozen experiment configuration (written to ``configs/experiment.json``)."""
    return {
        "schema": "lcrc-support-joint-function-config-v1",
        "round_id": ROUND_ID,
        "frozen_utc": "2026-11-25T00:00:00Z",
        "execution_regime": EXECUTION_REGIME,
        "seed": SEED,
        "task_order": list(TASK_ORDER),
        "tasks": {
            k: {
                "dataset": v["dataset"],
                "variant": v["variant"],
                "task": v["task"],
                "epochs": v["epochs"],
                "primary": v["primary"],
                "secondary": v["secondary"],
                "direction": v["direction"],
                "split_sizes": v["split_sizes"],
                "test_role": v["test_role"],
                "seed_timing": SEED_TIMING[k],
                **({"repeat": v["repeat"]} if "repeat" in v else {}),
            }
            for k, v in TASKS.items()
        },
        "recipe": {k: RECIPE[k] for k in sorted(RECIPE)},
        "arms": {a: ARM_LABEL[a] for a in ARMS},
        "trained_arms": list(TRAINED_ARMS),
        "support": SUPPORT,
        "reuse": {
            k: {
                "round": v["round"],
                "run_dir": track_rel(v["run_dir"]),
                "checkpoint_sha256": sha256_file(v["checkpoint"]),
                "expected": v["expected"],
            }
            for k, v in REUSE.items()
        },
        "analysis": {
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "scaffold_source": SCAFFOLD_SOURCE,
            "contrast_max_candidates": CONTRAST_MAX_CANDIDATES,
            "contrast_min_pair_docfreq": CONTRAST_MIN_PAIR_DOCFREQ,
            "tolerances": TOL,
        },
        "reference_delta": REFERENCE_DELTA,
        "budget": {
            "gpu_soft_cap_hours": GPU_SOFT_CAP_HOURS,
            "gpu_hard_cap_hours": GPU_HARD_CAP_HOURS,
            "wall_clock_hard_cap_hours": WALL_CLOCK_HARD_CAP_HOURS,
            "preferred_gpu": PREFERRED_GPU,
        },
    }


__all__ = [n for n in dir() if not n.startswith("_")]
