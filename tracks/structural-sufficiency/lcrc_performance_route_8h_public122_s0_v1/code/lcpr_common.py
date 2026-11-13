"""Frozen constants, paths, dataset registry, candidate registry and IO.

Round: ``lcrc-performance-route-8h-public122-s0-v1``.

Nothing in this module is chosen after seeing a score.  Every dataset, split,
recipe, candidate definition, selection rule and seed is written here before the
first formal run, mirrored in ``configs/experiment.yaml`` and ``PROTOCOL.md``.

Read-only upstream modules (hashed into every run record):

* ``lcrc_lipo_model_dissection_v1``  - FULL -> C1 state remap, C1 graph, B5
  marginal operators, public122 loader.
* ``lcrc_molnet_benchmark_pilot_v2`` - label-free structure caches for all
  seven datasets (same ``lipo_v1`` FeatureSpec), published sources and targets.
* ``lcrc_performance_longrun_v1``    - historical chem8 atom channels and the
  frozen eight-column artefact.
* ``lcrc_lipo_oracle_task_relevance_public122_s0_v1`` - WC_LOGP / MolWt oracle
  definition (used only as an auxiliary *training target*).
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROUND_ID = "lcrc-performance-route-8h-public122-s0-v1"
PROTOCOL_ID = ROUND_ID

CODE_DIR = Path(__file__).resolve().parent
TRACK = CODE_DIR.parent
REPO = TRACK.parents[2]

DISSECT = REPO / "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1"
DISSECT_CODE = DISSECT / "code"
MOLNET = REPO / "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v2"
MOLNET_CODE = MOLNET / "code"
LONGRUN = REPO / "tracks/structural-sufficiency/lcrc_performance_longrun_v1"
ORACLE = REPO / "tracks/structural-sufficiency/lcrc_lipo_oracle_task_relevance_public122_s0_v1"

for _p in (str(CODE_DIR), str(DISSECT_CODE), str(MOLNET_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dissect_common as DC  # noqa: E402
import dissect_data as DD  # noqa: E402
import dissect_model as DM  # noqa: E402
import dissect_source as DS  # noqa: E402

ANALYSIS = TRACK / "analysis"
CONFIGS = TRACK / "configs"
RUNS = TRACK / "runs"
CACHE = TRACK / "cache"
TESTS = TRACK / "tests"

#: read-only upstream label-free structure caches (dataset -> {split: path})
UPSTREAM_CACHE = MOLNET / "cache"

# ---------------------------------------------------------------------------
# frozen data declarations
# ---------------------------------------------------------------------------

#: Lipophilicity: the published ChemBench random122 split (4200 = 3360/420/420).
LIPO = {
    "dataset": "Lipophilicity",
    "task": "regression",
    "split_method": "random122",
    "sizes": (3360, 420, 420),
    "n_total": 4200,
    "rows_sha256": tuple(DC.SPLIT_ROWS_SHA256[122]),
    "csv_sha256": DC.LIPOP_CSV_SHA256,
    "split_file_sha256": DC.SPLIT_FILE_SHA256[122],
    "label_column": "exp",
    "units": "logD (experimental octanol/water distribution coefficient)",
}

#: Second/third task registry.  Split method per dataset, fixed before scores.
DATASETS: dict[str, dict[str, Any]] = {
    "Lipophilicity": {
        "task": "regression",
        "split_method": "random122",
        "sizes": (3360, 420, 420),
        "primary": "rmse",
        "units": "logD",
        "role": "primary diagnosis field",
    },
    "ESOL": {
        "task": "regression",
        "split_method": "random122",
        "sizes": (902, 113, 113),
        "primary": "rmse",
        "units": "log mol/L (aqueous solubility)",
        "role": "transfer task 1 (regression)",
    },
    "BACE": {
        "task": "binary",
        "split_method": "scaffold122",
        "sizes": (1210, 151, 152),
        "primary": "roc_auc",
        "units": "binary class (BACE-1 inhibition)",
        "role": "transfer task 2 (classification, scaffold split)",
    },
    "FreeSolv": {
        "task": "regression",
        "split_method": "random122",
        "sizes": (513, 64, 65),
        "primary": "rmse",
        "units": "kcal/mol (hydration free energy)",
        "role": "extension task 3 (regression, small)",
    },
    "ogbg-molhiv": {
        "task": "binary",
        "split_method": "ogb_official_scaffold",
        "sizes": (32901, 4113, 4113),
        "primary": "roc_auc",
        "units": "binary class (HIV replication inhibition)",
        "role": "HIV stress test (official scaffold split)",
    },
}

PRIMARY_DATASET = "Lipophilicity"
TRANSFER_DATASETS = ("ESOL", "BACE")
EXTENSION_DATASET = "FreeSolv"
HIV = "ogbg-molhiv"

SPLITS = ("train", "valid", "test")
SPLIT_SIZES = (3360, 420, 420)
N_MOLECULES = 4200

FEATURE_SPEC_HASH = DC.FEATURE_SPEC_HASH
ATOM_FEAT_DIM = DC.ATOM_FEAT_DIM
BOND_FEAT_DIM = DC.BOND_FEAT_DIM
ATOM_SCALAR_DIM = DC.ATOM_SCALAR_DIM
GLOBAL_DIM = DC.GLOBAL_DIM
N_NBR_CODES = DC.N_NBR_CODES
N_PATH_CODES = DC.N_PATH_CODES
N_BOND_TYPES = DC.N_BOND_TYPES
LOCAL_DIM = DC.LOCAL_DIM
RHO_DIM = DC.RHO_DIM

TRAIN_SEED = 0
SHUFFLE_OFFSET = 917_000
SHUFFLE_SEED = TRAIN_SEED + SHUFFLE_OFFSET
AUX_SHUFFLE_SEED = 90217
BOOTSTRAP_SEED = 20261113
BOOTSTRAP_REPLICATES = 2000

#: C1 (NO_BOND_MEAN) default configuration, frozen.
C1_VARIANT = "C1"
C1_PARAMS = 228674
C1_CFG = {
    "width": 64,
    "stat_rank": 32,
    "local_hidden": 192,
    "atom_hidden": 128,
    "atom_out": 96,
    "bond_hidden": 160,
    "bond_out": 96,
    "rho_hidden": 64,
}
#: A0 (FULL) is only used for the historical bridge; 247106 trainable parameters.
A0_PARAMS = 247106

# ---------------------------------------------------------------------------
# recipes (R0 is the frozen historical recipe; R1..R3 are the finite screen)
# ---------------------------------------------------------------------------

BASE_RECIPE: dict[str, Any] = {
    "optimizer": "AdamW",
    "adam_betas": (0.9, 0.999),
    "adam_eps": 1e-8,
    "weight_decay": 1.0e-3,
    "peak_lr": 3.0e-4,
    "min_lr": 1.0e-5,
    "batch_size": 128,
    "dropout": 0.10,
    "dropout_position": "final atom-level local representation h, before f and rho",
    "gradient_clip": 5.0,
    "epochs": 360,
    "scheduler": "per_step_cosine",
    "amp": False,
    "ema": False,
    "swa": False,
    "soup": False,
    "early_stopping": False,
}

#: The four frozen recipes.  R0 = historical; only lr / weight_decay change.
RECIPES: dict[str, dict[str, Any]] = {
    name: {**BASE_RECIPE, "peak_lr": lr, "weight_decay": wd, "name": name}
    for name, lr, wd in (
        ("R0", 3.0e-4, 1.0e-3),
        ("R1", 1.0e-3, 1.0e-3),
        ("R2", 3.0e-4, 1.0e-2),
        ("R3", 1.0e-3, 1.0e-2),
    )
}
RECIPE_TIE_ORDER = ("R0", "R1", "R2", "R3")
LONG_EPOCHS = 720
HIV_EPOCHS = 160

EVAL_BATCH_SIZE = 4096
REPRO_TOL_PRED = 2e-5
PARITY_TOL = 1e-6

#: ---------------------------------------------------------------------------
#: candidate registry.  ``family`` groups arms for selection; ``kind`` selects
#: the implementation path in ``lcpr_model``.
#: ---------------------------------------------------------------------------
CANDIDATES: dict[str, dict[str, Any]] = {
    # --- A1: recipe screen on plain C1 -----------------------------------
    "C1_R0": {"kind": "c1", "family": "VANILLA", "recipe": "R0", "dataset_role": "base"},
    "C1_R1": {"kind": "c1", "family": "VANILLA", "recipe": "R1", "dataset_role": "base"},
    "C1_R2": {"kind": "c1", "family": "VANILLA", "recipe": "R2", "dataset_role": "base"},
    "C1_R3": {"kind": "c1", "family": "VANILLA", "recipe": "R3", "dataset_role": "base"},
    # --- A2: cheap capacity / optimisation-budget controls ---------------
    "WIDE_LOCAL": {"kind": "wide_local", "family": "WIDE", "recipe": "R*", "dataset_role": "control"},
    "LONG": {"kind": "c1", "family": "LONG", "recipe": "R*", "epochs": LONG_EPOCHS, "dataset_role": "control"},
    # --- A3: historical positive-signal bridge ---------------------------
    "A0_R0": {"kind": "a0", "family": "BRIDGE", "recipe": "R0", "dataset_role": "mother"},
    "A0_CHEM8_CONTRIB_R0": {"kind": "a0_combo", "family": "BRIDGE", "recipe": "R0", "dataset_role": "bridge"},
    "K_COMBO": {"kind": "k_combo", "family": "K", "recipe": "R*", "dataset_role": "candidate"},
    "C1_CHEM8": {"kind": "k_chem8", "family": "K", "recipe": "R*", "dataset_role": "control"},
    "C1_CONTRIB": {"kind": "k_contrib", "family": "K", "recipe": "R*", "dataset_role": "control"},
    # --- B/H_M: dropout + second-order pooling ---------------------------
    "M_UNBIASED": {"kind": "m_unbiased", "family": "M", "recipe": "R*", "dataset_role": "candidate"},
    "M_CLEANPOOL": {"kind": "m_cleanpool", "family": "M", "recipe": "R*", "dataset_role": "candidate"},
    "M_NODROP": {"kind": "m_nodrop", "family": "M", "recipe": "R*", "dataset_role": "control"},
    # --- B/H_S: support-conditional contraction of the J pair block ------
    "S_SELECT_K1": {"kind": "s_select", "family": "S", "recipe": "R*", "kappa_mult": 1.0, "dataset_role": "candidate"},
    "S_UNIFORM_K1": {"kind": "s_uniform", "family": "S", "recipe": "R*", "kappa_mult": 1.0, "dataset_role": "candidate"},
    "S_SELECT_K4": {"kind": "s_select", "family": "S", "recipe": "R*", "kappa_mult": 4.0, "dataset_role": "candidate"},
    "S_UNIFORM_K4": {"kind": "s_uniform", "family": "S", "recipe": "R*", "kappa_mult": 4.0, "dataset_role": "candidate"},
    # --- B/H_A: computational task-relevance auxiliary supervision -------
    "A_LOGP": {"kind": "a_logp", "family": "A", "recipe": "R*", "dataset_role": "candidate"},
    "A_MW": {"kind": "a_mw", "family": "A", "recipe": "R*", "dataset_role": "control"},
    "A_SHUFFLE": {"kind": "a_shuffle", "family": "A", "recipe": "R*", "dataset_role": "control"},
    # --- 7.6 failure-diagnosis branch (only if triggered) ----------------
    "MP_CONTEXT": {"kind": "mp_context", "family": "CTX", "recipe": "R*", "dataset_role": "diagnostic"},
    "POOL_CONTEXT": {"kind": "pool_context", "family": "CTX", "recipe": "R*", "dataset_role": "control"},
    # --- A5: D-MPNN reference (its own fixed recipe) ---------------------
    "DMPNN_LR3E4": {"kind": "dmpnn", "family": "DMPNN", "recipe": "D1", "dataset_role": "reference"},
    "DMPNN_LR1E3": {"kind": "dmpnn", "family": "DMPNN", "recipe": "D2", "dataset_role": "reference"},
}

#: D-MPNN fixed recipes (chemprop v2.2.1 semantics, not our R0..R3).
DMPNN_RECIPES: dict[str, dict[str, Any]] = {
    "D1": {"batch_size": 64, "epochs": 100, "peak_lr": 3.0e-4, "init_lr": 1.0e-4, "min_lr": 1.0e-4,
           "warmup_epochs": 2, "decay": "exponential", "weight_decay": 0.0, "gradient_clip": 5.0,
           "optimizer": "Adam"},
    "D2": {"batch_size": 64, "epochs": 100, "peak_lr": 1.0e-3, "init_lr": 1.0e-4, "min_lr": 1.0e-4,
           "warmup_epochs": 2, "decay": "exponential", "weight_decay": 0.0, "gradient_clip": 5.0,
           "optimizer": "Adam"},
}
DMPNN_FAMILY = {"hidden": 300, "depth": 3, "bias": False, "undirected": False,
                "activation": "relu", "message_dropout": 0.0, "ffn": [300, 300, 1], "ffn_dropout": 0.0}

# ---------------------------------------------------------------------------
# frozen selection rules (written before any score is read)
# ---------------------------------------------------------------------------

#: Lipo "clear signal" purchase rule (round-local priority rule, not a
#: significance or generality threshold).
CLEAR_SIGNAL = {"test_rel_rmse_gain_min": 0.01, "valid_rel_rmse_gain_min": -0.005,
                "mae_degradation_max": 0.01}
#: transfer rule
TRANSFER_RULE = {"esol_rel_rmse_gain_min": 0.01, "bace_auc_gain_min": 0.005,
                 "regression_reverse_max": 0.02, "auc_reverse_max": 0.01}
#: 7.6 failure-branch trigger
CONTEXT_TRIGGER = {"dmpnn_rel_rmse_advantage_min": 0.03}
#: tail sensitivity fractions (fixed, on the frozen Lipo base row set)
TAIL_FRACTIONS = (0.01, 0.05)

#: 7.2/7.7 fixed ordering of previously-unadvanced families for the
#: remaining-budget sweep.
FAMILY_ORDER = ("M", "S", "A", "K")
MAX_TRANSFER_FAMILIES = 2

#: analytic dropout fixture
MOMENT_FIXTURE_SEED = 4401
MOMENT_MC_DRAWS = 256

GPU_SOFT_CAP_HOURS = 14.0
GPU_HARD_CAP_HOURS = 16.0
MAX_NEW_TRAININGS = 64


# ---------------------------------------------------------------------------
# hashing / IO
# ---------------------------------------------------------------------------


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def array_sha256(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(arr), dtype=np.int64)
    return sha256_bytes(a.tobytes())


def row_sha256(rows: Iterable[int]) -> str:
    return array_sha256(np.asarray(list(rows), dtype=np.int64))


def _json_default(obj: Any):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)


def write_json(path: Path | str, obj: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return p


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def track_rel(path: Path | str) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO.resolve()))
    except ValueError:
        return str(p)


def log(msg: str) -> None:
    print(f"[{utcnow()}] {msg}", flush=True)


def run_name(candidate: str, dataset: str) -> str:
    return f"{candidate}__{dataset}"


def run_dir(candidate: str, dataset: str) -> Path:
    return RUNS / run_name(candidate, dataset)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # pragma: no cover
        return ""


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
            text=True, stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except Exception:  # pragma: no cover
        return True


def env_info() -> dict[str, Any]:
    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": np.__version__,
        "hostname": platform.node(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def rdkit_version() -> str:
    from rdkit import rdBase

    return str(rdBase.rdkitVersion)


REUSED_MODULES = (
    "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code/dissect_common.py",
    "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code/dissect_data.py",
    "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code/dissect_model.py",
    "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code/dissect_source.py",
    "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/feature_spec.py",
    "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/molecular_graph.py",
    "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/adapters.py",
    "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/lcrc_data.py",
    "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/lcrc_model.py",
    "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/lcrc_common.py",
    "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v2/code/mb2_common.py",
    "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v2/code/mb2_source.py",
    "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v2/code/mb2_data.py",
    "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v2/code/mb2_metrics.py",
    "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v2/code/mb2_baseline.py",
    "tracks/structural-sufficiency/lcrc_performance_longrun_v1/code/cl_chem8.py",
    "tracks/structural-sufficiency/lcrc_lipo_oracle_task_relevance_public122_s0_v1/code/oracle.py",
)


def code_hashes() -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted(CODE_DIR.rglob("*.py")):
        out[str(p.relative_to(TRACK))] = sha256_file(p)
    for p in sorted(TESTS.glob("*.py")):
        out[f"tests/{p.name}"] = sha256_file(p)
    for p in sorted(CONFIGS.glob("*.yaml")):
        out[f"configs/{p.name}"] = sha256_file(p)
    if (TRACK / "PROTOCOL.md").exists():
        out["PROTOCOL.md"] = sha256_file(TRACK / "PROTOCOL.md")
    for rel in REUSED_MODULES:
        out[rel] = sha256_file(REPO / rel)
    return out


__all__ = [n for n in dir() if not n.startswith("_")]
