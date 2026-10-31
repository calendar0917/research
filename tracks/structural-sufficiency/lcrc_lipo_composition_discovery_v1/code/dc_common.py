"""Shared constants and provenance helpers for LCRC composition discovery v1.

This round is exploratory architecture discovery on the frozen
Lipophilicity random42 split (Train3360 / Validation420 / Test420).
Validation420 is the screening metric; Test420 is only reported for the
candidates selected before it is read and is always marked
``EXPOSED_TEST_EXPLORATORY``.

Nothing in a historical track is written.  Historical code/checkpoints are
imported read-only; all new artifacts live under this experiment directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
EXP_DIR = CODE_DIR.parent
REPO = EXP_DIR.parents[2]

# Historical read-only code roots.
V1_PREP = REPO / "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1"
V1_PREP_CODE = V1_PREP / "code"
V1_CONTEXT = REPO / "tracks/structural-sufficiency/lcrc_lipo_contextual_contrib_v1"
V1_CONTEXT_CODE = V1_CONTEXT / "code"
V2_MECH = REPO / "tracks/structural-sufficiency/lcrc_lipo_context_mechanism_check_v2"
V2_MECH_CODE = V2_MECH / "code"
V2_SSL = REPO / "tracks/structural-sufficiency/lcrc_lipo_mechanism_generalization_ssl_v2"
V2_SSL_CODE = V2_SSL / "code"
ESA = REPO / "tracks/structural-sufficiency/lcrc_lipo_esa_random42_v1"
ESA_CODE = ESA / "code"

for _path in (str(V1_PREP_CODE), str(V1_CONTEXT_CODE), str(V2_MECH_CODE), str(V2_SSL_CODE), str(ESA_CODE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import x1_common as X1C  # noqa: E402

# ---------------------------------------------------------------------------
# paths / round identity
# ---------------------------------------------------------------------------

CONFIG_PATH = EXP_DIR / "configs/discovery.yaml"
PROTOCOL_PATH = EXP_DIR / "PROTOCOL.md"
ANALYSIS = EXP_DIR / "analysis"
RUNS = ANALYSIS / "runs"
PREDICTIONS = ANALYSIS / "predictions"
SMOKE_CPU = ANALYSIS / "smoke_cpu"
SMOKE_CUDA = ANALYSIS / "smoke_cuda"
RR_STATUS = ANALYSIS / "rr_status"
EVAL_DIR = ANALYSIS / "eval"

DATA_AUDIT = ANALYSIS / "data_audit.json"
CPU_GATES = ANALYSIS / "cpu_gates.json"
PROTOCOL_FREEZE = ANALYSIS / "protocol_freeze.json"
PROTOCOL_SHA256 = ANALYSIS / "protocol_sha256.json"
GPU_LEDGER = ANALYSIS / "gpu_ledger.json"
TABLE_SCREEN = ANALYSIS / "table_screen.csv"
TABLE_MECHANISM = ANALYSIS / "table_mechanism.csv"
TABLE_SEED = ANALYSIS / "table_seed.csv"
TABLE_SCAFFOLD = ANALYSIS / "table_scaffold.csv"
DECISION_JSON = ANALYSIS / "decision.json"

ROUND_ID = "lcrc-lipo-composition-discovery-v1"
PROTOCOL_ID = ROUND_ID

# ---------------------------------------------------------------------------
# frozen data constants (identical to the context rounds)
# ---------------------------------------------------------------------------

CSV_PATH = REPO / "data/lipophilicity/Lipophilicity.csv"
CSV_SHA256 = "aed41590cb30609d51d8e08ad3ff06495a76e80e211358801f596b10da69bacd"
N_MOLECULES = 4200
SPLIT_SEED = 42
SPLIT_SIZES = {"train": 3360, "valid": 420, "test": 420}

V1_SPLIT_FILE = ESA / "data/split_esa_random42_v1.json"
V1_SPLIT_SHA256 = "1834414a6da73e9d2c5942ad157a345f08653110acfc4b9fef83c3064bc233b4"
V1_LABELS_FILE = ESA / "data/labels_train_valid_v1.json"
V1_LABELS_SHA256 = "1da269356e0e89974a7388d1e430d7e810a0ab1056be5cd747a1c8ed3d2009ca"

SPLIT_SHA256 = {
    "train": "0df18c72753974d502f8bede1cd2aa5c0ebdc1df450b8871fffdb1f67d2648ee",
    "valid": "8f9c35171ac2125e678d2ad44d87b293719bb9806a528810ac271e9aa39f9b44",
    "test": "4bc449106a64833973b8e8d3d16097bf5d2906c788d7dfcdf02a4b3a30ffa49b",
}
SPLIT_FIRST_ROWS = {
    "train": [3942, 3415, 1426, 2365, 3462],
    "valid": [501, 1649, 3579, 2873, 106],
    "test": [967, 3557, 3492, 462, 2258],
}

TARGET_MEAN = 2.2081011904761905
TARGET_SCALE = 1.1906281212426577
TARGET_VECTOR_SHA256 = "85a4c7c559f4b7fd58ad78b1b05d5a4160e0f60c351b7fa842eb0354aa47abd8"
VALID_VECTOR_SHA256 = "09bc1a1e1756bbd09dd17a5fc77a1b3c33e7010aef55fc352a4bba1d9822636f"
TEST_VECTOR_SHA256 = "a0724b4c1ca6f7bb6b69b720ac2d020825685d9666544c7312e0f24d7e3a4021"

LCRC_CACHE_CONTENT_FINGERPRINT = "23ec00be4fb7e3cf93967cd62911bd932159b1d7e0b43226b23cf7c6a3718e03"
LCRC_CACHE_SHA256_REFERENCE = "450e749045216f32a7a388f0fc039adf6bc5729ef3e9319379ba78f0d9662bd8"
LCRC_TRAIN_NORM_FINGERPRINT = "4d6e5bf456ff9d2217eb11a2c4a29d3f7cbbcb95665bf56e62b93c3ae93d0e26"

#: Historical descriptive references on the same frozen random42 split.
PNA_NATIVE_TEST_RMSE = 0.5288742856070585
PNA_NATIVE_TEST_MAE = 0.37377129447390695
HISTORICAL_LCRC_TEST_RMSE = 0.5839847887469122

#: Frozen checkpoints used for the matched baselines (read-only).
FROZEN_CHECKPOINTS = {
    "base_s0": {
        "path": V1_CONTEXT / "analysis/runs/base_s0/checkpoint.pt",
        "sha256": "1a2820e376e6a426cc470b80bdf3f28a9f6bb9044d0028792463aeddabadc60d",
        "family": "base",
        "seed": 0,
        "train_json": V1_CONTEXT / "analysis/runs/base_s0/train.json",
        "valid_rmse_reference": 0.5804162585770505,
    },
    "roleblind_s0": {
        "path": V2_MECH / "analysis/runs/roleblind_s0/checkpoint.pt",
        "sha256": "7f7bc2e7e40fb25f97f1982db37453ad50c952029ec5249424d9385268b79942",
        "family": "roleblind",
        "seed": 0,
        "train_json": V2_MECH / "analysis/runs/roleblind_s0/train.json",
        "valid_rmse_reference": 0.534232,
    },
    "base_s1": {
        "path": V2_MECH / "analysis/runs/base_s1/checkpoint.pt",
        "sha256": "6cf238117189e1edf0a7be8a7e72b9749f3b32ec800efa1ff89920c4bd354092",
        "family": "base",
        "seed": 1,
        "train_json": V2_MECH / "analysis/runs/base_s1/train.json",
        "valid_rmse_reference": 0.576780,
    },
}

# ---------------------------------------------------------------------------
# frozen recipe / model constants
# ---------------------------------------------------------------------------

WIDTH = 64
CORR_DIM = 32
ATOM_OUT = 96
HETERO_ELEMENTS = ("N", "O", "S", "P")
CARBON_HALOGEN_ELEMENTS = ("C", "F", "Cl", "Br", "I")
W_OUT_INIT_STD = 1.0e-3

LCRC_BASE_PARAMS_REFERENCE = 247_106
CONTEXT_NEW_PARAMS_REFERENCE = (WIDTH * CORR_DIM + CORR_DIM) * 3 + (CORR_DIM * WIDTH + WIDTH)
CONTEXT_TOTAL_PARAMS_REFERENCE = LCRC_BASE_PARAMS_REFERENCE + CONTEXT_NEW_PARAMS_REFERENCE

TRAIN_RECIPE = {
    "batch_size": 128,
    "epochs": 360,
    "optimizer": "AdamW",
    "adam_betas": [0.9, 0.999],
    "adam_eps": 1.0e-8,
    "peak_lr": 0.0003,
    "min_lr": 0.00001,
    "scheduler": "per_step_cosine",
    "weight_decay": 0.001,
    "dropout": 0.10,
    "dropout_position": "final_atom_level_local_h_before_readout",
    "gradient_clip": 5.0,
    "loss": "mse_on_train3360_standardised_target",
    "checkpoint": "average_last_5_epoch_end_state_dicts",
    "early_stopping": False,
    "ssl": False,
}
SOUP_LAST = 5
SHUFFLE_OFFSET = 917_000

#: Families implemented in this round.  ``base`` / ``context`` / ``roleblind``
#: are imported unchanged from the frozen historical code.
NEW_FAMILIES = (
    "g2l_all",
    "g2l_loo",
    "g2l_local_cap",
    "g2l_scalar",
    "dist_true",
    "dist_blind",
    "path_bind",
    "path_unbound",
)
FAMILIES = ("base", "context", "roleblind", *NEW_FAMILIES)

#: Stage-I screening arms.  Seed-1 variants are declared up front so the
#: protocol freeze is stable, but they are only run for the two candidates
#: selected by the pre-registered Stage-II rule.
ARM_SPECS: dict[str, dict[str, Any]] = {}
for _family in NEW_FAMILIES:
    _pool = "res-gpu0" if _family.startswith(("dist_", "path_")) else "res-gpu1"
    ARM_SPECS[f"{_family}_s0"] = {"family": _family, "seed": 0, "pool": _pool, "stage": 1}
    ARM_SPECS[f"{_family}_s1"] = {"family": _family, "seed": 1, "pool": _pool, "stage": 2}
ARM_SPECS["roleblind_s1"] = {"family": "roleblind", "seed": 1, "pool": "res-gpu1", "stage": 2}
STAGE1_ARMS = tuple(f"{family}_s0" for family in NEW_FAMILIES)

ARM_MODEL_NAME = {
    "base": "LCRC-A-R3-scratch-BASE",
    "context": "LCRC-A-R3-contextual-contribution",
    "roleblind": "LCRC-A-R3-contextual-roleblind-allatom",
    "g2l_all": "LCRC-A-G2L-ALL",
    "g2l_loo": "LCRC-A-G2L-LOO",
    "g2l_local_cap": "LCRC-A-G2L-LOCAL-CAP",
    "g2l_scalar": "LCRC-A-G2L-SCALAR",
    "dist_true": "LCRC-A-DIST-TRUE",
    "dist_blind": "LCRC-A-DIST-BLIND",
    "path_bind": "LCRC-A-PATH-BIND",
    "path_unbound": "LCRC-A-PATH-UNBOUND",
}

#: Fixed mechanism pairs.  ``reference`` is the matched negative/control arm.
MECHANISM_PAIRS = {
    "path_bind_vs_path_unbound": ("path_bind", "path_unbound", "bind", "unbound"),
    "dist_true_vs_dist_blind": ("dist_true", "dist_blind", "true", "blind"),
    "roleblind_vs_g2l_local_cap": ("roleblind", "g2l_local_cap", "roleblind", "local_cap"),
}

BOOTSTRAP = {
    "n_resamples": 10000,
    "seed": 20261101,
    "ci_level": 0.95,
    "method": "percentile_paired_molecule_resampling",
    "unit": "molecule",
}
TAIL_ROWS = (1630, 2924, 822, 1153, 3634)
NON_TAIL_COUNT = 415

#: Stage-II resource gate (exploratory, not a significance test).
STAGE2_RMSE_GAIN_THRESHOLD = 0.008
STAGE2_MECHANISM_GAIN_THRESHOLD = 0.008

GPU_HARD_CAP_HOURS = 14.0

# ---------------------------------------------------------------------------
# generic helpers
# ---------------------------------------------------------------------------

sha256_file = X1C.sha256_file
sha256_bytes = X1C.sha256_bytes
sha256_json = X1C.sha256_json
write_json = X1C.write_json
read_json = X1C.read_json
track_rel = X1C.track_rel
utcnow = X1C.utcnow
state_dict_fingerprint = X1C.state_dict_fingerprint
rows_sha256_int64 = X1C.rows_sha256_int64
vector_sha256_float64 = X1C.vector_sha256_float64
log = X1C.log


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "UNKNOWN"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"], text=True
        )
        return bool(out.strip())
    except Exception:
        return True


def env_info() -> dict[str, Any]:
    import torch

    return {
        "round_id": ROUND_ID,
        "timestamp": utcnow(),
        "git_commit": git_commit(),
        "git_dirty_tracked": git_dirty(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "numpy": np.__version__,
        "pid": os.getpid(),
    }


def code_file_hashes() -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted(CODE_DIR.glob("*.py")):
        out[str(path.relative_to(EXP_DIR))] = sha256_file(path)
    return out


def config_file_sha256() -> str:
    return sha256_file(CONFIG_PATH)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def _need(actual, expected, name) -> None:
    if actual != expected:
        raise ValueError(f"config drift at {name}: {actual!r} != frozen {expected!r}")


def validate_config(cfg: Mapping[str, Any]) -> None:
    _need(cfg.get("protocol_id"), PROTOCOL_ID, "protocol_id")
    _need(cfg.get("round_id"), ROUND_ID, "round_id")
    _need(int(cfg["split_seed"]), SPLIT_SEED, "split_seed")
    d = cfg["data"]
    _need(d["split_file_sha256"], V1_SPLIT_SHA256, "data.split_file_sha256")
    _need(d["labels_file_sha256"], V1_LABELS_SHA256, "data.labels_file_sha256")
    _need(d["test_rows_sha256"], SPLIT_SHA256["test"], "data.test_rows_sha256")
    _need(d["test_label_vector_sha256"], TEST_VECTOR_SHA256, "data.test_label_vector_sha256")
    _need({k: int(v) for k, v in d["split_sizes"].items()}, SPLIT_SIZES, "data.split_sizes")
    _need(d["test_access"], "EXPOSED_TEST_EXPLORATORY", "data.test_access")
    t = cfg["training"]
    for key in (
        "batch_size",
        "epochs",
        "optimizer",
        "peak_lr",
        "min_lr",
        "scheduler",
        "weight_decay",
        "dropout",
        "gradient_clip",
        "loss",
        "checkpoint",
        "early_stopping",
        "ssl",
    ):
        _need(t[key], TRAIN_RECIPE[key], f"training.{key}")
    _need([float(x) for x in t["adam_betas"]], [0.9, 0.999], "training.adam_betas")
    _need(float(t["adam_eps"]), 1.0e-8, "training.adam_eps")
    _need(int(t["soup_last"]), SOUP_LAST, "training.soup_last")
    _need(int(t["shuffle_offset"]), SHUFFLE_OFFSET, "training.shuffle_offset")
    _need(int(cfg["model"]["width"]), WIDTH, "model.width")
    _need(int(cfg["model"]["correction_dim"]), CORR_DIM, "model.correction_dim")
    _need(tuple(cfg["model"]["hetero_elements"]), HETERO_ELEMENTS, "model.hetero_elements")
    _need(float(cfg["budget"]["gpu_hard_cap_hours"]), GPU_HARD_CAP_HOURS, "budget.gpu_hard_cap_hours")


def load_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    import yaml

    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    validate_config(cfg)
    return cfg


__all__ = [name for name in dir() if not name.startswith("_")]
