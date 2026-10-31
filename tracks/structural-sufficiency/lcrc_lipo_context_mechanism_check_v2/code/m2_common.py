"""Shared constants, guards and helpers for the LCRC context mechanism check v2.

Round: ``lcrc-lipo-context-mechanism-check-v2``.

Scope
-----
* **A** Evaluate the already-frozen seed-0 BASE / CONTEXT checkpoints from
  ``lcrc_lipo_contextual_contrib_v1`` on the frozen random42 Test420 split.
  This is explicitly ``EXPOSED_TEST_EXPLORATORY``: Test was used by earlier
  rounds on this split and this round's numbers will feed the next research
  decision.  Test labels are read from the hash-verified raw CSV.
* **B** Train ``BASE_s1`` and ``CONTEXT_s1`` (training seed 1, split seed 42)
  with the byte-identical recipe and shared public initialisation.
* **C** Train ``ROLEBLIND_s0``: parameter-structure- and initialisation-identical
  to CONTEXT, but ``m_het = m_ch = mean(h_i over all atoms of the molecule)``.

Nothing in a historical directory is written.  Historical checkpoints, the
frozen split / labels / LCRC cache and the historical model definitions are read
only.  Every new artifact lives under this experiment directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
EXP_DIR = CODE_DIR.parent
REPO = EXP_DIR.parents[2]

V1 = REPO / "tracks/structural-sufficiency/lcrc_lipo_contextual_contrib_v1"
V1_CODE = V1 / "code"
V1_ANALYSIS = V1 / "analysis"
V1_RUNS = V1_ANALYSIS / "runs"
V1_DATA = REPO / "tracks/structural-sufficiency/lcrc_lipo_esa_random42_v1/data"
V1_ESA = REPO / "tracks/structural-sufficiency/lcrc_lipo_esa_random42_v1"
V1_ESA_CACHE = V1_ESA / "cache" / "lcrc_all4200.pt"
V1_CONTEXT_CACHE = V1 / "cache" / "lcrc_all4200.pt"

for _path in (str(V1_CODE),):
    if _path not in __import__("sys").path:
        __import__("sys").path.insert(0, _path)

# Read-only reuse of the frozen v1 helpers.  Importing it also puts the
# historical code directories on ``sys.path`` (v1, preparation, v2), which is
# how the exact same model definitions and data helpers are reused.
import x1_common as X1C  # noqa: E402

CONFIG_PATH = EXP_DIR / "configs/context_mechanism_check_v2.yaml"
PROTOCOL_PATH = EXP_DIR / "PROTOCOL.md"
ANALYSIS = EXP_DIR / "analysis"
RUNS = ANALYSIS / "runs"
RR_STATUS = ANALYSIS / "rr_status"
SMOKE_CPU = ANALYSIS / "smoke_cpu"
SMOKE_CUDA = ANALYSIS / "smoke_cuda"
EVAL_DIR = ANALYSIS / "eval"
EVAL_FROZEN_S0 = ANALYSIS / "eval_frozen_s0"
PREDICTIONS = ANALYSIS / "predictions"

DATA_AUDIT = ANALYSIS / "data_audit.json"
CPU_GATES = ANALYSIS / "cpu_gates.json"
PROTOCOL_FREEZE = ANALYSIS / "protocol_freeze.json"
PROTOCOL_SHA256 = ANALYSIS / "protocol_sha256.json"
GPU_LEDGER = ANALYSIS / "gpu_ledger.json"
COMPARISON = ANALYSIS / "comparison_seed01.json"
ROLEBLIND_COMPARISON = ANALYSIS / "roleblind_comparison.json"
TAIL_AUDIT = ANALYSIS / "tail_audit.json"
DECISION = ANALYSIS / "decision.json"
TRAINING_CURVES = ANALYSIS / "training_curves_summary.json"

ROUND_ID = "lcrc-lipo-context-mechanism-check-v2"
PROTOCOL_ID = ROUND_ID

# ---------------------------------------------------------------------------
# frozen random42 data constants (identical to every earlier round)
# ---------------------------------------------------------------------------

CSV_PATH = REPO / "data/lipophilicity/Lipophilicity.csv"
CSV_SHA256 = "aed41590cb30609d51d8e08ad3ff06495a76e80e211358801f596b10da69bacd"
N_MOLECULES = 4200
SPLIT_SEED = 42
SPLIT_SIZES = {"train": 3360, "valid": 420, "test": 420}

V1_SPLIT_FILE = V1_DATA / "split_esa_random42_v1.json"
V1_SPLIT_SHA256 = "1834414a6da73e9d2c5942ad157a345f08653110acfc4b9fef83c3064bc233b4"
V1_LABELS_FILE = V1_DATA / "labels_train_valid_v1.json"
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
#: One-shot Test label vector hash written by
#: ``lcrc_lipo_esa_random42_v1/analysis/test_read.lock``.  Recomputing the same
#: vector from the raw CSV is the tie-back to the historical scoring.
TEST_VECTOR_SHA256 = "a0724b4c1ca6f7bb6b69b720ac2d020825685d9666544c7312e0f24d7e3a4021"

LCRC_CACHE_CONTENT_FINGERPRINT = "23ec00be4fb7e3cf93967cd62911bd932159b1d7e0b43226b23cf7c6a3718e03"
LCRC_CACHE_SHA256_REFERENCE = "450e749045216f32a7a388f0fc039adf6bc5729ef3e9319379ba78f0d9662bd8"
LCRC_TRAIN_NORM_FINGERPRINT = "4d6e5bf456ff9d2217eb11a2c4a29d3f7cbbcb95665bf56e62b93c3ae93d0e26"

#: Historical Test numbers on the same frozen random42 split.  Descriptive
#: references only; no PNA model is retrained in this round.
PNA_NATIVE_TEST_RMSE = 0.5288742856070585
PNA_NATIVE_TEST_MAE = 0.37377129447390695
HISTORICAL_LCRC_TEST_RMSE = 0.5839847887469122

#: Historical seed-0 checkpoints evaluated in part A (read only).
HISTORICAL_CHECKPOINTS = {
    "base_s0": {
        "path": V1_RUNS / "base_s0/checkpoint.pt",
        "sha256": "1a2820e376e6a426cc470b80bdf3f28a9f6bb9044d0028792463aeddabadc60d",
        "family": "base",
        "seed": 0,
        "train_json": V1_RUNS / "base_s0/train.json",
        "valid_rmse_reference": 0.5804162585770505,
    },
    "context_s0": {
        "path": V1_RUNS / "context_s0/checkpoint.pt",
        "sha256": "38bf6926c32db02f4c10f72da5c84abf2194c8b954681e0334aca88c91eccaee",
        "family": "context",
        "seed": 0,
        "train_json": V1_RUNS / "context_s0/train.json",
        "valid_rmse_reference": 0.568587464048014,
    },
}

# ---------------------------------------------------------------------------
# frozen model / training constants
# ---------------------------------------------------------------------------

WIDTH = 64
CONTEXT_DIM = 32
HETERO_ELEMENTS = ("N", "O", "S", "P")
CARBON_HALOGEN_ELEMENTS = ("C", "F", "Cl", "Br", "I")
W_OUT_INIT_STD = 1.0e-3

LCRC_BASE_PARAMS_REFERENCE = 247_106
CONTEXT_NEW_PARAMS_REFERENCE = (WIDTH * CONTEXT_DIM + CONTEXT_DIM) * 3 + (CONTEXT_DIM * WIDTH + WIDTH)
CONTEXT_TOTAL_PARAMS_REFERENCE = LCRC_BASE_PARAMS_REFERENCE + CONTEXT_NEW_PARAMS_REFERENCE

TRAIN_RECIPE = {
    "model_family": "LCRC-A-R3 (lipo_v1, width 64)",
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

#: Formal arms of this round.  ``family`` selects the model definition, ``seed``
#: is the training seed (split seed is always 42).
ARM_SPECS: dict[str, dict[str, Any]] = {
    "base_s1": {"family": "base", "seed": 1, "pool": "res-gpu0", "run_key": "base_s1"},
    "context_s1": {"family": "context", "seed": 1, "pool": "res-gpu1", "run_key": "context_s1"},
    "roleblind_s0": {"family": "roleblind", "seed": 0, "pool": "res-gpu1", "run_key": "roleblind_s0"},
}
FAMILIES = ("base", "context", "roleblind")

ARM_MODEL_NAME = {
    "base": "LCRC-A-R3-scratch-BASE",
    "context": "LCRC-A-R3-contextual-contribution",
    "roleblind": "LCRC-A-R3-contextual-roleblind-allatom",
}

# ---------------------------------------------------------------------------
# evaluation constants (declared before any Test number is computed)
# ---------------------------------------------------------------------------

BOOTSTRAP = {
    "n_resamples": 10000,
    "seed": 20261101,
    "ci_level": 0.95,
    "method": "percentile_paired_molecule_resampling",
    "unit": "molecule",
}
TAIL_ROWS = (1630, 2924, 822, 1153, 3634)
NON_TAIL_COUNT = 415

PERFORMANCE_RULE = (
    "PERFORMANCE_SIGNAL=PRESENT iff all four BASE->CONTEXT RMSE gains "
    "(BASE RMSE - CONTEXT RMSE; positive = CONTEXT better) are > 0, i.e. "
    "seed 0 and seed 1 on both Validation420 and Test420.  Otherwise ABSENT. "
    "A sign disagreement between Validation and Test inside any seed is "
    "reported explicitly as a reversal."
)
ROLE_SPECIFICITY_RULE = (
    "ROLE_SPECIFICITY_SIGNAL=PRESENT iff CONTEXT_s0 beats ROLEBLIND_s0 on both "
    "Validation420 and Test420 by RMSE (ROLEBLIND RMSE - CONTEXT RMSE; positive "
    "= CONTEXT better) AND both paired-bootstrap 95% percentile intervals of "
    "that RMSE gain exclude 0.  Otherwise ABSENT.  Descriptive, not a "
    "confirmatory hypothesis test."
)

GPU_HARD_CAP_HOURS = 0.45

# ---------------------------------------------------------------------------
# generic helpers
# ---------------------------------------------------------------------------

sha256_file = X1C.sha256_file
sha256_bytes = X1C.sha256_bytes
sha256_json = X1C.sha256_json
stable_json = X1C.stable_json
write_json = X1C.write_json
read_json = X1C.read_json
track_rel = X1C.track_rel
utcnow = X1C.utcnow
config_hash = X1C.config_hash
state_dict_fingerprint = X1C.state_dict_fingerprint
rows_sha256_int64 = X1C.rows_sha256_int64
vector_sha256_float64 = X1C.vector_sha256_float64


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


def log(message: str) -> None:
    print(message, flush=True)


def verify_sha256(path: Path | str, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise PermissionError(f"{label}: sha256 {actual} != frozen {expected}")
    return actual


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
    _need(d["source_round"], "lcrc_lipo_esa_random42_v1", "data.source_round")
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
    m = cfg["model"]
    _need(int(m["width"]), WIDTH, "model.width")
    _need(int(m["context_dim"]), CONTEXT_DIM, "model.context_dim")
    _need(tuple(m["hetero_elements"]), HETERO_ELEMENTS, "model.hetero_elements")
    _need(tuple(m["carbon_halogen_elements"]), CARBON_HALOGEN_ELEMENTS, "model.carbon_halogen_elements")
    _need(float(m["w_out_init"]["std"]), W_OUT_INIT_STD, "model.w_out_init.std")
    _need(float(m["w_out_init"]["bias"]), 0.0, "model.w_out_init.bias")
    _need(m["update_scope"], "hetero_atoms_only", "model.update_scope")
    _need(m["roleblind_definition"], "m_het = m_ch = mean over all atoms", "model.roleblind_definition")
    arms = cfg["arms"]
    _need(set(arms), set(ARM_SPECS), "arms")
    for key, spec in ARM_SPECS.items():
        _need(arms[key]["family"], spec["family"], f"arms.{key}.family")
        _need(int(arms[key]["seed"]), spec["seed"], f"arms.{key}.seed")
        _need(arms[key]["pool"], spec["pool"], f"arms.{key}.pool")
    ev = cfg["evaluation"]
    _need(int(ev["bootstrap"]["n_resamples"]), BOOTSTRAP["n_resamples"], "evaluation.bootstrap.n_resamples")
    _need(int(ev["bootstrap"]["seed"]), BOOTSTRAP["seed"], "evaluation.bootstrap.seed")
    _need(float(ev["bootstrap"]["ci_level"]), BOOTSTRAP["ci_level"], "evaluation.bootstrap.ci_level")
    _need(tuple(int(x) for x in ev["tail_rows"]), TAIL_ROWS, "evaluation.tail_rows")
    _need(float(cfg["budget"]["gpu_hard_cap_hours"]), GPU_HARD_CAP_HOURS, "budget.gpu_hard_cap_hours")


def load_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    import yaml

    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    validate_config(cfg)
    return cfg


__all__ = [name for name in dir() if not name.startswith("_")]
