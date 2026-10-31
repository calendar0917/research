"""Shared constants, paths and provenance helpers.

Round: ``lcrc-lipo-best-epoch-distance-fix-v1``

Two changes relative to ``lcrc_lipo_composition_discovery_v1``:

1. ``DIST-TRUE`` is re-implemented with the mathematically correct bucket
   aggregation (the historical implementation wrote every bucket sum back to
   the *target* atom instead of the *root* atom, which reduced the context to a
   scaled copy of the atom's own representation).  The fixed model is called
   ``DIST-TRUE-FIXED`` and the historical checkpoint/report are untouched.
2. Every run uses the single best Validation420 epoch state dict as the
   checkpoint.  There is no soup, no SWA, no EMA, no post-hoc averaging and no
   Test-based epoch selection.

The frozen random42 split, the label-free LCRC cache, the Train3360-only
normalisation and the frozen R3 training recipe are imported read-only from the
historical rounds.  Nothing in a historical directory is written.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
EXP_DIR = CODE_DIR.parent
REPO = EXP_DIR.parents[2]

# Historical read-only code roots (the same set the composition round used).
HIST_COMPOSITION = REPO / "tracks/structural-sufficiency/lcrc_lipo_composition_discovery_v1"
HIST_COMPOSITION_CODE = HIST_COMPOSITION / "code"
HIST_CONTEXT = REPO / "tracks/structural-sufficiency/lcrc_lipo_contextual_contrib_v1"
HIST_CONTEXT_CODE = HIST_CONTEXT / "code"
HIST_MECH = REPO / "tracks/structural-sufficiency/lcrc_lipo_context_mechanism_check_v2"
HIST_MECH_CODE = HIST_MECH / "code"
HIST_PREP = REPO / "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1"
HIST_PREP_CODE = HIST_PREP / "code"
HIST_ESA = REPO / "tracks/structural-sufficiency/lcrc_lipo_esa_random42_v1"
HIST_ESA_CODE = HIST_ESA / "code"

for _path in (
    str(CODE_DIR),
    str(HIST_COMPOSITION_CODE),
    str(HIST_PREP_CODE),
    str(HIST_CONTEXT_CODE),
    str(HIST_MECH_CODE),
    str(HIST_ESA_CODE),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Read-only reuse of the frozen constants/helpers of the composition round.
# Importing it also puts the preparation / context / mechanism / ESA code
# directories on ``sys.path``, which is how the frozen model and data
# definitions are reused unchanged.
import dc_common as DC  # noqa: E402


# ---------------------------------------------------------------------------
# round identity / paths
# ---------------------------------------------------------------------------

ROUND_ID = "lcrc-lipo-best-epoch-distance-fix-v1"
PROTOCOL_ID = ROUND_ID

CONFIG_PATH = EXP_DIR / "configs/best_epoch.yaml"
PROTOCOL_PATH = EXP_DIR / "PROTOCOL.md"
ANALYSIS = EXP_DIR / "analysis"
RUNS = ANALYSIS / "runs"
PREDICTIONS = ANALYSIS / "predictions"
SMOKE_CPU = ANALYSIS / "smoke_cpu"
SMOKE_CUDA = ANALYSIS / "smoke_cuda"
RR_STATUS = ANALYSIS / "rr_status"
EVAL_VALID = ANALYSIS / "eval_valid"
EVAL_TEST = ANALYSIS / "eval_test"

DATA_AUDIT = ANALYSIS / "data_audit.json"
CPU_GATES = ANALYSIS / "cpu_gates.json"
PROTOCOL_FREEZE = ANALYSIS / "protocol_freeze.json"
GPU_LEDGER = ANALYSIS / "gpu_ledger.json"
CHECKPOINT_VERIFY = ANALYSIS / "checkpoint_verify.json"
CHECKPOINT_MANIFEST = ANALYSIS / "checkpoint_manifest.json"
MANIFEST_FROZEN = ANALYSIS / "MANIFEST_FROZEN"

TABLE_MAIN = ANALYSIS / "table_main.csv"
TABLE_SUMMARY = ANALYSIS / "table_summary.csv"
TABLE_PAIRS = ANALYSIS / "table_pairs.csv"
TABLE_SEED = ANALYSIS / "table_seed.csv"
TABLE_STABILITY = ANALYSIS / "table_stability.csv"
ANSWERS = ANALYSIS / "answers.json"
HISTORY_CURVE_CHECK = ANALYSIS / "history_curve_check.json"


# ---------------------------------------------------------------------------
# frozen data constants (identical to every previous random42 round)
# ---------------------------------------------------------------------------

CSV_PATH = DC.CSV_PATH
CSV_SHA256 = DC.CSV_SHA256
N_MOLECULES = DC.N_MOLECULES
SPLIT_SEED = DC.SPLIT_SEED
SPLIT_SIZES = dict(DC.SPLIT_SIZES)
V1_SPLIT_FILE = DC.V1_SPLIT_FILE
V1_SPLIT_SHA256 = DC.V1_SPLIT_SHA256
V1_LABELS_FILE = DC.V1_LABELS_FILE
V1_LABELS_SHA256 = DC.V1_LABELS_SHA256
SPLIT_SHA256 = dict(DC.SPLIT_SHA256)
TARGET_MEAN = DC.TARGET_MEAN
TARGET_SCALE = DC.TARGET_SCALE
TARGET_VECTOR_SHA256 = DC.TARGET_VECTOR_SHA256
VALID_VECTOR_SHA256 = DC.VALID_VECTOR_SHA256
TEST_VECTOR_SHA256 = DC.TEST_VECTOR_SHA256
LCRC_CACHE_CONTENT_FINGERPRINT = DC.LCRC_CACHE_CONTENT_FINGERPRINT
LCRC_TRAIN_NORM_FINGERPRINT = DC.LCRC_TRAIN_NORM_FINGERPRINT

#: Historical descriptive references (different protocol / checkpoint rule).
PNA_NATIVE_VALID_RMSE = 0.554034
PNA_NATIVE_TEST_RMSE = 0.5288742856070585
PNA_NATIVE_TEST_MAE = 0.37377129447390695

# ---------------------------------------------------------------------------
# models / arms
# ---------------------------------------------------------------------------

WIDTH = DC.WIDTH
CORR_DIM = DC.CORR_DIM
ATOM_OUT = DC.ATOM_OUT
W_OUT_INIT_STD = DC.W_OUT_INIT_STD

#: The six families trained this round.
FAMILIES = ("base", "roleblind", "g2l_loo", "g2l_all", "dist_true_fixed", "dist_blind")

MODEL_DISPLAY = {
    "base": "BASE",
    "roleblind": "ROLEBLIND",
    "g2l_loo": "G2L-LOO",
    "g2l_all": "G2L-ALL",
    "dist_true_fixed": "DIST-TRUE-FIXED",
    "dist_blind": "DIST-BLIND",
}

SEEDS = (0, 1, 2)

#: Pair keys ``(reference, candidate)``; the reported gain is
#: ``rmse(reference) - rmse(candidate)`` so a positive number means the
#: candidate is better.
PAIRS = (
    ("base", "roleblind"),
    ("roleblind", "g2l_loo"),
    ("roleblind", "g2l_all"),
    ("dist_blind", "dist_true_fixed"),
    ("roleblind", "dist_true_fixed"),
)

#: Display string for a pair, e.g. ``ROLEBLIND - BASE``.
PAIR_LABEL = {
    ("base", "roleblind"): "ROLEBLIND - BASE",
    ("roleblind", "g2l_loo"): "G2L-LOO - ROLEBLIND",
    ("roleblind", "g2l_all"): "G2L-ALL - ROLEBLIND",
    ("dist_blind", "dist_true_fixed"): "DIST-TRUE-FIXED - DIST-BLIND",
    ("roleblind", "dist_true_fixed"): "DIST-TRUE-FIXED - ROLEBLIND",
}


def arm_key(family: str, seed: int) -> str:
    return f"{family}_s{int(seed)}"


ARM_SPECS: dict[str, dict[str, Any]] = {}
for _family in FAMILIES:
    for _seed in SEEDS:
        ARM_SPECS[arm_key(_family, _seed)] = {"family": _family, "seed": int(_seed)}

ARMS = tuple(ARM_SPECS)


def arm_pool(arm: str, preferred: bool = True) -> str:
    """Default GPU pool assignment used when launching through ``rr``.

    The split simply balances the two pools; the launch plan records it and the
    runner never needs it to be canonical (provenance is taken from ``rr``).
    """
    del preferred
    family = ARM_SPECS[arm]["family"]
    return "res-gpu1" if family in ("roleblind", "g2l_loo", "g2l_all") else "res-gpu0"


# ---------------------------------------------------------------------------
# training recipe (frozen R3, checkpoint rule replaced)
# ---------------------------------------------------------------------------

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
    "checkpoint": "single_best_validation420_rmse_epoch",
    "early_stopping": False,
    "ssl": False,
}
SHUFFLE_OFFSET = DC.SHUFFLE_OFFSET  # 917_000, identical batch order per seed

#: Fields that must equal the frozen historical recipe (everything except the
#: checkpoint rule, which is the one deliberate change of this round).
_FROZEN_RECIPE_FIELDS = tuple(k for k in TRAIN_RECIPE if k != "checkpoint")
for _field in _FROZEN_RECIPE_FIELDS:
    if TRAIN_RECIPE[_field] != DC.TRAIN_RECIPE[_field]:
        raise RuntimeError(
            f"recipe drift vs the frozen historical recipe at {_field}: "
            f"{TRAIN_RECIPE[_field]!r} != {DC.TRAIN_RECIPE[_field]!r}"
        )
if DC.TRAIN_RECIPE["checkpoint"] != "average_last_5_epoch_end_state_dicts":
    raise RuntimeError("the historical checkpoint rule is not the last-5 soup any more")

BOOTSTRAP = dict(DC.BOOTSTRAP)

GPU_HARD_CAP_HOURS = 14.0
WALL_CAP_HOURS = 7.0

# ---------------------------------------------------------------------------
# generic helpers
# ---------------------------------------------------------------------------

sha256_file = DC.sha256_file
sha256_bytes = DC.sha256_bytes
sha256_json = DC.sha256_json
write_json = DC.write_json
read_json = DC.read_json
track_rel = DC.track_rel
utcnow = DC.utcnow
state_dict_fingerprint = DC.state_dict_fingerprint
rows_sha256_int64 = DC.rows_sha256_int64
vector_sha256_float64 = DC.vector_sha256_float64
log = DC.log


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
    }


def code_file_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(EXP_DIR)): sha256_file(path)
        for path in sorted(CODE_DIR.glob("*.py"))
    }


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
    for key in _FROZEN_RECIPE_FIELDS:
        _need(t[key], TRAIN_RECIPE[key], f"training.{key}")
    _need(t["checkpoint"], TRAIN_RECIPE["checkpoint"], "training.checkpoint")
    _need([float(x) for x in t["adam_betas"]], [0.9, 0.999], "training.adam_betas")
    _need(float(t["adam_eps"]), 1.0e-8, "training.adam_eps")
    _need(int(t["shuffle_offset"]), SHUFFLE_OFFSET, "training.shuffle_offset")
    _need(int(cfg["model"]["width"]), WIDTH, "model.width")
    _need(int(cfg["model"]["correction_dim"]), CORR_DIM, "model.correction_dim")
    _need(list(cfg["models"]), list(FAMILIES), "models")
    _need([int(s) for s in cfg["seeds"]], list(SEEDS), "seeds")
    _need(float(cfg["budget"]["gpu_hard_cap_hours"]), GPU_HARD_CAP_HOURS, "budget.gpu_hard_cap_hours")


def load_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    import yaml

    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    validate_config(cfg)
    return cfg


def run_dir(arm: str) -> Path:
    return RUNS / str(arm)


def load_train_json(arm: str) -> dict[str, Any]:
    path = run_dir(arm) / "train.json"
    if not path.exists():
        raise FileNotFoundError(f"{arm}: train.json missing ({track_rel(path)})")
    return read_json(path)


def completed_arms() -> list[str]:
    return [a for a in ARMS if (run_dir(a) / "train.json").exists()]


__all__ = [name for name in dir() if not name.startswith("_")]
