"""Shared frozen constants, paths and integrity helpers for the random42 round.

This round is a *downstream* consumer of three read-only historical artifacts:

* the original ``data/lipophilicity/Lipophilicity.csv`` (hash-verified);
* the committed, label-free raw-graph artifact produced by
  ``tracks/structure-contribution/results/bjoc_cross_task_lipophilicity_v1``;
* the frozen ``lipo_v1`` LCRC feature spec / data assembly in
  ``tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code``.

Nothing in a historical directory is written.  Every new artifact lives under
this experiment directory.

The protocol constants below are duplicated in ``configs/benchmark_v1.yaml``
and hard-checked by :func:`validate_config`; a config edit that drifts from the
frozen task specification fails before any training starts.
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
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
EXP_DIR = CODE_DIR.parent
REPO = EXP_DIR.parents[2]

VENDOR = CODE_DIR / "upstream"
PREP = REPO / "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1"
PREP_CODE = PREP / "code"
HIST_V2 = REPO / "tracks/structural-sufficiency/lcrc_lipo_mechanism_generalization_ssl_v2"

CONFIG_PATH = EXP_DIR / "configs/benchmark_v1.yaml"
DATA_DIR = EXP_DIR / "data"
CACHE_DIR = EXP_DIR / "cache"
ANALYSIS = EXP_DIR / "analysis"
CHECKPOINTS = EXP_DIR / "checkpoints"
PREDICTIONS = EXP_DIR / "predictions"
RUNS = ANALYSIS / "runs"

SPLIT_FILE = DATA_DIR / "split_esa_random42_v1.json"
LABELS_TRAIN_VALID = DATA_DIR / "labels_train_valid_v1.json"
STRUCTURES_FILE = DATA_DIR / "structures_esa_random42_v1.npz"
STRUCTURES_META = DATA_DIR / "structures_esa_random42_v1_meta.json"

CSV_PATH = REPO / "data/lipophilicity/Lipophilicity.csv"
RAW_GRAPH = REPO / "tracks/structure-contribution/results/bjoc_cross_task_lipophilicity_v1/dataset/lipo_raw_v1.json.gz"
ROW_MANIFEST = REPO / "tracks/structure-contribution/results/bjoc_cross_task_lipophilicity_v1/row_manifest.json"

ROUND_ID = "lcrc-lipo-esa-random42-v1"
PROTOCOL_ID = ROUND_ID

# ---------------------------------------------------------------------------
# frozen data constants
# ---------------------------------------------------------------------------

CSV_SHA256 = "aed41590cb30609d51d8e08ad3ff06495a76e80e211358801f596b10da69bacd"
N_MOLECULES = 4200
SPLIT_SEED = 42
TRAIN_SEED = 0
SPLIT_SIZES = {"train": 3360, "valid": 420, "test": 420}
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
MULTI_COMPONENT_ROW = 1646

UPSTREAM_COMMIT = "861a0b12d445ae80cc69b3948af623b4331321d4"
UPSTREAM_REPO = "https://github.com/davidbuterez/edge-set-attention"

# ---------------------------------------------------------------------------
# frozen model / training constants
# ---------------------------------------------------------------------------

LCRC_CONFIG = {
    "model": "LCRC-A-R3-scratch",
    "seed": TRAIN_SEED,
    "batch_size": 128,
    "epochs": 360,
    "optimizer": "AdamW",
    "peak_lr": 0.0003,
    "min_lr": 0.00001,
    "weight_decay": 0.001,
    "scheduler": "per_step_cosine",
    "dropout": 0.10,
    "gradient_clip": 5.0,
    "loss": "standardized_mse",
    "checkpoint": "average_last_5_epoch_end_state_dicts",
    "early_stopping": False,
    "ssl": False,
}
PNA_CONFIG = {
    "model": "PNA-native",
    "seed": TRAIN_SEED,
    "conv_type": "PNA",
    "num_layers": 4,
    "output_node_dim": 256,
    "gnn_intermediate_dim": 256,
    "batch_size": 128,
    "optimizer": "AdamW",
    "lr": 0.0001,
    "weight_decay": 0.0000000001,
    "gradient_clip": 0.5,
    "loss": "standardized_mse",
    "max_epochs": 360,
    "early_stopping_patience": 30,
    "lr_scheduler": "ReduceLROnPlateau",
    "lr_scheduler_factor": 0.5,
    "lr_scheduler_patience": 15,
    "checkpoint": "best_validation_loss",
    "precision": "bf16",
    "aggregation": ["mean", "min", "max", "std"],
    "scalers": ["identity", "amplification", "attenuation"],
    "towers": 5,
}
ESA_CONFIG = {
    "model": "ESA-native",
    "seed": TRAIN_SEED,
    "apply_attention_on": "edge",
    "layer_types": ["S", "S", "M", "M", "M", "S", "P"],
    "hidden_dims": [256] * 7,
    "num_heads": [8] * 7,
    "graph_dim": 512,
    "pma_seeds": 32,
    "norm_type": "LN",
    "pre_or_post": "post",
    "use_mlps": True,
    "mlp_type": "gated_mlp",
    "mlp_layers": 2,
    "mlp_hidden_size": 512,
    "sab_dropout": 0.0,
    "mab_dropout": 0.0,
    "pma_dropout": 0.0,
    "mlp_dropout": 0.0,
    "batch_size": 128,
    "optimizer": "AdamW",
    "lr": 0.0001,
    "weight_decay": 0.0000000001,
    "gradient_clip": 0.5,
    "loss": "standardized_mse",
    "max_epochs": 360,
    "early_stopping_patience": 30,
    "lr_scheduler": "ReduceLROnPlateau",
    "lr_scheduler_factor": 0.5,
    "lr_scheduler_patience": 15,
    "checkpoint": "best_validation_loss",
    "positional_encoding": "none",
    "external_pretraining": False,
    "attention_backend": "torch",
    "precision": "bf16",
    "attn_residual_dropout": 0.0,
    "pma_residual_dropout": 0.0,
    "use_mlp_ln": True,
}
XGB_CONFIG = {
    "model": "XGB-P3",
    "n_estimators": 800,
    "max_depth": 5,
    "min_child_weight": 1,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
    "tree_method": "hist",
    "random_state": 0,
    "n_jobs": 4,
}
GPU_HARD_CAP_HOURS = 6.0
SOUP_LAST = 5
SHUFFLE_OFFSET = 917_000
TARGET_SCALER = "sklearn.preprocessing.StandardScaler (train3360 only; population std)"

# ---------------------------------------------------------------------------
# generic helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_bytes(stable_json(obj).encode("utf-8"))


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_json(path: Path | str, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def track_rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "numpy": np.__version__,
        "pid": os.getpid(),
    }


def log(message: str) -> None:
    print(message, flush=True)


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def verify_sha256(path: Path | str, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise PermissionError(f"{label}: sha256 {actual} != frozen {expected}")
    return actual


def rows_sha256_int64(rows: Sequence[int] | np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(np.asarray(rows, dtype="<i8")).tobytes())


def iter_chunks(n: int, size: int) -> Iterable[list[int]]:
    for start in range(0, int(n), int(size)):
        yield list(range(start, min(start + int(size), int(n))))


def config_hash(cfg: Mapping[str, Any]) -> str:
    return sha256_json(cfg)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def load_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    import yaml

    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    validate_config(cfg)
    return cfg


def _need(actual, expected, name) -> None:
    if actual != expected:
        raise ValueError(f"config drift at {name}: {actual!r} != frozen {expected!r}")


def validate_config(cfg: Mapping[str, Any]) -> None:
    _need(cfg.get("protocol_id"), PROTOCOL_ID, "protocol_id")
    _need(cfg.get("round_id"), ROUND_ID, "round_id")
    _need(cfg["data"]["csv_sha256"], CSV_SHA256, "data.csv_sha256")
    _need(int(cfg["data"]["n_molecules"]), N_MOLECULES, "data.n_molecules")
    _need(int(cfg["split"]["seed"]), SPLIT_SEED, "split.seed")
    _need(dict(cfg["split"]["sizes"]), SPLIT_SIZES, "split.sizes")
    _need(dict(cfg["split"]["sha256"]), SPLIT_SHA256, "split.sha256")
    _need(int(cfg["training"]["seed"]), TRAIN_SEED, "training.seed")
    _need(float(cfg["budget"]["gpu_hard_cap_hours"]), GPU_HARD_CAP_HOURS, "budget.gpu_hard_cap_hours")
    _need(cfg["upstream"]["commit"], UPSTREAM_COMMIT, "upstream.commit")
    for key, frozen in (("lcrc", LCRC_CONFIG), ("pna", PNA_CONFIG), ("esa", ESA_CONFIG), ("xgb", XGB_CONFIG)):
        got = cfg["models"][key]
        for field, expected in frozen.items():
            _need(got[field], expected, f"models.{key}.{field}")
    _need(cfg["evaluation"]["primary_metric"], "rmse", "evaluation.primary_metric")
    _need(cfg["evaluation"]["test_policy"], "terminal_one_shot", "evaluation.test_policy")
    _need(cfg["evaluation"]["target_scaler"], TARGET_SCALER, "evaluation.target_scaler")


__all__ = [name for name in dir() if not name.startswith("_")]
