"""Shared constants, read-only reuse guards and integrity helpers.

Round: ``lcrc-lipo-contextual-contrib-v1``.

Scope
-----
One frozen split (random42: Train3360 / Validation420), one training seed (0),
two arms that differ *only* in the readout of the atom contribution:

    BASE     ``lcrc_model.LCRC`` (``lipo_v1``, width 64), unchanged, with the
             v2 ``DropoutAtLocal`` hook actually executed at ``model.local``;
    CONTEXT  BASE plus four new ``Linear`` layers that compute a per-molecule
             element-class context correction on the hetero-atom inputs of the
             per-atom contribution MLP ``f``.

The frozen random42 artifacts from ``lcrc_lipo_esa_random42_v1`` are consumed
read-only and hash-verified.  No Test label, Test prediction, Test metric or
per-molecule Test information is read anywhere in this round.

Nothing in a historical directory is written.  Every new artifact lives under
this experiment directory.
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

V1 = REPO / "tracks/structural-sufficiency/lcrc_lipo_esa_random42_v1"
V1_CODE = V1 / "code"
V1_DATA = V1 / "data"
V1_ANALYSIS = V1 / "analysis"
PREP = REPO / "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1"
PREP_CODE = PREP / "code"
V2 = REPO / "tracks/structural-sufficiency/lcrc_lipo_mechanism_generalization_ssl_v2"
V2_CODE = V2 / "code"
GAP_AUDIT = REPO / "tracks/structural-sufficiency/lcrc_lipo_random42_gap_audit_v1"

for _path in (str(V1_CODE), str(PREP_CODE), str(V2_CODE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

CONFIG_PATH = EXP_DIR / "configs/contextual_contrib_v1.yaml"
DATA_DIR = EXP_DIR / "data"
CACHE_DIR = EXP_DIR / "cache"
ANALYSIS = EXP_DIR / "analysis"
CHECKPOINTS = EXP_DIR / "checkpoints"
PREDICTIONS = EXP_DIR / "predictions"
RUNS = ANALYSIS / "runs"

PROTOCOL_PATH = EXP_DIR / "PROTOCOL.md"
PROTOCOL_FREEZE = ANALYSIS / "protocol_freeze.json"
PROTOCOL_SHA256 = ANALYSIS / "protocol_sha256.json"
DATA_AUDIT = ANALYSIS / "data_audit.json"
CPU_GATES = ANALYSIS / "cpu_gates.json"
GPU_LEDGER = ANALYSIS / "gpu_ledger.json"
COMPARISON = ANALYSIS / "comparison.json"
TAIL_AUDIT = ANALYSIS / "tail_audit.json"
SUBGROUPS = ANALYSIS / "subgroups.json"
CONTEXT_USAGE = ANALYSIS / "context_usage.json"
DECISION = ANALYSIS / "decision.json"
PER_MOLECULE = ANALYSIS / "per_molecule_valid.csv"
METRICS_VALIDATION = ANALYSIS / "metrics_validation.json"
RR_STATUS = ANALYSIS / "rr_status"

ROUND_ID = "lcrc-lipo-contextual-contrib-v1"
PROTOCOL_ID = ROUND_ID

# ---------------------------------------------------------------------------
# frozen random42 data constants (identical to the frozen v1 round)
# ---------------------------------------------------------------------------

CSV_PATH = REPO / "data/lipophilicity/Lipophilicity.csv"
CSV_SHA256 = "aed41590cb30609d51d8e08ad3ff06495a76e80e211358801f596b10da69bacd"
N_MOLECULES = 4200
SPLIT_SEED = 42
TRAIN_SEED = 0
SPLIT_SIZES = {"train": 3360, "valid": 420}
SPLIT_SHA256 = {
    "train": "0df18c72753974d502f8bede1cd2aa5c0ebdc1df450b8871fffdb1f67d2648ee",
    "valid": "8f9c35171ac2125e678d2ad44d87b293719bb9806a528810ac271e9aa39f9b44",
}
SPLIT_FIRST_ROWS = {
    "train": [3942, 3415, 1426, 2365, 3462],
    "valid": [501, 1649, 3579, 2873, 106],
}
MULTI_COMPONENT_ROW = 1646

V1_SPLIT_FILE = V1_DATA / "split_esa_random42_v1.json"
V1_SPLIT_SHA256 = "1834414a6da73e9d2c5942ad157a345f08653110acfc4b9fef83c3064bc233b4"
V1_LABELS_FILE = V1_DATA / "labels_train_valid_v1.json"
V1_LABELS_SHA256 = "1da269356e0e89974a7388d1e430d7e810a0ab1056be5cd747a1c8ed3d2009ca"
V1_METRICS_VALID = V1_ANALYSIS / "metrics_validation.json"
V1_METRICS_VALID_SHA256 = "c540f9ac89a391ea13c8235b7b14cb7faf10ae2a9d60616d2e4716659c576fdd"
V1_LCRC_CKPT = V1_ANALYSIS / "runs/lcrc_a_r3_s0/checkpoint.pt"
V1_LCRC_VALID = V1_ANALYSIS / "runs/lcrc_a_r3_s0/valid_predictions.npz"

TARGET_MEAN = 2.2081011904761905
TARGET_SCALE = 1.1906281212426577
TARGET_VECTOR_SHA256 = "85a4c7c559f4b7fd58ad78b1b05d5a4160e0f60c351b7fa842eb0354aa47abd8"
VALID_VECTOR_SHA256 = "09bc1a1e1756bbd09dd17a5fc77a1b3c33e7010aef55fc352a4bba1d9822636f"
LCRC_TRAIN_NORM_FINGERPRINT = "4d6e5bf456ff9d2217eb11a2c4a29d3f7cbbcb95665bf56e62b93c3ae93d0e26"

#: Historical random42 LCRC-A-R3 Validation RMSE.  Descriptive reference only:
#: the historical training loop never executed the configured dropout, so this
#: round's matched BASE (dropout actually applied) is not directly comparable.
HISTORICAL_LCRC_VALID_RMSE = 0.5890072081930064

#: Read-only source hashes of the exact implementations this round imports.
LCRC_MODEL_SHA256 = "d5af76cb9b5998e6bfa068afd28c483c93dde579cfb85344ed928582b05c6d37"
FEATURE_SPEC_SHA256 = "37f860a90bd805669b570d44394a791947c9eabdd7cc68a089264ced7c7a4260"
LCRC_DATA_SHA256 = "fab6ce8ef61e4997f67516b2b2e5d861c72ed870d57a2599bc2787b8cd3ac3d3"
V2_MODELS_SHA256 = "00b3d3c3657cb7320affd79d8829110ad8d34ca8327191781882f225853c2988"
V1_COMMON_CODE_SHA256 = "6caa31951a8a22b333953ec377e0376a612a803661111c1fb8903d70345173be"
V1_DATA_CODE_SHA256 = "cd08aec7134cf0c31a5e323c269e319164de59432416d8fcb91a399c671f2c3f"
V1_LABELS_CODE_SHA256 = "78bf0adf3a24a182a490c2fd1bff5f6eb13aaa393d5e2421e0899d6262bb1214"
V1_METRICS_CODE_SHA256 = "8535816d8bd8bbfd46f7a8571f0cb318e6e349709be42150c62a7eea207d7228"
V1_NATIVE_TRAIN_CODE_SHA256 = "98f7583d704b06594f63978e5cea2367c0a79014d68bdbf2f0f7b983804b6d7f"

# ---------------------------------------------------------------------------
# frozen model / training constants
# ---------------------------------------------------------------------------

WIDTH = 64
CONTEXT_DIM = 32
HETERO_ELEMENTS = ("N", "O", "S", "P")
CARBON_HALOGEN_ELEMENTS = ("C", "F", "Cl", "Br", "I")
W_OUT_INIT_STD = 1.0e-3

LCRC_BASE_PARAMS_REFERENCE = 247_106
# W_atom + W_het + W_ch = 3 * (64*32 + 32) = 6 240; W_out = 32*64 + 64 = 2 112.
CONTEXT_NEW_PARAMS_REFERENCE = (WIDTH * CONTEXT_DIM + CONTEXT_DIM) * 3 + (CONTEXT_DIM * WIDTH + WIDTH)
CONTEXT_TOTAL_PARAMS_REFERENCE = LCRC_BASE_PARAMS_REFERENCE + CONTEXT_NEW_PARAMS_REFERENCE

TRAIN_RECIPE = {
    "model_family": "LCRC-A-R3 (lipo_v1, width 64)",
    "seed": TRAIN_SEED,
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

ARMS = ("base", "context")
ARM_MODEL_NAME = {
    "base": "LCRC-A-R3-scratch-BASE",
    "context": "LCRC-A-R3-contextual-contribution",
}
ARM_POOL = {"base": "res-gpu0", "context": "res-gpu1"}
ARM_RUN_KEY = {"base": "base_s0", "context": "context_s0"}

# ---------------------------------------------------------------------------
# frozen decision rules (declared before any formal run)
# ---------------------------------------------------------------------------

GAIN_THRESHOLD = 0.020
MAE_WORSENING_TOLERANCE = 0.010
TAIL_ROWS = (1630, 2924, 822, 1153, 3634)
TAIL_SHARE_THRESHOLD = 0.50

GPU_HARD_CAP_HOURS = 0.45

SCAFFOLD_FINGERPRINT_RADIUS = 2
SCAFFOLD_FINGERPRINT_BITS = 2048

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
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=_jsonable) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


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


def vector_sha256_float64(values: Sequence[float] | np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(np.asarray(values, dtype="<f8")).tobytes())


def iter_chunks(n: int, size: int) -> Iterable[list[int]]:
    for start in range(0, int(n), int(size)):
        yield list(range(start, min(start + int(size), int(n))))


def config_hash(cfg: Mapping[str, Any]) -> str:
    return sha256_json(cfg)


def state_dict_fingerprint(state: Mapping[str, Any]) -> str:
    import torch

    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        tensor = value.detach().cpu() if torch.is_tensor(value) else torch.as_tensor(value)
        h.update(key.encode("utf-8"))
        h.update(np.ascontiguousarray(tensor.numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def code_file_hashes() -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted(CODE_DIR.glob("*.py")):
        out[str(path.relative_to(EXP_DIR))] = sha256_file(path)
    return out


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def _need(actual, expected, name) -> None:
    if actual != expected:
        raise ValueError(f"config drift at {name}: {actual!r} != frozen {expected!r}")


def validate_config(cfg: Mapping[str, Any]) -> None:
    _need(cfg.get("protocol_id"), PROTOCOL_ID, "protocol_id")
    _need(cfg.get("round_id"), ROUND_ID, "round_id")
    _need(int(cfg["seed"]), TRAIN_SEED, "seed")
    _need(int(cfg["split_seed"]), SPLIT_SEED, "split_seed")
    _need(cfg["data"]["source_round"], "lcrc_lipo_esa_random42_v1", "data.source_round")
    _need(cfg["data"]["split_file_sha256"], V1_SPLIT_SHA256, "data.split_file_sha256")
    _need(cfg["data"]["labels_file_sha256"], V1_LABELS_SHA256, "data.labels_file_sha256")
    _need(int(cfg["data"]["n_molecules"]), N_MOLECULES, "data.n_molecules")
    _need({k: int(v) for k, v in cfg["data"]["split_sizes"].items()}, SPLIT_SIZES, "data.split_sizes")
    _need(cfg["data"]["test_access"], "forbidden", "data.test_access")
    t = cfg["training"]
    for key in (
        "seed",
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
    d = cfg["decision"]
    _need(float(d["gain_threshold"]), GAIN_THRESHOLD, "decision.gain_threshold")
    _need(float(d["mae_worsening_tolerance"]), MAE_WORSENING_TOLERANCE, "decision.mae_worsening_tolerance")
    _need(tuple(int(x) for x in d["tail_rows"]), TAIL_ROWS, "decision.tail_rows")
    _need(float(d["tail_share_threshold"]), TAIL_SHARE_THRESHOLD, "decision.tail_share_threshold")
    _need(float(cfg["budget"]["gpu_hard_cap_hours"]), GPU_HARD_CAP_HOURS, "budget.gpu_hard_cap_hours")
    _need(dict(cfg["budget"]["pools"]), ARM_POOL, "budget.pools")


def load_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    import yaml

    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    validate_config(cfg)
    return cfg


__all__ = [name for name in dir() if not name.startswith("_")]
