"""Shared plumbing for the direct-local-object round.

Nothing scientific lives here: only paths, identity, IO, seeding, metric
helpers and the official OGB category dimensions.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROUND_DIR = CODE_DIR.parent
REPO_ROOT = ROUND_DIR.parents[2]
CONFIG_PATH = ROUND_DIR / "configs" / "dlo_s0.json"

ATOM_COLUMNS = 9
BOND_COLUMNS = 3
MAX_ATOMS = 6
#: canonical slot pairs, lexicographic (01,02,...,05,12,...,45)
EDGE_PAIRS = (
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
    (1, 2), (1, 3), (1, 4), (1, 5),
    (2, 3), (2, 4), (2, 5),
    (3, 4), (3, 5),
    (4, 5),
)
SPLITS = ("train", "valid", "test")
SPLIT_ID = {"train": 0, "valid": 1, "test": 2}
SPLIT_NAME = {0: "train", 1: "valid", 2: "test"}
ARM_S56 = "DIRECT_S56"
ARM_W34 = "MATCHED_W34"
ARMS = (ARM_S56, ARM_W34)
ARM_PREFIX = {ARM_S56: "s56", ARM_W34: "w34"}


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def round_dir() -> Path:
    return ROUND_DIR


def data_dir() -> Path:
    return ROUND_DIR / "data"


def analysis_dir() -> Path:
    return ROUND_DIR / "analysis"


def results_root() -> Path:
    override = os.environ.get("DLO_RESULTS_DIR", "").strip()
    if override:
        base = Path(override).expanduser()
        if not base.is_absolute():
            base = REPO_ROOT / base
        return base
    return ROUND_DIR / "results"


def run_dir(run_id: str) -> Path:
    """Resolve an output directory: absolute, repo-relative, or results/<name>."""
    raw = Path(run_id)
    if raw.is_absolute():
        path = raw
    elif len(raw.parts) > 1:
        path = REPO_ROOT / raw
    else:
        path = results_root() / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode())
    digest.update(str(arr.shape).encode())
    digest.update(arr.tobytes())
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return str(value)
    return value


def write_json(path: Path | str, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: Path | str, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> Path:
    import csv

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return target
    if columns is None:
        ordered: list[str] = []
        for row in rows:
            for key in row:
                if key not in ordered:
                    ordered.append(key)
        columns = ordered
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in columns})
    return target


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    diff = np.asarray(pred, np.float64) - np.asarray(target, np.float64)
    return float(np.sqrt(np.mean(diff * diff)))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    diff = np.asarray(pred, np.float64) - np.asarray(target, np.float64)
    return float(np.mean(np.abs(diff)))


def target_scaler(labels: np.ndarray, train_rows: np.ndarray) -> dict[str, float]:
    y = np.asarray(labels, np.float64)[np.asarray(train_rows, np.int64)]
    return {
        "mean": float(np.mean(y)),
        "std": float(np.std(y)),
        "n": int(y.size),
    }


def official_feature_dims() -> tuple[list[int], list[int]]:
    from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

    atom = [int(x) for x in get_atom_feature_dims()]
    bond = [int(x) for x in get_bond_feature_dims()]
    if len(atom) != ATOM_COLUMNS or len(bond) != BOND_COLUMNS:
        raise ValueError("unexpected official OGB feature definition")
    return atom, bond


def set_deterministic(seed: int) -> dict[str, Any]:
    import torch

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {
        "seed": int(seed),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "numpy_seed": int(seed),
        "torch_manual_seed": int(seed),
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "use_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }


def enable_tf32(enabled: bool) -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {"requested": bool(enabled), "available": bool(torch.cuda.is_available())}
    if not info["available"]:
        info["enabled"] = False
        return info
    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
    torch.backends.cudnn.allow_tf32 = bool(enabled)
    info["matmul_allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
    info["cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)
    info["enabled"] = True
    return info


def environment() -> dict[str, Any]:
    import platform
    import socket
    import sys

    import torch

    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = int(torch.cuda.device_count())
        info["gpu_capability"] = ".".join(str(x) for x in torch.cuda.get_device_capability(0))
        try:
            import subprocess

            raw = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True, timeout=10,
            ).strip().splitlines()
            info["nvidia_driver"] = raw[0] if raw else None
        except Exception as exc:  # noqa: BLE001
            info["nvidia_driver_error"] = repr(exc)
    return info


def git_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"
