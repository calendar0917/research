"""FGCR-v0 shared plumbing: paths, identity, config, seeding, metrics, IO.

No scientific choice lives here.  Every frozen definition is in
``PROTOCOL.md`` and ``configs/fgcr.json``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROUND_DIR = CODE_DIR.parent
REPO_ROOT = ROUND_DIR.parents[2]
CONFIG_PATH = ROUND_DIR / "configs" / "fgcr.json"
ANALYSIS_DIR = ROUND_DIR / "analysis"
RESULTS_DIR = ROUND_DIR / "results"

DATASET = "ogbg-mollipo"
CACHE_SHA256 = "fe002287fa9c581278bc5430f5b4a5b26b35d9d4d5207d138f15956d93123c7c"
EXPECTED_SPLIT_SIZES = {"train": 3360, "valid": 420, "test": 420}

ATOM_COLUMNS = 9
BOND_COLUMNS = 3
CHANNELS = 16
STATES = 2


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        arr = np.ascontiguousarray(array)
        digest.update(str(arr.dtype).encode())
        digest.update(str(arr.shape).encode())
        digest.update(arr.tobytes())
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(dumps_json(payload).encode("utf-8")).hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def git_dirty() -> bool | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        )
        return bool(out.stdout.strip())
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# config / JSON / CSV
# ---------------------------------------------------------------------------


def load_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return None if not math.isfinite(value) else value
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def dumps_json(payload: Any) -> str:
    return json.dumps(jsonable(payload), sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


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


def write_csv(path: Path | str, rows: Sequence[Mapping[str, Any]],
              columns: Sequence[str] | None = None) -> Path:
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


def append_csv(path: Path | str, row: Mapping[str, Any],
               columns: Sequence[str]) -> Path:
    """Append one row, writing the header when the file is new/empty."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    new = (not target.is_file()) or target.stat().st_size == 0
    with target.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in columns})
    return target


# ---------------------------------------------------------------------------
# seeding
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # noqa: BLE001 - provenance/CPU-only checks
        pass


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    diff = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.sqrt(np.mean(diff * diff)))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    diff = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.mean(np.abs(diff)))


def train_target_scaler(labels: np.ndarray) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    mean = float(y.mean())
    std = float(y.std(ddof=0))
    if not np.isfinite(mean) or not np.isfinite(std):
        raise ValueError("non-finite target statistics")
    if std <= 0.0:
        raise ValueError("Train target std is zero")
    return {"mean": mean, "std": std, "n": int(y.size),
            "min": float(y.min()), "max": float(y.max())}


def metrics_raw(pred_std: np.ndarray, target_raw: np.ndarray,
                scaler: Mapping[str, float]) -> dict[str, float]:
    pred_raw = np.asarray(pred_std, dtype=np.float64) * float(scaler["std"]) + float(scaler["mean"])
    target = np.asarray(target_raw, dtype=np.float64)
    return {"rmse": rmse(pred_raw, target), "mae": mae(pred_raw, target)}


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------


def environment_snapshot() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "cwd": os.getcwd(),
        "env": {k: os.environ.get(k) for k in (
            "CUDA_VISIBLE_DEVICES", "RR_RUN_ID", "SLURM_JOB_ID", "SLURM_NODELIST",
        ) if os.environ.get(k) is not None},
    }
    try:
        import torch

        payload["torch"] = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "device_count": int(torch.cuda.device_count()),
            "devices": [torch.cuda.get_device_name(i)
                        for i in range(torch.cuda.device_count())],
        }
    except Exception as exc:  # noqa: BLE001
        payload["torch"] = {"error": repr(exc)}
    try:
        import numpy

        payload["numpy"] = numpy.__version__
    except Exception as exc:  # noqa: BLE001
        payload["numpy"] = {"error": repr(exc)}
    return payload


def utc_stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_dir(path: Path | str) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def relative(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def import_sq_data():
    """Import the pinned, read-only SQ data module and verify its identity."""
    sq_code = REPO_ROOT / "tracks" / "structural-query-learning" / "code"
    if str(sq_code) not in sys.path:
        sys.path.insert(0, str(sq_code))
    import sq_data  # noqa: PLC0415

    expected = {
        "sq_data.py": "89ad5436be3ddb8825fe877535e7fbb13679f4cb012a3a3073d63deaf9a83647",
        "sq_common.py": "fe7333978fd748cf2a61fcdb4d3fa73bf855ebbdc6b05c439477b2c921b9d8cb",
    }
    found = {}
    for name, sha in expected.items():
        actual = sha256_file(sq_code / name)
        if actual != sha:
            raise RuntimeError(f"pinned SQ source changed: {name} {actual} != {sha}")
        found[name] = actual
    return sq_data, found