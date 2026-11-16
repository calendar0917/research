"""Shared paths, hashing, JSON helpers and the resource ledger.

Round: ``sq-pilot-small-v1`` (structural query learning, small-data prototype).

Nothing scientific lives here: only identity (hashes), IO and bookkeeping.
Every path is resolved from this file's location so the module works both in the
local checkout and in the read-only compute checkout on ``res``.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROUND_ID = "sq-pilot-small-v1"
PROTOCOL_ID = ROUND_ID
STUDY_ID = ROUND_ID

CODE_DIR = Path(__file__).resolve().parent
TRACK_ROOT = CODE_DIR.parent
REPO_ROOT = TRACK_ROOT.parents[1]

DATA_DIR = TRACK_ROOT / "data"          # committed label-free cache (small)
CACHE_DIR = TRACK_ROOT / "cache"        # local scratch caches (git-ignored)
ANALYSIS_DIR = TRACK_ROOT / "analysis"


def _results_dir() -> Path:
    """Durable artefact root; ``SQ_RESULTS_DIR`` allows a scratch destination."""
    override = os.environ.get("SQ_RESULTS_DIR", "").strip()
    if override:
        base = Path(override).expanduser()
        if not base.is_absolute():
            base = REPO_ROOT / base
        return base / ROUND_ID
    return TRACK_ROOT / "results" / ROUND_ID


RESULTS_DIR = _results_dir()

#: The two official OGB tasks of this round, in pipeline order.
DATASETS: tuple[str, ...] = ("ogbg-molesol", "ogbg-mollipo")

#: Expected official split sizes (asserted against the real manifest at prepare).
EXPECTED_SPLIT_SIZES = {
    "ogbg-molesol": {"train": 902, "valid": 113, "test": 113},
    "ogbg-mollipo": {"train": 3360, "valid": 420, "test": 420},
}

# Frozen file fingerprints (SHA-256) of the committed label-free caches.
# Comparability of this round means: identical caches byte for byte.  A run
# refuses to start when a cache changed, so a "reused" artefact can never be a
# different dataset than the one the protocol names.
EXPECTED_CACHE_SHA256 = {
    "ogbg_molesol_sq-pilot-cache-v1.npz": "3b7a8e16ca9c7ab240add84fbe26050dc81e22359bcc36f1c8a8111b77aec24a",
    "ogbg_mollipo_sq-pilot-cache-v1.npz": "fe002287fa9c581278bc5430f5b4a5b26b35d9d4d5207d138f15956d93123c7c",
    "ogbg_molesol_ecfp4_sq-pilot-cache-v1.npz": "653fb0190abc4af43822ec643bacbcbdbfdd7555f5e632574c9f61728d574f2e",
    "ogbg_mollipo_ecfp4_sq-pilot-cache-v1.npz": "7990ce4c90a0fdb032897076e19221b62b1361fc98f36fea248cf097bb424ae4",
    "ogbg_molesol_sq-pilot-lipo-raw-v1.npz": "a3ab9f7e22798a8f6fdc7f0913b1136b7cf59d4ffc7168faecc7f8b3094710b1",
    "ogbg_mollipo_sq-pilot-lipo-raw-v1.npz": "3005aa55e766d83a237b1659ad2cabb51b8702eef6d82c203fdecf9cb3c42cc3",
}

TRAIN_SEED = 0
SEARCH_RNG_SEED = 1701
RANDOM_LIBRARY_SEED = 1702
BOOTSTRAP_SEED = 1703


# ---------------------------------------------------------------------------
# hashing / identity
# ---------------------------------------------------------------------------


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


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


def sha256_json(payload: Any) -> str:
    return sha256_text(dumps_json(payload))


def dumps_json(payload: Any) -> str:
    return json.dumps(jsonable(payload), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return str(value)
    return value


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path | str, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target


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


# ---------------------------------------------------------------------------
# environment / provenance
# ---------------------------------------------------------------------------


def environment_snapshot() -> dict[str, Any]:
    import importlib

    versions: dict[str, Any] = {}
    for module_name in ("torch", "torch_geometric", "ogb", "xgboost", "numpy", "rdkit"):
        try:
            module = importlib.import_module(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001 - provenance, never fatal
            versions[module_name] = f"MISSING ({type(exc).__name__})"
    payload: dict[str, Any] = {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": versions,
        "cpu_count": os.cpu_count(),
        "env": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "RESEARCH_TRACK_ROOT",
                "RESEARCH_EXTRA_RUNNERS",
                "RR_RUN_ID",
                "SLURM_JOB_ID",
            )
            if os.environ.get(key) is not None
        },
    }
    try:
        import torch

        payload["torch"] = {
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
            "device_count": int(torch.cuda.device_count()),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
        if hasattr(torch.version, "git_version"):
            payload["torch"]["git_version"] = torch.version.git_version
    except Exception as exc:  # noqa: BLE001
        payload["torch"] = {"error": repr(exc)}
    try:
        import xgboost

        payload["xgboost"] = {
            "version": xgboost.__version__,
            "compiled_with_cuda": bool(xgboost.build_info().get("USE_CUDA")),
        }
    except Exception as exc:  # noqa: BLE001
        payload["xgboost"] = {"error": repr(exc)}
    return payload


def git_commit() -> str | None:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def git_dirty() -> bool | None:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(out.stdout.strip())
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# resource ledger
# ---------------------------------------------------------------------------


@dataclass
class Ledger:
    """Append-only wall/CPU time ledger (JSONL) for one worker process."""

    path: Path
    entries: list[dict[str, Any]] = field(default_factory=list)

    def record(self, stage: str, seconds: float, *, cpu_seconds: float = 0.0, **extra: Any) -> dict[str, Any]:
        entry = {
            "stage": stage,
            "seconds": float(seconds),
            "cpu_seconds": float(cpu_seconds),
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **extra,
        }
        self.entries.append(entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(jsonable(entry), ensure_ascii=False) + "\n")
        return entry

    def summary(self) -> dict[str, Any]:
        total = sum(entry["seconds"] for entry in self.entries)
        cpu = sum(entry["cpu_seconds"] for entry in self.entries)
        by_stage: dict[str, float] = {}
        for entry in self.entries:
            by_stage[entry["stage"]] = by_stage.get(entry["stage"], 0.0) + entry["seconds"]
        return {
            "wall_seconds_total": total,
            "cpu_seconds_total": cpu,
            "wall_seconds_by_stage": by_stage,
            "n_entries": len(self.entries),
        }


class stage_timer:
    """Context manager recording one stage into a :class:`Ledger`."""

    def __init__(self, ledger: Ledger, stage: str, **extra: Any) -> None:
        self.ledger = ledger
        self.stage = stage
        self.extra = extra
        self.t0 = 0.0
        self.c0 = 0.0

    def __enter__(self) -> "stage_timer":
        self.t0 = time.perf_counter()
        self.c0 = time.process_time()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.ledger.record(
            self.stage,
            time.perf_counter() - self.t0,
            cpu_seconds=time.process_time() - self.c0,
            **self.extra,
        )
        return False


# ---------------------------------------------------------------------------
# small numerical helpers
# ---------------------------------------------------------------------------


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    diff = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.sqrt(np.mean(diff * diff)))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    diff = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.mean(np.abs(diff)))


def train_target_scaler(labels: np.ndarray) -> dict[str, float]:
    """Mean and population std (ddof=0) fitted on Train labels only."""
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    mean = float(y.mean())
    std = float(y.std(ddof=0))
    if not np.isfinite(mean) or not np.isfinite(std):
        raise ValueError("non-finite target statistics")
    if std <= 0.0:
        raise ValueError("Train target std is 0: data error, refusing to continue")
    return {"mean": mean, "std": std, "n": int(y.size), "min": float(y.min()), "max": float(y.max())}


def ordered_index_hash(indices: Iterable[int]) -> str:
    arr = np.asarray(list(indices), dtype=np.int64)
    return sha256_array(arr)


def ensure_dir(path: Path | str) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target
