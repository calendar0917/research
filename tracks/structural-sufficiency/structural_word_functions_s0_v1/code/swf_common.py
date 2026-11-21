"""Paths, identity, frozen config access, IO and numeric helpers.

Nothing scientific lives here: only hashing, IO, bookkeeping and the shared
numeric primitives (RMSE / MAE / Train-only target scaler).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROUND_DIR = CODE_DIR.parent
REPO_ROOT = ROUND_DIR.parents[2]
CONFIG_PATH = ROUND_DIR / "configs" / "swf_s0.json"

SQ_CODE = REPO_ROOT / "tracks" / "structural-query-learning" / "code"
WP_CODE = REPO_ROOT / "tracks" / "structural-sufficiency" / "word_sentence_port_binding_context_v2" / "code"
for _p in (SQ_CODE, WP_CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

ATOM_COLUMNS = 9
BOND_COLUMNS = 3
MAX_WORD_ATOMS = 4
WORD_EDGE_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
SPLITS = ("train", "valid", "test")
SPLIT_ID = {"train": 0, "valid": 1, "test": 2}
SPLIT_NAME = {0: "train", 1: "valid", 2: "test"}


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def round_dir() -> Path:
    return ROUND_DIR


def data_dir() -> Path:
    return ROUND_DIR / "data"


def analysis_dir() -> Path:
    return ROUND_DIR / "analysis"


def results_root() -> Path:
    override = os.environ.get("SWF_RESULTS_DIR", "").strip()
    if override:
        base = Path(override).expanduser()
        if not base.is_absolute():
            base = REPO_ROOT / base
        return base
    return ROUND_DIR / "results"


def run_dir(run_id: str) -> Path:
    path = results_root() / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# hashing / json / csv
# ---------------------------------------------------------------------------


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


def sha256_json(payload: Any) -> str:
    text = json.dumps(jsonable(payload), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def append_jsonl(path: Path | str, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(payload), ensure_ascii=False) + "\n")


def read_jsonl(path: Path | str) -> list[Any]:
    target = Path(path)
    if not target.is_file():
        return []
    return [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# numeric helpers
# ---------------------------------------------------------------------------


def rmse(pred, target) -> float:
    diff = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.sqrt(np.mean(diff * diff)))


def mae(pred, target) -> float:
    diff = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.mean(np.abs(diff)))


def train_target_scaler(labels: np.ndarray) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    mean = float(y.mean())
    std = float(y.std(ddof=0))
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0.0:
        raise ValueError("non-finite or degenerate Train target statistics")
    return {"mean": mean, "std": std, "n": int(y.size), "min": float(y.min()), "max": float(y.max())}


# ---------------------------------------------------------------------------
# seeds / provenance
# ---------------------------------------------------------------------------


def set_deterministic(seed: int) -> dict[str, Any]:
    import random

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    info: dict[str, Any] = {"seed": int(seed), "numpy_seed": int(seed)}
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        info["torch_manual_seed"] = int(seed)
        info["cudnn_deterministic"] = True
    except Exception as exc:  # noqa: BLE001
        info["torch_error"] = repr(exc)
    return info


def enable_tf32(enabled: bool) -> dict[str, Any]:
    info = {"requested": bool(enabled)}
    try:
        import torch

        if enabled:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        info["matmul_allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
        info["cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)
    except Exception as exc:  # noqa: BLE001
        info["error"] = repr(exc)
    return info


def environment_snapshot() -> dict[str, Any]:
    import importlib

    versions: dict[str, Any] = {}
    for name in ("torch", "torch_geometric", "ogb", "numpy", "scipy", "sklearn", "rdkit"):
        try:
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001
            versions[name] = f"MISSING ({type(exc).__name__})"
    payload: dict[str, Any] = {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": versions,
        "cpu_count": os.cpu_count(),
        "env": {
            key: os.environ.get(key)
            for key in ("CUDA_VISIBLE_DEVICES", "RR_RUN_ID", "SLURM_JOB_ID", "SWF_RESULTS_DIR")
            if os.environ.get(key) is not None
        },
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
    }
    try:
        import torch

        payload["torch"] = {
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
            "device_count": int(torch.cuda.device_count()),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "uuid": [
                str(torch.cuda.get_device_properties(i).uuid) if hasattr(torch.cuda.get_device_properties(i), "uuid") else None
                for i in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:  # noqa: BLE001
        payload["torch"] = {"error": repr(exc)}
    try:
        import torch

        if torch.cuda.is_available():
            payload["gpu_peak_vram_mb"] = float(torch.cuda.max_memory_allocated() / 2 ** 20)
    except Exception:  # noqa: BLE001
        pass
    return payload


def git_commit() -> str | None:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def git_dirty() -> bool | None:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        )
        return bool(out.stdout.strip())
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# tiny resource ledger
# ---------------------------------------------------------------------------


class Ledger:
    """Append-only wall/cpu ledger (JSONL) for one worker process."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.entries: list[dict[str, Any]] = []

    def record(self, stage: str, seconds: float, *, cpu_seconds: float = 0.0, **extra: Any) -> dict[str, Any]:
        entry = {
            "stage": stage,
            "seconds": float(seconds),
            "cpu_seconds": float(cpu_seconds),
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **extra,
        }
        self.entries.append(entry)
        append_jsonl(self.path, entry)
        return entry

    def summary(self) -> dict[str, Any]:
        by_stage: dict[str, float] = {}
        for entry in self.entries:
            by_stage[entry["stage"]] = by_stage.get(entry["stage"], 0.0) + entry["seconds"]
        return {
            "wall_seconds_total": float(sum(e["seconds"] for e in self.entries)),
            "cpu_seconds_total": float(sum(e["cpu_seconds"] for e in self.entries)),
            "wall_seconds_by_stage": by_stage,
            "n_entries": len(self.entries),
        }


class stage_timer:
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


def ensure_dir(path: Path | str) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def iter_npz(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}
