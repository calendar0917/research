"""Identity, paths, IO and bookkeeping for ``lcrc-readout-context-pilot-small-v1``.

Nothing scientific lives here.  Every frozen scientific constant (recipe,
readout form, parameter budget, context fields, diagnostic tolerances) is in
:mod:`cr_models` / :mod:`cr_train` / :mod:`cr_analysis` and is hashed into
``analysis/protocol_freeze.json`` at prepare time.

Path discipline: every path is resolved from this file's location, so the module
behaves identically in the local checkout and in the read-only compute checkout
on ``res``.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROUND_ID = "lcrc-readout-context-pilot-small-v1"
PROTOCOL_ID = ROUND_ID
STUDY_ID = ROUND_ID

CODE_DIR = Path(__file__).resolve().parent
ROUND_DIR = CODE_DIR.parent
TRACK_DIR = ROUND_DIR.parent                       # tracks/structural-sufficiency
REPO_ROOT = TRACK_DIR.parents[1]                   # repository root

RESULTS_DIR = ROUND_DIR / "results"
ANALYSIS_DIR = ROUND_DIR / "analysis"

#: Read-only reuse roots (never written by this round).
SQL_DIR = REPO_ROOT / "tracks/structural-query-learning"
SQL_CODE = SQL_DIR / "code"
SQL_DATA = SQL_DIR / "data"
SQL_RESULTS = SQL_DIR / "results/sq-pilot-small-v1"
PREP_DIR = REPO_ROOT / "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1"
PREP_CODE = PREP_DIR / "code"
DISSECT_DIR = REPO_ROOT / "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1"
DISSECT_CODE = DISSECT_DIR / "code"

#: The two authorised tasks, in pipeline order.
DATASETS: tuple[str, ...] = ("ogbg-molesol", "ogbg-mollipo")
SPLITS: tuple[str, ...] = ("train", "valid", "test")

EXPECTED_SPLIT_SIZES = {
    "ogbg-molesol": {"train": 902, "valid": 113, "test": 113},
    "ogbg-mollipo": {"train": 3360, "valid": 420, "test": 420},
}
EXPECTED_TOTAL_ROWS = {"ogbg-molesol": 1128, "ogbg-mollipo": 4200}

#: The four frozen caches this round consumes (verified by real file bytes).
EXPECTED_CACHE_SHA256 = {
    "ogbg_molesol_sq-pilot-cache-v1.npz": "3b7a8e16ca9c7ab240add84fbe26050dc81e22359bcc36f1c8a8111b77aec24a",
    "ogbg_mollipo_sq-pilot-cache-v1.npz": "fe002287fa9c581278bc5430f5b4a5b26b35d9d4d5207d138f15956d93123c7c",
    "ogbg_molesol_sq-pilot-lipo-raw-v1.npz": "a3ab9f7e22798a8f6fdc7f0913b1136b7cf59d4ffc7168faecc7f8b3094710b1",
    "ogbg_mollipo_sq-pilot-lipo-raw-v1.npz": "3005aa55e766d83a237b1659ad2cabb51b8702eef6d82c203fdecf9cb3c42cc3",
}

#: Single training seed for the whole round; no multi-seed stability claim.
TRAIN_SEED = 0
BOOTSTRAP_SEED = 1703
#: Isolated initialisation substreams for the new readouts (module init only;
#: they are *not* extra training seeds).
INIT_STREAM_GLOBAL = 1001
INIT_STREAM_CONTEXT = 1002
INIT_STREAM_LOCAL = 1003
#: Sub-seed used to draw the canonical C1 initial state.
INIT_SEED_CANONICAL = 0

ARMS: tuple[str, ...] = ("A", "B", "C")

#: The two frozen, named context fields of arm C (K = 2, fixed in advance).
CONTEXT_FIELDS: tuple[str, ...] = ("log1p_n_atoms", "ring_atom_fraction")
CONTEXT_QUANTILES: tuple[float, float] = (25.0, 75.0)
#: A raw global column is "active" when its Train span exceeds this threshold.
ACTIVE_SPAN_EPS = 1e-8

SPLIT_FILE = {  # local name -> payload key
    "train": "train_idx",
    "valid": "valid_idx",
    "test": "test_idx",
}


# ---------------------------------------------------------------------------
# hashing
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


def sha256_state(state: Mapping[str, Any], *, keys: Iterable[str] | None = None) -> str:
    """Order-independent fingerprint of a tensor state dict (float32 bytes)."""
    import torch

    digest = hashlib.sha256()
    wanted = sorted(state.keys()) if keys is None else sorted(keys)
    for key in wanted:
        value = state[key]
        tensor = value.detach().to("cpu") if hasattr(value, "detach") else torch.as_tensor(value)
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(np.ascontiguousarray(tensor.numpy(), dtype=np.float32).tobytes())
    return digest.hexdigest()


def rng_state_hash(*, include_cuda: bool = True) -> str:
    """Fingerprint of the live training RNG streams (process start / resume)."""
    import torch

    parts = [sha256_text(json.dumps(random.getstate()[1][:8]))]
    parts.append(sha256_bytes(np.random.get_state()[1].tobytes()))
    parts.append(sha256_bytes(torch.get_rng_state().numpy().tobytes()))
    payload = {"python": parts[0], "numpy": parts[1], "torch_cpu": parts[2]}
    if include_cuda and torch.cuda.is_available():
        payload["torch_cuda"] = sha256_bytes(torch.cuda.get_rng_state().cpu().numpy().tobytes())
    return sha256_json(payload)


# ---------------------------------------------------------------------------
# JSON / CSV IO
# ---------------------------------------------------------------------------


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
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
    return value


def dumps_json(payload: Any) -> str:
    return json.dumps(jsonable(payload), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_json(payload: Any) -> str:
    return sha256_text(dumps_json(payload))


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_write_text(path: Path | str, text: str) -> Path:
    """Write via a sibling temp file then ``os.replace`` (never a torn file)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, target)
    return target


def atomic_write_json(path: Path | str, payload: Any) -> Path:
    return atomic_write_text(
        path, json.dumps(jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )


def atomic_write_npz(path: Path | str, arrays: Mapping[str, np.ndarray]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp-{os.getpid()}.npz")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **{str(k): np.asarray(v) for k, v in arrays.items()})
    os.replace(tmp, target)
    return target


def atomic_torch_save(path: Path | str, payload: Any) -> Path:
    import torch

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    with tmp.open("wb") as handle:
        torch.save(payload, handle)
    os.replace(tmp, target)
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


def read_csv(path: Path | str) -> list[dict[str, str]]:
    import csv

    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def ensure_dir(path: Path | str) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


# ---------------------------------------------------------------------------
# seeding
# ---------------------------------------------------------------------------


def set_global_seeds(seed: int = TRAIN_SEED) -> dict[str, Any]:
    """Set every global RNG this round uses, in one documented place."""
    import torch

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    return {"seed": int(seed), "rng_state_hash_after_set": rng_state_hash()}


def capture_rng_state() -> dict[str, Any]:
    import torch

    payload = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": torch.cuda.get_rng_state().clone() if torch.cuda.is_available() else None,
    }
    return payload


def restore_rng_state(payload: Mapping[str, Any]) -> None:
    import torch

    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch_cpu"].cpu())
    if payload.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(payload["torch_cuda"].cpu())


# ---------------------------------------------------------------------------
# environment / provenance
# ---------------------------------------------------------------------------


def _cmd(args: Sequence[str], cwd: Path | str | None = None) -> str | None:
    try:
        out = subprocess.run(args, capture_output=True, text=True, check=True, cwd=cwd)
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 - provenance only
        return None


def git_commit() -> str | None:
    return _cmd(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])


def git_branch() -> str | None:
    return _cmd(["git", "-C", str(REPO_ROOT), "rev-parse", "--abbrev-ref", "HEAD"])


def git_dirty_tracked() -> bool | None:
    out = _cmd(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"])
    return None if out is None else bool(out)


def lockfile_sha256() -> str | None:
    path = REPO_ROOT / "uv.lock"
    return sha256_file(path) if path.is_file() else None


def environment_snapshot() -> dict[str, Any]:
    import importlib

    versions: dict[str, Any] = {}
    for name in ("torch", "torch_geometric", "numpy", "scipy", "rdkit", "ogb", "matplotlib", "xgboost"):
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
            for key in ("CUDA_VISIBLE_DEVICES", "RESEARCH_TRACK_ROOT", "RESEARCH_EXTRA_RUNNERS", "RR_RUN_ID")
            if os.environ.get(key) is not None
        },
        "git_commit": git_commit(),
        "git_branch": git_branch(),
        "git_dirty_tracked": git_dirty_tracked(),
        "uv_lock_sha256": lockfile_sha256(),
    }
    try:
        import torch

        payload["torch"] = {
            "cuda_available": bool(torch.cuda.is_available()),
            "torch_cuda": getattr(torch.version, "cuda", None),
            "device_count": int(torch.cuda.device_count()),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "device_uuid": [
                str(torch.cuda.get_device_properties(i).uuid) if hasattr(torch.cuda.get_device_properties(i), "uuid") else None
                for i in range(torch.cuda.device_count())
            ],
            "cudnn_version": int(torch.backends.cudnn.version() or 0),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        }
    except Exception as exc:  # noqa: BLE001
        payload["torch"] = {"error": repr(exc)}
    payload["nvidia_smi"] = _cmd(
        ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,uuid", "--format=csv,noheader"]
    )
    return payload


# ---------------------------------------------------------------------------
# resource ledger
# ---------------------------------------------------------------------------


@dataclass
class Ledger:
    """Append-only wall/CPU time ledger (JSONL) written by one worker process."""

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
        by_stage: dict[str, float] = {}
        by_device: dict[str, float] = {}
        for entry in self.entries:
            by_stage[entry["stage"]] = by_stage.get(entry["stage"], 0.0) + float(entry["seconds"])
            device = str(entry.get("device", "cpu"))
            by_device[device] = by_device.get(device, 0.0) + float(entry["seconds"])
        return {
            "wall_seconds_total": float(sum(e["seconds"] for e in self.entries)),
            "cpu_seconds_total": float(sum(e["cpu_seconds"] for e in self.entries)),
            "wall_seconds_by_stage": by_stage,
            "wall_seconds_by_device": by_device,
            "gpu_hours_by_device": {k: v / 3600.0 for k, v in by_device.items() if k.startswith("cuda")},
            "n_entries": len(self.entries),
        }


class stage_timer:
    def __init__(self, ledger: Ledger | None, stage: str, **extra: Any) -> None:
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
        if self.ledger is not None:
            self.ledger.record(
                self.stage,
                time.perf_counter() - self.t0,
                cpu_seconds=time.process_time() - self.c0,
                **self.extra,
            )
        return False


# ---------------------------------------------------------------------------
# metrics / tolerances
# ---------------------------------------------------------------------------


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    diff = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.sqrt(np.mean(diff * diff)))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    diff = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.mean(np.abs(diff)))


def trace_tolerance(value: float | np.ndarray) -> np.ndarray | float:
    """Frozen traceback tolerance: ``1e-5 + 1e-6 * max(1, |x|)``."""
    arr = np.abs(np.asarray(value, dtype=np.float64))
    tol = 1e-5 + 1e-6 * np.maximum(1.0, arr)
    return float(tol) if np.isscalar(value) else tol


def target_scaler(labels: np.ndarray) -> dict[str, float]:
    """Train-only target mean and population std (ddof=0)."""
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    mean = float(y.mean())
    std = float(y.std(ddof=0))
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0.0:
        raise ValueError("invalid Train target statistics")
    return {
        "mean": mean,
        "std": std,
        "n": int(y.size),
        "min": float(y.min()),
        "max": float(y.max()),
        "ddof": 0,
        "fit_split": "train",
    }
