"""Shared paths, hashing and JSON helpers for ``lcrc-failure-mechanism-p0-v1``.

This round is CPU-only and read-only with respect to every historical artefact.
Nothing in this module writes outside ``tracks/structural-sufficiency/
lcrc_failure_mechanism_p0_v1/analysis/``.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROUND_ID = "lcrc-failure-mechanism-p0-v1"

TRACK = Path(__file__).resolve().parents[1]
REPO = TRACK.parents[2]
CODE = TRACK / "code"
ANALYSIS = TRACK / "analysis"

#: frozen historical artefacts (read-only inputs)
A_ROOT = TRACK.parent / "lcrc_lipo_esa_random42_v1"
B_ROOT = TRACK.parent / "lcrc_performance_longrun_v1"
CHEMBENCH_V1 = TRACK.parent / "lcrc_molnet_benchmark_pilot_v1/analysis/source/chembench"
CHEMBENCH_V2_ROOT = TRACK.parent / "lcrc_molnet_benchmark_pilot_v2"

#: fixed analysis constants, frozen before any result was inspected
TOL = 1e-5                 # metric reproduction tolerance required by the brief
N_BOOT = 2000              # scaffold-cluster bootstrap replicates
BOOTSTRAP_SEED = 20261103  # deterministic bootstrap stream
CI_PERCENTILES = (2.5, 97.5)
LOW_SUPPORT_N = 30         # below this group size a group is LOW_SUPPORT
TOP_FRACTION = 0.10        # tail definition: top 10% by BASE absolute error
N_FOLDS = 5                # cross-fitting folds for the conditional model
RIDGE_ALPHA = 10.0         # fixed, no hyper-parameter search
RARE_DF_MAX = 4            # df in [1, 4] counts as "rare"
SIGNATURE_RADIUS = 2       # rooted radius-2 structural signature
MORGAN_RADIUS = 2
MORGAN_BITS = 2048


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_int64(values: Iterable[int]) -> str:
    array = np.ascontiguousarray(np.asarray(list(values), dtype="<i8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def sha256_float64(values: Iterable[float]) -> str:
    array = np.ascontiguousarray(np.asarray(list(values), dtype="<f8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def sha256_text(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(str(v) for v in values).encode("utf-8")).hexdigest()


def row_sha(values: Iterable[int]) -> str:
    """Repository-wide row-id hash definition (``<i8`` contiguous bytes)."""
    return sha256_int64(values)


def relpath(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(path)


def jsonable(value: Any) -> Any:
    """Recursively convert numpy / Path / non-finite values to JSON-safe values."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, np.float64, np.float32)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2, sort_keys=False, allow_nan=False)
        handle.write("\n")


def read_json(path: Path | str) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def git_state() -> dict[str, Any]:
    def run(args: list[str]) -> str:
        return subprocess.check_output(args, cwd=REPO, text=True).strip()

    try:
        commit = run(["git", "rev-parse", "HEAD"])
        short = run(["git", "rev-parse", "--short", "HEAD"])
        status = run(["git", "status", "--porcelain"])
    except Exception as exc:  # pragma: no cover - git is present in practice
        return {"commit": None, "commit_short": None, "dirty": None, "error": str(exc)}
    return {
        "commit": commit,
        "commit_short": short,
        "dirty": bool(status),
        "dirty_files": [line for line in status.splitlines() if line.strip()],
    }


def environment() -> dict[str, Any]:
    import scipy
    import sklearn

    record: dict[str, Any] = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "gpu_used": False,
        "training_or_finetuning": False,
    }
    try:
        import rdkit

        record["rdkit"] = rdkit.__version__
    except Exception:  # pragma: no cover - rdkit is present locally
        record["rdkit"] = None
    try:
        import pandas

        record["pandas"] = pandas.__version__
    except Exception:  # pragma: no cover
        record["pandas"] = None
    return record


def ensure_analysis() -> Path:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    return ANALYSIS
