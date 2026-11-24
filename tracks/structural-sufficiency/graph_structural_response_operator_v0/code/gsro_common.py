"""Shared paths, configuration, hashing and IO for GSRO v0.

Nothing scientific lives here: only identity (hashes), IO and bookkeeping.
Every path is resolved from this file's location so the module works both in
the local checkout and in the compute checkout on ``res``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

ROUND_ID = "graph-structural-response-operator-v0"
PROTOCOL_ID = "gsro-v0-lipo-s0"

CODE_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = CODE_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
CONFIG_PATH = EXPERIMENT_DIR / "configs" / "gsro_v0.json"

#: Reused committed label-free raw-graph cache of the structural-query-learning
#: track (official OGB rows, official scaffold split, physical bonds).
SQ_DATA_DIR = REPO_ROOT / "tracks" / "structural-query-learning" / "data"
SQ_CODE_DIR = REPO_ROOT / "tracks" / "structural-query-learning" / "code"


def results_dir() -> Path:
    """Durable artefact root; ``GSRO_RESULTS_DIR`` allows a scratch destination."""
    override = os.environ.get("GSRO_RESULTS_DIR", "").strip()
    if override:
        base = Path(override).expanduser()
        if not base.is_absolute():
            base = REPO_ROOT / base
        return base
    return EXPERIMENT_DIR / "results"


def runs_dir() -> Path:
    """Per-arm run store; ``GSRO_RUNS_DIR`` allows a scratch destination."""
    override = os.environ.get("GSRO_RUNS_DIR", "").strip()
    if override:
        base = Path(override).expanduser()
        if not base.is_absolute():
            base = REPO_ROOT / base
        return base
    return EXPERIMENT_DIR / "runs"


def load_config() -> dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# hashing / IO
# ---------------------------------------------------------------------------


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_array(array) -> str:
    import numpy as np

    arr = np.ascontiguousarray(array)
    return sha256_bytes(arr.tobytes())


def write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
    return path


def _json_default(obj):
    import numpy as np
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def set_seed(seed: int) -> None:
    """Fix every RNG this round uses (python, numpy, torch, cuda)."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def stable_hash(*parts: object) -> str:
    text = "|".join(str(p) for p in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def import_sq_data():
    """Import the reused loader of the structural-query-learning track."""
    import sys

    if str(SQ_CODE_DIR) not in sys.path:
        sys.path.insert(0, str(SQ_CODE_DIR))
    import sq_data  # noqa: WPS433 (reused upstream module)

    return sq_data


def git_commit() -> str:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:  # pragma: no cover - best effort provenance
        return "unknown"


def environment() -> dict[str, Any]:
    import platform

    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "git_commit": git_commit(),
    }


def config_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "config_path": str(CONFIG_PATH.relative_to(REPO_ROOT)),
        "config_sha256": sha256_file(CONFIG_PATH),
        "config": dict(config),
    }
