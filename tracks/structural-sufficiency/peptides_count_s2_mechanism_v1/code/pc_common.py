"""Shared constants, IO and the Test-access guard for the S2 count-mechanism round.

Round: ``peptides-count-s2-mechanism-v1``
Result label: ``EXPLORATORY_VALIDATION_ONLY_MECHANISM``

This module is deliberately torch-free so the isolated CPU fingerprint
environment can import it.  The historical Fast v1 code is imported read-only.
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
from typing import Any, Iterable, Mapping

import numpy as np

ROUND_ID = "peptides-count-s2-mechanism-v1"
PROTOCOL_ID = ROUND_ID
RESULT_LABEL = "EXPLORATORY_VALIDATION_ONLY_MECHANISM"

CODE_DIR = Path(__file__).resolve().parent
ROUND = CODE_DIR.parent
TRACK = ROUND.parent
REPO = TRACK.parents[1]
PREV = REPO / "tracks/structural-sufficiency/lcrc_benchmark_positioning_fast_v1"
PREV_CODE = PREV / "code"
PREV_CACHE = PREV / "cache"
PREV_ANALYSIS = PREV / "analysis"

DATA = REPO / "data"
LRGB_ROOT = DATA / "LRGB"

ANALYSIS = ROUND / "analysis"
RESULTS = ROUND / "results"
CONFIGS = ROUND / "configs"
RUNS = ROUND / "runs"
TESTS = ROUND / "tests"
PROTOCOL_JSON = CONFIGS / "protocol.json"

DATASET = "peptides-func"
SPLIT_SIZES = {"train": 10873, "valid": 2331}
N_LABELS = 10
N_BUCKETS = 2048
HASH_PERSON = b"pcs2v1"

LABELS = (
    "antifungal",
    "cell_cell_communication",
    "anticancer",
    "drug_delivery_vehicle",
    "antimicrobial",
    "antiviral",
    "antihypertensive",
    "antibacterial",
    "antiparasitic",
    "toxic",
)

#: Exact raw Peptides-func discrete column vocab sizes (atoms: 9 columns,
#: bonds: 3 columns).  These are the legal categorical ranges of the official
#: PyG ``x`` / ``edge_attr`` tensors, not fitted parameters.
ATOM_VOCABS = (119, 4, 12, 12, 6, 6, 6, 2, 2)
BOND_VOCABS = (5, 6, 2)

#: The only split names this round may materialise.
ALLOWED_SPLITS = ("train", "valid")

#: Fail-closed Test protection.  These substrings cover every Peptides-func
#: Test artifact used by this repository and the upstream LRGB CSV/split
#: mirrors.  The audit hook records every attempted open before raising.
TEST_FORBIDDEN = (
    "peptides_func_test",
    "peptides_func_y_test",
    "test_predictions.npz",
    "peptides-func/processed/test.pt",
    "peptides-func/raw/test.pt",
    "LRGB_Peptides-func",
    "lrgb_splits_peptides_func.json",
)

if str(PREV_CODE) not in sys.path:
    sys.path.insert(0, str(PREV_CODE))

_GUARD_STATE: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(arr).tobytes())


def sha256_json(payload: Any) -> str:
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def read_json(path: Path | str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path | str, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def track_rel(path: Path | str) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def log(message: str) -> None:
    print(f"[{utcnow()}] {message}", flush=True)


# ---------------------------------------------------------------------------
# Deterministic content hash for S2 patterns
# ---------------------------------------------------------------------------


def stable_digest_int(key: bytes) -> int:
    """Fixed, process-independent 64-bit digest of one canonical key."""
    return int.from_bytes(hashlib.blake2b(key, digest_size=8, person=HASH_PERSON).digest(), "little")


def stable_bucket(key: bytes) -> int:
    return stable_digest_int(key) % N_BUCKETS


# ---------------------------------------------------------------------------
# Test-access guard
# ---------------------------------------------------------------------------


class TestAccessBlocked(RuntimeError):
    pass


def install_test_access_guard(
    forbidden: Iterable[str] = TEST_FORBIDDEN, *, strict: bool = True
) -> dict[str, Any]:
    """Refuse to open Test data/labels and record every attempted path.

    ``sys.addaudithook`` is process-wide and cannot be removed.  Entry points
    that must never touch Test install this before importing data loaders.
    """
    global _GUARD_STATE
    if _GUARD_STATE is not None:
        return _GUARD_STATE
    names = tuple(forbidden)
    state: dict[str, Any] = {
        "forbidden": list(names),
        "attempts": [],
        "installed_utc": utcnow(),
        "strict": bool(strict),
    }

    def _hook(event: str, args: tuple[Any, ...]) -> None:
        if event != "open" or not args:
            return
        target = args[0]
        if not isinstance(target, (str, bytes, os.PathLike)):
            return
        text = os.fsdecode(target)
        for name in names:
            if name in text:
                state["attempts"].append(text)
                if strict:
                    raise TestAccessBlocked(f"TEST_ACCESS_BLOCKED: {text}")

    sys.addaudithook(_hook)
    _GUARD_STATE = state
    return state


# ---------------------------------------------------------------------------
# Environment / Git provenance
# ---------------------------------------------------------------------------


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def git_status_porcelain() -> str | None:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True, check=True
        )
        return out.stdout
    except Exception:  # noqa: BLE001
        return None


def git_dirty() -> bool | None:
    status = git_status_porcelain()
    return None if status is None else bool(status.strip())


def git_tracked_dirty() -> bool | None:
    status = git_status_porcelain()
    if status is None:
        return None
    return any(line.strip() and not line.startswith("??") for line in status.splitlines())


def environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "git_tracked_dirty": git_tracked_dirty(),
        "hostname": platform.node(),
        "utc": utcnow(),
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["torch_cuda"] = torch.version.cuda
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception as exc:  # noqa: BLE001
        env["torch_error"] = f"{type(exc).__name__}: {exc}"
    return env


def distribution(values: Iterable[float]) -> dict[str, Any]:
    arr = np.asarray(list(values), dtype=np.float64)
    if not arr.size:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "sum": float(arr.sum()),
    }


__all__ = [
    "ALLOWED_SPLITS",
    "ANALYSIS",
    "ATOM_VOCABS",
    "BOND_VOCABS",
    "CONFIGS",
    "CODE_DIR",
    "DATASET",
    "DATA",
    "HASH_PERSON",
    "LABELS",
    "LRGB_ROOT",
    "N_BUCKETS",
    "N_LABELS",
    "PREV",
    "PREV_ANALYSIS",
    "PREV_CACHE",
    "PREV_CODE",
    "PROTOCOL_ID",
    "PROTOCOL_JSON",
    "REPO",
    "RESULT_LABEL",
    "RESULTS",
    "ROUND",
    "ROUND_ID",
    "RUNS",
    "SPLIT_SIZES",
    "TESTS",
    "TEST_FORBIDDEN",
    "TestAccessBlocked",
    "distribution",
    "environment",
    "git_commit",
    "git_dirty",
    "git_status_porcelain",
    "git_tracked_dirty",
    "install_test_access_guard",
    "log",
    "read_json",
    "sha256_array",
    "sha256_bytes",
    "sha256_file",
    "sha256_json",
    "sha256_text",
    "stable_bucket",
    "stable_digest_int",
    "track_rel",
    "utcnow",
    "write_json",
]
