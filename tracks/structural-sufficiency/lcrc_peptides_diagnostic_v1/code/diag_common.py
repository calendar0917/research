"""Shared constants and IO helpers for ``lcrc-peptides-diagnostic-v1``.

Every path here is fixed at import time.  The historical Fast v1 round is
read-only: its code is imported (never edited) and its artifacts are copied
under ``history/`` with SHA-256 verification.
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
from typing import Any, Iterable

import numpy as np

ROUND_ID = "lcrc-peptides-diagnostic-v1"
PROTOCOL_ID = ROUND_ID
RESULT_LABEL = "EXPLORATORY_VALIDATION_ONLY_DIAGNOSTIC"

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
SKFP_DATA = DATA / "lrgb_skfp"
PEPTIDES_CSV = SKFP_DATA / "LRGB_Peptides-func/peptides_func.csv"
PEPTIDES_SPLITS = SKFP_DATA / "LRGB_Peptides-func/lrgb_splits_peptides_func.json"

ANALYSIS = ROUND / "analysis"
RESULTS = ROUND / "results"
CONFIGS = ROUND / "configs"
HISTORY = ROUND / "history"
RUNS = ROUND / "runs"
TESTS = ROUND / "tests"
PROTOCOL_JSON = CONFIGS / "protocol.json"

DATASET = "peptides-func"
SPLITS = ("train", "valid", "test")
SPLIT_SIZES = {"train": 10873, "valid": 2331, "test": 2331}
LEGACY_SEEDS = (0, 1, 2)
N_LABELS = 10

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

#: The exact files that must never be opened by a Validation-only code path.
TEST_ARTIFACT_NAMES = (
    "peptides_func_test.pt",
    "peptides_func_y_test.npy",
    "test_predictions.npz",
)

for _p in (str(PREV_CODE),):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def track_rel(path: Path | str) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def log(message: str) -> None:
    print(f"[{utcnow()}] {message}", flush=True)


# ---------------------------------------------------------------------------
# Test-access guard
# ---------------------------------------------------------------------------


class TestAccessBlocked(RuntimeError):
    pass


def install_test_access_guard(
    forbidden: Iterable[str] = TEST_ARTIFACT_NAMES, *, strict: bool = True
) -> dict[str, Any]:
    """Refuse to open any Test artifact, and record every attempted path.

    ``sys.addaudithook`` is process-wide and cannot be removed, so this file
    must only be imported by Validation-only entry points.
    """

    names = tuple(forbidden)
    state: dict[str, Any] = {"forbidden": list(names), "attempts": [], "installed_utc": utcnow()}

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
    return state


def environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": None,
        "torch_cuda": None,
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
        env["torch_error"] = str(exc)
    return env


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def git_tracked_dirty() -> bool | None:
    """True only when tracked files differ from HEAD (untracked run outputs excluded)."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True, check=True
        )
        return any(line.strip() and not line.startswith("??") for line in out.stdout.splitlines())
    except Exception:  # noqa: BLE001
        return None


def git_dirty() -> bool | None:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True, check=True
        )
        return bool(out.stdout.strip())
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Frozen historical artifacts
# ---------------------------------------------------------------------------


def history_dir(seed: int) -> Path:
    return HISTORY / f"peptide_seed{int(seed)}"


def history_curve_path(seed: int) -> Path:
    return history_dir(seed) / "curve.csv"


def history_checkpoint_path(seed: int) -> Path:
    return history_dir(seed) / "checkpoint.pt"


def history_run(seed: int) -> dict[str, Any]:
    return read_json(history_dir(seed) / "run.json")


__all__ = [
    "ANALYSIS",
    "CODE_DIR",
    "CONFIGS",
    "DATASET",
    "DATA",
    "HISTORY",
    "LABELS",
    "LEGACY_SEEDS",
    "LRGB_ROOT",
    "N_LABELS",
    "PEPTIDES_CSV",
    "PEPTIDES_SPLITS",
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
    "SKFP_DATA",
    "SPLITS",
    "SPLIT_SIZES",
    "TEST_ARTIFACT_NAMES",
    "TestAccessBlocked",
    "environment",
    "git_commit",
    "git_dirty",
    "git_tracked_dirty",
    "history_checkpoint_path",
    "history_curve_path",
    "history_dir",
    "history_run",
    "install_test_access_guard",
    "log",
    "read_json",
    "sha256_file",
    "sha256_json",
    "sha256_text",
    "track_rel",
    "utcnow",
    "write_json",
]
