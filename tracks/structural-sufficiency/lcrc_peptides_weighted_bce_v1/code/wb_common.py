"""Shared constants, frozen references and IO for ``lcrc-peptides-weighted-bce-v1``.

This round reuses the historical Fast v1 Peptides-func objects (feature caches,
``bp_data`` / ``bp_model`` / ``bp_train``) and the read-only Diagnostic v1 run.
Nothing under either historical track is written.

Importing this module installs the Validation-only audit hook: any attempt to
open a historical Test artifact raises ``TestAccessBlocked``.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROUND_ID = "lcrc-peptides-weighted-bce-v1"
PROTOCOL_ID = ROUND_ID
RESULT_LABEL = "EXPLORATORY_VALIDATION_ONLY_DIAGNOSTIC"

CODE_DIR = Path(__file__).resolve().parent
ROUND = CODE_DIR.parent
TRACK = ROUND.parent
REPO = TRACK.parents[1]

DIAG = REPO / "tracks/structural-sufficiency/lcrc_peptides_diagnostic_v1"
DIAG_CODE = DIAG / "code"
FASTV1 = REPO / "tracks/structural-sufficiency/lcrc_benchmark_positioning_fast_v1"

if str(DIAG_CODE) not in sys.path:
    sys.path.insert(0, str(DIAG_CODE))

import diag_common as H  # noqa: E402  (adds the Fast v1 code directory to sys.path)

DATASET = H.DATASET
LABELS = H.LABELS
N_LABELS = H.N_LABELS
SPLIT_SIZES = H.SPLIT_SIZES

ANALYSIS = ROUND / "analysis"
RESULTS = ROUND / "results"
CONFIGS = ROUND / "configs"
RUNS = ROUND / "runs"
TESTS = ROUND / "tests"
PROTOCOL_JSON = CONFIGS / "protocol.json"
POS_WEIGHT_JSON = ANALYSIS / "pos_weight.json"

TRAIN_LABELS = H.PREV_ANALYSIS / "peptides_func_y_train.npy"
VALID_LABELS = H.PREV_ANALYSIS / "peptides_func_y_valid.npy"

#: Frozen historical reference run (ordinary BCE, seed 0, 240 epochs).
HISTORY_RUN_DIR = DIAG / "runs/peptide_c1_seed0_240ep"
HISTORY_RUN_JSON = HISTORY_RUN_DIR / "run.json"
HISTORY_CURVE = HISTORY_RUN_DIR / "curve.csv"
HISTORY_CHECKPOINT = HISTORY_RUN_DIR / "checkpoint.pt"
#: Fast v1 seed0 checkpoint, source of the frozen initialisation hash.
FASTV1_SEED0_CHECKPOINT = H.history_checkpoint_path(0)
#: Fixed background reference (not a single-variable control).
ECFP_SUMMARY = DIAG / "results/ecfp_validation.csv"
LABEL_STATISTICS = DIAG / "analysis/label_statistics.csv"

FORMAL_RUN_DIR = RUNS / "weighted_bce_seed0"

TEST_ARTIFACT_TOKENS = tuple(H.TEST_ARTIFACT_NAMES)
GUARD = H.install_test_access_guard()


# ---------------------------------------------------------------------------
# IO / provenance
# ---------------------------------------------------------------------------


def sha256_file(path: Path | str) -> str:
    return H.sha256_file(path)


def sha256_text(text: str) -> str:
    return H.sha256_text(text)


def track_rel(path: Path | str) -> str:
    return H.track_rel(path)


def read_json(path: Path | str) -> Any:
    return H.read_json(path)


def write_json(path: Path | str, payload: Any) -> None:
    H.write_json(path, payload)


def utcnow() -> str:
    return H.utcnow()


def log(message: str) -> None:
    H.log(message)


def environment() -> dict[str, Any]:
    return H.environment()


def git_commit() -> str | None:
    return H.git_commit()


def git_dirty() -> bool | None:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True, check=True
        )
        return bool(out.stdout.strip())
    except Exception:  # noqa: BLE001
        return None


def platform_summary() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": platform.node(),
    }


def data_sha256(values: np.ndarray) -> str:
    """Canonical SHA-256 over a C-contiguous float64 array (for label equality)."""
    arr = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(arr.tobytes()).hexdigest()


__all__ = [
    "ANALYSIS",
    "CONFIGS",
    "CODE_DIR",
    "DATASET",
    "DIAG",
    "DIAG_CODE",
    "ECFP_SUMMARY",
    "FASTV1",
    "FASTV1_SEED0_CHECKPOINT",
    "FORMAL_RUN_DIR",
    "GUARD",
    "H",
    "HISTORY_CHECKPOINT",
    "HISTORY_CURVE",
    "HISTORY_RUN_DIR",
    "HISTORY_RUN_JSON",
    "LABELS",
    "LABEL_STATISTICS",
    "N_LABELS",
    "POS_WEIGHT_JSON",
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
    "TRAIN_LABELS",
    "VALID_LABELS",
    "data_sha256",
    "environment",
    "git_commit",
    "git_dirty",
    "log",
    "platform_summary",
    "read_json",
    "sha256_file",
    "sha256_text",
    "track_rel",
    "utcnow",
    "write_json",
]
