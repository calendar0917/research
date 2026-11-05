"""Frozen constants and provenance for the edit-context screen.

Round: ``lcrc-edit-context-screen-v1``
Result label: ``EXPLORATORY_EXPOSED_DATA``

This round re-uses the frozen matched molecular pairs, component split,
per-pair traces and audit label event of ``lcrc-local-edit-error-diagnosis-v1``.
No model is trained and no historical file is modified: everything below is
read-only input plus new outputs in this round's ``analysis/`` directory.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROUND_ID = "lcrc-edit-context-screen-v1"
PROTOCOL_ID = ROUND_ID
RESULT_LABEL = "EXPLORATORY_EXPOSED_DATA"

CODE_DIR = Path(__file__).resolve().parent
ROUND = CODE_DIR.parent
TRACK = ROUND.parent
REPO = TRACK.parents[1]

HIST = REPO / "tracks/structural-sufficiency/lcrc_local_edit_error_diagnosis_v1"
HIST_CODE = HIST / "code"
#: The frozen repeat-122 Lipophilicity loader used by the historical round.
DISSECT = REPO / "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1"
DISSECT_CODE = DISSECT / "code"
HIST_ANALYSIS = HIST / "analysis"
PAIRS_CSV = HIST_ANALYSIS / "pairs.csv"
COMPONENT_SPLIT = HIST_ANALYSIS / "component_split.json"
TRACES_CSV = HIST_ANALYSIS / "pair_traces.csv"
AUDIT_SCORES = HIST_ANALYSIS / "probe_audit_scores.json"
SPLIT_JSON = HIST_ANALYSIS / "component_split.json"

ANALYSIS = ROUND / "analysis"
TESTS = ROUND / "tests"

#: Only these two historical partitions may contribute evaluation samples.
PARTITIONS = ("diagnose", "audit")
#: Frozen model units whose saved predictions are used (no re-training).
MODELS = ("C1", "A0")
SEEDS = (0, 1, 2)
MODEL_UNITS = tuple(f"{m}:s{s}" for m in MODELS for s in SEEDS)
MAIN_MODEL = "C1"
REFERENCE_MODEL = "A0"

#: Minimum structural support (fixed before any label is read).
SUPPORT_GATES = {"min_families": 5, "min_components": 15, "min_comparisons": 20}

#: Descriptive stratification threshold (raw logD), fixed, not tuned.
CONTEXT_EFFECT_THRESHOLD = 0.2

PERMUTATION_REPLICATES = 1000
BOOTSTRAP_REPLICATES = 2000
RANDOM_SEED = 20261105

#: Pre-registered resource decision (not a significance claim).
DECISION = {
    "min_positive_main_seeds": 2,
    "min_mean_context_gain": 0.05,
    "must_beat_permutation_median": True,
    "no_single_family_flip": True,
    "labels": ("PILOT_PROMISING", "PILOT_NOT_ESTABLISHED", "INSUFFICIENT_CONTEXT_SUPPORT"),
}

#: Frozen label source: the historical round's own repeat-122 frozen loader.
LABEL_REPEAT = 122
LABEL_SCOPE = "train_valid"
LABEL_SOURCE = "lcrc_lipo_model_dissection_v1/code/dissect_source.load_lipop(122, labels='train_valid')"

REQUIRED_COLUMNS = ("pair_id", "row_a", "row_b", "class_id", "core_class")
TRACE_COLUMNS = (
    "pair_id",
    "partition",
    "model",
    "seed",
    "row_a",
    "row_b",
    "class_id",
    "core_class",
    "pred_a_raw",
    "pred_b_raw",
    "d_pred_raw",
)


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_default(obj: Any):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def write_json(path: Path | str, obj: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    tmp.replace(p)
    return p


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def track_rel(path: Path | str) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def log(msg: str) -> None:
    print(f"[{utcnow()}] {msg}", flush=True)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # pragma: no cover
        return ""


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except Exception:  # pragma: no cover
        return True


def environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "hostname": platform.node(),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
    }


def input_manifest() -> dict[str, Any]:
    files = {
        "pairs": PAIRS_CSV,
        "component_split": COMPONENT_SPLIT,
        "pair_traces": TRACES_CSV,
        "probe_audit_scores": AUDIT_SCORES,
    }
    return {
        name: {"path": track_rel(p), "sha256": sha256_file(p)}
        for name, p in files.items()
    }


def index_hash(rows: Iterable[int]) -> str:
    arr = np.asarray(list(rows), dtype=np.int64)
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


__all__ = [
    "ANALYSIS",
    "AUDIT_SCORES",
    "DISSECT",
    "DISSECT_CODE",
    "BOOTSTRAP_REPLICATES",
    "CODE_DIR",
    "COMPONENT_SPLIT",
    "CONTEXT_EFFECT_THRESHOLD",
    "DECISION",
    "HIST",
    "HIST_CODE",
    "LABEL_REPEAT",
    "LABEL_SCOPE",
    "LABEL_SOURCE",
    "MAIN_MODEL",
    "MODEL_UNITS",
    "MODELS",
    "PAIRS_CSV",
    "PARTITIONS",
    "PERMUTATION_REPLICATES",
    "PROTOCOL_ID",
    "RANDOM_SEED",
    "REFERENCE_MODEL",
    "REPO",
    "RESULT_LABEL",
    "ROUND",
    "ROUND_ID",
    "SEEDS",
    "SUPPORT_GATES",
    "TRACES_CSV",
    "environment",
    "git_commit",
    "index_hash",
    "input_manifest",
    "log",
    "read_json",
    "sha256_file",
    "track_rel",
    "utcnow",
    "write_json",
]
