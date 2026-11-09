"""Frozen paths, identities and IO for the LCRC MolPCBA bottleneck diagnosis v1.

Everything here is read-only with respect to historical rounds.  The round label
is ``POST_HOC_EXPOSED_TEST_DIAGNOSIS``: official Test labels are used only for
descriptive statistics on already-frozen checkpoints.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

CODE = Path(__file__).resolve().parent
ROUND = CODE.parent
TRACK = ROUND.parent
REPO = TRACK.parents[1]

ANALYSIS = ROUND / "analysis"
LOCAL = ROUND / "_local"
LOCAL_VALID = LOCAL / "valid_predictions"
TESTS = ROUND / "tests"

ROUND_ID = "lcrc-pcba-bottleneck-diagnosis-v1"
RESULT_LABEL = "POST_HOC_EXPOSED_TEST_DIAGNOSIS"
TEST_ROLE = "EXPOSED_TEST_ENGINEERING_COMPARISON_SET"

DATASET = "ogbg-molpcba"
N_TASKS = 128
N_VALID_ROWS = 43_793
N_TEST_ROWS = 43_793

BS_ROUND = REPO / "tracks/structural-sufficiency/lcrc_pcba_backbone_scaling_v1"
S3_ROUND = REPO / "tracks/structural-sufficiency/lcrc_pcba_s3_v1"
S3SEL_ROUND = REPO / "tracks/structural-sufficiency/lcrc_pcba_s3_selective_v1"
RC_ROUND = REPO / "tracks/structural-sufficiency/lcrc_pcba_rc_v1"
BOC_ROUND = REPO / "tracks/structural-sufficiency/lcrc_pcba_boc_v1"
CC_ROUND = REPO / "tracks/structural-sufficiency/lcrc_pcba_continuation_recipe_v1"

OGB_ROOT = REPO / "data/ogb/ogbg_molpcba"
OGB_MAPPING = OGB_ROOT / "mapping/mol.csv.gz"
OGB_LABELS = OGB_ROOT / "raw/graph-label.csv.gz"
SPLIT_FILES = {name: OGB_ROOT / f"split/scaffold/{name}.csv.gz" for name in ("train", "valid", "test")}

BOC_FREEZE = BOC_ROUND / "analysis/test_prediction_manifest.json"
S3_FREEZE = S3SEL_ROUND / "analysis/test_prediction_manifest.json"
BOC_TEST_SCORES = BOC_ROUND / "analysis/test_scores.json"
S3_TEST_SCORES = S3SEL_ROUND / "analysis/test_scores.json"

#: The five frozen best-Validation models.  The manifest keys are the historical
#: prediction labels; every path/hash below is re-checked against the manifests
#: in :mod:`lc_inventory` before any analysis runs.
MODELS: dict[str, dict[str, Any]] = {
    "W2": {
        "label": "W2",
        "model": "W2",
        "display": "W2",
        "role": "primary reference",
        "round": "lcrc-pcba-backbone-scaling-v1",
        "epoch": 24,
        "checkpoint": BS_ROUND / "runs/W2/checkpoint_best.pt",
        "manifest": "boc",
        "manifest_key": "W2_best_valid",
    },
    "S3": {
        "label": "S3",
        "model": "S3",
        "display": "S3",
        "role": "primary new architecture",
        "round": "lcrc-pcba-s3-v1",
        "epoch": 29,
        "checkpoint": S3_ROUND / "runs/S3/checkpoint_best.pt",
        "manifest": "s3",
        "manifest_key": "S3_best_valid_epoch29",
    },
    "RC_REAL": {
        "label": "RC_REAL",
        "model": "RC-REAL",
        "display": "RC-REAL",
        "role": "primary new architecture",
        "round": "lcrc-pcba-rc-v1",
        "epoch": 33,
        "checkpoint": RC_ROUND / "runs/REAL/checkpoint_best.pt",
        "manifest": "boc",
        "manifest_key": "RC_REAL_best_valid",
    },
    "BOC_FULL": {
        "label": "BOC_FULL",
        "model": "FULL",
        "display": "BOC-FULL",
        "role": "primary new architecture",
        "round": "lcrc-pcba-boc-v1",
        "epoch": 29,
        "checkpoint": BOC_ROUND / "runs/FULL/checkpoint_best.pt",
        "manifest": "boc",
        "manifest_key": "BOC_FULL_best_valid",
    },
    "BOC_BOND": {
        "label": "BOC_BOND",
        "model": "BOND",
        "display": "BOC-BOND",
        "role": "auxiliary",
        "round": "lcrc-pcba-boc-v1",
        "epoch": 30,
        "checkpoint": BOC_ROUND / "runs/BOND/checkpoint_best.pt",
        "manifest": "boc",
        "manifest_key": "BOC_BOND_best_valid",
    },
}
MODEL_ORDER = ("W2", "S3", "RC_REAL", "BOC_FULL", "BOC_BOND")

#: (new, reference) after the first pair element is W2.  The first three are the
#: pre-registered primary comparisons; the last two are auxiliary.
PRIMARY_PAIRS = (("S3", "W2"), ("RC_REAL", "W2"), ("BOC_FULL", "W2"))
AUX_PAIRS = (("BOC_FULL", "RC_REAL"), ("BOC_BOND", "BOC_FULL"))
ALL_PAIRS = PRIMARY_PAIRS + AUX_PAIRS

#: Support / novelty constants.  Single definition, shared by both splits.
HIGH_SUPPORT_POS = 20
DESCRIPTIVE_POS_RANGE = (5, 19)
RECORD_POS_RANGE = (1, 4)
RARE_SUPPORT_MAX = 4
NOVELTY_PARTIAL_MAX = 0.25

SUPPORT_STRATA = ("ge20", "5_19", "1_4", "na")
NOVELTY_GROUPS = ("SEEN", "PARTIAL_NOVEL", "HIGH_NOVEL", "UNRESOLVED")

#: Pre-registered decision thresholds (PROTOCOL.md sections 5-8).
RHO_SIMILAR = 0.80
TOPK1_JACCARD_SIMILAR = 0.50
TOPK5_JACCARD_SIMILAR = 0.60
NOVELTY_GAIN_MARGIN = 0.001
LOW_SUPPORT_SHARE_STRONG = 0.25
LOW_SUPPORT_SHARE_WEAK = 0.10
VALID_REPLAY_TOLERANCE = 1e-4
PER_TASK_MAIN_TOLERANCE = 1e-3
AP_AGREEMENT_TOLERANCE = 1e-6


# ---------------------------------------------------------------------------
# hashing / IO
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_int64(values: Iterable[int]) -> str:
    arr = np.ascontiguousarray(np.asarray(list(values), dtype=np.int64))
    return sha256_bytes(arr.tobytes())


def array_sha256(array: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(array).tobytes())


def rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols = list(columns) if columns is not None else list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in cols})


def file_record(path: Path, role: str) -> dict[str, Any]:
    path = Path(path)
    return {
        "path": rel(path),
        "role": role,
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


# ---------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------


def quantiles(values: Sequence[float]) -> dict[str, Any]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "median": None, "q1": None, "q3": None, "min": None, "max": None, "mean": None}
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "q1": float(np.percentile(arr, 25)),
        "q3": float(np.percentile(arr, 75)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def summary_stats(values: Sequence[float]) -> dict[str, Any]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": None, "median": None, "sum": 0.0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "sum": float(arr.sum()),
    }


def support_stratum(n_pos: int, n_neg: int) -> str:
    """Frozen reliability stratum from positive/negative support only."""
    if n_pos <= 0 or n_neg <= 0:
        return "na"
    if n_pos >= HIGH_SUPPORT_POS:
        return "ge20"
    if n_pos >= DESCRIPTIVE_POS_RANGE[0]:
        return "5_19"
    return "1_4"


def novelty_group(fraction_unseen: float | None) -> str:
    if fraction_unseen is None or not np.isfinite(fraction_unseen):
        return "UNRESOLVED"
    if fraction_unseen == 0.0:
        return "SEEN"
    if fraction_unseen <= NOVELTY_PARTIAL_MAX:
        return "PARTIAL_NOVEL"
    return "HIGH_NOVEL"


def pair_key(a: str, b: str) -> str:
    return f"{a}_minus_{b}"


def utcnow() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = [name for name in dir() if not name.startswith("_")]
