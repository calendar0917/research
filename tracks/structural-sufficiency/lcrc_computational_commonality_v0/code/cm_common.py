"""Shared definitions for the LCRC computational-commonality round v0.

Read-only inheritance (never modified):

* ``lcrc_pcba_bottleneck_diagnosis_v1/code`` -- task/split/label access, the
  frozen Valid logits of the historical models and the official AP helpers.
* ``direct_structural_computation_recurrence_v0/code`` -- DSCR arms, their
  frozen best-Validation checkpoints and the read-only split evaluation.
* ``lcrc_pcba_backbone_scaling_v1/code/bs_model.py`` -- the W2 constructor.
* ``lcrc_pcba_dynamic_v1/code`` -- the historical W2-DYNAMIC checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE = Path(__file__).resolve().parent
ROUND = CODE.parent
TRACK = ROUND.parent
REPO = TRACK.parents[1]

BOTTLENECK = REPO / "tracks/structural-sufficiency/lcrc_pcba_bottleneck_diagnosis_v1"
DSCR = REPO / "tracks/structural-sufficiency/direct_structural_computation_recurrence_v0"
BS_ROUND = REPO / "tracks/structural-sufficiency/lcrc_pcba_backbone_scaling_v1"
DY_ROUND = REPO / "tracks/structural-sufficiency/lcrc_pcba_dynamic_v1"

for _p in (BOTTLENECK / "code", DSCR / "code", BS_ROUND / "code", DY_ROUND / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import lc_data as LD  # noqa: E402
import lc_metrics as LM  # noqa: E402
import ds_common as DC  # noqa: E402
import ds_model as DM  # noqa: E402

MET = LD.MET
DATA = LD

ROUND_ID = "lcrc-computational-commonality-v0"
PROTOCOL_ID = ROUND_ID
ANALYSIS = ROUND / "analysis"
LOCAL = ROUND / "_local"
PREDICTIONS = LOCAL / "predictions"
N_TASKS = 128
N_VALID = 43_793
N_TEST = 43_793
N_TRAIN = 350_343
SEED = 0
TEST_ROLE = "DEVELOPMENT_TEST / POST_HOC_EXPOSED_TEST_DIAGNOSIS"

#: Every model in the comparison.  ``role`` records the architecture family;
#: ``valid_logits``/``test_logits`` are the frozen per-molecule arrays (either a
#: historical freeze or a materialised read-only forward of the same checkpoint).
MODELS: dict[str, dict[str, Any]] = {
    "W2_s0": {
        "arch": "W2",
        "seed": 0,
        "family": "bs",
        "display": "W2 s0",
        "checkpoint": BS_ROUND / "runs/W2/checkpoint_best.pt",
        "parameters": 994_324,
        "horizon_epochs": 40,
        "best_epoch": 24,
        "valid_ap_recorded": 0.26527331610028143,
        "test_ap_recorded": 0.2536164787414065,
        "valid_logits": BOTTLENECK / "_local/valid_predictions/W2.npy",
        "test_logits": REPO / "tracks/structural-sufficiency/lcrc_pcba_boc_v1/runs/freeze/test_predictions/W2_best_valid.npy",
        "recipe_note": "historical 40-epoch scratch recipe (backbone-scaling-v1)",
    },
    "W2_s1": {
        "arch": "W2",
        "seed": 1,
        "family": "bs",
        "display": "W2 s1",
        "checkpoint": DSCR / "runs/W2_seed1/checkpoint_best.pt",
        "parameters": 994_324,
        "horizon_epochs": 25,
        "best_epoch": 23,
        "valid_ap_recorded": 0.25705818131531044,
        "test_ap_recorded": 0.24627545258665828,
        "recipe_note": "25-epoch prefix of the frozen 40-epoch cosine (DSCR horizon)",
    },
    "FIXED1_s0": {
        "arch": "FIXED1",
        "seed": 0,
        "family": "ds",
        "display": "FIXED1 s0",
        "checkpoint": DSCR / "runs/FIXED1_seed0/checkpoint_best.pt",
        "parameters": 1_128_468,
        "horizon_epochs": 25,
        "best_epoch": 25,
        "valid_ap_recorded": 0.27584697853384577,
        "test_ap_recorded": 0.25700143219823374,
        "recipe_note": "25-epoch prefix of the frozen 40-epoch cosine (DSCR horizon)",
    },
    "FIXED1_s1": {
        "arch": "FIXED1",
        "seed": 1,
        "family": "ds",
        "display": "FIXED1 s1",
        "checkpoint": DSCR / "runs/FIXED1_seed1/checkpoint_best.pt",
        "parameters": 1_128_468,
        "horizon_epochs": 25,
        "best_epoch": 19,
        "valid_ap_recorded": 0.2731070161937686,
        "test_ap_recorded": 0.2603009814576431,
        "recipe_note": "25-epoch prefix of the frozen 40-epoch cosine (DSCR horizon)",
    },
    "DYNAMIC_s0": {
        "arch": "DYNAMIC",
        "seed": 0,
        "family": "dy",
        "display": "DYNAMIC s0",
        "checkpoint": DY_ROUND / "runs/W2_DYNAMIC/checkpoint_best.pt",
        "parameters": 1_128_468,
        "horizon_epochs": 40,
        "best_epoch": 25,
        "valid_ap_recorded": 0.2750671855066329,
        "test_ap_recorded": 0.262226,
        "test_logits": DY_ROUND / "analysis/test_predictions/W2_DYNAMIC_best_valid.npy",
        "recipe_note": "historical 40-epoch scratch recipe (dynamic-v1)",
    },
    "DYNAMIC_s1": {
        "arch": "DYNAMIC",
        "seed": 1,
        "family": "ds",
        "display": "DYNAMIC s1",
        "checkpoint": DSCR / "runs/DYNAMIC_seed1/checkpoint_best.pt",
        "parameters": 1_128_468,
        "horizon_epochs": 25,
        "best_epoch": 19,
        "valid_ap_recorded": 0.27298095032591224,
        "test_ap_recorded": 0.2538025460485901,
        "recipe_note": "25-epoch prefix of the frozen 40-epoch cosine (DSCR horizon)",
    },
    "S3_s0": {
        "arch": "S3",
        "seed": 0,
        "family": "frozen",
        "display": "S3 s0",
        "checkpoint": REPO / "tracks/structural-sufficiency/lcrc_pcba_s3_v1/runs/S3/checkpoint_best.pt",
        "parameters": 1_018_532,
        "horizon_epochs": 40,
        "best_epoch": 29,
        "valid_ap_recorded": 0.2714881532227929,
        "test_ap_recorded": 0.25063338202304364,
        "valid_logits": BOTTLENECK / "_local/valid_predictions/S3.npy",
        "test_logits": REPO / "tracks/structural-sufficiency/lcrc_pcba_s3_selective_v1/analysis/test_predictions/S3_best_valid_epoch29.npy",
        "recipe_note": "historical 40-epoch scratch recipe (s3-v1)",
    },
    "RC_REAL_s0": {
        "arch": "RC_REAL",
        "seed": 0,
        "family": "frozen",
        "display": "RC-REAL s0",
        "checkpoint": REPO / "tracks/structural-sufficiency/lcrc_pcba_rc_v1/runs/REAL/checkpoint_best.pt",
        "parameters": 1_051_022,
        "horizon_epochs": 40,
        "best_epoch": 33,
        "valid_ap_recorded": 0.2677142738486777,
        "test_ap_recorded": 0.2559409168065797,
        "valid_logits": BOTTLENECK / "_local/valid_predictions/RC_REAL.npy",
        "test_logits": REPO / "tracks/structural-sufficiency/lcrc_pcba_boc_v1/runs/freeze/test_predictions/RC_REAL_best_valid.npy",
        "recipe_note": "historical 40-epoch scratch recipe (rc-v1)",
    },
    "BOC_FULL_s0": {
        "arch": "BOC_FULL",
        "seed": 0,
        "family": "frozen",
        "display": "BOC-FULL s0",
        "checkpoint": REPO / "tracks/structural-sufficiency/lcrc_pcba_boc_v1/runs/FULL/checkpoint_best.pt",
        "parameters": 1_352_596,
        "horizon_epochs": 40,
        "best_epoch": 29,
        "valid_ap_recorded": 0.26995450035573837,
        "test_ap_recorded": 0.25414238787847127,
        "valid_logits": BOTTLENECK / "_local/valid_predictions/BOC_FULL.npy",
        "test_logits": REPO / "tracks/structural-sufficiency/lcrc_pcba_boc_v1/runs/freeze/test_predictions/BOC_FULL_best_valid.npy",
        "recipe_note": "historical 40-epoch scratch recipe (boc-v1)",
    },
}

CORE = ("W2_s0", "W2_s1", "FIXED1_s0", "FIXED1_s1", "DYNAMIC_s0", "DYNAMIC_s1")
EXTENDED_ARMS = ("W2_s0", "FIXED1_s0", "DYNAMIC_s0", "S3_s0", "RC_REAL_s0", "BOC_FULL_s0")

#: Models whose frozen logits are on disk (no inference required).
FROZEN = tuple(k for k, v in MODELS.items() if "valid_logits" in v and "test_logits" in v)
NEEDS_INFERENCE = tuple(k for k in MODELS if k not in FROZEN)

TEST_ROWS_PATH = BOTTLENECK / "_local/valid_predictions/valid_row_ids.npy"


# ---------------------------------------------------------------------------
# io helpers
# ---------------------------------------------------------------------------


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float32).tobytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True, default=_default)
    tmp.replace(path)


def _default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj))


def rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except Exception:
        return str(path)


def save_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(columns) if columns else sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


# ---------------------------------------------------------------------------
# labels / scores
# ---------------------------------------------------------------------------


def y_valid() -> np.ndarray:
    return np.asarray(DATA.split_labels("valid"), dtype=np.float32)


def y_test() -> np.ndarray:
    out = DATA.official_test_labels()
    return np.asarray(out["labels"], dtype=np.float32)


def score_logits(logits: np.ndarray) -> np.ndarray:
    """Historical scoring: ``sigmoid(float32 logits)``."""
    x = np.asarray(logits, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def rank_score(logits: np.ndarray) -> np.ndarray:
    """Per-task percentile rank over ALL molecules of the split.

    ``r = (average_rank - 0.5) / N``; ties get average ranks.  Rows with the
    task label missing still receive a rank (transductive candidate set).
    """
    arr = np.asarray(logits, dtype=np.float64)
    n = arr.shape[0]
    out = np.empty_like(arr, dtype=np.float64)
    denom = float(n)
    for t in range(arr.shape[1]):
        order = np.argsort(arr[:, t], kind="mergesort")
        vals = arr[order, t]
        ranks = np.empty(n, dtype=np.float64)
        i = 0
        # average ranks for exact ties
        while i < n:
            j = i + 1
            while j < n and vals[j] == vals[i]:
                j += 1
            ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
            i = j
        out[:, t] = (ranks - 0.5) / denom
    return out.astype(np.float32)


def macro_ap(y: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    """Official-style macro AP: sklearn per-task AP averaged over scorable tasks."""
    per = MET.per_task_ap(np.asarray(y), np.asarray(scores, dtype=np.float32))
    return {
        "macro_ap": float(per["mean_over_scored_tasks"]),
        "per_task_ap": per["per_task"],
        "observed": per["observed"],
        "positive": per["positive"],
        "negative": per["negative"],
        "n_tasks_scored": int(per["n_tasks_scored"]),
    }


def official_macro_ap(y: np.ndarray, scores: np.ndarray) -> float:
    return float(MET.official_ap(np.asarray(y), np.asarray(scores, dtype=np.float32))["value"])


def inference_path(label: str, split: str) -> Path:
    return PREDICTIONS / f"{label}_{split}.npy"


def load_logits(label: str, split: str) -> np.ndarray:
    """Frozen logits for one model/split (historical freeze or local forward)."""
    spec = MODELS[label]
    key = f"{split}_logits"
    path = spec.get(key)
    if path is None:
        path = inference_path(label, split)
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"missing {label}/{split} logits: {rel(path)}")
    arr = np.load(path, allow_pickle=False)
    n = N_VALID if split == "valid" else N_TEST
    if arr.shape != (n, N_TASKS) or arr.dtype != np.float32:
        raise RuntimeError(f"{label}/{split} logits shape/dtype {arr.shape} {arr.dtype}")
    if not np.isfinite(arr).all():
        raise RuntimeError(f"{label}/{split} logits are not finite")
    return np.ascontiguousarray(arr, dtype=np.float32)


def available(label: str) -> dict[str, bool]:
    spec = MODELS[label]
    out = {}
    for split in ("valid", "test"):
        path = spec.get(f"{split}_logits") or inference_path(label, split)
        out[split] = Path(path).is_file()
    return out


__all__ = [n for n in dir() if not n.startswith("_")]
