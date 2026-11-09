"""Read-only data access: official splits, labels, frozen predictions.

The official Test labels are opened **only** through the inherited
``cc_data.read_official_test_labels`` gate, which verifies the historical freeze
manifest and re-checks the raw Train/Valid label blocks against the frozen label
artifact before returning the Test block.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import lc_common as C

for _p in (str(C.BOC_ROUND / "code"), str(C.CC_ROUND / "code"), str(C.RC_ROUND / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import boc_common as BC  # noqa: E402  (sets up inherited code dirs / frozen data)
import cc_data as CD  # noqa: E402

MET = BC.MET  # pc_metrics: official OGB AP + independent per-task AP


@lru_cache(maxsize=1)
def load_task() -> Any:
    """Frozen MolPCBA task (labels with Test rows NaN, official split rows)."""
    task = CD.load_task()
    if task.labels.shape != (437_929, C.N_TASKS):
        raise RuntimeError(f"label artifact shape {task.labels.shape}")
    if np.isfinite(task.labels[np.asarray(task.test_rows, np.int64)]).any():
        raise RuntimeError("Test labels leaked into the training artifact")
    return task


@lru_cache(maxsize=1)
def official_test_labels() -> dict[str, Any]:
    """Official Test labels via the historical gate (freeze-verified)."""
    task = load_task()
    out = CD.read_official_test_labels(task, freeze_path=C.BOC_FREEZE)
    y = np.asarray(out["labels"], dtype=np.float32)
    if y.shape != (C.N_TEST_ROWS, C.N_TASKS):
        raise RuntimeError(f"test label block shape {y.shape}")
    out["labels"] = y
    return out


def split_rows(split: str) -> np.ndarray:
    task = load_task()
    if split == "valid":
        return np.asarray(task.valid_rows, dtype=np.int64)
    if split == "test":
        return np.asarray(task.test_rows, dtype=np.int64)
    if split == "train":
        return np.asarray(task.train_rows, dtype=np.int64)
    raise KeyError(split)


def split_labels(split: str) -> np.ndarray:
    """Observed-label matrix for ``split``, shape [n_rows, 128] (NaN = missing)."""
    if split == "test":
        return official_test_labels()["labels"]
    rows = split_rows(split)
    return np.asarray(load_task().labels[rows], dtype=np.float32)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """Historical scoring transform: float64 sigmoid, stored as float32."""
    x = np.asarray(logits, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def manifest(which: str) -> dict[str, Any]:
    path = {"boc": C.BOC_FREEZE, "s3": C.S3_FREEZE}[which]
    return C.read_json(path)


def manifest_record(label: str) -> dict[str, Any]:
    spec = C.MODELS[label]
    return manifest(spec["manifest"])["predictions"][spec["manifest_key"]]


def test_logits_path(label: str) -> Path:
    return C.REPO / manifest_record(label)["test_logits_path"]


def load_valid_manifest() -> dict[str, Any]:
    path = C.LOCAL_VALID / "valid_prediction_manifest.json"
    if not path.is_file():
        raise RuntimeError(f"validation predictions missing: {path} (run lc_valid.py first)")
    return C.read_json(path)


def load_logits(label: str, split: str, *, verify: bool = True) -> np.ndarray:
    """Frozen logits [n_rows,128] float32 for one model / split."""
    if split == "test":
        path = test_logits_path(label)
        expected_sha = manifest_record(label)["test_logits_sha256"]
    elif split == "valid":
        entry = load_valid_manifest()["models"][label]
        path = C.REPO / entry["path"]
        expected_sha = entry["sha256"]
    else:
        raise KeyError(split)
    if not path.is_file():
        raise RuntimeError(f"missing logits: {path}")
    if verify:
        actual = C.sha256_file(path)
        if actual != expected_sha:
            raise RuntimeError(f"{label}/{split} logits sha {actual} != recorded {expected_sha}")
    arr = np.load(path, allow_pickle=False)
    if arr.shape != (C.N_TEST_ROWS if split == "test" else C.N_VALID_ROWS, C.N_TASKS):
        raise RuntimeError(f"{label}/{split} logits shape {arr.shape}")
    if arr.dtype != np.float32 or not np.isfinite(arr).all():
        raise RuntimeError(f"{label}/{split} logits dtype/finite check failed: {arr.dtype}")
    return np.ascontiguousarray(arr, dtype=np.float32)


def ap_block(y: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    """Official + independent AP for a [rows,128] label/logits block."""
    probs = sigmoid(logits)
    official = MET.official_ap(y, probs)
    indep = MET.per_task_ap(y, probs)
    return {
        "official_ap": float(official["value"]),
        "independent_ap": float(indep["mean_over_scored_tasks"]),
        "abs_delta": float(abs(official["value"] - indep["mean_over_scored_tasks"])),
        "per_task_ap": indep["per_task"],
        "observed": indep["observed"],
        "positive": indep["positive"],
        "negative": indep["negative"],
        "n_tasks_scored": int(indep["n_tasks_scored"]),
        "n_tasks_with_observed_labels": int(indep["n_tasks_with_observed_labels"]),
    }


def task_ap(y_col: np.ndarray, logits_col: np.ndarray) -> float | None:
    """AP of one task column on its observed rows (needs both classes).

    The score is the historical ``sigmoid(logits) -> float32`` probability, so a
    recomputed task AP is directly comparable to the frozen records.
    """
    from sklearn.metrics import average_precision_score

    yy = np.asarray(y_col, dtype=np.float64)
    obs = np.isfinite(yy)
    yt = yy[obs]
    if yt.size == 0:
        return None
    n_pos = int((yt == 1.0).sum())
    n_neg = int((yt == 0.0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    probs = sigmoid(np.asarray(logits_col, dtype=np.float32))[obs].astype(np.float64)
    return float(average_precision_score(yt, probs))


def task_support(y_col: np.ndarray) -> dict[str, int]:
    yy = np.asarray(y_col, dtype=np.float64)
    obs = np.isfinite(yy)
    yt = yy[obs]
    return {
        "observed": int(obs.sum()),
        "positive": int((yt == 1.0).sum()),
        "negative": int((yt == 0.0).sum()),
    }


__all__ = [
    "MET",
    "ap_block",
    "load_logits",
    "load_task",
    "load_valid_manifest",
    "manifest",
    "manifest_record",
    "official_test_labels",
    "sigmoid",
    "split_labels",
    "split_rows",
    "task_ap",
    "task_support",
    "test_logits_path",
]
