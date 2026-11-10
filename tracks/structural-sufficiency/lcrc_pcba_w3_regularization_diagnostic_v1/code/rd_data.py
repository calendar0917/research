"""Three-split PCBA evaluation with isolated, official Test labels.

Training data still comes from the frozen ``sg_data`` path (same cache, same
official split, same Train-only normalization, same label artifact whose Test
rows are NaN, same official OGB AP).  This module adds exactly two things:

* a **read-only** reader for the official OGB raw label file
  ``data/ogb/ogbg_molpcba/raw/graph-label.csv.gz`` that proves the file's
  Train/Valid blocks equal the frozen artifact (NaN mask included) before any
  Test row is used, and returns the Test labels for metric computation only;
* a per-split evaluation that records official macro AP, masked BCE, per-task
  AP and per-task support for Train, Validation and Test.

Nothing here is used by the training step: the trainer builds its batches from
``task.labels`` (Test rows NaN) and this module's Test labels are never passed
to it.
"""

from __future__ import annotations

import csv
import gzip
import time
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

import rd_common as C

MET = C.MET
PC = C.PC


# ---------------------------------------------------------------------------
# training-side task (frozen artifact, Test rows NaN)
# ---------------------------------------------------------------------------


def load_task() -> Any:
    """The frozen Task, with every inherited identity re-checked."""
    task = C.SD.load_task()
    if task.labels.shape != (C.N_MOLECULES, C.N_TASKS):
        raise RuntimeError(f"label artifact shape {task.labels.shape}")
    if int(task.train_rows.size) != C.N_TRAIN or int(task.valid_rows.size) != C.N_VALID:
        raise RuntimeError(
            f"split sizes {task.train_rows.size}/{task.valid_rows.size} != {C.N_TRAIN}/{C.N_VALID}"
        )
    if int(task.test_rows.size) != C.N_TEST:
        raise RuntimeError(f"Test split size {task.test_rows.size} != {C.N_TEST}")
    # Official Test targets must stay NaN inside the *training* artifact.
    if np.isfinite(task.labels[task.test_rows]).any():
        raise RuntimeError("frozen training artifact exposes Test labels; refusing to continue")
    for name, rows in (("train", task.train_rows), ("valid", task.valid_rows), ("test", task.test_rows)):
        if np.unique(rows).size != rows.size:
            raise RuntimeError(f"{name} split has duplicate row ids")
    pairs = (("train", "valid"), ("train", "test"), ("valid", "test"))
    for a, b in pairs:
        if np.intersect1d(task.split[a], task.split[b]).size:
            raise RuntimeError(f"{a}/{b} split rows overlap")
    return task


def split_rows(task: Any) -> dict[str, np.ndarray]:
    return {"train": task.train_rows, "valid": task.valid_rows, "test": task.test_rows}


def split_partition_audit(task: Any) -> dict[str, Any]:
    rows = split_rows(task)
    union = np.concatenate([rows[s] for s in C.SPLITS])
    counts = {s: int(rows[s].size) for s in C.SPLITS}
    return {
        "counts": counts,
        "expected_counts": dict(C.ROW_COUNTS),
        "counts_match": counts == dict(C.ROW_COUNTS),
        "union_size": int(union.size),
        "union_unique": int(np.unique(union).size),
        "covers_all_molecules": int(np.unique(union).size) == C.N_MOLECULES,
        "min_row": int(union.min()),
        "max_row": int(union.max()),
        "train_valid_overlap": int(np.intersect1d(rows["train"], rows["valid"]).size),
        "train_test_overlap": int(np.intersect1d(rows["train"], rows["test"]).size),
        "valid_test_overlap": int(np.intersect1d(rows["valid"], rows["test"]).size),
        "row_id_sha256": {s: C.sha256_bytes(np.ascontiguousarray(rows[s], dtype=np.int64).tobytes()) for s in C.SPLITS},
        "split_partition_sha256": task.split_partition_sha256,
    }


# ---------------------------------------------------------------------------
# official raw label file: Test labels, aligned and isolated
# ---------------------------------------------------------------------------


def _blank_labels() -> np.ndarray:
    out = np.empty((C.N_MOLECULES, C.N_TASKS), dtype=np.float32)
    out.fill(np.nan)
    return out


def read_raw_label_matrix() -> tuple[np.ndarray, dict[str, Any]]:
    """Read the official OGB label matrix into a preallocated float32 array.

    Chunked row-wise fill: the file is never expanded into a Python float list.
    ``""`` -> NaN (missing), everything else -> float32.
    """
    if not C.OGB_GRAPH_LABEL.is_file():
        raise RuntimeError(f"official OGB label file missing: {C.OGB_GRAPH_LABEL}")
    t0 = time.perf_counter()
    labels = _blank_labels()
    row = 0
    bad_rows: list[int] = []
    with gzip.open(C.OGB_GRAPH_LABEL, "rt", newline="") as fh:
        reader = csv.reader(fh)
        for record in reader:
            if row >= C.N_MOLECULES:
                raise RuntimeError(f"official label file has more than {C.N_MOLECULES} rows")
            if len(record) != C.N_TASKS:
                bad_rows.append(row)
                raise RuntimeError(f"official label row {row} has {len(record)} columns != {C.N_TASKS}")
            labels[row] = np.asarray(
                [np.nan if v == "" else float(v) for v in record], dtype=np.float32
            )
            row += 1
    if row != C.N_MOLECULES:
        raise RuntimeError(f"official label file has {row} rows != {C.N_MOLECULES}")
    return labels, {
        "path": C.track_rel(C.OGB_GRAPH_LABEL),
        "sha256": C.sha256_file(C.OGB_GRAPH_LABEL),
        "shape": list(labels.shape),
        "rows_read": int(row),
        "bad_rows": bad_rows,
        "read_seconds": float(time.perf_counter() - t0),
        "finite_entries": int(np.isfinite(labels).sum()),
        "chunked_read": True,
        "python_float_list_materialised": False,
    }


def _nan_mask(arr: np.ndarray) -> np.ndarray:
    return np.isnan(arr)


def read_official_test_labels(task: Any) -> dict[str, Any]:
    """Test labels from the official raw file, after full alignment proof."""
    raw, meta = read_raw_label_matrix()
    train = np.asarray(task.train_rows, dtype=np.int64)
    valid = np.asarray(task.valid_rows, dtype=np.int64)
    test = np.asarray(task.test_rows, dtype=np.int64)

    # 1) Train/Valid must equal the frozen artifact *including* the NaN mask.
    for name, rows in (("train", train), ("valid", valid)):
        a = raw[rows]
        b = np.asarray(task.labels[rows], dtype=np.float32)
        if not np.array_equal(_nan_mask(a), _nan_mask(b)):
            raise RuntimeError(f"raw OGB {name} NaN mask differs from the frozen artifact")
        finite = np.isfinite(a)
        if not np.array_equal(a[finite], b[finite]):
            raise RuntimeError(f"raw OGB {name} labels differ from the frozen artifact")
    # 2) The frozen artifact must not expose Test labels.
    if np.isfinite(task.labels[test]).any():
        raise RuntimeError("frozen artifact exposes Test labels")

    y_test = np.ascontiguousarray(raw[test], dtype=np.float32)
    finite = np.isfinite(y_test)
    values = y_test[finite]
    if not np.isin(values, (0.0, 1.0)).all():
        raise RuntimeError("official Test labels are not binary 0/1")
    if y_test.shape != (C.N_TEST, C.N_TASKS):
        raise RuntimeError(f"Test label matrix shape {y_test.shape} != {(C.N_TEST, C.N_TASKS)}")
    meta.update(
        {
            "test_shape": list(y_test.shape),
            "test_observed_entries": int(finite.sum()),
            "test_positive_entries": int((values == 1.0).sum()),
            "test_negative_entries": int((values == 0.0).sum()),
            "test_rows_sha256": C.sha256_bytes(np.ascontiguousarray(test, dtype=np.int64).tobytes()),
            "test_labels_sha256": C.sha256_bytes(np.ascontiguousarray(y_test, dtype=np.float32).tobytes()),
            "train_valid_alignment_verified": True,
            "train_artifact_label_sha256": task.label_artifact_sha256,
            "test_labels_from": "official OGB raw/graph-label.csv.gz",
            "test_labels_used_for": "metric computation only",
            "test_labels_enter_training": False,
        }
    )
    return {"labels": y_test, "audit": meta}


# ---------------------------------------------------------------------------
# per-split label / count bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class SplitData:
    name: str
    rows: np.ndarray
    labels: np.ndarray
    observed: list[int]
    positive: list[int]
    negative: list[int]

    @property
    def observed_entries(self) -> int:
        return int(sum(self.observed))

    @property
    def positive_entries(self) -> int:
        return int(sum(self.positive))


def build_split_data(task: Any, test_labels: np.ndarray | None) -> dict[str, SplitData]:
    out: dict[str, SplitData] = {}
    for name in C.SPLITS:
        rows = np.asarray(task.split[name], dtype=np.int64)
        y = np.ascontiguousarray(task.labels[rows], dtype=np.float32) if name != "test" else test_labels
        if y is None:
            raise RuntimeError("Test labels are required for the three-split evaluation")
        if y.shape != (rows.size, C.N_TASKS):
            raise RuntimeError(f"{name} label matrix {y.shape} != {(rows.size, C.N_TASKS)}")
        finite = np.isfinite(y)
        pos = ((y == 1.0) & finite).sum(axis=0)
        neg = ((y == 0.0) & finite).sum(axis=0)
        out[name] = SplitData(
            name=name,
            rows=rows,
            labels=y,
            observed=[int(x) for x in finite.sum(axis=0)],
            positive=[int(x) for x in pos],
            negative=[int(x) for x in neg],
        )
    return out


def task_counts(task: Any, splits: Mapping[str, SplitData]) -> dict[str, Any]:
    """Per-task observed/positive/negative counts, all three splits (labels only)."""
    return {
        name: {
            "observed": splits[name].observed,
            "positive": splits[name].positive,
            "negative": splits[name].negative,
            "observed_entries": splits[name].observed_entries,
            "positive_entries": splits[name].positive_entries,
            "rows": int(splits[name].rows.size),
        }
        for name in C.SPLITS
    }


def scorable_tasks(split: SplitData) -> list[int]:
    """Tasks with both classes present (the official evaluator's inclusion rule)."""
    return [t for t in range(C.N_TASKS) if split.positive[t] > 0 and split.negative[t] > 0]


# ---------------------------------------------------------------------------
# inference + metrics
# ---------------------------------------------------------------------------


def _array_sha(arr: np.ndarray) -> str:
    return C.sha256_bytes(np.ascontiguousarray(arr, dtype=np.float32).tobytes())


def predict_rows(
    model: Any, task: Any, rows: np.ndarray, device: Any, *, batch_size: int
) -> np.ndarray:
    """The frozen ``pc_model.predict_logits`` path (same collate, same transfer)."""
    return PC.predict_logits(model, task.cache, rows, device, batch_size=int(batch_size))


def evaluate_split(
    model: Any,
    task: Any,
    split: SplitData,
    device: Any,
    *,
    batch_size: int,
    reason: str = "",
) -> dict[str, Any]:
    """Full-split inference + official AP / masked BCE / per-task AP.

    The logits and probability arrays are dropped before returning; only the
    per-task summary and hashes survive, so no epoch's full Train logits are
    written to disk.
    """
    t0 = time.perf_counter()
    logits = predict_rows(model, task, split.rows, device, batch_size=batch_size)
    infer_seconds = float(time.perf_counter() - t0)
    if logits.shape != (split.rows.size, C.N_TASKS):
        raise RuntimeError(f"{split.name} logits {logits.shape} != {(split.rows.size, C.N_TASKS)}")
    if not np.isfinite(logits).all():
        raise RuntimeError(f"non-finite logits on split {split.name}")

    t1 = time.perf_counter()
    bce = float(MET.masked_bce_numpy(logits, split.labels))
    probs = PC.sigmoid(logits)
    report = MET.validation_report(split.labels, probs)
    metric_seconds = float(time.perf_counter() - t1)
    elapsed = float(time.perf_counter() - t0)

    if int(report["n_tasks"]) != C.N_TASKS:
        raise RuntimeError("metric report does not cover 128 tasks")
    out = {
        "split": split.name,
        "reason": reason,
        "rows": int(split.rows.size),
        "ap": float(report["ap"]),
        "bce": bce,
        "per_task_ap": list(report["per_task_ap"]),
        "observed": [int(x) for x in report["observed"]],
        "positive": [int(x) for x in report["positive"]],
        "negative": [int(x) for x in report["negative"]],
        "n_tasks_scored": int(report["n_tasks_scored"]),
        "n_tasks_with_observed_labels": int(report["n_tasks_with_observed_labels"]),
        "observed_entries": int(sum(report["observed"])),
        "positive_entries": int(sum(report["positive"])),
        "logits_sha256": _array_sha(logits),
        "probs_sha256": _array_sha(probs),
        "infer_seconds": infer_seconds,
        "metric_seconds": metric_seconds,
        "seconds": elapsed,
        "batch_size": int(batch_size),
        "ap_source": report["ap_source"],
    }
    # release the big arrays before the next split
    del logits, probs, report
    return out


def cross_check_ap(y_true: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    """Official OGB AP vs an independent per-task sklearn recomputation."""
    return MET.ap_agreement(y_true, probs)


__all__ = [
    "SplitData",
    "build_split_data",
    "cross_check_ap",
    "evaluate_split",
    "load_task",
    "predict_rows",
    "read_official_test_labels",
    "read_raw_label_matrix",
    "scorable_tasks",
    "split_partition_audit",
    "split_rows",
    "task_counts",
]
