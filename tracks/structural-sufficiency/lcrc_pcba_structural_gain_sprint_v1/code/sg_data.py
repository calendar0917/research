"""Read-only PCBA access, the frozen 40-epoch order plan and evaluation.

Everything here delegates to the already-audited loader and metric code:

* data/order: ``lr_data`` (which refuses Test labels and checks every hash),
* transfer: ``PinnedStager`` (already gated by the pipeline-perf round),
* metrics: ``pc_metrics`` with the OGB official Evaluator as the only AP.

No new data protocol is introduced in this round.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import sg_common as C
import lr_data as LD
import pc_metrics as MET
import pc_model as BM
from pf_pipeline import PinnedStager  # noqa: F401  (re-exported for the trainer)


def load_task() -> Any:
    task = LD.load_task()
    if task.labels.shape != (C.N_MOLECULES, C.N_TASKS):
        raise RuntimeError(f"label artifact shape {task.labels.shape}")
    if np.isfinite(task.labels[task.test_rows]).any():
        raise RuntimeError("official Test labels are exposed; refusing to continue")
    if task.spec.hash() != C.SOURCE["feature_spec_hash"]:
        raise RuntimeError("feature spec hash drifted")
    if task.norm["norm_fingerprint"] != C.SOURCE["norm_fingerprint"]:
        raise RuntimeError("Train-only normalization fingerprint drifted")
    if task.split_partition_sha256 != C.SOURCE["split_partition_sha256"]:
        raise RuntimeError("split partition hash drifted")
    if task.label_artifact_sha256 != C.SOURCE["label_artifact_sha256"]:
        raise RuntimeError("label artifact hash drifted")
    return task


def make_plan40(task: Any, *, seed: int = C.SEED) -> list[dict[str, Any]]:
    """The complete 40-epoch order plan (one plan per run seed group)."""
    plan = LD.make_order_plan(task.train_rows, batch_size=C.BATCH_SIZE, seed=int(seed), epochs=C.TOTAL_EPOCHS)
    if len(plan) != C.TOTAL_EPOCHS:
        raise RuntimeError(f"order plan has {len(plan)} epochs, expected {C.TOTAL_EPOCHS}")
    return plan


def plan_summary(plan: Sequence[Mapping[str, Any]], task: Any) -> dict[str, Any]:
    summary = LD.order_summary(plan, n_train_rows=int(task.train_rows.size))
    isolation = LD.assert_order_isolation(plan, task.train_rows, task.valid_rows, task.test_rows)
    if not isolation["passed"]:
        raise RuntimeError(f"order isolation gate failed: {isolation}")
    return {"summary": summary, "isolation": isolation}


def plan_prefix_check(plan: Sequence[Mapping[str, Any]], prefix_len: int) -> dict[str, Any]:
    rows = [p["row_order_sha256"] for p in plan[: int(prefix_len)]]
    return {
        "prefix_len": int(prefix_len),
        "prefix_hashes": rows,
        "prefix_sha256": C.sha256_bytes("|".join(rows).encode()),
        "global_plan_sha256": plan[0]["global_plan_sha256"],
    }


def task_counts(task: Any) -> dict[str, Any]:
    """Per-task Train/Valid observed, positive and negative counts (labels only)."""
    y_tr = np.asarray(task.labels[task.train_rows], dtype=np.float64)
    y_va = np.asarray(task.labels[task.valid_rows], dtype=np.float64)
    out = {"train": [], "valid": []}
    for name, y in (("train", y_tr), ("valid", y_va)):
        obs = np.isfinite(y)
        pos = ((y == 1.0) & obs).sum(axis=0)
        neg = ((y == 0.0) & obs).sum(axis=0)
        out[name] = {
            "observed": [int(x) for x in obs.sum(axis=0)],
            "positive": [int(x) for x in pos],
            "negative": [int(x) for x in neg],
        }
    return out


def _array_sha(arr: np.ndarray) -> str:
    return C.sha256_bytes(np.ascontiguousarray(arr, dtype=np.float32).tobytes())


def evaluate_validation(model: torch.nn.Module, task: Any, device: torch.device, *, reason: str = "") -> dict[str, Any]:
    logits = BM.predict_logits(model, task.cache, task.valid_rows, device, batch_size=256)
    y = np.asarray(task.labels[task.valid_rows])
    probs = BM.sigmoid(logits)
    report = MET.validation_report(y, probs)
    report.update(
        {
            "reason": reason,
            "valid_bce": float(MET.masked_bce_numpy(logits, y)),
            "valid_logits_sha256": _array_sha(logits),
            "valid_probs_sha256": _array_sha(probs),
            "valid_logits_shape": list(logits.shape),
            "evaluated_rows": int(len(task.valid_rows)),
            "bce_and_ap_are_nan_masked": True,
        }
    )
    return report


def evaluate_train(model: torch.nn.Module, task: Any, device: torch.device) -> dict[str, Any]:
    """Full-Train official macro AP and per-task AP at a pre-registered final epoch."""
    t0 = time.perf_counter()
    logits = BM.predict_logits(model, task.cache, task.train_rows, device, batch_size=512)
    y = np.asarray(task.labels[task.train_rows])
    probs = BM.sigmoid(logits)
    report = MET.validation_report(y, probs)
    report.update(
        {
            "reason": "fixed_final_epoch_train_eval",
            "train_bce": float(MET.masked_bce_numpy(logits, y)),
            "train_logits_sha256": _array_sha(logits),
            "evaluated_rows": int(len(task.train_rows)),
            "seconds": float(time.perf_counter() - t0),
        }
    )
    return report


def predict_rows(
    model: torch.nn.Module, task: Any, rows: np.ndarray, device: torch.device, *, batch_size: int = 256
) -> np.ndarray:
    return BM.predict_logits(model, task.cache, rows, device, batch_size=batch_size)


__all__ = [
    "PinnedStager",
    "evaluate_train",
    "evaluate_validation",
    "load_task",
    "make_plan40",
    "plan_prefix_check",
    "plan_summary",
    "predict_rows",
    "task_counts",
]
