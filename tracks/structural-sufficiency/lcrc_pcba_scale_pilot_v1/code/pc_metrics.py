"""Masked BCE, official OGB AP and an independent AP recomputation.

The official ``ogb.graphproppred.Evaluator("ogbg-molpcba")`` is the *only*
selection metric.  The independent recomputation exists to prove the official
number is reproducible and that missing labels are ignored, not to replace it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def masked_bce_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """BCEWithLogits over observed entries only; NaN targets never enter."""
    if logits.shape != y.shape:
        raise ValueError(f"BCE logits {tuple(logits.shape)} != target {tuple(y.shape)}")
    mask = torch.isfinite(y)
    if not bool(mask.any()):
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[mask], y[mask])


def masked_bce_numpy(logits: np.ndarray, y: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(y)
    if not mask.any():
        return 0.0
    z = logits[mask]
    t = y[mask]
    # BCE(z,t) = max(z,0) - z*t + log1p(exp(-|z|))
    return float(np.mean(np.maximum(z, 0.0) - z * t + np.log1p(np.exp(-np.abs(z)))))


def per_task_ap(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    """Task-wise AP with the official inclusion rule and NaN masking."""
    from sklearn.metrics import average_precision_score

    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.shape != y_score.shape or y_true.ndim != 2:
        raise ValueError(f"AP arrays {y_true.shape} / {y_score.shape}")
    per_task: list[float | None] = []
    observed: list[int] = []
    positives: list[int] = []
    negatives: list[int] = []
    for t in range(y_true.shape[1]):
        labeled = y_true[:, t] == y_true[:, t]
        yt = y_true[labeled, t]
        n_pos = int((yt == 1.0).sum())
        n_neg = int((yt == 0.0).sum())
        observed.append(int(labeled.sum()))
        positives.append(n_pos)
        negatives.append(n_neg)
        if n_pos > 0 and n_neg > 0:
            per_task.append(float(average_precision_score(yt, y_score[labeled, t])))
        else:
            per_task.append(None)
    return {
        "per_task": per_task,
        "observed": observed,
        "positive": positives,
        "negative": negatives,
        "n_tasks": int(y_true.shape[1]),
        "n_tasks_with_observed_labels": int(sum(1 for x in observed if x > 0)),
        "n_tasks_scored": int(sum(1 for x in per_task if x is not None)),
        "mean_over_scored_tasks": _mean_finite(per_task),
    }


def _mean_finite(values) -> float:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not vals:
        raise ValueError("no finite AP value")
    return float(np.mean(vals))


def official_ap(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    """Official OGB evaluator for ``ogbg-molpcba`` (Validation AP)."""
    from ogb.graphproppred import Evaluator

    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.shape != y_score.shape or y_true.ndim != 2:
        raise ValueError(f"OGB metric arrays {y_true.shape} / {y_score.shape}")
    if y_true.shape[1] != 128:
        raise ValueError(f"expected 128 tasks, got {y_true.shape[1]}")
    score = Evaluator(name="ogbg-molpcba").eval({"y_true": y_true, "y_pred": y_score})
    value = float(score["ap"])
    if not np.isfinite(value):
        raise ValueError(f"official evaluator returned non-finite AP {value}")
    return {
        "value": value,
        "source": "OGB official Evaluator(ogbg-molpcba)",
        "official": True,
        "metric": "ap",
    }


def ap_agreement(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    """Official AP vs an independent per-task recomputation."""
    official = official_ap(y_true, y_score)
    independent = per_task_ap(y_true, y_score)
    delta = abs(float(official["value"]) - float(independent["mean_over_scored_tasks"]))
    return {
        "official_ap": float(official["value"]),
        "independent_ap": float(independent["mean_over_scored_tasks"]),
        "abs_delta": float(delta),
        "n_tasks_scored": int(independent["n_tasks_scored"]),
        "n_tasks_with_observed_labels": int(independent["n_tasks_with_observed_labels"]),
        "official_source": official["source"],
        "independent_source": "sklearn average_precision_score per task with NaN masking",
        "missing_label_policy": "NaN rows are excluded per task; tasks without both classes are skipped by both paths",
    }


def validation_report(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    """Full per-epoch Validation record: official AP + task coverage."""
    official = official_ap(y_true, y_score)
    task = per_task_ap(y_true, y_score)
    return {
        "ap": float(official["value"]),
        "ap_source": official["source"],
        "per_task_ap": task["per_task"],
        "observed": task["observed"],
        "positive": task["positive"],
        "negative": task["negative"],
        "n_tasks_with_observed_labels": task["n_tasks_with_observed_labels"],
        "n_tasks_scored": task["n_tasks_scored"],
        "n_tasks": task["n_tasks"],
    }


__all__ = [
    "ap_agreement",
    "masked_bce_loss",
    "masked_bce_numpy",
    "official_ap",
    "per_task_ap",
    "validation_report",
]
