"""Metric functions with explicit missing-label and dimension handling."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F


def classification_metrics(logits: np.ndarray, y: np.ndarray, n_classes: int) -> dict[str, Any]:
    logits_t = torch.as_tensor(np.asarray(logits, dtype=np.float32))
    y_t = torch.as_tensor(np.asarray(y, dtype=np.int64))
    if logits_t.ndim != 2 or logits_t.shape[1] != int(n_classes):
        raise ValueError(f"logits shape {tuple(logits_t.shape)} != [N,{n_classes}]")
    if y_t.shape != (logits_t.shape[0],):
        raise ValueError("label shape")
    ce = float(F.cross_entropy(logits_t, y_t, reduction="mean").item())
    pred = logits_t.argmax(dim=1).numpy()
    y_np = y_t.numpy()
    n = int(y_np.size)
    acc = float((pred == y_np).sum() / max(n, 1))
    confusion = np.zeros((int(n_classes), int(n_classes)), dtype=np.int64)
    for t, p in zip(y_np.tolist(), pred.tolist()):
        confusion[int(t), int(p)] += 1
    f1s = []
    for c in range(int(n_classes)):
        tp = float(confusion[c, c])
        fp = float(confusion[:, c].sum() - confusion[c, c])
        fn = float(confusion[c, :].sum() - confusion[c, c])
        denom = 2.0 * tp + fp + fn
        f1s.append((2.0 * tp / denom) if denom > 0 else 0.0)
    return {
        "n": n,
        "cross_entropy": ce,
        "accuracy": acc,
        "macro_f1": float(np.mean(f1s)),
        "per_class_f1": [float(x) for x in f1s],
        "confusion_matrix": confusion.tolist(),
    }


def masked_bce_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if logits.shape != y.shape:
        raise ValueError(f"BCE logits {tuple(logits.shape)} != target {tuple(y.shape)}")
    mask = torch.isfinite(y)
    if not bool(mask.any()):
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[mask], y[mask])


def per_task_roc_auc(y: np.ndarray, prob: np.ndarray) -> list[float | None]:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y, dtype=np.float64)
    prob = np.asarray(prob, dtype=np.float64)
    if y.shape != prob.shape or y.ndim != 2:
        raise ValueError(f"ROC-AUC arrays {y.shape} / {prob.shape}")
    out: list[float | None] = []
    for t in range(y.shape[1]):
        obs = np.isfinite(y[:, t])
        yt = y[obs, t]
        if yt.size == 0 or np.unique(yt).size < 2:
            out.append(None)
            continue
        out.append(float(roc_auc_score(yt, prob[obs, t])))
    return out


def mean_finite(values: list[float | None]) -> float:
    vals = [float(x) for x in values if x is not None and np.isfinite(x)]
    if not vals:
        raise ValueError("no finite metric values")
    return float(np.mean(vals))


def ogb_rocauc(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Official OGB Evaluator, with a recorded per-task fallback if it raises."""
    from ogb.graphproppred import Evaluator

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape or y_true.ndim != 2:
        raise ValueError(f"OGB metric arrays {y_true.shape} / {y_pred.shape}")
    try:
        score = Evaluator(name=name).eval({"y_true": y_true, "y_pred": y_pred})
        value = float(score["rocauc"])
        if not np.isfinite(value):
            raise ValueError(f"official evaluator returned non-finite {value}")
        return {
            "value": value,
            "source": "OGB official Evaluator",
            "official": True,
            "per_task": per_task_roc_auc(y_true, y_pred),
        }
    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
        per_task = per_task_roc_auc(y_true, y_pred)
        value = mean_finite(per_task)
        return {
            "value": value,
            "source": "per-task sklearn ROC-AUC fallback",
            "official": False,
            "official_error": f"{type(exc).__name__}: {exc}",
            "per_task": per_task,
        }


def macro_ap(y: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score

    y = np.asarray(y, dtype=np.float64)
    prob = np.asarray(prob, dtype=np.float64)
    if y.shape != prob.shape or y.ndim != 2:
        raise ValueError(f"AP arrays {y.shape} / {prob.shape}")
    per_task: list[float | None] = []
    for t in range(y.shape[1]):
        obs = np.isfinite(y[:, t])
        yt = y[obs, t]
        if yt.size == 0 or np.unique(yt).size < 2:
            per_task.append(None)
            continue
        per_task.append(float(average_precision_score(yt, prob[obs, t])))
    return {
        "value": mean_finite(per_task),
        "macro_ap": mean_finite(per_task),
        "per_label": per_task,
        "n_labels": int(y.shape[1]),
        "missing_policy": "labels with no observed row or a single class are excluded from the macro mean and recorded as null",
    }


def macro_f1_from_predictions(pred: np.ndarray, y: np.ndarray, n_classes: int) -> float:
    confusion = np.zeros((int(n_classes), int(n_classes)), dtype=np.int64)
    for t, p in zip(np.asarray(y, dtype=np.int64).tolist(), np.asarray(pred, dtype=np.int64).tolist()):
        confusion[int(t), int(p)] += 1
    f1s = []
    for c in range(int(n_classes)):
        tp = float(confusion[c, c])
        fp = float(confusion[:, c].sum() - confusion[c, c])
        fn = float(confusion[c, :].sum() - confusion[c, c])
        denom = 2.0 * tp + fp + fn
        f1s.append((2.0 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(f1s))


__all__ = [
    "classification_metrics",
    "masked_bce_loss",
    "per_task_roc_auc",
    "mean_finite",
    "ogb_rocauc",
    "macro_ap",
    "macro_f1_from_predictions",
]
