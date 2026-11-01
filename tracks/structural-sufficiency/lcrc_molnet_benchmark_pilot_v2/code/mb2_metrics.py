"""Metrics for the seven-task round.

Three task families, one implementation each:

``regression``
    RMSE / MAE / R² / bias on raw units (the model trains on standardised
    targets and the metrics are computed after de-standardisation).

``binary``
    logits -> sigmoid -> ROC-AUC (primary), PR-AUC, BCE, accuracy@0.5.  The
    0.5 accuracy is reported for completeness only; no threshold is ever tuned.

``multitask`` (Tox21)
    per-endpoint ROC-AUC and PR-AUC computed **only on observed cells**, plus a
    macro average over the endpoints where both classes are present in that
    split, plus the masked BCE actually used for training.  Missing cells are
    ``NaN`` upstream and never enter a metric or a loss.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


# ---------------------------------------------------------------------------
# regression / binary
# ---------------------------------------------------------------------------


def regression_metrics(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if pred.shape != y.shape:
        raise ValueError("prediction/target shape mismatch")
    err = pred - y
    ss_res = float((err * err).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "n": int(y.size),
        "rmse": float(math.sqrt(ss_res / max(y.size, 1))),
        "mae": float(np.abs(err).mean()),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "bias": float(err.mean()),
    }


def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-logits))


def binary_metrics(logits: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    prob = sigmoid(logits)
    classes = np.unique(y)
    if classes.size != 2:
        raise ValueError(f"ROC-AUC undefined for {classes.size} class(es)")
    return {
        "n": int(y.size),
        "roc_auc": float(roc_auc_score(y, prob)),
        "pr_auc": float(average_precision_score(y, prob)),
        "bce": float(np.mean(np.logaddexp(0.0, logits) - y * logits)),
        "log_loss": float(
            np.mean(-(y * np.log(np.clip(prob, 1e-12, 1.0)) + (1 - y) * np.log(np.clip(1 - prob, 1e-12, 1.0))))
        ),
        "accuracy_at_0.5": float(((prob >= 0.5).astype(np.float64) == y).mean()),
        "positive_rate": float((y == 1).mean()),
        "pred_positive_rate_at_0.5": float((prob >= 0.5).mean()),
        "sigmoid_probability": True,
        "threshold_used_for_selection": None,
    }


# ---------------------------------------------------------------------------
# multitask
# ---------------------------------------------------------------------------


def masked_bce(logits: np.ndarray, y: np.ndarray) -> float:
    """``(per_label_loss * observed).sum() / observed.sum()`` -- the training loss."""
    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if logits.shape != y.shape:
        raise ValueError(f"logit/target shape mismatch {logits.shape} vs {y.shape}")
    mask = np.isfinite(y)
    if mask.sum() == 0:
        return float("nan")
    loss = np.logaddexp(0.0, logits) - np.where(mask, y, 0.0) * logits
    return float((loss * mask).sum() / mask.sum())


def multitask_metrics(logits: np.ndarray, y: np.ndarray, task_names: list[str]) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if logits.shape != y.shape:
        raise ValueError(f"logit/target shape mismatch {logits.shape} vs {y.shape}")
    prob = sigmoid(logits)
    per_task: dict[str, Any] = {}
    definable: list[str] = []
    undefined: dict[str, str] = {}
    for i, name in enumerate(task_names):
        obs = np.isfinite(y[:, i])
        yy = y[obs, i]
        pp = prob[obs, i]
        entry: dict[str, Any] = {
            "n_observed": int(obs.sum()),
            "n_positive": int(yy.sum()),
            "n_negative": int(obs.sum() - yy.sum()),
            "positive_rate": float(yy.mean()) if yy.size else float("nan"),
        }
        if yy.size == 0 or np.unique(yy).size != 2:
            entry["roc_auc"] = None
            entry["pr_auc"] = None
            entry["definable"] = False
            undefined[name] = "no positives" if yy.sum() == 0 else ("no negatives" if yy.sum() == yy.size else "empty")
        else:
            entry["roc_auc"] = float(roc_auc_score(yy, pp))
            entry["pr_auc"] = float(average_precision_score(yy, pp))
            entry["definable"] = True
            definable.append(name)
        per_task[name] = entry
    out: dict[str, Any] = {
        "n": int(y.shape[0]),
        "n_tasks": int(y.shape[1]),
        "per_task": per_task,
        "n_definable_tasks": len(definable),
        "definable_tasks": definable,
        "undefined_tasks": undefined,
        "masked_bce": masked_bce(logits, y),
        "macro_roc_auc": float(np.mean([per_task[t]["roc_auc"] for t in definable])) if definable else None,
        "macro_pr_auc": float(np.mean([per_task[t]["pr_auc"] for t in definable])) if definable else None,
        "missing_policy": "observed cells only; NaN never treated as negative",
        "sigmoid_probability": True,
        "threshold_used_for_selection": None,
    }
    return out


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------


def task_metrics(
    task: str, raw: np.ndarray, y: np.ndarray, target_norm: Mapping[str, Any] | None, task_names: list[str] | None = None
) -> dict[str, Any]:
    if task == "regression":
        if target_norm is None:
            raise ValueError("regression requires a target normaliser")
        pred = np.asarray(raw, dtype=np.float64).reshape(-1) * float(target_norm["scale"]) + float(
            target_norm["mean"]
        )
        return regression_metrics(pred, y)
    if task == "binary":
        return binary_metrics(np.asarray(raw, dtype=np.float64).reshape(-1), y)
    if task == "multitask":
        if not task_names:
            raise ValueError("multitask metrics need the frozen task names")
        return multitask_metrics(raw, y, list(task_names))
    raise ValueError(f"unknown task {task!r}")


def primary_value(task: str, metrics: Mapping[str, Any]) -> float:
    if task == "regression":
        return float(metrics["rmse"])
    if task == "binary":
        return float(metrics["roc_auc"])
    value = metrics["macro_roc_auc"]
    if value is None:
        raise ValueError("macro ROC-AUC is undefined (no definable endpoint)")
    return float(value)


def primary_name(task: str) -> str:
    return {"regression": "rmse", "binary": "roc_auc", "multitask": "macro_roc_auc"}[task]


def better(task: str, candidate: float, incumbent: float | None) -> bool:
    """Strict improvement in the frozen direction (ties keep the earlier epoch)."""
    if incumbent is None:
        return True
    return candidate < incumbent if task == "regression" else candidate > incumbent


# ---------------------------------------------------------------------------
# official OGB evaluator (ogbg-molhiv primary metric)
# ---------------------------------------------------------------------------


def ogb_evaluator_roc_auc(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Primary HIV metric from the official ``ogb`` Evaluator.

    ``y_true``/``y_pred`` are passed as ``[N, 1]`` raw labels / raw scores, as the
    task book requires.  The sklearn value is computed alongside as a
    cross-check; a disagreement is reported, not hidden.
    """
    from ogb.graphproppred import Evaluator

    evaluator = Evaluator(name="ogbg-molhiv")
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1, 1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1, 1)
    official = float(evaluator.eval({"y_true": y_true, "y_pred": y_pred})["rocauc"])
    from sklearn.metrics import roc_auc_score

    local = float(roc_auc_score(y_true.reshape(-1), y_pred.reshape(-1)))
    return {
        "ogb_evaluator": "ogb.graphproppred.Evaluator(name='ogbg-molhiv')",
        "ogb_rocauc": official,
        "sklearn_roc_auc": local,
        "abs_diff": abs(official - local),
        "consistent": bool(abs(official - local) <= 1e-12),
        "y_true_shape": list(y_true.shape),
        "y_pred_shape": list(y_pred.shape),
        "y_pred_is_raw_score": True,
    }


__all__ = [
    "regression_metrics",
    "binary_metrics",
    "sigmoid",
    "masked_bce",
    "multitask_metrics",
    "task_metrics",
    "primary_value",
    "primary_name",
    "better",
    "ogb_evaluator_roc_auc",
]
