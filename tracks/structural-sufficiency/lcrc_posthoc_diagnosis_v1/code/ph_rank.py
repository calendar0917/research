"""Rank-error decomposition for binary ROC-AUC."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _check(scores: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    if s.size != yy.size:
        raise ValueError("score/label length mismatch")
    if not np.isin(yy, (0.0, 1.0)).all():
        raise ValueError("binary labels required")
    return s, yy


def positive_rank_losses(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-positive mean error against all negatives (ties count 0.5)."""
    s, yy = _check(scores, y)
    pos = s[yy == 1]
    neg = np.sort(s[yy == 0])
    if pos.size == 0 or neg.size == 0:
        raise ValueError("both classes required")
    right = np.searchsorted(neg, pos, side="right")
    left = np.searchsorted(neg, pos, side="left")
    greater = neg.size - right
    equal = right - left
    return (greater + 0.5 * equal) / neg.size


def negative_rank_losses(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-negative mean error of positives ranking above it (ties count 0.5)."""
    s, yy = _check(scores, y)
    pos = np.sort(s[yy == 1])
    neg = s[yy == 0]
    if pos.size == 0 or neg.size == 0:
        raise ValueError("both classes required")
    right = np.searchsorted(pos, neg, side="right")
    left = np.searchsorted(pos, neg, side="left")
    # error for a negative = positives that rank BELOW it (ties half)
    less = left
    equal = right - left
    return (less + 0.5 * equal) / pos.size


def per_sample_rank_loss(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    s, yy = _check(scores, y)
    out = np.empty_like(s, dtype=np.float64)
    out[yy == 1] = positive_rank_losses(s, yy)
    out[yy == 0] = negative_rank_losses(s, yy)
    return out


def rank_summary(scores: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    s, yy = _check(scores, y)
    pos = positive_rank_losses(s, yy)
    neg = negative_rank_losses(s, yy)
    auc = float(roc_auc_score(yy, s))
    pr = float(average_precision_score(yy, s))
    identity = 1.0 - float(pos.mean())
    return {
        "n": int(yy.size),
        "n_pos": int((yy == 1).sum()),
        "n_neg": int((yy == 0).sum()),
        "positive_rate": float((yy == 1).mean()),
        "positive_rank_loss_mean": float(pos.mean()),
        "negative_rank_loss_mean": float(neg.mean()),
        "roc_auc": auc,
        "pr_auc": pr,
        "one_minus_positive_rank_loss": float(identity),
        "rank_identity_abs_error": abs(identity - auc),
        "rank_identity_ok": bool(abs(identity - auc) <= 1e-12),
    }
