"""Ranking / similarity statistics used by P1 and P3 (pure numpy, no torch)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import rankdata


def average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based average ranks; ties get the mean rank (scipy convention)."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    return rankdata(x, method="average")


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    """Spearman rho = Pearson correlation of average ranks.

    Returns ``None`` when either vector is constant (rho undefined) or shorter
    than three points.
    """
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    b = np.asarray(y, dtype=np.float64).reshape(-1)
    if a.size != b.size:
        raise ValueError(f"spearman length mismatch {a.size} != {b.size}")
    if a.size < 3:
        return None
    ra = average_ranks(a)
    rb = average_ranks(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt(np.sum(ra * ra) * np.sum(rb * rb)))
    if denom <= 0.0:
        return None
    return float(np.sum(ra * rb) / denom)


def descending_order(logits: np.ndarray, rows: Sequence[int]) -> np.ndarray:
    """Indices sorted by logit descending; ties broken by ascending official row id."""
    x = np.asarray(logits, dtype=np.float64).reshape(-1)
    r = np.asarray(rows, dtype=np.int64).reshape(-1)
    if x.size != r.size:
        raise ValueError("logits/rows length mismatch")
    return np.lexsort((r, -x))


def top_k_indices(logits: np.ndarray, rows: Sequence[int], k: int) -> np.ndarray:
    """Positions (into ``logits``) of the deterministic Top-K, ``k`` in [0, n]."""
    n = int(np.asarray(logits).size)
    k = int(k)
    if k < 0 or k > n:
        raise ValueError(f"k={k} outside [0,{n}]")
    if k == 0:
        return np.zeros((0,), dtype=np.int64)
    return descending_order(logits, rows)[:k]


def topk_size(n_observed: int, fraction: float) -> int:
    """``ceil(fraction * n)`` with a floor of 1 when n>0."""
    n = int(n_observed)
    if n <= 0:
        return 0
    import math

    return max(1, int(math.ceil(float(fraction) * n)))


def jaccard(set_a: Sequence[int], set_b: Sequence[int]) -> float | None:
    a = set(int(x) for x in set_a)
    b = set(int(x) for x in set_b)
    if not a and not b:
        return None
    return float(len(a & b) / len(a | b))


def normalized_percentile(logits: np.ndarray) -> np.ndarray:
    """Fraction-of-others-below percentile in [0,1]; 1.0 = highest score."""
    x = np.asarray(logits, dtype=np.float64).reshape(-1)
    n = int(x.size)
    if n == 0:
        return np.zeros((0,), dtype=np.float64)
    if n == 1:
        return np.full((1,), 0.5, dtype=np.float64)
    return (average_ranks(x) - 1.0) / (n - 1.0)


def positive_rank_table(
    reference_logits: np.ndarray, new_logits: np.ndarray, y: np.ndarray
) -> dict[str, float]:
    """Per-positive normalized percentile and ``delta = pct_new - pct_reference``.

    ``delta > 0`` means the new model ranks that true positive higher than the
    reference model does.
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    pct_ref = normalized_percentile(reference_logits)
    pct_new = normalized_percentile(new_logits)
    pos = y == 1.0
    if not np.any(pos):
        return {"n_positive": 0, "median_pct_reference": None, "median_pct_new": None,
                "median_delta_pct": None, "mean_delta_pct": None, "frac_improved": None}
    da = pct_new[pos] - pct_ref[pos]
    return {
        "n_positive": int(pos.sum()),
        "median_pct_reference": float(np.median(pct_ref[pos])),
        "median_pct_new": float(np.median(pct_new[pos])),
        "median_delta_pct": float(np.median(da)),
        "mean_delta_pct": float(np.mean(da)),
        "frac_improved": float(np.mean(da > 0.0)),
    }


def mutual_positive_table(
    idx_a: Sequence[int], idx_b: Sequence[int], y: np.ndarray
) -> dict[str, int]:
    """Positive discovery counts for two deterministic Top-K index sets."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    a = set(int(i) for i in idx_a)
    b = set(int(i) for i in idx_b)
    pos = set(int(i) for i in np.flatnonzero(y == 1.0))
    return {
        "both": int(len(pos & a & b)),
        "only_a": int(len(pos & a - b)),
        "only_b": int(len(pos & b - a)),
        "neither": int(len(pos - a - b)),
        "positives_total": int(len(pos)),
        "pos_in_a": int(len(pos & a)),
        "pos_in_b": int(len(pos & b)),
    }


__all__ = [
    "average_ranks",
    "descending_order",
    "jaccard",
    "mutual_positive_table",
    "normalized_percentile",
    "positive_rank_table",
    "spearman",
    "top_k_indices",
    "topk_size",
]
