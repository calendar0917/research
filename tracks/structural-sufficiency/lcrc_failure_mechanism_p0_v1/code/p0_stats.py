"""Metrics, tail definitions, quartiles and scaffold-cluster bootstrap.

All inference in this round treats the Bemis--Murcko scaffold as the resampling
unit.  No molecule-independent interval is used to narrow any interval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from p0_common import (
    BOOTSTRAP_SEED,
    CI_PERCENTILES,
    LOW_SUPPORT_N,
    N_BOOT,
    TOP_FRACTION,
)


# ---------------------------------------------------------------------------
# point metrics
# ---------------------------------------------------------------------------


def squared_error(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    if y.shape != pred.shape:
        raise ValueError(f"y/pred shape mismatch {y.shape} vs {pred.shape}")
    return (pred - y) ** 2


def absolute_error(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(pred, dtype=np.float64).reshape(-1) - np.asarray(y, dtype=np.float64).reshape(-1))


def sse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sum(squared_error(y, pred)))


def mse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(squared_error(y, pred)))


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(math.sqrt(mse(y, pred)))


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(absolute_error(y, pred)))


def p90_abs(y: np.ndarray, pred: np.ndarray) -> float:
    """90th percentile absolute error, numpy linear interpolation (documented)."""
    return float(np.percentile(absolute_error(y, pred), 90.0))


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    errors = absolute_error(y, pred)
    sq = squared_error(y, pred)
    n = int(len(np.asarray(y).reshape(-1)))
    return {
        "n": n,
        "sse": float(np.sum(sq)),
        "mse": float(np.mean(sq)),
        "rmse": float(math.sqrt(float(np.mean(sq)))),
        "mae": float(np.mean(errors)),
        "p90_abs": float(np.percentile(errors, 90.0)),
        "mean_residual": float(np.mean(np.asarray(pred).reshape(-1) - np.asarray(y).reshape(-1))),
    }


def metrics_from_errors(errors: np.ndarray) -> dict[str, float]:
    """Same metric semantics, taking a signed error vector (pred - y)."""
    errors = np.asarray(errors, dtype=np.float64).reshape(-1)
    sq = errors**2
    return {
        "n": int(len(errors)),
        "sse": float(np.sum(sq)),
        "mse": float(np.mean(sq)),
        "rmse": float(math.sqrt(float(np.mean(sq)))),
        "mae": float(np.mean(np.abs(errors))),
        "p90_abs": float(np.percentile(np.abs(errors), 90.0)),
        "mean_residual": float(np.mean(errors)),
    }


def top_k_count(n: int, fraction: float = TOP_FRACTION) -> int:
    """Pre-declared tail size: round(0.10 * n), at least one molecule."""
    return max(1, int(round(float(fraction) * int(n))))


def top_error_indices(abs_error: np.ndarray, fraction: float = TOP_FRACTION) -> np.ndarray:
    """Indices of the largest ``k`` absolute errors.

    Ties are broken by ascending row position (``np.argsort(kind="stable")``),
    which is deterministic for a fixed row order and never used for selection.
    """
    values = np.asarray(abs_error, dtype=np.float64).reshape(-1)
    k = top_k_count(len(values), fraction)
    order = np.argsort(-values, kind="stable")
    return order[:k]


# ---------------------------------------------------------------------------
# quartiles / correlations
# ---------------------------------------------------------------------------


def quartile_bins(values: Sequence[float], n_bins: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Fixed sample quartile bins.

    Boundaries are the empirical quantiles of the *validation sample* of the
    feature itself (label-free).  A value equal to a boundary falls into the
    higher bin index (``side="right"``); this is stated in the report.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = np.quantile(values, qs)
    labels = np.searchsorted(edges, values, side="right")
    return labels.astype(np.int64), edges


def spearman(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    """Spearman rank correlation with explicit LOW_SUPPORT/undefined handling."""
    from scipy import stats

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("spearman inputs differ in length")
    n = int(len(x))
    if n < LOW_SUPPORT_N:
        return {"n": n, "rho": None, "p": None, "status": "LOW_SUPPORT"}
    if len(np.unique(x)) < 3 or len(np.unique(y)) < 3:
        return {"n": n, "rho": None, "p": None, "status": "UNDEFINED_CONSTANT"}
    rho, p = stats.spearmanr(x, y)
    rho = float(rho)
    p = float(p)
    if not math.isfinite(rho):
        return {"n": n, "rho": None, "p": None, "status": "UNDEFINED_NONFINITE"}
    return {"n": n, "rho": rho, "p": p, "status": "OK"}


# ---------------------------------------------------------------------------
# deterministic scaffold folds
# ---------------------------------------------------------------------------


def deterministic_scaffold_folds(scaffolds: Sequence[str], n_splits: int = 5) -> np.ndarray:
    """Assign whole scaffold groups to folds without depending on row order.

    Unique scaffolds are ordered by descending molecule count then by the
    scaffold string and greedily placed into the currently smallest fold (ties
    to the lower fold index).  The assignment is a pure function of the
    scaffold multiset, so shuffling rows cannot change the folds.
    """
    scaffolds = [str(s) for s in scaffolds]
    counts: dict[str, int] = {}
    for s in scaffolds:
        counts[s] = counts.get(s, 0) + 1
    order = sorted(counts, key=lambda s: (-counts[s], s))
    totals = [0] * n_splits
    fold_of: dict[str, int] = {}
    for scaffold in order:
        fold = min(range(n_splits), key=lambda i: (totals[i], i))
        fold_of[scaffold] = fold
        totals[fold] += counts[scaffold]
    return np.asarray([fold_of[s] for s in scaffolds], dtype=np.int64)


# ---------------------------------------------------------------------------
# scaffold-cluster bootstrap
# ---------------------------------------------------------------------------


@dataclass
class ClusterIndex:
    """Grouping of molecules by scaffold, used as the bootstrap unit."""

    labels: np.ndarray
    cluster_names: np.ndarray
    ids: np.ndarray
    sizes: np.ndarray
    n: int
    n_clusters: int

    @classmethod
    def from_values(cls, values: Sequence[str]) -> "ClusterIndex":
        names, ids = np.unique(np.asarray([str(v) for v in values]), return_inverse=True)
        sizes = np.bincount(ids, minlength=len(names))
        return cls(
            labels=np.asarray(values, dtype=object),
            cluster_names=names,
            ids=ids.astype(np.int64),
            sizes=sizes.astype(np.int64),
            n=int(len(ids)),
            n_clusters=int(len(names)),
        )

    def sample_positions(self, rng: np.random.Generator) -> np.ndarray:
        """Cluster bootstrap: draw ``n_clusters`` clusters with replacement.

        The returned index vector may contain a molecule several times (once per
        time its cluster is drawn) and may omit molecules; that is the correct
        cluster-level resampling distribution.
        """
        drawn = rng.integers(0, self.n_clusters, size=self.n_clusters)
        counts = np.bincount(drawn, minlength=self.n_clusters)
        weights = counts[self.ids]
        return np.repeat(np.arange(self.n, dtype=np.int64), weights)

    def profile(self) -> dict[str, Any]:
        sizes = self.sizes.astype(float)
        max_share = float(sizes.max() / self.n) if self.n else float("nan")
        low = bool(self.n_clusters < LOW_SUPPORT_N or max_share > 0.10)
        return {
            "n": self.n,
            "n_clusters": self.n_clusters,
            "median_cluster_size": float(np.median(sizes)),
            "max_cluster_size": int(sizes.max()),
            "max_cluster_molecule_share": max_share,
            "singleton_clusters": int(np.sum(sizes == 1)),
            "low_cluster_support": low,
            "low_cluster_support_rule": (
                "n_clusters < 30 OR largest scaffold cluster > 10% of evaluation molecules"
            ),
        }


def cluster_bootstrap(
    index: ClusterIndex,
    compute: Callable[[np.ndarray], Mapping[str, float]],
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, np.ndarray]:
    """Run a cluster bootstrap; ``compute`` returns named scalar statistics."""
    rng = np.random.default_rng(int(seed))
    draws: dict[str, list[float]] = {}
    for _ in range(int(n_boot)):
        sample = index.sample_positions(rng)
        out = compute(sample)
        for key, value in out.items():
            draws.setdefault(key, []).append(float(value))
    return {key: np.asarray(values, dtype=np.float64) for key, values in draws.items()}


def bootstrap_summary(
    point: Mapping[str, float], draws: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, values in draws.items():
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        entry: dict[str, Any] = {
            "point": float(point[key]) if key in point else None,
            "n_boot": int(len(np.asarray(values))),
            "n_finite": int(len(finite)),
        }
        if len(finite):
            lo, hi = np.percentile(finite, CI_PERCENTILES)
            entry.update(
                {
                    "mean": float(np.mean(finite)),
                    "sd": float(np.std(finite, ddof=0)),
                    "ci_low": float(lo),
                    "ci_high": float(hi),
                    "ci_percentiles": list(CI_PERCENTILES),
                    "frac_gt_zero": float(np.mean(finite > 0.0)),
                }
            )
        else:
            entry.update(
                {
                    "mean": None,
                    "sd": None,
                    "ci_low": None,
                    "ci_high": None,
                    "frac_gt_zero": None,
                }
            )
        summary[key] = entry
    return summary


def sign_agreement(residuals: np.ndarray) -> dict[str, float]:
    """Residual sign consistency across seeds for one molecule.

    ``residuals`` has one entry per seed (``pred - y``).  Positive and negative
    counts are compared; exact zeros are counted in neither sign.
    """
    values = np.asarray(residuals, dtype=np.float64).reshape(-1)
    n = int(len(values))
    pos = int(np.sum(values > 0))
    neg = int(np.sum(values < 0))
    zero = int(np.sum(values == 0))
    majority = max(pos, neg)
    return {
        "n_seeds": n,
        "n_positive": pos,
        "n_negative": neg,
        "n_zero": zero,
        "sign_agreement": float(majority / n) if n else float("nan"),
        "sign_agreement_count": majority,
        "all_same_sign": bool(zero == 0 and (pos == n or neg == n)),
    }


def overlap_summary(sets: Sequence[Sequence[int]], universe_size: int) -> dict[str, Any]:
    """Pairwise overlap of tail index sets across seeds."""
    sets = [set(int(x) for x in s) for s in sets]
    k = len(sets)
    pair_rows = []
    for i in range(k):
        for j in range(i + 1, k):
            inter = len(sets[i] & sets[j])
            union = len(sets[i] | sets[j])
            pair_rows.append(
                {
                    "seed_a": i,
                    "seed_b": j,
                    "intersection": inter,
                    "jaccard": float(inter / union) if union else float("nan"),
                    "overlap_of_k": float(inter / max(1, min(len(sets[i]), len(sets[j])))),
                }
            )
    counts = np.zeros(universe_size, dtype=np.int64)
    for s in sets:
        idx = np.fromiter(s, dtype=np.int64)
        if len(idx):
            counts[idx] += 1
    return {
        "k_sets": k,
        "set_sizes": [len(s) for s in sets],
        "pairwise": pair_rows,
        "n_in_all_sets": int(np.sum(counts == k)),
        "n_in_at_least_two": int(np.sum(counts >= 2)),
        "n_in_exactly_one": int(np.sum(counts == 1)),
        "appearance_counts": counts.tolist(),
    }


def p90(values: Sequence[float]) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), 90.0))
