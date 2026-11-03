"""Metrics, pairing and scaffold-cluster bootstrap for the LCRC-P1 round.

All metrics are computed in the original logD units (de-standardised
predictions against raw labels).  Every paired statistic is computed on the
same frozen row set after a strict alignment by row id; nothing here drops a
molecule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from p1_common import (
    CI_PERCENTILES,
    LOW_SUPPORT_N,
    N_BOOT,
    TOP_FRACTION,
    BOOTSTRAP_SEED,
)


# ---------------------------------------------------------------------------
# point metrics
# ---------------------------------------------------------------------------


def squared_error(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    if y.shape != pred.shape:
        raise ValueError(f"shape mismatch: y {y.shape} vs pred {pred.shape}")
    return (pred - y) ** 2


def absolute_error(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(pred, dtype=np.float64).reshape(-1) - np.asarray(y, dtype=np.float64).reshape(-1))


def sse_from_se(se: np.ndarray) -> float:
    return float(np.sum(np.asarray(se, dtype=np.float64)))


def rmse_from_se(se: np.ndarray) -> float:
    se = np.asarray(se, dtype=np.float64)
    return float(math.sqrt(float(np.mean(se))))


def mae_from_ae(ae: np.ndarray) -> float:
    return float(np.mean(np.asarray(ae, dtype=np.float64)))


def p90_abs(ae: np.ndarray) -> float:
    """90th percentile of the absolute error (numpy linear interpolation)."""
    return float(np.percentile(np.asarray(ae, dtype=np.float64), 90.0))


def tail_count(n: int, fraction: float = TOP_FRACTION) -> int:
    """``ceil(fraction * n)`` — the CVaR90 tail size."""
    return int(math.ceil(float(fraction) * int(n) - 1e-12))


def cvar90_ae(ae: np.ndarray) -> float:
    """Mean absolute error of this model's own worst ``ceil(10% n)`` molecules."""
    ae = np.asarray(ae, dtype=np.float64)
    k = tail_count(len(ae))
    if k <= 0:
        return float("nan")
    worst = np.sort(ae)[::-1][:k]
    return float(np.mean(worst))


def top_abs_indices(ae: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` largest absolute errors (descending, stable on ties)."""
    ae = np.asarray(ae, dtype=np.float64)
    order = np.argsort(-ae, kind="stable")
    return order[: int(k)]


def metrics_from_errors(ae: np.ndarray, se: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(ae)),
        "rmse": rmse_from_se(se),
        "mae": mae_from_ae(ae),
        "p90_ae": p90_abs(ae),
        "cvar90_ae": cvar90_ae(ae),
        "sse": sse_from_se(se),
    }


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    se = squared_error(y, pred)
    ae = absolute_error(y, pred)
    out = metrics_from_errors(ae, se)
    out["bias"] = float(np.mean(pred - y))
    return out


# ---------------------------------------------------------------------------
# scaffold-cluster bootstrap
# ---------------------------------------------------------------------------


@dataclass
class ClusterIndex:
    """Scaffold grouping of the evaluation molecules (the bootstrap unit)."""

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
            cluster_names=names,
            ids=ids.astype(np.int64),
            sizes=sizes.astype(np.int64),
            n=int(len(ids)),
            n_clusters=int(len(names)),
        )

    def sample_positions(self, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n_clusters`` clusters with replacement; return row positions."""
        drawn = rng.integers(0, self.n_clusters, size=self.n_clusters)
        counts = np.bincount(drawn, minlength=self.n_clusters)
        weights = counts[self.ids]
        return np.repeat(np.arange(self.n, dtype=np.int64), weights)

    def profile(self) -> dict[str, Any]:
        sizes = self.sizes.astype(float)
        max_share = float(sizes.max() / self.n) if self.n else float("nan")
        return {
            "n_molecules": int(self.n),
            "n_clusters": int(self.n_clusters),
            "median_cluster_size": float(np.median(sizes)) if self.n else None,
            "max_cluster_size": int(self.sizes.max()) if self.n else None,
            "max_cluster_molecule_share": max_share,
            "singleton_clusters": int(np.sum(self.sizes == 1)),
            "low_cluster_support": bool(
                self.n_clusters < LOW_SUPPORT_N or (self.n and max_share > 0.10)
            ),
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
    """Run a cluster bootstrap; ``compute`` maps a position vector to statistics."""
    rng = np.random.default_rng(int(seed))
    draws: dict[str, list[float]] = {}
    for _ in range(int(n_boot)):
        positions = index.sample_positions(rng)
        out = compute(positions)
        for key, value in out.items():
            draws.setdefault(key, []).append(float(value))
    return {key: np.asarray(values, dtype=np.float64) for key, values in draws.items()}


def interval_summary(point: Mapping[str, float], draws: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Percentile interval plus the fraction of replicates on one side of zero."""
    summary: dict[str, Any] = {}
    for key, values in draws.items():
        values = np.asarray(values, dtype=np.float64)
        finite = values[np.isfinite(values)]
        entry: dict[str, Any] = {
            "point": float(point[key]) if key in point else None,
            "n_boot": int(len(values)),
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
                    "frac_lt_zero": float(np.mean(finite < 0.0)),
                }
            )
        else:  # pragma: no cover - defensive
            entry.update(
                {
                    "mean": None,
                    "sd": None,
                    "ci_low": None,
                    "ci_high": None,
                    "frac_gt_zero": None,
                    "frac_lt_zero": None,
                }
            )
        summary[key] = entry
    return summary


# ---------------------------------------------------------------------------
# quantile binning (boundaries always come from Train of the same split)
# ---------------------------------------------------------------------------


def quantile_edges(values: Sequence[float], n_bins: int = 4) -> list[float]:
    """Interior quantile boundaries (``n_bins - 1`` values) of ``values``."""
    arr = np.asarray([v for v in np.asarray(values, dtype=np.float64) if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return []
    qs = [100.0 * i / n_bins for i in range(1, n_bins)]
    return [float(x) for x in np.percentile(arr, qs)]


def bin_index(values: Sequence[float], edges: Sequence[float]) -> np.ndarray:
    """0-based bin index using ``searchsorted`` on frozen interior edges.

    Values below the first edge fall in bin 0, values equal to an edge belong to
    the lower bin, and values above the last edge fall in the last bin.
    """
    arr = np.asarray(values, dtype=np.float64)
    edges = np.asarray(list(edges), dtype=np.float64)
    out = np.searchsorted(edges, arr, side="left")
    out = np.where(np.isfinite(arr), out, -1)
    return out.astype(np.int64)


def bin_labels(values: Sequence[float], edges: Sequence[float], prefix: str) -> list[str]:
    idx = bin_index(values, edges)
    n_bins = len(edges) + 1
    out = []
    for i in idx:
        out.append(f"{prefix}_EMPTY" if i < 0 else f"{prefix}_Q{int(i) + 1}")
    assert n_bins >= 1
    return out


def cvar_point_and_interval(
    ae_a: np.ndarray,
    ae_b: np.ndarray,
    index: ClusterIndex,
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """CVaR90-AE difference (b - a) with a cluster bootstrap interval.

    The tail is recomputed inside every replicate on the resampled molecules,
    so the interval describes the CVaR statistic rather than a fixed tail set.
    """
    point = {"delta_cvar90_ae": cvar90_ae(ae_b) - cvar90_ae(ae_a)}

    def compute(positions: np.ndarray) -> Mapping[str, float]:
        a = np.asarray(ae_a, dtype=np.float64)[positions]
        b = np.asarray(ae_b, dtype=np.float64)[positions]
        return {"delta_cvar90_ae": cvar90_ae(b) - cvar90_ae(a)}

    draws = cluster_bootstrap(index, compute, n_boot=n_boot, seed=seed)
    return interval_summary(point, draws)["delta_cvar90_ae"]


__all__ = [
    "squared_error",
    "absolute_error",
    "sse_from_se",
    "rmse_from_se",
    "mae_from_ae",
    "p90_abs",
    "tail_count",
    "cvar90_ae",
    "top_abs_indices",
    "metrics_from_errors",
    "metrics",
    "ClusterIndex",
    "cluster_bootstrap",
    "interval_summary",
    "quantile_edges",
    "bin_index",
    "bin_labels",
    "cvar_point_and_interval",
]
