"""Independent NumPy metrics, component-equal pair metrics and bootstrap.

All performance numbers are in original logD units.  The main edit metric is
the component-equal mean absolute pair-difference error (CE-MAE):

    for each pair:  | (pred_b - pred_a) - (y_b - y_a) |
    within a component: mean over that component's pairs in the set
    across components: equal weight per component that has a pair in the set

Bootstrap resamples *components*, never individual pairs as independent units.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

import common as C


def _values(pred_by_row, y_by_row, rows: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(pred_by_row, Mapping):
        pred = np.asarray([float(pred_by_row[int(r)]) for r in rows], dtype=np.float64)
    else:
        arr = np.asarray(pred_by_row, dtype=np.float64)
        pred = np.asarray([float(arr[int(r)]) for r in rows], dtype=np.float64)
    if isinstance(y_by_row, Mapping):
        y = np.asarray([float(y_by_row[int(r)]) for r in rows], dtype=np.float64)
    else:
        arr = np.asarray(y_by_row, dtype=np.float64)
        y = np.asarray([float(arr[int(r)]) for r in rows], dtype=np.float64)
    return pred, y


def regression_metrics(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if pred.size != y.size:
        raise ValueError("prediction/target size mismatch")
    err = pred - y
    ss_res = float((err * err).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "n": int(y.size),
        "rmse": float(math.sqrt(ss_res / max(y.size, 1))),
        "mae": float(np.abs(err).mean()) if y.size else float("nan"),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "bias": float(err.mean()) if y.size else float("nan"),
        "sse": ss_res,
    }


def raw_metrics(pred_standardised: np.ndarray, y: np.ndarray, target_norm: Mapping[str, Any]) -> dict[str, float]:
    pred = np.asarray(pred_standardised, dtype=np.float64) * float(target_norm["scale"]) + float(target_norm["mean"])
    return regression_metrics(pred, y)


def pair_arrays(pred_by_row, y_by_row, pairs: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    if not pairs:
        return {
            "pair_id": np.asarray([], dtype=object),
            "component": np.asarray([], dtype=np.int64),
            "row_a": np.asarray([], dtype=np.int64),
            "row_b": np.asarray([], dtype=np.int64),
            "d_pred": np.asarray([], dtype=np.float64),
            "d_y": np.asarray([], dtype=np.float64),
            "err": np.asarray([], dtype=np.float64),
        }
    rows_a = [int(p["row_a"]) for p in pairs]
    rows_b = [int(p["row_b"]) for p in pairs]
    pa, ya = _values(pred_by_row, y_by_row, rows_a)
    pb, yb = _values(pred_by_row, y_by_row, rows_b)
    d_pred = pb - pa
    d_y = yb - ya
    return {
        "pair_id": np.asarray([str(p["pair_id"]) for p in pairs], dtype=object),
        "component": np.asarray([int(p["eval_component"]) for p in pairs], dtype=np.int64),
        "row_a": np.asarray(rows_a, dtype=np.int64),
        "row_b": np.asarray(rows_b, dtype=np.int64),
        "d_pred": d_pred,
        "d_y": d_y,
        "err": d_pred - d_y,
    }


def component_means(values: np.ndarray, components: np.ndarray) -> dict[int, float]:
    values = np.asarray(values, dtype=np.float64)
    components = np.asarray(components, dtype=np.int64)
    out: dict[int, float] = {}
    for c in sorted(set(int(x) for x in components.tolist())):
        mask = components == c
        out[int(c)] = float(values[mask].mean())
    return out


def component_equal_mean(values: np.ndarray, components: np.ndarray) -> float:
    means = component_means(values, components)
    return float(np.mean(list(means.values()))) if means else float("nan")


def pair_metrics(pred_by_row, y_by_row, pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    arr = pair_arrays(pred_by_row, y_by_row, pairs)
    err = arr["err"]
    abs_err = np.abs(err)
    d_y = arr["d_y"]
    comps = arr["component"]
    return {
        "n_pairs": int(len(pairs)),
        "n_components": int(len(set(int(x) for x in comps.tolist()))) if len(pairs) else 0,
        "ce_mae": component_equal_mean(abs_err, comps) if len(pairs) else float("nan"),
        "ce_rmse": component_equal_mean(err * err, comps) ** 0.5 if len(pairs) else float("nan"),
        "pair_equal_mae": float(abs_err.mean()) if len(pairs) else float("nan"),
        "pair_equal_rmse": float(math.sqrt(float((err * err).mean()))) if len(pairs) else float("nan"),
        "zero_change_reference_mean_abs_dy": float(np.abs(d_y).mean()) if len(pairs) else float("nan"),
        "bias": float(err.mean()) if len(pairs) else float("nan"),
        "arrays": arr,
    }


# ---------------------------------------------------------------------------
# component bootstrap
# ---------------------------------------------------------------------------


def percentile_ci(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    lo = float(np.percentile(vals, 100.0 * (1.0 - level) / 2.0))
    hi = float(np.percentile(vals, 100.0 * (1.0 + level) / 2.0))
    return lo, hi


def bootstrap_component_equal(
    values_by_component: Mapping[int, float],
    *,
    replicates: int = C.BOOTSTRAP_REPLICATES,
    rng: np.random.Generator,
    index: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    comps = np.asarray(sorted(values_by_component), dtype=np.int64)
    vals = np.asarray([values_by_component[int(c)] for c in comps], dtype=np.float64)
    if comps.size == 0:
        return np.full(replicates, np.nan), comps
    if index is None:
        index = rng.integers(0, comps.size, size=(int(replicates), comps.size))
    boots = vals[index].mean(axis=1)
    return boots, comps


def bootstrap_test_rmse(
    rows_by_component: Mapping[int, Sequence[int]],
    sq_err_by_row: Mapping[int, float],
    *,
    replicates: int = C.BOOTSTRAP_REPLICATES,
    rng: np.random.Generator,
    index: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[int, float], dict[int, int]]:
    comps = np.asarray(sorted(rows_by_component), dtype=np.int64)
    sse = np.asarray(
        [float(sum(float(sq_err_by_row[int(r)]) for r in rows_by_component[int(c)])) for c in comps],
        dtype=np.float64,
    )
    counts = np.asarray([int(len(rows_by_component[int(c)])) for c in comps], dtype=np.int64)
    if comps.size == 0:
        return np.full(replicates, np.nan), comps, {}, {}
    if index is None:
        index = rng.integers(0, comps.size, size=(int(replicates), comps.size))
    num = sse[index].sum(axis=1)
    den = counts[index].sum(axis=1)
    boots = np.sqrt(num / np.maximum(den, 1))
    return boots, comps, {int(c): float(s) for c, s in zip(comps, sse)}, {int(c): int(n) for c, n in zip(comps, counts)}


def gain_summary(
    point_reference: float,
    point_candidate: float,
    boots_reference: np.ndarray,
    boots_candidate: np.ndarray,
) -> dict[str, Any]:
    gain = float(point_reference) - float(point_candidate)
    boots = np.asarray(boots_reference, dtype=np.float64) - np.asarray(boots_candidate, dtype=np.float64)
    lo, hi = percentile_ci(boots)
    return {
        "reference": float(point_reference),
        "candidate": float(point_candidate),
        "gain": gain,
        "gain_ci95_lower": lo,
        "gain_ci95_upper": hi,
        "ci_lower_gt_zero": bool(np.isfinite(lo) and lo > 0.0),
        "ci_upper_lt_zero": bool(np.isfinite(hi) and hi < 0.0),
        "ci_crosses_zero": bool((not np.isfinite(lo)) or (not np.isfinite(hi)) or lo <= 0.0 <= hi),
        "n_bootstrap": int(len(boots)),
    }


__all__ = [
    "regression_metrics",
    "raw_metrics",
    "pair_arrays",
    "component_means",
    "component_equal_mean",
    "pair_metrics",
    "percentile_ci",
    "bootstrap_component_equal",
    "bootstrap_test_rmse",
    "gain_summary",
]
