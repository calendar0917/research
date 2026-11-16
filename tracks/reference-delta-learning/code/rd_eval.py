"""Aggregation math shared by training-time curves, selected states and reports.

All four arms are expressed against the same frozen reference table and the
same weight matrices:

    N(G) = sum_j w_j y_j
    R(G) = D(G) + sum_j w_j (y_j - D(R_j))
    C(G) = sum_j w_j (y_j + delta(G, R_j))
    delta(G, R) = sigma/2 * [ h(z(G,R)) - h(z(R,G)) ]      (raw units)

``delta`` is a raw-unit difference: it is multiplied by sigma and **never** has
mu added.  ``mu`` appears only when a standardized *absolute* output is mapped
back (the D arm).
"""

from __future__ import annotations

from typing import Any

import numpy as np

import rd_common as C


def rmse_mae(pred_raw: np.ndarray, y_raw: np.ndarray) -> tuple[float, float]:
    return C.rmse(pred_raw, y_raw), C.mae(pred_raw, y_raw)


def reference_targets(task, ref_local: np.ndarray) -> np.ndarray:
    """``y_j`` for every (query, slot) reference; padding slots map to the query itself."""
    ref = np.asarray(ref_local, dtype=np.int64)
    safe = np.clip(ref, 0, None)
    values = task.y["train"][safe]
    return values.astype(np.float64)


def aggregate_n(weights: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    return np.sum(w * np.asarray(y_ref, dtype=np.float64), axis=1)


def aggregate_r(
    d_query: np.ndarray,
    d_ref: np.ndarray,
    weights: np.ndarray,
    y_ref: np.ndarray,
) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    residual = np.asarray(y_ref, dtype=np.float64) - np.asarray(d_ref, dtype=np.float64)
    return np.asarray(d_query, dtype=np.float64) + np.sum(w * residual, axis=1)


def aggregate_c(weights: np.ndarray, y_ref: np.ndarray, delta_raw: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    merged = np.asarray(y_ref, dtype=np.float64) + np.asarray(delta_raw, dtype=np.float64)
    return np.sum(w * merged, axis=1)


def delta_from_margins(h_forward: np.ndarray, h_reverse: np.ndarray, sigma: float) -> np.ndarray:
    """Raw-unit antisymmetric delta; mu is deliberately absent."""
    return 0.5 * float(sigma) * (
        np.asarray(h_forward, dtype=np.float64) - np.asarray(h_reverse, dtype=np.float64)
    )


def raw_from_standardised(margin_std: np.ndarray, scaler: dict[str, float]) -> np.ndarray:
    return np.asarray(margin_std, dtype=np.float64) * float(scaler["std"]) + float(scaler["mean"])


def weighted_rmse(pred: np.ndarray, target: np.ndarray, weight: np.ndarray) -> float:
    diff = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("zero total weight")
    return float(np.sqrt(np.sum(w * diff * diff) / total))


def leave_one_out_delta(
    pred_c: np.ndarray,
    pred_b: np.ndarray,
    y: np.ndarray,
    rmse_fn=C.rmse,
) -> dict[str, Any]:
    """Recompute ``rmse(C) - rmse(B)`` after dropping the largest B-error query."""
    err_b = np.abs(np.asarray(pred_b, dtype=np.float64) - np.asarray(y, dtype=np.float64))
    drop = int(np.argmax(err_b))
    keep = np.ones(err_b.size, dtype=bool)
    keep[drop] = False
    full = float(rmse_fn(pred_c, y) - rmse_fn(pred_b, y))
    reduced = float(rmse_fn(pred_c[keep], y[keep]) - rmse_fn(pred_b[keep], y[keep]))
    flips = bool(np.sign(full) != np.sign(reduced)) if full != 0.0 else bool(reduced != 0.0)
    return {
        "dropped_index": drop,
        "dropped_error_b": float(err_b[drop]),
        "delta_full": full,
        "delta_drop1": reduced,
        "sign_flip": flips,
    }


def paired_bootstrap(
    pred_c: np.ndarray,
    pred_b: np.ndarray,
    y: np.ndarray,
    *,
    n_resamples: int,
    rng: np.random.Generator,
    rmse_fn=C.rmse,
) -> dict[str, float]:
    n = int(np.asarray(y).size)
    base_c = float(rmse_fn(pred_c, y))
    base_b = float(rmse_fn(pred_b, y))
    deltas = np.empty(int(n_resamples), dtype=np.float64)
    for index in range(int(n_resamples)):
        rows = rng.integers(0, n, size=n)
        deltas[index] = rmse_fn(np.asarray(pred_c)[rows], np.asarray(y)[rows]) - rmse_fn(
            np.asarray(pred_b)[rows], np.asarray(y)[rows]
        )
    return {
        "rmse_c": base_c,
        "rmse_b": base_b,
        "delta": base_c - base_b,
        "delta_median": float(np.median(deltas)),
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
        "n_resamples": int(n_resamples),
    }
