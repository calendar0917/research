"""Independent, unit-tested evaluation metrics in original logD units."""

from __future__ import annotations

from typing import Mapping

import numpy as np


def _pair(pred, y) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    t = np.asarray(y, dtype=np.float64).reshape(-1)
    if p.shape != t.shape:
        raise ValueError(f"prediction shape {p.shape} != target shape {t.shape}")
    if p.size == 0:
        raise ValueError("empty metric input")
    if not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError("non-finite prediction or target")
    return p, t


def rmse(pred, y) -> float:
    p, t = _pair(pred, y)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def mae(pred, y) -> float:
    p, t = _pair(pred, y)
    return float(np.mean(np.abs(p - t)))


def r2(pred, y) -> float:
    p, t = _pair(pred, y)
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - np.mean(t)) ** 2))
    if ss_tot == 0.0:
        raise ValueError("R^2 is undefined for a constant target")
    return float(1.0 - ss_res / ss_tot)


def all_metrics(pred, y) -> dict[str, float]:
    return {"rmse": rmse(pred, y), "mae": mae(pred, y), "r2": r2(pred, y)}


def metrics_payload(pred, y, *, split: str, units: str = "raw_logD") -> Mapping[str, float | int | str]:
    values = all_metrics(pred, y)
    return {**values, "n": int(np.asarray(y).size), "split": split, "units": units}


__all__ = ["rmse", "mae", "r2", "all_metrics", "metrics_payload"]
