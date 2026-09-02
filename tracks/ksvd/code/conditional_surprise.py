"""Label-free cross-view residuals for chemistry-aware graph patches.

The common part of a structural patch and its chemical patch is often already
available in the raw representation.  This module isolates the part that one
view cannot linearly predict from the other.  It deliberately uses no graph
label: the predictors and all later dictionaries are fitted only on training
patch pairs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge


@dataclass(frozen=True)
class CrossViewPredictors:
    """Two train-fold-only maps between paired structural and chemical views."""

    structure_to_chemistry: Ridge
    chemistry_to_structure: Ridge


def fit_cross_view_predictors(
    structure: np.ndarray,
    chemistry: np.ndarray,
    *,
    alpha: float,
) -> CrossViewPredictors:
    """Fit the two regularized maps on columns of paired patch data."""
    S, C = np.asarray(structure, dtype=np.float64), np.asarray(chemistry, dtype=np.float64)
    if S.ndim != 2 or C.ndim != 2 or S.shape[1] != C.shape[1]:
        raise ValueError("structure and chemistry must be paired column matrices")
    if len(S.T) < 2:
        raise ValueError("at least two patch pairs are required")
    if alpha < 0 or not np.isfinite(alpha):
        raise ValueError("alpha must be a finite non-negative number")
    s_to_c = Ridge(alpha=float(alpha), fit_intercept=True).fit(S.T, C.T)
    c_to_s = Ridge(alpha=float(alpha), fit_intercept=True).fit(C.T, S.T)
    return CrossViewPredictors(s_to_c, c_to_s)


def cross_view_residuals(
    structure: np.ndarray,
    chemistry: np.ndarray,
    predictors: CrossViewPredictors,
) -> tuple[np.ndarray, np.ndarray]:
    """Return structural and chemical information unexplained by the other view."""
    S, C = np.asarray(structure, dtype=np.float64), np.asarray(chemistry, dtype=np.float64)
    if S.ndim != 2 or C.ndim != 2 or S.shape[1] != C.shape[1]:
        raise ValueError("structure and chemistry must be paired column matrices")
    predicted_c = predictors.structure_to_chemistry.predict(S.T).T
    predicted_s = predictors.chemistry_to_structure.predict(C.T).T
    return S - predicted_s, C - predicted_c


def residual_error(
    residual_structure: np.ndarray,
    residual_chemistry: np.ndarray,
) -> dict[str, float]:
    """Per-coordinate mean squared residual, separately and jointly."""
    Rs = np.asarray(residual_structure, dtype=np.float64)
    Rc = np.asarray(residual_chemistry, dtype=np.float64)
    if Rs.ndim != 2 or Rc.ndim != 2 or Rs.shape[1] != Rc.shape[1]:
        raise ValueError("residual matrices must have paired columns")
    s = float(np.mean(Rs**2))
    c = float(np.mean(Rc**2))
    return {"structure_mse": s, "chemistry_mse": c, "mean_mse": 0.5 * (s + c)}


def normalize_residual_columns(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit-normalize residual directions while retaining their original norms."""
    X = np.asarray(values, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("values must be a column matrix")
    norms = np.linalg.norm(X, axis=0)
    return X / np.maximum(norms[None, :], 1e-12), norms
