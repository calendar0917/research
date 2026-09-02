from __future__ import annotations

import numpy as np

from .conditional_surprise import (
    cross_view_residuals,
    fit_cross_view_predictors,
    residual_error,
)


def test_true_cross_view_pairs_have_smaller_residual_than_shuffled_pairs() -> None:
    rng = np.random.default_rng(9)
    structure = rng.normal(size=(4, 120))
    transform = rng.normal(size=(3, 4))
    chemistry = transform @ structure + 0.03 * rng.normal(size=(3, 120))
    model = fit_cross_view_predictors(structure[:, :80], chemistry[:, :80], alpha=0.1)
    true_s, true_c = cross_view_residuals(structure[:, 80:], chemistry[:, 80:], model)
    order = np.roll(np.arange(40), 1)
    wrong_s, wrong_c = cross_view_residuals(
        structure[:, 80:], chemistry[:, 80:][:, order], model
    )
    assert residual_error(true_s, true_c)["mean_mse"] < residual_error(wrong_s, wrong_c)["mean_mse"]
