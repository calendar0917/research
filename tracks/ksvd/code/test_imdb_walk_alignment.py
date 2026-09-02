"""Self-tests for R0-X objective-alignment diagnostics."""
from __future__ import annotations

import numpy as np

from .imdb_walk_alignment import (
    fit_linear_residuals,
    label_effect_alignment,
    patch_frequency_diagnosis,
    scalar_label_effect,
)


def main() -> int:
    rng = np.random.default_rng(17)
    train_x = rng.normal(size=(80, 3))
    test_x = rng.normal(size=(40, 3))
    coefficients = np.asarray([[1.0, -0.5], [0.2, 0.7], [-0.4, 0.3]])
    train_y = train_x @ coefficients
    test_y = test_x @ coefficients
    train_residual, test_residual, explained = fit_linear_residuals(
        train_x, test_x, train_y, test_y
    )
    assert np.max(np.abs(train_residual)) < 1e-10
    assert np.max(np.abs(test_residual)) < 1e-10
    assert explained["test_explained_fraction"] > 0.999999

    train_labels = np.tile(np.asarray([0, 1]), 40)
    test_labels = np.tile(np.asarray([0, 1]), 20)
    train_values = np.column_stack([train_labels, 2.0 * train_labels]).astype(float)
    test_values = np.column_stack([test_labels, 2.0 * test_labels]).astype(float)
    alignment = label_effect_alignment(
        train_values, test_values, train_labels, test_labels
    )
    assert alignment["train_test_effect_cosine"] > 0.999999
    assert alignment["test_effect_projection_on_train_direction"] > 0.0

    scalar = scalar_label_effect(
        train_labels.astype(float),
        test_labels.astype(float),
        train_labels,
        test_labels,
    )
    assert scalar["effect_sign_consistent"]
    assert scalar["test"]["cohen_d"] > 0.0

    frequency = patch_frequency_diagnosis(
        np.asarray([6, 6, 6, 7, 7, 8]),
        np.asarray([2.0, 2.0, 2.0, 1.0, 1.0, 1.0]),
        np.asarray([1.0, 1.0, 1.0, 0.5, 0.5, 1.0]),
    )
    assert frequency["patch_count"] == 6
    assert frequency["top3_frequency_edge_counts"] == [6, 7, 8]
    assert np.isclose(frequency["top3_frequency_positive_gain_contribution"], 1.0)

    print("imdb_walk_alignment self-tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
