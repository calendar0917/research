from __future__ import annotations

import numpy as np

from tracks.ksvd.experiments.luyin16.cross_center_interaction_terminal import (
    _in_range,
    _official_train_scaffold_splits,
    _strip_search_params,
)


def test_official_train_scaffold_splits_cover_train_without_overlap() -> None:
    archive = {
        "original_indices": np.arange(6, dtype=np.int64),
        "official_train_indices": np.arange(4, dtype=np.int64),
        "official_valid_indices": np.arange(4, 6, dtype=np.int64),
        "fold_0_train_indices": np.asarray([2, 3], dtype=np.int64),
        "fold_0_valid_indices": np.asarray([0, 1], dtype=np.int64),
    }
    splits, metadata = _official_train_scaffold_splits(archive, np.arange(4))
    train, valid = splits[0]
    np.testing.assert_array_equal(train, [2, 3])
    np.testing.assert_array_equal(valid, [0, 1])
    assert metadata[0]["n_train"] == 2
    assert metadata[0]["n_valid"] == 2
    assert np.intersect1d(train, valid).size == 0


def test_search_params_are_filtered_and_checked_against_declared_ranges() -> None:
    params = {
        "n_estimators": 120,
        "max_depth": 3,
        "learning_rate": 0.05,
        "min_child_weight": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_lambda": 10.0,
        "reg_alpha": 0.01,
        "gamma": 0.0,
        "objective": "binary:logistic",
    }
    ranges = {
        "n_estimators": [100, 200],
        "max_depth": [2, 5],
        "learning_rate": [0.01, 0.1],
        "min_child_weight": [2, 20],
        "subsample": [0.5, 1.0],
        "colsample_bytree": [0.5, 1.0],
        "reg_lambda": [1, 20],
        "reg_alpha": [0.001, 1],
        "gamma": [0, 1],
    }
    stripped = _strip_search_params(params)
    assert "objective" not in stripped
    assert _in_range(stripped, ranges)
