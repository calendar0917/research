"""Preprocessing isolation: every scaler is fitted on one training partition only."""

from __future__ import annotations

import numpy as np
import pytest

import nb_common as C
import nb_cv as CV


def _synthetic(n: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    ecfp = (rng.random((n, 16)) < 0.3).astype(np.uint8)
    g = rng.normal(size=(n, C.GLOBAL_DIM)).astype(np.float64)
    g[:, 0] = 0.5  # constant column -> zero variance
    real = rng.random((n, C.N_PAIR_FEATURES))
    null = real * 0.5 + 0.01
    null[:, 3] = 0.0
    real[:, 3] = 0.0  # shared constant column
    return ecfp, g, real, null


def test_g42_scaler_uses_the_training_partition_only():
    ecfp, g, real, null = _synthetic()
    train = np.arange(40)
    pre = CV.fit_preprocessing(train, ecfp, g, real, null)
    assert np.allclose(pre["g_mean"], g[train].mean(axis=0))
    assert np.allclose(pre["g_scale"], np.where(g[train].std(axis=0) > 0, g[train].std(axis=0), 1.0))
    # changing the validation rows must not change the scaler
    g2 = g.copy()
    g2[45:] += 100.0
    pre2 = CV.fit_preprocessing(train, ecfp, g2, real, null)
    assert np.allclose(pre2["g_mean"], pre["g_mean"])
    assert np.allclose(pre2["g_scale"], pre["g_scale"])


def test_pair_scaler_is_shared_by_real_and_null():
    ecfp, g, real, null = _synthetic()
    train = np.arange(40)
    pre = CV.fit_preprocessing(train, ecfp, g, real, null)
    stacked = np.vstack([real[train], null[train]])
    assert np.allclose(pre["pair_mean"], stacked.mean(axis=0))
    assert np.allclose(pre["pair_scale"], np.where(stacked.std(axis=0) > 0, stacked.std(axis=0), 1.0))
    # a REAL-NULL swap of the input slots must reproduce the identical affine map
    X_real = CV.transform(train, pre, "BASE_REAL", ecfp, g, real, null)
    pre_swapped = CV.fit_preprocessing(train, ecfp, g, null, real)
    X_swapped = CV.transform(train, pre_swapped, "BASE_NULL", ecfp, g, null, real)
    assert np.allclose(X_real, X_swapped)


def test_zero_variance_columns_use_denominator_one():
    ecfp, g, real, null = _synthetic()
    pre = CV.fit_preprocessing(np.arange(40), ecfp, g, real, null)
    assert pre["g_zero_variance_columns"] >= 1
    assert pre["pair_zero_variance_columns"] >= 1
    zero_g = np.flatnonzero(g.std(axis=0) == 0)
    assert zero_g.size >= 1
    assert np.all(pre["g_scale"][zero_g] == 1.0)
    zero_pair = np.flatnonzero(np.vstack([real, null]).std(axis=0) == 0)
    assert zero_pair.size >= 1
    assert np.all(pre["pair_scale"][zero_pair] == 1.0)


def test_standardised_blocks_are_clipped():
    ecfp, g, real, null = _synthetic()
    g = g.copy()
    g[0, 1] = 1e6  # extreme outlier in the training partition
    pre = CV.fit_preprocessing(np.arange(40), ecfp, g, real, null)
    X = CV.transform(np.arange(60), pre, "BASE_REAL", ecfp, g, real, null)
    continuous = X[:, 16:]
    assert continuous.max() <= C.CLIP
    assert continuous.min() >= -C.CLIP


def test_ecfp_block_is_not_standardised_and_keeps_binary_values():
    ecfp, g, real, null = _synthetic()
    pre = CV.fit_preprocessing(np.arange(40), ecfp, g, real, null)
    X = CV.transform(np.arange(60), pre, "BASE", ecfp, g, real, null)
    assert set(np.unique(X[:, :16]).tolist()) <= {0.0, 1.0}
    assert np.array_equal(X[:, :16], ecfp.astype(np.float64))


def test_branch_widths_and_block_layout():
    ecfp, g, real, null = _synthetic()
    pre = CV.fit_preprocessing(np.arange(40), ecfp, g, real, null)
    rows = np.arange(20, 30)
    base = CV.transform(rows, pre, "BASE", ecfp, g, real, null)
    with_null = CV.transform(rows, pre, "BASE_NULL", ecfp, g, real, null)
    with_real = CV.transform(rows, pre, "BASE_REAL", ecfp, g, real, null)
    assert base.shape == (10, 16 + C.GLOBAL_DIM)
    assert with_null.shape == (10, 16 + C.GLOBAL_DIM + C.N_PAIR_FEATURES)
    assert with_real.shape == (10, 16 + C.GLOBAL_DIM + C.N_PAIR_FEATURES)
    # the shared baseline part is identical across branches
    assert np.array_equal(base, with_null[:, : base.shape[1]])
    assert np.array_equal(base, with_real[:, : base.shape[1]])
    # NULL and REAL blocks differ
    assert not np.array_equal(with_null[:, base.shape[1] :], with_real[:, base.shape[1] :])
    widths = CV.feature_widths(16)
    assert widths["BASE"] == 16 + C.GLOBAL_DIM
    assert widths["BASE_REAL"] == 16 + C.GLOBAL_DIM + 126


def test_inner_fold_preprocessing_is_fit_on_inner_train_only():
    ecfp, g, real, null = _synthetic()
    inner_train = np.arange(0, 20)
    inner_valid = np.arange(20, 40)
    outer_train = np.arange(0, 40)
    pre_inner = CV.fit_preprocessing(inner_train, ecfp, g, real, null)
    pre_outer = CV.fit_preprocessing(outer_train, ecfp, g, real, null)
    assert pre_inner["n_fit_rows"] == 20 and pre_outer["n_fit_rows"] == 40
    assert not np.allclose(pre_inner["g_mean"], pre_outer["g_mean"])
    # the evaluated rows never enter the fit
    g_shift = g.copy()
    g_shift[inner_valid] += 5.0
    pre_shift = CV.fit_preprocessing(inner_train, ecfp, g_shift, real, null)
    assert np.allclose(pre_shift["g_mean"], pre_inner["g_mean"])


def test_logreg_configuration_is_frozen():
    assert C.LOGREG == {
        "penalty": "l2",
        "solver": "liblinear",
        "class_weight": None,
        "fit_intercept": True,
        "tol": 1e-4,
        "max_iter": 4000,
        "random_state": 20261103,
    }
    assert C.C_GRID == (0.01, 0.1, 1.0)


def test_fit_logreg_records_the_requested_penalty():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    y = (X[:, 0] + 0.3 * rng.normal(size=200) > 0).astype(np.int64)
    model = CV.fit_logreg(X, y, 0.1)
    assert model.get_params()["C"] == 0.1
    assert model.get_params()["solver"] == "liblinear"
    assert model.get_params()["penalty"] == "l2"
    assert model.get_params()["class_weight"] is None
    assert model.get_params()["fit_intercept"] is True
    assert model.get_params()["max_iter"] == 4000
    assert model.n_iter_[0] < 4000
