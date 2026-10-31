"""Denormalisation and metric identities in original logD units."""

from __future__ import annotations

import numpy as np

import x1_common as C
import x1_data as XD

import r42_metrics as MET


def test_denormalisation_is_inverse_of_standardisation():
    norm = {"mean": C.TARGET_MEAN, "scale": C.TARGET_SCALE}
    y = np.asarray([0.0, 1.0, 2.5, 4.0], dtype=np.float64)
    z = (y - norm["mean"]) / norm["scale"]
    assert np.allclose(XD.denormalise(z, norm), y, atol=1e-14, rtol=0.0)


def test_metrics_match_independent_numpy(frozen_data):
    _split, labels, views = frozen_data
    rng = np.random.default_rng(7)
    y = np.asarray(labels.y_valid, dtype=np.float64)
    pred = y + rng.normal(0.0, 0.7, size=y.size)
    expected = {
        "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        "mae": float(np.mean(np.abs(pred - y))),
        "r2": float(1.0 - np.sum((y - pred) ** 2) / np.sum((y - np.mean(y)) ** 2)),
    }
    got = MET.all_metrics(pred, y)
    for key in expected:
        assert abs(got[key] - expected[key]) <= 1.0e-12


def test_rmse_scales_linearly_but_r2_does_not(frozen_data):
    _split, labels, views = frozen_data
    norm = views["target_norm"]
    rng = np.random.default_rng(11)
    y = np.asarray(labels.y_valid, dtype=np.float64)
    pred = y + rng.normal(0.0, 0.5, size=y.size)
    rmse_raw = MET.rmse(pred, y)
    pred_std = (pred - norm["mean"]) / norm["scale"]
    y_std = (y - norm["mean"]) / norm["scale"]
    rmse_std = MET.rmse(pred_std, y_std)
    assert abs(rmse_std * norm["scale"] - rmse_raw) <= 1.0e-12
    assert abs(MET.r2(pred_std, y_std) - MET.r2(pred, y)) <= 1.0e-12


def test_metric_shape_and_finiteness_guard():
    import pytest

    with pytest.raises(ValueError):
        MET.rmse(np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError):
        MET.rmse(np.asarray([1.0, np.nan]), np.asarray([1.0, 2.0]))
