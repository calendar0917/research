from __future__ import annotations

import numpy as np
import pytest

import c2_common as C

import r42_metrics as MET


def test_metrics_match_numpy_and_sklearn():
    pred = np.asarray([0.1, -1.2, 2.4, 3.3, 0.0], dtype=np.float64)
    y = np.asarray([0.0, -1.0, 2.0, 3.0, 0.5], dtype=np.float64)
    ours = MET.all_metrics(pred, y)
    ref = {
        "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        "mae": float(np.mean(np.abs(pred - y))),
        "r2": float(1.0 - np.sum((y - pred) ** 2) / np.sum((y - np.mean(y)) ** 2)),
    }
    for key in ref:
        assert abs(ours[key] - ref[key]) <= 1e-15
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    assert abs(ours["rmse"] - np.sqrt(mean_squared_error(y, pred))) <= 1e-14
    assert abs(ours["mae"] - mean_absolute_error(y, pred)) <= 1e-14
    assert abs(ours["r2"] - r2_score(y, pred)) <= 1e-14


@pytest.mark.skipif(not C.PAIRED_BOOTSTRAP.exists(), reason="paired bootstrap has not been run yet")
def test_paired_bootstrap_is_fixed_and_validation_only():
    payload = C.read_json(C.PAIRED_BOOTSTRAP)
    assert payload["resamples"] == 2000
    assert payload["seed"] == 0
    assert payload["n_molecules"] == 420
    assert payload["test_labels_read"] is False
    lo, hi = payload["bootstrap_delta_rmse"]["ci95_percentile"]
    assert lo <= hi
