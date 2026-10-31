"""Protocol requirement 2 and 3: metrics match an independent NumPy reference and
a standardised-space RMSE can never be mistaken for raw logD."""

from __future__ import annotations

import numpy as np
import pytest


def _reference(pred, y):
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(y, dtype=np.float64)
    rmse = np.sqrt(np.mean((p - t) ** 2))
    mae = np.mean(np.abs(p - t))
    r2 = 1.0 - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2)
    return float(rmse), float(mae), float(r2)


@pytest.mark.parametrize(
    "pred,y",
    [
        ([0.1, -1.2, 2.4, 3.3], [0.0, -1.0, 2.0, 3.0]),
        ([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]),
        ([-2.5, 0.0, 4.75], [0.25, -0.5, 5.0]),
    ],
)
def test_metrics_match_numpy_reference(pred, y):
    import r42_metrics as MET

    expected = _reference(pred, y)
    assert MET.rmse(pred, y) == pytest.approx(expected[0], abs=1e-12)
    assert MET.mae(pred, y) == pytest.approx(expected[1], abs=1e-12)
    assert MET.r2(pred, y) == pytest.approx(expected[2], abs=1e-12)


def test_metrics_match_sklearn_on_validation(labels):
    import r42_metrics as MET

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    rng = np.random.default_rng(0)
    pred = labels.y_valid + rng.normal(0.0, 0.4, size=labels.y_valid.size)
    assert MET.rmse(pred, labels.y_valid) == pytest.approx(float(np.sqrt(mean_squared_error(labels.y_valid, pred))), abs=1e-12)
    assert MET.mae(pred, labels.y_valid) == pytest.approx(float(mean_absolute_error(labels.y_valid, pred)), abs=1e-12)
    assert MET.r2(pred, labels.y_valid) == pytest.approx(float(r2_score(labels.y_valid, pred)), abs=1e-12)


def test_standardized_rmse_is_not_raw_rmse(labels):
    import r42_common as C
    import r42_data as D
    import r42_metrics as MET
    import r42_native_train as NT

    norm = D.fit_target_scaler(labels.y_train)
    rng = np.random.default_rng(1)
    pred_std = NT.target_standardise(labels.y_valid, norm).numpy() + rng.normal(0.0, 0.25, size=420)
    y_std = NT.target_standardise(labels.y_valid, norm).numpy()
    std_rmse = MET.rmse(pred_std, y_std)
    raw_pred = NT.denormalise(pred_std, norm)
    raw_rmse = MET.rmse(raw_pred, labels.y_valid)
    assert abs(raw_rmse - std_rmse * norm["scale"]) < 1e-9
    # The two numbers must differ for this dataset, and the ratio is the scale.
    assert abs(raw_rmse - std_rmse) > 1e-3
    assert raw_rmse / std_rmse == pytest.approx(norm["scale"], rel=1e-9)
    assert norm["scale"] > 1.0
    assert C.SPLIT_SIZES["valid"] == 420


def test_metric_inputs_are_validated():
    import r42_metrics as MET

    with pytest.raises(ValueError):
        MET.rmse([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        MET.mae([float("nan")], [1.0])
    with pytest.raises(ValueError):
        MET.r2([1.0, 1.0], [2.0, 2.0])
