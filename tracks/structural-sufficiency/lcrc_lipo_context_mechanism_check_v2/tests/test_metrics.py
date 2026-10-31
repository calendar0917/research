from __future__ import annotations

import numpy as np

import m2_common as C
import m2_evaluate as E


def test_paired_bootstrap_matches_direct_metrics():
    rng = np.random.default_rng(7)
    y = rng.normal(2.0, 1.0, size=420)
    ref = y + rng.normal(0.0, 0.5, size=420)
    other = y + rng.normal(0.0, 0.45, size=420)
    boot = E.paired_bootstrap(ref, other, y, n_resamples=200, seed=11)
    assert boot["observed"]["gain_rmse"] == np.sqrt(np.mean((ref - y) ** 2)) - np.sqrt(np.mean((other - y) ** 2))
    assert boot["observed"]["mae_change"] == np.mean(np.abs(other - y)) - np.mean(np.abs(ref - y))
    lower, median, upper = boot["interval"]["gain_rmse"]
    assert lower <= median <= upper
    assert boot["n_resamples"] == 200
    assert boot["seed"] == 11


def test_metric_block_sse_identity():
    rng = np.random.default_rng(13)
    y = rng.normal(2.0, 1.2, size=420)
    pred = y + rng.normal(0.0, 0.4, size=420)
    metrics = E.metric_block(pred, y)
    assert abs(metrics["sse"] - np.sum((pred - y) ** 2)) < 1e-12
    assert abs(metrics["rmse"] - np.sqrt(metrics["sse"] / y.size)) < 1e-12
    assert abs(metrics["mae"] - np.mean(np.abs(pred - y))) < 1e-12
