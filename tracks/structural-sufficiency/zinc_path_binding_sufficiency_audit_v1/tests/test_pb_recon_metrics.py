"""Metric plumbing used by the audit's reconstruction layer."""

from __future__ import annotations

import numpy as np

import pb_recon as R


def test_standardizer_is_train_only_and_roundtrips():
    rng = np.random.default_rng(0)
    train = rng.normal(3.0, 2.0, size=(500, 4)).astype(np.float32)
    std = R.Standardizer.fit(train)
    z = std.transform(train)
    assert np.abs(z.mean(0)).max() < 1e-5
    assert np.abs(z.std(0) - 1.0).max() < 1e-4
    # valid points are transformed with the *train* statistics
    valid = rng.normal(5.0, 1.0, size=(10, 4)).astype(np.float32)
    zv = std.transform(valid)
    assert np.abs(zv.mean(0) - (5.0 - 3.0) / 2.0).max() < 0.3


def test_block_metrics_identity_and_constant_channel_handling():
    rng = np.random.default_rng(1)
    true = rng.normal(size=(200, 6)).astype(np.float64)
    true[:, 5] = 0.0  # degenerate channel
    blocks = {"a": (0, 3), "b": (3, 6)}
    deg, near = R.degenerate_channels(true.std(0))
    assert bool(deg[5])
    m = R.block_metrics(true.copy(), true, deg | near, blocks, true.std(0))
    assert abs(m["a"]["r2"] - 1.0) < 1e-12
    assert m["b"]["n_degenerate"] == 1
    # a constant predictor must not be credited through the constant channel
    m0 = R.block_metrics(np.zeros_like(true[:, :3]).repeat(2, axis=1)[:, :6] * 0.0,
                         true, deg | near, blocks, true.std(0))
    assert m0["a"]["r2"] <= 0.0


def test_effective_rank_bounds():
    cov = np.diag([4.0, 1.0, 0.0, 0.0])
    r = R.effective_rank(cov)
    assert 1.0 <= r["entropy_rank"] <= 2.0 + 1e-9
    assert 1.0 <= r["participation_ratio"] <= 2.0 + 1e-9
    assert r["top1_share"] > 0.5
