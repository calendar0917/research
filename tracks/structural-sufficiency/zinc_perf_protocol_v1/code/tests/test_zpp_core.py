"""Lightweight CPU tests for the zinc-perf-protocol-v1 plumbing.

These tests do not load ZINC, do not train, and do not read valid/test labels.
Run with: uv run pytest -q tracks/structural-sufficiency/zinc_perf_protocol_v1/code/tests
"""

from __future__ import annotations

import math

import numpy as np
import torch

import zpp_common as C


def test_mae_and_rmse():
    pred = np.array([1.0, 2.0, 3.0])
    y = np.array([0.0, 2.0, 5.0])
    assert abs(C.mae(pred, y) - 1.0) < 1e-12
    assert abs(C.rmse(pred, y) - math.sqrt(5.0 / 3.0)) < 1e-12


def test_choose_best_keeps_earlier_on_tie():
    curve = [
        {"epoch": 1, "valid_mae": 0.10},
        {"epoch": 2, "valid_mae": 0.08},
        {"epoch": 3, "valid_mae": 0.08},
        {"epoch": 4, "valid_mae": 0.09},
    ]
    best = C.choose_best(curve)
    assert best["epoch"] == 2
    assert best["valid_mae"] == 0.08


def test_choose_best_ignores_nonfinite():
    curve = [
        {"epoch": 1, "valid_mae": float("nan")},
        {"epoch": 2, "valid_mae": float("inf")},
        {"epoch": 3, "valid_mae": 0.5},
    ]
    assert C.choose_best(curve)["epoch"] == 3


def test_cosine_schedule_shape():
    total, warm, base, low = 100, 10, 1e-3, 1e-6
    assert abs(C.cosine_with_warmup_lr(1, total, warm, base, low) - base / warm) < 1e-12
    assert abs(C.cosine_with_warmup_lr(warm, total, warm, base, low) - base) < 1e-12
    assert abs(C.cosine_with_warmup_lr(total, total, warm, base, low) - low) < 1e-12
    vals = [C.cosine_with_warmup_lr(s, total, warm, base, low) for s in range(warm, total + 1)]
    assert all(vals[i] >= vals[i + 1] - 1e-15 for i in range(len(vals) - 1))


def test_isolated_rng_does_not_advance_training_stream():
    torch.manual_seed(7)
    expected = torch.rand(4)
    torch.manual_seed(7)
    with C.isolated_rng():
        torch.rand(100)
    got = torch.rand(4)
    assert torch.equal(expected, got)


def test_checkpoint_selector_reads_no_test_symbol(tmp_path):
    # The selector function has no data path argument at all.
    curve = [{"epoch": 1, "valid_mae": 0.2}, {"epoch": 2, "valid_mae": 0.1}]
    best = C.choose_best(curve)
    assert set(best) == {"epoch", "valid_mae"}


def test_zsrt_variant_param_counts():
    import zpp_zsrt as Z

    for name, expected in (("A_main", 380161), ("B_lr_low", 380161),
                           ("C_wd_high", 380161), ("D_no_rrwp", 377473)):
        variant = Z.load_variant(name)
        _, _, n = Z.build_model(variant, 0, "cpu")
        assert n == expected, (name, n, expected)
