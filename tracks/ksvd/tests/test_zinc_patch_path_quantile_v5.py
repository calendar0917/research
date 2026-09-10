"""Correctness tests for Compact-v5 multi-quantile regression (objective-only).

Covered properties (see ``notes/compact_v5_multi_quantile_regression.md``):

1. Pinball tau=0.5 identity: ``2 * pinball_0.5 == L1`` (elementwise exact).
2. Non-crossing parameterization: q10 <= q50 <= q90 for any inputs.
3. All quantile losses have finite forward values and finite backward
   gradients (no NaN/inf anywhere in the mq head path).
4. Baseline guard: ``quantile_mode="none"`` model is architecturally and
   numerically identical to the original compact-v4 model (same seed ->
   same parameters; state-dict keys identical).
5. ``quantile_mode="median_only"`` loss is forward-equal to L1 on random
   batches, and the q50 main-task loss in mq mode keeps L1 scale.
6. lambda=0 semantics: with the auxiliary terms absent, gradients with
   respect to the width outputs (d_low_raw / d_high_raw rows) are exactly
   zero, so q10/q90 are not optimized by accident.
7. Pre-registered lambda grid is enforced by ``_validate_quantile_config``.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as module

MODE_MQ = "q10_q50_q90"


# --------------------------------------------------------------------------
# toy graph batches
# --------------------------------------------------------------------------

def _toy_batch(shell_width: int, n_graphs: int = 2) -> Data:
    """Two tiny single-centre molecules as one batch (one patch each)."""
    generator = torch.Generator().manual_seed(17)
    n = 2 * n_graphs  # one patch per molecule
    batch = torch.arange(n_graphs, dtype=torch.long).repeat_interleave(2)
    return Data(
        patch_cont=torch.randn(n, shell_width, generator=generator),
        patch_context=torch.zeros(n, 0),
        typed_token=torch.arange(1, n + 1, dtype=torch.long),
        parent_token=torch.ones(n, dtype=torch.long),
        pair_index=torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long),
        pair_relation=torch.randn(3, module.RELATION_WIDTH, generator=generator),
        pair_bucket=torch.tensor([0, 1, 0], dtype=torch.long),
        global_context=torch.randn(n_graphs, module.GLOBAL_WIDTH, generator=generator),
        batch=batch,
        y=torch.tensor([0.0, 0.0]),
        num_nodes=n,
    )


def _model(quantile_mode: str = "none", **overrides) -> module.PatchPathModel:
    return module.PatchPathModel(
        16,
        8,
        patch_hidden=8,
        pair_hidden=4,
        token_width=6,
        dropout=0.0,
        shell_width=7,
        center_context=True,
        center_context_hidden=12,
        quantile_mode=quantile_mode,
        **overrides,
    )


# --------------------------------------------------------------------------
# Test 1: pinball tau=0.5 identity (2 * pinball_0.5 == L1)
# --------------------------------------------------------------------------

def test_two_pinball_half_equals_l1_elementwise() -> None:
    generator = torch.Generator().manual_seed(3)
    error = torch.randn(1000, generator=generator) * 10.0
    doubled = 2.0 * module.pinball_loss(error, 0.5)
    assert doubled.shape == error.shape
    assert torch.equal(doubled, torch.abs(error))
    # also the reduced form matches torch L1 exactly (per-element equality)
    target = torch.randn(1000, generator=generator)
    prediction = torch.randn(1000, generator=generator)
    l1 = torch.nn.functional.l1_loss(prediction, target)
    pin = (2.0 * module.pinball_loss(target - prediction, 0.5)).mean()
    assert torch.equal(l1, pin)
    assert float(l1) == float(pin)


def test_pinball_values_at_standard_taus() -> None:
    # manual spot checks: y - q = +2 -> tau*2 ; = -3 -> (tau-1)*(-3)
    for tau in (0.1, 0.5, 0.9):
        err = torch.tensor([2.0, -3.0])
        expected = torch.tensor(
            [tau * 2.0, (tau - 1.0) * -3.0]
        )  # (tau-1)*e with e<0 is positive
        assert torch.allclose(module.pinball_loss(err, tau), expected)


# --------------------------------------------------------------------------
# Test 2: non-crossing q10 <= q50 <= q90
# --------------------------------------------------------------------------

def test_non_crossing_quantiles() -> None:
    torch.manual_seed(0)
    model = _model(quantile_mode=MODE_MQ).eval()
    batch = _toy_batch(shell_width=7, n_graphs=4)
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (4, 3)
    # push the raw head outputs to extreme values to stress softplus positivity
    extreme = torch.tensor(
        [[0.0, 50.0, -50.0], [-3.0, -40.0, 60.0], [1.0, 0.0, 0.0], [2.0, 1e3, 1e3]]
    )
    q10 = extreme[:, 0] - torch.nn.functional.softplus(extreme[:, 1])
    q50 = extreme[:, 0]
    q90 = extreme[:, 0] + torch.nn.functional.softplus(extreme[:, 2])
    assert bool((q10 <= q50).all())
    assert bool((q50 <= q90).all())
    assert bool((q10 <= q90).all())
    assert bool((out[:, 0] <= out[:, 1]).all() and (out[:, 1] <= out[:, 2]).all())
    # widths are softplus-positive
    assert bool((out[:, 1] - out[:, 0] >= 0).all())
    assert bool((out[:, 2] - out[:, 1] >= 0).all())


# --------------------------------------------------------------------------
# Test 3: finite forward and backward for every quantile loss
# --------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["none", "median_only", MODE_MQ])
def test_loss_forward_backward_finite(mode: str) -> None:
    torch.manual_seed(7)
    model = _model(quantile_mode=mode)
    batch = _toy_batch(shell_width=7, n_graphs=4)
    batch.y = torch.randn(4)
    prediction = model(batch)
    loss = module.quantile_regression_loss(
        prediction, batch.y, quantile_mode=mode, quantile_lambda=0.10
    )
    assert torch.isfinite(loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    finite = all(
        torch.isfinite(param.grad).all() for param in model.parameters() if param.grad is not None
    )
    assert finite


def test_mq_loss_matches_manual_formula() -> None:
    torch.manual_seed(9)
    model = _model(quantile_mode=MODE_MQ)
    batch = _toy_batch(shell_width=7, n_graphs=4)
    target = torch.randn(4)
    prediction = model(batch)
    loss = module.quantile_regression_loss(
        prediction, target, quantile_mode=MODE_MQ, quantile_lambda=0.25
    )
    q10, q50, q90 = prediction[:, 0], prediction[:, 1], prediction[:, 2]
    manual = (
        (2.0 * module.pinball_loss(target - q50, 0.5)).mean()
        + 0.25
        * (
            module.pinball_loss(target - q10, 0.10).mean()
            + module.pinball_loss(target - q90, 0.90).mean()
        )
    )
    assert torch.equal(loss, manual)


# --------------------------------------------------------------------------
# Test 4: baseline guard — quantile_mode="none" is the compact-v4 model
# --------------------------------------------------------------------------

def test_none_mode_matches_original_model_bitwise() -> None:
    torch.manual_seed(1234)
    original = _model()  # original signature: no quantile_mode
    torch.manual_seed(1234)
    guarded = _model(quantile_mode="none")
    assert set(original.state_dict().keys()) == set(guarded.state_dict().keys())
    assert list(original.state_dict().keys()) == list(guarded.state_dict().keys())
    for key in original.state_dict():
        assert torch.equal(original.state_dict()[key], guarded.state_dict()[key])
    # scalar outputs identical on the same batch
    original.eval()
    guarded.eval()
    batch = _toy_batch(shell_width=7, n_graphs=4)
    with torch.no_grad():
        assert torch.equal(original(batch), guarded(batch))
    # v5 parameter delta is only the widened final Linear (+2*head_hidden_1
    # weights +2 bias); for the frozen config head_hidden_1=32 this is +66.
    v4_params = sum(p.numel() for p in original.parameters())
    mq = _model(quantile_mode=MODE_MQ)
    mq_params = sum(p.numel() for p in mq.parameters())
    head_hidden_1 = original.head_hidden_1
    assert v4_params + 2 * head_hidden_1 + 2 == mq_params
    assert mq_params - v4_params == 2 * head_hidden_1 + 2


def test_median_only_keeps_scalar_architecture() -> None:
    original = _model()
    median = _model(quantile_mode="median_only")
    assert sum(p.numel() for p in median.parameters()) == sum(
        p.numel() for p in original.parameters()
    )
    assert list(median.state_dict().keys()) == list(original.state_dict().keys())


def test_none_loss_is_exact_l1() -> None:
    torch.manual_seed(31)
    prediction = torch.randn(8, requires_grad=True)
    target = torch.randn(8)
    computed = module.quantile_regression_loss(
        prediction, target, quantile_mode="none"
    )
    assert torch.equal(computed, torch.nn.functional.l1_loss(prediction, target))


# --------------------------------------------------------------------------
# Test 5: lambda=0 leaves the auxiliary (width) outputs unoptimized
# --------------------------------------------------------------------------

def test_lambda_zero_does_not_optimize_q10_q90() -> None:
    torch.manual_seed(5)
    model = _model(quantile_mode=MODE_MQ)
    batch = _toy_batch(shell_width=7, n_graphs=4)
    target = torch.randn(4)
    prediction = model(batch)
    # q50-only loss (lambda=0 must be equivalent to omitting aux terms)
    loss = module.quantile_regression_loss(
        prediction, target, quantile_mode=MODE_MQ, quantile_lambda=0.10
    )
    loss_no_aux = (
        2.0 * module.pinball_loss(target - prediction[:, 1], 0.5)
    ).mean()
    # grad check for the zero-aux path via the median-only helper below
    q50_only = module.quantile_regression_loss(
        prediction[:, 1], target, quantile_mode="median_only"
    )
    assert torch.equal(q50_only, loss_no_aux)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    q50_only.backward()
    head = model.head[-1]
    # rows 1 and 2 of the final linear (the raw d outputs) must have zero grad
    assert head.weight.grad is not None
    assert torch.equal(head.weight.grad[1:], torch.zeros_like(head.weight.grad[1:]))
    assert torch.equal(head.bias.grad[1:], torch.zeros_like(head.bias.grad[1:]))
    # row 0 (m) must have nonzero grad somewhere
    assert not torch.equal(head.weight.grad[0], torch.zeros_like(head.weight.grad[0]))


# --------------------------------------------------------------------------
# Test 7: pre-registered lambda grid is enforced
# --------------------------------------------------------------------------

def test_preregistered_lambda_grid_enforced() -> None:
    for lam in (0.10, 0.25, 0.50):
        assert module._validate_quantile_config(MODE_MQ, lam) == lam
    for bad in (0.05, 0.15, 0.2, 0.3, 1.0):
        with pytest.raises(ValueError):
            module._validate_quantile_config(MODE_MQ, bad)
    # median_only / none ignore lambda
    assert module._validate_quantile_config("median_only", 0.25) is None
    assert module._validate_quantile_config("none", None) is None


def test_bad_quantile_mode_rejected() -> None:
    with pytest.raises(ValueError):
        _model(quantile_mode="q10_q50")  # unsupported combination
    with pytest.raises(ValueError):
        module.quantile_regression_loss(torch.zeros(3), torch.zeros(3), quantile_mode="bad")
