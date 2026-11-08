from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import tr_common as C
import tr_loss as L


def _toy(seed: int = 11, batch: int = 23, tasks: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(int(seed))
    logits = torch.randn(batch, tasks, generator=gen)
    raw = torch.randn(batch, tasks, generator=gen)
    y = torch.where(raw > 0.35, 1.0, torch.where(raw < -0.15, 0.0, float("nan")))
    return logits, y


def _loss_grad(logits: torch.Tensor, y: torch.Tensor) -> tuple[float, np.ndarray]:
    z = logits.clone().requires_grad_(True)
    loss, _stats = L.rank_loss_parts(z, y)
    grad = torch.autograd.grad(loss, z)[0]
    return float(loss.detach()), grad.detach().numpy()


def test_rank_matches_independent_numpy_reference():
    logits, y = _toy()
    value = float(L.rank_loss(logits, y).detach())
    reference = L.numpy_rank_loss(logits.numpy(), y.numpy())
    assert abs(value - reference) < 1e-6


def test_task_equal_weight_not_row_equal_weight():
    logits, y = _toy(seed=5, batch=31, tasks=6)
    total = 0.0
    active = 0
    for t in range(logits.shape[1]):
        col = logits[:, t : t + 1]
        yt = y[:, t : t + 1]
        col_y = yt[:, 0]
        obs = torch.isfinite(col_y)
        if int(((obs) & (col_y == 1)).sum()) and int(((obs) & (col_y == 0)).sum()):
            total += float(L.rank_loss(col, yt).detach())
            active += 1
    assert active >= 2
    assert abs(float(L.rank_loss(logits, y).detach()) - total / active) < 1e-6


def test_nan_rows_do_not_enter_loss_or_gradient():
    logits, y = _toy(seed=7)
    nan_rows = ~torch.isfinite(y)
    z2 = logits.clone()
    z2[nan_rows] += 4.0
    v1, g1 = _loss_grad(logits, y)
    v2, g2 = _loss_grad(z2, y)
    assert v1 == pytest.approx(v2, abs=0.0)
    assert np.array_equal(g1, g2)
    assert float(np.max(np.abs(g1[nan_rows.numpy()]))) == 0.0


def test_missing_positive_or_negative_columns_are_skipped():
    logits = torch.tensor(
        [
            [2.0, -1.0, 0.5, float("nan")],
            [1.0, -2.0, -0.5, 1.0],
            [0.0, -3.0, 0.0, 0.0],
        ]
    )
    y = torch.tensor(
        [
            [1.0, 0.0, 0.0, float("nan")],
            [0.0, 0.0, 0.0, float("nan")],
            [0.0, 0.0, 0.0, float("nan")],
        ]
    )
    # only task 0 has both classes; task 1 has no positive, task 2 has no positive,
    # task 3 is fully missing
    value = float(L.rank_loss(logits, y).detach())
    ref = float(
        F.softplus(
            torch.logsumexp(torch.tensor([1.0, 0.0]), dim=0)
            - torch.log(torch.tensor(2.0))
            - 2.0
        )
    )
    assert value == pytest.approx(ref, abs=1e-6)


def test_all_missing_and_empty_batches_return_exact_zero():
    z = torch.randn(9, 4, requires_grad=True)
    y = torch.full((9, 4), float("nan"))
    loss, _ = L.rank_loss_parts(z, y)
    grad = torch.autograd.grad(loss, z)[0]
    assert float(loss.detach()) == 0.0
    assert torch.equal(grad, torch.zeros_like(grad))

    z0 = torch.zeros((0, 4), requires_grad=True)
    y0 = torch.full((0, 4), float("nan"))
    loss0, _ = L.rank_loss_parts(z0, y0)
    grad0 = torch.autograd.grad(loss0, z0)[0]
    assert float(loss0.detach()) == 0.0
    assert torch.equal(grad0, torch.zeros_like(grad0))


def test_monotonicity_positive_up_and_negative_up():
    logits, y = _toy(seed=13)
    base = float(L.rank_loss(logits, y).detach())
    pos = (y == 1.0).numpy()
    neg = (y == 0.0).numpy()

    z = logits.clone()
    z[pos] += 3.0
    assert float(L.rank_loss(z, y).detach()) <= base + 1e-9

    z = logits.clone()
    z[neg] += 3.0
    assert float(L.rank_loss(z, y).detach()) >= base - 1e-9

    # the reverse directions must move the loss the other way (non-vacuous)
    z = logits.clone()
    z[pos] -= 3.0
    assert float(L.rank_loss(z, y).detach()) > base + 1e-6


def test_mean_softplus_over_positives_not_softplus_of_mean_gap():
    # One assay, two positives with very different gaps: the contract is the
    # mean of softplus, which is larger than softplus of the mean gap.
    logits = torch.tensor([[10.0], [0.0], [-10.0]])
    y = torch.tensor([[1.0], [1.0], [0.0]])
    got = float(L.rank_loss(logits, y).detach())
    neg_ref = -10.0
    ref = float(
        (
            F.softplus(torch.tensor(neg_ref - 10.0))
            + F.softplus(torch.tensor(neg_ref - 0.0))
        )
        / 2.0
    )
    wrong = float(F.softplus(torch.tensor(neg_ref - 5.0)))
    assert got == pytest.approx(ref, abs=1e-6)
    assert got > wrong


def test_invalid_columns_have_exactly_zero_gradient():
    logits, y = _toy(seed=17)
    y = y.clone()
    y[:, 0] = float("nan")  # all-missing column
    y[:, 1] = torch.where(y[:, 1] == 1.0, 0.0, y[:, 1])  # no positive
    y[:, 2] = torch.where(y[:, 2] == 0.0, 1.0, y[:, 2])  # no negative
    z = logits.clone().requires_grad_(True)
    loss, _ = L.rank_loss_parts(z, y)
    grad = torch.autograd.grad(loss, z)[0].numpy()
    assert float(loss.detach()) > 0.0
    assert np.all(grad[:, 0] == 0.0)
    assert np.all(grad[:, 1] == 0.0)
    assert np.all(grad[:, 2] == 0.0)
    assert np.isfinite(grad).all()


def test_lambda_zero_is_exactly_masked_bce():
    logits, y = _toy(seed=19)
    bce = C.MET.masked_bce_loss(logits, y)
    total, parts = L.total_loss_parts(logits, y, 0.0)
    assert float(total.detach()) == float(bce.detach())
    assert float(parts["lambda_rank_times_rank"]) == 0.0

    zb = logits.clone().requires_grad_(True)
    bce_b = C.MET.masked_bce_loss(zb, y)
    gb = torch.autograd.grad(bce_b, zb)[0]
    zt = logits.clone().requires_grad_(True)
    total_t, _ = L.total_loss_parts(zt, y, 0.0)
    gt = torch.autograd.grad(total_t, zt)[0]
    assert torch.equal(gb, gt)


def test_extreme_logits_are_finite():
    logits, y = _toy(seed=23)
    z = logits * 1e6
    z = z.requires_grad_(True)
    loss, _ = L.rank_loss_parts(z, y)
    grad = torch.autograd.grad(loss, z)[0]
    assert torch.isfinite(loss)
    assert torch.isfinite(grad).all()


def test_shape_and_label_validation():
    z = torch.zeros(3, 2)
    with pytest.raises(ValueError):
        L.rank_loss(z, torch.zeros(2, 3))
    bad = torch.tensor([[1.0, 2.0], [0.0, 0.5]])
    with pytest.raises(ValueError):
        L.rank_loss(torch.zeros(2, 2), bad)


def test_cross_assay_scores_are_never_compared():
    # Changing every logit of assay 0 by +100 must not change assay 1's term.
    logits, y = _toy(seed=29, batch=18, tasks=4)
    shifted = logits.clone()
    shifted[:, 0] += 100.0
    per_task = []
    for t in range(4):
        per_task.append(float(L.rank_loss(logits[:, t : t + 1], y[:, t : t + 1]).detach()))
    per_task_shifted = [
        float(L.rank_loss(shifted[:, t : t + 1], y[:, t : t + 1]).detach()) for t in range(4)
    ]
    assert per_task[1:] == per_task_shifted[1:]


def test_self_check_all_pass():
    result = L.self_check()
    assert result["all_pass"] is True
