"""Masked BCE + per-assay hard-negative ranking loss.

The model emits ``logits`` with shape ``[B, 128]``; every assay is a separate
binary task.  For assay ``t`` let ``P_t`` be the rows of this batch whose label
is observed **and** positive, ``N_t`` the rows whose label is observed **and**
negative (NaN rows belong to neither set).  When both sets are non-empty:

    negative_reference_t = logsumexp_t(logits[N_t]) - log(|N_t|)

    rank_loss_t = mean_{i in P_t} softplus(negative_reference_t - logits[i, t])

    loss_rank = mean_{t : |P_t| > 0 and |N_t| > 0} rank_loss_t

This is a *hard-negative ranking surrogate*: the log-sum-exp makes every
negative compete through a soft maximum, and softplus penalises positives that
do not clear that soft maximum.  It is **not** an exact Average-Precision loss:
AP depends on the full ranking of all negatives, whereas this term is a smooth
one-sided surrogate that only compares each assay's positives with the soft
maximum of that assay's observed negatives.  Scores of different assays are
never compared.

Implementation is vectorised over the 128 assay columns: one masked
``logsumexp`` and one masked mean per batch, ``O(B x 128)`` work, no dense
positive x negative pair tensor.  Assays with no positive or no negative in the
batch contribute exactly zero (value and gradient); an all-missing batch returns
``logits.sum() * 0``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

import tr_common as C
from tr_common import MET


def _validate(logits: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits.shape != y.shape:
        raise ValueError(f"rank logits {tuple(logits.shape)} != target {tuple(y.shape)}")
    if logits.ndim != 2:
        raise ValueError(f"rank loss expects [B, T] logits, got {tuple(logits.shape)}")
    if not logits.dtype.is_floating_point:
        raise TypeError(f"logits must be floating point, got {logits.dtype}")
    observed = torch.isfinite(y)
    positive = observed & (y == 1)
    negative = observed & (y == 0)
    if bool((observed & ~(positive | negative)).any()):
        raise ValueError("observed labels must be exactly 0 or 1")
    return positive, negative, observed


def rank_loss_parts(
    logits: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return ``(loss_rank, stats)``; stats are scalar tensors (no .item()).

    The returned stats are detached integer/float scalars, safe to accumulate
    on the device and convert once per epoch.
    """
    positive, negative, _observed = _validate(logits, y)
    batch, tasks = logits.shape

    n_pos = positive.sum(dim=0)  # [T] int64
    n_neg = negative.sum(dim=0)  # [T] int64
    valid = (n_pos > 0) & (n_neg > 0)

    dtype = logits.dtype
    pos_f = positive.to(dtype)
    pos_count = n_pos.clamp(min=1).to(dtype)
    neg_count = n_neg.clamp(min=1).to(dtype)

    # Zero the whole column for assays that are not rank-active.  Their term is
    # multiplied out below, but this also makes a fully-missing column safe even
    # if its logits happen to be non-finite (which never happens in training;
    # the trainer rejects non-finite logits).
    safe_logits = torch.where(valid.unsqueeze(0), logits, torch.zeros_like(logits))

    # A masked logsumexp over negatives.  For an assay with no negative the mask
    # is filled with every row instead of being left all -inf, which would make
    # logsumexp NaN.  Its term is multiplied by zero and never enters the mean.
    safe_negative = negative | (~negative.any(dim=0)).unsqueeze(0)
    neg_inf = torch.finfo(dtype).min
    neg_reference = torch.logsumexp(safe_logits.masked_fill(~safe_negative, neg_inf), dim=0)
    neg_reference = neg_reference - torch.log(neg_count)

    # The requirement is a mean of softplus terms over the assay's positives:
    # mean_i softplus(neg_reference_t - logits[i, t]).  Softplus is convex, so
    # this is *not* softplus(neg_reference_t - mean_i logits[i, t]); the two
    # differ whenever an assay has more than one positive in the batch.
    task_loss = (F.softplus(neg_reference.unsqueeze(0) - safe_logits) * pos_f).sum(dim=0) / pos_count

    valid_f = valid.to(dtype)
    n_valid = valid_f.sum()
    mean_valid = (task_loss * valid_f).sum() / n_valid.clamp(min=1.0)
    zero = logits.sum() * 0.0
    loss = torch.where(valid.any(), mean_valid, zero)

    stats = {
        "n_active_tasks": n_valid.detach().to(torch.float32),
        "n_tasks": torch.as_tensor(tasks, device=logits.device, dtype=torch.float32),
        "n_no_positive_tasks": (n_pos == 0).sum().detach().to(torch.float32),
        "n_no_negative_tasks": (n_neg == 0).sum().detach().to(torch.float32),
        "n_positive_entries": n_pos.sum().detach().to(torch.float32),
        "n_negative_entries": n_neg.sum().detach().to(torch.float32),
        "n_positive_entries_active": n_pos[valid].sum().detach().to(torch.float32)
        if bool(valid.any())
        else torch.zeros((), device=logits.device),
        "n_negative_entries_active": n_neg[valid].sum().detach().to(torch.float32)
        if bool(valid.any())
        else torch.zeros((), device=logits.device),
        "implicit_pairs": (n_pos * n_neg).sum().detach().to(torch.float32),
        "batch_rows": torch.as_tensor(batch, device=logits.device, dtype=torch.float32),
        "active_mask": valid.detach(),
    }
    return loss, stats


def rank_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The per-assay hard-negative ranking term alone."""
    loss, _stats = rank_loss_parts(logits, y)
    return loss


def total_loss_parts(
    logits: torch.Tensor,
    y: torch.Tensor,
    lambda_rank: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """``loss_bce + lambda_rank * loss_rank`` with finite scalar parts."""
    if not (float(lambda_rank) >= 0.0):
        raise ValueError(f"lambda_rank must be non-negative, got {lambda_rank}")
    loss_bce = MET.masked_bce_loss(logits, y)
    loss_rank, stats = rank_loss_parts(logits, y)
    loss_total = loss_bce + float(lambda_rank) * loss_rank
    parts = {
        "loss_bce": loss_bce.detach(),
        "loss_rank": loss_rank.detach(),
        "loss_total": loss_total.detach(),
        "lambda_rank_times_rank": (float(lambda_rank) * loss_rank).detach(),
        **stats,
    }
    return loss_total, parts


def total_loss(logits: torch.Tensor, y: torch.Tensor, lambda_rank: float) -> torch.Tensor:
    """Training objective.  At ``lambda_rank == 0`` this is exactly masked BCE."""
    loss, _parts = total_loss_parts(logits, y, float(lambda_rank))
    return loss


# ---------------------------------------------------------------------------
# numpy reference used by the audit / tests
# ---------------------------------------------------------------------------


def numpy_rank_loss(logits: Any, y: Any) -> float:
    """Loop reference: per-assay logsumexp + softplus, then equal-weight mean."""
    import numpy as np

    z = np.asarray(logits, dtype=np.float64)
    t = np.asarray(y, dtype=np.float64)
    if z.shape != t.shape or z.ndim != 2:
        raise ValueError(f"reference arrays {z.shape} / {t.shape}")
    terms: list[float] = []
    for task in range(z.shape[1]):
        col_y = t[:, task]
        obs = np.isfinite(col_y)
        pos = obs & (col_y == 1.0)
        neg = obs & (col_y == 0.0)
        n_pos = int(pos.sum())
        n_neg = int(neg.sum())
        if n_pos == 0 or n_neg == 0:
            continue
        z_neg = z[neg, task]
        m = float(np.max(z_neg))
        neg_ref = float(m + np.log(np.exp(z_neg - m).sum()) - np.log(n_neg))
        diff = neg_ref - z[pos, task]
        # softplus(x) = log1p(exp(x)), evaluated stably
        sp = np.where(diff > 0, diff + np.log1p(np.exp(-np.abs(diff))), np.log1p(np.exp(-np.abs(diff))))
        terms.append(float(np.mean(sp)))
    if not terms:
        return 0.0
    return float(np.mean(np.asarray(terms, dtype=np.float64)))


def self_check(*, seed: int = 20261108) -> dict[str, Any]:
    """Deterministic CPU checks of the edge cases and the monotonicity signs."""
    import numpy as np

    gen = torch.Generator().manual_seed(int(seed))
    checks: dict[str, Any] = {}

    def _run(z: torch.Tensor, t: torch.Tensor) -> tuple[float, np.ndarray]:
        zz = z.clone().requires_grad_(True)
        loss, stats = rank_loss_parts(zz, t)
        grad = torch.autograd.grad(loss, zz)[0].detach().numpy()
        return float(loss.detach()), grad

    z = torch.randn(17, 5, generator=gen)
    t = torch.randn(17, 5, generator=gen)
    t = torch.where(t > 0.1, 1.0, torch.where(t < -0.1, 0.0, float("nan")))
    value, grad = _run(z, t)
    checks["matches_numpy_reference"] = {
        "torch": value,
        "numpy": numpy_rank_loss(z.numpy(), t.numpy()),
        "abs_delta": abs(value - numpy_rank_loss(z.numpy(), t.numpy())),
    }
    checks["matches_numpy_reference"]["pass"] = (
        checks["matches_numpy_reference"]["abs_delta"] < 1e-6
    )

    # NaN entries are invisible: perturbing only the logits of NaN rows cannot
    # change the loss or any gradient (those rows are in neither P_t nor N_t).
    nan_mask = ~np.isfinite(t.numpy())
    z2 = z.clone()
    z2[nan_mask] += 5.0
    value2, grad2 = _run(z2, t)
    checks["nan_rows_invisible"] = {
        "perturbed_entries": int(nan_mask.sum()),
        "loss_delta": abs(value - value2),
        "grad_max_abs_delta": float(np.max(np.abs(grad - grad2))),
        "grad_at_nan_rows_max": float(np.max(np.abs(grad[nan_mask]))) if nan_mask.any() else 0.0,
    }
    checks["nan_rows_invisible"]["pass"] = (
        checks["nan_rows_invisible"]["loss_delta"] < 1e-7
        and checks["nan_rows_invisible"]["grad_max_abs_delta"] < 1e-7
        and checks["nan_rows_invisible"]["grad_at_nan_rows_max"] < 1e-7
    )

    # all-NaN column, no-positive column, no-negative column
    t3 = t.clone()
    t3[:, 0] = float("nan")
    t3[:, 1] = torch.where(t3[:, 1] == 1.0, 0.0, t3[:, 1])
    t3[:, 2] = torch.where(t3[:, 2] == 0.0, 1.0, t3[:, 2])
    v3, g3 = _run(z, t3)
    checks["edge_columns_finite"] = {
        "loss": v3,
        "grad_finite": bool(np.isfinite(g3).all()),
    }
    checks["edge_columns_finite"]["pass"] = bool(
        np.isfinite(v3) and np.isfinite(g3).all()
    )
    # an all-invalid column must have exactly zero gradient
    checks["invalid_column_zero_grad"] = bool(np.all(g3[:, 0] == 0.0))

    # positive shift cannot increase the loss
    pos_mask = (t == 1.0).numpy()
    neg_mask = (t == 0.0).numpy()
    zp = z.clone()
    zp[pos_mask] += 2.0
    vp, _ = _run(zp, t)
    checks["positive_shift_non_increasing"] = {
        "base": value,
        "shifted": vp,
        "delta": vp - value,
        "pass": bool(vp <= value + 1e-9),
    }
    zn = z.clone()
    zn[neg_mask] += 2.0
    vn, _ = _run(zn, t)
    checks["negative_up_shift_non_decreasing"] = {
        "base": value,
        "shifted": vn,
        "delta": vn - value,
        "pass": bool(vn >= value - 1e-9),
    }
    zdown = z.clone()
    zdown[neg_mask] -= 2.0
    vdown, _ = _run(zdown, t)
    checks["negative_down_shift_decreases"] = {
        "base": value,
        "shifted": vdown,
        "delta": vdown - value,
        "pass": bool(vdown <= value + 1e-9),
    }
    # the reverse signs must violate monotonicity (sanity that the test is not vacuous)
    zbad = z.clone()
    zbad[pos_mask] -= 2.0
    vbad, _ = _run(zbad, t)
    checks["positive_down_shift_increases"] = {
        "base": value,
        "shifted": vbad,
        "delta": vbad - value,
        "pass": bool(vbad > value + 1e-6),
    }

    # huge finite logits stay finite
    zbig = z * 1e4
    vbig, gbig = _run(zbig, t)
    checks["extreme_logits_finite"] = {
        "loss": vbig,
        "grad_finite": bool(np.isfinite(gbig).all()),
        "pass": bool(np.isfinite(vbig) and np.isfinite(gbig).all()),
    }

    # empty batch and all-missing batch return exactly zero
    z0 = torch.zeros((0, 5), requires_grad=True)
    t0 = torch.full((0, 5), float("nan"))
    l0, _ = rank_loss_parts(z0, t0)
    g0 = torch.autograd.grad(l0, z0)[0]
    checks["empty_batch_zero"] = {
        "loss": float(l0.detach()),
        "grad_all_zero": bool(torch.equal(g0, torch.zeros_like(g0))),
        "pass": float(l0.detach()) == 0.0 and bool(torch.equal(g0, torch.zeros_like(g0))),
    }
    zall = torch.randn(9, 5, generator=gen, requires_grad=True)
    tall = torch.full((9, 5), float("nan"))
    lall, _ = rank_loss_parts(zall, tall)
    gall = torch.autograd.grad(lall, zall)[0]
    checks["all_missing_zero"] = {
        "loss": float(lall.detach()),
        "grad_all_zero": bool(torch.equal(gall, torch.zeros_like(gall))),
        "pass": float(lall.detach()) == 0.0
        and bool(torch.equal(gall, torch.zeros_like(gall))),
    }

    # equal-weight mean over active tasks, not over rows or pairs
    zt = torch.tensor([[2.0, -2.0], [1.0, -1.0], [0.0, 5.0]])
    yt = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, float("nan")]])
    term_a = torch.nn.functional.softplus(
        torch.logsumexp(torch.tensor([1.0, 0.0]), dim=0)
        - torch.log(torch.tensor(2.0))
        - 2.0
    )
    term_b = torch.nn.functional.softplus(torch.tensor(-2.0) - torch.tensor(-1.0))
    ref = float((term_a + term_b) / 2.0)
    got = float(rank_loss(zt, yt).detach())
    checks["task_equal_weight"] = {
        "torch": got,
        "reference": ref,
        "abs_delta": abs(got - ref),
        "pass": abs(got - ref) < 1e-6,
    }

    checks["all_pass"] = bool(
        all(
            (v.get("pass") if isinstance(v, dict) else True)
            for v in checks.values()
            if isinstance(v, dict)
        )
        and checks["invalid_column_zero_grad"]
    )
    return checks


def masked_bce_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The historical BCE, re-exported (never re-implemented here)."""
    return MET.masked_bce_loss(logits, y)


__all__ = [
    "C",
    "masked_bce_loss",
    "numpy_rank_loss",
    "rank_loss",
    "rank_loss_parts",
    "self_check",
    "total_loss",
    "total_loss_parts",
]
