"""The four frozen losses and the R/W algebraic identity.

Standardised units throughout: ``z = (y - mu) / sigma``, ``p = f_theta(x)``,
``e = p - z``.  The ordinary molecule term is always ``mean(e_i^2)``.

R: ``mean[(e_b - e_a)^2 / 2]``       real-edit residual coupling
W: ``mean[(e_a^2 + e_b^2) / 2]``     endpoint reweighting (same endpoint weight)
N: ``mean[(e_v - e_u)^2 / 2]``       random token-pair residual coupling

Identity: ``R = W - mean(e_a * e_b)``.
"""

from __future__ import annotations

from typing import Any

import torch


def residual(pred: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return pred.reshape(-1) - z.reshape(-1)


def mol_loss(pred: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    e = residual(pred, z)
    return (e * e).mean()


def aux_loss(arm: str, e_a: torch.Tensor, e_b: torch.Tensor) -> torch.Tensor:
    if arm == "R":
        return ((e_b - e_a) ** 2 / 2.0).mean()
    if arm == "W":
        return ((e_a ** 2 + e_b ** 2) / 2.0).mean()
    if arm == "N":
        return ((e_b - e_a) ** 2 / 2.0).mean()
    if arm == "A":
        return (e_a * 0.0).sum()
    raise KeyError(arm)


def total_loss(loss_mol: torch.Tensor, loss_aux: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    return loss_mol + float(lam) * loss_aux


def identity_gap(e_a: torch.Tensor, e_b: torch.Tensor) -> dict[str, Any]:
    r = ((e_b - e_a) ** 2 / 2.0).mean()
    w = ((e_a ** 2 + e_b ** 2) / 2.0).mean()
    cross = (e_a * e_b).mean()
    return {
        "R": float(r.detach()),
        "W": float(w.detach()),
        "mean_ea_eb": float(cross.detach()),
        "W_minus_cross": float((w - cross).detach()),
        "R_minus_W_plus_cross": float((r - w + cross).detach()),
    }


__all__ = ["residual", "mol_loss", "aux_loss", "total_loss", "identity_gap"]
