"""Pure-torch ANOVA machinery for HIS-v0.

Object
------
``LocalContent.path_emb`` is an nn.Embedding of shape ``[189, 32]`` indexed by

    path_cat = type(middle_atom) * 9 + (bond1 - 1) * 3 + (bond2 - 1)

(see ``lc_data.build_from_arrays``), i.e. a ``(21, 3, 3)`` categorical table of
32-d embedding vectors.  Its *effective linear response* on the first
LocalContent MLP layer is

    H[a, b, c, :] = path_emb.weight[a, b, c, :] @ W_path.T          (192-d)

where ``W_path = local.mlp.lins[0].weight[:, s : s + 32]`` is the block of the
first-layer weight that reads the ``path_emb`` slice of the MLP input vector.
The slice offset ``s = 3 * atom_out + stat_rank`` follows from the frozen input
layout in ``lc_models.LocalContent.forward``:

    [a_v (96) | sum_e b_e (96) | mean_e b_e (96) | nbr (32) |
     path_emb (32) | end_emb (32) | path_joint (32) | closure (8) |
     content_agg (96) | scalars (9)]                      -> 529 input columns

so ``s = 320`` and the path block is ``[320, 352)``.  ``path_slice`` derives it
from module attributes and ``his_cpu_checks`` verifies it *empirically* against
the real forward pass (a random perturbation of ``path_emb`` must only change
input columns 320..351), never by trusting the hard-coded constant.

ANOVA
-----
``H`` is a ``(A=21, B=3, C=3, F)`` tensor.  Write ``Hbar_a`` for the mean over
axes ``(b, c)`` with the ``a`` axis kept, ``Hbar_ab`` for the mean over ``c``,
and ``mu`` for the grand mean.  The best (least-squares) approximation that
contains no pure three-way interaction is the standard balanced ANOVA fit

    H_le2 = Hbar_ab + Hbar_ac + Hbar_bc - (Hbar_a + Hbar_b + Hbar_c) + mu
    H_3   = H - H_le2
          = H - Hbar_a - Hbar_b - Hbar_c + Hbar_ab + Hbar_ac + Hbar_bc - mu

(the double counting of the one-way terms inside the two-way marginals is why
the pairwise marginals appear with ``+`` and the main effects with ``-``).

NOTE ON THE TASK TEXT: the formula written in the HIS-v0 task brief,
``H - mean_a - mean_b - mean_c + mean_ab + mean_ac + mean_bc - grand``, is the
*negated* main-effect part; applied literally it fails every property the same
brief demands (it does not annihilate a pure 1st/2nd-order tensor, and its
marginals are not zero).  See ``tests/test_his_v0.py::test_literal_brief_formula_is_not_the_residual``.
The four required properties (exact reconstruction, zero margins on all three
axes, ``H_3 = 0`` for any constant/1st/2nd-order tensor, ``H_3 != 0`` for a pure
third-order tensor) hold exactly for ``H_3`` as implemented here.

Penalty
-------
    loss_HIS = mean(H_3 ** 2)        (mean over all 189 * 192 elements)

This is invariant under the re-parameterisation ``E -> cE, W_path -> W_path/c``
for ``c > 0`` (the product ``H = E @ W_path.T`` is what is penalised, not the
bare embedding), so it cannot be dodged by rescaling the embedding table.
"""

from __future__ import annotations

from typing import Any

import torch

#: Frozen categorical axes of the S2 typed-path code.
PATH_SHAPE: tuple[int, int, int] = (21, 3, 3)


# ---------------------------------------------------------------------------
# input layout
# ---------------------------------------------------------------------------

def path_slice_from_layout(atom_out: int, stat_rank: int) -> slice:
    """Column block of the LocalContent MLP input occupied by ``path_emb``."""
    start = 3 * int(atom_out) + int(stat_rank)
    return slice(start, start + int(stat_rank))


def local_out_dim(atom_out: int, stat_rank: int, n_scalars: int) -> int:
    """Total LocalContent MLP input width implied by the frozen layout."""
    return (3 * int(atom_out)          # a_v, sum b, mean b
            + int(stat_rank)           # S1 joint neighbour
            + 3 * int(stat_rank) + 8   # path_emb, end_emb, path_joint, closure
            + int(atom_out)            # gated endpoint content
            + int(n_scalars))


def path_slice(model: torch.nn.Module) -> slice:
    """Derive and structurally validate the ``path_emb`` column block."""
    local = model.local
    sl = path_slice_from_layout(local.atom_out, local.rank)
    expected = local_out_dim(local.atom_out, local.rank, 9)
    got = int(local.mlp.lins[0].in_features)
    if got != expected:
        raise RuntimeError(
            f"LocalContent layout changed: first-layer in_features={got}, "
            f"frozen layout implies {expected}")
    if int(local.mlp.lins[0].out_features) != int(local.mlp.lins[1].in_features):
        raise RuntimeError("unexpected LocalContent MLP shape")
    return sl


def effective_response(model: torch.nn.Module) -> torch.Tensor:
    """``H[a, b, c, :] = E[a, b, c, :] @ W_path.T``  (differentiable)."""
    local = model.local
    rank = int(local.rank)
    sl = path_slice(model)
    # ``reshape`` (not ``view``) so this also works if the table is ever
    # non-contiguous; the (21, 3, 3) interpretation is the path_cat encoding.
    emb = local.path_emb.weight.reshape(*PATH_SHAPE, rank)
    w_path = local.mlp.lins[0].weight[:, sl]           # [hidden, rank]
    return torch.einsum("abcd,ed->abce", emb, w_path)  # [A, B, C, hidden]


# ---------------------------------------------------------------------------
# ANOVA on the three categorical axes (axes 0, 1, 2 of ``H``)
# ---------------------------------------------------------------------------

def _marginals(h: torch.Tensor) -> dict[str, torch.Tensor]:
    m = h.mean(dim=(0, 1, 2), keepdim=True)
    return {
        "a": h.mean(dim=(1, 2), keepdim=True),
        "b": h.mean(dim=(0, 2), keepdim=True),
        "c": h.mean(dim=(0, 1), keepdim=True),
        "ab": h.mean(dim=2, keepdim=True),
        "ac": h.mean(dim=1, keepdim=True),
        "bc": h.mean(dim=0, keepdim=True),
        "mu": m,
    }


def anova_le2(h: torch.Tensor) -> torch.Tensor:
    """Best constant + one-way + two-way fit of ``h`` (no three-way term)."""
    g = _marginals(h)
    return g["ab"] + g["ac"] + g["bc"] - (g["a"] + g["b"] + g["c"]) + g["mu"]


def third_order(h: torch.Tensor) -> torch.Tensor:
    """Pure three-way interaction residual ``H_3 = H - H_le2``."""
    return h - anova_le2(h)


def margin_norms(h: torch.Tensor) -> dict[str, float]:
    """Frobenius norm of the marginals of ``h`` (zero for a pure 3-way tensor)."""
    g = _marginals(h)
    return {k: float(v.reshape(-1).pow(2).sum().sqrt()) for k, v in g.items()}


def fro(t: torch.Tensor) -> float:
    return float(t.reshape(-1).pow(2).sum().sqrt())


def project_embedding(emb: torch.Tensor, order: int = 2) -> torch.Tensor:
    """ANOVA projection of the ``(21, 3, 3, rank)`` embedding table.

    ``order=2`` returns ``E - E_3``.  Because the ANOVA projector acts only on
    the three categorical axes and ``H = E @ W_path.T`` right-multiplies each
    cell by the same matrix, ``project(E) @ W_path.T == project(E @ W_path.T)``
    exactly (checked in the CPU gates).
    """
    if order != 2:
        raise ValueError("only the <=2-order projection is defined here")
    e = emb.reshape(*PATH_SHAPE, emb.shape[-1])
    return e - third_order(e)


# ---------------------------------------------------------------------------
# penalty
# ---------------------------------------------------------------------------

def his_penalty(model: torch.nn.Module) -> tuple[torch.Tensor, dict[str, Any]]:
    """``mean(H_3 ** 2)`` plus diagnostics.  Gradients reach only E and W_path."""
    h = effective_response(model)
    h3 = third_order(h)
    pen = h3.pow(2).mean()
    stats = {
        "h_fro": fro(h),
        "h3_fro": fro(h3),
        "h_rms": float(h.pow(2).mean().sqrt()),
        "h3_rms": float(h3.pow(2).mean().sqrt()),
        "h3_ratio": fro(h3) / max(fro(h), 1e-30),
        "h3_ratio_centered": fro(h3) / max(fro(h - h.mean(dim=(0, 1, 2), keepdim=True)), 1e-30),
        "h_le2_fro": fro(anova_le2(h)),
        "penalty": float(pen),
    }
    return pen, stats


def affected_parameters(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """The parameter tensors the HIS penalty can touch (used by calibration)."""
    local = model.local
    sl = path_slice(model)
    return {
        "path_emb.weight": local.path_emb.weight,
        # NOTE: a *view*; for gradients read ``mlp.lins.0.weight.grad[:, sl]``.
        "W_path": local.mlp.lins[0].weight[:, sl],
    }


def affected_grad_vector(model: torch.nn.Module) -> torch.Tensor | None:
    """Concatenated gradient of the affected parameters (``None`` if absent)."""
    local = model.local
    sl = path_slice(model)
    g_emb = local.path_emb.weight.grad
    w = local.mlp.lins[0].weight
    g_w = w.grad
    if g_emb is None and g_w is None:
        return None
    parts = []
    if g_emb is not None:
        parts.append(g_emb.reshape(-1))
    if g_w is not None:
        parts.append(g_w[:, sl].reshape(-1))
    return torch.cat(parts)


def outside_slice_grad_max(model: torch.nn.Module) -> float:
    """Max |grad| on first-layer weight columns *outside* the path block."""
    local = model.local
    sl = path_slice(model)
    w = local.mlp.lins[0].weight
    if w.grad is None:
        return 0.0
    mask = torch.ones_like(w.grad, dtype=torch.bool)
    mask[:, sl] = False
    if not bool(mask.any()):
        return 0.0
    return float(w.grad[mask].abs().max())


__all__ = [
    "PATH_SHAPE", "path_slice_from_layout", "local_out_dim", "path_slice",
    "effective_response", "anova_le2", "third_order", "margin_norms", "fro",
    "project_embedding", "his_penalty", "affected_parameters",
    "affected_grad_vector", "outside_slice_grad_max",
]
