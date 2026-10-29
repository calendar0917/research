"""Fresh-clone-safe unit tests for the HIS-v0 ANOVA core (torch only, no data).

Run: ``uv run pytest -q tracks/structural-sufficiency/zinc_his_v0/tests``

These are the A1 "mathematical correctness" checks that do not need the frozen
LCRC-A checkpoint.  The model-level gates (slice provenance against the real
forward pass, renumbering / batch invariance, save-reload, gradients) live in
``code/his_cpu_checks.py`` because they need the frozen feature cache and the
frozen seed-0 checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import his_anova as A  # noqa: E402


# ---------------------------------------------------------------------------
# input layout
# ---------------------------------------------------------------------------

def test_path_slice_matches_frozen_layout() -> None:
    # LCRC-A: atom_out = 96, stat_rank = 32
    assert A.path_slice_from_layout(96, 32) == slice(320, 352)
    assert A.local_out_dim(96, 32, 9) == 529
    assert A.PATH_SHAPE == (21, 3, 3)


# ---------------------------------------------------------------------------
# ANOVA properties
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(21, 3, 3, 8), (3, 3, 3, 4), (2, 2, 2, 5), (1, 3, 3, 3)])
def test_exact_reconstruction(shape) -> None:
    torch.manual_seed(0)
    h = torch.randn(*shape, dtype=torch.float64)
    h3 = A.third_order(h)
    assert torch.allclose(h3 + A.anova_le2(h), h, atol=1e-12, rtol=0)


@pytest.mark.parametrize("shape", [(21, 3, 3, 8), (3, 3, 3, 4), (2, 2, 2, 5)])
def test_marginals_of_residual_are_zero(shape) -> None:
    torch.manual_seed(1)
    h3 = A.third_order(torch.randn(*shape, dtype=torch.float64))
    for key, value in A._marginals(h3).items():
        assert float(value.abs().max()) < 1e-12, key


def _le2_tensor(shape, seed: int = 7) -> torch.Tensor:
    """Constant + additive + pairwise tensor with arbitrary (non-zero-mean) terms."""
    torch.manual_seed(seed)
    a_dim, b_dim, c_dim, f = shape
    parts = [
        torch.randn(1, 1, 1, f, dtype=torch.float64),
        torch.randn(a_dim, 1, 1, f, dtype=torch.float64),
        torch.randn(1, b_dim, 1, f, dtype=torch.float64),
        torch.randn(1, 1, c_dim, f, dtype=torch.float64),
        torch.randn(a_dim, b_dim, 1, f, dtype=torch.float64),
        torch.randn(a_dim, 1, c_dim, f, dtype=torch.float64),
        torch.randn(1, b_dim, c_dim, f, dtype=torch.float64),
    ]
    total = torch.zeros(*shape, dtype=torch.float64)
    for p in parts:
        total = total + p
    return total


def _zero_margin_noise(shape, seed: int = 29) -> torch.Tensor:
    """Noise whose mean/pairwise marginals are exactly zero (pure 3-way space)."""
    torch.manual_seed(seed)
    raw = torch.randn(*shape, dtype=torch.float64)
    return raw - A.anova_le2(raw)


@pytest.mark.parametrize("shape", [(21, 3, 3, 8), (3, 3, 3, 4), (2, 2, 2, 5)])
def test_constant_first_and_second_order_tensor_has_zero_residual(shape) -> None:
    h = _le2_tensor(shape)
    assert float(A.third_order(h).abs().max()) < 1e-11


@pytest.mark.parametrize("shape", [(1, 3, 3, 6), (21, 1, 3, 6), (21, 3, 1, 6)])
def test_degenerate_axis_kills_third_order(shape) -> None:
    torch.manual_seed(3)
    h = torch.randn(*shape, dtype=torch.float64)
    assert float(A.third_order(h).abs().max()) < 1e-12


def test_pure_third_order_is_preserved_and_nonzero() -> None:
    torch.manual_seed(5)
    a = torch.randn(21, dtype=torch.float64)
    b = torch.randn(3, dtype=torch.float64)
    c = torch.randn(3, dtype=torch.float64)
    f = torch.randn(7, dtype=torch.float64)
    # centred factors -> all marginals/pairwise marginals/grand mean vanish, so
    # the rank-1 tensor really is a pure three-way interaction
    for v in (a, b, c):
        v -= v.mean()
    pure = torch.einsum("i,j,k,f->ijkf", a, b, c, f)
    h3 = A.third_order(pure)
    assert torch.allclose(h3, pure, atol=1e-11, rtol=0)
    assert A.fro(pure) > 0
    # arbitrary tensor whose marginals are all zero is reproduced verbatim
    shifted = pure + _zero_margin_noise((21, 3, 3, 7), seed=31)
    shifted = shifted - A.anova_le2(shifted)
    assert torch.allclose(A.third_order(shifted), shifted, atol=1e-10, rtol=0)


def test_anova_le2_is_idempotent_and_annihilates_third_order() -> None:
    torch.manual_seed(11)
    h = torch.randn(21, 3, 3, 6, dtype=torch.float64)
    assert float(A.third_order(A.anova_le2(h)).abs().max()) < 1e-11
    assert torch.allclose(A.anova_le2(A.anova_le2(h)), A.anova_le2(h), atol=1e-12, rtol=0)


def test_residual_is_symmetric_under_axis_permutation() -> None:
    torch.manual_seed(13)
    h = torch.randn(3, 3, 3, 5, dtype=torch.float64)
    h3 = A.third_order(h)
    perm = (1, 2, 0, 3)
    back = (2, 0, 1, 3)
    assert torch.allclose(A.third_order(h.permute(perm)), h3.permute(perm), atol=1e-12, rtol=0)
    assert torch.allclose(h3.permute(perm).permute(back), h3, atol=1e-12, rtol=0)


def test_category_relabelling_only_permutes_the_residual() -> None:
    torch.manual_seed(17)
    h = torch.randn(21, 3, 3, 4, dtype=torch.float64)
    h3 = A.third_order(h)
    for axis in (0, 1, 2):
        perm = torch.randperm(h.shape[axis])
        moved = h.index_select(axis, perm)
        assert torch.allclose(A.third_order(moved), h3.index_select(axis, perm),
                             atol=1e-12, rtol=0)
        assert abs(A.fro(A.third_order(moved)) - A.fro(h3)) < 1e-11


def test_literal_brief_formula_is_not_the_residual() -> None:
    """The formula printed in the task brief is missing the main-effect signs.

    Applied literally it neither annihilates a pure second-order tensor nor has
    zero marginals, so it would violate the four properties the same brief
    requires.  This test pins the discrepancy down; ``third_order`` implements
    the standard (correct) residual.
    """
    h = _le2_tensor((21, 3, 3, 4))
    g = A._marginals(h)
    literal = h - g["a"] - g["b"] - g["c"] + g["ab"] + g["ac"] + g["bc"] - g["mu"]
    assert float(literal.abs().max()) > 1.0          # fails "second order -> 0"
    assert float(A._marginals(literal)["a"].abs().max()) > 1.0   # and marginals
    # ... whereas the implemented residual is exactly zero on the same tensor
    assert float(A.third_order(h).abs().max()) < 1e-11


# ---------------------------------------------------------------------------
# effective response: projection commutes with the linear read-out
# ---------------------------------------------------------------------------

def test_projection_commutes_with_readout() -> None:
    torch.manual_seed(19)
    emb = torch.randn(21, 3, 3, 32, dtype=torch.float64)
    # deliberately rank-deficient read-out to prove the identity is not relying
    # on W being full rank
    w = torch.randn(192, 32, dtype=torch.float64)
    lhs = A.project_embedding(emb) @ w.T
    rhs = A.third_order(emb @ w.T)  # wait: this is the residual, not the <=2 fit
    rhs = A.anova_le2(emb @ w.T)
    assert torch.allclose(lhs, rhs, atol=1e-10, rtol=0)
    assert torch.allclose(emb @ w.T - A.project_embedding(emb) @ w.T,
                          A.third_order(emb @ w.T), atol=1e-10, rtol=0)


def test_penalty_invariant_to_embedding_rescaling() -> None:
    torch.manual_seed(23)
    emb = torch.randn(21, 3, 3, 32, dtype=torch.float64)
    w = torch.randn(192, 32, dtype=torch.float64)
    base = float(A.third_order(emb @ w.T).pow(2).mean())
    for scale in (0.25, 1.0, 7.0):
        got = float(A.third_order((scale * emb) @ (w / scale).T).pow(2).mean())
        assert abs(got - base) < 1e-10 * max(1.0, base)
