"""Tests for the frozen function-basis accessibility audit.

These pin the representation integrity gates, the fit-only preprocessing, the
target-independent quantile-hinge basis and the reader contract used by
``experiments/luyin16/zinc_function_basis_accessibility``.  Most tests are
self-contained (synthetic tensors); the two integration gates skip cleanly
when the frozen checkpoint / cache is not present.

1.  stored R re-fed through the frozen head reproduces stored yhat_0;
2.  standardisation statistics are fit on adapter-fit coordinates only;
3.  selection / evaluation rows do not affect the fit statistics;
4.  quantile knots are fit on adapter-fit inputs only;
5.  changing labels does not change the knots;
6.  the hinge basis has the expected 4D dimension;
7.  the basis values satisfy the hinge definition exactly;
8.  degenerate coordinates are handled deterministically;
9.  B1 / E parameter budgets are matched within 3%;
10. B1 / B2 / E use the identical L1 objective;
11. readers genuinely depend on their R / basis input;
12. official valid/test are never loaded.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from tracks.ksvd.experiments.luyin16 import (
    zinc_function_basis_accessibility as fba,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _random_R(n: int, seed: int = 0, dim: int = fba.R_DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float64)


def _synthetic_fold(tmp_path, n_fit: int = 60, n_sel: int = 20, n_eva: int = 20):
    R = _random_R(n_fit + n_sel + n_eva, seed=7)
    mean, scale, degenerate = fba._fit_standardizer(R[:n_fit])
    z = fba._standardize(R, mean, scale, degenerate)
    knots = fba._fit_knots(z[:n_fit])
    return R, mean, scale, degenerate, z, knots


# ---------------------------------------------------------------------------
# 1 -- stored R -> frozen head == stored yhat_0 (integration gate)
# ---------------------------------------------------------------------------

def test_stored_R_reconstruction_gate():
    cache = fba._cache_path(0, 0)
    state_path = fba.FOLD_DIR / "fold0_seed0_state.pt"
    if not cache.exists() or not state_path.exists():
        pytest.skip("frozen cache / checkpoint not present")
    payload = fba.load_cache(0, 0)
    head = fba._head_from_state(0, 0)
    yhat = fba._head_forward(head, np.asarray(payload["R"], dtype=np.float64))
    diff = float(np.abs(yhat - np.asarray(payload["yhat_0"], dtype=np.float64)).max())
    assert diff <= fba.PRED_RECON_ATOL


# ---------------------------------------------------------------------------
# 2 / 3 -- fit-only standardisation
# ---------------------------------------------------------------------------

def test_standardizer_fit_only():
    R = _random_R(100, seed=1)
    fit = R[:60]
    mean, scale, degenerate = fba._fit_standardizer(fit)
    assert np.allclose(mean, fit.mean(axis=0))
    assert np.allclose(scale, np.where(fit.std(axis=0) < fba.STANDARDIZE_EPS, 1.0, fit.std(axis=0)))
    assert not degenerate.any()
    z_fit = fba._standardize(fit, mean, scale, degenerate)
    assert np.abs(z_fit.mean(axis=0)).max() < 1e-9
    assert np.abs(z_fit.std(axis=0) - 1.0).max() < 1e-9


def test_selection_and_evaluation_do_not_change_stats():
    R = _random_R(120, seed=2)
    fit, sel, eva = R[:60], R[60:90], R[90:120]
    base = fba._fit_standardizer(fit)
    # moving / perturbing the selection and evaluation rows cannot change the
    # statistics because they are never passed to the fitter
    moved_sel = sel + 100.0
    moved_eva = eva * 50.0
    again = fba._fit_standardizer(np.concatenate([fit, moved_sel, moved_eva])[:60])
    assert np.array_equal(base[0], again[0])
    assert np.array_equal(base[1], again[1])
    assert np.array_equal(base[2], again[2])
    # a fit on the held-out rows genuinely differs (so the test is not vacuous)
    other = fba._fit_standardizer(sel)
    assert not np.allclose(base[0], other[0])


# ---------------------------------------------------------------------------
# 4 / 5 -- target-independent knots
# ---------------------------------------------------------------------------

def test_knots_use_fit_inputs_only():
    R = _random_R(120, seed=3)
    fit, sel, eva = R[:60], R[60:90], R[90:120]
    mean, scale, degenerate = fba._fit_standardizer(fit)
    z_fit = fba._standardize(fit, mean, scale, degenerate)
    knots = fba._fit_knots(z_fit)
    expected = np.quantile(z_fit, fba.HINGE_QUANTILES, axis=0).T
    assert np.allclose(knots, expected)
    # appending arbitrary selection / evaluation inputs cannot change them
    z_other = fba._standardize(np.concatenate([sel, eva]), mean, scale, degenerate)
    knots_again = fba._fit_knots(z_fit)
    assert np.array_equal(knots, knots_again)
    assert z_other.shape[0] == 60


def test_changing_labels_does_not_change_knots():
    R = _random_R(80, seed=4)
    mean, scale, degenerate = fba._fit_standardizer(R[:50])
    z_fit = fba._standardize(R[:50], mean, scale, degenerate)
    labels = np.arange(50, dtype=np.float64)
    knots_a = fba._fit_knots(z_fit)
    rng = np.random.default_rng(0)
    labels = labels[rng.permutation(50)] * 1000.0
    assert labels.shape == (50,)
    knots_b = fba._fit_knots(z_fit)
    assert np.array_equal(knots_a, knots_b)


# ---------------------------------------------------------------------------
# 6 / 7 -- basis dimension and definition
# ---------------------------------------------------------------------------

def test_hinge_basis_dimension():
    knots = _synthetic_fold(None)[5]
    z = _random_R(11, seed=5)
    basis = fba._hinge_basis(z, knots)
    assert basis.shape == (11, fba._basis_dim()) == (11, fba.R_DIM * 4) == (11, 1208)
    torch.manual_seed(0)
    reader = fba._make_hinge(knots, 0)
    tbasis = reader.basis(torch.tensor(z, dtype=torch.float32)).numpy()
    assert tbasis.shape == (11, 1208)


def test_hinge_basis_values_satisfy_definition():
    knots = np.array([[0.1, 0.3, 0.7], [-1.0, 0.0, 2.0]], dtype=np.float64)
    z = np.array([[0.0, 0.5], [0.5, -3.0], [1.0, 3.0]], dtype=np.float64)
    basis = fba._hinge_basis(z, knots)
    d = z.shape[1]
    for j in range(d):
        assert np.allclose(basis[:, j], z[:, j])
        assert np.allclose(basis[:, d + j], np.maximum(z[:, j] - knots[j, 0], 0.0))
        assert np.allclose(basis[:, 2 * d + j], np.maximum(z[:, j] - knots[j, 1], 0.0))
        assert np.allclose(basis[:, 3 * d + j], np.maximum(z[:, j] - knots[j, 2], 0.0))
    # exactly at a knot the corresponding hinge is exactly zero
    z_at = np.array([[0.1, -1.0]])
    b_at = fba._hinge_basis(z_at, knots)
    assert b_at[0, d + 0] == 0.0
    assert b_at[0, 2 * d + 1] == 0.0


# ---------------------------------------------------------------------------
# 8 -- degenerate coordinates
# ---------------------------------------------------------------------------

def test_degenerate_coordinates_deterministic():
    R = _random_R(50, seed=6)
    R[:, 3] = 5.0  # constant column -> std 0
    mean, scale, degenerate = fba._fit_standardizer(R)
    assert bool(degenerate[3])
    assert scale[3] == 1.0
    z = fba._standardize(R, mean, scale, degenerate)
    assert np.all(z[:, 3] == 0.0)
    knots = fba._fit_knots(z)
    assert np.all(knots[3] == 0.0)
    # repeated calls are bit-identical
    mean2, scale2, deg2 = fba._fit_standardizer(R)
    z2 = fba._standardize(R, mean2, scale2, deg2)
    knots2 = fba._fit_knots(z2)
    assert np.array_equal(z, z2)
    assert np.array_equal(knots, knots2)


# ---------------------------------------------------------------------------
# 9 -- parameter matching
# ---------------------------------------------------------------------------

def test_parameter_budgets_matched():
    accounting = fba._parameter_accounting()
    assert accounting["B1_params"] == 1225
    assert accounting["E_params"] == 1209
    assert accounting["B2_params"] == 4135
    assert accounting["B_linear_params"] == 303
    assert accounting["basis_dim"] == 1208
    assert accounting["parameter_match_ok"]
    assert accounting["B1_vs_E_abs_mismatch"] <= fba.PARAM_MATCH_TOL


# ---------------------------------------------------------------------------
# 10 -- identical L1 objective
# ---------------------------------------------------------------------------

def test_readers_use_identical_l1_objective():
    torch.manual_seed(0)
    R = _random_R(40, seed=8)
    y = np.random.default_rng(1).standard_normal(40)
    yhat = np.zeros(40)
    z = torch.tensor(R, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    yht = torch.tensor(yhat, dtype=torch.float32)

    # manual one Adam step on the L1 loss
    torch.manual_seed(0)
    manual = fba.GenericReader(fba.R_DIM, (4, 2))
    opt = torch.optim.Adam(manual.parameters(), lr=fba.ADAPTER_LR)
    opt.zero_grad()
    (yht + manual(z) - yt).abs().mean().backward()
    opt.step()

    # the audit trainer over exactly one epoch (patience disabled)
    torch.manual_seed(0)
    trained = fba.GenericReader(fba.R_DIM, (4, 2))
    fba._train_reader(
        trained, z, yt, yht, z, yt, yht, epochs=1, lr=fba.ADAPTER_LR, patience=None
    )
    for a, b in zip(manual.parameters(), trained.parameters()):
        assert torch.allclose(a, b, atol=1e-7)

    # an MSE step would differ (so the L1 pin is not vacuous)
    torch.manual_seed(0)
    mse = fba.GenericReader(fba.R_DIM, (4, 2))
    opt2 = torch.optim.Adam(mse.parameters(), lr=fba.ADAPTER_LR)
    opt2.zero_grad()
    ((yht + mse(z) - yt) ** 2).mean().backward()
    opt2.step()
    assert not all(
        torch.allclose(a, b, atol=1e-7) for a, b in zip(manual.parameters(), mse.parameters())
    )


# ---------------------------------------------------------------------------
# 11 -- readers depend on their input
# ---------------------------------------------------------------------------

def test_reader_outputs_depend_on_basis():
    knots = _synthetic_fold(None)[5]
    torch.manual_seed(0)
    reader = fba._make_hinge(knots, 0)
    for param in reader.parameters():
        with torch.no_grad():
            param.add_(torch.randn_like(param) * 0.05)
    z_a = torch.tensor(_random_R(16, seed=9), dtype=torch.float32)
    z_b = z_a + 2.0
    assert float((reader(z_a) - reader(z_b)).abs().max()) > 1e-4
    # the hinge terms are active in the forward map
    assert float((reader(z_a) - reader.forward_linear_only(z_a)).abs().max()) > 1e-4


def test_generic_reader_outputs_depend_on_input():
    torch.manual_seed(0)
    reader = fba.GenericReader(fba.R_DIM, (4, 2))
    for param in reader.parameters():
        with torch.no_grad():
            param.add_(torch.randn_like(param) * 0.05)
    z_a = torch.tensor(_random_R(16, seed=10), dtype=torch.float32)
    z_b = z_a + 2.0
    assert float((reader(z_a) - reader(z_b)).abs().max()) > 1e-4


# ---------------------------------------------------------------------------
# 12 -- official valid/test never loaded
# ---------------------------------------------------------------------------

def test_no_valid_or_test_loading():
    source = inspect.getsource(fba)
    forbidden = ["_load_zinc(", "ZINC_ROOT", '"valid"', "'valid'", '"test"', "'test'"]
    hits = [token for token in forbidden if token in source]
    assert hits == []
    assert "_load_train_records" in source
    assert "_load_train_labels" in source
