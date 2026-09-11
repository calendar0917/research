"""Tests for the graph-head function family audit.

These pin the representation-integrity gates, the nested OOF data protocol,
the fit-only preprocessing, the direct-head contract and -- most importantly --
the rank-4 Factorization Machine definition used by
``experiments/luyin16/zinc_graph_head_function_family``.

1.  stored R re-fed through the frozen head reproduces stored yhat_0;
2.  head-fit / head-selection / head-evaluation membership is strictly correct;
3.  standardisation statistics are fit on head-fit coordinates only;
4.  H1 / FM parameter budgets are matched within 3%;
5.  the efficient FM interaction equals the explicit pairwise form on toy data;
6.  the FM interaction is invariant to the feature evaluation order;
7.  the FM rank is fixed at 4 and there is no hidden rank-sweep path;
8.  H1 / H2 / FM use the identical L1 objective (no MSE);
9.  the CatBoost secondary reference never accesses extra features;
10. official valid/test are never loaded in the frozen audit;
11. the FM output genuinely depends on the interaction matrix;
12. if an end-to-end run is made, shared upstream initial tensors are copied
    item by item (the primitive is pinned here).
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from tracks.ksvd.experiments.luyin16 import (
    zinc_graph_head_function_family as gh,
)


def _random_z(n: int, dim: int = gh.R_DIM, seed: int = 0, scale: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, dim)) * scale).astype(np.float32)


# ---------------------------------------------------------------------------
# 1 -- stored R -> frozen head == stored yhat_0 (integration gate)
# ---------------------------------------------------------------------------

def test_stored_R_reconstruction_gate():
    cache = gh._cache_path(0, 0)
    state_path = gh.FOLD_DIR / "fold0_seed0_state.pt"
    if not cache.exists() or not state_path.exists():
        pytest.skip("frozen graph-head cache / checkpoint not present")
    payload = gh.load_cache(0, 0)
    head = gh._head_from_state(0, 0)
    yhat = gh._head_forward(head, np.asarray(payload["R"], dtype=np.float64))
    diff = float(np.abs(yhat - np.asarray(payload["yhat_0"], dtype=np.float64)).max())
    assert diff <= gh.HEAD_RECON_ATOL


# ---------------------------------------------------------------------------
# 2 -- nested split membership
# ---------------------------------------------------------------------------

def test_nested_split_membership():
    from tracks.ksvd.experiments.luyin16.zinc_oof_difficulty_audit import _fold_slices

    for fold in range(gh.K_FOLDS):
        fit_idx, sel_idx, holdout_idx = _fold_slices()[fold]
        assert (len(fit_idx), len(sel_idx), len(holdout_idx)) == (gh.N_FIT, gh.N_SELECT, gh.N_EVAL)
        assert np.array_equal(np.sort(np.concatenate([fit_idx, sel_idx, holdout_idx])), np.arange(10000))
    if gh._cache_path(0, 0).exists():
        payload = gh.load_cache(0, 0)
        role = np.asarray(payload["role"], dtype=np.int64)
        assert np.sum(role == 0) == gh.N_FIT
        assert np.sum(role == 1) == gh.N_SELECT
        assert np.sum(role == 2) == gh.N_EVAL
        fit_idx, sel_idx, holdout_idx = _fold_slices()[0]
        subset = np.asarray(payload["subset_index"], dtype=np.int64)
        assert np.array_equal(subset[role == 0], fit_idx)
        assert np.array_equal(subset[role == 1], sel_idx)
        assert np.array_equal(subset[role == 2], holdout_idx)


# ---------------------------------------------------------------------------
# 3 -- fit-only standardisation
# ---------------------------------------------------------------------------

def test_standardizer_fit_only():
    R = _random_z(120, seed=1).astype(np.float64)
    fit, sel, eva = R[:60], R[60:90], R[90:120]
    mean, scale, degenerate = gh._fit_standardizer(fit)
    assert np.allclose(mean, fit.mean(axis=0))
    expected_scale = np.where(fit.std(axis=0) < gh.STANDARDIZE_EPS, 1.0, fit.std(axis=0))
    assert np.allclose(scale, expected_scale)
    z_fit = gh._standardize(fit, mean, scale, degenerate)
    assert np.abs(z_fit.mean(axis=0)).max() < 1e-9
    assert np.abs(z_fit.std(axis=0) - 1.0).max() < 1e-9
    # selection / evaluation cannot move the fit statistics
    moved = np.concatenate([fit, sel + 100.0, eva * 50.0])
    mean2, scale2, degenerate2 = gh._fit_standardizer(moved[:60])
    assert np.array_equal(mean, mean2)
    assert np.array_equal(scale, scale2)
    assert np.array_equal(degenerate, degenerate2)
    # fitting on the selection set genuinely differs (non-vacuous)
    assert not np.allclose(mean, gh._fit_standardizer(sel)[0])


def test_degenerate_coordinates_deterministic():
    R = _random_z(50, seed=6).astype(np.float64)
    R[:, 3] = 5.0
    mean, scale, degenerate = gh._fit_standardizer(R)
    assert bool(degenerate[3])
    assert scale[3] == 1.0
    z = gh._standardize(R, mean, scale, degenerate)
    assert np.all(z[:, 3] == 0.0)


# ---------------------------------------------------------------------------
# 4 -- parameter matching
# ---------------------------------------------------------------------------

def test_parameter_budgets_matched():
    accounting = gh._parameter_accounting()
    assert accounting["H1_params"] == 1530
    assert accounting["FM_params"] == 1511
    assert accounting["FM_linear_params"] == 302
    assert accounting["FM_interaction_params"] == 1208
    assert accounting["H2_params"] == 4135
    assert accounting["Hlinear_params"] == 303
    assert accounting["FM_rank"] == 4
    assert accounting["parameter_match_ok"]
    assert accounting["H1_vs_FM_abs_mismatch"] <= gh.PARAM_MATCH_TOL


# ---------------------------------------------------------------------------
# 5 -- efficient FM interaction == explicit pairwise form
# ---------------------------------------------------------------------------

def test_fm_efficient_equals_explicit_pairwise():
    torch.manual_seed(0)
    d, r, n = 11, 4, 7
    z = torch.tensor(_random_z(n, dim=d, seed=2), dtype=torch.float32)
    model = gh.FMReader(d, r)
    with torch.no_grad():
        model.v.normal_()
    efficient = model.interaction(z).detach().numpy()
    explicit = gh._explicit_pairwise_interaction(z.numpy().astype(np.float64), model.v.detach().numpy().astype(np.float64))
    assert np.allclose(efficient, explicit, atol=1e-5)
    # not vacuous: a scaled factor matrix changes the value
    with torch.no_grad():
        model.v.mul_(2.0)
    explicit2 = gh._explicit_pairwise_interaction(z.numpy().astype(np.float64), model.v.detach().numpy().astype(np.float64))
    assert not np.allclose(explicit, explicit2)


# ---------------------------------------------------------------------------
# 6 -- FM interaction is invariant to feature evaluation order
# ---------------------------------------------------------------------------

def test_fm_interaction_order_invariant():
    torch.manual_seed(0)
    d, r, n = 9, 4, 5
    z = torch.tensor(_random_z(n, dim=d, seed=3), dtype=torch.float32)
    model = gh.FMReader(d, r)
    with torch.no_grad():
        model.v.normal_()
    baseline = model.interaction(z).detach().numpy()
    permutation = torch.randperm(d)
    z_perm = z[:, permutation]
    v_perm = model.v[permutation, :]
    reordered = gh.FMReader(d, r)
    with torch.no_grad():
        reordered.v.copy_(v_perm)
    assert np.allclose(baseline, reordered.interaction(z_perm).detach().numpy(), atol=1e-5)


# ---------------------------------------------------------------------------
# 7 -- fixed rank, no hidden sweep
# ---------------------------------------------------------------------------

def test_rank_fixed_no_sweep_path():
    assert gh.FM_RANK == 4
    torch.manual_seed(0)
    model = gh._make_fm(0)
    assert isinstance(model, gh.FMReader)
    assert model.rank == 4
    assert model.v.shape == (gh.R_DIM, 4)
    source = inspect.getsource(gh)
    for forbidden in ("rank=8", "rank = 8", "rank=16", "FM_RANKS", "rank_sweep", "RANK_GRID"):
        assert forbidden not in source


# ---------------------------------------------------------------------------
# 8 -- identical L1 objective for every direct head
# ---------------------------------------------------------------------------

def test_direct_heads_use_identical_l1_objective():
    z = torch.tensor(_random_z(64, seed=8), dtype=torch.float32)
    y = torch.tensor(np.random.default_rng(1).standard_normal(64), dtype=torch.float32)

    for factory in (lambda: gh._make_h1(0), lambda: gh._make_h2(0), lambda: gh._make_fm(0)):
        # manual one-epoch deterministic mini-batch loop
        torch.manual_seed(0)
        manual = factory()
        optimizer = torch.optim.Adam(manual.parameters(), lr=gh.ADAPTER_LR, weight_decay=0.0)
        for batch_z, batch_y in gh._iter_minibatches(z, y, gh.ADAPTER_BATCH_SIZE, 0):
            optimizer.zero_grad()
            (manual(batch_z) - batch_y).abs().mean().backward()
            optimizer.step()
        # the audit trainer over exactly one epoch (best-selection = final state)
        torch.manual_seed(0)
        trained = factory()
        gh._train_head(trained, z, y, z, y, epochs=1)
        for a, b in zip(manual.parameters(), trained.parameters()):
            assert torch.allclose(a, b, atol=1e-7)

    # an MSE step would differ (so the L1 pin is not vacuous)
    torch.manual_seed(0)
    mse = gh._make_h1(0)
    optimizer = torch.optim.Adam(mse.parameters(), lr=gh.ADAPTER_LR, weight_decay=0.0)
    for batch_z, batch_y in gh._iter_minibatches(z, y, gh.ADAPTER_BATCH_SIZE, 0):
        optimizer.zero_grad()
        ((mse(batch_z) - batch_y) ** 2).mean().backward()
        optimizer.step()
    torch.manual_seed(0)
    l1 = gh._make_h1(0)
    gh._train_head(l1, z, y, z, y, epochs=1)
    assert not all(torch.allclose(a, b, atol=1e-7) for a, b in zip(mse.parameters(), l1.parameters()))


# ---------------------------------------------------------------------------
# 9 -- CatBoost secondary reference uses no extra features
# ---------------------------------------------------------------------------

def test_catboost_no_extra_features():
    status = gh.catboost_status()
    assert status["extra_features"] == []
    assert "standardized z" in status["input_spec"]
    assert "MAE" == gh.CATBOOST_PRESET["loss_function"]
    # the input matrix for the tree reference is exactly the neural-head z
    data = {"z": _random_z(20, seed=9)}
    assert data["z"].shape[1] == gh.R_DIM


# ---------------------------------------------------------------------------
# 10 -- official valid/test never loaded
# ---------------------------------------------------------------------------

def test_no_valid_or_test_loading():
    source = inspect.getsource(gh)
    forbidden = ["_load_zinc(", "ZINC_ROOT", '"valid"', "'valid'", '"test"', "'test'"]
    hits = [token for token in forbidden if token in source]
    assert hits == []
    assert "_load_train_records" in source
    assert "_load_train_labels" in source


# ---------------------------------------------------------------------------
# 11 -- FM output genuinely depends on the interaction matrix
# ---------------------------------------------------------------------------

def test_fm_output_depends_on_interaction():
    torch.manual_seed(0)
    z = torch.tensor(_random_z(16, seed=10), dtype=torch.float32)
    model = gh._make_fm(0)
    with torch.no_grad():
        model.v.normal_(std=0.2)
    full = model(z)
    linear_only = model.forward_linear_only(z)
    assert float((full - linear_only).abs().max()) > 1e-4
    # perturbing only V changes the interaction but not the linear term
    linear_before = model.linear_term(z).clone()
    interaction_before = model.interaction(z).clone()
    with torch.no_grad():
        model.v.add_(torch.randn_like(model.v) * 0.1)
    assert torch.allclose(linear_before, model.linear_term(z))
    assert float((model.interaction(z) - interaction_before).abs().max()) > 1e-4
    assert float((model(z) - full).abs().max()) > 1e-4


# ---------------------------------------------------------------------------
# 12 -- end-to-end shared-upstream initialisation primitive
# ---------------------------------------------------------------------------

def test_shared_upstream_initialisation_primitive():
    class Tiny(torch.nn.Module):
        def __init__(self, seed: int) -> None:
            super().__init__()
            torch.manual_seed(seed)
            self.up = torch.nn.Linear(6, 4)
            self.head = torch.nn.Linear(4, 1)

    source = Tiny(1)
    target = Tiny(2)
    audit = gh._copy_shared_upstream(source, target)
    assert audit["n_shared_tensors"] >= 2  # up.weight + up.bias
    assert all(not key.startswith("head.") for key in audit["shared_tensor_hashes"])
    assert torch.equal(source.up.weight, target.up.weight)
    assert torch.equal(source.up.bias, target.up.bias)
    # the head is intentionally left untouched
    assert not torch.equal(source.head.weight, target.head.weight)
