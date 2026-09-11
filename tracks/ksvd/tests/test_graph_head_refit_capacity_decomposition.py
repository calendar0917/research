"""Tests for the graph-head refit & capacity decomposition audit.

These pin the pre-registered protocol of
``experiments/luyin16/zinc_graph_head_refit_capacity_decomposition``:

1.  stored R re-fed through the frozen reconstructed head reproduces yhat_0;
2.  head-fit / head-selection / head-evaluation membership is correct;
3.  standardisation statistics use the 7200 head-fit molecules only;
4.  the original raw head and the first-layer-reparameterised head agree at
    optimisation step 0;
5.  the first-layer standardisation transform is algebraically exact;
6.  S-scratch / L-scratch / L-warm share loss, optimizer, batch size, horizon;
7.  L-warm step 0 is function-equivalent to the original checkpoint head;
8.  the scratch architectures are fixed (no hidden-width sweep path);
9.  official valid / official test are never loaded;
10. parameter counts are exact;
11. the learning-curve logger does not change training;
12. the bootstrap pairs molecules across the two backbones.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from tracks.ksvd.experiments.luyin16 import (
    zinc_graph_head_refit_capacity_decomposition as cd,
)


def _random(n: int, dim: int = cd.R_DIM, seed: int = 0, scale: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, dim)) * scale).astype(np.float32)


def _cache_available() -> bool:
    return all(cd.gh._cache_path(f, s).exists() for s in (0, 1) for f in range(cd.K_FOLDS))


# ---------------------------------------------------------------------------
# 1 -- stored R -> frozen reconstructed head == stored yhat_0
# ---------------------------------------------------------------------------

def test_stored_R_reconstruction_gate():
    if not cd.gh._cache_path(0, 0).exists():
        pytest.skip("frozen R cache not present")
    bundle = cd.fold_bundle(0, 0)
    head = cd.load_original_head(0, 0)
    pred = cd.head_forward(head, bundle["R"])
    diff = float(np.abs(pred - bundle["yhat_0"]).max())
    assert diff <= cd.HEAD_RECON_ATOL


# ---------------------------------------------------------------------------
# 2 -- nested split membership
# ---------------------------------------------------------------------------

def test_nested_split_role_membership():
    from tracks.ksvd.experiments.luyin16.zinc_oof_difficulty_audit import _fold_slices

    for fold in range(cd.K_FOLDS):
        fit, sel, eva = _fold_slices()[fold]
        assert (len(fit), len(sel), len(eva)) == (cd.N_FIT, cd.N_SELECT, cd.N_EVAL)
        assert np.array_equal(
            np.sort(np.concatenate([fit, sel, eva])), np.arange(10000)
        )
    if cd.gh._cache_path(0, 0).exists():
        bundle = cd.fold_bundle(0, 0)
        role = bundle["role"]
        fit, sel, eva = _fold_slices()[0]
        idx = bundle["subset_index"]
        assert np.array_equal(idx[role == 0], fit)
        assert np.array_equal(idx[role == 1], sel)
        assert np.array_equal(idx[role == 2], eva)


# ---------------------------------------------------------------------------
# 3 -- fit-only standardisation
# ---------------------------------------------------------------------------

def test_standardizer_uses_fit_only():
    R = _random(120, seed=1).astype(np.float64)
    fit, sel, eva = R[:60], R[60:90], R[90:120]
    mean, scale, degenerate = cd.gh._fit_standardizer(fit)
    z = cd.gh._standardize(R, mean, scale, degenerate)
    assert np.abs(z[:60].mean(axis=0)).max() < 1e-9
    # moving the selection / evaluation rows cannot move the statistics
    moved = np.concatenate([fit, sel + 100.0, eva * 50.0])
    mean2, scale2, deg2 = cd.gh._fit_standardizer(moved[:60])
    assert np.array_equal(mean, mean2)
    assert np.array_equal(scale, scale2)
    assert np.array_equal(degenerate, deg2)
    # fitting on selection alone differs (non-vacuous)
    assert not np.allclose(mean, cd.gh._fit_standardizer(sel)[0])


def test_fold_bundle_statistics_only_fit():
    if not cd.gh._cache_path(0, 0).exists():
        pytest.skip("frozen R cache not present")
    bundle = cd.fold_bundle(0, 0)
    fit = bundle["fit_pos"]
    mean, scale, degenerate = cd.gh._fit_standardizer(bundle["R"][fit])
    assert np.array_equal(mean, bundle["mean"])
    assert np.array_equal(scale, bundle["scale"])
    assert np.array_equal(degenerate, bundle["degenerate"])


# ---------------------------------------------------------------------------
# 4 -- original raw head == transformed standardised head at step 0
# ---------------------------------------------------------------------------

def test_standardisation_reparameterisation_step0_equivalent():
    if not cd.gh._cache_path(0, 0).exists():
        pytest.skip("frozen R cache not present")
    bundle = cd.fold_bundle(0, 0)
    original = cd.load_original_head(0, 0)
    transformed = cd.transform_first_layer(original, bundle["mean"], bundle["scale"])
    raw_pred = cd.head_forward(original, bundle["R"])
    std_pred = cd.head_forward(transformed, bundle["z"])
    diff = float(np.abs(raw_pred - std_pred).max())
    assert diff <= cd.STANDARDIZATION_EQUIV_ATOL
    # the float64 identity on the reconstructed input is exact
    assert float(np.abs(raw_pred - bundle["yhat_0"]).max()) <= cd.HEAD_RECON_ATOL


# ---------------------------------------------------------------------------
# 5 -- first-layer transform formula (toy)
# ---------------------------------------------------------------------------

def test_first_layer_transform_formula_toy():
    rng = np.random.default_rng(0)
    d, h, n = 7, 5, 11
    w = rng.standard_normal((h, d))
    b = rng.standard_normal(h)
    mean = rng.standard_normal(d) * 3.0
    scale = np.abs(rng.standard_normal(d)) + 0.1
    R = rng.standard_normal((n, d)) * scale + mean
    z = (R - mean) / scale
    w2, b2 = cd.transform_first_layer_reference(w, b, mean, scale)
    assert np.allclose(R @ w.T + b, z @ w2.T + b2, atol=1e-10)
    # non-vacuous: without the bias correction the identity breaks
    assert not np.allclose(R @ w.T + b, z @ w2.T + b, atol=1e-3)

    # module-level round trip on a small torch head with the same layout
    d2, h2 = 7, 5
    head = torch.nn.Sequential(
        torch.nn.Linear(d2, h2),
        torch.nn.LayerNorm(h2),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(h2, 1),
    )
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.normal_()
    head.eval()
    transformed = cd.transform_first_layer(head, mean, scale)
    in64 = cd.deepcopy(head).double()
    in64_t = cd.deepcopy(transformed).double()
    with torch.no_grad():
        a = in64(torch.tensor(R)).view(-1).numpy()
        bpred = in64_t(torch.tensor(z)).view(-1).numpy()
    assert np.allclose(a, bpred, atol=1e-10)


# ---------------------------------------------------------------------------
# 6 -- identical loss / optimizer / batch / horizon for S / L / L-warm
# ---------------------------------------------------------------------------

def test_scratch_heads_share_training_protocol():
    x = torch.tensor(_random(256, seed=3), dtype=torch.float32)
    y = torch.tensor(np.random.default_rng(4).standard_normal(256), dtype=torch.float32)

    for hidden in (cd.S_HIDDEN, cd.L_HIDDEN):
        torch.manual_seed(cd.HEAD_SEED)
        manual = cd.GenericReader(cd.R_DIM, hidden)
        optimizer = torch.optim.Adam(
            manual.parameters(), lr=cd.REFIT_LR, weight_decay=cd.REFIT_WEIGHT_DECAY
        )
        for bx, by in cd.gh._iter_minibatches(x, y, cd.REFIT_BATCH_SIZE, 0, cd.HEAD_SEED):
            optimizer.zero_grad()
            (cd._flat_forward(manual, bx) - by).abs().mean().backward()
            optimizer.step()

        torch.manual_seed(cd.HEAD_SEED)
        trained = cd.GenericReader(cd.R_DIM, hidden)
        cd.train_refit(trained, x, y, x, y, epochs=1, trace_epochs=())

        for a, b in zip(manual.parameters(), trained.parameters()):
            assert torch.allclose(a, b, atol=1e-7)

    # an MSE step would differ, so the L1 pin is not vacuous
    torch.manual_seed(cd.HEAD_SEED)
    mse = cd.GenericReader(cd.R_DIM, cd.S_HIDDEN)
    optimizer = torch.optim.Adam(mse.parameters(), lr=cd.REFIT_LR)
    for bx, by in cd.gh._iter_minibatches(x, y, cd.REFIT_BATCH_SIZE, 0, cd.HEAD_SEED):
        optimizer.zero_grad()
        ((cd._flat_forward(mse, bx) - by) ** 2).mean().backward()
        optimizer.step()
    torch.manual_seed(cd.HEAD_SEED)
    l1 = cd.GenericReader(cd.R_DIM, cd.S_HIDDEN)
    cd.train_refit(l1, x, y, x, y, epochs=1, trace_epochs=())
    assert not all(torch.allclose(a, b, atol=1e-7) for a, b in zip(mse.parameters(), l1.parameters()))


# ---------------------------------------------------------------------------
# 7 -- L-warm initialises to the original checkpoint function
# ---------------------------------------------------------------------------

def test_lwarm_initialises_to_original_function():
    if not cd.gh._cache_path(0, 0).exists():
        pytest.skip("frozen R cache not present")
    bundle = cd.fold_bundle(0, 0)
    original = cd.load_original_head(0, 0)
    warm = cd.transform_first_layer(original, bundle["mean"], bundle["scale"])
    # before any training, warm(z) == original(R) on every molecule
    diff = float(
        np.abs(cd.head_forward(original, bundle["R"]) - cd.head_forward(warm, bundle["z"])).max()
    )
    assert diff <= cd.STANDARDIZATION_EQUIV_ATOL
    # and warm's very first selection/fit MAE equals the original head's
    sel = bundle["sel_pos"]
    y = torch.tensor(bundle["y"][sel], dtype=torch.float32)
    warm.eval()
    with torch.no_grad():
        z_sel = torch.tensor(bundle["z"][sel], dtype=torch.float32)
        warm_mae = float((cd._flat_forward(warm, z_sel) - y).abs().mean())
    orig_mae = float(
        np.abs(bundle["yhat_0"][sel] - bundle["y"][sel]).mean()
    )
    assert abs(warm_mae - orig_mae) < 1e-4


# ---------------------------------------------------------------------------
# 8 -- fixed architectures, no hidden-width sweep path
# ---------------------------------------------------------------------------

def test_fixed_architectures_no_sweep_path():
    assert cd.S_HIDDEN == (13, 13)
    assert cd.L_HIDDEN == (64, 32)
    assert cd.ORIGINAL_ARCH == (64, 32)
    source = inspect.getsource(cd)
    for forbidden in ("WIDTH_GRID", "width_sweep", "HIDDEN_GRID", "sweep_widths"):
        assert forbidden not in source


# ---------------------------------------------------------------------------
# 9 -- official valid / test never loaded
# ---------------------------------------------------------------------------

def test_no_valid_or_test_loading():
    source = inspect.getsource(cd)
    forbidden = ["_load_zinc(", "ZINC_ROOT", '"valid"', "'valid'", '"test"', "'test'"]
    hits = [token for token in forbidden if token in source]
    assert hits == []
    assert "_load_train_labels" in source


# ---------------------------------------------------------------------------
# 10 -- exact parameter counts
# ---------------------------------------------------------------------------

def test_parameter_counts_exact():
    counts = cd.parameter_counts()
    assert counts["H0_original_head"]["params"] == 21633
    assert counts["H0_original_head"]["linear_params"] == 21505
    assert counts["H0_original_head"]["layernorm_params"] == 128
    assert counts["S_scratch_head"]["params"] == 4135
    assert counts["L_scratch_head"]["params"] == 21505
    assert cd.count_parameters(cd.make_scratch_head(cd.S_HIDDEN)) == 4135
    assert cd.count_parameters(cd.make_scratch_head(cd.L_HIDDEN)) == 21505
    # replacement saving is exact: original head minus small head
    per_fold = counts["per_fold_total_model"]
    for entry in per_fold.values():
        assert entry["small_head_replaced_total"] == entry["total_params"] - 21633 + 4135
        assert entry["saving_vs_original_head"] == 17498


# ---------------------------------------------------------------------------
# 11 -- learning-curve logger cannot change training
# ---------------------------------------------------------------------------

def test_trace_logger_does_not_change_training():
    x = torch.tensor(_random(200, dim=16, seed=5), dtype=torch.float32)
    y = torch.tensor(np.random.default_rng(6).standard_normal(200), dtype=torch.float32)
    torch.manual_seed(cd.HEAD_SEED)
    a = cd.GenericReader(16, (5, 3))
    torch.manual_seed(cd.HEAD_SEED)
    b = cd.GenericReader(16, (5, 3))
    info_a = cd.train_refit(a, x, y, x, y, epochs=6, trace_epochs=())
    info_b = cd.train_refit(b, x, y, x, y, epochs=6, trace_epochs=(2, 4))
    assert info_b["trace"] and not info_a["trace"]
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)
    assert info_a["best_selection_mae"] == info_b["best_selection_mae"]


# ---------------------------------------------------------------------------
# 12 -- bootstrap pairs molecules across the two backbones
# ---------------------------------------------------------------------------

def test_bootstrap_molecule_pairing(tmp_path):
    path = tmp_path / "molecule_errors.npz"
    subset_index = np.array([10, 10, 20, 20, 30, 30], dtype=np.int64)
    fold = np.array([0, 0, 1, 1, 0, 0], dtype=np.int64)
    err_h0 = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    err_s = np.array([0.5, 0.7, 1.5, 1.7, 2.5, 2.7])
    err_lscratch = np.array([0.6, 0.8, 1.6, 1.8, 2.6, 2.8])
    err_lwarm = np.array([0.4, 0.6, 1.4, 1.6, 2.4, 2.6])
    err_sraw = np.array([0.45, 0.65, 1.45, 1.65, 2.45, 2.65])
    np.savez_compressed(
        path,
        subset_index=subset_index,
        fold=fold,
        err_h0=err_h0,
        err_s=err_s,
        err_lscratch=err_lscratch,
        err_lwarm=err_lwarm,
        err_sraw=err_sraw,
    )
    delta = cd.molecule_deltas(errors_path=path)
    # one unit per molecule, not per (molecule, backbone) row
    assert delta["delta_cap"].shape == (3,)
    assert np.array_equal(delta["_molecule_ids"], np.array([10, 20, 30]))
    assert np.array_equal(delta["_fold_ids"], np.array([0, 1, 0]))
    # values are means over the two backbone rows
    # delta_cap(m=10) = mean([0.6,0.8]) - mean([0.5,0.7]) = 0.1
    assert abs(delta["delta_cap"][0] - 0.1) < 1e-12
    # delta_s(m=20) = mean([2,2]) - mean([1.5,1.7]) = 0.4
    assert abs(delta["delta_s"][1] - 0.4) < 1e-12

    # the audit's own stratified bootstrap consumes these molecule units
    boot = cd.ciw._stratified_bootstrap(delta["delta_cap"], delta["_fold_ids"], n_boot=200, seed=1)
    assert boot["n_molecules"] == 3
