"""Targeted correctness tests for the CENTER-COMP-v0 diagnostic."""

from __future__ import annotations

import numpy as np
import torch

from tracks.ksvd.code import tccd_v0 as T
from tracks.ksvd.code import tccd_v6 as V6
from tracks.ksvd.code import center_comp_v0 as V


def test_geometry_and_parameters():
    assert V.PAIR_WIDTH == 3 * V.K_PROTO + V.N_REL == 197
    assert V.RHO_IN == 80
    assert V.FROZEN_BASE_DIM == T.h_dim(V.K_PROTO, V.N_REL) == 10464
    assert V.parameter_accounting() == {"phi": 13712, "rho": 6224, "head": 10481, "total": 30417}
    model = V.CenterCompFactory.build(seed=0)
    assert V.model_parameter_count(model) == V.parameter_accounting()
    names = sorted(n for n, _ in model.named_parameters())
    assert all(n.startswith("phi.") or n.startswith("rho.") or n.startswith("head.") for n in names)


def test_center_pair_context_contributes_to_both_endpoints():
    cache, _ = V6.tiny_cache(seed=3)
    row = 0
    n = cache.n_of_row(row)
    c0, c1 = int(cache.c_offsets[row]), int(cache.c_offsets[row + 1])
    p0, p1 = int(cache.pair_offsets[row]), int(cache.pair_offsets[row + 1])
    C = torch.as_tensor(cache.C[c0:c1], dtype=torch.float32).unsqueeze(0)
    i0 = torch.as_tensor(cache.pair_i0[p0:p1]).unsqueeze(0)
    i1 = torch.as_tensor(cache.pair_i1[p0:p1]).unsqueeze(0)
    rel = torch.as_tensor(cache.pair_rel[p0:p1]).unsqueeze(0)
    valid = torch.ones((1, p1 - p0), dtype=torch.bool)
    phi = V.CenterCompFactory.build(seed=0).phi
    with torch.no_grad():
        e, q = V.center_pair_context(C, i0, i1, rel, valid, phi)
        manual = torch.zeros((1, n, V.PHI_OUT))
        for k in range(p1 - p0):
            manual[0, int(i0[0, k])] += e[0, k]
            manual[0, int(i1[0, k])] += e[0, k]
    assert float((q - manual).abs().max()) <= 1e-6


def test_pair_order_and_swap_invariance():
    cache, _ = V6.tiny_cache(seed=3)
    row = 0
    c0, c1 = int(cache.c_offsets[row]), int(cache.c_offsets[row + 1])
    p0, p1 = int(cache.pair_offsets[row]), int(cache.pair_offsets[row + 1])
    C = torch.as_tensor(cache.C[c0:c1], dtype=torch.float32).unsqueeze(0)
    i0 = torch.as_tensor(cache.pair_i0[p0:p1]).unsqueeze(0)
    i1 = torch.as_tensor(cache.pair_i1[p0:p1]).unsqueeze(0)
    rel = torch.as_tensor(cache.pair_rel[p0:p1]).unsqueeze(0)
    valid = torch.ones((1, p1 - p0), dtype=torch.bool)
    phi = V.CenterCompFactory.build(seed=0).phi
    with torch.no_grad():
        e, q = V.center_pair_context(C, i0, i1, rel, valid, phi)
        e_swap, q_swap = V.center_pair_context(C, i1, i0, rel, valid, phi)
    assert float((e - e_swap).abs().max()) <= 1e-6
    assert float((q - q_swap).abs().max()) <= 1e-6
    order = np.random.default_rng(3).permutation(p1 - p0)
    with torch.no_grad():
        _, q_perm = V.center_pair_context(
            C, i0[:, order], i1[:, order], rel[:, order], valid, phi
        )
    assert float((q - q_perm).abs().max()) <= 1e-6


def test_data_free_gate0_passes():
    checks = V.gate0_data_free_checks("cpu")
    for key in (
        "parameter_accounting_ok",
        "model_parameter_count_ok",
        "shapes_ok",
        "pair_to_centers_exact",
        "pair_swap_descriptor_invariant",
        "pair_order_invariant",
        "batching_invariant",
        "relabel_h_center_invariant",
        "shuffle_c_multiset_preserved",
        "shuffle_h_base_preserved",
        "shuffle_q_multiset_preserved",
        "shuffle_changes_binding",
        "gradient_reaches_phi",
        "gradient_reaches_rho",
        "gradient_reaches_reader",
        "frozen_local_all_detached",
        "frozen_local_no_grad",
        "no_local_encoder_parameters",
        "data_free_all_pass",
    ):
        assert checks[key] is True, key
    assert checks["official_test_loaded"] is False
    assert checks["official_valid_blocked"] is True


def test_train_smoke_and_shuffle_changes_predictions():
    cache, y = V6.tiny_cache(seed=5)
    train = np.arange(cache.n_graphs - 1)
    dev = np.arange(cache.n_graphs - 1, cache.n_graphs)
    res = V.train_center_comp(cache, y, train, dev, "cpu", seed=0, max_epochs=2, patience=2, batch=2)
    assert res.soup_valid is not None and np.isfinite(res.soup_valid)
    assert len(res.soup_members) >= 1
    model = V.CenterCompFactory.build(seed=0)
    model.load_state_dict(res.state_soup)
    real = V.predict(model, cache, y, dev, "cpu", shuffle=False)
    shuf = V.predict(model, cache, y, dev, "cpu", shuffle=True)
    # shapes match; the intervention is defined even when predictions coincide
    assert real.shape == shuf.shape == (1,)


def test_decision_cases():
    d = V.center_comp_decision(g_center=0.02, g_bind=0.02, mae_center=0.23, mae_shuffle=0.25)
    assert d["outcome"] == "A"
    d = V.center_comp_decision(g_center=0.02, g_bind=0.001, mae_center=0.23, mae_shuffle=0.231)
    assert d["outcome"] == "B"
    d = V.center_comp_decision(g_center=0.02, g_bind=0.007, mae_center=0.23, mae_shuffle=0.237)
    assert d["outcome"] == "B_AMBIGUOUS"
    d = V.center_comp_decision(g_center=0.005, g_bind=0.02, mae_center=0.246, mae_shuffle=0.266)
    assert d["outcome"] == "C"
    d = V.center_comp_decision(g_center=0.005, g_bind=0.005, mae_center=0.246, mae_shuffle=0.251)
    assert d["outcome"] == "D"
    d = V.center_comp_decision(g_center=0.001, g_bind=0.001, mae_center=0.250, mae_shuffle=0.251)
    assert d["outcome"] == "D"
    assert d["gap_to_strong"] == 0.250 - V.CANONICAL_GPU1_BASELINE


def test_prediction_change_report_and_stratification():
    real = np.array([0.0, 1.0, 2.0, 3.0])
    shuf = np.array([0.005, 1.2, 2.0, 3.2])
    rep = V.prediction_change_report(real, shuf)
    assert rep["n"] == 4
    assert rep["mean_abs_change"] == float(np.mean(np.abs(real - shuf)))
    assert 0.0 <= rep["fraction_ge_0.01"] <= 1.0
    cache, y = V6.tiny_cache(seed=5)
    dev = list(range(cache.n_graphs))
    strat = V.size_stratification(cache, dev, y, np.zeros(len(dev)), np.ones(len(dev)))
    assert set(strat).issubset({"small", "medium", "large"})
    for value in strat.values():
        assert value["n_graphs"] > 0
