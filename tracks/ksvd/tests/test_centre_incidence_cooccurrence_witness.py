"""Tests for the centre-incidence co-occurrence witness audit.

These pin the v4 export, the reconstructed centre path and the witness /
control contract used by
``experiments/luyin16/zinc_centre_incidence_cooccurrence_witness`` without
touching the real dataset or any frozen checkpoint:

1.  reconstructed per-centre 165D context equals the true forward context;
2.  each unordered pair contributes exactly once to each endpoint centre;
3.  the reconstructed centre update equals the forward post-update state;
4.  the reconstructed pre-head R equals the forward R;
5.  the reconstructed prediction equals the forward prediction;
6.  the covariance off-diagonal has the expected dimension / symmetry;
7.  covariance is zero for cells with a single incident relation;
8.  the centre witness summary is invariant to incident-pair row order;
9.  the deterministic random projections are identical across calls;
10. marginal / covariance graph summaries have matched dimensionality;
11. B1 / B2 / E parameter budgets are matched;
12. the covariance adapter genuinely depends on its witness input.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import (
    zinc_centre_incidence_cooccurrence_witness as wz,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _tiny_model() -> zpp.PatchPathModel:
    torch.manual_seed(0)
    return zpp.PatchPathModel(
        8,
        4,
        patch_hidden=wz.PATCH_DIM,
        pair_hidden=wz.PAIR_DIM,
        token_width=8,
        dropout=0.0,
        embedding_mode="full",
        center_context=True,
        center_context_hidden=12,
        graph_head_hidden_0=16,
        graph_head_hidden_1=8,
        shell_width=zpp._shell_width_for_radius(zpp.PATCH_RADIUS),
        context_width=0,
        topology_mode="hinge",
        topology_input_width=25,
        topology_hidden_dim=8,
        topology_out_dim=8,
    )


def _dummy_graph(n: int = 6, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pair_index = torch.tensor([[i for i, _ in pairs], [j for _, j in pairs]], dtype=torch.long)
    buckets = torch.tensor([min(max((j - i), 1), 5) - 1 for i, j in pairs], dtype=torch.long)
    return zpp.Data(
        patch_cont=torch.randn(n, zpp._shell_width_for_radius(zpp.PATCH_RADIUS), generator=generator),
        patch_context=torch.zeros(n, 0),
        typed_token=torch.randint(0, 8, (n,), generator=generator),
        parent_token=torch.randint(0, 4, (n,), generator=generator),
        structural_token=torch.zeros(n, dtype=torch.long),
        structural_coarse=torch.zeros(n, 4),
        pair_index=pair_index,
        pair_relation=torch.randn(len(pairs), zpp.RELATION_WIDTH, generator=generator),
        pair_bucket=buckets,
        global_context=torch.randn(1, zpp.GLOBAL_WIDTH, generator=generator),
        batch=torch.zeros(n, dtype=torch.long),
        y=torch.tensor([0.0]),
        num_nodes=n,
        topology_features=torch.randn(1, 25, generator=generator),
    )


def _capture(n: int = 6, seed: int = 0):
    model = _tiny_model()
    model.eval()
    graph = _dummy_graph(n=n, seed=seed)
    capture = wz._capture_states_v4(model, [graph], batch_size=1)["capture"]
    capture["n_patches"] = np.asarray(capture["n_patches"])
    capture["n_pairs"] = np.asarray(capture["n_pairs"])
    return model, graph, capture


# ---------------------------------------------------------------------------
# 1 -- reconstructed centre context equals the forward context
# ---------------------------------------------------------------------------

def test_reconstructed_centre_context_equals_forward():
    _model, _graph, capture = _capture(seed=1)
    rebuilt = wz._reconstruct_centre_context(capture)
    truth = np.asarray(capture["center_context"], dtype=np.float64)
    assert rebuilt.shape == truth.shape
    assert rebuilt.shape[1] == wz.CENTRE_CONTEXT_WIDTH == 165
    # model computes the context in float32; the offline reconstruction in
    # float64 differs only by float32 accumulation error (< gate atol).
    assert np.abs(rebuilt - truth).max() < wz.CONTEXT_RECON_ATOL


# ---------------------------------------------------------------------------
# 2 -- pair incidence: each pair contributes once to each endpoint
# ---------------------------------------------------------------------------

def test_pair_incidence_contributes_to_both_endpoints():
    _model, _graph, capture = _capture(n=6, seed=2)
    counts, totals = wz._incidence_count(capture)
    n_patches = int(capture["n_patches"][0])
    n_pairs = int(capture["n_pairs"][0])
    assert n_pairs == n_patches * (n_patches - 1) // 2
    assert int(totals[0].sum()) == 2 * n_pairs
    # each centre is incident to exactly n_patches-1 relations
    assert np.allclose(totals[0], n_patches - 1)
    src = np.asarray(capture["pair_source_local"])
    tgt = np.asarray(capture["pair_target_local"])
    assert np.all(src < tgt)


# ---------------------------------------------------------------------------
# 3 -- reconstructed centre update equals the forward post-update state
# ---------------------------------------------------------------------------

def test_reconstructed_centre_update_equals_forward():
    model, _graph, capture = _capture(seed=3)
    pre = torch.tensor(np.asarray(capture["patch_states_pre"], dtype=np.float32))
    context = torch.tensor(wz._reconstruct_centre_context(capture), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        rebuilt = (pre + model.center_update(torch.cat([pre, context], dim=1))).numpy()
    assert np.abs(rebuilt - np.asarray(capture["patch_states_post"])).max() < 1e-6


# ---------------------------------------------------------------------------
# 4 -- reconstructed R equals the forward R
# ---------------------------------------------------------------------------

def test_reconstructed_R_equals_forward():
    capture = _capture(seed=4)[2]
    rebuilt = wz._reconstruct_R(capture)
    assert rebuilt.shape[1] == wz.R_DIM
    assert np.abs(rebuilt.astype(np.float64) - np.asarray(capture["R"], dtype=np.float64)).max() < 1e-4


# ---------------------------------------------------------------------------
# 5 -- reconstructed prediction equals the forward prediction
# ---------------------------------------------------------------------------

def test_reconstructed_prediction_equals_forward():
    model, _graph, capture = _capture(seed=5)
    rebuilt = wz._reconstruct_R(capture)
    model.eval()
    with torch.no_grad():
        yhat = model.head(torch.tensor(rebuilt, dtype=torch.float32)).view(-1).numpy()
    assert np.abs(yhat - np.asarray(capture["yhat_0"])).max() < 1e-5


# ---------------------------------------------------------------------------
# 6 / 7 -- covariance shape and small-n behaviour
# ---------------------------------------------------------------------------

def test_covariance_offdiag_dimension_and_small_n_zero():
    # one centre, one pair (n=1) in bucket 0 -> zero covariance
    q = np.asarray([[0.0, 1.0] + [0.0] * (wz.PAIR_DIM - 2)])
    src = np.asarray([0])
    tgt = np.asarray([1])
    bucket = np.asarray([0])
    count = np.asarray([[1.0] + [0.0] * (wz.N_BUCKETS - 1),
                        [1.0] + [0.0] * (wz.N_BUCKETS - 1)])
    rows = wz._centre_covariance_rows(q, src, tgt, bucket, 2, count)
    assert rows.shape == (2 * wz.N_BUCKETS, wz.OFFDIAG_DIM) == (10, 120)
    assert np.abs(rows).max() == 0.0  # single relation -> zero covariance
    iu = np.triu_indices(wz.PAIR_DIM, k=1)
    assert (iu[0][0], iu[1][0]) == (0, 1)

    # centre 0 owns two bucket-0 relations with anti-correlated channels
    q2 = np.asarray([
        [1.0, 0.0] + [0.0] * (wz.PAIR_DIM - 2),
        [0.0, 1.0] + [0.0] * (wz.PAIR_DIM - 2),
    ])
    src2 = np.asarray([0, 0])
    tgt2 = np.asarray([1, 2])
    bucket2 = np.asarray([0, 0])
    count2 = np.asarray([[2.0] + [0.0] * (wz.N_BUCKETS - 1),
                         [1.0] + [0.0] * (wz.N_BUCKETS - 1),
                         [1.0] + [0.0] * (wz.N_BUCKETS - 1)])
    rows2 = wz._centre_covariance_rows(q2, src2, tgt2, bucket2, 3, count2)
    assert abs(rows2[0, 0]) > 0.1  # cov(ch0, ch1) nonzero for centre 0


# ---------------------------------------------------------------------------
# 8 -- incident-pair row order invariance
# ---------------------------------------------------------------------------

def test_witness_invariant_to_pair_row_order():
    capture = _capture(n=6, seed=8)[2]
    features = wz._per_centre_features(capture)
    cov_stat = (np.zeros(wz.OFFDIAG_DIM), np.ones(wz.OFFDIAG_DIM))
    marg_stat = (np.zeros(2 * wz.PAIR_DIM + 1), np.ones(2 * wz.PAIR_DIM + 1))
    n = int(capture["n_patches"][0])
    count = features["count"]
    cov = features["cov_rows"].reshape(n, wz.N_BUCKETS, wz.OFFDIAG_DIM)
    marg = features["marg_rows"].reshape(n, wz.N_BUCKETS, 2 * wz.PAIR_DIM + 1)
    base = wz._residual_witness_signature(
        count.reshape(-1), features["cov_rows"], features["marg_rows"], n, cov_stat, marg_stat
    )

    rng = np.random.default_rng(0)
    order = rng.permutation(n)
    permuted = wz._residual_witness_signature(
        count[order].reshape(-1),
        cov[order].reshape(-1, wz.OFFDIAG_DIM),
        marg[order].reshape(-1, 2 * wz.PAIR_DIM + 1),
        n,
        cov_stat,
        marg_stat,
    )
    assert base.shape == (2 * wz.WITNESS_GRAPH_DIM,) == (160,)
    assert np.abs(base - permuted).max() < 1e-9


# ---------------------------------------------------------------------------
# 9 -- deterministic projections
# ---------------------------------------------------------------------------

def test_random_projections_deterministic():
    cov_a = wz._cov_projection()
    cov_b = wz._cov_projection()
    marg_a = wz._marg_projection()
    marg_b = wz._marg_projection()
    assert np.array_equal(cov_a, cov_b)
    assert np.array_equal(marg_a, marg_b)
    assert cov_a.shape == (wz.OFFDIAG_DIM, wz.COV_PROJ_DIM) == (120, 8)
    assert marg_a.shape == (2 * wz.PAIR_DIM + 1, wz.MARG_PROJ_DIM) == (33, 8)
    assert np.abs(cov_a.T @ cov_a - np.eye(wz.COV_PROJ_DIM)).max() < 1e-5
    assert np.abs(marg_a.T @ marg_a - np.eye(wz.MARG_PROJ_DIM)).max() < 1e-5
    fingerprint = wz._projection_fingerprint()
    assert fingerprint["covariance"]["sha256"] == wz._sha256_array(cov_a)
    assert fingerprint["marginal"]["sha256"] == wz._sha256_array(marg_a)


# ---------------------------------------------------------------------------
# 10 -- matched graph dimensionality
# ---------------------------------------------------------------------------

def test_graph_summary_dimensions_match():
    assert wz.WITNESS_GRAPH_DIM == wz.MARGINAL_GRAPH_DIM == 80
    assert wz.CENTRE_CONTEXT_WIDTH == 165
    assert wz.OFFDIAG_DIM == 120


# ---------------------------------------------------------------------------
# 11 -- parameter budgets
# ---------------------------------------------------------------------------

def test_parameter_budgets_matched():
    b1 = wz._make_head(0, "ronly")
    b2 = wz._make_head(0, "marginal")
    e = wz._make_head(0, "covariance")
    p1, p2, pe = wz._count_parameters(b1), wz._count_parameters(b2), wz._count_parameters(e)
    assert p2 == pe  # B2 and E share architecture exactly
    assert b1.net[0].in_features == wz.R_DIM
    assert e.net[0].in_features == wz.R_DIM + wz.WITNESS_GRAPH_DIM == 382
    assert abs(pe - p1) / p1 <= 0.02  # within the pre-registered +-2% budget


# ---------------------------------------------------------------------------
# 12 -- covariance adapter depends on the witness
# ---------------------------------------------------------------------------

def test_covariance_adapter_depends_on_witness():
    torch.manual_seed(0)
    model = wz._make_head(0, "covariance")
    for param in model.parameters():
        with torch.no_grad():
            param.add_(torch.randn_like(param) * 0.05)
    x = torch.randn(32, wz.R_DIM + wz.WITNESS_GRAPH_DIM)
    yhat0 = torch.zeros(32)
    with torch.no_grad():
        with_witness = model(x, yhat0, use_witness=True)
        without = model(x, yhat0, use_witness=False)
    assert float((with_witness - without).abs().max()) > 1e-3


# ---------------------------------------------------------------------------
# extra: legacy cache rejected
# ---------------------------------------------------------------------------

def test_legacy_cache_rejected(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez_compressed(
        path,
        fingerprint_json=np.asarray(
            '{"export_version": "frozen_state_export_v3_pair_endpoint"}'
        ),
    )
    with pytest.raises(RuntimeError):
        wz.load_export(path)
