"""Tests for the pair-endpoint-association witness audit.

These pin the v3 export, the reconstructed pair path and the witness / control
contract used by ``experiments/luyin16/zinc_pair_endpoint_association_witness``
without touching the real dataset or any frozen checkpoint:

1.  exported ``u_i`` equals the true forward pair-projection state;
2.  the reconstructed pair encoder input equals the forward pair input;
3.  the reconstructed ``q_ij`` equals the original ``q_ij``;
4.  the off-diagonal association is invariant under endpoint swap;
5.  the known algebraic collision has identical current c_ij but different a_ij;
6.  node permutation preserves the graph-level witness summary;
7.  the deterministic random projection is identical across calls;
8.  B1 / B2 / E parameter budgets are matched;
9.  a legacy v2 cache cannot masquerade as a v3 export;
10. the endpoint adapter genuinely depends on its witness input.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import zinc_pair_endpoint_association_witness as wz


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
    return zpp.Data(
        patch_cont=torch.randn(n, zpp._shell_width_for_radius(zpp.PATCH_RADIUS), generator=generator),
        patch_context=torch.zeros(n, 0),
        typed_token=torch.randint(0, 8, (n,), generator=generator),
        parent_token=torch.randint(0, 4, (n,), generator=generator),
        structural_token=torch.zeros(n, dtype=torch.long),
        structural_coarse=torch.zeros(n, 4),
        pair_index=pair_index,
        pair_relation=torch.randn(len(pairs), zpp.RELATION_WIDTH, generator=generator),
        pair_bucket=torch.randint(0, wz.N_BUCKETS, (len(pairs),), generator=generator),
        global_context=torch.randn(1, zpp.GLOBAL_WIDTH, generator=generator),
        batch=torch.zeros(n, dtype=torch.long),
        y=torch.tensor([0.0]),
        num_nodes=n,
        topology_features=torch.randn(1, 25, generator=generator),
    )


# ---------------------------------------------------------------------------
# Test 1 / 2 / 3 -- export path correctness
# ---------------------------------------------------------------------------

def test_exported_u_equals_forward_pair_projection():
    model = _tiny_model()
    model.eval()
    graph = _dummy_graph(n=6, seed=3)
    result = wz._capture_states_v3(model, [graph], batch_size=1)
    capture = result["capture"]
    assert result["projected_max_diff"] < wz.PAIR_RECON_ATOL

    projected = []
    handle = model.pair_projection.register_forward_hook(
        lambda _m, _i, o: projected.append(o.detach().numpy())
    )
    with torch.no_grad():
        model(graph)
    handle.remove()
    src = capture["pair_source"]
    tgt = capture["pair_target"]
    assert np.abs(capture["projected_patch_state"][src] - projected[0]).max() < 1e-6
    assert np.abs(capture["projected_patch_state"][tgt] - projected[1]).max() < 1e-6


def test_reconstructed_pair_input_matches_forward():
    model = _tiny_model()
    model.eval()
    graph = _dummy_graph(n=6, seed=4)
    capture = wz._capture_states_v3(model, [graph], batch_size=1)["capture"]

    true_input = []
    handle = model.pair_encoder.register_forward_pre_hook(
        lambda _m, args: true_input.append(args[0].detach().numpy())
    )
    with torch.no_grad():
        model(graph)
    handle.remove()

    with torch.no_grad():
        rebuilt = wz._reconstruct_pair_input(
            capture["projected_patch_state"],
            capture["pair_source"],
            capture["pair_target"],
            torch.tensor(capture["pair_relation"]),
            torch.tensor(capture["pair_bucket"]),
            model,
        ).numpy()
    assert rebuilt.shape == (capture["pair_states"].shape[0], 4 * wz.PAIR_DIM)
    assert np.abs(rebuilt - true_input[0]).max() < 1e-6


def test_reconstructed_q_equals_original():
    model = _tiny_model()
    model.eval()
    graph = _dummy_graph(n=5, seed=5)
    capture = wz._capture_states_v3(model, [graph], batch_size=1)["capture"]
    pair_input = wz._reconstruct_pair_input(
        capture["projected_patch_state"],
        capture["pair_source"],
        capture["pair_target"],
        torch.tensor(capture["pair_relation"]),
        torch.tensor(capture["pair_bucket"]),
        model,
    )
    with torch.no_grad():
        q = model.pair_encoder(pair_input).numpy()
    assert np.abs(q - capture["pair_states"]).max() < 1e-6


# ---------------------------------------------------------------------------
# Test 4 / 5 -- algebra
# ---------------------------------------------------------------------------

def test_offdiag_invariant_under_endpoint_swap():
    rng = np.random.default_rng(0)
    left = rng.standard_normal((7, wz.PAIR_DIM))
    right = rng.standard_normal((7, wz.PAIR_DIM))
    assert np.allclose(wz._offdiag_outer(left, right), wz._offdiag_outer(right, left))
    assert wz._offdiag_outer(left, right).shape == (7, wz.OFFDIAG_DIM)
    assert wz.OFFDIAG_DIM == wz.PAIR_DIM * (wz.PAIR_DIM - 1) // 2


def test_algebraic_collision_same_c_different_a():
    # u_i=(0,0), u_j=(1,1)  vs  u'_i=(0,1), u'_j=(1,0) on the first two coords.
    left = np.zeros((1, wz.PAIR_DIM))
    right = np.zeros((1, wz.PAIR_DIM))
    right[0, 0] = 1.0
    right[0, 1] = 1.0
    left2 = np.zeros((1, wz.PAIR_DIM))
    right2 = np.zeros((1, wz.PAIR_DIM))
    left2[0, 1] = 1.0
    right2[0, 0] = 1.0

    def current_feature(a, b):
        return np.concatenate([a + b, np.abs(a - b), a * b], axis=1)

    assert np.allclose(current_feature(left, right), current_feature(left2, right2))
    a1 = wz._offdiag_outer(left, right)
    a2 = wz._offdiag_outer(left2, right2)
    assert np.abs(a1 - a2).max() > 0.5
    # the discarded off-diagonal coordinate (0,1) is the first triu(k=1) entry
    iu = np.triu_indices(wz.PAIR_DIM, k=1)
    assert (iu[0][0], iu[1][0]) == (0, 1)
    assert a2[0, 0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test 6 -- node permutation invariance of the graph witness summary
# ---------------------------------------------------------------------------

def _permute_graph(graph, perm):
    n = int(graph.num_nodes)
    inv = np.argsort(perm)
    patch_cont = graph.patch_cont[torch.tensor(inv, dtype=torch.long)]
    typed = graph.typed_token[torch.tensor(inv, dtype=torch.long)]
    parent = graph.parent_token[torch.tensor(inv, dtype=torch.long)]
    pairs = graph.pair_index.numpy()
    new_pairs = np.empty_like(pairs)
    for col in range(pairs.shape[1]):
        a, b = int(pairs[0, col]), int(pairs[1, col])
        na, nb = int(perm[a]), int(perm[b])
        new_pairs[0, col], new_pairs[1, col] = (na, nb) if na < nb else (nb, na)
    return zpp.Data(
        patch_cont=patch_cont,
        patch_context=graph.patch_context,
        typed_token=typed,
        parent_token=parent,
        structural_token=graph.structural_token,
        structural_coarse=graph.structural_coarse,
        pair_index=torch.tensor(new_pairs, dtype=torch.long),
        pair_relation=graph.pair_relation,
        pair_bucket=graph.pair_bucket,
        global_context=graph.global_context,
        topology_features=graph.topology_features,
        y=graph.y,
        num_nodes=n,
    )


def test_node_permutation_preserves_witness_summary():
    model = _tiny_model()
    model.eval()
    graph = _dummy_graph(n=6, seed=7)
    perm = np.asarray([3, 1, 5, 0, 2, 4], dtype=np.int64)
    permuted = _permute_graph(graph, perm)
    cap_a = wz._capture_states_v3(model, [graph], batch_size=1)["capture"]
    cap_b = wz._capture_states_v3(model, [permuted], batch_size=1)["capture"]
    projection = wz._random_projection()
    stat_a = (np.zeros(wz.OFFDIAG_DIM), np.ones(wz.OFFDIAG_DIM))
    stat_d = (np.zeros(wz.PAIR_DIM), np.ones(wz.PAIR_DIM))
    wa, da = wz._witness_from_capture(cap_a, projection, stat_a, stat_d)
    wb, db = wz._witness_from_capture(cap_b, projection, stat_a, stat_d)
    assert np.abs(wa - wb).max() < 1e-5
    assert np.abs(da - db).max() < 1e-5


# ---------------------------------------------------------------------------
# Test 7 -- deterministic projection
# ---------------------------------------------------------------------------

def test_random_projection_deterministic():
    a = wz._random_projection()
    b = wz._random_projection()
    assert np.array_equal(a, b)
    assert a.shape == (wz.OFFDIAG_DIM, wz.PROJ_DIM)
    # orthonormal columns
    gram = a.T @ a
    assert np.abs(gram - np.eye(wz.PROJ_DIM)).max() < 1e-5
    fingerprint = wz._projection_fingerprint()
    assert fingerprint["sha256"] == wz._sha256_array(a)


# ---------------------------------------------------------------------------
# Test 8 -- parameter budgets
# ---------------------------------------------------------------------------

def test_parameter_budgets_matched():
    b1 = wz._make_head(0, "ronly")
    b2 = wz._make_head(0, "diag")
    e = wz._make_head(0, "endpoint")
    p1, p2, pe = wz._count_parameters(b1), wz._count_parameters(b2), wz._count_parameters(e)
    assert p2 == pe  # B2 and E share architecture exactly
    assert b1.net[0].in_features == wz.R_DIM
    assert e.net[0].in_features == wz.R_DIM + wz.DIAG_GRAPH_DIM
    assert abs(pe - p1) / p1 <= 0.02  # within the pre-registered +-2% budget


# ---------------------------------------------------------------------------
# Test 9 -- legacy cache rejected
# ---------------------------------------------------------------------------

def test_legacy_cache_rejected(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez_compressed(
        path,
        fingerprint_json=np.asarray(
            '{"export_version": "frozen_state_export_v2_corrected"}'
        ),
    )
    with pytest.raises(RuntimeError):
        wz.load_export(path)


# ---------------------------------------------------------------------------
# Test 10 -- endpoint adapter depends on the witness
# ---------------------------------------------------------------------------

def test_endpoint_adapter_depends_on_witness():
    torch.manual_seed(0)
    model = wz._make_head(0, "endpoint")
    # make the input-side weights demonstrably non-trivial
    for param in model.parameters():
        with torch.no_grad():
            param.add_(torch.randn_like(param) * 0.05)
    x = torch.randn(32, wz.R_DIM + wz.DIAG_GRAPH_DIM)
    yhat0 = torch.zeros(32)
    with torch.no_grad():
        with_witness = model(x, yhat0, use_witness=True)
        without = model(x, yhat0, use_witness=False)
    assert float((with_witness - without).abs().max()) > 1e-3
