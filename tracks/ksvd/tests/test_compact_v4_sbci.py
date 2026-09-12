"""Static / unit tests for the compact-v4-SBCI experiment.

These tests never train, never load the official valid split for a decision and
never load official test.  They exercise the pre-registered architecture
properties (parameter cap, removed pathways, shared initialisation, exact
factorised pair interaction, bucket semantics, permutation invariance,
branch gradients) on synthetic batches.
"""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from tracks.ksvd.code.graph import from_edges, ring_chords
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_sbci as sb
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ztopo


def _synthetic_graph() -> Data:
    graph = ring_chords(6, [])
    n = len(graph.nodes)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pair_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    buckets = torch.tensor([min(i, 4) for i in range(len(pairs))], dtype=torch.long)
    return Data(
        patch_cont=torch.randn(n, int(zpp.SHELL_WIDTH)),
        patch_context=torch.zeros(n, 0),
        typed_token=torch.zeros(n, dtype=torch.long),
        parent_token=torch.zeros(n, dtype=torch.long),
        structural_token=torch.zeros(n, dtype=torch.long),
        structural_coarse=torch.zeros(n, 0),
        pair_index=pair_index,
        pair_relation=torch.randn(len(pairs), int(zpp.RELATION_WIDTH)),
        pair_bucket=buckets,
        global_context=torch.zeros(1, int(zpp.GLOBAL_WIDTH)),
        topology_features=torch.zeros(1, int(ztopo.raw_width("hinge"))),
        y=torch.zeros(1),
        num_nodes=n,
    )


def _synthetic_batch():
    graphs = [_synthetic_graph() for _ in range(3)]
    return next(iter(zpp._make_loader(graphs, 128, False, 0)))


@pytest.fixture(scope="module")
def sbci():
    torch.manual_seed(0)
    model = sb.build_sbci(0)
    model.eval()
    return model


def test_01_baseline_and_candidate_params():
    baseline = sb.build_baseline(0)
    candidate = sb.build_sbci(0)
    assert sb._n_params(baseline) == 99613
    assert sb._n_params(candidate) == 62045
    assert sb._n_params(candidate) <= sb.BASELINE_TOTAL == 82115


def test_02_parameter_ledger_mechanical_account():
    model = sb.build_sbci(0)
    blocks = {
        "identity_token_storage": sb._n_params(model.typed_embedding) + sb._n_params(model.parent_embedding),
        "patch_encoder": sb._n_params(model.patch_encoder),
        "basis_encoder_phi": sb._n_params(model.basis_encoder),
        "relation_modulator_psi": sb._n_params(model.relation_modulator),
        "centre_composer": sb._n_params(model.centre_composer),
        "global_branch": sb._n_params(model.global_encoder),
        "topology_branch": sb._n_params(model.topology_encoder),
        "final_head": sb._n_params(model.head),
    }
    assert sum(blocks.values()) == sb._n_params(model) == 62045
    assert blocks["basis_encoder_phi"] == 2096
    assert blocks["relation_modulator_psi"] == 1296
    assert blocks["centre_composer"] == 2656
    assert blocks["final_head"] == 1441


def test_03_removed_pathways_absent(sbci):
    for name in ("pair_projection", "relation_encoder", "distance_gate", "pair_encoder", "center_update"):
        assert not hasattr(sbci, name)
    # the head is the SBCI head, not the 302D small head
    assert int(sbci.head[0].in_features) == 88
    assert int(sbci.head[2].out_features) == 1


def test_04_phi_psi_architecture(sbci):
    assert int(sbci.basis_encoder[0].in_features) == 48
    assert int(sbci.basis_encoder[0].out_features) == 32
    assert int(sbci.basis_encoder[2].out_features) == 16
    assert int(sbci.relation_modulator[0].in_features) == sb.RELATION_WIDTH == 23
    assert int(sbci.relation_modulator[2].out_features) == 16
    assert int(sbci.centre_composer.in_features) == 165
    assert int(sbci.centre_composer.out_features) == 16


def test_05_shared_init_exact_both_seeds():
    for seed in (0, 1):
        baseline = sb.build_baseline(seed)
        model = sb.build_sbci(seed)
        base_state = {k: v.detach().clone() for k, v in baseline.state_dict().items()}
        state = model.state_dict()
        keys = sb._shared_keys(base_state, model)
        assert len(keys) == 20
        for key in keys:
            assert torch.equal(state[key], base_state[key]), key


def test_06_pair_factorization_exact(sbci):
    batch = _synthetic_batch()
    with torch.no_grad():
        e = sbci.typed_embedding(batch.typed_token)
        h = sbci.patch_encoder(
            torch.cat([batch.patch_cont, batch.patch_context, e, sbci.parent_embedding(batch.parent_token)], dim=1)
        )
        z, pair = sbci.pair_states(batch, h)
        gate = 1.0 + torch.tanh(sbci.relation_modulator(batch.pair_relation))
        manual = z[batch.pair_index[0]] * z[batch.pair_index[1]] * gate
    assert torch.equal(pair, manual)
    assert bool((gate > 0.0).all() and (gate < 2.0).all())


def test_07_endpoint_symmetry(sbci):
    batch = _synthetic_batch()
    with torch.no_grad():
        e = sbci.typed_embedding(batch.typed_token)
        h = sbci.patch_encoder(
            torch.cat([batch.patch_cont, batch.patch_context, e, sbci.parent_embedding(batch.parent_token)], dim=1)
        )
        z, pair = sbci.pair_states(batch, h)
        gate = 1.0 + torch.tanh(sbci.relation_modulator(batch.pair_relation))
        swapped = z[batch.pair_index[1]] * z[batch.pair_index[0]] * gate
    assert torch.equal(pair, swapped)


def test_08_bucket_width_and_empty_semantics(sbci):
    batch = _synthetic_batch()
    with torch.no_grad():
        ctx = sbci._pool_pairs_to_centres(
            torch.randn(batch.pair_index.shape[1], 16),
            batch.pair_index[0],
            batch.pair_index[1],
            batch.pair_bucket,
            int(batch.num_nodes),
        )
    assert int(ctx.shape[1]) == 165
    empty = sbci._pool_pairs_to_centres(
        torch.zeros((0, 16)), torch.zeros(0, dtype=torch.long),
        torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long), 3,
    )
    assert torch.equal(empty, torch.zeros_like(empty))
    assert bool(torch.isfinite(empty).all())


def test_09_R_dimension(sbci):
    batch = _synthetic_batch()
    with torch.no_grad():
        R = sbci.encode(batch)
    assert int(R.shape[1]) == 88 == sb.FINAL_R_WIDTH


def test_10_permutation_invariance(sbci):
    graph = _synthetic_graph()
    graph.batch = torch.zeros(int(graph.num_nodes), dtype=torch.long)
    perm = torch.randperm(int(graph.num_nodes), generator=torch.Generator().manual_seed(1))
    permuted = sb._permute_single_graph(graph, perm)
    with torch.no_grad():
        y0 = sbci(graph).item()
        y1 = sbci(permuted).item()
    assert abs(y0 - y1) < 1e-4


def test_11_branch_gradients_nonzero():
    torch.manual_seed(0)
    model = sb.build_sbci(0)
    model.train()
    batch = _synthetic_batch()
    loss = torch.nn.functional.l1_loss(model(batch).view(-1), batch.y.view(-1))
    model.zero_grad()
    loss.backward()
    for module in (model.basis_encoder, model.relation_modulator, model.centre_composer, model.head):
        total = sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)
        assert total > 0.0


def test_12_pre_registered_thresholds():
    assert sb.SBCI_NOGO == 0.008
    assert sb.SBCI_STRONG == 0.015
    assert sb.K_BASIS == 16
    assert sb.BOOTSTRAP_B == 2000
