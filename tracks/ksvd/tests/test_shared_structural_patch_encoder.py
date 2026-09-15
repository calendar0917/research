"""Tests for the shared connectivity-aware structural patch encoder.

All tests are fast unit / static tests.  They never load official test data and
never train a full model.  Requirements covered (numbered as in the experiment
brief):

1. node relabeling preserves ``e_struct``
2. changing an atom type changes the representation
3. changing a bond type changes the representation
4. changing the root position changes the rooted representation
5. the same rooted typed graph under permutation gives the same representation
6. an unseen certificate needs no vocabulary row
7. the model contains no vocab-sized ``typed_embedding.weight``
8. the parent path is preserved
9. q=16, h=64, T=2
10. recurrent weight tying is preserved
11. forward / backward are finite
12. total trainable params are in the 80k-90k budget
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_pair_centre_capacity_decomposition as cd,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_shared_structural_patch_encoder as sspe,
)
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16.structural_patch_encoder import (
    SharedStructuralPatchEncoder,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _path_data() -> Data:
    """A small typed path 0-1-2-3 with distinct atom and bond types."""
    edges = [(0, 1), (1, 2), (2, 3)]
    bond_types = [0, 1, 2]
    bi = edges + [(v, u) for u, v in edges]
    edge_attr = bond_types + bond_types
    return Data(
        edge_index=torch.tensor(bi, dtype=torch.long).t().contiguous(),
        x=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.long),
        y=torch.tensor([0.0]),
        num_nodes=4,
    )


class _Batch:
    def __init__(self, atom, root, dist, patch, src, dst, bond) -> None:
        self.struct_atom = atom
        self.struct_root = root
        self.struct_dist = dist
        self.struct_patch = patch
        self.struct_src = src
        self.struct_dst = dst
        self.struct_bond = bond


# ---------------------------------------------------------------------------
# 1-5: encoder invariance / sensitivity
# ---------------------------------------------------------------------------


def test_node_relabel_is_invariant():
    torch.manual_seed(0)
    encoder = SharedStructuralPatchEncoder(output_dim=16).eval()
    atom = torch.tensor([0, 1, 2, 3])
    root = torch.tensor([1, 0, 0, 0])
    dist = torch.tensor([0, 1, 2, 1])
    patch = torch.tensor([0, 0, 0, 0])
    src = torch.tensor([0, 1, 0, 2, 1, 3])
    dst = torch.tensor([1, 0, 2, 0, 3, 1])
    bond = torch.tensor([0, 0, 1, 1, 2, 2])
    base = _Batch(atom, root, dist, patch, src, dst, bond)
    with torch.no_grad():
        e = encoder(base)
    perm = torch.tensor([0, 2, 3, 1])
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(perm.numel())
    permuted = _Batch(
        atom[perm], root[perm], dist[perm], patch[perm],
        inverse[src], inverse[dst], bond,
    )
    with torch.no_grad():
        e2 = encoder(permuted)
    assert float((e - e2).abs().max()) < 1.0e-5


def test_edge_order_is_irrelevant():
    torch.manual_seed(1)
    encoder = SharedStructuralPatchEncoder(output_dim=16).eval()
    atom = torch.tensor([0, 1, 2])
    root = torch.tensor([1, 0, 0])
    dist = torch.tensor([0, 1, 2])
    patch = torch.tensor([0, 0, 0])
    # both directions of each undirected bond
    src = torch.tensor([0, 1, 1, 2])
    dst = torch.tensor([1, 0, 2, 1])
    bond = torch.tensor([0, 0, 1, 1])
    with torch.no_grad():
        e = encoder(_Batch(atom, root, dist, patch, src, dst, bond))
    order = torch.tensor([3, 0, 2, 1])
    with torch.no_grad():
        e_perm = encoder(
            _Batch(atom, root, dist, patch, src[order], dst[order], bond[order])
        )
    assert float((e - e_perm).abs().max()) < 1.0e-6


def test_atom_bond_root_changes_matter():
    torch.manual_seed(2)
    encoder = SharedStructuralPatchEncoder(output_dim=16).eval()
    atom = torch.tensor([0, 1, 2])
    root = torch.tensor([1, 0, 0])
    dist = torch.tensor([0, 1, 2])
    patch = torch.tensor([0, 0, 0])
    src = torch.tensor([0, 1])
    dst = torch.tensor([1, 2])
    bond = torch.tensor([0, 1])
    with torch.no_grad():
        e = encoder(_Batch(atom, root, dist, patch, src, dst, bond))
        atom2 = atom.clone()
        atom2[1] = 7
        e_atom = encoder(_Batch(atom2, root, dist, patch, src, dst, bond))
        bond2 = bond.clone()
        bond2[0] = 3
        e_bond = encoder(_Batch(atom, root, dist, patch, src, dst, bond2))
        root2 = torch.tensor([0, 1, 0])
        dist2 = torch.tensor([1, 0, 1])
        e_root = encoder(_Batch(atom, root2, dist2, patch, src, dst, bond))
    assert float((e - e_atom).abs().max()) > 1.0e-6
    assert float((e - e_bond).abs().max()) > 1.0e-6
    assert float((e - e_root).abs().max()) > 1.0e-6


def test_output_width_matches_token_width():
    encoder = SharedStructuralPatchEncoder(output_dim=16)
    atom = torch.tensor([0])
    root = torch.tensor([1])
    dist = torch.tensor([0])
    patch = torch.tensor([0])
    src = torch.zeros(0, dtype=torch.long)
    dst = torch.zeros(0, dtype=torch.long)
    bond = torch.zeros(0, dtype=torch.long)
    with torch.no_grad():
        out = encoder(_Batch(atom, root, dist, patch, src, dst, bond))
    assert out.shape == (1, 16)


# ---------------------------------------------------------------------------
# patch-graph reconstruction from the canonical pipeline
# ---------------------------------------------------------------------------


def test_patch_graph_reconstruction():
    graphs = sspe._patch_graphs_from_dataset([_path_data()])
    assert len(graphs) == 1
    graph = graphs[0]
    assert graph.n_patches == 4
    # patch 0 (centre node 0): nodes 0,1,2 at distances 0,1,2; atoms 0,1,2
    nodes0 = graph.patch == 0
    assert sorted(graph.atom[nodes0].tolist()) == [0, 1, 2]
    assert sorted(graph.dist[nodes0].tolist()) == [0, 1, 2]
    assert int(graph.root[nodes0].sum()) == 1
    patch0_edges = graph.edge_patch == 0
    assert sorted(graph.bond[patch0_edges].tolist()) == [0, 0, 1, 1]
    # every patch has exactly one root and no self loops
    for patch_id in range(graph.n_patches):
        assert int(graph.root[graph.patch == patch_id].sum()) == 1
    assert bool((graph.src != graph.dst).all())


def test_patch_graph_plain_cache_roundtrip():
    graphs = sspe._patch_graphs_from_dataset([_path_data()])
    plain = sspe._graphs_to_plain(graphs)
    restored = sspe._graphs_from_plain(plain)
    assert len(restored) == 1
    assert np.array_equal(restored[0].atom, graphs[0].atom)
    assert np.array_equal(restored[0].src, graphs[0].src)
    assert restored[0].n_patches == graphs[0].n_patches


def test_struct_collate_offsets():
    data_a = Data(num_nodes=2, y=torch.tensor([0.0]))
    data_b = Data(num_nodes=3, y=torch.tensor([0.0]))
    # graph-local structural tensors
    data_a.struct_atom = torch.tensor([0, 1, 2])
    data_a.struct_root = torch.tensor([1, 0, 0])
    data_a.struct_dist = torch.tensor([0, 1, 1])
    data_a.struct_patch = torch.tensor([0, 0, 1])
    data_a.struct_src = torch.tensor([0, 1])
    data_a.struct_dst = torch.tensor([1, 0])
    data_a.struct_bond = torch.tensor([0, 0])
    data_b.struct_atom = torch.tensor([3, 4])
    data_b.struct_root = torch.tensor([1, 0])
    data_b.struct_dist = torch.tensor([0, 1])
    data_b.struct_patch = torch.tensor([0, 0])
    data_b.struct_src = torch.tensor([0])
    data_b.struct_dst = torch.tensor([1])
    data_b.struct_bond = torch.tensor([1])
    # minimal required fields for PyG Batch
    for data in (data_a, data_b):
        data.patch_cont = torch.zeros((data.num_nodes, 4))
        data.patch_context = torch.zeros((data.num_nodes, 0))
        data.typed_token = torch.zeros(data.num_nodes, dtype=torch.long)
        data.parent_token = torch.zeros(data.num_nodes, dtype=torch.long)
        data.structural_token = torch.zeros(data.num_nodes, dtype=torch.long)
        data.structural_coarse = torch.zeros((data.num_nodes, 4))
        data.global_context = torch.zeros((1, 3))
        data.pair_index = torch.zeros((2, 0), dtype=torch.long)
        data.pair_relation = torch.zeros((0, 23))
        data.pair_bucket = torch.zeros(0, dtype=torch.long)
        data.topology_features = torch.zeros((1, 25))
    batch = sspe.struct_collate([data_a, data_b])
    # patch ids become globally contiguous
    assert batch.struct_patch.tolist() == [0, 0, 1, 2, 2]
    # graph b node block starts at 3 -> edges offset by 3
    assert batch.struct_src.tolist() == [0, 1, 3]
    assert batch.struct_dst.tolist() == [1, 0, 4]
    assert batch.struct_n_patches == 5


# ---------------------------------------------------------------------------
# 7-12: model-level checks
# ---------------------------------------------------------------------------


def test_candidate_has_no_typed_lookup_and_budget():
    model = sspe.build_candidate(0)
    assert getattr(model, "typed_embedding", None) is None
    assert [k for k in model.state_dict() if "typed_embedding" in k] == []
    assert [k for k, _ in model.named_parameters() if "typed_embedding" in k] == []
    total = sspe._n_params(model)
    assert sspe.PARAM_LOWER <= total <= sspe.PARAM_UPPER
    assert total == 84495


def test_candidate_preserves_cell_a_geometry():
    model = sspe.build_candidate(0)
    baseline = cd.build_cell("A", 0)
    assert int(model.pair_hidden) == 16
    assert int(model.patch_hidden) == 64
    assert int(model.recurrence_rounds) == 2
    assert int(model.patch_encoder_hidden) == 64
    assert int(model.global_encoder_hidden) == 32
    # parent path preserved (same module shape and parameter count)
    assert sspe._n_params(model.parent_embedding) == sspe._n_params(
        baseline.parent_embedding
    )
    # the patch encoder input width is unchanged because e_struct is token_width
    assert model.patch_encoder.layers[0].in_features == (
        baseline.patch_encoder.layers[0].in_features
    )
    assert model.unified_graph_width == baseline.unified_graph_width == 334


def test_recurrent_weight_tying_and_forward_backward():
    model = sspe.build_candidate(0)
    model.eval()
    atom = torch.tensor([0, 1, 2, 3])
    root = torch.tensor([1, 0, 0, 1])
    dist = torch.tensor([0, 1, 1, 0])
    patch = torch.tensor([0, 0, 0, 1])
    src = torch.tensor([0, 1])
    dst = torch.tensor([1, 2])
    bond = torch.tensor([0, 1])

    # a minimal but valid batch for the full recurrent forward
    data = Data(num_nodes=2, y=torch.tensor([0.0]))
    data.patch_cont = torch.zeros((2, zpp.SHELL_WIDTH))
    data.patch_context = torch.zeros((2, 0))
    data.typed_token = torch.zeros(2, dtype=torch.long)
    data.parent_token = torch.zeros(2, dtype=torch.long)
    data.structural_token = torch.zeros(2, dtype=torch.long)
    data.structural_coarse = torch.zeros((2, 4))
    data.global_context = torch.zeros((1, zpp.GLOBAL_WIDTH))
    data.pair_index = torch.tensor([[0], [1]], dtype=torch.long)
    data.pair_relation = torch.zeros((1, zpp.RELATION_WIDTH))
    data.pair_bucket = torch.zeros(1, dtype=torch.long)
    data.topology_features = torch.zeros((1, 25))
    data.struct_atom = atom
    data.struct_root = root
    data.struct_dist = dist
    data.struct_patch = patch
    data.struct_src = src
    data.struct_dst = dst
    data.struct_bond = bond
    batch = sspe.struct_collate([data])

    counts = model.module_call_counts(batch)
    assert counts == {"pair_projection": 4, "pair_encoder": 2, "center_update": 2}
    model.train()
    out = model(batch)
    loss = torch.nn.functional.l1_loss(out.view(-1), batch.y.view(-1))
    loss.backward()
    assert bool(torch.isfinite(out).all())
    assert all(
        p.grad is None or bool(torch.isfinite(p.grad).all())
        for p in model.parameters()
    )
    assert any(
        p.grad is not None and float(p.grad.abs().max()) > 0.0
        for p in model.parameters()
    )


# ---------------------------------------------------------------------------
# MolHIV projection accounting
# ---------------------------------------------------------------------------


def test_molhiv_projection_arithmetic():
    encoder = sspe._molhiv_projected_encoder()
    encoder_params = sspe._n_params(encoder)
    typed = sspe.MOLHIV_TYPED_LOOKUP_PARAMS
    projected = 1076589 - typed + encoder_params
    assert typed == 839456
    assert projected == 1076589 - 839456 + encoder_params
    # output width matches the MolHIV token width, so the MolHIV patch encoder
    # input width is unchanged
    assert encoder.output_dim == sspe.MOLHIV_TYPED_TOKEN_WIDTH == 32
    assert encoder_params == 35936
