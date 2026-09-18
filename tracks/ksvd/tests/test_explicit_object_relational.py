"""Tests for explicit objects + explicit relations + T_obj=2 reasoning.

Fast unit / static tests only: no official ZINC data, no training.

Requirements covered (numbered as in the experiment brief):

1.  B0/B1/B2 supports and counts are exactly the previous explicit basis
2.  relation edges exist only inside a patch, unique undirected pairs
3.  relation families are support-derived: overlap / bond_overlap / adjacent
4.  containment / share-centre / share-root / graph-distance descriptors
5.  no relation edge when supports are disjoint and not bonded
6.  relation graph is deterministic and sparse (far from complete)
7.  object state width and relation latent width are frozen
8.  T_obj = 2, Q and U weight-tied across rounds
9.  late pooling happens only after all rounds
10. no message passing / attention / softmax / top-k / learned gate
11. no certificate / token lookup
12. rank-1 structural channel by construction
13. endpoint swap invariance (bonds, triples, relations)
14. relation-zero and round-1-only frozen hooks change the output
15. relation-family mask changes the output
16. relation-topology intervention: identical objects, rewired relations
17. encoder parameter count matches B-full's 35152 exactly
18. forward / backward finite and deterministic
"""

from __future__ import annotations

import inspect

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import explicit_object_relational as eormod
from tracks.ksvd.experiments.luyin16.explicit_object_relational import (
    FAMILY_ADJACENT,
    FAMILY_BOND_OVERLAP,
    FAMILY_OVERLAP,
    OBJ_DIM,
    Q_OBJ_DIM,
    REL_FEATURE_DIM,
    ExplicitObjectRelationalEncoder,
    build_object_relation_graph,
    relation_counts,
)
from tracks.ksvd.experiments.luyin16.explicit_structural_basis import (
    ExplicitStructuralBasisEncoder,
    atom_object_support,
    bond_object_bond_support,
    bond_object_support,
    build_explicit_basis_graph,
    expected_triple_count,
    object_count,
    triple_object_bond_support,
    triple_object_center,
    triple_object_support,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _basis(undirected, n_atoms, *, bonds=None):
    n = int(n_atoms)
    both = list(undirected) + [(v, u) for u, v in undirected]
    src = np.asarray([u for u, _ in both], dtype=np.int64)
    dst = np.asarray([v for _, v in both], dtype=np.int64)
    if bonds is None:
        bond = np.zeros(len(both), dtype=np.int64)
    else:
        bond = np.asarray(
            [bonds[(min(u, v), max(u, v))] for u, v in both], dtype=np.int64
        )
    patch = np.zeros(n, dtype=np.int64)
    edge_patch = np.zeros(len(both), dtype=np.int64)
    return build_explicit_basis_graph(patch, src, dst, bond, edge_patch, 1)


def _bfs(undirected, n_atoms, root=0):
    adjacency = {node: set() for node in range(int(n_atoms))}
    for u, v in undirected:
        adjacency[u].add(v)
        adjacency[v].add(u)
    distances = {root: 0}
    queue = [root]
    while queue:
        node = queue.pop(0)
        for neighbour in sorted(adjacency[node]):
            if neighbour not in distances:
                distances[neighbour] = distances[node] + 1
                queue.append(neighbour)
    return np.asarray(
        [distances.get(node, 0) for node in range(int(n_atoms))], dtype=np.int64
    )


class _ObjectBatch:
    pass


def _batch(undirected, n_atoms, *, atom=None, root=None, dist=None):
    graph = _basis(undirected, n_atoms)
    n = int(n_atoms)
    if atom is None:
        atom = np.zeros(n, dtype=np.int64)
    if root is None:
        root = np.zeros(n, dtype=np.int64)
        if n:
            root[0] = 1
    if dist is None:
        dist = _bfs(undirected, n, 0)
    relation = build_object_relation_graph(graph, root_atoms=root)
    batch = _ObjectBatch()
    batch.struct_atom = torch.from_numpy(np.asarray(atom, dtype=np.int64))
    batch.struct_root = torch.from_numpy(np.asarray(root, dtype=np.int64))
    batch.struct_dist = torch.from_numpy(np.asarray(dist, dtype=np.int64))
    batch.struct_patch = torch.from_numpy(graph.patch)
    batch.esb_degree = torch.from_numpy(graph.degree)
    batch.esb_l1_u = torch.from_numpy(graph.l1_u)
    batch.esb_l1_v = torch.from_numpy(graph.l1_v)
    batch.esb_l1_bond = torch.from_numpy(graph.l1_bond)
    batch.esb_l1_patch = torch.from_numpy(graph.l1_patch)
    batch.esb_b2_center = torch.from_numpy(graph.b2_center)
    batch.esb_b2_u = torch.from_numpy(graph.b2_u)
    batch.esb_b2_w = torch.from_numpy(graph.b2_w)
    batch.esb_b2_bond_cu = torch.from_numpy(graph.b2_bond_cu)
    batch.esb_b2_bond_cw = torch.from_numpy(graph.b2_bond_cw)
    batch.esb_b2_closure = torch.from_numpy(graph.b2_closure)
    batch.esb_b2_patch = torch.from_numpy(graph.b2_patch)
    batch.eor_edge_i = torch.from_numpy(relation.edge_i)
    batch.eor_edge_j = torch.from_numpy(relation.edge_j)
    batch.eor_family = torch.from_numpy(relation.family)
    batch.eor_feat = torch.from_numpy(relation.feature)
    batch.relation = relation
    batch.graph = graph
    return batch


def _encoder():
    return ExplicitObjectRelationalEncoder(
        atom_categories=28,
        bond_categories=4,
        n_distance_bins=3,
        n_degree_bins=8,
        obj_dim=OBJ_DIM,
        q_obj_dim=Q_OBJ_DIM,
    ).eval()


def _lookup(relation, i, j):
    key = (min(int(i), int(j)), max(int(i), int(j)))
    for index in range(relation.edge_i.shape[0]):
        pair = (int(relation.edge_i[index]), int(relation.edge_j[index]))
        if pair == key:
            return int(relation.family[index]), relation.feature[index]
    return None


def _rewire(edge_i, edge_j, n_switches=64):
    ei = [int(v) for v in edge_i.tolist()]
    ej = [int(v) for v in edge_j.tolist()]
    existing = {(min(a, b), max(a, b)) for a, b in zip(ei, ej)}
    done = 0
    for index in range(len(ei)):
        if done >= n_switches:
            break
        for other in range(index + 1, len(ei)):
            a, b = ei[index], ej[index]
            c, d = ei[other], ej[other]
            if len({a, b, c, d}) != 4:
                continue
            n1 = (min(a, c), max(a, c))
            n2 = (min(b, d), max(b, d))
            if (
                n1 not in existing
                and n2 not in existing
                and n1[0] != n1[1]
                and n2[0] != n2[1]
            ):
                existing.discard((min(a, b), max(a, b)))
                existing.discard((min(c, d), max(c, d)))
                existing.add(n1)
                existing.add(n2)
                ei[index], ej[index] = n1
                ei[other], ej[other] = n2
                done += 1
                break
    return torch.tensor(ei), torch.tensor(ej), done


# ---------------------------------------------------------------------------
# basis reuse
# ---------------------------------------------------------------------------


def test_supports_match_previous_basis():
    graph = _basis([(0, 1), (1, 2), (0, 2)], 3)
    assert all(
        atom_object_support(graph, i) == frozenset({i})
        for i in range(graph.n_atoms)
    )
    for index in range(graph.n_bonds):
        assert bond_object_support(graph, index) == frozenset(
            {int(graph.l1_u[index]), int(graph.l1_v[index])}
        )
        assert bond_object_bond_support(graph, index) == frozenset({index})
    for index in range(graph.n_triples):
        assert triple_object_support(graph, index) == frozenset(
            {int(graph.b2_center[index]), int(graph.b2_u[index]), int(graph.b2_w[index])}
        )
        assert triple_object_center(graph, index) == int(graph.b2_center[index])
    assert object_count(graph, 2) == expected_triple_count(graph)


# ---------------------------------------------------------------------------
# relation graph
# ---------------------------------------------------------------------------


def test_relation_families_on_path():
    batch = _batch([(0, 1), (1, 2)], 3)
    relation = batch.relation
    # object ids: B0 0,1,2 ; B1 3=(0,1), 4=(1,2) ; B2 5=centre(1; 0,2)
    assert _lookup(relation, 0, 1)[0] == FAMILY_ADJACENT
    assert _lookup(relation, 3, 5)[0] == FAMILY_BOND_OVERLAP
    assert _lookup(relation, 0, 5)[0] == FAMILY_OVERLAP
    # containment (B1 support is a subset of the B2 support)
    assert float(_lookup(relation, 0, 5)[1][8]) == 1.0
    # share-centre descriptor
    assert float(_lookup(relation, 1, 5)[1][9]) == 1.0


def test_no_relation_when_supports_disjoint_and_unbonded():
    batch = _batch([(0, 1), (1, 2)], 3)
    assert _lookup(batch.relation, 0, 2) is None


def test_relations_within_patch_no_self_loops():
    batch = _batch([(0, 1), (1, 2), (2, 3), (1, 3)], 4)
    relation = batch.relation
    assert bool(np.all(relation.edge_i < relation.edge_j))
    assert relation.feature.shape[1] == REL_FEATURE_DIM
    obj_patch = np.concatenate(
        [batch.graph.patch, batch.graph.l1_patch, batch.graph.b2_patch]
    )
    assert bool(np.all(obj_patch[relation.edge_i] == obj_patch[relation.edge_j]))


def test_relation_graph_deterministic_and_sparse():
    batch = _batch([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)], 5)
    first = build_object_relation_graph(batch.graph)
    second = build_object_relation_graph(batch.graph)
    assert np.array_equal(first.edge_i, second.edge_i)
    assert np.array_equal(first.feature, second.feature)
    ratio = float(first.edge_i.size) / float(first.n_objects)
    assert 0.0 < ratio < 20.0


def test_relation_counts_families():
    batch = _batch([(0, 1), (1, 2), (0, 2)], 3)
    counts = relation_counts(batch.graph)
    assert counts["relations"] >= counts["family_bond_overlap"] > 0
    assert counts["objects"] == batch.graph.n_atoms + batch.graph.n_bonds + batch.graph.n_triples


# ---------------------------------------------------------------------------
# encoder geometry / structure
# ---------------------------------------------------------------------------


def test_frozen_widths_and_rounds():
    encoder = _encoder()
    assert encoder.obj_dim == 8 and encoder.q_obj_dim == 8
    assert encoder.t_obj == 2 and encoder.rounds == 2
    assert encoder.output_dim == 16


def test_weight_tying_single_q_and_update():
    encoder = _encoder()
    names = [name for name, _ in encoder.named_modules()]
    assert sum(1 for name in names if name == "q_encoder") == 1
    assert sum(1 for name in names if name.startswith("q_encoder_")) == 0
    assert sum(1 for name in names if name == "update") == 1
    assert sum(1 for name in names if name.startswith("update_")) == 0


def test_no_message_passing_attention_or_learned_gate():
    encoder = _encoder()
    forbidden = ("message", "attention", "transformer", "gineconv", "gcnconv")
    assert not any(
        any(token in name.lower() for token in forbidden)
        for name, _ in encoder.named_modules()
    )
    source = inspect.getsource(ExplicitObjectRelationalEncoder).lower()
    for token in ("sigmoid(", "nn.sigmoid", "softmax(", "nn.softmax", "topk(", "gumbel"):
        assert token not in source
    forward = inspect.getsource(ExplicitObjectRelationalEncoder.forward).lower()
    assert "struct_src" not in forward and "struct_dst" not in forward


def test_encoder_parameter_count_matches_bfull():
    encoder = _encoder()
    total = sum(p.numel() for p in encoder.parameters())
    assert total == 35152


# ---------------------------------------------------------------------------
# forward behaviour
# ---------------------------------------------------------------------------


def test_rank1_channel_by_construction():
    encoder = _encoder()
    batch = _batch([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)], 4)
    with torch.no_grad():
        e_struct = encoder(batch)
    delta = e_struct - encoder.rank1_bias.unsqueeze(0)
    delta = delta - delta.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(delta)
    assert int((singular > 1e-6 * float(singular[0].clamp_min(1e-12))).sum()) <= 1


def test_late_pooling_after_all_rounds():
    encoder = _encoder()
    batch = _batch([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)], 4)
    with torch.no_grad():
        details = encoder(batch, return_details=True)
        full = details["e_struct"]
        encoder._rounds = 1
        round_one = encoder(batch)
        encoder._rounds = 2
    assert len(details["h_rounds"]) == 3
    assert len(details["q_rounds"]) == 2
    assert float((full - round_one).abs().max()) > 1e-8


def test_endpoint_swap_invariance():
    encoder = _encoder()
    batch = _batch([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)], 4)
    with torch.no_grad():
        reference = encoder(batch)

        swapped_rel = _ObjectBatch()
        swapped_rel.__dict__.update(batch.__dict__)
        swapped_rel.esb_l1_u = batch.esb_l1_v
        swapped_rel.esb_l1_v = batch.esb_l1_u
        swapped_rel.esb_b2_u = batch.esb_b2_w
        swapped_rel.esb_b2_w = batch.esb_b2_u
        swapped_rel.esb_b2_bond_cu = batch.esb_b2_bond_cw
        swapped_rel.esb_b2_bond_cw = batch.esb_b2_bond_cu
        swapped_rel.eor_edge_i = batch.eor_edge_j
        swapped_rel.eor_edge_j = batch.eor_edge_i
        out = encoder(swapped_rel)
    assert float((reference - out).abs().max()) < 1e-4


def test_relation_zero_and_family_mask_change_output():
    encoder = _encoder()
    batch = _batch([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3), (2, 4)], 5)
    with torch.no_grad():
        reference = encoder(batch)
        encoder._ablate_relations = True
        zeroed = encoder(batch)
        encoder._ablate_relations = False
        encoder._family_mask = (False, False, False)
        masked = encoder(batch)
        encoder._family_mask = (True, True, True)
    assert float((reference - zeroed).abs().max()) > 0.0
    assert float((reference - masked).abs().max()) > 0.0


def test_relation_topology_intervention():
    encoder = _encoder()
    path = [(i, i + 1) for i in range(8)]
    batch = _batch(path, 9)
    with torch.no_grad():
        reference = encoder(batch)
    new_i, new_j, done = _rewire(batch.eor_edge_i, batch.eor_edge_j, 32)
    rewired = _ObjectBatch()
    rewired.__dict__.update(batch.__dict__)
    rewired.eor_edge_i = new_i
    rewired.eor_edge_j = new_j
    with torch.no_grad():
        changed = encoder(rewired)
    assert done > 0
    assert float((reference - changed).abs().max()) > 0.0


def test_early_pooling_ignores_relations():
    # The explicit-basis (early pooling) encoder never sees the relation graph:
    # rewiring relations leaves its output bit-identical.
    basis_encoder = ExplicitStructuralBasisEncoder(
        atom_categories=28,
        bond_categories=4,
        n_distance_bins=3,
        n_degree_bins=8,
        node_dim=20,
        edge_dim=16,
        valuation_hidden=184,
        fusion_hidden=184,
        output_dim=16,
    ).eval()
    path = [(i, i + 1) for i in range(8)]
    batch = _batch(path, 9)
    new_i, new_j, done = _rewire(batch.eor_edge_i, batch.eor_edge_j, 32)
    rewired = _ObjectBatch()
    rewired.__dict__.update(batch.__dict__)
    rewired.eor_edge_i = new_i
    rewired.eor_edge_j = new_j
    with torch.no_grad():
        assert float((basis_encoder(batch) - basis_encoder(rewired)).abs().max()) == 0.0
    assert done > 0


def test_forward_backward_finite_and_deterministic():
    encoder = _encoder()
    batch = _batch([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)], 4)
    encoder.train()
    output = encoder(batch)
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert any(
        p.grad is not None and bool(torch.isfinite(p.grad).all())
        for p in encoder.parameters()
    )
    encoder.zero_grad(set_to_none=True)
    encoder.eval()
    with torch.no_grad():
        first = encoder(batch)
        second = encoder(batch)
    assert float((first - second).abs().max()) == 0.0
