"""Tests for the explicit structural basis with learned scalar valuation.

Fast unit / static tests only: no official ZINC data, no training.

Requirements covered (numbered as in the experiment brief):

1.  node relabeling preserves the patch output
2.  atom object support is exact
3.  bond object support is exact
4.  centred-3 object support is exact
5.  B1 count == number of real patch bonds
6.  B2 count == sum_v C(deg(v), 2)
7.  B1 endpoint swap does not change the valuation
8.  B2 endpoint u <-> w swap does not change the valuation
9.  changing an atom type changes the matching object valuation
10. changing a bond type changes the matching object valuation
11. same primitive counts but different connectivity -> different B2 basis
12. closure edge appears / disappears changes the explicit B2 feature
13. no message-passing modules
14. no attention modules
15. no vocab-sized lookup / certificate dependence
16. no learned support / existence gate (no sigmoid / softmax / top-k)
17. structural output is rank <= 1 by construction
18. output width is 16
19. h/q/T = 64/16/2
20. forward / backward finite
"""

from __future__ import annotations

import inspect

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import explicit_structural_basis as esbmod
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


def _batch(
    graph,
    *,
    atom=None,
    root=None,
    dist=None,
):
    n = int(graph.n_atoms)
    if atom is None:
        atom = np.zeros(n, dtype=np.int64)
    if root is None:
        root = np.zeros(n, dtype=np.int64)
        if n:
            root[0] = 1
    if dist is None:
        dist = np.zeros(n, dtype=np.int64)
    to_t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.int64))  # noqa: E731
    return _HostBatch(
        atom=to_t(atom),
        root=to_t(root),
        dist=to_t(dist),
        degree=torch.from_numpy(graph.degree),
        patch=torch.from_numpy(graph.patch),
        l1_u=torch.from_numpy(graph.l1_u),
        l1_v=torch.from_numpy(graph.l1_v),
        l1_bond=torch.from_numpy(graph.l1_bond),
        l1_patch=torch.from_numpy(graph.l1_patch),
        b2_center=torch.from_numpy(graph.b2_center),
        b2_u=torch.from_numpy(graph.b2_u),
        b2_w=torch.from_numpy(graph.b2_w),
        b2_bond_cu=torch.from_numpy(graph.b2_bond_cu),
        b2_bond_cw=torch.from_numpy(graph.b2_bond_cw),
        b2_closure=torch.from_numpy(graph.b2_closure),
        b2_patch=torch.from_numpy(graph.b2_patch),
        n_patches=graph.n_patches,
    )


class _HostBatch:
    def __init__(
        self,
        atom,
        root,
        dist,
        degree,
        patch,
        l1_u,
        l1_v,
        l1_bond,
        l1_patch,
        b2_center,
        b2_u,
        b2_w,
        b2_bond_cu,
        b2_bond_cw,
        b2_closure,
        b2_patch,
        n_patches,
    ) -> None:
        self.struct_atom = atom
        self.struct_root = root
        self.struct_dist = dist
        self.esb_degree = degree
        self.struct_patch = patch
        self.esb_l1_u = l1_u
        self.esb_l1_v = l1_v
        self.esb_l1_bond = l1_bond
        self.esb_l1_patch = l1_patch
        self.esb_b2_center = b2_center
        self.esb_b2_u = b2_u
        self.esb_b2_w = b2_w
        self.esb_b2_bond_cu = b2_bond_cu
        self.esb_b2_bond_cw = b2_bond_cw
        self.esb_b2_closure = b2_closure
        self.esb_b2_patch = b2_patch
        self.struct_n_patches = int(n_patches)


def _swap(batch, which):
    clone = _HostBatch(
        batch.struct_atom,
        batch.struct_root,
        batch.struct_dist,
        batch.esb_degree,
        batch.struct_patch,
        batch.esb_l1_u,
        batch.esb_l1_v,
        batch.esb_l1_bond,
        batch.esb_l1_patch,
        batch.esb_b2_center,
        batch.esb_b2_u,
        batch.esb_b2_w,
        batch.esb_b2_bond_cu,
        batch.esb_b2_bond_cw,
        batch.esb_b2_closure,
        batch.esb_b2_patch,
        batch.struct_n_patches,
    )
    if which == "b1":
        clone.esb_l1_u, clone.esb_l1_v = batch.esb_l1_v, batch.esb_l1_u
    else:
        clone.esb_b2_u, clone.esb_b2_w = batch.esb_b2_w, batch.esb_b2_u
        clone.esb_b2_bond_cu, clone.esb_b2_bond_cw = (
            batch.esb_b2_bond_cw,
            batch.esb_b2_bond_cu,
        )
    return clone


def _triangle_with_tail():
    return _basis([(0, 1), (1, 2), (0, 2), (2, 3)], 4)


# ---------------------------------------------------------------------------
# 2/3/4/5/6: exact supports and counts
# ---------------------------------------------------------------------------


def test_atom_object_support_exact():
    graph = _triangle_with_tail()
    for index in range(graph.n_atoms):
        assert atom_object_support(graph, index) == frozenset({index})


def test_bond_object_support_exact():
    graph = _triangle_with_tail()
    for index in range(graph.n_bonds):
        assert bond_object_support(graph, index) == frozenset(
            {int(graph.l1_u[index]), int(graph.l1_v[index])}
        )
        assert bond_object_bond_support(graph, index) == frozenset({index})


def test_triple_object_support_exact():
    graph = _triangle_with_tail()
    for index in range(graph.n_triples):
        assert triple_object_support(graph, index) == frozenset(
            {
                int(graph.b2_center[index]),
                int(graph.b2_u[index]),
                int(graph.b2_w[index]),
            }
        )
        assert triple_object_center(graph, index) == int(graph.b2_center[index])


def test_triple_bond_support_includes_closure():
    graph = _triangle_with_tail()
    closure_seen = False
    for index in range(graph.n_triples):
        support = triple_object_bond_support(graph, index)
        assert int(graph.b2_l1_cu[index]) in support
        assert int(graph.b2_l1_cw[index]) in support
        closure = int(graph.b2_l1_closure[index])
        if closure >= 0:
            assert closure in support
            closure_seen = True
        else:
            assert len(support) == 2
    assert closure_seen


def test_counts_match_topology():
    graph = _triangle_with_tail()
    # real undirected bonds: 0-1, 1-2, 0-2, 2-3
    assert object_count(graph, 1) == 4
    assert object_count(graph, 1) == graph.n_bonds
    assert object_count(graph, 2) == expected_triple_count(graph)
    # degrees = [2, 2, 3, 1] -> C(2,2)+C(2,2)+C(3,2) = 1 + 1 + 3 = 5
    assert expected_triple_count(graph) == 5


def test_centred_objects_can_share_support():
    graph = _triangle_with_tail()
    supports = [triple_object_support(graph, i) for i in range(graph.n_triples)]
    # the triangle contributes three centred objects over the same atom support
    assert supports.count(frozenset({0, 1, 2})) == 3
    centers = {triple_object_center(graph, i) for i in range(graph.n_triples)}
    assert centers == {0, 1, 2}


# ---------------------------------------------------------------------------
# 7/8: endpoint-swap invariance of the valuations
# ---------------------------------------------------------------------------


def test_bond_endpoint_swap_invariant():
    graph = _basis([(0, 1), (1, 2), (2, 3)], 4)
    encoder = ExplicitStructuralBasisEncoder().eval()
    batch = _batch(graph, dist=np.asarray([0, 1, 2, 3], dtype=np.int64))
    with torch.no_grad():
        reference = encoder(batch)
        swapped = encoder(_swap(batch, "b1"))
    assert float((reference - swapped).abs().max()) < 1.0e-6


def test_triple_endpoint_swap_invariant():
    graph = _triangle_with_tail()
    encoder = ExplicitStructuralBasisEncoder().eval()
    batch = _batch(graph, dist=np.asarray([0, 1, 1, 2], dtype=np.int64))
    with torch.no_grad():
        reference = encoder(batch)
        swapped = encoder(_swap(batch, "b2"))
    assert float((reference - swapped).abs().max()) < 1.0e-6


# ---------------------------------------------------------------------------
# 9/10: primitive perturbations move the matching valuation
# ---------------------------------------------------------------------------


def test_atom_and_bond_type_change_valuations():
    graph = _triangle_with_tail()
    encoder = ExplicitStructuralBasisEncoder().eval()
    batch = _batch(graph)
    with torch.no_grad():
        base = encoder(batch, return_details=True)
        atom_batch = _swap(batch, "b1")  # reuse clone-with-same-tables
        atom_batch.struct_atom = batch.struct_atom.clone()
        atom_batch.struct_atom[0] = (int(batch.struct_atom[0]) + 1) % 28
        atom_out = encoder(atom_batch, return_details=True)
        bond_batch = _swap(batch, "b1")
        bond_batch.esb_l1_bond = batch.esb_l1_bond.clone()
        bond_batch.esb_l1_bond[0] = (int(batch.esb_l1_bond[0]) + 1) % 4
        bond_out = encoder(bond_batch, return_details=True)
        triple_batch = _swap(batch, "b2")
        triple_batch.esb_b2_bond_cu = batch.esb_b2_bond_cu.clone()
        triple_batch.esb_b2_bond_cu[0] = (int(batch.esb_b2_bond_cu[0]) + 1) % 4
        triple_out = encoder(triple_batch, return_details=True)
    assert float((atom_out["t0"] - base["t0"]).abs().max()) > 1.0e-6
    assert float((bond_out["t1"] - base["t1"]).abs().max()) > 1.0e-6
    assert float((triple_out["t2"] - base["t2"]).abs().max()) > 1.0e-6


# ---------------------------------------------------------------------------
# 11/12: connectivity and closure
# ---------------------------------------------------------------------------


def test_same_primitives_different_connectivity_changes_b2():
    # Seven-atom graphs: same degree multiset, same rooted distance multiset,
    # same B1 endpoint-primitive multiset, different connectivity.
    tree_a = _basis(
        [
            (0, 1), (0, 2), (0, 4), (0, 5), (1, 2), (1, 5), (1, 6),
            (2, 4), (2, 6), (3, 5), (3, 6), (4, 5), (4, 6),
        ],
        7,
    )
    tree_b = _basis(
        [
            (0, 3), (0, 4), (0, 5), (0, 6), (1, 2), (1, 3), (1, 4),
            (1, 6), (2, 6), (3, 4), (3, 5), (4, 5), (5, 6),
        ],
        7,
    )
    dist_a = np.asarray([0, 1, 1, 2, 1, 1, 2], dtype=np.int64)
    dist_b = np.asarray([0, 2, 2, 1, 1, 1, 1], dtype=np.int64)
    assert sorted(zip(tree_a.degree.tolist(), dist_a.tolist())) == sorted(
        zip(tree_b.degree.tolist(), dist_b.tolist())
    )
    assert sorted(tree_a.l1_bond.tolist()) == sorted(tree_b.l1_bond.tolist())

    def _b1(graph, types):
        return tuple(
            sorted(
                tuple(
                    sorted(
                        [
                            types[int(graph.l1_u[i])],
                            types[int(graph.l1_v[i])],
                        ]
                    )
                )
                for i in range(graph.n_bonds)
            )
        )

    types_a = list(zip(tree_a.degree.tolist(), dist_a.tolist()))
    types_b = list(zip(tree_b.degree.tolist(), dist_b.tolist()))
    assert _b1(tree_a, types_a) == _b1(tree_b, types_b)

    def _b2(graph, types):
        rows = []
        for index in range(graph.n_triples):
            center = int(graph.b2_center[index])
            endpoint_u = int(graph.b2_u[index])
            endpoint_w = int(graph.b2_w[index])
            rows.append(
                (
                    types[center],
                    tuple(sorted([types[endpoint_u], types[endpoint_w]])),
                    int(graph.b2_closure[index]) >= 0,
                )
            )
        return tuple(sorted(rows))

    assert _b2(tree_a, types_a) != _b2(tree_b, types_b)
    encoder = ExplicitStructuralBasisEncoder().eval()
    root = np.zeros(7, dtype=np.int64)
    root[0] = 1
    with torch.no_grad():
        out_a = encoder(_batch(tree_a, root=root, dist=dist_a))
        out_b = encoder(_batch(tree_b, root=root, dist=dist_b))
    assert float((out_a - out_b).abs().max()) > 1.0e-6


def test_closure_edge_changes_b2_feature():
    path = _basis([(0, 1), (0, 2)], 3)
    closed = _basis([(0, 1), (0, 2), (1, 2)], 3)
    path_closure = int(path.b2_closure[0])
    closed_closure = int(closed.b2_closure[0])
    assert path_closure == -1
    assert closed_closure >= 0
    assert path.n_bonds != closed.n_bonds
    # the unique B2 object has the same centre/endpoints but gains a closure
    assert int(path.b2_center[0]) == int(closed.b2_center[0])
    assert {int(path.b2_u[0]), int(path.b2_w[0])} == {
        int(closed.b2_u[0]),
        int(closed.b2_w[0]),
    }


# ---------------------------------------------------------------------------
# 1: node relabel invariance
# ---------------------------------------------------------------------------


def test_node_relabel_invariant():
    graph = _basis([(0, 1), (1, 2), (0, 2)], 3)
    encoder = ExplicitStructuralBasisEncoder().eval()
    with torch.no_grad():
        reference = encoder(_batch(graph, dist=np.asarray([0, 1, 2])))
    permutation = {0: 2, 1: 0, 2: 1}
    relabelled = [
        (permutation[u], permutation[v]) for u, v in [(0, 1), (1, 2), (0, 2)]
    ]
    graph2 = _basis(relabelled, 3)
    root = np.zeros(3, dtype=np.int64)
    root[permutation[0]] = 1
    dist = np.zeros(3, dtype=np.int64)
    dist[permutation[1]] = 1
    dist[permutation[2]] = 2
    with torch.no_grad():
        moved = encoder(_batch(graph2, root=root, dist=dist))
    assert float((reference - moved).abs().max()) < 1.0e-6


# ---------------------------------------------------------------------------
# 13/14/15/16: no message passing / attention / lookup / gate
# ---------------------------------------------------------------------------


def test_no_message_passing_or_attention_modules():
    forbidden = (
        "gineconv",
        "gcnconv",
        "graphconv",
        "messagepassing",
        "transformer",
        "multiheadattention",
    )
    encoder = ExplicitStructuralBasisEncoder()
    assert encoder.rounds == 0
    names = [name for name, _ in encoder.named_modules()]
    assert not any("message" in name for name in names)
    assert not any("attention" in name for name in names)
    for module in encoder.modules():
        class_name = type(module).__name__.lower()
        assert not any(token in class_name for token in forbidden)


def test_forward_never_reads_edge_endpoints():
    graph = _triangle_with_tail()
    encoder = ExplicitStructuralBasisEncoder().eval()
    batch = _batch(graph)

    class _NoEndpoints:
        def __getattr__(self, name):
            if name in {"struct_src", "struct_dst"}:
                raise AttributeError(name)
            return getattr(batch, name)

    with torch.no_grad():
        reference = encoder(batch)
        stripped = encoder(_NoEndpoints())
    assert torch.equal(reference, stripped)
    source = inspect.getsource(ExplicitStructuralBasisEncoder.forward).lower()
    assert "struct_src" not in source
    assert "struct_dst" not in source


def test_no_learned_support_gate():
    source = inspect.getsource(ExplicitStructuralBasisEncoder).lower()
    for token in ("sigmoid(", "nn.sigmoid", "softmax(", "nn.softmax", "topk(", "gumbel"):
        assert token not in source


# ---------------------------------------------------------------------------
# 17/18: rank-1 channel + width
# ---------------------------------------------------------------------------


def test_rank1_channel_by_construction():
    graph = _triangle_with_tail()
    encoder = ExplicitStructuralBasisEncoder().eval()
    batch = _batch(graph, dist=np.asarray([0, 1, 1, 2]))
    with torch.no_grad():
        e_struct = encoder(batch)
    delta = e_struct - encoder.rank1_bias.unsqueeze(0)
    delta = delta - delta.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(delta)
    assert int((singular > 1.0e-6 * float(singular[0].clamp_min(1.0e-12))).sum()) <= 1
    assert e_struct.shape[1] == 16


def test_zeroing_two_objects_still_rank_one():
    encoder = ExplicitStructuralBasisEncoder().eval()
    graph = _basis([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)], 5)
    batch = _batch(graph)
    with torch.no_grad():
        e_struct = encoder(batch)
    delta = e_struct - encoder.rank1_bias.unsqueeze(0)
    delta = delta - delta.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(delta)
    rank = int((singular > 1.0e-6 * float(singular[0].clamp_min(1.0e-12))).sum())
    assert rank <= 1


# ---------------------------------------------------------------------------
# 19/20 + model wiring
# ---------------------------------------------------------------------------


def test_forward_backward_finite():
    graph = _triangle_with_tail()
    encoder = ExplicitStructuralBasisEncoder()
    out = encoder(_batch(graph))
    assert out.shape[1] == 16
    assert bool(torch.isfinite(out).all())
    out.sum().backward()
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in encoder.parameters()
    )


def test_encoder_parameter_count():
    encoder = ExplicitStructuralBasisEncoder()
    total = sum(p.numel() for p in encoder.parameters())
    assert total == 35184


def test_model_budget_and_geometry():
    from tracks.ksvd.experiments.luyin16 import (
        zinc_explicit_structural_basis as esb,
    )

    model = esb.build_candidate(0)
    total = sum(p.numel() for p in model.parameters())
    assert abs(total - 84495) / 84495 <= 0.01
    assert int(model.pair_hidden) == 16
    assert int(model.patch_hidden) == 64
    assert int(model.recurrence_rounds) == 2
    assert getattr(model, "typed_embedding", None) is None
    assert not [k for k in model.state_dict() if "typed_embedding" in k]
    assert model.structural_encoder.output_dim == 16
    assert model.structural_encoder.kind == "explicit_basis_rank1"


def test_build_explicit_basis_graph_deterministic():
    first = _triangle_with_tail()
    second = _triangle_with_tail()
    assert first.b2_center.tolist() == second.b2_center.tolist()
    assert first.b2_u.tolist() == second.b2_u.tolist()
    assert first.b2_closure.tolist() == second.b2_closure.tolist()
    assert first.l1_u.tolist() == second.l1_u.tolist()
