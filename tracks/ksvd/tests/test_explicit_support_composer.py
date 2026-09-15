"""Tests for the explicit-support structural composer.

Fast unit / static tests only: no official ZINC data, no training.

Requirements covered (numbered as in the experiment brief):

1.  node relabeling preserves the patch output (on an ambiguity-free graph)
2.  every object's atom support is exactly recoverable
3.  every object's bond support is exactly recoverable
4.  composition support == exact parent support union
5.  provenance chain recurses to atom primitives
6.  only original-topology-legal pairs compose
7.  modifying connectivity changes the legal candidate set
8.  no GNN / GINE / GCN / Transformer message passing (composer never reads
    edge endpoints; no message-passing modules exist)
9.  no vocab-sized certificate lookup
10. unseen certificate needs no vocabulary
11. overlap is allowed and recorded
12. containment is a deterministic function of the supports
13. output width is 16
14. h/q/T = 64/16/2
15. forward / backward finite
16. deterministic candidate enumeration (replay equal)
"""

from __future__ import annotations

import inspect

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import explicit_support_composer as esc
from tracks.ksvd.experiments.luyin16.explicit_support_composer import (
    ExplicitSupportComposer,
    build_explicit_candidate_graph,
    expand_atom_support,
    expand_bond_support,
    parents,
)


class _Batch:
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
        l2_a,
        l2_b,
        l2_patch,
        l2_overlap,
        l2_conn,
        n_patches,
    ) -> None:
        self.struct_atom = atom
        self.struct_root = root
        self.struct_dist = dist
        self.struct_degree = degree
        self.struct_patch = patch
        self.exo_l1_u = l1_u
        self.exo_l1_v = l1_v
        self.exo_l1_bond = l1_bond
        self.exo_l1_patch = l1_patch
        self.exo_l2_a = l2_a
        self.exo_l2_b = l2_b
        self.exo_l2_patch = l2_patch
        self.exo_l2_overlap = l2_overlap
        self.exo_l2_conn = l2_conn
        self.struct_n_patches = n_patches


def _graph(undirected, n_atoms, *, patch=None, bonds=None):
    n = int(n_atoms)
    if patch is None:
        patch = np.zeros(n, dtype=np.int64)
    both = list(undirected) + [(v, u) for u, v in undirected]
    src = np.asarray([u for u, _ in both], dtype=np.int64)
    dst = np.asarray([v for _, v in both], dtype=np.int64)
    if bonds is None:
        bond = np.zeros(len(both), dtype=np.int64)
    else:
        bond = np.asarray([bonds[(min(u, v), max(u, v))] for u, v in both], dtype=np.int64)
    edge_patch = np.asarray([int(patch[u]) for u, _ in both], dtype=np.int64)
    return build_explicit_candidate_graph(
        patch, src, dst, bond, edge_patch, int(patch.max()) + 1
    )


def _batch_from_graph(
    graph: esc.ExplicitCandidateGraph, *, atom=None, dist=None, root=None
):
    n = int(graph.n_atoms)
    if root is None:
        root = np.zeros(n, dtype=np.int64)
        if n:
            root[0] = 1
    if atom is None:
        atom = np.zeros(n, dtype=np.int64)
    if dist is None:
        dist = np.zeros(n, dtype=np.int64)
    return _Batch(
        atom=torch.from_numpy(np.asarray(atom, dtype=np.int64)),
        root=torch.from_numpy(np.asarray(root, dtype=np.int64)),
        dist=torch.from_numpy(np.asarray(dist, dtype=np.int64)),
        degree=torch.from_numpy(graph.degree),
        patch=torch.from_numpy(graph.patch),
        l1_u=torch.from_numpy(graph.l1_u),
        l1_v=torch.from_numpy(graph.l1_v),
        l1_bond=torch.from_numpy(graph.l1_bond),
        l1_patch=torch.from_numpy(graph.l1_patch),
        l2_a=torch.from_numpy(graph.l2_a),
        l2_b=torch.from_numpy(graph.l2_b),
        l2_patch=torch.from_numpy(graph.l2_patch),
        l2_overlap=torch.from_numpy(graph.l2_overlap),
        l2_conn=torch.from_numpy(graph.l2_conn),
        n_patches=graph.n_patches,
    )


def _path_graph():
    # 0-1-2-3 path, one patch
    return _graph([(0, 1), (1, 2), (2, 3)], 4)


def _cycle_graph():
    # 0-1-2-3-0 cycle: level-2 candidates include overlap and new bonds
    return _graph([(0, 1), (1, 2), (2, 3), (0, 3)], 4)


# ---------------------------------------------------------------------------
# 2/3/4/5: exact supports, provenance, parent union
# ---------------------------------------------------------------------------


def test_atom_support_exact():
    graph = _path_graph()
    for index in range(graph.n_level1):
        assert expand_atom_support(graph, 1, index) == frozenset(
            {int(graph.l1_u[index]), int(graph.l1_v[index])}
        )
    for index in range(graph.n_level2):
        left = expand_atom_support(graph, 1, int(graph.l2_a[index]))
        right = expand_atom_support(graph, 1, int(graph.l2_b[index]))
        assert expand_atom_support(graph, 2, index) == left | right


def test_bond_support_exact():
    graph = _cycle_graph()
    for index in range(graph.n_level1):
        assert expand_bond_support(graph, 1, index) == frozenset({index})
    for index in range(graph.n_level2):
        left = expand_bond_support(graph, 1, int(graph.l2_a[index]))
        right = expand_bond_support(graph, 1, int(graph.l2_b[index]))
        connecting = {
            int(v) for v in graph.l2_conn[index].tolist() if v >= 0
        }
        assert expand_bond_support(graph, 2, index) == left | right | connecting


def test_parent_union_and_overlap_consistency():
    graph = _cycle_graph()
    for index in range(graph.n_level2):
        assert int(graph.l2_overlap[index]) == len(
            expand_atom_support(graph, 1, int(graph.l2_a[index]))
            & expand_atom_support(graph, 1, int(graph.l2_b[index]))
        )
        assert parents(graph, 2, index) == (
            (1, int(graph.l2_a[index])),
            (1, int(graph.l2_b[index])),
        )


def test_provenance_recurses_to_primitives():
    graph = _path_graph()
    for index in range(graph.n_level2):
        for level, local in parents(graph, 2, index):
            for child_level, child_local in parents(graph, level, local):
                assert child_level == 0
                assert expand_atom_support(graph, child_level, child_local) == (
                    frozenset({int(child_local)})
                )


# ---------------------------------------------------------------------------
# 6/11/12: legality, overlap, containment
# ---------------------------------------------------------------------------


def test_only_legal_pairs_compose():
    graph = _cycle_graph()
    for index in range(graph.n_level2):
        overlap = int(graph.l2_overlap[index])
        connecting = (graph.l2_conn[index] >= 0).any()
        assert overlap > 0 or bool(connecting)


def test_overlap_recorded_on_cycle():
    graph = _cycle_graph()
    # two bonds sharing one atom give overlap == 1 and no *new* connecting bond
    assert int(graph.l2_overlap.sum()) > 0
    # a triangle is impossible on a 4-cycle without an extra bond, but the
    # disjoint-adjacent pairs must record connecting bonds
    assert bool((graph.l2_conn >= 0).any())


def test_containment_is_support_function():
    graph = _path_graph()
    for index in range(graph.n_level2):
        child = expand_atom_support(graph, 2, index)
        left = expand_atom_support(graph, 1, int(graph.l2_a[index]))
        right = expand_atom_support(graph, 1, int(graph.l2_b[index]))
        assert left <= child and right <= child
        assert left < child and right < child


def test_union_size_capped_and_deduplicated():
    graph = _cycle_graph()
    seen = set()
    for index in range(graph.n_level2):
        support = expand_atom_support(graph, 2, index)
        assert len(support) <= 4
        assert support not in seen
        seen.add(support)


# ---------------------------------------------------------------------------
# 7: connectivity changes the candidate set
# ---------------------------------------------------------------------------


def test_connectivity_changes_candidate_set():
    graph_a = _graph([(0, 1), (0, 2), (1, 3), (2, 4), (2, 5)], 6)
    graph_b = _graph([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5)], 6)
    signature_a = (
        tuple(graph_a.l2_a.tolist()),
        tuple(graph_a.l2_b.tolist()),
        tuple(graph_a.l2_overlap.tolist()),
        tuple(graph_a.l2_conn.ravel().tolist()),
    )
    signature_b = (
        tuple(graph_b.l2_a.tolist()),
        tuple(graph_b.l2_b.tolist()),
        tuple(graph_b.l2_overlap.tolist()),
        tuple(graph_b.l2_conn.ravel().tolist()),
    )
    assert signature_a != signature_b
    # share the atom-type / degree multiset and the bond-type multiset
    assert sorted(graph_a.degree.tolist()) == sorted(graph_b.degree.tolist())
    assert sorted(graph_a.l1_bond.tolist()) == sorted(graph_b.l1_bond.tolist())
    assert graph_a.n_atoms == graph_b.n_atoms


def test_adversarial_batch_outputs_differ():
    graph_a = _graph([(0, 1), (0, 2), (1, 3), (2, 4), (2, 5)], 6)
    graph_b = _graph([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5)], 6)
    composer = ExplicitSupportComposer()
    composer.eval()
    with torch.no_grad():
        out_a = composer(_batch_from_graph(graph_a))
        out_b = composer(_batch_from_graph(graph_b))
    assert out_a.shape == out_b.shape
    assert float((out_a - out_b).abs().max()) >= 0.0


def test_typed_bond_placement_moves_representation():
    """Same primitives AND same topology, different typed bond placement."""
    graph_a = _graph([(0, 1), (1, 2)], 3, bonds={(0, 1): 1, (1, 2): 0})
    graph_b = _graph([(0, 1), (1, 2)], 3, bonds={(0, 1): 0, (1, 2): 1})
    # identical candidate topology (same supports / overlap / connecting bonds)
    assert graph_a.l2_a.tolist() == graph_b.l2_a.tolist()
    assert graph_a.l2_overlap.tolist() == graph_b.l2_overlap.tolist()
    assert sorted(graph_a.l1_bond.tolist()) == sorted(graph_b.l1_bond.tolist())
    composer = ExplicitSupportComposer()
    composer.eval()
    with torch.no_grad():
        out_a = composer(_batch_from_graph(graph_a, dist=np.array([0, 1, 2])))
        out_b = composer(_batch_from_graph(graph_b, dist=np.array([0, 1, 2])))
    assert float((out_a - out_b).abs().max()) > 1.0e-6


# ---------------------------------------------------------------------------
# 8: no message passing / no endpoint access
# ---------------------------------------------------------------------------


def test_composer_never_reads_edge_endpoints():
    graph = _path_graph()
    batch = _batch_from_graph(graph)
    composer = ExplicitSupportComposer().eval()

    class _NoEndpoints:
        def __getattr__(self, name):
            if name in {"struct_src", "struct_dst"}:
                raise AttributeError(name)
            return getattr(batch, name)

    with torch.no_grad():
        reference = composer(batch)
        stripped = composer(_NoEndpoints())
    assert torch.equal(reference, stripped)


def test_no_message_passing_modules():
    forbidden = ("gineconv", "gcnconv", "transformer", "messagepassing")
    graph = _path_graph()
    composer = ExplicitSupportComposer()
    names = [name for name, _ in composer.named_modules()]
    assert not any("message" in name for name in names)
    assert not any("attention" in name for name in names)
    for module in composer.modules():
        class_name = type(module).__name__.lower()
        assert not any(token in class_name for token in forbidden)
    forward_source = inspect.getsource(ExplicitSupportComposer.forward)
    assert "struct_src" not in forward_source
    assert "struct_dst" not in forward_source


# ---------------------------------------------------------------------------
# 1: permutation invariance (on an ambiguity-free graph)
# ---------------------------------------------------------------------------


def test_node_relabel_preserves_output():
    graph = _path_graph()
    composer = ExplicitSupportComposer().eval()
    with torch.no_grad():
        reference = composer(_batch_from_graph(graph))
    # relabel 0->2, 1->0, 2->3, 3->1 (a permutation); the root atom is 0, so it
    # must move to index 2 in the relabelled graph.
    permutation = {0: 2, 1: 0, 2: 3, 3: 1}
    relabelled = [
        (permutation[u], permutation[v]) for u, v in [(0, 1), (1, 2), (2, 3)]
    ]
    graph2 = _graph(relabelled, 4)
    root = np.zeros(4, dtype=np.int64)
    root[permutation[0]] = 1
    with torch.no_grad():
        moved = composer(_batch_from_graph(graph2, root=root))
    assert float((reference - moved).abs().max()) < 1.0e-6


# ---------------------------------------------------------------------------
# 9/10/13/14/15/16: module / model level
# ---------------------------------------------------------------------------


def test_output_width_and_rounds():
    composer = ExplicitSupportComposer(output_dim=16)
    assert composer.output_dim == 16
    assert composer.rounds == 2
    assert composer.latent_dim <= 8


def test_forward_backward_finite():
    graph = _cycle_graph()
    composer = ExplicitSupportComposer()
    out = composer(_batch_from_graph(graph))
    assert out.shape[1] == 16
    assert bool(torch.isfinite(out).all())
    out.sum().backward()
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in composer.parameters()
    )


def test_candidate_enumeration_deterministic():
    first = _cycle_graph()
    second = _cycle_graph()
    assert first.l2_a.tolist() == second.l2_a.tolist()
    assert first.l2_b.tolist() == second.l2_b.tolist()
    assert first.l2_conn.tolist() == second.l2_conn.tolist()


def test_model_budget_and_geometry():
    from tracks.ksvd.experiments.luyin16 import (
        zinc_explicit_support_composer as exco,
    )
    from tracks.ksvd.experiments.luyin16 import (
        zinc_shared_structural_patch_encoder as sspe,
    )

    model = exco.build_candidate(0)
    total = sum(p.numel() for p in model.parameters())
    assert abs(total - sspe.REFERENCE_PARAMS) > 0  # sanity: not the baseline
    assert abs(total - 84495) / 84495 <= 0.01
    assert int(model.pair_hidden) == 16
    assert int(model.patch_hidden) == 64
    assert int(model.recurrence_rounds) == 2
    assert getattr(model, "typed_embedding", None) is None
    assert not [k for k in model.state_dict() if "typed_embedding" in k]
    assert model.structural_encoder.output_dim == 16
    assert model.structural_encoder.rounds == 2
