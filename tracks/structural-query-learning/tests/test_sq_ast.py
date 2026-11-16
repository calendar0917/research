"""Query-language semantics: the hand-bound 12-cycle witness and BRANCH distinctness.

The witness graphs are pure graphs; passing these tests is an implementation
guarantee, not chemical evidence.
"""

from __future__ import annotations

import numpy as np
import pytest

import sq_ast as A
from sq_ast import PackedGraph

ATOM_A = 6
ATOM_B = 7


def cycle_packed(types_a: str, types_b: str) -> PackedGraph:
    """Two disjoint 12-cycles with the given A/B patterns (all bonds one type)."""
    assert len(types_a) == 12 and len(types_b) == 12
    atom_feat = []
    bond_u = []
    bond_v = []
    bond_feat = []
    node_graph = []
    n_bonds = 0
    for graph, pattern in enumerate((types_a, types_b)):
        offset = len(atom_feat)
        for symbol in pattern:
            row = np.zeros(9, dtype=np.int16)
            row[0] = ATOM_A if symbol == "A" else ATOM_B
            atom_feat.append(row)
            node_graph.append(graph)
        for i in range(12):
            u, v = offset + i, offset + (i + 1) % 12
            bond_u.append(min(u, v))
            bond_v.append(max(u, v))
            bond_feat.append(np.asarray([0, 0, 1], dtype=np.int16))
            n_bonds += 1
    n_nodes = len(atom_feat)
    degree = np.full(n_nodes, 2, dtype=np.int16)
    # all nodes are on a 12-cycle
    ring = np.full(n_nodes, 9, dtype=np.int16)
    return PackedGraph(
        dataset="witness",
        split="train",
        n_graphs=2,
        n_nodes=n_nodes,
        node_graph=np.asarray(node_graph, dtype=np.int32),
        atom_feat=np.stack(atom_feat),
        degree=degree,
        min_ring=ring,
        in_cycle=np.ones(n_nodes, dtype=bool),
        n_bonds=n_bonds,
        bond_u=np.asarray(bond_u, dtype=np.int32),
        bond_v=np.asarray(bond_v, dtype=np.int32),
        bond_feat=np.stack(bond_feat),
        graph_n_atoms=np.asarray([12, 12], dtype=np.int32),
        graph_n_bonds=np.asarray([12, 12], dtype=np.int32),
        graph_n_components=np.asarray([1, 1], dtype=np.int32),
        graph_cycle_rank=np.asarray([1, 1], dtype=np.int32),
        row_ids=np.asarray([0, 1], dtype=np.int64),
        labels=np.asarray([0.0, 0.0], dtype=np.float32),
        smiles=["c1" * 0 + "witness0", "witness1"],
    )


@pytest.fixture(scope="module")
def witness():
    return cycle_packed("ABAABBAAABBB", "ABBAABAAABBB")


def counts(packed: PackedGraph, ast: tuple) -> np.ndarray:
    return packed.graph_counts(A.evaluate(packed, ast))


def test_base_counts_are_identical(witness):
    """A/B counts, degree, min-ring and the basic one-hop type conditions agree."""
    assert counts(witness, A.ATOM_EQ(0, ATOM_A)).tolist() == [6, 6]
    assert counts(witness, A.ATOM_EQ(0, ATOM_B)).tolist() == [6, 6]
    assert counts(witness, A.DEG_EQ(2)).tolist() == [12, 12]
    assert counts(witness, A.MIN_RING_GE(9)).tolist() == [12, 12]
    assert counts(witness, A.IN_CYCLE()).tolist() == [12, 12]
    assert counts(witness, A.ACYCLIC()).tolist() == [0, 0]
    for atom_type in (ATOM_A, ATOM_B):
        for m in (1, 2):
            assert counts(witness, A.NBR(A.ATOM_EQ(0, atom_type), A.ANY_BOND_PRED(), m)).tolist() == [
                10 if m == 1 else 2,
                10 if m == 1 else 2,
            ]
    # both graphs have exactly one atom whose two neighbours are both A and one
    # whose two neighbours are both B
    assert counts(witness, A.NBR(A.ATOM_EQ(0, ATOM_A), A.ANY_BOND_PRED(), 2)).tolist() == [2, 2]
    assert counts(witness, A.NBR(A.ATOM_EQ(0, ATOM_B), A.ANY_BOND_PRED(), 2)).tolist() == [2, 2]


def test_branch_witness_separates_the_two_cycles(witness):
    t = A.AND(A.ATOM_EQ(0, ATOM_B), A.NBR(A.ATOM_EQ(0, ATOM_A), A.ANY_BOND_PRED(), 2))
    q = A.AND(A.ATOM_EQ(0, ATOM_A), A.BRANCH(A.ATOM_EQ(0, ATOM_B), t, A.ANY_BOND_PRED(), A.ANY_BOND_PRED()))
    assert counts(witness, t).tolist() == [1, 1]
    assert counts(witness, q).tolist() == [1, 0]
    assert A.ast_nodes(q) == 8
    assert A.validates(q)


def test_branch_requires_two_distinct_neighbours():
    """A single neighbour satisfying both sides must NOT satisfy BRANCH."""
    # 3 atoms: 0 - 1 - 2, atom 1 has one neighbour of each kind is fine, but a
    # triangle-free chain where only one neighbour satisfies both sides fails.
    atom_feat = np.zeros((3, 9), dtype=np.int16)
    atom_feat[0, 0] = ATOM_B
    atom_feat[1, 0] = ATOM_A
    atom_feat[2, 0] = ATOM_A
    packed = PackedGraph(
        dataset="chain",
        split="train",
        n_graphs=1,
        n_nodes=3,
        node_graph=np.zeros(3, dtype=np.int32),
        atom_feat=atom_feat,
        degree=np.asarray([1, 2, 1], dtype=np.int16),
        min_ring=np.zeros(3, dtype=np.int16),
        in_cycle=np.zeros(3, dtype=bool),
        n_bonds=2,
        bond_u=np.asarray([0, 1], dtype=np.int32),
        bond_v=np.asarray([1, 2], dtype=np.int32),
        bond_feat=np.zeros((2, 3), dtype=np.int16),
        graph_n_atoms=np.asarray([3], dtype=np.int32),
        graph_n_bonds=np.asarray([2], dtype=np.int32),
        graph_n_components=np.asarray([1], dtype=np.int32),
        graph_cycle_rank=np.asarray([0], dtype=np.int32),
        row_ids=np.asarray([0], dtype=np.int64),
        labels=np.asarray([0.0], dtype=np.float32),
        smiles=["CCC"],
    )
    # centre atom 1: neighbours are 0 (B) and 2 (A); both branches require an A
    # neighbour -> only node 2 qualifies on both sides, so BRANCH must be False.
    q = A.BRANCH(A.ATOM_EQ(0, ATOM_A), A.ATOM_EQ(0, ATOM_A), A.ANY_BOND_PRED(), A.ANY_BOND_PRED())
    assert A.evaluate(packed, q).tolist() == [False, False, False]
    # with two distinct A neighbours it becomes True (centre of an A-B-A path)
    atom_feat2 = atom_feat.copy()
    atom_feat2[0, 0] = ATOM_A
    packed2 = PackedGraph(**{**packed.__dict__, "atom_feat": atom_feat2})
    assert A.evaluate(packed2, q).tolist() == [False, True, False]
    # witness is a pair of distinct neighbours
    w = A.witness(packed2, q, 1)
    assert w["branch"]["u_neighbor"] != w["branch"]["w_neighbor"]


def test_nbr_is_a_counting_modality_not_a_path():
    """NBR may revisit nodes: a triangle satisfies NBR(x, ANY, 2) for every node."""
    atom_feat = np.zeros((3, 9), dtype=np.int16)
    atom_feat[:, 0] = ATOM_A
    packed = PackedGraph(
        dataset="triangle",
        split="train",
        n_graphs=1,
        n_nodes=3,
        node_graph=np.zeros(3, dtype=np.int32),
        atom_feat=atom_feat,
        degree=np.asarray([2, 2, 2], dtype=np.int16),
        min_ring=np.asarray([3, 3, 3], dtype=np.int16),
        in_cycle=np.ones(3, dtype=bool),
        n_bonds=3,
        bond_u=np.asarray([0, 0, 1], dtype=np.int32),
        bond_v=np.asarray([1, 2, 2], dtype=np.int32),
        bond_feat=np.zeros((3, 3), dtype=np.int16),
        graph_n_atoms=np.asarray([3], dtype=np.int32),
        graph_n_bonds=np.asarray([3], dtype=np.int32),
        graph_n_components=np.asarray([1], dtype=np.int32),
        graph_cycle_rank=np.asarray([1], dtype=np.int32),
        row_ids=np.asarray([0], dtype=np.int64),
        labels=np.asarray([0.0], dtype=np.float32),
        smiles=["C1CC1"],
    )
    assert A.evaluate(packed, A.NBR(A.ATOM_EQ(0, ATOM_A), A.ANY_BOND_PRED(), 2)).all()
    assert A.evaluate(packed, A.NBR(A.ATOM_EQ(0, ATOM_A), A.ANY_BOND_PRED(), 3)).sum() == 0
    assert A.evaluate(packed, A.MIN_RING_EQ(3)).all()


def test_validation_limits():
    deep = A.TRUE()
    for _ in range(4):
        deep = A.NBR(deep, A.ANY_BOND_PRED(), 1)
    assert A.ast_hops(deep) == 4
    assert not A.validates(deep)

    wide = A.AND(*[A.ATOM_EQ(0, k) for k in range(6)])  # 1 + 6 nodes > 9? (7) but arity 6
    assert not A.validates(wide)  # AND arity is limited to 2..3 children

    ok = A.AND(A.ATOM_EQ(0, 6), A.ATOM_EQ(0, 7), A.ATOM_EQ(0, 8))
    assert A.validates(ok)
    assert A.ast_nodes(ok) == 4
    # canonicalisation dedups and sorts
    assert A.canon(A.AND(A.ATOM_EQ(0, 7), A.ATOM_EQ(0, 6))) == A.canon(
        A.AND(A.ATOM_EQ(0, 6), A.ATOM_EQ(0, 7))
    )
    assert A.AND(A.ATOM_EQ(0, 6), A.ATOM_EQ(0, 6)) == A.ATOM_EQ(0, 6)


def test_permutation_invariance(witness):
    """Every query count is invariant under an atom relabelling."""
    rng = np.random.default_rng(0)
    queries = [
        A.ATOM_EQ(0, ATOM_A),
        A.ATOM_EQ(0, ATOM_B),
        A.NBR(A.ATOM_EQ(0, ATOM_A), A.ANY_BOND_PRED(), 2),
        A.NBR(A.NBR(A.ATOM_EQ(0, ATOM_B), A.ANY_BOND_PRED(), 1), A.ANY_BOND_PRED(), 1),
        A.BRANCH(A.ATOM_EQ(0, ATOM_A), A.ATOM_EQ(0, ATOM_B), A.ANY_BOND_PRED(), A.ANY_BOND_PRED()),
        A.AND(A.ATOM_EQ(0, ATOM_B), A.NBR(A.ATOM_EQ(0, ATOM_A), A.ANY_BOND_PRED(), 2)),
    ]
    base = [witness.graph_counts(A.evaluate(witness, q)).tolist() for q in queries]
    for _ in range(5):
        perm = rng.permutation(witness.n_nodes)
        inverse = np.argsort(perm)
        atom_feat = witness.atom_feat[perm]
        node_graph = witness.node_graph[perm]
        degree = witness.degree[perm]
        ring = witness.min_ring[perm]
        in_cycle = witness.in_cycle[perm]
        bu = np.minimum(inverse[witness.bond_u], inverse[witness.bond_v])
        bv = np.maximum(inverse[witness.bond_u], inverse[witness.bond_v])
        order = np.lexsort((bv, bu))
        relabelled = PackedGraph(
            dataset=witness.dataset,
            split=witness.split,
            n_graphs=witness.n_graphs,
            n_nodes=witness.n_nodes,
            node_graph=node_graph,
            atom_feat=atom_feat,
            degree=degree,
            min_ring=ring,
            in_cycle=in_cycle,
            n_bonds=witness.n_bonds,
            bond_u=bu[order].astype(np.int32),
            bond_v=bv[order].astype(np.int32),
            bond_feat=witness.bond_feat[order],
            graph_n_atoms=witness.graph_n_atoms,
            graph_n_bonds=witness.graph_n_bonds,
            graph_n_components=witness.graph_n_components,
            graph_cycle_rank=witness.graph_cycle_rank,
            row_ids=witness.row_ids,
            labels=witness.labels,
            smiles=witness.smiles,
        )
        after = [relabelled.graph_counts(A.evaluate(relabelled, q)).tolist() for q in queries]
        assert after == base
