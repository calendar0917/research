"""Structural correctness of the explicit three-hop enumeration.

These tests are torch-free and use hand-built molecules, so they pin the
definition itself: every path is a *simple* bond walk of exactly three bonds
over four distinct atoms, with the ordered bond types and the
``dist(root, end) in {1, 2, 3}`` class that the model consumes.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import s3_paths as P


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def ring(n: int, bond_type: int = 1, atom_type: int = 6) -> dict:
    edges = np.asarray([[i, (i + 1) % n] for i in range(n)], dtype=np.int64)
    types = np.full(n, bond_type, dtype=np.int64)
    adj = P.molecule_adjacency(n, edges[:, 0], edges[:, 1], types)
    atoms = np.full(n, atom_type, dtype=np.int64)
    dist = P.all_pairs_shortest_paths(n, adj)
    return {"n": n, "adj": adj, "atom_type": atoms, "dist": dist, "edges": edges, "bond_type": types}


def chain(n: int, bond_type: int = 1, atom_type: int = 6) -> dict:
    edges = np.asarray([[i, i + 1] for i in range(n - 1)], dtype=np.int64)
    types = np.full(n - 1, bond_type, dtype=np.int64)
    adj = P.molecule_adjacency(n, edges[:, 0], edges[:, 1], types)
    atoms = np.full(n, atom_type, dtype=np.int64)
    dist = P.all_pairs_shortest_paths(n, adj)
    return {"n": n, "adj": adj, "atom_type": atoms, "dist": dist, "edges": edges, "bond_type": types}


def random_molecule(rng: np.random.Generator, *, n_max: int = 9, p: float = 0.35) -> dict:
    n = int(rng.integers(1, n_max + 1))
    pairs = list(itertools.combinations(range(n), 2))
    keep = [pair for pair in pairs if rng.random() < p]
    if keep:
        edges = np.asarray(keep, dtype=np.int64)
        types = rng.integers(1, 6, size=len(keep)).astype(np.int64)
    else:
        edges = np.zeros((0, 2), dtype=np.int64)
        types = np.zeros(0, dtype=np.int64)
    adj = P.molecule_adjacency(n, edges[:, 0], edges[:, 1], types)
    atoms = rng.integers(0, 16, size=n).astype(np.int64)
    dist = P.all_pairs_shortest_paths(n, adj)
    return {"n": n, "adj": adj, "atom_type": atoms, "dist": dist, "edges": edges, "bond_type": types}


def enumerate_molecule(mol: dict) -> P.S3Paths:
    return P.enumerate_s3(mol["n"], mol["adj"], mol["atom_type"], mol["dist"])


# ---------------------------------------------------------------------------
# hand-checked cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "builder,n,expected_paths,expected_dist",
    [
        (chain, 4, 2, {3: 2}),          # butane: 1-2-3-4 and 4-3-2-1
        (chain, 6, 6, {3: 6}),          # hexane: one three-bond walk per root
        (ring, 4, 8, {1: 8}),           # cyclobutane: closure at distance 1
        (ring, 5, 10, {2: 10}),         # cyclopentane: closure at distance 2
        (ring, 6, 12, {3: 12}),         # cyclohexane: closure at distance 3
    ],
)
def test_hand_checked_counts(builder, n, expected_paths, expected_dist):
    mol = builder(n)
    paths = enumerate_molecule(mol)
    assert paths.n_paths == expected_paths
    values, counts = np.unique(paths.dist, return_counts=True)
    assert {int(v): int(c) for v, c in zip(values, counts)} == expected_dist


def test_benzene_is_all_distance_three():
    mol = ring(6)
    paths = enumerate_molecule(mol)
    # every (root, direction) pair: 6 roots x 2 directions, all closing at dist 3
    assert paths.n_paths == 12
    assert set(paths.dist.tolist()) == {3}
    for root in range(6):
        records = paths.records(root)
        assert len(records) == 2
        # both directions end on the para atom, i.e. the ring closes at distance 3
        assert [r["end_atom"] for r in records] == [(root + 3) % 6, (root - 3) % 6]
        assert all(r["shortest_distance"] == 3 for r in records)


def test_chain_endpoints_and_counts_per_root():
    mol = chain(6)
    paths = enumerate_molecule(mol)
    per_root = paths.per_root_counts()
    # every root supports exactly one simple three-bond walk on a hexane chain
    assert per_root.tolist() == [1, 1, 1, 1, 1, 1]
    assert int(per_root.sum()) == paths.n_paths == 6
    assert all(int(d) == 3 for d in paths.dist)


def test_no_paths_in_separated_components():
    mol = chain(3)
    assert enumerate_molecule(mol).n_paths == 0
    single = random_molecule(np.random.default_rng(0), n_max=1)
    assert enumerate_molecule(single).n_paths == 0


# ---------------------------------------------------------------------------
# definition-level invariants on random molecules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_every_path_is_simple_ordered_and_typed(seed):
    rng = np.random.default_rng(seed)
    for _ in range(20):
        mol = random_molecule(rng)
        paths = enumerate_molecule(mol)
        nbrs = [dict(adj) for adj in mol["adj"]]
        for root in range(mol["n"]):
            for rec in paths.records(root):
                u, v, x = rec["mid_atom_1"], rec["mid_atom_2"], rec["end_atom"]
                # four distinct atoms, exactly three bonds, root != end
                assert len({root, u, v, x}) == 4
                # the walk is real and the ordered bond types match the graph
                assert nbrs[root][u] == rec["bond_type_1"]
                assert nbrs[u][v] == rec["bond_type_2"]
                assert nbrs[v][x] == rec["bond_type_3"]
                # distance class is the true BFS distance and is in {1, 2, 3}
                assert rec["shortest_distance"] == int(mol["dist"][root, x])
                assert rec["shortest_distance"] in (1, 2, 3)
                # distance 1 means the ring closes: x is a neighbour of root
                assert (x in nbrs[root]) == (rec["shortest_distance"] == 1)
                # mid-atom types come from the atom vocabulary, not from ids
                assert rec["mid_type_1"] == int(mol["atom_type"][u])
                assert rec["mid_type_2"] == int(mol["atom_type"][v])


def test_paths_are_unique_per_root():
    rng = np.random.default_rng(17)
    for _ in range(20):
        paths = enumerate_molecule(random_molecule(rng))
        for root in range(int(paths.ptr.size) - 1):
            seen = {
                (r["mid_atom_1"], r["mid_atom_2"], r["end_atom"]) for r in paths.records(root)
            }
            assert len(seen) == len(paths.records(root))


def test_closed_form_matches_enumeration():
    rng = np.random.default_rng(29)
    for _ in range(40):
        mol = random_molecule(rng, n_max=12)
        paths = enumerate_molecule(mol)
        expected = P.count_s3_closed_form(mol["n"], mol["adj"])
        assert np.array_equal(paths.per_root_counts(), expected)


def test_reference_enumerator_matches():
    rng = np.random.default_rng(31)
    for _ in range(10):
        mol = random_molecule(rng, n_max=6, p=0.6)
        fast = enumerate_molecule(mol)
        slow = P.reference_enumerate_s3(mol["n"], mol["adj"], mol["atom_type"])
        assert fast.n_paths == sum(len(rs) for rs in slow)
        key_fast = sorted(
            (r, rec["mid_atom_1"], rec["mid_atom_2"], rec["end_atom"], rec["bond_type_1"], rec["bond_type_2"], rec["bond_type_3"])
            for r in range(mol["n"])
            for rec in fast.records(r)
        )
        key_slow = sorted(
            (rec["root_atom"], rec["mid_atom_1"], rec["mid_atom_2"], rec["end_atom"], rec["bond_type_1"], rec["bond_type_2"], rec["bond_type_3"])
            for rs in slow
            for rec in rs
        )
        assert key_fast == key_slow


def test_bond_edit_changes_the_encoding():
    mol = ring(6)
    base = enumerate_molecule(mol)
    # a different bond type changes the ordered type triples, not the topology
    edited = ring(6, bond_type=2)
    other = enumerate_molecule(edited)
    assert other.n_paths == base.n_paths
    assert sorted(P.path_type_keys(base).tolist()) != sorted(P.path_type_keys(other).tolist())
    # removing one bond changes the topology and the path set
    edges = mol["edges"][:-1]
    adj = P.molecule_adjacency(mol["n"], edges[:, 0], edges[:, 1], mol["bond_type"][:-1])
    broken = P.enumerate_s3(mol["n"], adj, mol["atom_type"], P.all_pairs_shortest_paths(mol["n"], adj))
    assert broken.n_paths != base.n_paths


def test_path_type_roundtrip():
    rng = np.random.default_rng(41)
    paths = enumerate_molecule(random_molecule(rng, n_max=12))
    keys = P.path_type_keys(paths)
    assert keys.shape == (paths.n_paths,)
    decoded = [P.decode_path_type(int(k)) for k in np.unique(keys)]
    assert all(d["shortest_distance"] in (1, 2, 3) for d in decoded)
    assert all(1 <= d["bond_type_1"] <= 5 for d in decoded)
    assert all(0 <= d["mid_type_1"] <= 15 for d in decoded)


# ---------------------------------------------------------------------------
# two-ball vs three-ball distinguishability
# ---------------------------------------------------------------------------


def test_cyclohexane_and_hexane_differ_only_at_three_hops():
    cyclo = ring(6)
    hexa = chain(6)
    ball2_cyclo = P.rooted_ball(cyclo["n"], cyclo["adj"], cyclo["atom_type"], 0, radius=2)
    ball3_cyclo = P.rooted_ball(cyclo["n"], cyclo["adj"], cyclo["atom_type"], 0, radius=3)
    ball2_hexa = P.rooted_ball(hexa["n"], hexa["adj"], hexa["atom_type"], 2, radius=2)
    ball3_hexa = P.rooted_ball(hexa["n"], hexa["adj"], hexa["atom_type"], 2, radius=3)
    # the two-hop neighbourhoods are identical: both are a path of four atoms
    assert P.ball_isomorphic(ball2_cyclo, ball2_hexa)
    assert P.ball_signature(ball2_cyclo) == P.ball_signature(ball2_hexa)
    # the three-hop neighbourhoods are not: only the ring closes back on the root
    assert not P.ball_isomorphic(ball3_cyclo, ball3_hexa)
    assert P.ball_signature(ball3_cyclo) != P.ball_signature(ball3_hexa)
    # and the S3 path sets differ accordingly (12 vs 6, closure at distance 1/2 vs a chain)
    assert enumerate_molecule(cyclo).n_paths == 12
    assert enumerate_molecule(hexa).n_paths == 6


def test_ball_isomorphism_respects_atom_types_and_bond_types():
    mol = ring(6)
    ball = P.rooted_ball(mol["n"], mol["adj"], mol["atom_type"], 0, radius=3)
    # relabelling atoms is an isomorphism (same rooted ball)
    perm = np.asarray([3, 4, 5, 0, 1, 2], dtype=np.int64)
    remap = np.empty_like(perm)
    remap[perm] = np.arange(perm.size)
    edges = remap[mol["edges"]]
    adj = P.molecule_adjacency(mol["n"], edges[:, 0], edges[:, 1], mol["bond_type"])
    ball_perm = P.rooted_ball(mol["n"], adj, mol["atom_type"], int(remap[0]), radius=3)
    assert P.ball_isomorphic(ball, ball_perm)
    assert P.ball_signature(ball) == P.ball_signature(ball_perm)
    # an atom-type edit at radius 1 is not isomorphic
    atoms = mol["atom_type"].copy()
    atoms[1] = (int(atoms[1]) + 1) % 16
    ball_edit = P.rooted_ball(mol["n"], mol["adj"], atoms, 0, radius=3)
    assert not P.ball_isomorphic(ball, ball_edit)


def test_ball_signature_is_a_screen_not_a_proof():
    """Equal signature implies isomorphic, never the other way round."""
    rng = np.random.default_rng(59)
    balls = []
    for _ in range(15):
        mol = random_molecule(rng, n_max=8)
        for root in range(mol["n"]):
            balls.append(P.rooted_ball(mol["n"], mol["adj"], mol["atom_type"], root, radius=2))
    by_signature: dict[str, list[P.RootedBall]] = {}
    for ball in balls:
        by_signature.setdefault(P.ball_signature(ball), []).append(ball)
    for group in by_signature.values():
        for a, b in itertools.combinations(group[:4], 2):
            assert P.ball_isomorphic(a, b)
