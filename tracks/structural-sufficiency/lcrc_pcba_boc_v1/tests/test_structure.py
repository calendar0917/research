"""Independent structural checks for BOC objects and ring systems."""

from __future__ import annotations

import numpy as np

import boc_structures as S


def _edges(*pairs):
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def test_chain_counts_and_no_ring():
    # 0-1-2-3 chain: L={0}, R={2} for the 1-2 bond, no pairs, no rings.
    rec = S.enumerate_molecule(4, _edges((0, 1), (1, 2), (2, 3)))
    counts = rec["counts"]
    np.testing.assert_array_equal(counts[0], [0, 1, 0, 0, 0])
    np.testing.assert_array_equal(counts[1], [1, 1, 0, 0, 1])
    np.testing.assert_array_equal(counts[2], [1, 0, 0, 0, 0])
    assert len(rec["rings"]) == 0
    assert all(bool(x) for x in rec["bridges"])


def test_triangle_keeps_both_neighbour_roles_and_skips_diagonal_cross():
    rec = S.enumerate_molecule(3, _edges((0, 1), (1, 2), (0, 2)))
    counts = rec["counts"]
    # every bond has one L and one R; that single atom is in both sides
    for row in counts:
        np.testing.assert_array_equal(row, [1, 1, 0, 0, 0])
    # the triangle closure must be present as is_bond in each neighbour entry
    for info in rec["neighbours"]:
        assert info["nL"] == 1 and info["nR"] == 1
        for _atom, _side, _bond_side, bond_other in info["entries"]:
            assert bond_other >= 0
    # a triangle has one ring system of 3 atoms and 3 pairs with distance 1
    assert len(rec["rings"]) == 1
    ring = rec["rings"][0]
    assert ring["atoms"] == [0, 1, 2]
    assert ring["n_edges"] == 3
    assert [p[2] for p in ring["pairs"]] == [0, 0, 0]
    assert all(p[3] >= 0 for p in ring["pairs"])


def test_four_cycle_chord_and_fork():
    # square 0-1-2-3-0 plus chord 0-2: ring pairs include direct and distance 1
    rec = S.enumerate_molecule(4, _edges((0, 1), (1, 2), (2, 3), (3, 0), (0, 2)))
    assert len(rec["rings"]) == 1
    ring = rec["rings"][0]
    assert ring["n_edges"] == 5
    dist = {(i, j): d for i, j, d, _ in ring["pairs"]}
    assert dist[(0, 1)] == 0 and dist[(0, 2)] == 0 and dist[(0, 3)] == 0
    assert dist[(1, 2)] == 0 and dist[(2, 3)] == 0
    assert dist[(1, 3)] == 1  # distance 2 -> bucket index 1
    # no bridges in this graph


def test_bridges_match_edge_removal_reference_on_random_graphs():
    rng = np.random.default_rng(7)
    for n in range(2, 10):
        for _ in range(20):
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if rng.random() < 0.28:
                        edges.append((u, v))
            if not edges:
                continue
            e = np.asarray(edges, dtype=np.int64)
            got = S.find_bridges(n, e)
            ref = S.reference_bridges_by_removal(n, e)
            np.testing.assert_array_equal(got, ref)


def test_benzene_ortho_meta_para_distance_buckets():
    # benzene 0-1-2-3-4-5-0; ortho=1 (bucket 0), meta=2 (bucket 1), para=3 (bucket 2)
    rec = S.enumerate_molecule(
        6, _edges((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0))
    )
    assert len(rec["rings"]) == 1
    ring = rec["rings"][0]
    dist = {(i, j): d for i, j, d, _ in ring["pairs"]}
    assert dist[(0, 1)] == 0  # ortho (direct)
    assert dist[(0, 2)] == 1  # meta
    assert dist[(0, 3)] == 2  # para
    assert dist[(1, 3)] == 1
    assert dist[(1, 4)] == 2


def test_fused_rings_share_one_nonbridge_component():
    # naphthalene skeleton: two fused six-rings sharing edge 4-5
    edges = _edges(
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
        (4, 6), (6, 7), (7, 8), (8, 9), (9, 5),
    )
    rec = S.enumerate_molecule(10, edges)
    assert len(rec["rings"]) == 1
    ring = rec["rings"][0]
    assert len(ring["atoms"]) == 10
    assert ring["n_edges"] == 11
    assert ring["n_external"] == 0


def test_bridge_connected_ring_systems_stay_separate():
    # cyclopropane 0-1-2 and cyclopropane 3-4-5 joined by bridge 2-3
    rec = S.enumerate_molecule(6, _edges((0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)))
    assert len(rec["rings"]) == 2
    assert rec["rings"][0]["atoms"] == [0, 1, 2]
    assert rec["rings"][1]["atoms"] == [3, 4, 5]
    ext = [row for system in rec["rings"] for row in system["external_entries"]]
    assert [(a, b) for a, b, _e in ext] == [(2, 3), (3, 2)]


def test_pair_enumeration_agrees_with_brute_force():
    rng = np.random.default_rng(11)
    for n in range(2, 12):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < 0.25:
                    edges.append((u, v))
        if not edges:
            continue
        e = np.asarray(edges, dtype=np.int64)
        rec = S.enumerate_molecule(n, e)
        adj = [set() for _ in range(n)]
        for u, v in e.tolist():
            adj[u].add(v)
            adj[v].add(u)
        for eid, (u, v) in enumerate(e.tolist()):
            left = sorted(adj[u] - {v})
            right = sorted(adj[v] - {u})
            expected = []
            for i in range(len(left)):
                for j in range(i + 1, len(left)):
                    expected.append((i, j, 0, -1 if left[j] not in adj[left[i]] else 1))
            for i in range(len(right)):
                for j in range(i + 1, len(right)):
                    expected.append((len(left) + i, len(left) + j, 1, -1 if right[j] not in adj[right[i]] else 1))
            for i, w in enumerate(left):
                for j, z in enumerate(right):
                    if w != z:
                        expected.append((i, len(left) + j, 2, -1 if z not in adj[w] else 1))
            got = rec["pairs"][eid]
            assert len(got) == len(expected)
            for (a1, b1, k1, e1), (a2, b2, k2, e2) in zip(got, expected):
                assert (a1, b1, k1) == (a2, b2, k2)
                assert (e1 >= 0) == (e2 >= 0)


def test_cache_roundtrip_on_real_rows(tmp_path):
    import boc_common as C

    rows = list(range(0, 48))
    out = tmp_path / "struct"
    meta = S.build_cache(out, rows=rows, reference_fraction=1.0, log_every=0)
    assert meta["labels_read"] is False
    assert meta["reference_audit"]["mismatches"] == 0
    cache = S.load_cache(out)
    task = C.BC.SD.load_task()
    batch = S.collate_boc(task, cache, rows[:8])
    assert int(batch["boc_n_bonds"]) == batch["bond_index"].shape[0]
    # each bond has exactly one centre object and a non-empty neighbour block
    assert int(batch["boc_nb_ptr"][-1]) == int(batch["boc_n_nb"])
    assert int(batch["boc_pair_ptr"][-1]) == int(batch["boc_n_pairs"])
    assert int(batch["boc_nL"].sum() + batch["boc_nR"].sum()) == int(batch["boc_n_nb"])
    assert int(batch["boc_nLL"].sum() + batch["boc_nRR"].sum() + batch["boc_nLR"].sum()) == int(batch["boc_n_pairs"])
    # every ring pair points into this batch's ring atom range
    if int(batch["boc_n_ring_pairs"]):
        assert int(batch["boc_ring_pair_i"].max()) < int(batch["boc_n_ring_atoms"])
        assert int(batch["boc_ring_pair_j"].max()) < int(batch["boc_n_ring_atoms"])
    # no cross-molecule object: every neighbour of a bond is in that bond's molecule
    atom_batch = batch["atom_batch"].numpy()
    nb_atom = batch["boc_nb_atom"].numpy()
    bond_batch = batch["bond_batch"].numpy()
    nb_ptr = batch["boc_nb_ptr"].numpy()
    mol_atoms = {b: set(np.nonzero(atom_batch == b)[0].tolist()) for b in range(int(batch["n_graphs"]))}
    for bond in range(int(batch["boc_n_bonds"])):
        lo, hi = nb_ptr[bond], nb_ptr[bond + 1]
        assert set(nb_atom[lo:hi].tolist()) <= mol_atoms[int(bond_batch[bond])]
