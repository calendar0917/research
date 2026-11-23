from __future__ import annotations


import numpy as np
import pytest

import bbpa_structure as S
import bbpa_synthetic as Y


def test_bridges_match_removal_reference():
    rng = np.random.default_rng(0)
    for n in range(2, 12):
        for _ in range(20):
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if rng.random() < 0.25:
                        edges.append((u, v))
            bu = np.asarray([e[0] for e in edges], np.int64)
            bv = np.asarray([e[1] for e in edges], np.int64)
            got = S.find_bridges(n, bu, bv)
            # reference: an edge is a bridge iff its removal increases components
            def ncomp(bu2, bv2):
                dsu = S.DSU(n)
                for u, v in zip(bu2.tolist(), bv2.tolist()):
                    dsu.union(u, v)
                return len({dsu.find(i) for i in range(n)})
            base = ncomp(bu, bv)
            for e in range(len(edges)):
                keep = np.ones(len(edges), bool)
                keep[e] = False
                ref = ncomp(bu[keep], bv[keep]) > base
                assert bool(got[e]) == bool(ref), (n, edges, e)
    # explicit triangle + tail
    bu = np.asarray([0, 1, 2, 2, 3], np.int64)
    bv = np.asarray([1, 2, 0, 3, 4], np.int64)
    got = S.find_bridges(5, bu, bv)
    assert got.tolist() == [False, False, False, True, True]


def test_decomposition_cases():
    cases = {
        "benzene": (Y.benzene(), 1, 0),
        "ethane": (Y.ethane(), 2, 1),
        "chain4": (Y.chain4(), 4, 3),
        "star4": (Y.star4(), 4, 3),
        "ringchain4": (Y.concatenated_chain(4), 4, 3),
    }
    for name, (g, n_blocks, n_bridges) in cases.items():
        d = S.decompose(g["n_atoms"], g["bu"], g["bv"])
        info = S.validate(d)
        assert info["n_blocks"] == n_blocks, name
        assert info["n_bridges"] == n_bridges, name
        assert info["n_components"] == 1, name


def test_two_component_forest():
    bu = np.asarray([0, 2, 3, 4, 5, 6, 7], np.int64)
    bv = np.asarray([1, 3, 4, 5, 6, 7, 2], np.int64)
    d = S.decompose(8, bu, bv)
    info = S.validate(d)
    assert info["n_components"] == 2
    assert info["n_bridges"] == 1
    assert info["n_blocks"] == 3


def test_bridge_sides_complementary():
    g = Y.concatenated_chain(4)
    d = S.decompose(g["n_atoms"], g["bu"], g["bv"])
    for br in d.bridges:
        keep = np.ones(d.n_bonds, bool)
        keep[br.bond] = False
        a = S.bfs_dist(d.n_atoms, d.bu[keep], d.bv[keep], br.atom_u)
        b = S.bfs_dist(d.n_atoms, d.bu[keep], d.bv[keep], br.atom_v)
        sa = set(np.where(a >= 0)[0].tolist())
        sb = set(np.where(b >= 0)[0].tolist())
        assert not (sa & sb)
        assert sa | sb == set(range(d.n_atoms))


def test_port_distances_and_ring_positions():
    expected = {1: 1, 2: 2, 3: 3}
    for pos, dist in expected.items():
        g = Y.disubstituted_benzene(pos)
        d = S.decompose(g["n_atoms"], g["bu"], g["bv"])
        ring = [b for b in d.blocks if len(b.atoms) == 6][0]
        dm = d.block_port_dist[ring.block_id]
        assert int(dm[0, 1]) == dist


def test_schedule_is_topological_and_root_free_value():
    g = Y.concatenated_chain(4)
    d1 = S.decompose(g["n_atoms"], g["bu"], g["bv"])
    # reverse the atom numbering; the block-tree root flips, values must not
    n = g["n_atoms"]
    inv = np.arange(n - 1, -1, -1)
    d2 = S.decompose(n, inv[g["bu"]], inv[g["bv"]])
    for d in (d1, d2):
        for e, deps in enumerate(d.edge_deps):
            for dep in deps:
                assert d.edge_step[dep["dep_edge"]] < d.edge_step[e]
    assert len(d1.bridges) == len(d2.bridges) == 3
    # port distance matrices are permutation-invariant as multisets
    assert sorted(d1.block_port_dist[0].ravel().tolist()) == \
        sorted(d2.block_port_dist[0].ravel().tolist())


def test_single_atom_block_port_distance_zero():
    g = Y.star4()
    d = S.decompose(g["n_atoms"], g["bu"], g["bv"])
    for b in d.blocks:
        if len(b.atoms) == 1 and len(b.ports) > 1:
            dm = d.block_port_dist[b.block_id]
            assert np.all(dm == 0)


def test_block_size_and_bridge_counts_on_real_subset():
    import bbpa_common as C
    import bbpa_data as D

    if not C.RAW_CACHE.is_file():
        pytest.skip("raw cache unavailable")
    b = D.load_bundle()
    rows = np.union1d(b.fit_rows[:12], b.eval_rows[:4])
    decs = S.build_all(b.raw, rows)
    for d in decs:
        S.validate(d)
        assert len(d.bridges) == len(d.blocks) - len(d.components)


def test_root_choice_changes_only_schedule():
    g = Y.concatenated_chain(4)
    d1 = S.decompose(g["n_atoms"], g["bu"], g["bv"])
    d2 = S.reschedule(S.decompose(g["n_atoms"], g["bu"], g["bv"]), "last")
    assert d1.edge_step.tolist() != d2.edge_step.tolist()
    assert sorted(d1.edge_step.tolist()) == sorted(d2.edge_step.tolist())
    for e, deps in enumerate(d2.edge_deps):
        for dep in deps:
            assert d2.edge_step[dep["dep_edge"]] < d2.edge_step[e]
