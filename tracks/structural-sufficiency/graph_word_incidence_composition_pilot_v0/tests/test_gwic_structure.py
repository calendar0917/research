from __future__ import annotations

import numpy as np
import pytest

import gwic_data as D
import gwic_structure as S
import gwic_synthetic as SYN


def _bfs(n, adj, src):
    dist = np.full(n, -1, np.int64)
    dist[src] = 0
    q = [src]
    while q:
        x = q.pop(0)
        for y in adj[x]:
            if dist[y] < 0:
                dist[y] = dist[x] + 1
                q.append(y)
    return dist


def test_atom_bond_attributes_match_raw(bundle, rows, structs):
    node_ptr = np.asarray(bundle.raw["node_ptr"], np.int64)
    bond_ptr = np.asarray(bundle.raw["bond_ptr"], np.int64)
    for row in rows[:5]:
        st = structs[int(row)]
        a0, a1 = int(node_ptr[row]), int(node_ptr[row + 1])
        b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
        assert np.array_equal(st.atom_feat, np.asarray(bundle.raw["atom_feat"])[a0:a1])
        assert np.array_equal(st.bond_feat, np.asarray(bundle.raw["bond_feat"])[b0:b1])
        assert np.array_equal(st.bond_u, np.asarray(bundle.raw["bond_u"])[b0:b1])
        assert np.array_equal(st.bond_v, np.asarray(bundle.raw["bond_v"])[b0:b1])
        assert st.n_atoms == a1 - a0 and st.n_bonds == b1 - b0


def test_every_candidate_is_a_connected_induced_support(structs, rows):
    for row in rows[:4]:
        st = structs[int(row)]
        n = st.n_atoms
        adj = [[] for _ in range(n)]
        for e in range(st.n_bonds):
            u, v = int(st.bond_u[e]), int(st.bond_v[e])
            adj[u].append(v)
            adj[v].append(u)
        for w in range(st.n_words):
            rec = S.word_reference(st, w)
            support = rec["support"]
            root = rec["root"]
            assert 2 <= len(support) <= 4
            assert root in support
            assert len(set(support)) == len(support)
            allowed = set(support)
            dist = _bfs(n, adj, root)
            assert all(dist[x] >= 0 for x in support)
            # every internal recorded bond is real and induced
            for e in rec["internal_bonds"]:
                assert int(st.bond_u[e]) in allowed and int(st.bond_v[e]) in allowed
            expected_internal = [e for e in range(st.n_bonds)
                                 if int(st.bond_u[e]) in allowed and int(st.bond_v[e]) in allowed]
            assert rec["internal_bonds"] == sorted(expected_internal)
            # induced subgraph connectivity by BFS inside S
            seen = {root}
            stack = [root]
            while stack:
                x = stack.pop()
                for y in adj[x]:
                    if y in allowed and y not in seen:
                        seen.add(y); stack.append(y)
            assert seen == allowed


def test_boundary_bonds_are_exact(structs, rows):
    for row in rows[:4]:
        st = structs[int(row)]
        n = st.n_atoms
        adj = [[] for _ in range(n)]
        for e in range(st.n_bonds):
            adj[int(st.bond_u[e])].append(int(st.bond_v[e]))
            adj[int(st.bond_v[e])].append(int(st.bond_u[e]))
        for w in range(st.n_words):
            rec = S.word_reference(st, w)
            root = rec["root"]
            dist = _bfs(n, adj, root)
            r2 = {i for i in range(n) if 0 <= dist[i] <= 2}
            Sset = set(rec["support"])
            expected = sorted(e for e in range(st.n_bonds)
                              if (int(st.bond_u[e]) in Sset) != (int(st.bond_v[e]) in Sset)
                              and int(st.bond_u[e]) in r2 and int(st.bond_v[e]) in r2)
            assert rec["boundary_bonds"] == expected
            # no boundary edge is ever counted as internal
            assert not (set(rec["boundary_bonds"]) & set(rec["internal_bonds"]))


def test_synthetic_same_counts_different_connection():
    path = SYN.make_view("butane_path", 4, [(0, 1, 1), (1, 2, 1), (2, 3, 1)])
    star = SYN.make_view("isobutane_star", 4, [(0, 1, 1), (0, 2, 1), (0, 3, 1)])
    s_path = S.build_molecule_struct(path)
    s_star = S.build_molecule_struct(star)
    assert s_path.n_atoms == s_star.n_atoms == 4
    assert s_path.n_bonds == s_star.n_bonds == 3
    def supports(st):
        out = set()
        for w in range(st.n_words):
            out.add(tuple(sorted(int(x) for x in st.word_support[w].tolist() if x >= 0)))
        return out
    assert supports(s_path) != supports(s_star)
    # uniform-alpha coverage is also different: this is a real incidence contrast
    def coverage(st):
        N = st.n_atoms
        C = np.zeros((N, N))
        for w in range(st.n_words):
            root = int(st.word_root[w])
            sup = [int(x) for x in st.word_support[w].tolist() if x >= 0]
            for t in sup:
                C[root, t] += 1.0 / max(1, int((st.word_root == root).sum()))
        return C
    assert not np.allclose(coverage(s_path), coverage(s_star))
