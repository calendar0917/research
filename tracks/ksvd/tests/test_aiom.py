"""Data-free tests for the AIOM representation audit.

These lock the correctness invariants the audit depends on and need no ZINC
data, checkpoints or results artifacts (fresh-clone safe):

* the AIOM moments match their definition;
* ``Phi_T`` is exactly permutation invariant;
* the two interventions preserve the invariants they claim to preserve;
* the collision statistic detects a planted non-isomorphic collision.
"""

from __future__ import annotations

import numpy as np

from tracks.ksvd.code.run_aiom_representation_audit import (
    Mol,
    aiom_moments,
    collision_statistics,
    incidence_matrix,
    permute_mol,
    phi_from_moments,
    perturb_bond_type_swap,
    two_switch,
    upper_tri_indices,
)


def _mol(n, edges, node_types, bond_types):
    return Mol(
        n=n,
        bonds=[(min(a, b), max(a, b)) for a, b in edges],
        bond_types=np.asarray(bond_types, dtype=np.int64),
        node_types=np.asarray(node_types, dtype=np.int64),
    )


def _codebook(mol):
    atoms = sorted({int(c) for c in mol.node_types.tolist()})
    bonds = sorted({int(c) for c in mol.bond_types.tolist()})
    return atoms, bonds, {c: i for i, c in enumerate(atoms)}, {c: i for i, c in enumerate(bonds)}


def _phi(mol):
    atoms, bonds, ni, bi = _codebook(mol)
    triu = upper_tri_indices(len(atoms) + len(bonds))
    moments = aiom_moments(mol, len(atoms), len(bonds), ni, bi, T_max=8)
    return phi_from_moments(moments, triu), moments


def _cycle_mol():
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 6)]
    nt = [0, 1, 0, 1, 0, 0, 1]
    bt = [1 + (i % 2) for i in range(len(edges))]
    return _mol(7, edges, nt, bt)


def test_incidence_matrix_is_bipartite_symmetric():
    mol = _cycle_mol()
    atoms, bonds, ni, bi = _codebook(mol)
    A, X = incidence_matrix(mol, len(atoms), len(bonds), ni, bi)
    n, m = mol.n, mol.m
    assert A.shape == (n + m, n + m)
    assert np.array_equal(A, A.T)
    # atom--atom and bond--bond blocks are zero
    assert A[:n, :n].sum() == 0
    assert A[n:, n:].sum() == 0
    # every bond has exactly two endpoints
    assert np.all(A[n:].sum(axis=1) == 2)
    # X rows: atoms use the atom block, bonds use the bond block
    C_V = len(atoms)
    assert np.all(X[:n, :C_V].sum(axis=1) == 1) and np.all(X[:n, C_V:].sum() == 0)
    assert np.all(X[n:, C_V:].sum(axis=1) == 1) and np.all(X[n:, :C_V].sum() == 0)


def test_moment_zero_is_attribute_counts():
    mol = _cycle_mol()
    atoms, bonds, ni, bi = _codebook(mol)
    moments = aiom_moments(mol, len(atoms), len(bonds), ni, bi, T_max=2)
    M0 = moments[0]
    # atom block counts atom categories
    for c in range(len(atoms)):
        assert M0[c, c] == int((mol.node_types == atoms[c]).sum())
    # bond block counts bond categories
    for j, c in enumerate(bonds):
        assert M0[len(atoms) + j, len(atoms) + j] == int((mol.bond_types == c).sum())
    # M1 counts atom-type x bond-type incidences (odd hops)
    M1 = moments[1]
    assert M1.shape == M0.shape


def test_phi_is_permutation_invariant():
    mol = _cycle_mol()
    base, _ = _phi(mol)
    rng = np.random.default_rng(0)
    for _ in range(20):
        pa = rng.permutation(mol.n)
        pb = rng.permutation(mol.m)
        pm = permute_mol(mol, pa, pb)
        other, _ = _phi(pm)
        assert np.max(np.abs(other - base)) < 1e-12


def _degree_sequence(mol):
    deg = np.zeros(mol.n, dtype=np.int64)
    for a, b in mol.bonds:
        deg[a] += 1
        deg[b] += 1
    return deg


def _bond_multiset(mol):
    return sorted(int(x) for x in mol.bond_types.tolist())


def _is_simple(mol):
    seen = set()
    for a, b in mol.bonds:
        if a == b or (a, b) in seen:
            return False
        seen.add((a, b))
    return True


def test_two_switch_preserves_structural_invariants():
    mol = _cycle_mol()
    found = False
    for seed in range(50):
        out = two_switch(np.random.default_rng(seed), mol)
        if out is None:
            continue
        found = True
        assert _is_simple(out)
        assert np.array_equal(_degree_sequence(out), _degree_sequence(mol))
        assert _bond_multiset(out) == _bond_multiset(mol)
        assert np.array_equal(out.node_types, mol.node_types)
        break
    assert found, "expected at least one successful two-switch"


def test_two_switch_none_on_star():
    mol = _mol(5, [(0, 1), (0, 2), (0, 3), (0, 4)], [0] * 5, [1, 1, 1, 1])
    assert all(two_switch(np.random.default_rng(s), mol) is None for s in range(20))


def test_bond_type_swap_preserves_multiset():
    mol = _cycle_mol()
    out = None
    for seed in range(50):
        out = perturb_bond_type_swap(np.random.default_rng(seed), mol)
        if out is not None:
            break
    assert out is not None
    assert _bond_multiset(out) == _bond_multiset(mol)
    assert np.array_equal(out.node_types, mol.node_types)
    # topology unchanged
    assert sorted(out.bonds) == sorted(mol.bonds)


def test_collision_statistics_detects_planted_noniso_collision():
    # 3 rows: rows 0/1 identical features but different iso classes -> collision
    phi = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]])
    iso = np.array([0, 1, 2], dtype=np.int64)
    st = collision_statistics(phi, iso, decimals=10)
    assert st["n_collision_groups"] == 1
    assert st["n_graphs_in_collision"] == 2
    assert st["n_noniso_collision_pairs"] == 1


def test_collision_statistics_ignores_isomorphic_duplicates():
    phi = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]])
    iso = np.array([0, 0, 2], dtype=np.int64)
    st = collision_statistics(phi, iso, decimals=10)
    assert st["n_collision_groups"] == 0
    assert st["n_duplicate_only_groups"] == 1
    assert st["n_graphs_in_collision"] == 0
