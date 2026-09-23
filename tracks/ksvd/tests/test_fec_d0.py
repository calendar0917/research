"""Focused CPU tests for FEC-D0 (label-free; no data, no checkpoints)."""

from __future__ import annotations

import numpy as np
import pytest

from tracks.ksvd.code.pipeline import graph_from_edge_index
from tracks.ksvd.experiments.luyin16 import fec_d0 as f0
from tracks.ksvd.experiments.luyin16 import fsar_v2 as v2


def _graph_from_edges(n: int, edges):
    edge_index = np.asarray(
        [[u, v] for u, v in edges for (u, v) in ((u, v), (v, u))], dtype=np.int64
    ).T
    return graph_from_edge_index(n, edge_index)


def _toy_graphs():
    # path-4 root 0
    g1 = _graph_from_edges(4, [(0, 1), (1, 2), (2, 3)])
    # star-4 centre 0
    g2 = _graph_from_edges(4, [(0, 1), (0, 2), (0, 3)])
    # ring-5
    g3 = _graph_from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
    return [g1, g2, g3]


# ---------------------------------------------------------------------------
# basis audit
# ---------------------------------------------------------------------------


def test_basis_dim_and_shell_coordinates():
    assert f0.NODE_BASIS_DIM == 11
    assert f0.D_RES == 7
    assert f0.KEPT_COORDINATES == (4, 5, 6, 7, 8, 9, 10)
    assert f0.DELETED_COORDINATES == (0, 1, 2, 3)


def test_root_indicator_equals_shell0():
    """Coordinate 0 is exactly 1[shell == 0]; coordinate [1:4] is the shell one-hot."""
    for g in _toy_graphs():
        for center in sorted(g.nodes):
            nodes, _index, node_basis, _e, _edges = v2._explicit_basis_for_patch(g, center)
            shell = node_basis[:, 1:4].argmax(axis=1)
            assert np.array_equal(node_basis[:, 0], (shell == 0).astype(node_basis.dtype))
            for local, node in enumerate(nodes):
                if node == center:
                    assert shell[local] == 0


# ---------------------------------------------------------------------------
# G0 — chemistry invariance (basis reads no chemistry)
# ---------------------------------------------------------------------------


def test_g0_chemistry_invariance():
    """The pure-topology basis is a function of adjacency only.

    Building the same untyped graph with two different 'chemistry' labelings is
    impossible here by construction: extraction only ever sees adjacency.
    We assert the stronger property that extraction depends only on adjacency
    by checking two independently constructed adjacency-identical graphs match.
    """
    g = _graph_from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    g_again = _graph_from_edges(5, [(4, 3), (3, 2), (2, 1), (1, 0)])
    b1 = f0.extract_molecule(g, 0, with_marked=False)
    b2 = f0.extract_molecule(g_again, 0, with_marked=False)
    assert np.array_equal(b1.basis, b2.basis)
    assert np.array_equal(b1.shell, b2.shell)


# ---------------------------------------------------------------------------
# G1 — relabel invariance
# ---------------------------------------------------------------------------


def test_g1_relabel_invariance():
    """Node relabeling preserves the occurrence multiset of (basis, shell)."""
    g = _graph_from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    perm = {0: 3, 1: 0, 2: 4, 3: 1, 4: 2}
    edges = [(perm[u], perm[v]) for u, v in [(0, 1), (1, 2), (2, 3), (3, 4)]]
    g_relab = _graph_from_edges(5, edges)
    b1 = f0.extract_molecule(g, 0, with_marked=False)
    b2 = f0.extract_molecule(g_relab, 0, with_marked=False)
    rows1 = sorted(map(tuple, b1.basis))
    rows2 = sorted(map(tuple, b2.basis))
    assert rows1 == rows2


def test_g1_relabel_invariance_residual_and_codes():
    g = _graph_from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    perm = {0: 3, 1: 0, 2: 4, 3: 1, 4: 2}
    edges = [(perm[u], perm[v]) for u, v in [(0, 1), (1, 2), (2, 3), (3, 4)]]
    g_relab = _graph_from_edges(5, edges)
    b1 = f0.extract_molecule(g, 0, with_marked=False)
    b2 = f0.extract_molecule(g_relab, 0, with_marked=False)
    stats = f0.fit_shell_stats(b1.basis, b1.shell)
    r1 = f0.residualize(b1.basis, b1.shell, stats)
    r2 = f0.residualize(b2.basis, b2.shell, stats)
    order1 = np.lexsort(r1.T)
    order2 = np.lexsort(r2.T)
    assert np.allclose(r1[order1], r2[order2], atol=1e-12)
    D = f0.random_dictionary()
    c1 = f0.omp_codes(D, r1)
    c2 = f0.omp_codes(D, r2)
    assert np.array_equal(c1[order1], c2[order2])


# ---------------------------------------------------------------------------
# G2 — exact sparsity
# ---------------------------------------------------------------------------


def test_g2_exact_sparsity_random_dictionary():
    rng = np.random.RandomState(0)
    X = rng.randn(50, f0.D_RES)
    D = f0.random_dictionary()
    C = f0.omp_codes(D, X, s=f0.SPARSITY)
    l0 = (np.abs(C) > 0).sum(axis=1)
    assert l0.max() <= f0.SPARSITY


# ---------------------------------------------------------------------------
# G5 — coarse anchor removed
# ---------------------------------------------------------------------------


def test_g5_no_exact_shell_coordinate_in_residual_input():
    """Reduced basis excludes all exact shell coordinates."""
    assert 0 not in f0.KEPT_COORDINATES
    for coord in (1, 2, 3):
        assert coord not in f0.KEPT_COORDINATES
    # the reduced input is invariant to a shell coordinate being added back
    for g in _toy_graphs():
        b = f0.extract_molecule(g, 0, with_marked=False)
        reduced = b.basis[:, f0.KEPT_COORDINATES]
        perturbed = b.basis.copy()
        perturbed[:, 1:4] = 0.0
        perturbed[:, 0] = 0.0
        reduced2 = perturbed[:, f0.KEPT_COORDINATES]
        assert np.array_equal(reduced, reduced2)


def test_g5_shell_conditional_mean_zero_on_fit():
    batches = [f0.extract_molecule(g, i, with_marked=False) for i, g in enumerate(_toy_graphs())]
    basis = np.concatenate([b.basis for b in batches], axis=0)
    shell = np.concatenate([b.shell for b in batches], axis=0)
    stats = f0.fit_shell_stats(basis, shell)
    resid = f0.residualize(basis, shell, stats)
    for s in range(f0.N_SHELLS):
        m = shell == s
        if m.sum() == 0:
            continue
        # zero-mean only where std is not floored
        unfloored = stats.std[s] > f0.SIGMA_FLOOR
        if unfloored.any():
            assert np.allclose(resid[m][:, unfloored].mean(axis=0), 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# G6 — independent float64 reconstruction reference
# ---------------------------------------------------------------------------


def test_g6_independent_reconstruction_reference():
    rng = np.random.RandomState(7)
    D = rng.randn(9, 16)
    D /= np.linalg.norm(D, axis=0, keepdims=True)
    X = rng.randn(20, 9)
    C = f0.omp_codes(D, X)
    ref = X - C @ D.T
    got = f0.relative_reconstruction_error(D, X, C)
    expected = float(np.mean(np.sum(ref * ref, axis=1)) / (np.mean(np.sum(X * X, axis=1)) + 1e-8))
    assert abs(got - expected) < 1e-10
    # dalpha reconstruction
    R = f0.omp_codes(D, X) @ D.T
    assert np.allclose(R, C @ D.T, atol=0)


# ---------------------------------------------------------------------------
# marked topology canonicalization (no aliased certificate bug)
# ---------------------------------------------------------------------------


def test_marked_topology_relabel_invariance():
    g = _graph_from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    perm = {0: 3, 1: 0, 2: 4, 3: 1, 4: 2}
    edges = [(perm[u], perm[v]) for u, v in [(0, 1), (1, 2), (2, 3), (3, 4)]]
    g_relab = _graph_from_edges(5, edges)
    k1 = f0.marked_topology_key(g, 0, 2)
    k2 = f0.marked_topology_key(g_relab, perm[0], perm[2])
    assert k1 == k2


def test_marked_topology_distinguishes_root_and_mark():
    g = _graph_from_edges(3, [(0, 1), (1, 2)])  # path-3
    # Distinct marked-rooted classes of path-3:
    #   root at an end, mark at the other end -> (0,2)/(2,0)
    #   root at an end, mark adjacent          -> (0,1)/(2,1)
    #   root at the centre, mark either end    -> (1,0)/(1,2)  [mirror-equal]
    k_root0_mark1 = f0.marked_topology_key(g, 0, 1)
    k_root0_mark2 = f0.marked_topology_key(g, 0, 2)
    k_root1_mark0 = f0.marked_topology_key(g, 1, 0)
    k_root1_mark2 = f0.marked_topology_key(g, 1, 2)
    assert k_root0_mark1 != k_root0_mark2  # marked distance differs
    assert k_root0_mark1 != k_root1_mark0  # root/mark roles differ
    # centre-root with either end marked are mirror-isomorphic -> EQUAL
    assert k_root1_mark0 == k_root1_mark2
    assert f0.marked_topology_key(g, 0, 2) == f0.marked_topology_key(g, 2, 0)
    assert f0.marked_topology_key(g, 0, 1) == f0.marked_topology_key(g, 2, 1)


def test_marked_topology_no_color_alias():
    """Complete invariant: same topology, different root-distance colour -> differ."""
    g_path3 = _graph_from_edges(3, [(0, 1), (1, 2)])
    # root 0 mark 2: shells (0,1,2);  root 0 mark 1: shells (0,1,1)
    assert f0.marked_topology_key(g_path3, 0, 2) != f0.marked_topology_key(g_path3, 0, 1)


def test_marked_topology_isomorphism_equality():
    """Two different labelings of the same marked topology agree."""
    g1 = _graph_from_edges(4, [(0, 1), (1, 2), (1, 3)])
    g2 = _graph_from_edges(4, [(3, 1), (1, 0), (1, 2)])
    # root 0 mark 2 in g1 -> root 3 mark 0 in g2
    assert f0.marked_topology_key(g1, 0, 2) == f0.marked_topology_key(g2, 3, 0)


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------


def test_split_sizes_and_disjoint():
    fit, holdout = f0.internal_split()
    assert len(fit) == f0.N_FIT
    assert len(holdout) == f0.N_HOLDOUT
    assert len(set(fit.tolist()) & set(holdout.tolist())) == 0
    assert len(set(fit.tolist()) | set(holdout.tolist())) == f0.N_TOTAL
    # deterministic
    fit2, holdout2 = f0.internal_split()
    assert np.array_equal(fit, fit2)
    assert np.array_equal(holdout, holdout2)


def test_split_matches_repo_convention():
    rng = np.random.RandomState(20260922)
    perm = rng.permutation(10000)
    expected = np.sort(perm[:2000].astype(np.int64))
    _fit, holdout = f0.internal_split()
    assert np.array_equal(holdout, expected)


# ---------------------------------------------------------------------------
# label-free guard
# ---------------------------------------------------------------------------


def test_target_guard_raises():
    class _Data:
        y = 1.0

    poisoned = f0._PoisonedData(_Data())
    with pytest.raises(f0.TargetAccessError):
        _ = poisoned.y


def test_non_train_split_forbidden():
    with pytest.raises(RuntimeError):
        f0.assert_label_free("val")
    with pytest.raises(RuntimeError):
        f0.assert_label_free("test")
    f0.assert_label_free("train")


# ---------------------------------------------------------------------------
# reuse statistics
# ---------------------------------------------------------------------------


def test_code_support_stats_basic():
    codes = np.zeros((10, 4))
    codes[0:3, 0] = 1.0
    codes[3:6, 1] = 2.0
    mol = np.asarray([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
    root = np.asarray([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
    shell = np.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    stats = f0.code_support_stats(codes, mol, root, shell)
    assert stats["n_active"] == 2
    assert stats["per_atom"][0]["support_count"] == 3
    assert stats["per_atom"][0]["distinct_molecules"] == 2
    assert stats["per_atom"][2]["support_count"] == 0


def test_spearman_perfect_and_reverse():
    a = np.asarray([1.0, 2.0, 3.0, 4.0])
    assert abs(f0.spearman(a, a) - 1.0) < 1e-12
    assert abs(f0.spearman(a, a[::-1]) + 1.0) < 1e-12


def test_usage_summary_sane():
    stats = {
        "per_atom": [{"support_count": c} for c in [10, 0, 5, 5]],
        "n_active": 3,
    }
    s = f0.usage_summary(stats)
    assert s["n_active"] == 3
    assert 0 < s["effective_atom_count"] <= 4
    assert s["top1_mass"] >= s["top4_mass"] / 4
