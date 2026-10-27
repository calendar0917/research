"""Fast CPU tests for the frozen-transfer audit primitives.

These tests do not load any model or label; they exercise the deterministic
split, the structure fingerprints, the block masks and the statistics helpers.
They are safe for the fresh-clone CI allowlist (numpy/torch only, no data).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fta_common as K


def _toy_cache():
    """Two molecules: a 3-atom chain and a 4-ring, as the ad cache layout."""
    atom_types = [0, 1, 1, 2, 2, 2, 2]
    bond_types = [1, 2, 1, 1, 1, 1]
    atom_ptr = [0, 3, 7]
    bond_ptr = [0, 2, 6]
    bond_index = [[0, 1], [1, 2], [0, 1], [1, 2], [2, 3], [3, 0]]
    return {
        "atom_type": np.asarray(atom_types, dtype=np.int64),
        "bond_type": np.asarray(bond_types, dtype=np.int64),
        "atom_ptr": np.asarray(atom_ptr, dtype=np.int64),
        "bond_ptr": np.asarray(bond_ptr, dtype=np.int64),
        "bond_index": np.asarray(bond_index, dtype=np.int64),
    }


def test_molecule_graphs_shapes():
    cache = _toy_cache()
    graphs = K.molecule_graphs(cache)
    assert [len(g[0]) for g in graphs] == [3, 4]
    assert [len(g[1]) for g in graphs] == [2, 4]
    assert graphs[1][1][-1] == (3, 0, 1)


def test_fingerprint_bond_order_and_label_content():
    """The split key is a *labelled-graph* hash: edge order must not matter, but
    the atom-type assignment does.  (It is not an isomorphism invariant, and the
    split rule does not need one: it only orders Fit4k rows deterministically
    without reading y.)"""
    cache = _toy_cache()
    atoms, bonds = K.molecule_graphs(cache, [1])[0]
    assert K.structure_fingerprint((atoms, bonds)) == K.structure_fingerprint(
        (atoms, list(reversed(bonds))))
    other = ([atoms[0], atoms[1], atoms[2], 9], bonds)
    assert K.structure_fingerprint((atoms, bonds)) != K.structure_fingerprint(other)
    # documented non-property: two isomorphic node numbering can give a different
    # key (the function normalises the *edge set*, it is not a canonical labeling).
    chain = K.molecule_graphs(cache, [0])[0]
    chain_renumbered = ([chain[0][2], chain[0][1], chain[0][0]],
                        [bonds for bonds in [(2, 1, chain[1][0][2]),
                                             (1, 0, chain[1][1][2])]])
    assert K.structure_fingerprint(chain) != K.structure_fingerprint(chain_renumbered)


def test_radius2_signature_and_support_tiers():
    cache = _toy_cache()
    graphs = K.molecule_graphs(cache)
    sigs = K.radius2_signatures(graphs[1])
    assert len(sigs) == 4
    # a 4-ring of identical atoms has one orbit, so all signatures coincide
    assert len(set(sigs)) == 1
    freq = K.support_frequency([graphs[1]])
    tier = K.support_tiers([graphs[1]], freq)
    assert tier["unseen_frac"].tolist() == [0.0]
    assert tier["low4_frac"].tolist() == [1.0]   # support == 1 molecule
    assert tier["supported_frac"].tolist() == [0.0]
    # the chain's atoms are unseen w.r.t. the ring fit set
    tier2 = K.support_tiers([graphs[0]], freq)
    assert tier2["unseen_frac"].tolist() == [1.0]


def test_ring_stats():
    cache = _toy_cache()
    chain, ring = K.molecule_graphs(cache)
    s_chain = K.ring_stats(chain)
    assert s_chain["n_rings"] == 0 and s_chain["max_ring_size"] == 0
    assert s_chain["cycle_excess"] == 0
    s_ring = K.ring_stats(ring)
    assert s_ring["n_rings"] >= 1 and s_ring["max_ring_size"] >= 4
    assert s_ring["cycle_excess"] == 0


def test_row_hash_is_order_invariant():
    a = [7, 3, 9]
    assert K.row_hash(a) == K.row_hash([9, 3, 7])
    assert K.row_hash(a) != K.row_hash([7, 3, 8])
    assert len(K.row_hash(a)) == 64


def test_block_layout_and_masks():
    total = 0
    for name, (a, b) in K.BLOCKS.items():
        assert 0 <= a < b <= 513
        total += b - a
    assert total == 513
    assert int(K.block_mask("atom_global").sum()) == 225
    assert int(K.block_mask("local_global").sum()) == 225
    assert int(K.block_mask("full").sum()) == 513
    m1 = K.block_mask("atom_global")
    m2 = K.block_mask("local_global")
    # the two blocks share only the global33 channel
    assert int((m1 & m2).sum()) == 33


def test_linear_cka_properties():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 9))
    Y = X @ rng.normal(size=(9, 4)) + 0.05 * rng.normal(size=(300, 4))
    assert K.linear_cka(X, X) == pytest.approx(1.0, abs=1e-10)
    assert K.linear_cka(X, Y) == pytest.approx(K.linear_cka(Y, X), abs=1e-12)
    assert 0.0 <= K.linear_cka(X, rng.normal(size=X.shape)) <= 1.0
    # isotropic scaling must not change CKA
    assert K.linear_cka(X, Y) == pytest.approx(K.linear_cka(X, 37.0 * Y), abs=1e-10)


def test_participation_ratio_and_rank():
    # exact case: two orthogonal zero-mean directions with equal norm -> PR = 2
    k = np.arange(100, dtype=np.float64)
    Z = np.zeros((100, 5))
    Z[:, 1] = np.cos(2 * np.pi * k / 100.0)
    Z[:, 2] = np.sin(2 * np.pi * k / 100.0)
    sv = np.linalg.svd(Z - Z.mean(0), compute_uv=False)
    assert K.participation_ratio(sv) == pytest.approx(2.0, abs=1e-10)
    spec = K.spectral_summary(Z)
    assert spec["numerical_rank_tol_1e-06"] == 2
    assert spec["rank_99pct_energy"] == 2
    # empirical case: a centred isotropic rank-3 cloud has PR close to but not
    # exactly 3 (eigenvalue spread); the helper must track that.
    rng = np.random.default_rng(1)
    X = rng.normal(size=(1000, 3))
    assert 2.8 < K.participation_ratio(np.linalg.svd(X - X.mean(0), compute_uv=False)) <= 3.0


def test_standardizer_roundtrip():
    rng = np.random.default_rng(2)
    z = rng.normal(scale=3.0, size=(50, 513)).astype(np.float32)
    mu, sd = K.standardizer(z)
    zs = K.apply_standardizer(z, mu, sd)
    assert np.allclose(zs.mean(0), 0.0, atol=1e-5)
    assert np.allclose(zs.std(0), 1.0, atol=1e-4)
    const = np.zeros((10, 513), dtype=np.float32)
    mu, sd = K.standardizer(const)
    assert np.all(sd == 1e-6)
    assert np.all(K.apply_standardizer(const, mu, sd) == 0.0)


def test_label_permutation_shape():
    rng = np.random.default_rng(20261028)
    y = rng.normal(size=4000)
    perm = rng.permutation(len(y))
    yp = y[perm]
    assert np.array_equal(np.sort(yp), np.sort(y))
    assert not np.array_equal(yp, y)
