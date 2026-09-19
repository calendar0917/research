"""Tests for the whole-graph canonicalization / registration analysis.

These are data-free (no ZINC, no checkpoints, no results artifacts) so they are
fresh-clone safe.  They lock the two correctness invariants the analysis relies
on:

* exact canonicalization of the attributed graph is permutation invariant;
* the soft doubly-stochastic registration transform has the right direction and
  recovers a permutation on a permuted copy (the Section-11 control).
"""

from __future__ import annotations

import numpy as np
import pytest

from tracks.ksvd.code.graph import from_edges
from tracks.ksvd.code.run_wholegraph_canonical_registration_audit import (
    Codebook,
    alignment_loss_numpy,
    canonical_atom_order,
    fit_registration,
)


def _cycle_attributed():
    # 6-cycle with a pendant leaf; node types 0/1, bond types 1/2.
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 6)]
    graph = from_edges(7, edges)
    node_types = np.array([0, 1, 0, 1, 0, 0, 1], dtype=np.int64)
    edge_types = {}
    for i, (a, b) in enumerate(edges):
        edge_types[(min(a, b), max(a, b))] = 1 + (i % 2)
    return graph, node_types, edge_types


def _relabel(graph, node_types, edge_types, perm):
    inv = np.empty(graph.n, dtype=np.int64)
    inv[np.asarray(perm)] = np.arange(graph.n)
    edges = []
    for u in graph.nodes:
        for v in graph.neighbors(u):
            a, b = int(inv[u]), int(inv[v])
            if a < b:
                edges.append((a, b))
    new_graph = from_edges(graph.n, edges)
    new_nt = np.zeros_like(node_types)
    for u in graph.nodes:
        new_nt[int(inv[u])] = node_types[u]
    new_et = {}
    for (a, b), bond in edge_types.items():
        na, nb = int(inv[a]), int(inv[b])
        new_et[(min(na, nb), max(na, nb))] = bond
    return new_graph, new_nt, new_et


def test_canonical_order_is_permutation_invariant():
    graph, nt, et = _cycle_attributed()
    codebook = Codebook(sorted(set(nt.tolist())), sorted(set(et.values())))
    base = codebook.encode(graph, nt, et, q=8)
    rng = np.random.default_rng(0)
    for _ in range(25):
        ng, nnt, net = _relabel(graph, nt, et, rng.permutation(graph.n))
        other = codebook.encode(ng, nnt, net, q=8)
        assert np.array_equal(base["node"], other["node"])
        assert np.array_equal(base["relation"], other["relation"])
        assert base["key"] == other["key"]


def test_canonical_key_distinguishes_non_isomorphic():
    graph, nt, et = _cycle_attributed()
    # same 7 atoms, same node/bond colours, but a 7-cycle (no pendant leaf)
    cycle_edges = [(i, (i + 1) % 7) for i in range(7)]
    other = from_edges(7, cycle_edges)
    other_nt = np.array([0, 1, 0, 1, 0, 0, 1], dtype=np.int64)
    other_et = {(min(a, b), max(a, b)): (1 + (i % 2)) for i, (a, b) in enumerate(cycle_edges)}
    key_a = canonical_atom_order(graph, nt, et)[1]
    key_b = canonical_atom_order(other, other_nt, other_et)[1]
    assert key_a != key_b


def test_alignment_loss_identity_zero_for_identical():
    graph, nt, et = _cycle_attributed()
    codebook = Codebook(sorted(set(nt.tolist())), sorted(set(et.values())))
    code = codebook.encode(graph, nt, et, q=8)
    node, edge, total = alignment_loss_numpy(code, code)
    assert node == 0.0 and edge == 0.0 and total == 0.0


def test_registration_recovers_permutation():
    pytest.importorskip("torch")
    graph, nt, et = _cycle_attributed()
    codebook = Codebook(sorted(set(nt.tolist())), sorted(set(et.values())))
    q = 8
    code_g = codebook.encode(graph, nt, et, q=q)
    ng, nnt, net = _relabel(graph, nt, et, np.random.default_rng(3).permutation(graph.n))
    code_h = codebook.encode(ng, nnt, net, q=q)
    # H is a relabelled copy, but canonicalization makes it identical to G.
    node, edge, total = alignment_loss_numpy(code_g, code_h)
    assert total == 0.0
    free = fit_registration(code_g, code_h, rho=0.0, lam=0.0, steps=150,
                            sinkhorn_iters=30, lr=0.3, init_scale=4.0, seed=0)
    assert free["d"] < 1e-3


def test_uncanonicalized_permuted_copy_has_nonzero_identity_loss():
    graph, nt, et = _cycle_attributed()
    codebook = Codebook(sorted(set(nt.tolist())), sorted(set(et.values())))
    q = 8
    from tracks.ksvd.code.run_wholegraph_canonical_registration_audit import _encode_in_order

    base = _encode_in_order(graph, nt, et, q, codebook)
    ng, nnt, net = _relabel(graph, nt, et, np.array([3, 0, 5, 1, 6, 2, 4]))
    other = _encode_in_order(ng, nnt, net, q, codebook)
    _, _, total = alignment_loss_numpy(base, other)
    assert total > 0.0
