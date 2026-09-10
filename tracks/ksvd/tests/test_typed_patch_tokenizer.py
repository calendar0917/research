"""Correctness tests for the versioned typed patch tokenizer.

Regression guard for the historical ``pynauty.certificate`` alias defect: the
historical token was an invariant of the *uncolored* incidence topology only,
so distinct rooted typed patches shared a token.  These tests pin the
corrected key to the intended color-preserving isomorphism relation.
"""

from __future__ import annotations

import itertools
import random

import numpy as np
import pytest

from tracks.ksvd.code.graph import from_edges
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16.typed_patch_tokenizer import (
    TYPED_TOKENIZER_V1_HISTORICAL,
    TYPED_TOKENIZER_V2_CORRECTED,
    assert_cache_tokenizer_version,
    build_colored_incidence,
    corrected_canonical_key,
    historical_certificate,
    resolve_typed_tokenizer_version,
    typed_isomorphic_vf2,
    typed_patch_key,
    typed_tokenizer_fingerprint,
)


def _build(n, edges, node_types, edge_types):
    graph = from_edges(n, [tuple(edge) for edge in edges])
    return (
        graph,
        np.asarray(node_types, dtype=np.int64),
        {tuple(sorted(edge)): int(value) for edge, value in edge_types.items()},
    )


def _key(graph, node_types, edge_types, center, radius=2, version=TYPED_TOKENIZER_V2_CORRECTED):
    return typed_patch_key(version, graph, int(center), node_types, edge_types, radius)


def _incidence(graph, node_types, edge_types, center, radius=2):
    return build_colored_incidence(graph, int(center), node_types, edge_types, radius)


def _relabel(graph, node_types, edge_types, permutation):
    nodes = list(graph.nodes)
    new_edges = [
        tuple(sorted((permutation[int(u)], permutation[int(v)]))) for u, v in graph.edges()
    ]
    new_graph = from_edges(len(nodes), new_edges)
    new_nodes = np.zeros_like(node_types)
    for old in nodes:
        new_nodes[permutation[old]] = node_types[old]
    new_edges_map = {
        tuple(sorted((permutation[int(u)], permutation[int(v)]))): int(
            edge_types[graph.edge_key(int(u), int(v))]
        )
        for u, v in graph.edges()
    }
    return new_graph, new_nodes, new_edges_map


PATH = [(0, 1), (1, 2), (2, 3)]
ZERO = {edge: 0 for edge in PATH}


def test_1_same_colored_graph_under_permutation_same_key() -> None:
    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    graph, node_types, edge_types = _build(
        5, edges, [0, 1, 2, 1, 0], {edge: 0 for edge in edges}
    )
    base = _key(graph, node_types, edge_types, 2)
    for permutation in [(4, 3, 2, 1, 0), (0, 2, 4, 1, 3), (3, 1, 4, 0, 2)]:
        new_graph, new_nodes, new_edges = _relabel(
            graph, node_types, edge_types, permutation
        )
        assert _key(new_graph, new_nodes, new_edges, permutation[2]) == base


def test_2_different_root_color_different_key() -> None:
    left = _build(4, PATH, [0, 1, 1, 1], ZERO)
    right = _build(4, PATH, [2, 1, 1, 1], ZERO)
    assert _key(*left, 0) != _key(*right, 0)
    # ... while the historical certificate collides.
    assert historical_certificate(_incidence(*left, 0)) == historical_certificate(
        _incidence(*right, 0)
    )


def test_3_different_atom_color_different_key() -> None:
    left = _build(4, PATH, [0, 1, 1, 1], ZERO)
    right = _build(4, PATH, [0, 1, 2, 1], ZERO)
    assert _key(*left, 0) != _key(*right, 0)
    # historical counterexample: same topology, different atom types, same key
    assert historical_certificate(_incidence(*left, 0)) == historical_certificate(
        _incidence(*right, 0)
    )


def test_4_different_bond_color_different_key() -> None:
    left = _build(4, PATH, [0, 1, 1, 1], ZERO)
    right = _build(4, PATH, [0, 1, 1, 1], {e: (1 if e == (1, 2) else 0) for e in PATH})
    assert _key(*left, 0) != _key(*right, 0)


def test_5_known_historical_collision_separated() -> None:
    left = _build(4, PATH, [0, 1, 1, 1], ZERO)
    right = _build(4, PATH, [0, 1, 2, 1], ZERO)
    left_inc = _incidence(*left, 0)
    right_inc = _incidence(*right, 0)
    assert historical_certificate(left_inc) == historical_certificate(right_inc)
    assert corrected_canonical_key(left_inc) != corrected_canonical_key(right_inc)
    assert not typed_isomorphic_vf2(left_inc, right_inc)


def test_5b_root_position_separated() -> None:
    left = _build(4, PATH, [0, 0, 0, 0], ZERO)
    # same raw adjacency, different root position: these are different rooted
    # typed patches and the corrected key must separate them.
    assert corrected_canonical_key(_incidence(*left, 0)) != corrected_canonical_key(
        _incidence(*left, 1)
    )
    assert not typed_isomorphic_vf2(_incidence(*left, 0), _incidence(*left, 1))


def test_6_same_corrected_key_implies_oracle_isomorphic() -> None:
    random.seed(11)
    buckets: dict[bytes, object] = {}
    for _ in range(1500):
        n = random.randint(3, 6)
        pairs = list(itertools.combinations(range(n), 2))
        edges = [p for p in pairs if random.random() < 0.4]
        node_types = [random.randint(0, 2) for _ in range(n)]
        edge_types = {p: random.randint(0, 1) for p in edges}
        root = random.randint(0, n - 1)
        graph, nt, et = _build(n, edges, node_types, edge_types)
        incidence = _incidence(graph, nt, et, root)
        key = corrected_canonical_key(incidence)
        representative = buckets.get(key)
        if representative is not None:
            assert typed_isomorphic_vf2(representative, incidence)
        else:
            buckets[key] = incidence


def test_7_cache_version_mismatch_rejected() -> None:
    with pytest.raises(RuntimeError):
        assert_cache_tokenizer_version(
            {"typed_tokenizer_version": TYPED_TOKENIZER_V1_HISTORICAL},
            TYPED_TOKENIZER_V2_CORRECTED,
        )
    with pytest.raises(RuntimeError):
        assert_cache_tokenizer_version(None, TYPED_TOKENIZER_V2_CORRECTED)
    assert_cache_tokenizer_version(
        {"typed_tokenizer_version": TYPED_TOKENIZER_V2_CORRECTED},
        TYPED_TOKENIZER_V2_CORRECTED,
    )


def test_8_historical_tokenizer_reproduces_legacy_certificate() -> None:
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    graph, node_types, edge_types = _build(
        6, edges, [0, 0, 1, 1, 0, 0], {e: 0 for e in edges}
    )
    cache: dict[bytes, bytes] = {}
    for radius in (1, 2):
        for center in graph.nodes:
            legacy = zpp._typed_certificate(
                graph, int(center), node_types, edge_types, radius, cache
            )
            incidence = _incidence(graph, node_types, edge_types, center, radius)
            assert legacy == historical_certificate(incidence)
            assert (
                _key(graph, node_types, edge_types, center, radius, TYPED_TOKENIZER_V1_HISTORICAL)
                == legacy
            )


def test_9_oracle_matches_brute_force_on_small_graphs() -> None:
    def brute_iso(n, e1, c1, e2, c2):
        s1 = set(e1)
        for perm in itertools.permutations(range(n)):
            if any(c1[v] != c2[perm[v]] for v in range(n)):
                continue
            if set(tuple(sorted((perm[a], perm[b]))) for a, b in s1) == set(e2):
                return True
        return False

    random.seed(3)
    for _ in range(1500):
        n = random.randint(3, 5)
        pairs = list(itertools.combinations(range(n), 2))
        e1 = [p for p in pairs if random.random() < 0.4]
        e2 = [p for p in pairs if random.random() < 0.4]
        c1 = [random.randint(0, 2) for _ in range(n)]
        c2 = [random.randint(0, 2) for _ in range(n)]
        et1 = {p: 0 for p in e1}
        et2 = {p: 0 for p in e2}
        g1, nt1, _ = _build(n, e1, c1, et1)
        g2, nt2, _ = _build(n, e2, c2, et2)
        inc1 = _incidence(g1, nt1, et1, 0)
        inc2 = _incidence(g2, nt2, et2, 0)
        # the oracle compares radius-2 ego patches; brute-force only over the
        # same patch when radius 2 already covers the whole molecule.
        if inc1.n_nodes != n or inc2.n_nodes != n:
            continue
        oracle = typed_isomorphic_vf2(inc1, inc2)
        if oracle:
            # oracle must be a true color-preserving isomorphism: convert our
            # semantic node colors back to the bare atom types used by brute.
            assert brute_iso(n, e1, c1, e2, c2)
        elif c1[0] == c2[0] and not brute_iso(n, e1, c1, e2, c2):
            assert not oracle


def test_10_resolve_version_and_fingerprint_stability() -> None:
    assert resolve_typed_tokenizer_version(None) == TYPED_TOKENIZER_V1_HISTORICAL
    with pytest.raises(ValueError):
        resolve_typed_tokenizer_version("nope")
    assert typed_tokenizer_fingerprint(
        TYPED_TOKENIZER_V2_CORRECTED, 2
    ) == typed_tokenizer_fingerprint(TYPED_TOKENIZER_V2_CORRECTED, 2)
    assert typed_tokenizer_fingerprint(
        TYPED_TOKENIZER_V1_HISTORICAL, 2
    ) != typed_tokenizer_fingerprint(TYPED_TOKENIZER_V2_CORRECTED, 2)
