from __future__ import annotations

from collections import Counter

import numpy as np

from ksvd_research.core import from_edges
from tracks.ksvd.experiments.luyin16.exact_orbit_fusion_screen import (
    _cross_patch_attribute_pairs,
    _deterministic_farthest_prototypes,
    _extract_graph_data,
    _record_token,
    _shuffle_entity_attributes,
)
from tracks.ksvd.experiments.luyin16.patch_object_audit import (
    relabel_graph_features,
)


REPRESENTATION = {
    "radius": 2,
    "wl_rounds": 2,
    "node_role_bins": 64,
    "edge_role_bins": 32,
    "max_centers_per_graph": None,
}


def _features(graph):
    nodes = np.zeros((graph.n, 9), dtype=np.int64)
    nodes[:, 0] = np.arange(graph.n) % 10
    nodes[:, 1] = np.arange(graph.n) % 4
    nodes[:, 3] = 5
    nodes[:, 4] = np.arange(graph.n) % 4
    nodes[:, 5] = np.arange(graph.n) % 3
    nodes[:, 6] = np.arange(graph.n) % 5
    nodes[:, 7] = np.arange(graph.n) % 2
    edges = {
        edge: np.asarray([index % 4, index % 6, index % 2], dtype=np.int64)
        for index, edge in enumerate(graph.edges())
    }
    return nodes, edges


def test_rooted_star_uses_structural_automorphism_orbits() -> None:
    graph = from_edges(5, [(0, 1), (0, 2), (0, 3), (0, 4)])
    nodes, edges = _features(graph)
    data = _extract_graph_data(
        graph,
        nodes,
        edges,
        REPRESENTATION,
        shuffle_repeats=1,
        shuffle_seed=17,
    )
    root_patch = data.patches[0]
    assert root_patch.n_node_orbits == 2
    assert root_patch.n_edge_orbits == 1
    assert root_patch.template.shape == (2 * 40 + 13,)


def test_matched_shuffle_preserves_marginal_and_is_deterministic() -> None:
    rows = np.eye(4, dtype=np.float32)
    canonical_order = np.asarray([2, 0, 3, 1], dtype=np.int64)
    key = b"topology"
    first = _shuffle_entity_attributes(
        rows,
        canonical_order,
        key,
        base_seed=11,
        repeat=0,
        modality="node",
    )
    second = _shuffle_entity_attributes(
        rows,
        canonical_order,
        key,
        base_seed=11,
        repeat=0,
        modality="node",
    )
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first.sum(axis=0), rows.sum(axis=0))
    assert not np.array_equal(first, rows)


def test_exact_orbit_patch_multiset_is_relabel_invariant() -> None:
    graph = from_edges(
        8,
        [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (5, 6), (5, 7)],
    )
    nodes, edges = _features(graph)
    base = _extract_graph_data(
        graph,
        nodes,
        edges,
        REPRESENTATION,
        shuffle_repeats=2,
        shuffle_seed=23,
    )
    permutation = np.asarray([4, 2, 7, 0, 6, 1, 5, 3], dtype=np.int64)
    changed_graph, changed_nodes, changed_edges = relabel_graph_features(
        graph, nodes, edges, permutation
    )
    changed = _extract_graph_data(
        changed_graph,
        changed_nodes,
        changed_edges,
        REPRESENTATION,
        shuffle_repeats=2,
        shuffle_seed=23,
    )
    assert Counter(map(_record_token, base.patches)) == Counter(
        map(_record_token, changed.patches)
    )
    np.testing.assert_allclose(base.wl_structure, changed.wl_structure, atol=1e-7)
    np.testing.assert_allclose(base.context, changed.context, atol=1e-7)


def test_deterministic_prototype_selection_has_no_seed() -> None:
    rows = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.5, 0.5]],
        dtype=np.float32,
    )
    first = _deterministic_farthest_prototypes(rows, maximum=3)
    second = _deterministic_farthest_prototypes(rows[::-1], maximum=3)
    np.testing.assert_allclose(first, second, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-6)


def test_cross_patch_pair_shuffle_is_relabel_invariant() -> None:
    graph = from_edges(
        7,
        [(0, 1), (0, 2), (1, 3), (2, 4), (4, 5), (4, 6)],
    )
    nodes, edges = _features(graph)
    base = _extract_graph_data(
        graph,
        nodes,
        edges,
        REPRESENTATION,
        shuffle_repeats=1,
        shuffle_seed=29,
    )
    permutation = np.asarray([3, 6, 0, 4, 1, 5, 2], dtype=np.int64)
    changed_graph, changed_nodes, changed_edges = relabel_graph_features(
        graph, nodes, edges, permutation
    )
    changed = _extract_graph_data(
        changed_graph,
        changed_nodes,
        changed_edges,
        REPRESENTATION,
        shuffle_repeats=1,
        shuffle_seed=29,
    )

    def paired_tokens(data):
        return Counter(
            (patch.topology_key, row.tobytes())
            for patch, row in _cross_patch_attribute_pairs(
                data, base_seed=31, repeat=0
            )
        )

    assert paired_tokens(base) == paired_tokens(changed)
