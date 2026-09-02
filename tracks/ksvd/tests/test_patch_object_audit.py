from __future__ import annotations

import numpy as np

from ksvd_research.core import from_edges
from tracks.ksvd.experiments.luyin16.patch_object_audit import (
    build_graph_patch_views,
    relabel_graph_features,
)


def _features(graph):
    node_features = np.zeros((graph.n, 9), dtype=np.int64)
    node_features[:, 0] = np.arange(graph.n)
    edge_features = {
        edge: np.asarray([index % 4, 0, 0], dtype=np.int64)
        for index, edge in enumerate(graph.edges())
    }
    return node_features, edge_features


def test_relabel_graph_features_preserves_node_and_edge_semantics() -> None:
    graph = from_edges(4, [(0, 1), (1, 2), (1, 3)])
    nodes, edges = _features(graph)
    permutation = np.asarray([2, 0, 3, 1], dtype=np.int64)
    changed, changed_nodes, changed_edges = relabel_graph_features(
        graph, nodes, edges, permutation
    )
    for old in graph.nodes:
        np.testing.assert_array_equal(changed_nodes[permutation[old]], nodes[old])
    for left, right in graph.edges():
        mapped = tuple(sorted((int(permutation[left]), int(permutation[right]))))
        assert changed.has_edge(*mapped)
        np.testing.assert_array_equal(changed_edges[mapped], edges[(left, right)])


def test_current_coordinate_readout_can_change_under_node_relabeling() -> None:
    # The two first-shell branches have equal degree but different structure.
    graph = from_edges(
        7,
        [
            (0, 1),
            (0, 2),
            (1, 3),
            (1, 4),
            (3, 4),
            (2, 5),
            (2, 6),
        ],
    )
    nodes, edges = _features(graph)
    base = build_graph_patch_views(
        graph, nodes, edges, radius=2, max_nodes=8
    )
    permutation = np.asarray([0, 2, 1, 5, 6, 3, 4], dtype=np.int64)
    changed_graph, changed_nodes, changed_edges = relabel_graph_features(
        graph, nodes, edges, permutation
    )
    changed = build_graph_patch_views(
        changed_graph,
        changed_nodes,
        changed_edges,
        radius=2,
        max_nodes=8,
        centers=[int(permutation[center]) for center in graph.nodes],
    )
    assert not np.array_equal(base["topology_readout"], changed["topology_readout"])


def test_matched_scope_drops_attributes_outside_the_adjacency_cap() -> None:
    graph = from_edges(10, [(0, node) for node in range(1, 10)])
    nodes, edges = _features(graph)
    nodes[9, 0] = 15
    views = build_graph_patch_views(graph, nodes, edges, radius=1, max_nodes=8)
    assert np.any(views["full_sizes"] > views["kept_sizes"])
    assert not np.array_equal(views["legacy_readout"], views["matched_readout"])
