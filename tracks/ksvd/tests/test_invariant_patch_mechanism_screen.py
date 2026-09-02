from __future__ import annotations

import numpy as np

from ksvd_research.core import from_edges
from tracks.ksvd.experiments.luyin16.invariant_patch_mechanism_screen import (
    build_graph_patch_matrices,
    distribution_readout,
    invariant_patch_blocks,
    relabel_graph_features,
    representation_dimensions,
)


REPRESENTATION = {
    "radius": 2,
    "degree_bins": 12,
    "atom_categories": 119,
    "bond_categories": 5,
    "normalize_patches": True,
    "node_scale": 16.0,
    "edge_scale": 24.0,
    "degree_scale": 8.0,
    "cycle_scale": 8.0,
    "triangle_scale": 8.0,
}


def _features(graph):
    nodes = np.zeros((graph.n, 9), dtype=np.int64)
    nodes[:, 0] = (np.arange(graph.n) * 16) % 119
    edges = {
        edge: np.asarray([index % 4, 0, 0], dtype=np.int64)
        for index, edge in enumerate(graph.edges())
    }
    return nodes, edges


def test_invariant_readouts_are_exact_under_relabeling() -> None:
    graph = from_edges(
        9,
        [(0, 1), (0, 2), (1, 3), (1, 4), (3, 4), (2, 5), (2, 6), (6, 7), (6, 8)],
    )
    nodes, edges = _features(graph)
    base = build_graph_patch_matrices(graph, nodes, edges, REPRESENTATION)
    permutation = np.asarray([4, 2, 8, 1, 7, 5, 0, 6, 3], dtype=np.int64)
    changed_graph, changed_nodes, changed_edges = relabel_graph_features(
        graph, nodes, edges, permutation
    )
    changed = build_graph_patch_matrices(
        changed_graph,
        changed_nodes,
        changed_edges,
        REPRESENTATION,
        centers=[int(permutation[center]) for center in graph.nodes],
    )
    for name in ("topology", "attributes", "joint"):
        np.testing.assert_array_equal(
            distribution_readout(base[name]), distribution_readout(changed[name])
        )


def test_full_ego_scope_keeps_nodes_beyond_eight() -> None:
    graph = from_edges(10, [(0, node) for node in range(1, 10)])
    nodes, edges = _features(graph)
    _topology, attributes, meta = invariant_patch_blocks(
        graph, 0, nodes, edges, REPRESENTATION
    )
    assert meta["n_nodes"] == 10
    # Categories 0 and 16 occupy distinct direct-index coordinates in shells 0 and 1.
    assert attributes[0] > 0
    assert attributes[119 + 16] > 0


def test_dimensions_and_readout_width_are_explicit() -> None:
    dims = representation_dimensions(REPRESENTATION)
    assert dims == {"topology": 52, "attributes": 387, "joint": 439}
    matrix = np.eye(dims["joint"], 3, dtype=np.float64)
    assert distribution_readout(matrix).shape == (6 * dims["joint"],)
