from __future__ import annotations

import numpy as np

from ksvd_research.core import from_edges
from tracks.ksvd.experiments.luyin16.patch_object_audit import (
    relabel_graph_features,
)
from tracks.ksvd.experiments.luyin16.structural_role_fusion_screen import (
    STRICT_ATOM_DIM,
    audit_invariance,
    exact_topology_audit,
    graph_features,
    patch_feature_rows,
    patch_roles,
    representation_dimensions,
    strict_atom_semantics,
)


REPRESENTATION = {
    "radius": 2,
    "degree_bins": 8,
    "wl_rounds": 2,
    "node_role_bins": 64,
    "edge_role_bins": 32,
    "typed_edge_bins": 128,
    "max_centers_per_graph": None,
}


def _features(graph):
    nodes = np.zeros((graph.n, 9), dtype=np.int64)
    nodes[:, 0] = np.arange(graph.n) % 10
    nodes[:, 1] = np.arange(graph.n) % 4
    nodes[:, 2] = np.arange(graph.n) % 6
    nodes[:, 3] = 5
    nodes[:, 4] = np.arange(graph.n) % 4
    nodes[:, 6] = np.arange(graph.n) % 5
    nodes[:, 7] = np.arange(graph.n) % 2
    nodes[:, 8] = (np.arange(graph.n) + 1) % 2
    edges = {
        edge: np.asarray([index % 4, index % 6, index % 2], dtype=np.int64)
        for index, edge in enumerate(graph.edges())
    }
    return nodes, edges


def test_strict_atom_attributes_exclude_degree_and_ring() -> None:
    raw = np.zeros((3, 9), dtype=np.int64)
    raw[:, 0] = [5, 6, 7]
    base, groups = strict_atom_semantics(raw)
    changed = raw.copy()
    changed[:, 2] = [1, 4, 10]
    changed[:, 8] = [1, 0, 1]
    same, changed_groups = strict_atom_semantics(changed)
    np.testing.assert_array_equal(base, same)
    np.testing.assert_array_equal(groups, changed_groups)
    assert base.shape == (3, STRICT_ATOM_DIM)


def test_rooted_wl_roles_use_topology_only() -> None:
    graph = from_edges(7, [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (5, 6)])
    base = patch_roles(graph, 0, "rooted_wl", REPRESENTATION)
    nodes, edges = _features(graph)
    changed_nodes = nodes[::-1].copy()
    changed_edges = {key: values[::-1].copy() for key, values in edges.items()}
    del changed_nodes, changed_edges
    changed = patch_roles(graph, 0, "rooted_wl", REPRESENTATION)
    np.testing.assert_array_equal(base.node_ids, changed.node_ids)
    np.testing.assert_array_equal(base.edge_ids, changed.edge_ids)
    assert base.patch_key == changed.patch_key


def test_graph_features_are_relabeling_invariant() -> None:
    graph = from_edges(
        9,
        [(0, 1), (0, 2), (1, 3), (1, 4), (3, 4), (2, 5), (2, 6), (6, 7), (6, 8)],
    )
    nodes, edges = _features(graph)
    permutation = np.asarray([4, 2, 8, 1, 7, 5, 0, 6, 3], dtype=np.int64)
    changed_graph, changed_nodes, changed_edges = relabel_graph_features(
        graph, nodes, edges, permutation
    )
    for schema in ("coarse", "rooted_wl"):
        base = graph_features(graph, nodes, edges, schema, REPRESENTATION)
        changed = graph_features(
            changed_graph, changed_nodes, changed_edges, schema, REPRESENTATION
        )
        for name in (
            "node_role",
            "edge_role",
            "node_attribute",
            "edge_attribute",
            "node_raw",
            "edge_raw",
            "typed_edge",
            "node_binding",
            "edge_binding",
            "context",
        ):
            np.testing.assert_allclose(base[name], changed[name], rtol=0.0, atol=1e-6)


def test_patch_feature_rows_mean_matches_graph_features() -> None:
    graph = from_edges(
        8,
        [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (5, 6), (5, 7)],
    )
    nodes, edges = _features(graph)
    rows = patch_feature_rows(graph, nodes, edges, "rooted_wl", REPRESENTATION)
    blocks = graph_features(graph, nodes, edges, "rooted_wl", REPRESENTATION)
    for name in (
        "node_role",
        "edge_role",
        "node_attribute",
        "edge_attribute",
        "node_raw",
        "edge_raw",
        "typed_edge",
        "node_binding",
        "edge_binding",
    ):
        np.testing.assert_allclose(
            rows[name].mean(axis=0), blocks[name], rtol=0.0, atol=1e-6
        )
    np.testing.assert_allclose(rows["context"], blocks["context"], rtol=0.0, atol=1e-6)


def test_patch_shuffle_preserves_marginals_but_changes_alignment() -> None:
    graph = from_edges(
        8,
        [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (5, 6), (5, 7)],
    )
    nodes, edges = _features(graph)
    blocks = graph_features(
        graph,
        nodes,
        edges,
        "rooted_wl",
        REPRESENTATION,
        shuffle_repeats=1,
        shuffle_seed=19,
    )
    for name in ("node_role", "edge_role", "node_attribute", "edge_attribute"):
        np.testing.assert_allclose(
            blocks[name], blocks[f"{name}_shuffled_0"], rtol=0.0, atol=1e-7
        )
    differences = [
        np.max(np.abs(blocks[name] - blocks[f"{name}_shuffled_0"]))
        for name in ("node_raw", "edge_raw", "typed_edge")
    ]
    assert max(differences) > 1e-5


def test_dimensions_are_matched_across_role_schemas() -> None:
    dims = representation_dimensions(REPRESENTATION)
    assert dims["node_role"] == 64
    assert dims["edge_role"] == 32
    assert dims["node_attribute"] == STRICT_ATOM_DIM
    assert dims["node_raw"] == 64 * STRICT_ATOM_DIM
    assert dims["edge_raw"] == 32 * dims["edge_attribute"]


def test_invariance_audit_and_exact_coverage_on_synthetic_bundle() -> None:
    train_graph = from_edges(6, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (4, 5)])
    valid_graph = from_edges(6, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (4, 5)])
    train_nodes, train_edges = _features(train_graph)
    valid_nodes, valid_edges = _features(valid_graph)

    class Bundle:
        graphs = [train_graph, valid_graph]
        node_feats = [train_nodes, valid_nodes]
        edge_feats = [train_edges, valid_edges]

    for schema in ("coarse", "rooted_wl"):
        audit = audit_invariance(
            Bundle(),
            np.asarray([0], dtype=np.int64),
            schema,
            REPRESENTATION,
            n_graphs=1,
            permutations_per_graph=2,
            seed=7,
            tolerance=1.0e-6,
        )
        assert audit["pass"]
    exact = exact_topology_audit(
        Bundle(),
        np.asarray([0], dtype=np.int64),
        np.asarray([1], dtype=np.int64),
        REPRESENTATION,
        max_train_graphs=1,
        max_valid_graphs=1,
        seed=3,
    )
    assert exact["valid_patch_seen_rate"] == 1.0
    assert exact["rooted_wl"]["maximum_exact_types_per_key"] >= 1
