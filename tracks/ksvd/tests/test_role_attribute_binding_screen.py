from __future__ import annotations

import numpy as np

from ksvd_research.core import from_edges
from tracks.ksvd.experiments.luyin16.patch_object_audit import relabel_graph_features
from tracks.ksvd.experiments.luyin16.role_attribute_binding_screen import (
    ATOM_DIM,
    BOND_DIM,
    _entity_statistics,
    compact_atom_semantics,
    compact_bond_semantics,
    graph_role_attribute_features,
    audit_invariance,
    role_dimensions,
)


REPRESENTATION = {"radius": 2, "degree_bins": 8}


def _features(graph):
    nodes = np.zeros((graph.n, 9), dtype=np.int64)
    nodes[:, 0] = (np.arange(graph.n) * 16) % 119
    nodes[:, 1] = np.arange(graph.n) % 5
    nodes[:, 2] = np.arange(graph.n) % 6
    edges = {
        edge: np.asarray([index % 4, index % 6, index % 2], dtype=np.int64)
        for index, edge in enumerate(graph.edges())
    }
    return nodes, edges


def test_compact_attribute_dimensions_are_explicit() -> None:
    graph = from_edges(3, [(0, 1), (1, 2)])
    nodes, edges = _features(graph)
    assert compact_atom_semantics(nodes).shape == (3, ATOM_DIM)
    assert compact_bond_semantics(edges[(0, 1)]).shape == (BOND_DIM,)


def test_roles_do_not_change_when_attributes_change() -> None:
    graph = from_edges(6, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (4, 5)])
    nodes, edges = _features(graph)
    base = graph_role_attribute_features(graph, nodes, edges, REPRESENTATION)
    changed_nodes = nodes.copy()
    changed_nodes[:, :] = np.arange(changed_nodes.size).reshape(changed_nodes.shape) % 6
    changed_edges = {key: np.asarray([(i + 2) % 5, 0, 1]) for i, key in enumerate(edges)}
    changed = graph_role_attribute_features(graph, changed_nodes, changed_edges, REPRESENTATION)
    np.testing.assert_allclose(base["node_role"], changed["node_role"], atol=0.0)
    np.testing.assert_allclose(base["edge_role"], changed["edge_role"], atol=0.0)
    assert not np.allclose(base["node_attribute"], changed["node_attribute"])


def test_true_blocks_are_relabeling_invariant() -> None:
    graph = from_edges(
        9,
        [(0, 1), (0, 2), (1, 3), (1, 4), (3, 4), (2, 5), (2, 6), (6, 7), (6, 8)],
    )
    nodes, edges = _features(graph)
    permutation = np.asarray([4, 2, 8, 1, 7, 5, 0, 6, 3], dtype=np.int64)
    changed_graph, changed_nodes, changed_edges = relabel_graph_features(
        graph, nodes, edges, permutation
    )
    base = graph_role_attribute_features(graph, nodes, edges, REPRESENTATION)
    changed = graph_role_attribute_features(changed_graph, changed_nodes, changed_edges, REPRESENTATION)
    for name in ("node_role", "edge_role", "node_attribute", "edge_attribute", "node_raw", "edge_raw", "node_binding", "edge_binding", "context"):
        np.testing.assert_allclose(base[name], changed[name], rtol=0.0, atol=1e-6)


def test_shuffle_preserves_marginals_and_changes_joint() -> None:
    graph = from_edges(9, [(0, 1), (0, 2), (1, 3), (1, 4), (3, 4), (2, 5), (2, 6), (6, 7), (6, 8)])
    nodes, edges = _features(graph)
    blocks = graph_role_attribute_features(
        graph, nodes, edges, REPRESENTATION, shuffle_repeats=1, shuffle_seed=91
    )
    for name in ("node_role", "edge_role", "node_attribute", "edge_attribute"):
        np.testing.assert_allclose(blocks[name], blocks[f"{name}_shuffled_0"], rtol=0.0, atol=1e-7)
    assert np.max(np.abs(blocks["node_binding"] - blocks["node_binding_shuffled_0"])) > 1e-5
    assert np.max(np.abs(blocks["edge_binding"] - blocks["edge_binding_shuffled_0"])) > 1e-5


def test_centered_joint_has_zero_role_and_attribute_marginals() -> None:
    roles = np.asarray([0, 0, 1, 1], dtype=np.int64)
    attrs = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32)
    role, attribute, joint = _entity_statistics(roles, attrs, 2)
    binding = joint - role[:, None] * attribute[None, :]
    np.testing.assert_allclose(binding.sum(axis=1), 0.0, atol=1e-7)
    np.testing.assert_allclose(binding.sum(axis=0), 0.0, atol=1e-7)


def test_role_dimensions_match_schema() -> None:
    dimensions = role_dimensions(REPRESENTATION)
    assert dimensions["node_role"] == 48
    assert dimensions["edge_role"] == 12
    assert dimensions["node_raw"] == 48 * ATOM_DIM
    assert dimensions["edge_raw"] == 12 * BOND_DIM


def test_invariance_audit_passes_on_synthetic_bundle() -> None:
    graph = from_edges(6, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (4, 5)])
    nodes, edges = _features(graph)

    class Bundle:
        graphs = [graph]
        node_feats = [nodes]
        edge_feats = [edges]

    audit = audit_invariance(
        Bundle(),
        np.asarray([0], dtype=np.int64),
        REPRESENTATION,
        n_graphs=1,
        permutations_per_graph=2,
        seed=7,
        tolerance=1.0e-6,
    )
    assert audit["pass"]
