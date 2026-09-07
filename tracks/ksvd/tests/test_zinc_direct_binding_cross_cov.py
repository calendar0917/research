from __future__ import annotations

import numpy as np

from ksvd_research.core import from_edges
from tracks.ksvd.experiments.luyin16.zinc_direct_binding_cross_cov import (
    _cross_covariance,
    _direct_graph_features_from_parts,
    _entity_statistics,
    _relabel_parts,
    direct_patch_roles,
    role_dimensions,
)


REPRESENTATION = {
    "radius": 2,
    "degree_bins": 5,
    "atom_categories": 4,
    "bond_categories": 3,
}


def test_direct_roles_ignore_attributes() -> None:
    graph = from_edges(6, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (4, 5)])
    first = direct_patch_roles(graph, 0, REPRESENTATION)
    second = direct_patch_roles(graph, 0, REPRESENTATION)
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(first[3], second[3])


def test_binding_has_zero_role_and_attribute_marginals() -> None:
    roles = np.asarray([0, 0, 1, 1], dtype=np.int64)
    attributes = np.asarray(
        [[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32
    )
    role, attribute, joint = _entity_statistics(roles, attributes, 2)
    binding = joint - role[:, None] * attribute[None, :]
    np.testing.assert_allclose(binding.sum(axis=1), 0.0, atol=1.0e-7)
    np.testing.assert_allclose(binding.sum(axis=0), 0.0, atol=1.0e-7)


def test_cross_covariance_uses_aligned_centre_rows() -> None:
    topology = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    attributes = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    shuffled = attributes[[1, 0]]
    aligned = _cross_covariance(topology, attributes).reshape(2, 2)
    broken = _cross_covariance(topology, shuffled).reshape(2, 2)
    assert np.linalg.norm(aligned) > 0.0
    np.testing.assert_allclose(broken, -aligned, atol=1.0e-7)


def test_direct_graph_statistics_are_relabeling_invariant() -> None:
    graph = from_edges(
        7,
        [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (4, 5), (5, 6)],
    )
    nodes = np.asarray([0, 1, 0, 2, 1, 3, 2], dtype=np.int64)
    edges = {
        (0, 1): 1,
        (1, 2): 2,
        (2, 3): 1,
        (0, 3): 2,
        (0, 4): 1,
        (4, 5): 2,
        (5, 6): 1,
    }
    base = _direct_graph_features_from_parts(graph, nodes, edges, REPRESENTATION)
    changed_graph, changed_nodes, changed_edges = _relabel_parts(
        graph, nodes, edges, np.asarray([4, 2, 6, 1, 5, 0, 3], dtype=np.int64)
    )
    changed = _direct_graph_features_from_parts(
        changed_graph, changed_nodes, changed_edges, REPRESENTATION
    )
    for name in ("marginal", "context", "cross_cov", "binding"):
        np.testing.assert_allclose(base[name], changed[name], atol=1.0e-6, rtol=0.0)


def test_direct_dimensions_are_explicit() -> None:
    dimensions = role_dimensions(REPRESENTATION)
    assert dimensions["node_role"] == 30
    assert dimensions["edge_role"] == 12
    assert dimensions["topology_row"] == 72
    assert dimensions["attribute_row"] == 14
    assert dimensions["cross_cov_raw"] == 72 * 14


def test_direct_statistics_have_expected_width() -> None:
    graph = from_edges(3, [(0, 1), (1, 2)])
    nodes = np.asarray([0, 1, 2], dtype=np.int64)
    edges = {(0, 1): 1, (1, 2): 2}
    features = _direct_graph_features_from_parts(graph, nodes, edges, REPRESENTATION)
    dimensions = role_dimensions(REPRESENTATION)
    assert features["marginal"].shape == (2 * (dimensions["topology_row"] + dimensions["attribute_row"]),)
    assert features["cross_cov"].shape == (dimensions["cross_cov_raw"],)
    assert features["binding"].shape == (2 * dimensions["binding_raw"],)
