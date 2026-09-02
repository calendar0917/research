"""Pytest checks for the maintained patch-vectorization boundary."""

from __future__ import annotations

import numpy as np
import pytest

from ksvd_research.core.graph import from_edges
from ksvd_research.features.vectorize import (
    adjacency_padded,
    canonical_adjacency_features,
    flatten_upper,
    wl_patch_features,
)


def _relabel(g, permutation: np.ndarray):
    return from_edges(g.n, [(int(permutation[u]), int(permutation[v])) for u, v in g.edges()])


def test_wl_and_padded_vectors_have_stable_shapes() -> None:
    graph = from_edges(4, [(0, 1), (1, 2), (2, 3)])
    padded = adjacency_padded(graph, set(graph.nodes), 6)
    assert len(padded) == 6
    assert len(flatten_upper(padded)) == 6 * 5 // 2
    assert len(wl_patch_features(graph, set(graph.nodes), 6)) == 6 + 6 * 7 // 2 + 64 * 3 + 3


def test_wl_vector_is_invariant_to_node_relabeling() -> None:
    graph = from_edges(6, [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5)])
    permutation = np.array([3, 5, 1, 4, 0, 2])
    changed = _relabel(graph, permutation)
    base = np.asarray(wl_patch_features(graph, set(graph.nodes), 8))
    relabeled = np.asarray(wl_patch_features(changed, set(changed.nodes), 8))
    assert np.array_equal(base, relabeled)


def test_exact_canonical_vector_is_invariant_when_pynauty_is_available() -> None:
    pytest.importorskip("pynauty")
    prism = from_edges(
        6,
        [
            (0, 1),
            (1, 2),
            (2, 0),
            (3, 4),
            (4, 5),
            (5, 3),
            (0, 3),
            (1, 4),
            (2, 5),
        ],
    )
    permutation = np.array([2, 5, 0, 4, 1, 3])
    changed = _relabel(prism, permutation)
    base = np.asarray(canonical_adjacency_features(prism, set(prism.nodes), 8))
    relabeled = np.asarray(canonical_adjacency_features(changed, set(changed.nodes), 8))
    assert np.array_equal(base, relabeled)
