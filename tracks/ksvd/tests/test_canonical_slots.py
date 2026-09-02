"""Pytest form of the historical canonical-slot self-test."""

from __future__ import annotations

import numpy as np

from ksvd_research.features.canonical_slots import (
    exact_canonical_order,
    reorder_cover_structurally,
)
from ksvd_research.sampling.overlap import make_cover, make_patch, remap_cover


def _example_adjacency() -> np.ndarray:
    adjacency = np.zeros((8, 8), dtype=np.int8)
    for left, right in (
        (0, 1),
        (1, 2),
        (2, 0),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 3),
        (5, 6),
        (6, 7),
    ):
        adjacency[left, right] = adjacency[right, left] = 1
    return adjacency


def test_exact_order_is_invariant_to_node_relabeling() -> None:
    adjacency = _example_adjacency()
    nodes = (0, 1, 2, 3, 4, 5)
    base = exact_canonical_order(adjacency[np.ix_(nodes, nodes)], nodes)
    rng = np.random.default_rng(20260802)
    for _ in range(20):
        permutation = rng.permutation(len(nodes))
        relabeled_nodes = tuple(int(nodes[index]) for index in permutation)
        changed = exact_canonical_order(
            adjacency[np.ix_(relabeled_nodes, relabeled_nodes)], relabeled_nodes
        )
        assert changed.adjacency_code == base.adjacency_code


def test_symmetric_patch_reports_ambiguity() -> None:
    clique = np.ones((6, 6), dtype=np.int8) - np.eye(6, dtype=np.int8)
    result = exact_canonical_order(clique, tuple(range(6)))
    assert result.ambiguous
    assert result.symmetry_pruned


def test_cover_reordering_preserves_relabel_invariant_patch_adjacency() -> None:
    adjacency = _example_adjacency()
    cover = make_cover(
        "manual",
        [
            make_patch(adjacency, (0, 1, 2, 3, 4), 2),
            make_patch(adjacency, (2, 3, 4, 5, 6), 5),
            make_patch(adjacency, (4, 5, 6, 7, 3), 7),
        ],
    )
    rng = np.random.default_rng(20260802)
    permutation = rng.permutation(adjacency.shape[0])
    inverse = np.empty(adjacency.shape[0], dtype=np.int64)
    inverse[permutation] = np.arange(adjacency.shape[0])
    relabeled_adjacency = adjacency[np.ix_(permutation, permutation)]
    mapped = remap_cover(cover, inverse, relabeled_adjacency)
    for mode in ("canonical", "rooted_canonical", "overlap_canonical"):
        original_ordered, _ = reorder_cover_structurally(adjacency, cover, mode)
        mapped_ordered, _ = reorder_cover_structurally(relabeled_adjacency, mapped, mode)
        assert all(
            np.array_equal(left.adjacency, right.adjacency)
            for left, right in zip(original_ordered.patches, mapped_ordered.patches)
        )
