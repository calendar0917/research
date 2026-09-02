"""Self-tests for ID-free exact and chain-anchored patch slot orderings."""
from __future__ import annotations

import numpy as np

from .canonical_slots import exact_canonical_order, reorder_cover_structurally
from .overlap_cover import _make_cover, _make_patch, remap_cover


def main() -> int:
    adjacency = np.zeros((8, 8), dtype=np.int8)
    for left, right in ((0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3), (5, 6), (6, 7)):
        adjacency[left, right] = adjacency[right, left] = 1
    nodes = (0, 1, 2, 3, 4, 5)
    patch_adjacency = adjacency[np.ix_(nodes, nodes)]
    base = exact_canonical_order(patch_adjacency, nodes)
    rng = np.random.default_rng(20260802)
    for _ in range(20):
        permutation = rng.permutation(len(nodes))
        relabeled_nodes = tuple(int(nodes[index]) for index in permutation)
        relabeled_patch = adjacency[np.ix_(relabeled_nodes, relabeled_nodes)]
        changed = exact_canonical_order(relabeled_patch, relabeled_nodes)
        assert changed.adjacency_code == base.adjacency_code

    clique = np.ones((6, 6), dtype=np.int8) - np.eye(6, dtype=np.int8)
    clique_result = exact_canonical_order(clique, tuple(range(6)))
    assert clique_result.ambiguous
    assert clique_result.symmetry_pruned

    cover = _make_cover(
        "manual",
        [
            _make_patch(adjacency, (0, 1, 2, 3, 4), 2),
            _make_patch(adjacency, (2, 3, 4, 5, 6), 5),
            _make_patch(adjacency, (4, 5, 6, 7, 3), 7),
        ],
    )
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
    print("canonical_slots self-tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
