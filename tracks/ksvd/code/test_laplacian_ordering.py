"""Self-tests for graph-global Fiedler patch ordering."""
from __future__ import annotations

import numpy as np

from .from_scratch_unplanted_representation import is_connected
from .laplacian_ordering import (
    coordinate_tie_diagnostics,
    mapped_relabel_fiedler_invariance,
    normalized_fiedler_coordinate,
    one_swap_fiedler_stability,
    pair_order_agreement,
    reorder_cover_by_fiedler,
)
from .overlap_cover import patch_budget, sample_edge_target_bridge_cover
from .run_overlap_cover_audit import generate_graph


def _path_adjacency(n_nodes: int) -> np.ndarray:
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.int8)
    for node in range(n_nodes - 1):
        adjacency[node, node + 1] = 1
        adjacency[node + 1, node] = 1
    return adjacency


def main() -> int:
    path = _path_adjacency(30)
    coordinate = normalized_fiedler_coordinate(path)
    assert coordinate.values.shape == (30,)
    assert np.array_equal(np.sort(coordinate.ranks), np.arange(30))
    assert coordinate.eigenvalues.shape == (3,)
    assert pair_order_agreement(coordinate.ranks, coordinate.ranks) == 1.0
    assert pair_order_agreement(coordinate.ranks, -coordinate.ranks) == 0.0
    ties = coordinate_tie_diagnostics(np.asarray([0.0, 0.0, 1.0, 2.0]))
    assert ties["unique_coordinate_group_count"] == 3
    assert ties["tied_node_fraction"] == 0.5

    adjacency = generate_graph("small_world", 30, 8, seed=704)
    budget = patch_budget(adjacency, patch_size=8, target_overlap=4)
    cover = sample_edge_target_bridge_cover(
        adjacency,
        np.random.default_rng(701),
        n_patches=budget,
        patch_size=8,
        target_overlap=4,
    )
    reordered, _coordinate = reorder_cover_by_fiedler(adjacency, cover)
    assert all(
        set(left.node_ids) == set(right.node_ids)
        for left, right in zip(cover.patches, reordered.patches)
    )
    assert all(
        list(patch.node_ids)
        == sorted(patch.node_ids, key=lambda node: _coordinate.ranks[node])
        for patch in reordered.patches
    )
    permutation = np.random.default_rng(702).permutation(adjacency.shape[0])
    replay = mapped_relabel_fiedler_invariance(adjacency, cover, permutation)
    assert replay["rank_match_rate"] == 1.0
    assert replay["patch_adjacency_match_rate"] == 1.0
    assert replay["transition_slot_map_match_rate"] == 1.0

    stability = one_swap_fiedler_stability(
        adjacency, np.random.default_rng(703)
    )
    assert 0.0 <= stability["canonical_pair_order_agreement"] <= 1.0
    assert 0.5 <= stability["sign_invariant_pair_order_agreement"] <= 1.0
    assert stability["swap_attempts"] >= 1
    assert is_connected(adjacency)
    print("laplacian_ordering self-tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
