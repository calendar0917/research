"""Self-tests for the U0-R unplanted adjacency representation audit."""
from __future__ import annotations

import numpy as np

from .from_scratch_unplanted_representation import (
    adjacency_to_upper_vector,
    audit_patch_collection,
    exact_rooted_graph_edit_distance,
    generate_rewired_graph,
    is_connected,
    relabel_adjacency_and_order,
    rooted_exact_canonical_vector,
    sample_graph_walk_patches,
    upper_vector_to_adjacency,
)


def main() -> int:
    graph = generate_rewired_graph(seed=732001, n_accepted_swaps=25)
    adjacency = graph.adjacency
    assert graph.accepted_swaps == 25
    assert adjacency.shape == (60, 60)
    assert int(adjacency.sum() // 2) == 120
    assert np.all(adjacency.sum(axis=1) == 4)
    assert is_connected(adjacency)

    patches = sample_graph_walk_patches(
        adjacency,
        np.random.default_rng(91),
        n_patches=8,
    )
    assert len(patches) == 8
    for patch in patches:
        assert len(set(patch.node_ids)) == 6
        assert patch.node_ids[0] == patch.root
        assert patch.adjacency_walk_order.shape == (6, 6)
        assert int(patch.adjacency_walk_order.sum() // 2) >= 5
        round_trip = upper_vector_to_adjacency(patch.walk_order_vector, 6)
        assert np.array_equal(round_trip, patch.adjacency_walk_order)

        permutation = np.asarray([0, 3, 1, 5, 2, 4])
        permuted = patch.adjacency_walk_order[np.ix_(permutation, permutation)]
        assert np.array_equal(
            rooted_exact_canonical_vector(permuted).vector,
            patch.canonical_vector,
        )
        assert exact_rooted_graph_edit_distance(patch.adjacency_walk_order, permuted) == 0

        flipped = patch.adjacency_walk_order.copy()
        flipped[0, 1] = 1 - flipped[0, 1]
        flipped[1, 0] = flipped[0, 1]
        assert exact_rooted_graph_edit_distance(patch.adjacency_walk_order, flipped) == 1
        assert np.count_nonzero(
            adjacency_to_upper_vector(flipped) != patch.walk_order_vector
        ) == 1

    global_permutation = np.random.default_rng(123).permutation(60)
    for patch in patches:
        relabeled, mapped_order = relabel_adjacency_and_order(
            adjacency,
            patch.node_ids,
            global_permutation,
        )
        mapped_induced = relabeled[np.ix_(mapped_order, mapped_order)]
        assert np.array_equal(mapped_induced, patch.adjacency_walk_order)
        assert np.array_equal(
            adjacency_to_upper_vector(mapped_induced),
            patch.walk_order_vector,
        )

    audit = audit_patch_collection(
        patches,
        np.random.default_rng(314),
        pair_count=20,
        root_permutation_trials=3,
    )
    assert audit["canonical_permutation_invariance_rate"] == 1.0
    assert audit["one_edge_flip"]["exact_distance_all_one"]
    assert audit["one_edge_flip"]["walk_distance_all_one"]
    assert audit["pairwise"]["canonical_injectivity_disagreement_count"] == 0
    print("u0r_adjacency_representation self-tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
