"""Self-tests for graph-global stable structural IDs."""
from __future__ import annotations

import numpy as np

from .global_stable_ids import (
    ambiguity_audit,
    compute_global_wl_ids,
    compute_rooted_wl_ids,
    mapped_id_audit,
    reorder_by_stable_ids,
)
from .run_overlap_cover_audit import generate_graph


def main() -> int:
    adjacency = generate_graph("small_world", 18, 6, 20260805)
    rng = np.random.default_rng(20260806)
    permutation = rng.permutation(adjacency.shape[0])
    relabeled_adjacency = adjacency[np.ix_(permutation, permutation)]
    for compute in (compute_global_wl_ids, compute_rooted_wl_ids):
        base = compute(adjacency)
        relabeled = compute(relabeled_adjacency)
        audit = mapped_id_audit(base, relabeled, permutation)
        assert audit["stable_class_match_rate"] == 1.0
        assert audit["singleton_unique_id_match_rate"] == 1.0
        if base.fully_singleton:
            assert audit["all_node_concrete_id_match_rate"] == 1.0
            assert np.array_equal(
                reorder_by_stable_ids(adjacency, base),
                reorder_by_stable_ids(relabeled_adjacency, relabeled),
            )

    cycle = np.zeros((8, 8), dtype=np.int8)
    for node in range(8):
        cycle[node, (node + 1) % 8] = 1
        cycle[(node + 1) % 8, node] = 1
    cycle_ids = compute_rooted_wl_ids(cycle)
    assert cycle_ids.singleton_fraction == 0.0
    assert cycle_ids.largest_class == 8
    permutation = rng.permutation(8)
    changed = cycle[np.ix_(permutation, permutation)]
    cycle_audit = mapped_id_audit(
        cycle_ids, compute_rooted_wl_ids(changed), permutation
    )
    assert cycle_audit["stable_class_match_rate"] == 1.0
    assert cycle_audit["singleton_trial_count"] == 0
    cycle_ambiguity = ambiguity_audit(cycle, cycle_ids)
    assert cycle_ambiguity["ambiguous_pair_count"] == 28
    # Cycle vertices share one rooted-WL class, but arbitrary swaps are not automorphisms.
    assert cycle_ambiguity["ambiguous_swap_automorphism_fraction"] == 0.0
    print("global_stable_ids self-tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
