"""Self-tests for the G0B-R representation-only gate."""
from __future__ import annotations

import numpy as np

from .from_scratch_hidden_motif import MOTIF_EDGE_LISTS, motif_template
from .from_scratch_representation_gate import (
    canonicalize_with_hidden_core,
    evaluate_representation_condition,
    make_add_one_noncore_variants,
    make_clean_variants,
    make_edge_flip_variants,
)
from .run_from_scratch_g0b_representation_gate import classify


def main() -> int:
    rng = np.random.default_rng(31)
    for motif_index, core_edges in enumerate(MOTIF_EDGE_LISTS):
        adjacency = motif_template(motif_index)
        reference = canonicalize_with_hidden_core(adjacency, core_edges)
        for _ in range(10):
            permutation = rng.permutation(6)
            inverse = np.empty(6, dtype=np.int64)
            inverse[permutation] = np.arange(6)
            permuted_adjacency = adjacency[np.ix_(permutation, permutation)]
            permuted_core = tuple(
                (int(inverse[left]), int(inverse[right])) for left, right in core_edges
            )
            result = canonicalize_with_hidden_core(permuted_adjacency, permuted_core)
            assert np.array_equal(reference.vector, result.vector)
            assert set(reference.possible_core_supports) == set(result.possible_core_supports)

    clean = evaluate_representation_condition(make_clean_variants())
    assert clean["gate_passed"]
    assert clean["label_identifiability"]["equal_prior_bayes_accuracy"] > 0.999999
    assert clean["core_coordinate_consistency"]["minimum_best_robust_fixed_support_mean_f1"] > 0.999999

    add_one = evaluate_representation_condition(make_add_one_noncore_variants())
    assert add_one["variant_count"] > 40
    assert add_one["label_identifiability"]["cross_family_collision_vector_count"] > 0

    flips = make_edge_flip_variants(seed=31, samples_per_motif=40)
    assert len(flips) == 160
    flip_result = evaluate_representation_condition(flips)
    assert 0.0 <= flip_result["label_identifiability"]["equal_prior_bayes_accuracy"] <= 1.0

    decision = classify([clean, add_one, flip_result])
    assert decision["clean_control_passed"]
    print("g0b_representation_gate self-tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
