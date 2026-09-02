"""Decision self-tests for the global stable node ID audit."""
from __future__ import annotations

import numpy as np

from .global_stable_ids import compute_global_wl_ids, reorder_by_stable_ids
from .run_global_stable_node_id_audit import (
    classify,
    relabeled_stable_to_base_mapping,
)


def _id(method: str, **updates: float) -> dict:
    row = {
        "id_method": method,
        "graph_index": 0,
        "family": "regular",
        "target_degree": 6,
        "permutation_seed": 1,
        "stable_class_match_rate": 1.0,
        "singleton_unique_id_match_rate": 1.0,
        "all_node_concrete_id_match_rate": 1.0,
        "singleton_trial_count": 18,
        "singleton_fraction": 1.0,
        "fully_singleton": 1.0,
        "largest_class": 1,
        "ambiguous_class_count": 0,
        "canonical_adjacency_match": 1.0,
        "global_rooted_order_match": 1.0,
        "ambiguous_pair_count": 0,
        "ambiguous_swap_automorphism_fraction": 1.0,
        "id_seconds": 0.01,
    }
    row.update(updates)
    return row


def _record(method: str) -> dict:
    return {
        "record_method": method,
        "graph_index": 0,
        "family": "regular",
        "target_degree": 6,
        "permutation_seed": 1,
        "ordered_membership_row_match": 1.0,
        "rooted_membership_row_match": 1.0,
        "class_membership_row_match": 1.0,
        "transition_record_match": 1.0,
        "residual_concrete_match": 1.0,
        "residual_class_match": 1.0,
    }


def _sampler(branch: str, exact: float, vectors: float) -> dict:
    return {
        "sampler_branch": branch,
        "graph_index": 0,
        "family": "regular",
        "target_degree": 6,
        "permutation_seed": 1,
        "sampling_seconds": 0.1,
        "fully_singleton_graph": True,
        "exact_ordered_chain_match": exact,
        "patch_set_jaccard": exact,
        "equivalence_class_chain_match": exact,
        "equivalence_class_patch_jaccard": exact,
        "canonical_coordinate_chain_match": exact,
        "rooted_vector_row_match": vectors,
        "rooted_transition_map_match": vectors,
        "left_edge_coverage": 0.9,
        "right_edge_coverage": 0.9,
        "left_pair_coverage": 0.6,
        "right_pair_coverage": 0.6,
        "absolute_edge_coverage_delta": 0.0,
        "absolute_pair_coverage_delta": 0.0,
        "absolute_raw_rmse_delta": 0.0,
        "left_raw_rmse": 0.2,
        "right_raw_rmse": 0.2,
        "left_invariants": True,
        "right_invariants": True,
    }


def _test_abstract_coordinate_mapping() -> None:
    rng = np.random.default_rng(44)
    upper = np.triu((rng.random((9, 9)) < 0.35).astype(np.int8), k=1)
    adjacency = upper + upper.T
    permutation = rng.permutation(adjacency.shape[0])
    relabeled = adjacency[np.ix_(permutation, permutation)]
    base_ids = compute_global_wl_ids(adjacency)
    relabeled_ids = compute_global_wl_ids(relabeled)
    mapping = relabeled_stable_to_base_mapping(
        base_ids, relabeled_ids, permutation
    )
    base_stable = reorder_by_stable_ids(adjacency, base_ids)
    relabeled_stable = reorder_by_stable_ids(relabeled, relabeled_ids)
    assert np.array_equal(
        relabeled_stable, base_stable[np.ix_(mapping, mapping)]
    )


def main() -> int:
    _test_abstract_coordinate_mapping()
    ids = [_id("GLOBAL_WL"), _id("ROOTED_WL")]
    records = [_record("INPUT_ID"), _record("GLOBAL_WL"), _record("ROOTED_WL")]
    samplers = [
        _sampler("RAW_BEAM8", 0.0, 0.1),
        _sampler("STABLE_ID_PREORDER_BEAM8", 1.0, 1.0),
    ]
    passed = classify(ids, records, samplers)
    assert passed["classification"] == "ADOPT_GLOBAL_STABLE_ID_PREORDER"
    assert passed["id_gate"] and passed["sampler_gate"]

    weak_ids = [
        _id("GLOBAL_WL", singleton_fraction=0.5),
        _id("ROOTED_WL", singleton_fraction=0.5),
    ]
    weak = classify(weak_ids, records, samplers)
    assert weak["classification"] == "STABLE_EQUIVALENCE_CLASSES_ONLY"

    broken_ids = [
        _id("GLOBAL_WL", stable_class_match_rate=0.9),
        _id("ROOTED_WL"),
    ]
    broken = classify(broken_ids, records, samplers)
    assert broken["classification"] == "FAIL_GLOBAL_STABLE_ID_INVARIANTS"
    print("global_stable_node_id_audit self-tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
