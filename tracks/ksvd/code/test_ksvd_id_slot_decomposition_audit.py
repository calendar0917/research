"""Self-tests for KSVD ID/slot instability decomposition helpers."""
from __future__ import annotations

import numpy as np

from .marginal_candidate_cover import sample_marginal_candidate_cover
from .overlap_cover import patch_budget, remap_cover
from .overlap_stitching import make_cover_example
from .run_ksvd_id_slot_decomposition_audit import (
    CONDITIONS,
    classify,
    random_slot_cover,
    sorted_id_cover,
)
from .run_overlap_cover_audit import generate_graph


def _fold(index: int) -> dict:
    means = {}
    for condition in CONDITIONS:
        cosine = 1.0
        if condition == "FROZEN_SET_SLOT_SHUFFLE":
            cosine = 0.7
        elif condition == "RELABEL_RESAMPLE":
            cosine = 0.75
        means[condition] = {
            "graph_embedding_cosine": cosine,
            "graph_embedding_relative_l2": 0.0,
            "patch_vector_row_match": 1.0
            if condition == "MAPPED_GLOBAL_RELABEL"
            else None,
            "matched_patch_code_cosine": None,
            "matched_patch_support_jaccard": None,
        }
    return {
        "fold_index": index,
        "conditions": means,
        "invariants": {"passed": True},
    }


def main() -> int:
    adjacency = generate_graph("small_world", 30, 8, seed=961)
    budget = patch_budget(adjacency, patch_size=8, target_overlap=3)
    cover = sample_marginal_candidate_cover(
        adjacency,
        np.random.default_rng(962),
        n_patches=budget,
        patch_size=8,
        target_overlap=3,
        retained_beam=8,
        candidate_restarts=1,
    )
    shuffled = random_slot_cover(adjacency, cover, 963)
    sorted_cover = sorted_id_cover(adjacency, cover)
    assert all(
        set(left.node_ids) == set(right.node_ids) == set(third.node_ids)
        for left, right, third in zip(cover.patches, shuffled.patches, sorted_cover.patches)
    )
    assert any(
        left.node_ids != right.node_ids
        for left, right in zip(cover.patches, shuffled.patches)
    )
    permutation = np.random.default_rng(964).permutation(adjacency.shape[0])
    inverse = np.empty(adjacency.shape[0], dtype=np.int64)
    inverse[permutation] = np.arange(adjacency.shape[0])
    relabeled = adjacency[np.ix_(permutation, permutation)]
    mapped = remap_cover(cover, inverse, relabeled)
    original_example = make_cover_example(0, "small_world", 8, adjacency, cover)
    mapped_example = make_cover_example(0, "small_world", 8, relabeled, mapped)
    assert np.array_equal(original_example.patch_vectors, mapped_example.patch_vectors)
    decision = classify([_fold(0), _fold(1), _fold(2)])
    assert decision["classification"] == "MIXED_SAMPLER_AND_SLOT_INSTABILITY"
    print("ksvd_id_slot_decomposition_audit self-tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
