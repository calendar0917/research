"""Self-tests for masked patch-relation representation features."""
from __future__ import annotations

import numpy as np

from .marginal_candidate_cover import sample_marginal_candidate_cover
from .overlap_cover import patch_budget
from .run_overlap_cover_audit import generate_graph
from .run_patch_relation_representation_audit import (
    BRANCHES,
    all_pairs_shortest_paths,
    bag_features,
    classify,
    graph_embedding,
    patch_invariant_descriptor,
    relation_features,
)


def _fold(index: int) -> dict:
    values = {
        "RAW_BAG": 0.110,
        "KSVD_BAG": 0.100,
        "KSVD_TRUE_RELATION": 0.095,
        "KSVD_SHUFFLED_RELATION": 0.105,
    }
    return {
        "fold_index": index,
        "branches": {
            branch: {
                "summary": {
                    "overall_rmse": value,
                    "degree_rmse": value,
                    "spectrum_rmse": value,
                    "density_triangle_rmse": value,
                }
            }
            for branch, value in values.items()
        },
        "stability": {
            branch: {"cosine_similarity": 0.95, "relative_l2_difference": 0.1}
            for branch in BRANCHES
        },
        "invariants": {"passed": True},
    }


def main() -> int:
    adjacency = generate_graph("small_world", 30, 8, seed=941)
    budget = patch_budget(adjacency, patch_size=8, target_overlap=3)
    cover = sample_marginal_candidate_cover(
        adjacency,
        np.random.default_rng(942),
        n_patches=budget,
        patch_size=8,
        target_overlap=3,
        retained_beam=8,
        candidate_restarts=1,
    )
    patch = cover.patches[0].adjacency
    permutation = np.random.default_rng(943).permutation(patch.shape[0])
    assert np.allclose(
        patch_invariant_descriptor(patch),
        patch_invariant_descriptor(patch[np.ix_(permutation, permutation)]),
    )
    tokens = np.random.default_rng(944).normal(size=(len(cover.patches), 24))
    distances = all_pairs_shortest_paths(adjacency)
    bag = bag_features(tokens, 0)
    true = relation_features(tokens, cover, distances, 0, shuffled=False)
    shuffled = relation_features(tokens, cover, distances, 0, shuffled=True)
    assert bag.shape == (72,)
    assert true.shape == shuffled.shape == (124,)
    assert not np.allclose(true, shuffled)
    raw = np.stack(
        [patch.adjacency[np.triu_indices(8, k=1)] for patch in cover.patches], axis=0
    )
    for branch in BRANCHES:
        embedding = graph_embedding(branch, raw, tokens, cover, adjacency)
        assert embedding.ndim == 1
        assert np.all(np.isfinite(embedding))
    decision = classify([_fold(0), _fold(1), _fold(2)])
    assert decision["classification"] == "PATCH_RELATIONS_ADD_MASKED_VALUE"
    print("patch_relation_representation_audit self-tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
