"""Self-tests for the ID-free invariant patch-relation follow-up."""
from __future__ import annotations

import numpy as np

from .marginal_candidate_cover import sample_marginal_candidate_cover
from .overlap_cover import patch_budget
from .overlap_stitching import make_cover_example
from .run_invariant_patch_relation_followup import (
    BRANCHES,
    build_masked_matrices,
    classify,
    graph_embedding,
)
from .run_overlap_cover_audit import generate_graph


def _fold(index: int) -> dict:
    values = {
        "INVARIANT_BAG": 0.100,
        "INVARIANT_TRUE_RELATION": 0.090,
        "INVARIANT_SHUFFLED_RELATION": 0.095,
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
    adjacency = generate_graph("regular", 30, 8, seed=951)
    budget = patch_budget(adjacency, patch_size=8, target_overlap=3)
    cover = sample_marginal_candidate_cover(
        adjacency,
        np.random.default_rng(952),
        n_patches=budget,
        patch_size=8,
        target_overlap=3,
        retained_beam=8,
        candidate_restarts=1,
    )
    example = make_cover_example(0, "regular", 8, adjacency, cover)
    features, targets, graph_rows = build_masked_matrices([example])
    assert targets.shape == (len(cover.patches), 18)
    assert graph_rows.shape == (len(cover.patches),)
    assert features["INVARIANT_BAG"].shape[1] == 54
    assert features["INVARIANT_TRUE_RELATION"].shape[1] == 94
    for branch in BRANCHES:
        embedding = graph_embedding(branch, example)
        assert embedding.ndim == 1
        assert np.all(np.isfinite(embedding))
    decision = classify([_fold(0), _fold(1), _fold(2)])
    assert decision["classification"] == "ID_FREE_PATCH_RELATION_SUBSTRATE_SUPPORTED"
    print("invariant_patch_relation_followup self-tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
