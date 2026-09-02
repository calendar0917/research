"""Self-tests for invariant-space KSVD token auditing."""
from __future__ import annotations

import numpy as np

from .run_invariant_space_ksvd_token_audit import (
    BRANCHES,
    classify,
    graph_embedding,
)
from .run_invariant_patch_relation_followup import invariant_tokens
from .marginal_candidate_cover import sample_marginal_candidate_cover
from .overlap_cover import patch_budget
from .overlap_stitching import make_cover_example
from .run_overlap_cover_audit import generate_graph


def _fold(index: int) -> dict:
    values = {
        "INV_KSVD_BAG": 0.140,
        "INV_KSVD_TRUE_RELATION": 0.120,
        "INV_KSVD_SHUFFLED_RELATION": 0.130,
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
            branch: {"cosine_similarity": 0.98, "relative_l2_difference": 0.05}
            for branch in BRANCHES
        },
        "invariants": {"passed": True},
    }


def main() -> int:
    adjacency = generate_graph("regular", 30, 8, seed=971)
    budget = patch_budget(adjacency, patch_size=8, target_overlap=3)
    cover = sample_marginal_candidate_cover(
        adjacency,
        np.random.default_rng(972),
        n_patches=budget,
        patch_size=8,
        target_overlap=3,
        retained_beam=8,
        candidate_restarts=1,
    )
    example = make_cover_example(0, "regular", 8, adjacency, cover)
    tokens = invariant_tokens(example)
    assert tokens.shape == (len(cover.patches), 18)
    codes = np.random.default_rng(973).normal(size=(len(cover.patches), 24))
    for branch in BRANCHES:
        embedding = graph_embedding(branch, example, codes)
        assert embedding.ndim == 1
        assert np.all(np.isfinite(embedding))
    decision = classify([_fold(0), _fold(1), _fold(2)])
    assert decision["classification"] == "ADOPT_INVARIANT_SPACE_KSVD_TOKEN"
    print("invariant_space_ksvd_token_audit self-tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
