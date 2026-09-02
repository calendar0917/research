"""Self-tests for scalable marginal candidate covers."""
from __future__ import annotations

import numpy as np

from .marginal_candidate_cover import sample_marginal_candidate_cover
from .overlap_cover import audit_cover, patch_budget
from .run_overlap_cover_audit import generate_graph


def main() -> int:
    adjacency = generate_graph("small_world", 30, 8, seed=911)
    budget = patch_budget(adjacency, patch_size=8, target_overlap=3)
    cover = sample_marginal_candidate_cover(
        adjacency,
        np.random.default_rng(912),
        n_patches=budget,
        patch_size=8,
        target_overlap=3,
        retained_beam=12,
        candidate_restarts=2,
    )
    audit = audit_cover(adjacency, cover)
    assert all(transition.overlap_size == 3 for transition in cover.transitions)
    assert audit["patch_connected_rate"] == 1.0
    assert audit["continuous_transition_fraction"] == 1.0
    assert audit["observed_pair_accuracy"] == 1.0
    print("marginal_candidate_cover self-tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
