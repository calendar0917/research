"""Self-tests for exhaustive marginal coverage on small graphs."""
from __future__ import annotations

import numpy as np

from .marginal_coverage_cover import sample_exhaustive_marginal_cover
from .overlap_cover import audit_cover
from .run_overlap_cover_audit import generate_graph


def main() -> int:
    adjacency = generate_graph("small_world", 14, 4, seed=901)
    cover = sample_exhaustive_marginal_cover(
        adjacency,
        np.random.default_rng(902),
        n_patches=5,
        patch_size=5,
        target_overlap=2,
    )
    audit = audit_cover(adjacency, cover)
    assert len(cover.patches) == 5
    assert all(transition.overlap_size == 2 for transition in cover.transitions)
    assert audit["patch_connected_rate"] == 1.0
    assert audit["continuous_transition_fraction"] == 1.0
    assert audit["observed_pair_accuracy"] == 1.0
    print("marginal_coverage_cover self-tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
