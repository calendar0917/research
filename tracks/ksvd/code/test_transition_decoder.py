"""Self-tests for the explicit transition decoder feature contract."""
from __future__ import annotations

import numpy as np

from .from_scratch_unplanted_representation import ring_lattice_adjacency
from .overlap_cover import patch_budget, sample_edge_target_bridge_cover
from .overlap_stitching import make_cover_example
from .transition_decoder import (
    fit_ridge_decoder,
    make_transition_features,
    predict_ridge_decoder,
    shuffle_transition_context,
)


def main() -> int:
    adjacency = ring_lattice_adjacency(30, 4)
    budget = patch_budget(adjacency, patch_size=8, target_overlap=4)
    cover = sample_edge_target_bridge_cover(
        adjacency,
        np.random.default_rng(601),
        n_patches=budget,
        patch_size=8,
        target_overlap=4,
    )
    example = make_cover_example(0, "ring", 8, adjacency, cover)
    codes = np.random.default_rng(602).normal(size=(6, budget))
    current, transition = make_transition_features(
        codes, example.patch_vectors, cover
    )
    assert current.shape == (budget, 6)
    assert transition.shape == (budget - 1, 6 * 2 + 8 * 3)
    shuffled = shuffle_transition_context(transition, 6)
    assert np.array_equal(shuffled[:, :6], transition[:, :6])
    assert not np.array_equal(shuffled[:, 6:], transition[:, 6:])

    targets = np.random.default_rng(603).normal(size=(budget, 5))
    model = fit_ridge_decoder(current, targets, alpha=1e-2)
    prediction = predict_ridge_decoder(model, current)
    assert prediction.shape == targets.shape
    assert np.all(np.isfinite(prediction))
    print("transition_decoder self-tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
