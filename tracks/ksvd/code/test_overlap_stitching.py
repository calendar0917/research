"""Self-tests for overlap-cover compression and graph stitching."""
from __future__ import annotations

from itertools import combinations

import numpy as np

from .from_scratch_unplanted_representation import ring_lattice_adjacency
from .overlap_cover import patch_budget, sample_edge_target_bridge_cover
from .overlap_stitching import (
    fit_pca_basis,
    make_cover_example,
    reconstruct_with_pca,
    stack_cover_examples,
    stitch_patch_predictions,
)


def main() -> int:
    adjacency = ring_lattice_adjacency(n_nodes=30, neighbors_each_side=4)
    budget = patch_budget(adjacency, patch_size=8, target_overlap=4)
    cover = sample_edge_target_bridge_cover(
        adjacency,
        np.random.default_rng(501),
        n_patches=budget,
        patch_size=8,
        target_overlap=4,
    )
    example = make_cover_example(0, "ring", 8, adjacency, cover)
    stacked = stack_cover_examples([example])
    assert stacked.shape == (28, budget)

    raw = stitch_patch_predictions(example, example.patch_vectors)
    assert raw["patch_relative_error"] == 0.0
    assert raw["observed_pair_rmse"] == 0.0
    assert raw["observed_edge_f1"] == 1.0
    assert raw["repeated_pair_disagreement_std_mean"] == 0.0
    assert 0.0 < raw["full_edge_recall"] <= 1.0
    observed_pairs = {
        tuple(sorted(pair))
        for patch in cover.patches
        for pair in combinations(patch.node_ids, 2)
    }
    residual_edges = {
        (left, right)
        for left in range(adjacency.shape[0])
        for right in range(left + 1, adjacency.shape[0])
        if adjacency[left, right] and (left, right) not in observed_pairs
    }
    corrected = stitch_patch_predictions(
        example,
        example.patch_vectors,
        exact_residual_edges=residual_edges,
    )
    assert corrected["full_adjacency_rmse"] == 0.0
    assert corrected["full_edge_recall"] == 1.0

    mean = np.mean(stacked, axis=1, keepdims=True)
    centered = stacked - mean
    basis = fit_pca_basis(centered, rank=3)
    reconstructed = reconstruct_with_pca(centered, basis) + mean
    pca = stitch_patch_predictions(example, reconstructed.T)
    assert pca["patch_relative_error"] >= 0.0
    assert pca["observed_pair_rmse"] >= 0.0
    assert np.allclose(basis.T @ basis, np.eye(3))
    print("overlap_stitching self-tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
