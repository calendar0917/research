"""Self-tests for per-graph IMDB dictionary descriptors."""
from __future__ import annotations

import numpy as np

from .imdb_pergraph_dictionary import (
    extract_pergraph_dictionary_features,
    invariant_dictionary_readout,
    legacy_ordered_dictionary_readout,
    run_pergraph_downstream_fold,
    summarize_pergraph_view,
)
from .imdb_walk_dictionary import FoldSplit
from .imdb_walk_substrate import IMDBPatchGraph


def _example(index: int, label: int, rng: np.random.Generator) -> IMDBPatchGraph:
    patch_count = 8 + index % 3
    values = (rng.random((patch_count, 21)) < (0.25 + 0.20 * label)).astype(np.float64)
    # Ensure every patch is nonzero for deterministic real-column initialization.
    values[:, 0] = 1.0
    edge_counts = np.sum(values, axis=1).astype(np.int64)
    histogram = np.bincount(np.clip(edge_counts - 6, 0, 15), minlength=16).astype(np.float64)
    histogram /= patch_count
    stats = np.asarray(
        [
            12.0 + index % 4,
            float(np.mean(edge_counts)),
            0.2 + 0.1 * label,
            2.0 + label,
            0.5,
            1.0,
            4.0 + label,
            float(label),
            0.2,
            2.0,
            4.0,
            1.0,
        ],
        dtype=np.float64,
    )
    return IMDBPatchGraph(
        graph_index=index,
        label=label,
        n_nodes=12 + index % 4,
        n_edges=int(np.mean(edge_counts)),
        n_patches=patch_count,
        root_coverage=1.0,
        walk_vectors=values,
        canonical_vectors=values.copy(),
        edge_counts=edge_counts,
        features={
            "walk_mean_std": np.concatenate(
                [np.mean(values, axis=0), np.std(values, axis=0, ddof=0)]
            ),
            "edge_count_histogram": histogram,
        },
        graph_statistics=stats,
    )


def main() -> int:
    rng = np.random.default_rng(41)
    dictionary = rng.standard_normal((21, 4))
    dictionary /= np.linalg.norm(dictionary, axis=0, keepdims=True)
    codes = rng.standard_normal((4, 9))
    codes[np.abs(codes) < 0.5] = 0.0
    values = dictionary @ codes + 0.01 * rng.standard_normal((21, 9))
    reference = invariant_dictionary_readout(dictionary, codes, values)
    permutation = np.asarray([2, 0, 3, 1])
    signs = np.asarray([-1.0, 1.0, -1.0, 1.0])
    transformed_dictionary = dictionary[:, permutation] * signs[None, :]
    transformed_codes = codes[permutation, :] * signs[:, None]
    transformed = invariant_dictionary_readout(
        transformed_dictionary, transformed_codes, values
    )
    assert np.allclose(reference, transformed, atol=1e-10, rtol=1e-10)
    legacy = legacy_ordered_dictionary_readout(dictionary, codes)
    legacy_permuted = legacy_ordered_dictionary_readout(
        transformed_dictionary, transformed_codes
    )
    assert legacy.shape == legacy_permuted.shape
    assert not np.allclose(legacy, legacy_permuted)

    example = _example(0, 0, np.random.default_rng(42))
    descriptor = extract_pergraph_dictionary_features(
        example, n_atoms=4, sparsity=2, minimum_sparsity=1, n_iterations=2
    )
    assert descriptor.invariant_init.shape == descriptor.invariant_final.shape
    assert descriptor.invariant_pca.shape == descriptor.invariant_final.shape
    assert descriptor.legacy_init.shape == descriptor.legacy_final.shape
    assert descriptor.raw_patch.shape == (58,)
    assert descriptor.invariant_permutation_max_abs_difference <= 1e-9
    assert descriptor.init_reconstruction_error >= 0.0
    assert descriptor.final_reconstruction_error >= 0.0

    examples = tuple(
        _example(index, index % 2, np.random.default_rng(1000 + index))
        for index in range(30)
    )
    descriptors = tuple(
        extract_pergraph_dictionary_features(
            item,
            n_atoms=4,
            sparsity=2,
            minimum_sparsity=1,
            n_iterations=1,
        )
        for item in examples
    )
    outer = FoldSplit(
        fold_index=0,
        train_indices=tuple(range(24)),
        test_indices=tuple(range(24, 30)),
    )
    inner = FoldSplit(
        fold_index=0,
        train_indices=tuple(range(18)),
        test_indices=tuple(range(18, 24)),
    )
    result = run_pergraph_downstream_fold(
        examples,
        descriptors,
        outer,
        inner,
        graph_shuffle_seed=732121,
        label_shuffle_seed=732131,
    )
    assert result["invariance"]["test_maximum_atom_permutation_abs_difference"] <= 1e-9
    assert "stats_raw_invariant_final" in result["evaluations"]
    assert "stats_raw_invariant_pca" in result["evaluations"]
    assert "stats_raw_legacy_final_atom_permuted" in result["evaluations"]
    assert np.isclose(
        result["attribution"]["primary_update_gain"],
        result["evaluations"]["stats_raw_invariant_final"]["test_balanced_accuracy"]
        - result["evaluations"]["stats_raw_invariant_init"]["test_balanced_accuracy"],
    )
    summary = summarize_pergraph_view([result])
    assert summary["fold_count"] == 1
    assert summary["maximum_invariant_permutation_difference"] <= 1e-9
    print("imdb_pergraph_dictionary self-tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
