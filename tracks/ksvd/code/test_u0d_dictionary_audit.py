"""Self-tests for the U0-D unplanted dictionary audit."""
from __future__ import annotations

import numpy as np

from .from_scratch_unplanted_dictionary import (
    cross_replicate_dictionary_similarity,
    deterministic_maximin_initialization,
    exact_maximum_cosine_assignment,
    run_u0d_dictionary_audit,
)
from .from_scratch_unplanted_signal import generate_u0p_dataset


def main() -> int:
    values = np.asarray(
        [
            [1.0, 0.0, -1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, -1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    dictionary_a, info_a = deterministic_maximin_initialization(values, 3)
    dictionary_b, info_b = deterministic_maximin_initialization(values, 3)
    assert np.array_equal(dictionary_a, dictionary_b)
    assert info_a == info_b
    assert np.allclose(np.linalg.norm(dictionary_a, axis=0), 1.0)

    score, assignment = exact_maximum_cosine_assignment(
        np.asarray([[0.1, 0.9, 0.2], [0.8, 0.1, 0.2], [0.1, 0.2, 0.7]])
    )
    assert assignment == (1, 0, 2)
    assert np.isclose(score, 0.8)

    root = np.random.SeedSequence(731101)
    data_sequence, _shuffle_sequence = root.spawn(2)
    dataset = generate_u0p_dataset(
        data_sequence,
        train_per_class=5,
        validation_per_class=2,
        test_per_class=3,
        patches_per_graph=6,
    )
    result, initial, final = run_u0d_dictionary_audit(
        dataset,
        n_atoms=6,
        sparsity=2,
        minimum_sparsity=1,
        n_iterations=2,
    )
    assert result["patch_counts"] == {"train": 60, "validation": 24, "test": 36}
    assert initial.shape == (15, 6)
    assert final.shape == (15, 6)
    for stage in ("init", "final"):
        for split in ("train", "validation", "test"):
            metrics = result["stages"][stage][split]
            assert metrics["patch_count"] == result["patch_counts"][split]
            assert 0.0 <= metrics["relative_reconstruction_error"]
            assert 1.0 <= metrics["mean_nonzeros_per_patch"] <= 2.0
    similarity = cross_replicate_dictionary_similarity([initial, initial.copy()])
    assert np.isclose(similarity["matched_cosine_mean"], 1.0)
    assert np.isclose(similarity["subspace_cosine_mean"], 1.0)
    print("u0d_dictionary_audit self-tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
