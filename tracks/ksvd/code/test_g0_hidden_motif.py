"""Fast self-tests for G0 complete-graph hidden-motif discovery."""
from __future__ import annotations

import numpy as np

from .from_scratch_hidden_motif import (
    MOTIF_NAMES,
    deterministic_maximin_initialization,
    exact_canonical_vector,
    is_connected,
    make_g0_dataset,
    motif_template,
    oracle_control,
    random_column_initialization,
    run_g0_stage,
    true_motif_dictionary,
)
from .run_from_scratch_g0_hidden_motif import classify


def main() -> int:
    rng = np.random.default_rng(31)
    for motif_index in range(len(MOTIF_NAMES)):
        adjacency = motif_template(motif_index)
        reference = exact_canonical_vector(adjacency)
        for _ in range(10):
            permutation = rng.permutation(6)
            permuted = adjacency[np.ix_(permutation, permutation)]
            assert np.array_equal(reference, exact_canonical_vector(permuted))

    D_true, supports = true_motif_dictionary()
    assert D_true.shape == (15, 4)
    assert len(supports) == 4
    assert np.unique(D_true.T, axis=0).shape[0] == 4

    dataset = make_g0_dataset(
        seed=20260731,
        n_train_graphs=6,
        n_test_graphs=3,
        cells_per_graph=12,
    )
    assert dataset.Y_train.shape == (15, 72)
    assert dataset.Y_test.shape == (15, 36)
    assert np.unique(dataset.Y_train.T, axis=0).shape[0] == 4
    assert all(is_connected(graph.adjacency) for graph in dataset.train_graphs)
    assert min(np.bincount(dataset.train_labels, minlength=4)) > 0
    assert min(np.bincount(dataset.test_labels, minlength=4)) > 0

    oracle = oracle_control(dataset)
    assert oracle["passed"]

    deterministic, deterministic_info = deterministic_maximin_initialization(dataset.Y_train, 4)
    assert deterministic_info["selected_unique_column_count"] == 4
    _D_init, init_metrics = run_g0_stage(dataset, deterministic, n_iter=0)
    assert init_metrics["minimum_atom_cosine"] > 0.999999
    assert init_metrics["primary_decode"]["exact_motif_count"] == 4
    assert init_metrics["test_occurrence"]["occurrence_macro_f1"] > 0.999999

    random_dictionary, random_info = random_column_initialization(dataset.Y_train, 4, seed=0)
    assert random_dictionary.shape == (15, 4)
    assert 1 <= random_info["selected_unique_column_count"] <= 4

    fake_stability = {
        "pair_count": 1,
        "pairwise_matched_atom_cosine_mean": 1.0,
        "pairwise_matched_atom_cosine_minimum": 1.0,
        "pairs": [],
    }
    perfect_metrics = {
        key: {"mean": value, "std": 0.0, "minimum": value, "maximum": value}
        for key, value in {
            "mean_atom_cosine": 1.0,
            "minimum_atom_cosine": 1.0,
            "primary_decode_exact_motif_count": 4.0,
            "test_occurrence_macro_f1": 1.0,
            "exact_occurrence_accuracy": 1.0,
        }.items()
    }
    summary = {
        "data_seed_count": 10,
        "oracle_pass_count": 10,
        "connected_data_count": 10,
        "four_unique_patch_data_count": 10,
        "full_motif_coverage_count": 10,
        "conditions": {
            "deterministic_maximin": {
                "init": {"strict_discovery_count": 10, "metrics": perfect_metrics, "dictionary_stability": fake_stability},
                "final": {"strict_discovery_count": 10, "metrics": perfect_metrics, "dictionary_stability": fake_stability},
            },
            "fixed_random_columns": {
                "init": {"strict_discovery_count": 0},
                "final": {"strict_discovery_count": 5},
            },
        },
    }
    decision = classify(summary)
    assert decision["classification"] == "PASS_INITIALIZER_DISCOVERY_ONLY"

    print("g0_hidden_motif self-tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
