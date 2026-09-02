"""Fast contract tests for the E1-T2 initialization audit."""
from __future__ import annotations

import numpy as np

from .from_scratch_recovery import make_e1_dataset
from .run_from_scratch_e1_t2_initialization_audit import (
    average_ranks,
    classify_r5,
    exact_containment_probability,
    grouped_initialization_success,
    is_strict_success,
    pearson_correlation,
    reconstruct_initialization,
    simulate_restart_budget,
    spearman_correlation,
)


def fake_run(seed: int, train: float, atom: float, support: float = 1.0) -> dict:
    return {
        "learner_seed": seed,
        "train_reconstruction_relative": train,
        "test_reconstruction_relative": 0.0 if atom >= 0.99 else 0.2,
        "mean_atom_cosine": atom,
        "support_f1": support,
    }


def main() -> int:
    dataset = make_e1_dataset(seed=23, n_train=80, n_test=40, max_sparsity=2)
    init_a = reconstruct_initialization(dataset, 7)
    init_b = reconstruct_initialization(dataset, 7)
    assert init_a == init_b
    assert len(init_a["training_column_indices"]) == 4
    assert 0 <= init_a["singleton_column_count"] <= 4
    assert 0.0 <= init_a["initial_mean_atom_cosine"] <= 1.0 + 1e-12

    assert np.allclose(average_ranks([10.0, 20.0, 20.0, 40.0]), [1.0, 2.5, 2.5, 4.0])
    assert pearson_correlation([1, 2, 3], [3, 2, 1]) < -0.999999
    assert spearman_correlation([1, 2, 3], [10, 20, 30]) > 0.999999
    assert abs(exact_containment_probability(population_size=4, success_count=1, budget=2) - 0.5) < 1e-12

    runs = [
        fake_run(0, 0.30, 0.50),
        fake_run(1, 0.10, 1.00),
        fake_run(2, 0.20, 0.60),
        fake_run(3, 0.40, 0.70),
    ]
    assert is_strict_success(runs[1])
    for run, coverage in zip(runs, (2, 4, 3, 4)):
        run["initialization_audit"] = {
            "all_column_atom_coverage_count": coverage,
            "initial_mean_atom_cosine": 0.5 + 0.1 * coverage,
        }
    coverage_groups = grouped_initialization_success(
        runs, "all_column_atom_coverage_count"
    )
    assert [group["value"] for group in coverage_groups] == [2, 3, 4]
    assert coverage_groups[-1]["strict_success_count"] == 1

    result = simulate_restart_budget(
        runs,
        budget=4,
        n_trials=20,
        rng=np.random.default_rng(5),
    )
    assert result["group_contains_strict_success_probability"] == 1.0
    assert result["selector_selects_strict_success_probability"] == 1.0
    assert result["selection_efficiency"] == 1.0
    assert classify_r5({
        "group_contains_strict_success_probability": 0.95,
        "selector_selects_strict_success_probability": 0.92,
        "selection_efficiency": 0.968,
    })["classification"] == "A_MANAGEABLE_LOCAL_OPTIMUM"

    print("e1_t2_initialization_audit self-tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
