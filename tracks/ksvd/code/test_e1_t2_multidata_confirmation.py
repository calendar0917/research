"""Fast contract tests for E1-T2 cross-data-seed confirmation helpers."""
from __future__ import annotations

from .run_from_scratch_e1_t2_multidata_confirmation import (
    aggregate_results,
    classify_confirmation,
    oracle_passes,
    select_by_train_error,
    summarize_data_seed,
)


def fake_run(seed: int, train: float, success: bool) -> dict:
    atom = 1.0 if success else 0.8
    return {
        "learner_seed": seed,
        "strict_success": success,
        "train_reconstruction_relative": train,
        "test_reconstruction_relative": 0.0 if success else 0.2,
        "mean_atom_cosine": atom,
        "minimum_atom_cosine": atom,
        "support_f1": 1.0 if success else 0.7,
        "mean_atom_edge_support_f1": 1.0 if success else 0.7,
        "edge_f1": 1.0,
        "exact_patch_recovery": 1.0,
    }


def fake_oracle() -> dict:
    return {
        "test_reconstruction_relative": 0.0,
        "support_f1": 1.0,
        "edge_f1": 1.0,
        "exact_patch_recovery": 1.0,
    }


def main() -> int:
    oracle = fake_oracle()
    assert oracle_passes(oracle)
    runs = [fake_run(0, 0.2, False), fake_run(1, 0.01, True)]
    assert select_by_train_error(runs)["learner_seed"] == 1
    result = summarize_data_seed(data_seed=7, oracle=oracle, runs=runs)
    assert result["group_contains_strict_success"]
    assert result["selected_strict_success"]
    assert not result["selection_miss"]

    results = []
    for data_seed in range(10):
        candidates = [
            fake_run(0, 0.2, False),
            fake_run(1, 0.01, True),
        ]
        results.append(
            summarize_data_seed(data_seed=data_seed, oracle=oracle, runs=candidates)
        )
    summary = aggregate_results(results)
    decision = classify_confirmation(summary)
    assert decision["passed"]
    assert decision["classification"] == "PASS_FREEZE_FIVE_RESTART_RULE"

    missed = summarize_data_seed(
        data_seed=99,
        oracle=oracle,
        runs=[fake_run(0, 0.001, False), fake_run(1, 0.01, True)],
    )
    assert missed["selection_miss"]

    print("e1_t2_multidata_confirmation self-tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
