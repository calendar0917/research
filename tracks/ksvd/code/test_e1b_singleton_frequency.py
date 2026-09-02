"""Fast tests for controlled E1 singleton-frequency experiments."""
from __future__ import annotations

import numpy as np

from .from_scratch_recovery import evaluate_dictionary
from .from_scratch_singleton_frequency import (
    code_cardinality_summary,
    make_e1_singleton_frequency_dataset,
)
from .run_from_scratch_e1b_s20 import aggregate_results, classify, summarize_data_seed


def fake_run(seed: int, success: bool) -> dict:
    atom = 1.0 if success else 0.8
    return {
        "learner_seed": seed,
        "strict_success": success,
        "train_reconstruction_relative": 0.001 if success else 0.2,
        "test_reconstruction_relative": 0.001 if success else 0.2,
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
    codes = np.asarray([[1, 0, 1], [0, 1, 1]], dtype=np.float64)
    cardinality = code_cardinality_summary(codes)
    assert cardinality["singleton_count"] == 2
    assert cardinality["pair_count"] == 1

    dataset = make_e1_singleton_frequency_dataset(
        seed=31,
        singleton_probability=0.2,
        n_train=200,
        n_test=100,
    )
    assert np.all((dataset.Y_train == 0.0) | (dataset.Y_train == 1.0))
    train_rate = dataset.metadata["train_code_cardinality"]["singleton_rate"]
    assert 0.10 <= train_rate <= 0.30
    assert dataset.metadata["train_code_cardinality"]["singleton_count"] >= 4
    _oracle_x, oracle = evaluate_dictionary(dataset, dataset.D_true, T=2)
    assert oracle["support_f1"] > 0.999999
    assert oracle["exact_patch_recovery"] > 0.999999

    results = []
    for data_seed in range(10):
        metadata = {
            "train_code_cardinality": {"singleton_rate": 0.2},
            "test_code_cardinality": {"singleton_rate": 0.2},
        }
        runs = [fake_run(0, False), fake_run(1, True)]
        results.append(
            summarize_data_seed(
                data_seed=data_seed,
                dataset_metadata=metadata,
                oracle=fake_oracle(),
                runs=runs,
            )
        )
    summary = aggregate_results(results)
    decision = classify(summary)
    assert decision["passed"]
    assert decision["classification"] == "PASS_CONTINUE_TO_SINGLETON_P005"

    print("e1b_singleton_frequency self-tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
