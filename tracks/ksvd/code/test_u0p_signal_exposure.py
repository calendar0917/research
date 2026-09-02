"""Self-tests for the U0-P signal-exposure implementation."""
from __future__ import annotations

import numpy as np

from .from_scratch_unplanted_signal import (
    FEATURE_KEYS,
    balanced_accuracy,
    evaluate_feature_control,
    fit_l2_logistic,
    generate_u0p_dataset,
    predict_l2_logistic,
    shuffled_labels,
)


def main() -> int:
    rng = np.random.default_rng(12)
    negative = rng.normal(loc=-1.0, scale=0.4, size=(80, 2))
    positive = rng.normal(loc=1.0, scale=0.4, size=(80, 2))
    values = np.vstack([negative, positive])
    labels = np.concatenate([np.zeros(80, dtype=np.int64), np.ones(80, dtype=np.int64)])
    model = fit_l2_logistic(values, labels, regularization=1e-2)
    predictions = predict_l2_logistic(model, values)
    assert model["converged"]
    assert balanced_accuracy(labels, predictions) > 0.98

    dataset = generate_u0p_dataset(
        731101,
        train_per_class=4,
        validation_per_class=2,
        test_per_class=3,
        patches_per_graph=6,
    )
    assert {key: len(value) for key, value in dataset.items()} == {
        "train": 8,
        "validation": 4,
        "test": 6,
    }
    for examples in dataset.values():
        for example in examples:
            assert example.walk_vectors.shape == (6, 15)
            assert example.canonical_vectors.shape == (6, 15)
            assert set(example.features) == set(FEATURE_KEYS)
            assert np.isclose(example.features["edge_count_histogram"].sum(), 1.0)
            if example.label == 0:
                assert 20 <= example.requested_swaps <= 40
            else:
                assert 60 <= example.requested_swaps <= 80

    repeated = generate_u0p_dataset(
        731101,
        train_per_class=4,
        validation_per_class=2,
        test_per_class=3,
        patches_per_graph=6,
    )
    for split in dataset:
        for left, right in zip(dataset[split], repeated[split]):
            assert left.label == right.label
            assert left.requested_swaps == right.requested_swaps
            assert left.graph_seed == right.graph_seed
            assert np.array_equal(left.walk_vectors, right.walk_vectors)
            assert np.array_equal(left.canonical_vectors, right.canonical_vectors)

    result = evaluate_feature_control(dataset, "walk_mean_std")
    assert result["raw_dimension"] == 30
    assert 0.0 <= result["test_balanced_accuracy"] <= 1.0
    shuffle = shuffled_labels(dataset, np.random.SeedSequence(99))
    shuffled_result = evaluate_feature_control(
        dataset, "walk_mean_std", label_override=shuffle
    )
    assert shuffled_result["class_counts"]["train"] == {"0": 4, "1": 4}
    print("u0p_signal_exposure self-tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
