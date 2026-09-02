"""Self-tests for U1A graph-level code construction and evaluation."""
from __future__ import annotations

import numpy as np

from .from_scratch_unplanted_dictionary import run_u0d_dictionary_audit
from .from_scratch_unplanted_downstream import (
    build_u1a_features,
    evaluate_u1a_features,
    graph_code_readout,
)
from .from_scratch_unplanted_signal import generate_u0p_dataset


def main() -> int:
    codes = np.asarray([[1, 0, 2, 0], [0, 3, 0, 4]], dtype=np.float64)
    readout = graph_code_readout(codes, graph_count=2, patches_per_graph=2)
    assert readout.shape == (2, 6)
    assert np.allclose(readout[:, :2], 0.5)

    root = np.random.SeedSequence(731102)
    data_sequence, _u0p_shuffle, graph_shuffle, label_shuffle = root.spawn(4)
    dataset = generate_u0p_dataset(
        data_sequence,
        train_per_class=5,
        validation_per_class=2,
        test_per_class=3,
        patches_per_graph=6,
    )
    audit, initial, final = run_u0d_dictionary_audit(
        dataset, n_atoms=6, sparsity=2, minimum_sparsity=1, n_iterations=2
    )
    features, labels = build_u1a_features(
        dataset,
        initial_dictionary=initial,
        final_dictionary=final,
        selected_training_indices=audit["initialization"]["selected_training_indices"],
        n_atoms=6,
        sparsity=2,
        minimum_sparsity=1,
        patches_per_graph=6,
    )
    assert features["final_codes"]["train"].shape == (10, 18)
    assert features["simple_graph_statistics"]["train"].shape == (10, 12)
    results = evaluate_u1a_features(
        features,
        labels,
        graph_shuffle_sequence=graph_shuffle,
        label_shuffle_sequence=label_shuffle,
    )
    assert "init_codes" in results and "final_codes" in results
    assert 0.0 <= results["final_codes"]["test_balanced_accuracy"] <= 1.0
    print("u1a_graph_code_signal self-tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
