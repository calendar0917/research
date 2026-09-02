"""Controlled E1 datasets for auditing singleton-observation frequency."""
from __future__ import annotations

from typing import Any

import numpy as np

from .from_scratch_recovery import (
    RecoveryDataset,
    make_fixed_slot_graph_atoms,
    sample_sparse_codes,
)


def code_cardinality_summary(X: np.ndarray) -> dict[str, Any]:
    cardinalities = np.count_nonzero(np.abs(X) > 1e-12, axis=0)
    values, counts = np.unique(cardinalities, return_counts=True)
    histogram = {str(int(value)): int(count) for value, count in zip(values, counts)}
    singleton_count = int(np.sum(cardinalities == 1))
    pair_count = int(np.sum(cardinalities == 2))
    return {
        "sample_count": int(X.shape[1]),
        "singleton_count": singleton_count,
        "singleton_rate": singleton_count / max(X.shape[1], 1),
        "pair_count": pair_count,
        "pair_rate": pair_count / max(X.shape[1], 1),
        "cardinality_histogram": histogram,
    }


def make_e1_singleton_frequency_dataset(
    *,
    seed: int = 20260731,
    singleton_probability: float = 0.2,
    n_train: int = 1000,
    n_test: int = 300,
) -> RecoveryDataset:
    if not 0.0 <= singleton_probability <= 1.0:
        raise ValueError("singleton_probability must lie in [0, 1]")
    rng = np.random.default_rng(seed)
    D_true, scales, supports, atom_meta = make_fixed_slot_graph_atoms(6)
    n_atoms = D_true.shape[1]
    X_train = sample_sparse_codes(
        n_atoms=n_atoms,
        n_samples=n_train,
        max_sparsity=2,
        rng=rng,
        singleton_probability=singleton_probability,
        coefficient_scales=scales,
        signed=False,
        random_amplitude=False,
    )
    X_test = sample_sparse_codes(
        n_atoms=n_atoms,
        n_samples=n_test,
        max_sparsity=2,
        rng=rng,
        singleton_probability=singleton_probability,
        coefficient_scales=scales,
        signed=False,
        random_amplitude=False,
    )
    Y_train = D_true @ X_train
    Y_test = D_true @ X_test
    if not np.all((Y_train == 0.0) | (Y_train == 1.0)):
        raise RuntimeError("E1 train signals must be binary adjacency vectors")
    if not np.all((Y_test == 0.0) | (Y_test == 1.0)):
        raise RuntimeError("E1 test signals must be binary adjacency vectors")

    train_cardinality = code_cardinality_summary(X_train)
    test_cardinality = code_cardinality_summary(X_test)
    return RecoveryDataset(
        name=f"E1_fixed_slot_singleton_p{singleton_probability:g}",
        D_true=D_true,
        X_train_true=X_train,
        Y_train=Y_train,
        X_test_true=X_test,
        Y_test=Y_test,
        metadata={
            "seed": seed,
            "n_atoms": n_atoms,
            "max_sparsity": 2,
            "n_train": n_train,
            "n_test": n_test,
            "singleton_probability": singleton_probability,
            "guaranteed_singleton_per_atom": 1,
            "coefficient_rule": "sqrt(atom_edge_count)",
            "train_code_cardinality": train_cardinality,
            "test_code_cardinality": test_cardinality,
            **atom_meta,
        },
        binary_graph_signal=True,
        true_atom_edge_supports=supports,
    )
