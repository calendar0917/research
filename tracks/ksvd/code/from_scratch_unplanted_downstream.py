"""U1A graph-level controls for the unplanted walk-patch KSVD route."""
from __future__ import annotations

from typing import Any

import numpy as np

from .from_scratch_unplanted_dictionary import encode_with_dictionary, stack_walk_patch_matrix
from .from_scratch_unplanted_signal import (
    GraphPatchExample,
    evaluate_feature_matrices,
)


CODE_CONTROLS = ("fixed_gaussian", "medoid_bag", "pca12", "init_codes", "final_codes")
RAW_CONTROLS = (
    "simple_graph_statistics",
    "walk_mean_std",
    "canonical_mean_std",
    "edge_count_histogram",
)


def graph_code_readout(codes: np.ndarray, graph_count: int, patches_per_graph: int) -> np.ndarray:
    codes = np.asarray(codes, dtype=np.float64)
    if codes.ndim != 2 or codes.shape[1] != graph_count * patches_per_graph:
        raise ValueError("code matrix does not match graph/patch counts")
    n_atoms = codes.shape[0]
    grouped = codes.reshape(n_atoms, graph_count, patches_per_graph).transpose(1, 0, 2)
    absolute = np.abs(grouped)
    frequency = np.mean(absolute > 1e-10, axis=2)
    mean_absolute = np.mean(absolute, axis=2)
    rms = np.sqrt(np.mean(grouped**2, axis=2))
    return np.concatenate([frequency, mean_absolute, rms], axis=1)


def _labels(dataset: dict[str, tuple[GraphPatchExample, ...]]) -> dict[str, np.ndarray]:
    return {
        split: np.asarray([example.label for example in examples], dtype=np.int64)
        for split, examples in dataset.items()
    }


def _raw_graph_features(
    dataset: dict[str, tuple[GraphPatchExample, ...]], key: str
) -> dict[str, np.ndarray]:
    if key == "simple_graph_statistics":
        return {
            split: np.stack([example.simple_graph_statistics for example in examples], axis=0)
            for split, examples in dataset.items()
        }
    return {
        split: np.stack([example.features[key] for example in examples], axis=0)
        for split, examples in dataset.items()
    }


def _medoid_one_hot_codes(values: np.ndarray, medoids: np.ndarray) -> np.ndarray:
    value_norms = np.sum(values**2, axis=0)[:, None]
    medoid_norms = np.sum(medoids**2, axis=0)[None, :]
    distances = value_norms + medoid_norms - 2.0 * values.T @ medoids
    winners = np.argmin(distances, axis=1)
    codes = np.zeros((medoids.shape[1], values.shape[1]), dtype=np.float64)
    codes[winners, np.arange(values.shape[1])] = 1.0
    return codes


def build_u1a_features(
    dataset: dict[str, tuple[GraphPatchExample, ...]],
    *,
    initial_dictionary: np.ndarray,
    final_dictionary: np.ndarray,
    selected_training_indices: list[int],
    n_atoms: int = 12,
    sparsity: int = 2,
    minimum_sparsity: int = 1,
    patches_per_graph: int = 24,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    labels = _labels(dataset)
    raw_patches = {split: stack_walk_patch_matrix(examples) for split, examples in dataset.items()}
    train_mean = np.mean(raw_patches["train"], axis=1, keepdims=True)
    centered = {split: values - train_mean for split, values in raw_patches.items()}
    if not np.allclose(train_mean[:, 0], np.mean(raw_patches["train"], axis=1)):
        raise RuntimeError("train centering failed")

    rng = np.random.default_rng(0)
    gaussian = rng.standard_normal((centered["train"].shape[0], n_atoms))
    gaussian /= np.linalg.norm(gaussian, axis=0, keepdims=True)
    u, _singular, _vt = np.linalg.svd(centered["train"], full_matrices=False)
    pca_basis = u[:, :n_atoms]
    medoids = centered["train"][:, np.asarray(selected_training_indices, dtype=np.int64)]

    features: dict[str, dict[str, np.ndarray]] = {
        key: _raw_graph_features(dataset, key) for key in RAW_CONTROLS
    }
    for split in ("train", "validation", "test"):
        graph_count = len(dataset[split])
        code_matrices = {
            "fixed_gaussian": encode_with_dictionary(
                centered[split], gaussian, sparsity=sparsity, minimum_sparsity=minimum_sparsity
            ),
            "medoid_bag": _medoid_one_hot_codes(centered[split], medoids),
            "pca12": pca_basis.T @ centered[split],
            "init_codes": encode_with_dictionary(
                centered[split], initial_dictionary, sparsity=sparsity, minimum_sparsity=minimum_sparsity
            ),
            "final_codes": encode_with_dictionary(
                centered[split], final_dictionary, sparsity=sparsity, minimum_sparsity=minimum_sparsity
            ),
        }
        for key, codes in code_matrices.items():
            features.setdefault(key, {})[split] = graph_code_readout(
                codes, graph_count, patches_per_graph
            )
    return features, labels


def permute_feature_rows(
    matrices: dict[str, np.ndarray], sequence: np.random.SeedSequence
) -> dict[str, np.ndarray]:
    result = {}
    for split, split_sequence in zip(("train", "validation", "test"), sequence.spawn(3)):
        values = np.asarray(matrices[split], dtype=np.float64)
        order = np.random.default_rng(split_sequence).permutation(values.shape[0])
        result[split] = values[order]
    return result


def permute_label_rows(
    labels: dict[str, np.ndarray], sequence: np.random.SeedSequence
) -> dict[str, np.ndarray]:
    result = {}
    for split, split_sequence in zip(("train", "validation", "test"), sequence.spawn(3)):
        values = np.asarray(labels[split], dtype=np.int64)
        order = np.random.default_rng(split_sequence).permutation(values.size)
        result[split] = values[order]
    return result


def evaluate_u1a_features(
    features: dict[str, dict[str, np.ndarray]],
    labels: dict[str, np.ndarray],
    *,
    graph_shuffle_sequence: np.random.SeedSequence,
    label_shuffle_sequence: np.random.SeedSequence,
) -> dict[str, Any]:
    results = {
        key: evaluate_feature_matrices(matrices, labels, feature_key=key)
        for key, matrices in features.items()
    }
    shuffled_features = permute_feature_rows(features["final_codes"], graph_shuffle_sequence)
    results["graph_code_shuffle"] = evaluate_feature_matrices(
        shuffled_features, labels, feature_key="graph_code_shuffle"
    )
    shuffled_labels = permute_label_rows(labels, label_shuffle_sequence)
    results["label_shuffle"] = evaluate_feature_matrices(
        features["final_codes"], shuffled_labels, feature_key="label_shuffle"
    )
    return results
