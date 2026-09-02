"""Per-graph dictionary descriptors for the raw IMDB-BINARY falsification route.

Each graph is factorized independently.  The primary readout is invariant to
dictionary-column / sparse-code-row permutations; a legacy ordered readout is
retained only as a historical sensitivity control.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .from_scratch_unplanted_dictionary import (
    ACTIVATION_THRESHOLD,
    EPS,
    deterministic_maximin_initialization,
)
from .from_scratch_unplanted_signal import evaluate_feature_matrices
from .imdb_walk_dictionary import FoldSplit, encode_with_minimum_sparsity
from .imdb_walk_substrate import IMDBPatchGraph
from .ksvd import ksvd


ATOM_SCALAR_NAMES = (
    "usage_frequency",
    "coefficient_mean_absolute",
    "coefficient_rms",
    "coefficient_maximum_absolute",
    "coefficient_energy_share",
    "dictionary_mean_absolute",
    "dictionary_std",
    "dictionary_maximum_absolute",
)


@dataclass(frozen=True)
class PerGraphDictionaryFeatures:
    graph_index: int
    label: int
    raw_patch: np.ndarray
    invariant_init: np.ndarray
    invariant_final: np.ndarray
    invariant_pca: np.ndarray
    legacy_init: np.ndarray
    legacy_final: np.ndarray
    legacy_final_atom_permuted: np.ndarray
    invariant_permutation_max_abs_difference: float
    init_reconstruction_error: float
    final_reconstruction_error: float
    pca_reconstruction_error: float
    init_mean_nnz: float
    final_mean_nnz: float
    training_reconstruction_curve: tuple[float, ...]


def _canonicalize_atom_signs(
    dictionary: np.ndarray, codes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve rank-one signs without changing reconstruction."""
    dictionary = np.asarray(dictionary, dtype=np.float64).copy()
    codes = np.asarray(codes, dtype=np.float64).copy()
    if dictionary.ndim != 2 or codes.shape[0] != dictionary.shape[1]:
        raise ValueError("dictionary and codes have incompatible shapes")
    for atom in range(dictionary.shape[1]):
        column = dictionary[:, atom]
        total = float(np.sum(column))
        flip = total < -1e-12
        if abs(total) <= 1e-12:
            pivot = int(np.argmax(np.abs(column)))
            flip = bool(column[pivot] < 0.0)
        if flip:
            dictionary[:, atom] *= -1.0
            codes[atom, :] *= -1.0
    return dictionary, codes


def _offdiagonal_absolute_summary(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if matrix.shape[0] <= 1:
        return np.zeros(4, dtype=np.float64)
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    values = np.abs(matrix[mask])
    return np.asarray(
        [np.mean(values), np.std(values), np.max(values), np.quantile(values, 0.90)],
        dtype=np.float64,
    )


def legacy_ordered_dictionary_readout(
    dictionary: np.ndarray, codes: np.ndarray
) -> np.ndarray:
    """Approximate the advisor script's atom-index-sensitive D/X/Gram readout."""
    dictionary, codes = _canonicalize_atom_signs(dictionary, codes)
    d_mean = np.mean(dictionary, axis=0)
    d_std = np.std(dictionary, axis=0, ddof=0)
    d_max = np.max(dictionary, axis=0)
    x_abs = np.abs(codes)
    x_mean = np.mean(x_abs, axis=1)
    x_std = np.std(codes, axis=1, ddof=0)
    x_max = np.max(x_abs, axis=1)
    gram = (codes @ codes.T) / max(codes.shape[1], 1)
    upper = gram[np.triu_indices(gram.shape[0])]
    return np.concatenate([d_mean, d_std, d_max, x_mean, x_std, x_max, upper])


def invariant_dictionary_readout(
    dictionary: np.ndarray,
    codes: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Permutation/sign-invariant fixed-length descriptor of one factorization."""
    dictionary = np.asarray(dictionary, dtype=np.float64)
    codes = np.asarray(codes, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if dictionary.ndim != 2 or values.ndim != 2:
        raise ValueError("dictionary and values must be matrices")
    if codes.shape != (dictionary.shape[1], values.shape[1]):
        raise ValueError("codes have the wrong shape")
    if dictionary.shape[0] != values.shape[0]:
        raise ValueError("dictionary and values must share a coordinate dimension")

    absolute = np.abs(codes)
    usage = np.mean(absolute > ACTIVATION_THRESHOLD, axis=1)
    coefficient_mean = np.mean(absolute, axis=1)
    coefficient_rms = np.sqrt(np.mean(codes**2, axis=1))
    coefficient_max = np.max(absolute, axis=1)
    energy = np.sum(codes**2, axis=1)
    energy_share = energy / max(float(np.sum(energy)), EPS)
    dictionary_mean = np.mean(np.abs(dictionary), axis=0)
    dictionary_std = np.std(dictionary, axis=0, ddof=0)
    dictionary_max = np.max(np.abs(dictionary), axis=0)
    atom_scalars = np.column_stack(
        [
            usage,
            coefficient_mean,
            coefficient_rms,
            coefficient_max,
            energy_share,
            dictionary_mean,
            dictionary_std,
            dictionary_max,
        ]
    )
    set_summary = np.concatenate(
        [
            np.mean(atom_scalars, axis=0),
            np.std(atom_scalars, axis=0, ddof=0),
            np.min(atom_scalars, axis=0),
            np.max(atom_scalars, axis=0),
        ]
    )

    spectra = np.concatenate(
        [
            np.sort(usage)[::-1],
            np.sort(coefficient_mean)[::-1],
            np.sort(coefficient_rms)[::-1],
            np.sort(energy_share)[::-1],
        ]
    )
    code_gram = (codes @ codes.T) / max(codes.shape[1], 1)
    dictionary_gram = dictionary.T @ dictionary
    eigen_spectra = np.concatenate(
        [
            np.sort(np.linalg.eigvalsh(code_gram))[::-1],
            np.sort(np.linalg.eigvalsh(dictionary_gram))[::-1],
        ]
    )
    residual = values - dictionary @ codes
    reconstruction_error = float(
        np.linalg.norm(residual, "fro") / max(np.linalg.norm(values, "fro"), EPS)
    )
    mean_nnz = float(np.mean(np.sum(absolute > ACTIVATION_THRESHOLD, axis=0)))
    tail = np.concatenate(
        [
            _offdiagonal_absolute_summary(dictionary_gram),
            _offdiagonal_absolute_summary(code_gram),
            np.asarray([reconstruction_error, mean_nnz], dtype=np.float64),
        ]
    )
    output = np.concatenate([set_summary, spectra, eigen_spectra, tail])
    if not np.all(np.isfinite(output)):
        raise RuntimeError("invariant dictionary readout produced non-finite values")
    return output


def _raw_patch_readout(example: IMDBPatchGraph) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(example.features["walk_mean_std"], dtype=np.float64),
            np.asarray(example.features["edge_count_histogram"], dtype=np.float64),
        ]
    )


def _relative_reconstruction_error(
    values: np.ndarray, dictionary: np.ndarray, codes: np.ndarray
) -> float:
    return float(
        np.linalg.norm(values - dictionary @ codes, "fro")
        / max(np.linalg.norm(values, "fro"), EPS)
    )


def extract_pergraph_dictionary_features(
    example: IMDBPatchGraph,
    *,
    n_atoms: int = 8,
    sparsity: int = 2,
    minimum_sparsity: int = 1,
    n_iterations: int = 10,
    atom_permutation_seed: int = 732141,
) -> PerGraphDictionaryFeatures:
    """Fit INIT/FINAL/PCA independently on a single graph's WALK patches."""
    values = np.asarray(example.walk_vectors, dtype=np.float64).T
    if values.shape[1] < n_atoms:
        raise ValueError("each graph must contain at least n_atoms patches")
    initial_dictionary, _initialization = deterministic_maximin_initialization(
        values, n_atoms
    )
    final_dictionary, _training_codes, training_info = ksvd(
        values,
        n_atoms=n_atoms,
        T=sparsity,
        T_min=minimum_sparsity,
        n_iter=n_iterations,
        seed=0,
        initial_dictionary=initial_dictionary,
        coherence_step=0.0,
        anchor_strength=0.0,
    )
    init_codes = encode_with_minimum_sparsity(
        values,
        initial_dictionary,
        sparsity=sparsity,
        minimum_sparsity=minimum_sparsity,
    )
    final_codes = encode_with_minimum_sparsity(
        values,
        final_dictionary,
        sparsity=sparsity,
        minimum_sparsity=minimum_sparsity,
    )
    left, _singular, _right = np.linalg.svd(values, full_matrices=False)
    if left.shape[1] < n_atoms:
        raise ValueError("per-graph patch matrix has insufficient PCA rank capacity")
    pca_dictionary = left[:, :n_atoms]
    pca_codes = pca_dictionary.T @ values

    invariant_init = invariant_dictionary_readout(
        initial_dictionary, init_codes, values
    )
    invariant_final = invariant_dictionary_readout(
        final_dictionary, final_codes, values
    )
    invariant_pca = invariant_dictionary_readout(
        pca_dictionary, pca_codes, values
    )
    legacy_init = legacy_ordered_dictionary_readout(initial_dictionary, init_codes)
    legacy_final = legacy_ordered_dictionary_readout(final_dictionary, final_codes)

    permutation = np.random.default_rng(
        int(atom_permutation_seed) + int(example.graph_index)
    ).permutation(n_atoms)
    permuted_dictionary = final_dictionary[:, permutation]
    permuted_codes = final_codes[permutation, :]
    invariant_permuted = invariant_dictionary_readout(
        permuted_dictionary, permuted_codes, values
    )
    legacy_permuted = legacy_ordered_dictionary_readout(
        permuted_dictionary, permuted_codes
    )
    return PerGraphDictionaryFeatures(
        graph_index=int(example.graph_index),
        label=int(example.label),
        raw_patch=_raw_patch_readout(example),
        invariant_init=invariant_init,
        invariant_final=invariant_final,
        invariant_pca=invariant_pca,
        legacy_init=legacy_init,
        legacy_final=legacy_final,
        legacy_final_atom_permuted=legacy_permuted,
        invariant_permutation_max_abs_difference=float(
            np.max(np.abs(invariant_final - invariant_permuted))
        ),
        init_reconstruction_error=_relative_reconstruction_error(
            values, initial_dictionary, init_codes
        ),
        final_reconstruction_error=_relative_reconstruction_error(
            values, final_dictionary, final_codes
        ),
        pca_reconstruction_error=_relative_reconstruction_error(
            values, pca_dictionary, pca_codes
        ),
        init_mean_nnz=float(
            np.mean(np.sum(np.abs(init_codes) > ACTIVATION_THRESHOLD, axis=0))
        ),
        final_mean_nnz=float(
            np.mean(np.sum(np.abs(final_codes) > ACTIVATION_THRESHOLD, axis=0))
        ),
        training_reconstruction_curve=tuple(
            float(value) for value in training_info["recon_curve"]
        ),
    )


def _permute_rows(values: np.ndarray, seed: int) -> np.ndarray:
    values = np.asarray(values)
    return values[np.random.default_rng(int(seed)).permutation(values.shape[0])]


def run_pergraph_downstream_fold(
    examples: Sequence[IMDBPatchGraph],
    descriptors: Sequence[PerGraphDictionaryFeatures],
    outer_split: FoldSplit,
    inner_split: FoldSplit,
    *,
    graph_shuffle_seed: int,
    label_shuffle_seed: int,
) -> dict[str, Any]:
    """Evaluate one frozen outer fold using precomputed label-free descriptors."""
    example_by_index = {int(item.graph_index): item for item in examples}
    descriptor_by_index = {int(item.graph_index): item for item in descriptors}
    outer_train = set(int(index) for index in outer_split.train_indices)
    if set(inner_split.train_indices) | set(inner_split.test_indices) != outer_train:
        raise ValueError("inner split must partition outer training graphs")
    split_indices = {
        "train": tuple(int(index) for index in inner_split.train_indices),
        "validation": tuple(int(index) for index in inner_split.test_indices),
        "test": tuple(int(index) for index in outer_split.test_indices),
    }

    labels: dict[str, np.ndarray] = {}
    blocks: dict[str, dict[str, np.ndarray]] = {
        key: {} for key in (
            "stats", "raw", "invariant_init", "invariant_final", "invariant_pca",
            "legacy_init", "legacy_final", "legacy_final_atom_permuted",
        )
    }
    diagnostics: dict[str, dict[str, np.ndarray]] = {
        key: {} for key in (
            "init_reconstruction_error", "final_reconstruction_error",
            "pca_reconstruction_error", "init_mean_nnz", "final_mean_nnz",
            "invariant_permutation_max_abs_difference",
        )
    }
    for split, indices in split_indices.items():
        labels[split] = np.asarray(
            [example_by_index[index].label for index in indices], dtype=np.int64
        )
        blocks["stats"][split] = np.stack(
            [example_by_index[index].graph_statistics for index in indices]
        )
        descriptor_attributes = {
            "raw": "raw_patch",
            "invariant_init": "invariant_init",
            "invariant_final": "invariant_final",
            "invariant_pca": "invariant_pca",
            "legacy_init": "legacy_init",
            "legacy_final": "legacy_final",
            "legacy_final_atom_permuted": "legacy_final_atom_permuted",
        }
        for key, attribute in descriptor_attributes.items():
            blocks[key][split] = np.stack(
                [
                    np.asarray(getattr(descriptor_by_index[index], attribute))
                    for index in indices
                ]
            )
        for key in diagnostics:
            diagnostics[key][split] = np.asarray(
                [float(getattr(descriptor_by_index[index], key)) for index in indices],
                dtype=np.float64,
            )

    def combine(*names: str) -> dict[str, np.ndarray]:
        return {
            split: np.column_stack([blocks[name][split] for name in names])
            for split in ("train", "validation", "test")
        }

    features = {
        "stats": combine("stats"),
        "stats_plus_raw": combine("stats", "raw"),
        "invariant_init": combine("invariant_init"),
        "invariant_final": combine("invariant_final"),
        "stats_plus_invariant_init": combine("stats", "invariant_init"),
        "stats_plus_invariant_final": combine("stats", "invariant_final"),
        "stats_raw_invariant_init": combine("stats", "raw", "invariant_init"),
        "stats_raw_invariant_final": combine("stats", "raw", "invariant_final"),
        "stats_raw_invariant_pca": combine("stats", "raw", "invariant_pca"),
        "stats_raw_legacy_init": combine("stats", "raw", "legacy_init"),
        "stats_raw_legacy_final": combine("stats", "raw", "legacy_final"),
        "stats_raw_legacy_final_atom_permuted": combine(
            "stats", "raw", "legacy_final_atom_permuted"
        ),
    }
    evaluations = {
        key: evaluate_feature_matrices(values, labels, feature_key=key)
        for key, values in features.items()
    }

    shuffle_sequences = np.random.SeedSequence(int(graph_shuffle_seed)).spawn(3)
    shuffled_final = {
        split: _permute_rows(
            blocks["invariant_final"][split],
            int(sequence.generate_state(1)[0]),
        )
        for split, sequence in zip(
            ("train", "validation", "test"), shuffle_sequences
        )
    }
    shuffled_feature = {
        split: np.column_stack(
            [blocks["stats"][split], blocks["raw"][split], shuffled_final[split]]
        )
        for split in ("train", "validation", "test")
    }
    evaluations["stats_raw_shuffled_invariant_final"] = evaluate_feature_matrices(
        shuffled_feature, labels, feature_key="stats_raw_shuffled_invariant_final"
    )

    label_sequences = np.random.SeedSequence(int(label_shuffle_seed)).spawn(3)
    shuffled_labels = {
        split: _permute_rows(labels[split], int(sequence.generate_state(1)[0]))
        for split, sequence in zip(
            ("train", "validation", "test"), label_sequences
        )
    }
    evaluations["label_shuffle_stats_raw_invariant_final"] = evaluate_feature_matrices(
        features["stats_raw_invariant_final"],
        shuffled_labels,
        feature_key="label_shuffle_stats_raw_invariant_final",
    )

    score = lambda key: float(evaluations[key]["test_balanced_accuracy"])
    test_init_error = diagnostics["init_reconstruction_error"]["test"]
    test_final_error = diagnostics["final_reconstruction_error"]["test"]
    return {
        "fold_index": int(outer_split.fold_index),
        "outer_train_graph_count": int(len(outer_split.train_indices)),
        "inner_train_graph_count": int(len(inner_split.train_indices)),
        "validation_graph_count": int(len(inner_split.test_indices)),
        "test_graph_count": int(len(outer_split.test_indices)),
        "feature_dimensions": {
            key: int(value["train"].shape[1]) for key, value in features.items()
        },
        "evaluations": evaluations,
        "reconstruction": {
            "test_mean_init": float(np.mean(test_init_error)),
            "test_mean_final": float(np.mean(test_final_error)),
            "test_mean_pca": float(
                np.mean(diagnostics["pca_reconstruction_error"]["test"])
            ),
            "test_graphs_final_better_count": int(
                np.count_nonzero(test_final_error < test_init_error - 1e-12)
            ),
            "test_graph_count": int(test_init_error.size),
            "test_mean_relative_reduction": float(
                np.mean(
                    (test_init_error - test_final_error)
                    / np.maximum(test_init_error, EPS)
                )
            ),
            "test_mean_init_nnz": float(
                np.mean(diagnostics["init_mean_nnz"]["test"])
            ),
            "test_mean_final_nnz": float(
                np.mean(diagnostics["final_mean_nnz"]["test"])
            ),
        },
        "invariance": {
            "test_maximum_atom_permutation_abs_difference": float(
                np.max(
                    diagnostics["invariant_permutation_max_abs_difference"]["test"]
                )
            ),
        },
        "attribution": {
            "primary_update_gain": float(
                score("stats_raw_invariant_final")
                - score("stats_raw_invariant_init")
            ),
            "beyond_stats_raw_gain": float(
                score("stats_raw_invariant_final") - score("stats_plus_raw")
            ),
            "pca_contrast": float(
                score("stats_raw_invariant_final")
                - score("stats_raw_invariant_pca")
            ),
            "aligned_over_row_shuffle": float(
                score("stats_raw_invariant_final")
                - score("stats_raw_shuffled_invariant_final")
            ),
            "legacy_update_gain": float(
                score("stats_raw_legacy_final") - score("stats_raw_legacy_init")
            ),
            "legacy_atom_order_sensitivity": float(
                score("stats_raw_legacy_final")
                - score("stats_raw_legacy_final_atom_permuted")
            ),
        },
    }


def summarize_pergraph_view(folds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    folds = tuple(folds)
    if not folds:
        raise ValueError("folds cannot be empty")
    evaluation_keys = tuple(folds[0]["evaluations"])
    scores = {
        key: np.asarray(
            [fold["evaluations"][key]["test_balanced_accuracy"] for fold in folds],
            dtype=np.float64,
        )
        for key in evaluation_keys
    }
    gains = np.asarray(
        [fold["attribution"]["primary_update_gain"] for fold in folds],
        dtype=np.float64,
    )
    reconstruction_positive = np.asarray(
        [
            fold["reconstruction"]["test_mean_final"]
            < fold["reconstruction"]["test_mean_init"] - 1e-12
            for fold in folds
        ],
        dtype=bool,
    )
    return {
        "fold_count": int(len(folds)),
        "feature_test_balanced_accuracy": {
            key: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=0)),
                "per_fold": values.tolist(),
            }
            for key, values in scores.items()
        },
        "primary_update_direction_count": int(np.count_nonzero(gains > 0.0)),
        "mean_primary_update_gain": float(np.mean(gains)),
        "per_fold_primary_update_gain": gains.tolist(),
        "reconstruction_positive_fold_count": int(np.count_nonzero(reconstruction_positive)),
        "mean_test_reconstruction_reduction": float(
            np.mean(
                [fold["reconstruction"]["test_mean_relative_reduction"] for fold in folds]
            )
        ),
        "maximum_invariant_permutation_difference": float(
            np.max(
                [
                    fold["invariance"]["test_maximum_atom_permutation_abs_difference"]
                    for fold in folds
                ]
            )
        ),
        "mean_aligned_over_row_shuffle": float(
            np.mean(
                [fold["attribution"]["aligned_over_row_shuffle"] for fold in folds]
            )
        ),
        "mean_pca_contrast": float(
            np.mean([fold["attribution"]["pca_contrast"] for fold in folds])
        ),
        "mean_beyond_stats_raw_gain": float(
            np.mean(
                [fold["attribution"]["beyond_stats_raw_gain"] for fold in folds]
            )
        ),
        "mean_legacy_atom_order_sensitivity": float(
            np.mean(
                [
                    fold["attribution"]["legacy_atom_order_sensitivity"]
                    for fold in folds
                ]
            )
        ),
        "mean_label_shuffle_test_balanced_accuracy": float(
            np.mean(scores["label_shuffle_stats_raw_invariant_final"])
        ),
    }
