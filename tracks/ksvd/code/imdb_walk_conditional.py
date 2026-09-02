"""Statistics-conditioned residual KSVD audit for raw IMDB-BINARY."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .from_scratch_unplanted_dictionary import (
    ACTIVATION_THRESHOLD,
    deterministic_maximin_initialization,
    dictionary_metrics,
)
from .from_scratch_unplanted_signal import evaluate_feature_matrices
from .imdb_walk_dictionary import (
    FoldSplit,
    encode_with_minimum_sparsity,
    stack_patch_graphs,
)
from .imdb_walk_downstream import variable_graph_code_readout
from .imdb_walk_substrate import IMDBPatchGraph, TUStructureGraph
from .ksvd import ksvd

EPS = 1e-12


def fit_statistics_residualizer(
    examples: Sequence[IMDBPatchGraph],
) -> dict[str, Any]:
    """Fit a graph-balanced linear predictor of each graph's patch mean."""
    examples = tuple(examples)
    if not examples:
        raise ValueError("examples cannot be empty")
    stats = np.stack([example.graph_statistics for example in examples], axis=0).astype(
        np.float64
    )
    patch_means = np.stack(
        [np.mean(np.asarray(example.walk_vectors, dtype=np.float64), axis=0) for example in examples],
        axis=0,
    )
    mean = np.mean(stats, axis=0)
    scale = np.std(stats, axis=0, ddof=0)
    active = scale > EPS
    safe_scale = scale.copy()
    safe_scale[~active] = 1.0
    standardized = (stats[:, active] - mean[active]) / safe_scale[active]
    design = np.column_stack([standardized, np.ones(len(examples), dtype=np.float64)])
    coefficients = np.linalg.lstsq(design, patch_means, rcond=None)[0]
    return {
        "stats_mean": mean,
        "stats_scale": safe_scale,
        "stats_active": active,
        "coefficients": coefficients,
        "stats_active_dimension": int(np.count_nonzero(active)),
        "target_dimension": int(patch_means.shape[1]),
    }


def predict_statistics_patch_mean(
    examples: Sequence[IMDBPatchGraph], residualizer: dict[str, Any]
) -> np.ndarray:
    """Predict one patch-coordinate mean per graph using frozen train coefficients."""
    examples = tuple(examples)
    stats = np.stack([example.graph_statistics for example in examples], axis=0).astype(
        np.float64
    )
    mean = residualizer["stats_mean"]
    scale = residualizer["stats_scale"]
    active = residualizer["stats_active"]
    standardized = (stats[:, active] - mean[active]) / scale[active]
    design = np.column_stack([standardized, np.ones(len(examples), dtype=np.float64)])
    return design @ residualizer["coefficients"]


def build_conditional_targets(
    examples: Sequence[IMDBPatchGraph], residualizer: dict[str, Any]
) -> tuple[np.ndarray, tuple[tuple[int, int, int, int], ...], np.ndarray, np.ndarray]:
    """Stack raw patches and subtract the frozen graph-specific predicted mean."""
    raw, slices = stack_patch_graphs(examples)
    predictions = predict_statistics_patch_mean(examples, residualizer)
    predicted_columns = np.concatenate(
        [np.repeat(predictions[index][:, None], stop - start, axis=1) for index, (_graph, _label, start, stop) in enumerate(slices)],
        axis=1,
    )
    return raw - predicted_columns, slices, raw, predicted_columns


def _graph_errors(
    values: np.ndarray,
    reconstruction: np.ndarray,
    slices: Sequence[tuple[int, int, int, int]],
) -> np.ndarray:
    errors = []
    for _graph_index, _label, start, stop in slices:
        signal = values[:, start:stop]
        residual = signal - reconstruction[:, start:stop]
        errors.append(float(np.linalg.norm(residual, "fro") / max(np.linalg.norm(signal, "fro"), EPS)))
    return np.asarray(errors, dtype=np.float64)


def _stage(
    targets: dict[str, np.ndarray],
    slices: dict[str, Sequence[tuple[int, int, int, int]]],
    dictionary: np.ndarray,
    *,
    sparsity: int,
    minimum_sparsity: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    result: dict[str, Any] = {}
    codes_by_split: dict[str, np.ndarray] = {}
    recon_by_split: dict[str, np.ndarray] = {}
    for split in ("train", "validation", "test"):
        codes = encode_with_minimum_sparsity(
            targets[split], dictionary, sparsity=sparsity, minimum_sparsity=minimum_sparsity
        )
        reconstruction = dictionary @ codes
        codes_by_split[split] = codes
        recon_by_split[split] = reconstruction
        graph_errors = _graph_errors(targets[split], reconstruction, slices[split])
        result[split] = {
            "dictionary_health": dictionary_metrics(targets[split], dictionary, codes),
            "graph_balanced_mean_relative_error": float(np.mean(graph_errors)),
            "graph_balanced_errors": graph_errors.tolist(),
            "patch_weighted_relative_error": float(
                np.linalg.norm(targets[split] - reconstruction, "fro")
                / max(np.linalg.norm(targets[split], "fro"), EPS)
            ),
        }
    return result, codes_by_split, recon_by_split


def _labels(examples: Sequence[IMDBPatchGraph]) -> np.ndarray:
    return np.asarray([int(example.label) for example in examples], dtype=np.int64)


def _stats(examples: Sequence[IMDBPatchGraph]) -> np.ndarray:
    return np.stack([example.graph_statistics for example in examples], axis=0).astype(np.float64)


def _features_from_codes(
    codes_by_split: dict[str, np.ndarray],
    slices: dict[str, Sequence[tuple[int, int, int, int]]],
) -> dict[str, np.ndarray]:
    return {
        split: variable_graph_code_readout(codes_by_split[split], slices[split])
        for split in ("train", "validation", "test")
    }


def _shuffle_rows(values: np.ndarray, seed: int) -> np.ndarray:
    values = np.asarray(values)
    return values[np.random.default_rng(int(seed)).permutation(values.shape[0])]


def run_conditional_fold(
    graphs: Sequence[TUStructureGraph],
    examples: Sequence[IMDBPatchGraph],
    outer_split: FoldSplit,
    inner_split: FoldSplit,
    *,
    n_atoms: int = 12,
    sparsity: int = 2,
    minimum_sparsity: int = 1,
    n_iterations: int = 25,
    graph_shuffle_seed: int,
    label_shuffle_seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run paired standard and statistics-conditioned dictionary branches."""
    del graphs
    example_by_index = {int(example.graph_index): example for example in examples}
    outer_train = tuple(example_by_index[index] for index in outer_split.train_indices)
    split_indices = {
        "train": tuple(inner_split.train_indices),
        "validation": tuple(inner_split.test_indices),
        "test": tuple(outer_split.test_indices),
    }
    split_examples = {
        split: tuple(example_by_index[index] for index in indices)
        for split, indices in split_indices.items()
    }
    residualizer = fit_statistics_residualizer(outer_train)
    residual_targets: dict[str, np.ndarray] = {}
    raw_targets: dict[str, np.ndarray] = {}
    predicted_means: dict[str, np.ndarray] = {}
    slices: dict[str, tuple[tuple[int, int, int, int], ...]] = {}
    for split, items in split_examples.items():
        residual, split_slices, raw, predicted = build_conditional_targets(items, residualizer)
        residual_targets[split] = residual
        raw_targets[split] = raw
        predicted_means[split] = predicted
        slices[split] = split_slices
    outer_raw, _outer_slices = stack_patch_graphs(outer_train)
    outer_graph_predictions = predict_statistics_patch_mean(outer_train, residualizer)
    outer_predicted = np.concatenate(
        [
            np.repeat(
                outer_graph_predictions[index][:, None],
                stop - start,
                axis=1,
            )
            for index, (_graph, _label, start, stop) in enumerate(_outer_slices)
        ],
        axis=1,
    )
    standard_mean = np.mean(outer_raw, axis=1, keepdims=True)
    outer_standard_target = outer_raw - standard_mean
    outer_residual_target = outer_raw - outer_predicted
    standard_targets = {
        split: raw_targets[split] - standard_mean for split in raw_targets
    }

    dictionaries: dict[str, np.ndarray] = {}
    initial_standard, init_standard_info = deterministic_maximin_initialization(
        outer_standard_target, n_atoms
    )
    final_standard, _codes, standard_training = ksvd(
        outer_standard_target,
        n_atoms=n_atoms,
        T=sparsity,
        T_min=minimum_sparsity,
        n_iter=n_iterations,
        seed=0,
        initial_dictionary=initial_standard,
        coherence_step=0.0,
        anchor_strength=0.0,
    )
    initial_residual, init_residual_info = deterministic_maximin_initialization(
        outer_residual_target, n_atoms
    )
    final_residual, _codes, residual_training = ksvd(
        outer_residual_target,
        n_atoms=n_atoms,
        T=sparsity,
        T_min=minimum_sparsity,
        n_iter=n_iterations,
        seed=0,
        initial_dictionary=initial_residual,
        coherence_step=0.0,
        anchor_strength=0.0,
    )
    dictionaries.update(
        {
            "standard_init": initial_standard,
            "standard_final": final_standard,
            "residual_init": initial_residual,
            "residual_final": final_residual,
        }
    )
    standard_stages: dict[str, Any] = {}
    residual_stages: dict[str, Any] = {}
    standard_codes: dict[str, dict[str, np.ndarray]] = {}
    residual_codes: dict[str, dict[str, np.ndarray]] = {}
    standard_recons: dict[str, dict[str, np.ndarray]] = {}
    residual_recons: dict[str, dict[str, np.ndarray]] = {}
    for name, dictionary in (("init", initial_standard), ("final", final_standard)):
        standard_stages[name], standard_codes[name], standard_recons[name] = _stage(
            standard_targets, slices, dictionary,
            sparsity=sparsity, minimum_sparsity=minimum_sparsity,
        )
    for name, dictionary in (("init", initial_residual), ("final", final_residual)):
        residual_stages[name], residual_codes[name], residual_recons[name] = _stage(
            residual_targets, slices, dictionary,
            sparsity=sparsity, minimum_sparsity=minimum_sparsity,
        )

    labels = {split: _labels(items) for split, items in split_examples.items()}
    stats = {split: _stats(items) for split, items in split_examples.items()}
    features: dict[str, dict[str, np.ndarray]] = {
        "stats": stats,
        "standard_init": _features_from_codes(standard_codes["init"], slices),
        "standard_final": _features_from_codes(standard_codes["final"], slices),
        "residual_init": _features_from_codes(residual_codes["init"], slices),
        "residual_final": _features_from_codes(residual_codes["final"], slices),
    }
    features.update(
        {
            "stats_plus_standard_init": {
                split: np.column_stack([stats[split], features["standard_init"][split]])
                for split in stats
            },
            "stats_plus_standard_final": {
                split: np.column_stack([stats[split], features["standard_final"][split]])
                for split in stats
            },
            "stats_plus_residual_init": {
                split: np.column_stack([stats[split], features["residual_init"][split]])
                for split in stats
            },
            "stats_plus_residual_final": {
                split: np.column_stack([stats[split], features["residual_final"][split]])
                for split in stats
            },
        }
    )
    evaluations = {
        key: evaluate_feature_matrices(value, labels, feature_key=key)
        for key, value in features.items()
    }
    shuffled_residual = {
        split: _shuffle_rows(features["residual_final"][split], seed)
        for split, seed in zip(
            ("train", "validation", "test"),
            [graph_shuffle_seed, graph_shuffle_seed + 1, graph_shuffle_seed + 2],
        )
    }
    evaluations["stats_plus_shuffled_residual_final"] = evaluate_feature_matrices(
        {
            split: np.column_stack([stats[split], shuffled_residual[split]])
            for split in stats
        },
        labels,
        feature_key="stats_plus_shuffled_residual_final",
    )
    shuffled_labels = {
        split: _shuffle_rows(labels[split], seed)
        for split, seed in zip(
            ("train", "validation", "test"),
            [label_shuffle_seed, label_shuffle_seed + 1, label_shuffle_seed + 2],
        )
    }
    evaluations["label_shuffle_stats_plus_residual_final"] = evaluate_feature_matrices(
        features["stats_plus_residual_final"],
        shuffled_labels,
        feature_key="label_shuffle_stats_plus_residual_final",
    )

    residual_test_init = residual_stages["init"]["test"]["graph_balanced_mean_relative_error"]
    residual_test_final = residual_stages["final"]["test"]["graph_balanced_mean_relative_error"]
    full_patch_reconstruction: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        full_patch_reconstruction[split] = {
            "predictor_only_graph_balanced_mean_relative_error": float(
                np.mean(_graph_errors(raw_targets[split], predicted_means[split], slices[split]))
            ),
            "standard_init_graph_balanced_mean_relative_error": float(
                np.mean(
                    _graph_errors(
                        raw_targets[split],
                        standard_mean + standard_recons["init"][split],
                        slices[split],
                    )
                )
            ),
            "standard_final_graph_balanced_mean_relative_error": float(
                np.mean(
                    _graph_errors(
                        raw_targets[split],
                        standard_mean + standard_recons["final"][split],
                        slices[split],
                    )
                )
            ),
            "residual_init_graph_balanced_mean_relative_error": float(
                np.mean(
                    _graph_errors(
                        raw_targets[split],
                        predicted_means[split] + residual_recons["init"][split],
                        slices[split],
                    )
                )
            ),
            "residual_final_graph_balanced_mean_relative_error": float(
                np.mean(
                    _graph_errors(
                        raw_targets[split],
                        predicted_means[split] + residual_recons["final"][split],
                        slices[split],
                    )
                )
            ),
        }
    result = {
        "fold_index": int(outer_split.fold_index),
        "outer_train_graph_count": int(len(outer_split.train_indices)),
        "inner_train_graph_count": int(len(inner_split.train_indices)),
        "validation_graph_count": int(len(inner_split.test_indices)),
        "test_graph_count": int(len(outer_split.test_indices)),
        "outer_train_patch_count": int(outer_raw.shape[1]),
        "split_patch_counts": {split: int(raw_targets[split].shape[1]) for split in raw_targets},
        "residualizer": {
            "stats_active_dimension": int(residualizer["stats_active_dimension"]),
            "target_dimension": int(residualizer["target_dimension"]),
            "train_graph_mean_prediction_relative_error": float(
                np.mean(
                    np.linalg.norm(
                        np.stack([item.walk_vectors.mean(axis=0) for item in outer_train])
                        - predict_statistics_patch_mean(outer_train, residualizer),
                        axis=1,
                    )
                )
            ),
        },
        "config": {
            "patch_dimension": int(outer_raw.shape[0]),
            "n_atoms": int(n_atoms),
            "sparsity": int(sparsity),
            "minimum_sparsity": int(minimum_sparsity),
            "n_iterations": int(n_iterations),
            "readout_dimension": int(3 * n_atoms),
            "readout": ["activation_frequency", "mean_absolute", "rms"],
            "residual_target": "raw_patch_minus_outer_train_stats_predicted_graph_patch_mean",
            "residualizer_fit": "graph_balanced_outer_train_multivariate_least_squares",
            "restarts": 0,
            "graph_shuffle_seed": int(graph_shuffle_seed),
            "label_shuffle_seed": int(label_shuffle_seed),
        },
        "initializations": {
            "standard": init_standard_info,
            "residual": init_residual_info,
        },
        "training_reconstruction_curves": {
            "standard": [float(value) for value in standard_training["recon_curve"]],
            "residual": [float(value) for value in residual_training["recon_curve"]],
        },
        "stages": {
            "standard": standard_stages,
            "residual": residual_stages,
        },
        "full_patch_reconstruction": full_patch_reconstruction,
        "evaluations": evaluations,
        "attribution": {
            "residual_graph_balanced_relative_reduction": float(
                (residual_test_init - residual_test_final) / max(residual_test_init, EPS)
            ),
            "residual_update_gain": float(
                evaluations["stats_plus_residual_final"]["test_balanced_accuracy"]
                - evaluations["stats_plus_residual_init"]["test_balanced_accuracy"]
            ),
            "residual_beyond_stats_gain": float(
                evaluations["stats_plus_residual_final"]["test_balanced_accuracy"]
                - evaluations["stats"]["test_balanced_accuracy"]
            ),
            "residual_over_standard_final_gain": float(
                evaluations["stats_plus_residual_final"]["test_balanced_accuracy"]
                - evaluations["stats_plus_standard_final"]["test_balanced_accuracy"]
            ),
            "residual_alignment_gain_over_shuffle": float(
                evaluations["stats_plus_residual_final"]["test_balanced_accuracy"]
                - evaluations["stats_plus_shuffled_residual_final"]["test_balanced_accuracy"]
            ),
        },
    }
    return result, dictionaries
