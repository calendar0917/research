"""Fold-local R0-A/R0-B downstream attribution for raw IMDB-BINARY WALK codes.

The module keeps the registered R0-A mechanism audit separate from older
REAL_STRUCTURE and synthetic downstream pipelines.  Dictionaries and all
unsupervised controls are fitted on the outer training fold; the classifier
uses one deterministic inner train/validation split.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .from_scratch_unplanted_dictionary import (
    ACTIVATION_THRESHOLD,
    deterministic_maximin_initialization,
)
from .from_scratch_unplanted_signal import evaluate_feature_matrices
from .imdb_walk_dictionary import (
    FoldSplit,
    audit_fold_splits,
    deterministic_lloyd_medoids,
    encode_with_minimum_sparsity,
    grouped_isomorphism_folds,
    stack_patch_graphs,
    stratified_graph_folds,
)
from .imdb_walk_substrate import IMDBPatchGraph, IsomorphismGroup, TUStructureGraph
from .ksvd import ksvd


PRIMARY_FEATURE_KEYS = ("stats", "stats_plus_init", "stats_plus_final")
SECONDARY_FEATURE_KEYS = (
    "init",
    "final",
    "stats_plus_raw_walk",
    "stats_plus_pca12",
    "stats_plus_medoid_bag",
    "stats_plus_fixed_gaussian",
)


def variable_graph_code_readout(
    codes: np.ndarray,
    graph_slices: Sequence[tuple[int, int, int, int]],
) -> np.ndarray:
    """Pool variable-count patch codes into frequency/mean-abs/RMS graph codes."""
    codes = np.asarray(codes, dtype=np.float64)
    if codes.ndim != 2 or codes.shape[0] == 0 or codes.shape[1] == 0:
        raise ValueError("codes must be a non-empty atom-by-patch matrix")
    if not graph_slices:
        raise ValueError("graph_slices cannot be empty")
    rows = []
    expected_start = 0
    for _graph_index, _label, start, stop in graph_slices:
        if start != expected_start or not (0 <= start < stop <= codes.shape[1]):
            raise ValueError("graph slices must be contiguous and ordered")
        graph_codes = codes[:, start:stop]
        frequency = np.mean(np.abs(graph_codes) > ACTIVATION_THRESHOLD, axis=1)
        mean_absolute = np.mean(np.abs(graph_codes), axis=1)
        rms = np.sqrt(np.mean(graph_codes**2, axis=1))
        rows.append(np.concatenate([frequency, mean_absolute, rms]))
        expected_start = stop
    if expected_start != codes.shape[1]:
        raise ValueError("graph slices must cover every patch code exactly once")
    return np.stack(rows, axis=0)


def variable_graph_pair_readout(
    codes: np.ndarray,
    graph_slices: Sequence[tuple[int, int, int, int]],
) -> np.ndarray:
    """Pool within-patch atom-pair co-activation for variable-size graphs.

    The support threshold is the same one used by the marginal readout.  For
    each graph and each k<l, the feature is the fraction of patches in which
    both atoms are active.  This deliberately ignores atom signs and values:
    it is a support-composition readout, not a second dictionary objective.
    """
    codes = np.asarray(codes, dtype=np.float64)
    if codes.ndim != 2 or codes.shape[0] < 2 or codes.shape[1] == 0:
        raise ValueError("codes must have at least two atoms and one patch")
    pair_indices = tuple(
        (left, right)
        for left in range(codes.shape[0])
        for right in range(left + 1, codes.shape[0])
    )
    active = np.abs(codes) > ACTIVATION_THRESHOLD
    rows = []
    expected_start = 0
    for _graph_index, _label, start, stop in graph_slices:
        if start != expected_start or not (0 <= start < stop <= codes.shape[1]):
            raise ValueError("graph slices must be contiguous and ordered")
        graph_active = active[:, start:stop]
        rows.append(
            np.asarray(
                [
                    np.mean(graph_active[left] & graph_active[right])
                    for left, right in pair_indices
                ],
                dtype=np.float64,
            )
        )
        expected_start = stop
    if expected_start != codes.shape[1]:
        raise ValueError("graph slices must cover every patch code exactly once")
    return np.stack(rows, axis=0)


def nearest_medoid_one_hot_codes(values: np.ndarray, medoids: np.ndarray) -> np.ndarray:
    """Assign every patch to its nearest real-patch medoid."""
    values = np.asarray(values, dtype=np.float64)
    medoids = np.asarray(medoids, dtype=np.float64)
    if values.ndim != 2 or medoids.ndim != 2 or values.shape[0] != medoids.shape[0]:
        raise ValueError("values and medoids must share a coordinate dimension")
    value_norms = np.sum(values**2, axis=0)[:, None]
    medoid_norms = np.sum(medoids**2, axis=0)[None, :]
    distances = value_norms + medoid_norms - 2.0 * values.T @ medoids
    winners = np.argmin(distances, axis=1)
    codes = np.zeros((medoids.shape[1], values.shape[1]), dtype=np.float64)
    codes[winners, np.arange(values.shape[1])] = 1.0
    return codes


def _restricted_groups(
    groups: Sequence[IsomorphismGroup], indices: set[int]
) -> tuple[IsomorphismGroup, ...]:
    result = []
    covered: set[int] = set()
    for group in groups:
        members = tuple(int(index) for index in group.member_indices if int(index) in indices)
        if not members:
            continue
        if len(members) != len(group.member_indices):
            raise ValueError("grouped outer training set split an exact-isomorphism group")
        result.append(group)
        covered.update(members)
    if covered != indices:
        raise ValueError("restricted groups do not partition outer training graphs")
    return tuple(result)


def make_inner_train_validation_split(
    graphs: Sequence[TUStructureGraph],
    outer_split: FoldSplit,
    *,
    groups: Sequence[IsomorphismGroup] | None,
    grouped: bool,
    seed: int,
    n_splits: int = 5,
) -> tuple[FoldSplit, dict[str, Any]]:
    """Select registered inner fold 0 as validation, preserving groups if required."""
    graph_by_index = {int(graph.index): graph for graph in graphs}
    outer_train_set = set(int(index) for index in outer_split.train_indices)
    train_graphs = tuple(graph_by_index[index] for index in sorted(outer_train_set))
    if grouped:
        if groups is None:
            raise ValueError("groups are required for grouped inner splitting")
        train_groups = _restricted_groups(groups, outer_train_set)
        folds = grouped_isomorphism_folds(
            train_graphs, train_groups, n_splits=n_splits, seed=seed
        )
        split_audit = audit_fold_splits(
            train_graphs,
            folds,
            groups=train_groups,
            require_group_integrity=True,
        )
    else:
        folds = stratified_graph_folds(train_graphs, n_splits=n_splits, seed=seed)
        split_audit = audit_fold_splits(train_graphs, folds)
    selected = folds[0]
    if set(selected.train_indices) | set(selected.test_indices) != outer_train_set:
        raise RuntimeError("inner split does not partition outer train")
    audit = {
        "seed": int(seed),
        "n_splits": int(n_splits),
        "selected_validation_fold": 0,
        "group_integrity_required": bool(grouped),
        "full_inner_partition_audit": split_audit,
        "inner_train_graph_count": int(len(selected.train_indices)),
        "validation_graph_count": int(len(selected.test_indices)),
        "inner_train_class_counts": {
            str(label): int(sum(graph_by_index[index].label == label for index in selected.train_indices))
            for label in (0, 1)
        },
        "validation_class_counts": {
            str(label): int(sum(graph_by_index[index].label == label for index in selected.test_indices))
            for label in (0, 1)
        },
    }
    return FoldSplit(
        fold_index=int(outer_split.fold_index),
        train_indices=selected.train_indices,
        test_indices=selected.test_indices,
    ), audit


def _permute_rows(values: np.ndarray, seed: int) -> np.ndarray:
    values = np.asarray(values)
    return values[np.random.default_rng(int(seed)).permutation(values.shape[0])]


def _labels_for_examples(examples: Sequence[IMDBPatchGraph]) -> np.ndarray:
    return np.asarray([int(example.label) for example in examples], dtype=np.int64)


def _stack_attribute(examples: Sequence[IMDBPatchGraph], attribute: str) -> np.ndarray:
    return np.stack([np.asarray(getattr(example, attribute), dtype=np.float64) for example in examples])


def run_imdb_downstream_fold(
    graphs: Sequence[TUStructureGraph],
    examples: Sequence[IMDBPatchGraph],
    outer_split: FoldSplit,
    inner_split: FoldSplit,
    *,
    n_atoms: int = 12,
    sparsity: int = 2,
    minimum_sparsity: int = 1,
    n_iterations: int = 25,
    gaussian_seed: int = 0,
    graph_shuffle_seed: int,
    label_shuffle_seed: int,
    include_pair_readout: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Fit outer-train dictionaries and evaluate registered R0-A/R0-B features."""
    del graphs  # Graph metadata is already frozen into examples and split indices.
    example_by_index = {int(example.graph_index): example for example in examples}
    outer_train_set = set(int(index) for index in outer_split.train_indices)
    if set(inner_split.train_indices) | set(inner_split.test_indices) != outer_train_set:
        raise ValueError("inner train/validation must partition the outer train")
    if set(inner_split.train_indices) & set(inner_split.test_indices):
        raise ValueError("inner train and validation must be disjoint")

    split_indices = {
        "train": tuple(int(index) for index in inner_split.train_indices),
        "validation": tuple(int(index) for index in inner_split.test_indices),
        "test": tuple(int(index) for index in outer_split.test_indices),
    }
    split_examples = {
        split: tuple(example_by_index[index] for index in indices)
        for split, indices in split_indices.items()
    }
    outer_train_examples = tuple(example_by_index[index] for index in outer_split.train_indices)
    raw_outer_train, _outer_train_slices = stack_patch_graphs(outer_train_examples)
    raw_by_split: dict[str, np.ndarray] = {}
    slices_by_split: dict[str, tuple[tuple[int, int, int, int], ...]] = {}
    for split, items in split_examples.items():
        raw_by_split[split], slices_by_split[split] = stack_patch_graphs(items)

    train_mean = np.mean(raw_outer_train, axis=1, keepdims=True)
    centered_outer_train = raw_outer_train - train_mean
    centered = {split: values - train_mean for split, values in raw_by_split.items()}

    initial_dictionary, initialization = deterministic_maximin_initialization(
        centered_outer_train, n_atoms
    )
    final_dictionary, _training_codes, training_info = ksvd(
        centered_outer_train,
        n_atoms=n_atoms,
        T=sparsity,
        T_min=minimum_sparsity,
        n_iter=n_iterations,
        seed=0,
        initial_dictionary=initial_dictionary,
        coherence_step=0.0,
        anchor_strength=0.0,
    )
    rng = np.random.default_rng(int(gaussian_seed))
    gaussian_dictionary = rng.standard_normal((centered_outer_train.shape[0], n_atoms))
    gaussian_dictionary /= np.linalg.norm(gaussian_dictionary, axis=0, keepdims=True)
    medoid_dictionary, medoid_info = deterministic_lloyd_medoids(
        centered_outer_train,
        n_medoids=n_atoms,
        initial_indices=initialization["selected_training_indices"],
    )
    left, _singular, _right = np.linalg.svd(centered_outer_train, full_matrices=False)
    pca_basis = left[:, : min(n_atoms, left.shape[1])]

    labels = {split: _labels_for_examples(items) for split, items in split_examples.items()}
    stats = {split: _stack_attribute(items, "graph_statistics") for split, items in split_examples.items()}
    raw_walk = {
        split: np.stack([item.features["walk_mean_std"] for item in items], axis=0)
        for split, items in split_examples.items()
    }
    code_features: dict[str, dict[str, np.ndarray]] = {
        key: {} for key in ("init", "final", "pca12", "medoid_bag", "fixed_gaussian")
    }
    pair_features: dict[str, dict[str, np.ndarray]] = {
        key: {} for key in ("init", "final")
    }
    for split in ("train", "validation", "test"):
        code_matrices = {
            "init": encode_with_minimum_sparsity(
                centered[split], initial_dictionary,
                sparsity=sparsity, minimum_sparsity=minimum_sparsity,
            ),
            "final": encode_with_minimum_sparsity(
                centered[split], final_dictionary,
                sparsity=sparsity, minimum_sparsity=minimum_sparsity,
            ),
            "pca12": pca_basis.T @ centered[split],
            "medoid_bag": nearest_medoid_one_hot_codes(centered[split], medoid_dictionary),
            "fixed_gaussian": encode_with_minimum_sparsity(
                centered[split], gaussian_dictionary,
                sparsity=sparsity, minimum_sparsity=minimum_sparsity,
            ),
        }
        for key, codes in code_matrices.items():
            code_features[key][split] = variable_graph_code_readout(
                codes, slices_by_split[split]
            )
        if include_pair_readout:
            for key in ("init", "final"):
                pair_features[key][split] = variable_graph_pair_readout(
                    code_matrices[key], slices_by_split[split]
                )

    features: dict[str, dict[str, np.ndarray]] = {
        "stats": stats,
        "init": code_features["init"],
        "final": code_features["final"],
        "stats_plus_init": {
            split: np.column_stack([stats[split], code_features["init"][split]])
            for split in stats
        },
        "stats_plus_final": {
            split: np.column_stack([stats[split], code_features["final"][split]])
            for split in stats
        },
        "stats_plus_raw_walk": {
            split: np.column_stack([stats[split], raw_walk[split]]) for split in stats
        },
        "stats_plus_pca12": {
            split: np.column_stack([stats[split], code_features["pca12"][split]])
            for split in stats
        },
        "stats_plus_medoid_bag": {
            split: np.column_stack([stats[split], code_features["medoid_bag"][split]])
            for split in stats
        },
        "stats_plus_fixed_gaussian": {
            split: np.column_stack([stats[split], code_features["fixed_gaussian"][split]])
            for split in stats
        },
    }
    if include_pair_readout:
        features.update(
            {
                "stats_plus_init_pair": {
                    split: np.column_stack(
                        [stats[split], code_features["init"][split], pair_features["init"][split]]
                    )
                    for split in stats
                },
                "stats_plus_final_pair": {
                    split: np.column_stack(
                        [stats[split], code_features["final"][split], pair_features["final"][split]]
                    )
                    for split in stats
                },
            }
        )
    evaluations = {
        key: evaluate_feature_matrices(matrices, labels, feature_key=key)
        for key, matrices in features.items()
    }

    shuffle_sequences = np.random.SeedSequence(int(graph_shuffle_seed)).spawn(3)
    shuffled_final = {
        split: _permute_rows(code_features["final"][split], int(sequence.generate_state(1)[0]))
        for split, sequence in zip(("train", "validation", "test"), shuffle_sequences)
    }
    shuffled_feature = {
        split: np.column_stack([stats[split], shuffled_final[split]]) for split in stats
    }
    evaluations["stats_plus_shuffled_final"] = evaluate_feature_matrices(
        shuffled_feature, labels, feature_key="stats_plus_shuffled_final"
    )

    if include_pair_readout:
        final_pair_feature = {
            split: np.column_stack(
                [code_features["final"][split], pair_features["final"][split]]
            )
            for split in stats
        }
        pair_shuffle_sequences = np.random.SeedSequence(int(graph_shuffle_seed) + 1).spawn(3)
        shuffled_final_pair = {
            split: _permute_rows(
                final_pair_feature[split], int(sequence.generate_state(1)[0])
            )
            for split, sequence in zip(
                ("train", "validation", "test"), pair_shuffle_sequences
            )
        }
        shuffled_pair_feature = {
            split: np.column_stack([stats[split], shuffled_final_pair[split]])
            for split in stats
        }
        evaluations["stats_plus_shuffled_final_pair"] = evaluate_feature_matrices(
            shuffled_pair_feature,
            labels,
            feature_key="stats_plus_shuffled_final_pair",
        )

    label_sequences = np.random.SeedSequence(int(label_shuffle_seed)).spawn(3)
    shuffled_labels = {
        split: _permute_rows(labels[split], int(sequence.generate_state(1)[0]))
        for split, sequence in zip(("train", "validation", "test"), label_sequences)
    }
    evaluations["label_shuffle_stats_plus_final"] = evaluate_feature_matrices(
        features["stats_plus_final"],
        shuffled_labels,
        feature_key="label_shuffle_stats_plus_final",
    )
    if include_pair_readout:
        evaluations["label_shuffle_stats_plus_final_pair"] = evaluate_feature_matrices(
            features["stats_plus_final_pair"],
            shuffled_labels,
            feature_key="label_shuffle_stats_plus_final_pair",
        )

    test_scores = {
        key: float(value["test_balanced_accuracy"]) for key, value in evaluations.items()
    }
    result = {
        "fold_index": int(outer_split.fold_index),
        "outer_train_graph_count": int(len(outer_split.train_indices)),
        "inner_train_graph_count": int(len(inner_split.train_indices)),
        "validation_graph_count": int(len(inner_split.test_indices)),
        "test_graph_count": int(len(outer_split.test_indices)),
        "outer_train_patch_count": int(raw_outer_train.shape[1]),
        "split_patch_counts": {
            split: int(raw_by_split[split].shape[1]) for split in raw_by_split
        },
        "train_coordinate_mean": train_mean[:, 0].tolist(),
        "initialization": initialization,
        "medoid_initialization": medoid_info,
        "training_reconstruction_curve_internal": [
            float(value) for value in training_info["recon_curve"]
        ],
        "config": {
            "patch_dimension": int(raw_outer_train.shape[0]),
            "n_atoms": int(n_atoms),
            "sparsity": int(sparsity),
            "minimum_sparsity": int(minimum_sparsity),
            "n_iterations": int(n_iterations),
            "ksvd_internal_seed": 0,
            "gaussian_seed": int(gaussian_seed),
            "graph_shuffle_seed": int(graph_shuffle_seed),
            "label_shuffle_seed": int(label_shuffle_seed),
            "graph_code_dimension": int(3 * n_atoms),
            "graph_code_readout": ["activation_frequency", "mean_absolute", "rms"],
            "pair_readout": (
                "within_patch_coactivation_fraction" if include_pair_readout else None
            ),
            "pair_code_dimension": int(n_atoms * (n_atoms - 1) // 2)
            if include_pair_readout
            else 0,
            "centering": "outer_train_coordinate_mean",
        },
        "evaluations": evaluations,
        "attribution": {
            "update_gain": float(
                test_scores["stats_plus_final"] - test_scores["stats_plus_init"]
            ),
            "beyond_stats_gain": float(
                test_scores["stats_plus_final"] - test_scores["stats"]
            ),
            "correct_alignment_gain_over_shuffle": float(
                test_scores["stats_plus_final"] - test_scores["stats_plus_shuffled_final"]
            ),
        },
    }
    if include_pair_readout:
        result["attribution"].update(
            {
                "pair_update_gain": float(
                    test_scores["stats_plus_final_pair"]
                    - test_scores["stats_plus_init_pair"]
                ),
                "pair_added_value_over_marginal": float(
                    test_scores["stats_plus_final_pair"]
                    - test_scores["stats_plus_final"]
                ),
                "pair_correct_alignment_gain_over_shuffle": float(
                    test_scores["stats_plus_final_pair"]
                    - test_scores["stats_plus_shuffled_final_pair"]
                ),
            }
        )
    dictionaries = {
        "init": initial_dictionary,
        "final": final_dictionary,
        "pca12": pca_basis,
        "medoid": medoid_dictionary,
        "fixed_gaussian": gaussian_dictionary,
    }
    return result, dictionaries
