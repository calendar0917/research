"""Fold-local R0-D dictionary audit utilities for raw IMDB-BINARY WALK patches.

This module deliberately keeps the real-data audit separate from the older
REAL_STRUCTURE pipeline.  It implements the frozen single-initialization,
no-restart comparison between deterministic INIT and 25-update FINAL KSVD.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from .from_scratch_unplanted_dictionary import (
    ACTIVATION_THRESHOLD,
    EPS,
    deterministic_maximin_initialization,
    dictionary_metrics,
)
from .imdb_walk_substrate import IMDBPatchGraph, IsomorphismGroup, TUStructureGraph
from .ksvd import ksvd


@dataclass(frozen=True)
class FoldSplit:
    fold_index: int
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]


def _validate_split_inputs(graphs: Sequence[TUStructureGraph], n_splits: int) -> None:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if len(graphs) < n_splits:
        raise ValueError("number of graphs must be at least n_splits")
    indices = [int(graph.index) for graph in graphs]
    if len(set(indices)) != len(indices):
        raise ValueError("graph indices must be unique")


def stratified_graph_folds(
    graphs: Sequence[TUStructureGraph],
    *,
    n_splits: int = 5,
    seed: int = 731301,
) -> tuple[FoldSplit, ...]:
    """Create deterministic class-stratified folds without external ML deps."""
    graphs = tuple(graphs)
    _validate_split_inputs(graphs, n_splits)
    rng = np.random.default_rng(int(seed))
    fold_tests: list[list[int]] = [[] for _ in range(n_splits)]
    labels = sorted({int(graph.label) for graph in graphs})
    for label in labels:
        class_indices = np.asarray(
            sorted(int(graph.index) for graph in graphs if int(graph.label) == label),
            dtype=np.int64,
        )
        class_indices = class_indices[rng.permutation(class_indices.size)]
        for fold_index, part in enumerate(np.array_split(class_indices, n_splits)):
            fold_tests[fold_index].extend(int(value) for value in part)
    all_indices = {int(graph.index) for graph in graphs}
    return tuple(
        FoldSplit(
            fold_index=fold_index,
            train_indices=tuple(sorted(all_indices - set(test_indices))),
            test_indices=tuple(sorted(test_indices)),
        )
        for fold_index, test_indices in enumerate(fold_tests)
    )


def grouped_isomorphism_folds(
    graphs: Sequence[TUStructureGraph],
    groups: Sequence[IsomorphismGroup],
    *,
    n_splits: int = 5,
    seed: int = 731301,
) -> tuple[FoldSplit, ...]:
    """Keep exact-isomorphism groups intact while greedily balancing labels.

    Groups are placed largest/most label-skewed first.  For each placement, the
    selected fold minimizes squared deviation from target class, graph, and
    structure-group counts.  A seeded rank is used only for deterministic
    tie-breaking; it is not a restart or a selected split.
    """
    graphs = tuple(graphs)
    groups = tuple(groups)
    _validate_split_inputs(graphs, n_splits)
    graph_by_index = {int(graph.index): graph for graph in graphs}
    expected = set(graph_by_index)
    covered = [int(index) for group in groups for index in group.member_indices]
    if len(covered) != len(set(covered)) or set(covered) != expected:
        raise ValueError("isomorphism groups must partition the supplied graphs")

    labels = sorted({int(graph.label) for graph in graphs})
    label_to_position = {label: position for position, label in enumerate(labels)}
    group_vectors: list[np.ndarray] = []
    for group in groups:
        vector = np.zeros(len(labels), dtype=np.float64)
        for index in group.member_indices:
            vector[label_to_position[int(graph_by_index[int(index)].label)]] += 1.0
        group_vectors.append(vector)

    rng = np.random.default_rng(int(seed))
    tie_order = rng.permutation(len(groups))
    tie_rank = np.empty(len(groups), dtype=np.int64)
    tie_rank[tie_order] = np.arange(len(groups), dtype=np.int64)
    order = sorted(
        range(len(groups)),
        key=lambda position: (
            -len(groups[position].member_indices),
            -float(np.max(group_vectors[position])),
            -float(np.max(group_vectors[position]) - np.min(group_vectors[position])),
            int(tie_rank[position]),
            int(groups[position].group_id),
        ),
    )

    total_class = np.sum(np.stack(group_vectors), axis=0)
    target_class = total_class / float(n_splits)
    target_graphs = len(graphs) / float(n_splits)
    target_groups = len(groups) / float(n_splits)
    fold_class = np.zeros((n_splits, len(labels)), dtype=np.float64)
    fold_graphs = np.zeros(n_splits, dtype=np.float64)
    fold_group_counts = np.zeros(n_splits, dtype=np.float64)
    fold_members: list[list[int]] = [[] for _ in range(n_splits)]

    # Seed every fold with one group so that a pathological objective cannot
    # leave an empty test fold.
    for placement_index, group_position in enumerate(order):
        group = groups[group_position]
        vector = group_vectors[group_position]
        size = float(len(group.member_indices))
        if placement_index < n_splits:
            chosen = placement_index
        else:
            candidates: list[tuple[float, float, float, int]] = []
            for fold_index in range(n_splits):
                new_class = fold_class[fold_index] + vector
                new_graphs = fold_graphs[fold_index] + size
                new_groups = fold_group_counts[fold_index] + 1.0
                old_class_cost = float(
                    np.sum(((fold_class[fold_index] - target_class) / np.maximum(target_class, 1.0)) ** 2)
                )
                new_class_cost = float(
                    np.sum(((new_class - target_class) / np.maximum(target_class, 1.0)) ** 2)
                )
                old_graph_cost = float(
                    ((fold_graphs[fold_index] - target_graphs) / target_graphs) ** 2
                )
                new_graph_cost = float(((new_graphs - target_graphs) / target_graphs) ** 2)
                old_group_cost = float(
                    ((fold_group_counts[fold_index] - target_groups) / target_groups) ** 2
                )
                new_group_cost = float(((new_groups - target_groups) / target_groups) ** 2)
                # Compare the *increment* in the global objective. Comparing
                # absolute post-placement costs would incorrectly favor an
                # already balanced fold over an empty, initially costly fold.
                score = (new_class_cost - old_class_cost) + 0.35 * (
                    new_graph_cost - old_graph_cost
                ) + 0.05 * (new_group_cost - old_group_cost)
                candidates.append(
                    (score, float(fold_graphs[fold_index]), float(fold_group_counts[fold_index]), fold_index)
                )
            chosen = min(candidates)[-1]
        fold_class[chosen] += vector
        fold_graphs[chosen] += size
        fold_group_counts[chosen] += 1.0
        fold_members[chosen].extend(int(index) for index in group.member_indices)

    all_indices = set(expected)
    return tuple(
        FoldSplit(
            fold_index=fold_index,
            train_indices=tuple(sorted(all_indices - set(test_indices))),
            test_indices=tuple(sorted(test_indices)),
        )
        for fold_index, test_indices in enumerate(fold_members)
    )


def audit_fold_splits(
    graphs: Sequence[TUStructureGraph],
    folds: Sequence[FoldSplit],
    *,
    groups: Sequence[IsomorphismGroup] | None = None,
    require_group_integrity: bool | None = None,
) -> dict[str, Any]:
    """Check partition integrity, class balance, and optional group leakage."""
    graphs = tuple(graphs)
    folds = tuple(folds)
    if require_group_integrity is None:
        require_group_integrity = groups is not None
    if require_group_integrity and groups is None:
        raise ValueError("groups are required when group integrity is enforced")
    graph_by_index = {int(graph.index): graph for graph in graphs}
    all_indices = set(graph_by_index)
    test_occurrences = {index: 0 for index in all_indices}
    fold_rows = []
    for fold in folds:
        train = set(fold.train_indices)
        test = set(fold.test_indices)
        if train & test:
            raise ValueError("train/test overlap within a fold")
        if train | test != all_indices:
            raise ValueError("each fold must cover every graph")
        for index in test:
            test_occurrences[index] += 1
        labels = [int(graph_by_index[index].label) for index in test]
        class_counts = {
            str(label): int(sum(value == label for value in labels))
            for label in sorted(set(labels) | {int(graph.label) for graph in graphs})
        }
        row: dict[str, Any] = {
            "fold_index": int(fold.fold_index),
            "train_graph_count": int(len(train)),
            "test_graph_count": int(len(test)),
            "test_class_counts": class_counts,
        }
        if groups is not None:
            test_group_count = 0
            leaked = 0
            for group in groups:
                members = set(int(index) for index in group.member_indices)
                in_test = bool(members & test)
                in_train = bool(members & train)
                test_group_count += int(in_test)
                leaked += int(in_test and in_train)
            row["test_structure_group_count"] = int(test_group_count)
            row["leaked_structure_group_count"] = int(leaked)
        fold_rows.append(row)
    every_graph_once = all(value == 1 for value in test_occurrences.values())
    leakage = int(sum(row.get("leaked_structure_group_count", 0) for row in fold_rows))
    return {
        "n_splits": int(len(folds)),
        "every_graph_appears_once_in_test": bool(every_graph_once),
        "total_group_leakage_count": leakage if groups is not None else None,
        "group_integrity_required": bool(require_group_integrity),
        "passes_partition_gate": bool(
            every_graph_once and (not require_group_integrity or leakage == 0)
        ),
        "folds": fold_rows,
    }


def stack_patch_graphs(
    examples: Iterable[IMDBPatchGraph],
) -> tuple[np.ndarray, tuple[tuple[int, int, int, int], ...]]:
    """Stack variable-count graph patches and retain graph slice metadata."""
    examples = tuple(examples)
    if not examples:
        raise ValueError("examples cannot be empty")
    dimension = int(examples[0].walk_vectors.shape[1])
    matrices = []
    slices = []
    start = 0
    for example in examples:
        values = np.asarray(example.walk_vectors, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != dimension or values.shape[0] == 0:
            raise ValueError("all examples must contain nonempty equal-dimensional patches")
        stop = start + values.shape[0]
        matrices.append(values)
        slices.append((int(example.graph_index), int(example.label), start, stop))
        start = stop
    return np.concatenate(matrices, axis=0).T, tuple(slices)


def encode_with_minimum_sparsity(
    values: np.ndarray,
    dictionary: np.ndarray,
    *,
    sparsity: int = 2,
    minimum_sparsity: int = 1,
) -> np.ndarray:
    """OMP encode held-out columns and explicitly honor T_min."""
    values = np.asarray(values, dtype=np.float64)
    dictionary = np.asarray(dictionary, dtype=np.float64)
    if minimum_sparsity < 0 or minimum_sparsity > sparsity:
        raise ValueError("minimum_sparsity must lie in [0, sparsity]")
    _, codes, _ = ksvd(
        values,
        n_atoms=dictionary.shape[1],
        T=sparsity,
        T_min=minimum_sparsity,
        n_iter=0,
        seed=0,
        initial_dictionary=dictionary,
        coherence_step=0.0,
        anchor_strength=0.0,
    )
    if minimum_sparsity:
        nnz = np.sum(np.abs(codes) > ACTIVATION_THRESHOLD, axis=0)
        for column in np.flatnonzero(nnz < minimum_sparsity):
            correlations = np.abs(dictionary.T @ values[:, column])
            support = np.argsort(-correlations, kind="stable")[:minimum_sparsity]
            coefficients, _, _, _ = np.linalg.lstsq(
                dictionary[:, support], values[:, column], rcond=None
            )
            codes[:, column] = 0.0
            codes[support, column] = coefficients
    return codes


def graph_balanced_reconstruction_metrics(
    values: np.ndarray,
    reconstruction: np.ndarray,
    graph_slices: Sequence[tuple[int, int, int, int]],
) -> dict[str, Any]:
    """Average each graph's relative Frobenius error with equal graph weight."""
    values = np.asarray(values, dtype=np.float64)
    reconstruction = np.asarray(reconstruction, dtype=np.float64)
    if values.shape != reconstruction.shape:
        raise ValueError("values and reconstruction must have identical shapes")
    errors = []
    labels = []
    for graph_index, label, start, stop in graph_slices:
        if not (0 <= start < stop <= values.shape[1]):
            raise ValueError("invalid graph slice")
        signal = values[:, start:stop]
        residual = signal - reconstruction[:, start:stop]
        error = float(np.linalg.norm(residual, "fro") / max(np.linalg.norm(signal, "fro"), EPS))
        errors.append(error)
        labels.append(int(label))
    array = np.asarray(errors, dtype=np.float64)
    class_means = {
        str(label): float(np.mean(array[np.asarray(labels) == label]))
        for label in sorted(set(labels))
    }
    return {
        "graph_count": int(array.size),
        "mean_relative_reconstruction_error": float(np.mean(array)),
        "median_relative_reconstruction_error": float(np.median(array)),
        "std_relative_reconstruction_error": float(np.std(array, ddof=0)),
        "minimum_relative_reconstruction_error": float(np.min(array)),
        "maximum_relative_reconstruction_error": float(np.max(array)),
        "class_mean_relative_reconstruction_error": class_means,
    }


def _reconstruction_metrics(
    values: np.ndarray,
    reconstruction: np.ndarray,
    graph_slices: Sequence[tuple[int, int, int, int]],
) -> dict[str, Any]:
    residual = values - reconstruction
    relative = float(np.linalg.norm(residual, "fro") / max(np.linalg.norm(values, "fro"), EPS))
    return {
        "patch_weighted": {
            "patch_count": int(values.shape[1]),
            "relative_reconstruction_error": relative,
            "nmse": float(np.sum(residual**2) / max(float(np.sum(values**2)), EPS)),
        },
        "graph_balanced": graph_balanced_reconstruction_metrics(values, reconstruction, graph_slices),
    }


def deterministic_lloyd_medoids(
    values: np.ndarray,
    *,
    n_medoids: int,
    initial_indices: Sequence[int],
    max_iterations: int = 20,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Refine real-patch representatives by deterministic squared-Euclidean Lloyd steps.

    For squared Euclidean loss, the member nearest a cluster mean is the exact
    within-cluster medoid.  This avoids an O(N^2) pairwise distance matrix.
    """
    values = np.asarray(values, dtype=np.float64)
    indices = np.asarray(initial_indices, dtype=np.int64).copy()
    if values.ndim != 2 or len(indices) != n_medoids:
        raise ValueError("invalid values or initial_indices")
    if np.any(indices < 0) or np.any(indices >= values.shape[1]):
        raise ValueError("medoid indices out of bounds")
    history = [indices.astype(int).tolist()]
    for _ in range(max_iterations):
        medoids = values[:, indices]
        distances = (
            np.sum(values**2, axis=0)[:, None]
            + np.sum(medoids**2, axis=0)[None, :]
            - 2.0 * values.T @ medoids
        )
        assignments = np.argmin(distances, axis=1)
        updated = indices.copy()
        for cluster in range(n_medoids):
            members = np.flatnonzero(assignments == cluster)
            if members.size == 0:
                continue
            mean = np.mean(values[:, members], axis=1)
            distance_to_mean = np.sum((values[:, members] - mean[:, None]) ** 2, axis=0)
            best_value = float(np.min(distance_to_mean))
            candidates = members[np.abs(distance_to_mean - best_value) <= 1e-12]
            updated[cluster] = int(np.min(candidates))
        history.append(updated.astype(int).tolist())
        if np.array_equal(updated, indices):
            break
        indices = updated
    dictionary = values[:, indices].copy()
    norms = np.linalg.norm(dictionary, axis=0)
    if np.any(norms <= EPS):
        raise ValueError("medoid refinement selected a zero centered patch")
    dictionary /= norms[None, :]
    return dictionary, {
        "name": "deterministic_lloyd_real_patch_medoids",
        "selected_training_indices": indices.astype(int).tolist(),
        "iterations": int(len(history) - 1),
        "index_history": history,
    }


def _sparse_stage_metrics(
    centered: dict[str, np.ndarray],
    slices: dict[str, Sequence[tuple[int, int, int, int]]],
    dictionary: np.ndarray,
    *,
    sparsity: int,
    minimum_sparsity: int,
) -> dict[str, Any]:
    result = {}
    for split in ("train", "test"):
        codes = encode_with_minimum_sparsity(
            centered[split], dictionary, sparsity=sparsity, minimum_sparsity=minimum_sparsity
        )
        result[split] = {
            "dictionary_health": dictionary_metrics(centered[split], dictionary, codes),
            **_reconstruction_metrics(centered[split], dictionary @ codes, slices[split]),
        }
    return result


def run_imdb_dictionary_fold(
    examples: Sequence[IMDBPatchGraph],
    split: FoldSplit,
    *,
    n_atoms: int = 12,
    sparsity: int = 2,
    minimum_sparsity: int = 1,
    n_iterations: int = 25,
    gaussian_seed: int = 0,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run one completely fold-local INIT-vs-FINAL reconstruction audit."""
    example_by_index = {int(example.graph_index): example for example in examples}
    train_examples = tuple(example_by_index[index] for index in split.train_indices)
    test_examples = tuple(example_by_index[index] for index in split.test_indices)
    raw_train, train_slices = stack_patch_graphs(train_examples)
    raw_test, test_slices = stack_patch_graphs(test_examples)
    train_mean = np.mean(raw_train, axis=1, keepdims=True)
    centered = {"train": raw_train - train_mean, "test": raw_test - train_mean}
    slices = {"train": train_slices, "test": test_slices}

    initial_dictionary, initialization = deterministic_maximin_initialization(
        centered["train"], n_atoms
    )
    learned_dictionary, _training_codes, training_info = ksvd(
        centered["train"],
        n_atoms=n_atoms,
        T=sparsity,
        T_min=minimum_sparsity,
        n_iter=n_iterations,
        seed=0,
        initial_dictionary=initial_dictionary,
        coherence_step=0.0,
        anchor_strength=0.0,
    )

    stages = {
        "init": _sparse_stage_metrics(
            centered, slices, initial_dictionary,
            sparsity=sparsity, minimum_sparsity=minimum_sparsity,
        ),
        "final": _sparse_stage_metrics(
            centered, slices, learned_dictionary,
            sparsity=sparsity, minimum_sparsity=minimum_sparsity,
        ),
    }

    # Fixed Gaussian is independent of data beyond the frozen dimension.
    rng = np.random.default_rng(int(gaussian_seed))
    gaussian = rng.standard_normal((raw_train.shape[0], n_atoms))
    gaussian /= np.linalg.norm(gaussian, axis=0, keepdims=True)
    stages["fixed_gaussian"] = _sparse_stage_metrics(
        centered, slices, gaussian,
        sparsity=sparsity, minimum_sparsity=minimum_sparsity,
    )

    medoid_dictionary, medoid_info = deterministic_lloyd_medoids(
        centered["train"],
        n_medoids=n_atoms,
        initial_indices=initialization["selected_training_indices"],
    )
    stages["medoid"] = _sparse_stage_metrics(
        centered, slices, medoid_dictionary,
        sparsity=sparsity, minimum_sparsity=minimum_sparsity,
    )

    # Rank-12 PCA is a non-sparse reconstruction floor fitted only on train.
    left, _singular, _right = np.linalg.svd(centered["train"], full_matrices=False)
    pca_basis = left[:, : min(n_atoms, left.shape[1])]
    pca = {}
    for split_name in ("train", "test"):
        reconstruction = pca_basis @ (pca_basis.T @ centered[split_name])
        pca[split_name] = _reconstruction_metrics(
            centered[split_name], reconstruction, slices[split_name]
        )
    stages["pca12"] = pca

    init_graph_error = stages["init"]["test"]["graph_balanced"][
        "mean_relative_reconstruction_error"
    ]
    final_graph_error = stages["final"]["test"]["graph_balanced"][
        "mean_relative_reconstruction_error"
    ]
    init_patch_error = stages["init"]["test"]["patch_weighted"][
        "relative_reconstruction_error"
    ]
    final_patch_error = stages["final"]["test"]["patch_weighted"][
        "relative_reconstruction_error"
    ]
    final_health = stages["final"]["test"]["dictionary_health"]
    result = {
        "fold_index": int(split.fold_index),
        "train_graph_count": int(len(train_examples)),
        "test_graph_count": int(len(test_examples)),
        "train_patch_count": int(raw_train.shape[1]),
        "test_patch_count": int(raw_test.shape[1]),
        "train_class_counts": {
            str(label): int(sum(example.label == label for example in train_examples))
            for label in sorted({example.label for example in examples})
        },
        "test_class_counts": {
            str(label): int(sum(example.label == label for example in test_examples))
            for label in sorted({example.label for example in examples})
        },
        "train_coordinate_mean": train_mean[:, 0].tolist(),
        "initialization": initialization,
        "medoid_initialization": medoid_info,
        "config": {
            "patch_dimension": int(raw_train.shape[0]),
            "n_atoms": int(n_atoms),
            "sparsity": int(sparsity),
            "minimum_sparsity": int(minimum_sparsity),
            "n_iterations": int(n_iterations),
            "ksvd_internal_seed": 0,
            "gaussian_seed": int(gaussian_seed),
            "centering": "outer_train_coordinate_mean",
            "per_patch_normalization": False,
        },
        "stages": stages,
        "training_reconstruction_curve_internal": [
            float(value) for value in training_info["recon_curve"]
        ],
        "attribution": {
            "test_graph_balanced_relative_reduction": float(
                (init_graph_error - final_graph_error) / max(init_graph_error, EPS)
            ),
            "test_patch_weighted_relative_reduction": float(
                (init_patch_error - final_patch_error) / max(init_patch_error, EPS)
            ),
        },
        "health_gate_components": {
            "positive_graph_balanced_reduction": bool(final_graph_error < init_graph_error),
            "final_test_nondead_atoms_at_least_10": bool(final_health["nondead_atom_count"] >= 10),
            "final_test_maximum_activation_share_at_most_0_60": bool(
                final_health["maximum_activation_share"] <= 0.60
            ),
        },
    }
    dictionaries = {
        "init": initial_dictionary,
        "final": learned_dictionary,
        "fixed_gaussian": gaussian,
        "medoid": medoid_dictionary,
        "pca12": pca_basis,
    }
    return result, dictionaries
