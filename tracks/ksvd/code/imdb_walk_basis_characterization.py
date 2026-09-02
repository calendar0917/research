"""Label-free R1-A basis characterization for raw IMDB WALK dictionaries.

This module reuses the registered R0-D fold dictionaries and does not retrain
KSVD.  Labels never enter the characterization gates; they may only appear as
descriptive diagnostics.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .from_scratch_unplanted_dictionary import (
    ACTIVATION_THRESHOLD,
    cross_replicate_dictionary_similarity,
    dictionary_metrics,
    exact_maximum_cosine_assignment,
)
from .imdb_walk_dictionary import (
    FoldSplit,
    encode_with_minimum_sparsity,
    stack_patch_graphs,
)
from .imdb_walk_substrate import IMDBPatchGraph


EPS = 1e-12
TOP_K = 5


def _row_keys(values: np.ndarray) -> list[tuple[int, ...]]:
    values = np.asarray(values)
    return [tuple(int(item) for item in row) for row in values]


def nearest_real_patch_proximity(
    dictionary: np.ndarray,
    centered_patches: np.ndarray,
    edge_counts: np.ndarray,
) -> dict[str, Any]:
    """Measure how close each atom is to some real centered train patch."""
    dictionary = np.asarray(dictionary, dtype=np.float64)
    centered_patches = np.asarray(centered_patches, dtype=np.float64)
    edge_counts = np.asarray(edge_counts, dtype=np.int64)
    if dictionary.ndim != 2 or centered_patches.ndim != 2:
        raise ValueError("dictionary and patches must be matrices")
    if dictionary.shape[0] != centered_patches.shape[0]:
        raise ValueError("dictionary and patches must share dimension")
    if edge_counts.shape != (centered_patches.shape[1],):
        raise ValueError("edge_counts must align with patch columns")

    atom_norms = np.linalg.norm(dictionary, axis=0)
    patch_norms = np.linalg.norm(centered_patches, axis=0)
    safe_atoms = np.maximum(atom_norms, EPS)
    safe_patches = np.maximum(patch_norms, EPS)
    cosine = np.abs(dictionary.T @ centered_patches) / (safe_atoms[:, None] * safe_patches[None, :])
    nearest_index = np.argmax(cosine, axis=1)
    nearest_cosine = cosine[np.arange(dictionary.shape[1]), nearest_index]
    return {
        "nearest_absolute_cosine": nearest_cosine.tolist(),
        "mean_nearest_absolute_cosine": float(np.mean(nearest_cosine)),
        "minimum_nearest_absolute_cosine": float(np.min(nearest_cosine)),
        "nearest_patch_edge_counts": edge_counts[nearest_index].astype(int).tolist(),
        "mean_nearest_patch_edge_count": float(np.mean(edge_counts[nearest_index])),
        "nearest_patch_indices": nearest_index.astype(int).tolist(),
    }


def top_activating_patch_summary(
    codes: np.ndarray,
    walk_vectors: np.ndarray,
    canonical_vectors: np.ndarray,
    edge_counts: np.ndarray,
    graph_slices: Sequence[tuple[int, int, int, int]],
    *,
    top_k: int = TOP_K,
) -> list[dict[str, Any]]:
    """Summarize the strongest held-out activations for every atom."""
    codes = np.asarray(codes, dtype=np.float64)
    walk_vectors = np.asarray(walk_vectors, dtype=np.float64)
    canonical_vectors = np.asarray(canonical_vectors, dtype=np.float64)
    edge_counts = np.asarray(edge_counts, dtype=np.int64)
    if codes.ndim != 2:
        raise ValueError("codes must be atom-by-patch")
    if walk_vectors.shape[0] != codes.shape[1] or canonical_vectors.shape != walk_vectors.shape:
        raise ValueError("walk/canonical matrices must align with patch codes")
    if edge_counts.shape != (codes.shape[1],):
        raise ValueError("edge_counts must align with patch codes")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    graph_labels = np.empty(codes.shape[1], dtype=np.int64)
    graph_indices = np.empty(codes.shape[1], dtype=np.int64)
    for graph_index, label, start, stop in graph_slices:
        graph_indices[start:stop] = int(graph_index)
        graph_labels[start:stop] = int(label)

    summaries = []
    for atom_index in range(codes.shape[0]):
        coefficients = codes[atom_index]
        order = np.argsort(-np.abs(coefficients), kind="stable")[: min(top_k, coefficients.size)]
        selected_walk = walk_vectors[order]
        selected_canonical = canonical_vectors[order]
        selected_edges = edge_counts[order]
        selected_abs = np.abs(coefficients[order])
        walk_keys = _row_keys(np.rint(selected_walk))
        canonical_keys = _row_keys(np.rint(selected_canonical))
        edge_values, edge_counts_selected = np.unique(selected_edges, return_counts=True)
        dominant_mass = float(np.max(edge_counts_selected) / selected_edges.size)
        summaries.append(
            {
                "atom_index": int(atom_index),
                "top_k": int(order.size),
                "mean_absolute_coefficient": float(np.mean(selected_abs)),
                "minimum_absolute_coefficient": float(np.min(selected_abs)),
                "edge_count_mean": float(np.mean(selected_edges)),
                "edge_count_std": float(np.std(selected_edges, ddof=0)),
                "unique_walk_vector_count": int(len(set(walk_keys))),
                "unique_canonical_signature_count": int(len(set(canonical_keys))),
                "dominant_edge_count": int(edge_values[int(np.argmax(edge_counts_selected))]),
                "dominant_edge_count_mass": dominant_mass,
                "graph_indices": graph_indices[order].astype(int).tolist(),
                "graph_labels_descriptive_only": graph_labels[order].astype(int).tolist(),
                "edge_counts": selected_edges.astype(int).tolist(),
                "absolute_coefficients": selected_abs.tolist(),
            }
        )
    return summaries


def atom_mass_summary(dictionary: np.ndarray) -> list[dict[str, Any]]:
    dictionary = np.asarray(dictionary, dtype=np.float64)
    rows = []
    for atom_index in range(dictionary.shape[1]):
        atom = dictionary[:, atom_index]
        l1 = float(np.sum(np.abs(atom)))
        positive = float(np.sum(np.clip(atom, 0.0, None)))
        rows.append(
            {
                "atom_index": int(atom_index),
                "l1_mass": l1,
                "positive_mass_fraction": float(positive / max(l1, EPS)),
                "maximum_absolute_coordinate": float(np.max(np.abs(atom))),
            }
        )
    return rows


def characterize_dictionary_on_split(
    dictionary: np.ndarray,
    centered_train: np.ndarray,
    train_edge_counts: np.ndarray,
    centered_test: np.ndarray,
    test_walk_vectors: np.ndarray,
    test_canonical_vectors: np.ndarray,
    test_edge_counts: np.ndarray,
    test_slices: Sequence[tuple[int, int, int, int]],
    *,
    sparsity: int = 2,
    minimum_sparsity: int = 1,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    codes = encode_with_minimum_sparsity(
        centered_test,
        dictionary,
        sparsity=sparsity,
        minimum_sparsity=minimum_sparsity,
    )
    health = dictionary_metrics(centered_test, dictionary, codes)
    active = np.abs(codes) > ACTIVATION_THRESHOLD
    graph_coverage = []
    mean_abs_active = []
    for atom_index in range(dictionary.shape[1]):
        covered = 0
        for _graph_index, _label, start, stop in test_slices:
            if np.any(active[atom_index, start:stop]):
                covered += 1
        graph_coverage.append(float(covered / max(len(test_slices), 1)))
        atom_active = active[atom_index]
        if np.any(atom_active):
            mean_abs_active.append(float(np.mean(np.abs(codes[atom_index, atom_active]))))
        else:
            mean_abs_active.append(0.0)
    proximity = nearest_real_patch_proximity(dictionary, centered_train, train_edge_counts)
    tops = top_activating_patch_summary(
        codes,
        test_walk_vectors,
        test_canonical_vectors,
        test_edge_counts,
        test_slices,
        top_k=top_k,
    )
    return {
        "dictionary_health": health,
        "graph_coverage": graph_coverage,
        "mean_absolute_coefficient_among_active": mean_abs_active,
        "nearest_real_patch_proximity": proximity,
        "top_activating_patches": tops,
        "atom_mass": atom_mass_summary(dictionary),
        "top5_unique_walk_mean": float(
            np.mean([item["unique_walk_vector_count"] for item in tops])
        ),
        "top5_edge_count_mean_std_across_atoms": float(
            np.std([item["edge_count_mean"] for item in tops], ddof=0)
        ),
        "top5_edge_count_means": [float(item["edge_count_mean"]) for item in tops],
    }


def matched_dictionary_similarity(
    left: np.ndarray, right: np.ndarray
) -> dict[str, Any]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    cosine = np.abs(left.T @ right)
    matched_mean, assignment = exact_maximum_cosine_assignment(cosine)
    matched_values = [float(cosine[row, column]) for row, column in enumerate(assignment)]
    left_q, _ = np.linalg.qr(left)
    right_q, _ = np.linalg.qr(right)
    singular = np.linalg.svd(left_q.T @ right_q, compute_uv=False)
    return {
        "mean_absolute_matched_atom_cosine": float(matched_mean),
        "minimum_absolute_matched_atom_cosine": float(min(matched_values)),
        "assignment": list(assignment),
        "matched_absolute_cosines": matched_values,
        "mean_principal_subspace_cosine": float(np.mean(singular)),
        "minimum_principal_subspace_cosine": float(np.min(singular)),
    }


def run_basis_characterization_fold(
    examples: Sequence[IMDBPatchGraph],
    split: FoldSplit,
    dictionaries: dict[str, np.ndarray],
    train_coordinate_mean: np.ndarray,
    *,
    sparsity: int = 2,
    minimum_sparsity: int = 1,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    example_by_index = {int(example.graph_index): example for example in examples}
    train_examples = tuple(example_by_index[index] for index in split.train_indices)
    test_examples = tuple(example_by_index[index] for index in split.test_indices)
    raw_train, _train_slices = stack_patch_graphs(train_examples)
    raw_test, test_slices = stack_patch_graphs(test_examples)
    mean = np.asarray(train_coordinate_mean, dtype=np.float64).reshape(-1, 1)
    if mean.shape[0] != raw_train.shape[0]:
        raise ValueError("train coordinate mean has the wrong dimension")
    centered_train = raw_train - mean
    centered_test = raw_test - mean
    train_edge_counts = np.concatenate(
        [np.asarray(example.edge_counts, dtype=np.int64) for example in train_examples]
    )
    test_edge_counts = np.concatenate(
        [np.asarray(example.edge_counts, dtype=np.int64) for example in test_examples]
    )
    test_walk = np.concatenate(
        [np.asarray(example.walk_vectors, dtype=np.float64) for example in test_examples],
        axis=0,
    )
    test_canonical = np.concatenate(
        [np.asarray(example.canonical_vectors, dtype=np.float64) for example in test_examples],
        axis=0,
    )

    stages = {}
    for stage_name in ("init", "final", "fixed_gaussian"):
        stages[stage_name] = characterize_dictionary_on_split(
            dictionaries[stage_name],
            centered_train,
            train_edge_counts,
            centered_test,
            test_walk,
            test_canonical,
            test_edge_counts,
            test_slices,
            sparsity=sparsity,
            minimum_sparsity=minimum_sparsity,
            top_k=top_k,
        )
    init_to_final = matched_dictionary_similarity(
        dictionaries["init"], dictionaries["final"]
    )
    return {
        "fold_index": int(split.fold_index),
        "train_graph_count": int(len(train_examples)),
        "test_graph_count": int(len(test_examples)),
        "train_patch_count": int(raw_train.shape[1]),
        "test_patch_count": int(raw_test.shape[1]),
        "stages": stages,
        "init_to_final_similarity": init_to_final,
        "gate_components": {
            "final_nondead_atoms_all_12": bool(
                stages["final"]["dictionary_health"]["nondead_atom_count"] == 12
            ),
            "final_nearest_cosine_beats_gaussian": bool(
                stages["final"]["nearest_real_patch_proximity"]["mean_nearest_absolute_cosine"]
                - stages["fixed_gaussian"]["nearest_real_patch_proximity"][
                    "mean_nearest_absolute_cosine"
                ]
                >= 0.10
            ),
            "final_nearest_cosine_not_below_init": bool(
                stages["final"]["nearest_real_patch_proximity"]["mean_nearest_absolute_cosine"]
                + 1e-12
                >= stages["init"]["nearest_real_patch_proximity"]["mean_nearest_absolute_cosine"]
            ),
            "top5_unique_walk_mean_at_least_2": bool(
                stages["final"]["top5_unique_walk_mean"] >= 2.0
            ),
            "top5_edge_count_dispersion_at_least_1": bool(
                stages["final"]["top5_edge_count_mean_std_across_atoms"] >= 1.0
            ),
        },
    }


def summarize_basis_view(folds: list[dict[str, Any]], cross_fold: dict[str, Any]) -> dict[str, Any]:
    if len(folds) != 5:
        raise ValueError("registered R1-A view requires all five folds")
    final_nondead = [
        fold["stages"]["final"]["dictionary_health"]["nondead_atom_count"] for fold in folds
    ]
    final_nearest = np.asarray(
        [
            fold["stages"]["final"]["nearest_real_patch_proximity"][
                "mean_nearest_absolute_cosine"
            ]
            for fold in folds
        ],
        dtype=np.float64,
    )
    init_nearest = np.asarray(
        [
            fold["stages"]["init"]["nearest_real_patch_proximity"][
                "mean_nearest_absolute_cosine"
            ]
            for fold in folds
        ],
        dtype=np.float64,
    )
    gaussian_nearest = np.asarray(
        [
            fold["stages"]["fixed_gaussian"]["nearest_real_patch_proximity"][
                "mean_nearest_absolute_cosine"
            ]
            for fold in folds
        ],
        dtype=np.float64,
    )
    unique_walk = np.asarray(
        [fold["stages"]["final"]["top5_unique_walk_mean"] for fold in folds], dtype=np.float64
    )
    edge_dispersion = np.asarray(
        [
            fold["stages"]["final"]["top5_edge_count_mean_std_across_atoms"]
            for fold in folds
        ],
        dtype=np.float64,
    )
    init_to_final = np.asarray(
        [
            fold["init_to_final_similarity"]["mean_absolute_matched_atom_cosine"]
            for fold in folds
        ],
        dtype=np.float64,
    )
    conditions = {
        "final_nondead_all_folds": bool(all(value == 12 for value in final_nondead)),
        "final_cross_fold_matched_cosine_mean_at_least_0_70": bool(
            float(cross_fold["final"]["matched_cosine_mean"]) >= 0.70
        ),
        "final_cross_fold_more_stable_than_init": bool(
            float(cross_fold["final"]["matched_cosine_mean"])
            > float(cross_fold["init"]["matched_cosine_mean"])
        ),
        "final_nearest_real_patch_beats_gaussian_by_0_10": bool(
            float(np.mean(final_nearest - gaussian_nearest)) >= 0.10
        ),
        "final_nearest_real_patch_not_below_init": bool(
            float(np.mean(final_nearest - init_nearest)) >= -1e-12
        ),
        "top5_unique_walk_mean_at_least_2": bool(float(np.mean(unique_walk)) >= 2.0),
        "top5_edge_count_dispersion_at_least_1": bool(float(np.mean(edge_dispersion)) >= 1.0),
    }
    return {
        "fold_count": int(len(folds)),
        "final_nondead_atom_counts": final_nondead,
        "mean_final_nearest_real_patch_cosine": float(np.mean(final_nearest)),
        "mean_init_nearest_real_patch_cosine": float(np.mean(init_nearest)),
        "mean_gaussian_nearest_real_patch_cosine": float(np.mean(gaussian_nearest)),
        "mean_final_minus_gaussian_nearest_cosine": float(
            np.mean(final_nearest - gaussian_nearest)
        ),
        "mean_final_minus_init_nearest_cosine": float(np.mean(final_nearest - init_nearest)),
        "mean_top5_unique_walk_vectors": float(np.mean(unique_walk)),
        "mean_top5_edge_count_dispersion": float(np.mean(edge_dispersion)),
        "mean_init_to_final_matched_cosine": float(np.mean(init_to_final)),
        "cross_fold_init_matched_cosine_mean": float(cross_fold["init"]["matched_cosine_mean"]),
        "cross_fold_final_matched_cosine_mean": float(cross_fold["final"]["matched_cosine_mean"]),
        "registered_gate_conditions": conditions,
        "passes_registered_view_gate": bool(all(conditions.values())),
    }


def classify_basis_views(views: dict[str, Any]) -> dict[str, Any]:
    if set(views) != {"stratified", "exact_isomorphism_grouped"}:
        return {
            "classification": "INCOMPLETE_R1A_VIEWS",
            "passes_r1a": False,
            "next_step": "Run both registered raw views before classifying R1-A.",
        }
    stratified = views["stratified"]["summary"]
    grouped = views["exact_isomorphism_grouped"]["summary"]
    if stratified["passes_registered_view_gate"] and grouped["passes_registered_view_gate"]:
        return {
            "classification": "PASS_R1A_BASIS_CHARACTERIZATION",
            "passes_r1a": True,
            "stratified_view_passes": True,
            "grouped_view_passes": True,
            "next_step": (
                "Treat KSVD as a validated label-free sparse patch basis learner on raw IMDB; "
                "optional next work is multi-view/attribute compressor analysis, not task reopening."
            ),
        }
    return {
        "classification": "FAIL_R1A_BASIS_CHARACTERIZATION",
        "passes_r1a": False,
        "stratified_view_passes": bool(stratified["passes_registered_view_gate"]),
        "grouped_view_passes": bool(grouped["passes_registered_view_gate"]),
        "next_step": (
            "Do not add restarts or reopen classification. Either accept reconstruction-valid but "
            "weak structural characterization, or design a new multi-view compressor protocol."
        ),
    }
