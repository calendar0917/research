"""U0-D dictionary optimization and health utilities for walk-order patches."""
from __future__ import annotations

from typing import Any

import numpy as np

from .from_scratch_unplanted_signal import GraphPatchExample
from .ksvd import ksvd


EPS = 1e-12
ACTIVATION_THRESHOLD = 1e-10


def stack_walk_patch_matrix(examples: tuple[GraphPatchExample, ...] | list[GraphPatchExample]) -> np.ndarray:
    examples = tuple(examples)
    if not examples:
        raise ValueError("examples cannot be empty")
    patches = np.concatenate([example.walk_vectors for example in examples], axis=0)
    if patches.ndim != 2 or patches.shape[1] != 15:
        raise ValueError("expected walk patches with dimension 15")
    return patches.T.astype(np.float64, copy=False)


def deterministic_maximin_initialization(
    centered_train: np.ndarray,
    n_atoms: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select one deployable, deterministic, diverse set of training columns."""
    values = np.asarray(centered_train, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < n_atoms:
        raise ValueError("centered_train must contain at least n_atoms columns")
    norms = np.linalg.norm(values, axis=0)
    valid = np.flatnonzero(norms > EPS)
    if valid.size < n_atoms:
        raise ValueError("not enough nonzero centered training columns")
    normalized = values[:, valid] / norms[valid][None, :]

    def lex_key(position: int) -> tuple[float, ...]:
        return tuple(float(value) for value in normalized[:, position])

    maximum_norm = float(np.max(norms[valid]))
    first_candidates = [
        position
        for position, source in enumerate(valid)
        if abs(float(norms[source]) - maximum_norm) <= 1e-12
    ]
    first = min(first_candidates, key=lambda position: (lex_key(position), int(valid[position])))
    selected_positions = [int(first)]
    selection_scores: list[float | None] = [None]

    while len(selected_positions) < n_atoms:
        selected = normalized[:, selected_positions]
        novelty = 1.0 - np.max(np.abs(selected.T @ normalized), axis=0)
        novelty[selected_positions] = -np.inf
        best_score = float(np.max(novelty))
        candidates = [
            position
            for position in range(normalized.shape[1])
            if np.isfinite(novelty[position])
            and abs(float(novelty[position]) - best_score) <= 1e-12
        ]
        best = min(candidates, key=lambda position: (lex_key(position), int(valid[position])))
        selected_positions.append(int(best))
        selection_scores.append(best_score)

    selected_indices = [int(valid[position]) for position in selected_positions]
    dictionary = normalized[:, selected_positions].copy()
    return dictionary, {
        "name": "deterministic_maximin",
        "selected_training_indices": selected_indices,
        "selected_raw_norms": [float(norms[index]) for index in selected_indices],
        "selection_novelty_scores": selection_scores,
        "selected_unique_column_count": int(
            np.unique(values[:, selected_indices].T, axis=0).shape[0]
        ),
    }


def encode_with_dictionary(
    values: np.ndarray,
    dictionary: np.ndarray,
    *,
    sparsity: int = 2,
    minimum_sparsity: int = 1,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    dictionary = np.asarray(dictionary, dtype=np.float64)
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
    return codes


def dictionary_metrics(
    values: np.ndarray,
    dictionary: np.ndarray,
    codes: np.ndarray,
    *,
    activation_threshold: float = ACTIVATION_THRESHOLD,
    dead_frequency: float = 0.005,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    dictionary = np.asarray(dictionary, dtype=np.float64)
    codes = np.asarray(codes, dtype=np.float64)
    if values.ndim != 2 or dictionary.ndim != 2 or codes.ndim != 2:
        raise ValueError("values, dictionary, and codes must be matrices")
    if dictionary.shape[0] != values.shape[0] or codes.shape != (
        dictionary.shape[1],
        values.shape[1],
    ):
        raise ValueError("dictionary/code shapes are incompatible with values")

    residual = values - dictionary @ codes
    residual_frobenius = float(np.linalg.norm(residual, "fro"))
    signal_frobenius = float(np.linalg.norm(values, "fro"))
    relative = residual_frobenius / max(signal_frobenius, EPS)
    nmse = float(np.sum(residual**2) / max(float(np.sum(values**2)), EPS))

    active = np.abs(codes) > activation_threshold
    activation_counts = np.sum(active, axis=1).astype(np.int64)
    frequencies = activation_counts / max(codes.shape[1], 1)
    total_activations = int(np.sum(activation_counts))
    if total_activations > 0:
        shares = activation_counts.astype(np.float64) / total_activations
        positive = shares > 0
        entropy = float(-np.sum(shares[positive] * np.log(shares[positive])))
        effective = float(np.exp(entropy))
        maximum_share = float(np.max(shares))
    else:
        entropy = 0.0
        effective = 0.0
        maximum_share = 0.0

    gram = np.abs(dictionary.T @ dictionary)
    np.fill_diagonal(gram, 0.0)
    maximum_coherence = float(np.max(gram)) if dictionary.shape[1] > 1 else 0.0
    nonzeros_per_patch = np.sum(active, axis=0)
    return {
        "patch_count": int(values.shape[1]),
        "relative_reconstruction_error": float(relative),
        "nmse": nmse,
        "mean_nonzeros_per_patch": float(np.mean(nonzeros_per_patch)),
        "minimum_nonzeros_per_patch": int(np.min(nonzeros_per_patch)),
        "maximum_nonzeros_per_patch": int(np.max(nonzeros_per_patch)),
        "activation_counts": activation_counts.tolist(),
        "activation_frequencies": frequencies.tolist(),
        "activation_entropy": entropy,
        "effective_atom_count": effective,
        "dead_atom_count": int(np.count_nonzero(frequencies < dead_frequency)),
        "nondead_atom_count": int(np.count_nonzero(frequencies >= dead_frequency)),
        "dead_frequency_threshold": float(dead_frequency),
        "maximum_absolute_offdiagonal_coherence": maximum_coherence,
        "maximum_activation_share": maximum_share,
    }


def run_u0d_dictionary_audit(
    dataset: dict[str, tuple[GraphPatchExample, ...]],
    *,
    n_atoms: int = 12,
    sparsity: int = 2,
    minimum_sparsity: int = 1,
    n_iterations: int = 25,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    raw = {split: stack_walk_patch_matrix(examples) for split, examples in dataset.items()}
    train_mean = np.mean(raw["train"], axis=1, keepdims=True)
    centered = {split: values - train_mean for split, values in raw.items()}
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

    stage_metrics: dict[str, dict[str, Any]] = {"init": {}, "final": {}}
    for stage, dictionary in (("init", initial_dictionary), ("final", learned_dictionary)):
        for split in ("train", "validation", "test"):
            codes = encode_with_dictionary(
                centered[split],
                dictionary,
                sparsity=sparsity,
                minimum_sparsity=minimum_sparsity,
            )
            stage_metrics[stage][split] = dictionary_metrics(
                centered[split], dictionary, codes
            )

    init_test = stage_metrics["init"]["test"]["relative_reconstruction_error"]
    final_test = stage_metrics["final"]["test"]["relative_reconstruction_error"]
    relative_reduction = float((init_test - final_test) / max(init_test, EPS))
    final_test_health = stage_metrics["final"]["test"]
    health = {
        "test_relative_reconstruction_reduction": relative_reduction,
        "passes_10_percent_reduction": bool(relative_reduction >= 0.10),
        "passes_nondead_atoms": bool(final_test_health["nondead_atom_count"] >= 6),
        "passes_effective_atom_count": bool(final_test_health["effective_atom_count"] >= 4.0),
        "passes_coherence": bool(
            final_test_health["maximum_absolute_offdiagonal_coherence"] < 0.95
        ),
    }
    health["passes_all_per_replicate_health"] = bool(
        health["passes_10_percent_reduction"]
        and health["passes_nondead_atoms"]
        and health["passes_effective_atom_count"]
        and health["passes_coherence"]
    )

    result = {
        "patch_counts": {split: int(values.shape[1]) for split, values in raw.items()},
        "train_coordinate_mean": train_mean[:, 0].tolist(),
        "initialization": initialization,
        "config": {
            "patch_dimension": int(raw["train"].shape[0]),
            "n_atoms": int(n_atoms),
            "sparsity": int(sparsity),
            "minimum_sparsity": int(minimum_sparsity),
            "n_iterations": int(n_iterations),
            "ksvd_internal_seed": 0,
            "per_patch_normalization": False,
            "centering": "train_coordinate_mean",
        },
        "stages": stage_metrics,
        "training_reconstruction_curve_internal": [
            float(value) for value in training_info["recon_curve"]
        ],
        "training_final_internal_relative_reconstruction_error": float(
            training_info["recon_rel"]
        ),
        "health": health,
    }
    return result, initial_dictionary, learned_dictionary


def exact_maximum_cosine_assignment(similarity: np.ndarray) -> tuple[float, tuple[int, ...]]:
    """Exact maximum-sum assignment by bitmask DP; intended for K=12 audits."""
    similarity = np.asarray(similarity, dtype=np.float64)
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("similarity must be square")
    size = similarity.shape[0]
    if size > 16:
        raise ValueError("bitmask assignment is restricted to at most 16 atoms")
    state_count = 1 << size
    scores = np.full(state_count, -np.inf, dtype=np.float64)
    parents = np.full(state_count, -1, dtype=np.int16)
    scores[0] = 0.0
    for mask in range(state_count):
        row = int(mask.bit_count())
        if row >= size or not np.isfinite(scores[mask]):
            continue
        for column in range(size):
            bit = 1 << column
            if mask & bit:
                continue
            candidate_mask = mask | bit
            candidate_score = scores[mask] + similarity[row, column]
            if candidate_score > scores[candidate_mask] + 1e-15:
                scores[candidate_mask] = candidate_score
                parents[candidate_mask] = column
            elif abs(candidate_score - scores[candidate_mask]) <= 1e-15:
                if parents[candidate_mask] < 0 or column < parents[candidate_mask]:
                    parents[candidate_mask] = column
    assignment = [-1] * size
    mask = state_count - 1
    for row in range(size - 1, -1, -1):
        column = int(parents[mask])
        assignment[row] = column
        mask ^= 1 << column
    return float(scores[-1] / size), tuple(assignment)


def cross_replicate_dictionary_similarity(
    dictionaries: list[np.ndarray],
) -> dict[str, Any]:
    if len(dictionaries) < 2:
        return {"pairs": [], "matched_cosine_mean": None, "subspace_cosine_mean": None}
    pairs = []
    for left_index in range(len(dictionaries)):
        left = np.asarray(dictionaries[left_index], dtype=np.float64)
        left_q, _ = np.linalg.qr(left)
        for right_index in range(left_index + 1, len(dictionaries)):
            right = np.asarray(dictionaries[right_index], dtype=np.float64)
            right_q, _ = np.linalg.qr(right)
            cosine = np.abs(left.T @ right)
            matched_mean, assignment = exact_maximum_cosine_assignment(cosine)
            singular_values = np.linalg.svd(left_q.T @ right_q, compute_uv=False)
            pairs.append(
                {
                    "left_replicate_index": int(left_index),
                    "right_replicate_index": int(right_index),
                    "mean_absolute_matched_atom_cosine": float(matched_mean),
                    "minimum_absolute_matched_atom_cosine": float(
                        min(cosine[row, column] for row, column in enumerate(assignment))
                    ),
                    "assignment": list(assignment),
                    "mean_principal_subspace_cosine": float(np.mean(singular_values)),
                    "minimum_principal_subspace_cosine": float(np.min(singular_values)),
                }
            )
    return {
        "pairs": pairs,
        "matched_cosine_mean": float(
            np.mean([item["mean_absolute_matched_atom_cosine"] for item in pairs])
        ),
        "matched_cosine_minimum_pair": float(
            np.min([item["mean_absolute_matched_atom_cosine"] for item in pairs])
        ),
        "subspace_cosine_mean": float(
            np.mean([item["mean_principal_subspace_cosine"] for item in pairs])
        ),
        "subspace_cosine_minimum_pair": float(
            np.min([item["mean_principal_subspace_cosine"] for item in pairs])
        ),
    }
