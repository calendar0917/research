"""Patch compression and graph stitching metrics for ordered overlap covers."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Sequence

import numpy as np

from .from_scratch_unplanted_representation import upper_triangle_edges, validate_simple_adjacency
from .overlap_cover import PatchCover, cover_vectors


EPS = 1e-12


@dataclass(frozen=True)
class CoverExample:
    graph_index: int
    family: str
    target_degree: int
    adjacency: np.ndarray
    cover: PatchCover
    patch_vectors: np.ndarray


def make_cover_example(
    graph_index: int,
    family: str,
    target_degree: int,
    adjacency: np.ndarray,
    cover: PatchCover,
) -> CoverExample:
    validate_simple_adjacency(adjacency)
    vectors = cover_vectors(cover).astype(np.float64, copy=False)
    return CoverExample(
        graph_index=int(graph_index),
        family=str(family),
        target_degree=int(target_degree),
        adjacency=np.asarray(adjacency, dtype=np.int8),
        cover=cover,
        patch_vectors=vectors,
    )


def stack_cover_examples(examples: Sequence[CoverExample]) -> np.ndarray:
    examples = tuple(examples)
    if not examples:
        raise ValueError("examples cannot be empty")
    dimension = examples[0].patch_vectors.shape[1]
    if any(
        example.patch_vectors.ndim != 2
        or example.patch_vectors.shape[1] != dimension
        for example in examples
    ):
        raise ValueError("all examples must have equal-dimensional patch matrices")
    return np.concatenate([example.patch_vectors for example in examples], axis=0).T


def fit_pca_basis(centered_train: np.ndarray, rank: int) -> np.ndarray:
    values = np.asarray(centered_train, dtype=np.float64)
    if values.ndim != 2 or not 1 <= rank <= min(values.shape):
        raise ValueError("invalid centered train matrix or PCA rank")
    left, _singular, _right = np.linalg.svd(values, full_matrices=False)
    return left[:, :rank]


def reconstruct_with_pca(centered_values: np.ndarray, basis: np.ndarray) -> np.ndarray:
    values = np.asarray(centered_values, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    if values.ndim != 2 or basis.ndim != 2 or values.shape[0] != basis.shape[0]:
        raise ValueError("PCA values and basis are incompatible")
    return basis @ (basis.T @ values)


def _binary_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth = np.asarray(truth, dtype=np.int8)
    prediction = np.asarray(prediction, dtype=np.int8)
    if truth.shape != prediction.shape or truth.ndim != 1:
        raise ValueError("binary metrics require equal vectors")
    true_positive = int(np.count_nonzero((truth == 1) & (prediction == 1)))
    false_positive = int(np.count_nonzero((truth == 0) & (prediction == 1)))
    false_negative = int(np.count_nonzero((truth == 1) & (prediction == 0)))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, EPS)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def stitch_patch_predictions(
    example: CoverExample,
    predicted_patch_vectors: np.ndarray,
    *,
    threshold: float = 0.5,
    local_pair_weights: np.ndarray | None = None,
    exact_residual_edges: set[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Map local pair predictions to global pairs and evaluate their mean stitch."""
    predicted = np.asarray(predicted_patch_vectors, dtype=np.float64)
    truth_patches = np.asarray(example.patch_vectors, dtype=np.float64)
    if predicted.shape != truth_patches.shape:
        raise ValueError("predicted patches must match the cover patch matrix")
    patch_size = len(example.cover.patches[0].node_ids)
    local_edges = upper_triangle_edges(patch_size)
    if predicted.shape[1] != len(local_edges):
        raise ValueError("patch vector dimension does not match patch_size")
    if local_pair_weights is None:
        pair_weights = np.ones(len(local_edges), dtype=np.float64)
    else:
        pair_weights = np.asarray(local_pair_weights, dtype=np.float64)
        if (
            pair_weights.shape != (len(local_edges),)
            or not np.all(np.isfinite(pair_weights))
            or np.any(pair_weights <= 0.0)
        ):
            raise ValueError("local_pair_weights must be finite positive slot weights")

    occurrence_predictions: dict[tuple[int, int], list[float]] = {}
    occurrence_weights: dict[tuple[int, int], list[float]] = {}
    for patch, vector in zip(example.cover.patches, predicted):
        for edge_index, (value, (left_slot, right_slot)) in enumerate(
            zip(vector, local_edges)
        ):
            pair = tuple(
                sorted((patch.node_ids[left_slot], patch.node_ids[right_slot]))
            )
            occurrence_predictions.setdefault(pair, []).append(float(value))
            occurrence_weights.setdefault(pair, []).append(
                float(pair_weights[edge_index])
            )

    n_nodes = example.adjacency.shape[0]
    all_pairs = list(combinations(range(n_nodes), 2))
    truth = np.asarray(
        [example.adjacency[left, right] for left, right in all_pairs],
        dtype=np.float64,
    )
    pair_to_position = {pair: index for index, pair in enumerate(all_pairs)}
    full_prediction = np.zeros(len(all_pairs), dtype=np.float64)
    observed_positions = []
    repeated_stds = []
    repeated_ranges = []
    for pair, values in occurrence_predictions.items():
        position = pair_to_position[pair]
        full_prediction[position] = float(
            np.average(values, weights=occurrence_weights[pair])
        )
        observed_positions.append(position)
        if len(values) > 1:
            repeated_stds.append(float(np.std(values, ddof=0)))
            repeated_ranges.append(float(np.max(values) - np.min(values)))
    if exact_residual_edges is not None:
        observed_pairs = set(occurrence_predictions)
        for raw_pair in exact_residual_edges:
            pair = tuple(sorted((int(raw_pair[0]), int(raw_pair[1]))))
            if pair in observed_pairs:
                raise ValueError("residual edges must be unobserved by all patches")
            if pair not in pair_to_position:
                raise ValueError("residual edge endpoint is outside the graph")
            if example.adjacency[pair[0], pair[1]] == 0:
                raise ValueError("exact residual sidecar may contain only true edges")
            full_prediction[pair_to_position[pair]] = 1.0

    observed_positions_array = np.asarray(sorted(observed_positions), dtype=np.int64)
    observed_truth = truth[observed_positions_array]
    observed_prediction = full_prediction[observed_positions_array]

    patch_residual = predicted - truth_patches
    patch_relative = float(
        np.linalg.norm(patch_residual, "fro")
        / max(np.linalg.norm(truth_patches, "fro"), EPS)
    )
    observed_binary = (observed_prediction >= threshold).astype(np.int8)
    full_binary = (full_prediction >= threshold).astype(np.int8)
    observed_metrics = _binary_metrics(observed_truth.astype(np.int8), observed_binary)
    full_metrics = _binary_metrics(truth.astype(np.int8), full_binary)
    return {
        "patch_relative_error": patch_relative,
        "observed_pair_count": int(observed_positions_array.size),
        "observed_pair_rmse": float(
            np.sqrt(np.mean((observed_prediction - observed_truth) ** 2))
        ),
        "observed_pair_accuracy": float(np.mean(observed_binary == observed_truth)),
        "observed_edge_precision": observed_metrics["precision"],
        "observed_edge_recall": observed_metrics["recall"],
        "observed_edge_f1": observed_metrics["f1"],
        "full_adjacency_rmse": float(
            np.sqrt(np.mean((full_prediction - truth) ** 2))
        ),
        "full_adjacency_accuracy": float(np.mean(full_binary == truth)),
        "full_edge_precision": full_metrics["precision"],
        "full_edge_recall": full_metrics["recall"],
        "full_edge_f1": full_metrics["f1"],
        "repeated_pair_count": int(len(repeated_stds)),
        "repeated_pair_disagreement_std_mean": float(np.mean(repeated_stds))
        if repeated_stds
        else 0.0,
        "repeated_pair_disagreement_range_mean": float(np.mean(repeated_ranges))
        if repeated_ranges
        else 0.0,
        "prediction_minimum": float(np.min(predicted)),
        "prediction_maximum": float(np.max(predicted)),
    }


def graph_balanced_stitch_summary(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    rows = tuple(rows)
    if not rows:
        raise ValueError("stitch rows cannot be empty")
    metrics = (
        "patch_relative_error",
        "observed_pair_rmse",
        "observed_pair_accuracy",
        "observed_edge_precision",
        "observed_edge_recall",
        "observed_edge_f1",
        "full_adjacency_rmse",
        "full_adjacency_accuracy",
        "full_edge_precision",
        "full_edge_recall",
        "full_edge_f1",
        "repeated_pair_count",
        "repeated_pair_disagreement_std_mean",
        "repeated_pair_disagreement_range_mean",
    )
    return {
        metric: float(np.mean([float(row[metric]) for row in rows]))
        for metric in metrics
    }
