"""U0-P utilities for the unplanted graph patch signal-exposure gate.

This module intentionally stops before dictionary learning.  It asks whether
fixed summaries of the exact patch signals proposed for U0 contain enough
LOW/HIGH regime information to justify running KSVD at all.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .from_scratch_unplanted_representation import (
    generate_rewired_graph,
    is_connected,
    sample_graph_walk_patches,
    validate_simple_adjacency,
)


REGULARIZATION_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
FEATURE_KEYS = ("walk_mean_std", "canonical_mean_std", "edge_count_histogram")


@dataclass(frozen=True)
class GraphPatchExample:
    split: str
    label: int
    requested_swaps: int
    graph_seed: int
    features: dict[str, np.ndarray]
    walk_vectors: np.ndarray
    canonical_vectors: np.ndarray
    edge_counts: np.ndarray
    attempted_swaps: int
    simple_graph_statistics: np.ndarray


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray
    active: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.mean.size:
            raise ValueError("feature matrix has the wrong shape")
        return (values[:, self.active] - self.mean[self.active]) / self.scale[self.active]


def _seed_to_int(sequence: np.random.SeedSequence) -> int:
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def summarize_patch_vectors(
    walk_vectors: np.ndarray,
    canonical_vectors: np.ndarray,
) -> dict[str, np.ndarray]:
    walk_vectors = np.asarray(walk_vectors, dtype=np.float64)
    canonical_vectors = np.asarray(canonical_vectors, dtype=np.float64)
    if walk_vectors.ndim != 2 or walk_vectors.shape[1] != 15:
        raise ValueError("walk vectors must have shape (n_patches, 15)")
    if canonical_vectors.shape != walk_vectors.shape:
        raise ValueError("canonical vectors must match walk vector shape")
    if walk_vectors.shape[0] == 0:
        raise ValueError("at least one patch is required")

    edge_counts = np.sum(walk_vectors, axis=1).astype(np.int64)
    if np.any(edge_counts < 5) or np.any(edge_counts > 15):
        raise ValueError("walk-induced connected six-node patches must have 5..15 edges")
    histogram = np.bincount(edge_counts - 5, minlength=11).astype(np.float64)
    histogram /= float(walk_vectors.shape[0])
    return {
        "walk_mean_std": np.concatenate(
            [np.mean(walk_vectors, axis=0), np.std(walk_vectors, axis=0, ddof=0)]
        ),
        "canonical_mean_std": np.concatenate(
            [np.mean(canonical_vectors, axis=0), np.std(canonical_vectors, axis=0, ddof=0)]
        ),
        "edge_count_histogram": histogram,
    }



def simple_graph_statistics(adjacency: np.ndarray) -> np.ndarray:
    """Frozen U1A global-stat baseline in the protocol's listed order."""
    adjacency = np.asarray(adjacency, dtype=np.int8)
    validate_simple_adjacency(adjacency)
    n_nodes = adjacency.shape[0]
    degrees = np.sum(adjacency, axis=1).astype(np.float64)
    n_edges = float(np.sum(degrees) / 2.0)
    density = float(2.0 * n_edges / max(n_nodes * (n_nodes - 1), 1))
    triangle_count = float(np.trace(adjacency.astype(np.int64) @ adjacency @ adjacency) / 6.0)
    connected_triples = float(np.sum(degrees * (degrees - 1.0) / 2.0))
    transitivity = float(3.0 * triangle_count / connected_triples) if connected_triples else 0.0

    all_distances: list[int] = []
    component_count = 0
    unseen = set(range(n_nodes))
    for source in range(n_nodes):
        distances = np.full(n_nodes, -1, dtype=np.int64)
        distances[source] = 0
        queue = [source]
        for node in queue:
            for neighbor in np.flatnonzero(adjacency[node]):
                neighbor_int = int(neighbor)
                if distances[neighbor_int] < 0:
                    distances[neighbor_int] = distances[node] + 1
                    queue.append(neighbor_int)
        if source in unseen:
            component_count += 1
            unseen.difference_update(queue)
        all_distances.extend(int(value) for value in distances[source + 1 :] if value >= 0)
    average_shortest_path = float(np.mean(all_distances)) if all_distances else 0.0
    diameter = float(np.max(all_distances)) if all_distances else 0.0
    return np.asarray(
        [
            float(n_nodes),
            n_edges,
            density,
            float(np.mean(degrees)),
            float(np.std(degrees, ddof=0)),
            float(np.min(degrees)),
            float(np.max(degrees)),
            triangle_count,
            transitivity,
            average_shortest_path,
            diameter,
            float(component_count),
        ],
        dtype=np.float64,
    )

def _validate_generated_graph(adjacency: np.ndarray, accepted_swaps: int, requested: int) -> None:
    validate_simple_adjacency(adjacency)
    if adjacency.shape != (60, 60):
        raise RuntimeError("generator produced the wrong graph shape")
    if int(np.sum(adjacency) // 2) != 120:
        raise RuntimeError("generator changed the edge count")
    if not np.all(np.sum(adjacency, axis=1) == 4):
        raise RuntimeError("generator changed the degree sequence")
    if not is_connected(adjacency):
        raise RuntimeError("generator produced a disconnected graph")
    if accepted_swaps != requested:
        raise RuntimeError("generator did not accept the requested number of swaps")


def generate_graph_patch_example(
    *,
    split: str,
    label: int,
    graph_sequence: np.random.SeedSequence,
    patches_per_graph: int = 24,
    low_swap_range: tuple[int, int] = (20, 40),
    high_swap_range: tuple[int, int] = (60, 80),
) -> GraphPatchExample:
    if label not in (0, 1):
        raise ValueError("label must be 0 (LOW) or 1 (HIGH)")
    swap_sequence, generator_sequence, patch_sequence = graph_sequence.spawn(3)
    swap_rng = np.random.default_rng(swap_sequence)
    lower, upper = low_swap_range if label == 0 else high_swap_range
    requested_swaps = int(swap_rng.integers(lower, upper + 1))
    graph_seed = _seed_to_int(generator_sequence)
    graph = generate_rewired_graph(
        seed=graph_seed,
        n_accepted_swaps=requested_swaps,
    )
    _validate_generated_graph(graph.adjacency, graph.accepted_swaps, requested_swaps)

    patches = sample_graph_walk_patches(
        graph.adjacency,
        np.random.default_rng(patch_sequence),
        n_patches=patches_per_graph,
        patch_size=6,
    )
    walk_vectors = np.stack([patch.walk_order_vector for patch in patches], axis=0)
    canonical_vectors = np.stack([patch.canonical_vector for patch in patches], axis=0)
    features = summarize_patch_vectors(walk_vectors, canonical_vectors)
    return GraphPatchExample(
        split=str(split),
        label=int(label),
        requested_swaps=requested_swaps,
        graph_seed=graph_seed,
        features=features,
        walk_vectors=walk_vectors,
        canonical_vectors=canonical_vectors,
        edge_counts=np.sum(walk_vectors, axis=1).astype(np.int64),
        attempted_swaps=graph.attempted_swaps,
        simple_graph_statistics=simple_graph_statistics(graph.adjacency),
    )


def generate_balanced_split(
    *,
    split: str,
    graphs_per_class: int,
    split_sequence: np.random.SeedSequence,
    patches_per_graph: int = 24,
) -> tuple[GraphPatchExample, ...]:
    if graphs_per_class <= 0:
        raise ValueError("graphs_per_class must be positive")
    graph_parent, order_sequence = split_sequence.spawn(2)
    graph_sequences = graph_parent.spawn(2 * graphs_per_class)
    examples: list[GraphPatchExample] = []
    for index, graph_sequence in enumerate(graph_sequences):
        label = 0 if index < graphs_per_class else 1
        examples.append(
            generate_graph_patch_example(
                split=split,
                label=label,
                graph_sequence=graph_sequence,
                patches_per_graph=patches_per_graph,
            )
        )
    order = np.random.default_rng(order_sequence).permutation(len(examples))
    return tuple(examples[int(index)] for index in order)


def generate_u0p_dataset(
    master_seed: int | np.random.SeedSequence,
    *,
    train_per_class: int = 150,
    validation_per_class: int = 50,
    test_per_class: int = 100,
    patches_per_graph: int = 24,
) -> dict[str, tuple[GraphPatchExample, ...]]:
    data_sequence = (
        master_seed
        if isinstance(master_seed, np.random.SeedSequence)
        else np.random.SeedSequence(int(master_seed))
    )
    train_sequence, validation_sequence, test_sequence = data_sequence.spawn(3)
    return {
        "train": generate_balanced_split(
            split="train",
            graphs_per_class=train_per_class,
            split_sequence=train_sequence,
            patches_per_graph=patches_per_graph,
        ),
        "validation": generate_balanced_split(
            split="validation",
            graphs_per_class=validation_per_class,
            split_sequence=validation_sequence,
            patches_per_graph=patches_per_graph,
        ),
        "test": generate_balanced_split(
            split="test",
            graphs_per_class=test_per_class,
            split_sequence=test_sequence,
            patches_per_graph=patches_per_graph,
        ),
    }


def feature_matrix(
    examples: Iterable[GraphPatchExample],
    feature_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    examples = tuple(examples)
    if feature_key not in FEATURE_KEYS:
        raise ValueError(f"unknown feature key: {feature_key}")
    if not examples:
        raise ValueError("examples cannot be empty")
    values = np.stack([example.features[feature_key] for example in examples], axis=0)
    labels = np.asarray([example.label for example in examples], dtype=np.int64)
    return values, labels


def fit_standardizer(values: np.ndarray, *, tolerance: float = 1e-12) -> Standardizer:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("values must be a non-empty matrix")
    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0, ddof=0)
    active = scale > tolerance
    safe_scale = scale.copy()
    safe_scale[~active] = 1.0
    return Standardizer(mean=mean, scale=safe_scale, active=active)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    output = np.empty_like(logits)
    positive = logits >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    negative_exp = np.exp(logits[~positive])
    output[~positive] = negative_exp / (1.0 + negative_exp)
    return output


def logistic_objective(
    design: np.ndarray,
    labels: np.ndarray,
    parameters: np.ndarray,
    regularization: float,
) -> float:
    logits = design @ parameters
    loss = np.mean(np.logaddexp(0.0, logits) - labels * logits)
    return float(loss + 0.5 * regularization * np.dot(parameters[:-1], parameters[:-1]))


def fit_l2_logistic(
    values: np.ndarray,
    labels: np.ndarray,
    regularization: float,
    *,
    max_iterations: int = 100,
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Fit mean-log-loss logistic regression with an unpenalized intercept.

    Objective: mean binary cross entropy + lambda/2 * ||w||_2^2.
    A damped Newton solver is used so the regularization grid has explicit,
    package-independent semantics.
    """
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if values.ndim != 2 or labels.shape != (values.shape[0],):
        raise ValueError("incompatible values/labels shapes")
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("labels must be binary")
    if regularization <= 0:
        raise ValueError("regularization must be positive")

    design = np.column_stack([values, np.ones(values.shape[0], dtype=np.float64)])
    parameters = np.zeros(design.shape[1], dtype=np.float64)
    converged = False
    objective = logistic_objective(design, labels, parameters, regularization)
    gradient_norm = float("inf")

    for iteration in range(1, max_iterations + 1):
        probabilities = _sigmoid(design @ parameters)
        gradient = design.T @ (probabilities - labels) / values.shape[0]
        gradient[:-1] += regularization * parameters[:-1]
        gradient_norm = float(np.max(np.abs(gradient)))
        if gradient_norm <= tolerance:
            converged = True
            break

        weights = probabilities * (1.0 - probabilities)
        hessian = (design.T * weights) @ design / values.shape[0]
        hessian[:-1, :-1] += regularization * np.eye(values.shape[1])
        # Tiny damping only protects the unregularized intercept in degenerate tests.
        hessian[-1, -1] += 1e-12
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]

        directional = float(gradient @ step)
        step_scale = 1.0
        accepted = False
        while step_scale >= 2.0**-20:
            candidate = parameters - step_scale * step
            candidate_objective = logistic_objective(
                design, labels, candidate, regularization
            )
            if candidate_objective <= objective - 1e-4 * step_scale * directional:
                parameters = candidate
                objective = candidate_objective
                accepted = True
                break
            step_scale *= 0.5
        if not accepted:
            parameters = parameters - 1e-3 * gradient
            objective = logistic_objective(design, labels, parameters, regularization)
    else:
        iteration = max_iterations

    return {
        "weights": parameters[:-1],
        "intercept": float(parameters[-1]),
        "objective": float(objective),
        "iterations": int(iteration),
        "converged": bool(converged),
        "gradient_inf_norm": float(gradient_norm),
    }


def predict_l2_logistic(model: dict[str, Any], values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    logits = values @ np.asarray(model["weights"], dtype=np.float64) + float(model["intercept"])
    return (_sigmoid(logits) >= 0.5).astype(np.int64)


def balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if labels.shape != predictions.shape or labels.ndim != 1:
        raise ValueError("labels and predictions must be equal-length vectors")
    recalls = []
    for label in (0, 1):
        mask = labels == label
        if not np.any(mask):
            raise ValueError("balanced accuracy requires both classes")
        recalls.append(float(np.mean(predictions[mask] == label)))
    return float(np.mean(recalls))



def evaluate_feature_matrices(
    matrices: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    *,
    feature_key: str,
    regularization_grid: tuple[float, ...] = REGULARIZATION_GRID,
) -> dict[str, Any]:
    matrices = {split: np.asarray(matrices[split], dtype=np.float64) for split in ("train", "validation", "test")}
    labels = {split: np.asarray(labels[split], dtype=np.int64) for split in ("train", "validation", "test")}
    for split in ("train", "validation", "test"):
        if matrices[split].ndim != 2 or labels[split].shape != (matrices[split].shape[0],):
            raise ValueError("incompatible feature matrix and label shapes")

    standardizer = fit_standardizer(matrices["train"])
    standardized = {split: standardizer.transform(matrix) for split, matrix in matrices.items()}
    candidates = []
    selected: dict[str, Any] | None = None
    for regularization in regularization_grid:
        model = fit_l2_logistic(standardized["train"], labels["train"], float(regularization))
        validation_predictions = predict_l2_logistic(model, standardized["validation"])
        validation_score = balanced_accuracy(labels["validation"], validation_predictions)
        candidate = {
            "regularization": float(regularization),
            "validation_balanced_accuracy": float(validation_score),
            "model": model,
        }
        candidates.append(candidate)
        if selected is None or validation_score > selected["validation_balanced_accuracy"] + 1e-12:
            selected = candidate
        elif abs(validation_score - selected["validation_balanced_accuracy"]) <= 1e-12:
            if regularization > selected["regularization"]:
                selected = candidate
    assert selected is not None
    train_score = balanced_accuracy(labels["train"], predict_l2_logistic(selected["model"], standardized["train"]))
    test_score = balanced_accuracy(labels["test"], predict_l2_logistic(selected["model"], standardized["test"]))
    return {
        "feature_key": feature_key,
        "raw_dimension": int(matrices["train"].shape[1]),
        "active_dimension": int(np.count_nonzero(standardizer.active)),
        "selected_regularization": float(selected["regularization"]),
        "train_balanced_accuracy": float(train_score),
        "validation_balanced_accuracy": float(selected["validation_balanced_accuracy"]),
        "test_balanced_accuracy": float(test_score),
        "validation_grid": [
            {
                "regularization": item["regularization"],
                "balanced_accuracy": item["validation_balanced_accuracy"],
                "converged": item["model"]["converged"],
                "iterations": item["model"]["iterations"],
                "gradient_inf_norm": item["model"]["gradient_inf_norm"],
            }
            for item in candidates
        ],
        "selected_model_converged": bool(selected["model"]["converged"]),
        "selected_model_iterations": int(selected["model"]["iterations"]),
        "class_counts": {
            split: {str(label): int(np.count_nonzero(labels[split] == label)) for label in (0, 1)}
            for split in ("train", "validation", "test")
        },
    }

def evaluate_feature_control(
    dataset: dict[str, tuple[GraphPatchExample, ...]],
    feature_key: str,
    *,
    regularization_grid: tuple[float, ...] = REGULARIZATION_GRID,
    label_override: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    matrices: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    for split in ("train", "validation", "test"):
        matrices[split], labels[split] = feature_matrix(dataset[split], feature_key)
        if label_override is not None:
            candidate = np.asarray(label_override[split], dtype=np.int64)
            if candidate.shape != labels[split].shape:
                raise ValueError("label override has the wrong shape")
            labels[split] = candidate
    return evaluate_feature_matrices(
        matrices, labels, feature_key=feature_key, regularization_grid=regularization_grid
    )

def shuffled_labels(
    dataset: dict[str, tuple[GraphPatchExample, ...]],
    sequence: np.random.SeedSequence,
) -> dict[str, np.ndarray]:
    split_sequences = sequence.spawn(3)
    result: dict[str, np.ndarray] = {}
    for split, split_sequence in zip(("train", "validation", "test"), split_sequences):
        labels = np.asarray([example.label for example in dataset[split]], dtype=np.int64)
        result[split] = labels[np.random.default_rng(split_sequence).permutation(labels.size)]
    return result


def dataset_summary(dataset: dict[str, tuple[GraphPatchExample, ...]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split, examples in dataset.items():
        requested = np.asarray([example.requested_swaps for example in examples], dtype=np.int64)
        attempted = np.asarray([example.attempted_swaps for example in examples], dtype=np.int64)
        edge_counts = np.concatenate([example.edge_counts for example in examples])
        labels = np.asarray([example.label for example in examples], dtype=np.int64)
        summary[split] = {
            "graph_count": int(len(examples)),
            "patch_count": int(edge_counts.size),
            "class_counts": {
                str(label): int(np.count_nonzero(labels == label)) for label in (0, 1)
            },
            "requested_swaps": {
                "minimum": int(np.min(requested)),
                "maximum": int(np.max(requested)),
                "mean": float(np.mean(requested)),
            },
            "swap_attempts": {
                "minimum": int(np.min(attempted)),
                "maximum": int(np.max(attempted)),
                "mean": float(np.mean(attempted)),
            },
            "patch_edges": {
                "minimum": int(np.min(edge_counts)),
                "maximum": int(np.max(edge_counts)),
                "mean": float(np.mean(edge_counts)),
            },
        }
    return summary
