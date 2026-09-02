"""G0 hidden-motif discovery from complete synthetic graphs.

The learner receives only canonicalized patch columns. Hidden cell boundaries and
motif IDs are retained exclusively for extraction/evaluation controls.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations
from typing import Any, Iterable

import numpy as np

from .from_scratch_recovery import (
    EPS,
    align_codes_to_truth,
    binary_reconstruction_metrics,
    normalize_columns,
    optimal_atom_alignment,
    sparse_code_matrix,
    support_metrics,
    upper_triangle_edges,
)
from .ksvd import ksvd


MOTIF_NAMES = ("triangle", "four_cycle", "three_star", "five_path")
MOTIF_PROBABILITIES = np.asarray([0.35, 0.25, 0.25, 0.15], dtype=np.float64)
MOTIF_EDGE_LISTS: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 1), (1, 2), (0, 2)),
    ((0, 1), (1, 2), (2, 3), (3, 0)),
    ((0, 1), (0, 2), (0, 3)),
    ((0, 1), (1, 2), (2, 3), (3, 4)),
)


@dataclass(frozen=True)
class HiddenCell:
    graph_index: int
    cell_index: int
    motif_index: int
    node_ids: tuple[int, ...]


@dataclass(frozen=True)
class HiddenMotifGraph:
    adjacency: np.ndarray
    cells: tuple[HiddenCell, ...]


@dataclass(frozen=True)
class HiddenMotifDataset:
    name: str
    D_true: np.ndarray
    true_edge_supports: tuple[tuple[int, ...], ...]
    motif_names: tuple[str, ...]
    train_graphs: tuple[HiddenMotifGraph, ...]
    test_graphs: tuple[HiddenMotifGraph, ...]
    Y_train: np.ndarray
    train_labels: np.ndarray
    train_graph_ids: np.ndarray
    train_cell_ids: np.ndarray
    Y_test: np.ndarray
    test_labels: np.ndarray
    test_graph_ids: np.ndarray
    test_cell_ids: np.ndarray
    metadata: dict[str, Any]


@lru_cache(maxsize=None)
def _canonical_permutation_index_map(n_nodes: int) -> np.ndarray:
    """Map each permuted upper-triangle position back to a raw vector index."""
    edges = upper_triangle_edges(n_nodes)
    edge_to_index = {edge: index for index, edge in enumerate(edges)}
    rows: list[list[int]] = []
    for perm in permutations(range(n_nodes)):
        row: list[int] = []
        for left, right in edges:
            source_left = int(perm[left])
            source_right = int(perm[right])
            source = tuple(sorted((source_left, source_right)))
            row.append(edge_to_index[source])
        rows.append(row)
    return np.asarray(rows, dtype=np.int16)


def adjacency_to_upper_vector(adjacency: np.ndarray) -> np.ndarray:
    adjacency = np.asarray(adjacency)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    return np.asarray(
        [adjacency[left, right] for left, right in upper_triangle_edges(adjacency.shape[0])],
        dtype=np.float64,
    )


def upper_vector_to_adjacency(vector: np.ndarray, n_nodes: int = 6) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    edges = upper_triangle_edges(n_nodes)
    if vector.shape != (len(edges),):
        raise ValueError(f"expected vector shape {(len(edges),)}, got {vector.shape}")
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for value, (left, right) in zip(vector, edges):
        adjacency[left, right] = value
        adjacency[right, left] = value
    return adjacency


def exact_canonical_vector(adjacency: np.ndarray) -> np.ndarray:
    """Return lexicographically minimum upper-triangle vector over all permutations."""
    adjacency = np.asarray(adjacency)
    if adjacency.shape != (6, 6):
        raise ValueError("G0 canonicalization requires a 6x6 adjacency")
    if not np.allclose(adjacency, adjacency.T):
        raise ValueError("adjacency must be symmetric")
    raw = adjacency_to_upper_vector(adjacency).astype(np.int8)
    if not np.all((raw == 0) | (raw == 1)):
        raise ValueError("G0 exact canonicalization expects binary adjacency")
    candidates = raw[_canonical_permutation_index_map(6)]
    # Binary integer ordering with the first vector coordinate as the most
    # significant bit is exactly lexicographic ordering for equal-length rows.
    weights = (1 << np.arange(candidates.shape[1] - 1, -1, -1)).astype(np.int64)
    codes = candidates.astype(np.int64) @ weights
    return candidates[int(np.argmin(codes))].astype(np.float64)


def motif_template(motif_index: int) -> np.ndarray:
    adjacency = np.zeros((6, 6), dtype=np.int8)
    for left, right in MOTIF_EDGE_LISTS[motif_index]:
        adjacency[left, right] = 1
        adjacency[right, left] = 1
    return adjacency


def true_motif_dictionary() -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    columns = [exact_canonical_vector(motif_template(index)) for index in range(len(MOTIF_NAMES))]
    raw = np.column_stack(columns)
    if np.unique(raw.T, axis=0).shape[0] != len(MOTIF_NAMES):
        raise RuntimeError("motif templates are not canonically distinct")
    supports = tuple(
        tuple(int(index) for index in np.flatnonzero(raw[:, atom] > 0.5))
        for atom in range(raw.shape[1])
    )
    return normalize_columns(raw), supports


def _forced_motif_assignments(
    *,
    rng: np.random.Generator,
    n_graphs: int,
    cells_per_graph: int,
) -> np.ndarray:
    labels = rng.choice(
        len(MOTIF_NAMES),
        size=(n_graphs, cells_per_graph),
        p=MOTIF_PROBABILITIES,
    ).astype(np.int64)
    if labels.size < len(MOTIF_NAMES):
        raise ValueError("dataset must have at least one cell per motif")
    labels.reshape(-1)[: len(MOTIF_NAMES)] = np.arange(len(MOTIF_NAMES))
    return labels


def generate_hidden_motif_graphs(
    *,
    seed: int,
    n_graphs: int,
    cells_per_graph: int = 12,
) -> tuple[HiddenMotifGraph, ...]:
    if n_graphs < 1 or cells_per_graph < 2:
        raise ValueError("need at least one graph and two cells per graph")
    rng = np.random.default_rng(seed)
    assignments = _forced_motif_assignments(
        rng=rng,
        n_graphs=n_graphs,
        cells_per_graph=cells_per_graph,
    )
    graphs: list[HiddenMotifGraph] = []
    nodes_per_graph = cells_per_graph * 6
    for graph_index in range(n_graphs):
        adjacency = np.zeros((nodes_per_graph, nodes_per_graph), dtype=np.int8)
        for cell_index, motif_index in enumerate(assignments[graph_index]):
            offset = 6 * cell_index
            template = motif_template(int(motif_index))
            adjacency[offset : offset + 6, offset : offset + 6] = template

        # A cross-cell backbone visits all nodes and makes the full graph
        # connected without adding any edge inside a cell.
        backbone = [6 * cell + slot for slot in range(6) for cell in range(cells_per_graph)]
        for left, right in zip(backbone[:-1], backbone[1:]):
            if left // 6 == right // 6:
                raise RuntimeError("bridge backbone accidentally entered one cell")
            adjacency[left, right] = 1
            adjacency[right, left] = 1

        old_to_new = rng.permutation(nodes_per_graph)
        permuted = np.zeros_like(adjacency)
        permuted[np.ix_(old_to_new, old_to_new)] = adjacency
        cells: list[HiddenCell] = []
        for cell_index, motif_index in enumerate(assignments[graph_index]):
            old_nodes = np.arange(6 * cell_index, 6 * cell_index + 6)
            # Sorting is deliberate: the extractor receives only a node set,
            # not the original semantic local ordering.
            node_ids = tuple(sorted(int(value) for value in old_to_new[old_nodes]))
            cells.append(
                HiddenCell(
                    graph_index=graph_index,
                    cell_index=cell_index,
                    motif_index=int(motif_index),
                    node_ids=node_ids,
                )
            )
        graphs.append(HiddenMotifGraph(adjacency=permuted, cells=tuple(cells)))
    return tuple(graphs)


def is_connected(adjacency: np.ndarray) -> bool:
    adjacency = np.asarray(adjacency)
    if adjacency.shape[0] == 0:
        return True
    seen = {0}
    frontier = [0]
    while frontier:
        node = frontier.pop()
        for neighbor in np.flatnonzero(adjacency[node] != 0):
            neighbor_int = int(neighbor)
            if neighbor_int not in seen:
                seen.add(neighbor_int)
                frontier.append(neighbor_int)
    return len(seen) == adjacency.shape[0]


def extract_oracle_cell_patches(
    graphs: Iterable[HiddenMotifGraph],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache: dict[bytes, np.ndarray] = {}
    columns: list[np.ndarray] = []
    labels: list[int] = []
    graph_ids: list[int] = []
    cell_ids: list[int] = []
    for graph in graphs:
        for cell in graph.cells:
            nodes = np.asarray(cell.node_ids, dtype=np.int64)
            induced = graph.adjacency[np.ix_(nodes, nodes)]
            raw_key = adjacency_to_upper_vector(induced).astype(np.int8).tobytes()
            canonical = cache.get(raw_key)
            if canonical is None:
                canonical = exact_canonical_vector(induced)
                cache[raw_key] = canonical
            columns.append(canonical)
            labels.append(cell.motif_index)
            graph_ids.append(cell.graph_index)
            cell_ids.append(cell.cell_index)
    if not columns:
        raise ValueError("no cells were extracted")
    return (
        np.column_stack(columns),
        np.asarray(labels, dtype=np.int64),
        np.asarray(graph_ids, dtype=np.int64),
        np.asarray(cell_ids, dtype=np.int64),
    )


def make_g0_dataset(
    *,
    seed: int = 20260731,
    n_train_graphs: int = 100,
    n_test_graphs: int = 30,
    cells_per_graph: int = 12,
) -> HiddenMotifDataset:
    train_graphs = generate_hidden_motif_graphs(
        seed=seed,
        n_graphs=n_train_graphs,
        cells_per_graph=cells_per_graph,
    )
    # Keep test generation independent but reproducible.
    test_graphs = generate_hidden_motif_graphs(
        seed=seed + 1_000_003,
        n_graphs=n_test_graphs,
        cells_per_graph=cells_per_graph,
    )
    Y_train, train_labels, train_graph_ids, train_cell_ids = extract_oracle_cell_patches(train_graphs)
    Y_test, test_labels, test_graph_ids, test_cell_ids = extract_oracle_cell_patches(test_graphs)
    D_true, supports = true_motif_dictionary()
    train_counts = np.bincount(train_labels, minlength=len(MOTIF_NAMES))
    test_counts = np.bincount(test_labels, minlength=len(MOTIF_NAMES))
    expected_train = D_true[:, train_labels] * np.linalg.norm(Y_train, axis=0)[None, :]
    expected_test = D_true[:, test_labels] * np.linalg.norm(Y_test, axis=0)[None, :]
    if not np.array_equal(Y_train, expected_train) or not np.array_equal(Y_test, expected_test):
        raise RuntimeError("extracted canonical patches disagree with hidden motif records")
    return HiddenMotifDataset(
        name="G0_complete_graph_hidden_motif_cells",
        D_true=D_true,
        true_edge_supports=supports,
        motif_names=MOTIF_NAMES,
        train_graphs=train_graphs,
        test_graphs=test_graphs,
        Y_train=Y_train,
        train_labels=train_labels,
        train_graph_ids=train_graph_ids,
        train_cell_ids=train_cell_ids,
        Y_test=Y_test,
        test_labels=test_labels,
        test_graph_ids=test_graph_ids,
        test_cell_ids=test_cell_ids,
        metadata={
            "seed": int(seed),
            "n_train_graphs": int(n_train_graphs),
            "n_test_graphs": int(n_test_graphs),
            "cells_per_graph": int(cells_per_graph),
            "nodes_per_graph": int(6 * cells_per_graph),
            "patch_size": 6,
            "vector_dimension": 15,
            "motif_names": list(MOTIF_NAMES),
            "motif_probabilities": MOTIF_PROBABILITIES.tolist(),
            "train_patch_count": int(Y_train.shape[1]),
            "test_patch_count": int(Y_test.shape[1]),
            "train_motif_counts": train_counts.tolist(),
            "test_motif_counts": test_counts.tolist(),
            "train_unique_canonical_patch_count": int(np.unique(Y_train.T, axis=0).shape[0]),
            "test_unique_canonical_patch_count": int(np.unique(Y_test.T, axis=0).shape[0]),
            "all_train_graphs_connected": bool(all(is_connected(graph.adjacency) for graph in train_graphs)),
            "all_test_graphs_connected": bool(all(is_connected(graph.adjacency) for graph in test_graphs)),
        },
    )


def _lex_key(vector: np.ndarray) -> tuple[float, ...]:
    return tuple(float(value) for value in np.asarray(vector).tolist())


def deterministic_maximin_initialization(
    Y: np.ndarray,
    n_atoms: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Single deterministic diversity initialization from training columns."""
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2 or Y.shape[1] < n_atoms:
        raise ValueError("Y must contain at least n_atoms columns")
    norms = np.linalg.norm(Y, axis=0)
    valid = np.flatnonzero(norms > EPS)
    if valid.size < n_atoms:
        raise ValueError("not enough nonzero training columns")
    normalized = Y[:, valid] / norms[valid][None, :]

    max_norm = float(np.max(norms[valid]))
    first_candidates = [
        pos for pos, source in enumerate(valid)
        if abs(float(norms[source]) - max_norm) <= 1e-12
    ]
    first_pos = min(first_candidates, key=lambda pos: (_lex_key(normalized[:, pos]), int(valid[pos])))
    selected_positions = [int(first_pos)]

    while len(selected_positions) < n_atoms:
        selected = normalized[:, selected_positions]
        scores = 1.0 - np.max(np.abs(selected.T @ normalized), axis=0)
        scores[selected_positions] = -np.inf
        best_score = float(np.max(scores))
        candidates = [
            pos for pos in range(normalized.shape[1])
            if np.isfinite(scores[pos]) and abs(float(scores[pos]) - best_score) <= 1e-12
        ]
        best_pos = min(candidates, key=lambda pos: (_lex_key(normalized[:, pos]), int(valid[pos])))
        selected_positions.append(int(best_pos))

    selected_indices = [int(valid[pos]) for pos in selected_positions]
    dictionary = normalized[:, selected_positions].copy()
    return dictionary, {
        "name": "deterministic_maximin",
        "selected_training_indices": selected_indices,
        "selected_unique_column_count": int(np.unique(Y[:, selected_indices].T, axis=0).shape[0]),
    }


def random_column_initialization(
    Y: np.ndarray,
    n_atoms: int,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2 or Y.shape[1] < n_atoms:
        raise ValueError("Y must contain at least n_atoms columns")
    rng = np.random.default_rng(seed)
    selected = rng.choice(Y.shape[1], size=n_atoms, replace=False).astype(np.int64)
    dictionary = normalize_columns(Y[:, selected])
    return dictionary, {
        "name": "fixed_random_training_columns",
        "seed": int(seed),
        "selected_training_indices": selected.tolist(),
        "selected_unique_column_count": int(np.unique(Y[:, selected].T, axis=0).shape[0]),
    }


def _binary_prf(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    truth = np.asarray(truth, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    tp = int(np.sum(truth & prediction))
    fp = int(np.sum(~truth & prediction))
    fn = int(np.sum(truth & ~prediction))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, EPS)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1), "tp": tp, "fp": fp, "fn": fn}


def _nonisolated_connected(edge_support: set[int], n_nodes: int = 6) -> bool:
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.int8)
    edges = upper_triangle_edges(n_nodes)
    for index in edge_support:
        left, right = edges[index]
        adjacency[left, right] = 1
        adjacency[right, left] = 1
    active_nodes = np.flatnonzero(np.sum(adjacency, axis=0) > 0)
    if active_nodes.size == 0:
        return False
    return is_connected(adjacency[np.ix_(active_nodes, active_nodes)])


def decode_atom_metrics(
    D: np.ndarray,
    alignment: dict[str, Any],
    true_supports: tuple[tuple[int, ...], ...],
    *,
    thresholds: tuple[float, ...] = (0.3, 0.5, 0.7),
) -> dict[str, Any]:
    normalized = normalize_columns(D)
    permutation = np.asarray(alignment["true_to_learned"], dtype=np.int64)
    signs = np.asarray(alignment["signs"], dtype=np.float64)
    threshold_results: dict[str, Any] = {}
    for alpha in thresholds:
        per_atom: list[dict[str, Any]] = []
        for true_atom, learned_atom in enumerate(permutation):
            atom = signs[true_atom] * normalized[:, learned_atom]
            maximum = float(np.max(np.abs(atom)))
            predicted = set(int(index) for index in np.flatnonzero(np.abs(atom) >= alpha * maximum - 1e-12))
            truth = set(int(index) for index in true_supports[true_atom])
            binary = _binary_prf(
                np.asarray([index in truth for index in range(atom.shape[0])]),
                np.asarray([index in predicted for index in range(atom.shape[0])]),
            )
            per_atom.append({
                "motif_index": int(true_atom),
                "predicted_edge_indices": sorted(predicted),
                "predicted_edge_count": len(predicted),
                "true_edge_count": len(truth),
                "edge_precision": binary["precision"],
                "edge_recall": binary["recall"],
                "edge_f1": binary["f1"],
                "exact_support": predicted == truth,
                "nonisolated_connected": _nonisolated_connected(predicted),
            })
        threshold_results[f"{alpha:.1f}"] = {
            "relative_threshold": float(alpha),
            "mean_edge_f1": float(np.mean([item["edge_f1"] for item in per_atom])),
            "minimum_edge_f1": float(np.min([item["edge_f1"] for item in per_atom])),
            "exact_motif_count": int(sum(item["exact_support"] for item in per_atom)),
            "connected_decode_count": int(sum(item["nonisolated_connected"] for item in per_atom)),
            "per_atom": per_atom,
        }
    return {
        "decode_thresholds": threshold_results,
        "primary_decode": threshold_results["0.5"],
    }


def occurrence_metrics(
    labels: np.ndarray,
    X_aligned: np.ndarray,
    motif_names: tuple[str, ...] = MOTIF_NAMES,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    truth = np.zeros_like(X_aligned, dtype=bool)
    truth[labels, np.arange(labels.shape[0])] = True
    sample_max = np.max(np.abs(X_aligned), axis=0, keepdims=True)
    prediction = (np.abs(X_aligned) > 1e-8) & (np.abs(X_aligned) >= 1e-6 * np.maximum(sample_max, EPS))
    global_support = support_metrics(truth.astype(np.float64), prediction.astype(np.float64))
    per_motif: list[dict[str, Any]] = []
    for motif_index, motif_name in enumerate(motif_names):
        binary = _binary_prf(truth[motif_index], prediction[motif_index])
        per_motif.append({"motif_index": motif_index, "motif_name": motif_name, **binary})
    predicted_labels = np.argmax(np.abs(X_aligned), axis=0)
    nonzero = np.max(np.abs(X_aligned), axis=0) > 1e-8
    exact = float(np.mean(nonzero & (predicted_labels == labels)))
    return {
        **global_support,
        "occurrence_macro_precision": float(np.mean([item["precision"] for item in per_motif])),
        "occurrence_macro_recall": float(np.mean([item["recall"] for item in per_motif])),
        "occurrence_macro_f1": float(np.mean([item["f1"] for item in per_motif])),
        "exact_occurrence_accuracy": exact,
        "per_motif_occurrence": per_motif,
    }


def evaluate_g0_dictionary(
    dataset: HiddenMotifDataset,
    D: np.ndarray,
    *,
    X_train: np.ndarray | None = None,
    train_reconstruction_relative: float | None = None,
) -> dict[str, Any]:
    if X_train is None:
        X_train = sparse_code_matrix(D, dataset.Y_train, T=1)
    X_test = sparse_code_matrix(D, dataset.Y_test, T=1)
    train_reconstruction = D @ X_train
    test_reconstruction = D @ X_test
    train_rel = float(
        np.linalg.norm(dataset.Y_train - train_reconstruction, "fro")
        / max(np.linalg.norm(dataset.Y_train, "fro"), EPS)
    )
    test_rel = float(
        np.linalg.norm(dataset.Y_test - test_reconstruction, "fro")
        / max(np.linalg.norm(dataset.Y_test, "fro"), EPS)
    )
    alignment = optimal_atom_alignment(dataset.D_true, D)
    aligned_train = align_codes_to_truth(X_train, alignment)
    aligned_test = align_codes_to_truth(X_test, alignment)
    return {
        **alignment,
        "train_reconstruction_relative": float(train_reconstruction_relative if train_reconstruction_relative is not None else train_rel),
        "train_reconstruction_recomputed": train_rel,
        "test_reconstruction_relative": test_rel,
        **decode_atom_metrics(D, alignment, dataset.true_edge_supports),
        "train_occurrence": occurrence_metrics(dataset.train_labels, aligned_train, dataset.motif_names),
        "test_occurrence": occurrence_metrics(dataset.test_labels, aligned_test, dataset.motif_names),
        "test_binary_reconstruction": binary_reconstruction_metrics(dataset.Y_test, test_reconstruction),
    }


def run_g0_stage(
    dataset: HiddenMotifDataset,
    initial_dictionary: np.ndarray,
    *,
    n_iter: int,
    ksvd_seed: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    D, X_train, info = ksvd(
        dataset.Y_train,
        n_atoms=initial_dictionary.shape[1],
        T=1,
        T_min=1,
        n_iter=n_iter,
        seed=ksvd_seed,
        initial_dictionary=initial_dictionary,
    )
    metrics = evaluate_g0_dictionary(
        dataset,
        D,
        X_train=X_train,
        train_reconstruction_relative=float(info["recon_rel"]),
    )
    metrics["n_iter"] = int(n_iter)
    metrics["train_atoms_used"] = int(info["atoms_used"])
    metrics["train_mean_nnz"] = float(info["mean_nnz"])
    metrics["reconstruction_curve"] = [float(value) for value in info["recon_curve"]]
    return D, metrics


def oracle_control(dataset: HiddenMotifDataset) -> dict[str, Any]:
    metrics = evaluate_g0_dictionary(dataset, dataset.D_true)
    passed = bool(
        metrics["test_reconstruction_relative"] <= 1e-10
        and metrics["minimum_atom_cosine"] >= 1.0 - 1e-10
        and metrics["primary_decode"]["exact_motif_count"] == len(MOTIF_NAMES)
        and metrics["test_occurrence"]["occurrence_macro_f1"] >= 1.0 - 1e-10
        and metrics["test_occurrence"]["exact_occurrence_accuracy"] >= 1.0 - 1e-10
        and metrics["test_binary_reconstruction"]["exact_patch_recovery"] >= 1.0 - 1e-10
    )
    return {"passed": passed, **metrics}
