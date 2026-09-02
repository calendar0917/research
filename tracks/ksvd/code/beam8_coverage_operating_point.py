"""Prefix metrics and checkpoint selection for Beam8 coverage operating points."""
from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Sequence

import numpy as np

from .beam8_edge_residual import (
    direct_enumerative_bits,
    identity_proxy_bits,
    residual_subset_bits,
    true_edge_set,
)
from .overlap_cover import PatchCover


CHECKPOINTS = ("BASE", "EDGE90", "EDGE95", "FAIR95", "EDGE99", "EDGE100")


def _pairs(nodes: Sequence[int]) -> set[tuple[int, int]]:
    return {tuple(sorted(pair)) for pair in combinations(nodes, 2)}


def _bridge_edges(adjacency: np.ndarray) -> set[tuple[int, int]]:
    """Return graph bridges using a deterministic Tarjan traversal."""
    values = np.asarray(adjacency, dtype=np.int8)
    n_nodes = values.shape[0]
    discovery = [-1] * n_nodes
    low = [-1] * n_nodes
    parent = [-1] * n_nodes
    timer = 0
    bridges: set[tuple[int, int]] = set()

    def visit(node: int) -> None:
        nonlocal timer
        discovery[node] = low[node] = timer
        timer += 1
        for raw_neighbor in np.flatnonzero(values[node]):
            neighbor = int(raw_neighbor)
            if discovery[neighbor] < 0:
                parent[neighbor] = node
                visit(neighbor)
                low[node] = min(low[node], low[neighbor])
                if low[neighbor] > discovery[node]:
                    bridges.add(tuple(sorted((node, neighbor))))
            elif neighbor != parent[node]:
                low[node] = min(low[node], discovery[neighbor])

    for node in range(n_nodes):
        if discovery[node] < 0:
            visit(node)
    return bridges


def edge_categories(
    adjacency: np.ndarray,
    *,
    block_groups: Sequence[int] | None = None,
) -> dict[str, set[tuple[int, int]]]:
    """Construct invariant residual-edge diagnostic categories."""
    values = np.asarray(adjacency, dtype=np.int8)
    true_edges = true_edge_set(values)
    degrees = np.sum(values, axis=1).astype(np.int64)
    q25 = float(np.quantile(degrees, 0.25))
    q75 = float(np.quantile(degrees, 0.75))
    categories = {
        "low_degree_incident": {
            edge
            for edge in true_edges
            if min(int(degrees[edge[0]]), int(degrees[edge[1]])) <= q25
        },
        "high_degree_incident": {
            edge
            for edge in true_edges
            if max(int(degrees[edge[0]]), int(degrees[edge[1]])) >= q75
        },
        "zero_common_neighbor": {
            edge
            for edge in true_edges
            if int(np.dot(values[edge[0]], values[edge[1]])) == 0
        },
        "bridge": _bridge_edges(values),
    }
    if block_groups is not None:
        groups = tuple(int(group) for group in block_groups)
        if len(groups) != values.shape[0]:
            raise ValueError("block_groups must align with adjacency")
        categories["cross_block"] = {
            edge for edge in true_edges if groups[edge[0]] != groups[edge[1]]
        }
    return categories


def _category_metrics(
    categories: dict[str, set[tuple[int, int]]],
    covered_edges: set[tuple[int, int]],
    residual_edges: set[tuple[int, int]],
) -> dict[str, float | int | None]:
    metrics: dict[str, float | int | None] = {}
    for name, edges in categories.items():
        covered = len(edges & covered_edges)
        residual = len(edges & residual_edges)
        metrics[f"{name}_edge_count"] = len(edges)
        metrics[f"{name}_edge_recall"] = (
            covered / len(edges) if edges else None
        )
        metrics[f"{name}_residual_share"] = (
            residual / len(residual_edges) if residual_edges else 0.0
        )
    return metrics


def prefix_coverage_trajectory(
    adjacency: np.ndarray,
    cover: PatchCover,
    *,
    patch_size: int,
    overlap: int,
    maximum_patches: int,
    block_groups: Sequence[int] | None = None,
    n_atoms: int = 24,
    sparsity: int = 3,
    coefficient_bits: int = 8,
) -> list[dict[str, Any]]:
    """Measure coverage, fairness, residual structure, and rate at every prefix."""
    values = np.asarray(adjacency, dtype=np.int8)
    n_nodes = values.shape[0]
    pair_count = n_nodes * (n_nodes - 1) // 2
    true_edges = true_edge_set(values)
    degrees = np.sum(values, axis=1).astype(np.int64)
    nonisolated = np.flatnonzero(degrees > 0)
    categories = edge_categories(values, block_groups=block_groups)
    observed_pairs: set[tuple[int, int]] = set()
    covered_edges: set[tuple[int, int]] = set()
    covered_nodes: set[int] = set()
    edge_multiplicity: dict[tuple[int, int], int] = {}
    marginal_new_edges: list[int] = []
    atom_index_bits = int(math.ceil(math.log2(n_atoms)))
    code_bits_per_patch = sparsity * (atom_index_bits + coefficient_bits)
    framing_bits = int(math.ceil(math.log2(maximum_patches + 1)))
    direct_bits = direct_enumerative_bits(pair_count, len(true_edges))
    rows: list[dict[str, Any]] = []

    for patch_count in range(len(cover.patches) + 1):
        if patch_count:
            patch = cover.patches[patch_count - 1]
            patch_pairs = _pairs(patch.node_ids)
            patch_edges = true_edges & patch_pairs
            marginal_new_edges.append(len(patch_edges - covered_edges))
            observed_pairs.update(patch_pairs)
            covered_edges.update(patch_edges)
            covered_nodes.update(int(node) for node in patch.node_ids)
            for edge in patch_edges:
                edge_multiplicity[edge] = edge_multiplicity.get(edge, 0) + 1

        residual_edges = true_edges - covered_edges
        covered_incident = np.zeros(n_nodes, dtype=np.int64)
        residual_incident = np.zeros(n_nodes, dtype=np.int64)
        for left, right in covered_edges:
            covered_incident[left] += 1
            covered_incident[right] += 1
        for left, right in residual_edges:
            residual_incident[left] += 1
            residual_incident[right] += 1
        recalls = (
            covered_incident[nonisolated] / degrees[nonisolated]
            if nonisolated.size
            else np.ones(0, dtype=np.float64)
        )
        multiplicities = np.asarray(tuple(edge_multiplicity.values()), dtype=np.float64)
        identity = identity_proxy_bits(
            n_nodes=n_nodes,
            patch_size=patch_size,
            overlap=overlap,
            patch_count=patch_count,
        )
        unobserved_count = pair_count - len(observed_pairs)
        residual_bits = residual_subset_bits(unobserved_count, len(residual_edges))
        ksvd_bits = patch_count * code_bits_per_patch
        canonical_bits = (
            framing_bits
            + identity["canonical_set_identity_bits"]
            + ksvd_bits
            + residual_bits
        )
        row = {
            "patch_count": patch_count,
            "raw_pair_slots": patch_count * patch_size * (patch_size - 1) // 2,
            "unique_observed_pair_count": len(observed_pairs),
            "covered_edge_count": len(covered_edges),
            "residual_edge_count": len(residual_edges),
            "node_coverage": len(covered_nodes) / max(n_nodes, 1),
            "pair_coverage": len(observed_pairs) / max(pair_count, 1),
            "edge_coverage": len(covered_edges) / max(len(true_edges), 1),
            "raw_zero_fill_rmse": math.sqrt(len(residual_edges) / max(pair_count, 1)),
            "covered_edge_density": len(covered_edges) / max(len(observed_pairs), 1),
            "last_marginal_new_edges": marginal_new_edges[-1] if marginal_new_edges else 0,
            "mean_marginal_new_edges": float(np.mean(marginal_new_edges))
            if marginal_new_edges
            else 0.0,
            "node_incident_recall_mean": float(np.mean(recalls)) if recalls.size else 1.0,
            "node_incident_recall_p10": float(
                np.quantile(recalls, 0.10, method="linear")
            )
            if recalls.size
            else 1.0,
            "node_incident_recall_minimum": float(np.min(recalls))
            if recalls.size
            else 1.0,
            "fully_covered_node_fraction": float(np.mean(recalls == 1.0))
            if recalls.size
            else 1.0,
            "nodes_with_residual_fraction": float(np.mean(residual_incident > 0)),
            "maximum_residual_incident_count": int(np.max(residual_incident)),
            "edge_multiplicity_mean": float(np.mean(multiplicities))
            if multiplicities.size
            else 0.0,
            "edge_multiplicity_cv": float(
                np.std(multiplicities) / max(np.mean(multiplicities), 1e-12)
            )
            if multiplicities.size
            else 0.0,
            **_category_metrics(categories, covered_edges, residual_edges),
            **identity,
            "prefix_length_bits": framing_bits,
            "ksvd_code_proxy_bits": ksvd_bits,
            "residual_subset_bits": residual_bits,
            "canonical_hybrid_proxy_bits": canonical_bits,
            "direct_bitset_bits": pair_count,
            "direct_enumerative_exact_bits": direct_bits,
            "dictionary_scalars": patch_size * (patch_size - 1) // 2 * n_atoms,
            "code_scalars": patch_count * sparsity,
        }
        rows.append(row)
    return rows


def select_operating_checkpoints(
    trajectory: Sequence[dict[str, Any]],
    *,
    base_patch_count: int,
) -> dict[str, dict[str, Any] | None]:
    by_count = {int(row["patch_count"]): dict(row) for row in trajectory}
    selected: dict[str, dict[str, Any] | None] = {
        "BASE": by_count.get(int(base_patch_count))
    }
    for threshold in (0.90, 0.95, 0.99, 1.0):
        label = f"EDGE{int(round(100 * threshold))}"
        selected[label] = next(
            (
                dict(row)
                for row in trajectory
                if float(row["edge_coverage"]) + 1e-12 >= threshold
            ),
            None,
        )
    selected["FAIR95"] = next(
        (
            dict(row)
            for row in trajectory
            if float(row["edge_coverage"]) + 1e-12 >= 0.95
            and float(row["node_incident_recall_p10"]) + 1e-12 >= 0.90
            and float(row["node_coverage"]) + 1e-12 >= 1.0
        ),
        None,
    )
    return selected


def nondominated_candidates(
    rows: Sequence[dict[str, Any]],
) -> list[str]:
    """Return candidate labels on the frozen mixed min/max coverage axes."""
    minimization = (
        "patch_count",
        "unique_observed_pair_count",
        "code_scalars",
        "dictionary_scalars",
    )
    maximization = (
        "edge_coverage",
        "node_incident_recall_p10",
        "low_degree_incident_edge_recall",
    )

    def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
        not_worse = all(float(left[key]) <= float(right[key]) for key in minimization)
        not_worse = not_worse and all(
            float(left[key]) >= float(right[key]) for key in maximization
        )
        strictly = any(float(left[key]) < float(right[key]) for key in minimization)
        strictly = strictly or any(
            float(left[key]) > float(right[key]) for key in maximization
        )
        return bool(not_worse and strictly)

    return [
        str(row["candidate"])
        for row in rows
        if not any(
            other is not row and dominates(other, row) for other in rows
        )
    ]
