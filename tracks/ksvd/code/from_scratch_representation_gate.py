"""G0B-R: representation-only gates before noisy motif dictionary learning."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, permutations
from typing import Any

import numpy as np

from .from_scratch_hidden_motif import (
    MOTIF_EDGE_LISTS,
    MOTIF_NAMES,
    _canonical_permutation_index_map,
    adjacency_to_upper_vector,
    motif_template,
)
from .from_scratch_recovery import EPS, upper_triangle_edges


@dataclass(frozen=True)
class CanonicalRoleResult:
    vector: np.ndarray
    deterministic_core_support: tuple[int, ...]
    possible_core_supports: tuple[tuple[int, ...], ...]
    minimizing_permutation_count: int


@dataclass(frozen=True)
class MotifVariant:
    condition: str
    motif_index: int
    raw_adjacency: np.ndarray
    hidden_core_edges: tuple[tuple[int, int], ...]
    canonical_vector: np.ndarray
    deterministic_core_support: tuple[int, ...]
    possible_core_supports: tuple[tuple[int, ...], ...]
    minimizing_permutation_count: int


def canonicalize_with_hidden_core(
    adjacency: np.ndarray,
    hidden_core_edges: tuple[tuple[int, int], ...],
) -> CanonicalRoleResult:
    """Canonicalize an untyped graph and map hidden edge roles through all minima.

    The canonical vector is observable. Core supports are evaluator-only records.
    Multiple possible supports mean graph automorphisms make the hidden core role
    ambiguous in canonical coordinates.
    """
    raw = adjacency_to_upper_vector(adjacency).astype(np.int8)
    if raw.shape != (15,) or not np.all((raw == 0) | (raw == 1)):
        raise ValueError("expected a binary six-node graph")
    index_map = _canonical_permutation_index_map(6)
    candidates = raw[index_map]
    weights = (1 << np.arange(14, -1, -1)).astype(np.int64)
    codes = candidates.astype(np.int64) @ weights
    minimum = int(np.min(codes))
    minimizing_rows = np.flatnonzero(codes == minimum)

    edge_to_index = {edge: index for index, edge in enumerate(upper_triangle_edges(6))}
    core_raw_indices = {
        edge_to_index[tuple(sorted((int(left), int(right))))]
        for left, right in hidden_core_edges
    }
    supports = {
        tuple(
            int(position)
            for position, raw_index in enumerate(index_map[row])
            if int(raw_index) in core_raw_indices
        )
        for row in minimizing_rows
    }
    ordered_supports = tuple(sorted(supports))
    return CanonicalRoleResult(
        vector=candidates[int(minimizing_rows[0])].astype(np.float64),
        deterministic_core_support=tuple(
            int(position)
            for position, raw_index in enumerate(index_map[int(minimizing_rows[0])])
            if int(raw_index) in core_raw_indices
        ),
        possible_core_supports=ordered_supports,
        minimizing_permutation_count=int(minimizing_rows.size),
    )


def _record(condition: str, motif_index: int, adjacency: np.ndarray) -> MotifVariant:
    hidden_core_edges = tuple(
        tuple(int(value) for value in edge) for edge in MOTIF_EDGE_LISTS[motif_index]
    )
    result = canonicalize_with_hidden_core(adjacency, hidden_core_edges)
    return MotifVariant(
        condition=condition,
        motif_index=motif_index,
        raw_adjacency=np.asarray(adjacency, dtype=np.int8).copy(),
        hidden_core_edges=hidden_core_edges,
        canonical_vector=result.vector,
        deterministic_core_support=result.deterministic_core_support,
        possible_core_supports=result.possible_core_supports,
        minimizing_permutation_count=result.minimizing_permutation_count,
    )


def make_clean_variants() -> list[MotifVariant]:
    return [_record("R0_clean", motif_index, motif_template(motif_index)) for motif_index in range(len(MOTIF_NAMES))]


def make_add_one_noncore_variants() -> list[MotifVariant]:
    variants: list[MotifVariant] = []
    edges = upper_triangle_edges(6)
    for motif_index in range(len(MOTIF_NAMES)):
        base = motif_template(motif_index)
        for left, right in edges:
            if base[left, right] != 0:
                continue
            adjacency = base.copy()
            adjacency[left, right] = 1
            adjacency[right, left] = 1
            variants.append(_record("R1_add_one_noncore_exhaustive", motif_index, adjacency))
    return variants


def make_edge_flip_variants(
    *,
    seed: int = 20260731,
    flip_probability: float = 0.05,
    samples_per_motif: int = 2000,
) -> list[MotifVariant]:
    if not 0.0 <= flip_probability <= 1.0:
        raise ValueError("flip_probability must be in [0, 1]")
    rng = np.random.default_rng(seed)
    edges = upper_triangle_edges(6)
    variants: list[MotifVariant] = []
    cache: dict[tuple[int, bytes], MotifVariant] = {}
    for motif_index in range(len(MOTIF_NAMES)):
        base = motif_template(motif_index)
        for _ in range(samples_per_motif):
            flips = rng.random(len(edges)) < flip_probability
            adjacency = base.copy()
            for flip, (left, right) in zip(flips, edges):
                if flip:
                    adjacency[left, right] = 1 - adjacency[left, right]
                    adjacency[right, left] = adjacency[left, right]
            key = (motif_index, adjacency_to_upper_vector(adjacency).astype(np.int8).tobytes())
            cached = cache.get(key)
            if cached is None:
                cached = _record("R2_edge_flip_p005", motif_index, adjacency)
                cache[key] = cached
            variants.append(cached)
    return variants


def _group_by_label(variants: list[MotifVariant]) -> list[list[MotifVariant]]:
    groups = [[variant for variant in variants if variant.motif_index == label] for label in range(len(MOTIF_NAMES))]
    if any(not group for group in groups):
        raise ValueError("each motif family must have at least one variant")
    return groups


def label_identifiability_metrics(variants: list[MotifVariant]) -> dict[str, Any]:
    groups = _group_by_label(variants)
    masses: dict[bytes, np.ndarray] = {}
    vectors: dict[bytes, list[int]] = {}
    for label, group in enumerate(groups):
        unit_mass = 1.0 / (len(groups) * len(group))
        for variant in group:
            key = variant.canonical_vector.astype(np.int8).tobytes()
            masses.setdefault(key, np.zeros(len(groups), dtype=np.float64))[label] += unit_mass
            vectors.setdefault(key, variant.canonical_vector.astype(int).tolist())
    bayes_accuracy = float(sum(float(np.max(mass)) for mass in masses.values()))
    collision_keys = [key for key, mass in masses.items() if np.count_nonzero(mass > EPS) > 1]
    collision_mass = float(sum(float(np.sum(masses[key])) for key in collision_keys))
    collisions = []
    for key in collision_keys:
        mass = masses[key]
        collisions.append({
            "canonical_vector": vectors[key],
            "motif_names": [MOTIF_NAMES[index] for index in np.flatnonzero(mass > EPS)],
            "equal_prior_probability_mass": float(np.sum(mass)),
            "per_motif_mass": mass.tolist(),
        })
    collisions.sort(key=lambda item: -item["equal_prior_probability_mass"])
    return {
        "unique_canonical_vector_count": len(masses),
        "cross_family_collision_vector_count": len(collision_keys),
        "cross_family_collision_mass": collision_mass,
        "equal_prior_bayes_accuracy": bayes_accuracy,
        "collisions": collisions,
    }


def _support_f1(left_mask: int, right_mask: int, support_size: int) -> float:
    return (left_mask & right_mask).bit_count() / max(support_size, 1)


def _support_to_mask(support: tuple[int, ...]) -> int:
    mask = 0
    for index in support:
        mask |= 1 << int(index)
    return mask


def core_coordinate_metrics(variants: list[MotifVariant]) -> dict[str, Any]:
    groups = _group_by_label(variants)
    family_results = []
    for motif_index, group in enumerate(groups):
        support_size = len(group[0].deterministic_core_support)
        candidate_masks = [
            sum(1 << index for index in combination)
            for combination in combinations(range(15), support_size)
        ]
        deterministic_masks = [_support_to_mask(variant.deterministic_core_support) for variant in group]
        option_masks = [tuple(_support_to_mask(support) for support in variant.possible_core_supports) for variant in group]

        def best_candidate(mode: str) -> tuple[int, float, float]:
            best_mask = candidate_masks[0]
            best_mean = -1.0
            best_minimum = -1.0
            for candidate in candidate_masks:
                if mode == "deterministic":
                    values = [_support_f1(candidate, target, support_size) for target in deterministic_masks]
                elif mode == "optimistic":
                    values = [max(_support_f1(candidate, target, support_size) for target in targets) for targets in option_masks]
                elif mode == "robust":
                    values = [min(_support_f1(candidate, target, support_size) for target in targets) for targets in option_masks]
                else:
                    raise ValueError(mode)
                mean = float(np.mean(values))
                minimum = float(np.min(values))
                if mean > best_mean + 1e-12 or (abs(mean - best_mean) <= 1e-12 and candidate < best_mask):
                    best_mask, best_mean, best_minimum = candidate, mean, minimum
            return best_mask, best_mean, best_minimum

        occupancy = np.mean(
            np.asarray([[index in variant.deterministic_core_support for index in range(15)] for variant in group], dtype=np.float64),
            axis=0,
        )
        deterministic_best = best_candidate("deterministic")
        optimistic_best = best_candidate("optimistic")
        robust_best = best_candidate("robust")
        family_results.append({
            "motif_index": motif_index,
            "motif_name": MOTIF_NAMES[motif_index],
            "variant_count": len(group),
            "core_edge_count": support_size,
            "role_identifiable_variant_rate": float(np.mean([len(variant.possible_core_supports) == 1 for variant in group])),
            "mean_distinct_core_supports_per_variant": float(np.mean([len(variant.possible_core_supports) for variant in group])),
            "mean_minimizing_permutation_count": float(np.mean([variant.minimizing_permutation_count for variant in group])),
            "deterministic_core_coordinate_occupancy": occupancy.tolist(),
            "best_deterministic_fixed_support": [index for index in range(15) if deterministic_best[0] & (1 << index)],
            "best_deterministic_fixed_support_mean_f1": deterministic_best[1],
            "best_deterministic_fixed_support_minimum_f1": deterministic_best[2],
            "best_optimistic_fixed_support_mean_f1": optimistic_best[1],
            "best_optimistic_fixed_support_minimum_f1": optimistic_best[2],
            "best_robust_fixed_support_mean_f1": robust_best[1],
            "best_robust_fixed_support_minimum_f1": robust_best[2],
        })
    return {
        "families": family_results,
        "minimum_role_identifiable_variant_rate": float(min(item["role_identifiable_variant_rate"] for item in family_results)),
        "minimum_best_deterministic_fixed_support_mean_f1": float(min(item["best_deterministic_fixed_support_mean_f1"] for item in family_results)),
        "minimum_best_optimistic_fixed_support_mean_f1": float(min(item["best_optimistic_fixed_support_mean_f1"] for item in family_results)),
        "minimum_best_robust_fixed_support_mean_f1": float(min(item["best_robust_fixed_support_mean_f1"] for item in family_results)),
    }


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, np.maximum(norms, EPS), out=np.zeros_like(matrix), where=norms > EPS)


def prototype_separability_metrics(variants: list[MotifVariant]) -> dict[str, Any]:
    groups = _group_by_label(variants)
    arrays = [np.vstack([variant.canonical_vector for variant in group]).astype(np.float64) for group in groups]
    normalized_arrays = [_normalize_rows(array) for array in arrays]
    centroids = np.vstack([np.mean(array, axis=0) for array in arrays])
    normalized_centroids = _normalize_rows(centroids)
    medoids = []
    for array, normalized, centroid in zip(arrays, normalized_arrays, normalized_centroids):
        similarities = normalized @ centroid
        best_value = float(np.max(similarities))
        candidates = np.flatnonzero(np.abs(similarities - best_value) <= 1e-12)
        best = min((int(index) for index in candidates), key=lambda index: tuple(array[index].tolist()))
        medoids.append(array[best])
    normalized_medoids = _normalize_rows(np.vstack(medoids))

    centroid_accuracies = []
    medoid_accuracies = []
    within_cosines = []
    for label, normalized in enumerate(normalized_arrays):
        centroid_predictions = np.argmax(normalized @ normalized_centroids.T, axis=1)
        medoid_predictions = np.argmax(normalized @ normalized_medoids.T, axis=1)
        centroid_accuracies.append(float(np.mean(centroid_predictions == label)))
        medoid_accuracies.append(float(np.mean(medoid_predictions == label)))
        within_cosines.append(float(np.mean(normalized @ normalized_centroids[label])))
    centroid_gram = normalized_centroids @ normalized_centroids.T
    between = centroid_gram[~np.eye(len(groups), dtype=bool)]
    return {
        "nearest_centroid_accuracy_per_motif": centroid_accuracies,
        "nearest_centroid_macro_accuracy": float(np.mean(centroid_accuracies)),
        "nearest_medoid_accuracy_per_motif": medoid_accuracies,
        "nearest_medoid_macro_accuracy": float(np.mean(medoid_accuracies)),
        "within_family_mean_cosine_to_centroid": within_cosines,
        "minimum_within_family_mean_cosine_to_centroid": float(min(within_cosines)),
        "between_family_centroid_cosine_mean": float(np.mean(between)),
        "between_family_centroid_cosine_maximum": float(np.max(between)),
        "centroids": centroids.tolist(),
        "medoids": np.vstack(medoids).tolist(),
    }


def evaluate_representation_condition(variants: list[MotifVariant]) -> dict[str, Any]:
    condition_names = sorted(set(variant.condition for variant in variants))
    if len(condition_names) != 1:
        raise ValueError("condition evaluation expects one condition")
    label = label_identifiability_metrics(variants)
    core = core_coordinate_metrics(variants)
    prototype = prototype_separability_metrics(variants)
    checks = {
        "collision_mass_le_0.01": label["cross_family_collision_mass"] <= 0.01,
        "bayes_accuracy_ge_0.95": label["equal_prior_bayes_accuracy"] >= 0.95,
        "fixed_core_support_mean_f1_ge_0.90": core["minimum_best_robust_fixed_support_mean_f1"] >= 0.90,
        "nearest_medoid_macro_accuracy_ge_0.90": prototype["nearest_medoid_macro_accuracy"] >= 0.90,
    }
    return {
        "condition": condition_names[0],
        "variant_count": len(variants),
        "motif_variant_counts": [sum(variant.motif_index == index for variant in variants) for index in range(len(MOTIF_NAMES))],
        "label_identifiability": label,
        "core_coordinate_consistency": core,
        "prototype_separability": prototype,
        "gate_checks": checks,
        "gate_passed": bool(all(checks.values())),
    }


@lru_cache(maxsize=None)
def _permutation_array(n_nodes: int) -> np.ndarray:
    return np.asarray(list(permutations(range(n_nodes))), dtype=np.int16)


def canonicalize_rooted_with_hidden_core(
    adjacency: np.ndarray,
    hidden_core_edges: tuple[tuple[int, int], ...],
    root: int,
) -> CanonicalRoleResult:
    """Canonicalize while fixing an observed root at canonical node zero."""
    raw = adjacency_to_upper_vector(adjacency).astype(np.int8)
    if raw.shape != (15,) or not 0 <= root < 6:
        raise ValueError("expected a six-node graph and valid root")
    permutations_array = _permutation_array(6)
    rows = np.flatnonzero(permutations_array[:, 0] == int(root))
    index_map = _canonical_permutation_index_map(6)[rows]
    candidates = raw[index_map]
    weights = (1 << np.arange(14, -1, -1)).astype(np.int64)
    codes = candidates.astype(np.int64) @ weights
    minimum = int(np.min(codes))
    local_minima = np.flatnonzero(codes == minimum)

    edge_to_index = {edge: index for index, edge in enumerate(upper_triangle_edges(6))}
    core_raw_indices = {
        edge_to_index[tuple(sorted((int(left), int(right))))]
        for left, right in hidden_core_edges
    }
    supports = {
        tuple(
            int(position)
            for position, raw_index in enumerate(index_map[local_row])
            if int(raw_index) in core_raw_indices
        )
        for local_row in local_minima
    }
    ordered_supports = tuple(sorted(supports))
    first = int(local_minima[0])
    return CanonicalRoleResult(
        vector=candidates[first].astype(np.float64),
        deterministic_core_support=tuple(
            int(position)
            for position, raw_index in enumerate(index_map[first])
            if int(raw_index) in core_raw_indices
        ),
        possible_core_supports=ordered_supports,
        minimizing_permutation_count=int(local_minima.size),
    )


def transform_variants_oracle_root(
    variants: list[MotifVariant],
    *,
    condition_suffix: str = "oracle_root0",
) -> list[MotifVariant]:
    """Expose generator node zero as a root, analogous to a known patch anchor."""
    transformed = []
    for variant in variants:
        result = canonicalize_rooted_with_hidden_core(
            variant.raw_adjacency,
            variant.hidden_core_edges,
            root=0,
        )
        transformed.append(MotifVariant(
            condition=f"{variant.condition}__{condition_suffix}",
            motif_index=variant.motif_index,
            raw_adjacency=variant.raw_adjacency.copy(),
            hidden_core_edges=variant.hidden_core_edges,
            canonical_vector=result.vector,
            deterministic_core_support=result.deterministic_core_support,
            possible_core_supports=result.possible_core_supports,
            minimizing_permutation_count=result.minimizing_permutation_count,
        ))
    return transformed


def transform_variants_structural_max_degree_root(
    variants: list[MotifVariant],
) -> list[MotifVariant]:
    """Use only graph-observable max-degree roots; ties remain set-valued.

    This adds no side information, so it cannot resolve two latent families that
    produce the same unrooted observed graph. It can only improve coordinates.
    """
    transformed = []
    weights = (1 << np.arange(14, -1, -1)).astype(np.int64)
    for variant in variants:
        degrees = np.sum(variant.raw_adjacency, axis=1)
        roots = np.flatnonzero(degrees == np.max(degrees))
        rooted_results = [
            canonicalize_rooted_with_hidden_core(
                variant.raw_adjacency,
                variant.hidden_core_edges,
                root=int(root),
            )
            for root in roots
        ]
        codes = [int(result.vector.astype(np.int64) @ weights) for result in rooted_results]
        minimum = min(codes)
        selected = [result for result, code in zip(rooted_results, codes) if code == minimum]
        supports = tuple(sorted({support for result in selected for support in result.possible_core_supports}))
        first = selected[0]
        transformed.append(MotifVariant(
            condition=f"{variant.condition}__structural_max_degree_root",
            motif_index=variant.motif_index,
            raw_adjacency=variant.raw_adjacency.copy(),
            hidden_core_edges=variant.hidden_core_edges,
            canonical_vector=first.vector,
            deterministic_core_support=first.deterministic_core_support,
            possible_core_supports=supports,
            minimizing_permutation_count=int(sum(result.minimizing_permutation_count for result in selected)),
        ))
    return transformed
