"""Rate proxies and prefix selection for Beam8 edge-residual hybrids."""
from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Sequence

import numpy as np

from .overlap_cover import PatchCover


EPS = 1e-12


def log2_combination(n: int, k: int) -> float:
    """Return log2(C(n, k)) without constructing the integer combination."""
    if not 0 <= k <= n:
        raise ValueError("expected 0 <= k <= n")
    return float(
        (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))
        / math.log(2.0)
    )


def log2_permutation(n: int, k: int) -> float:
    """Return log2(P(n, k)) without constructing the integer permutation."""
    if not 0 <= k <= n:
        raise ValueError("expected 0 <= k <= n")
    return float((math.lgamma(n + 1) - math.lgamma(n - k + 1)) / math.log(2.0))


def true_edge_set(adjacency: np.ndarray) -> set[tuple[int, int]]:
    values = np.asarray(adjacency)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("adjacency must be square")
    return {
        (left, right)
        for left in range(values.shape[0])
        for right in range(left + 1, values.shape[0])
        if values[left, right] != 0
    }


def identity_proxy_bits(
    *,
    n_nodes: int,
    patch_size: int,
    overlap: int,
    patch_count: int,
) -> dict[str, int]:
    """Return ordered and canonical-set explicit membership proxies.

    Exact overlap means every successor chooses its new nodes from the ``n-s``
    nodes outside the previous patch.  The canonical-set branch assumes local
    slot order is derivable and therefore deliberately favors the patch route.
    """
    if not 0 <= patch_count:
        raise ValueError("patch_count must be nonnegative")
    if not 0 < overlap < patch_size <= n_nodes:
        raise ValueError("expected 0 < overlap < patch_size <= n_nodes")
    if patch_count == 0:
        return {"ordered_identity_bits": 0, "canonical_set_identity_bits": 0}
    new_count = patch_size - overlap
    outside_count = n_nodes - patch_size
    if new_count > outside_count:
        raise ValueError("not enough outside nodes for exact-overlap transition")
    retained = log2_combination(patch_size, overlap)
    ordered = log2_permutation(n_nodes, patch_size) + (patch_count - 1) * (
        retained + log2_permutation(outside_count, new_count)
    )
    canonical = (
        log2_combination(n_nodes, patch_size)
        + math.log2(patch_size)
        + (patch_count - 1)
        * (
            retained
            + log2_combination(outside_count, new_count)
            + math.log2(new_count)
        )
    )
    return {
        "ordered_identity_bits": int(math.ceil(ordered)),
        "canonical_set_identity_bits": int(math.ceil(canonical)),
    }


def residual_subset_bits(unobserved_pair_count: int, residual_edge_count: int) -> int:
    """Encode residual cardinality and its subset of the unobserved universe."""
    if not 0 <= residual_edge_count <= unobserved_pair_count:
        raise ValueError("residual edges must be a subset of unobserved pairs")
    cardinality_bits = math.ceil(math.log2(unobserved_pair_count + 1))
    subset_bits = math.ceil(
        log2_combination(unobserved_pair_count, residual_edge_count)
    )
    return int(cardinality_bits + subset_bits)


def direct_enumerative_bits(pair_count: int, edge_count: int) -> int:
    if not 0 <= edge_count <= pair_count:
        raise ValueError("edge_count must lie inside pair universe")
    return int(
        math.ceil(math.log2(pair_count + 1))
        + math.ceil(log2_combination(pair_count, edge_count))
    )


def prefix_rate_trajectory(
    adjacency: np.ndarray,
    cover: PatchCover,
    *,
    patch_size: int,
    overlap: int,
    n_atoms: int = 24,
    sparsity: int = 3,
    coefficient_bits: int = 8,
) -> list[dict[str, Any]]:
    """Audit all prefixes of one frozen Beam8 chain, including the zero prefix."""
    if len(cover.patches) < 1:
        raise ValueError("cover must contain at least one patch")
    if any(len(patch.node_ids) != patch_size for patch in cover.patches):
        raise ValueError("cover patch size does not match configuration")
    if n_atoms < 1 or sparsity < 1 or coefficient_bits < 1:
        raise ValueError("invalid KSVD proxy configuration")
    n_nodes = int(np.asarray(adjacency).shape[0])
    pair_count = n_nodes * (n_nodes - 1) // 2
    true_edges = true_edge_set(adjacency)
    observed_pairs: set[tuple[int, int]] = set()
    covered_edges: set[tuple[int, int]] = set()
    atom_index_bits = int(math.ceil(math.log2(n_atoms)))
    code_bits_per_patch = sparsity * (atom_index_bits + coefficient_bits)
    prefix_length_bits = int(math.ceil(math.log2(len(cover.patches) + 1)))
    direct_bits = direct_enumerative_bits(pair_count, len(true_edges))
    rows: list[dict[str, Any]] = []

    for patch_count in range(len(cover.patches) + 1):
        if patch_count > 0:
            patch = cover.patches[patch_count - 1]
            patch_pairs = {
                tuple(sorted((int(left), int(right))))
                for left, right in combinations(patch.node_ids, 2)
            }
            observed_pairs.update(patch_pairs)
            covered_edges.update(true_edges & patch_pairs)
        residual_count = len(true_edges - covered_edges)
        unobserved_count = pair_count - len(observed_pairs)
        if not residual_count <= unobserved_count:
            raise RuntimeError("covered-edge accounting is inconsistent")
        identity = identity_proxy_bits(
            n_nodes=n_nodes,
            patch_size=patch_size,
            overlap=overlap,
            patch_count=patch_count,
        )
        residual_bits = residual_subset_bits(unobserved_count, residual_count)
        code_bits = patch_count * code_bits_per_patch
        edge_coverage = len(covered_edges) / max(len(true_edges), 1)
        pair_coverage = len(observed_pairs) / max(pair_count, 1)
        rows.append(
            {
                "patch_count": patch_count,
                "observed_pair_count": len(observed_pairs),
                "covered_edge_count": len(covered_edges),
                "residual_edge_count": residual_count,
                "unobserved_pair_count": unobserved_count,
                "edge_coverage": float(edge_coverage),
                "pair_coverage": float(pair_coverage),
                "raw_zero_fill_rmse": float(math.sqrt(residual_count / pair_count)),
                **identity,
                "prefix_length_bits": prefix_length_bits,
                "ksvd_code_proxy_bits": int(code_bits),
                "residual_subset_bits": int(residual_bits),
                "ordered_hybrid_proxy_bits": int(
                    prefix_length_bits
                    + identity["ordered_identity_bits"]
                    + code_bits
                    + residual_bits
                ),
                "canonical_hybrid_proxy_bits": int(
                    prefix_length_bits
                    + identity["canonical_set_identity_bits"]
                    + code_bits
                    + residual_bits
                ),
                "ordered_raw_exact_bits": int(
                    prefix_length_bits
                    + identity["ordered_identity_bits"]
                    + len(observed_pairs)
                    + residual_bits
                ),
                "canonical_raw_exact_bits": int(
                    prefix_length_bits
                    + identity["canonical_set_identity_bits"]
                    + len(observed_pairs)
                    + residual_bits
                ),
                "direct_bitset_bits": pair_count,
                "direct_enumerative_exact_bits": direct_bits,
            }
        )
    if rows[0]["canonical_hybrid_proxy_bits"] != direct_bits + prefix_length_bits:
        raise RuntimeError(
            "zero-prefix hybrid must equal direct enumerative code plus framing"
        )
    return rows


def first_at_coverage(
    trajectory: Sequence[dict[str, Any]], threshold: float
) -> dict[str, Any] | None:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("coverage threshold must lie in [0, 1]")
    return next(
        (dict(row) for row in trajectory if float(row["edge_coverage"]) + EPS >= threshold),
        None,
    )


def minimum_cost_prefix(
    trajectory: Sequence[dict[str, Any]],
    *,
    minimum_patch_count: int = 0,
    cost_key: str = "canonical_hybrid_proxy_bits",
) -> dict[str, Any]:
    eligible = [
        row
        for row in trajectory
        if int(row["patch_count"]) >= int(minimum_patch_count)
    ]
    if not eligible:
        raise ValueError("no prefix satisfies minimum_patch_count")
    # Prefer fewer patches on an exact cost tie.
    selected = min(
        eligible,
        key=lambda row: (float(row[cost_key]), int(row["patch_count"])),
    )
    return dict(selected)


def select_checkpoints(
    trajectory: Sequence[dict[str, Any]],
    *,
    base_patch_count: int,
    thresholds: Sequence[float] = (0.90, 0.95, 0.99, 1.0),
) -> dict[str, dict[str, Any] | None]:
    by_count = {int(row["patch_count"]): dict(row) for row in trajectory}
    if base_patch_count not in by_count:
        raise ValueError("base prefix is outside trajectory")
    selected: dict[str, dict[str, Any] | None] = {
        "ZERO": by_count[0],
        "BASE": by_count[base_patch_count],
    }
    for threshold in thresholds:
        label = f"EDGE{int(round(100 * threshold))}"
        selected[label] = first_at_coverage(trajectory, threshold)
    selected["HYBRID_ALL"] = minimum_cost_prefix(trajectory)
    selected["HYBRID_AFTER_BASE"] = minimum_cost_prefix(
        trajectory, minimum_patch_count=base_patch_count
    )
    return selected
