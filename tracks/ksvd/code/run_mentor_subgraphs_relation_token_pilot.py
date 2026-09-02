"""Frozen no-label relation-token pilot on the real mentor 50-node subgraph bank.

One geometry (``s8_o2``) at the BASE checkpoint, rooted-canonical slot
ordering, root-candidate-grouped 3-fold split, per-fold train-only centering,
deterministic maximin INIT and same-init FINAL KSVD (K=24, T=3, T_min=1, 25
updates).  Three token families — INVARIANT (dynamic 2*s+2 invariant patch
descriptor: sorted normalized degrees + eigenvalues + density + triangle
density), INIT sparse codes, FINAL sparse codes — are each evaluated under
masked target prediction with BAG / TRUE_RELATION / SHUFFLED_RELATION context
features.  The target is the invariant descriptor of the masked patch; the
target patch token is always excluded from its own context.  TRUE/SHUFFLED
relation features use overlap fraction and center shortest-path distance only;
numeric source IDs never enter features.  Ridge decoders (alpha=0.01) are fit
on train folds only.  Reports graph-balanced overall/degree/spectrum/
density-triangle RMSE per graph and per density stratum, and classifies via
FINAL-focused gates (see ``classify``).

No residual tokens are built anywhere in this pilot: the masked target is the
invariant descriptor, so residual patch content would leak the target through
the context.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .beam8_coverage_operating_point import (
    prefix_coverage_trajectory,
    select_operating_checkpoints,
)
from .canonical_slots import reorder_cover_structurally
from .data_mentor_subgraphs import (
    STRATUM_NAMES,
    default_cache_path,
    default_source_path,
    density_stratum,
    load_bundle,
    select_pilot_indices,
)
from .from_scratch_unplanted_dictionary import (
    deterministic_maximin_initialization,
)
from .global_stable_ids import compute_global_wl_ids, reorder_by_stable_ids
from .imdb_walk_dictionary import encode_with_minimum_sparsity
from .ksvd import ksvd
from .marginal_candidate_cover import sample_marginal_candidate_cover
from .mentor_grouped_splits import (
    audit_fold_partition,
    audit_source_exposure,
    balanced_group_folds,
)
from .overlap_cover import _make_cover, patch_budget
from .overlap_stitching import (
    CoverExample,
    make_cover_example,
    stack_cover_examples,
)
from .run_beam8_coverage_operating_point_audit import GEOMETRIES
from .run_patch_relation_representation_audit import (
    EPS,
    _relation_arrays,
    all_pairs_shortest_paths,
    bag_features,
    patch_invariant_descriptor,
    pool_tokens,
    relation_features,
)
from .transition_decoder import fit_ridge_decoder, predict_ridge_decoder

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JSON = ROOT / "tracks/ksvd/results/mentor_subgraphs/relation_token_pilot_20260806.json"
DEFAULT_REPORT = ROOT / "tracks/ksvd/results/mentor_subgraphs/RELATION_TOKEN_PILOT_20260806.md"
PROTOCOL = "tracks/ksvd/docs/KSVD_MENTOR_SUBGRAPHS_RELATION_TOKEN_PILOT_PROTOCOL_20260806.md"
PROTOCOL_BEAM8 = "tracks/ksvd/docs/KSVD_MENTOR_SUBGRAPHS_BEAM8_PILOT_PROTOCOL_20260806.md"
PROTOCOL_GROUPED = "tracks/ksvd/docs/KSVD_MENTOR_SUBGRAPHS_GROUPED_RECONSTRUCTION_PROTOCOL_20260806.md"

# Frozen pilot parameters (2026-08-06).
GEOMETRY = "s8_o2"
CHECKPOINT = "BASE"
PILOT_SIZE = 100
PILOT_SEED = 20260806
COVER_SEED = 970201
MAXIMUM_PATCHES = 60
RETAINED_BEAM = 8
CANDIDATE_RESTARTS = 1
MULTIPLIER = 1.5
N_SPLITS = 3
SPLIT_SEED = 20260807
VIEW = "root_candidate_grouped"
N_ATOMS = 24
SPARSITY = 3
T_MIN = 1
ITERATIONS = 25
RIDGE_ALPHA = 0.01

FAMILIES = ("INVARIANT", "INIT", "FINAL")
BRANCHES = ("BAG", "TRUE_RELATION", "SHUFFLED_RELATION")
METRIC_KEYS = ("overall_rmse", "degree_rmse", "spectrum_rmse", "density_triangle_rmse")


def invariant_tokens(example: CoverExample) -> np.ndarray:
    """Per-patch invariant descriptors: rows (patches, 2*s + 2)."""
    return np.stack(
        [patch_invariant_descriptor(patch.adjacency) for patch in example.cover.patches],
        axis=0,
    )


def _split_codes(
    codes: np.ndarray, examples: Sequence[CoverExample]
) -> dict[int, np.ndarray]:
    """Slice stacked code columns (patches, atoms) back to per-graph rows."""
    result: dict[int, np.ndarray] = {}
    cursor = 0
    for example in examples:
        count = int(example.patch_vectors.shape[0])
        result[int(example.graph_index)] = np.asarray(
            codes.T[cursor : cursor + count], dtype=np.float64
        )
        cursor += count
    if cursor != codes.shape[1]:
        raise ValueError("code columns do not match the example patch counts")
    return result


def build_masked_matrices(
    examples: Sequence[CoverExample],
    tokens_by_graph: dict[int, np.ndarray],
    family: str,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Masked target-prediction rows for one token family.

    The target of each row is the invariant descriptor of the masked patch;
    the target patch token is excluded from every context (BAG deletes the
    target row, TRUE/SHUFFLED contexts only contain other patch indices).
    """
    if family not in FAMILIES:
        raise ValueError(f"unknown token family: {family}")
    feature_rows: dict[str, list[np.ndarray]] = {
        branch: [] for branch in BRANCHES
    }
    targets: list[np.ndarray] = []
    graph_indices: list[int] = []
    for example in examples:
        tokens = np.asarray(tokens_by_graph[int(example.graph_index)], dtype=np.float64)
        if tokens.ndim != 2 or tokens.shape[0] != len(example.cover.patches):
            raise ValueError("token matrix must have one row per cover patch")
        distances = all_pairs_shortest_paths(example.adjacency)
        for target_index, patch in enumerate(example.cover.patches):
            feature_rows["BAG"].append(bag_features(tokens, target_index))
            feature_rows["TRUE_RELATION"].append(
                relation_features(
                    tokens,
                    example.cover,
                    distances,
                    target_index,
                    shuffled=False,
                )
            )
            feature_rows["SHUFFLED_RELATION"].append(
                relation_features(
                    tokens,
                    example.cover,
                    distances,
                    target_index,
                    shuffled=True,
                )
            )
            targets.append(patch_invariant_descriptor(patch.adjacency))
            graph_indices.append(int(example.graph_index))
    return (
        {branch: np.stack(rows, axis=0) for branch, rows in feature_rows.items()},
        np.stack(targets, axis=0),
        np.asarray(graph_indices, dtype=np.int64),
    )


def _target_exclusion_audit(
    examples: Sequence[CoverExample],
    tokens_by_graph: dict[int, np.ndarray],
) -> bool:
    """Structural check that the target patch token never enters its own context."""
    for example in examples:
        tokens = np.asarray(tokens_by_graph[int(example.graph_index)], dtype=np.float64)
        if tokens.shape[0] != len(example.cover.patches):
            return False
        distances = all_pairs_shortest_paths(example.adjacency)
        for target_index in range(tokens.shape[0]):
            context = np.delete(tokens, target_index, axis=0)
            if context.shape[0] != tokens.shape[0] - 1:
                return False
            if not np.allclose(bag_features(tokens, target_index), pool_tokens(context)):
                return False
            context_indices, _overlap, _center = _relation_arrays(
                example.cover, distances, target_index
            )
            if any(int(index) == int(target_index) for index in context_indices):
                return False
    return True


def _metric_blocks(
    targets: np.ndarray, predictions: np.ndarray, degree_dim: int
) -> dict[str, float]:
    """Dynamic-dimension RMSE blocks: degree, spectrum, density/triangle."""
    residual = np.asarray(predictions, dtype=np.float64) - np.asarray(
        targets, dtype=np.float64
    )
    if residual.shape != targets.shape or targets.shape[1] != 2 * degree_dim + 2:
        raise ValueError(
            "masked target/prediction shape mismatch "
            f"(expected dimension {2 * degree_dim + 2})"
        )

    def rmse(block: np.ndarray) -> float:
        return float(np.sqrt(np.mean(block**2))) if block.size else 0.0

    return {
        "overall_rmse": rmse(residual),
        "degree_rmse": rmse(residual[:, :degree_dim]),
        "spectrum_rmse": rmse(residual[:, degree_dim : 2 * degree_dim]),
        "density_triangle_rmse": rmse(residual[:, 2 * degree_dim :]),
    }


def graph_balanced_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    graph_indices: np.ndarray,
    *,
    degree_dim: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Per-graph rows plus the graph-balanced (not per-row) mean summary."""
    rows: list[dict[str, Any]] = []
    for graph_index in sorted(set(int(value) for value in graph_indices)):
        mask = graph_indices == graph_index
        rows.append(
            {
                "graph_index": int(graph_index),
                **_metric_blocks(targets[mask], predictions[mask], degree_dim),
            }
        )
    return rows, {
        key: float(np.mean([float(row[key]) for row in rows])) for key in METRIC_KEYS
    }


def _run_fold(
    train_examples: Sequence[CoverExample],
    test_examples: Sequence[CoverExample],
    *,
    fold_index: int,
    patch_size: int,
    n_atoms: int = N_ATOMS,
    sparsity: int = SPARSITY,
    iterations: int = ITERATIONS,
    ridge_alpha: float = RIDGE_ALPHA,
    strata_by_source_index: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Train-only masked relation-token evaluation for one root-grouped fold."""
    train_examples = tuple(train_examples)
    test_examples = tuple(test_examples)
    if not train_examples or not test_examples:
        raise ValueError("a fold needs both train and test examples")
    raw_train = stack_cover_examples(train_examples)
    raw_test = stack_cover_examples(test_examples)
    train_mean = np.mean(raw_train, axis=1, keepdims=True)
    centered_train = raw_train - train_mean
    centered_test = raw_test - train_mean

    # Deterministic maximin INIT; FINAL continues from the same dictionary.
    initial_dictionary, initialization = deterministic_maximin_initialization(
        centered_train, n_atoms
    )
    final_dictionary, _training_codes, training_info = ksvd(
        centered_train,
        n_atoms=n_atoms,
        T=sparsity,
        T_min=T_MIN,
        n_iter=iterations,
        seed=0,
        initial_dictionary=initial_dictionary,
    )
    init_train_codes = encode_with_minimum_sparsity(
        centered_train, initial_dictionary, sparsity=sparsity, minimum_sparsity=T_MIN
    )
    init_test_codes = encode_with_minimum_sparsity(
        centered_test, initial_dictionary, sparsity=sparsity, minimum_sparsity=T_MIN
    )
    final_train_codes = encode_with_minimum_sparsity(
        centered_train, final_dictionary, sparsity=sparsity, minimum_sparsity=T_MIN
    )
    final_test_codes = encode_with_minimum_sparsity(
        centered_test, final_dictionary, sparsity=sparsity, minimum_sparsity=T_MIN
    )

    train_tokens_by_graph: dict[str, dict[int, np.ndarray]] = {
        "INVARIANT": {
            int(example.graph_index): invariant_tokens(example)
            for example in train_examples
        },
        "INIT": _split_codes(init_train_codes, train_examples),
        "FINAL": _split_codes(final_train_codes, train_examples),
    }
    test_tokens_by_graph: dict[str, dict[int, np.ndarray]] = {
        "INVARIANT": {
            int(example.graph_index): invariant_tokens(example)
            for example in test_examples
        },
        "INIT": _split_codes(init_test_codes, test_examples),
        "FINAL": _split_codes(final_test_codes, test_examples),
    }

    expected_train_rows = sum(
        int(example.patch_vectors.shape[0]) for example in train_examples
    )
    expected_test_rows = sum(
        int(example.patch_vectors.shape[0]) for example in test_examples
    )
    train_ids = {int(example.graph_index) for example in train_examples}
    test_ids = {int(example.graph_index) for example in test_examples}

    branches: dict[str, Any] = {}
    target_dimension: int | None = None
    finite = True
    for family in FAMILIES:
        train_features, train_targets, train_graph_rows = build_masked_matrices(
            train_examples, train_tokens_by_graph[family], family
        )
        test_features, test_targets, test_graph_rows = build_masked_matrices(
            test_examples, test_tokens_by_graph[family], family
        )
        if target_dimension is None:
            target_dimension = int(train_targets.shape[1])
        if int(train_targets.shape[1]) != target_dimension or int(
            test_targets.shape[1]
        ) != target_dimension:
            raise ValueError("target dimension changed across families/folds")
        finite = finite and all(
            np.all(np.isfinite(matrix)) for matrix in train_features.values()
        ) and all(np.all(np.isfinite(matrix)) for matrix in test_features.values())
        finite = finite and np.all(np.isfinite(train_targets)) and np.all(
            np.isfinite(test_targets)
        )
        branches[family] = {}
        for branch in BRANCHES:
            model = fit_ridge_decoder(
                train_features[branch], train_targets, alpha=ridge_alpha
            )
            predictions = predict_ridge_decoder(model, test_features[branch])
            graph_rows, summary = graph_balanced_metrics(
                test_targets,
                predictions,
                test_graph_rows,
                degree_dim=patch_size,
            )
            branches[family][branch] = {
                "summary": summary,
                "graphs": [
                    {
                        "source_index": int(row["graph_index"]),
                        "density_stratum": (
                            strata_by_source_index.get(int(row["graph_index"]), "unknown")
                            if strata_by_source_index is not None
                            else "unknown"
                        ),
                        **{key: float(row[key]) for key in METRIC_KEYS},
                    }
                    for row in graph_rows
                ],
            }

    target_excluded = _target_exclusion_audit(
        test_examples, test_tokens_by_graph["INVARIANT"]
    ) and _target_exclusion_audit(
        train_examples, train_tokens_by_graph["INVARIANT"]
    )
    invariants = {
        "train_test_graph_isolation": bool(not (train_ids & test_ids)),
        "target_dimension": int(target_dimension) if target_dimension is not None else -1,
        "target_dimension_matches_2s_plus_2": bool(
            target_dimension is not None and target_dimension == 2 * patch_size + 2
        ),
        "train_row_count": int(train_targets.shape[0]),
        "test_row_count": int(test_targets.shape[0]),
        "expected_train_row_count": int(expected_train_rows),
        "expected_test_row_count": int(expected_test_rows),
        "train_row_count_matches": bool(train_targets.shape[0] == expected_train_rows),
        "test_row_count_matches": bool(test_targets.shape[0] == expected_test_rows),
        "finite": bool(finite),
        "target_excluded_from_context": bool(target_excluded),
        "graph_balanced_per_graph_rows": all(
            len(branches[family][branch]["graphs"]) == len(test_examples)
            for family in FAMILIES
            for branch in BRANCHES
        ),
        "residual_tokens_used": False,
    }
    check_keys = (
        "train_test_graph_isolation",
        "target_dimension_matches_2s_plus_2",
        "train_row_count_matches",
        "test_row_count_matches",
        "finite",
        "target_excluded_from_context",
        "graph_balanced_per_graph_rows",
    )
    invariants["passed"] = all(invariants[key] for key in check_keys)

    stratum_accum: dict[tuple[str, str, str], dict[str, float]] = {}
    for family in FAMILIES:
        for branch in BRANCHES:
            for row in branches[family][branch]["graphs"]:
                accumulator = stratum_accum.setdefault(
                    (family, branch, row["density_stratum"]),
                    {
                        "count": 0.0,
                        "overall": 0.0,
                        "degree": 0.0,
                        "spectrum": 0.0,
                        "density_triangle": 0.0,
                    },
                )
                accumulator["count"] += 1.0
                accumulator["overall"] += float(row["overall_rmse"])
                accumulator["degree"] += float(row["degree_rmse"])
                accumulator["spectrum"] += float(row["spectrum_rmse"])
                accumulator["density_triangle"] += float(row["density_triangle_rmse"])
    stratum_rows = [
        {
            "family": key[0],
            "branch": key[1],
            "stratum": key[2],
            "test_graph_count": int(acc["count"]),
            "overall_rmse": acc["overall"] / max(acc["count"], 1.0),
            "degree_rmse": acc["degree"] / max(acc["count"], 1.0),
            "spectrum_rmse": acc["spectrum"] / max(acc["count"], 1.0),
            "density_triangle_rmse": acc["density_triangle"] / max(acc["count"], 1.0),
        }
        for key, acc in sorted(stratum_accum.items())
    ]

    return {
        "fold_index": int(fold_index),
        "train_graph_count": len(train_examples),
        "test_graph_count": len(test_examples),
        "train_patch_count": int(expected_train_rows),
        "test_patch_count": int(expected_test_rows),
        "train_mean_norm": float(np.linalg.norm(train_mean)),
        "initialization": initialization,
        "training_recon_curve": [float(value) for value in training_info.get("recon_curve", [])],
        "final_train_recon_rel": training_info.get("recon_rel"),
        "branches": branches,
        "stratum_rows": stratum_rows,
        "invariants": invariants,
    }


def _merge_stratum_rows(folds: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge per-fold per-stratum graph-balanced RMSE, weighted by test graphs."""
    merged: dict[tuple[str, str, str], dict[str, float]] = {}
    for fold in folds:
        for row in fold["stratum_rows"]:
            key = (row["family"], row["branch"], row["stratum"])
            accumulator = merged.setdefault(
                key,
                {
                    "count": 0.0,
                    "overall": 0.0,
                    "degree": 0.0,
                    "spectrum": 0.0,
                    "density_triangle": 0.0,
                },
            )
            count = float(row["test_graph_count"])
            accumulator["count"] += count
            accumulator["overall"] += count * float(row["overall_rmse"])
            accumulator["degree"] += count * float(row["degree_rmse"])
            accumulator["spectrum"] += count * float(row["spectrum_rmse"])
            accumulator["density_triangle"] += count * float(row["density_triangle_rmse"])
    return [
        {
            "family": key[0],
            "branch": key[1],
            "stratum": key[2],
            "test_graph_count": int(acc["count"]),
            "overall_rmse": acc["overall"] / max(acc["count"], 1.0),
            "degree_rmse": acc["degree"] / max(acc["count"], 1.0),
            "spectrum_rmse": acc["spectrum"] / max(acc["count"], 1.0),
            "density_triangle_rmse": acc["density_triangle"] / max(acc["count"], 1.0),
        }
        for key, acc in sorted(merged.items())
    ]


def classify(
    folds: Sequence[dict[str, Any]], fold_audit: dict[str, Any] | None = None
) -> dict[str, Any]:
    """FINAL-focused gates plus the invariant gate and fold integrity.

    Frozen labels (protocol section 8):
      * MENTOR_RELATION_TOKEN_PILOT_SUPPORTED — every gate passes;
      * RELATION_SIGNAL_PRESENT_BUT_NO_FINAL_KSVD_VALUE — relation gates and
        the invariant gate pass but FINAL_TRUE does not improve INIT_TRUE;
      * MENTOR_RELATION_SIGNAL_BELOW_GATE — positive TRUE gains below 2% or
        unstable wins;
      * REJECT_CURRENT_MENTOR_RELATION_TOKEN — TRUE not better than matched
        controls, or a data/invariant gate fails.
    """
    folds = tuple(folds)
    valid = [fold for fold in folds if not fold.get("skipped")]
    if len(valid) != N_SPLITS:
        raise ValueError(f"expected {N_SPLITS} available folds, got {len(valid)}")

    mean_branches: dict[str, dict[str, dict[str, float]]] = {}
    for family in FAMILIES:
        mean_branches[family] = {}
        for branch in BRANCHES:
            metric_keys = valid[0]["branches"][family][branch]["summary"].keys()
            mean_branches[family][branch] = {
                key: float(
                    np.mean(
                        [
                            float(fold["branches"][family][branch]["summary"][key])
                            for fold in valid
                        ]
                    )
                )
                for key in metric_keys
            }

    merged = _merge_stratum_rows(valid)
    strata_by_key: dict[tuple[str, str], dict[str, float]] = {}
    for row in merged:
        strata_by_key.setdefault((row["family"], row["branch"]), {})[
            row["stratum"]
        ] = float(row["overall_rmse"])

    final_bag = mean_branches["FINAL"]["BAG"]["overall_rmse"]
    final_true = mean_branches["FINAL"]["TRUE_RELATION"]["overall_rmse"]
    final_shuffled = mean_branches["FINAL"]["SHUFFLED_RELATION"]["overall_rmse"]
    final_bag_gain = float((final_bag - final_true) / max(final_bag, EPS))
    final_shuffled_gain = float(
        (final_shuffled - final_true) / max(final_shuffled, EPS)
    )
    wins_both_by_fold = [
        bool(
            float(fold["branches"]["FINAL"]["TRUE_RELATION"]["summary"]["overall_rmse"])
            < float(fold["branches"]["FINAL"]["BAG"]["summary"]["overall_rmse"])
            and float(
                fold["branches"]["FINAL"]["TRUE_RELATION"]["summary"]["overall_rmse"]
            )
            < float(
                fold["branches"]["FINAL"]["SHUFFLED_RELATION"]["summary"]["overall_rmse"]
            )
        )
        for fold in valid
    ]
    strata_wins: dict[str, bool] = {}
    for stratum in STRATUM_NAMES:
        bag = strata_by_key.get(("FINAL", "BAG"), {}).get(stratum)
        true = strata_by_key.get(("FINAL", "TRUE_RELATION"), {}).get(stratum)
        shuffled = strata_by_key.get(("FINAL", "SHUFFLED_RELATION"), {}).get(stratum)
        strata_wins[stratum] = bool(
            bag is not None
            and true is not None
            and shuffled is not None
            and true < bag
            and true < shuffled
        )
    strata_won_count = int(sum(strata_wins.values()))

    init_true = mean_branches["INIT"]["TRUE_RELATION"]["overall_rmse"]
    final_improves_init = bool(final_true < init_true)
    invariant_true = mean_branches["INVARIANT"]["TRUE_RELATION"]["overall_rmse"]
    invariant_shuffled = mean_branches["INVARIANT"]["SHUFFLED_RELATION"]["overall_rmse"]
    invariant_gain = float(
        (invariant_shuffled - invariant_true) / max(invariant_shuffled, EPS)
    )
    integrity = all(bool(fold["invariants"]["passed"]) for fold in valid) and bool(
        fold_audit is None or fold_audit.get("passed", False)
    )

    checks: dict[str, Any] = {
        "integrity_passed": integrity,
        "final_true_vs_bag_gain_at_least_002": bool(final_bag_gain >= 0.02),
        "final_true_vs_shuffled_gain_at_least_002": bool(final_shuffled_gain >= 0.02),
        "final_true_wins_both_all_folds": bool(all(wins_both_by_fold)),
        "final_true_wins_both_strata_at_least_4_of_5": bool(strata_won_count >= 4),
        "final_true_improves_init_true": final_improves_init,
        "invariant_true_vs_shuffled_gain_at_least_002": bool(invariant_gain >= 0.02),
    }
    relation_checks = (
        "final_true_vs_bag_gain_at_least_002",
        "final_true_vs_shuffled_gain_at_least_002",
        "final_true_wins_both_all_folds",
        "final_true_wins_both_strata_at_least_4_of_5",
    )
    relation_gate = bool(integrity and all(checks[key] for key in relation_checks))
    invariant_gate = bool(
        integrity and checks["invariant_true_vs_shuffled_gain_at_least_002"]
    )
    final_vs_init_gate = bool(integrity and checks["final_true_improves_init_true"])
    positive_gains = bool(final_bag_gain > 0.0 and final_shuffled_gain > 0.0)
    if not integrity:
        label = "REJECT_CURRENT_MENTOR_RELATION_TOKEN"
    elif relation_gate and invariant_gate and final_vs_init_gate:
        label = "MENTOR_RELATION_TOKEN_PILOT_SUPPORTED"
    elif relation_gate and invariant_gate and not final_vs_init_gate:
        label = "RELATION_SIGNAL_PRESENT_BUT_NO_FINAL_KSVD_VALUE"
    elif positive_gains:
        label = "MENTOR_RELATION_SIGNAL_BELOW_GATE"
    else:
        label = "REJECT_CURRENT_MENTOR_RELATION_TOKEN"
    return {
        "classification": label,
        "checks": checks,
        "relation_gate": relation_gate,
        "invariant_gate": invariant_gate,
        "final_vs_init_gate": final_vs_init_gate,
        "positive_gains": positive_gains,
        "final_true_vs_bag_rmse_reduction": final_bag_gain,
        "final_true_vs_shuffled_rmse_reduction": final_shuffled_gain,
        "invariant_true_vs_shuffled_rmse_reduction": invariant_gain,
        "final_true_improves_init_true_delta": float(init_true - final_true),
        "final_wins_both_by_fold": wins_both_by_fold,
        "final_strata_wins": strata_wins,
        "final_strata_won_count": strata_won_count,
        "mean_branches": mean_branches,
    }


def _fmt(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def render_report(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    lines = [
        "# 导师 50-node 真实子图：no-label relation-token pilot",
        "",
        f"> 日期：2026-08-06",
        f"> 协议：`{payload['protocol']}`",
        f"> 上游协议：`{payload['protocol_beam8']}`；`{payload['protocol_grouped']}`",
        f"> 判定：`{decision['classification']}`",
        "",
        "## 1. 数据与 selection",
        "",
        f"- source SHA-256：`{payload['data']['source_sha256']}`；",
        f"- cache graphs：`{payload['data']['cache_graph_count']}`；pilot：`{payload['selection']['size']}`；",
        f"- pilot roots：`{payload['selection']['n_distinct_roots']}`；density quotas：`{payload['selection']['selected_counts']}`；",
        f"- geometry：`{payload['config']['geometry']}`（BASE）；cover seed：`{payload['config']['cover_seed']}`；",
        f"- max patches：`{payload['config']['maximum_patches']}`；Beam `{payload['config']['retained_beam']}`/R`{payload['config']['candidate_restarts']}`；rooted-canonical slots；",
        f"- split：`{payload['config']['n_splits']}`-fold `{payload['config']['view']}`，seed `{payload['config']['split_seed']}`；group leakage：`{payload['fold_audit']['group_leakage_count']}`；",
        f"- source exposure（仅审计，不进 features）：node seen=`{_fmt(payload['source_exposure_summary']['test_source_node_seen_fraction'])}`，edge seen=`{_fmt(payload['source_exposure_summary']['test_source_edge_seen_fraction'])}`；",
        f"- KSVD：K=`{payload['config']['n_atoms']}`，T=`{payload['config']['sparsity']}`，T_min=`{payload['config']['T_min']}`，updates=`{payload['config']['iterations']}`；",
        f"  centering=train-only，INIT=deterministic maximin，FINAL=same-init；ridge alpha=`{payload['config']['ridge_alpha']}`。",
        f"- token families：`{payload['config']['token_families']}`；branches：`{payload['config']['branches']}`；target=invariant patch descriptor（2·s+2）。",
        f"- labels used：`False`；source IDs in features：`False`；residual tokens：`False`；strata used as labels：`False`。",
        "",
        "## 2. 总判定",
        "",
        f"- integrity gate：`{decision['checks']['integrity_passed']}`；",
        f"- FINAL relation gate：`{decision['relation_gate']}`；",
        f"- invariant gate：`{decision['invariant_gate']}`；",
        f"- FINAL_TRUE vs INIT_TRUE gate：`{decision['final_vs_init_gate']}`；",
        f"- FINAL TRUE vs BAG RMSE reduction：`{_fmt(decision['final_true_vs_bag_rmse_reduction'])}`；",
        f"- FINAL TRUE vs SHUFFLED RMSE reduction：`{_fmt(decision['final_true_vs_shuffled_rmse_reduction'])}`；",
        f"- FINAL TRUE wins both by fold：`{decision['final_wins_both_by_fold']}`；",
        f"- FINAL TRUE wins both strata：`{decision['final_strata_won_count']}/5`（`{decision['final_strata_wins']}`）；",
        f"- FINAL TRUE vs INIT TRUE delta：`{_fmt(decision['final_true_improves_init_true_delta'])}`；",
        f"- invariant TRUE vs SHUFFLED reduction：`{_fmt(decision['invariant_true_vs_shuffled_rmse_reduction'])}`；",
        f"- registered checks：`{decision['checks']}`。",
        "",
        "## 3. Masked prediction（graph-balanced，fold 均值）",
        "",
        "| family | branch | overall RMSE | degree | spectrum | density/triangle |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for family in FAMILIES:
        for branch in BRANCHES:
            summary = decision["mean_branches"][family][branch]
            lines.append(
                f"| {family} | {branch} | {_fmt(summary['overall_rmse'], 5)} | "
                f"{_fmt(summary['degree_rmse'], 5)} | {_fmt(summary['spectrum_rmse'], 5)} | "
                f"{_fmt(summary['density_triangle_rmse'], 5)} |"
            )
    lines.extend(
        [
            "",
            "## 4. Density strata（FINAL，fold 合并，test-graph 加权）",
            "",
            "| stratum | graphs | BAG | TRUE | SHUFFLED | TRUE<BAG & TRUE<SHUFFLED |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for stratum in STRATUM_NAMES:
        bag = next(
            (
                row
                for row in payload["stratum_rows"]
                if row["family"] == "FINAL"
                and row["branch"] == "BAG"
                and row["stratum"] == stratum
            ),
            None,
        )
        true = next(
            (
                row
                for row in payload["stratum_rows"]
                if row["family"] == "FINAL"
                and row["branch"] == "TRUE_RELATION"
                and row["stratum"] == stratum
            ),
            None,
        )
        shuffled = next(
            (
                row
                for row in payload["stratum_rows"]
                if row["family"] == "FINAL"
                and row["branch"] == "SHUFFLED_RELATION"
                and row["stratum"] == stratum
            ),
            None,
        )
        if bag is None or true is None or shuffled is None:
            continue
        won = bool(
            true["overall_rmse"] < bag["overall_rmse"]
            and true["overall_rmse"] < shuffled["overall_rmse"]
        )
        lines.append(
            f"| {stratum} | {true['test_graph_count']} | {_fmt(bag['overall_rmse'], 5)} | "
            f"{_fmt(true['overall_rmse'], 5)} | {_fmt(shuffled['overall_rmse'], 5)} | {won} |"
        )
    lines.extend(
        [
            "",
            "## 5. Per-graph rows",
            "",
            f"每 graph、每 family、每 branch 的 overall/degree/spectrum/density_triangle RMSE "
            f"见 JSON `folds[].branches[family][branch].graphs`（共 "
            f"{sum(len(fold['branches']['FINAL']['TRUE_RELATION']['graphs']) for fold in payload['folds'])} "
            f"test-graph × {len(FAMILIES)} × {len(BRANCHES)} 行）。",
            "",
            "## 6. 边界",
            "",
            "- density strata 与 root_candidate 只用于 split/条件报告，不是 labels；numeric source IDs 不进入任何 feature。",
            "- target 是 invariant patch descriptor（2·s+2）；target patch token 从自身 context 中排除（BAG 删除 target 行，TRUE/SHUFFLED 只用其他 patch 下标）。",
            "- TRUE/SHUFFLED 只差 context 的 token 对齐（overlap fraction 与 center shortest-path distance 相同）；SHUFFLED 用 cyclic roll 打乱对齐。",
            "- 完全不用 residual tokens：masked target 是 invariant descriptor，residual 会把 target 内容泄漏进 context。",
            "- 本 pilot 不训练 Transformer、不使用 graph labels；只判定 FINAL 字典的 relation token 是否通过冻结 gates。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="No-label relation-token pilot on the mentor subgraph bank "
        "(s8_o2 BASE, root-grouped folds)."
    )
    parser.add_argument("--source", type=Path, default=default_source_path())
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--pilot-size", type=int, default=PILOT_SIZE)
    parser.add_argument("--pilot-seed", type=int, default=PILOT_SEED)
    parser.add_argument("--cover-seed", type=int, default=COVER_SEED)
    parser.add_argument("--maximum-patches", type=int, default=MAXIMUM_PATCHES)
    parser.add_argument("--n-atoms", type=int, default=N_ATOMS)
    parser.add_argument("--sparsity", type=int, default=SPARSITY)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    cache_path = args.cache or default_cache_path(args.source)
    bundle = load_bundle(
        source_pkl=args.source, cache_path=cache_path, force=args.force_cache
    )
    selection = select_pilot_indices(
        bundle, n_pilot=args.pilot_size, seed=args.pilot_seed
    )
    pilot_indices = selection.indices
    all_strata = density_stratum(bundle.avg_degrees)
    strata_values = all_strata[pilot_indices]
    root_values = bundle.roots[pilot_indices]

    patch_size, overlap = GEOMETRIES[GEOMETRY]
    geometry_index = tuple(GEOMETRIES).index(GEOMETRY)

    prepared: dict[int, CoverExample] = {}
    coverage_missing: list[dict[str, Any]] = []
    for position, raw_source_index in enumerate(pilot_indices):
        source_index = int(raw_source_index)
        source_adjacency = bundle.adjacency[source_index].astype(np.int8, copy=False)
        ids = compute_global_wl_ids(source_adjacency)
        stable_adjacency = reorder_by_stable_ids(source_adjacency, ids)
        base_budget = patch_budget(
            stable_adjacency,
            patch_size=patch_size,
            target_overlap=overlap,
            edge_capacity_multiplier=MULTIPLIER,
        )
        sampler_seed = int(
            np.random.SeedSequence([args.cover_seed, source_index, geometry_index])
            .generate_state(1, dtype=np.uint32)[0]
        )
        try:
            stable_cover = sample_marginal_candidate_cover(
                stable_adjacency,
                np.random.default_rng(sampler_seed),
                n_patches=args.maximum_patches,
                patch_size=patch_size,
                target_overlap=overlap,
                retained_beam=RETAINED_BEAM,
                candidate_restarts=CANDIDATE_RESTARTS,
                allow_partial=True,
            )
        except Exception as exc:  # noqa: BLE001 - recorded per graph
            coverage_missing.append(
                {
                    "source_index": source_index,
                    "reason": f"sampling failed: {type(exc).__name__}: {exc}",
                }
            )
            continue
        trajectory = prefix_coverage_trajectory(
            stable_adjacency,
            stable_cover,
            patch_size=patch_size,
            overlap=overlap,
            maximum_patches=args.maximum_patches,
        )
        selected = select_operating_checkpoints(
            trajectory, base_patch_count=base_budget
        )
        row = selected[CHECKPOINT]
        if row is None:
            coverage_missing.append(
                {
                    "source_index": source_index,
                    "reason": "BASE checkpoint unreachable",
                }
            )
            continue
        prefix = _make_cover(
            f"{GEOMETRY}_{CHECKPOINT}",
            stable_cover.patches[: int(row["patch_count"])],
            stable_cover.segment_ids[: int(row["patch_count"])],
            stable_cover.target_edges[: int(row["patch_count"])],
            stable_cover.bridge_lengths[: int(row["patch_count"])],
        )
        ordered, _diagnostics = reorder_cover_structurally(
            stable_adjacency, prefix, "rooted_canonical"
        )
        prepared[source_index] = make_cover_example(
            source_index,
            STRATUM_NAMES[int(all_strata[source_index])],
            int(round(float(bundle.avg_degrees[source_index]))),
            stable_adjacency,
            ordered,
        )
        if (position + 1) % 25 == 0 or position + 1 == len(pilot_indices):
            print(f"prepared={position + 1}/{len(pilot_indices)}", flush=True)

    folds = balanced_group_folds(
        pilot_indices,
        strata_values,
        root_values,
        n_splits=N_SPLITS,
        seed=args.split_seed,
        view=VIEW,
    )
    fold_audit = audit_fold_partition(
        folds,
        pilot_indices,
        groups_array_by_index=dict(zip(pilot_indices, root_values)),
        require_group_integrity=True,
    )

    source_exposure = [
        audit_source_exposure(bundle, fold, all_strata) for fold in folds
    ]
    source_exposure_summary = {
        "test_source_node_seen_fraction": float(
            np.mean([float(row["test_source_node_seen_fraction"]) for row in source_exposure])
        ),
        "source_node_jaccard": float(
            np.mean([float(row["source_node_jaccard"]) for row in source_exposure])
        ),
        "test_source_edge_seen_fraction": float(
            np.mean([float(row["test_source_edge_seen_fraction"]) for row in source_exposure])
        ),
        "source_edge_jaccard": float(
            np.mean([float(row["source_edge_jaccard"]) for row in source_exposure])
        ),
        "root_intersection_count": int(
            sum(int(row["root_intersection_count"]) for row in source_exposure)
        ),
    }

    fold_payloads: list[dict[str, Any]] = []
    for fold in folds:
        train_examples = [
            prepared[int(index)]
            for index in fold.train_indices
            if int(index) in prepared
        ]
        test_examples = [
            prepared[int(index)]
            for index in fold.test_indices
            if int(index) in prepared
        ]
        if not train_examples or not test_examples:
            fold_payloads.append(
                {
                    "fold_index": int(fold.fold_index),
                    "skipped": True,
                    "train_graph_count": len(train_examples),
                    "test_graph_count": len(test_examples),
                }
            )
            continue
        fold_payloads.append(
            _run_fold(
                train_examples,
                test_examples,
                fold_index=fold.fold_index,
                patch_size=patch_size,
                n_atoms=args.n_atoms,
                sparsity=args.sparsity,
                iterations=args.iterations,
                ridge_alpha=RIDGE_ALPHA,
                strata_by_source_index={
                    int(index): prepared[int(index)].family
                    for index in fold.test_indices
                    if int(index) in prepared
                },
            )
        )
        print(
            f"fold={fold.fold_index} graphs={len(test_examples)} "
            f"final_true={fold_payloads[-1]['branches']['FINAL']['TRUE_RELATION']['summary']['overall_rmse']:.5f}",
            flush=True,
        )

    decision = classify(fold_payloads, fold_audit)
    stratum_rows = _merge_stratum_rows(
        [fold for fold in fold_payloads if not fold.get("skipped")]
    )
    payload = {
        "protocol": PROTOCOL,
        "protocol_beam8": PROTOCOL_BEAM8,
        "protocol_grouped": PROTOCOL_GROUPED,
        "config": {
            "pilot_size": args.pilot_size,
            "pilot_seed": args.pilot_seed,
            "cover_seed": args.cover_seed,
            "maximum_patches": args.maximum_patches,
            "multiplier": MULTIPLIER,
            "retained_beam": RETAINED_BEAM,
            "candidate_restarts": CANDIDATE_RESTARTS,
            "geometry": GEOMETRY,
            "checkpoint": CHECKPOINT,
            "slots": "rooted_canonical",
            "n_splits": N_SPLITS,
            "split_seed": args.split_seed,
            "view": VIEW,
            "n_atoms": args.n_atoms,
            "sparsity": args.sparsity,
            "T_min": T_MIN,
            "iterations": args.iterations,
            "ridge_alpha": RIDGE_ALPHA,
            "centering": "train_only",
            "initialization": "deterministic_maximin",
            "same_init_final": True,
            "target": "invariant_patch_descriptor",
            "target_dimension_formula": "2*patch_size+2",
            "token_families": list(FAMILIES),
            "branches": list(BRANCHES),
            "labels_used": False,
            "source_ids_in_features": False,
            "residual_tokens_used": False,
            "strata_used_as_labels": False,
        },
        "data": {
            "source_path": bundle.metadata.get("source_path"),
            "source_sha256": bundle.metadata.get("source_sha256"),
            "cache_path": str(cache_path),
            "cache_graph_count": bundle.n_graphs,
            "metadata": bundle.metadata,
        },
        "selection": selection.summary(),
        "pilot_indices": pilot_indices.tolist(),
        "coverage_missing": coverage_missing,
        "fold_audit": fold_audit,
        "source_exposure": source_exposure,
        "source_exposure_summary": source_exposure_summary,
        "folds": fold_payloads,
        "stratum_rows": stratum_rows,
        "decision": decision,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.report.write_text(render_report(payload), encoding="utf-8")
    print(f"decision={decision['classification']}")
    print(f"wrote {args.json}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
