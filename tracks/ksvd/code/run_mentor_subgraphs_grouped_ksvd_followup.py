"""Frozen grouped-split KSVD reconstruction follow-up on the mentor subgraph bank.

Implements the 2026-08-06 grouped reconstruction protocol: three geometries,
BASE/FAIR95 checkpoints, random-reference and root-candidate-grouped split
views, train-only centering, RAW/PCA3/RANDOM/INIT/FINAL stages, uncorrected and
exact-residual-corrected stitching, source-exposure and exact canonical
patch-vector overlap audits, plus the frozen split/data, KSVD optimization, and
control/Pareto gates.

Source identities never enter patch vectors or dictionaries.  Source graph
indices are noncontiguous; every fold stores source indices.  When auditing
patch source-node sets, a stable slot ``k`` corresponds to source row position
``ids.order[k]`` of the stable ordering, so slot sets are mapped back through
``ids.order`` before touching ``bundle.global_node_ids``.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
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
    dictionary_metrics,
)
from .global_stable_ids import StableNodeIDs, compute_global_wl_ids, reorder_by_stable_ids
from .imdb_walk_dictionary import encode_with_minimum_sparsity
from .ksvd import ksvd
from .marginal_candidate_cover import sample_marginal_candidate_cover
from .mentor_grouped_splits import (
    MentorFold,
    audit_source_exposure,
    make_mentor_folds,
)
from .overlap_cover import PatchCover, _make_cover, patch_budget
from .overlap_stitching import (
    CoverExample,
    fit_pca_basis,
    graph_balanced_stitch_summary,
    make_cover_example,
    stack_cover_examples,
    stitch_patch_predictions,
)
from .run_beam8_coverage_operating_point_audit import GEOMETRIES

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JSON = ROOT / "tracks/ksvd/results/mentor_subgraphs/grouped_ksvd_followup_20260806.json"
DEFAULT_REPORT = ROOT / "tracks/ksvd/results/mentor_subgraphs/GROUPED_KSVD_FOLLOWUP_20260806.md"
DEFAULT_PILOT_JSON = ROOT / "tracks/ksvd/results/mentor_subgraphs/beam8_pilot_20260806.json"
PROTOCOL = "tracks/ksvd/docs/KSVD_MENTOR_SUBGRAPHS_GROUPED_RECONSTRUCTION_PROTOCOL_20260806.md"

CHECKPOINTS = ("BASE", "FAIR95")
VIEWS = ("random_reference", "root_candidate_grouped")
STAGES = ("RAW", "PCA3", "RANDOM", "INIT", "FINAL")
CORRECTIONS = ("uncorrected", "residual_corrected")
STITCH_METRICS = (
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

# Frozen protocol parameters (section 5 of the protocol).
N_ATOMS = 24
SPARSITY = 3
T_MIN = 1
KSVD_SEED = 0
PCA_RANK = 3
RANDOM_DICTIONARY_SEED_BASE = 970301
DECOMPOSITION_TOLERANCE = 1e-10
EPS = 1e-12


@dataclass(frozen=True)
class PreparedGraph:
    """One regenerated BASE/FAIR95 rooted-canonical prefix for a pilot graph."""

    source_index: int
    pilot_position: int
    density_stratum: str
    average_degree: float
    geometry: str
    checkpoint: str
    patch_count: int
    ids: StableNodeIDs
    example: CoverExample


def _prefix_cover(cover: PatchCover, patch_count: int, method: str) -> PatchCover:
    if not 1 <= patch_count <= len(cover.patches):
        raise ValueError("prefix patch count is outside cover")
    return _make_cover(
        method,
        cover.patches[:patch_count],
        cover.segment_ids[:patch_count],
        cover.target_edges[:patch_count],
        cover.bridge_lengths[:patch_count],
    )


def _residual_edges(example: CoverExample) -> set[tuple[int, int]]:
    observed = {
        tuple(sorted(pair))
        for patch in example.cover.patches
        for pair in combinations(patch.node_ids, 2)
    }
    return {
        (left, right)
        for left in range(example.adjacency.shape[0])
        for right in range(left + 1, example.adjacency.shape[0])
        if example.adjacency[left, right] and (left, right) not in observed
    }


def _evaluate(
    examples: Sequence[CoverExample],
    reconstructed: np.ndarray | None,
    *,
    residual_corrected: bool,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Stitch patch predictions and return per-graph rows plus the graph-balanced summary."""
    rows = []
    cursor = 0
    for example in examples:
        patch_count = example.patch_vectors.shape[0]
        predictions = (
            example.patch_vectors
            if reconstructed is None
            else reconstructed[:, cursor : cursor + patch_count].T
        )
        metrics = stitch_patch_predictions(
            example,
            predictions,
            exact_residual_edges=(
                _residual_edges(example) if residual_corrected else None
            ),
        )
        rows.append(
            {
                "source_index": example.graph_index,
                **metrics,
            }
        )
        cursor += patch_count
    if reconstructed is not None and cursor != reconstructed.shape[1]:
        raise ValueError("reconstruction columns do not match examples")
    return rows, graph_balanced_stitch_summary(rows)


def _compact_health(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "patch_count",
        "relative_reconstruction_error",
        "mean_nonzeros_per_patch",
        "minimum_nonzeros_per_patch",
        "maximum_nonzeros_per_patch",
        "activation_entropy",
        "effective_atom_count",
        "dead_atom_count",
        "nondead_atom_count",
        "dead_frequency_threshold",
        "maximum_absolute_offdiagonal_coherence",
        "maximum_activation_share",
    )
    return {key: metrics[key] for key in keys}


def _random_dictionary_seed(geometry: str, checkpoint: str, fold_index: int) -> int:
    """Deterministic branch/fold offset: 970301 + branch_index*100 + fold_index.

    ``geometry_index`` is the position in the full frozen GEOMETRIES ordering so
    subset runs keep the same seeds as the full run.
    """
    geometry_index = tuple(GEOMETRIES).index(geometry)
    checkpoint_index = CHECKPOINTS.index(checkpoint)
    branch_index = geometry_index * len(CHECKPOINTS) + checkpoint_index
    return RANDOM_DICTIONARY_SEED_BASE + branch_index * 100 + int(fold_index)


def _run_fold(
    train_examples: Sequence[CoverExample],
    test_examples: Sequence[CoverExample],
    *,
    view: str,
    fold_index: int,
    n_atoms: int = N_ATOMS,
    sparsity: int = SPARSITY,
    iterations: int = 25,
    random_seed: int,
    strata_by_source_index: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Train-only reconstruction for one fold; returns summaries, health, audits."""
    raw_train = stack_cover_examples(train_examples)
    raw_test = stack_cover_examples(test_examples)
    train_mean = np.mean(raw_train, axis=1, keepdims=True)
    centered_train = raw_train - train_mean
    centered_test = raw_test - train_mean

    # RAW: identity reconstruction (cover/stitch ceiling).
    reconstructed: dict[str, np.ndarray | None] = {"RAW": None}

    pca_basis = fit_pca_basis(centered_train, PCA_RANK)
    reconstructed["PCA3"] = pca_basis @ (pca_basis.T @ centered_test) + train_mean

    # RANDOM: 24 train-centered nonzero patch columns, uniform without
    # replacement, normalized; no updates; test coded with T=3 OMP.
    rng = np.random.default_rng(int(random_seed))
    column_norms = np.linalg.norm(centered_train, axis=0)
    nonzero_columns = np.flatnonzero(column_norms > EPS)
    if nonzero_columns.size < n_atoms:
        raise ValueError(
            f"not enough nonzero centered train columns "
            f"({nonzero_columns.size} < {n_atoms})"
        )
    chosen_columns = rng.choice(nonzero_columns, size=n_atoms, replace=False)
    random_dictionary = centered_train[:, chosen_columns].copy()
    random_dictionary = random_dictionary / np.linalg.norm(random_dictionary, axis=0)[None, :]
    random_train_codes = encode_with_minimum_sparsity(
        centered_train, random_dictionary, sparsity=sparsity, minimum_sparsity=T_MIN
    )
    random_codes = encode_with_minimum_sparsity(
        centered_test, random_dictionary, sparsity=sparsity, minimum_sparsity=T_MIN
    )
    reconstructed["RANDOM"] = random_dictionary @ random_codes + train_mean

    # INIT: deterministic maximin on train-centered columns; no updates.
    initial_dictionary, initialization = deterministic_maximin_initialization(
        centered_train, n_atoms
    )
    init_train_codes = encode_with_minimum_sparsity(
        centered_train, initial_dictionary, sparsity=sparsity, minimum_sparsity=T_MIN
    )
    init_codes = encode_with_minimum_sparsity(
        centered_test, initial_dictionary, sparsity=sparsity, minimum_sparsity=T_MIN
    )
    reconstructed["INIT"] = initial_dictionary @ init_codes + train_mean

    # FINAL: ordinary KSVD updates from the same INIT dictionary.
    final_dictionary, _training_codes, training_info = ksvd(
        centered_train,
        n_atoms=n_atoms,
        T=sparsity,
        T_min=T_MIN,
        n_iter=iterations,
        seed=KSVD_SEED,
        initial_dictionary=initial_dictionary,
    )
    final_train_codes = encode_with_minimum_sparsity(
        centered_train, final_dictionary, sparsity=sparsity, minimum_sparsity=T_MIN
    )
    final_codes = encode_with_minimum_sparsity(
        centered_test, final_dictionary, sparsity=sparsity, minimum_sparsity=T_MIN
    )
    reconstructed["FINAL"] = final_dictionary @ final_codes + train_mean

    stages: dict[str, Any] = {}
    keep_rows = {"RAW": True, "FINAL": True}
    stratum_accum: dict[tuple[str, str, str], dict[str, float]] = {}
    for stage in STAGES:
        stages[stage] = {}
        for correction in CORRECTIONS:
            rows, summary = _evaluate(
                test_examples,
                reconstructed[stage],
                residual_corrected=(correction == "residual_corrected"),
            )
            stages[stage][correction] = {"summary": summary}
            if keep_rows.get(stage):
                stages[stage][correction]["rows"] = rows
            for row in rows:
                stratum = (
                    strata_by_source_index.get(int(row["source_index"]), "unknown")
                    if strata_by_source_index
                    else "unknown"
                )
                accumulator = stratum_accum.setdefault(
                    (stratum, stage, correction),
                    {"count": 0.0, "patch_error": 0.0, "observed_rmse": 0.0, "full_rmse": 0.0, "recall": 0.0},
                )
                accumulator["count"] += 1.0
                accumulator["patch_error"] += float(row["patch_relative_error"])
                accumulator["observed_rmse"] += float(row["observed_pair_rmse"])
                accumulator["full_rmse"] += float(row["full_adjacency_rmse"])
                accumulator["recall"] += float(row["full_edge_recall"])
    stratum_rows = [
        {
            "stratum": key[0],
            "stage": key[1],
            "correction": key[2],
            "test_graph_count": int(acc["count"]),
            "patch_relative_error_mean": acc["patch_error"] / max(acc["count"], 1.0),
            "observed_pair_rmse_mean": acc["observed_rmse"] / max(acc["count"], 1.0),
            "full_adjacency_rmse_mean": acc["full_rmse"] / max(acc["count"], 1.0),
            "full_edge_recall_mean": acc["recall"] / max(acc["count"], 1.0),
        }
        for key, acc in sorted(stratum_accum.items())
    ]

    stages["RANDOM"]["dictionary_health"] = {
        "train": _compact_health(
            dictionary_metrics(centered_train, random_dictionary, random_train_codes)
        ),
    }
    stages["RANDOM"]["random_seed"] = int(random_seed)
    stages["RANDOM"]["chosen_train_columns"] = [int(value) for value in chosen_columns]
    stages["INIT"]["dictionary_health"] = {
        "train": _compact_health(
            dictionary_metrics(centered_train, initial_dictionary, init_train_codes)
        ),
    }
    stages["INIT"]["initialization"] = {
        "name": initialization["name"],
        "selection_novelty_scores": initialization["selection_novelty_scores"],
    }
    stages["FINAL"]["dictionary_health"] = {
        "train": _compact_health(
            dictionary_metrics(centered_train, final_dictionary, final_train_codes)
        ),
        "test": _compact_health(
            dictionary_metrics(centered_test, final_dictionary, final_codes)
        ),
    }
    stages["FINAL"]["training"] = {
        "training_iterations": len(training_info.get("recon_curve", [])),
        "recon_curve": [float(value) for value in training_info.get("recon_curve", [])],
        "final_train_recon_rel": training_info.get("recon_rel"),
        "mean_train_nnz": training_info.get("mean_nnz"),
    }

    decomposition_delta = 0.0
    for raw_row, final_unc_row, final_cor_row in zip(
        stages["RAW"]["uncorrected"]["rows"],
        stages["FINAL"]["uncorrected"]["rows"],
        stages["FINAL"]["residual_corrected"]["rows"],
    ):
        decomposition_delta = max(
            decomposition_delta,
            abs(
                final_unc_row["full_adjacency_rmse"] ** 2
                - raw_row["full_adjacency_rmse"] ** 2
                - final_cor_row["full_adjacency_rmse"] ** 2
            ),
        )

    return {
        "view": view,
        "fold_index": int(fold_index),
        "train_graph_count": len(train_examples),
        "test_graph_count": len(test_examples),
        "train_patch_count": int(raw_train.shape[1]),
        "test_patch_count": int(raw_test.shape[1]),
        "train_mean_norm": float(np.linalg.norm(train_mean)),
        "stages": stages,
        "dictionaries": {
            "RANDOM": random_dictionary.tolist(),
            "INIT": initial_dictionary.tolist(),
            "FINAL": final_dictionary.tolist(),
        },
        "decomposition_delta": float(decomposition_delta),
        "stratum_rows": stratum_rows,
    }


def _weighted_stage_mean(
    folds: Sequence[dict[str, Any]], stage: str, correction: str, key: str
) -> float | None:
    folds = [fold for fold in folds if not fold.get("skipped")]
    if not folds:
        return None
    values = np.asarray(
        [float(fold["stages"][stage][correction]["summary"][key]) for fold in folds],
        dtype=np.float64,
    )
    weights = np.asarray([float(fold["test_graph_count"]) for fold in folds], dtype=np.float64)
    return float(np.average(values, weights=weights))


def _mean_stage(
    folds: Sequence[dict[str, Any]], stage: str, correction: str
) -> dict[str, float]:
    folds = [fold for fold in folds if not fold.get("skipped")]
    rows = [fold["stages"][stage][correction]["summary"] for fold in folds]
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in rows[0]}


def _merge_stratum_rows(folds: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str], dict[str, float]] = {}
    for fold in folds:
        for row in fold["stratum_rows"]:
            key = (row["stratum"], row["stage"], row["correction"])
            accumulator = merged.setdefault(
                key, {"count": 0.0, "patch_error": 0.0, "observed_rmse": 0.0, "full_rmse": 0.0, "recall": 0.0}
            )
            count = float(row["test_graph_count"])
            accumulator["count"] += count
            accumulator["patch_error"] += count * float(row["patch_relative_error_mean"])
            accumulator["observed_rmse"] += count * float(row["observed_pair_rmse_mean"])
            accumulator["full_rmse"] += count * float(row["full_adjacency_rmse_mean"])
            accumulator["recall"] += count * float(row["full_edge_recall_mean"])
    return [
        {
            "stratum": key[0],
            "stage": key[1],
            "correction": key[2],
            "test_graph_count": int(acc["count"]),
            "patch_relative_error_mean": acc["patch_error"] / max(acc["count"], 1.0),
            "observed_pair_rmse_mean": acc["observed_rmse"] / max(acc["count"], 1.0),
            "full_adjacency_rmse_mean": acc["full_rmse"] / max(acc["count"], 1.0),
            "full_edge_recall_mean": acc["recall"] / max(acc["count"], 1.0),
        }
        for key, acc in sorted(merged.items())
    ]


# ---------------------------------------------------------------------------
# Patch-vector overlap and patch source-node exposure audits
# ---------------------------------------------------------------------------


def _vector_keys(examples: Sequence[CoverExample]) -> list[bytes]:
    """Exact rooted-canonical adjacency bit-vectors; no source identities."""
    keys = []
    for example in examples:
        vectors = np.ascontiguousarray(
            np.asarray(example.patch_vectors, dtype=np.int8)
        )
        for row in vectors:
            keys.append(row.tobytes())
    return keys


def _overlap_counts(train_keys: Sequence[bytes], test_keys: Sequence[bytes]) -> dict[str, float | int]:
    train_counts = Counter(train_keys)
    train_unique = set(train_counts)
    test_unique = set(test_keys)
    exact_seen = sum(1 for key in test_keys if key in train_counts)
    return {
        "occurrence_fraction": float(exact_seen / max(len(test_keys), 1)),
        "unique_overlap": float(
            len(train_unique & test_unique) / max(len(train_unique | test_unique), 1)
        ),
        "exact_seen_test_vector_count": int(exact_seen),
        "train_unique_vector_count": len(train_unique),
        "test_unique_vector_count": len(test_unique),
    }


def _covered_source_nodes(
    bundle: Any, source_index: int, ids: StableNodeIDs, slots: np.ndarray
) -> set[int]:
    """Map stable patch slots back to source nodes via ``ids.order``.

    Stable slot ``k`` corresponds to source row position ``ids.order[k]``;
    source identities are used only for the exposure audit, never in vectors.
    """
    order = np.asarray(ids.order, dtype=np.int64)
    return {
        int(bundle.global_node_ids[source_index, int(order[int(slot)])])
        for slot in slots
    }


def _patch_overlap_fold(
    bundle: Any, prepared_by_index: dict[int, PreparedGraph], fold: MentorFold
) -> dict[str, Any]:
    train_indices = [
        int(index) for index in fold.train_indices if int(index) in prepared_by_index
    ]
    test_indices = [
        int(index) for index in fold.test_indices if int(index) in prepared_by_index
    ]
    if not train_indices or not test_indices:
        return {"fold_index": int(fold.fold_index), "skipped": True}
    train_keys = _vector_keys([prepared_by_index[index].example for index in train_indices])
    test_keys = _vector_keys([prepared_by_index[index].example for index in test_indices])
    counts = _overlap_counts(train_keys, test_keys)
    train_nodes = set().union(
        *(set(int(value) for value in bundle.global_node_ids[index]) for index in train_indices)
    )
    train_roots = {int(bundle.roots[index]) for index in train_indices}
    node_seen_fractions = []
    root_covered_flags = []
    covered_root_seen_flags = []
    for index in test_indices:
        graph = prepared_by_index[index]
        slots = np.unique(
            np.concatenate(
                [
                    np.asarray(patch.node_ids, dtype=np.int64)
                    for patch in graph.example.cover.patches
                ]
            )
        )
        covered_sources = _covered_source_nodes(bundle, index, graph.ids, slots)
        node_seen_fractions.append(
            len(covered_sources & train_nodes) / max(len(covered_sources), 1)
        )
        order = np.asarray(graph.ids.order, dtype=np.int64)
        root_slot = int(np.flatnonzero(order == 0)[0])  # roots == global_node_ids[:, 0]
        root_covered = bool(root_slot in set(int(slot) for slot in slots))
        root_covered_flags.append(root_covered)
        covered_root_seen_flags.append(
            bool(root_covered and int(bundle.roots[index]) in train_roots)
        )
    return {
        "fold_index": int(fold.fold_index),
        "train_graph_count": len(train_indices),
        "test_graph_count": len(test_indices),
        "train_patch_count": len(train_keys),
        "test_patch_count": len(test_keys),
        "skipped": False,
        **counts,
        "patch_covered_node_seen_fraction": float(np.mean(node_seen_fractions)),
        "root_covered_fraction": float(np.mean(root_covered_flags)),
        "covered_root_seen_fraction": float(np.mean(covered_root_seen_flags)),
    }


def _patch_overlap_audit(
    bundle: Any,
    folds_by_view: dict[str, tuple[MentorFold, ...]],
    prepared_by_branch: dict[tuple[str, str], dict[int, PreparedGraph]],
    *,
    views: Sequence[str],
    geometries: Sequence[str],
    checkpoints: Sequence[str],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for view in views:
        output[view] = {}
        for geometry in geometries:
            for checkpoint in checkpoints:
                branch = f"{geometry}_{checkpoint}"
                prepared = prepared_by_branch[(geometry, checkpoint)]
                fold_rows = [
                    _patch_overlap_fold(bundle, prepared, fold)
                    for fold in folds_by_view[view]
                ]
                valid = [row for row in fold_rows if not row.get("skipped")]
                output[view][branch] = {
                    "fold_rows": fold_rows,
                    "weighted_occurrence_fraction": (
                        float(
                            np.average(
                                [float(row["occurrence_fraction"]) for row in valid],
                                weights=[float(row["test_patch_count"]) for row in valid],
                            )
                        )
                        if valid
                        else None
                    ),
                    "mean_unique_overlap": (
                        float(np.mean([float(row["unique_overlap"]) for row in valid]))
                        if valid
                        else None
                    ),
                    "mean_patch_covered_node_seen_fraction": (
                        float(
                            np.mean(
                                [
                                    float(row["patch_covered_node_seen_fraction"])
                                    for row in valid
                                ]
                            )
                        )
                        if valid
                        else None
                    ),
                    "mean_root_covered_fraction": (
                        float(np.mean([float(row["root_covered_fraction"]) for row in valid]))
                        if valid
                        else None
                    ),
                    "mean_covered_root_seen_fraction": (
                        float(
                            np.mean(
                                [float(row["covered_root_seen_fraction"]) for row in valid]
                            )
                        )
                        if valid
                        else None
                    ),
                }
    return output


# ---------------------------------------------------------------------------
# Frozen gates (protocol sections 7.1-7.3) and classification (section 8)
# ---------------------------------------------------------------------------


def _split_data_checks(
    branches: dict[str, Any],
    exposure_rows: Sequence[dict[str, Any]],
    *,
    views: Sequence[str],
    n_folds: int,
) -> dict[str, Any]:
    per_view: dict[str, Any] = {}
    for view in views:
        view_exposure = [row for row in exposure_rows if row["view"] == view]
        exposure_values: list[float] = []
        for row in view_exposure:
            for key in (
                "test_source_node_seen_fraction",
                "source_node_jaccard",
                "test_source_edge_seen_fraction",
                "source_edge_jaccard",
            ):
                exposure_values.append(float(row[key]))
            for distribution_key in (
                "test_graph_source_node_exposure",
                "test_graph_source_edge_exposure",
            ):
                exposure_values.extend(
                    float(value) for value in row[distribution_key].values()
                )
        array = np.asarray(exposure_values, dtype=np.float64)
        checks: dict[str, Any] = {
            "exposure_finite_in_unit_interval": bool(
                array.size > 0
                and np.all(np.isfinite(array))
                and np.all((array >= 0.0) & (array <= 1.0))
            ),
            "fold_count_matches": len(view_exposure) == n_folds,
        }
        if view == "root_candidate_grouped":
            checks["root_intersection_zero"] = all(
                int(row["root_intersection_count"]) == 0 for row in view_exposure
            )
        raw_checks: list[bool] = []
        for branch_rows in branches[view].values():
            for fold in branch_rows["fold_rows"]:
                if fold.get("skipped"):
                    continue
                raw = fold["stages"]["RAW"]["uncorrected"]["summary"]
                raw_checks.append(
                    raw["patch_relative_error"] == 0.0
                    and raw["observed_pair_rmse"] == 0.0
                    and raw["repeated_pair_disagreement_std_mean"] == 0.0
                    and raw["repeated_pair_disagreement_range_mean"] == 0.0
                    and fold["decomposition_delta"] <= DECOMPOSITION_TOLERANCE
                )
        checks["raw_identity_and_decomposition"] = bool(raw_checks) and all(raw_checks)
        per_view[view] = {"checks": checks, "passed": all(checks.values())}
    return {
        "per_view": per_view,
        "passed": bool(per_view) and all(value["passed"] for value in per_view.values()),
    }


def _optimization_gate(
    branches: dict[str, Any],
    branch_name: str,
    *,
    n_folds: int,
    view: str = "root_candidate_grouped",
) -> dict[str, Any]:
    folds = [
        fold
        for fold in branches[view][branch_name]["fold_rows"]
        if not fold.get("skipped")
    ]
    if len(folds) != n_folds:
        return {
            "passed": False,
            "view": view,
            "available_fold_count": len(folds),
            "reason": "not all folds available",
        }
    init_error = np.asarray(
        [float(fold["stages"]["INIT"]["uncorrected"]["summary"]["patch_relative_error"]) for fold in folds],
        dtype=np.float64,
    )
    final_error = np.asarray(
        [float(fold["stages"]["FINAL"]["uncorrected"]["summary"]["patch_relative_error"]) for fold in folds],
        dtype=np.float64,
    )
    weights = np.asarray([float(fold["test_graph_count"]) for fold in folds], dtype=np.float64)
    reductions = (init_error - final_error) / np.maximum(init_error, EPS)
    weighted_reduction = float(np.average(reductions, weights=weights))
    init_observed = [
        float(fold["stages"]["INIT"]["uncorrected"]["summary"]["observed_pair_rmse"])
        for fold in folds
    ]
    final_observed = [
        float(fold["stages"]["FINAL"]["uncorrected"]["summary"]["observed_pair_rmse"])
        for fold in folds
    ]
    nondead = [
        int(fold["stages"]["FINAL"]["dictionary_health"]["train"]["nondead_atom_count"])
        for fold in folds
    ]
    max_share = [
        float(fold["stages"]["FINAL"]["dictionary_health"]["train"]["maximum_activation_share"])
        for fold in folds
    ]
    positive_count = int(np.count_nonzero(reductions > 0.0))
    checks: dict[str, Any] = {
        "final_better_than_init_all_folds": bool(np.all(final_error < init_error)),
        "weighted_reduction_at_least_005": bool(weighted_reduction >= 0.05),
        "final_nondead_atoms_ge20_all_folds": all(value >= 20 for value in nondead),
        "final_max_activation_share_lt050_all_folds": all(value < 0.5 for value in max_share),
        "final_observed_rmse_not_worse_all_folds": all(
            final_value <= init_value + EPS
            for init_value, final_value in zip(init_observed, final_observed)
        ),
        "positive_reduction_fold_count": positive_count,
        "positive_reduction_folds_at_least_n_minus_1": bool(
            positive_count >= max(n_folds - 1, 1)
        ),
        "weighted_mean_reduction_positive": bool(weighted_reduction > 0.0),
    }
    if "random_reference" in branches:
        random_folds = [
            fold
            for fold in branches["random_reference"][branch_name]["fold_rows"]
            if not fold.get("skipped")
        ]
        if len(random_folds) == n_folds:
            random_init = np.asarray(
                [float(fold["stages"]["INIT"]["uncorrected"]["summary"]["patch_relative_error"]) for fold in random_folds],
                dtype=np.float64,
            )
            random_final = np.asarray(
                [float(fold["stages"]["FINAL"]["uncorrected"]["summary"]["patch_relative_error"]) for fold in random_folds],
                dtype=np.float64,
            )
            random_weights = np.asarray(
                [float(fold["test_graph_count"]) for fold in random_folds], dtype=np.float64
            )
            random_reductions = (random_init - random_final) / np.maximum(random_init, EPS)
            checks["random_reference_weighted_reduction"] = float(
                np.average(random_reductions, weights=random_weights)
            )
            checks["root_not_systematically_reversed"] = bool(
                positive_count >= max(n_folds - 1, 1)
                and weighted_reduction > 0.0
                and checks["random_reference_weighted_reduction"] > 0.0
            )
        else:
            checks["random_reference_weighted_reduction"] = None
            checks["root_not_systematically_reversed"] = None
    else:
        checks["random_reference_weighted_reduction"] = None
        checks["root_not_systematically_reversed"] = None
    required = (
        "final_better_than_init_all_folds",
        "weighted_reduction_at_least_005",
        "final_nondead_atoms_ge20_all_folds",
        "final_max_activation_share_lt050_all_folds",
        "final_observed_rmse_not_worse_all_folds",
        "positive_reduction_folds_at_least_n_minus_1",
        "weighted_mean_reduction_positive",
    )
    required_values = [checks[key] for key in required]
    if checks["root_not_systematically_reversed"] is not None:
        required_values.append(checks["root_not_systematically_reversed"])
    checks["passed"] = bool(all(required_values))
    return {
        "passed": checks["passed"],
        "view": view,
        "weighted_reduction": weighted_reduction,
        "fold_reductions": [float(value) for value in reductions],
        "positive_reduction_fold_count": positive_count,
        "nondead_atom_counts": nondead,
        "maximum_activation_shares": max_share,
        "checks": checks,
    }


def _control_gate(
    branches: dict[str, Any],
    branch_name: str,
    *,
    view: str = "root_candidate_grouped",
) -> dict[str, Any]:
    folds = [
        fold
        for fold in branches[view][branch_name]["fold_rows"]
        if not fold.get("skipped")
    ]
    final_error = _weighted_stage_mean(folds, "FINAL", "uncorrected", "patch_relative_error")
    random_error = _weighted_stage_mean(folds, "RANDOM", "uncorrected", "patch_relative_error")
    pca3_error = _weighted_stage_mean(folds, "PCA3", "uncorrected", "patch_relative_error")
    checks = {
        "final_beats_random": bool(final_error < random_error),
        "final_beats_pca3": bool(final_error < pca3_error),
        "final_patch_error": final_error,
        "random_patch_error": random_error,
        "pca3_patch_error": pca3_error,
    }
    checks["passed"] = bool(checks["final_beats_random"] and checks["final_beats_pca3"])
    return checks


def _candidate_row(branch_payload: dict[str, Any], branch_name: str) -> dict[str, Any]:
    final = branch_payload["weighted_mean_stages"]["FINAL"]
    return {
        "candidate": branch_name,
        "full_adjacency_rmse": final["uncorrected"]["full_adjacency_rmse"],
        "corrected_full_rmse": final["residual_corrected"]["full_adjacency_rmse"],
        "full_edge_recall": final["uncorrected"]["full_edge_recall"],
        "dictionary_scalars": branch_payload["dictionary_scalars"],
        "code_scalars_per_graph": branch_payload["code_scalars_per_graph"],
    }


def _pareto_names(rows: Sequence[dict[str, Any]]) -> list[str]:
    minimization = (
        "full_adjacency_rmse",
        "corrected_full_rmse",
        "dictionary_scalars",
        "code_scalars_per_graph",
    )

    def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
        not_worse = all(left[key] <= right[key] for key in minimization)
        not_worse = not_worse and left["full_edge_recall"] >= right["full_edge_recall"]
        strict = any(left[key] < right[key] for key in minimization)
        strict = strict or left["full_edge_recall"] > right["full_edge_recall"]
        return bool(not_worse and strict)

    return [
        str(row["candidate"])
        for row in rows
        if not any(other is not row and dominates(other, row) for other in rows)
    ]


def classify(
    branches: dict[str, Any],
    exposure_rows: Sequence[dict[str, Any]],
    *,
    views: Sequence[str],
    geometries: Sequence[str],
    checkpoints: Sequence[str],
    n_folds: int,
) -> dict[str, Any]:
    split_data = _split_data_checks(branches, exposure_rows, views=views, n_folds=n_folds)
    optimization: dict[str, Any] = {}
    control: dict[str, Any] = {}
    if "root_candidate_grouped" in branches:
        for geometry in geometries:
            for checkpoint in checkpoints:
                branch = f"{geometry}_{checkpoint}"
                if branch not in branches["root_candidate_grouped"]:
                    continue
                optimization[branch] = _optimization_gate(
                    branches, branch, n_folds=n_folds
                )
                if optimization[branch]["passed"]:
                    control[branch] = _control_gate(branches, branch)
    passed_optimization = [branch for branch, gate in optimization.items() if gate["passed"]]
    passed_control = [
        branch for branch in passed_optimization if control.get(branch, {}).get("passed", False)
    ]
    if not split_data["passed"]:
        classification = "FAIL_MENTOR_GROUPED_RECONSTRUCTION_CONTRACT"
    elif not passed_optimization:
        classification = "KEEP_RAW_PATCH_OR_INIT_BASELINE_NO_KSVD_GAIN"
    elif not passed_control:
        classification = "KSVD_OPTIMIZES_BUT_NO_RATE_DISTORTION_ADVANTAGE"
    else:
        classification = "KSVD_GROUPED_RECONSTRUCTION_READY_FOR_RELATION_TOKEN_ABLATION"
    candidate_rows = {
        branch: _candidate_row(branches["root_candidate_grouped"][branch], branch)
        for branch in passed_control
    }
    return {
        "classification": classification,
        "split_data_gate": split_data,
        "optimization_gate": {
            "per_branch": optimization,
            "passed_branches": passed_optimization,
        },
        "control_gate": {
            "per_branch": control,
            "passed_branches": passed_control,
        },
        "ready_branches": passed_control,
        "candidate_rows": candidate_rows,
        "pareto": _pareto_names(list(candidate_rows.values())),
        "views_complete": len(views) == len(VIEWS),
    }


# ---------------------------------------------------------------------------
# Pilot checkpoint-count verification
# ---------------------------------------------------------------------------


def _verify_pilot_counts(
    args: argparse.Namespace,
    pilot_payload: dict[str, Any] | None,
    selection_indices: np.ndarray,
    prepared_by_branch: dict[tuple[str, str], dict[int, PreparedGraph]],
    coverage_missing: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if pilot_payload is None:
        return {"checked": False, "reason": "pilot json not found"}
    config = pilot_payload.get("config", {})
    if (
        int(args.pilot_size) != int(config.get("pilot_size"))
        or int(args.pilot_seed) != int(config.get("pilot_seed"))
        or int(args.cover_seed) != int(config.get("cover_seed"))
    ):
        return {
            "checked": False,
            "reason": "selection does not match the pilot config",
        }
    pilot_indices = np.sort(
        np.asarray(pilot_payload.get("pilot_indices", []), dtype=np.int64)
    )
    indices_match = bool(
        pilot_indices.size == selection_indices.size
        and np.array_equal(pilot_indices, np.sort(np.asarray(selection_indices, dtype=np.int64)))
    )
    rows_by_key = {
        (int(row["source_graph_index"]), row["geometry"], row["checkpoint"]): row
        for row in pilot_payload.get("checkpoint_rows", [])
    }
    mismatches = []
    compared = 0
    for (geometry, checkpoint), prepared in prepared_by_branch.items():
        for source_index, graph in prepared.items():
            row = rows_by_key.get((source_index, geometry, checkpoint))
            if row is None:
                continue
            compared += 1
            if int(row["patch_count"]) != int(graph.patch_count):
                mismatches.append(
                    {
                        "source_index": source_index,
                        "geometry": geometry,
                        "checkpoint": checkpoint,
                        "pilot_patch_count": int(row["patch_count"]),
                        "regenerated_patch_count": int(graph.patch_count),
                    }
                )
    missing_but_in_pilot = [
        {
            "source_index": int(item["source_index"]),
            "geometry": item["geometry"],
            "checkpoint": item["checkpoint"],
            "reason": item.get("reason"),
        }
        for item in coverage_missing
        if (int(item["source_index"]), item["geometry"], item["checkpoint"]) in rows_by_key
    ]
    return {
        "checked": True,
        "pilot_indices_match": bool(indices_match),
        "compared": compared,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "missing_but_in_pilot_count": len(missing_but_in_pilot),
        "missing_but_in_pilot": missing_but_in_pilot,
        "passed": bool(indices_match) and not mismatches and not missing_but_in_pilot,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def render_report(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    verification = payload["verification"]
    lines = [
        "# 导师 50-node 真实子图：Grouped-split KSVD reconstruction follow-up",
        "",
        f"> 日期：2026-08-06",
        f"> 协议：`{payload['protocol']}`",
        f"> 判定：`{decision['classification']}`",
        "",
        "## 1. 数据与 selection",
        "",
        f"- source SHA-256：`{payload['data']['source_sha256']}`；",
        f"- cache graphs：`{payload['data']['cache_graph_count']}`；pilot：`{payload['selection']['size']}`；",
        f"- pilot roots：`{payload['selection']['n_distinct_roots']}`；density quotas：`{payload['selection']['selected_counts']}`；",
        f"- labels used：`False`；source IDs in patch vectors：`False`；source IDs in dictionaries：`False`。",
        f"- splits：`{payload['config']['n_splits']}`-fold，seed `{payload['config']['split_seed']}`；views：`{payload['config']['views']}`。",
        "",
        f"- pilot 冻结验证：checked=`{verification['checked']}`；passed=`{verification.get('passed')}`；",
    ]
    if verification.get("checked"):
        lines.append(
            f"  pilot_indices_match=`{verification['pilot_indices_match']}`，"
            f"compared=`{verification['compared']}`，mismatches=`{verification['mismatch_count']}`，"
            f"missing_but_in_pilot=`{verification.get('missing_but_in_pilot_count')}`。"
        )
    else:
        lines.append(f"  原因：`{verification.get('reason')}`。")
    lines.extend([
        "",
        "## 2. 总判定",
        "",
        f"- split/data gate：`{decision['split_data_gate']['passed']}`；",
        f"- optimization gate passed branches：`{decision['optimization_gate']['passed_branches']}`；",
        f"- control/Pareto passed branches：`{decision['control_gate']['passed_branches']}`；",
        f"- ready branches：`{decision['ready_branches']}`；Pareto：`{decision['pareto']}`。",
        "",
        "## 3. Split views 与 source exposure audit",
        "",
        "| view | fold | train/test | roots train/test/∩ | node seen | node Jaccard | edge seen | edge Jaccard | node exposure mean/p10/min | edge exposure mean/p10/min |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in payload["exposure_rows"]:
        node_dist = row["test_graph_source_node_exposure"]
        edge_dist = row["test_graph_source_edge_exposure"]
        lines.append(
            f"| {row['view']} | {row['fold_index']} | {row['train_graph_count']}/{row['test_graph_count']} | "
            f"{row['train_distinct_root_count']}/{row['test_distinct_root_count']}/{row['root_intersection_count']} | "
            f"{_fmt(row['test_source_node_seen_fraction'])} | {_fmt(row['source_node_jaccard'])} | "
            f"{_fmt(row['test_source_edge_seen_fraction'])} | {_fmt(row['source_edge_jaccard'])} | "
            f"{_fmt(node_dist['mean'])}/{_fmt(node_dist['p10'])}/{_fmt(node_dist['minimum'])} | "
            f"{_fmt(edge_dist['mean'])}/{_fmt(edge_dist['p10'])}/{_fmt(edge_dist['minimum'])} |"
        )
    lines.extend([
        "",
        "## 4. Exact canonical patch-vector overlap 与 patch source-node exposure",
        "",
        "Patch vectors 是 rooted-canonical adjacency bit vectors（不含 source IDs）；occurrence 是 test patch vectors 在 train 中 exact-seen 的比例（按 multiplicity），unique overlap 是 unique-vector Jaccard。patch source-node exposure 把 patch 覆盖的 stable slots 经 `ids.order` 映回 source nodes 后与 train source nodes 比较。",
        "",
        "| view | geometry | checkpoint | occurrence | unique overlap | covered-node seen | root covered | covered-root seen |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for view, branches in payload["patch_overlap"].items():
        for branch, row in branches.items():
            geometry, checkpoint = branch.rsplit("_", 1)
            lines.append(
                f"| {view} | {geometry} | {checkpoint} | {_fmt(row['weighted_occurrence_fraction'])} | "
                f"{_fmt(row['mean_unique_overlap'])} | {_fmt(row['mean_patch_covered_node_seen_fraction'])} | "
                f"{_fmt(row['mean_root_covered_fraction'])} | {_fmt(row['mean_covered_root_seen_fraction'])} |"
            )
    lines.extend([
        "",
        "## 5. Reconstruction（test-count-weighted means）",
        "",
        "| view | branch | stage | patch err | obs RMSE | obs F1 | full RMSE | full acc | full recall | full F1 | corrected RMSE | corrected recall |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for view, branches in payload["branches"].items():
        for branch_name, branch in branches.items():
            if not branch["weighted_mean_stages"]:
                continue
            for stage in STAGES:
                uncorrected = branch["weighted_mean_stages"][stage]["uncorrected"]
                corrected = branch["weighted_mean_stages"][stage]["residual_corrected"]
                lines.append(
                    f"| {view} | {branch_name} | {stage} | {_fmt(uncorrected['patch_relative_error'])} | "
                    f"{_fmt(uncorrected['observed_pair_rmse'])} | {_fmt(uncorrected['observed_edge_f1'])} | "
                    f"{_fmt(uncorrected['full_adjacency_rmse'])} | {_fmt(uncorrected['full_adjacency_accuracy'])} | "
                    f"{_fmt(uncorrected['full_edge_recall'])} | {_fmt(uncorrected['full_edge_f1'])} | "
                    f"{_fmt(corrected['full_adjacency_rmse'])} | {_fmt(corrected['full_edge_recall'])} |"
                )
    lines.extend([
        "",
        "## 6. Dictionary health 与 scalars",
        "",
        "health 基于 train codes（字典在其上训练）；FINAL 同时记录 test codes。",
        "",
        "| view | branch | stage | dict scalars | code scalars/graph | nondead | effective | max share | coherence | train rel err |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for view, branches in payload["branches"].items():
        for branch_name, branch in branches.items():
            for stage in ("RANDOM", "INIT", "FINAL"):
                if not branch["weighted_mean_stages"]:
                    continue
                health = None
                for fold in branch["fold_rows"]:
                    if fold.get("skipped"):
                        continue
                    health = fold["stages"][stage].get("dictionary_health")
                    break
                if health is None:
                    continue
                train_health = health["train"]
                lines.append(
                    f"| {view} | {branch_name} | {stage} | {branch['dictionary_scalars']} | "
                    f"{_fmt(branch['code_scalars_per_graph'], 1)} | {train_health['nondead_atom_count']} | "
                    f"{_fmt(train_health['effective_atom_count'], 2)} | {_fmt(train_health['maximum_activation_share'])} | "
                    f"{_fmt(train_health['maximum_absolute_offdiagonal_coherence'])} | "
                    f"{_fmt(train_health['relative_reconstruction_error'])} |"
                )
    lines.extend([
        "",
        "## 7. 冻结 gates",
        "",
        "### 7.1 Split/data gate",
        "",
    ])
    for view, row in decision["split_data_gate"]["per_view"].items():
        lines.append(
            f"- `{view}`：passed=`{row['passed']}`；checks=`{row['checks']}`"
        )
    lines.extend([
        "",
        "### 7.2 KSVD optimization gate（root-grouped 为主）",
        "",
        "| branch | FINAL<INIT 3/3 | weighted reduction | ≥0.05 | nondead≥20 | max share<0.5 | obs RMSE not worse | positive folds | random-view reduction | pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for branch, gate in decision["optimization_gate"]["per_branch"].items():
        checks = gate.get("checks", {})
        lines.append(
            f"| {branch} | {checks.get('final_better_than_init_all_folds', '—')} | "
            f"{_fmt(gate.get('weighted_reduction'))} | {checks.get('weighted_reduction_at_least_005', '—')} | "
            f"{checks.get('final_nondead_atoms_ge20_all_folds', '—')} | "
            f"{checks.get('final_max_activation_share_lt050_all_folds', '—')} | "
            f"{checks.get('final_observed_rmse_not_worse_all_folds', '—')} | "
            f"{checks.get('positive_reduction_fold_count', '—')} | "
            f"{_fmt(checks.get('random_reference_weighted_reduction'))} | {gate['passed']} |"
        )
    lines.extend([
        "",
        "### 7.3 Control / Pareto gate（root-grouped weighted mean patch error）",
        "",
        "| branch | FINAL err | RANDOM err | PCA3 err | beats RANDOM | beats PCA3 | pass | Pareto |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    pareto = set(decision["pareto"])
    for branch, gate in decision["control_gate"]["per_branch"].items():
        lines.append(
            f"| {branch} | {_fmt(gate['final_patch_error'])} | {_fmt(gate['random_patch_error'])} | "
            f"{_fmt(gate['pca3_patch_error'])} | {gate['final_beats_random']} | {gate['final_beats_pca3']} | "
            f"{gate['passed']} | {branch in pareto} |"
        )
    lines.extend([
        "",
        "## 8. Density-stratum summaries（per view × branch）",
        "",
        "| view | branch | stratum | graphs | RAW err | PCA3 | RANDOM | INIT | FINAL | FINAL corr | obs RMSE | full recall |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for view, branches in payload["branches"].items():
        for branch_name, branch in branches.items():
            rows_by_stratum: dict[str, dict[str, Any]] = {}
            for row in branch["stratum_rows"]:
                rows_by_stratum.setdefault(row["stratum"], {})[(row["stage"], row["correction"])] = row
            for stratum in STRATUM_NAMES:
                if stratum not in rows_by_stratum:
                    continue
                rows = rows_by_stratum[stratum]
                uncorrected = {stage: rows.get((stage, "uncorrected")) for stage in STAGES}
                corrected = rows.get(("FINAL", "residual_corrected"))
                count = next(
                    (row["test_graph_count"] for row in rows.values() if row["test_graph_count"]),
                    None,
                )
                final = uncorrected["FINAL"]
                lines.append(
                    f"| {view} | {branch_name} | {stratum} | {count} | "
                    f"{_fmt(uncorrected['RAW']['patch_relative_error_mean'] if uncorrected['RAW'] else None)} | "
                    f"{_fmt(uncorrected['PCA3']['patch_relative_error_mean'] if uncorrected['PCA3'] else None)} | "
                    f"{_fmt(uncorrected['RANDOM']['patch_relative_error_mean'] if uncorrected['RANDOM'] else None)} | "
                    f"{_fmt(uncorrected['INIT']['patch_relative_error_mean'] if uncorrected['INIT'] else None)} | "
                    f"{_fmt(final['patch_relative_error_mean'] if final else None)} | "
                    f"{_fmt(corrected['patch_relative_error_mean'] if corrected else None)} | "
                    f"{_fmt(final['observed_pair_rmse_mean'] if final else None)} | "
                    f"{_fmt(final['full_edge_recall_mean'] if final else None)} |"
                )
    lines.extend([
        "",
        "## 9. 边界",
        "",
        "- density strata 与 root_candidate 只用于 split/audit，不进入 patch vectors；source IDs 只用于 exposure audit 的 bookkeeping。",
        "- root-grouped view 是 root-candidate-grouped；`root_candidate` 尚未由生成方确认是真实 sampling root。",
        "- patch-vector overlap 是结构模板重复（canonical adjacency bit vectors），不是 source identity 泄漏。",
        "- residual-corrected 只隔离 patch compression error；residual 的 bit 成本不计入 codec，也不参与 dictionary training。",
        "- 本协议只判定 KSVD 是否值得进入 relation-token ablation；不自动证明语义 motif，不自动开始 Transformer 调参。",
        "",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Grouped-split KSVD reconstruction follow-up on the mentor subgraph bank."
    )
    parser.add_argument("--source", type=Path, default=default_source_path())
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--pilot-size", type=int, default=500)
    parser.add_argument("--pilot-seed", type=int, default=20260806)
    parser.add_argument("--cover-seed", type=int, default=970201)
    parser.add_argument("--maximum-patches", type=int, default=60)
    parser.add_argument("--multiplier", type=float, default=1.5)
    parser.add_argument("--retained-beam", type=int, default=8)
    parser.add_argument("--candidate-restarts", type=int, default=1)
    parser.add_argument("--geometries", nargs="+", choices=tuple(GEOMETRIES), default=list(GEOMETRIES))
    parser.add_argument("--checkpoints", nargs="+", choices=tuple(CHECKPOINTS), default=list(CHECKPOINTS))
    parser.add_argument("--views", nargs="+", choices=tuple(VIEWS), default=list(VIEWS))
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--split-seed", type=int, default=20260807)
    parser.add_argument("--pilot-json", type=Path, default=DEFAULT_PILOT_JSON)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    cache_path = args.cache or default_cache_path(args.source)
    bundle = load_bundle(source_pkl=args.source, cache_path=cache_path, force=args.force_cache)
    selection = select_pilot_indices(bundle, n_pilot=args.pilot_size, seed=args.pilot_seed)
    pilot_indices = selection.indices
    all_strata = density_stratum(bundle.avg_degrees)
    strata_values = all_strata[pilot_indices]
    root_values = bundle.roots[pilot_indices]
    geometries = tuple(args.geometries)
    checkpoints = tuple(args.checkpoints)
    views = tuple(args.views)

    prepared_by_branch: dict[tuple[str, str], dict[int, PreparedGraph]] = {
        (geometry, checkpoint): {}
        for geometry in geometries
        for checkpoint in checkpoints
    }
    coverage_missing: list[dict[str, Any]] = []
    for position, raw_source_index in enumerate(pilot_indices):
        source_index = int(raw_source_index)
        source_adjacency = bundle.adjacency[source_index].astype(np.int8, copy=False)
        ids = compute_global_wl_ids(source_adjacency)
        stable_adjacency = reorder_by_stable_ids(source_adjacency, ids)
        stratum = STRATUM_NAMES[int(all_strata[source_index])]
        for geometry in geometries:
            patch_size, overlap = GEOMETRIES[geometry]
            geometry_index = tuple(GEOMETRIES).index(geometry)
            base_budget = patch_budget(
                stable_adjacency,
                patch_size=patch_size,
                target_overlap=overlap,
                edge_capacity_multiplier=args.multiplier,
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
                    retained_beam=args.retained_beam,
                    candidate_restarts=args.candidate_restarts,
                    allow_partial=True,
                )
            except Exception as exc:  # noqa: BLE001 - recorded per graph
                for checkpoint in checkpoints:
                    coverage_missing.append(
                        {
                            "source_index": source_index,
                            "geometry": geometry,
                            "checkpoint": checkpoint,
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
            selected = select_operating_checkpoints(trajectory, base_patch_count=base_budget)
            for checkpoint in checkpoints:
                row = selected[checkpoint]
                if row is None:
                    coverage_missing.append(
                        {
                            "source_index": source_index,
                            "geometry": geometry,
                            "checkpoint": checkpoint,
                            "reason": "checkpoint unreachable",
                        }
                    )
                    continue
                prefix = _prefix_cover(
                    stable_cover, int(row["patch_count"]), f"{geometry}_{checkpoint}"
                )
                ordered, _diagnostics = reorder_cover_structurally(
                    stable_adjacency, prefix, "rooted_canonical"
                )
                example = make_cover_example(
                    source_index,
                    stratum,
                    int(round(float(bundle.avg_degrees[source_index]))),
                    stable_adjacency,
                    ordered,
                )
                prepared_by_branch[(geometry, checkpoint)][source_index] = PreparedGraph(
                    source_index=source_index,
                    pilot_position=position,
                    density_stratum=stratum,
                    average_degree=float(bundle.avg_degrees[source_index]),
                    geometry=geometry,
                    checkpoint=checkpoint,
                    patch_count=int(row["patch_count"]),
                    ids=ids,
                    example=example,
                )
        if (position + 1) % 25 == 0 or position + 1 == len(pilot_indices):
            print(f"prepared={position + 1}/{len(pilot_indices)}", flush=True)

    folds_by_view = make_mentor_folds(
        pilot_indices,
        strata_values,
        root_values,
        n_splits=args.n_splits,
        seed=args.split_seed,
    )

    exposure_rows: list[dict[str, Any]] = []
    for view in views:
        for fold in folds_by_view[view]:
            exposure_rows.append(audit_source_exposure(bundle, fold, all_strata))

    branches: dict[str, Any] = {view: {} for view in views}
    for view in views:
        folds = folds_by_view[view]
        for geometry in geometries:
            for checkpoint in checkpoints:
                branch = f"{geometry}_{checkpoint}"
                prepared = prepared_by_branch[(geometry, checkpoint)]
                fold_rows = []
                for fold in folds:
                    train_examples = [
                        prepared[int(index)].example
                        for index in fold.train_indices
                        if int(index) in prepared
                    ]
                    test_examples = [
                        prepared[int(index)].example
                        for index in fold.test_indices
                        if int(index) in prepared
                    ]
                    if not train_examples or not test_examples:
                        fold_rows.append(
                            {
                                "view": view,
                                "fold_index": int(fold.fold_index),
                                "skipped": True,
                                "train_graph_count": len(train_examples),
                                "test_graph_count": len(test_examples),
                            }
                        )
                        continue
                    random_seed = _random_dictionary_seed(geometry, checkpoint, fold.fold_index)
                    fold_rows.append(
                        _run_fold(
                            train_examples,
                            test_examples,
                            view=view,
                            fold_index=fold.fold_index,
                            iterations=args.iterations,
                            random_seed=random_seed,
                            strata_by_source_index={
                                int(index): prepared[int(index)].density_stratum
                                for index in fold.test_indices
                                if int(index) in prepared
                            },
                        )
                    )
                valid_folds = [fold for fold in fold_rows if not fold.get("skipped")]
                if valid_folds:
                    weighted_mean_stages = {
                        stage: {
                            correction: {
                                key: _weighted_stage_mean(valid_folds, stage, correction, key)
                                for key in STITCH_METRICS
                            }
                            for correction in CORRECTIONS
                        }
                        for stage in STAGES
                    }
                    graph_balanced_mean_stages = {
                        stage: {
                            correction: _mean_stage(valid_folds, stage, correction)
                            for correction in CORRECTIONS
                        }
                        for stage in STAGES
                    }
                    code_scalars = float(
                        sum(float(fold["test_patch_count"]) for fold in valid_folds)
                        / max(sum(float(fold["test_graph_count"]) for fold in valid_folds), 1.0)
                        * SPARSITY
                    )
                else:
                    weighted_mean_stages = {}
                    graph_balanced_mean_stages = {}
                    code_scalars = None
                branch_payload = {
                    "view": view,
                    "geometry": geometry,
                    "checkpoint": checkpoint,
                    "fold_rows": fold_rows,
                    "weighted_mean_stages": weighted_mean_stages,
                    "graph_balanced_mean_stages": graph_balanced_mean_stages,
                    "stratum_rows": _merge_stratum_rows(valid_folds),
                    "dictionary_scalars": int(
                        GEOMETRIES[geometry][0] * (GEOMETRIES[geometry][0] - 1) // 2 * N_ATOMS
                    ),
                    "code_scalars_per_graph": code_scalars,
                    "random_dictionary_seed": _random_dictionary_seed(
                        geometry, checkpoint, 0
                    ),
                }
                branches[view][branch] = branch_payload
                if valid_folds:
                    print(
                        f"branch={view}/{branch} folds={len(valid_folds)} "
                        f"final={weighted_mean_stages['FINAL']['uncorrected']['patch_relative_error']:.4f} "
                        f"init={weighted_mean_stages['INIT']['uncorrected']['patch_relative_error']:.4f}",
                        flush=True,
                    )

    decision = classify(
        branches,
        exposure_rows,
        views=views,
        geometries=geometries,
        checkpoints=checkpoints,
        n_folds=args.n_splits,
    )

    pilot_payload = None
    if args.pilot_json.exists():
        pilot_payload = json.loads(args.pilot_json.read_text(encoding="utf-8"))
    verification = _verify_pilot_counts(
        args, pilot_payload, pilot_indices, prepared_by_branch, coverage_missing
    )

    patch_overlap = _patch_overlap_audit(
        bundle,
        folds_by_view,
        prepared_by_branch,
        views=views,
        geometries=geometries,
        checkpoints=checkpoints,
    )

    payload = {
        "protocol": PROTOCOL,
        "config": {
            "pilot_size": args.pilot_size,
            "pilot_seed": args.pilot_seed,
            "cover_seed": args.cover_seed,
            "maximum_patches": args.maximum_patches,
            "multiplier": args.multiplier,
            "retained_beam": args.retained_beam,
            "candidate_restarts": args.candidate_restarts,
            "geometries": {key: list(value) for key, value in GEOMETRIES.items()},
            "checkpoints": list(checkpoints),
            "views": list(views),
            "iterations": args.iterations,
            "n_splits": args.n_splits,
            "split_seed": args.split_seed,
            "n_atoms": N_ATOMS,
            "sparsity": SPARSITY,
            "T_min": T_MIN,
            "pca_rank": PCA_RANK,
            "ksvd_seed": KSVD_SEED,
            "random_dictionary_seed_base": RANDOM_DICTIONARY_SEED_BASE,
            "labels_used": False,
            "source_ids_in_patch_vectors": False,
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
        "verification": verification,
        "exposure_rows": exposure_rows,
        "patch_overlap": patch_overlap,
        "branches": branches,
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
