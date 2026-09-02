"""Frozen Stable-ID Beam8 pilot on the mentor 50-node subgraph bank."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .beam8_coverage_operating_point import (
    CHECKPOINTS,
    prefix_coverage_trajectory,
    select_operating_checkpoints,
)
from .data_mentor_subgraphs import (
    STRATUM_NAMES,
    default_cache_path,
    default_source_path,
    density_stratum,
    load_bundle,
    select_pilot_indices,
)
from .global_stable_ids import (
    ambiguity_audit,
    compute_global_wl_ids,
    mapped_id_audit,
    reorder_by_stable_ids,
)
from .marginal_candidate_cover import sample_marginal_candidate_cover
from .overlap_cover import audit_cover, patch_budget, remap_cover
from .run_beam8_coverage_operating_point_audit import GEOMETRIES, SUMMARY_METRICS
from .run_global_stable_node_id_audit import (
    _class_chain_metrics,
    _sampler_comparison,
    relabeled_stable_to_base_mapping,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JSON = ROOT / "tracks/ksvd/results/mentor_subgraphs/beam8_pilot_20260806.json"
DEFAULT_REPORT = ROOT / "tracks/ksvd/results/mentor_subgraphs/BEAM8_PILOT_20260806.md"
PROTOCOL = "tracks/ksvd/docs/KSVD_MENTOR_SUBGRAPHS_BEAM8_PILOT_PROTOCOL_20260806.md"
DEFAULT_STABILITY_PERMUTATIONS = (960201, 960202)


def _cover_invariants(audit: dict[str, Any], overlap: int) -> bool:
    return bool(
        audit["patch_connected_rate"] == 1.0
        and audit["continuous_transition_fraction"] == 1.0
        and all(value == overlap for value in audit["transition_overlap_exact"])
    )


def _mean(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def _summary(
    checkpoint_rows: Sequence[dict[str, Any]],
    graph_rows: Sequence[dict[str, Any]],
    *,
    geometry: str,
    checkpoint: str,
    stratum: str | None = None,
) -> dict[str, Any]:
    trials = [
        row
        for row in graph_rows
        if row["geometry"] == geometry
        and (stratum is None or row["density_stratum"] == stratum)
    ]
    selected = [
        row
        for row in checkpoint_rows
        if row["geometry"] == geometry
        and row["checkpoint"] == checkpoint
        and (stratum is None or row["density_stratum"] == stratum)
    ]
    return {
        "geometry": geometry,
        "checkpoint": checkpoint,
        "density_stratum": stratum,
        "trial_count": len(trials),
        "reachable_count": len(selected),
        "reachable_fraction": len(selected) / max(len(trials), 1),
        **{metric: _mean(selected, metric) for metric in SUMMARY_METRICS},
    }


def _select_stability_indices(
    pilot_indices: np.ndarray,
    strata: np.ndarray,
    *,
    count: int,
    seed: int,
) -> np.ndarray:
    if count <= 0:
        return np.zeros(0, dtype=np.int64)
    count = min(int(count), int(pilot_indices.size))
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    base, extra = divmod(count, len(STRATUM_NAMES))
    for stratum in range(len(STRATUM_NAMES)):
        candidates = pilot_indices[strata[pilot_indices] == stratum]
        take = min(base + int(stratum < extra), int(candidates.size))
        if take:
            selected.extend(int(value) for value in rng.permutation(candidates)[:take])
    if len(selected) < count:
        remaining = np.setdiff1d(pilot_indices, np.asarray(selected, dtype=np.int64))
        selected.extend(int(value) for value in rng.permutation(remaining)[: count - len(selected)])
    return np.sort(np.asarray(selected, dtype=np.int64))


def _geometry_decision(
    graph_rows: Sequence[dict[str, Any]],
    summaries: Sequence[dict[str, Any]],
    geometry: str,
) -> dict[str, Any]:
    rows = [row for row in graph_rows if row["geometry"] == geometry]
    summary = {
        row["checkpoint"]: row
        for row in summaries
        if row["geometry"] == geometry and row["density_stratum"] is None
    }
    stratum_checks = []
    for stratum in STRATUM_NAMES:
        subset = [row for row in rows if row["density_stratum"] == stratum]
        if len(subset) < 20:
            continue
        success = float(np.mean([row["sampling_success"] for row in subset]))
        stratum_checks.append(
            {"density_stratum": stratum, "trial_count": len(subset), "success": success, "passed": success >= 0.95}
        )
    checks = {
        "all_invariants": bool(rows) and all(row["cover_invariants"] for row in rows),
        "sampler_success_at_least_099": bool(rows)
        and float(np.mean([row["sampling_success"] for row in rows])) >= 0.99,
        "base_reach_at_least_099": summary["BASE"]["reachable_fraction"] >= 0.99,
        "edge95_reach_at_least_095": summary["EDGE95"]["reachable_fraction"] >= 0.95,
        "fair95_reach_at_least_090": summary["FAIR95"]["reachable_fraction"] >= 0.90,
        "all_large_strata_success_at_least_095": all(row["passed"] for row in stratum_checks),
    }
    return {
        "geometry": geometry,
        "checks": checks,
        "stratum_checks": stratum_checks,
        "passed": all(checks.values()),
    }


def classify(
    graph_rows: Sequence[dict[str, Any]],
    checkpoint_rows: Sequence[dict[str, Any]],
    stability_rows: Sequence[dict[str, Any]],
    geometries: Sequence[str] | None = None,
) -> dict[str, Any]:
    selected_geometries = tuple(geometries or GEOMETRIES)
    summaries = [
        _summary(checkpoint_rows, graph_rows, geometry=geometry, checkpoint=checkpoint, stratum=stratum)
        for geometry in selected_geometries
        for checkpoint in CHECKPOINTS
        for stratum in (None, *STRATUM_NAMES)
    ]
    geometry_decisions = [
        _geometry_decision(graph_rows, summaries, geometry) for geometry in selected_geometries
    ]
    feasible = [row["geometry"] for row in geometry_decisions if row["passed"]]

    if stability_rows:
        singleton_trials = sum(int(row["singleton_trial_count"]) for row in stability_rows)
        singleton_matches = sum(int(row["singleton_match_count"]) for row in stability_rows)
        stability_metrics = {
            "stable_class_match_rate": _mean(stability_rows, "stable_class_match_rate"),
            "singleton_unique_id_match_rate": (
                singleton_matches / singleton_trials if singleton_trials else None
            ),
            "canonical_adjacency_match": _mean(stability_rows, "canonical_adjacency_match"),
            "rooted_vector_row_match": _mean(stability_rows, "rooted_vector_row_match"),
            "rooted_transition_map_match": _mean(stability_rows, "rooted_transition_map_match"),
            "absolute_edge_coverage_delta": _mean(stability_rows, "absolute_edge_coverage_delta"),
            "absolute_pair_coverage_delta": _mean(stability_rows, "absolute_pair_coverage_delta"),
            "exact_ordered_chain_match": _mean(stability_rows, "exact_ordered_chain_match"),
            "patch_set_jaccard": _mean(stability_rows, "patch_set_jaccard"),
            "trial_count": len(stability_rows),
            "singleton_trial_count": singleton_trials,
        }
        stable_checks = {
            "stable_class_match_is_one": stability_metrics["stable_class_match_rate"] == 1.0,
            "singleton_match_at_least_0999_or_null": stability_metrics["singleton_unique_id_match_rate"] is None
            or stability_metrics["singleton_unique_id_match_rate"] >= 0.999,
            "canonical_adjacency_at_least_099": stability_metrics["canonical_adjacency_match"] >= 0.99,
            "rooted_vector_at_least_099": stability_metrics["rooted_vector_row_match"] >= 0.99,
            "transition_at_least_099": stability_metrics["rooted_transition_map_match"] >= 0.99,
            "edge_delta_at_most_001": stability_metrics["absolute_edge_coverage_delta"] <= 0.01,
            "pair_delta_at_most_001": stability_metrics["absolute_pair_coverage_delta"] <= 0.01,
        }
        stable_gate = all(stable_checks.values())
    else:
        stability_metrics = {}
        stable_checks = {"stability_trials_present": False}
        stable_gate = False

    data_gate = bool(graph_rows) and all(row["data_contract"] for row in graph_rows)
    if not data_gate:
        label = "FAIL_MENTOR_SUBGRAPH_DATA_CONTRACT"
    elif stable_gate and feasible:
        label = "READY_FOR_GROUPED_KSVD_RECONSTRUCTION_FOLLOWUP"
    elif feasible:
        label = "COVERAGE_READY_STABILITY_REQUIRES_REPAIR"
    elif stable_gate:
        label = "STABLE_PREORDER_READY_BEAM8_GEOMETRY_NO_GO"
    else:
        label = "BEAM8_PILOT_NO_GO"
    return {
        "classification": label,
        "data_gate": data_gate,
        "stable_gate": stable_gate,
        "stable_checks": stable_checks,
        "stability_metrics": stability_metrics,
        "feasible_geometries": feasible,
        "geometry_decisions": geometry_decisions,
        "summaries": summaries,
    }


def _f(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def render_report(payload: dict[str, Any]) -> str:
    geometries = tuple(payload["config"]["geometries"])
    decision = payload["decision"]
    aggregate = {
        (row["geometry"], row["checkpoint"]): row
        for row in decision["summaries"]
        if row["density_stratum"] is None
    }
    lines = [
        "# 导师 50-node 真实子图：Stable-ID Beam8 pilot",
        "",
        f"> 协议：`{payload['protocol']}`  ",
        f"> 判定：`{decision['classification']}`",
        "",
        "## 1. 数据与抽样",
        "",
        f"- source SHA-256：`{payload['data']['source_sha256']}`；",
        f"- cache graphs：`{payload['data']['cache_graph_count']}`；pilot：`{payload['selection']['size']}`；",
        f"- pilot roots：`{payload['selection']['n_distinct_roots']}`；density quotas：`{payload['selection']['selected_counts']}`；",
        f"- labels used：`False`；source IDs in patch vectors：`False`。",
        "",
        "## 2. 总判定",
        "",
        f"- feasible geometries：`{decision['feasible_geometries']}`；",
        f"- stable gate：`{decision['stable_gate']}`；checks：`{decision['stable_checks']}`。",
        "",
        "| geometry | invariants | sampler | BASE | EDGE95 | FAIR95 | strata | pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in decision["geometry_decisions"]:
        c = row["checks"]
        lines.append(
            f"| {row['geometry']} | {c['all_invariants']} | {c['sampler_success_at_least_099']} | "
            f"{c['base_reach_at_least_099']} | {c['edge95_reach_at_least_095']} | "
            f"{c['fair95_reach_at_least_090']} | {c['all_large_strata_success_at_least_095']} | {row['passed']} |"
        )
    lines.extend([
        "",
        "## 3. Coverage checkpoints",
        "",
        "| geometry | checkpoint | reach | patches | edge/pair | node p10/min | residual | RAW RMSE | unique pairs | code scalars |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for geometry in geometries:
        for checkpoint in CHECKPOINTS:
            row = aggregate[(geometry, checkpoint)]
            lines.append(
                f"| {geometry} | {checkpoint} | {row['reachable_fraction']:.3f} | {_f(row['patch_count'], 2)} | "
                f"{_f(row['edge_coverage'])}/{_f(row['pair_coverage'])} | "
                f"{_f(row['node_incident_recall_p10'])}/{_f(row['node_incident_recall_minimum'])} | "
                f"{_f(row['residual_edge_count'], 2)} | {_f(row['raw_zero_fill_rmse'])} | "
                f"{_f(row['unique_observed_pair_count'], 1)} | {_f(row['code_scalars'], 1)} |"
            )
    lines.extend([
        "",
        "## 4. Density strata at BASE",
        "",
        "| geometry | stratum | trials | reach | patches | edge | pair | node p10 | residual |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for geometry in geometries:
        for stratum in STRATUM_NAMES:
            row = next(
                item for item in decision["summaries"]
                if item["geometry"] == geometry and item["checkpoint"] == "BASE" and item["density_stratum"] == stratum
            )
            lines.append(
                f"| {geometry} | {stratum} | {row['trial_count']} | {row['reachable_fraction']:.3f} | "
                f"{_f(row['patch_count'], 2)} | {_f(row['edge_coverage'])} | {_f(row['pair_coverage'])} | "
                f"{_f(row['node_incident_recall_p10'])} | {_f(row['residual_edge_count'], 2)} |"
            )
    s = decision["stability_metrics"]
    lines.extend([
        "",
        "## 5. Relabel stability",
        "",
        "| trials | class | singleton ID | canonical adjacency | exact chain | patch Jaccard | vectors | transition | edge/pair delta |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {s.get('trial_count', 0)} | {_f(s.get('stable_class_match_rate'))} | "
        f"{_f(s.get('singleton_unique_id_match_rate'))} | {_f(s.get('canonical_adjacency_match'))} | "
        f"{_f(s.get('exact_ordered_chain_match'))} | {_f(s.get('patch_set_jaccard'))} | "
        f"{_f(s.get('rooted_vector_row_match'))} | {_f(s.get('rooted_transition_map_match'))} | "
        f"{_f(s.get('absolute_edge_coverage_delta'))}/{_f(s.get('absolute_pair_coverage_delta'))} |",
        "",
        "## 6. Ordering control",
        "",
        "SOURCE_INPUT_ORDER 与 GLOBAL_WL_PREORDER 使用同 seed、同 BASE budget；比较只衡量 numeric ordering 对 Beam8 tie/path 的影响。Source IDs 未进入 local patch vectors。",
        "",
        "| geometry | trials | stable-source edge | stable-source pair | stable-source RAW RMSE | patch Jaccard | vector rows |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for geometry, row in payload["ordering_control"].items():
        lines.append(
            f"| {geometry} | {row['trial_count']} | {_f(row['stable_minus_source_edge_coverage'])} | "
            f"{_f(row['stable_minus_source_pair_coverage'])} | {_f(row['stable_minus_source_raw_rmse'])} | "
            f"{_f(row['patch_set_jaccard'])} | {_f(row['rooted_vector_row_match'])} |"
        )
    lines.extend([
        "",
        "## 7. 边界",
        "",
        "- density strata 与 root_candidate 只用于抽样/条件报告，不是 labels。",
        "- source IDs 只用于 bookkeeping、映射和 overlap audit；KSVD local coordinates 仍是 rooted-canonical adjacency。",
        "- root_candidate 只是 NetworkX 第一个 insertion-order 节点，生成方尚未确认其语义。",
        "- hybrid bits 是 optimistic proxy；本轮不恢复 graph codec 主张。",
        "- pilot 只决定是否进入 grouped-split KSVD reconstruction，不直接训练 Transformer。",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--stability-size", type=int, default=50)
    parser.add_argument("--permutation-seeds", type=int, nargs="+", default=list(DEFAULT_STABILITY_PERMUTATIONS))
    parser.add_argument("--geometries", nargs="+", choices=tuple(GEOMETRIES), default=list(GEOMETRIES))
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    cache_path = args.cache or default_cache_path(args.source)
    bundle = load_bundle(source_pkl=args.source, cache_path=cache_path, force=args.force_cache)
    selection = select_pilot_indices(bundle, n_pilot=args.pilot_size, seed=args.pilot_seed)
    all_strata = density_stratum(bundle.avg_degrees)
    pilot_indices = selection.indices
    selected_geometries = {name: GEOMETRIES[name] for name in args.geometries}

    graph_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    order_rows: list[dict[str, Any]] = []
    stable_cache: dict[int, tuple[Any, np.ndarray, float]] = {}

    for position, source_index_raw in enumerate(pilot_indices):
        source_index = int(source_index_raw)
        source_adjacency = bundle.adjacency[source_index].astype(np.int8, copy=False)
        started = time.perf_counter()
        ids = compute_global_wl_ids(source_adjacency)
        stable_seconds = time.perf_counter() - started
        stable_adjacency = reorder_by_stable_ids(source_adjacency, ids)
        stable_cache[source_index] = (ids, stable_adjacency, stable_seconds)
        stratum = STRATUM_NAMES[int(all_strata[source_index])]
        for geometry_index, (geometry, (patch_size, overlap)) in enumerate(selected_geometries.items()):
            base_budget = patch_budget(
                stable_adjacency,
                patch_size=patch_size,
                target_overlap=overlap,
                edge_capacity_multiplier=args.multiplier,
            )
            sampler_seed = int(np.random.SeedSequence([args.cover_seed, source_index, geometry_index]).generate_state(1, dtype=np.uint32)[0])
            metadata = {
                "source_graph_index": source_index,
                "pilot_position": position,
                "density_stratum": stratum,
                "average_degree": float(bundle.avg_degrees[source_index]),
                "edge_count": int(bundle.edge_counts[source_index]),
                "root_global_id": int(bundle.roots[source_index]),
                "geometry": geometry,
                "patch_size": patch_size,
                "target_overlap": overlap,
                "base_budget": int(base_budget),
                "singleton_fraction": ids.singleton_fraction,
                "fully_singleton": ids.fully_singleton,
                "largest_class": ids.largest_class,
                "stable_id_seconds": stable_seconds,
                "data_contract": True,
            }
            error = None
            started = time.perf_counter()
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
            except Exception as exc:
                stable_cover = None
                error = f"{type(exc).__name__}: {exc}"
            sampling_seconds = time.perf_counter() - started
            if stable_cover is None:
                graph_rows.append({**metadata, "sampling_success": False, "sampled_patch_count": 0, "cover_invariants": False, "sampling_seconds": sampling_seconds, "error": error})
                continue
            stable_audit = audit_cover(stable_adjacency, stable_cover)
            invariants = _cover_invariants(stable_audit, overlap)
            trajectory = prefix_coverage_trajectory(
                stable_adjacency,
                stable_cover,
                patch_size=patch_size,
                overlap=overlap,
                maximum_patches=args.maximum_patches,
            )
            selected = select_operating_checkpoints(trajectory, base_patch_count=base_budget)
            graph_rows.append({**metadata, "sampling_success": True, "sampled_patch_count": len(stable_cover.patches), "cover_invariants": invariants, "sampling_seconds": sampling_seconds, "error": error})
            for checkpoint, row in selected.items():
                if row is not None:
                    checkpoint_rows.append({**metadata, "checkpoint": checkpoint, **row})

            # Matched source-input-order control, restricted to BASE cost.
            started = time.perf_counter()
            source_cover = sample_marginal_candidate_cover(
                source_adjacency,
                np.random.default_rng(sampler_seed),
                n_patches=base_budget,
                patch_size=patch_size,
                target_overlap=overlap,
                retained_beam=args.retained_beam,
                candidate_restarts=args.candidate_restarts,
            )
            source_seconds = time.perf_counter() - started
            source_to_stable = np.asarray(ids.canonical_ids, dtype=np.int64)
            mapped_source = remap_cover(source_cover, source_to_stable, stable_adjacency)
            stable_prefix = remap_cover(stable_cover, np.arange(stable_adjacency.shape[0]), stable_adjacency)
            from .run_beam8_coverage_ksvd_followup import _prefix_cover
            stable_prefix = _prefix_cover(stable_prefix, base_budget, "stable_base")
            comparison = _sampler_comparison(stable_adjacency, mapped_source, stable_adjacency, stable_prefix, overlap=overlap)
            order_rows.append({**metadata, "source_sampling_seconds": source_seconds, "stable_sampling_seconds": sampling_seconds, **comparison})
        if (position + 1) % 25 == 0 or position + 1 == len(pilot_indices):
            print(f"prepared={position + 1}/{len(pilot_indices)}", flush=True)

    stability_indices = _select_stability_indices(
        pilot_indices, all_strata, count=args.stability_size, seed=args.pilot_seed + 1
    )
    stability_rows: list[dict[str, Any]] = []
    stability_geometry = "s10_o3" if "s10_o3" in selected_geometries else next(iter(selected_geometries))
    patch_size, overlap = selected_geometries[stability_geometry]
    for source_index_raw in stability_indices:
        source_index = int(source_index_raw)
        source_adjacency = bundle.adjacency[source_index].astype(np.int8, copy=False)
        base_ids, base_stable_adjacency, _seconds = stable_cache[source_index]
        budget = patch_budget(base_stable_adjacency, patch_size=patch_size, target_overlap=overlap, edge_capacity_multiplier=args.multiplier)
        sampler_seed = int(np.random.SeedSequence([args.cover_seed, source_index, 99]).generate_state(1, dtype=np.uint32)[0])
        base_cover = sample_marginal_candidate_cover(
            base_stable_adjacency, np.random.default_rng(sampler_seed), n_patches=budget,
            patch_size=patch_size, target_overlap=overlap,
            retained_beam=args.retained_beam, candidate_restarts=args.candidate_restarts,
        )
        base_classes = tuple(base_ids.class_ids[node] for node in base_ids.order)
        for permutation_seed in args.permutation_seeds:
            permutation = np.random.default_rng(np.random.SeedSequence([permutation_seed, source_index]).generate_state(1, dtype=np.uint32)[0]).permutation(50)
            relabeled = source_adjacency[np.ix_(permutation, permutation)]
            relabeled_ids = compute_global_wl_ids(relabeled)
            id_audit = mapped_id_audit(base_ids, relabeled_ids, permutation)
            relabeled_stable = reorder_by_stable_ids(relabeled, relabeled_ids)
            canonical_match = bool(np.array_equal(base_stable_adjacency, relabeled_stable))
            relabeled_cover = sample_marginal_candidate_cover(
                relabeled_stable, np.random.default_rng(sampler_seed), n_patches=budget,
                patch_size=patch_size, target_overlap=overlap,
                retained_beam=args.retained_beam, candidate_restarts=args.candidate_restarts,
            )
            mapping = relabeled_stable_to_base_mapping(base_ids, relabeled_ids, permutation)
            mapped_cover = remap_cover(relabeled_cover, mapping, base_stable_adjacency)
            comparison = _sampler_comparison(base_stable_adjacency, base_cover, base_stable_adjacency, mapped_cover, overlap=overlap)
            class_metrics = _class_chain_metrics(base_cover, base_classes, mapped_cover, base_classes)
            ambiguity = ambiguity_audit(source_adjacency, base_ids)
            stability_rows.append({
                "source_graph_index": source_index,
                "density_stratum": STRATUM_NAMES[int(all_strata[source_index])],
                "permutation_seed": int(permutation_seed),
                "fully_singleton": base_ids.fully_singleton,
                "canonical_adjacency_match": canonical_match,
                "singleton_match_count": int(round(id_audit["singleton_unique_id_match_rate"] * id_audit["singleton_trial_count"])),
                **ambiguity,
                **id_audit,
                **class_metrics,
                **comparison,
            })
        print(f"stability_graph={source_index}", flush=True)

    decision = classify(
        graph_rows,
        checkpoint_rows,
        stability_rows,
        geometries=tuple(selected_geometries),
    )

    ordering_control = {}
    for geometry in selected_geometries:
        rows = [row for row in order_rows if row["geometry"] == geometry]
        ordering_control[geometry] = {
            "trial_count": len(rows),
            "stable_minus_source_edge_coverage": _mean(rows, "right_edge_coverage") - _mean(rows, "left_edge_coverage"),
            "stable_minus_source_pair_coverage": _mean(rows, "right_pair_coverage") - _mean(rows, "left_pair_coverage"),
            "stable_minus_source_raw_rmse": _mean(rows, "right_raw_rmse") - _mean(rows, "left_raw_rmse"),
            "patch_set_jaccard": _mean(rows, "patch_set_jaccard"),
            "rooted_vector_row_match": _mean(rows, "rooted_vector_row_match"),
        }

    payload = {
        "protocol": PROTOCOL,
        "config": {
            "pilot_size": args.pilot_size, "pilot_seed": args.pilot_seed,
            "cover_seed": args.cover_seed, "maximum_patches": args.maximum_patches,
            "multiplier": args.multiplier, "retained_beam": args.retained_beam,
            "candidate_restarts": args.candidate_restarts,
            "geometries": {key: list(value) for key, value in selected_geometries.items()},
            "stability_size": args.stability_size, "permutation_seeds": args.permutation_seeds,
            "labels_used": False, "source_ids_in_patch_vectors": False,
            "strata_used_as_labels": False,
        },
        "data": {
            "source_path": bundle.metadata.get("source_path"),
            "source_sha256": bundle.metadata.get("source_sha256"),
            "cache_path": str(cache_path), "cache_graph_count": bundle.n_graphs,
            "metadata": bundle.metadata,
        },
        "selection": selection.summary(),
        "pilot_indices": pilot_indices.tolist(),
        "stability_indices": stability_indices.tolist(),
        "graph_rows": graph_rows,
        "checkpoint_rows": checkpoint_rows,
        "ordering_rows": order_rows,
        "ordering_control": ordering_control,
        "stability_rows": stability_rows,
        "decision": decision,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(render_report(payload), encoding="utf-8")
    print(f"decision={decision['classification']}")
    print(f"wrote {args.json}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
