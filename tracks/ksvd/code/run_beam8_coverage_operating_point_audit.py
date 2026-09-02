"""Audit stable-ID Beam8 geometry and prefix coverage operating points."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .beam8_coverage_operating_point import (
    CHECKPOINTS,
    nondominated_candidates,
    prefix_coverage_trajectory,
    select_operating_checkpoints,
)
from .global_stable_ids import compute_global_wl_ids, reorder_by_stable_ids
from .marginal_candidate_cover import sample_marginal_candidate_cover
from .overlap_cover import audit_cover, patch_budget
from .run_overlap_cover_audit import FAMILIES, generate_graph


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JSON = ROOT / "tracks/ksvd/results/from_scratch/beam8_coverage_operating_point_audit_20260805.json"
DEFAULT_REPORT = ROOT / "tracks/ksvd/results/from_scratch/BEAM8_COVERAGE_OPERATING_POINT_AUDIT_20260805.md"
PROTOCOL = "tracks/ksvd/docs/KSVD_BEAM8_COVERAGE_OPERATING_POINT_PROTOCOL_20260805.md"
GEOMETRIES = {"s8_o2": (8, 2), "s10_o3": (10, 3), "s12_o4": (12, 4)}
DOMAINS = {
    "main": {"graph_bank_seed": 810001, "degrees": (15, 20, 25), "graphs_per_cell": 8},
    "sparse": {"graph_bank_seed": 810002, "degrees": (3, 4), "graphs_per_cell": 4},
}
SUMMARY_METRICS = (
    "patch_count",
    "raw_pair_slots",
    "unique_observed_pair_count",
    "node_coverage",
    "pair_coverage",
    "edge_coverage",
    "residual_edge_count",
    "raw_zero_fill_rmse",
    "covered_edge_density",
    "last_marginal_new_edges",
    "mean_marginal_new_edges",
    "node_incident_recall_mean",
    "node_incident_recall_p10",
    "node_incident_recall_minimum",
    "fully_covered_node_fraction",
    "nodes_with_residual_fraction",
    "maximum_residual_incident_count",
    "edge_multiplicity_mean",
    "edge_multiplicity_cv",
    "low_degree_incident_edge_recall",
    "high_degree_incident_edge_recall",
    "zero_common_neighbor_edge_recall",
    "bridge_edge_recall",
    "cross_block_edge_recall",
    "canonical_set_identity_bits",
    "ksvd_code_proxy_bits",
    "residual_subset_bits",
    "canonical_hybrid_proxy_bits",
    "dictionary_scalars",
    "code_scalars",
)


def _mean(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def _summary(
    checkpoint_rows: Sequence[dict[str, Any]],
    graph_rows: Sequence[dict[str, Any]],
    *,
    domain: str,
    geometry: str,
    checkpoint: str,
    cover_seed: int | None = None,
) -> dict[str, Any]:
    trials = [
        row
        for row in graph_rows
        if row["domain"] == domain
        and row["geometry"] == geometry
        and (cover_seed is None or row["cover_seed"] == cover_seed)
    ]
    rows = [
        row
        for row in checkpoint_rows
        if row["domain"] == domain
        and row["geometry"] == geometry
        and row["checkpoint"] == checkpoint
        and (cover_seed is None or row["cover_seed"] == cover_seed)
    ]
    return {
        "domain": domain,
        "geometry": geometry,
        "checkpoint": checkpoint,
        "cover_seed": cover_seed,
        "trial_count": len(trials),
        "reachable_count": len(rows),
        "reachable_fraction": len(rows) / max(len(trials), 1),
        **{metric: _mean(rows, metric) for metric in SUMMARY_METRICS},
    }


def _geometry_decision(
    summaries: Sequence[dict[str, Any]],
    graph_rows: Sequence[dict[str, Any]],
    geometry: str,
    cover_seeds: Sequence[int],
) -> dict[str, Any]:
    seed_checks = []
    for seed in cover_seeds:
        edge95 = next(
            row
            for row in summaries
            if row["domain"] == "main"
            and row["geometry"] == geometry
            and row["checkpoint"] == "EDGE95"
            and row["cover_seed"] == seed
        )
        fair = next(
            row
            for row in summaries
            if row["domain"] == "main"
            and row["geometry"] == geometry
            and row["checkpoint"] == "FAIR95"
            and row["cover_seed"] == seed
        )
        trials = [
            row
            for row in graph_rows
            if row["domain"] == "main"
            and row["geometry"] == geometry
            and row["cover_seed"] == seed
        ]
        checks = {
            "invariants": bool(trials) and all(row["cover_invariants"] for row in trials),
            "edge95_reach_at_least_099": edge95["reachable_fraction"] >= 0.99,
            "fair95_reach_at_least_095": fair["reachable_fraction"] >= 0.95,
            "fair95_edge_at_least_095": fair["edge_coverage"] is not None
            and fair["edge_coverage"] >= 0.95,
            "fair95_node_p10_at_least_090": fair["node_incident_recall_p10"]
            is not None
            and fair["node_incident_recall_p10"] >= 0.90,
        }
        seed_checks.append(
            {"cover_seed": int(seed), "checks": checks, "passed": all(checks.values())}
        )
    return {
        "geometry": geometry,
        "passed": all(row["passed"] for row in seed_checks),
        "seed_checks": seed_checks,
    }


def classify(
    checkpoint_rows: Sequence[dict[str, Any]],
    graph_rows: Sequence[dict[str, Any]],
    cover_seeds: Sequence[int],
) -> dict[str, Any]:
    summaries = []
    for domain in DOMAINS:
        for geometry in GEOMETRIES:
            for checkpoint in CHECKPOINTS:
                summaries.append(
                    _summary(
                        checkpoint_rows,
                        graph_rows,
                        domain=domain,
                        geometry=geometry,
                        checkpoint=checkpoint,
                    )
                )
                for seed in cover_seeds:
                    summaries.append(
                        _summary(
                            checkpoint_rows,
                            graph_rows,
                            domain=domain,
                            geometry=geometry,
                            checkpoint=checkpoint,
                            cover_seed=int(seed),
                        )
                    )
    geometry_decisions = [
        _geometry_decision(summaries, graph_rows, geometry, cover_seeds)
        for geometry in GEOMETRIES
    ]
    feasible = [row["geometry"] for row in geometry_decisions if row["passed"]]
    candidate_rows = []
    for geometry in feasible:
        for checkpoint in ("BASE", "FAIR95"):
            row = next(
                summary
                for summary in summaries
                if summary["domain"] == "main"
                and summary["geometry"] == geometry
                and summary["checkpoint"] == checkpoint
                and summary["cover_seed"] is None
            )
            if row["reachable_fraction"] == 1.0:
                candidate_rows.append(
                    {"candidate": f"{geometry}_{checkpoint}", **row}
                )
    pareto = nondominated_candidates(candidate_rows) if candidate_rows else []
    sparse_success = {}
    for geometry in GEOMETRIES:
        rows = [
            row
            for row in graph_rows
            if row["domain"] == "sparse" and row["geometry"] == geometry
        ]
        sparse_success[geometry] = {
            "trial_count": len(rows),
            "maximum_chain_reached_fraction": float(
                np.mean([row["maximum_chain_reached"] for row in rows])
            )
            if rows
            else 0.0,
            "invariant_fraction": float(
                np.mean([row["cover_invariants"] for row in rows])
            )
            if rows
            else 0.0,
        }
    label = (
        "COVERAGE_CANDIDATES_READY_FOR_KSVD"
        if feasible
        else "NO_GEOMETRY_PASSES_FAIR95_GATE"
    )
    return {
        "classification": label,
        "feasible_geometries": feasible,
        "geometry_decisions": geometry_decisions,
        "coverage_pareto_candidates": pareto,
        "candidate_rows": candidate_rows,
        "sparse_sampler_success": sparse_success,
        "summaries": summaries,
    }


def _fmt(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def render_report(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    aggregate = [row for row in decision["summaries"] if row["cover_seed"] is None]
    lines = [
        "# Stable-ID Beam8 coverage operating-point 审计",
        "",
        "> 日期：2026-08-05",
        f"> 协议：`{payload['protocol']}`",
        f"> 判定：`{decision['classification']}`",
        "",
        "## 1. 判定",
        "",
        f"- feasible geometries：`{decision['feasible_geometries']}`；",
        f"- coverage Pareto candidates：`{decision['coverage_pareto_candidates']}`。",
        "",
        "| geometry | seed | invariants | EDGE95 reach | FAIR95 reach | FAIR edge | FAIR node p10 | pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    summary_lookup = {
        (row["domain"], row["geometry"], row["checkpoint"], row["cover_seed"]): row
        for row in decision["summaries"]
    }
    for geometry in decision["geometry_decisions"]:
        for seed_row in geometry["seed_checks"]:
            seed = seed_row["cover_seed"]
            edge95 = summary_lookup[("main", geometry["geometry"], "EDGE95", seed)]
            fair = summary_lookup[("main", geometry["geometry"], "FAIR95", seed)]
            lines.append(
                f"| {geometry['geometry']} | {seed} | {seed_row['checks']['invariants']} | "
                f"{edge95['reachable_fraction']:.3f} | {fair['reachable_fraction']:.3f} | "
                f"{_fmt(fair['edge_coverage'])} | {_fmt(fair['node_incident_recall_p10'])} | "
                f"{seed_row['passed']} |"
            )
    lines.extend(
        [
            "",
            "## 2. Main bank checkpoints",
            "",
            "| geometry | checkpoint | reach | patches | edge/pair | residual | node p10/min | full nodes | residual nodes | unique pairs | last/mean new edges | hybrid bits |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for geometry in GEOMETRIES:
        for checkpoint in CHECKPOINTS:
            row = summary_lookup[("main", geometry, checkpoint, None)]
            lines.append(
                f"| {geometry} | {checkpoint} | {row['reachable_fraction']:.3f} | "
                f"{_fmt(row['patch_count'], 2)} | {_fmt(row['edge_coverage'])}/"
                f"{_fmt(row['pair_coverage'])} | {_fmt(row['residual_edge_count'], 2)} | "
                f"{_fmt(row['node_incident_recall_p10'])}/{_fmt(row['node_incident_recall_minimum'])} | "
                f"{_fmt(row['fully_covered_node_fraction'])} | "
                f"{_fmt(row['nodes_with_residual_fraction'])} | "
                f"{_fmt(row['unique_observed_pair_count'], 1)} | "
                f"{_fmt(row['last_marginal_new_edges'], 2)}/"
                f"{_fmt(row['mean_marginal_new_edges'], 2)} | "
                f"{_fmt(row['canonical_hybrid_proxy_bits'], 1)} |"
            )
    lines.extend(
        [
            "",
            "## 3. Residual structure（main）",
            "",
            "| geometry | checkpoint | low-degree recall | high-degree recall | zero-common recall | bridge recall | cross-block recall |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for geometry in GEOMETRIES:
        for checkpoint in ("BASE", "FAIR95"):
            row = summary_lookup[("main", geometry, checkpoint, None)]
            lines.append(
                f"| {geometry} | {checkpoint} | "
                f"{_fmt(row['low_degree_incident_edge_recall'])} | "
                f"{_fmt(row['high_degree_incident_edge_recall'])} | "
                f"{_fmt(row['zero_common_neighbor_edge_recall'])} | "
                f"{_fmt(row['bridge_edge_recall'])} | "
                f"{_fmt(row['cross_block_edge_recall'])} |"
            )
    lines.extend(
        [
            "",
            "## 4. Sparse transfer probe",
            "",
            "| geometry | max-chain reach | invariants | checkpoint | checkpoint reach | patches | edge | node p10 | residual |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for geometry in GEOMETRIES:
        success = decision["sparse_sampler_success"][geometry]
        for checkpoint in ("BASE", "FAIR95", "EDGE100"):
            row = summary_lookup[("sparse", geometry, checkpoint, None)]
            lines.append(
                f"| {geometry} | {success['maximum_chain_reached_fraction']:.3f} | "
                f"{success['invariant_fraction']:.3f} | {checkpoint} | "
                f"{row['reachable_fraction']:.3f} | {_fmt(row['patch_count'], 2)} | "
                f"{_fmt(row['edge_coverage'])} | {_fmt(row['node_incident_recall_p10'])} | "
                f"{_fmt(row['residual_edge_count'], 2)} |"
            )
    lines.extend(
        [
            "",
            "## 5. Coverage candidates",
            "",
            "| candidate | patches | dictionary/code scalars | edge | node p10 | low-degree recall | unique pairs | hybrid bits | Pareto |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    pareto = set(decision["coverage_pareto_candidates"])
    for row in decision["candidate_rows"]:
        lines.append(
            f"| {row['candidate']} | {_fmt(row['patch_count'], 2)} | "
            f"{_fmt(row['dictionary_scalars'], 0)}/{_fmt(row['code_scalars'], 1)} | "
            f"{_fmt(row['edge_coverage'])} | {_fmt(row['node_incident_recall_p10'])} | "
            f"{_fmt(row['low_degree_incident_edge_recall'])} | "
            f"{_fmt(row['unique_observed_pair_count'], 1)} | "
            f"{_fmt(row['canonical_hybrid_proxy_bits'], 1)} | "
            f"{row['candidate'] in pareto} |"
        )
    lines.extend(
        [
            "",
            "## 6. 边界",
            "",
            "- coverage Pareto 只筛选进入 rooted-canonical KSVD follow-up 的候选，不直接冻结最终配置。",
            "- residual structure 按 stable-ID 坐标正确搬运原 block membership；不把 numeric ID 顺序误当社区。",
            "- bridge 等空 category 以 null 排除均值，不用 vacuous 1.0。",
            "- hybrid bits 仍是 optimistic proxy，只用于 operating-point 相对比较。",
            "",
        ]
    )
    return "\n".join(lines)


def _domain_graphs(
    domain: str,
    *,
    n_nodes: int,
    graph_bank_seed: int,
    degrees: Sequence[int],
    graphs_per_cell: int,
) -> list[dict[str, Any]]:
    count = len(FAMILIES) * len(degrees) * graphs_per_cell
    sequences = np.random.SeedSequence(graph_bank_seed).spawn(count)
    output = []
    graph_index = 0
    for family in FAMILIES:
        for degree in degrees:
            for _replicate in range(graphs_per_cell):
                seed = int(sequences[graph_index].generate_state(1, dtype=np.uint32)[0])
                adjacency = generate_graph(family, n_nodes, int(degree), seed)
                stable_ids = compute_global_wl_ids(adjacency)
                stable_adjacency = reorder_by_stable_ids(adjacency, stable_ids)
                groups = (
                    tuple(
                        0 if int(old_node) < n_nodes // 2 else 1
                        for old_node in stable_ids.order
                    )
                    if family == "block"
                    else None
                )
                output.append(
                    {
                        "domain": domain,
                        "graph_index": graph_index,
                        "family": family,
                        "target_degree": int(degree),
                        "adjacency": stable_adjacency,
                        "block_groups": groups,
                    }
                )
                graph_index += 1
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cover-seeds", type=int, nargs="+", default=[970101, 970102, 970103])
    parser.add_argument("--n-nodes", type=int, default=50)
    parser.add_argument("--maximum-patches", type=int, default=60)
    parser.add_argument("--multiplier", type=float, default=1.5)
    parser.add_argument("--retained-beam", type=int, default=8)
    parser.add_argument("--candidate-restarts", type=int, default=1)
    parser.add_argument("--main-graphs-per-cell", type=int, default=8)
    parser.add_argument("--sparse-graphs-per-cell", type=int, default=4)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    domains = {
        "main": _domain_graphs(
            "main",
            n_nodes=args.n_nodes,
            graph_bank_seed=810001,
            degrees=(15, 20, 25),
            graphs_per_cell=args.main_graphs_per_cell,
        ),
        "sparse": _domain_graphs(
            "sparse",
            n_nodes=args.n_nodes,
            graph_bank_seed=810002,
            degrees=(3, 4),
            graphs_per_cell=args.sparse_graphs_per_cell,
        ),
    }
    checkpoint_rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    for domain, graphs in domains.items():
        for graph in graphs:
            adjacency = graph["adjacency"]
            for geometry_index, (geometry, (patch_size, overlap)) in enumerate(GEOMETRIES.items()):
                base_budget = patch_budget(
                    adjacency,
                    patch_size=patch_size,
                    target_overlap=overlap,
                    edge_capacity_multiplier=args.multiplier,
                )
                for cover_seed in args.cover_seeds:
                    seed = int(
                        np.random.SeedSequence(
                            [cover_seed, graph["graph_index"], geometry_index, 0 if domain == "main" else 1]
                        ).generate_state(1, dtype=np.uint32)[0]
                    )
                    started = time.perf_counter()
                    error = None
                    try:
                        cover = sample_marginal_candidate_cover(
                            adjacency,
                            np.random.default_rng(seed),
                            n_patches=args.maximum_patches,
                            patch_size=patch_size,
                            target_overlap=overlap,
                            retained_beam=args.retained_beam,
                            candidate_restarts=args.candidate_restarts,
                            allow_partial=True,
                        )
                    except Exception as exc:  # audit must retain failures
                        cover = None
                        error = f"{type(exc).__name__}: {exc}"
                    elapsed = time.perf_counter() - started
                    metadata = {
                        "domain": domain,
                        "graph_index": graph["graph_index"],
                        "family": graph["family"],
                        "target_degree": graph["target_degree"],
                        "geometry": geometry,
                        "patch_size": patch_size,
                        "target_overlap": overlap,
                        "cover_seed": int(cover_seed),
                    }
                    if cover is None:
                        graph_rows.append(
                            {
                                **metadata,
                                "base_budget": base_budget,
                                "sampled_patch_count": 0,
                                "maximum_chain_reached": False,
                                "cover_invariants": False,
                                "sampling_seconds": elapsed,
                                "error": error,
                            }
                        )
                        continue
                    audit = audit_cover(adjacency, cover)
                    invariants = (
                        audit["patch_connected_rate"] == 1.0
                        and audit["continuous_transition_fraction"] == 1.0
                        and all(
                            value == overlap
                            for value in audit["transition_overlap_exact"]
                        )
                    )
                    trajectory = prefix_coverage_trajectory(
                        adjacency,
                        cover,
                        patch_size=patch_size,
                        overlap=overlap,
                        maximum_patches=args.maximum_patches,
                        block_groups=graph["block_groups"],
                    )
                    selected = select_operating_checkpoints(
                        trajectory, base_patch_count=base_budget
                    )
                    graph_rows.append(
                        {
                            **metadata,
                            "base_budget": base_budget,
                            "sampled_patch_count": len(cover.patches),
                            "maximum_chain_reached": len(cover.patches) == args.maximum_patches,
                            "cover_invariants": bool(invariants),
                            "sampling_seconds": elapsed,
                            "error": error,
                        }
                    )
                    for checkpoint, row in selected.items():
                        if row is not None:
                            checkpoint_rows.append(
                                {**metadata, "checkpoint": checkpoint, **row}
                            )
                    base = selected["BASE"]
                    fair = selected["FAIR95"]
                    base_edge_text = (
                        "none" if base is None else f"{base['edge_coverage']:.3f}"
                    )
                    print(
                        f"{domain} graph={graph['graph_index']} {graph['family']}/d{graph['target_degree']} "
                        f"{geometry} seed={cover_seed} sampled={len(cover.patches)} "
                        f"base_edge={base_edge_text} "
                        f"fair_t={None if fair is None else fair['patch_count']}",
                        flush=True,
                    )

    decision = classify(checkpoint_rows, graph_rows, args.cover_seeds)
    payload = {
        "protocol": PROTOCOL,
        "config": {
            "cover_seeds": args.cover_seeds,
            "n_nodes": args.n_nodes,
            "maximum_patches": args.maximum_patches,
            "multiplier": args.multiplier,
            "retained_beam": args.retained_beam,
            "candidate_restarts": args.candidate_restarts,
            "geometries": {key: list(value) for key, value in GEOMETRIES.items()},
            "domains": DOMAINS,
            "main_graphs_per_cell": args.main_graphs_per_cell,
            "sparse_graphs_per_cell": args.sparse_graphs_per_cell,
            "labels_used": False,
        },
        "graph_rows": graph_rows,
        "checkpoint_rows": checkpoint_rows,
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
