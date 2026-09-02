"""Measure marginal-beam coverage gains against sampling cost."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .marginal_candidate_cover import sample_marginal_candidate_cover
from .overlap_cover import audit_cover, patch_budget, sample_edge_target_bridge_cover
from .overlap_stitching import make_cover_example, stitch_patch_predictions
from .run_overlap_cover_audit import FAMILIES, generate_graph


DEFAULT_JSON = Path(
    "tracks/ksvd/results/from_scratch/marginal_beam_sensitivity_audit_20260802.json"
)
DEFAULT_REPORT = Path(
    "tracks/ksvd/results/from_scratch/MARGINAL_BEAM_SENSITIVITY_AUDIT_20260802.md"
)
CELLS = {
    "TARGET": None,
    "B8_R1": (8, 1),
    "B16_R1": (16, 1),
    "B32_R1": (32, 1),
    "B32_R2": (32, 2),
}


def _summary(rows: Sequence[dict[str, Any]], cell: str) -> dict[str, float]:
    selected = [row for row in rows if row["cell"] == cell]
    metrics = (
        "true_edge_coverage",
        "node_pair_coverage",
        "raw_full_adjacency_rmse",
        "sampling_seconds",
    )
    return {
        metric: float(np.mean([row[metric] for row in selected]))
        for metric in metrics
    }


def _relative_reduction(left: float, right: float) -> float:
    return float((left - right) / max(left, 1e-12))


def _is_dominated(name: str, summaries: dict[str, dict[str, float]]) -> bool:
    value = summaries[name]
    for other_name, candidate in summaries.items():
        if other_name == name:
            continue
        weak = (
            candidate["sampling_seconds"] <= value["sampling_seconds"] + 1e-12
            and candidate["raw_full_adjacency_rmse"]
            <= value["raw_full_adjacency_rmse"] + 1e-12
            and candidate["true_edge_coverage"]
            >= value["true_edge_coverage"] - 1e-12
            and candidate["node_pair_coverage"]
            >= value["node_pair_coverage"] - 1e-12
        )
        strict = (
            candidate["sampling_seconds"] < value["sampling_seconds"] - 1e-12
            or candidate["raw_full_adjacency_rmse"]
            < value["raw_full_adjacency_rmse"] - 1e-12
            or candidate["true_edge_coverage"]
            > value["true_edge_coverage"] + 1e-12
            or candidate["node_pair_coverage"]
            > value["node_pair_coverage"] + 1e-12
        )
        if weak and strict:
            return True
    return False


def classify(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summaries = {cell: _summary(rows, cell) for cell in CELLS}
    target = summaries["TARGET"]
    effects = {}
    eligible = []
    for cell in CELLS:
        if cell == "TARGET":
            continue
        summary = summaries[cell]
        edge_gain = summary["true_edge_coverage"] - target["true_edge_coverage"]
        rmse_gain = _relative_reduction(
            target["raw_full_adjacency_rmse"], summary["raw_full_adjacency_rmse"]
        )
        passed = edge_gain >= 0.08 and rmse_gain >= 0.20
        effects[cell] = {
            "edge_coverage_gain": float(edge_gain),
            "pair_coverage_gain": float(
                summary["node_pair_coverage"] - target["node_pair_coverage"]
            ),
            "raw_full_rmse_reduction": rmse_gain,
            "passed_effect_gate": passed,
        }
        if passed:
            eligible.append(cell)
    invariant_gate = all(
        row["connected_rate"] == 1.0
        and row["continuous_fraction"] == 1.0
        and row["overlap_exact"]
        for row in rows
    )
    selected = None
    if eligible:
        minimum_time = min(summaries[cell]["sampling_seconds"] for cell in eligible)
        near_fastest = [
            cell
            for cell in eligible
            if summaries[cell]["sampling_seconds"] <= minimum_time * 1.05
        ]
        selected = max(
            near_fastest, key=lambda cell: summaries[cell]["true_edge_coverage"]
        )
    pareto = [cell for cell in CELLS if not _is_dominated(cell, summaries)]
    if not invariant_gate:
        label = "FAIL_BEAM_SENSITIVITY_INVARIANTS"
    elif selected is None:
        label = "NO_STABLE_BEAM_KNEE"
    elif selected in ("B8_R1", "B16_R1"):
        label = "SMALL_BEAM_SUFFICIENT"
    else:
        label = "BEAM32_REQUIRED"
    return {
        "classification": label,
        "invariant_gate": invariant_gate,
        "selected_knee_cell": selected,
        "summaries": summaries,
        "effects": effects,
        "pareto_cells": pareto,
    }


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def render_report(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    lines = [
        "# Marginal candidate beam：cost–coverage sensitivity 审计",
        "",
        "> 日期：2026-08-02  ",
        "> s10/o3/m1.5；72 graphs；wall-clock sampling cost。",
        "",
        "## 1. 判定",
        "",
        f"**{decision['classification']}**",
        "",
        f"Selected knee：`{decision['selected_knee_cell']}`。",
        "",
        "## 2. Cells",
        "",
        "| cell | sec/graph | edge cover | pair cover | RAW full RMSE | edge gain | RMSE reduction | effect gate | Pareto |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in CELLS:
        summary = decision["summaries"][cell]
        effect = decision["effects"].get(cell)
        lines.append(
            f"| {cell} | {_fmt(summary['sampling_seconds'])} | "
            f"{_fmt(summary['true_edge_coverage'])} | "
            f"{_fmt(summary['node_pair_coverage'])} | "
            f"{_fmt(summary['raw_full_adjacency_rmse'])} | "
            f"{_fmt(effect['edge_coverage_gain']) if effect else '-'} | "
            f"{_fmt(effect['raw_full_rmse_reduction']) if effect else '-'} | "
            f"{effect['passed_effect_gate'] if effect else '-'} | "
            f"{cell in decision['pareto_cells']} |"
        )
    lines.extend(
        [
            "",
            f"Pareto cells：`{decision['pareto_cells']}`。",
            "",
            "## 3. 边界",
            "",
            "- wall-clock 只在当前机器和实现内比较，不外推绝对部署延迟。",
            "- sensitivity 只回答 beam search effort；不重新扫描 patch size、overlap 或 KSVD capacity。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-bank-seed", type=int, default=810001)
    parser.add_argument("--cover-seed", type=int, default=900101)
    parser.add_argument("--n-nodes", type=int, default=50)
    parser.add_argument("--degrees", type=int, nargs="+", default=[15, 20, 25])
    parser.add_argument("--graphs-per-cell", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=10)
    parser.add_argument("--target-overlap", type=int, default=3)
    parser.add_argument("--multiplier", type=float, default=1.5)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    sequences = np.random.SeedSequence(args.graph_bank_seed).spawn(
        len(FAMILIES) * len(args.degrees) * args.graphs_per_cell
    )
    rows = []
    graph_index = 0
    for family in FAMILIES:
        for degree in args.degrees:
            for _replicate in range(args.graphs_per_cell):
                graph_seed = int(
                    sequences[graph_index].generate_state(1, dtype=np.uint32)[0]
                )
                adjacency = generate_graph(family, args.n_nodes, degree, graph_seed)
                budget = patch_budget(
                    adjacency,
                    patch_size=args.patch_size,
                    target_overlap=args.target_overlap,
                    edge_capacity_multiplier=args.multiplier,
                )
                for cell_index, (cell, parameters) in enumerate(CELLS.items()):
                    seed = int(
                        np.random.SeedSequence(
                            [args.cover_seed, graph_index, cell_index]
                        ).generate_state(1, dtype=np.uint32)[0]
                    )
                    started = time.perf_counter()
                    if parameters is None:
                        cover = sample_edge_target_bridge_cover(
                            adjacency,
                            np.random.default_rng(seed),
                            n_patches=budget,
                            patch_size=args.patch_size,
                            target_overlap=args.target_overlap,
                        )
                    else:
                        beam, restarts = parameters
                        cover = sample_marginal_candidate_cover(
                            adjacency,
                            np.random.default_rng(seed),
                            n_patches=budget,
                            patch_size=args.patch_size,
                            target_overlap=args.target_overlap,
                            retained_beam=beam,
                            candidate_restarts=restarts,
                        )
                    elapsed = time.perf_counter() - started
                    audit = audit_cover(adjacency, cover)
                    example = make_cover_example(
                        graph_index, family, degree, adjacency, cover
                    )
                    raw = stitch_patch_predictions(example, example.patch_vectors)
                    rows.append(
                        {
                            "graph_index": graph_index,
                            "family": family,
                            "target_degree": degree,
                            "cell": cell,
                            "sampling_seconds": float(elapsed),
                            "true_edge_coverage": audit["true_edge_coverage"],
                            "node_pair_coverage": audit["node_pair_coverage"],
                            "raw_full_adjacency_rmse": raw[
                                "full_adjacency_rmse"
                            ],
                            "connected_rate": audit["patch_connected_rate"],
                            "continuous_fraction": audit[
                                "continuous_transition_fraction"
                            ],
                            "overlap_exact": all(
                                value == args.target_overlap
                                for value in audit["transition_overlap_exact"]
                            ),
                        }
                    )
                print(f"graph={graph_index} family={family} degree={degree}", flush=True)
                graph_index += 1
    decision = classify(rows)
    payload = {
        "protocol": "marginal-beam-sensitivity-audit-v0-20260802",
        "config": {
            "graph_bank_seed": args.graph_bank_seed,
            "cover_seed": args.cover_seed,
            "families": list(FAMILIES),
            "n_nodes": args.n_nodes,
            "target_degrees": args.degrees,
            "graphs_per_family_degree": args.graphs_per_cell,
            "patch_size": args.patch_size,
            "target_overlap": args.target_overlap,
            "multiplier": args.multiplier,
            "cells": {key: value for key, value in CELLS.items()},
            "labels_used": False,
        },
        "rows": rows,
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
