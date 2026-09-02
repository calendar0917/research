"""Audit continuous-cover hyperparameters on coverage/cost Pareto axes."""
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .overlap_cover import audit_cover, patch_budget, sample_edge_target_bridge_cover
from .overlap_stitching import make_cover_example, stitch_patch_predictions
from .run_overlap_cover_audit import FAMILIES, generate_graph


DEFAULT_JSON = Path(
    "tracks/ksvd/results/from_scratch/patch_chain_cover_pareto_audit_20260802.json"
)
DEFAULT_REPORT = Path(
    "tracks/ksvd/results/from_scratch/PATCH_CHAIN_COVER_PARETO_AUDIT_20260802.md"
)
OVERLAPS = {8: (2, 4, 6), 10: (3, 5, 7), 12: (4, 6, 8)}


def _mean_rows(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    metrics = (
        "patch_count",
        "raw_pair_slots",
        "node_coverage",
        "true_edge_coverage",
        "node_pair_coverage",
        "bridge_length_mean",
        "new_pairs_per_patch_mean",
        "edge_observation_multiplicity_cv",
        "raw_full_adjacency_rmse",
        "raw_full_edge_recall",
    )
    return {
        metric: float(np.mean([float(row[metric]) for row in rows]))
        for metric in metrics
    }


def _is_dominated(cell: dict[str, Any], cells: Sequence[dict[str, Any]]) -> bool:
    if not cell["hard_gate"]:
        return True
    value = cell["summary"]
    for other in cells:
        if other is cell or not other["hard_gate"]:
            continue
        candidate = other["summary"]
        weak = (
            candidate["patch_count"] <= value["patch_count"] + 1e-12
            and candidate["raw_pair_slots"] <= value["raw_pair_slots"] + 1e-12
            and candidate["raw_full_adjacency_rmse"]
            <= value["raw_full_adjacency_rmse"] + 1e-12
            and candidate["true_edge_coverage"]
            >= value["true_edge_coverage"] - 1e-12
            and candidate["node_pair_coverage"]
            >= value["node_pair_coverage"] - 1e-12
        )
        strict = (
            candidate["patch_count"] < value["patch_count"] - 1e-12
            or candidate["raw_pair_slots"] < value["raw_pair_slots"] - 1e-12
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


def classify(cells: list[dict[str, Any]]) -> dict[str, Any]:
    for cell in cells:
        cell["pareto_nondominated"] = not _is_dominated(cell, cells)
    current = next(
        (
            cell
            for cell in cells
            if cell["patch_size"] == 10
            and cell["target_overlap"] == 5
            and cell["edge_capacity_multiplier"] == 1.5
        ),
        None,
    )
    pareto = [cell["cell_id"] for cell in cells if cell["pareto_nondominated"]]
    dominators = []
    if current is not None and not current["pareto_nondominated"]:
        current_value = current["summary"]
        for other in cells:
            if other is current or not other["hard_gate"]:
                continue
            candidate = other["summary"]
            if (
                candidate["patch_count"] <= current_value["patch_count"] + 1e-12
                and candidate["raw_pair_slots"]
                <= current_value["raw_pair_slots"] + 1e-12
                and candidate["raw_full_adjacency_rmse"]
                <= current_value["raw_full_adjacency_rmse"] + 1e-12
                and candidate["true_edge_coverage"]
                >= current_value["true_edge_coverage"] - 1e-12
                and candidate["node_pair_coverage"]
                >= current_value["node_pair_coverage"] - 1e-12
            ):
                dominators.append(other["cell_id"])
    return {
        "current_cell_id": current["cell_id"] if current is not None else None,
        "current_hard_gate": current["hard_gate"] if current is not None else None,
        "current_pareto_nondominated": current["pareto_nondominated"]
        if current is not None
        else None,
        "current_summary": current["summary"] if current is not None else None,
        "pareto_cell_ids": pareto,
        "current_dominator_cell_ids": dominators,
        "feasible_cell_count": int(sum(cell["hard_gate"] for cell in cells)),
        "total_cell_count": len(cells),
        "classification": "GRID_COMPLETED_WITHOUT_REGISTERED_CURRENT"
        if current is None
        else (
            "CURRENT_COVER_PARETO_SUPPORTED"
            if current["pareto_nondominated"]
            else "REVISE_COVER_CONFIGURATION"
        ),
    }


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def render_report(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    lines = [
        "# Patch-chain cover hyperparameter Pareto 审计",
        "",
        "> 日期：2026-08-02  ",
        "> 72 graphs；无 labels；27 个预注册 cover cells。",
        "",
        "## 1. 判定",
        "",
        f"**{decision['classification']}**",
        "",
        f"当前 `10/5/1.5` nondominated：`{decision['current_pareto_nondominated']}`；feasible cells：`{decision['feasible_cell_count']}/{decision['total_cell_count']}`。",
        "",
        "## 2. 全部 cells",
        "",
        "| cell | feasible | patches | raw pair slots | edge cover | pair cover | RAW full RMSE | bridge | new pairs/patch | Pareto |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in payload["cells"]:
        summary = cell["summary"]
        lines.append(
            f"| {cell['cell_id']} | {_fmt(cell['feasible_graph_fraction'])} | "
            f"{_fmt(summary['patch_count'])} | "
            f"{_fmt(summary['raw_pair_slots'])} | "
            f"{_fmt(summary['true_edge_coverage'])} | "
            f"{_fmt(summary['node_pair_coverage'])} | "
            f"{_fmt(summary['raw_full_adjacency_rmse'])} | "
            f"{_fmt(summary['bridge_length_mean'])} | "
            f"{_fmt(summary['new_pairs_per_patch_mean'])} | "
            f"{cell['pareto_nondominated']} |"
        )
    lines.extend(
        [
            "",
            "## 3. Current cell",
            "",
            f"`{decision['current_summary']}`",
            "",
            f"Pareto cells：`{decision['pareto_cell_ids']}`。",
            "",
            f"Current dominators：`{decision['current_dominator_cell_ids']}`。",
            "",
            "## 4. 边界",
            "",
            "- raw pair slots 是 patch count × patch pair capacity；它衡量未压缩观察成本，不等于最终 sparse-code 存储。",
            "- Pareto 不选单一冠军，只排除在成本和 coverage/error 上同时被支配的 cells。",
            "- sampler failure 计入 feasibility；不因高 coverage 子集而忽略失败图。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-bank-seed", type=int, default=810001)
    parser.add_argument("--cover-seed", type=int, default=870101)
    parser.add_argument("--n-nodes", type=int, default=50)
    parser.add_argument("--degrees", type=int, nargs="+", default=[15, 20, 25])
    parser.add_argument("--graphs-per-cell", type=int, default=8)
    parser.add_argument("--patch-sizes", type=int, nargs="+", default=[8, 10, 12])
    parser.add_argument(
        "--multipliers", type=float, nargs="+", default=[1.0, 1.5, 2.0]
    )
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    graphs = []
    graph_sequences = np.random.SeedSequence(args.graph_bank_seed).spawn(
        len(FAMILIES) * len(args.degrees) * args.graphs_per_cell
    )
    graph_index = 0
    for family in FAMILIES:
        for degree in args.degrees:
            for _replicate in range(args.graphs_per_cell):
                seed = int(
                    graph_sequences[graph_index].generate_state(1, dtype=np.uint32)[0]
                )
                graphs.append(
                    (graph_index, family, degree, generate_graph(family, args.n_nodes, degree, seed))
                )
                graph_index += 1

    cells = []
    for patch_size, multiplier in product(args.patch_sizes, args.multipliers):
        if patch_size not in OVERLAPS:
            raise ValueError(f"no frozen overlaps registered for patch_size={patch_size}")
        for overlap in OVERLAPS[patch_size]:
            rows = []
            failures = []
            for index, family, degree, adjacency in graphs:
                budget = patch_budget(
                    adjacency,
                    patch_size=patch_size,
                    target_overlap=overlap,
                    edge_capacity_multiplier=multiplier,
                )
                seed = int(
                    np.random.SeedSequence(
                        [args.cover_seed, index, patch_size, overlap, int(multiplier * 100)]
                    ).generate_state(1, dtype=np.uint32)[0]
                )
                try:
                    cover = sample_edge_target_bridge_cover(
                        adjacency,
                        np.random.default_rng(seed),
                        n_patches=budget,
                        patch_size=patch_size,
                        target_overlap=overlap,
                    )
                except RuntimeError as error:
                    failures.append(
                        {
                            "graph_index": index,
                            "family": family,
                            "target_degree": degree,
                            "error": str(error),
                        }
                    )
                    continue
                audit = audit_cover(adjacency, cover)
                example = make_cover_example(index, family, degree, adjacency, cover)
                raw = stitch_patch_predictions(example, example.patch_vectors)
                rows.append(
                    {
                        "graph_index": index,
                        "family": family,
                        "target_degree": degree,
                        "patch_count": audit["patch_count"],
                        "raw_pair_slots": audit["patch_count"]
                        * patch_size
                        * (patch_size - 1)
                        // 2,
                        "node_coverage": audit["node_coverage"],
                        "true_edge_coverage": audit["true_edge_coverage"],
                        "node_pair_coverage": audit["node_pair_coverage"],
                        "bridge_length_mean": audit["bridge_length_mean"],
                        "new_pairs_per_patch_mean": audit[
                            "new_pairs_per_patch_mean"
                        ],
                        "edge_observation_multiplicity_cv": audit[
                            "edge_observation_multiplicity_cv"
                        ],
                        "raw_full_adjacency_rmse": raw["full_adjacency_rmse"],
                        "raw_full_edge_recall": raw["full_edge_recall"],
                        "continuous_fraction": audit[
                            "continuous_transition_fraction"
                        ],
                        "connected_rate": audit["patch_connected_rate"],
                        "target_hit_rate": audit["target_edge_hit_rate"],
                        "overlap_exact": all(
                            value == overlap
                            for value in audit["transition_overlap_exact"]
                        ),
                    }
                )
            feasible_fraction = len(rows) / len(graphs)
            hard_gate = bool(
                feasible_fraction == 1.0
                and all(row["continuous_fraction"] == 1.0 for row in rows)
                and all(row["connected_rate"] == 1.0 for row in rows)
                and all(row["target_hit_rate"] == 1.0 for row in rows)
                and all(row["overlap_exact"] for row in rows)
            )
            empty_summary = {
                key: float("nan")
                for key in (
                    "patch_count",
                    "raw_pair_slots",
                    "node_coverage",
                    "true_edge_coverage",
                    "node_pair_coverage",
                    "bridge_length_mean",
                    "new_pairs_per_patch_mean",
                    "edge_observation_multiplicity_cv",
                    "raw_full_adjacency_rmse",
                    "raw_full_edge_recall",
                )
            }
            summary = _mean_rows(rows) if rows else empty_summary
            family_summaries = {
                family: _mean_rows([row for row in rows if row["family"] == family])
                if any(row["family"] == family for row in rows)
                else empty_summary
                for family in FAMILIES
            }
            cell_id = f"s{patch_size}_o{overlap}_m{multiplier:g}"
            cells.append(
                {
                    "cell_id": cell_id,
                    "patch_size": patch_size,
                    "target_overlap": overlap,
                    "edge_capacity_multiplier": multiplier,
                    "feasible_graph_fraction": feasible_fraction,
                    "hard_gate": hard_gate,
                    "summary": summary,
                    "family_summaries": family_summaries,
                    "failures": failures,
                    "rows": rows,
                }
            )
            print(
                f"cell={cell_id} feasible={feasible_fraction:.3f} "
                f"patches={summary['patch_count']:.2f} "
                f"edge={summary['true_edge_coverage']:.4f} "
                f"pair={summary['node_pair_coverage']:.4f} "
                f"rmse={summary['raw_full_adjacency_rmse']:.4f}",
                flush=True,
            )

    decision = classify(cells)
    payload = {
        "protocol": "patch-chain-cover-pareto-audit-v0-20260802",
        "config": {
            "graph_bank_seed": args.graph_bank_seed,
            "cover_seed": args.cover_seed,
            "families": list(FAMILIES),
            "n_nodes": args.n_nodes,
            "target_degrees": args.degrees,
            "graphs_per_family_degree": args.graphs_per_cell,
            "patch_sizes": args.patch_sizes,
            "overlaps": {str(key): list(value) for key, value in OVERLAPS.items()},
            "multipliers": args.multipliers,
            "labels_used": False,
        },
        "cells": cells,
        "decision": decision,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(render_report(payload), encoding="utf-8")
    print(f"decision={decision['classification']}")
    print(f"wrote {args.json}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
