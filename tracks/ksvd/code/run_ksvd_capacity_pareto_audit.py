"""Audit KSVD dictionary/sparsity capacity on fixed continuous covers."""
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .overlap_cover import patch_budget, sample_edge_target_bridge_cover
from .overlap_stitching import CoverExample, make_cover_example
from .run_ksvd_stitched_reconstruction_audit import _run_fold, classify
from .run_overlap_cover_audit import FAMILIES, generate_graph


DEFAULT_JSON = Path(
    "tracks/ksvd/results/from_scratch/ksvd_capacity_pareto_audit_20260802.json"
)
DEFAULT_REPORT = Path(
    "tracks/ksvd/results/from_scratch/KSVD_CAPACITY_PARETO_AUDIT_20260802.md"
)


def _compact_fold(fold: dict[str, Any]) -> dict[str, Any]:
    result = dict(fold)
    result["stages"] = {
        stage: {"summary": values["summary"]}
        for stage, values in fold["stages"].items()
    }
    return result


def _is_dominated(cell: dict[str, Any], cells: Sequence[dict[str, Any]]) -> bool:
    for other in cells:
        if other is cell:
            continue
        weak = (
            other["dictionary_scalar_count"] <= cell["dictionary_scalar_count"]
            and other["mean_code_scalars_per_graph"]
            <= cell["mean_code_scalars_per_graph"] + 1e-12
            and other["final_observed_rmse"] <= cell["final_observed_rmse"] + 1e-12
        )
        strict = (
            other["dictionary_scalar_count"] < cell["dictionary_scalar_count"]
            or other["mean_code_scalars_per_graph"]
            < cell["mean_code_scalars_per_graph"] - 1e-12
            or other["final_observed_rmse"] < cell["final_observed_rmse"] - 1e-12
        )
        if weak and strict:
            return True
    return False


def classify_cells(cells: list[dict[str, Any]]) -> dict[str, Any]:
    main_cells = [cell for cell in cells if cell["updates"] == 25]
    for cell in cells:
        cell["compression_pareto_nondominated"] = (
            not _is_dominated(cell, main_cells) if cell in main_cells else None
        )
    current = next(
        cell
        for cell in cells
        if cell["n_atoms"] == 24 and cell["sparsity"] == 3 and cell["updates"] == 25
    )
    pareto = [
        cell["cell_id"]
        for cell in main_cells
        if cell["compression_pareto_nondominated"]
    ]
    same_or_lower_cost_better = [
        cell["cell_id"]
        for cell in main_cells
        if cell is not current
        and cell["dictionary_scalar_count"] <= current["dictionary_scalar_count"]
        and cell["mean_code_scalars_per_graph"]
        <= current["mean_code_scalars_per_graph"] + 1e-12
        and cell["final_observed_rmse"] < current["final_observed_rmse"] - 1e-12
    ]
    convergence = {
        str(cell["updates"]): cell["final_observed_rmse"]
        for cell in cells
        if cell["n_atoms"] == 24 and cell["sparsity"] == 3
    }
    return {
        "current_cell_id": current["cell_id"],
        "current_pareto_nondominated": current["compression_pareto_nondominated"],
        "pareto_cell_ids": pareto,
        "same_or_lower_cost_better_cell_ids": same_or_lower_cost_better,
        "convergence_rmse_by_updates": convergence,
        "classification": "CURRENT_KSVD_CAPACITY_PARETO_SUPPORTED"
        if current["compression_pareto_nondominated"]
        else "REVISE_KSVD_CAPACITY",
    }


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def render_report(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    lines = [
        "# KSVD capacity–compression Pareto 审计",
        "",
        "> 日期：2026-08-02  ",
        "> fixed `s=10,o=5,m=1.5` cover；3-fold graph isolation。",
        "",
        "## 1. 判定",
        "",
        f"**{decision['classification']}**",
        "",
        f"当前 `K24/T3/u25` nondominated：`{decision['current_pareto_nondominated']}`。",
        "",
        "## 2. Capacity cells",
        "",
        "| cell | dict scalars | code scalars/graph | patch err | observed RMSE | full RMSE | F1 | recall | disagree | nondead/max share | Pareto |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in payload["cells"]:
        lines.append(
            f"| {cell['cell_id']} | {cell['dictionary_scalar_count']} | "
            f"{_fmt(cell['mean_code_scalars_per_graph'])} | "
            f"{_fmt(cell['final_patch_error'])} | "
            f"{_fmt(cell['final_observed_rmse'])} | "
            f"{_fmt(cell['final_full_rmse'])} | "
            f"{_fmt(cell['final_observed_f1'])} | "
            f"{_fmt(cell['final_full_recall'])} | "
            f"{_fmt(cell['final_disagreement'])} | "
            f"{_fmt(cell['mean_nondead_atoms'])}/"
            f"{_fmt(cell['mean_maximum_activation_share'])} | "
            f"{cell['compression_pareto_nondominated']} |"
        )
    lines.extend(
        [
            "",
            "## 3. Registered comparisons",
            "",
            f"Pareto cells：`{decision['pareto_cell_ids']}`。",
            "",
            f"Same-or-lower cost but lower RMSE：`{decision['same_or_lower_cost_better_cell_ids']}`。",
            "",
            f"K24/T3 convergence：`{decision['convergence_rmse_by_updates']}`。",
            "",
            "## 4. 边界",
            "",
            "- dictionary scalars 是共享模型成本；code scalars/graph 是每图稀疏表示成本，两者不混成一个任意加权总数。",
            "- 更大 K/T 若只靠增加成本降低误差，只能成为另一个 Pareto 点，不能自动替换当前配置。",
            "- PCA rank 与 T 匹配，但 PCA 不稀疏；它仍只作为重构对照。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-bank-seed", type=int, default=810001)
    parser.add_argument("--cover-seed", type=int, default=830101)
    parser.add_argument("--n-nodes", type=int, default=50)
    parser.add_argument("--degrees", type=int, nargs="+", default=[15, 20, 25])
    parser.add_argument("--graphs-per-cell", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=10)
    parser.add_argument("--target-overlap", type=int, default=5)
    parser.add_argument("--edge-capacity-multiplier", type=float, default=1.5)
    parser.add_argument("--atom-grid", type=int, nargs="+", default=[16, 24, 32])
    parser.add_argument("--sparsity-grid", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--updates", type=int, default=25)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    examples: list[CoverExample] = []
    graph_sequences = np.random.SeedSequence(args.graph_bank_seed).spawn(
        len(FAMILIES) * len(args.degrees) * args.graphs_per_cell
    )
    graph_index = 0
    for family in FAMILIES:
        for degree in args.degrees:
            for _replicate in range(args.graphs_per_cell):
                graph_seed = int(
                    graph_sequences[graph_index].generate_state(1, dtype=np.uint32)[0]
                )
                adjacency = generate_graph(family, args.n_nodes, degree, graph_seed)
                budget = patch_budget(
                    adjacency,
                    patch_size=args.patch_size,
                    target_overlap=args.target_overlap,
                    edge_capacity_multiplier=args.edge_capacity_multiplier,
                )
                sampler_seed = int(
                    np.random.SeedSequence(
                        [args.cover_seed, graph_index, 5501]
                    ).generate_state(1, dtype=np.uint32)[0]
                )
                cover = sample_edge_target_bridge_cover(
                    adjacency,
                    np.random.default_rng(sampler_seed),
                    n_patches=budget,
                    patch_size=args.patch_size,
                    target_overlap=args.target_overlap,
                )
                examples.append(
                    make_cover_example(graph_index, family, degree, adjacency, cover)
                )
                graph_index += 1

    configurations = sorted(
        {
            *(
                (atoms, sparsity, args.updates)
                for atoms, sparsity in product(args.atom_grid, args.sparsity_grid)
            ),
            (24, 3, 10),
            (24, 3, 25),
            (24, 3, 50),
        }
    )
    cells = []
    mean_patch_count = float(
        np.mean([example.patch_vectors.shape[0] for example in examples])
    )
    patch_dimension = args.patch_size * (args.patch_size - 1) // 2
    for atoms, sparsity, updates in configurations:
        folds = []
        for fold_index in range(3):
            fold = _run_fold(
                examples,
                fold_index,
                n_atoms=atoms,
                sparsity=sparsity,
                minimum_sparsity=1,
                n_iterations=updates,
                pca_rank=sparsity,
            )
            folds.append(_compact_fold(fold))
        decision = classify(folds)
        final = decision["mean_stages"]["final"]
        train_health = [fold["dictionary_health"]["final"]["train"] for fold in folds]
        cell_id = f"K{atoms}_T{sparsity}_u{updates}"
        cell = {
            "cell_id": cell_id,
            "n_atoms": atoms,
            "sparsity": sparsity,
            "updates": updates,
            "dictionary_scalar_count": patch_dimension * atoms,
            "mean_code_scalars_per_graph": mean_patch_count * sparsity,
            "final_patch_error": final["patch_relative_error"],
            "final_observed_rmse": final["observed_pair_rmse"],
            "final_full_rmse": final["full_adjacency_rmse"],
            "final_observed_f1": final["observed_edge_f1"],
            "final_full_recall": final["full_edge_recall"],
            "final_disagreement": final[
                "repeated_pair_disagreement_std_mean"
            ],
            "mean_nondead_atoms": float(
                np.mean([health["nondead_atom_count"] for health in train_health])
            ),
            "mean_maximum_activation_share": float(
                np.mean(
                    [health["maximum_activation_share"] for health in train_health]
                )
            ),
            "mean_patch_init_final_reduction": decision[
                "mean_patch_relative_reduction"
            ],
            "optimization_gate": decision["optimization_gate"],
            "beats_pca3": decision["stitched_checks"]["beats_pca3_rmse"],
            "folds": folds,
            "decision": decision,
        }
        cells.append(cell)
        print(
            f"cell={cell_id} patch={cell['final_patch_error']:.4f} "
            f"observed_rmse={cell['final_observed_rmse']:.4f} "
            f"full_rmse={cell['final_full_rmse']:.4f} "
            f"nondead={cell['mean_nondead_atoms']:.1f}",
            flush=True,
        )

    decision = classify_cells(cells)
    payload = {
        "protocol": "ksvd-capacity-pareto-audit-v0-20260802",
        "config": {
            "graph_bank_seed": args.graph_bank_seed,
            "cover_seed": args.cover_seed,
            "families": list(FAMILIES),
            "n_nodes": args.n_nodes,
            "target_degrees": args.degrees,
            "graphs_per_family_degree": args.graphs_per_cell,
            "patch_size": args.patch_size,
            "target_overlap": args.target_overlap,
            "edge_capacity_multiplier": args.edge_capacity_multiplier,
            "atom_grid": args.atom_grid,
            "sparsity_grid": args.sparsity_grid,
            "updates": args.updates,
            "labels_used": False,
        },
        "cells": cells,
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
