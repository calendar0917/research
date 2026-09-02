"""Matched KSVD follow-up for RAW-Pareto cover configurations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .overlap_cover import patch_budget, sample_edge_target_bridge_cover
from .overlap_stitching import CoverExample, make_cover_example
from .run_ksvd_stitched_reconstruction_audit import _run_fold, classify
from .run_overlap_cover_audit import FAMILIES, generate_graph


DEFAULT_JSON = Path(
    "tracks/ksvd/results/from_scratch/revised_cover_compression_audit_20260802.json"
)
DEFAULT_REPORT = Path(
    "tracks/ksvd/results/from_scratch/REVISED_COVER_COMPRESSION_AUDIT_20260802.md"
)
BRANCHES = {
    "current": (10, 5, 1.5),
    "lower_overlap": (10, 3, 1.5),
    "larger_patch": (12, 4, 1.5),
}


def _compact_fold(fold: dict[str, Any]) -> dict[str, Any]:
    result = dict(fold)
    result["stages"] = {
        stage: {"summary": values["summary"]}
        for stage, values in fold["stages"].items()
    }
    return result


def _relative_reduction(left: float, right: float) -> float:
    return float((left - right) / max(left, 1e-12))


def classify_branches(branches: dict[str, Any]) -> dict[str, Any]:
    current = branches["current"]
    lower = branches["lower_overlap"]
    larger = branches["larger_patch"]
    current_final = current["decision"]["mean_stages"]["final"]
    lower_final = lower["decision"]["mean_stages"]["final"]
    larger_final = larger["decision"]["mean_stages"]["final"]
    full_gain = _relative_reduction(
        current_final["full_adjacency_rmse"],
        lower_final["full_adjacency_rmse"],
    )
    observed_change = _relative_reduction(
        current_final["observed_pair_rmse"],
        lower_final["observed_pair_rmse"],
    )
    lower_checks = {
        "patch_count_not_higher": lower["mean_patch_count"]
        <= current["mean_patch_count"] + 1e-12,
        "dictionary_not_larger": lower["dictionary_scalar_count"]
        <= current["dictionary_scalar_count"],
        "code_not_larger": lower["mean_code_scalars_per_graph"]
        <= current["mean_code_scalars_per_graph"] + 1e-12,
        "full_rmse_reduction_at_least_002": full_gain >= 0.02,
        "observed_rmse_not_worse_by_001": observed_change >= -0.01,
        "full_recall_not_worse": lower_final["full_edge_recall"]
        >= current_final["full_edge_recall"] - 1e-12,
        "full_f1_not_worse": lower_final["full_edge_f1"]
        >= current_final["full_edge_f1"] - 1e-12,
        "raw_gates": current["decision"]["raw_gate"]
        and lower["decision"]["raw_gate"],
        "patch_final_better_all_folds": all(
            check["patch_final_better"]
            for check in lower["decision"]["optimization_checks_by_fold"]
        ),
    }
    invariant_gate = all(
        branch["decision"]["raw_gate"] for branch in branches.values()
    )
    lower_gate = all(lower_checks.values())
    larger_tradeoff = {
        "full_rmse_reduction_vs_current": _relative_reduction(
            current_final["full_adjacency_rmse"],
            larger_final["full_adjacency_rmse"],
        ),
        "code_scalar_reduction_vs_current": _relative_reduction(
            current["mean_code_scalars_per_graph"],
            larger["mean_code_scalars_per_graph"],
        ),
        "dictionary_scalar_increase_vs_current": float(
            (larger["dictionary_scalar_count"] - current["dictionary_scalar_count"])
            / current["dictionary_scalar_count"]
        ),
    }
    if not invariant_gate:
        label = "FAIL_REVISED_COVER_INVARIANTS"
    elif lower_gate:
        label = "ADOPT_LOWER_OVERLAP_COVER_V2"
    else:
        label = "RAW_COVER_GAIN_LOST_AFTER_COMPRESSION"
    return {
        "classification": label,
        "invariant_gate": invariant_gate,
        "lower_overlap_gate": lower_gate,
        "lower_overlap_checks": lower_checks,
        "lower_overlap_full_rmse_reduction": full_gain,
        "lower_overlap_observed_rmse_reduction": observed_change,
        "larger_patch_tradeoff": larger_tradeoff,
    }


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def render_report(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    lines = [
        "# Revised cover configurations：matched KSVD 审计",
        "",
        "> 日期：2026-08-02  ",
        "> RAW Pareto follow-up；K24/T3/u25；3-fold isolation。",
        "",
        "## 1. 判定",
        "",
        f"**{decision['classification']}**",
        "",
        "## 2. Costs and FINAL metrics",
        "",
        "| branch | s/o/m | patches | dict scalars | code scalars/graph | raw pair slots | patch err | observed RMSE | full RMSE | full recall/F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for branch_name in ("current", "lower_overlap", "larger_patch"):
        branch = payload["branches"][branch_name]
        final = branch["decision"]["mean_stages"]["final"]
        lines.append(
            f"| {branch_name} | {branch['patch_size']}/{branch['target_overlap']}/"
            f"{branch['multiplier']} | {_fmt(branch['mean_patch_count'])} | "
            f"{branch['dictionary_scalar_count']} | "
            f"{_fmt(branch['mean_code_scalars_per_graph'])} | "
            f"{_fmt(branch['mean_raw_pair_slots'])} | "
            f"{_fmt(final['patch_relative_error'])} | "
            f"{_fmt(final['observed_pair_rmse'])} | "
            f"{_fmt(final['full_adjacency_rmse'])} | "
            f"{_fmt(final['full_edge_recall'])}/"
            f"{_fmt(final['full_edge_f1'])} |"
        )
    lines.extend(
        [
            "",
            "## 3. Registered lower-overlap gate",
            "",
            f"- full RMSE reduction：`{_fmt(decision['lower_overlap_full_rmse_reduction'])}`；",
            f"- observed RMSE reduction：`{_fmt(decision['lower_overlap_observed_rmse_reduction'])}`；",
            f"- checks：`{decision['lower_overlap_checks']}`；",
            f"- gate：`{decision['lower_overlap_gate']}`。",
            "",
            "## 4. Larger-patch tradeoff",
            "",
            f"`{decision['larger_patch_tradeoff']}`",
            "",
            "## 5. 边界",
            "",
            "- lower_overlap 与 current 的 patch dimension、dictionary cost 和 mean patch count匹配，因此可以直接比较。",
            "- larger_patch 减少 code length，但增加 dictionary dimension；保留为独立 Pareto 点。",
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
    parser.add_argument("--n-atoms", type=int, default=24)
    parser.add_argument("--sparsity", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    graph_data = []
    sequences = np.random.SeedSequence(args.graph_bank_seed).spawn(
        len(FAMILIES) * len(args.degrees) * args.graphs_per_cell
    )
    graph_index = 0
    for family in FAMILIES:
        for degree in args.degrees:
            for _replicate in range(args.graphs_per_cell):
                seed = int(
                    sequences[graph_index].generate_state(1, dtype=np.uint32)[0]
                )
                graph_data.append(
                    (
                        graph_index,
                        family,
                        degree,
                        generate_graph(family, args.n_nodes, degree, seed),
                    )
                )
                graph_index += 1

    branch_payload = {}
    for branch_name, (patch_size, overlap, multiplier) in BRANCHES.items():
        examples: list[CoverExample] = []
        for index, family, degree, adjacency in graph_data:
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
            cover = sample_edge_target_bridge_cover(
                adjacency,
                np.random.default_rng(seed),
                n_patches=budget,
                patch_size=patch_size,
                target_overlap=overlap,
            )
            examples.append(
                make_cover_example(index, family, degree, adjacency, cover)
            )
        folds = []
        for fold_index in range(3):
            fold = _run_fold(
                examples,
                fold_index,
                n_atoms=args.n_atoms,
                sparsity=args.sparsity,
                minimum_sparsity=1,
                n_iterations=args.iterations,
                pca_rank=args.sparsity,
            )
            folds.append(_compact_fold(fold))
        branch_decision = classify(folds)
        mean_patch_count = float(
            np.mean([example.patch_vectors.shape[0] for example in examples])
        )
        patch_dimension = patch_size * (patch_size - 1) // 2
        branch_payload[branch_name] = {
            "patch_size": patch_size,
            "target_overlap": overlap,
            "multiplier": multiplier,
            "mean_patch_count": mean_patch_count,
            "dictionary_scalar_count": patch_dimension * args.n_atoms,
            "mean_code_scalars_per_graph": mean_patch_count * args.sparsity,
            "mean_raw_pair_slots": mean_patch_count * patch_dimension,
            "folds": folds,
            "decision": branch_decision,
        }
        final = branch_decision["mean_stages"]["final"]
        print(
            f"branch={branch_name} observed/full_rmse="
            f"{final['observed_pair_rmse']:.4f}/{final['full_adjacency_rmse']:.4f} "
            f"recall={final['full_edge_recall']:.4f}",
            flush=True,
        )
    decision = classify_branches(branch_payload)
    payload = {
        "protocol": "revised-cover-compression-audit-v0-20260802",
        "config": {
            "graph_bank_seed": args.graph_bank_seed,
            "cover_seed": args.cover_seed,
            "families": list(FAMILIES),
            "n_nodes": args.n_nodes,
            "target_degrees": args.degrees,
            "graphs_per_family_degree": args.graphs_per_cell,
            "n_atoms": args.n_atoms,
            "sparsity": args.sparsity,
            "iterations": args.iterations,
            "labels_used": False,
        },
        "branches": branch_payload,
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
