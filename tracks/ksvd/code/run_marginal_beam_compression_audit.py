"""Matched KSVD audit for target and marginal-beam continuous covers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .marginal_candidate_cover import sample_marginal_candidate_cover
from .overlap_cover import patch_budget, sample_edge_target_bridge_cover
from .overlap_stitching import CoverExample, make_cover_example
from .run_ksvd_stitched_reconstruction_audit import _run_fold, classify
from .run_overlap_cover_audit import FAMILIES, generate_graph


DEFAULT_JSON = Path(
    "tracks/ksvd/results/from_scratch/marginal_beam_compression_audit_20260802.json"
)
DEFAULT_REPORT = Path(
    "tracks/ksvd/results/from_scratch/MARGINAL_BEAM_COMPRESSION_AUDIT_20260802.md"
)
BRANCHES = {
    "target_o5": ("target", 5),
    "target_o3": ("target", 3),
    "beam_o3": ("beam", 3),
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
    target = branches["target_o3"]["decision"]
    beam = branches["beam_o3"]["decision"]
    target_final = target["mean_stages"]["final"]
    beam_final = beam["mean_stages"]["final"]
    full_gain = _relative_reduction(
        target_final["full_adjacency_rmse"], beam_final["full_adjacency_rmse"]
    )
    observed_gain = _relative_reduction(
        target_final["observed_pair_rmse"], beam_final["observed_pair_rmse"]
    )
    checks = {
        "full_rmse_reduction_at_least_005": full_gain >= 0.05,
        "full_recall_improves": beam_final["full_edge_recall"]
        > target_final["full_edge_recall"],
        "full_f1_improves": beam_final["full_edge_f1"]
        > target_final["full_edge_f1"],
        "observed_rmse_not_worse_by_005": observed_gain >= -0.05,
        "raw_gates": target["raw_gate"] and beam["raw_gate"],
        "patch_final_better_all_folds": all(
            check["patch_final_better"]
            for check in beam["optimization_checks_by_fold"]
        ),
        "beats_own_pca3_full_rmse": beam_final["full_adjacency_rmse"]
        < beam["mean_stages"]["pca3"]["full_adjacency_rmse"],
    }
    invariant_gate = all(
        branch["decision"]["raw_gate"] for branch in branches.values()
    )
    if not invariant_gate:
        label = "FAIL_BEAM_COMPRESSION_INVARIANTS"
    elif all(checks.values()):
        label = "ADOPT_MARGINAL_BEAM_COVER_V2"
    elif beam_final["full_adjacency_rmse"] < target_final["full_adjacency_rmse"]:
        label = "BEAM_RAW_GAIN_SURVIVES_PARTIALLY"
    else:
        label = "BEAM_RAW_GAIN_LOST_AFTER_COMPRESSION"
    return {
        "classification": label,
        "invariant_gate": invariant_gate,
        "beam_gate": all(checks.values()),
        "checks": checks,
        "full_rmse_reduction": full_gain,
        "observed_rmse_reduction": observed_gain,
    }


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def render_report(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    lines = [
        "# Marginal beam cover：matched KSVD compression 审计",
        "",
        "> 日期：2026-08-02  ",
        "> same s=10, K24/T3/u25；3-fold graph isolation。",
        "",
        "## 1. 判定",
        "",
        f"**{decision['classification']}**",
        "",
        "## 2. Fold-balanced means",
        "",
        "| branch | stage | patch err | observed RMSE | full RMSE | observed F1 | full recall/F1 | disagreement |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for branch_name in BRANCHES:
        branch = payload["branches"][branch_name]
        for stage in ("raw", "init", "final", "pca3"):
            mean = branch["decision"]["mean_stages"][stage]
            lines.append(
                f"| {branch_name} | {stage} | "
                f"{_fmt(mean['patch_relative_error'])} | "
                f"{_fmt(mean['observed_pair_rmse'])} | "
                f"{_fmt(mean['full_adjacency_rmse'])} | "
                f"{_fmt(mean['observed_edge_f1'])} | "
                f"{_fmt(mean['full_edge_recall'])}/"
                f"{_fmt(mean['full_edge_f1'])} | "
                f"{_fmt(mean['repeated_pair_disagreement_std_mean'])} |"
            )
    lines.extend(
        [
            "",
            "## 3. Registered gate",
            "",
            f"- full RMSE reduction：`{_fmt(decision['full_rmse_reduction'])}`；",
            f"- observed RMSE reduction：`{_fmt(decision['observed_rmse_reduction'])}`；",
            f"- checks：`{decision['checks']}`；",
            f"- gate：`{decision['beam_gate']}`。",
            "",
            "## 4. 边界",
            "",
            "- beam 偏向 edge-dense patches；必须同时阅读 observed RMSE、pair coverage 和 full graph metrics。",
            "- 本轮不含 completion；unseen pair 仍 zero fill。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-bank-seed", type=int, default=810001)
    parser.add_argument("--cover-seed", type=int, default=890101)
    parser.add_argument("--n-nodes", type=int, default=50)
    parser.add_argument("--degrees", type=int, nargs="+", default=[15, 20, 25])
    parser.add_argument("--graphs-per-cell", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=10)
    parser.add_argument("--multiplier", type=float, default=1.5)
    parser.add_argument("--retained-beam", type=int, default=32)
    parser.add_argument("--candidate-restarts", type=int, default=2)
    parser.add_argument("--n-atoms", type=int, default=24)
    parser.add_argument("--sparsity", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    sequences = np.random.SeedSequence(args.graph_bank_seed).spawn(
        len(FAMILIES) * len(args.degrees) * args.graphs_per_cell
    )
    graph_data = []
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
    for branch_index, (branch_name, (method, overlap)) in enumerate(BRANCHES.items()):
        examples: list[CoverExample] = []
        for index, family, degree, adjacency in graph_data:
            budget = patch_budget(
                adjacency,
                patch_size=args.patch_size,
                target_overlap=overlap,
                edge_capacity_multiplier=args.multiplier,
            )
            seed = int(
                np.random.SeedSequence(
                    [args.cover_seed, index, branch_index]
                ).generate_state(1, dtype=np.uint32)[0]
            )
            if method == "target":
                cover = sample_edge_target_bridge_cover(
                    adjacency,
                    np.random.default_rng(seed),
                    n_patches=budget,
                    patch_size=args.patch_size,
                    target_overlap=overlap,
                )
            else:
                cover = sample_marginal_candidate_cover(
                    adjacency,
                    np.random.default_rng(seed),
                    n_patches=budget,
                    patch_size=args.patch_size,
                    target_overlap=overlap,
                    retained_beam=args.retained_beam,
                    candidate_restarts=args.candidate_restarts,
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
        branch_payload[branch_name] = {
            "method": method,
            "target_overlap": overlap,
            "mean_patch_count": float(
                np.mean([example.patch_vectors.shape[0] for example in examples])
            ),
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
        "protocol": "marginal-beam-compression-audit-v0-20260802",
        "config": {
            "graph_bank_seed": args.graph_bank_seed,
            "cover_seed": args.cover_seed,
            "families": list(FAMILIES),
            "n_nodes": args.n_nodes,
            "target_degrees": args.degrees,
            "graphs_per_family_degree": args.graphs_per_cell,
            "patch_size": args.patch_size,
            "multiplier": args.multiplier,
            "retained_beam": args.retained_beam,
            "candidate_restarts": args.candidate_restarts,
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
