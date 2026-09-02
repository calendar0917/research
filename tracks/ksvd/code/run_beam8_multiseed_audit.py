"""Audit Beam8 continuous-cover gains across three cover seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .marginal_candidate_cover import sample_marginal_candidate_cover
from .overlap_cover import audit_cover, patch_budget, sample_edge_target_bridge_cover
from .overlap_stitching import make_cover_example, stitch_patch_predictions
from .run_overlap_cover_audit import FAMILIES, generate_graph


DEFAULT_JSON = Path(
    "tracks/ksvd/results/from_scratch/beam8_multiseed_audit_20260802.json"
)
DEFAULT_REPORT = Path(
    "tracks/ksvd/results/from_scratch/BEAM8_MULTISEED_AUDIT_20260802.md"
)


def _summary(rows: Sequence[dict[str, Any]], seed: int, branch: str) -> dict[str, float]:
    selected = [
        row for row in rows if row["cover_seed"] == seed and row["branch"] == branch
    ]
    return {
        metric: float(np.mean([row[metric] for row in selected]))
        for metric in (
            "true_edge_coverage",
            "node_pair_coverage",
            "raw_full_adjacency_rmse",
        )
    }


def classify(rows: Sequence[dict[str, Any]], seeds: Sequence[int]) -> dict[str, Any]:
    seed_rows = []
    for seed in seeds:
        target = _summary(rows, seed, "target")
        beam = _summary(rows, seed, "beam8")
        edge_gain = beam["true_edge_coverage"] - target["true_edge_coverage"]
        rmse_gain = (
            target["raw_full_adjacency_rmse"] - beam["raw_full_adjacency_rmse"]
        ) / max(target["raw_full_adjacency_rmse"], 1e-12)
        seed_rows.append(
            {
                "cover_seed": seed,
                "target": target,
                "beam8": beam,
                "edge_coverage_gain": float(edge_gain),
                "pair_coverage_gain": float(
                    beam["node_pair_coverage"] - target["node_pair_coverage"]
                ),
                "raw_full_rmse_reduction": float(rmse_gain),
                "passed": edge_gain >= 0.08 and rmse_gain >= 0.20,
            }
        )
    invariant_gate = all(
        row["connected_rate"] == 1.0
        and row["continuous_fraction"] == 1.0
        and row["overlap_exact"]
        for row in rows
    )
    pass_count = sum(row["passed"] for row in seed_rows)
    if not invariant_gate:
        label = "FAIL_BEAM8_MULTISEED_INVARIANTS"
    elif pass_count == len(seeds):
        label = "PASS_BEAM8_THREE_SEED_ROBUSTNESS"
    elif pass_count > 0:
        label = "BEAM8_SEED_SENSITIVE"
    else:
        label = "REJECT_BEAM8_ROBUSTNESS"
    return {
        "classification": label,
        "invariant_gate": invariant_gate,
        "pass_count": int(pass_count),
        "seed_rows": seed_rows,
    }


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def render_report(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    lines = [
        "# Beam8 continuous cover：3-seed robustness 审计",
        "",
        "> 日期：2026-08-02  ",
        "> s10/o3/m1.5；72 fixed graphs；RAW comparison。",
        "",
        "## 1. 判定",
        "",
        f"**{decision['classification']}**",
        "",
        "| seed | target edge | Beam8 edge | edge gain | target RMSE | Beam8 RMSE | RMSE reduction | pair gain | pass |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in decision["seed_rows"]:
        lines.append(
            f"| {row['cover_seed']} | "
            f"{_fmt(row['target']['true_edge_coverage'])} | "
            f"{_fmt(row['beam8']['true_edge_coverage'])} | "
            f"{_fmt(row['edge_coverage_gain'])} | "
            f"{_fmt(row['target']['raw_full_adjacency_rmse'])} | "
            f"{_fmt(row['beam8']['raw_full_adjacency_rmse'])} | "
            f"{_fmt(row['raw_full_rmse_reduction'])} | "
            f"{_fmt(row['pair_coverage_gain'])} | "
            f"{row['passed']} |"
        )
    lines.extend(
        [
            "",
            f"Passed seeds：`{decision['pass_count']}/3`；invariants：`{decision['invariant_gate']}`。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-bank-seed", type=int, default=810001)
    parser.add_argument(
        "--cover-seeds", type=int, nargs="+", default=[900101, 900102, 900103]
    )
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
    graphs = []
    graph_index = 0
    for family in FAMILIES:
        for degree in args.degrees:
            for _replicate in range(args.graphs_per_cell):
                graph_seed = int(
                    sequences[graph_index].generate_state(1, dtype=np.uint32)[0]
                )
                graphs.append(
                    (
                        graph_index,
                        family,
                        degree,
                        generate_graph(family, args.n_nodes, degree, graph_seed),
                    )
                )
                graph_index += 1

    rows = []
    for cover_seed in args.cover_seeds:
        for index, family, degree, adjacency in graphs:
            budget = patch_budget(
                adjacency,
                patch_size=args.patch_size,
                target_overlap=args.target_overlap,
                edge_capacity_multiplier=args.multiplier,
            )
            for branch_index, branch in enumerate(("target", "beam8")):
                seed = int(
                    np.random.SeedSequence(
                        [cover_seed, index, branch_index]
                    ).generate_state(1, dtype=np.uint32)[0]
                )
                if branch == "target":
                    cover = sample_edge_target_bridge_cover(
                        adjacency,
                        np.random.default_rng(seed),
                        n_patches=budget,
                        patch_size=args.patch_size,
                        target_overlap=args.target_overlap,
                    )
                else:
                    cover = sample_marginal_candidate_cover(
                        adjacency,
                        np.random.default_rng(seed),
                        n_patches=budget,
                        patch_size=args.patch_size,
                        target_overlap=args.target_overlap,
                        retained_beam=8,
                        candidate_restarts=1,
                    )
                audit = audit_cover(adjacency, cover)
                example = make_cover_example(index, family, degree, adjacency, cover)
                raw = stitch_patch_predictions(example, example.patch_vectors)
                rows.append(
                    {
                        "cover_seed": cover_seed,
                        "graph_index": index,
                        "family": family,
                        "target_degree": degree,
                        "branch": branch,
                        "true_edge_coverage": audit["true_edge_coverage"],
                        "node_pair_coverage": audit["node_pair_coverage"],
                        "raw_full_adjacency_rmse": raw["full_adjacency_rmse"],
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
        print(f"cover_seed={cover_seed} complete", flush=True)
    decision = classify(rows, args.cover_seeds)
    payload = {
        "protocol": "beam8-multiseed-audit-v0-20260802",
        "config": {
            "graph_bank_seed": args.graph_bank_seed,
            "cover_seeds": args.cover_seeds,
            "families": list(FAMILIES),
            "n_nodes": args.n_nodes,
            "target_degrees": args.degrees,
            "graphs_per_family_degree": args.graphs_per_cell,
            "patch_size": args.patch_size,
            "target_overlap": args.target_overlap,
            "multiplier": args.multiplier,
            "retained_beam": 8,
            "candidate_restarts": 1,
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
