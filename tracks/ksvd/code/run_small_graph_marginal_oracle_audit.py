"""Compare target-edge covers with exhaustive one-step marginal coverage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .marginal_coverage_cover import sample_exhaustive_marginal_cover
from .overlap_cover import audit_cover, patch_budget, sample_edge_target_bridge_cover
from .run_overlap_cover_audit import FAMILIES, generate_graph


DEFAULT_JSON = Path(
    "tracks/ksvd/results/from_scratch/small_graph_marginal_oracle_audit_20260802.json"
)
DEFAULT_REPORT = Path(
    "tracks/ksvd/results/from_scratch/SMALL_GRAPH_MARGINAL_ORACLE_AUDIT_20260802.md"
)


def _summary(rows: Sequence[dict[str, Any]], branch: str) -> dict[str, float]:
    metrics = (
        "true_edge_coverage",
        "node_pair_coverage",
        "node_coverage",
        "new_pairs_per_patch_mean",
        "edge_observation_multiplicity_cv",
    )
    selected = [row for row in rows if row["branch"] == branch]
    return {
        metric: float(np.mean([row[metric] for row in selected]))
        for metric in metrics
    }


def classify(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    target = _summary(rows, "target_edge")
    oracle = _summary(rows, "exhaustive_marginal")
    edge_gap = oracle["true_edge_coverage"] - target["true_edge_coverage"]
    pair_gap = oracle["node_pair_coverage"] - target["node_pair_coverage"]
    per_graph_edge_gaps = []
    graph_ids = sorted({row["graph_index"] for row in rows})
    for graph_id in graph_ids:
        graph_rows = [row for row in rows if row["graph_index"] == graph_id]
        values = {row["branch"]: row for row in graph_rows}
        per_graph_edge_gaps.append(
            values["exhaustive_marginal"]["true_edge_coverage"]
            - values["target_edge"]["true_edge_coverage"]
        )
    return {
        "classification": "HEURISTIC_HAS_MATERIAL_ORACLE_GAP"
        if edge_gap >= 0.03
        else "TARGET_HEURISTIC_NEAR_ONE_STEP_ORACLE",
        "target_summary": target,
        "oracle_summary": oracle,
        "mean_edge_coverage_gap": float(edge_gap),
        "mean_pair_coverage_gap": float(pair_gap),
        "oracle_edge_better_graph_fraction": float(
            np.mean([gap > 0.0 for gap in per_graph_edge_gaps])
        ),
        "oracle_edge_not_worse_graph_fraction": float(
            np.mean([gap >= 0.0 for gap in per_graph_edge_gaps])
        ),
    }


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def render_report(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    lines = [
        "# Small-graph exhaustive marginal coverage 审计",
        "",
        "> 日期：2026-08-02  ",
        "> exact one-step candidate enumeration；不声称全链 global optimum。",
        "",
        "## 1. 判定",
        "",
        f"**{decision['classification']}**",
        "",
        "## 2. Mean results",
        "",
        "| branch | edge cover | pair cover | node cover | new pairs/patch | edge multiplicity CV |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for branch, key in (
        ("target_edge", "target_summary"),
        ("exhaustive_marginal", "oracle_summary"),
    ):
        summary = decision[key]
        lines.append(
            f"| {branch} | {_fmt(summary['true_edge_coverage'])} | "
            f"{_fmt(summary['node_pair_coverage'])} | "
            f"{_fmt(summary['node_coverage'])} | "
            f"{_fmt(summary['new_pairs_per_patch_mean'])} | "
            f"{_fmt(summary['edge_observation_multiplicity_cv'])} |"
        )
    lines.extend(
        [
            "",
            "## 3. Gaps",
            "",
            f"- edge coverage gap：`{_fmt(decision['mean_edge_coverage_gap'])}`；",
            f"- pair coverage gap：`{_fmt(decision['mean_pair_coverage_gap'])}`；",
            f"- oracle edge better graph fraction：`{_fmt(decision['oracle_edge_better_graph_fraction'])}`；",
            f"- oracle edge not-worse graph fraction：`{_fmt(decision['oracle_edge_not_worse_graph_fraction'])}`。",
            "",
            "## 4. 边界",
            "",
            "- exhaustive branch 只保证当前一步的 lexicographic marginal objective 最优，后续 chain 仍是 greedy。",
            "- 小图结果用于量化 heuristic gap；不能直接把枚举算法扩展到 50 nodes。",
            "- 若 gap 明显，下一步应做 candidate beam/search，而不是继续修改 target deficit 权重。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-bank-seed", type=int, default=880101)
    parser.add_argument("--cover-seed", type=int, default=880201)
    parser.add_argument("--n-nodes", type=int, default=18)
    parser.add_argument("--degrees", type=int, nargs="+", default=[4, 6])
    parser.add_argument("--graphs-per-cell", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=6)
    parser.add_argument("--target-overlap", type=int, default=3)
    parser.add_argument("--edge-capacity-multiplier", type=float, default=1.5)
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
                    edge_capacity_multiplier=args.edge_capacity_multiplier,
                )
                for offset, (branch, sampler) in enumerate(
                    (
                        ("target_edge", sample_edge_target_bridge_cover),
                        ("exhaustive_marginal", sample_exhaustive_marginal_cover),
                    )
                ):
                    seed = int(
                        np.random.SeedSequence(
                            [args.cover_seed, graph_index, offset]
                        ).generate_state(1, dtype=np.uint32)[0]
                    )
                    cover = sampler(
                        adjacency,
                        np.random.default_rng(seed),
                        n_patches=budget,
                        patch_size=args.patch_size,
                        target_overlap=args.target_overlap,
                    )
                    audit = audit_cover(adjacency, cover)
                    rows.append(
                        {
                            "graph_index": graph_index,
                            "family": family,
                            "target_degree": degree,
                            "branch": branch,
                            "patch_count": budget,
                            "true_edge_coverage": audit["true_edge_coverage"],
                            "node_pair_coverage": audit["node_pair_coverage"],
                            "node_coverage": audit["node_coverage"],
                            "new_pairs_per_patch_mean": audit[
                                "new_pairs_per_patch_mean"
                            ],
                            "edge_observation_multiplicity_cv": audit[
                                "edge_observation_multiplicity_cv"
                            ],
                            "patch_connected_rate": audit["patch_connected_rate"],
                            "continuous_fraction": audit[
                                "continuous_transition_fraction"
                            ],
                        }
                    )
                print(f"graph={graph_index} family={family} degree={degree}", flush=True)
                graph_index += 1

    if not all(
        row["patch_connected_rate"] == 1.0
        and row["continuous_fraction"] == 1.0
        for row in rows
    ):
        raise RuntimeError("small-graph cover invariants failed")
    decision = classify(rows)
    payload = {
        "protocol": "small-graph-marginal-oracle-audit-v0-20260802",
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
