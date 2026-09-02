"""Compare target-edge and scalable marginal candidate covers."""
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
    "tracks/ksvd/results/from_scratch/marginal_candidate_beam_audit_20260802.json"
)
DEFAULT_REPORT = Path(
    "tracks/ksvd/results/from_scratch/MARGINAL_CANDIDATE_BEAM_AUDIT_20260802.md"
)
BRANCHES = {
    "target_s10_o5": ("target", 10, 5),
    "beam_s10_o5": ("beam", 10, 5),
    "target_s10_o3": ("target", 10, 3),
    "beam_s10_o3": ("beam", 10, 3),
}


def _summary(rows: Sequence[dict[str, Any]], branch: str) -> dict[str, float]:
    selected = [row for row in rows if row["branch"] == branch]
    metrics = (
        "true_edge_coverage",
        "node_pair_coverage",
        "raw_full_adjacency_rmse",
        "new_pairs_per_patch_mean",
        "bridge_length_mean",
    )
    return {
        metric: float(np.mean([row[metric] for row in selected]))
        for metric in metrics
    }


def _relative_reduction(left: float, right: float) -> float:
    return float((left - right) / max(left, 1e-12))


def classify(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summaries = {branch: _summary(rows, branch) for branch in BRANCHES}
    comparisons = {}
    invariant_gate = all(
        row["connected_rate"] == 1.0
        and row["continuous_fraction"] == 1.0
        and row["overlap_exact"]
        for row in rows
    )
    for overlap in (5, 3):
        target = summaries[f"target_s10_o{overlap}"]
        beam = summaries[f"beam_s10_o{overlap}"]
        edge_gain = beam["true_edge_coverage"] - target["true_edge_coverage"]
        pair_gain = beam["node_pair_coverage"] - target["node_pair_coverage"]
        rmse_gain = _relative_reduction(
            target["raw_full_adjacency_rmse"], beam["raw_full_adjacency_rmse"]
        )
        checks = {
            "edge_gain_at_least_003": edge_gain >= 0.03,
            "pair_not_worse_by_001": pair_gain >= -0.01,
            "full_rmse_reduction_at_least_002": rmse_gain >= 0.02,
        }
        comparisons[f"o{overlap}"] = {
            "edge_coverage_gain": float(edge_gain),
            "pair_coverage_gain": float(pair_gain),
            "raw_full_rmse_reduction": rmse_gain,
            "checks": checks,
            "passed": all(checks.values()),
        }
    if not invariant_gate:
        label = "FAIL_MARGINAL_BEAM_INVARIANTS"
    elif any(value["passed"] for value in comparisons.values()):
        label = "PASS_MARGINAL_CANDIDATE_BEAM"
    elif any(value["edge_coverage_gain"] > 0.0 for value in comparisons.values()):
        label = "BEAM_IMPROVES_COVERAGE_BELOW_GATE"
    else:
        label = "REJECT_SCALABLE_MARGINAL_BEAM"
    return {
        "classification": label,
        "invariant_gate": invariant_gate,
        "summaries": summaries,
        "comparisons": comparisons,
    }


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def render_report(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    lines = [
        "# Marginal candidate beam continuous-cover 审计",
        "",
        "> 日期：2026-08-02  ",
        "> 72 graphs；single chain；exact overlap；RAW cover comparison。",
        "",
        "## 1. 判定",
        "",
        f"**{decision['classification']}**",
        "",
        "## 2. Mean results",
        "",
        "| branch | edge cover | pair cover | RAW full RMSE | new pairs/patch | bridge length |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for branch in BRANCHES:
        summary = decision["summaries"][branch]
        lines.append(
            f"| {branch} | {_fmt(summary['true_edge_coverage'])} | "
            f"{_fmt(summary['node_pair_coverage'])} | "
            f"{_fmt(summary['raw_full_adjacency_rmse'])} | "
            f"{_fmt(summary['new_pairs_per_patch_mean'])} | "
            f"{_fmt(summary['bridge_length_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## 3. Registered comparisons",
            "",
            f"`{decision['comparisons']}`",
            "",
            "## 4. 边界",
            "",
            "- beam 近似每一步 marginal objective，不是全链 global optimum。",
            "- 第一 patch 与 target branch 共用同类 seed；差异来自后续 candidate selection。",
            "- beam 不显式追逐单一 target edge，因此 bridge length 字段为 0；连续性由 exact overlap 和 connected patch 保证。",
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
    parser.add_argument("--multiplier", type=float, default=1.5)
    parser.add_argument("--retained-beam", type=int, default=32)
    parser.add_argument("--candidate-restarts", type=int, default=2)
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
                for branch_index, (branch, (method, patch_size, overlap)) in enumerate(
                    BRANCHES.items()
                ):
                    budget = patch_budget(
                        adjacency,
                        patch_size=patch_size,
                        target_overlap=overlap,
                        edge_capacity_multiplier=args.multiplier,
                    )
                    seed = int(
                        np.random.SeedSequence(
                            [args.cover_seed, graph_index, branch_index]
                        ).generate_state(1, dtype=np.uint32)[0]
                    )
                    if method == "target":
                        cover = sample_edge_target_bridge_cover(
                            adjacency,
                            np.random.default_rng(seed),
                            n_patches=budget,
                            patch_size=patch_size,
                            target_overlap=overlap,
                        )
                    else:
                        cover = sample_marginal_candidate_cover(
                            adjacency,
                            np.random.default_rng(seed),
                            n_patches=budget,
                            patch_size=patch_size,
                            target_overlap=overlap,
                            retained_beam=args.retained_beam,
                            candidate_restarts=args.candidate_restarts,
                        )
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
                            "branch": branch,
                            "patch_count": budget,
                            "true_edge_coverage": audit["true_edge_coverage"],
                            "node_pair_coverage": audit["node_pair_coverage"],
                            "raw_full_adjacency_rmse": raw[
                                "full_adjacency_rmse"
                            ],
                            "new_pairs_per_patch_mean": audit[
                                "new_pairs_per_patch_mean"
                            ],
                            "bridge_length_mean": audit["bridge_length_mean"],
                            "connected_rate": audit["patch_connected_rate"],
                            "continuous_fraction": audit[
                                "continuous_transition_fraction"
                            ],
                            "overlap_exact": all(
                                value == overlap
                                for value in audit["transition_overlap_exact"]
                            ),
                        }
                    )
                print(f"graph={graph_index} family={family} degree={degree}", flush=True)
                graph_index += 1
    decision = classify(rows)
    payload = {
        "protocol": "marginal-candidate-beam-audit-v0-20260802",
        "config": {
            "graph_bank_seed": args.graph_bank_seed,
            "cover_seed": args.cover_seed,
            "families": list(FAMILIES),
            "n_nodes": args.n_nodes,
            "target_degrees": args.degrees,
            "graphs_per_family_degree": args.graphs_per_cell,
            "multiplier": args.multiplier,
            "retained_beam": args.retained_beam,
            "candidate_restarts": args.candidate_restarts,
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
