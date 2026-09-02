"""Compare construction-order and slot-persistent target-bridge covers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .overlap_cover import (
    audit_cover,
    make_slot_persistent_cover,
    mapped_replay_relabel_invariance,
    patch_budget,
    sample_edge_target_bridge_cover,
)
from .overlap_stitching import CoverExample, make_cover_example
from .run_ksvd_stitched_reconstruction_audit import _run_fold, classify
from .run_overlap_cover_audit import FAMILIES, generate_graph


DEFAULT_JSON = Path(
    "tracks/ksvd/results/from_scratch/slot_persistent_ordering_audit_20260801.json"
)
DEFAULT_REPORT = Path(
    "tracks/ksvd/results/from_scratch/SLOT_PERSISTENT_ORDERING_AUDIT_20260801.md"
)


def classify_comparison(
    construction_folds: list[dict[str, Any]],
    persistent_folds: list[dict[str, Any]],
    invariants: dict[str, bool],
) -> dict[str, Any]:
    construction_decision = classify(construction_folds)
    persistent_decision = classify(persistent_folds)
    construction = construction_decision["mean_stages"]["final"]
    persistent = persistent_decision["mean_stages"]["final"]
    rmse_reduction = (
        construction["observed_pair_rmse"] - persistent["observed_pair_rmse"]
    ) / max(construction["observed_pair_rmse"], 1e-12)
    disagreement_reduction = (
        construction["repeated_pair_disagreement_std_mean"]
        - persistent["repeated_pair_disagreement_std_mean"]
    ) / max(construction["repeated_pair_disagreement_std_mean"], 1e-12)
    comparisons = {
        "rmse_relative_reduction_at_least_002": rmse_reduction >= 0.02,
        "disagreement_relative_reduction_at_least_002": disagreement_reduction
        >= 0.02,
        "observed_f1_not_worse": persistent["observed_edge_f1"]
        >= construction["observed_edge_f1"] - 1e-12,
        "full_edge_recall_not_worse": persistent["full_edge_recall"]
        >= construction["full_edge_recall"] - 1e-12,
        "beats_persistent_pca3": persistent["observed_pair_rmse"]
        < persistent_decision["mean_stages"]["pca3"]["observed_pair_rmse"],
    }
    comparative_gate = all(comparisons.values())
    if not all(invariants.values()):
        label = "FAIL_SLOT_PERSISTENCE_INVARIANTS"
    elif persistent_decision["optimization_gate"] and comparative_gate:
        label = "PASS_SLOT_PERSISTENT_KSVD"
    elif comparative_gate:
        label = "PERSISTENT_COORDINATES_HELP_BUT_BELOW_GATE"
    else:
        label = "REJECT_SLOT_PERSISTENT_ORDERING"
    return {
        "classification": label,
        "invariants": invariants,
        "construction_decision": construction_decision,
        "persistent_decision": persistent_decision,
        "rmse_relative_reduction": float(rmse_reduction),
        "disagreement_relative_reduction": float(disagreement_reduction),
        "comparisons": comparisons,
        "comparative_gate": comparative_gate,
    }


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def render_report(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    lines = [
        "# Slot-persistent continuous cover：KSVD matched audit",
        "",
        "> 日期：2026-08-01  ",
        "> node sets、cover budget 和 train/test folds 完全匹配；只改变 local ordering。",
        "",
        "## 1. 判定",
        "",
        f"**{decision['classification']}**",
        "",
        "## 2. Ordering invariants",
        "",
        f"`{decision['invariants']}`",
        "",
        "## 3. Fold-balanced stage means",
        "",
        "| branch | stage | patch rel err | observed RMSE | observed F1 | full edge recall | overlap disagreement |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for branch in ("construction", "persistent"):
        branch_decision = decision[f"{branch}_decision"]
        for stage in ("raw", "init", "final", "pca3"):
            mean = branch_decision["mean_stages"][stage]
            lines.append(
                f"| {branch} | {stage} | {_fmt(mean['patch_relative_error'])} | "
                f"{_fmt(mean['observed_pair_rmse'])} | "
                f"{_fmt(mean['observed_edge_f1'])} | "
                f"{_fmt(mean['full_edge_recall'])} | "
                f"{_fmt(mean['repeated_pair_disagreement_std_mean'])} |"
            )

    lines.extend(
        [
            "",
            "## 4. Registered comparison",
            "",
            f"- persistent FINAL vs construction FINAL observed RMSE relative reduction：`{_fmt(decision['rmse_relative_reduction'])}`；",
            f"- overlap disagreement relative reduction：`{_fmt(decision['disagreement_relative_reduction'])}`；",
            f"- comparisons：`{decision['comparisons']}`；",
            f"- persistent mean patch INIT→FINAL reduction：`{_fmt(decision['persistent_decision']['mean_patch_relative_reduction'])}`；",
            f"- persistent optimization gate：`{decision['persistent_decision']['optimization_gate']}`。",
            "",
            "## 5. 解释边界",
            "",
            "- persistent slots 只运输同一条 traversal 中的局部坐标，不产生跨图绝对位置。",
            "- node sets 和 edge coverage 完全不变，因此差异可以归因于 ordering/coordinate continuity，而不是 sampler。",
            "- 若 ordering 无法通过 comparison，下一步必须让 learner 显式使用 transition map，而不是继续手工排序。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-bank-seed", type=int, default=810001)
    parser.add_argument("--cover-seed", type=int, default=830101)
    parser.add_argument("--relabel-seed", type=int, default=840101)
    parser.add_argument("--n-nodes", type=int, default=50)
    parser.add_argument("--degrees", type=int, nargs="+", default=[15, 20, 25])
    parser.add_argument("--graphs-per-cell", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=10)
    parser.add_argument("--target-overlap", type=int, default=5)
    parser.add_argument("--edge-capacity-multiplier", type=float, default=1.5)
    parser.add_argument("--n-atoms", type=int, default=24)
    parser.add_argument("--sparsity", type=int, default=3)
    parser.add_argument("--minimum-sparsity", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--pca-rank", type=int, default=3)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    construction_examples: list[CoverExample] = []
    persistent_examples: list[CoverExample] = []
    invariant_rows = []
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
                construction = sample_edge_target_bridge_cover(
                    adjacency,
                    np.random.default_rng(sampler_seed),
                    n_patches=budget,
                    patch_size=args.patch_size,
                    target_overlap=args.target_overlap,
                )
                persistent = make_slot_persistent_cover(adjacency, construction)
                construction_audit = audit_cover(adjacency, construction)
                persistent_audit = audit_cover(adjacency, persistent)
                permutation = np.random.default_rng(
                    np.random.SeedSequence([args.relabel_seed, graph_index])
                ).permutation(args.n_nodes)
                mapped = mapped_replay_relabel_invariance(
                    adjacency, persistent, permutation
                )
                invariant_rows.append(
                    {
                        "graph_index": graph_index,
                        "node_sets_match": all(
                            set(left.node_ids) == set(right.node_ids)
                            for left, right in zip(
                                construction.patches, persistent.patches
                            )
                        ),
                        "node_coverage_match": construction_audit["node_coverage"]
                        == persistent_audit["node_coverage"],
                        "edge_coverage_match": construction_audit[
                            "true_edge_coverage"
                        ]
                        == persistent_audit["true_edge_coverage"],
                        "pair_coverage_match": construction_audit[
                            "node_pair_coverage"
                        ]
                        == persistent_audit["node_pair_coverage"],
                        "persistent_slot_rate": persistent_audit[
                            "continuous_shared_slot_persistence_rate"
                        ],
                        "mapped_patch_rate": mapped["patch_adjacency_match_rate"],
                        "mapped_transition_rate": mapped[
                            "transition_slot_map_match_rate"
                        ],
                    }
                )
                construction_examples.append(
                    make_cover_example(
                        graph_index, family, degree, adjacency, construction
                    )
                )
                persistent_examples.append(
                    make_cover_example(
                        graph_index, family, degree, adjacency, persistent
                    )
                )
                graph_index += 1

    branch_folds = {"construction": [], "persistent": []}
    for branch, examples in (
        ("construction", construction_examples),
        ("persistent", persistent_examples),
    ):
        for fold_index in range(3):
            fold = _run_fold(
                examples,
                fold_index,
                n_atoms=args.n_atoms,
                sparsity=args.sparsity,
                minimum_sparsity=args.minimum_sparsity,
                n_iterations=args.iterations,
                pca_rank=args.pca_rank,
            )
            branch_folds[branch].append(fold)
            print(
                f"branch={branch} fold={fold_index} "
                f"final_patch={fold['stages']['final']['summary']['patch_relative_error']:.4f} "
                f"final_stitch={fold['stages']['final']['summary']['observed_pair_rmse']:.4f} "
                f"disagree={fold['stages']['final']['summary']['repeated_pair_disagreement_std_mean']:.4f}",
                flush=True,
            )

    invariants = {
        "node_sets_match": all(row["node_sets_match"] for row in invariant_rows),
        "node_coverage_match": all(
            row["node_coverage_match"] for row in invariant_rows
        ),
        "edge_coverage_match": all(
            row["edge_coverage_match"] for row in invariant_rows
        ),
        "pair_coverage_match": all(
            row["pair_coverage_match"] for row in invariant_rows
        ),
        "persistent_slot_rate_one": all(
            row["persistent_slot_rate"] == 1.0 for row in invariant_rows
        ),
        "mapped_patch_rate_one": all(
            row["mapped_patch_rate"] == 1.0 for row in invariant_rows
        ),
        "mapped_transition_rate_one": all(
            row["mapped_transition_rate"] == 1.0 for row in invariant_rows
        ),
    }
    decision = classify_comparison(
        branch_folds["construction"], branch_folds["persistent"], invariants
    )
    payload = {
        "protocol": "ksvd-slot-persistent-ordering-audit-v0-20260801",
        "config": {
            "graph_bank_seed": args.graph_bank_seed,
            "cover_seed": args.cover_seed,
            "relabel_seed": args.relabel_seed,
            "families": list(FAMILIES),
            "n_nodes": args.n_nodes,
            "target_degrees": args.degrees,
            "graphs_per_family_degree": args.graphs_per_cell,
            "patch_size": args.patch_size,
            "target_overlap": args.target_overlap,
            "n_atoms": args.n_atoms,
            "sparsity": args.sparsity,
            "minimum_sparsity": args.minimum_sparsity,
            "iterations": args.iterations,
            "pca_rank": args.pca_rank,
        },
        "invariant_rows": invariant_rows,
        "branches": branch_folds,
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
