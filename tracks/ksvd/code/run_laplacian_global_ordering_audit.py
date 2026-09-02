"""Compare construction and graph-global Fiedler patch orderings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .laplacian_ordering import (
    coordinate_tie_diagnostics,
    mapped_relabel_fiedler_invariance,
    one_swap_fiedler_stability,
    reorder_cover_by_fiedler,
)
from .overlap_cover import audit_cover, patch_budget, sample_edge_target_bridge_cover
from .overlap_stitching import CoverExample, make_cover_example
from .run_ksvd_stitched_reconstruction_audit import _run_fold, classify
from .run_overlap_cover_audit import FAMILIES, generate_graph


DEFAULT_JSON = Path(
    "tracks/ksvd/results/from_scratch/laplacian_global_ordering_audit_20260801.json"
)
DEFAULT_REPORT = Path(
    "tracks/ksvd/results/from_scratch/LAPLACIAN_GLOBAL_ORDERING_AUDIT_20260801.md"
)


def _relative_reduction(left: float, right: float) -> float:
    return float((left - right) / max(left, 1e-12))


def _coordinate_group_summary(
    rows: Sequence[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, Any]]] = {"all": list(rows)}
    for family in FAMILIES:
        groups[family] = [row for row in rows if row["family"] == family]
    summaries = {}
    for name, group in groups.items():
        summaries[name] = {
            "graph_count": int(len(group)),
            "canonical_pair_order_agreement_mean": float(
                np.mean([row["canonical_pair_order_agreement"] for row in group])
            ),
            "sign_invariant_pair_order_agreement_mean": float(
                np.mean(
                    [row["sign_invariant_pair_order_agreement"] for row in group]
                )
            ),
            "canonical_stable_fraction": float(
                np.mean(
                    [row["canonical_pair_order_agreement"] >= 0.80 for row in group]
                )
            ),
            "relative_eigengap_mean": float(
                np.mean([row["original_relative_eigengap"] for row in group])
            ),
            "relative_eigengap_median": float(
                np.median([row["original_relative_eigengap"] for row in group])
            ),
            "tied_node_fraction_mean": float(
                np.mean([row["tied_node_fraction"] for row in group])
            ),
            "graphs_with_ties_fraction": float(
                np.mean([row["tied_node_fraction"] > 0.0 for row in group])
            ),
        }
    return summaries


def classify_comparison(
    construction_folds: Sequence[dict[str, Any]],
    fiedler_folds: Sequence[dict[str, Any]],
    invariant_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    construction_decision = classify(construction_folds)
    fiedler_decision = classify(fiedler_folds)
    construction = construction_decision["mean_stages"]["final"]
    fiedler = fiedler_decision["mean_stages"]["final"]
    fiedler_pca = fiedler_decision["mean_stages"]["pca3"]
    rmse_reduction = _relative_reduction(
        construction["observed_pair_rmse"], fiedler["observed_pair_rmse"]
    )
    comparisons = {
        "rmse_relative_reduction_at_least_002": rmse_reduction >= 0.02,
        "disagreement_not_worse": fiedler[
            "repeated_pair_disagreement_std_mean"
        ]
        <= construction["repeated_pair_disagreement_std_mean"] + 1e-12,
        "observed_f1_not_worse": fiedler["observed_edge_f1"]
        >= construction["observed_edge_f1"] - 1e-12,
        "full_edge_recall_preserved": fiedler["full_edge_recall"]
        >= construction["full_edge_recall"] - 0.01,
        "beats_fiedler_pca3": fiedler["observed_pair_rmse"]
        < fiedler_pca["observed_pair_rmse"],
        "final_patch_better_all_folds": all(
            check["patch_final_better"]
            for check in fiedler_decision["optimization_checks_by_fold"]
        ),
    }
    comparative_gate = all(comparisons.values())
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
        "mapped_rank_match_one": all(
            row["mapped_rank_match_rate"] == 1.0 for row in invariant_rows
        ),
        "mapped_patch_rate_one": all(
            row["mapped_patch_rate"] == 1.0 for row in invariant_rows
        ),
        "mapped_transition_rate_one": all(
            row["mapped_transition_rate"] == 1.0 for row in invariant_rows
        ),
        "construction_raw_gate": construction_decision["raw_gate"],
        "fiedler_raw_gate": fiedler_decision["raw_gate"],
    }
    coordinate_summary = _coordinate_group_summary(invariant_rows)
    overall = coordinate_summary["all"]
    coordinate_stability_checks = {
        "mapped_relabel_rank_exact": invariants["mapped_rank_match_one"],
        "mean_canonical_pair_order_agreement_at_least_080": overall[
            "canonical_pair_order_agreement_mean"
        ]
        >= 0.80,
        "stable_graph_fraction_at_least_two_thirds": overall[
            "canonical_stable_fraction"
        ]
        >= 2.0 / 3.0,
    }
    coordinate_stability_gate = all(coordinate_stability_checks.values())
    invariant_gate = all(invariants.values())
    if not invariant_gate:
        label = "FAIL_LAPLACIAN_ORDERING_INVARIANTS"
    elif coordinate_stability_gate and comparative_gate:
        if fiedler_decision["optimization_gate"]:
            label = "PASS_LAPLACIAN_GLOBAL_ORDERING"
        else:
            label = "LAPLACIAN_ORDERING_HELPS_BELOW_KSVD_GATE"
    elif comparative_gate:
        label = "LAPLACIAN_ORDERING_HELP_BUT_UNSTABLE"
    elif coordinate_stability_gate:
        label = "STABLE_LAPLACIAN_COORDINATE_NO_RECON_GAIN"
    else:
        label = "REJECT_UNSTABLE_LAPLACIAN_ORDERING"
    return {
        "classification": label,
        "invariants": invariants,
        "invariant_gate": invariant_gate,
        "coordinate_summary": coordinate_summary,
        "coordinate_stability_checks": coordinate_stability_checks,
        "coordinate_stability_gate": coordinate_stability_gate,
        "comparisons": comparisons,
        "comparative_gate": comparative_gate,
        "rmse_relative_reduction": rmse_reduction,
        "construction_decision": construction_decision,
        "fiedler_decision": fiedler_decision,
    }


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def render_report(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    lines = [
        "# Laplacian graph-global ordering：matched KSVD 审计",
        "",
        "> 日期：2026-08-01  ",
        "> normalized-Laplacian Fiedler rank；node sets、coverage、folds 和容量完全匹配。",
        "",
        "## 1. 判定",
        "",
        f"**{decision['classification']}**",
        "",
        "## 2. Spectral coordinate audit",
        "",
        "| graph group | n | canonical pair-order agreement | sign-invariant agreement | stable fraction | relative eigengap mean/median |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group in ("all", *FAMILIES):
        summary = decision["coordinate_summary"][group]
        lines.append(
            f"| {group} | {summary['graph_count']} | "
            f"{_fmt(summary['canonical_pair_order_agreement_mean'])} | "
            f"{_fmt(summary['sign_invariant_pair_order_agreement_mean'])} | "
            f"{_fmt(summary['canonical_stable_fraction'])} | "
            f"{_fmt(summary['relative_eigengap_mean'])}/"
            f"{_fmt(summary['relative_eigengap_median'])} |"
        )

    lines.extend(
        [
            "",
            f"Coordinate stability checks：`{decision['coordinate_stability_checks']}`。",
            "",
            "### Post-result tie diagnostic（不参与 gate）",
            "",
            "| graph group | mean tied-node fraction | graphs with ties |",
            "|---|---:|---:|",
        ]
    )
    for group in ("all", *FAMILIES):
        summary = decision["coordinate_summary"][group]
        lines.append(
            f"| {group} | {_fmt(summary['tied_node_fraction_mean'])} | "
            f"{_fmt(summary['graphs_with_ties_fraction'])} |"
        )

    lines.extend(
        [
            "",
            "该 diagnostic 是在正式 invariant failure 后用于定位原因，不改变预注册判定。tied nodes 的 Fiedler 值在 `1e-10` 内不可区分，其内部 rank 会依赖 tie-breaking。",
            "",
            "## 3. Fold-balanced reconstruction means",
            "",
            "| branch | stage | patch rel err | observed RMSE | observed F1 | full edge recall | overlap disagreement |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for branch in ("construction", "fiedler"):
        branch_decision = decision[f"{branch}_decision"]
        for stage in ("raw", "init", "final", "pca3"):
            mean = branch_decision["mean_stages"][stage]
            lines.append(
                f"| {branch} | {stage} | "
                f"{_fmt(mean['patch_relative_error'])} | "
                f"{_fmt(mean['observed_pair_rmse'])} | "
                f"{_fmt(mean['observed_edge_f1'])} | "
                f"{_fmt(mean['full_edge_recall'])} | "
                f"{_fmt(mean['repeated_pair_disagreement_std_mean'])} |"
            )

    lines.extend(
        [
            "",
            "## 4. Registered gates",
            "",
            f"- representation invariants：`{decision['invariants']}`。",
            f"- coordinate stability gate：`{decision['coordinate_stability_gate']}`。",
            f"- Fiedler FINAL vs construction FINAL RMSE relative reduction：`{_fmt(decision['rmse_relative_reduction'])}`。",
            f"- comparative checks：`{decision['comparisons']}`。",
            f"- comparative gate：`{decision['comparative_gate']}`。",
            f"- Fiedler optimization gate：`{decision['fiedler_decision']['optimization_gate']}`；mean patch INIT→FINAL reduction：`{_fmt(decision['fiedler_decision']['mean_patch_relative_reduction'])}`。",
            "",
            "## 5. 如何理解“位置”",
            "",
            "- Fiedler rank 是每张完整图内部的 global spectral coordinate，不是跨图共享的物理坐标。",
            "- node relabeling 不应改变它；一次边交换稳定性则检查它是否对轻微结构变化过度敏感。",
            "- 即使 rank 稳定，也只有 matched reconstruction 改善后，才能说它对当前 patch/KSVD 表示有实际价值。",
            "- 本轮不使用 labels，也不回答 LapPE 输入 GNN/Transformer 后的分类效果。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-bank-seed", type=int, default=810001)
    parser.add_argument("--cover-seed", type=int, default=830101)
    parser.add_argument("--relabel-seed", type=int, default=850101)
    parser.add_argument("--perturb-seed", type=int, default=860101)
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
    fiedler_examples: list[CoverExample] = []
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
                fiedler, coordinate = reorder_cover_by_fiedler(
                    adjacency, construction
                )
                construction_audit = audit_cover(adjacency, construction)
                fiedler_audit = audit_cover(adjacency, fiedler)
                permutation = np.random.default_rng(
                    np.random.SeedSequence([args.relabel_seed, graph_index])
                ).permutation(args.n_nodes)
                mapped = mapped_relabel_fiedler_invariance(
                    adjacency, construction, permutation
                )
                stability = one_swap_fiedler_stability(
                    adjacency,
                    np.random.default_rng(
                        np.random.SeedSequence([args.perturb_seed, graph_index])
                    ),
                )
                ties = coordinate_tie_diagnostics(coordinate.values)
                invariant_rows.append(
                    {
                        "graph_index": graph_index,
                        "family": family,
                        "target_degree": degree,
                        "node_sets_match": all(
                            set(left.node_ids) == set(right.node_ids)
                            for left, right in zip(
                                construction.patches, fiedler.patches
                            )
                        ),
                        "node_coverage_match": construction_audit["node_coverage"]
                        == fiedler_audit["node_coverage"],
                        "edge_coverage_match": construction_audit[
                            "true_edge_coverage"
                        ]
                        == fiedler_audit["true_edge_coverage"],
                        "pair_coverage_match": construction_audit[
                            "node_pair_coverage"
                        ]
                        == fiedler_audit["node_pair_coverage"],
                        "mapped_rank_match_rate": mapped["rank_match_rate"],
                        "mapped_patch_rate": mapped[
                            "patch_adjacency_match_rate"
                        ],
                        "mapped_transition_rate": mapped[
                            "transition_slot_map_match_rate"
                        ],
                        "fiedler_lambda_2": float(coordinate.eigenvalues[1]),
                        "fiedler_lambda_3": float(coordinate.eigenvalues[2]),
                        **ties,
                        **stability,
                    }
                )
                construction_examples.append(
                    make_cover_example(
                        graph_index, family, degree, adjacency, construction
                    )
                )
                fiedler_examples.append(
                    make_cover_example(graph_index, family, degree, adjacency, fiedler)
                )
                graph_index += 1

    branch_folds = {"construction": [], "fiedler": []}
    for branch, examples in (
        ("construction", construction_examples),
        ("fiedler", fiedler_examples),
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
                f"final_rmse={fold['stages']['final']['summary']['observed_pair_rmse']:.4f} "
                f"disagree={fold['stages']['final']['summary']['repeated_pair_disagreement_std_mean']:.4f}",
                flush=True,
            )

    decision = classify_comparison(
        branch_folds["construction"], branch_folds["fiedler"], invariant_rows
    )
    payload = {
        "protocol": "ksvd-laplacian-global-ordering-audit-v0-20260801",
        "config": {
            "graph_bank_seed": args.graph_bank_seed,
            "cover_seed": args.cover_seed,
            "relabel_seed": args.relabel_seed,
            "perturb_seed": args.perturb_seed,
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
            "laplacian": "symmetric_normalized",
            "spectral_coordinate": "sign-canonicalized Fiedler rank",
            "perturbation": "one connected degree-preserving double-edge swap",
            "labels_used": False,
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
