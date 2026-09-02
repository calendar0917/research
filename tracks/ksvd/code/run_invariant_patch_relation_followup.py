"""ID-free invariant-token follow-up to the patch relation masked audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .marginal_candidate_cover import sample_marginal_candidate_cover
from .overlap_cover import patch_budget
from .overlap_stitching import CoverExample, make_cover_example
from .run_overlap_cover_audit import FAMILIES, generate_graph
from .run_patch_relation_representation_audit import (
    EPS,
    _cosine,
    _relative_l2,
    all_pairs_shortest_paths,
    bag_features,
    graph_balanced_metrics,
    patch_invariant_descriptor,
    pool_tokens,
    relation_features,
)
from .transition_decoder import fit_ridge_decoder, predict_ridge_decoder


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JSON = ROOT / "tracks/ksvd/results/from_scratch/invariant_patch_relation_followup_20260802.json"
DEFAULT_REPORT = ROOT / "tracks/ksvd/results/from_scratch/INVARIANT_PATCH_RELATION_FOLLOWUP_20260802.md"
PROTOCOL = "tracks/ksvd/docs/KSVD_INVARIANT_PATCH_RELATION_FOLLOWUP_PROTOCOL_20260802.md"
BRANCHES = (
    "INVARIANT_BAG",
    "INVARIANT_TRUE_RELATION",
    "INVARIANT_SHUFFLED_RELATION",
)


def invariant_tokens(example: CoverExample) -> np.ndarray:
    return np.stack(
        [patch_invariant_descriptor(patch.adjacency) for patch in example.cover.patches],
        axis=0,
    )


def build_masked_matrices(
    examples: Sequence[CoverExample],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    rows = {branch: [] for branch in BRANCHES}
    targets = []
    graph_indices = []
    for example in examples:
        tokens = invariant_tokens(example)
        distances = all_pairs_shortest_paths(example.adjacency)
        for target_index, target in enumerate(tokens):
            rows["INVARIANT_BAG"].append(bag_features(tokens, target_index))
            rows["INVARIANT_TRUE_RELATION"].append(
                relation_features(
                    tokens,
                    example.cover,
                    distances,
                    target_index,
                    shuffled=False,
                )
            )
            rows["INVARIANT_SHUFFLED_RELATION"].append(
                relation_features(
                    tokens,
                    example.cover,
                    distances,
                    target_index,
                    shuffled=True,
                )
            )
            targets.append(target)
            graph_indices.append(example.graph_index)
    return (
        {branch: np.stack(values, axis=0) for branch, values in rows.items()},
        np.stack(targets, axis=0),
        np.asarray(graph_indices, dtype=np.int64),
    )


def relation_graph_embedding(
    example: CoverExample,
    *,
    shuffled: bool,
) -> np.ndarray:
    tokens = invariant_tokens(example)
    distances = all_pairs_shortest_paths(example.adjacency)
    rows = []
    for target_index, token in enumerate(tokens):
        context = relation_features(
            tokens,
            example.cover,
            distances,
            target_index,
            shuffled=shuffled,
        )
        rows.append(np.concatenate([token, context]))
    return pool_tokens(np.stack(rows, axis=0))


def graph_embedding(branch: str, example: CoverExample) -> np.ndarray:
    if branch == "INVARIANT_BAG":
        return pool_tokens(invariant_tokens(example))
    if branch == "INVARIANT_TRUE_RELATION":
        return relation_graph_embedding(example, shuffled=False)
    if branch == "INVARIANT_SHUFFLED_RELATION":
        return relation_graph_embedding(example, shuffled=True)
    raise ValueError(f"unknown branch: {branch}")


def classify(folds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    folds = tuple(folds)
    if len(folds) != 3:
        raise ValueError("expected three folds")
    means = {}
    for branch in BRANCHES:
        keys = folds[0]["branches"][branch]["summary"].keys()
        means[branch] = {
            key: float(
                np.mean([fold["branches"][branch]["summary"][key] for fold in folds])
            )
            for key in keys
        }
    bag = means["INVARIANT_BAG"]["overall_rmse"]
    true = means["INVARIANT_TRUE_RELATION"]["overall_rmse"]
    shuffled = means["INVARIANT_SHUFFLED_RELATION"]["overall_rmse"]
    bag_gain = float((bag - true) / max(bag, EPS))
    shuffled_gain = float((shuffled - true) / max(shuffled, EPS))
    better_folds = [
        fold["branches"]["INVARIANT_TRUE_RELATION"]["summary"]["overall_rmse"]
        < fold["branches"]["INVARIANT_BAG"]["summary"]["overall_rmse"]
        and fold["branches"]["INVARIANT_TRUE_RELATION"]["summary"]["overall_rmse"]
        < fold["branches"]["INVARIANT_SHUFFLED_RELATION"]["summary"]["overall_rmse"]
        for fold in folds
    ]
    stability = {}
    for branch in BRANCHES:
        stability[branch] = {
            "cosine_similarity": float(
                np.mean([fold["stability"][branch]["cosine_similarity"] for fold in folds])
            ),
            "relative_l2_difference": float(
                np.mean(
                    [fold["stability"][branch]["relative_l2_difference"] for fold in folds]
                )
            ),
        }
    invariant_gate = all(bool(fold["invariants"]["passed"]) for fold in folds)
    checks = {
        "true_vs_bag_gain_at_least_002": bag_gain >= 0.02,
        "true_vs_shuffled_gain_at_least_002": shuffled_gain >= 0.02,
        "true_better_both_all_folds": all(better_folds),
        "true_embedding_cosine_at_least_090": stability[
            "INVARIANT_TRUE_RELATION"
        ]["cosine_similarity"]
        >= 0.90,
        "invariants": invariant_gate,
    }
    relation_checks = all(
        checks[key]
        for key in (
            "true_vs_bag_gain_at_least_002",
            "true_vs_shuffled_gain_at_least_002",
            "true_better_both_all_folds",
        )
    )
    if all(checks.values()):
        label = "ID_FREE_PATCH_RELATION_SUBSTRATE_SUPPORTED"
    elif relation_checks and invariant_gate:
        label = "INVARIANT_TOKEN_RELATION_USEFUL_BUT_SAMPLER_UNSTABLE"
    else:
        label = "REJECT_INVARIANT_PATCH_RELATION_SUBSTRATE"
    return {
        "classification": label,
        "checks": checks,
        "true_vs_bag_rmse_reduction": bag_gain,
        "true_vs_shuffled_rmse_reduction": shuffled_gain,
        "true_better_both_by_fold": better_folds,
        "mean_branches": means,
        "stability": stability,
    }


def render_report(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    lines = [
        "# ID-free invariant patch token + relation：跟进结果",
        "",
        "> 日期：2026-08-02  ",
        f"> 协议：`{payload['protocol']}`  ",
        f"> 判定：`{decision['classification']}`",
        "",
        "## 1. Held-out masked prediction",
        "",
        "| branch | overall RMSE | degree | spectrum | density/triangle |",
        "|---|---:|---:|---:|---:|",
    ]
    for branch in BRANCHES:
        values = decision["mean_branches"][branch]
        lines.append(
            f"| {branch} | {values['overall_rmse']:.5f} | "
            f"{values['degree_rmse']:.5f} | {values['spectrum_rmse']:.5f} | "
            f"{values['density_triangle_rmse']:.5f} |"
        )
    lines.extend(
        [
            "",
            f"- TRUE vs BAG reduction：`{decision['true_vs_bag_rmse_reduction']:.4f}`；",
            f"- TRUE vs SHUFFLED reduction：`{decision['true_vs_shuffled_rmse_reduction']:.4f}`；",
            f"- 3-fold simultaneous wins：`{decision['true_better_both_by_fold']}`。",
            "",
            "## 2. Relabel + resampling stability",
            "",
            "| branch | cosine | relative L2 |",
            "|---|---:|---:|",
        ]
    )
    for branch in BRANCHES:
        values = decision["stability"][branch]
        lines.append(
            f"| {branch} | {values['cosine_similarity']:.4f} | "
            f"{values['relative_l2_difference']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Registered checks：`{decision['checks']}`。",
            "",
            "## 3. 解释边界",
            "",
            "该分支证明的是 ID-free local structural tokens 能否保留 relation signal。descriptor 是手工 control；后续 learned encoder 必须在相同 relabel 和 shuffled gates 下比较。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-bank-seed", type=int, default=810001)
    parser.add_argument("--cover-seed", type=int, default=930101)
    parser.add_argument("--n-nodes", type=int, default=50)
    parser.add_argument("--degrees", type=int, nargs="+", default=[15, 20, 25])
    parser.add_argument("--graphs-per-cell", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=10)
    parser.add_argument("--target-overlap", type=int, default=3)
    parser.add_argument("--multiplier", type=float, default=1.5)
    parser.add_argument("--retained-beam", type=int, default=8)
    parser.add_argument("--candidate-restarts", type=int, default=1)
    parser.add_argument("--ridge-alpha", type=float, default=0.01)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    graph_count = len(FAMILIES) * len(args.degrees) * args.graphs_per_cell
    sequences = np.random.SeedSequence(args.graph_bank_seed).spawn(graph_count)
    examples = []
    sampler_seeds = {}
    graph_index = 0
    for family in FAMILIES:
        for degree in args.degrees:
            for _replicate in range(args.graphs_per_cell):
                graph_seed = int(sequences[graph_index].generate_state(1, dtype=np.uint32)[0])
                adjacency = generate_graph(family, args.n_nodes, degree, graph_seed)
                budget = patch_budget(
                    adjacency,
                    patch_size=args.patch_size,
                    target_overlap=args.target_overlap,
                    edge_capacity_multiplier=args.multiplier,
                )
                sampler_seed = int(
                    np.random.SeedSequence([args.cover_seed, graph_index]).generate_state(
                        1, dtype=np.uint32
                    )[0]
                )
                sampler_seeds[graph_index] = sampler_seed
                cover = sample_marginal_candidate_cover(
                    adjacency,
                    np.random.default_rng(sampler_seed),
                    n_patches=budget,
                    patch_size=args.patch_size,
                    target_overlap=args.target_overlap,
                    retained_beam=args.retained_beam,
                    candidate_restarts=args.candidate_restarts,
                )
                examples.append(
                    make_cover_example(graph_index, family, degree, adjacency, cover)
                )
                graph_index += 1

    folds = []
    for fold_index in range(3):
        test_examples = tuple(
            example for example in examples if example.graph_index % 8 % 3 == fold_index
        )
        test_ids = {example.graph_index for example in test_examples}
        train_examples = tuple(
            example for example in examples if example.graph_index not in test_ids
        )
        train_features, train_targets, _train_graph_rows = build_masked_matrices(
            train_examples
        )
        test_features, test_targets, test_graph_rows = build_masked_matrices(
            test_examples
        )
        branch_payload = {}
        for branch in BRANCHES:
            model = fit_ridge_decoder(
                train_features[branch], train_targets, alpha=args.ridge_alpha
            )
            predictions = predict_ridge_decoder(model, test_features[branch])
            graph_rows, summary = graph_balanced_metrics(
                test_targets, predictions, test_graph_rows
            )
            branch_payload[branch] = {"summary": summary, "graphs": graph_rows}

        stability_rows = []
        for example in test_examples:
            permutation = np.random.default_rng(
                np.random.SeedSequence(
                    [args.graph_bank_seed, example.graph_index, 8801]
                ).generate_state(1, dtype=np.uint32)[0]
            ).permutation(args.n_nodes)
            relabeled_adjacency = example.adjacency[np.ix_(permutation, permutation)]
            relabeled_cover = sample_marginal_candidate_cover(
                relabeled_adjacency,
                np.random.default_rng(sampler_seeds[example.graph_index]),
                n_patches=len(example.cover.patches),
                patch_size=args.patch_size,
                target_overlap=args.target_overlap,
                retained_beam=args.retained_beam,
                candidate_restarts=args.candidate_restarts,
            )
            relabeled_example = make_cover_example(
                example.graph_index,
                example.family,
                example.target_degree,
                relabeled_adjacency,
                relabeled_cover,
            )
            row = {"graph_index": example.graph_index}
            for branch in BRANCHES:
                original = graph_embedding(branch, example)
                relabeled = graph_embedding(branch, relabeled_example)
                row[branch] = {
                    "cosine_similarity": _cosine(original, relabeled),
                    "relative_l2_difference": _relative_l2(original, relabeled),
                }
            stability_rows.append(row)
        stability = {
            branch: {
                "cosine_similarity": float(
                    np.mean([row[branch]["cosine_similarity"] for row in stability_rows])
                ),
                "relative_l2_difference": float(
                    np.mean(
                        [row[branch]["relative_l2_difference"] for row in stability_rows]
                    )
                ),
            }
            for branch in BRANCHES
        }
        expected_train_rows = sum(len(example.cover.patches) for example in train_examples)
        expected_test_rows = sum(len(example.cover.patches) for example in test_examples)
        invariants = {
            "train_test_graph_isolation": not (
                {example.graph_index for example in train_examples} & test_ids
            ),
            "train_row_count": train_targets.shape[0] == expected_train_rows,
            "test_row_count": test_targets.shape[0] == expected_test_rows,
            "finite": all(
                np.all(np.isfinite(matrix)) for matrix in train_features.values()
            )
            and all(np.all(np.isfinite(matrix)) for matrix in test_features.values()),
            "target_dimension_22": train_targets.shape[1] == 22
            and test_targets.shape[1] == 22,
        }
        invariants["passed"] = all(invariants.values())
        folds.append(
            {
                "fold_index": fold_index,
                "train_graph_count": len(train_examples),
                "test_graph_count": len(test_examples),
                "branches": branch_payload,
                "stability": stability,
                "stability_graphs": stability_rows,
                "invariants": invariants,
            }
        )
        print(
            f"fold={fold_index} bag/true/shuffled="
            f"{branch_payload['INVARIANT_BAG']['summary']['overall_rmse']:.5f}/"
            f"{branch_payload['INVARIANT_TRUE_RELATION']['summary']['overall_rmse']:.5f}/"
            f"{branch_payload['INVARIANT_SHUFFLED_RELATION']['summary']['overall_rmse']:.5f}",
            flush=True,
        )

    decision = classify(folds)
    payload = {
        "protocol": PROTOCOL,
        "config": {
            "graph_bank_seed": args.graph_bank_seed,
            "cover_seed": args.cover_seed,
            "families": list(FAMILIES),
            "degrees": args.degrees,
            "graphs_per_cell": args.graphs_per_cell,
            "graph_count": graph_count,
            "n_nodes": args.n_nodes,
            "patch_size": args.patch_size,
            "target_overlap": args.target_overlap,
            "multiplier": args.multiplier,
            "retained_beam": args.retained_beam,
            "candidate_restarts": args.candidate_restarts,
            "ridge_alpha": args.ridge_alpha,
            "labels_used": False,
        },
        "folds": folds,
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
