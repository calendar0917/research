"""Audit an explicit transition-aware decoder on continuous patch chains."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .from_scratch_unplanted_dictionary import deterministic_maximin_initialization
from .imdb_walk_dictionary import encode_with_minimum_sparsity
from .ksvd import ksvd
from .overlap_cover import patch_budget, sample_edge_target_bridge_cover
from .overlap_stitching import (
    CoverExample,
    graph_balanced_stitch_summary,
    make_cover_example,
    stack_cover_examples,
    stitch_patch_predictions,
)
from .run_overlap_cover_audit import FAMILIES, generate_graph
from .transition_decoder import (
    fit_ridge_decoder,
    make_transition_features,
    predict_ridge_decoder,
    shuffle_transition_context,
)


DEFAULT_JSON = Path(
    "tracks/ksvd/results/from_scratch/transition_decoder_audit_20260801.json"
)
DEFAULT_REPORT = Path(
    "tracks/ksvd/results/from_scratch/TRANSITION_DECODER_AUDIT_20260801.md"
)


def _split_examples(
    examples: Sequence[CoverExample], fold_index: int
) -> tuple[tuple[CoverExample, ...], tuple[CoverExample, ...]]:
    test = tuple(
        example for example in examples if example.graph_index % 8 % 3 == fold_index
    )
    test_ids = {example.graph_index for example in test}
    train = tuple(example for example in examples if example.graph_index not in test_ids)
    if not train or not test:
        raise RuntimeError("fold split produced an empty train or test set")
    return train, test


def _encode_examples(
    examples: Sequence[CoverExample],
    dictionary: np.ndarray,
    train_mean: np.ndarray,
    *,
    sparsity: int,
    minimum_sparsity: int,
) -> list[dict[str, Any]]:
    examples = tuple(examples)
    if not examples:
        raise ValueError("examples cannot be empty")
    stacked = stack_cover_examples(examples)
    centered_stacked = stacked - train_mean
    stacked_codes = encode_with_minimum_sparsity(
        centered_stacked,
        dictionary,
        sparsity=sparsity,
        minimum_sparsity=minimum_sparsity,
    )
    encoded = []
    cursor = 0
    for example in examples:
        patch_count = example.patch_vectors.shape[0]
        stop = cursor + patch_count
        centered = centered_stacked[:, cursor:stop]
        codes = stacked_codes[:, cursor:stop]
        base = (dictionary @ codes + train_mean).T
        current, transition = make_transition_features(codes, base, example.cover)
        encoded.append(
            {
                "example": example,
                "codes": codes,
                "base": base,
                "current_features": current,
                "transition_features": transition,
                "centered_targets": centered.T,
            }
        )
        cursor = stop
    if cursor != stacked.shape[1]:
        raise RuntimeError("encoded graph slices do not consume the stacked matrix")
    return encoded


def _stage_summary(
    encoded: Sequence[dict[str, Any]],
    predictions: Sequence[np.ndarray] | None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    if predictions is not None and len(predictions) != len(encoded):
        raise ValueError("predictions must align with encoded examples")
    rows = []
    for index, item in enumerate(encoded):
        example = item["example"]
        predicted = example.patch_vectors if predictions is None else predictions[index]
        metrics = stitch_patch_predictions(example, predicted)
        rows.append(
            {
                "graph_index": example.graph_index,
                "family": example.family,
                "target_degree": example.target_degree,
                **metrics,
            }
        )
    return rows, graph_balanced_stitch_summary(rows)


def _run_fold(
    examples: Sequence[CoverExample],
    fold_index: int,
    *,
    n_atoms: int,
    sparsity: int,
    minimum_sparsity: int,
    n_iterations: int,
    ridge_alpha: float,
) -> dict[str, Any]:
    train_examples, test_examples = _split_examples(examples, fold_index)
    train_ids = {example.graph_index for example in train_examples}
    test_ids = {example.graph_index for example in test_examples}

    raw_train = stack_cover_examples(train_examples)
    train_mean = np.mean(raw_train, axis=1, keepdims=True)
    centered_train = raw_train - train_mean
    initial_dictionary, initialization = deterministic_maximin_initialization(
        centered_train, n_atoms
    )
    final_dictionary, _training_codes, training_info = ksvd(
        centered_train,
        n_atoms=n_atoms,
        T=sparsity,
        T_min=minimum_sparsity,
        n_iter=n_iterations,
        seed=0,
        initial_dictionary=initial_dictionary,
    )

    train_encoded = _encode_examples(
        train_examples,
        final_dictionary,
        train_mean,
        sparsity=sparsity,
        minimum_sparsity=minimum_sparsity,
    )
    test_encoded = _encode_examples(
        test_examples,
        final_dictionary,
        train_mean,
        sparsity=sparsity,
        minimum_sparsity=minimum_sparsity,
    )

    current_features = np.concatenate(
        [item["current_features"] for item in train_encoded], axis=0
    )
    current_targets = np.concatenate(
        [item["centered_targets"] for item in train_encoded], axis=0
    )
    transition_features = np.concatenate(
        [item["transition_features"] for item in train_encoded], axis=0
    )
    transition_targets = np.concatenate(
        [item["centered_targets"][1:] for item in train_encoded], axis=0
    )
    current_decoder = fit_ridge_decoder(
        current_features, current_targets, alpha=ridge_alpha
    )
    transition_decoder = fit_ridge_decoder(
        transition_features, transition_targets, alpha=ridge_alpha
    )

    stage_predictions: dict[str, list[np.ndarray]] = {
        "base_final": [],
        "current_only": [],
        "true_transition": [],
        "shuffled_transition": [],
    }
    shuffle_rows = []
    for item in test_encoded:
        current_centered = predict_ridge_decoder(
            current_decoder, item["current_features"]
        )
        true_centered = current_centered.copy()
        true_centered[1:] = predict_ridge_decoder(
            transition_decoder, item["transition_features"]
        )
        shuffled_features = shuffle_transition_context(
            item["transition_features"], n_atoms
        )
        shuffled_centered = current_centered.copy()
        shuffled_centered[1:] = predict_ridge_decoder(
            transition_decoder, shuffled_features
        )
        mean_row = train_mean.T
        stage_predictions["base_final"].append(item["base"])
        stage_predictions["current_only"].append(current_centered + mean_row)
        stage_predictions["true_transition"].append(true_centered + mean_row)
        stage_predictions["shuffled_transition"].append(
            shuffled_centered + mean_row
        )
        transition = item["transition_features"]
        shuffle_rows.append(
            {
                "graph_index": item["example"].graph_index,
                "current_code_preserved": bool(
                    np.array_equal(shuffled_features[:, :n_atoms], transition[:, :n_atoms])
                ),
                "context_changed": bool(
                    transition.shape[0] <= 1
                    or not np.array_equal(
                        shuffled_features[:, n_atoms:], transition[:, n_atoms:]
                    )
                ),
            }
        )

    stages: dict[str, Any] = {}
    raw_rows, raw_summary = _stage_summary(test_encoded, None)
    stages["raw"] = {"summary": raw_summary, "graphs": raw_rows}
    for stage, predictions in stage_predictions.items():
        rows, summary = _stage_summary(test_encoded, predictions)
        stages[stage] = {"summary": summary, "graphs": rows}

    patch_dimension = raw_train.shape[0]
    expected_transition_dimension = n_atoms * 2 + 3 * len(
        train_examples[0].cover.patches[0].node_ids
    )
    invariants = {
        "train_test_graph_isolation": train_ids.isdisjoint(test_ids),
        "current_feature_dimension": current_features.shape[1] == n_atoms,
        "transition_feature_dimension": transition_features.shape[1]
        == expected_transition_dimension,
        "target_dimension": current_targets.shape[1] == patch_dimension,
        "current_training_rows": current_features.shape[0]
        == sum(len(item["example"].cover.patches) for item in train_encoded),
        "transition_training_rows": transition_features.shape[0]
        == sum(len(item["example"].cover.patches) - 1 for item in train_encoded),
        "finite_features_and_predictions": bool(
            np.all(np.isfinite(current_features))
            and np.all(np.isfinite(transition_features))
            and all(
                np.all(np.isfinite(prediction))
                for predictions in stage_predictions.values()
                for prediction in predictions
            )
        ),
        "shuffle_current_code_preserved": all(
            row["current_code_preserved"] for row in shuffle_rows
        ),
        "shuffle_context_changed_within_graph": all(
            row["context_changed"] for row in shuffle_rows
        ),
    }
    return {
        "fold_index": fold_index,
        "train_graph_count": len(train_examples),
        "test_graph_count": len(test_examples),
        "train_patch_count": int(raw_train.shape[1]),
        "test_patch_count": int(
            sum(len(example.cover.patches) for example in test_examples)
        ),
        "train_graph_indices": sorted(train_ids),
        "test_graph_indices": sorted(test_ids),
        "initialization": initialization,
        "training_reconstruction_curve": [
            float(value) for value in training_info["recon_curve"]
        ],
        "feature_dimensions": {
            "current": int(current_features.shape[1]),
            "transition": int(transition_features.shape[1]),
            "target": int(current_targets.shape[1]),
        },
        "invariants": invariants,
        "shuffle_rows": shuffle_rows,
        "stages": stages,
    }


def _relative_reduction(left: float, right: float) -> float:
    return float((left - right) / max(left, 1e-12))


def classify(folds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    folds = tuple(folds)
    mean_stages = {}
    for stage in (
        "raw",
        "base_final",
        "current_only",
        "true_transition",
        "shuffled_transition",
    ):
        keys = folds[0]["stages"][stage]["summary"].keys()
        mean_stages[stage] = {
            key: float(
                np.mean([fold["stages"][stage]["summary"][key] for fold in folds])
            )
            for key in keys
        }

    raw_invariants = []
    fold_comparisons = []
    current_vs_base_reductions = []
    true_vs_current_reductions = []
    true_vs_shuffled_reductions = []
    for fold in folds:
        stages = fold["stages"]
        raw = stages["raw"]["summary"]
        base = stages["base_final"]["summary"]
        current = stages["current_only"]["summary"]
        true = stages["true_transition"]["summary"]
        shuffled = stages["shuffled_transition"]["summary"]
        raw_invariants.append(
            {
                "patch_exact": raw["patch_relative_error"] == 0.0,
                "observed_exact": raw["observed_pair_rmse"] == 0.0,
                "observed_f1_exact": raw["observed_edge_f1"] == 1.0,
                "overlap_consistent": raw[
                    "repeated_pair_disagreement_std_mean"
                ]
                == 0.0,
            }
        )
        current_gain = _relative_reduction(
            base["observed_pair_rmse"], current["observed_pair_rmse"]
        )
        true_current_gain = _relative_reduction(
            current["observed_pair_rmse"], true["observed_pair_rmse"]
        )
        true_shuffled_gain = _relative_reduction(
            shuffled["observed_pair_rmse"], true["observed_pair_rmse"]
        )
        current_vs_base_reductions.append(current_gain)
        true_vs_current_reductions.append(true_current_gain)
        true_vs_shuffled_reductions.append(true_shuffled_gain)
        comparisons = {
            "rmse_vs_current_at_least_002": true_current_gain >= 0.02,
            "rmse_vs_shuffled_at_least_002": true_shuffled_gain >= 0.02,
            "disagreement_not_worse": true[
                "repeated_pair_disagreement_std_mean"
            ]
            <= current["repeated_pair_disagreement_std_mean"] + 1e-12,
            "observed_f1_not_worse": true["observed_edge_f1"]
            >= current["observed_edge_f1"] - 1e-12,
            "full_edge_recall_preserved": true["full_edge_recall"]
            >= current["full_edge_recall"] - 0.01,
        }
        fold_comparisons.append(comparisons)

    current = mean_stages["current_only"]
    true = mean_stages["true_transition"]
    shuffled = mean_stages["shuffled_transition"]
    base = mean_stages["base_final"]
    mean_true_current_gain = _relative_reduction(
        current["observed_pair_rmse"], true["observed_pair_rmse"]
    )
    mean_true_shuffled_gain = _relative_reduction(
        shuffled["observed_pair_rmse"], true["observed_pair_rmse"]
    )
    mean_comparisons = {
        "rmse_vs_current_at_least_002": mean_true_current_gain >= 0.02,
        "rmse_vs_shuffled_at_least_002": mean_true_shuffled_gain >= 0.02,
        "disagreement_not_worse": true["repeated_pair_disagreement_std_mean"]
        <= current["repeated_pair_disagreement_std_mean"] + 1e-12,
        "observed_f1_not_worse": true["observed_edge_f1"]
        >= current["observed_edge_f1"] - 1e-12,
        "full_edge_recall_preserved": true["full_edge_recall"]
        >= current["full_edge_recall"] - 0.01,
    }
    fold_support_count = sum(all(checks.values()) for checks in fold_comparisons)
    decoder_invariants = all(
        all(fold["invariants"].values()) for fold in folds
    )
    raw_gate = all(all(checks.values()) for checks in raw_invariants)
    transition_gate = (
        fold_support_count >= 2 and all(mean_comparisons.values())
    )
    mean_current_base_gain = _relative_reduction(
        base["observed_pair_rmse"], current["observed_pair_rmse"]
    )
    current_support_count = sum(value >= 0.02 for value in current_vs_base_reductions)
    current_recalibration_gate = (
        current_support_count >= 2 and mean_current_base_gain >= 0.02
    )
    if not raw_gate or not decoder_invariants:
        label = "FAIL_TRANSITION_DECODER_INVARIANTS"
    elif transition_gate:
        label = "PASS_EXPLICIT_TRANSITION_CONTEXT"
    elif current_recalibration_gate:
        label = "CURRENT_DECODER_RECALIBRATION_ONLY"
    else:
        label = "REJECT_LINEAR_TRANSITION_CONTEXT"
    return {
        "classification": label,
        "raw_gate": raw_gate,
        "decoder_invariants": decoder_invariants,
        "raw_invariants_by_fold": raw_invariants,
        "fold_comparisons": fold_comparisons,
        "fold_support_count": int(fold_support_count),
        "mean_comparisons": mean_comparisons,
        "transition_gate": transition_gate,
        "current_recalibration_gate": current_recalibration_gate,
        "current_support_count": int(current_support_count),
        "current_vs_base_rmse_reductions": current_vs_base_reductions,
        "true_vs_current_rmse_reductions": true_vs_current_reductions,
        "true_vs_shuffled_rmse_reductions": true_vs_shuffled_reductions,
        "mean_current_vs_base_rmse_reduction": mean_current_base_gain,
        "mean_true_vs_current_rmse_reduction": mean_true_current_gain,
        "mean_true_vs_shuffled_rmse_reduction": mean_true_shuffled_gain,
        "mean_stages": mean_stages,
    }


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def render_report(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    lines = [
        "# Explicit transition-aware decoder 审计",
        "",
        "> 日期：2026-08-01  ",
        "> FINAL KSVD codes frozen；ridge 只用 train graphs；无 labels。",
        "",
        "## 1. 判定",
        "",
        f"**{decision['classification']}**",
        "",
        "## 2. 三折 held-out 结果",
        "",
        "| fold | train/test graphs | stage | patch rel err | observed RMSE | observed F1 | full edge recall | overlap disagreement |",
        "|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for fold in payload["folds"]:
        for stage in (
            "raw",
            "base_final",
            "current_only",
            "true_transition",
            "shuffled_transition",
        ):
            summary = fold["stages"][stage]["summary"]
            lines.append(
                f"| {fold['fold_index']} | {fold['train_graph_count']}/"
                f"{fold['test_graph_count']} | {stage} | "
                f"{_fmt(summary['patch_relative_error'])} | "
                f"{_fmt(summary['observed_pair_rmse'])} | "
                f"{_fmt(summary['observed_edge_f1'])} | "
                f"{_fmt(summary['full_edge_recall'])} | "
                f"{_fmt(summary['repeated_pair_disagreement_std_mean'])} |"
            )

    lines.extend(
        [
            "",
            "## 3. Fold-balanced stage means",
            "",
            "| stage | patch rel err | observed RMSE | observed F1 | full edge recall | overlap disagreement |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for stage in (
        "raw",
        "base_final",
        "current_only",
        "true_transition",
        "shuffled_transition",
    ):
        mean = decision["mean_stages"][stage]
        lines.append(
            f"| {stage} | {_fmt(mean['patch_relative_error'])} | "
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
            f"- RAW gate：`{decision['raw_gate']}`；decoder invariants：`{decision['decoder_invariants']}`。",
            f"- CURRENT vs BASE mean RMSE reduction：`{_fmt(decision['mean_current_vs_base_rmse_reduction'])}`；2% fold support：`{decision['current_support_count']}/3`。",
            f"- TRUE vs CURRENT mean RMSE reduction：`{_fmt(decision['mean_true_vs_current_rmse_reduction'])}`。",
            f"- TRUE vs SHUFFLED mean RMSE reduction：`{_fmt(decision['mean_true_vs_shuffled_rmse_reduction'])}`。",
            f"- TRUE complete fold support：`{decision['fold_support_count']}/3`；mean checks：`{decision['mean_comparisons']}`。",
            f"- transition gate：`{decision['transition_gate']}`；current recalibration gate：`{decision['current_recalibration_gate']}`。",
            "",
            "## 5. Feature isolation",
            "",
            "- current decoder 输入只有当前 FINAL sparse code。",
            "- transition decoder 的 previous degrees 只由 BASE_FINAL previous reconstruction 计算；不读取 previous raw adjacency。",
            "- SHUFFLED 在每张测试图内部循环错位 context tail，current code 保持逐行不变。",
            "- 每折 feature dimensions、row counts、finite values、train/test graph isolation 均写入 JSON invariants。",
            "",
            "## 6. 解释边界",
            "",
            "- 这是线性、低容量的机制探针；失败不等价于所有非线性 transition model 都失败。",
            "- TRUE 若不优于 SHUFFLED，说明收益不能归因于正确 correspondence binding。",
            "- 本轮只回答无标签 reconstruction，不回答分类或 Transformer 的有效性。",
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
    parser.add_argument("--n-atoms", type=int, default=24)
    parser.add_argument("--sparsity", type=int, default=3)
    parser.add_argument("--minimum-sparsity", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--ridge-alpha", type=float, default=1e-2)
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

    folds = []
    for fold_index in range(3):
        fold = _run_fold(
            examples,
            fold_index,
            n_atoms=args.n_atoms,
            sparsity=args.sparsity,
            minimum_sparsity=args.minimum_sparsity,
            n_iterations=args.iterations,
            ridge_alpha=args.ridge_alpha,
        )
        folds.append(fold)
        stage = fold["stages"]
        print(
            f"fold={fold_index} base/current/true/shuffled_rmse="
            f"{stage['base_final']['summary']['observed_pair_rmse']:.4f}/"
            f"{stage['current_only']['summary']['observed_pair_rmse']:.4f}/"
            f"{stage['true_transition']['summary']['observed_pair_rmse']:.4f}/"
            f"{stage['shuffled_transition']['summary']['observed_pair_rmse']:.4f}",
            flush=True,
        )

    decision = classify(folds)
    payload = {
        "protocol": "ksvd-explicit-transition-decoder-audit-v0-20260801",
        "config": {
            "graph_bank_seed": args.graph_bank_seed,
            "cover_seed": args.cover_seed,
            "families": list(FAMILIES),
            "n_nodes": args.n_nodes,
            "target_degrees": args.degrees,
            "graphs_per_family_degree": args.graphs_per_cell,
            "folds": 3,
            "patch_size": args.patch_size,
            "target_overlap": args.target_overlap,
            "edge_capacity_multiplier": args.edge_capacity_multiplier,
            "n_atoms": args.n_atoms,
            "sparsity": args.sparsity,
            "minimum_sparsity": args.minimum_sparsity,
            "iterations": args.iterations,
            "ridge_alpha": args.ridge_alpha,
            "threshold": 0.5,
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
