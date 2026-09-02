#!/usr/bin/env python3
"""Run R0-P on the complete raw IMDB-BINARY structure-only dataset.

The run has three separate outputs:
1. exact-isomorphism repetition/label-conflict audit on raw and cleaned variants;
2. six-node WALK patch substrate audit on all 1,000 raw graphs;
3. exploratory, non-KSVD signal exposure under ordinary raw stratified folds.

No dictionary is trained here and no test result is used to tune K/T/readout.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .from_scratch_unplanted_signal import evaluate_feature_matrices
from .imdb_walk_substrate import (
    exact_isomorphism_groups,
    extract_walk_patch_graphs,
    feature_matrices,
    isomorphism_summary,
    load_tu_structure_text,
    mapped_trajectory_invariance_audit,
    patch_substrate_summary,
)


DEFAULT_SPLIT_SEEDS = (731201, 731202, 731203)
DEFAULT_DATASET_ROOT = Path("data/TUD/IMDB-BINARY")
DEFAULT_JSON = Path("tracks/ksvd/results/from_scratch/imdb_binary_r0p_walk_audit_20260731.json")
DEFAULT_REPORT = Path("tracks/ksvd/results/from_scratch/IMDB_BINARY_R0P_WALK_AUDIT_20260731.md")


def _stratified_folds(labels: np.ndarray, n_splits: int, seed: int) -> list[np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    per_class = {}
    for label in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == label)
        per_class[label] = np.array_split(indices[rng.permutation(indices.size)], n_splits)
    return [
        np.sort(np.concatenate([per_class[label][fold] for label in sorted(per_class)]))
        for fold in range(n_splits)
    ]


def _train_validation_split(indices: np.ndarray, labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    train_parts = []
    validation_parts = []
    for label in sorted(set(labels[indices].tolist())):
        class_indices = indices[labels[indices] == label]
        shuffled = class_indices[rng.permutation(class_indices.size)]
        n_validation = max(1, int(round(0.2 * shuffled.size)))
        validation_parts.append(shuffled[:n_validation])
        train_parts.append(shuffled[n_validation:])
    return np.sort(np.concatenate(train_parts)), np.sort(np.concatenate(validation_parts))


def _evaluate_raw_signal(
    matrices: dict[str, np.ndarray],
    labels: np.ndarray,
    *,
    split_seeds: tuple[int, ...],
    n_splits: int = 5,
) -> dict[str, Any]:
    feature_keys = tuple(matrices)
    seed_results = []
    for split_seed in split_seeds:
        folds = _stratified_folds(labels, n_splits, split_seed)
        shuffle_rng = np.random.default_rng(split_seed + 10_000_019)
        shuffled = labels[shuffle_rng.permutation(labels.size)]
        fold_results = []
        for fold, test_indices in enumerate(folds):
            outer_train = np.setdiff1d(np.arange(labels.size), test_indices, assume_unique=True)
            train_indices, validation_indices = _train_validation_split(
                outer_train, labels, split_seed * 101 + fold
            )
            results = {}
            for feature_key in feature_keys:
                split_matrices = {
                    "train": matrices[feature_key][train_indices],
                    "validation": matrices[feature_key][validation_indices],
                    "test": matrices[feature_key][test_indices],
                }
                split_labels = {
                    "train": labels[train_indices],
                    "validation": labels[validation_indices],
                    "test": labels[test_indices],
                }
                results[feature_key] = evaluate_feature_matrices(
                    split_matrices, split_labels, feature_key=feature_key
                )
            shuffled_result = evaluate_feature_matrices(
                {
                    "train": matrices["walk_mean_std"][train_indices],
                    "validation": matrices["walk_mean_std"][validation_indices],
                    "test": matrices["walk_mean_std"][test_indices],
                },
                {
                    "train": shuffled[train_indices],
                    "validation": shuffled[validation_indices],
                    "test": shuffled[test_indices],
                },
                feature_key="walk_mean_std_label_shuffle",
            )
            fold_results.append(
                {
                    "fold": int(fold),
                    "train_size": int(train_indices.size),
                    "validation_size": int(validation_indices.size),
                    "test_size": int(test_indices.size),
                    "features": results,
                    "walk_label_shuffle": shuffled_result,
                }
            )
        seed_results.append({"split_seed": int(split_seed), "folds": fold_results})

    summary = {}
    for feature_key in feature_keys:
        per_seed = []
        all_folds = []
        for seed_result in seed_results:
            scores = [
                fold["features"][feature_key]["test_balanced_accuracy"]
                for fold in seed_result["folds"]
            ]
            per_seed.append(float(np.mean(scores)))
            all_folds.extend(scores)
        summary[feature_key] = {
            "per_split_seed_mean_test_balanced_accuracy": per_seed,
            "mean_test_balanced_accuracy": float(np.mean(per_seed)),
            "std_across_split_seed_means": float(np.std(per_seed, ddof=0)),
            "minimum_fold_test_balanced_accuracy": float(np.min(all_folds)),
            "maximum_fold_test_balanced_accuracy": float(np.max(all_folds)),
        }
    shuffle_per_seed = []
    shuffle_folds = []
    for seed_result in seed_results:
        scores = [
            fold["walk_label_shuffle"]["test_balanced_accuracy"]
            for fold in seed_result["folds"]
        ]
        shuffle_per_seed.append(float(np.mean(scores)))
        shuffle_folds.extend(scores)
    summary["walk_label_shuffle"] = {
        "per_split_seed_mean_test_balanced_accuracy": shuffle_per_seed,
        "mean_test_balanced_accuracy": float(np.mean(shuffle_per_seed)),
        "std_across_split_seed_means": float(np.std(shuffle_per_seed, ddof=0)),
        "minimum_fold_test_balanced_accuracy": float(np.min(shuffle_folds)),
        "maximum_fold_test_balanced_accuracy": float(np.max(shuffle_folds)),
    }
    return {"summary": summary, "seed_results": seed_results}


def _decision(payload: dict[str, Any]) -> dict[str, Any]:
    substrate = payload["raw_patch_substrate"]
    invariance = payload["mapped_trajectory_invariance"]
    signal = payload["raw_stratified_signal"]["summary"]
    substrate_checks = {
        "dominant_canonical_signature_le_0_50": substrate["dominant_canonical_signature_fraction"] <= 0.50,
        "canonical_effective_signature_count_ge_10": substrate["canonical_effective_signature_count"] >= 10.0,
        "median_within_graph_canonical_unique_fraction_ge_0_20": substrate["within_graph_canonical_unique_fraction"]["median"] >= 0.20,
        "mapped_trajectory_exact_invariance": invariance["passes_exact_mapped_trajectory_gate"],
    }
    substrate_pass = all(substrate_checks.values())
    walk_score = signal["walk_mean_std"]["mean_test_balanced_accuracy"]
    shuffle_score = signal["walk_label_shuffle"]["mean_test_balanced_accuracy"]
    signal_checks = {
        "walk_mean_ba_ge_0_60": walk_score >= 0.60,
        "walk_exceeds_shuffle_by_0_05": walk_score - shuffle_score >= 0.05,
        "shuffle_mean_in_0_45_0_55": 0.45 <= shuffle_score <= 0.55,
    }
    signal_pass = all(signal_checks.values())
    if substrate_pass and signal_pass:
        classification = "PASS_R0P_RAW_WALK_SUBSTRATE_AND_SIGNAL"
        next_step = (
            "Freeze raw-stratified and exact-isomorphism-grouped folds, then run a paired "
            "single-INIT versus FINAL held-out reconstruction audit. Keep cleaned as sensitivity only."
        )
    elif substrate_pass:
        classification = "PASS_R0P_SUBSTRATE_BUT_PATCH_SIGNAL_WEAK"
        next_step = "Stop before KSVD downstream evaluation; inspect sampler scale without adding restarts."
    else:
        classification = "FAIL_R0P_RAW_WALK_SUBSTRATE"
        next_step = "Stop before dictionary learning and revise only the patch substrate/representation."
    return {
        "classification": classification,
        "substrate_checks": substrate_checks,
        "signal_checks": signal_checks,
        "passes_substrate": substrate_pass,
        "passes_signal": signal_pass,
        "next_step": next_step,
        "status_note": "Exploratory transfer gate, not a confirmatory benchmark claim.",
    }


def _f(value: float) -> str:
    return f"{value:.4f}"


def render_report(payload: dict[str, Any]) -> str:
    raw_iso = payload["raw_isomorphism"]
    cleaned_iso = payload["cleaned_isomorphism"]
    substrate = payload["raw_patch_substrate"]
    invariant = payload["mapped_trajectory_invariance"]
    signal = payload["raw_stratified_signal"]["summary"]
    decision = payload["decision"]
    lines = [
        "# IMDB-BINARY R0-P：完整 raw 数据上的 WALK substrate 审计",
        "",
        "> 日期：2026-07-31",
        ">",
        f"> 探索性结论：**{decision['classification']}**",
        "",
        "## 1. 本轮边界",
        "",
        "- 主数据是完整 raw IMDB-BINARY（1000 图），不是 cleaned 替代 benchmark。",
        "- cleaned 只用于解释重复图和冲突标签，不用于替代 raw 分数。",
        "- 本轮不训练 KSVD；只审计真实 patch substrate、置换性质和非 KSVD signal exposure。",
        "- 普通 raw stratified CV 是 benchmark view；exact-isomorphism grouped CV 留到下一阶段。",
        "",
        "## 2. Raw 与 cleaned 的精确同构结构",
        "",
        "| 指标 | raw | cleaned |",
        "|---|---:|---:|",
        f"| 图数 | {raw_iso['graph_count']} | {cleaned_iso['graph_count']} |",
        f"| exact structure groups | {raw_iso['exact_structure_group_count']} | {cleaned_iso['exact_structure_group_count']} |",
        f"| duplicate groups | {raw_iso['duplicate_group_count']} | {cleaned_iso['duplicate_group_count']} |",
        f"| graphs in duplicate groups | {raw_iso['graphs_in_duplicate_groups']} | {cleaned_iso['graphs_in_duplicate_groups']} |",
        f"| label-conflict groups | {raw_iso['label_conflict_group_count']} | {cleaned_iso['label_conflict_group_count']} |",
        f"| graphs in label-conflict groups | {raw_iso['graphs_in_label_conflict_groups']} | {cleaned_iso['graphs_in_label_conflict_groups']} |",
        f"| pure-structure empirical ceiling | {_f(raw_iso['structure_only_deterministic_empirical_ceiling_accuracy'])} | {_f(cleaned_iso['structure_only_deterministic_empirical_ceiling_accuracy'])} |",
        "",
        f"Raw 有 {raw_iso['exact_structure_group_count']} 个不同结构；其中 "
        f"{raw_iso['label_conflict_group_count']} 个结构组包含互相冲突的 graph labels，"
        f"涉及 {raw_iso['graphs_in_label_conflict_groups']} 张图。"
        f"去掉冲突组后剩 {raw_iso['label_consistent_group_count']} 个 label-consistent structure groups；"
        f"其类别代表数为 {raw_iso['label_consistent_representative_counts']}。",
        f"因此任何对精确同构严格不变的纯结构确定性分类器，即使在这 1000 张图上记住每个结构并按组内多数标签预测，"
        f"经验 accuracy 上限也只有 {_f(raw_iso['structure_only_deterministic_empirical_ceiling_accuracy'])}；"
        f"至少 {raw_iso['structure_only_unavoidable_conflict_errors']} 个样本不可同时判对。",
        "",
        "这解释了为什么 cleaned 适合敏感性分析，但也说明它改变了任务：它不仅去重，还排除了 raw 中无法由纯结构唯一判定的冲突样本。",
        "",
        "## 3. Raw WALK patch substrate",
        "",
        f"- 图数：{substrate['graph_count']}；patch 数：{substrate['patch_count']}。",
        f"- 每图 patch 数 min/median/max：{substrate['patches_per_graph']['minimum']}/"
        f"{substrate['patches_per_graph']['median']:.1f}/{substrate['patches_per_graph']['maximum']}。",
        f"- 根覆盖率 min/median/mean：{_f(substrate['root_coverage']['minimum'])}/"
        f"{_f(substrate['root_coverage']['median'])}/{_f(substrate['root_coverage']['mean'])}；"
        f"全根覆盖图数：{substrate['root_coverage']['full_coverage_graph_count']}。",
        f"- patch edge mean：{_f(substrate['patch_edge_mean'])}。",
        f"- tree fraction：{_f(substrate['tree_fraction'])}。",
        f"- clique fraction：{_f(substrate['clique_fraction'])}。",
        f"- missing-at-most-one-edge fraction：{_f(substrate['near_clique_fraction_missing_at_most_one_edge'])}。",
        f"- WALK unique vectors：{substrate['walk_unique_vector_count']}。",
        f"- rooted-canonical unique signatures：{substrate['canonical_unique_signature_count']}。",
        f"- canonical effective signature count：{_f(substrate['canonical_effective_signature_count'])}。",
        f"- dominant canonical signature mass：{_f(substrate['dominant_canonical_signature_fraction'])}。",
        f"- within-graph canonical unique fraction min/median/mean："
        f"{_f(substrate['within_graph_canonical_unique_fraction']['minimum'])}/"
        f"{_f(substrate['within_graph_canonical_unique_fraction']['median'])}/"
        f"{_f(substrate['within_graph_canonical_unique_fraction']['mean'])}。",
        "",
        "Patch edge-count histogram：",
        "",
        "```text",
        json.dumps(substrate["edge_count_histogram"], ensure_ascii=False, sort_keys=True),
        "```",
        "",
        "## 4. 节点重编号审计",
        "",
        f"对 {invariant['graphs_checked']} 张图、{invariant['patches_checked']} 条已采样 walk trajectory 进行整体节点重编号后映射："
        f"mismatch={invariant['mismatch_count']}，maximum L1={_f(invariant['maximum_l1_difference'])}。",
        "",
        "这里验证的是严格命题：给定同一条抽象 walk trajectory，first-discovery-order induced adjacency 不随原始节点 ID 改变。独立随机采样的逐样本结果不要求完全相同。",
        "",
        "## 5. Raw stratified signal exposure（尚未训练 KSVD）",
        "",
        "| 特征 | 3 split-seed mean BA | split-seed std | per-seed means |",
        "|---|---:|---:|---|",
    ]
    ordered = [
        "graph_stats",
        "walk_mean_std",
        "canonical_mean_std",
        "edge_count_histogram",
        "graph_stats_plus_walk_mean_std",
        "graph_stats_plus_canonical_mean_std",
        "graph_stats_plus_edge_count_histogram",
        "walk_label_shuffle",
    ]
    for key in ordered:
        item = signal[key]
        means = ", ".join(_f(value) for value in item["per_split_seed_mean_test_balanced_accuracy"])
        lines.append(
            f"| {key} | {_f(item['mean_test_balanced_accuracy'])} | "
            f"{_f(item['std_across_split_seed_means'])} | [{means}] |"
        )
    lines.extend(
        [
            "",
            "这些分数只证明 raw WALK summaries 是否暴露标签信号。KSVD 的 added value 必须在下一阶段通过同一 fold 内的 `STATS+INIT` vs `STATS+FINAL` 单独归因。",
            "",
            "## 6. Gate",
            "",
            "### Substrate checks",
            "",
        ]
    )
    for key, value in decision["substrate_checks"].items():
        lines.append(f"- [{'x' if value else ' '}] `{key}`")
    lines.extend(["", "### Signal checks", ""])
    for key, value in decision["signal_checks"].items():
        lines.append(f"- [{'x' if value else ' '}] `{key}`")
    lines.extend(
        [
            "",
            f"下一步：{decision['next_step']}",
            "",
            "## 7. 下一阶段必须保留的三种视角",
            "",
            "1. `raw/stratified`：与常见 IMDB-BINARY benchmark 设置接近，用于外部参考。",
            "2. `raw/exact-isomorphism-grouped`：保留 1000 图，但同构组不跨折，用于结构泛化判断。",
            "3. `cleaned/stratified`：只做去重与冲突样本移除后的敏感性分析，不替代 raw benchmark。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--sampling-seed", type=int, default=20260731)
    parser.add_argument("--split-seeds", nargs="+", type=int, default=list(DEFAULT_SPLIT_SEEDS))
    parser.add_argument("--patch-size", type=int, default=6)
    parser.add_argument("--max-patches-per-graph", type=int, default=24)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    warnings.filterwarnings(
        "ignore", message="The hashes produced for graphs without node or edge attributes changed"
    )
    raw = load_tu_structure_text(args.dataset_root, cleaned=False)
    cleaned = load_tu_structure_text(args.dataset_root, cleaned=True)
    print(f"loaded raw={len(raw)} cleaned={len(cleaned)}", flush=True)

    raw_groups = exact_isomorphism_groups(raw)
    cleaned_groups = exact_isomorphism_groups(cleaned)
    raw_iso = isomorphism_summary(raw_groups, len(raw))
    cleaned_iso = isomorphism_summary(cleaned_groups, len(cleaned))
    print(
        f"raw groups={raw_iso['exact_structure_group_count']} conflicts={raw_iso['label_conflict_group_count']} "
        f"cleaned groups={cleaned_iso['exact_structure_group_count']}",
        flush=True,
    )

    examples = extract_walk_patch_graphs(
        raw,
        sampling_seed=args.sampling_seed,
        patch_size=args.patch_size,
        max_patches_per_graph=args.max_patches_per_graph,
    )
    substrate = patch_substrate_summary(examples)
    invariant = mapped_trajectory_invariance_audit(
        raw, examples, seed=args.sampling_seed, graph_limit=100
    )
    print(
        f"patches={substrate['patch_count']} clique={substrate['clique_fraction']:.4f} "
        f"dominant={substrate['dominant_canonical_signature_fraction']:.4f} "
        f"effective={substrate['canonical_effective_signature_count']:.2f}",
        flush=True,
    )

    matrices, labels = feature_matrices(examples)
    signal = _evaluate_raw_signal(
        matrices, labels, split_seeds=tuple(args.split_seeds), n_splits=5
    )
    for key, item in signal["summary"].items():
        print(f"{key}: BA={item['mean_test_balanced_accuracy']:.4f}", flush=True)

    payload: dict[str, Any] = {
        "protocol": "imdb-binary-r0p-raw-walk-substrate-v0-20260731",
        "config": {
            "dataset_root": str(args.dataset_root),
            "sampling_seed": int(args.sampling_seed),
            "split_seeds": [int(value) for value in args.split_seeds],
            "patch_size": int(args.patch_size),
            "max_patches_per_graph": int(args.max_patches_per_graph),
            "outer_folds": 5,
            "validation_fraction_within_outer_train": 0.2,
            "structure_only": True,
            "dictionary_trained": False,
        },
        "raw_isomorphism": raw_iso,
        "cleaned_isomorphism": cleaned_iso,
        "raw_patch_substrate": substrate,
        "mapped_trajectory_invariance": invariant,
        "raw_stratified_signal": signal,
    }
    payload["decision"] = _decision(payload)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    args.report.write_text(render_report(payload))
    print(f"decision={payload['decision']['classification']}", flush=True)
    print(f"wrote {args.json}", flush=True)
    print(f"wrote {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
