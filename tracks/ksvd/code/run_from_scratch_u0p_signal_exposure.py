#!/usr/bin/env python3
"""Run the registered U0-P patch representation signal-exposure gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .from_scratch_unplanted_signal import (
    FEATURE_KEYS,
    REGULARIZATION_GRID,
    dataset_summary,
    evaluate_feature_control,
    generate_u0p_dataset,
    shuffled_labels,
)


DEFAULT_SEEDS = (731101, 731102, 731103, 731104, 731105)
DEFAULT_JSON = Path(
    "tracks/ksvd/results/from_scratch/u0p_signal_exposure_20260731.json"
)
DEFAULT_REPORT = Path(
    "tracks/ksvd/results/from_scratch/U0P_SIGNAL_EXPOSURE_20260731.md"
)


def _seed_result(
    master_seed: int,
    *,
    train_per_class: int,
    validation_per_class: int,
    test_per_class: int,
    patches_per_graph: int,
) -> dict[str, Any]:
    root_sequence = np.random.SeedSequence(int(master_seed))
    data_sequence, shuffle_sequence = root_sequence.spawn(2)
    dataset = generate_u0p_dataset(
        data_sequence,
        train_per_class=train_per_class,
        validation_per_class=validation_per_class,
        test_per_class=test_per_class,
        patches_per_graph=patches_per_graph,
    )
    controls = {
        feature_key: evaluate_feature_control(dataset, feature_key)
        for feature_key in FEATURE_KEYS
    }
    label_override = shuffled_labels(dataset, shuffle_sequence)
    label_shuffle = evaluate_feature_control(
        dataset,
        "walk_mean_std",
        label_override=label_override,
    )
    return {
        "master_seed": int(master_seed),
        "dataset": dataset_summary(dataset),
        "controls": controls,
        "label_shuffle": label_shuffle,
    }


def classify(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    controls: dict[str, Any] = {}
    for feature_key in FEATURE_KEYS:
        scores = np.asarray(
            [seed["controls"][feature_key]["test_balanced_accuracy"] for seed in seed_results],
            dtype=np.float64,
        )
        controls[feature_key] = {
            "test_balanced_accuracy_mean": float(np.mean(scores)),
            "test_balanced_accuracy_std": float(np.std(scores, ddof=0)),
            "test_balanced_accuracy_minimum": float(np.min(scores)),
            "replicates_at_least_0_65": int(np.count_nonzero(scores >= 0.65)),
            "passes_real_control_gate": bool(
                np.count_nonzero(scores >= 0.65) >= 4 and np.mean(scores) >= 0.70
            ),
        }
    shuffle_scores = np.asarray(
        [seed["label_shuffle"]["test_balanced_accuracy"] for seed in seed_results],
        dtype=np.float64,
    )
    shuffle_mean = float(np.mean(shuffle_scores))
    shuffle_pass = bool(0.45 <= shuffle_mean <= 0.55)
    passing_controls = [
        key for key, value in controls.items() if value["passes_real_control_gate"]
    ]
    overall_pass = bool(passing_controls and shuffle_pass)
    walk_pass = bool(controls["walk_mean_std"]["passes_real_control_gate"])
    if overall_pass and walk_pass:
        classification = "PASS_U0P_WALK_SIGNAL_EXPOSED"
        next_step = (
            "Proceed to U0-D with walk first-discovery-order adjacency as the primary "
            "15-D KSVD signal; retain INIT versus FINAL and all registered controls."
        )
    elif overall_pass:
        classification = "PASS_U0P_SAMPLER_ONLY_WALK_SUMMARY_WEAK"
        next_step = (
            "Do not automatically run walk-adjacency KSVD.  The patch sampler exposes "
            "regime signal through another registered control, but the frozen walk mean/std "
            "readout did not itself pass; inspect which information the planned code readout can preserve."
        )
    else:
        classification = "FAIL_U0P_SIGNAL_EXPOSURE"
        next_step = (
            "Stop before KSVD.  Reconsider the sampler or use fixed-semantic invariant "
            "statistics; do not add optimization restarts."
        )
    return {
        "classification": classification,
        "passes_u0p": overall_pass,
        "passing_real_controls": passing_controls,
        "walk_primary_passes": walk_pass,
        "controls": controls,
        "label_shuffle": {
            "test_balanced_accuracy_mean": shuffle_mean,
            "test_balanced_accuracy_std": float(np.std(shuffle_scores, ddof=0)),
            "test_balanced_accuracy_minimum": float(np.min(shuffle_scores)),
            "test_balanced_accuracy_maximum": float(np.max(shuffle_scores)),
            "passes_chance_gate": shuffle_pass,
        },
        "next_step": next_step,
    }


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def render_report(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    lines = [
        "# KSVD U0-P：无人工原子路线的 patch signal-exposure gate",
        "",
        "> 日期：2026-07-31",
        ">",
        f"> 正式结论：**{decision['classification']}**",
        "",
        "## 1. 本实验只回答什么",
        "",
        "在训练任何字典之前，检验同一批 walk-induced 6-node patches 是否暴露了 LOW/HIGH rewiring regime 信号。",
        "这不是 KSVD 实验，也不评价 atom；它用于避免在输入本身无信号时靠增加 restart 或字典复杂度补救。",
        "",
        "三组真实 control 使用完全相同的每图 24 个 patches：",
        "",
        "1. `walk_mean_std`：15 个 first-discovery-order adjacency 坐标的均值与标准差（30 维）；",
        "2. `canonical_mean_std`：15 个 rooted-canonical adjacency 坐标的均值与标准差（30 维 baseline）；",
        "3. `edge_count_histogram`：每 patch 的 induced edge count 5..15 频率（11 维）。",
        "",
        "负对照只对 primary `walk_mean_std` 完整管线做一次固定 label shuffle。",
        "",
        "## 2. 冻结实现口径",
        "",
        f"- master data seeds：`{', '.join(str(seed) for seed in payload['config']['seeds'])}`；",
        f"- train / validation / test：每类 `{payload['config']['train_per_class']} / {payload['config']['validation_per_class']} / {payload['config']['test_per_class']}` 图；",
        f"- patches per graph：`{payload['config']['patches_per_graph']}`；",
        "- 图生成：60-node degree-4 ring lattice，LOW 20..40、HIGH 60..80 accepted connected double-edge swaps；",
        "- patch：随机 root，随机游走首次发现 6 个节点，取这 6 个节点的完整 induced adjacency；",
        "- 标准化只拟合 train；常数维删除；",
        "- classifier：确定性 Newton solver 的 L2 logistic regression；",
        "- 目标：mean binary cross entropy + `lambda/2 * ||w||^2`，intercept 不正则；",
        f"- lambda grid：`{list(payload['config']['regularization_grid'])}`；validation 并列时选更大的 lambda；",
        "- 每个候选只在 train 拟合，test 只在 validation 选定后评估一次。",
        "",
        "## 3. 五个 replicate 的 test balanced accuracy",
        "",
        "| seed | WALK mean/std | CAN mean/std | edge histogram | WALK label shuffle |",
        "|---:|---:|---:|---:|---:|",
    ]
    for seed in payload["seeds"]:
        lines.append(
            "| {seed} | {walk} | {can} | {hist} | {shuffle} |".format(
                seed=seed["master_seed"],
                walk=_fmt(seed["controls"]["walk_mean_std"]["test_balanced_accuracy"]),
                can=_fmt(seed["controls"]["canonical_mean_std"]["test_balanced_accuracy"]),
                hist=_fmt(seed["controls"]["edge_count_histogram"]["test_balanced_accuracy"]),
                shuffle=_fmt(seed["label_shuffle"]["test_balanced_accuracy"]),
            )
        )
    lines.extend(
        [
            "",
            "## 4. 聚合 gate",
            "",
            "| control | mean | std | min | count >= 0.65 | real-control gate |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for feature_key in FEATURE_KEYS:
        item = decision["controls"][feature_key]
        lines.append(
            f"| `{feature_key}` | {_fmt(item['test_balanced_accuracy_mean'])} | "
            f"{_fmt(item['test_balanced_accuracy_std'])} | "
            f"{_fmt(item['test_balanced_accuracy_minimum'])} | "
            f"{item['replicates_at_least_0_65']}/5 | "
            f"{'PASS' if item['passes_real_control_gate'] else 'FAIL'} |"
        )
    shuffle = decision["label_shuffle"]
    lines.extend(
        [
            "",
            f"Label shuffle mean = `{_fmt(shuffle['test_balanced_accuracy_mean'])}` "
            f"(range `{_fmt(shuffle['test_balanced_accuracy_minimum'])}`–"
            f"`{_fmt(shuffle['test_balanced_accuracy_maximum'])}`): "
            f"**{'PASS' if shuffle['passes_chance_gate'] else 'FAIL'}** for the registered `[0.45, 0.55]` gate.",
            "",
            f"Passing real controls：`{decision['passing_real_controls']}`。",
            "",
            "## 5. 结论与下一步",
            "",
            f"**{decision['classification']}**",
            "",
            decision["next_step"],
            "",
        ]
    )
    if decision["passes_u0p"]:
        lines.extend(
            [
                "本结论只说明 patch pipeline 暴露了可线性读出的 regime 信息。它不说明：",
                "",
                "- KSVD 一定能保留或增强该信息；",
                "- canonical adjacency 的欧氏几何问题已经消失；",
                "- learned atoms 必须是合法或可命名的图；",
                "- KSVD 会超过 edge histogram 或简单全局图统计。",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "因此本轮不应运行 KSVD。失败发生在 representation/sampler 层，而不是优化层。",
                "",
            ]
        )
    lines.extend(
        [
            "## 6. 每个 seed 的 validation 选择",
            "",
            "| seed | control | selected lambda | train BA | validation BA | test BA | active dim |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for seed in payload["seeds"]:
        for key in (*FEATURE_KEYS, "label_shuffle"):
            item = seed["label_shuffle"] if key == "label_shuffle" else seed["controls"][key]
            lines.append(
                f"| {seed['master_seed']} | `{key}` | {item['selected_regularization']:.4g} | "
                f"{_fmt(item['train_balanced_accuracy'])} | "
                f"{_fmt(item['validation_balanced_accuracy'])} | "
                f"{_fmt(item['test_balanced_accuracy'])} | {item['active_dimension']} |"
            )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--train-per-class", type=int, default=150)
    parser.add_argument("--validation-per-class", type=int, default=50)
    parser.add_argument("--test-per-class", type=int, default=100)
    parser.add_argument("--patches-per-graph", type=int, default=24)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_results = []
    for seed in args.seeds:
        result = _seed_result(
            seed,
            train_per_class=args.train_per_class,
            validation_per_class=args.validation_per_class,
            test_per_class=args.test_per_class,
            patches_per_graph=args.patches_per_graph,
        )
        seed_results.append(result)
        print(
            f"seed={seed} "
            f"walk={result['controls']['walk_mean_std']['test_balanced_accuracy']:.4f} "
            f"can={result['controls']['canonical_mean_std']['test_balanced_accuracy']:.4f} "
            f"hist={result['controls']['edge_count_histogram']['test_balanced_accuracy']:.4f} "
            f"shuffle={result['label_shuffle']['test_balanced_accuracy']:.4f}",
            flush=True,
        )
    decision = classify(seed_results)
    payload = {
        "protocol": "ksvd-u0p-unplanted-signal-exposure-v0-20260731",
        "config": {
            "seeds": [int(seed) for seed in args.seeds],
            "train_per_class": int(args.train_per_class),
            "validation_per_class": int(args.validation_per_class),
            "test_per_class": int(args.test_per_class),
            "patches_per_graph": int(args.patches_per_graph),
            "patch_size": 6,
            "low_accepted_swaps": [20, 40],
            "high_accepted_swaps": [60, 80],
            "regularization_grid": list(REGULARIZATION_GRID),
            "regularization_semantics": "mean_log_loss_plus_lambda_half_l2",
            "ksvd_run": False,
        },
        "seeds": seed_results,
        "decision": decision,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.report.write_text(render_report(payload), encoding="utf-8")
    print(f"decision={decision['classification']}")
    print(f"wrote {args.json}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
