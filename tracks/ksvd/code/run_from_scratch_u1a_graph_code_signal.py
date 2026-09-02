#!/usr/bin/env python3
"""Run U1A graph-level signal and KSVD-added-value evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .from_scratch_unplanted_downstream import build_u1a_features, evaluate_u1a_features
from .from_scratch_unplanted_signal import generate_u0p_dataset


DEFAULT_U0D = Path("tracks/ksvd/results/from_scratch/u0d_dictionary_audit_20260731.json")
DEFAULT_JSON = Path("tracks/ksvd/results/from_scratch/u1a_graph_code_signal_20260731.json")
DEFAULT_REPORT = Path("tracks/ksvd/results/from_scratch/U1A_GRAPH_CODE_SIGNAL_20260731.md")
CONTROL_ORDER = (
    "simple_graph_statistics",
    "walk_mean_std",
    "canonical_mean_std",
    "edge_count_histogram",
    "fixed_gaussian",
    "medoid_bag",
    "pca12",
    "init_codes",
    "final_codes",
    "graph_code_shuffle",
    "label_shuffle",
)


def aggregate(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for key in CONTROL_ORDER:
        scores = np.asarray(
            [seed["controls"][key]["test_balanced_accuracy"] for seed in seed_results],
            dtype=np.float64,
        )
        summary[key] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores, ddof=0)),
            "minimum": float(np.min(scores)),
            "maximum": float(np.max(scores)),
            "replicates_at_least_0_65": int(np.count_nonzero(scores >= 0.65)),
        }
    return summary


def classify(summary: dict[str, Any], seed_results: list[dict[str, Any]], u0d_pass: bool) -> dict[str, Any]:
    final = summary["final_codes"]
    gaussian = summary["fixed_gaussian"]
    init = summary["init_codes"]
    shuffle_pass = all(
        0.45 <= summary[key]["mean"] <= 0.55
        for key in ("graph_code_shuffle", "label_shuffle")
    )
    level2 = bool(
        final["mean"] >= 0.70
        and final["replicates_at_least_0_65"] >= 4
        and final["mean"] - gaussian["mean"] >= 0.03
        and shuffle_pass
    )
    paired_differences = np.asarray(
        [
            seed["controls"]["final_codes"]["test_balanced_accuracy"]
            - seed["controls"]["init_codes"]["test_balanced_accuracy"]
            for seed in seed_results
        ],
        dtype=np.float64,
    )
    level3 = bool(
        u0d_pass
        and level2
        and np.mean(paired_differences) >= 0.02
        and np.count_nonzero(paired_differences > 0.0) >= 4
    )
    if level3:
        classification = "PASS_KSVD_ADDED_VALUE"
        next_step = "Proceed to U1B design: test incremental utility on a target not identical to the generator regime."
    elif level2:
        classification = "PASS_GRAPH_CODE_SIGNAL_BUT_NOT_KSVD_ADDED_VALUE"
        next_step = (
            "Do not attribute graph-level performance to KSVD updates.  The content/code chain is viable, "
            "but INIT versus FINAL evidence does not satisfy the registered added-value gate."
        )
    elif u0d_pass:
        classification = "PASS_OPTIMIZATION_BUT_FAIL_GRAPH_CODE_SIGNAL"
        next_step = (
            "Stop before U1B.  KSVD improved reconstruction, but FINAL graph codes did not satisfy the "
            "registered downstream signal gate against the fixed Gaussian control."
        )
    else:
        classification = "FAIL_LEVEL1_OPTIMIZATION"
        next_step = "U1A cannot support the route because its required U0-D optimization gate failed."
    return {
        "classification": classification,
        "level1_pass_ksvd_optimization": bool(u0d_pass),
        "level2_pass_graph_code_signal": level2,
        "level3_pass_ksvd_added_value": level3,
        "final_minus_gaussian_mean": float(final["mean"] - gaussian["mean"]),
        "final_minus_init": {
            "mean": float(np.mean(paired_differences)),
            "std": float(np.std(paired_differences, ddof=0)),
            "replicates_final_above_init": int(np.count_nonzero(paired_differences > 0.0)),
            "values": paired_differences.tolist(),
        },
        "shuffle_controls_pass": shuffle_pass,
        "next_step": next_step,
    }


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def render_report(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    summary = payload["summary"]
    lines = [
        "# KSVD U1A：无人工原子路线的 graph-code signal 与 added-value gate",
        "",
        "> 日期：2026-07-31",
        ">",
        f"> 正式结论：**{decision['classification']}**",
        "",
        "## 1. 本实验的归因问题",
        "",
        "U0-P 已确认输入 patches 有 regime signal，U0-D 已确认单次初始化 KSVD 明显改善 held-out reconstruction。U1A 现在区分：",
        "",
        "1. FINAL codes 是否包含图级预测信号；",
        "2. 信号是否只是 raw patches、随机投影或 initializer 已经提供；",
        "3. KSVD 的 25 次更新本身是否带来 paired downstream 增益。",
        "",
        "字典训练仍不使用 graph labels。每个 atom 的 graph readout 固定为 activation frequency、mean absolute coefficient 和 coefficient RMS，共 36 维。",
        "",
        "## 2. 五个 replicate 的 test balanced accuracy",
        "",
        "| seed | simple stats | WALK raw | edge hist | Gaussian | medoid | PCA | INIT | FINAL | code shuffle | label shuffle |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in payload["seeds"]:
        controls = seed["controls"]
        lines.append(
            f"| {seed['master_seed']} | {_fmt(controls['simple_graph_statistics']['test_balanced_accuracy'])} | "
            f"{_fmt(controls['walk_mean_std']['test_balanced_accuracy'])} | "
            f"{_fmt(controls['edge_count_histogram']['test_balanced_accuracy'])} | "
            f"{_fmt(controls['fixed_gaussian']['test_balanced_accuracy'])} | "
            f"{_fmt(controls['medoid_bag']['test_balanced_accuracy'])} | "
            f"{_fmt(controls['pca12']['test_balanced_accuracy'])} | "
            f"{_fmt(controls['init_codes']['test_balanced_accuracy'])} | "
            f"{_fmt(controls['final_codes']['test_balanced_accuracy'])} | "
            f"{_fmt(controls['graph_code_shuffle']['test_balanced_accuracy'])} | "
            f"{_fmt(controls['label_shuffle']['test_balanced_accuracy'])} |"
        )
    lines.extend(
        [
            "",
            "Rooted-canonical raw baseline 也被运行并保存在 JSON；表中省略一列仅为可读性。",
            "",
            "## 3. 聚合结果",
            "",
            "| control | mean | std | min | count >= 0.65 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key in CONTROL_ORDER:
        item = summary[key]
        lines.append(
            f"| `{key}` | {_fmt(item['mean'])} | {_fmt(item['std'])} | "
            f"{_fmt(item['minimum'])} | {item['replicates_at_least_0_65']}/5 |"
        )
    lines.extend(
        [
            "",
            "## 4. 分层判定",
            "",
            f"- Level 1 `PASS_KSVD_OPTIMIZATION`：**{'PASS' if decision['level1_pass_ksvd_optimization'] else 'FAIL'}**；",
            f"- Level 2 `PASS_GRAPH_CODE_SIGNAL`：**{'PASS' if decision['level2_pass_graph_code_signal'] else 'FAIL'}**；",
            f"- Level 3 `PASS_KSVD_ADDED_VALUE`：**{'PASS' if decision['level3_pass_ksvd_added_value'] else 'FAIL'}**。",
            "",
            f"FINAL − fixed Gaussian mean = `{_fmt(decision['final_minus_gaussian_mean'])}`。",
            f"FINAL − INIT mean = `{_fmt(decision['final_minus_init']['mean'])}`；"
            f"FINAL 高于 INIT 的 replicate 数 = `{decision['final_minus_init']['replicates_final_above_init']}/5`。",
            f"Graph-code shuffle mean = `{_fmt(summary['graph_code_shuffle']['mean'])}`；"
            f"label shuffle mean = `{_fmt(summary['label_shuffle']['mean'])}`。",
            "",
            "## 5. 结论",
            "",
            f"**{decision['classification']}**",
            "",
            decision["next_step"],
            "",
            "解释时必须把 reconstruction 与 downstream 分开：U0-D 的通过只证明 FINAL 是更好的稀疏重建 basis；只有 Level 3 才能说 KSVD update 对当前图级任务有 added value。",
            "",
            "此外，simple stats 或 edge histogram 更强并不否定 atom 的统计存在，但说明当前生成任务主要由低阶结构统计决定，不能借此声称发现了不可替代的高阶图原子。",
            "",
            "## 6. validation 选择明细",
            "",
            "| seed | control | lambda | validation BA | test BA | active dim |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for seed in payload["seeds"]:
        for key in CONTROL_ORDER:
            item = seed["controls"][key]
            lines.append(
                f"| {seed['master_seed']} | `{key}` | {item['selected_regularization']:.4g} | "
                f"{_fmt(item['validation_balanced_accuracy'])} | {_fmt(item['test_balanced_accuracy'])} | "
                f"{item['active_dimension']} |"
            )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--u0d-json", type=Path, default=DEFAULT_U0D)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    u0d = json.loads(args.u0d_json.read_text(encoding="utf-8"))
    config = u0d["config"]
    seed_results = []
    for stored in u0d["seeds"]:
        master_seed = int(stored["master_seed"])
        root = np.random.SeedSequence(master_seed)
        data_sequence, _u0p_shuffle, graph_shuffle, label_shuffle = root.spawn(4)
        dataset = generate_u0p_dataset(
            data_sequence,
            train_per_class=int(config["train_per_class"]),
            validation_per_class=int(config["validation_per_class"]),
            test_per_class=int(config["test_per_class"]),
            patches_per_graph=int(config["patches_per_graph"]),
        )
        initial = np.asarray(stored["dictionaries"]["init"], dtype=np.float64)
        final = np.asarray(stored["dictionaries"]["final"], dtype=np.float64)
        features, labels = build_u1a_features(
            dataset,
            initial_dictionary=initial,
            final_dictionary=final,
            selected_training_indices=stored["initialization"]["selected_training_indices"],
            n_atoms=int(config["n_atoms"]),
            sparsity=int(config["sparsity"]),
            minimum_sparsity=int(config["minimum_sparsity"]),
            patches_per_graph=int(config["patches_per_graph"]),
        )
        controls = evaluate_u1a_features(
            features,
            labels,
            graph_shuffle_sequence=graph_shuffle,
            label_shuffle_sequence=label_shuffle,
        )
        seed_results.append({"master_seed": master_seed, "controls": controls})
        print(
            f"seed={master_seed} simple={controls['simple_graph_statistics']['test_balanced_accuracy']:.4f} "
            f"gauss={controls['fixed_gaussian']['test_balanced_accuracy']:.4f} "
            f"init={controls['init_codes']['test_balanced_accuracy']:.4f} "
            f"final={controls['final_codes']['test_balanced_accuracy']:.4f} "
            f"shuffle={controls['graph_code_shuffle']['test_balanced_accuracy']:.4f}/"
            f"{controls['label_shuffle']['test_balanced_accuracy']:.4f}",
            flush=True,
        )
    summary = aggregate(seed_results)
    decision = classify(summary, seed_results, bool(u0d["decision"]["passes_u0d"]))
    payload = {
        "protocol": "ksvd-u1a-unplanted-graph-code-signal-v0-20260731",
        "source_u0d_json": str(args.u0d_json),
        "config": config,
        "seeds": seed_results,
        "summary": summary,
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
