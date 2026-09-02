"""Run the frozen E0/E1 from-scratch K-SVD recovery protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .from_scratch_recovery import (
    RecoveryDataset,
    evaluate_dictionary,
    fit_and_evaluate,
    make_e0_dataset,
    make_e1_dataset,
    pairwise_dictionary_stability,
)


DEFAULT_JSON = Path("tracks/ksvd/results/from_scratch/e0_e1_recovery_20260731.json")
DEFAULT_REPORT = Path("tracks/ksvd/results/from_scratch/E0_E1_RECOVERY_20260731.md")


def parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("values must be unique")
    return values


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = sorted(
        key
        for key, value in runs[0].items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and key not in {"learner_seed", "support_tp", "support_fp", "support_fn"}
    )
    out: dict[str, Any] = {}
    for key in numeric_keys:
        values = np.asarray([float(run[key]) for run in runs], dtype=np.float64)
        out[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "values": values.tolist(),
        }
    return out


def gates_for(
    family: str,
    oracle: dict[str, Any],
    summary: dict[str, Any],
    stability: dict[str, Any],
) -> dict[str, Any]:
    if family == "E0":
        checks = {
            "oracle_reconstruction_relative_le_1e-8": oracle["test_reconstruction_relative"] <= 1e-8,
            "oracle_support_f1_ge_0.999": oracle["support_f1"] >= 0.999,
            "learned_mean_atom_cosine_ge_0.95": summary["mean_atom_cosine"]["mean"] >= 0.95,
            "learned_worst_run_mean_atom_cosine_ge_0.90": summary["mean_atom_cosine"]["minimum"] >= 0.90,
            "learned_mean_support_f1_ge_0.95": summary["support_f1"]["mean"] >= 0.95,
            "learned_mean_test_reconstruction_le_0.05": summary["test_reconstruction_relative"]["mean"] <= 0.05,
            "pairwise_seed_stability_ge_0.90": stability["pairwise_matched_atom_cosine_mean"] >= 0.90,
        }
    elif family == "E1":
        checks = {
            "oracle_edge_f1_eq_1": oracle["edge_f1"] >= 1.0 - 1e-12,
            "oracle_exact_patch_eq_1": oracle["exact_patch_recovery"] >= 1.0 - 1e-12,
            "oracle_support_f1_ge_0.999": oracle["support_f1"] >= 0.999,
            "learned_mean_atom_cosine_ge_0.90": summary["mean_atom_cosine"]["mean"] >= 0.90,
            "learned_mean_atom_edge_support_f1_ge_0.90": summary["mean_atom_edge_support_f1"]["mean"] >= 0.90,
            "learned_mean_code_support_f1_ge_0.90": summary["support_f1"]["mean"] >= 0.90,
            "learned_mean_edge_f1_ge_0.95": summary["edge_f1"]["mean"] >= 0.95,
            "learned_mean_exact_patch_ge_0.90": summary["exact_patch_recovery"]["mean"] >= 0.90,
            "pairwise_seed_stability_ge_0.90": stability["pairwise_matched_atom_cosine_mean"] >= 0.90,
        }
    else:
        raise ValueError(f"unknown experiment family {family}")
    return {"passed": bool(all(checks.values())), "checks": checks}


def run_dataset(
    family: str,
    dataset: RecoveryDataset,
    *,
    learner_seeds: list[int],
    T: int,
    n_iter: int,
) -> dict[str, Any]:
    _oracle_X, oracle = evaluate_dictionary(dataset, dataset.D_true, T=T)
    runs: list[dict[str, Any]] = []
    dictionaries: list[np.ndarray] = []
    for learner_seed in learner_seeds:
        D, metrics = fit_and_evaluate(
            dataset,
            learner_seed=learner_seed,
            T=T,
            n_iter=n_iter,
        )
        dictionaries.append(D)
        runs.append(metrics)
    summary = summarize_runs(runs)
    stability = pairwise_dictionary_stability(dictionaries)
    gates = gates_for(family, oracle, summary, stability)
    best_train_run = min(runs, key=lambda run: float(run["train_reconstruction_relative"]))
    strict_recovery_count = sum(
        float(run["mean_atom_cosine"]) >= 0.99
        and float(run["test_reconstruction_relative"]) <= 0.01
        for run in runs
    )
    return {
        "family": family,
        "dataset": dataset.name,
        "sparsity_condition": T,
        "metadata": dataset.metadata,
        "oracle_dictionary_control": oracle,
        "learner_runs": runs,
        "learner_summary": summary,
        "seed_stability": stability,
        "best_train_reconstruction_run": best_train_run,
        "strict_recovery_seed_count": int(strict_recovery_count),
        "learner_seed_count": len(learner_seeds),
        "gates": gates,
    }


def fmt(summary: dict[str, Any], key: str, digits: int = 4) -> str:
    item = summary[key]
    return f"{item['mean']:.{digits}f} ± {item['std']:.{digits}f}"


def experiment_row(label: str, experiment: dict[str, Any], family: str) -> str:
    summary = experiment["learner_summary"]
    stability = experiment["seed_stability"]["pairwise_matched_atom_cosine_mean"]
    gate = "PASS" if experiment["gates"]["passed"] else "FAIL"
    if family == "E0":
        return (
            f"| {label} | {fmt(summary, 'test_reconstruction_relative')} | "
            f"{fmt(summary, 'mean_atom_cosine')} | {fmt(summary, 'support_f1')} | "
            f"{stability:.4f} | {experiment['strict_recovery_seed_count']}/{experiment['learner_seed_count']} | **{gate}** |"
        )
    return (
        f"| {label} | {fmt(summary, 'test_reconstruction_relative')} | "
        f"{fmt(summary, 'mean_atom_cosine')} | {fmt(summary, 'mean_atom_edge_support_f1')} | "
        f"{fmt(summary, 'support_f1')} | {fmt(summary, 'edge_f1')} | "
        f"{fmt(summary, 'exact_patch_recovery')} | {stability:.4f} | "
        f"{experiment['strict_recovery_seed_count']}/{experiment['learner_seed_count']} | **{gate}** |"
    )


def gate_lines(experiment: dict[str, Any]) -> list[str]:
    return [
        f"- [{'x' if passed else ' '}] `{name}`"
        for name, passed in experiment["gates"]["checks"].items()
    ]


def make_report(payload: dict[str, Any]) -> str:
    experiments = payload["experiments"]
    lines = [
        "# KSVD 从零实验：E0/E1 首轮结果",
        "",
        "> 日期：2026-07-31  ",
        "> 协议：`KSVD_FROM_SCRATCH_PROTOCOL_V0_20260731.md`  ",
        "> 数字源：`e0_e1_recovery_20260731.json`",
        "",
        "## 1. 运行配置",
        "",
        f"- learner seeds：`{payload['config']['learner_seeds']}`",
        f"- sparsity conditions：`{payload['config']['sparsity_conditions']}`",
        f"- train/test patches：{payload['config']['n_train']}/{payload['config']['n_test']}",
        f"- KSVD iterations：{payload['config']['n_iter']}",
        "- support 激活阈值：`max(1e-6, 1e-3 × 该样本最大系数绝对值)`",
        "",
        "## 2. E0：纯数值恢复",
        "",
        "| 条件 | test relative reconstruction | atom cosine | code support F1 | seed stability | strict recovery seeds | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for sparsity in payload["config"]["sparsity_conditions"]:
        key = f"E0_T{sparsity}"
        lines.append(experiment_row(f"T{sparsity}", experiments[key], "E0"))
    lines.extend(
        [
            "",
            "## 3. E1：固定槽位图结构基恢复",
            "",
            "| 条件 | test relative reconstruction | atom cosine | atom edge-support F1 | code support F1 | reconstructed edge F1 | exact patch | seed stability | strict recovery seeds | Gate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for sparsity in payload["config"]["sparsity_conditions"]:
        key = f"E1_T{sparsity}"
        lines.append(experiment_row(f"T{sparsity}", experiments[key], "E1"))

    all_pass = all(exp["gates"]["passed"] for exp in experiments.values())
    e1_t1 = experiments.get("E1_T1")
    e1_t2 = experiments.get("E1_T2")
    lines.extend(
        [
            "",
            "## 4. 当前判断",
            "",
            f"- 全部预注册 gate：**{'PASS' if all_pass else 'NOT YET'}**。",
        ]
    )
    if e1_t1 is not None:
        lines.append(
            f"- E1-T1：{'通过' if e1_t1['gates']['passed'] else '未通过'}。这回答最基础问题：单个固定槽位结构基能否被 ordinary KSVD 恢复。"
        )
    if e1_t2 is not None:
        best = e1_t2["best_train_reconstruction_run"]
        lines.extend(
            [
                f"- E1-T2：{'通过' if e1_t2['gates']['passed'] else '未通过'}；严格恢复 {e1_t2['strict_recovery_seed_count']}/{e1_t2['learner_seed_count']} 个初始化 seed。",
                f"- E1-T2 中按最低训练重构误差选择的 seed 为 {best['learner_seed']}，其 test reconstruction={best['test_reconstruction_relative']:.6f}、atom cosine={best['mean_atom_cosine']:.6f}。该值只作为多启动诊断，不改变 single-start gate。",
            ]
        )
    lines.extend(
        [
            "- 若 T1 通过而 T2 single-start 不稳定，下一步应先研究初始化/多启动与可辨识性，不应直接增加随机游走、置换或真实数据复杂度。",
            "- 高 edge/exact reconstruction 与低 atom recovery 可以同时发生；因此重构 patch 成功不等于恢复了真实生成结构基。",
            "",
            "## 5. 判定边界",
            "",
            "- 本结果只判断 ordinary KSVD 在已知稀疏生成条件下的恢复能力。",
            "- 它不证明随机游走、置换不变表示、图级 readout、真实数据分类或合法 motif prototype 有效。",
            "",
            "## 6. Gate 明细",
            "",
        ]
    )
    for key, experiment in experiments.items():
        lines.extend([f"### {key}", ""])
        lines.extend(gate_lines(experiment))
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=parse_int_list, default=parse_int_list("0,1,2,3,4,5,6,7,8,9"))
    parser.add_argument("--sparsities", type=parse_int_list, default=parse_int_list("1,2"))
    parser.add_argument("--data-seed", type=int, default=20260731)
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=300)
    parser.add_argument("--n-iter", type=int, default=25)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    if any(value not in {1, 2} for value in args.sparsities):
        parser.error("the frozen v0 protocol supports only sparsities 1 and 2")

    builders: tuple[tuple[str, Callable[..., RecoveryDataset]], ...] = (
        ("E0", make_e0_dataset),
        ("E1", make_e1_dataset),
    )
    experiments: dict[str, Any] = {}
    for family, builder in builders:
        for sparsity in args.sparsities:
            dataset = builder(
                seed=args.data_seed,
                n_train=args.n_train,
                n_test=args.n_test,
                max_sparsity=sparsity,
            )
            key = f"{family}_T{sparsity}"
            experiments[key] = run_dataset(
                family,
                dataset,
                learner_seeds=args.seeds,
                T=sparsity,
                n_iter=args.n_iter,
            )

    payload = {
        "protocol": "ksvd-from-scratch-e0-e1-v0-20260731",
        "config": {
            "data_seed": args.data_seed,
            "learner_seeds": args.seeds,
            "sparsity_conditions": args.sparsities,
            "n_train": args.n_train,
            "n_test": args.n_test,
            "n_iter": args.n_iter,
        },
        "experiments": experiments,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.report.write_text(make_report(payload), encoding="utf-8")

    for key, experiment in experiments.items():
        print(f"{key} gate: {'PASS' if experiment['gates']['passed'] else 'FAIL'}")
    print(f"JSON: {args.json}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
