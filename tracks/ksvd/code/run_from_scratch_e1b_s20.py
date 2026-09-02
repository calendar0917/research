"""Run frozen E1B-S20: E1-T2 recovery at singleton probability 0.2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .from_scratch_recovery import evaluate_dictionary, fit_and_evaluate
from .from_scratch_singleton_frequency import make_e1_singleton_frequency_dataset
from .run_from_scratch_e1_t2_initialization_audit import (
    is_strict_success,
    numeric_summary,
    parse_int_list,
    run_identifier,
)
from .run_from_scratch_e1_t2_multidata_confirmation import (
    oracle_passes,
    select_by_train_error,
)


DEFAULT_JSON = Path("tracks/ksvd/results/from_scratch/e1b_s20_20260731.json")
DEFAULT_REPORT = Path("tracks/ksvd/results/from_scratch/E1B_S20_20260731.md")
FROZEN_SINGLETON_PROBABILITY = 0.2


def summarize_data_seed(
    *,
    data_seed: int,
    dataset_metadata: dict[str, Any],
    oracle: dict[str, Any],
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = select_by_train_error(runs)
    strict_count = sum(is_strict_success(run) for run in runs)
    contains = strict_count > 0
    selected_success = is_strict_success(selected)
    train_cardinality = dataset_metadata["train_code_cardinality"]
    test_cardinality = dataset_metadata["test_code_cardinality"]
    return {
        "data_seed": int(data_seed),
        "oracle_passed": oracle_passes(oracle),
        "oracle_dictionary_control": oracle,
        "train_code_cardinality": train_cardinality,
        "test_code_cardinality": test_cardinality,
        "train_singleton_rate_in_range": bool(
            0.15 <= float(train_cardinality["singleton_rate"]) <= 0.25
        ),
        "strict_success_candidate_count": int(strict_count),
        "group_contains_strict_success": contains,
        "selected_strict_success": selected_success,
        "selection_miss": bool(contains and not selected_success),
        "selected_run": run_identifier(selected),
        "candidate_runs": runs,
    }


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [result["selected_run"] for result in results]
    all_candidates = [run for result in results for run in result["candidate_runs"]]
    metric_keys = (
        "train_reconstruction_relative",
        "test_reconstruction_relative",
        "mean_atom_cosine",
        "minimum_atom_cosine",
        "support_f1",
        "mean_atom_edge_support_f1",
        "edge_f1",
        "exact_patch_recovery",
    )
    return {
        "data_seed_count": len(results),
        "oracle_pass_count": sum(result["oracle_passed"] for result in results),
        "singleton_rate_check_pass_count": sum(
            result["train_singleton_rate_in_range"] for result in results
        ),
        "group_contains_strict_success_count": sum(
            result["group_contains_strict_success"] for result in results
        ),
        "selected_strict_success_count": sum(
            result["selected_strict_success"] for result in results
        ),
        "selection_miss_count": sum(result["selection_miss"] for result in results),
        "all_candidate_strict_success_count": sum(
            is_strict_success(run) for run in all_candidates
        ),
        "all_candidate_count": len(all_candidates),
        "train_singleton_rate": numeric_summary(
            [float(result["train_code_cardinality"]["singleton_rate"]) for result in results]
        ),
        "test_singleton_rate": numeric_summary(
            [float(result["test_code_cardinality"]["singleton_rate"]) for result in results]
        ),
        "strict_candidates_per_data_seed": numeric_summary(
            [float(result["strict_success_candidate_count"]) for result in results]
        ),
        "selected_metrics": {
            key: numeric_summary([float(run[key]) for run in selected])
            for key in metric_keys
        },
    }


def classify(summary: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "oracle_controls_10_of_10": summary["oracle_pass_count"] == 10,
        "groups_containing_success_ge_9_of_10": summary[
            "group_contains_strict_success_count"
        ]
        >= 9,
        "selectors_successful_ge_9_of_10": summary[
            "selected_strict_success_count"
        ]
        >= 9,
        "selection_miss_count_eq_0": summary["selection_miss_count"] == 0,
        "selected_mean_atom_cosine_ge_0.99": summary["selected_metrics"][
            "mean_atom_cosine"
        ]["mean"]
        >= 0.99,
        "selected_mean_test_reconstruction_le_0.01": summary["selected_metrics"][
            "test_reconstruction_relative"
        ]["mean"]
        <= 0.01,
        "all_train_singleton_rates_in_0.15_0.25": summary[
            "singleton_rate_check_pass_count"
        ]
        == 10,
    }
    passed = all(checks.values())
    if passed:
        label = "PASS_CONTINUE_TO_SINGLETON_P005"
    elif summary["oracle_pass_count"] < summary["data_seed_count"] or summary[
        "singleton_rate_check_pass_count"
    ] < summary["data_seed_count"]:
        label = "FAIL_ORACLE_OR_DATA"
    elif summary["selection_miss_count"] > 0:
        label = "FAIL_SELECTOR"
    elif summary["group_contains_strict_success_count"] < 9:
        label = "FAIL_NO_SUCCESS_BASIN"
    else:
        label = "FAIL_RECOVERY_GATE"
    return {"passed": passed, "classification": label, "checks": checks}


def fmt(summary: dict[str, Any], key: str, digits: int = 4) -> str:
    value = summary[key]
    return f"{value['mean']:.{digits}f} ± {value['std']:.{digits}f}"


def make_report(payload: dict[str, Any]) -> str:
    config = payload["config"]
    summary = payload["summary"]
    decision = payload["decision"]
    metrics = summary["selected_metrics"]
    lines = [
        "# KSVD E1B-S20：singleton probability 0.2 结果",
        "",
        "> 日期：2026-07-31  ",
        "> 协议：`KSVD_E1B_S20_PROTOCOL_20260731.md`  ",
        "> 数字源：`e1b_s20_20260731.json`",
        "",
        "## 1. 冻结配置",
        "",
        f"- singleton probability：`{config['singleton_probability']}`",
        f"- data seeds：`{config['data_seeds'][0]}..{config['data_seeds'][-1]}`",
        f"- learner seeds：`{config['learner_seeds']}`",
        f"- train/test：{config['n_train']}/{config['n_test']}",
        f"- K/T/iterations：4/2/{config['n_iter']}",
        "- selector：5 restarts 中最低 train reconstruction。",
        "",
        "## 2. 数据检查",
        "",
        f"- train singleton rate：{summary['train_singleton_rate']['mean']:.4f} ± {summary['train_singleton_rate']['std']:.4f}，范围 [{summary['train_singleton_rate']['minimum']:.4f}, {summary['train_singleton_rate']['maximum']:.4f}]。",
        f"- test singleton rate：{summary['test_singleton_rate']['mean']:.4f} ± {summary['test_singleton_rate']['std']:.4f}。",
        f"- 位于预注册 `[0.15, 0.25]` 范围的训练集：{summary['singleton_rate_check_pass_count']}/{summary['data_seed_count']}。",
        "",
        "## 3. 恢复结果",
        "",
        f"- oracle controls：**{summary['oracle_pass_count']}/{summary['data_seed_count']}**",
        f"- 五启动 group 包含正确解：**{summary['group_contains_strict_success_count']}/{summary['data_seed_count']}**",
        f"- selector 选中正确解：**{summary['selected_strict_success_count']}/{summary['data_seed_count']}**",
        f"- selection misses：**{summary['selection_miss_count']}**",
        f"- 所有 single-start 候选：{summary['all_candidate_strict_success_count']}/{summary['all_candidate_count']} = {summary['all_candidate_strict_success_count']/summary['all_candidate_count']:.1%}",
        "",
        "### 被选模型指标",
        "",
        "| test reconstruction | atom cosine | minimum atom cosine | support F1 | atom edge-support F1 | edge F1 | exact patch |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {fmt(metrics, 'test_reconstruction_relative')} | {fmt(metrics, 'mean_atom_cosine')} | {fmt(metrics, 'minimum_atom_cosine')} | {fmt(metrics, 'support_f1')} | {fmt(metrics, 'mean_atom_edge_support_f1')} | {fmt(metrics, 'edge_f1')} | {fmt(metrics, 'exact_patch_recovery')} |",
        "",
        "## 4. 每个 data seed",
        "",
        "| data seed | train singleton | strict candidates / 5 | selected seed | train recon | test recon | atom cosine | support F1 | strict | miss |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for result in payload["data_seed_results"]:
        selected = result["selected_run"]
        lines.append(
            f"| {result['data_seed']} | {result['train_code_cardinality']['singleton_rate']:.3f} | {result['strict_success_candidate_count']}/5 | {selected['learner_seed']} | {selected['train_reconstruction_relative']:.6f} | {selected['test_reconstruction_relative']:.6f} | {selected['mean_atom_cosine']:.4f} | {selected['support_f1']:.4f} | {'Y' if result['selected_strict_success'] else 'N'} | {'Y' if result['selection_miss'] else 'N'} |"
        )
    lines.extend(
        [
            "",
            "## 5. 预注册判断",
            "",
            f"- 分类：**{decision['classification']}**。",
        ]
    )
    if decision["passed"]:
        lines.extend(
            [
                "- 结论：当 singleton frequency 从约 50% 降至约 20% 时，五启动 KSVD 仍能稳定恢复结构基。",
                "- 下一步只把 singleton probability 降到 0.05；其他因素继续冻结。",
            ]
        )
    elif decision["classification"] == "FAIL_NO_SUCCESS_BASIN":
        lines.append("- 结论：直接 atom 观测减少后，五启动中正确 basin 不再稳定出现。")
    elif decision["classification"] == "FAIL_SELECTOR":
        lines.append("- 结论：候选中存在正确解，但训练重构 selector 出现误选。")
    elif decision["classification"] == "FAIL_ORACLE_OR_DATA":
        lines.append("- 结论：oracle 或 singleton frequency 检查失败，停止解释 learner。")
    else:
        lines.append("- 结论：恢复指标未达到门槛，暂不继续降低 singleton frequency。")
    lines.extend(["", "## 6. Gate 明细", ""])
    for name, passed in decision["checks"].items():
        lines.append(f"- [{'x' if passed else ' '}] `{name}`")
    lines.extend(
        [
            "",
            "## 7. 判定边界",
            "",
            "- 本轮仍保留每个 atom 的 guaranteed singleton，不能解释为完全从无 singleton 的组合中恢复。",
            "- 本结果不涉及边重叠、噪声、节点置换、随机游走、图级 readout 或真实数据。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-seeds",
        type=parse_int_list,
        default=parse_int_list(",".join(str(seed) for seed in range(20260731, 20260741))),
    )
    parser.add_argument(
        "--learner-seeds", type=parse_int_list, default=parse_int_list("0,1,2,3,4")
    )
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=300)
    parser.add_argument("--n-iter", type=int, default=25)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    if len(args.data_seeds) != 10:
        parser.error("the frozen protocol requires exactly 10 data seeds")
    if len(args.learner_seeds) != 5:
        parser.error("the frozen protocol requires exactly 5 learner seeds")

    results: list[dict[str, Any]] = []
    for position, data_seed in enumerate(args.data_seeds, start=1):
        dataset = make_e1_singleton_frequency_dataset(
            seed=data_seed,
            singleton_probability=FROZEN_SINGLETON_PROBABILITY,
            n_train=args.n_train,
            n_test=args.n_test,
        )
        _oracle_x, oracle = evaluate_dictionary(dataset, dataset.D_true, T=2)
        runs: list[dict[str, Any]] = []
        for learner_seed in args.learner_seeds:
            _dictionary, metrics = fit_and_evaluate(
                dataset,
                learner_seed=learner_seed,
                T=2,
                n_iter=args.n_iter,
            )
            metrics["strict_success"] = is_strict_success(metrics)
            runs.append(metrics)
        result = summarize_data_seed(
            data_seed=data_seed,
            dataset_metadata=dataset.metadata,
            oracle=oracle,
            runs=runs,
        )
        results.append(result)
        selected = result["selected_run"]
        print(
            f"[{position:02d}/{len(args.data_seeds):02d}] data_seed={data_seed} "
            f"singleton={result['train_code_cardinality']['singleton_rate']:.3f} "
            f"strict={result['strict_success_candidate_count']}/5 "
            f"selected_seed={selected['learner_seed']} "
            f"train={selected['train_reconstruction_relative']:.6f} "
            f"atom={selected['mean_atom_cosine']:.4f} "
            f"success={'Y' if result['selected_strict_success'] else 'N'}",
            flush=True,
        )

    summary = aggregate_results(results)
    decision = classify(summary)
    payload = {
        "protocol": "ksvd-e1b-s20-v0-20260731",
        "config": {
            "singleton_probability": FROZEN_SINGLETON_PROBABILITY,
            "guaranteed_singleton_per_atom": 1,
            "data_seeds": args.data_seeds,
            "learner_seeds": args.learner_seeds,
            "n_train": args.n_train,
            "n_test": args.n_test,
            "n_atoms": 4,
            "sparsity": 2,
            "n_iter": args.n_iter,
            "selector": "minimum_train_reconstruction_then_lower_learner_seed",
        },
        "data_seed_results": results,
        "summary": summary,
        "decision": decision,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    args.report.write_text(make_report(payload), encoding="utf-8")
    print(f"Decision: {decision['classification']}")
    print(f"JSON: {args.json}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
