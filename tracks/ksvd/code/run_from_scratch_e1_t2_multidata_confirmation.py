"""Confirm the E1-T2 five-restart rule across independent data seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .from_scratch_recovery import evaluate_dictionary, fit_and_evaluate, make_e1_dataset
from .run_from_scratch_e1_t2_initialization_audit import (
    is_strict_success,
    numeric_summary,
    parse_int_list,
    run_identifier,
)


DEFAULT_JSON = Path(
    "tracks/ksvd/results/from_scratch/e1_t2_multidata_confirmation_20260731.json"
)
DEFAULT_REPORT = Path(
    "tracks/ksvd/results/from_scratch/E1_T2_MULTIDATA_CONFIRMATION_20260731.md"
)


def oracle_passes(oracle: dict[str, Any]) -> bool:
    return bool(
        float(oracle["test_reconstruction_relative"]) <= 1e-8
        and float(oracle["support_f1"]) >= 0.999
        and float(oracle["edge_f1"]) >= 1.0 - 1e-12
        and float(oracle["exact_patch_recovery"]) >= 1.0 - 1e-12
    )


def select_by_train_error(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return min(
        runs,
        key=lambda run: (
            float(run["train_reconstruction_relative"]),
            int(run["learner_seed"]),
        ),
    )


def summarize_data_seed(
    *,
    data_seed: int,
    oracle: dict[str, Any],
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = select_by_train_error(runs)
    strict_count = sum(is_strict_success(run) for run in runs)
    contains = strict_count > 0
    selected_success = is_strict_success(selected)
    return {
        "data_seed": int(data_seed),
        "oracle_passed": oracle_passes(oracle),
        "oracle_dictionary_control": oracle,
        "candidate_count": len(runs),
        "strict_success_candidate_count": int(strict_count),
        "group_contains_strict_success": contains,
        "selected_strict_success": selected_success,
        "selection_miss": bool(contains and not selected_success),
        "selected_run": run_identifier(selected),
        "candidate_runs": runs,
    }


def classify_confirmation(summary: dict[str, Any]) -> dict[str, Any]:
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
    }
    passed = all(checks.values())
    if passed:
        classification = "PASS_FREEZE_FIVE_RESTART_RULE"
    elif summary["oracle_pass_count"] < summary["data_seed_count"]:
        classification = "FAIL_ORACLE_OR_METRIC"
    elif summary["selection_miss_count"] > 0:
        classification = "FAIL_SELECTOR"
    elif summary["group_contains_strict_success_count"] < 9:
        classification = "FAIL_NO_SUCCESS_BASIN"
    else:
        classification = "FAIL_CONFIRMATION_GATE"
    return {"passed": passed, "classification": classification, "checks": checks}


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [result["selected_run"] for result in results]
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
    all_candidates = [run for result in results for run in result["candidate_runs"]]
    return {
        "data_seed_count": len(results),
        "oracle_pass_count": sum(result["oracle_passed"] for result in results),
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
        "selected_metrics": {
            key: numeric_summary([float(run[key]) for run in selected])
            for key in metric_keys
        },
        "strict_candidates_per_data_seed": numeric_summary(
            [float(result["strict_success_candidate_count"]) for result in results]
        ),
    }


def fmt(summary: dict[str, Any], key: str, digits: int = 4) -> str:
    value = summary[key]
    return f"{value['mean']:.{digits}f} ± {value['std']:.{digits}f}"


def make_report(payload: dict[str, Any]) -> str:
    config = payload["config"]
    summary = payload["summary"]
    decision = payload["decision"]
    metrics = summary["selected_metrics"]
    lines = [
        "# KSVD E1-T2 跨数据种子五启动确认结果",
        "",
        "> 日期：2026-07-31  ",
        "> 协议：`KSVD_E1_T2_MULTIDATA_CONFIRMATION_PROTOCOL_20260731.md`  ",
        "> 数字源：`e1_t2_multidata_confirmation_20260731.json`",
        "",
        "## 1. 冻结配置",
        "",
        f"- data seeds：`{config['data_seeds'][0]}..{config['data_seeds'][-1]}`（{len(config['data_seeds'])} 个）",
        f"- 每个 data seed 的 learner seeds：`{config['learner_seeds']}`",
        f"- train/test patches：{config['n_train']}/{config['n_test']}",
        f"- K/T/iterations：4/2/{config['n_iter']}",
        "- selector：最低 train reconstruction，相同则选择较小 learner seed。",
        "- strict success：mean atom cosine >= 0.99 且 test reconstruction <= 0.01。",
        "",
        "## 2. 总结果",
        "",
        f"- oracle controls：**{summary['oracle_pass_count']}/{summary['data_seed_count']}**",
        f"- 五候选中包含正确解：**{summary['group_contains_strict_success_count']}/{summary['data_seed_count']}**",
        f"- selector 选中正确解：**{summary['selected_strict_success_count']}/{summary['data_seed_count']}**",
        f"- selection misses：**{summary['selection_miss_count']}**",
        f"- 全部 single-start 候选成功率：{summary['all_candidate_strict_success_count']}/{summary['all_candidate_count']} = {summary['all_candidate_strict_success_count']/summary['all_candidate_count']:.1%}",
        "",
        "### 被选模型指标",
        "",
        "| test reconstruction | atom cosine | minimum atom cosine | support F1 | atom edge-support F1 | edge F1 | exact patch |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {fmt(metrics, 'test_reconstruction_relative')} | {fmt(metrics, 'mean_atom_cosine')} | {fmt(metrics, 'minimum_atom_cosine')} | {fmt(metrics, 'support_f1')} | {fmt(metrics, 'mean_atom_edge_support_f1')} | {fmt(metrics, 'edge_f1')} | {fmt(metrics, 'exact_patch_recovery')} |",
        "",
        "## 3. 每个 data seed",
        "",
        "| data seed | oracle | strict candidates / 5 | selected learner seed | train recon | test recon | atom cosine | support F1 | selected strict | selection miss |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for result in payload["data_seed_results"]:
        selected = result["selected_run"]
        lines.append(
            f"| {result['data_seed']} | {'PASS' if result['oracle_passed'] else 'FAIL'} | {result['strict_success_candidate_count']}/5 | {selected['learner_seed']} | {selected['train_reconstruction_relative']:.6f} | {selected['test_reconstruction_relative']:.6f} | {selected['mean_atom_cosine']:.4f} | {selected['support_f1']:.4f} | {'Y' if result['selected_strict_success'] else 'N'} | {'Y' if result['selection_miss'] else 'N'} |"
        )
    lines.extend(
        [
            "",
            "## 4. 预注册判断",
            "",
            f"- 分类：**{decision['classification']}**。",
        ]
    )
    if decision["passed"]:
        lines.extend(
            [
                "- 结论：E1-T2 后续实验冻结使用 `5 restarts + minimum train reconstruction selector`。",
                "- 下一步只降低 singleton probability，不同时引入边重叠、噪声、置换或采样。",
            ]
        )
    elif decision["classification"] == "FAIL_NO_SUCCESS_BASIN":
        lines.append("- 结论：五启动在部分数据实现上找不到正确 basin，暂不冻结该规则。")
    elif decision["classification"] == "FAIL_SELECTOR":
        lines.append("- 结论：训练 selector 跨数据实现出现误选，需要先研究 objective。")
    elif decision["classification"] == "FAIL_ORACLE_OR_METRIC":
        lines.append("- 结论：oracle 或 evaluator 未通过，停止解释 learner 结果。")
    else:
        lines.append("- 结论：未达到确认门槛，暂不增加数据难度。")
    lines.extend(["", "## 5. Gate 明细", ""])
    for name, passed in decision["checks"].items():
        lines.append(f"- [{'x' if passed else ' '}] `{name}`")
    lines.extend(
        [
            "",
            "## 6. 判定边界",
            "",
            "- 本结果只冻结 E1-T2 synthetic setting 的优化规则。",
            "- 它不证明 singleton 稀缺、边支持重叠、噪声、节点置换、随机游走或真实数据条件下仍可恢复。",
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
    for data_position, data_seed in enumerate(args.data_seeds, start=1):
        dataset = make_e1_dataset(
            seed=data_seed,
            max_sparsity=2,
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
        result = summarize_data_seed(data_seed=data_seed, oracle=oracle, runs=runs)
        results.append(result)
        selected = result["selected_run"]
        print(
            f"[{data_position:02d}/{len(args.data_seeds):02d}] data_seed={data_seed} "
            f"strict={result['strict_success_candidate_count']}/5 "
            f"selected_seed={selected['learner_seed']} "
            f"train={selected['train_reconstruction_relative']:.6f} "
            f"atom={selected['mean_atom_cosine']:.4f} "
            f"success={'Y' if result['selected_strict_success'] else 'N'}",
            flush=True,
        )

    summary = aggregate_results(results)
    decision = classify_confirmation(summary)
    payload = {
        "protocol": "ksvd-e1-t2-multidata-confirmation-v0-20260731",
        "config": {
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
