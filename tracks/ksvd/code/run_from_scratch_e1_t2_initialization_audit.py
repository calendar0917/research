"""Audit E1-T2 K-SVD initialization basins and train-error multi-start selection."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .from_scratch_recovery import (
    fit_and_evaluate,
    make_e1_dataset,
    normalize_columns,
    optimal_atom_alignment,
)


DEFAULT_JSON = Path(
    "tracks/ksvd/results/from_scratch/e1_t2_initialization_audit_20260731.json"
)
DEFAULT_REPORT = Path(
    "tracks/ksvd/results/from_scratch/E1_T2_INITIALIZATION_AUDIT_20260731.md"
)
STRICT_ATOM_COSINE = 0.99
STRICT_TEST_RECONSTRUCTION = 0.01


def parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("values must be unique")
    return values


def is_strict_success(run: dict[str, Any]) -> bool:
    return bool(
        float(run["mean_atom_cosine"]) >= STRICT_ATOM_COSINE
        and float(run["test_reconstruction_relative"])
        <= STRICT_TEST_RECONSTRUCTION
    )


def reconstruct_initialization(
    dataset: Any,
    learner_seed: int,
) -> dict[str, Any]:
    """Reproduce the default initialization calls in ksvd() without fitting."""
    rng = np.random.default_rng(learner_seed)
    n_atoms = dataset.D_true.shape[1]
    n_samples = dataset.Y_train.shape[1]
    indices = rng.choice(n_samples, size=n_atoms, replace=n_samples < n_atoms)
    initial = dataset.Y_train[:, indices].astype(np.float64).copy()
    initial += 1e-3 * rng.standard_normal(initial.shape)
    initial = normalize_columns(initial)
    alignment = optimal_atom_alignment(dataset.D_true, initial)

    support_sets: list[list[int]] = []
    singleton_atoms: list[int] = []
    all_atoms: set[int] = set()
    for index in indices:
        support = np.flatnonzero(dataset.X_train_true[:, int(index)] > 1e-12)
        support_list = [int(value) for value in support]
        support_sets.append(support_list)
        all_atoms.update(support_list)
        if len(support_list) == 1:
            singleton_atoms.append(support_list[0])

    return {
        "training_column_indices": [int(value) for value in indices],
        "training_column_true_supports": support_sets,
        "singleton_column_count": len(singleton_atoms),
        "distinct_singleton_atom_count": len(set(singleton_atoms)),
        "all_column_atom_coverage_count": len(all_atoms),
        "covers_all_atoms": len(all_atoms) == n_atoms,
        "initial_mean_atom_cosine": float(alignment["mean_atom_cosine"]),
        "initial_minimum_atom_cosine": float(alignment["minimum_atom_cosine"]),
    }


def pearson_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.size != y.size or x.size < 2:
        return None
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    if denominator <= 1e-15:
        return None
    return float((x_centered @ y_centered) / denominator)


def average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and array[order[end]] == array[order[start]]:
            end += 1
        average = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average
        start = end
    return ranks


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    return pearson_correlation(average_ranks(left), average_ranks(right))


def numeric_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "std": None, "minimum": None, "maximum": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def exact_containment_probability(
    *,
    population_size: int,
    success_count: int,
    budget: int,
) -> float:
    if not 0 <= success_count <= population_size:
        raise ValueError("invalid success count")
    if not 1 <= budget <= population_size:
        raise ValueError("budget must lie in [1, population_size]")
    failure_count = population_size - success_count
    if budget > failure_count:
        return 1.0
    return float(
        1.0
        - math.comb(failure_count, budget) / math.comb(population_size, budget)
    )


def select_by_train_error(indices: np.ndarray, runs: list[dict[str, Any]]) -> int:
    return min(
        (int(index) for index in indices),
        key=lambda index: (
            float(runs[index]["train_reconstruction_relative"]),
            int(runs[index]["learner_seed"]),
        ),
    )


def simulate_restart_budget(
    runs: list[dict[str, Any]],
    *,
    budget: int,
    n_trials: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    n_candidates = len(runs)
    success = np.asarray([is_strict_success(run) for run in runs], dtype=bool)
    contains_count = 0
    selected_success_count = 0
    selected_atom_cosines: list[float] = []
    selected_support_f1: list[float] = []
    selected_test_reconstruction: list[float] = []
    selected_train_reconstruction: list[float] = []
    atom_cosine_regrets: list[float] = []
    support_f1_regrets: list[float] = []

    for _ in range(n_trials):
        group = rng.choice(n_candidates, size=budget, replace=False)
        contains_count += int(bool(np.any(success[group])))
        selected = select_by_train_error(group, runs)
        selected_success_count += int(success[selected])
        chosen = runs[selected]
        selected_atom_cosines.append(float(chosen["mean_atom_cosine"]))
        selected_support_f1.append(float(chosen["support_f1"]))
        selected_test_reconstruction.append(
            float(chosen["test_reconstruction_relative"])
        )
        selected_train_reconstruction.append(
            float(chosen["train_reconstruction_relative"])
        )
        best_atom_cosine = max(float(runs[int(i)]["mean_atom_cosine"]) for i in group)
        best_support_f1 = max(float(runs[int(i)]["support_f1"]) for i in group)
        atom_cosine_regrets.append(
            best_atom_cosine - float(chosen["mean_atom_cosine"])
        )
        support_f1_regrets.append(best_support_f1 - float(chosen["support_f1"]))

    contains_probability = contains_count / n_trials
    selected_probability = selected_success_count / n_trials
    efficiency = (
        selected_probability / contains_probability
        if contains_probability > 0.0
        else None
    )
    return {
        "budget": budget,
        "n_trials": n_trials,
        "group_contains_strict_success_probability": float(contains_probability),
        "exact_group_contains_strict_success_probability": exact_containment_probability(
            population_size=n_candidates,
            success_count=int(success.sum()),
            budget=budget,
        ),
        "selector_selects_strict_success_probability": float(selected_probability),
        "selection_efficiency": float(efficiency) if efficiency is not None else None,
        "selected_mean_atom_cosine": numeric_summary(selected_atom_cosines),
        "selected_support_f1": numeric_summary(selected_support_f1),
        "selected_test_reconstruction_relative": numeric_summary(
            selected_test_reconstruction
        ),
        "selected_train_reconstruction_relative": numeric_summary(
            selected_train_reconstruction
        ),
        "atom_cosine_regret": numeric_summary(atom_cosine_regrets),
        "support_f1_regret": numeric_summary(support_f1_regrets),
    }


def run_identifier(run: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "learner_seed",
        "strict_success",
        "train_reconstruction_relative",
        "test_reconstruction_relative",
        "mean_atom_cosine",
        "minimum_atom_cosine",
        "support_f1",
        "mean_atom_edge_support_f1",
        "edge_f1",
        "exact_patch_recovery",
    )
    return {key: run[key] for key in keys}


def grouped_initialization_success(
    runs: list[dict[str, Any]], key: str
) -> list[dict[str, Any]]:
    values = sorted(
        {run["initialization_audit"][key] for run in runs},
        key=lambda value: (str(type(value)), value),
    )
    groups: list[dict[str, Any]] = []
    for value in values:
        members = [
            run for run in runs if run["initialization_audit"][key] == value
        ]
        success_count = sum(is_strict_success(run) for run in members)
        groups.append(
            {
                "value": value,
                "candidate_count": len(members),
                "strict_success_count": int(success_count),
                "strict_success_rate": success_count / len(members),
                "initial_mean_atom_cosine": numeric_summary(
                    [
                        float(run["initialization_audit"]["initial_mean_atom_cosine"])
                        for run in members
                    ]
                ),
            }
        )
    return groups


def pool_diagnostics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    strict = [run for run in runs if is_strict_success(run)]
    non_strict = [run for run in runs if not is_strict_success(run)]
    train_errors = [float(run["train_reconstruction_relative"]) for run in runs]
    atom_cosines = [float(run["mean_atom_cosine"]) for run in runs]
    support_f1 = [float(run["support_f1"]) for run in runs]
    initial_cosines = [
        float(run["initialization_audit"]["initial_mean_atom_cosine"])
        for run in runs
    ]
    global_selected = min(
        runs,
        key=lambda run: (
            float(run["train_reconstruction_relative"]),
            int(run["learner_seed"]),
        ),
    )

    best_strict = (
        min(strict, key=lambda run: float(run["train_reconstruction_relative"]))
        if strict
        else None
    )
    best_non_strict = (
        min(non_strict, key=lambda run: float(run["train_reconstruction_relative"]))
        if non_strict
        else None
    )
    margin = None
    if best_strict is not None and best_non_strict is not None:
        margin = float(
            best_non_strict["train_reconstruction_relative"]
            - best_strict["train_reconstruction_relative"]
        )

    return {
        "candidate_count": len(runs),
        "strict_success_count": len(strict),
        "strict_success_rate": len(strict) / len(runs),
        "global_train_error_selector": run_identifier(global_selected),
        "global_selector_is_strict_success": is_strict_success(global_selected),
        "best_strict_success_by_train_error": (
            run_identifier(best_strict) if best_strict is not None else None
        ),
        "best_non_strict_by_train_error": (
            run_identifier(best_non_strict) if best_non_strict is not None else None
        ),
        "best_non_strict_minus_best_strict_train_error": margin,
        "strict_train_error": numeric_summary(
            [float(run["train_reconstruction_relative"]) for run in strict]
        ),
        "non_strict_train_error": numeric_summary(
            [float(run["train_reconstruction_relative"]) for run in non_strict]
        ),
        "initialization_groups": {
            "all_column_atom_coverage_count": grouped_initialization_success(
                runs, "all_column_atom_coverage_count"
            ),
            "covers_all_atoms": grouped_initialization_success(
                runs, "covers_all_atoms"
            ),
            "distinct_singleton_atom_count": grouped_initialization_success(
                runs, "distinct_singleton_atom_count"
            ),
        },
        "correlations": {
            "train_error_vs_final_atom_cosine_pearson": pearson_correlation(
                train_errors, atom_cosines
            ),
            "train_error_vs_final_atom_cosine_spearman": spearman_correlation(
                train_errors, atom_cosines
            ),
            "train_error_vs_support_f1_pearson": pearson_correlation(
                train_errors, support_f1
            ),
            "train_error_vs_support_f1_spearman": spearman_correlation(
                train_errors, support_f1
            ),
            "initial_atom_cosine_vs_final_atom_cosine_pearson": pearson_correlation(
                initial_cosines, atom_cosines
            ),
            "initial_atom_cosine_vs_final_atom_cosine_spearman": spearman_correlation(
                initial_cosines, atom_cosines
            ),
        },
    }


def classify_r5(result: dict[str, Any]) -> dict[str, Any]:
    contains = float(result["group_contains_strict_success_probability"])
    selected = float(result["selector_selects_strict_success_probability"])
    efficiency_value = result["selection_efficiency"]
    efficiency = float(efficiency_value) if efficiency_value is not None else 0.0
    if contains >= 0.90 and selected >= 0.90 and efficiency >= 0.90:
        label = "A_MANAGEABLE_LOCAL_OPTIMUM"
    elif contains >= 0.90 and (selected < 0.80 or efficiency < 0.80):
        label = "B_SELECTION_OBJECTIVE_OR_IDENTIFIABILITY"
    elif contains < 0.90:
        label = "C_SUCCESS_BASIN_TOO_RARE"
    else:
        label = "INCONCLUSIVE"
    return {
        "primary_budget": 5,
        "classification": label,
        "checks": {
            "contains_success_ge_0.90": contains >= 0.90,
            "selected_success_ge_0.90": selected >= 0.90,
            "selection_efficiency_ge_0.90": efficiency >= 0.90,
            "selected_success_lt_0.80": selected < 0.80,
            "selection_efficiency_lt_0.80": efficiency < 0.80,
        },
    }


def format_optional(value: float | None, digits: int = 4) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def make_report(payload: dict[str, Any]) -> str:
    config = payload["config"]
    diagnostics = payload["pool_diagnostics"]
    simulations = payload["multi_start_simulation"]
    decision = payload["decision"]
    correlations = diagnostics["correlations"]
    initialization_groups = diagnostics["initialization_groups"]
    global_selected = diagnostics["global_train_error_selector"]
    best_strict = diagnostics["best_strict_success_by_train_error"]
    best_non_strict = diagnostics["best_non_strict_by_train_error"]

    lines = [
        "# KSVD E1-T2 初始化与多启动审计结果",
        "",
        "> 日期：2026-07-31  ",
        "> 协议：`KSVD_E1_T2_INITIALIZATION_AUDIT_PROTOCOL_20260731.md`  ",
        "> 数字源：`e1_t2_initialization_audit_20260731.json`",
        "",
        "## 1. 冻结配置",
        "",
        f"- data seed：`{config['data_seed']}`",
        f"- learner seeds：`{config['learner_seeds'][0]}..{config['learner_seeds'][-1]}`（{len(config['learner_seeds'])} 个）",
        f"- train/test patches：{config['n_train']}/{config['n_test']}",
        f"- K/T/iterations：4/2/{config['n_iter']}",
        f"- restart budgets：`{config['restart_budgets']}`；每个 budget {config['simulation_trials']} trials",
        f"- strict success：atom cosine >= {STRICT_ATOM_COSINE} 且 test reconstruction <= {STRICT_TEST_RECONSTRUCTION}",
        "- 多启动只按最低 train reconstruction 选模型；test/atom/support 指标均不参与选择。",
        "",
        "## 2. 50 个初始化候选",
        "",
        f"- strict success：**{diagnostics['strict_success_count']}/{diagnostics['candidate_count']}**（{diagnostics['strict_success_rate']:.1%}）",
        f"- 全局最低 train error seed：**{global_selected['learner_seed']}**；strict success={'是' if global_selected['strict_success'] else '否'}；train={global_selected['train_reconstruction_relative']:.8f}；test={global_selected['test_reconstruction_relative']:.8f}；atom cosine={global_selected['mean_atom_cosine']:.6f}。",
    ]
    if best_strict is not None:
        lines.append(
            f"- 最低 train error 的正确候选：seed {best_strict['learner_seed']}，train={best_strict['train_reconstruction_relative']:.8f}。"
        )
    if best_non_strict is not None:
        lines.append(
            f"- 最低 train error 的错误候选：seed {best_non_strict['learner_seed']}，train={best_non_strict['train_reconstruction_relative']:.8f}，atom cosine={best_non_strict['mean_atom_cosine']:.6f}。"
        )
    margin = diagnostics["best_non_strict_minus_best_strict_train_error"]
    if margin is not None:
        lines.append(
            f"- `best non-strict train error - best strict train error` = {margin:.8f}；正值表示全局上训练目标更偏好正确候选。"
        )
    coverage_groups = {
        bool(group["value"]): group
        for group in initialization_groups["covers_all_atoms"]
    }
    lines.extend(
        [
            "",
            "### 初始化覆盖的直接诊断",
            "",
            "| 初始 4 列是否联合覆盖全部真实 atoms | 候选数 | strict success | success rate | 初始 atom cosine |",
            "|:---:|---:|---:|---:|---:|",
        ]
    )
    for covers_all in (False, True):
        group = coverage_groups.get(covers_all)
        if group is None:
            continue
        lines.append(
            f"| {'是' if covers_all else '否'} | {group['candidate_count']} | {group['strict_success_count']} | {group['strict_success_rate']:.1%} | {group['initial_mean_atom_cosine']['mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "该分组只做初始化机制诊断，不参与 selector，也不改变预注册 A/B/C 判断。",
            "",
            "## 3. 多启动模拟",
            "",
            "| R | group 含正确解 | 精确 containment | train selector 选中正确解 | selection efficiency | selected atom cosine | selected support F1 | atom-cosine regret |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in simulations:
        lines.append(
            "| {budget} | {contains:.4f} | {exact:.4f} | {selected:.4f} | {efficiency} | {atom:.4f} | {support:.4f} | {regret:.4f} |".format(
                budget=result["budget"],
                contains=result["group_contains_strict_success_probability"],
                exact=result["exact_group_contains_strict_success_probability"],
                selected=result["selector_selects_strict_success_probability"],
                efficiency=format_optional(result["selection_efficiency"]),
                atom=result["selected_mean_atom_cosine"]["mean"],
                support=result["selected_support_f1"]["mean"],
                regret=result["atom_cosine_regret"]["mean"],
            )
        )
    lines.extend(
        [
            "",
            "## 4. 训练目标与语义指标",
            "",
            "| 关系 | Pearson | Spearman |",
            "|---|---:|---:|",
            f"| train error vs final atom cosine | {format_optional(correlations['train_error_vs_final_atom_cosine_pearson'])} | {format_optional(correlations['train_error_vs_final_atom_cosine_spearman'])} |",
            f"| train error vs support F1 | {format_optional(correlations['train_error_vs_support_f1_pearson'])} | {format_optional(correlations['train_error_vs_support_f1_spearman'])} |",
            f"| initial atom cosine vs final atom cosine | {format_optional(correlations['initial_atom_cosine_vs_final_atom_cosine_pearson'])} | {format_optional(correlations['initial_atom_cosine_vs_final_atom_cosine_spearman'])} |",
            "",
            "train error 越低越好，而 atom/support 越高越好；因此前两行负相关表示目标与语义恢复方向一致。",
            "",
            "## 5. 预注册判断",
            "",
            f"- 主判断 budget：R={decision['primary_budget']}。",
            f"- 分类：**{decision['classification']}**。",
        ]
    )
    if decision["classification"] == "A_MANAGEABLE_LOCAL_OPTIMUM":
        lines.append(
            "- 解释：正确解在多启动候选中足够常见，且最低训练重构能够可靠选中；E1-T2 的主要问题是可管理的初始化局部最优。"
        )
    elif decision["classification"] == "B_SELECTION_OBJECTIVE_OR_IDENTIFIABILITY":
        lines.append(
            "- 解释：group 中经常已有正确解，但训练重构 selector 仍常选错；仅增加 restart 不足以解决问题。"
        )
    elif decision["classification"] == "C_SUCCESS_BASIN_TOO_RARE":
        lines.append(
            "- 解释：R=5 时正确 basin 仍不够常见；下一步应先研究初始化策略。"
        )
    else:
        lines.append("- 解释：结果落在预注册灰区，需要扩大候选池后再判断。")
    lines.extend(
        [
            "",
            "## 6. 判定边界与下一步",
            "",
            "- 本结果只针对固定 E1-T2 synthetic setting，不外推到采样、置换、图级 readout 或真实数据。",
            "- 下一步必须根据上面的 A/B/C 分支行动，不增加 CIN/MolHIV、随机游走或分类器复杂度。",
            "",
            "## 7. 每个 seed 明细",
            "",
            "| seed | init atom cosine | init singleton cols | init singleton atom coverage | train recon | test recon | atom cosine | support F1 | strict |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for run in payload["candidate_runs"]:
        init = run["initialization_audit"]
        lines.append(
            f"| {run['learner_seed']} | {init['initial_mean_atom_cosine']:.4f} | {init['singleton_column_count']} | {init['distinct_singleton_atom_count']} | {run['train_reconstruction_relative']:.6f} | {run['test_reconstruction_relative']:.6f} | {run['mean_atom_cosine']:.4f} | {run['support_f1']:.4f} | {'Y' if run['strict_success'] else 'N'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=parse_int_list,
        default=parse_int_list(",".join(str(value) for value in range(50))),
    )
    parser.add_argument("--data-seed", type=int, default=20260731)
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=300)
    parser.add_argument("--n-iter", type=int, default=25)
    parser.add_argument(
        "--restart-budgets", type=parse_int_list, default=parse_int_list("1,2,3,5,10")
    )
    parser.add_argument("--simulation-trials", type=int, default=5000)
    parser.add_argument("--audit-seed", type=int, default=20260731)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    if args.simulation_trials <= 0:
        parser.error("simulation-trials must be positive")
    if any(budget < 1 or budget > len(args.seeds) for budget in args.restart_budgets):
        parser.error("each restart budget must lie in [1, number of seeds]")
    if 5 not in args.restart_budgets:
        parser.error("the frozen protocol requires restart budget 5")

    dataset = make_e1_dataset(
        seed=args.data_seed,
        max_sparsity=2,
        n_train=args.n_train,
        n_test=args.n_test,
    )
    runs: list[dict[str, Any]] = []
    for position, learner_seed in enumerate(args.seeds, start=1):
        _dictionary, metrics = fit_and_evaluate(
            dataset,
            learner_seed=learner_seed,
            T=2,
            n_iter=args.n_iter,
        )
        metrics["initialization_audit"] = reconstruct_initialization(
            dataset, learner_seed
        )
        metrics["strict_success"] = is_strict_success(metrics)
        runs.append(metrics)
        print(
            f"[{position:02d}/{len(args.seeds):02d}] seed={learner_seed:>3d} "
            f"train={metrics['train_reconstruction_relative']:.6f} "
            f"atom={metrics['mean_atom_cosine']:.4f} "
            f"strict={'Y' if metrics['strict_success'] else 'N'}",
            flush=True,
        )

    diagnostics = pool_diagnostics(runs)
    rng = np.random.default_rng(args.audit_seed)
    simulations = [
        simulate_restart_budget(
            runs,
            budget=budget,
            n_trials=args.simulation_trials,
            rng=rng,
        )
        for budget in args.restart_budgets
    ]
    r5 = next(result for result in simulations if result["budget"] == 5)
    payload = {
        "protocol": "ksvd-e1-t2-initialization-audit-v0-20260731",
        "config": {
            "data_seed": args.data_seed,
            "learner_seeds": args.seeds,
            "n_train": args.n_train,
            "n_test": args.n_test,
            "n_atoms": 4,
            "sparsity": 2,
            "n_iter": args.n_iter,
            "restart_budgets": args.restart_budgets,
            "simulation_trials": args.simulation_trials,
            "audit_seed": args.audit_seed,
            "selector": "minimum_train_reconstruction_then_lower_learner_seed",
            "strict_atom_cosine_threshold": STRICT_ATOM_COSINE,
            "strict_test_reconstruction_threshold": STRICT_TEST_RECONSTRUCTION,
        },
        "dataset_metadata": dataset.metadata,
        "candidate_runs": runs,
        "pool_diagnostics": diagnostics,
        "multi_start_simulation": simulations,
        "decision": classify_r5(r5),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    args.report.write_text(make_report(payload), encoding="utf-8")
    print(f"Decision: {payload['decision']['classification']}")
    print(f"JSON: {args.json}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
