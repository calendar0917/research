"""Run frozen G0 hidden-motif discovery across independent graph datasets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .from_scratch_hidden_motif import (
    MOTIF_NAMES,
    deterministic_maximin_initialization,
    make_g0_dataset,
    oracle_control,
    random_column_initialization,
    run_g0_stage,
)
from .from_scratch_recovery import pairwise_dictionary_stability


DEFAULT_JSON = Path("tracks/ksvd/results/from_scratch/g0_hidden_motif_20260731.json")
DEFAULT_REPORT = Path("tracks/ksvd/results/from_scratch/G0_HIDDEN_MOTIF_20260731.md")
CONDITIONS = ("deterministic_maximin", "fixed_random_columns")
STAGES = ("init", "final")


def numeric_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "minimum": float("nan"), "maximum": float("nan")}
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def compact_stage_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    primary = metrics["primary_decode"]
    occurrence = metrics["test_occurrence"]
    binary = metrics["test_binary_reconstruction"]
    return {
        "mean_atom_cosine": float(metrics["mean_atom_cosine"]),
        "minimum_atom_cosine": float(metrics["minimum_atom_cosine"]),
        "matched_cosines": metrics["matched_cosines"],
        "train_reconstruction_relative": float(metrics["train_reconstruction_relative"]),
        "test_reconstruction_relative": float(metrics["test_reconstruction_relative"]),
        "primary_decode_mean_edge_f1": float(primary["mean_edge_f1"]),
        "primary_decode_minimum_edge_f1": float(primary["minimum_edge_f1"]),
        "primary_decode_exact_motif_count": int(primary["exact_motif_count"]),
        "primary_decode_connected_count": int(primary["connected_decode_count"]),
        "decode_thresholds": metrics["decode_thresholds"],
        "test_support_precision": float(occurrence["support_precision"]),
        "test_support_recall": float(occurrence["support_recall"]),
        "test_support_f1": float(occurrence["support_f1"]),
        "test_occurrence_macro_precision": float(occurrence["occurrence_macro_precision"]),
        "test_occurrence_macro_recall": float(occurrence["occurrence_macro_recall"]),
        "test_occurrence_macro_f1": float(occurrence["occurrence_macro_f1"]),
        "exact_occurrence_accuracy": float(occurrence["exact_occurrence_accuracy"]),
        "per_motif_occurrence": occurrence["per_motif_occurrence"],
        "test_edge_f1": float(binary["edge_f1"]),
        "test_exact_patch_recovery": float(binary["exact_patch_recovery"]),
        "train_atoms_used": int(metrics["train_atoms_used"]),
        "train_mean_nnz": float(metrics["train_mean_nnz"]),
        "reconstruction_curve": metrics["reconstruction_curve"],
    }


def aggregate_results(
    results: list[dict[str, Any]],
    dictionaries: dict[str, dict[str, list[np.ndarray]]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "data_seed_count": len(results),
        "oracle_pass_count": sum(bool(result["oracle_control"]["passed"]) for result in results),
        "connected_data_count": sum(
            bool(result["dataset"]["all_train_graphs_connected"])
            and bool(result["dataset"]["all_test_graphs_connected"])
            for result in results
        ),
        "four_unique_patch_data_count": sum(
            int(result["dataset"]["train_unique_canonical_patch_count"]) == 4
            and int(result["dataset"]["test_unique_canonical_patch_count"]) == 4
            for result in results
        ),
        "full_motif_coverage_count": sum(
            min(result["dataset"]["train_motif_counts"]) > 0
            and min(result["dataset"]["test_motif_counts"]) > 0
            for result in results
        ),
        "conditions": {},
    }
    metric_keys = (
        "mean_atom_cosine",
        "minimum_atom_cosine",
        "train_reconstruction_relative",
        "test_reconstruction_relative",
        "primary_decode_mean_edge_f1",
        "primary_decode_minimum_edge_f1",
        "primary_decode_exact_motif_count",
        "test_support_f1",
        "test_occurrence_macro_f1",
        "exact_occurrence_accuracy",
        "test_edge_f1",
        "test_exact_patch_recovery",
    )
    for condition in CONDITIONS:
        condition_summary: dict[str, Any] = {}
        for stage in STAGES:
            stage_records = [result["conditions"][condition][stage] for result in results]
            condition_summary[stage] = {
                "metrics": {
                    key: numeric_summary([float(record[key]) for record in stage_records])
                    for key in metric_keys
                },
                "strict_discovery_count": sum(stage_passes(record) for record in stage_records),
                "dictionary_stability": pairwise_dictionary_stability(dictionaries[condition][stage]),
            }
        condition_summary["initial_selected_unique_column_count"] = numeric_summary(
            [float(result["conditions"][condition]["initialization"]["selected_unique_column_count"]) for result in results]
        )
        summary["conditions"][condition] = condition_summary
    return summary


def stage_passes(record: dict[str, Any]) -> bool:
    return bool(
        float(record["mean_atom_cosine"]) >= 0.99
        and float(record["minimum_atom_cosine"]) >= 0.99
        and int(record["primary_decode_exact_motif_count"]) == len(MOTIF_NAMES)
        and float(record["test_occurrence_macro_f1"]) >= 0.99
        and float(record["exact_occurrence_accuracy"]) >= 0.99
    )


def classify(summary: dict[str, Any]) -> dict[str, Any]:
    count = int(summary["data_seed_count"])
    data_ok = bool(
        summary["oracle_pass_count"] == count
        and summary["connected_data_count"] == count
        and summary["four_unique_patch_data_count"] == count
        and summary["full_motif_coverage_count"] == count
    )
    primary = summary["conditions"]["deterministic_maximin"]
    final_metrics = primary["final"]["metrics"]
    init_metrics = primary["init"]["metrics"]
    final_ok = bool(
        primary["final"]["strict_discovery_count"] == count
        and final_metrics["mean_atom_cosine"]["minimum"] >= 0.99
        and final_metrics["primary_decode_exact_motif_count"]["minimum"] >= 4
        and final_metrics["test_occurrence_macro_f1"]["minimum"] >= 0.99
        and final_metrics["exact_occurrence_accuracy"]["minimum"] >= 0.99
        and primary["final"]["dictionary_stability"]["pairwise_matched_atom_cosine_minimum"] >= 0.99
    )
    init_ok = bool(
        primary["init"]["strict_discovery_count"] == count
        and init_metrics["mean_atom_cosine"]["minimum"] >= 0.99
        and init_metrics["primary_decode_exact_motif_count"]["minimum"] >= 4
        and init_metrics["test_occurrence_macro_f1"]["minimum"] >= 0.99
        and init_metrics["exact_occurrence_accuracy"]["minimum"] >= 0.99
        and primary["init"]["dictionary_stability"]["pairwise_matched_atom_cosine_minimum"] >= 0.99
    )
    if not data_ok:
        label = "FAIL_DATA_OR_EVALUATOR"
    elif not final_ok:
        label = "FAIL_SINGLE_RUN_DISCOVERY"
    elif init_ok:
        label = "PASS_INITIALIZER_DISCOVERY_ONLY"
    else:
        label = "PASS_KSVD_REFINEMENT"
    random_condition = summary["conditions"]["fixed_random_columns"]
    return {
        "classification": label,
        "passed_data_controls": data_ok,
        "primary_init_passed": init_ok,
        "primary_final_passed": final_ok,
        "random_init_strict_count": int(random_condition["init"]["strict_discovery_count"]),
        "random_final_strict_count": int(random_condition["final"]["strict_discovery_count"]),
        "interpretation": {
            "PASS_INITIALIZER_DISCOVERY_ONLY": "The exact four-prototype vocabulary is already recovered by deterministic maximin at n_iter=0; G0 validates the pipeline but does not establish a KSVD discovery contribution.",
            "PASS_KSVD_REFINEMENT": "The primary single initialization fails the frozen discovery gate at n_iter=0 and passes after KSVD updates, supporting a KSVD refinement contribution.",
            "FAIL_SINGLE_RUN_DISCOVERY": "The deployable single-run primary route does not reliably recover hidden motif atoms and occurrences.",
            "FAIL_DATA_OR_EVALUATOR": "Graph generation, canonicalization, motif coverage, connectivity, or oracle evaluation failed.",
        }[label],
    }


def render_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    decision = payload["decision"]
    lines = [
        "# G0 Hidden Motif Discovery 结果",
        "",
        "> 日期：2026-07-31  ",
        "> 数字唯一源：`g0_hidden_motif_20260731.json`",
        "",
        "## 1. 判定",
        "",
        f"**{decision['classification']}**",
        "",
        decision["interpretation"],
        "",
        "本轮每个 data seed 对每种初始化只运行一次；没有 restart selection。",
        "",
        "## 2. 数据与 oracle controls",
        "",
        f"- oracle：{summary['oracle_pass_count']}/{summary['data_seed_count']}",
        f"- 完整图全连通：{summary['connected_data_count']}/{summary['data_seed_count']}",
        f"- train/test 均恰有 4 种 canonical patch：{summary['four_unique_patch_data_count']}/{summary['data_seed_count']}",
        f"- 四种 motif 全覆盖：{summary['full_motif_coverage_count']}/{summary['data_seed_count']}",
        "",
        "## 3. INIT 与 FINAL 对照",
        "",
        "| condition | stage | strict seeds | atom cosine mean/min(seed) | exact motifs mean | occurrence macro-F1 mean/min | exact occurrence mean/min | test recon mean | pairwise stability min |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        for stage in STAGES:
            block = summary["conditions"][condition][stage]
            metrics = block["metrics"]
            stability = block["dictionary_stability"]
            lines.append(
                "| "
                f"{condition} | {stage.upper()} | {block['strict_discovery_count']}/{summary['data_seed_count']} | "
                f"{metrics['mean_atom_cosine']['mean']:.4f}/{metrics['mean_atom_cosine']['minimum']:.4f} | "
                f"{metrics['primary_decode_exact_motif_count']['mean']:.2f} | "
                f"{metrics['test_occurrence_macro_f1']['mean']:.4f}/{metrics['test_occurrence_macro_f1']['minimum']:.4f} | "
                f"{metrics['exact_occurrence_accuracy']['mean']:.4f}/{metrics['exact_occurrence_accuracy']['minimum']:.4f} | "
                f"{metrics['test_reconstruction_relative']['mean']:.6f} | "
                f"{stability['pairwise_matched_atom_cosine_minimum']:.4f} |"
            )
    lines.extend([
        "",
        "## 4. 每个 data seed",
        "",
        "| data seed | train motif counts | maximin unique init | maximin init/final atom | maximin init/final occurrence | random unique init | random init/final atom | random init/final occurrence |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for result in payload["data_seed_results"]:
        det = result["conditions"]["deterministic_maximin"]
        rnd = result["conditions"]["fixed_random_columns"]
        lines.append(
            f"| {result['data_seed']} | {result['dataset']['train_motif_counts']} | "
            f"{det['initialization']['selected_unique_column_count']} | "
            f"{det['init']['mean_atom_cosine']:.3f}/{det['final']['mean_atom_cosine']:.3f} | "
            f"{det['init']['test_occurrence_macro_f1']:.3f}/{det['final']['test_occurrence_macro_f1']:.3f} | "
            f"{rnd['initialization']['selected_unique_column_count']} | "
            f"{rnd['init']['mean_atom_cosine']:.3f}/{rnd['final']['mean_atom_cosine']:.3f} | "
            f"{rnd['init']['test_occurrence_macro_f1']:.3f}/{rnd['final']['test_occurrence_macro_f1']:.3f} |"
        )
    lines.extend([
        "",
        "## 5. 科学解释",
        "",
        "- `INIT` 与 `FINAL` 使用完全相同的初始字典；差异只来自 KSVD iterations。",
        "- deterministic maximin 若在 INIT 已通过，说明四种 exact canonical prototypes 被初始化器直接枚举出来，不能宣称 KSVD 自动提炼了 motif。",
        "- fixed random-column 是单次弱初始化诊断，不参与选择，也不用于救 primary 结果。",
        "- 即使 G0 通过，oracle cell extractor 仍是强条件；随机游走覆盖与 patch 间关联尚未验证。",
        "",
        "## 6. 下一步",
        "",
        "若判定为 `PASS_INITIALIZER_DISCOVERY_ONLY`，进入 G0B：加入 within-motif variation / nuisance edges，使训练集不再只有四种唯一 canonical columns，再检查 INIT→FINAL 是否出现真正的 KSVD refinement。",
        "",
    ])
    return "\n".join(lines)


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-seeds", type=parse_int_list, default=list(range(20260731, 20260741)))
    parser.add_argument("--n-train-graphs", type=int, default=100)
    parser.add_argument("--n-test-graphs", type=int, default=30)
    parser.add_argument("--cells-per-graph", type=int, default=12)
    parser.add_argument("--n-iter", type=int, default=25)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    results: list[dict[str, Any]] = []
    dictionaries: dict[str, dict[str, list[np.ndarray]]] = {
        condition: {stage: [] for stage in STAGES} for condition in CONDITIONS
    }
    for position, data_seed in enumerate(args.data_seeds, start=1):
        dataset = make_g0_dataset(
            seed=data_seed,
            n_train_graphs=args.n_train_graphs,
            n_test_graphs=args.n_test_graphs,
            cells_per_graph=args.cells_per_graph,
        )
        oracle = oracle_control(dataset)
        initializers = {
            "deterministic_maximin": deterministic_maximin_initialization(dataset.Y_train, 4),
            "fixed_random_columns": random_column_initialization(dataset.Y_train, 4, seed=0),
        }
        condition_results: dict[str, Any] = {}
        for condition, (initial_dictionary, initialization_info) in initializers.items():
            D_init, init_metrics = run_g0_stage(dataset, initial_dictionary, n_iter=0, ksvd_seed=0)
            D_final, final_metrics = run_g0_stage(dataset, initial_dictionary, n_iter=args.n_iter, ksvd_seed=0)
            dictionaries[condition]["init"].append(D_init)
            dictionaries[condition]["final"].append(D_final)
            condition_results[condition] = {
                "initialization": initialization_info,
                "init": compact_stage_metrics(init_metrics),
                "final": compact_stage_metrics(final_metrics),
            }
        result = {
            "data_seed": int(data_seed),
            "dataset": dataset.metadata,
            "oracle_control": {
                "passed": bool(oracle["passed"]),
                "test_reconstruction_relative": float(oracle["test_reconstruction_relative"]),
                "minimum_atom_cosine": float(oracle["minimum_atom_cosine"]),
                "exact_motif_count": int(oracle["primary_decode"]["exact_motif_count"]),
                "occurrence_macro_f1": float(oracle["test_occurrence"]["occurrence_macro_f1"]),
                "exact_occurrence_accuracy": float(oracle["test_occurrence"]["exact_occurrence_accuracy"]),
                "exact_patch_recovery": float(oracle["test_binary_reconstruction"]["exact_patch_recovery"]),
            },
            "conditions": condition_results,
        }
        results.append(result)
        det = condition_results["deterministic_maximin"]
        rnd = condition_results["fixed_random_columns"]
        print(
            f"[{position:02d}/{len(args.data_seeds):02d}] seed={data_seed} "
            f"oracle={'Y' if oracle['passed'] else 'N'} "
            f"det_unique={det['initialization']['selected_unique_column_count']} "
            f"det_atom={det['init']['mean_atom_cosine']:.3f}->{det['final']['mean_atom_cosine']:.3f} "
            f"rnd_unique={rnd['initialization']['selected_unique_column_count']} "
            f"rnd_atom={rnd['init']['mean_atom_cosine']:.3f}->{rnd['final']['mean_atom_cosine']:.3f}",
            flush=True,
        )

    summary = aggregate_results(results, dictionaries)
    decision = classify(summary)
    payload = {
        "protocol": "ksvd-g0-hidden-motif-discovery-v0-20260731",
        "config": {
            "data_seeds": args.data_seeds,
            "n_train_graphs": args.n_train_graphs,
            "n_test_graphs": args.n_test_graphs,
            "cells_per_graph": args.cells_per_graph,
            "nodes_per_graph": 6 * args.cells_per_graph,
            "n_atoms": 4,
            "sparsity": 1,
            "n_iter_init": 0,
            "n_iter_final": args.n_iter,
            "ksvd_seed": 0,
            "random_initialization_seed": 0,
            "restart_count_per_condition": 1,
            "model_selection": "none",
        },
        "data_seed_results": results,
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
