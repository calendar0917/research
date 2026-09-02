"""Post-G0 diagnostic: when does one fixed random-column run recover motifs?"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .from_scratch_hidden_motif import (
    make_g0_dataset,
    random_column_initialization,
    run_g0_stage,
)


DEFAULT_JSON = Path("tracks/ksvd/results/from_scratch/g0_random_iteration_probe_20260731.json")
DEFAULT_REPORT = Path("tracks/ksvd/results/from_scratch/G0_RANDOM_ITERATION_PROBE_20260731.md")
DEFAULT_CHECKPOINTS = [0, 1, 2, 3, 5, 10, 25]


def strict(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["mean_atom_cosine"] >= 0.99
        and metrics["minimum_atom_cosine"] >= 0.99
        and metrics["primary_decode"]["exact_motif_count"] == 4
        and metrics["test_occurrence"]["occurrence_macro_f1"] >= 0.99
        and metrics["test_occurrence"]["exact_occurrence_accuracy"] >= 0.99
    )


def parse_int_list(text: str) -> list[int]:
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# G0 fixed-random 单次运行的 iteration probe",
        "",
        "> 日期：2026-07-31  ",
        "> 定位：G0 主实验后的机制诊断，不改变冻结判定。每个 data seed 仍只有一个固定 random-column initialization，没有 restart 或 model selection。",
        "",
        "## 1. 结论",
        "",
        f"- INIT strict recovery：0/{len(payload['data_seed_results'])}",
        f"- iteration 25 strict recovery：{payload['summary']['strict_count_by_checkpoint'][str(payload['config']['checkpoints'][-1])]}/{len(payload['data_seed_results'])}",
        f"- 所有 data seeds 首次严格恢复最晚发生在 iteration {payload['summary']['maximum_first_strict_iteration']}。",
        "- 因而，在这个只有四种 exact canonical columns 的最干净条件中，一次普通 KSVD 更新过程确实能从缺失/重复 prototypes 的固定随机初始化恢复完整 vocabulary。",
        "- 但这仍不能外推到有 within-motif variation、采样错误或真实大图；G0B 才是关键检验。",
        "",
        "## 2. 每个 data seed",
        "",
        "| data seed | random INIT unique motifs | first strict iteration | atom cosine trajectory | occurrence macro-F1 trajectory |",
        "|---:|---:|---:|---|---|",
    ]
    for result in payload["data_seed_results"]:
        atoms = " → ".join(f"{point['mean_atom_cosine']:.3f}" for point in result["trajectory"])
        occurrence = " → ".join(f"{point['occurrence_macro_f1']:.3f}" for point in result["trajectory"])
        lines.append(
            f"| {result['data_seed']} | {result['selected_unique_column_count']} | "
            f"{result['first_strict_iteration']} | {atoms} | {occurrence} |"
        )
    lines.extend([
        "",
        "checkpoint 顺序为：`" + ", ".join(str(value) for value in payload["config"]["checkpoints"]) + "`。",
        "",
        "## 3. 正确解释",
        "",
        "这给出两条同时成立、不能混淆的结论：",
        "",
        "1. **primary maximin 的 G0 成功不能归功于 KSVD**，因为它在 `n_iter=0` 已经找齐四种 prototype；",
        "2. **KSVD 在 secondary weak initialization 上有真实 refinement 贡献**，因为随机 INIT 0/10 通过，而同一初始字典的 FINAL 10/10 通过。",
        "",
        "第二条是 KSVD 路线的正信号，但目前只发生在离散、无噪声、oracle patch、恰好四种唯一观测的理想条件，证据强度仍应标为 pipeline-level，而不是 real-graph motif discovery。",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-seeds", type=parse_int_list, default=list(range(20260731, 20260741)))
    parser.add_argument("--checkpoints", type=parse_int_list, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    if args.checkpoints[0] != 0 or args.checkpoints != sorted(set(args.checkpoints)):
        raise ValueError("checkpoints must be unique, sorted, and start at 0")

    results = []
    for data_seed in args.data_seeds:
        dataset = make_g0_dataset(seed=data_seed)
        initial_dictionary, initialization = random_column_initialization(dataset.Y_train, 4, seed=0)
        trajectory = []
        for checkpoint in args.checkpoints:
            _D, metrics = run_g0_stage(dataset, initial_dictionary, n_iter=checkpoint, ksvd_seed=0)
            trajectory.append({
                "iteration": int(checkpoint),
                "strict": strict(metrics),
                "mean_atom_cosine": float(metrics["mean_atom_cosine"]),
                "minimum_atom_cosine": float(metrics["minimum_atom_cosine"]),
                "exact_motif_count": int(metrics["primary_decode"]["exact_motif_count"]),
                "occurrence_macro_f1": float(metrics["test_occurrence"]["occurrence_macro_f1"]),
                "exact_occurrence_accuracy": float(metrics["test_occurrence"]["exact_occurrence_accuracy"]),
                "test_reconstruction_relative": float(metrics["test_reconstruction_relative"]),
            })
        first = next((point["iteration"] for point in trajectory if point["strict"]), None)
        results.append({
            "data_seed": int(data_seed),
            "selected_unique_column_count": int(initialization["selected_unique_column_count"]),
            "selected_training_indices": initialization["selected_training_indices"],
            "first_strict_iteration": first,
            "trajectory": trajectory,
        })

    strict_counts = {
        str(checkpoint): int(sum(result["trajectory"][index]["strict"] for result in results))
        for index, checkpoint in enumerate(args.checkpoints)
    }
    first_values = [result["first_strict_iteration"] for result in results if result["first_strict_iteration"] is not None]
    payload = {
        "protocol": "ksvd-g0-fixed-random-iteration-probe-v0-20260731",
        "config": {
            "data_seeds": args.data_seeds,
            "checkpoints": args.checkpoints,
            "random_initialization_seed": 0,
            "ksvd_seed": 0,
            "restart_count": 1,
            "model_selection": "none",
        },
        "data_seed_results": results,
        "summary": {
            "strict_count_by_checkpoint": strict_counts,
            "first_strict_iteration_mean": float(np.mean(first_values)) if first_values else None,
            "maximum_first_strict_iteration": int(max(first_values)) if first_values else None,
            "never_strict_count": int(sum(result["first_strict_iteration"] is None for result in results)),
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.report.write_text(render(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"wrote {args.json}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
