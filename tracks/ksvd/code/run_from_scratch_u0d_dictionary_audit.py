#!/usr/bin/env python3
"""Run the registered U0-D INIT-versus-FINAL KSVD optimization/health audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .from_scratch_unplanted_dictionary import (
    cross_replicate_dictionary_similarity,
    run_u0d_dictionary_audit,
)
from .from_scratch_unplanted_signal import dataset_summary, generate_u0p_dataset


DEFAULT_SEEDS = (731101, 731102, 731103, 731104, 731105)
DEFAULT_JSON = Path(
    "tracks/ksvd/results/from_scratch/u0d_dictionary_audit_20260731.json"
)
DEFAULT_REPORT = Path(
    "tracks/ksvd/results/from_scratch/U0D_DICTIONARY_AUDIT_20260731.md"
)


def _seed_result(
    master_seed: int,
    *,
    train_per_class: int,
    validation_per_class: int,
    test_per_class: int,
    patches_per_graph: int,
    n_atoms: int,
    sparsity: int,
    minimum_sparsity: int,
    n_iterations: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    root_sequence = np.random.SeedSequence(int(master_seed))
    data_sequence, _shuffle_sequence = root_sequence.spawn(2)
    dataset = generate_u0p_dataset(
        data_sequence,
        train_per_class=train_per_class,
        validation_per_class=validation_per_class,
        test_per_class=test_per_class,
        patches_per_graph=patches_per_graph,
    )
    audit, initial, final = run_u0d_dictionary_audit(
        dataset,
        n_atoms=n_atoms,
        sparsity=sparsity,
        minimum_sparsity=minimum_sparsity,
        n_iterations=n_iterations,
    )
    audit["master_seed"] = int(master_seed)
    audit["dataset"] = dataset_summary(dataset)
    audit["dictionaries"] = {
        "init": initial.tolist(),
        "final": final.tolist(),
    }
    return audit, initial, final


def classify(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    reductions = np.asarray(
        [seed["health"]["test_relative_reconstruction_reduction"] for seed in seed_results],
        dtype=np.float64,
    )
    reduction_gate = bool(
        np.count_nonzero(reductions >= 0.10) >= 4 and np.mean(reductions) >= 0.15
    )
    nondead_passes = int(
        sum(seed["health"]["passes_nondead_atoms"] for seed in seed_results)
    )
    effective_passes = int(
        sum(seed["health"]["passes_effective_atom_count"] for seed in seed_results)
    )
    coherence_passes = int(sum(seed["health"]["passes_coherence"] for seed in seed_results))
    # Dictionary-health thresholds are required in every registered replicate.
    health_gate = bool(
        nondead_passes == len(seed_results)
        and effective_passes == len(seed_results)
        and coherence_passes == len(seed_results)
    )
    overall = bool(reduction_gate and health_gate)
    if overall:
        classification = "PASS_KSVD_OPTIMIZATION"
        next_step = (
            "Proceed to U1A.  Evaluate graph-level INIT and FINAL code readouts on the "
            "same datasets, retaining raw patch, Gaussian, medoid, PCA, simple-stat, and shuffle controls."
        )
    elif not reduction_gate:
        classification = "FAIL_U0D_HELDOUT_RECONSTRUCTION_GAIN"
        next_step = (
            "Stop before downstream attribution.  Ordinary single-start KSVD did not meet the "
            "registered held-out reconstruction improvement gate; do not repair this with restarts."
        )
    else:
        classification = "FAIL_U0D_DICTIONARY_HEALTH"
        next_step = (
            "Stop before downstream attribution.  Reconstruction improved, but the learned "
            "dictionary violated at least one registered collapse/health gate."
        )
    return {
        "classification": classification,
        "passes_u0d": overall,
        "test_relative_reconstruction_reduction": {
            "mean": float(np.mean(reductions)),
            "std": float(np.std(reductions, ddof=0)),
            "minimum": float(np.min(reductions)),
            "maximum": float(np.max(reductions)),
            "replicates_at_least_0_10": int(np.count_nonzero(reductions >= 0.10)),
            "passes_gate": reduction_gate,
        },
        "health_replicate_counts": {
            "nondead_atoms_at_least_6": nondead_passes,
            "effective_atom_count_at_least_4": effective_passes,
            "maximum_coherence_below_0_95": coherence_passes,
            "required": len(seed_results),
            "passes_gate": health_gate,
        },
        "next_step": next_step,
    }


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def render_report(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    lines = [
        "# KSVD U0-D：无人工原子路线的 dictionary optimization / health audit",
        "",
        "> 日期：2026-07-31",
        ">",
        f"> 正式结论：**{decision['classification']}**",
        "",
        "## 1. 本实验回答什么",
        "",
        "U0-P 已确认 walk-order adjacency patches 暴露 LOW/HIGH regime 信号。U0-D 现在只问：",
        "",
        "> 在每个独立数据 replicate 上，只使用一次 deterministic maximin initialization，普通 KSVD 是否比完全相同的 INIT 改善 held-out sparse reconstruction，并保持非坍缩字典？",
        "",
        "本阶段不使用 graph labels 训练字典，不做 atom-to-motif matching，也不做多 restart 或模型挑选。",
        "",
        "## 2. 冻结配置",
        "",
        f"- master seeds：`{', '.join(str(seed) for seed in payload['config']['seeds'])}`；",
        f"- train / validation / test：每类 `{payload['config']['train_per_class']} / {payload['config']['validation_per_class']} / {payload['config']['test_per_class']}` 图；",
        f"- patches per graph：`{payload['config']['patches_per_graph']}`；",
        "- primary signal：15-D walk first-discovery-order induced adjacency upper triangle；",
        "- 只减去 train patch coordinate mean；不做 per-patch normalization；",
        f"- `K={payload['config']['n_atoms']}`, `T={payload['config']['sparsity']}`, `T_min={payload['config']['minimum_sparsity']}`；",
        f"- FINAL updates：`{payload['config']['n_iterations']}`；",
        "- initialization：一次 deterministic maximin，第一列取 centered norm 最大训练 patch，后续最小化与已选 atoms 的最大 absolute cosine；",
        "- INIT 和 FINAL 使用完全相同的 `D_init`；dead-atom internal seed 固定为 0；",
        "- validation/test 不参与初始化、字典训练或超参数选择。",
        "",
        "## 3. held-out reconstruction",
        "",
        "| seed | INIT test rel. err | FINAL test rel. err | relative reduction | FINAL test NMSE |",
        "|---:|---:|---:|---:|---:|",
    ]
    for seed in payload["seeds"]:
        init = seed["stages"]["init"]["test"]
        final = seed["stages"]["final"]["test"]
        lines.append(
            f"| {seed['master_seed']} | {_fmt(init['relative_reconstruction_error'])} | "
            f"{_fmt(final['relative_reconstruction_error'])} | "
            f"{_fmt(seed['health']['test_relative_reconstruction_reduction'])} | "
            f"{_fmt(final['nmse'])} |"
        )
    reduction = decision["test_relative_reconstruction_reduction"]
    lines.extend(
        [
            "",
            f"五次 relative reduction：mean `{_fmt(reduction['mean'])}`，std `{_fmt(reduction['std'])}`，"
            f"minimum `{_fmt(reduction['minimum'])}`；`{reduction['replicates_at_least_0_10']}/5` 达到 10%。",
            "",
            f"Registered reconstruction gate：**{'PASS' if reduction['passes_gate'] else 'FAIL'}**。",
            "",
            "## 4. FINAL test dictionary health",
            "",
            "| seed | mean nnz | nondead atoms | effective atoms | max coherence | max activation share | health |",
            "|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for seed in payload["seeds"]:
        metrics = seed["stages"]["final"]["test"]
        lines.append(
            f"| {seed['master_seed']} | {_fmt(metrics['mean_nonzeros_per_patch'])} | "
            f"{metrics['nondead_atom_count']}/12 | {_fmt(metrics['effective_atom_count'])} | "
            f"{_fmt(metrics['maximum_absolute_offdiagonal_coherence'])} | "
            f"{_fmt(metrics['maximum_activation_share'])} | "
            f"{'PASS' if seed['health']['passes_all_per_replicate_health'] else 'FAIL'} |"
        )
    health = decision["health_replicate_counts"]
    lines.extend(
        [
            "",
            f"Across replicates：nondead gate `{health['nondead_atoms_at_least_6']}/5`，"
            f"effective-count gate `{health['effective_atom_count_at_least_4']}/5`，"
            f"coherence gate `{health['maximum_coherence_below_0_95']}/5`。",
            "",
            f"Registered health gate：**{'PASS' if health['passes_gate'] else 'FAIL'}**。",
            "",
            "## 5. 跨 replicate 描述性稳定性",
            "",
            "这里不设置通过门槛。atom matching 使用 absolute cosine 的 exact maximum-sum assignment；subspace 指标使用 principal-angle cosines。",
            "",
            "| stage | mean pair matched atom cosine | minimum pair mean | mean subspace cosine | minimum pair mean |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for stage in ("init", "final"):
        item = payload["cross_replicate_similarity"][stage]
        lines.append(
            f"| `{stage}` | {_fmt(item['matched_cosine_mean'])} | "
            f"{_fmt(item['matched_cosine_minimum_pair'])} | "
            f"{_fmt(item['subspace_cosine_mean'])} | "
            f"{_fmt(item['subspace_cosine_minimum_pair'])} |"
        )
    lines.extend(
        [
            "",
            "## 6. 结论边界",
            "",
            f"**{decision['classification']}**",
            "",
            decision["next_step"],
            "",
        ]
    )
    if decision["passes_u0d"]:
        lines.extend(
            [
                "该结论只支持：普通单次初始化 KSVD 在当前 walk-induced patch distribution 上学到了更好的 held-out sparse reconstruction basis，且未发生注册定义下的严重坍缩。",
                "",
                "它仍不支持：",
                "",
                "- 每个 atom 都是合法 adjacency 或可命名 motif；",
                "- FINAL codes 比 INIT codes 更适合图级任务；",
                "- KSVD 超过 edge-count histogram、PCA 或 simple graph statistics；",
                "- 该结论可直接外推到真实大图。",
                "",
                "因此下一步必须是 paired U1A：同数据、同初始化、同 classifier protocol 下比较 INIT 与 FINAL，而不是直接展示 FINAL accuracy。",
                "",
            ]
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--train-per-class", type=int, default=150)
    parser.add_argument("--validation-per-class", type=int, default=50)
    parser.add_argument("--test-per-class", type=int, default=100)
    parser.add_argument("--patches-per-graph", type=int, default=24)
    parser.add_argument("--n-atoms", type=int, default=12)
    parser.add_argument("--sparsity", type=int, default=2)
    parser.add_argument("--minimum-sparsity", type=int, default=1)
    parser.add_argument("--n-iterations", type=int, default=25)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_results = []
    initial_dictionaries = []
    final_dictionaries = []
    for seed in args.seeds:
        result, initial, final = _seed_result(
            seed,
            train_per_class=args.train_per_class,
            validation_per_class=args.validation_per_class,
            test_per_class=args.test_per_class,
            patches_per_graph=args.patches_per_graph,
            n_atoms=args.n_atoms,
            sparsity=args.sparsity,
            minimum_sparsity=args.minimum_sparsity,
            n_iterations=args.n_iterations,
        )
        seed_results.append(result)
        initial_dictionaries.append(initial)
        final_dictionaries.append(final)
        init_test = result["stages"]["init"]["test"]["relative_reconstruction_error"]
        final_test = result["stages"]["final"]["test"]["relative_reconstruction_error"]
        health = result["stages"]["final"]["test"]
        print(
            f"seed={seed} recon={init_test:.4f}->{final_test:.4f} "
            f"reduction={result['health']['test_relative_reconstruction_reduction']:.4f} "
            f"nondead={health['nondead_atom_count']} effective={health['effective_atom_count']:.2f} "
            f"coherence={health['maximum_absolute_offdiagonal_coherence']:.4f}",
            flush=True,
        )
    decision = classify(seed_results)
    payload = {
        "protocol": "ksvd-u0d-unplanted-dictionary-audit-v0-20260731",
        "config": {
            "seeds": [int(seed) for seed in args.seeds],
            "train_per_class": int(args.train_per_class),
            "validation_per_class": int(args.validation_per_class),
            "test_per_class": int(args.test_per_class),
            "patches_per_graph": int(args.patches_per_graph),
            "patch_size": 6,
            "representation": "walk_first_discovery_order_adjacency_upper_triangle",
            "n_atoms": int(args.n_atoms),
            "sparsity": int(args.sparsity),
            "minimum_sparsity": int(args.minimum_sparsity),
            "n_iterations": int(args.n_iterations),
            "ksvd_internal_seed": 0,
            "restart_count": 1,
            "label_used_for_dictionary": False,
        },
        "seeds": seed_results,
        "cross_replicate_similarity": {
            "init": cross_replicate_dictionary_similarity(initial_dictionaries),
            "final": cross_replicate_dictionary_similarity(final_dictionaries),
        },
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
