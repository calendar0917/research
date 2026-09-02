#!/usr/bin/env python3
"""Run raw IMDB-BINARY R0-D fold-local INIT-versus-FINAL audit."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .from_scratch_unplanted_dictionary import cross_replicate_dictionary_similarity
from .imdb_walk_dictionary import (
    audit_fold_splits,
    grouped_isomorphism_folds,
    run_imdb_dictionary_fold,
    stratified_graph_folds,
)
from .imdb_walk_substrate import (
    exact_isomorphism_groups,
    extract_walk_patch_graphs,
    load_tu_structure_text,
)


DEFAULT_DATASET = Path("data/TUD/IMDB-BINARY")
DEFAULT_JSON = Path(
    "tracks/ksvd/results/from_scratch/imdb_binary_r0d_dictionary_audit_20260731.json"
)
DEFAULT_REPORT = Path(
    "tracks/ksvd/results/from_scratch/IMDB_BINARY_R0D_DICTIONARY_AUDIT_20260731.md"
)


def summarize_view(folds: list[dict[str, Any]]) -> dict[str, Any]:
    reductions = np.asarray(
        [fold["attribution"]["test_graph_balanced_relative_reduction"] for fold in folds],
        dtype=np.float64,
    )
    positive_count = int(np.count_nonzero(reductions > 0.0))
    patch_reductions = np.asarray(
        [fold["attribution"]["test_patch_weighted_relative_reduction"] for fold in folds],
        dtype=np.float64,
    )
    init_train_graph_errors = np.asarray(
        [fold["stages"]["init"]["train"]["graph_balanced"]["mean_relative_reconstruction_error"] for fold in folds],
        dtype=np.float64,
    )
    init_test_graph_errors = np.asarray(
        [fold["stages"]["init"]["test"]["graph_balanced"]["mean_relative_reconstruction_error"] for fold in folds],
        dtype=np.float64,
    )
    final_train_graph_errors = np.asarray(
        [fold["stages"]["final"]["train"]["graph_balanced"]["mean_relative_reconstruction_error"] for fold in folds],
        dtype=np.float64,
    )
    final_test_graph_errors = np.asarray(
        [fold["stages"]["final"]["test"]["graph_balanced"]["mean_relative_reconstruction_error"] for fold in folds],
        dtype=np.float64,
    )
    nondead = [
        fold["stages"]["final"]["test"]["dictionary_health"]["nondead_atom_count"]
        for fold in folds
    ]
    max_shares = [
        fold["stages"]["final"]["test"]["dictionary_health"]["maximum_activation_share"]
        for fold in folds
    ]
    full_five_fold = len(folds) == 5
    direction_gate = bool(full_five_fold and positive_count >= 4 and np.mean(reductions) >= 0.10)
    health_gate = bool(
        full_five_fold
        and all(value >= 10 for value in nondead)
        and all(value <= 0.60 for value in max_shares)
    )
    return {
        "fold_count": int(len(folds)),
        "graph_balanced_relative_reductions": reductions.tolist(),
        "positive_reduction_fold_count": positive_count,
        "mean_graph_balanced_relative_reduction": float(np.mean(reductions)),
        "std_graph_balanced_relative_reduction": float(np.std(reductions, ddof=0)),
        "minimum_graph_balanced_relative_reduction": float(np.min(reductions)),
        "maximum_graph_balanced_relative_reduction": float(np.max(reductions)),
        "mean_patch_weighted_relative_reduction": float(np.mean(patch_reductions)),
        "mean_init_train_graph_balanced_error": float(np.mean(init_train_graph_errors)),
        "mean_init_test_graph_balanced_error": float(np.mean(init_test_graph_errors)),
        "mean_init_train_to_test_gap": float(np.mean(init_test_graph_errors - init_train_graph_errors)),
        "mean_final_train_graph_balanced_error": float(np.mean(final_train_graph_errors)),
        "mean_final_test_graph_balanced_error": float(np.mean(final_test_graph_errors)),
        "mean_final_train_to_test_gap": float(np.mean(final_test_graph_errors - final_train_graph_errors)),
        "minimum_final_test_nondead_atom_count": int(min(nondead)),
        "maximum_final_test_activation_share": float(max(max_shares)),
        "passes_registered_direction_gate": direction_gate,
        "passes_registered_health_gate": health_gate,
        "passes_registered_view_gate": bool(direction_gate and health_gate),
        "provisional_due_to_partial_folds": bool(not full_five_fold),
    }


def classify(views: dict[str, Any]) -> dict[str, Any]:
    required = ("stratified", "exact_isomorphism_grouped")
    if any(name not in views or views[name]["summary"]["fold_count"] != 5 for name in required):
        return {
            "classification": "PARTIAL_R0D_TIMING_OR_DEBUG_RUN",
            "passes_r0d": False,
            "next_step": "Complete all five folds in both frozen raw views before making a route decision.",
        }
    stratified_view = views["stratified"]
    grouped_view = views["exact_isomorphism_grouped"]
    stratified = stratified_view["summary"]
    grouped = grouped_view["summary"]
    split_pass = bool(
        stratified_view["split_audit"]["passes_partition_gate"]
        and grouped_view["split_audit"]["passes_partition_gate"]
    )
    health_pass = bool(
        stratified["passes_registered_health_gate"] and grouped["passes_registered_health_gate"]
    )
    direction_pass = bool(
        stratified["passes_registered_direction_gate"] and grouped["passes_registered_direction_gate"]
    )
    grouped_opposite = bool(
        grouped["mean_graph_balanced_relative_reduction"] <= 0.0
        or grouped["positive_reduction_fold_count"] < 4
    )
    if not split_pass:
        label = "FAIL_R0D_SPLIT_INTEGRITY"
        next_step = "Stop and repair only the split implementation before interpreting dictionary results."
    elif not health_pass:
        label = "FAIL_R0D_DICTIONARY_HEALTH"
        next_step = "Stop this frozen route; do not add restarts or scan K/T to repair the result."
    elif not direction_pass and grouped_opposite:
        label = "FAIL_R0D_GROUPED_GENERALIZATION"
        next_step = "Stop before R0-A: learned reconstruction does not generalize across exact structures."
    elif not direction_pass:
        label = "FAIL_R0D_HELDOUT_RECONSTRUCTION_GAIN"
        next_step = "Stop before R0-A: FINAL lacks the registered held-out gain over the same INIT."
    else:
        label = "PASS_R0D_REAL_DICTIONARY_OPTIMIZATION"
        next_step = "Freeze these dictionaries/readouts and design R0-A around STATS+INIT versus STATS+FINAL."
    return {
        "classification": label,
        "passes_r0d": bool(split_pass and health_pass and direction_pass and not grouped_opposite),
        "passes_split_gate": split_pass,
        "passes_health_gate": health_pass,
        "passes_direction_gate": direction_pass,
        "grouped_systematic_opposite_failure": grouped_opposite,
        "next_step": next_step,
    }


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def render_report(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    lines = [
        "# IMDB-BINARY R0-D：raw WALK dictionary optimization / health audit",
        "",
        "> 日期：2026-07-31",
        ">",
        f"> 结论：**{decision['classification']}**",
        "",
        "## 1. 本轮问题与边界",
        "",
        "本轮只检验：同一个 outer-train fold 上的一次 deterministic INIT，经普通 KSVD 更新后，是否稳定改善 held-out sparse reconstruction，并保持字典不坍缩。",
        "",
        "- 数据：完整 raw IMDB-BINARY；cleaned 未进入本轮。",
        "- 两个视图：普通 stratified 5-fold 与 exact-isomorphism-grouped 5-fold。",
        "- 表示：7-node WALK first-discovery-order adjacency，21 维。",
        "- 每图 `min(n,24)` patches，sampling seed `20260731`。",
        "- `K=12, T=2, T_min=1, updates=25`，每 fold 仅一个 deterministic maximin INIT，restart=0。",
        "- centering、INIT、KSVD、PCA 和 medoid 均只用 outer train。",
        "- atom 是否可命名不构成 gate；本轮也不使用 graph labels 学字典。",
        "",
        "## 2. Split 审计",
        "",
    ]
    for view_name, title in (
        ("stratified", "raw/stratified"),
        ("exact_isomorphism_grouped", "raw/exact-isomorphism-grouped"),
    ):
        if view_name not in payload["views"]:
            continue
        audit = payload["views"][view_name]["split_audit"]
        lines.extend([
            f"### {title}",
            "",
            f"- partition gate：`{'PASS' if audit['passes_partition_gate'] else 'FAIL'}`；",
            f"- exact-group integrity required：`{audit['group_integrity_required']}`；",
            f"- fold-group leakage count：`{audit['total_group_leakage_count']}`；",
            "- test folds：" + ", ".join(
                f"fold {row['fold_index']} n={row['test_graph_count']} class={row['test_class_counts']}"
                + (f" groups={row['test_structure_group_count']}" if 'test_structure_group_count' in row else "")
                for row in audit["folds"]
            ) + "。",
            "",
        ])

    lines.extend([
        "fold-group leakage count 表示某个 exact group 同时出现在一个 fold 的 train/test 的事件数，不是泄漏图数；stratified 不以此为 gate，grouped 必须为 0。",
        "",
        "## 3. INIT vs FINAL held-out reconstruction",
        "",
        "主误差是先对每张 held-out 图计算 patch-matrix relative Frobenius error，再对图等权平均。",
        "",
    ])
    for view_name, title in (
        ("stratified", "raw/stratified"),
        ("exact_isomorphism_grouped", "raw/exact-isomorphism-grouped"),
    ):
        if view_name not in payload["views"]:
            continue
        view = payload["views"][view_name]
        lines.extend([
            f"### {title}",
            "",
            "| fold | train/test graphs | train/test patches | INIT test GB err | FINAL test GB err | relative reduction | nondead | max usage share | sec |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for fold in view["folds"]:
            init = fold["stages"]["init"]["test"]["graph_balanced"]["mean_relative_reconstruction_error"]
            final = fold["stages"]["final"]["test"]["graph_balanced"]["mean_relative_reconstruction_error"]
            health = fold["stages"]["final"]["test"]["dictionary_health"]
            lines.append(
                f"| {fold['fold_index']} | {fold['train_graph_count']}/{fold['test_graph_count']} | "
                f"{fold['train_patch_count']}/{fold['test_patch_count']} | {_fmt(init)} | {_fmt(final)} | "
                f"{_fmt(fold['attribution']['test_graph_balanced_relative_reduction'])} | "
                f"{health['nondead_atom_count']}/12 | {_fmt(health['maximum_activation_share'])} | "
                f"{fold['runtime_seconds']:.1f} |"
            )
        summary = view["summary"]
        lines.extend([
            "",
            f"- positive folds：`{summary['positive_reduction_fold_count']}/{summary['fold_count']}`；",
            f"- mean graph-balanced reduction：`{_fmt(summary['mean_graph_balanced_relative_reduction'])}`；",
            f"- mean patch-weighted reduction：`{_fmt(summary['mean_patch_weighted_relative_reduction'])}`；",
            f"- FINAL mean train/test graph-balanced error：`{_fmt(summary['mean_final_train_graph_balanced_error'])}` / `{_fmt(summary['mean_final_test_graph_balanced_error'])}`，gap `{_fmt(summary['mean_final_train_to_test_gap'])}`；",
            f"- minimum nondead atoms：`{summary['minimum_final_test_nondead_atom_count']}/12`；",
            f"- maximum atom activation share：`{_fmt(summary['maximum_final_test_activation_share'])}`；",
            f"- registered view gate：`{'PASS' if summary['passes_registered_view_gate'] else 'FAIL'}`。",
            "",
        ])

    lines.extend([
        "## 4. Reconstruction controls",
        "",
        "PCA-12 是非稀疏 rank-12 reconstruction baseline；Gaussian、medoid、INIT、FINAL 均以同一 T=2 OMP 评估。medoid 控制从 INIT 的真实训练 patch 出发，以 deterministic Lloyd real-patch medoid steps 细化，不做超参数搜索。",
        "",
    ])
    for view_name, view in payload["views"].items():
        lines.extend([
            f"### {view_name}",
            "",
            "| fold | Gaussian | Medoid | INIT | FINAL | PCA-12 |",
            "|---:|---:|---:|---:|---:|---:|",
        ])
        for fold in view["folds"]:
            def error(stage: str) -> float:
                return fold["stages"][stage]["test"]["graph_balanced"]["mean_relative_reconstruction_error"]
            lines.append(
                f"| {fold['fold_index']} | {_fmt(error('fixed_gaussian'))} | {_fmt(error('medoid'))} | "
                f"{_fmt(error('init'))} | {_fmt(error('final'))} | {_fmt(error('pca12'))} |"
            )
        lines.append("")

    lines.extend([
        "## 5. 跨 fold 字典描述性稳定性",
        "",
        "matched atom cosine 与 principal-subspace cosine 只作描述，不用于选择 fold、restart 或模型。",
        "",
        "| view | stage | matched cosine mean | minimum pair mean | subspace cosine mean |",
        "|---|---|---:|---:|---:|",
    ])
    for view_name, view in payload["views"].items():
        for stage in ("init", "final"):
            similarity = view["cross_fold_similarity"][stage]
            lines.append(
                f"| {view_name} | {stage} | {_fmt(similarity['matched_cosine_mean']) if similarity['matched_cosine_mean'] is not None else 'NA'} | "
                f"{_fmt(similarity['matched_cosine_minimum_pair']) if similarity.get('matched_cosine_minimum_pair') is not None else 'NA'} | "
                f"{_fmt(similarity['subspace_cosine_mean']) if similarity['subspace_cosine_mean'] is not None else 'NA'} |"
            )

    lines.extend([
        "",
        "## 6. 冻结 gate 判定",
        "",
        "1. 每个 view 至少 4/5 folds 的 graph-balanced reduction 为正；",
        "2. 每个 view 的 mean reduction 至少 10%；",
        "3. 每 fold 至少 10/12 non-dead atoms；",
        "4. 每 fold maximum single-atom activation share 不超过 0.60；",
        "5. grouped view 不得系统性反向失败。",
        "",
        f"最终判定：**{decision['classification']}**。",
        "",
        f"下一步：{decision['next_step']}",
        "",
        "无论通过或失败，本结果都不等价于下游分类收益；只有 R0-D 通过后，才允许在 R0-A 检验 `STATS+FINAL - STATS+INIT`。",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--views", nargs="+", choices=("stratified", "grouped"),
        default=("stratified", "grouped"),
    )
    parser.add_argument("--folds", nargs="+", type=int, default=(0, 1, 2, 3, 4))
    parser.add_argument("--n-iterations", type=int, default=25)
    args = parser.parse_args()
    selected_folds = sorted(set(args.folds))
    if any(index < 0 or index >= 5 for index in selected_folds):
        raise ValueError("--folds must be within 0..4")

    started = time.perf_counter()
    graphs = load_tu_structure_text(args.dataset_root, cleaned=False)
    groups = exact_isomorphism_groups(graphs)
    examples = extract_walk_patch_graphs(
        graphs,
        sampling_seed=20260731,
        patch_size=7,
        max_patches_per_graph=24,
    )
    split_sets = {}
    if "stratified" in args.views:
        split_sets["stratified"] = stratified_graph_folds(graphs, n_splits=5, seed=731301)
    if "grouped" in args.views:
        split_sets["exact_isomorphism_grouped"] = grouped_isomorphism_folds(
            graphs, groups, n_splits=5, seed=731301
        )

    views: dict[str, Any] = {}
    for view_name, folds in split_sets.items():
        split_audit = audit_fold_splits(
            graphs,
            folds,
            groups=groups,
            require_group_integrity=(view_name == "exact_isomorphism_grouped"),
        )
        fold_results = []
        dictionaries = {"init": [], "final": []}
        for fold in folds:
            if fold.fold_index not in selected_folds:
                continue
            fold_started = time.perf_counter()
            result, fold_dictionaries = run_imdb_dictionary_fold(
                examples,
                fold,
                n_atoms=12,
                sparsity=2,
                minimum_sparsity=1,
                n_iterations=args.n_iterations,
                gaussian_seed=0,
            )
            result["runtime_seconds"] = float(time.perf_counter() - fold_started)
            result["dictionaries"] = {
                key: value.tolist() for key, value in fold_dictionaries.items()
            }
            fold_results.append(result)
            dictionaries["init"].append(fold_dictionaries["init"])
            dictionaries["final"].append(fold_dictionaries["final"])
            print(
                f"[{view_name}] fold={fold.fold_index} sec={result['runtime_seconds']:.1f} "
                f"reduction={result['attribution']['test_graph_balanced_relative_reduction']:.4f}",
                flush=True,
            )
        views[view_name] = {
            "split_audit": split_audit,
            "folds": fold_results,
            "summary": summarize_view(fold_results),
            "cross_fold_similarity": {
                stage: cross_replicate_dictionary_similarity(dictionaries[stage])
                for stage in ("init", "final")
            },
        }

    payload = {
        "experiment": "IMDB_BINARY_R0D_WALK_DICTIONARY_AUDIT",
        "date": "2026-07-31",
        "dataset": {
            "name": "IMDB-BINARY",
            "variant": "raw",
            "graph_count": int(len(graphs)),
            "exact_isomorphism_group_count": int(len(groups)),
        },
        "config": {
            "patch_sampling_seed": 20260731,
            "split_seed": 731301,
            "patch_size": 7,
            "patch_dimension": 21,
            "max_patches_per_graph": 24,
            "n_atoms": 12,
            "sparsity": 2,
            "minimum_sparsity": 1,
            "n_iterations": int(args.n_iterations),
            "restarts": 0,
            "selected_folds": selected_folds,
        },
        "views": views,
        "decision": classify(views),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    if args.n_iterations != 25:
        payload["decision"] = {
            "classification": "NONREGISTERED_R0D_DEBUG_RUN",
            "passes_r0d": False,
            "registered_n_iterations": 25,
            "executed_n_iterations": int(args.n_iterations),
            "unregistered_result_before_override": payload["decision"],
            "next_step": "Run the frozen 25-update configuration before making a route decision.",
        }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.report.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps(payload["decision"], indent=2, ensure_ascii=False))
    print(f"wrote {args.json}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
