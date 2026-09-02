#!/usr/bin/env python3
"""Run registered raw IMDB-BINARY R0-B atom-pair readout audit."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .imdb_walk_dictionary import (
    audit_fold_splits,
    grouped_isomorphism_folds,
    stratified_graph_folds,
)
from .imdb_walk_downstream import (
    make_inner_train_validation_split,
    run_imdb_downstream_fold,
)
from .imdb_walk_substrate import (
    exact_isomorphism_groups,
    extract_walk_patch_graphs,
    load_tu_structure_text,
)


DEFAULT_DATASET = Path("data/TUD/IMDB-BINARY")
DEFAULT_JSON = Path(
    "tracks/ksvd/results/from_scratch/imdb_binary_r0b_pair_readout_audit_20260801.json"
)
DEFAULT_REPORT = Path(
    "tracks/ksvd/results/from_scratch/IMDB_BINARY_R0B_PAIR_READOUT_AUDIT_20260801.md"
)


def summarize_view(folds: list[dict[str, Any]], *, grouped: bool) -> dict[str, Any]:
    if not folds:
        raise ValueError("at least one fold is required")
    pair_update = np.asarray(
        [fold["attribution"]["pair_update_gain"] for fold in folds], dtype=np.float64
    )
    pair_added = np.asarray(
        [fold["attribution"]["pair_added_value_over_marginal"] for fold in folds],
        dtype=np.float64,
    )
    alignment = np.asarray(
        [fold["attribution"]["pair_correct_alignment_gain_over_shuffle"] for fold in folds],
        dtype=np.float64,
    )
    scores = {
        key: np.asarray(
            [fold["evaluations"][key]["test_balanced_accuracy"] for fold in folds],
            dtype=np.float64,
        )
        for key in (
            "stats",
            "stats_plus_init",
            "stats_plus_final",
            "stats_plus_init_pair",
            "stats_plus_final_pair",
            "stats_plus_shuffled_final_pair",
            "label_shuffle_stats_plus_final_pair",
        )
    }
    label_shuffle = scores["label_shuffle_stats_plus_final_pair"]
    direction_count = int(
        np.count_nonzero(pair_update >= 0.0)
        if grouped
        else np.count_nonzero(pair_update > 0.0)
    )
    required_count = 3 if grouped else 4
    conditions = {
        "pair_update_direction_count": bool(direction_count >= required_count),
        "mean_pair_update_gain": bool(
            float(np.mean(pair_update)) > 0.0
            if grouped
            else float(np.mean(pair_update)) >= 0.02
        ),
        "mean_pair_added_value": bool(
            float(np.mean(pair_added)) >= 0.0
            if grouped
            else float(np.mean(pair_added)) > 0.0
        ),
        "correct_alignment_beats_shuffle": bool(float(np.mean(alignment)) > 0.0),
        "label_shuffle_near_chance": bool(0.45 <= float(np.mean(label_shuffle)) <= 0.55),
    }
    return {
        "fold_count": int(len(folds)),
        "feature_test_balanced_accuracy": {
            key: {
                "per_fold": [float(value) for value in values],
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=0)),
            }
            for key, values in scores.items()
        },
        "pair_update_gain_per_fold": [float(value) for value in pair_update],
        "pair_update_direction_count": direction_count,
        "required_pair_update_direction_count": required_count,
        "mean_pair_update_gain": float(np.mean(pair_update)),
        "pair_added_value_per_fold": [float(value) for value in pair_added],
        "mean_pair_added_value_over_marginal": float(np.mean(pair_added)),
        "pair_correct_alignment_gain_over_shuffle_per_fold": [
            float(value) for value in alignment
        ],
        "mean_pair_correct_alignment_gain_over_shuffle": float(np.mean(alignment)),
        "mean_label_shuffle_test_balanced_accuracy": float(np.mean(label_shuffle)),
        "registered_gate_conditions": conditions,
        "passes_registered_view_gate": bool(all(conditions.values()) and len(folds) == 5),
    }


def classify(views: dict[str, Any]) -> dict[str, Any]:
    stratified = views.get("stratified", {}).get("summary")
    grouped = views.get("exact_isomorphism_grouped", {}).get("summary")
    if stratified is None or grouped is None:
        return {
            "classification": "INCOMPLETE_R0B_VIEWS",
            "passes_r0b": False,
            "next_step": "Run all five folds for both registered raw views.",
        }
    controls_ok = all(
        summary["registered_gate_conditions"]["correct_alignment_beats_shuffle"]
        and summary["registered_gate_conditions"]["label_shuffle_near_chance"]
        for summary in (stratified, grouped)
    )
    if not controls_ok:
        classification = "FAIL_R0B_CONTROL_INTEGRITY"
        next_step = (
            "Stop pair-readout attribution and inspect only the deterministic shuffle/control "
            "implementation; do not tune pair thresholds or dictionary parameters."
        )
    elif not stratified["passes_registered_view_gate"]:
        classification = "FAIL_R0B_PAIR_READOUT_UTILITY"
        next_step = (
            "Conclude that within-patch atom-pair co-activation did not provide the registered "
            "stable bridge; do not add more readout statistics on these outer tests."
        )
    elif not grouped["passes_registered_view_gate"]:
        classification = "FAIL_R0B_GROUPED_GENERALIZATION"
        next_step = (
            "Treat stratified pair gain as insufficient for a mechanism claim because exact-structure "
            "novelty did not preserve it."
        )
    else:
        classification = "PASS_R0B_PAIR_READOUT_UTILITY"
        next_step = (
            "Freeze a separate follow-up protocol to test pair-readout utility on another raw graph "
            "view or dataset; do not optimize this outer test further."
        )
    return {
        "classification": classification,
        "passes_r0b": bool(classification == "PASS_R0B_PAIR_READOUT_UTILITY"),
        "control_integrity_passes": bool(controls_ok),
        "stratified_view_passes": bool(stratified["passes_registered_view_gate"]),
        "grouped_view_passes": bool(grouped["passes_registered_view_gate"]),
        "next_step": next_step,
    }


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def render_report(payload: dict[str, Any]) -> str:
    config = payload["config"]
    lines = [
        "# IMDB-BINARY R0-B atom-pair readout audit",
        "",
        "> 日期：2026-08-01  ",
        "> 数据：完整 raw IMDB-BINARY（1000 图）  ",
        "> 协议：`tracks/ksvd/docs/KSVD_IMDB_BINARY_R0B_PAIR_READOUT_PROTOCOL_20260801.md`",
        "",
        "## 1. Frozen design",
        "",
        f"- outer split seed：`{config['outer_split_seed']}`；5 folds；不扫描额外 split seeds。",
        "- readout-only repair：在 R0-A 的 36-D marginal code 后增加 66-D within-patch atom-pair co-activation。",
        f"- patch：s={config['patch_size']}，d={config['patch_dimension']}，每图 `min(n,{config['max_patches_per_graph']})`；sampling seed `{config['patch_sampling_seed']}`。",
        f"- K={config['n_atoms']}，T={config['sparsity']}，T_min={config['minimum_sparsity']}，updates={config['n_iterations']}，deterministic INIT，restart={config['restarts']}。",
        "- 主归因：`STATS+MARGINAL+PAIR_FINAL` vs `STATS+MARGINAL+PAIR_INIT`。",
        "",
        "## 2. Split audit",
        "",
    ]
    for view_name, title in (
        ("stratified", "raw/stratified"),
        ("exact_isomorphism_grouped", "raw/exact-isomorphism-grouped"),
    ):
        view = payload["views"].get(view_name)
        if view is None:
            continue
        audit = view["outer_split_audit"]
        lines.extend(
            [
                f"### {title}",
                "",
                f"- outer partition gate：`{'PASS' if audit['passes_partition_gate'] else 'FAIL'}`；group integrity required：`{audit['group_integrity_required']}`；group leakage events：`{audit['total_group_leakage_count']}`。",
                "- outer test folds："
                + ", ".join(
                    f"fold {row['fold_index']} n={row['test_graph_count']} class={row['test_class_counts']}"
                    for row in audit["folds"]
                )
                + "。",
                "- selected inner validation sizes："
                + ", ".join(
                    f"fold {fold['fold_index']} {fold['inner_split_audit']['validation_graph_count']} class={fold['inner_split_audit']['validation_class_counts']}"
                    for fold in view["folds"]
                )
                + "。",
                "",
            ]
        )

    lines.extend(["## 3. Fold-level primary attribution", ""])
    for view_name, title in (
        ("stratified", "raw/stratified"),
        ("exact_isomorphism_grouped", "raw/exact-isomorphism-grouped"),
    ):
        view = payload["views"].get(view_name)
        if view is None:
            continue
        lines.extend(
            [
                f"### {title}",
                "",
                "| fold | STATS+M+INIT | STATS+M+FINAL | STATS+M+P+INIT | STATS+M+P+FINAL | pair update | pair added | shuffled pair | FINALpair-shuffle | label shuffle | sec |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for fold in view["folds"]:
            ev = fold["evaluations"]
            score = lambda key: ev[key]["test_balanced_accuracy"]
            attr = fold["attribution"]
            lines.append(
                f"| {fold['fold_index']} | {_fmt(score('stats_plus_init'))} | {_fmt(score('stats_plus_final'))} | "
                f"{_fmt(score('stats_plus_init_pair'))} | {_fmt(score('stats_plus_final_pair'))} | "
                f"{_fmt(attr['pair_update_gain'])} | {_fmt(attr['pair_added_value_over_marginal'])} | "
                f"{_fmt(score('stats_plus_shuffled_final_pair'))} | "
                f"{_fmt(attr['pair_correct_alignment_gain_over_shuffle'])} | "
                f"{_fmt(score('label_shuffle_stats_plus_final_pair'))} | {fold['runtime_seconds']:.1f} |"
            )
        summary = view["summary"]
        lines.extend(
            [
                "",
                f"- pair update direction count：`{summary['pair_update_direction_count']}/{summary['fold_count']}`（required `{summary['required_pair_update_direction_count']}`）；",
                f"- mean pair update gain：`{_fmt(summary['mean_pair_update_gain'])}`；",
                f"- mean pair added value over marginal FINAL：`{_fmt(summary['mean_pair_added_value_over_marginal'])}`；",
                f"- mean correct-alignment over shuffle：`{_fmt(summary['mean_pair_correct_alignment_gain_over_shuffle'])}`；",
                f"- mean label-shuffle BA：`{_fmt(summary['mean_label_shuffle_test_balanced_accuracy'])}`；",
                f"- registered view gate：`{'PASS' if summary['passes_registered_view_gate'] else 'FAIL'}`。",
                "",
            ]
        )

    lines.extend(
        [
            "## 4. Marginal reference",
            "",
            "| view | STATS | STATS+MARGINAL INIT | STATS+MARGINAL FINAL | STATS+M+PAIR INIT | STATS+M+PAIR FINAL |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for view_name, view in payload["views"].items():
        scores = view["summary"]["feature_test_balanced_accuracy"]
        mean = lambda key: scores[key]["mean"]
        lines.append(
            f"| {view_name} | {_fmt(mean('stats'))} | {_fmt(mean('stats_plus_init'))} | "
            f"{_fmt(mean('stats_plus_final'))} | {_fmt(mean('stats_plus_init_pair'))} | "
            f"{_fmt(mean('stats_plus_final_pair'))} |"
        )

    decision = payload["decision"]
    lines.extend(
        [
            "",
            "## 5. Registered decision",
            "",
            f"> **{decision['classification']}**",
            "",
            f"- control integrity：`{'PASS' if decision.get('control_integrity_passes') else 'FAIL'}`；",
            f"- stratified gate：`{'PASS' if decision.get('stratified_view_passes') else 'FAIL'}`；",
            f"- grouped gate：`{'PASS' if decision.get('grouped_view_passes') else 'FAIL'}`；",
            f"- next step：{decision['next_step']}",
            "",
            "## 6. Interpretation boundary",
            "",
            "R0-B 只测试同一 patch 内 learned atoms 的 support-composition readout。通过不等于已经表达跨 patch 的空间关系；失败也不等于所有 relation readout 都不可能。无论结果如何，不在本轮 outer tests 上继续调 pair threshold、K/T、iterations 或 restart。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--views",
        nargs="+",
        choices=("stratified", "grouped"),
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
        split_sets["stratified"] = stratified_graph_folds(
            graphs, n_splits=5, seed=731501
        )
    if "grouped" in args.views:
        split_sets["exact_isomorphism_grouped"] = grouped_isomorphism_folds(
            graphs, groups, n_splits=5, seed=731501
        )

    views: dict[str, Any] = {}
    for view_offset, (view_name, folds) in enumerate(split_sets.items()):
        grouped = view_name == "exact_isomorphism_grouped"
        outer_audit = audit_fold_splits(
            graphs,
            folds,
            groups=groups,
            require_group_integrity=grouped,
        )
        fold_results = []
        for fold in folds:
            if fold.fold_index not in selected_folds:
                continue
            fold_started = time.perf_counter()
            inner_split, inner_audit = make_inner_train_validation_split(
                graphs,
                fold,
                groups=groups if grouped else None,
                grouped=grouped,
                seed=731511 + fold.fold_index,
            )
            result, _dictionaries = run_imdb_downstream_fold(
                graphs,
                examples,
                fold,
                inner_split,
                n_atoms=12,
                sparsity=2,
                minimum_sparsity=1,
                n_iterations=args.n_iterations,
                gaussian_seed=0,
                graph_shuffle_seed=731521 + 100 * view_offset + fold.fold_index,
                label_shuffle_seed=731531 + 100 * view_offset + fold.fold_index,
                include_pair_readout=True,
            )
            result["inner_split_audit"] = inner_audit
            result["runtime_seconds"] = float(time.perf_counter() - fold_started)
            fold_results.append(result)
            print(
                f"[{view_name}] fold={fold.fold_index} sec={result['runtime_seconds']:.1f} "
                f"pair_update={result['attribution']['pair_update_gain']:.4f} "
                f"pair_added={result['attribution']['pair_added_value_over_marginal']:.4f}",
                flush=True,
            )
        views[view_name] = {
            "outer_split_audit": outer_audit,
            "folds": fold_results,
            "summary": summarize_view(fold_results, grouped=grouped),
        }

    payload = {
        "experiment": "IMDB_BINARY_R0B_PAIR_READOUT_AUDIT",
        "date": "2026-08-01",
        "dataset": {
            "name": "IMDB-BINARY",
            "variant": "raw",
            "graph_count": int(len(graphs)),
            "exact_isomorphism_group_count": int(len(groups)),
        },
        "config": {
            "patch_sampling_seed": 20260731,
            "outer_split_seed": 731501,
            "inner_split_seed_rule": "731511 + outer_fold_index",
            "graph_shuffle_seed_rule": "731521 + 100 * view_offset + outer_fold_index",
            "pair_shuffle_seed_rule": "graph_shuffle_seed + 1",
            "label_shuffle_seed_rule": "731531 + 100 * view_offset + outer_fold_index",
            "patch_size": 7,
            "patch_dimension": 21,
            "max_patches_per_graph": 24,
            "n_atoms": 12,
            "sparsity": 2,
            "minimum_sparsity": 1,
            "n_iterations": int(args.n_iterations),
            "restarts": 0,
            "outer_folds": 5,
            "inner_folds": 5,
            "selected_inner_validation_fold": 0,
            "selected_outer_folds": selected_folds,
            "pair_dimension": 66,
            "pair_threshold": 1e-10,
            "regularization_grid": [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
        },
        "views": views,
        "decision": classify(views),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    if (
        args.n_iterations != 25
        or selected_folds != [0, 1, 2, 3, 4]
        or set(args.views) != {"stratified", "grouped"}
    ):
        payload["decision"] = {
            "classification": "NONREGISTERED_R0B_DEBUG_RUN",
            "passes_r0b": False,
            "registered_n_iterations": 25,
            "executed_n_iterations": int(args.n_iterations),
            "registered_outer_folds": [0, 1, 2, 3, 4],
            "executed_outer_folds": selected_folds,
            "registered_views": ["stratified", "grouped"],
            "executed_views": list(args.views),
            "unregistered_result_before_override": payload["decision"],
            "next_step": "Run the complete frozen configuration before making a route decision.",
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
