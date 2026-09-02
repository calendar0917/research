#!/usr/bin/env python3
"""Run registered raw IMDB-BINARY R0-C statistics-conditioned KSVD audit."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .imdb_walk_conditional import run_conditional_fold
from .imdb_walk_dictionary import (
    audit_fold_splits,
    grouped_isomorphism_folds,
    stratified_graph_folds,
)
from .imdb_walk_downstream import make_inner_train_validation_split
from .imdb_walk_substrate import (
    exact_isomorphism_groups,
    extract_walk_patch_graphs,
    load_tu_structure_text,
)

DEFAULT_DATASET = Path("data/TUD/IMDB-BINARY")
DEFAULT_JSON = Path(
    "tracks/ksvd/results/from_scratch/imdb_binary_r0c_stats_conditional_audit_20260801.json"
)
DEFAULT_REPORT = Path(
    "tracks/ksvd/results/from_scratch/IMDB_BINARY_R0C_STATS_CONDITIONAL_AUDIT_20260801.md"
)


def _scores(folds: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray(
        [fold["evaluations"][key]["test_balanced_accuracy"] for fold in folds],
        dtype=np.float64,
    )


def summarize_view(folds: list[dict[str, Any]], *, grouped: bool) -> dict[str, Any]:
    if not folds:
        raise ValueError("at least one fold is required")
    residual_reduction = np.asarray(
        [fold["attribution"]["residual_graph_balanced_relative_reduction"] for fold in folds],
        dtype=np.float64,
    )
    update = np.asarray(
        [fold["attribution"]["residual_update_gain"] for fold in folds], dtype=np.float64
    )
    beyond_stats = np.asarray(
        [fold["attribution"]["residual_beyond_stats_gain"] for fold in folds], dtype=np.float64
    )
    over_standard = np.asarray(
        [fold["attribution"]["residual_over_standard_final_gain"] for fold in folds],
        dtype=np.float64,
    )
    alignment = np.asarray(
        [fold["attribution"]["residual_alignment_gain_over_shuffle"] for fold in folds],
        dtype=np.float64,
    )
    final_health = [
        fold["stages"]["residual"]["final"]["test"]["dictionary_health"]
        for fold in folds
    ]
    label_shuffle = _scores(folds, "label_shuffle_stats_plus_residual_final")
    full_error_keys = (
        "predictor_only_graph_balanced_mean_relative_error",
        "standard_init_graph_balanced_mean_relative_error",
        "standard_final_graph_balanced_mean_relative_error",
        "residual_init_graph_balanced_mean_relative_error",
        "residual_final_graph_balanced_mean_relative_error",
    )
    direction_count = int(
        np.count_nonzero(update >= 0.0)
        if grouped
        else np.count_nonzero(update > 0.0)
    )
    required_direction = 3 if grouped else 4
    reconstruction_positive = int(np.count_nonzero(residual_reduction > 0.0))
    conditions = {
        "residual_reconstruction_direction_count": bool(reconstruction_positive >= 4),
        "residual_mean_reduction": bool(float(np.mean(residual_reduction)) >= 0.10),
        "residual_nondead_atoms": bool(min(item["nondead_atom_count"] for item in final_health) >= 10),
        "residual_no_collapsed_atom": bool(max(item["maximum_activation_share"] for item in final_health) <= 0.60),
        "update_direction_count": bool(direction_count >= required_direction),
        "mean_update_gain": bool(float(np.mean(update)) > 0.0 if grouped else float(np.mean(update)) >= 0.02),
        "mean_beyond_stats_gain": bool(float(np.mean(beyond_stats)) >= 0.0 if grouped else float(np.mean(beyond_stats)) > 0.0),
        "mean_over_standard_gain": bool(float(np.mean(over_standard)) >= 0.0 if grouped else float(np.mean(over_standard)) > 0.0),
        "alignment_beats_shuffle": bool(float(np.mean(alignment)) > 0.0),
        "label_shuffle_near_chance": bool(0.45 <= float(np.mean(label_shuffle)) <= 0.55),
    }
    return {
        "fold_count": int(len(folds)),
        "feature_test_balanced_accuracy": {
            key: {
                "per_fold": [float(value) for value in _scores(folds, key)],
                "mean": float(np.mean(_scores(folds, key))),
                "std": float(np.std(_scores(folds, key), ddof=0)),
            }
            for key in (
                "stats",
                "stats_plus_standard_init",
                "stats_plus_standard_final",
                "stats_plus_residual_init",
                "stats_plus_residual_final",
                "stats_plus_shuffled_residual_final",
                "label_shuffle_stats_plus_residual_final",
            )
        },
        "residual_reduction_per_fold": [float(value) for value in residual_reduction],
        "positive_residual_reduction_fold_count": reconstruction_positive,
        "mean_residual_reduction": float(np.mean(residual_reduction)),
        "residual_update_gain_per_fold": [float(value) for value in update],
        "residual_update_direction_count": direction_count,
        "required_update_direction_count": required_direction,
        "mean_residual_update_gain": float(np.mean(update)),
        "residual_beyond_stats_gain_per_fold": [float(value) for value in beyond_stats],
        "mean_residual_beyond_stats_gain": float(np.mean(beyond_stats)),
        "residual_over_standard_gain_per_fold": [float(value) for value in over_standard],
        "mean_residual_over_standard_gain": float(np.mean(over_standard)),
        "alignment_gain_per_fold": [float(value) for value in alignment],
        "mean_alignment_gain": float(np.mean(alignment)),
        "mean_label_shuffle_test_balanced_accuracy": float(np.mean(label_shuffle)),
        "mean_test_full_patch_reconstruction_error": {
            key: float(
                np.mean([fold["full_patch_reconstruction"]["test"][key] for fold in folds])
            )
            for key in full_error_keys
        },
        "minimum_residual_final_nondead_atom_count": int(min(item["nondead_atom_count"] for item in final_health)),
        "maximum_residual_final_activation_share": float(max(item["maximum_activation_share"] for item in final_health)),
        "registered_gate_conditions": conditions,
        "passes_registered_view_gate": bool(all(conditions.values()) and len(folds) == 5),
    }


def classify(views: dict[str, Any]) -> dict[str, Any]:
    stratified = views.get("stratified", {}).get("summary")
    grouped = views.get("exact_isomorphism_grouped", {}).get("summary")
    if stratified is None or grouped is None:
        return {
            "classification": "INCOMPLETE_R0C_VIEWS",
            "passes_r0c": False,
            "next_step": "Run all five folds for both registered raw views.",
        }
    controls_ok = all(
        summary["registered_gate_conditions"]["alignment_beats_shuffle"]
        and summary["registered_gate_conditions"]["label_shuffle_near_chance"]
        for summary in (stratified, grouped)
    )
    residual_dictionary_ok = all(
        all(
            summary["registered_gate_conditions"][key]
            for key in (
                "residual_reconstruction_direction_count",
                "residual_mean_reduction",
                "residual_nondead_atoms",
                "residual_no_collapsed_atom",
            )
        )
        for summary in (stratified, grouped)
    )
    if not controls_ok:
        classification = "FAIL_R0C_CONTROL_INTEGRITY"
        next_step = "Stop attribution and inspect only the fixed shuffle/control implementation."
    elif not residual_dictionary_ok:
        classification = "FAIL_R0C_RESIDUAL_DICTIONARY_OPTIMIZATION"
        next_step = "Residual target did not produce a healthy registered dictionary improvement; do not interpret task results as a repair."
    elif not stratified["passes_registered_view_gate"]:
        classification = "FAIL_R0C_STATS_CONDITIONAL_UTILITY"
        next_step = "Statistics-conditioned residual target did not establish primary raw/stratified utility; stop unsupervised task-utility expansion."
    elif not grouped["passes_registered_view_gate"]:
        classification = "FAIL_R0C_GROUPED_GENERALIZATION"
        next_step = "Residual utility did not survive exact-isomorphism grouped generalization."
    else:
        classification = "PASS_R0C_STATS_CONDITIONAL_UTILITY"
        next_step = "Freeze a separate confirmation or attribute-graph protocol; do not tune this outer test further."
    return {
        "classification": classification,
        "passes_r0c": bool(classification == "PASS_R0C_STATS_CONDITIONAL_UTILITY"),
        "control_integrity_passes": bool(controls_ok),
        "residual_dictionary_gate_passes": bool(residual_dictionary_ok),
        "stratified_view_passes": bool(stratified["passes_registered_view_gate"]),
        "grouped_view_passes": bool(grouped["passes_registered_view_gate"]),
        "next_step": next_step,
    }


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def render_report(payload: dict[str, Any]) -> str:
    config = payload["config"]
    lines = [
        "# IMDB-BINARY R0-C statistics-conditioned residual KSVD audit",
        "",
        "> 日期：2026-08-01  ",
        "> 数据：完整 raw IMDB-BINARY（1000 图）  ",
        "> 协议：`tracks/ksvd/docs/KSVD_IMDB_BINARY_R0C_STATS_CONDITIONAL_PROTOCOL_20260801.md`",
        "",
        "## 1. Frozen design",
        "",
        f"- outer split seed：`{config['outer_split_seed']}`；5 folds；不扫描额外 split seeds。",
        "- dictionary target：raw patch minus outer-train graph-statistics-predicted graph patch mean。",
        "- residualizer：graph-balanced outer-train multivariate least squares；不使用 labels。",
        f"- patch：s={config['patch_size']}，d={config['patch_dimension']}，每图 `min(n,{config['max_patches_per_graph']})`；sampling seed `{config['patch_sampling_seed']}`。",
        f"- K={config['n_atoms']}，T={config['sparsity']}，T_min={config['minimum_sparsity']}，updates={config['n_iterations']}，每 branch deterministic INIT，restart={config['restarts']}。",
        "- readout：36-D activation frequency + mean absolute + RMS。",
        "",
        "## 2. Fold-level attribution",
        "",
    ]
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
                "| fold | residual recon reduction | STATS | standard FINAL | residual INIT | residual FINAL | residual update | beyond STATS | over standard | shuffle | label shuffle | nondead | max share |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for fold in view["folds"]:
            ev = fold["evaluations"]
            score = lambda key: ev[key]["test_balanced_accuracy"]
            health = fold["stages"]["residual"]["final"]["test"]["dictionary_health"]
            attr = fold["attribution"]
            lines.append(
                f"| {fold['fold_index']} | {_fmt(attr['residual_graph_balanced_relative_reduction'])} | "
                f"{_fmt(score('stats'))} | {_fmt(score('stats_plus_standard_final'))} | "
                f"{_fmt(score('stats_plus_residual_init'))} | {_fmt(score('stats_plus_residual_final'))} | "
                f"{_fmt(attr['residual_update_gain'])} | {_fmt(attr['residual_beyond_stats_gain'])} | "
                f"{_fmt(attr['residual_over_standard_final_gain'])} | "
                f"{_fmt(score('stats_plus_shuffled_residual_final'))} | "
                f"{_fmt(score('label_shuffle_stats_plus_residual_final'))} | "
                f"{health['nondead_atom_count']}/12 | {_fmt(health['maximum_activation_share'])} |"
            )
        summary = view["summary"]
        full = summary["mean_test_full_patch_reconstruction_error"]
        lines.extend(
            [
                "",
                f"- residual reconstruction：positive `{summary['positive_residual_reduction_fold_count']}/{summary['fold_count']}`；mean reduction `{_fmt(summary['mean_residual_reduction'])}`；minimum nondead `{summary['minimum_residual_final_nondead_atom_count']}/12`；maximum share `{_fmt(summary['maximum_residual_final_activation_share'])}`。",
                f"- mean residual update：`{_fmt(summary['mean_residual_update_gain'])}`；direction `{summary['residual_update_direction_count']}/{summary['fold_count']}`；",
                f"- mean beyond-STATS：`{_fmt(summary['mean_residual_beyond_stats_gain'])}`；mean over standard FINAL：`{_fmt(summary['mean_residual_over_standard_gain'])}`；",
                f"- mean alignment gain over shuffle：`{_fmt(summary['mean_alignment_gain'])}`；label shuffle BA：`{_fmt(summary['mean_label_shuffle_test_balanced_accuracy'])}`；",
                f"- mean full-patch test error：predictor-only `{_fmt(full['predictor_only_graph_balanced_mean_relative_error'])}`；standard INIT/FINAL `{_fmt(full['standard_init_graph_balanced_mean_relative_error'])}/{_fmt(full['standard_final_graph_balanced_mean_relative_error'])}`；residual INIT/FINAL `{_fmt(full['residual_init_graph_balanced_mean_relative_error'])}/{_fmt(full['residual_final_graph_balanced_mean_relative_error'])}`。",
                f"- registered view gate：`{'PASS' if summary['passes_registered_view_gate'] else 'FAIL'}`。",
                "",
            ]
        )

    decision = payload["decision"]
    lines.extend(
        [
            "## 3. Registered decision",
            "",
            f"> **{decision['classification']}**",
            "",
            f"- controls：`{'PASS' if decision.get('control_integrity_passes') else 'FAIL'}`；",
            f"- residual dictionary gate：`{'PASS' if decision.get('residual_dictionary_gate_passes') else 'FAIL'}`；",
            f"- stratified gate：`{'PASS' if decision.get('stratified_view_passes') else 'FAIL'}`；",
            f"- grouped gate：`{'PASS' if decision.get('grouped_view_passes') else 'FAIL'}`；",
            f"- next step：{decision['next_step']}",
            "",
            "## 4. Interpretation boundary",
            "",
            "R0-C 只测试 statistics-conditioned residual target。通过不等于已经证明 labels 参与字典学习是必要的；失败也不等于节点属性路线必然失败。若失败，按照冻结协议停止普通无监督 KSVD 的 IMDB task-utility 扩展。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--views", nargs="+", choices=("stratified", "grouped"), default=("stratified", "grouped"))
    parser.add_argument("--folds", nargs="+", type=int, default=(0, 1, 2, 3, 4))
    parser.add_argument("--n-iterations", type=int, default=25)
    args = parser.parse_args()
    selected_folds = sorted(set(args.folds))
    if any(index < 0 or index >= 5 for index in selected_folds):
        raise ValueError("--folds must be within 0..4")
    started = time.perf_counter()
    graphs = load_tu_structure_text(args.dataset_root, cleaned=False)
    groups = exact_isomorphism_groups(graphs)
    examples = extract_walk_patch_graphs(graphs, sampling_seed=20260731, patch_size=7, max_patches_per_graph=24)
    split_sets = {}
    if "stratified" in args.views:
        split_sets["stratified"] = stratified_graph_folds(graphs, n_splits=5, seed=731601)
    if "grouped" in args.views:
        split_sets["exact_isomorphism_grouped"] = grouped_isomorphism_folds(graphs, groups, n_splits=5, seed=731601)
    views: dict[str, Any] = {}
    for view_offset, (view_name, folds) in enumerate(split_sets.items()):
        grouped = view_name == "exact_isomorphism_grouped"
        outer_audit = audit_fold_splits(graphs, folds, groups=groups, require_group_integrity=grouped)
        fold_results = []
        for fold in folds:
            if fold.fold_index not in selected_folds:
                continue
            fold_started = time.perf_counter()
            inner_split, inner_audit = make_inner_train_validation_split(
                graphs, fold, groups=groups if grouped else None, grouped=grouped, seed=731611 + fold.fold_index
            )
            result, _dictionaries = run_conditional_fold(
                graphs, examples, fold, inner_split, n_atoms=12, sparsity=2, minimum_sparsity=1,
                n_iterations=args.n_iterations,
                graph_shuffle_seed=731621 + 100 * view_offset + fold.fold_index,
                label_shuffle_seed=731631 + 100 * view_offset + fold.fold_index,
            )
            result["inner_split_audit"] = inner_audit
            result["runtime_seconds"] = float(time.perf_counter() - fold_started)
            fold_results.append(result)
            print(
                f"[{view_name}] fold={fold.fold_index} sec={result['runtime_seconds']:.1f} "
                f"recon={result['attribution']['residual_graph_balanced_relative_reduction']:.4f} "
                f"update={result['attribution']['residual_update_gain']:.4f}", flush=True
            )
        views[view_name] = {"outer_split_audit": outer_audit, "folds": fold_results, "summary": summarize_view(fold_results, grouped=grouped)}
    payload = {
        "experiment": "IMDB_BINARY_R0C_STATS_CONDITIONAL_AUDIT",
        "date": "2026-08-01",
        "dataset": {"name": "IMDB-BINARY", "variant": "raw", "graph_count": int(len(graphs)), "exact_isomorphism_group_count": int(len(groups))},
        "config": {
            "patch_sampling_seed": 20260731, "outer_split_seed": 731601,
            "inner_split_seed_rule": "731611 + outer_fold_index",
            "graph_shuffle_seed_rule": "731621 + 100 * view_offset + outer_fold_index",
            "label_shuffle_seed_rule": "731631 + 100 * view_offset + outer_fold_index",
            "patch_size": 7, "patch_dimension": 21, "max_patches_per_graph": 24,
            "n_atoms": 12, "sparsity": 2, "minimum_sparsity": 1,
            "n_iterations": int(args.n_iterations), "restarts": 0, "outer_folds": 5,
            "inner_folds": 5, "selected_inner_validation_fold": 0,
            "selected_outer_folds": selected_folds,
            "residualizer": "graph_balanced_outer_train_multivariate_least_squares",
            "residual_target": "raw_patch_minus_stats_predicted_graph_patch_mean",
        },
        "views": views,
        "decision": classify(views),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    if args.n_iterations != 25 or selected_folds != [0, 1, 2, 3, 4] or set(args.views) != {"stratified", "grouped"}:
        payload["decision"] = {"classification": "NONREGISTERED_R0C_DEBUG_RUN", "passes_r0c": False, "next_step": "Run the complete frozen configuration before making a route decision.", "unregistered_result_before_override": payload["decision"]}
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
