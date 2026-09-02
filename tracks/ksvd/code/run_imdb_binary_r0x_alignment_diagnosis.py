#!/usr/bin/env python3
"""Run the frozen-dictionary R0-X objective-alignment diagnosis."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .imdb_walk_alignment import run_alignment_fold
from .imdb_walk_dictionary import (
    audit_fold_splits,
    grouped_isomorphism_folds,
    stratified_graph_folds,
)
from .imdb_walk_substrate import (
    exact_isomorphism_groups,
    extract_walk_patch_graphs,
    load_tu_structure_text,
)


DEFAULT_DATASET = Path("data/TUD/IMDB-BINARY")
DEFAULT_SOURCE = Path(
    "tracks/ksvd/results/from_scratch/imdb_binary_r0d_dictionary_audit_20260731.json"
)
DEFAULT_JSON = Path(
    "tracks/ksvd/results/from_scratch/imdb_binary_r0x_alignment_diagnosis_20260801.json"
)
DEFAULT_REPORT = Path(
    "tracks/ksvd/results/from_scratch/IMDB_BINARY_R0X_ALIGNMENT_DIAGNOSIS_20260801.md"
)


def _values(view: dict[str, Any], path: tuple[str, ...]) -> np.ndarray:
    return np.asarray(
        [
            _nested(f, path)
            for f in view["folds"]
        ],
        dtype=np.float64,
    )


def _nested(value: dict[str, Any], path: tuple[str, ...]) -> Any:
    for key in path:
        value = value[key]
    return value


def summarize_view(folds: list[dict[str, Any]], *, grouped: bool) -> dict[str, Any]:
    if not folds:
        raise ValueError("at least one fold is required")
    final_code_r2 = _values(
        {"folds": folds}, ("redundancy", "final_code", "test_explained_fraction")
    )
    init_code_r2 = _values(
        {"folds": folds}, ("redundancy", "init_code", "test_explained_fraction")
    )
    walk_r2 = _values(
        {"folds": folds}, ("redundancy", "raw_walk", "test_explained_fraction")
    )
    gain_r2 = _values(
        {"folds": folds}, ("redundancy", "reconstruction_gain", "test_explained_fraction")
    )
    init_projection = _values(
        {"folds": folds},
        (
            "label_alignment",
            "init_code",
            "after_stats_residualization",
            "test_effect_projection_on_train_direction",
        ),
    )
    final_projection = _values(
        {"folds": folds},
        (
            "label_alignment",
            "final_code",
            "after_stats_residualization",
            "test_effect_projection_on_train_direction",
        ),
    )
    init_cosine = _values(
        {"folds": folds},
        (
            "label_alignment",
            "init_code",
            "after_stats_residualization",
            "train_test_effect_cosine",
        ),
    )
    final_cosine = _values(
        {"folds": folds},
        (
            "label_alignment",
            "final_code",
            "after_stats_residualization",
            "train_test_effect_cosine",
        ),
    )
    gain_cohen_d = _values(
        {"folds": folds},
        ("reconstruction_gain_label_effect", "test", "cohen_d"),
    )
    top3_mass = _values(
        {"folds": folds},
        ("patch_frequency", "test", "top3_frequency_patch_mass"),
    )
    top3_gain = _values(
        {"folds": folds},
        ("patch_frequency", "test", "top3_frequency_positive_gain_contribution"),
    )
    gain_bin_correlation = _values(
        {"folds": folds},
        ("patch_frequency", "test", "bin_mass_positive_total_gain_correlation"),
    )
    projection_delta = final_projection - init_projection
    direction_count = int(
        np.count_nonzero(projection_delta >= 0.0)
        if grouped
        else np.count_nonzero(projection_delta > 0.0)
    )
    required_direction_count = 3 if grouped else 4
    code_r2_delta = final_code_r2 - init_code_r2
    return {
        "fold_count": int(len(folds)),
        "test_explained_fraction": {
            "init_code": [float(value) for value in init_code_r2],
            "final_code": [float(value) for value in final_code_r2],
            "raw_walk": [float(value) for value in walk_r2],
            "reconstruction_gain": [float(value) for value in gain_r2],
        },
        "mean_test_explained_fraction": {
            "init_code": float(np.mean(init_code_r2)),
            "final_code": float(np.mean(final_code_r2)),
            "raw_walk": float(np.mean(walk_r2)),
            "reconstruction_gain": float(np.mean(gain_r2)),
        },
        "code_r2_final_minus_init_per_fold": [float(value) for value in code_r2_delta],
        "mean_code_r2_final_minus_init": float(np.mean(code_r2_delta)),
        "residual_label_projection": {
            "init_per_fold": [float(value) for value in init_projection],
            "final_per_fold": [float(value) for value in final_projection],
            "final_minus_init_per_fold": [float(value) for value in projection_delta],
            "mean_init": float(np.mean(init_projection)),
            "mean_final": float(np.mean(final_projection)),
            "mean_final_minus_init": float(np.mean(projection_delta)),
            "direction_count": direction_count,
            "required_direction_count": required_direction_count,
        },
        "residual_label_effect_cosine": {
            "init_mean": float(np.mean(init_cosine)),
            "final_mean": float(np.mean(final_cosine)),
            "init_per_fold": [float(value) for value in init_cosine],
            "final_per_fold": [float(value) for value in final_cosine],
        },
        "reconstruction_gain_test_cohen_d": [float(value) for value in gain_cohen_d],
        "mean_reconstruction_gain_test_cohen_d": float(np.mean(gain_cohen_d)),
        "patch_frequency": {
            "top3_mass_per_fold": [float(value) for value in top3_mass],
            "top3_positive_gain_contribution_per_fold": [float(value) for value in top3_gain],
            "mean_top3_mass": float(np.mean(top3_mass)),
            "mean_top3_positive_gain_contribution": float(np.mean(top3_gain)),
            "mean_bin_mass_gain_correlation": float(np.mean(gain_bin_correlation)),
        },
        "readout_route_condition": bool(
            direction_count >= required_direction_count and float(np.mean(projection_delta)) > 0.0
        ),
    }


def classify_route(views: dict[str, Any]) -> dict[str, Any]:
    strat = views["stratified"]["summary"]
    grouped = views["exact_isomorphism_grouped"]["summary"]
    readout_plausible = bool(
        strat["readout_route_condition"] and grouped["readout_route_condition"]
    )
    stats_redundancy = bool(
        strat["mean_code_r2_final_minus_init"] > 0.0
        and grouped["mean_code_r2_final_minus_init"] > 0.0
    )
    frequency_candidate = bool(
        min(
            strat["patch_frequency"]["mean_top3_mass"],
            grouped["patch_frequency"]["mean_top3_mass"],
        )
        >= 0.60
        and min(
            strat["patch_frequency"]["mean_top3_positive_gain_contribution"],
            grouped["patch_frequency"]["mean_top3_positive_gain_contribution"],
        )
        >= 0.70
    )
    if readout_plausible:
        route = "READOUT_OR_CROSS_PATCH_RELATION_REMAINS_PLAUSIBLE"
        next_step = (
            "Only now design a new-seed cross-patch relation protocol; retain STATS+INIT vs "
            "STATS+FINAL attribution and do not alter the dictionary objective yet."
        )
    elif stats_redundancy:
        route = "PRIORITIZE_STATS_CONDITIONAL_OBJECTIVE"
        next_step = (
            "Freeze a residual/conditional objective protocol that removes train-fitted graph-stat "
            "nuisance before dictionary learning; do not add a richer readout first."
        )
    elif frequency_candidate:
        route = "PRIORITIZE_FREQUENCY_REWEIGHTED_OBJECTIVE"
        next_step = (
            "Freeze a single frequency-aware objective repair protocol; keep dictionary and readout "
            "otherwise fixed."
        )
    else:
        route = "PRIORITIZE_TASK_ALIGNED_OBJECTIVE_OR_STOP_TASK_CLAIM"
        next_step = (
            "Treat ordinary reconstruction KSVD as a compressor/basis-discovery route; if classification "
            "is mandatory, freeze a new label-conditioned dictionary objective instead of stacking readout features."
        )
    return {
        "route": route,
        "readout_route_plausible": readout_plausible,
        "stats_redundancy_supported": stats_redundancy,
        "frequency_reweighting_candidate": frequency_candidate,
        "next_step": next_step,
    }


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# IMDB-BINARY R0-X objective-alignment diagnosis",
        "",
        "> 日期：2026-08-01  ",
        "> 类型：复用 R0-D frozen dictionaries 的机制诊断，不是新的分类 benchmark。  ",
        "> 协议：`tracks/ksvd/docs/KSVD_IMDB_BINARY_R0X_OBJECTIVE_ALIGNMENT_PROTOCOL_20260801.md`",
        "",
        "## 1. Frozen source",
        "",
        f"- source audit：`{payload['source_audit']}`；不重新训练字典。",
        f"- patch sampling seed：`{payload['config']['patch_sampling_seed']}`；dictionary split seed：`{payload['config']['dictionary_split_seed']}`。",
        "- 检查：STATS redundancy、label residual alignment、reconstruction gain label effect、patch-frequency allocation。",
        "",
        "## 2. Fold-level diagnosis",
        "",
    ]
    for view_name, title in (
        ("stratified", "raw/stratified"),
        ("exact_isomorphism_grouped", "raw/exact-isomorphism-grouped"),
    ):
        view = payload["views"][view_name]
        lines.extend(
            [
                f"### {title}",
                "",
                "| fold | INIT code R2 | FINAL code R2 | gain R2 | residual FINAL−INIT label projection | INIT residual cosine | FINAL residual cosine | gain Cohen d | top3 patch mass | top3 gain share |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for fold in view["folds"]:
            r2 = fold["redundancy"]
            align = fold["label_alignment"]
            lines.append(
                f"| {fold['fold_index']} | {_fmt(r2['init_code']['test_explained_fraction'])} | "
                f"{_fmt(r2['final_code']['test_explained_fraction'])} | "
                f"{_fmt(r2['reconstruction_gain']['test_explained_fraction'])} | "
                f"{_fmt(align['final_code']['after_stats_residualization']['test_effect_projection_on_train_direction'] - align['init_code']['after_stats_residualization']['test_effect_projection_on_train_direction'])} | "
                f"{_fmt(align['init_code']['after_stats_residualization']['train_test_effect_cosine'])} | "
                f"{_fmt(align['final_code']['after_stats_residualization']['train_test_effect_cosine'])} | "
                f"{_fmt(fold['reconstruction_gain_label_effect']['test']['cohen_d'])} | "
                f"{_fmt(fold['patch_frequency']['test']['top3_frequency_patch_mass'])} | "
                f"{_fmt(fold['patch_frequency']['test']['top3_frequency_positive_gain_contribution'])} |"
            )
        summary = view["summary"]
        lines.extend(
            [
                "",
                f"- mean held-out explained fraction：INIT code `{_fmt(summary['mean_test_explained_fraction']['init_code'])}`；FINAL code `{_fmt(summary['mean_test_explained_fraction']['final_code'])}`；raw WALK `{_fmt(summary['mean_test_explained_fraction']['raw_walk'])}`；reconstruction gain `{_fmt(summary['mean_test_explained_fraction']['reconstruction_gain'])}`。",
                f"- mean residual label projection：INIT `{_fmt(summary['residual_label_projection']['mean_init'])}`；FINAL `{_fmt(summary['residual_label_projection']['mean_final'])}`；FINAL−INIT `{_fmt(summary['residual_label_projection']['mean_final_minus_init'])}`；direction `{summary['residual_label_projection']['direction_count']}/{summary['fold_count']}`。",
                f"- mean residual effect cosine：INIT `{_fmt(summary['residual_label_effect_cosine']['init_mean'])}`；FINAL `{_fmt(summary['residual_label_effect_cosine']['final_mean'])}`。",
                f"- mean reconstruction-gain test Cohen d：`{_fmt(summary['mean_reconstruction_gain_test_cohen_d'])}`。",
                f"- top-3 frequency bins：patch mass `{_fmt(summary['patch_frequency']['mean_top3_mass'])}`；positive gain contribution `{_fmt(summary['patch_frequency']['mean_top3_positive_gain_contribution'])}`；mass/gain correlation `{_fmt(summary['patch_frequency']['mean_bin_mass_gain_correlation'])}`。",
                f"- readout route condition：`{'PASS' if summary['readout_route_condition'] else 'FAIL'}`。",
                "",
            ]
        )

    route = payload["route_decision"]
    lines.extend(
        [
            "## 3. Route decision",
            "",
            f"> **{route['route']}**",
            "",
            f"- readout/cross-patch remains plausible：`{route['readout_route_plausible']}`；",
            f"- stats redundancy supported：`{route['stats_redundancy_supported']}`；",
            f"- frequency reweighting candidate：`{route['frequency_reweighting_candidate']}`；",
            f"- next step：{route['next_step']}",
            "",
            "## 4. Boundary",
            "",
            "本诊断没有训练新模型，也没有用结果选择超参数。outer-test labels 只用于固定的 effect consistency 描述，因此不能把本报告中的 projection/R2 当作新的分类性能。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--source-json", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--folds", nargs="+", type=int, default=(0, 1, 2, 3, 4))
    args = parser.parse_args()
    selected_folds = sorted(set(args.folds))
    if any(index < 0 or index >= 5 for index in selected_folds):
        raise ValueError("--folds must be within 0..4")

    started = time.perf_counter()
    source = json.loads(args.source_json.read_text(encoding="utf-8"))
    if source["decision"]["classification"] != "PASS_R0D_REAL_DICTIONARY_OPTIMIZATION":
        raise ValueError("R0-X requires the registered R0-D pass artifact")
    graphs = load_tu_structure_text(args.dataset_root, cleaned=False)
    groups = exact_isomorphism_groups(graphs)
    examples = extract_walk_patch_graphs(
        graphs, sampling_seed=20260731, patch_size=7, max_patches_per_graph=24
    )
    split_sets = {
        "stratified": stratified_graph_folds(graphs, n_splits=5, seed=731301),
        "exact_isomorphism_grouped": grouped_isomorphism_folds(
            graphs, groups, n_splits=5, seed=731301
        ),
    }
    views: dict[str, Any] = {}
    for view_name, folds in split_sets.items():
        grouped = view_name == "exact_isomorphism_grouped"
        outer_audit = audit_fold_splits(
            graphs, folds, groups=groups, require_group_integrity=grouped
        )
        source_folds = {
            int(row["fold_index"]): row for row in source["views"][view_name]["folds"]
        }
        fold_results = []
        for fold in folds:
            if fold.fold_index not in selected_folds:
                continue
            source_fold = source_folds[fold.fold_index]
            result = run_alignment_fold(
                examples,
                fold,
                train_mean=np.asarray(source_fold["train_coordinate_mean"], dtype=np.float64),
                initial_dictionary=np.asarray(source_fold["dictionaries"]["init"], dtype=np.float64),
                final_dictionary=np.asarray(source_fold["dictionaries"]["final"], dtype=np.float64),
                sparsity=2,
                minimum_sparsity=1,
            )
            fold_results.append(result)
            print(
                f"[{view_name}] fold={fold.fold_index} "
                f"final_code_r2={result['redundancy']['final_code']['test_explained_fraction']:.4f} "
                f"residual_delta={result['label_alignment']['final_code']['after_stats_residualization']['test_effect_projection_on_train_direction'] - result['label_alignment']['init_code']['after_stats_residualization']['test_effect_projection_on_train_direction']:.4f}",
                flush=True,
            )
        views[view_name] = {
            "outer_split_audit": outer_audit,
            "folds": fold_results,
            "summary": summarize_view(fold_results, grouped=grouped),
        }

    payload = {
        "experiment": "IMDB_BINARY_R0X_OBJECTIVE_ALIGNMENT_DIAGNOSIS",
        "date": "2026-08-01",
        "source_audit": str(args.source_json),
        "dataset": {
            "name": "IMDB-BINARY",
            "variant": "raw",
            "graph_count": int(len(graphs)),
            "exact_isomorphism_group_count": int(len(groups)),
        },
        "config": {
            "patch_sampling_seed": 20260731,
            "dictionary_split_seed": 731301,
            "patch_size": 7,
            "patch_dimension": 21,
            "max_patches_per_graph": 24,
            "n_atoms": 12,
            "sparsity": 2,
            "minimum_sparsity": 1,
            "n_iterations": 25,
            "restarts": 0,
            "selected_folds": selected_folds,
            "statistics_to_code_fit": "outer_train_least_squares",
            "label_effect": "train_direction_to_heldout_projection_no_classifier",
        },
        "views": views,
        "route_decision": classify_route(views) if selected_folds == [0, 1, 2, 3, 4] else {
            "route": "NONREGISTERED_DEBUG_RUN",
            "readout_route_plausible": False,
            "stats_redundancy_supported": False,
            "frequency_reweighting_candidate": False,
            "next_step": "Run all five registered folds in both views before routing the research path.",
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.report.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps(payload["route_decision"], indent=2, ensure_ascii=False))
    print(f"wrote {args.json}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
