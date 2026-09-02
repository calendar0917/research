"""Run the frozen raw IMDB-BINARY per-graph dictionary falsification audit."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .imdb_pergraph_dictionary import (
    extract_pergraph_dictionary_features,
    run_pergraph_downstream_fold,
    summarize_pergraph_view,
)
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


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = ROOT / "data" / "TUD" / "IMDB-BINARY"
DEFAULT_JSON = (
    ROOT
    / "tracks"
    / "ksvd"
    / "results"
    / "from_scratch"
    / "imdb_binary_g0_pergraph_dictionary_audit_20260801.json"
)
DEFAULT_REPORT = (
    ROOT
    / "tracks"
    / "ksvd"
    / "results"
    / "from_scratch"
    / "IMDB_BINARY_G0_PERGRAPH_DICTIONARY_AUDIT_20260801.md"
)


def _feature_extraction_summary(descriptors) -> dict[str, Any]:
    init = np.asarray([item.init_reconstruction_error for item in descriptors])
    final = np.asarray([item.final_reconstruction_error for item in descriptors])
    pca = np.asarray([item.pca_reconstruction_error for item in descriptors])
    return {
        "graph_count": int(len(descriptors)),
        "raw_patch_dimension": int(descriptors[0].raw_patch.size),
        "invariant_descriptor_dimension": int(descriptors[0].invariant_final.size),
        "legacy_descriptor_dimension": int(descriptors[0].legacy_final.size),
        "mean_init_reconstruction_error": float(np.mean(init)),
        "mean_final_reconstruction_error": float(np.mean(final)),
        "mean_pca_reconstruction_error": float(np.mean(pca)),
        "mean_init_to_final_relative_reduction": float(
            np.mean((init - final) / np.maximum(init, 1e-12))
        ),
        "final_better_graph_count": int(np.count_nonzero(final < init - 1e-12)),
        "final_better_graph_fraction": float(np.mean(final < init - 1e-12)),
        "maximum_invariant_atom_permutation_abs_difference": float(
            np.max(
                [item.invariant_permutation_max_abs_difference for item in descriptors]
            )
        ),
        "mean_init_nnz": float(np.mean([item.init_mean_nnz for item in descriptors])),
        "mean_final_nnz": float(np.mean([item.final_mean_nnz for item in descriptors])),
    }


def _decision(views: dict[str, Any], selected_folds: list[int]) -> dict[str, Any]:
    stratified = views.get("stratified")
    if stratified is None or len(selected_folds) != 5:
        return {
            "classification": "PARTIAL_PERGRAPH_DIAGNOSTIC",
            "formal_gate_evaluated": False,
            "next_step": "Run all five stratified folds before making a registered decision.",
        }
    summary = stratified["summary"]
    components = {
        "invariant_permutation_difference_le_1e_8": bool(
            summary["maximum_invariant_permutation_difference"] <= 1e-8
        ),
        "reconstruction_positive_5_of_5": bool(
            summary["reconstruction_positive_fold_count"] == 5
        ),
        "primary_update_positive_at_least_4_of_5": bool(
            summary["primary_update_direction_count"] >= 4
        ),
        "mean_primary_update_gain_ge_0_01": bool(
            summary["mean_primary_update_gain"] >= 0.01
        ),
        "aligned_not_below_row_shuffle": bool(
            summary["mean_aligned_over_row_shuffle"] >= 0.0
        ),
        "label_shuffle_in_0_45_0_55": bool(
            0.45
            <= summary["mean_label_shuffle_test_balanced_accuracy"]
            <= 0.55
        ),
    }
    if not components["invariant_permutation_difference_le_1e_8"]:
        classification = "READOUT_INVALID_PERGRAPH_DICTIONARY"
    elif not components["reconstruction_positive_5_of_5"]:
        classification = "FAIL_PERGRAPH_DICTIONARY_FACTORIZATION"
    elif all(components.values()):
        classification = "PASS_PERGRAPH_DICTIONARY_DESCRIPTOR"
    else:
        classification = "RECON_ONLY_PERGRAPH_DICTIONARY"
    return {
        "classification": classification,
        "formal_gate_evaluated": True,
        "gate_components": components,
        "all_components_pass": bool(all(components.values())),
        "next_step": (
            "Only after a PASS consider a separate dynamic-ego replication. "
            "A reconstruction-only result stops the current per-graph readout branch."
        ),
    }


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _render_report(payload: dict[str, Any]) -> str:
    extraction = payload["feature_extraction_summary"]
    lines = [
        "# IMDB-BINARY G0 per-graph dictionary audit",
        "",
        "> Frozen protocol: `KSVD_IMDB_PERGRAPH_DICTIONARY_PROTOCOL_20260801.md`",
        "",
        "## 1. Method",
        "",
        "Each raw IMDB graph independently factorizes its frozen 7-node WALK patch matrix. "
        "The primary descriptor is invariant to atom permutations; the legacy ordered "
        "D/X/Gram readout is secondary only.",
        "",
        "```text",
        "s=7, d=21, patches=min(n,24), K=8, T=2, updates=10",
        "INIT=deterministic maximin, FINAL=same INIT + KSVD, PCA=rank 8",
        "```",
        "",
        "## 2. Label-free feature extraction audit",
        "",
        f"- graphs: `{extraction['graph_count']}`;",
        f"- raw descriptor dimension: `{extraction['raw_patch_dimension']}`;",
        f"- invariant factorization descriptor dimension: `{extraction['invariant_descriptor_dimension']}`;",
        f"- legacy descriptor dimension: `{extraction['legacy_descriptor_dimension']}`;",
        f"- mean reconstruction INIT / FINAL / PCA: `{_fmt(extraction['mean_init_reconstruction_error'])}` / "
        f"`{_fmt(extraction['mean_final_reconstruction_error'])}` / `{_fmt(extraction['mean_pca_reconstruction_error'])}`;",
        f"- mean INIT→FINAL relative reduction: `{_fmt(extraction['mean_init_to_final_relative_reduction'])}`;",
        f"- FINAL-better graphs: `{extraction['final_better_graph_count']}/{extraction['graph_count']}`;",
        f"- invariant atom-permutation max difference: "
        f"`{extraction['maximum_invariant_atom_permutation_abs_difference']:.3e}`;",
        f"- mean nnz INIT / FINAL: `{_fmt(extraction['mean_init_nnz'])}` / `{_fmt(extraction['mean_final_nnz'])}`.",
        "",
        "## 3. Fold results",
        "",
    ]
    for view_name, view in payload["views"].items():
        lines.extend(
            [
                f"### {view_name}",
                "",
                "| fold | STATS | STATS+RAW | +INIT | +FINAL | +PCA | FINAL−INIT | FINAL−PCA | aligned−shuffle |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for fold in view["folds"]:
            score = lambda key: fold["evaluations"][key]["test_balanced_accuracy"]
            attr = fold["attribution"]
            lines.append(
                f"| {fold['fold_index']} | {_fmt(score('stats'))} | "
                f"{_fmt(score('stats_plus_raw'))} | "
                f"{_fmt(score('stats_raw_invariant_init'))} | "
                f"{_fmt(score('stats_raw_invariant_final'))} | "
                f"{_fmt(score('stats_raw_invariant_pca'))} | "
                f"{_fmt(attr['primary_update_gain'])} | "
                f"{_fmt(attr['pca_contrast'])} | "
                f"{_fmt(attr['aligned_over_row_shuffle'])} |"
            )
        summary = view["summary"]
        lines.extend(
            [
                "",
                f"- primary update direction: `{summary['primary_update_direction_count']}/{summary['fold_count']}`;",
                f"- mean primary update gain: `{_fmt(summary['mean_primary_update_gain'])}`;",
                f"- mean beyond STATS+RAW: `{_fmt(summary['mean_beyond_stats_raw_gain'])}`;",
                f"- mean FINAL−PCA: `{_fmt(summary['mean_pca_contrast'])}`;",
                f"- reconstruction-positive folds: `{summary['reconstruction_positive_fold_count']}/{summary['fold_count']}`;",
                f"- mean aligned−row-shuffle: `{_fmt(summary['mean_aligned_over_row_shuffle'])}`;",
                f"- mean legacy atom-order sensitivity: `{_fmt(summary['mean_legacy_atom_order_sensitivity'])}`;",
                f"- mean label-shuffle BA: `{_fmt(summary['mean_label_shuffle_test_balanced_accuracy'])}`.",
                "",
            ]
        )

    decision = payload["decision"]
    lines.extend(
        [
            "## 4. Registered decision",
            "",
            f"> **{decision['classification']}**",
            "",
        ]
    )
    for key, value in decision.get("gate_components", {}).items():
        lines.append(f"- [{'x' if value else ' '}] `{key}`")
    lines.extend(
        [
            "",
            "## 5. Interpretation boundary",
            "",
            "A PASS would support per-graph sparse-factorization statistics as a graph descriptor. "
            "It would not establish a dataset-shared motif vocabulary or aligned atom identities. "
            "A reconstruction-only result means KSVD optimizes each graph's patch cloud but does not "
            "add stable task information beyond the same INIT and raw/statistical controls.",
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
    parser.add_argument("--n-atoms", type=int, default=8)
    parser.add_argument("--sparsity", type=int, default=2)
    parser.add_argument("--n-iterations", type=int, default=10)
    args = parser.parse_args()
    selected_folds = sorted(set(int(value) for value in args.folds))
    if any(value < 0 or value >= 5 for value in selected_folds):
        raise ValueError("--folds must lie in 0..4")

    started = time.perf_counter()
    graphs = load_tu_structure_text(args.dataset_root, cleaned=False)
    groups = exact_isomorphism_groups(graphs)
    examples = extract_walk_patch_graphs(
        graphs,
        sampling_seed=20260731,
        patch_size=7,
        max_patches_per_graph=24,
    )
    descriptors = []
    extraction_started = time.perf_counter()
    for position, example in enumerate(examples, start=1):
        descriptors.append(
            extract_pergraph_dictionary_features(
                example,
                n_atoms=args.n_atoms,
                sparsity=args.sparsity,
                minimum_sparsity=1,
                n_iterations=args.n_iterations,
                atom_permutation_seed=732141,
            )
        )
        if position % 100 == 0:
            print(
                f"[extract] {position}/{len(examples)} "
                f"sec={time.perf_counter() - extraction_started:.1f}",
                flush=True,
            )
    descriptors = tuple(descriptors)

    split_sets = {}
    if "stratified" in args.views:
        split_sets["stratified"] = stratified_graph_folds(
            graphs, n_splits=5, seed=732101
        )
    if "grouped" in args.views:
        split_sets["exact_isomorphism_grouped"] = grouped_isomorphism_folds(
            graphs, groups, n_splits=5, seed=732101
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
                seed=732111 + fold.fold_index,
            )
            result = run_pergraph_downstream_fold(
                examples,
                descriptors,
                fold,
                inner_split,
                graph_shuffle_seed=732121 + 100 * view_offset + fold.fold_index,
                label_shuffle_seed=732131 + 100 * view_offset + fold.fold_index,
            )
            result["inner_split_audit"] = inner_audit
            result["runtime_seconds"] = float(time.perf_counter() - fold_started)
            fold_results.append(result)
            print(
                f"[{view_name}] fold={fold.fold_index} "
                f"update={result['attribution']['primary_update_gain']:.4f} "
                f"pca={result['attribution']['pca_contrast']:.4f} "
                f"sec={result['runtime_seconds']:.1f}",
                flush=True,
            )
        views[view_name] = {
            "outer_split_audit": outer_audit,
            "folds": fold_results,
            "summary": summarize_pergraph_view(fold_results),
        }

    payload = {
        "experiment": "IMDB_BINARY_G0_PERGRAPH_DICTIONARY_AUDIT",
        "dataset": "raw IMDB-BINARY",
        "config": {
            "sampling_seed": 20260731,
            "patch_size": 7,
            "max_patches_per_graph": 24,
            "signal": "walk_first_discovery_upper_triangle",
            "n_atoms": int(args.n_atoms),
            "sparsity": int(args.sparsity),
            "minimum_sparsity": 1,
            "n_iterations": int(args.n_iterations),
            "centering": "none",
            "restarts": 0,
            "outer_split_seed": 732101,
            "selected_folds": selected_folds,
        },
        "feature_extraction_summary": _feature_extraction_summary(descriptors),
        "views": views,
    }
    payload["decision"] = _decision(views, selected_folds)
    payload["runtime_seconds"] = float(time.perf_counter() - started)

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.report.write_text(_render_report(payload), encoding="utf-8")
    print(json.dumps(payload["decision"], indent=2), flush=True)
    print(f"wrote {args.json}", flush=True)
    print(f"wrote {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
