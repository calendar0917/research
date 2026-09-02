#!/usr/bin/env python3
"""Run registered label-free R1-A characterization of R0-D IMDB dictionaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .from_scratch_unplanted_dictionary import cross_replicate_dictionary_similarity
from .imdb_walk_basis_characterization import (
    classify_basis_views,
    run_basis_characterization_fold,
    summarize_basis_view,
)
from .imdb_walk_dictionary import grouped_isomorphism_folds, stratified_graph_folds
from .imdb_walk_substrate import (
    exact_isomorphism_groups,
    extract_walk_patch_graphs,
    load_tu_structure_text,
)


DEFAULT_DATASET = Path("data/TUD/IMDB-BINARY")
DEFAULT_R0D = Path(
    "tracks/ksvd/results/from_scratch/imdb_binary_r0d_dictionary_audit_20260731.json"
)
DEFAULT_JSON = Path(
    "tracks/ksvd/results/from_scratch/imdb_binary_r1a_basis_characterization_20260801.json"
)
DEFAULT_REPORT = Path(
    "tracks/ksvd/results/from_scratch/IMDB_BINARY_R1A_BASIS_CHARACTERIZATION_20260801.md"
)


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# IMDB-BINARY R1-A label-free basis characterization",
        "",
        "> 日期：2026-08-01  ",
        "> 输入：R0-D registered dictionaries；不重新训练、不使用 labels 作 gate  ",
        "> 协议：`tracks/ksvd/docs/KSVD_IMDB_BINARY_R1A_BASIS_CHARACTERIZATION_PROTOCOL_20260801.md`",
        "",
        "## 1. Frozen design",
        "",
        "- raw IMDB-BINARY；stratified/grouped 各 5 folds。",
        "- 复用 R0-D：s=7,d=21,K=12,T=2,updates=25，单 deterministic INIT，restart=0。",
        "- top-activating patches：每 atom 在 held-out test 中取 absolute coefficient top-5。",
        "- 主 gate：usage、cross-fold stability、nearest-real-patch proximity、top-patch diversity。",
        "- graph labels 仅保存在 top-patch descriptive metadata，不进入任何 gate。",
        "",
        "## 2. View summary",
        "",
        "| view | nondead | INIT cross-fold | FINAL cross-fold | INIT nearest | FINAL nearest | Gaussian nearest | FINAL-Gaussian | top5 unique WALK | edge-regime dispersion | INIT->FINAL | gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for view_name, view in payload["views"].items():
        summary = view["summary"]
        lines.append(
            f"| {view_name} | {min(summary['final_nondead_atom_counts'])}/12 | "
            f"{_fmt(summary['cross_fold_init_matched_cosine_mean'])} | "
            f"{_fmt(summary['cross_fold_final_matched_cosine_mean'])} | "
            f"{_fmt(summary['mean_init_nearest_real_patch_cosine'])} | "
            f"{_fmt(summary['mean_final_nearest_real_patch_cosine'])} | "
            f"{_fmt(summary['mean_gaussian_nearest_real_patch_cosine'])} | "
            f"{_fmt(summary['mean_final_minus_gaussian_nearest_cosine'])} | "
            f"{_fmt(summary['mean_top5_unique_walk_vectors'])} | "
            f"{_fmt(summary['mean_top5_edge_count_dispersion'])} | "
            f"{_fmt(summary['mean_init_to_final_matched_cosine'])} | "
            f"{'PASS' if summary['passes_registered_view_gate'] else 'FAIL'} |"
        )

    lines.extend(["", "## 3. Fold details", ""])
    for view_name, view in payload["views"].items():
        lines.extend([
            f"### {view_name}",
            "",
            "| fold | nondead | INIT nearest | FINAL nearest | Gaussian nearest | top5 unique WALK | edge-regime dispersion | INIT->FINAL cosine |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for fold in view["folds"]:
            stage = fold["stages"]
            nearest = lambda key: stage[key]["nearest_real_patch_proximity"]["mean_nearest_absolute_cosine"]
            lines.append(
                f"| {fold['fold_index']} | {stage['final']['dictionary_health']['nondead_atom_count']}/12 | "
                f"{_fmt(nearest('init'))} | {_fmt(nearest('final'))} | {_fmt(nearest('fixed_gaussian'))} | "
                f"{_fmt(stage['final']['top5_unique_walk_mean'])} | "
                f"{_fmt(stage['final']['top5_edge_count_mean_std_across_atoms'])} | "
                f"{_fmt(fold['init_to_final_similarity']['mean_absolute_matched_atom_cosine'])} |"
            )
        lines.append("")

    lines.extend(["## 4. Registered conditions", ""])
    for view_name, view in payload["views"].items():
        lines.append(f"### {view_name}")
        lines.append("")
        for key, value in view["summary"]["registered_gate_conditions"].items():
            lines.append(f"- [{'x' if value else ' '}] `{key}`")
        lines.append("")

    decision = payload["decision"]
    lines.extend([
        "## 5. Decision",
        "",
        f"> **{decision['classification']}**",
        "",
        f"- next step：{decision['next_step']}",
        "",
        "## 6. Interpretation boundary",
        "",
        "本轮不重新打开 classification claim。通过只说明 learned basis 具有 label-free compressor/basis-discovery 意义；不要求 atom 都可命名，也不表示它们是 task-optimal motifs。",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--r0d-json", type=Path, default=DEFAULT_R0D)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    r0d = json.loads(args.r0d_json.read_text(encoding="utf-8"))
    if r0d["decision"]["classification"] != "PASS_R0D_REAL_DICTIONARY_OPTIMIZATION":
        raise ValueError("R1-A requires the registered passing R0-D artifact")
    if r0d["config"] != {
        "patch_sampling_seed": 20260731,
        "split_seed": 731301,
        "patch_size": 7,
        "patch_dimension": 21,
        "max_patches_per_graph": 24,
        "n_atoms": 12,
        "sparsity": 2,
        "minimum_sparsity": 1,
        "n_iterations": 25,
        "restarts": 0,
        "selected_folds": [0, 1, 2, 3, 4],
    }:
        raise ValueError("R0-D artifact does not match the frozen R1-A input")

    graphs = load_tu_structure_text(args.dataset_root, cleaned=False)
    groups = exact_isomorphism_groups(graphs)
    examples = extract_walk_patch_graphs(
        graphs, sampling_seed=20260731, patch_size=7, max_patches_per_graph=24
    )
    splits = {
        "stratified": stratified_graph_folds(graphs, n_splits=5, seed=731301),
        "exact_isomorphism_grouped": grouped_isomorphism_folds(
            graphs, groups, n_splits=5, seed=731301
        ),
    }

    views = {}
    for view_name, view_splits in splits.items():
        source_folds = sorted(r0d["views"][view_name]["folds"], key=lambda item: item["fold_index"])
        source_by_index = {int(item["fold_index"]): item for item in source_folds}
        folds = []
        dictionaries = {"init": [], "final": []}
        for split in view_splits:
            source = source_by_index[split.fold_index]
            fold_dictionaries = {
                key: np.asarray(value, dtype=np.float64)
                for key, value in source["dictionaries"].items()
            }
            result = run_basis_characterization_fold(
                examples,
                split,
                fold_dictionaries,
                np.asarray(source["train_coordinate_mean"], dtype=np.float64),
                sparsity=2,
                minimum_sparsity=1,
                top_k=5,
            )
            folds.append(result)
            dictionaries["init"].append(fold_dictionaries["init"])
            dictionaries["final"].append(fold_dictionaries["final"])
            print(
                f"[{view_name}] fold={split.fold_index} "
                f"nearest={result['stages']['final']['nearest_real_patch_proximity']['mean_nearest_absolute_cosine']:.4f} "
                f"unique={result['stages']['final']['top5_unique_walk_mean']:.2f}",
                flush=True,
            )
        cross_fold = {
            stage: cross_replicate_dictionary_similarity(dictionaries[stage])
            for stage in ("init", "final")
        }
        views[view_name] = {
            "folds": folds,
            "cross_fold_similarity": cross_fold,
            "summary": summarize_basis_view(folds, cross_fold),
        }

    payload = {
        "experiment": "IMDB_BINARY_R1A_LABEL_FREE_BASIS_CHARACTERIZATION",
        "date": "2026-08-01",
        "dataset": {"name": "IMDB-BINARY", "variant": "raw", "graph_count": len(graphs)},
        "source_r0d_json": str(args.r0d_json),
        "config": {
            "reuse_r0d_dictionaries": True,
            "retrain_dictionary": False,
            "labels_used_in_gate": False,
            "top_k": 5,
            **r0d["config"],
        },
        "views": views,
        "decision": classify_basis_views(views),
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
