"""Run held-out KSVD compression and stitched graph reconstruction audits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .from_scratch_unplanted_dictionary import (
    deterministic_maximin_initialization,
    dictionary_metrics,
)
from .imdb_walk_dictionary import encode_with_minimum_sparsity
from .ksvd import ksvd
from .overlap_cover import patch_budget, sample_edge_target_bridge_cover
from .overlap_stitching import (
    CoverExample,
    fit_pca_basis,
    graph_balanced_stitch_summary,
    make_cover_example,
    reconstruct_with_pca,
    stack_cover_examples,
    stitch_patch_predictions,
)
from .run_overlap_cover_audit import FAMILIES, generate_graph


DEFAULT_JSON = Path(
    "tracks/ksvd/results/from_scratch/ksvd_stitched_reconstruction_audit_20260801.json"
)
DEFAULT_REPORT = Path(
    "tracks/ksvd/results/from_scratch/KSVD_STITCHED_RECONSTRUCTION_AUDIT_20260801.md"
)


def _evaluate_reconstruction(
    examples: Sequence[CoverExample],
    reconstructed: np.ndarray | None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows = []
    cursor = 0
    for example in examples:
        patch_count = example.patch_vectors.shape[0]
        if reconstructed is None:
            predictions = example.patch_vectors
        else:
            predictions = reconstructed[:, cursor : cursor + patch_count].T
        metrics = stitch_patch_predictions(example, predictions)
        rows.append(
            {
                "graph_index": example.graph_index,
                "family": example.family,
                "target_degree": example.target_degree,
                **metrics,
            }
        )
        cursor += patch_count
    if reconstructed is not None and cursor != reconstructed.shape[1]:
        raise ValueError("reconstruction column count does not match graph patches")
    return rows, graph_balanced_stitch_summary(rows)


def _run_fold(
    examples: Sequence[CoverExample],
    fold_index: int,
    *,
    n_atoms: int,
    sparsity: int,
    minimum_sparsity: int,
    n_iterations: int,
    pca_rank: int,
) -> dict[str, Any]:
    test_examples = tuple(
        example for example in examples if example.graph_index % 8 % 3 == fold_index
    )
    test_ids = {example.graph_index for example in test_examples}
    train_examples = tuple(
        example for example in examples if example.graph_index not in test_ids
    )
    if not train_examples or not test_examples:
        raise RuntimeError("fold split produced an empty train or test set")

    raw_train = stack_cover_examples(train_examples)
    raw_test = stack_cover_examples(test_examples)
    train_mean = np.mean(raw_train, axis=1, keepdims=True)
    centered_train = raw_train - train_mean
    centered_test = raw_test - train_mean

    initial_dictionary, initialization = deterministic_maximin_initialization(
        centered_train, n_atoms
    )
    final_dictionary, _training_codes, training_info = ksvd(
        centered_train,
        n_atoms=n_atoms,
        T=sparsity,
        T_min=minimum_sparsity,
        n_iter=n_iterations,
        seed=0,
        initial_dictionary=initial_dictionary,
    )
    dictionaries = {"init": initial_dictionary, "final": final_dictionary}
    reconstructed: dict[str, np.ndarray] = {}
    health: dict[str, Any] = {}
    for stage, dictionary in dictionaries.items():
        train_codes = encode_with_minimum_sparsity(
            centered_train,
            dictionary,
            sparsity=sparsity,
            minimum_sparsity=minimum_sparsity,
        )
        test_codes = encode_with_minimum_sparsity(
            centered_test,
            dictionary,
            sparsity=sparsity,
            minimum_sparsity=minimum_sparsity,
        )
        reconstructed[stage] = dictionary @ test_codes + train_mean
        health[stage] = {
            "train": dictionary_metrics(centered_train, dictionary, train_codes),
            "test": dictionary_metrics(centered_test, dictionary, test_codes),
        }

    pca_basis = fit_pca_basis(centered_train, rank=pca_rank)
    reconstructed["pca3"] = reconstruct_with_pca(centered_test, pca_basis) + train_mean

    stages: dict[str, Any] = {}
    raw_rows, raw_summary = _evaluate_reconstruction(test_examples, None)
    stages["raw"] = {"summary": raw_summary, "graphs": raw_rows}
    for stage in ("init", "final", "pca3"):
        rows, summary = _evaluate_reconstruction(test_examples, reconstructed[stage])
        stages[stage] = {"summary": summary, "graphs": rows}

    return {
        "fold_index": fold_index,
        "train_graph_count": len(train_examples),
        "test_graph_count": len(test_examples),
        "train_patch_count": int(raw_train.shape[1]),
        "test_patch_count": int(raw_test.shape[1]),
        "test_graph_indices": sorted(test_ids),
        "initialization": initialization,
        "training_reconstruction_curve": [
            float(value) for value in training_info["recon_curve"]
        ],
        "dictionary_health": health,
        "stages": stages,
    }


def classify(folds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    folds = tuple(folds)
    raw_invariants = []
    patch_reductions = []
    stitched_reductions = []
    optimization_checks = []
    for fold in folds:
        stages = fold["stages"]
        raw = stages["raw"]["summary"]
        init = stages["init"]["summary"]
        final = stages["final"]["summary"]
        patch_reduction = (
            init["patch_relative_error"] - final["patch_relative_error"]
        ) / max(init["patch_relative_error"], 1e-12)
        stitched_reduction = (
            init["observed_pair_rmse"] - final["observed_pair_rmse"]
        ) / max(init["observed_pair_rmse"], 1e-12)
        patch_reductions.append(float(patch_reduction))
        stitched_reductions.append(float(stitched_reduction))
        final_health = fold["dictionary_health"]["final"]["train"]
        raw_invariants.append(
            {
                "patch_exact": raw["patch_relative_error"] == 0.0,
                "observed_exact": raw["observed_pair_rmse"] == 0.0,
                "observed_f1_exact": raw["observed_edge_f1"] == 1.0,
                "overlap_consistent": raw[
                    "repeated_pair_disagreement_std_mean"
                ]
                == 0.0,
            }
        )
        optimization_checks.append(
            {
                "patch_final_better": final["patch_relative_error"]
                < init["patch_relative_error"],
                "nondead_atoms": final_health["nondead_atom_count"] >= 20,
                "activation_share": final_health["maximum_activation_share"] < 0.50,
            }
        )

    mean_stage = {}
    for stage in ("raw", "init", "final", "pca3"):
        keys = folds[0]["stages"][stage]["summary"].keys()
        mean_stage[stage] = {
            key: float(
                np.mean([fold["stages"][stage]["summary"][key] for fold in folds])
            )
            for key in keys
        }
    init = mean_stage["init"]
    final = mean_stage["final"]
    pca = mean_stage["pca3"]
    optimization_gate = bool(
        all(all(check.values()) for check in optimization_checks)
        and min(patch_reductions) > 0.0
        and float(np.mean(patch_reductions)) >= 0.10
    )
    stitched_checks = {
        "final_better_all_folds": all(value > 0.0 for value in stitched_reductions),
        "mean_rmse_reduction": float(np.mean(stitched_reductions)) >= 0.05,
        "disagreement_not_worse": final[
            "repeated_pair_disagreement_std_mean"
        ]
        <= init["repeated_pair_disagreement_std_mean"] + 1e-12,
        "full_edge_recall_preserved": final["full_edge_recall"]
        >= init["full_edge_recall"] - 0.02,
        "observed_f1_not_worse": final["observed_edge_f1"]
        >= init["observed_edge_f1"] - 1e-12,
        "beats_pca3_rmse": final["observed_pair_rmse"]
        < pca["observed_pair_rmse"],
    }
    raw_gate = all(all(check.values()) for check in raw_invariants)
    stitched_gate = all(stitched_checks.values())
    ksvd_stitch_without_pca = all(
        value for key, value in stitched_checks.items() if key != "beats_pca3_rmse"
    )
    if not raw_gate:
        label = "FAIL_RAW_STITCH_INVARIANTS"
    elif not optimization_gate:
        label = "FAIL_KSVD_COVER_COMPRESSION"
    elif stitched_gate:
        label = "PASS_KSVD_STITCHED_COMPRESSOR"
    elif ksvd_stitch_without_pca and not stitched_checks["beats_pca3_rmse"]:
        label = "KSVD_RECON_ONLY_NO_PCA_ADVANTAGE"
    else:
        label = "FAIL_KSVD_STITCH_CONSISTENCY"
    return {
        "classification": label,
        "raw_gate": raw_gate,
        "optimization_gate": optimization_gate,
        "stitched_gate": stitched_gate,
        "raw_invariants_by_fold": raw_invariants,
        "optimization_checks_by_fold": optimization_checks,
        "patch_relative_reductions": patch_reductions,
        "mean_patch_relative_reduction": float(np.mean(patch_reductions)),
        "stitched_rmse_reductions": stitched_reductions,
        "mean_stitched_rmse_reduction": float(np.mean(stitched_reductions)),
        "stitched_checks": stitched_checks,
        "mean_stages": mean_stage,
    }


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def render_report(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    lines = [
        "# KSVD continuous-cover stitched reconstruction 审计",
        "",
        "> 日期：2026-08-01  ",
        "> train/test graph isolation；无 labels、无 classification。",
        "",
        "## 1. 判定",
        "",
        f"**{decision['classification']}**",
        "",
        "## 2. 三折 held-out 结果",
        "",
        "| fold | train/test graphs | train/test patches | stage | patch rel err | observed RMSE | observed F1 | full edge recall/F1 | overlap disagreement | nondead/max share |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for fold in payload["folds"]:
        for stage in ("raw", "init", "final", "pca3"):
            summary = fold["stages"][stage]["summary"]
            if stage in ("init", "final"):
                health = fold["dictionary_health"][stage]["train"]
                health_text = (
                    f"{health['nondead_atom_count']}/"
                    f"{_fmt(health['maximum_activation_share'])}"
                )
            else:
                health_text = "-"
            lines.append(
                f"| {fold['fold_index']} | {fold['train_graph_count']}/"
                f"{fold['test_graph_count']} | {fold['train_patch_count']}/"
                f"{fold['test_patch_count']} | {stage} | "
                f"{_fmt(summary['patch_relative_error'])} | "
                f"{_fmt(summary['observed_pair_rmse'])} | "
                f"{_fmt(summary['observed_edge_f1'])} | "
                f"{_fmt(summary['full_edge_recall'])}/"
                f"{_fmt(summary['full_edge_f1'])} | "
                f"{_fmt(summary['repeated_pair_disagreement_std_mean'])} | "
                f"{health_text} |"
            )

    lines.extend(
        [
            "",
            "## 3. Fold-balanced stage means",
            "",
            "| stage | patch rel err | observed RMSE | observed F1 | full edge recall | overlap disagreement |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for stage in ("raw", "init", "final", "pca3"):
        mean = decision["mean_stages"][stage]
        lines.append(
            f"| {stage} | {_fmt(mean['patch_relative_error'])} | "
            f"{_fmt(mean['observed_pair_rmse'])} | "
            f"{_fmt(mean['observed_edge_f1'])} | "
            f"{_fmt(mean['full_edge_recall'])} | "
            f"{_fmt(mean['repeated_pair_disagreement_std_mean'])} |"
        )

    lines.extend(
        [
            "",
            "## 4. Registered gates",
            "",
            f"- raw gate：`{decision['raw_gate']}`；",
            f"- KSVD optimization gate：`{decision['optimization_gate']}`；mean patch reduction `{_fmt(decision['mean_patch_relative_reduction'])}`；",
            f"- stitched gate：`{decision['stitched_gate']}`；mean observed RMSE reduction `{_fmt(decision['mean_stitched_rmse_reduction'])}`；",
            f"- stitched checks：`{decision['stitched_checks']}`。",
            "",
            "## 5. 解释边界",
            "",
            "- RAW 是当前 sampler 的 ceiling，不是可部署压缩模型；未观察 pairs 仍固定预测为 0。",
            "- PCA3 与 KSVD 都使用 3 个连续系数，但 PCA 没有 sparse atom identity；胜负只回答压缩误差，不回答 motif 语义。",
            "- overlap disagreement 衡量同一 global pair 从不同 local slots 重构时是否一致，它不会被单纯 patch Frobenius error 自动保证。",
            "- 本轮不评估下游任务；通过也只能把 KSVD 定位为 continuous-cover sparse compressor。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-bank-seed", type=int, default=810001)
    parser.add_argument("--cover-seed", type=int, default=830101)
    parser.add_argument("--n-nodes", type=int, default=50)
    parser.add_argument("--degrees", type=int, nargs="+", default=[15, 20, 25])
    parser.add_argument("--graphs-per-cell", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=10)
    parser.add_argument("--target-overlap", type=int, default=5)
    parser.add_argument("--edge-capacity-multiplier", type=float, default=1.5)
    parser.add_argument("--n-atoms", type=int, default=24)
    parser.add_argument("--sparsity", type=int, default=3)
    parser.add_argument("--minimum-sparsity", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--pca-rank", type=int, default=3)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    examples: list[CoverExample] = []
    graph_sequences = np.random.SeedSequence(args.graph_bank_seed).spawn(
        len(FAMILIES) * len(args.degrees) * args.graphs_per_cell
    )
    graph_index = 0
    for family in FAMILIES:
        for degree in args.degrees:
            for _replicate in range(args.graphs_per_cell):
                graph_seed = int(
                    graph_sequences[graph_index].generate_state(1, dtype=np.uint32)[0]
                )
                adjacency = generate_graph(family, args.n_nodes, degree, graph_seed)
                budget = patch_budget(
                    adjacency,
                    patch_size=args.patch_size,
                    target_overlap=args.target_overlap,
                    edge_capacity_multiplier=args.edge_capacity_multiplier,
                )
                sampler_seed = int(
                    np.random.SeedSequence(
                        [args.cover_seed, graph_index, 5501]
                    ).generate_state(1, dtype=np.uint32)[0]
                )
                cover = sample_edge_target_bridge_cover(
                    adjacency,
                    np.random.default_rng(sampler_seed),
                    n_patches=budget,
                    patch_size=args.patch_size,
                    target_overlap=args.target_overlap,
                )
                examples.append(
                    make_cover_example(
                        graph_index, family, degree, adjacency, cover
                    )
                )
                graph_index += 1

    folds = []
    for fold_index in range(3):
        fold = _run_fold(
            examples,
            fold_index,
            n_atoms=args.n_atoms,
            sparsity=args.sparsity,
            minimum_sparsity=args.minimum_sparsity,
            n_iterations=args.iterations,
            pca_rank=args.pca_rank,
        )
        folds.append(fold)
        summary = fold["stages"]
        print(
            f"fold={fold_index} "
            f"patch_init/final={summary['init']['summary']['patch_relative_error']:.4f}/"
            f"{summary['final']['summary']['patch_relative_error']:.4f} "
            f"stitch_init/final/pca={summary['init']['summary']['observed_pair_rmse']:.4f}/"
            f"{summary['final']['summary']['observed_pair_rmse']:.4f}/"
            f"{summary['pca3']['summary']['observed_pair_rmse']:.4f}",
            flush=True,
        )

    decision = classify(folds)
    payload = {
        "protocol": "ksvd-stitched-reconstruction-audit-v0-20260801",
        "config": {
            "graph_bank_seed": args.graph_bank_seed,
            "cover_seed": args.cover_seed,
            "families": list(FAMILIES),
            "n_nodes": args.n_nodes,
            "target_degrees": args.degrees,
            "graphs_per_family_degree": args.graphs_per_cell,
            "folds": 3,
            "patch_size": args.patch_size,
            "target_overlap": args.target_overlap,
            "edge_capacity_multiplier": args.edge_capacity_multiplier,
            "n_atoms": args.n_atoms,
            "sparsity": args.sparsity,
            "minimum_sparsity": args.minimum_sparsity,
            "iterations": args.iterations,
            "pca_rank": args.pca_rank,
            "threshold": 0.5,
            "labels_used": False,
        },
        "folds": folds,
        "decision": decision,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.report.write_text(render_report(payload), encoding="utf-8")
    print(f"decision={decision['classification']}")
    print(f"wrote {args.json}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
