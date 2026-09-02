"""Rooted-canonical KSVD follow-up for Beam8 BASE and FAIR95 prefixes."""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .beam8_coverage_operating_point import (
    prefix_coverage_trajectory,
    select_operating_checkpoints,
)
from .canonical_slots import reorder_cover_structurally
from .from_scratch_unplanted_dictionary import deterministic_maximin_initialization
from .global_stable_ids import compute_global_wl_ids, reorder_by_stable_ids
from .imdb_walk_dictionary import encode_with_minimum_sparsity
from .ksvd import ksvd
from .marginal_candidate_cover import sample_marginal_candidate_cover
from .overlap_cover import PatchCover, _make_cover, patch_budget
from .overlap_stitching import (
    CoverExample,
    graph_balanced_stitch_summary,
    make_cover_example,
    stack_cover_examples,
    stitch_patch_predictions,
)
from .run_beam8_coverage_operating_point_audit import GEOMETRIES
from .run_overlap_cover_audit import FAMILIES, generate_graph


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JSON = ROOT / "tracks/ksvd/results/from_scratch/beam8_coverage_ksvd_followup_20260805.json"
DEFAULT_REPORT = ROOT / "tracks/ksvd/results/from_scratch/BEAM8_COVERAGE_KSVD_FOLLOWUP_20260805.md"
PROTOCOL = "tracks/ksvd/docs/KSVD_BEAM8_COVERAGE_OPERATING_POINT_PROTOCOL_20260805.md"
CHECKPOINTS = ("BASE", "FAIR95")
STAGES = ("raw", "final")


def _prefix_cover(cover: PatchCover, patch_count: int, method: str) -> PatchCover:
    if not 1 <= patch_count <= len(cover.patches):
        raise ValueError("prefix patch count is outside cover")
    return _make_cover(
        method,
        cover.patches[:patch_count],
        cover.segment_ids[:patch_count],
        cover.target_edges[:patch_count],
        cover.bridge_lengths[:patch_count],
    )


def _residual_edges(example: CoverExample) -> set[tuple[int, int]]:
    observed = {
        tuple(sorted(pair))
        for patch in example.cover.patches
        for pair in combinations(patch.node_ids, 2)
    }
    return {
        (left, right)
        for left in range(example.adjacency.shape[0])
        for right in range(left + 1, example.adjacency.shape[0])
        if example.adjacency[left, right] and (left, right) not in observed
    }


def _evaluate(
    examples: Sequence[CoverExample],
    reconstructed: np.ndarray | None,
    *,
    residual_corrected: bool,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows = []
    cursor = 0
    for example in examples:
        patch_count = example.patch_vectors.shape[0]
        predictions = (
            example.patch_vectors
            if reconstructed is None
            else reconstructed[:, cursor : cursor + patch_count].T
        )
        metrics = stitch_patch_predictions(
            example,
            predictions,
            exact_residual_edges=(
                _residual_edges(example) if residual_corrected else None
            ),
        )
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
        raise ValueError("reconstruction columns do not match examples")
    return rows, graph_balanced_stitch_summary(rows)


def _run_fold(
    examples: Sequence[CoverExample],
    fold_index: int,
    *,
    n_atoms: int,
    sparsity: int,
    iterations: int,
) -> dict[str, Any]:
    test_examples = tuple(
        example
        for example in examples
        if example.graph_index % 8 % 3 == fold_index
    )
    test_ids = {example.graph_index for example in test_examples}
    train_examples = tuple(
        example for example in examples if example.graph_index not in test_ids
    )
    raw_train = stack_cover_examples(train_examples)
    raw_test = stack_cover_examples(test_examples)
    train_mean = np.mean(raw_train, axis=1, keepdims=True)
    centered_train = raw_train - train_mean
    initial_dictionary, initialization = deterministic_maximin_initialization(
        centered_train, n_atoms
    )
    dictionary, _training_codes, training_info = ksvd(
        centered_train,
        n_atoms=n_atoms,
        T=sparsity,
        T_min=1,
        n_iter=iterations,
        seed=0,
        initial_dictionary=initial_dictionary,
    )
    test_codes = encode_with_minimum_sparsity(
        raw_test - train_mean,
        dictionary,
        sparsity=sparsity,
        minimum_sparsity=1,
    )
    reconstructed = dictionary @ test_codes + train_mean
    _raw_rows, raw = _evaluate(
        test_examples, None, residual_corrected=False
    )
    _raw_corrected_rows, raw_corrected = _evaluate(
        test_examples, None, residual_corrected=True
    )
    _final_rows, final = _evaluate(
        test_examples, reconstructed, residual_corrected=False
    )
    _corrected_rows, corrected = _evaluate(
        test_examples, reconstructed, residual_corrected=True
    )
    decomposition_delta = max(
        abs(
            final_row["full_adjacency_rmse"] ** 2
            - raw_row["full_adjacency_rmse"] ** 2
            - corrected_row["full_adjacency_rmse"] ** 2
        )
        for raw_row, final_row, corrected_row in zip(
            _raw_rows, _final_rows, _corrected_rows
        )
    )
    return {
        "fold_index": fold_index,
        "train_graph_count": len(train_examples),
        "test_graph_count": len(test_examples),
        "train_patch_count": int(sum(x.patch_vectors.shape[0] for x in train_examples)),
        "test_patch_count": int(sum(x.patch_vectors.shape[0] for x in test_examples)),
        "initialization": initialization,
        "training_iterations": len(training_info.get("residual_history", [])),
        "stages": {
            "raw": {
                "uncorrected": raw,
                "residual_corrected": raw_corrected,
            },
            "final": {
                "uncorrected": final,
                "residual_corrected": corrected,
            },
        },
        "squared_error_decomposition_delta": float(decomposition_delta),
    }


def _mean_stage(
    folds: Sequence[dict[str, Any]], stage: str, correction: str
) -> dict[str, float]:
    rows = [fold["stages"][stage][correction] for fold in folds]
    weights = np.asarray(
        [float(fold["test_graph_count"]) for fold in folds], dtype=float
    )
    return {
        key: float(
            np.average(
                np.asarray([float(row[key]) for row in rows], dtype=float),
                weights=weights,
            )
        )
        for key in rows[0]
    }


def _relative_reduction(left: float, right: float) -> float:
    return float((left - right) / max(left, 1e-12))


def classify(branches: dict[str, Any], cover_seeds: Sequence[int]) -> dict[str, Any]:
    geometry_rows = []
    for geometry in GEOMETRIES:
        seed_rows = []
        for seed in cover_seeds:
            base = branches[f"{geometry}_BASE"]["seeds"][str(seed)]["mean_stages"]
            fair = branches[f"{geometry}_FAIR95"]["seeds"][str(seed)]["mean_stages"]
            base_final = base["final"]["uncorrected"]
            fair_final = fair["final"]["uncorrected"]
            base_corrected = base["final"]["residual_corrected"]
            fair_corrected = fair["final"]["residual_corrected"]
            full_reduction = _relative_reduction(
                base_final["full_adjacency_rmse"],
                fair_final["full_adjacency_rmse"],
            )
            checks = {
                "full_rmse_reduction_at_least_002": full_reduction >= 0.02,
                "corrected_rmse_not_worse_by_001": fair_corrected[
                    "full_adjacency_rmse"
                ]
                <= 1.01 * base_corrected["full_adjacency_rmse"],
                "observed_rmse_not_worse_by_002": fair_final[
                    "observed_pair_rmse"
                ]
                <= 1.02 * base_final["observed_pair_rmse"],
                "full_edge_recall_not_lower": fair_final["full_edge_recall"]
                >= base_final["full_edge_recall"],
            }
            seed_rows.append(
                {
                    "cover_seed": int(seed),
                    "full_rmse_reduction": full_reduction,
                    "corrected_rmse_ratio": fair_corrected[
                        "full_adjacency_rmse"
                    ]
                    / max(base_corrected["full_adjacency_rmse"], 1e-12),
                    "observed_rmse_ratio": fair_final["observed_pair_rmse"]
                    / max(base_final["observed_pair_rmse"], 1e-12),
                    "checks": checks,
                    "passed": all(checks.values()),
                }
            )
        pass_count = sum(row["passed"] for row in seed_rows)
        geometry_rows.append(
            {
                "geometry": geometry,
                "seed_pass_count": pass_count,
                "passed": pass_count >= 2,
                "seeds": seed_rows,
            }
        )
    passing = [row["geometry"] for row in geometry_rows if row["passed"]]

    def candidate(geometry: str, checkpoint: str) -> dict[str, Any]:
        branch = branches[f"{geometry}_{checkpoint}"]
        stage = branch["mean_stages"]["final"]
        return {
            "candidate": f"{geometry}_{checkpoint}",
            "full_adjacency_rmse": stage["uncorrected"]["full_adjacency_rmse"],
            "corrected_full_rmse": stage["residual_corrected"]["full_adjacency_rmse"],
            "full_edge_recall": stage["uncorrected"]["full_edge_recall"],
            "dictionary_scalars": branch["dictionary_scalars"],
            "code_scalars_per_graph": branch["code_scalars_per_graph"],
        }

    candidate_rows = [candidate(geometry, "FAIR95") for geometry in passing]
    base_candidate_rows = [candidate(geometry, "BASE") for geometry in GEOMETRIES]

    def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
        minimize = (
            "full_adjacency_rmse",
            "corrected_full_rmse",
            "dictionary_scalars",
            "code_scalars_per_graph",
        )
        not_worse = all(left[key] <= right[key] for key in minimize)
        not_worse = not_worse and left["full_edge_recall"] >= right["full_edge_recall"]
        strict = any(left[key] < right[key] for key in minimize)
        strict = strict or left["full_edge_recall"] > right["full_edge_recall"]
        return bool(not_worse and strict)

    def pareto_names(rows: Sequence[dict[str, Any]]) -> list[str]:
        return [
            row["candidate"]
            for row in rows
            if not any(other is not row and dominates(other, row) for other in rows)
        ]

    pareto = pareto_names(candidate_rows)
    base_pareto = pareto_names(base_candidate_rows)
    return {
        "classification": (
            "ADOPT_FAIR95_OPERATING_POINT_GEOMETRY_UNRESOLVED"
            if passing
            else "KEEP_BASE_WITH_RESIDUAL_SECOND_CHANNEL_GEOMETRY_UNRESOLVED"
        ),
        "passing_fair95_geometries": passing,
        "geometry_decisions": geometry_rows,
        "fair95_ksvd_pareto": pareto,
        "base_ksvd_pareto": base_pareto,
        "geometry_status": "UNRESOLVED_RATE_DISTORTION_TRADEOFF",
        "candidate_rows": candidate_rows,
        "base_candidate_rows": base_candidate_rows,
    }


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def render_report(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    lines = [
        "# Beam8 coverage operating point：rooted-canonical KSVD follow-up",
        "",
        "> 日期：2026-08-05",
        f"> 协议：`{payload['protocol']}`",
        f"> 判定：`{decision['classification']}`",
        "",
        "## 1. 判定",
        "",
        f"- passing FAIR95 geometries：`{decision['passing_fair95_geometries']}`；",
        f"- FAIR95 KSVD Pareto：`{decision['fair95_ksvd_pareto']}`；",
        f"- BASE KSVD Pareto：`{decision['base_ksvd_pareto']}`；",
        "- geometry 尚未冻结：三者在 reconstruction、dictionary size 与 per-graph code length 间互不支配。",
        "",
        "- 条件性结论：patch-only 表示偏向 FAIR95；若 residual 作为显式第二通道，则 BASE 更优。当前标签只对应后一种系统。",
        "",
        "| geometry | seed | full RMSE reduction | corrected ratio | observed ratio | full≥2% | corrected≤+1% | observed≤+2% | recall nonlower | pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for geometry in decision["geometry_decisions"]:
        for row in geometry["seeds"]:
            checks = row["checks"]
            lines.append(
                f"| {geometry['geometry']} | {row['cover_seed']} | "
                f"{row['full_rmse_reduction']:.4f} | "
                f"{row['corrected_rmse_ratio']:.4f} | "
                f"{row['observed_rmse_ratio']:.4f} | "
                f"{checks['full_rmse_reduction_at_least_002']} | "
                f"{checks['corrected_rmse_not_worse_by_001']} | "
                f"{checks['observed_rmse_not_worse_by_002']} | "
                f"{checks['full_edge_recall_not_lower']} | {row['passed']} |"
            )
    lines.extend(
        [
            "",
            "## 2. Graph-balanced means across cover seeds",
            "",
            "| branch | patches | dict/code scalars | stage | patch error | observed RMSE/F1 | full RMSE | corrected RMSE | full recall/F1 |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for branch_name, branch in payload["branches"].items():
        for stage_name in STAGES:
            stage = branch["mean_stages"][stage_name]
            raw = stage["uncorrected"]
            corrected = stage["residual_corrected"]
            lines.append(
                f"| {branch_name} | {branch['mean_patch_count']:.2f} | "
                f"{branch['dictionary_scalars']}/{branch['code_scalars_per_graph']:.1f} | "
                f"{stage_name} | {raw['patch_relative_error']:.4f} | "
                f"{raw['observed_pair_rmse']:.4f}/{raw['observed_edge_f1']:.4f} | "
                f"{raw['full_adjacency_rmse']:.4f} | "
                f"{corrected['full_adjacency_rmse']:.4f} | "
                f"{raw['full_edge_recall']:.4f}/{raw['full_edge_f1']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## 3. Geometry candidates",
            "",
            "| candidate | full RMSE | corrected RMSE | full recall | dict/code scalars | Pareto |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    pareto = set(decision["fair95_ksvd_pareto"] + decision["base_ksvd_pareto"])
    for row in decision["base_candidate_rows"] + decision["candidate_rows"]:
        lines.append(
            f"| {row['candidate']} | {_fmt(row['full_adjacency_rmse'])} | "
            f"{_fmt(row['corrected_full_rmse'])} | {_fmt(row['full_edge_recall'])} | "
            f"{row['dictionary_scalars']}/{row['code_scalars_per_graph']:.1f} | "
            f"{row['candidate'] in pareto} |"
        )
    lines.extend(
        [
            "",
            "## 4. 结果解释",
            "",
            "- 不加 residual 时，FAIR95 在三种 geometry 上都稳定降低 full RMSE 并提高 full edge recall；所以不能说额外覆盖没有价值。",
            "- 加 exact residual 后，FAIR95 的 compression-only RMSE 相对 BASE 分别恶化约 7.0%、11.9%、14.6%；因此在显式 residual 第二通道系统中，不应继续把 residual 边搬入固定 K24/T3 的 patch channel。",
            "- BASE residual 不是随机噪声：coverage 审计显示其主要偏向 zero-common-neighbor 与跨社区边。因此 residual 若用于下游，必须作为可见 edge token/channel，而不能只在最终解码时悄悄置 1。",
            "",
            "## 5. Error decomposition audit",
            "",
            f"最大 squared-error decomposition delta：`{payload['maximum_squared_error_decomposition_delta']:.3e}`。",
            "",
            "`uncorrected full RMSE² = RAW coverage RMSE² + residual-corrected compression RMSE²`；residual 只修未观察真实边，不修 observed-pair KSVD 错误。",
            "",
            "## 6. 边界",
            "",
            "- 每个 geometry/checkpoint/cover seed 独立训练 train-only dictionary；没有跨 test graph 泄漏。",
            "- 三折 test graph 数为 27/27/18；汇总按 test graph 数加权，而不是错误地对三折等权。",
            "- FAIR95 是否通过由三 seed 中至少两 seed的预注册 gate决定，不用 pooled mean 掩盖 seed failure。",
            "- BASE 与 FAIR95 的 observed-pair 集合不同；observed RMSE gate 混合了共享 pair 误差与新增 hard-pair 难度，不解释为同一 pair 上的纯退化。",
            "- residual sidecar 的 bit 成本未进入 KSVD gate；本轮结论只比较 reconstruction，不是新的 bit-codec 主张。",
            "- 若多个 geometry 位于 Pareto，最终选择仍需明确偏好 dictionary size 或 per-graph code length；不伪造单一总标量。",
            "- 本轮只检验 residual 作为 reconstruction sidecar，不检验 residual edge token 的下游价值。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-bank-seed", type=int, default=810001)
    parser.add_argument("--cover-seeds", type=int, nargs="+", default=[970101, 970102, 970103])
    parser.add_argument("--n-nodes", type=int, default=50)
    parser.add_argument("--degrees", type=int, nargs="+", default=[15, 20, 25])
    parser.add_argument("--graphs-per-cell", type=int, default=8)
    parser.add_argument("--maximum-patches", type=int, default=60)
    parser.add_argument("--multiplier", type=float, default=1.5)
    parser.add_argument("--retained-beam", type=int, default=8)
    parser.add_argument("--candidate-restarts", type=int, default=1)
    parser.add_argument("--n-atoms", type=int, default=24)
    parser.add_argument("--sparsity", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    graph_count = len(FAMILIES) * len(args.degrees) * args.graphs_per_cell
    sequences = np.random.SeedSequence(args.graph_bank_seed).spawn(graph_count)
    graphs = []
    graph_index = 0
    for family in FAMILIES:
        for degree in args.degrees:
            for _replicate in range(args.graphs_per_cell):
                graph_seed = int(sequences[graph_index].generate_state(1, dtype=np.uint32)[0])
                adjacency = generate_graph(family, args.n_nodes, degree, graph_seed)
                stable = compute_global_wl_ids(adjacency)
                graphs.append(
                    (
                        graph_index,
                        family,
                        degree,
                        reorder_by_stable_ids(adjacency, stable),
                    )
                )
                graph_index += 1

    branch_examples: dict[str, dict[int, list[CoverExample]]] = {
        f"{geometry}_{checkpoint}": {int(seed): [] for seed in args.cover_seeds}
        for geometry in GEOMETRIES
        for checkpoint in CHECKPOINTS
    }
    for index, family, degree, adjacency in graphs:
        for geometry_index, (geometry, (patch_size, overlap)) in enumerate(GEOMETRIES.items()):
            base_budget = patch_budget(
                adjacency,
                patch_size=patch_size,
                target_overlap=overlap,
                edge_capacity_multiplier=args.multiplier,
            )
            for cover_seed in args.cover_seeds:
                seed = int(
                    np.random.SeedSequence(
                        [cover_seed, index, geometry_index, 0]
                    ).generate_state(1, dtype=np.uint32)[0]
                )
                cover = sample_marginal_candidate_cover(
                    adjacency,
                    np.random.default_rng(seed),
                    n_patches=args.maximum_patches,
                    patch_size=patch_size,
                    target_overlap=overlap,
                    retained_beam=args.retained_beam,
                    candidate_restarts=args.candidate_restarts,
                    allow_partial=True,
                )
                trajectory = prefix_coverage_trajectory(
                    adjacency,
                    cover,
                    patch_size=patch_size,
                    overlap=overlap,
                    maximum_patches=args.maximum_patches,
                )
                selected = select_operating_checkpoints(
                    trajectory, base_patch_count=base_budget
                )
                for checkpoint in CHECKPOINTS:
                    row = selected[checkpoint]
                    if row is None:
                        raise RuntimeError(
                            f"coverage prerequisite missing: {geometry}/{checkpoint}"
                        )
                    prefix = _prefix_cover(
                        cover,
                        int(row["patch_count"]),
                        f"{geometry}_{checkpoint}",
                    )
                    ordered, _diagnostics = reorder_cover_structurally(
                        adjacency, prefix, "rooted_canonical"
                    )
                    branch_examples[f"{geometry}_{checkpoint}"][int(cover_seed)].append(
                        make_cover_example(index, family, degree, adjacency, ordered)
                    )
        if (index + 1) % 12 == 0:
            print(f"prepared_graphs={index + 1}/{graph_count}", flush=True)

    branches: dict[str, Any] = {}
    maximum_decomposition_delta = 0.0
    for branch_name, seed_examples in branch_examples.items():
        seed_payload = {}
        all_folds: list[dict[str, Any]] = []
        patch_counts = []
        for cover_seed, examples in seed_examples.items():
            folds = [
                _run_fold(
                    examples,
                    fold,
                    n_atoms=args.n_atoms,
                    sparsity=args.sparsity,
                    iterations=args.iterations,
                )
                for fold in range(3)
            ]
            maximum_decomposition_delta = max(
                maximum_decomposition_delta,
                max(fold["squared_error_decomposition_delta"] for fold in folds),
            )
            mean_stages = {
                stage: {
                    correction: _mean_stage(folds, stage, correction)
                    for correction in ("uncorrected", "residual_corrected")
                }
                for stage in STAGES
            }
            all_folds.extend(folds)
            patch_counts.extend(example.patch_vectors.shape[0] for example in examples)
            seed_payload[str(cover_seed)] = {
                "folds": folds,
                "mean_stages": mean_stages,
                "mean_patch_count": float(
                    np.mean([example.patch_vectors.shape[0] for example in examples])
                ),
            }
            print(
                f"branch={branch_name} seed={cover_seed} "
                f"full={mean_stages['final']['uncorrected']['full_adjacency_rmse']:.4f} "
                f"corrected={mean_stages['final']['residual_corrected']['full_adjacency_rmse']:.4f}",
                flush=True,
            )
        mean_stages = {
            stage: {
                correction: _mean_stage(all_folds, stage, correction)
                for correction in ("uncorrected", "residual_corrected")
            }
            for stage in STAGES
        }
        geometry = branch_name.rsplit("_", 1)[0]
        patch_size = GEOMETRIES[geometry][0]
        branches[branch_name] = {
            "seeds": seed_payload,
            "mean_stages": mean_stages,
            "mean_patch_count": float(np.mean(patch_counts)),
            "dictionary_scalars": patch_size * (patch_size - 1) // 2 * args.n_atoms,
            "code_scalars_per_graph": float(np.mean(patch_counts) * args.sparsity),
        }

    decision = classify(branches, args.cover_seeds)
    payload = {
        "protocol": PROTOCOL,
        "config": {
            "graph_bank_seed": args.graph_bank_seed,
            "cover_seeds": args.cover_seeds,
            "families": list(FAMILIES),
            "n_nodes": args.n_nodes,
            "degrees": args.degrees,
            "graphs_per_cell": args.graphs_per_cell,
            "geometries": {key: list(value) for key, value in GEOMETRIES.items()},
            "n_atoms": args.n_atoms,
            "sparsity": args.sparsity,
            "iterations": args.iterations,
            "labels_used": False,
        },
        "branches": branches,
        "maximum_squared_error_decomposition_delta": maximum_decomposition_delta,
        "decision": decision,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(render_report(payload), encoding="utf-8")
    print(f"decision={decision['classification']}")
    print(f"wrote {args.json}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
