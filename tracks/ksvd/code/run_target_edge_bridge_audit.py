"""Run the registered target-edge bridge and multi-chain cover audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .overlap_cover import (
    PatchCover,
    audit_cover,
    cover_set_similarity,
    mapped_replay_relabel_invariance,
    patch_budget,
    remap_cover,
    sample_edge_target_bridge_cover,
    sample_frontier_cover,
    sample_independent_walk_cover,
    sample_multi_chain_target_cover,
)
from .run_overlap_cover_audit import FAMILIES, generate_graph


DEFAULT_JSON = Path(
    "tracks/ksvd/results/from_scratch/target_edge_bridge_audit_20260801.json"
)
DEFAULT_REPORT = Path(
    "tracks/ksvd/results/from_scratch/TARGET_EDGE_BRIDGE_AUDIT_20260801.md"
)
DEFAULT_AUDIT_SEEDS = (820101, 820102, 820103)
METHODS = (
    "independent_walk",
    "frontier_cover",
    "edge_target_bridge",
    "multi_chain_target",
)


def _sampler(method: str) -> Callable[..., PatchCover]:
    if method == "independent_walk":
        return sample_independent_walk_cover
    if method == "frontier_cover":
        return sample_frontier_cover
    if method == "edge_target_bridge":
        return sample_edge_target_bridge_cover
    if method == "multi_chain_target":
        return sample_multi_chain_target_cover
    raise ValueError(method)


def _sample(
    method: str,
    adjacency: np.ndarray,
    seed: int,
    *,
    n_patches: int,
    patch_size: int,
    target_overlap: int,
    segment_length: int,
) -> PatchCover:
    kwargs: dict[str, Any] = {
        "n_patches": n_patches,
        "patch_size": patch_size,
    }
    if method != "independent_walk":
        kwargs["target_overlap"] = target_overlap
    if method == "multi_chain_target":
        kwargs["segment_length"] = segment_length
    return _sampler(method)(adjacency, np.random.default_rng(seed), **kwargs)


METRICS = (
    "node_coverage",
    "true_edge_coverage",
    "node_pair_coverage",
    "observed_pair_consistency",
    "observed_pair_accuracy",
    "full_adjacency_accuracy",
    "patch_connected_rate",
    "segment_count",
    "continuous_transition_fraction",
    "within_segment_overlap_mean",
    "within_segment_jaccard_mean",
    "within_segment_nonlocal_jaccard_gap",
    "edge_observation_multiplicity_mean",
    "edge_observation_multiplicity_cv",
    "new_nodes_per_patch_mean",
    "new_pairs_per_patch_mean",
    "target_edge_hit_rate",
    "bridge_length_mean",
    "bridge_length_maximum",
    "mapped_patch_adjacency_match_rate",
    "mapped_transition_slot_map_match_rate",
    "fresh_relabel_patch_set_similarity",
)


def _means(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot summarize empty rows")
    return {
        metric: float(np.mean([float(row[metric]) for row in rows]))
        for metric in METRICS
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_method = {}
    block_by_method = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        block = [row for row in selected if row["family"] == "block"]
        by_method[method] = {"graph_count": len(selected), "mean": _means(selected)}
        block_by_method[method] = {"graph_count": len(block), "mean": _means(block)}
    by_family_degree: dict[str, Any] = {}
    degrees = sorted(set(int(row["target_degree"]) for row in rows))
    for family in FAMILIES:
        for degree in degrees:
            key = f"{family}_d{degree}"
            by_family_degree[key] = {}
            for method in METHODS:
                selected = [
                    row
                    for row in rows
                    if row["family"] == family
                    and row["target_degree"] == degree
                    and row["method"] == method
                ]
                by_family_degree[key][method] = _means(selected)
    return {
        "by_method": by_method,
        "block_by_method": block_by_method,
        "by_family_degree": by_family_degree,
    }


def _run_seed(
    audit_seed: int,
    graph_bank: list[dict[str, Any]],
    *,
    patch_size: int,
    target_overlap: int,
    edge_capacity_multiplier: float,
    segment_length: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for graph_index, graph_record in enumerate(graph_bank):
        adjacency = graph_record["adjacency"]
        budget = patch_budget(
            adjacency,
            patch_size=patch_size,
            target_overlap=target_overlap,
            edge_capacity_multiplier=edge_capacity_multiplier,
        )
        for method_index, method in enumerate(METHODS):
            sampler_seed = int(
                np.random.SeedSequence(
                    [audit_seed, graph_index, method_index, 3301]
                ).generate_state(1, dtype=np.uint32)[0]
            )
            cover = _sample(
                method,
                adjacency,
                sampler_seed,
                n_patches=budget,
                patch_size=patch_size,
                target_overlap=target_overlap,
                segment_length=segment_length,
            )
            audit = audit_cover(adjacency, cover)

            relabel_rng = np.random.default_rng(
                np.random.SeedSequence([audit_seed, graph_index, method_index, 4409])
            )
            permutation = relabel_rng.permutation(adjacency.shape[0])
            mapped = mapped_replay_relabel_invariance(adjacency, cover, permutation)
            relabeled = adjacency[np.ix_(permutation, permutation)]
            fresh = _sample(
                method,
                relabeled,
                sampler_seed,
                n_patches=budget,
                patch_size=patch_size,
                target_overlap=target_overlap,
                segment_length=segment_length,
            )
            fresh_original = remap_cover(fresh, permutation, adjacency)
            rows.append(
                {
                    "audit_seed": audit_seed,
                    "graph_index": graph_index,
                    "family": graph_record["family"],
                    "target_degree": graph_record["target_degree"],
                    "realized_average_degree": graph_record[
                        "realized_average_degree"
                    ],
                    "edge_count": graph_record["edge_count"],
                    "budget": budget,
                    **audit,
                    "mapped_patch_adjacency_match_rate": mapped[
                        "patch_adjacency_match_rate"
                    ],
                    "mapped_transition_slot_map_match_rate": mapped[
                        "transition_slot_map_match_rate"
                    ],
                    "fresh_relabel_patch_set_similarity": cover_set_similarity(
                        cover, fresh_original
                    ),
                }
            )
    return {
        "audit_seed": audit_seed,
        "rows": rows,
        "summary": _summarize(rows),
    }


def classify(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for result in seed_results for row in result["rows"]]
    invariants = {
        "observed_pair_consistency": all(
            row["observed_pair_consistency"] == 1.0 for row in rows
        ),
        "observed_pair_accuracy": all(
            row["observed_pair_accuracy"] == 1.0 for row in rows
        ),
        "mapped_patch_adjacency": all(
            row["mapped_patch_adjacency_match_rate"] == 1.0 for row in rows
        ),
        "mapped_transition_maps": all(
            row["mapped_transition_slot_map_match_rate"] == 1.0 for row in rows
        ),
    }
    seed_gates = []
    for result in seed_results:
        means = result["summary"]["by_method"]
        blocks = result["summary"]["block_by_method"]
        frontier = means["frontier_cover"]["mean"]
        frontier_block = blocks["frontier_cover"]["mean"]

        def deltas(method: str) -> dict[str, Any]:
            candidate = means[method]["mean"]
            candidate_block = blocks[method]["mean"]
            return {
                "edge_gain": candidate["true_edge_coverage"]
                - frontier["true_edge_coverage"],
                "block_edge_gain": candidate_block["true_edge_coverage"]
                - frontier_block["true_edge_coverage"],
                "pair_gain": candidate["node_pair_coverage"]
                - frontier["node_pair_coverage"],
                "continuous_fraction": candidate[
                    "continuous_transition_fraction"
                ],
                "overlap": candidate["within_segment_overlap_mean"],
                "jaccard_gap": candidate[
                    "within_segment_nonlocal_jaccard_gap"
                ],
                "connected_rate": candidate["patch_connected_rate"],
                "target_hit_rate": candidate["target_edge_hit_rate"],
            }

        single = deltas("edge_target_bridge")
        single["passed"] = bool(
            single["edge_gain"] >= 0.03
            and single["block_edge_gain"] >= 0.05
            and single["pair_gain"] >= -0.03
            and single["continuous_fraction"] == 1.0
            and single["overlap"] == 5.0
            and single["jaccard_gap"] > 0.0
            and single["connected_rate"] >= 0.99
            and single["target_hit_rate"] == 1.0
        )
        multi = deltas("multi_chain_target")
        multi["passed"] = bool(
            multi["edge_gain"] >= 0.05
            and multi["block_edge_gain"] >= 0.08
            and multi["pair_gain"] >= 0.0
            and multi["continuous_fraction"] >= 0.70
            and multi["overlap"] == 5.0
            and multi["jaccard_gap"] > 0.0
            and multi["connected_rate"] >= 0.99
            and multi["target_hit_rate"] == 1.0
        )
        seed_gates.append(
            {"audit_seed": result["audit_seed"], "single": single, "multi": multi}
        )

    single_passes = sum(item["single"]["passed"] for item in seed_gates)
    multi_passes = sum(item["multi"]["passed"] for item in seed_gates)
    target_rows = [
        row
        for row in rows
        if row["method"] in ("edge_target_bridge", "multi_chain_target")
    ]
    target_invariants = {
        "patch_connected": all(row["patch_connected_rate"] >= 0.99 for row in target_rows),
        "target_edges_hit": all(row["target_edge_hit_rate"] == 1.0 for row in target_rows),
    }
    if not all(invariants.values()) or not all(target_invariants.values()):
        label = "FAIL_TARGET_BRIDGE_INVARIANTS"
    elif single_passes >= 2:
        label = "PASS_SINGLE_CHAIN_TARGET_BRIDGE"
    elif multi_passes >= 2:
        label = "SELECT_MULTI_CHAIN_PATCH_COVER"
    else:
        label = "FAIL_TARGET_EDGE_SCHEDULING"
    return {
        "classification": label,
        "invariants": invariants,
        "target_invariants": target_invariants,
        "single_seed_pass_count": single_passes,
        "multi_seed_pass_count": multi_passes,
        "seed_gates": seed_gates,
    }


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def render_report(payload: dict[str, Any]) -> str:
    rows = [row for result in payload["seeds"] for row in result["rows"]]
    summary = _summarize(rows)
    lines = [
        "# Target-edge bridge / multi-chain patch cover 审计",
        "",
        "> 日期：2026-08-01  ",
        "> 本轮不训练 KSVD、不使用 labels。",
        "",
        "## 1. 判定",
        "",
        f"**{payload['decision']['classification']}**",
        "",
        "## 2. 全局结果",
        "",
        "| method | edge cover | block edge cover | pair cover | segments | continuous fraction | overlap | gap | connected | target hit | bridge mean/max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        mean = summary["by_method"][method]["mean"]
        block = summary["block_by_method"][method]["mean"]
        lines.append(
            f"| {method} | {_fmt(mean['true_edge_coverage'])} | "
            f"{_fmt(block['true_edge_coverage'])} | "
            f"{_fmt(mean['node_pair_coverage'])} | {_fmt(mean['segment_count'])} | "
            f"{_fmt(mean['continuous_transition_fraction'])} | "
            f"{_fmt(mean['within_segment_overlap_mean'])} | "
            f"{_fmt(mean['within_segment_nonlocal_jaccard_gap'])} | "
            f"{_fmt(mean['patch_connected_rate'])} | "
            f"{_fmt(mean['target_edge_hit_rate'])} | "
            f"{_fmt(mean['bridge_length_mean'])}/{_fmt(mean['bridge_length_maximum'])} |"
        )

    lines.extend(
        [
            "",
            "## 3. Registered seed gates",
            "",
            "| seed | single edge/block/pair gain | continuity/overlap/gap | pass | multi edge/block/pair gain | continuity/overlap/gap | pass |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in payload["decision"]["seed_gates"]:
        single = item["single"]
        multi = item["multi"]
        lines.append(
            f"| {item['audit_seed']} | {_fmt(single['edge_gain'])}/"
            f"{_fmt(single['block_edge_gain'])}/{_fmt(single['pair_gain'])} | "
            f"{_fmt(single['continuous_fraction'])}/{_fmt(single['overlap'])}/"
            f"{_fmt(single['jaccard_gap'])} | {single['passed']} | "
            f"{_fmt(multi['edge_gain'])}/{_fmt(multi['block_edge_gain'])}/"
            f"{_fmt(multi['pair_gain'])} | {_fmt(multi['continuous_fraction'])}/"
            f"{_fmt(multi['overlap'])}/{_fmt(multi['jaccard_gap'])} | "
            f"{multi['passed']} |"
        )

    lines.extend(
        [
            "",
            "## 4. Family / degree breakdown",
            "",
            "| family-degree | method | edge cover | pair cover | continuity |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for key, methods in summary["by_family_degree"].items():
        for method in METHODS:
            mean = methods[method]
            lines.append(
                f"| {key} | {method} | {_fmt(mean['true_edge_coverage'])} | "
                f"{_fmt(mean['node_pair_coverage'])} | "
                f"{_fmt(mean['continuous_transition_fraction'])} |"
            )

    lines.extend(
        [
            "",
            "## 5. 解释边界",
            "",
            "- target-edge scheduling 读取输入 adjacency，但不读取 graph labels；它优化的是结构覆盖，不是下游任务。",
            "- multi-chain 若胜出，只能说明一般图需要多个局部连续坐标系，不能把 segment 顺序称为跨图共享位置。",
            "- fresh relabel similarity 仍是随机稳健性指标，不是严格 invariance；严格声明只来自 mapped replay。",
            "- 只有本轮 cover gate 通过后，才允许进入 raw patch 与 KSVD patch 的 stitched reconstruction 对照。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-seeds", type=int, nargs="+", default=list(DEFAULT_AUDIT_SEEDS))
    parser.add_argument("--n-nodes", type=int, default=50)
    parser.add_argument("--degrees", type=int, nargs="+", default=[15, 20, 25])
    parser.add_argument("--graphs-per-cell", type=int, default=8)
    parser.add_argument("--graph-bank-seed", type=int, default=810001)
    parser.add_argument("--patch-size", type=int, default=10)
    parser.add_argument("--target-overlap", type=int, default=5)
    parser.add_argument("--edge-capacity-multiplier", type=float, default=1.5)
    parser.add_argument("--segment-length", type=int, default=4)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    graph_bank: list[dict[str, Any]] = []
    graph_sequences = np.random.SeedSequence(args.graph_bank_seed).spawn(
        len(FAMILIES) * len(args.degrees) * args.graphs_per_cell
    )
    cursor = 0
    for family in FAMILIES:
        for degree in args.degrees:
            for replicate in range(args.graphs_per_cell):
                graph_seed = int(
                    graph_sequences[cursor].generate_state(1, dtype=np.uint32)[0]
                )
                cursor += 1
                adjacency = generate_graph(family, args.n_nodes, degree, graph_seed)
                graph_bank.append(
                    {
                        "family": family,
                        "target_degree": degree,
                        "replicate": replicate,
                        "graph_seed": graph_seed,
                        "adjacency": adjacency,
                        "edge_count": int(adjacency.sum() // 2),
                        "realized_average_degree": float(np.mean(adjacency.sum(axis=1))),
                    }
                )

    seed_results = []
    for audit_seed in args.audit_seeds:
        result = _run_seed(
            audit_seed,
            graph_bank,
            patch_size=args.patch_size,
            target_overlap=args.target_overlap,
            edge_capacity_multiplier=args.edge_capacity_multiplier,
            segment_length=args.segment_length,
        )
        seed_results.append(result)
        means = result["summary"]["by_method"]
        print(
            f"seed={audit_seed} "
            + " ".join(
                f"{method}:edge={means[method]['mean']['true_edge_coverage']:.3f},"
                f"pair={means[method]['mean']['node_pair_coverage']:.3f},"
                f"cont={means[method]['mean']['continuous_transition_fraction']:.2f}"
                for method in METHODS
            ),
            flush=True,
        )

    decision = classify(seed_results)
    payload = {
        "protocol": "ksvd-target-edge-bridge-audit-v0-20260801",
        "config": {
            "audit_seeds": args.audit_seeds,
            "graph_bank_seed": args.graph_bank_seed,
            "families": list(FAMILIES),
            "n_nodes": args.n_nodes,
            "target_degrees": args.degrees,
            "graphs_per_family_degree": args.graphs_per_cell,
            "patch_size": args.patch_size,
            "target_overlap": args.target_overlap,
            "edge_capacity_multiplier": args.edge_capacity_multiplier,
            "segment_length": args.segment_length,
            "ksvd_run": False,
            "classification_run": False,
        },
        "graph_bank": [
            {key: value for key, value in graph.items() if key != "adjacency"}
            for graph in graph_bank
        ],
        "seeds": seed_results,
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
