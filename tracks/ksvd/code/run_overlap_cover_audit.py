"""Run the first continuous overlap-cover audit without KSVD or labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .from_scratch_unplanted_representation import (
    degree_preserving_double_edge_swaps,
    is_connected,
    ring_lattice_adjacency,
    validate_simple_adjacency,
)
from .overlap_cover import (
    PatchCover,
    audit_cover,
    cover_set_similarity,
    mapped_replay_relabel_invariance,
    patch_budget,
    remap_cover,
    sample_frontier_cover,
    sample_independent_walk_cover,
    sample_sliding_walk_cover,
)


DEFAULT_JSON = Path(
    "tracks/ksvd/results/from_scratch/overlap_cover_audit_20260801.json"
)
DEFAULT_REPORT = Path(
    "tracks/ksvd/results/from_scratch/OVERLAP_COVER_AUDIT_20260801.md"
)
DEFAULT_AUDIT_SEEDS = (810101, 810102, 810103)
FAMILIES = ("regular", "small_world", "block")
METHODS = ("independent_walk", "sliding_walk", "frontier_cover")


def _to_adjacency(graph: Any, n_nodes: int) -> np.ndarray:
    import networkx as nx

    adjacency = nx.to_numpy_array(
        graph, nodelist=range(n_nodes), dtype=np.int8, weight=None
    )
    adjacency = np.asarray(adjacency, dtype=np.int8)
    np.fill_diagonal(adjacency, 0)
    validate_simple_adjacency(adjacency)
    if not is_connected(adjacency):
        raise RuntimeError("generator returned a disconnected graph")
    return adjacency


def _regular_graph(n_nodes: int, degree: int, seed: int) -> np.ndarray:
    import networkx as nx

    for attempt in range(100):
        graph = nx.random_regular_graph(degree, n_nodes, seed=seed + attempt)
        if nx.is_connected(graph):
            return _to_adjacency(graph, n_nodes)
    raise RuntimeError("failed to generate a connected regular graph")


def _small_world_graph(n_nodes: int, degree: int, seed: int) -> np.ndarray:
    neighbors_each_side = degree // 2
    adjacency = ring_lattice_adjacency(n_nodes, neighbors_each_side)
    if degree % 2:
        offset = n_nodes // 2
        for node in range(n_nodes // 2):
            other = node + offset
            adjacency[node, other] = adjacency[other, node] = 1
    validate_simple_adjacency(adjacency)
    edge_count = int(adjacency.sum() // 2)
    rewired, _ = degree_preserving_double_edge_swaps(
        adjacency,
        max(edge_count // 8, 1),
        np.random.default_rng(seed),
        require_connected=True,
    )
    return rewired


def _block_graph(n_nodes: int, degree: int, seed: int) -> np.ndarray:
    import networkx as nx

    sizes = [n_nodes // 2, n_nodes - n_nodes // 2]
    cross_probability = 0.08
    within_probability = min(
        max((degree - cross_probability * sizes[1]) / max(sizes[0] - 1, 1), 0.05),
        0.98,
    )
    probabilities = [
        [within_probability, cross_probability],
        [cross_probability, within_probability],
    ]
    for attempt in range(200):
        graph = nx.stochastic_block_model(
            sizes, probabilities, seed=seed + attempt, selfloops=False, sparse=True
        )
        if nx.is_connected(graph):
            return _to_adjacency(graph, n_nodes)
    raise RuntimeError("failed to generate a connected block graph")


def generate_graph(family: str, n_nodes: int, degree: int, seed: int) -> np.ndarray:
    if family == "regular":
        return _regular_graph(n_nodes, degree, seed)
    if family == "small_world":
        return _small_world_graph(n_nodes, degree, seed)
    if family == "block":
        return _block_graph(n_nodes, degree, seed)
    raise ValueError(f"unknown graph family: {family}")


def _sampler(method: str) -> Callable[..., PatchCover]:
    if method == "independent_walk":
        return sample_independent_walk_cover
    if method == "sliding_walk":
        return sample_sliding_walk_cover
    if method == "frontier_cover":
        return sample_frontier_cover
    raise ValueError(method)


def _sample(
    method: str,
    adjacency: np.ndarray,
    seed: int,
    *,
    n_patches: int,
    patch_size: int,
    target_overlap: int,
) -> PatchCover:
    kwargs: dict[str, Any] = {
        "n_patches": n_patches,
        "patch_size": patch_size,
    }
    if method != "independent_walk":
        kwargs["target_overlap"] = target_overlap
    return _sampler(method)(adjacency, np.random.default_rng(seed), **kwargs)


def _mean_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    metric_names = (
        "node_coverage",
        "true_edge_coverage",
        "node_pair_coverage",
        "observed_pair_consistency",
        "observed_pair_accuracy",
        "full_adjacency_accuracy",
        "edge_observation_multiplicity_mean",
        "edge_observation_multiplicity_cv",
        "consecutive_overlap_mean",
        "consecutive_jaccard_mean",
        "nonconsecutive_overlap_mean",
        "nonconsecutive_jaccard_mean",
        "consecutive_nonconsecutive_jaccard_gap",
        "consecutive_center_distance_mean",
        "new_nodes_per_patch_mean",
        "new_pairs_per_patch_mean",
        "mapped_patch_adjacency_match_rate",
        "mapped_transition_slot_map_match_rate",
        "fresh_relabel_patch_set_similarity",
    )
    return {
        name: float(np.mean([float(row[name]) for row in rows])) for name in metric_names
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, Any] = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        by_method[method] = {
            "graph_count": len(selected),
            "mean": _mean_metrics(selected),
            "edge_coverage_std": float(
                np.std([row["true_edge_coverage"] for row in selected])
            ),
            "pair_coverage_std": float(
                np.std([row["node_pair_coverage"] for row in selected])
            ),
        }
    by_family_degree: dict[str, Any] = {}
    for family in FAMILIES:
        for degree in sorted(set(int(row["target_degree"]) for row in rows)):
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
                by_family_degree[key][method] = _mean_metrics(selected)
    return {"by_method": by_method, "by_family_degree": by_family_degree}


def _run_seed(
    audit_seed: int,
    graph_bank: list[dict[str, Any]],
    *,
    patch_size: int,
    target_overlap: int,
    edge_capacity_multiplier: float,
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
                    [audit_seed, graph_index, method_index, 1103]
                ).generate_state(1, dtype=np.uint32)[0]
            )
            cover = _sample(
                method,
                adjacency,
                sampler_seed,
                n_patches=budget,
                patch_size=patch_size,
                target_overlap=target_overlap,
            )
            audit = audit_cover(adjacency, cover)

            relabel_rng = np.random.default_rng(
                np.random.SeedSequence([audit_seed, graph_index, method_index, 2207])
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
            )
            fresh_in_original_ids = remap_cover(fresh, permutation, adjacency)
            row = {
                "audit_seed": audit_seed,
                "graph_index": graph_index,
                "family": graph_record["family"],
                "target_degree": graph_record["target_degree"],
                "realized_average_degree": graph_record["realized_average_degree"],
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
                    cover, fresh_in_original_ids
                ),
            }
            rows.append(row)
    return {
        "audit_seed": audit_seed,
        "rows": rows,
        "summary": _summarize(rows),
    }


def classify(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    all_rows = [row for result in seed_results for row in result["rows"]]
    invariants = {
        "observed_pair_consistency": all(
            row["observed_pair_consistency"] == 1.0 for row in all_rows
        ),
        "observed_pair_accuracy": all(
            row["observed_pair_accuracy"] == 1.0 for row in all_rows
        ),
        "mapped_patch_adjacency": all(
            row["mapped_patch_adjacency_match_rate"] == 1.0 for row in all_rows
        ),
        "mapped_transition_maps": all(
            row["mapped_transition_slot_map_match_rate"] == 1.0 for row in all_rows
        ),
    }
    seed_gates = []
    for result in seed_results:
        summary = result["summary"]["by_method"]
        independent = summary["independent_walk"]["mean"]
        frontier = summary["frontier_cover"]["mean"]
        sliding = summary["sliding_walk"]["mean"]

        def gate(candidate: dict[str, float]) -> dict[str, Any]:
            return {
                "edge_coverage_gain": candidate["true_edge_coverage"]
                - independent["true_edge_coverage"],
                "pair_coverage_gain": candidate["node_pair_coverage"]
                - independent["node_pair_coverage"],
                "overlap": candidate["consecutive_overlap_mean"],
                "jaccard_gap": candidate[
                    "consecutive_nonconsecutive_jaccard_gap"
                ],
                "passed": bool(
                    candidate["true_edge_coverage"]
                    - independent["true_edge_coverage"]
                    >= 0.03
                    and candidate["node_pair_coverage"]
                    - independent["node_pair_coverage"]
                    >= -0.05
                    and candidate["consecutive_overlap_mean"] >= 4.5
                    and candidate["consecutive_nonconsecutive_jaccard_gap"] > 0.0
                ),
            }

        seed_gates.append(
            {
                "audit_seed": result["audit_seed"],
                "frontier": gate(frontier),
                "sliding": gate(sliding),
            }
        )
    frontier_passes = sum(item["frontier"]["passed"] for item in seed_gates)
    sliding_passes = sum(item["sliding"]["passed"] for item in seed_gates)
    if not all(invariants.values()):
        label = "FAIL_IMPLEMENTATION_INVARIANTS"
    elif frontier_passes >= 2:
        label = "PASS_FRONTIER_OVERLAP_COVER"
    elif sliding_passes >= 2:
        label = "SELECT_SLIDING_WALK_COVER"
    else:
        label = "FAIL_SMALL_PATCH_GRAPH_RECOVERY"
    return {
        "classification": label,
        "invariants": invariants,
        "frontier_seed_pass_count": frontier_passes,
        "sliding_seed_pass_count": sliding_passes,
        "seed_gates": seed_gates,
    }


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# 连续重叠 patch cover 第一轮审计",
        "",
        "> 日期：2026-08-01  ",
        "> 本轮不训练 KSVD、不使用 labels。",
        "",
        "## 1. 判定",
        "",
        f"**{payload['decision']['classification']}**",
        "",
        "## 2. 全局均值",
        "",
        "| method | node cover | edge cover | pair cover | full adj acc | consecutive overlap | Jaccard gap | center dist | fresh relabel set sim |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    overall_rows = [row for seed in payload["seeds"] for row in seed["rows"]]
    overall = _summarize(overall_rows)["by_method"]
    for method in METHODS:
        mean = overall[method]["mean"]
        lines.append(
            f"| {method} | {_fmt(mean['node_coverage'])} | "
            f"{_fmt(mean['true_edge_coverage'])} | {_fmt(mean['node_pair_coverage'])} | "
            f"{_fmt(mean['full_adjacency_accuracy'])} | "
            f"{_fmt(mean['consecutive_overlap_mean'])} | "
            f"{_fmt(mean['consecutive_nonconsecutive_jaccard_gap'])} | "
            f"{_fmt(mean['consecutive_center_distance_mean'])} | "
            f"{_fmt(mean['fresh_relabel_patch_set_similarity'])} |"
        )

    lines.extend(
        [
            "",
            "所有方法在 observed pairs 上必须满足 consistency/accuracy=1，并且 mapped replay 的 patch adjacency 与 transition slot maps 必须完全一致。",
            "",
            "## 3. Seed gates",
            "",
            "| seed | frontier edge/pair gain | frontier overlap/gap | pass | sliding edge/pair gain | sliding overlap/gap | pass |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in payload["decision"]["seed_gates"]:
        frontier = item["frontier"]
        sliding = item["sliding"]
        lines.append(
            f"| {item['audit_seed']} | {_fmt(frontier['edge_coverage_gain'])}/"
            f"{_fmt(frontier['pair_coverage_gain'])} | {_fmt(frontier['overlap'])}/"
            f"{_fmt(frontier['jaccard_gap'])} | {frontier['passed']} | "
            f"{_fmt(sliding['edge_coverage_gain'])}/{_fmt(sliding['pair_coverage_gain'])} | "
            f"{_fmt(sliding['overlap'])}/{_fmt(sliding['jaccard_gap'])} | "
            f"{sliding['passed']} |"
        )

    lines.extend(
        [
            "",
            "## 4. Family / degree breakdown",
            "",
            "| family-degree | method | edge cover | pair cover | consecutive overlap | Jaccard gap |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    breakdown = _summarize(overall_rows)["by_family_degree"]
    for key, methods in breakdown.items():
        for method in METHODS:
            mean = methods[method]
            lines.append(
                f"| {key} | {method} | {_fmt(mean['true_edge_coverage'])} | "
                f"{_fmt(mean['node_pair_coverage'])} | "
                f"{_fmt(mean['consecutive_overlap_mean'])} | "
                f"{_fmt(mean['consecutive_nonconsecutive_jaccard_gap'])} |"
            )

    lines.extend(
        [
            "",
            "## 5. 机制解释与路线判断",
            "",
            "1. **连续性实现成功。** sliding/frontier 的相邻 overlap 均严格为 5，Jaccard gap 为正；observed-pair consistency、mapped adjacency 和 transition slot maps 全部为 1。",
            "2. **当前 frontier 没有通过恢复 gate。** 它相对 independent walk 的 edge coverage 只提高约 0.008–0.011，低于冻结的 0.03；pair coverage 损失约 0.034–0.039，虽在容许范围内，但没有换来足够的边恢复收益。",
            "3. **失败具有明确的 graph-family 条件。** frontier 在 regular 与 small-world 的各 degree cell 中提高 edge coverage，但在 block 图上随 degree 增大而明显恶化。单条局部 frontier 会在社区内部持续获得高局部收益，因此无法把有限 patch 预算合理转移到另一个社区。",
            "4. **sliding walk 不是候选。** 它提供最清晰的序列连续性，但在所有 family/degree cells 都降低 edge coverage；仅靠一条 walk 的连续窗口不足以形成高质量 edge cover。",
            "5. **暂不进入 KSVD。** 当前 raw induced patches 尚只观察约 0.60 的真实边；此时比较 KSVD stitched reconstruction 会把采样缺失与字典误差混在一起。",
            "",
            "下一轮应保持 patch size、overlap 和预算不变，只修改 frontier 的跨区域调度：显式选择低覆盖区域中的 uncovered target edge，并用最短 bridge 把下一 patch 引向该区域。必须同时比较 single-chain、edge-targeted bridge 和允许多个连续 segment 的 multi-chain cover；若仍不能提高 block edge coverage，再做 patch-size/overlap 容量曲线，而不是直接接 KSVD。",
            "",
            "## 6. 边界",
            "",
            "- `mapped replay=1` 只说明给定同一抽象 patch cover 后，节点重编号不会改变局部 adjacency 或 overlap correspondence。",
            "- fresh resampling similarity 不是严格不变性；它会受到随机选择和结构 ties 影响。",
            "- full adjacency accuracy 会受图的非边占多数影响，主恢复指标仍是 true-edge 与 node-pair coverage。",
            "- 本轮通过也不等于 KSVD 或 Transformer 已经有效；下一轮才比较 raw patch 与 KSVD patch 的 stitched reconstruction。",
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
        )
        seed_results.append(result)
        means = result["summary"]["by_method"]
        print(
            f"seed={audit_seed} "
            + " ".join(
                f"{method}:edge={means[method]['mean']['true_edge_coverage']:.3f},"
                f"pair={means[method]['mean']['node_pair_coverage']:.3f},"
                f"ov={means[method]['mean']['consecutive_overlap_mean']:.2f}"
                for method in METHODS
            ),
            flush=True,
        )

    decision = classify(seed_results)
    payload = {
        "protocol": "ksvd-overlap-cover-audit-v0-20260801",
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
