"""Run U0-R before selecting an adjacency representation for KSVD."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .from_scratch_unplanted_representation import (
    WalkPatch,
    adjacency_to_upper_vector,
    audit_patch_collection,
    generate_rewired_graph,
    is_connected,
    relabel_adjacency_and_order,
    sample_graph_walk_patches,
)


DEFAULT_JSON = Path("tracks/ksvd/results/from_scratch/u0r_adjacency_audit_20260731.json")
DEFAULT_REPORT = Path("tracks/ksvd/results/from_scratch/U0R_ADJACENCY_AUDIT_20260731.md")
DEFAULT_SEEDS = (732001, 732002, 732003)


def _generate_seed_audit(
    master_seed: int,
    *,
    graphs_per_regime: int,
    patches_per_graph: int,
    pair_count: int,
    root_permutation_trials: int,
) -> dict[str, Any]:
    total_graphs = 2 * graphs_per_regime
    graph_sequences = np.random.SeedSequence(master_seed).spawn(total_graphs)
    all_patches: list[WalkPatch] = []
    generator_failures = 0
    walk_relabel_matches = 0
    walk_relabel_trials = 0
    requested_swaps: list[int] = []
    attempted_swaps: list[int] = []
    trace_lengths: list[int] = []
    induced_edge_counts: list[int] = []

    graph_index = 0
    for regime in ("LOW", "HIGH"):
        low, high = (20, 40) if regime == "LOW" else (60, 80)
        for _ in range(graphs_per_regime):
            generation_sequence, patch_sequence, audit_sequence = graph_sequences[graph_index].spawn(3)
            generation_rng = np.random.default_rng(generation_sequence)
            swap_count = int(generation_rng.integers(low, high + 1))
            graph_seed = int(generation_rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
            graph = generate_rewired_graph(seed=graph_seed, n_accepted_swaps=swap_count)
            adjacency = graph.adjacency
            requested_swaps.append(swap_count)
            attempted_swaps.append(graph.attempted_swaps)
            valid = (
                graph.accepted_swaps == swap_count
                and adjacency.shape == (60, 60)
                and int(adjacency.sum() // 2) == 120
                and bool(np.all(adjacency.sum(axis=1) == 4))
                and is_connected(adjacency)
            )
            generator_failures += int(not valid)

            patch_rng = np.random.default_rng(patch_sequence)
            patches = sample_graph_walk_patches(
                adjacency,
                patch_rng,
                n_patches=patches_per_graph,
                patch_size=6,
            )
            all_patches.extend(patches)
            trace_lengths.extend(len(patch.walk_trace) for patch in patches)
            induced_edge_counts.extend(int(patch.adjacency_walk_order.sum() // 2) for patch in patches)

            audit_rng = np.random.default_rng(audit_sequence)
            global_permutation = audit_rng.permutation(adjacency.shape[0])
            for patch in patches:
                relabeled, mapped_order = relabel_adjacency_and_order(
                    adjacency,
                    patch.node_ids,
                    global_permutation,
                )
                mapped_induced = relabeled[np.ix_(mapped_order, mapped_order)]
                mapped_vector = adjacency_to_upper_vector(mapped_induced)
                walk_relabel_trials += 1
                walk_relabel_matches += int(np.array_equal(mapped_vector, patch.walk_order_vector))
            graph_index += 1

    collection_rng = np.random.default_rng(np.random.SeedSequence([master_seed, 99173]))
    representation = audit_patch_collection(
        all_patches,
        collection_rng,
        pair_count=pair_count,
        root_permutation_trials=root_permutation_trials,
    )
    representation["walk_mapped_trace_global_relabel_invariance_rate"] = float(
        walk_relabel_matches / walk_relabel_trials
    )
    representation["walk_mapped_trace_global_relabel_trial_count"] = int(walk_relabel_trials)
    return {
        "master_seed": int(master_seed),
        "graph_count": total_graphs,
        "generator_invariant_failure_count": int(generator_failures),
        "requested_swaps": {
            "minimum": int(min(requested_swaps)),
            "maximum": int(max(requested_swaps)),
            "mean": float(np.mean(requested_swaps)),
        },
        "swap_attempts": {
            "minimum": int(min(attempted_swaps)),
            "maximum": int(max(attempted_swaps)),
            "mean": float(np.mean(attempted_swaps)),
        },
        "walk_trace_length": {
            "mean": float(np.mean(trace_lengths)),
            "p95": float(np.quantile(trace_lengths, 0.95)),
            "maximum": int(max(trace_lengths)),
        },
        "induced_patch_edge_count": {
            "mean": float(np.mean(induced_edge_counts)),
            "minimum": int(min(induced_edge_counts)),
            "maximum": int(max(induced_edge_counts)),
        },
        "representation": representation,
    }


def classify(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    canonical_checks = []
    walk_checks = []
    for result in seed_results:
        representation = result["representation"]
        flip = representation["one_edge_flip"]
        pairwise = representation["pairwise"]
        canonical_checks.append({
            "generator_valid": result["generator_invariant_failure_count"] == 0,
            "permutation_invariance": representation["canonical_permutation_invariance_rate"] == 1.0,
            "injective_on_sampled_pairs": pairwise["canonical_injectivity_disagreement_count"] == 0,
            "amplification_rate": flip["canonical_amplification_rate_gt_1"] <= 0.20,
            "severe_jump_rate": flip["canonical_severe_jump_rate_ge_4"] <= 0.05,
            "metric_spearman": pairwise["canonical"]["spearman"] >= 0.80,
        })
        walk_checks.append({
            "generator_valid": result["generator_invariant_failure_count"] == 0,
            "mapped_trace_invariance": representation["walk_mapped_trace_global_relabel_invariance_rate"] == 1.0,
            "one_edge_continuity": flip["walk_distance_all_one"],
            "metric_not_much_worse": pairwise["walk_order"]["spearman"] >= pairwise["canonical"]["spearman"] - 0.05,
        })
    canonical_pass = all(all(check.values()) for check in canonical_checks)
    walk_geometry_eligible = all(all(check.values()) for check in walk_checks)
    if canonical_pass:
        label = "SELECT_ROOTED_CANONICAL_ADJACENCY"
        interpretation = (
            "Rooted canonical adjacency passes invariance, local-continuity, and exact-distance "
            "fidelity gates on every audit seed. It can be the primary U0 representation."
        )
    elif walk_geometry_eligible:
        label = "SELECT_WALK_ORDER_PENDING_U0P"
        interpretation = (
            "Canonical adjacency is invariant but its linear geometry fails at least one gate. "
            "Walk-order adjacency preserves sampler-slot semantics and is eligible, pending the "
            "separate U0-P signal-exposure gate."
        )
    else:
        label = "REJECT_BOTH_ADJACENCY_GEOMETRIES"
        interpretation = (
            "Canonical geometry fails and walk-order geometry is materially worse under exact "
            "rooted graph distance. Do not run adjacency KSVD; move to fixed-semantic invariant statistics."
        )
    return {
        "classification": label,
        "canonical_passed": bool(canonical_pass),
        "walk_geometry_eligible": bool(walk_geometry_eligible),
        "canonical_checks_by_seed": canonical_checks,
        "walk_checks_by_seed": walk_checks,
        "interpretation": interpretation,
    }


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# U0-R 邻接表示审计结果",
        "",
        "> 日期：2026-07-31",
        ">",
        "> 本实验不训练 KSVD；它检查 exact canonicalization 是否不仅同构不变，而且具有可供线性字典学习使用的结构几何。",
        "",
        "## 1. 总判定",
        "",
        f"**{payload['decision']['classification']}**",
        "",
        payload["decision"]["interpretation"],
        "",
        "## 2. 三个 audit seeds",
        "",
        "| seed | patches | CAN invariance | CAN one-flip mean | CAN amplify >1 | CAN severe >=4 | CAN GED Spearman | WALK GED Spearman | WALK relabel invariance |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["seeds"]:
        representation = result["representation"]
        flip = representation["one_edge_flip"]
        pairwise = representation["pairwise"]
        lines.append(
            f"| {result['master_seed']} | {representation['patch_count']} | "
            f"{representation['canonical_permutation_invariance_rate']:.4f} | "
            f"{flip['canonical_mean_distance']:.4f} | "
            f"{flip['canonical_amplification_rate_gt_1']:.4f} | "
            f"{flip['canonical_severe_jump_rate_ge_4']:.4f} | "
            f"{pairwise['canonical']['spearman']:.4f} | "
            f"{pairwise['walk_order']['spearman']:.4f} | "
            f"{representation['walk_mapped_trace_global_relabel_invariance_rate']:.4f} |"
        )
    lines.extend([
        "",
        "## 3. Generator / sampler sanity",
        "",
        "| seed | graph failures | swap attempts mean | trace length mean/p95/max | induced edges mean/min/max |",
        "|---:|---:|---:|---:|---:|",
    ])
    for result in payload["seeds"]:
        trace = result["walk_trace_length"]
        edges = result["induced_patch_edge_count"]
        lines.append(
            f"| {result['master_seed']} | {result['generator_invariant_failure_count']} | "
            f"{result['swap_attempts']['mean']:.2f} | "
            f"{trace['mean']:.2f}/{trace['p95']:.2f}/{trace['maximum']} | "
            f"{edges['mean']:.2f}/{edges['minimum']}/{edges['maximum']} |"
        )
    lines.extend([
        "",
        "## 4. 如何理解两个表示",
        "",
        "- `R-CAN` 的成功点是：同一个 rooted isomorphism class 只有一个 vector；它不自动保证相邻 graph classes 在 15 维空间中也相邻。",
        "- `R-WALK` 的 coordinate 不是 canonical graph role，而是首次发现顺序。它牺牲同一 subgraph 在不同 walk 下的唯一表示，换取固定 sampler-slot 语义和严格的一条边/一个坐标连续性。",
        "- continuous KSVD atom 不需要本身是合法 adjacency；真正需要避免的是输入坐标因 canonical relabeling 产生大幅、无结构依据的跳变。",
        "",
        "## 5. 下一步",
        "",
    ])
    decision = payload["decision"]["classification"]
    if decision == "SELECT_ROOTED_CANONICAL_ADJACENCY":
        lines.append("以 R-CAN 进入 U0-P；R-WALK 保留为 representation baseline。")
    elif decision == "SELECT_WALK_ORDER_PENDING_U0P":
        lines.append("以 R-WALK 进入 U0-P signal-exposure gate；在 U0-P 通过前不运行 KSVD。R-CAN 仅作为不变性 baseline。")
    else:
        lines.append("停止 adjacency KSVD，下一轮设计 rooted distance/walk-return/degree-distance histogram 等固定语义统计向量。")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--graphs-per-regime", type=int, default=40)
    parser.add_argument("--patches-per-graph", type=int, default=12)
    parser.add_argument("--pair-count", type=int, default=2000)
    parser.add_argument("--root-permutation-trials", type=int, default=20)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    seed_results = []
    for seed in args.seeds:
        result = _generate_seed_audit(
            seed,
            graphs_per_regime=args.graphs_per_regime,
            patches_per_graph=args.patches_per_graph,
            pair_count=args.pair_count,
            root_permutation_trials=args.root_permutation_trials,
        )
        seed_results.append(result)
        representation = result["representation"]
        print(
            f"seed={seed} patches={representation['patch_count']} "
            f"can_amp={representation['one_edge_flip']['canonical_amplification_rate_gt_1']:.4f} "
            f"can_spear={representation['pairwise']['canonical']['spearman']:.4f} "
            f"walk_spear={representation['pairwise']['walk_order']['spearman']:.4f}",
            flush=True,
        )
    decision = classify(seed_results)
    payload = {
        "protocol": "ksvd-u0r-adjacency-representation-audit-v0-20260731",
        "config": {
            "audit_seeds": [int(seed) for seed in args.seeds],
            "graphs_per_regime": args.graphs_per_regime,
            "low_accepted_swaps": [20, 40],
            "high_accepted_swaps": [60, 80],
            "patches_per_graph": args.patches_per_graph,
            "patch_size": 6,
            "pair_count": args.pair_count,
            "root_permutation_trials": args.root_permutation_trials,
            "ksvd_run": False,
        },
        "seeds": seed_results,
        "decision": decision,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.report.write_text(render_report(payload), encoding="utf-8")
    print(f"decision={decision['classification']}")
    print(f"wrote {args.json}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
