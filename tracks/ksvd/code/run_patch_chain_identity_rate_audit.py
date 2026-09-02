"""Audit node-label dependence and complete per-graph payload lower bounds."""
from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .marginal_candidate_cover import sample_marginal_candidate_cover
from .overlap_cover import (
    PatchCover,
    audit_cover,
    cover_set_similarity,
    cover_vectors,
    mapped_replay_relabel_invariance,
    patch_budget,
    remap_cover,
)
from .run_overlap_cover_audit import generate_graph


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JSON = ROOT / "tracks/ksvd/results/from_scratch/patch_chain_identity_rate_audit_20260802.json"
DEFAULT_REPORT = ROOT / "tracks/ksvd/results/from_scratch/PATCH_CHAIN_IDENTITY_RATE_AUDIT_20260802.md"
PROTOCOL = "tracks/ksvd/docs/KSVD_PATCH_CHAIN_IDENTITY_RATE_PROTOCOL_20260802.md"
FAMILIES = ("regular", "small_world", "block")


def _log2_permutation(n: int, k: int) -> float:
    if not 0 <= k <= n:
        raise ValueError("expected 0 <= k <= n")
    return float((math.lgamma(n + 1) - math.lgamma(n - k + 1)) / math.log(2.0))


def _log2_combination(n: int, k: int) -> float:
    if not 0 <= k <= n:
        raise ValueError("expected 0 <= k <= n")
    return float(
        (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))
        / math.log(2.0)
    )


def identity_bit_bounds(
    *, n_nodes: int, patch_size: int, overlap: int, patch_count: int
) -> dict[str, float]:
    """Optimistic chain-aware lower bound and fixed-width membership upper bound."""
    if patch_count < 1:
        raise ValueError("patch_count must be positive")
    new_count = patch_size - overlap
    first = _log2_permutation(n_nodes, patch_size)
    transition = _log2_combination(patch_size, overlap) + _log2_permutation(
        n_nodes - overlap, new_count
    )
    lower = math.ceil(first + (patch_count - 1) * transition)
    fixed = patch_count * patch_size * math.ceil(math.log2(n_nodes))
    return {
        "chain_aware_lower_bits": float(lower),
        "fixed_width_ordered_membership_bits": float(fixed),
        "first_patch_log2_permutation": float(first),
        "per_transition_log2_cost": float(transition),
    }


def _observed_pair_count(cover: PatchCover) -> int:
    observed: set[tuple[int, int]] = set()
    for patch in cover.patches:
        observed.update(tuple(sorted(pair)) for pair in combinations(patch.node_ids, 2))
    return len(observed)


def _ordered_chain_match(left: PatchCover, right: PatchCover) -> bool:
    return len(left.patches) == len(right.patches) and all(
        left_patch.node_ids == right_patch.node_ids
        for left_patch, right_patch in zip(left.patches, right.patches)
    )


def _mean_transition_match(left: PatchCover, right: PatchCover) -> float:
    if len(left.transitions) != len(right.transitions):
        return 0.0
    if not left.transitions:
        return 1.0
    return float(
        np.mean(
            [
                left_map.left_to_right_slots == right_map.left_to_right_slots
                for left_map, right_map in zip(left.transitions, right.transitions)
            ]
        )
    )


def _mean_local_vector_row_match(left: PatchCover, right: PatchCover) -> float:
    left_vectors = cover_vectors(left)
    right_vectors = cover_vectors(right)
    if left_vectors.shape != right_vectors.shape:
        return 0.0
    return float(np.mean(np.all(left_vectors == right_vectors, axis=1)))


def _mean(rows: Sequence[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def classify(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = tuple(rows)
    if not rows:
        raise ValueError("rows cannot be empty")
    summary_keys = (
        "mapped_patch_adjacency_match_rate",
        "mapped_transition_slot_map_match_rate",
        "resampled_exact_ordered_chain_match",
        "resampled_patch_set_jaccard",
        "resampled_local_vector_row_match",
        "resampled_transition_map_match",
        "absolute_edge_coverage_delta",
        "absolute_pair_coverage_delta",
        "absolute_raw_full_rmse_delta",
        "identity_chain_lower_bits",
        "identity_fixed_width_bits",
        "raw_naive_lower_bits",
        "raw_unique_lower_bits",
        "direct_enumerative_exact_bits",
        "ksvd_break_even_bits_per_coefficient",
    )
    means = {key: _mean(rows, key) for key in summary_keys}
    strict_checks = {
        "exact_chain_at_least_099": means["resampled_exact_ordered_chain_match"] >= 0.99,
        "local_vectors_at_least_099": means["resampled_local_vector_row_match"] >= 0.99,
        "transition_maps_at_least_099": means["resampled_transition_map_match"] >= 0.99,
    }
    strict_pass = all(strict_checks.values())
    stability_checks = {
        "edge_delta_at_most_002": means["absolute_edge_coverage_delta"] <= 0.02,
        "pair_delta_at_most_002": means["absolute_pair_coverage_delta"] <= 0.02,
        "raw_rmse_delta_at_most_002": means["absolute_raw_full_rmse_delta"] <= 0.02,
    }
    raw_bit_pass = means["raw_unique_lower_bits"] < float(rows[0]["direct_bitset_bits"])
    coefficient_bit_pass = means["ksvd_break_even_bits_per_coefficient"] >= 8.0
    replay_pass = (
        means["mapped_patch_adjacency_match_rate"] == 1.0
        and means["mapped_transition_slot_map_match_rate"] == 1.0
    )
    invariants = all(
        bool(row["base_invariants"]) and bool(row["relabeled_invariants"])
        for row in rows
    )
    if strict_pass and raw_bit_pass and coefficient_bit_pass and replay_pass and invariants:
        classification = "PATCH_CHAIN_IDENTITY_RATE_SUPPORTED"
    else:
        classification = "REVISE_PATCH_CHAIN_REPRESENTATION_CLAIM"
    by_degree = {}
    if all("target_degree" in row for row in rows):
        for degree in sorted({int(row["target_degree"]) for row in rows}):
            degree_rows = [row for row in rows if int(row["target_degree"]) == degree]
            by_degree[str(degree)] = {
                "patch_count": _mean(degree_rows, "patch_count"),
                "identity_chain_estimate_bits": _mean(
                    degree_rows, "identity_chain_lower_bits"
                ),
                "raw_unique_estimate_bits": _mean(
                    degree_rows, "raw_unique_lower_bits"
                ),
                "raw_unique_over_bitset_ratio": _mean(
                    [
                        {
                            "ratio": float(row["raw_unique_lower_bits"])
                            / float(row["direct_bitset_bits"])
                        }
                        for row in degree_rows
                    ],
                    "ratio",
                ),
                "ksvd_break_even_bits_per_coefficient": _mean(
                    degree_rows, "ksvd_break_even_bits_per_coefficient"
                ),
            }
    return {
        "classification": classification,
        "strict_relabel_classification": (
            "STRICT_RELABEL_EQUIVARIANCE_PASS"
            if strict_pass else "SAMPLER_NOT_STRICTLY_RELABEL_EQUIVARIANT"
        ),
        "raw_rate_classification": (
            "RAW_CHAIN_MAY_BE_BIT_COMPETITIVE"
            if raw_bit_pass else "RAW_CHAIN_NOT_BIT_COMPETITIVE_WITH_ADJACENCY"
        ),
        "ksvd_rate_classification": (
            "KSVD_HAS_AT_LEAST_8_COEFFICIENT_BITS"
            if coefficient_bit_pass
            else "KSVD_REQUIRES_UNVALIDATED_AGGRESSIVE_QUANTIZATION"
        ),
        "mapped_replay_gate": replay_pass,
        "invariant_gate": invariants,
        "strict_checks": strict_checks,
        "coverage_stability_checks": stability_checks,
        "coverage_stability_gate": all(stability_checks.values()),
        "raw_bit_gate": raw_bit_pass,
        "coefficient_bit_gate": coefficient_bit_pass,
        "means": means,
        "by_degree": by_degree,
        "raw_unique_over_bitset_ratio": (
            means["raw_unique_lower_bits"] / float(rows[0]["direct_bitset_bits"])
        ),
        "raw_unique_not_smaller_graph_fraction": float(
            np.mean(
                [
                    float(row["raw_unique_lower_bits"])
                    >= float(row["direct_bitset_bits"])
                    for row in rows
                ]
            )
        ),
    }


def render_report(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    means = decision["means"]
    config = payload["config"]
    lines = [
        "# Patch-chain 节点身份与 bit-rate 审计",
        "",
        "> 日期：2026-08-02  ",
        f"> 协议：`{payload['protocol']}`  ",
        f"> 总判定：`{decision['classification']}`",
        "",
        "## 1. Relabel-resampling",
        "",
        f"图数：{config['graph_count']}；每图 cover seeds：{len(config['cover_seeds'])}。",
        "",
        "| metric | mean |",
        "|---|---:|",
        f"| mapped replay patch adjacency match | {means['mapped_patch_adjacency_match_rate']:.4f} |",
        f"| mapped replay transition match | {means['mapped_transition_slot_map_match_rate']:.4f} |",
        f"| resampled exact ordered-chain match | {means['resampled_exact_ordered_chain_match']:.4f} |",
        f"| resampled patch-set Jaccard | {means['resampled_patch_set_jaccard']:.4f} |",
        f"| resampled local-vector row match | {means['resampled_local_vector_row_match']:.4f} |",
        f"| resampled transition-map match | {means['resampled_transition_map_match']:.4f} |",
        f"| absolute edge-coverage delta | {means['absolute_edge_coverage_delta']:.4f} |",
        f"| absolute pair-coverage delta | {means['absolute_pair_coverage_delta']:.4f} |",
        f"| absolute RAW full-RMSE delta | {means['absolute_raw_full_rmse_delta']:.4f} |",
        "",
        f"严格重编号判定：`{decision['strict_relabel_classification']}`。",
        "",
        "Mapped replay 为 1 只说明一条已经生成的抽象 cover 可以随 node permutation 一起搬运。重新运行 sampler 后，chain/slot substrate 是否复现是更强、也更相关的检验。",
        "",
        "## 2. 完整 payload 下界",
        "",
        "| payload | mean bits / graph |",
        "|---|---:|",
        f"| direct upper-triangle bitset | {config['direct_bitset_bits']:.1f} |",
        f"| direct enumerative exact code | {means['direct_enumerative_exact_bits']:.1f} |",
        f"| patch identity optimistic fixed-field estimate | {means['identity_chain_lower_bits']:.1f} |",
        f"| patch identity fixed-width ordered IDs | {means['identity_fixed_width_bits']:.1f} |",
        f"| RAW local slots + identity estimate | {means['raw_naive_lower_bits']:.1f} |",
        f"| RAW unique observed pairs + identity estimate | {means['raw_unique_lower_bits']:.1f} |",
        "",
        f"RAW unique estimate / direct bitset：`{decision['raw_unique_over_bitset_ratio']:.3f}`；图级不更小比例：`{decision['raw_unique_not_smaller_graph_fraction']:.3f}`。",
        "",
        f"RAW 判定：`{decision['raw_rate_classification']}`。",
        "",
        "按目标度数拆分：",
        "",
        "| degree | patches | identity bits | RAW unique bits | / bitset | KSVD coefficient break-even bits |",
        "|---:|---:|---:|---:|---:|---:|",
        *[
            (
                f"| {degree} | {values['patch_count']:.2f} | "
                f"{values['identity_chain_estimate_bits']:.1f} | "
                f"{values['raw_unique_estimate_bits']:.1f} | "
                f"{values['raw_unique_over_bitset_ratio']:.3f} | "
                f"{values['ksvd_break_even_bits_per_coefficient']:.2f} |"
            )
            for degree, values in decision["by_degree"].items()
        ],
        "",
        "`degree=15` 在该乐观显式编码下接近 bitset break-even；随着密度和 matched patch budget 增加，identity sidecar 很快成为主导成本。",
        "",
        "## 3. KSVD break-even",
        "",
        f"在先支付 optimistic identity estimate 和每个 nonzero 的 5-bit atom index 后，为了不超过 1225-bit adjacency，每个约 `3×patch_count` coefficient 平均只剩 `{means['ksvd_break_even_bits_per_coefficient']:.2f}` bits。",
        "",
        f"判定：`{decision['ksvd_rate_classification']}`。这个预算尚未包含 dictionary、quantizer、completion model、framing 或误差校验，因此不是已经实现的 codec rate。",
        "",
        "## 4. 结论",
        "",
        "当前 Beam8 的 coverage/reconstruction 对重编号扰动较稳定，但具体 chain、local slots 和 transition maps 不是严格 relabel-equivariant。另一方面，在本协议对 patch chain 有利的显式身份编码下，平均 RAW payload 仍不能胜过直接 bit-packed adjacency。该结果否定当前表示已经 bit-competitive 的主张，不构成对所有可能联合熵编码的不可行性证明。",
        "",
        "因此应把现有方法限定为：",
        "",
        "> 完整已知图上的结构引导 patch extractor 与可选局部 sparse representation。",
        "",
        "不能继续无条件表述为编号无关图表示，或已经证明优于直接邻接矩阵的 bit-level compression。后续必须拆成 labeled rate-distortion 与 ID-free structural representation 两条不同路线。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-bank-seed", type=int, default=820201)
    parser.add_argument("--cover-seeds", type=int, nargs="+", default=[920101, 920102, 920103])
    parser.add_argument("--n-nodes", type=int, default=50)
    parser.add_argument("--degrees", type=int, nargs="+", default=[15, 20, 25])
    parser.add_argument("--graphs-per-cell", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=10)
    parser.add_argument("--target-overlap", type=int, default=3)
    parser.add_argument("--multiplier", type=float, default=1.5)
    parser.add_argument("--retained-beam", type=int, default=8)
    parser.add_argument("--candidate-restarts", type=int, default=1)
    parser.add_argument("--n-atoms", type=int, default=24)
    parser.add_argument("--sparsity", type=int, default=3)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    graph_count = len(FAMILIES) * len(args.degrees) * args.graphs_per_cell
    graph_sequences = np.random.SeedSequence(args.graph_bank_seed).spawn(graph_count)
    pair_count = args.n_nodes * (args.n_nodes - 1) // 2
    bitset_bits = pair_count
    atom_index_bits = math.ceil(math.log2(args.n_atoms))
    rows: list[dict[str, Any]] = []
    graph_index = 0
    for family in FAMILIES:
        for degree in args.degrees:
            for _replicate in range(args.graphs_per_cell):
                graph_seed = int(
                    graph_sequences[graph_index].generate_state(1, dtype=np.uint32)[0]
                )
                adjacency = generate_graph(family, args.n_nodes, degree, graph_seed)
                permutation = np.random.default_rng(
                    np.random.SeedSequence([args.graph_bank_seed, graph_index, 991]).generate_state(
                        1, dtype=np.uint32
                    )[0]
                ).permutation(args.n_nodes)
                relabeled_adjacency = adjacency[np.ix_(permutation, permutation)]
                budget = patch_budget(
                    adjacency,
                    patch_size=args.patch_size,
                    target_overlap=args.target_overlap,
                    edge_capacity_multiplier=args.multiplier,
                )
                for cover_seed in args.cover_seeds:
                    seed = int(
                        np.random.SeedSequence(
                            [cover_seed, graph_index]
                        ).generate_state(1, dtype=np.uint32)[0]
                    )
                    base = sample_marginal_candidate_cover(
                        adjacency,
                        np.random.default_rng(seed),
                        n_patches=budget,
                        patch_size=args.patch_size,
                        target_overlap=args.target_overlap,
                        retained_beam=args.retained_beam,
                        candidate_restarts=args.candidate_restarts,
                    )
                    relabeled = sample_marginal_candidate_cover(
                        relabeled_adjacency,
                        np.random.default_rng(seed),
                        n_patches=budget,
                        patch_size=args.patch_size,
                        target_overlap=args.target_overlap,
                        retained_beam=args.retained_beam,
                        candidate_restarts=args.candidate_restarts,
                    )
                    mapped = remap_cover(relabeled, permutation, adjacency)
                    base_audit = audit_cover(adjacency, base)
                    mapped_audit = audit_cover(adjacency, mapped)
                    replay = mapped_replay_relabel_invariance(
                        adjacency, base, permutation
                    )
                    identity = identity_bit_bounds(
                        n_nodes=args.n_nodes,
                        patch_size=args.patch_size,
                        overlap=args.target_overlap,
                        patch_count=len(base.patches),
                    )
                    observed_pair_count = _observed_pair_count(base)
                    local_pair_slots = len(base.patches) * (
                        args.patch_size * (args.patch_size - 1) // 2
                    )
                    edge_count = int(adjacency.sum() // 2)
                    enumerative_bits = math.ceil(_log2_combination(pair_count, edge_count)) + math.ceil(
                        math.log2(pair_count + 1)
                    )
                    nonzeros = args.sparsity * len(base.patches)
                    remaining = (
                        bitset_bits
                        - identity["chain_aware_lower_bits"]
                        - nonzeros * atom_index_bits
                    )
                    base_raw_rmse = math.sqrt(max(0.0, 1.0 - base_audit["full_adjacency_accuracy"]))
                    mapped_raw_rmse = math.sqrt(
                        max(0.0, 1.0 - mapped_audit["full_adjacency_accuracy"])
                    )
                    base_invariants = (
                        base_audit["patch_connected_rate"] == 1.0
                        and base_audit["continuous_transition_fraction"] == 1.0
                        and all(
                            overlap == args.target_overlap
                            for overlap in base_audit["transition_overlap_exact"]
                        )
                    )
                    mapped_invariants = (
                        mapped_audit["patch_connected_rate"] == 1.0
                        and mapped_audit["continuous_transition_fraction"] == 1.0
                        and all(
                            overlap == args.target_overlap
                            for overlap in mapped_audit["transition_overlap_exact"]
                        )
                    )
                    rows.append(
                        {
                            "graph_index": graph_index,
                            "family": family,
                            "target_degree": degree,
                            "cover_seed": int(cover_seed),
                            "patch_count": len(base.patches),
                            "base_invariants": bool(base_invariants),
                            "relabeled_invariants": bool(mapped_invariants),
                            "mapped_patch_adjacency_match_rate": replay[
                                "patch_adjacency_match_rate"
                            ],
                            "mapped_transition_slot_map_match_rate": replay[
                                "transition_slot_map_match_rate"
                            ],
                            "resampled_exact_ordered_chain_match": float(
                                _ordered_chain_match(base, mapped)
                            ),
                            "resampled_patch_set_jaccard": cover_set_similarity(base, mapped),
                            "resampled_local_vector_row_match": _mean_local_vector_row_match(
                                base, mapped
                            ),
                            "resampled_transition_map_match": _mean_transition_match(
                                base, mapped
                            ),
                            "absolute_edge_coverage_delta": abs(
                                base_audit["true_edge_coverage"]
                                - mapped_audit["true_edge_coverage"]
                            ),
                            "absolute_pair_coverage_delta": abs(
                                base_audit["node_pair_coverage"]
                                - mapped_audit["node_pair_coverage"]
                            ),
                            "absolute_raw_full_rmse_delta": abs(
                                base_raw_rmse - mapped_raw_rmse
                            ),
                            "direct_bitset_bits": bitset_bits,
                            "direct_enumerative_exact_bits": enumerative_bits,
                            "identity_chain_lower_bits": identity[
                                "chain_aware_lower_bits"
                            ],
                            "identity_fixed_width_bits": identity[
                                "fixed_width_ordered_membership_bits"
                            ],
                            "local_pair_slots": local_pair_slots,
                            "observed_pair_count": observed_pair_count,
                            "raw_naive_lower_bits": identity[
                                "chain_aware_lower_bits"
                            ]
                            + local_pair_slots,
                            "raw_unique_lower_bits": identity[
                                "chain_aware_lower_bits"
                            ]
                            + observed_pair_count,
                            "ksvd_nonzero_count": nonzeros,
                            "ksvd_atom_index_bits": nonzeros * atom_index_bits,
                            "ksvd_break_even_bits_per_coefficient": remaining
                            / max(nonzeros, 1),
                        }
                    )
                graph_index += 1

    decision = classify(rows)
    payload = {
        "protocol": PROTOCOL,
        "config": {
            "graph_bank_seed": args.graph_bank_seed,
            "cover_seeds": args.cover_seeds,
            "families": list(FAMILIES),
            "degrees": args.degrees,
            "graphs_per_cell": args.graphs_per_cell,
            "graph_count": graph_count,
            "n_nodes": args.n_nodes,
            "patch_size": args.patch_size,
            "target_overlap": args.target_overlap,
            "multiplier": args.multiplier,
            "retained_beam": args.retained_beam,
            "candidate_restarts": args.candidate_restarts,
            "n_atoms": args.n_atoms,
            "sparsity": args.sparsity,
            "direct_bitset_bits": bitset_bits,
            "labels_used": False,
        },
        "rows": rows,
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
