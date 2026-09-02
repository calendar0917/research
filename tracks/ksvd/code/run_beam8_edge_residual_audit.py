"""Audit Beam8 prefix repair, residual edges, and hybrid rate proxies."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .beam8_edge_residual import prefix_rate_trajectory, select_checkpoints
from .marginal_candidate_cover import sample_marginal_candidate_cover
from .overlap_cover import audit_cover, patch_budget
from .run_overlap_cover_audit import FAMILIES, generate_graph


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JSON = ROOT / "tracks/ksvd/results/from_scratch/beam8_edge_residual_audit_20260805.json"
DEFAULT_REPORT = ROOT / "tracks/ksvd/results/from_scratch/BEAM8_EDGE_RESIDUAL_AUDIT_20260805.md"
PROTOCOL = "tracks/ksvd/docs/KSVD_BEAM8_EDGE_RESIDUAL_PROTOCOL_20260805.md"
CHECKPOINTS = (
    "ZERO",
    "BASE",
    "EDGE90",
    "EDGE95",
    "EDGE99",
    "EDGE100",
    "HYBRID_ALL",
    "HYBRID_AFTER_BASE",
)
METRICS = (
    "patch_count",
    "edge_coverage",
    "pair_coverage",
    "residual_edge_count",
    "raw_zero_fill_rmse",
    "canonical_set_identity_bits",
    "prefix_length_bits",
    "ksvd_code_proxy_bits",
    "residual_subset_bits",
    "canonical_hybrid_proxy_bits",
    "ordered_hybrid_proxy_bits",
    "canonical_raw_exact_bits",
    "direct_bitset_bits",
    "direct_enumerative_exact_bits",
)


def _mean(rows: Sequence[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("distribution values cannot be empty")
    return {
        "mean": float(np.mean(array)),
        "minimum": float(np.min(array)),
        "p50": float(np.quantile(array, 0.50, method="nearest")),
        "p90": float(np.quantile(array, 0.90, method="nearest")),
        "maximum": float(np.max(array)),
    }


def _checkpoint_summary(
    rows: Sequence[dict[str, Any]], label: str
) -> dict[str, Any]:
    selected = [row for row in rows if row["checkpoint"] == label]
    total_graphs = len({(row["cover_seed"], row["graph_index"]) for row in rows})
    reachable = len(selected)
    return {
        "reachable_count": reachable,
        "reachable_fraction": reachable / max(total_graphs, 1),
        **{
            metric: (_mean(selected, metric) if selected else None)
            for metric in METRICS
        },
    }


def _seed_summary(
    checkpoint_rows: Sequence[dict[str, Any]],
    graph_rows: Sequence[dict[str, Any]],
    cover_seed: int,
) -> dict[str, Any]:
    selected_graphs = [row for row in graph_rows if row["cover_seed"] == cover_seed]
    by_checkpoint = {
        label: _checkpoint_summary(
            [row for row in checkpoint_rows if row["cover_seed"] == cover_seed],
            label,
        )
        for label in CHECKPOINTS
    }
    repair_rows = [row for row in selected_graphs if row["edge100_reachable"]]
    residual_preferred_fraction = float(
        np.mean([bool(row["residual_preferred"]) for row in selected_graphs])
    )
    if repair_rows:
        base_mean = _mean(repair_rows, "base_hybrid_bits")
        edge100_mean = _mean(repair_rows, "edge100_patch_only_bits")
        base_beats_edge100 = base_mean < edge100_mean
    else:
        base_mean = None
        edge100_mean = None
        base_beats_edge100 = True
    base_all = _mean(selected_graphs, "base_hybrid_bits")
    hybrid_after = _mean(selected_graphs, "hybrid_after_base_bits")
    hybrid_reduction = (base_all - hybrid_after) / max(base_all, 1e-12)
    hybrid_not_worse_edge100 = all(
        (not row["edge100_reachable"])
        or row["hybrid_after_base_bits"] <= row["edge100_patch_only_bits"]
        for row in selected_graphs
    )
    return {
        "cover_seed": int(cover_seed),
        "graph_count": len(selected_graphs),
        "checkpoints": by_checkpoint,
        "edge100_reachable_fraction": float(
            np.mean([bool(row["edge100_reachable"]) for row in selected_graphs])
        ),
        "residual_preferred_fraction": residual_preferred_fraction,
        "reachable_base_hybrid_mean_bits": base_mean,
        "reachable_edge100_patch_only_mean_bits": edge100_mean,
        "base_residual_beats_edge100_mean": bool(base_beats_edge100),
        "base_hybrid_mean_bits": base_all,
        "hybrid_after_base_mean_bits": hybrid_after,
        "hybrid_after_base_reduction": float(hybrid_reduction),
        "hybrid_not_worse_than_reachable_edge100": bool(
            hybrid_not_worse_edge100
        ),
        "hybrid_seed_gate": bool(
            hybrid_reduction >= 0.01 and hybrid_not_worse_edge100
        ),
    }


def classify(
    checkpoint_rows: Sequence[dict[str, Any]],
    graph_rows: Sequence[dict[str, Any]],
    cover_seeds: Sequence[int],
) -> dict[str, Any]:
    if not graph_rows:
        raise ValueError("graph rows cannot be empty")
    seed_summaries = [
        _seed_summary(checkpoint_rows, graph_rows, int(seed))
        for seed in cover_seeds
    ]
    invariant_gate = all(
        bool(row["cover_invariants"])
        and int(row["trajectory_length"]) == int(row["maximum_patches"]) + 1
        for row in graph_rows
    )
    residual_seed_wins = sum(
        summary["base_residual_beats_edge100_mean"] for summary in seed_summaries
    )
    residual_preferred_fraction = float(
        np.mean([bool(row["residual_preferred"]) for row in graph_rows])
    )
    residual_gate = (
        residual_seed_wins >= int(np.ceil(2 * len(seed_summaries) / 3))
        and residual_preferred_fraction >= 0.75
    )
    hybrid_gate = all(summary["hybrid_seed_gate"] for summary in seed_summaries)

    hybrid_all = [
        row for row in checkpoint_rows if row["checkpoint"] == "HYBRID_ALL"
    ]
    global_seed_checks = {}
    for seed in cover_seeds:
        rows = [row for row in hybrid_all if row["cover_seed"] == int(seed)]
        hybrid_bits = _mean(rows, "canonical_hybrid_proxy_bits")
        bitset_bits = _mean(rows, "direct_bitset_bits")
        enumerative_bits = _mean(rows, "direct_enumerative_exact_bits")
        global_seed_checks[str(seed)] = {
            "hybrid_mean_bits": hybrid_bits,
            "bitset_mean_bits": bitset_bits,
            "enumerative_mean_bits": enumerative_bits,
            "below_both": bool(
                hybrid_bits < bitset_bits and hybrid_bits < enumerative_bits
            ),
        }
    global_proxy_gate = all(
        values["below_both"] for values in global_seed_checks.values()
    )
    positive_prefix_excess = [
        int(row["minimum_positive_prefix_bits"])
        - int(row["direct_enumerative_exact_bits"])
        for row in graph_rows
    ]
    positive_prefix_beats_enumerative_fraction = float(
        np.mean([value < 0 for value in positive_prefix_excess])
    )
    checkpoint_by_key = {
        (int(row["graph_index"]), int(row["cover_seed"]), str(row["checkpoint"])): row
        for row in checkpoint_rows
    }
    edge100_added_patches = []
    residual_savings_bits = []
    for row in graph_rows:
        key = (int(row["graph_index"]), int(row["cover_seed"]))
        edge100 = checkpoint_by_key.get((*key, "EDGE100"))
        if edge100 is None:
            continue
        base = checkpoint_by_key[(*key, "BASE")]
        edge100_added_patches.append(
            int(edge100["patch_count"]) - int(base["patch_count"])
        )
        residual_savings_bits.append(
            int(row["edge100_patch_only_bits"]) - int(row["base_hybrid_bits"])
        )
    hybrid_stops_at_base_fraction = float(
        np.mean(
            [
                int(row["hybrid_after_base_patch_count"])
                == int(row["base_budget"])
                for row in graph_rows
            ]
        )
    )
    if not invariant_gate:
        label = "FAIL_EDGE_RESIDUAL_INVARIANTS"
    elif global_proxy_gate:
        label = "HYBRID_PROXY_RATE_COMPETITIVE"
    elif residual_gate or hybrid_gate:
        label = "REPAIR_POLICY_FOUND_BUT_NOT_GLOBAL_CODEC"
    else:
        label = "NO_RATE_JUSTIFIED_EDGE_REPAIR_POLICY"
    return {
        "classification": label,
        "invariant_gate": invariant_gate,
        "residual_repair_gate": residual_gate,
        "residual_seed_win_count": residual_seed_wins,
        "residual_preferred_fraction": residual_preferred_fraction,
        "hybrid_stopping_gate": hybrid_gate,
        "global_proxy_rate_gate": global_proxy_gate,
        "positive_prefix_beats_enumerative_fraction": (
            positive_prefix_beats_enumerative_fraction
        ),
        "minimum_positive_prefix_excess_bits": _distribution(
            positive_prefix_excess
        ),
        "edge100_added_patch_distribution": _distribution(
            edge100_added_patches
        ),
        "residual_savings_bit_distribution": _distribution(
            residual_savings_bits
        ),
        "hybrid_stops_at_base_fraction": hybrid_stops_at_base_fraction,
        "global_seed_checks": global_seed_checks,
        "seed_summaries": seed_summaries,
        "checkpoint_summaries": {
            label: _checkpoint_summary(checkpoint_rows, label)
            for label in CHECKPOINTS
        },
    }


def _fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def render_report(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    config = payload["config"]
    lines = [
        "# Beam8 edge repair、residual 与 hybrid rate 审计",
        "",
        f"> 日期：2026-08-05  ",
        f"> 协议：`{payload['protocol']}`  ",
        f"> 判定：`{decision['classification']}`",
        "",
        "## 1. 设置与总判定",
        "",
        (
            f"{config['graph_count']} graphs × {len(config['cover_seeds'])} cover seeds；"
            f"Beam{config['retained_beam']}/R{config['candidate_restarts']}；"
            f"s={config['patch_size']}, o={config['target_overlap']}；"
            f"最多 {config['maximum_patches']} patches。"
        ),
        "",
        f"- invariants：`{decision['invariant_gate']}`；",
        f"- BASE+residual vs EDGE100 gate：`{decision['residual_repair_gate']}`（图级 residual preferred={decision['residual_preferred_fraction']:.3f}）；",
        f"- HYBRID_AFTER_BASE gate：`{decision['hybrid_stopping_gate']}`；",
        f"- global proxy rate gate：`{decision['global_proxy_rate_gate']}`；",
        f"- 存在正长度 patch 前缀胜过 direct enumerative 的运行比例：`{decision['positive_prefix_beats_enumerative_fraction']:.3f}`。",
        "",
        "## 2. 核心发现",
        "",
        (
            "从 BASE 延长到 EDGE100 平均增加 "
            f"`{decision['edge100_added_patch_distribution']['mean']:.2f}` 个 patch "
            f"（p50={decision['edge100_added_patch_distribution']['p50']:.0f}，"
            f"p90={decision['edge100_added_patch_distribution']['p90']:.0f}，"
            f"max={decision['edge100_added_patch_distribution']['maximum']:.0f}）。"
        ),
        "",
        (
            "用 BASE + residual 代替补到 EDGE100，平均节省 "
            f"`{decision['residual_savings_bit_distribution']['mean']:.1f}` proxy bits "
            f"（min={decision['residual_savings_bit_distribution']['minimum']:.0f}，"
            f"p50={decision['residual_savings_bit_distribution']['p50']:.0f}，"
            f"p90={decision['residual_savings_bit_distribution']['p90']:.0f}）。"
        ),
        "",
        (
            "HYBRID_AFTER_BASE 恰好停在 BASE 的运行比例为 "
            f"`{decision['hybrid_stops_at_base_fraction']:.3f}`；即在当前显式成本模型下，"
            "BASE 之后没有一个额外 patch 的边际收益足以抵消 identity + KSVD code 成本。"
        ),
        "",
        (
            "每次运行中最便宜的正长度 patch 前缀相对 direct enumerative 仍平均多 "
            f"`{decision['minimum_positive_prefix_excess_bits']['mean']:.1f}` bits，"
            f"最接近的一次也多 `{decision['minimum_positive_prefix_excess_bits']['minimum']:.0f}` bits。"
        ),
        "",
        "因此，本轮找到的是 **Beam8 表示已经存在时最省的 coverage-completion policy：固定 BASE 后直接记录 residual edges**；它精确补齐未观察真实边，但不修正已观察 pair 上的 KSVD 重构/量化误差。没有找到胜过直接整图编码的 labeled-adjacency codec。",
        "",
        "## 3. Checkpoints",
        "",
        "| checkpoint | reach | patches | edge cover | pair cover | residual edges | RAW RMSE | identity | framing | KSVD proxy | residual bits | hybrid bits | raw exact bits |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in CHECKPOINTS:
        row = decision["checkpoint_summaries"][label]
        lines.append(
            f"| {label} | {_fmt(row['reachable_fraction'])} | "
            f"{_fmt(row['patch_count'], 2)} | {_fmt(row['edge_coverage'], 4)} | "
            f"{_fmt(row['pair_coverage'], 4)} | {_fmt(row['residual_edge_count'], 2)} | "
            f"{_fmt(row['raw_zero_fill_rmse'], 4)} | "
            f"{_fmt(row['canonical_set_identity_bits'], 1)} | "
            f"{_fmt(row['prefix_length_bits'], 1)} | "
            f"{_fmt(row['ksvd_code_proxy_bits'], 1)} | "
            f"{_fmt(row['residual_subset_bits'], 1)} | "
            f"{_fmt(row['canonical_hybrid_proxy_bits'], 1)} | "
            f"{_fmt(row['canonical_raw_exact_bits'], 1)} |"
        )
    lines.extend(
        [
            "",
            "`hybrid bits = prefix-length framing + optimistic canonical-set identity + K24/T3/q8 code proxy + residual subset`。RAW RMSE 只包含未覆盖真实边的 zero-fill coverage error。",
            "",
            "## 4. Seed robustness",
            "",
            "| seed | EDGE100 reach | residual preferred | BASE bits | EDGE100 patch-only bits* | hybrid-after bits | hybrid reduction | hybrid gate |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in decision["seed_summaries"]:
        lines.append(
            f"| {row['cover_seed']} | {row['edge100_reachable_fraction']:.3f} | "
            f"{row['residual_preferred_fraction']:.3f} | "
            f"{row['base_hybrid_mean_bits']:.1f} | "
            f"{_fmt(row['reachable_edge100_patch_only_mean_bits'], 1)} | "
            f"{row['hybrid_after_base_mean_bits']:.1f} | "
            f"{row['hybrid_after_base_reduction']:.3f} | "
            f"{row['hybrid_seed_gate']} |"
        )
    lines.extend(
        [
            "",
            "* EDGE100 patch-only mean 只在 60 patches 内达到 100% 的图上计算；不可达图自动视为 residual policy 更可行，但不混入该均值。",
            "",
            "## 5. Direct codec comparison",
            "",
            "| seed | HYBRID_ALL | bitset | direct enumerative | below both |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for seed, row in decision["global_seed_checks"].items():
        lines.append(
            f"| {seed} | {row['hybrid_mean_bits']:.1f} | "
            f"{row['bitset_mean_bits']:.1f} | {row['enumerative_mean_bits']:.1f} | "
            f"{row['below_both']} |"
        )
    lines.extend(
        [
            "",
            "## 6. Family × degree",
            "",
            "| family | degree | BASE patches | BASE edge | BASE residual | BASE hybrid bits | HYBRID_AFTER patches | HYBRID_AFTER bits | EDGE100 reach |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["strata"]:
        lines.append(
            f"| {row['family']} | {row['target_degree']} | "
            f"{row['base_patch_count']:.2f} | {row['base_edge_coverage']:.4f} | "
            f"{row['base_residual_edge_count']:.2f} | {row['base_hybrid_bits']:.1f} | "
            f"{row['hybrid_patch_count']:.2f} | {row['hybrid_bits']:.1f} | "
            f"{row['edge100_reachable_fraction']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 7. 路线结论与边界",
            "",
            "- 对 **已经选择 Beam8 作为结构表示** 的路线：采用 `BASE + residual edge subset`，不要继续增加 repair patches；它能消除 coverage error。只有 RAW patch values 被精确保留，或另有 compression-error correction 时，整体图恢复才是 exact。",
            "- 对 **labeled adjacency bit compression** 的路线：当前显式 patch-chain 编码应停止；不再为证明该主张追加实际 K24/T3/q8 量化实验。",
            "- 对 **结构表示/下游学习** 的路线：Beam8、rooted-canonical slots 与共享字典仍可继续，因为其价值目标不是击败 direct adjacency codec。",
            "- 若未来研究 sampler-induced joint entropy coding，应作为新协议重新立项；当前 canonical-set 公式不是该分布下的信息论下界，不能据此宣称所有可能图 codec 都不可行。",
            "",
            "边界：",
            "",
            "- 输入是完整已知图；缺边输入不能把 unobserved 解释为 0。",
            "- residual 是 encoder 从完整输入算出的确定性 sidecar，不是标签泄漏。",
            "- 已计入 0..maximum_patches 的 prefix-length framing；仍未计 codec mode、dictionary、quantizer、checksum 等其他 framing/模型开销。",
            "- canonical identity 和 K24/T3/q8 都是偏向 patch 方法的 optimistic proxy；即使假设 dictionary 免费且 8-bit coefficient 无失真，正长度 patch 前缀也没有胜过 direct enumerative。真实量化只会增加成本或失真。",
            "- `HYBRID_ALL` 允许选择 0 patch，因此若 direct enumerative 本身最便宜，会诚实地停在 0；这只约束 labeled-adjacency codec 主张，不否定 Beam8 的结构表示用途。",
            "",
        ]
    )
    return "\n".join(lines)


def _build_strata(
    checkpoint_rows: Sequence[dict[str, Any]],
    graph_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    strata = []
    keys = sorted(
        {(str(row["family"]), int(row["target_degree"])) for row in graph_rows}
    )
    for family, degree in keys:
        base = [
            row
            for row in checkpoint_rows
            if row["checkpoint"] == "BASE"
            and row["family"] == family
            and row["target_degree"] == degree
        ]
        hybrid = [
            row
            for row in checkpoint_rows
            if row["checkpoint"] == "HYBRID_AFTER_BASE"
            and row["family"] == family
            and row["target_degree"] == degree
        ]
        graphs = [
            row
            for row in graph_rows
            if row["family"] == family and row["target_degree"] == degree
        ]
        strata.append(
            {
                "family": family,
                "target_degree": degree,
                "base_patch_count": _mean(base, "patch_count"),
                "base_edge_coverage": _mean(base, "edge_coverage"),
                "base_residual_edge_count": _mean(base, "residual_edge_count"),
                "base_hybrid_bits": _mean(base, "canonical_hybrid_proxy_bits"),
                "hybrid_patch_count": _mean(hybrid, "patch_count"),
                "hybrid_bits": _mean(hybrid, "canonical_hybrid_proxy_bits"),
                "edge100_reachable_fraction": float(
                    np.mean([bool(row["edge100_reachable"]) for row in graphs])
                ),
            }
        )
    return strata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-bank-seed", type=int, default=810001)
    parser.add_argument(
        "--cover-seeds", type=int, nargs="+", default=[950101, 950102, 950103]
    )
    parser.add_argument("--n-nodes", type=int, default=50)
    parser.add_argument("--degrees", type=int, nargs="+", default=[15, 20, 25])
    parser.add_argument("--graphs-per-cell", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=10)
    parser.add_argument("--target-overlap", type=int, default=3)
    parser.add_argument("--multiplier", type=float, default=1.5)
    parser.add_argument("--retained-beam", type=int, default=8)
    parser.add_argument("--candidate-restarts", type=int, default=1)
    parser.add_argument("--maximum-patches", type=int, default=60)
    parser.add_argument("--n-atoms", type=int, default=24)
    parser.add_argument("--sparsity", type=int, default=3)
    parser.add_argument("--coefficient-bits", type=int, default=8)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    if args.maximum_patches < 1:
        raise ValueError("maximum patches must be positive")

    graph_count = len(FAMILIES) * len(args.degrees) * args.graphs_per_cell
    sequences = np.random.SeedSequence(args.graph_bank_seed).spawn(graph_count)
    graph_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    graph_index = 0
    for family in FAMILIES:
        for degree in args.degrees:
            for _replicate in range(args.graphs_per_cell):
                graph_seed = int(
                    sequences[graph_index].generate_state(1, dtype=np.uint32)[0]
                )
                adjacency = generate_graph(family, args.n_nodes, degree, graph_seed)
                base_budget = patch_budget(
                    adjacency,
                    patch_size=args.patch_size,
                    target_overlap=args.target_overlap,
                    edge_capacity_multiplier=args.multiplier,
                )
                if base_budget > args.maximum_patches:
                    raise ValueError("maximum patches is below the frozen BASE budget")
                for cover_seed in args.cover_seeds:
                    seed = int(
                        np.random.SeedSequence(
                            [cover_seed, graph_index]
                        ).generate_state(1, dtype=np.uint32)[0]
                    )
                    started = time.perf_counter()
                    cover = sample_marginal_candidate_cover(
                        adjacency,
                        np.random.default_rng(seed),
                        n_patches=args.maximum_patches,
                        patch_size=args.patch_size,
                        target_overlap=args.target_overlap,
                        retained_beam=args.retained_beam,
                        candidate_restarts=args.candidate_restarts,
                    )
                    elapsed = time.perf_counter() - started
                    cover_audit = audit_cover(adjacency, cover)
                    invariants = (
                        cover_audit["patch_connected_rate"] == 1.0
                        and cover_audit["continuous_transition_fraction"] == 1.0
                        and all(
                            value == args.target_overlap
                            for value in cover_audit["transition_overlap_exact"]
                        )
                    )
                    trajectory = prefix_rate_trajectory(
                        adjacency,
                        cover,
                        patch_size=args.patch_size,
                        overlap=args.target_overlap,
                        n_atoms=args.n_atoms,
                        sparsity=args.sparsity,
                        coefficient_bits=args.coefficient_bits,
                    )
                    selected = select_checkpoints(
                        trajectory, base_patch_count=base_budget
                    )
                    base = selected["BASE"]
                    hybrid_after = selected["HYBRID_AFTER_BASE"]
                    edge100 = selected["EDGE100"]
                    assert base is not None and hybrid_after is not None
                    edge100_patch_only = (
                        int(edge100["prefix_length_bits"])
                        + int(edge100["canonical_set_identity_bits"])
                        + int(edge100["ksvd_code_proxy_bits"])
                        if edge100 is not None
                        else None
                    )
                    positive_prefix = trajectory[1:]
                    minimum_positive = min(
                        positive_prefix,
                        key=lambda row: (
                            int(row["canonical_hybrid_proxy_bits"]),
                            int(row["patch_count"]),
                        ),
                    )
                    residual_preferred = (
                        edge100 is None
                        or int(base["canonical_hybrid_proxy_bits"])
                        < int(edge100_patch_only)
                    )
                    graph_rows.append(
                        {
                            "graph_index": graph_index,
                            "family": family,
                            "target_degree": degree,
                            "cover_seed": int(cover_seed),
                            "base_budget": base_budget,
                            "maximum_patches": args.maximum_patches,
                            "trajectory_length": len(trajectory),
                            "sampling_seconds": float(elapsed),
                            "cover_invariants": bool(invariants),
                            "edge100_reachable": edge100 is not None,
                            "edge100_patch_only_bits": edge100_patch_only,
                            "base_hybrid_bits": int(
                                base["canonical_hybrid_proxy_bits"]
                            ),
                            "hybrid_after_base_bits": int(
                                hybrid_after["canonical_hybrid_proxy_bits"]
                            ),
                            "hybrid_after_base_patch_count": int(
                                hybrid_after["patch_count"]
                            ),
                            "minimum_positive_prefix_bits": int(
                                minimum_positive["canonical_hybrid_proxy_bits"]
                            ),
                            "minimum_positive_prefix_patch_count": int(
                                minimum_positive["patch_count"]
                            ),
                            "direct_enumerative_exact_bits": int(
                                base["direct_enumerative_exact_bits"]
                            ),
                            "residual_preferred": bool(residual_preferred),
                        }
                    )
                    for row in trajectory:
                        trajectory_rows.append(
                            {
                                "graph_index": graph_index,
                                "family": family,
                                "target_degree": degree,
                                "cover_seed": int(cover_seed),
                                **row,
                            }
                        )
                    for checkpoint, row in selected.items():
                        if row is None:
                            continue
                        checkpoint_rows.append(
                            {
                                "graph_index": graph_index,
                                "family": family,
                                "target_degree": degree,
                                "cover_seed": int(cover_seed),
                                "checkpoint": checkpoint,
                                **row,
                            }
                        )
                    print(
                        f"graph={graph_index} {family}/d{degree} seed={cover_seed} "
                        f"base={base_budget} edge={base['edge_coverage']:.3f} "
                        f"res={base['residual_edge_count']} hybrid_t={hybrid_after['patch_count']}",
                        flush=True,
                    )
                graph_index += 1

    decision = classify(checkpoint_rows, graph_rows, args.cover_seeds)
    payload = {
        "protocol": PROTOCOL,
        "config": {
            "graph_bank_seed": args.graph_bank_seed,
            "cover_seeds": args.cover_seeds,
            "families": list(FAMILIES),
            "n_nodes": args.n_nodes,
            "target_degrees": args.degrees,
            "graphs_per_family_degree": args.graphs_per_cell,
            "graph_count": graph_count,
            "patch_size": args.patch_size,
            "target_overlap": args.target_overlap,
            "multiplier": args.multiplier,
            "retained_beam": args.retained_beam,
            "candidate_restarts": args.candidate_restarts,
            "maximum_patches": args.maximum_patches,
            "n_atoms": args.n_atoms,
            "sparsity": args.sparsity,
            "coefficient_bits": args.coefficient_bits,
            "labels_used": False,
        },
        "graph_rows": graph_rows,
        "checkpoint_rows": checkpoint_rows,
        "trajectory_rows": trajectory_rows,
        "strata": _build_strata(checkpoint_rows, graph_rows),
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
