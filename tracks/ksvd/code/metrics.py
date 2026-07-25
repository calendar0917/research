from __future__ import annotations

import math
from typing import Any

from .graph import Graph
from .sample import SampleBundle


def _gini(counts: list[int]) -> float:
    if not counts:
        return 0.0
    xs = sorted(counts)
    n = len(xs)
    total = sum(xs)
    if total == 0:
        return 0.0
    acc = 0.0
    for i, x in enumerate(xs, start=1):
        acc += i * x
    return (2 * acc) / (n * total) - (n + 1) / n


def evaluate_bundle(g: Graph, bundle: SampleBundle) -> dict[str, Any]:
    all_edges = g.edges()
    m_e = max(len(all_edges), 1)
    counts = [bundle.edge_counts.get(e, 0) for e in all_edges]
    hit = sum(1 for c in counts if c > 0)
    repeat_mass = sum(max(c - 1, 0) for c in counts)
    sizes = [len(s) for s in bundle.node_sets]
    esizes = [sub.num_edges() for sub in bundle.subgraphs]
    cap = max(sizes) if sizes else 0

    n_sg = max(len(bundle.subgraphs), 1)
    mean_count = sum(counts) / m_e
    return {
        "method": bundle.method,
        "n_subgraphs": len(bundle.subgraphs),
        "edge_cover": hit / m_e,
        # raw inclusion surplus (sensitive to n_subgraphs & |S|); use for same-budget RW compare
        "edge_repeat": repeat_mass / m_e,
        "edge_mean_count": mean_count,
        # normalized: mean times each edge appears **per subgraph** (fairer across methods)
        "edge_count_per_sg": mean_count / n_sg,
        "edge_gini": _gini(counts),
        "mean_|S|": sum(sizes) / max(len(sizes), 1),
        "p95_|S|": _percentile(sizes, 95),
        "mean_|Es|": sum(esizes) / max(len(esizes), 1),
        "max_|S|": max(sizes) if sizes else 0,
        "min_|S|": min(sizes) if sizes else 0,
    }


def _percentile(xs: list[int], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    k = (len(ys) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(ys[int(k)])
    return ys[f] * (c - k) + ys[c] * (k - f)


def compare_methods(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by = {r["method"]: r for r in rows}
    out: dict[str, Any] = {"methods": by}
    # B0 vs M0: different |S| budgets — only compare cover & count_per_sg, not raw repeat
    if "B0" in by and "M0" in by:
        out["M0_cover"] = by["M0"]["edge_cover"]
        out["B0_cover"] = by["B0"]["edge_cover"]
        out["M0_count_per_sg"] = by["M0"].get("edge_count_per_sg")
        out["B0_count_per_sg"] = by["B0"].get("edge_count_per_sg")
    # same-family RW: M0 downweight should not raise count_per_sg vs B1 (soft check)
    if "B1" in by and "M0" in by:
        out["M0_count_per_sg_lte_B1"] = by["M0"].get("edge_count_per_sg", 1e9) <= by["B1"].get(
            "edge_count_per_sg", 0
        ) * 1.05
        out["B1_count_per_sg"] = by["B1"].get("edge_count_per_sg")
        out["M0_count_per_sg"] = by["M0"].get("edge_count_per_sg")
    return out
