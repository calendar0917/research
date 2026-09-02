"""Binary-only FAIR95 completion audit for typed Mutagenicity Beam8 covers."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .canonical_slots import reorder_cover_structurally
from .data_tud import load_tud
from .overlap_cover import _make_cover, _make_patch
from .run_beam8_nci1_chain_classification import (
    DEFAULT_ROOT,
    ROOT,
    _atomic_json,
    _atomic_text,
    _relation_matrices,
    prepare_graph,
)
from .run_beam8_mutagenicity_typed_feasibility import _typed_adjacency
from .run_luyin14_edge_aware_joint import _load_edge_pyg
from .run_luyin14_route import (
    _complete_edges,
    _fair_prefix_from_completed,
    _prefix_metrics,
)


PROTOCOL = "tracks/ksvd/docs/KSVD_BEAM8_MUTAGENICITY_FAIR95_COMPLETION_PROTOCOL_20260813.md"
DEFAULT_JSON = ROOT / "tracks/ksvd/results/luyin14/beam8_mutagenicity_fair95_completion_20260813.json"
DEFAULT_REPORT = ROOT / "tracks/ksvd/results/luyin14/BEAM8_MUTAGENICITY_FAIR95_COMPLETION_20260813.md"
GEOMETRIES = {"s10_o3": (10, 3), "s8_o2": (8, 2)}


def _adjacency(graph: Any) -> np.ndarray:
    values = np.zeros((graph.n, graph.n), dtype=np.int8)
    for left, right in graph.edges():
        values[left, right] = values[right, left] = 1
    return values


def _base_cover(item: Any, adjacency: np.ndarray) -> Any:
    patches = [
        _make_patch(adjacency, nodes, center)
        for nodes, center in zip(item.slot_nodes, item.centers)
    ]
    return _make_cover(
        "beam8_component_base",
        patches,
        item.segment_ids,
        [None] * len(patches),
        [0] * len(patches),
    )


def fair95_cover(item: Any, adjacency: np.ndarray, *, patch_size: int) -> tuple[Any, bool]:
    base = _base_cover(item, adjacency)
    completed = _complete_edges(adjacency, base, patch_size=patch_size)
    fair, reached = _fair_prefix_from_completed(adjacency, completed)
    fair, _audit = reorder_cover_structurally(adjacency, fair, "rooted_canonical")
    return fair, reached


def _covered_by_type(cover: Any, typed: np.ndarray, edge_dim: int) -> tuple[np.ndarray, np.ndarray]:
    total = np.asarray(
        [
            np.sum(typed[np.triu_indices(len(typed), k=1)] == edge_type)
            for edge_type in range(1, edge_dim + 1)
        ],
        dtype=np.int64,
    )
    covered_pairs: set[tuple[int, int]] = set()
    for patch in cover.patches:
        nodes = patch.node_ids
        for left_index, left in enumerate(nodes):
            for right in nodes[left_index + 1 :]:
                if typed[left, right] != 0:
                    covered_pairs.add(tuple(sorted((int(left), int(right)))))
    covered = np.zeros(edge_dim, dtype=np.int64)
    for left, right in covered_pairs:
        covered[int(typed[left, right]) - 1] += 1
    return total, covered


def _summary(
    items: Sequence[Any],
    graphs: Sequence[Any],
    typed_graphs: Sequence[np.ndarray],
    edge_dim: int,
    *,
    patch_size: int,
) -> dict[str, Any]:
    base_counts, fair_counts, reached_values = [], [], []
    fair_edge, fair_node, fair_incident = [], [], []
    chain_edges = []
    total = np.zeros(edge_dim, dtype=np.int64)
    covered = np.zeros(edge_dim, dtype=np.int64)
    graph_recalls = [[] for _ in range(edge_dim)]
    rare_graphs = rare_graph_hits = 0
    completion_new = np.zeros(edge_dim, dtype=np.int64)
    for item, graph, typed in zip(items, graphs, typed_graphs):
        adjacency = _adjacency(graph)
        base = _base_cover(item, adjacency)
        fair, reached = fair95_cover(item, adjacency, patch_size=patch_size)
        metrics = _prefix_metrics(adjacency, fair.patches)
        base_counts.append(len(base.patches))
        fair_counts.append(len(fair.patches))
        reached_values.append(reached)
        fair_edge.append(metrics["edge_coverage"])
        fair_node.append(metrics["node_coverage"])
        fair_incident.append(metrics["incident_p10"])
        chain_edges.append(np.count_nonzero(_relation_matrices(item)[0]))
        graph_total, graph_covered = _covered_by_type(fair, typed, edge_dim)
        total += graph_total
        covered += graph_covered
        for edge_type in range(edge_dim):
            if graph_total[edge_type] > 0:
                graph_recalls[edge_type].append(
                    graph_covered[edge_type] / graph_total[edge_type]
                )
        if graph_total[-1] > 0:
            rare_graphs += 1
            rare_graph_hits += int(graph_covered[-1] > 0)
        base_total, base_covered = _covered_by_type(base, typed, edge_dim)
        if not np.array_equal(base_total, graph_total):
            raise RuntimeError("base/fair totals disagree")
        completion_new += graph_covered - base_covered
    base_counts = np.asarray(base_counts, dtype=np.float64)
    fair_counts = np.asarray(fair_counts, dtype=np.float64)
    extra = fair_counts - base_counts
    per_type = []
    for edge_type in range(edge_dim):
        per_type.append(
            {
                "edge_type": edge_type + 1,
                "total_edges": int(total[edge_type]),
                "covered_edges": int(covered[edge_type]),
                "aggregate_recall": float(covered[edge_type] / max(total[edge_type], 1)),
                "mean_graph_recall": float(np.mean(graph_recalls[edge_type])) if graph_recalls[edge_type] else None,
                "completion_new_edges": int(completion_new[edge_type]),
            }
        )
    return {
        "graph_count": len(items),
        "all_reached": bool(all(reached_values)),
        "mean_base_patches": float(base_counts.mean()),
        "mean_fair_patches": float(fair_counts.mean()),
        "mean_extra_patches": float(extra.mean()),
        "mean_extra_fraction": float(np.mean(extra / np.maximum(base_counts, 1.0))),
        "mean_edge_coverage": float(np.mean(fair_edge)),
        "mean_node_coverage": float(np.mean(fair_node)),
        "mean_incident_p10": float(np.mean(fair_incident)),
        "graphs_with_chain_fraction": float(np.mean(np.asarray(chain_edges) > 0)),
        "mean_directed_chain_edges": float(np.mean(chain_edges)),
        "per_edge_type": per_type,
        "rare_type_graph_coverage": float(rare_graph_hits / max(rare_graphs, 1)),
    }


def _checks(row: dict[str, Any]) -> dict[str, bool]:
    return {
        "all_reached": bool(row["all_reached"]),
        "edge_node_coverage": bool(row["mean_edge_coverage"] >= 0.95 and row["mean_node_coverage"] >= 1.0 - 1e-12),
        "rare_aggregate_recall": bool(row["per_edge_type"][-1]["aggregate_recall"] >= 0.80),
        "rare_graph_coverage": bool(row["rare_type_graph_coverage"] >= 0.80),
        "extra_patch_fraction": bool(row["mean_extra_fraction"] <= 0.75),
        "chain_graph_fraction": bool(row["graphs_with_chain_fraction"] >= 0.50),
    }


def _decision(results: dict[str, Any]) -> dict[str, Any]:
    checks = {name: _checks(row) for name, row in results.items()}
    passed = [name for name in GEOMETRIES if all(checks[name].values())]
    selected = None
    if passed:
        selected = min(
            passed,
            key=lambda name: (
                round(results[name]["mean_extra_patches"] / 0.25),
                -results[name]["mean_directed_chain_edges"],
            ),
        )
    return {
        "checks": checks,
        "passed_geometries": passed,
        "selected_geometry": selected,
        "decision": "FAIR95_READY_FOR_TYPED_CLASSIFICATION" if selected else "FAIR95_COMPLETION_BELOW_GATE",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    graphs, labels, features, metadata = load_tud(args.dataset, args.dataset_root)
    raw, raw_labels, _classes, edge_dim = _load_edge_pyg(args.dataset, args.dataset_root)
    if not np.array_equal(labels, raw_labels):
        raise RuntimeError("loaders disagree on labels")
    typed_graphs = [_typed_adjacency(data, edge_dim) for data in raw]
    results = {}
    for name, (patch_size, overlap) in GEOMETRIES.items():
        items = []
        for index, (graph, node_features) in enumerate(zip(graphs, features)):
            items.append(
                prepare_graph(index, graph, 0, node_features, patch_size=patch_size, overlap=overlap,
                              retained_beam=8, edge_capacity_multiplier=1.5, seed=20260813)
            )
            if (index + 1) % 500 == 0 or index + 1 == len(graphs):
                print(f"{name} cover {index + 1}/{len(graphs)}", flush=True)
        results[name] = _summary(items, graphs, typed_graphs, edge_dim, patch_size=patch_size)
        print(f"{name} fair95 done", flush=True)
    return {
        "protocol": PROTOCOL,
        "dataset": metadata | {"edge_dim": edge_dim},
        "geometries": results,
        "decision": _decision(results),
        "seconds": time.time() - started,
    }


def render_report(payload: dict[str, Any]) -> str:
    lines = ["# Beam8/Mutagenicity FAIR95 completion audit", "",
             f"> 协议：`{payload['protocol']}`  ", f"> 判定：`{payload['decision']['decision']}`  ",
             f"> selected：`{payload['decision']['selected_geometry']}`", "",
             "| metric | s10/o3 | s8/o2 |", "|---|---:|---:|"]
    metrics = ("mean_base_patches", "mean_fair_patches", "mean_extra_patches", "mean_extra_fraction",
               "mean_edge_coverage", "mean_node_coverage", "mean_incident_p10",
               "graphs_with_chain_fraction", "mean_directed_chain_edges", "rare_type_graph_coverage")
    for metric in metrics:
        lines.append(f"| {metric} | {payload['geometries']['s10_o3'][metric]:.4f} | {payload['geometries']['s8_o2'][metric]:.4f} |")
    lines.extend(["", "## Per-bond FAIR95 recall", "", "| geometry | bond | recall | mean graph recall | completion new edges |", "|---|---:|---:|---:|---:|"])
    for geometry in GEOMETRIES:
        for row in payload["geometries"][geometry]["per_edge_type"]:
            lines.append(f"| {geometry} | {row['edge_type']} | {row['aggregate_recall']:.4f} | {row['mean_graph_recall']:.4f} | {row['completion_new_edges']} |")
    lines.extend(["", "## Frozen checks", ""])
    for geometry, checks in payload["decision"]["checks"].items():
        lines.append(f"### {geometry}")
        lines.append("")
        for name, value in checks.items():
            lines.append(f"- {name}：`{value}`；")
    lines.extend(["", "## Boundary", "", "- completion 不读取 bond type；新增 patch 都是新 segment。", "- 分类必须同时报告 BASE 与 FAIR95，不能把 completion 伪装成连续 Beam8 chain。"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Mutagenicity")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    payload = run(args)
    _atomic_json(args.json, payload)
    _atomic_text(args.report, render_report(payload))
    print(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
