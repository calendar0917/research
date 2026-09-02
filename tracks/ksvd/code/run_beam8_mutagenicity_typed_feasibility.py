"""Label-free typed Beam8 geometry audit on TU Mutagenicity."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .data_tud import load_tud
from .run_luyin14_edge_aware_joint import _load_edge_pyg
from .run_beam8_nci1_chain_classification import (
    DEFAULT_ROOT,
    ROOT,
    _atomic_json,
    _atomic_text,
    _relation_matrices,
    _shuffle_permutation,
    prepare_graph,
)


PROTOCOL = "tracks/ksvd/docs/KSVD_BEAM8_MUTAGENICITY_TYPED_FEASIBILITY_PROTOCOL_20260813.md"
DEFAULT_JSON = ROOT / "tracks/ksvd/results/luyin14/beam8_mutagenicity_typed_feasibility_20260813.json"
DEFAULT_REPORT = ROOT / "tracks/ksvd/results/luyin14/BEAM8_MUTAGENICITY_TYPED_FEASIBILITY_20260813.md"
GEOMETRIES = {"s10_o3": (10, 3), "s8_o2": (8, 2)}


def _typed_adjacency(data: Any, edge_dim: int) -> np.ndarray:
    typed = np.zeros((int(data.num_nodes), int(data.num_nodes)), dtype=np.int16)
    edges = data.edge_index.cpu().numpy()
    types = data.edge_attr.argmax(dim=1).cpu().numpy() + 1
    typed[edges[0], edges[1]] = types
    if np.any(typed != typed.T):
        raise RuntimeError("expected symmetric typed TU edges")
    if int(typed.max(initial=0)) > edge_dim:
        raise RuntimeError("edge type exceeds declared dimension")
    return typed


def _patch_typed_features(item: Any, typed: np.ndarray, edge_dim: int) -> tuple[np.ndarray, np.ndarray]:
    patch_hist = []
    new_hist = []
    covered: set[tuple[int, int]] = set()
    for nodes in item.slot_nodes:
        local = typed[np.ix_(nodes, nodes)]
        upper = local[np.triu_indices(len(nodes), k=1)]
        patch_hist.append(
            [float(np.sum(upper == edge_type)) for edge_type in range(1, edge_dim + 1)]
        )
        row = np.zeros(edge_dim, dtype=np.float64)
        for left_index, left in enumerate(nodes):
            for right in nodes[left_index + 1 :]:
                edge_type = int(typed[left, right])
                pair = tuple(sorted((int(left), int(right))))
                if edge_type and pair not in covered:
                    row[edge_type - 1] += 1.0
                    covered.add(pair)
        new_hist.append(row)
    return np.asarray(patch_hist), np.asarray(new_hist)


def _chain_margin(items: Sequence[Any], tokens: Sequence[np.ndarray], *, seed: int) -> dict[str, float | int]:
    true_values = []
    shuffled_values = []
    usable = 0
    for item, values in zip(items, tokens):
        weights = _relation_matrices(item)[0]
        target, source = np.nonzero(weights > 0)
        if target.size == 0:
            continue
        permutation = _shuffle_permutation(len(values), seed=seed + item.index * 1009)

        def cosine(current: np.ndarray) -> np.ndarray:
            left, right = current[target], current[source]
            return np.sum(left * right, axis=1) / np.maximum(
                np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1), 1e-12
            )

        true_values.extend(cosine(values).tolist())
        shuffled_values.extend(cosine(values[permutation]).tolist())
        usable += 1
    true = np.asarray(true_values, dtype=np.float64)
    shuffled = np.asarray(shuffled_values, dtype=np.float64)
    return {
        "usable_graphs": usable,
        "chain_edges": int(len(true)),
        "true_mean": float(true.mean()) if true.size else 0.0,
        "shuffled_mean": float(shuffled.mean()) if shuffled.size else 0.0,
        "true_minus_shuffled": float(true.mean() - shuffled.mean()) if true.size else 0.0,
    }


def _summary(items: Sequence[Any], typed_graphs: Sequence[np.ndarray], edge_dim: int, *, seed: int) -> dict[str, Any]:
    patches = np.asarray([len(item.vectors) for item in items], dtype=np.float64)
    chain_edges = np.asarray([np.count_nonzero(_relation_matrices(item)[0]) for item in items])
    component_count = sum(item.sampling["components"] for item in items)
    partial_count = sum(item.sampling["partial_components"] for item in items)
    edge_coverage = np.asarray([item.sampling["edge_coverage"] for item in items])
    node_coverage = np.asarray([item.sampling["node_coverage"] for item in items])
    patch_hists, new_hists = [], []
    total = np.zeros(edge_dim, dtype=np.int64)
    covered = np.zeros(edge_dim, dtype=np.int64)
    graph_recalls = [[] for _ in range(edge_dim)]
    rare_graphs = rare_covered_graphs = 0
    for item, typed in zip(items, typed_graphs):
        patch_hist, new_hist = _patch_typed_features(item, typed, edge_dim)
        patch_hists.append(patch_hist)
        new_hists.append(new_hist)
        graph_total = np.asarray(
            [np.sum(typed[np.triu_indices(len(typed), k=1)] == value) for value in range(1, edge_dim + 1)],
            dtype=np.int64,
        )
        graph_covered = new_hist.sum(axis=0).astype(np.int64)
        total += graph_total
        covered += graph_covered
        for edge_type in range(edge_dim):
            if graph_total[edge_type] > 0:
                graph_recalls[edge_type].append(graph_covered[edge_type] / graph_total[edge_type])
        if graph_total[-1] > 0:
            rare_graphs += 1
            rare_covered_graphs += int(graph_covered[-1] > 0)
    per_type = []
    for edge_type in range(edge_dim):
        per_type.append(
            {
                "edge_type": edge_type + 1,
                "total_edges": int(total[edge_type]),
                "covered_edges": int(covered[edge_type]),
                "aggregate_recall": float(covered[edge_type] / max(total[edge_type], 1)),
                "mean_graph_recall": float(np.mean(graph_recalls[edge_type])) if graph_recalls[edge_type] else None,
                "graphs_with_type": len(graph_recalls[edge_type]),
            }
        )
    node_tokens = [item.node_histograms for item in items]
    return {
        "graph_count": len(items),
        "mean_patches": float(patches.mean()),
        "single_patch_fraction": float(np.mean(patches == 1)),
        "graphs_with_chain_fraction": float(np.mean(chain_edges > 0)),
        "mean_directed_chain_edges": float(chain_edges.mean()),
        "partial_component_fraction": float(partial_count / max(component_count, 1)),
        "mean_edge_coverage": float(edge_coverage.mean()),
        "edge_coverage_p10": float(np.quantile(edge_coverage, 0.1)),
        "mean_node_coverage": float(node_coverage.mean()),
        "per_edge_type": per_type,
        "rare_type_graph_coverage": float(rare_covered_graphs / max(rare_graphs, 1)),
        "node_chain_binding": _chain_margin(items, node_tokens, seed=seed),
        "new_bond_chain_binding": _chain_margin(items, new_hists, seed=seed + 100000),
    }


def _decision(s10: dict[str, Any], s8: dict[str, Any]) -> dict[str, Any]:
    type_noninferior = all(
        right["aggregate_recall"] >= left["aggregate_recall"] - 0.02
        for left, right in zip(s10["per_edge_type"], s8["per_edge_type"])
    )
    checks = {
        "single_patch_relative_reduction_ge_20pct": bool(s8["single_patch_fraction"] <= 0.8 * max(s10["single_patch_fraction"], 1e-12)),
        "chain_graph_fraction_noninferior": bool(s8["graphs_with_chain_fraction"] >= s10["graphs_with_chain_fraction"] - 1e-12),
        "mean_chain_edges_relative_gain_ge_20pct": bool(s8["mean_directed_chain_edges"] >= 1.2 * max(s10["mean_directed_chain_edges"], 1e-12)),
        "edge_coverage_noninferior_1pt": bool(s8["mean_edge_coverage"] >= s10["mean_edge_coverage"] - 0.01),
        "partial_component_fraction_noninferior": bool(s8["partial_component_fraction"] <= s10["partial_component_fraction"] + 1e-12),
        "all_bond_types_noninferior_2pt": bool(type_noninferior),
        "node_chain_margin_positive": bool(s8["node_chain_binding"]["true_minus_shuffled"] > 0),
        "new_bond_chain_margin_positive": bool(s8["new_bond_chain_binding"]["true_minus_shuffled"] > 0),
    }
    gate = all(checks.values())
    return {
        "checks": checks,
        "gate": gate,
        "rare_bond_evidence_sufficient": bool(s8["rare_type_graph_coverage"] >= 0.8),
        "decision": "S8_O2_READY_FOR_TYPED_CLASSIFICATION" if gate else "TYPED_GEOMETRY_GATE_FAILED",
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
                print(f"{name} {index + 1}/{len(graphs)}", flush=True)
        results[name] = _summary(items, typed_graphs, edge_dim, seed=731421)
    return {
        "protocol": PROTOCOL,
        "dataset": metadata | {"edge_dim": edge_dim},
        "geometries": results,
        "decision": _decision(results["s10_o3"], results["s8_o2"]),
        "seconds": time.time() - started,
    }


def render_report(payload: dict[str, Any]) -> str:
    lines = ["# Beam8/Mutagenicity typed feasibility audit", "",
             f"> 协议：`{payload['protocol']}`  ", f"> 判定：`{payload['decision']['decision']}`", "",
             "| metric | s10/o3 | s8/o2 |", "|---|---:|---:|"]
    metrics = ("mean_patches", "single_patch_fraction", "graphs_with_chain_fraction",
               "mean_directed_chain_edges", "partial_component_fraction", "mean_edge_coverage",
               "edge_coverage_p10", "mean_node_coverage", "rare_type_graph_coverage")
    for metric in metrics:
        lines.append(f"| {metric} | {payload['geometries']['s10_o3'][metric]:.4f} | {payload['geometries']['s8_o2'][metric]:.4f} |")
    lines.extend(["", "## Per-bond coverage", "", "| geometry | bond type | edges | aggregate recall | mean graph recall | graphs |", "|---|---:|---:|---:|---:|---:|"])
    for geometry in GEOMETRIES:
        for row in payload["geometries"][geometry]["per_edge_type"]:
            mean = "—" if row["mean_graph_recall"] is None else f"{row['mean_graph_recall']:.4f}"
            lines.append(f"| {geometry} | {row['edge_type']} | {row['total_edges']} | {row['aggregate_recall']:.4f} | {mean} | {row['graphs_with_type']} |")
    lines.extend(["", "## Binding margins", ""])
    for geometry in GEOMETRIES:
        node = payload["geometries"][geometry]["node_chain_binding"]
        bond = payload["geometries"][geometry]["new_bond_chain_binding"]
        lines.append(f"- {geometry} node TRUE−SHUFFLED：`{node['true_minus_shuffled']:+.4f}`；new-bond TRUE−SHUFFLED：`{bond['true_minus_shuffled']:+.4f}`。")
    lines.extend(["", "## Frozen checks", ""])
    for name, value in payload["decision"]["checks"].items():
        lines.append(f"- {name}：`{value}`；")
    lines.append(f"- rare bond evidence sufficient：`{payload['decision']['rare_bond_evidence_sufficient']}`；")
    lines.extend(["", "## Boundary", "", "- cover 不读取 bond type；bond labels 只用于无标签 coverage/binding audit。", "- graph labels 未进入本审计。"])
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

