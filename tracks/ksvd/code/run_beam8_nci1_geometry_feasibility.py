"""Label-free NCI1 feasibility audit for frozen Beam8 geometries."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .data_tud import load_tud
from .run_beam8_nci1_chain_classification import (
    DEFAULT_ROOT,
    ROOT,
    _atomic_json,
    _atomic_text,
    _relation_matrices,
    _shuffle_permutation,
    prepare_graph,
)


PROTOCOL = "tracks/ksvd/docs/KSVD_BEAM8_NCI1_GEOMETRY_FEASIBILITY_PROTOCOL_20260813.md"
DEFAULT_JSON = ROOT / "tracks/ksvd/results/luyin14/beam8_nci1_geometry_feasibility_20260813.json"
DEFAULT_REPORT = ROOT / "tracks/ksvd/results/luyin14/BEAM8_NCI1_GEOMETRY_FEASIBILITY_20260813.md"
GEOMETRIES = {
    "s10_o3": (10, 3),
    "s8_o2": (8, 2),
}


def _chain_cosine_margin(items: Sequence[Any], *, seed: int) -> dict[str, float | int]:
    true_values = []
    shuffled_values = []
    usable_graphs = 0
    for item in items:
        weights = _relation_matrices(item)[0]
        target, source = np.nonzero(weights > 0)
        if target.size == 0:
            continue
        tokens = item.node_histograms
        permutation = _shuffle_permutation(len(tokens), seed=seed + item.index * 1009)
        shuffled = tokens[permutation]

        def cosine(values: np.ndarray) -> np.ndarray:
            left = values[target]
            right = values[source]
            dot = np.sum(left * right, axis=1)
            return dot / np.maximum(
                np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1),
                1e-12,
            )

        true_values.extend(cosine(tokens).tolist())
        shuffled_values.extend(cosine(shuffled).tolist())
        usable_graphs += 1
    true = np.asarray(true_values, dtype=np.float64)
    shuffled = np.asarray(shuffled_values, dtype=np.float64)
    return {
        "usable_graphs": usable_graphs,
        "chain_edges": int(len(true)),
        "true_mean": float(true.mean()) if true.size else 0.0,
        "shuffled_mean": float(shuffled.mean()) if shuffled.size else 0.0,
        "true_minus_shuffled": float(true.mean() - shuffled.mean()) if true.size else 0.0,
    }


def _summary(items: Sequence[Any], *, seed: int) -> dict[str, Any]:
    patches = np.asarray([len(item.vectors) for item in items], dtype=np.float64)
    chain_edges = np.asarray(
        [np.count_nonzero(_relation_matrices(item)[0]) for item in items],
        dtype=np.float64,
    )
    partial = np.asarray(
        [item.sampling["partial_components"] for item in items], dtype=np.float64
    )
    components = np.asarray(
        [item.sampling["components"] for item in items], dtype=np.float64
    )
    edge = np.asarray(
        [item.sampling["edge_coverage"] for item in items], dtype=np.float64
    )
    node = np.asarray(
        [item.sampling["node_coverage"] for item in items], dtype=np.float64
    )
    nonchain = []
    for item in items:
        previous, following, overlap, _slot = _relation_matrices(item)
        nonchain.append(
            np.count_nonzero(overlap)
            > np.count_nonzero(previous) + np.count_nonzero(following)
        )
    return {
        "graph_count": len(items),
        "mean_patches": float(patches.mean()),
        "patch_quantiles": {
            str(q): float(np.quantile(patches, q)) for q in (0.1, 0.5, 0.9)
        },
        "single_patch_fraction": float(np.mean(patches == 1)),
        "graphs_with_chain_fraction": float(np.mean(chain_edges > 0)),
        "mean_directed_chain_edges": float(chain_edges.mean()),
        "partial_component_fraction": float(partial.sum() / max(components.sum(), 1.0)),
        "graphs_with_partial_component_fraction": float(np.mean(partial > 0)),
        "mean_edge_coverage": float(edge.mean()),
        "edge_coverage_p10": float(np.quantile(edge, 0.1)),
        "mean_node_coverage": float(node.mean()),
        "nonchain_overlap_graph_fraction": float(np.mean(nonchain)),
        "attribute_chain_binding": _chain_cosine_margin(items, seed=seed),
    }


def _decision(s10: dict[str, Any], s8: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "single_patch_relative_reduction_ge_20pct": bool(
            s8["single_patch_fraction"]
            <= 0.8 * max(s10["single_patch_fraction"], 1e-12)
        ),
        "chain_graph_fraction_noninferior": bool(
            s8["graphs_with_chain_fraction"] >= s10["graphs_with_chain_fraction"] - 1e-12
        ),
        "mean_chain_edges_relative_gain_ge_20pct": bool(
            s8["mean_directed_chain_edges"]
            >= 1.2 * max(s10["mean_directed_chain_edges"], 1e-12)
        ),
        "edge_coverage_noninferior_1pt": bool(
            s8["mean_edge_coverage"] >= s10["mean_edge_coverage"] - 0.01
        ),
        "partial_component_fraction_noninferior": bool(
            s8["partial_component_fraction"]
            <= s10["partial_component_fraction"] + 1e-12
        ),
        "attribute_true_minus_shuffled_positive": bool(
            s8["attribute_chain_binding"]["true_minus_shuffled"] > 0.0
        ),
    }
    passed = all(checks.values())
    return {
        "checks": checks,
        "gate": passed,
        "decision": "S8_O2_READY_FOR_FIXED_CLASSIFICATION" if passed else "KEEP_S10_O3_NO_GEOMETRY_SWITCH",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    graphs, _labels, features, metadata = load_tud(args.dataset, args.dataset_root)
    results = {}
    for name, (patch_size, overlap) in GEOMETRIES.items():
        items = []
        for index, (graph, node_features) in enumerate(zip(graphs, features)):
            items.append(
                prepare_graph(
                    index,
                    graph,
                    0,
                    node_features,
                    patch_size=patch_size,
                    overlap=overlap,
                    retained_beam=args.retained_beam,
                    edge_capacity_multiplier=args.edge_capacity_multiplier,
                    seed=args.cover_seed,
                )
            )
            if (index + 1) % 500 == 0 or index + 1 == len(graphs):
                print(f"{name} {index + 1}/{len(graphs)}", flush=True)
        results[name] = _summary(items, seed=args.shuffle_seed)
    return {
        "protocol": PROTOCOL,
        "dataset": metadata,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "geometries": results,
        "decision": _decision(results["s10_o3"], results["s8_o2"]),
        "seconds": time.time() - started,
    }


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Beam8/NCI1 geometry feasibility audit",
        "",
        f"> 协议：`{payload['protocol']}`  ",
        f"> 判定：`{payload['decision']['decision']}`",
        "",
        "| metric | s10/o3 | s8/o2 |",
        "|---|---:|---:|",
    ]
    metrics = (
        "mean_patches",
        "single_patch_fraction",
        "graphs_with_chain_fraction",
        "mean_directed_chain_edges",
        "partial_component_fraction",
        "graphs_with_partial_component_fraction",
        "mean_edge_coverage",
        "edge_coverage_p10",
        "mean_node_coverage",
        "nonchain_overlap_graph_fraction",
    )
    for metric in metrics:
        lines.append(
            f"| {metric} | {payload['geometries']['s10_o3'][metric]:.4f} | {payload['geometries']['s8_o2'][metric]:.4f} |"
        )
    lines.extend(["", "## Attribute relation binding", ""])
    for name in ("s10_o3", "s8_o2"):
        row = payload["geometries"][name]["attribute_chain_binding"]
        lines.append(
            f"- {name}：TRUE `{row['true_mean']:.4f}`，SHUFFLED `{row['shuffled_mean']:.4f}`，margin `{row['true_minus_shuffled']:+.4f}`，usable graphs `{row['usable_graphs']}`。"
        )
    lines.extend(["", "## Frozen checks", ""])
    for name, value in payload["decision"]["checks"].items():
        lines.append(f"- {name}：`{value}`；")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- 本审计没有使用 graph labels，也没有训练分类器或字典。",
            "- 只有全部冻结 checks 通过才允许用 s8/o2 重跑分类；不得以分类结果反向选择 geometry。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="NCI1")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--retained-beam", type=int, default=8)
    parser.add_argument("--edge-capacity-multiplier", type=float, default=1.5)
    parser.add_argument("--cover-seed", type=int, default=20260813)
    parser.add_argument("--shuffle-seed", type=int, default=731421)
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

