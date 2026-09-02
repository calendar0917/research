"""Audit natural Beam8 chain extension for rare typed edges."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from .data_tud import load_tud
from .run_beam8_mutagenicity_typed_feasibility import _typed_adjacency
from .run_luyin14_edge_aware_joint import _load_edge_pyg
from .run_beam8_nci1_chain_classification import (
    DEFAULT_ROOT, ROOT, _atomic_json, _atomic_text,
    _components, _adjacency, _component_seed,
)
from .run_beam8_mutagenicity_fair95_completion import _covered_by_type
from .run_luyin14_route import _true_edges, _covered_edges
from .canonical_slots import reorder_cover_structurally
from .global_stable_ids import compute_global_wl_ids, reorder_by_stable_ids
from .marginal_candidate_cover import sample_marginal_candidate_cover
from .overlap_cover import _make_cover, _make_patch, patch_budget


PROTOCOL = "tracks/ksvd/docs/KSVD_BEAM8_MUTAGENICITY_MAXCHAIN_PROTOCOL_20260813.md"
DEFAULT_JSON = ROOT / "tracks/ksvd/results/luyin14/beam8_mutagenicity_maxchain_20260813.json"
DEFAULT_REPORT = ROOT / "tracks/ksvd/results/luyin14/BEAM8_MUTAGENICITY_MAXCHAIN_20260813.md"


def _cover_for_components(graph: Any, *, patch_size: int, overlap: int) -> tuple[Any, Any]:
    adjacency = _adjacency(graph)
    base_patches, base_segments = [], []
    max_patches, max_segments = [], []
    for segment, component in enumerate(_components(adjacency)):
        nodes = np.asarray(component, dtype=np.int64)
        local_source = adjacency[np.ix_(nodes, nodes)]
        stable = compute_global_wl_ids(local_source)
        local = reorder_by_stable_ids(local_source, stable)
        ordered_global = nodes[np.asarray(stable.order, dtype=np.int64)]
        if len(nodes) <= patch_size:
            local_nodes = tuple(range(len(nodes)))
            local_patches = [_make_patch(local, local_nodes, 0)]
            base_count = 1
        else:
            base_count = patch_budget(
                local, patch_size=patch_size, target_overlap=overlap,
                edge_capacity_multiplier=1.5,
            )
            cover = sample_marginal_candidate_cover(
                local,
                np.random.default_rng(_component_seed(local, 20260813)),
                n_patches=128,
                patch_size=patch_size,
                target_overlap=overlap,
                retained_beam=8,
                candidate_restarts=1,
                allow_partial=True,
            )
            cover, _audit = reorder_cover_structurally(local, cover, "rooted_canonical")
            local_patches = list(cover.patches)
        for local_index, patch in enumerate(local_patches):
            global_nodes = tuple(int(ordered_global[node]) for node in patch.node_ids)
            entry = (global_nodes, int(ordered_global[patch.center]))
            max_patches.append(entry)
            max_segments.append(segment)
            if local_index < base_count:
                base_patches.append(entry)
                base_segments.append(segment)
    return (
        _as_patch_cover(adjacency, base_patches, base_segments, "beam8_base"),
        _as_patch_cover(adjacency, max_patches, max_segments, "beam8_maxchain"),
    )


def _as_patch_cover(
    adjacency: np.ndarray,
    patches: list[tuple[tuple[int, ...], int]],
    segments: list[int],
    method: str,
) -> Any:
    ordered = [_make_patch(adjacency, nodes, center) for nodes, center in patches]
    return _make_cover(method, ordered, segments, [None] * len(ordered), [0] * len(ordered))


def _chain_edge_count(cover: Any) -> int:
    return int(
        sum(
            left == right
            for left, right in zip(cover.segment_ids, cover.segment_ids[1:])
        )
    )


def _summary(rows: list[dict[str, Any]], edge_dim: int) -> dict[str, Any]:
    base_patches = np.asarray([row["base_patches"] for row in rows], dtype=np.float64)
    max_patches = np.asarray([row["max_patches"] for row in rows], dtype=np.float64)
    base_chain = np.asarray([row["base_chain_edges"] for row in rows])
    max_chain = np.asarray([row["max_chain_edges"] for row in rows])
    base_recall = np.asarray([row["base_edge_recall"] for row in rows])
    max_recall = np.asarray([row["max_edge_recall"] for row in rows])
    base_node = np.asarray([row["base_node_coverage"] for row in rows])
    max_node = np.asarray([row["max_node_coverage"] for row in rows])
    base_type = np.sum([row["base_type_covered"] for row in rows], axis=0)
    max_type = np.sum([row["max_type_covered"] for row in rows], axis=0)
    total_type = np.sum([row["type_total"] for row in rows], axis=0)
    rare_graphs = sum(row["rare_total"] > 0 for row in rows)
    rare_hits = sum(row["rare_covered"] > 0 for row in rows)
    return {
        "graph_count": len(rows),
        "mean_base_patches": float(base_patches.mean()),
        "mean_max_patches": float(max_patches.mean()),
        "mean_extra_patches": float((max_patches - base_patches).mean()),
        "mean_extra_fraction": float(np.mean((max_patches - base_patches) / np.maximum(base_patches, 1))),
        "graphs_with_chain_base": float(np.mean(base_chain > 0)),
        "graphs_with_chain_max": float(np.mean(max_chain > 0)),
        "mean_chain_edges_base": float(base_chain.mean()),
        "mean_chain_edges_max": float(max_chain.mean()),
        "mean_edge_recall_base": float(base_recall.mean()),
        "mean_edge_recall_max": float(max_recall.mean()),
        "mean_node_coverage_base": float(base_node.mean()),
        "mean_node_coverage_max": float(max_node.mean()),
        "type_recall_base": (base_type / np.maximum(total_type, 1)).tolist(),
        "type_recall_max": (max_type / np.maximum(total_type, 1)).tolist(),
        "rare_graph_coverage_max": float(rare_hits / max(rare_graphs, 1)),
        "new_type_edges": (max_type - base_type).tolist(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    graphs, labels, features, metadata = load_tud("Mutagenicity", args.dataset_root)
    raw, raw_labels, _classes, edge_dim = _load_edge_pyg("Mutagenicity", args.dataset_root)
    rows = []
    for index, (graph, data) in enumerate(zip(graphs, raw)):
        adjacency = _adjacency(graph)
        base_cover, max_cover = _cover_for_components(
            graph, patch_size=args.patch_size, overlap=args.overlap
        )
        typed = _typed_adjacency(data, edge_dim)
        truth = _true_edges(adjacency)
        def metrics(cover):
            covered = _covered_edges(adjacency, cover.patches)
            nodes = set().union(*(set(p.node_ids) for p in cover.patches))
            total, by_type = _covered_by_type(cover, typed, edge_dim)
            return len(covered) / max(len(truth), 1), len(nodes) / max(adjacency.shape[0], 1), by_type, total
        be, bn, bc, total = metrics(base_cover)
        me, mn, mc, _ = metrics(max_cover)
        rare_total = int(total[-1]); rare_covered = int(mc[-1])
        rows.append({"base_patches": len(base_cover.patches), "max_patches": len(max_cover.patches),
                     "base_chain_edges": _chain_edge_count(base_cover),
                     "max_chain_edges": _chain_edge_count(max_cover),
                     "base_edge_recall": be, "max_edge_recall": me, "base_node_coverage": bn, "max_node_coverage": mn,
                     "base_type_covered": bc.tolist(), "max_type_covered": mc.tolist(), "type_total": total.tolist(),
                     "rare_total": rare_total, "rare_covered": rare_covered})
        if (index + 1) % 500 == 0 or index + 1 == len(graphs): print(f"{index+1}/{len(graphs)}", flush=True)
    summary = _summary(rows, edge_dim)
    checks = {
        "rare_recall_ge_80": summary["type_recall_max"][-1] >= 0.8,
        "rare_graph_coverage_ge_80": summary["rare_graph_coverage_max"] >= 0.8,
        "extra_fraction_le_100": summary["mean_extra_fraction"] <= 1.0,
        "chain_fraction_noninferior": summary["graphs_with_chain_max"] >= summary["graphs_with_chain_base"],
    }
    return {"protocol": PROTOCOL, "dataset": metadata | {"edge_dim": edge_dim}, "summary": summary,
            "checks": checks, "decision": "MAXCHAIN_READY_FOR_TYPED_CLASSIFICATION" if all(checks.values()) else "MAXCHAIN_COST_OR_COVERAGE_BELOW_GATE",
            "seconds": time.time() - started}


def render(payload: dict[str, Any]) -> str:
    s = payload["summary"]
    lines = ["# Beam8/Mutagenicity MAXCHAIN audit", "", f"> 协议：`{payload['protocol']}`  ", f"> 判定：`{payload['decision']}`", "",
             "| metric | value |", "|---|---:|"]
    for key in ("mean_base_patches", "mean_max_patches", "mean_extra_patches", "mean_extra_fraction", "graphs_with_chain_base", "graphs_with_chain_max", "mean_chain_edges_base", "mean_chain_edges_max", "mean_edge_recall_base", "mean_edge_recall_max", "mean_node_coverage_base", "mean_node_coverage_max", "rare_graph_coverage_max"):
        lines.append(f"| {key} | {s[key]:.4f} |")
    lines.extend(["", f"- type recall BASE：`{s['type_recall_base']}`；", f"- type recall MAXCHAIN：`{s['type_recall_max']}`；", f"- new type edges：`{s['new_type_edges']}`；", "", "## Checks", ""])
    for k,v in payload["checks"].items(): lines.append(f"- {k}：`{v}`；")
    return "\n".join(lines)+"\n"


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument('--dataset-root',type=Path,default=DEFAULT_ROOT); p.add_argument('--patch-size',type=int,default=8); p.add_argument('--overlap',type=int,default=2); p.add_argument('--json',type=Path,default=DEFAULT_JSON); p.add_argument('--report',type=Path,default=DEFAULT_REPORT); a=p.parse_args(); payload=run(a); _atomic_json(a.json,payload); _atomic_text(a.report,render(payload)); print(a.report); return 0

if __name__=='__main__': raise SystemExit(main())
