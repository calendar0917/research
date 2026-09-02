"""Label-free feasibility audit for Beam8 patch-to-node incidence features."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from .attributed_beam8 import attributed_automorphism_orbits
from .data_tud import load_tud
from .run_attributed_beam8_mutagenicity import prepare_attributed_beam_graph
from .run_attributed_beam8_mutagenicity_invariance import _graph
from .run_beam8_mutagenicity_typed_feasibility import _typed_adjacency
from .run_beam8_nci1_chain_classification import (
    DEFAULT_ROOT,
    ROOT,
    _adjacency,
    _atomic_json,
    _atomic_text,
    _relation_matrices,
)
from .run_luyin14_edge_aware_joint import _load_edge_pyg


PROTOCOL = "tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_NODE_INCIDENCE_FEASIBILITY_PROTOCOL_20260814.md"
DEFAULT_JSON = ROOT / "tracks/ksvd/results/luyin14/attributed_beam8_node_incidence_feasibility_20260814.json"
DEFAULT_REPORT = ROOT / "tracks/ksvd/results/luyin14/ATTRIBUTED_BEAM8_NODE_INCIDENCE_FEASIBILITY_20260814.md"


def node_incidence_features(
    item: Any,
    n_nodes: int,
    patch_size: int = 8,
    *,
    tokens: np.ndarray | None = None,
    include_relation: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    tokens = (
        np.concatenate([item.vectors, item.node_histograms], axis=1)
        if tokens is None
        else np.asarray(tokens, dtype=np.float64)
    )
    if tokens.shape[0] != len(item.slot_nodes):
        raise ValueError("one token row is required for every Beam8 patch")
    previous, following, overlap, slot = _relation_matrices(item)
    relation_degree = np.stack(
        [previous.sum(1), following.sum(1), overlap.sum(1), slot.sum(1)], axis=1
    )
    token_sum = np.zeros((n_nodes, tokens.shape[1]), dtype=np.float64)
    token_max = np.full((n_nodes, tokens.shape[1]), -np.inf, dtype=np.float64)
    slots = np.zeros((n_nodes, patch_size), dtype=np.float64)
    centers = np.zeros((n_nodes, 1), dtype=np.float64)
    positions = np.zeros((n_nodes, item.position_features.shape[1]), dtype=np.float64)
    relations = np.zeros((n_nodes, relation_degree.shape[1]), dtype=np.float64)
    counts = np.zeros(n_nodes, dtype=np.float64)
    chain_active = np.zeros(n_nodes, dtype=bool)
    active_patch = (previous.sum(0) + previous.sum(1) + following.sum(0) + following.sum(1)) > 1e-12
    for patch_index, nodes in enumerate(item.slot_nodes):
        for slot_index, node in enumerate(nodes):
            node = int(node)
            counts[node] += 1.0
            token_sum[node] += tokens[patch_index]
            token_max[node] = np.maximum(token_max[node], tokens[patch_index])
            slots[node, slot_index] += 1.0
            centers[node, 0] += float(node == int(item.centers[patch_index]))
            positions[node] += item.position_features[patch_index]
            relations[node] += relation_degree[patch_index]
            chain_active[node] |= bool(active_patch[patch_index])
    covered = counts > 0
    denominator = np.maximum(counts[:, None], 1.0)
    token_mean = token_sum / denominator
    token_max[~covered] = 0.0
    output = np.concatenate(
        [
            token_mean,
            token_max,
            slots / denominator if include_relation else np.zeros_like(slots),
            centers / denominator if include_relation else np.zeros_like(centers),
            positions / denominator if include_relation else np.zeros_like(positions),
            relations / denominator if include_relation else np.zeros_like(relations),
            np.log1p(counts)[:, None],
            covered.astype(np.float64)[:, None],
        ],
        axis=1,
    )
    diagnostics = {
        "covered": covered,
        "counts": counts,
        "chain_active": chain_active,
        "feature_dim": int(output.shape[1]),
    }
    return output, diagnostics


def _same_type_pair_stats(features: np.ndarray, node_features: np.ndarray) -> tuple[int, int]:
    node_types = np.argmax(node_features, axis=1)
    distinguishable = total = 0
    for node_type in np.unique(node_types):
        nodes = np.flatnonzero(node_types == node_type)
        for left_index, left in enumerate(nodes):
            for right in nodes[left_index + 1 :]:
                total += 1
                distinguishable += int(not np.array_equal(features[left], features[right]))
    return distinguishable, total


def _readout(values: np.ndarray) -> np.ndarray:
    return np.concatenate([values.sum(0), values.max(0)])


def orbit_safe_features(
    values: np.ndarray,
    typed: np.ndarray,
    node_features: np.ndarray,
    *,
    canonical_node_features: np.ndarray | None = None,
    canonical_node_types: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    canonical = np.asarray(
        node_features if canonical_node_features is None else canonical_node_features
    )
    node_types = np.asarray(
        np.argmax(canonical, axis=1)
        if canonical_node_types is None
        else canonical_node_types,
        dtype=np.int64,
    )
    orbits = attributed_automorphism_orbits(typed, node_types)
    output = values.copy()
    for orbit in orbits:
        nodes = np.asarray(orbit, dtype=np.int64)
        output[nodes] = values[nodes].mean(axis=0, keepdims=True)
    return output, orbits


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    graphs, _labels, features, metadata = load_tud("Mutagenicity", args.dataset_root)
    raw, _raw_labels, _classes, edge_dim = _load_edge_pyg("Mutagenicity", args.dataset_root)
    coverage = []
    full_coverage = []
    mean_incidence = []
    covered_incidence = []
    multi_patch = []
    chain_active = []
    pair_distinguishable = pair_total = 0
    feature_dim = None
    invariance_rows = []
    orbit_sizes = []
    audit_graphs = min(args.graphs, len(graphs))
    for index, (graph, node_features, data) in enumerate(zip(graphs, features, raw)):
        typed = _typed_adjacency(data, edge_dim)
        item = prepare_attributed_beam_graph(
            index, graph, 0, node_features, typed, edge_dim=edge_dim
        )
        incidence, diagnostic = node_incidence_features(item, graph.n)
        orbit_safe, orbits = orbit_safe_features(incidence, typed, np.asarray(node_features))
        orbit_sizes.extend(len(orbit) for orbit in orbits)
        feature_dim = diagnostic["feature_dim"]
        covered = diagnostic["covered"]
        counts = diagnostic["counts"]
        coverage.append(float(np.mean(covered)))
        full_coverage.append(bool(np.all(covered)))
        mean_incidence.append(float(np.mean(counts)))
        covered_incidence.append(float(np.mean(counts[covered])) if np.any(covered) else 0.0)
        multi_patch.append(float(np.mean(counts >= 2)))
        chain_active.append(float(np.mean(diagnostic["chain_active"])))
        different, total = _same_type_pair_stats(orbit_safe, np.asarray(node_features))
        pair_distinguishable += different
        pair_total += total
        if index < audit_graphs:
            adjacency = _adjacency(graph)
            base_readout = _readout(orbit_safe)
            for trial in range(args.permutations):
                permutation = np.random.default_rng(
                    940000 + index * 1009 + trial
                ).permutation(graph.n)
                relabeled_item = prepare_attributed_beam_graph(
                    index,
                    _graph(adjacency[np.ix_(permutation, permutation)]),
                    0,
                    np.asarray(node_features)[permutation],
                    typed[np.ix_(permutation, permutation)],
                    edge_dim=edge_dim,
                )
                relabeled, _relabeled_diagnostic = node_incidence_features(
                    relabeled_item, graph.n
                )
                relabeled_orbit_safe, _relabeled_orbits = orbit_safe_features(
                    relabeled,
                    typed[np.ix_(permutation, permutation)],
                    np.asarray(node_features)[permutation],
                )
                invariance_rows.append(
                    {
                        "graph_index": index,
                        "permutation": trial,
                        "direct_node_equivariance": np.array_equal(
                            relabeled, incidence[permutation]
                        ),
                        "node_equivariance": np.allclose(
                            relabeled_orbit_safe,
                            orbit_safe[permutation],
                            atol=1e-12,
                            rtol=0,
                        ),
                        "graph_readout": np.allclose(
                            _readout(relabeled_orbit_safe), base_readout, atol=1e-12, rtol=0
                        ),
                    }
                )
        if (index + 1) % 500 == 0 or index + 1 == len(graphs):
            print(f"{index + 1}/{len(graphs)}", flush=True)
    invariance = {
        key: float(np.mean([row[key] for row in invariance_rows]))
        for key in ("direct_node_equivariance", "node_equivariance", "graph_readout")
    }
    summary = {
        "feature_dim": int(feature_dim or 0),
        "mean_node_coverage": float(np.mean(coverage)),
        "full_coverage_graph_fraction": float(np.mean(full_coverage)),
        "mean_incidence_per_node": float(np.mean(mean_incidence)),
        "mean_incidence_per_covered_node": float(np.mean(covered_incidence)),
        "multi_patch_node_fraction": float(np.mean(multi_patch)),
        "chain_active_node_fraction": float(np.mean(chain_active)),
        "same_atom_pair_disambiguation": float(pair_distinguishable / max(pair_total, 1)),
        "non_singleton_orbit_node_fraction": float(
            np.sum([size for size in orbit_sizes if size > 1]) / max(np.sum(orbit_sizes), 1)
        ),
    }
    checks = {
        "usable_node_coverage": summary["mean_node_coverage"] >= 0.70,
        "nontrivial_multi_patch": summary["multi_patch_node_fraction"] >= 0.10,
        "nontrivial_chain_context": summary["chain_active_node_fraction"] >= 0.20,
        "atom_complement": summary["same_atom_pair_disambiguation"] >= 0.50,
        "node_equivariance": invariance["node_equivariance"] == 1.0,
        "graph_readout_invariance": invariance["graph_readout"] == 1.0,
    }
    ready = all(checks.values())
    failures = [row for row in invariance_rows if not all((row["node_equivariance"], row["graph_readout"]))][:20]
    return {
        "protocol": PROTOCOL,
        "dataset": metadata | {"edge_dim": int(edge_dim)},
        "config": {"graphs": audit_graphs, "permutations": args.permutations},
        "summary": summary,
        "invariance": invariance,
        "checks": checks,
        "failure_examples": failures,
        "decision": (
            "BEAM8_NODE_INCIDENCE_READY_FOR_LOW_CAPACITY_CLASSIFICATION"
            if ready
            else "BEAM8_NODE_INCIDENCE_REQUIRES_SYMMETRY_OR_COVERAGE_FIX"
        ),
        "seconds": time.time() - started,
    }


def render(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Attributed Beam8 node-incidence feasibility audit",
        "",
        f"> 协议：`{payload['protocol']}`  ",
        f"> 判定：`{payload['decision']}`",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key, value in summary.items():
        lines.append(f"| {key} | {value:.6f} |" if isinstance(value, float) else f"| {key} | {value} |")
    lines.extend(["", "## Relabel audit", ""])
    for key, value in payload["invariance"].items():
        lines.append(f"- {key}：`{value:.6f}`；")
    lines.extend(["", "## Checks", ""])
    for key, value in payload["checks"].items():
        lines.append(f"- {key}：`{value}`；")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- 本轮未使用分类标签；通过只授权低容量 GINE fusion screen。",
            "- graph readout invariant 但 node equivariance 不通过时，不能直接接节点级 GNN。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--graphs", type=int, default=512)
    parser.add_argument("--permutations", type=int, default=3)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    payload = run(args)
    _atomic_json(args.json, payload)
    _atomic_text(args.report, render(payload))
    print(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
