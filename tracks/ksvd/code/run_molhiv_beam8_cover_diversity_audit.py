"""Audit whether MolHIV Beam8 covers actually vary across sampling seeds.

Only molecules from the OGB official-train split are materialized.  The audit
compares the unordered patch sets produced by the marginal Beam sampler with a
robust randomized-BFS reference; labels are not read or used.
"""
from __future__ import annotations

import argparse
import json
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .attributed_beam8 import (
    attributed_component_signature,
    attributed_stable_order,
)
from .data_molhiv import _edges_from_pyg, _patch_torch_load_weights_only
from .molhiv_beam8_incidence import (
    _atomic_number_one_hot,
    _feature_row_colors,
    typed_bond_adjacency,
)
from .overlap_cover import patch_budget
from .run_attributed_beam8_mutagenicity import prepare_attributed_beam_graph
from .run_beam8_nci1_chain_classification import (
    _adjacency,
    _component_seed,
    _components,
)


DEFAULT_JSON = Path(
    "tracks/ksvd/results/molhiv/"
    "molhiv_beam8_cover_diversity_audit_20260816.json"
)
DEFAULT_REPORT = Path(
    "tracks/ksvd/docs/KSVD_MOLHIV_BEAM8_COVER_DIVERSITY_AUDIT_20260816.md"
)


def _edge_feature_map(graph_dict: dict[str, Any]) -> dict[tuple[int, int], np.ndarray]:
    edge_index = np.asarray(graph_dict["edge_index"], dtype=np.int64)
    edge_features = np.asarray(graph_dict["edge_feat"], dtype=np.int64)
    output: dict[tuple[int, int], np.ndarray] = {}
    for offset, (left, right) in enumerate(zip(edge_index[0], edge_index[1])):
        left_value, right_value = int(left), int(right)
        if left_value == right_value:
            continue
        key = tuple(sorted((left_value, right_value)))
        output.setdefault(key, edge_features[offset].copy())
    return output


def _cover_signature(slot_nodes: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    """Ignore patch order and within-patch slot order for diversity scoring."""

    return tuple(sorted(tuple(sorted(int(node) for node in patch)) for patch in slot_nodes))


def _patch_jaccard(
    left: Sequence[Sequence[int]], right: Sequence[Sequence[int]]
) -> float:
    left_set = set(_cover_signature(left))
    right_set = set(_cover_signature(right))
    union = left_set | right_set
    return float(len(left_set & right_set) / max(len(union), 1))


def _occurrences(
    slot_nodes: Sequence[Sequence[int]],
    n_nodes: int,
    edges: Sequence[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    node_counts = np.zeros(n_nodes, dtype=np.float64)
    edge_counts = np.zeros(len(edges), dtype=np.float64)
    for patch in slot_nodes:
        nodes = set(int(node) for node in patch)
        node_counts[list(nodes)] += 1.0
        for edge_index, (left, right) in enumerate(edges):
            edge_counts[edge_index] += float(left in nodes and right in nodes)
    scale = float(max(len(slot_nodes), 1))
    return node_counts / scale, edge_counts / scale


def summarize_cover_family(
    covers: Sequence[Sequence[Sequence[int]]],
    *,
    n_nodes: int,
    edges: Sequence[tuple[int, int]],
) -> dict[str, float | int]:
    """Return graph-local seed diversity metrics for one cover family."""

    signatures = [_cover_signature(cover) for cover in covers]
    occurrence = [_occurrences(cover, n_nodes, edges) for cover in covers]
    pair_exact: list[float] = []
    pair_jaccard: list[float] = []
    node_l1: list[float] = []
    edge_l1: list[float] = []
    for left, right in combinations(range(len(covers)), 2):
        pair_exact.append(float(signatures[left] == signatures[right]))
        pair_jaccard.append(_patch_jaccard(covers[left], covers[right]))
        node_l1.append(float(np.mean(np.abs(occurrence[left][0] - occurrence[right][0]))))
        edge_l1.append(
            float(np.mean(np.abs(occurrence[left][1] - occurrence[right][1])))
            if edges
            else 0.0
        )
    node_matrix = np.stack([values[0] for values in occurrence])
    edge_matrix = (
        np.stack([values[1] for values in occurrence])
        if edges
        else np.empty((len(covers), 0), dtype=np.float64)
    )
    return {
        "unique_covers": len(set(signatures)),
        "pair_exact_rate": float(np.mean(pair_exact)) if pair_exact else 1.0,
        "pair_patch_jaccard": float(np.mean(pair_jaccard)) if pair_jaccard else 1.0,
        "pair_node_occurrence_l1": float(np.mean(node_l1)) if node_l1 else 0.0,
        "pair_edge_occurrence_l1": float(np.mean(edge_l1)) if edge_l1 else 0.0,
        "variable_node_fraction": float(np.mean(np.ptp(node_matrix, axis=0) > 0)),
        "variable_edge_fraction": (
            float(np.mean(np.ptp(edge_matrix, axis=0) > 0)) if edges else 0.0
        ),
    }


def _canonical_components(
    adjacency: np.ndarray, typed: np.ndarray, canonical_types: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    components = []
    for raw_component in _components(adjacency):
        nodes = np.asarray(raw_component, dtype=np.int64)
        local_typed = typed[np.ix_(nodes, nodes)]
        local_types = canonical_types[nodes]
        signature = attributed_component_signature(local_typed, local_types)
        stable = attributed_stable_order(local_typed, local_types)
        ordered_global = nodes[np.asarray(stable.order, dtype=np.int64)]
        binary = adjacency[np.ix_(ordered_global, ordered_global)]
        components.append((signature, ordered_global, binary))
    components.sort(key=lambda item: item[0])
    return [(ordered, binary) for _signature, ordered, binary in components]


def _random_bfs_patch(
    adjacency: np.ndarray, root: int, rng: np.random.Generator, patch_size: int
) -> tuple[int, ...]:
    """Collect one connected patch with randomized BFS tie-breaking."""

    nodes = [int(root)]
    seen = {int(root)}
    queue = [int(root)]
    cursor = 0
    while cursor < len(queue) and len(nodes) < patch_size:
        current = queue[cursor]
        cursor += 1
        neighbors = np.asarray(np.flatnonzero(adjacency[current]), dtype=np.int64)
        for raw_neighbor in neighbors[rng.permutation(len(neighbors))]:
            neighbor = int(raw_neighbor)
            if neighbor in seen:
                continue
            seen.add(neighbor)
            queue.append(neighbor)
            nodes.append(neighbor)
            if len(nodes) == patch_size:
                break
    if len(nodes) != patch_size:
        raise RuntimeError("randomized BFS did not span the connected component")
    return tuple(nodes)


def _random_bfs_slots(
    adjacency: np.ndarray,
    typed: np.ndarray,
    canonical_types: np.ndarray,
    *,
    seed: int,
    patch_size: int,
    overlap: int,
    edge_capacity_multiplier: float,
) -> tuple[tuple[int, ...], ...]:
    slots: list[tuple[int, ...]] = []
    for ordered_global, binary in _canonical_components(adjacency, typed, canonical_types):
        if len(ordered_global) <= patch_size:
            slots.append(tuple(int(node) for node in ordered_global))
            continue
        budget = patch_budget(
            binary,
            patch_size=patch_size,
            target_overlap=overlap,
            edge_capacity_multiplier=edge_capacity_multiplier,
        )
        rng = np.random.default_rng(_component_seed(binary, seed))
        roots: list[int] = []
        while len(roots) < budget:
            roots.extend(int(node) for node in rng.permutation(len(binary)))
        for root in roots[:budget]:
            patch = _random_bfs_patch(binary, root, rng, patch_size)
            slots.append(tuple(int(ordered_global[node]) for node in patch))
    return tuple(slots)


def _aggregate(rows: Sequence[dict[str, Any]], family: str) -> dict[str, float]:
    selected = [row[family] for row in rows]
    eligible = [row[family] for row in rows if row["eligible_for_stochastic_cover"]]

    def mean(key: str, values: Sequence[dict[str, Any]]) -> float:
        return float(np.mean([float(value[key]) for value in values])) if values else 0.0

    return {
        "graphs": float(len(selected)),
        "eligible_graphs": float(len(eligible)),
        "changed_graph_fraction_all": (
            float(np.mean([value["unique_covers"] > 1 for value in selected]))
            if selected
            else 0.0
        ),
        "changed_graph_fraction_eligible": (
            float(np.mean([value["unique_covers"] > 1 for value in eligible]))
            if eligible
            else 0.0
        ),
        "mean_unique_covers_eligible": mean("unique_covers", eligible),
        "pair_exact_rate_eligible": mean("pair_exact_rate", eligible),
        "pair_patch_jaccard_eligible": mean("pair_patch_jaccard", eligible),
        "pair_node_occurrence_l1_eligible": mean(
            "pair_node_occurrence_l1", eligible
        ),
        "pair_edge_occurrence_l1_eligible": mean(
            "pair_edge_occurrence_l1", eligible
        ),
        "variable_node_fraction_eligible": mean("variable_node_fraction", eligible),
        "variable_edge_fraction_eligible": mean("variable_edge_fraction", eligible),
    }


def classify(beam: dict[str, float]) -> str:
    changed = beam["changed_graph_fraction_eligible"]
    jaccard = beam["pair_patch_jaccard_eligible"]
    if changed < 0.10 or jaccard > 0.98:
        return "BEAM_SEED_DIVERSITY_INSUFFICIENT"
    if changed >= 0.50 and jaccard <= 0.90:
        return "BEAM_SEED_DIVERSITY_MEANINGFUL"
    return "BEAM_SEED_DIVERSITY_LIMITED"


def render_report(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    beam = payload["summary"]["beam"]
    random_bfs = payload["summary"]["random_bfs"]
    lines = [
        "# MolHIV Beam8 cover seed-diversity audit",
        "",
        "> 日期：2026-08-16  ",
        "> 范围：OGB official-train only；不读取 official-valid/test 分子。",
        "",
        "## 结论",
        "",
        f"**{decision}**",
        "",
        "该审计只比较真实 `slot_nodes` patch 集合；token shuffle、chain shuffle "
        "和 mapping shuffle 均不计为 cover 多样性。",
        "",
        "## 汇总（仅统计至少一个连通分量大于 patch size 的分子）",
        "",
        "| cover | changed graphs | unique covers | exact pair | patch Jaccard | node occurrence L1 | edge occurrence L1 | variable nodes | variable edges |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary in (
        ("Beam8", beam),
        ("random BFS", random_bfs),
    ):
        lines.append(
            f"| {name} | {summary['changed_graph_fraction_eligible']:.4f} | "
            f"{summary['mean_unique_covers_eligible']:.3f} | "
            f"{summary['pair_exact_rate_eligible']:.4f} | "
            f"{summary['pair_patch_jaccard_eligible']:.4f} | "
            f"{summary['pair_node_occurrence_l1_eligible']:.4f} | "
            f"{summary['pair_edge_occurrence_l1_eligible']:.4f} | "
            f"{summary['variable_node_fraction_eligible']:.4f} | "
            f"{summary['variable_edge_fraction_eligible']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 协议边界",
            "",
            f"- 抽样 official-train 分子：`{payload['protocol']['sampled_graphs']}`。",
            f"- cover seeds：`{payload['protocol']['seeds']}`。",
            f"- patch size / overlap：`{payload['protocol']['patch_size']} / {payload['protocol']['overlap']}`。",
            "- 标签未参与 cover 构造、选择或审计。",
            "- `official_valid_evaluations = 0`，`official_test_evaluations = 0`。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/ogb"))
    parser.add_argument("--sample-graphs", type=int, default=1000)
    parser.add_argument("--sample-seed", type=int, default=20260816)
    parser.add_argument(
        "--cover-seeds",
        type=int,
        nargs="+",
        default=[20260813, 20260814, 20260815, 20260816, 20260817, 20260818, 20260819, 20260820],
    )
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--overlap", type=int, default=2)
    parser.add_argument("--retained-beam", type=int, default=8)
    parser.add_argument("--edge-capacity-multiplier", type=float, default=1.5)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    if args.sample_graphs <= 0 or len(set(args.cover_seeds)) < 2:
        raise ValueError("need positive graph count and at least two distinct cover seeds")

    _patch_torch_load_weights_only()
    from ogb.graphproppred import GraphPropPredDataset
    from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

    started = time.time()
    dataset = GraphPropPredDataset(name="ogbg-molhiv", root=str(args.root))
    split = dataset.get_idx_split()
    official_train = np.asarray(split["train"], dtype=np.int64)
    official_valid = np.asarray(split["valid"], dtype=np.int64)
    official_test = np.asarray(split["test"], dtype=np.int64)
    rng = np.random.default_rng(args.sample_seed)
    selected = np.sort(
        rng.choice(
            official_train,
            size=min(args.sample_graphs, len(official_train)),
            replace=False,
        )
    )
    if set(selected) & (set(official_valid) | set(official_test)):
        raise AssertionError("audit sample leaks official valid/test")

    atom_dims = tuple(int(value) for value in get_atom_feature_dims())
    bond_dims = tuple(int(value) for value in get_bond_feature_dims())
    rows: list[dict[str, Any]] = []
    for offset, graph_index in enumerate(selected, 1):
        graph_dict, _label = dataset[int(graph_index)]
        n_nodes = int(graph_dict["num_nodes"])
        graph = _edges_from_pyg(np.asarray(graph_dict["edge_index"]), n_nodes)
        atoms = np.asarray(graph_dict["node_feat"], dtype=np.int64)
        edge_features = _edge_feature_map(graph_dict)
        typed = typed_bond_adjacency(graph, edge_features, bond_dims)
        adjacency = _adjacency(graph)
        atomic_one_hot = _atomic_number_one_hot(atoms, atom_dims[0])
        canonical_types = _feature_row_colors(atoms)
        edges = sorted(graph.edges())
        beam_covers = []
        random_bfs_covers = []
        for cover_seed in args.cover_seeds:
            item = prepare_attributed_beam_graph(
                int(graph_index),
                graph,
                0,
                atomic_one_hot,
                typed,
                patch_size=args.patch_size,
                overlap=args.overlap,
                retained_beam=args.retained_beam,
                edge_capacity_multiplier=args.edge_capacity_multiplier,
                seed=int(cover_seed),
                edge_dim=int(np.prod(bond_dims)),
                canonical_node_features=atomic_one_hot,
                canonical_node_types=canonical_types,
            )
            beam_covers.append(item.slot_nodes)
            random_bfs_covers.append(
                _random_bfs_slots(
                    adjacency,
                    typed,
                    canonical_types,
                    seed=int(cover_seed),
                    patch_size=args.patch_size,
                    overlap=args.overlap,
                    edge_capacity_multiplier=args.edge_capacity_multiplier,
                )
            )
        eligible = any(len(component) > args.patch_size for component in _components(adjacency))
        rows.append(
            {
                "graph_index": int(graph_index),
                "n_nodes": n_nodes,
                "n_edges": len(edges),
                "eligible_for_stochastic_cover": bool(eligible),
                "beam": summarize_cover_family(
                    beam_covers, n_nodes=n_nodes, edges=edges
                ),
                "random_bfs": summarize_cover_family(
                    random_bfs_covers, n_nodes=n_nodes, edges=edges
                ),
            }
        )
        if offset == 1 or offset % 100 == 0 or offset == len(selected):
            print(f"audited official-train molecules {offset}/{len(selected)}", flush=True)

    summary = {
        family: _aggregate(rows, family)
        for family in ("beam", "random_bfs")
    }
    payload = {
        "protocol": {
            "dataset": "ogbg-molhiv",
            "split": "official-train only",
            "sampled_graphs": len(selected),
            "sample_seed": args.sample_seed,
            "seeds": [int(value) for value in args.cover_seeds],
            "patch_size": args.patch_size,
            "overlap": args.overlap,
            "retained_beam": args.retained_beam,
            "edge_capacity_multiplier": args.edge_capacity_multiplier,
            "selected_intersection_official_valid": 0,
            "selected_intersection_official_test": 0,
            "official_valid_evaluations": 0,
            "official_test_evaluations": 0,
        },
        "decision": classify(summary["beam"]),
        "summary": summary,
        "rows": rows,
        "elapsed_seconds": float(time.time() - started),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.report.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({"decision": payload["decision"], "summary": summary}, indent=2))
    print(f"wrote {args.json}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
