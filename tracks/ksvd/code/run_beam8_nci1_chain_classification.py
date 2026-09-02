"""Frozen Beam8 patch-chain classification Stage A on TU NCI1.

The runner keeps one label-free Beam8 cover per graph and compares bag, true
token-to-position binding, and shuffled binding under fixed relation message
passing.  Dictionaries and every statistical transform are outer-train only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .canonical_slots import exact_canonical_order, reorder_cover_structurally
from .data_tud import load_tud, node_feature_readout
from .from_scratch_unplanted_dictionary import deterministic_maximin_initialization
from .global_stable_ids import compute_global_wl_ids, reorder_by_stable_ids
from .imdb_walk_dictionary import encode_with_minimum_sparsity
from .ksvd import ksvd
from .marginal_candidate_cover import sample_marginal_candidate_cover
from .overlap_cover import PatchCover, _make_cover, _make_patch, patch_budget
from .run_luyin14_route import _atomic_json, _atomic_text, _pad_upper
from .run_real_structure_ksvd import graph_basic_features


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = ROOT / "data" / "TUD"
DEFAULT_JSON = ROOT / "tracks/ksvd/results/luyin14/beam8_nci1_chain_stage_a_20260813.json"
DEFAULT_REPORT = ROOT / "tracks/ksvd/results/luyin14/BEAM8_NCI1_CHAIN_STAGE_A_20260813.md"
PROTOCOL = "tracks/ksvd/docs/KSVD_BEAM8_NCI1_CHAIN_CLASSIFICATION_PROTOCOL_20260813.md"
S8O2_PROTOCOL = "tracks/ksvd/docs/KSVD_BEAM8_NCI1_S8O2_CLASSIFICATION_ADDENDUM_20260813.md"
TOKEN_FAMILIES = ("RAW", "INIT", "FINAL")
VARIANTS = (
    "FEATURE_ONLY",
    "FEATURE_STATS",
    "RAW_BAG",
    "RAW_CHAIN_TRUE",
    "RAW_CHAIN_SHUFFLED",
    "INIT_BAG",
    "INIT_CHAIN_TRUE",
    "FINAL_BAG",
    "FINAL_CHAIN_TRUE",
    "FINAL_CHAIN_SHUFFLED",
)


@dataclass(frozen=True)
class BeamGraph:
    index: int
    label: int
    vectors: np.ndarray
    node_histograms: np.ndarray
    node_sets: tuple[frozenset[int], ...]
    slot_nodes: tuple[tuple[int, ...], ...]
    centers: tuple[int, ...]
    segment_ids: tuple[int, ...]
    position_features: np.ndarray
    residual_features: np.ndarray
    graph_features: np.ndarray
    stats: np.ndarray
    sampling: dict[str, Any]


def _adjacency(graph: Any) -> np.ndarray:
    adjacency = np.zeros((graph.n, graph.n), dtype=np.int8)
    for left, right in graph.edges():
        adjacency[int(left), int(right)] = 1
        adjacency[int(right), int(left)] = 1
    return adjacency


def _components(adjacency: np.ndarray) -> tuple[tuple[int, ...], ...]:
    seen: set[int] = set()
    output = []
    for start in range(adjacency.shape[0]):
        if start in seen:
            continue
        seen.add(start)
        queue = [start]
        for node in queue:
            for raw_neighbor in np.flatnonzero(adjacency[node]):
                neighbor = int(raw_neighbor)
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        output.append(tuple(queue))
    return tuple(output)


def _pairs(nodes: Iterable[int]) -> set[tuple[int, int]]:
    return {tuple(sorted((int(left), int(right)))) for left, right in combinations(nodes, 2)}


def _true_edges(adjacency: np.ndarray) -> set[tuple[int, int]]:
    left, right = np.nonzero(np.triu(adjacency, k=1))
    return {(int(u), int(v)) for u, v in zip(left, right)}


def _component_seed(adjacency: np.ndarray, base_seed: int) -> int:
    digest = hashlib.sha256()
    digest.update(np.asarray(adjacency, dtype=np.int8).tobytes())
    digest.update(int(base_seed).to_bytes(8, "big", signed=False))
    return int.from_bytes(digest.digest()[:8], "big") % (2**63 - 1)


def _full_component_cover(adjacency: np.ndarray) -> PatchCover:
    nodes = tuple(range(adjacency.shape[0]))
    center = 0
    others = tuple(node for node in nodes if node != center)
    result = exact_canonical_order(
        adjacency,
        nodes,
        color_cells=((center,), others) if others else ((center,),),
    )
    return _make_cover(
        "full_component",
        [_make_patch(adjacency, result.node_ids, center)],
    )


def _component_cover(
    adjacency: np.ndarray,
    *,
    patch_size: int,
    overlap: int,
    retained_beam: int,
    edge_capacity_multiplier: float,
    seed: int,
) -> PatchCover:
    if adjacency.shape[0] <= patch_size:
        return _full_component_cover(adjacency)
    budget = patch_budget(
        adjacency,
        patch_size=patch_size,
        target_overlap=overlap,
        edge_capacity_multiplier=edge_capacity_multiplier,
    )
    cover = sample_marginal_candidate_cover(
        adjacency,
        np.random.default_rng(seed),
        n_patches=budget,
        patch_size=patch_size,
        target_overlap=overlap,
        retained_beam=retained_beam,
        candidate_restarts=1,
        allow_partial=True,
    )
    reordered, _audit = reorder_cover_structurally(
        adjacency, cover, "rooted_canonical"
    )
    return reordered


def _position_and_residual_features(
    adjacency: np.ndarray,
    node_sets: Sequence[frozenset[int]],
    segment_ids: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    truth = _true_edges(adjacency)
    total_nodes = max(adjacency.shape[0], 1)
    total_edges = max(len(truth), 1)
    covered_nodes: set[int] = set()
    covered_edges: set[tuple[int, int]] = set()
    segment_nodes = {
        segment: set().union(
            *(set(nodes) for nodes, value in zip(node_sets, segment_ids) if value == segment)
        )
        for segment in set(segment_ids)
    }
    segment_edges = {
        segment: {
            edge
            for edge in truth
            if edge[0] in nodes and edge[1] in nodes
        }
        for segment, nodes in segment_nodes.items()
    }
    segment_covered_nodes = {segment: set() for segment in segment_nodes}
    segment_covered_edges = {segment: set() for segment in segment_nodes}
    position_rows = []
    for index, nodes in enumerate(node_sets):
        pairs = _pairs(nodes)
        patch_edges = pairs & truth
        new_nodes = set(nodes) - covered_nodes
        new_edges = patch_edges - covered_edges
        segment = int(segment_ids[index])
        segment_start = index == 0 or segment_ids[index] != segment_ids[index - 1]
        previous_overlap = 0.0
        if not segment_start:
            previous = node_sets[index - 1]
            previous_overlap = len(nodes & previous) / max(len(nodes | previous), 1)
        covered_nodes.update(nodes)
        covered_edges.update(patch_edges)
        segment_covered_nodes[segment].update(nodes)
        segment_covered_edges[segment].update(patch_edges)
        pair_capacity = max(len(nodes) * (len(nodes) - 1) // 2, 1)
        position_rows.append(
            [
                len(new_nodes) / total_nodes,
                len(new_edges) / total_edges,
                len(segment_covered_nodes[segment]) / max(len(segment_nodes[segment]), 1),
                len(segment_covered_edges[segment]) / max(len(segment_edges[segment]), 1),
                float(segment_start),
                len(patch_edges) / pair_capacity,
                previous_overlap,
                len(nodes) / total_nodes,
            ]
        )
    residual = truth - covered_edges
    degrees = np.sum(adjacency, axis=1).astype(np.float64)
    incident = {node for edge in residual for node in edge}
    endpoint_degrees = [degrees[node] for edge in residual for node in edge]
    common = [float(np.dot(adjacency[left], adjacency[right])) for left, right in residual]
    residual_features = np.asarray(
        [
            len(residual) / total_edges,
            len(incident) / total_nodes,
            float(np.mean(endpoint_degrees)) / max(total_nodes - 1, 1)
            if endpoint_degrees
            else 0.0,
            float(np.max(endpoint_degrees)) / max(total_nodes - 1, 1)
            if endpoint_degrees
            else 0.0,
            float(np.mean(common)) / max(total_nodes - 2, 1) if common else 0.0,
            len(covered_edges) / total_edges,
            len(covered_nodes) / total_nodes,
            len(node_sets) / total_nodes,
        ],
        dtype=np.float64,
    )
    return (
        np.asarray(position_rows, dtype=np.float64),
        residual_features,
        {
            "edge_coverage": len(covered_edges) / total_edges,
            "node_coverage": len(covered_nodes) / total_nodes,
            "residual_edges": float(len(residual)),
        },
    )


def prepare_graph(
    index: int,
    graph: Any,
    label: int,
    node_features: np.ndarray | None,
    *,
    patch_size: int,
    overlap: int,
    retained_beam: int,
    edge_capacity_multiplier: float,
    seed: int,
) -> BeamGraph:
    adjacency = _adjacency(graph)
    features = (
        np.ones((graph.n, 1), dtype=np.float64)
        if node_features is None
        else np.asarray(node_features, dtype=np.float64)
    )
    node_sets: list[frozenset[int]] = []
    slot_nodes: list[tuple[int, ...]] = []
    vectors: list[np.ndarray] = []
    histograms: list[np.ndarray] = []
    segment_ids: list[int] = []
    centers: list[int] = []
    component_sizes = []
    partial_components = 0
    for segment_id, component in enumerate(_components(adjacency)):
        component_array = np.asarray(component, dtype=np.int64)
        local_source = adjacency[np.ix_(component_array, component_array)]
        stable = compute_global_wl_ids(local_source)
        local = reorder_by_stable_ids(local_source, stable)
        ordered_global = component_array[np.asarray(stable.order, dtype=np.int64)]
        cover = _component_cover(
            local,
            patch_size=patch_size,
            overlap=overlap,
            retained_beam=retained_beam,
            edge_capacity_multiplier=edge_capacity_multiplier,
            seed=_component_seed(local, seed),
        )
        component_sizes.append(len(component))
        expected_budget = (
            1
            if len(component) <= patch_size
            else patch_budget(
                local,
                patch_size=patch_size,
                target_overlap=overlap,
                edge_capacity_multiplier=edge_capacity_multiplier,
            )
        )
        partial_components += int(len(cover.patches) < expected_budget)
        for patch in cover.patches:
            global_nodes = tuple(int(ordered_global[node]) for node in patch.node_ids)
            node_sets.append(frozenset(global_nodes))
            slot_nodes.append(global_nodes)
            centers.append(int(ordered_global[patch.center]))
            segment_ids.append(segment_id)
            vectors.append(_pad_upper(patch.adjacency, patch_size))
            histograms.append(features[np.asarray(global_nodes)].mean(axis=0))
    position, residual, coverage = _position_and_residual_features(
        adjacency, node_sets, segment_ids
    )
    return BeamGraph(
        index=int(index),
        label=int(label),
        vectors=np.stack(vectors),
        node_histograms=np.stack(histograms),
        node_sets=tuple(node_sets),
        slot_nodes=tuple(slot_nodes),
        centers=tuple(centers),
        segment_ids=tuple(segment_ids),
        position_features=position,
        residual_features=residual,
        graph_features=node_feature_readout(features),
        stats=graph_basic_features(graph),
        sampling={
            "components": len(component_sizes),
            "component_sizes": component_sizes,
            "patches": len(vectors),
            "partial_components": partial_components,
            **coverage,
        },
    )


def _fit_dictionary(
    prepared: Sequence[BeamGraph],
    train_indices: Sequence[int],
    *,
    n_atoms: int,
    sparsity: int,
    iterations: int,
    max_train_patches: int,
    seed: int,
) -> dict[str, Any]:
    raw = np.concatenate(
        [prepared[int(index)].vectors for index in train_indices], axis=0
    ).T
    raw_count = raw.shape[1]
    if raw_count > max_train_patches:
        rng = np.random.default_rng(seed)
        selected = np.sort(
            rng.choice(raw_count, size=max_train_patches, replace=False)
        )
        raw = raw[:, selected]
    mean = raw.mean(axis=1, keepdims=True)
    centered = raw - mean
    nonzero = np.flatnonzero(np.linalg.norm(centered, axis=0) > 1e-12)
    atoms = min(n_atoms, len(nonzero), centered.shape[0])
    if atoms < 2:
        raise RuntimeError("too few nonzero Beam8 patches for a dictionary")
    initial, initialization = deterministic_maximin_initialization(
        centered[:, nonzero], atoms
    )
    final, _codes, training = ksvd(
        centered,
        n_atoms=atoms,
        T=min(sparsity, atoms),
        T_min=1,
        n_iter=iterations,
        seed=0,
        initial_dictionary=initial,
    )
    return {
        "mean": mean,
        "initial": initial,
        "final": final,
        "atoms": atoms,
        "raw_train_patches": int(raw_count),
        "used_train_patches": int(raw.shape[1]),
        "initialization": initialization,
        "training": training,
    }


def _encode(
    item: BeamGraph,
    dictionary: np.ndarray,
    mean: np.ndarray,
    sparsity: int,
) -> np.ndarray:
    codes = encode_with_minimum_sparsity(
        item.vectors.T - mean,
        dictionary,
        sparsity=min(sparsity, dictionary.shape[1]),
        minimum_sparsity=1,
    )
    return np.abs(codes.T)


def _normalize_patch_tokens(
    tokens: Sequence[np.ndarray], train_indices: Sequence[int]
) -> list[np.ndarray]:
    train = np.concatenate([tokens[int(index)] for index in train_indices], axis=0)
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return [(values - mean) / scale for values in tokens]


def _relation_matrices(item: BeamGraph) -> tuple[np.ndarray, ...]:
    count = len(item.node_sets)
    previous = np.zeros((count, count), dtype=np.float64)
    following = np.zeros((count, count), dtype=np.float64)
    overlap = np.zeros((count, count), dtype=np.float64)
    slot = np.zeros((count, count), dtype=np.float64)
    for left in range(count):
        for right in range(left + 1, count):
            shared = item.node_sets[left] & item.node_sets[right]
            if not shared:
                continue
            jaccard = len(shared) / max(
                len(item.node_sets[left] | item.node_sets[right]), 1
            )
            overlap[left, right] = overlap[right, left] = jaccard
            left_slots = {node: index for index, node in enumerate(item.slot_nodes[left])}
            right_slots = {node: index for index, node in enumerate(item.slot_nodes[right])}
            persistence = np.mean(
                [left_slots[node] == right_slots[node] for node in shared]
            )
            slot[left, right] = slot[right, left] = jaccard * float(persistence)
            if (
                right == left + 1
                and item.segment_ids[left] == item.segment_ids[right]
            ):
                previous[right, left] = jaccard
                following[left, right] = jaccard
    return previous, following, overlap, slot


def _row_normalized_message(weights: np.ndarray, values: np.ndarray) -> np.ndarray:
    denominator = weights.sum(axis=1, keepdims=True)
    normalized = np.divide(
        weights,
        denominator,
        out=np.zeros_like(weights),
        where=denominator > 1e-12,
    )
    return normalized @ values


def _moments(values: np.ndarray) -> np.ndarray:
    if values.shape[0] == 0:
        return np.zeros(3 * values.shape[1], dtype=np.float64)
    return np.concatenate(
        [values.mean(axis=0), values.std(axis=0), values.max(axis=0)]
    )


def _relation_readout(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    message = _row_normalized_message(weights, values)
    active = weights.sum(axis=1) > 1e-12
    if not np.any(active):
        return np.zeros(2 * values.shape[1] + 6, dtype=np.float64)
    local = values[active]
    received = message[active]
    dot = np.sum(local * received, axis=1)
    cosine = dot / np.maximum(
        np.linalg.norm(local, axis=1) * np.linalg.norm(received, axis=1), 1e-12
    )
    distance = np.mean(np.abs(local - received), axis=1)
    return np.concatenate(
        [
            received.mean(axis=0),
            received.std(axis=0),
            np.asarray(
                [
                    cosine.mean(),
                    cosine.std(),
                    cosine.max(initial=0.0),
                    distance.mean(),
                    distance.std(),
                    distance.max(initial=0.0),
                ],
                dtype=np.float64,
            ),
        ]
    )


def _relation_graph_summary(item: BeamGraph) -> np.ndarray:
    output = []
    for weights in _relation_matrices(item):
        degree = weights.sum(axis=1)
        output.extend(
            [
                float(weights.sum()),
                float(np.mean(degree)),
                float(np.std(degree)),
                float(np.max(degree, initial=0.0)),
            ]
        )
    output.extend(
        [
            len(item.vectors),
            len(set(item.segment_ids)),
            len(item.vectors) / max(len(set(item.segment_ids)), 1),
        ]
    )
    return np.asarray(output, dtype=np.float64)


def _shuffle_permutation(count: int, *, seed: int) -> np.ndarray:
    if count <= 1:
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(count)
    if np.array_equal(permutation, np.arange(count)):
        permutation = np.roll(permutation, 1)
    return permutation.astype(np.int64)


def _graph_token_feature(
    item: BeamGraph,
    tokens: np.ndarray,
    *,
    chain: bool,
    permutation: np.ndarray | None = None,
) -> np.ndarray:
    values = tokens if permutation is None else tokens[permutation]
    common = np.concatenate(
        [
            _moments(values),
            _moments(item.position_features),
            _relation_graph_summary(item),
            item.residual_features,
        ]
    )
    if not chain:
        return common
    relation = np.concatenate(
        [_relation_readout(values, weights) for weights in _relation_matrices(item)]
    )
    return np.concatenate([common, relation])


def _fit_score(
    train: np.ndarray,
    train_labels: np.ndarray,
    test: np.ndarray,
    test_labels: np.ndarray,
    *,
    seed: int,
) -> dict[str, float]:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, C=1.0, random_state=seed),
    )
    model.fit(train, train_labels)
    prediction = model.predict(test)
    return {
        "balanced_accuracy": float(
            balanced_accuracy_score(test_labels, prediction)
        ),
        "accuracy": float(accuracy_score(test_labels, prediction)),
    }


def _paired(
    folds: Sequence[dict[str, Any]], left: str, right: str
) -> dict[str, Any]:
    values = np.asarray(
        [
            fold["scores"][left]["balanced_accuracy"]
            - fold["scores"][right]["balanced_accuracy"]
            for fold in folds
        ],
        dtype=np.float64,
    )
    return {
        "mean": float(values.mean()),
        "wins": int(np.sum(values > 1e-12)),
        "ties": int(np.sum(np.abs(values) <= 1e-12)),
        "losses": int(np.sum(values < -1e-12)),
        "values": values.tolist(),
    }


def _summarize(folds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    variants = {
        variant: {
            "balanced_accuracy_mean": float(
                np.mean([fold["scores"][variant]["balanced_accuracy"] for fold in folds])
            ),
            "balanced_accuracy_std": float(
                np.std([fold["scores"][variant]["balanced_accuracy"] for fold in folds])
            ),
            "accuracy_mean": float(
                np.mean([fold["scores"][variant]["accuracy"] for fold in folds])
            ),
        }
        for variant in VARIANTS
    }
    paired = {
        "raw_true_vs_shuffled": _paired(
            folds, "RAW_CHAIN_TRUE", "RAW_CHAIN_SHUFFLED"
        ),
        "raw_true_vs_bag": _paired(folds, "RAW_CHAIN_TRUE", "RAW_BAG"),
        "final_true_vs_shuffled": _paired(
            folds, "FINAL_CHAIN_TRUE", "FINAL_CHAIN_SHUFFLED"
        ),
        "final_true_vs_bag": _paired(folds, "FINAL_CHAIN_TRUE", "FINAL_BAG"),
        "final_vs_init_true": _paired(
            folds, "FINAL_CHAIN_TRUE", "INIT_CHAIN_TRUE"
        ),
    }
    raw = paired["raw_true_vs_shuffled"]
    final = paired["final_true_vs_shuffled"]
    final_bag = paired["final_true_vs_bag"]
    update = paired["final_vs_init_true"]
    gates = {
        "raw_substrate_gate": bool(raw["mean"] >= 0.01 and raw["wins"] >= 2),
        "relation_gate": bool(
            final["mean"] >= 0.01
            and final["wins"] >= 2
            and final_bag["mean"] >= -1e-12
        ),
        "ksvd_update_gate": bool(update["mean"] >= 0.01 and update["wins"] >= 2),
    }
    if gates["relation_gate"] and gates["ksvd_update_gate"]:
        decision = "BEAM8_RELATION_AND_KSVD_UPDATE_PASS"
    elif gates["relation_gate"] or gates["raw_substrate_gate"]:
        decision = "BEAM8_SUBSTRATE_PASS_KSVD_UPDATE_NOT_ESTABLISHED"
    else:
        decision = "BEAM8_CHAIN_STAGE_A_BELOW_GATE"
    return {"variants": variants, "paired": paired, "gates": gates, "decision": decision}


def _feature_matrices(
    prepared: Sequence[BeamGraph],
    train_indices: np.ndarray,
    dictionary: dict[str, Any],
    *,
    sparsity: int,
    shuffle_seed: int,
) -> dict[str, np.ndarray]:
    raw_tokens = [
        np.concatenate([item.vectors, item.node_histograms], axis=1)
        for item in prepared
    ]
    init_tokens = [
        np.concatenate(
            [
                _encode(
                    item, dictionary["initial"], dictionary["mean"], sparsity
                ),
                item.node_histograms,
            ],
            axis=1,
        )
        for item in prepared
    ]
    final_tokens = [
        np.concatenate(
            [
                _encode(item, dictionary["final"], dictionary["mean"], sparsity),
                item.node_histograms,
            ],
            axis=1,
        )
        for item in prepared
    ]
    families = {
        "RAW": _normalize_patch_tokens(raw_tokens, train_indices),
        "INIT": _normalize_patch_tokens(init_tokens, train_indices),
        "FINAL": _normalize_patch_tokens(final_tokens, train_indices),
    }
    rows: dict[str, list[np.ndarray]] = {variant: [] for variant in VARIANTS}
    for graph_index, item in enumerate(prepared):
        rows["FEATURE_ONLY"].append(item.graph_features)
        rows["FEATURE_STATS"].append(
            np.concatenate([item.graph_features, item.stats])
        )
        for family, tokens in families.items():
            rows[f"{family}_BAG"].append(
                _graph_token_feature(item, tokens[graph_index], chain=False)
            )
            rows[f"{family}_CHAIN_TRUE"].append(
                _graph_token_feature(item, tokens[graph_index], chain=True)
            )
            shuffled_name = f"{family}_CHAIN_SHUFFLED"
            if shuffled_name in rows:
                permutation = _shuffle_permutation(
                    len(tokens[graph_index]),
                    seed=shuffle_seed + int(item.index) * 1009,
                )
                rows[shuffled_name].append(
                    _graph_token_feature(
                        item,
                        tokens[graph_index],
                        chain=True,
                        permutation=permutation,
                    )
                )
    return {name: np.stack(values) for name, values in rows.items()}


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    graphs, labels, node_features, metadata = load_tud(
        args.dataset, args.dataset_root
    )
    if args.limit is not None and args.limit < len(graphs):
        rng = np.random.default_rng(20260813)
        selected = []
        per_class = max(2, args.limit // len(np.unique(labels)))
        for label in np.unique(labels):
            candidates = np.flatnonzero(labels == label)
            selected.extend(
                rng.choice(
                    candidates,
                    size=min(per_class, len(candidates)),
                    replace=False,
                ).tolist()
            )
        selected = sorted(int(index) for index in selected)[: args.limit]
        graphs = [graphs[index] for index in selected]
        node_features = [node_features[index] for index in selected]
        labels = labels[np.asarray(selected, dtype=np.int64)]
        metadata = {**metadata, "limited_graph_count": len(graphs)}
    prepared = []
    for index, (graph, label, features) in enumerate(
        zip(graphs, labels, node_features)
    ):
        prepared.append(
            prepare_graph(
                index,
                graph,
                int(label),
                features,
                patch_size=args.patch_size,
                overlap=args.overlap,
                retained_beam=args.retained_beam,
                edge_capacity_multiplier=args.edge_capacity_multiplier,
                seed=args.cover_seed,
            )
        )
        if (index + 1) % 250 == 0 or index + 1 == len(graphs):
            print(f"cover {index + 1}/{len(graphs)}", flush=True)
    splitter = StratifiedKFold(
        n_splits=args.n_splits, shuffle=True, random_state=args.split_seed
    )
    folds = []
    for fold_index, (train_indices, test_indices) in enumerate(
        splitter.split(np.zeros(len(labels)), labels)
    ):
        fold_seed = args.split_seed * 100 + fold_index
        dictionary = _fit_dictionary(
            prepared,
            train_indices,
            n_atoms=args.n_atoms,
            sparsity=args.sparsity,
            iterations=args.iterations,
            max_train_patches=args.max_train_patches,
            seed=fold_seed,
        )
        matrices = _feature_matrices(
            prepared,
            train_indices,
            dictionary,
            sparsity=args.sparsity,
            shuffle_seed=args.shuffle_seed + fold_seed,
        )
        scores = {
            variant: _fit_score(
                values[train_indices],
                labels[train_indices],
                values[test_indices],
                labels[test_indices],
                seed=fold_seed,
            )
            for variant, values in matrices.items()
        }
        folds.append(
            {
                "fold_index": fold_index,
                "train_count": len(train_indices),
                "test_count": len(test_indices),
                "dictionary": {
                    "atoms": dictionary["atoms"],
                    "raw_train_patches": dictionary["raw_train_patches"],
                    "used_train_patches": dictionary["used_train_patches"],
                    "training_recon_rel": dictionary["training"].get("recon_rel"),
                    "training_recon_curve": dictionary["training"].get("recon_curve", []),
                },
                "scores": scores,
            }
        )
        print(
            f"fold {fold_index}: "
            f"FINAL_TRUE={scores['FINAL_CHAIN_TRUE']['balanced_accuracy']:.4f} "
            f"SHUFFLED={scores['FINAL_CHAIN_SHUFFLED']['balanced_accuracy']:.4f} "
            f"INIT={scores['INIT_CHAIN_TRUE']['balanced_accuracy']:.4f}",
            flush=True,
        )
    sampling = {
        "graph_count": len(prepared),
        "disconnected_graphs": int(
            sum(item.sampling["components"] > 1 for item in prepared)
        ),
        "mean_patches": float(np.mean([len(item.vectors) for item in prepared])),
        "mean_edge_coverage": float(
            np.mean([item.sampling["edge_coverage"] for item in prepared])
        ),
        "mean_node_coverage": float(
            np.mean([item.sampling["node_coverage"] for item in prepared])
        ),
        "partial_component_count": int(
            sum(item.sampling["partial_components"] for item in prepared)
        ),
        "single_patch_graphs": int(sum(len(item.vectors) == 1 for item in prepared)),
        "graphs_with_chain_edges": int(
            sum(np.count_nonzero(_relation_matrices(item)[0]) > 0 for item in prepared)
        ),
        "mean_chain_edges": float(
            np.mean(
                [np.count_nonzero(_relation_matrices(item)[0]) for item in prepared]
            )
        ),
        "graphs_with_nonchain_overlap": int(
            sum(
                np.count_nonzero(_relation_matrices(item)[2])
                > np.count_nonzero(_relation_matrices(item)[0])
                + np.count_nonzero(_relation_matrices(item)[1])
                for item in prepared
            )
        ),
    }
    protocol = S8O2_PROTOCOL if (args.patch_size, args.overlap) == (8, 2) else PROTOCOL
    return {
        "protocol": protocol,
        "dataset": metadata,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "sampling": sampling,
        "folds": folds,
        "summary": _summarize(folds),
        "seconds": time.time() - started,
    }


def render_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Beam8/NCI1 patch-chain classification Stage A",
        "",
        f"> 协议：`{payload['protocol']}`  ",
        f"> 判定：`{summary['decision']}`",
        "",
        "## 1. Sampling",
        "",
        f"- graphs：`{payload['sampling']['graph_count']}`；非连通图：`{payload['sampling']['disconnected_graphs']}`；",
        f"- mean patches：`{payload['sampling']['mean_patches']:.3f}`；",
        f"- mean edge/node coverage：`{payload['sampling']['mean_edge_coverage']:.4f}` / `{payload['sampling']['mean_node_coverage']:.4f}`；",
        f"- partial components：`{payload['sampling']['partial_component_count']}`。",
        f"- single-patch graphs：`{payload['sampling']['single_patch_graphs']}`；graphs with chain edges：`{payload['sampling']['graphs_with_chain_edges']}`；mean directed chain edges：`{payload['sampling']['mean_chain_edges']:.3f}`；",
        f"- graphs with non-chain overlap：`{payload['sampling']['graphs_with_nonchain_overlap']}`。",
        "",
        "## 2. Classification",
        "",
        "| variant | balanced accuracy | accuracy |",
        "|---|---:|---:|",
    ]
    for variant in VARIANTS:
        row = summary["variants"][variant]
        lines.append(
            f"| {variant} | {row['balanced_accuracy_mean']:.4f} ± {row['balanced_accuracy_std']:.4f} | {row['accuracy_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 3. Paired attribution",
            "",
            "| comparison | mean difference | W/T/L |",
            "|---|---:|---:|",
        ]
    )
    for name, row in summary["paired"].items():
        lines.append(
            f"| {name} | {row['mean']:+.4f} | {row['wins']}/{row['ties']}/{row['losses']} |"
        )
    lines.extend(
        [
            "",
            "## 4. Gates",
            "",
            f"- RAW substrate gate：`{summary['gates']['raw_substrate_gate']}`；",
            f"- Beam8 relation gate：`{summary['gates']['relation_gate']}`；",
            f"- KSVD update gate：`{summary['gates']['ksvd_update_gate']}`；",
            "",
            "## 5. 解释边界",
            "",
            "- TRUE/SHUFFLED 共用 cover、token multiset、关系图、位置元数据和 residual sidecar，只改变 token 到 patch position 的绑定。",
            "- TRUE 优于 SHUFFLED 但低于 BAG，表示真实绑定可检测，却还没有转化为超过无序聚合的分类价值。",
            "- relation 或 RAW 通过但 KSVD gate 失败，只支持 Beam8 chain substrate，不支持普通无监督 KSVD updates 具有分类增益。",
            "- 本轮为固定无参数 relation message passing；只有 relation gate 通过后才进入可学习 patch GNN。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="NCI1")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--patch-size", type=int, default=10)
    parser.add_argument("--overlap", type=int, default=3)
    parser.add_argument("--retained-beam", type=int, default=8)
    parser.add_argument("--edge-capacity-multiplier", type=float, default=1.5)
    parser.add_argument("--n-atoms", type=int, default=24)
    parser.add_argument("--sparsity", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--max-train-patches", type=int, default=3000)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--cover-seed", type=int, default=20260813)
    parser.add_argument("--shuffle-seed", type=int, default=731421)
    parser.add_argument("--limit", type=int, default=None)
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
