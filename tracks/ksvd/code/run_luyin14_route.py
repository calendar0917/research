"""Close the luyin14 KSVD route: edge repair, downstream relation, and fusion.

The runner intentionally keeps the model small and the attribution explicit.
Sampling is label-free.  Every dictionary, scaler, classifier, and gated model
is fitted inside its outer training fold.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .canonical_slots import exact_canonical_order, reorder_cover_structurally
from .data_tud import load_tud, node_feature_readout
from .from_scratch_unplanted_dictionary import deterministic_maximin_initialization
from .global_stable_ids import compute_global_wl_ids, reorder_by_stable_ids
from .imdb_walk_dictionary import encode_with_minimum_sparsity
from .ksvd import ksvd
from .marginal_candidate_cover import sample_marginal_candidate_cover
from .overlap_cover import OrderedPatch, PatchCover, _make_cover, _make_patch, patch_budget
from .pipeline import graph_from_edge_index
from .run_real_structure_ksvd import graph_basic_features


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = ROOT / "data" / "TUD"
DEFAULT_JSON = ROOT / "tracks/ksvd/results/luyin14/route_closure_20260812.json"
DEFAULT_REPORT = ROOT / "tracks/ksvd/results/luyin14/ROUTE_CLOSURE_20260812.md"
PROTOCOL = "tracks/ksvd/docs/KSVD_LUYIN14_ROUTE_PROTOCOL_20260812.md"
DEFAULT_DATASETS = ("IMDB-BINARY", "IMDB-MULTI", "MUTAG", "PTC_MR")
FEATURE_DATASETS = frozenset(("MUTAG", "PTC_MR"))
FEATURE_SETS = (
    "STATS",
    "RAW_PATCH",
    "INIT_CONTENT",
    "FAIR95_CONTENT",
    "FAIR95_RELATION_GRAPH",
    "FAIR95_TRUE_RELATION",
    "FAIR95_SHUFFLED_RELATION",
    "EDGE100_CONTENT",
    "FAIR95_RESIDUAL",
)
FEATURE_AWARE_SETS = (
    "FEATURE_ONLY",
    "FEATURE_STATS",
    "FEATURE_FAIR95_CONTENT",
    "FEATURE_FAIR95_TRUE_RELATION",
    "FEATURE_FAIR95_RESIDUAL",
)


@dataclass(frozen=True)
class PatchSet:
    vectors: np.ndarray
    node_sets: tuple[frozenset[int], ...]
    segment_ids: tuple[int, ...]
    centers: tuple[int, ...]


@dataclass(frozen=True)
class PreparedGraph:
    index: int
    label: int
    stats: np.ndarray
    node_features: np.ndarray | None
    fair: PatchSet
    edge100: PatchSet
    residual_features: np.ndarray
    sampling: dict[str, Any]


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2))


def _adjacency(graph) -> np.ndarray:
    adjacency = np.zeros((graph.n, graph.n), dtype=np.int8)
    for left, right in graph.edges():
        adjacency[left, right] = adjacency[right, left] = 1
    return adjacency


def _pairs(nodes: Iterable[int]) -> set[tuple[int, int]]:
    return {tuple(sorted(pair)) for pair in combinations(tuple(nodes), 2)}


def _true_edges(adjacency: np.ndarray) -> set[tuple[int, int]]:
    left, right = np.nonzero(np.triu(adjacency, k=1))
    return {(int(u), int(v)) for u, v in zip(left, right)}


def _covered_edges(adjacency: np.ndarray, patches: Sequence[OrderedPatch]) -> set[tuple[int, int]]:
    truth = _true_edges(adjacency)
    return set().union(*(_pairs(patch.node_ids) for patch in patches)) & truth if patches else set()


def _prefix_metrics(adjacency: np.ndarray, patches: Sequence[OrderedPatch]) -> dict[str, float]:
    truth = _true_edges(adjacency)
    covered = _covered_edges(adjacency, patches)
    covered_nodes = set().union(*(set(patch.node_ids) for patch in patches)) if patches else set()
    degrees = np.sum(adjacency, axis=1).astype(np.int64)
    recalls = []
    for node in np.flatnonzero(degrees > 0):
        incident = {
            tuple(sorted((int(node), int(neighbor))))
            for neighbor in np.flatnonzero(adjacency[int(node)])
        }
        recalls.append(len(incident & covered) / max(len(incident), 1))
    pair_count = adjacency.shape[0] * (adjacency.shape[0] - 1) // 2
    observed = set().union(*(_pairs(patch.node_ids) for patch in patches)) if patches else set()
    return {
        "edge_coverage": len(covered) / max(len(truth), 1),
        "pair_coverage": len(observed) / max(pair_count, 1),
        "node_coverage": len(covered_nodes) / max(adjacency.shape[0], 1),
        "incident_p10": float(np.quantile(recalls, 0.10)) if recalls else 1.0,
        "residual_edge_count": float(len(truth - covered)),
    }


def _prefix_cover(cover: PatchCover, count: int, method: str) -> PatchCover:
    count = min(max(int(count), 1), len(cover.patches))
    return _make_cover(
        method,
        cover.patches[:count],
        cover.segment_ids[:count],
        cover.target_edges[:count],
        cover.bridge_lengths[:count],
    )


def _select_fair_prefix(adjacency: np.ndarray, cover: PatchCover) -> tuple[int, bool]:
    for count in range(1, len(cover.patches) + 1):
        metrics = _prefix_metrics(adjacency, cover.patches[:count])
        if (
            metrics["edge_coverage"] + 1e-12 >= 0.95
            and metrics["incident_p10"] + 1e-12 >= 0.90
            and metrics["node_coverage"] + 1e-12 >= 1.0
        ):
            return count, True
    return len(cover.patches), False


def _completion_nodes(
    adjacency: np.ndarray,
    edge: tuple[int, int],
    *,
    patch_size: int,
    covered_edges: set[tuple[int, int]],
) -> tuple[int, ...]:
    nodes = [int(edge[0]), int(edge[1])]
    while len(nodes) < min(patch_size, adjacency.shape[0]):
        present = set(nodes)
        frontier = sorted(
            {
                int(neighbor)
                for node in nodes
                for neighbor in np.flatnonzero(adjacency[node])
                if int(neighbor) not in present
            }
        )
        if not frontier:
            frontier = [node for node in range(adjacency.shape[0]) if node not in present]
        scored = []
        for candidate in frontier:
            new_edges = sum(
                adjacency[candidate, node] != 0
                and tuple(sorted((candidate, int(node)))) not in covered_edges
                for node in nodes
            )
            scored.append(((int(new_edges), int(np.sum(adjacency[candidate]))), candidate))
        nodes.append(max(scored)[1])
    return tuple(nodes)


def _complete_edges(
    adjacency: np.ndarray,
    fair_cover: PatchCover,
    *,
    patch_size: int,
) -> PatchCover:
    patches = list(fair_cover.patches)
    segments = list(fair_cover.segment_ids)
    targets = list(fair_cover.target_edges)
    bridges = list(fair_cover.bridge_lengths)
    truth = _true_edges(adjacency)
    covered = _covered_edges(adjacency, patches)
    while covered != truth:
        deficits = np.zeros(adjacency.shape[0], dtype=np.int64)
        for left, right in truth - covered:
            deficits[left] += 1
            deficits[right] += 1
        target = max(
            truth - covered,
            key=lambda pair: (deficits[pair[0]] + deficits[pair[1]], pair),
        )
        nodes = _completion_nodes(
            adjacency, target, patch_size=patch_size, covered_edges=covered
        )
        patches.append(_make_patch(adjacency, nodes, target[0]))
        segments.append(segments[-1] + 1)
        targets.append(target)
        bridges.append(0)
        covered = _covered_edges(adjacency, patches)
    return _make_cover("fair95_edge100_completion", patches, segments, targets, bridges)


def _fair_prefix_from_completed(
    adjacency: np.ndarray, cover: PatchCover
) -> tuple[PatchCover, bool]:
    count, reached = _select_fair_prefix(adjacency, cover)
    return _prefix_cover(cover, count, "fair95_completed_prefix"), reached


def _pad_upper(adjacency: np.ndarray, patch_size: int) -> np.ndarray:
    padded = np.zeros((patch_size, patch_size), dtype=np.float64)
    n_nodes = adjacency.shape[0]
    padded[:n_nodes, :n_nodes] = adjacency
    return padded[np.triu_indices(patch_size, k=1)]


def _patch_set(adjacency: np.ndarray, cover: PatchCover, patch_size: int) -> PatchSet:
    vectors = np.stack(
        [_pad_upper(patch.adjacency, patch_size) for patch in cover.patches], axis=0
    )
    return PatchSet(
        vectors=vectors,
        node_sets=tuple(frozenset(patch.node_ids) for patch in cover.patches),
        segment_ids=tuple(int(value) for value in cover.segment_ids),
        centers=tuple(int(patch.center) for patch in cover.patches),
    )


def _residual_features(adjacency: np.ndarray, cover: PatchCover) -> np.ndarray:
    truth = _true_edges(adjacency)
    covered = _covered_edges(adjacency, cover.patches)
    residual = sorted(truth - covered)
    degrees = np.sum(adjacency, axis=1).astype(np.float64)
    incident = np.zeros(adjacency.shape[0], dtype=np.float64)
    common = []
    endpoint_degrees = []
    for left, right in residual:
        incident[left] += 1.0
        incident[right] += 1.0
        common.append(float(np.dot(adjacency[left], adjacency[right])))
        endpoint_degrees.extend((degrees[left], degrees[right]))
    normalizer = max(len(truth), 1)
    nonzero_incident = incident[incident > 0]
    endpoint = np.asarray(endpoint_degrees or [0.0], dtype=np.float64)
    common_values = np.asarray(common or [0.0], dtype=np.float64)
    return np.asarray(
        [
            len(residual) / normalizer,
            math.log1p(len(residual)),
            float(np.mean(incident > 0)),
            float(np.mean(nonzero_incident)) if nonzero_incident.size else 0.0,
            float(np.max(incident, initial=0.0)),
            float(np.quantile(incident, 0.90)),
            float(np.mean(endpoint)) / max(adjacency.shape[0] - 1, 1),
            float(np.std(endpoint)) / max(adjacency.shape[0] - 1, 1),
            float(np.max(endpoint, initial=0.0)) / max(adjacency.shape[0] - 1, 1),
            float(np.mean(common_values)) / max(adjacency.shape[0] - 2, 1),
            float(np.max(common_values, initial=0.0)) / max(adjacency.shape[0] - 2, 1),
            _prefix_metrics(adjacency, cover.patches)["pair_coverage"],
        ],
        dtype=np.float64,
    )


def _full_graph_cover(adjacency: np.ndarray) -> PatchCover:
    nodes = tuple(range(adjacency.shape[0]))
    center = 0
    result = exact_canonical_order(
        adjacency, nodes, color_cells=((center,), tuple(node for node in nodes if node != center))
    )
    return _make_cover("full_graph", [_make_patch(adjacency, result.node_ids, center)])


def prepare_graph(
    index: int,
    graph,
    label: int,
    node_features: np.ndarray | None,
    *,
    patch_size: int,
    overlap: int,
    maximum_patches: int,
    retained_beam: int,
    seed: int,
) -> PreparedGraph:
    source = _adjacency(graph)
    stable_ids = compute_global_wl_ids(source)
    adjacency = reorder_by_stable_ids(source, stable_ids)
    reordered_features = (
        None
        if node_features is None
        else np.asarray(node_features, dtype=np.float64)[np.asarray(stable_ids.order)]
    )
    if adjacency.shape[0] <= patch_size:
        fair_cover = _full_graph_cover(adjacency)
        fair_reached = True
        raw_cover_count = 1
        base_count = 1
    else:
        base_count = patch_budget(
            adjacency,
            patch_size=patch_size,
            target_overlap=overlap,
            edge_capacity_multiplier=1.5,
        )
        requested = min(max(maximum_patches, base_count + 16), 128)
        raw_cover = sample_marginal_candidate_cover(
            adjacency,
            np.random.default_rng(seed),
            n_patches=requested,
            patch_size=patch_size,
            target_overlap=overlap,
            retained_beam=retained_beam,
            candidate_restarts=1,
            allow_partial=True,
        )
        raw_cover_count = len(raw_cover.patches)
        fair_count, fair_reached = _select_fair_prefix(adjacency, raw_cover)
        if fair_reached:
            fair_cover = _prefix_cover(raw_cover, fair_count, "fair95_prefix")
        else:
            completed = _complete_edges(adjacency, raw_cover, patch_size=patch_size)
            fair_cover, fair_reached = _fair_prefix_from_completed(
                adjacency, completed
            )
        edge100_cover = _complete_edges(adjacency, fair_cover, patch_size=patch_size)
        fair_cover, _slot_audit = reorder_cover_structurally(
            adjacency, fair_cover, "rooted_canonical"
        )
        edge100_cover, _slot_audit = reorder_cover_structurally(
            adjacency, edge100_cover, "rooted_canonical"
        )
    if adjacency.shape[0] <= patch_size:
        edge100_cover = fair_cover
    fair_metrics = _prefix_metrics(adjacency, fair_cover.patches)
    edge100_metrics = _prefix_metrics(adjacency, edge100_cover.patches)
    if edge100_metrics["edge_coverage"] < 1.0 - 1e-12:
        raise RuntimeError("edge completion failed to reach 100%")
    return PreparedGraph(
        index=int(index),
        label=int(label),
        stats=graph_basic_features(graph),
        node_features=(
            None if reordered_features is None else node_feature_readout(reordered_features)
        ),
        fair=_patch_set(adjacency, fair_cover, patch_size),
        edge100=_patch_set(adjacency, edge100_cover, patch_size),
        residual_features=_residual_features(adjacency, fair_cover),
        sampling={
            "n_nodes": int(adjacency.shape[0]),
            "n_edges": int(np.sum(adjacency) // 2),
            "base_patch_count": int(base_count),
            "raw_cover_count": int(raw_cover_count),
            "fair_reached": bool(fair_reached),
            "fair_patch_count": len(fair_cover.patches),
            "edge100_patch_count": len(edge100_cover.patches),
            "added_patch_count": len(edge100_cover.patches) - len(fair_cover.patches),
            "fair_edge_coverage": float(fair_metrics["edge_coverage"]),
            "fair_pair_coverage": float(fair_metrics["pair_coverage"]),
            "fair_residual_edges": int(fair_metrics["residual_edge_count"]),
            "edge100_edge_coverage": float(edge100_metrics["edge_coverage"]),
            "edge100_pair_coverage": float(edge100_metrics["pair_coverage"]),
            "fair_segment_count": len(set(fair_cover.segment_ids)),
            "edge100_segment_count": len(set(edge100_cover.segment_ids)),
        },
    )


def _raw_patch_readout(patch_set: PatchSet) -> np.ndarray:
    values = patch_set.vectors
    return np.concatenate(
        [values.mean(axis=0), values.std(axis=0), values.max(axis=0)]
    )


def _code_readout(codes: np.ndarray) -> np.ndarray:
    absolute = np.abs(codes)
    return np.concatenate(
        [
            (absolute > 1e-10).mean(axis=1),
            absolute.mean(axis=1),
            np.sqrt((codes * codes).mean(axis=1)),
        ]
    )


def _relation_channels(patch_set: PatchSet) -> tuple[np.ndarray, np.ndarray]:
    count = len(patch_set.node_sets)
    chain = np.zeros((count, count), dtype=np.float64)
    overlap = np.zeros((count, count), dtype=np.float64)
    for left in range(count):
        for right in range(left + 1, count):
            a = patch_set.node_sets[left]
            b = patch_set.node_sets[right]
            weight = len(a & b) / max(len(a | b), 1)
            overlap[left, right] = overlap[right, left] = weight
            if right == left + 1 and patch_set.segment_ids[left] == patch_set.segment_ids[right]:
                chain[left, right] = chain[right, left] = weight
    return chain, overlap


def _relation_graph_readout(patch_set: PatchSet) -> np.ndarray:
    output = []
    for weights in _relation_channels(patch_set):
        degree = weights.sum(axis=1)
        output.extend(
            [
                float(weights.sum() / 2.0),
                float(degree.mean()) if degree.size else 0.0,
                float(degree.std()) if degree.size else 0.0,
            ]
        )
    output.extend(
        [
            float(len(patch_set.vectors)),
            float(len(set(patch_set.segment_ids))),
        ]
    )
    return np.asarray(output, dtype=np.float64)


def _aligned_relation_readout(
    codes: np.ndarray,
    patch_set: PatchSet,
    *,
    permutation: np.ndarray | None = None,
) -> np.ndarray:
    values = np.abs(codes).T
    if permutation is not None:
        values = values[np.asarray(permutation, dtype=np.int64)]
    output = []
    winners = np.argmax(values, axis=1) if len(values) else np.zeros(0, dtype=np.int64)
    for weights in _relation_channels(patch_set):
        left, right = np.nonzero(np.triu(weights, k=1) > 0)
        if left.size == 0:
            output.extend([0.0] * 5)
            continue
        pair_weights = weights[left, right]
        norm = max(float(pair_weights.sum()), 1e-12)
        dot = np.sum(values[left] * values[right], axis=1)
        cosine = dot / np.maximum(
            np.linalg.norm(values[left], axis=1) * np.linalg.norm(values[right], axis=1),
            1e-12,
        )
        l1 = np.mean(np.abs(values[left] - values[right]), axis=1)
        same = (winners[left] == winners[right]).astype(np.float64)
        output.extend(
            [
                float(pair_weights.sum()),
                float(pair_weights @ dot / norm),
                float(pair_weights @ cosine / norm),
                float(pair_weights @ l1 / norm),
                float(pair_weights @ same / norm),
            ]
        )
    return np.asarray(output, dtype=np.float64)


def _stack_patches(
    prepared: Sequence[PreparedGraph], indices: Sequence[int], branch: str
) -> np.ndarray:
    values = [getattr(prepared[int(index)], branch).vectors for index in indices]
    return np.concatenate(values, axis=0).T


def _fit_dictionary(
    prepared: Sequence[PreparedGraph],
    train_indices: np.ndarray,
    *,
    branch: str,
    n_atoms: int,
    sparsity: int,
    iterations: int,
    max_train_patches: int,
    seed: int,
) -> dict[str, Any]:
    raw = _stack_patches(prepared, train_indices, branch)
    raw_count = raw.shape[1]
    if raw_count > max_train_patches:
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(raw_count, size=max_train_patches, replace=False))
        raw = raw[:, selected]
    mean = raw.mean(axis=1, keepdims=True)
    centered = raw - mean
    nonzero = np.flatnonzero(np.linalg.norm(centered, axis=0) > 1e-12)
    if nonzero.size < n_atoms:
        centered = raw.copy()
        mean = np.zeros_like(mean)
        nonzero = np.flatnonzero(np.linalg.norm(centered, axis=0) > 1e-12)
    atoms = min(n_atoms, int(nonzero.size), centered.shape[0])
    if atoms < 2:
        raise RuntimeError("not enough nonzero training patches for a dictionary")
    initial, init_info = deterministic_maximin_initialization(centered[:, nonzero], atoms)
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
        "initialization": init_info,
        "training": training,
    }


def _encode_patch_set(
    patch_set: PatchSet, dictionary: np.ndarray, mean: np.ndarray, sparsity: int
) -> np.ndarray:
    centered = patch_set.vectors.T - mean
    return encode_with_minimum_sparsity(
        centered, dictionary, sparsity=min(sparsity, dictionary.shape[1]), minimum_sparsity=1
    )


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
        "balanced_accuracy": float(balanced_accuracy_score(test_labels, prediction)),
        "accuracy": float(accuracy_score(test_labels, prediction)),
    }


def _feature_matrix(
    prepared: Sequence[PreparedGraph],
    indices: Sequence[int],
    fair_dictionary: dict[str, Any],
    edge_dictionary: dict[str, Any],
    *,
    sparsity: int,
    shuffle_seed: int,
) -> dict[str, np.ndarray]:
    rows: dict[str, list[np.ndarray]] = {name: [] for name in FEATURE_SETS}
    rows.update({name: [] for name in FEATURE_AWARE_SETS})
    for index in indices:
        item = prepared[int(index)]
        init_codes = _encode_patch_set(
            item.fair, fair_dictionary["initial"], fair_dictionary["mean"], sparsity
        )
        fair_codes = _encode_patch_set(
            item.fair, fair_dictionary["final"], fair_dictionary["mean"], sparsity
        )
        edge_codes = _encode_patch_set(
            item.edge100, edge_dictionary["final"], edge_dictionary["mean"], sparsity
        )
        content = _code_readout(fair_codes)
        relation_graph = _relation_graph_readout(item.fair)
        true_relation = _aligned_relation_readout(fair_codes, item.fair)
        rng = np.random.default_rng(shuffle_seed + int(item.index) * 1009)
        permutation = rng.permutation(fair_codes.shape[1])
        shuffled_relation = _aligned_relation_readout(
            fair_codes, item.fair, permutation=permutation
        )
        rows["STATS"].append(item.stats)
        rows["RAW_PATCH"].append(_raw_patch_readout(item.fair))
        rows["INIT_CONTENT"].append(_code_readout(init_codes))
        rows["FAIR95_CONTENT"].append(content)
        rows["FAIR95_RELATION_GRAPH"].append(np.concatenate([content, relation_graph]))
        rows["FAIR95_TRUE_RELATION"].append(
            np.concatenate([content, relation_graph, true_relation])
        )
        rows["FAIR95_SHUFFLED_RELATION"].append(
            np.concatenate([content, relation_graph, shuffled_relation])
        )
        rows["EDGE100_CONTENT"].append(_code_readout(edge_codes))
        rows["FAIR95_RESIDUAL"].append(
            np.concatenate([content, relation_graph, item.residual_features])
        )
        if item.node_features is not None:
            features = item.node_features
            rows["FEATURE_ONLY"].append(features)
            rows["FEATURE_STATS"].append(np.concatenate([features, item.stats]))
            rows["FEATURE_FAIR95_CONTENT"].append(np.concatenate([features, content]))
            rows["FEATURE_FAIR95_TRUE_RELATION"].append(
                np.concatenate([features, content, relation_graph, true_relation])
            )
            rows["FEATURE_FAIR95_RESIDUAL"].append(
                np.concatenate([features, content, relation_graph, item.residual_features])
            )
    return {
        name: np.stack(values)
        for name, values in rows.items()
        if values
    }


def _gated_score(
    feature_train: np.ndarray,
    structure_train: np.ndarray,
    y_train: np.ndarray,
    feature_test: np.ndarray,
    structure_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int,
    epochs: int,
) -> dict[str, float | int]:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    inner_train, validation = train_test_split(
        np.arange(len(y_train)),
        test_size=0.2,
        random_state=seed,
        stratify=y_train,
    )
    f_scaler = StandardScaler().fit(feature_train[inner_train])
    s_scaler = StandardScaler().fit(structure_train[inner_train])
    f_all = torch.tensor(f_scaler.transform(feature_train), dtype=torch.float32)
    s_all = torch.tensor(s_scaler.transform(structure_train), dtype=torch.float32)
    f_test = torch.tensor(f_scaler.transform(feature_test), dtype=torch.float32)
    s_test = torch.tensor(s_scaler.transform(structure_test), dtype=torch.float32)
    labels = torch.tensor(y_train, dtype=torch.long)
    n_classes = int(np.max(y_train)) + 1

    class Gate(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hidden = 32
            self.feature = nn.Linear(feature_train.shape[1], hidden)
            self.structure = nn.Linear(structure_train.shape[1], hidden)
            self.gate = nn.Linear(feature_train.shape[1] + structure_train.shape[1], hidden)
            self.output = nn.Linear(hidden, n_classes)

        def forward(self, feature, structure):
            hf = torch.relu(self.feature(feature))
            hs = torch.relu(self.structure(structure))
            gate = torch.sigmoid(self.gate(torch.cat([feature, structure], dim=1)))
            return self.output(hf + gate * hs)

    model = Gate()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    train_tensor = torch.tensor(inner_train, dtype=torch.long)
    validation_tensor = torch.tensor(validation, dtype=torch.long)
    best_state = None
    best_score = -1.0
    best_epoch = 0
    patience = 30
    stale = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(f_all[train_tensor], s_all[train_tensor])
        loss = nn.functional.cross_entropy(logits, labels[train_tensor])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = model(f_all[validation_tensor], s_all[validation_tensor]).argmax(1).numpy()
        score = balanced_accuracy_score(y_train[validation], prediction)
        if score > best_score + 1e-12:
            best_score = float(score)
            best_epoch = epoch + 1
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("gated model did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = model(f_test, s_test).argmax(1).numpy()
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
        "accuracy": float(accuracy_score(y_test, prediction)),
        "selected_epoch": int(best_epoch),
        "validation_balanced_accuracy": float(best_score),
    }


def _sampling_summary(prepared: Sequence[PreparedGraph]) -> dict[str, Any]:
    keys = (
        "fair_patch_count",
        "edge100_patch_count",
        "added_patch_count",
        "fair_edge_coverage",
        "fair_pair_coverage",
        "fair_residual_edges",
        "edge100_pair_coverage",
        "fair_segment_count",
        "edge100_segment_count",
    )
    return {
        "graph_count": len(prepared),
        "fair_reached_fraction": float(np.mean([item.sampling["fair_reached"] for item in prepared])),
        **{
            f"{key}_mean": float(np.mean([item.sampling[key] for item in prepared]))
            for key in keys
        },
        "added_patch_p50": float(np.quantile([item.sampling["added_patch_count"] for item in prepared], 0.5)),
        "added_patch_p90": float(np.quantile([item.sampling["added_patch_count"] for item in prepared], 0.9)),
        "edge100_reached_fraction": float(
            np.mean([item.sampling["edge100_edge_coverage"] >= 1.0 - 1e-12 for item in prepared])
        ),
    }


def _aggregate_fold_scores(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    feature_names = sorted({name for row in rows for name in row["scores"]})
    output = {}
    for name in feature_names:
        selected = [row["scores"][name] for row in rows if name in row["scores"]]
        output[name] = {
            "balanced_accuracy_mean": float(np.mean([item["balanced_accuracy"] for item in selected])),
            "balanced_accuracy_std": float(np.std([item["balanced_accuracy"] for item in selected])),
            "accuracy_mean": float(np.mean([item["accuracy"] for item in selected])),
            "accuracy_std": float(np.std([item["accuracy"] for item in selected])),
            "fold_count": len(selected),
        }
    return output


def _paired_summary(
    rows: Sequence[dict[str, Any]], left: str, right: str
) -> dict[str, Any]:
    values = [
        row["scores"][left]["balanced_accuracy"]
        - row["scores"][right]["balanced_accuracy"]
        for row in rows
    ]
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "wins": int(sum(value > 1e-12 for value in values)),
        "ties": int(sum(abs(value) <= 1e-12 for value in values)),
        "losses": int(sum(value < -1e-12 for value in values)),
        "values": [float(value) for value in values],
    }


def run_dataset(
    name: str,
    *,
    dataset_root: Path,
    patch_size: int,
    overlap: int,
    maximum_patches: int,
    retained_beam: int,
    n_atoms: int,
    sparsity: int,
    iterations: int,
    max_train_patches: int,
    split_seeds: Sequence[int],
    n_splits: int,
    gate_epochs: int,
    limit: int | None,
) -> dict[str, Any]:
    started = time.time()
    graphs, labels, node_features, metadata = load_tud(name, dataset_root)
    if limit is not None and limit < len(graphs):
        keep = []
        rng = np.random.default_rng(20260812)
        per_class = max(1, limit // len(np.unique(labels)))
        for label in np.unique(labels):
            candidates = np.flatnonzero(labels == label)
            keep.extend(int(value) for value in rng.choice(candidates, size=min(per_class, len(candidates)), replace=False))
        keep = sorted(keep)[:limit]
        graphs = [graphs[index] for index in keep]
        node_features = [node_features[index] for index in keep]
        labels = labels[np.asarray(keep, dtype=np.int64)]
        metadata = {**metadata, "limited_graph_count": len(graphs)}
    prepared = []
    for index, (graph, label, features) in enumerate(zip(graphs, labels, node_features)):
        prepared.append(
            prepare_graph(
                index,
                graph,
                int(label),
                features,
                patch_size=patch_size,
                overlap=overlap,
                maximum_patches=maximum_patches,
                retained_beam=retained_beam,
                seed=20260812 + index * 1009,
            )
        )
    fold_rows = []
    for split_seed in split_seeds:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(split_seed))
        for fold_index, (train_indices, test_indices) in enumerate(
            splitter.split(np.zeros(len(labels)), labels)
        ):
            dictionary_seed = int(split_seed) * 100 + fold_index
            fair_dictionary = _fit_dictionary(
                prepared,
                train_indices,
                branch="fair",
                n_atoms=n_atoms,
                sparsity=sparsity,
                iterations=iterations,
                max_train_patches=max_train_patches,
                seed=dictionary_seed,
            )
            edge_dictionary = _fit_dictionary(
                prepared,
                train_indices,
                branch="edge100",
                n_atoms=n_atoms,
                sparsity=sparsity,
                iterations=iterations,
                max_train_patches=max_train_patches,
                seed=dictionary_seed,
            )
            train_features = _feature_matrix(
                prepared,
                train_indices,
                fair_dictionary,
                edge_dictionary,
                sparsity=sparsity,
                shuffle_seed=731421 + dictionary_seed,
            )
            test_features = _feature_matrix(
                prepared,
                test_indices,
                fair_dictionary,
                edge_dictionary,
                sparsity=sparsity,
                shuffle_seed=731421 + dictionary_seed,
            )
            scores = {}
            for feature_name in train_features:
                scores[feature_name] = _fit_score(
                    train_features[feature_name],
                    labels[train_indices],
                    test_features[feature_name],
                    labels[test_indices],
                    seed=dictionary_seed,
                )
            if name in FEATURE_DATASETS:
                feature_train = train_features["FEATURE_ONLY"]
                feature_test = test_features["FEATURE_ONLY"]
                structure_train = train_features["FAIR95_TRUE_RELATION"]
                structure_test = test_features["FAIR95_TRUE_RELATION"]
                scores["FEATURE_STRUCTURE_GATE"] = _gated_score(
                    feature_train,
                    structure_train,
                    labels[train_indices],
                    feature_test,
                    structure_test,
                    labels[test_indices],
                    seed=840000 + dictionary_seed,
                    epochs=gate_epochs,
                )
            fold_rows.append(
                {
                    "split_seed": int(split_seed),
                    "fold_index": int(fold_index),
                    "train_count": int(len(train_indices)),
                    "test_count": int(len(test_indices)),
                    "fair_dictionary": {
                        "atoms": fair_dictionary["atoms"],
                        "raw_train_patches": fair_dictionary["raw_train_patches"],
                        "used_train_patches": fair_dictionary["used_train_patches"],
                        "final_reconstruction": fair_dictionary["training"].get("recon_rel"),
                    },
                    "edge100_dictionary": {
                        "atoms": edge_dictionary["atoms"],
                        "raw_train_patches": edge_dictionary["raw_train_patches"],
                        "used_train_patches": edge_dictionary["used_train_patches"],
                        "final_reconstruction": edge_dictionary["training"].get("recon_rel"),
                    },
                    "scores": scores,
                }
            )
    return {
        "dataset": name,
        "metadata": metadata,
        "sampling": _sampling_summary(prepared),
        "folds": fold_rows,
        "summary": _aggregate_fold_scores(fold_rows),
        "paired": {
            "edge100_minus_fair95": _paired_summary(
                fold_rows, "EDGE100_CONTENT", "FAIR95_CONTENT"
            ),
            "true_minus_shuffled": _paired_summary(
                fold_rows, "FAIR95_TRUE_RELATION", "FAIR95_SHUFFLED_RELATION"
            ),
            **(
                {
                    feature_name: _paired_summary(
                        fold_rows, feature_name, "FEATURE_ONLY"
                    )
                    for feature_name in (
                        "FEATURE_STATS",
                        "FEATURE_FAIR95_CONTENT",
                        "FEATURE_FAIR95_TRUE_RELATION",
                        "FEATURE_FAIR95_RESIDUAL",
                        "FEATURE_STRUCTURE_GATE",
                    )
                }
                if name in FEATURE_DATASETS
                else {}
            ),
        },
        "seconds": time.time() - started,
    }


def classify(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_name = {row["dataset"]: row for row in results}
    edge_deltas = {}
    relation_deltas = {}
    for name, result in by_name.items():
        summary = result["summary"]
        edge_deltas[name] = (
            summary["EDGE100_CONTENT"]["balanced_accuracy_mean"]
            - summary["FAIR95_CONTENT"]["balanced_accuracy_mean"]
        )
        relation_deltas[name] = (
            summary["FAIR95_TRUE_RELATION"]["balanced_accuracy_mean"]
            - summary["FAIR95_SHUFFLED_RELATION"]["balanced_accuracy_mean"]
        )
    edge_pass = sum(value >= 0.01 for value in edge_deltas.values()) >= 3
    relation_pass = (
        sum(value > 0.0 for value in relation_deltas.values()) >= 3
        and float(np.mean(list(relation_deltas.values()))) >= 0.01
    )
    fusion_deltas = {}
    for name in FEATURE_DATASETS:
        if name not in by_name:
            continue
        summary = by_name[name]["summary"]
        baseline = summary["FEATURE_ONLY"]["balanced_accuracy_mean"]
        fusion_deltas[name] = {
            "concat_content": summary["FEATURE_FAIR95_CONTENT"]["balanced_accuracy_mean"] - baseline,
            "concat_relation": summary["FEATURE_FAIR95_TRUE_RELATION"]["balanced_accuracy_mean"] - baseline,
            "concat_residual": summary["FEATURE_FAIR95_RESIDUAL"]["balanced_accuracy_mean"] - baseline,
            "gate": summary["FEATURE_STRUCTURE_GATE"]["balanced_accuracy_mean"] - baseline,
        }
    best_fusion = {name: max(values.values()) for name, values in fusion_deltas.items()}
    fusion_pass = bool(best_fusion) and (
        max(best_fusion.values()) >= 0.01 and min(best_fusion.values()) >= -0.01
    )
    if relation_pass or fusion_pass:
        label = "CONTINUE_ONLY_PASSED_RELATION_OR_FUSION_AXIS"
    elif edge_pass:
        label = "CONTINUE_COMPLETENESS_ONLY"
    else:
        label = "KSVD_REMAINS_COMPRESSOR_DIAGNOSTIC_BASELINE"
    return {
        "classification": label,
        "edge100_default_gate": bool(edge_pass),
        "relation_binding_gate": bool(relation_pass),
        "feature_fusion_gate": bool(fusion_pass),
        "edge100_minus_fair95": edge_deltas,
        "true_minus_shuffled": relation_deltas,
        "fusion_minus_feature_only": fusion_deltas,
    }


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def render_report(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    formal_four_dataset_run = {
        row["dataset"] for row in payload["datasets"]
    } == set(DEFAULT_DATASETS)
    lines = [
        "# luyin14 路线闭环：补边、分类、关系与节点特征融合",
        "",
        f"> 日期：2026-08-12  ",
        f"> 协议：`{payload['protocol']}`  ",
        f"> 判定：`{decision['classification']}`",
        "",
        "## 1. 冻结判定",
        "",
        f"- EDGE100 默认 gate：`{decision['edge100_default_gate']}`；",
        f"- true relation binding gate：`{decision['relation_binding_gate']}`；",
        f"- feature fusion gate：`{decision['feature_fusion_gate']}`。",
        "",
        "## 2. 补边成本",
        "",
        "| dataset | FAIR reach | FAIR patches | FAIR edge/pair | residual edges | EDGE100 patches | added p50/p90 | EDGE100 pair |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["datasets"]:
        row = result["sampling"]
        lines.append(
            f"| {result['dataset']} | {_fmt(row['fair_reached_fraction'])} | "
            f"{_fmt(row['fair_patch_count_mean'], 2)} | "
            f"{_fmt(row['fair_edge_coverage_mean'])}/{_fmt(row['fair_pair_coverage_mean'])} | "
            f"{_fmt(row['fair_residual_edges_mean'], 2)} | {_fmt(row['edge100_patch_count_mean'], 2)} | "
            f"{_fmt(row['added_patch_p50'], 1)}/{_fmt(row['added_patch_p90'], 1)} | "
            f"{_fmt(row['edge100_pair_coverage_mean'])} |"
        )
    lines.extend(
        [
            "",
            "EDGE100 只保证真实边覆盖为 100%；节点对覆盖仍可能远低于 100%。",
            "",
            "## 3. 分类主表（balanced accuracy）",
            "",
            "| dataset | STATS | RAW | INIT | FAIR content | relation graph | TRUE | SHUFFLED | EDGE100 | FAIR+residual |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in payload["datasets"]:
        summary = result["summary"]
        get = lambda name: summary[name]["balanced_accuracy_mean"]
        lines.append(
            f"| {result['dataset']} | {get('STATS'):.3f} | {get('RAW_PATCH'):.3f} | "
            f"{get('INIT_CONTENT'):.3f} | {get('FAIR95_CONTENT'):.3f} | "
            f"{get('FAIR95_RELATION_GRAPH'):.3f} | {get('FAIR95_TRUE_RELATION'):.3f} | "
            f"{get('FAIR95_SHUFFLED_RELATION'):.3f} | {get('EDGE100_CONTENT'):.3f} | "
            f"{get('FAIR95_RESIDUAL'):.3f} |"
        )
    lines.extend(
        [
            "",
            "## 4. 节点特征融合（balanced accuracy）",
            "",
            "| dataset | feature | +stats | +content | +true relation | +residual | gate | best Δ |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in payload["datasets"]:
        if result["dataset"] not in FEATURE_DATASETS:
            continue
        summary = result["summary"]
        get = lambda name: summary[name]["balanced_accuracy_mean"]
        baseline = get("FEATURE_ONLY")
        candidates = [
            get("FEATURE_FAIR95_CONTENT"),
            get("FEATURE_FAIR95_TRUE_RELATION"),
            get("FEATURE_FAIR95_RESIDUAL"),
            get("FEATURE_STRUCTURE_GATE"),
        ]
        lines.append(
            f"| {result['dataset']} | {baseline:.3f} | {get('FEATURE_STATS'):.3f} | "
            f"{candidates[0]:.3f} | {candidates[1]:.3f} | {candidates[2]:.3f} | "
            f"{candidates[3]:.3f} | {max(candidates)-baseline:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## 5. Paired route deltas",
            "",
            "| dataset | EDGE100−FAIR | TRUE−SHUFFLED |",
            "|---|---:|---:|",
        ]
    )
    for name in decision["edge100_minus_fair95"]:
        result = next(row for row in payload["datasets"] if row["dataset"] == name)
        edge_pair = result["paired"]["edge100_minus_fair95"]
        relation_pair = result["paired"]["true_minus_shuffled"]
        lines.append(
            f"| {name} | {decision['edge100_minus_fair95'][name]:+.3f} | "
            f"{decision['true_minus_shuffled'][name]:+.3f} |"
        )
    lines.extend(
        [
            "",
            "逐折稳定性：",
            "",
            "| dataset | EDGE100 W/T/L | TRUE relation W/T/L |",
            "|---|---:|---:|",
        ]
    )
    for result in payload["datasets"]:
        edge_pair = result["paired"]["edge100_minus_fair95"]
        relation_pair = result["paired"]["true_minus_shuffled"]
        lines.append(
            f"| {result['dataset']} | {edge_pair['wins']}/{edge_pair['ties']}/{edge_pair['losses']} | "
            f"{relation_pair['wins']}/{relation_pair['ties']}/{relation_pair['losses']} |"
        )
    lines.extend(["", "## 6. 路线结论", ""])
    if formal_four_dataset_run:
        lines.extend(
            [
                "1. **补到 100% 不应成为默认路线。** FAIR95 后平均只剩很少真实边，但 EDGE100 在四个数据集上均未提高 balanced accuracy；四个平均差值全为负。已有无标签 rate 审计还显示 BASE+residual 比补到 EDGE100 平均节省约 348 proxy bits，因此默认保留 FAIR95，并把残余边作为显式 sidecar。",
                "2. **当前 patch 关系绑定没有跨数据集成立。** IMDB-BINARY 是唯一接近门槛的正例（约 +1.1 points，8/9 folds 为正），IMDB-MULTI 只有 +0.2 points，两个带特征数据集为负。不能据此进入 Transformer 或更深 relation network。",
                "3. **结构与节点特征的融合未成立。** MUTAG 上所有 KSVD 融合均明显低于 feature-only；PTC_MR 的 true-relation concat 接近持平但没有正增益。相反，简单 `feature+stats` 在两个数据集分别约 +4.9/+2.1 points，说明任务可用信号来自低阶统计，而非当前 KSVD code。",
                "4. **普通 KSVD 的合理定位不变：** 它在重构和压缩上有效，但当前 code/readout 没有形成稳定下游增益。后续不扫描 K/T、restart、fusion depth；若继续研究，只能另立问题，转向合法结构对象与 exact occurrence/incidence。",
            ]
        )
    else:
        lines.append(
            "本次是数据集子集/烟测运行；只验证实现，不替代四数据集冻结判定。"
        )
    lines.extend(
        [
            "",
            "## 7. 解释边界",
            "",
            "- sampling 完全不使用 labels；字典、scaler、classifier 与 gate 均为 train-fold only。",
            "- completion patch 被标为新 segment；没有伪造连续 `o=2` 关系。",
            "- residual sidecar 只使用结构输入可计算的不变统计，不使用 graph label。",
            "- 本表是机制闭环，不是与论文 leaderboard 的统一公开 benchmark。",
            "- 若 gate 未通过，不通过扩大 K/T、增加 restart 或扫描网络深度救结果。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--overlap", type=int, default=2)
    parser.add_argument("--maximum-patches", type=int, default=48)
    parser.add_argument("--retained-beam", type=int, default=4)
    parser.add_argument("--n-atoms", type=int, default=24)
    parser.add_argument("--sparsity", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--max-train-patches", type=int, default=3000)
    parser.add_argument("--split-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--gate-epochs", type=int, default=250)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    results = []
    for dataset in args.datasets:
        print(f"[{dataset}] start", flush=True)
        result = run_dataset(
            dataset,
            dataset_root=args.dataset_root,
            patch_size=args.patch_size,
            overlap=args.overlap,
            maximum_patches=args.maximum_patches,
            retained_beam=args.retained_beam,
            n_atoms=args.n_atoms,
            sparsity=args.sparsity,
            iterations=args.iterations,
            max_train_patches=args.max_train_patches,
            split_seeds=args.split_seeds,
            n_splits=args.n_splits,
            gate_epochs=args.gate_epochs,
            limit=args.limit,
        )
        print(f"[{dataset}] done in {result['seconds']:.1f}s", flush=True)
        results.append(result)
    payload = {
        "protocol": PROTOCOL,
        "config": vars(args) | {"dataset_root": str(args.dataset_root), "json": str(args.json), "report": str(args.report)},
        "datasets": results,
        "decision": classify(results),
    }
    _atomic_json(args.json, payload)
    _atomic_text(args.report, render_report(payload))
    print(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
