"""MolHIV patch--path pooling with one explicit, non-message-passing MLP.

This is the MolHIV counterpart of ``zinc_patch_path_pooling``.  Every atom is
used as a centre.  Its radius-2 induced neighbourhood is represented by an
exact rooted typed certificate over the complete OGB atom/bond attributes,
plus a fixed-width shell descriptor.  Every unordered pair of centres is
then evaluated independently using a shortest-path-conditioned relation.  The
pair values are pooled by distance bucket with invariant first/second moments
and pair mass, followed by one graph-level MLP head.

There is deliberately no centre-to-centre message passing, attention, or
second predictive model.  The official MolHIV scaffold validation split
selects the epoch; test is evaluated only after a train+validation refit at
that frozen epoch.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter, deque
from dataclasses import dataclass
import gc
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from ksvd_research.data import load_molhiv


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = REPO_ROOT / "tracks/ksvd/configs/luyin16/molhiv_patch_path_pooling.yaml"

# OGB's ogbg-molhiv feature schema.  The values are categorical indices, not
# numerical measurements; the exact patch certificate retains all coordinates.
ATOM_FEATURE_DIMS = (119, 5, 12, 12, 10, 6, 6, 2, 2)
BOND_FEATURE_DIMS = (5, 6, 2)
ATOM_WIDTH = int(sum(ATOM_FEATURE_DIMS))
BOND_WIDTH = int(sum(BOND_FEATURE_DIMS))

PATCH_RADIUS = 2
DISTANCE_BUCKETS = 6  # 1, 2, 3, 4, 5+, disconnected
DISCONNECTED_BUCKET = DISTANCE_BUCKETS - 1
SHELL_PAIRS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
SHELL_WIDTH = (
    (PATCH_RADIUS + 1) * ATOM_WIDTH
    + len(SHELL_PAIRS) * BOND_WIDTH
    + ATOM_WIDTH
    + BOND_WIDTH
    + 6
)

# distance one-hot, log distance, five overlap/size fields, three boundary
# overlap fields, shortest-path bond-feature composition, log path count, and
# adjacent bond-feature one-hot.
RELATION_WIDTH = (
    DISTANCE_BUCKETS
    + 1
    + 5
    + 3
    + BOND_WIDTH
    + 1
    + BOND_WIDTH
)

# The global context is only a label-free graph summary.  It is included to
# make this a direct analogue of the ZINC implementation's global context,
# while the main representation remains the explicit patch-pair route.
GLOBAL_SHORT_WIDTH = 15
GLOBAL_LONG_WIDTH = 15
GLOBAL_WIDTH = GLOBAL_SHORT_WIDTH + GLOBAL_LONG_WIDTH + ATOM_WIDTH + BOND_WIDTH

PATCH_HIDDEN = 64
PAIR_HIDDEN = 32


@dataclass(frozen=True)
class PatchRecord:
    typed_certificate: bytes
    parent_certificate: bytes
    nodes: frozenset[int]
    boundary: frozenset[int]
    shell_descriptor: np.ndarray


@dataclass(frozen=True)
class GraphRecord:
    patches: tuple[PatchRecord, ...]
    pair_index: np.ndarray
    pair_relation: np.ndarray
    pair_bucket: np.ndarray
    global_context: np.ndarray
    y: float


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _feature_one_hot(values: Sequence[int], dimensions: Sequence[int]) -> np.ndarray:
    values_array = np.asarray(values, dtype=np.int64).reshape(-1)
    if values_array.size != len(dimensions):
        raise ValueError(
            f"feature width {values_array.size} does not match schema {len(dimensions)}"
        )
    output = np.zeros(int(sum(dimensions)), dtype=np.float32)
    offset = 0
    for value, width in zip(values_array, dimensions, strict=True):
        value_int = int(value)
        if value_int < 0 or value_int >= int(width):
            raise ValueError(f"feature value {value_int} outside [0,{int(width)})")
        output[offset + value_int] = 1.0
        offset += int(width)
    return output


def _histogram_features(
    values: np.ndarray,
    dimensions: Sequence[int],
    *,
    denominator: float,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[1] != len(dimensions):
        raise ValueError(f"unexpected categorical matrix shape {matrix.shape}")
    rows: list[np.ndarray] = []
    for column, width in enumerate(dimensions):
        counts = np.bincount(matrix[:, column], minlength=int(width)).astype(np.float32)
        rows.append(counts / max(float(denominator), 1.0))
    return np.concatenate(rows).astype(np.float32, copy=False)


def _ego_distances(graph: Any, center: int, radius: int) -> dict[int, int]:
    distances = {int(center): 0}
    queue: deque[int] = deque([int(center)])
    while queue:
        node = queue.popleft()
        if distances[node] >= int(radius):
            continue
        for neighbour in sorted(graph.neighbors(node)):
            neighbour = int(neighbour)
            if neighbour not in distances:
                distances[neighbour] = distances[node] + 1
                queue.append(neighbour)
    return distances


def _canonical_typed_patch(
    graph: Any,
    center: int,
    node_types: np.ndarray,
    edge_types: Mapping[tuple[int, int], tuple[int, ...]],
    *,
    radius: int,
) -> bytes:
    """Canonical certificate for a rooted patch with vector-valued labels."""
    try:
        import pynauty
    except ImportError as exc:  # pragma: no cover - dependency diagnostic
        raise RuntimeError("this experiment requires pynauty==2.8.8.1") from exc

    distances = _ego_distances(graph, int(center), int(radius))
    original_nodes = tuple(sorted(distances))
    node_to_local = {node: index for index, node in enumerate(original_nodes)}
    induced = graph.induced(set(original_nodes))
    original_edges = tuple(sorted(induced.edges()))
    local_edges = tuple(
        (node_to_local[int(left)], node_to_local[int(right)])
        for left, right in original_edges
    )
    n_nodes = len(original_nodes)
    n_edges = len(local_edges)

    adjacency: dict[int, list[int]] = {
        vertex: [] for vertex in range(n_nodes + n_edges)
    }
    color_groups: dict[tuple[Any, ...], set[int]] = {}
    root_local = node_to_local[int(center)]
    for local, node in enumerate(original_nodes):
        node_key = (
            "node",
            int(local == root_local),
            int(distances[node]),
            tuple(int(value) for value in node_types[int(node)]),
        )
        color_groups.setdefault(node_key, set()).add(local)
    for edge_local, (left, right) in enumerate(local_edges):
        edge_vertex = n_nodes + edge_local
        adjacency[left].append(edge_vertex)
        adjacency[right].append(edge_vertex)
        adjacency[edge_vertex] = [left, right]
        edge_values = edge_types[
            graph.edge_key(int(original_nodes[left]), int(original_nodes[right]))
        ]
        color_groups.setdefault(("edge", tuple(int(value) for value in edge_values)), set()).add(
            edge_vertex
        )

    coloring = [color_groups[key] for key in sorted(color_groups, key=repr)]
    incidence = pynauty.Graph(
        number_of_vertices=n_nodes + n_edges,
        directed=False,
        adjacency_dict=adjacency,
        vertex_coloring=coloring,
    )
    return bytes(pynauty.certificate(incidence))


def _patch_cache_key(
    graph: Any,
    center: int,
    node_types: np.ndarray,
    edge_types: Mapping[tuple[int, int], tuple[int, ...]],
    radius: int,
) -> bytes:
    distances = _ego_distances(graph, int(center), int(radius))
    nodes = tuple(sorted(distances))
    node_to_local = {node: index for index, node in enumerate(nodes)}
    induced = graph.induced(set(nodes))
    payload: list[Any] = [node_to_local[int(center)], int(radius)]
    payload.extend(
        (
            int(distances[node]),
            tuple(int(value) for value in node_types[int(node)]),
        )
        for node in nodes
    )
    for left, right in sorted(induced.edges()):
        key = graph.edge_key(int(left), int(right))
        payload.append(
            (
                node_to_local[int(left)],
                node_to_local[int(right)],
                tuple(int(value) for value in edge_types[key]),
            )
        )
    return repr(tuple(payload)).encode("ascii")


def _typed_certificate(
    graph: Any,
    center: int,
    node_types: np.ndarray,
    edge_types: Mapping[tuple[int, int], tuple[int, ...]],
    radius: int,
    cache: dict[bytes, bytes],
) -> bytes:
    key = _patch_cache_key(graph, center, node_types, edge_types, radius)
    certificate = cache.get(key)
    if certificate is None:
        certificate = _canonical_typed_patch(
            graph,
            center,
            node_types,
            edge_types,
            radius=int(radius),
        )
        cache[key] = certificate
    return certificate


def _shell_descriptor(
    graph: Any,
    center: int,
    node_types: np.ndarray,
    edge_types: Mapping[tuple[int, int], tuple[int, ...]],
    distances: Mapping[int, int],
) -> tuple[np.ndarray, frozenset[int], frozenset[int]]:
    nodes = frozenset(int(node) for node in distances)
    induced = graph.induced(set(nodes))
    n_nodes = len(nodes)
    n_edges = induced.num_edges()

    atom_shell = np.zeros((PATCH_RADIUS + 1, ATOM_WIDTH), dtype=np.float32)
    for node in nodes:
        atom_shell[int(distances[node])] += _feature_one_hot(
            node_types[int(node)], ATOM_FEATURE_DIMS
        )
    atom_shell /= max(float(n_nodes), 1.0)

    bond_shell = np.zeros((len(SHELL_PAIRS), BOND_WIDTH), dtype=np.float32)
    shell_pair_index = {pair: index for index, pair in enumerate(SHELL_PAIRS)}
    for left, right in induced.edges():
        pair = tuple(sorted((int(distances[left]), int(distances[right]))))
        shell_index = shell_pair_index.get(pair)
        if shell_index is None:
            raise RuntimeError(f"unexpected shell pair {pair}")
        bond_shell[shell_index] += _feature_one_hot(
            edge_types[graph.edge_key(int(left), int(right))], BOND_FEATURE_DIMS
        )
    bond_shell /= max(float(n_edges), 1.0)

    root_atom = _feature_one_hot(node_types[int(center)], ATOM_FEATURE_DIMS)
    incident_bonds = np.zeros(BOND_WIDTH, dtype=np.float32)
    neighbours = list(graph.neighbors(int(center)))
    for neighbour in neighbours:
        incident_bonds += _feature_one_hot(
            edge_types[graph.edge_key(int(center), int(neighbour))], BOND_FEATURE_DIMS
        )
    incident_bonds /= max(float(len(neighbours)), 1.0)

    cycle_rank = max(int(n_edges) - int(n_nodes) + 1, 0)
    degrees = np.asarray([len(graph.neighbors(node)) for node in nodes], dtype=np.float32)
    scalars = np.asarray(
        [
            np.log1p(float(n_nodes)),
            np.log1p(float(n_edges)),
            float(sum(int(distance) == PATCH_RADIUS for distance in distances.values()))
            / max(float(n_nodes), 1.0),
            float(cycle_rank) / max(float(n_nodes), 1.0),
            float(len(neighbours)) / 4.0,
            float(degrees.mean()) / 4.0 if degrees.size else 0.0,
        ],
        dtype=np.float32,
    )
    descriptor = np.concatenate(
        [atom_shell.reshape(-1), bond_shell.reshape(-1), root_atom, incident_bonds, scalars]
    ).astype(np.float32, copy=False)
    if descriptor.shape != (SHELL_WIDTH,):
        raise RuntimeError(f"shell descriptor width changed: {descriptor.shape}")
    boundary = frozenset(
        int(node) for node, distance in distances.items() if int(distance) == PATCH_RADIUS
    )
    return descriptor, nodes, boundary


def _one_hot_bucket(value: int, width: int) -> np.ndarray:
    output = np.zeros(int(width), dtype=np.float32)
    output[int(value)] = 1.0
    return output


def _shortest_path_summary(
    graph: Any,
    source: int,
    edge_types: Mapping[tuple[int, int], tuple[int, ...]],
) -> dict[int, tuple[int, float, np.ndarray]]:
    """Distance, shortest-path count and mean bond composition to each node."""
    source = int(source)
    distances = {source: 0}
    path_counts: dict[int, float] = {source: 1.0}
    bond_sums: dict[int, np.ndarray] = {
        source: np.zeros(BOND_WIDTH, dtype=np.float64)
    }
    queue: deque[int] = deque([source])
    while queue:
        node = queue.popleft()
        for neighbour in sorted(graph.neighbors(node)):
            neighbour = int(neighbour)
            bond_vector = _feature_one_hot(
                edge_types[graph.edge_key(node, neighbour)], BOND_FEATURE_DIMS
            ).astype(np.float64, copy=False)
            if neighbour not in distances:
                distances[neighbour] = distances[node] + 1
                path_counts[neighbour] = 0.0
                bond_sums[neighbour] = np.zeros(BOND_WIDTH, dtype=np.float64)
                queue.append(neighbour)
            if distances[neighbour] != distances[node] + 1:
                continue
            path_counts[neighbour] += path_counts[node]
            bond_sums[neighbour] += bond_sums[node]
            bond_sums[neighbour] += path_counts[node] * bond_vector

    output: dict[int, tuple[int, float, np.ndarray]] = {}
    for node, distance in distances.items():
        count = max(float(path_counts[node]), 1.0)
        output[int(node)] = (
            int(distance),
            float(path_counts[node]),
            (bond_sums[node] / count).astype(np.float32),
        )
    return output


def _pair_relation(
    left: PatchRecord,
    right: PatchRecord,
    path_summary: tuple[int, float, np.ndarray] | None,
    adjacent_bond: tuple[int, ...] | None,
) -> tuple[np.ndarray, int]:
    if path_summary is None:
        # MolHIV contains salt/disconnected molecular graphs.  Keep those
        # centre pairs as a separate relation rather than inventing a path or
        # dropping the graph.  The explicit bucket is the indicator; the
        # continuous path fields are neutral/zero for this relation.
        distance = 0
        path_count = 0.0
        path_bond_mean = np.zeros(BOND_WIDTH, dtype=np.float32)
        bucket = DISCONNECTED_BUCKET
    else:
        distance, path_count, path_bond_mean = path_summary
        bucket = min(max(int(distance), 1), DISCONNECTED_BUCKET) - 1
    distance_one_hot = _one_hot_bucket(bucket, DISTANCE_BUCKETS)

    intersection = len(left.nodes & right.nodes)
    union = len(left.nodes | right.nodes)
    left_size = len(left.nodes)
    right_size = len(right.nodes)
    overlap = np.asarray(
        [
            float(intersection) / 20.0,
            float(intersection) / max(float(union), 1.0),
            float(intersection) / max(float(min(left_size, right_size)), 1.0),
            float(intersection) / max(float(max(left_size, right_size)), 1.0),
            float(abs(left_size - right_size)) / 20.0,
        ],
        dtype=np.float32,
    )
    boundary_intersection = len(left.boundary & right.boundary)
    boundary_union = len(left.boundary | right.boundary)
    boundary_min = min(len(left.boundary), len(right.boundary))
    boundary_features = np.asarray(
        [
            float(boundary_intersection) / max(float(boundary_union), 1.0),
            float(boundary_intersection) / max(float(boundary_min), 1.0),
            float(boundary_intersection > 0),
        ],
        dtype=np.float32,
    )
    adjacent = np.zeros(BOND_WIDTH, dtype=np.float32)
    if adjacent_bond is not None:
        adjacent = _feature_one_hot(adjacent_bond, BOND_FEATURE_DIMS)
    relation = np.concatenate(
        [
            distance_one_hot,
            np.asarray([np.log1p(float(distance))], dtype=np.float32),
            overlap,
            boundary_features,
            np.asarray(path_bond_mean, dtype=np.float32),
            np.asarray([np.log1p(float(path_count))], dtype=np.float32),
            adjacent,
        ]
    ).astype(np.float32, copy=False)
    if relation.shape != (RELATION_WIDTH,):
        raise RuntimeError(
            f"relation width changed: {relation.shape}; expected {RELATION_WIDTH}"
        )
    return relation, bucket


def _global_structure_blocks(graph: Any) -> tuple[np.ndarray, np.ndarray]:
    degree = np.asarray([len(graph.neighbors(node)) for node in graph.nodes], dtype=np.float64)
    density = 2.0 * graph.num_edges() / max(graph.n * (graph.n - 1), 1)
    triangles = sum(
        len(graph.neighbors(left) & graph.neighbors(right))
        for left, right in graph.edges()
    ) / 3.0
    clustering_values: list[float] = []
    for node in graph.nodes:
        neighbours = graph.neighbors(node)
        possible = len(neighbours) * (len(neighbours) - 1) / 2
        links = sum(
            1
            for left in neighbours
            for right in neighbours
            if left < right and right in graph.neighbors(left)
        )
        clustering_values.append(links / possible if possible else 0.0)
    degree_quantiles = (
        np.quantile(degree, [0.25, 0.50, 0.75])
        if degree.size
        else (0.0, 0.0, 0.0)
    )
    short = np.asarray(
        [
            graph.n,
            graph.num_edges(),
            density,
            degree.mean() if degree.size else 0.0,
            degree.std() if degree.size else 0.0,
            degree.min() if degree.size else 0.0,
            degree.max() if degree.size else 0.0,
            *degree_quantiles,
            float(np.mean(degree == 1)) if degree.size else 0.0,
            float(np.mean(degree >= 3)) if degree.size else 0.0,
            graph.num_edges() - graph.n + 1,
            triangles,
            float(np.mean(clustering_values)) if clustering_values else 0.0,
        ],
        dtype=np.float32,
    )

    pair_distances: list[float] = []
    eccentricities: list[float] = []
    for source in graph.nodes:
        distances = {int(source): 0}
        queue: deque[int] = deque([int(source)])
        while queue:
            node = queue.popleft()
            for neighbour in sorted(graph.neighbors(node)):
                neighbour = int(neighbour)
                if neighbour not in distances:
                    distances[neighbour] = distances[node] + 1
                    queue.append(neighbour)
        eccentricities.append(float(max(distances.values(), default=0)))
        pair_distances.extend(
            float(distances[target])
            for target in graph.nodes
            if target > source and target in distances
        )
    distance_array = np.asarray(pair_distances, dtype=np.float64)
    eccentricity_array = np.asarray(eccentricities, dtype=np.float64)
    if distance_array.size:
        long = np.asarray(
            [
                distance_array.mean(),
                distance_array.std(),
                distance_array.min(),
                distance_array.max(),
                *np.quantile(distance_array, [0.25, 0.50, 0.75, 0.90]),
                eccentricity_array.min(),
                eccentricity_array.mean(),
                eccentricity_array.std(),
                eccentricity_array.max(),
                float(np.mean(distance_array > 2)),
                float(np.mean(distance_array > 3)),
                float(np.mean(distance_array > 4)),
            ],
            dtype=np.float32,
        )
    else:
        long = np.zeros(GLOBAL_LONG_WIDTH, dtype=np.float32)
    if short.shape != (GLOBAL_SHORT_WIDTH,) or long.shape != (GLOBAL_LONG_WIDTH,):
        raise RuntimeError(f"global structure width changed: {short.shape}, {long.shape}")
    return short, long


def _global_context(
    graph: Any,
    node_types: np.ndarray,
    edge_types: Mapping[tuple[int, int], tuple[int, ...]],
) -> np.ndarray:
    short, long = _global_structure_blocks(graph)
    atom_hist = _histogram_features(
        np.asarray(node_types, dtype=np.int64),
        ATOM_FEATURE_DIMS,
        denominator=float(graph.n),
    )
    if edge_types:
        bond_matrix = np.asarray(list(edge_types.values()), dtype=np.int64)
    else:
        bond_matrix = np.zeros((0, len(BOND_FEATURE_DIMS)), dtype=np.int64)
    bond_hist = _histogram_features(
        bond_matrix,
        BOND_FEATURE_DIMS,
        denominator=float(len(edge_types)),
    )
    context = np.concatenate([short, long, atom_hist, bond_hist]).astype(
        np.float32, copy=False
    )
    if context.shape != (GLOBAL_WIDTH,):
        raise RuntimeError(f"global context width changed: {context.shape}; expected {GLOBAL_WIDTH}")
    return context


def _graph_record(
    graph: Any,
    node_types: np.ndarray,
    edge_types: Mapping[tuple[int, int], tuple[int, ...]],
    y: float,
    certificate_cache: dict[bytes, bytes],
) -> GraphRecord:
    centers = list(graph.nodes)
    patches: list[PatchRecord] = []
    for center in centers:
        distances = _ego_distances(graph, int(center), PATCH_RADIUS)
        descriptor, nodes, boundary = _shell_descriptor(
            graph, int(center), node_types, edge_types, distances
        )
        typed = _typed_certificate(
            graph, int(center), node_types, edge_types, PATCH_RADIUS, certificate_cache
        )
        parent = _typed_certificate(
            graph, int(center), node_types, edge_types, 1, certificate_cache
        )
        patches.append(
            PatchRecord(
                typed_certificate=typed,
                parent_certificate=parent,
                nodes=nodes,
                boundary=boundary,
                shell_descriptor=descriptor,
            )
        )

    shortest_paths = {
        int(source): _shortest_path_summary(graph, int(source), edge_types)
        for source in centers
    }
    pair_sources: list[int] = []
    pair_targets: list[int] = []
    pair_relations: list[np.ndarray] = []
    pair_buckets: list[int] = []
    for left_index, left_center in enumerate(centers):
        for right_index in range(left_index + 1, len(centers)):
            right_center = int(centers[right_index])
            path_summary = shortest_paths[int(left_center)].get(right_center)
            distance = int(path_summary[0]) if path_summary is not None else 0
            adjacent_bond = (
                edge_types[graph.edge_key(int(left_center), right_center)]
                if distance == 1
                else None
            )
            relation, bucket = _pair_relation(
                patches[left_index], patches[right_index], path_summary, adjacent_bond
            )
            pair_sources.append(int(left_index))
            pair_targets.append(int(right_index))
            pair_relations.append(relation)
            pair_buckets.append(int(bucket))

    pair_index = np.asarray([pair_sources, pair_targets], dtype=np.int64)
    if pair_relations:
        pair_relation = np.stack(pair_relations, axis=0).astype(np.float32, copy=False)
    else:
        pair_relation = np.zeros((0, RELATION_WIDTH), dtype=np.float32)
    pair_bucket = np.asarray(pair_buckets, dtype=np.int64)
    if pair_relation.shape != (len(pair_sources), RELATION_WIDTH):
        raise RuntimeError(f"pair relation shape changed: {pair_relation.shape}")
    return GraphRecord(
        patches=tuple(patches),
        pair_index=pair_index,
        pair_relation=pair_relation,
        pair_bucket=pair_bucket,
        global_context=_global_context(graph, node_types, edge_types),
        y=float(y),
    )


def _extract_split(
    bundle: Any,
    indices: Sequence[int],
    split: str,
    certificate_cache: dict[bytes, bytes],
) -> tuple[list[GraphRecord], dict[str, Any]]:
    started = time.perf_counter()
    records: list[GraphRecord] = []
    for position, index in enumerate(np.asarray(indices, dtype=np.int64)):
        index_int = int(index)
        graph = bundle.graphs[index_int]
        node_types = np.asarray(bundle.node_feats[index_int], dtype=np.int64)
        raw_edge_types = bundle.edge_feats[index_int]
        edge_types = {
            (int(left), int(right)): tuple(int(value) for value in values)
            for (left, right), values in raw_edge_types.items()
        }
        records.append(
            _graph_record(
                graph,
                node_types,
                edge_types,
                float(bundle.y[index_int]),
                certificate_cache,
            )
        )
        if (position + 1) % 500 == 0 or position + 1 == len(indices):
            print(
                f"MolHIV patch-path features {split}: {position + 1}/{len(indices)} "
                f"certificate_cache={len(certificate_cache)}",
                flush=True,
            )
    metadata = {
        "n_graphs": int(len(records)),
        "positive_graphs": int(sum(record.y > 0.5 for record in records)),
        "mean_centres": float(np.mean([len(record.patches) for record in records]))
        if records
        else 0.0,
        "mean_pairs": float(np.mean([record.pair_relation.shape[0] for record in records]))
        if records
        else 0.0,
        "mean_patch_nodes": float(
            np.mean([len(patch.nodes) for record in records for patch in record.patches])
        )
        if records
        else 0.0,
        "max_patch_nodes": int(
            max(
                (len(patch.nodes) for record in records for patch in record.patches),
                default=0,
            )
        ),
        "seconds": float(time.perf_counter() - started),
    }
    return records, metadata


class Standardizer:
    def __init__(self, mean: np.ndarray, scale: np.ndarray) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError(f"standardizer expects non-empty matrix, got {matrix.shape}")
        mean = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = matrix.std(axis=0, dtype=np.float64).astype(np.float32)
        scale[~np.isfinite(scale) | (scale < 1.0e-6)] = 1.0
        return cls(mean, scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        output = ((np.asarray(values, dtype=np.float32) - self.mean) / self.scale).astype(
            np.float32, copy=False
        )
        if not np.isfinite(output).all():
            raise FloatingPointError("non-finite standardized values")
        return output


def _fit_vocabulary(
    records: Sequence[GraphRecord],
    field: str,
    maximum: int,
    minimum_frequency: int,
) -> dict[bytes, int]:
    counts: Counter[bytes] = Counter()
    for record in records:
        for patch in record.patches:
            counts[getattr(patch, field)] += 1
    ordered = sorted(
        (key for key, count in counts.items() if count >= int(minimum_frequency)),
        key=lambda key: (-counts[key], key),
    )[: int(maximum)]
    return {key: index + 1 for index, key in enumerate(ordered)}


def _vocabulary_stats(
    records: Sequence[GraphRecord],
    vocabulary: Mapping[bytes, int],
    field: str,
) -> dict[str, Any]:
    values = [getattr(patch, field) for record in records for patch in record.patches]
    known = sum(value in vocabulary for value in values)
    types = set(values)
    known_types = sum(value in vocabulary for value in types)
    graph_all = sum(
        all(getattr(patch, field) in vocabulary for patch in record.patches)
        for record in records
    )
    return {
        "occurrences": int(len(values)),
        "unique_types": int(len(types)),
        "vocabulary_size": int(len(vocabulary)),
        "known_occurrence_fraction": float(known / max(len(values), 1)),
        "known_type_fraction": float(known_types / max(len(types), 1)),
        "all_patches_known_graph_fraction": float(graph_all / max(len(records), 1)),
    }


def _patch_matrix(records: Sequence[GraphRecord]) -> np.ndarray:
    return np.stack(
        [patch.shell_descriptor for record in records for patch in record.patches], axis=0
    ).astype(np.float32, copy=False)


def _context_matrix(records: Sequence[GraphRecord]) -> np.ndarray:
    return np.stack([record.global_context for record in records], axis=0).astype(
        np.float32, copy=False
    )


def _encode_records(
    records: Sequence[GraphRecord],
    typed_vocabulary: Mapping[bytes, int],
    parent_vocabulary: Mapping[bytes, int],
    patch_standardizer: Standardizer,
    context_standardizer: Standardizer,
) -> list[Data]:
    output: list[Data] = []
    for record in records:
        patch_cont = patch_standardizer.transform(
            np.stack([patch.shell_descriptor for patch in record.patches], axis=0)
        )
        typed = np.asarray(
            [typed_vocabulary.get(patch.typed_certificate, 0) for patch in record.patches],
            dtype=np.int64,
        )
        parent = np.asarray(
            [parent_vocabulary.get(patch.parent_certificate, 0) for patch in record.patches],
            dtype=np.int64,
        )
        output.append(
            Data(
                patch_cont=torch.from_numpy(patch_cont),
                typed_token=torch.from_numpy(typed),
                parent_token=torch.from_numpy(parent),
                pair_index=torch.from_numpy(record.pair_index),
                pair_relation=torch.from_numpy(record.pair_relation),
                pair_bucket=torch.from_numpy(record.pair_bucket),
                global_context=torch.from_numpy(
                    context_standardizer.transform(record.global_context[None, :])
                ),
                y=torch.tensor([record.y], dtype=torch.float32),
                num_nodes=len(record.patches),
            )
        )
    return output


class _MLPBlock(nn.Module):
    def __init__(self, width: int, hidden: int, output: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(int(width), int(hidden)),
            nn.LayerNorm(int(hidden)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), int(output)),
            nn.ReLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class PatchPathModel(nn.Module):
    def __init__(
        self,
        typed_vocabulary_size: int,
        parent_vocabulary_size: int,
        *,
        patch_hidden: int,
        pair_hidden: int,
        token_width: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.patch_hidden = int(patch_hidden)
        self.pair_hidden = int(pair_hidden)
        self.typed_embedding = nn.Embedding(int(typed_vocabulary_size), int(token_width))
        parent_width = max(int(token_width // 2), 1)
        self.parent_embedding = nn.Embedding(int(parent_vocabulary_size), parent_width)
        self.patch_encoder = _MLPBlock(
            SHELL_WIDTH + int(token_width) + parent_width,
            max(int(patch_hidden), 64),
            int(patch_hidden),
            float(dropout),
        )
        self.global_encoder = _MLPBlock(
            GLOBAL_WIDTH,
            max(int(patch_hidden // 2), 32),
            32,
            float(dropout),
        )
        self.pair_projection = nn.Linear(int(patch_hidden), int(pair_hidden), bias=False)
        self.relation_encoder = _MLPBlock(
            RELATION_WIDTH,
            max(int(pair_hidden), 32),
            int(pair_hidden),
            float(dropout),
        )
        self.distance_gate = nn.Embedding(DISTANCE_BUCKETS, int(pair_hidden))
        self.pair_encoder = _MLPBlock(
            4 * int(pair_hidden),
            max(2 * int(pair_hidden), 64),
            int(pair_hidden),
            float(dropout),
        )
        readout_width = (
            2 * int(patch_hidden)
            + 1
            + DISTANCE_BUCKETS * (2 * int(pair_hidden) + 1)
        )
        self.head = nn.Sequential(
            nn.Linear(readout_width + 32, max(int(patch_hidden) * 2, 96)),
            nn.LayerNorm(max(int(patch_hidden) * 2, 96)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(max(int(patch_hidden) * 2, 96), int(patch_hidden)),
            nn.ReLU(),
            nn.Linear(int(patch_hidden), 1),
        )

    @staticmethod
    def _pool_nodes(value: torch.Tensor, batch: torch.Tensor, n_graphs: int) -> torch.Tensor:
        total = torch.zeros(
            (n_graphs, value.shape[1]), device=value.device, dtype=value.dtype
        )
        total.index_add_(0, batch, value)
        squared = torch.zeros_like(total)
        squared.index_add_(0, batch, value * value)
        counts = torch.bincount(batch, minlength=n_graphs).to(value.dtype).unsqueeze(1)
        return torch.cat([total, squared, torch.log1p(counts)], dim=1)

    @staticmethod
    def _pool_pairs(
        value: torch.Tensor,
        pair_batch: torch.Tensor,
        pair_bucket: torch.Tensor,
        n_graphs: int,
    ) -> torch.Tensor:
        blocks: list[torch.Tensor] = []
        for bucket in range(DISTANCE_BUCKETS):
            mask = pair_bucket == int(bucket)
            current = value[mask]
            current_batch = pair_batch[mask]
            total = torch.zeros(
                (n_graphs, value.shape[1]), device=value.device, dtype=value.dtype
            )
            squared = torch.zeros_like(total)
            counts = torch.zeros((n_graphs, 1), device=value.device, dtype=value.dtype)
            if current.numel():
                total.index_add_(0, current_batch, current)
                squared.index_add_(0, current_batch, current * current)
                counts.index_add_(
                    0,
                    current_batch,
                    torch.ones(
                        (current_batch.shape[0], 1),
                        device=value.device,
                        dtype=value.dtype,
                    ),
                )
            blocks.append(torch.cat([total, squared, torch.log1p(counts)], dim=1))
        return torch.cat(blocks, dim=1)

    def forward(self, data: Data) -> torch.Tensor:
        global_context = data.global_context
        if global_context.ndim == 1:
            global_context = global_context.unsqueeze(0)
        n_graphs = int(global_context.shape[0])
        patch = self.patch_encoder(
            torch.cat(
                [
                    data.patch_cont,
                    self.typed_embedding(data.typed_token),
                    self.parent_embedding(data.parent_token),
                ],
                dim=1,
            )
        )
        unary = self._pool_nodes(patch, data.batch, n_graphs)

        source = data.pair_index[0]
        target = data.pair_index[1]
        projected_left = self.pair_projection(patch[source])
        projected_right = self.pair_projection(patch[target])
        relation = self.relation_encoder(data.pair_relation)
        product = projected_left * projected_right
        gate = 1.0 + torch.tanh(self.distance_gate(data.pair_bucket))
        pair_input = torch.cat(
            [
                projected_left + projected_right,
                torch.abs(projected_left - projected_right),
                product * gate,
                relation,
            ],
            dim=1,
        )
        pair_value = self.pair_encoder(pair_input)
        pair_batch = data.batch[source]
        relation_readout = self._pool_pairs(
            pair_value, pair_batch, data.pair_bucket, n_graphs
        )
        graph_hidden = self.global_encoder(global_context)
        return self.head(torch.cat([unary, relation_readout, graph_hidden], dim=1)).view(-1)


def _make_loader(
    graphs: Sequence[Data], batch_size: int, shuffle: bool, seed: int
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed)) if shuffle else None
    return DataLoader(
        list(graphs),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=0,
    )


def _evaluate_auc(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions.append(torch.sigmoid(model(batch)).cpu().numpy())
            targets.append(batch.y.view(-1).cpu().numpy())
    return float(
        roc_auc_score(
            np.concatenate(targets).astype(np.float64),
            np.concatenate(predictions).astype(np.float64),
        )
    )


def _positive_weight(records: Sequence[GraphRecord]) -> float:
    labels = np.asarray([record.y for record in records], dtype=np.float32)
    positives = int(np.sum(labels > 0.5))
    negatives = int(labels.size - positives)
    if positives <= 0 or negatives <= 0:
        raise ValueError(f"balanced BCE needs both classes: {positives=} {negatives=}")
    return float(negatives / positives)


def _positive_weight_data(graphs: Sequence[Data]) -> float:
    labels = np.asarray([float(graph.y.item()) for graph in graphs], dtype=np.float32)
    positives = int(np.sum(labels > 0.5))
    negatives = int(labels.size - positives)
    if positives <= 0 or negatives <= 0:
        raise ValueError(f"balanced BCE needs both classes: {positives=} {negatives=}")
    return float(negatives / positives)


def _train_phase(
    train_graphs: Sequence[Data],
    eval_graphs: Sequence[Data],
    *,
    typed_vocabulary_size: int,
    parent_vocabulary_size: int,
    config: Mapping[str, Any],
    seed: int,
    select_best: bool,
    epochs: int,
    pos_weight: float,
) -> dict[str, Any]:
    model_config = config["model"]
    device = torch.device(str(model_config.get("device", "cpu")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("model requested CUDA but CUDA is unavailable")
    _seed_everything(seed)
    model = PatchPathModel(
        typed_vocabulary_size,
        parent_vocabulary_size,
        patch_hidden=int(model_config.get("patch_hidden", PATCH_HIDDEN)),
        pair_hidden=int(model_config.get("pair_hidden", PAIR_HIDDEN)),
        token_width=int(model_config.get("token_width", 32)),
        dropout=float(model_config.get("dropout", 0.05)),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(model_config.get("learning_rate", 1.0e-3)),
        weight_decay=float(model_config.get("weight_decay", 1.0e-5)),
    )
    loader = _make_loader(
        train_graphs,
        int(model_config.get("batch_size", 64)),
        True,
        seed + 91011,
    )
    eval_loader = _make_loader(
        eval_graphs,
        int(model_config.get("batch_size", 64)),
        False,
        seed + 91012,
    )
    patience = int(model_config.get("patience", 10))
    evaluate_every = int(model_config.get("evaluate_every", 1))
    weight = torch.as_tensor(float(pos_weight), dtype=torch.float32, device=device)
    best_auc = -float("inf")
    best_epoch = 1
    best_state: dict[str, Any] | None = None
    trace: list[dict[str, float | int]] = []
    stale = 0
    losses: list[float] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch in loader:
            batch = batch.to(device)
            target = batch.y.view(-1)
            loss = F.binary_cross_entropy_with_logits(
                model(batch), target, pos_weight=weight
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(target)
            seen += len(target)
        epoch_loss = total_loss / max(seen, 1)
        losses.append(float(epoch_loss))
        current_auc = None
        if select_best and (
            epoch == 1 or epoch % max(evaluate_every, 1) == 0 or epoch == int(epochs)
        ):
            current_auc = _evaluate_auc(model, eval_loader, device)
            trace.append({"epoch": int(epoch), "train_bce": float(epoch_loss), "auc": float(current_auc)})
            if current_auc > best_auc:
                best_auc = float(current_auc)
                best_epoch = int(epoch)
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            if (
                epoch == 1
                or epoch % max(1, int(epochs) // 10) == 0
                or current_auc == best_auc
            ):
                print(
                    f"molhiv patch-path phase=select epoch={epoch:03d}/{epochs} "
                    f"bce={epoch_loss:.6f} valid_auc={current_auc:.6f}",
                    flush=True,
                )
        elif not select_best and (
            epoch == 1 or epoch == int(epochs) or epoch % max(1, int(epochs) // 10) == 0
        ):
            print(
                f"molhiv patch-path phase=refit epoch={epoch:03d}/{epochs} "
                f"bce={epoch_loss:.6f}",
                flush=True,
            )
        if select_best and stale >= patience:
            print(
                f"molhiv patch-path early_stop epoch={epoch} best_epoch={best_epoch}",
                flush=True,
            )
            break
    if select_best and best_state is not None:
        model.load_state_dict(best_state)
    final_auc = _evaluate_auc(model, eval_loader, device)
    return {
        "model": model,
        "auc": float(final_auc),
        "best_auc": None if not select_best else float(best_auc),
        "selected_epoch": int(best_epoch if select_best else epochs),
        "epochs_run": int(len(losses)),
        "trace": trace,
        "losses": losses,
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    }


def _phase_data(
    fit_records: Sequence[GraphRecord],
    other_records: Sequence[GraphRecord],
    *,
    config: Mapping[str, Any],
) -> tuple[list[Data], list[Data], dict[str, Any]]:
    representation = config["representation"]
    typed_vocabulary = _fit_vocabulary(
        fit_records,
        "typed_certificate",
        int(representation.get("max_typed_tokens", 32768)),
        int(representation.get("minimum_typed_frequency", 1)),
    )
    parent_vocabulary = _fit_vocabulary(
        fit_records,
        "parent_certificate",
        int(representation.get("max_parent_tokens", 4096)),
        int(representation.get("minimum_parent_frequency", 1)),
    )
    patch_standardizer = Standardizer.fit(_patch_matrix(fit_records))
    context_standardizer = Standardizer.fit(_context_matrix(fit_records))
    encoded_fit = _encode_records(
        fit_records,
        typed_vocabulary,
        parent_vocabulary,
        patch_standardizer,
        context_standardizer,
    )
    encoded_other = _encode_records(
        other_records,
        typed_vocabulary,
        parent_vocabulary,
        patch_standardizer,
        context_standardizer,
    )
    audit = {
        "typed_vocabulary": {
            "fit": _vocabulary_stats(fit_records, typed_vocabulary, "typed_certificate"),
            "other": _vocabulary_stats(other_records, typed_vocabulary, "typed_certificate"),
        },
        "parent_vocabulary": {
            "fit": _vocabulary_stats(fit_records, parent_vocabulary, "parent_certificate"),
            "other": _vocabulary_stats(other_records, parent_vocabulary, "parent_certificate"),
        },
        "typed_vocabulary_size_with_oov": int(len(typed_vocabulary) + 1),
        "parent_vocabulary_size_with_oov": int(len(parent_vocabulary) + 1),
        "patch_standardizer_fit_graphs": int(len(fit_records)),
        "context_standardizer_fit_graphs": int(len(fit_records)),
    }
    return encoded_fit, encoded_other, audit


def _render_markdown(result: Mapping[str, Any]) -> str:
    evaluation = result["evaluation"]
    lines = [
        f"# {result['protocol_id']}",
        "",
        "Exact rooted OGB-typed patch tokens with shortest-path-conditioned pair pooling; one MLP, no message passing.",
        "",
        f"- split: `{result['data']['split']}`; sizes `{result['data']['sizes']}`",
        f"- patch shell descriptor: `{result['representation']['shell_width']}D`; relation descriptor: `{result['representation']['relation_width']}D`",
        f"- readout: `{result['representation']['readout']}`",
        f"- message passing: `{result['representation']['message_passing']}`; attention: `{result['representation']['attention']}`",
        "",
        "| head | valid ROC-AUC | test ROC-AUC after train+valid refit | selected epoch |",
        "|---|---:|---:|---:|",
        f"| `single_mlp` | {evaluation['valid']['auc']:.6f} | {evaluation['test_after_train_valid_refit']['auc']:.6f} | {evaluation['valid']['selected_epoch']} |",
        "",
        f"- train-only valid typed token coverage: `{result['vocabulary_audit']['valid']['typed_vocabulary']['other']['known_occurrence_fraction']:.4f}`",
        f"- train+valid test typed token coverage: `{result['vocabulary_audit']['test_after_train_valid_refit']['typed_vocabulary']['other']['known_occurrence_fraction']:.4f}`",
        f"- trainable parameters: `{evaluation['parameters']}`",
        f"- runtime: `{result['runtime']['seconds']:.1f}s`",
        "",
    ]
    return "\n".join(lines)


def run(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_config = config["data"]
    model_config = config["model"]
    started = time.perf_counter()
    seed = int(config.get("seed", 0))
    torch.set_num_threads(max(int(config.get("runtime", {}).get("torch_threads", 4)), 1))

    bundle = load_molhiv(
        root=_resolve(data_config["root"]),
        with_features=True,
    )
    if bundle.node_feats is None or bundle.edge_feats is None:
        raise RuntimeError("MolHIV OGB node/edge features were not loaded")
    split_indices = {
        name: np.asarray(bundle.split[name], dtype=np.int64)
        for name in ("train", "valid", "test")
    }
    split_meta = {
        name: {
            "n_graphs": int(values.size),
            "positive_graphs": int(bundle.y[values].sum()),
        }
        for name, values in split_indices.items()
    }

    certificate_cache: dict[bytes, bytes] = {}
    feature_metadata: dict[str, Any] = {}

    # Phase 1: build only train and valid records, then release the raw record
    # objects before fitting.  This keeps the full MolHIV run below local RAM.
    train_records, feature_metadata["train"] = _extract_split(
        bundle, split_indices["train"], "train", certificate_cache
    )
    valid_records, feature_metadata["valid"] = _extract_split(
        bundle, split_indices["valid"], "valid", certificate_cache
    )
    valid_train_data, valid_eval_data, valid_audit = _phase_data(
        train_records, valid_records, config=config
    )
    valid_pos_weight = _positive_weight(train_records)
    del train_records, valid_records
    gc.collect()

    device = torch.device(str(model_config.get("device", "cpu")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("model requested CUDA but CUDA is unavailable")
    valid_phase = _train_phase(
        valid_train_data,
        valid_eval_data,
        typed_vocabulary_size=int(valid_audit["typed_vocabulary_size_with_oov"]),
        parent_vocabulary_size=int(valid_audit["parent_vocabulary_size_with_oov"]),
        config=config,
        seed=seed,
        select_best=True,
        epochs=int(model_config.get("epochs", 50)),
        pos_weight=valid_pos_weight,
    )
    selected_epoch = int(valid_phase["selected_epoch"])
    del valid_train_data, valid_eval_data, valid_phase["model"]
    gc.collect()

    # Phase 2: refit all label-free transforms on official train+valid and
    # evaluate the fresh model once on official test.
    refit_indices = np.concatenate(
        [split_indices["train"], split_indices["valid"]]
    ).astype(np.int64, copy=False)
    refit_records, feature_metadata["train_valid_refit"] = _extract_split(
        bundle, refit_indices, "train+valid", certificate_cache
    )
    test_records, feature_metadata["test"] = _extract_split(
        bundle, split_indices["test"], "test", certificate_cache
    )
    refit_train_data, refit_test_data, test_audit = _phase_data(
        refit_records, test_records, config=config
    )
    refit_pos_weight = _positive_weight(refit_records)
    del refit_records, test_records
    gc.collect()
    refit_phase = _train_phase(
        refit_train_data,
        refit_test_data,
        typed_vocabulary_size=int(test_audit["typed_vocabulary_size_with_oov"]),
        parent_vocabulary_size=int(test_audit["parent_vocabulary_size_with_oov"]),
        config=config,
        seed=seed,
        select_best=False,
        epochs=selected_epoch,
        pos_weight=refit_pos_weight,
    )

    result = {
        "protocol_id": str(config["protocol_id"]),
        "status": "completed",
        "seed": seed,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "data": {
            "root": str(_resolve(data_config["root"])),
            "split": "OGB ogbg-molhiv official scaffold train/valid/test",
            "sizes": {name: int(values.size) for name, values in split_indices.items()},
            "positive": {
                name: int(split_meta[name]["positive_graphs"])
                for name in split_indices
            },
            "positive_rate": {
                name: float(split_meta[name]["positive_graphs"] / max(int(values.size), 1))
                for name, values in split_indices.items()
            },
            "test_labels_used_for_selection": False,
        },
        "representation": {
            "radius": PATCH_RADIUS,
            "centres": "every atom",
            "exact_patch": "rooted colored-incidence certificate over full OGB atom/bond feature tuples",
            "atom_feature_dims": list(ATOM_FEATURE_DIMS),
            "bond_feature_dims": list(BOND_FEATURE_DIMS),
            "shell_width": SHELL_WIDTH,
            "relation_width": RELATION_WIDTH,
            "global_width": GLOBAL_WIDTH,
            "distance_buckets": ["1", "2", "3", "4", "5+", "disconnected"],
            "relation_definition": "distance bucket, shortest-path bond-feature composition averaged over all shortest paths, path count, patch overlap/boundary overlap, and adjacent bond feature",
            "readout": "unary sum and sum-of-squares plus distance-conditioned pair sum and sum-of-squares with log pair mass",
            "message_passing": False,
            "attention": False,
            "label_free": True,
        },
        "feature_build": {
            **feature_metadata,
            "certificate_cache_entries": int(len(certificate_cache)),
        },
        "vocabulary_audit": {
            "valid": valid_audit,
            "test_after_train_valid_refit": test_audit,
        },
        "training": {
            **dict(model_config),
            "loss": "balanced binary cross entropy with logits",
            "metric": "ROC-AUC",
            "device": str(device),
            "selected_epoch": selected_epoch,
            "one_predictive_head": True,
        },
        "evaluation": {
            "valid": {
                "auc": float(valid_phase["auc"]),
                "best_auc": float(valid_phase["best_auc"]),
                "selected_epoch": selected_epoch,
                "epochs_run": int(valid_phase["epochs_run"]),
                "trace": valid_phase["trace"],
            },
            "test_after_train_valid_refit": {
                "auc": float(refit_phase["auc"]),
                "epochs_run": int(refit_phase["epochs_run"]),
            },
            "parameters": int(valid_phase["parameters"]),
        },
        "runtime": {
            "seconds": float(time.perf_counter() - started),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": importlib.metadata.version("numpy"),
            "torch": importlib.metadata.version("torch"),
            "torch_geometric": importlib.metadata.version("torch-geometric"),
            "pynauty": importlib.metadata.version("pynauty"),
            "script_sha256": _sha256(Path(__file__).resolve()),
        },
    }
    output_json = _resolve(config["output"]["json"])
    output_markdown = _resolve(config["output"]["markdown"])
    _write_json_atomic(output_json, result)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(_render_markdown(result), encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    config_path = _resolve(args.config)
    result = run(config_path)
    print(
        json.dumps(
            {
                "valid_auc": result["evaluation"]["valid"]["auc"],
                "test_auc": result["evaluation"]["test_after_train_valid_refit"]["auc"],
                "selected_epoch": result["evaluation"]["valid"]["selected_epoch"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
