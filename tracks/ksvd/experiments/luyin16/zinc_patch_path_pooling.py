"""ZINC patch--path pooling with one explicit, non-message-passing MLP.

This is the next representation/readout candidate after the dense canonical
patch experiment.  It keeps an exact rooted typed patch as a train-only
categorical token, adds a small continuous shell descriptor for unseen/OOV
patches, and forms a symmetric relation object for every unordered pair of
centres.  Pair interactions are conditioned on the shortest-path relation
*before* graph pooling.  Pair outputs are then pooled separately by distance
bucket using first/second moments and pair mass.

There is no centre-to-centre message passing: a patch state is computed once,
and each pair is evaluated independently before the invariant readout.  The
only predictive head is the final MLP, which is trained end-to-end with the
patch encoder and pair function.

The official validation phase fits token vocabularies and continuous
standardisation on official train only.  The test phase refits those
label-free transforms on train+validation after the validation-selected epoch
has been frozen.  The official test labels never affect representation or
epoch selection.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter, deque
from dataclasses import dataclass
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
from sklearn.metrics import mean_absolute_error
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from tracks.ksvd.experiments.luyin16.zinc_exact_patch_relation import (
    _canonical_typed_patch,
    _patch_cache_key,
)
from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import (
    REPO_ROOT,
    _data_to_graph,
    _load_zinc,
    _resolve,
    global_feature_views,
    source_audit,
)


DEFAULT_CONFIG = REPO_ROOT / "tracks/ksvd/configs/luyin16/zinc_patch_path_pooling.yaml"

ATOM_CATEGORIES = 28
BOND_CATEGORIES = 4
PATCH_RADIUS = 2
DISTANCE_BUCKETS = 5  # 1, 2, 3, 4, 5+
SHELL_PAIRS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))

# Shell-wise atom proportions (3 x 28), shell-pair bond proportions (6 x 4),
# root/incident chemistry (28 + 4), and a few size/cycle/degree scalars.
SHELL_WIDTH = 3 * ATOM_CATEGORIES + len(SHELL_PAIRS) * BOND_CATEGORIES + 28 + 4 + 6
RELATION_WIDTH = (
    DISTANCE_BUCKETS  # distance bucket one-hot
    + 1  # log shortest-path distance
    + 5  # patch overlap and size relation
    + 3  # boundary overlap and centre containment
    + BOND_CATEGORIES  # bond composition averaged over shortest paths
    + 1  # log number of shortest paths
    + BOND_CATEGORIES  # adjacent bond type, zero for non-adjacent pairs
)
PATCH_HIDDEN = 64
PAIR_HIDDEN = 32
GLOBAL_WIDTH = 62  # global_feature_views(...)["global_all"]


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


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _ego_distances(graph: Any, center: int, radius: int) -> dict[int, int]:
    distances = {int(center): 0}
    queue: deque[int] = deque([int(center)])
    while queue:
        node = queue.popleft()
        if distances[node] >= int(radius):
            continue
        for neighbor in sorted(graph.neighbors(node)):
            if neighbor not in distances:
                distances[int(neighbor)] = distances[node] + 1
                queue.append(int(neighbor))
    return distances


def _shortest_path_summary(
    graph: Any,
    source: int,
    edge_types: Mapping[tuple[int, int], int],
) -> dict[int, tuple[int, float, np.ndarray]]:
    """Return distance, path count, and mean bond composition to every node.

    The bond composition is averaged over all shortest paths.  This avoids
    choosing an arbitrary node-ID-dependent path when multiple shortest paths
    exist, and keeps the relation permutation invariant.
    """
    source = int(source)
    distances = {source: 0}
    path_counts: dict[int, float] = {source: 1.0}
    bond_sums: dict[int, np.ndarray] = {
        source: np.zeros(BOND_CATEGORIES, dtype=np.float64)
    }
    queue: deque[int] = deque([source])
    while queue:
        node = queue.popleft()
        for neighbor in sorted(graph.neighbors(node)):
            neighbor = int(neighbor)
            bond = int(edge_types[graph.edge_key(node, neighbor)])
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                path_counts[neighbor] = 0.0
                bond_sums[neighbor] = np.zeros(BOND_CATEGORIES, dtype=np.float64)
                queue.append(neighbor)
            if distances[neighbor] != distances[node] + 1:
                continue
            path_counts[neighbor] += path_counts[node]
            bond_sums[neighbor] += bond_sums[node]
            bond_sums[neighbor][bond] += path_counts[node]
    output: dict[int, tuple[int, float, np.ndarray]] = {}
    for node, distance in distances.items():
        count = max(float(path_counts[node]), 1.0)
        output[int(node)] = (
            int(distance),
            float(path_counts[node]),
            (bond_sums[node] / count).astype(np.float32),
        )
    return output


def _shell_descriptor(
    graph: Any,
    center: int,
    node_types: np.ndarray,
    edge_types: Mapping[tuple[int, int], int],
    distances: Mapping[int, int],
) -> tuple[np.ndarray, frozenset[int], frozenset[int]]:
    nodes = frozenset(int(node) for node in distances)
    induced = graph.induced(set(nodes))
    n_nodes = len(nodes)
    n_edges = induced.num_edges()
    atom_shell = np.zeros((PATCH_RADIUS + 1, ATOM_CATEGORIES), dtype=np.float32)
    for node in nodes:
        atom = int(node_types[node])
        shell = int(distances[node])
        if atom < 0 or atom >= ATOM_CATEGORIES:
            raise ValueError(f"atom category {atom} outside schema")
        atom_shell[shell, atom] += 1.0
    atom_shell /= max(float(n_nodes), 1.0)

    bond_shell = np.zeros((len(SHELL_PAIRS), BOND_CATEGORIES), dtype=np.float32)
    shell_pair_index = {pair: index for index, pair in enumerate(SHELL_PAIRS)}
    for left, right in induced.edges():
        pair = tuple(sorted((int(distances[left]), int(distances[right]))))
        index = shell_pair_index.get(pair)
        if index is None:
            raise RuntimeError(f"unexpected shell pair {pair}")
        bond = int(edge_types[graph.edge_key(int(left), int(right))])
        if bond < 0 or bond >= BOND_CATEGORIES:
            raise ValueError(f"bond category {bond} outside schema")
        bond_shell[index, bond] += 1.0
    bond_shell /= max(float(n_edges), 1.0)

    root_atom = np.zeros(ATOM_CATEGORIES, dtype=np.float32)
    root_atom[int(node_types[int(center)])] = 1.0
    incident_bonds = np.zeros(BOND_CATEGORIES, dtype=np.float32)
    for neighbor in graph.neighbors(int(center)):
        bond = int(edge_types[graph.edge_key(int(center), int(neighbor))])
        incident_bonds[bond] += 1.0
    incident_bonds /= max(float(len(graph.neighbors(int(center)))), 1.0)

    cycle_rank = max(int(n_edges) - int(n_nodes) + 1, 0)
    degrees = np.asarray([len(graph.neighbors(node)) for node in nodes], dtype=np.float32)
    scalars = np.asarray(
        [
            np.log1p(float(n_nodes)),
            np.log1p(float(n_edges)),
            float(len(nodes) and sum(int(distance) == PATCH_RADIUS for distance in distances.values()))
            / max(float(n_nodes), 1.0),
            float(cycle_rank) / max(float(n_nodes), 1.0),
            float(len(graph.neighbors(int(center)))) / 4.0,
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


def _typed_certificate(
    graph: Any,
    center: int,
    node_types: np.ndarray,
    edge_types: Mapping[tuple[int, int], int],
    radius: int,
    cache: dict[bytes, bytes],
) -> bytes:
    key = _patch_cache_key(graph, int(center), node_types, edge_types, int(radius))
    certificate = cache.get(key)
    if certificate is None:
        certificate = _canonical_typed_patch(
            graph, int(center), node_types, edge_types, radius=int(radius)
        )[2]
        cache[key] = certificate
    return certificate


def _pair_relation(
    left: PatchRecord,
    right: PatchRecord,
    path_summary: tuple[int, float, np.ndarray],
    adjacent_bond: int | None,
) -> tuple[np.ndarray, int]:
    distance, path_count, path_bond_mean = path_summary
    bucket = min(max(int(distance), 1), DISTANCE_BUCKETS) - 1
    distance_one_hot = np.zeros(DISTANCE_BUCKETS, dtype=np.float32)
    distance_one_hot[bucket] = 1.0

    intersection = len(left.nodes & right.nodes)
    union = len(left.nodes | right.nodes)
    left_size = len(left.nodes)
    right_size = len(right.nodes)
    overlap = np.asarray(
        [
            float(intersection) / max(float(PATCH_RADIUS * PATCH_RADIUS + 10), 1.0),
            float(intersection) / max(float(union), 1.0),
            float(intersection) / max(float(min(left_size, right_size)), 1.0),
            float(intersection) / max(float(max(left_size, right_size)), 1.0),
            float(abs(left_size - right_size)) / max(float(PATCH_RADIUS * PATCH_RADIUS + 10), 1.0),
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
    adjacent = np.zeros(BOND_CATEGORIES, dtype=np.float32)
    if adjacent_bond is not None:
        adjacent[int(adjacent_bond)] = 1.0
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
        raise RuntimeError(f"relation width changed: {relation.shape}; expected {RELATION_WIDTH}")
    return relation, bucket


def _graph_record(
    data: Any,
    global_context: np.ndarray,
    certificate_cache: dict[bytes, bytes],
) -> GraphRecord:
    graph, node_types, edge_types = _data_to_graph(data)
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
            path_summary = shortest_paths[int(left_center)][right_center]
            distance = int(path_summary[0])
            adjacent_bond = None
            if distance == 1:
                adjacent_bond = int(edge_types[graph.edge_key(int(left_center), right_center)])
            relation, bucket = _pair_relation(
                patches[left_index], patches[right_index], path_summary, adjacent_bond
            )
            pair_sources.append(int(left_index))
            pair_targets.append(int(right_index))
            pair_relations.append(relation)
            pair_buckets.append(int(bucket))

    pair_index = np.asarray([pair_sources, pair_targets], dtype=np.int64)
    pair_relation = np.stack(pair_relations, axis=0).astype(np.float32, copy=False)
    pair_bucket = np.asarray(pair_buckets, dtype=np.int64)
    if pair_relation.shape[1] != RELATION_WIDTH:
        raise RuntimeError(f"pair relation matrix width changed: {pair_relation.shape}")
    return GraphRecord(
        patches=tuple(patches),
        pair_index=pair_index,
        pair_relation=pair_relation,
        pair_bucket=pair_bucket,
        global_context=np.asarray(global_context, dtype=np.float32),
        y=float(data.y.view(-1)[0]),
    )


def _extract_split(
    dataset: Any,
    split: str,
    certificate_cache: dict[bytes, bytes],
) -> tuple[list[GraphRecord], dict[str, Any]]:
    started = time.perf_counter()
    contexts = global_feature_views(dataset)["global_all"]
    records: list[GraphRecord] = []
    for index, data in enumerate(dataset):
        records.append(_graph_record(data, contexts[index], certificate_cache))
        if index and index % 500 == 0:
            print(
                f"patch-path features {split}: {index}/{len(dataset)} "
                f"certificate_cache={len(certificate_cache)}",
                flush=True,
            )
    metadata = {
        "n_graphs": int(len(records)),
        "mean_centres": float(np.mean([len(row.patches) for row in records])),
        "mean_pairs": float(np.mean([row.pair_relation.shape[0] for row in records])),
        "mean_patch_nodes": float(
            np.mean([len(patch.nodes) for row in records for patch in row.patches])
        ),
        "max_patch_nodes": int(
            max((len(patch.nodes) for row in records for patch in row.patches), default=0)
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
    # ID zero is a learned OOV token; known tokens start at one.
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


class _FactorizedEmbedding(nn.Module):
    """Exact-token lookup with a low-rank parameterization.

    The lookup still has one row per exact rooted patch, so no patch identity
    is hashed or merged.  Only the embedding table is factorized as
    ``V x rank`` followed by a shared linear map to the requested output
    width.  This lets us keep a 32D token representation under the CIN-sized
    parameter budget.
    """

    def __init__(self, vocabulary_size: int, output_width: int, rank: int) -> None:
        super().__init__()
        if int(rank) < 1 or int(rank) > int(output_width):
            raise ValueError("embedding rank must be in [1, output_width]")
        self.rank = int(rank)
        self.output_width = int(output_width)
        self.embedding = nn.Embedding(int(vocabulary_size), int(rank))
        self.projection = nn.Linear(int(rank), int(output_width), bias=False)

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        return self.projection(self.embedding(token))


def _make_embedding(
    vocabulary_size: int,
    output_width: int,
    *,
    mode: str,
    rank: int,
) -> nn.Module:
    if mode == "full":
        return nn.Embedding(int(vocabulary_size), int(output_width))
    if mode == "factorized":
        return _FactorizedEmbedding(int(vocabulary_size), int(output_width), int(rank))
    raise ValueError(f"unknown embedding_mode={mode!r}; expected full or factorized")


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
        embedding_mode: str = "full",
        embedding_rank: int = 16,
    ) -> None:
        super().__init__()
        self.patch_hidden = int(patch_hidden)
        self.pair_hidden = int(pair_hidden)
        self.embedding_mode = str(embedding_mode)
        self.embedding_rank = int(embedding_rank)
        self.typed_embedding = _make_embedding(
            int(typed_vocabulary_size),
            int(token_width),
            mode=self.embedding_mode,
            rank=self.embedding_rank,
        )
        parent_width = max(int(token_width // 2), 1)
        parent_rank = min(self.embedding_rank, parent_width)
        self.parent_embedding = _make_embedding(
            int(parent_vocabulary_size),
            parent_width,
            mode=self.embedding_mode,
            rank=parent_rank,
        )
        self.patch_encoder = _MLPBlock(
            SHELL_WIDTH + int(token_width) + parent_width,
            max(int(patch_hidden), 64),
            int(patch_hidden),
            float(dropout),
        )
        self.global_encoder = _MLPBlock(GLOBAL_WIDTH, max(int(patch_hidden // 2), 32), 32, float(dropout))
        self.pair_projection = nn.Linear(int(patch_hidden), int(pair_hidden), bias=False)
        self.relation_encoder = _MLPBlock(
            RELATION_WIDTH, max(int(pair_hidden), 32), int(pair_hidden), float(dropout)
        )
        self.distance_gate = nn.Embedding(DISTANCE_BUCKETS, int(pair_hidden))
        self.pair_encoder = _MLPBlock(
            4 * int(pair_hidden),
            max(2 * int(pair_hidden), 64),
            int(pair_hidden),
            float(dropout),
        )
        # Unary: sum, sum-of-squares, log mass.  Each distance bucket has the
        # same first/second moment plus log pair mass.  The final module is the
        # sole graph-level prediction head.
        readout_width = 2 * int(patch_hidden) + 1 + DISTANCE_BUCKETS * (2 * int(pair_hidden) + 1)
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
        total = torch.zeros((n_graphs, value.shape[1]), device=value.device, dtype=value.dtype)
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
            total = torch.zeros((n_graphs, value.shape[1]), device=value.device, dtype=value.dtype)
            squared = torch.zeros_like(total)
            counts = torch.zeros((n_graphs, 1), device=value.device, dtype=value.dtype)
            if current.numel():
                total.index_add_(0, current_batch, current)
                squared.index_add_(0, current_batch, current * current)
                counts.index_add_(
                    0,
                    current_batch,
                    torch.ones((current_batch.shape[0], 1), device=value.device, dtype=value.dtype),
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


def _make_loader(graphs: Sequence[Data], batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed)) if shuffle else None
    return DataLoader(
        list(graphs),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=0,
    )


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions.append(model(batch).cpu().numpy())
            targets.append(batch.y.view(-1).cpu().numpy())
    return float(
        mean_absolute_error(
            np.concatenate(targets).astype(np.float64),
            np.concatenate(predictions).astype(np.float64),
        )
    )


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
        embedding_mode=str(model_config.get("embedding_mode", "full")),
        embedding_rank=int(model_config.get("embedding_rank", 16)),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(model_config.get("learning_rate", 1.0e-3)),
        weight_decay=float(model_config.get("weight_decay", 1.0e-5)),
    )
    loader = _make_loader(
        train_graphs,
        int(model_config.get("batch_size", 128)),
        True,
        seed + 91011,
    )
    eval_loader = _make_loader(
        eval_graphs,
        int(model_config.get("batch_size", 128)),
        False,
        seed + 91012,
    )
    patience = int(model_config.get("patience", 10))
    best_mae = float("inf")
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
            prediction = model(batch)
            target = batch.y.view(-1)
            loss = F.l1_loss(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(target)
            seen += len(target)
        epoch_loss = total_loss / max(seen, 1)
        losses.append(float(epoch_loss))
        current_mae = None
        if select_best:
            current_mae = _evaluate(model, eval_loader, device)
            trace.append({"epoch": int(epoch), "mae": float(current_mae)})
            if current_mae < best_mae:
                best_mae = float(current_mae)
                best_epoch = int(epoch)
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
        if epoch == 1 or epoch == int(epochs) or epoch % max(1, int(epochs) // 10) == 0:
            suffix = "" if current_mae is None else f" valid_mae={current_mae:.6f}"
            print(
                f"patch-path phase={'select' if select_best else 'refit'} "
                f"epoch={epoch:03d}/{epochs} l1={epoch_loss:.6f}{suffix}",
                flush=True,
            )
        if select_best and stale >= patience:
            print(
                f"patch-path early_stop epoch={epoch} best_epoch={best_epoch}", flush=True
            )
            break
    if select_best and best_state is not None:
        model.load_state_dict(best_state)
    final_mae = _evaluate(model, eval_loader, device)
    return {
        "model": model,
        "mae": float(final_mae),
        "best_mae": None if not select_best else float(best_mae),
        "selected_epoch": int(best_epoch if select_best else epochs),
        "epochs_run": int(len(losses)),
        "trace": trace,
        "losses": losses,
        "device": str(device),
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
        int(representation.get("max_typed_tokens", 8192)),
        int(representation.get("minimum_typed_frequency", 1)),
    )
    parent_vocabulary = _fit_vocabulary(
        fit_records,
        "parent_certificate",
        int(representation.get("max_parent_tokens", 2048)),
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
        "Exact rooted typed patch tokens with shortest-path-conditioned pair pooling; one MLP, no message passing.",
        "",
        f"- split: `{result['data']['split']}`; sizes `{result['data']['sizes']}`",
        f"- patch shell descriptor: `{result['representation']['shell_width']}D`; relation descriptor: `{result['representation']['relation_width']}D`",
        f"- readout: `{result['representation']['readout']}`",
        f"- message passing: `{result['representation']['message_passing']}`; attention: `{result['representation']['attention']}`",
        "",
        "| head | valid MAE | test MAE after train+valid refit | selected epoch |",
        "|---|---:|---:|---:|",
        f"| `single_mlp` | {evaluation['valid']['mae']:.6f} | {evaluation['test_after_train_valid_refit']['mae']:.6f} | {evaluation['valid']['selected_epoch']} |",
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
    data_root = _resolve(config["data"]["root"])
    result_json = _resolve(config["output"]["json"])
    result_markdown = _resolve(config["output"]["markdown"])
    seed = int(config.get("seed", 0))
    started = time.perf_counter()
    thread_count = int(config.get("runtime", {}).get("torch_threads", 4))
    torch.set_num_threads(max(thread_count, 1))

    datasets = tuple(_load_zinc(data_root, split) for split in ("train", "val", "test"))
    labels = tuple(
        np.asarray([float(data.y.view(-1)[0]) for data in dataset], dtype=np.float32)
        for dataset in datasets
    )
    certificate_cache: dict[bytes, bytes] = {}
    records: list[list[GraphRecord]] = []
    feature_metadata: dict[str, Any] = {}
    for split, dataset in zip(("train", "valid", "test"), datasets, strict=True):
        split_records, metadata = _extract_split(dataset, split, certificate_cache)
        records.append(split_records)
        feature_metadata[split] = metadata
    feature_metadata["certificate_cache_entries"] = int(len(certificate_cache))

    train_records, valid_records, test_records = records
    model_config = config["model"]
    device = torch.device(str(model_config.get("device", "cpu")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("model requested CUDA but CUDA is unavailable")

    valid_train_data, valid_eval_data, valid_audit = _phase_data(
        train_records, valid_records, config=config
    )
    valid_phase = _train_phase(
        valid_train_data,
        valid_eval_data,
        typed_vocabulary_size=int(valid_audit["typed_vocabulary_size_with_oov"]),
        parent_vocabulary_size=int(valid_audit["parent_vocabulary_size_with_oov"]),
        config=config,
        seed=seed,
        select_best=True,
        epochs=int(model_config.get("epochs", 50)),
    )
    selected_epoch = int(valid_phase["selected_epoch"])

    refit_train_records = list(train_records) + list(valid_records)
    refit_train_data, refit_test_data, test_audit = _phase_data(
        refit_train_records, test_records, config=config
    )
    refit_phase = _train_phase(
        refit_train_data,
        refit_test_data,
        typed_vocabulary_size=int(test_audit["typed_vocabulary_size_with_oov"]),
        parent_vocabulary_size=int(test_audit["parent_vocabulary_size_with_oov"]),
        config=config,
        seed=seed,
        select_best=False,
        epochs=selected_epoch,
    )

    result = {
        "protocol_id": str(config["protocol_id"]),
        "status": "completed",
        "seed": seed,
        "data": {
            "root": str(data_root),
            "split": "PyG ZINC subset=True official train/val/test",
            "sizes": {
                name: len(dataset)
                for name, dataset in zip(("train", "valid", "test"), datasets, strict=True)
            },
            "source": source_audit(data_root),
            "valid_vocab_scope": "official train only",
            "test_vocab_scope": "official train+valid only",
            "test_labels_used_for_selection": False,
        },
        "representation": {
            "radius": PATCH_RADIUS,
            "centres": "every atom",
            "exact_patch": "rooted colored-incidence typed certificate; radius-2 token with radius-1 parent token",
            "shell_width": SHELL_WIDTH,
            "relation_width": RELATION_WIDTH,
            "distance_buckets": ["1", "2", "3", "4", "5+"],
            "relation_definition": "distance bucket, shortest-path bond composition averaged over all shortest paths, path count, patch overlap/boundary overlap, and adjacent bond type",
            "readout": "unary sum and sum-of-squares plus distance-conditioned pair sum and sum-of-squares with log pair mass",
            "message_passing": False,
            "attention": False,
            "label_free": True,
        },
        "feature_build": feature_metadata,
        "vocabulary_audit": {
            "valid": valid_audit,
            "test_after_train_valid_refit": test_audit,
        },
        "training": {
            **dict(model_config),
            "loss": "L1 / mean absolute error",
            "device": str(device),
            "selected_epoch": selected_epoch,
            "one_predictive_head": True,
        },
        "evaluation": {
            "valid": {
                "mae": float(valid_phase["mae"]),
                "best_mae": float(valid_phase["best_mae"]),
                "selected_epoch": selected_epoch,
                "epochs_run": int(valid_phase["epochs_run"]),
                "trace": valid_phase["trace"],
            },
            "test_after_train_valid_refit": {
                "mae": float(refit_phase["mae"]),
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
    _write_json_atomic(result_json, result)
    result_markdown.parent.mkdir(parents=True, exist_ok=True)
    result_markdown.write_text(_render_markdown(result), encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    result = run(_resolve(args.config))
    print(
        json.dumps(
            {
                "valid_mae": result["evaluation"]["valid"]["mae"],
                "test_mae": result["evaluation"]["test_after_train_valid_refit"]["mae"],
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
