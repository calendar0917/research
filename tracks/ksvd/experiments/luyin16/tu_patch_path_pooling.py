"""TU graph-classification port of the compact recurrent pair--centre backbone.

This is a new benchmark track, not a molecular experiment.  It ports the same
architectural family used by the ZINC/MolHIV channel-necessity work:

    rooted radius-2 patch per node  ->  patch encoder
    all unordered patch-centre pairs ->  distance/overlap/adjacency relation
    T=2 weight-tied recurrent pair--centre refresh
    graph-level readout (patch moments + pair moments + global context)
    local-token channel: ``typed_lookup`` / ``null`` / ``constant``

so the same necessity question can be asked on TU datasets.

Strict split protocol (pre-registered, see
``notes/tu_channel_necessity_preregistration.md``):

* outer stratified k-fold; each graph is the *test* set in exactly one fold;
* the remaining folds are split once into train / valid (stratified, fixed
  seed); valid is the only selection signal;
* the test fold is evaluated once per fold after the model is frozen
  (best-valid checkpoint and fixed equal-weight Top-5 soup);
* identical splits are used for every condition and baseline;
* split indices are written to ``splits.json`` with a SHA-256 fingerprint
  before any training.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

# ---------------------------------------------------------------------------
# layout / constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[4]
TU_ROOT = REPO_ROOT / "data/TUD"

PATCH_RADIUS = 2
DISTANCE_BUCKETS = 4
DEGREE_BINS = 16

SPLIT_SEED = 20260918
VALID_FRAC = 0.2

TOKEN_WIDTH = 16
PATCH_HIDDEN = 64
PAIR_HIDDEN = 16
CENTER_HIDDEN = 60
DROPOUT = 0.05
RECURRENCE_ROUNDS = 2

MAX_EPOCHS = 150
PATIENCE = 30
BATCH_SIZE = 64
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 5.0
SOUP_K = 5

MAX_VOCAB = 65536
MIN_VOCAB_FREQ = 1

TRAINABLE_REPRESENTATIONS = ("typed_lookup", "null", "constant")


# ---------------------------------------------------------------------------
# dataset loading
# ---------------------------------------------------------------------------


def load_dataset(name: str) -> tuple[list[Data], np.ndarray, dict[str, Any]]:
    ds = TUDataset(root=str(TU_ROOT), name=str(name))
    graphs = [ds[i] for i in range(len(ds))]
    y = np.asarray([int(g.y.view(-1)[0]) for g in graphs], dtype=np.int64)
    uniq = sorted(set(y.tolist()))
    remap = {int(v): i for i, v in enumerate(uniq)}
    y = np.asarray([remap[int(v)] for v in y], dtype=np.int64)
    n_node_types = _n_node_types(graphs)
    n_edge_types = _n_edge_types(graphs)
    meta = {
        "name": str(name),
        "n_graphs": int(len(graphs)),
        "n_classes": int(len(uniq)),
        "classes": [int(v) for v in uniq],
        "mean_nodes": float(np.mean([int(g.num_nodes) for g in graphs])),
        "mean_edges": float(np.mean([int(g.num_edges) for g in graphs])),
        "n_node_types": int(n_node_types),
        "n_edge_types": int(n_edge_types),
        "has_node_features": bool(getattr(graphs[0], "x", None) is not None),
        "has_edge_features": bool(getattr(graphs[0], "edge_attr", None) is not None),
    }
    return graphs, y, meta


def _n_node_types(graphs: Sequence[Data]) -> int:
    first = graphs[0]
    x = getattr(first, "x", None)
    if x is None:
        return DEGREE_BINS
    return max(int(x.shape[1]), 1)


def _n_edge_types(graphs: Sequence[Data]) -> int:
    first = graphs[0]
    ea = getattr(first, "edge_attr", None)
    if ea is None:
        return 1
    return max(int(ea.shape[1]), 1)


def node_types_of(graph: Data, n_node_types: int) -> np.ndarray:
    x = getattr(graph, "x", None)
    if x is None:
        # structure-only dataset: clipped degree as the discrete node type
        degree = torch.bincount(
            graph.edge_index[0], minlength=int(graph.num_nodes)
        ).cpu().numpy()
        return np.minimum(degree, int(n_node_types) - 1).astype(np.int64)
    return np.argmax(x.cpu().numpy(), axis=1).astype(np.int64)


def edge_types_of(graph: Data) -> np.ndarray:
    ea = getattr(graph, "edge_attr", None)
    if ea is None:
        return np.zeros(int(graph.edge_index.shape[1]), dtype=np.int64)
    return np.argmax(ea.cpu().numpy(), axis=1).astype(np.int64)


# ---------------------------------------------------------------------------
# graph helpers
# ---------------------------------------------------------------------------


def adjacency(graph: Data) -> dict[int, dict[int, int]]:
    """Undirected adjacency: adj[u][v] = edge type."""
    adj: dict[int, dict[int, int]] = {
        int(v): {} for v in range(int(graph.num_nodes))
    }
    edges = graph.edge_index.cpu().numpy()
    etypes = edge_types_of(graph)
    for index in range(edges.shape[1]):
        u = int(edges[0, index])
        v = int(edges[1, index])
        adj[u][v] = int(etypes[index])
        adj[v][u] = int(etypes[index])
    return adj


def bfs_distances(adj: Mapping[int, Mapping[int, int]], source: int, n: int) -> np.ndarray:
    dist = np.full(int(n), -1, dtype=np.int64)
    dist[int(source)] = 0
    frontier = [int(source)]
    while frontier:
        nxt: list[int] = []
        for u in frontier:
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    nxt.append(v)
        frontier = nxt
    return dist


# ---------------------------------------------------------------------------
# patch records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PatchRecord:
    typed_certificate: bytes
    parent_certificate: bytes
    shell_descriptor: np.ndarray
    nodes: frozenset[int]
    boundary: frozenset[int]


@dataclass(frozen=True)
class GraphRecord:
    patches: tuple[PatchRecord, ...]
    pair_index: np.ndarray
    pair_relation: np.ndarray
    pair_bucket: np.ndarray
    global_context: np.ndarray
    y: int


def _shell_descriptor(
    node_types: np.ndarray,
    adj: Mapping[int, Mapping[int, int]],
    centre: int,
    distances: np.ndarray,
    n_node_types: int,
    n_edge_types: int,
) -> np.ndarray:
    n = int(node_types.shape[0])
    node_hist = np.zeros(((PATCH_RADIUS + 1) * n_node_types,), dtype=np.float32)
    nodes = [v for v in range(n) if 0 <= distances[v] <= PATCH_RADIUS]
    for v in nodes:
        d = int(distances[v])
        node_hist[d * n_node_types + int(node_types[v])] += 1.0
    edge_hist = np.zeros((n_edge_types,), dtype=np.float32)
    for u in nodes:
        for v, etype in adj[u].items():
            if v in nodes and u < v:
                edge_hist[int(etype)] += 1.0
    degree = np.asarray(
        [len(adj[v]) for v in nodes], dtype=np.float32
    )
    degree_stats = np.asarray(
        [
            float(len(adj[centre])),
            float(np.log1p(len(adj[centre]))),
            float(degree.mean()),
            float(degree.max()),
        ],
        dtype=np.float32,
    )
    patch_size = np.asarray(
        [
            float(len(nodes)),
            float(sum(1 for v in nodes if distances[v] == PATCH_RADIUS)),
            float(len(nodes)) / max(float(n), 1.0),
            float(len(adj[centre])) / max(float(n - 1), 1.0),
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [node_hist, edge_hist, degree_stats, patch_size]
    ).astype(np.float32)


def _certificate(
    node_types: np.ndarray,
    adj: Mapping[int, Mapping[int, int]],
    centre: int,
    radius: int,
    cache: dict[tuple[int, int, bytes], bytes],
) -> bytes:
    distances = bfs_distances(adj, centre, int(node_types.shape[0]))
    items: list[tuple[int, int, int]] = []
    for v in range(int(node_types.shape[0])):
        d = int(distances[v])
        if 0 <= d <= int(radius):
            parent_edge = -1
            if d > 0:
                # canonical parent-edge type: min over shortest-path parents
                parents = [
                    adj[p].get(v, -1)
                    for p in range(int(node_types.shape[0]))
                    if distances[p] == d - 1 and v in adj[p]
                ]
                parent_edge = min(parents) if parents else -1
            items.append((d, int(node_types[v]), int(parent_edge)))
    items.sort()
    payload = repr((int(radius), int(node_types[centre]), tuple(items))).encode()
    key = (int(radius), int(centre), payload)
    cached = cache.get(key)
    if cached is not None:
        return cached
    digest = hashlib.blake2b(payload, digest_size=16).digest()
    cache[key] = digest
    return digest


def _pair_relation(
    left: PatchRecord,
    right: PatchRecord,
    distance: int,
    adjacent_edge: int | None,
    n_edge_types: int,
) -> tuple[np.ndarray, int]:
    bucket = min(max(int(distance), 1), DISTANCE_BUCKETS) - 1
    distance_one_hot = np.zeros((DISTANCE_BUCKETS,), dtype=np.float32)
    distance_one_hot[bucket] = 1.0
    intersection = len(left.nodes & right.nodes)
    union = len(left.nodes | right.nodes)
    left_size = len(left.nodes)
    right_size = len(right.nodes)
    overlap = np.asarray(
        [
            float(intersection) / max(float(union), 1.0),
            float(intersection) / max(float(min(left_size, right_size)), 1.0),
            float(intersection) / max(float(max(left_size, right_size)), 1.0),
            float(abs(left_size - right_size)) / max(float(PATCH_RADIUS ** 2 + 10), 1.0),
            float(intersection) / max(float(PATCH_RADIUS ** 2 + 10), 1.0),
        ],
        dtype=np.float32,
    )
    boundary_intersection = len(left.boundary & right.boundary)
    boundary_union = len(left.boundary | right.boundary)
    boundary_min = min(len(left.boundary), len(right.boundary))
    boundary = np.asarray(
        [
            float(boundary_intersection) / max(float(boundary_union), 1.0),
            float(boundary_intersection) / max(float(boundary_min), 1.0),
            float(boundary_intersection > 0),
        ],
        dtype=np.float32,
    )
    adjacent = np.zeros((n_edge_types,), dtype=np.float32)
    if adjacent_edge is not None:
        adjacent[int(adjacent_edge)] = 1.0
    relation = np.concatenate(
        [
            distance_one_hot,
            np.asarray([np.log1p(float(distance))], dtype=np.float32),
            overlap,
            boundary,
            adjacent,
        ]
    ).astype(np.float32)
    return relation, bucket


def relation_width(n_edge_types: int) -> int:
    return DISTANCE_BUCKETS + 1 + 5 + 3 + int(n_edge_types)


def shell_width(n_node_types: int, n_edge_types: int) -> int:
    return (PATCH_RADIUS + 1) * int(n_node_types) + int(n_edge_types) + 8


GLOBAL_WIDTH = 8


def extract_graph(
    graph: Data,
    n_node_types: int,
    n_edge_types: int,
    certificate_cache: dict[tuple[int, int, bytes], bytes],
) -> GraphRecord:
    n = int(graph.num_nodes)
    node_types = node_types_of(graph, n_node_types)
    adj = adjacency(graph)
    distances_all = [bfs_distances(adj, v, n) for v in range(n)]
    patches: list[PatchRecord] = []
    for centre in range(n):
        distances = distances_all[centre]
        nodes = frozenset(v for v in range(n) if 0 <= distances[v] <= PATCH_RADIUS)
        boundary = frozenset(
            v for v in nodes if any(w not in nodes for w in adj[v])
        )
        patches.append(
            PatchRecord(
                typed_certificate=_certificate(
                    node_types, adj, centre, PATCH_RADIUS, certificate_cache
                ),
                parent_certificate=_certificate(
                    node_types, adj, centre, 1, certificate_cache
                ),
                shell_descriptor=_shell_descriptor(
                    node_types, adj, centre, distances, n_node_types, n_edge_types
                ),
                nodes=nodes,
                boundary=boundary,
            )
        )
    pair_sources: list[int] = []
    pair_targets: list[int] = []
    relations: list[np.ndarray] = []
    buckets: list[int] = []
    for left in range(n):
        for right in range(left + 1, n):
            distance = int(distances_all[left][right])
            if distance < 0:
                continue
            adjacent_edge = adj[left].get(right)
            relation, bucket = _pair_relation(
                patches[left], patches[right], distance, adjacent_edge, n_edge_types
            )
            pair_sources.append(left)
            pair_targets.append(right)
            relations.append(relation)
            buckets.append(bucket)
    if relations:
        pair_relation = np.stack(relations, axis=0).astype(np.float32)
    else:
        pair_relation = np.zeros((0, relation_width(n_edge_types)), dtype=np.float32)
    m = int(graph.num_edges) // 2
    degrees = np.asarray(
        [len(adj[v]) for v in range(n)], dtype=np.float32
    )
    global_context = np.asarray(
        [
            float(n),
            float(m),
            float(m) / max(float(n), 1.0),
            float(degrees.mean()) if n else 0.0,
            float(degrees.max()) if n else 0.0,
            float(degrees.std()) if n else 0.0,
            float(n_node_types),
            float(n_edge_types),
        ],
        dtype=np.float32,
    )
    return GraphRecord(
        patches=tuple(patches),
        pair_index=np.asarray([pair_sources, pair_targets], dtype=np.int64),
        pair_relation=pair_relation,
        pair_bucket=np.asarray(buckets, dtype=np.int64),
        global_context=global_context,
        y=int(graph.y.view(-1)[0]),
    )


# ---------------------------------------------------------------------------
# standardizer / vocabulary
# ---------------------------------------------------------------------------


class Standardizer:
    def __init__(self, mean: np.ndarray, scale: np.ndarray) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError("standardizer expects a non-empty 2-D matrix")
        mean = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = matrix.std(axis=0, dtype=np.float64).astype(np.float32)
        scale[~np.isfinite(scale) | (scale < 1.0e-6)] = 1.0
        return cls(mean, scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        output = (
            (np.asarray(values, dtype=np.float32) - self.mean) / self.scale
        ).astype(np.float32, copy=False)
        if not np.isfinite(output).all():
            raise FloatingPointError("non-finite standardized values")
        return output


def fit_vocabulary(
    records: Sequence[GraphRecord], field: str
) -> dict[bytes, int]:
    counts: Counter[bytes] = Counter()
    for record in records:
        for patch in record.patches:
            counts[getattr(patch, field)] += 1
    kept = [
        key for key, count in counts.most_common() if count >= MIN_VOCAB_FREQ
    ][:MAX_VOCAB]
    return {key: index + 1 for index, key in enumerate(kept)}


def encode_records(
    records: Sequence[GraphRecord],
    typed_vocabulary: Mapping[bytes, int],
    parent_vocabulary: Mapping[bytes, int],
    patch_standardizer: Standardizer,
    global_standardizer: Standardizer,
) -> list[Data]:
    output: list[Data] = []
    for record in records:
        patch_cont = patch_standardizer.transform(
            np.stack([patch.shell_descriptor for patch in record.patches], axis=0)
        )
        typed = np.asarray(
            [typed_vocabulary.get(p.typed_certificate, 0) for p in record.patches],
            dtype=np.int64,
        )
        parent = np.asarray(
            [parent_vocabulary.get(p.parent_certificate, 0) for p in record.patches],
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
                    global_standardizer.transform(record.global_context[None, :])
                ),
                y=torch.tensor([int(record.y)], dtype=torch.long),
                num_nodes=len(record.patches),
            )
        )
    return output


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


class _MLP(nn.Module):
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


class TuPatchPathModel(nn.Module):
    """Compact recurrent pair--centre backbone with a switchable local token."""

    def __init__(
        self,
        typed_vocabulary_size: int,
        parent_vocabulary_size: int,
        *,
        shell_width: int,
        relation_width: int,
        global_width: int = GLOBAL_WIDTH,
        n_classes: int = 2,
        token_width: int = TOKEN_WIDTH,
        patch_hidden: int = PATCH_HIDDEN,
        pair_hidden: int = PAIR_HIDDEN,
        center_hidden: int = CENTER_HIDDEN,
        dropout: float = DROPOUT,
        patch_representation: str = "typed_lookup",
        recurrence_rounds: int = RECURRENCE_ROUNDS,
    ) -> None:
        super().__init__()
        if patch_representation not in TRAINABLE_REPRESENTATIONS:
            raise ValueError(
                f"unknown patch_representation={patch_representation!r}; "
                f"expected {TRAINABLE_REPRESENTATIONS}"
            )
        self.patch_representation = str(patch_representation)
        self.token_width = int(token_width)
        self.patch_hidden = int(patch_hidden)
        self.pair_hidden = int(pair_hidden)
        self.recurrence_rounds = int(recurrence_rounds)

        self.local_token_constant: nn.Parameter | None = None
        self.typed_embedding: nn.Embedding | None = nn.Embedding(
            int(typed_vocabulary_size), int(token_width)
        )
        if self.patch_representation == "null":
            del self.typed_embedding
            self.typed_embedding = None
        elif self.patch_representation == "constant":
            del self.typed_embedding
            self.typed_embedding = None
            self.local_token_constant = nn.Parameter(torch.zeros(int(token_width)))

        parent_width = max(int(token_width // 2), 1)
        self.parent_embedding = nn.Embedding(int(parent_vocabulary_size), parent_width)
        self.patch_encoder = _MLP(
            int(shell_width) + int(token_width) + parent_width,
            max(int(patch_hidden), 64),
            int(patch_hidden),
            float(dropout),
        )
        self.global_encoder = _MLP(
            int(global_width),
            max(int(patch_hidden // 2), 32),
            32,
            float(dropout),
        )
        self.pair_projection = nn.Linear(int(patch_hidden), int(pair_hidden), bias=False)
        self.relation_encoder = _MLP(
            int(relation_width),
            max(int(pair_hidden), 32),
            int(pair_hidden),
            float(dropout),
        )
        self.distance_gate = nn.Embedding(DISTANCE_BUCKETS, int(pair_hidden))
        self.pair_encoder = _MLP(
            4 * int(pair_hidden),
            max(2 * int(pair_hidden), 64),
            int(pair_hidden),
            float(dropout),
        )
        self.center_context_width = DISTANCE_BUCKETS * (2 * int(pair_hidden) + 1)
        self.center_update = nn.Sequential(
            nn.Linear(int(patch_hidden) + self.center_context_width, int(center_hidden)),
            nn.LayerNorm(int(center_hidden)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(center_hidden), int(patch_hidden)),
        )
        self.readout_width = (
            2 * int(patch_hidden) + 1 + DISTANCE_BUCKETS * (2 * int(pair_hidden) + 1)
        )
        hidden = max(int(patch_hidden) * 2, 96)
        self.head = nn.Sequential(
            nn.Linear(self.readout_width + 32, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, int(patch_hidden)),
            nn.ReLU(),
            nn.Linear(int(patch_hidden), int(n_classes)),
        )

    def _patch_token_value(self, data: Data) -> torch.Tensor:
        if self.typed_embedding is not None:
            return self.typed_embedding(data.typed_token)
        if self.local_token_constant is not None:
            count = int(data.patch_cont.shape[0])
            return self.local_token_constant.view(1, -1).expand(count, -1)
        count = int(data.patch_cont.shape[0])
        return data.patch_cont.new_zeros((count, int(self.token_width)))

    def _pair_value(
        self,
        patch: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        relation: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        left = self.pair_projection(patch[source])
        right = self.pair_projection(patch[target])
        pair_input = torch.cat(
            [
                left + right,
                torch.abs(left - right),
                left * right * gate,
                relation,
            ],
            dim=1,
        )
        return self.pair_encoder(pair_input)

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

    @staticmethod
    def _pool_pairs_to_centres(
        value: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        pair_bucket: torch.Tensor,
        n_centres: int,
    ) -> torch.Tensor:
        blocks: list[torch.Tensor] = []
        width = int(value.shape[1])
        for bucket in range(DISTANCE_BUCKETS):
            mask = pair_bucket == int(bucket)
            current = value[mask]
            current_source = source[mask]
            current_target = target[mask]
            total = torch.zeros(
                (int(n_centres), width), device=value.device, dtype=value.dtype
            )
            squared = torch.zeros_like(total)
            counts = torch.zeros(
                (int(n_centres), 1), device=value.device, dtype=value.dtype
            )
            if current.numel():
                endpoints = torch.cat([current_source, current_target], dim=0)
                duplicated = torch.cat([current, current], dim=0)
                total.index_add_(0, endpoints, duplicated)
                squared.index_add_(0, endpoints, duplicated * duplicated)
                counts.index_add_(
                    0,
                    endpoints,
                    torch.ones(
                        (endpoints.shape[0], 1), device=value.device, dtype=value.dtype
                    ),
                )
            denominator = counts.clamp_min(1.0)
            mean = total / denominator
            variance = (squared / denominator - mean * mean).clamp_min(0.0)
            std = torch.sqrt(variance + 1.0e-8)
            occupied = (counts > 0).to(value.dtype)
            blocks.append(torch.cat([mean, std * occupied, torch.log1p(counts)], dim=1))
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
                    self._patch_token_value(data),
                    self.parent_embedding(data.parent_token),
                ],
                dim=1,
            )
        )
        source = data.pair_index[0]
        target = data.pair_index[1]
        relation = self.relation_encoder(data.pair_relation)
        gate = 1.0 + torch.tanh(self.distance_gate(data.pair_bucket))
        pair_value = None
        for _round in range(self.recurrence_rounds):
            pair_value = self._pair_value(patch, source, target, relation, gate)
            center_context = self._pool_pairs_to_centres(
                pair_value, source, target, data.pair_bucket, int(patch.shape[0])
            )
            patch = patch + self.center_update(
                torch.cat([patch, center_context], dim=1)
            )
        unary = self._pool_nodes(patch, data.batch, n_graphs)
        assert pair_value is not None
        relation_readout = self._pool_pairs(
            pair_value, data.batch[source], data.pair_bucket, n_graphs
        )
        graph_hidden = self.global_encoder(global_context)
        return self.head(
            torch.cat([unary, relation_readout, graph_hidden], dim=1)
        )

    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))


def make_loader(
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


# ---------------------------------------------------------------------------
# strict splits
# ---------------------------------------------------------------------------


def build_folds(
    y: np.ndarray, k: int, *, seed: int = SPLIT_SEED, valid_frac: float = VALID_FRAC
) -> list[dict[str, np.ndarray]]:
    """Outer stratified k-fold; train/valid carved from the non-test folds."""
    from sklearn.model_selection import StratifiedKFold, train_test_split

    y = np.asarray(y)
    outer = StratifiedKFold(n_splits=int(k), shuffle=True, random_state=int(seed))
    folds: list[dict[str, np.ndarray]] = []
    for fold, (train_valid, test) in enumerate(outer.split(np.zeros_like(y), y)):
        train, valid = train_test_split(
            train_valid,
            test_size=float(valid_frac),
            random_state=int(seed) + 1000 * (fold + 1),
            stratify=y[train_valid],
        )
        folds.append(
            {
                "fold": np.int64(fold),
                "train": np.sort(np.asarray(train, dtype=np.int64)),
                "valid": np.sort(np.asarray(valid, dtype=np.int64)),
                "test": np.sort(np.asarray(test, dtype=np.int64)),
            }
        )
    return folds


def folds_fingerprint(folds: Sequence[Mapping[str, np.ndarray]]) -> str:
    payload = [
        {
            "fold": int(f["fold"]),
            "train": [int(v) for v in f["train"]],
            "valid": [int(v) for v in f["valid"]],
            "test": [int(v) for v in f["test"]],
        }
        for f in folds
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()


def check_folds(
    folds: Sequence[Mapping[str, np.ndarray]], n: int
) -> dict[str, Any]:
    covered = np.zeros(int(n), dtype=np.int64)
    disjoint = True
    for f in folds:
        train, valid, test = f["train"], f["valid"], f["test"]
        if set(train.tolist()) & set(valid.tolist()):
            disjoint = False
        if set(train.tolist()) & set(test.tolist()):
            disjoint = False
        if set(valid.tolist()) & set(test.tolist()):
            disjoint = False
        covered[test] += 1
    return {
        "all_disjoint": bool(disjoint),
        "each_graph_tested_once": bool(np.all(covered == 1)),
        "n_graphs": int(n),
        "fold_sizes": [
            {
                "fold": int(f["fold"]),
                "train": int(len(f["train"])),
                "valid": int(len(f["valid"])),
                "test": int(len(f["test"])),
            }
            for f in folds
        ],
    }
