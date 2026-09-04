"""A small explicit centre-level structure--attribute relation network.

This runner tests whether the current graph-level count/readout route loses
useful information before structure and chemistry are allowed to interact.
It is deliberately not a GPS variant:

* topology-only rooted-WL features are computed independently for every atom
  centre;
* chemical features are computed for the same centre;
* conditional fusion happens before graph pooling;
* relation propagation is over explicit molecular centre pairs with bounded
  shortest-path distance and patch-overlap features;
* the graph readout is sum/mean/std, with no global attention.

The train/validation phase selects an epoch using only official validation.
The test phase refits on official train+validation for that selected epoch and
then evaluates the frozen model once.  ``--max-*`` limits are intended only
for smoke/development runs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import mean_absolute_error
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import (
    REPO_ROOT,
    _data_to_graph,
    _load_zinc,
    _resolve,
    source_audit,
)
from tracks.ksvd.experiments.luyin16.zinc_s_marginal import (
    ATOM_CATEGORIES,
    BOND_CATEGORIES,
    _ego_distances,
    _rooted_wl_roles,
)


DEFAULT_CONFIG = REPO_ROOT / "tracks/ksvd/configs/luyin16/center_relation_network.yaml"
NODE_ROLE_BINS = 64
EDGE_ROLE_BINS = 32
TOPOLOGY_WIDTH = NODE_ROLE_BINS + NODE_ROLE_BINS + EDGE_ROLE_BINS + 8
ATTRIBUTE_WIDTH = ATOM_CATEGORIES + BOND_CATEGORIES + ATOM_CATEGORIES + BOND_CATEGORIES
RELATION_WIDTH = 3 + 2 + BOND_CATEGORIES
MODES = (
    "attribute_only",
    "structure_only",
    "center_concat",
    "conditional_fusion",
    "conditional_relation",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
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


def _one_hot(value: int, width: int) -> np.ndarray:
    output = np.zeros(int(width), dtype=np.float32)
    if value < 0 or value >= int(width):
        raise ValueError(f"category {value} outside [0, {int(width) - 1}]")
    output[int(value)] = 1.0
    return output


def _normalized_histogram(values: Sequence[int], width: int) -> np.ndarray:
    output = np.bincount(np.asarray(list(values), dtype=np.int64), minlength=int(width)).astype(
        np.float32
    )
    return output / max(float(output.sum()), 1.0)


def _shuffle_permutation(n_rows: int, seed: int, key: int) -> np.ndarray:
    """Return a deterministic non-trivial permutation for one graph.

    ``key`` identifies the graph within the complete experiment rather than
    within one split.  This makes the train+valid refit use exactly the same
    null assignment that was used during the train-to-valid phase.
    """
    if int(n_rows) <= 1:
        return np.arange(int(n_rows), dtype=np.int64)
    digest = hashlib.blake2b(digest_size=8)
    digest.update(b"luyin16-centre-attribute-shuffle-v1")
    digest.update(np.asarray([int(seed), int(key)], dtype=np.int64).tobytes())
    rng = np.random.default_rng(int.from_bytes(digest.digest(), "little"))
    permutation = np.asarray(rng.permutation(int(n_rows)), dtype=np.int64)
    # A random identity permutation is a valid draw, but would make an
    # individual graph contribute no control perturbation.  Use a cyclic
    # shift in that rare case while preserving the exact attribute bag.
    if np.array_equal(permutation, np.arange(int(n_rows), dtype=np.int64)):
        permutation = np.roll(permutation, 1)
    return permutation


def _shuffle_center_attributes(data: Data, *, seed: int, key: int) -> Data:
    """Break centre-to-attribute binding while preserving every other field."""
    if not hasattr(data, "node_attributes"):
        raise ValueError("centre data has no node_attributes field")
    attributes = data.node_attributes
    permutation = _shuffle_permutation(int(attributes.shape[0]), seed, key)
    shuffled = data.clone()
    permutation_tensor = torch.from_numpy(permutation).to(attributes.device)
    shuffled.node_attributes = attributes.index_select(0, permutation_tensor)
    return shuffled


class _ShuffledCenterDataset(Dataset):
    """Lazy matched null dataset; only the attribute tensor is copied."""

    def __init__(self, graphs: Sequence[Data], *, seed: int, keys: Sequence[int]) -> None:
        if len(graphs) != len(keys):
            raise ValueError("shuffle keys are not aligned with graph rows")
        self.graphs = graphs
        self.seed = int(seed)
        self.keys = tuple(int(key) for key in keys)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: int) -> Data:
        return _shuffle_center_attributes(
            self.graphs[int(index)], seed=self.seed, key=self.keys[int(index)]
        )


def _all_distances(graph, source: int) -> dict[int, int]:
    distances = {int(source): 0}
    queue: deque[int] = deque([int(source)])
    while queue:
        node = queue.popleft()
        for neighbor in sorted(graph.neighbors(node)):
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    return distances


def _centre_arrays(
    graph,
    node_types: np.ndarray,
    edge_types: Mapping[tuple[int, int], int],
    *,
    radius: int,
    wl_rounds: int,
    node_role_bins: int,
    edge_role_bins: int,
    relation_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Build aligned centre features and explicit centre-pair relations.

    The first return block is topology-only.  In particular, the rooted-WL
    encoder is called with ``typed_wl=False`` and contains no atom or bond
    category.  Chemistry enters only in the second block and in the optional
    adjacent-bond relation component.
    """
    if node_types.shape != (graph.n,):
        raise ValueError("node type vector is not aligned with graph")
    topology_rows: list[np.ndarray] = []
    attribute_rows: list[np.ndarray] = []
    patches: list[set[int]] = []

    for center in graph.nodes:
        nodes, edges, node_roles, edge_roles = _rooted_wl_roles(
            graph,
            int(center),
            int(radius),
            typed_wl=False,
            wl_rounds=int(wl_rounds),
            node_role_bins=int(node_role_bins),
            edge_role_bins=int(edge_role_bins),
        )
        node_set = set(int(value) for value in nodes)
        patches.append(node_set)
        center_position = nodes.index(int(center))
        root_role = int(node_roles[center_position])
        root_role_vector = _one_hot(root_role, int(node_role_bins))
        node_role_hist = _normalized_histogram(node_roles.tolist(), int(node_role_bins))
        edge_role_hist = _normalized_histogram(edge_roles.tolist(), int(edge_role_bins))
        shell_distances = _ego_distances(graph, int(center), int(radius))
        shell = np.asarray(
            [
                float(sum(distance == shell_id for distance in shell_distances.values()))
                / max(float(len(nodes)), 1.0)
                for shell_id in range(4)
            ],
            dtype=np.float32,
        )
        local_cycle_rank = max(len(edges) - len(nodes) + 1, 0)
        topology_rows.append(
            np.concatenate(
                [
                    root_role_vector,
                    node_role_hist,
                    edge_role_hist,
                    shell,
                    np.asarray(
                        [
                            float(len(nodes)) / max(float(graph.n), 1.0),
                            float(len(edges)) / max(float(graph.n), 1.0),
                            float(len(graph.neighbors(int(center)))) / 4.0,
                            float(local_cycle_rank) / max(float(graph.n), 1.0),
                        ],
                        dtype=np.float32,
                    ),
                ]
            ).astype(np.float32, copy=False)
        )

        local_atoms = _normalized_histogram(
            [int(node_types[node]) for node in nodes], ATOM_CATEGORIES
        )
        local_bonds = _normalized_histogram(
            [int(edge_types[graph.edge_key(left, right)]) for left, right in edges],
            BOND_CATEGORIES,
        )
        incident_bonds = _normalized_histogram(
            [int(edge_types[graph.edge_key(int(center), neighbor)]) for neighbor in graph.neighbors(int(center))],
            BOND_CATEGORIES,
        )
        attribute_rows.append(
            np.concatenate(
                [
                    _one_hot(int(node_types[int(center)]), ATOM_CATEGORIES),
                    incident_bonds,
                    local_atoms,
                    local_bonds,
                ]
            ).astype(np.float32, copy=False)
        )

    edge_sources: list[int] = []
    edge_targets: list[int] = []
    relation_rows: list[np.ndarray] = []
    for source in graph.nodes:
        distances = _all_distances(graph, int(source))
        for target in graph.nodes:
            if int(source) == int(target):
                continue
            distance = distances.get(int(target))
            if distance is None or distance > int(relation_radius):
                continue
            distance_one_hot = _one_hot(int(distance) - 1, int(relation_radius))
            intersection = len(patches[int(source)] & patches[int(target)])
            union = len(patches[int(source)] | patches[int(target)])
            minimum = min(len(patches[int(source)]), len(patches[int(target)]))
            bond_one_hot = np.zeros(BOND_CATEGORIES, dtype=np.float32)
            if distance == 1:
                bond_one_hot = _one_hot(
                    int(edge_types[graph.edge_key(int(source), int(target))]),
                    BOND_CATEGORIES,
                )
            edge_sources.append(int(source))
            edge_targets.append(int(target))
            relation_rows.append(
                np.concatenate(
                    [
                        distance_one_hot,
                        np.asarray(
                            [
                                float(intersection) / max(float(union), 1.0),
                                float(intersection) / max(float(minimum), 1.0),
                            ],
                            dtype=np.float32,
                        ),
                        bond_one_hot,
                    ]
                ).astype(np.float32, copy=False)
            )

    topology = np.stack(topology_rows, axis=0).astype(np.float32, copy=False)
    attributes = np.stack(attribute_rows, axis=0).astype(np.float32, copy=False)
    if topology.shape[1] != int(node_role_bins) * 2 + int(edge_role_bins) + 8:
        raise RuntimeError(f"topology width changed: {topology.shape}")
    if attributes.shape[1] != ATTRIBUTE_WIDTH:
        raise RuntimeError(f"attribute width changed: {attributes.shape}")
    if relation_rows:
        relation_index = np.asarray([edge_sources, edge_targets], dtype=np.int64)
        relations = np.stack(relation_rows, axis=0).astype(np.float32, copy=False)
    else:
        relation_index = np.zeros((2, 0), dtype=np.int64)
        relations = np.zeros((0, RELATION_WIDTH), dtype=np.float32)
    metadata = {
        "n_nodes": int(graph.n),
        "n_edges": int(graph.num_edges()),
        "n_relations_directed": int(relations.shape[0]),
        "mean_patch_nodes": float(np.mean([len(patch) for patch in patches])),
        "relation_radius": int(relation_radius),
    }
    return topology, attributes, relation_index, relations, metadata


def _data_to_center_data(
    data: Any,
    *,
    radius: int,
    wl_rounds: int,
    node_role_bins: int,
    edge_role_bins: int,
    relation_radius: int,
) -> tuple[Data, dict[str, Any]]:
    graph, node_types, edge_types = _data_to_graph(data)
    if node_types.size and (node_types.min() < 0 or node_types.max() >= ATOM_CATEGORIES):
        raise ValueError(f"atom category outside schema: {node_types.min()}..{node_types.max()}")
    if edge_types and (min(edge_types.values()) < 0 or max(edge_types.values()) >= BOND_CATEGORIES):
        raise ValueError(f"bond category outside schema: {min(edge_types.values())}..{max(edge_types.values())}")
    topology, attributes, relation_index, relations, metadata = _centre_arrays(
        graph,
        node_types,
        edge_types,
        radius=radius,
        wl_rounds=wl_rounds,
        node_role_bins=node_role_bins,
        edge_role_bins=edge_role_bins,
        relation_radius=relation_radius,
    )
    result = Data(
        node_topology=torch.from_numpy(topology),
        node_attributes=torch.from_numpy(attributes),
        center_edge_index=torch.from_numpy(relation_index),
        center_edge_attr=torch.from_numpy(relations),
        y=torch.tensor([float(data.y.view(-1)[0])], dtype=torch.float32),
    )
    result.num_nodes = int(graph.n)
    return result, metadata


def _load_data_split(dataset: Any, limits: int | None, representation: Mapping[str, Any]) -> tuple[list[Data], dict[str, Any]]:
    selected = dataset if limits is None else dataset[: int(limits)]
    values: list[Data] = []
    metadata: list[dict[str, Any]] = []
    for position, data in enumerate(selected):
        value, details = _data_to_center_data(
            data,
            radius=int(representation["radius"]),
            wl_rounds=int(representation["wl_rounds"]),
            node_role_bins=int(representation["node_role_bins"]),
            edge_role_bins=int(representation["edge_role_bins"]),
            relation_radius=int(representation["relation_radius"]),
        )
        values.append(value)
        metadata.append(details)
        if position and position % 500 == 0:
            print(f"center features: {position}/{len(selected)}", flush=True)
    return values, {
        "n_graphs": int(len(values)),
        "mean_nodes": float(np.mean([row["n_nodes"] for row in metadata])) if metadata else 0.0,
        "mean_relations_directed": float(np.mean([row["n_relations_directed"] for row in metadata])) if metadata else 0.0,
        "mean_patch_nodes": float(np.mean([row["mean_patch_nodes"] for row in metadata])) if metadata else 0.0,
    }


def _save_center_cache(path: Path, graph_sets: Sequence[Sequence[Data]], metadata: Sequence[Mapping[str, Any]], signature: str) -> None:
    """Persist CPU tensors so an interrupted training run skips feature building."""
    arrays: dict[str, np.ndarray] = {"signature": np.asarray([signature])}
    for split, graphs in zip(("train", "valid", "test"), graph_sets, strict=True):
        arrays[f"{split}_topology"] = np.concatenate(
            [graph.node_topology.numpy() for graph in graphs], axis=0
        ).astype(np.float32, copy=False)
        arrays[f"{split}_attributes"] = np.concatenate(
            [graph.node_attributes.numpy() for graph in graphs], axis=0
        ).astype(np.float32, copy=False)
        arrays[f"{split}_edge_index"] = np.concatenate(
            [graph.center_edge_index.numpy() + int(offset) for offset, graph in zip(
                np.cumsum([0] + [int(item.num_nodes) for item in graphs[:-1]]), graphs, strict=True
            )],
            axis=1,
        ).astype(np.int64, copy=False)
        arrays[f"{split}_edge_attr"] = np.concatenate(
            [graph.center_edge_attr.numpy() for graph in graphs], axis=0
        ).astype(np.float32, copy=False)
        arrays[f"{split}_node_offsets"] = np.asarray(
            [0, *np.cumsum([int(item.num_nodes) for item in graphs])], dtype=np.int64
        )
        arrays[f"{split}_edge_offsets"] = np.asarray(
            [0, *np.cumsum([int(item.center_edge_attr.shape[0]) for item in graphs])], dtype=np.int64
        )
        arrays[f"{split}_labels"] = np.asarray(
            [float(item.y.view(-1)[0]) for item in graphs], dtype=np.float32
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _load_center_cache(path: Path, signature: str) -> tuple[list[list[Data]], list[dict[str, Any]]]:
    with np.load(path, allow_pickle=False) as archive:
        cached_signature = str(np.asarray(archive["signature"]).reshape(-1)[0])
        if cached_signature != signature:
            raise ValueError(f"center cache signature mismatch: {path}")
        graph_sets: list[list[Data]] = []
        metadata: list[dict[str, Any]] = []
        for split in ("train", "valid", "test"):
            node_offsets = np.asarray(archive[f"{split}_node_offsets"], dtype=np.int64)
            edge_offsets = np.asarray(archive[f"{split}_edge_offsets"], dtype=np.int64)
            topology = np.asarray(archive[f"{split}_topology"], dtype=np.float32)
            attributes = np.asarray(archive[f"{split}_attributes"], dtype=np.float32)
            edge_index = np.asarray(archive[f"{split}_edge_index"], dtype=np.int64)
            edge_attr = np.asarray(archive[f"{split}_edge_attr"], dtype=np.float32)
            labels = np.asarray(archive[f"{split}_labels"], dtype=np.float32)
            graphs: list[Data] = []
            for index in range(labels.size):
                node_start, node_stop = int(node_offsets[index]), int(node_offsets[index + 1])
                edge_start, edge_stop = int(edge_offsets[index]), int(edge_offsets[index + 1])
                local_edges = edge_index[:, edge_start:edge_stop] - node_start
                item = Data(
                    node_topology=torch.from_numpy(topology[node_start:node_stop]),
                    node_attributes=torch.from_numpy(attributes[node_start:node_stop]),
                    center_edge_index=torch.from_numpy(local_edges),
                    center_edge_attr=torch.from_numpy(edge_attr[edge_start:edge_stop]),
                    y=torch.tensor([float(labels[index])], dtype=torch.float32),
                )
                item.num_nodes = node_stop - node_start
                graphs.append(item)
            graph_sets.append(graphs)
            metadata.append({
                "n_graphs": int(len(graphs)),
                "mean_nodes": float(np.mean([item.num_nodes for item in graphs])) if graphs else 0.0,
                "mean_relations_directed": float(np.mean([item.center_edge_attr.shape[0] for item in graphs])) if graphs else 0.0,
                "mean_patch_nodes": 0.0,
                "cache_hit": True,
            })
    return graph_sets, metadata


class _MLP(nn.Module):
    def __init__(self, input_width: int, output_width: int, hidden_width: int | None = None) -> None:
        super().__init__()
        hidden = int(hidden_width or output_width)
        self.layers = nn.Sequential(
            nn.Linear(int(input_width), hidden),
            nn.ReLU(),
            nn.Linear(hidden, int(output_width)),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class CenterRelationNetwork(nn.Module):
    def __init__(self, mode: str, topology_width: int, attribute_width: int, relation_width: int, hidden: int, relation_layers: int) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
        self.mode = str(mode)
        self.structural_encoder = _MLP(topology_width, hidden)
        self.attribute_encoder = _MLP(attribute_width, hidden)
        self.concat_fusion = _MLP(2 * hidden, hidden)
        self.conditional_fusion = _MLP(3 * hidden, hidden)
        self.relation_encoder = _MLP(relation_width, hidden)
        self.message_layers = nn.ModuleList(
            _MLP(2 * hidden, hidden) for _ in range(int(relation_layers))
        )
        self.update_layers = nn.ModuleList(
            _MLP(2 * hidden, hidden) for _ in range(int(relation_layers))
        )
        self.readout = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    @staticmethod
    def _pool(value: torch.Tensor, batch: torch.Tensor, n_graphs: int) -> torch.Tensor:
        total = torch.zeros((n_graphs, value.shape[1]), device=value.device, dtype=value.dtype)
        total.index_add_(0, batch, value)
        counts = torch.bincount(batch, minlength=n_graphs).to(value.dtype).unsqueeze(1).clamp_min(1.0)
        mean = total / counts
        second = torch.zeros_like(total)
        second.index_add_(0, batch, value * value)
        # A zero-variance centre population has an infinite derivative for
        # sqrt(var) at var=0.  MolHIV contains batches with identical hidden
        # rows early in training, so keep the same std readout while adding a
        # tiny numerical floor for a finite backward pass.
        variance = (second / counts - mean * mean).clamp_min(0.0)
        std = (variance + 1.0e-12).sqrt()
        return torch.cat([total, mean, std], dim=1)

    def forward(self, data: Data) -> torch.Tensor:
        pooled = self.encode(data)
        return self.readout(pooled).view(-1)

    def encode(self, data: Data) -> torch.Tensor:
        """Return the pooled centre representation before the scalar head.

        Keeping this representation accessible lets downstream controls add
        an independently scaled graph-level branch without duplicating the
        centre encoder.  The default ``forward`` path remains unchanged.
        """
        structural = self.structural_encoder(data.node_topology)
        attribute = self.attribute_encoder(data.node_attributes)
        if self.mode == "attribute_only":
            hidden = attribute
        elif self.mode == "structure_only":
            hidden = structural
        elif self.mode == "center_concat":
            hidden = self.concat_fusion(torch.cat([structural, attribute], dim=1))
        else:
            hidden = self.conditional_fusion(
                torch.cat([structural, attribute, structural * attribute], dim=1)
            )

        if self.mode == "conditional_relation":
            source = data.center_edge_index[0]
            target = data.center_edge_index[1]
            relation = self.relation_encoder(data.center_edge_attr)
            for message_layer, update_layer in zip(self.message_layers, self.update_layers, strict=True):
                messages = message_layer(torch.cat([hidden[source], relation], dim=1))
                aggregate = torch.zeros_like(hidden)
                degree = torch.zeros((hidden.shape[0], 1), device=hidden.device, dtype=hidden.dtype)
                if messages.numel():
                    aggregate.index_add_(0, target, messages)
                    degree.index_add_(0, target, torch.ones((target.shape[0], 1), device=hidden.device, dtype=hidden.dtype))
                    aggregate = aggregate / degree.clamp_min(1.0)
                hidden = hidden + update_layer(torch.cat([hidden, aggregate], dim=1))
                hidden = F.relu(hidden)

        batch = data.batch
        n_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
        return self._pool(hidden, batch, n_graphs)


def _make_loader(graphs: Sequence[Data], batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed)) if shuffle else None
    return DataLoader(
        list(graphs),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=0,
    )


def _make_graph_loader(graphs: Any, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed)) if shuffle else None
    return DataLoader(
        graphs,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=0,
    )


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions.append(model(batch).cpu().numpy())
            targets.append(batch.y.view(-1).cpu().numpy())
    prediction = np.concatenate(predictions).astype(np.float64, copy=False)
    target = np.concatenate(targets).astype(np.float64, copy=False)
    return float(mean_absolute_error(target, prediction)), prediction


def _train_phase(
    model: nn.Module,
    train_graphs: Any,
    eval_graphs: Any,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    evaluate_every: int,
    patience: int | None,
    phase: str,
    select_best: bool = True,
) -> dict[str, Any]:
    _seed_everything(seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    train_loader = _make_graph_loader(train_graphs, batch_size, True, seed + 91011)
    eval_loader = _make_graph_loader(eval_graphs, batch_size, False, seed + 91012)
    best_mae = float("inf")
    best_epoch = 1
    best_state: dict[str, Any] | None = None
    trace: list[dict[str, float | int]] = []
    epochs_without_improvement = 0
    losses: list[float] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch in train_loader:
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
        train_loss = total_loss / max(seen, 1)
        losses.append(float(train_loss))
        should_evaluate = select_best and (
            epoch == 1 or epoch % max(int(evaluate_every), 1) == 0 or epoch == int(epochs)
        )
        current_mae: float | None = None
        if should_evaluate:
            current_mae, _ = _evaluate(model, eval_loader, device)
            if select_best:
                trace.append({"epoch": int(epoch), "mae": float(current_mae)})
            if select_best and current_mae < best_mae:
                best_mae = float(current_mae)
                best_epoch = int(epoch)
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += max(int(evaluate_every), 1)
        if epoch == 1 or epoch % max(1, int(epochs) // 10) == 0 or (
            current_mae is not None and current_mae == best_mae
        ):
            suffix = "" if current_mae is None else f" valid_mae={current_mae:.6f}"
            print(
                f"center phase={phase} epoch={epoch:03d}/{epochs} l1={train_loss:.6f}{suffix}",
                flush=True,
            )
        if select_best and patience is not None and epochs_without_improvement >= int(patience):
            print(f"center phase={phase} early_stop epoch={epoch} best_epoch={best_epoch}", flush=True)
            break
    if select_best and best_state is not None:
        model.load_state_dict(best_state)
    final_mae, prediction = _evaluate(model, eval_loader, device)
    return {
        "phase": phase,
        "epochs_run": int(len(losses)),
        "selected_epoch": int(best_epoch),
        "best_valid_mae": float(best_mae),
        "final_eval_mae": float(final_mae),
        "validation_trace": trace,
        "train_l1": losses,
        "prediction": prediction,
    }


def _run_mode(
    mode: str,
    train_graphs: Any,
    valid_graphs: Any,
    test_graphs: Any,
    *,
    config: Mapping[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    model_config = config["model"]
    hidden = int(model_config["hidden"])
    relation_layers = int(model_config["relation_layers"])
    # Seed before constructing the model so the reported model seed controls
    # parameter initialization as well as loader order and torch operations.
    _seed_everything(seed)
    train_model = CenterRelationNetwork(
        mode,
        TOPOLOGY_WIDTH,
        ATTRIBUTE_WIDTH,
        RELATION_WIDTH,
        hidden,
        relation_layers,
    )
    valid_phase = _train_phase(
        train_model,
        train_graphs,
        valid_graphs,
        epochs=int(model_config["epochs"]),
        batch_size=int(model_config["batch_size"]),
        learning_rate=float(model_config["learning_rate"]),
        weight_decay=float(model_config.get("weight_decay", 0.0)),
        seed=int(seed),
        device=device,
        evaluate_every=int(model_config.get("evaluate_every", 5)),
        patience=None if model_config.get("patience") is None else int(model_config["patience"]),
        phase=f"{mode}:train-to-valid",
        select_best=True,
    )
    selected_epoch = int(valid_phase["selected_epoch"])

    # Reset before constructing the refit model as well; otherwise its initial
    # weights would depend on the preceding validation run's RNG state.
    _seed_everything(seed)
    refit_model = CenterRelationNetwork(
        mode,
        TOPOLOGY_WIDTH,
        ATTRIBUTE_WIDTH,
        RELATION_WIDTH,
        hidden,
        relation_layers,
    )
    combined = list(train_graphs) + list(valid_graphs)
    test_phase = _train_phase(
        refit_model,
        combined,
        test_graphs,
        epochs=selected_epoch,
        batch_size=int(model_config["batch_size"]),
        learning_rate=float(model_config["learning_rate"]),
        weight_decay=float(model_config.get("weight_decay", 0.0)),
        seed=int(seed),
        device=device,
        evaluate_every=max(selected_epoch + 1, 1),
        patience=None,
        phase=f"{mode}:train+valid-to-test",
        select_best=False,
    )
    return {
        "mode": mode,
        "seed": int(seed),
        "valid": {
            "mae": float(valid_phase["final_eval_mae"]),
            "best_mae": float(valid_phase["best_valid_mae"]),
            "selected_epoch": selected_epoch,
            "epochs_run": int(valid_phase["epochs_run"]),
            "trace": valid_phase["validation_trace"],
        },
        "test_after_train_valid_refit": {
            "mae": float(test_phase["final_eval_mae"]),
            "epochs_run": int(test_phase["epochs_run"]),
        },
        "parameters": int(sum(parameter.numel() for parameter in train_model.parameters() if parameter.requires_grad)),
    }


def _render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        f"# {result['protocol_id']}",
        "",
        "Explicit centre-level conditional fusion and bounded centre-relation propagation on official PyG ZINC.",
        "",
        f"- split sizes: `{result['data']['sizes']}`",
        f"- representation: topology-only rooted-WL radius `{result['representation']['radius']}`, relation radius `{result['representation']['relation_radius']}`",
        f"- device: `{result['training']['device']}`; epoch budget `{result['training']['epochs']}` with early stopping",
        "",
        "| mode | valid MAE | selected epoch | test MAE after train+valid refit |",
        "|---|---:|---:|---:|",
    ]
    for name, row in result["modes"].items():
        lines.append(
            f"| `{name}` | {row['valid']['mean_mae']:.6f} | "
            f"{row['valid']['selected_epoch_mean']:.1f} | "
            f"{row['test_after_train_valid_refit']['mean_mae']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: `center_concat` tests centre alignment, `conditional_fusion` adds the same-centre multiplicative interaction, and `conditional_relation` additionally propagates over explicit distance/overlap/bond relations.",
            "",
            f"Runtime: `{result['runtime_seconds']:.1f}s`.",
            "",
        ]
    )
    return "\n".join(lines)


def run(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_root = _resolve(config["data"]["root"])
    output = config["output"]
    result_path = _resolve(output["json"])
    markdown_path = _resolve(output["markdown"])
    representation = dict(config["representation"])
    limits = config.get("limits", {})
    start = time.perf_counter()

    datasets = tuple(_load_zinc(data_root, split) for split in ("train", "val", "test"))
    graph_sets: list[list[Data]] = []
    split_metadata: list[dict[str, Any]] = []
    for split, dataset in zip(("train", "valid", "test"), datasets, strict=True):
        graphs, metadata = _load_data_split(dataset, limits.get(split), representation)
        graph_sets.append(graphs)
        split_metadata.append(metadata)
        print(f"loaded {split}: {metadata}", flush=True)

    device_name = str(config["model"].get("device", "cpu"))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(device_name)
    requested_modes = tuple(str(value) for value in config["model"].get("modes", ("conditional_fusion",)))
    control_modes = tuple(str(value) for value in config["model"].get("control_modes", ()))
    unknown_modes = sorted(
        set(requested_modes) - set(MODES)
        | set(control_modes) - {"conditional_fusion_shuffle"}
    )
    if unknown_modes:
        raise ValueError(f"unknown modes: {unknown_modes}")
    seeds = [int(value) for value in config["model"].get("model_seeds", [0])]
    all_modes = requested_modes + tuple(mode for mode in control_modes if mode not in requested_modes)
    mode_results: dict[str, list[dict[str, Any]]] = {mode: [] for mode in all_modes}
    for mode in requested_modes:
        for seed in seeds:
            print(f"starting mode={mode} seed={seed}", flush=True)
            mode_results[mode].append(
                _run_mode(
                    mode,
                    graph_sets[0],
                    graph_sets[1],
                    graph_sets[2],
                    config=config,
                    seed=seed,
                    device=device,
                )
            )
    if "conditional_fusion_shuffle" in control_modes:
        shuffle_seed = int(config["model"].get("shuffle_seed", 20260903))
        shuffled_sets = [
            _ShuffledCenterDataset(
                graph_sets[split_id],
                seed=shuffle_seed,
                keys=range(
                    sum(len(values) for values in graph_sets[:split_id]),
                    sum(len(values) for values in graph_sets[: split_id + 1]),
                ),
            )
            for split_id in range(3)
        ]
        for seed in seeds:
            print("starting mode=conditional_fusion_shuffle seed=" + str(seed), flush=True)
            mode_results["conditional_fusion_shuffle"].append(
                _run_mode(
                    "conditional_fusion",
                    shuffled_sets[0],
                    shuffled_sets[1],
                    shuffled_sets[2],
                    config=config,
                    seed=seed,
                    device=device,
                )
            )

    summarized: dict[str, Any] = {}
    for mode, rows in mode_results.items():
        valid = np.asarray([row["valid"]["mae"] for row in rows], dtype=np.float64)
        test = np.asarray([row["test_after_train_valid_refit"]["mae"] for row in rows], dtype=np.float64)
        epochs = np.asarray([row["valid"]["selected_epoch"] for row in rows], dtype=np.float64)
        summarized[mode] = {
            "seeds": rows,
            "valid": {
                "mean_mae": float(valid.mean()),
                "std_mae": float(valid.std()),
                "seed_values": valid.tolist(),
                "selected_epoch_mean": float(epochs.mean()),
            },
            "test_after_train_valid_refit": {
                "mean_mae": float(test.mean()),
                "std_mae": float(test.std()),
                "seed_values": test.tolist(),
            },
        }
        if mode == "conditional_fusion_shuffle":
            summarized[mode]["control"] = "within-graph permutation of centre attribute rows; topology and attribute bag preserved"
    result = {
        "protocol_id": str(config["protocol_id"]),
        "status": "completed",
        "data": {
            "root": str(data_root),
            "split": "PyG ZINC subset=True official train/val/test",
            "sizes": {name: len(values) for name, values in zip(("train", "valid", "test"), graph_sets, strict=True)},
            "source": source_audit(data_root),
            "test_used_for_selection": False,
        },
        "representation": {
            **representation,
            "topology_width": TOPOLOGY_WIDTH,
            "attribute_width": ATTRIBUTE_WIDTH,
            "relation_width": RELATION_WIDTH,
            "topology_definition": "untyped rooted-WL root role + topology role histograms + shell/size/cycle statistics",
            "attribute_definition": "center atom + incident bond histogram + local atom/bond histograms",
            "relation_definition": "directed centre pairs within relation radius with shortest distance, patch overlap, and adjacent bond type",
        },
        "feature_build": {name: metadata for name, metadata in zip(("train", "valid", "test"), split_metadata, strict=True)},
        "training": {
            **dict(config["model"]),
            "device": str(device),
            "loss": "L1 / mean absolute error",
            "readout": "sum + mean + std over centre states",
        },
        "modes": summarized,
        "runtime_seconds": float(time.perf_counter() - start),
        "script_sha256": _sha256(Path(__file__).resolve()),
    }
    _write_json_atomic(result_path, result)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_render_markdown(result), encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    result = run(_resolve(args.config))
    compact = {
        mode: {
            "valid": row["valid"]["mean_mae"],
            "test": row["test_after_train_valid_refit"]["mean_mae"],
            "epoch": row["valid"]["selected_epoch_mean"],
        }
        for mode, row in result["modes"].items()
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
