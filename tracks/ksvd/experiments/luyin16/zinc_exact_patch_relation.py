"""ZINC exact typed patches with direct, non-message-passing relations.

The experiment is intentionally a graph-level readout model rather than a
message-passing GNN.  For every atom centre it builds a radius-2 induced ego
graph and canonicalises its typed incidence graph with nauty.  The resulting
fixed-width descriptor contains the canonical node slots, shell/root/boundary
flags, and typed internal edges.

For one molecule, unary patch descriptors are pooled with sum/mean/max.  For
every unordered pair of centres, the runner computes explicit relation
statistics (shortest-path distance, patch overlap, adjacent bond type, and
descriptor shared bits).  Those pair rows are pooled with sum/mean/max as
well.  No centre state is updated from another centre and no attention or
message passing is used.

Two downstream heads consume the same frozen graph-level feature matrix:
one XGBoost regressor and one small MLP.  Feature construction is label-free;
the official validation split selects the MLP epoch and the XGBoost settings
are fixed in the configuration.  Test is evaluated once after train+valid
refitting.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import random
import sys
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
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import (
    REPO_ROOT,
    _data_to_graph,
    _load_zinc,
    _resolve,
    source_audit,
)


DEFAULT_CONFIG = REPO_ROOT / "tracks/ksvd/configs/luyin16/zinc_exact_patch_relation.yaml"

ATOM_CATEGORIES = 28
BOND_CATEGORIES = 4
RADIUS = 2
MAX_PATCH_NODES = 14
DISTANCE_BINS = 12

# Every occupied canonical node slot stores atom type, shell, boundary, root,
# and occupancy.  Edge slots store one of the four bond types; zero means no
# induced edge and is therefore represented by an all-zero pair block.
NODE_SLOT_WIDTH = ATOM_CATEGORIES + (RADIUS + 1) + 1 + 1 + 1
PAIR_SLOTS = MAX_PATCH_NODES * (MAX_PATCH_NODES - 1) // 2
PATCH_WIDTH = MAX_PATCH_NODES * NODE_SLOT_WIDTH + PAIR_SLOTS * BOND_CATEGORIES

# Pair scalar relation fields:
# distance one-hot, four overlap quantities, same root atom, exact patch
# equality, four adjacent-bond categories, cosine, and shared-bit density.
PAIR_SCALAR_WIDTH = DISTANCE_BINS + 4 + 1 + 1 + BOND_CATEGORIES + 1 + 1
RELATION_WIDTH = PATCH_WIDTH + PAIR_SCALAR_WIDTH
GRAPH_CONTEXT_WIDTH = 4
FEATURE_WIDTH = 3 * PATCH_WIDTH + 3 * PATCH_WIDTH + 3 * RELATION_WIDTH + GRAPH_CONTEXT_WIDTH


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


def _all_distances(graph: Any, source: int) -> dict[int, int]:
    distances = {int(source): 0}
    queue: deque[int] = deque([int(source)])
    while queue:
        node = queue.popleft()
        for neighbor in sorted(graph.neighbors(node)):
            if neighbor not in distances:
                distances[int(neighbor)] = distances[node] + 1
                queue.append(int(neighbor))
    return distances


def _canonical_typed_patch(
    graph: Any,
    center: int,
    node_types: np.ndarray,
    edge_types: Mapping[tuple[int, int], int],
    *,
    radius: int = RADIUS,
) -> tuple[np.ndarray, frozenset[int], bytes, dict[str, int]]:
    """Return an invariant fixed-width descriptor for one exact typed patch."""
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
    if n_nodes > MAX_PATCH_NODES:
        raise RuntimeError(
            f"radius-{radius} patch has {n_nodes} nodes, exceeding fixed width "
            f"MAX_PATCH_NODES={MAX_PATCH_NODES}"
        )

    # Replace each bond by an incidence vertex.  Node colors contain all typed
    # rooted-patch information needed for exact canonicalisation; bond colors
    # retain the edge type without relying on the input edge order.
    adjacency: dict[int, list[int]] = {
        vertex: [] for vertex in range(n_nodes + n_edges)
    }
    color_groups: dict[tuple[Any, ...], set[int]] = {}
    for local, node in enumerate(original_nodes):
        key = (
            "node",
            int(local == node_to_local[int(center)]),
            int(distances[node]),
            int(node_types[int(node)]),
        )
        color_groups.setdefault(key, set()).add(local)
    for edge_local, (left, right) in enumerate(local_edges):
        edge_vertex = n_nodes + edge_local
        adjacency[left].append(edge_vertex)
        adjacency[right].append(edge_vertex)
        adjacency[edge_vertex] = [left, right]
        bond_type = int(edge_types[graph.edge_key(int(original_nodes[left]), int(original_nodes[right]))])
        color_groups.setdefault(("edge", bond_type), set()).add(edge_vertex)

    coloring = [color_groups[key] for key in sorted(color_groups, key=repr)]
    incidence = pynauty.Graph(
        number_of_vertices=n_nodes + n_edges,
        directed=False,
        adjacency_dict=adjacency,
        vertex_coloring=coloring,
    )
    certificate = bytes(pynauty.certificate(incidence))
    canonical = tuple(int(value) for value in pynauty.canon_label(incidence))
    canonical_position = {vertex: position for position, vertex in enumerate(canonical)}
    root_local = node_to_local[int(center)]
    ordered_nodes = [root_local]
    ordered_nodes.extend(
        sorted(
            (local for local in range(n_nodes) if local != root_local),
            key=canonical_position.__getitem__,
        )
    )
    position = {local: slot for slot, local in enumerate(ordered_nodes)}

    descriptor = np.zeros(PATCH_WIDTH, dtype=np.float32)
    for slot, local in enumerate(ordered_nodes):
        atom = int(node_types[int(original_nodes[local])])
        shell = int(distances[int(original_nodes[local])])
        if atom < 0 or atom >= ATOM_CATEGORIES:
            raise ValueError(f"atom category {atom} outside [0,{ATOM_CATEGORIES})")
        if shell < 0 or shell > RADIUS:
            raise ValueError(f"shell {shell} outside [0,{RADIUS}]")
        base = slot * NODE_SLOT_WIDTH
        descriptor[base + atom] = 1.0
        descriptor[base + ATOM_CATEGORIES + shell] = 1.0
        descriptor[base + ATOM_CATEGORIES + RADIUS + 1] = float(shell == radius)
        descriptor[base + ATOM_CATEGORIES + RADIUS + 2] = float(local == root_local)
        descriptor[base + ATOM_CATEGORIES + RADIUS + 3] = 1.0

    edge_offset = MAX_PATCH_NODES * NODE_SLOT_WIDTH
    for edge_local, (left, right) in enumerate(local_edges):
        bond_type = int(edge_types[graph.edge_key(int(original_nodes[left]), int(original_nodes[right]))])
        if bond_type < 0 or bond_type >= BOND_CATEGORIES:
            raise ValueError(f"bond category {bond_type} outside [0,{BOND_CATEGORIES})")
        left_slot, right_slot = position[left], position[right]
        if left_slot > right_slot:
            left_slot, right_slot = right_slot, left_slot
        pair_index = 0
        for first in range(MAX_PATCH_NODES):
            for second in range(first + 1, MAX_PATCH_NODES):
                if first == left_slot and second == right_slot:
                    descriptor[edge_offset + pair_index * BOND_CATEGORIES + bond_type] = 1.0
                    break
                pair_index += 1

    if not np.isfinite(descriptor).all():
        raise FloatingPointError("non-finite exact patch descriptor")
    metadata = {
        "n_nodes": int(n_nodes),
        "n_edges": int(n_edges),
        "n_boundary_nodes": int(sum(distance == radius for distance in distances.values())),
    }
    return descriptor, frozenset(int(node) for node in original_nodes), certificate, metadata


def _patch_cache_key(
    graph: Any,
    center: int,
    node_types: np.ndarray,
    edge_types: Mapping[tuple[int, int], int],
    radius: int,
) -> bytes:
    """Raw-order key used only to avoid repeating identical canonicalisation."""
    distances = _ego_distances(graph, int(center), int(radius))
    nodes = tuple(sorted(distances))
    node_to_local = {node: index for index, node in enumerate(nodes)}
    induced = graph.induced(set(nodes))
    payload: list[Any] = [node_to_local[int(center)], radius]
    payload.extend((int(distances[node]), int(node_types[int(node)])) for node in nodes)
    for left, right in sorted(induced.edges()):
        bond = int(edge_types[graph.edge_key(int(left), int(right))])
        payload.append((node_to_local[int(left)], node_to_local[int(right)], bond))
    return repr(tuple(payload)).encode("ascii")


def _pair_relation_row(
    left_descriptor: np.ndarray,
    right_descriptor: np.ndarray,
    left_nodes: frozenset[int],
    right_nodes: frozenset[int],
    left_certificate: bytes,
    right_certificate: bytes,
    distance: int,
    adjacent_bond: int | None,
) -> np.ndarray:
    shared = left_descriptor * right_descriptor
    left_size = len(left_nodes)
    right_size = len(right_nodes)
    intersection = len(left_nodes & right_nodes)
    union = len(left_nodes | right_nodes)
    boundary_left = left_nodes  # boundary is encoded in the patch descriptor;
    boundary_right = right_nodes  # overlap itself is the invariant relation.
    # The two raw containment ratios plus Jaccard are intentionally retained;
    # they distinguish a small patch contained in a large one from two equal
    # sized patches with the same Jaccard overlap.
    overlap = np.asarray(
        [
            float(intersection) / MAX_PATCH_NODES,
            float(intersection) / max(float(union), 1.0),
            float(intersection) / max(float(min(left_size, right_size)), 1.0),
            float(intersection) / max(float(max(left_size, right_size)), 1.0),
        ],
        dtype=np.float32,
    )
    distance_one_hot = np.zeros(DISTANCE_BINS, dtype=np.float32)
    distance_one_hot[min(max(int(distance), 1), DISTANCE_BINS) - 1] = 1.0
    bond_one_hot = np.zeros(BOND_CATEGORIES, dtype=np.float32)
    if adjacent_bond is not None:
        if int(adjacent_bond) < 0 or int(adjacent_bond) >= BOND_CATEGORIES:
            raise ValueError(f"bond category {adjacent_bond} outside schema")
        bond_one_hot[int(adjacent_bond)] = 1.0
    denominator = float(np.linalg.norm(left_descriptor) * np.linalg.norm(right_descriptor))
    cosine = float(left_descriptor @ right_descriptor / denominator) if denominator > 1.0e-12 else 0.0
    scalar = np.concatenate(
        [
            distance_one_hot,
            overlap,
            np.asarray(
                [
                    float(left_descriptor[:NODE_SLOT_WIDTH][0:ATOM_CATEGORIES]
                          @ right_descriptor[:NODE_SLOT_WIDTH][0:ATOM_CATEGORIES] > 0.0),
                    float(left_certificate == right_certificate),
                ],
                dtype=np.float32,
            ),
            bond_one_hot,
            np.asarray(
                [cosine, float(shared.sum()) / max(float(PATCH_WIDTH), 1.0)],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32, copy=False)
    if scalar.shape != (PAIR_SCALAR_WIDTH,):
        raise RuntimeError(f"pair scalar width changed: {scalar.shape}")
    return np.concatenate([shared, scalar]).astype(np.float32, copy=False)


def _aggregate(values: np.ndarray, width: int) -> np.ndarray:
    if values.size == 0:
        return np.zeros(3 * int(width), dtype=np.float32)
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != int(width):
        raise ValueError(f"aggregate expects [n,{width}], got {matrix.shape}")
    return np.concatenate(
        [matrix.sum(axis=0), matrix.mean(axis=0), matrix.max(axis=0)]
    ).astype(np.float32, copy=False)


def _graph_feature_row(
    graph: Any,
    node_types: np.ndarray,
    edge_types: Mapping[tuple[int, int], int],
    *,
    patch_cache: dict[bytes, tuple[np.ndarray, frozenset[int], bytes, dict[str, int]]],
) -> tuple[np.ndarray, dict[str, Any]]:
    centers = list(graph.nodes)
    patches: list[tuple[np.ndarray, frozenset[int], bytes, dict[str, int]]] = []
    for center in centers:
        key = _patch_cache_key(graph, int(center), node_types, edge_types, RADIUS)
        patch = patch_cache.get(key)
        if patch is None:
            patch = _canonical_typed_patch(
                graph, int(center), node_types, edge_types, radius=RADIUS
            )
            patch_cache[key] = patch
        patches.append(patch)

    descriptors = np.stack([patch[0] for patch in patches], axis=0).astype(np.float32, copy=False)
    unary = _aggregate(descriptors, PATCH_WIDTH)
    shared_counts = descriptors.sum(axis=0, dtype=np.float64)
    n_pairs = len(centers) * (len(centers) - 1) // 2
    shared_sum = shared_counts * np.maximum(shared_counts - 1.0, 0.0) / 2.0
    shared_mean = shared_sum / max(float(n_pairs), 1.0)
    shared_max = (shared_counts >= 2.0).astype(np.float64)
    shared_readout = np.concatenate([shared_sum, shared_mean, shared_max]).astype(np.float32)

    distances = {center: _all_distances(graph, int(center)) for center in centers}
    pair_rows: list[np.ndarray] = []
    for left_index, left in enumerate(centers):
        for right_index in range(left_index + 1, len(centers)):
            right = centers[right_index]
            distance = int(distances[left][int(right)])
            adjacent_bond = None
            if distance == 1:
                adjacent_bond = int(edge_types[graph.edge_key(int(left), int(right))])
            pair_rows.append(
                _pair_relation_row(
                    patches[left_index][0],
                    patches[right_index][0],
                    patches[left_index][1],
                    patches[right_index][1],
                    patches[left_index][2],
                    patches[right_index][2],
                    distance,
                    adjacent_bond,
                )
            )
    pair_matrix = np.stack(pair_rows, axis=0).astype(np.float32, copy=False)
    relation = _aggregate(pair_matrix, RELATION_WIDTH)
    context = np.asarray(
        [
            np.log1p(float(graph.n)),
            np.log1p(float(graph.num_edges())),
            np.log1p(float(len(centers))),
            np.log1p(float(n_pairs)),
        ],
        dtype=np.float32,
    )
    features = np.concatenate([unary, shared_readout, relation, context]).astype(np.float32, copy=False)
    if features.shape != (FEATURE_WIDTH,):
        raise RuntimeError(f"feature width changed: {features.shape}; expected {(FEATURE_WIDTH,)}")
    if not np.isfinite(features).all():
        raise FloatingPointError("non-finite graph feature row")
    metadata = {
        "n_nodes": int(graph.n),
        "n_edges": int(graph.num_edges()),
        "n_centres": int(len(centers)),
        "n_pairs": int(n_pairs),
        "patch_nodes_mean": float(np.mean([patch[3]["n_nodes"] for patch in patches])),
        "patch_nodes_max": int(max(patch[3]["n_nodes"] for patch in patches)),
        "patch_edges_mean": float(np.mean([patch[3]["n_edges"] for patch in patches])),
        "unique_patch_certificates": int(len({patch[2] for patch in patches})),
    }
    return features, metadata


def _build_split(
    dataset: Any,
    *,
    split: str,
    patch_cache: dict[bytes, tuple[np.ndarray, frozenset[int], bytes, dict[str, int]]],
) -> tuple[np.ndarray, dict[str, Any]]:
    rows: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, data in enumerate(dataset):
        graph, node_types, edge_types = _data_to_graph(data)
        row, details = _graph_feature_row(
            graph, node_types, edge_types, patch_cache=patch_cache
        )
        rows.append(row)
        metadata.append(details)
        if index and index % 500 == 0:
            print(
                f"exact patch features {split}: {index}/{len(dataset)} "
                f"cache={len(patch_cache)}",
                flush=True,
            )
    matrix = np.stack(rows, axis=0).astype(np.float32, copy=False)
    summary = {
        "n_graphs": int(len(matrix)),
        "feature_width": int(matrix.shape[1]),
        "mean_nodes": float(np.mean([row["n_nodes"] for row in metadata])),
        "mean_pairs": float(np.mean([row["n_pairs"] for row in metadata])),
        "mean_patch_nodes": float(np.mean([row["patch_nodes_mean"] for row in metadata])),
        "max_patch_nodes": int(max(row["patch_nodes_max"] for row in metadata)),
        "mean_unique_patch_certificates": float(
            np.mean([row["unique_patch_certificates"] for row in metadata])
        ),
        "seconds": float(time.perf_counter() - started),
    }
    return matrix, summary


class Standardizer:
    def __init__(self, mean: np.ndarray, scale: np.ndarray) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        matrix = np.asarray(values, dtype=np.float32)
        mean = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = matrix.std(axis=0, dtype=np.float64).astype(np.float32)
        scale[~np.isfinite(scale) | (scale < 1.0e-6)] = 1.0
        return cls(mean, scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        output = ((np.asarray(values, dtype=np.float32) - self.mean) / self.scale).astype(
            np.float32, copy=False
        )
        if not np.isfinite(output).all():
            raise FloatingPointError("non-finite standardized features")
        return output


def _xgb_params(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    values = dict(config["xgboost"]["params"])
    values.update(
        {
            "objective": str(config["xgboost"].get("objective", "reg:absoluteerror")),
            "eval_metric": "mae",
            "tree_method": "hist",
            "max_bin": int(config["xgboost"].get("max_bin", 256)),
            "random_state": int(seed),
            "n_jobs": int(config["xgboost"].get("n_jobs", 4)),
        }
    )
    return values


def _fit_xgb(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    config: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    model = XGBRegressor(**_xgb_params(config, seed))
    model.fit(x_train, y_train)
    prediction = np.asarray(model.predict(x_eval), dtype=np.float64)
    return {"mae": float(mean_absolute_error(y_eval, prediction)), "prediction": prediction}


class GraphMLP(nn.Module):
    def __init__(self, width: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(int(width), int(hidden)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), int(hidden)),
            nn.ReLU(),
            nn.Linear(int(hidden), 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).view(-1)


def _mlp_predict(model: nn.Module, values: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(values).to(device)).cpu().numpy().astype(np.float64)


def _train_mlp_phase(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    config: Mapping[str, Any],
    seed: int,
    *,
    select_best: bool,
    epochs: int,
) -> dict[str, Any]:
    model_config = config["mlp"]
    device = torch.device(str(model_config.get("device", "cpu")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("MLP requested CUDA but CUDA is not available")
    _seed_everything(seed)
    model = GraphMLP(
        x_train.shape[1],
        int(model_config.get("hidden", 128)),
        float(model_config.get("dropout", 0.05)),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(model_config.get("learning_rate", 1.0e-3)),
        weight_decay=float(model_config.get("weight_decay", 1.0e-5)),
    )
    train_dataset = TensorDataset(
        torch.from_numpy(x_train), torch.from_numpy(np.asarray(y_train, dtype=np.float32))
    )
    loader = DataLoader(
        train_dataset,
        batch_size=int(model_config.get("batch_size", 128)),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(seed) + 91011),
        num_workers=0,
    )
    patience = int(model_config.get("patience", 12))
    best_mae = float("inf")
    best_epoch = 1
    best_state: dict[str, Any] | None = None
    trace: list[dict[str, float | int]] = []
    stale = 0
    losses: list[float] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        total = 0.0
        count = 0
        for batch_x, batch_y in loader:
            prediction = model(batch_x.to(device))
            loss = F.l1_loss(prediction, batch_y.to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(batch_y)
            count += len(batch_y)
        losses.append(total / max(count, 1))
        current_mae = None
        if select_best:
            prediction = _mlp_predict(model, x_eval, device)
            current_mae = float(mean_absolute_error(y_eval, prediction))
            trace.append({"epoch": int(epoch), "mae": current_mae})
            if current_mae < best_mae:
                best_mae = current_mae
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
        if epoch == 1 or epoch == int(epochs) or epoch % max(1, int(epochs) // 10) == 0:
            suffix = "" if current_mae is None else f" valid_mae={current_mae:.6f}"
            print(
                f"exact-patch MLP epoch={epoch:03d}/{epochs} "
                f"l1={losses[-1]:.6f}{suffix}",
                flush=True,
            )
        if select_best and stale >= patience:
            print(
                f"exact-patch MLP early_stop epoch={epoch} best_epoch={best_epoch}",
                flush=True,
            )
            break
    if select_best and best_state is not None:
        model.load_state_dict(best_state)
    prediction = _mlp_predict(model, x_eval, device)
    return {
        "model": model,
        "mae": float(mean_absolute_error(y_eval, prediction)),
        "best_mae": None if not select_best else float(best_mae),
        "selected_epoch": int(best_epoch if select_best else epochs),
        "epochs_run": int(len(losses)),
        "trace": trace,
        "losses": losses,
        "prediction": prediction,
        "device": str(device),
    }


def _render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        f"# {result['protocol_id']}",
        "",
        "Exact rooted typed radius-2 patches with direct explicit pair-relation readout; no message passing.",
        "",
        f"- split: `{result['data']['split']}`; sizes `{result['data']['sizes']}`",
        f"- descriptor width: `{result['representation']['patch_width']}`; graph feature width: `{result['representation']['feature_width']}`",
        f"- canonicalizer: `{result['representation']['canonicalizer']}`",
        f"- pair relation: `{result['representation']['pair_relation']}`",
        "",
        "| head | valid MAE | test MAE after train+valid refit | selected epoch |",
        "|---|---:|---:|---:|",
    ]
    for name, row in result["evaluation"].items():
        lines.append(
            f"| `{name}` | {row['valid']['mae']:.6f} | "
            f"{row['test_after_train_valid_refit']['mae']:.6f} | "
            f"{row.get('selected_epoch', 'fixed')} |"
        )
    lines.extend(
        [
            "",
            "The feature matrix is label-free and uses all centres and all unordered centre pairs. "
            "For each patch and relation block, sum/mean/max are concatenated; the heads are the only predictors.",
            "",
            f"- feature build cache: `{result['feature_cache']['path']}`; cache hit: `{result['feature_cache']['cache_hit']}`",
            f"- runtime: `{result['runtime']['seconds']:.1f}s`",
            "",
        ]
    )
    return "\n".join(lines)


def run(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_root = _resolve(config["data"]["root"])
    feature_cache = _resolve(config["output"]["feature_cache"])
    result_json = _resolve(config["output"]["json"])
    result_markdown = _resolve(config["output"]["markdown"])
    seed = int(config.get("seed", 0))
    started = time.perf_counter()
    datasets = tuple(_load_zinc(data_root, split) for split in ("train", "val", "test"))
    labels = tuple(
        np.asarray([float(data.y.view(-1)[0]) for data in dataset], dtype=np.float32)
        for dataset in datasets
    )

    signature_payload = {
        "protocol_id": config["protocol_id"],
        "radius": RADIUS,
        "max_patch_nodes": MAX_PATCH_NODES,
        "atom_categories": ATOM_CATEGORIES,
        "bond_categories": BOND_CATEGORIES,
        "distance_bins": DISTANCE_BINS,
        "script_sha256": _sha256(Path(__file__).resolve()),
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    cache_hit = False
    feature_metadata: dict[str, Any]
    if feature_cache.exists():
        try:
            cached = np.load(feature_cache, allow_pickle=False)
            cached_signature = str(cached["signature"].item())
            if cached_signature == signature:
                features = [
                    np.asarray(cached[split], dtype=np.float32)
                    for split in ("train", "valid", "test")
                ]
                feature_metadata = json.loads(str(cached["metadata"].item()))
                cache_hit = True
                print(f"feature cache hit: {feature_cache}", flush=True)
            else:
                cached.close()
                features = []
                feature_metadata = {}
        except Exception as exc:
            print(f"feature cache ignored: {exc!r}", flush=True)
            features = []
            feature_metadata = {}
    else:
        features = []
        feature_metadata = {}

    if not cache_hit:
        patch_cache: dict[bytes, tuple[np.ndarray, frozenset[int], bytes, dict[str, int]]] = {}
        features = []
        feature_metadata = {}
        for split_name, dataset in zip(("train", "valid", "test"), datasets, strict=True):
            matrix, metadata = _build_split(
                dataset, split=split_name, patch_cache=patch_cache
            )
            features.append(matrix)
            feature_metadata[split_name] = metadata
        feature_metadata["raw_patch_cache_entries"] = int(len(patch_cache))
        feature_metadata["signature_payload"] = signature_payload
        feature_cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = feature_cache.with_suffix(feature_cache.suffix + ".tmp.npz")
        np.savez_compressed(
            temporary,
            signature=np.asarray(signature),
            metadata=np.asarray(json.dumps(_jsonable(feature_metadata), sort_keys=True)),
            train=features[0],
            valid=features[1],
            test=features[2],
        )
        temporary.replace(feature_cache)

    train_x, valid_x, test_x = features
    train_y, valid_y, test_y = labels
    model_seed = seed

    xgb_valid = _fit_xgb(train_x, train_y, valid_x, valid_y, config, model_seed)
    xgb_test = _fit_xgb(
        np.concatenate([train_x, valid_x]),
        np.concatenate([train_y, valid_y]),
        test_x,
        test_y,
        config,
        model_seed,
    )

    standardizer = Standardizer.fit(train_x)
    train_x_std = standardizer.transform(train_x)
    valid_x_std = standardizer.transform(valid_x)
    test_x_std = standardizer.transform(test_x)
    mlp_valid = _train_mlp_phase(
        train_x_std,
        train_y,
        valid_x_std,
        valid_y,
        config,
        model_seed,
        select_best=True,
        epochs=int(config["mlp"].get("epochs", 50)),
    )
    selected_epoch = int(mlp_valid["selected_epoch"])
    mlp_test = _train_mlp_phase(
        np.concatenate([train_x_std, valid_x_std]),
        np.concatenate([train_y, valid_y]),
        test_x_std,
        test_y,
        config,
        model_seed,
        select_best=False,
        epochs=selected_epoch,
    )

    result = {
        "protocol_id": str(config["protocol_id"]),
        "status": "completed",
        "seed": model_seed,
        "data": {
            "root": str(data_root),
            "split": "PyG ZINC subset=True official train/val/test",
            "sizes": {
                name: len(dataset)
                for name, dataset in zip(("train", "valid", "test"), datasets, strict=True)
            },
            "source": source_audit(data_root),
            "test_used_for_feature_selection": False,
        },
        "representation": {
            "radius": RADIUS,
            "centres": "every atom",
            "patch_definition": "induced rooted ego graph with atom types, shell/root/boundary flags, and typed internal bonds",
            "canonicalizer": "pynauty 2.8.8.1 colored incidence-graph certificate + canonical label",
            "max_patch_nodes": MAX_PATCH_NODES,
            "node_slot_width": NODE_SLOT_WIDTH,
            "pair_slots": PAIR_SLOTS,
            "patch_width": PATCH_WIDTH,
            "pair_scalar_width": PAIR_SCALAR_WIDTH,
            "relation_width": RELATION_WIDTH,
            "feature_width": FEATURE_WIDTH,
            "pair_relation": "unordered centre pairs: distance, four overlap ratios, root-atom equality, exact typed-patch equality, adjacent bond type, cosine and shared descriptor bits",
            "readout": "sum + mean + max for unary patch descriptors, shared pair bits, and explicit pair relation rows",
            "message_passing": False,
            "attention": False,
        },
        "feature_build": feature_metadata,
        "feature_cache": {
            "path": str(feature_cache),
            "signature": signature,
            "sha256": _sha256(feature_cache),
            "cache_hit": cache_hit,
        },
        "heads": {
            "xgboost": _xgb_params(config, model_seed),
            "mlp": {
                **dict(config["mlp"]),
                "standardization": "train-only mean/std",
                "loss": "L1 / mean absolute error",
            },
        },
        "evaluation": {
            "xgboost": {
                "valid": {"mae": xgb_valid["mae"]},
                "test_after_train_valid_refit": {"mae": xgb_test["mae"]},
                "selected_epoch": "fixed n_estimators",
            },
            "mlp": {
                "valid": {
                    "mae": mlp_valid["mae"],
                    "best_mae": mlp_valid["best_mae"],
                    "trace": mlp_valid["trace"],
                },
                "test_after_train_valid_refit": {"mae": mlp_test["mae"]},
                "selected_epoch": selected_epoch,
            },
        },
        "runtime": {
            "seconds": float(time.perf_counter() - started),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": importlib.metadata.version("numpy"),
            "torch": importlib.metadata.version("torch"),
            "xgboost": importlib.metadata.version("xgboost"),
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
                name: {
                    "valid_mae": row["valid"]["mae"],
                    "test_mae": row["test_after_train_valid_refit"]["mae"],
                    "selected_epoch": row["selected_epoch"],
                }
                for name, row in result["evaluation"].items()
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
