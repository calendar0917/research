"""ZINC direct-structure binding and cross-centre covariance.

This is the non-WL implementation of the luyin16 interaction proposal.  Every
atom is a centre and defines one induced radius-``r`` ego patch.  The patch
object is fixed before any statistics are computed:

* ``T`` is topology-only: node shell, induced-patch degree and cycle
  membership; edge shell-pair and cycle membership;
* ``A`` is semantic: ZINC atom and bond categories;
* within a patch, ``binding = P(T,A) - P(T)P(A)`` is computed separately for
  nodes and edges;
* across centres, ``cross_cov = Cov_v(T_v, A_v)`` is computed from the
  topology and attribute rows of those same patches.

The downstream views are ``S``, ``S + direct-marginal``, and the latter with
train-only PCA projections of ``cross_cov`` and/or ``binding`` appended.  PCA
is refit inside every training fold during XGBoost tuning, fit on official
train for official validation, and refit on train+validation for official
test.  A single XGBoost model seed is used; official test is evaluated only
after parameters are frozen by training-fold CV.
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import optuna
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
import yaml

from ksvd_research.core import from_edges
from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import (
    REPO_ROOT,
    _data_to_graph,
    _load_zinc,
    _resolve,
    global_feature_views,
    source_audit,
)


DEFAULT_CONFIG = REPO_ROOT / "tracks/ksvd/configs/luyin16/zinc_direct_binding_cross_cov.yaml"
VIEW_NAMES = ("s", "s_marginal", "s_cross_cov", "s_binding", "s_both")
SEARCH_KEYS = (
    "n_estimators",
    "max_depth",
    "learning_rate",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
    "reg_lambda",
    "reg_alpha",
    "gamma",
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


def _one_hot(value: int, width: int) -> np.ndarray:
    result = np.zeros(int(width), dtype=np.float32)
    if not 0 <= int(value) < int(width):
        raise ValueError(f"category {value} outside [0, {int(width) - 1}]")
    result[int(value)] = 1.0
    return result


def _histogram(values: Sequence[int], width: int) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.int64)
    if array.size and (array.min() < 0 or array.max() >= int(width)):
        raise ValueError(f"category outside [0, {int(width) - 1}]: {array.tolist()}")
    result = np.bincount(array, minlength=int(width)).astype(np.float32, copy=False)
    return result / max(float(result.sum()), 1.0)


def _ego_distances(graph: Any, centre: int, radius: int) -> dict[int, int]:
    distances = {int(centre): 0}
    queue: deque[int] = deque([int(centre)])
    while queue:
        node = queue.popleft()
        if distances[node] >= int(radius):
            continue
        for neighbour in sorted(graph.neighbors(node)):
            if neighbour not in distances:
                distances[int(neighbour)] = distances[node] + 1
                queue.append(int(neighbour))
    return distances


def _cycle_membership(graph: Any) -> tuple[set[int], set[tuple[int, int]]]:
    """Return vertices and edges that are not bridges in an induced patch."""
    discovery: dict[int, int] = {}
    low: dict[int, int] = {}
    bridges: set[tuple[int, int]] = set()
    clock = 0

    def visit(node: int, parent: int | None) -> None:
        nonlocal clock
        clock += 1
        discovery[node] = clock
        low[node] = clock
        for neighbour in sorted(graph.neighbors(node)):
            if neighbour == parent:
                continue
            edge = graph.edge_key(node, neighbour)
            if neighbour not in discovery:
                visit(int(neighbour), int(node))
                low[node] = min(low[node], low[neighbour])
                if low[neighbour] > discovery[node]:
                    bridges.add(edge)
            else:
                low[node] = min(low[node], discovery[neighbour])

    for node in graph.nodes:
        if node not in discovery:
            visit(int(node), None)

    cycle_edges = {
        graph.edge_key(left, right)
        for left, right in graph.edges()
        if graph.edge_key(left, right) not in bridges
    }
    cycle_nodes: set[int] = set()
    for left, right in cycle_edges:
        cycle_nodes.update((int(left), int(right)))
    return cycle_nodes, cycle_edges


def _shell_pairs(radius: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (left, right)
        for left in range(int(radius) + 1)
        for right in range(left, int(radius) + 1)
    )


def role_dimensions(representation: Mapping[str, Any]) -> dict[str, int]:
    radius = int(representation["radius"])
    degree_bins = int(representation["degree_bins"])
    node_roles = (radius + 1) * degree_bins * 2
    edge_roles = len(_shell_pairs(radius)) * 2
    atom_categories = int(representation.get("atom_categories", 28))
    bond_categories = int(representation.get("bond_categories", 4))
    topology_row = node_roles + node_roles + edge_roles
    attribute_row = atom_categories + atom_categories + bond_categories + bond_categories
    node_binding = node_roles * atom_categories
    edge_binding = edge_roles * bond_categories
    return {
        "node_role": node_roles,
        "edge_role": edge_roles,
        "topology_row": topology_row,
        "attribute_row": attribute_row,
        "node_attribute": atom_categories,
        "edge_attribute": bond_categories,
        "node_binding": node_binding,
        "edge_binding": edge_binding,
        "binding_raw": node_binding + edge_binding,
        "cross_cov_raw": topology_row * attribute_row,
        "context": 5,
    }


def _node_role_id(
    shell: int,
    induced_degree: int,
    on_cycle: bool,
    radius: int,
    degree_bins: int,
) -> int:
    if not 0 <= int(shell) <= int(radius):
        raise ValueError(f"shell {shell} outside radius {radius}")
    degree = min(max(int(induced_degree), 0), int(degree_bins) - 1)
    return ((int(shell) * int(degree_bins) + degree) * 2) + int(bool(on_cycle))


def _edge_role_id(
    left_shell: int,
    right_shell: int,
    on_cycle: bool,
    radius: int,
) -> int:
    pair = tuple(sorted((int(left_shell), int(right_shell))))
    try:
        pair_index = _shell_pairs(int(radius)).index(pair)
    except ValueError as exc:
        raise ValueError(f"edge shell pair {pair} outside radius {radius}") from exc
    return pair_index * 2 + int(bool(on_cycle))


def direct_patch_roles(
    graph: Any,
    centre: int,
    representation: Mapping[str, Any],
) -> tuple[tuple[int, ...], np.ndarray, tuple[tuple[int, int], ...], np.ndarray]:
    """Return invariant topology roles for one induced radius patch.

    No atom or bond attribute is read in this function.  The only role inputs
    are graph distance from the centre, induced-patch degree, and bridge/cycle
    membership.
    """
    radius = int(representation["radius"])
    degree_bins = int(representation["degree_bins"])
    distances = _ego_distances(graph, int(centre), radius)
    nodes = tuple(sorted(distances))
    induced = graph.induced(set(nodes))
    cycle_nodes, cycle_edges = _cycle_membership(induced)
    node_roles = np.asarray(
        [
            _node_role_id(
                distances[node],
                len(induced.neighbors(node)),
                node in cycle_nodes,
                radius,
                degree_bins,
            )
            for node in nodes
        ],
        dtype=np.int64,
    )
    edges = tuple(sorted(induced.edges()))
    edge_roles = np.asarray(
        [
            _edge_role_id(
                distances[left],
                distances[right],
                graph.edge_key(left, right) in cycle_edges,
                radius,
            )
            for left, right in edges
        ],
        dtype=np.int64,
    )
    return nodes, node_roles, edges, edge_roles


def _entity_statistics(
    roles: np.ndarray,
    attributes: np.ndarray,
    n_roles: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return P(T), P(A), and P(T,A) for one patch entity population."""
    role_values = np.asarray(roles, dtype=np.int64)
    attribute_values = np.asarray(attributes, dtype=np.float32)
    if (
        role_values.ndim != 1
        or attribute_values.ndim != 2
        or role_values.shape[0] != attribute_values.shape[0]
    ):
        raise ValueError("roles and attributes are not aligned")
    if role_values.size and (
        role_values.min() < 0 or role_values.max() >= int(n_roles)
    ):
        raise ValueError("role id outside declared role width")
    count = int(role_values.shape[0])
    if count == 0:
        return (
            np.zeros(int(n_roles), dtype=np.float32),
            np.zeros(attribute_values.shape[1], dtype=np.float32),
            np.zeros((int(n_roles), attribute_values.shape[1]), dtype=np.float32),
        )
    role_marginal = np.bincount(role_values, minlength=int(n_roles)).astype(
        np.float32, copy=False
    ) / float(count)
    attribute_marginal = attribute_values.mean(axis=0, dtype=np.float32)
    joint = np.zeros((int(n_roles), attribute_values.shape[1]), dtype=np.float32)
    np.add.at(joint, role_values, attribute_values)
    joint /= float(count)
    return role_marginal, attribute_marginal, joint


def _cross_covariance(topology_rows: np.ndarray, attribute_rows: np.ndarray) -> np.ndarray:
    """Return Cov_v(T_v, A_v), flattened row-major."""
    topology = np.asarray(topology_rows, dtype=np.float32)
    attributes = np.asarray(attribute_rows, dtype=np.float32)
    if topology.ndim != 2 or attributes.ndim != 2:
        raise ValueError("cross-covariance inputs must be two-dimensional")
    if topology.shape[0] != attributes.shape[0]:
        raise ValueError(f"unaligned centre rows: {topology.shape} vs {attributes.shape}")
    if topology.shape[0] <= 1:
        return np.zeros(topology.shape[1] * attributes.shape[1], dtype=np.float32)
    topology_centered = topology - topology.mean(axis=0, keepdims=True)
    attributes_centered = attributes - attributes.mean(axis=0, keepdims=True)
    return (
        (topology_centered.T @ attributes_centered) / float(topology.shape[0])
    ).reshape(-1).astype(np.float32, copy=False)


def _direct_graph_features_from_parts(
    graph: Any,
    node_types: np.ndarray,
    edge_types: Mapping[tuple[int, int], int],
    representation: Mapping[str, Any],
) -> dict[str, np.ndarray | float | int]:
    dims = role_dimensions(representation)
    atom_categories = int(representation.get("atom_categories", 28))
    bond_categories = int(representation.get("bond_categories", 4))
    node_values = np.asarray(node_types, dtype=np.int64).reshape(-1)
    if node_values.shape[0] != graph.n:
        raise ValueError("node feature count does not match graph")
    if node_values.size and (
        node_values.min() < 0 or node_values.max() >= atom_categories
    ):
        raise ValueError("ZINC atom category outside the declared schema")
    if edge_types and (
        min(int(value) for value in edge_types.values()) < 0
        or max(int(value) for value in edge_types.values()) >= bond_categories
    ):
        raise ValueError("ZINC bond category outside the declared schema")

    topology_rows: list[np.ndarray] = []
    attribute_rows: list[np.ndarray] = []
    binding_rows: list[np.ndarray] = []
    node_counts: list[int] = []
    edge_counts: list[int] = []
    for centre in graph.nodes:
        nodes, node_roles, edges, edge_roles = direct_patch_roles(
            graph, int(centre), representation
        )
        induced = graph.induced(set(nodes))
        root_position = nodes.index(int(centre))
        topology = np.concatenate(
            [
                _one_hot(int(node_roles[root_position]), dims["node_role"]),
                _histogram(node_roles.tolist(), dims["node_role"]),
                _histogram(edge_roles.tolist(), dims["edge_role"]),
            ]
        ).astype(np.float32, copy=False)

        local_atom_types = [int(node_values[node]) for node in nodes]
        local_bond_types = [
            int(edge_types[graph.edge_key(left, right)]) for left, right in edges
        ]
        incident_bond_types = [
            int(edge_types[graph.edge_key(int(centre), neighbour)])
            for neighbour in graph.neighbors(int(centre))
        ]
        attributes = np.concatenate(
            [
                _one_hot(int(node_values[int(centre)]), atom_categories),
                _histogram(local_atom_types, atom_categories),
                _histogram(incident_bond_types, bond_categories),
                _histogram(local_bond_types, bond_categories),
            ]
        ).astype(np.float32, copy=False)

        node_attributes = np.stack(
            [_one_hot(int(node_values[node]), atom_categories) for node in nodes],
            axis=0,
        ).astype(np.float32, copy=False)
        if edges:
            edge_attributes = np.stack(
                [
                    _one_hot(int(edge_types[graph.edge_key(left, right)]), bond_categories)
                    for left, right in edges
                ],
                axis=0,
            ).astype(np.float32, copy=False)
        else:
            edge_attributes = np.zeros((0, bond_categories), dtype=np.float32)
        node_role, node_attribute, node_joint = _entity_statistics(
            node_roles, node_attributes, dims["node_role"]
        )
        edge_role, edge_attribute, edge_joint = _entity_statistics(
            edge_roles, edge_attributes, dims["edge_role"]
        )
        node_binding = node_joint - node_role[:, None] * node_attribute[None, :]
        edge_binding = edge_joint - edge_role[:, None] * edge_attribute[None, :]
        binding = np.concatenate([node_binding.reshape(-1), edge_binding.reshape(-1)]).astype(
            np.float32, copy=False
        )

        if topology.shape != (dims["topology_row"],):
            raise RuntimeError(f"direct topology row width changed: {topology.shape}")
        if attributes.shape != (dims["attribute_row"],):
            raise RuntimeError(f"direct attribute row width changed: {attributes.shape}")
        if binding.shape != (dims["binding_raw"],):
            raise RuntimeError(f"direct binding row width changed: {binding.shape}")
        topology_rows.append(topology)
        attribute_rows.append(attributes)
        binding_rows.append(binding)
        node_counts.append(len(nodes))
        edge_counts.append(len(edges))
        # ``induced`` is intentionally constructed above as an explicit audit
        # that the patch edge set is the induced edge set; it also guards
        # against accidental use of edges outside the fixed patch.
        if induced.num_edges() != len(edges):
            raise RuntimeError("patch edge list is not induced")

    topology_matrix = np.stack(topology_rows, axis=0).astype(np.float32, copy=False)
    attribute_matrix = np.stack(attribute_rows, axis=0).astype(np.float32, copy=False)
    binding_matrix = np.stack(binding_rows, axis=0).astype(np.float32, copy=False)
    marginal = np.concatenate(
        [
            topology_matrix.mean(axis=0),
            topology_matrix.std(axis=0),
            attribute_matrix.mean(axis=0),
            attribute_matrix.std(axis=0),
        ]
    ).astype(np.float32, copy=False)
    context = np.asarray(
        [
            np.log1p(float(graph.n)),
            np.log1p(float(graph.num_edges())),
            np.log1p(float(len(topology_rows))),
            float(np.mean(node_counts)) if node_counts else 0.0,
            float(np.mean(edge_counts)) if edge_counts else 0.0,
        ],
        dtype=np.float32,
    )
    binding = np.concatenate(
        [binding_matrix.mean(axis=0), binding_matrix.std(axis=0)]
    ).astype(np.float32, copy=False)
    cross_cov = _cross_covariance(topology_matrix, attribute_matrix)
    if marginal.shape[0] != 2 * (dims["topology_row"] + dims["attribute_row"]):
        raise RuntimeError("direct marginal width changed")
    if cross_cov.shape != (dims["cross_cov_raw"],):
        raise RuntimeError("direct cross-covariance width changed")
    if binding.shape != (2 * dims["binding_raw"],):
        raise RuntimeError("direct binding width changed")
    return {
        "marginal": marginal,
        "context": context,
        "cross_cov": cross_cov,
        "binding": binding,
        "n_centres": int(topology_matrix.shape[0]),
        "mean_ego_nodes": float(np.mean(node_counts)) if node_counts else 0.0,
        "mean_ego_edges": float(np.mean(edge_counts)) if edge_counts else 0.0,
    }


def direct_graph_features(data: Any, representation: Mapping[str, Any]) -> dict[str, np.ndarray | float | int]:
    graph, node_types, edge_types = _data_to_graph(data)
    return _direct_graph_features_from_parts(graph, node_types, edge_types, representation)


def _relabel_parts(
    graph: Any,
    node_features: np.ndarray,
    edge_features: Mapping[tuple[int, int], int],
    permutation: np.ndarray,
) -> tuple[Any, np.ndarray, dict[tuple[int, int], int]]:
    permutation = np.asarray(permutation, dtype=np.int64)
    if sorted(permutation.tolist()) != list(range(graph.n)):
        raise ValueError("permutation must contain every node id exactly once")
    changed_graph = from_edges(
        graph.n,
        [
            (int(permutation[left]), int(permutation[right]))
            for left, right in graph.edges()
        ],
    )
    changed_nodes = np.empty_like(node_features)
    changed_nodes[permutation] = node_features
    changed_edges: dict[tuple[int, int], int] = {}
    for (left, right), value in edge_features.items():
        mapped_left, mapped_right = int(permutation[left]), int(permutation[right])
        changed_edges[changed_graph.edge_key(mapped_left, mapped_right)] = int(value)
    return changed_graph, changed_nodes, changed_edges


def audit_invariance(
    dataset: Any,
    representation: Mapping[str, Any],
    *,
    n_graphs: int,
    seed: int,
    tolerance: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    candidates = np.arange(len(dataset), dtype=np.int64)
    if candidates.size > int(n_graphs):
        candidates = rng.choice(candidates, size=int(n_graphs), replace=False)
    maxima = {name: 0.0 for name in ("marginal", "context", "cross_cov", "binding")}
    for raw_index in candidates:
        graph, node_types, edge_types = _data_to_graph(dataset[int(raw_index)])
        base = _direct_graph_features_from_parts(graph, node_types, edge_types, representation)
        changed_graph, changed_nodes, changed_edges = _relabel_parts(
            graph, node_types, edge_types, rng.permutation(graph.n)
        )
        changed = _direct_graph_features_from_parts(
            changed_graph, changed_nodes, changed_edges, representation
        )
        for name in maxima:
            maxima[name] = max(
                maxima[name],
                float(np.max(np.abs(np.asarray(base[name]) - np.asarray(changed[name])))),
            )
    return {
        "pass": bool(all(value <= float(tolerance) for value in maxima.values())),
        "n_graphs": int(candidates.size),
        "tolerance": float(tolerance),
        "maximum_drift": maxima,
    }


def _feature_signature(config: Mapping[str, Any], data_root: Path) -> str:
    payload = {
        "schema": "zinc_direct_structure_binding_cross_cov_v1",
        "representation": _jsonable(config["representation"]),
        "data_root": str(data_root.resolve()),
        "encoder_sha256": _sha256(Path(__file__).resolve()),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _build_or_load_features(
    datasets: Sequence[Any],
    config: Mapping[str, Any],
    data_root: Path,
    cache_path: Path,
) -> tuple[dict[str, np.ndarray], bool, dict[str, Any]]:
    signature = _feature_signature(config, data_root)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as archive:
            cached_signature = str(np.asarray(archive["signature"]).reshape(-1)[0])
            if cached_signature != signature:
                raise ValueError(f"direct feature cache signature mismatch: {cache_path}")
            arrays = {
                name: np.asarray(archive[name])
                for name in archive.files
                if name not in {"signature", "metadata"}
            }
            metadata = json.loads(str(np.asarray(archive["metadata"]).reshape(-1)[0]))
        return arrays, True, metadata

    representation = config["representation"]
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {"splits": {}, "signature": signature}
    for split_name, dataset in zip(("train", "valid", "test"), datasets, strict=True):
        global_values = global_feature_views(dataset)["global_all"]
        marginal_rows: list[np.ndarray] = []
        context_rows: list[np.ndarray] = []
        cross_rows: list[np.ndarray] = []
        binding_rows: list[np.ndarray] = []
        graph_meta: list[dict[str, float | int]] = []
        for position, data in enumerate(dataset):
            features = direct_graph_features(data, representation)
            marginal_rows.append(np.asarray(features["marginal"], dtype=np.float32))
            context_rows.append(np.asarray(features["context"], dtype=np.float32))
            cross_rows.append(np.asarray(features["cross_cov"], dtype=np.float32))
            binding_rows.append(np.asarray(features["binding"], dtype=np.float32))
            graph_meta.append(
                {
                    "n_centres": int(features["n_centres"]),
                    "mean_ego_nodes": float(features["mean_ego_nodes"]),
                    "mean_ego_edges": float(features["mean_ego_edges"]),
                }
            )
            if position and position % 500 == 0:
                print(f"[{split_name}] direct interaction graphs: {position}/{len(dataset)}", flush=True)
        arrays[f"{split_name}_s"] = np.asarray(global_values, dtype=np.float32)
        arrays[f"{split_name}_marginal"] = np.stack(marginal_rows).astype(np.float32, copy=False)
        arrays[f"{split_name}_context"] = np.stack(context_rows).astype(np.float32, copy=False)
        arrays[f"{split_name}_cross_cov"] = np.stack(cross_rows).astype(np.float32, copy=False)
        arrays[f"{split_name}_binding"] = np.stack(binding_rows).astype(np.float32, copy=False)
        metadata["splits"][split_name] = {
            "n_graphs": int(len(dataset)),
            "mean_centres": float(np.mean([row["n_centres"] for row in graph_meta])),
            "mean_ego_nodes": float(np.mean([row["mean_ego_nodes"] for row in graph_meta])),
            "mean_ego_edges": float(np.mean([row["mean_ego_edges"] for row in graph_meta])),
        }
        print(f"[{split_name}] direct interaction features complete", flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            signature=np.asarray([signature]),
            metadata=np.asarray([json.dumps(_jsonable(metadata), sort_keys=True)]),
            **arrays,
        )
    temporary.replace(cache_path)
    return arrays, False, metadata


def _raw_split(arrays: Mapping[str, np.ndarray], split: str) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(arrays[f"{split}_{name}"], dtype=np.float32)
        for name in ("s", "marginal", "context", "cross_cov", "binding")
    }


def _slice_raw(raw: Mapping[str, np.ndarray], rows: np.ndarray) -> dict[str, np.ndarray]:
    indices = np.asarray(rows, dtype=np.int64)
    return {name: np.asarray(value[indices], dtype=np.float32) for name, value in raw.items()}


def _concat_raw(left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    if set(left) != set(right):
        raise ValueError("raw feature blocks differ")
    return {
        name: np.concatenate([np.asarray(left[name]), np.asarray(right[name])], axis=0).astype(
            np.float32, copy=False
        )
        for name in left
    }


def _base_matrix(raw: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([raw["s"], raw["marginal"], raw["context"]], axis=1).astype(
        np.float32, copy=False
    )


def _fit_projection_model(
    train: np.ndarray,
    rank: int,
    seed: int,
) -> tuple[PCA | None, np.ndarray, dict[str, Any]]:
    values = np.asarray(train, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"PCA input must be two-dimensional, got {values.shape}")
    components = min(int(rank), values.shape[0], values.shape[1])
    if components <= 0:
        return None, np.zeros((values.shape[0], 0), dtype=np.float32), {
            "n_components": 0,
            "explained_variance_ratio_sum": 0.0,
            "fit_rows": int(values.shape[0]),
            "raw_width": int(values.shape[1]),
        }
    model = PCA(n_components=components, svd_solver="randomized", random_state=int(seed))
    projected = model.fit_transform(values).astype(np.float32, copy=False)
    return model, projected, {
        "n_components": int(components),
        "explained_variance_ratio_sum": float(np.sum(model.explained_variance_ratio_)),
        "fit_rows": int(values.shape[0]),
        "raw_width": int(values.shape[1]),
    }


def _transform_projection(model: PCA | None, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if model is None:
        return np.zeros((array.shape[0], 0), dtype=np.float32)
    return model.transform(array).astype(np.float32, copy=False)


def _scope_views(
    train_raw: Mapping[str, np.ndarray],
    eval_raw: Mapping[str, np.ndarray],
    pca_rank: int,
    pca_seed: int,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    train_base = _base_matrix(train_raw)
    eval_base = _base_matrix(eval_raw)
    cross_model, cross_train, cross_meta = _fit_projection_model(
        train_raw["cross_cov"], pca_rank, pca_seed
    )
    binding_model, binding_train, binding_meta = _fit_projection_model(
        train_raw["binding"], pca_rank, pca_seed
    )
    cross_eval = _transform_projection(cross_model, eval_raw["cross_cov"])
    binding_eval = _transform_projection(binding_model, eval_raw["binding"])
    views = {
        "s": (train_raw["s"], eval_raw["s"]),
        "s_marginal": (train_base, eval_base),
        "s_cross_cov": (
            np.concatenate([train_base, cross_train], axis=1).astype(np.float32, copy=False),
            np.concatenate([eval_base, cross_eval], axis=1).astype(np.float32, copy=False),
        ),
        "s_binding": (
            np.concatenate([train_base, binding_train], axis=1).astype(np.float32, copy=False),
            np.concatenate([eval_base, binding_eval], axis=1).astype(np.float32, copy=False),
        ),
        "s_both": (
            np.concatenate([train_base, cross_train, binding_train], axis=1).astype(
                np.float32, copy=False
            ),
            np.concatenate([eval_base, cross_eval, binding_eval], axis=1).astype(
                np.float32, copy=False
            ),
        ),
    }
    return views, {"cross_cov": cross_meta, "binding": binding_meta}


def _build_cv_views(
    train_raw: Mapping[str, np.ndarray],
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    pca_rank: int,
    pca_seed: int,
) -> tuple[list[dict[str, tuple[np.ndarray, np.ndarray]]], list[dict[str, Any]]]:
    all_views: list[dict[str, tuple[np.ndarray, np.ndarray]]] = []
    metadata: list[dict[str, Any]] = []
    for fold_id, (train_rows, valid_rows) in enumerate(folds):
        views, projection = _scope_views(
            _slice_raw(train_raw, train_rows),
            _slice_raw(train_raw, valid_rows),
            pca_rank,
            pca_seed,
        )
        all_views.append(views)
        metadata.append(
            {
                "fold": int(fold_id),
                "n_train": int(len(train_rows)),
                "n_valid": int(len(valid_rows)),
                "projection": projection,
            }
        )
        print(f"CV PCA fold {fold_id} complete", flush=True)
    return all_views, metadata


def _params_from_trial(
    trial: optuna.Trial,
    ranges: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int(
            "n_estimators", int(ranges["n_estimators"][0]), int(ranges["n_estimators"][1])
        ),
        "max_depth": trial.suggest_int(
            "max_depth", int(ranges["max_depth"][0]), int(ranges["max_depth"][1])
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            float(ranges["learning_rate"][0]),
            float(ranges["learning_rate"][1]),
            log=True,
        ),
        "min_child_weight": trial.suggest_float(
            "min_child_weight",
            float(ranges["min_child_weight"][0]),
            float(ranges["min_child_weight"][1]),
            log=True,
        ),
        "subsample": trial.suggest_float(
            "subsample", float(ranges["subsample"][0]), float(ranges["subsample"][1])
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            float(ranges["colsample_bytree"][0]),
            float(ranges["colsample_bytree"][1]),
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            float(ranges["reg_lambda"][0]),
            float(ranges["reg_lambda"][1]),
            log=True,
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            float(ranges["reg_alpha"][0]),
            float(ranges["reg_alpha"][1]),
            log=True,
        ),
        "gamma": trial.suggest_float(
            "gamma", float(ranges["gamma"][0]), float(ranges["gamma"][1])
        ),
    }


def _base_xgb_params(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    xgb_config = config["xgboost"]
    params = dict(xgb_config.get("params", {}))
    params.update(
        {
            "objective": str(xgb_config.get("objective", "reg:absoluteerror")),
            "eval_metric": "mae",
            "tree_method": "hist",
            "max_bin": int(xgb_config.get("max_bin", 256)),
            "random_state": int(seed),
            "n_jobs": int(xgb_config.get("n_jobs", 4)),
        }
    )
    return params


def _fit_mae(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    params: Mapping[str, Any],
    config: Mapping[str, Any],
) -> float:
    model_params = _base_xgb_params(config, int(config.get("seed", 0)))
    model_params.update(dict(params))
    model = XGBRegressor(**model_params)
    model.fit(x_train, y_train, verbose=False)
    return float(mean_absolute_error(y_eval, model.predict(x_eval)))


def _tune_view(
    view: str,
    fold_views: Sequence[Mapping[str, tuple[np.ndarray, np.ndarray]]],
    y_train: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    tuning = config["tuning"]
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=int(tuning["seed"])),
    )
    ranges = tuning["ranges"]
    warm_start = dict(config["xgboost"].get("params", {}))
    if all(
        key in warm_start
        and float(ranges[key][0]) <= float(warm_start[key]) <= float(ranges[key][1])
        for key in SEARCH_KEYS
    ):
        study.enqueue_trial({key: warm_start[key] for key in SEARCH_KEYS})

    def score(params: Mapping[str, Any]) -> list[float]:
        values: list[float] = []
        for fold_id, ((x_train, x_valid), (train_rows, valid_rows)) in enumerate(
            zip((fold[view] for fold in fold_views), folds, strict=True)
        ):
            if x_train.shape[0] != len(train_rows) or x_valid.shape[0] != len(valid_rows):
                raise ValueError(f"{view} fold matrix and index rows differ at fold {fold_id}")
            values.append(
                _fit_mae(
                    x_train,
                    y_train[np.asarray(train_rows, dtype=np.int64)],
                    x_valid,
                    y_train[np.asarray(valid_rows, dtype=np.int64)],
                    params,
                    config,
                )
            )
        return values

    def objective(trial: optuna.Trial) -> float:
        return float(np.mean(score(_params_from_trial(trial, ranges))))

    study.optimize(objective, n_trials=int(tuning["n_trials"]), show_progress_bar=False)
    best_params = dict(study.best_trial.params)
    best_fold_mae = score(best_params)
    return {
        "view": view,
        "n_trials": int(tuning["n_trials"]),
        "n_folds": int(len(folds)),
        "best_trial": int(study.best_trial.number),
        "best_cv_mae": float(np.mean(best_fold_mae)),
        "best_cv_fold_mae": best_fold_mae,
        "best_params": best_params,
        "trials": [
            {
                "trial": int(trial.number),
                "state": str(trial.state),
                "value": None if trial.value is None else float(trial.value),
                "params": dict(trial.params),
            }
            for trial in study.trials
        ],
    }


def _evaluate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    params: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    model_params = _base_xgb_params(config, int(config.get("seed", 0)))
    model_params.update(dict(params))
    model = XGBRegressor(**model_params)
    model.fit(x_train, y_train, verbose=False)
    prediction = np.asarray(model.predict(x_eval), dtype=np.float64)
    return {
        "seed": int(config.get("seed", 0)),
        "mae": float(mean_absolute_error(y_eval, prediction)),
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std()),
    }


def _label_summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "median": float(np.median(array)),
    }


def _render_markdown(result: Mapping[str, Any]) -> str:
    evaluation = result["evaluation"]
    tuning = result["tuning"]["views"]
    lines = [
        f"# {result['protocol_id']}",
        "",
        "ZINC direct topology-role × attribute binding and cross-centre covariance; no WL, K-SVD, GNN, or attention.",
        "",
        "## Protocol",
        "",
        f"- split: `{result['data']['split']}`; sizes: `{result['data']['sizes']}`",
        f"- one XGBoost/model seed: `{result['protocol']['seed']}`; objective: `{result['protocol']['objective']}`; metric: `MAE`",
        f"- tuning: `{result['tuning']['n_trials']}` trials on `{result['tuning']['n_folds']}` official-train folds only",
        "- valid: PCA fit on official train, then model fit on official train",
        "- test: PCA and model refit on official train+valid after parameters are frozen",
        "",
        "## Direct patch definition",
        "",
        f"- patch: every atom centre, induced radius-{result['representation']['radius']} ego patch",
        "- `T`: node `(shell, induced degree, cycle)` and edge `(unordered shell pair, cycle)` roles; topology-only",
        "- `A`: centre atom one-hot, patch atom histogram, incident-bond histogram, patch bond histogram",
        f"- node binding: `{result['representation']['node_binding_raw_dimension']}D`; edge binding: `{result['representation']['edge_binding_raw_dimension']}D`",
        f"- cross covariance: `Cov_v(T_v,A_v)` with raw width `{result['representation']['cross_cov_raw_dimension']}D`",
        f"- interaction PCA: separate train-only `{result['representation']['pca_rank']}D` projection for cross_cov and binding",
        f"- invariance audit: **{result['audit']['pass']}**, maximum drift `{max(result['audit']['maximum_drift'].values()):.3e}`",
        "",
        "## Results",
        "",
        "| view | dimension | train CV MAE | valid MAE | test MAE after train+valid refit |",
        "|---|---:|---:|---:|---:|",
    ]
    for view in VIEW_NAMES:
        lines.append(
            f"| `{view}` | {evaluation[view]['dimension']} | {tuning[view]['best_cv_mae']:.6f} | "
            f"{evaluation[view]['valid']['mae']:.6f} | "
            f"{evaluation[view]['test_after_train_valid_refit']['mae']:.6f} |"
        )
    baseline = evaluation["s_marginal"]
    proposed = evaluation["s_both"]
    lines.extend(
        [
            "",
            "Negative deltas are improvements in MAE.",
            f"- `s_both - s_marginal`: valid `{proposed['valid']['mae'] - baseline['valid']['mae']:+.6f}`; "
            f"test `{proposed['test_after_train_valid_refit']['mae'] - baseline['test_after_train_valid_refit']['mae']:+.6f}`",
            f"- train-CV selected view: `{result['selection']['selected_view_by_cv']}`",
            "",
            "## PCA scope metadata",
            "",
            "```json",
            json.dumps(result["pca_scope"], ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
            f"Runtime: `{result['runtime']['seconds']:.1f}s`; feature cache hit: `{result['runtime']['feature_cache_hit']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def run(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    start = time.perf_counter()
    config_path = _resolve(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_root = _resolve(config["data"]["root"])
    cache_path = _resolve(config["output"]["feature_cache"])
    datasets = tuple(_load_zinc(data_root, split) for split in ("train", "val", "test"))
    labels = tuple(
        np.asarray([float(data.y.view(-1)[0]) for data in dataset], dtype=np.float32)
        for dataset in datasets
    )
    arrays, cache_hit, feature_metadata = _build_or_load_features(
        datasets, config, data_root, cache_path
    )
    audit_config = config.get("audit", {})
    audit = audit_invariance(
        datasets[0],
        config["representation"],
        n_graphs=int(audit_config.get("n_graphs", 16)),
        seed=int(config.get("seed", 0)),
        tolerance=float(audit_config.get("tolerance", 1.0e-6)),
    )
    if not audit["pass"]:
        raise RuntimeError(f"direct patch invariance audit failed: {audit}")

    train_raw = _raw_split(arrays, "train")
    valid_raw = _raw_split(arrays, "valid")
    test_raw = _raw_split(arrays, "test")
    split_seed = int(config["tuning"]["split_seed"])
    folds = [
        (train.astype(np.int64), valid.astype(np.int64))
        for train, valid in KFold(
            n_splits=int(config["tuning"]["n_folds"]),
            shuffle=True,
            random_state=split_seed,
        ).split(np.arange(labels[0].shape[0], dtype=np.int64))
    ]
    fold_views, fold_projection = _build_cv_views(
        train_raw,
        folds,
        pca_rank=int(config["representation"]["pca_rank"]),
        pca_seed=int(config.get("seed", 0)),
    )
    tuning_results: dict[str, Any] = {}
    for view in VIEW_NAMES:
        print(f"starting direct XGBoost tuning: {view}", flush=True)
        tuning_results[view] = _tune_view(
            view, fold_views, labels[0], folds, config
        )
        print(
            f"{view}: CV MAE={tuning_results[view]['best_cv_mae']:.6f}",
            flush=True,
        )

    valid_views, valid_projection = _scope_views(
        train_raw,
        valid_raw,
        pca_rank=int(config["representation"]["pca_rank"]),
        pca_seed=int(config.get("seed", 0)),
    )
    train_valid_raw = _concat_raw(train_raw, valid_raw)
    test_views, test_projection = _scope_views(
        train_valid_raw,
        test_raw,
        pca_rank=int(config["representation"]["pca_rank"]),
        pca_seed=int(config.get("seed", 0)),
    )
    evaluation: dict[str, Any] = {}
    train_valid_labels = np.concatenate(labels[:2], axis=0)
    for view in VIEW_NAMES:
        valid_train, valid_eval = valid_views[view]
        test_train, test_eval = test_views[view]
        params = tuning_results[view]["best_params"]
        evaluation[view] = {
            "dimension": int(valid_train.shape[1]),
            "valid": _evaluate(valid_train, labels[0], valid_eval, labels[1], params, config),
            "test_after_train_valid_refit": _evaluate(
                test_train, train_valid_labels, test_eval, labels[2], params, config
            ),
        }
    selected_view = min(
        VIEW_NAMES, key=lambda view: float(tuning_results[view]["best_cv_mae"])
    )
    dims = role_dimensions(config["representation"])
    result = {
        "protocol_id": str(config["protocol_id"]),
        "status": "completed",
        "data": {
            "root": str(data_root),
            "split": "PyG ZINC subset=True official train/val/test",
            "sizes": {
                "train": len(datasets[0]),
                "valid": len(datasets[1]),
                "test": len(datasets[2]),
            },
            "source": source_audit(data_root),
            "target": {
                name: _label_summary(values)
                for name, values in zip(("train", "valid", "test"), labels, strict=True)
            },
        },
        "representation": {
            "patch": "all atom centres; induced radius-r ego patch",
            "radius": int(config["representation"]["radius"]),
            "degree_bins": int(config["representation"]["degree_bins"]),
            "atom_categories": int(config["representation"].get("atom_categories", 28)),
            "bond_categories": int(config["representation"].get("bond_categories", 4)),
            "T_definition": "node=(shell, induced-patch degree, cycle membership); edge=(unordered endpoint shell pair, cycle membership)",
            "A_definition": "centre atom one-hot + patch atom histogram + incident bond histogram + patch bond histogram",
            "binding_definition": "P(T,A) - P(T)P(A), separately for node/atom and edge/bond entities; mean+std over centres",
            "cross_cov_definition": "Cov_v(T_v, A_v) over aligned centre rows, denominator n_centres",
            "node_role_dimension": dims["node_role"],
            "edge_role_dimension": dims["edge_role"],
            "topology_row_dimension": dims["topology_row"],
            "attribute_row_dimension": dims["attribute_row"],
            "node_binding_raw_dimension": dims["node_binding"],
            "edge_binding_raw_dimension": dims["edge_binding"],
            "binding_raw_dimension": dims["binding_raw"],
            "cross_cov_raw_dimension": dims["cross_cov_raw"],
            "pca_rank": int(config["representation"]["pca_rank"]),
            "pca_scope": "separate block-wise PCA for cross_cov and binding; fit only on the relevant training scope",
            "centres": "every atom",
            "wl": False,
            "ksvd": False,
            "feature_metadata": feature_metadata,
        },
        "audit": audit,
        "protocol": {
            "seed": int(config.get("seed", 0)),
            "objective": str(config["xgboost"].get("objective", "reg:absoluteerror")),
            "metric": "MAE",
            "valid_scope": "official train fit; PCA fit on official train after train-fold parameter search",
            "test_scope": "official train+valid refit; PCA refit on official train+valid after frozen parameter search",
            "official_test_used_for_selection": False,
        },
        "tuning": {
            "n_folds": int(config["tuning"]["n_folds"]),
            "split_seed": split_seed,
            "n_trials": int(config["tuning"]["n_trials"]),
            "seed": int(config["tuning"]["seed"]),
            "selection_scope": "official train only; fold-local PCA and XGBoost",
            "views": tuning_results,
        },
        "evaluation": evaluation,
        "selection": {
            "rule": "minimum official-train fold-CV MAE",
            "selected_view_by_cv": selected_view,
        },
        "pca_scope": {
            "official_train_cv": fold_projection,
            "official_train_to_valid": valid_projection,
            "official_train_valid_to_test": test_projection,
        },
        "runtime": {
            "seconds": float(time.perf_counter() - start),
            "python": platform.python_version(),
            "numpy": importlib.metadata.version("numpy"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "xgboost": importlib.metadata.version("xgboost"),
            "optuna": importlib.metadata.version("optuna"),
            "feature_cache_hit": bool(cache_hit),
            "feature_cache": str(cache_path),
            "feature_cache_sha256": _sha256(cache_path),
            "script_sha256": _sha256(Path(__file__).resolve()),
        },
    }
    output = config["output"]
    result_json = _resolve(output["json"])
    result_markdown = _resolve(output["markdown"])
    _write_json_atomic(result_json, result)
    result_markdown.parent.mkdir(parents=True, exist_ok=True)
    result_markdown.write_text(_render_markdown(result), encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    result = run(args.config)
    print(
        json.dumps(
            {
                "protocol_id": result["protocol_id"],
                "selected_view_by_cv": result["selection"]["selected_view_by_cv"],
                "valid_mae": {
                    view: result["evaluation"][view]["valid"]["mae"] for view in VIEW_NAMES
                },
                "test_mae_after_train_valid_refit": {
                    view: result["evaluation"][view]["test_after_train_valid_refit"]["mae"]
                    for view in VIEW_NAMES
                },
                "runtime_seconds": result["runtime"]["seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
