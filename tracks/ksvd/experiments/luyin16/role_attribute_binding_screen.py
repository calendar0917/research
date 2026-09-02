"""Fast MolHIV screen for topology-role x attribute binding.

This experiment deliberately does not use K-SVD.  For every radius-2 rooted
ego patch it builds two independent descriptions:

* structural roles, obtained only from the induced unlabelled topology;
* compact OGB atom/bond semantic one-hot attributes.

The graph-level candidate is the mean over rooted patches of
``P(role, attribute)`` and the centered binding
``P(role, attribute) - P(role) P(attribute)``.  A within-patch attribute
permutation is the null control.  It preserves every patch's role marginal,
attribute marginal, number of entities, and the graph's patch/size context,
while breaking the correspondence between roles and attributes.

The runner uses only the official-train scaffold folds for supervised scores.
The frozen mentor ``S`` feature is used as the existing baseline; official
validation/test labels are not used by this protocol.
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from ksvd_research.data import load_molhiv
from tracks.ksvd.experiments.luyin16.patch_object_audit import relabel_graph_features


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = REPO_ROOT / "tracks/ksvd/configs/luyin16/role_attribute_binding_screen.yaml"
DEFAULT_SMOKE_CONFIG = (
    REPO_ROOT / "tracks/ksvd/configs/luyin16/role_attribute_binding_screen_smoke.yaml"
)

SHELL_PAIRS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))

# These are the same compact semantic buckets used by the mentor proxy.  The
# role side never reads them; they are only an explicitly documented attribute
# channel.  Keeping the compact 48/13 representation makes the first
# mechanism screen inexpensive and interpretable.
ELEMENT_GROUPS = (
    ("H", {1}),
    ("B", {5}),
    ("C", {6}),
    ("N", {7}),
    ("O", {8}),
    ("F", {9}),
    ("Si", {14}),
    ("P", {15}),
    ("S", {16}),
    ("Cl", {17}),
    ("Se", {34}),
    ("Br", {35}),
    ("I", {53}),
    ("other", set()),
)
ATOM_DIM = 48
BOND_DIM = 13


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
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


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"n": 0, "mean": 0.0, "std": 0.0, "minimum": 0.0, "maximum": 0.0}
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def one_hot_index(index: int, size: int) -> np.ndarray:
    output = np.zeros(int(size), dtype=np.float32)
    index = int(index)
    output[index if 0 <= index < size else size - 1] = 1.0
    return output


def atomic_number_group(raw_atomic_index: int) -> int:
    raw_atomic_index = int(raw_atomic_index)
    if 0 <= raw_atomic_index < 118:
        atomic_number = raw_atomic_index + 1
        for group_id, (_, atomic_numbers) in enumerate(ELEMENT_GROUPS[:-1]):
            if atomic_number in atomic_numbers:
                return group_id
    return len(ELEMENT_GROUPS) - 1


def compact_atom_semantics(raw_atom_features: np.ndarray) -> np.ndarray:
    """Return the fixed 48D compact OGB atom attribute block."""
    raw = np.asarray(raw_atom_features, dtype=np.int64)
    if raw.ndim != 2 or raw.shape[1] != 9:
        raise ValueError(f"Expected OGB atom features [N,9], got {raw.shape}")
    rows: list[np.ndarray] = []
    for feature in raw:
        degree = int(feature[2])
        charge = int(feature[3])
        charge_bucket = 3 if charge == 11 else (0 if charge < 5 else 1 if charge == 5 else 2)
        num_h = int(feature[4])
        radical = int(feature[5])
        radical_bucket = 3 if radical == 5 else (0 if radical == 0 else 1 if radical == 1 else 2)
        pieces = (
            one_hot_index(atomic_number_group(int(feature[0])), 14),
            one_hot_index(int(feature[1]), 5),
            one_hot_index(degree if 0 <= degree <= 4 else 5, 6),
            one_hot_index(charge_bucket, 4),
            one_hot_index(num_h if 0 <= num_h <= 3 else 4, 5),
            one_hot_index(radical_bucket, 4),
            one_hot_index(int(feature[6]), 6),
            one_hot_index(int(feature[7]), 2),
            one_hot_index(int(feature[8]), 2),
        )
        row = np.concatenate(pieces).astype(np.float32, copy=False)
        if row.shape != (ATOM_DIM,):
            raise RuntimeError(f"compact atom dimension mismatch: {row.shape}")
        rows.append(row)
    return np.stack(rows, axis=0) if rows else np.zeros((0, ATOM_DIM), dtype=np.float32)


def compact_bond_semantics(raw_bond_feature: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw_bond_feature, dtype=np.int64).reshape(-1)
    if raw.shape != (3,):
        raise ValueError(f"Expected OGB bond feature [3], got {raw.shape}")
    return np.concatenate(
        [one_hot_index(int(raw[0]), 5), one_hot_index(int(raw[1]), 6), one_hot_index(int(raw[2]), 2)]
    ).astype(np.float32, copy=False)


def _ego_distances(graph, center: int, radius: int) -> dict[int, int]:
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


def _cycle_membership(graph) -> tuple[set[int], set[tuple[int, int]]]:
    """Return nodes/edges lying on a cycle using a bridge DFS."""
    discovery: dict[int, int] = {}
    low: dict[int, int] = {}
    bridges: set[tuple[int, int]] = set()
    clock = 0

    def visit(node: int, parent: int | None) -> None:
        nonlocal clock
        clock += 1
        discovery[node] = clock
        low[node] = clock
        for neighbor in sorted(graph.neighbors(node)):
            if neighbor == parent:
                continue
            edge = graph.edge_key(node, neighbor)
            if neighbor not in discovery:
                visit(int(neighbor), node)
                low[node] = min(low[node], low[neighbor])
                if low[neighbor] > discovery[node]:
                    bridges.add(edge)
            else:
                low[node] = min(low[node], discovery[neighbor])

    for node in graph.nodes:
        if node not in discovery:
            visit(int(node), None)
    cycle_edges = {edge for edge in graph.edges() if graph.edge_key(*edge) not in bridges}
    cycle_nodes: set[int] = set()
    for left, right in cycle_edges:
        cycle_nodes.update((int(left), int(right)))
    return cycle_nodes, cycle_edges


def role_dimensions(config: Mapping[str, Any]) -> dict[str, int]:
    degree_bins = int(config["degree_bins"])
    node_role = 3 * degree_bins * 2
    edge_role = len(SHELL_PAIRS) * 2
    return {
        "node_role": node_role,
        "edge_role": edge_role,
        "node_attribute": ATOM_DIM,
        "edge_attribute": BOND_DIM,
        "node_raw": node_role * ATOM_DIM,
        "edge_raw": edge_role * BOND_DIM,
        "node_binding": node_role * ATOM_DIM,
        "edge_binding": edge_role * BOND_DIM,
        "context": 5,
    }


def _node_role_id(shell: int, degree: int, on_cycle: bool, degree_bins: int) -> int:
    degree_bin = min(max(int(degree), 0), int(degree_bins) - 1)
    return ((int(shell) * int(degree_bins) + degree_bin) * 2) + int(bool(on_cycle))


def _edge_role_id(shell_pair: tuple[int, int], on_cycle: bool) -> int:
    pair = tuple(sorted((int(shell_pair[0]), int(shell_pair[1]))))
    try:
        pair_index = SHELL_PAIRS.index(pair)
    except ValueError as exc:
        raise ValueError(f"unsupported shell pair {pair}") from exc
    return pair_index * 2 + int(bool(on_cycle))


def _patch_entities(
    graph,
    center: int,
    atom_semantics: np.ndarray,
    bond_semantics: Mapping[tuple[int, int], np.ndarray],
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    distances = _ego_distances(graph, int(center), int(config["radius"]))
    nodes = set(distances)
    induced = graph.induced(nodes)
    cycle_nodes, cycle_edges = _cycle_membership(induced)
    degree_bins = int(config["degree_bins"])

    node_roles = np.asarray(
        [
            _node_role_id(
                distances[node],
                len(induced.neighbors(node)),
                node in cycle_nodes,
                degree_bins,
            )
            for node in sorted(nodes)
        ],
        dtype=np.int64,
    )
    node_attributes = np.stack(
        [atom_semantics[int(node)] for node in sorted(nodes)], axis=0
    ).astype(np.float32, copy=False)

    edge_rows = sorted(induced.edges())
    edge_roles = np.asarray(
        [
            _edge_role_id(
                (distances[left], distances[right]),
                graph.edge_key(left, right) in cycle_edges,
            )
            for left, right in edge_rows
        ],
        dtype=np.int64,
    )
    if edge_rows:
        edge_attributes = np.stack(
            [bond_semantics[graph.edge_key(left, right)] for left, right in edge_rows], axis=0
        ).astype(np.float32, copy=False)
    else:
        edge_attributes = np.zeros((0, BOND_DIM), dtype=np.float32)
    return node_roles, node_attributes, edge_roles, edge_attributes


def _entity_statistics(
    roles: np.ndarray,
    attributes: np.ndarray,
    n_roles: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return role marginal, attribute marginal, and uncentered joint."""
    if roles.ndim != 1 or attributes.ndim != 2 or roles.shape[0] != attributes.shape[0]:
        raise ValueError("roles and attributes are not aligned")
    n_entities = int(roles.shape[0])
    if n_entities == 0:
        return (
            np.zeros(int(n_roles), dtype=np.float32),
            np.zeros(attributes.shape[1], dtype=np.float32),
            np.zeros((int(n_roles), attributes.shape[1]), dtype=np.float32),
        )
    role_marginal = np.bincount(roles, minlength=int(n_roles)).astype(np.float32) / float(n_entities)
    attribute_marginal = attributes.mean(axis=0, dtype=np.float32)
    joint = np.zeros((int(n_roles), attributes.shape[1]), dtype=np.float32)
    np.add.at(joint, roles, attributes)
    joint /= float(n_entities)
    return role_marginal, attribute_marginal, joint


def _patch_statistics(
    node_roles: np.ndarray,
    node_attributes: np.ndarray,
    edge_roles: np.ndarray,
    edge_attributes: np.ndarray,
    config: Mapping[str, Any],
    *,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    dims = role_dimensions(config)
    node_attr = node_attributes
    edge_attr = edge_attributes
    if rng is not None:
        if node_attr.shape[0] > 1:
            node_attr = node_attr[rng.permutation(node_attr.shape[0])]
        if edge_attr.shape[0] > 1:
            edge_attr = edge_attr[rng.permutation(edge_attr.shape[0])]
    node_role, node_attribute, node_joint = _entity_statistics(
        node_roles, node_attr, dims["node_role"]
    )
    edge_role, edge_attribute, edge_joint = _entity_statistics(
        edge_roles, edge_attr, dims["edge_role"]
    )
    node_binding = node_joint - node_role[:, None] * node_attribute[None, :]
    edge_binding = edge_joint - edge_role[:, None] * edge_attribute[None, :]
    return {
        "node_role": node_role,
        "edge_role": edge_role,
        "node_attribute": node_attribute,
        "edge_attribute": edge_attribute,
        "node_raw": node_joint.ravel(),
        "edge_raw": edge_joint.ravel(),
        "node_binding": node_binding.ravel(),
        "edge_binding": edge_binding.ravel(),
    }


def _graph_context(graph, n_patches: int, node_counts: Sequence[int], edge_counts: Sequence[int]) -> np.ndarray:
    return np.asarray(
        [
            np.log1p(float(graph.n)),
            np.log1p(float(graph.num_edges())),
            np.log1p(float(n_patches)),
            float(np.mean(node_counts)) if node_counts else 0.0,
            float(np.mean(edge_counts)) if edge_counts else 0.0,
        ],
        dtype=np.float32,
    )


def graph_role_attribute_features(
    graph,
    node_features: np.ndarray,
    edge_features: Mapping[tuple[int, int], np.ndarray],
    config: Mapping[str, Any],
    *,
    shuffle_repeats: int = 0,
    shuffle_seed: int = 0,
) -> dict[str, np.ndarray]:
    """Build all graph-level blocks, optionally including shuffled controls."""
    if int(config["radius"]) != 2:
        raise ValueError("the first role schema is fixed to radius=2")
    atom_semantics = compact_atom_semantics(node_features)
    bond_semantics = {
        graph.edge_key(left, right): compact_bond_semantics(values)
        for (left, right), values in edge_features.items()
    }
    centers = list(graph.nodes)
    max_centers = config.get("max_centers_per_graph")
    if max_centers is not None and len(centers) > int(max_centers):
        rng = np.random.default_rng(int(shuffle_seed) + 17)
        centers = sorted(rng.choice(centers, size=int(max_centers), replace=False).tolist())
    true_accumulator: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "node_role",
            "edge_role",
            "node_attribute",
            "edge_attribute",
            "node_raw",
            "edge_raw",
            "node_binding",
            "edge_binding",
        )
    }
    shuffled_accumulators: list[dict[str, list[np.ndarray]]] = [
        {key: [] for key in true_accumulator} for _ in range(int(shuffle_repeats))
    ]
    node_counts: list[int] = []
    edge_counts: list[int] = []
    for center in centers:
        entities = _patch_entities(graph, int(center), atom_semantics, bond_semantics, config)
        node_roles, node_attrs, edge_roles, edge_attrs = entities
        node_counts.append(int(node_roles.size))
        edge_counts.append(int(edge_roles.size))
        true_stats = _patch_statistics(*entities, config)
        for key, value in true_stats.items():
            true_accumulator[key].append(value)
        for repeat, accumulator in enumerate(shuffled_accumulators):
            rng = np.random.default_rng(
                int(shuffle_seed) + 1000003 * (repeat + 1) + 7919 * int(center)
            )
            shuffled_stats = _patch_statistics(*entities, config, rng=rng)
            for key, value in shuffled_stats.items():
                accumulator[key].append(value)

    dims = role_dimensions(config)
    context = _graph_context(graph, len(centers), node_counts, edge_counts)

    def average(accumulator: Mapping[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
        output: dict[str, np.ndarray] = {}
        for key, values in accumulator.items():
            if values:
                output[key] = np.mean(np.stack(values, axis=0), axis=0, dtype=np.float32)
            else:
                output[key] = np.zeros(
                    dims[key] if key in dims else 0, dtype=np.float32
                )
            if output[key].ndim != 1:
                output[key] = output[key].reshape(-1)
            if not np.all(np.isfinite(output[key])):
                raise FloatingPointError(f"non-finite graph block {key}")
        return output

    output = average(true_accumulator)
    output["context"] = context
    for repeat, accumulator in enumerate(shuffled_accumulators):
        shuffled = average(accumulator)
        for key, value in shuffled.items():
            output[f"{key}_shuffled_{repeat}"] = value
    return output


def audit_invariance(
    bundle,
    indices: np.ndarray,
    representation: Mapping[str, Any],
    *,
    n_graphs: int,
    permutations_per_graph: int,
    seed: int,
    tolerance: float,
) -> dict[str, Any]:
    """Audit true (unshuffled) blocks under arbitrary node relabelings."""
    if bundle.node_feats is None or bundle.edge_feats is None:
        raise ValueError("invariance audit requires node and edge features")
    candidates = np.asarray(indices, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    if candidates.size > int(n_graphs):
        candidates = rng.choice(candidates, size=int(n_graphs), replace=False)
    drift: dict[str, list[float]] = {
        key: []
        for key in (
            "node_role",
            "edge_role",
            "node_attribute",
            "edge_attribute",
            "node_raw",
            "edge_raw",
            "node_binding",
            "edge_binding",
            "context",
        )
    }
    for raw_index in candidates:
        index = int(raw_index)
        graph = bundle.graphs[index]
        node_features = bundle.node_feats[index]
        edge_features = bundle.edge_feats[index]
        base = graph_role_attribute_features(graph, node_features, edge_features, representation)
        for _ in range(int(permutations_per_graph)):
            permutation = rng.permutation(graph.n)
            changed_graph, changed_nodes, changed_edges = relabel_graph_features(
                graph, node_features, edge_features, permutation
            )
            changed = graph_role_attribute_features(
                changed_graph, changed_nodes, changed_edges, representation
            )
            for key in drift:
                drift[key].append(float(np.max(np.abs(base[key] - changed[key]), initial=0.0)))
    summaries = {key: _summary(values) for key, values in drift.items()}
    return {
        "n_graphs": int(candidates.size),
        "permutations_per_graph": int(permutations_per_graph),
        "tolerance": float(tolerance),
        "maximum_drift": {key: row["maximum"] for key, row in summaries.items()},
        "summaries": summaries,
        "pass": bool(all(row["maximum"] <= float(tolerance) for row in summaries.values())),
    }


def assemble_feature_views(blocks: Mapping[str, np.ndarray], config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Map primitive blocks to the registered model views for one graph."""
    context = np.asarray(blocks["context"], dtype=np.float32)
    axis = 0 if context.ndim == 1 else 1

    def join(*parts: np.ndarray) -> np.ndarray:
        return np.concatenate(parts, axis=axis).astype(np.float32, copy=False)

    structure = join(blocks["node_role"], blocks["edge_role"], context)
    attribute = join(blocks["node_attribute"], blocks["edge_attribute"], context)
    marginals = join(
        blocks["node_role"], blocks["edge_role"], blocks["node_attribute"], blocks["edge_attribute"], context
    )
    raw_node = join(blocks["node_raw"], context)
    raw_edge = join(blocks["edge_raw"], context)
    raw = join(blocks["node_raw"], blocks["edge_raw"], context)
    binding_node = join(blocks["node_binding"], context)
    binding_edge = join(blocks["edge_binding"], context)
    binding = join(blocks["node_binding"], blocks["edge_binding"], context)
    return {
        "structure": structure,
        "attribute": attribute,
        "marginals": marginals,
        "raw_node": raw_node,
        "raw_edge": raw_edge,
        "raw": raw,
        "binding_node": binding_node,
        "binding_edge": binding_edge,
        "binding": binding,
    }


def _select_rows(labels: np.ndarray, candidates: np.ndarray, maximum: int | None, seed: int) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=np.int64)
    if maximum is None or candidates.size <= int(maximum):
        return np.sort(candidates)
    rng = np.random.default_rng(int(seed))
    positive = candidates[labels[candidates] == 1]
    negative = candidates[labels[candidates] == 0]
    n_positive = min(max(int(round(int(maximum) * positive.size / max(candidates.size, 1))), 1), positive.size)
    n_negative = min(int(maximum) - n_positive, negative.size)
    selected = np.concatenate(
        [rng.choice(positive, n_positive, replace=False), rng.choice(negative, n_negative, replace=False)]
    )
    return np.sort(selected.astype(np.int64))


def _fold_indices(
    archive: Mapping[str, np.ndarray], fold: int, labels: np.ndarray, screen: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    original = np.asarray(archive["original_indices"], dtype=np.int64)
    train = original[np.asarray(archive[f"fold_{fold}_train_indices"], dtype=np.int64)]
    valid = original[np.asarray(archive[f"fold_{fold}_valid_indices"], dtype=np.int64)]
    seed = int(screen["seed"]) + 1009 * int(fold)
    return (
        _select_rows(labels, train, screen.get("max_train_graphs_per_fold"), seed),
        _select_rows(labels, valid, screen.get("max_valid_graphs_per_fold"), seed + 1),
    )


def _frozen_s_rows(frozen: Mapping[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    mapping = {
        int(dataset_index): position
        for position, dataset_index in enumerate(np.asarray(frozen["dataset_indices"], dtype=np.int64))
    }
    rows = []
    for index in np.asarray(indices, dtype=np.int64):
        if int(index) not in mapping:
            raise ValueError(f"dataset index {int(index)} is absent from frozen S")
        rows.append(mapping[int(index)])
    return np.asarray(frozen["s"][np.asarray(rows, dtype=np.int64)], dtype=np.float32)


def _fit_auc_views(
    views: Mapping[str, np.ndarray],
    labels: np.ndarray,
    n_train: int,
    classifier: Mapping[str, Any],
    model_seeds: Sequence[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    y_train = np.asarray(labels[:n_train], dtype=np.int64)
    y_valid = np.asarray(labels[n_train:], dtype=np.int64)
    if np.unique(y_train).size < 2 or np.unique(y_valid).size < 2:
        raise ValueError("both train and validation rows need two classes")
    weight = float(np.sum(y_train == 0) / max(1, np.sum(y_train == 1)))
    fixed = {
        key: classifier[key]
        for key in (
            "n_estimators",
            "max_depth",
            "learning_rate",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "reg_lambda",
            "reg_alpha",
            "n_jobs",
        )
        if key in classifier
    }
    by_seed: dict[str, list[dict[str, float]]] = {name: [] for name in views}
    for seed in model_seeds:
        for name, matrix in views.items():
            if matrix.shape[0] != labels.shape[0]:
                raise ValueError(f"view {name} has {matrix.shape[0]} rows, expected {labels.shape[0]}")
            model = XGBClassifier(
                **fixed,
                objective="binary:logistic",
                eval_metric="auc",
                scale_pos_weight=weight,
                random_state=int(seed),
                tree_method="hist",
            )
            model.fit(matrix[:n_train], y_train)
            train_pred = model.predict_proba(matrix[:n_train])[:, 1]
            valid_pred = model.predict_proba(matrix[n_train:])[:, 1]
            by_seed[name].append(
                {
                    "train_auc": float(roc_auc_score(y_train, train_pred)),
                    "valid_auc": float(roc_auc_score(y_valid, valid_pred)),
                }
            )
    aggregate: dict[str, Any] = {}
    for name, records in by_seed.items():
        aggregate[name] = {
            "dimension": int(views[name].shape[1]),
            "train_auc": float(np.mean([row["train_auc"] for row in records])),
            "valid_auc": float(np.mean([row["valid_auc"] for row in records])),
            "train_auc_std": float(np.std([row["train_auc"] for row in records])),
            "valid_auc_std": float(np.std([row["valid_auc"] for row in records])),
        }
    return aggregate, {name: records for name, records in by_seed.items()}


def _aggregate_folds(folds: Sequence[Mapping[str, Any]], view: str) -> dict[str, Any]:
    values = [float(row["scores"][view]["valid_auc"]) for row in folds]
    return {"fold_auc": values, "mean_auc": float(np.mean(values)), "std_auc": float(np.std(values))}


def _delta(
    folds: Sequence[Mapping[str, Any]], candidate: str, baseline: str
) -> dict[str, Any]:
    values = [
        float(row["scores"][candidate]["valid_auc"] - row["scores"][baseline]["valid_auc"])
        for row in folds
    ]
    return {
        "candidate": candidate,
        "baseline": baseline,
        "fold_deltas": values,
        "mean_delta": float(np.mean(values)),
        "fold_wins": int(np.sum(np.asarray(values) > 0.0)),
    }


def _delta_against_controls(
    folds: Sequence[Mapping[str, Any]], candidate: str, controls: Sequence[str]
) -> dict[str, Any]:
    """Compare a candidate with the mean control score in each fold."""
    if not controls:
        raise ValueError("at least one control is required")
    values: list[float] = []
    for row in folds:
        candidate_score = float(row["scores"][candidate]["valid_auc"])
        control_score = float(
            np.mean([row["scores"][control]["valid_auc"] for control in controls])
        )
        values.append(candidate_score - control_score)
    return {
        "candidate": candidate,
        "baseline": "mean(" + ",".join(controls) + ")",
        "fold_deltas": values,
        "mean_delta": float(np.mean(values)),
        "fold_wins": int(np.sum(np.asarray(values) > 0.0)),
    }


def _build_dataset_blocks(bundle, indices: np.ndarray, representation: Mapping[str, Any], repeats: int, seed: int):
    assert bundle.node_feats is not None and bundle.edge_feats is not None
    unique_indices = np.asarray(indices, dtype=np.int64)
    if unique_indices.size == 0:
        raise ValueError("no graph indices selected")
    first = graph_role_attribute_features(
        bundle.graphs[int(unique_indices[0])],
        bundle.node_feats[int(unique_indices[0])],
        bundle.edge_feats[int(unique_indices[0])],
        representation,
        shuffle_repeats=repeats,
        shuffle_seed=seed + int(unique_indices[0]),
    )
    primitive_names = list(first)
    primitive_arrays = {
        name: np.zeros((unique_indices.size, first[name].shape[0]), dtype=np.float32)
        for name in primitive_names
    }
    for position, raw_index in enumerate(unique_indices):
        index = int(raw_index)
        blocks = first if position == 0 else graph_role_attribute_features(
            bundle.graphs[index],
            bundle.node_feats[index],
            bundle.edge_feats[index],
            representation,
            shuffle_repeats=repeats,
            shuffle_seed=seed + index,
        )
        for name in primitive_names:
            if blocks[name].shape != (first[name].shape[0],):
                raise RuntimeError(f"block shape changed for {name}: {blocks[name].shape}")
            primitive_arrays[name][position] = blocks[name]
        if position and position % 500 == 0:
            print(f"feature graphs: {position}/{unique_indices.size}", flush=True)
    return unique_indices, primitive_arrays


def _make_views_for_indices(
    primitive_arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
    requested: Sequence[str],
    frozen: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    all_view_names = {
        "structure",
        "attribute",
        "marginals",
        "raw_node",
        "raw_edge",
        "raw",
        "binding_node",
        "binding_edge",
        "binding",
    }
    # Assemble primitive true/shuffled blocks once per requested view.  The
    # caller requests names such as binding_shuffled_0; the suffix is kept
    # explicit in output so controls cannot be confused with the true view.
    result: dict[str, np.ndarray] = {}
    s = _frozen_s_rows(frozen, np.asarray(indices, dtype=np.int64))

    def block_rows(prefix: str, suffix: str = "") -> dict[str, np.ndarray]:
        output: dict[str, np.ndarray] = {}
        for key in (
            "node_role",
            "edge_role",
            "node_attribute",
            "edge_attribute",
            "node_raw",
            "edge_raw",
            "node_binding",
            "edge_binding",
        ):
            source = f"{key}{suffix}"
            if source in primitive_arrays:
                output[key] = np.asarray(primitive_arrays[source], dtype=np.float32)
        # Context is graph-level and therefore is not shuffled.
        output["context"] = np.asarray(primitive_arrays["context"], dtype=np.float32)
        return output

    # ``indices`` is already the union order used when the primitive arrays
    # were built.  No fitted or fold-specific transform is applied here.
    for name in requested:
        if name == "s":
            result[name] = s
            continue
        suffix = ""
        base_name = name
        if "_shuffled_" in name:
            base_name, repeat_text = name.rsplit("_shuffled_", 1)
            suffix = f"_shuffled_{int(repeat_text)}"
        if base_name not in all_view_names:
            raise ValueError(f"unknown requested view {name}")
        blocks = block_rows(base_name, suffix)
        assembled = assemble_feature_views(blocks, {})
        if base_name not in assembled:
            raise ValueError(f"cannot assemble view {name}")
        # The primitive arrays are currently already restricted to ``indices``;
        # retaining this explicit branch makes the alignment contract visible.
        result[name] = assembled[base_name]
        if result[name].shape[0] != len(indices):
            raise RuntimeError(f"view {name} row alignment failed")
    return result


def _make_s_views(
    primitive_arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
    requested: Sequence[str],
    frozen: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Assemble candidate blocks and prepend the frozen S baseline.

    ``requested`` names are primitive candidate names (``binding``,
    ``binding_shuffled_0``, ...).  The returned supervised view names are
    explicitly prefixed with ``s_`` so a score cannot be mistaken for a
    standalone candidate score.
    """
    base = _make_views_for_indices(primitive_arrays, indices, requested, frozen)
    result: dict[str, np.ndarray] = {}
    s = _frozen_s_rows(frozen, indices)
    result["s"] = s
    for name, matrix in base.items():
        if name == "s":
            continue
        result[f"s_{name}"] = np.concatenate([s, matrix], axis=1)
    return result


def _render_markdown(result: Mapping[str, Any]) -> str:
    if "representation" not in result:
        audit = result.get("object_audit", {})
        return "\n".join(
            [
                "# MolHIV role × attribute binding screen",
                "",
                f"Protocol: `{result.get('protocol_id', 'unknown')}`",
                "",
                f"Object audit: **{'PASS' if audit.get('pass') else 'FAIL'}**.",
                "",
                f"Decision: **{result.get('decision', 'UNKNOWN')}**.",
                "",
            ]
        )
    lines = [
        "# MolHIV role × attribute binding screen",
        "",
        f"Protocol: `{result['protocol_id']}`",
        "",
        "The structural role side uses only radius-2 induced topology; the attribute side uses compact OGB atom/bond semantics. Each patch contributes an entity-normalized joint table and its centered residual.",
        "",
        "## Feature schema",
        "",
        f"- node role dimension: {result['representation']['dimensions']['node_role']}",
        f"- edge role dimension: {result['representation']['dimensions']['edge_role']}",
        f"- compact atom attribute dimension: {result['representation']['dimensions']['node_attribute']}",
        f"- compact bond attribute dimension: {result['representation']['dimensions']['edge_attribute']}",
        "- role = shell × induced-degree-bin × cycle-membership (nodes), or shell-pair × cycle-membership (edges)",
        "- centered binding = joint − role marginal × attribute marginal, averaged over rooted patches",
        "- shuffle = independent within-patch permutation of node and edge attribute rows",
        "",
        "## Fold scores",
        "",
        "| view | mean validation ROC-AUC | fold std |",
        "|---|---:|---:|",
    ]
    for name, row in result.get("aggregate", {}).items():
        lines.append(f"| `{name}` | {row['mean_auc']:.6f} | {row['std_auc']:.6f} |")
    lines.extend(["", "## Gates", ""])
    gates = result.get("gates", {})
    for name, gate in gates.items():
        lines.append(
            f"- `{name}`: mean delta {gate['mean_delta']:+.6f}, wins {gate['fold_wins']}/{result['n_folds']} — **{'PASS' if gate['passed'] else 'FAIL'}**"
        )
    lines.extend(
        [
            "",
            "Interpretation: true-vs-shuffle is evidence that the representation contains role–attribute dependence; only a positive binding-vs-marginals (and S+binding-vs-S) increment is evidence that this dependence helps the present MolHIV task.",
            "",
            f"Decision: **{result.get('decision', 'UNKNOWN')}**.",
            "",
            "Official test was not encoded or evaluated, and no hyperparameter search was run.",
            "",
        ]
    )
    return "\n".join(lines)


def run(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_config = config["data"]
    representation = config["representation"]
    screen = config["screen"]
    classifier = config["classifier"]
    result_json = _resolve(config["output_json"])
    result_md = _resolve(config["output_markdown"])
    frozen_path = _resolve(data_config["frozen_features"])
    folds_path = _resolve(data_config["scaffold_folds"])

    with np.load(frozen_path, allow_pickle=False) as archive:
        frozen = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(folds_path, allow_pickle=False) as archive:
        fold_archive = {name: np.asarray(archive[name]) for name in archive.files}
    bundle = load_molhiv(root=_resolve(data_config["root"]), with_features=True)
    labels = bundle.y.astype(np.int64)
    start = time.perf_counter()

    original = np.asarray(fold_archive["original_indices"], dtype=np.int64)
    audit_config = config.get(
        "audit",
        {
            "n_graphs": 64,
            "permutations_per_graph": 2,
            "seed": int(screen["seed"]) + 41,
            "tolerance": 1.0e-6,
        },
    )
    audit_candidates = original[np.asarray(fold_archive["official_train_indices"], dtype=np.int64)]
    object_audit = audit_invariance(
        bundle,
        audit_candidates,
        representation,
        n_graphs=int(audit_config["n_graphs"]),
        permutations_per_graph=int(audit_config["permutations_per_graph"]),
        seed=int(audit_config["seed"]),
        tolerance=float(audit_config["tolerance"]),
    )
    if not object_audit["pass"]:
        result = {
            "protocol_id": config["protocol_id"],
            "config": config,
            "object_audit": object_audit,
            "decision": "STOP_INVARIANCE_AUDIT_FAILED",
            "runtime": {"seconds": float(time.perf_counter() - start)},
        }
        _write_json(result_json, result)
        result_md.parent.mkdir(parents=True, exist_ok=True)
        result_md.write_text(_render_markdown(result), encoding="utf-8")
        return result

    split_mode = str(screen.get("split", "scaffold_folds"))
    if split_mode == "official_valid":
        train = original[np.asarray(fold_archive["official_train_indices"], dtype=np.int64)]
        valid = original[np.asarray(fold_archive["official_valid_indices"], dtype=np.int64)]
        train = _select_rows(labels, train, screen.get("max_train_graphs_per_fold"), int(screen["seed"]))
        valid = _select_rows(labels, valid, screen.get("max_valid_graphs_per_fold"), int(screen["seed"]) + 1)
        fold_ids = [0]
        selections = [(train, valid)]
    elif split_mode == "scaffold_folds":
        fold_ids = sorted(
            int(name.removeprefix("fold_").removesuffix("_train_indices"))
            for name in fold_archive
            if name.startswith("fold_") and name.endswith("_train_indices")
        )
        selections = [_fold_indices(fold_archive, fold, labels, screen) for fold in fold_ids]
    else:
        raise ValueError(f"unknown screen split mode: {split_mode}")
    selected_union = np.unique(np.concatenate([np.concatenate(pair) for pair in selections])).astype(np.int64)
    print(
        f"selected graphs={selected_union.size}; folds={fold_ids}; "
        f"shuffle_repeats={int(screen['shuffle_repeats'])}",
        flush=True,
    )
    unique_indices, primitive_arrays = _build_dataset_blocks(
        bundle,
        selected_union,
        representation,
        int(screen["shuffle_repeats"]),
        int(screen["seed"]),
    )
    requested = [str(value) for value in config["views"]]
    # Build all S+candidate views on the selected union once.  Fold scoring
    # then only slices rows; no fold-specific feature fitting occurs.
    all_views = _make_s_views(primitive_arrays, unique_indices, requested, frozen)
    index_to_row = {int(index): position for position, index in enumerate(unique_indices)}

    folds: list[dict[str, Any]] = []
    model_seeds = [int(value) for value in classifier.get("model_seeds", [0])]
    for fold, (train_indices, valid_indices) in zip(fold_ids, selections, strict=True):
        fold_indices = np.concatenate([train_indices, valid_indices]).astype(np.int64)
        rows = np.asarray([index_to_row[int(index)] for index in fold_indices], dtype=np.int64)
        fold_views = {name: matrix[rows] for name, matrix in all_views.items()}
        fold_scores, by_seed = _fit_auc_views(
            fold_views,
            labels[fold_indices],
            len(train_indices),
            classifier,
            model_seeds,
        )
        folds.append(
            {
                "fold": int(fold),
                "n_train": int(train_indices.size),
                "n_valid": int(valid_indices.size),
                "n_train_positive": int(labels[train_indices].sum()),
                "n_valid_positive": int(labels[valid_indices].sum()),
                "scores": fold_scores,
                "scores_by_model_seed": by_seed,
            }
        )
        print(
            f"fold {fold}: S={fold_scores['s']['valid_auc']:.6f}; "
            f"S+binding={fold_scores.get('s_binding', {'valid_auc': float('nan')})['valid_auc']:.6f}",
            flush=True,
        )

    aggregate = {
        name: _aggregate_folds(folds, name)
        for name in folds[0]["scores"]
    }
    minimum_delta = float(screen["minimum_delta"])
    minimum_wins = int(screen["minimum_fold_wins"])
    shuffle_repeats = int(screen["shuffle_repeats"])
    binding_shuffle_controls = [f"s_binding_shuffled_{repeat}" for repeat in range(shuffle_repeats)]
    raw_shuffle_controls = [f"s_raw_shuffled_{repeat}" for repeat in range(shuffle_repeats)]
    gates = {
        "binding_vs_shuffle": _delta_against_controls(
            folds, "s_binding", binding_shuffle_controls
        ),
        "binding_vs_marginals": _delta(folds, "s_binding", "s_marginals"),
        "binding_vs_s": _delta(folds, "s_binding", "s"),
        "binding_vs_raw": _delta(folds, "s_binding", "s_raw"),
        "raw_vs_shuffle": _delta_against_controls(folds, "s_raw", raw_shuffle_controls),
        "raw_vs_marginals": _delta(folds, "s_raw", "s_marginals"),
        "raw_vs_s": _delta(folds, "s_raw", "s"),
    }
    for repeat in range(shuffle_repeats):
        gates[f"binding_vs_shuffle_{repeat}"] = _delta(
            folds, "s_binding", f"s_binding_shuffled_{repeat}"
        )
        gates[f"raw_vs_shuffle_{repeat}"] = _delta(
            folds, "s_raw", f"s_raw_shuffled_{repeat}"
        )
    for gate in gates.values():
        gate["minimum_mean_delta"] = minimum_delta
        gate["minimum_fold_wins"] = minimum_wins
        gate["passed"] = bool(
            gate["mean_delta"] >= minimum_delta and gate["fold_wins"] >= minimum_wins
        )
    binding_gate = gates["binding_vs_shuffle"]["passed"]
    raw_gate = gates["raw_vs_shuffle"]["passed"]
    centered_increment_gate = gates["binding_vs_marginals"]["passed"] and gates["binding_vs_s"]["passed"]
    raw_increment_gate = gates["raw_vs_marginals"]["passed"] and gates["raw_vs_s"]["passed"]
    if raw_increment_gate and centered_increment_gate and binding_gate:
        decision = "JOINT_AND_CENTERED_BINDING_TASK_INCREMENT_PASS"
    elif raw_increment_gate and centered_increment_gate:
        decision = "JOINT_TASK_INCREMENT_PASS_CENTERED_BINDING_UNCONFIRMED"
    elif raw_increment_gate:
        decision = "JOINT_TASK_INCREMENT_PASS_CENTERED_NO_STABLE_INCREMENT"
    elif centered_increment_gate and binding_gate:
        decision = "CENTERED_BINDING_TASK_INCREMENT_PASS"
    elif centered_increment_gate:
        decision = "CENTERED_TASK_INCREMENT_BINDING_UNCONFIRMED"
    elif binding_gate or raw_gate:
        decision = "BINDING_DETECTED_NO_STABLE_TASK_INCREMENT"
    else:
        decision = "NO_STABLE_BINDING_SIGNAL"

    dimensions = role_dimensions(representation)
    result: dict[str, Any] = {
        "protocol_id": config["protocol_id"],
        "config": config,
        "data": {
            "dataset": data_config["dataset"],
            "selected_graphs": int(unique_indices.size),
            "official_test_encoded_or_evaluated": False,
            "supervised_scope": (
                "official-train-only scaffold folds"
                if split_mode == "scaffold_folds"
                else "official train versus official validation (frozen evaluation)"
            ),
            "split_mode": split_mode,
        },
        "audit_boundary": {
            "frozen_s_sha256": _sha256(frozen_path),
            "scaffold_folds_sha256": _sha256(folds_path),
            "config_sha256": _sha256(config_path),
            "role_uses_attributes": False,
            "shuffle_preserves_patch_marginals": "within-patch row permutation",
        },
        "object_audit": object_audit,
        "representation": {
            "dimensions": dimensions,
            "radius": int(representation["radius"]),
            "degree_bins": int(representation["degree_bins"]),
            "attribute_mode": "compact OGB semantics: atom 48D, bond 13D",
            "centers": "all graph nodes unless max_centers_per_graph is set",
            "normalization": "mean of entity-normalized patch statistics",
            "joint_definition": "P(role,attribute)",
            "binding_definition": "P(role,attribute)-P(role)P(attribute)",
        },
        "n_folds": len(folds),
        "aggregate": aggregate,
        "folds": folds,
        "gates": gates,
        "decision": decision,
        "runtime": {
            "seconds": float(time.perf_counter() - start),
            "python": sys.version,
            "platform": platform.platform(),
        },
    }
    _write_json(result_json, result)
    result_md.parent.mkdir(parents=True, exist_ok=True)
    result_md.write_text(_render_markdown(result), encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    result = run(_resolve(args.config))
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "binding_vs_shuffle": result["gates"]["binding_vs_shuffle"],
                "binding_vs_marginals": result["gates"]["binding_vs_marginals"],
                "runtime_seconds": result["runtime"]["seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
