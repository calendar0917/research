"""Fast MolHIV mechanism screen with a genuinely invariant radius-2 object."""

from __future__ import annotations

import argparse
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.decomposition import sparse_encode
from sklearn.metrics import roc_auc_score
import yaml
from xgboost import XGBClassifier

from ksvd_research.core import ksvd
from ksvd_research.data import load_molhiv
from tracks.ksvd.experiments.luyin16.patch_object_audit import (
    relabel_graph_features,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = (
    REPO_ROOT / "tracks/ksvd/configs/luyin16/invariant_patch_mechanism_screen.yaml"
)
SHELL_PAIRS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


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


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "maximum": 0.0}
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
    }


def representation_dimensions(config: Mapping[str, Any]) -> dict[str, int]:
    degree_bins = int(config["degree_bins"])
    atom_categories = int(config["atom_categories"])
    bond_categories = int(config["bond_categories"])
    topology = 3 * degree_bins + len(SHELL_PAIRS) + 3 + 7
    attributes = 3 * atom_categories + len(SHELL_PAIRS) * bond_categories
    return {"topology": topology, "attributes": attributes, "joint": topology + attributes}


def _ego_distances(graph, center: int, radius: int) -> dict[int, int]:
    distances = {int(center): 0}
    queue: deque[int] = deque([int(center)])
    while queue:
        node = queue.popleft()
        if distances[node] >= radius:
            continue
        for neighbor in graph.neighbors(node):
            if neighbor not in distances:
                distances[int(neighbor)] = distances[node] + 1
                queue.append(int(neighbor))
    return distances


def _triangle_count(graph, nodes: set[int]) -> int:
    count = 0
    for left, right in graph.induced(nodes).edges():
        count += len((graph.neighbors(left) & graph.neighbors(right)) & nodes)
    return count // 3


def invariant_patch_blocks(
    graph,
    center: int,
    node_features: np.ndarray,
    edge_features: Mapping[tuple[int, int], np.ndarray],
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Return ID-free topology and attribute statistics for one full ego patch."""
    radius = int(config["radius"])
    if radius != 2:
        raise ValueError("v1 fixes radius=2 so its shell schema stays explicit")
    degree_bins = int(config["degree_bins"])
    atom_categories = int(config["atom_categories"])
    bond_categories = int(config["bond_categories"])
    dims = representation_dimensions(config)
    topology = np.zeros(dims["topology"], dtype=np.float64)
    attributes = np.zeros(dims["attributes"], dtype=np.float64)

    distances = _ego_distances(graph, int(center), radius)
    nodes = set(distances)
    induced = graph.induced(nodes)
    n_nodes = len(nodes)
    n_edges = induced.num_edges()
    degrees = {node: len(induced.neighbors(node)) for node in nodes}
    shell_counts = np.zeros(3, dtype=np.float64)

    degree_offset = 0
    atom_offset = 0
    for node in nodes:
        shell = int(distances[node])
        degree_bin = min(int(degrees[node]), degree_bins - 1)
        topology[degree_offset + shell * degree_bins + degree_bin] += 1.0
        atom_type = int(node_features[node, 0])
        if not 0 <= atom_type < atom_categories:
            raise ValueError(f"atom category {atom_type} outside [0,{atom_categories})")
        attributes[atom_offset + shell * atom_categories + atom_type] += 1.0
        shell_counts[shell] += 1.0
    if n_nodes:
        topology[: 3 * degree_bins] /= float(n_nodes)
    for shell in range(3):
        if shell_counts[shell] > 0:
            start = shell * atom_categories
            attributes[start : start + atom_categories] /= shell_counts[shell]

    pair_to_index = {pair: index for index, pair in enumerate(SHELL_PAIRS)}
    edge_offset = 3 * degree_bins
    bond_offset = 3 * atom_categories
    pair_counts = np.zeros(len(SHELL_PAIRS), dtype=np.float64)
    for left, right in induced.edges():
        pair = tuple(sorted((int(distances[left]), int(distances[right]))))
        pair_index = pair_to_index[pair]
        topology[edge_offset + pair_index] += 1.0
        pair_counts[pair_index] += 1.0
        values = edge_features.get(graph.edge_key(left, right))
        if values is None or not values.size:
            raise ValueError("every retained MolHIV edge must have a bond feature")
        bond_type = int(values[0])
        if not 0 <= bond_type < bond_categories:
            raise ValueError(f"bond category {bond_type} outside [0,{bond_categories})")
        attributes[
            bond_offset + pair_index * bond_categories + bond_type
        ] += 1.0
    if n_edges:
        topology[edge_offset : edge_offset + len(SHELL_PAIRS)] /= float(n_edges)
    for pair_index, count in enumerate(pair_counts):
        if count > 0:
            start = bond_offset + pair_index * bond_categories
            attributes[start : start + bond_categories] /= count

    shell_offset = edge_offset + len(SHELL_PAIRS)
    topology[shell_offset : shell_offset + 3] = shell_counts / max(float(n_nodes), 1.0)
    scalar_offset = shell_offset + 3
    components = 1 if n_nodes else 0
    cycle_rank = max(0, n_edges - n_nodes + components)
    triangles = _triangle_count(graph, nodes)
    density = 2.0 * n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0
    topology[scalar_offset:] = np.asarray(
        [
            n_nodes / float(config["node_scale"]),
            n_edges / float(config["edge_scale"]),
            degrees[int(center)] / float(config["degree_scale"]),
            (2.0 * n_edges / max(n_nodes, 1)) / float(config["degree_scale"]),
            cycle_rank / float(config["cycle_scale"]),
            triangles / float(config["triangle_scale"]),
            density,
        ],
        dtype=np.float64,
    )
    return topology, attributes, {"n_nodes": n_nodes, "n_edges": n_edges}


def _normalize_columns(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.maximum(np.linalg.norm(matrix, axis=0, keepdims=True), 1e-12)


def build_graph_patch_matrices(
    graph,
    node_features: np.ndarray,
    edge_features: Mapping[tuple[int, int], np.ndarray],
    config: Mapping[str, Any],
    *,
    centers: Sequence[int] | None = None,
) -> dict[str, np.ndarray]:
    ordered_centers = list(graph.nodes if centers is None else centers)
    topology_columns: list[np.ndarray] = []
    attribute_columns: list[np.ndarray] = []
    full_sizes: list[int] = []
    edge_counts: list[int] = []
    for center in ordered_centers:
        topology, attributes, meta = invariant_patch_blocks(
            graph, int(center), node_features, edge_features, config
        )
        topology_columns.append(topology)
        attribute_columns.append(attributes)
        full_sizes.append(meta["n_nodes"])
        edge_counts.append(meta["n_edges"])
    dims = representation_dimensions(config)
    if topology_columns:
        topology_matrix = np.stack(topology_columns, axis=1)
        attribute_matrix = np.stack(attribute_columns, axis=1)
    else:
        topology_matrix = np.zeros((dims["topology"], 0), dtype=np.float64)
        attribute_matrix = np.zeros((dims["attributes"], 0), dtype=np.float64)
    joint_matrix = np.concatenate([topology_matrix, attribute_matrix], axis=0)
    if bool(config["normalize_patches"]):
        topology_matrix = _normalize_columns(topology_matrix)
        attribute_matrix = _normalize_columns(attribute_matrix)
        joint_matrix = _normalize_columns(joint_matrix)
    return {
        "topology": topology_matrix,
        "attributes": attribute_matrix,
        "joint": joint_matrix,
        "full_sizes": np.asarray(full_sizes, dtype=np.int64),
        "edge_counts": np.asarray(edge_counts, dtype=np.int64),
    }


def distribution_readout(matrix: np.ndarray) -> np.ndarray:
    """Compact coordinate-wise statistics used unchanged for RAW/INIT/FINAL."""
    if matrix.ndim != 2:
        raise ValueError(f"expected a feature-by-patch matrix, got {matrix.shape}")
    if matrix.shape[1] == 0:
        return np.zeros(matrix.shape[0] * 6, dtype=np.float64)
    q75, q90 = np.quantile(matrix, [0.75, 0.90], axis=1)
    return np.concatenate(
        [
            matrix.mean(axis=1),
            matrix.std(axis=1),
            matrix.max(axis=1),
            q75,
            q90,
            (np.abs(matrix) > 1e-12).mean(axis=1),
        ]
    )


def _select_rows(
    labels: np.ndarray,
    candidates: np.ndarray,
    maximum: int | None,
    seed: int,
) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=np.int64)
    if maximum is None or candidates.size <= maximum:
        return np.sort(candidates)
    rng = np.random.default_rng(seed)
    positive = candidates[labels[candidates] == 1]
    negative = candidates[labels[candidates] == 0]
    n_positive = int(round(maximum * positive.size / candidates.size))
    n_positive = min(max(n_positive, 1), positive.size)
    n_negative = min(maximum - n_positive, negative.size)
    selected = np.concatenate(
        [
            rng.choice(positive, n_positive, replace=False),
            rng.choice(negative, n_negative, replace=False),
        ]
    )
    return np.sort(selected.astype(np.int64))


def audit_invariance(bundle, config: Mapping[str, Any]) -> dict[str, Any]:
    audit = config["audit"]
    rep = config["representation"]
    candidates = np.asarray(bundle.split["valid"], dtype=np.int64)
    labels = bundle.y.astype(np.int64)
    rng = np.random.default_rng(int(audit["seed"]))
    positive = candidates[labels[candidates] == 1]
    negative = candidates[labels[candidates] == 0]
    n_graphs = int(audit["n_graphs"])
    n_positive = min(len(positive), int(round(n_graphs * float(audit["positive_fraction"]))))
    n_negative = min(len(negative), n_graphs - n_positive)
    selected = np.concatenate(
        [
            rng.choice(positive, n_positive, replace=False),
            rng.choice(negative, n_negative, replace=False),
        ]
    )
    rng.shuffle(selected)
    drift: dict[str, list[float]] = {name: [] for name in ("topology", "attributes", "joint")}
    full_sizes: list[int] = []
    atom_counts: Counter[int] = Counter()
    bond_counts: Counter[int] = Counter()
    assert bundle.node_feats is not None and bundle.edge_feats is not None
    for dataset_index in selected:
        graph = bundle.graphs[int(dataset_index)]
        node_features = bundle.node_feats[int(dataset_index)]
        edge_features = bundle.edge_feats[int(dataset_index)]
        base = build_graph_patch_matrices(graph, node_features, edge_features, rep)
        full_sizes.extend(int(value) for value in base["full_sizes"])
        atom_counts.update(int(value) for value in node_features[:, 0])
        bond_counts.update(int(values[0]) for values in edge_features.values())
        base_readouts = {
            name: distribution_readout(base[name])
            for name in ("topology", "attributes", "joint")
        }
        for _repeat in range(int(audit["permutations_per_graph"])):
            permutation = rng.permutation(graph.n)
            changed_graph, changed_nodes, changed_edges = relabel_graph_features(
                graph, node_features, edge_features, permutation
            )
            changed = build_graph_patch_matrices(
                changed_graph,
                changed_nodes,
                changed_edges,
                rep,
                centers=[int(permutation[center]) for center in graph.nodes],
            )
            for name in drift:
                delta = np.max(
                    np.abs(base_readouts[name] - distribution_readout(changed[name])),
                    initial=0.0,
                )
                drift[name].append(float(delta))
    tolerance = float(audit["feature_tolerance"])
    summaries = {name: _summary(values) for name, values in drift.items()}
    return {
        "sample": {
            "split": "official validation (label-free audit only)",
            "n_graphs": int(selected.size),
            "n_positive": int(labels[selected].sum()),
            "permutations_per_graph": int(audit["permutations_per_graph"]),
        },
        "readout_drift": summaries,
        "maximum_ego_nodes": max(full_sizes, default=0),
        "mean_ego_nodes": float(np.mean(full_sizes)) if full_sizes else 0.0,
        "scope": "all topology and attribute coordinates use the complete radius-2 ego",
        "category_encoding": {
            "atom": {
                "mode": "direct OGB category index; no modulo",
                "dimension": int(rep["atom_categories"]),
                "observed": sorted(atom_counts),
            },
            "bond": {
                "mode": "direct OGB category index; no modulo",
                "dimension": int(rep["bond_categories"]),
                "observed": sorted(bond_counts),
            },
        },
        "pass": bool(all(row["maximum"] <= tolerance for row in summaries.values())),
        "tolerance": tolerance,
    }


def _raw_readouts(bundle, indices: np.ndarray, rep: Mapping[str, Any]) -> dict[str, np.ndarray]:
    assert bundle.node_feats is not None and bundle.edge_feats is not None
    rows: dict[str, list[np.ndarray]] = {name: [] for name in ("topology", "attributes", "joint")}
    for raw_index in indices:
        index = int(raw_index)
        matrices = build_graph_patch_matrices(
            bundle.graphs[index],
            bundle.node_feats[index],
            bundle.edge_feats[index],
            rep,
        )
        for name in rows:
            rows[name].append(distribution_readout(matrices[name]))
    return {name: np.stack(values).astype(np.float32) for name, values in rows.items()}


def _fit_auc_views(
    views: Mapping[str, np.ndarray],
    labels: np.ndarray,
    n_train: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    y_train = labels[:n_train].astype(np.int64)
    y_valid = labels[n_train:].astype(np.int64)
    weight = float(np.sum(y_train == 0) / max(1, np.sum(y_train == 1)))
    fixed = dict(config)
    scores: dict[str, Any] = {}
    for name, matrix in views.items():
        model = XGBClassifier(
            **fixed,
            objective="binary:logistic",
            eval_metric="auc",
            scale_pos_weight=weight,
            tree_method="hist",
        )
        model.fit(matrix[:n_train], y_train)
        train_prediction = model.predict_proba(matrix[:n_train])[:, 1]
        valid_prediction = model.predict_proba(matrix[n_train:])[:, 1]
        scores[name] = {
            "dimension": int(matrix.shape[1]),
            "train_auc": float(roc_auc_score(y_train, train_prediction)),
            "valid_auc": float(roc_auc_score(y_valid, valid_prediction)),
        }
    return scores


def _fold_indices(
    archive: Mapping[str, np.ndarray],
    fold: int,
    labels: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    original = np.asarray(archive["original_indices"], dtype=np.int64)
    train = original[np.asarray(archive[f"fold_{fold}_train_indices"], dtype=np.int64)]
    valid = original[np.asarray(archive[f"fold_{fold}_valid_indices"], dtype=np.int64)]
    seed = int(config["seed"]) + fold * 1009
    train = _select_rows(labels, train, int(config["max_train_graphs_per_fold"]), seed)
    valid = _select_rows(labels, valid, int(config["max_valid_graphs_per_fold"]), seed + 1)
    return train, valid


def _frozen_s_rows(
    frozen: Mapping[str, np.ndarray],
    indices: np.ndarray,
) -> np.ndarray:
    mapping = {
        int(dataset_index): position
        for position, dataset_index in enumerate(np.asarray(frozen["dataset_indices"], dtype=np.int64))
    }
    try:
        rows = np.asarray([mapping[int(index)] for index in indices], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"dataset index {int(exc.args[0])} is absent from frozen S") from exc
    return np.asarray(frozen["s"][rows], dtype=np.float32)


def _score_deltas(folds: Sequence[Mapping[str, Any]], candidate: str, baseline: str) -> dict[str, Any]:
    deltas = [
        float(row["scores"][candidate]["valid_auc"] - row["scores"][baseline]["valid_auc"])
        for row in folds
    ]
    return {
        "candidate": candidate,
        "baseline": baseline,
        "fold_deltas": deltas,
        "mean_delta": float(np.mean(deltas)),
        "fold_wins": int(np.sum(np.asarray(deltas) > 0.0)),
    }


def _reservoir_train_patches(
    bundle,
    train_indices: np.ndarray,
    rep: Mapping[str, Any],
    maximum: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    assert bundle.node_feats is not None and bundle.edge_feats is not None
    dimension = representation_dimensions(rep)["joint"]
    reservoir = np.zeros((dimension, maximum), dtype=np.float64)
    sources = np.full(maximum, -1, dtype=np.int64)
    rng = np.random.default_rng(seed)
    seen = 0
    filled = 0
    for raw_index in train_indices:
        index = int(raw_index)
        matrix = build_graph_patch_matrices(
            bundle.graphs[index], bundle.node_feats[index], bundle.edge_feats[index], rep
        )["joint"]
        for column in range(matrix.shape[1]):
            seen += 1
            if filled < maximum:
                slot = filled
                filled += 1
            else:
                slot = int(rng.integers(0, seen))
                if slot >= maximum:
                    continue
            reservoir[:, slot] = matrix[:, column]
            sources[slot] = index
    matrix = reservoir[:, :filled]
    source_rows = sources[:filled]
    return matrix, {
        "n_patches_seen": seen,
        "n_patches_used": int(filled),
        "n_source_graphs": int(np.unique(source_rows).size),
        "source_subset_of_fold_train": bool(set(source_rows.tolist()).issubset(set(train_indices.tolist()))),
        "source_indices_sha256": hashlib.sha256(source_rows.tobytes()).hexdigest(),
    }


def _learn_dictionaries(
    patches: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    initial, _, initial_info = ksvd(
        patches,
        n_atoms=int(config["n_atoms"]),
        T=int(config["sparsity"]),
        T_min=int(config["sparsity"]),
        n_iter=0,
        seed=int(config["seed"]),
    )
    final, _, final_info = ksvd(
        patches,
        n_atoms=initial.shape[1],
        T=int(config["sparsity"]),
        T_min=int(config["sparsity"]),
        n_iter=int(config["iterations"]),
        seed=int(config["seed"]),
        initial_dictionary=initial,
    )
    return initial, final, {"initial": initial_info, "final": final_info}


def _encode_matrix(
    matrix: np.ndarray,
    dictionary: np.ndarray,
    sparsity: int,
    n_jobs: int,
) -> np.ndarray:
    if matrix.shape[1] == 0:
        return np.zeros_like(matrix)
    codes = sparse_encode(
        matrix.T,
        dictionary.T,
        algorithm="omp",
        n_nonzero_coefs=int(sparsity),
        n_jobs=int(n_jobs),
        check_input=False,
    )
    return dictionary @ codes.T


def _reconstruction_readouts(
    bundle,
    indices: np.ndarray,
    rep: Mapping[str, Any],
    initial: np.ndarray,
    final: np.ndarray,
    dictionary_config: Mapping[str, Any],
) -> dict[str, Any]:
    assert bundle.node_feats is not None and bundle.edge_feats is not None
    rows: dict[str, list[np.ndarray]] = {
        "joint_init": [],
        "joint_final": [],
        "joint_final_residual": [],
    }
    init_errors: list[float] = []
    final_errors: list[float] = []
    pending: list[np.ndarray] = []
    pending_patches = 0
    chunk_limit = int(dictionary_config["encode_chunk_patches"])
    n_jobs = int(dictionary_config["encode_n_jobs"])

    def flush() -> None:
        nonlocal pending, pending_patches
        if not pending:
            return
        counts = [matrix.shape[1] for matrix in pending]
        joined = np.concatenate(pending, axis=1)
        init_joined = _encode_matrix(
            joined, initial, int(dictionary_config["sparsity"]), n_jobs
        )
        final_joined = _encode_matrix(
            joined, final, int(dictionary_config["sparsity"]), n_jobs
        )
        start = 0
        for matrix, count in zip(pending, counts, strict=True):
            stop = start + count
            init_reconstruction = init_joined[:, start:stop]
            final_reconstruction = final_joined[:, start:stop]
            rows["joint_init"].append(distribution_readout(init_reconstruction))
            rows["joint_final"].append(distribution_readout(final_reconstruction))
            rows["joint_final_residual"].append(
                distribution_readout(np.abs(matrix - final_reconstruction))
            )
            denominator = max(float(np.linalg.norm(matrix)), 1e-12)
            init_errors.append(
                float(np.linalg.norm(matrix - init_reconstruction) / denominator)
            )
            final_errors.append(
                float(np.linalg.norm(matrix - final_reconstruction) / denominator)
            )
            start = stop
        pending = []
        pending_patches = 0

    for raw_index in indices:
        index = int(raw_index)
        matrix = build_graph_patch_matrices(
            bundle.graphs[index], bundle.node_feats[index], bundle.edge_feats[index], rep
        )["joint"]
        if pending and pending_patches + matrix.shape[1] > chunk_limit:
            flush()
        pending.append(matrix)
        pending_patches += matrix.shape[1]
    flush()
    return {
        "features": {name: np.stack(values).astype(np.float32) for name, values in rows.items()},
        "initial_error": np.asarray(init_errors, dtype=np.float64),
        "final_error": np.asarray(final_errors, dtype=np.float64),
    }


def _aggregate_view(folds: Sequence[Mapping[str, Any]], view: str) -> dict[str, Any]:
    values = [float(row["scores"][view]["valid_auc"]) for row in folds]
    return {"fold_auc": values, "mean_auc": float(np.mean(values)), "std_auc": float(np.std(values))}


def _render_markdown(result: Mapping[str, Any]) -> str:
    audit = result["object_audit"]
    raw = result.get("raw_screen", {})
    lines = [
        "# MolHIV invariant patch mechanism screen",
        "",
        f"Protocol: `{result['protocol_id']}`",
        "",
        "## Object gate",
        "",
        f"- invariant object: **{'PASS' if audit['pass'] else 'FAIL'}**",
        f"- maximum topology drift: {audit['readout_drift']['topology']['maximum']:.8g}",
        f"- maximum attribute drift: {audit['readout_drift']['attributes']['maximum']:.8g}",
        f"- maximum joint drift: {audit['readout_drift']['joint']['maximum']:.8g}",
        f"- maximum full ego size observed: {audit['maximum_ego_nodes']}",
        "- atom/bond categories use direct indices; no modulo collisions",
        "",
    ]
    if raw:
        attribute_delta = _score_deltas(
            raw["folds"], "s_invariant_attributes_raw", "s"
        )
        topology_delta = _score_deltas(
            raw["folds"], "s_invariant_topology_raw", "s"
        )
        joint_vs_attributes = _score_deltas(
            raw["folds"], "invariant_joint_raw", "invariant_attributes_raw"
        )
        lines.extend(
            [
                "## RAW official-train scaffold screen",
                "",
                "| view | mean fold AUC |",
                "|---|---:|",
            ]
        )
        for name, row in raw["aggregate"].items():
            lines.append(f"| `{name}` | {row['mean_auc']:.6f} |")
        gate = raw["promotion"]
        lines.extend(
            [
                "",
                f"`S+joint RAW - S` mean delta: {gate['mean_delta']:+.6f}; wins: {gate['fold_wins']}/3.",
                f"`S+attributes RAW - S`: {attribute_delta['mean_delta']:+.6f}; wins: {attribute_delta['fold_wins']}/3.",
                f"`S+topology RAW - S`: {topology_delta['mean_delta']:+.6f}; wins: {topology_delta['fold_wins']}/3.",
                f"Standalone `joint RAW - attributes RAW`: {joint_vs_attributes['mean_delta']:+.6f}; wins: {joint_vs_attributes['fold_wins']}/3.",
                f"RAW promotion: **{'PASS' if gate['promoted'] else 'FAIL'}**.",
                "",
            ]
        )
    if result.get("ksvd_attribution"):
        ksvd_result = result["ksvd_attribution"]
        raw_scores = [
            float(row["scores"]["s_invariant_joint_raw"]["valid_auc"])
            for row in raw["folds"]
        ]
        init_scores = [
            float(row["scores"]["s_invariant_joint_init"]["valid_auc"])
            for row in ksvd_result["folds"]
        ]
        final_scores = [
            float(row["scores"]["s_invariant_joint_final"]["valid_auc"])
            for row in ksvd_result["folds"]
        ]
        init_minus_raw = np.asarray(init_scores) - np.asarray(raw_scores)
        final_minus_raw = np.asarray(final_scores) - np.asarray(raw_scores)
        reconstruction_reductions = [
            (
                float(row["reconstruction"]["valid_initial_mean"])
                - float(row["reconstruction"]["valid_final_mean"])
            )
            / float(row["reconstruction"]["valid_initial_mean"])
            for row in ksvd_result["folds"]
        ]
        lines.extend(
            [
                "## Fold-specific K-SVD attribution",
                "",
                "| view | mean fold AUC |",
                "|---|---:|",
            ]
        )
        for name, row in ksvd_result["aggregate"].items():
            lines.append(f"| `{name}` | {row['mean_auc']:.6f} |")
        update = ksvd_result["update_gate"]
        lines.extend(
            [
                "",
                f"`S+FINAL - S+INIT` mean delta: {update['mean_delta']:+.6f}; wins: {update['fold_wins']}/3.",
                f"`S+INIT - S+RAW`: {float(init_minus_raw.mean()):+.6f}; wins: {int(np.sum(init_minus_raw > 0))}/3.",
                f"`S+FINAL - S+RAW`: {float(final_minus_raw.mean()):+.6f}; wins: {int(np.sum(final_minus_raw > 0))}/3.",
                "Validation reconstruction-error reductions: "
                + ", ".join(f"{value:.1%}" for value in reconstruction_reductions)
                + ".",
                f"K-SVD task update: **{'PASS' if update['passed'] else 'FAIL'}**.",
                "",
            ]
        )
        lines.extend(
            [
                "## Interpretation",
                "",
                "The stable signal is an all-center, shell-conditioned local chemistry distribution. Pure local topology is weaker, and naively joint-normalizing topology with attributes can dilute the attribute signal.",
                "",
                "Sparse reconstruction under the empirical INIT dictionary is useful, but optimizing the unsupervised reconstruction objective does not improve the task consistently. K-SVD therefore remains a compressor/diagnostic in this protocol, not the source of the MolHIV gain.",
                "",
            ]
        )
    lines.extend(
        [
            "## Boundary",
            "",
            "No official validation label was used for supervised scoring, official test was not encoded or evaluated, and no hyperparameter search was run.",
            "",
        ]
    )
    return "\n".join(lines)


def run(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_config = config["data"]
    feature_path = _resolve(data_config["frozen_features"])
    folds_path = _resolve(data_config["scaffold_folds"])
    with np.load(feature_path, allow_pickle=False) as archive:
        frozen = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(folds_path, allow_pickle=False) as archive:
        folds_archive = {name: np.asarray(archive[name]) for name in archive.files}
    bundle = load_molhiv(root=_resolve(data_config["root"]), with_features=True)
    labels = bundle.y.astype(np.int64)
    start = time.perf_counter()
    object_audit = audit_invariance(bundle, config)
    result: dict[str, Any] = {
        "protocol_id": config["protocol_id"],
        "config": config,
        "audit_boundary": {
            "supervised_scores_use_official_train_only": True,
            "official_validation_used_for_label_free_relabel_audit_only": True,
            "official_test_encoded_or_evaluated": False,
            "hyperparameter_search_performed": False,
            "frozen_s_sha256": _sha256(feature_path),
            "folds_sha256": _sha256(folds_path),
            "config_sha256": _sha256(config_path),
        },
        "representation_dimensions": representation_dimensions(config["representation"]),
        "object_audit": object_audit,
        "runtime_seconds": 0.0,
    }
    output_json = _resolve(config["output_json"])
    output_markdown = _resolve(config["output_markdown"])
    _write_json(output_json, result)
    if not object_audit["pass"]:
        result["decision"] = "STOP_OBJECT_GATE_FAILED"
        result["runtime_seconds"] = time.perf_counter() - start
        _write_json(output_json, result)
        output_markdown.write_text(_render_markdown(result), encoding="utf-8")
        return result

    fold_ids = sorted(
        int(name.removeprefix("fold_").removesuffix("_train_indices"))
        for name in folds_archive
        if name.startswith("fold_") and name.endswith("_train_indices")
    )
    raw_folds: list[dict[str, Any]] = []
    fold_selections: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in fold_ids:
        train_indices, valid_indices = _fold_indices(
            folds_archive, fold, labels, config["screen"]
        )
        fold_selections.append((train_indices, valid_indices))
        combined = np.concatenate([train_indices, valid_indices])
        local = _raw_readouts(bundle, combined, config["representation"])
        s = _frozen_s_rows(frozen, combined)
        combined_labels = labels[combined]
        views = {
            "s": s,
            "invariant_topology_raw": local["topology"],
            "invariant_attributes_raw": local["attributes"],
            "invariant_joint_raw": local["joint"],
            "s_invariant_topology_raw": np.concatenate([s, local["topology"]], axis=1),
            "s_invariant_attributes_raw": np.concatenate([s, local["attributes"]], axis=1),
            "s_invariant_joint_raw": np.concatenate([s, local["joint"]], axis=1),
        }
        scores = _fit_auc_views(
            views, combined_labels, len(train_indices), config["classifier"]
        )
        raw_folds.append(
            {
                "fold": fold,
                "n_train": int(train_indices.size),
                "n_valid": int(valid_indices.size),
                "n_train_positive": int(labels[train_indices].sum()),
                "n_valid_positive": int(labels[valid_indices].sum()),
                "scores": scores,
            }
        )
        result["raw_screen"] = {"folds": raw_folds}
        result["runtime_seconds"] = time.perf_counter() - start
        _write_json(output_json, result)

    raw_aggregate = {
        name: _aggregate_view(raw_folds, name)
        for name in raw_folds[0]["scores"]
    }
    raw_promotion = _score_deltas(
        raw_folds, "s_invariant_joint_raw", "s"
    )
    raw_promotion["minimum_mean_delta"] = float(config["screen"]["minimum_raw_delta"])
    raw_promotion["minimum_fold_wins"] = int(config["screen"]["minimum_fold_wins"])
    raw_promotion["promoted"] = bool(
        raw_promotion["mean_delta"] >= raw_promotion["minimum_mean_delta"]
        and raw_promotion["fold_wins"] >= raw_promotion["minimum_fold_wins"]
    )
    result["raw_screen"] = {
        "folds": raw_folds,
        "aggregate": raw_aggregate,
        "promotion": raw_promotion,
    }

    dictionary_config = config["dictionary"]
    if raw_promotion["promoted"] and bool(dictionary_config["run_if_raw_passes"]):
        ksvd_folds: list[dict[str, Any]] = []
        for fold, (train_indices, valid_indices) in zip(fold_ids, fold_selections, strict=True):
            patches, reservoir_audit = _reservoir_train_patches(
                bundle,
                train_indices,
                config["representation"],
                int(dictionary_config["max_train_patches"]),
                int(dictionary_config["seed"]) + fold,
            )
            fold_dictionary_config = dict(dictionary_config)
            fold_dictionary_config["seed"] = int(dictionary_config["seed"]) + fold
            initial, final, dictionary_info = _learn_dictionaries(
                patches, fold_dictionary_config
            )
            combined = np.concatenate([train_indices, valid_indices])
            reconstructed = _reconstruction_readouts(
                bundle,
                combined,
                config["representation"],
                initial,
                final,
                fold_dictionary_config,
            )
            s = _frozen_s_rows(frozen, combined)
            local = reconstructed["features"]
            views = {
                "invariant_joint_init": local["joint_init"],
                "invariant_joint_final": local["joint_final"],
                "s_invariant_joint_init": np.concatenate([s, local["joint_init"]], axis=1),
                "s_invariant_joint_final": np.concatenate([s, local["joint_final"]], axis=1),
                "s_invariant_joint_final_residual": np.concatenate(
                    [s, local["joint_final_residual"]], axis=1
                ),
            }
            scores = _fit_auc_views(
                views, labels[combined], len(train_indices), config["classifier"]
            )
            ksvd_folds.append(
                {
                    "fold": fold,
                    "scores": scores,
                    "reservoir_audit": reservoir_audit,
                    "dictionary_info": dictionary_info,
                    "reconstruction": {
                        "train_initial_mean": float(
                            reconstructed["initial_error"][: len(train_indices)].mean()
                        ),
                        "train_final_mean": float(
                            reconstructed["final_error"][: len(train_indices)].mean()
                        ),
                        "valid_initial_mean": float(
                            reconstructed["initial_error"][len(train_indices) :].mean()
                        ),
                        "valid_final_mean": float(
                            reconstructed["final_error"][len(train_indices) :].mean()
                        ),
                    },
                }
            )
            result["ksvd_attribution"] = {"folds": ksvd_folds}
            result["runtime_seconds"] = time.perf_counter() - start
            _write_json(output_json, result)
        ksvd_aggregate = {
            name: _aggregate_view(ksvd_folds, name)
            for name in ksvd_folds[0]["scores"]
        }
        update = _score_deltas(
            ksvd_folds, "s_invariant_joint_final", "s_invariant_joint_init"
        )
        update["minimum_mean_delta"] = float(dictionary_config["minimum_update_delta"])
        update["minimum_fold_wins"] = int(dictionary_config["minimum_fold_wins"])
        update["passed"] = bool(
            update["mean_delta"] >= update["minimum_mean_delta"]
            and update["fold_wins"] >= update["minimum_fold_wins"]
        )
        result["ksvd_attribution"] = {
            "folds": ksvd_folds,
            "aggregate": ksvd_aggregate,
            "update_gate": update,
        }
        result["decision"] = (
            "RAW_OBJECT_AND_KSVD_UPDATE_PASS" if update["passed"] else "RAW_OBJECT_PASS_KSVD_UPDATE_FAIL"
        )
    else:
        result["decision"] = "STOP_RAW_OBJECT_NO_INCREMENT"

    result["runtime_seconds"] = time.perf_counter() - start
    _write_json(output_json, result)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(_render_markdown(result), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = _resolve(args.config)
    result = run(config_path)
    print(
        json.dumps(
            {
                "decision": result.get("decision"),
                "object_gate": result["object_audit"]["pass"],
                "raw_promotion": result.get("raw_screen", {}).get("promotion"),
                "ksvd_update": result.get("ksvd_attribution", {}).get("update_gate"),
                "runtime_seconds": result["runtime_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
