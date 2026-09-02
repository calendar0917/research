"""Independent reproduction of the mentor's fixed-feature MolHIV idea.

This runner intentionally reproduces the *research structure*, not the
unavailable 69/624-dimensional upstream implementation:

S_v1
    Explicit graph topology plus normalized OGB atom/bond composition.
T_v1
    Rich graph-level sparse-code readout under one train-only initial
    dictionary and the K-SVD-updated dictionary from the same initialization.

The default protocol evaluates train/validation only.  Official test
evaluation requires an explicit ``--evaluate-test`` flag after the feature
view and classifier have been frozen on validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
from collections import deque
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from ksvd_research.core import ksvd
from ksvd_research.data import MolhivBundle, load_molhiv
from ksvd_research.evaluation.graph_level import (
    GraphLevelConfig,
    bundle_to_Y,
    sample_patches_B0,
    sample_patches_graph_level,
    sparse_code_patch_matrix,
    sparse_code_readouts,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = REPO_ROOT / "tracks/ksvd/configs/luyin16/mentor_concept_v1.yaml"
DEFAULT_RESULT_DIR = REPO_ROOT / "tracks/ksvd/results/luyin16/mentor_concept_v1"

TOPOLOGY_FEATURE_NAMES = (
    "n_nodes",
    "log1p_n_nodes",
    "n_edges",
    "log1p_n_edges",
    "density",
    "degree_mean",
    "degree_std",
    "degree_max",
    "degree_q25",
    "degree_median",
    "degree_q75",
    "fraction_isolated",
    "fraction_leaves",
    "fraction_degree_ge_3",
    "triangle_count",
    "log1p_triangle_count",
    "connected_components",
    "cycle_rank",
)

ALLOWED_VIEWS = (
    "s",
    "t_init",
    "t_final",
    "s_t_init",
    "s_t_final",
    "r_raw",
    "r_init",
    "r_final",
    "s_r_raw",
    "s_r_init",
    "s_r_final",
)

TYPED_RECONSTRUCTION_BLOCKS = (
    "signed_mean",
    "std",
    "minimum",
    "maximum",
    "q10",
    "q25",
    "q50",
    "q75",
    "q90",
    "mean_absolute",
    "root_mean_square",
    "nonzero_fraction",
)


def experiment_status(protocol_id: str, max_graphs: int | None) -> str:
    if max_graphs is None:
        return "full"
    if protocol_id.endswith("-smoke"):
        return "smoke"
    return "development"


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must be a YAML mapping: {path}")
    for key in ("protocol_id", "data", "sampler", "dictionary", "classifier", "views"):
        if key not in payload:
            raise KeyError(f"missing configuration key: {key}")
    views = list(payload["views"])
    unknown = sorted(set(views) - set(ALLOWED_VIEWS))
    if unknown:
        raise ValueError(f"unknown views {unknown}; expected a subset of {ALLOWED_VIEWS}")
    if not views:
        raise ValueError("at least one feature view is required")
    return payload


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
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
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _index_hash(indices: Iterable[int]) -> str:
    values = np.asarray(list(indices), dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def runtime_environment() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "ogb", "scikit-learn", "xgboost"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "git_head": git_head,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def _connected_components(graph) -> int:
    remaining = set(graph.nodes)
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            node = stack.pop()
            unseen = set(graph.neighbors(node)) & remaining
            remaining.difference_update(unseen)
            stack.extend(unseen)
    return count


def _triangle_count(graph) -> int:
    count_by_edges = 0
    for left, right in graph.edges():
        count_by_edges += len(set(graph.neighbors(left)) & set(graph.neighbors(right)))
    return count_by_edges // 3


def _topology_row(graph) -> np.ndarray:
    n_nodes = int(graph.n)
    n_edges = int(graph.num_edges())
    degrees = np.asarray([len(graph.neighbors(node)) for node in graph.nodes], dtype=np.float64)
    if degrees.size:
        degree_mean = float(degrees.mean())
        degree_std = float(degrees.std())
        degree_max = float(degrees.max())
        degree_q25, degree_median, degree_q75 = np.quantile(degrees, [0.25, 0.50, 0.75])
        fraction_isolated = float(np.mean(degrees == 0))
        fraction_leaves = float(np.mean(degrees == 1))
        fraction_branch = float(np.mean(degrees >= 3))
    else:
        degree_mean = degree_std = degree_max = 0.0
        degree_q25 = degree_median = degree_q75 = 0.0
        fraction_isolated = fraction_leaves = fraction_branch = 0.0
    components = _connected_components(graph) if n_nodes else 0
    triangles = _triangle_count(graph)
    density = (2.0 * n_edges / (n_nodes * (n_nodes - 1))) if n_nodes > 1 else 0.0
    cycle_rank = max(0, n_edges - n_nodes + components)
    return np.asarray(
        [
            n_nodes,
            math.log1p(n_nodes),
            n_edges,
            math.log1p(n_edges),
            density,
            degree_mean,
            degree_std,
            degree_max,
            degree_q25,
            degree_median,
            degree_q75,
            fraction_isolated,
            fraction_leaves,
            fraction_branch,
            triangles,
            math.log1p(triangles),
            components,
            cycle_rank,
        ],
        dtype=np.float32,
    )


def _normalized_categorical_histogram(values: np.ndarray, dimensions: Sequence[int]) -> np.ndarray:
    if values.ndim != 2 or values.shape[1] != len(dimensions):
        raise ValueError(
            f"categorical feature shape {values.shape} does not match {len(dimensions)} channels"
        )
    blocks: list[np.ndarray] = []
    for channel, dimension in enumerate(dimensions):
        column = values[:, channel].astype(np.int64, copy=False)
        if column.size and (column.min() < 0 or column.max() >= int(dimension)):
            raise ValueError(f"channel {channel} has category outside [0,{int(dimension) - 1}]")
        hist = np.bincount(column, minlength=int(dimension)).astype(np.float32)
        if hist.sum() > 0:
            hist /= hist.sum()
        blocks.append(hist)
    return np.concatenate(blocks) if blocks else np.empty(0, dtype=np.float32)


def build_composition_features(
    bundle: MolhivBundle,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Build the explicit S_v1 schema without labels or split-dependent fitting."""
    if bundle.node_feats is None or bundle.edge_feats is None:
        raise ValueError("S_v1 requires load_molhiv(..., with_features=True)")
    from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

    atom_dims = [int(value) for value in get_atom_feature_dims()]
    bond_dims = [int(value) for value in get_bond_feature_dims()]
    names = list(TOPOLOGY_FEATURE_NAMES)
    names.extend(
        f"atom_channel_{channel}_category_{category}"
        for channel, dimension in enumerate(atom_dims)
        for category in range(dimension)
    )
    names.extend(
        f"bond_channel_{channel}_category_{category}"
        for channel, dimension in enumerate(bond_dims)
        for category in range(dimension)
    )

    rows: list[np.ndarray] = []
    for graph, node_features, edge_features in zip(
        bundle.graphs, bundle.node_feats, bundle.edge_feats, strict=True
    ):
        if edge_features:
            bond_values = np.stack(list(edge_features.values()), axis=0)
        else:
            bond_values = np.empty((0, len(bond_dims)), dtype=np.int64)
        row = np.concatenate(
            [
                _topology_row(graph),
                _normalized_categorical_histogram(node_features, atom_dims),
                _normalized_categorical_histogram(bond_values, bond_dims),
            ]
        ).astype(np.float32, copy=False)
        rows.append(row)
    matrix = np.stack(rows, axis=0)
    if matrix.shape[1] != len(names) or not np.all(np.isfinite(matrix)):
        raise RuntimeError("invalid S_v1 feature matrix")
    meta = {
        "name": "S_v1",
        "dimension": int(matrix.shape[1]),
        "topology_dimension": len(TOPOLOGY_FEATURE_NAMES),
        "atom_dimensions": atom_dims,
        "bond_dimensions": bond_dims,
        "fit_scope": "label-free per-graph; no fitted parameters",
    }
    return matrix, names, meta


def graph_level_config(config: Mapping[str, Any]) -> GraphLevelConfig:
    sampler = dict(config["sampler"])
    dictionary = dict(config["dictionary"])
    return GraphLevelConfig(
        p=float(sampler["p"]),
        q=float(sampler["q"]),
        walk_length=int(sampler["walk_length"]),
        max_nodes=int(sampler["max_nodes"]),
        max_walks=int(sampler["max_walks"]),
        edge_decay=float(sampler["edge_decay"]),
        seed_policy=str(sampler["seed_policy"]),
        cover_target=(None if sampler.get("cover_target") is None else float(sampler["cover_target"])),
        no_backtrack=bool(sampler["no_backtrack"]),
        n_atoms=int(dictionary["n_atoms"]),
        T=int(dictionary["sparsity"]),
        T_min=int(dictionary["minimum_sparsity"]),
        ksvd_iter=int(dictionary["iterations"]),
        seed=int(dictionary["seed"]),
        max_train_patches=int(dictionary["max_train_patches"]),
        patch_feat=str(sampler["patch_feat"]),
        normalize_patches=bool(sampler["normalize_patches"]),
        max_patches_per_graph=(
            None if sampler.get("max_patches_per_graph") is None else int(sampler["max_patches_per_graph"])
        ),
        radius=int(sampler.get("radius", 2)),
        allow_overcomplete=bool(dictionary.get("allow_overcomplete", False)),
    )


def _radius_ego_nodes(graph, center: int, radius: int) -> set[int]:
    """Return the induced radius-r node set around one atom deterministically."""
    if radius < 0:
        raise ValueError("radius must be non-negative")
    distances = {int(center): 0}
    queue: deque[int] = deque([int(center)])
    while queue:
        node = queue.popleft()
        if distances[node] >= radius:
            continue
        for neighbor in sorted(graph.neighbors(node)):
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    return set(distances)


def _rooted_bfs_order(graph, nodes: set[int], center: int) -> list[int]:
    """Order an ego patch with its center first, then a deterministic BFS."""
    if center not in nodes:
        raise ValueError("center must belong to the patch")
    order: list[int] = []
    seen = {center}
    queue: deque[int] = deque([center])
    while queue:
        node = queue.popleft()
        order.append(node)
        neighbors = sorted(
            (neighbor for neighbor in graph.neighbors(node) if neighbor in nodes),
            key=lambda neighbor: (-len(graph.neighbors(neighbor) & nodes), neighbor),
        )
        for neighbor in neighbors:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    for node in sorted(nodes):
        if node not in seen:
            seen.add(node)
            order.append(node)
    return order


def _radius2_rooted_chem_vector(
    graph,
    nodes: set[int],
    center: int,
    max_nodes: int,
    node_features: np.ndarray,
    edge_features: Mapping[tuple[int, int], np.ndarray],
) -> np.ndarray:
    """52-D chem patch vector with an explicit center-preserving node order.

    The historical ``chem`` schema is 28 padded upper-triangular adjacency
    coordinates plus 16 atom-type and 8 bond-type histograms.  Radius-2
    sampling changes only the patch substrate; keeping this width makes the
    624-D (12 x 52) typed readout directly comparable to the earlier proxy.
    """
    order = _rooted_bfs_order(graph, nodes, center)[:max_nodes]
    position = {node: index for index, node in enumerate(order)}
    adjacency = np.zeros((max_nodes, max_nodes), dtype=np.float64)
    for left, right in graph.induced(set(order)).edges():
        i, j = position[left], position[right]
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0
    upper = [adjacency[i, j] for i in range(max_nodes) for j in range(i + 1, max_nodes)]

    atom_hist = np.zeros(16, dtype=np.float64)
    for node in nodes:
        atom_hist[int(node_features[node, 0]) % 16] += 1.0
    if atom_hist.sum() > 0:
        atom_hist /= atom_hist.sum()
    bond_hist = np.zeros(8, dtype=np.float64)
    for left, right in graph.induced(nodes).edges():
        key = (left, right) if left < right else (right, left)
        values = edge_features.get(key)
        if values is not None and values.size:
            bond_hist[int(values[0]) % 8] += 1.0
    if bond_hist.sum() > 0:
        bond_hist /= bond_hist.sum()
    vector = np.concatenate([np.asarray(upper), atom_hist, bond_hist]).astype(np.float64)
    if vector.shape != (52,):
        raise AssertionError(f"radius-2 chem vector changed dimension: {vector.shape}")
    return vector


def _radius2_rooted_topology_vector(
    graph,
    nodes: set[int],
    center: int,
    max_nodes: int,
) -> np.ndarray:
    """28-D center-preserving adjacency vector for a pure-topology control."""
    order = _rooted_bfs_order(graph, nodes, center)[:max_nodes]
    position = {node: index for index, node in enumerate(order)}
    adjacency = np.zeros((max_nodes, max_nodes), dtype=np.float64)
    for left, right in graph.induced(set(order)).edges():
        i, j = position[left], position[right]
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0
    vector = np.asarray(
        [adjacency[i, j] for i in range(max_nodes) for j in range(i + 1, max_nodes)],
        dtype=np.float64,
    )
    if vector.shape != (28,):
        raise AssertionError(f"radius-2 topology vector changed dimension: {vector.shape}")
    return vector


def vectorize_graphs(
    bundle: MolhivBundle,
    cfg: GraphLevelConfig,
    mode: str,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Sample every graph once so INIT and FINAL use identical patches."""
    if bundle.node_feats is None or bundle.edge_feats is None:
        raise ValueError("typed patch vectors require node and edge features")
    matrices: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for graph_index, graph in enumerate(bundle.graphs):
        local_seed = cfg.seed + graph_index * 13
        if mode == "B0":
            sampled = sample_patches_B0(graph, cfg.max_nodes, local_seed)
            sample_meta = {
                "n_walks": graph.n,
                "traj_edge_cover": None,
                "traj_edge_repeat": None,
            }
        elif mode == "coverage":
            sampled, sample_meta = sample_patches_graph_level(graph, cfg, seed=local_seed)
        elif mode in {"radius2_atom", "radius2_atom_unrooted", "radius2_atom_topo"}:
            if bundle.node_feats is None or bundle.edge_feats is None:
                raise ValueError("radius2_atom sampling requires node and edge features")
            centers = list(graph.nodes)
            columns: list[np.ndarray] = []
            raw_nodes: list[set[int]] = []
            for center in centers:
                nodes = _radius_ego_nodes(graph, int(center), int(cfg.radius))
                if mode == "radius2_atom":
                    vector = _radius2_rooted_chem_vector(
                        graph,
                        nodes,
                        int(center),
                        cfg.max_nodes,
                        bundle.node_feats[graph_index],
                        bundle.edge_feats[graph_index],
                    )
                    columns.append(vector)
                elif mode == "radius2_atom_topo":
                    columns.append(
                        _radius2_rooted_topology_vector(
                            graph, nodes, int(center), cfg.max_nodes
                        )
                    )
                else:
                    # Ablation: same radius-2 atom-centered patches but the
                    # historical unrooted BFS vectorizer chooses its own root.
                    from types import SimpleNamespace

                    patch, _ = bundle_to_Y(
                        graph,
                        SimpleNamespace(node_sets=[nodes]),
                        cfg.max_nodes,
                        cfg.order_mode,
                        patch_feat="chem",
                        node_feat=bundle.node_feats[graph_index],
                        edge_feat=bundle.edge_feats[graph_index],
                    )
                    columns.append(patch[:, 0])
                raw_nodes.append(nodes)
            if columns:
                matrix = np.stack(columns, axis=1)
            else:
                fallback_dimension = 28 if mode == "radius2_atom_topo" else 52
                matrix = np.zeros((fallback_dimension, 1), dtype=np.float64)
            sample_meta = {
                "sampling": mode,
                "radius": int(cfg.radius),
                "n_centers_raw": int(len(centers)),
                "mean_ego_nodes": float(np.mean([len(nodes) for nodes in raw_nodes]))
                if raw_nodes
                else 0.0,
            }
            # Skip the generic bundle_to_Y call below: this branch already
            # produced the fixed-width matrix for every atom center.
            if cfg.normalize_patches and matrix.shape[1]:
                norms = np.linalg.norm(matrix, axis=0, keepdims=True)
                matrix = matrix / np.maximum(norms, 1e-12)
            if cfg.max_patches_per_graph is not None and matrix.shape[1] > cfg.max_patches_per_graph:
                rng = np.random.default_rng(local_seed)
                chosen = np.sort(
                    rng.choice(matrix.shape[1], size=cfg.max_patches_per_graph, replace=False)
                )
                matrix = matrix[:, chosen]
            matrices.append(np.asarray(matrix, dtype=np.float64))
            metadata.append(
                {
                    "graph_index": graph_index,
                    "local_seed": local_seed,
                    "n_patches": int(matrix.shape[1]),
                    **sample_meta,
                    "n_features": int(matrix.shape[0]),
                }
            )
            continue
        else:
            raise ValueError(f"unsupported sampler mode: {mode}")
        matrix, vector_meta = bundle_to_Y(
            graph,
            sampled,
            cfg.max_nodes,
            cfg.order_mode,
            patch_feat=cfg.patch_feat,
            node_feat=bundle.node_feats[graph_index],
            edge_feat=bundle.edge_feats[graph_index],
        )
        if vector_meta.get("n_cols", matrix.shape[1]) == 0:
            matrix = matrix[:, :0]
        if cfg.normalize_patches and matrix.shape[1]:
            norms = np.linalg.norm(matrix, axis=0, keepdims=True)
            matrix = matrix / np.maximum(norms, 1e-12)
        if cfg.max_patches_per_graph is not None and matrix.shape[1] > cfg.max_patches_per_graph:
            rng = np.random.default_rng(local_seed)
            chosen = np.sort(rng.choice(matrix.shape[1], size=cfg.max_patches_per_graph, replace=False))
            matrix = matrix[:, chosen]
        matrices.append(np.asarray(matrix, dtype=np.float64))
        metadata.append(
            {
                "graph_index": graph_index,
                "local_seed": local_seed,
                "n_patches": int(matrix.shape[1]),
                **sample_meta,
                **vector_meta,
            }
        )
    return matrices, metadata


def collect_train_patch_matrix(
    patch_matrices: Sequence[np.ndarray],
    train_indices: np.ndarray,
    maximum: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_set = {int(value) for value in np.asarray(train_indices, dtype=np.int64)}
    columns: list[np.ndarray] = []
    sources: list[int] = []
    for graph_index in sorted(train_set):
        matrix = patch_matrices[graph_index]
        for column in range(matrix.shape[1]):
            if np.linalg.norm(matrix[:, column]) > 1e-12:
                columns.append(matrix[:, column])
                sources.append(graph_index)
    if not columns:
        raise RuntimeError("no non-zero training patches were collected")
    raw_count = len(columns)
    if raw_count > maximum:
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(raw_count, size=maximum, replace=False))
        columns = [columns[int(index)] for index in chosen]
        sources = [sources[int(index)] for index in chosen]
    if not set(sources).issubset(train_set):
        raise AssertionError("dictionary patch pool contains a non-training graph")
    matrix = np.stack(columns, axis=1)
    audit = {
        "fit_split": "train",
        "train_index_sha256": _index_hash(sorted(train_set)),
        "source_index_sha256": _index_hash(sources),
        "n_train_graphs_contributing": len(set(sources)),
        "n_patches_raw": raw_count,
        "n_patches_used": int(matrix.shape[1]),
        "source_subset_of_train": True,
    }
    return matrix, audit


def learn_matched_dictionaries(
    train_patches: np.ndarray,
    cfg: GraphLevelConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return INIT and FINAL dictionaries sharing the exact initialization."""
    if cfg.allow_overcomplete and cfg.n_atoms > train_patches.shape[0]:
        # The historical ksvd() convenience path caps K at the feature width.
        # For the mentor-shaped K64/52D experiment we explicitly construct the
        # overcomplete initialization, then pass it through the same OMP
        # zero-update audit and matched K-SVD update path.
        rng = np.random.default_rng(cfg.seed)
        indices = rng.choice(
            train_patches.shape[1], size=cfg.n_atoms, replace=train_patches.shape[1] < cfg.n_atoms
        )
        supplied = train_patches[:, indices].astype(np.float64, copy=True)
        supplied += 1e-3 * rng.standard_normal(supplied.shape)
        supplied /= np.maximum(np.linalg.norm(supplied, axis=0, keepdims=True), 1e-12)
        initial, _, initial_info = ksvd(
            train_patches,
            n_atoms=cfg.n_atoms,
            T=cfg.T,
            T_min=cfg.T_min,
            n_iter=0,
            seed=cfg.seed,
            initial_dictionary=supplied,
        )
    else:
        initial, _, initial_info = ksvd(
            train_patches,
            n_atoms=cfg.n_atoms,
            T=cfg.T,
            T_min=cfg.T_min,
            n_iter=0,
            seed=cfg.seed,
        )
    final, _, final_info = ksvd(
        train_patches,
        n_atoms=initial.shape[1],
        T=cfg.T,
        T_min=cfg.T_min,
        n_iter=cfg.ksvd_iter,
        seed=cfg.seed,
        initial_dictionary=initial,
    )
    return initial, final, {"initial": initial_info, "final": final_info}


def encode_sparse_readout(
    patch_matrices: Sequence[np.ndarray],
    dictionary: np.ndarray,
    cfg: GraphLevelConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    rows: list[np.ndarray] = []
    graph_errors: list[float] = []
    patch_counts: list[int] = []
    for matrix in patch_matrices:
        if matrix.shape[1]:
            encoded_matrix, codes = sparse_code_patch_matrix(matrix, dictionary, cfg)
            residual = encoded_matrix - dictionary @ codes
            denominators = np.maximum(np.linalg.norm(encoded_matrix, axis=0), 1e-12)
            patch_errors = np.linalg.norm(residual, axis=0) / denominators
            graph_error = float(np.linalg.norm(residual) / max(np.linalg.norm(encoded_matrix), 1e-12))
        else:
            codes = np.zeros((dictionary.shape[1], 0), dtype=np.float64)
            patch_errors = np.empty(0, dtype=np.float64)
            graph_error = 0.0
        rows.append(sparse_code_readouts(codes, patch_errors)["rich"])
        graph_errors.append(graph_error)
        patch_counts.append(int(matrix.shape[1]))
    features = np.stack(rows, axis=0).astype(np.float32)
    if not np.all(np.isfinite(features)):
        raise FloatingPointError("T_v1 contains non-finite values")
    meta = {
        "dimension": int(features.shape[1]),
        "n_atoms": int(dictionary.shape[1]),
        "readout": "rich = 10K sparse-code summaries + 6 reconstruction summaries + 2 counts",
        "mean_graph_reconstruction_error": float(np.mean(graph_errors)),
        "mean_patch_count": float(np.mean(patch_counts)),
    }
    return features, meta


def typed_matrix_distribution_readout(matrix: np.ndarray) -> np.ndarray:
    """Preserve typed input coordinates with 12 fixed distribution summaries."""
    if matrix.ndim != 2:
        raise ValueError(f"expected a feature-by-patch matrix, got {matrix.shape}")
    dimension, n_patches = matrix.shape
    if n_patches == 0:
        return np.zeros(len(TYPED_RECONSTRUCTION_BLOCKS) * dimension, dtype=np.float64)
    quantiles = np.quantile(matrix, [0.10, 0.25, 0.50, 0.75, 0.90], axis=1)
    blocks = (
        matrix.mean(axis=1),
        matrix.std(axis=1),
        matrix.min(axis=1),
        matrix.max(axis=1),
        *quantiles,
        np.abs(matrix).mean(axis=1),
        np.sqrt(np.mean(matrix * matrix, axis=1)),
        (np.abs(matrix) > 1e-10).mean(axis=1),
    )
    return np.concatenate(blocks)


def encode_typed_matrix_readout(
    patch_matrices: Sequence[np.ndarray],
    dictionary: np.ndarray | None,
    cfg: GraphLevelConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read out raw patches or their sparse reconstruction in typed coordinates."""
    rows: list[np.ndarray] = []
    errors: list[float] = []
    for matrix in patch_matrices:
        if dictionary is None:
            represented = matrix
        elif matrix.shape[1]:
            encoded_matrix, codes = sparse_code_patch_matrix(matrix, dictionary, cfg)
            represented = dictionary @ codes
            errors.append(
                float(
                    np.linalg.norm(encoded_matrix - represented)
                    / max(np.linalg.norm(encoded_matrix), 1e-12)
                )
            )
        else:
            represented = np.zeros((dictionary.shape[0], 0), dtype=np.float64)
            errors.append(0.0)
        rows.append(typed_matrix_distribution_readout(represented))
    features = np.stack(rows, axis=0).astype(np.float32)
    if not np.all(np.isfinite(features)):
        raise FloatingPointError("typed reconstruction readout contains non-finite values")
    input_dimension = int(patch_matrices[0].shape[0])
    return features, {
        "name": "R_v1_raw" if dictionary is None else "R_v1_reconstruction",
        "input_dimension": input_dimension,
        "blocks": list(TYPED_RECONSTRUCTION_BLOCKS),
        "dimension": int(features.shape[1]),
        "expected_dimension": len(TYPED_RECONSTRUCTION_BLOCKS) * input_dimension,
        "mean_graph_reconstruction_error": None if dictionary is None else float(np.mean(errors)),
    }


def assemble_views(
    composition: np.ndarray,
    initial: np.ndarray,
    final: np.ndarray,
    requested: Sequence[str],
    *,
    r_raw: np.ndarray | None = None,
    r_initial: np.ndarray | None = None,
    r_final: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    available = {
        "s": composition,
        "t_init": initial,
        "t_final": final,
        "s_t_init": np.concatenate([composition, initial], axis=1),
        "s_t_final": np.concatenate([composition, final], axis=1),
    }
    reconstruction_blocks = {
        "r_raw": r_raw,
        "r_init": r_initial,
        "r_final": r_final,
        "s_r_raw": None if r_raw is None else np.concatenate([composition, r_raw], axis=1),
        "s_r_init": (
            None if r_initial is None else np.concatenate([composition, r_initial], axis=1)
        ),
        "s_r_final": None if r_final is None else np.concatenate([composition, r_final], axis=1),
    }
    available.update({name: value for name, value in reconstruction_blocks.items() if value is not None})
    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError(f"views require unavailable reconstruction features: {missing}")
    views = {name: np.asarray(available[name], dtype=np.float32) for name in requested}
    for name, matrix in views.items():
        if matrix.ndim != 2 or matrix.shape[0] != composition.shape[0]:
            raise RuntimeError(f"invalid view {name}: {matrix.shape}")
    return views


def official_rocauc(evaluator, labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.float64).reshape(-1, 1)
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1, 1)
    if np.unique(labels).size < 2:
        return float("nan")
    return float(evaluator.eval({"y_true": labels, "y_pred": predictions})["rocauc"])


def _mean_std(values: Sequence[float | None]) -> dict[str, float | None]:
    finite = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if not finite.size:
        return {"mean": None, "std_sample": None}
    return {
        "mean": float(finite.mean()),
        "std_sample": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
    }


def fit_xgboost_views(
    views: Mapping[str, np.ndarray],
    labels: np.ndarray,
    split: Mapping[str, np.ndarray],
    classifier_config: Mapping[str, Any],
    evaluate_test: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from ogb.graphproppred import Evaluator
    from xgboost import XGBClassifier

    train = np.asarray(split["train"], dtype=np.int64)
    valid = np.asarray(split["valid"], dtype=np.int64)
    test = np.asarray(split["test"], dtype=np.int64)
    y_train = labels[train].astype(np.int64)
    y_valid = labels[valid].astype(np.int64)
    y_test = labels[test].astype(np.int64)
    if np.unique(y_train).size != 2 or np.unique(y_valid).size != 2:
        raise RuntimeError("train and validation splits must both contain two classes")
    class_weight = float(np.sum(y_train == 0) / max(1, np.sum(y_train == 1)))
    evaluator = Evaluator(name="ogbg-molhiv")
    seeds = [int(value) for value in classifier_config["seeds"]]
    fixed = {
        key: classifier_config[key]
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
    }
    records: list[dict[str, Any]] = []
    predictions: dict[str, Any] = {}
    for view_name, matrix in views.items():
        valid_predictions: list[np.ndarray] = []
        test_predictions: list[np.ndarray] = []
        for seed in seeds:
            model = XGBClassifier(
                **fixed,
                objective="binary:logistic",
                eval_metric="auc",
                scale_pos_weight=class_weight,
                random_state=seed,
                tree_method="hist",
            )
            model.fit(matrix[train], y_train)
            pred_train = model.predict_proba(matrix[train])[:, 1]
            pred_valid = model.predict_proba(matrix[valid])[:, 1]
            pred_test = model.predict_proba(matrix[test])[:, 1] if evaluate_test else None
            valid_predictions.append(pred_valid.astype(np.float64))
            if pred_test is not None:
                test_predictions.append(pred_test.astype(np.float64))
            records.append(
                {
                    "view": view_name,
                    "seed": seed,
                    "feature_dim": int(matrix.shape[1]),
                    "train_rocauc": official_rocauc(evaluator, y_train, pred_train),
                    "valid_rocauc": official_rocauc(evaluator, y_valid, pred_valid),
                    "test_rocauc": (
                        official_rocauc(evaluator, y_test, pred_test) if pred_test is not None else None
                    ),
                }
            )
        predictions[view_name] = {
            "valid": np.stack(valid_predictions, axis=0),
            "test": (
                np.stack(test_predictions, axis=0)
                if test_predictions
                else np.empty((0, len(test)), dtype=np.float64)
            ),
        }

    per_view: dict[str, Any] = {}
    for view_name in views:
        rows = [record for record in records if record["view"] == view_name]
        per_view[view_name] = {
            "dimension": int(views[view_name].shape[1]),
            "train": _mean_std([row["train_rocauc"] for row in rows]),
            "valid": _mean_std([row["valid_rocauc"] for row in rows]),
            "test": _mean_std([row["test_rocauc"] for row in rows]),
        }
    best_valid = max(
        per_view,
        key=lambda name: (
            float(per_view[name]["valid"]["mean"]),
            -list(views).index(name),
        ),
    )
    summary = {
        "classifier": "XGBClassifier(binary:logistic)",
        "class_weight": class_weight,
        "seeds": seeds,
        "evaluate_test": evaluate_test,
        "fixed_params": fixed,
        "per_view": per_view,
        "best_by_valid_only": best_valid,
        "selection_rule": "validation mean only; test is never used as a tie-breaker",
    }
    return records, {"summary": summary, "predictions": predictions}


def _write_records(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "view",
        "seed",
        "feature_dim",
        "train_rocauc",
        "valid_rocauc",
        "test_rocauc",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _render_summary(payload: Mapping[str, Any]) -> str:
    classifier = payload["classifier"]["per_view"]
    lines = [
        "# Mentor concept replication v1",
        "",
        f"Protocol: `{payload['protocol_id']}`",
        "",
        "This is an independent concept reproduction; it does not claim to reproduce the mentor's unknown 69/624-D features.",
        "",
        "| view | dim | valid ROC-AUC | test ROC-AUC |",
        "|---|---:|---:|---:|",
    ]
    for name, result in classifier.items():
        valid = result["valid"]
        test = result["test"]
        valid_text = f"{valid['mean']:.6f} ± {valid['std_sample']:.6f}"
        test_text = (
            "not evaluated" if test["mean"] is None else f"{test['mean']:.6f} ± {test['std_sample']:.6f}"
        )
        lines.append(f"| {name} | {result['dimension']} | {valid_text} | {test_text} |")
    lines.extend(
        [
            "",
            f"Best by validation only: `{payload['classifier']['best_by_valid_only']}`.",
            "",
            "Primary attribution checks:",
            "",
            "- `t_final - t_init`: contribution of K-SVD updates under matched initialization.",
            "- `s_t_final - s`: structural dictionary features beyond explicit composition.",
            "- Test evaluation is disabled unless explicitly requested after freezing the protocol.",
            "",
        ]
    )
    if "r_final" in classifier:
        lines.extend(
            [
                "Typed-reconstruction proxy checks:",
                "",
                "- `r_final - r_init`: K-SVD update contribution in typed reconstruction coordinates.",
                "- `r_final - r_raw`: information retained or lost through sparse reconstruction.",
                "- `s_r_final - s`: typed reconstruction beyond explicit composition.",
                "",
            ]
        )
    return "\n".join(lines)


def run_experiment(
    config: Mapping[str, Any],
    result_dir: Path,
    *,
    data_root: Path | None = None,
    max_graphs: int | None = None,
    evaluate_test: bool | None = None,
    save_features: bool = True,
) -> dict[str, Any]:
    data_config = dict(config["data"])
    classifier_config = dict(config["classifier"])
    root = data_root or _resolve_repo_path(data_config["root"])
    configured_max = data_config.get("max_graphs")
    maximum = max_graphs if max_graphs is not None else configured_max
    test_enabled = (
        bool(classifier_config.get("evaluate_test", False)) if evaluate_test is None else bool(evaluate_test)
    )
    result_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_molhiv(
        root=root,
        max_graphs=None if maximum is None else int(maximum),
        seed=int(data_config.get("subsample_seed", 0)),
        with_features=True,
    )
    composition, composition_names, composition_meta = build_composition_features(bundle)
    cfg = graph_level_config(config)
    mode = str(config["sampler"]["mode"])
    patch_matrices, sampling_meta = vectorize_graphs(bundle, cfg, mode)
    train_patches, leakage_audit = collect_train_patch_matrix(
        patch_matrices,
        bundle.split["train"],
        cfg.max_train_patches,
        cfg.seed,
    )
    initial_dictionary, final_dictionary, dictionary_info = learn_matched_dictionaries(train_patches, cfg)
    initial_features, initial_meta = encode_sparse_readout(patch_matrices, initial_dictionary, cfg)
    final_features, final_meta = encode_sparse_readout(patch_matrices, final_dictionary, cfg)
    requested_views = list(config["views"])
    reconstruction_requested = any("r_" in name for name in requested_views)
    raw_reconstruction = initial_reconstruction = final_reconstruction = None
    reconstruction_meta: dict[str, Any] = {}
    if reconstruction_requested:
        raw_reconstruction, raw_meta = encode_typed_matrix_readout(patch_matrices, None, cfg)
        initial_reconstruction, reconstruction_initial_meta = encode_typed_matrix_readout(
            patch_matrices, initial_dictionary, cfg
        )
        final_reconstruction, reconstruction_final_meta = encode_typed_matrix_readout(
            patch_matrices, final_dictionary, cfg
        )
        reconstruction_meta = {
            "r_raw_v1": raw_meta,
            "r_init_v1": reconstruction_initial_meta,
            "r_final_v1": reconstruction_final_meta,
        }
    views = assemble_views(
        composition,
        initial_features,
        final_features,
        requested_views,
        r_raw=raw_reconstruction,
        r_initial=initial_reconstruction,
        r_final=final_reconstruction,
    )
    records, fitted = fit_xgboost_views(
        views,
        bundle.y,
        bundle.split,
        classifier_config,
        test_enabled,
    )

    manifest = {
        "protocol_id": str(config["protocol_id"]),
        "status": experiment_status(str(config["protocol_id"]), maximum),
        "concept_replication": True,
        "not_exact_mentor_feature_replication": True,
        "resolved_config": dict(config),
        "runtime_environment": runtime_environment(),
        "data": bundle.meta,
        "split_hashes": {
            name: _index_hash(np.asarray(indices, dtype=np.int64)) for name, indices in bundle.split.items()
        },
        "leakage_audit": leakage_audit,
        "sampler": dict(config["sampler"]),
        "dictionary_config": dict(config["dictionary"]),
        "dictionary_info": dictionary_info,
        "features": {
            "s_v1": composition_meta,
            "t_init_v1": initial_meta,
            "t_final_v1": final_meta,
            **reconstruction_meta,
            "view_dimensions": {name: int(matrix.shape[1]) for name, matrix in views.items()},
            "composition_schema_file": "composition_schema.json",
        },
        "sampling": {
            "n_graphs": len(sampling_meta),
            "mean_patches": float(np.mean([item["n_patches"] for item in sampling_meta])),
            "minimum_patches": int(min(item["n_patches"] for item in sampling_meta)),
            "maximum_patches": int(max(item["n_patches"] for item in sampling_meta)),
        },
        "classifier": fitted["summary"],
    }

    _write_records(result_dir / "records.csv", records)
    _write_json(result_dir / "summary.json", manifest)
    _write_json(result_dir / "resolved_config.json", dict(config))
    _write_json(
        result_dir / "composition_schema.json",
        {"feature_names": composition_names, **composition_meta},
    )
    (result_dir / "MENTOR_CONCEPT_V1_SUMMARY.md").write_text(_render_summary(manifest), encoding="utf-8")
    if save_features:
        prediction_payload: dict[str, np.ndarray] = {}
        for name, values in fitted["predictions"].items():
            prediction_payload[f"{name}_valid_predictions"] = values["valid"]
            prediction_payload[f"{name}_test_predictions"] = values["test"]
        np.savez_compressed(
            result_dir / "features_and_predictions.npz",
            labels=bundle.y,
            train_indices=bundle.split["train"],
            valid_indices=bundle.split["valid"],
            test_indices=bundle.split["test"],
            dictionary_initial=initial_dictionary,
            dictionary_final=final_dictionary,
            composition=composition,
            t_initial=initial_features,
            t_final=final_features,
            **(
                {
                    "r_raw": raw_reconstruction,
                    "r_initial": initial_reconstruction,
                    "r_final": final_reconstruction,
                }
                if reconstruction_requested
                else {}
            ),
            **prediction_payload,
        )
        # A train/official-valid frozen view is convenient for the staged
        # Optuna screen.  It deliberately omits official-test rows and keeps
        # the official train rows first, matching tune_xgb_fused_proxy.py.
        train = np.asarray(bundle.split["train"], dtype=np.int64)
        valid = np.asarray(bundle.split["valid"], dtype=np.int64)
        ordered = np.concatenate([train, valid])
        original_indices = np.asarray(bundle.meta["original_indices"], dtype=np.int64)
        frozen_payload: dict[str, np.ndarray] = {
            "dataset_indices": original_indices[ordered],
            "labels": bundle.y[ordered].astype(np.int64),
            "train_count": np.asarray([train.size], dtype=np.int64),
        }
        for name, matrix in views.items():
            frozen_payload[name] = matrix[ordered]
        np.savez_compressed(result_dir / "feature_views_train_valid.npz", **frozen_payload)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--max-graphs", type=int)
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="explicitly enable official test evaluation after protocol freeze",
    )
    parser.add_argument("--no-save-features", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "validated",
                    "config": str(config_path),
                    "protocol_id": config["protocol_id"],
                    "views": config["views"],
                    "evaluate_test": bool(args.evaluate_test),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    manifest = run_experiment(
        config,
        args.result_dir.expanduser().resolve(),
        data_root=args.data_root.expanduser().resolve() if args.data_root else None,
        max_graphs=args.max_graphs,
        evaluate_test=True if args.evaluate_test else None,
        save_features=not args.no_save_features,
    )
    print(
        json.dumps(
            {
                "protocol_id": manifest["protocol_id"],
                "status": manifest["status"],
                "result_dir": str(args.result_dir.expanduser().resolve()),
                "n_used": manifest["data"]["n_used"],
                "dictionary_train_only": manifest["leakage_audit"]["source_subset_of_train"],
                "view_dimensions": manifest["features"]["view_dimensions"],
                "best_by_valid_only": manifest["classifier"]["best_by_valid_only"],
                "test_evaluated": manifest["classifier"]["evaluate_test"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
