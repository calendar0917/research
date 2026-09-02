"""ZINC-12K long-range and representation attribution for luyin16.

The experiment uses the official PyG ZINC ``subset=True`` split. For one
radius it samples every atom centre once, freezes those patch objects, and
derives all statistical/K-SVD views from the same objects. Dictionaries and
model selection use train data (and the official validation split where
explicitly stated); test is only evaluated after a view/configuration has been
selected. This diagnoses the mentor route, not an unavailable exact schema.

Exit code 2 is reserved for unavailable official ZINC data. PyG's ZINC class
downloads from its dataset URLs (Dropbox plus the benchmarking-gnns split
indices), never from a Python package mirror.
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from ksvd_research.core import ksvd
from ksvd_research.evaluation.graph_level import (
    GraphLevelConfig,
    sparse_code_patch_matrix,
    sparse_code_readouts,
)
from tracks.ksvd.code.pipeline import graph_from_edge_index


REPO_ROOT = Path(__file__).resolve().parents[4]
ATOM_BINS = 28
BOND_BINS = 4
EXPECTED_SPLIT_SIZES = {"train": 10_000, "val": 1_000, "test": 1_000}

DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.03,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
}

PRIMARY_VIEWS = {
    "global_all",
    "global_structure",
    "local_topology_raw",
    "local_typed_raw",
    "local_topology_ksvd_init",
    "local_topology_ksvd_final",
    "local_typed_ksvd_init",
    "local_typed_ksvd_final",
    "global_all_plus_local_typed_raw",
    "global_all_plus_topology_ksvd_final",
    "global_all_plus_typed_ksvd_init",
    "global_all_plus_typed_ksvd_final",
}

TUNE_CANDIDATES = {
    "global_all",
    "global_all_plus_local_geometry",
    "global_all_plus_local_typed_raw",
    "global_all_plus_topology_ksvd_final",
    "global_all_plus_typed_ksvd_init",
    "global_all_plus_typed_ksvd_final",
    "global_all_plus_local_typed_raw_plus_typed_ksvd_final",
}


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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _zinc_class():
    try:
        from torch_geometric.datasets import ZINC
    except ImportError as exc:  # pragma: no cover - dependency is in the ksvd group
        raise RuntimeError("ZINC proxy requires torch-geometric") from exc
    return ZINC


def _load_zinc(root: Path, split: str):
    return _zinc_class()(root=str(root), subset=True, split=split)


def _raw_file_audit(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted((root / "raw").glob("*")):
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        records.append({"name": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    return records


def source_audit(root: Path) -> dict[str, Any]:
    zinc = _zinc_class()
    return {
        "policy": "official_pyg_zinc_only",
        "dataset_url": zinc.url,
        "split_url": zinc.split_url,
        "raw_files": _raw_file_audit(root),
    }


def preflight(root: Path, result_path: Path | None = None) -> int:
    try:
        sizes = {split: len(_load_zinc(root, split)) for split in ("train", "val", "test")}
        payload = {
            "status": "available" if sizes == EXPECTED_SPLIT_SIZES else "unexpected_split_sizes",
            "root": str(root),
            "sizes": sizes,
            "expected_sizes": EXPECTED_SPLIT_SIZES,
            "source": source_audit(root),
        }
    except Exception as exc:  # official download/network errors are resumable
        payload = {"status": "blocked", "reason": repr(exc), "root": str(root)}
        if result_path is not None:
            _write_json(result_path, payload)
        print(json.dumps(payload, ensure_ascii=False))
        return 2
    if result_path is not None:
        _write_json(result_path, payload)
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload["status"] == "available" else 1


def _ego_nodes(graph, center: int, radius: int) -> set[int]:
    distance = {int(center): 0}
    queue: deque[int] = deque([int(center)])
    while queue:
        node = queue.popleft()
        if distance[node] >= radius:
            continue
        for neighbor in sorted(graph.neighbors(node)):
            if neighbor not in distance:
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)
    return set(distance)


def _rooted_order(graph, nodes: set[int], center: int) -> list[int]:
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
    order.extend(node for node in sorted(nodes) if node not in seen)
    return order


def _patch_vector(
    graph,
    nodes: set[int],
    center: int,
    node_types: np.ndarray,
    edge_types: Mapping[tuple[int, int], int],
    max_nodes: int,
) -> tuple[np.ndarray, set[int], int]:
    kept_nodes = set(_rooted_order(graph, nodes, center)[:max_nodes])
    order = _rooted_order(graph, kept_nodes, center)
    position = {node: index for index, node in enumerate(order)}
    adjacency = np.zeros((max_nodes, max_nodes), dtype=np.float32)
    induced = graph.induced(kept_nodes)
    for left, right in induced.edges():
        i, j = position[left], position[right]
        adjacency[i, j] = adjacency[j, i] = 1.0
    upper = np.asarray(
        [adjacency[i, j] for i in range(max_nodes) for j in range(i + 1, max_nodes)],
        dtype=np.float32,
    )
    atom_hist = np.zeros(ATOM_BINS, dtype=np.float32)
    for node in kept_nodes:
        atom_hist[int(node_types[node]) % ATOM_BINS] += 1.0
    atom_hist /= max(float(atom_hist.sum()), 1.0)
    bond_hist = np.zeros(BOND_BINS, dtype=np.float32)
    for left, right in induced.edges():
        bond_type = edge_types.get((left, right), edge_types.get((right, left), 0))
        bond_hist[int(bond_type) % BOND_BINS] += 1.0
    if bond_hist.sum() > 0:
        bond_hist /= bond_hist.sum()
    return np.concatenate([upper, atom_hist, bond_hist]), kept_nodes, induced.num_edges()


def _data_to_graph(data):
    edge_index = data.edge_index.detach().cpu().numpy()
    graph = graph_from_edge_index(int(data.num_nodes), edge_index)
    node_types = data.x.detach().cpu().numpy().reshape(-1).astype(np.int64, copy=False)
    edge_attr = data.edge_attr.detach().cpu().numpy().reshape(-1).astype(np.int64, copy=False)
    edge_types: dict[tuple[int, int], int] = {}
    for index in range(edge_index.shape[1]):
        left, right = int(edge_index[0, index]), int(edge_index[1, index])
        if left == right:
            continue
        key = (left, right) if left < right else (right, left)
        edge_types.setdefault(key, int(edge_attr[index]))
    return graph, node_types, edge_types


def _pair_coverage(n_nodes: int, node_sets: Sequence[set[int]]) -> float:
    possible = n_nodes * (n_nodes - 1) // 2
    if possible <= 0:
        return 1.0
    observed: set[tuple[int, int]] = set()
    for nodes in node_sets:
        ordered = sorted(nodes)
        observed.update((left, right) for i, left in enumerate(ordered) for right in ordered[i + 1 :])
    return len(observed) / possible


def vectorize_dataset(dataset, radius: int, max_nodes: int, max_patches: int | None, seed: int):
    matrices: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    dimension = max_nodes * (max_nodes - 1) // 2 + ATOM_BINS + BOND_BINS
    for graph_index, data in enumerate(dataset):
        graph, node_types, edge_types = _data_to_graph(data)
        centers = list(graph.nodes)
        columns: list[np.ndarray] = []
        kept_sets: list[set[int]] = []
        full_sizes: list[int] = []
        kept_sizes: list[int] = []
        induced_edges: list[int] = []
        for center in centers:
            full_nodes = _ego_nodes(graph, center, radius)
            vector, kept_nodes, edge_count = _patch_vector(
                graph, full_nodes, center, node_types, edge_types, max_nodes
            )
            columns.append(vector)
            kept_sets.append(kept_nodes)
            full_sizes.append(len(full_nodes))
            kept_sizes.append(len(kept_nodes))
            induced_edges.append(edge_count)
        if max_patches is not None and len(columns) > max_patches:
            rng = np.random.default_rng(seed + graph_index * 13)
            chosen = np.sort(rng.choice(len(columns), size=max_patches, replace=False))
            columns = [columns[int(index)] for index in chosen]
            kept_sets = [kept_sets[int(index)] for index in chosen]
            full_sizes = [full_sizes[int(index)] for index in chosen]
            kept_sizes = [kept_sizes[int(index)] for index in chosen]
            induced_edges = [induced_edges[int(index)] for index in chosen]
        matrix = np.stack(columns, axis=1) if columns else np.zeros((dimension, 1), dtype=np.float32)
        matrices.append(matrix.astype(np.float32, copy=False))
        covered_nodes = set().union(*kept_sets) if kept_sets else set()
        metadata.append(
            {
                "graph_index": graph_index,
                "n_nodes": graph.n,
                "n_edges": graph.num_edges(),
                "n_centers_total": len(centers),
                "n_patches": int(matrix.shape[1]),
                "center_fraction": len(columns) / max(len(centers), 1),
                "mean_ego_nodes_full": float(np.mean(full_sizes)) if full_sizes else 0.0,
                "std_ego_nodes_full": float(np.std(full_sizes)) if full_sizes else 0.0,
                "max_ego_nodes_full": int(max(full_sizes, default=0)),
                "mean_ego_nodes_kept": float(np.mean(kept_sizes)) if kept_sizes else 0.0,
                "std_ego_nodes_kept": float(np.std(kept_sizes)) if kept_sizes else 0.0,
                "mean_patch_edges": float(np.mean(induced_edges)) if induced_edges else 0.0,
                "truncation_rate": float(np.mean(np.asarray(full_sizes) > max_nodes)) if full_sizes else 0.0,
                "node_coverage": len(covered_nodes) / max(graph.n, 1),
                "pair_coverage": _pair_coverage(graph.n, kept_sets),
            }
        )
    return matrices, metadata


def _all_pair_distances(graph) -> tuple[np.ndarray, np.ndarray]:
    pair_distances: list[float] = []
    eccentricities: list[float] = []
    for source in graph.nodes:
        distance = {int(source): 0}
        queue: deque[int] = deque([int(source)])
        while queue:
            node = queue.popleft()
            for neighbor in graph.neighbors(node):
                if neighbor not in distance:
                    distance[neighbor] = distance[node] + 1
                    queue.append(neighbor)
        eccentricities.append(float(max(distance.values(), default=0)))
        pair_distances.extend(float(distance[target]) for target in graph.nodes if target > source and target in distance)
    return np.asarray(pair_distances, dtype=np.float64), np.asarray(eccentricities, dtype=np.float64)


def _global_structure_blocks(graph) -> tuple[np.ndarray, np.ndarray]:
    degree = np.asarray([len(graph.neighbors(node)) for node in graph.nodes], dtype=np.float64)
    density = 2.0 * graph.num_edges() / max(graph.n * (graph.n - 1), 1)
    triangles = sum(
        len(graph.neighbors(left) & graph.neighbors(right))
        for left, right in graph.edges()
    ) / 3.0
    clustering_values = []
    for node in graph.nodes:
        neighbors = graph.neighbors(node)
        possible = len(neighbors) * (len(neighbors) - 1) / 2
        links = sum(1 for left in neighbors for right in neighbors if left < right and right in graph.neighbors(left))
        clustering_values.append(links / possible if possible else 0.0)
    degree_quantiles = np.quantile(degree, [0.25, 0.50, 0.75]) if degree.size else (0.0, 0.0, 0.0)
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
        dtype=np.float64,
    )
    distances, eccentricities = _all_pair_distances(graph)
    if distances.size:
        long = np.asarray(
            [
                distances.mean(),
                distances.std(),
                distances.min(),
                distances.max(),
                *np.quantile(distances, [0.25, 0.50, 0.75, 0.90]),
                eccentricities.min(),
                eccentricities.mean(),
                eccentricities.std(),
                eccentricities.max(),
                float(np.mean(distances > 2)),
                float(np.mean(distances > 3)),
                float(np.mean(distances > 4)),
            ],
            dtype=np.float64,
        )
    else:
        long = np.zeros(15, dtype=np.float64)
    return short, long


def global_feature_views(dataset) -> dict[str, np.ndarray]:
    short_rows: list[np.ndarray] = []
    long_rows: list[np.ndarray] = []
    attribute_rows: list[np.ndarray] = []
    for data in dataset:
        graph, node_types, edge_types = _data_to_graph(data)
        short, long = _global_structure_blocks(graph)
        atom_hist = np.bincount(node_types % ATOM_BINS, minlength=ATOM_BINS).astype(np.float64)
        atom_hist /= max(atom_hist.sum(), 1.0)
        bond_values = np.asarray(list(edge_types.values()), dtype=np.int64)
        bond_hist = np.bincount(bond_values % BOND_BINS, minlength=BOND_BINS).astype(np.float64)
        bond_hist /= max(bond_hist.sum(), 1.0)
        short_rows.append(short)
        long_rows.append(long)
        attribute_rows.append(np.concatenate([atom_hist, bond_hist]))
    short_array = np.stack(short_rows).astype(np.float32)
    long_array = np.stack(long_rows).astype(np.float32)
    attribute_array = np.stack(attribute_rows).astype(np.float32)
    structure = np.concatenate([short_array, long_array], axis=1)
    return {
        "global_structure_short": short_array,
        "global_structure_long": long_array,
        "global_structure": structure,
        "global_attributes": attribute_array,
        "global_all": np.concatenate([structure, attribute_array], axis=1),
    }


def _block_slice(max_nodes: int, block: str) -> slice:
    topology_dim = max_nodes * (max_nodes - 1) // 2
    if block == "topology":
        return slice(0, topology_dim)
    if block == "attributes":
        return slice(topology_dim, topology_dim + ATOM_BINS + BOND_BINS)
    if block == "typed":
        return slice(0, topology_dim + ATOM_BINS + BOND_BINS)
    raise ValueError(f"unknown patch block: {block}")


def select_and_normalize(matrices: Sequence[np.ndarray], max_nodes: int, block: str) -> list[np.ndarray]:
    selected: list[np.ndarray] = []
    block_slice = _block_slice(max_nodes, block)
    for matrix in matrices:
        value = np.asarray(matrix[block_slice], dtype=np.float64)
        norms = np.linalg.norm(value, axis=0, keepdims=True)
        selected.append(value / np.maximum(norms, 1e-12))
    return selected


def raw_readout(matrices: Sequence[np.ndarray]) -> np.ndarray:
    rows = []
    for matrix in matrices:
        if matrix.shape[1] == 0:
            rows.append(np.zeros(matrix.shape[0] * 5 + 2, dtype=np.float64))
            continue
        rows.append(
            np.concatenate(
                [
                    matrix.mean(axis=1),
                    matrix.std(axis=1),
                    *np.quantile(matrix, [0.25, 0.50, 0.75], axis=1),
                    np.asarray([matrix.shape[1], np.log1p(matrix.shape[1])], dtype=np.float64),
                ]
            )
        )
    return np.stack(rows).astype(np.float32)


def geometry_readout(metadata: Sequence[Mapping[str, Any]]) -> np.ndarray:
    keys = (
        "n_patches",
        "center_fraction",
        "mean_ego_nodes_full",
        "std_ego_nodes_full",
        "max_ego_nodes_full",
        "mean_ego_nodes_kept",
        "std_ego_nodes_kept",
        "mean_patch_edges",
        "truncation_rate",
        "node_coverage",
        "pair_coverage",
    )
    return np.asarray([[float(row[key]) for key in keys] for row in metadata], dtype=np.float32)


def collect_train_patches(matrices: Sequence[np.ndarray], maximum: int, seed: int) -> np.ndarray:
    n_columns = sum(matrix.shape[1] for matrix in matrices)
    if n_columns == 0:
        raise RuntimeError("no non-empty ZINC training patches")
    rng = np.random.default_rng(seed)
    if n_columns <= maximum:
        return np.concatenate(matrices, axis=1).astype(np.float64, copy=False)
    chosen = set(int(index) for index in rng.choice(n_columns, size=maximum, replace=False))
    columns: list[np.ndarray] = []
    offset = 0
    for matrix in matrices:
        for column in range(matrix.shape[1]):
            if offset + column in chosen:
                columns.append(matrix[:, column])
        offset += matrix.shape[1]
    return np.stack(columns, axis=1).astype(np.float64, copy=False)


def learn_matched_dictionaries(
    train_matrices: Sequence[np.ndarray],
    args: argparse.Namespace,
    block: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train_patches = collect_train_patches(train_matrices, args.max_train_patches, args.seed)
    n_atoms = min(args.n_atoms, train_patches.shape[0], train_patches.shape[1])
    initial, _, initial_info = ksvd(
        train_patches,
        n_atoms=n_atoms,
        T=args.sparsity,
        T_min=min(args.minimum_sparsity, args.sparsity),
        n_iter=0,
        seed=args.seed,
    )
    final, _, final_info = ksvd(
        train_patches,
        n_atoms=n_atoms,
        T=args.sparsity,
        T_min=min(args.minimum_sparsity, args.sparsity),
        n_iter=args.ksvd_iterations,
        seed=args.seed,
        initial_dictionary=initial,
    )
    info = {
        "block": block,
        "patch_dimension": int(train_patches.shape[0]),
        "n_train_patches": int(train_patches.shape[1]),
        "initial": initial_info,
        "final": final_info,
        "reconstruction_delta_final_minus_init": float(final_info["recon_rel"] - initial_info["recon_rel"]),
    }
    return initial, final, info


def dictionary_readouts(
    matrices: Sequence[np.ndarray],
    dictionary: np.ndarray,
    cfg: GraphLevelConfig,
) -> dict[str, np.ndarray]:
    code_rows: list[np.ndarray] = []
    reconstruction_matrices: list[np.ndarray] = []
    residual_matrices: list[np.ndarray] = []
    for matrix in matrices:
        normalized, codes = sparse_code_patch_matrix(matrix, dictionary, cfg)
        reconstruction = dictionary @ codes
        residual = np.abs(normalized - reconstruction)
        errors = np.linalg.norm(residual, axis=0) / np.maximum(np.linalg.norm(normalized, axis=0), 1e-12)
        code_rows.append(sparse_code_readouts(codes, errors)["rich"])
        reconstruction_matrices.append(reconstruction)
        residual_matrices.append(residual)
    return {
        "code": np.stack(code_rows).astype(np.float32),
        "reconstruction": raw_readout(reconstruction_matrices),
        "absolute_residual": raw_readout(residual_matrices),
    }


def _fit_regressor(
    x_train,
    y_train,
    x_eval,
    y_eval,
    seed: int,
    n_jobs: int,
    parameters: Mapping[str, Any] | None = None,
) -> float:
    model_parameters = {**DEFAULT_XGB_PARAMS, **dict(parameters or {})}
    model = XGBRegressor(
        **model_parameters,
        objective="reg:squarederror",
        eval_metric="mae",
        random_state=seed,
        n_jobs=n_jobs,
        tree_method="hist",
    )
    model.fit(x_train, y_train)
    prediction = model.predict(x_eval)
    return float(mean_absolute_error(y_eval, prediction))


def evaluate_views(
    views: Mapping[str, Sequence[np.ndarray]],
    labels: Sequence[np.ndarray],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, float], list[str]]:
    train_y, valid_y, test_y = labels
    seeds = list(dict.fromkeys(int(seed) for seed in args.seeds))
    screen_seed = seeds[0]
    screen = {
        name: _fit_regressor(arrays[0], train_y, arrays[1], valid_y, screen_seed, args.n_jobs)
        for name, arrays in views.items()
    }
    baseline = screen["global_all"]
    promoted = sorted(
        name
        for name in views
        if name in PRIMARY_VIEWS or screen[name] <= baseline + args.promotion_margin
    )
    outputs: dict[str, Any] = {}
    for name, arrays in views.items():
        valid_rows = [{"seed": screen_seed, "mae": screen[name]}]
        test_rows: list[dict[str, Any]] = []
        if name in promoted:
            for seed in seeds[1:]:
                valid_rows.append(
                    {
                        "seed": seed,
                        "mae": _fit_regressor(arrays[0], train_y, arrays[1], valid_y, seed, args.n_jobs),
                    }
                )
            train_valid_x = np.concatenate([arrays[0], arrays[1]], axis=0)
            train_valid_y = np.concatenate([train_y, valid_y])
            for seed in seeds:
                test_rows.append(
                    {
                        "seed": seed,
                        "mae": _fit_regressor(
                            train_valid_x,
                            train_valid_y,
                            arrays[2],
                            test_y,
                            seed,
                            args.n_jobs,
                        ),
                    }
                )
        outputs[name] = {
            "dimension": int(arrays[0].shape[1]),
            "screen_valid_mae": screen[name],
            "promotion": "primary" if name in PRIMARY_VIEWS else ("automatic" if name in promoted else "screen_only"),
            "valid_rows": valid_rows,
            "valid_mae_mean": float(np.mean([row["mae"] for row in valid_rows])),
            "test_rows_after_train_valid_refit": test_rows,
            "test_mae_mean": float(np.mean([row["mae"] for row in test_rows])) if test_rows else None,
        }
    return outputs, screen, promoted


def _tune_one_view(
    arrays: Sequence[np.ndarray],
    labels: Sequence[np.ndarray],
    args: argparse.Namespace,
    view_name: str,
) -> dict[str, Any]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    train_y, valid_y, test_y = labels

    def objective(trial) -> float:
        parameters = {
            "n_estimators": trial.suggest_int("n_estimators", 250, 900),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        return _fit_regressor(
            arrays[0], train_y, arrays[1], valid_y, args.seed, args.n_jobs, parameters
        )

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=args.optuna_trials, show_progress_bar=False)
    train_valid_x = np.concatenate([arrays[0], arrays[1]], axis=0)
    train_valid_y = np.concatenate([train_y, valid_y])
    test_mae = _fit_regressor(
        train_valid_x,
        train_valid_y,
        arrays[2],
        test_y,
        args.seed,
        args.n_jobs,
        study.best_params,
    )
    return {
        "view": view_name,
        "selection_scope": "official train -> official validation",
        "test_scope": "train+validation refit -> held-out official test",
        "n_trials": args.optuna_trials,
        "best_valid_mae": float(study.best_value),
        "best_parameters": study.best_params,
        "test_mae_after_train_valid_refit": test_mae,
        "trials": [
            {"number": trial.number, "value": trial.value, "parameters": trial.params}
            for trial in study.trials
        ],
    }


def tune_selected_views(
    views: Mapping[str, Sequence[np.ndarray]],
    labels: Sequence[np.ndarray],
    screen: Mapping[str, float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.optuna_trials <= 0 or args.tune_top_k <= 0:
        return {"status": "disabled", "views": {}}
    eligible = sorted((name for name in TUNE_CANDIDATES if name in views), key=lambda name: screen[name])
    selected = ["global_all"]
    selected.extend(name for name in eligible if name != "global_all")
    selected = selected[: args.tune_top_k]
    return {
        "status": "completed",
        "selection": "global_all plus best fixed-parameter candidate views on validation",
        "selected_views": selected,
        "views": {name: _tune_one_view(views[name], labels, args, name) for name in selected},
    }


def _concatenate_views(
    views: Mapping[str, Sequence[np.ndarray]],
    *names: str,
) -> list[np.ndarray]:
    return [np.concatenate([views[name][split] for name in names], axis=1) for split in range(3)]


def _aggregate_metadata(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = (
        "n_patches",
        "center_fraction",
        "mean_ego_nodes_full",
        "mean_ego_nodes_kept",
        "truncation_rate",
        "node_coverage",
        "pair_coverage",
    )
    return {f"mean_{key}": float(np.mean([float(row[key]) for row in rows])) for key in keys}


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = _resolve(args.data_root)
    train = _load_zinc(root, "train")
    valid = _load_zinc(root, "val")
    test = _load_zinc(root, "test")
    if args.max_graphs is not None:
        train = train[: min(len(train), args.max_graphs)]
        valid = valid[: min(len(valid), max(1, args.max_graphs // 10))]
        test = test[: min(len(test), max(1, args.max_graphs // 10))]
    datasets = (train, valid, test)
    vectorized = [
        vectorize_dataset(ds, args.radius, args.max_nodes, args.max_patches, args.seed + offset)
        for offset, ds in enumerate(datasets)
    ]
    raw_matrices = [item[0] for item in vectorized]
    metadata = [item[1] for item in vectorized]
    labels = [
        np.asarray([float(data.y.view(-1)[0]) for data in dataset], dtype=np.float32)
        for dataset in datasets
    ]

    global_blocks = [global_feature_views(dataset) for dataset in datasets]
    views: dict[str, list[np.ndarray]] = {
        name: [global_blocks[split][name] for split in range(3)]
        for name in global_blocks[0]
    }
    views["local_geometry"] = [geometry_readout(rows) for rows in metadata]

    block_dictionaries: dict[str, Any] = {}
    for block in ("attributes", "typed", "topology"):
        block_matrices = [select_and_normalize(matrices, args.max_nodes, block) for matrices in raw_matrices]
        views[f"local_{block}_raw"] = [raw_readout(matrices) for matrices in block_matrices]
        if block == "attributes":
            continue
        initial, final, dictionary_info = learn_matched_dictionaries(block_matrices[0], args, block)
        block_dictionaries[block] = dictionary_info
        cfg = GraphLevelConfig(
            T=args.sparsity,
            T_min=min(args.minimum_sparsity, args.sparsity),
            normalize_patches=False,
            seed=args.seed,
        )
        for stage, dictionary in (("init", initial), ("final", final)):
            encoded = [dictionary_readouts(matrices, dictionary, cfg) for matrices in block_matrices]
            views[f"local_{block}_ksvd_{stage}"] = [item["code"] for item in encoded]
            views[f"local_{block}_reconstruction_{stage}"] = [item["reconstruction"] for item in encoded]
            views[f"local_{block}_absolute_residual_{stage}"] = [item["absolute_residual"] for item in encoded]

    views["global_structure_plus_local_topology_raw"] = _concatenate_views(
        views, "global_structure", "local_topology_raw"
    )
    views["global_attributes_plus_local_attributes_raw"] = _concatenate_views(
        views, "global_attributes", "local_attributes_raw"
    )
    views["global_all_plus_local_geometry"] = _concatenate_views(views, "global_all", "local_geometry")
    views["global_all_plus_local_typed_raw"] = _concatenate_views(
        views, "global_all", "local_typed_raw"
    )
    views["global_all_plus_local_topology_attributes_late"] = _concatenate_views(
        views, "global_all", "local_topology_raw", "local_attributes_raw"
    )
    views["global_all_plus_topology_ksvd_init"] = _concatenate_views(
        views, "global_all", "local_topology_ksvd_init"
    )
    views["global_all_plus_topology_ksvd_final"] = _concatenate_views(
        views, "global_all", "local_topology_ksvd_final"
    )
    views["global_all_plus_typed_ksvd_init"] = _concatenate_views(
        views, "global_all", "local_typed_ksvd_init"
    )
    views["global_all_plus_typed_ksvd_final"] = _concatenate_views(
        views, "global_all", "local_typed_ksvd_final"
    )
    views["global_all_plus_local_typed_raw_plus_typed_ksvd_final"] = _concatenate_views(
        views, "global_all", "local_typed_raw", "local_typed_ksvd_final"
    )

    outputs, screen, promoted = evaluate_views(views, labels, args)
    tuned = tune_selected_views(views, labels, screen, args)
    return {
        "protocol_id": f"luyin16-zinc-radius{args.radius}-factorial-v2",
        "status": "full" if args.max_graphs is None else "development",
        "data": {
            "root": str(root),
            "train": len(train),
            "valid": len(valid),
            "test": len(test),
            "split": "PyG ZINC subset=True official train/val/test",
            "source": source_audit(root),
            "target": {
                "train_mean": float(labels[0].mean()),
                "train_std": float(labels[0].std()),
                "constant_median_valid_mae": float(
                    mean_absolute_error(labels[1], np.full_like(labels[1], np.median(labels[0])))
                ),
            },
        },
        "sampling": {
            "center": "every atom unless max_patches is explicitly set",
            "radius": args.radius,
            "max_nodes": args.max_nodes,
            "max_patches_per_graph": args.max_patches,
            "split_summaries": {
                name: _aggregate_metadata(rows)
                for name, rows in zip(("train", "valid", "test"), metadata)
            },
        },
        "representation": {
            "topology_dimension": args.max_nodes * (args.max_nodes - 1) // 2,
            "attribute_dimension": ATOM_BINS + BOND_BINS,
            "typed_dimension": args.max_nodes * (args.max_nodes - 1) // 2 + ATOM_BINS + BOND_BINS,
            "dictionary_atoms": args.n_atoms,
            "sparsity": args.sparsity,
            "dictionary_fit_scope": "official train patches only",
            "dictionaries": block_dictionaries,
        },
        "evaluation": {
            "metric": "MAE (lower is better)",
            "fixed_parameters": DEFAULT_XGB_PARAMS,
            "screen_seed": int(args.seeds[0]),
            "promotion_margin_from_global_all": args.promotion_margin,
            "promoted_views": promoted,
            "test_policy": "only promoted views; train+validation refit after validation selection",
        },
        "views": outputs,
        "optuna": tuned,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": importlib.metadata.version("numpy"),
            "torch_geometric": importlib.metadata.version("torch-geometric"),
            "xgboost": importlib.metadata.version("xgboost"),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/ZINC")
    parser.add_argument("--radius", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--max-nodes", type=int, default=12)
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--n-atoms", type=int, default=24)
    parser.add_argument("--sparsity", type=int, default=3)
    parser.add_argument("--minimum-sparsity", type=int, default=1)
    parser.add_argument("--ksvd-iterations", type=int, default=6)
    parser.add_argument("--max-train-patches", type=int, default=24_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-graphs", type=int)
    parser.add_argument("--promotion-margin", type=float, default=0.01)
    parser.add_argument("--optuna-trials", type=int, default=0)
    parser.add_argument("--tune-top-k", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    result_path = _resolve(args.result) if args.result else None
    if args.preflight:
        return preflight(_resolve(args.data_root), result_path)
    try:
        payload = run(args)
    except Exception as exc:
        message = str(exc).lower()
        if isinstance(exc, (OSError, RuntimeError)) and ("download" in message or "zinc" in message):
            blocked = {"status": "blocked", "reason": repr(exc)}
            if result_path is not None:
                _write_json(result_path, blocked)
            print(json.dumps(blocked, ensure_ascii=False))
            return 2
        raise
    if result_path is not None:
        _write_json(result_path, payload)
    compact = {
        name: {
            "dimension": row["dimension"],
            "valid_mae_mean": row["valid_mae_mean"],
            "test_mae_mean": row["test_mae_mean"],
            "promotion": row["promotion"],
        }
        for name, row in payload["views"].items()
    }
    print(
        json.dumps(
            {"status": payload["status"], "protocol_id": payload["protocol_id"], "views": compact},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
