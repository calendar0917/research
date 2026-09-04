"""ZINC transfer of the current ``S + marginal -> XGBoost`` route.

This runner intentionally excludes K-SVD and interaction blocks.  It mirrors
the current luyin16 MolHIV baseline at the level of information flow:

* ``S``: graph-level structure and atom/bond composition statistics;
* ``marginal``: mean/std over all atom-centred local rows, where each row
  contains rooted-WL topology-role marginals and atom/bond attribute
  marginals.  The WL encoder can optionally be typed: atom categories enter
  node colours and bond categories enter edge messages/roles;
* XGBoost regression for the ZINC penalized-logP target.

Optuna sees only internal folds of the official ZINC training split.  The
official validation split is evaluated once with the selected parameters, and
the official test split is evaluated after a train+validation refit.  No
K-SVD, joint/co-occurrence, attention, or test-driven selection is performed.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import optuna
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
import yaml

from tracks.ksvd.code.pipeline import graph_from_edge_index
from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import (
    REPO_ROOT,
    _load_zinc,
    _resolve,
    global_feature_views,
    source_audit,
)


DEFAULT_CONFIG = REPO_ROOT / "tracks/ksvd/configs/luyin16/zinc_s_marginal.yaml"

ATOM_CATEGORIES = 28
BOND_CATEGORIES = 4
NODE_ROLE_BINS = 64
EDGE_ROLE_BINS = 32
LOCAL_WIDTH = NODE_ROLE_BINS + EDGE_ROLE_BINS + ATOM_CATEGORIES + BOND_CATEGORIES
MARGINAL_WIDTH = 2 * LOCAL_WIDTH
CONTEXT_WIDTH = 5

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


def _stable_int(namespace: str, token: object) -> int:
    payload = (namespace + "|" + repr(token)).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def _stable_bin(namespace: str, token: object, width: int) -> int:
    return _stable_int(namespace, token) % int(width)


def _ego_distances(graph, center: int, radius: int) -> dict[int, int]:
    distances = {int(center): 0}
    queue: deque[int] = deque([int(center)])
    while queue:
        node = queue.popleft()
        if distances[node] >= int(radius):
            continue
        for neighbor in sorted(graph.neighbors(node)):
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    return distances


def _rooted_wl_roles(
    graph,
    center: int,
    radius: int,
    *,
    node_types: np.ndarray | None = None,
    edge_types: Mapping[tuple[int, int], int] | None = None,
    typed_wl: bool = False,
    wl_rounds: int = 2,
    node_role_bins: int = NODE_ROLE_BINS,
    edge_role_bins: int = EDGE_ROLE_BINS,
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...], np.ndarray, np.ndarray]:
    """Return an invariant rooted-WL role assignment for one ego graph."""
    if wl_rounds < 0:
        raise ValueError("wl_rounds must be non-negative")
    if typed_wl and (node_types is None or edge_types is None):
        raise ValueError("typed WL requires node and edge categories")
    distances = _ego_distances(graph, int(center), int(radius))
    nodes = tuple(sorted(distances))
    induced = graph.induced(set(nodes))
    if typed_wl:
        colors = {
            node: _stable_int(
                "zinc-rooted-wl-typed-initial",
                (
                    int(node == center),
                    int(distances[node]),
                    len(induced.neighbors(node)),
                    int(node_types[node]),  # type: ignore[index]
                ),
            )
            for node in nodes
        }
    else:
        colors = {
            node: _stable_int(
                "zinc-rooted-wl-initial",
                (int(node == center), int(distances[node]), len(induced.neighbors(node))),
            )
            for node in nodes
        }
    for iteration in range(int(wl_rounds)):
        next_colors: dict[int, int] = {}
        for node in nodes:
            if typed_wl:
                messages = tuple(
                    sorted(
                        (
                            colors[neighbor],
                            int(edge_types[graph.edge_key(node, neighbor)]),  # type: ignore[index]
                        )
                        for neighbor in induced.neighbors(node)
                    )
                )
            else:
                messages = tuple(sorted(colors[neighbor] for neighbor in induced.neighbors(node)))
            next_colors[node] = _stable_int(
                f"zinc-rooted-wl{'-typed' if typed_wl else ''}-{iteration}",
                (colors[node], messages),
            )
        colors = next_colors
    node_roles = np.asarray(
        [
            _stable_bin("zinc-node-role", colors[node], int(node_role_bins))
            for node in nodes
        ],
        dtype=np.int64,
    )
    edges = tuple(sorted(induced.edges()))
    edge_roles = np.asarray(
        [
            _stable_bin(
                "zinc-edge-role-typed" if typed_wl else "zinc-edge-role",
                (
                    tuple(sorted((colors[left], colors[right]))),
                    tuple(sorted((int(distances[left]), int(distances[right])))),
                    (
                        int(edge_types[graph.edge_key(left, right)])  # type: ignore[index]
                        if typed_wl
                        else None
                    ),
                ),
                int(edge_role_bins),
            )
            for left, right in edges
        ],
        dtype=np.int64,
    )
    return nodes, edges, node_roles, edge_roles


def _one_hot_histogram(values: Sequence[int], width: int) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.int64)
    if array.size and (array.min() < 0 or array.max() >= int(width)):
        raise ValueError(f"category outside [0, {int(width) - 1}]: {array.tolist()}")
    output = np.bincount(array, minlength=int(width)).astype(np.float32)
    if output.sum() > 0:
        output /= output.sum()
    return output


def _local_row(
    graph,
    center: int,
    node_types: np.ndarray,
    edge_types: Mapping[tuple[int, int], int],
    radius: int,
    *,
    typed_wl: bool = False,
    wl_rounds: int = 2,
    node_role_bins: int = NODE_ROLE_BINS,
    edge_role_bins: int = EDGE_ROLE_BINS,
) -> tuple[np.ndarray, int, int, set[int]]:
    nodes, edges, node_roles, edge_roles = _rooted_wl_roles(
        graph,
        center,
        radius,
        node_types=node_types,
        edge_types=edge_types,
        typed_wl=typed_wl,
        wl_rounds=wl_rounds,
        node_role_bins=node_role_bins,
        edge_role_bins=edge_role_bins,
    )
    node_role = np.bincount(node_roles, minlength=int(node_role_bins)).astype(np.float32)
    node_role /= max(float(node_roles.size), 1.0)
    edge_role = np.bincount(edge_roles, minlength=int(edge_role_bins)).astype(np.float32)
    edge_role /= max(float(edge_roles.size), 1.0)
    node_attribute = _one_hot_histogram(
        [int(node_types[node]) for node in nodes], ATOM_CATEGORIES
    )
    bond_values = [
        int(edge_types[graph.edge_key(left, right)])
        for left, right in edges
    ]
    edge_attribute = _one_hot_histogram(bond_values, BOND_CATEGORIES)
    row = np.concatenate([node_role, edge_role, node_attribute, edge_attribute]).astype(
        np.float32, copy=False
    )
    expected_width = int(node_role_bins) + int(edge_role_bins) + ATOM_CATEGORIES + BOND_CATEGORIES
    if row.shape != (expected_width,):
        raise RuntimeError(f"local marginal width changed: {row.shape}; expected {(expected_width,)}")
    return row, len(nodes), len(edges), set(nodes)


def _distribution_mean_std(rows: np.ndarray, local_width: int = LOCAL_WIDTH) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != int(local_width):
        raise ValueError(f"expected [n_centres,{int(local_width)}], got {values.shape}")
    if values.shape[0] == 0:
        return np.zeros(2 * int(local_width), dtype=np.float32)
    return np.concatenate([values.mean(axis=0), values.std(axis=0)]).astype(
        np.float32, copy=False
    )


def _graph_marginal_features(
    data,
    radius: int,
    *,
    typed_wl: bool = False,
    wl_rounds: int = 2,
    node_role_bins: int = NODE_ROLE_BINS,
    edge_role_bins: int = EDGE_ROLE_BINS,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    edge_index = data.edge_index.detach().cpu().numpy()
    graph = graph_from_edge_index(int(data.num_nodes), edge_index)
    node_types = data.x.detach().cpu().numpy().reshape(-1).astype(np.int64, copy=False)
    edge_attr = data.edge_attr.detach().cpu().numpy().reshape(-1).astype(np.int64, copy=False)
    edge_types: dict[tuple[int, int], int] = {}
    for position in range(edge_index.shape[1]):
        left, right = int(edge_index[0, position]), int(edge_index[1, position])
        if left == right:
            continue
        edge_types.setdefault(graph.edge_key(left, right), int(edge_attr[position]))
    if node_types.size and (node_types.min() < 0 or node_types.max() >= ATOM_CATEGORIES):
        raise ValueError(
            f"ZINC atom category outside the declared schema: "
            f"min={int(node_types.min())}, max={int(node_types.max())}, "
            f"width={ATOM_CATEGORIES}"
        )
    if edge_types and (min(edge_types.values()) < 0 or max(edge_types.values()) >= BOND_CATEGORIES):
        raise ValueError(
            f"ZINC bond category outside the declared schema: "
            f"min={min(edge_types.values())}, max={max(edge_types.values())}, "
            f"width={BOND_CATEGORIES}"
        )
    rows: list[np.ndarray] = []
    node_counts: list[int] = []
    edge_counts: list[int] = []
    covered_nodes: set[int] = set()
    for center in graph.nodes:
        row, node_count, edge_count, kept = _local_row(
            graph,
            int(center),
            node_types,
            edge_types,
            radius,
            typed_wl=typed_wl,
            wl_rounds=wl_rounds,
            node_role_bins=node_role_bins,
            edge_role_bins=edge_role_bins,
        )
        rows.append(row)
        node_counts.append(node_count)
        edge_counts.append(edge_count)
        covered_nodes.update(kept)
    local_width = int(node_role_bins) + int(edge_role_bins) + ATOM_CATEGORIES + BOND_CATEGORIES
    matrix = np.stack(rows, axis=0) if rows else np.zeros((0, local_width), dtype=np.float32)
    expected_width = int(node_role_bins) + int(edge_role_bins) + ATOM_CATEGORIES + BOND_CATEGORIES
    if matrix.shape[1] != expected_width:
        raise RuntimeError(
            f"local feature width {matrix.shape[1]} does not match configured width {expected_width}"
        )
    marginal = _distribution_mean_std(matrix, expected_width)
    context = np.asarray(
        [
            np.log1p(float(graph.n)),
            np.log1p(float(graph.num_edges())),
            np.log1p(float(len(rows))),
            float(np.mean(node_counts)) if node_counts else 0.0,
            float(np.mean(edge_counts)) if edge_counts else 0.0,
        ],
        dtype=np.float32,
    )
    metadata = {
        "n_nodes": int(graph.n),
        "n_edges": int(graph.num_edges()),
        "n_centres": int(len(rows)),
        "mean_ego_nodes": float(np.mean(node_counts)) if node_counts else 0.0,
        "mean_ego_edges": float(np.mean(edge_counts)) if edge_counts else 0.0,
        "node_coverage": float(len(covered_nodes) / max(graph.n, 1)),
    }
    return marginal, context, metadata


def _feature_signature(config: Mapping[str, Any], data_root: Path) -> str:
    payload = {
        "schema": "zinc_invariant_s_marginal_v2",
        "representation": dict(config["representation"]),
        "data_root": str(data_root.resolve()),
        "encoder_sha256": _sha256(Path(__file__).resolve()),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


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
                raise ValueError(f"feature cache signature mismatch: {cache_path}")
            arrays = {name: np.asarray(archive[name]) for name in archive.files if name != "signature"}
        metadata = {
            "cache_path": str(cache_path),
            "cache_hit": True,
            "signature": signature,
        }
        return arrays, True, metadata

    representation = dict(config["representation"])
    radius = int(representation["radius"])
    typed_wl = bool(representation.get("typed_wl", False))
    wl_rounds = int(representation.get("wl_rounds", 2))
    node_role_bins = int(representation.get("node_role_bins", NODE_ROLE_BINS))
    edge_role_bins = int(representation.get("edge_role_bins", EDGE_ROLE_BINS))
    if node_role_bins <= 0 or edge_role_bins <= 0:
        raise ValueError("WL role bin counts must be positive")
    global_blocks = [global_feature_views(dataset) for dataset in datasets]
    arrays: dict[str, np.ndarray] = {}
    split_metadata: dict[str, Any] = {}
    for split_name, dataset, global_values in zip(
        ("train", "valid", "test"), datasets, global_blocks, strict=True
    ):
        marginal_rows: list[np.ndarray] = []
        context_rows: list[np.ndarray] = []
        graph_metadata: list[dict[str, Any]] = []
        for position, data in enumerate(dataset):
            marginal, context, metadata = _graph_marginal_features(
                data,
                radius,
                typed_wl=typed_wl,
                wl_rounds=wl_rounds,
                node_role_bins=node_role_bins,
                edge_role_bins=edge_role_bins,
            )
            marginal_rows.append(marginal)
            context_rows.append(context)
            graph_metadata.append(metadata)
            if position and position % 500 == 0:
                print(f"[{split_name}] local marginal graphs: {position}/{len(dataset)}", flush=True)
        arrays[f"{split_name}_s"] = np.asarray(global_values["global_all"], dtype=np.float32)
        arrays[f"{split_name}_marginal"] = np.stack(marginal_rows).astype(np.float32, copy=False)
        arrays[f"{split_name}_context"] = np.stack(context_rows).astype(np.float32, copy=False)
        split_metadata[split_name] = {
            "n_graphs": int(len(dataset)),
            "mean_nodes": float(np.mean([row["n_nodes"] for row in graph_metadata])),
            "mean_edges": float(np.mean([row["n_edges"] for row in graph_metadata])),
            "mean_centres": float(np.mean([row["n_centres"] for row in graph_metadata])),
            "mean_ego_nodes": float(np.mean([row["mean_ego_nodes"] for row in graph_metadata])),
            "mean_ego_edges": float(np.mean([row["mean_ego_edges"] for row in graph_metadata])),
            "mean_node_coverage": float(np.mean([row["node_coverage"] for row in graph_metadata])),
        }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, signature=np.asarray([signature]), **arrays)
    temporary.replace(cache_path)
    metadata = {
        "cache_path": str(cache_path),
        "cache_hit": False,
        "signature": signature,
        "splits": split_metadata,
    }
    return arrays, False, metadata


def _build_views(arrays: Mapping[str, np.ndarray], split: str) -> dict[str, np.ndarray]:
    s = np.asarray(arrays[f"{split}_s"], dtype=np.float32)
    marginal = np.asarray(arrays[f"{split}_marginal"], dtype=np.float32)
    context = np.asarray(arrays[f"{split}_context"], dtype=np.float32)
    if s.shape[0] != marginal.shape[0] or marginal.shape[0] != context.shape[0]:
        raise RuntimeError(f"feature rows are not aligned for {split}")
    return {
        "s": s,
        "s_marginal": np.concatenate([s, marginal, context], axis=1).astype(
            np.float32, copy=False
        ),
    }


def _base_params(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    return {
        "objective": str(config.get("xgboost_objective", "reg:squarederror")),
        "eval_metric": "mae",
        "tree_method": "hist",
        "max_bin": int(config.get("max_bin", 256)),
        "random_state": int(seed),
        "n_jobs": int(config["n_jobs"]),
    }


def _params_from_trial(trial: optuna.Trial, tuning: Mapping[str, Any]) -> dict[str, Any]:
    ranges = tuning["ranges"]
    return {
        "n_estimators": trial.suggest_int("n_estimators", int(ranges["n_estimators"][0]), int(ranges["n_estimators"][1])),
        "max_depth": trial.suggest_int("max_depth", int(ranges["max_depth"][0]), int(ranges["max_depth"][1])),
        "learning_rate": trial.suggest_float("learning_rate", float(ranges["learning_rate"][0]), float(ranges["learning_rate"][1]), log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", float(ranges["min_child_weight"][0]), float(ranges["min_child_weight"][1]), log=True),
        "subsample": trial.suggest_float("subsample", float(ranges["subsample"][0]), float(ranges["subsample"][1])),
        "colsample_bytree": trial.suggest_float("colsample_bytree", float(ranges["colsample_bytree"][0]), float(ranges["colsample_bytree"][1])),
        "reg_lambda": trial.suggest_float("reg_lambda", float(ranges["reg_lambda"][0]), float(ranges["reg_lambda"][1]), log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", float(ranges["reg_alpha"][0]), float(ranges["reg_alpha"][1]), log=True),
        "gamma": trial.suggest_float("gamma", float(ranges["gamma"][0]), float(ranges["gamma"][1])),
    }


def _fit_mae(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    params: Mapping[str, Any],
    seed: int,
    n_jobs: int,
    objective: str,
    max_bin: int,
) -> float:
    model_params = dict(params)
    model_params.update(
        {
            "objective": str(objective),
            "eval_metric": "mae",
            "tree_method": "hist",
            "max_bin": int(max_bin),
            "random_state": int(seed),
            "n_jobs": int(n_jobs),
        }
    )
    model = XGBRegressor(**model_params)
    model.fit(x_train, y_train)
    return float(mean_absolute_error(y_eval, model.predict(x_eval)))


def _cv_scores(
    x: np.ndarray,
    y: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    params: Mapping[str, Any],
    seed: int,
    n_jobs: int,
    objective: str,
    max_bin: int,
) -> list[float]:
    return [
        _fit_mae(
            x[train],
            y[train],
            x[valid],
            y[valid],
            params,
            seed,
            n_jobs,
            objective,
            max_bin,
        )
        for train, valid in folds
    ]


def _tune_view(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    tuning = config["tuning"]
    objective_name = str(config.get("xgboost_objective", "reg:squarederror"))
    max_bin = int(config.get("max_bin", 256))
    seed = int(tuning["seed"]) + (0 if name == "s" else 100003)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    warm_start = {
        key: config["warm_start"][name][key]
        for key in SEARCH_KEYS
        if key in config.get("warm_start", {}).get(name, {})
    }
    ranges = tuning["ranges"]
    if len(warm_start) == len(SEARCH_KEYS) and all(
        float(ranges[key][0]) <= float(warm_start[key]) <= float(ranges[key][1])
        for key in SEARCH_KEYS
    ):
        study.enqueue_trial(warm_start)

    def objective(trial: optuna.Trial) -> float:
        params = _params_from_trial(trial, tuning)
        values = _cv_scores(
            x,
            y,
            folds,
            params,
            seed,
            int(config["n_jobs"]),
            objective_name,
            max_bin,
        )
        return float(np.mean(values))

    n_trials = int(tuning["n_trials"])
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = dict(study.best_trial.params)
    best_cv = _cv_scores(
        x,
        y,
        folds,
        best_params,
        seed,
        int(config["n_jobs"]),
        objective_name,
        max_bin,
    )
    return {
        "view": name,
        "objective": objective_name,
        "eval_metric": "mae",
        "selection_scope": "official train internal KFold only",
        "n_trials": n_trials,
        "n_folds": int(len(folds)),
        "best_trial": int(study.best_trial.number),
        "best_cv_mae": float(np.mean(best_cv)),
        "best_cv_fold_mae": best_cv,
        "best_params": best_params,
        "trials": [
            {
                "trial": int(trial.number),
                "value": None if trial.value is None else float(trial.value),
                "params": dict(trial.params),
                "state": str(trial.state),
            }
            for trial in study.trials
        ],
    }


def _evaluate_final(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    params: Mapping[str, Any],
    seeds: Sequence[int],
    n_jobs: int,
    objective: str,
    max_bin: int,
) -> dict[str, Any]:
    predictions: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        model_params = dict(params)
        model_params.update(
            {
                "objective": str(objective),
                "eval_metric": "mae",
                "tree_method": "hist",
                "max_bin": int(max_bin),
                "random_state": int(seed),
                "n_jobs": int(n_jobs),
            }
        )
        model = XGBRegressor(**model_params)
        model.fit(x_train, y_train)
        prediction = np.asarray(model.predict(x_eval), dtype=np.float64)
        predictions.append(prediction)
        rows.append({"seed": int(seed), "mae": float(mean_absolute_error(y_eval, prediction))})
    ensemble = np.mean(np.stack(predictions, axis=0), axis=0)
    values = [row["mae"] for row in rows]
    return {
        "rows": rows,
        "mean_mae": float(np.mean(values)),
        "std_mae": float(np.std(values)),
        "seed_ensemble_mae": float(mean_absolute_error(y_eval, ensemble)),
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


def run(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_config = config["data"]
    xgboost_objective = str(config.get("xgboost_objective", "reg:squarederror"))
    max_bin = int(config.get("max_bin", 256))
    data_root = _resolve(data_config["root"])
    result_json = _resolve(config["output"]["json"])
    result_markdown = _resolve(config["output"]["markdown"])
    cache_path = _resolve(config["output"]["feature_cache"])
    start = time.perf_counter()

    datasets = tuple(_load_zinc(data_root, split) for split in ("train", "val", "test"))
    labels = tuple(
        np.asarray([float(data.y.view(-1)[0]) for data in dataset], dtype=np.float32)
        for dataset in datasets
    )
    arrays, cache_hit, feature_meta = _build_or_load_features(
        datasets, config, data_root, cache_path
    )
    views = {
        "train": _build_views(arrays, "train"),
        "valid": _build_views(arrays, "valid"),
        "test": _build_views(arrays, "test"),
    }
    for split, split_views in views.items():
        for name, matrix in split_views.items():
            if not np.all(np.isfinite(matrix)):
                raise FloatingPointError(f"non-finite values in {split}:{name}")

    n_splits = int(config["tuning"]["n_splits"])
    kfold = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=int(config["tuning"]["split_seed"]),
    )
    folds = [(train.astype(np.int64), valid.astype(np.int64)) for train, valid in kfold.split(views["train"]["s_marginal"])]
    tuning_results = {}
    for name in ("s", "s_marginal"):
        print(f"Optuna tuning: {name} ({int(config['tuning']['n_trials'])} trials)", flush=True)
        tuning_results[name] = _tune_view(
            name,
            views["train"][name],
            labels[0],
            folds,
            config,
        )
        print(
            f"  {name}: CV MAE={tuning_results[name]['best_cv_mae']:.6f}",
            flush=True,
        )

    model_seeds = [int(value) for value in config["model_seeds"]]
    evaluation: dict[str, Any] = {}
    for name in ("s", "s_marginal"):
        params = tuning_results[name]["best_params"]
        valid_result = _evaluate_final(
            views["train"][name],
            labels[0],
            views["valid"][name],
            labels[1],
            params,
            model_seeds,
            int(config["n_jobs"]),
            xgboost_objective,
            max_bin,
        )
        train_valid_x = np.concatenate([views["train"][name], views["valid"][name]], axis=0)
        train_valid_y = np.concatenate([labels[0], labels[1]], axis=0)
        test_result = _evaluate_final(
            train_valid_x,
            train_valid_y,
            views["test"][name],
            labels[2],
            params,
            model_seeds,
            int(config["n_jobs"]),
            xgboost_objective,
            max_bin,
        )
        evaluation[name] = {
            "dimension": int(views["train"][name].shape[1]),
            "valid": valid_result,
            "test_after_train_valid_refit": test_result,
        }

    result = {
        "protocol_id": config["protocol_id"],
        "status": "completed",
        "data": {
            "root": str(data_root),
            "split": "PyG ZINC subset=True official train/val/test",
            "sizes": {"train": len(datasets[0]), "valid": len(datasets[1]), "test": len(datasets[2])},
            "source": source_audit(data_root),
            "target": {
                "train": _label_summary(labels[0]),
                "valid": _label_summary(labels[1]),
                "test": _label_summary(labels[2]),
            },
        },
        "representation": {
            "S": "global_all = global structure statistics + global atom/bond composition",
            "marginal": "all-centre radius patch local row mean/std",
            "local_row": "rooted-WL role histogram + edge-role histogram + atom-type histogram + bond-type histogram",
            "radius": int(config["representation"]["radius"]),
            "typed_wl": bool(config["representation"].get("typed_wl", False)),
            "wl_rounds": int(config["representation"].get("wl_rounds", 2)),
            "node_role_bins": int(config["representation"].get("node_role_bins", NODE_ROLE_BINS)),
            "edge_role_bins": int(config["representation"].get("edge_role_bins", EDGE_ROLE_BINS)),
            "atom_categories": ATOM_CATEGORIES,
            "bond_categories": BOND_CATEGORIES,
            "S_dimension": int(views["train"]["s"].shape[1]),
            "marginal_dimension": int(
                2
                * (
                    int(config["representation"].get("node_role_bins", NODE_ROLE_BINS))
                    + int(config["representation"].get("edge_role_bins", EDGE_ROLE_BINS))
                    + ATOM_CATEGORIES
                    + BOND_CATEGORIES
                )
                + CONTEXT_WIDTH
            ),
            "S_plus_marginal_dimension": int(views["train"]["s_marginal"].shape[1]),
            "centres": "every atom",
            "ksvd": False,
            "interaction_blocks": False,
        },
        "feature_cache": {
            **feature_meta,
            "sha256": _sha256(cache_path),
        },
        "tuning": {
            "selection_scope": "official train internal shuffled KFold only",
            "n_splits": n_splits,
            "split_seed": int(config["tuning"]["split_seed"]),
            "n_trials_per_view": int(config["tuning"]["n_trials"]),
            "results": tuning_results,
        },
        "evaluation": {
            "metric": "MAE (lower is better)",
            "xgboost_objective": xgboost_objective,
            "xgboost_eval_metric": "mae",
            "model_seeds": model_seeds,
            "valid_scope": "fit on official train with train-only selected parameters",
            "test_scope": "fit on official train+valid with parameters selected without test",
            "views": evaluation,
        },
        "runtime": {
            "seconds": float(time.perf_counter() - start),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": importlib.metadata.version("numpy"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "xgboost": importlib.metadata.version("xgboost"),
            "optuna": importlib.metadata.version("optuna"),
            "script_sha256": _sha256(Path(__file__).resolve()),
            "cache_hit": bool(cache_hit),
        },
    }
    _write_json_atomic(result_json, result)
    result_markdown.parent.mkdir(parents=True, exist_ok=True)
    result_markdown.write_text(_render_markdown(result), encoding="utf-8")
    return result


def _render_markdown(result: Mapping[str, Any]) -> str:
    evaluation = result["evaluation"]["views"]
    tuning = result["tuning"]["results"]
    lines = [
        f"# {result['protocol_id']}",
        "",
        "ZINC transfer of `S + marginal -> XGBoost`; no K-SVD or interaction block.",
        "",
        "## Protocol",
        "",
        f"- split: `{result['data']['split']}`; sizes `{result['data']['sizes']}`",
        f"- target: penalized logP / constrained solubility; metric: MAE",
        f"- tuning: {result['tuning']['n_splits']}-fold shuffled KFold inside official train, {result['tuning']['n_trials_per_view']} Optuna trials per view",
        "- official valid: evaluated after train-only tuning; official test: train+valid refit",
        "",
        "## Representation",
        "",
        f"- `S`: {result['representation']['S_dimension']}D global structure + atom/bond composition",
        f"- `marginal`: {result['representation']['marginal_dimension']}D local mean/std + context",
        f"- `S+marginal`: {result['representation']['S_plus_marginal_dimension']}D",
        f"- local object: invariant all-centre radius-{result['representation']['radius']} "
        f"{'typed' if result['representation']['typed_wl'] else 'untyped'}-WL "
        f"({result['representation']['wl_rounds']} rounds) role/attribute marginals",
        f"- XGBoost objective: `{result['evaluation']['xgboost_objective']}`; eval metric: `mae`",
        "",
        "## Results",
        "",
        "| view | dim | train CV MAE | valid MAE mean ± std | valid seed ensemble | test MAE mean ± std | test seed ensemble |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("s", "s_marginal"):
        cv = tuning[name]["best_cv_mae"]
        valid = evaluation[name]["valid"]
        test = evaluation[name]["test_after_train_valid_refit"]
        lines.append(
            f"| `{name}` | {evaluation[name]['dimension']} | {cv:.6f} | "
            f"{valid['mean_mae']:.6f} ± {valid['std_mae']:.6f} | "
            f"{valid['seed_ensemble_mae']:.6f} | "
            f"{test['mean_mae']:.6f} ± {test['std_mae']:.6f} | "
            f"{test['seed_ensemble_mae']:.6f} |"
        )
    s_valid = evaluation["s"]["valid"]["mean_mae"]
    sm_valid = evaluation["s_marginal"]["valid"]["mean_mae"]
    s_test = evaluation["s"]["test_after_train_valid_refit"]["mean_mae"]
    sm_test = evaluation["s_marginal"]["test_after_train_valid_refit"]["mean_mae"]
    lines.extend(
        [
            "",
            "## Increment over S",
            "",
            f"- valid: `S+marginal - S = {sm_valid - s_valid:+.6f} MAE`",
            f"- test after train+valid refit: `S+marginal - S = {sm_test - s_test:+.6f} MAE`",
            "",
            "## Selected parameters",
            "",
            "```json",
            json.dumps(
                {name: tuning[name]["best_params"] for name in ("s", "s_marginal")},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
            f"Runtime: `{result['runtime']['seconds']:.1f}s`; feature cache hit: `{result['runtime']['cache_hit']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    result = run(_resolve(args.config))
    compact = {
        name: {
            "dimension": row["dimension"],
            "valid_mae": row["valid"]["mean_mae"],
            "valid_ensemble_mae": row["valid"]["seed_ensemble_mae"],
            "test_mae": row["test_after_train_valid_refit"]["mean_mae"],
            "test_ensemble_mae": row["test_after_train_valid_refit"]["seed_ensemble_mae"],
        }
        for name, row in result["evaluation"]["views"].items()
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
