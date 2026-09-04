"""ZINC: decomposable structure--attribute statistics with one XGBoost.

This experiment tests the hypothesis that the useful part of centre-level
structure--attribute fusion can be retained without making an exact joint
motif ID.  Every atom centre gets:

* a topology-only rooted-WL role representation;
* an aligned atom/bond attribute representation.

The graph-level views are:

``s_factorized``
    Global statistics plus the mean/std distribution of topology and
    attributes, with no centre-level interaction.
``s_conditional_r3``
    ``s_factorized`` plus a reusable role-by-attribute cross moment at the
    final radius-3 WL level.
``s_conditional_multiscale``
    ``s_factorized`` plus the same cross moment at WL levels 0, 1, 2 and 3.

The role-by-attribute block is a fixed 64 x 64 table of aggregate moments,
not an exact token vocabulary.  It therefore shares statistical strength
between molecules whose local environments are similar.  Optuna sees only
internal folds of the official ZINC training split.  Official validation is
evaluated after each view's search, and official test is evaluated only after
the selected parameters have been frozen and the model is refit on train plus
validation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import optuna
from scipy import sparse
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
import yaml

from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import (
    REPO_ROOT,
    _data_to_graph,
    _load_zinc,
    _resolve,
    global_feature_views,
    source_audit,
)
from tracks.ksvd.experiments.luyin16.zinc_s_marginal import (
    ATOM_CATEGORIES,
    BOND_CATEGORIES,
    _stable_bin,
    _stable_int,
    _ego_distances,
)


DEFAULT_CONFIG = REPO_ROOT / "tracks/ksvd/configs/luyin16/zinc_conditional_statistics.yaml"
VIEW_NAMES = ("s_factorized", "s_conditional_r3", "s_conditional_multiscale")


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
    output = np.zeros(int(width), dtype=np.float32)
    if value < 0 or value >= int(width):
        raise ValueError(f"category {value} outside [0, {int(width) - 1}]")
    output[int(value)] = 1.0
    return output


def _histogram(values: Sequence[int], width: int) -> np.ndarray:
    output = np.bincount(np.asarray(list(values), dtype=np.int64), minlength=int(width)).astype(
        np.float32
    )
    return output / max(float(output.sum()), 1.0)


def _rooted_role_levels(
    graph: Any,
    centre: int,
    *,
    radius: int,
    levels: int,
    node_role_bins: int,
    edge_role_bins: int,
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...], tuple[np.ndarray, ...], tuple[np.ndarray, ...], dict[int, int]]:
    """Return invariant root roles and role histograms for all WL levels."""
    distances = _ego_distances(graph, int(centre), int(radius))
    nodes = tuple(sorted(distances))
    induced = graph.induced(set(nodes))
    colours = {
        node: _stable_int(
            "zinc-rooted-wl-initial",
            (int(node == centre), int(distances[node]), len(induced.neighbors(node))),
        )
        for node in nodes
    }
    edges = tuple(sorted(induced.edges()))
    root_roles: list[int] = []
    node_hists: list[np.ndarray] = []
    edge_hists: list[np.ndarray] = []
    for iteration in range(int(levels) + 1):
        node_roles = np.asarray(
            [_stable_bin("zinc-node-role", colours[node], int(node_role_bins)) for node in nodes],
            dtype=np.int64,
        )
        edge_roles = np.asarray(
            [
                _stable_bin(
                    "zinc-edge-role",
                    (
                        tuple(sorted((colours[left], colours[right]))),
                        tuple(sorted((int(distances[left]), int(distances[right])))),
                        None,
                    ),
                    int(edge_role_bins),
                )
                for left, right in edges
            ],
            dtype=np.int64,
        )
        root_roles.append(int(node_roles[nodes.index(int(centre))]))
        node_hists.append(_histogram(node_roles.tolist(), int(node_role_bins)))
        edge_hists.append(_histogram(edge_roles.tolist(), int(edge_role_bins)))
        if iteration == int(levels):
            break
        next_colours: dict[int, int] = {}
        for node in nodes:
            messages = tuple(
                sorted(colours[neighbour] for neighbour in induced.neighbors(node))
            )
            next_colours[node] = _stable_int(
                f"zinc-rooted-wl-{iteration}", (colours[node], messages)
            )
        colours = next_colours
    return nodes, tuple(root_roles), tuple(node_hists), tuple(edge_hists), distances


def _centre_rows(
    data: Any,
    *,
    radius: int,
    levels: int,
    node_role_bins: int,
    edge_role_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Build aligned topology rows, attribute rows and root roles."""
    graph, node_types, edge_types = _data_to_graph(data)
    structure_rows: list[np.ndarray] = []
    attribute_rows: list[np.ndarray] = []
    role_levels: list[list[int]] = [[] for _ in range(int(levels) + 1)]
    for centre in graph.nodes:
        nodes, roots, node_hists, edge_hists, distances = _rooted_role_levels(
            graph,
            int(centre),
            radius=int(radius),
            levels=int(levels),
            node_role_bins=int(node_role_bins),
            edge_role_bins=int(edge_role_bins),
        )
        for level, role in enumerate(roots):
            role_levels[level].append(int(role))
        shell = np.asarray(
            [
                float(sum(distance == shell_id for distance in distances.values()))
                / max(float(len(nodes)), 1.0)
                for shell_id in range(int(radius) + 1)
            ],
            dtype=np.float32,
        )
        local_edges = tuple(
            (left, right)
            for left, right in graph.induced(set(nodes)).edges()
        )
        local_cycle_rank = max(len(local_edges) - len(nodes) + 1, 0)
        structure_rows.append(
            np.concatenate(
                [
                    _one_hot(roots[-1], int(node_role_bins)),
                    node_hists[-1],
                    edge_hists[-1],
                    shell,
                    np.asarray(
                        [
                            float(len(nodes)) / max(float(graph.n), 1.0),
                            float(len(local_edges)) / max(float(graph.n), 1.0),
                            float(len(graph.neighbors(int(centre)))) / 4.0,
                            float(local_cycle_rank) / max(float(graph.n), 1.0),
                        ],
                        dtype=np.float32,
                    ),
                ]
            ).astype(np.float32, copy=False)
        )
        local_atoms = _histogram([int(node_types[node]) for node in nodes], ATOM_CATEGORIES)
        local_bonds = _histogram(
            [int(edge_types[graph.edge_key(left, right)]) for left, right in local_edges],
            BOND_CATEGORIES,
        )
        incident_bonds = _histogram(
            [
                int(edge_types[graph.edge_key(int(centre), neighbour)])
                for neighbour in graph.neighbors(int(centre))
            ],
            BOND_CATEGORIES,
        )
        attribute_rows.append(
            np.concatenate(
                [
                    _one_hot(int(node_types[int(centre)]), ATOM_CATEGORIES),
                    incident_bonds,
                    local_atoms,
                    local_bonds,
                ]
            ).astype(np.float32, copy=False)
        )
    structure = np.stack(structure_rows).astype(np.float32, copy=False)
    attributes = np.stack(attribute_rows).astype(np.float32, copy=False)
    roots = np.asarray(role_levels, dtype=np.int16).T
    expected_structure = int(node_role_bins) * 2 + int(edge_role_bins) + int(radius) + 5
    if structure.shape[1] != expected_structure:
        raise RuntimeError(f"unexpected structure width {structure.shape}; expected {expected_structure}")
    if attributes.shape[1] != ATOM_CATEGORIES * 2 + BOND_CATEGORIES * 2:
        raise RuntimeError(f"unexpected attribute width {attributes.shape}")
    return structure, attributes, roots, {
        "n_centres": int(structure.shape[0]),
        "observed_roles": [int(np.unique(roots[:, level]).size) for level in range(roots.shape[1])],
        "structure_width": int(structure.shape[1]),
        "attribute_width": int(attributes.shape[1]),
    }


def _distribution(rows: np.ndarray) -> np.ndarray:
    return np.concatenate([rows.mean(axis=0), rows.std(axis=0)]).astype(np.float32, copy=False)


def _condition_graphs(
    graph_rows: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    levels: Sequence[int],
    role_bins: int,
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    """Build a sparse graph-by-condition table from centre-local arrays."""
    attr_width = int(graph_rows[0][1].shape[1]) if graph_rows else 0
    total_width = len(levels) * int(role_bins) * attr_width
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    nnz_per_graph: list[int] = []
    for graph_index, (_, attributes, roots) in enumerate(graph_rows):
        n_centres = max(int(attributes.shape[0]), 1)
        graph_nnz = 0
        for block, level in enumerate(levels):
            for role in range(int(role_bins)):
                selected = roots[:, int(level)] == int(role)
                if not np.any(selected):
                    continue
                aggregate = attributes[selected].sum(axis=0) / float(n_centres)
                nonzero = np.flatnonzero(aggregate > 0.0)
                base = block * int(role_bins) * attr_width + int(role) * attr_width
                for attribute in nonzero.tolist():
                    row_indices.append(graph_index)
                    column_indices.append(base + int(attribute))
                    values.append(float(aggregate[int(attribute)]))
                    graph_nnz += 1
        nnz_per_graph.append(graph_nnz)
    matrix = sparse.coo_matrix(
        (
            np.asarray(values, dtype=np.float32),
            (np.asarray(row_indices, dtype=np.int64), np.asarray(column_indices, dtype=np.int64)),
        ),
        shape=(len(graph_rows), total_width),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    return matrix, {
        "width": int(total_width),
        "nnz": int(matrix.nnz),
        "mean_nnz_per_graph": float(np.mean(nnz_per_graph)) if nnz_per_graph else 0.0,
        "levels": [int(level) for level in levels],
    }


def _build_features(
    datasets: Sequence[Any],
    *,
    radius: int,
    levels: int,
    node_role_bins: int,
    edge_role_bins: int,
) -> tuple[dict[str, np.ndarray | sparse.csr_matrix], dict[str, Any]]:
    global_views = tuple(global_feature_views(dataset)["global_all"] for dataset in datasets)
    factorized_rows: list[np.ndarray] = []
    graph_rows_by_split: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
    metadata: dict[str, Any] = {}
    for split_name, dataset, global_array in zip(
        ("train", "valid", "test"), datasets, global_views, strict=True
    ):
        graph_rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        split_factorized: list[np.ndarray] = []
        for index, data in enumerate(dataset):
            structure, attributes, roots, row_meta = _centre_rows(
                data,
                radius=int(radius),
                levels=int(levels),
                node_role_bins=int(node_role_bins),
                edge_role_bins=int(edge_role_bins),
            )
            graph_rows.append((structure, attributes, roots))
            split_factorized.append(
                np.concatenate([_distribution(structure), _distribution(attributes)])
            )
            if index and index % 500 == 0:
                print(f"[{split_name}] conditional features: {index}/{len(dataset)}", flush=True)
        factorized = np.concatenate(
            [np.asarray(global_array, dtype=np.float32), np.stack(split_factorized)], axis=1
        ).astype(np.float32, copy=False)
        graph_rows_by_split.append(graph_rows)
        factorized_rows.append(factorized)
        metadata[split_name] = {
            "n_graphs": int(len(dataset)),
            "mean_centres": float(np.mean([row[0].shape[0] for row in graph_rows])) if graph_rows else 0.0,
            "observed_roles_by_level": [
                int(np.unique(np.concatenate([row[2][:, level] for row in graph_rows])).size)
                for level in range(int(levels) + 1)
            ],
        }
    conditional_r3: list[sparse.csr_matrix] = []
    conditional_multi: list[sparse.csr_matrix] = []
    for split_name, graph_rows in zip(("train", "valid", "test"), graph_rows_by_split, strict=True):
        final, final_meta = _condition_graphs(
            graph_rows, levels=(int(levels),), role_bins=int(node_role_bins)
        )
        multi, multi_meta = _condition_graphs(
            graph_rows, levels=tuple(range(int(levels) + 1)), role_bins=int(node_role_bins)
        )
        conditional_r3.append(final)
        conditional_multi.append(multi)
        metadata[split_name]["condition_r3"] = final_meta
        metadata[split_name]["condition_multiscale"] = multi_meta
    features: dict[str, np.ndarray | sparse.csr_matrix] = {}
    for split_index, split_name in enumerate(("train", "valid", "test")):
        factorized = factorized_rows[split_index]
        features[f"{split_name}_s_factorized"] = factorized
        features[f"{split_name}_s_conditional_r3"] = sparse.hstack(
            [sparse.csr_matrix(factorized), conditional_r3[split_index]], format="csr", dtype=np.float32
        )
        features[f"{split_name}_s_conditional_multiscale"] = sparse.hstack(
            [sparse.csr_matrix(factorized), conditional_multi[split_index]], format="csr", dtype=np.float32
        )
    return features, metadata


def _cache_signature(config: Mapping[str, Any], script_path: Path) -> str:
    representation = config["representation"]
    payload = {
        "script_sha256": _sha256(script_path),
        "data_root": str(_resolve(config["data"]["root"])),
        "representation": _jsonable(representation),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _cache_paths(cache_dir: Path, split: str, view: str) -> Path:
    suffix = ".npy" if view == "s_factorized" else ".npz"
    return cache_dir / f"{split}_{view}{suffix}"


def _save_features(cache_dir: Path, features: Mapping[str, np.ndarray | sparse.csr_matrix], metadata: Mapping[str, Any], signature: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        for view in VIEW_NAMES:
            value = features[f"{split}_{view}"]
            path = _cache_paths(cache_dir, split, view)
            if view == "s_factorized":
                np.save(path, np.asarray(value, dtype=np.float32))
            else:
                sparse.save_npz(path, sparse.csr_matrix(value))
    _write_json_atomic(cache_dir / "manifest.json", {"signature": signature, "metadata": metadata})


def _load_or_build_features(
    datasets: Sequence[Any], config: Mapping[str, Any], script_path: Path
) -> tuple[dict[str, np.ndarray | sparse.csr_matrix], bool, dict[str, Any]]:
    cache_dir = _resolve(config["output"]["feature_cache_dir"])
    signature = _cache_signature(config, script_path)
    manifest_path = cache_dir / "manifest.json"
    complete = manifest_path.exists() and all(
        _cache_paths(cache_dir, split, view).exists()
        for split in ("train", "valid", "test")
        for view in VIEW_NAMES
    )
    if complete:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("signature") == signature:
            features: dict[str, np.ndarray | sparse.csr_matrix] = {}
            for split in ("train", "valid", "test"):
                for view in VIEW_NAMES:
                    path = _cache_paths(cache_dir, split, view)
                    features[f"{split}_{view}"] = (
                        np.load(path).astype(np.float32, copy=False)
                        if view == "s_factorized"
                        else sparse.load_npz(path).tocsr().astype(np.float32)
                    )
            return features, True, dict(manifest.get("metadata", {}))
    features, metadata = _build_features(
        datasets,
        radius=int(config["representation"]["radius"]),
        levels=int(config["representation"]["wl_levels"]),
        node_role_bins=int(config["representation"]["node_role_bins"]),
        edge_role_bins=int(config["representation"]["edge_role_bins"]),
    )
    _save_features(cache_dir, features, metadata, signature)
    return features, False, metadata


def _base_params(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    xgb_config = config["xgboost"]
    params = dict(xgb_config["params"])
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


def _params_from_trial(trial: optuna.Trial, ranges: Mapping[str, Sequence[float]]) -> dict[str, Any]:
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


def _fit_mae(x_train: Any, y_train: np.ndarray, x_valid: Any, y_valid: np.ndarray, params: Mapping[str, Any], config: Mapping[str, Any]) -> float:
    model_params = _base_params(config, int(config["xgboost"]["model_seed"]))
    model_params.update(dict(params))
    model = XGBRegressor(**model_params)
    model.fit(x_train, y_train, verbose=False)
    return float(mean_absolute_error(y_valid, model.predict(x_valid)))


def _tune_view(
    view: str,
    x_train: Any,
    y_train: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    tuning = config["tuning"]
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=int(tuning["seed"]) + VIEW_NAMES.index(view) * 1009),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ranges = tuning["ranges"]

    def objective(trial: optuna.Trial) -> float:
        params = _params_from_trial(trial, ranges)
        scores = [
            _fit_mae(x_train[train], y_train[train], x_train[valid], y_train[valid], params, config)
            for train, valid in folds
        ]
        return float(np.mean(scores))

    study.optimize(objective, n_trials=int(tuning["n_trials"]), show_progress_bar=False)
    best_params = dict(study.best_trial.params)
    fold_scores = [
        _fit_mae(x_train[train], y_train[train], x_train[valid], y_train[valid], best_params, config)
        for train, valid in folds
    ]
    return {
        "view": view,
        "objective": str(config["xgboost"].get("objective", "reg:absoluteerror")),
        "eval_metric": "mae",
        "selection_scope": "official-train inner shuffled KFold only",
        "n_trials": int(tuning["n_trials"]),
        "n_folds": int(len(folds)),
        "best_trial": int(study.best_trial.number),
        "best_cv_mae": float(np.mean(fold_scores)),
        "best_cv_fold_mae": fold_scores,
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


def _evaluate(x_train: Any, y_train: np.ndarray, x_eval: Any, y_eval: np.ndarray, params: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    model_params = _base_params(config, int(config["xgboost"]["model_seed"]))
    model_params.update(dict(params))
    model = XGBRegressor(**model_params)
    model.fit(x_train, y_train, verbose=False)
    prediction = np.asarray(model.predict(x_eval), dtype=np.float64)
    return {
        "seed": int(config["xgboost"]["model_seed"]),
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
    first_tuning = result["tuning"][VIEW_NAMES[0]]
    lines = [
        "# ZINC decomposable conditional structure--attribute statistics",
        "",
        "## Protocol",
        "",
        f"- split: `{result['data']['split']}`; sizes: `{result['data']['sizes']}`",
        f"- objective: `{result['protocol']['objective']}`; model seed: `{result['protocol']['model_seed']}`",
        f"- Optuna: `{first_tuning['n_trials']}` trials, `{first_tuning['n_folds']}` train-only folds",
        "- condition block: reusable root-WL-role × aligned attribute cross moments; no exact joint IDs",
        "",
        "## Results",
        "",
        "| view | dimension | train CV MAE | valid MAE | test MAE after train+valid refit |",
        "|---|---:|---:|---:|---:|",
    ]
    for view in VIEW_NAMES:
        tuning = result["tuning"][view]
        evaluation = result["evaluation"][view]
        lines.append(
            f"| `{view}` | {evaluation['dimension']} | {tuning['best_cv_mae']:.6f} | "
            f"{evaluation['valid']['mae']:.6f} | {evaluation['test_after_train_valid_refit']['mae']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"- CV-selected view: `{result['tuning']['selected_view_by_cv']}`",
            f"- feature cache hit: `{result['runtime']['feature_cache_hit']}`",
            "",
            "## Representation",
            "",
            "```json",
            json.dumps(result["representation"], ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
            f"Runtime: `{result['runtime']['seconds']:.1f}s`; script SHA-256: `{result['runtime']['script_sha256']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def run(config_path: Path) -> dict[str, Any]:
    start = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_root = _resolve(config["data"]["root"])
    script_path = Path(__file__).resolve()
    datasets = tuple(_load_zinc(data_root, split) for split in ("train", "val", "test"))
    sizes = tuple(len(dataset) for dataset in datasets)
    labels = tuple(
        np.asarray([float(data.y.view(-1)[0]) for data in dataset], dtype=np.float32)
        for dataset in datasets
    )
    features, cache_hit, feature_metadata = _load_or_build_features(datasets, config, script_path)
    x_train = {view: features[f"train_{view}"] for view in VIEW_NAMES}
    x_valid = {view: features[f"valid_{view}"] for view in VIEW_NAMES}
    x_test = {view: features[f"test_{view}"] for view in VIEW_NAMES}
    folds = [
        (train.astype(np.int64), valid.astype(np.int64))
        for train, valid in KFold(
            n_splits=int(config["tuning"]["n_folds"]),
            shuffle=True,
            random_state=int(config["tuning"]["split_seed"]),
        ).split(np.arange(sizes[0], dtype=np.int64))
    ]
    tuning: dict[str, Any] = {}
    for view in VIEW_NAMES:
        print(f"starting Optuna view={view}", flush=True)
        tuning[view] = _tune_view(view, x_train[view], labels[0], folds, config)
    selected_view = min(VIEW_NAMES, key=lambda view: tuning[view]["best_cv_mae"])
    tuning["selected_view_by_cv"] = selected_view
    evaluation: dict[str, Any] = {}
    for view in VIEW_NAMES:
        best_params = tuning[view]["best_params"]
        combined_x = sparse.vstack([x_train[view], x_valid[view]], format="csr") if sparse.issparse(x_train[view]) else np.concatenate([x_train[view], x_valid[view]], axis=0)
        evaluation[view] = {
            "dimension": int(x_train[view].shape[1]),
            "valid": _evaluate(x_train[view], labels[0], x_valid[view], labels[1], best_params, config),
            "test_after_train_valid_refit": _evaluate(
                combined_x,
                np.concatenate(labels[:2]),
                x_test[view],
                labels[2],
                best_params,
                config,
            ),
        }
    result = {
        "protocol_id": str(config["protocol_id"]),
        "status": "completed",
        "data": {
            "root": str(data_root),
            "split": "PyG ZINC subset=True official train/val/test",
            "sizes": {"train": sizes[0], "valid": sizes[1], "test": sizes[2]},
            "source": source_audit(data_root),
            "target": {
                name: _label_summary(values)
                for name, values in zip(("train", "valid", "test"), labels, strict=True)
            },
        },
        "representation": {
            "radius": int(config["representation"]["radius"]),
            "wl_levels": list(range(int(config["representation"]["wl_levels"]) + 1)),
            "centres": "every atom",
            "node_role_bins": int(config["representation"]["node_role_bins"]),
            "edge_role_bins": int(config["representation"]["edge_role_bins"]),
            "structure_definition": "topology-only rooted-WL final root role + role/edge histograms + shell/size/cycle statistics",
            "attribute_definition": "centre atom + incident bond histogram + local atom/bond histograms",
            "conditional_definition": "E[1{root role=r} * aligned attribute vector] per selected WL level",
            "feature_metadata": feature_metadata,
        },
        "protocol": {
            "objective": str(config["xgboost"].get("objective", "reg:absoluteerror")),
            "eval_metric": "mae",
            "model_seed": int(config["xgboost"]["model_seed"]),
            "valid_scope": "official train only after train-internal Optuna",
            "test_scope": "official train+valid refit after frozen train-internal Optuna",
        },
        "tuning": tuning,
        "evaluation": evaluation,
        "runtime": {
            "seconds": float(time.perf_counter() - start),
            "feature_cache_hit": bool(cache_hit),
            "python": platform.python_version(),
            "numpy": importlib.metadata.version("numpy"),
            "scipy": importlib.metadata.version("scipy"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "xgboost": importlib.metadata.version("xgboost"),
            "script_sha256": _sha256(script_path),
        },
    }
    output = config["output"]
    result_json = _resolve(output["json"])
    result_markdown = _resolve(output["markdown"])
    _write_json_atomic(result_json, result)
    result_markdown.parent.mkdir(parents=True, exist_ok=True)
    result_markdown.write_text(_render_markdown(result), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    result = run(args.config)
    for view in VIEW_NAMES:
        print(
            f"{view}: cv_mae={result['tuning'][view]['best_cv_mae']:.6f} "
            f"valid_mae={result['evaluation'][view]['valid']['mae']:.6f} "
            f"test_mae={result['evaluation'][view]['test_after_train_valid_refit']['mae']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
