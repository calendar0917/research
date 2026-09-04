"""ZINC Step A: collision-free typed motif counts and matched baselines.

The experiment is deliberately narrower than the older ``S + marginal``
runner.  A radius-3 rooted typed-WL object is constructed for every atom
centre.  Its root colour is retained at WL rounds 0, 1, 2 and 3.  A
train-fold-only top-K vocabulary maps those discrete objects to sparse graph
features; no fixed hash bins are used for the proposed representation.

The primary comparison is:

``S``
    Existing 62-dimensional graph statistics.
``S + WL-count``
    Raw and centre-normalized counts of the four typed-WL vocabularies,
    with one explicit OOV column per round and per count type.
``S + Morgan-count``
    A fixed 2048-bin count-ECFP reference implemented directly on the ZINC
    labelled graph.  This is an RDKit-free reference because the current
    environment does not ship RDKit; its atom/bond invariants are recorded in
    the manifest.
``matched GINE``
    A fixed edge-aware message-passing regressor trained with L1 loss on the
    same official split.

XGBoost vocabulary fitting and parameter selection use only inner folds of
the official training split.  The official validation split is evaluated once
after that search.  The official test split is evaluated only after the
selected view is frozen, with a train+validation refit.  The module is also
usable for small smoke runs through the YAML configuration.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import random
import sys
import time
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


DEFAULT_CONFIG = REPO_ROOT / "tracks/ksvd/configs/luyin16/zinc_motif_count.yaml"
WL_ROUNDS = 4
MORGAN_BITS = 2048
ATOM_CATEGORIES = 28
BOND_CATEGORIES = 4

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


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _ExactTokenRegistry:
    """Stable compact labels for exact recursive WL expressions.

    The feature vocabulary never uses a small modulo bucket.  Internally a
    128-bit digest makes recursively nested colours cheap to carry between
    rounds, while the registry retains the complete immutable expression and
    raises if two different expressions ever receive the same digest.  A
    registry shared by the whole token cache therefore gives an explicit
    observed collision audit, rather than silently accepting a hash collision.
    """

    def __init__(self, namespace: str) -> None:
        self.namespace = str(namespace)
        self._digest_to_key: dict[bytes, object] = {}
        self.collision_count = 0

    def intern(self, key: object) -> bytes:
        namespaced = (self.namespace, key)
        payload = repr(namespaced).encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=16).digest()
        existing = self._digest_to_key.get(digest)
        if existing is not None and existing != namespaced:
            self.collision_count += 1
            raise RuntimeError("exact token digest collision detected")
        self._digest_to_key[digest] = namespaced
        return digest

    @property
    def size(self) -> int:
        return len(self._digest_to_key)


def _stable_hash(namespace: str, value: object) -> int:
    """Stable hash reserved for the Morgan bit reference only."""
    payload = (namespace + "|" + repr(value)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _distances(graph: Any, centre: int, radius: int) -> dict[int, int]:
    distances = {int(centre): 0}
    queue: deque[int] = deque([int(centre)])
    while queue:
        node = queue.popleft()
        if distances[node] >= int(radius):
            continue
        for neighbour in sorted(graph.neighbors(node)):
            if neighbour not in distances:
                distances[neighbour] = distances[node] + 1
                queue.append(neighbour)
    return distances


def _typed_wl_tokens(
    data: Any,
    *,
    radius: int = 3,
    rounds: int = WL_ROUNDS - 1,
    registry: _ExactTokenRegistry | None = None,
) -> tuple[tuple[tuple[object, ...], ...], ...]:
    """Return exact rooted typed-WL tokens for every centre and round.

    The nested tuples are intentional.  They are exact Python keys rather
    than 64-bit hash bins.  Neighbour lists are sorted by their semantic
    colours and bond types, so graph-node relabeling does not change a token.
    The return shape is ``(round, centre)``.
    """
    if radius < 0 or rounds < 0:
        raise ValueError("radius and rounds must be non-negative")
    graph, node_types, edge_types = _data_to_graph(data)
    token_registry = registry or _ExactTokenRegistry("typed-rooted-wl")
    round_rows: list[list[tuple[int, bytes]]] = [[] for _ in range(int(rounds) + 1)]
    for centre in graph.nodes:
        distance = _distances(graph, int(centre), int(radius))
        nodes = tuple(sorted(distance))
        induced = graph.induced(set(nodes))
        colours: dict[int, bytes] = {
            node: token_registry.intern(
                (
                    "initial",
                    int(node == centre),
                    int(distance[node]),
                    int(len(induced.neighbors(node))),
                    int(node_types[node]),
                )
            )
            for node in nodes
        }
        for current_round in range(int(rounds) + 1):
            # Round is kept outside the token map, but adding it to the
            # semantic key makes accidental cross-round mixing impossible in
            # downstream exact-vocabulary code.
            round_rows[current_round].append((current_round, colours[int(centre)]))
            if current_round == int(rounds):
                break
            next_colours: dict[int, bytes] = {}
            for node in nodes:
                messages = tuple(
                    sorted(
                        (
                            int(edge_types[graph.edge_key(node, neighbour)]),
                            colours[neighbour],
                        )
                        for neighbour in induced.neighbors(node)
                    )
                )
                next_colours[node] = token_registry.intern(("refine", colours[node], messages))
            colours = next_colours
    return tuple(tuple(row) for row in round_rows)


def _initial_morgan_invariants(
    graph: Any,
    node_types: np.ndarray,
    edge_types: Mapping[tuple[int, int], int],
) -> dict[int, object]:
    values: dict[int, object] = {}
    for node in graph.nodes:
        bond_counts = [0, 0, 0, 0]
        for neighbour in graph.neighbors(node):
            bond_counts[int(edge_types[graph.edge_key(node, neighbour)])] += 1
        values[int(node)] = (
            "morgan-v1",
            int(node_types[node]),
            int(len(graph.neighbors(node))),
            tuple(bond_counts),
        )
    return values


def _morgan_tokens(
    data: Any,
    rounds: int = WL_ROUNDS - 1,
    registry: _ExactTokenRegistry | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Return radius-0..3 count-ECFP environments for all atoms.

    This is the standard circular update pattern, using the discrete ZINC
    atom and bond categories.  The implementation avoids an RDKit dependency;
    the fixed hashed reference is intentionally labelled as an in-repo
    count-ECFP/Morgan analogue in the result manifest.
    """
    graph, node_types, edge_types = _data_to_graph(data)
    token_registry = registry or _ExactTokenRegistry("morgan")
    initial = _initial_morgan_invariants(graph, node_types, edge_types)
    colours: dict[int, bytes] = {
        node: token_registry.intern(("initial", value)) for node, value in initial.items()
    }
    rows: list[list[tuple[int, bytes]]] = [[] for _ in range(int(rounds) + 1)]
    for current_round in range(int(rounds) + 1):
        rows[current_round].extend((current_round, colours[int(node)]) for node in graph.nodes)
        if current_round == int(rounds):
            break
        next_colours: dict[int, bytes] = {}
        for node in graph.nodes:
            messages = tuple(
                sorted(
                    (
                        int(edge_types[graph.edge_key(node, neighbour)]),
                        colours[neighbour],
                    )
                    for neighbour in graph.neighbors(node)
                )
            )
            next_colours[int(node)] = token_registry.intern(("refine", colours[int(node)], messages))
        colours = next_colours
    return tuple(tuple(row) for row in rows)


def _count_ecfp_row(
    tokens: Sequence[Sequence[object]],
    *,
    bits: int = MORGAN_BITS,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros(int(bits), dtype=np.float32)
    for current_round, row in enumerate(tokens):
        for token in row:
            counts[_stable_hash(f"count-ecfp-r{current_round}", token) % int(bits)] += 1.0
    normalized = counts / max(float(sum(len(row) for row in tokens)), 1.0)
    return counts, normalized.astype(np.float32, copy=False)


def _build_token_cache(
    datasets: Sequence[Any],
    *,
    radius: int,
    rounds: int,
    progress_every: int = 500,
) -> tuple[list[tuple[tuple[object, ...], ...]], list[tuple[tuple[object, ...], ...]], dict[str, Any]]:
    wl_rows: list[tuple[tuple[object, ...], ...]] = []
    morgan_rows: list[tuple[tuple[object, ...], ...]] = []
    split_meta: dict[str, Any] = {}
    wl_registry = _ExactTokenRegistry("typed-rooted-wl")
    morgan_registry = _ExactTokenRegistry("morgan")
    for split_name, dataset in zip(("train", "valid", "test"), datasets, strict=True):
        start = len(wl_rows)
        node_counts: list[int] = []
        for index, data in enumerate(dataset):
            wl = _typed_wl_tokens(data, radius=radius, rounds=rounds, registry=wl_registry)
            morgan = _morgan_tokens(data, rounds=rounds, registry=morgan_registry)
            wl_rows.append(wl)
            morgan_rows.append(morgan)
            node_counts.append(int(data.num_nodes))
            if progress_every > 0 and (index + 1) % progress_every == 0:
                print(f"[{split_name}] token graphs: {index + 1}/{len(dataset)}", flush=True)
        split_meta[split_name] = {
            "n_graphs": int(len(dataset)),
            "mean_centres": float(np.mean(node_counts)) if node_counts else 0.0,
            "row_offset": int(start),
        }
    split_meta["registry"] = {
        "typed_wl_unique_tokens": int(wl_registry.size),
        "morgan_unique_tokens": int(morgan_registry.size),
        "typed_wl_digest_collisions": int(wl_registry.collision_count),
        "morgan_digest_collisions": int(morgan_registry.collision_count),
        "digest_bits": 128,
    }
    return wl_rows, morgan_rows, split_meta


def _cache_to_rows(archive: Any) -> tuple[list[Any], list[Any], dict[str, Any]]:
    """Decode object arrays without relying on an accidental ndarray shape."""
    wl_array = np.asarray(archive["wl_rows"], dtype=object).reshape(-1)
    morgan_array = np.asarray(archive["morgan_rows"], dtype=object).reshape(-1)
    wl_rows = [item for item in wl_array.tolist()]
    morgan_rows = [item for item in morgan_array.tolist()]
    split_meta = json.loads(str(np.asarray(archive["split_meta"]).reshape(-1)[0]))
    return wl_rows, morgan_rows, split_meta


def _object_row_array(rows: Sequence[Any]) -> np.ndarray:
    output = np.empty(len(rows), dtype=object)
    output[:] = list(rows)
    return output


def _fit_topk_vocab(
    rows: Sequence[tuple[tuple[object, ...], ...]],
    indices: Sequence[int],
    *,
    rounds: int,
    top_k: int,
) -> tuple[dict[object, int], ...]:
    vocabularies: list[dict[object, int]] = []
    for current_round in range(int(rounds)):
        counts: Counter[object] = Counter()
        for index in indices:
            counts.update(rows[int(index)][current_round])
        ranked = sorted(counts.items(), key=lambda item: (-int(item[1]), repr(item[0])))[: int(top_k)]
        vocabularies.append({token: column for column, (token, _) in enumerate(ranked)})
    return tuple(vocabularies)


def _encode_wl_counts(
    rows: Sequence[tuple[tuple[object, ...], ...]],
    vocabularies: Sequence[Mapping[object, int]],
    *,
    rounds: int,
    top_k: int,
) -> sparse.csr_matrix:
    width_per_round = 2 * (int(top_k) + 1)
    width = int(rounds) * width_per_round
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    for graph_index, graph_rows in enumerate(rows):
        for current_round in range(int(rounds)):
            tokens = graph_rows[current_round]
            counts = Counter(tokens)
            raw_offset = current_round * width_per_round
            norm_offset = raw_offset + int(top_k) + 1
            unknown = 0
            for token, count in counts.items():
                column = vocabularies[current_round].get(token)
                if column is None:
                    unknown += int(count)
                    continue
                row_indices.append(graph_index)
                column_indices.append(raw_offset + int(column))
                values.append(float(count))
                row_indices.append(graph_index)
                column_indices.append(norm_offset + int(column))
                values.append(float(count) / max(float(len(tokens)), 1.0))
            if unknown:
                row_indices.append(graph_index)
                column_indices.append(raw_offset + int(top_k))
                values.append(float(unknown))
                row_indices.append(graph_index)
                column_indices.append(norm_offset + int(top_k))
                values.append(float(unknown) / max(float(len(tokens)), 1.0))
    matrix = sparse.coo_matrix(
        (np.asarray(values, dtype=np.float32), (row_indices, column_indices)),
        shape=(len(rows), width),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    return matrix


def _build_morgan_counts(
    rows: Sequence[tuple[tuple[object, ...], ...]],
    *,
    bits: int,
) -> sparse.csr_matrix:
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    for graph_index, tokens in enumerate(rows):
        counts, normalized = _count_ecfp_row(tokens, bits=bits)
        for column in np.flatnonzero(counts):
            row_indices.append(graph_index)
            column_indices.append(int(column))
            values.append(float(counts[column]))
            row_indices.append(graph_index)
            column_indices.append(int(bits) + int(column))
            values.append(float(normalized[column]))
    return sparse.coo_matrix(
        (np.asarray(values, dtype=np.float32), (row_indices, column_indices)),
        shape=(len(rows), 2 * int(bits)),
        dtype=np.float32,
    ).tocsr()


def _hstack_dense_sparse(dense: np.ndarray, extra: sparse.spmatrix) -> sparse.csr_matrix:
    return sparse.hstack(
        [sparse.csr_matrix(np.asarray(dense, dtype=np.float32)), extra], format="csr", dtype=np.float32
    )


def _xgb_base(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    return {
        "objective": str(config.get("xgboost_objective", "reg:absoluteerror")),
        "eval_metric": "mae",
        "tree_method": "hist",
        "max_bin": int(config.get("max_bin", 256)),
        "random_state": int(seed),
        "n_jobs": int(config.get("n_jobs", 4)),
    }


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


def _fit_xgb_mae(
    x_train: sparse.spmatrix | np.ndarray,
    y_train: np.ndarray,
    x_valid: sparse.spmatrix | np.ndarray,
    y_valid: np.ndarray,
    params: Mapping[str, Any],
    config: Mapping[str, Any],
    seed: int,
) -> float:
    model = XGBRegressor(**_xgb_base(config, seed), **dict(params))
    model.fit(x_train, y_train)
    return float(mean_absolute_error(y_valid, model.predict(x_valid)))


def _tune_xgb_view(
    name: str,
    fold_matrices: Sequence[tuple[sparse.spmatrix | np.ndarray, sparse.spmatrix | np.ndarray]],
    y: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    tuning = config["tuning"]
    ranges = tuning["ranges"]
    # Keep the sampler seed deterministic across processes.
    seed = int(tuning["seed"]) + _stable_hash("xgb-view-seed", name) % 100003
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = _params_from_trial(trial, ranges)
        scores = []
        for fold_id, (train_indices, valid_indices) in enumerate(folds):
            x_train, x_valid = fold_matrices[fold_id]
            scores.append(
                _fit_xgb_mae(
                    x_train[train_indices],
                    y[train_indices],
                    x_train[valid_indices],
                    y[valid_indices],
                    params,
                    config,
                    seed,
                )
            )
        return float(np.mean(scores))

    study.optimize(objective, n_trials=int(tuning["n_trials"]), show_progress_bar=False)
    best = dict(study.best_trial.params)
    # Recompute fold scores explicitly so the reported values are tied to the
    # exact selected parameters and not to a pruned/interrupted trial.
    best_scores = []
    for fold_id, (train_indices, valid_indices) in enumerate(folds):
        x_train, x_valid = fold_matrices[fold_id]
        best_scores.append(
            _fit_xgb_mae(
                x_train[train_indices],
                y[train_indices],
                x_train[valid_indices],
                y[valid_indices],
                best,
                config,
                seed,
            )
        )
    return {
        "view": name,
        "objective": str(config.get("xgboost_objective", "reg:absoluteerror")),
        "eval_metric": "mae",
        "selection_scope": "official-train inner KFold only",
        "n_trials": int(tuning["n_trials"]),
        "n_folds": int(len(folds)),
        "best_trial": int(study.best_trial.number),
        "best_cv_mae": float(np.mean(best_scores)),
        "best_cv_fold_mae": best_scores,
        "best_params": best,
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


def _evaluate_xgb(
    x_train: sparse.spmatrix | np.ndarray,
    y_train: np.ndarray,
    x_eval: sparse.spmatrix | np.ndarray,
    y_eval: np.ndarray,
    params: Mapping[str, Any],
    config: Mapping[str, Any],
    seeds: Sequence[int],
) -> dict[str, Any]:
    predictions: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        model = XGBRegressor(**_xgb_base(config, int(seed)), **dict(params))
        model.fit(x_train, y_train)
        prediction = np.asarray(model.predict(x_eval), dtype=np.float64)
        predictions.append(prediction)
        rows.append({"seed": int(seed), "mae": float(mean_absolute_error(y_eval, prediction))})
    ensemble = np.mean(np.stack(predictions, axis=0), axis=0)
    return {
        "rows": rows,
        "mean_mae": float(np.mean([row["mae"] for row in rows])),
        "std_mae": float(np.std([row["mae"] for row in rows])),
        "seed_ensemble_mae": float(mean_absolute_error(y_eval, ensemble)),
    }


def _seed_everything(seed: int, torch: Any) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def _make_gine_model(torch: Any, nn: Any, F: Any, hidden: int, layers: int) -> Any:
    from torch_geometric.nn import GINEConv

    class MatchedGINE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.atom_embedding = nn.Embedding(ATOM_CATEGORIES, hidden)
            self.bond_embedding = nn.Embedding(BOND_CATEGORIES, hidden)
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            self.readout_projections = nn.ModuleList()
            for _ in range(int(layers)):
                mlp = nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden),
                )
                self.convs.append(GINEConv(mlp, train_eps=True))
                self.bns.append(nn.BatchNorm1d(hidden))
                self.readout_projections.append(nn.Linear(hidden, hidden))
            self.head = nn.Sequential(nn.ReLU(), nn.Linear(hidden, 1))

        def forward(self, data: Any) -> Any:
            from torch_geometric.nn import global_add_pool

            x = self.atom_embedding(data.x.view(-1).long())
            edge_attr = self.bond_embedding(data.edge_attr.view(-1).long())
            states = []
            for conv, batch_norm, projection in zip(
                self.convs, self.bns, self.readout_projections, strict=True
            ):
                x = F.relu(batch_norm(conv(x, data.edge_index, edge_attr)))
                states.append(projection(global_add_pool(x, data.batch)))
            graph_state = torch.stack(states, dim=0).sum(dim=0) / max(float(len(states)), 1.0)
            return self.head(graph_state).view(-1)

    return MatchedGINE


def _run_gine_phase(
    config: Mapping[str, Any],
    *,
    train_rows_source: Sequence[Any],
    eval_rows_source: Sequence[Any],
    train_y: np.ndarray,
    eval_y: np.ndarray,
    phase: str,
) -> dict[str, Any]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.loader import DataLoader
    except ImportError as exc:  # pragma: no cover - dependency is in ksvd group
        raise RuntimeError(f"matched GINE dependencies unavailable: {exc}") from exc

    gine_config = config["gine"]
    hidden = int(gine_config["hidden"])
    layers = int(gine_config["layers"])
    epochs = int(gine_config["epochs"])
    batch_size = int(gine_config["batch_size"])
    lr = float(gine_config["learning_rate"])
    device = torch.device(str(gine_config.get("device", "cpu")))
    model_seeds = [int(value) for value in gine_config["model_seeds"]]
    all_rows = list(train_rows_source)
    eval_rows = list(eval_rows_source)
    train_y = np.asarray(train_y, dtype=np.float32)
    eval_y = np.asarray(eval_y, dtype=np.float32)
    # Dataset rows already carry y, but recreate it so train+valid refits use
    # exactly the requested index arrays and no hidden split state.
    train_data = []
    eval_data = []
    for data, y_value in zip(all_rows, train_y, strict=True):
        copied = data.clone()
        copied.y = torch.tensor([float(y_value)], dtype=torch.float32)
        train_data.append(copied)
    for data, y_value in zip(eval_rows, eval_y, strict=True):
        copied = data.clone()
        copied.y = torch.tensor([float(y_value)], dtype=torch.float32)
        eval_data.append(copied)

    predictions: list[np.ndarray] = []
    seed_rows: list[dict[str, Any]] = []
    history: dict[str, Any] = {}
    trainable_parameters: int | None = None
    for model_seed in model_seeds:
        _seed_everything(model_seed, torch)
        model_class = _make_gine_model(torch, nn, F, hidden, layers)
        model = model_class().to(device)
        if trainable_parameters is None:
            trainable_parameters = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_generator = torch.Generator().manual_seed(model_seed + 91011)
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            generator=train_generator,
            num_workers=0,
        )
        eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False, num_workers=0)
        losses: list[float] = []
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            seen = 0
            for batch in train_loader:
                batch = batch.to(device)
                prediction = model(batch)
                target = batch.y.view(-1).float()
                loss = F.l1_loss(prediction, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * len(target)
                seen += len(target)
            epoch_loss = total_loss / max(seen, 1)
            losses.append(epoch_loss)
            if epoch == 1 or epoch % max(1, epochs // 5) == 0 or epoch == epochs:
                print(
                    f"gine phase={phase} seed={model_seed} epoch={epoch:03d} l1={epoch_loss:.6f}",
                    flush=True,
                )
        model.eval()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for batch in eval_loader:
                outputs.append(model(batch.to(device)).cpu().numpy())
        prediction = np.concatenate(outputs).astype(np.float64, copy=False)
        predictions.append(prediction)
        seed_rows.append({"seed": model_seed, "mae": float(mean_absolute_error(eval_y, prediction))})
        history[str(model_seed)] = losses
        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    ensemble = np.mean(np.stack(predictions, axis=0), axis=0)
    return {
        "phase": phase,
        "rows": seed_rows,
        "mean_mae": float(np.mean([row["mae"] for row in seed_rows])),
        "std_mae": float(np.std([row["mae"] for row in seed_rows])),
        "seed_ensemble_mae": float(mean_absolute_error(eval_y, ensemble)),
        "training": {
            "hidden": hidden,
            "layers": layers,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "loss": "L1 / mean absolute error",
            "readout": "mean of layer-wise global-add states",
            "trainable_parameters": int(trainable_parameters or 0),
            "history": history,
        },
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


def _feature_signature(config: Mapping[str, Any], data_root: Path) -> str:
    payload = {
        "schema": "zinc-step-a-motif-count-v1",
        "representation": dict(config["representation"]),
        "data_root": str(data_root.resolve()),
        "encoder_sha256": _sha256_path(Path(__file__).resolve()),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _build_global_s(datasets: Sequence[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    blocks = [global_feature_views(dataset)["global_all"] for dataset in datasets]
    return tuple(np.asarray(block, dtype=np.float32) for block in blocks)  # type: ignore[return-value]


def _limit_dataset(dataset: Any, limit: int | None) -> Any:
    if limit is None or int(limit) <= 0 or int(limit) >= len(dataset):
        return dataset
    return [dataset[index] for index in range(int(limit))]


def _vocab_summary(
    rows: Sequence[tuple[tuple[object, ...], ...]],
    indices: Sequence[int],
    vocabularies: Sequence[Mapping[object, int]],
    *,
    rounds: int,
) -> dict[str, Any]:
    unknown_rates = []
    vocabulary_sizes = []
    total_counts = []
    for current_round in range(int(rounds)):
        total = 0
        unknown = 0
        for index in indices:
            tokens = rows[int(index)][current_round]
            total += len(tokens)
            unknown += sum(1 for token in tokens if token not in vocabularies[current_round])
        total_counts.append(int(total))
        unknown_rates.append(float(unknown / max(total, 1)))
        vocabulary_sizes.append(int(len(vocabularies[current_round])))
    return {
        "vocabulary_sizes": vocabulary_sizes,
        "token_counts": total_counts,
        "unknown_rates": unknown_rates,
    }


def run(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_root = _resolve(config["data"]["root"])
    output = config["output"]
    result_json = _resolve(output["json"])
    result_markdown = _resolve(output["markdown"])
    start = time.perf_counter()
    loaded_datasets = tuple(_load_zinc(data_root, split) for split in ("train", "val", "test"))
    limits = config.get("limits", {})
    datasets = tuple(
        _limit_dataset(loaded, limits.get(split))
        for loaded, split in zip(loaded_datasets, ("train", "valid", "test"), strict=True)
    )
    labels = tuple(
        np.asarray([float(data.y.view(-1)[0]) for data in dataset], dtype=np.float32)
        for dataset in datasets
    )
    representation = config["representation"]
    radius = int(representation.get("radius", 3))
    if radius != 3:
        raise ValueError("Step A is fixed to radius-3")
    rounds = int(representation.get("wl_rounds", 3)) + 1
    if rounds != WL_ROUNDS:
        raise ValueError("Step A requires WL rounds 0/1/2/3")
    top_k = int(representation.get("top_k", 2048))
    morgan_bits = int(representation.get("morgan_bits", MORGAN_BITS))
    if top_k <= 0 or morgan_bits <= 0:
        raise ValueError("top_k and morgan_bits must be positive")

    token_cache_path = _resolve(output["token_cache"])
    token_signature = _feature_signature(config, data_root)
    cache_hit = False
    if token_cache_path.exists():
        with np.load(token_cache_path, allow_pickle=True) as archive:
            cached_signature = str(np.asarray(archive["signature"]).reshape(-1)[0])
            if cached_signature != token_signature:
                raise ValueError(f"token cache signature mismatch: {token_cache_path}")
            # Pickle is necessary for exact nested tuple keys; this file is a
            # local generated cache and never enters a model input directly.
            wl_rows, morgan_rows, split_meta = _cache_to_rows(archive)
        cache_hit = True
    else:
        wl_rows, morgan_rows, split_meta = _build_token_cache(
            datasets,
            radius=radius,
            rounds=rounds - 1,
            progress_every=int(config.get("progress_every", 500)),
        )
        token_cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = token_cache_path.with_suffix(token_cache_path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                signature=np.asarray([token_signature]),
                # Force one object per graph.  Without the explicit empty
                # allocation NumPy sees the uniform four-round prefix and
                # silently creates an (n_graphs, 4) object matrix, which
                # would flatten the graph/round boundary on reload.
                wl_rows=_object_row_array(wl_rows),
                morgan_rows=_object_row_array(morgan_rows),
                split_meta=np.asarray([json.dumps(_jsonable(split_meta), sort_keys=True)]),
            )
        temporary.replace(token_cache_path)

    split_offsets = [0, len(datasets[0]), len(datasets[0]) + len(datasets[1])]
    all_wl_rows = wl_rows
    all_morgan_rows = morgan_rows
    train_rows = all_wl_rows[split_offsets[0] : split_offsets[1]]
    valid_rows = all_wl_rows[split_offsets[1] : split_offsets[2]]
    test_rows = all_wl_rows[split_offsets[2] :]
    train_morgan = all_morgan_rows[split_offsets[0] : split_offsets[1]]
    valid_morgan = all_morgan_rows[split_offsets[1] : split_offsets[2]]
    test_morgan = all_morgan_rows[split_offsets[2] :]
    s_train, s_valid, s_test = _build_global_s(datasets)

    n_train = len(datasets[0])
    kfold = KFold(
        n_splits=int(config["tuning"]["n_splits"]),
        shuffle=True,
        random_state=int(config["tuning"]["split_seed"]),
    )
    folds = [
        (train.astype(np.int64), valid.astype(np.int64))
        for train, valid in kfold.split(np.arange(n_train, dtype=np.int64))
    ]
    # Every outer fold gets its own top-K vocabulary.  The matrices contain
    # all official-train rows only, but the vocabulary itself sees only the
    # corresponding fold's fit indices.
    vocab_meta: list[dict[str, Any]] = []

    # Build the actual candidate matrices for each fold.  Keeping the vocab
    # fit and matrix construction explicit makes leakage audits straightforward.
    fold_views: dict[str, list[tuple[sparse.spmatrix | np.ndarray, sparse.spmatrix | np.ndarray]]] = {
        "s": [],
        "wl_count": [],
        "s_wl_count": [],
        "morgan_count": [],
        "s_morgan_count": [],
    }
    for fold_id, (fold_train, fold_valid) in enumerate(folds):
        vocab = _fit_topk_vocab(train_rows, fold_train, rounds=rounds, top_k=top_k)
        wl_matrix = _encode_wl_counts(train_rows, vocab, rounds=rounds, top_k=top_k)
        morgan_matrix = _build_morgan_counts(train_morgan, bits=morgan_bits)
        vocab_meta.append(
            {
                "fold": int(fold_id),
                "fit_graphs": int(len(fold_train)),
                "heldout_graphs": int(len(fold_valid)),
                "wl": _vocab_summary(train_rows, fold_train, vocab, rounds=rounds),
                "heldout_wl": _vocab_summary(train_rows, fold_valid, vocab, rounds=rounds),
            }
        )
        fold_views["s"].append((s_train, s_train))
        fold_views["wl_count"].append((wl_matrix, wl_matrix))
        fold_views["s_wl_count"].append((_hstack_dense_sparse(s_train, wl_matrix), _hstack_dense_sparse(s_train, wl_matrix)))
        fold_views["morgan_count"].append((morgan_matrix, morgan_matrix))
        fold_views["s_morgan_count"].append((_hstack_dense_sparse(s_train, morgan_matrix), _hstack_dense_sparse(s_train, morgan_matrix)))

    y_train = labels[0]
    tuning_results: dict[str, Any] = {}
    for view_name in ("s", "wl_count", "s_wl_count", "morgan_count", "s_morgan_count"):
        print(f"Optuna tuning: {view_name} ({int(config['tuning']['n_trials'])} trials)", flush=True)
        tuning_results[view_name] = _tune_xgb_view(
            view_name,
            fold_views[view_name],
            y_train,
            folds,
            config,
        )
        print(
            f"  {view_name}: CV MAE={tuning_results[view_name]['best_cv_mae']:.6f}",
            flush=True,
        )

    # Final train-only vocabulary for the one-time official-valid report.
    full_train_indices = np.arange(n_train, dtype=np.int64)
    full_vocab = _fit_topk_vocab(train_rows, full_train_indices, rounds=rounds, top_k=top_k)
    wl_train = _encode_wl_counts(train_rows, full_vocab, rounds=rounds, top_k=top_k)
    wl_valid = _encode_wl_counts(valid_rows, full_vocab, rounds=rounds, top_k=top_k)
    wl_test_train_vocab = _encode_wl_counts(test_rows, full_vocab, rounds=rounds, top_k=top_k)
    morgan_train = _build_morgan_counts(train_morgan, bits=morgan_bits)
    morgan_valid = _build_morgan_counts(valid_morgan, bits=morgan_bits)
    morgan_test = _build_morgan_counts(test_morgan, bits=morgan_bits)
    final_views = {
        "s": (s_train, s_valid, s_test),
        "wl_count": (wl_train, wl_valid, wl_test_train_vocab),
        "s_wl_count": (
            _hstack_dense_sparse(s_train, wl_train),
            _hstack_dense_sparse(s_valid, wl_valid),
            _hstack_dense_sparse(s_test, wl_test_train_vocab),
        ),
        "morgan_count": (morgan_train, morgan_valid, morgan_test),
        "s_morgan_count": (
            _hstack_dense_sparse(s_train, morgan_train),
            _hstack_dense_sparse(s_valid, morgan_valid),
            _hstack_dense_sparse(s_test, morgan_test),
        ),
    }
    model_seeds = [int(value) for value in config["model_seeds"]]
    xgb_evaluation: dict[str, Any] = {}
    for view_name, (x_train, x_valid, x_test_train_vocab) in final_views.items():
        params = tuning_results[view_name]["best_params"]
        valid_result = _evaluate_xgb(x_train, y_train, x_valid, labels[1], params, config, model_seeds)

        # For the frozen test refit, the validation graphs become labelled
        # training data.  Refit the top-K vocab on train+valid (never test),
        # preserving the fixed K-dimensional schema and the OOV column.
        train_valid_wl = None
        test_wl = None
        if view_name in ("wl_count", "s_wl_count"):
            combined_wl = train_rows + valid_rows
            combined_indices = np.arange(len(combined_wl), dtype=np.int64)
            combined_vocab = _fit_topk_vocab(combined_wl, combined_indices, rounds=rounds, top_k=top_k)
            train_valid_wl = _encode_wl_counts(combined_wl, combined_vocab, rounds=rounds, top_k=top_k)
            test_wl = _encode_wl_counts(test_rows, combined_vocab, rounds=rounds, top_k=top_k)
        if view_name in ("morgan_count", "s_morgan_count"):
            combined_morgan_rows = train_morgan + valid_morgan
            train_valid_morgan = _build_morgan_counts(combined_morgan_rows, bits=morgan_bits)
            test_morgan_refit = morgan_test
        else:
            train_valid_morgan = None
            test_morgan_refit = None
        if view_name == "s":
            refit_train = np.concatenate([s_train, s_valid], axis=0)
            refit_test = s_test
        elif view_name == "wl_count":
            refit_train = train_valid_wl
            refit_test = test_wl
        elif view_name == "s_wl_count":
            refit_train = _hstack_dense_sparse(np.concatenate([s_train, s_valid], axis=0), train_valid_wl)
            refit_test = _hstack_dense_sparse(s_test, test_wl)
        elif view_name == "morgan_count":
            refit_train = train_valid_morgan
            refit_test = test_morgan_refit
        elif view_name == "s_morgan_count":
            refit_train = _hstack_dense_sparse(np.concatenate([s_train, s_valid], axis=0), train_valid_morgan)
            refit_test = _hstack_dense_sparse(s_test, test_morgan_refit)
        else:  # pragma: no cover
            raise AssertionError(view_name)
        refit_y = np.concatenate([labels[0], labels[1]], axis=0)
        test_result = _evaluate_xgb(refit_train, refit_y, refit_test, labels[2], params, config, model_seeds)
        xgb_evaluation[view_name] = {
            "dimension": int(x_train.shape[1]),
            "sparsity": {
                "train_nnz": int(x_train.nnz if sparse.issparse(x_train) else np.count_nonzero(x_train)),
                "valid_nnz": int(x_valid.nnz if sparse.issparse(x_valid) else np.count_nonzero(x_valid)),
            },
            "valid": valid_result,
            "test_after_train_valid_refit": test_result,
        }

    gine_result: dict[str, Any] | None = None
    if bool(config["gine"].get("enabled", True)):
        gine_valid = _run_gine_phase(
            config,
            train_rows_source=datasets[0],
            eval_rows_source=datasets[1],
            train_y=labels[0],
            eval_y=labels[1],
            phase="official-train-to-valid",
        )
        combined_rows = list(datasets[0]) + list(datasets[1])
        combined_y = np.concatenate([labels[0], labels[1]], axis=0)
        gine_test = _run_gine_phase(
            config,
            train_rows_source=combined_rows,
            eval_rows_source=datasets[2],
            train_y=combined_y,
            eval_y=labels[2],
            phase="official-train-plus-valid-to-test",
        )
        gine_result = {"valid": gine_valid, "test_after_train_valid_refit": gine_test}

    selected_view = min(
        tuning_results,
        key=lambda name: (float(tuning_results[name]["best_cv_mae"]), name),
    )
    result = {
        "protocol_id": config["protocol_id"],
        "status": "completed",
        "data": {
            "root": str(data_root),
            "split": "PyG ZINC subset=True official train/val/test",
            "sizes": {"train": len(datasets[0]), "valid": len(datasets[1]), "test": len(datasets[2])},
            "official_sizes": {"train": len(loaded_datasets[0]), "valid": len(loaded_datasets[1]), "test": len(loaded_datasets[2])},
            "limits": {str(key): (None if value is None else int(value)) for key, value in limits.items()},
            "source": source_audit(data_root),
            "target": {name: _label_summary(value) for name, value in zip(("train", "valid", "test"), labels, strict=True)},
        },
        "representation": {
            "radius": radius,
            "typed_wl": True,
            "wl_rounds": [0, 1, 2, 3],
            "centres": "every atom",
            "vocabulary": "train-fold-only top-K exact nested tuple tokens; no proposed-feature hash bins",
            "top_k_per_round": top_k,
            "oov_per_round": True,
            "readout": "raw count + count / number of centres",
            "token_cache": str(token_cache_path),
            "token_cache_signature": token_signature,
            "token_cache_hit": cache_hit,
            "morgan_reference": {
                "kind": "RDKit-free count-ECFP/Morgan analogue",
                "bits": morgan_bits,
                "rounds": [0, 1, 2, 3],
                "atom_invariant": "ZINC atom category + degree + four bond-type degree counts",
                "update": "sorted typed neighbor environments",
            },
            "S_dimension": int(s_train.shape[1]),
            "WL_count_dimension": int(wl_train.shape[1]),
            "Morgan_count_dimension": int(morgan_train.shape[1]),
        },
        "protocol": {
            "xgboost_objective": str(config.get("xgboost_objective", "reg:absoluteerror")),
            "xgboost_eval_metric": "mae",
            "tuning_scope": "official-train inner shuffled KFold only",
            "official_valid_scope": "fit on official train with train-only selected parameters and vocabulary",
            "official_test_scope": "fit on official train+valid after freeze; vocabulary fit excludes test",
            "model_seeds": model_seeds,
            "folds": [{"train": int(len(train)), "valid": int(len(valid))} for train, valid in folds],
            "vocabulary_audit": vocab_meta,
        },
        "tuning": {"results": tuning_results, "selected_view_by_cv": selected_view},
        "evaluation": {"xgb": xgb_evaluation, "gine": gine_result},
        "runtime": {
            "seconds": float(time.perf_counter() - start),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": importlib.metadata.version("numpy"),
            "scipy": importlib.metadata.version("scipy"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "xgboost": importlib.metadata.version("xgboost"),
            "optuna": importlib.metadata.version("optuna"),
            "script_sha256": _sha256_path(Path(__file__).resolve()),
        },
    }
    _write_json_atomic(result_json, result)
    result_markdown.parent.mkdir(parents=True, exist_ok=True)
    result_markdown.write_text(_render_markdown(result), encoding="utf-8")
    return result


def _render_markdown(result: Mapping[str, Any]) -> str:
    representation = result["representation"]
    tuning = result["tuning"]["results"]
    evaluation = result["evaluation"]
    lines = [
        "# ZINC Step A：collision-free typed motif count",
        "",
        f"Protocol: `{result['protocol_id']}`",
        "",
        "## Representation",
        "",
        f"- radius: `{representation['radius']}`; centres: `{representation['centres']}`",
        "- typed rooted-WL rounds: `0/1/2/3`",
        f"- vocabulary: `{representation['vocabulary']}`; K=`{representation['top_k_per_round']}` + OOV",
        f"- readout: `{representation['readout']}`",
        f"- Morgan reference: `{representation['morgan_reference']['kind']}`, bits=`{representation['morgan_reference']['bits']}`",
        "",
        "## XGBoost results",
        "",
        "| view | dim | train CV MAE | valid MAE | valid ensemble | test MAE after train+valid | test ensemble |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in evaluation["xgb"].items():
        valid = row["valid"]
        test = row["test_after_train_valid_refit"]
        lines.append(
            f"| `{name}` | {row['dimension']} | {tuning[name]['best_cv_mae']:.6f} | "
            f"{valid['mean_mae']:.6f} ± {valid['std_mae']:.6f} | {valid['seed_ensemble_mae']:.6f} | "
            f"{test['mean_mae']:.6f} ± {test['std_mae']:.6f} | {test['seed_ensemble_mae']:.6f} |"
        )
    if evaluation.get("gine") is not None:
        valid = evaluation["gine"]["valid"]
        test = evaluation["gine"]["test_after_train_valid_refit"]
        lines.extend(
            [
                "",
                "## Matched GINE",
                "",
                f"- valid MAE: `{valid['mean_mae']:.6f} ± {valid['std_mae']:.6f}`; ensemble `{valid['seed_ensemble_mae']:.6f}`",
                f"- test after train+valid refit MAE: `{test['mean_mae']:.6f} ± {test['std_mae']:.6f}`; ensemble `{test['seed_ensemble_mae']:.6f}`",
            ]
        )
    lines.extend(
        [
            "",
            f"Selected XGBoost view by train-only CV: `{result['tuning']['selected_view_by_cv']}`.",
            "",
            f"Runtime: `{result['runtime']['seconds']:.1f}s`; token cache hit: `{representation['token_cache_hit']}`.",
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
        "selected_view": result["tuning"]["selected_view_by_cv"],
        "xgb": {
            name: {
                "cv": row["best_cv_mae"],
                "valid": result["evaluation"]["xgb"][name]["valid"]["mean_mae"],
                "test": result["evaluation"]["xgb"][name]["test_after_train_valid_refit"]["mean_mae"],
            }
            for name, row in result["tuning"]["results"].items()
        },
        "gine": None
        if result["evaluation"].get("gine") is None
        else {
            "valid": result["evaluation"]["gine"]["valid"]["mean_mae"],
            "test": result["evaluation"]["gine"]["test_after_train_valid_refit"]["mean_mae"],
        },
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
