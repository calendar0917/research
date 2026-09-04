"""ZINC hierarchical backoff plus topology-conditioned attribute matching.

This is a gated follow-up to ``zinc_hierarchical_backoff``.  The existing
hierarchical typed-WL count representation is kept fixed, and a single
additional representation is tested:

``topology-only rooted WL -> conditional attribute prototype -> match/residual``

For every atom centre, topology-only rooted-WL colours are used as the
condition.  A train-only bank stores the mean centre attribute vector for the
most frequent topology colour at each WL level.  At encoding time the deepest
known level for that same centre is selected; the centre's attribute vector is
then compared with the corresponding prototype.  The graph readout contains
per-prototype normalized mass, agreement and residual channels plus compact
depth summaries.  It is concatenated with the hierarchical count features and
sent to one XGBoost regressor.

The first stage is deliberately a hard representation gate.  It uses three
shuffled folds inside official train and fixed parameters selected by the
earlier hierarchical experiment.  True centre-to-attribute binding must beat
both the hierarchical baseline and an in-graph attribute-shuffle control.  If
the gate fails, no Optuna search and no official valid/test model evaluation is
performed.  Only a passing gate unlocks one 20-trial candidate search and the
frozen official valid/test report.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
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

from tracks.ksvd.experiments.luyin16.zinc_hierarchical_backoff import (
    N_ROUNDS,
    _encode_hierarchical,
    _fit_vocabularies,
    _params_from_trial,
)
from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import (
    REPO_ROOT,
    _data_to_graph,
    _load_zinc,
    _resolve,
    global_feature_views,
    source_audit,
)
from tracks.ksvd.experiments.luyin16.zinc_motif_count import _cache_to_rows


DEFAULT_CONFIG = REPO_ROOT / "tracks/ksvd/configs/luyin16/zinc_hierarchical_conditioned_match.yaml"
ATTRIBUTE_WIDTH = 28 + 4 + 28 + 4
TOPOLOGY_SCHEMA = "zinc-topology-only-rooted-wl-v1"
MATCH_CHANNELS = ("mass", "agreement", "residual", "centre_agreement")


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


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _label_summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "median": float(np.median(array)),
    }


def _stable_digest(namespace: str, value: object) -> bytes:
    payload = repr((str(namespace), value)).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).digest()


def _ego_distances(graph: Any, centre: int, radius: int) -> dict[int, int]:
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


def _normalised_histogram(values: Sequence[int], width: int) -> np.ndarray:
    output = np.bincount(np.asarray(list(values), dtype=np.int64), minlength=int(width)).astype(
        np.float32
    )
    return output / max(float(output.sum()), 1.0)


def _topology_and_attributes(
    data: Any,
    *,
    radius: int,
    rounds: int,
) -> tuple[tuple[tuple[object, ...], ...], np.ndarray, dict[str, Any]]:
    """Build aligned topology-only WL rows and centre attribute rows.

    Atom and bond categories are intentionally absent from the topology
    colours.  They enter only through the aligned 64-dimensional attribute
    rows, so a topology--attribute binding can be independently shuffled.
    """
    graph, node_types, edge_types = _data_to_graph(data)
    if node_types.size and (node_types.min() < 0 or node_types.max() >= 28):
        raise ValueError(
            f"ZINC atom category outside [0, 27]: {node_types.min()}..{node_types.max()}"
        )
    if edge_types and (min(edge_types.values()) < 0 or max(edge_types.values()) >= 4):
        raise ValueError(
            f"ZINC bond category outside [0, 3]: {min(edge_types.values())}..{max(edge_types.values())}"
        )

    topology_rows: list[list[bytes]] = [[] for _ in range(int(rounds) + 1)]
    attribute_rows: list[np.ndarray] = []
    for centre in graph.nodes:
        centre = int(centre)
        distances = _ego_distances(graph, centre, int(radius))
        nodes = tuple(sorted(distances))
        induced = graph.induced(set(nodes))
        colours: dict[int, bytes] = {
            node: _stable_digest(
                f"{TOPOLOGY_SCHEMA}-initial",
                (
                    int(node == centre),
                    int(distances[node]),
                    int(len(induced.neighbors(node))),
                ),
            )
            for node in nodes
        }
        for level in range(int(rounds) + 1):
            topology_rows[level].append(colours[centre])
            if level == int(rounds):
                break
            next_colours: dict[int, bytes] = {}
            for node in nodes:
                messages = tuple(sorted(colours[neighbour] for neighbour in induced.neighbors(node)))
                next_colours[node] = _stable_digest(
                    f"{TOPOLOGY_SCHEMA}-refine-{level}",
                    (colours[node], messages),
                )
            colours = next_colours

        incident_bonds = [
            int(edge_types[graph.edge_key(centre, neighbour)])
            for neighbour in graph.neighbors(centre)
        ]
        local_edges = tuple(induced.edges())
        local_bonds = [
            int(edge_types[graph.edge_key(left, right)])
            for left, right in local_edges
        ]
        centre_one_hot = np.zeros(28, dtype=np.float32)
        centre_one_hot[int(node_types[centre])] = 1.0
        attribute_rows.append(
            np.concatenate(
                [
                    centre_one_hot,
                    _normalised_histogram(incident_bonds, 4),
                    _normalised_histogram([int(node_types[node]) for node in nodes], 28),
                    _normalised_histogram(local_bonds, 4),
                ]
            ).astype(np.float32, copy=False)
        )

    topology = tuple(tuple(row) for row in topology_rows)
    attributes = np.stack(attribute_rows).astype(np.float32, copy=False)
    if len(topology) != N_ROUNDS or any(len(row) != attributes.shape[0] for row in topology):
        raise RuntimeError("topology rows and attribute rows are not centre-aligned")
    if attributes.shape[1] != ATTRIBUTE_WIDTH:
        raise RuntimeError(f"attribute width changed: {attributes.shape}")
    return topology, attributes, {
        "n_nodes": int(graph.n),
        "n_edges": int(graph.num_edges()),
        "n_centres": int(attributes.shape[0]),
        "radius": int(radius),
        "wl_rounds": int(rounds),
    }


def _object_array(rows: Sequence[Any]) -> np.ndarray:
    output = np.empty(len(rows), dtype=object)
    for index, row in enumerate(rows):
        output[index] = row
    return output


def _feature_signature(config: Mapping[str, Any]) -> str:
    representation = config["representation"]
    payload = {
        "schema": TOPOLOGY_SCHEMA,
        "radius": int(representation["radius"]),
        "wl_rounds": int(representation["wl_rounds"]),
        "script_sha256": _sha256_path(Path(__file__).resolve()),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _feature_cache_path(config: Mapping[str, Any], split: str) -> Path:
    cache_dir = _resolve(config["output"]["feature_cache_dir"])
    return cache_dir / f"{split}_topology_attributes.npz"


def _load_or_build_split_features(
    dataset: Any,
    split: str,
    config: Mapping[str, Any],
) -> tuple[list[Any], list[np.ndarray], dict[str, Any], bool]:
    path = _feature_cache_path(config, split)
    signature = _feature_signature(config)
    if path.exists():
        with np.load(path, allow_pickle=True) as archive:
            cached_signature = str(np.asarray(archive["signature"]).reshape(-1)[0])
            if cached_signature == signature:
                topology_array = np.asarray(archive["topology_rows"], dtype=object).reshape(-1)
                attribute_array = np.asarray(archive["attribute_rows"], dtype=object).reshape(-1)
                metadata = json.loads(str(np.asarray(archive["metadata"]).reshape(-1)[0]))
                topology_rows = topology_array.tolist()
                attribute_rows = [np.asarray(value, dtype=np.float32) for value in attribute_array.tolist()]
                if len(topology_rows) == len(dataset) and len(attribute_rows) == len(dataset):
                    return topology_rows, attribute_rows, metadata, True

    representation = config["representation"]
    topology_rows: list[Any] = []
    attribute_rows: list[np.ndarray] = []
    graph_metadata: list[dict[str, Any]] = []
    progress_every = int(config.get("progress_every", 500))
    for index, data in enumerate(dataset):
        topology, attributes, details = _topology_and_attributes(
            data,
            radius=int(representation["radius"]),
            rounds=int(representation["wl_rounds"]),
        )
        topology_rows.append(topology)
        attribute_rows.append(attributes)
        graph_metadata.append(details)
        if progress_every > 0 and (index + 1) % progress_every == 0:
            print(f"[{split}] topology-conditioned rows: {index + 1}/{len(dataset)}", flush=True)

    metadata = {
        "split": str(split),
        "n_graphs": int(len(dataset)),
        "mean_centres": float(np.mean([row["n_centres"] for row in graph_metadata]))
        if graph_metadata
        else 0.0,
        "mean_nodes": float(np.mean([row["n_nodes"] for row in graph_metadata]))
        if graph_metadata
        else 0.0,
        "mean_edges": float(np.mean([row["n_edges"] for row in graph_metadata]))
        if graph_metadata
        else 0.0,
        "schema": TOPOLOGY_SCHEMA,
        "signature": signature,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            signature=np.asarray([signature]),
            topology_rows=_object_array(topology_rows),
            attribute_rows=_object_array(attribute_rows),
            metadata=np.asarray([json.dumps(metadata, sort_keys=True)]),
        )
    temporary.replace(path)
    return topology_rows, attribute_rows, metadata, False


def _fit_condition_bank(
    topology_rows: Sequence[tuple[tuple[object, ...], ...]],
    attribute_rows: Sequence[np.ndarray],
    indices: Sequence[int],
    *,
    top_k: int,
) -> dict[str, Any]:
    """Fit topology vocabularies and train-only conditional attribute means."""
    vocabularies: list[dict[object, int]] = []
    prototypes: list[np.ndarray] = []
    selected_counts: list[np.ndarray] = []
    level_metadata: list[dict[str, Any]] = []
    for level in range(N_ROUNDS):
        counts: Counter[object] = Counter()
        for graph_index in indices:
            counts.update(topology_rows[int(graph_index)][level])
        ranked = sorted(counts.items(), key=lambda item: (-int(item[1]), repr(item[0])))[: int(top_k)]
        vocabulary = {token: column for column, (token, _count) in enumerate(ranked)}
        sums = np.zeros((len(ranked), ATTRIBUTE_WIDTH), dtype=np.float64)
        selected = np.asarray([int(count) for _token, count in ranked], dtype=np.int64)
        for graph_index in indices:
            tokens = topology_rows[int(graph_index)][level]
            attributes = np.asarray(attribute_rows[int(graph_index)], dtype=np.float32)
            for token, attributes_row in zip(tokens, attributes, strict=True):
                column = vocabulary.get(token)
                if column is not None:
                    sums[int(column)] += attributes_row
        means = (sums / np.maximum(selected[:, None], 1)).astype(np.float32, copy=False)
        vocabularies.append(vocabulary)
        prototypes.append(means)
        selected_counts.append(selected)
        level_metadata.append(
            {
                "level": int(level),
                "unique_train_tokens": int(len(counts)),
                "selected_prototypes": int(len(ranked)),
                "selected_occurrence_fraction": float(
                    selected.sum() / max(float(sum(counts.values())), 1.0)
                ),
                "fit_occurrences": int(sum(counts.values())),
            }
        )
    return {
        "vocabularies": tuple(vocabularies),
        "prototypes": tuple(prototypes),
        "selected_counts": tuple(selected_counts),
        "metadata": {
            "fit_graphs": int(len(indices)),
            "fit_centres": int(sum(len(topology_rows[int(index)][0]) for index in indices)),
            "top_k": int(top_k),
            "levels": level_metadata,
        },
    }


def _shuffle_permutation(n_rows: int, seed: int, key: int) -> np.ndarray:
    if int(n_rows) <= 1:
        return np.arange(int(n_rows), dtype=np.int64)
    digest = hashlib.blake2b(digest_size=8)
    digest.update(b"luyin16-zinc-conditioned-attribute-shuffle-v1")
    digest.update(np.asarray([int(seed), int(key)], dtype=np.int64).tobytes())
    rng = np.random.default_rng(int.from_bytes(digest.digest(), "little"))
    permutation = np.asarray(rng.permutation(int(n_rows)), dtype=np.int64)
    if np.array_equal(permutation, np.arange(int(n_rows), dtype=np.int64)):
        permutation = np.roll(permutation, 1)
    return permutation


def _attribute_match(attribute: np.ndarray, prototype: np.ndarray) -> tuple[float, float, float]:
    blocks = (
        (slice(0, 28), 1.0),
        (slice(28, 32), 1.0),
        (slice(32, 60), 1.0),
        (slice(60, 64), 1.0),
    )
    dots = [
        float(np.dot(attribute[block], prototype[block]))
        for block, _weight in blocks
    ]
    agreement = float(np.mean(dots))
    residual = float(np.mean(np.abs(attribute - prototype)))
    centre_agreement = float(np.dot(attribute[:28], prototype[:28]))
    return agreement, residual, centre_agreement


def _summary(values: Sequence[float]) -> list[float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return [0.0] * 6
    return [
        float(array.mean()),
        float(array.std()),
        float(np.quantile(array, 0.25)),
        float(np.quantile(array, 0.50)),
        float(np.quantile(array, 0.75)),
    ] + [float(array.max())]


def _encode_conditioned_match(
    topology_rows: Sequence[tuple[tuple[object, ...], ...]],
    attribute_rows: Sequence[np.ndarray],
    bank: Mapping[str, Any],
    *,
    top_k: int,
    keys: Sequence[int],
    shuffle: bool,
    shuffle_seed: int,
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    """Encode deepest-known same-centre topology--attribute matches."""
    if len(topology_rows) != len(attribute_rows) or len(keys) != len(topology_rows):
        raise ValueError("topology, attribute and shuffle-key rows are not aligned")
    vocabularies = bank["vocabularies"]
    prototypes = bank["prototypes"]
    total_prototype_width = N_ROUNDS * int(top_k) * len(MATCH_CHANNELS)
    summary_width = (N_ROUNDS + 1) + N_ROUNDS * 12 + 12
    total_width = total_prototype_width + summary_width
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    depth_totals = np.zeros(N_ROUNDS + 1, dtype=np.float64)
    assigned_agreement: list[list[float]] = [[] for _ in range(N_ROUNDS)]
    assigned_residual: list[list[float]] = [[] for _ in range(N_ROUNDS)]
    centre_total = 0.0

    for graph_index, (graph_topology, graph_attributes, key) in enumerate(
        zip(topology_rows, attribute_rows, keys, strict=True)
    ):
        attributes = np.asarray(graph_attributes, dtype=np.float32)
        if shuffle:
            permutation = _shuffle_permutation(attributes.shape[0], int(shuffle_seed), int(key))
            attributes = attributes[permutation]
        n_centres = int(attributes.shape[0])
        if len(graph_topology) != N_ROUNDS or any(
            len(graph_topology[level]) != n_centres for level in range(N_ROUNDS)
        ):
            raise ValueError(f"graph {graph_index} topology rows are not aligned")
        row_counts: dict[int, float] = {}
        depth_counts = np.zeros(N_ROUNDS + 1, dtype=np.float64)
        per_level_agreement: list[list[float]] = [[] for _ in range(N_ROUNDS)]
        per_level_residual: list[list[float]] = [[] for _ in range(N_ROUNDS)]
        for centre in range(n_centres):
            matched_level = -1
            matched_column = -1
            for level in range(N_ROUNDS - 1, -1, -1):
                column = vocabularies[level].get(graph_topology[level][centre])
                if column is not None:
                    matched_level = int(level)
                    matched_column = int(column)
                    break
            bucket = matched_level + 1
            depth_counts[bucket] += 1.0
            if matched_level < 0:
                continue
            agreement, residual, centre_agreement = _attribute_match(
                attributes[centre], prototypes[matched_level][matched_column]
            )
            per_level_agreement[matched_level].append(agreement)
            per_level_residual[matched_level].append(residual)
            block = (
                matched_level * int(top_k) + matched_column
            ) * len(MATCH_CHANNELS)
            mass = 1.0 / max(float(n_centres), 1.0)
            channel_values = (mass, mass * agreement, mass * residual, mass * centre_agreement)
            for channel, value in enumerate(channel_values):
                column = block + int(channel)
                row_counts[column] = row_counts.get(column, 0.0) + float(value)

        depth_totals += depth_counts
        centre_total += float(n_centres)
        for level in range(N_ROUNDS):
            assigned_agreement[level].extend(per_level_agreement[level])
            assigned_residual[level].extend(per_level_residual[level])
        offset = total_prototype_width
        row_counts.update(
            {
                offset + bucket: float(depth_counts[bucket] / max(float(n_centres), 1.0))
                for bucket in range(N_ROUNDS + 1)
            }
        )
        offset += N_ROUNDS + 1
        for level in range(N_ROUNDS):
            agreement_stats = _summary(per_level_agreement[level])
            residual_stats = _summary(per_level_residual[level])
            for value in agreement_stats + residual_stats:
                row_counts[offset] = float(value)
                offset += 1
        all_agreement = [value for values in per_level_agreement for value in values]
        all_residual = [value for values in per_level_residual for value in values]
        for value in _summary(all_agreement) + _summary(all_residual):
            row_counts[offset] = float(value)
            offset += 1
        if offset != total_width:
            raise RuntimeError(f"conditioned-match summary width changed: {offset} != {total_width}")
        for column, value in row_counts.items():
            if value == 0.0:
                continue
            row_indices.append(int(graph_index))
            column_indices.append(int(column))
            values.append(float(value))

    matrix = sparse.coo_matrix(
        (
            np.asarray(values, dtype=np.float32),
            (
                np.asarray(row_indices, dtype=np.int64),
                np.asarray(column_indices, dtype=np.int64),
            ),
        ),
        shape=(len(topology_rows), total_width),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    return matrix, {
        "dimension": int(total_width),
        "prototype_dimension": int(total_prototype_width),
        "summary_dimension": int(summary_width),
        "shuffle": bool(shuffle),
        "shuffle_seed": int(shuffle_seed),
        "deepest_match_fraction": [
            float(depth_totals[bucket] / max(centre_total, 1.0))
            for bucket in range(N_ROUNDS + 1)
        ],
        "assigned_level_agreement": [_summary(values) for values in assigned_agreement],
        "assigned_level_residual": [_summary(values) for values in assigned_residual],
        "centres": int(centre_total),
    }


def _hstack(global_values: np.ndarray, hierarchical: sparse.spmatrix, match: sparse.spmatrix) -> sparse.csr_matrix:
    return sparse.hstack(
        [
            sparse.csr_matrix(np.asarray(global_values, dtype=np.float32)),
            hierarchical,
            match,
        ],
        format="csr",
        dtype=np.float32,
    )


def _xgb_params(config: Mapping[str, Any], params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    xgb_config = config["xgboost"]
    output = {
        "objective": str(xgb_config.get("objective", "reg:absoluteerror")),
        "eval_metric": "mae",
        "tree_method": "hist",
        "max_bin": int(xgb_config.get("max_bin", 256)),
        "random_state": int(xgb_config.get("model_seed", 0)),
        "n_jobs": int(xgb_config.get("n_jobs", 4)),
    }
    output.update(dict(xgb_config.get("params", {})))
    if params is not None:
        output.update(dict(params))
    return output


def _fit_mae(
    matrix: sparse.spmatrix,
    labels: np.ndarray,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    config: Mapping[str, Any],
    params: Mapping[str, Any] | None = None,
) -> float:
    model = XGBRegressor(**_xgb_params(config, params))
    model.fit(matrix[train_indices], labels[train_indices])
    prediction = model.predict(matrix[valid_indices])
    return float(mean_absolute_error(labels[valid_indices], prediction))


def _evaluate(
    train_matrix: sparse.spmatrix,
    train_labels: np.ndarray,
    eval_matrix: sparse.spmatrix,
    eval_labels: np.ndarray,
    config: Mapping[str, Any],
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    model = XGBRegressor(**_xgb_params(config, params))
    model.fit(train_matrix, train_labels)
    prediction = np.asarray(model.predict(eval_matrix), dtype=np.float64)
    return {
        "seed": int(config["xgboost"].get("model_seed", 0)),
        "mae": float(mean_absolute_error(eval_labels, prediction)),
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std()),
    }


def _folds(n_rows: int, config: Mapping[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    gate = config["gate"]
    splitter = KFold(
        n_splits=int(gate["n_splits"]),
        shuffle=True,
        random_state=int(gate["split_seed"]),
    )
    return [
        (train.astype(np.int64), valid.astype(np.int64))
        for train, valid in splitter.split(np.arange(int(n_rows), dtype=np.int64))
    ]


def _run_gate(
    typed_rows: Sequence[tuple[tuple[object, ...], ...]],
    topology_rows: Sequence[tuple[tuple[object, ...], ...]],
    attribute_rows: Sequence[np.ndarray],
    global_train: np.ndarray,
    labels: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    top_k = int(config["representation"]["top_k"])
    folds = _folds(len(labels), config)
    fold_results: list[dict[str, Any]] = []
    candidate_matrices: list[sparse.csr_matrix] = []
    baseline_matrices: list[sparse.csr_matrix] = []
    model_params = dict(config["xgboost"]["params"])
    shuffle_seed = int(config["gate"]["shuffle_seed"])

    for fold_id, (fold_train, fold_valid) in enumerate(folds):
        hierarchical_vocabularies = _fit_vocabularies(
            typed_rows,
            fold_train,
            top_k=top_k,
        )
        hierarchical, hierarchical_summary = _encode_hierarchical(
            typed_rows,
            hierarchical_vocabularies,
            top_k=top_k,
        )
        bank = _fit_condition_bank(
            topology_rows,
            attribute_rows,
            fold_train,
            top_k=top_k,
        )
        true_match, true_summary = _encode_conditioned_match(
            topology_rows,
            attribute_rows,
            bank,
            top_k=top_k,
            keys=np.arange(len(labels), dtype=np.int64),
            shuffle=False,
            shuffle_seed=shuffle_seed,
        )
        shuffled_match, shuffled_summary = _encode_conditioned_match(
            topology_rows,
            attribute_rows,
            bank,
            top_k=top_k,
            keys=np.arange(len(labels), dtype=np.int64),
            shuffle=True,
            shuffle_seed=shuffle_seed,
        )
        baseline = _hstack(global_train, hierarchical, sparse.csr_matrix((len(labels), 0)))
        candidate = _hstack(global_train, hierarchical, true_match)
        shuffled = _hstack(global_train, hierarchical, shuffled_match)
        baseline_mae = _fit_mae(baseline, labels, fold_train, fold_valid, config, model_params)
        true_mae = _fit_mae(candidate, labels, fold_train, fold_valid, config, model_params)
        shuffled_mae = _fit_mae(shuffled, labels, fold_train, fold_valid, config, model_params)
        fold_results.append(
            {
                "fold": int(fold_id),
                "train_graphs": int(fold_train.size),
                "valid_graphs": int(fold_valid.size),
                "baseline_mae": float(baseline_mae),
                "true_match_mae": float(true_mae),
                "shuffle_mae": float(shuffled_mae),
                "baseline_minus_true": float(baseline_mae - true_mae),
                "shuffle_minus_true": float(shuffled_mae - true_mae),
                "true_wins": bool(true_mae < baseline_mae),
                "hierarchical_summary": hierarchical_summary,
                "bank": bank["metadata"],
                "true_match_summary": true_summary,
                "shuffle_match_summary": shuffled_summary,
            }
        )
        baseline_matrices.append(baseline)
        candidate_matrices.append(candidate)
        print(
            f"gate fold {fold_id}: hierarchical={baseline_mae:.6f} "
            f"true={true_mae:.6f} shuffle={shuffled_mae:.6f} "
            f"delta={baseline_mae - true_mae:+.6f}",
            flush=True,
        )

    mean_baseline = float(np.mean([row["baseline_mae"] for row in fold_results]))
    mean_true = float(np.mean([row["true_match_mae"] for row in fold_results]))
    mean_shuffle = float(np.mean([row["shuffle_mae"] for row in fold_results]))
    mean_reduction = float(mean_baseline - mean_true)
    mean_shuffle_gap = float(mean_shuffle - mean_true)
    wins = int(sum(bool(row["true_wins"]) for row in fold_results))
    conditions = {
        "mean_reduction_min": float(config["gate"]["min_mean_reduction"]),
        "fold_wins_min": int(config["gate"]["min_fold_wins"]),
        "true_vs_shuffle_min": float(config["gate"]["min_true_vs_shuffle"]),
        "target_mean_reduction": float(config["gate"].get("target_mean_reduction", 0.015)),
    }
    checks = {
        "mean_reduction": bool(mean_reduction >= conditions["mean_reduction_min"]),
        "fold_wins": bool(wins >= conditions["fold_wins_min"]),
        "true_vs_shuffle": bool(mean_shuffle_gap >= conditions["true_vs_shuffle_min"]),
    }
    passed = bool(all(checks.values()))
    gate_result = {
        "status": "passed" if passed else "failed",
        "protocol": {
            "scope": "official-train only",
            "n_splits": int(config["gate"]["n_splits"]),
            "split_seed": int(config["gate"]["split_seed"]),
            "shuffle_seed": shuffle_seed,
            "model_seed": int(config["xgboost"].get("model_seed", 0)),
            "fixed_xgboost_params": model_params,
        },
        "conditions": conditions,
        "checks": checks,
        "fold_wins": wins,
        "mean_baseline_mae": mean_baseline,
        "mean_true_match_mae": mean_true,
        "mean_shuffle_mae": mean_shuffle,
        "mean_reduction": mean_reduction,
        "mean_true_vs_shuffle": mean_shuffle_gap,
        "folds": fold_results,
    }
    state = None
    if passed:
        state = {
            "folds": folds,
            "candidate_matrices": candidate_matrices,
            "baseline_matrices": baseline_matrices,
        }
    return gate_result, state


def _tune_candidate(
    candidate_matrices: Sequence[sparse.csr_matrix],
    labels: np.ndarray,
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

    def objective(trial: optuna.Trial) -> float:
        params = _params_from_trial(trial, ranges)
        scores = [
            _fit_mae(
                candidate_matrices[fold_id],
                labels,
                fold_train,
                fold_valid,
                config,
                params,
            )
            for fold_id, (fold_train, fold_valid) in enumerate(folds)
        ]
        return float(np.mean(scores))

    study.optimize(
        objective,
        n_trials=int(tuning["n_trials"]),
        show_progress_bar=False,
    )
    best_params = dict(study.best_trial.params)
    fold_scores = [
        _fit_mae(
            candidate_matrices[fold_id],
            labels,
            fold_train,
            fold_valid,
            config,
            best_params,
        )
        for fold_id, (fold_train, fold_valid) in enumerate(folds)
    ]
    return {
        "view": "s_hierarchical_topology_conditioned_match",
        "objective": str(config["xgboost"].get("objective", "reg:absoluteerror")),
        "eval_metric": "mae",
        "selection_scope": "official-train inner shuffled KFold only; fold-local vocabularies and attribute prototypes",
        "n_trials": int(tuning["n_trials"]),
        "n_folds": int(len(folds)),
        "seed": int(tuning["seed"]),
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


def _render_markdown(result: Mapping[str, Any]) -> str:
    gate = result["gate"]
    lines = [
        "# ZINC hierarchical topology-conditioned match gate",
        "",
        "Topology-only rooted-WL conditions a train-only centre-attribute prototype bank; the resulting same-centre agreement/residual readout is concatenated with hierarchical typed-WL backoff counts and sent to one XGBoost.",
        "",
        "## Gate protocol",
        "",
        "- scope: `official-train` only; three shuffled folds",
        f"- fixed objective: `{result['protocol']['xgboost_objective']}`; model seed: `{result['protocol']['model_seed']}`",
        f"- topology/attribute shuffle: in-graph row permutation, seed `{gate['protocol']['shuffle_seed']}`; prototype bank is not refit",
        "- each fold fits the hierarchical vocabulary and conditional prototype bank on fold-train only",
        "",
        "| fold | hierarchical MAE | true match MAE | shuffle MAE | baseline − true | shuffle − true | win |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in gate["folds"]:
        lines.append(
            f"| {row['fold']} | {row['baseline_mae']:.6f} | {row['true_match_mae']:.6f} | "
            f"{row['shuffle_mae']:.6f} | {row['baseline_minus_true']:+.6f} | "
            f"{row['shuffle_minus_true']:+.6f} | {'yes' if row['true_wins'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"- mean hierarchical MAE: `{gate['mean_baseline_mae']:.6f}`",
            f"- mean true-match MAE: `{gate['mean_true_match_mae']:.6f}`",
            f"- mean shuffle MAE: `{gate['mean_shuffle_mae']:.6f}`",
            f"- mean reduction: `{gate['mean_reduction']:+.6f}`; fold wins: `{gate['fold_wins']}/{len(gate['folds'])}`; true-vs-shuffle gap: `{gate['mean_true_vs_shuffle']:+.6f}`",
            f"- gate: **{gate['status']}**",
            "",
        ]
    )
    if gate["status"] != "passed":
        lines.extend(
            [
                "No Optuna search was run and official valid/test were not evaluated because the representation gate did not pass.",
                "",
            ]
        )
    else:
        tuning = result["tuning"]
        evaluation = result["evaluation"]
        lines.extend(
            [
                "## Unlocked main-model search",
                "",
                f"- Optuna: `{tuning['n_trials']}` trials, `{tuning['n_folds']}` folds; best CV MAE `{tuning['best_cv_mae']:.6f}`",
                "",
                "| view | dimension | official valid MAE | official test MAE after train+valid refit |",
                "|---|---:|---:|---:|",
                f"| `S + hierarchical + topology-conditioned match` | {evaluation['candidate']['dimension']} | {evaluation['candidate']['valid']['mae']:.6f} | {evaluation['candidate']['test_after_train_valid_refit']['mae']:.6f} |",
                f"| `S + hierarchical` fixed reference | {evaluation['hierarchical']['dimension']} | {evaluation['hierarchical']['valid']['mae']:.6f} | {evaluation['hierarchical']['test_after_train_valid_refit']['mae']:.6f} |",
                "",
                "### Selected parameters",
                "",
                "```json",
                json.dumps(tuning["best_params"], ensure_ascii=False, indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Leakage boundary",
            "",
            f"- official valid vocabulary/prototype scope: `{result['protocol']['valid_scope']}`",
            f"- official test vocabulary/prototype scope: `{result['protocol']['test_scope']}`",
            f"- test labels used for selection: `{result['protocol']['test_labels_used_for_selection']}`",
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
    representation = config["representation"]
    top_k = int(representation["top_k"])
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if int(representation["wl_rounds"]) + 1 != N_ROUNDS:
        raise ValueError(f"this protocol expects {N_ROUNDS - 1} WL refinement rounds")

    token_cache_path = _resolve(representation["token_cache"])
    if not token_cache_path.exists():
        raise FileNotFoundError(f"typed-WL token cache is missing: {token_cache_path}")
    with np.load(token_cache_path, allow_pickle=True) as archive:
        all_typed_rows, _unused_morgan, token_cache_meta = _cache_to_rows(archive)

    # Gate stage intentionally loads and encodes official train only.  The
    # official validation/test datasets are not loaded until the gate passes.
    train_dataset = _load_zinc(data_root, "train")
    train_size = len(train_dataset)
    if len(all_typed_rows) < train_size:
        raise RuntimeError("typed token cache has fewer rows than official train")
    typed_train = all_typed_rows[:train_size]
    labels_train = np.asarray(
        [float(data.y.view(-1)[0]) for data in train_dataset], dtype=np.float32
    )
    global_train = np.asarray(global_feature_views(train_dataset)["global_all"], dtype=np.float32)
    topology_train, attributes_train, train_feature_meta, train_cache_hit = _load_or_build_split_features(
        train_dataset, "train", config
    )
    if len(topology_train) != train_size or global_train.shape[0] != train_size:
        raise RuntimeError("official train feature rows are not aligned")

    gate, gate_state = _run_gate(
        typed_train,
        topology_train,
        attributes_train,
        global_train,
        labels_train,
        config,
    )
    output = config["output"]
    result_json = _resolve(output["json"])
    result_markdown = _resolve(output["markdown"])
    result: dict[str, Any] = {
        "protocol_id": str(config["protocol_id"]),
        "status": "gate_passed" if gate["status"] == "passed" else "gate_failed",
        "data": {
            "root": str(data_root),
            "split": "PyG ZINC subset=True official train/val/test; gate computed on official train only",
            "sizes": {"train": int(train_size)},
            "source": source_audit(data_root),
            "target": {"train": _label_summary(labels_train)},
        },
        "representation": {
            "hierarchical_base": "typed-WL radius-3 exact count plus nearest-known same-centre lower-order backoff",
            "condition": "topology-only rooted-WL levels 0..3",
            "attribute": "centre atom one-hot + incident bond histogram + local atom/bond histograms",
            "conditional_prototype": "train-only mean attribute vector for top-K topology tokens at each level",
            "match": "deepest-known same-centre topology condition; normalized mass, agreement, residual and centre-agreement per prototype",
            "top_k": top_k,
            "radius": int(representation["radius"]),
            "wl_rounds": int(representation["wl_rounds"]),
            "attribute_width": ATTRIBUTE_WIDTH,
            "match_channels": list(MATCH_CHANNELS),
            "centres": "every atom",
            "one_xgboost": True,
            "ksvd": False,
            "feature_cache": {
                "path": str(_feature_cache_path(config, "train")),
                "cache_hit": bool(train_cache_hit),
                "metadata": train_feature_meta,
            },
            "token_cache": str(token_cache_path),
            "token_cache_metadata": token_cache_meta,
        },
        "protocol": {
            "xgboost_objective": str(config["xgboost"].get("objective", "reg:absoluteerror")),
            "xgboost_eval_metric": "mae",
            "model_seed": int(config["xgboost"].get("model_seed", 0)),
            "valid_scope": "not evaluated unless gate passes; then official train-only vocabulary/prototypes and labels",
            "test_scope": "not evaluated unless gate passes; then official train+valid-only vocabulary/prototypes and labels",
            "test_labels_used_for_selection": False,
            "gate_scope": "official train only; fold-local hierarchical vocabularies and topology-conditioned attribute prototypes",
        },
        "gate": gate,
        "tuning": None,
        "evaluation": None,
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
            "token_cache_sha256": _sha256_path(token_cache_path),
        },
    }

    if gate_state is not None:
        assert gate["status"] == "passed"
        tuning = _tune_candidate(
            gate_state["candidate_matrices"],
            labels_train,
            gate_state["folds"],
            config,
        )
        print(
            f"gate passed; candidate Optuna best CV MAE={tuning['best_cv_mae']:.6f} "
            f"params={tuning['best_params']}",
            flush=True,
        )

        valid_dataset = _load_zinc(data_root, "val")
        test_dataset = _load_zinc(data_root, "test")
        valid_size, test_size = len(valid_dataset), len(test_dataset)
        expected_total = train_size + valid_size + test_size
        if len(all_typed_rows) != expected_total:
            raise RuntimeError(
                f"typed token cache rows={len(all_typed_rows)} but loaded data rows={expected_total}"
            )
        labels_valid = np.asarray(
            [float(data.y.view(-1)[0]) for data in valid_dataset], dtype=np.float32
        )
        labels_test = np.asarray(
            [float(data.y.view(-1)[0]) for data in test_dataset], dtype=np.float32
        )
        global_valid = np.asarray(global_feature_views(valid_dataset)["global_all"], dtype=np.float32)
        global_test = np.asarray(global_feature_views(test_dataset)["global_all"], dtype=np.float32)
        topology_valid, attributes_valid, valid_feature_meta, valid_cache_hit = _load_or_build_split_features(
            valid_dataset, "valid", config
        )
        topology_test, attributes_test, test_feature_meta, test_cache_hit = _load_or_build_split_features(
            test_dataset, "test", config
        )
        typed_valid = all_typed_rows[train_size : train_size + valid_size]
        typed_test = all_typed_rows[train_size + valid_size :]

        train_indices = np.arange(train_size, dtype=np.int64)
        valid_indices = np.arange(valid_size, dtype=np.int64)
        train_vocab = _fit_vocabularies(typed_train, train_indices, top_k=top_k)
        train_hierarchical, _train_hier_summary = _encode_hierarchical(
            typed_train, train_vocab, top_k=top_k
        )
        valid_hierarchical, _valid_hier_summary = _encode_hierarchical(
            typed_valid, train_vocab, top_k=top_k
        )
        train_bank = _fit_condition_bank(
            topology_train,
            attributes_train,
            train_indices,
            top_k=top_k,
        )
        train_match, train_match_summary = _encode_conditioned_match(
            topology_train,
            attributes_train,
            train_bank,
            top_k=top_k,
            keys=np.arange(train_size, dtype=np.int64),
            shuffle=False,
            shuffle_seed=int(config["gate"]["shuffle_seed"]),
        )
        valid_match, valid_match_summary = _encode_conditioned_match(
            topology_valid,
            attributes_valid,
            train_bank,
            top_k=top_k,
            keys=np.arange(train_size, train_size + valid_size, dtype=np.int64),
            shuffle=False,
            shuffle_seed=int(config["gate"]["shuffle_seed"]),
        )
        train_candidate = _hstack(global_train, train_hierarchical, train_match)
        valid_candidate = _hstack(global_valid, valid_hierarchical, valid_match)
        train_baseline = _hstack(global_train, train_hierarchical, sparse.csr_matrix((train_size, 0)))
        valid_baseline = _hstack(global_valid, valid_hierarchical, sparse.csr_matrix((valid_size, 0)))
        selected_params = tuning["best_params"]
        candidate_valid = _evaluate(
            train_candidate, labels_train, valid_candidate, labels_valid, config, selected_params
        )
        hierarchical_valid = _evaluate(
            train_baseline, labels_train, valid_baseline, labels_valid, config, config["xgboost"]["params"]
        )

        combined_topology = topology_train + topology_valid
        combined_attributes = attributes_train + attributes_valid
        combined_typed = typed_train + typed_valid
        combined_global = np.concatenate([global_train, global_valid], axis=0)
        combined_labels = np.concatenate([labels_train, labels_valid], axis=0)
        combined_indices = np.arange(train_size + valid_size, dtype=np.int64)
        combined_vocab = _fit_vocabularies(
            combined_typed, combined_indices, top_k=top_k
        )
        combined_hierarchical, _combined_hier_summary = _encode_hierarchical(
            combined_typed, combined_vocab, top_k=top_k
        )
        test_hierarchical, _test_hier_summary = _encode_hierarchical(
            typed_test, combined_vocab, top_k=top_k
        )
        combined_bank = _fit_condition_bank(
            combined_topology,
            combined_attributes,
            combined_indices,
            top_k=top_k,
        )
        combined_match, combined_match_summary = _encode_conditioned_match(
            combined_topology,
            combined_attributes,
            combined_bank,
            top_k=top_k,
            keys=np.arange(train_size + valid_size, dtype=np.int64),
            shuffle=False,
            shuffle_seed=int(config["gate"]["shuffle_seed"]),
        )
        test_match, test_match_summary = _encode_conditioned_match(
            topology_test,
            attributes_test,
            combined_bank,
            top_k=top_k,
            keys=np.arange(train_size + valid_size, train_size + valid_size + test_size, dtype=np.int64),
            shuffle=False,
            shuffle_seed=int(config["gate"]["shuffle_seed"]),
        )
        combined_candidate = _hstack(combined_global, combined_hierarchical, combined_match)
        test_candidate = _hstack(global_test, test_hierarchical, test_match)
        combined_baseline = _hstack(
            combined_global,
            combined_hierarchical,
            sparse.csr_matrix((train_size + valid_size, 0)),
        )
        test_baseline = _hstack(
            global_test,
            test_hierarchical,
            sparse.csr_matrix((test_size, 0)),
        )
        candidate_test = _evaluate(
            combined_candidate, combined_labels, test_candidate, labels_test, config, selected_params
        )
        hierarchical_test = _evaluate(
            combined_baseline,
            combined_labels,
            test_baseline,
            labels_test,
            config,
            config["xgboost"]["params"],
        )
        result["status"] = "completed"
        result["data"]["sizes"] = {
            "train": int(train_size),
            "valid": int(valid_size),
            "test": int(test_size),
        }
        result["data"]["target"] = {
            "train": _label_summary(labels_train),
            "valid": _label_summary(labels_valid),
            "test": _label_summary(labels_test),
        }
        result["representation"]["feature_cache"]["splits"] = {
            "valid": {"path": str(_feature_cache_path(config, "valid")), "cache_hit": bool(valid_cache_hit), "metadata": valid_feature_meta},
            "test": {"path": str(_feature_cache_path(config, "test")), "cache_hit": bool(test_cache_hit), "metadata": test_feature_meta},
        }
        result["tuning"] = tuning
        result["evaluation"] = {
            "metric": "MAE (lower is better)",
            "candidate": {
                "dimension": int(train_candidate.shape[1]),
                "valid": candidate_valid,
                "test_after_train_valid_refit": candidate_test,
            },
            "hierarchical": {
                "dimension": int(train_baseline.shape[1]),
                "note": "fixed hierarchical parameters used for the gate; no new baseline search",
                "valid": hierarchical_valid,
                "test_after_train_valid_refit": hierarchical_test,
            },
        }
        result["coverage"] = {
            "official_train_bank": train_bank["metadata"],
            "official_train_match": train_match_summary,
            "official_valid_match": valid_match_summary,
            "combined_bank": combined_bank["metadata"],
            "combined_train_valid_match": combined_match_summary,
            "official_test_match": test_match_summary,
        }
    result["runtime"]["seconds"] = float(time.perf_counter() - start)
    _write_json_atomic(result_json, result)
    result_markdown.parent.mkdir(parents=True, exist_ok=True)
    result_markdown.write_text(_render_markdown(result), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    result = run(args.config)
    gate = result["gate"]
    print(
        f"gate={gate['status']} mean_reduction={gate['mean_reduction']:+.6f} "
        f"wins={gate['fold_wins']}/{len(gate['folds'])} "
        f"true_vs_shuffle={gate['mean_true_vs_shuffle']:+.6f}",
        flush=True,
    )
    if result["status"] == "completed":
        print(
            f"candidate: valid_mae={result['evaluation']['candidate']['valid']['mae']:.6f} "
            f"test_mae={result['evaluation']['candidate']['test_after_train_valid_refit']['mae']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
