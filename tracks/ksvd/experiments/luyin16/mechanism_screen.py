"""Fast mechanism screen for the luyin16 ZINC hypothesis.

This runner deliberately avoids a full K-SVD/XGBoost sweep.  It freezes one
set of atom-centred patches and compares, on a small official train/valid
subset, (1) marginal readouts, (2) block-wise joint/co-occurrence statistics,
(3) typed attribute-shuffle controls, and (4) a low-dimensional patch-relation
readout using centre distance, patch overlap and patch similarity.  It is a
screen, not a terminal ZINC result; no test split is loaded.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import (
    ATOM_BINS,
    BOND_BINS,
    REPO_ROOT,
    _data_to_graph,
    _ego_nodes,
    _patch_vector,
    _resolve,
    _load_zinc,
    global_feature_views,
    raw_readout,
    select_and_normalize,
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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _vectorize_with_relations(
    dataset,
    radius: int,
    max_nodes: int,
    max_patches: int | None,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, Any]]]:
    matrices: list[np.ndarray] = []
    relations: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    dimension = max_nodes * (max_nodes - 1) // 2 + ATOM_BINS + BOND_BINS
    for graph_index, data in enumerate(dataset):
        graph, node_types, edge_types = _data_to_graph(data)
        centers = list(graph.nodes)
        columns: list[np.ndarray] = []
        kept_sets: list[set[int]] = []
        full_sizes: list[int] = []
        for center in centers:
            full_nodes = _ego_nodes(graph, center, radius)
            vector, kept_nodes, _ = _patch_vector(
                graph, full_nodes, center, node_types, edge_types, max_nodes
            )
            columns.append(vector)
            kept_sets.append(kept_nodes)
            full_sizes.append(len(full_nodes))
        if max_patches is not None and len(columns) > max_patches:
            rng = np.random.default_rng(seed + graph_index * 13)
            chosen = np.sort(rng.choice(len(columns), size=max_patches, replace=False))
            columns = [columns[int(index)] for index in chosen]
            kept_sets = [kept_sets[int(index)] for index in chosen]
            centers = [centers[int(index)] for index in chosen]
            full_sizes = [full_sizes[int(index)] for index in chosen]
        matrix = np.stack(columns, axis=1) if columns else np.zeros((dimension, 0), dtype=np.float32)
        matrices.append(matrix.astype(np.float32, copy=False))
        relation = _patch_relation_readout(graph, centers, kept_sets, matrix, max_nodes)
        relations.append(relation)
        metadata.append(
            {
                "graph_index": graph_index,
                "n_nodes": graph.n,
                "n_edges": graph.num_edges(),
                "n_patches": int(matrix.shape[1]),
                "mean_ego_nodes_full": float(np.mean(full_sizes)) if full_sizes else 0.0,
                "truncation_rate": float(np.mean(np.asarray(full_sizes) > max_nodes)) if full_sizes else 0.0,
            }
        )
    return matrices, relations, metadata


def _shortest_paths(graph, source: int) -> dict[int, int]:
    distances = {int(source): 0}
    queue: deque[int] = deque([int(source)])
    while queue:
        node = queue.popleft()
        for neighbor in graph.neighbors(node):
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    return distances


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(left @ right / denominator) if denominator > 1e-12 else 0.0


def _summary(values: Sequence[float]) -> np.ndarray:
    if not values:
        return np.zeros(6, dtype=np.float32)
    array = np.asarray(values, dtype=np.float64)
    return np.asarray(
        [
            array.mean(),
            array.std(),
            *np.quantile(array, [0.25, 0.50, 0.75]),
            array.max(),
        ],
        dtype=np.float32,
    )


def _patch_relation_readout(
    graph,
    centers: Sequence[int],
    kept_sets: Sequence[set[int]],
    matrix: np.ndarray,
    max_nodes: int,
) -> np.ndarray:
    if len(centers) < 2:
        return np.zeros(8 + 6 * 3 + 3, dtype=np.float32)
    paths = {center: _shortest_paths(graph, center) for center in centers}
    topology_dim = max_nodes * (max_nodes - 1) // 2
    distance_hist = np.zeros(8, dtype=np.float64)
    overlaps: list[float] = []
    topology_similarities: list[float] = []
    attribute_similarities: list[float] = []
    long_overlaps: list[float] = []
    for left_index, left in enumerate(centers):
        for right_index in range(left_index + 1, len(centers)):
            right = centers[right_index]
            distance = paths[left].get(right, 0)
            distance_hist[min(max(distance, 1), 8) - 1] += 1.0
            left_nodes, right_nodes = kept_sets[left_index], kept_sets[right_index]
            union = len(left_nodes | right_nodes)
            overlap = len(left_nodes & right_nodes) / max(union, 1)
            overlaps.append(overlap)
            topology_similarities.append(
                _cosine(matrix[:topology_dim, left_index], matrix[:topology_dim, right_index])
            )
            attribute_similarities.append(
                _cosine(matrix[topology_dim:, left_index], matrix[topology_dim:, right_index])
            )
            if distance >= 3:
                long_overlaps.append(overlap)
    distance_hist /= max(distance_hist.sum(), 1.0)
    output = np.concatenate(
        [
            distance_hist,
            _summary(overlaps),
            _summary(topology_similarities),
            _summary(attribute_similarities),
            np.asarray(
                [
                    float(np.mean(np.asarray(overlaps) > 0.25)),
                    float(np.mean(np.asarray(topology_similarities) > 0.75)),
                    float(np.mean(long_overlaps)) if long_overlaps else 0.0,
                ],
                dtype=np.float32,
            ),
        ]
    )
    return output.astype(np.float32, copy=False)


def _shuffle_attribute_columns(
    matrices: Sequence[np.ndarray],
    max_nodes: int,
    seed: int,
) -> list[np.ndarray]:
    topology_dim = max_nodes * (max_nodes - 1) // 2
    shuffled: list[np.ndarray] = []
    for graph_index, matrix in enumerate(matrices):
        if matrix.shape[1] < 2:
            shuffled.append(matrix.copy())
            continue
        rng = np.random.default_rng(seed + graph_index * 97)
        permutation = rng.permutation(matrix.shape[1])
        shuffled.append(
            np.concatenate([matrix[:topology_dim], matrix[topology_dim:, permutation]], axis=0)
        )
    return shuffled


def _joint_readout(
    matrices: Sequence[np.ndarray],
    max_nodes: int,
    *,
    shuffle_seed: int | None = None,
) -> np.ndarray:
    topology_dim = max_nodes * (max_nodes - 1) // 2
    rows: list[np.ndarray] = []
    for graph_index, original in enumerate(matrices):
        matrix = original
        if shuffle_seed is not None:
            matrix = _shuffle_attribute_columns([matrix], max_nodes, shuffle_seed + graph_index)[0]
        topology = matrix[:topology_dim]
        atom = matrix[topology_dim : topology_dim + ATOM_BINS]
        bond = matrix[topology_dim + ATOM_BINS :]
        n_patches = matrix.shape[1]
        joint_topology_atom = np.zeros((9, ATOM_BINS), dtype=np.float64)
        joint_topology_bond = np.zeros((9, BOND_BINS), dtype=np.float64)
        joint_atom_bond = np.zeros((ATOM_BINS, BOND_BINS), dtype=np.float64)
        if n_patches:
            edge_bins = np.clip(np.rint(topology.sum(axis=0)).astype(np.int64), 0, 8)
            for column, edge_bin in enumerate(edge_bins):
                joint_topology_atom[edge_bin] += atom[:, column]
                joint_topology_bond[edge_bin] += bond[:, column]
                joint_atom_bond += np.outer(atom[:, column], bond[:, column])
            joint_topology_atom /= n_patches
            joint_topology_bond /= n_patches
            joint_atom_bond /= n_patches
        rows.append(
            np.concatenate(
                [
                    joint_topology_atom.ravel(),
                    joint_topology_bond.ravel(),
                    joint_atom_bond.ravel(),
                    np.asarray([n_patches, np.log1p(n_patches)], dtype=np.float64),
                ]
            )
        )
    return np.stack(rows).astype(np.float32)


def _fit_mae(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    seed: int,
    n_jobs: int,
) -> float:
    model = XGBRegressor(
        n_estimators=260,
        max_depth=5,
        learning_rate=0.05,
        min_child_weight=4,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        reg_alpha=0.0,
        objective="reg:squarederror",
        eval_metric="mae",
        random_state=seed,
        n_jobs=n_jobs,
        tree_method="hist",
    )
    model.fit(x_train, y_train)
    return float(mean_absolute_error(y_valid, model.predict(x_valid)))


def _concat(*arrays: Sequence[np.ndarray]) -> list[np.ndarray]:
    return [np.concatenate([array[index] for array in arrays], axis=1) for index in range(3)]


def _run_radius(args: argparse.Namespace, radius: int) -> dict[str, Any]:
    root = _resolve(args.data_root)
    train_dataset = _load_zinc(root, "train")
    valid_dataset = _load_zinc(root, "val")
    train = train_dataset[args.train_offset : args.train_offset + args.max_train_graphs]
    valid = valid_dataset[args.valid_offset : args.valid_offset + args.max_valid_graphs]
    datasets = (train, valid)
    labels = [
        np.asarray([float(data.y.view(-1)[0]) for data in dataset], dtype=np.float32)
        for dataset in datasets
    ]
    # A third empty slot keeps feature concatenation code identical while test
    # remains completely unloaded and unreported.
    labels = [labels[0], labels[1], np.zeros(0, dtype=np.float32)]
    vectorized = [
        _vectorize_with_relations(ds, radius, args.max_nodes, args.max_patches, args.seed + split)
        for split, ds in enumerate(datasets)
    ]
    matrices = [item[0] for item in vectorized]
    relation_arrays = [
        np.stack(item[1]).astype(np.float32)
        if item[1]
        else np.zeros((0, 29), dtype=np.float32)
        for item in vectorized
    ]
    metadata = [item[2] for item in vectorized]
    relation_rows = relation_arrays + [np.zeros((0, relation_arrays[0].shape[1]), dtype=np.float32)]
    raw_typed = [select_and_normalize(rows, args.max_nodes, "typed") for rows in matrices]
    raw_topology = [select_and_normalize(rows, args.max_nodes, "topology") for rows in matrices]
    raw_attributes = [select_and_normalize(rows, args.max_nodes, "attributes") for rows in matrices]
    global_blocks = [global_feature_views(ds) for ds in datasets]
    empty_global = {name: np.zeros((0, values.shape[1]), dtype=np.float32) for name, values in global_blocks[0].items()}
    global_blocks.append(empty_global)
    views: dict[str, list[np.ndarray]] = {
        name: [global_blocks[split][name] for split in range(3)]
        for name in global_blocks[0]
    }
    views["local_typed_raw"] = [raw_readout(rows) for rows in raw_typed] + [np.zeros((0, raw_readout(raw_typed[0]).shape[1]), dtype=np.float32)]
    views["local_topology_raw"] = [raw_readout(rows) for rows in raw_topology] + [np.zeros((0, raw_readout(raw_topology[0]).shape[1]), dtype=np.float32)]
    views["local_attributes_raw"] = [raw_readout(rows) for rows in raw_attributes] + [np.zeros((0, raw_readout(raw_attributes[0]).shape[1]), dtype=np.float32)]
    views["local_typed_joint"] = [_joint_readout(rows, args.max_nodes) for rows in matrices] + [np.zeros((0, 402), dtype=np.float32)]
    for repeat in range(args.shuffle_repeats):
        views[f"local_typed_raw_shuffled_{repeat}"] = [
            raw_readout(select_and_normalize(_shuffle_attribute_columns(rows, args.max_nodes, args.seed + 1000 + repeat), args.max_nodes, "typed"))
            for rows in matrices
        ]
        views[f"local_typed_joint_shuffled_{repeat}"] = [
            _joint_readout(rows, args.max_nodes, shuffle_seed=args.seed + 2000 + repeat)
            for rows in matrices
        ]
        views[f"local_typed_raw_shuffled_{repeat}"].append(np.zeros((0, views[f"local_typed_raw_shuffled_{repeat}"][0].shape[1]), dtype=np.float32))
        views[f"local_typed_joint_shuffled_{repeat}"].append(np.zeros((0, 402), dtype=np.float32))
    views["local_patch_relation"] = relation_rows
    views["global_all_plus_local_typed_raw"] = _concat(views["global_all"], views["local_typed_raw"])
    views["global_all_plus_local_typed_joint"] = _concat(views["global_all"], views["local_typed_joint"])
    views["global_all_plus_local_patch_relation"] = _concat(views["global_all"], views["local_patch_relation"])
    views["global_all_plus_local_typed_raw_plus_relation"] = _concat(
        views["global_all"], views["local_typed_raw"], views["local_patch_relation"]
    )
    for repeat in range(args.shuffle_repeats):
        views[f"global_all_plus_local_typed_joint_shuffled_{repeat}"] = _concat(
            views["global_all"], views[f"local_typed_joint_shuffled_{repeat}"]
        )
    model_seeds = list(dict.fromkeys(int(seed) for seed in args.model_seeds))
    scores_by_seed: dict[str, list[float]] = {name: [] for name in views}
    for model_seed in model_seeds:
        for name, arrays in views.items():
            scores_by_seed[name].append(
                _fit_mae(arrays[0], labels[0], arrays[1], labels[1], model_seed, args.n_jobs)
            )
    screen = {name: float(np.mean(values)) for name, values in scores_by_seed.items()}
    score_std = {name: float(np.std(values)) for name, values in scores_by_seed.items()}
    baseline = screen["global_all"]
    raw_fused = screen["global_all_plus_local_typed_raw"]
    joint_fused = screen["global_all_plus_local_typed_joint"]
    relation_fused = screen["global_all_plus_local_patch_relation"]
    raw_relation_fused = screen["global_all_plus_local_typed_raw_plus_relation"]
    shuffle_joint = [screen[f"global_all_plus_local_typed_joint_shuffled_{i}"] for i in range(args.shuffle_repeats)]
    shuffle_raw = [screen[f"local_typed_raw_shuffled_{i}"] for i in range(args.shuffle_repeats)]
    joint_shuffle_gaps = [value - joint_fused for value in shuffle_joint]
    payload = {
        "protocol_id": f"luyin16-zinc-mechanism-screen-radius{radius}-v1",
        "status": "screen_only",
        "data": {
            "root": str(root),
            "train": len(train),
            "valid": len(valid),
            "train_offset": args.train_offset,
            "valid_offset": args.valid_offset,
            "test_loaded": False,
            "split": "PyG ZINC subset=True official train/valid slice for mechanism screen",
        },
        "representation": {
            "radius": radius,
            "max_nodes": args.max_nodes,
            "max_patches_per_graph": args.max_patches,
            "typed_patch_dimension": args.max_nodes * (args.max_nodes - 1) // 2 + ATOM_BINS + BOND_BINS,
            "joint_dimension": 402,
            "relation_dimension": int(relation_rows[0].shape[1]) if relation_rows[0].ndim == 2 else 0,
            "joint_definition": "topology-edge-bin x atom histogram, topology-edge-bin x bond histogram, atom x bond co-occurrence",
            "shuffle_definition": "permute attribute columns within each graph while preserving topology columns and marginal patch counts",
        },
        "sampling": {
            "train_mean_n_patches": float(np.mean([row["n_patches"] for row in metadata[0]])),
            "train_mean_ego_nodes": float(np.mean([row["mean_ego_nodes_full"] for row in metadata[0]])),
            "train_mean_truncation": float(np.mean([row["truncation_rate"] for row in metadata[0]])),
            "valid_mean_n_patches": float(np.mean([row["n_patches"] for row in metadata[1]])),
        },
        "screen": {
            "metric": "MAE (lower is better)",
            "fixed_xgb": {"n_estimators": 260, "max_depth": 5, "learning_rate": 0.05},
            "scores": screen,
            "score_std_over_model_seeds": score_std,
            "model_seeds": model_seeds,
            "baseline_global_all": baseline,
            "gain_global_plus_raw_over_global": baseline - raw_fused,
            "gain_global_plus_joint_over_global": baseline - joint_fused,
            "gain_global_plus_relation_over_global": baseline - relation_fused,
            "gain_raw_plus_relation_over_raw": raw_fused - raw_relation_fused,
            "joint_vs_shuffle_gap": joint_shuffle_gaps,
            "joint_vs_shuffle_mean_gap": float(np.mean(joint_shuffle_gaps)),
            "joint_vs_shuffle_min_gap": float(np.min(joint_shuffle_gaps)),
            "raw_shuffle_scores": shuffle_raw,
            "joint_shuffle_scores": shuffle_joint,
            "promotion_rule": {
                "relation": "candidate slice passes if global+raw+relation beats global+raw by >= 0.01",
                "binding": "candidate slice passes if mean(joint shuffle - joint) >= 0.01",
            },
        },
        "runtime": {"seed": args.seed, "shuffle_repeats": args.shuffle_repeats},
    }
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/ZINC")
    parser.add_argument("--radii", type=int, nargs="+", choices=[1, 2, 3], default=[2, 3])
    parser.add_argument("--max-train-graphs", type=int, default=2000)
    parser.add_argument("--max-valid-graphs", type=int, default=200)
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--valid-offset", type=int, default=0)
    parser.add_argument("--max-nodes", type=int, default=12)
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--shuffle-repeats", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    payloads = {f"radius{radius}": _run_radius(args, radius) for radius in args.radii}
    output = {
        "protocol_id": "luyin16-zinc-mechanism-screen-v1",
        "radii": payloads,
    }
    _write_json(_resolve(args.result), output)
    compact = {
        name: {
            "global": round(payload["screen"]["baseline_global_all"], 5),
            "global+raw": round(payload["screen"]["scores"]["global_all_plus_local_typed_raw"], 5),
            "global+joint": round(payload["screen"]["scores"]["global_all_plus_local_typed_joint"], 5),
            "global+relation": round(payload["screen"]["scores"]["global_all_plus_local_patch_relation"], 5),
            "global+raw+relation": round(
                payload["screen"]["scores"]["global_all_plus_local_typed_raw_plus_relation"], 5
            ),
            "raw_relation_gain": round(payload["screen"]["gain_raw_plus_relation_over_raw"], 5),
            "joint_shuffle_gap": [round(value, 5) for value in payload["screen"]["joint_vs_shuffle_gap"]],
        }
        for name, payload in payloads.items()
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
