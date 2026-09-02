"""Fast ZINC screen for long-range relations between local rooted objects.

Each node-centred radius patch becomes an object type consisting of a rooted
structural signature and the centre atom type.  Graph features count pairs of
object types conditioned on centre shortest-path distance.  Position-shuffle
controls preserve the object bag and graph topology while breaking their
assignment.  The official test split is never loaded.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tracks.ksvd.experiments.luyin16.conditional_joint_screen import (
    Signature,
    _signature_rows,
)
from tracks.ksvd.experiments.luyin16.mechanism_screen import _fit_mae, _write_json
from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import (
    REPO_ROOT,
    _data_to_graph,
    _load_zinc,
    _resolve,
    global_feature_views,
    raw_readout,
    select_and_normalize,
    vectorize_dataset,
)

ObjectType = tuple[int, ...]
RelationToken = tuple[int, ...]
PairRow = list[tuple[int, int, int]]


def _object_type(signature: Signature, atom_type: int, schema: str) -> ObjectType:
    if schema == "atom":
        return (int(atom_type),)
    if schema == "atom_degree":
        return int(atom_type), int(signature[3])
    if schema == "signature_atom":
        return (*signature, int(atom_type))
    raise ValueError(f"unknown object schema: {schema}")


def _relation_token(distance_bin: int, left: ObjectType, right: ObjectType) -> RelationToken:
    first, second = sorted((left, right))
    return (int(distance_bin), *first, *second)


def _pair_rows(graph, min_distance: int, max_distance_bin: int) -> PairRow:
    pairs: PairRow = []
    nodes = list(graph.nodes)
    position = {node: index for index, node in enumerate(nodes)}
    for source_index, source in enumerate(nodes):
        distances = {int(source): 0}
        queue: deque[int] = deque([int(source)])
        while queue:
            node = queue.popleft()
            for neighbor in graph.neighbors(node):
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        for target in nodes[source_index + 1 :]:
            distance = distances.get(target)
            if distance is None or distance < min_distance:
                continue
            pairs.append((source_index, position[target], min(distance, max_distance_bin)))
    return pairs


def _build_object_graphs(
    dataset,
    signatures: Sequence[Sequence[Signature]],
    min_distance: int,
    max_distance_bin: int,
    object_schema: str,
) -> tuple[list[list[ObjectType]], list[PairRow]]:
    object_rows: list[list[ObjectType]] = []
    graph_pairs: list[PairRow] = []
    for data, graph_signatures in zip(dataset, signatures, strict=True):
        graph, node_types, _ = _data_to_graph(data)
        centers = list(graph.nodes)
        if len(centers) != len(graph_signatures):
            raise ValueError("all-centre vectorization is required for object relations")
        object_rows.append(
            [
                _object_type(signature, int(node_types[center]), object_schema)
                for center, signature in zip(centers, graph_signatures, strict=True)
            ]
        )
        graph_pairs.append(_pair_rows(graph, min_distance, max_distance_bin))
    return object_rows, graph_pairs


def _tokens(objects: Sequence[ObjectType], pairs: PairRow) -> list[RelationToken]:
    return [
        _relation_token(distance, objects[left], objects[right])
        for left, right, distance in pairs
    ]


def _fit_vocabulary(
    object_rows: Sequence[Sequence[ObjectType]],
    pair_rows: Sequence[PairRow],
    max_relations: int,
) -> tuple[dict[RelationToken, int], Counter[RelationToken]]:
    counts: Counter[RelationToken] = Counter()
    for objects, pairs in zip(object_rows, pair_rows, strict=True):
        counts.update(_tokens(objects, pairs))
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:max_relations]
    return {token: index for index, (token, _) in enumerate(ordered)}, counts


def _relation_readout(
    object_rows: Sequence[Sequence[ObjectType]],
    pair_rows: Sequence[PairRow],
    vocabulary: Mapping[RelationToken, int],
    *,
    shuffle_seed: int | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    unknown_index = len(vocabulary)
    rows: list[np.ndarray] = []
    unknown = 0
    total = 0
    for graph_index, (original_objects, pairs) in enumerate(
        zip(object_rows, pair_rows, strict=True)
    ):
        objects = list(original_objects)
        if shuffle_seed is not None and len(objects) > 1:
            rng = np.random.default_rng(shuffle_seed + graph_index * 97)
            permutation = rng.permutation(len(objects))
            objects = [objects[int(index)] for index in permutation]
        counts = np.zeros(unknown_index + 1, dtype=np.float64)
        for token in _tokens(objects, pairs):
            token_index = vocabulary.get(token, unknown_index)
            counts[token_index] += 1.0
            unknown += int(token_index == unknown_index)
            total += 1
        n_pairs = len(pairs)
        rows.append(
            np.concatenate(
                [
                    counts / max(n_pairs, 1),
                    np.asarray(
                        [len(objects), np.log1p(len(objects)), n_pairs, np.log1p(n_pairs)],
                        dtype=np.float64,
                    ),
                ]
            )
        )
    return np.stack(rows).astype(np.float32), {
        "unknown_relation_rate": unknown / max(total, 1),
        "n_relation_pairs": float(total),
    }


def _concat(left: Sequence[np.ndarray], right: Sequence[np.ndarray]) -> list[np.ndarray]:
    return [np.concatenate([left[index], right[index]], axis=1) for index in range(2)]


def _run(args: argparse.Namespace) -> dict[str, Any]:
    root = _resolve(args.data_root)
    train_source = _load_zinc(root, "train")
    valid_source = _load_zinc(root, "val")
    train = train_source[args.train_offset : args.train_offset + args.max_train_graphs]
    valid = valid_source[args.valid_offset : args.valid_offset + args.max_valid_graphs]
    datasets = (train, valid)
    labels = [
        np.asarray([float(data.y.view(-1)[0]) for data in dataset], dtype=np.float32)
        for dataset in datasets
    ]
    vectorized = [
        vectorize_dataset(dataset, args.radius, args.max_nodes, None, args.seed + split)
        for split, dataset in enumerate(datasets)
    ]
    matrices = [item[0] for item in vectorized]
    metadata = [item[1] for item in vectorized]
    signatures = [_signature_rows(rows, args.max_nodes) for rows in matrices]
    object_graphs = [
        _build_object_graphs(
            dataset,
            row_signatures,
            args.min_distance,
            args.max_distance_bin,
            args.object_schema,
        )
        for dataset, row_signatures in zip(datasets, signatures, strict=True)
    ]
    object_rows = [item[0] for item in object_graphs]
    pair_rows = [item[1] for item in object_graphs]
    vocabulary, vocabulary_counts = _fit_vocabulary(
        object_rows[0], pair_rows[0], args.max_relations
    )
    true_relation, true_metadata = _relation_readout(
        object_rows[0], pair_rows[0], vocabulary
    )
    valid_relation, valid_metadata = _relation_readout(
        object_rows[1], pair_rows[1], vocabulary
    )

    global_all = [global_feature_views(dataset)["global_all"] for dataset in datasets]
    typed_raw = [
        raw_readout(select_and_normalize(rows, args.max_nodes, "typed")) for rows in matrices
    ]
    global_raw = _concat(global_all, typed_raw)
    views: dict[str, list[np.ndarray]] = {
        "global_all_plus_typed_raw": global_raw,
        "global_all_plus_typed_raw_plus_long_relation": _concat(
            global_raw, [true_relation, valid_relation]
        ),
    }
    shuffled_metadata: list[dict[str, Any]] = []
    for repeat in range(args.shuffle_repeats):
        shuffle_seed = args.seed + 2000 + repeat
        shuffled_train_objects = []
        shuffled_valid_objects = []
        for graph_index, objects in enumerate(object_rows[0]):
            rng = np.random.default_rng(shuffle_seed + graph_index * 97)
            permutation = rng.permutation(len(objects))
            shuffled_train_objects.append([objects[int(index)] for index in permutation])
        for graph_index, objects in enumerate(object_rows[1]):
            rng = np.random.default_rng(shuffle_seed + graph_index * 97)
            permutation = rng.permutation(len(objects))
            shuffled_valid_objects.append([objects[int(index)] for index in permutation])
        shuffled_vocabulary, shuffled_counts = _fit_vocabulary(
            shuffled_train_objects, pair_rows[0], args.max_relations
        )
        shuffled_train, train_details = _relation_readout(
            shuffled_train_objects, pair_rows[0], shuffled_vocabulary
        )
        shuffled_valid, valid_details = _relation_readout(
            shuffled_valid_objects, pair_rows[1], shuffled_vocabulary
        )
        views[f"global_all_plus_typed_raw_plus_long_relation_shuffled_{repeat}"] = _concat(
            global_raw, [shuffled_train, shuffled_valid]
        )
        kept_mass = sum(shuffled_counts[token] for token in shuffled_vocabulary)
        shuffled_metadata.append(
            {
                "n_unique_train_relations": len(shuffled_counts),
                "n_kept_relations": len(shuffled_vocabulary),
                "train_vocabulary_pair_coverage": kept_mass / max(sum(shuffled_counts.values()), 1),
                "train_unknown_relation_rate": train_details["unknown_relation_rate"],
                "valid_unknown_relation_rate": valid_details["unknown_relation_rate"],
            }
        )

    model_seeds = list(dict.fromkeys(int(seed) for seed in args.model_seeds))
    scores_by_seed: dict[str, list[float]] = {name: [] for name in views}
    for model_seed in model_seeds:
        for name, arrays in views.items():
            scores_by_seed[name].append(
                _fit_mae(arrays[0], labels[0], arrays[1], labels[1], model_seed, args.n_jobs)
            )
    scores = {name: float(np.mean(values)) for name, values in scores_by_seed.items()}
    score_std = {name: float(np.std(values)) for name, values in scores_by_seed.items()}
    baseline = scores["global_all_plus_typed_raw"]
    true_score = scores["global_all_plus_typed_raw_plus_long_relation"]
    shuffle_scores = [
        scores[f"global_all_plus_typed_raw_plus_long_relation_shuffled_{repeat}"]
        for repeat in range(args.shuffle_repeats)
    ]
    shuffle_gaps = [score - true_score for score in shuffle_scores]
    kept_patch_mass = sum(vocabulary_counts[token] for token in vocabulary)
    total_patch_mass = sum(vocabulary_counts.values())

    return {
        "protocol_id": "luyin16-zinc-long-range-local-object-relation-screen-v1",
        "status": "screen_only",
        "data": {
            "root": str(root),
            "train": len(train),
            "valid": len(valid),
            "train_offset": args.train_offset,
            "valid_offset": args.valid_offset,
            "test_loaded": False,
            "split": "PyG ZINC subset=True official train/valid slice",
        },
        "representation": {
            "radius": args.radius,
            "max_nodes": args.max_nodes,
            "object_schema": args.object_schema,
            "object_type": {
                "atom": "centre atom type",
                "atom_degree": "centre atom type plus rooted patch centre degree",
                "signature_atom": "full rooted structural signature plus centre atom type",
            }[args.object_schema],
            "relation_token": "unordered object-type pair plus centre shortest-path distance bin",
            "min_distance": args.min_distance,
            "max_distance_bin": args.max_distance_bin,
            "vocabulary_fit": "top-frequency relation tokens on training graphs only",
            "max_relations": args.max_relations,
            "n_unique_train_relations": len(vocabulary_counts),
            "n_kept_relations": len(vocabulary),
            "train_vocabulary_pair_coverage": kept_patch_mass / max(total_patch_mass, 1),
            "relation_dimension": int(true_relation.shape[1]),
            "train_unknown_relation_rate": true_metadata["unknown_relation_rate"],
            "valid_unknown_relation_rate": valid_metadata["unknown_relation_rate"],
            "shuffle_definition": "permute complete local-object types over graph centres; preserve topology and object bag; fit matched train-only relation vocabulary",
            "shuffled_vocabulary": shuffled_metadata,
            "top_relations": [
                {"token": token, "count": vocabulary_counts[token]} for token in vocabulary
            ],
        },
        "sampling": {
            "train_mean_n_patches": float(np.mean([row["n_patches"] for row in metadata[0]])),
            "train_mean_ego_nodes": float(np.mean([row["mean_ego_nodes_full"] for row in metadata[0]])),
            "train_mean_truncation": float(np.mean([row["truncation_rate"] for row in metadata[0]])),
            "train_mean_long_relation_pairs": true_metadata["n_relation_pairs"] / max(len(train), 1),
            "valid_mean_long_relation_pairs": valid_metadata["n_relation_pairs"] / max(len(valid), 1),
        },
        "screen": {
            "metric": "MAE (lower is better)",
            "fixed_xgb": {"n_estimators": 260, "max_depth": 5, "learning_rate": 0.05},
            "model_seeds": model_seeds,
            "scores": scores,
            "scores_by_seed": scores_by_seed,
            "score_std_over_model_seeds": score_std,
            "gain_long_relation_over_typed_raw": baseline - true_score,
            "long_relation_vs_shuffle_gap": shuffle_gaps,
            "long_relation_vs_shuffle_mean_gap": float(np.mean(shuffle_gaps)),
            "long_relation_vs_shuffle_min_gap": float(np.min(shuffle_gaps)),
            "small_slice_gate": {
                "prediction": "long relation beats typed raw by >=0.01 on both slices",
                "relation": "mean(shuffle - true) >=0.01 on both slices",
            },
        },
        "runtime": {
            "seed": args.seed,
            "shuffle_repeats": args.shuffle_repeats,
            "n_jobs": args.n_jobs,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/ZINC")
    parser.add_argument("--radius", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--max-train-graphs", type=int, default=2000)
    parser.add_argument("--max-valid-graphs", type=int, default=200)
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--valid-offset", type=int, default=0)
    parser.add_argument("--max-nodes", type=int, default=12)
    parser.add_argument("--min-distance", type=int, default=3)
    parser.add_argument("--max-distance-bin", type=int, default=7)
    parser.add_argument(
        "--object-schema",
        choices=["atom", "atom_degree", "signature_atom"],
        default="atom_degree",
    )
    parser.add_argument("--max-relations", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--shuffle-repeats", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = _run(args)
    _write_json(_resolve(args.result), payload)
    compact = {
        "global+raw": round(payload["screen"]["scores"]["global_all_plus_typed_raw"], 5),
        "global+raw+long_relation": round(
            payload["screen"]["scores"]["global_all_plus_typed_raw_plus_long_relation"], 5
        ),
        "relation_gain": round(payload["screen"]["gain_long_relation_over_typed_raw"], 5),
        "shuffle_gap": [
            round(value, 5) for value in payload["screen"]["long_relation_vs_shuffle_gap"]
        ],
        "train_pair_coverage": round(
            payload["representation"]["train_vocabulary_pair_coverage"], 5
        ),
        "valid_unknown_rate": round(
            payload["representation"]["valid_unknown_relation_rate"], 5
        ),
    }
    print(compact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
