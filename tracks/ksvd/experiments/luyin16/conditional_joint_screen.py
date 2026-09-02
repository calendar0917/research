"""Fast train-only screen for conditional structure--attribute statistics.

Each radius patch is assigned a permutation-invariant rooted structural
signature.  The readout then records atom and bond distributions conditioned
on that signature.  Within-graph attribute shuffles preserve structural and
attribute marginals while breaking their patch-wise binding.  The official
ZINC test split is never loaded.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tracks.ksvd.experiments.luyin16.mechanism_screen import _fit_mae, _write_json
from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import (
    ATOM_BINS,
    BOND_BINS,
    REPO_ROOT,
    _load_zinc,
    _resolve,
    global_feature_views,
    raw_readout,
    select_and_normalize,
    vectorize_dataset,
)

Signature = tuple[int, int, int, int, int]
SIGNATURE_FIELDS = (
    "n_nodes",
    "n_edges",
    "cycle_rank",
    "center_degree_or_shell1",
    "shell2_size",
)


def _upper_indices(max_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    left: list[int] = []
    right: list[int] = []
    for i in range(max_nodes):
        for j in range(i + 1, max_nodes):
            left.append(i)
            right.append(j)
    return np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)


def _structural_signature(
    topology: np.ndarray,
    max_nodes: int,
    upper_indices: tuple[np.ndarray, np.ndarray],
) -> Signature:
    adjacency = np.zeros((max_nodes, max_nodes), dtype=np.float32)
    left, right = upper_indices
    adjacency[left, right] = topology
    adjacency[right, left] = topology
    adjacency = adjacency > 0.5
    degree = adjacency.sum(axis=1).astype(np.int64)
    n_edges = int(np.rint(topology.sum()))
    n_nodes = int(np.count_nonzero(degree)) if n_edges else 1
    cycle_rank = max(n_edges - n_nodes + 1, 0)
    center_degree = int(degree[0])

    distances = np.full(max_nodes, -1, dtype=np.int64)
    distances[0] = 0
    queue: deque[int] = deque([0])
    while queue:
        node = queue.popleft()
        for neighbor in np.flatnonzero(adjacency[node]):
            neighbor = int(neighbor)
            if distances[neighbor] < 0:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    shell2_size = int(np.count_nonzero(distances == 2))
    return n_nodes, n_edges, cycle_rank, center_degree, shell2_size


def _signature_rows(matrices: Sequence[np.ndarray], max_nodes: int) -> list[list[Signature]]:
    topology_dim = max_nodes * (max_nodes - 1) // 2
    indices = _upper_indices(max_nodes)
    return [
        [
            _structural_signature(matrix[:topology_dim, column], max_nodes, indices)
            for column in range(matrix.shape[1])
        ]
        for matrix in matrices
    ]


def _fit_vocabulary(
    signatures: Sequence[Sequence[Signature]],
    max_signatures: int,
) -> tuple[dict[Signature, int], Counter[Signature]]:
    counts: Counter[Signature] = Counter(signature for graph in signatures for signature in graph)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:max_signatures]
    return {signature: index for index, (signature, _) in enumerate(ordered)}, counts


def _conditional_readout(
    matrices: Sequence[np.ndarray],
    signatures: Sequence[Sequence[Signature]],
    vocabulary: Mapping[Signature, int],
    max_nodes: int,
    *,
    shuffle_seed: int | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    topology_dim = max_nodes * (max_nodes - 1) // 2
    unknown_index = len(vocabulary)
    n_signature_bins = unknown_index + 1
    rows: list[np.ndarray] = []
    unknown = 0
    total = 0
    for graph_index, (matrix, graph_signatures) in enumerate(zip(matrices, signatures, strict=True)):
        n_patches = matrix.shape[1]
        attribute_columns = np.arange(n_patches)
        if shuffle_seed is not None and n_patches > 1:
            rng = np.random.default_rng(shuffle_seed + graph_index * 97)
            attribute_columns = rng.permutation(n_patches)
        signature_mass = np.zeros(n_signature_bins, dtype=np.float64)
        atom_joint = np.zeros((n_signature_bins, ATOM_BINS), dtype=np.float64)
        bond_joint = np.zeros((n_signature_bins, BOND_BINS), dtype=np.float64)
        for patch_index, signature in enumerate(graph_signatures):
            signature_index = vocabulary.get(signature, unknown_index)
            attribute_index = int(attribute_columns[patch_index])
            signature_mass[signature_index] += 1.0
            atom_joint[signature_index] += matrix[
                topology_dim : topology_dim + ATOM_BINS, attribute_index
            ]
            bond_joint[signature_index] += matrix[
                topology_dim + ATOM_BINS : topology_dim + ATOM_BINS + BOND_BINS,
                attribute_index,
            ]
            unknown += int(signature_index == unknown_index)
            total += 1
        denominator = max(n_patches, 1)
        rows.append(
            np.concatenate(
                [
                    signature_mass / denominator,
                    atom_joint.ravel() / denominator,
                    bond_joint.ravel() / denominator,
                    np.asarray([n_patches, np.log1p(n_patches)], dtype=np.float64),
                ]
            )
        )
    return np.stack(rows).astype(np.float32), {
        "unknown_patch_rate": unknown / max(total, 1),
        "n_patches": float(total),
    }


def _structure_signature_readout(
    signatures: Sequence[Sequence[Signature]],
    vocabulary: Mapping[Signature, int],
) -> np.ndarray:
    unknown_index = len(vocabulary)
    rows: list[np.ndarray] = []
    for graph_signatures in signatures:
        counts = np.zeros(unknown_index + 1, dtype=np.float64)
        for signature in graph_signatures:
            counts[vocabulary.get(signature, unknown_index)] += 1.0
        n_patches = len(graph_signatures)
        rows.append(
            np.concatenate(
                [
                    counts / max(n_patches, 1),
                    np.asarray([n_patches, np.log1p(n_patches)], dtype=np.float64),
                ]
            )
        )
    return np.stack(rows).astype(np.float32)


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
        vectorize_dataset(dataset, args.radius, args.max_nodes, args.max_patches, args.seed + split)
        for split, dataset in enumerate(datasets)
    ]
    matrices = [item[0] for item in vectorized]
    metadata = [item[1] for item in vectorized]
    signatures = [_signature_rows(rows, args.max_nodes) for rows in matrices]
    vocabulary, vocabulary_counts = _fit_vocabulary(signatures[0], args.max_signatures)

    conditional: list[np.ndarray] = []
    conditional_metadata: list[dict[str, float]] = []
    for rows, row_signatures in zip(matrices, signatures, strict=True):
        values, details = _conditional_readout(
            rows, row_signatures, vocabulary, args.max_nodes
        )
        conditional.append(values)
        conditional_metadata.append(details)

    global_all = [global_feature_views(dataset)["global_all"] for dataset in datasets]
    typed_raw = [
        raw_readout(select_and_normalize(rows, args.max_nodes, "typed")) for rows in matrices
    ]
    signature_only = [
        _structure_signature_readout(row_signatures, vocabulary)
        for row_signatures in signatures
    ]
    views: dict[str, list[np.ndarray]] = {
        "global_all": global_all,
        "global_all_plus_typed_raw": _concat(global_all, typed_raw),
        "global_all_plus_signature": _concat(global_all, signature_only),
        "global_all_plus_conditional_v2": _concat(global_all, conditional),
    }
    for repeat in range(args.shuffle_repeats):
        shuffled = [
            _conditional_readout(
                rows,
                row_signatures,
                vocabulary,
                args.max_nodes,
                shuffle_seed=args.seed + 2000 + repeat,
            )[0]
            for rows, row_signatures in zip(matrices, signatures, strict=True)
        ]
        views[f"global_all_plus_conditional_v2_shuffled_{repeat}"] = _concat(
            global_all, shuffled
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
    conditional_score = scores["global_all_plus_conditional_v2"]
    shuffle_scores = [
        scores[f"global_all_plus_conditional_v2_shuffled_{repeat}"]
        for repeat in range(args.shuffle_repeats)
    ]
    shuffle_gaps = [score - conditional_score for score in shuffle_scores]
    raw_score = scores["global_all_plus_typed_raw"]
    kept_patch_mass = sum(vocabulary_counts[signature] for signature in vocabulary)
    total_patch_mass = sum(vocabulary_counts.values())

    return {
        "protocol_id": "luyin16-zinc-conditional-joint-v2-screen-v1",
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
            "max_patches_per_graph": args.max_patches,
            "signature_fields": SIGNATURE_FIELDS,
            "vocabulary_fit": "training patches only; frequency-descending with lexical tie break",
            "max_signatures": args.max_signatures,
            "n_unique_train_signatures": len(vocabulary_counts),
            "n_kept_signatures": len(vocabulary),
            "train_vocabulary_patch_coverage": kept_patch_mass / max(total_patch_mass, 1),
            "conditional_dimension": int(conditional[0].shape[1]),
            "train_unknown_patch_rate": conditional_metadata[0]["unknown_patch_rate"],
            "valid_unknown_patch_rate": conditional_metadata[1]["unknown_patch_rate"],
            "top_signatures": [
                {"signature": signature, "count": vocabulary_counts[signature]}
                for signature in vocabulary
            ],
            "shuffle_definition": "permute atom+bond columns together within each graph while preserving structural signatures and both marginals",
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
            "model_seeds": model_seeds,
            "scores": scores,
            "scores_by_seed": scores_by_seed,
            "score_std_over_model_seeds": score_std,
            "gain_conditional_over_global": scores["global_all"] - conditional_score,
            "gain_conditional_over_typed_raw": raw_score - conditional_score,
            "conditional_vs_shuffle_gap": shuffle_gaps,
            "conditional_vs_shuffle_mean_gap": float(np.mean(shuffle_gaps)),
            "conditional_vs_shuffle_min_gap": float(np.min(shuffle_gaps)),
            "small_slice_gate": {
                "binding": "mean(shuffle - conditional) >= 0.01 on both non-overlapping slices",
                "prediction": "conditional beats typed raw on at least one small slice",
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
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--max-signatures", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--shuffle-repeats", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = _run(args)
    _write_json(_resolve(args.result), payload)
    compact = {
        "global": round(payload["screen"]["scores"]["global_all"], 5),
        "global+raw": round(payload["screen"]["scores"]["global_all_plus_typed_raw"], 5),
        "global+signature": round(
            payload["screen"]["scores"]["global_all_plus_signature"], 5
        ),
        "global+conditional_v2": round(
            payload["screen"]["scores"]["global_all_plus_conditional_v2"], 5
        ),
        "conditional_over_raw_gain": round(
            payload["screen"]["gain_conditional_over_typed_raw"], 5
        ),
        "shuffle_gap": [
            round(value, 5) for value in payload["screen"]["conditional_vs_shuffle_gap"]
        ],
        "valid_unknown_rate": round(
            payload["representation"]["valid_unknown_patch_rate"], 5
        ),
    }
    print(compact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
