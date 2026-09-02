"""Screen a low-rank conditional binding residual on official ZINC train/valid.

The representation removes structure and attribute marginals from their joint
distribution, then learns a train-only low-rank SVD projection.  Within-graph
attribute shuffles receive independently fitted train-only projections with
the same budget.  The official test split is never loaded.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.decomposition import TruncatedSVD

from tracks.ksvd.experiments.luyin16.conditional_joint_screen import (
    SIGNATURE_FIELDS,
    _conditional_readout,
    _fit_vocabulary,
    _signature_rows,
)
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


def _centered_conditional_residual(
    conditional: np.ndarray,
    n_signature_bins: int,
) -> np.ndarray:
    signature_mass = conditional[:, :n_signature_bins]
    atom_start = n_signature_bins
    atom_stop = atom_start + n_signature_bins * ATOM_BINS
    bond_stop = atom_stop + n_signature_bins * BOND_BINS
    atom_joint = conditional[:, atom_start:atom_stop].reshape(
        -1, n_signature_bins, ATOM_BINS
    )
    bond_joint = conditional[:, atom_stop:bond_stop].reshape(
        -1, n_signature_bins, BOND_BINS
    )
    atom_marginal = atom_joint.sum(axis=1)
    bond_marginal = bond_joint.sum(axis=1)
    atom_expected = signature_mass[:, :, None] * atom_marginal[:, None, :]
    bond_expected = signature_mass[:, :, None] * bond_marginal[:, None, :]
    return np.concatenate(
        [
            (atom_joint - atom_expected).reshape(conditional.shape[0], -1),
            (bond_joint - bond_expected).reshape(conditional.shape[0], -1),
        ],
        axis=1,
    ).astype(np.float32)


def _fit_svd(
    train: np.ndarray,
    valid: np.ndarray,
    n_components: int,
    seed: int,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    train_mean = train.mean(axis=0, keepdims=True)
    centered_train = train - train_mean
    centered_valid = valid - train_mean
    maximum = min(centered_train.shape[0] - 1, centered_train.shape[1] - 1)
    actual_components = max(1, min(n_components, maximum))
    projector = TruncatedSVD(
        n_components=actual_components,
        algorithm="randomized",
        n_iter=7,
        random_state=seed,
    )
    transformed = [
        projector.fit_transform(centered_train).astype(np.float32),
        projector.transform(centered_valid).astype(np.float32),
    ]
    return transformed, {
        "requested_components": n_components,
        "actual_components": actual_components,
        "explained_variance_ratio_sum": float(projector.explained_variance_ratio_.sum()),
        "train_mean_l2": float(np.linalg.norm(train_mean)),
        "train_residual_mean_l2": float(np.linalg.norm(train, axis=1).mean()),
        "valid_residual_mean_l2": float(np.linalg.norm(valid, axis=1).mean()),
    }


def _concat(*blocks: Sequence[np.ndarray]) -> list[np.ndarray]:
    return [np.concatenate([block[index] for block in blocks], axis=1) for index in range(2)]


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
    n_signature_bins = len(vocabulary) + 1

    conditional = [
        _conditional_readout(rows, row_signatures, vocabulary, args.max_nodes)[0]
        for rows, row_signatures in zip(matrices, signatures, strict=True)
    ]
    residuals = [
        _centered_conditional_residual(values, n_signature_bins) for values in conditional
    ]
    residual_svd, residual_svd_metadata = _fit_svd(
        residuals[0], residuals[1], args.svd_components, args.seed
    )

    global_all = [global_feature_views(dataset)["global_all"] for dataset in datasets]
    typed_raw = [
        raw_readout(select_and_normalize(rows, args.max_nodes, "typed")) for rows in matrices
    ]
    global_raw = _concat(global_all, typed_raw)
    views: dict[str, list[np.ndarray]] = {
        "global_all_plus_typed_raw": global_raw,
        "global_all_plus_typed_raw_plus_residual_svd": _concat(global_raw, residual_svd),
    }
    shuffled_metadata: list[dict[str, Any]] = []
    for repeat in range(args.shuffle_repeats):
        shuffled_conditional = [
            _conditional_readout(
                rows,
                row_signatures,
                vocabulary,
                args.max_nodes,
                shuffle_seed=args.seed + 2000 + repeat,
            )[0]
            for rows, row_signatures in zip(matrices, signatures, strict=True)
        ]
        shuffled_residuals = [
            _centered_conditional_residual(values, n_signature_bins)
            for values in shuffled_conditional
        ]
        shuffled_svd, details = _fit_svd(
            shuffled_residuals[0],
            shuffled_residuals[1],
            args.svd_components,
            args.seed + 100 + repeat,
        )
        shuffled_metadata.append(details)
        views[f"global_all_plus_typed_raw_plus_residual_svd_shuffled_{repeat}"] = _concat(
            global_raw, shuffled_svd
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
    true_score = scores["global_all_plus_typed_raw_plus_residual_svd"]
    shuffle_scores = [
        scores[f"global_all_plus_typed_raw_plus_residual_svd_shuffled_{repeat}"]
        for repeat in range(args.shuffle_repeats)
    ]
    shuffle_gaps = [score - true_score for score in shuffle_scores]
    kept_patch_mass = sum(vocabulary_counts[signature] for signature in vocabulary)
    total_patch_mass = sum(vocabulary_counts.values())

    return {
        "protocol_id": "luyin16-zinc-centered-conditional-residual-svd16-screen-v1",
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
            "vocabulary_fit": "training patches only",
            "max_signatures": args.max_signatures,
            "n_unique_train_signatures": len(vocabulary_counts),
            "n_kept_signatures": len(vocabulary),
            "train_vocabulary_patch_coverage": kept_patch_mass / max(total_patch_mass, 1),
            "raw_residual_dimension": int(residuals[0].shape[1]),
            "residual_definition": "P(signature,attribute)-P(signature)P(attribute)",
            "svd_fit": "subtract training mean; fit randomized TruncatedSVD on training residuals only",
            "true_svd": residual_svd_metadata,
            "shuffled_svd": shuffled_metadata,
            "shuffle_definition": "within-graph joint atom+bond column permutation; independent train-only SVD with matched dimension",
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
            "gain_residual_svd_over_typed_raw": baseline - true_score,
            "residual_svd_vs_shuffle_gap": shuffle_gaps,
            "residual_svd_vs_shuffle_mean_gap": float(np.mean(shuffle_gaps)),
            "residual_svd_vs_shuffle_min_gap": float(np.min(shuffle_gaps)),
            "small_slice_gate": {
                "prediction": "residual SVD beats typed raw by >=0.01 on both slices",
                "binding": "mean(shuffle - true) >=0.01 on both slices",
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
    parser.add_argument("--svd-components", type=int, default=16)
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
        "global+raw+residual_svd": round(
            payload["screen"]["scores"]["global_all_plus_typed_raw_plus_residual_svd"], 5
        ),
        "residual_gain": round(payload["screen"]["gain_residual_svd_over_typed_raw"], 5),
        "shuffle_gap": [
            round(value, 5) for value in payload["screen"]["residual_svd_vs_shuffle_gap"]
        ],
        "svd_explained": round(
            payload["representation"]["true_svd"]["explained_variance_ratio_sum"], 5
        ),
    }
    print(compact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
