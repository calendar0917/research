"""Fast ZINC screen for radius-2/radius-3 typed-readout complementarity.

The runner compares global+r2, global+r3, and global+r2+r3 under one fixed
XGBoost budget on official train/valid slices.  Radius-specific node caps are
chosen to avoid material patch truncation.  The official test split is never
loaded.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tracks.ksvd.experiments.luyin16.mechanism_screen import _fit_mae, _write_json
from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import (
    REPO_ROOT,
    _load_zinc,
    _resolve,
    global_feature_views,
    raw_readout,
    select_and_normalize,
    vectorize_dataset,
)


def _concat(*blocks: Sequence[np.ndarray]) -> list[np.ndarray]:
    return [np.concatenate([block[index] for block in blocks], axis=1) for index in range(2)]


def _sampling_summary(metadata: Sequence[dict[str, Any]]) -> dict[str, float]:
    return {
        "mean_n_patches": float(np.mean([row["n_patches"] for row in metadata])),
        "mean_ego_nodes": float(np.mean([row["mean_ego_nodes_full"] for row in metadata])),
        "mean_truncation": float(np.mean([row["truncation_rate"] for row in metadata])),
        "mean_pair_coverage": float(np.mean([row["pair_coverage"] for row in metadata])),
    }


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

    radius_settings = {
        "r2": (2, args.r2_max_nodes),
        "r3": (3, args.r3_max_nodes),
    }
    local_views: dict[str, list[np.ndarray]] = {}
    sampling: dict[str, list[dict[str, float]]] = {}
    for name, (radius, max_nodes) in radius_settings.items():
        vectorized = [
            vectorize_dataset(dataset, radius, max_nodes, None, args.seed + split)
            for split, dataset in enumerate(datasets)
        ]
        matrices = [item[0] for item in vectorized]
        metadata = [item[1] for item in vectorized]
        local_views[name] = [
            raw_readout(select_and_normalize(rows, max_nodes, "typed")) for rows in matrices
        ]
        sampling[name] = [_sampling_summary(rows) for rows in metadata]

    global_all = [global_feature_views(dataset)["global_all"] for dataset in datasets]
    views = {
        "global_all_plus_r2_raw": _concat(global_all, local_views["r2"]),
        "global_all_plus_r3_raw": _concat(global_all, local_views["r3"]),
        "global_all_plus_r2_raw_plus_r3_raw": _concat(
            global_all, local_views["r2"], local_views["r3"]
        ),
    }
    model_seeds = list(dict.fromkeys(int(seed) for seed in args.model_seeds))
    scores_by_seed: dict[str, list[float]] = {name: [] for name in views}
    for model_seed in model_seeds:
        for name, arrays in views.items():
            scores_by_seed[name].append(
                _fit_mae(arrays[0], labels[0], arrays[1], labels[1], model_seed, args.n_jobs)
            )
    scores = {name: float(np.mean(values)) for name, values in scores_by_seed.items()}
    score_std = {name: float(np.std(values)) for name, values in scores_by_seed.items()}
    r3_score = scores["global_all_plus_r3_raw"]
    multiscale_score = scores["global_all_plus_r2_raw_plus_r3_raw"]

    return {
        "protocol_id": "luyin16-zinc-r2-r3-multiscale-screen-v1",
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
            "readout": "typed raw coordinate-wise mean/std/q25/q50/q75 plus patch count",
            "r2": {"radius": 2, "max_nodes": args.r2_max_nodes},
            "r3": {"radius": 3, "max_nodes": args.r3_max_nodes},
            "fusion": "feature concatenation before fixed XGBoost",
        },
        "sampling": {
            "r2": {"train": sampling["r2"][0], "valid": sampling["r2"][1]},
            "r3": {"train": sampling["r3"][0], "valid": sampling["r3"][1]},
        },
        "screen": {
            "metric": "MAE (lower is better)",
            "fixed_xgb": {"n_estimators": 260, "max_depth": 5, "learning_rate": 0.05},
            "model_seeds": model_seeds,
            "scores": scores,
            "scores_by_seed": scores_by_seed,
            "score_std_over_model_seeds": score_std,
            "gain_r3_over_r2": scores["global_all_plus_r2_raw"] - r3_score,
            "gain_multiscale_over_r3": r3_score - multiscale_score,
            "small_slice_gate": "multiscale beats r3 by >=0.01 MAE on both non-overlapping slices",
        },
        "runtime": {"seed": args.seed, "n_jobs": args.n_jobs},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/ZINC")
    parser.add_argument("--max-train-graphs", type=int, default=2000)
    parser.add_argument("--max-valid-graphs", type=int, default=200)
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--valid-offset", type=int, default=0)
    parser.add_argument("--r2-max-nodes", type=int, default=12)
    parser.add_argument("--r3-max-nodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = _run(args)
    _write_json(_resolve(args.result), payload)
    compact = {
        "global+r2": round(payload["screen"]["scores"]["global_all_plus_r2_raw"], 5),
        "global+r3": round(payload["screen"]["scores"]["global_all_plus_r3_raw"], 5),
        "global+r2+r3": round(
            payload["screen"]["scores"]["global_all_plus_r2_raw_plus_r3_raw"], 5
        ),
        "r3_over_r2_gain": round(payload["screen"]["gain_r3_over_r2"], 5),
        "multiscale_over_r3_gain": round(payload["screen"]["gain_multiscale_over_r3"], 5),
    }
    print(compact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
