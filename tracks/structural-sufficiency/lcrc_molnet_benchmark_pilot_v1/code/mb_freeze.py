"""Freeze every successful P1 run into one manifest, *before* any Test access.

The frozen manifest is the gate for P2: ``mb_eval.py`` refuses to score a Test
split whose checkpoint is not listed here with a matching sha256.  It records,
per dataset:

* the checkpoint path, sha256, best epoch and best Validation metric;
* the training-run manifest (``train.json``) and its sha256, plus the curve and
  prediction artifacts;
* the frozen data identities (split-file hash, split row hashes, train/valid
  structure-cache fingerprints, train-only normalisers, target normaliser);
* the model identity (parameter count, feature-spec hash, initial state hash);
* the code revision and per-module hashes.

Nothing in this step reads a Test label.

Usage
-----
    uv run python .../code/mb_freeze.py [--datasets ESOL,Lipophilicity,...]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Sequence

import mb_common as C


def run_dir(dataset: str, seed: int = C.TRAIN_SEED) -> Path:
    return C.RUNS / f"{dataset}_s{seed}"


def freeze_dataset(dataset: str) -> dict[str, Any]:
    d = run_dir(dataset)
    train_json = d / "train.json"
    checkpoint = d / "checkpoint.pt"
    curve = d / "curve.csv"
    if not train_json.exists():
        raise FileNotFoundError(f"{C.track_rel(train_json)} is missing; run P1 first")
    payload = C.read_json(train_json)
    if payload.get("smoke"):
        raise ValueError(f"{C.track_rel(train_json)} is a smoke run and cannot be frozen")
    if payload["data"].get("test_labels_read") or payload["data"].get("test_cache_built"):
        raise PermissionError(f"{C.track_rel(train_json)} reports Test access; refusing to freeze")
    for path in (checkpoint, curve, d / "train_predictions.npz", d / "valid_predictions.npz"):
        if not path.exists():
            raise FileNotFoundError(path)
    recorded = payload["checkpoint"]["sha256"]
    actual = C.sha256_file(checkpoint)
    if recorded != actual:
        raise ValueError(f"{dataset}: checkpoint sha256 {actual} != recorded {recorded}")
    if payload["recipe"]["epochs_run"] != int(C.RECIPE["epochs"]):
        raise ValueError(f"{dataset}: {payload['recipe']['epochs_run']} epochs != frozen {C.RECIPE['epochs']}")
    if int(payload["seed"]) != C.TRAIN_SEED:
        raise ValueError(f"{dataset}: training seed {payload['seed']} != {C.TRAIN_SEED}")
    if int(payload["split_repeat_index"]) != C.SPLIT_REPEAT_INDEX:
        raise ValueError(f"{dataset}: split repeat {payload['split_repeat_index']} != {C.SPLIT_REPEAT_INDEX}")
    gpu_seconds = float(payload["timing"]["train_wall_seconds"])
    return {
        "dataset": dataset,
        "run_dir": C.track_rel(d),
        "task": payload["task"],
        "label_column": payload["label_column"],
        "units": payload["units"],
        "primary_metric": C.DATASETS[dataset]["primary_metric"],
        "seed": int(payload["seed"]),
        "split_repeat_index": int(payload["split_repeat_index"]),
        "split_file": payload["split_file"],
        "split_file_sha256": payload["split_file_sha256"],
        "split_rows_sha256": payload["split_rows_sha256"],
        "feature_spec_hash": payload["data"]["feature_spec_hash"],
        "parameters": int(payload["model"]["parameters"]),
        "model_definition": "LCRC-A (one-shot local content + per-atom contribution + global rho)",
        "iterative_hidden_message_passing": False,
        "cache_fingerprints": {
            "train": payload["data"]["train_cache_fingerprint"],
            "valid": payload["data"]["valid_cache_fingerprint"],
        },
        "train_norm_fingerprint": payload["data"]["train_norm_fingerprint"],
        "target_norm": payload["data"]["target_norm"],
        "data_view": payload["data_view"],
        "checkpoint": {
            "path": payload["checkpoint"]["path"],
            "sha256": actual,
            "best_epoch": int(payload["checkpoint"]["best_epoch"]),
            "best_metric_name": payload["checkpoint"]["best_metric_name"],
            "best_metric_value": float(payload["checkpoint"]["best_metric_value"]),
            "best_state_hash": payload["checkpoint"]["best_state_hash"],
            "init_state_hash": payload["model"]["init_state_hash"],
            "curve_argbest_epoch": int(payload["checkpoint"]["curve_argbest_epoch"]),
            "epochs_run": int(payload["checkpoint"]["epochs_run"]),
            "checkpoint_is_final_epoch_state": bool(payload["checkpoint"]["checkpoint_is_final_epoch_state"]),
            "valid_recheck_abs_diff": float(payload["checkpoint"]["valid_recheck_abs_diff"]),
        },
        "metrics_before_test": {
            "train": payload["metrics"]["train"],
            "valid": payload["metrics"]["valid"],
        },
        "recipe": payload["recipe"],
        "timing": {
            "train_wall_seconds": gpu_seconds,
            "mean_epoch_seconds": float(payload["timing"]["mean_epoch_seconds"]),
            "inference_seconds_per_valid_split": float(payload["timing"]["inference_seconds_per_valid_split"]),
            "process_wall_seconds": float(payload["timing"]["process_wall_seconds"]),
        },
        "gpu": payload["gpu"],
        "git_commit": payload["git_commit"],
        "code_hashes": payload["code_hashes"],
        "artifacts": {
            "train_json": C.track_rel(train_json),
            "train_json_sha256": C.sha256_file(train_json),
            "curve": C.track_rel(curve),
            "curve_sha256": C.sha256_file(curve),
            "checkpoint": C.track_rel(checkpoint),
            "checkpoint_sha256": actual,
            "train_predictions": payload["predictions"]["train"],
            "train_predictions_sha256": payload["predictions"]["train_sha256"],
            "valid_predictions": payload["predictions"]["valid"],
            "valid_predictions_sha256": payload["predictions"]["valid_sha256"],
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze P1 checkpoints before P2 Test evaluation")
    parser.add_argument("--datasets", default=",".join(C.DATASET_ORDER))
    parser.add_argument("--out", default=str(C.ANALYSIS / "checkpoint_manifest.json"))
    args = parser.parse_args(argv)
    keys = [k.strip() for k in args.datasets.split(",") if k.strip()]
    frozen: dict[str, Any] = {}
    missing: list[str] = []
    for key in keys:
        try:
            frozen[key] = freeze_dataset(key)
        except FileNotFoundError as exc:
            missing.append(f"{key}: {exc}")
    if not frozen:
        raise SystemExit("no completed runs to freeze")
    total_gpu_seconds = sum(entry["timing"]["train_wall_seconds"] for entry in frozen.values())
    manifest = {
        "schema": "lcrc-molnet-checkpoint-manifest-v1",
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "train_seed": C.TRAIN_SEED,
        "split_repeat_index": C.SPLIT_REPEAT_INDEX,
        "datasets": frozen,
        "missing_datasets": missing,
        "gpu_ledger_before_test": {
            "n_runs": len(frozen),
            "training_wall_seconds": total_gpu_seconds,
            "gpu_hours": total_gpu_seconds / 3600.0,
            "cap_hours": C.GPU_HARD_CAP_HOURS,
            "charging_rule": "one A100 per run; training-process wall clock from train.json",
        },
        "frozen": True,
        "test_evaluation_rule": (
            "one shot per dataset; mb_eval.py refuses a dataset absent from this manifest or a "
            "checkpoint whose sha256 differs; no Test result may change an epoch, model, split, "
            "learning rate, loss or metric definition"
        ),
        "test_labels_read": False,
    }
    C.ensure_dir(Path(args.out).parent)
    C.write_json(args.out, manifest)
    C.log(
        f"froze {len(frozen)} run(s) -> {C.track_rel(args.out)}; "
        f"training GPU time {manifest['gpu_ledger_before_test']['gpu_hours']:.4f} h"
    )
    if missing:
        C.log("missing runs: " + "; ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
