"""Assemble the benchmark table, the GPU ledger and the round summary.

Reads only committed artifacts:

* ``analysis/checkpoint_manifest.json`` (frozen P1 checkpoints);
* ``analysis/runs/<dataset>_s0/{train.json,test_eval.json}``;
* ``analysis/p0_audit.json``, ``analysis/dataset_manifest.json``;
* optionally ``analysis/rr_jobs.json`` (raw ``rr status --json`` output used for
  the wall-clock ledger; measured GPU time always comes from ``train.json``).

Writes ``analysis/benchmark_table.csv``, ``analysis/gpu_ledger.json`` and
``analysis/summary.json``.  A dataset without a one-shot Test evaluation is
reported with ``status=NO_TEST_EVAL`` rather than a fabricated number.

Usage
-----
    uv run python .../code/mb_tables.py
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import mb_common as C

TABLE_COLUMNS = [
    "dataset",
    "task",
    "split_method",
    "split_repeat_index",
    "train_seed",
    "n_train",
    "n_valid",
    "n_test",
    "primary_metric",
    "units",
    "train",
    "valid",
    "test",
    "test_secondary_name",
    "test_secondary_value",
    "best_epoch",
    "epochs_run",
    "parameters",
    "train_wall_seconds",
    "gpu_hours",
    "peak_memory_allocated_bytes",
    "checkpoint_sha256",
    "train_cache_fingerprint",
    "status",
]


def _load_optional(path: Path) -> Any | None:
    return C.read_json(path) if path.exists() else None


def _secondary(task: str, metrics: dict[str, Any]) -> tuple[str, float | None]:
    if task == "regression":
        return "test_mae", float(metrics["mae"]) if "mae" in metrics else None
    return "test_pr_auc", float(metrics["pr_auc"]) if "pr_auc" in metrics else None


def build() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    manifest_path = C.ANALYSIS / "checkpoint_manifest.json"
    manifest = _load_optional(manifest_path)
    if manifest is None:
        raise SystemExit("checkpoint_manifest.json is missing: freeze P1 before building tables")
    rr_jobs = _load_optional(C.ANALYSIS / "rr_jobs.json") or {}
    p0 = _load_optional(C.ANALYSIS / "p0_audit.json") or {}

    rows: list[dict[str, Any]] = []
    ledger_entries: list[dict[str, Any]] = []
    for dataset in C.DATASET_ORDER:
        run_dir = C.RUNS / f"{dataset}_s{C.TRAIN_SEED}"
        train_json = _load_optional(run_dir / "train.json")
        test_json = _load_optional(run_dir / "test_eval.json")
        frozen = (manifest.get("datasets") or {}).get(dataset)
        split_manifest = _load_optional(C.ANALYSIS / f"split_manifest_{dataset}.json") or {}
        sizes = split_manifest.get("sizes") or {}
        if train_json is None or frozen is None:
            rows.append(
                {
                    "dataset": dataset,
                    "task": C.DATASETS[dataset]["task"],
                    "split_method": C.DATASETS[dataset]["split_method"],
                    "split_repeat_index": C.SPLIT_REPEAT_INDEX,
                    "train_seed": C.TRAIN_SEED,
                    "status": "BLOCKED_NO_FORMAL_RUN",
                }
            )
            continue
        task = train_json["task"]
        primary = C.DATASETS[dataset]["primary_metric"]
        valid_key = "rmse" if task == "regression" else "roc_auc"
        rr = rr_jobs.get(f"{dataset}_s{C.TRAIN_SEED}_train") or {}
        rr_wall = rr.get("wall_seconds") or rr.get("elapsed_seconds")
        train_seconds = float(train_json["timing"]["train_wall_seconds"])
        entry = {
            "dataset": dataset,
            "task": task,
            "phase": "train",
            "run_dir": C.track_rel(run_dir),
            "experiment_name": f"molnet-{dataset.lower()}-s0",
            "rr_wall_seconds": rr_wall,
            "measured_training_seconds": train_seconds,
            "charged_gpu_seconds": train_seconds,
            "charged_gpu_hours": train_seconds / 3600.0,
            "peak_memory_allocated_bytes": int(train_json["gpu"]["peak_memory_allocated_bytes"]),
            "cuda_device_name": train_json["gpu"]["device_name"],
            "git_commit": train_json["git_commit"],
            "hostname": train_json["hostname"],
            "epochs_run": int(train_json["recipe"]["epochs_run"]),
            "parameters": int(train_json["model"]["parameters"]),
            "charging_rule": "one A100 per run; training-process wall clock from train.json",
        }
        if test_json is not None:
            eval_seconds = float(test_json["timing"]["test_seconds"]) + float(test_json["timing"]["valid_seconds"])
            entry["test_eval"] = {
                "measured_inference_seconds": eval_seconds,
                "test_seconds": float(test_json["timing"]["test_seconds"]),
                "test_molecules_per_second": float(test_json["timing"]["test_molecules_per_second"]),
                "test_cache_build_seconds": float(test_json["timing"]["test_cache_build_seconds"]),
                "peak_memory_allocated_bytes": int(test_json["gpu"]["peak_memory_allocated_bytes"]),
                "charged_gpu_seconds": eval_seconds,
                "charged_gpu_hours": eval_seconds / 3600.0,
            }
            entry["charged_gpu_seconds"] += eval_seconds
            entry["charged_gpu_hours"] = entry["charged_gpu_seconds"] / 3600.0
        ledger_entries.append(entry)

        if test_json is None:
            rows.append(
                {
                    "dataset": dataset,
                    "task": task,
                    "split_method": C.DATASETS[dataset]["split_method"],
                    "split_repeat_index": C.SPLIT_REPEAT_INDEX,
                    "train_seed": C.TRAIN_SEED,
                    "n_train": int(frozen["data_view"]["n_train"]),
                    "n_valid": int(frozen["data_view"]["n_valid"]),
                    "n_test": sizes.get("test"),
                    "primary_metric": primary,
                    "units": frozen["units"],
                    "train": float(train_json["metrics"]["train"][valid_key]),
                    "valid": float(train_json["metrics"]["valid"][valid_key]),
                    "test": None,
                    "best_epoch": int(frozen["checkpoint"]["best_epoch"]),
                    "epochs_run": int(frozen["checkpoint"]["epochs_run"]),
                    "parameters": int(frozen["parameters"]),
                    "train_wall_seconds": round(train_seconds, 3),
                    "gpu_hours": round(entry["charged_gpu_hours"], 6),
                    "peak_memory_allocated_bytes": int(train_json["gpu"]["peak_memory_allocated_bytes"]),
                    "checkpoint_sha256": frozen["checkpoint"]["sha256"],
                    "train_cache_fingerprint": frozen["cache_fingerprints"]["train"],
                    "status": "FROZEN_NO_TEST_EVAL",
                }
            )
            continue

        test_metrics = test_json["metrics"]["test"]
        sec_name, sec_value = _secondary(task, test_metrics)
        rows.append(
            {
                "dataset": dataset,
                "task": task,
                "split_method": C.DATASETS[dataset]["split_method"],
                "split_repeat_index": C.SPLIT_REPEAT_INDEX,
                "train_seed": C.TRAIN_SEED,
                "n_train": int(train_json["data"]["n_train"]),
                "n_valid": int(train_json["data"]["n_valid"]),
                "n_test": int(test_json["test_split"]["n"]),
                "primary_metric": primary,
                "units": frozen["units"],
                "train": float(train_json["metrics"]["train"][valid_key]),
                "valid": float(train_json["metrics"]["valid"][valid_key]),
                "test": float(test_metrics[valid_key]),
                "test_secondary_name": sec_name,
                "test_secondary_value": sec_value,
                "best_epoch": int(frozen["checkpoint"]["best_epoch"]),
                "epochs_run": int(frozen["checkpoint"]["epochs_run"]),
                "parameters": int(frozen["parameters"]),
                "train_wall_seconds": round(train_seconds, 3),
                "gpu_hours": round(entry["charged_gpu_hours"], 6),
                "peak_memory_allocated_bytes": int(train_json["gpu"]["peak_memory_allocated_bytes"]),
                "checkpoint_sha256": frozen["checkpoint"]["sha256"],
                "train_cache_fingerprint": frozen["cache_fingerprints"]["train"],
                "status": "COMPLETE",
            }
        )

    total_seconds = sum(e["charged_gpu_seconds"] for e in ledger_entries)
    ledger = {
        "schema": "lcrc-molnet-gpu-ledger-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "gpu_hard_cap_hours": C.GPU_HARD_CAP_HOURS,
        "charging_rule": "one A100 per run; training wall clock from train.json plus inference seconds from test_eval.json",
        "derived_from": {
            "checkpoint_manifest": C.track_rel(manifest_path),
            "checkpoint_manifest_sha256": C.sha256_file(manifest_path),
            "rr_jobs": C.track_rel(C.ANALYSIS / "rr_jobs.json") if rr_jobs else None,
        },
        "entries": ledger_entries,
        "total": {
            "n_formal_runs": len(ledger_entries),
            "measured_seconds": total_seconds,
            "gpu_hours": total_seconds / 3600.0,
            "fraction_of_cap": (total_seconds / 3600.0) / C.GPU_HARD_CAP_HOURS,
            "within_cap": (total_seconds / 3600.0) <= C.GPU_HARD_CAP_HOURS,
            "smoke_gpu_seconds": 0.0,
            "smoke_note": "engineering smoke runs used 2 epochs; their time is recorded separately in summary.json",
        },
    }
    summary = {
        "schema": "lcrc-molnet-summary-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "train_seed": C.TRAIN_SEED,
        "split_repeat_index": C.SPLIT_REPEAT_INDEX,
        "source_commit": C.read_json(C.SOURCE_MANIFEST)["upstream"]["commit"],
        "p0_status": p0.get("status"),
        "p0_sha256": C.sha256_file(C.ANALYSIS / "p0_audit.json") if (C.ANALYSIS / "p0_audit.json").exists() else None,
        "feature_spec_hash": C.read_json(C.ANALYSIS / "dataset_manifest.json")["feature_spec"]["hash"],
        "rows": rows,
    }
    return rows, ledger, summary


def write_table(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    rows, ledger, summary = build()
    write_table(rows, C.ANALYSIS / "benchmark_table.csv")
    C.write_json(C.ANALYSIS / "gpu_ledger.json", ledger)
    C.write_json(C.ANALYSIS / "summary.json", summary)
    C.log(
        f"benchmark_table.csv: {sum(1 for r in rows if r.get('status') == 'COMPLETE')} complete row(s); "
        f"GPU {ledger['total']['gpu_hours']:.4f} h of {C.GPU_HARD_CAP_HOURS} h cap"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
