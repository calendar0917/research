"""Aggregate the 12 formal runs into ``results.csv`` and the comparison table.

Every row is read from the recorded per-run artefacts; nothing is recomputed
from a fresh forward pass.  Integrity checks refuse a silently inconsistent run
(checkpoint hash, Test-label alignment, best-epoch re-derivation, test access
after selection).
"""

from __future__ import annotations

import argparse
import csv
from typing import Any, Mapping, Sequence

import numpy as np

import tu_common as C

RESULTS_CSV = C.TRACK / "results.csv"
SUMMARY_JSON = C.ANALYSIS / "summary.json"
COMPARISON_MD = C.ANALYSIS / "comparison.md"

CSV_COLUMNS = [
    "dataset",
    "model",
    "model_label",
    "seed",
    "split_seed",
    "n_train",
    "n_valid",
    "n_test",
    "train_accuracy",
    "valid_accuracy",
    "test_accuracy",
    "test_macro_f1",
    "test_cross_entropy",
    "train_cross_entropy",
    "valid_cross_entropy",
    "best_epoch",
    "epochs_run",
    "train_seconds",
    "preprocessing_seconds",
    "parameters",
    "peak_cuda_allocated_bytes",
    "peak_cuda_reserved_bytes",
    "checkpoint_sha256",
    "result_label",
]


def _load_run(dataset: str, model: str) -> dict[str, Any]:
    path = C.RUNS / f"{dataset}_{model}_s{C.TRAINING_SEED}" / "run.json"
    if not path.exists():
        raise FileNotFoundError(f"missing formal run record {C.track_rel(path)}")
    return C.read_json(path)


def _check_run(payload: Mapping[str, Any], dataset: str, model: str) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    run_dir = C.RUNS / f"{dataset}_{model}_s{C.TRAINING_SEED}"
    selection = C.read_json(run_dir / "selection.json")
    checks["status_completed"] = str(payload["status"]) == "completed"
    checks["seed_zero"] = int(payload["seed"]) == C.TRAINING_SEED
    checks["split_seed_zero"] = int(payload["split_seed"]) == C.SPLIT_SEED
    checks["checkpoint_hash_matches_selection"] = bool(
        selection["checkpoint_sha256"] == payload["checkpoint"]["sha256"]
    )
    checks["checkpoint_file_hash"] = bool(
        C.sha256_file(C.REPO / payload["checkpoint"]["path"]) == payload["checkpoint"]["sha256"]
    )
    checks["test_evaluated_after_selection"] = bool(selection.get("test_evaluated")) and bool(
        payload["test_metrics"] is not None
    )
    curve_path = C.REPO / payload["curve"]["path"]
    with open(curve_path, newline="", encoding="utf-8") as fh:
        curve = list(csv.DictReader(fh))
    best = None
    for row in curve:
        key = (float(row["valid_accuracy"]), -float(row["valid_ce"]), -int(row["epoch"]))
        if best is None or key > best[0]:
            best = (key, int(row["epoch"]))
    checks["best_epoch_matches_curve"] = bool(best is not None and best[1] == int(payload["best_epoch"]))
    checks["curve_rows_equal_epochs"] = len(curve) == int(payload["epochs_run"])
    pred = np.load(C.REPO / payload["predictions"]["test"])
    y_test = np.load(C.ANALYSIS / f"{dataset}_y_test.npy")
    checks["test_prediction_rows"] = int(pred["y_true"].size) == int(y_test.size) == int(payload["n_test"])
    checks["test_labels_aligned"] = bool(np.array_equal(pred["y_true"], y_test))
    checks["test_gids_unique_complete"] = bool(
        np.unique(pred["gids"]).size == pred["gids"].size == int(payload["n_test"])
    )
    checks["reload_valid_recheck"] = bool(
        payload["valid_recheck"]["accuracy_abs_diff"] <= 1e-12
        and payload["valid_recheck"]["ce_abs_diff"] <= 1e-6
    )
    checks["deterministic_seeded"] = bool(payload["determinism"]["torch_seed"] == C.TRAINING_SEED)
    return {"checks": checks, "all_ok": bool(all(checks.values()))}


def _majority_baseline(dataset: str) -> dict[str, Any]:
    y_train = np.load(C.ANALYSIS / f"{dataset}_y_train.npy")
    y_test = np.load(C.ANALYSIS / f"{dataset}_y_test.npy")
    n_classes = int(C.DATASETS[dataset]["n_classes"])
    counts = np.bincount(y_train, minlength=n_classes)
    cls = int(np.argmax(counts))
    return {
        "dataset": dataset,
        "train_majority_class": cls,
        "train_class_counts": [int(x) for x in counts],
        "test_class_counts": [int(x) for x in np.bincount(y_test, minlength=n_classes)],
        "test_accuracy": float((y_test == cls).mean()),
        "n_test": int(y_test.size),
        "training_cost": "zero (no parameters, no training steps)",
    }


def aggregate() -> dict[str, Any]:
    runs: dict[tuple[str, str], dict[str, Any]] = {}
    integrity: dict[str, Any] = {}
    for dataset, model in C.RUN_MATRIX:
        payload = _load_run(dataset, model)
        check = _check_run(payload, dataset, model)
        integrity[f"{dataset}:{model}"] = check
        if not check["all_ok"]:
            raise RuntimeError(f"integrity check failed for {dataset}:{model}: {check['checks']}")
        runs[(dataset, model)] = payload

    rows = []
    for dataset, model in C.RUN_MATRIX:
        p = runs[(dataset, model)]
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "model_label": p["model_label"],
                "seed": p["seed"],
                "split_seed": p["split_seed"],
                "n_train": p["n_train"],
                "n_valid": p["n_valid"],
                "n_test": p["n_test"],
                "train_accuracy": p["train_metrics_at_best"]["accuracy"],
                "valid_accuracy": p["valid_metrics_at_best"]["accuracy"],
                "test_accuracy": p["test_metrics"]["accuracy"],
                "test_macro_f1": p["test_metrics"]["macro_f1"],
                "test_cross_entropy": p["test_metrics"]["cross_entropy"],
                "train_cross_entropy": p["train_metrics_at_best"]["cross_entropy"],
                "valid_cross_entropy": p["valid_metrics_at_best"]["cross_entropy"],
                "best_epoch": p["best_epoch"],
                "epochs_run": p["epochs_run"],
                "train_seconds": round(float(p["timing"]["training_and_eval_seconds"]), 3),
                "preprocessing_seconds": round(float(p["timing"]["preprocessing_seconds"]), 3),
                "parameters": p["parameters"],
                "peak_cuda_allocated_bytes": p["memory"]["peak_cuda_allocated_bytes"],
                "peak_cuda_reserved_bytes": p["memory"]["peak_cuda_reserved_bytes"],
                "checkpoint_sha256": p["checkpoint"]["sha256"],
                "result_label": C.RESULT_LABEL,
            }
        )
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    baselines = {ds: _majority_baseline(ds) for ds in C.DATASETS}
    comparison = {}
    for ds in C.DATASETS:
        comparison[ds] = {
            "majority_baseline": baselines[ds],
            "test_accuracy": {m: runs[(ds, m)]["test_metrics"]["accuracy"] for m in C.MODEL_NAMES},
            "test_macro_f1": {m: runs[(ds, m)]["test_metrics"]["macro_f1"] for m in C.MODEL_NAMES},
            "test_cross_entropy": {m: runs[(ds, m)]["test_metrics"]["cross_entropy"] for m in C.MODEL_NAMES},
            "best_epoch": {m: runs[(ds, m)]["best_epoch"] for m in C.MODEL_NAMES},
            "parameters": {m: runs[(ds, m)]["parameters"] for m in C.MODEL_NAMES},
            "confusion_matrix": {m: runs[(ds, m)]["test_metrics"]["confusion_matrix"] for m in C.MODEL_NAMES},
            "valid_accuracy": {m: runs[(ds, m)]["valid_metrics_at_best"]["accuracy"] for m in C.MODEL_NAMES},
            "train_accuracy": {m: runs[(ds, m)]["train_metrics_at_best"]["accuracy"] for m in C.MODEL_NAMES},
        }

    prepped = C.read_json(C.ANALYSIS / "prep_summary.json")
    payload = {
        "round_id": C.ROUND_ID,
        "result_label": C.RESULT_LABEL,
        "created_utc": C.utcnow(),
        "n_runs": len(rows),
        "run_matrix": [{"dataset": d, "model": m} for d, m in C.RUN_MATRIX],
        "seed": C.TRAINING_SEED,
        "split_seed": C.SPLIT_SEED,
        "recipe": C.RECIPE,
        "model_config": C.MODEL_CONFIG,
        "gin_config": C.GIN_CONFIG,
        "prep": prepped,
        "rows": rows,
        "comparison": comparison,
        "integrity": integrity,
        "results_csv": C.track_rel(RESULTS_CSV),
        "results_csv_sha256": C.sha256_file(RESULTS_CSV),
    }
    C.write_json(SUMMARY_JSON, payload)

    lines = ["# FULL / C1 / GIN Test accuracy by dataset", ""]
    lines.append("| dataset | majority | FULL | C1 | GIN | FULL−C1 | C1−GIN |")
    lines.append("|---|---|---|---|---|---|---|")
    for ds in C.DATASETS:
        c = comparison[ds]
        a = c["test_accuracy"]
        lines.append(
            f"| {ds} | {c['majority_baseline']['test_accuracy']:.4f} | "
            f"{a['FULL']:.4f} | {a['C1']:.4f} | {a['GIN']:.4f} | "
            f"{a['FULL'] - a['C1']:+.4f} | {a['C1'] - a['GIN']:+.4f} |"
        )
    lines.append("")
    COMPARISON_MD.parent.mkdir(parents=True, exist_ok=True)
    COMPARISON_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    payload = aggregate()
    print(C.stable_json({"n_runs": payload["n_runs"], "results_csv": payload["results_csv"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
