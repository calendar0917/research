"""Assemble the seven-dataset tables, the matched-baseline table and the GPU ledger.

Reads only committed artifacts:

* ``analysis/checkpoint_manifest.json`` (frozen P1 checkpoints, new datasets);
* ``analysis/runs/<dataset>_s0/{train.json,test_eval.json}`` (new datasets);
* ``lcrc_molnet_benchmark_pilot_v1`` reports, read-only, for the four exposed
  datasets (their checkpoints are not re-trained and their Test is not re-read);
* ``analysis/baseline/baseline_manifest.json`` + ``<dataset>.json`` +
  ``<dataset>_test.json``;
* ``analysis/validation_tuning_probe.csv`` and ``analysis/literature_matrix.csv``.

Writes:

* ``analysis/seven_dataset_table.csv``  -- the headline table (LCRC vs matched XGB)
* ``analysis/matched_baseline.csv``     -- same-row comparison with the split tier
* ``analysis/benchmark_table.csv``      -- LCRC only, with the exposure flag
* ``analysis/gpu_ledger.json``          -- GPU wall clock, charged per run
* ``analysis/summary.json``             -- machine-readable roll-up

A dataset with no one-shot Test evaluation is reported as ``NO_TEST_EVAL`` rather
than with a fabricated number.

Usage
-----
    uv run python .../code/mb2_tables.py
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path
from typing import Any, Sequence

import mb2_common as C

SEVEN_COLUMNS = [
    "dataset",
    "split_method",
    "test_exposure",
    "train_seed",
    "n_train",
    "n_valid",
    "n_test",
    "primary_metric",
    "units",
    "lcrc_valid",
    "lcrc_test",
    "lcrc_test_secondary_name",
    "lcrc_test_secondary_value",
    "xgb_valid",
    "xgb_test",
    "lcrc_minus_xgb",
    "lcrc_relative_gain_pct",
    "better_on_this_split",
    "same_rows",
    "split_match_tier",
    "literature_best_reference",
    "literature_grade",
    "literature_exact_match",
    "lcrc_gpu_hours",
    "xgb_cpu_seconds",
    "lcrc_checkpoint_sha256",
    "status",
]

MATCHED_COLUMNS = [
    "dataset",
    "primary_metric",
    "test_exposure",
    "n_test",
    "lcrc_test",
    "xgb_test",
    "lcrc_minus_xgb",
    "lcrc_relative_gain_pct",
    "lcrc_valid",
    "xgb_valid",
    "lcrc_minus_xgb_valid",
    "same_test_rows",
    "split_match_tier",
    "comparison_strength",
    "note",
]

BENCH_COLUMNS = [
    "dataset",
    "task",
    "split_method",
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
    "gpu_hours",
    "peak_memory_allocated_bytes",
    "test_exposure",
    "status",
]


def _load(path: Path) -> Any | None:
    return C.read_json(path) if path.exists() else None


def _git_commit_for(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", C.track_rel(path)],
            cwd=C.REPO,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # pragma: no cover
        return ""


def _metric(task: str, metrics: dict[str, Any]) -> float:
    if task == "regression":
        return float(metrics["rmse"])
    if task == "multitask":
        return float(metrics["macro_roc_auc"])
    return float(metrics["roc_auc"])


def _secondary(task: str, metrics: dict[str, Any]) -> tuple[str, float | None]:
    if task == "regression":
        return "mae", float(metrics["mae"])
    if task == "multitask":
        return "macro_pr_auc", float(metrics["macro_pr_auc"])
    return "pr_auc", float(metrics["pr_auc"])


def _lcrc_record(dataset: str) -> dict[str, Any] | None:
    """LCRC Train/Validation/Test record for one dataset, from the frozen artifacts."""
    if dataset in C.EXPOSED_TEST_DATASETS:
        run_dir = C.OLD_V1_RUNS / f"{dataset}_s0"
        train_path = run_dir / "train.json"
        test_path = run_dir / "test_eval.json"
        if not train_path.exists() or not test_path.exists():
            return None
        train = C.read_json(train_path)
        test = C.read_json(test_path)
        checkpoint_path = C.REPO / train["checkpoint"]["path"]
        if checkpoint_path.exists() and C.sha256_file(checkpoint_path) != test["checkpoint"]["sha256"]:
            raise SystemExit(f"{dataset}: Sprint-01 checkpoint sha256 no longer matches its test_eval.json")
        return {
            "task": train["task"],
            "train": _metric(train["task"], train["metrics"]["train"]),
            "valid": _metric(train["task"], train["metrics"]["valid"]),
            "test": _metric(train["task"], test["metrics"]["test"]),
            "secondary": _secondary(train["task"], test["metrics"]["test"]),
            "best_epoch": int(train["checkpoint"]["best_epoch"]),
            "epochs_run": int(train["checkpoint"]["epochs_run"]),
            "parameters": int(train["model"]["parameters"]),
            "gpu_hours": float(train["timing"]["train_wall_seconds"]) / 3600.0,
            "train_wall_seconds": float(train["timing"]["train_wall_seconds"]),
            "process_wall_seconds": float(train["timing"]["process_wall_seconds"]),
            "train_cache_build_seconds": float(train["data"].get("train_cache_build_seconds", 0.0)),
            "peak_memory_allocated_bytes": int(train["gpu"]["peak_memory_allocated_bytes"]),
            "checkpoint_sha256": test["checkpoint"]["sha256"],
            "n_train": int(train["data"]["n_train"]),
            "n_valid": int(train["data"]["n_valid"]),
            "n_test": int(test["test_split"]["n"]),
            "source": "lcrc_molnet_benchmark_pilot_v1 (read-only)",
            "provenance_commit": _git_commit_for(test_path),
            "test_exposure": "EXPOSED_TEST_EXPLORATORY",
            "test_eval_path": C.track_rel(test_path),
        }
    run_dir = C.RUNS / f"{dataset}_s{C.TRAIN_SEED}"
    train_path = run_dir / "train.json"
    test_path = run_dir / "test_eval.json"
    if not train_path.exists():
        return None
    train = C.read_json(train_path)
    if not test_path.exists():
        return {
            "task": train["task"],
            "train": _metric(train["task"], train["metrics"]["train"]),
            "valid": _metric(train["task"], train["metrics"]["valid"]),
            "test": None,
            "secondary": ("", None),
            "best_epoch": int(train["checkpoint"]["best_epoch"]),
            "epochs_run": int(train["checkpoint"]["epochs_run"]),
            "parameters": int(train["model"]["parameters"]),
            "gpu_hours": float(train["timing"]["train_wall_seconds"]) / 3600.0,
            "train_wall_seconds": float(train["timing"]["train_wall_seconds"]),
            "process_wall_seconds": float(train["timing"]["process_wall_seconds"]),
            "train_cache_build_seconds": float(train["data"].get("train_cache_build_seconds", 0.0)),
            "peak_memory_allocated_bytes": int(train["gpu"]["peak_memory_allocated_bytes"]),
            "checkpoint_sha256": train["checkpoint"]["sha256"],
            "n_train": int(train["data"]["n_train"]),
            "n_valid": int(train["data"]["n_valid"]),
            "n_test": None,
            "source": "lcrc_molnet_benchmark_pilot_v2",
            "provenance_commit": train["git_commit"],
            "test_exposure": "NOT_YET_READ",
            "test_eval_path": None,
        }
    test = C.read_json(test_path)
    return {
        "task": train["task"],
        "train": _metric(train["task"], train["metrics"]["train"]),
        "valid": _metric(train["task"], train["metrics"]["valid"]),
        "test": _metric(train["task"], test["metrics"]["test"]),
        "secondary": _secondary(train["task"], test["metrics"]["test"]),
        "best_epoch": int(train["checkpoint"]["best_epoch"]),
        "epochs_run": int(train["checkpoint"]["epochs_run"]),
        "parameters": int(train["model"]["parameters"]),
        "gpu_hours": float(train["timing"]["train_wall_seconds"]) / 3600.0,
        "train_wall_seconds": float(train["timing"]["train_wall_seconds"]),
        "process_wall_seconds": float(train["timing"]["process_wall_seconds"]),
        "train_cache_build_seconds": float(train["data"].get("train_cache_build_seconds", 0.0)),
        "peak_memory_allocated_bytes": int(train["gpu"]["peak_memory_allocated_bytes"]),
        "checkpoint_sha256": test["checkpoint"]["sha256"],
        "n_train": int(train["data"]["n_train"]),
        "n_valid": int(train["data"]["n_valid"]),
        "n_test": int(test["test_split"]["n"]),
        "source": "lcrc_molnet_benchmark_pilot_v2",
        "provenance_commit": test["git_commit"],
        "test_exposure": "FIRST_TEST_READ",
        "test_eval_path": C.track_rel(test_path),
    }


def _baseline_record(dataset: str) -> dict[str, Any] | None:
    report = _load(C.BASELINE_DIR / f"{dataset}.json")
    if report is None:
        return None
    test = _load(C.BASELINE_DIR / f"{dataset}_test.json")
    task = report["task"]
    out = {
        "task": task,
        "valid": _metric(task, report["metrics"]["valid"]),
        "test": _metric(task, test["metrics"]["test"]) if test else None,
        "fit_seconds": float(report["fit_seconds"]),
        "report": C.track_rel(C.BASELINE_DIR / f"{dataset}.json"),
        "report_sha256": C.sha256_file(C.BASELINE_DIR / f"{dataset}.json"),
        "n_used_columns": int(report["preprocessing"]["n_used_columns"]),
        "n_dropped_columns": int(report["preprocessing"]["n_dropped_columns"]),
        "test_rows_sha256": test["test_rows_sha256"] if test else None,
        "comparison_label": test["comparison_label"] if test else None,
        "test_exposure": (
            "EXPOSED_TEST_EXPLORATORY" if dataset in C.EXPOSED_TEST_DATASETS else "FIRST_TEST_READ"
        ),
    }
    return out


def _literature_best(dataset: str) -> tuple[str, str, str]:
    path = C.ANALYSIS / "literature_matrix.csv"
    if not path.exists():
        return "", "", ""
    rows = [r for r in csv.DictReader(path.open(encoding="utf-8")) if r["dataset"] == dataset]
    if not rows:
        return "", "", ""
    rank = {"A": 0, "B": 1, "C": 2}
    rows.sort(key=lambda r: (rank[r["grade"]], -float(r["value"]) if r["metric"] == "ROC-AUC" else float(r["value"])))
    best = rows[0]
    return (
        f"{best['model']}={best['value']} ({best['metric']}, {best['split_method']})",
        best["grade"],
        best["grade_meaning"],
    )


def _significance_note(dataset: str) -> str:
    if dataset in C.EXPOSED_TEST_DATASETS:
        return (
            "EXPOSED_TEST_EXPLORATORY: the Test labels were already read in Sprint 01, so this matched "
            "comparison is descriptive only and was not used to change any configuration"
        )
    return (
        "FIRST_TEST_READ: LCRC and the baseline were both frozen before the single Test read; single seed, "
        "so a difference of one or two points is not a significance claim"
    )


def build() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    manifest = _load(C.ANALYSIS / "checkpoint_manifest.json")
    baseline_manifest = _load(C.ANALYSIS / "baseline_manifest.json")
    seven: list[dict[str, Any]] = []
    matched: list[dict[str, Any]] = []
    bench: list[dict[str, Any]] = []
    ledger_entries: list[dict[str, Any]] = []
    missing: list[str] = []

    for dataset in C.DATASET_ORDER:
        meta = C.DATASETS[dataset]
        lcrc = _lcrc_record(dataset)
        base = _baseline_record(dataset)
        if lcrc is None:
            missing.append(f"{dataset}: no LCRC record")
            seven.append(
                {
                    "dataset": dataset,
                    "split_method": meta["split_method"],
                    "test_exposure": "MISSING",
                    "status": "BLOCKED_NO_FORMAL_RUN",
                    "primary_metric": meta["primary_metric"],
                    "units": meta["units"],
                }
            )
            continue
        task = lcrc["task"]
        lit_value, lit_grade, lit_meaning = _literature_best(dataset)
        delta = None
        relative = None
        better = ""
        if lcrc["test"] is not None and base is not None and base["test"] is not None:
            delta = lcrc["test"] - base["test"]
            tighter = task == "regression"
            better = "LCRC" if (delta < 0) == tighter else "ECFP4+RDKit+XGBoost"
            if base["test"]:
                relative = delta / base["test"] * 100.0
        seven.append(
            {
                "dataset": dataset,
                "split_method": meta["split_method"],
                "test_exposure": lcrc["test_exposure"],
                "train_seed": C.TRAIN_SEED,
                "n_train": lcrc["n_train"],
                "n_valid": lcrc["n_valid"],
                "n_test": lcrc["n_test"],
                "primary_metric": meta["primary_metric"],
                "units": meta["units"],
                "lcrc_valid": lcrc["valid"],
                "lcrc_test": lcrc["test"],
                "lcrc_test_secondary_name": lcrc["secondary"][0],
                "lcrc_test_secondary_value": lcrc["secondary"][1],
                "xgb_valid": base["valid"] if base else None,
                "xgb_test": base["test"] if base else None,
                "lcrc_minus_xgb": delta,
                "lcrc_relative_gain_pct": -relative if relative is not None else None,
                "better_on_this_split": better,
                "same_rows": bool(base and base["test_rows_sha256"]),
                "split_match_tier": "SAME_ROW_IDS (LCRC and ECFP4 baseline trained and scored on identical indices)",
                "literature_best_reference": lit_value,
                "literature_grade": lit_grade,
                "literature_exact_match": "YES" if lit_grade == "A" else "NO_EXACT_MATCH_FOUND",
                "lcrc_gpu_hours": lcrc["gpu_hours"],
                "xgb_cpu_seconds": base["fit_seconds"] if base else None,
                "lcrc_checkpoint_sha256": lcrc["checkpoint_sha256"],
                "status": "COMPLETE" if lcrc["test"] is not None and base and base["test"] is not None else "PARTIAL",
            }
        )
        matched.append(
            {
                "dataset": dataset,
                "primary_metric": meta["primary_metric"],
                "test_exposure": lcrc["test_exposure"],
                "n_test": lcrc["n_test"],
                "lcrc_test": lcrc["test"],
                "xgb_test": base["test"] if base else None,
                "lcrc_minus_xgb": delta,
                "lcrc_relative_gain_pct": -relative if relative is not None else None,
                "lcrc_valid": lcrc["valid"],
                "xgb_valid": base["valid"] if base else None,
                "lcrc_minus_xgb_valid": (lcrc["valid"] - base["valid"]) if base else None,
                "same_test_rows": bool(base and base["test_rows_sha256"]),
                "split_match_tier": "SAME_ROW_IDS",
                "comparison_strength": (
                    "descriptive single-seed, exposed Test" if dataset in C.EXPOSED_TEST_DATASETS
                    else "descriptive single-seed, first Test read, frozen before scoring"
                ),
                "note": _significance_note(dataset),
            }
        )
        bench.append(
            {
                "dataset": dataset,
                "task": task,
                "split_method": meta["split_method"],
                "train_seed": C.TRAIN_SEED,
                "n_train": lcrc["n_train"],
                "n_valid": lcrc["n_valid"],
                "n_test": lcrc["n_test"],
                "primary_metric": meta["primary_metric"],
                "units": meta["units"],
                "train": lcrc["train"],
                "valid": lcrc["valid"],
                "test": lcrc["test"],
                "test_secondary_name": lcrc["secondary"][0],
                "test_secondary_value": lcrc["secondary"][1],
                "best_epoch": lcrc["best_epoch"],
                "epochs_run": lcrc["epochs_run"],
                "parameters": lcrc["parameters"],
                "gpu_hours": lcrc["gpu_hours"],
                "peak_memory_allocated_bytes": lcrc["peak_memory_allocated_bytes"],
                "test_exposure": lcrc["test_exposure"],
                "status": "COMPLETE" if lcrc["test"] is not None else "AWAITING_TEST",
            }
        )
        ledger_entries.append(
            {
                "dataset": dataset,
                "phase": "train",
                "run_dir": lcrc.get("test_eval_path") or "",
                "source": lcrc["source"],
                "provenance_commit": lcrc["provenance_commit"],
                "measured_training_seconds": lcrc["train_wall_seconds"],
                "measured_process_wall_seconds": lcrc["process_wall_seconds"],
                "train_cache_build_seconds": lcrc["train_cache_build_seconds"],
                "charged_gpu_hours": lcrc["gpu_hours"],
                "process_wall_gpu_hours": lcrc["process_wall_seconds"] / 3600.0,
                "peak_memory_allocated_bytes": lcrc["peak_memory_allocated_bytes"],
                "n_train": lcrc["n_train"],
                "epochs_run": lcrc["epochs_run"],
                "charging_rule": "one A100 per run; training-process wall clock from train.json",
            }
        )

    # ---- probe rows (never charged to the main table) --------------------
    probe_path = C.ANALYSIS / "validation_tuning_probe.csv"
    probe_rows = list(csv.DictReader(probe_path.open(encoding="utf-8"))) if probe_path.exists() else []
    for row in probe_rows:
        if row["kind"] != "validation_only_probe":
            continue
        ledger_entries.append(
            {
                "dataset": row["dataset"],
                "phase": "probe_validation_only",
                "run_dir": row["artifact"],
                "source": "lcrc_molnet_benchmark_pilot_v2 (P3)",
                "measured_training_seconds": float(row["train_wall_seconds"]),
                "measured_process_wall_seconds": float(row.get("process_wall_seconds") or row["train_wall_seconds"]),
                "charged_gpu_hours": float(row["gpu_hours"]),
                "process_wall_gpu_hours": float(row.get("process_wall_seconds") or row["train_wall_seconds"]) / 3600.0,
                "peak_memory_allocated_bytes": int(row["peak_memory_allocated_bytes"]),
                "epochs_run": int(row["epochs_run"]),
                "test_evaluated": False,
                "charging_rule": "one A100 per run; training-process wall clock from train.json",
            }
        )

    v2_total = sum(
        e["charged_gpu_hours"] for e in ledger_entries if e["source"].startswith("lcrc_molnet_benchmark_pilot_v2")
    )
    process_total = sum(
        e.get("process_wall_gpu_hours", e["charged_gpu_hours"])
        for e in ledger_entries
        if e["source"].startswith("lcrc_molnet_benchmark_pilot_v2")
    )
    smoke = 0.0
    for run_dir in sorted(C.RUNS.glob("smoke_*")):
        payload = _load(run_dir / "train.json")
        if payload:
            smoke += float(payload["timing"]["train_wall_seconds"]) / 3600.0
    ledger = {
        "schema": "lcrc-molnet-v2-gpu-ledger-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "gpu_hard_cap_hours": C.GPU_HARD_CAP_HOURS,
        "charging_rule": (
            "one A100 per run; charged from the training process wall clock in train.json; CPU baseline "
            "and CPU Test evaluation are charged 0 GPU-hours and reported separately"
        ),
        "derived_from": {
            "checkpoint_manifest": (
                C.track_rel(C.ANALYSIS / "checkpoint_manifest.json")
                if (C.ANALYSIS / "checkpoint_manifest.json").exists()
                else None
            ),
            "checkpoint_manifest_sha256": (
                C.sha256_file(C.ANALYSIS / "checkpoint_manifest.json")
                if (C.ANALYSIS / "checkpoint_manifest.json").exists()
                else None
            ),
            "baseline_manifest": (
                C.track_rel(C.ANALYSIS / "baseline_manifest.json")
                if (C.ANALYSIS / "baseline_manifest.json").exists()
                else None
            ),
            "probe_table": C.track_rel(probe_path) if probe_path.exists() else None,
            "sprint01_checkpoint_manifest": C.track_rel(C.V1 / "analysis/checkpoint_manifest.json"),
        },
        "entries": ledger_entries,
        "total": {
            "n_runs_this_round": sum(
                1 for e in ledger_entries if e["source"].startswith("lcrc_molnet_benchmark_pilot_v2")
            ),
            "gpu_hours_this_round": v2_total,
            "fraction_of_cap": v2_total / C.GPU_HARD_CAP_HOURS,
            "within_cap": v2_total <= C.GPU_HARD_CAP_HOURS,
            "gpu_hours_this_round_process_wall": process_total,
            "fraction_of_cap_process_wall": process_total / C.GPU_HARD_CAP_HOURS,
            "within_cap_process_wall": process_total <= C.GPU_HARD_CAP_HOURS,
            "cap_accounting_note": (
                "the frozen charging rule charges the training loop wall clock; the process wall clock "
                "(which also contains the label-free structure-cache build and the final predictions) is "
                "reported alongside, and the cap holds under both accountings"
            ),
            "engineering_smoke_gpu_hours": smoke,
            "smoke_note": (
                "2-epoch engineering smokes are not results; they are reported separately and are not "
                "counted in the formal per-dataset ledger entries"
            ),
            "cpu_baseline_seconds_total": sum(e["fit_seconds"] for e in (_baseline_record(d) or {"fit_seconds": 0} for d in C.DATASET_ORDER)),
            "sprint01_gpu_hours_read_only": sum(
                e["charged_gpu_hours"]
                for e in ledger_entries
                if e["source"].startswith("lcrc_molnet_benchmark_pilot_v1")
            ),
        },
    }
    summary = {
        "schema": "lcrc-molnet-v2-summary-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "train_seed": C.TRAIN_SEED,
        "split_policy": "ChemBench repeat 0 (suffix 122) for six datasets; official OGB scaffold split for ogbg-molhiv",
        "p0_status": (_load(C.ANALYSIS / "p0_audit.json") or {}).get("status"),
        "protocol_freeze_sha256": (
            C.sha256_file(C.ANALYSIS / "protocol_freeze.json")
            if (C.ANALYSIS / "protocol_freeze.json").exists()
            else None
        ),
        "checkpoint_manifest_present": manifest is not None,
        "baseline_manifest_present": baseline_manifest is not None,
        "missing": missing,
        "seven_dataset_rows": seven,
        "probe_rows": len(probe_rows),
        "statistics": {
            "n_seeds": 1,
            "confidence_intervals": False,
            "significance_tests": None,
            "note": (
                "every LCRC and baseline number is a single-seed point estimate on a public split; "
                "differences of one or two points are not interpreted as significant"
            ),
        },
    }
    return seven, matched, bench, ledger, summary


def write_csv(rows: list[dict[str, Any]], columns: Sequence[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="assemble the seven-dataset tables")
    parser.parse_args(argv)
    if not (C.ANALYSIS / "baseline_manifest.json").exists():
        C.log("warning: baseline_manifest.json missing - the matched columns will be empty")
    seven, matched, bench, ledger, summary = build()
    write_csv(seven, SEVEN_COLUMNS, C.ANALYSIS / "seven_dataset_table.csv")
    write_csv(matched, MATCHED_COLUMNS, C.ANALYSIS / "matched_baseline.csv")
    write_csv(bench, BENCH_COLUMNS, C.ANALYSIS / "benchmark_table.csv")
    C.write_json(C.ANALYSIS / "gpu_ledger.json", ledger)
    C.write_json(C.ANALYSIS / "summary.json", summary)
    C.log(
        f"seven_dataset_table.csv: {sum(1 for r in seven if r.get('status') == 'COMPLETE')}/7 complete; "
        f"GPU {ledger['total']['gpu_hours_this_round']:.4f} h of {C.GPU_HARD_CAP_HOURS} h"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
