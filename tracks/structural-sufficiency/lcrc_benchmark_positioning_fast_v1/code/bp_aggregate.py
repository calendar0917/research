"""Aggregate completed runs into the positioning tables and resource ledger."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import bp_common as C
import bp_metrics as MET


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return C.read_json(path)
    except Exception:
        return None


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fields is None:
        fields = []
        for row in rows:
            for k in row:
                if k not in fields:
                    fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    vals = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if not vals.size:
        return None, None
    mean = float(vals.mean())
    std = float(vals.std(ddof=1)) if vals.size > 1 else None
    return mean, std


def _complete(run: dict[str, Any] | None) -> bool:
    return bool(run and run.get("status") == "complete")


def main() -> int:
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    tu_folds: list[dict[str, Any]] = []
    mol_seeds: list[dict[str, Any]] = []
    peptide_seeds: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    recheck: list[dict[str, Any]] = []
    all_tasks: list[dict[str, Any]] = []
    planned: list[dict[str, Any]] = []
    plan_path = C.ANALYSIS / "formal_plan.json"
    if plan_path.is_file():
        planned = C.read_json(plan_path)["order"]
    else:
        planned = []
        for ds in C.TU_DATASETS:
            for f in range(C.TU_N_SPLITS):
                planned.append(C.plan_entry("tu", ds, f))
        for ds in C.MOL_DATASETS:
            for s in C.MOL_SEEDS:
                planned.append(C.plan_entry("mol", ds, s))
        for s in C.PEPTIDE_SEEDS:
            planned.append(C.plan_entry("peptide", C.PEPTIDE_DATASET, s))
    blocked: dict[str, dict[str, Any]] = {}
    for ds in C.TU_DATASETS:
        payload = _read_json(C.ANALYSIS / f"{ds}_blocked.json")
        if payload:
            blocked[ds] = payload
    for entry in planned:
        kind, dataset, unit = entry["kind"], entry["dataset"], int(entry["unit"])
        run_dir = C.result_paths(kind, dataset, unit)
        run = _read_json(run_dir / "run.json")
        blocked_payload = blocked.get(dataset)
        status = (run or {}).get("status", "NOT_RUN")
        if run is None and blocked_payload:
            status = str(blocked_payload.get("status", "RESOURCE_BLOCKED"))
        task = {
            "kind": kind,
            "dataset": dataset,
            "unit": unit,
            "task_id": entry["task_id"],
            "status": status,
            "run_json": C.track_rel(run_dir / "run.json") if (run_dir / "run.json").is_file() else "",
            "reason": (run or {}).get("reason", ""),
            "detail": (run or {}).get("detail", ""),
        }
        all_tasks.append(task)
        if run is None:
            if blocked_payload:
                task["reason"] = str(blocked_payload.get("reason", "PREPROCESSING_BLOCKED"))
                task["detail"] = str(blocked_payload.get("detail", ""))
                task["source"] = str(blocked_payload.get("counts_file", ""))
                failures.append({**task, "status": status})
            else:
                failures.append({**task, "reason": "NOT_RUN_NO_RUN_JSON"})
            continue
        ledger.append(
            {
                "kind": kind,
                "dataset": dataset,
                "unit": unit,
                "task_id": entry["task_id"],
                "status": run.get("status"),
                "best_epoch": run.get("best_epoch"),
                "epochs_run": run.get("epochs_run"),
                "seconds": run.get("seconds_including_test", run.get("seconds")),
                "train_seconds": run.get("train_seconds"),
                "valid_seconds": run.get("valid_seconds"),
                "gpu_peak_allocated_bytes": run.get("gpu_peak_allocated_bytes"),
                "gpu_peak_reserved_bytes": run.get("gpu_peak_reserved_bytes"),
                "parameters": run.get("parameters"),
                "checkpoint_sha256": run.get("checkpoint_sha256"),
                "seed": run.get("seed"),
                "batch_size": run.get("batch_size"),
                "reason": run.get("reason"),
                "detail": run.get("detail"),
            }
        )
        if run.get("status") != "complete":
            failures.append(
                {
                    "kind": kind,
                    "dataset": dataset,
                    "unit": unit,
                    "task_id": entry["task_id"],
                    "status": run.get("status"),
                    "reason": run.get("reason", ""),
                    "detail": run.get("detail", ""),
                    "run_json": C.track_rel(run_dir / "run.json"),
                }
            )
            continue
        tm = run.get("test_metrics", {})
        if kind == "tu":
            tu_folds.append(
                {
                    "dataset": dataset,
                    "fold": unit,
                    "seed": run.get("seed"),
                    "accuracy": float(tm["accuracy"]),
                    "macro_f1": float(tm["macro_f1"]),
                    "cross_entropy": float(tm["cross_entropy"]),
                    "n_test": int(tm.get("n", 0)),
                    "best_epoch": run.get("best_epoch"),
                    "epochs_run": run.get("epochs_run"),
                    "checkpoint_sha256": run.get("checkpoint_sha256"),
                }
            )
        elif kind == "mol":
            mol_seeds.append(
                {
                    "dataset": dataset,
                    "seed": unit,
                    "roc_auc": float(tm["value"]),
                    "metric_source": tm.get("source", ""),
                    "official_evaluator": bool(tm.get("official", False)),
                    "n_test": int(tm.get("n", 0)),
                    "best_epoch": run.get("best_epoch"),
                    "epochs_run": run.get("epochs_run"),
                    "checkpoint_sha256": run.get("checkpoint_sha256"),
                }
            )
        elif kind == "peptide":
            peptide_seeds.append(
                {
                    "dataset": dataset,
                    "seed": unit,
                    "mean_ap": float(tm["value"]),
                    "n_test": int(tm.get("n", 0)),
                    "best_epoch": run.get("best_epoch"),
                    "epochs_run": run.get("epochs_run"),
                    "checkpoint_sha256": run.get("checkpoint_sha256"),
                }
            )
        # Independent recomputation from the saved predictions.
        pred_path = run_dir / "test_predictions.npz"
        if pred_path.is_file():
            z = np.load(pred_path)
            if kind == "tu":
                rec = MET.classification_metrics(z["logits"], z["y_true"].astype(np.int64), int(np.max(z["y_true"])) + 1)
                recheck.append(
                    {
                        "task_id": entry["task_id"],
                        "metric": "accuracy",
                        "stored": tm.get("accuracy"),
                        "recomputed": rec["accuracy"],
                        "delta": float(rec["accuracy"] - float(tm["accuracy"])),
                    }
                )
            elif kind == "mol":
                rec = MET.ogb_rocauc(dataset, z["y_true"].astype(np.float64), z["probs"].astype(np.float64))
                recheck.append(
                    {
                        "task_id": entry["task_id"],
                        "metric": "roc_auc",
                        "stored": tm.get("value"),
                        "recomputed": rec["value"],
                        "delta": float(rec["value"] - float(tm["value"])),
                    }
                )
            elif kind == "peptide":
                rec = MET.macro_ap(z["y_true"].astype(np.float64), z["probs"].astype(np.float64))
                recheck.append(
                    {
                        "task_id": entry["task_id"],
                        "metric": "mean_ap",
                        "stored": tm.get("value"),
                        "recomputed": rec["value"],
                        "delta": float(rec["value"] - float(tm["value"])),
                    }
                )
    _write_csv(C.RESULTS / "tu_folds.csv", tu_folds)
    _write_csv(C.RESULTS / "mol_seeds.csv", mol_seeds)
    _write_csv(C.RESULTS / "peptide_seeds.csv", peptide_seeds)
    _write_csv(C.RESULTS / "resource_ledger.csv", ledger)
    _write_csv(C.RESULTS / "failure_list.csv", failures)
    _write_csv(C.RESULTS / "metric_recheck.csv", recheck)
    _write_csv(C.RESULTS / "run_manifest.csv", all_tasks)

    tu_summary: list[dict[str, Any]] = []
    for ds in C.TU_DATASETS:
        rows = [r for r in tu_folds if r["dataset"] == ds]
        acc = [float(r["accuracy"]) for r in rows]
        f1 = [float(r["macro_f1"]) for r in rows]
        mean, std = _mean_std(acc)
        f1_mean, f1_std = _mean_std(f1)
        tu_summary.append(
            {
                "dataset": ds,
                "n_folds_complete": len(rows),
                "n_folds_required": C.TU_N_SPLITS,
                "accuracy_mean_pct": None if mean is None else 100.0 * mean,
                "accuracy_std_pct": None if std is None else 100.0 * std,
                "macro_f1_mean_pct": None if f1_mean is None else 100.0 * f1_mean,
                "macro_f1_std_pct": None if f1_std is None else 100.0 * f1_std,
                "complete": len(rows) == C.TU_N_SPLITS,
            }
        )
    _write_csv(C.RESULTS / "tu_summary.csv", tu_summary)

    mol_summary: list[dict[str, Any]] = []
    for ds in list(C.MOL_DATASETS) + [C.PEPTIDE_DATASET]:
        if ds == C.PEPTIDE_DATASET:
            rows = [r for r in peptide_seeds if r["dataset"] == ds]
            vals = [float(r["mean_ap"]) for r in rows]
            metric = "mean_ap"
            required = len(C.PEPTIDE_SEEDS)
        else:
            rows = [r for r in mol_seeds if r["dataset"] == ds]
            vals = [float(r["roc_auc"]) for r in rows]
            metric = "roc_auc"
            required = len(C.MOL_SEEDS)
        mean, std = _mean_std(vals)
        mol_summary.append(
            {
                "dataset": ds,
                "metric": metric,
                "n_seeds_complete": len(rows),
                "n_seeds_required": required,
                "mean_pct": None if mean is None else 100.0 * mean,
                "std_pct": None if std is None else 100.0 * std,
                "complete": len(rows) == required,
            }
        )
    _write_csv(C.RESULTS / "molecule_summary.csv", mol_summary)

    pos_tu: list[dict[str, Any]] = []
    for row in tu_summary:
        ds = row["dataset"]
        ref = C.REFERENCE[ds]
        ours = row["accuracy_mean_pct"]
        pos_tu.append(
            {
                "dataset": ds,
                "screenshot_reference_pct": ref,
                "lcrc_c1_mean_accuracy_pct": "NOT_RUN" if ours is None else ours,
                "cross_fold_std_pct": "NOT_RUN" if row["accuracy_std_pct"] is None else row["accuracy_std_pct"],
                "delta_pct": "NOT_RUN" if ours is None else float(ours - ref),
                "n_folds": row["n_folds_complete"],
                "protocol_note": C.SCREENSHOT_LABEL,
            }
        )
    _write_csv(C.RESULTS / "positioning_tu.csv", pos_tu)

    pos_mol: list[dict[str, Any]] = []
    for row in mol_summary:
        ds = row["dataset"]
        ref = C.REFERENCE[ds]
        ours = row["mean_pct"]
        pos_mol.append(
            {
                "dataset": ds,
                "metric": row["metric"],
                "screenshot_reference_pct": ref,
                "lcrc_c1_mean_pct": "NOT_RUN" if ours is None else ours,
                "across_seed_std_pct": "NOT_RUN" if row["std_pct"] is None else row["std_pct"],
                "delta_pct": "NOT_RUN" if ours is None else float(ours - ref),
                "n_seeds": row["n_seeds_complete"],
                "protocol_note": C.SCREENSHOT_LABEL,
            }
        )
    _write_csv(C.RESULTS / "positioning_mol.csv", pos_mol)

    budget = _read_json(C.ANALYSIS / "budget_ledger.json") or {}
    workers = []
    for p in sorted(C.ANALYSIS.glob("worker_*.json")):
        payload = _read_json(p)
        if payload:
            workers.append(
                {
                    "worker": payload.get("worker"),
                    "gpu_visible": payload.get("gpu_visible"),
                    "gpu_name": payload.get("gpu_name"),
                    "wall_seconds": payload.get("wall_seconds"),
                    "n_started": payload.get("n_started"),
                    "started_epoch": payload.get("started_epoch"),
                    "finished_epoch": payload.get("finished_epoch"),
                }
            )
    total_gpu_seconds = float(budget.get("gpu_seconds", 0.0))
    total_wall_seconds = sum(float(w.get("wall_seconds") or 0.0) for w in workers)
    summary = {
        "round_id": C.ROUND_ID,
        "result_label": C.RESULT_LABEL,
        "generated_utc": C.utcnow(),
        "n_planned_tasks": len(planned),
        "n_complete_tasks": sum(1 for r in all_tasks if r["status"] == "complete"),
        "n_not_run": sum(1 for r in all_tasks if r["status"] == "NOT_RUN"),
        "n_failed_or_blocked": sum(1 for r in failures if r.get("status") != "NOT_RUN"),
        "tu_summary": tu_summary,
        "molecule_summary": mol_summary,
        "total_gpu_hours": total_gpu_seconds / 3600.0,
        "workers": workers,
        "total_worker_wall_seconds": total_wall_seconds,
        "budget": {
            "hard_gpu_hours": C.BUDGET["gpu_total_hours"],
            "wall_soft_hours": C.BUDGET["wall_soft_hours"],
            "wall_hard_hours": C.BUDGET["wall_hard_hours"],
        },
        "screenshot_reference_label": C.SCREENSHOT_LABEL,
    }
    C.write_json(C.RESULTS / "summary.json", summary)
    C.write_json(C.RESULTS / "budget.json", budget)
    print(json.dumps({k: summary[k] for k in ("n_planned_tasks", "n_complete_tasks", "n_not_run", "n_failed_or_blocked", "total_gpu_hours")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
