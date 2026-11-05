"""Phase A: frozen epoch-9 Train/Validation diagnostic (no training).

The model is ``eval()`` and every forward runs under ``torch.inference_mode()``.
Only the official Train and Validation rows are inferred; official Test targets
are all NaN in the label artifact and are never indexed.  Predictions are
written to memory-mapped arrays so the full split is never copied unnecessarily.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _configure_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    os.environ.setdefault("OMP_NUM_THREADS", "8")


_configure_imports()

import torch  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

import ls_common as C  # noqa: E402
import ls_checkpoint as CK  # noqa: E402
import lr_data  # noqa: E402
import pc_model as BM  # noqa: E402

CSV_FIELDS = [
    "task",
    "stratum",
    "train_observed",
    "train_positive",
    "train_negative",
    "train_prevalence",
    "train_ap",
    "train_roc_auc",
    "train_ap_minus_prevalence",
    "train_ap_computable",
    "train_auc_computable",
    "valid_observed",
    "valid_positive",
    "valid_negative",
    "valid_prevalence",
    "valid_ap",
    "valid_roc_auc",
    "valid_ap_minus_prevalence",
    "valid_ap_computable",
    "valid_auc_computable",
    "ap_diff_train_minus_valid",
    "auc_diff_train_minus_valid",
]


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(row.get(k)) for k in CSV_FIELDS})
    return C.sha256_file(path)


def _official_ap(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    """Official OGB Evaluator without an extra float64 copy of the full split."""
    from ogb.graphproppred import Evaluator

    if y_true.shape != y_score.shape or y_true.ndim != 2:
        raise ValueError(f"AP arrays {y_true.shape} / {y_score.shape}")
    if y_true.shape[1] != C.N_TASKS:
        raise ValueError(f"expected {C.N_TASKS} tasks, got {y_true.shape[1]}")
    score = Evaluator(name="ogbg-molpcba").eval({"y_true": y_true, "y_pred": y_score})
    value = float(score["ap"])
    if not np.isfinite(value):
        raise ValueError(f"official AP is non-finite: {value}")
    return {
        "value": value,
        "source": "OGB official Evaluator(ogbg-molpcba)",
        "metric": "ap",
    }


def _per_task_metrics(y: np.ndarray, score: np.ndarray) -> list[dict[str, Any]]:
    """Task-wise AP/ROC-AUC and coverage; None when a class is absent."""
    if y.shape != score.shape or y.ndim != 2:
        raise ValueError(f"metric arrays {y.shape} / {score.shape}")
    rows: list[dict[str, Any]] = []
    for t in range(y.shape[1]):
        labeled = np.isfinite(y[:, t])
        if not bool(labeled.any()):
            rows.append(
                {
                    "task": t,
                    "observed": 0,
                    "positive": 0,
                    "negative": 0,
                    "prevalence": None,
                    "ap": None,
                    "roc_auc": None,
                    "ap_computable": False,
                    "auc_computable": False,
                }
            )
            continue
        yt = np.asarray(y[labeled, t], dtype=np.float64)
        ys = np.asarray(score[labeled, t], dtype=np.float64)
        if not np.isfinite(ys).all():
            raise RuntimeError(f"non-finite scores in task {t}")
        if not np.all((yt == 0.0) | (yt == 1.0)):
            raise RuntimeError(f"non-binary observed labels in task {t}")
        n_pos = int((yt == 1.0).sum())
        n_neg = int((yt == 0.0).sum())
        n_obs = int(labeled.sum())
        ap = None
        auc = None
        if n_pos > 0 and n_neg > 0:
            ap = float(average_precision_score(yt, ys))
            auc = float(roc_auc_score(yt, ys))
            if not np.isfinite(ap) or not np.isfinite(auc):
                raise RuntimeError(f"non-finite AP/AUC in task {t}: ap={ap} auc={auc}")
        rows.append(
            {
                "task": t,
                "observed": n_obs,
                "positive": n_pos,
                "negative": n_neg,
                "prevalence": float(n_pos / n_obs) if n_obs else None,
                "ap": ap,
                "roc_auc": auc,
                "ap_computable": ap is not None,
                "auc_computable": auc is not None,
            }
        )
    return rows


def _mean(values: Sequence[float | None]) -> float | None:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(finite)) if finite else None


def _median(values: Sequence[float | None]) -> float | None:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.median(finite)) if finite else None


def _min(values: Sequence[float | None]) -> float | None:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.min(finite)) if finite else None


def _max(values: Sequence[float | None]) -> float | None:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.max(finite)) if finite else None


def _stratum(train_positive: int) -> str:
    if train_positive <= 100:
        return "sparse_le_100"
    if train_positive <= 1000:
        return "medium_101_1000"
    return "ample_gt_1000"


def _predict_to_memmap(
    model: torch.nn.Module,
    cache: Mapping[str, Any],
    rows: np.ndarray,
    device: torch.device,
    *,
    logits_path: Path,
    probs_path: Path,
    batch_size: int = 512,
) -> dict[str, Any]:
    rows = np.asarray(rows, dtype=np.int64)
    logits_path.parent.mkdir(parents=True, exist_ok=True)
    logits_mm = np.lib.format.open_memmap(
        logits_path, mode="w+", dtype=np.float32, shape=(len(rows), C.N_TASKS)
    )
    probs_mm = np.lib.format.open_memmap(
        probs_path, mode="w+", dtype=np.float32, shape=(len(rows), C.N_TASKS)
    )
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    n_batches = 0
    with torch.inference_mode():
        for start in range(0, len(rows), int(batch_size)):
            stop = min(start + int(batch_size), len(rows))
            chunk = rows[start:stop]
            batch = BM.to_device(BM.collate(cache, chunk.tolist()), device)
            logits = model(batch)
            if tuple(logits.shape) != (stop - start, C.N_TASKS):
                raise RuntimeError(f"model output {tuple(logits.shape)} != {(stop - start, C.N_TASKS)}")
            arr = logits.detach().cpu().numpy().astype(np.float32, copy=False)
            if not np.isfinite(arr).all():
                raise RuntimeError(f"non-finite logits in rows {start}:{stop}")
            logits_mm[start:stop] = arr
            probs_mm[start:stop] = (1.0 / (1.0 + np.exp(-arr.astype(np.float64)))).astype(np.float32)
            n_batches += 1
    logits_mm.flush()
    probs_mm.flush()
    del logits_mm, probs_mm
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = float(time.perf_counter() - t0)
    out = {
        "rows": int(len(rows)),
        "tasks": int(C.N_TASKS),
        "batch_size": int(batch_size),
        "n_batches": int(n_batches),
        "seconds": elapsed,
        "molecules_per_second": float(len(rows) / elapsed) if elapsed > 0 else None,
        "logits_path": C.track_rel(logits_path),
        "probs_path": C.track_rel(probs_path),
        "logits_sha256": C.sha256_file(logits_path),
        "probs_sha256": C.sha256_file(probs_path),
        "dtype": "float32",
        "inference_mode": True,
        "model_eval": True,
    }
    if device.type == "cuda":
        out["gpu_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        out["gpu_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    return out


def _summarize_metric_block(rows: Sequence[Mapping[str, Any]], prefix: str) -> dict[str, Any]:
    ap = [r[f"{prefix}_ap"] for r in rows]
    auc = [r[f"{prefix}_roc_auc"] for r in rows]
    gain = [
        None
        if r[f"{prefix}_ap"] is None or r[f"{prefix}_prevalence"] is None
        else float(r[f"{prefix}_ap"]) - float(r[f"{prefix}_prevalence"])
        for r in rows
    ]
    return {
        "n_tasks": len(rows),
        "n_ap_computable": sum(1 for x in ap if x is not None),
        "n_auc_computable": sum(1 for x in auc if x is not None),
        "n_observed_tasks": sum(1 for r in rows if int(r[f"{prefix}_observed"]) > 0),
        "macro_ap": _mean(ap),
        "median_ap": _median(ap),
        "macro_roc_auc": _mean(auc),
        "median_roc_auc": _median(auc),
        "macro_ap_minus_prevalence": _mean(gain),
        "median_ap_minus_prevalence": _median(gain),
        "total_observed": int(sum(int(r[f"{prefix}_observed"]) for r in rows)),
        "total_positive": int(sum(int(r[f"{prefix}_positive"]) for r in rows)),
        "total_negative": int(sum(int(r[f"{prefix}_negative"]) for r in rows)),
        "positive_rate": (
            float(sum(int(r[f"{prefix}_positive"]) for r in rows))
            / float(sum(int(r[f"{prefix}_observed"]) for r in rows))
            if sum(int(r[f"{prefix}_observed"]) for r in rows) > 0
            else None
        ),
    }


def _strata_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("sparse_le_100", "medium_101_1000", "ample_gt_1000"):
        group = [r for r in rows if r["stratum"] == name]
        if not group:
            out[name] = {"n_tasks": 0}
            continue
        out[name] = {
            "n_tasks": len(group),
            "train_positive_sum": int(sum(int(r["train_positive"]) for r in group)),
            "train_observed_sum": int(sum(int(r["train_observed"]) for r in group)),
            "train_ap_mean": _mean([r["train_ap"] for r in group]),
            "train_ap_median": _median([r["train_ap"] for r in group]),
            "valid_ap_mean": _mean([r["valid_ap"] for r in group]),
            "valid_ap_median": _median([r["valid_ap"] for r in group]),
            "train_auc_mean": _mean([r["train_roc_auc"] for r in group]),
            "valid_auc_mean": _mean([r["valid_roc_auc"] for r in group]),
            "ap_diff_mean": _mean([r["ap_diff_train_minus_valid"] for r in group]),
            "ap_diff_median": _median([r["ap_diff_train_minus_valid"] for r in group]),
            "auc_diff_mean": _mean([r["auc_diff_train_minus_valid"] for r in group]),
            "auc_diff_median": _median([r["auc_diff_train_minus_valid"] for r in group]),
            "mean_train_prevalence": _mean([r["train_prevalence"] for r in group]),
            "mean_valid_prevalence": _mean([r["valid_prevalence"] for r in group]),
        }
    return out


def _delta_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ap_diff = [r["ap_diff_train_minus_valid"] for r in rows]
    auc_diff = [r["auc_diff_train_minus_valid"] for r in rows]
    return {
        "n_paired_ap": sum(1 for x in ap_diff if x is not None),
        "ap_diff_mean": _mean(ap_diff),
        "ap_diff_median": _median(ap_diff),
        "ap_diff_min": _min(ap_diff),
        "ap_diff_max": _max(ap_diff),
        "n_paired_auc": sum(1 for x in auc_diff if x is not None),
        "auc_diff_mean": _mean(auc_diff),
        "auc_diff_median": _median(auc_diff),
        "auc_diff_min": _min(auc_diff),
        "auc_diff_max": _max(auc_diff),
    }


def _top_gaps(rows: Sequence[Mapping[str, Any]], n: int = 10) -> dict[str, Any]:
    paired = [r for r in rows if r["ap_diff_train_minus_valid"] is not None]
    ranked = sorted(paired, key=lambda r: float(r["ap_diff_train_minus_valid"]), reverse=True)

    def slim(r: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "task": int(r["task"]),
            "stratum": r["stratum"],
            "train_positive": int(r["train_positive"]),
            "train_observed": int(r["train_observed"]),
            "valid_positive": int(r["valid_positive"]),
            "valid_observed": int(r["valid_observed"]),
            "train_ap": r["train_ap"],
            "valid_ap": r["valid_ap"],
            "ap_diff_train_minus_valid": r["ap_diff_train_minus_valid"],
        }

    return {
        "largest_train_minus_valid": [slim(r) for r in ranked[:n]],
        "smallest_train_minus_valid": [slim(r) for r in ranked[-n:]],
    }


def run(
    *,
    device: torch.device,
    out_dir: Path,
    analysis_dir: Path,
    batch_size: int = 512,
) -> dict[str, Any]:
    t_start = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema": "lcrc-pcba-loss-screen-phase-a-v1",
        "round_id": C.ROUND_ID,
        "phase": "A_frozen_train_valid_diagnostic",
        "source_checkpoint": C.track_rel(C.SOURCE["checkpoint"]),
        "source_checkpoint_sha256": C.SOURCE["checkpoint_sha256"],
        "historical_valid_ap": float(C.SOURCE["best_valid_ap"]),
        "device": str(device),
        "environment": C.environment(),
        "gpu_occupancy_start": C.gpu_occupancy(),
        "test_evaluated": False,
        "started_utc": C.utcnow(),
    }
    gates: dict[str, bool] = {}
    try:
        C.assert_inherited_constants()
        task = lr_data.load_task()
        result["data_audit"] = task.data_audit
        result["data_load_seconds"] = float(task.load_seconds)

        checkpoint = CK.load_source_checkpoint()
        result["checkpoint_metadata"] = CK.verify_source_metadata(checkpoint)

        model, model_info, load_audit = CK.build_model_from_checkpoint(
            task, checkpoint, device=device, group="A_diagnostic"
        )
        result["initialization"] = load_audit
        result["model_info"] = model_info
        gates["checkpoint_sha256"] = True
        gates["checkpoint_metadata"] = True
        gates["model_state_bitwise_equal"] = bool(load_audit["state_dict_bitwise_equal"])

        labels = task.labels
        train_rows = np.asarray(task.train_rows, dtype=np.int64)
        valid_rows = np.asarray(task.valid_rows, dtype=np.int64)
        test_rows = np.asarray(task.test_rows, dtype=np.int64)
        train_labels = labels[train_rows]
        valid_labels = labels[valid_rows]

        train_obs = int(np.isfinite(train_labels).sum())
        valid_obs = int(np.isfinite(valid_labels).sum())
        test_finite = int(np.isfinite(labels[test_rows]).sum())
        all_finite = int(np.isfinite(labels).sum())
        outside = int(all_finite - train_obs - valid_obs)
        row_checks = {
            "train_rows": int(train_rows.size),
            "train_unique_rows": int(np.unique(train_rows).size),
            "valid_rows": int(valid_rows.size),
            "valid_unique_rows": int(np.unique(valid_rows).size),
            "test_rows": int(test_rows.size),
            "test_unique_rows": int(np.unique(test_rows).size),
            "train_valid_overlap": int(np.intersect1d(train_rows, valid_rows).size),
            "train_test_overlap": int(np.intersect1d(train_rows, test_rows).size),
            "valid_test_overlap": int(np.intersect1d(valid_rows, test_rows).size),
            "train_observed_entries": train_obs,
            "valid_observed_entries": valid_obs,
            "test_finite_entries": test_finite,
            "finite_entries_outside_train_valid": outside,
            "train_positive_entries": int((train_labels[np.isfinite(train_labels)] == 1).sum()),
            "valid_positive_entries": int((valid_labels[np.isfinite(valid_labels)] == 1).sum()),
            "train_rows_sha256": C.sha256_int64(train_rows),
            "valid_rows_sha256": C.sha256_int64(valid_rows),
            "test_rows_sha256": C.sha256_int64(test_rows),
            "train_rows_sorted_sha256": C.sha256_int64(np.sort(train_rows)),
            "valid_rows_sorted_sha256": C.sha256_int64(np.sort(valid_rows)),
        }
        result["row_checks"] = row_checks
        gates["train_rows_complete_unique"] = (
            int(train_rows.size) == C.N_TRAIN and int(np.unique(train_rows).size) == C.N_TRAIN
        )
        gates["valid_rows_complete_unique"] = (
            int(valid_rows.size) == C.N_VALID and int(np.unique(valid_rows).size) == C.N_VALID
        )
        gates["train_valid_disjoint"] = row_checks["train_valid_overlap"] == 0
        gates["test_never_used"] = (
            test_finite == 0
            and row_checks["train_test_overlap"] == 0
            and row_checks["valid_test_overlap"] == 0
            and outside == 0
        )
        gates["observed_entry_counts"] = (
            train_obs == int(task.data_audit["train_observed_entries"])
            and valid_obs == int(task.data_audit["valid_observed_entries"])
        )

        # Inference: Train and Validation only.
        train_pred = _predict_to_memmap(
            model,
            task.cache,
            train_rows,
            device,
            logits_path=out_dir / "preds" / "train_logits.npy",
            probs_path=out_dir / "preds" / "train_probs.npy",
            batch_size=int(batch_size),
        )
        valid_pred = _predict_to_memmap(
            model,
            task.cache,
            valid_rows,
            device,
            logits_path=out_dir / "preds" / "valid_logits.npy",
            probs_path=out_dir / "preds" / "valid_probs.npy",
            batch_size=int(batch_size),
        )
        result["inference"] = {"train": train_pred, "valid": valid_pred}
        gates["train_predictions_shape"] = train_pred["rows"] == C.N_TRAIN
        gates["valid_predictions_shape"] = valid_pred["rows"] == C.N_VALID

        train_probs = np.load(out_dir / "preds" / "train_probs.npy", mmap_mode="r")
        valid_probs = np.load(out_dir / "preds" / "valid_probs.npy", mmap_mode="r")
        train_task = _per_task_metrics(train_labels, train_probs)
        valid_task = _per_task_metrics(valid_labels, valid_probs)

        official_train = _official_ap(train_labels, train_probs)
        official_valid = _official_ap(valid_labels, valid_probs)
        independent_train = _mean([r["ap"] for r in train_task])
        independent_valid = _mean([r["ap"] for r in valid_task])
        train_agreement = abs(float(official_train["value"]) - float(independent_train))
        valid_agreement = abs(float(official_valid["value"]) - float(independent_valid))
        historical_delta = abs(float(official_valid["value"]) - float(C.SOURCE["best_valid_ap"]))
        result["official_ap"] = {
            "train": official_train,
            "valid": official_valid,
            "independent_train": independent_train,
            "independent_valid": independent_valid,
            "train_abs_delta": float(train_agreement),
            "valid_abs_delta": float(valid_agreement),
            "agreement_tolerance": 1e-8,
            "historical_valid_ap": float(C.SOURCE["best_valid_ap"]),
            "historical_abs_delta": float(historical_delta),
            "historical_tolerance": 1e-4,
        }
        gates["official_ap_matches_independent_train"] = train_agreement <= 1e-8
        gates["official_ap_matches_independent_valid"] = valid_agreement <= 1e-8
        gates["valid_ap_reproduces_historical"] = historical_delta <= 1e-4

        # Merge task tables.
        rows: list[dict[str, Any]] = []
        for t in range(C.N_TASKS):
            tr = train_task[t]
            va = valid_task[t]
            stratum = _stratum(int(tr["positive"]))
            ap_diff = (
                None
                if tr["ap"] is None or va["ap"] is None
                else float(tr["ap"]) - float(va["ap"])
            )
            auc_diff = (
                None
                if tr["roc_auc"] is None or va["roc_auc"] is None
                else float(tr["roc_auc"]) - float(va["roc_auc"])
            )
            rows.append(
                {
                    "task": t,
                    "stratum": stratum,
                    "train_observed": tr["observed"],
                    "train_positive": tr["positive"],
                    "train_negative": tr["negative"],
                    "train_prevalence": tr["prevalence"],
                    "train_ap": tr["ap"],
                    "train_roc_auc": tr["roc_auc"],
                    "train_ap_minus_prevalence": (
                        None
                        if tr["ap"] is None or tr["prevalence"] is None
                        else float(tr["ap"]) - float(tr["prevalence"])
                    ),
                    "train_ap_computable": tr["ap_computable"],
                    "train_auc_computable": tr["auc_computable"],
                    "valid_observed": va["observed"],
                    "valid_positive": va["positive"],
                    "valid_negative": va["negative"],
                    "valid_prevalence": va["prevalence"],
                    "valid_ap": va["ap"],
                    "valid_roc_auc": va["roc_auc"],
                    "valid_ap_minus_prevalence": (
                        None
                        if va["ap"] is None or va["prevalence"] is None
                        else float(va["ap"]) - float(va["prevalence"])
                    ),
                    "valid_ap_computable": va["ap_computable"],
                    "valid_auc_computable": va["auc_computable"],
                    "ap_diff_train_minus_valid": ap_diff,
                    "auc_diff_train_minus_valid": auc_diff,
                }
            )

        csv_path = analysis_dir / "train_valid_diagnostic.csv"
        csv_sha = _write_csv(csv_path, rows)
        result["diagnostic_csv"] = C.track_rel(csv_path)
        result["diagnostic_csv_sha256"] = csv_sha

        train_summary = _summarize_metric_block(rows, "train")
        valid_summary = _summarize_metric_block(rows, "valid")
        result["summary"] = {
            "train": train_summary,
            "valid": valid_summary,
            "delta": _delta_summary(rows),
            "strata": _strata_summary(rows),
            "top_ap_gaps": _top_gaps(rows),
            "n_tasks": C.N_TASKS,
            "n_tasks_with_both_ap": sum(
                1 for r in rows if r["train_ap"] is not None and r["valid_ap"] is not None
            ),
        }
        gates["all_ap_auc_finite_when_computable"] = all(
            (r["train_ap"] is None or np.isfinite(r["train_ap"]))
            and (r["train_roc_auc"] is None or np.isfinite(r["train_roc_auc"]))
            and (r["valid_ap"] is None or np.isfinite(r["valid_ap"]))
            and (r["valid_roc_auc"] is None or np.isfinite(r["valid_roc_auc"]))
            for r in rows
        )
        gates["no_test_metric"] = True

        result["gates"] = {k: bool(v) for k, v in gates.items()}
        result["all_passed"] = bool(all(gates.values()))
        result["gpu_occupancy_end"] = C.gpu_occupancy()
    except Exception as exc:  # noqa: BLE001 - the failure is the deliverable
        result["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        result["gates"] = {k: bool(v) for k, v in gates.items()}
        result["all_passed"] = False
    result["gpu_seconds"] = float(time.perf_counter() - t_start)
    result["gpu_hours"] = float(result["gpu_seconds"] / 3600.0)
    result["finished_utc"] = C.utcnow()
    C.write_json(analysis_dir / "diagnostic_summary.json", result)
    C.log(
        f"phase A all_passed={result['all_passed']} "
        f"trainAP={result.get('official_ap', {}).get('train', {}).get('value')} "
        f"validAP={result.get('official_ap', {}).get('valid', {}).get('value')} "
        f"gpu_hours={result['gpu_hours']:.6f}"
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default=str(C.RUNS / "A_diagnostic"))
    parser.add_argument("--analysis-dir", default=str(C.ANALYSIS))
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        C.log("CUDA requested but unavailable")
        return 2
    result = run(
        device=device,
        out_dir=Path(args.out_dir),
        analysis_dir=Path(args.analysis_dir),
        batch_size=int(args.batch_size),
    )
    return 0 if result.get("all_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
