"""Final paired comparison for the six-epoch loss screen.

The main comparison is fixed at epoch 6.  The best-Validation checkpoints are
reported only as auxiliary artefacts and are never substituted for epoch 6.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _configure_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_configure_imports()

import torch  # noqa: E402

import ls_common as C  # noqa: E402

try:  # scipy is a project dependency; keep the comparison usable without it.
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover
    spearmanr = None


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _mean(values: Sequence[float | None]) -> float | None:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(finite)) if finite else None


def _median(values: Sequence[float | None]) -> float | None:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.median(finite)) if finite else None


def _score_map(run: Mapping[str, Any]) -> dict[int, float]:
    out: dict[int, float] = {}
    for row in run.get("scores") or []:
        out[int(row["epoch"])] = float(row["metric"])
    return out


def _curve_map(run_dir: Path) -> dict[int, dict[str, Any]]:
    path = run_dir / "train_curve.csv"
    return {int(r["epoch"]): r for r in _read_csv(path)}


def _task_ap_map(run_dir: Path, epoch: int) -> dict[int, dict[str, Any]]:
    rows = _read_csv(run_dir / "per_task_ap.csv")
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        if int(row["epoch"]) != int(epoch):
            continue
        out[int(row["task"])] = {
            "ap": _float_or_none(row.get("ap")),
            "observed": int(row.get("observed") or 0),
            "positive": int(row.get("positive") or 0),
            "negative": int(row.get("negative") or 0),
            "train_observed": int(row.get("train_observed") or 0),
            "train_positive": int(row.get("train_positive") or 0),
            "stratum": row.get("stratum") or "",
        }
    return out


def _diagnostic_map(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[int, dict[str, Any]] = {}
    for row in _read_csv(path):
        t = int(row["task"])
        out[t] = {
            "train_observed": int(row.get("train_observed") or 0),
            "train_positive": int(row.get("train_positive") or 0),
            "valid_observed": int(row.get("valid_observed") or 0),
            "valid_positive": int(row.get("valid_positive") or 0),
            "stratum": row.get("stratum") or "",
            "train_ap": _float_or_none(row.get("train_ap")),
            "valid_ap": _float_or_none(row.get("valid_ap")),
        }
    return out


def _stratum(train_positive: int) -> str:
    if train_positive <= 100:
        return "sparse_le_100"
    if train_positive <= 1000:
        return "medium_101_1000"
    return "ample_gt_1000"


def _config_value(run: Mapping[str, Any], dotted: str) -> Any:
    cur: Any = run
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _config_audit(m: Mapping[str, Any], t: Mapping[str, Any]) -> dict[str, Any]:
    paths = [
        "source_checkpoint_sha256",
        "dataset",
        "variant",
        "seed",
        "n_outputs",
        "parameters",
        "batch_size",
        "max_epochs",
        "eval_epochs_requested",
        "eval_epochs_completed",
        "recipe.optimizer",
        "recipe.learning_rate",
        "recipe.weight_decay",
        "recipe.adam_betas",
        "recipe.adam_eps",
        "recipe.dropout",
        "recipe.batch_size",
        "recipe.gradient_clip",
        "recipe.scheduler",
        "recipe.precision",
        "recipe.epochs",
        "recipe.early_stopping",
        "n_task_sha256",
        "training_order.global_plan_sha256",
        "training_order.generator_state_sha256",
        "training_order.not_historical_epoch_10_plus_order",
        "data.label_artifact_sha256",
        "data.split_partition_sha256",
        "data.feature_spec_hash",
        "data.norm_fingerprint",
        "initialization.source_checkpoint_sha256",
        "initialization.loaded_parameter_fingerprint",
        "initialization.source_parameter_fingerprint",
        "optimizer_state.source_optimizer_state_fingerprint",
        "optimizer_state.loaded_optimizer_state_fingerprint",
        "rng_start_hashes.combined_sha256",
        "history.source_checkpoint_sha256",
        "history.source_epoch",
        "history.source_optimizer_state_loaded",
        "history.adamw_reinitialised",
        "history.new_six_epoch_order",
        "history.not_bitwise_continuation_of_epoch_10_plus",
        "first_step.rows_sha256",
        "first_step.labels_sha256",
        "test_policy",
    ]
    rows = []
    mismatches = []
    for path in paths:
        vm = _config_value(m, path)
        vt = _config_value(t, path)
        same = vm == vt
        rows.append({"field": path, "M": vm, "T": vt, "equal": bool(same)})
        if not same:
            mismatches.append({"field": path, "M": vm, "T": vt})
    return {
        "fields": rows,
        "mismatches": mismatches,
        "passed": len(mismatches) == 0,
    }


def _top_tasks(rows: Sequence[Mapping[str, Any]], key: str, n: int = 10) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda r: float(r[key]), reverse=True)
    out = []
    for r in ranked[:n]:
        out.append(
            {
                "task": int(r["task"]),
                "delta_ap_T_minus_M": float(r[key]),
                "ap_M": r["ap_M"],
                "ap_T": r["ap_T"],
                "stratum": r["stratum"],
                "train_positive": int(r["train_positive"]),
                "train_observed": int(r["train_observed"]),
                "valid_observed": int(r["valid_observed"]),
                "valid_positive": int(r["valid_positive"]),
            }
        )
    return out


def _strata(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("sparse_le_100", "medium_101_1000", "ample_gt_1000"):
        group = [r for r in rows if r["stratum"] == name]
        scored = [r for r in group if r["delta_ap_T_minus_M"] is not None]
        out[name] = {
            "n_tasks": len(group),
            "n_scored_both": len(scored),
            "train_positive_sum": int(sum(int(r["train_positive"]) for r in group)),
            "train_observed_sum": int(sum(int(r["train_observed"]) for r in group)),
            "ap_M_mean": _mean([r["ap_M"] for r in scored]),
            "ap_T_mean": _mean([r["ap_T"] for r in scored]),
            "delta_ap_mean": _mean([r["delta_ap_T_minus_M"] for r in scored]),
            "delta_ap_median": _median([r["delta_ap_T_minus_M"] for r in scored]),
            "n_T_better": int(sum(1 for r in scored if r["delta_ap_T_minus_M"] > 1e-12)),
            "n_M_better": int(sum(1 for r in scored if r["delta_ap_T_minus_M"] < -1e-12)),
            "n_tied": int(sum(1 for r in scored if abs(r["delta_ap_T_minus_M"]) <= 1e-12)),
        }
    return out


def _correlation(rows: Sequence[Mapping[str, Any]], x_key: str) -> dict[str, Any]:
    pairs = [
        (float(r[x_key]), float(r["delta_ap_T_minus_M"]))
        for r in rows
        if r["delta_ap_T_minus_M"] is not None and r.get(x_key) is not None
    ]
    if len(pairs) < 3 or spearmanr is None:
        return {"n": len(pairs), "spearman_rho": None, "p_value": None}
    x = np.asarray([p[0] for p in pairs], dtype=np.float64)
    y = np.asarray([p[1] for p in pairs], dtype=np.float64)
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return {"n": len(pairs), "spearman_rho": None, "p_value": None, "note": "constant input"}
    rho, p = spearmanr(x, y)
    return {"n": len(pairs), "spearman_rho": float(rho), "p_value": float(p)}


def _first_step_check(run_m_dir: Path, run_t_dir: Path, run_m: Mapping[str, Any], run_t: Mapping[str, Any]) -> dict[str, Any]:
    pm = run_m_dir / "first_step_logits.pt"
    pt = run_t_dir / "first_step_logits.pt"
    if not pm.exists() or not pt.exists():
        return {"available": False, "passed": False}
    a = torch.load(pm, map_location="cpu", weights_only=False)
    b = torch.load(pt, map_location="cpu", weights_only=False)
    diff = float((a["logits"].float() - b["logits"].float()).abs().max().item())
    return {
        "available": True,
        "max_abs_diff_logits": diff,
        "rows_sha256_M": a.get("rows_sha256"),
        "rows_sha256_T": b.get("rows_sha256"),
        "labels_sha256_M": a.get("labels_sha256"),
        "labels_sha256_T": b.get("labels_sha256"),
        "rows_labels_match": bool(
            a.get("rows_sha256") == b.get("rows_sha256")
            and a.get("labels_sha256") == b.get("labels_sha256")
        ),
        "raw_masked_bce_M": _config_value(run_m, "first_step.raw_masked_bce"),
        "raw_masked_bce_T": _config_value(run_t, "first_step.raw_masked_bce"),
        "passed": bool(
            diff <= 1e-5
            and a.get("rows_sha256") == b.get("rows_sha256")
            and a.get("labels_sha256") == b.get("labels_sha256")
        ),
    }


def compare(
    *,
    run_m_dir: Path,
    run_t_dir: Path,
    diagnostic_csv: Path,
    out_path: Path,
    phase_a_summary: Path | None = None,
    paired_initialization: Path | None = None,
) -> dict[str, Any]:
    run_m = C.read_json(run_m_dir / "run.json")
    run_t = C.read_json(run_t_dir / "run.json")
    config = _config_audit(run_m, run_t)
    first_step = _first_step_check(run_m_dir, run_t_dir, run_m, run_t)

    scores_m = _score_map(run_m)
    scores_t = _score_map(run_t)
    ap_m6 = scores_m.get(6)
    ap_t6 = scores_t.get(6)
    ap_m0 = scores_m.get(0)
    ap_t0 = scores_t.get(0)
    d6 = None if ap_m6 is None or ap_t6 is None else float(ap_t6) - float(ap_m6)

    status_ok = bool(
        run_m.get("status") == "COMPLETED"
        and run_t.get("status") == "COMPLETED"
        and int(run_m.get("epochs_run", -1)) == C.TOTAL_EPOCHS
        and int(run_t.get("epochs_run", -1)) == C.TOTAL_EPOCHS
        and set(run_m.get("eval_epochs_completed") or []) == set(C.EVAL_EPOCHS)
        and set(run_t.get("eval_epochs_completed") or []) == set(C.EVAL_EPOCHS)
        and int(run_m.get("non_finite_events", -1)) == 0
        and int(run_t.get("non_finite_events", -1)) == 0
        and run_m.get("test_evaluated") is False
        and run_t.get("test_evaluated") is False
    )
    epoch0_ok = bool(
        ap_m0 is not None
        and ap_t0 is not None
        and abs(float(ap_m0) - float(C.SOURCE["best_valid_ap"])) <= 1e-4
        and abs(float(ap_t0) - float(C.SOURCE["best_valid_ap"])) <= 1e-4
        and abs(float(ap_m0) - float(ap_t0)) <= 1e-5
    )

    task_m = _task_ap_map(run_m_dir, 6)
    task_t = _task_ap_map(run_t_dir, 6)
    diag = _diagnostic_map(diagnostic_csv)
    rows: list[dict[str, Any]] = []
    for task in range(C.N_TASKS):
        rm = task_m.get(task, {})
        rt = task_t.get(task, {})
        dd = diag.get(task, {})
        train_observed = int(dd.get("train_observed", rm.get("train_observed", 0)))
        train_positive = int(dd.get("train_positive", rm.get("train_positive", 0)))
        stratum = dd.get("stratum") or _stratum(train_positive)
        ap_m = rm.get("ap")
        ap_t = rt.get("ap")
        delta = None if ap_m is None or ap_t is None else float(ap_t) - float(ap_m)
        rows.append(
            {
                "task": task,
                "stratum": stratum,
                "train_observed": train_observed,
                "train_positive": train_positive,
                "valid_observed": int(rt.get("observed", 0)),
                "valid_positive": int(rt.get("positive", 0)),
                "valid_negative": int(rt.get("negative", 0)),
                "ap_M": ap_m,
                "ap_T": ap_t,
                "delta_ap_T_minus_M": delta,
                "scored_both": delta is not None,
            }
        )
    scored = [r for r in rows if r["delta_ap_T_minus_M"] is not None]
    deltas = [r["delta_ap_T_minus_M"] for r in scored]
    positive_deltas = sorted([float(x) for x in deltas if x > 1e-12], reverse=True)
    negative_deltas = sorted([float(x) for x in deltas if x < -1e-12])
    positive_sum = float(sum(positive_deltas)) if positive_deltas else 0.0
    negative_sum = float(sum(negative_deltas)) if negative_deltas else 0.0

    def share(k: int) -> float | None:
        if positive_sum <= 0 or not positive_deltas:
            return None
        return float(sum(positive_deltas[:k]) / positive_sum)

    n_t_better = sum(1 for x in deltas if x > 1e-12)
    n_m_better = sum(1 for x in deltas if x < -1e-12)
    n_tied = sum(1 for x in deltas if abs(x) <= 1e-12)

    task_level = {
        "n_scored_both": len(scored),
        "n_tasks_total": C.N_TASKS,
        "n_T_better": int(n_t_better),
        "n_M_better": int(n_m_better),
        "n_tied": int(n_tied),
        "delta_ap_mean": _mean(deltas),
        "delta_ap_median": _median(deltas),
        "delta_ap_min": float(min(deltas)) if deltas else None,
        "delta_ap_max": float(max(deltas)) if deltas else None,
        "positive_delta_sum": positive_sum,
        "negative_delta_sum": negative_sum,
        "n_positive_delta": len(positive_deltas),
        "n_negative_delta": len(negative_deltas),
        "top1_positive_share": share(1),
        "top3_positive_share": share(3),
        "top5_positive_share": share(5),
        "top10_positive_share": share(10),
        "top_improved": _top_tasks(scored, "delta_ap_T_minus_M", 10),
        "top_degraded": [
            {
                "task": int(r["task"]),
                "delta_ap_T_minus_M": float(r["delta_ap_T_minus_M"]),
                "ap_M": r["ap_M"],
                "ap_T": r["ap_T"],
                "stratum": r["stratum"],
                "train_positive": int(r["train_positive"]),
                "train_observed": int(r["train_observed"]),
                "valid_observed": int(r["valid_observed"]),
                "valid_positive": int(r["valid_positive"]),
            }
            for r in sorted(scored, key=lambda r: float(r["delta_ap_T_minus_M"]))[:10]
        ],
        "task_table": rows,
    }

    strata = _strata(rows)
    coverage_relation = {
        "delta_vs_train_positive": _correlation(scored, "train_positive"),
        "delta_vs_train_observed": _correlation(scored, "train_observed"),
        "delta_vs_valid_observed": _correlation(scored, "valid_observed"),
        "delta_vs_valid_positive": _correlation(scored, "valid_positive"),
        "note": "Spearman correlations are descriptive, not a significance claim.",
    }

    curves = {}
    for name, run_dir, run in (("M", run_m_dir, run_m), ("T", run_t_dir, run_t)):
        cmap = _curve_map(run_dir)
        curves[name] = {
            "validation_ap": {str(e): _float_or_none(cmap.get(e, {}).get("valid_ap")) for e in C.EVAL_EPOCHS},
            "train_loss": {str(e): _float_or_none(cmap.get(e, {}).get("train_loss")) for e in range(1, C.TOTAL_EPOCHS + 1)},
            "raw_masked_bce": {str(e): _float_or_none(cmap.get(e, {}).get("raw_masked_bce")) for e in range(1, C.TOTAL_EPOCHS + 1)},
            "grad_norm_mean": {str(e): _float_or_none(cmap.get(e, {}).get("grad_norm_mean")) for e in range(1, C.TOTAL_EPOCHS + 1)},
            "step_ms_p50": {str(e): _float_or_none(cmap.get(e, {}).get("step_ms_p50")) for e in range(1, C.TOTAL_EPOCHS + 1)},
            "best_epoch": run.get("best_epoch"),
            "best_valid_ap_auxiliary": run.get("best_valid_ap"),
            "final_epoch": run.get("final_epoch"),
            "final_valid_ap": run.get("final_valid_ap"),
        }

    phase_a_hours = None
    if phase_a_summary and phase_a_summary.exists():
        pa = C.read_json(phase_a_summary)
        phase_a_hours = pa.get("gpu_hours")
    paired_hours = None
    if paired_initialization and paired_initialization.exists():
        pi = C.read_json(paired_initialization)
        paired_hours = _config_value(pi, "details.gpu_hours.sum")
    hours = {
        "M": run_m.get("gpu_hours"),
        "T": run_t.get("gpu_hours"),
        "phase_A": phase_a_hours,
        "preflight_pair": paired_hours,
        "total": float(
            (run_m.get("gpu_hours") or 0.0)
            + (run_t.get("gpu_hours") or 0.0)
            + (phase_a_hours or 0.0)
            + (paired_hours or 0.0)
        ),
    }
    budget_ok = bool(hours["total"] <= float(C.BUDGET["gpu_hours_total"]) + 1e-9)

    decision = C.decide_screen_label(
        ap_m6=ap_m6,
        ap_t6=ap_t6,
        status_m=str(run_m.get("status")),
        status_t=str(run_t.get("status")),
        n_t_better=int(n_t_better),
        n_scored_both=len(scored),
        config_valid=bool(config["passed"] and status_ok and epoch0_ok and first_step.get("passed")),
    )
    if not budget_ok:
        decision["label"] = "INCOMPLETE_OR_INVALID"
        decision.setdefault("conditions", {})["budget_within_limit"] = False
        decision["reason"] = "total GPU-hour budget exceeded"
    else:
        decision["conditions"]["budget_within_limit"] = True

    out = {
        "schema": "lcrc-pcba-loss-screen-paired-comparison-v1",
        "round_id": C.ROUND_ID,
        "phase": "B_fixed_epoch6_comparison",
        "source_checkpoint": C.track_rel(C.SOURCE["checkpoint"]),
        "source_checkpoint_sha256": C.SOURCE["checkpoint_sha256"],
        "historical_valid_ap": float(C.SOURCE["best_valid_ap"]),
        "run_M": C.track_rel(run_m_dir),
        "run_T": C.track_rel(run_t_dir),
        "status": {"M": run_m.get("status"), "T": run_t.get("status")},
        "config_audit": config,
        "status_ok": status_ok,
        "epoch0_ok": epoch0_ok,
        "first_step_audit": first_step,
        "epoch6": {
            "AP_M6": ap_m6,
            "AP_T6": ap_t6,
            "D6_T_minus_M": d6,
            "AP0": float(C.SOURCE["best_valid_ap"]),
            "T6_minus_AP0": None if ap_t6 is None else float(ap_t6) - float(C.SOURCE["best_valid_ap"]),
            "M6_minus_AP0": None if ap_m6 is None else float(ap_m6) - float(C.SOURCE["best_valid_ap"]),
            "epoch0_AP_M": ap_m0,
            "epoch0_AP_T": ap_t0,
        },
        "task_level": task_level,
        "strata": strata,
        "coverage_relation": coverage_relation,
        "curves": curves,
        "auxiliary_best_checkpoints": {
            "M": {"epoch": run_m.get("best_epoch"), "valid_ap": run_m.get("best_valid_ap")},
            "T": {"epoch": run_t.get("best_epoch"), "valid_ap": run_t.get("best_valid_ap")},
            "note": "auxiliary only; main comparison is fixed epoch 6",
        },
        "resource_accounting_gpu_hours": hours,
        "budget_within_limit": budget_ok,
        "decision": decision,
        "decision_label": decision["label"],
        "observations": {
            "main_metric_epoch": 6,
            "D6_raw": d6,
            "AP_T6_ge_AP0": None if ap_t6 is None else bool(float(ap_t6) >= float(C.SOURCE["best_valid_ap"])),
            "task_majority_64_of_127": bool(len(scored) == 127 and n_t_better >= 64),
            "n_T_better": int(n_t_better),
            "n_scored_both": len(scored),
        },
        "created_utc": C.utcnow(),
        "test_evaluated": False,
    }
    C.write_json(out_path, out)
    C.log(
        f"paired comparison label={decision['label']} AP_M6={ap_m6} AP_T6={ap_t6} "
        f"D6={d6} n_T_better={n_t_better}/{len(scored)} total_gpu_hours={hours['total']:.6f}"
    )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-M", required=True)
    parser.add_argument("--run-T", required=True)
    parser.add_argument("--diagnostic-csv", default=str(C.ANALYSIS / "train_valid_diagnostic.csv"))
    parser.add_argument("--phase-a-summary", default=str(C.ANALYSIS / "diagnostic_summary.json"))
    parser.add_argument("--paired-initialization", default=str(C.ANALYSIS / "paired_initialization.json"))
    parser.add_argument("--out", default=str(C.ANALYSIS / "paired_comparison.json"))
    args = parser.parse_args(argv)
    compare(
        run_m_dir=Path(args.run_M),
        run_t_dir=Path(args.run_T),
        diagnostic_csv=Path(args.diagnostic_csv),
        out_path=Path(args.out),
        phase_a_summary=Path(args.phase_a_summary) if args.phase_a_summary else None,
        paired_initialization=Path(args.paired_initialization)
        if args.paired_initialization
        else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
