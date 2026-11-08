"""Checkpoint selection manifest and the frozen comparison / decision.

``select`` reads only the completed Validation record and writes the
pre-Test selection manifest; it never looks at a Test metric.
``compare`` reads the post-freeze ``test_scores.json`` and produces the
pre-registered W2-versus-TopRank comparison, support strata, dominant-task
diagnostics and the decision label.

The decision thresholds are fixed in ``PROTOCOL.md`` before the run; this file
does not search them.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import tr_common as C

ANALYSIS = C.ANALYSIS
SELECTION_PATH = ANALYSIS / "selected_checkpoint.json"
SCORES_PATH = ANALYSIS / "test_scores.json"
COMPARISON_PATH = ANALYSIS / "comparison.json"
DECISION_PATH = ANALYSIS / "decision.json"
PER_TASK_DELTA_PATH = ANALYSIS / "per_task_test_deltas.csv"
DECISION_LABELS = ("VALID_AND_TEST_GAIN", "VALID_ONLY", "EXPOSED_TEST_ONLY_SIGNAL", "NO_CLEAR_GAIN")


def _abs_path(value: str | Path) -> Path:
    return C.relative_or_absolute(value)


def _sha_ok(path: Path, expected: str) -> bool:
    return path.is_file() and C.sha256_file(path) == expected


def run_select(run_path: Path, out_path: Path) -> int:
    run = C.read_json(run_path)
    if run.get("schema") != C.SCHEMA_RUN:
        raise RuntimeError("training run has the wrong schema")
    if run.get("round_id") != C.ROUND_ID:
        raise RuntimeError("training run belongs to another round")
    if run.get("status") != "COMPLETED":
        raise RuntimeError(f"training run is {run.get('status')}, not COMPLETED")
    if not bool(run.get("formal_run")):
        raise RuntimeError("selection requires the formal 40-epoch run")
    if int(run.get("epochs_run", 0)) != int(C.TOTAL_EPOCHS):
        raise RuntimeError("selection requires 40 completed epochs")
    if int(run.get("global_steps_run", 0)) != int(C.TOTAL_STEPS_40):
        raise RuntimeError("selection requires 109520 global steps")
    if len(run.get("eval_epochs_completed", [])) != int(C.TOTAL_EPOCHS):
        raise RuntimeError("selection requires Validation AP on every epoch")
    if run.get("best_epoch") is None or run.get("best_valid_ap") is None:
        raise RuntimeError("run has no best Validation record")
    if run.get("test_evaluated") is not False:
        raise RuntimeError("the training run must declare test_evaluated=false")
    if run.get("batch_plan", {}).get("global_plan_sha256") != C.GLOBAL_PLAN_SHA256:
        raise RuntimeError("training batch plan drifted")

    checkpoint = _abs_path(run["best_checkpoint"])
    if not _sha_ok(checkpoint, run["best_checkpoint_sha256"]):
        raise RuntimeError(f"best checkpoint {checkpoint} is missing or changed")

    best_epoch = int(run["best_epoch"])
    by_epoch = {int(e): r for e, r in run.get("fixed_epoch_reports", {}).items()}
    if best_epoch not in by_epoch:
        raise RuntimeError("best epoch report is missing")
    # The selection rule is restated from the per-epoch Validation AP only.
    candidates = sorted(
        (int(e), float(r["ap"])) for e, r in by_epoch.items() if r.get("ap") is not None
    )
    argmax_epoch = None
    argmax_ap = -math.inf
    for epoch, ap in candidates:
        if ap > argmax_ap:  # strict: equal floats keep the earlier epoch
            argmax_epoch, argmax_ap = epoch, ap
    if argmax_epoch != best_epoch or abs(argmax_ap - float(run["best_valid_ap"])) > 1e-12:
        raise RuntimeError(
            f"selection rule mismatch: argmax epoch {argmax_epoch} ap {argmax_ap} "
            f"vs recorded {best_epoch} / {run['best_valid_ap']}"
        )

    payload = {
        "schema": C.SCHEMA_SELECTION,
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "selection_rule": "argmax official Validation AP over epochs 1..40; strict > keeps the earlier epoch on exact float ties",
        "selection_uses_test": False,
        "test_read": False,
        "test_role": C.TEST_ROLE,
        "source_run": C.track_rel(run_path),
        "source_run_sha256": C.sha256_file(run_path),
        "source_run_git_commit": run.get("git_commit"),
        "best_epoch": best_epoch,
        "best_valid_ap": float(run["best_valid_ap"]),
        "epoch40_valid_ap": run.get("epoch40_valid_ap"),
        "checkpoint": C.track_rel(checkpoint),
        "checkpoint_sha256": run["best_checkpoint_sha256"],
        "parameters": int(run["parameters"]),
        "lambda_rank": float(run["calibration"]["lambda_rank"]),
        "calibration_sha256": run["calibration"]["sha256"],
        "batch_plan_sha256": run["batch_plan"]["global_plan_sha256"],
        "note": "The Test split is exposed; the checkpoint was selected on Validation only.",
    }
    C.write_json(out_path, payload)
    C.log(
        f"[select] epoch={best_epoch} valid={payload['best_valid_ap']:.9f} "
        f"checkpoint={payload['checkpoint_sha256'][:12]}"
    )
    return 0


def _decision(
    *,
    best_valid: float,
    test_ap: float,
    high_support: Mapping[str, Any],
    work_target: Mapping[str, Any],
) -> dict[str, Any]:
    w2_valid = float(C.W2["best_valid_ap"])
    w2_test = float(C.W2["test_ap_official"])
    valid_improved = bool(best_valid > w2_valid)
    test_improved = bool(test_ap > w2_test)
    test_target_met = bool(test_ap >= float(work_target["test_ap_min"]))
    high_mean = high_support.get("mean_delta")
    high_improved = int(high_support.get("improved", 0))
    high_n = int(high_support.get("n_tasks", 0))
    high_mean_ok = bool(high_mean is not None and float(high_mean) >= float(work_target["high_support_mean_delta_min"]))
    high_count_ok = bool(high_improved >= int(work_target["high_support_improved_min"]))
    high_n_ok = bool(high_n == int(work_target["high_support_tasks_historical"]))
    engineering_signal = bool(test_target_met and high_mean_ok and high_count_ok and high_n_ok)

    if test_target_met and valid_improved:
        label = "VALID_AND_TEST_GAIN"
    elif test_target_met and not valid_improved:
        label = "EXPOSED_TEST_ONLY_SIGNAL"
    elif valid_improved and not test_target_met:
        label = "VALID_ONLY"
    else:
        label = "NO_CLEAR_GAIN"
    if label not in DECISION_LABELS:
        raise RuntimeError(label)

    return {
        "label": label,
        "valid_improved_vs_w2": valid_improved,
        "test_improved_vs_w2": test_improved,
        "test_target_met": test_target_met,
        "high_support_mean_ok": high_mean_ok,
        "high_support_count_ok": high_count_ok,
        "high_support_n_ok": high_n_ok,
        "engineering_signal": engineering_signal,
        "rule": {
            "valid_improved": f"Best Valid AP > W2 {w2_valid:.9f}",
            "test_target": f"Test AP >= {float(work_target['test_ap_min']):.6f}",
            "high_support": (
                f"Test-positive >= {int(work_target['high_support_min_test_positives'])} tasks: "
                f"mean delta >= {float(work_target['high_support_mean_delta_min']):.3f} and "
                f"improved >= {int(work_target['high_support_improved_min'])}/"
                f"{int(work_target['high_support_tasks_historical'])}"
            ),
            "labels": {
                "VALID_AND_TEST_GAIN": "test target met and Best Valid AP above W2",
                "VALID_ONLY": "Best Valid AP above W2, Test target not met",
                "EXPOSED_TEST_ONLY_SIGNAL": "test target met but Best Valid AP not above W2",
                "NO_CLEAR_GAIN": "neither signal met",
            },
        },
        "note": (
            "This is a pre-registered resource decision, not a statistical significance test. "
            "The Test split is exposed; this is not a blind generalisation estimate."
        ),
    }


def run_compare(
    run_path: Path,
    selection_path: Path,
    scores_path: Path,
    comparison_path: Path,
    decision_path: Path,
    per_task_path: Path,
) -> int:
    run = C.read_json(run_path)
    selection = C.read_json(selection_path)
    scores = C.read_json(scores_path)
    if run.get("round_id") != C.ROUND_ID or selection.get("round_id") != C.ROUND_ID:
        raise RuntimeError("round identity mismatch")
    if scores.get("round_id") != C.ROUND_ID:
        raise RuntimeError("test scores belong to another round")
    if scores.get("test_role") != C.TEST_ROLE:
        raise RuntimeError("unexpected Test role")
    if selection.get("test_read") is not False or selection.get("selection_uses_test") is not False:
        raise RuntimeError("selection used Test or is not sealed")
    if C.track_rel(_abs_path(selection["checkpoint"])) != C.track_rel(_abs_path(run["best_checkpoint"])):
        raise RuntimeError("selection checkpoint differs from the run best checkpoint")
    if selection["checkpoint_sha256"] != run["best_checkpoint_sha256"]:
        raise RuntimeError("selection checkpoint sha differs from the run")
    if abs(float(selection["best_valid_ap"]) - float(run["best_valid_ap"])) > 1e-12:
        raise RuntimeError("selection Valid AP differs from the run")

    top = scores["models"]["TopRank_best_valid"]
    w2 = scores["models"]["W2_best_valid_epoch24"]
    if abs(float(w2["test_ap_official"]) - float(C.W2["test_ap_official"])) > 1e-9:
        raise RuntimeError("historical W2 Test AP drifted")
    if w2["checkpoint_sha256"] != C.W2["checkpoint_best_sha256"]:
        raise RuntimeError("historical W2 checkpoint identity drifted")

    best_valid = float(run["best_valid_ap"])
    epoch40_valid = run.get("epoch40_valid_ap")
    test_ap = float(top["test_ap_official"])
    test_bce = float(top["test_bce"])
    valid_delta = best_valid - float(C.W2["best_valid_ap"])
    test_delta = test_ap - float(C.W2["test_ap_official"])

    high = dict(scores["high_support"])
    low = dict(scores["low_support"])
    if int(high.get("n_tasks", 0)) != int(C.WORK_TARGET["high_support_tasks_historical"]):
        raise RuntimeError(
            f"high-support task count {high.get('n_tasks')} != "
            f"{C.WORK_TARGET['high_support_tasks_historical']}"
        )
    decision = _decision(
        best_valid=best_valid,
        test_ap=test_ap,
        high_support=high,
        work_target=C.WORK_TARGET,
    )

    train_eval = run.get("train_eval_final") or {}
    epoch_seconds = float(run.get("epoch_seconds_mean") or 0.0)
    gpu_hours = float(run.get("gpu_hours") or 0.0)
    w2_hours = float(C.W2["gpu_hours"])
    comparison = {
        "schema": "lcrc-pcba-toprank-comparison-v1",
        "round_id": C.ROUND_ID,
        "generated_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "test_role": C.TEST_ROLE,
        "test_scores": C.track_rel(scores_path),
        "test_scores_sha256": C.sha256_file(scores_path),
        "selection": C.track_rel(selection_path),
        "selection_sha256": C.sha256_file(selection_path),
        "run": C.track_rel(run_path),
        "run_sha256": C.sha256_file(run_path),
        "baseline": {
            "name": "W2_best_valid_epoch24",
            "parameters": int(C.W2["parameters"]),
            "best_epoch": int(C.W2["best_epoch"]),
            "best_valid_ap": float(C.W2["best_valid_ap"]),
            "test_ap": float(C.W2["test_ap_official"]),
            "test_bce": float(C.W2["test_bce"]),
            "checkpoint_sha256": C.W2["checkpoint_best_sha256"],
        },
        "toprank": {
            "parameters": int(run["parameters"]),
            "best_epoch": int(run["best_epoch"]),
            "best_valid_ap": best_valid,
            "epoch40_valid_ap": None if epoch40_valid is None else float(epoch40_valid),
            "test_ap": test_ap,
            "test_ap_independent": float(top["test_ap_independent"]),
            "test_bce": test_bce,
            "train_ap40": None if train_eval.get("ap") is None else float(train_eval["ap"]),
            "train_bce40": None if train_eval.get("bce") is None else float(train_eval["bce"]),
            "lambda_rank": float(run["calibration"]["lambda_rank"]),
            "g_bce_median": float(run["calibration"]["g_bce_median"]),
            "g_rank_median": float(run["calibration"]["g_rank_median"]),
            "ratio_raw": float(run["calibration"]["ratio_raw"]),
            "lambda_clipped": bool(run["calibration"]["clipped"]),
            "calibration_sha256": run["calibration"]["sha256"],
            "checkpoint_sha256": run["best_checkpoint_sha256"],
        },
        "deltas": {
            "valid": float(valid_delta),
            "test": float(test_delta),
            "test_bce": float(test_bce) - float(C.W2["test_bce"]),
        },
        "test_per_task": {
            "scored_tasks": int(scores["scored_tasks"]),
            "macro_delta_from_per_task": float(scores["macro_test_delta"]),
            "tasks_improved": int(scores["comparisons"]["test_TopRank_minus_W2"]["tasks_improved"]),
            "tasks_degraded": int(scores["comparisons"]["test_TopRank_minus_W2"]["tasks_degraded"]),
            "tasks_scored_both": int(scores["comparisons"]["test_TopRank_minus_W2"]["tasks_scored_both"]),
        },
        "support_strata": scores["support_strata"],
        "high_support": high,
        "low_support": low,
        "dominant_tasks": scores["dominant_tasks"],
        "valid_test_direction": scores["valid_test_direction"],
        "label_stats": scores["label_stats"],
        "w2_reference_reproduction": scores["w2_reference_reproduction"],
        "work_target": dict(C.WORK_TARGET),
        "engineering_signal": bool(decision["engineering_signal"]),
        "decision": decision,
        "extra_training_time": {
            "toprank_gpu_hours": gpu_hours,
            "w2_gpu_hours": w2_hours,
            "delta_gpu_hours": gpu_hours - w2_hours,
            "ratio": gpu_hours / w2_hours if w2_hours > 0 else None,
            "toprank_epoch_seconds_mean": epoch_seconds,
            "budget_target_gpu_hours": float(C.BUDGET["gpu_target_hours"]),
            "budget_hard_gpu_hours": float(C.BUDGET["gpu_hard_hours"]),
            "within_target": bool(gpu_hours <= float(C.BUDGET["gpu_target_hours"])),
            "within_hard_cap": bool(gpu_hours <= float(C.BUDGET["gpu_hard_hours"])),
        },
        "test_exposure_history": scores["test_exposure_history"],
        "descriptive_only": True,
    }
    C.write_json(comparison_path, comparison)

    decision_payload = {
        "schema": "lcrc-pcba-toprank-decision-v1",
        "round_id": C.ROUND_ID,
        "generated_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        **decision,
        "numbers": {
            "w2_best_valid_ap": float(C.W2["best_valid_ap"]),
            "w2_test_ap": float(C.W2["test_ap_official"]),
            "toprank_best_valid_ap": best_valid,
            "toprank_epoch40_valid_ap": None if epoch40_valid is None else float(epoch40_valid),
            "toprank_test_ap": test_ap,
            "toprank_test_bce": test_bce,
            "valid_delta": float(valid_delta),
            "test_delta": float(test_delta),
        },
        "high_support": high,
        "engineering_signal": bool(decision["engineering_signal"]),
        "comparison": C.track_rel(comparison_path),
        "comparison_sha256": C.sha256_file(comparison_path),
        "note": decision["note"],
    }
    C.write_json(decision_path, decision_payload)

    _write_per_task_deltas(per_task_path, scores)
    C.log(
        f"[compare] valid {best_valid:.9f} ({valid_delta:+.9f}) "
        f"test {test_ap:.9f} ({test_delta:+.9f}) decision={decision['label']}"
    )
    return 0


def _write_per_task_deltas(path: Path, scores: Mapping[str, Any]) -> None:
    fields = [
        "task",
        "test_positive",
        "support_stratum",
        "test_ap_toprank",
        "test_ap_w2",
        "delta",
        "valid_positive",
        "valid_ap_toprank",
        "valid_ap_w2",
        "valid_delta",
    ]
    rows = scores["per_task"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fields})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    sel = sub.add_parser("select")
    sel.add_argument("--run", required=True)
    sel.add_argument("--out", default=str(SELECTION_PATH))
    cmp = sub.add_parser("compare")
    cmp.add_argument("--run", required=True)
    cmp.add_argument("--selection", default=str(SELECTION_PATH))
    cmp.add_argument("--scores", default=str(SCORES_PATH))
    cmp.add_argument("--out", default=str(COMPARISON_PATH))
    cmp.add_argument("--decision", default=str(DECISION_PATH))
    cmp.add_argument("--per-task", default=str(PER_TASK_DELTA_PATH))
    args = parser.parse_args(argv)
    if args.mode == "select":
        return run_select(Path(args.run), Path(args.out))
    if args.mode == "compare":
        return run_compare(
            Path(args.run),
            Path(args.selection),
            Path(args.scores),
            Path(args.out),
            Path(args.decision),
            Path(args.per_task),
        )
    raise KeyError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
