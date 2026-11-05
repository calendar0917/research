"""Phase C: deterministic local reduction of the pulled artifacts.

Reads the Phase-A baseline replay and the two Phase-B arm records, then writes

* ``analysis/per_task_deltas.csv``   -- per-task AP, deltas and improvement flags
* ``analysis/readout_curve.csv``     -- merged both-arm curve with cross-head deltas
* ``analysis/readout_comparison.json`` -- the paired comparison and decision label

No GPU work, no network access, no Test labels.  Every number here is derived
from artifacts whose SHA-256 is recorded in the JSON.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fr_common as C  # noqa: E402
import fr_metrics as FM  # noqa: E402

DELTA_FIELDS = [
    "task",
    "scorable",
    "valid_observed",
    "valid_positive",
    "valid_negative",
    "valid_prevalence",
    "stratum",
    "ap_frozen",
    "ap_linear_epoch8",
    "ap_nonlinear_epoch8",
    "delta_linear",
    "delta_nonlinear",
    "delta_nonlinear_minus_linear",
    "improved_linear",
    "improved_nonlinear",
    "improved_nonlinear_over_linear",
]

CURVE_FIELDS = [
    "epoch",
    "arm",
    "evaluated",
    "train_bce_full",
    "train_ap",
    "valid_ap",
    "n_valid_tasks_scored",
    "delta_vs_ap0",
    "train_minus_valid_ap",
    "delta_vs_other_head",
    "train_seconds",
    "eval_seconds",
    "cumulative_seconds",
    "gpu_peak_allocated_bytes",
    "gpu_peak_reserved_bytes",
    "logits_bitwise_equal_frozen",
]


def _require(path: Path) -> Any:
    if not path.exists():
        raise RuntimeError(f"missing required artifact: {path}")
    return C.read_json(path)


def _stratum(n_positive: int) -> str:
    if int(n_positive) <= 100:
        return "sparse_le_100"
    if int(n_positive) <= 1000:
        return "medium_101_1000"
    return "ample_gt_1000"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            out = {}
            for key in fields:
                value = row.get(key)
                if value is None:
                    value = ""
                elif isinstance(value, (np.floating, np.integer, np.bool_)):
                    value = value.item()
                out[key] = value
            writer.writerow(out)
    return C.sha256_file(path)


def _arm_epoch8(arm_json: Mapping[str, Any]) -> dict[str, Any]:
    metrics = arm_json["epoch_metrics"]
    return dict(metrics[str(C.MAIN_EPOCH)])


def _validate_arm_csvs(arm_json: Mapping[str, Any]) -> dict[str, Any]:
    curve_path = C.REPO / arm_json["curve"]
    task_path = C.REPO / arm_json["per_task_ap"]
    for path, expected in ((curve_path, arm_json["curve_sha256"]), (task_path, arm_json["per_task_ap_sha256"])):
        if not path.exists():
            raise RuntimeError(f"missing arm artifact: {path}")
        if C.sha256_file(path) != expected:
            raise RuntimeError(f"arm artifact hash mismatch: {path}")
    rows = [row for row in csv.DictReader(open(task_path, encoding="utf-8")) if int(row["epoch"]) == C.MAIN_EPOCH]
    expected = arm_json["epoch_metrics"][str(C.MAIN_EPOCH)]["valid_per_task"]
    if len(rows) != C.N_TASKS:
        raise RuntimeError(f"epoch-8 per-task rows {len(rows)} != {C.N_TASKS}")
    max_diff = 0.0
    for row, reference in zip(rows, expected):
        if int(row["task"]) >= C.N_TASKS:
            raise RuntimeError("task id out of range in per_task_ap.csv")
        if reference is None:
            if row["ap"] != "":
                raise RuntimeError(f"task {row['task']} should have no AP at epoch 8")
        else:
            max_diff = max(max_diff, abs(float(row["ap"]) - float(reference)))
    if max_diff > 1e-12:
        raise RuntimeError(f"per_task_ap.csv epoch-8 rows disagree with arm.json (max diff {max_diff})")
    return {"curve_sha256": C.sha256_file(curve_path), "per_task_csv_sha256": C.sha256_file(task_path), "max_abs_diff": max_diff}


def build(analysis_dir: Path = C.ANALYSIS, *, write: bool = True) -> dict[str, Any]:
    baseline = _require(analysis_dir / "frozen_baseline_replay.json")
    extraction = _require(analysis_dir / "feature_extraction_audit.json")
    heads = _require(C.HEADS_RUNS / "run.json")

    arm_jsons = {arm: _require(C.HEADS_RUNS / arm / "arm.json") for arm in C.ARM_ORDER}
    csv_audit = {arm: _validate_arm_csvs(arm_jsons[arm]) for arm in C.ARM_ORDER}
    # Both arms must describe the same Validation label coverage.
    coverage = {
        arm: [
            (int(r['observed']), int(r['positive']), int(r['negative']))
            for r in csv.DictReader(open(C.REPO / arm_jsons[arm]['per_task_ap'], encoding='utf-8'))
            if int(r['epoch']) == C.MAIN_EPOCH
        ]
        for arm in C.ARM_ORDER
    }
    if coverage['L'] != coverage['N']:
        raise RuntimeError('the two arms score different Validation label coverage')
    coverage_identical = True

    base_per_task = list(baseline["valid"]["per_task_ap"])
    if len(base_per_task) != C.N_TASKS:
        raise RuntimeError(f"baseline per-task AP has {len(base_per_task)} entries")
    n_scorable = sum(1 for v in base_per_task if v is not None)
    if n_scorable != C.N_SCORABLE_TASKS_EXPECTED:
        raise RuntimeError(f"baseline scorable tasks {n_scorable} != {C.N_SCORABLE_TASKS_EXPECTED}")

    l8 = _arm_epoch8(arm_jsons["L"])
    n8 = _arm_epoch8(arm_jsons["N"])
    ap_l = float(l8["valid_ap"])
    ap_n = float(n8["valid_ap"])
    d_l = ap_l - float(C.AP0)
    d_n = ap_n - float(C.AP0)
    d_nl = ap_n - ap_l

    per_task_l = list(l8["valid_per_task"])
    per_task_n = list(n8["valid_per_task"])
    delta_l = FM.delta_summary(base_per_task, per_task_l)
    delta_n = FM.delta_summary(base_per_task, per_task_n)
    delta_nl = FM.delta_summary(per_task_l, per_task_n)

    # Per-task table (all 128 tasks; task 45 has no positive Validation label).
    task_rows: list[dict[str, Any]] = []
    per_task_csv_l = {
        int(row["task"]): row
        for row in csv.DictReader(open(C.REPO / arm_jsons["L"]["per_task_ap"], encoding="utf-8"))
        if int(row["epoch"]) == C.MAIN_EPOCH
    }
    for t in range(C.N_TASKS):
        row = per_task_csv_l[t]
        observed = int(row["observed"])
        positive = int(row["positive"])
        negative = int(row["negative"])
        scorable = base_per_task[t] is not None and per_task_l[t] is not None and per_task_n[t] is not None
        task_rows.append(
            {
                "task": int(t),
                "scorable": bool(scorable),
                "valid_observed": observed,
                "valid_positive": positive,
                "valid_negative": negative,
                "valid_prevalence": float(positive / observed) if observed else None,
                "stratum": _stratum(positive),
                "ap_frozen": base_per_task[t],
                "ap_linear_epoch8": per_task_l[t],
                "ap_nonlinear_epoch8": per_task_n[t],
                "delta_linear": None if not scorable else float(per_task_l[t]) - float(base_per_task[t]),
                "delta_nonlinear": None if not scorable else float(per_task_n[t]) - float(base_per_task[t]),
                "delta_nonlinear_minus_linear": None if not scorable else float(per_task_n[t]) - float(per_task_l[t]),
                "improved_linear": None if not scorable else bool(float(per_task_l[t]) > float(base_per_task[t])),
                "improved_nonlinear": None if not scorable else bool(float(per_task_n[t]) > float(base_per_task[t])),
                "improved_nonlinear_over_linear": None
                if not scorable
                else bool(float(per_task_n[t]) > float(per_task_l[t])),
            }
        )

    def _strata(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in ("sparse_le_100", "medium_101_1000", "ample_gt_1000"):
            group = [r for r in rows if r["stratum"] == name and r["scorable"]]
            deltas = [float(r[key]) for r in group]
            out[name] = {
                "n_tasks": len(group),
                "mean_delta": float(np.mean(deltas)) if deltas else None,
                "median_delta": float(np.median(deltas)) if deltas else None,
                "n_improved": int(sum(1 for d in deltas if d > 0)),
                "sum_delta": float(np.sum(deltas)) if deltas else None,
            }
        return out

    # Merged curve with the cross-head difference at every evaluation epoch.
    merged_curve: list[dict[str, Any]] = []
    for arm in C.ARM_ORDER:
        other = "N" if arm == "L" else "L"
        own = arm_jsons[arm]["epoch_metrics"]
        other_metrics = arm_jsons[other]["epoch_metrics"]
        curve_rows = {
            int(row["epoch"]): row
            for row in csv.DictReader(open(C.REPO / arm_jsons[arm]["curve"], encoding="utf-8"))
        }
        for epoch in sorted(int(k) for k in own):
            row = curve_rows[int(epoch)]
            merged_curve.append(
                {
                    "epoch": int(epoch),
                    "arm": arm,
                    "evaluated": True,
                    "train_bce_full": float(own[str(epoch)]["train_bce_full"]),
                    "train_ap": float(own[str(epoch)]["train_ap"]),
                    "valid_ap": float(own[str(epoch)]["valid_ap"]),
                    "n_valid_tasks_scored": int(own[str(epoch)]["n_valid_tasks_scored"]),
                    "delta_vs_ap0": float(own[str(epoch)]["delta_vs_ap0"]),
                    "train_minus_valid_ap": float(own[str(epoch)]["train_minus_valid_ap"]),
                    "delta_vs_other_head": float(own[str(epoch)]["valid_ap"])
                    - float(other_metrics[str(epoch)]["valid_ap"]),
                    "train_seconds": float(row["train_seconds"]),
                    "eval_seconds": float(row["eval_seconds"]),
                    "cumulative_seconds": float(row["cumulative_seconds"]),
                    "gpu_peak_allocated_bytes": int(row["gpu_peak_allocated_bytes"]),
                    "gpu_peak_reserved_bytes": int(row["gpu_peak_reserved_bytes"]),
                    "logits_bitwise_equal_frozen": bool(own[str(epoch)]["logits_bitwise_equal_frozen"]),
                }
            )
    merged_curve.sort(key=lambda r: (int(r["epoch"]), str(r["arm"])))

    status_l = str(arm_jsons["L"].get("status"))
    status_n = str(arm_jsons["N"].get("status"))
    gates_ok = bool(
        extraction.get("all_passed")
        and heads.get("all_passed")
        and baseline.get("all_passed")
        and status_l == "COMPLETED"
        and status_n == "COMPLETED"
        and len(merged_curve) == 2 * len(C.EVAL_EPOCHS)
    )
    decision = C.decide_label(
        status_l=status_l,
        status_n=status_n,
        gates_ok=gates_ok,
        d_l=d_l,
        d_n=d_n,
        d_nl=d_nl,
        improved_l=int(delta_l["n_improved"]),
        improved_n=int(delta_n["n_improved"]),
        n_scorable=int(n_scorable),
    )

    gap_frozen = float(baseline["train_minus_valid_ap"])
    gap_l = float(l8["train_minus_valid_ap"])
    gap_n = float(n8["train_minus_valid_ap"])

    preflight = None
    preflight_path = C.PREFLIGHT_RUN / "preflight.json"
    if preflight_path.exists():
        preflight = C.read_json(preflight_path)

    extraction_hours = float(extraction.get("gpu_hours") or 0.0)
    heads_hours = float(heads.get("gpu_hours") or 0.0)
    preflight_hours = float((preflight or {}).get("gpu_hours") or 0.0)
    total_hours = float(extraction_hours + heads_hours + preflight_hours)

    result = {
        "schema": "lcrc-pcba-frozen-readout-comparison-v1",
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "phase": "C_paired_comparison",
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "inputs": {
            "frozen_baseline_replay": {
                "path": C.track_rel(analysis_dir / "frozen_baseline_replay.json"),
                "sha256": C.sha256_file(analysis_dir / "frozen_baseline_replay.json"),
            },
            "feature_extraction_audit": {
                "path": C.track_rel(analysis_dir / "feature_extraction_audit.json"),
                "sha256": C.sha256_file(analysis_dir / "feature_extraction_audit.json"),
            },
            "heads_run": {
                "path": C.track_rel(C.HEADS_RUNS / "run.json"),
                "sha256": C.sha256_file(C.HEADS_RUNS / "run.json"),
            },
            "arms": {
                arm: {
                    "path": C.track_rel(C.HEADS_RUNS / arm / "arm.json"),
                    "sha256": C.sha256_file(C.HEADS_RUNS / arm / "arm.json"),
                    "artifacts": csv_audit[arm],
                }
                for arm in C.ARM_ORDER
            },
        },
        "gates": {
            "extraction_all_passed": bool(extraction.get("all_passed")),
            "baseline_replay_all_passed": bool(baseline.get("all_passed")),
            "heads_all_passed": bool(heads.get("all_passed")),
            "both_arms_completed": status_l == "COMPLETED" and status_n == "COMPLETED",
            "all_four_evaluation_epochs_present": len(merged_curve) == 2 * len(C.EVAL_EPOCHS),
            "same_validation_label_coverage": bool(coverage_identical),
            "epoch0_identity_both_arms": bool(
                all(arm_jsons[a]["epoch_metrics"]["0"]["logits_bitwise_equal_frozen"] for a in C.ARM_ORDER)
            ),
            "base_model_unchanged": bool(heads.get("gates", {}).get("base_model_unchanged")),
            "same_row_order_both_arms": bool(heads.get("gates", {}).get("shared_row_plan_identical")),
            "test_never_evaluated": True,
            "gates_ok_for_decision": bool(gates_ok),
        },
        "baseline": {
            "ap0": float(C.AP0),
            "train_ap0_reference": float(C.TRAIN_AP0),
            "valid_ap_replayed": float(baseline["valid"]["official_ap"]),
            "valid_ap_abs_delta_vs_ap0": float(baseline["valid"]["abs_delta_vs_ap0"]),
            "train_ap_replayed": float(baseline["train"]["macro_ap"]),
            "train_ap_abs_delta_vs_reference": float(baseline["train"]["abs_delta_vs_reference"]),
            "train_masked_bce": float(baseline["train"]["masked_bce"]),
            "valid_masked_bce": float(baseline["valid"]["masked_bce"]),
            "n_scorable_valid_tasks": int(n_scorable),
            "n_unscorable_tasks": [int(t) for t, v in enumerate(base_per_task) if v is None],
            "rho_in_dim": int(extraction["model"]["rho_dimension"]["rho_input_dim"]),
            "n_zero_variance_rho_columns": int(extraction["standardizer"]["n_zero_variance_columns"]),
        },
        "main_comparison": {
            "metric": "official OGB Validation macro AP at fixed epoch 8",
            "epoch": int(C.MAIN_EPOCH),
            "AP_L": ap_l,
            "AP_N": ap_n,
            "D_L": float(d_l),
            "D_N": float(d_n),
            "D_NL": float(d_nl),
            "thresholds": {
                "improvement_floor": float(C.DECISION["improvement_floor"]),
                "task_majority": int(C.DECISION["task_majority"]),
                "nonlinear_extra_floor": float(C.DECISION["nonlinear_extra_floor"]),
            },
            "linear_vs_frozen": delta_l,
            "nonlinear_vs_frozen": delta_n,
            "nonlinear_vs_linear": delta_nl,
            "auxiliary_best_valid_ap": {
                arm: {
                    "best_valid_ap": float(arm_jsons[arm]["primary"]["auxiliary_best_valid_ap"]),
                    "best_epoch": int(arm_jsons[arm]["primary"]["auxiliary_best_epoch"]),
                    "primary_is_best": bool(arm_jsons[arm]["primary"]["auxiliary_best_is_primary_epoch"]),
                }
                for arm in C.ARM_ORDER
            },
            "auxiliary_note": "best-Validation AP is auxiliary and does not replace fixed epoch 8",
        },
        "train_valid_gap": {
            "frozen": {
                "train_ap": float(baseline["train"]["macro_ap"]),
                "valid_ap": float(baseline["valid"]["official_ap"]),
                "train_minus_valid": gap_frozen,
            },
            "L_epoch8": {
                "train_ap": float(l8["train_ap"]),
                "valid_ap": ap_l,
                "train_minus_valid": gap_l,
                "gap_change_vs_frozen": float(gap_l - gap_frozen),
            },
            "N_epoch8": {
                "train_ap": float(n8["train_ap"]),
                "valid_ap": ap_n,
                "train_minus_valid": gap_n,
                "gap_change_vs_frozen": float(gap_n - gap_frozen),
            },
            "larger_gap_expansion": "N" if abs(gap_n - gap_frozen) > abs(gap_l - gap_frozen) else "L",
        },
        "strata_by_valid_positive_count": {
            "definition": {
                "sparse_le_100": "Validation positives <= 100",
                "medium_101_1000": "101..1000",
                "ample_gt_1000": "> 1000",
            },
            "linear_vs_frozen": _strata(task_rows, "delta_linear"),
            "nonlinear_vs_frozen": _strata(task_rows, "delta_nonlinear"),
            "nonlinear_vs_linear": _strata(task_rows, "delta_nonlinear_minus_linear"),
        },
        "curve": merged_curve,
        "gpu_accounting": {
            "extraction_gpu_hours": extraction_hours,
            "heads_gpu_hours": heads_hours,
            "preflight_gpu_hours": preflight_hours,
            "preflight_recorded": preflight is not None,
            "total_gpu_hours": total_hours,
            "budget_gpu_hours": float(C.BUDGET["gpu_hours_total"]),
            "within_budget": bool(total_hours <= float(C.BUDGET["gpu_hours_total"])),
            "peak_allocated_bytes": {
                "extraction": int((extraction.get("gpu_peak_during_extraction") or {}).get("peak_allocated_bytes", 0)),
                "heads": int(max(
                    (arm_jsons[a].get("gpu_peak_allocated_bytes") or 0) for a in C.ARM_ORDER
                )),
            },
            "peak_reserved_bytes": {
                "extraction": int((extraction.get("gpu_peak_during_extraction") or {}).get("peak_reserved_bytes", 0)),
                "heads": int(max(
                    (arm_jsons[a].get("gpu_peak_reserved_bytes") or 0) for a in C.ARM_ORDER
                )),
            },
        },
        "reproducibility": {
            "source_checkpoint": C.track_rel(C.SOURCE["checkpoint"]),
            "source_checkpoint_sha256": C.SOURCE["checkpoint_sha256"],
            "feature_cache_manifest_sha256": heads["cache"]["manifest_sha256"],
            "feature_cache_manifest_path": heads["cache"]["manifest"],
            "standardized_train_sha256": heads["cache"]["standardizer"]["train_standardized_sha256"],
            "standardized_valid_sha256": heads["cache"]["standardizer"]["valid_standardized_sha256"],
            "standardizer_reproduces_phase_a": bool(
                heads["cache"]["standardizer"]["train_standardized_sha256_matches_phase_a"]
                and heads["cache"]["standardizer"]["valid_standardized_sha256_matches_phase_a"]
            ),
            "row_plan_global_sha256": heads["row_plan"]["global_plan_sha256"],
            "row_plan_shared": bool(heads["gates"]["shared_row_plan_identical"]),
            "base_model_fingerprint": heads["base_model"]["parameter_fingerprint"],
            "frozen_logits_unchanged": bool(heads["gates"]["frozen_logits_unchanged"]),
            "seeds": {"init": C.SEED, "row_order": C.SEED},
            "single_seed_limitation": "one seed per head; no seed replication in this round",
        },
        "decision": decision,
        "interpretation_guardrails": C.DECISION["interpretation_guardrails"],
        "test_evaluated": False,
        "per_task_deltas_csv": C.track_rel(analysis_dir / "per_task_deltas.csv"),
        "readout_curve_csv": C.track_rel(analysis_dir / "readout_curve.csv"),
    }

    if write:
        delta_sha = _write_csv(analysis_dir / "per_task_deltas.csv", task_rows, DELTA_FIELDS)
        curve_sha = _write_csv(analysis_dir / "readout_curve.csv", merged_curve, CURVE_FIELDS)
        result["per_task_deltas_csv_sha256"] = delta_sha
        result["readout_curve_csv_sha256"] = curve_sha
        C.write_json(analysis_dir / "readout_comparison.json", result)
    C.log(
        f"decision={decision['label']} nonlinear_extra={decision['nonlinear_extra_signal']} "
        f"AP_L={ap_l:.10f} AP_N={ap_n:.10f} D_L={d_l:+.6f} D_N={d_n:+.6f} D_NL={d_nl:+.6f} "
        f"improved L/N={delta_l['n_improved']}/{delta_n['n_improved']}"
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", default=str(C.ANALYSIS))
    args = parser.parse_args(argv)
    result = build(Path(args.analysis_dir))
    return 0 if result["gates"]["gates_ok_for_decision"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
