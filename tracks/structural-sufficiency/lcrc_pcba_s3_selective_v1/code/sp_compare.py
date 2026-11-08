"""Selection manifest, performance comparison, GPU ledger and decision.

``select``  writes ``analysis/selected_checkpoint.json`` from the SP run's
            Validation curve only (no Test access, declared explicitly).
``compare`` writes ``analysis/performance_comparison.json``,
            ``analysis/gpu_ledger.json`` and ``analysis/decision.json`` from the
            run artifacts and (if present) the frozen Test scores.

Both subcommands are deterministic functions of on-disk artifacts; nothing is
re-computed on the GPU and nothing here can change a recipe or a checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import sp_common as C  # noqa: E402

SP_RUN_DIR = C.RUNS / "SP"
SP_RUN = SP_RUN_DIR / "run.json"
SELECTION = C.ANALYSIS / "selected_checkpoint.json"
SCORES = C.ANALYSIS / "test_scores.json"
COMPARISON = C.ANALYSIS / "performance_comparison.json"
LEDGER = C.ANALYSIS / "gpu_ledger.json"
DECISION = C.ANALYSIS / "decision.json"

#: Historical, read-only references (all selected on Validation AP).
W2_REF = {
    "model": "W2",
    "parameters": 994_324,
    "best_epoch": int(C.W2["best_epoch"]),
    "best_valid_ap": float(C.W2["best_valid_ap"]),
    "ap40": float(C.W2["ap40"]),
    "train_ap40": float(C.W2["train_ap40"]),
    "train_bce40": float(C.W2["train_bce40"]),
    "valid_bce40": float(C.W2["valid_bce40"]),
    "gpu_hours": float(C.W2["gpu_hours"]),
    "epoch_seconds_mean": float(C.W2["epoch_seconds_mean"]),
    "checkpoint": C.track_rel(C.W2_CHECKPOINT_BEST),
    "checkpoint_sha256": C.W2["best_checkpoint_sha256"],
}
S3_REF = {
    "model": "S3",
    "parameters": int(C.S3_PARAMETERS),
    "new_parameters": int(C.S3_ADDED),
    "best_epoch": int(C.S3["best_epoch"]),
    "best_valid_ap": float(C.S3["best_valid_ap"]),
    "ap40": float(C.S3["ap40"]),
    "train_ap40": float(C.read_json(Path(C.S3["run_dir"]) / "run.json")["train_eval_final"]["ap"]),
    "train_bce40": float(C.read_json(Path(C.S3["run_dir"]) / "run.json")["train_eval_final"]["bce"]),
    "gpu_hours": float(C.S3["gpu_hours"]),
    "epoch_seconds_mean": float(C.S3["epoch_seconds_mean"]),
    "checkpoint": C.track_rel(C.S3_CHECKPOINT_BEST),
    "checkpoint_sha256": C.S3["best_checkpoint_sha256"],
}


def _read_curve(path: Path) -> list[dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def run_select() -> dict[str, Any]:
    run = C.read_json(SP_RUN)
    if run.get("test_evaluated") is not False:
        raise RuntimeError("selection requires a run that never touched Test")
    if run["status"] != "COMPLETED":
        raise RuntimeError(f"selection requires a COMPLETED run, got {run['status']}")
    best_epoch = int(run["best_epoch"])
    best_ap = float(run["best_valid_ap"])
    curve = _read_curve(SP_RUN_DIR / "train_curve.csv")
    ap_by_epoch = {int(r["epoch"]): _float(r["valid_ap"]) for r in curve}
    if int(max(ap_by_epoch, key=lambda e: (ap_by_epoch[e], -e))) != best_epoch:
        raise RuntimeError("recorded best_epoch is not the argmax of the Validation curve")
    if abs(float(ap_by_epoch[best_epoch]) - best_ap) > 1e-12:
        raise RuntimeError("recorded best_valid_ap does not match the curve")
    selection = {
        "schema": "lcrc-pcba-s3-selective-selection-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "test_read": False,
        "selection_uses_test": False,
        "test_role": "EXPOSED_TEST_PERFORMANCE_BENCHMARK",
        "selection_rule": run["selection_rule"],
        "selection_source": C.track_rel(SP_RUN),
        "models": {
            "SP": {
                "best_epoch": best_epoch,
                "best_valid_ap": best_ap,
                "ap40": _float(run.get("epoch40_ap")),
                "valid_bce40": _float(run.get("epoch40_valid_bce")),
                "path": run["best_checkpoint"],
                "sha256": run["best_checkpoint_sha256"],
                "epoch40_path": run["epoch40_checkpoint"],
                "epoch40_sha256": run["epoch40_checkpoint_sha256"],
                "checkpoint_kind": "best_valid_ap_selected",
                "parameters": int(run["parameters"]),
                "new_parameters": int(run["new_parameters"]),
            }
        },
        "curve": {
            "epochs": len(curve),
            "ap_by_epoch": {str(k): v for k, v in sorted(ap_by_epoch.items())},
            "argmax_epoch": best_epoch,
            "argmax_ap": best_ap,
        },
        "note": (
            "epoch chosen by argmax official Validation AP over epochs 1..40 with the earliest epoch "
            "winning exact ties; the official Test score was not read at selection time"
        ),
    }
    C.write_json(SELECTION, selection)
    C.log(f"[select] SP best epoch {best_epoch} valid AP {best_ap:.9f}")
    return selection


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------


def _task_stats(
    a: Sequence[Any],
    b: Sequence[Any],
    support: Sequence[int],
    *,
    min_support: int,
    max_support: int | None = None,
) -> dict[str, Any]:
    deltas = []
    improved = degraded = 0
    for x, y, s in zip(a, b, support):
        if x is None or y is None or int(s) < int(min_support):
            continue
        if max_support is not None and int(s) >= int(max_support):
            continue
        d = float(x) - float(y)
        deltas.append(d)
        improved += int(d > 0)
        degraded += int(d < 0)
    if not deltas:
        return {"n_tasks": 0}
    arr = np.asarray(deltas, dtype=np.float64)
    return {
        "n_tasks": int(arr.size),
        "mean_delta": float(arr.mean()),
        "median_delta": float(np.median(arr)),
        "improved": int(improved),
        "degraded": int(degraded),
        "min_delta": float(arr.min()),
        "max_delta": float(arr.max()),
        "share_improved": float(improved / arr.size),
        "abs_lt_0.01": int((np.abs(arr) < 0.01).sum()),
    }


def _curve_summary(curve: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ap = [float(r["valid_ap"]) for r in curve if r["valid_ap"] not in ("", None)]
    ent = [_float(r.get("attention_entropy_ratio")) for r in curve]
    ent = [x for x in ent if x is not None]
    onehot = [_float(r.get("attention_near_one_hot_fraction")) for r in curve]
    onehot = [x for x in onehot if x is not None]
    sel = [_float(r.get("selective_over_residual_total")) for r in curve]
    sel = [x for x in sel if x is not None]
    wsum = [_float(r.get("attention_weight_sum_error")) for r in curve]
    wsum = [x for x in wsum if x is not None]
    return {
        "epochs": len(curve),
        "valid_ap_first": ap[0] if ap else None,
        "valid_ap_last": ap[-1] if ap else None,
        "valid_ap_max": max(ap) if ap else None,
        "valid_ap_min": min(ap) if ap else None,
        "attention_entropy_ratio_first": ent[0] if ent else None,
        "attention_entropy_ratio_last": ent[-1] if ent else None,
        "attention_entropy_ratio_mean": float(np.mean(ent)) if ent else None,
        "attention_entropy_ratio_min": float(np.min(ent)) if ent else None,
        "attention_near_one_hot_fraction_first": onehot[0] if onehot else None,
        "attention_near_one_hot_fraction_last": onehot[-1] if onehot else None,
        "selective_share_of_residual_first": sel[0] if sel else None,
        "selective_share_of_residual_last": sel[-1] if sel else None,
        "selective_share_of_residual_mean": float(np.mean(sel)) if sel else None,
        "attention_weight_sum_error_max": float(np.max(wsum)) if wsum else None,
        "all_weights_normalised": bool(wsum and max(wsum) < 1e-5),
        "curve": [
            {
                "epoch": int(r["epoch"]),
                "valid_ap": _float(r["valid_ap"]),
                "valid_bce": _float(r["valid_bce"]),
                "train_bce": _float(r["train_bce"]),
                "learning_rate": _float(r["learning_rate"]),
                "epoch_seconds": _float(r["epoch_seconds"]),
                "step_ms_p50": _float(r["step_ms_p50"]),
                "selective_over_residual_total": _float(r.get("selective_over_residual_total")),
                "attention_entropy_ratio": _float(r.get("attention_entropy_ratio")),
                "attention_near_one_hot_fraction": _float(r.get("attention_near_one_hot_fraction")),
                "selective_param_grad_norm": _float(r.get("selective_param_grad_norm")),
            }
            for r in curve
        ],
    }


def run_compare(*, extra_hours: float = 0.0) -> dict[str, Any]:
    run = C.read_json(SP_RUN)
    selection = C.read_json(SELECTION)
    curve = _read_curve(SP_RUN_DIR / "train_curve.csv")
    scores = C.read_json(SCORES) if SCORES.is_file() else None
    if scores is not None and bool(scores.get("test_label_artifact", {}).get("read_after_freeze")) is not True:
        raise RuntimeError("Test scores exist but were not produced after a freeze")

    sp_epoch = int(selection["models"]["SP"]["best_epoch"])
    sp_valid = float(selection["models"]["SP"]["best_valid_ap"])
    sp_ap40 = _float(selection["models"]["SP"].get("ap40"))
    test_ap = {}
    per_task: dict[str, Any] = {}
    if scores is not None:
        for label, rec in scores["models"].items():
            test_ap[label] = float(rec["test_ap_official"])
            per_task[label] = rec["test_per_task_ap"]

    models: dict[str, Any] = {
        "W2": dict(W2_REF),
        "S3": dict(S3_REF),
        "SP": {
            "model": "SP",
            "parameters": int(run["parameters"]),
            "new_parameters": int(run["new_parameters"]),
            "best_epoch": sp_epoch,
            "best_valid_ap": sp_valid,
            "ap40": sp_ap40,
            "valid_bce40": _float(run.get("epoch40_valid_bce")),
            "train_ap40": _float((run.get("train_eval_final") or {}).get("ap")),
            "train_bce40": _float((run.get("train_eval_final") or {}).get("bce")),
            "gpu_hours": float(run["gpu_hours"]),
            "epoch_seconds_mean": float(run["epoch_seconds_mean"]),
            "checkpoint": run["best_checkpoint"],
            "checkpoint_sha256": run["best_checkpoint_sha256"],
            "epoch40_checkpoint": run["epoch40_checkpoint"],
            "epoch40_checkpoint_sha256": run["epoch40_checkpoint_sha256"],
        },
    }
    for key, label in (("W2", "W2_best_valid_epoch24"), ("S3", "S3_best_valid_epoch29"), ("SP", "SP_best_valid_selected")):
        models[key]["test_ap"] = test_ap.get(label)
        models[key]["test_bce"] = (
            float(scores["models"][label]["test_bce"]) if scores is not None and label in scores["models"] else None
        )
        models[key]["n_tasks_scored_test"] = (
            int(scores["models"][label]["n_tasks_scored"]) if scores is not None and label in scores["models"] else None
        )
        models[key]["n_tasks_scored_valid"] = (
            int(scores["models"][label]["n_tasks_scored_valid"]) if scores is not None and label in scores["models"] else None
        )
        models[key]["valid_test_ap_delta"] = (
            float(models[key]["test_ap"] - models[key]["best_valid_ap"])
            if models[key]["test_ap"] is not None
            else None
        )

    valid_gain = sp_valid - S3_REF["best_valid_ap"]
    test_gain = (models["SP"]["test_ap"] - models["S3"]["test_ap"]) if models["SP"]["test_ap"] is not None else None
    test_gain_vs_w2 = (
        (models["SP"]["test_ap"] - models["W2"]["test_ap"]) if models["SP"]["test_ap"] is not None else None
    )
    s3_test_gain_vs_w2 = (
        (models["S3"]["test_ap"] - models["W2"]["test_ap"]) if models["S3"]["test_ap"] is not None else None
    )

    task_comparison: dict[str, Any] = {}
    if per_task and "SP_best_valid_selected" in per_task and "S3_best_valid_epoch29" in per_task:
        sp_task = per_task["SP_best_valid_selected"]
        s3_task = per_task["S3_best_valid_epoch29"]
        w2_task = per_task["W2_best_valid_epoch24"]
        sp = scores["models"]["SP_best_valid_selected"]
        support_test = np.asarray(sp["test_positive"], dtype=np.int64)
        up, down, both = 0, 0, 0
        deltas = []
        for a, b in zip(sp_task, s3_task):
            if a is None or b is None:
                continue
            both += 1
            d = float(a) - float(b)
            deltas.append(d)
            up += int(d > 0)
            down += int(d < 0)
        arr = np.asarray(deltas, dtype=np.float64) if deltas else np.zeros(0)
        positive_gain = arr[arr > 0].sum() if arr.size else 0.0
        top = np.argsort(-arr)[:10].tolist() if arr.size else []
        task_comparison = {
            "test_tasks_scored_both": int(both),
            "test_tasks_improved": int(up),
            "test_tasks_degraded": int(down),
            "test_tasks_unchanged": int(both - up - down),
            "test_delta_mean": float(arr.mean()) if arr.size else None,
            "test_delta_median": float(np.median(arr)) if arr.size else None,
            "test_delta_min": float(arr.min()) if arr.size else None,
            "test_delta_max": float(arr.max()) if arr.size else None,
            "top10_positive_share_of_positive_gain": (
                float(np.sort(arr)[-10:].clip(min=0).sum() / positive_gain) if positive_gain > 0 else None
            ),
            "test_ge20": _task_stats(sp_task, s3_task, support_test, min_support=20),
            "test_ge5_lt20": _task_stats(sp_task, s3_task, support_test, min_support=5, max_support=20),
            "test_lt5": _task_stats(sp_task, s3_task, support_test, min_support=0, max_support=5),
            "test_vs_w2_ge20": _task_stats(sp_task, w2_task, support_test, min_support=20),
            "valid_ge20": _task_stats(
                sp["valid_per_task_ap"], scores["models"]["S3_best_valid_epoch29"]["valid_per_task_ap"],
                np.asarray(sp["valid_positive"], np.int64), min_support=20,
            ),
            "valid_ge5_lt20": _task_stats(
                sp["valid_per_task_ap"], scores["models"]["S3_best_valid_epoch29"]["valid_per_task_ap"],
                np.asarray(sp["valid_positive"], np.int64), min_support=5, max_support=20,
            ),
        }

    arch_path = C.ANALYSIS / "architecture_audit.json"
    arch = C.read_json(arch_path) if arch_path.is_file() else None
    cost = {
        "parameters": {k: int(v["parameters"]) for k, v in models.items()},
        "new_parameters_vs_w2": {k: int(v["parameters"] - 994_324) for k, v in models.items()},
        "s3_sp_parameter_ratio": float(C.SP_PARAMETERS / C.S3_PARAMETERS),
        "sp_new_fraction_of_s3": float(C.SP_ADDED / C.S3_PARAMETERS),
        "epoch_seconds_mean": {k: float(v["epoch_seconds_mean"]) for k, v in models.items()},
        "sp_epoch_over_s3": float(models["SP"]["epoch_seconds_mean"] / S3_REF["epoch_seconds_mean"]),
        "sp_epoch_over_w2": float(models["SP"]["epoch_seconds_mean"] / W2_REF["epoch_seconds_mean"]),
        "training_gpu_hours": {k: float(v["gpu_hours"]) for k, v in models.items()},
        "step_ms_p50_median": float(np.median([_float(r["step_ms_p50"]) for r in curve if _float(r["step_ms_p50"])])),
        "analytical_flops": None if arch is None else {
            "per_batch": arch["cost"],
            "selective_over_s3": arch["cost"]["selective_over_s3_flops"],
        },
        "gpu_peak_allocated_bytes": int(run["gpu_peak_allocated_bytes"]),
        "gpu_peak_reserved_bytes": int(run["gpu_peak_reserved_bytes"]),
    }

    curve_summary = _curve_summary(curve)
    work_target = {
        "best_valid_ap_min": float(C.WORK_TARGET["best_valid_ap_min"]),
        "best_valid_ap_observed": sp_valid,
        "best_valid_ap_pass": bool(sp_valid >= float(C.WORK_TARGET["best_valid_ap_min"])),
        "test_gain_min": float(C.WORK_TARGET["test_gain_min"]),
        "test_gain_observed": test_gain,
        "test_gain_pass": bool(test_gain is not None and test_gain >= float(C.WORK_TARGET["test_gain_min"])),
    }
    work_target["both_pass"] = bool(work_target["best_valid_ap_pass"] and work_target["test_gain_pass"])

    comparison: dict[str, Any] = {
        "schema": "lcrc-pcba-s3-selective-performance-comparison-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "models": models,
        "valid_gain_vs_s3": float(valid_gain),
        "valid_gain_vs_w2": float(sp_valid - W2_REF["best_valid_ap"]),
        "test_gain_vs_s3": test_gain,
        "test_gain_vs_w2": test_gain_vs_w2,
        "s3_test_gain_vs_w2": s3_test_gain_vs_w2,
        "ap40_delta_vs_s3": None if sp_ap40 is None else float(sp_ap40 - S3_REF["ap40"]),
        "best_epoch_minus_40_ap": None if sp_ap40 is None else float(sp_valid - sp_ap40),
        "train_valid_gap": {
            k: (
                None
                if models[k].get("train_ap40") is None
                else float(models[k]["train_ap40"] - models[k]["best_valid_ap"])
            )
            for k in models
        },
        "valid_test_gap": {k: v["valid_test_ap_delta"] for k, v in models.items()},
        "task_comparison": task_comparison,
        "support_strata_from_test_scores": None if scores is None else scores.get("support_strata"),
        "label_stats": None if scores is None else scores.get("label_stats"),
        "cost": cost,
        "sp_validation_curve": curve_summary,
        "attention_diagnostics": {
            "all_roots_normalised": curve_summary["all_weights_normalised"],
            "entropy_ratio_mean": curve_summary["attention_entropy_ratio_mean"],
            "entropy_ratio_min": curve_summary["attention_entropy_ratio_min"],
            "near_one_hot_fraction_first": curve_summary["attention_near_one_hot_fraction_first"],
            "near_one_hot_fraction_last": curve_summary["attention_near_one_hot_fraction_last"],
            "selective_share_of_residual_mean": curve_summary["selective_share_of_residual_mean"],
            "all_uniform": bool(
                curve_summary["attention_entropy_ratio_mean"] is not None
                and curve_summary["attention_entropy_ratio_mean"] > 0.999
            ),
            "all_saturated": bool(
                curve_summary["attention_near_one_hot_fraction_last"] is not None
                and curve_summary["attention_near_one_hot_fraction_last"] > 0.999
            ),
            "selective_residual_never_used": bool(
                curve_summary["selective_share_of_residual_last"] is not None
                and curve_summary["selective_share_of_residual_last"] < 1e-6
            ),
            "note": "diagnostic only; no ablation training was run",
        },
        "work_target": work_target,
        "test_role": "EXPOSED_TEST_PERFORMANCE_BENCHMARK",
        "test_is_blind": False,
        "test_exposure_note": (
            "the MolPCBA official Test split was read by lcrc_pcba_continuation_recipe_v1 and is therefore "
            "not a blind generalisation estimate; this round uses it only as a benchmark of a "
            "Validation-selected checkpoint"
        ),
        "single_seed": True,
        "statistical_significance_claimed": False,
    }
    C.write_json(COMPARISON, comparison)

    ledger = _ledger(run, scores=scores, extra_hours=float(extra_hours))
    C.write_json(LEDGER, ledger)

    decision = _decide(comparison, work_target)
    C.write_json(DECISION, decision)
    C.log(f"[compare] decision {decision['label']}")
    return comparison


def _ledger(run: Mapping[str, Any], *, scores: Mapping[str, Any] | None, extra_hours: float) -> dict[str, Any]:
    freeze_seconds = 0.0
    if scores is not None:
        freeze_path = C.ANALYSIS / "test_prediction_manifest.json"
        if freeze_path.is_file():
            freeze_seconds = float(C.read_json(freeze_path).get("total_seconds", 0.0))
    sp_hours = float(run["gpu_hours"])
    inferred_hours = freeze_seconds / 3600.0
    total = sp_hours + inferred_hours + float(extra_hours)
    return {
        "schema": "lcrc-pcba-s3-selective-gpu-ledger-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "items": {
            "sp_training_40_epochs": {
                "gpu_hours": sp_hours,
                "source": C.track_rel(SP_RUN),
                "epochs": int(run["epochs_run"]),
                "epoch_seconds_mean": float(run["epoch_seconds_mean"]),
                "includes_validation": True,
            },
            "test_freeze_w2_s3_sp": {
                "gpu_hours": inferred_hours,
                "seconds": freeze_seconds,
                "source": C.track_rel(C.ANALYSIS / "test_prediction_manifest.json"),
                "includes_validation_replay": True,
            },
            "audits_and_smoke": {
                "gpu_hours": float(extra_hours),
                "source": "rr job durations (analysis/rr_jobs.json) or the CLI --extra-hours value",
            },
        },
        "training_gpu_hours": sp_hours,
        "total_round_gpu_hours": total,
        "historical_reference_gpu_hours": {
            "W2_training": float(C.W2["gpu_hours"]),
            "S3_training": float(C.S3["gpu_hours"]),
        },
        "budget": dict(C.BUDGET),
        "within_target": bool(total <= float(C.BUDGET["gpu_target_hours"])),
        "within_hard_cap": bool(total <= float(C.BUDGET["gpu_hard_hours"])),
        "host": run.get("host_facts", {}),
        "gpu_occupancy_at_start": run.get("gpu_occupancy_start"),
        "single_trajectory": True,
        "note": "one new training trajectory only; no capacity control and no mechanism ablation were trained",
    }


def _decide(comparison: Mapping[str, Any], work_target: Mapping[str, Any]) -> dict[str, Any]:
    th = float(C.DECISION_RULES["valid_gain_threshold"])
    valid_gain = float(comparison["valid_gain_vs_s3"])
    test_gain = comparison["test_gain_vs_s3"]
    test_gain = None if test_gain is None else float(test_gain)
    valid_improved = bool(valid_gain >= th)
    test_improved = bool(test_gain is not None and test_gain >= th)
    if test_gain is None:
        label = "PENDING_TEST"
        explanation = "Test scores are missing; no performance decision can be written."
    elif valid_improved and test_improved:
        label = C.DECISION_RULES["labels"]["valid_and_test_gain"]
        explanation = "Best Validation AP and the selected-checkpoint Test AP both improve over S3."
    elif valid_improved and not test_improved:
        label = C.DECISION_RULES["labels"]["valid_only_gain"]
        explanation = "Validation improves but Test does not; the gain did not transfer."
    elif (not valid_improved) and test_improved:
        label = C.DECISION_RULES["labels"]["test_only_gain"]
        explanation = (
            "Test improves without a Validation improvement; this is exploratory exposed-Test behaviour "
            "and must not be read as evidence that Validation selection is reliable."
        )
    else:
        label = C.DECISION_RULES["labels"]["no_clear_gain"]
        explanation = "Neither dimension improves beyond the pre-registered 0.002 threshold."
    return {
        "schema": "lcrc-pcba-s3-selective-decision-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "label": label,
        "explanation": explanation,
        "thresholds": {
            "valid_gain_min": th,
            "test_gain_min": th,
            "valid_gain_observed": valid_gain,
            "test_gain_observed": test_gain,
        },
        "work_target": dict(work_target),
        "work_target_note": (
            "Best Valid AP >= 0.2745 and Test AP >= TestAP(S3) + 0.003 is an engineering target, "
            "not a statistical significance threshold"
        ),
        "test_role": "EXPOSED_TEST_PERFORMANCE_BENCHMARK",
        "keep_as_main_backbone": bool(label in ("VALID_AND_TEST_GAIN",)),
        "further_rounds_authorised": False,
        "forbidden_followups": [
            "additional attention heads",
            "longer training",
            "a second architecture in this round",
            "seed 1",
            "COUNT/ROOT ablations",
            "new scaffold splits",
        ],
        "test_not_used_for_selection": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["select", "compare", "all"])
    parser.add_argument("--extra-hours", type=float, default=0.0)
    args = parser.parse_args(argv)
    if args.mode in ("select", "all"):
        run_select()
    if args.mode in ("compare", "all"):
        run_compare(extra_hours=float(args.extra_hours))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
