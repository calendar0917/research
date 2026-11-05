"""Build ``analysis/comparison.json`` from the two formal 12-epoch arms.

The decision rule is applied verbatim.  The label is a resource-decision label,
not a statistical-significance claim.  Both curves, the best/final AP, the
per-task differences and every limitation are reported even when the label is
``NO_CLEAR_INCREMENT`` or ``INCOMPLETE``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _configure_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_configure_imports()

import lr_common as C  # noqa: E402

_INT_FIELDS = {
    "epoch",
    "n_tasks_with_observed_labels",
    "n_tasks_scored",
    "gpu_peak_allocated_bytes",
    "gpu_peak_reserved_bytes",
    "best_epoch_so_far",
    "task",
    "observed",
    "positive",
    "negative",
}
_BOOL_FIELDS = {"evaluated", "is_best", "strict_improve"}


def _parse_cell(key: str, value: str) -> Any:
    if value == "":
        return None
    if key in _BOOL_FIELDS:
        return value.lower() in ("true", "1")
    if key in _INT_FIELDS:
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value


def load_curve(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            rows.append({k: _parse_cell(k, v) for k, v in raw.items()})
    return rows


def load_per_task(path: Path | str, epoch: int) -> dict[int, float | None]:
    out: dict[int, float | None] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            if int(raw["epoch"]) != int(epoch):
                continue
            out[int(raw["task"])] = None if raw["ap"] == "" else float(raw["ap"])
    return out


def _epoch_row(curve: Sequence[Mapping[str, Any]], epoch: int) -> Mapping[str, Any] | None:
    for row in curve:
        if int(row["epoch"]) == int(epoch):
            return row
    return None


def _diff_stats(h: Mapping[int, float | None], low: Mapping[int, float | None]) -> dict[str, Any]:
    tasks = sorted(set(h) & set(low))
    pairs = [(t, float(h[t]), float(low[t])) for t in tasks if h[t] is not None and low[t] is not None]
    if not pairs:
        return {"n_tasks_compared": 0, "per_task": []}
    diffs = np.asarray([lp - hp for _, hp, lp in pairs], dtype=np.float64)
    order_pos = sorted(pairs, key=lambda x: (x[2] - x[1]), reverse=True)
    order_neg = sorted(pairs, key=lambda x: (x[2] - x[1]))
    return {
        "n_tasks_compared": len(pairs),
        "mean_L_minus_H": float(diffs.mean()),
        "median_L_minus_H": float(np.median(diffs)),
        "min_L_minus_H": float(diffs.min()),
        "max_L_minus_H": float(diffs.max()),
        "stdev_L_minus_H": float(diffs.std(ddof=0)),
        "n_L_better": int((diffs > 0).sum()),
        "n_H_better": int((diffs < 0).sum()),
        "n_tied": int((diffs == 0).sum()),
        "top_L_advantages": [
            {"task": int(t), "H": hp, "L": lp, "L_minus_H": float(lp - hp)}
            for t, hp, lp in order_pos[:5]
            if lp - hp > 0
        ],
        "top_H_advantages": [
            {"task": int(t), "H": hp, "L": lp, "L_minus_H": float(lp - hp)}
            for t, hp, lp in order_neg[:5]
            if lp - hp < 0
        ],
        "per_task": [
            {"task": int(t), "H": hp, "L": lp, "L_minus_H": float(lp - hp)}
            for t, hp, lp in pairs
        ],
    }


def _curve_summary(curve: Sequence[Mapping[str, Any]], eval_epochs: Sequence[int]) -> dict[str, Any]:
    rows = []
    for e in eval_epochs:
        r = _epoch_row(curve, e)
        if r is None:
            rows.append({"epoch": int(e), "present": False})
            continue
        rows.append(
            {
                "epoch": int(e),
                "present": True,
                "lr": r.get("lr"),
                "train_loss": r.get("train_loss"),
                "valid_ap": r.get("valid_ap"),
                "n_tasks_scored": r.get("n_tasks_scored"),
                "step_ms_p50": r.get("step_ms_p50"),
                "train_seconds": r.get("train_seconds"),
                "valid_seconds": r.get("valid_seconds"),
                "epoch_seconds": r.get("epoch_seconds"),
            }
        )
    e1 = _epoch_row(curve, 1)
    e12 = _epoch_row(curve, 12)
    e0 = _epoch_row(curve, 0)
    ap0 = None if e0 is None else e0.get("valid_ap")
    ap12 = None if e12 is None else e12.get("valid_ap")
    loss1 = None if e1 is None else e1.get("train_loss")
    loss12 = None if e12 is None else e12.get("train_loss")
    ap_improved = ap0 is not None and ap12 is not None and float(ap12) > float(ap0)
    loss_improved = loss1 is not None and loss12 is not None and float(loss12) < float(loss1)
    return {
        "eval_epochs": rows,
        "epoch1_train_loss": loss1,
        "epoch12_train_loss": loss12,
        "train_loss_delta_1_to_12": None if loss1 is None or loss12 is None else float(loss12) - float(loss1),
        "epoch0_valid_ap": ap0,
        "epoch12_valid_ap": ap12,
        "valid_ap_delta_0_to_12": None if ap0 is None or ap12 is None else float(ap12) - float(ap0),
        "valid_ap_improved_0_to_12": bool(ap_improved),
        "train_loss_improved_1_to_12": bool(loss_improved),
        "validation_improved_but_train_loss_worsened": bool(ap_improved and not loss_improved),
        "train_loss_improved_but_validation_worsened": bool(loss_improved and not ap_improved),
        "train_loss_min": min(
            [float(r["train_loss"]) for r in curve if r.get("train_loss") is not None], default=None
        ),
        "train_loss_max": max(
            [float(r["train_loss"]) for r in curve if r.get("train_loss") is not None], default=None
        ),
    }


def _resource_summary(run: Mapping[str, Any]) -> dict[str, Any]:
    curve = load_curve(run["curve"]) if run.get("curve") and Path(C.REPO / run["curve"]).exists() else []
    train_rows = int(run.get("data", {}).get("split_sizes", {}).get("train", 0))
    train_seconds = float(run.get("train_seconds", 0.0))
    epoch_seconds = float(run.get("epoch_seconds_mean", 0.0))
    return {
        "status": run.get("status"),
        "gpu_seconds": run.get("gpu_seconds"),
        "gpu_hours": run.get("gpu_hours"),
        "data_load_seconds": run.get("data_load_seconds"),
        "train_seconds": train_seconds,
        "valid_seconds": run.get("valid_seconds"),
        "epoch_seconds_mean": epoch_seconds,
        "step_ms_p50_mean": run.get("step_ms_p50_mean"),
        "steps_per_second_mean": run.get("steps_per_second_mean"),
        "train_molecules_per_second": (
            float(train_rows / train_seconds) if train_seconds > 0 and train_rows else None
        ),
        "epoch_molecules_per_second": (
            float(train_rows / epoch_seconds) if epoch_seconds > 0 and train_rows else None
        ),
        "gpu_peak_allocated_bytes": run.get("gpu_peak_allocated_bytes"),
        "gpu_peak_reserved_bytes": run.get("gpu_peak_reserved_bytes"),
        "grad_clip_events": run.get("grad_clip_events"),
        "grad_norm_max": run.get("grad_norm_max"),
        "non_finite_events": run.get("non_finite_events"),
        "trajectory": [
            {
                "epoch": int(r["epoch"]),
                "lr": r.get("lr"),
                "train_loss": r.get("train_loss"),
                "valid_ap": r.get("valid_ap"),
                "step_ms_p50": r.get("step_ms_p50"),
                "train_seconds": r.get("train_seconds"),
            }
            for r in curve
        ],
    }


def _rng_pair_equal(h: Mapping[str, Any], low: Mapping[str, Any]) -> dict[str, Any] | None:
    hh = h.get("rng_epoch_hashes") or []
    ll = low.get("rng_epoch_hashes") or []
    if not hh or not ll or len(hh) != len(ll):
        return None
    rows = []
    for a, b in zip(hh, ll):
        rows.append(
            {
                "epoch": int(a["epoch"]),
                "cpu_equal": a.get("torch_cpu_sha256") == b.get("torch_cpu_sha256"),
                "cuda_equal": a.get("torch_cuda_sha256") == b.get("torch_cuda_sha256"),
            }
        )
    return {
        "epochs": rows,
        "all_cpu_equal": all(r["cpu_equal"] for r in rows),
        "all_cuda_equal": all(r["cuda_equal"] for r in rows),
        "note": "identical RNG state means identical dropout masks given identical weights and batch order",
    }


def build_comparison(
    run_h: Mapping[str, Any],
    run_l: Mapping[str, Any],
    *,
    init_audit: Mapping[str, Any] | None = None,
    paired_order: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    curve_h = load_curve(C.REPO / run_h["curve"])
    curve_l = load_curve(C.REPO / run_l["curve"])
    e12h = _epoch_row(curve_h, 12)
    e12l = _epoch_row(curve_l, 12)
    h_ap12 = None if e12h is None else e12h.get("valid_ap")
    l_ap12 = None if e12l is None else e12l.get("valid_ap")
    decision = C.decide_label(
        None if h_ap12 is None else float(h_ap12),
        None if l_ap12 is None else float(l_ap12),
        str(run_h.get("status")),
        str(run_l.get("status")),
    )
    decision["D_old_vs_historical_checkpoint"] = (
        None if l_ap12 is None else float(l_ap12) - C.HIST["best_valid_ap"]
    )
    decision["H_improvement_vs_0.23449"] = (
        None if h_ap12 is None else float(h_ap12) - float(C.DECISION["historical_ap_literal"])
    )
    decision["L_improvement_vs_0.23449"] = (
        None if l_ap12 is None else float(l_ap12) - float(C.DECISION["historical_ap_literal"])
    )
    decision["both_arms_improved_by_floor"] = bool(
        h_ap12 is not None
        and l_ap12 is not None
        and (float(h_ap12) - float(C.DECISION["historical_ap_literal"])) >= float(C.DECISION["improvement_floor"])
        and (float(l_ap12) - float(C.DECISION["historical_ap_literal"])) >= float(C.DECISION["improvement_floor"])
    )

    # Per-task differences at epoch 12 (the primary comparison) and at each
    # arm's selected best epoch (secondary, descriptive only).
    h_task12 = load_per_task(C.REPO / run_h["per_task_ap"], 12)
    l_task12 = load_per_task(C.REPO / run_l["per_task_ap"], 12)
    per_task_12 = _diff_stats(h_task12, l_task12)
    best_task_h = load_per_task(C.REPO / run_h["per_task_ap"], int(run_h["best_epoch"]))
    best_task_l = load_per_task(C.REPO / run_l["per_task_ap"], int(run_l["best_epoch"]))
    per_task_best = _diff_stats(best_task_h, best_task_l)

    resource_h = _resource_summary(run_h)
    resource_l = _resource_summary(run_l)
    preflight_hours = 0.0
    preflight_files = []
    for name in ("preflight_H.json", "preflight_L.json"):
        p = C.ANALYSIS / name
        if p.exists():
            payload = C.read_json(p)
            preflight_hours += float(payload.get("gpu_hours", 0.0))
            preflight_files.append(
                {
                    "path": C.track_rel(p),
                    "gpu_seconds": payload.get("gpu_seconds"),
                    "gpu_hours": payload.get("gpu_hours"),
                    "all_passed": payload.get("all_passed"),
                }
            )
    total_hours = float(run_h.get("gpu_hours", 0.0)) + float(run_l.get("gpu_hours", 0.0)) + preflight_hours

    rng_equal = _rng_pair_equal(run_h, run_l)
    order_h = run_h.get("training_order", {}).get("global_plan_sha256")
    order_l = run_l.get("training_order", {}).get("global_plan_sha256")

    init_passed = bool(init_audit.get("passed")) if init_audit is not None else None
    order_passed = bool(paired_order.get("passed")) if paired_order is not None else None
    completed = run_h.get("status") == "COMPLETED" and run_l.get("status") == "COMPLETED"

    limitations = [
        "single seed (0); no seed replication and no statistical significance claim",
        "12 fixed epochs; a short-horizon diagnostic, not a full benchmark",
        "no learning-rate scheduler in either arm; both use a fixed LR",
        "AdamW moments were re-initialised in both arms because the historical checkpoint has no optimizer state",
        "the historical epoch-29 training trajectory cannot be resumed seamlessly; this is a fresh-optimizer probe",
        "official Test targets were never opened or evaluated",
        "the decision label is a resource-decision label, not evidence of a model-capability ceiling",
    ]
    if run_h.get("environment", {}).get("cuda_visible_devices") == run_l.get("environment", {}).get("cuda_visible_devices"):
        limitations.append(
            "both arms ran with the same CUDA_VISIBLE_DEVICES; this is recorded as an execution-regime "
            "deviation from the planned two-GPU parallel assignment"
        )
    if not completed:
        limitations.append(
            "at least one arm is INCOMPLETE; the pair must not be read as a completed LR comparison"
        )

    return {
        "schema": "lcrc-pcba-lr-probe-comparison-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "question": (
            "From the same historical epoch-29 weights, does a lower fixed learning rate "
            "produce better short-horizon official Validation AP than the higher LR?"
        ),
        "arms": {
            "H": {
                "role": C.GROUPS["H"]["role"],
                "learning_rate": C.GROUPS["H"]["learning_rate"],
                "status": run_h.get("status"),
                "best_epoch": run_h.get("best_epoch"),
                "best_valid_ap": run_h.get("best_valid_ap"),
                "final_epoch": run_h.get("final_epoch"),
                "final_valid_ap": run_h.get("final_valid_ap"),
                "curve_summary": _curve_summary(curve_h, C.EVAL_EPOCHS),
                "resource": resource_h,
                "run_json": C.track_rel(Path(run_h["_path"])) if "_path" in run_h else None,
                "device": run_h.get("environment", {}).get("cuda_visible_devices"),
            },
            "L": {
                "role": C.GROUPS["L"]["role"],
                "learning_rate": C.GROUPS["L"]["learning_rate"],
                "status": run_l.get("status"),
                "best_epoch": run_l.get("best_epoch"),
                "best_valid_ap": run_l.get("best_valid_ap"),
                "final_epoch": run_l.get("final_epoch"),
                "final_valid_ap": run_l.get("final_valid_ap"),
                "curve_summary": _curve_summary(curve_l, C.EVAL_EPOCHS),
                "resource": resource_l,
                "run_json": C.track_rel(Path(run_l["_path"])) if "_path" in run_l else None,
                "device": run_l.get("environment", {}).get("cuda_visible_devices"),
            },
        },
        "decision": decision,
        "per_task": {
            "epoch12_L_minus_H": per_task_12,
            "best_epoch_L_minus_H": per_task_best,
            "note": "L minus H; positive means the low-LR arm has higher per-task AP",
        },
        "paired_integrity": {
            "initialization_audit": init_audit,
            "initialization_audit_passed": init_passed,
            "paired_training_order": paired_order,
            "paired_training_order_passed": order_passed,
            "order_plan_sha256_H": order_h,
            "order_plan_sha256_L": order_l,
            "order_plan_equal": order_h == order_l,
            "rng_pair": rng_equal,
            "same_source_checkpoint_sha256": run_h.get("history", {}).get("source_checkpoint_sha256")
            == run_l.get("history", {}).get("source_checkpoint_sha256"),
            "same_loaded_parameter_fingerprint": run_h.get("initialization", {}).get(
                "loaded_parameter_fingerprint"
            )
            == run_l.get("initialization", {}).get("loaded_parameter_fingerprint"),
            "both_adamw_reinitialised": bool(
                run_h.get("history", {}).get("adamw_reinitialised")
                and run_l.get("history", {}).get("adamw_reinitialised")
            ),
        },
        "resources": {
            "H_gpu_hours": run_h.get("gpu_hours"),
            "L_gpu_hours": run_l.get("gpu_hours"),
            "preflight_gpu_hours": preflight_hours,
            "preflight_files": preflight_files,
            "total_gpu_hours": total_hours,
            "budget_gpu_hours": C.BUDGET["gpu_hours_total"],
            "within_budget": bool(total_hours <= float(C.BUDGET["gpu_hours_total"])),
            "budget_note": (
                "GPU-hours are wall-clock seconds of the process holding the device; "
                "the two arms were not necessarily concurrent"
            ),
        },
        "comparison_flags": {
            "completed_pair": bool(completed),
            "D_LR": decision["D_LR"],
            "D_old": decision["D_old"],
            "validation_improved_but_train_loss_worsened_H": run_h.get("_curve_flags", {}).get(
                "validation_improved_but_train_loss_worsened"
            ),
            "validation_improved_but_train_loss_worsened_L": run_l.get("_curve_flags", {}).get(
                "validation_improved_but_train_loss_worsened"
            ),
            "train_loss_improved_but_validation_worsened_H": run_h.get("_curve_flags", {}).get(
                "train_loss_improved_but_validation_worsened"
            ),
            "train_loss_improved_but_validation_worsened_L": run_l.get("_curve_flags", {}).get(
                "train_loss_improved_but_validation_worsened"
            ),
            "L_validation_better_while_L_loss_worse_than_H": bool(
                decision["D_LR"] is not None
                and decision["D_LR"] > 0
                and e12h is not None
                and e12l is not None
                and e12l.get("train_loss") is not None
                and e12h.get("train_loss") is not None
                and float(e12l["train_loss"]) > float(e12h["train_loss"])
            ),
            "L_validation_worse_while_L_loss_better_than_H": bool(
                decision["D_LR"] is not None
                and decision["D_LR"] < 0
                and e12h is not None
                and e12l is not None
                and e12l.get("train_loss") is not None
                and e12h.get("train_loss") is not None
                and float(e12l["train_loss"]) < float(e12h["train_loss"])
            ),
        },
        "test_evaluated": False,
        "test_policy": "official Test targets were NaN in the label artifact and were never indexed",
        "limitations": limitations,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-h", default=str(C.RUNS / "H_high_lr" / "run.json"))
    parser.add_argument("--run-l", default=str(C.RUNS / "L_low_lr" / "run.json"))
    parser.add_argument("--init-audit", default=str(C.ANALYSIS / "initialization_audit.json"))
    parser.add_argument("--paired-order", default=str(C.ANALYSIS / "paired_training_order.json"))
    parser.add_argument("--out", default=str(C.ANALYSIS / "comparison.json"))
    args = parser.parse_args(argv)
    run_h = C.read_json(args.run_h)
    run_l = C.read_json(args.run_l)
    run_h["_path"] = args.run_h
    run_l["_path"] = args.run_l
    # Attach the curve flags used in comparison_flags.
    for run in (run_h, run_l):
        curve = load_curve(C.REPO / run["curve"])
        summary = _curve_summary(curve, C.EVAL_EPOCHS)
        run["_curve_flags"] = {
            "validation_improved_but_train_loss_worsened": summary[
                "validation_improved_but_train_loss_worsened"
            ],
            "train_loss_improved_but_validation_worsened": summary[
                "train_loss_improved_but_validation_worsened"
            ],
        }
    init_audit = C.read_json(args.init_audit) if Path(args.init_audit).exists() else None
    paired_order = C.read_json(args.paired_order) if Path(args.paired_order).exists() else None
    comparison = build_comparison(
        run_h, run_l, init_audit=init_audit, paired_order=paired_order
    )
    C.write_json(args.out, comparison)
    C.log(
        f"comparison label={comparison['decision']['label']} "
        f"D_LR={comparison['decision']['D_LR']} D_old={comparison['decision']['D_old']} "
        f"total_gpu_hours={comparison['resources']['total_gpu_hours']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
