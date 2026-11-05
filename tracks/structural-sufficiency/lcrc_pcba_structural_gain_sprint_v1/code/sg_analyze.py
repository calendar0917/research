"""Pre-registered analysis of the structural-gain sprint.

Reads only the fixed-epoch artifacts written by ``sg_train.py``.  Every
comparison uses the epoch fixed in ``PROTOCOL.md``; best-epoch numbers are
reported as auxiliary only.  Official Test is never read.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import sg_common as C


def _read_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if value is None or value == "":
                    row[key] = None
                else:
                    try:
                        row[key] = float(value) if "." in value or "e" in value.lower() else int(value)
                    except ValueError:
                        row[key] = value
            rows.append(row)
    return rows


def load_run(name: str) -> dict[str, Any]:
    d = C.RUNS / name
    run = C.read_json(d / "run.json")
    out = {"dir": d, "run": run, "name": name}
    if (d / "train_curve.csv").exists():
        out["curve"] = _read_csv(d / "train_curve.csv")
    if (d / "per_task_ap.csv").exists():
        out["per_task"] = _read_csv(d / "per_task_ap.csv")
    for path in sorted(d.glob("train_eval_epoch*.json")):
        out[path.stem] = C.read_json(path)
    return out


def _per_task_at(run: Mapping[str, Any], epoch: int) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for row in run.get("per_task", []):
        if int(row["epoch"]) != int(epoch):
            continue
        out[int(row["task"])] = {
            "ap": None if row["ap"] is None else float(row["ap"]),
            "positive": int(row["positive"]),
            "observed": int(row["observed"]),
            "negative": int(row["negative"]),
            "train_positive": int(row["train_positive"]) if row.get("train_positive") is not None else None,
        }
    return out


def _ap_at_fixed(run: Mapping[str, Any], epoch: int) -> float | None:
    report = (run["run"].get("fixed_epoch_reports") or {}).get(str(epoch))
    return None if report is None else float(report["ap"])


def task_comparison(
    arm_run: Mapping[str, Any],
    control_run: Mapping[str, Any],
    epoch: int,
    *,
    min_positive: int = 20,
) -> dict[str, Any]:
    a = _per_task_at(arm_run, epoch)
    c = _per_task_at(control_run, epoch)
    common = sorted(set(a) & set(c))
    scored = [t for t in common if a[t]["ap"] is not None and c[t]["ap"] is not None]
    diffs = {t: float(a[t]["ap"] - c[t]["ap"]) for t in scored}
    improved = [t for t in scored if diffs[t] > 0]
    regressed = [t for t in scored if diffs[t] < 0]
    support = [t for t in scored if a[t]["positive"] >= int(min_positive)]
    support_diffs = [diffs[t] for t in support]
    best_t = max(diffs, key=diffs.get) if diffs else None
    worst_t = min(diffs, key=diffs.get) if diffs else None
    return {
        "epoch": int(epoch),
        "n_tasks_compared": len(common),
        "n_tasks_scored_both": len(scored),
        "n_tasks_improved": len(improved),
        "n_tasks_regressed": len(regressed),
        "n_tasks_tied": len(scored) - len(improved) - len(regressed),
        "mean_ap_difference": float(np.mean(list(diffs.values()))) if diffs else None,
        "median_ap_difference": float(np.median(list(diffs.values()))) if diffs else None,
        "support_subset": {
            "min_valid_positive": int(min_positive),
            "n_tasks": len(support),
            "mean_ap_difference": float(np.mean(support_diffs)) if support_diffs else None,
            "n_improved": sum(1 for d in support_diffs if d > 0),
            "n_regressed": sum(1 for d in support_diffs if d < 0),
        },
        "largest_gain": None
        if best_t is None
        else {
            "task": int(best_t),
            "ap_difference": float(diffs[best_t]),
            "valid_positive": int(a[best_t]["positive"]),
            "valid_observed": int(a[best_t]["observed"]),
            "arm_ap": float(a[best_t]["ap"]),
            "control_ap": float(c[best_t]["ap"]),
        },
        "largest_loss": None
        if worst_t is None
        else {
            "task": int(worst_t),
            "ap_difference": float(diffs[worst_t]),
            "valid_positive": int(a[worst_t]["positive"]),
            "valid_observed": int(a[worst_t]["observed"]),
            "arm_ap": float(a[worst_t]["ap"]),
            "control_ap": float(c[worst_t]["ap"]),
        },
        "per_task_differences": {str(t): diffs[t] for t in scored},
    }


def _run_health(run: Mapping[str, Any]) -> dict[str, Any]:
    r = run["run"]
    return {
        "status": r.get("status"),
        "epochs_run": r.get("epochs_run"),
        "eval_completed": r.get("eval_epochs_completed"),
        "non_finite_events": r.get("non_finite_events"),
        "source_state_bitwise_equal": bool(
            (r.get("model_audit") or {}).get("warm_start", {}).get("state_dict_bitwise_equal", False)
        )
        if r.get("phase") in (1, 2)
        else None,
        "optimizer_group0_identical": bool(
            (r.get("optimizer_state_audit") or {}).get("group0_identical_to_source", False)
        )
        if r.get("phase") in (1, 2)
        else None,
        "test_evaluated": r.get("test_evaluated"),
        "complete": r.get("status") == "COMPLETED",
    }


def _curve_summary(run: Mapping[str, Any]) -> dict[str, Any]:
    rows = run.get("curve", [])
    return {
        "fixed_epoch_ap": {
            str(int(r["epoch"])): float(r["valid_ap"])
            for r in rows
            if r.get("evaluated") and r.get("valid_ap") is not None
        },
        "train_loss_by_epoch": {
            str(int(r["epoch"])): float(r["train_loss"]) for r in rows if r.get("train_loss") is not None
        },
        "gpu_peak_allocated_bytes": run["run"].get("gpu_peak_allocated_bytes"),
        "step_ms_p50_mean": run["run"].get("step_ms_p50_mean")
        or (
            float(np.mean([r["step_ms_p50"] for r in rows if r.get("step_ms_p50") is not None]))
            if any(r.get("step_ms_p50") is not None for r in rows)
            else None
        ),
        "steps_per_second_mean": run["run"].get("steps_per_second_mean"),
        "train_seconds": run["run"].get("train_seconds"),
        "gpu_hours": run["run"].get("gpu_hours"),
        "parameters": run["run"].get("parameters"),
        "new_parameters": run["run"].get("new_parameters"),
    }


def _train_eval(run: Mapping[str, Any], epoch: int) -> dict[str, Any] | None:
    payload = run.get(f"train_eval_epoch{epoch}")
    if payload is None:
        return None
    return {
        "ap": float(payload["ap"]),
        "bce": float(payload["bce"]),
        "n_tasks_scored": int(payload["n_tasks_scored"]),
        "evaluated_rows": int(payload["evaluated_rows"]),
        "per_task_ap": payload["per_task_ap"],
        "seconds": float(payload["seconds"]),
    }


def analyze_phase1() -> dict[str, Any]:
    names = {arm: f"phase1_{arm}" for arm in C.ARM_ORDER}
    runs: dict[str, Any] = {}
    missing = []
    for arm, name in names.items():
        if not (C.RUNS / name / "run.json").exists():
            missing.append(name)
        else:
            runs[arm] = load_run(name)
    result: dict[str, Any] = {
        "schema": "lcrc-pcba-structural-gain-phase1-v1",
        "round_id": C.ROUND_ID,
        "phase": 1,
        "ap0": C.AP0,
        "rules": C.DECISION["phase1"],
        "runs": {arm: names[arm] for arm in names},
        "missing_runs": missing,
    }
    if missing:
        result["label"] = "SCREEN_INCOMPLETE"
        return result

    c_run = runs["C"]
    c12 = _ap_at_fixed(c_run, 12)
    result["control"] = {
        "arm": "C",
        "ap12": c12,
        "health": _run_health(c_run),
        "curve": _curve_summary(c_run),
        "train_eval_epoch12": _train_eval(c_run, 12),
    }
    candidates = []
    arm_rows: dict[str, Any] = {}
    for arm in ("F", "M", "V"):
        run = runs[arm]
        ap12 = _ap_at_fixed(run, 12)
        health = _run_health(run)
        cmp12 = task_comparison(run, c_run, 12)
        d_base = None if ap12 is None else float(ap12 - C.AP0)
        d_control = None if ap12 is None or c12 is None else float(ap12 - c12)
        checks = {
            "completed_12_epochs": bool(health["complete"] and health["epochs_run"] == 12),
            "d_base_ge_floor": bool(d_base is not None and d_base >= C.DECISION["phase1"]["d_base_min"]),
            "d_control_ge_floor": bool(
                d_control is not None and d_control >= C.DECISION["phase1"]["d_control_min"]
            ),
            "improved_tasks_ge_70": bool(cmp12["n_tasks_improved"] >= 70),
            "support_subset_mean_ge_0": bool(
                cmp12["support_subset"]["mean_ap_difference"] is not None
                and cmp12["support_subset"]["mean_ap_difference"] >= 0.0
            ),
            "no_anomaly": bool(
                health["non_finite_events"] == 0
                and health["source_state_bitwise_equal"] is not False
                and health["optimizer_group0_identical"] is not False
                and health["test_evaluated"] is False
            ),
        }
        qualified = all(checks.values())
        arm_rows[arm] = {
            "ap12": ap12,
            "d_base": d_base,
            "d_control": d_control,
            "health": health,
            "checks": {k: bool(v) for k, v in checks.items()},
            "qualified": bool(qualified),
            "task_comparison_epoch12": cmp12,
            "curve": _curve_summary(run),
            "train_eval_epoch12": _train_eval(run, 12),
            "best_epoch": run["run"].get("best_epoch"),
            "best_valid_ap": run["run"].get("best_valid_ap"),
        }
        if qualified:
            candidates.append((arm, float(ap12)))
    result["control"]["ap12"] = c12
    result["arms"] = arm_rows
    if candidates:
        candidates.sort(key=lambda x: (-x[1], C.DECISION["phase1"]["tie_break_order"].index(x[0])))
        top_ap = candidates[0][1]
        near = [c for c in candidates if top_ap - c[1] <= C.DECISION["phase1"]["tie_break_ap"]]
        near.sort(key=lambda x: C.DECISION["phase1"]["tie_break_order"].index(x[0]))
        promoted = near[0][0]
        result["label"] = f"PROMOTE_{promoted}"
        result["promoted_arm"] = promoted
        result["promoted_ap12"] = top_ap
        result["promotion_reason"] = (
            f"qualified candidates {candidates}; highest epoch-12 AP with pre-registered "
            f"F<M<V tie-break within {C.DECISION['phase1']['tie_break_ap']}"
        )
    else:
        result["label"] = "SCREEN_NO_PROMOTION"
        result["promoted_arm"] = None
        result["promotion_reason"] = "no F/M/V arm met all six pre-registered rules"
    return result


def analyze_phase2() -> dict[str, Any]:
    screen = C.read_json(C.ANALYSIS / "phase1_screen.json")
    result: dict[str, Any] = {
        "schema": "lcrc-pcba-structural-gain-phase2-v1",
        "round_id": C.ROUND_ID,
        "phase": 2,
        "phase1_label": screen.get("label"),
        "rules": C.DECISION["phase2"],
    }
    if screen.get("label") not in ("PROMOTE_F", "PROMOTE_M", "PROMOTE_V"):
        result["label"] = "NOT_RUN_PHASE1_NO_PROMOTION"
        return result
    w = screen["promoted_arm"]
    result["promoted_arm"] = w
    names = {"C": "phase2_C", "W": f"phase2_{w}"}
    runs = {}
    missing = []
    for key, name in names.items():
        if not (C.RUNS / name / "run.json").exists():
            missing.append(name)
        else:
            runs[key] = load_run(name)
    result["runs"] = names
    if missing:
        result["label"] = "NO_SUSTAINED_GAIN"
        result["missing_runs"] = missing
        return result
    ap_c40 = _ap_at_fixed(runs["C"], 40)
    ap_w40 = _ap_at_fixed(runs["W"], 40)
    cmp40 = task_comparison(runs["W"], runs["C"], 40)
    health_w = _run_health(runs["W"])
    health_c = _run_health(runs["C"])
    d40 = None if ap_w40 is None or ap_c40 is None else float(ap_w40 - ap_c40)
    checks = {
        "both_completed_40_epochs": bool(
            health_w["complete"] and health_c["complete"] and health_w["epochs_run"] == 40 and health_c["epochs_run"] == 40
        ),
        "ap_w40_ge_0.245": bool(ap_w40 is not None and ap_w40 >= C.DECISION["phase2"]["ap_w40_min"]),
        "d40_ge_0.006": bool(d40 is not None and d40 >= C.DECISION["phase2"]["d40_min"]),
        "improved_tasks_ge_70": bool(cmp40["n_tasks_improved"] >= 70),
        "support_subset_mean_ge_0": bool(
            cmp40["support_subset"]["mean_ap_difference"] is not None
            and cmp40["support_subset"]["mean_ap_difference"] >= 0.0
        ),
        "no_anomaly": bool(
            health_w["non_finite_events"] == 0
            and health_c["non_finite_events"] == 0
            and health_w["test_evaluated"] is False
            and health_c["test_evaluated"] is False
        ),
    }
    result.update(
        {
            "ap_c40": ap_c40,
            "ap_w40": ap_w40,
            "d40": d40,
            "checks": {k: bool(v) for k, v in checks.items()},
            "task_comparison_epoch40": cmp40,
            "W": {
                "health": health_w,
                "curve": _curve_summary(runs["W"]),
                "best_epoch": runs["W"]["run"].get("best_epoch"),
                "best_valid_ap": runs["W"]["run"].get("best_valid_ap"),
                "train_eval_epoch40": _train_eval(runs["W"], 40),
            },
            "C": {
                "health": health_c,
                "curve": _curve_summary(runs["C"]),
                "best_epoch": runs["C"]["run"].get("best_epoch"),
                "best_valid_ap": runs["C"]["run"].get("best_valid_ap"),
                "train_eval_epoch40": _train_eval(runs["C"], 40),
            },
            "historical_references": {
                "ap0": C.AP0,
                "ap_readout": C.AP_READOUT,
                "d40_vs_ap0": None if ap_w40 is None else float(ap_w40 - C.AP0),
                "d40_vs_readout": None if ap_w40 is None else float(ap_w40 - C.AP_READOUT),
            },
        }
    )
    result["label"] = "PROMOTE_PHASE3" if all(checks.values()) else "NO_SUSTAINED_GAIN"
    return result


def analyze_phase3() -> dict[str, Any]:
    phase2 = C.read_json(C.ANALYSIS / "phase2_continuation.json")
    result: dict[str, Any] = {
        "schema": "lcrc-pcba-structural-gain-phase3-v1",
        "round_id": C.ROUND_ID,
        "phase": 3,
        "phase2_label": phase2.get("label"),
        "rules": C.DECISION["phase3"],
    }
    if phase2.get("label") != "PROMOTE_PHASE3":
        result["label"] = "NOT_RUN_PHASE2_NOT_PROMOTED"
        return result
    w = phase2["promoted_arm"]
    result["winner_arm"] = w
    runs: dict[str, dict[int, Any]] = {}
    missing = []
    for arm in ("C", w):
        for seed in (0, 1):
            name = f"phase3_seed{seed}_{arm}"
            if not (C.RUNS / name / "run.json").exists():
                missing.append(name)
            else:
                runs.setdefault(arm, {})[seed] = load_run(name)
    result["runs"] = {
        arm: {str(seed): f"phase3_seed{seed}_{arm}" for seed in seeds} for arm, seeds in runs.items()
    }
    if missing or len(runs.get("C", {})) != 2 or len(runs.get(w, {})) != 2:
        result["label"] = "CONFIRMATION_INCOMPLETE"
        result["missing_runs"] = missing
        return result

    seeds = {}
    for seed in (0, 1):
        c_run = runs["C"][seed]
        w_run = runs[w][seed]
        ap_c40 = _ap_at_fixed(c_run, 40)
        ap_w40 = _ap_at_fixed(w_run, 40)
        d_seed = None if ap_c40 is None or ap_w40 is None else float(ap_w40 - ap_c40)
        cmp40 = task_comparison(w_run, c_run, 40)
        health_c = _run_health(c_run)
        health_w = _run_health(w_run)
        seeds[seed] = {
            "ap_c40": ap_c40,
            "ap_w40": ap_w40,
            "d_seed": d_seed,
            "w_ge_025": bool(ap_w40 is not None and ap_w40 >= 0.250),
            "d_seed_ge_0.005": bool(d_seed is not None and d_seed >= 0.005),
            "task_comparison_epoch40": cmp40,
            "w": {
                "health": health_w,
                "curve": _curve_summary(w_run),
                "best_epoch": w_run["run"].get("best_epoch"),
                "best_valid_ap": w_run["run"].get("best_valid_ap"),
                "train_eval_epoch40": _train_eval(w_run, 40),
                "gpu_hours": w_run["run"].get("gpu_hours"),
                "parameters": w_run["run"].get("parameters"),
            },
            "c": {
                "health": health_c,
                "curve": _curve_summary(c_run),
                "best_epoch": c_run["run"].get("best_epoch"),
                "best_valid_ap": c_run["run"].get("best_valid_ap"),
                "train_eval_epoch40": _train_eval(c_run, 40),
                "gpu_hours": c_run["run"].get("gpu_hours"),
                "parameters": c_run["run"].get("parameters"),
            },
            "complete": bool(health_c["complete"] and health_w["complete"]),
        }
    complete = all(seeds[s]["complete"] for s in (0, 1))
    both_w = all(seeds[s]["w_ge_025"] for s in (0, 1))
    both_d = all(seeds[s]["d_seed_ge_0.005"] for s in (0, 1))
    if not complete:
        label = "CONFIRMATION_INCOMPLETE"
    elif both_w and both_d:
        label = "PCBA_025_REPRODUCED"
    elif any((seeds[s]["d_seed"] or -1) > 0 for s in (0, 1)):
        label = "POSITIVE_BUT_NOT_REPRODUCED"
    else:
        label = "NO_REPRODUCIBLE_GAIN"
    w_aps = [seeds[s]["ap_w40"] for s in (0, 1) if seeds[s]["ap_w40"] is not None]
    ds = [seeds[s]["d_seed"] for s in (0, 1) if seeds[s]["d_seed"] is not None]
    result.update(
        {
            "seeds": {str(k): v for k, v in seeds.items()},
            "w_ap40_mean": float(statistics.mean(w_aps)) if w_aps else None,
            "w_ap40_std": float(statistics.pstdev(w_aps)) if len(w_aps) > 1 else None,
            "d_seed_mean": float(statistics.mean(ds)) if ds else None,
            "d_seed_std": float(statistics.pstdev(ds)) if len(ds) > 1 else None,
            "label": label,
        }
    )
    return result


def analyze_gpu_ledger() -> dict[str, Any]:
    entries = []
    total = 0.0
    for d in sorted(p for p in C.RUNS.iterdir() if p.is_dir()):
        run_path = d / "run.json"
        if not run_path.exists():
            continue
        run = C.read_json(run_path)
        hours = float(run.get("gpu_hours") or 0.0)
        entries.append(
            {
                "kind": "training",
                "name": d.name,
                "phase": run.get("phase"),
                "arm": run.get("arm"),
                "seed": run.get("seed"),
                "status": run.get("status"),
                "stop_reason": run.get("stop_reason"),
                "gpu_hours": hours,
                "gpu_peak_allocated_bytes": run.get("gpu_peak_allocated_bytes"),
                "parameters": run.get("parameters"),
                "git_commit": run.get("git_commit"),
                "started_utc": run.get("started_utc"),
                "finished_utc": run.get("finished_utc"),
            }
        )
        total += hours
    preflights = []
    for path in sorted(C.ANALYSIS.glob("preflight_*.json")):
        p = C.read_json(path)
        hours = float(p.get("gpu_hours") or 0.0)
        preflights.append(
            {
                "kind": "preflight",
                "name": path.stem,
                "arms": p.get("arms"),
                "status": p.get("status"),
                "gpu_hours": hours,
                "wall_seconds": p.get("wall_seconds"),
                "git_commit": (p.get("environment") or {}).get("git_commit"),
            }
        )
        total += hours
    return {
        "schema": "lcrc-pcba-structural-gain-gpu-ledger-v1",
        "round_id": C.ROUND_ID,
        "budget": C.BUDGET,
        "training_entries": entries,
        "preflight_entries": preflights,
        "gpu_hours_total": float(total),
        "target_gpu_hours": C.BUDGET["gpu_target_hours"],
        "hard_gpu_hours": C.BUDGET["gpu_hard_hours"],
        "note": "every preflight, failed and formal GPU process is included",
    }


def build_registry(rr_jobs_path: Path | None) -> dict[str, Any]:
    jobs = []
    if rr_jobs_path is not None and rr_jobs_path.exists():
        payload = json.loads(rr_jobs_path.read_text())
        for job in payload.get("jobs", []):
            if not str(job.get("experiment", "")).startswith("sgs-"):
                continue
            jobs.append(
                {
                    "experiment": job.get("experiment"),
                    "run_id": job.get("run_id"),
                    "host": job.get("host"),
                    "pool": job.get("pool"),
                    "state": job.get("state"),
                    "node": job.get("node"),
                    "gpu": job.get("cuda_visible_devices"),
                    "commit": job.get("commit"),
                    "created_at": job.get("created_at"),
                    "started_at": job.get("started_at"),
                    "completed_at": job.get("completed_at"),
                    "exit_code": job.get("exit_code"),
                    "torch_version": job.get("torch_version"),
                    "torch_cuda": job.get("torch_cuda"),
                    "driver_version": job.get("driver_version"),
                }
            )
    return {
        "schema": "lcrc-pcba-structural-gain-run-registry-v1",
        "round_id": C.ROUND_ID,
        "jobs": jobs,
        "source": str(rr_jobs_path) if rr_jobs_path else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("phase1", "phase2", "phase3", "ledger", "registry", "all"),
    )
    parser.add_argument("--rr-jobs", default="")
    args = parser.parse_args(argv)
    out = None
    if args.command in ("phase1", "all"):
        payload = analyze_phase1()
        C.write_json(C.ANALYSIS / "phase1_screen.json", payload)
        C.log(f"phase1 label={payload['label']}")
        out = payload
    if args.command in ("phase2", "all"):
        payload = analyze_phase2()
        C.write_json(C.ANALYSIS / "phase2_continuation.json", payload)
        C.log(f"phase2 label={payload['label']}")
        out = payload
    if args.command in ("phase3", "all"):
        payload = analyze_phase3()
        C.write_json(C.ANALYSIS / "phase3_seed_confirmation.json", payload)
        C.log(f"phase3 label={payload['label']}")
        out = payload
    if args.command in ("ledger", "all"):
        payload = analyze_gpu_ledger()
        C.write_json(C.ANALYSIS / "gpu_ledger.json", payload)
        C.log(f"gpu_hours_total={payload['gpu_hours_total']:.4f}")
        out = payload
    if args.command in ("registry", "all"):
        payload = build_registry(Path(args.rr_jobs) if args.rr_jobs else None)
        C.write_json(C.ANALYSIS / "run_registry.json", payload)
        C.log(f"registry jobs={len(payload['jobs'])}")
        out = payload
    return 0 if out is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
