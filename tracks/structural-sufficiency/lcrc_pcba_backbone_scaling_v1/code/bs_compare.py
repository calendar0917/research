"""Comparison, per-task deltas, resource labels and the GPU ledger.

Reads only round artifacts (the two new runs + the read-only historical A0
record + the pre-training analysis).  Never evaluates a model and never touches
an official Test row.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import bs_common as C

SCORABLE_MIN_SUPPORT = int(C.DECISION_RULES["support_strata"]["ge20"])


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_curve(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(run_dir / "train_curve.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            parsed: dict[str, Any] = {}
            for k, v in row.items():
                if v == "":
                    parsed[k] = None
                else:
                    try:
                        parsed[k] = int(v) if k in ("epoch", "global_step", "steps_per_epoch", "n_tasks_scored",
                                                    "n_tasks_with_observed_labels", "gradient_clip_events",
                                                    "gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes") else float(v)
                    except ValueError:
                        parsed[k] = v
            rows.append(parsed)
    return rows


def curve_at(curve: Sequence[Mapping[str, Any]], epoch: int) -> dict[str, Any] | None:
    for row in curve:
        if int(row["epoch"]) == int(epoch):
            return dict(row)
    return None


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def epoch_metrics(run: Mapping[str, Any], curve: Sequence[Mapping[str, Any]], epochs: Sequence[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for e in epochs:
        row = curve_at(curve, e)
        out[str(e)] = {
            "valid_ap": None if row is None else _finite(row.get("valid_ap")),
            "valid_bce": None if row is None else _finite(row.get("valid_bce")),
            "train_bce": None if row is None else _finite(row.get("train_bce")),
            "learning_rate": None if row is None else _finite(row.get("learning_rate")),
            "global_step": None if row is None else row.get("global_step"),
            "epoch_train_seconds": None if row is None else _finite(row.get("epoch_train_seconds")),
            "epoch_valid_seconds": None if row is None else _finite(row.get("epoch_valid_seconds")),
            "step_ms_p50": None if row is None else _finite(row.get("step_ms_p50")),
            "gpu_peak_allocated_bytes": None if row is None else row.get("gpu_peak_allocated_bytes"),
        }
    return out


# ---------------------------------------------------------------------------
# per-task table
# ---------------------------------------------------------------------------


def per_task_table(task: Any, runs: Mapping[str, Mapping[str, Any]], run_dirs: Mapping[str, Path]) -> dict[str, Any]:
    a0 = runs["A0"]
    valid = [(None if v is None else float(v)) for v in a0["fixed_epoch_reports"]["40"]["per_task_ap"]]
    w2 = [(None if v is None else float(v)) for v in runs["W2"]["fixed_epoch_reports"]["40"]["per_task_ap"]]
    w3 = [(None if v is None else float(v)) for v in runs["W3"]["fixed_epoch_reports"]["40"]["per_task_ap"]]
    train_a0 = [float(v) for v in read_json(C.A0_TRAIN_EVAL)["per_task_ap"]]
    train_w2 = [float(v) for v in read_json(Path(run_dirs["W2"]) / "train_eval_epoch40.json")["per_task_ap"]]
    train_w3 = [float(v) for v in read_json(Path(run_dirs["W3"]) / "train_eval_epoch40.json")["per_task_ap"]]

    counts = C.SD.task_counts(task)
    valid_pos = counts["valid"]["positive"]
    valid_obs = counts["valid"]["observed"]
    train_pos = counts["train"]["positive"]
    recorded_pos = [int(v) for v in a0["fixed_epoch_reports"]["40"]["positive"]]
    counts_match = recorded_pos == [int(v) for v in valid_pos]

    rows: list[dict[str, Any]] = []
    for t in range(C.N_TASKS):
        stratum = (
            "support_lt5"
            if valid_pos[t] < int(C.DECISION_RULES["support_strata"]["ge5_lt20"])
            else "support_5_19"
            if valid_pos[t] < SCORABLE_MIN_SUPPORT
            else "support_ge20"
        )
        rows.append(
            {
                "task": t,
                "valid_observed": int(valid_obs[t]),
                "valid_positive": int(valid_pos[t]),
                "train_positive": int(train_pos[t]),
                "stratum": stratum,
                "ap_a0": valid[t],
                "ap_w2": w2[t],
                "ap_w3": w3[t],
                "delta_w2": None if (valid[t] is None or w2[t] is None) else w2[t] - valid[t],
                "delta_w3": None if (valid[t] is None or w3[t] is None) else w3[t] - valid[t],
                "delta_w3_minus_w2": None if (valid[t] is None or w2[t] is None or w3[t] is None) else w3[t] - w2[t],
                "train_ap_a0": train_a0[t],
                "train_ap_w2": train_w2[t],
                "train_ap_w3": train_w3[t],
            }
        )

    scorable = [r for r in rows if r["ap_a0"] is not None and r["ap_w2"] is not None and r["ap_w3"] is not None]
    summary: dict[str, Any] = {
        "n_tasks": int(C.N_TASKS),
        "n_scorable": len(scorable),
        "unscorable_tasks": [r["task"] for r in rows if r["ap_a0"] is None],
        "valid_positive_counts_from_label_artifact_match_a0_record": bool(counts_match),
        "stratum_sizes": {
            s: int(sum(1 for r in scorable if r["stratum"] == s))
            for s in ("support_ge20", "support_5_19", "support_lt5")
        },
    }
    for key, dkey in (("W2", "delta_w2"), ("W3", "delta_w3")):
        deltas = np.asarray([r[dkey] for r in scorable], dtype=np.float64)
        pos_mass = float(deltas[deltas > 0].sum())
        order = np.argsort(-deltas)
        top5 = float(deltas[order[:5]].sum())
        top10 = float(deltas[order[:10]].sum())
        strata: dict[str, Any] = {}
        for s in ("support_ge20", "support_5_19", "support_lt5"):
            sub = [r[dkey] for r in scorable if r["stratum"] == s]
            arr = np.asarray(sub, dtype=np.float64)
            strata[s] = {
                "n": int(arr.size),
                "mean_delta": float(arr.mean()) if arr.size else None,
                "median_delta": float(np.median(arr)) if arr.size else None,
                "improved": int((arr > 0).sum()),
                "regressed": int((arr < 0).sum()),
                "tied": int((arr == 0).sum()),
            }
        summary[key] = {
            "mean_delta": float(deltas.mean()),
            "median_delta": float(np.median(deltas)),
            "improved": int((deltas > 0).sum()),
            "regressed": int((deltas < 0).sum()),
            "tied": int((deltas == 0).sum()),
            "positive_mass": pos_mass,
            "top5_share_of_positive_mass": float(top5 / pos_mass) if pos_mass > 0 else None,
            "top10_share_of_positive_mass": float(top10 / pos_mass) if pos_mass > 0 else None,
            "top5_tasks": [int(scorable[i]["task"]) for i in order[:5]],
            "top5_deltas": [float(deltas[i]) for i in order[:5]],
            "bottom5_tasks": [int(scorable[i]["task"]) for i in order[-5:]],
            "bottom5_deltas": [float(deltas[i]) for i in order[-5:]],
            "strata": strata,
            "spearman_delta_vs_valid_support": None,
        }
        support = np.asarray([r["valid_positive"] for r in scorable], dtype=np.float64)
        if deltas.size > 2:
            summary[key]["spearman_delta_vs_valid_support"] = float(
                np.corrcoef(np.argsort(np.argsort(support)), np.argsort(np.argsort(deltas)))[0, 1]
            )
    # W3 vs W2
    d32 = np.asarray([r["delta_w3_minus_w2"] for r in scorable], dtype=np.float64)
    summary["W3_minus_W2"] = {
        "mean_delta": float(d32.mean()),
        "median_delta": float(np.median(d32)),
        "improved": int((d32 > 0).sum()),
        "regressed": int((d32 < 0).sum()),
    }
    # train-side breadth
    tr_w2 = np.asarray([r["train_ap_w2"] - r["train_ap_a0"] for r in rows], dtype=np.float64)
    tr_w3 = np.asarray([r["train_ap_w3"] - r["train_ap_a0"] for r in rows], dtype=np.float64)
    summary["train_breadth"] = {
        "W2_mean_train_ap_delta": float(tr_w2.mean()),
        "W2_improved_train_tasks": int((tr_w2 > 0).sum()),
        "W3_mean_train_ap_delta": float(tr_w3.mean()),
        "W3_improved_train_tasks": int((tr_w3 > 0).sum()),
    }
    return {"rows": rows, "summary": summary}


def write_per_task_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    fields = [
        "task",
        "valid_observed",
        "valid_positive",
        "train_positive",
        "stratum",
        "ap_a0",
        "ap_w2",
        "ap_w3",
        "delta_w2",
        "delta_w3",
        "delta_w3_minus_w2",
        "train_ap_a0",
        "train_ap_w2",
        "train_ap_w3",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fields})
    return C.sha256_file(path)


# ---------------------------------------------------------------------------
# decision labels
# ---------------------------------------------------------------------------


def label_for(delta: float, strata: Mapping[str, Any], *, n_support_ge20: int) -> dict[str, Any]:
    rules = C.DECISION_RULES
    promising = rules["promising"]
    mean20 = strata["support_ge20"]["mean_delta"]
    improved20 = strata["support_ge20"]["improved"]
    checks = {
        "delta_ap40_ge_0.005": bool(delta >= promising["delta_ap40_min"]),
        "mean_delta_support_ge20_ge_0.003": bool(mean20 is not None and mean20 >= promising["mean_delta_support_ge20_min"]),
        "improved_support_ge20_ge_60": bool(improved20 >= promising["improved_support_ge20_min"]),
        "support_ge20_total_is_107": bool(n_support_ge20 == promising["support_ge20_total"]),
    }
    if all(checks[k] for k in ("delta_ap40_ge_0.005", "mean_delta_support_ge20_ge_0.003", "improved_support_ge20_ge_60")):
        label = promising["label"]
    elif delta >= promising["delta_ap40_min"]:
        label = rules["weak"]["label"]
    elif rules["weak"]["delta_ap40_min"] <= delta < rules["weak"]["delta_ap40_max"]:
        label = rules["weak"]["label"]
    elif rules["none"]["delta_ap40_min"] < delta < rules["none"]["delta_ap40_max"]:
        label = rules["none"]["label"]
    elif delta <= rules["regression"]["delta_ap40_max"]:
        label = rules["regression"]["label"]
    else:  # pragma: no cover - the ranges are exhaustive
        label = rules["none"]["label"]
    return {
        "label": label,
        "delta_ap40": float(delta),
        "mean_delta_support_ge20": mean20,
        "improved_support_ge20": int(improved20),
        "n_support_ge20": int(n_support_ge20),
        "checks": checks,
        "rule": {
            "promising": "delta>=+0.005 AND mean(ge20)>=+0.003 AND improved(ge20)>=60",
            "weak": "+0.002<=delta<+0.005 OR delta>=+0.005 without sufficient high-support coverage",
            "none": "-0.002<delta<+0.002",
            "regression": "delta<=-0.002",
        },
        "caveat": "resource-allocation label for one seed, not a significance test",
    }


# ---------------------------------------------------------------------------
# primary epoch-40 records
# ---------------------------------------------------------------------------


def epoch40_record(task: Any, key: str, run_dir: Path, run: Mapping[str, Any], curve: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    row = curve_at(curve, 40) or {}
    fixed = run["fixed_epoch_reports"]["40"]
    train_eval = read_json(run_dir / "train_eval_epoch40.json")
    counts = C.SD.task_counts(task)
    return {
        "schema": "lcrc-pcba-backbone-scaling-epoch40-v1",
        "round_id": C.ROUND_ID,
        "model": key,
        "model_name": C.MODELS[key]["name"],
        "scale": int(run["scale"]),
        "seed": int(run["seed"]),
        "parameters": int(run["parameters"]),
        "trainable_parameter_tensors": int(run["model_audit"]["parameter_tensors"]),
        "status": run["status"],
        "epoch": 40,
        "global_step": int(run["global_steps_run"]),
        "eval_every": int(run.get("eval_every", 0)),
        "eval_epochs_completed": run.get("eval_epochs_completed"),
        "validation": {
            "ap": float(fixed["ap"]),
            "ap_source": "OGB official Evaluator(ogbg-molpcba) on official Validation rows",
            "bce": float(fixed["valid_bce"]),
            "n_tasks_scored": int(fixed["n_tasks_scored"]),
            "n_tasks_with_observed_labels": int(fixed["n_tasks_with_observed_labels"]),
            "evaluated_rows": int(task.valid_rows.size),
            "valid_logits_sha256": fixed.get("valid_logits_sha256"),
            "valid_probs_sha256": fixed.get("valid_probs_sha256"),
            "per_task_ap": fixed["per_task_ap"],
            "positive": fixed["positive"],
            "observed": fixed["observed"],
            "step": fixed.get("step"),
        },
        "train": {
            "ap": float(train_eval["ap"]),
            "bce": float(train_eval["bce"]),
            "n_tasks_scored": int(train_eval["n_tasks_scored"]),
            "evaluated_rows": int(train_eval["evaluated_rows"]),
            "per_task_ap": train_eval["per_task_ap"],
            "logits_sha256": train_eval["logits_sha256"],
        },
        "train_valid_ap_gap": float(train_eval["ap"]) - float(fixed["ap"]),
        "curve_point": {
            "train_bce": row.get("train_bce"),
            "learning_rate": row.get("learning_rate"),
            "epoch_train_seconds": row.get("epoch_train_seconds"),
            "epoch_valid_seconds": row.get("epoch_valid_seconds"),
            "step_ms_p50": row.get("step_ms_p50"),
            "grad_norm_max": row.get("grad_norm_max"),
            "gradient_clip_events": row.get("gradient_clip_events"),
            "gpu_peak_allocated_bytes": row.get("gpu_peak_allocated_bytes"),
        },
        "best_diagnostic_validation": {
            "epoch": int(run["best_epoch"]),
            "ap": float(run["best_valid_ap"]),
            "used_for_primary_comparison": False,
        },
        "checkpoint": {
            "path": run.get("primary_checkpoint"),
            "sha256": run.get("primary_checkpoint_sha256"),
            "best_diagnostic_path": run.get("best_checkpoint"),
            "best_diagnostic_sha256": run.get("best_checkpoint_sha256"),
            "resumable_last_path": run.get("last_checkpoint"),
            "resumable_last_sha256": run.get("last_checkpoint_sha256"),
            "rng_state_sha256": run.get("rng_state_sha256"),
        },
        "resources": {
            "gpu_hours": float(run["gpu_hours"]),
            "epoch_seconds_mean": float(run.get("epoch_seconds_mean") or 0.0),
            "steps_per_second_mean": float(run.get("steps_per_second_mean") or 0.0),
            "gpu_peak_allocated_bytes": int(run.get("gpu_peak_allocated_bytes") or 0),
            "gpu_peak_reserved_bytes": int(run.get("gpu_peak_reserved_bytes") or 0),
            "gpu_occupancy_start": run.get("gpu_occupancy_start"),
        },
        "provenance": {
            "git_commit": run.get("git_commit"),
            "git_state": run.get("git_state"),
            "host_facts": run.get("host_facts"),
            "started_utc": run.get("started_utc"),
            "finished_utc": run.get("finished_utc"),
            "batch_plan": run.get("batch_plan"),
            "recipe": run.get("recipe"),
            "non_finite_events": int(run.get("non_finite_events") or 0),
            "test_evaluated": False,
        },
        "task_counts": counts["valid"],
    }


def protocol_conformance(runs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for key in C.MODEL_ORDER:
        run = runs[key]
        checks[key] = {
            "status_completed": run["status"] == "COMPLETED",
            "epochs_run_40": int(run["epochs_run"]) == 40,
            "global_steps_109520": int(run["global_steps_run"]) == 109_520,
            "parameters_exact": int(run["parameters"]) == C.EXPECTED_PARAMETERS[int(run["scale"])],
            "seed_0": int(run["seed"]) == 0,
            "eval_every_matches_protocol": int(run.get("eval_every", -1)) == int(C.EVAL_CADENCE["every"]),
            "all_eval_epochs_completed": bool(run.get("all_eval_epochs_completed")),
            "plan_matches_frozen": run["batch_plan"]["global_plan_sha256"] == C.A0["global_plan_sha256"],
            "dropout_0_10": abs(float(run["model_dropout"]) - 0.10) < 1e-12,
            "grad_clip_5": float(run["grad_clip"]) == 5.0,
            "precision_fp32": run["precision"] == "fp32",
            "loss_masked_bce": "masked" in str(run["recipe"]["loss"]).lower(),
            "test_evaluated_false": run.get("test_evaluated") is False,
            "no_non_finite": int(run.get("non_finite_events") or 0) == 0,
        }
        checks[key]["passed"] = all(checks[key].values())
    return checks


# ---------------------------------------------------------------------------
# GPU ledger
# ---------------------------------------------------------------------------


def build_ledger(analysis: Path, runs: Mapping[str, Mapping[str, Any]], extra: Mapping[str, Any]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    rr = extra.get("rr_jobs")
    rr_by_exp: dict[str, list[dict[str, Any]]] = {}
    if rr:
        for job in rr.get("jobs", []):
            rr_by_exp.setdefault(str(job.get("experiment")), []).append(job)
    rr_by_exp = {k: sorted(v, key=lambda j: str(j.get("created_at"))) for k, v in rr_by_exp.items()}

    def rr_match(experiment: str) -> dict[str, Any] | None:
        jobs = rr_by_exp.get(experiment, [])
        for job in jobs:
            if job.get("state") == "completed":
                return job
        return jobs[-1] if jobs else None

    probe = extra.get("probe") or {}
    preflight_seconds = float(probe.get("seconds") or 0.0)
    audit_job = rr_match("bs-audit")
    audit_elapsed = float((audit_job or {}).get("elapsed") or 0.0)
    # charge whichever is larger: the job's own wall clock (warm-up, data load,
    # all three audit modes) or the probe's measured span.
    audit_hours = max(audit_elapsed, preflight_seconds) / 3600.0
    entries.append(
        {
            "item": "pretraining_audit_and_cost_probe",
            "experiment": "bs-audit",
            "rr_run_id": None if audit_job is None else audit_job.get("run_id"),
            "rr_elapsed_seconds": audit_elapsed,
            "probe_mode_seconds": preflight_seconds,
            "gpu_hours": audit_hours,
            "source": "rr jobs:bs-audit elapsed (model + a0-replay + eval-probe), cross-checked against evaluation_cost_probe.json:seconds",
            "counted_as": "preflight",
        }
    )
    for key in C.MODEL_ORDER:
        run = runs[key]
        job = rr_match(f"bs-{key.lower()}")
        entries.append(
            {
                "item": f"{key}_formal_40_epoch",
                "model": key,
                "scale": int(run["scale"]),
                "parameters": int(run["parameters"]),
                "rr_run_id": None if job is None else job.get("run_id"),
                "rr_state": None if job is None else job.get("state"),
                "rr_pool": None if job is None else job.get("pool"),
                "rr_cuda_visible_devices": None if job is None else job.get("cuda_visible_devices"),
                "rr_started_at": None if job is None else job.get("started_at"),
                "rr_completed_at": None if job is None else job.get("completed_at"),
                "gpu_hours": float(run["gpu_hours"]),
                "status": run["status"],
                "epochs_run": int(run["epochs_run"]),
                "source": "runs/%s/run.json:gpu_hours" % key,
            }
        )
    total = float(sum(e["gpu_hours"] for e in entries))
    failures = extra.get("failed_attempts") or []
    ledger = {
        "captured_utc": C.utcnow(),
        "entries": entries,
        "failed_attempts": failures,
        "failed_attempt_gpu_hours": float(sum(float(f.get("gpu_hours", 0.0)) for f in failures)),
        "total_gpu_hours_including_failures": total + float(sum(float(f.get("gpu_hours", 0.0)) for f in failures)),
        "budget": dict(C.BUDGET),
        "budget_status": None,
        "note": (
            "GPU-hours are per-run wall clock measured inside the job (gpu_hours in run.json), "
            "which is what the budget charges. See scheduling.json for whether the two formal "
            "runs overlapped or ran serially, and why."
        ),
    }
    spend = ledger["total_gpu_hours_including_failures"]
    ledger["budget_status"] = (
        "WITHIN_TARGET"
        if spend <= float(C.BUDGET["gpu_target_hours"])
        else "WITHIN_HARD_CAP"
        if spend <= float(C.BUDGET["gpu_hard_hours"])
        else "OVER_HARD_CAP"
    )
    return ledger


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default=str(C.RUNS))
    parser.add_argument("--analysis-dir", default=str(C.ANALYSIS))
    parser.add_argument("--rr-jobs", default="")
    parser.add_argument("--out-dir", default=str(C.ANALYSIS))
    args = parser.parse_args(argv)
    runs_dir = Path(args.runs_dir)
    analysis = Path(args.analysis_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    task = C.SD.load_task()
    run_dirs = {"W2": runs_dir / "W2", "W3": runs_dir / "W3", "A0": C.A0_RUN_DIR}
    runs: dict[str, Any] = {k: read_json(p / "run.json") for k, p in run_dirs.items()}
    curves = {k: load_curve(p) for k, p in run_dirs.items() if k != "A0"}
    curves["A0"] = load_curve(C.A0_RUN_DIR)

    ap0 = float(C.A0["ap40"])
    recorded_ap0 = float(runs["A0"]["fixed_epoch_reports"]["40"]["ap"])
    if abs(recorded_ap0 - ap0) > 1e-12:
        raise RuntimeError(f"historical A0 run record AP {recorded_ap0} != frozen {ap0}")
    ap = {"A0": ap0, "W2": float(runs["W2"]["fixed_epoch_reports"]["40"]["ap"]),
          "W3": float(runs["W3"]["fixed_epoch_reports"]["40"]["ap"])}
    delta = {"W2": ap["W2"] - ap0, "W3": ap["W3"] - ap0, "W3_minus_W2": ap["W3"] - ap["W2"]}

    table = per_task_table(task, runs, run_dirs)
    n_ge20 = table["summary"]["stratum_sizes"]["support_ge20"]
    labels = {
        key: label_for(delta[key], table["summary"][key]["strata"], n_support_ge20=n_ge20)
        for key in C.MODEL_ORDER
    }

    eval_curve = {}
    for key in C.MODEL_ORDER:
        eval_curve[key] = [
            {"epoch": row["epoch"], "valid_ap": row["valid_ap"], "valid_bce": row["valid_bce"],
             "train_bce": row["train_bce"], "global_step": row["global_step"],
             "epoch_train_seconds": row["epoch_train_seconds"],
             "epoch_valid_seconds": row["epoch_valid_seconds"],
             "epoch_seconds": row["epoch_seconds"],
             "step_ms_p50": row["step_ms_p50"],
             "learning_rate": row["learning_rate"],
             "grad_norm_max": row["grad_norm_max"], "grad_norm_mean": row["grad_norm_mean"],
             "gradient_clip_events": row["gradient_clip_events"],
             "gpu_peak_allocated_bytes": row["gpu_peak_allocated_bytes"]}
            for row in curves[key]
        ]

    fixed_epochs = [10, 20, 30, 40]
    per_run: dict[str, Any] = {}
    for key in ("A0", "W2", "W3"):
        run = runs[key]
        curve = curves[key]
        metrics = epoch_metrics(run, curve, fixed_epochs)
        train_eval = (
            read_json(C.A0_TRAIN_EVAL)
            if key == "A0"
            else read_json(run_dirs[key] / "train_eval_epoch40.json")
        )
        best = None
        if key == "A0":
            best = {"epoch": int(run["best_epoch"]), "valid_ap": float(run["best_valid_ap"])}
        else:
            evals = [r for r in curve if r.get("valid_ap") is not None]
            if evals:
                top = max(evals, key=lambda r: float(r["valid_ap"]))
                best = {"epoch": int(top["epoch"]), "valid_ap": float(top["valid_ap"])}
        e40 = metrics["40"]
        e30 = metrics["30"]
        run_info = {
            "model": key,
            "scale": int(run.get("scale", 1)),
            "parameters": int(run["parameters"]),
            "seed": int(run["seed"]),
            "status": run["status"],
            "epochs_run": int(run["epochs_run"]),
            "global_steps_run": int(run["global_steps_run"]),
            "eval_epochs_completed": run.get("eval_epochs_completed"),
            "fixed_epoch_metrics": metrics,
            "ap10": None if metrics["10"] is None else metrics["10"]["valid_ap"],
            "ap20": None if metrics["20"] is None else metrics["20"]["valid_ap"],
            "ap30": None if metrics["30"] is None else metrics["30"]["valid_ap"],
            "ap40": ap[key],
            "valid_bce40": float(run["fixed_epoch_reports"]["40"]["valid_bce"]),
            "train_ap40": float(train_eval["ap"]),
            "train_bce40": float(train_eval["bce"]),
            "train_valid_ap_gap": float(train_eval["ap"]) - ap[key],
            "best_diagnostic": best,
            "epoch30_to_40_change": None if e30["valid_ap"] is None else ap[key] - float(e30["valid_ap"]),
            "gpu_hours": float(run["gpu_hours"]),
            "gpu_peak_allocated_bytes": int(run.get("gpu_peak_allocated_bytes") or 0),
            "gpu_peak_reserved_bytes": int(run.get("gpu_peak_reserved_bytes") or 0),
            "epoch_seconds_mean": float(run.get("epoch_seconds_mean") or 0.0),
            "steps_per_second_mean": float(run.get("steps_per_second_mean") or 0.0),
            "grad_clip_events": int(run.get("grad_clip_events") or 0),
            "non_finite_events": int(run.get("non_finite_events") or 0),
            "checkpoint": run.get("primary_checkpoint") or run.get("continuation_checkpoint"),
            "checkpoint_sha256": run.get("primary_checkpoint_sha256") or run.get("continuation_checkpoint_sha256"),
            "commit": run.get("git_commit"),
            "host": (run.get("host_facts") or {}).get("hostname"),
        }
        if key == "A0":
            run_info["eval_epochs_completed"] = [40]
            run_info["gpu_hours"] = float(run["gpu_hours"])
            run_info["ap10"] = None
            run_info["ap20"] = None
            run_info["ap30"] = None
            run_info["checkpoint"] = C.track_rel(C.A0_CHECKPOINT)
            run_info["checkpoint_sha256"] = C.A0["checkpoint_sha256"]
            run_info["commit"] = C.A0["commit"]
            run_info["host"] = "a100-2 (historical run)"
        per_run[key] = run_info

    # last-10-epoch trend: least-squares slope of the dense Validation curve
    trends: dict[str, Any] = {}
    for key in C.MODEL_ORDER:
        evals = [(int(r["epoch"]), float(r["valid_ap"])) for r in eval_curve[key] if r.get("valid_ap") is not None]
        if len(evals) >= 4:
            last = [e for e in evals if e[0] >= C.TOTAL_EPOCHS - 9]
            x = np.asarray([e[0] for e in last], dtype=np.float64)
            y = np.asarray([e[1] for e in last], dtype=np.float64)
            slope = float(np.polyfit(x, y, 1)[0]) if x.size >= 2 else None
            trends[key] = {
                "epochs_in_window": int(x.size),
                "epoch30_to_40_change": per_run[key]["epoch30_to_40_change"],
                "slope_per_epoch_last10": slope,
                "last10_std": float(y.std()) if y.size else None,
                "last10_range": float(y.max() - y.min()) if y.size else None,
                "rising_trend": bool(slope is not None and slope > 0.0005),
            }

    comparison = {
        "captured_utc": C.utcnow(),
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "git_commit": C.git_commit(),
        "git_state": C.git_state(),
        "dataset": C.DATASET,
        "split": "official scaffold",
        "seed": int(C.SEED),
        "primary_metric": "official OGB Validation macro AP at epoch 40 (fixed, no best-epoch substitution)",
        "ap0": ap0,
        "ap": ap,
        "delta": delta,
        "parameters": {k: int(per_run[k]["parameters"]) for k in per_run},
        "n_parameters_vs_a0": {k: int(per_run[k]["parameters"]) / int(C.A0["parameters"]) for k in per_run},
        "per_model": per_run,
        "validation_curve": eval_curve,
        "last10_epoch_trends": trends,
        "per_task": table["summary"],
        "eval_cadence": dict(C.EVAL_CADENCE),
        "decision_labels": labels,
        "efficiency": {
            key: {
                "parameters": int(per_run[key]["parameters"]),
                "gpu_hours": float(per_run[key]["gpu_hours"]),
                "epoch_seconds_mean": float(per_run[key]["epoch_seconds_mean"]),
                "steps_per_second_mean": float(per_run[key]["steps_per_second_mean"]),
                "valid_ap40": ap[key],
                "ap_per_100k_parameters": 100_000.0 * ap[key] / float(per_run[key]["parameters"]),
            }
            for key in per_run
        },
        "answers": {},
        "test_evaluated": False,
    }
    # the four required judgements
    tr = {k: per_run[k]["train_ap40"] for k in per_run}
    comparison["answers"] = {
        "train_ap_increases_with_width": bool(tr["W2"] > tr["A0"] and tr["W3"] > tr["W2"]),
        "train_ap_by_width": tr,
        "validation_ap_increases_with_width": bool(ap["W2"] > ap0 and ap["W3"] > ap["W2"]),
        "validation_ap_by_width": ap,
        "w3_better_than_w2": bool(ap["W3"] > ap["W2"]),
        "delta_w3_minus_w2": delta["W3_minus_W2"],
        "gain_broad_or_sparse_concentrated": {
            key: {
                "mean_delta_support_ge20": table["summary"][key]["strata"]["support_ge20"]["mean_delta"],
                "improved_support_ge20": table["summary"][key]["strata"]["support_ge20"]["improved"],
                "n_support_ge20": table["summary"][key]["strata"]["support_ge20"]["n"],
                "mean_delta_support_5_19": table["summary"][key]["strata"]["support_5_19"]["mean_delta"],
                "top10_share_of_positive_mass": table["summary"][key]["top10_share_of_positive_mass"],
                "improved_overall": table["summary"][key]["improved"],
            }
            for key in C.MODEL_ORDER
        },
    }

    probe = read_json(analysis / "evaluation_cost_probe.json") if (analysis / "evaluation_cost_probe.json").exists() else None
    rr_jobs = read_json(args.rr_jobs) if args.rr_jobs else None
    failed_path = analysis / "failed_attempts.json"
    extra = {
        "probe": probe,
        "rr_jobs": rr_jobs,
        "failed_attempts": read_json(failed_path) if failed_path.exists() else [],
    }
    ledger = build_ledger(analysis, runs, extra)

    decision = {
        "captured_utc": C.utcnow(),
        "round_id": C.ROUND_ID,
        "ap0": ap0,
        "ap_w2": ap["W2"],
        "ap_w3": ap["W3"],
        "d_w2": delta["W2"],
        "d_w3": delta["W3"],
        "d_32": delta["W3_minus_W2"],
        "labels": {k: labels[k]["label"] for k in C.MODEL_ORDER},
        "label_evidence": labels,
        "eval_cadence": dict(C.EVAL_CADENCE),
        "budget": {
            "total_gpu_hours": ledger["total_gpu_hours_including_failures"],
            "status": ledger["budget_status"],
            "target": C.BUDGET["gpu_target_hours"],
            "hard_cap": C.BUDGET["gpu_hard_hours"],
            "failed_retries": len(ledger["failed_attempts"]),
        },
        "answers": comparison["answers"],
        "conclusions": conclusion_text(comparison, labels),
        "not_established": [
            "no Test metric was produced; Test labels are NaN in the frozen artifact",
            "one seed per model: epoch-level AP noise is ~0.0014-0.0026, so |delta| < 0.002 is not interpretable",
            "a capacity null result would not prove LCRC has reached a capacity limit: the from-scratch recipe may simply not suit wider models",
            "the two new models are different shapes, so no per-tensor paired-init statement is made",
        ],
        "test_evaluated": False,
    }

    comparison["protocol_conformance"] = protocol_conformance(runs)
    comparison["protocol_conformance_passed"] = all(
        v["passed"] for v in comparison["protocol_conformance"].values()
    )
    C.write_json(out_dir / "scale_comparison.json", comparison)
    for key in C.MODEL_ORDER:
        C.write_json(
            out_dir / f"{key.lower()}_epoch40.json",
            epoch40_record(task, key, run_dirs[key], runs[key], curves[key]),
        )
    C.write_json(out_dir / "decision.json", decision)
    C.write_json(out_dir / "gpu_ledger.json", ledger)
    csv_sha = write_per_task_csv(out_dir / "per_task_deltas.csv", table["rows"])
    C.log(f"comparison written: AP0={ap0:.6f} W2={ap['W2']:.6f} W3={ap['W3']:.6f}")
    C.log(f"labels {decision['labels']}  per_task_csv_sha256={csv_sha}")
    return 0


def conclusion_text(comparison: Mapping[str, Any], labels: Mapping[str, Mapping[str, Any]]) -> list[str]:
    ap = comparison["ap"]
    d = comparison["delta"]
    tr = comparison["answers"]["train_ap_by_width"]
    out = []
    for key in C.MODEL_ORDER:
        out.append(
            f"{key}: AP40 {ap[key]:.6f} (delta {d[key]:+.6f}), Train AP40 {tr[key]:.6f}, "
            f"label {labels[key]['label']}"
        )
    out.append(f"W3 - W2 = {d['W3_minus_W2']:+.6f}")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
