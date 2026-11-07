"""Post-run comparison, per-task table, decision and GPU ledger for CPE v1.

Reads only committed-or-pulled artifacts:

* ``lcrc_pcba_backbone_scaling_v1/runs/W2`` (historical, read-only) -- never re-run
* ``lcrc_pcba_cpe_v1/runs/P`` and ``.../runs/S`` (git-ignored run directories)

Writes ``analysis/{p_epoch40,s_epoch40,paired_comparison,paired_training_audit,
per_task_deltas.csv,decision,gpu_ledger}.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import cpe_common as C


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def read_json(path: Path | str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_curve(run_dir: Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / "train_curve.csv"
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            row: dict[str, Any] = {}
            for k, v in raw.items():
                if k == "evaluated":
                    row[k] = str(v) == "True"
                elif k == "is_best":
                    row[k] = str(v) == "True"
                elif v == "":
                    row[k] = None
                else:
                    try:
                        row[k] = float(v) if ("." in v or "e" in v or "E" in v) else int(v)
                    except ValueError:
                        row[k] = v
            rows.append(row)
    return rows


def curve_at(curve: Sequence[Mapping[str, Any]], epoch: int) -> dict[str, Any] | None:
    for row in curve:
        if int(row.get("epoch", -1)) == int(epoch):
            return dict(row)
    return None


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def epoch_metrics(curve: Sequence[Mapping[str, Any]], epochs: Sequence[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for e in epochs:
        row = curve_at(curve, e)
        out[f"epoch_{e}"] = None if row is None else _finite(row.get("valid_ap"))
    return out


def _stratum(positives: int) -> str:
    if positives >= 20:
        return "support_ge20"
    if positives >= 5:
        return "support_5_19"
    return "support_lt5"


def _summary(deltas: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    d = np.asarray(deltas, dtype=np.float64)
    if mask is not None:
        d = d[np.asarray(mask, dtype=bool)]
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {"n": 0}
    return {
        "n": int(d.size),
        "improved": int((d > 0).sum()),
        "regressed": int((d < 0).sum()),
        "tied": int((d == 0).sum()),
        "mean": float(d.mean()),
        "median": float(np.median(d)),
        "std": float(d.std(ddof=0)),
        "min": float(d.min()),
        "max": float(d.max()),
        "sum_positive": float(d[d > 0].sum()),
    }


def _concentration(deltas: np.ndarray) -> dict[str, float]:
    d = np.asarray(deltas, dtype=np.float64)
    d = d[np.isfinite(d)]
    pos = np.sort(d[d > 0])[::-1]
    total = float(pos.sum()) if pos.size else 0.0
    return {
        "positive_sum": total,
        "top5_share": float(pos[:5].sum() / total) if total > 0 else 0.0,
        "top10_share": float(pos[:10].sum() / total) if total > 0 else 0.0,
        "n_positive": int(pos.size),
    }


# ---------------------------------------------------------------------------
# per-task table
# ---------------------------------------------------------------------------


def per_task_table(
    w2: Mapping[str, Any],
    p: Mapping[str, Any],
    s: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """One row per task with the four APs, three deltas and the support stratum."""
    rep_w2 = w2["fixed_epoch_reports"]["40"]
    rep_p = p["fixed_epoch_reports"]["40"]
    rep_s = s["fixed_epoch_reports"]["40"]
    rows: list[dict[str, Any]] = []
    for t in range(C.N_TASKS):
        ap_w2 = rep_w2["per_task_ap"][t]
        ap_p = rep_p["per_task_ap"][t]
        ap_s = rep_s["per_task_ap"][t]
        pos = int(rep_w2["positive"][t])
        stratum = _stratum(pos) if pos > 0 else "unscorable"
        row = {
            "task": t,
            "validation_positive": pos,
            "validation_observed": int(rep_w2["observed"][t]),
            "train_positive": int(w2["task_counts"]["train"]["positive"][t]),
            "stratum": stratum,
            "ap_w2": "" if ap_w2 is None else float(ap_w2),
            "ap_p": "" if ap_p is None else float(ap_p),
            "ap_s": "" if ap_s is None else float(ap_s),
        }
        for name, a, b in (
            ("delta_p_minus_w2", ap_p, ap_w2),
            ("delta_s_minus_w2", ap_s, ap_w2),
            ("delta_p_minus_s", ap_p, ap_s),
        ):
            row[name] = "" if (a is None or b is None) else float(a) - float(b)
        rows.append(row)
    return rows


def write_per_task_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    fields = [
        "task",
        "validation_positive",
        "validation_observed",
        "train_positive",
        "stratum",
        "ap_w2",
        "ap_p",
        "ap_s",
        "delta_p_minus_w2",
        "delta_s_minus_w2",
        "delta_p_minus_s",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    return C.sha256_file(path)


def _task_arrays(rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    def col(name: str) -> np.ndarray:
        return np.asarray([float(r[name]) if r[name] != "" else np.nan for r in rows], dtype=np.float64)

    pos = np.asarray([int(r["validation_positive"]) for r in rows], dtype=np.int64)
    return {
        "pos": pos,
        "ap_w2": col("ap_w2"),
        "ap_p": col("ap_p"),
        "ap_s": col("ap_s"),
        "d_pw": col("delta_p_minus_w2"),
        "d_sw": col("delta_s_minus_w2"),
        "d_ps": col("delta_p_minus_s"),
    }


# ---------------------------------------------------------------------------
# decision
# ---------------------------------------------------------------------------


def decide(ap_w2: float, ap_p: float, ap_s: float, table: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    arr = _task_arrays(table)
    scored = np.isfinite(arr["d_pw"])
    high = scored & (arr["pos"] >= int(C.DECISION_RULES["support_strata"]["ge20"]))
    mid = scored & (arr["pos"] >= 5) & (arr["pos"] < 20)
    low = scored & (arr["pos"] < 5)
    d_base = float(ap_p) - float(ap_w2)
    d_mech = float(ap_p) - float(ap_s)
    mean_high = float(np.nanmean(arr["d_pw"][high])) if high.any() else float("nan")
    improved_high = int((arr["d_pw"][high] > 0).sum()) if high.any() else 0
    n_high = int(high.sum())
    rules = C.DECISION_RULES["promising"]
    broad_high = bool(
        mean_high >= float(rules["mean_delta_support_ge20_min"])
        and improved_high >= int(rules["improved_support_ge20_min"])
    )
    checks = {
        "delta_base_ge_0.005": bool(d_base >= float(rules["delta_base_min"])),
        "delta_mech_ge_0.003": bool(d_mech >= float(rules["delta_mech_min"])),
        "high_support_mean_ge_0.003": bool(mean_high >= float(rules["mean_delta_support_ge20_min"])),
        "high_support_improved_ge_60_of_107": bool(
            improved_high >= int(rules["improved_support_ge20_min"]) and n_high == int(rules["support_ge20_total"])
        ),
        "support_ge20_total_is_107": n_high == int(rules["support_ge20_total"]),
    }
    promising = all(checks[k] for k in ("delta_base_ge_0.005", "delta_mech_ge_0.003", "high_support_mean_ge_0.003", "high_support_improved_ge_60_of_107"))

    flags = {
        "P_beats_W2": bool(d_base > 0.0),
        "S_beats_W2": bool(float(ap_s) - float(ap_w2) > 0.0),
        "P_beats_S": bool(d_mech > 0.0),
        "P_better_than_S_by_0.003": bool(d_mech >= 0.003),
        "high_support_mean_below_0.003": bool(mean_high < 0.003),
        "high_support_improved_below_60": bool(improved_high < 60),
        "broad_high_support_gain": broad_high,
        "gains_only_in_low_support": bool(
            (arr["d_pw"][mid].mean() if mid.any() else 0.0) > 0.003
            and (arr["d_pw"][low].mean() if low.any() else 0.0) > 0.0
            and not broad_high
        ),
        "regression_threshold": bool(d_base <= -0.002),
        "no_clear_band": bool(abs(d_base) < 0.002),
    }

    if flags["regression_threshold"]:
        label = C.DECISION_RULES["regression"]["label"]
    elif promising:
        label = rules["label"]
    elif d_base < float(C.DECISION_RULES["no_clear"]["delta_base_max_exclusive"]) and not broad_high:
        label = C.DECISION_RULES["no_clear"]["label"]
    elif d_base > 0.0 and flags["S_beats_W2"] and not flags["P_better_than_S_by_0.003"]:
        label = C.DECISION_RULES["capacity_without_advantage"]["label"]
    else:
        label = C.DECISION_RULES["weak_or_mixed"]["label"]

    return {
        "schema": "lcrc-pcba-cpe-decision-v1",
        "round_id": C.ROUND_ID,
        "ap_w2": float(ap_w2),
        "ap_p": float(ap_p),
        "ap_s": float(ap_s),
        "d_base": d_base,
        "d_mech": d_mech,
        "d_s_vs_w2": float(ap_s) - float(ap_w2),
        "label": label,
        "checks": checks,
        "flags": flags,
        "high_support": {
            "n": n_high,
            "mean_delta_p_minus_w2": mean_high,
            "improved": improved_high,
            "median_delta_p_minus_w2": float(np.nanmedian(arr["d_pw"][high])) if high.any() else float("nan"),
        },
        "rounded_medians_note": "thresholds are resource decisions for one seed, not significance tests",
        "required_reporting_disclaimer": [
            "single-seed architecture exploration",
            "historical W2 baseline",
            "same-round paired P/S mechanism control",
            "not independently confirmed",
        ],
        "test_evaluated": False,
    }


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------


def epoch40_record(key: str, run: Mapping[str, Any], curve: Sequence[Mapping[str, Any]], train_eval: Mapping[str, Any] | None) -> dict[str, Any]:
    formal = {f"epoch_{e}": None for e in (1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 40)}
    formal.update(epoch_metrics(curve, (1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 40)))
    return {
        "schema": "lcrc-pcba-cpe-epoch40-v1",
        "round_id": C.ROUND_ID,
        "model": key,
        "model_name": run["model_name"],
        "role": run["role"],
        "pair_mode": run["pair_mode"],
        "status": run["status"],
        "parameters": int(run["parameters"]),
        "base_parameters": int(run["base_parameters"]),
        "new_parameters": int(run["new_parameters"]),
        "seed": int(run["seed"]),
        "epochs_run": int(run["epochs_run"]),
        "eval_every": int(run["eval_every"]),
        "all_eval_epochs_completed": bool(run["all_eval_epochs_completed"]),
        "ap_epochs": formal,
        "ap40": float(run["fixed_epoch_reports"]["40"]["ap"]),
        "valid_bce40": float(run["fixed_epoch_reports"]["40"]["valid_bce"]),
        "n_tasks_scored40": int(run["fixed_epoch_reports"]["40"]["n_tasks_scored"]),
        "best_epoch": run.get("best_epoch"),
        "best_valid_ap": run.get("best_valid_ap"),
        "train_ap40": None if train_eval is None else float(train_eval["ap"]),
        "train_bce40": None if train_eval is None else float(train_eval["bce"]),
        "train_valid_ap_gap": None
        if train_eval is None
        else float(train_eval["ap"]) - float(run["fixed_epoch_reports"]["40"]["ap"]),
        "curve": [
            {
                "epoch": int(r["epoch"]),
                "valid_ap": r.get("valid_ap"),
                "valid_bce": r.get("valid_bce"),
                "train_bce": r.get("train_bce"),
                "alpha_A_l2": r.get("alpha_A_l2"),
                "alpha_B_l2": r.get("alpha_B_l2"),
                "alpha_pair_l2": r.get("alpha_pair_l2"),
                "pair_abs_mean": r.get("pair_abs_mean"),
                "main_residual_abs_mean": r.get("main_residual_abs_mean"),
                "pair_residual_abs_mean": r.get("pair_residual_abs_mean"),
            }
            for r in curve
        ],
        "epoch0_diagnostic": run.get("epoch0_diagnostic"),
        "resources": {
            "gpu_hours": float(run["gpu_hours"]),
            "gpu_seconds": float(run["gpu_seconds"]),
            "train_seconds": float(run["train_seconds"]),
            "valid_seconds": float(run["valid_seconds"]),
            "epoch_seconds_mean": float(run["epoch_seconds_mean"]),
            "steps_per_second_mean": float(run["steps_per_second_mean"]),
            "step_ms_p50_mean": float(np.mean([r["step_ms_p50"] for r in curve if r.get("step_ms_p50")])),
            "gpu_peak_allocated_bytes": int(run["gpu_peak_allocated_bytes"]),
            "gpu_peak_reserved_bytes": int(run["gpu_peak_reserved_bytes"]),
            "grad_clip_events": int(run["grad_clip_events"]),
        },
        "checkpoints": {
            "primary": run.get("primary_checkpoint"),
            "primary_sha256": run.get("primary_checkpoint_sha256"),
            "best_diagnostic": run.get("best_checkpoint"),
            "best_diagnostic_sha256": run.get("best_checkpoint_sha256"),
            "last": run.get("last_checkpoint"),
            "last_sha256": run.get("last_checkpoint_sha256"),
            "rng_state": run.get("rng_state"),
            "rng_state_sha256": run.get("rng_state_sha256"),
        },
        "batch_plan": run.get("batch_plan"),
        "evidence_diagnostics_epoch40": run.get("evidence_diagnostics_epoch40"),
        "evidence_diagnostics_path": run.get("evidence_diagnostics"),
        "evidence_diagnostics_sha256": run.get("evidence_diagnostics_sha256"),
        "recipe": run.get("recipe"),
        "git_commit": run.get("git_commit"),
        "environment": run.get("environment"),
        "host_facts": run.get("host_facts"),
        "started_utc": run.get("started_utc"),
        "finished_utc": run.get("finished_utc"),
        "test_evaluated": False,
    }


def w2_record(run: Mapping[str, Any], curve: Sequence[Mapping[str, Any]], train_eval: Mapping[str, Any] | None) -> dict[str, Any]:
    formal = {f"epoch_{e}": None for e in (1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 40)}
    formal.update(epoch_metrics(curve, (1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 40)))
    return {
        "source": C.track_rel(C.W2_RUN_DIR),
        "model": "W2",
        "parameters": int(run["parameters"]),
        "ap_epochs": formal,
        "ap40": float(run["fixed_epoch_reports"]["40"]["ap"]),
        "valid_bce40": float(run["fixed_epoch_reports"]["40"]["valid_bce"]),
        "train_ap40": None if train_eval is None else float(train_eval["ap"]),
        "train_bce40": None if train_eval is None else float(train_eval["bce"]),
        "best_epoch": run.get("best_epoch"),
        "best_valid_ap": run.get("best_valid_ap"),
        "curve": [
            {"epoch": int(r["epoch"]), "valid_ap": r.get("valid_ap"), "valid_bce": r.get("valid_bce"), "train_bce": r.get("train_bce")}
            for r in curve
        ],
        "resources": {
            "gpu_hours": float(run["gpu_hours"]),
            "train_seconds": float(run["train_seconds"]),
            "valid_seconds": float(run["valid_seconds"]),
            "epoch_seconds_mean": float(run["epoch_seconds_mean"]),
            "steps_per_second_mean": float(run["steps_per_second_mean"]),
            "step_ms_p50_mean": float(np.mean([r["step_ms_p50"] for r in curve if r.get("step_ms_p50")])),
            "gpu_peak_allocated_bytes": int(run["gpu_peak_allocated_bytes"]),
            "gpu_peak_reserved_bytes": int(run["gpu_peak_reserved_bytes"]),
        },
        "checkpoint_sha256": run.get("primary_checkpoint_sha256"),
        "batch_plan": run.get("batch_plan"),
        "git_commit": run.get("git_commit"),
        "not_retrained": True,
    }


# ---------------------------------------------------------------------------
# paired training audit
# ---------------------------------------------------------------------------


def _shared_recipe(recipe: Mapping[str, Any]) -> dict[str, Any]:
    """The recipe with the arm identity fields removed (they must differ)."""
    return {
        k: v
        for k, v in recipe.items()
        if k not in ("model", "model_name", "role", "pair_mode", "new_module")
    }


def training_parity(
    w2: Mapping[str, Any],
    p: Mapping[str, Any],
    s: Mapping[str, Any],
    curves: Mapping[str, Sequence[Mapping[str, Any]]],
    diag: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    base_state_p = p["model_audit"]["base_state_sha256"]
    base_state_s = s["model_audit"]["base_state_sha256"]
    checks = {
        "same_batch_plan": bool(
            p["batch_plan"]["global_plan_sha256"] == s["batch_plan"]["global_plan_sha256"] == C.BASE_GLOBAL_PLAN_SHA256
        ),
        "same_consumed_prefix": bool(
            p["batch_plan"]["prefix_sha256"] == s["batch_plan"]["prefix_sha256"] == C.BASE_CONSUMED40_PREFIX_SHA256
        ),
        "same_recipe": bool(_shared_recipe(p["recipe"]) == _shared_recipe(s["recipe"])),
        "same_parameters": bool(p["parameters"] == s["parameters"] == 1_027_732),
        "same_seed": bool(p["seed"] == s["seed"] == 0),
        "same_eval_cadence": bool(p["eval_every"] == s["eval_every"] == 1),
        "same_base_state_hash": bool(base_state_p == base_state_s),
        "same_trunk_fingerprint_as_historical_w2": bool(
            p["model_audit"]["base_trunk_init_fingerprint"]
            == s["model_audit"]["base_trunk_init_fingerprint"]
            == w2["model_audit"]["trunk_init_fingerprint"]
            == C.BASE_TRUNK_INIT_FINGERPRINT
        ),
        "same_full_trunk_fingerprint_as_historical_w2": bool(
            p["model_audit"]["base_full_trunk_init_fingerprint"]
            == s["model_audit"]["base_full_trunk_init_fingerprint"]
            == w2["model_audit"]["full_trunk_init_fingerprint"]
            == C.BASE_FULL_TRUNK_INIT_FINGERPRINT
        ),
        "both_completed_40_epochs": bool(
            p["status"] == s["status"] == "COMPLETED"
            and p["epochs_run"] == s["epochs_run"] == 40
            and p["all_eval_epochs_completed"]
            and s["all_eval_epochs_completed"]
        ),
        "epoch1_rng_equal": bool(
            curves["P"][0].get("rng_cpu_sha256") == curves["S"][0].get("rng_cpu_sha256")
        ),
        "test_never_evaluated": bool(not p["test_evaluated"] and not s["test_evaluated"]),
        "train_valid_rows_disjoint": bool(
            p["training_order_isolation"]["passed"] and s["training_order_isolation"]["passed"]
        ),
    }
    epoch0 = {
        "W2_replay": None if w2.get("epoch0_diagnostic") is None else float(w2["epoch0_diagnostic"]["ap"]),
        "P": None if p.get("epoch0_diagnostic") is None else float(p["epoch0_diagnostic"]["ap"]),
        "S": None if s.get("epoch0_diagnostic") is None else float(s["epoch0_diagnostic"]["ap"]),
        "W2_replay_source": "historical W2 run.json epoch0_diagnostic (not re-run in this round)",
    }
    vals = [v for v in (epoch0["W2_replay"], epoch0["P"], epoch0["S"]) if v is not None]
    if len(vals) >= 2:
        epoch0["max_abs_diff"] = max(vals) - min(vals)
        epoch0["tolerance"] = 1e-4
        epoch0["within_tolerance"] = bool(epoch0["max_abs_diff"] <= 1e-4)
    checks["epoch0_ap_parity"] = bool(epoch0.get("within_tolerance", True))
    return {
        "schema": "lcrc-pcba-cpe-paired-training-audit-v1",
        "round_id": C.ROUND_ID,
        "checks": checks,
        "all_passed": bool(all(checks.values())),
        "epoch0_validation_ap": epoch0,
        "batch_plan": {
            "global_plan_sha256": p["batch_plan"]["global_plan_sha256"],
            "prefix_sha256": p["batch_plan"]["prefix_sha256"],
            "expected_global": C.BASE_GLOBAL_PLAN_SHA256,
            "expected_prefix": C.BASE_CONSUMED40_PREFIX_SHA256,
        },
        "base_state_sha256": {
            "P": p["model_audit"]["base_state_sha256"],
            "S": s["model_audit"]["base_state_sha256"],
            "historical_W2": None,
            "historical_W2_note": (
                "the historical W2 run.json does not record a full base-state SHA-256; the "
                "bitwise subset equality is proved in analysis/initialization_parity.json "
                "and the trunk init fingerprints are compared here"
            ),
        },
        "init_fingerprints": {
            "P": p["model_audit"]["base_trunk_init_fingerprint"],
            "S": s["model_audit"]["base_trunk_init_fingerprint"],
            "historical_W2": w2["model_audit"]["trunk_init_fingerprint"],
            "frozen": C.BASE_TRUNK_INIT_FINGERPRINT,
            "full": {
                "P": p["model_audit"]["base_full_trunk_init_fingerprint"],
                "S": s["model_audit"]["base_full_trunk_init_fingerprint"],
                "historical_W2": w2["model_audit"]["full_trunk_init_fingerprint"],
                "frozen": C.BASE_FULL_TRUNK_INIT_FINGERPRINT,
            },
        },
        "model_state_sha256": {"P": p["model_audit"]["model_state_sha256"], "S": s["model_audit"]["model_state_sha256"]},
        "new_module_init": {"P": p["model_audit"]["new_module_init"], "S": s["model_audit"]["new_module_init"]},
        "recipe": {"P": p["recipe"], "S": s["recipe"]},
        "rng": {
            "P_start": p.get("rng_state_hashes_start"),
            "S_start": s.get("rng_state_hashes_start"),
            "P_end": p.get("rng_state_hashes_end"),
            "S_end": s.get("rng_state_hashes_end"),
        },
        "diagnostics_present": sorted(diag.keys()),
    }


# ---------------------------------------------------------------------------
# GPU ledger
# ---------------------------------------------------------------------------


def gpu_ledger(rr_jobs: Mapping[str, Any] | None, runs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    jobs: list[dict[str, Any]] = []
    total = 0.0
    if rr_jobs is not None:
        for job in rr_jobs.get("jobs", []):
            exp = str(job.get("experiment") or "")
            if not exp.startswith("cpe-"):
                continue
            started = job.get("started_at")
            completed = job.get("completed_at")
            seconds = 0.0
            if started and completed:
                seconds = max(0.0, (datetime.fromisoformat(completed) - datetime.fromisoformat(started)).total_seconds())
            jobs.append(
                {
                    "run_id": job.get("run_id"),
                    "experiment": exp,
                    "state": job.get("state"),
                    "pool": job.get("pool"),
                    "pool_or_host": job.get("pool_or_host"),
                    "cuda_visible_devices": job.get("cuda_visible_devices"),
                    "node": job.get("node"),
                    "gpu_names": job.get("gpu_names"),
                    "driver_version": job.get("driver_version"),
                    "torch_version": job.get("torch_version"),
                    "torch_cuda": job.get("torch_cuda"),
                    "commit": job.get("commit"),
                    "created_at": job.get("created_at"),
                    "started_at": started,
                    "completed_at": completed,
                    "exit_code": job.get("exit_code"),
                    "wall_seconds": seconds,
                    "wall_hours": seconds / 3600.0,
                }
            )
            total += seconds
    formal = {
        key: {
            "run_dir": C.track_rel(C.RUNS / key),
            "gpu_hours_process": float(run["gpu_hours"]),
            "gpu_seconds_process": float(run["gpu_seconds"]),
            "params": int(run["parameters"]),
            "checkpoint_sha256": run.get("primary_checkpoint_sha256"),
        }
        for key, run in runs.items()
    }
    return {
        "schema": "lcrc-pcba-cpe-gpu-ledger-v1",
        "round_id": C.ROUND_ID,
        "jobs": jobs,
        "jobs_total_wall_hours": total / 3600.0,
        "formal_runs": formal,
        "formal_process_hours": sum(v["gpu_hours_process"] for v in formal.values()),
        "total_charged_hours": total / 3600.0,
        "budget": dict(C.BUDGET),
        "budget_status": "WITHIN_TARGET"
        if total / 3600.0 <= C.BUDGET["gpu_target_hours"]
        else "WITHIN_HARD_CAP"
        if total / 3600.0 <= C.BUDGET["gpu_hard_hours"]
        else "OVER_HARD_CAP",
        "note": "wall clock includes queue/setup/audit; formal process hours are the measured in-run GPU seconds",
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default=str(C.RUNS))
    parser.add_argument("--rr-jobs", default="")
    parser.add_argument("--out-dir", default=str(C.ANALYSIS))
    args = parser.parse_args(argv)
    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    w2_json = read_json(C.W2_RUN_JSON)
    p_json = read_json(runs_root / "P" / "run.json")
    s_json = read_json(runs_root / "S" / "run.json")
    w2_curve = load_curve(C.W2_RUN_DIR)
    curves = {"P": load_curve(runs_root / "P"), "S": load_curve(runs_root / "S")}
    w2_train_eval = read_json(C.W2_TRAIN_EVAL) if C.W2_TRAIN_EVAL.exists() else None
    p_train_eval = read_json(runs_root / "P" / "train_eval_epoch40.json") if (runs_root / "P" / "train_eval_epoch40.json").exists() else None
    s_train_eval = read_json(runs_root / "S" / "train_eval_epoch40.json") if (runs_root / "S" / "train_eval_epoch40.json").exists() else None
    diag = {}
    for key in C.MODEL_ORDER:
        path = runs_root / key / "evidence_diagnostics.json"
        if path.exists():
            diag[key] = read_json(path)

    records = {
        "P": epoch40_record("P", p_json, curves["P"], p_train_eval),
        "S": epoch40_record("S", s_json, curves["S"], s_train_eval),
    }
    for key in C.MODEL_ORDER:
        C.write_json(out_dir / f"{key.lower()}_epoch40.json", records[key])

    table = per_task_table(w2_json, p_json, s_json)
    csv_sha = write_per_task_csv(out_dir / "per_task_deltas.csv", table)
    arr = _task_arrays(table)
    scored = np.isfinite(arr["d_pw"])
    high = scored & (arr["pos"] >= 20)
    mid = scored & (arr["pos"] >= 5) & (arr["pos"] < 20)
    low = scored & (arr["pos"] < 5)

    ap_w2 = float(w2_json["fixed_epoch_reports"]["40"]["ap"])
    ap_p = float(p_json["fixed_epoch_reports"]["40"]["ap"])
    ap_s = float(s_json["fixed_epoch_reports"]["40"]["ap"])
    decision = decide(ap_w2, ap_p, ap_s, table)

    w2_curve_metrics = epoch_metrics(w2_curve, (10, 20, 25, 30, 40))
    comparison = {
        "schema": "lcrc-pcba-cpe-paired-comparison-v1",
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "primary_metric": "official OGB Validation macro AP at epoch 40",
        "ap": {
            "W2_historical": ap_w2,
            "P": ap_p,
            "S": ap_s,
            "d_base_P_minus_W2": ap_p - ap_w2,
            "d_mech_P_minus_S": ap_p - ap_s,
            "d_S_minus_W2": ap_s - ap_w2,
        },
        "ap_by_epoch": {
            "W2": {**{f"epoch_{e}": None for e in (1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 40)}, **w2_curve_metrics, **epoch_metrics(w2_curve, (1, 2, 3, 5, 15, 35))},
            "P": {f"epoch_{e}": records["P"]["ap_epochs"].get(f"epoch_{e}") for e in (1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 40)},
            "S": {f"epoch_{e}": records["S"]["ap_epochs"].get(f"epoch_{e}") for e in (1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 40)},
        },
        "train_valid": {
            "W2": {"train_ap40": None if w2_train_eval is None else float(w2_train_eval["ap"]), "train_bce40": None if w2_train_eval is None else float(w2_train_eval["bce"]), "valid_ap40": ap_w2, "valid_bce40": float(w2_json["fixed_epoch_reports"]["40"]["valid_bce"])},
            "P": {"train_ap40": None if p_train_eval is None else float(p_train_eval["ap"]), "train_bce40": None if p_train_eval is None else float(p_train_eval["bce"]), "valid_ap40": ap_p, "valid_bce40": float(p_json["fixed_epoch_reports"]["40"]["valid_bce"])},
            "S": {"train_ap40": None if s_train_eval is None else float(s_train_eval["ap"]), "train_bce40": None if s_train_eval is None else float(s_train_eval["bce"]), "valid_ap40": ap_s, "valid_bce40": float(s_json["fixed_epoch_reports"]["40"]["valid_bce"])},
        },
        "per_task": {
            "P_minus_W2": {
                "all": _summary(arr["d_pw"], scored),
                "support_ge20": _summary(arr["d_pw"], high),
                "support_5_19": _summary(arr["d_pw"], mid),
                "support_lt5": _summary(arr["d_pw"], low),
                "concentration": _concentration(arr["d_pw"][scored]),
                "spearman_delta_vs_support": float(
                    np.corrcoef(
                        np.argsort(np.argsort(arr["d_pw"][scored])),
                        np.argsort(np.argsort(arr["pos"][scored].astype(np.float64))),
                    )[0, 1]
                ),
            },
            "S_minus_W2": {
                "all": _summary(arr["d_sw"], scored),
                "support_ge20": _summary(arr["d_sw"], high),
                "support_5_19": _summary(arr["d_sw"], mid),
                "support_lt5": _summary(arr["d_sw"], low),
                "concentration": _concentration(arr["d_sw"][scored]),
            },
            "P_minus_S": {
                "all": _summary(arr["d_ps"], scored),
                "support_ge20": _summary(arr["d_ps"], high),
                "support_5_19": _summary(arr["d_ps"], mid),
                "support_lt5": _summary(arr["d_ps"], low),
                "concentration": _concentration(arr["d_ps"][scored]),
            },
            "n_scored": int(scored.sum()),
            "n_unscorable": int((~scored).sum()),
            "stratum_sizes": {
                "support_ge20": int(high.sum()),
                "support_5_19": int(mid.sum()),
                "support_lt5": int(low.sum()),
            },
        },
        "best_diagnostic": {
            "W2": {"epoch": w2_json.get("best_epoch"), "ap": w2_json.get("best_valid_ap"), "used_for_selection": False},
            "P": {"epoch": p_json.get("best_epoch"), "ap": p_json.get("best_valid_ap"), "used_for_selection": False},
            "S": {"epoch": s_json.get("best_epoch"), "ap": s_json.get("best_valid_ap"), "used_for_selection": False},
        },
        "last_10_epochs": {
            key: _last10(curves[key]) for key in C.MODEL_ORDER
        } | {"W2": _last10(w2_curve)},
        "cost": {
            "W2_historical": w2_json["gpu_hours"],
            "P": p_json["gpu_hours"],
            "S": s_json["gpu_hours"],
            "P_over_W2_ratio": float(p_json["gpu_hours"] / w2_json["gpu_hours"]),
            "epoch_seconds_mean": {"W2": w2_json["epoch_seconds_mean"], "P": p_json["epoch_seconds_mean"], "S": s_json["epoch_seconds_mean"]},
            "step_ms_p50_mean": {
                "W2": float(np.mean([r["step_ms_p50"] for r in w2_curve if r.get("step_ms_p50")])),
                "P": float(np.mean([r["step_ms_p50"] for r in curves["P"] if r.get("step_ms_p50")])),
                "S": float(np.mean([r["step_ms_p50"] for r in curves["S"] if r.get("step_ms_p50")])),
            },
            "valid_seconds_total": {"W2": w2_json["valid_seconds"], "P": p_json["valid_seconds"], "S": s_json["valid_seconds"]},
            "peak_allocated_bytes": {"W2": w2_json["gpu_peak_allocated_bytes"], "P": p_json["gpu_peak_allocated_bytes"], "S": s_json["gpu_peak_allocated_bytes"]},
            "peak_reserved_bytes": {"W2": w2_json["gpu_peak_reserved_bytes"], "P": p_json["gpu_peak_reserved_bytes"], "S": s_json["gpu_peak_reserved_bytes"]},
            "P_vs_S_speed_ratio": float(p_json["epoch_seconds_mean"] / s_json["epoch_seconds_mean"]),
        },
        "provenance": {
            "git_commit_P": p_json.get("git_commit"),
            "git_commit_S": s_json.get("git_commit"),
            "git_commit_W2_historical": w2_json.get("git_commit"),
            "batch_plan_P": p_json.get("batch_plan"),
            "batch_plan_S": s_json.get("batch_plan"),
            "checkpoints": {
                "P": p_json.get("primary_checkpoint_sha256"),
                "S": s_json.get("primary_checkpoint_sha256"),
                "W2": w2_json.get("primary_checkpoint_sha256"),
            },
            "curve_sha256": {"P": p_json.get("curve_sha256"), "S": s_json.get("curve_sha256"), "W2": w2_json.get("curve_sha256")},
            "per_task_sha256": {"P": p_json.get("per_task_ap_sha256"), "S": s_json.get("per_task_ap_sha256")},
            "evidence_diagnostics_sha256": {"P": p_json.get("evidence_diagnostics_sha256"), "S": s_json.get("evidence_diagnostics_sha256")},
            "per_task_deltas_csv_sha256": csv_sha,
        },
        "test_isolation": {
            "test_evaluated": False,
            "official_test_used_for_selection": False,
            "note": "the official Test split is never indexed, scored or predicted on in this round",
        },
    }
    C.write_json(out_dir / "paired_comparison.json", comparison)

    parity = training_parity(w2_json, p_json, s_json, curves, diag)
    C.write_json(out_dir / "paired_training_audit.json", parity)

    decision["artifact_sha256"] = {
        "paired_comparison": C.sha256_file(out_dir / "paired_comparison.json"),
        "per_task_deltas_csv": csv_sha,
        "p_epoch40": C.sha256_file(out_dir / "p_epoch40.json"),
        "s_epoch40": C.sha256_file(out_dir / "s_epoch40.json"),
    }
    C.write_json(out_dir / "decision.json", decision)

    rr_jobs = read_json(args.rr_jobs) if args.rr_jobs and Path(args.rr_jobs).exists() else None
    ledger = gpu_ledger(rr_jobs, {"P": p_json, "S": s_json})
    C.write_json(out_dir / "gpu_ledger.json", ledger)

    print(
        json.dumps(
            {
                "ap_w2": ap_w2,
                "ap_p": records["P"]["ap40"],
                "ap_s": records["S"]["ap40"],
                "d_base": decision["d_base"],
                "d_mech": decision["d_mech"],
                "label": decision["label"],
                "high_support": decision["high_support"],
                "parity_all_passed": parity["all_passed"],
                "budget": ledger["budget_status"],
            },
            indent=2,
        )
    )
    return 0


def _last10(curve: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [r for r in curve if r.get("epoch") is not None and int(r["epoch"]) >= 31 and r.get("valid_ap") is not None]
    if len(rows) < 2:
        return {"n": len(rows)}
    x = np.asarray([int(r["epoch"]) for r in rows], dtype=np.float64)
    y = np.asarray([float(r["valid_ap"]) for r in rows], dtype=np.float64)
    slope = float(np.polyfit(x, y, 1)[0])
    e30 = next((float(r["valid_ap"]) for r in curve if r.get("epoch") is not None and int(r["epoch"]) == 30), None)
    return {
        "n": int(len(rows)),
        "slope_per_epoch": slope,
        "range": float(y.max() - y.min()),
        "std": float(y.std(ddof=0)),
        "e30_to_e40": None if e30 is None else float(y[-1] - e30),
    }


if __name__ == "__main__":
    raise SystemExit(main())
