"""Comparison, per-task deltas, resource labels and the GPU ledger.

Reads only committed round artifacts (the two new runs + the read-only
historical W2 record + the pre-training audits).  Never evaluates a model and
never touches an official Test row.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import cs2_common as C

SUPPORT_GE20 = int(C.DECISION_RULES["support_strata"]["ge20"])
SPARSE_MIN = int(C.DECISION_RULES["support_strata"]["ge5_lt20"])
EPOCHS_REPORTED = (10, 20, 25, 30, 40)


# ---------------------------------------------------------------------------
# helpers
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
                    continue
                if k in (
                    "epoch",
                    "global_step",
                    "steps_per_epoch",
                    "n_tasks_scored",
                    "n_tasks_with_observed_labels",
                    "gradient_clip_events",
                    "gpu_peak_allocated_bytes",
                    "gpu_peak_reserved_bytes",
                ):
                    parsed[k] = int(v)
                elif k in ("evaluated", "is_best"):
                    parsed[k] = v == "True"
                else:
                    try:
                        parsed[k] = float(v)
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
            "steps_per_second": None if row is None else _finite(row.get("steps_per_second")),
            "gpu_peak_allocated_bytes": None if row is None else row.get("gpu_peak_allocated_bytes"),
        }
    return out


def _stratum(positives: int) -> str:
    if positives < SPARSE_MIN:
        return "support_lt5"
    if positives < SUPPORT_GE20:
        return "support_5_19"
    return "support_ge20"


def _summary(deltas: np.ndarray, rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    pos_mass = float(deltas[deltas > 0].sum())
    order = np.argsort(-deltas)
    out: dict[str, Any] = {
        "n": int(deltas.size),
        "mean_delta": float(deltas.mean()) if deltas.size else None,
        "median_delta": float(np.median(deltas)) if deltas.size else None,
        "improved": int((deltas > 0).sum()),
        "regressed": int((deltas < 0).sum()),
        "tied": int((deltas == 0).sum()),
        "positive_mass": pos_mass,
        "top5_share_of_positive_mass": float(deltas[order[:5]].sum() / pos_mass) if pos_mass > 0 else None,
        "top10_share_of_positive_mass": float(deltas[order[:10]].sum() / pos_mass) if pos_mass > 0 else None,
        "top5_tasks": [int(rows[i]["task"]) for i in order[:5]],
        "top5_deltas": [float(deltas[i]) for i in order[:5]],
        "bottom5_tasks": [int(rows[i]["task"]) for i in order[-5:]],
        "bottom5_deltas": [float(deltas[i]) for i in order[-5:]],
        "strata": {},
    }
    for s in ("support_ge20", "support_5_19", "support_lt5"):
        sub = np.asarray([r[key] for r in rows if r["stratum"] == s], dtype=np.float64)
        out["strata"][s] = {
            "n": int(sub.size),
            "mean_delta": float(sub.mean()) if sub.size else None,
            "median_delta": float(np.median(sub)) if sub.size else None,
            "improved": int((sub > 0).sum()),
            "regressed": int((sub < 0).sum()),
            "tied": int((sub == 0).sum()),
        }
    support = np.asarray([r["valid_positive"] for r in rows], dtype=np.float64)
    if deltas.size > 2:
        out["spearman_delta_vs_valid_support"] = float(
            np.corrcoef(np.argsort(np.argsort(support)), np.argsort(np.argsort(deltas)))[0, 1]
        )
    return out


# ---------------------------------------------------------------------------
# per-task table
# ---------------------------------------------------------------------------


def per_task_table(
    runs: Mapping[str, Mapping[str, Any]],
    run_dirs: Mapping[str, Path],
    w2_run: Mapping[str, Any],
    baseline_dir: Path,
) -> dict[str, Any]:
    counts = runs["C2"]["task_counts"]
    valid_pos = counts["valid"]["positive"]
    valid_obs = counts["valid"]["observed"]
    train_pos = counts["train"]["positive"]
    w2_counts = w2_run["task_counts"]
    counts_match = (
        w2_counts["valid"]["positive"] == valid_pos
        and w2_counts["valid"]["observed"] == valid_obs
        and w2_counts["train"]["positive"] == train_pos
    )

    def valid_ap(run: Mapping[str, Any], e: int = 40) -> list[Any]:
        return list(run["fixed_epoch_reports"][str(e)]["per_task_ap"])

    ap_w2 = valid_ap(w2_run)
    ap_c2 = valid_ap(runs["C2"])
    ap_f = valid_ap(runs["F"])
    train_w2 = list(read_json(baseline_dir / "train_eval_epoch40.json")["per_task_ap"])
    train_c2 = list(read_json(run_dirs["C2"] / "train_eval_epoch40.json")["per_task_ap"])
    train_f = list(read_json(run_dirs["F"] / "train_eval_epoch40.json")["per_task_ap"])

    rows: list[dict[str, Any]] = []
    for t in range(C.N_TASKS):
        a, b, c = ap_w2[t], ap_c2[t], ap_f[t]
        a = None if a is None else float(a)
        b = None if b is None else float(b)
        c = None if c is None else float(c)
        rows.append(
            {
                "task": t,
                "valid_observed": int(valid_obs[t]),
                "valid_positive": int(valid_pos[t]),
                "train_positive": int(train_pos[t]),
                "stratum": _stratum(int(valid_pos[t])),
                "ap_w2": a,
                "ap_c2": b,
                "ap_f": c,
                "delta_c2_minus_w2": None if (a is None or b is None) else b - a,
                "delta_f_minus_w2": None if (a is None or c is None) else c - a,
                "delta_c2_minus_f": None if (b is None or c is None) else b - c,
                "train_ap_w2": float(train_w2[t]),
                "train_ap_c2": float(train_c2[t]),
                "train_ap_f": float(train_f[t]),
            }
        )

    scorable = [r for r in rows if r["ap_w2"] is not None and r["ap_c2"] is not None and r["ap_f"] is not None]
    summary: dict[str, Any] = {
        "n_tasks": int(C.N_TASKS),
        "n_scorable": len(scorable),
        "unscorable_tasks": [r["task"] for r in rows if r["ap_w2"] is None],
        "valid_positive_counts_match_w2": bool(counts_match),
        "stratum_sizes": {
            s: int(sum(1 for r in scorable if r["stratum"] == s))
            for s in ("support_ge20", "support_5_19", "support_lt5")
        },
    }
    for name, key in (
        ("C2_minus_W2", "delta_c2_minus_w2"),
        ("F_minus_W2", "delta_f_minus_w2"),
        ("C2_minus_F", "delta_c2_minus_f"),
    ):
        deltas = np.asarray([r[key] for r in scorable], dtype=np.float64)
        summary[name] = _summary(deltas, scorable, key)

    tr_c2 = np.asarray([r["train_ap_c2"] - r["train_ap_w2"] for r in rows], dtype=np.float64)
    tr_f = np.asarray([r["train_ap_f"] - r["train_ap_w2"] for r in rows], dtype=np.float64)
    summary["train_breadth"] = {
        "C2_mean_train_ap_delta": float(tr_c2.mean()),
        "C2_improved_train_tasks": int((tr_c2 > 0).sum()),
        "F_mean_train_ap_delta": float(tr_f.mean()),
        "F_improved_train_tasks": int((tr_f > 0).sum()),
    }
    return {"rows": rows, "summary": summary}


def write_per_task_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    fields = [
        "task",
        "valid_observed",
        "valid_positive",
        "train_positive",
        "stratum",
        "ap_w2",
        "ap_c2",
        "ap_f",
        "delta_c2_minus_w2",
        "delta_f_minus_w2",
        "delta_c2_minus_f",
        "train_ap_w2",
        "train_ap_c2",
        "train_ap_f",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fields})
    return C.sha256_file(path)


# ---------------------------------------------------------------------------
# decision
# ---------------------------------------------------------------------------


def decide(
    d_c2: float,
    d_f: float,
    d_cf: float,
    per_task: Mapping[str, Any],
    *,
    c2_ap_curve: Mapping[str, Any],
) -> dict[str, Any]:
    rules = C.DECISION_RULES
    high = per_task["C2_minus_W2"]["strata"]["support_ge20"]
    sparse = per_task["C2_minus_W2"]["strata"]
    mean20 = high["mean_delta"]
    improved20 = int(high["improved"])
    n20 = int(high["n"])
    promising = rules["promising"]

    best_epoch = c2_ap_curve.get("best_diagnostic_epoch")
    best_ap = c2_ap_curve.get("best_diagnostic_ap")
    ap40 = c2_ap_curve.get("ap40")
    early_peak_then_regress = bool(
        best_epoch is not None
        and best_ap is not None
        and ap40 is not None
        and int(best_epoch) <= 25
        and float(best_ap) - float(ap40) >= 0.005
    )
    sparse_mean = sparse["support_5_19"]["mean_delta"]
    tiny_mean = sparse["support_lt5"]["mean_delta"]
    gains_sparse_only = bool(
        mean20 is not None
        and sparse_mean is not None
        and mean20 < 0.003
        and sparse_mean > mean20
    )
    f_matches = bool(d_cf < 0.003)
    f_better = bool(d_cf <= 0.0)

    checks = {
        "D_C2_ge_0.005": bool(d_c2 >= promising["delta_c2_min"]),
        "D_CF_ge_0.003": bool(d_cf >= promising["delta_cf_min"]),
        "mean_delta_ge20_ge_0.003": bool(mean20 is not None and mean20 >= promising["mean_delta_support_ge20_min"]),
        "improved_ge20_ge_60_of_107": bool(improved20 >= promising["improved_support_ge20_min"]),
        "support_ge20_total_is_107": bool(n20 == promising["support_ge20_total"]),
    }
    all_promising = all(checks[k] for k in checks if k != "support_ge20_total_is_107") and not early_peak_then_regress

    if all_promising:
        label = promising["label"]
        rationale = "all four pre-registered conditions hold"
    elif d_c2 <= rules["regression"]["delta_c2_max"]:
        label = rules["regression"]["label"]
        rationale = "C2 epoch-40 AP is at least 0.002 below W2"
    elif d_c2 < rules["no_clear"]["delta_c2_max_exclusive"]:
        broad_stable = bool(
            mean20 is not None
            and mean20 >= promising["mean_delta_support_ge20_min"]
            and improved20 >= promising["improved_support_ge20_min"]
            and not gains_sparse_only
            and not early_peak_then_regress
        )
        if broad_stable:
            label = rules["weak"]["label"]
            rationale = "D_C2 < 0.002 but the high-support pattern is broad and stable"
        else:
            label = rules["no_clear"]["label"]
            rationale = "D_C2 < 0.002 without a broad stable high-support gain"
    elif d_c2 < rules["weak"]["delta_c2_max"]:
        label = rules["weak"]["label"]
        rationale = "0.002 <= D_C2 < 0.005"
    else:
        label = rules["capacity_or_mixed"]["label"]
        rationale = "D_C2 >= 0.005 but not all Conditional-S2 conditions hold"

    mixed_flags = {
        "c2_beats_w2": bool(d_c2 > 0.0),
        "f_matches_or_beats_c2": bool(f_matches),
        "f_actually_beats_c2": bool(f_better),
        "mean_high_support_gain_below_0.003": bool(mean20 is not None and mean20 < 0.003),
        "improved_high_support_below_60": bool(improved20 < 60),
        "gains_concentrated_in_sparse_assays": gains_sparse_only,
        "early_peak_then_regress_by_epoch40": early_peak_then_regress,
        "weak_condition_d_cf_positive_below_threshold": bool(0.0 < d_cf < promising["delta_cf_min"]),
    }
    return {
        "label": label,
        "rationale": rationale,
        "delta_c2": float(d_c2),
        "delta_f": float(d_f),
        "delta_cf": float(d_cf),
        "high_support": {
            "n": n20,
            "mean_delta_c2_minus_w2": mean20,
            "median_delta_c2_minus_w2": high["median_delta"],
            "improved": improved20,
            "regressed": int(high["regressed"]),
        },
        "checks": checks,
        "mixed_flags": mixed_flags,
        "thresholds": {
            "promising": {
                "D_C2_min": promising["delta_c2_min"],
                "D_CF_min": promising["delta_cf_min"],
                "mean_delta_support_ge20_min": promising["mean_delta_support_ge20_min"],
                "improved_support_ge20_min": promising["improved_support_ge20_min"],
            },
            "weak_band": [rules["weak"]["delta_c2_min"], rules["weak"]["delta_c2_max"]],
            "no_clear_below": rules["no_clear"]["delta_c2_max_exclusive"],
            "regression_at_or_below": rules["regression"]["delta_c2_max"],
        },
        "caveat": "one seed; resource-allocation label, not a significance test",
    }


# ---------------------------------------------------------------------------
# primary records / parity / ledger
# ---------------------------------------------------------------------------


def epoch40_record(
    key: str,
    run_dir: Path,
    run: Mapping[str, Any],
    curve: Sequence[Mapping[str, Any]],
    *,
    baseline: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row = curve_at(curve, 40) or {}
    fixed = run["fixed_epoch_reports"]["40"]
    train_eval = read_json(run_dir / "train_eval_epoch40.json")
    record: dict[str, Any] = {
        "schema": "lcrc-pcba-conditional-s2-epoch40-v1",
        "round_id": C.ROUND_ID,
        "model": key,
        "model_name": C.MODELS[key]["name"],
        "role": C.MODELS[key]["role"],
        "base_model": C.BASE_MODEL_KEY,
        "seed": int(run["seed"]),
        "parameters": int(run["parameters"]),
        "base_parameters": int(run["base_parameters"]),
        "new_parameters": int(run["new_parameters"]),
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
            "logits_sha256": train_eval["logits_sha256"],
        },
        "train_valid_ap_gap": float(train_eval["ap"]) - float(fixed["ap"]),
        "epochs": epoch_metrics(run, curve, EPOCHS_REPORTED),
        "epoch25_to_epoch40": None
        if curve_at(curve, 25) is None
        else float(fixed["ap"]) - float(curve_at(curve, 25)["valid_ap"]),
        "epoch30_to_epoch40": None
        if curve_at(curve, 30) is None
        else float(fixed["ap"]) - float(curve_at(curve, 30)["valid_ap"]),
        "best_diagnostic_validation": {
            "epoch": int(run["best_epoch"]),
            "ap": float(run["best_valid_ap"]),
            "used_for_primary_comparison": False,
        },
        "diagnostics": {
            "initial": (run.get("diagnostics") or {}).get("initial"),
            "epoch40": ((run.get("diagnostics") or {}).get("epochs") or {}).get("40"),
            "epoch10": ((run.get("diagnostics") or {}).get("epochs") or {}).get("10"),
            "epoch25": ((run.get("diagnostics") or {}).get("epochs") or {}).get("25"),
            "epoch30": ((run.get("diagnostics") or {}).get("epochs") or {}).get("30"),
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
            "train_seconds": float(run.get("train_seconds") or 0.0),
            "valid_seconds": float(run.get("valid_seconds") or 0.0),
            "diag_seconds": float(run.get("diag_seconds") or 0.0),
            "steps_per_second_mean": float(run.get("steps_per_second_mean") or 0.0),
            "step_ms_p50_epoch40": row.get("step_ms_p50"),
            "epoch_train_seconds_epoch40": row.get("epoch_train_seconds"),
            "epoch_valid_seconds_epoch40": row.get("epoch_valid_seconds"),
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
            "curve_sha256": run.get("curve_sha256"),
            "per_task_ap_sha256": run.get("per_task_ap_sha256"),
            "test_evaluated": bool(run.get("test_evaluated", False)),
        },
        "curve": [
            {
                "epoch": int(r["epoch"]),
                "valid_ap": r.get("valid_ap"),
                "valid_bce": r.get("valid_bce"),
                "train_bce": r.get("train_bce"),
                "gate_delta_rms": r.get("gate_delta_rms"),
                "ffn_branch_rms": r.get("ffn_branch_rms"),
            }
            for r in curve
        ],
    }
    if baseline is not None:
        record["baseline"] = dict(baseline)
    return record


def training_parity(
    runs: Mapping[str, Mapping[str, Any]],
    curves: Mapping[str, Sequence[Mapping[str, Any]]],
    w2_run: Mapping[str, Any],
    init_audit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    recipe_keys = (
        "optimizer",
        "learning_rate",
        "warmup_start_lr",
        "end_lr",
        "weight_decay",
        "adam_betas",
        "adam_eps",
        "batch_size",
        "epochs",
        "gradient_clip",
        "dropout",
        "precision",
        "early_stopping",
        "loss",
    )

    def recipe_subset(run: Mapping[str, Any]) -> dict[str, Any]:
        r = run.get("recipe") or {}
        return {k: r.get(k) for k in recipe_keys}

    c2, f = runs["C2"], runs["F"]
    c2r, fr = recipe_subset(c2), recipe_subset(f)
    w2r = recipe_subset(w2_run)
    plan_equal = bool(
        c2["batch_plan"]["global_plan_sha256"] == f["batch_plan"]["global_plan_sha256"]
        and c2["batch_plan"]["prefix_sha256"] == f["batch_plan"]["prefix_sha256"]
        and c2["batch_plan"]["global_plan_sha256"] == C.BASE_GLOBAL_PLAN_SHA256
        and c2["batch_plan"]["prefix_sha256"] == C.BASE_CONSUMED40_PREFIX_SHA256
    )
    recipe_equal = bool(c2r == fr == w2r)
    base_fp_equal = bool(
        c2["model_audit"]["base_state_sha256"] == f["model_audit"]["base_state_sha256"]
        and c2["model_audit"]["base_trunk_init_fingerprint"] == C.BASE_TRUNK_INIT_FINGERPRINT
        and f["model_audit"]["base_trunk_init_fingerprint"] == C.BASE_TRUNK_INIT_FINGERPRINT
    )
    epoch0_equal = bool(
        c2.get("epoch0_diagnostic") is not None
        and f.get("epoch0_diagnostic") is not None
        and abs(float(c2["epoch0_diagnostic"]["ap"]) - float(f["epoch0_diagnostic"]["ap"])) <= 1e-4
    )
    epoch0_vs_w2 = {
        "w2_recorded_ap": None if w2_run.get("epoch0_diagnostic") is None else float(w2_run["epoch0_diagnostic"]["ap"]),
        "c2_epoch0_ap": None if c2.get("epoch0_diagnostic") is None else float(c2["epoch0_diagnostic"]["ap"]),
        "f_epoch0_ap": None if f.get("epoch0_diagnostic") is None else float(f["epoch0_diagnostic"]["ap"]),
    }
    epoch0_vs_w2["c2_abs_delta_vs_w2"] = (
        None
        if epoch0_vs_w2["w2_recorded_ap"] is None or epoch0_vs_w2["c2_epoch0_ap"] is None
        else abs(epoch0_vs_w2["c2_epoch0_ap"] - epoch0_vs_w2["w2_recorded_ap"])
    )
    epoch0_vs_w2["f_abs_delta_vs_w2"] = (
        None
        if epoch0_vs_w2["w2_recorded_ap"] is None or epoch0_vs_w2["f_epoch0_ap"] is None
        else abs(epoch0_vs_w2["f_epoch0_ap"] - epoch0_vs_w2["w2_recorded_ap"])
    )
    rng_epoch1 = {
        "c2": (curve_at(curves["C2"], 1) or {}).get("rng_cpu_sha256"),
        "f": (curve_at(curves["F"], 1) or {}).get("rng_cpu_sha256"),
    }
    seed_states_equal = bool(c2.get("seed_state") == f.get("seed_state"))
    checks = {
        "same_40_epoch_batch_plan": plan_equal,
        "same_frozen_plan_as_w2": bool(
            c2["batch_plan"]["global_plan_sha256"] == C.BASE_GLOBAL_PLAN_SHA256
            and f["batch_plan"]["global_plan_sha256"] == C.BASE_GLOBAL_PLAN_SHA256
        ),
        "same_training_recipe_as_w2": recipe_equal,
        "same_original_w2_initial_subset": base_fp_equal,
        "same_global_training_seed_state": seed_states_equal,
        "epoch0_ap_matches_between_arms": epoch0_equal,
        "epoch0_ap_matches_w2_within_1e_4": bool(
            epoch0_vs_w2["c2_abs_delta_vs_w2"] is not None
            and epoch0_vs_w2["c2_abs_delta_vs_w2"] <= 1e-4
            and epoch0_vs_w2["f_abs_delta_vs_w2"] is not None
            and epoch0_vs_w2["f_abs_delta_vs_w2"] <= 1e-4
        ),
        "same_end_epoch1_torch_rng_state": bool(rng_epoch1["c2"] == rng_epoch1["f"]),
        "all_eval_epochs_completed": bool(
            c2.get("all_eval_epochs_completed") and f.get("all_eval_epochs_completed")
        ),
    }
    if init_audit is not None:
        audit_checks = init_audit.get("checks") or {}
        checks["independent_init_audit_passed"] = bool(init_audit.get("verdict") == "OK")
        checks["init_audit_cpu_logits_bitwise"] = bool(
            audit_checks.get("C2_cpu_logits_bitwise_equal_to_w2")
            and audit_checks.get("F_cpu_logits_bitwise_equal_to_w2")
        )
        checks["init_audit_parameter_counts"] = bool(
            audit_checks.get("C2_parameters_exact") and audit_checks.get("F_parameters_exact")
        )
    return {
        "schema": "lcrc-pcba-conditional-s2-training-parity-v1",
        "round_id": C.ROUND_ID,
        "recipe": {
            "c2": c2r,
            "f": fr,
            "w2": w2r,
            "identical": recipe_equal,
        },
        "batch_plan": {
            "c2": c2["batch_plan"],
            "f": f["batch_plan"],
            "frozen_w2_global_plan_sha256": C.BASE_GLOBAL_PLAN_SHA256,
            "frozen_w2_consumed40_prefix_sha256": C.BASE_CONSUMED40_PREFIX_SHA256,
            "identical": plan_equal,
        },
        "initial_subset": {
            "c2_base_state_sha256": c2["model_audit"]["base_state_sha256"],
            "f_base_state_sha256": f["model_audit"]["base_state_sha256"],
            "c2_trunk_init_fingerprint": c2["model_audit"]["base_trunk_init_fingerprint"],
            "f_trunk_init_fingerprint": f["model_audit"]["base_trunk_init_fingerprint"],
            "historical_trunk_init_fingerprint": C.BASE_TRUNK_INIT_FINGERPRINT,
            "equal": base_fp_equal,
        },
        "epoch0": {
            **epoch0_vs_w2,
            "c2_valid_logits_sha256": (c2.get("epoch0_diagnostic") or {}).get("valid_logits_sha256"),
            "f_valid_logits_sha256": (f.get("epoch0_diagnostic") or {}).get("valid_logits_sha256"),
            "w2_recorded_valid_logits_sha256": (w2_run.get("epoch0_diagnostic") or {}).get("valid_logits_sha256"),
        },
        "rng": {
            "epoch1_cpu_sha256": rng_epoch1,
            "seed_states_equal": seed_states_equal,
            "epoch40_c2": (curve_at(curves["C2"], 40) or {}).get("rng_cpu_sha256"),
            "epoch40_f": (curve_at(curves["F"], 40) or {}).get("rng_cpu_sha256"),
        },
        "checks": checks,
        "all_checks_pass": all(bool(v) for v in checks.values()),
        "caveat": "paired initialisation/data order does not imply identical trajectories after step 1",
    }


def gpu_ledger(rr_jobs: Mapping[str, Any] | None, runs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    jobs: list[dict[str, Any]] = []
    total = 0.0
    if rr_jobs is not None:
        for job in rr_jobs.get("jobs", []):
            exp = str(job.get("experiment") or "")
            if not exp.startswith("cs2-"):
                continue
            started = job.get("started_at")
            completed = job.get("completed_at")
            seconds = 0.0
            if started and completed:
                seconds = max(
                    0.0,
                    (datetime.fromisoformat(completed) - datetime.fromisoformat(started)).total_seconds(),
                )
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
        "schema": "lcrc-pcba-conditional-s2-gpu-ledger-v1",
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
    parser.add_argument("--c2-dir", default=str(C.RUNS / "C2"))
    parser.add_argument("--f-dir", default=str(C.RUNS / "F"))
    parser.add_argument("--baseline-dir", default=str(C.W2_RUN_DIR))
    parser.add_argument("--out-dir", default=str(C.ANALYSIS))
    parser.add_argument("--rr-jobs", default="")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = {"C2": Path(args.c2_dir), "F": Path(args.f_dir)}
    baseline_dir = Path(args.baseline_dir)
    runs = {k: read_json(d / "run.json") for k, d in run_dirs.items()}
    w2_run = read_json(baseline_dir / "run.json")
    curves = {k: load_curve(d) for k, d in run_dirs.items()}
    w2_curve = load_curve(baseline_dir)
    rr_jobs = read_json(args.rr_jobs) if args.rr_jobs else None
    init_audit_path = out_dir / "initialization_audit.json"
    init_audit = read_json(init_audit_path) if init_audit_path.exists() else None
    arch_audit_path = out_dir / "architecture_audit.json"
    arch_audit = read_json(arch_audit_path) if arch_audit_path.exists() else None

    baseline = {
        "model": C.BASE_MODEL_KEY,
        "parameters": int(C.BASE_PARAMETERS),
        "ap40": float(w2_run["fixed_epoch_reports"]["40"]["ap"]),
        "train_ap40": float(read_json(baseline_dir / "train_eval_epoch40.json")["ap"]),
        "train_bce40": float(read_json(baseline_dir / "train_eval_epoch40.json")["bce"]),
        "valid_bce40": float(w2_run["fixed_epoch_reports"]["40"]["valid_bce"]),
        "best_epoch": int(w2_run["best_epoch"]),
        "best_ap": float(w2_run["best_valid_ap"]),
        "epochs": epoch_metrics(w2_run, w2_curve, EPOCHS_REPORTED),
        "source": "tracks/structural-sufficiency/lcrc_pcba_backbone_scaling_v1/runs/W2 (read-only)",
        "checkpoint_sha256": w2_run.get("primary_checkpoint_sha256"),
    }

    table = per_task_table(runs, run_dirs, w2_run, baseline_dir)
    summary = table["summary"]
    csv_sha = write_per_task_csv(out_dir / "per_task_deltas.csv", table["rows"])

    c2_record = epoch40_record("C2", run_dirs["C2"], runs["C2"], curves["C2"], baseline=baseline)
    f_record = epoch40_record("F", run_dirs["F"], runs["F"], curves["F"], baseline=baseline)
    ap_w2 = float(baseline["ap40"])
    ap_c2 = float(c2_record["validation"]["ap"])
    ap_f = float(f_record["validation"]["ap"])
    d_c2 = ap_c2 - ap_w2
    d_f = ap_f - ap_w2
    d_cf = ap_c2 - ap_f
    c2_curve_summary = {
        "ap40": ap_c2,
        "best_diagnostic_epoch": c2_record["best_diagnostic_validation"]["epoch"],
        "best_diagnostic_ap": c2_record["best_diagnostic_validation"]["ap"],
    }
    decision = decide(d_c2, d_f, d_cf, summary, c2_ap_curve=c2_curve_summary)

    paired = {
        "schema": "lcrc-pcba-conditional-s2-paired-comparison-v1",
        "round_id": C.ROUND_ID,
        "baseline": baseline,
        "new_arms": {
            "C2": {
                "parameters": int(c2_record["parameters"]),
                "new_parameters": int(c2_record["new_parameters"]),
                "ap40": ap_c2,
            },
            "F": {
                "parameters": int(f_record["parameters"]),
                "new_parameters": int(f_record["new_parameters"]),
                "ap40": ap_f,
            },
        },
        "deltas": {
            "D_C2": d_c2,
            "D_F": d_f,
            "D_CF": d_cf,
            "C2_vs_W2_relative_pct": 100.0 * d_c2 / ap_w2,
            "F_vs_W2_relative_pct": 100.0 * d_f / ap_w2,
        },
        "per_task": summary,
        "per_task_csv": C.track_rel(out_dir / "per_task_deltas.csv"),
        "per_task_csv_sha256": csv_sha,
        "epochs": {
            "W2": {"ap10": baseline["epochs"]["10"]["valid_ap"], "ap20": baseline["epochs"]["20"]["valid_ap"],
                   "ap25": baseline["epochs"]["25"]["valid_ap"], "ap30": baseline["epochs"]["30"]["valid_ap"],
                   "ap40": baseline["epochs"]["40"]["valid_ap"]},
            "C2": {f"ap{e}": c2_record["epochs"][str(e)]["valid_ap"] for e in EPOCHS_REPORTED},
            "F": {f"ap{e}": f_record["epochs"][str(e)]["valid_ap"] for e in EPOCHS_REPORTED},
        },
        "budget": {
            "C2_gpu_hours": float(runs["C2"]["gpu_hours"]),
            "F_gpu_hours": float(runs["F"]["gpu_hours"]),
            "total_formal_gpu_hours": float(runs["C2"]["gpu_hours"] + runs["F"]["gpu_hours"]),
            "target_hours": float(C.BUDGET["gpu_target_hours"]),
            "hard_cap_hours": float(C.BUDGET["gpu_hard_hours"]),
        },
        "audits": {
            "initialization_audit_verdict": None if init_audit is None else init_audit.get("verdict"),
            "architecture_audit_verdict": None if arch_audit is None else arch_audit.get("verdict"),
        },
        "decision_label": decision["label"],
        "test_evaluated": False,
    }
    parity = training_parity(runs, curves, w2_run, init_audit)
    ledger = gpu_ledger(rr_jobs, runs)
    decision["training_parity_all_checks_pass"] = parity["all_checks_pass"]
    decision["architecture_audit_verdict"] = None if arch_audit is None else arch_audit.get("verdict")
    decision["initialization_audit_verdict"] = None if init_audit is None else init_audit.get("verdict")

    C.write_json(out_dir / "c2_epoch40.json", c2_record)
    C.write_json(out_dir / "ffn_epoch40.json", f_record)
    C.write_json(out_dir / "paired_comparison.json", paired)
    C.write_json(out_dir / "training_parity.json", parity)
    C.write_json(out_dir / "gpu_ledger.json", ledger)
    C.write_json(out_dir / "decision.json", decision)
    C.log(
        f"AP W2 {ap_w2:.10f} -> C2 {ap_c2:.10f} ({d_c2:+.10f}), F {ap_f:.10f} ({d_f:+.10f}); "
        f"D_CF {d_cf:+.10f}; label {decision['label']}"
    )
    return 0 if decision["label"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
