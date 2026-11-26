"""FSF follow-up v1 analysis: 1-25 curves, FIXED1 comparison, decision.

Reads only local artefacts (no data cache, no model execution): the v0 epoch-12
prefix, the new 13-25 segment, and the real historical FIXED1 seed-0 curve and
per-task Valid AP file.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fsf_common as F

BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 0
IMPROVE_TOL = 1e-6
SIGNAL_AP = 0.003
HIGH_SUPPORT_MIN = 20
DROP_TOP_K = 3


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def _f(x: Any) -> float | None:
    if x in (None, ""):
        return None
    return float(x)


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            out = {}
            for k in fields:
                v = row.get(k)
                if isinstance(v, (np.floating, np.integer, np.bool_)):
                    v = v.item()
                out[k] = "" if v is None else v
            writer.writerow(out)
    return F.sha256_file(path)


def rssc_curve() -> list[dict[str, Any]]:
    path = F.RUN_DIR / "train_curve.csv"
    if path.is_file():
        rows = _read_csv(path)
        return [{**r, "source": r.get("source") or "RSSC_followup_v1"} for r in rows]
    return F.source_curve()


def fixed1_curve(horizon: int = 25) -> list[dict[str, Any]]:
    rows = _read_csv(F.HIST_CURVE)
    out = []
    for r in rows:
        if int(r["epoch"]) > int(horizon):
            continue
        out.append({
            "arm": "FIXED1_hist", "source": "historical_seed0_run",
            "epoch": int(r["epoch"]), "global_step": int(r.get("global_step") or 0),
            "steps_per_epoch": int(r.get("steps_per_epoch") or 0),
            "learning_rate": _f(r.get("learning_rate")),
            "train_ap": _f(r.get("train_ap")), "train_bce": _f(r.get("train_bce")),
            "train_evaluated": True, "valid_ap": _f(r.get("valid_ap")),
            "valid_bce": _f(r.get("valid_bce")),
            "epoch_seconds": _f(r.get("epoch_seconds")), "is_best": "",
            "best_valid_ap_so_far": "",
        })
    return out


def per_task_from(path: Path, epoch: int) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        return out
    for r in _read_csv(path):
        if r["split"] != "valid" or int(r["epoch"]) != int(epoch):
            continue
        out[int(r["task"])] = {
            "ap": (None if r["ap"] == "" else float(r["ap"])),
            "observed": int(r["observed"]), "positive": int(r["positive"]),
            "negative": int(r["negative"]),
        }
    return out


def per_task(epoch: int) -> dict[int, dict[str, Any]]:
    """Union of the v0 prefix file and this round's file."""
    out = per_task_from(F.SOURCE_PER_TASK, epoch)
    out.update(per_task_from(F.RUN_DIR / "per_task_valid.csv", epoch))
    return out


def fixed1_per_task(epoch: int = 25) -> dict[int, dict[str, Any]]:
    return per_task_from(F.HIST_PER_TASK, epoch)


def curve_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if _f(r.get("valid_ap")) is not None]
    if not valid:
        return {}
    best = max(valid, key=lambda r: (float(r["valid_ap"]), -int(r["epoch"])))
    last3 = valid[-3:]
    last5 = valid[-5:]
    return {
        "best_epoch": int(best["epoch"]), "best_valid_ap": float(best["valid_ap"]),
        "best_valid_bce": _f(best.get("valid_bce")),
        "endpoint_epoch": int(valid[-1]["epoch"]),
        "endpoint_valid_ap": float(valid[-1]["valid_ap"]),
        "endpoint_valid_bce": _f(valid[-1].get("valid_bce")),
        "last3_mean_valid_ap": float(np.mean([float(r["valid_ap"]) for r in last3])),
        "last5_mean_valid_ap": float(np.mean([float(r["valid_ap"]) for r in last5])),
        "last5_epochs": [int(r["epoch"]) for r in last5],
        "last5_rows": [{k: r.get(k) for k in ("epoch", "valid_ap", "valid_bce", "train_ap", "train_bce")}
                       for r in last5],
        "best_train_ap_in_curve": _f(best.get("train_ap")),
        "best_train_bce_in_curve": _f(best.get("train_bce")),
    }


def paired(
    name: str,
    a: Mapping[int, float | None],
    b: Mapping[int, float | None],
    support: Mapping[int, Mapping[str, Any]],
    *,
    meta: Mapping[str, Any],
) -> dict[str, Any]:
    tasks = sorted(set(a) & set(b))
    tasks = [t for t in tasks if a[t] is not None and b[t] is not None]
    deltas = np.asarray([float(a[t]) - float(b[t]) for t in tasks], dtype=np.float64)
    high = np.asarray([t for t in tasks if support[t]["positive"] >= HIGH_SUPPORT_MIN
                       and support[t]["negative"] >= HIGH_SUPPORT_MIN], dtype=np.int64)
    high_deltas = np.asarray([float(a[t]) - float(b[t]) for t in high], dtype=np.float64) if high.size else np.zeros(0)
    order = np.argsort(-np.abs(deltas))
    keep = np.ones(len(deltas), dtype=bool)
    keep[order[: min(DROP_TOP_K, len(deltas))]] = False
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(BOOTSTRAP_N, dtype=np.float64)
    n = len(deltas)
    for i in range(BOOTSTRAP_N):
        idx = rng.integers(0, n, n)
        means[i] = float(deltas[idx].mean())
    return {
        "name": name, "meta": dict(meta), "n_tasks_paired": int(n),
        "mean_delta_ap": float(deltas.mean()) if n else None,
        "median_delta_ap": float(np.median(deltas)) if n else None,
        "std_delta_ap": float(deltas.std(ddof=1)) if n > 1 else None,
        "min_delta_ap": float(deltas.min()) if n else None,
        "max_delta_ap": float(deltas.max()) if n else None,
        "tasks_improved": int((deltas > IMPROVE_TOL).sum()),
        "tasks_degraded": int((deltas < -IMPROVE_TOL).sum()),
        "tasks_tied": int((np.abs(deltas) <= IMPROVE_TOL).sum()),
        "tasks_improved_gt_0.005": int((deltas > 0.005).sum()),
        "tasks_degraded_lt_-0.005": int((deltas < -0.005).sum()),
        "task_bootstrap_2000": {
            "mean_delta_ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "fraction_resamples_positive": float((means > 0).mean()),
            "note": "task-level resampling of the fixed trained results; not a seed-variance estimate",
        },
        "high_support": {
            "min_positive_and_negative": HIGH_SUPPORT_MIN,
            "n_tasks": int(high.size),
            "mean_delta_ap": float(high_deltas.mean()) if high.size else None,
            "tasks_improved": int((high_deltas > IMPROVE_TOL).sum()) if high.size else 0,
            "tasks_degraded": int((high_deltas < -IMPROVE_TOL).sum()) if high.size else 0,
        },
        "drop_top3_abs_delta": {
            "n_tasks": int(keep.sum()),
            "mean_delta_ap": float(deltas[keep].mean()) if keep.any() else None,
            "dropped_tasks": [int(tasks[i]) for i in order[: min(DROP_TOP_K, len(deltas))]],
        },
        "per_task": [{
            "task": int(t), "a_ap": float(a[t]), "b_ap": float(b[t]),
            "delta_ap": float(a[t]) - float(b[t]),
            "valid_positive": int(support[t]["positive"]),
            "valid_negative": int(support[t]["negative"]),
        } for t in tasks],
    }


def main() -> int:
    F.ANALYSIS.mkdir(parents=True, exist_ok=True)
    rssc = rssc_curve()
    fixed1 = fixed1_curve(25)
    if not rssc:
        raise RuntimeError("no RSSC curve available")
    rssc_endpoint_epoch = int(max(int(r["epoch"]) for r in rssc))
    rssc = [r for r in rssc if int(r["epoch"]) <= rssc_endpoint_epoch]
    fixed1 = [r for r in fixed1 if int(r["epoch"]) <= rssc_endpoint_epoch]

    # run/selection metadata ------------------------------------------------
    run_path = F.RUN_DIR / "run.json"
    run = F.read_json(run_path) if run_path.is_file() else {}
    status = run.get("status", "MISSING")
    complete = bool(status == "COMPLETED" and int(run.get("epochs_run") or 0) >= rssc_endpoint_epoch
                    and rssc_endpoint_epoch == F.TARGET_HORIZON)
    cs_rssc = curve_summary(rssc)
    cs_fixed1 = curve_summary(fixed1)
    best_epoch = int(cs_rssc["best_epoch"])
    # official record from run.json if available
    if run.get("best_epoch") is not None:
        best_epoch = int(run["best_epoch"])

    # curves CSV -------------------------------------------------------------
    curve_fields = ["arm", "source", "epoch", "global_step", "learning_rate", "train_ap",
                    "train_bce", "train_evaluated", "valid_ap", "valid_bce", "epoch_seconds",
                    "is_best", "best_valid_ap_so_far"]
    curve_rows = []
    for r in rssc:
        curve_rows.append({"arm": "RSSC" if r.get("arm") in (None, "") else r.get("arm"),
                           **{k: r.get(k) for k in curve_fields if k != "arm"}})
    for r in fixed1:
        curve_rows.append({**{k: r.get(k) for k in curve_fields}})
    curve_sha = _write_csv(F.ANALYSIS / "epoch_curves.csv", curve_fields, curve_rows)

    # best checkpoint evaluation -------------------------------------------
    best_eval_path = F.RUN_DIR / "best_checkpoint_eval.json"
    best_eval = F.read_json(best_eval_path) if best_eval_path.is_file() else {}
    best_train_ap = _f((best_eval.get("train") or {}).get("ap"))
    best_train_bce = _f((best_eval.get("train") or {}).get("bce"))
    best_valid_bce = _f((best_eval.get("valid") or {}).get("bce"))
    if best_valid_bce is None:
        row = next((r for r in rssc if int(r["epoch"]) == best_epoch), None)
        best_valid_bce = _f((row or {}).get("valid_bce"))
    if not best_eval and best_epoch <= 12:
        # the v0 epoch-12 best was already evaluated in the source round
        src_eval = F.read_json(F.SRC_RUNS / "best_checkpoint_eval.json")
        best_train_ap = _f(src_eval["train"]["ap"])
        best_train_bce = _f(src_eval["train"]["bce"])
        best_valid_bce = _f(src_eval["valid"]["bce"])

    # FIXED1 best checkpoint eval (historical, seeds 0, epoch 25) ------------
    hist_run = F.read_json(F.HIST_RUN_JSON)
    h_eval = hist_run["best_checkpoint_eval"]
    fixed1_best_train_ap = float(h_eval["train"]["ap"])
    fixed1_best_train_bce = float(h_eval["train"]["bce"])
    fixed1_best_valid_bce = float(h_eval["valid"]["bce"])
    fixed1_best_valid_ap = float(hist_run["best_valid_ap"])
    fixed1_best_epoch = int(hist_run["best_epoch"])

    main_fields = ["arm", "status", "horizon", "epochs_run", "best_epoch", "best_valid_ap",
                   "best_valid_bce", "endpoint_valid_ap", "endpoint_valid_bce",
                   "last3_mean_valid_ap", "last5_mean_valid_ap", "best_train_ap", "best_train_bce",
                   "parameters", "rank", "gpu_hours", "test_scored"]
    main_rows = [
        {"arm": "RSSC", "status": status, "horizon": F.TARGET_HORIZON,
         "epochs_run": rssc_endpoint_epoch, "best_epoch": best_epoch,
         "best_valid_ap": cs_rssc["best_valid_ap"], "best_valid_bce": best_valid_bce,
         "endpoint_valid_ap": cs_rssc["endpoint_valid_ap"],
         "endpoint_valid_bce": cs_rssc["endpoint_valid_bce"],
         "last3_mean_valid_ap": cs_rssc["last3_mean_valid_ap"],
         "last5_mean_valid_ap": cs_rssc["last5_mean_valid_ap"],
         "best_train_ap": best_train_ap, "best_train_bce": best_train_bce,
         "parameters": run.get("parameters", 1153044), "rank": 32,
         "gpu_hours": run.get("gpu_hours"), "test_scored": False},
        {"arm": "FIXED1_hist", "status": "COMPLETED", "horizon": 25, "epochs_run": 25,
         "best_epoch": fixed1_best_epoch, "best_valid_ap": fixed1_best_valid_ap,
         "best_valid_bce": fixed1_best_valid_bce,
         "endpoint_valid_ap": cs_fixed1["endpoint_valid_ap"],
         "endpoint_valid_bce": cs_fixed1["endpoint_valid_bce"],
         "last3_mean_valid_ap": cs_fixed1["last3_mean_valid_ap"],
         "last5_mean_valid_ap": cs_fixed1["last5_mean_valid_ap"],
         "best_train_ap": fixed1_best_train_ap, "best_train_bce": fixed1_best_train_bce,
         "parameters": 1128468, "rank": 0, "gpu_hours": None, "test_scored": False},
    ]
    main_sha = _write_csv(F.ANALYSIS / "main_table.csv", main_fields, main_rows)

    # per-task comparisons ---------------------------------------------------
    support_hist = fixed1_per_task(25)
    support = {t: {"positive": v["positive"], "negative": v["negative"]}
               for t, v in support_hist.items()}
    r_best = per_task(best_epoch)
    r_end = per_task(rssc_endpoint_epoch)
    f_best = fixed1_per_task(fixed1_best_epoch)
    f_end = fixed1_per_task(25)

    def ap_map(rows: Mapping[int, Mapping[str, Any]]) -> dict[int, float | None]:
        return {t: v["ap"] for t, v in rows.items()}

    comparisons: dict[str, Any] = {}
    comparisons["RSSC_minus_FIXED1_best"] = paired(
        "RSSC (best over 25 epochs) - FIXED1 (25-epoch best, Valid)",
        ap_map(r_best), ap_map(f_best), support,
        meta={"a_epoch": best_epoch, "b_epoch": fixed1_best_epoch,
              "a_valid_ap": cs_rssc["best_valid_ap"], "b_valid_ap": fixed1_best_valid_ap,
              "delta_ap": cs_rssc["best_valid_ap"] - fixed1_best_valid_ap})
    comparisons["RSSC_minus_FIXED1_endpoint"] = paired(
        "RSSC (epoch 25) - FIXED1 (epoch 25, Valid)",
        ap_map(r_end), ap_map(f_end), support,
        meta={"a_epoch": rssc_endpoint_epoch, "b_epoch": 25,
              "a_valid_ap": cs_rssc["endpoint_valid_ap"],
              "b_valid_ap": cs_fixed1["endpoint_valid_ap"],
              "delta_ap": cs_rssc["endpoint_valid_ap"] - cs_fixed1["endpoint_valid_ap"]})
    # prefix continuity check: v0 epoch-12 vs FIXED1 epoch-12
    r12 = per_task(12)
    f12 = fixed1_per_task(12)
    r12_ap = next((r for r in rssc if int(r["epoch"]) == 12), None)
    f12_ap = next((r for r in fixed1 if int(r["epoch"]) == 12), None)
    if r12 and f12 and r12_ap and f12_ap:
        comparisons["RSSC_epoch12_minus_FIXED1_epoch12"] = paired(
            "RSSC (epoch 12 prefix) - FIXED1 (epoch 12 prefix, Valid)",
            ap_map(r12), ap_map(f12), support,
            meta={"a_epoch": 12, "b_epoch": 12,
                  "a_valid_ap": _f(r12_ap["valid_ap"]), "b_valid_ap": _f(f12_ap["valid_ap"]),
                  "delta_ap": float(r12_ap["valid_ap"]) - float(f12_ap["valid_ap"])})

    # task delta table for the primary best-checkpoint comparison
    task_fields = ["task", "rssc_best_epoch", "fixed1_best_epoch", "rssc_ap", "fixed1_ap",
                   "delta_ap", "valid_positive", "valid_negative", "high_support"]
    task_rows = []
    for rec in comparisons["RSSC_minus_FIXED1_best"]["per_task"]:
        task_rows.append({**rec, "rssc_best_epoch": best_epoch,
                          "fixed1_best_epoch": fixed1_best_epoch,
                          "high_support": bool(rec["valid_positive"] >= HIGH_SUPPORT_MIN
                                               and rec["valid_negative"] >= HIGH_SUPPORT_MIN)})
    task_sha = _write_csv(F.ANALYSIS / "task_deltas.csv", task_fields, task_rows)

    # mechanism light --------------------------------------------------------
    mech_path = F.RUN_DIR / "mechanism_light.json"
    mech = F.read_json(mech_path) if mech_path.is_file() else {}
    hist = mech.get("history", [])
    baseline = mech.get("epoch12_baseline", {})
    mech_rows = []
    for rec in hist:
        light = rec.get("light") or {}
        ds = light.get("D_stat") or {}
        sc = light.get("scales") or {}
        row = {"epoch": int(rec["epoch"]), "global_step": int(rec.get("global_step") or 0)}
        for i in (0, 1):
            d = ds.get(str(i)) or {}
            s = sc.get(str(i)) or {}
            row[f"D_frobenius_{i}"] = d.get("D_frobenius")
            row[f"W_frobenius_{i}"] = d.get("W_frobenius")
            row[f"R_ratio_{i}"] = d.get("R_ratio")
            row[f"D_cosine_to_W_{i}"] = d.get("D_cosine_to_W")
            row[f"rms_fc1_{i}"] = s.get("rms_fc1_z_plus_m")
            row[f"rms_correction_{i}"] = s.get("rms_lowrank_correction")
            row[f"correction_to_fc1_{i}"] = s.get("correction_to_fc1_ratio")
        mech_rows.append(row)
    mech_fields = ["epoch", "global_step"]
    for i in (0, 1):
        mech_fields += [f"D_frobenius_{i}", f"W_frobenius_{i}", f"R_ratio_{i}",
                        f"D_cosine_to_W_{i}", f"rms_fc1_{i}", f"rms_correction_{i}",
                        f"correction_to_fc1_{i}"]
    mech_sha = _write_csv(F.ANALYSIS / "mechanism_light.csv", mech_fields, mech_rows)
    best_mech = next((r for r in mech_rows if int(r["epoch"]) == best_epoch), None)
    mech_summary = {
        "round_id": F.ROUND_ID,
        "epoch12_baseline": baseline,
        "best_epoch": best_epoch,
        "best_checkpoint": best_mech,
        "change_vs_epoch12": {
            str(i): {
                "R_ratio_delta": (
                    None if not best_mech or not baseline.get(str(i)) else
                    float(best_mech[f"R_ratio_{i}"]) - float(baseline[str(i)]["R_ratio"])),
                "correction_to_fc1_delta": (
                    None if not best_mech or not baseline.get(str(i)) else
                    float(best_mech[f"correction_to_fc1_{i}"]) - float(baseline[str(i)]["correction_to_fc1_ratio"])),
            } for i in (0, 1)
        },
        "history": mech_rows,
        "train_valid_gap": {
            "RSSC_best_checkpoint": {
                "train_ap": best_train_ap, "valid_ap": cs_rssc["best_valid_ap"],
                "ap_gap_train_minus_valid": (None if best_train_ap is None else best_train_ap - cs_rssc["best_valid_ap"]),
                "train_bce": best_train_bce, "valid_bce": best_valid_bce,
                "bce_gap_valid_minus_train": (None if best_train_bce is None or best_valid_bce is None else best_valid_bce - best_train_bce),
            },
            "FIXED1_best_checkpoint": {
                "train_ap": fixed1_best_train_ap, "valid_ap": fixed1_best_valid_ap,
                "ap_gap_train_minus_valid": fixed1_best_train_ap - fixed1_best_valid_ap,
                "train_bce": fixed1_best_train_bce, "valid_bce": fixed1_best_valid_bce,
                "bce_gap_valid_minus_train": fixed1_best_valid_bce - fixed1_best_train_bce,
            },
            "RSSC_epoch12_prefix": {
                "train_ap": _f((F.read_json(F.SRC_RUNS / "best_checkpoint_eval.json")["train"]).get("ap")),
                "valid_ap": _f(r12_ap["valid_ap"]) if r12_ap else None,
            },
        },
        "note": "light only: D/W and correction/fc1 RMS; no frozen deletion branch experiment",
    }
    F.write_json(F.ANALYSIS / "mechanism_light.json", mech_summary)

    # decision ---------------------------------------------------------------
    d_best = cs_rssc["best_valid_ap"] - fixed1_best_valid_ap
    d_end = cs_rssc["endpoint_valid_ap"] - cs_fixed1["endpoint_valid_ap"]
    d_last5 = cs_rssc["last5_mean_valid_ap"] - cs_fixed1["last5_mean_valid_ap"]
    d_high = comparisons["RSSC_minus_FIXED1_best"]["high_support"]["mean_delta_ap"]
    d_train = (None if best_train_ap is None else best_train_ap - fixed1_best_train_ap)
    flags = []
    if not complete:
        flags.append("INCOMPLETE_OR_BUDGET_LIMITED")
    if rssc_endpoint_epoch < F.TARGET_HORIZON:
        flags.append("HORIZON_TRUNCATED")
    conflict = (
        d_best * d_end < 0 and abs(d_best) >= SIGNAL_AP and abs(d_end) >= SIGNAL_AP
    ) or (
        d_best * d_last5 < 0 and abs(d_best) >= SIGNAL_AP and abs(d_last5) >= SIGNAL_AP
    ) or (
        d_best * (d_high if d_high is not None else d_best) < 0
        and abs(d_best) >= SIGNAL_AP and abs(d_high or 0.0) >= SIGNAL_AP
    )
    if flags:
        label = "INCONCLUSIVE"
    elif d_train is not None and d_train >= 0.010 and d_best <= 0.0 and d_end <= 0.0:
        label = "FIT_WITHOUT_GENERALIZATION_GAIN"
    elif d_best <= -0.005 or (d_best <= -SIGNAL_AP and d_end <= -0.001 and d_last5 <= -0.001):
        label = "REGRESSION"
    elif (
        d_best >= SIGNAL_AP and d_end >= -0.001 and d_last5 >= -0.001
        and (d_high is None or d_high >= -0.001)
    ):
        label = "PERFORMANCE_ADVANCE"
    elif conflict:
        label = "INCONCLUSIVE"
    elif -SIGNAL_AP < d_best < SIGNAL_AP:
        label = "NEAR_BASELINE"
    else:
        label = "INCONCLUSIVE"

    changed = bool(label == "PERFORMANCE_ADVANCE")
    summary = {
        "round_id": F.ROUND_ID,
        "decision": label,
        "decision_is_single_seed_exploratory": True,
        "flags": flags,
        "headline_answer": {
            "question": "did completing the 25-epoch horizon change the previous research judgement?",
            "previous_judgement": "ROLE_DIFFERENCE_WITHOUT_PERFORMANCE_ADVANCE (RSSC > TIED but < FIXED1 at 12 epochs)",
            "current_judgement": label,
            "changed": changed,
            "label_changed_vs_previous": bool(label != "ROLE_DIFFERENCE_WITHOUT_PERFORMANCE_ADVANCE"),
            "changed_meaning": "changed=True only for PERFORMANCE_ADVANCE; a different no-advance label is reported separately",
            "delta_best_valid_ap": d_best,
            "delta_endpoint_valid_ap": d_end,
            "delta_last5_valid_ap": d_last5,
            "delta_high_support_ap": d_high,
            "delta_train_ap": d_train,
        },
        "coordinates": {
            "horizon": int(rssc_endpoint_epoch),
            "RSSC_best_epoch": best_epoch,
            "RSSC_best_valid_ap": cs_rssc["best_valid_ap"],
            "FIXED1_best_epoch": fixed1_best_epoch,
            "FIXED1_best_valid_ap": fixed1_best_valid_ap,
            "RSSC_endpoint_valid_ap": cs_rssc["endpoint_valid_ap"],
            "FIXED1_endpoint_valid_ap": cs_fixed1["endpoint_valid_ap"],
            "RSSC_last5_mean_valid_ap": cs_rssc["last5_mean_valid_ap"],
            "FIXED1_last5_mean_valid_ap": cs_fixed1["last5_mean_valid_ap"],
            "RSSC_best_train_ap": best_train_ap,
            "FIXED1_best_train_ap": fixed1_best_train_ap,
            "RSSC_best_train_bce": best_train_bce,
            "FIXED1_best_train_bce": fixed1_best_train_bce,
            "RSSC_best_valid_bce": best_valid_bce,
            "FIXED1_best_valid_bce": fixed1_best_valid_bce,
        },
        "curve_summaries": {"RSSC": cs_rssc, "FIXED1_hist": cs_fixed1},
        "paired_deltas": {k: v["meta"] for k, v in comparisons.items()},
        "paired_headline": {
            k: {
                "mean_delta_ap": v["mean_delta_ap"],
                "tasks_improved": v["tasks_improved"], "tasks_degraded": v["tasks_degraded"],
                "bootstrap_ci95": v["task_bootstrap_2000"]["mean_delta_ci95"],
                "high_support_mean_delta_ap": v["high_support"]["mean_delta_ap"],
                "drop_top3_abs_delta_mean": v["drop_top3_abs_delta"]["mean_delta_ap"],
            } for k, v in comparisons.items()
        },
        "thresholds": {
            "signal_ap": SIGNAL_AP, "high_support_min_positive_and_negative": HIGH_SUPPORT_MIN,
            "bootstrap_resamples": BOOTSTRAP_N, "drop_top_k": DROP_TOP_K,
            "performance_advance": "delta_best >= +0.003 and endpoint/last5/high-support >= -0.001",
        },
        "budget": {
            "gpu_hours_this_round": run.get("gpu_hours"),
            "gpu_seconds_this_round": run.get("gpu_seconds"),
            "epochs_run": rssc_endpoint_epoch,
            "seeds": [0], "dataset": "ogbg-molpcba", "test_scored": False,
        },
        "artifacts": {
            "epoch_curves_csv_sha256": curve_sha,
            "main_table_csv_sha256": main_sha,
            "task_deltas_csv_sha256": task_sha,
            "mechanism_light_csv_sha256": mech_sha,
        },
        "created_utc": F.utcnow(),
    }
    F.write_json(F.ANALYSIS / "paired_comparisons.json", {
        "round_id": F.ROUND_ID, "horizon": int(rssc_endpoint_epoch),
        "reference": "real FIXED1 seed-0 25-epoch prefix (not W2, not W2_JOINT)",
        "comparisons": comparisons,
        "curve_summaries": {"RSSC": cs_rssc, "FIXED1_hist": cs_fixed1},
        "thresholds": summary["thresholds"],
    })
    F.write_json(F.ANALYSIS / "summary.json", summary)
    F.log(json.dumps({"decision": label, "delta_best": d_best, "delta_end": d_end,
                      "delta_last5": d_last5, "delta_high": d_high,
                      "delta_train": d_train}, indent=1))
    return 0 if complete else 3


if __name__ == "__main__":
    raise SystemExit(main())
