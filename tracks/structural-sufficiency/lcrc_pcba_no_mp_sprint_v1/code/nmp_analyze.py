"""Post-run analysis and pre-registered decision labels (no further training)."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _configure_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_configure_imports()

import nmp_common as C  # noqa: E402
import nmp_data as D  # noqa: E402

PHASE_DIRS = {
    "B": "phaseB_seed0_{arm}",
    "C": "phaseC_seed0_{arm}",
    "D": "phaseD_seed1_{arm}",
}
HISTORICAL_DIR = "phase3_seed{seed}_{arm}"


def load_run(rel: str) -> dict[str, Any] | None:
    path = C.RUNS / rel / "run.json"
    if not path.exists():
        return None
    run = C.read_json(path)
    run["_dir"] = C.track_rel(path.parent)
    run["_path"] = C.track_rel(path)
    return run


def historical(seed: int, arm: str) -> dict[str, Any] | None:
    path = C.SG_SPRINT / "runs" / HISTORICAL_DIR.format(seed=seed, arm=arm) / "run.json"
    if not path.exists():
        return None
    return C.read_json(path)


def fixed(run: Mapping[str, Any] | None, epoch: int) -> dict[str, Any] | None:
    if run is None:
        return None
    return (run.get("fixed_epoch_reports") or {}).get(str(int(epoch)))


def ap_at(run: Mapping[str, Any] | None, epoch: int) -> float | None:
    rep = fixed(run, epoch)
    return None if rep is None else float(rep["ap"])


def per_task(run: Mapping[str, Any] | None, epoch: int) -> list[float | None] | None:
    rep = fixed(run, epoch)
    return None if rep is None else list(rep["per_task_ap"])


def train_positive(task: Any) -> list[int]:
    return D.task_counts(task)["train"]["positive"]


def strata_from_counts(positives: list[int]) -> list[str]:
    return [
        "support_lt5" if p < 5 else "support_5_19" if p < 20 else "support_ge20" for p in positives
    ]


def improved(arm_ap: list[float | None], ref_ap: list[float | None]) -> dict[str, Any]:
    rows = []
    for a, b in zip(arm_ap, ref_ap):
        if a is None or b is None:
            continue
        rows.append(float(a) - float(b))
    arr = np.asarray(rows, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean_delta": float(arr.mean()) if arr.size else None,
        "median_delta": float(np.median(arr)) if arr.size else None,
        "n_improved": int((arr > 0).sum()),
        "n_regressed": int((arr < 0).sum()),
        "n_equal": int((arr == 0).sum()),
    }


def subset(arm_ap, ref_ap, positives, predicate) -> dict[str, Any]:
    deltas = []
    for a, b, p in zip(arm_ap, ref_ap, positives):
        if a is None or b is None:
            continue
        if predicate(p):
            deltas.append(float(a) - float(b))
    arr = np.asarray(deltas, dtype=np.float64)
    return {
        "n_tasks": int(arr.size),
        "mean_delta": float(arr.mean()) if arr.size else None,
        "n_improved": int((arr > 0).sum()),
    }


def arm_row(run: Mapping[str, Any] | None, epoch: int, hist_c, hist_v, positives) -> dict[str, Any]:
    if run is None:
        return {"available": False}
    rep = fixed(run, epoch)
    if rep is None:
        return {"available": False, "run_status": run.get("status")}
    a = list(rep["per_task_ap"])
    c = per_task(hist_c, epoch)
    v = per_task(hist_v, epoch)
    out = {
        "available": True,
        "run_dir": run.get("_dir"),
        "status": run.get("status"),
        "stop_reason": run.get("stop_reason"),
        "seed": int(run["seed"]),
        "phase": run["phase"],
        "ap": float(rep["ap"]),
        "valid_bce": float(rep["valid_bce"]),
        "epoch": int(epoch),
        "gpu_hours": float(run.get("gpu_hours", float("nan"))),
        "global_steps_run": int(run.get("global_steps_run", 0)),
        "parameters": int(run.get("parameters", 0)),
        "new_parameters": int(run.get("new_parameters", 0)),
        "train_eval_final": run.get("train_eval_final"),
        "per_task_ap": a,
    }
    out["vs_C"] = None if c is None else improved(a, c)
    out["vs_V"] = None if v is None else improved(a, v)
    if c is not None:
        out["vs_C_support_ge20"] = subset(a, c, positives, lambda p: p >= 20)
    if v is not None:
        out["vs_V_support_ge20"] = subset(a, v, positives, lambda p: p >= 20)
    return out


def build_phase_b(task, runs: Mapping[str, Any]) -> dict[str, Any]:
    positives = train_positive(task)
    arms = {}
    for arm in C.ARM_ORDER:
        run = load_run(PHASE_DIRS["B"].format(arm=arm))
        runs.setdefault("B", {})[arm] = run
        a10 = ap_at(run, 10)
        a20 = ap_at(run, 20)
        row = {
            "available": run is not None and a10 is not None and a20 is not None,
            "run_dir": None if run is None else run.get("_dir"),
            "status": None if run is None else run.get("status"),
            "gpu_hours": None if run is None else run.get("gpu_hours"),
            "ap10": a10,
            "ap20": a20,
            "ap20_minus_ap10": None if a10 is None or a20 is None else float(a20 - a10),
            "train_loss_at_20": None,
        }
        if run is not None:
            rows = [r for r in _curve(run) if int(r["epoch"]) == 20]
            row["train_loss_at_20"] = None if not rows else float(rows[-1]["train_loss"])
        arms[arm] = row
    complete = all(v["available"] for v in arms.values())
    early = bool(
        complete
        and all(
            float(v["ap20"]) < C.DECISION["phase_b"]["ap20_floor"]
            and float(v["ap20_minus_ap10"]) < C.DECISION["phase_b"]["ap20_minus_ap10_floor"]
            for v in arms.values()
        )
    )
    return {
        "schema": "lcrc-pcba-no-mp-phase-b-screen-v1",
        "round_id": C.ROUND_ID,
        "phase": "B",
        "seed": 0,
        "epochs": [1, 20],
        "eval_epochs": [10, 20],
        "arms": arms,
        "rule": C.DECISION["phase_b"],
        "all_three_complete": complete,
        "early_no_signal": early,
        "label": C.DECISION["phase_b"]["label_if_all_below"] if early else "CONTINUE_TO_PHASE_C",
        "test_evaluated": False,
    }


def _curve(run: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = Path(C.REPO) / (run.get("curve") or "")
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def build_phase_c(task, runs: Mapping[str, Any]) -> dict[str, Any]:
    positives = train_positive(task)
    hist_c = historical(0, "C")
    hist_v = historical(0, "V")
    arms = {}
    for arm in C.ARM_ORDER:
        run = load_run(PHASE_DIRS["C"].format(arm=arm))
        runs.setdefault("C", {})[arm] = run
        arms[arm] = arm_row(run, 40, hist_c, hist_v, positives)
    complete = all(v.get("available") for v in arms.values())
    stop = bool(
        complete and all(float(v["ap"]) < C.DECISION["phase_c"]["ap40_floor"] for v in arms.values())
    )
    return {
        "schema": "lcrc-pcba-no-mp-phase-c-full-v1",
        "round_id": C.ROUND_ID,
        "phase": "C",
        "seed": 0,
        "primary_epoch": 40,
        "arms": arms,
        "historical_C_seed0": None if hist_c is None else ap_at(hist_c, 40),
        "historical_V_seed0": None if hist_v is None else ap_at(hist_v, 40),
        "rule": C.DECISION["phase_c"],
        "all_three_complete": complete,
        "no_strong_seed0_signal": stop,
        "label": C.DECISION["phase_c"]["label_if_all_below"] if stop else "CONTINUE_TO_PHASE_D",
        "test_evaluated": False,
    }


def build_phase_d(task, runs: Mapping[str, Any]) -> dict[str, Any]:
    positives = train_positive(task)
    hist_c = {s: historical(s, "C") for s in (0, 1)}
    hist_v = {s: historical(s, "V") for s in (0, 1)}
    arms = {}
    for arm in C.ARM_ORDER:
        run = load_run(PHASE_DIRS["D"].format(arm=arm))
        runs.setdefault("D", {})[arm] = run
        row = arm_row(run, 40, hist_c[1], hist_v[1], positives)
        if row.get("available"):
            row["ap10"] = ap_at(run, 10)
            row["ap20"] = ap_at(run, 20)
            row["ap30"] = ap_at(run, 30)
            seed0 = load_run(PHASE_DIRS["C"].format(arm=arm))
            runs.setdefault("C", {})[arm] = seed0
            row["seed0_ap40"] = ap_at(seed0, 40)
            row["two_seed_mean_ap40"] = float(
                (float(row["ap"]) + float(row["seed0_ap40"])) / 2.0
            ) if row.get("seed0_ap40") is not None else None
        arms[arm] = row
    complete = all(v.get("available") for v in arms.values())
    return {
        "schema": "lcrc-pcba-no-mp-phase-d-confirmation-v1",
        "round_id": C.ROUND_ID,
        "phase": "D",
        "seed": 1,
        "primary_epoch": 40,
        "eval_epochs": [10, 20, 30, 40],
        "arms": arms,
        "historical_C_seed1": None if hist_c[1] is None else ap_at(hist_c[1], 40),
        "historical_V_seed1": None if hist_v[1] is None else ap_at(hist_v[1], 40),
        "all_three_complete": complete,
        "test_evaluated": False,
    }


def build_decisions(task, phase_b, phase_c, phase_d) -> dict[str, Any]:
    labels = []
    if not phase_b["all_three_complete"]:
        labels.append(C.DECISION["incomplete"]["label"])
        return {"labels": labels, "reason": "Phase B incomplete"}
    if phase_b["early_no_signal"]:
        labels.append(C.DECISION["no_static_gain"]["label"])
        labels.append(C.DECISION["phase_b"]["label_if_all_below"])
        return {"labels": labels, "reason": "Phase B pre-registered stop"}
    if not phase_c["all_three_complete"]:
        labels.append(C.DECISION["incomplete"]["label"])
        return {"labels": labels, "reason": "Phase C incomplete"}
    if phase_c["no_strong_seed0_signal"]:
        labels.append(C.DECISION["no_static_gain"]["label"])
        labels.append(C.DECISION["phase_c"]["label_if_all_below"])
        return {"labels": labels, "reason": "Phase C pre-registered stop"}
    if not phase_d["all_three_complete"]:
        labels.append(C.DECISION["incomplete"]["label"])
        labels.append(C.DECISION["second_seed_skip"])
        return {"labels": labels, "reason": "Phase D not complete"}

    rows = phase_d["arms"]
    means = {arm: float(r["two_seed_mean_ap40"]) for arm, r in rows.items()}
    static_candidates = {a: means[a] for a in ("S", "A")}
    best_static = max(static_candidates, key=static_candidates.get)
    best_mean = static_candidates[best_static]
    stats = {}
    for arm in C.ARM_ORDER:
        r = rows[arm]
        stats[arm] = {
            "seed0_ap40": r["seed0_ap40"],
            "seed1_ap40": r["ap"],
            "mean_ap40": means[arm],
            "gap_to_V_mean": means[arm] - C.AP_V_MEAN,
            "seed0_gain_vs_C": r["seed0_ap40"] - C.HISTORICAL["C"]["seed0_ap40"],
            "seed1_gain_vs_C": r["ap"] - C.HISTORICAL["C"]["seed1_ap40"],
            "mean_gain_vs_C": means[arm]
            - C.HISTORICAL["C"]["mean"],
            "gpu_hours_total": (r.get("gpu_hours") or 0.0)
            + ((phase_c["arms"].get(arm) or {}).get("gpu_hours") or 0.0),
        }
    best = stats[best_static]
    close = bool(
        best["seed0_ap40"] >= C.DECISION["static_close"]["per_seed_ap_min"]
        and best["seed1_ap40"] >= C.DECISION["static_close"]["per_seed_ap_min"]
        and abs(best["mean_ap40"] - C.AP_V_MEAN) <= C.DECISION["static_close"]["mean_gap_to_v_max"]
        and best["seed0_gain_vs_C"] >= C.DECISION["static_close"]["per_seed_gain_over_c_min"]
        and best["seed1_gain_vs_C"] >= C.DECISION["static_close"]["per_seed_gain_over_c_min"]
    )
    positive = bool(
        best["seed0_gain_vs_C"] >= C.DECISION["static_close"]["per_seed_gain_over_c_min"]
        and best["seed1_gain_vs_C"] >= C.DECISION["static_close"]["per_seed_gain_over_c_min"]
    )
    if close:
        static_label = C.DECISION["static_close"]["label"]
    elif positive:
        static_label = C.DECISION["static_positive"]["label"]
    else:
        static_label = C.DECISION["no_static_gain"]["label"]
    labels.append(static_label)

    g = stats["G"]
    g_promising = bool(
        abs(g["mean_ap40"] - C.AP_V_MEAN) <= 0.008
        and (g["mean_ap40"] - best_mean) >= 0.008
    )
    if g_promising:
        labels.append(C.DECISION["global_communication"]["label"])
    v_best = bool(C.AP_V_MEAN - max(means.values()) >= 0.008)
    if v_best:
        labels.append(C.DECISION["v_retains"]["label"])
    return {
        "labels": labels,
        "best_static_arm": best_static,
        "static_close": close,
        "static_positive": positive,
        "g_promising": g_promising,
        "v_retains_advantage": v_best,
        "statistics": stats,
        "v_mean": C.AP_V_MEAN,
    }


def build_per_task_summary(task, runs) -> dict[str, Any]:
    hist = {
        (arm, seed): historical(seed, arm) for arm in ("C", "V") for seed in (0, 1)
    }
    out: dict[str, Any] = {}

    def stats(a, b):
        aa = np.asarray([np.nan if x is None else float(x) for x in a], dtype=np.float64)
        bb = np.asarray([np.nan if x is None else float(x) for x in b], dtype=np.float64)
        diff = aa - bb
        return {
            "improved": int(np.nansum(diff > 0)),
            "regressed": int(np.nansum(diff < 0)),
            "mean": float(np.nanmean(diff)),
            "median": float(np.nanmedian(diff)),
        }

    for arm in C.ARM_ORDER:
        arrays = {}
        for seed, run in ((0, runs.get("C", {}).get(arm)), (1, runs.get("D", {}).get(arm))):
            a = per_task(run, 40)
            if a is None:
                continue
            arrays[seed] = [np.nan if x is None else float(x) for x in a]
            out[f"{arm}{seed}"] = {
                "ap": ap_at(run, 40),
                "vs_C": stats(a, per_task(hist[("C", seed)], 40)),
                "vs_V": stats(a, per_task(hist[("V", seed)], 40)),
            }
        if len(arrays) == 2:
            mean_a = list(np.mean(np.asarray([arrays[0], arrays[1]], dtype=np.float64), axis=0))
            mean_c = list(
                np.mean(
                    np.asarray([
                        [np.nan if x is None else float(x) for x in per_task(hist[("C", s)], 40)]
                        for s in (0, 1)
                    ], dtype=np.float64),
                    axis=0,
                )
            )
            mean_v = list(
                np.mean(
                    np.asarray([
                        [np.nan if x is None else float(x) for x in per_task(hist[("V", s)], 40)]
                        for s in (0, 1)
                    ], dtype=np.float64),
                    axis=0,
                )
            )
            out[f"{arm}_mean"] = {
                "ap": float(np.nanmean(mean_a)),
                "vs_C": stats(mean_a, mean_c),
                "vs_V": stats(mean_a, mean_v),
            }
    return out


def write_per_task(task, phase_c, phase_d, runs) -> str:
    positives = train_positive(task)
    strata = strata_from_counts(positives)
    hist = {
        ("C", 0): historical(0, "C"),
        ("C", 1): historical(1, "C"),
        ("V", 0): historical(0, "V"),
        ("V", 1): historical(1, "V"),
    }
    rows = []
    for arm in C.ARM_ORDER:
        for seed, run in ((0, runs.get("C", {}).get(arm)), (1, runs.get("D", {}).get(arm))):
            a = per_task(run, 40)
            rep = fixed(run, 40)
            obs = None if rep is None else rep.get("observed")
            c = per_task(hist[("C", seed)], 40)
            v = per_task(hist[("V", seed)], 40)
            if a is None:
                continue
            for t in range(C.N_TASKS):
                av = None if a[t] is None else float(a[t])
                cv = None if c is None or c[t] is None else float(c[t])
                vv = None if v is None or v[t] is None else float(v[t])
                rows.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "task": t,
                        "ap_arm40": "" if av is None else av,
                        "ap_C40": "" if cv is None else cv,
                        "delta_vs_C": "" if av is None or cv is None else av - cv,
                        "ap_V40": "" if vv is None else vv,
                        "delta_vs_V": "" if av is None or vv is None else av - vv,
                        "train_positive": int(positives[t]),
                        "valid_observed": "" if obs is None else int(obs[t]),
                        "stratum": strata[t],
                    }
                )
    path = C.ANALYSIS / "per_task_comparison.csv"
    fields = [
        "arm", "seed", "task", "ap_arm40", "ap_C40", "delta_vs_C",
        "ap_V40", "delta_vs_V", "train_positive", "valid_observed", "stratum",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return C.track_rel(path)


def build_ledger() -> dict[str, Any]:
    entries = []
    failed = []
    for phase, fmt in PHASE_DIRS.items():
        for arm in C.ARM_ORDER:
            run = load_run(fmt.format(arm=arm))
            if run is None:
                continue
            row = {
                "phase": phase,
                "arm": arm,
                "seed": int(run.get("seed", -1)),
                "run_dir": run.get("_dir"),
                "status": run.get("status"),
                "gpu_hours": float(run.get("gpu_hours", 0.0)),
                "started_utc": run.get("started_utc"),
                "finished_utc": run.get("finished_utc"),
                "git_commit": run.get("git_commit"),
                "environment": run.get("environment"),
            }
            if run.get("status") == "COMPLETED":
                entries.append(row)
            else:
                failed.append(row)
    preflight = []
    for path in sorted(C.ANALYSIS.glob("preflight_gpu*.json")):
        try:
            d = C.read_json(path)
            preflight.append(
                {
                    "file": C.track_rel(path),
                    "status": d.get("status"),
                    "gpu_hours": float(d.get("gpu_hours", 0.0)),
                    "gpu_seconds": float(d.get("gpu_seconds", 0.0)),
                    "device": d.get("device"),
                    "arms": d.get("arms"),
                }
            )
        except Exception:  # pragma: no cover
            continue
    extra_failed: list[dict[str, Any]] = []
    extra_path = C.ANALYSIS / "failed_gpu_attempts.json"
    if extra_path.exists():
        extra_failed = list(C.read_json(extra_path).get("entries") or [])
    total = sum(e["gpu_hours"] for e in entries) + sum(
        e["gpu_hours"] for e in failed
    ) + sum(e["gpu_hours"] for e in preflight) + sum(
        float(e.get("gpu_hours", 0.0)) for e in extra_failed
    )
    return {
        "schema": "lcrc-pcba-no-mp-gpu-ledger-v1",
        "round_id": C.ROUND_ID,
        "budget": C.BUDGET,
        "training_entries": entries,
        "failed_entries": failed,
        "failed_preflight_or_retry_entries": extra_failed,
        "preflight_entries": preflight,
        "gpu_hours_total": float(total),
        "remaining_to_target": float(C.BUDGET["gpu_target_hours"] - total),
        "remaining_to_hard": float(C.BUDGET["gpu_hard_hours"] - total),
        "within_target": bool(total <= C.BUDGET["gpu_target_hours"]),
        "within_hard": bool(total <= C.BUDGET["gpu_hard_hours"]),
        "note": "sum of run-process wall seconds / 3600 per GPU job; all attempts included",
        "test_evaluated": False,
    }


def build_registry() -> dict[str, Any]:
    jobs = []
    for phase, fmt in PHASE_DIRS.items():
        for arm in C.ARM_ORDER:
            run = load_run(fmt.format(arm=arm))
            if run is None:
                continue
            env = run.get("environment") or {}
            jobs.append(
                {
                    "experiment": f"nmp-p{phase.lower()}-s{run.get('seed')}-{arm}",
                    "run_dir": run.get("_dir"),
                    "phase": phase,
                    "arm": arm,
                    "seed": run.get("seed"),
                    "state": str(run.get("status", "")).lower(),
                    "stop_reason": run.get("stop_reason"),
                    "host": env.get("hostname"),
                    "device": env.get("cuda_device_name"),
                    "commit": run.get("git_commit"),
                    "torch": env.get("torch"),
                    "torch_cuda": env.get("torch_cuda"),
                    "driver_version": env.get("nvidia_driver"),
                    "started_utc": run.get("started_utc"),
                    "finished_utc": run.get("finished_utc"),
                    "gpu_hours": run.get("gpu_hours"),
                }
            )
    return {
        "schema": "lcrc-pcba-no-mp-run-registry-v1",
        "round_id": C.ROUND_ID,
        "jobs": jobs,
        "test_evaluated": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args(argv)
    task = D.load_task()
    runs: dict[str, dict[str, Any]] = {"B": {}, "C": {}, "D": {}}
    phase_b = build_phase_b(task, runs)
    phase_c = build_phase_c(task, runs)
    phase_d = build_phase_d(task, runs)
    decisions = build_decisions(task, phase_b, phase_c, phase_d)
    C.write_json(C.ANALYSIS / "phase_b_seed0_screen.json", phase_b)
    C.write_json(C.ANALYSIS / "phase_c_seed0_full.json", phase_c)
    C.write_json(C.ANALYSIS / "phase_d_seed1_confirmation.json", phase_d)
    C.write_json(C.ANALYSIS / "decision_labels.json", decisions)
    p = write_per_task(task, phase_c, phase_d, runs)
    C.write_json(C.ANALYSIS / "per_task_summary.json", build_per_task_summary(task, runs))
    ledger = build_ledger()
    C.write_json(C.ANALYSIS / "gpu_ledger.json", ledger)
    C.write_json(C.ANALYSIS / "run_registry.json", build_registry())
    C.log(f"phaseB={phase_b['label']} phaseC={phase_c['label']} decisions={decisions['labels']}")
    C.log(f"wrote per-task {p} and gpu ledger total={ledger['gpu_hours_total']:.4f} h")
    ok = phase_b["all_three_complete"]
    if not args.allow_missing and not ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
