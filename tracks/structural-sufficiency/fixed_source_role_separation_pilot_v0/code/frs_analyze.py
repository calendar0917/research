"""FSR v0 analysis: tables, paired task comparisons, decision label.

Reads only the two new runs plus the real historical FIXED1 seed-0 curve and
its per-task Valid AP file (same-horizon prefix, never the 25-epoch best value).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _configure_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_configure_imports()

import frs_common as FC

BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 0
IMPROVE_TOL = 1e-6
SIGNAL_AP = 0.003
TIE_AP = 0.001
HIGH_SUPPORT_MIN = 20
DROP_TOP_K = 3


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


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
    return FC.sha256_file(path)


def _f(x: Any) -> float | None:
    if x in (None, ""):
        return None
    return float(x)


def curve_rows(arm: str, horizon: int) -> list[dict[str, Any]]:
    run_dir = FC.arm_out_dir(arm)
    rows = _read_csv(run_dir / "train_curve.csv")
    out = []
    for r in rows:
        if int(r["epoch"]) > int(horizon):
            continue
        out.append(
            {
                "arm": arm,
                "source": "new_run",
                "epoch": int(r["epoch"]),
                "learning_rate": _f(r.get("learning_rate")),
                "train_ap": _f(r.get("train_ap")),
                "train_bce": _f(r.get("train_bce")),
                "train_evaluated": r.get("train_evaluated"),
                "valid_ap": _f(r.get("valid_ap")),
                "valid_bce": _f(r.get("valid_bce")),
                "epoch_seconds": _f(r.get("epoch_seconds")),
                "is_best": r.get("is_best"),
                "best_valid_ap_so_far": _f(r.get("best_valid_ap_so_far")),
                "R1": _f(r.get("R1")),
                "R2": _f(r.get("R2")),
                "rms_fc1_term_1": _f(r.get("rms_fc1_term_1")),
                "rms_correction_1": _f(r.get("rms_correction_1")),
                "grad_norm_lowrank_total": _f(r.get("grad_norm_lowrank_total")),
            }
        )
    return out


def hist_curve_rows(horizon: int) -> list[dict[str, Any]]:
    out = []
    for r in FC.load_hist_curve():
        if r["epoch"] > int(horizon):
            continue
        out.append(
            {
                "arm": "FIXED1_hist",
                "source": "historical_seed0_run",
                "epoch": int(r["epoch"]),
                "learning_rate": r["learning_rate"],
                "train_ap": r["train_ap"],
                "train_bce": r["train_bce"],
                "train_evaluated": "True",
                "valid_ap": r["valid_ap"],
                "valid_bce": r["valid_bce"],
                "epoch_seconds": r["epoch_seconds"],
                "is_best": "",
                "best_valid_ap_so_far": "",
                "R1": None,
                "R2": None,
                "rms_fc1_term_1": None,
                "rms_correction_1": None,
                "grad_norm_lowrank_total": None,
            }
        )
    return out


def run_table_row(arm: str) -> dict[str, Any]:
    run = FC.read_json(FC.arm_out_dir(arm) / "run.json")
    best_eval = FC.read_json(FC.arm_out_dir(arm) / "best_checkpoint_eval.json")
    return {
        "arm": arm,
        "status": run.get("status"),
        "horizon": int(run.get("horizon")),
        "epochs_run": int(run.get("epochs_run")),
        "best_epoch": run.get("best_epoch"),
        "best_valid_ap": run.get("best_valid_ap"),
        "best_valid_bce": run.get("best_valid_bce"),
        "endpoint_valid_ap": (run.get("horizon_endpoint") or {}).get("valid_ap"),
        "endpoint_valid_bce": (run.get("horizon_endpoint") or {}).get("valid_bce"),
        "last3_mean_valid_ap": run.get("last3_mean_valid_ap"),
        "best_train_ap": (best_eval.get("train") or {}).get("ap"),
        "best_train_bce": (best_eval.get("train") or {}).get("bce"),
        "parameters": run.get("parameters"),
        "rank": run.get("rank"),
        "gpu_hours": run.get("gpu_hours"),
        "gpu_seconds": run.get("gpu_seconds"),
        "lowrank_grad_first_epoch_total": None,
        "test_scored": run.get("test_scored"),
    }


def hist_table_row(horizon: int) -> dict[str, Any]:
    pre = FC.hist_prefix(horizon)
    ep = pre["endpoint_epoch"]
    hist_rows = [r for r in FC.load_hist_curve() if r["epoch"] == ep]
    row = hist_rows[0]
    return {
        "arm": "FIXED1_hist",
        "status": FC.HIST["status"],
        "horizon": int(horizon),
        "epochs_run": int(horizon),
        "best_epoch": pre["best_epoch"],
        "best_valid_ap": pre["best_valid_ap"],
        "best_valid_bce": row["valid_bce"],
        "endpoint_valid_ap": pre["endpoint_valid_ap"],
        "endpoint_valid_bce": pre["endpoint_valid_bce"],
        "last3_mean_valid_ap": pre["last3_mean_valid_ap"],
        "best_train_ap": row["train_ap"],
        "best_train_bce": row["train_bce"],
        "parameters": FC.PARAMS_FIXED1,
        "rank": 0,
        "gpu_hours": None,
        "gpu_seconds": None,
        "lowrank_grad_first_epoch_total": None,
        "test_scored": None,
    }


def per_task_at(arm: str, epoch: int) -> dict[int, dict[str, Any]]:
    path = FC.arm_out_dir(arm) / "per_task_valid.csv"
    out: dict[int, dict[str, Any]] = {}
    for r in _read_csv(path):
        if int(r["epoch"]) != int(epoch) or r["split"] != "valid":
            continue
        out[int(r["task"])] = {
            "ap": (None if r["ap"] == "" else float(r["ap"])),
            "observed": int(r["observed"]),
            "positive": int(r["positive"]),
            "negative": int(r["negative"]),
        }
    return out


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
    high = np.asarray(
        [t for t in tasks if support[t]["positive"] >= HIGH_SUPPORT_MIN and support[t]["negative"] >= HIGH_SUPPORT_MIN],
        dtype=np.int64,
    )
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
    out = {
        "name": name,
        "meta": dict(meta),
        "n_tasks_paired": int(n),
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
        "per_task": [
            {
                "task": int(t),
                "a_ap": float(a[t]),
                "b_ap": float(b[t]),
                "delta_ap": float(a[t]) - float(b[t]),
                "valid_positive": int(support[t]["positive"]),
                "valid_negative": int(support[t]["negative"]),
            }
            for t in tasks
        ],
    }
    return out


def curve_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if r.get("valid_ap") is not None]
    best = max(valid, key=lambda r: (float(r["valid_ap"]), -int(r["epoch"])))
    last3 = valid[-3:]
    late = [r for r in valid if int(r["epoch"]) >= max(1, int(valid[-1]["epoch"]) - 4)]
    return {
        "best_epoch": int(best["epoch"]),
        "best_valid_ap": float(best["valid_ap"]),
        "endpoint_epoch": int(valid[-1]["epoch"]),
        "endpoint_valid_ap": float(valid[-1]["valid_ap"]),
        "last3_mean_valid_ap": float(np.mean([float(r["valid_ap"]) for r in last3])),
        "last5_mean_valid_ap": float(np.mean([float(r["valid_ap"]) for r in late])),
    }


def mechanism_summary() -> dict[str, Any]:
    out: dict[str, Any] = {"arms": {}}
    for arm in FC.ARMS:
        path = FC.arm_out_dir(arm) / "mechanism.json"
        if not path.is_file():
            out["arms"][arm] = None
            continue
        rec = FC.read_json(path)
        blocks = {}
        for i in (0, 1):
            branch = rec["blocks"][str(i)]["rank_branch"]
            scales = rec["scales_on_real_valid_batch"][str(i)]
            blocks[str(i)] = {
                "D_frobenius": branch["D_frobenius"],
                "W_frobenius": branch["W_frobenius"],
                "R_ratio": branch["R_ratio"],
                "D_cosine_to_W": branch["D_cosine_to_W"],
                "rms_z": scales["rms_z"],
                "rms_m": scales["rms_m"],
                "rms_fc1_z_plus_m": scales["rms_fc1_z_plus_m"],
                "rms_lowrank_correction": scales["rms_lowrank_correction"],
                "correction_to_fc1_ratio": scales["correction_to_fc1_ratio"],
            }
        out["arms"][arm] = {
            "weights_source": rec.get("weights_source"),
            "best_epoch": rec.get("best_epoch"),
            "blocks": blocks,
            "lowrank_gradients": rec.get("lowrank_gradients"),
            "frozen_branch_zero": rec.get("frozen_branch_zero"),
            "molecules": rec.get("molecules"),
        }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(FC.ANALYSIS))
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = {}
    for arm in FC.ARMS:
        path = FC.arm_out_dir(arm) / "run.json"
        if path.is_file():
            runs[arm] = FC.read_json(path)
    missing = [a for a in FC.ARMS if a not in runs]
    statuses = {a: runs[a].get("status") for a in runs}
    complete = all(statuses.get(a) == "COMPLETED" for a in FC.ARMS)
    horizon = int(max(runs[a].get("horizon", 0) for a in runs)) if runs else FC.PREFERRED_HORIZON
    horizons = sorted({int(runs[a].get("horizon", 0)) for a in runs})
    same_horizon = len(horizons) == 1
    if not same_horizon:
        horizon = max(horizons) if horizons else FC.PREFERRED_HORIZON

    # ---- epoch curves -------------------------------------------------------
    curve_fields = [
        "arm", "source", "epoch", "learning_rate", "train_ap", "train_bce", "train_evaluated",
        "valid_ap", "valid_bce", "epoch_seconds", "is_best", "best_valid_ap_so_far",
        "R1", "R2", "rms_fc1_term_1", "rms_correction_1", "grad_norm_lowrank_total",
    ]
    curve: list[dict[str, Any]] = []
    for arm in FC.ARMS:
        if arm in runs:
            curve.extend(curve_rows(arm, horizon))
    curve.extend(hist_curve_rows(horizon))
    curve_sha = _write_csv(out_dir / "epoch_curves.csv", curve_fields, curve)

    hist = FC.hist_prefix(horizon)
    hist_ident = FC.hist_run_identity()
    reference_ok = bool(
        hist_ident["status"] == "COMPLETED"
        and int(hist_ident["horizon"]) >= horizon
        and hist["epochs_available"] >= horizon
    )

    # ---- main table ---------------------------------------------------------
    main_fields = [
        "arm", "status", "horizon", "epochs_run", "best_epoch", "best_valid_ap",
        "best_valid_bce", "endpoint_valid_ap", "endpoint_valid_bce",
        "last3_mean_valid_ap", "best_train_ap", "best_train_bce", "parameters",
        "rank", "gpu_hours", "test_scored",
    ]
    main_rows = [run_table_row(arm) for arm in FC.ARMS if arm in runs]
    main_rows.append(hist_table_row(horizon))
    main_sha = _write_csv(out_dir / "main_table.csv", main_fields, main_rows)

    # ---- paired comparisons -------------------------------------------------
    comparisons: dict[str, Any] = {}
    support_source: dict[int, dict[str, Any]] = {}
    per_task_cache: dict[tuple[str, int], dict[int, dict[str, Any]]] = {}
    for arm in FC.ARMS:
        if arm not in runs or runs[arm].get("best_epoch") is None:
            continue
        epoch = int(runs[arm]["best_epoch"])
        rows = per_task_at(arm, epoch)
        per_task_cache[(arm, epoch)] = rows
        support_source = rows
    hist_pts = FC.hist_per_task(horizon, epoch=horizon)
    if not support_source:
        support_source = hist_pts

    def _ap_map(rows: Mapping[int, Mapping[str, Any]]) -> dict[int, float | None]:
        return {t: r["ap"] for t, r in rows.items()}

    if "TIED" in runs and "RSSC" in runs:
        tied_best = int(runs["TIED"]["best_epoch"])
        rssc_best = int(runs["RSSC"]["best_epoch"])
        comparisons["RSSC_minus_TIED_best"] = paired(
            "RSSC - TIED (best checkpoints, Valid)",
            _ap_map(per_task_cache[("RSSC", rssc_best)]),
            _ap_map(per_task_cache[("TIED", tied_best)]),
            support_source,
            meta={"a_epoch": rssc_best, "b_epoch": tied_best,
                  "a_valid_ap": runs["RSSC"]["best_valid_ap"],
                  "b_valid_ap": runs["TIED"]["best_valid_ap"],
                  "delta_ap": float(runs["RSSC"]["best_valid_ap"]) - float(runs["TIED"]["best_valid_ap"])},
        )
        # endpoint (last epoch of the shared horizon)
        r_end = per_task_at("RSSC", horizon)
        t_end = per_task_at("TIED", horizon)
        comparisons["RSSC_minus_TIED_endpoint"] = paired(
            "RSSC - TIED (horizon endpoint, Valid)", _ap_map(r_end), _ap_map(t_end), support_source,
            meta={"a_epoch": horizon, "b_epoch": horizon,
                  "a_valid_ap": runs["RSSC"]["horizon_endpoint"]["valid_ap"],
                  "b_valid_ap": runs["TIED"]["horizon_endpoint"]["valid_ap"],
                  "delta_ap": float(runs["RSSC"]["horizon_endpoint"]["valid_ap"]) - float(runs["TIED"]["horizon_endpoint"]["valid_ap"])},
        )

    if reference_ok:
        for arm in FC.ARMS:
            if arm not in runs:
                continue
            best_epoch = int(runs[arm]["best_epoch"])
            comparisons[f"{arm}_minus_FIXED1_hist_best"] = paired(
                f"{arm} (best) - FIXED1 historical same-horizon best (Valid)",
                _ap_map(per_task_cache[(arm, best_epoch)]),
                _ap_map(hist_pts),
                support_source,
                meta={"a_epoch": best_epoch, "b_epoch": horizon,
                      "a_valid_ap": runs[arm]["best_valid_ap"], "b_valid_ap": hist["best_valid_ap"],
                      "delta_ap": float(runs[arm]["best_valid_ap"]) - float(hist["best_valid_ap"])},
            )
            comparisons[f"{arm}_minus_FIXED1_hist_endpoint"] = paired(
                f"{arm} (endpoint) - FIXED1 historical same-horizon endpoint (Valid)",
                _ap_map(per_task_at(arm, horizon)),
                _ap_map(hist_pts),
                support_source,
                meta={"a_epoch": horizon, "b_epoch": horizon,
                      "a_valid_ap": runs[arm]["horizon_endpoint"]["valid_ap"],
                      "b_valid_ap": hist["endpoint_valid_ap"],
                      "delta_ap": float(runs[arm]["horizon_endpoint"]["valid_ap"]) - hist["endpoint_valid_ap"]},
            )

    curve_summaries: dict[str, Any] = {}
    for arm in FC.ARMS:
        if arm in runs:
            curve_summaries[arm] = curve_summary(curve_rows(arm, horizon))
    curve_summaries["FIXED1_hist"] = curve_summary(hist_curve_rows(horizon))

    paired_payload = {
        "round_id": FC.ROUND_ID,
        "horizon": int(horizon),
        "horizons_per_arm": {a: int(runs[a].get("horizon", 0)) for a in runs},
        "same_horizon": bool(same_horizon),
        "historical_reference": {
            "available": bool(reference_ok),
            "source": "direct_structural_computation_recurrence_v0 / results/runs/FIXED1_seed0",
            "identity": hist_ident,
            "same_horizon": hist,
            "not_used": "the 25-epoch best (0.275847) is never substituted for a prefix value",
        },
        "curve_summaries": curve_summaries,
        "comparisons": comparisons,
        "thresholds": {
            "practical_signal_ap": SIGNAL_AP,
            "tie_ap": TIE_AP,
            "high_support_min_positive_and_negative": HIGH_SUPPORT_MIN,
            "bootstrap_resamples": BOOTSTRAP_N,
        },
    }
    FC.write_json(out_dir / "paired_comparisons.json", paired_payload)

    mech = mechanism_summary()
    FC.write_json(out_dir / "mechanism.json", {
        "round_id": FC.ROUND_ID,
        "note": "light diagnostics only: D=A@B, scale ratios, gradients, frozen branch-off on the RSSC best checkpoint",
        **mech,
    })

    # ---- decision -----------------------------------------------------------
    def cdelta(key: str) -> float | None:
        c = comparisons.get(key)
        return None if not c else c["meta"].get("delta_ap")

    d_rssc_tied = cdelta("RSSC_minus_TIED_best")
    d_rssc_fixed1 = cdelta("RSSC_minus_FIXED1_hist_best") if reference_ok else None
    d_tied_fixed1 = cdelta("TIED_minus_FIXED1_hist_best") if reference_ok else None
    high_rssc_tied = (comparisons.get("RSSC_minus_TIED_best", {}).get("high_support") or {}).get("mean_delta_ap")
    late = curve_summaries
    late_ok = (
        late.get("RSSC") is not None and late.get("TIED") is not None
        and late["RSSC"]["last3_mean_valid_ap"] >= late["TIED"]["last3_mean_valid_ap"] - TIE_AP
    )
    flags: list[str] = []
    if not complete:
        flags.append("BUDGET_LIMITED")
    if not reference_ok:
        flags.append("REFERENCE_MISSING")
    if not same_horizon:
        flags.append("HORIZON_MISMATCH")

    if not complete or not same_horizon:
        label = "BUDGET_LIMITED" if not complete else "NO_CLEAR_ROLE_SEPARATION_SIGNAL"
    elif d_rssc_tied is None:
        label = "NO_CLEAR_ROLE_SEPARATION_SIGNAL"
    elif (
        d_rssc_fixed1 is not None
        and d_rssc_tied >= SIGNAL_AP
        and d_rssc_fixed1 >= SIGNAL_AP
        and (high_rssc_tied is None or high_rssc_tied >= -TIE_AP)
        and late_ok
    ):
        label = "ROLE_SEPARATION_PROMISING"
    elif (
        d_tied_fixed1 is not None
        and d_tied_fixed1 >= SIGNAL_AP
        and d_rssc_fixed1 is not None
        and d_rssc_fixed1 >= SIGNAL_AP
        and d_rssc_tied < SIGNAL_AP
    ):
        label = "GENERIC_PARAMETERIZATION_SIGNAL"
    elif d_rssc_tied >= SIGNAL_AP and (d_rssc_fixed1 is None or d_rssc_fixed1 < SIGNAL_AP):
        label = "ROLE_DIFFERENCE_WITHOUT_PERFORMANCE_ADVANCE"
    else:
        label = "NO_CLEAR_ROLE_SEPARATION_SIGNAL"

    gpu_hours_total = float(sum(float(runs[a].get("gpu_hours") or 0.0) for a in runs))
    summary = {
        "round_id": FC.ROUND_ID,
        "decision": label,
        "flags": flags,
        "answers": {
            "Q1_RSSC_beats_TIED": (None if d_rssc_tied is None else d_rssc_tied > 0),
            "Q1_delta_valid_ap": d_rssc_tied,
            "Q2_RSSC_beats_same_horizon_FIXED1": (None if d_rssc_fixed1 is None else d_rssc_fixed1 > 0),
            "Q2_delta_valid_ap": d_rssc_fixed1,
            "Q2_reference_available": bool(reference_ok),
            "Q2_TIED_vs_FIXED1": d_tied_fixed1,
            "Q3_train_fit_vs_validation_generalisation": {
                arm: {
                    "best_train_ap": (FC.read_json(FC.arm_out_dir(arm) / "best_checkpoint_eval.json")["train"]["ap"]
                                      if (FC.arm_out_dir(arm) / "best_checkpoint_eval.json").is_file() else None),
                    "best_valid_ap": runs[arm].get("best_valid_ap"),
                    "endpoint_train_ap": runs[arm].get("horizon_endpoint", {}).get("train_ap"),
                    "endpoint_valid_ap": runs[arm].get("horizon_endpoint", {}).get("valid_ap"),
                }
                for arm in FC.ARMS if arm in runs
            },
            "Q4_branch_participates": {
                arm: ((mech["arms"].get(arm) or {}).get("frozen_branch_zero") or {})
                for arm in FC.ARMS
            },
        },
        "curve_summaries": curve_summaries,
        "paired_deltas": {k: v["meta"] for k, v in comparisons.items()},
        "budget": {
            "gpu_hours_total_two_arms": gpu_hours_total,
            "wall_clock_seconds_sum": float(sum(float(runs[a].get("gpu_seconds") or 0.0) for a in runs)),
            "units": len(runs),
            "seeds": [FC.SEED],
            "dataset": "ogbg-molpcba",
        },
        "artifacts": {"epoch_curves_csv_sha256": curve_sha, "main_table_csv_sha256": main_sha},
        "created_utc": FC.utcnow(),
    }
    FC.write_json(out_dir / "summary.json", summary)
    FC.log(json.dumps({"decision": label, "flags": flags, "delta_RSSC_TIED": d_rssc_tied,
                       "delta_RSSC_FIXED1": d_rssc_fixed1, "delta_TIED_FIXED1": d_tied_fixed1}, indent=1))
    return 0 if complete else 3


if __name__ == "__main__":
    raise SystemExit(main())
