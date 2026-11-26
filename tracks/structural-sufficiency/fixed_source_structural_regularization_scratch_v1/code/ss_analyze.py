"""FSS-v1 analysis: curves, main table, paired task comparison, verdict.

Primary metric: official full-task macro Valid AP (official OGB evaluator).

Comparisons (per-task, paired over the 127 scorable Valid tasks):
  * ``STRUCT_DROP_SCRATCH`` vs the historical FIXED1 seed-0 25-epoch prefix
    (best vs best, endpoint vs endpoint, last-5 mean vs last-5 mean);
  * phase means (1-12, 13-20, 21-25) against the same historical phases.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import ss_common as R

BOOTSTRAP_B = 10000
BOOTSTRAP_SEED = 20261126
NEGLIGIBLE = 1e-12
ADVANCE_MIN_AP = 0.003
NO_REVERSE_TOL = -0.001
SIGNAL_MIN_AP = 0.001
HIGH_SUPPORT_POS = 20
HIGH_SUPPORT_NEG = 20
PHASES = ((1, 12), (13, 20), (21, 25))


def _f(x: Any) -> float | None:
    if x is None or x == "":
        return None
    v = float(x)
    return None if math.isnan(v) else v


def _mean(values: Sequence[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None


def bootstrap(delta: np.ndarray) -> dict[str, Any]:
    d = np.asarray(delta, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {"n_tasks": 0}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, d.size, size=(BOOTSTRAP_B, d.size))
    means = d[idx].mean(axis=1)
    return {
        "n_tasks": int(d.size),
        "mean_delta": float(d.mean()),
        "median_delta": float(np.median(d)),
        "sd_delta": float(d.std(ddof=1)) if d.size > 1 else 0.0,
        "bootstrap_b": BOOTSTRAP_B,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
        "fraction_positive": float((means > 0).mean()),
        "n_improved": int((d > NEGLIGIBLE).sum()),
        "n_degraded": int((d < -NEGLIGIBLE).sum()),
        "n_equal": int((np.abs(d) <= NEGLIGIBLE).sum()),
        "delta": [float(x) for x in d],
    }


def drop_top_abs(delta: np.ndarray, k: int = 3) -> dict[str, Any]:
    d = np.asarray(delta, dtype=np.float64)
    finite = np.isfinite(d)
    if not finite.any():
        return {"n_tasks": 0}
    order = np.argsort(-np.abs(np.where(finite, d, 0.0)))[:k]
    keep = np.ones(d.size, dtype=bool)
    keep[order] = False
    kept = d[keep & finite]
    return {
        "removed_tasks": [int(i) for i in order],
        "removed_deltas": [float(d[i]) for i in order],
        "n_tasks_kept": int(kept.size),
        "mean_delta_kept": float(kept.mean()) if kept.size else None,
        "sd_delta_kept": float(kept.std(ddof=1)) if kept.size > 1 else 0.0,
    }


def _per_task_map(rows: Sequence[Mapping[str, Any]], split: str, epoch: int) -> dict[int, float | None]:
    out: dict[int, float | None] = {}
    for row in rows:
        if row["split"] == split and int(row["epoch"]) == int(epoch):
            out[int(row["task"])] = _f(row["ap"])
    return out


def _vec(mapping: Mapping[int, float | None], tasks: Sequence[int]) -> np.ndarray:
    return np.asarray(
        [np.nan if mapping.get(t) is None else float(mapping[t]) for t in tasks], dtype=np.float64
    )


def _mean_over(mapping: Mapping[int, float | None], tasks: Sequence[int]) -> float | None:
    v = _vec(mapping, tasks)
    return None if not np.isfinite(v).any() else float(np.nanmean(v))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(R.ANALYSIS))
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dir = R.arm_run_dir()
    curve = R.read_csv_rows(run_dir / "train_curve.csv")
    per_task_rows = R.read_csv_rows(run_dir / "per_task_ap.csv")
    run = json.loads((run_dir / "run.json").read_text())
    initial = json.loads((run_dir / "initial_eval.json").read_text())
    best = json.loads((run_dir / "best_checkpoint_eval.json").read_text())
    historical = R.load_historical_fixed1()
    hist_curve = historical["curve"]

    if run["status"] != "COMPLETED":
        raise RuntimeError(f"run status {run['status']} ({run.get('stop_reason')}); refusing to analyse")

    epochs = [int(r["epoch"]) for r in curve]
    if epochs != list(range(1, R.EPOCHS + 1)):
        raise RuntimeError(f"curve epochs {epochs} != 1..{R.EPOCHS}")

    tasks = list(range(R.C.N_TASKS))
    positive = [int(x) for x in historical["positive"]]
    negative = [int(x) for x in historical["negative"]]
    high_support = [t for t in tasks if positive[t] >= HIGH_SUPPORT_POS and negative[t] >= HIGH_SUPPORT_NEG]

    valid_by_epoch = {e: _per_task_map(per_task_rows, "valid", e) for e in epochs}
    train_by_epoch = {e: _per_task_map(per_task_rows, "train", e) for e in epochs if e in R.TRAIN_EVAL_EPOCHS}
    our_best_valid = {int(k): v for k, v in enumerate(best["valid"]["per_task_ap"])}
    our_best_train = {int(k): v for k, v in enumerate(best["train"]["per_task_ap"])}
    our_endpoint_valid = valid_by_epoch[R.EPOCHS]
    our_endpoint_train = train_by_epoch.get(R.EPOCHS, {})
    our_last5_valid = {
        t: _mean([valid_by_epoch[e].get(t) for e in range(R.EPOCHS - 4, R.EPOCHS + 1)]) for t in tasks
    }
    hist_valid = historical["valid_per_task"]
    hist_train = historical["train_per_task"]
    hist_valid_curve: dict[int, dict[int, float | None]] = {}
    for row in R.read_csv_rows(R.HIST_PER_TASK):
        if row["split"] == "valid":
            hist_valid_curve.setdefault(int(row["epoch"]), {})[int(row["task"])] = _f(row["ap"])
    hist_last5_valid = {
        t: _mean([hist_valid_curve.get(e, {}).get(t) for e in range(R.EPOCHS - 4, R.EPOCHS + 1)])
        for t in tasks
    }
    scorable = [
        t for t in tasks
        if our_best_valid.get(t) is not None and hist_valid.get(t) is not None
        and np.isfinite(float(our_best_valid[t])) and np.isfinite(float(hist_valid[t]))
        and our_endpoint_valid.get(t) is not None
    ]

    # ---------------- curves --------------------------------------------
    curve_rows: list[dict[str, Any]] = []
    for row in curve:
        e = int(row["epoch"])
        hist = hist_curve[e]
        curve_rows.append(
            {
                "epoch": e,
                "global_step": int(row["global_step"]),
                "learning_rate": float(row["learning_rate"]),
                "fss_valid_ap": float(row["valid_ap"]),
                "fss_valid_bce": float(row["valid_bce"]),
                "fss_train_ap": _f(row.get("train_ap")),
                "fss_train_bce": _f(row.get("train_bce")),
                "fss_train_bce_step_mean": _f(row.get("train_bce_step_mean")),
                "fss_mask_dropped_fraction": _f(row.get("mask_dropped_fraction_mean")),
                "fss_is_best": row.get("is_best"),
                "fss_epoch_train_seconds": _f(row.get("epoch_train_seconds")),
                "fss_valid_seconds": _f(row.get("valid_seconds")),
                "fss_train_eval_seconds": _f(row.get("train_eval_seconds")),
                "fss_gpu_peak_allocated_bytes": int(row["gpu_peak_allocated_bytes"]),
                "hist_fixed1_valid_ap": hist["valid_ap"],
                "hist_fixed1_valid_bce": hist["valid_bce"],
                "hist_fixed1_train_ap": hist["train_ap"],
                "hist_fixed1_train_bce": hist["train_bce"],
                "delta_valid_ap": float(row["valid_ap"]) - hist["valid_ap"],
                "delta_valid_bce": float(row["valid_bce"]) - hist["valid_bce"],
            }
        )
    R.write_csv(out_dir / "epoch_curves.csv", list(curve_rows[0].keys()), curve_rows)

    # ---------------- headline numbers ----------------------------------
    best_epoch = int(run["best_epoch"])
    best_valid_ap = float(best["valid"]["ap"])
    endpoint_ap = float(curve[-1]["valid_ap"])
    endpoint_bce = float(curve[-1]["valid_bce"])
    last5_ap = _mean([float(r["valid_ap"]) for r in curve[-5:]])
    last5_bce = _mean([float(r["valid_bce"]) for r in curve[-5:]])
    initial_ap = float(initial["valid"]["ap"])
    hist_best = float(R.HIST["best_valid_ap"])
    hist_best_bce = float(R.HIST["best_valid_bce"])
    hist_endpoint = float(R.HIST["endpoint_valid_ap"])
    hist_endpoint_bce = float(R.HIST["endpoint_valid_bce"])
    hist_last5 = float(R.HIST["last5_mean_valid_ap"])

    high_support_values = {
        "initial": _mean_over({int(k): v for k, v in enumerate(initial["valid"]["per_task_ap"])}, high_support),
        "best": _mean_over(our_best_valid, high_support),
        "endpoint": _mean_over(our_endpoint_valid, high_support),
        "last5": _mean_over(our_last5_valid, high_support),
        "hist_best": _mean_over(hist_valid, high_support),
        "hist_endpoint": _mean_over(hist_valid, high_support),
        "hist_last5": _mean_over(hist_last5_valid, high_support),
    }

    # ---------------- paired comparisons --------------------------------
    def compare(name: str, a: Mapping[int, float | None], b: Mapping[int, float | None],
                subset: Sequence[int], *, official_a: float, official_b: float, note: str = "") -> dict[str, Any]:
        d = _vec(b, subset) - _vec(a, subset)
        return {
            "comparison": name,
            "direction": "STRUCT_DROP_SCRATCH - FIXED1",
            "n_tasks": int(np.isfinite(d).sum()),
            "official_ap_fixed1": float(official_a),
            "official_ap_struct_drop_scratch": float(official_b),
            "official_ap_delta": float(official_b) - float(official_a),
            "mean_per_task_ap_fixed1": _mean_over(a, subset),
            "mean_per_task_ap_struct_drop_scratch": _mean_over(b, subset),
            "note": note,
            **bootstrap(d),
            "sensitivity_drop_top3": drop_top_abs(d, 3),
        }

    paired = {
        "schema": "fsrp-scratch-paired-v1",
        "round_id": R.ROUND_ID,
        "unit": "official OGB task (paired, 127 scorable Valid tasks)",
        "primary_best": compare(
            "best checkpoint", hist_valid, our_best_valid, scorable,
            official_a=hist_best, official_b=best_valid_ap,
        ),
        "endpoint": compare(
            "endpoint (epoch 25)", hist_valid, our_endpoint_valid, scorable,
            official_a=hist_endpoint, official_b=endpoint_ap,
        ),
        "last5": compare(
            "last-5 mean (epochs 21-25)", hist_last5_valid, our_last5_valid, scorable,
            official_a=hist_last5, official_b=last5_ap or float("nan"),
        ),
        "high_support": {
            "tasks": [int(t) for t in high_support],
            "n_tasks": len(high_support),
            "decision_rule": f"Valid positive >= {HIGH_SUPPORT_POS} and negative >= {HIGH_SUPPORT_NEG}",
            "values": high_support_values,
            "delta_best": (
                None if high_support_values["best"] is None or high_support_values["hist_best"] is None
                else high_support_values["best"] - high_support_values["hist_best"]
            ),
            "delta_endpoint": (
                None if high_support_values["endpoint"] is None or high_support_values["hist_endpoint"] is None
                else high_support_values["endpoint"] - high_support_values["hist_endpoint"]
            ),
            "delta_last5": (
                None if high_support_values["last5"] is None or high_support_values["hist_last5"] is None
                else high_support_values["last5"] - high_support_values["hist_last5"]
            ),
        },
    }

    # ---------------- phases --------------------------------------------
    phase_rows = []
    phases: dict[str, Any] = {}
    for lo, hi in PHASES:
        fss = [float(r["valid_ap"]) for r in curve if lo <= int(r["epoch"]) <= hi]
        his = [hist_curve[e]["valid_ap"] for e in range(lo, hi + 1)]
        fss_bce = [float(r["valid_bce"]) for r in curve if lo <= int(r["epoch"]) <= hi]
        his_bce = [hist_curve[e]["valid_bce"] for e in range(lo, hi + 1)]
        key = f"epochs_{lo}_{hi}"
        phases[key] = {
            "epochs": [lo, hi],
            "n_epochs": hi - lo + 1,
            "mean_valid_ap_struct_drop_scratch": float(np.mean(fss)),
            "mean_valid_ap_fixed1": float(np.mean(his)),
            "delta_mean_valid_ap": float(np.mean(fss) - np.mean(his)),
            "min_valid_ap_struct_drop_scratch": float(np.min(fss)),
            "max_valid_ap_struct_drop_scratch": float(np.max(fss)),
            "mean_valid_bce_struct_drop_scratch": float(np.mean(fss_bce)),
            "mean_valid_bce_fixed1": float(np.mean(his_bce)),
            "delta_mean_valid_bce": float(np.mean(fss_bce) - np.mean(his_bce)),
            "n_epochs_better": int(sum(1 for a, b in zip(fss, his) if a > b)),
        }
        phase_rows.append(
            {
                "phase": key,
                "epochs": f"{lo}-{hi}",
                "n_epochs": hi - lo + 1,
                **{k: v for k, v in phases[key].items() if k not in ("epochs",)},
            }
        )
    R.write_csv(out_dir / "phase_table.csv", list(phase_rows[0].keys()), phase_rows)

    # learning-speed landmarks (first epoch reaching a Valid AP level)
    landmarks = {}
    for level in (0.15, 0.20, 0.25, 0.27):
        fss_first = next((int(r["epoch"]) for r in curve if float(r["valid_ap"]) >= level), None)
        his_first = next((e for e in range(1, R.EPOCHS + 1) if hist_curve[e]["valid_ap"] >= level), None)
        landmarks[str(level)] = {
            "struct_drop_scratch_first_epoch": fss_first,
            "fixed1_first_epoch": his_first,
            "delta_epochs": None if fss_first is None or his_first is None else fss_first - his_first,
        }

    # ---------------- Train-Valid gap -----------------------------------
    def gap(train_map: Mapping[int, float | None], valid_map: Mapping[int, float | None]) -> dict[str, Any]:
        common = [t for t in tasks if train_map.get(t) is not None and valid_map.get(t) is not None]
        tv = _vec(train_map, common)
        vv = _vec(valid_map, common)
        finite = np.isfinite(tv) & np.isfinite(vv)
        return {
            "n_common_tasks": int(finite.sum()),
            "mean_train_ap": float(tv[finite].mean()) if finite.any() else None,
            "mean_valid_ap": float(vv[finite].mean()) if finite.any() else None,
            "train_minus_valid_mean_ap": float((tv[finite] - vv[finite]).mean()) if finite.any() else None,
        }

    gaps = {
        "struct_drop_scratch_best": {
            **gap(our_best_train, our_best_valid),
            "official_train_ap": float(best["train"]["ap"]),
            "official_valid_ap": float(best["valid"]["ap"]),
            "official_train_minus_valid_ap": float(best["train"]["ap"]) - float(best["valid"]["ap"]),
            "train_bce": float(best["train"]["bce"]),
            "valid_bce": float(best["valid"]["bce"]),
        },
        "fixed1_best": {
            **gap(hist_train, hist_valid),
            "official_train_ap": float(R.HIST["best_train_ap"]),
            "official_valid_ap": float(R.HIST["best_valid_ap"]),
            "official_train_minus_valid_ap": float(R.HIST["best_train_ap"]) - float(R.HIST["best_valid_ap"]),
            "train_bce": float(R.HIST["best_train_bce"]),
            "valid_bce": float(R.HIST["best_valid_bce"]),
        },
    }

    paired["train_valid_gap"] = gaps
    paired["phases"] = phases
    paired["learning_speed_landmarks"] = landmarks
    paired["train_fit"] = {
        "epoch": list(R.TRAIN_EVAL_EPOCHS),
        "struct_drop_scratch_train_ap": [
            _f(next(r for r in curve if int(r["epoch"]) == e).get("train_ap")) for e in R.TRAIN_EVAL_EPOCHS
        ],
        "fixed1_train_ap": [hist_curve[e]["train_ap"] for e in R.TRAIN_EVAL_EPOCHS],
        "struct_drop_scratch_train_bce": [
            _f(next(r for r in curve if int(r["epoch"]) == e).get("train_bce")) for e in R.TRAIN_EVAL_EPOCHS
        ],
        "fixed1_train_bce": [hist_curve[e]["train_bce"] for e in R.TRAIN_EVAL_EPOCHS],
        "struct_drop_scratch_best_train_ap": float(best["train"]["ap"]),
        "fixed1_best_train_ap": float(R.HIST["best_train_ap"]),
        "struct_drop_scratch_best_train_bce": float(best["train"]["bce"]),
        "fixed1_best_train_bce": float(R.HIST["best_train_bce"]),
        "delta_train_ap_best": float(best["train"]["ap"]) - float(R.HIST["best_train_ap"]),
        "delta_train_bce_best": float(best["train"]["bce"]) - float(R.HIST["best_train_bce"]),
    }
    paired["resource"] = {
        "gpu_hours_formal_run": float(run["gpu_hours"]),
        "gpu_seconds_formal_run": float(run["gpu_seconds"]),
        "train_seconds": float(run["train_seconds"]),
        "eval_seconds": float(run["eval_seconds"]),
        "gpu_peak_allocated_bytes": int(run["gpu_peak_allocated_bytes"]),
        "mean_epoch_train_seconds": float(np.mean([float(r["epoch_train_seconds"]) for r in curve])),
        "mean_valid_seconds": float(np.mean([float(r["valid_seconds"]) for r in curve])),
        "gpu_name": run["environment"].get("cuda_device_name"),
        "hostname": run["environment"].get("hostname"),
        "git_commit": run["git_commit"],
        "reference_fixed1_gpu_hours": R.HIST["gpu_hours_full_eval_every_epoch"],
        "reference_note": "the historical FIXED1 run scored full Train+Valid+Test every epoch; "
                          "this round scores full Valid every epoch and full Train at 12/20/25, "
                          "so GPU hours are not directly comparable as a cost of the same protocol",
    }
    paired["mask"] = {
        "q": float(R.Q),
        "mask_seed": int(R.MASK_SEED),
        "mean_dropped_fraction": run["structural_dropout"]["mean_epoch_dropped_fraction"],
        "per_epoch_dropped_fraction": {
            int(s["epoch"]): s.get("dropped_fraction_mean") for s in run["structural_dropout"]["per_epoch"]
        },
        "draws_per_epoch": [int(s["n"]) for s in run["structural_dropout"]["per_epoch"]],
        "same_tensor_both_blocks_all_epochs": bool(
            all(s.get("same_tensor_both_blocks") for s in run["structural_dropout"]["per_epoch"])
        ),
        "directions_shared_all_epochs": bool(
            all(s.get("directions_shared") for s in run["structural_dropout"]["per_epoch"])
        ),
        "mask_draws_total": int(run["structural_dropout"]["mask_draws_total"]),
        "mask_draws_in_eval": 0,
    }
    paired["common_start"] = {
        "struct_drop_scratch_initial_valid_ap": initial_ap,
        "struct_drop_scratch_initial_valid_bce": float(initial["valid"]["bce"]),
        "fixed1_epoch0_valid_ap": None,
        "note": "the initial model is bitwise the FIXED1 seed-0 constructor state, so the "
                "epoch-0 Valid AP is a property of that initialisation, not of this round's training",
        "initial_state_sha256": initial["initial_state_sha256"],
        "expected_initial_state_sha256": R.EXPECTED_INITIAL_STATE_SHA256,
        "initial_state_matches_fixed1": initial["initial_state_sha256"] == R.EXPECTED_INITIAL_STATE_SHA256,
    }
    paired["historical_reference"] = {
        "round_id": R.HIST["round_id"],
        "arm": R.HIST["arm"],
        "seed": R.HIST["seed"],
        "horizon": R.HIST["horizon"],
        "best_epoch": R.HIST["best_epoch"],
        "best_valid_ap": hist_best,
        "endpoint_valid_ap": hist_endpoint,
        "last5_mean_valid_ap": hist_last5,
        "best_train_ap": float(R.HIST["best_train_ap"]),
        "high_support_valid_ap": high_support_values["hist_best"],
        "verification": historical["verification"],
        "sources": historical["sources"],
        "retrained_in_this_round": False,
    }
    R.write_json(out_dir / "paired_comparisons.json", paired)

    # ---------------- main table ----------------------------------------
    def delta(a: float | None, b: float | None) -> float | None:
        return None if a is None or b is None else float(a) - float(b)

    main_rows = [
        {
            "arm": "FIXED1",
            "role": "historical reference (read-only, not retrained)",
            "best_epoch": int(R.HIST["best_epoch"]),
            "best_valid_ap": hist_best,
            "best_valid_bce": hist_best_bce,
            "endpoint_valid_ap": hist_endpoint,
            "endpoint_valid_bce": hist_endpoint_bce,
            "last5_mean_valid_ap": hist_last5,
            "train_ap_e12": R.HIST["train_ap_e12"],
            "train_ap_e20": R.HIST["train_ap_e20"],
            "train_ap_e25": R.HIST["train_ap_e25"],
            "train_bce_e12": R.HIST["train_bce_e12"],
            "train_bce_e20": R.HIST["train_bce_e20"],
            "train_bce_e25": R.HIST["train_bce_e25"],
            "best_train_ap": float(R.HIST["best_train_ap"]),
            "best_train_bce": float(R.HIST["best_train_bce"]),
            "train_minus_valid_ap_best_ckpt": gaps["fixed1_best"]["train_minus_valid_mean_ap"],
            "high_support_valid_ap_best": high_support_values["hist_best"],
            "high_support_valid_ap_endpoint": high_support_values["hist_endpoint"],
            "high_support_valid_ap_last5": high_support_values["hist_last5"],
            "n_tasks_scored_valid": 127,
            "parameters": R.PARAMS_TOTAL,
            "mask_dropped_fraction_mean": None,
            "gpu_hours": R.HIST["gpu_hours_full_eval_every_epoch"],
        },
        {
            "arm": R.ARM,
            "role": "structural-unit message dropout from training step 0 (q=0.10)",
            "best_epoch": best_epoch,
            "best_valid_ap": best_valid_ap,
            "best_valid_bce": float(best["valid"]["bce"]),
            "endpoint_valid_ap": endpoint_ap,
            "endpoint_valid_bce": endpoint_bce,
            "last5_mean_valid_ap": last5_ap,
            "train_ap_e12": _f(next(r for r in curve if int(r["epoch"]) == 12).get("train_ap")),
            "train_ap_e20": _f(next(r for r in curve if int(r["epoch"]) == 20).get("train_ap")),
            "train_ap_e25": _f(next(r for r in curve if int(r["epoch"]) == 25).get("train_ap")),
            "train_bce_e12": _f(next(r for r in curve if int(r["epoch"]) == 12).get("train_bce")),
            "train_bce_e20": _f(next(r for r in curve if int(r["epoch"]) == 20).get("train_bce")),
            "train_bce_e25": _f(next(r for r in curve if int(r["epoch"]) == 25).get("train_bce")),
            "best_train_ap": float(best["train"]["ap"]),
            "best_train_bce": float(best["train"]["bce"]),
            "train_minus_valid_ap_best_ckpt": gaps["struct_drop_scratch_best"]["train_minus_valid_mean_ap"],
            "high_support_valid_ap_best": high_support_values["best"],
            "high_support_valid_ap_endpoint": high_support_values["endpoint"],
            "high_support_valid_ap_last5": high_support_values["last5"],
            "n_tasks_scored_valid": int(best["valid"]["n_tasks_scored"]),
            "parameters": int(run["parameters"]),
            "mask_dropped_fraction_mean": run["structural_dropout"]["mean_epoch_dropped_fraction"],
            "gpu_hours": float(run["gpu_hours"]),
        },
    ]
    for row in main_rows:
        row["no_mask_reference"] = "FIXED1" if row["arm"] == R.ARM else None
    fields = list(main_rows[0].keys())
    delta_row = {
        "arm": "DELTA (STRUCT_DROP_SCRATCH - FIXED1)",
        "role": "difference",
        "best_epoch": best_epoch - int(R.HIST["best_epoch"]),
    }
    for key in fields:
        if key in ("arm", "role", "best_epoch", "no_mask_reference"):
            continue
        a = main_rows[0].get(key)
        b = main_rows[1].get(key)
        delta_row[key] = delta(b, a) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
    main_rows.append(delta_row)
    R.write_csv(out_dir / "main_table.csv", fields, main_rows)

    # ---------------- checkpoint manifest -------------------------------
    manifest = {
        "schema": R.SCHEMA_MANIFEST,
        "round_id": R.ROUND_ID,
        "arm": R.ARM,
        "seed": R.SEED,
        "parameters": int(run["parameters"]),
        "initial_state_sha256": run["initial_state_sha256"],
        "checkpoints": [
            {
                "role": "best_valid_ap",
                "epoch": best_epoch,
                "global_step": int(run["best_checkpoint_eval"]["global_step"]),
                "valid_ap": best_valid_ap,
                "path": run["best_checkpoint"],
                "sha256": run["best_checkpoint_sha256"],
            },
            {
                "role": "horizon_endpoint",
                "epoch": R.EPOCHS,
                "global_step": int(run["global_steps_run"]),
                "valid_ap": endpoint_ap,
                "path": run["horizon_checkpoint"],
                "sha256": run["horizon_checkpoint_sha256"],
            },
            {
                "role": "last_epoch_resumable",
                "epoch": R.EPOCHS,
                "global_step": int(run["global_steps_run"]),
                "valid_ap": endpoint_ap,
                "path": run["last_checkpoint"],
                "sha256": run["last_checkpoint_sha256"],
            },
        ],
        "mask_rng_fingerprint_end": run["structural_dropout"]["mask_rng_fingerprint_end"],
        "mask_rng_state_saved_in_every_checkpoint": True,
        "optimizer_state_saved_in_every_checkpoint": True,
        "test_scored": False,
    }
    R.write_json(out_dir / "checkpoint_manifest.json", manifest)

    # ---------------- verdict -------------------------------------------
    d_best = best_valid_ap - hist_best
    d_endpoint = endpoint_ap - hist_endpoint
    d_last5 = None if last5_ap is None else last5_ap - hist_last5
    d_hs_best = paired["high_support"]["delta_best"]
    d_hs_endpoint = paired["high_support"]["delta_endpoint"]
    d_hs_last5 = paired["high_support"]["delta_last5"]
    # same-stage fit comparison (epoch 25 vs epoch 25) is the honest one; the
    # best-checkpoint Train AP is reported separately because this run's best is
    # at epoch 19 while the historical best is at epoch 25.
    d_train_e25 = float(_f(next(r for r in curve if int(r["epoch"]) == 25).get("train_ap"))) - float(
        R.HIST["train_ap_e25"]
    )
    d_train_best = float(best["train"]["ap"]) - float(R.HIST["best_train_ap"])
    phase_late = phases["epochs_21_25"]["delta_mean_valid_ap"]
    phase_early = phases["epochs_1_12"]["delta_mean_valid_ap"]
    phase_mid = phases["epochs_13_20"]["delta_mean_valid_ap"]
    n_better = sum(1 for row in curve_rows if row["delta_valid_ap"] > 0)
    n_worse = sum(1 for row in curve_rows if row["delta_valid_ap"] < 0)

    no_reverse = all(
        v is None or v >= NO_REVERSE_TOL for v in (d_endpoint, d_last5, d_hs_best, d_hs_endpoint, d_hs_last5)
    )
    # ordered, pre-registered decision rule (PROTOCOL.md section "Decision labels")
    sustained_signal = bool(
        (phase_late is not None and phase_late >= SIGNAL_MIN_AP)
        or (d_hs_best is not None and d_hs_best >= SIGNAL_MIN_AP)
        or (d_hs_last5 is not None and d_hs_last5 >= SIGNAL_MIN_AP)
    )
    if d_best >= ADVANCE_MIN_AP and no_reverse:
        verdict = "SCRATCH_STRUCTURAL_REGULARIZATION_ADVANCE"
    elif sustained_signal:
        verdict = "WEAK_REGULARIZATION_SIGNAL"
    elif d_train_e25 < 0 and d_best < ADVANCE_MIN_AP:
        verdict = "FIT_REDUCTION_WITHOUT_GAIN"
    elif d_best <= 0:
        verdict = "NO_CLEAR_GAIN"
    else:
        verdict = "NO_CLEAR_GAIN"

    summary = {
        "schema": R.SCHEMA_SUMMARY,
        "round_id": R.ROUND_ID,
        "protocol_id": R.PROTOCOL_ID,
        "arm": R.ARM,
        "seed": R.SEED,
        "epochs": R.EPOCHS,
        "verdict": verdict,
        "verdict_rules": {
            "advance_min_best_valid_ap_delta": ADVANCE_MIN_AP,
            "no_reverse_tolerance": NO_REVERSE_TOL,
            "signal_min_ap": SIGNAL_MIN_AP,
            "order": [
                "ADVANCE if best >= +0.003 and endpoint/last-5/high-support >= -0.001",
                "WEAK_REGULARIZATION_SIGNAL if late-phase (21-25) mean or high-support delta >= +0.001",
                "FIT_REDUCTION_WITHOUT_GAIN if same-stage (epoch-25) Train AP is lower and best Valid AP < +0.003",
                "NO_CLEAR_GAIN otherwise",
            ],
            "applied": {
                "delta_best_valid_ap": d_best,
                "delta_endpoint_valid_ap": d_endpoint,
                "delta_last5_mean_valid_ap": d_last5,
                "delta_high_support_best": d_hs_best,
                "delta_high_support_endpoint": d_hs_endpoint,
                "delta_high_support_last5": d_hs_last5,
                "no_reverse_condition_satisfied": no_reverse,
                "delta_train_ap_best_checkpoint": d_train_best,
                "delta_train_ap_epoch25_same_stage": d_train_e25,
                "n_epochs_valid_better": int(n_better),
                "n_epochs_valid_worse": int(n_worse),
                "delta_phase_means": {"1-12": phase_early, "13-20": phase_mid, "21-25": phase_late},
            },
        },
        "headline": {
            "struct_drop_scratch_best_epoch": best_epoch,
            "struct_drop_scratch_best_valid_ap": best_valid_ap,
            "struct_drop_scratch_endpoint_valid_ap": endpoint_ap,
            "struct_drop_scratch_last5_mean_valid_ap": last5_ap,
            "fixed1_best_valid_ap": hist_best,
            "fixed1_endpoint_valid_ap": hist_endpoint,
            "fixed1_last5_mean_valid_ap": hist_last5,
            "delta_best_valid_ap": d_best,
            "delta_endpoint_valid_ap": d_endpoint,
            "delta_last5_mean_valid_ap": d_last5,
        },
        "paired": {
            "best": {k: paired["primary_best"][k] for k in
                     ("n_tasks", "mean_delta", "ci95_low", "ci95_high", "fraction_positive",
                      "n_improved", "n_degraded", "n_equal")},
            "endpoint": {k: paired["endpoint"][k] for k in
                          ("n_tasks", "mean_delta", "ci95_low", "ci95_high", "fraction_positive",
                           "n_improved", "n_degraded", "n_equal")},
            "last5": {k: paired["last5"][k] for k in
                       ("n_tasks", "mean_delta", "ci95_low", "ci95_high", "fraction_positive",
                        "n_improved", "n_degraded", "n_equal")},
            "sensitivity_drop_top3": {
                "best": paired["primary_best"]["sensitivity_drop_top3"]["mean_delta_kept"],
                "endpoint": paired["endpoint"]["sensitivity_drop_top3"]["mean_delta_kept"],
                "last5": paired["last5"]["sensitivity_drop_top3"]["mean_delta_kept"],
            },
        },
        "phases": phases,
        "learning_speed_landmarks": landmarks,
        "train_fit": {**paired["train_fit"], "delta_train_ap_epoch25_same_stage": d_train_e25,
                      "best_epoch_note": "this run's best checkpoint is epoch "
                                         f"{best_epoch}; the historical best is epoch {R.HIST['best_epoch']}, "
                                         "so best-checkpoint Train metrics are not like-for-like"},
        "curve_positivity": {"n_epochs_better": int(n_better), "n_epochs_worse": int(n_worse),
                             "n_epochs": len(curve_rows)},
        "train_valid_gap": gaps,
        "mask": paired["mask"],
        "resource": paired["resource"],
        "historical_reference": paired["historical_reference"],
        "created_utc": R.utcnow(),
        "environment": run["environment"],
        "git_commit": run["git_commit"],
        "test_scored": False,
        "test_used_for_selection": False,
    }
    R.write_json(out_dir / "summary.json", summary)

    # ---------------- markdown tables -----------------------------------
    def fmt(v: Any, nd: int = 6) -> str:
        if v is None:
            return "–"
        if isinstance(v, bool):
            return "yes" if v else "no"
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        try:
            return f"{float(v):.{nd}f}"
        except Exception:  # noqa: BLE001
            return str(v)

    lines = [
        "# FSRP-SCRATCH v1 — generated tables",
        "",
        "## 1. Complete Valid curve (official macro AP, full Valid split)",
        "",
        "| epoch | FSS Valid AP | FIXED1 Valid AP | Δ AP | FSS Valid BCE | FIXED1 Valid BCE | FSS Train AP | FIXED1 Train AP | mask dropped |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in curve_rows:
        lines.append(
            "| {e} | {a} | {b} | {d} | {c} | {f} | {g} | {h} | {m} |".format(
                e=row["epoch"], a=fmt(row["fss_valid_ap"]), b=fmt(row["hist_fixed1_valid_ap"]),
                d=fmt(row["delta_valid_ap"]), c=fmt(row["fss_valid_bce"], 7),
                f=fmt(row["hist_fixed1_valid_bce"], 7), g=fmt(row["fss_train_ap"]),
                h=fmt(row["hist_fixed1_train_ap"]), m=fmt(row["fss_mask_dropped_fraction"], 5),
            )
        )
    lines += ["", "## 2. Phase means", "", "| phase | FSS mean AP | FIXED1 mean AP | Δ | epochs better |", "|---|---:|---:|---:|---:|"]
    for row in phase_rows:
        lines.append(
            f"| {row['epochs']} | {fmt(row['mean_valid_ap_struct_drop_scratch'])} | "
            f"{fmt(row['mean_valid_ap_fixed1'])} | {fmt(row['delta_mean_valid_ap'])} | "
            f"{row['n_epochs_better']}/{row['n_epochs']} |"
        )
    lines += ["", "## 3. Paired task comparisons (127 tasks)", "",
              "| comparison | mean Δ | 95% CI | P(Δ>0) | + / − | drop-top-3 |", "|---|---:|---|---:|---:|---:|"]
    for key in ("primary_best", "endpoint", "last5"):
        c = paired[key]
        lines.append(
            f"| {c['comparison']} | {fmt(c['mean_delta'])} | [{fmt(c['ci95_low'])}, {fmt(c['ci95_high'])}] | "
            f"{fmt(c['fraction_positive'], 4)} | {c['n_improved']} / {c['n_degraded']} | "
            f"{fmt(c['sensitivity_drop_top3']['mean_delta_kept'])} |"
        )
    lines += ["", "## 4. Main table", "",
              "| metric | FIXED1 | STRUCT_DROP_SCRATCH | Δ |", "|---|---:|---:|---:|"]
    keys = [
        ("best Valid AP", "best_valid_ap"), ("best Valid BCE", "best_valid_bce"),
        ("endpoint (25) Valid AP", "endpoint_valid_ap"), ("last-5 mean Valid AP", "last5_mean_valid_ap"),
        ("Train AP @12", "train_ap_e12"), ("Train AP @20", "train_ap_e20"), ("Train AP @25", "train_ap_e25"),
        ("Train BCE @12", "train_bce_e12"), ("Train BCE @20", "train_bce_e20"), ("Train BCE @25", "train_bce_e25"),
        ("best-checkpoint Train AP", "best_train_ap"), ("best-checkpoint Train BCE", "best_train_bce"),
        ("high-support Valid AP (best)", "high_support_valid_ap_best"),
        ("high-support Valid AP (endpoint)", "high_support_valid_ap_endpoint"),
        ("Train−Valid gap (best ckpt)", "train_minus_valid_ap_best_ckpt"),
        ("mask dropped fraction", "mask_dropped_fraction_mean"),
        ("GPU hours", "gpu_hours"),
    ]
    for label, key in keys:
        lines.append(
            f"| {label} | {fmt(main_rows[0].get(key))} | {fmt(main_rows[1].get(key))} | "
            f"{fmt(delta_row.get(key))} |"
        )
    lines += ["", f"**Verdict: `{verdict}`**", ""]
    (out_dir / "report_tables.md").write_text("\n".join(lines), encoding="utf-8")

    R.C.log(f"analysis done verdict={verdict} d_best={d_best:+.6f} d_endpoint={d_endpoint:+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
