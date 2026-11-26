"""FSRP v0 analysis: main table, paired task comparison, verdict label.

Comparison identities:
  * ``STRUCT_DROP - BASE_CONT`` (primary, same continuation budget);
  * ``best - epoch-25 common start`` (extension gain);
  * ``BASE_CONT best - epoch-25 start`` (plain continuation gain).

Task-level bootstrap resamples the 127 OGB tasks (paired), never molecules.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import sr_common as R

ROUND = R.ROUND
ANALYSIS = R.ANALYSIS
RUNS = R.RUNS
BOOTSTRAP_B = 10000
BOOTSTRAP_SEED = 20261126
NEGLIGIBLE = 1e-12
PROMISING_MIN_AP = 0.003


def _rows(path: Path) -> list[dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fields})


def _f(x: Any) -> float | None:
    if x is None or x == "":
        return None
    v = float(x)
    return None if math.isnan(v) else v


def _task_ap_map(rows: Sequence[Mapping[str, Any]], split: str) -> dict[int, dict[int, float | None]]:
    out: dict[int, dict[int, float | None]] = {}
    for row in rows:
        if row["split"] != split:
            continue
        epoch = int(row["epoch"])
        out.setdefault(epoch, {})[int(row["task"])] = _f(row["ap"])
    return out


def _vec(mapping: Mapping[int, float | None], tasks: Sequence[int]) -> np.ndarray:
    return np.asarray([np.nan if mapping.get(t) is None else float(mapping[t]) for t in tasks], dtype=np.float64)


def _mean_over(mapping: Mapping[int, float | None], tasks: Sequence[int]) -> float | None:
    v = _vec(mapping, tasks)
    return None if not np.isfinite(v).any() else float(np.nanmean(v))


def bootstrap(delta: np.ndarray, *, seed: int = BOOTSTRAP_SEED, n_boot: int = BOOTSTRAP_B) -> dict[str, Any]:
    d = np.asarray(delta, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {"n_tasks": 0}
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, d.size, size=(int(n_boot), d.size))
    means = d[idx].mean(axis=1)
    return {
        "n_tasks": int(d.size),
        "mean_delta": float(d.mean()),
        "median_delta": float(np.median(d)),
        "sd_delta": float(d.std(ddof=1)) if d.size > 1 else 0.0,
        "bootstrap_b": int(n_boot),
        "bootstrap_seed": int(seed),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
        "fraction_positive": float((means > 0).mean()),
        "fraction_nonnegative": float((means >= 0).mean()),
        "n_improved": int((d > NEGLIGIBLE).sum()),
        "n_degraded": int((d < -NEGLIGIBLE).sum()),
        "n_equal": int((np.abs(d) <= NEGLIGIBLE).sum()),
        "delta": [None if not np.isfinite(x) else float(x) for x in d],
    }


def drop_top_abs(delta: np.ndarray, k: int = 3) -> dict[str, Any]:
    d = np.asarray(delta, dtype=np.float64)
    finite = np.isfinite(d)
    n = int(finite.sum())
    if n == 0:
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


def load_run(arm: str) -> dict[str, Any]:
    run_dir = RUNS / f"{arm}_seed{R.SEED}"
    curve = _rows(run_dir / "train_curve.csv")
    per_task_rows = _rows(run_dir / "per_task_ap.csv")
    baseline = json.loads((run_dir / "baseline_epoch25.json").read_text())
    best = json.loads((run_dir / "best_checkpoint_eval.json").read_text())
    run = json.loads((run_dir / "run.json").read_text())
    valid = _task_ap_map(per_task_rows, "valid")
    train = _task_ap_map(per_task_rows, "train")
    return {
        "arm": arm,
        "dir": run_dir,
        "curve": curve,
        "per_task_rows": per_task_rows,
        "baseline": baseline,
        "best": best,
        "run": run,
        "valid_by_epoch": valid,
        "train_by_epoch": train,
        "baseline_valid": {int(k): v for k, v in enumerate(baseline["valid"]["per_task_ap"])},
        "baseline_train": {int(k): v for k, v in enumerate(baseline["train"]["per_task_ap"])},
        "best_valid": {int(k): v for k, v in enumerate(best["valid"]["per_task_ap"])},
        "best_train": {int(k): v for k, v in enumerate(best["train"]["per_task_ap"])},
        "endpoint_valid": valid.get(R.END_EPOCH, {}),
        "endpoint_train": train.get(R.END_EPOCH, {}),
    }


def curve_value(rows: Sequence[Mapping[str, Any]], epoch: int, key: str) -> float | None:
    for row in rows:
        if int(row["epoch"]) == int(epoch):
            return _f(row.get(key))
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(ANALYSIS))
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = {arm: load_run(arm) for arm in R.ARMS}
    base = runs["BASE_CONT"]
    drop = runs["STRUCT_DROP"]

    tasks = list(range(R.C.N_TASKS))
    pos = base["baseline"]["valid"]["positive"]
    neg = base["baseline"]["valid"]["negative"]
    high_support = [t for t in tasks if pos[t] >= 20 and neg[t] >= 20]
    scorable_valid = [t for t in tasks if base["baseline_valid"].get(t) is not None]
    scorable_train = [t for t in tasks if base["baseline_train"].get(t) is not None]

    # ---- consistency of the common start --------------------------------
    start = {
        "base_valid_ap": float(base["baseline"]["valid"]["ap"]),
        "drop_valid_ap": float(drop["baseline"]["valid"]["ap"]),
        "base_valid_bce": float(base["baseline"]["valid"]["bce"]),
        "drop_valid_bce": float(drop["baseline"]["valid"]["bce"]),
        "base_train_ap": float(base["baseline"]["train"]["ap"]),
        "drop_train_ap": float(drop["baseline"]["train"]["ap"]),
        "base_train_bce": float(base["baseline"]["train"]["bce"]),
        "drop_train_bce": float(drop["baseline"]["train"]["bce"]),
        "recorded_expected_valid_ap": R.EXPECTED_BASE_VALID_AP,
        "max_abs_valid_ap_difference": abs(float(base["baseline"]["valid"]["ap"]) - float(drop["baseline"]["valid"]["ap"])),
        "max_abs_valid_bce_difference": abs(float(base["baseline"]["valid"]["bce"]) - float(drop["baseline"]["valid"]["bce"])),
        "max_abs_train_ap_difference": abs(float(base["baseline"]["train"]["ap"]) - float(drop["baseline"]["train"]["ap"])),
        "identical_initial_state": base["baseline"]["model_state_fingerprint"] == drop["baseline"]["model_state_fingerprint"],
        "identical_initial_optimizer": base["baseline"]["optimizer_fingerprint"] == drop["baseline"]["optimizer_fingerprint"],
        "identical_initial_rng": base["baseline"]["rng_state_hashes"]["combined_sha256"]
        == drop["baseline"]["rng_state_hashes"]["combined_sha256"],
    }

    # ---- curves ---------------------------------------------------------
    curve_rows = []
    for epoch in (R.START_EPOCH, *R.TRAIN_EPOCHS):
        row: dict[str, Any] = {"epoch": epoch}
        for arm, rr in runs.items():
            if epoch == R.START_EPOCH:
                row[f"{arm}_lr"] = rr["baseline"]["lr_at_restore"]
                row[f"{arm}_valid_ap"] = rr["baseline"]["valid"]["ap"]
                row[f"{arm}_valid_bce"] = rr["baseline"]["valid"]["bce"]
                row[f"{arm}_train_ap"] = rr["baseline"]["train"]["ap"]
                row[f"{arm}_train_bce"] = rr["baseline"]["train"]["bce"]
                row[f"{arm}_mask_dropped_fraction"] = None
            else:
                row[f"{arm}_lr"] = curve_value(rr["curve"], epoch, "learning_rate")
                row[f"{arm}_valid_ap"] = curve_value(rr["curve"], epoch, "valid_ap")
                row[f"{arm}_valid_bce"] = curve_value(rr["curve"], epoch, "valid_bce")
                row[f"{arm}_train_ap"] = curve_value(rr["curve"], epoch, "train_ap")
                row[f"{arm}_train_bce"] = curve_value(rr["curve"], epoch, "train_bce")
                row[f"{arm}_mask_dropped_fraction"] = curve_value(rr["curve"], epoch, "mask_dropped_fraction_mean")
        row["valid_ap_delta"] = (
            None if row.get("STRUCT_DROP_valid_ap") is None or row.get("BASE_CONT_valid_ap") is None
            else float(row["STRUCT_DROP_valid_ap"]) - float(row["BASE_CONT_valid_ap"])
        )
        curve_rows.append(row)
    curve_fields = ["epoch"]
    for arm in R.ARMS:
        curve_fields += [
            f"{arm}_lr", f"{arm}_valid_ap", f"{arm}_valid_bce",
            f"{arm}_train_ap", f"{arm}_train_bce", f"{arm}_mask_dropped_fraction",
        ]
    curve_fields += ["valid_ap_delta"]
    _write_csv(out_dir / "epoch_curves.csv", curve_fields, curve_rows)

    # ---- last-3 means ---------------------------------------------------
    for arm, rr in runs.items():
        last = [r for r in rr["curve"] if int(r["epoch"]) in (31, 32, 33)]
        rr["last3_epochs"] = [int(r["epoch"]) for r in last]
        vals = [_f(r["valid_ap"]) for r in last]
        vals = [v for v in vals if v is not None]
        rr["last3_mean_valid_ap"] = float(np.mean(vals)) if vals else None
        rr["last3_mean_valid_bce"] = float(
            np.mean([v for v in (_f(r["valid_bce"]) for r in last) if v is not None])
        )
        last3_tasks = {}
        for t in tasks:
            per = [rr["valid_by_epoch"].get(e, {}).get(t) for e in (31, 32, 33)]
            per = [v for v in per if v is not None]
            if per:
                last3_tasks[t] = float(np.mean(per))
        rr["last3_valid_per_task"] = last3_tasks

    # ---- high support ---------------------------------------------------
    hs = {}
    for arm, rr in runs.items():
        hs[arm] = {
            "best": _mean_over(rr["best_valid"], high_support),
            "endpoint": _mean_over(rr["endpoint_valid"], high_support),
            "last3": _mean_over(rr.get("last3_valid_per_task", {}), high_support),
            "start": _mean_over(rr["baseline_valid"], high_support),
        }

    # ---- Train-Valid gap at the best checkpoint -------------------------
    gaps = {}
    for arm, rr in runs.items():
        common = [t for t in tasks if rr["best_train"].get(t) is not None and rr["best_valid"].get(t) is not None]
        tv = _vec(rr["best_train"], common)
        vv = _vec(rr["best_valid"], common)
        finite = np.isfinite(tv) & np.isfinite(vv)
        gaps[arm] = {
            "n_common_tasks": int(finite.sum()),
            "tasks": [int(t) for t, keep in zip(common, finite) if keep],
            "mean_train_ap": float(tv[finite].mean()) if finite.any() else None,
            "mean_valid_ap": float(vv[finite].mean()) if finite.any() else None,
            "train_minus_valid_mean_ap": float((tv[finite] - vv[finite]).mean()) if finite.any() else None,
            "official_train_ap": float(rr["best"]["train"]["ap"]),
            "official_valid_ap": float(rr["best"]["valid"]["ap"]),
            "official_train_minus_valid_ap": float(rr["best"]["train"]["ap"]) - float(rr["best"]["valid"]["ap"]),
            "train_bce": float(rr["best"]["train"]["bce"]),
            "valid_bce": float(rr["best"]["valid"]["bce"]),
            "train_minus_valid_bce": float(rr["best"]["train"]["bce"]) - float(rr["best"]["valid"]["bce"]),
        }

    # ---- paired comparisons --------------------------------------------
    def compare(name: str, a_map: Mapping[int, float | None], b_map: Mapping[int, float | None],
                tasks_subset: Sequence[int], *, direction: str, official_a: float | None = None,
                official_b: float | None = None, note: str = "") -> dict[str, Any]:
        d = _vec(b_map, tasks_subset) - _vec(a_map, tasks_subset)
        out = {
            "comparison": f"{name}",
            "direction": direction,
            "mean_ap_a": _mean_over(a_map, tasks_subset),
            "mean_ap_b": _mean_over(b_map, tasks_subset),
            "official_ap_a": official_a,
            "official_ap_b": official_b,
            "official_ap_delta": (
                None if official_a is None or official_b is None else float(official_b) - float(official_a)
            ),
            "mean_delta": None if not np.isfinite(d).any() else float(np.nanmean(d)),
            "note": note,
            **bootstrap(d),
            "sensitivity_drop_top3": drop_top_abs(d, 3),
        }
        return out

    paired = {
        "schema": "fsrp-paired-v1",
        "round_id": R.ROUND_ID,
        "primary": compare(
            "best", base["best_valid"], drop["best_valid"], scorable_valid,
            direction="STRUCT_DROP - BASE_CONT",
            official_a=float(base["best"]["valid"]["ap"]), official_b=float(drop["best"]["valid"]["ap"]),
        ),
        "endpoint": compare(
            "endpoint", base["endpoint_valid"], drop["endpoint_valid"], scorable_valid,
            direction="STRUCT_DROP - BASE_CONT",
            official_a=curve_value(base["curve"], 33, "valid_ap"),
            official_b=curve_value(drop["curve"], 33, "valid_ap"),
        ),
        "last3": compare(
            "last3",
            base.get("last3_valid_per_task", {}),
            drop.get("last3_valid_per_task", {}),
            [t for t in scorable_valid if t in base.get("last3_valid_per_task", {}) and t in drop.get("last3_valid_per_task", {})],
            direction="STRUCT_DROP - BASE_CONT (mean of epochs 31-33)",
            official_a=base["last3_mean_valid_ap"], official_b=drop["last3_mean_valid_ap"],
        ),
        "struct_vs_start": compare(
            "STRUCT_DROP best vs epoch25 start", base["baseline_valid"], drop["best_valid"], scorable_valid,
            direction="STRUCT_DROP best - epoch25 common start",
            official_a=float(base["baseline"]["valid"]["ap"]), official_b=float(drop["best"]["valid"]["ap"]),
            note="common-start reference uses the BASE_CONT epoch-25 per-task AP (identical starting state)",
        ),
        "base_vs_start": compare(
            "BASE_CONT best vs epoch25 start", base["baseline_valid"], base["best_valid"], scorable_valid,
            direction="BASE_CONT best - epoch25 start",
            official_a=float(base["baseline"]["valid"]["ap"]), official_b=float(base["best"]["valid"]["ap"]),
            note="plain continuation gain",
        ),
        "high_support": {
            "tasks": [int(t) for t in high_support],
            "n_tasks": len(high_support),
            "struct_best_minus_base_best": (
                None if hs["STRUCT_DROP"]["best"] is None or hs["BASE_CONT"]["best"] is None
                else hs["STRUCT_DROP"]["best"] - hs["BASE_CONT"]["best"]
            ),
            "struct_endpoint_minus_base_endpoint": (
                None if hs["STRUCT_DROP"]["endpoint"] is None or hs["BASE_CONT"]["endpoint"] is None
                else hs["STRUCT_DROP"]["endpoint"] - hs["BASE_CONT"]["endpoint"]
            ),
            "struct_last3_minus_base_last3": (
                None if hs["STRUCT_DROP"]["last3"] is None or hs["BASE_CONT"]["last3"] is None
                else hs["STRUCT_DROP"]["last3"] - hs["BASE_CONT"]["last3"]
            ),
            "per_arm": hs,
        },
        "train_valid_gap": gaps,
        "common_start": start,
        "train_fit": {
            "base_train_ap_25_29_33": [
                curve_value(base["curve"], 29, "train_ap"), curve_value(base["curve"], 33, "train_ap")
            ],
            "struct_train_ap_29_33": [
                curve_value(drop["curve"], 29, "train_ap"), curve_value(drop["curve"], 33, "train_ap")
            ],
            "base_train_bce_29_33": [
                curve_value(base["curve"], 29, "train_bce"), curve_value(base["curve"], 33, "train_bce")
            ],
            "struct_train_bce_29_33": [
                curve_value(drop["curve"], 29, "train_bce"), curve_value(drop["curve"], 33, "train_bce")
            ],
            "base_best_train_ap": float(base["best"]["train"]["ap"]),
            "struct_best_train_ap": float(drop["best"]["train"]["ap"]),
            "base_best_train_bce": float(base["best"]["train"]["bce"]),
            "struct_best_train_bce": float(drop["best"]["train"]["bce"]),
        },
    }

    # ---- main table -----------------------------------------------------
    header = [
        "arm", "best_epoch", "best_valid_ap", "best_valid_bce", "endpoint_valid_ap", "endpoint_valid_bce",
        "last3_mean_valid_ap", "d_best_valid_ap_vs_base", "d_best_valid_ap_vs_start",
        "train_ap_25", "train_ap_29", "train_ap_33", "train_bce_25", "train_bce_29", "train_bce_33",
        "best_train_ap", "best_train_bce", "train_valid_gap_mean_ap", "high_support_valid_ap_best",
        "high_support_valid_ap_endpoint", "n_tasks_scored_valid", "parameters",
        "mask_dropped_fraction_mean", "gpu_hours", "wall_seconds",
    ]
    main_rows = []
    for arm, rr in runs.items():
        b = rr["best"]
        main_rows.append({
            "arm": arm,
            "best_epoch": int(b["epoch"]),
            "best_valid_ap": float(b["valid"]["ap"]),
            "best_valid_bce": float(b["valid"]["bce"]),
            "endpoint_valid_ap": curve_value(rr["curve"], 33, "valid_ap"),
            "endpoint_valid_bce": curve_value(rr["curve"], 33, "valid_bce"),
            "last3_mean_valid_ap": rr["last3_mean_valid_ap"],
            "d_best_valid_ap_vs_base": None,
            "d_best_valid_ap_vs_start": float(b["valid"]["ap"]) - float(rr["baseline"]["valid"]["ap"]),
            "train_ap_25": float(rr["baseline"]["train"]["ap"]),
            "train_ap_29": curve_value(rr["curve"], 29, "train_ap"),
            "train_ap_33": curve_value(rr["curve"], 33, "train_ap"),
            "train_bce_25": float(rr["baseline"]["train"]["bce"]),
            "train_bce_29": curve_value(rr["curve"], 29, "train_bce"),
            "train_bce_33": curve_value(rr["curve"], 33, "train_bce"),
            "best_train_ap": float(b["train"]["ap"]),
            "best_train_bce": float(b["train"]["bce"]),
            "train_valid_gap_mean_ap": gaps[arm]["train_minus_valid_mean_ap"],
            "high_support_valid_ap_best": hs[arm]["best"],
            "high_support_valid_ap_endpoint": hs[arm]["endpoint"],
            "n_tasks_scored_valid": int(rr["best"]["valid"]["n_tasks_scored"]),
            "parameters": int(rr["run"]["parameters"]),
            "mask_dropped_fraction_mean": (
                None if rr["run"].get("structural_dropout") is None
                else rr["run"]["structural_dropout"]["mean_epoch_dropped_fraction"]
            ),
            "gpu_hours": float(rr["run"]["gpu_hours"]),
            "wall_seconds": float(rr["run"]["gpu_seconds"]),
        })
    by_arm = {r["arm"]: r for r in main_rows}
    by_arm["STRUCT_DROP"]["d_best_valid_ap_vs_base"] = (
        by_arm["STRUCT_DROP"]["best_valid_ap"] - by_arm["BASE_CONT"]["best_valid_ap"]
    )
    by_arm["BASE_CONT"]["d_best_valid_ap_vs_base"] = 0.0
    _write_csv(out_dir / "main_table.csv", header, main_rows)

    # ---- verdict --------------------------------------------------------
    d_best = float(paired["primary"]["mean_delta"]) if paired["primary"].get("mean_delta") is not None else None
    d_best_official = by_arm["STRUCT_DROP"]["best_valid_ap"] - by_arm["BASE_CONT"]["best_valid_ap"]
    d_vs_start = by_arm["STRUCT_DROP"]["d_best_valid_ap_vs_start"]
    d_endpoint = (
        by_arm["STRUCT_DROP"]["endpoint_valid_ap"] - by_arm["BASE_CONT"]["endpoint_valid_ap"]
        if by_arm["STRUCT_DROP"]["endpoint_valid_ap"] is not None and by_arm["BASE_CONT"]["endpoint_valid_ap"] is not None
        else None
    )
    d_last3 = by_arm["STRUCT_DROP"]["last3_mean_valid_ap"] - by_arm["BASE_CONT"]["last3_mean_valid_ap"]
    d_hs = paired["high_support"]["struct_best_minus_base_best"]
    train_fit_drop = (
        by_arm["STRUCT_DROP"]["best_train_ap"] < by_arm["BASE_CONT"]["best_train_ap"]
        or (
            by_arm["STRUCT_DROP"]["train_bce_33"] is not None
            and by_arm["BASE_CONT"]["train_bce_33"] is not None
            and by_arm["STRUCT_DROP"]["train_bce_33"] > by_arm["BASE_CONT"]["train_bce_33"]
        )
    )
    complete = all(rr["run"]["status"] == "COMPLETED" for rr in runs.values())
    criteria = {
        "best_gain_vs_base_ge_0.003": d_best_official >= PROMISING_MIN_AP,
        "best_gain_vs_start_ge_0.003": d_vs_start is not None and d_vs_start >= PROMISING_MIN_AP,
        "endpoint_not_worse": d_endpoint is not None and d_endpoint >= 0.0,
        "last3_not_worse": d_last3 >= 0.0,
        "high_support_not_worse": d_hs is not None and d_hs >= 0.0,
        "both_runs_completed": complete,
    }
    promising = all(criteria[k] for k in criteria if k != "both_runs_completed") and complete
    no_validation_gain = d_best_official <= 0.0 and (d_vs_start is None or d_vs_start <= 0.0)
    if not complete:
        verdict = "INCONCLUSIVE"
        rationale = "at least one arm is incomplete; the short-horizon curves cannot be compared as planned"
    elif promising:
        verdict = "STRUCTURAL_REGULARIZATION_PROMISING"
        rationale = (
            "STRUCT_DROP best Valid AP beats BASE_CONT and the epoch-25 common start by >= 0.003, "
            "with endpoint, last-3 and high-support not worse than BASE_CONT"
        )
    elif train_fit_drop and no_validation_gain:
        verdict = "FIT_REDUCTION_WITHOUT_GAIN"
        rationale = "training fit is lower but neither the best nor the endpoint Valid AP improves"
    else:
        verdict = "NO_CLEAR_GAIN"
        rationale = "the pre-registered +0.003 best-Valid-AP practical threshold is not met"
    summary = {
        "schema": R.SCHEMA_SUMMARY,
        "round_id": R.ROUND_ID,
        "verdict": verdict,
        "rationale": rationale,
        "criteria": criteria,
        "best_valid_ap": {
            "BASE_CONT": by_arm["BASE_CONT"]["best_valid_ap"],
            "STRUCT_DROP": by_arm["STRUCT_DROP"]["best_valid_ap"],
            "delta": d_best_official,
        },
        "best_vs_common_start": {
            "BASE_CONT": by_arm["BASE_CONT"]["d_best_valid_ap_vs_start"],
            "STRUCT_DROP": d_vs_start,
        },
        "endpoint_valid_ap_delta": d_endpoint,
        "last3_mean_valid_ap_delta": d_last3,
        "high_support_valid_ap_delta_best": d_hs,
        "train_fit": paired["train_fit"],
        "train_fit_reduced": bool(train_fit_drop),
        "paired_task": paired["primary"],
        "paired_task_endpoint": paired["endpoint"],
        "paired_task_last3": paired["last3"],
        "sensitivity_drop_top3": paired["primary"]["sensitivity_drop_top3"],
        "train_valid_gap": gaps,
        "common_start": start,
        "resource": {
            arm: {
                "gpu_hours": runs[arm]["run"]["gpu_hours"],
                "wall_seconds": runs[arm]["run"]["gpu_seconds"],
                "gpu_peak_allocated_bytes": runs[arm]["run"]["gpu_peak_allocated_bytes"],
                "host": runs[arm]["run"]["host_facts"].get("hostname"),
                "git_commit": runs[arm]["run"]["git_commit"],
            }
            for arm in R.ARMS
        },
        "test_scored": False,
        "test_labels_loaded": False,
        "non_claims": [
            "no plain message-dropout control was run, so a positive result could not be attributed "
            "to the structural organisation of the mask",
            "single seed, 8-epoch continuation, PCBA Valid only; exploration label only",
        ],
        "utc": R.utcnow(),
    }
    R.write_json(out_dir / "summary.json", summary)
    R.write_json(out_dir / "paired_comparisons.json", paired)

    # ---- manifest -------------------------------------------------------
    manifest = {
        "schema": R.SCHEMA_MANIFEST,
        "round_id": R.ROUND_ID,
        "seed": R.SEED,
        "start_epoch": R.START_EPOCH,
        "end_epoch": R.END_EPOCH,
        "parameters": R.PARAMS_TOTAL,
        "base_checkpoint": base["run"]["base_checkpoint"],
        "checkpoints": [],
    }
    for arm in R.ARMS:
        rr = runs[arm]
        for kind, key in (("best_valid_ap", "best_checkpoint"), ("last_epoch", "last_checkpoint"),
                          ("horizon_endpoint", "horizon_checkpoint")):
            rel = rr["run"].get(key)
            if not rel:
                continue
            path = R.REPO / rel
            payload = torch_load_meta(path)
            manifest["checkpoints"].append({
                "arm": arm,
                "kind": kind,
                "epoch": payload.get("epoch"),
                "global_step": payload.get("global_step"),
                "path": rel,
                "bytes": int(path.stat().st_size),
                "sha256": R.sha256_file(path),
                "valid_ap": payload.get("valid_ap"),
                "parameters": payload.get("parameters"),
                "has_optimizer_state": bool(payload.get("optimizer_state_dict", {}).get("state")),
                "has_rng_state": bool(payload.get("rng_state")),
                "mask_rng_fingerprint": payload.get("mask_rng_fingerprint"),
            })
    R.write_json(out_dir / "checkpoint_manifest.json", manifest)

    # ---- markdown tables ------------------------------------------------
    lines = ["# FSRP v0 tables", ""]
    lines += ["## Main table", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in main_rows:
        lines.append("| " + " | ".join(_fmt(row.get(k)) for k in header) + " |")
    lines += ["", "## Curves", "", "| " + " | ".join(curve_fields) + " |", "|" + "---|" * len(curve_fields)]
    for row in curve_rows:
        lines.append("| " + " | ".join(_fmt(row.get(k)) for k in curve_fields) + " |")
    lines += ["", "## Paired comparisons (tasks)", ""]
    for key in ("primary", "endpoint", "last3", "struct_vs_start", "base_vs_start"):
        pc = paired[key]
        lines.append(
            f"- **{pc['comparison']}**: mean delta {_fmt(pc.get('mean_delta'))}, "
            f"95% CI [{_fmt(pc.get('ci95_low'))}, {_fmt(pc.get('ci95_high'))}], "
            f"tasks +{pc.get('n_improved')}/-{pc.get('n_degraded')}, "
            f"bootstrap P(delta>0)={_fmt(pc.get('fraction_positive'))}; "
            f"drop-top3 mean {_fmt((pc.get('sensitivity_drop_top3') or {}).get('mean_delta_kept'))}"
        )
    lines += ["", f"## Verdict: `{verdict}`", "", rationale, "", "Criteria:", ""]
    for k, v in criteria.items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Train-Valid gap (best checkpoint, common scorable tasks)", ""]
    for arm in R.ARMS:
        g = gaps[arm]
        lines.append(
            f"- {arm}: train mean AP {_fmt(g['mean_train_ap'])}, valid mean AP {_fmt(g['mean_valid_ap'])}, "
            f"gap {_fmt(g['train_minus_valid_mean_ap'])} over {g['n_common_tasks']} tasks"
        )
    (out_dir / "report_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "criteria": criteria, "delta_best": d_best_official}, indent=2))
    return 0


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{v:.6f}"
    return str(v)


def torch_load_meta(path: Path) -> dict[str, Any]:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


if __name__ == "__main__":
    raise SystemExit(main())
