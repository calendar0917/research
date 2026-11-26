"""Result analysis for the MEANFIELD pilot (no training, no Test access).

Produces:

* ``analysis/main_table.csv``            -- one row per arm at the 12-epoch horizon
* ``analysis/epoch_curves.csv``          -- every arm's epoch curve over the horizon
* ``analysis/per_task_metrics.csv``      -- task-level AP for MEANFIELD and FIXED1
* ``analysis/paired_comparisons.json``   -- paired task statistics + bootstrap + verdict
* ``analysis/allocation_checks.json``    -- structural checks + real-Valid contrast
* ``analysis/checkpoint_manifest.json``  -- checkpoint identity of the new run
* ``REPORT.md``                          -- Chinese report
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import mf_common as R
import ds_common as C

HORIZON = R.HORIZON
SUPPORT_MIN = C.SUPPORT_MIN
N_BOOT = 2000
BOOT_SEED = 0
TOP_K_SENSITIVITY = 3


# ---------------------------------------------------------------------------
# io helpers
# ---------------------------------------------------------------------------


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    rows = list(rows)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fields})


def fnum(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    try:
        return float(text)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# curve access
# ---------------------------------------------------------------------------


def curve_rows(path: Path) -> list[dict[str, Any]]:
    rows = read_csv(path)
    out = []
    for row in rows:
        epoch = int(float(row["epoch"]))
        if epoch < 1 or epoch > HORIZON:
            continue
        out.append(
            {
                "epoch": epoch,
                "valid_ap": fnum(row.get("valid_ap")),
                "valid_bce": fnum(row.get("valid_bce")),
                "train_ap": fnum(row.get("train_ap")),
                "train_bce": fnum(row.get("train_bce")),
            }
        )
    out.sort(key=lambda r: r["epoch"])
    return out


def curve_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scored = [r for r in rows if r["valid_ap"] is not None]
    if not scored:
        return {}
    best = max(scored, key=lambda r: (r["valid_ap"], -r["epoch"]))
    tail = scored[-3:]
    train_rows = [r for r in rows if r["train_ap"] is not None]
    return {
        "epochs": len(scored),
        "best_epoch": int(best["epoch"]),
        "best_valid_ap": float(best["valid_ap"]),
        "endpoint_epoch": int(scored[-1]["epoch"]),
        "endpoint_valid_ap": float(scored[-1]["valid_ap"]),
        "endpoint_valid_bce": scored[-1]["valid_bce"],
        "last3_valid_ap_mean": float(np.mean([r["valid_ap"] for r in tail])),
        "train_ap_last_available": (float(train_rows[-1]["train_ap"]) if train_rows else None),
        "train_ap_last_epoch": (int(train_rows[-1]["epoch"]) if train_rows else None),
        "train_bce_last_available": (float(train_rows[-1]["train_bce"]) if train_rows else None),
        "curve": [dict(r) for r in rows],
    }


# ---------------------------------------------------------------------------
# per-task access
# ---------------------------------------------------------------------------


def per_task_at(path: Path, *, split: str | None, epoch: int) -> dict[int, dict[str, float]]:
    rows = read_csv(path)
    out: dict[int, dict[str, float]] = {}
    for row in rows:
        if int(float(row["epoch"])) != int(epoch):
            continue
        if split is not None and str(row.get("split", "")) != split:
            continue
        task = int(float(row["task"]))
        ap = fnum(row.get("ap"))
        if ap is None:
            continue
        out[task] = {
            "ap": ap,
            "observed": fnum(row.get("observed")) or 0.0,
            "positive": fnum(row.get("positive")) or 0.0,
            "negative": fnum(row.get("negative")) or 0.0,
        }
    return out


def high_support_tasks(counts: Mapping[str, Any]) -> set[int]:
    positives = counts["valid"]["positive"]
    return {t for t, value in enumerate(positives) if int(value) >= SUPPORT_MIN}


def valid_positives_from_csv(path: Path, epoch: int) -> dict[int, float]:
    """Official Valid positive counts, read back from the run's own per-task table."""
    out: dict[int, float] = {}
    for row in read_csv(path):
        if str(row.get("split")) != "valid":
            continue
        if int(float(row["epoch"])) != int(epoch):
            continue
        out[int(float(row["task"]))] = fnum(row.get("positive")) or 0.0
    return out


# ---------------------------------------------------------------------------
# paired statistics
# ---------------------------------------------------------------------------


def paired_stats(
    mine: Mapping[int, dict[str, float]],
    ref: Mapping[int, dict[str, float]],
    *,
    label: str,
    support: set[int],
) -> dict[str, Any]:
    tasks = sorted(set(mine) & set(ref))
    deltas = np.asarray([mine[t]["ap"] - ref[t]["ap"] for t in tasks], dtype=np.float64)
    support_present = [i for i, t in enumerate(tasks) if t in support]
    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.integers(0, len(tasks), size=(N_BOOT, len(tasks)))
    boot = deltas[draws].mean(axis=1)
    order = np.argsort(-np.abs(deltas))
    keep = np.sort(order[TOP_K_SENSITIVITY:])
    out = {
        "comparison": label,
        "n_tasks": len(tasks),
        "mean_ap_mine": float(np.mean([mine[t]["ap"] for t in tasks])),
        "mean_ap_reference": float(np.mean([ref[t]["ap"] for t in tasks])),
        "delta_macro_ap": float(deltas.mean()),
        "delta_median": float(np.median(deltas)),
        "n_tasks_improved": int((deltas > 0).sum()),
        "n_tasks_degraded": int((deltas < 0).sum()),
        "n_tasks_tied": int((deltas == 0).sum()),
        "frac_tasks_improved": float((deltas > 0).mean()),
        "high_support": {
            "n_tasks": len(support_present),
            "support_min_valid_positives": int(SUPPORT_MIN),
            "mean_ap_mine": float(np.mean([mine[tasks[i]]["ap"] for i in support_present])) if support_present else None,
            "mean_ap_reference": float(np.mean([ref[tasks[i]]["ap"] for i in support_present])) if support_present else None,
            "delta": float(np.mean(deltas[support_present])) if support_present else None,
            "n_improved": int((deltas[support_present] > 0).sum()) if support_present else 0,
            "n_degraded": int((deltas[support_present] < 0).sum()) if support_present else 0,
        },
        "task_bootstrap": {
            "n_resamples": int(N_BOOT),
            "seed": int(BOOT_SEED),
            "unit": "task (official scorable task set), paired",
            "mean": float(boot.mean()),
            "ci95_low": float(np.percentile(boot, 2.5)),
            "ci95_high": float(np.percentile(boot, 97.5)),
            "prob_delta_positive": float((boot > 0).mean()),
        },
        "sensitivity_drop_top_abs_delta": {
            "k": int(TOP_K_SENSITIVITY),
            "dropped_tasks": [int(tasks[i]) for i in order[:TOP_K_SENSITIVITY]],
            "delta_excluding": float(deltas[keep].mean()),
            "n_tasks_remaining": int(keep.size),
        },
        "largest_abs_delta_tasks": [
            {"task": int(tasks[i]), "delta": float(deltas[i]), "mine": float(mine[tasks[i]]["ap"]),
             "reference": float(ref[tasks[i]]["ap"])}
            for i in order[:TOP_K_SENSITIVITY]
        ],
        "per_task_delta": {str(t): float(mine[t]["ap"] - ref[t]["ap"]) for t in tasks},
    }
    return out


def verdict(primary: Mapping[str, Any], reference_name: str) -> dict[str, Any]:
    gap = float(primary["delta_macro_ap"])
    hs = float(primary["high_support"]["delta"] or 0.0)
    late_conflict = hs < 0 and gap > 0
    if gap < -0.008 and hs < 0:
        label = "LOCAL_ALLOCATION_ADVANTAGE_SIGNAL"
        reason = (
            f"MEANFIELD trails {reference_name} by {gap:+.6f} macro AP and the high-support subset moves the "
            f"same way ({hs:+.6f}); removing the specific source->centre binding costs real performance."
        )
    elif abs(gap) <= 0.003 and not late_conflict:
        label = "DISTRIBUTIONAL_COMPUTATION_PROMISING"
        reason = (
            f"|gap| = {abs(gap):.6f} <= 0.003 with a high-support delta of {hs:+.6f}; keeping only the "
            f"intra-molecular content distribution and the per-centre relation counts is competitive at this horizon."
        )
    elif gap >= 0.005 and float(primary["frac_tasks_improved"]) >= 0.5 and hs >= 0:
        label = "DISTRIBUTIONAL_PERFORMANCE_ADVANCE"
        reason = (
            f"MEANFIELD leads {reference_name} by {gap:+.6f} with {primary['n_tasks_improved']}/"
            f"{primary['n_tasks']} tasks improved and high-support delta {hs:+.6f}."
        )
    else:
        label = "INCONCLUSIVE"
        reason = f"gap {gap:+.6f} with high-support delta {hs:+.6f} does not match any frozen decision band."
    return {
        "label": label,
        "reference": reference_name,
        "gap_macro_ap": gap,
        "high_support_delta": hs,
        "frac_tasks_improved": float(primary["frac_tasks_improved"]),
        "reason": reason,
        "bands": {
            "local_allocation_advantage": "gap < -0.008 and high-support same direction",
            "promising": "|gap| <= 0.003, no late/high-support conflict",
            "advance": "gap >= +0.005, >= 50% tasks improved, high-support not negative",
            "otherwise": "INCONCLUSIVE",
        },
        "caveat": "single seed, short horizon, practical scale only -- not a confirmatory claim",
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    R.ensure_dirs()
    run_dir = R.RUN_DIR
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    curve = curve_rows(run_dir / "train_curve.csv")
    summary = curve_summary(curve)
    counts = run.get("task_counts")
    if counts is None:
        positives = valid_positives_from_csv(run_dir / "per_task_ap.csv", int(run["best_epoch"]))
        counts = {"valid": {"positive": [int(positives.get(t, 0)) for t in range(C.N_TASKS)]}}
    support = high_support_tasks(counts)

    arms: dict[str, dict[str, Any]] = {}
    arms["MEANFIELD"] = {
        "role": "distributional message content, per-centre relation counts kept",
        "path": run_dir / "train_curve.csv",
        "per_task": run_dir / "per_task_ap.csv",
        "per_task_split": "valid",
        "source": "fsdp-v0 new run",
        "params": R.PARAMS_TOTAL,
    }
    arms["FIXED1"] = {
        "role": "real single-hop source->centre binding (primary reference)",
        "path": R.REF_RUN_DIR / "train_curve.csv",
        "per_task": R.REF_RUN_DIR / "per_task_ap.csv",
        "per_task_split": "valid",
        "source": "historical dscr-v0 FIXED1 seed 0 (read-only)",
        "params": R.PARAMS_TOTAL,
    }
    arms["SELF"] = {
        "role": "pointwise capacity control",
        "path": R.SELF_RUN_DIR / "train_curve.csv",
        "per_task": R.SELF_RUN_DIR / "per_task_ap.csv",
        "per_task_split": "valid",
        "source": "historical dscr-v0 SELF seed 0 (read-only)",
        "params": R.PARAMS_TOTAL,
    }
    arms["W2"] = {
        "role": "non-message-passing baseline",
        "path": R.W2_RUN_DIR / "train_curve.csv",
        "per_task": R.W2_RUN_DIR / "per_task_ap.csv",
        "per_task_split": None,
        "source": "historical lcrc-pcba-backbone-scaling-v1 W2 seed 0 (read-only)",
        "params": C.PARAMS_W2,
    }

    curves: dict[str, dict[str, Any]] = {}
    task_tables: dict[str, dict[int, dict[str, float]]] = {}
    table_rows: list[dict[str, Any]] = []
    for name, spec in arms.items():
        rows = curve_rows(spec["path"])
        curves[name] = curve_summary(rows)
        best_epoch = curves[name]["best_epoch"]
        task_tables[name] = per_task_at(spec["per_task"], split=spec["per_task_split"], epoch=best_epoch)
        hs_tasks = [t for t in task_tables[name] if t in support]
        hs_ap = float(np.mean([task_tables[name][t]["ap"] for t in hs_tasks])) if hs_tasks else None
        table_rows.append(
            {
                "model": name,
                "role": spec["role"],
                "seed": 0,
                "horizon": HORIZON,
                "params": spec["params"],
                "best_epoch": curves[name]["best_epoch"],
                "valid_ap": curves[name]["best_valid_ap"],
                "endpoint_epoch": curves[name]["endpoint_epoch"],
                "endpoint_valid_ap": curves[name]["endpoint_valid_ap"],
                "last3_valid_ap_mean": curves[name]["last3_valid_ap_mean"],
                "valid_bce_endpoint": curves[name]["endpoint_valid_bce"],
                "train_ap_at_epoch12": curves[name]["train_ap_last_available"],
                "train_bce_at_epoch12": curves[name]["train_bce_last_available"],
                "train_ap_epoch": curves[name]["train_ap_last_epoch"],
                "valid_high_support_ap": hs_ap,
                "valid_high_support_tasks": len(hs_tasks),
                "valid_minus_train_ap": (
                    None if curves[name]["train_ap_last_available"] is None
                    else curves[name]["best_valid_ap"] - curves[name]["train_ap_last_available"]
                ),
                "source": spec["source"],
            }
        )
        C.log(f"[analysis] {name}: best {curves[name]['best_valid_ap']:.9f} ep{curves[name]['best_epoch']}")

    write_csv(R.ANALYSIS / "main_table.csv", table_rows)

    curve_rows_out = []
    for name, spec in arms.items():
        for row in curves[name]["curve"]:
            curve_rows_out.append({"model": name, **row})
    write_csv(R.ANALYSIS / "epoch_curves.csv", curve_rows_out)

    primary = paired_stats(task_tables["MEANFIELD"], task_tables["FIXED1"], label="MEANFIELD-FIXED1", support=support)
    secondary = {
        "MEANFIELD-SELF": paired_stats(task_tables["MEANFIELD"], task_tables["SELF"], label="MEANFIELD-SELF", support=support),
        "MEANFIELD-W2": paired_stats(task_tables["MEANFIELD"], task_tables["W2"], label="MEANFIELD-W2", support=support),
    }
    decision = verdict(primary, "FIXED1")
    comparisons = {
        "schema": "fsdp-paired-comparisons-v1",
        "round_id": R.ROUND_ID,
        "horizon": HORIZON,
        "seed": R.SEED,
        "created_utc": C.utcnow(),
        "primary": primary,
        "secondary": secondary,
        "decision": decision,
        "reference_curves_frozen": {
            "FIXED1_12ep_prefix": R.HISTORICAL_FIXED1_PREFIX,
            "FIXED1_best_epoch": curves["FIXED1"]["best_epoch"],
            "FIXED1_best_valid_ap_recomputed": curves["FIXED1"]["best_valid_ap"],
            "SELF_12ep_prefix_best": curves["SELF"]["best_valid_ap"],
            "W2_12ep_prefix_best": curves["W2"]["best_valid_ap"],
        },
        "notes": [
            "macro AP is the official OGB Evaluator mean over scorable tasks; the task bootstrap resamples tasks",
            "high-support = tasks with >= %d official Valid positives; the subset never replaces the official metric" % SUPPORT_MIN,
            "Train AP/BCE are only available at epochs %s and at the best checkpoint for MEANFIELD" % (list(R.TRAIN_EVAL_EPOCHS),),
            "no Test split was scored in this round",
        ],
    }
    C.write_json(R.ANALYSIS / "paired_comparisons.json", comparisons)

    per_task_rows = []
    tasks = sorted(set(task_tables["MEANFIELD"]) | set(task_tables["FIXED1"]))
    for t in tasks:
        mf = task_tables["MEANFIELD"].get(t)
        f1 = task_tables["FIXED1"].get(t)
        per_task_rows.append(
            {
                "task": t,
                "valid_positive": int(counts["valid"]["positive"][t]),
                "valid_negative": (
                    int(counts["valid"]["negative"][t]) if "negative" in counts["valid"] else None
                ),
                "high_support": bool(t in support),
                "meanfield_ap": None if mf is None else mf["ap"],
                "fixed1_ap": None if f1 is None else f1["ap"],
                "delta": None if (mf is None or f1 is None) else mf["ap"] - f1["ap"],
            }
        )
    write_csv(R.ANALYSIS / "per_task_metrics.csv", per_task_rows)

    for name, filename in (
        ("allocation_checks.json", "allocation_checks.json"),
        ("checkpoint_manifest.json", "checkpoint_manifest.json"),
    ):
        source = run_dir / filename
        if source.is_file():
            payload = json.loads(source.read_text(encoding="utf-8"))
            C.write_json(R.ANALYSIS / filename, payload)

    summary_payload = {
        "schema": R.SCHEMA_SUMMARY,
        "round_id": R.ROUND_ID,
        "run_status": run.get("status"),
        "stop_reason": run.get("stop_reason"),
        "epochs_run": run.get("epochs_run"),
        "parameters": run.get("parameters"),
        "best_epoch": run.get("best_epoch"),
        "best_valid_ap": run.get("best_valid_ap"),
        "contrast_status": run.get("contrast_status"),
        "structural_checks_passed": run.get("structural_checks_passed"),
        "decision": decision,
        "main_table": table_rows,
        "gpu_hours": run.get("gpu_hours"),
        "created_utc": C.utcnow(),
    }
    C.write_json(R.ANALYSIS / "summary.json", summary_payload)
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
