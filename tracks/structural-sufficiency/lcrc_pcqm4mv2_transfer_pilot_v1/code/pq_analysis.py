"""Aggregate the two formal runs into the report artifacts (CPU only).

Reads ``runs/<run>/run.json`` + ``runs/<run>/train_curve.csv`` and writes:

* ``analysis/pcqm_summary.json``  — every number quoted in ``REPORT.md``
* ``analysis/pcqm_curves.csv``    — the two epoch-1..40 Validation curves merged
* ``analysis/pcqm_summary_table.md`` — the report's front table

Nothing here trains, selects or re-evaluates a model; it only restates recorded
metrics.  Test splits are never mentioned because they were never touched.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from collections.abc import Mapping as Mappingish
from typing import Any, Sequence

import numpy as np

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import pq_common as C  # noqa: E402

CURVE_KEYS = (
    "epoch",
    "learning_rate",
    "train_l1_z",
    "valid_mae_eV",
    "monitor_mae_eV",
    "step_ms_p50",
    "epoch_train_seconds",
    "epoch_valid_seconds",
    "cumulative_seconds",
    "grad_norm_max",
    "gradient_clip_events",
)


def _curve(run: str) -> list[dict[str, Any]]:
    path = C.RUNS / run / "train_curve.csv"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for row in rows:
        if str(row.get("evaluated", "")).lower() not in ("true", "1"):
            continue
        out.append({k: float(row[k]) for k in CURVE_KEYS if k in row and row[k] not in ("", "nan")})
    return out


def _trend(values: Sequence[float], last: int = 10) -> dict[str, Any]:
    vals = list(values[-last:])
    if len(vals) < 2:
        return {"n": len(vals), "slope_eV_per_epoch": float("nan")}
    x = np.arange(len(vals), dtype=np.float64)
    slope = float(np.polyfit(x, np.asarray(vals, dtype=np.float64), 1)[0])
    return {
        "n": len(vals),
        "first_epoch": None,
        "mean_eV": float(np.mean(vals)),
        "min_eV": float(np.min(vals)),
        "max_eV": float(np.max(vals)),
        "slope_eV_per_epoch": slope,
        "first_eV": float(vals[0]),
        "last_eV": float(vals[-1]),
        "drop_eV": float(vals[0] - vals[-1]),
    }


def _best_after(curve: Sequence[Mappingish], first_epoch: int) -> dict[str, Any]:
    tail = [r for r in curve if int(r.get("epoch", 0)) >= first_epoch and "valid_mae_eV" in r]
    if not tail:
        return {"n": 0, "epoch": None, "valid_mae_eV": None}
    best = min(tail, key=lambda r: (r["valid_mae_eV"], r["epoch"]))
    return {"n": len(tail), "epoch": int(best["epoch"]), "valid_mae_eV": float(best["valid_mae_eV"])}


def summarise_run(run: str) -> dict[str, Any]:
    meta_path = C.RUNS / run / "run.json"
    if not meta_path.exists():
        return {"run": run, "status": "MISSING", "run_json": None}
    meta = C.read_json(meta_path)
    curve = _curve(run)
    valid = [r["valid_mae_eV"] for r in curve if "valid_mae_eV" in r]
    monitor = [r["monitor_mae_eV"] for r in curve if "monitor_mae_eV" in r]
    best_row = min(
        (r for r in curve if "valid_mae_eV" in r),
        key=lambda r: (r["valid_mae_eV"], r["epoch"]),
        default=None,
    )
    last = curve[-1] if curve else {}
    ckpts = {}
    for name in ("checkpoint_best_valid.pt", "checkpoint_epoch40.pt", "checkpoint_last.pt"):
        path = C.RUNS / run / name
        if path.exists():
            ckpts[name] = {
                "path": C.track_rel(path),
                "bytes": int(path.stat().st_size),
                "sha256": C.sha256_file(path),
            }
    isolation = meta.get("data_isolation") or {}
    return {
        "run": run,
        "status": meta.get("status"),
        "stop_reason": meta.get("stop_reason"),
        "run_json": C.track_rel(meta_path),
        "train_rows": int(meta.get("train_size") or 0),
        "epochs_planned": int(meta.get("epochs") or 0),
        "epochs_run": int(meta.get("epochs_run") or 0),
        "steps_per_epoch": int(meta.get("steps_per_epoch") or 0),
        "steps_total": int(meta.get("steps_total") or 0),
        "global_steps": int(meta.get("global_steps_run") or 0),
        "parameters_total": int(meta.get("parameters") or 0),
        "parameters_base": int(meta.get("base_parameters") or 0),
        "parameters_readout": int(meta.get("readout_parameters") or 0),
        "best_epoch": int(meta.get("best_epoch") or 0),
        "best_epoch_from_curve": int(best_row["epoch"]) if best_row else 0,
        "best_valid_mae_eV": float(meta.get("best_valid_mae_eV") or float("nan")),
        "valid_mae_at_last_epoch_eV": float(last.get("valid_mae_eV", float("nan"))),
        "monitor_mae_at_last_epoch_eV": float(last.get("monitor_mae_eV", float("nan"))),
        "train_l1_z_at_last_epoch": float(last.get("train_l1_z", float("nan"))),
        "epochs_evaluated": len(curve),
        "gpu_hours": float(meta.get("gpu_hours") or 0.0),
        "gpu_seconds": float(meta.get("gpu_seconds") or 0.0),
        "mean_epoch_seconds": float(meta.get("epoch_seconds_mean") or 0.0),
        "train_seconds": float(meta.get("train_seconds") or 0.0),
        "valid_seconds": float(meta.get("valid_seconds") or 0.0),
        "monitor_seconds": float(meta.get("monitor_seconds") or 0.0),
        "steps_per_second_mean": float(meta.get("steps_per_second_mean") or 0.0),
        "step_ms_p50_median": float(np.median([r["step_ms_p50"] for r in curve]))
        if curve
        else float("nan"),
        "gradient_clip_events": int(meta.get("grad_clip_events") or 0),
        "gpu_peak_allocated_bytes": int(meta.get("gpu_peak_allocated_bytes") or 0),
        "peak_rss_mb": float(meta.get("peak_rss_mb") or 0.0),
        "n_valid": int(isolation.get("valid_rows") or 0),
        "n_monitor": int(isolation.get("monitor_rows") or 0),
        "data_isolation": isolation,
        "label_norm": meta.get("label_norm"),
        "valid_mae_curve": valid,
        "monitor_mae_curve": monitor,
        "trend_last10_valid": _trend(valid, 10),
        "trend_last10_monitor": _trend(monitor, 10),
        "best_after_epoch_20": _best_after(curve, 20),
        "checkpoints": ckpts,
        "git_commit": meta.get("git_commit"),
        "device": meta.get("device"),
        "host_facts": meta.get("host_facts"),
        "deterministic_settings": meta.get("deterministic_settings"),
        "test_evaluated": meta.get("test_evaluated"),
        "formal_run": meta.get("formal_run"),
    }


def build_summary() -> dict[str, Any]:
    baseline = C.read_json(C.ANALYSIS / "pcqm_constant_baseline.json")
    smoke = C.read_json(C.ANALYSIS / "pcqm_smoke.json") if (C.ANALYSIS / "pcqm_smoke.json").exists() else {}
    runs = {run: summarise_run(run) for run in C.RUN_ORDER}
    a = runs["PCQM_W2_300K"]
    b = runs["PCQM_W2_1M"]
    improvement: dict[str, Any] = {}
    if a.get("best_valid_mae_eV") and b.get("best_valid_mae_eV"):
        imp = float(a["best_valid_mae_eV"]) - float(b["best_valid_mae_eV"])
        improvement = {
            "best_valid_mae_delta_eV": imp,
            "best_valid_mae_relative": imp / float(a["best_valid_mae_eV"]),
            "at_epoch40_delta_eV": float(a["valid_mae_at_last_epoch_eV"] - b["valid_mae_at_last_epoch_eV"]),
            "at_epoch40_relative": float(
                (a["valid_mae_at_last_epoch_eV"] - b["valid_mae_at_last_epoch_eV"])
                / a["valid_mae_at_last_epoch_eV"]
            ),
            "monitor_at_epoch40_delta_eV": float(
                a["monitor_mae_at_last_epoch_eV"] - b["monitor_mae_at_last_epoch_eV"]
            ),
            "rows_ratio": float(b["train_rows"] / max(a["train_rows"], 1)),
            "gpu_hours_ratio": float(b["gpu_seconds"] / max(a["gpu_seconds"], 1e-9)),
        }
        # cost-normalised gain: eV removed per extra GPU-hour
        extra_h = float(b["gpu_hours"] - a["gpu_hours"])
        improvement["delta_eV_per_extra_gpu_hour"] = float(imp / extra_h) if extra_h > 0 else float("nan")
    positioning = {
        "note": (
            "public numbers are full-data (3.38M molecules) reference baselines copied from the "
            "official OGB-LSC PCQM4Mv2 leaderboard docs; they are NOT like-for-like with this pilot"
        ),
        "public_dev_mae_eV": C.PUBLIC_REFERENCE,
        "our_valid_mae_eV": {
            run: runs[run].get("best_valid_mae_eV") for run in C.RUN_ORDER if runs[run].get("run_json")
        },
        "difference_to_gin_virtual_test_dev_eV": {
            run: (
                float(runs[run]["best_valid_mae_eV"]) - float(C.PUBLIC_REFERENCE["GIN-virtual"]["test_dev_mae"])
                if runs[run].get("best_valid_mae_eV")
                else None
            )
            for run in C.RUN_ORDER
        },
        "data_used_fraction_of_official_train": {
            run: float(runs[run].get("train_rows", 0)) / 3_378_606 for run in C.RUN_ORDER
        },
    }
    ledger = {
        "projected_gpu_hours_from_smoke": (smoke.get("details", {}).get("cost") or {}).get(
            "projected_total_gpu_hours"
        ),
        "budget_target_hours": 6.0,
        "budget_hard_hours": 9.0,
        "measured_gpu_hours_total": float(sum(runs[r].get("gpu_seconds", 0.0) for r in C.RUN_ORDER) / 3600.0),
        "measured_gpu_hours": {run: runs[run].get("gpu_hours") for run in C.RUN_ORDER},
    }
    return {
        "schema": "lcrc-pcqm4mv2-summary-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "constant_baseline": baseline,
        "runs": runs,
        "improvement_300k_to_1m": improvement,
        "positioning": positioning,
        "resource_ledger": ledger,
        "recipe": {run: C.recipe_for_run(run) for run in C.RUN_ORDER},
        "smoke_status": smoke.get("status"),
    }


def write_curves(runs: dict[str, Any]) -> Path:
    path = C.ANALYSIS / "pcqm_curves.csv"
    fields = ["epoch", "run", "train_rows", *CURVE_KEYS[1:]]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for run in C.RUN_ORDER:
            rows = _curve(run)
            n = C.RUNS_TABLE[run]["train_size"]
            for row in rows:
                writer.writerow({"run": run, "train_rows": n, **{k: row.get(k, "") for k in CURVE_KEYS}})
    return path


def table_markdown(summary: dict[str, Any]) -> str:
    runs = summary["runs"]
    base = summary["constant_baseline"]
    head = (
        "| arm | Params | Best epoch | Best Valid MAE (eV) | Valid MAE @40 (eV) | "
        "Train-monitor MAE @40 (eV) | GPU-h |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = [
        f"| Constant median | 0 | — | {float(base['valid_mae_eV']):.4f} | {float(base['valid_mae_eV']):.4f} "
        f"| — | 0.00 |"
    ]
    for run in C.RUN_ORDER:
        r = runs[run]
        if not r.get("run_json"):
            lines.append(f"| {run} | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {run.replace('PCQM_W2_', 'LCRC-W2-')} | {r['parameters_total']:,} | {r['best_epoch']} "
            f"| {r['best_valid_mae_eV']:.4f} | {r['valid_mae_at_last_epoch_eV']:.4f} "
            f"| {r['monitor_mae_at_last_epoch_eV']:.4f} | {r['gpu_hours']:.2f} |"
        )
    return head + "\n".join(lines) + "\n"


def write_plot(summary: dict[str, Any]) -> Path | None:
    """Small two-panel figure: full Validation MAE and Train-monitor MAE."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001 - plotting is optional
        return None
    runs = summary["runs"]
    if not any(runs[r].get("run_json") for r in C.RUN_ORDER):
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharex=True)
    colors = {"PCQM_W2_300K": "#1f77b4", "PCQM_W2_1M": "#d62728"}
    for run in C.RUN_ORDER:
        r = runs[run]
        curve = _curve(run)
        if not curve:
            continue
        epochs = [row["epoch"] for row in curve]
        label = f"{run.replace('PCQM_W2_', 'LCRC-W2-')} ({r['train_rows']:,} rows)"
        axes[0].plot(epochs, [row["valid_mae_eV"] for row in curve], label=label, color=colors[run])
        axes[1].plot(epochs, [row["monitor_mae_eV"] for row in curve], label=label, color=colors[run])
        best = r["best_epoch"]
        axes[0].scatter([best], [r["best_valid_mae_eV"]], marker="o", s=22, color=colors[run], zorder=5)
    base = float(summary["constant_baseline"]["valid_mae_eV"])
    axes[0].axhline(base, linestyle=":", color="grey", linewidth=1.0, label=f"constant median ({base:.3f})")
    for ax, ylabel, title in (
        (axes[0], "Validation MAE (eV)", "official Validation split (73,545 molecules)"),
        (axes[1], "Train-monitor MAE (eV)", "fixed 20,000-row Train monitor"),
    ):
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3, linewidth=0.6)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("LCRC-W2 on PCQM4Mv2 (2D, HOMO-LUMO gap, MAE in eV)", fontsize=10)
    fig.tight_layout()
    path = C.ANALYSIS / "pcqm_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    summary = build_summary()
    C.write_json(C.ANALYSIS / "pcqm_summary.json", summary)
    curves = write_curves(summary["runs"])
    table = table_markdown(summary)
    (C.ANALYSIS / "pcqm_summary_table.md").write_text(table, encoding="utf-8")
    plot = write_plot(summary)
    if plot is not None and not args.quiet:
        print(f"plot -> {C.track_rel(plot)}")
    if not args.quiet:
        print(table)
        print(f"curves -> {C.track_rel(curves)}")
        print(
            f"improvement 1M vs 300K: {summary['improvement_300k_to_1m'].get('best_valid_mae_delta_eV')} eV "
            f"({summary['improvement_300k_to_1m'].get('best_valid_mae_relative')})"
        )
        ledger = summary["resource_ledger"]
        print(f"gpu-hours: {ledger['measured_gpu_hours']} total {ledger['measured_gpu_hours_total']:.2f}")
    ok = all(summary["runs"][r].get("status") == "COMPLETED" for r in C.RUN_ORDER)
    return 0 if ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
