"""Read-only analysis of the two full-Train runs.

Builds the comparison table (1M historical controls vs FULL W2/W3), the curves
(CSV + PNG), last-5-epoch means and the 1M→FULL / W3−W2 deltas, and provides an
independent CPU re-scoring of the best checkpoints.  No training and no Test
access.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import fs_common as C  # noqa: E402
import fs_data as D  # noqa: E402

import torch  # noqa: E402

SUMMARY = C.ANALYSIS / "fs_summary.json"
CURVES_CSV = C.ANALYSIS / "fs_curves.csv"
CURVES_PNG = C.ANALYSIS / "fs_curves.png"


def _curve(run: str) -> list[dict[str, Any]]:
    path = C.RUNS / run / "train_curve.csv"
    rows: list[dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if value == "":
                    parsed[key] = None
                    continue
                try:
                    parsed[key] = float(value) if ("." in value or "e" in value.lower()) else int(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def _summarize(run: str) -> dict[str, Any]:
    payload = C.read_json(C.RUNS / run / "run.json")
    curve = _curve(run)
    valid = [float(r["valid_mae_eV"]) for r in curve]
    monitor = [float(r["monitor_mae_eV"]) for r in curve]
    best_epoch = int(payload["best_epoch"])
    return {
        "run": run,
        "model": payload["model"],
        "train_size": int(payload["train_size"]),
        "parameters": int(payload["parameters"]),
        "epochs_run": int(payload["epochs_run"]),
        "status": payload["status"],
        "best_epoch": best_epoch,
        "best_valid_mae_eV": float(payload["best_valid_mae_eV"]),
        "valid_mae_eV_at_40": float(valid[-1]),
        "monitor_mae_eV_at_40": float(monitor[-1]),
        "last5_valid_mean_eV": float(np.mean(valid[-5:])),
        "last5_monitor_mean_eV": float(np.mean(monitor[-5:])),
        "valid_mae_eV_by_epoch": valid,
        "monitor_mae_eV_by_epoch": monitor,
        "train_l1_z_by_epoch": [float(r["train_l1_z"]) for r in curve],
        "step_ms_p50_by_epoch": [float(r["step_ms_p50"]) for r in curve],
        "gpu_hours": float(payload["gpu_hours"]),
        "molecules_per_second": float(payload["throughput"]["molecules_per_second_train"]),
        "steps_total": int(payload["steps_total"]),
        "steps_per_epoch": int(payload["steps_per_epoch"]),
        "curve_sha256": payload["curve_sha256"],
        "best_checkpoint_sha256": payload["best_checkpoint_sha256"],
        "primary_checkpoint_sha256": payload["primary_checkpoint_sha256"],
        "last_checkpoint_sha256": payload["last_checkpoint_sha256"],
        "git_commit": payload["git_commit"],
        "host": payload["host_facts"].get("hostname"),
        "gpu": (payload.get("gpu_occupancy_start") or {}).get("nvidia_smi_gpus"),
        "gpu_peak_allocated_bytes": int(payload["gpu_peak_allocated_bytes"]),
        "peak_rss_mb": float(payload["peak_rss_mb"]),
    }


def _historical() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, h in C.HISTORICAL.items():
        out[key] = {
            "run": key,
            "model": f"W{h['scale']}",
            "train_size": int(h["train_rows"]),
            "parameters": int(h["parameters"]),
            "best_epoch": int(h["best_epoch"]),
            "best_valid_mae_eV": float(h["best_valid_mae_eV"]),
            "valid_mae_eV_at_40": float(h["valid_mae_eV_at_40"]),
            "monitor_mae_eV_at_40": float(h["monitor_mae_eV_at_40"]),
            "gpu_hours": float(h["gpu_hours"]),
            "source_round": h["source_round"],
            "checkpoint_best_sha256": h["checkpoint_best_sha256"],
        }
    return out


def build_summary() -> dict[str, Any]:
    runs = {run: _summarize(run) for run in C.RUN_ORDER}
    hist = _historical()
    w2, w3 = runs["PCQM_W2_FULL"], runs["PCQM_W3_FULL"]
    summary = {
        "schema": "lcrc-pcqm4mv2-fullscale-summary-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "full_runs": runs,
        "historical_1m": hist,
        "comparisons": {
            "delta_w3_minus_w2_full_at_best_eV": w3["best_valid_mae_eV"] - w2["best_valid_mae_eV"],
            "delta_w3_minus_w2_full_at_40_eV": w3["valid_mae_eV_at_40"] - w2["valid_mae_eV_at_40"],
            "relative_w3_minus_w2_full_at_40_pct": 100.0
            * (w3["valid_mae_eV_at_40"] - w2["valid_mae_eV_at_40"])
            / w2["valid_mae_eV_at_40"],
            "delta_1m_to_full_w2_best_eV": w2["best_valid_mae_eV"] - hist["PCQM_W2_1M"]["best_valid_mae_eV"],
            "delta_1m_to_full_w3_best_eV": w3["best_valid_mae_eV"] - hist["PCQM_W3_1M"]["best_valid_mae_eV"],
            "delta_1m_to_full_w2_at_40_eV": w2["valid_mae_eV_at_40"] - hist["PCQM_W2_1M"]["valid_mae_eV_at_40"],
            "delta_1m_to_full_w3_at_40_eV": w3["valid_mae_eV_at_40"] - hist["PCQM_W3_1M"]["valid_mae_eV_at_40"],
            "monitor_minus_valid_w2_eV": w2["monitor_mae_eV_at_40"] - w2["valid_mae_eV_at_40"],
            "monitor_minus_valid_w3_eV": w3["monitor_mae_eV_at_40"] - w3["valid_mae_eV_at_40"],
            "delta_w1m_baseline_eV": hist["PCQM_W3_1M"]["valid_mae_eV_at_40"]
            - hist["PCQM_W2_1M"]["valid_mae_eV_at_40"],
            "delta_1m_to_full_w3_at_40_minus_w1m_delta_eV": (
                w3["valid_mae_eV_at_40"] - hist["PCQM_W3_1M"]["valid_mae_eV_at_40"]
            )
            - (hist["PCQM_W3_1M"]["valid_mae_eV_at_40"] - hist["PCQM_W2_1M"]["valid_mae_eV_at_40"]),
        },
        "resource": {
            "gpu_hours_w2_full": w2["gpu_hours"],
            "gpu_hours_w3_full": w3["gpu_hours"],
            "gpu_hours_full_total": w2["gpu_hours"] + w3["gpu_hours"],
            "budget_target_gpu_hours": C.BUDGET["target_gpu_hours"],
            "budget_hard_gpu_hours": C.BUDGET["hard_gpu_hours"],
            "within_target": bool((w2["gpu_hours"] + w3["gpu_hours"]) <= C.BUDGET["target_gpu_hours"]),
            "within_hard": bool((w2["gpu_hours"] + w3["gpu_hours"]) <= C.BUDGET["hard_gpu_hours"]),
        },
        "test_mae": "PENDING_OFFICIAL_EVALUATION",
        "test_labels_read": False,
    }
    C.write_json(SUMMARY, summary)
    write_curves_csv(runs, hist)
    write_png(runs)
    write_table(summary)
    return summary


def write_curves_csv(runs: Mapping[str, Mapping[str, Any]], hist: Mapping[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for run, payload in runs.items():
        for epoch, (valid, monitor, l1) in enumerate(
            zip(payload["valid_mae_eV_by_epoch"], payload["monitor_mae_eV_by_epoch"], payload["train_l1_z_by_epoch"]),
            start=1,
        ):
            rows.append(
                {
                    "run": run,
                    "model": payload["model"],
                    "train_size": payload["train_size"],
                    "epoch": epoch,
                    "valid_mae_eV": valid,
                    "monitor_mae_eV": monitor,
                    "train_l1_z": l1,
                }
            )
    with open(CURVES_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_png(runs: Mapping[str, Mapping[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    colors = {"PCQM_W2_FULL": "tab:blue", "PCQM_W3_FULL": "tab:orange"}
    for run, payload in runs.items():
        epochs = np.arange(1, len(payload["valid_mae_eV_by_epoch"]) + 1)
        axes[0, 0].plot(epochs, payload["valid_mae_eV_by_epoch"], label=f"{payload['model']} FULL", color=colors[run])
        axes[0, 1].plot(epochs, payload["monitor_mae_eV_by_epoch"], label=f"{payload['model']} FULL", color=colors[run])
        axes[1, 0].plot(epochs, payload["train_l1_z_by_epoch"], label=f"{payload['model']} FULL", color=colors[run])
        axes[1, 1].plot(epochs, payload["step_ms_p50_by_epoch"], label=f"{payload['model']} FULL", color=colors[run])
    for ax, title in zip(
        axes.flat,
        ("Validation MAE (eV)", "Train-monitor MAE (eV)", "Train L1(z)", "step p50 (ms)"),
    ):
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(CURVES_PNG, dpi=150)
    plt.close(fig)


def write_table(summary: Mapping[str, Any]) -> None:
    hist = summary["historical_1m"]
    runs = summary["full_runs"]
    lines = [
        "# Full-Train comparison (Validation MAE, eV)",
        "",
        "| model | Train size | Parameters | Best Valid MAE | Best epoch | Valid@40 | Monitor@40 | GPU-hours |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("PCQM_W2_1M", "PCQM_W3_1M"):
        h = hist[key]
        lines.append(
            f"| {h['model']}-1M historical | {h['train_size']:,} | {h['parameters']:,} | "
            f"{h['best_valid_mae_eV']:.6f} | {h['best_epoch']} | {h['valid_mae_eV_at_40']:.6f} | "
            f"{h['monitor_mae_eV_at_40']:.6f} | {h['gpu_hours']:.3f} |"
        )
    for key in ("PCQM_W2_FULL", "PCQM_W3_FULL"):
        r = runs[key]
        lines.append(
            f"| {r['model']}-FULL | {r['train_size']:,} | {r['parameters']:,} | "
            f"{r['best_valid_mae_eV']:.6f} | {r['best_epoch']} | {r['valid_mae_eV_at_40']:.6f} | "
            f"{r['monitor_mae_eV_at_40']:.6f} | {r['gpu_hours']:.3f} |"
        )
    c = summary["comparisons"]
    lines += [
        "",
        f"W3−W2 at FULL: best {c['delta_w3_minus_w2_full_at_best_eV']:+.6f} eV, "
        f"@40 {c['delta_w3_minus_w2_full_at_40_eV']:+.6f} eV "
        f"({c['relative_w3_minus_w2_full_at_40_pct']:+.2f} %).",
        f"1M→FULL: W2 {c['delta_1m_to_full_w2_at_40_eV']:+.6f} eV, "
        f"W3 {c['delta_1m_to_full_w3_at_40_eV']:+.6f} eV (at epoch 40).",
        "",
    ]
    (C.ANALYSIS / "fs_summary_table.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# independent CPU re-scoring of the best checkpoints
# ---------------------------------------------------------------------------


def recheck(run: str, *, device: torch.device) -> dict[str, Any]:
    payload = C.read_json(C.RUNS / run / "run.json")
    checkpoint = Path(C.REPO / payload["best_checkpoint"])
    task = D.load_task(run, verify_hashes=False, verbose=False)
    torch_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model, audit = C.EW_MODEL.build_model(
        task["spec"], task["norm"], scale=int(task["scale"]), seed=int(task["seed"]), device=device
    )
    model.load_state_dict(torch_payload["state_dict"], strict=True)
    model.eval()
    report, pred_z = C.PQ_TRAIN.evaluate_split(model, task, device, which="valid", batch_size=512)
    y = np.asarray(task["valid_y_eV"], dtype=np.float64)
    pred = np.asarray(C.PQ_DATA.from_z(pred_z, task["label_norm"]), dtype=np.float64)
    out = {
        "schema": "lcrc-pcqm4mv2-fullscale-recheck-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "run": run,
        "checkpoint": C.track_rel(checkpoint),
        "checkpoint_sha256": C.sha256_file(checkpoint),
        "checkpoint_epoch": int(torch_payload["epoch"]),
        "parameters": int(audit["total_parameters"]),
        "recorded_best_valid_mae_eV": float(payload["best_valid_mae_eV"]),
        "recomputed_official_mae_eV": float(report["mae_eV"]),
        "recomputed_numpy_mae_eV": float(np.mean(np.abs(pred - y))),
        "delta_vs_recorded_eV": float(report["mae_eV"]) - float(payload["best_valid_mae_eV"]),
        "n": int(y.size),
        "pred_sha256": C.sha256_bytes(pred.tobytes()),
        "device": str(device),
    }
    C.write_json(C.ANALYSIS / f"fs_recheck_{run}.json", out)
    C.log(
        f"[recheck:{run}] recorded {out['recorded_best_valid_mae_eV']:.9f} "
        f"recomputed {out['recomputed_official_mae_eV']:.9f} eV "
        f"(Δ {out['delta_vs_recorded_eV']:+.2e})"
    )
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="summary", choices=("summary", "recheck", "all"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.stage in ("summary", "all"):
        summary = build_summary()
        C.log(f"[analysis] summary written; W3-W2@40={summary['comparisons']['delta_w3_minus_w2_full_at_40_eV']:+.6f} eV")
    if args.stage in ("recheck", "all"):
        for run in C.RUN_ORDER:
            recheck(run, device=torch.device(args.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
