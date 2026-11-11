"""Analysis of the three-model comparison and the full-data projection.

Reads the frozen historical W2 curve from the parent round (read-only) and the
new W1/W3 curves (when present), and writes

* ``analysis/summary.json``   -- machine-readable summary
* ``analysis/summary_table.md`` -- the first-page table of REPORT.md
* ``analysis/curves.csv``     -- long-format curves for all available models
* ``analysis/curves.png``     -- Validation / monitor / train-L1 / step-ms panels

No training, no evaluation, no Test access; this only aggregates recorded
numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import ew_common as C  # noqa: E402

FIXED_EPOCHS = (20, 30, 35, 40)
LAST_N = 5


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_curve(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as fh:
        raw = list(csv.DictReader(fh))
    rows: list[dict[str, Any]] = []
    for r in raw:
        row: dict[str, Any] = {}
        for k, v in r.items():
            f = _float(v)
            row[k] = f if not (isinstance(v, str) and _is_text(v)) else v
        rows.append(row)
    return sorted(rows, key=lambda r: int(r["epoch"]))


def _is_text(value: str) -> bool:
    try:
        float(value)
        return False
    except ValueError:
        return True


def summarize_run(run: str, curve: list[dict[str, Any]], meta: Mapping[str, Any] | None) -> dict[str, Any]:
    meta = dict(meta or {})
    if not curve:
        return {"run": run, "available": False}
    by_epoch = {int(r["epoch"]): r for r in curve}
    valid = [r["valid_mae_eV"] for r in curve]
    best_idx = int(min(range(len(valid)), key=lambda i: valid[i]))
    best = curve[best_idx]
    last = curve[-1]
    last_n = curve[-LAST_N:]
    fixed = {
        str(e): {
            "valid_mae_eV": by_epoch[e]["valid_mae_eV"],
            "monitor_mae_eV": by_epoch[e]["monitor_mae_eV"],
            "train_l1_z": by_epoch[e]["train_l1_z"],
        }
        for e in FIXED_EPOCHS
        if e in by_epoch
    }
    train_seconds = _float(meta.get("train_seconds"))
    if not math.isfinite(train_seconds) or train_seconds <= 0:
        train_seconds = float(sum(r.get("epoch_train_seconds", 0.0) for r in curve))
    steps = int(last["global_step"])
    molecules = float(steps * C.BATCH_SIZE)
    gpu_hours = _float(meta.get("gpu_hours"))
    if not math.isfinite(gpu_hours) or gpu_hours <= 0:
        wall = _float(meta.get("wall_seconds"))
        gpu_hours = wall / 3600.0 if math.isfinite(wall) and wall > 0 else float("nan")
    eval_seconds = float(sum(r.get("epoch_valid_seconds", 0.0) + r.get("epoch_monitor_seconds", 0.0) for r in curve))
    ckpt_seconds = float(sum(r.get("checkpoint_seconds", 0.0) for r in curve))
    return {
        "run": run,
        "available": True,
        "model": meta.get("model") or C.RUNS_TABLE.get(run, {}).get("model"),
        "scale": int(meta.get("scale") or C.RUNS_TABLE.get(run, {}).get("scale", 0)) or None,
        "parameters": int(meta.get("parameters") or C.EXPECTED_PARAMETERS.get(run, 0)) or None,
        "epochs": len(curve),
        "best_epoch": int(best["epoch"]),
        "best_valid_mae_eV": float(best["valid_mae_eV"]),
        "final_valid_mae_eV": float(last["valid_mae_eV"]),
        "final_monitor_mae_eV": float(last["monitor_mae_eV"]),
        "final_train_l1_z": float(last["train_l1_z"]),
        "fixed_epochs": fixed,
        "last5_valid_mean": float(sum(r["valid_mae_eV"] for r in last_n) / len(last_n)),
        "last5_monitor_mean": float(sum(r["monitor_mae_eV"] for r in last_n) / len(last_n)),
        "monitor_minus_valid_eV": float(last["monitor_mae_eV"] - last["valid_mae_eV"]),
        "late_deltas_eV": {
            "valid_40_minus_35": float(
                by_epoch[40]["valid_mae_eV"] - by_epoch[35]["valid_mae_eV"] if 40 in by_epoch and 35 in by_epoch else float("nan")
            ),
            "valid_35_minus_30": float(
                by_epoch[35]["valid_mae_eV"] - by_epoch[30]["valid_mae_eV"] if 35 in by_epoch and 30 in by_epoch else float("nan")
            ),
            "valid_30_minus_20": float(
                by_epoch[30]["valid_mae_eV"] - by_epoch[20]["valid_mae_eV"] if 30 in by_epoch and 20 in by_epoch else float("nan")
            ),
        },
        "train_seconds": train_seconds,
        "eval_seconds": eval_seconds,
        "checkpoint_seconds": ckpt_seconds,
        "steps": steps,
        "molecules": molecules,
        "molecules_per_second_train_only": float(molecules / train_seconds) if train_seconds > 0 else float("nan"),
        "ms_per_step_train_only": float(1000.0 * train_seconds / steps) if steps > 0 else float("nan"),
        "gpu_hours": gpu_hours,
        "gpu_seconds": _float(meta.get("gpu_seconds")),
        "git_commit": meta.get("git_commit"),
        "curve": [
            {
                "epoch": int(r["epoch"]),
                "valid_mae_eV": r["valid_mae_eV"],
                "monitor_mae_eV": r["monitor_mae_eV"],
                "train_l1_z": r["train_l1_z"],
                "epoch_seconds": r.get("epoch_seconds"),
                "epoch_train_seconds": r.get("epoch_train_seconds"),
                "step_ms_mean": r.get("step_ms_mean"),
            }
            for r in curve
        ],
    }


def projection(rows: Mapping[str, Mapping[str, Any]], *, full_rows: int = C.FULL_TRAIN_ROWS) -> dict[str, Any]:
    steps_full = int(math.ceil(full_rows / C.BATCH_SIZE))
    epochs = int(C.RECIPE["epochs"])
    total_steps = steps_full * epochs
    out: dict[str, Any] = {
        "full_rows": int(full_rows),
        "batch_size": int(C.BATCH_SIZE),
        "steps_per_epoch": steps_full,
        "total_steps": total_steps,
        "total_molecules": float(steps_full * C.BATCH_SIZE * epochs),
        "assumption": (
            "per-step training time measured on the 1M subset is held constant; "
            "Validation cost is the measured full-split cost per epoch; first-epoch "
            "page-cache warmup is included in the measured per-step time"
        ),
        "widths": {},
    }
    for name, row in rows.items():
        if not row.get("available"):
            continue
        ms = float(row["ms_per_step_train_only"])
        per_epoch_eval = float(row["eval_seconds"]) / max(int(row["epochs"]), 1)
        per_epoch_ckpt = float(row["checkpoint_seconds"]) / max(int(row["epochs"]), 1)
        train_hours = total_steps * ms / 3.6e6
        eval_hours = epochs * (per_epoch_eval + per_epoch_ckpt) / 3600.0
        out["widths"][name] = {
            "parameters": row.get("parameters"),
            "measured_ms_per_step_train_only": ms,
            "measured_molecules_per_second": row["molecules_per_second_train_only"],
            "projected_train_hours": train_hours,
            "projected_eval_hours": eval_hours,
            "projected_total_hours": train_hours + eval_hours,
            "projected_gpu_hours": train_hours + eval_hours,
        }
    # frozen-path reference at width 2 from the interleaved benchmark (if present)
    bench_path = C.ANALYSIS / "bench.json"
    if bench_path.exists():
        bench = json.loads(bench_path.read_text(encoding="utf-8"))
        agg = bench.get("aggregates", {})
        for key in ("frozen", "E1"):
            if key in agg:
                mols = float(agg[key]["molecules_per_second_median"])
                train_hours = (steps_full * epochs * C.BATCH_SIZE) / mols / 3600.0
                out.setdefault("bench_reference", {})[key] = {
                    "molecules_per_second": mols,
                    "projected_train_hours": train_hours,
                }
    return out


def write_curves_csv(rows: Mapping[str, Mapping[str, Any]], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "epoch", "valid_mae_eV", "monitor_mae_eV", "train_l1_z", "epoch_seconds", "step_ms_mean"])
        for name, row in rows.items():
            if not row.get("available"):
                continue
            for r in row["curve"]:
                w.writerow(
                    [
                        name,
                        r["epoch"],
                        r["valid_mae_eV"],
                        r["monitor_mae_eV"],
                        r["train_l1_z"],
                        r.get("epoch_seconds"),
                        r.get("step_ms_mean"),
                    ]
                )


def write_table(rows: Mapping[str, Mapping[str, Any]], path: Path) -> None:
    order = ["W1", "W2", "W3"]
    lines = [
        "| model | parameters | best Valid MAE (eV) @ epoch | Valid@40 | monitor@40 | GPU-hours | molecules/s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in order:
        row = rows.get(name, {})
        if not row.get("available"):
            lines.append(f"| {name} | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {name} | {row['parameters']:,} | {row['best_valid_mae_eV']:.9f} @ {row['best_epoch']} | "
            f"{row['final_valid_mae_eV']:.9f} | {row['final_monitor_mae_eV']:.9f} | "
            f"{row['gpu_hours']:.3f} | {row['molecules_per_second_train_only']:.0f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(rows: Mapping[str, Mapping[str, Any]], path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    available = [n for n in ("W1", "W2", "W3") if rows.get(n, {}).get("available")]
    if not available:
        return False
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), dpi=150)
    colors = {"W1": "#1f77b4", "W2": "#444444", "W3": "#d62728"}
    for name in available:
        c = rows[name]["curve"]
        e = [r["epoch"] for r in c]
        for ax, key, title in (
            (axes[0, 0], "valid_mae_eV", "Validation MAE (full 73,545)"),
            (axes[0, 1], "monitor_mae_eV", "Train-monitor MAE (20k)"),
            (axes[1, 0], "train_l1_z", "Train L1 (standardised)"),
            (axes[1, 1], "step_ms_mean", "ms/step (training only)"),
        ):
            ax.plot(e, [r[key] for r in c], label=name, color=colors[name], lw=1.4)
            ax.set_title(title)
            ax.set_xlabel("epoch")
            ax.grid(alpha=0.25)
        axes[0, 1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(C.ANALYSIS / "summary.json"))
    args = parser.parse_args(argv)
    rows: dict[str, Any] = {}
    hist_dir = C.OLD_ROUND / "runs" / C.HISTORICAL["run"]
    hist_curve = load_curve(hist_dir / "train_curve.csv")
    hist_meta = json.loads((hist_dir / "run.json").read_text(encoding="utf-8")) if (hist_dir / "run.json").exists() else {}
    rows["W2"] = summarize_run(C.HISTORICAL["run"], hist_curve, hist_meta)
    for run in C.RUN_ORDER:
        d = C.RUNS / run
        meta_path = d / "run.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        rows[C.RUNS_TABLE[run]["model"]] = summarize_run(run, load_curve(d / "train_curve.csv"), meta)
    payload = {
        "schema": "lcrc-pcqm4mv2-efficiency-width-summary-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "rows": rows,
        "projection": projection(rows),
        "decision": decide(rows),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    C.write_json(out, payload)
    write_curves_csv(rows, out.parent / "curves.csv")
    write_table(rows, out.parent / "summary_table.md")
    plotted = write_plot(rows, out.parent / "curves.png")
    C.log(
        f"[analysis] W2 available={rows['W2'].get('available')} W1={rows['W1'].get('available')} "
        f"W3={rows['W3'].get('available')} plot={plotted} -> {C.track_rel(out)}"
    )
    print(json.dumps(payload["decision"], indent=1))
    return 0


def decide(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    w1, w2, w3 = rows.get("W1", {}), rows.get("W2", {}), rows.get("W3", {})
    out: dict[str, Any] = {
        "rule": "PROMISING_WIDTH_SIGNAL iff W3 Valid@40 improves over W2 by >= 0.005 eV with a consistent late trend",
        "valid40_delta_W3_minus_W2_eV": None,
        "valid40_delta_W1_minus_W2_eV": None,
        "w1_redundant_with_w2": None,
        "status": "INCONCLUSIVE",
        "reasons": [],
    }
    if not (w2.get("available") and w3.get("available")):
        out["reasons"].append("W3 (or W2 historical) curve missing")
        return out
    d32 = float(w3["final_valid_mae_eV"]) - float(w2["final_valid_mae_eV"])
    out["valid40_delta_W3_minus_W2_eV"] = d32
    late = w3.get("late_deltas_eV", {})
    consistent = (
        math.isfinite(late.get("valid_40_minus_35", float("nan")))
        and math.isfinite(late.get("valid_35_minus_30", float("nan")))
        and late["valid_40_minus_35"] <= 0
        and late["valid_35_minus_30"] <= 0
        and w3["last5_valid_mean"] <= w2["last5_valid_mean"]
    )
    out["late_trend_consistent_in_W3"] = bool(consistent)
    out["w3_last5_valid_mean"] = w3["last5_valid_mean"]
    out["w2_last5_valid_mean"] = w2["last5_valid_mean"]
    if d32 <= -0.005 and consistent:
        out["status"] = "PROMISING_WIDTH_SIGNAL"
    else:
        out["status"] = "NO_WIDTH_SIGNAL"
        out["reasons"].append(
            "W3 does not improve Valid@40 by >=0.005 eV with a consistent late trend"
            if d32 > -0.005
            else "Valid@40 improvement exists but the late trend is not consistent"
        )
    if w1.get("available"):
        d12 = float(w1["final_valid_mae_eV"]) - float(w2["final_valid_mae_eV"])
        out["valid40_delta_W1_minus_W2_eV"] = d12
        # "W1 redundant with W2" means W1 is not materially better AND not materially worse
        out["w1_redundant_with_w2"] = bool(abs(d12) < 0.005)
    else:
        # W1 missing limits the redundancy question only; the width decision is
        # W3 vs the historical W2 and does not depend on W1.
        out["reasons"].append("W1 not run (budget); redundancy of W1 with W2 unanswered")
        out["w1_status"] = "MISSING"
    out["w1_status"] = out.get("w1_status", "AVAILABLE" if w1.get("available") else "MISSING")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
