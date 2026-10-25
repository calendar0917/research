"""Reporting helpers for zinc-statistical-program-search-v1 (local, no GPU).

Reads run directories produced by sps_train.py, computes the frozen stable
measures (best valid / W20 / T100 / final / eval-train@best), and writes
analysis/summary.json + analysis/main_table.md.

W20   = min over 20-consecutive-epoch windows of the window median valid MAE
T100  = median valid MAE over the last 100 epochs of the run
GPUh  = wall seconds / 3600 (one A100 per run)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_AD = _HERE.parents[1] / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_AD)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402

SPS = _HERE.parent
REF = {
    "DCC-L s0 (historical)": 0.078537,
    "DCC-L 4-seed mean (historical)": 0.075990,
    "P3-Null s0 (historical)": 0.079105,
}


def window_min(curve: list[dict], width: int) -> tuple[float, int | None]:
    vals = [e["valid_mae"] for e in curve]
    epochs = [e["epoch"] for e in curve]
    if len(vals) < width:
        return float(np.median(vals)), int(epochs[0]) if epochs else None
    best = (float("inf"), None)
    for s in range(len(vals) - width + 1):
        m = float(np.median(vals[s:s + width]))
        if m < best[0]:
            best = (m, int(epochs[s]))
    return best


def summarize(run: Path) -> dict:
    info = C.read_json(run / "train.json")
    curve = info["curve"]
    vals = [e["valid_mae"] for e in curve]
    best = info["best_valid"]
    w20, w20_start = window_min(curve, 20)
    t100 = float(np.median(vals[-100:])) if len(vals) >= 100 else float(np.median(vals))
    best_i = int(np.argmin(vals))
    tail_start = max(0, best_i - 20)
    return {
        "run": run.name,
        "candidate": info["candidate"],
        "seed": info["seed"],
        "parameters": info["parameter_count"],
        "epochs_run": len(curve),
        "declared_epochs": info["recipe"]["epochs"],
        "best_valid": float(best["valid_mae"]),
        "best_epoch": int(best["epoch"]),
        "final_valid": float(vals[-1]),
        "w20": float(w20),
        "w20_start_epoch": w20_start,
        "t100": t100,
        "best_minus_w20": float(best["valid_mae"]) - float(w20),
        "train_eval_mae_at_best": info["fixed_objects"]["best"]["train_mae_eval_mode"],
        "train_eval_mae_final": info["fixed_objects"]["final"]["train_mae_eval_mode"],
        "online_train_mae_final": float(curve[-1]["online_train_mae"]),
        "gpu_hours": info.get("gpu_hours"),
        "delta_vs_dcc_s0": float(best["valid_mae"]) - REF["DCC-L s0 (historical)"],
        "delta_vs_dcc_mean": float(best["valid_mae"]) - REF["DCC-L 4-seed mean (historical)"],
        "eval_mode_train_curve_last20_median": float(
            np.median([e["online_train_mae"] for e in curve[-20:]])
        ),
        "sanity": {
            "gradient_audit": info["gradient_audit"],
            "parameter_audit": info["parameter_audit"],
            "round": info.get("round"),
            "device": info.get("device"),
            "official_test_loaded": info["data"]["official_test_loaded"],
            "ssl_sha256": (info.get("model_info", {}).get("ssl_sha256")),
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default=str(SPS / "analysis"))
    args = ap.parse_args(argv)
    rows = [summarize(Path(r)) for r in args.runs]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    C.write_json(out / "summary.json", {"reference": REF, "runs": rows})
    lines = [
        "| run | params | best valid | ep | W20 (start) | T100 | final | best−W20 | train@best | GPU·h | Δ vs DCC-L s0 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['run']} | {r['parameters']:,} | {r['best_valid']:.6f} | {r['best_epoch']} | "
            f"{r['w20']:.6f} ({r['w20_start_epoch']}) | {r['t100']:.6f} | {r['final_valid']:.6f} | "
            f"{r['best_minus_w20']:+.6f} | {r['train_eval_mae_at_best']:.6f} | "
            f"{r['gpu_hours'] if r['gpu_hours'] else float('nan'):.3f} | {r['delta_vs_dcc_s0']:+.6f} |"
        )
    (out / "main_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
