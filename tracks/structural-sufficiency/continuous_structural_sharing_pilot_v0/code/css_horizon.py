"""Frozen horizon decision for CSS-v0.

Reads the two 3-epoch timing runs (full three-split evaluation each epoch) and
applies the pre-registered rule from ``PROTOCOL.md`` §7: choose the largest
common horizon ``H`` in {120, 90, 60} whose measured per-run wall and total GPU
cost, with a 25% timing margin, fit the 10-minute per-run target and the
0.5 GPU-hour total.  Never reads validation performance.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

import css_common as C

ALLOWED = (120, 90, 60)
PER_RUN_TARGET_SECONDS = 600.0
TOTAL_GPU_SECONDS = 1800.0
MARGIN = 1.25


def _load_runs(results_dir: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for path in sorted(results_dir.glob("*/manifest.json")):
        payload = C.read_json(path)
        if payload.get("format") != "css-v0-run-v1":
            continue
        if "timing" not in str(payload.get("run_id", "")):
            continue
        size = int(payload["size"])
        out[size] = {"manifest": payload, "run_dir": str(path.parent)}
    if set(out) != {100, 50}:
        raise ValueError(f"need timing runs for sizes 100 and 50, got {sorted(out)}")
    return out


def main(results_dir: str = str(C.RESULTS_DIR), out: str | None = None) -> dict[str, Any]:
    runs = _load_runs(Path(results_dir))
    timing: dict[int, dict[str, float]] = {}
    for size, payload in runs.items():
        manifest = payload["manifest"]
        import csv

        with (Path(payload["run_dir"]) / "curves.csv").open(newline="", encoding="utf-8") as handle:
            curves = list(csv.DictReader(handle))
        epoch_seconds = [float(row["epoch_seconds"]) for row in curves]
        # exclude the first epoch (CUDA init / allocator warm-up)
        steady = epoch_seconds[1:] if len(epoch_seconds) > 1 else epoch_seconds
        train_seconds = float(manifest["train_seconds"])
        overhead = max(0.0, float(manifest["wall_seconds"]) - train_seconds)
        timing[size] = {
            "n_epochs": len(epoch_seconds),
            "epoch_seconds": epoch_seconds,
            "steady_epoch_seconds": float(np.mean(steady)),
            "measured_epoch_min": float(np.min(epoch_seconds)),
            "measured_epoch_max": float(np.max(epoch_seconds)),
            "startup_overhead_seconds": overhead,
            "train_seconds": train_seconds,
            "wall_seconds": float(manifest["wall_seconds"]),
            "run_id": manifest["run_id"],
            "commit": manifest.get("git_commit"),
            "gpu_name": manifest.get("gpu_name"),
            "device": manifest.get("device"),
        }
    t100 = timing[100]["steady_epoch_seconds"]
    t50 = timing[50]["steady_epoch_seconds"]
    o100 = timing[100]["startup_overhead_seconds"]
    o50 = timing[50]["startup_overhead_seconds"]
    candidates = []
    for horizon in ALLOWED:
        run_100 = horizon * t100 + o100
        run_50 = horizon * t50 + o50
        total = 2.0 * run_100 + 2.0 * run_50
        ok = (MARGIN * run_100 <= PER_RUN_TARGET_SECONDS
              and MARGIN * run_50 <= PER_RUN_TARGET_SECONDS
              and MARGIN * total <= TOTAL_GPU_SECONDS)
        candidates.append({
            "horizon": int(horizon),
            "projected_run_100_seconds": run_100,
            "projected_run_50_seconds": run_50,
            "projected_run_100_seconds_with_margin": MARGIN * run_100,
            "projected_run_50_seconds_with_margin": MARGIN * run_50,
            "projected_total_gpu_seconds": total,
            "projected_total_gpu_hours": total / 3600.0,
            "projected_total_with_margin_gpu_hours": MARGIN * total / 3600.0,
            "per_run_ok": bool(MARGIN * run_100 <= PER_RUN_TARGET_SECONDS
                               and MARGIN * run_50 <= PER_RUN_TARGET_SECONDS),
            "total_ok": bool(MARGIN * total <= TOTAL_GPU_SECONDS),
            "feasible": bool(ok),
        })
    feasible = [c for c in candidates if c["feasible"]]
    horizon = int(feasible[0]["horizon"]) if feasible else None
    decision = {
        "format": "css-v0-horizon-v1",
        "rule": "largest H in {120,90,60} with a 25% margin inside 600 s per run and 1800 s total GPU",
        "margin": MARGIN,
        "per_run_target_seconds": PER_RUN_TARGET_SECONDS,
        "total_gpu_seconds": TOTAL_GPU_SECONDS,
        "timing_runs": timing,
        "candidates": candidates,
        "chosen_horizon": horizon,
        "budget_limited": horizon is None,
    }
    out_path = Path(out) if out else (C.RESULTS_DIR / "horizon.json")
    C.write_json(out_path, decision)
    print(f"horizon decision: {horizon} epochs -> {out_path}", flush=True)
    for c in candidates:
        print(f"  H={c['horizon']:3d} run100={c['projected_run_100_seconds']:.1f}s "
              f"run50={c['projected_run_50_seconds']:.1f}s "
              f"total={c['projected_total_gpu_hours']:.3f} GPUh "
              f"(margin {c['projected_total_with_margin_gpu_hours']:.3f}) "
              f"feasible={c['feasible']}", flush=True)
    return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(C.RESULTS_DIR))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    main(args.results_dir, args.out)
