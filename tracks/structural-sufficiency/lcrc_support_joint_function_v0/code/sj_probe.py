"""Formal-training throughput probe (before any formal run).

One short probe per task measures the *full* three-split evaluation cost with the
formal batch size and data, extrapolates the whole queue, and refuses to start if
the frozen budget cannot hold the complete control set.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

import sj_common as C
import sj_train as T


def probe(task_key: str, arm: str, device, epochs: int = 2) -> dict[str, Any]:
    out = C.RUNS / "smoke" / f"probe_{task_key}_{arm}_s{C.SEED}"
    if out.exists():
        import shutil

        shutil.rmtree(out)
    record = T.train_unit(task_key, arm, C.SEED, out, device, epochs=epochs, smoke=True, force=True)
    import csv

    with open(record["curve"]["path"], newline="", encoding="utf-8") as handle:
        curve = list(csv.DictReader(handle))
    train_s = [float(r["train_seconds"]) for r in curve]
    eval_s = [float(r["eval_seconds"]) for r in curve]
    warm = 1 if len(curve) > 1 else 0
    mean_train = sum(train_s[warm:]) / max(len(train_s) - warm, 1)
    mean_eval = sum(eval_s[warm:]) / max(len(eval_s) - warm, 1)
    n_epochs = int(C.TASKS[task_key]["epochs"])
    return {
        "task": task_key,
        "arm": arm,
        "epochs_probed": int(len(curve)),
        "mean_train_seconds_per_epoch": float(mean_train),
        "mean_full_three_split_eval_seconds_per_epoch": float(mean_eval),
        "mean_epoch_seconds": float(mean_train + mean_eval),
        "planned_epochs": n_epochs,
        "projected_seconds_for_planned_epochs": float((mean_train + mean_eval) * n_epochs),
        "gpu_peak_allocated_bytes": int(record["timing"]["gpu_peak_allocated_bytes"]),
        "device": str(device),
        "smoke_run": C.track_rel(out),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="throughput probe for the formal queue")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args(argv)
    device = torch.device(args.device)
    payload: dict[str, Any] = {
        "schema": "lcrc-support-joint-function-throughput-v1",
        "round_id": C.ROUND_ID,
        "git_commit": C.git_commit(),
        "created_utc": C.utcnow(),
        "execution_regime": C.EXECUTION_REGIME,
        "env": C.env_info(),
        "probes": {},
    }
    for task_key in C.TASK_ORDER:
        payload["probes"][task_key] = probe(task_key, C.ARM_B, device, epochs=args.epochs)
        C.log(f"probe {task_key}: {payload['probes'][task_key]['mean_epoch_seconds']:.2f}s/epoch")
    projected = 0.0
    for task_key, entry in payload["probes"].items():
        projected += 2.0 * entry["projected_seconds_for_planned_epochs"]  # B and C per task
    payload["queue"] = {
        "units": [f"{t}-{a}" for t in C.TASK_ORDER for a in C.TRAINED_ARMS],
        "projected_gpu_seconds_sequential": float(projected),
        "projected_gpu_hours_sequential": float(projected / 3600.0),
        "gpu_soft_cap_hours": C.GPU_SOFT_CAP_HOURS,
        "gpu_hard_cap_hours": C.GPU_HARD_CAP_HOURS,
        "wall_clock_hard_cap_hours": C.WALL_CLOCK_HARD_CAP_HOURS,
        "gpus_used": 1,
        "note": "sequential on GPU1 only (GPU0 belongs to another round's job); wall clock ~= GPU time",
    }
    payload["budget_ok"] = bool(
        payload["queue"]["projected_gpu_hours_sequential"] <= C.GPU_HARD_CAP_HOURS
        and payload["queue"]["projected_gpu_hours_sequential"] <= C.WALL_CLOCK_HARD_CAP_HOURS
    )
    C.write_json(C.ANALYSIS / "throughput_probe.json", payload)
    C.log(f"projected {payload['queue']['projected_gpu_hours_sequential']:.3f} GPU-h; budget_ok={payload['budget_ok']}")
    return 0 if payload["budget_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
