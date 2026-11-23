"""Read the 2-epoch GPU preflight and project the frozen 240-epoch budget."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

import wobr_common as C

PREFLIGHT_DIR = C.RESULTS_DIR / "preflight_seed0"
MAX_FORMAL_GPU_HOURS = 2.0
MAX_TOTAL_GPU_HOURS = 2.5


def main() -> dict[str, Any]:
    run = C.read_json(PREFLIGHT_DIR / "run.json")
    with (PREFLIGHT_DIR / "curves.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("preflight curves are empty")
    epoch_seconds = np.asarray([float(r["epoch_seconds"]) for r in rows], np.float64)
    eval_seconds = np.asarray([float(r["eval_seconds"]) for r in rows], np.float64)
    mean_epoch = float(epoch_seconds.mean())
    projected_train_seconds = mean_epoch * 240.0
    # Be conservative: the formal process also pays structure construction,
    # startup and the final full-split prediction pass.  Use the preflight's
    # non-training wall time as the projection for that overhead.
    projected_nontrain_seconds = max(0.0, float(run["wall_seconds"]) -
                                     float(run["train_seconds"]))
    projected_formal_process_seconds = projected_train_seconds + projected_nontrain_seconds
    projected_gpu_hours = projected_train_seconds / 3600.0
    projected_formal_gpu_hours = projected_formal_process_seconds / 3600.0
    actual_preflight_gpu_hours = float(run["gpu_hours"])
    actual_preflight_total_hours = float(run.get("gpu_hours_total_process",
                                                 run["gpu_hours"]))
    projected_total_with_preflight = (projected_formal_gpu_hours
                                      + actual_preflight_total_hours)
    budget_ok = bool(projected_formal_gpu_hours <= MAX_FORMAL_GPU_HOURS
                     and projected_total_with_preflight <= MAX_TOTAL_GPU_HOURS)
    try:
        preflight_run = str((PREFLIGHT_DIR / "run.json").relative_to(C.REPO_ROOT))
    except ValueError:
        preflight_run = str(PREFLIGHT_DIR / "run.json")
    payload = {
        "preflight_run": preflight_run,
        "git_commit": run.get("git_commit"),
        "host": run.get("host"),
        "gpu_name": run.get("gpu_name"),
        "device": run.get("device"),
        "torch": run.get("torch"),
        "cuda": run.get("cuda"),
        "epochs_run": int(run["epochs_run"]),
        "preflight_train_seconds": float(run["train_seconds"]),
        "preflight_wall_seconds": float(run["wall_seconds"]),
        "preflight_gpu_hours": actual_preflight_gpu_hours,
        "preflight_total_process_gpu_hours": actual_preflight_total_hours,
        "peak_vram_bytes": int(run.get("peak_vram_bytes", 0)),
        "peak_vram_gb": float(run.get("peak_vram_bytes", 0)) / (1024 ** 3),
        "epoch_seconds": epoch_seconds.tolist(),
        "eval_seconds": eval_seconds.tolist(),
        "mean_epoch_seconds": mean_epoch,
        "projected_240_epoch_train_seconds": projected_train_seconds,
        "projected_nontrain_seconds": projected_nontrain_seconds,
        "projected_240_epoch_gpu_hours": projected_gpu_hours,
        "projected_formal_process_gpu_hours": projected_formal_gpu_hours,
        "projected_total_with_preflight_gpu_hours": projected_total_with_preflight,
        "formal_gpu_budget_hours": MAX_FORMAL_GPU_HOURS,
        "total_gpu_budget_with_preflight_hours": MAX_TOTAL_GPU_HOURS,
        "budget_ok": budget_ok,
        "candidate_stats": run.get("structure_stats"),
        "best_preflight_valid_rmse": float(run.get("best_valid_rmse", 0.0)),
        "preflight_valid_rmse_epochs": [float(r["valid_rmse"]) for r in rows],
        "preflight_train_rmse_epochs": [float(r["train_rmse"]) for r in rows],
        "preflight_test_rmse_epochs": [float(r["test_rmse"]) for r in rows],
        "no_config_change": True,
    }
    C.write_json(C.ANALYSIS_DIR / "preflight_gpu.json", payload)
    print(json.dumps(C.jsonable(payload), indent=1, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()