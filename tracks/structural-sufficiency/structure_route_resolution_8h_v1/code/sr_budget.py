"""Allocated-GPU-hour ledger and wall-clock guard for the sprint."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import sr_common as C

T0 = datetime(2026, 11, 23, 15, 23, 1, tzinfo=timezone.utc)
WALL_TARGET = 7.5
WALL_HARD = 8.0
GPU_TARGET = 14.0
GPU_HARD = 16.0


def main() -> int:
    cfg = C.load_config()
    now = datetime.now(timezone.utc)
    hours = (now - T0).total_seconds() / 3600.0
    runs = []
    total = 0.0
    for d in sorted(C.RESULTS_DIR.iterdir()) if C.RESULTS_DIR.is_dir() else []:
        mf = d / "manifest.json"
        if not mf.is_file():
            continue
        m = json.loads(mf.read_text(encoding="utf-8"))
        gpu_h = float(m.get("gpu_hours_total_process",
                            float(m.get("wall_seconds", 0.0)) / 3600.0))
        total += gpu_h
        runs.append({"run_id": m.get("run_id"), "arm": m.get("arm"),
                     "split": m.get("split"), "seed": m.get("seed"),
                     "state": "completed", "gpu_hours": gpu_h,
                     "best_epoch": m.get("best_epoch"),
                     "dev_rmse": m.get("dev_rmse")})
    payload: dict[str, Any] = {
        "t0_utc": T0.isoformat(), "now_utc": now.isoformat(),
        "wall_hours_elapsed": hours,
        "wall_target_hours": WALL_TARGET, "wall_hard_hours": WALL_HARD,
        "gpu_target_hours": GPU_TARGET, "gpu_hard_hours": GPU_HARD,
        "gpu_hours_allocated_completed": total,
        "gpu_hours_remaining_to_target": GPU_TARGET - total,
        "gpu_hours_remaining_to_hard": GPU_HARD - total,
        "stop_new_optional_utc": (T0 + timedelta(hours=6.5)).isoformat(),
        "finish_training_utc": (T0 + timedelta(hours=7.25)).isoformat(),
        "runs": runs,
        "budget": cfg["budget"],
    }
    C.write_json(C.ANALYSIS_DIR / "budget.json", payload)
    print(json.dumps({k: v for k, v in payload.items() if k != "runs"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())