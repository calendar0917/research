"""GPU-hour ledger for the v3 pilot only.

Every executed v3 run is charged its measured process wall time plus the same
8 s startup allowance used by the v2 ledger.  Historical v1/v2 hours are
reported separately and are not part of the 0.30 h v3 cap.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import v3_common as C
import v3_registry as R


def _rr_wall(run_dir: Path) -> float | None:
    meta = run_dir / "meta.json"
    if not meta.exists():
        return None
    try:
        payload = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:
        return None
    for key in ("wall_seconds", "duration_seconds", "elapsed_seconds"):
        if payload.get(key) is not None:
            try:
                return float(payload[key])
            except Exception:
                pass
    return None


def rebuild_ledger() -> dict[str, Any]:
    entries = []
    total = 0.0
    for row in R.planned_registry()["runs"]:
        key = row["key"]
        run_dir = C.RUNS / key
        train_path = run_dir / "train.json"
        if not train_path.exists():
            continue
        payload = C.read_json(train_path)
        measured = float(payload.get("wall_seconds", 0.0)) + 8.0
        rr_wall = _rr_wall(run_dir)
        charged = max(measured, rr_wall) if rr_wall is not None else measured
        hours = charged / 3600.0
        total += hours
        entries.append(
            {
                "key": key,
                "arm": row["arm"],
                "fold": row["fold"],
                "seed": row["seed"],
                "recipe": row["recipe"],
                "conditional": row["conditional"],
                "measured_process_seconds": float(payload.get("wall_seconds", 0.0)),
                "rr_wall_seconds": rr_wall,
                "charged_seconds": charged,
                "charged_gpu_hours": hours,
                "cuda_device_name": payload.get("cuda_device_name"),
                "torch": payload.get("torch"),
                "git_commit": payload.get("git_commit"),
                "file": C.track_rel(train_path),
            }
        )
    if not entries:
        status = "NO_GPU_RUNS"
    elif total <= C.GPU_HARD_CAP_HOURS:
        status = "WITHIN_CAP"
    else:
        status = "CAP_EXCEEDED"
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "hard_cap_gpu_hours": C.GPU_HARD_CAP_HOURS,
        "historical_v1_v2_hours_reference_only": C.HISTORICAL_V1_V2_HOURS_REFERENCE_ONLY,
        "n_jobs_charged": len(entries),
        "n_unconditional_executed": sum(1 for e in entries if not e["conditional"]),
        "n_conditional_executed": sum(1 for e in entries if e["conditional"]),
        "total_gpu_hours": total,
        "remaining_gpu_hours": C.GPU_HARD_CAP_HOURS - total,
        "entries": entries,
        "status": status,
    }
    C.write_json(C.ANALYSIS / "gpu_ledger_v3.json", payload)
    if total > C.GPU_HARD_CAP_HOURS:
        raise SystemExit(f"v3 GPU ledger exceeds the {C.GPU_HARD_CAP_HOURS} h cap: {total:.4f} h")
    return payload


__all__ = ["rebuild_ledger"]
