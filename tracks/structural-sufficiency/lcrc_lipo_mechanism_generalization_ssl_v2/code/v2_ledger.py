"""GPU-hour ledger for the v2 round.

Every registered GPU job is charged its full rr wall time when available,
otherwise the training/SSL wall time plus a fixed startup allowance.  The v1
0.5539 h history is reported separately and is not part of this cap.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import v2_common as C
import v2_registry as R


def _rr_wall(run_dir: Path) -> float | None:
    meta = run_dir / "meta.json"
    if not meta.exists():
        return None
    try:
        payload = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:
        return None
    for key in ("wall_seconds", "duration_seconds", "elapsed_seconds"):
        if key in payload and payload[key] is not None:
            try:
                return float(payload[key])
            except Exception:
                pass
    started = payload.get("started_at") or payload.get("start_time")
    finished = payload.get("finished_at") or payload.get("end_time")
    if started and finished:
        from datetime import datetime

        try:
            a = datetime.fromisoformat(str(started).replace("Z", "+00:00"))
            b = datetime.fromisoformat(str(finished).replace("Z", "+00:00"))
            return (b - a).total_seconds()
        except Exception:
            return None
    return None


def rebuild_ledger() -> dict[str, Any]:
    entries = []
    total = 0.0
    for row in R.planned_registry()["runs"]:
        key = row["key"]
        run_dir = C.RUNS / key
        payload = None
        for name in ("train.json", "ssl.json"):
            if (run_dir / name).exists():
                payload = C.read_json(run_dir / name)
                break
        if payload is None:
            continue
        measured = float(payload.get("wall_seconds", 0.0)) + 8.0
        rr_wall = _rr_wall(run_dir)
        charged = max(measured, rr_wall) if rr_wall is not None else measured
        hours = charged / 3600.0
        total += hours
        entries.append(
            {
                "key": key,
                "kind": row["kind"],
                "fold": row["fold"],
                "recipe": payload.get("recipe"),
                "seed": row["seed"],
                "measured_process_seconds": float(payload.get("wall_seconds", 0.0)),
                "rr_wall_seconds": rr_wall,
                "charged_seconds": charged,
                "charged_gpu_hours": hours,
                "cuda_device_name": payload.get("cuda_device_name"),
                "git_commit": payload.get("git_commit"),
                "file": C.track_rel(run_dir / ("train.json" if row["kind"] == "train" else "ssl.json")),
            }
        )
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "hard_cap_gpu_hours": C.GPU_HARD_CAP_HOURS,
        "v1_historical_gpu_hours_reference_only": C.V1_HISTORICAL_GPU_HOURS,
        "n_jobs_charged": len(entries),
        "total_gpu_hours": total,
        "remaining_gpu_hours": C.GPU_HARD_CAP_HOURS - total,
        "entries": entries,
        "status": "WITHIN_CAP" if total <= C.GPU_HARD_CAP_HOURS else "CAP_EXCEEDED",
    }
    C.write_json(C.ANALYSIS / "gpu_ledger_v2.json", payload)
    if total > C.GPU_HARD_CAP_HOURS:
        raise SystemExit(f"v2 GPU ledger exceeds the {C.GPU_HARD_CAP_HOURS} h cap: {total:.4f} h")
    return payload


__all__ = ["rebuild_ledger"]
