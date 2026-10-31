"""GPU budget ledger for this round (rr status JSON + in-process timings)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import be_common as C


def _seconds(started: str | None, completed: str | None) -> float | None:
    if not started or not completed:
        return None
    try:
        return (datetime.fromisoformat(completed) - datetime.fromisoformat(started)).total_seconds()
    except Exception:
        return None


def _status_records() -> list[dict[str, Any]]:
    records = []
    if not C.RR_STATUS.exists():
        return records
    for path in sorted(C.RR_STATUS.glob("*.json")):
        status = C.read_json(path)
        seconds = _seconds(status.get("started_at"), status.get("completed_at"))
        meta = status.get("meta") or {}
        records.append(
            {
                "experiment": status.get("experiment", path.stem),
                "run_id": status.get("run_id"),
                "state": status.get("state"),
                "host": status.get("host"),
                "node": status.get("node"),
                "pool": status.get("pool"),
                "cuda_visible_devices": status.get("cuda_visible_devices"),
                "gpu_names": status.get("gpu_names"),
                "driver_version": status.get("driver_version"),
                "torch_version": status.get("torch_version"),
                "torch_cuda": status.get("torch_cuda"),
                "commit": status.get("commit"),
                "dirty": status.get("dirty"),
                "created_at": status.get("created_at"),
                "started_at": status.get("started_at"),
                "completed_at": status.get("completed_at"),
                "exit_code": status.get("exit_code"),
                "failure_reason": status.get("failure_reason"),
                "command": (meta.get("command") if isinstance(meta, dict) else None),
                "gpu_seconds": seconds,
                "gpu_hours_charged": (seconds / 3600.0) if seconds is not None else None,
                "source_json": C.track_rel(path),
            }
        )
    return records


def _in_process() -> list[dict[str, Any]]:
    out = []
    if not C.RUNS.exists():
        return out
    for train_json in sorted(C.RUNS.glob("*/train.json")):
        payload = C.read_json(train_json)
        out.append(
            {
                "run_key": train_json.parent.name,
                "arm": payload.get("arm"),
                "family": payload.get("family"),
                "seed": payload.get("seed"),
                "best_epoch": payload.get("checkpoint", {}).get("best_epoch"),
                "device": payload.get("device"),
                "hostname": payload.get("hostname"),
                "cuda_device_name": payload.get("cuda_device_name"),
                "train_wall_seconds": payload.get("timing", {}).get("train_wall_seconds"),
                "process_wall_seconds": payload.get("timing", {}).get("process_wall_seconds"),
                "optimizer_steps": payload.get("recipe", {}).get("optimizer_steps"),
                "epochs_run": payload.get("recipe", {}).get("epochs_run"),
                "grad_clip_events": payload.get("anomalies", {}).get("grad_clip_events"),
                "train_json": C.track_rel(train_json),
            }
        )
    return out


def build_ledger() -> dict[str, Any]:
    records = _status_records()
    total_seconds = float(sum(r["gpu_seconds"] or 0.0 for r in records))
    total_hours = total_seconds / 3600.0
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "hard_cap_gpu_hours": C.GPU_HARD_CAP_HOURS,
        "charging_rule": (
            "full rr-measured process lifetime of every CUDA process of this round "
            "(smoke, formal training, checkpoint verification, Test420 evaluation); "
            "both pools summed, not wall-clocked"
        ),
        "runs": records,
        "n_runs": len(records),
        "total_gpu_seconds": total_seconds,
        "total_gpu_hours": total_hours,
        "within_cap": bool(total_hours <= C.GPU_HARD_CAP_HOURS),
        "remaining_gpu_hours": max(0.0, C.GPU_HARD_CAP_HOURS - total_hours),
        "in_process_timings": _in_process(),
        "gpu_models": sorted({name for r in records for name in (r.get("gpu_names") or []) if name}),
        "failures": [r for r in records if r.get("state") != "completed"],
    }
    C.write_json(C.GPU_LEDGER, payload)
    return payload


__all__ = ["build_ledger"]
