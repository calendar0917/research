"""GPU budget ledger for the contextual-contribution round.

Charging rule (fixed before any run): the full rr-measured process lifetime of
every process that touched CUDA, including the CUDA smoke, data loading,
validation and any retry.  Both parallel GPU pools are summed, not wall-clocked.
CPU-only work is recorded separately at zero GPU-hours.

Sources:
* ``analysis/rr_status/*.json`` -- ``rr status --json`` payloads (authoritative
  process lifetime, host / node / driver / GPU model / commit);
* ``analysis/runs/*/train.json`` -- in-process training / inference seconds for
  cross-checking;
* ``analysis/manual_gpu_charges.json`` -- only if a failed CUDA attempt produced
  no rr record (none is expected here).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import x1_common as C


def _seconds(started: str | None, completed: str | None) -> float | None:
    if not started or not completed:
        return None
    return (datetime.fromisoformat(completed) - datetime.fromisoformat(started)).total_seconds()


def _status_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
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
    out: list[dict[str, Any]] = []
    if not C.RUNS.exists():
        return out
    for train_json in sorted(C.RUNS.glob("*/train.json")):
        payload = C.read_json(train_json)
        out.append(
            {
                "run_key": train_json.parent.name,
                "arm": payload.get("arm"),
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
    manual = C.read_json(C.ANALYSIS / "manual_gpu_charges.json") if (C.ANALYSIS / "manual_gpu_charges.json").exists() else []
    for row in manual:
        seconds = float(row.get("gpu_seconds", 0.0))
        records.append(
            {
                "experiment": row["run_key"],
                "run_id": None,
                "state": row.get("status", "FAILED"),
                "host": row.get("hostname"),
                "node": row.get("node"),
                "pool": row.get("pool"),
                "cuda_visible_devices": None,
                "gpu_names": [row.get("cuda_device_name")],
                "driver_version": row.get("driver_version"),
                "torch_version": row.get("torch_version"),
                "torch_cuda": row.get("torch_cuda"),
                "commit": row.get("git_commit"),
                "dirty": None,
                "created_at": row.get("started_utc"),
                "started_at": row.get("started_utc"),
                "completed_at": row.get("finished_utc"),
                "exit_code": row.get("exit_code"),
                "failure_reason": row.get("error"),
                "command": row.get("command"),
                "gpu_seconds": seconds,
                "gpu_hours_charged": seconds / 3600.0,
                "source_json": None,
                "charged_from": "launcher timestamps (manual, conservative)",
            }
        )
    total_seconds = float(sum(r["gpu_seconds"] or 0.0 for r in records))
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "host": C.load_config()["budget"]["host"],
        "hard_cap_gpu_hours": C.GPU_HARD_CAP_HOURS,
        "charging_rule": (
            "full rr-measured process lifetime of every process that touched CUDA "
            "(CPU smoke not charged unless it ran on CUDA); parallel pools summed, not wall-clocked"
        ),
        "runs": records,
        "n_runs": len(records),
        "total_gpu_seconds": total_seconds,
        "total_gpu_hours": total_seconds / 3600.0,
        "within_cap": bool(total_seconds / 3600.0 <= C.GPU_HARD_CAP_HOURS),
        "remaining_gpu_hours": max(0.0, C.GPU_HARD_CAP_HOURS - total_seconds / 3600.0),
        "in_process_timings": _in_process(),
        "gpu_models": sorted({name for r in records for name in (r.get("gpu_names") or []) if name}),
        "failures": [r for r in records if r.get("state") != "completed"],
    }
    C.write_json(C.GPU_LEDGER, payload)
    return payload


__all__ = ["build_ledger"]
