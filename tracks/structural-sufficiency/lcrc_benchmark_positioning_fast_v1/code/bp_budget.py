"""Cross-worker GPU-hour and wall-clock budget ledger."""

from __future__ import annotations

import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import bp_common as C

LEDGER = C.ANALYSIS / "budget_ledger.json"
LOCK = C.ANALYSIS / ".budget.lock"


@contextmanager
def _locked() -> Iterator[None]:
    C.ANALYSIS.mkdir(parents=True, exist_ok=True)
    with open(LOCK, "a+", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _read() -> dict[str, Any]:
    if LEDGER.is_file():
        return C.read_json(LEDGER)
    return {
        "gpu_seconds": 0.0,
        "hard_gpu_hours": float(C.BUDGET["gpu_total_hours"]),
        "reservations": [],
        "tasks": [],
        "created_utc": C.utcnow(),
    }


def _write(payload: dict[str, Any]) -> None:
    C.write_json(LEDGER, payload)


def _prune(payload: dict[str, Any]) -> None:
    now = time.time()
    keep = []
    for r in payload.get("reservations", []):
        if now - float(r.get("created_epoch", 0)) < 1800:
            keep.append(r)
    payload["reservations"] = keep


def reserve(seconds: float, task_id: str, worker: str) -> dict[str, Any]:
    """Reserve ``seconds`` of one-GPU time if the hard cap allows it."""
    seconds = float(max(seconds, 1.0))
    with _locked():
        payload = _read()
        _prune(payload)
        reserved = sum(float(r["seconds"]) for r in payload["reservations"])
        cap = float(payload.get("hard_gpu_hours", C.BUDGET["gpu_total_hours"])) * 3600.0
        if float(payload["gpu_seconds"]) + reserved + seconds > cap:
            return {
                "ok": False,
                "reason": "GPU_HOURS_HARD_CAP",
                "gpu_seconds": float(payload["gpu_seconds"]),
                "reserved_seconds": reserved,
                "cap_seconds": cap,
            }
        reservation = {
            "task_id": task_id,
            "worker": worker,
            "seconds": seconds,
            "created_epoch": time.time(),
            "created_utc": C.utcnow(),
        }
        payload["reservations"].append(reservation)
        _write(payload)
        return {"ok": True, "reservation": reservation, "gpu_seconds": float(payload["gpu_seconds"]), "reserved_seconds": reserved}


def finish(reservation: dict[str, Any], actual_seconds: float, record: dict[str, Any]) -> dict[str, Any]:
    with _locked():
        payload = _read()
        _prune(payload)
        rid = reservation.get("task_id")
        kept = []
        found = False
        for r in payload["reservations"]:
            if not found and r.get("task_id") == rid and abs(float(r.get("seconds", 0)) - float(reservation.get("seconds", 0))) < 1e-9:
                found = True
                continue
            kept.append(r)
        payload["reservations"] = kept
        payload["gpu_seconds"] = float(payload["gpu_seconds"]) + float(max(actual_seconds, 0.0))
        payload["tasks"].append({**record, "actual_gpu_seconds": float(max(actual_seconds, 0.0)), "finished_utc": C.utcnow()})
        _write(payload)
        return {
            "gpu_seconds": float(payload["gpu_seconds"]),
            "gpu_hours": float(payload["gpu_seconds"]) / 3600.0,
            "n_tasks": len(payload["tasks"]),
        }


def snapshot() -> dict[str, Any]:
    with _locked():
        payload = _read()
        _prune(payload)
        return payload


__all__ = ["LEDGER", "reserve", "finish", "snapshot"]
