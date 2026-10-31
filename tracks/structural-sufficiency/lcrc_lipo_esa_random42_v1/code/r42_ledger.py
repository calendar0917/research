"""GPU / CPU cost ledger.

Every training and smoke invocation writes ``analysis/runs/<key>/envelope.json``
with wall-clock start/stop, hostname, device and status.  The ledger charges
**device-occupancy hours** = the full wall time of every process that ran on the
GPU (successful *and* failed attempts, including data loading and the in-process
CPU gates that preceded a training run).  CPU baselines are recorded separately
with ``gpu_hours = 0``.

The 6.0 A100 GPU-hour cap is enforced against this ledger, not against a guess.
"""

from __future__ import annotations

from typing import Any

import r42_common as C


def _entries() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not C.RUNS.exists():
        return rows
    for env_path in sorted(C.RUNS.glob("*/envelope.json")):
        env = C.read_json(env_path)
        run_dir = env_path.parent
        train_json = run_dir / "train.json"
        elapsed = env.get("elapsed_seconds")
        if elapsed is None:
            elapsed = float(env["finished_epoch_seconds"] - env["started_epoch_seconds"]) if "finished_epoch_seconds" in env else 0.0
        on_gpu = str(env.get("device", "cpu")).startswith("cuda")
        entry = {
            "run_key": env.get("run_key", run_dir.name),
            "model": env.get("model"),
            "status": env.get("status"),
            "device": env.get("device"),
            "cuda_device_name": env.get("cuda_device_name"),
            "hostname": env.get("hostname"),
            "git_commit": env.get("git_commit"),
            "started_utc": env.get("started_utc"),
            "finished_utc": env.get("finished_utc"),
            "wall_seconds": float(elapsed),
            "gpu_seconds": float(elapsed) if on_gpu else 0.0,
            "gpu_hours_charged": float(elapsed) / 3600.0 if on_gpu else 0.0,
            "envelope": C.track_rel(env_path),
            "error": env.get("error"),
        }
        if train_json.exists():
            payload = C.read_json(train_json)
            entry.update(
                {
                    "train_loop_seconds": payload.get("timing", {}).get("train_wall_seconds"),
                    "inference_seconds_per_valid420": payload.get("timing", {}).get("inference_seconds_per_valid420"),
                    "peak_memory_allocated_bytes": payload.get("gpu", {}).get("peak_memory_allocated_bytes"),
                    "checkpoint_sha256": payload.get("checkpoint_sha256") or payload.get("model_sha256"),
                    "valid_rmse_raw": payload.get("metrics", {}).get("valid", {}).get("rmse"),
                    "parameters_total": payload.get("model_audit", {}).get("parameters_total"),
                }
            )
        rows.append(entry)
    return rows


# GPU processes that died before writing their own envelope (e.g. an import
# error before model construction).  They are charged conservatively from the
# launcher timestamps so the ledger never under-reports.
MANUAL_CHARGES: list[dict[str, Any]] = [
    {
        "run_key": "smoke_esa_failed_attempt_1",
        "why": "ESA GPU smoke aborted on a missing RDKit import before model construction",
        "started_utc": "2026-10-31T15:08:53Z",
        "finished_utc": "2026-10-31T15:08:58Z",
        "gpu_seconds": 5.0,
        "device": "cuda",
        "cuda_device_name": "NVIDIA A100-SXM4-40GB",
        "status": "FAILED",
        "error": "ModuleNotFoundError: No module named 'rdkit'",
    }
]


def build_ledger() -> dict[str, Any]:
    entries = _entries()
    entries.extend(
        {
            "run_key": row["run_key"],
            "model": row["run_key"],
            "status": row["status"],
            "device": row["device"],
            "cuda_device_name": row["cuda_device_name"],
            "envelope": None,
            "started_utc": row["started_utc"],
            "finished_utc": row["finished_utc"],
            "wall_seconds": float(row["gpu_seconds"]),
            "gpu_seconds": float(row["gpu_seconds"]),
            "gpu_hours_charged": float(row["gpu_seconds"]) / 3600.0,
            "error": row["error"],
            "hostname": None,
            "git_commit": None,
            "checkpoint_sha256": None,
            "valid_rmse_raw": None,
            "parameters_total": None,
            "train_loop_seconds": None,
            "inference_seconds_per_valid420": None,
            "peak_memory_allocated_bytes": None,
            "charged_from": "launcher timestamps (manual, conservative)",
        }
        for row in MANUAL_CHARGES
    )
    gpu_seconds = sum(e["gpu_seconds"] for e in entries)
    cpu_hours = sum(e["wall_seconds"] for e in entries if e["gpu_seconds"] == 0.0) / 3600.0
    xgb_wall = None
    xgb_manifest = C.RUNS / "xgb_p3_s0" / "train.json"
    if xgb_manifest.exists():
        payload = C.read_json(xgb_manifest)
        xgb_wall = {
            "fit_seconds": payload["training"]["fit_seconds"],
            "predict_seconds_train_plus_valid": payload["training"]["predict_seconds_train_plus_valid"],
            "total_seconds": payload["training"]["total_seconds"],
            "device": "cpu",
            "gpu_hours": 0.0,
        }
    failures = [e for e in entries if e["status"] != "COMPLETED"]
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "gpu_hard_cap_hours": C.GPU_HARD_CAP_HOURS,
        "gpu_hours_charged": gpu_seconds / 3600.0,
        "gpu_hours_remaining": C.GPU_HARD_CAP_HOURS - gpu_seconds / 3600.0,
        "cpu_hours_recorded_separately": cpu_hours,
        "within_cap": (gpu_seconds / 3600.0) <= C.GPU_HARD_CAP_HOURS,
        "charging_rule": "full wall time of every process that ran on the GPU (train loop + data loading + validation), including failed attempts",
        "n_entries": len(entries),
        "n_failures": len(failures),
        "failures": failures,
        "gpu_models": sorted({e["cuda_device_name"] for e in entries if e["gpu_seconds"] > 0}),
        "xgb_cpu_baseline": xgb_wall,
        "entries": entries,
    }
    C.write_json(C.ANALYSIS / "gpu_ledger.json", payload)
    return payload


__all__ = ["build_ledger"]
