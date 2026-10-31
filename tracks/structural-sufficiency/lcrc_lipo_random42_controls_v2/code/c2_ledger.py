"""GPU / CPU cost ledger for controls v2.

Charging rule: the full wall time of every process that touched CUDA, including
data loading, smoke runs, the inference benchmark and failed attempts.  CPU-only
work is recorded separately at zero GPU-hours.
"""

from __future__ import annotations

from typing import Any

import c2_common as C


def _read_optional(path) -> Any:
    return C.read_json(path) if path.exists() else None


def _entries() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not C.RUNS.exists():
        return rows
    for env_path in sorted(C.RUNS.glob("*/envelope.json")):
        env = C.read_json(env_path)
        run_dir = env_path.parent
        elapsed = env.get("elapsed_seconds")
        if elapsed is None and "finished_epoch_seconds" in env and "started_epoch_seconds" in env:
            elapsed = float(env["finished_epoch_seconds"] - env["started_epoch_seconds"])
        elapsed = float(elapsed or 0.0)
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
            "wall_seconds": elapsed,
            "gpu_seconds": elapsed if on_gpu else 0.0,
            "gpu_hours_charged": elapsed / 3600.0 if on_gpu else 0.0,
            "envelope": C.track_rel(env_path),
            "error": env.get("error"),
        }
        train_json = run_dir / "train.json"
        if train_json.exists():
            payload = C.read_json(train_json)
            entry.update(
                {
                    "train_loop_seconds": payload.get("timing", {}).get("train_wall_seconds"),
                    "inference_seconds_per_valid420": payload.get("timing", {}).get("inference_seconds_per_valid420"),
                    "peak_memory_allocated_bytes": payload.get("gpu", {}).get("peak_memory_allocated_bytes"),
                    "checkpoint_sha256": payload.get("checkpoint_sha256"),
                    "valid_rmse_raw": payload.get("metrics", {}).get("valid", {}).get("rmse"),
                    "parameters_total": payload.get("model_audit", {}).get("parameters_total")
                    or payload.get("model_audit", {}).get("parameters_total_nominal"),
                    "optimizer_steps": payload.get("training", {}).get("optimizer_steps"),
                    "best_epoch": payload.get("training", {}).get("best_epoch"),
                }
            )
        if env.get("extra"):
            entry["extra"] = env["extra"]
        rows.append(entry)
    return rows


def build_ledger() -> dict[str, Any]:
    entries = _entries()
    manual = _read_optional(C.ANALYSIS / "manual_gpu_charges.json") or []
    for row in manual:
        entries.append(
            {
                "run_key": row["run_key"],
                "model": row.get("model", row["run_key"]),
                "status": row.get("status", "FAILED"),
                "device": "cuda",
                "cuda_device_name": row.get("cuda_device_name", "NVIDIA A100-SXM4-40GB"),
                "hostname": row.get("hostname"),
                "git_commit": row.get("git_commit"),
                "started_utc": row.get("started_utc"),
                "finished_utc": row.get("finished_utc"),
                "wall_seconds": float(row["gpu_seconds"]),
                "gpu_seconds": float(row["gpu_seconds"]),
                "gpu_hours_charged": float(row["gpu_seconds"]) / 3600.0,
                "envelope": None,
                "error": row.get("error"),
                "charged_from": "launcher timestamps (manual, conservative)",
            }
        )
    gpu_seconds = float(sum(e["gpu_seconds"] for e in entries))
    cpu_seconds = float(sum(e["wall_seconds"] for e in entries if e["gpu_seconds"] == 0.0))
    failures = [e for e in entries if e["status"] != "COMPLETED"]
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "gpu_hard_cap_hours": C.GPU_HARD_CAP_HOURS,
        "gpu_hours_charged": gpu_seconds / 3600.0,
        "gpu_hours_remaining": C.GPU_HARD_CAP_HOURS - gpu_seconds / 3600.0,
        "cpu_hours_recorded_separately": cpu_seconds / 3600.0,
        "within_cap": (gpu_seconds / 3600.0) <= C.GPU_HARD_CAP_HOURS,
        "charging_rule": "full wall time of every process that touched CUDA (data loading + train loop + validation + benchmark + failed attempts)",
        "n_entries": len(entries),
        "n_gpu_entries": int(sum(1 for e in entries if e["gpu_seconds"] > 0)),
        "n_failures": len(failures),
        "failures": failures,
        "gpu_models": sorted({e["cuda_device_name"] for e in entries if e["gpu_seconds"] > 0 and e["cuda_device_name"]}),
        "entries": entries,
    }
    C.write_json(C.GPU_LEDGER, payload)
    return payload


__all__ = ["build_ledger"]
