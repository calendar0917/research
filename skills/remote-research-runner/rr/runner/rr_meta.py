#!/usr/bin/env python3
"""Remote provenance helper for rr runs.

Invoked on the remote host by rr_runner.sh.  Records the *execution regime*
actually observed on the allocated node (driver, GPU model/count, torch/cuda,
python) into the run's meta.json so heterogeneous clusters never silently
produce incomparable results.

Stdlib only; safe to run with either the project venv python or system python.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone


def _iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _cmd(argv: list[str], timeout: float = 20.0) -> str:
    try:
        cp = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except Exception:
        return ""
    return cp.stdout.strip() if cp.returncode == 0 else ""


def probe_driver() -> str | None:
    try:
        with open("/proc/driver/nvidia/version") as fh:
            text = fh.read()
        m = re.search(r"Kernel Module\s+([0-9][0-9A-Za-z.\-+]*)", text)
        if m:
            return m.group(1)
    except OSError:
        pass
    out = _cmd(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    return out.splitlines()[0].strip() if out else None


def probe_gpus() -> dict:
    out = _cmd(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        timeout=15,
    )
    if not out:
        return {"count": 0, "names": []}
    names = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if parts and parts[0]:
            names.append(parts[0])
    return {"count": len(names), "names": names}


def probe_torch() -> dict:
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return {"torch_version": None, "torch_cuda": None, "cuda_available": None,
                "torch_error": f"{type(exc).__name__}: {exc}"[:200]}
    info = {
        "torch_version": getattr(torch, "__version__", None),
        "torch_cuda": getattr(getattr(torch, "version", None), "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    return info


def load(path: str) -> dict:
    try:
        with open(path) as fh:
            return json.load(fh)
    except FileNotFoundError:
        return {}


def save(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    os.replace(tmp, path)


def _driver_at_least(actual: str | None, minimum: str | None) -> bool:
    if not minimum:
        return True
    if not actual:
        return False

    def parse(v: str) -> tuple[int, ...]:
        parts = []
        for chunk in re.split(r"[.\-+ ]", v.strip()):
            m = re.match(r"^(\d+)", chunk)
            parts.append(int(m.group(1)) if m else 0)
        while len(parts) < 3:
            parts.append(0)
        return tuple(parts[:3])

    return parse(actual) >= parse(minimum)


def cmd_start(meta_path: str, probe_torch_flag: bool, min_driver: str | None = None) -> int:
    meta = load(meta_path)
    runtime = meta.setdefault("runtime", {})
    runtime["hostname"] = socket.gethostname()
    runtime["node"] = os.environ.get("SLURMD_NODENAME") or socket.gethostname()
    runtime["driver_version"] = probe_driver()
    runtime["gpus"] = probe_gpus()
    runtime["gpu_count"] = runtime["gpus"].get("count", 0)
    runtime["python_version"] = sys.version.split()[0]
    runtime["python_executable"] = sys.executable
    runtime["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    runtime["slurm_job_nodelist"] = os.environ.get("SLURM_JOB_NODELIST")
    runtime["slurm_partition"] = os.environ.get("SLURM_JOB_PARTITION")
    runtime["slurm_cpus"] = os.environ.get("SLURM_CPUS_PER_TASK")
    runtime["slurm_gpus"] = os.environ.get("SLURM_GPUS") or os.environ.get("CUDA_VISIBLE_DEVICES")
    # Explicit GPU isolation (process pools pin one device): record it so the
    # execution regime — not just the allocation — is auditable.
    runtime["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    runtime["started_at"] = _iso()
    if not _driver_at_least(runtime["driver_version"], min_driver):
        runtime["completed_at"] = _iso()
        runtime["exit_code"] = 78
        runtime["failure_reason"] = "driver_too_old"
        meta["state"] = "failed"
        save(meta_path, meta)
        print(
            f"rr: node {runtime['node']} driver {runtime['driver_version']} < required {min_driver}",
            file=sys.stderr,
        )
        return 78
    if probe_torch_flag:
        runtime.update(probe_torch())
    meta["state"] = "running"
    save(meta_path, meta)
    return 0


def cmd_finish(meta_path: str, exit_code: int) -> int:
    meta = load(meta_path)
    runtime = meta.setdefault("runtime", {})
    runtime["completed_at"] = _iso()
    runtime["exit_code"] = exit_code
    meta["state"] = "completed" if exit_code == 0 else "failed"
    save(meta_path, meta)
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: rr_meta.py {start|finish} <meta.json> [exit_code]", file=sys.stderr)
        return 2
    action = argv[1]
    meta_path = argv[2]
    min_driver = None
    for arg in argv[3:]:
        if arg.startswith("--min-driver="):
            min_driver = arg.split("=", 1)[1] or None
    if action == "start":
        probe_torch_flag = "--no-torch" not in argv
        return cmd_start(meta_path, probe_torch_flag, min_driver)
    if action == "finish":
        code = int(argv[3]) if len(argv) > 3 and argv[3].lstrip("-").isdigit() else 0
        return cmd_finish(meta_path, code)
    print(f"unknown action: {action}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
