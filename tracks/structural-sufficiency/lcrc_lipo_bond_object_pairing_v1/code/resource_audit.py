"""Preflight workspace / GPU isolation audit (G7).

Read-only SSH inspection is used only to observe GPU processes and the remote
checkout.  No job is launched here.  Formal execution still goes through ``rr``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import common as C

SHARED_REMOTE_CODE = "/home/hxy/cy/research"
OUR_REMOTE_CODE = "/home/hxy/cy/research-bond-object-pairing-v1"
OUR_REMOTE_RUNS = "/home/hxy/cy/rr-runs-bond-object-pairing"
SELECTED_PHYSICAL_GPU = 1
PEPTIDE_MARKER = "lcrc_peptides_weighted_bce_v1"


def _run(cmd: list[str], timeout: int = 60) -> dict[str, Any]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except Exception as exc:  # pragma: no cover - environment-specific
        return {"cmd": cmd, "returncode": None, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}"}


def _git(args: list[str]) -> str:
    p = subprocess.run(["git", *args], cwd=C.REPO, capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else ""


def _parse_gpu_lines(lines: list[str]) -> list[dict[str, Any]]:
    out = []
    for line in lines:
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            out.append({
                "index": int(parts[0]),
                "uuid": parts[1],
                "name": parts[2],
                "memory_total": parts[3],
                "memory_used": parts[4],
                "utilization": parts[5],
            })
        except ValueError:
            continue
    return out


def _parse_apps(lines: list[str]) -> list[dict[str, Any]]:
    out = []
    for line in lines:
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            out.append({"gpu_uuid": parts[0], "pid": int(parts[1]), "used_memory": parts[2]})
        except ValueError:
            continue
    return out


def run_resource_audit() -> dict[str, Any]:
    worktrees = _git(["worktree", "list", "--porcelain"])
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    commit = _git(["rev-parse", "HEAD"])
    status = _git(["status", "--porcelain", "--untracked-files=no"])
    local = {
        "worktree": str(C.EXP_DIR),
        "repo_root": str(C.REPO),
        "branch": branch,
        "commit": commit,
        "tracked_worktree_clean": status == "",
        "worktree_list": worktrees,
        "track_dir": C.track_rel(C.EXP_DIR),
        "cache_dir": C.track_rel(C.CACHE),
        "runs_dir": C.track_rel(C.RUNS),
        "results_dir": C.track_rel(C.RESULTS),
        "is_independent_worktree": str(C.EXP_DIR) != "/home/calendar/code/research",
        "branch_is_not_main": branch != "main",
        "historical_dirs_read_only": True,
    }

    gpu = _run(["ssh", "res", "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"])
    apps = _run(["ssh", "res", "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"])
    ps = _run(["ssh", "res", "ps -eo pid,ppid,etime,pcpu,pmem,cmd --sort=-pcpu"])
    remote_git = _run(["ssh", "res", f"if [ -d {OUR_REMOTE_CODE}/.git ]; then git -C {OUR_REMOTE_CODE} rev-parse HEAD; git -C {OUR_REMOTE_CODE} status --porcelain --untracked-files=no; else echo MISSING; fi"])
    shared_git = _run(["ssh", "res", f"if [ -d {SHARED_REMOTE_CODE}/.git ]; then git -C {SHARED_REMOTE_CODE} rev-parse HEAD; git -C {SHARED_REMOTE_CODE} status --porcelain --untracked-files=no; else echo MISSING; fi"])
    disk = _run(["ssh", "res", "df -h /home/hxy/cy | tail -1"])
    rr_jobs = _run(["rr", "jobs", "--json"], timeout=120)

    gpus = _parse_gpu_lines(gpu["stdout"].splitlines()) if gpu["returncode"] == 0 else []
    compute_apps = _parse_apps(apps["stdout"].splitlines()) if apps["returncode"] == 0 else []
    selected = next((g for g in gpus if g["index"] == SELECTED_PHYSICAL_GPU), None)
    selected_apps = [a for a in compute_apps if selected and a["gpu_uuid"] == selected["uuid"]]
    gpu0 = next((g for g in gpus if g["index"] == 0), None)
    gpu0_apps = [a for a in compute_apps if gpu0 and a["gpu_uuid"] == gpu0["uuid"]]
    peptide_pids = []
    for line in ps["stdout"].splitlines():
        if PEPTIDE_MARKER in line and "python" in line:
            parts = line.split(None, 5)
            if parts:
                try:
                    peptide_pids.append({"pid": int(parts[0]), "cmd": parts[-1]})
                except ValueError:
                    pass
    peptide_on_gpu0 = bool(gpu0_apps) and any(any(p["pid"] == a["pid"] for p in peptide_pids) for a in gpu0_apps)
    selected_memory_mib = None
    if selected:
        m = re.search(r"(\d+)\s*MiB", selected["memory_used"])
        selected_memory_mib = int(m.group(1)) if m else None
    selected_util = None
    if selected:
        m = re.search(r"(\d+)\s*%", selected["utilization"])
        selected_util = int(m.group(1)) if m else None

    jobs = []
    try:
        jobs = json.loads(rr_jobs["stdout"]).get("jobs", []) if rr_jobs["returncode"] == 0 else []
    except json.JSONDecodeError:
        jobs = []
    running_peptides = [
        {"experiment": j.get("experiment"), "run_id": j.get("run_id"), "pool": j.get("pool"), "state": j.get("state"), "cuda_visible_devices": j.get("cuda_visible_devices")}
        for j in jobs if j.get("state") == "running" and "peptides" in str(j.get("experiment", ""))
    ]
    running_ours = [
        {"experiment": j.get("experiment"), "run_id": j.get("run_id"), "state": j.get("state")}
        for j in jobs if j.get("state") == "running" and str(j.get("experiment", "")).startswith("lcrc-lipo-bond")
    ]

    remote_code_exists = "MISSING" not in remote_git["stdout"] and remote_git["returncode"] == 0
    remote_lines = [x for x in remote_git["stdout"].splitlines() if x.strip()]
    remote_commit = remote_lines[0] if remote_code_exists and remote_lines else None
    remote_clean = remote_code_exists and len([x for x in remote_lines[1:] if x.strip()]) == 0
    shared_lines = [x for x in shared_git["stdout"].splitlines() if x.strip()]
    shared_commit = shared_lines[0] if shared_lines and "MISSING" not in shared_lines[0] else None

    selected_free = bool(selected) and not selected_apps and (selected_memory_mib is not None and selected_memory_mib < 512) and (selected_util is not None and selected_util <= 5)
    checks = {
        "independent_local_worktree": local["is_independent_worktree"],
        "local_branch_not_main": local["branch_is_not_main"],
        "local_tracked_worktree_clean": local["tracked_worktree_clean"],
        "remote_code_independent": OUR_REMOTE_CODE != SHARED_REMOTE_CODE,
        "remote_runs_independent": OUR_REMOTE_RUNS != "/home/hxy/cy/rr-runs",
        "remote_code_not_nested_in_shared": not OUR_REMOTE_CODE.startswith(SHARED_REMOTE_CODE + "/"),
        "gpu_inventory_available": bool(gpus) and len(gpus) >= 2,
        "peptide_process_visible": bool(peptide_pids),
        "peptide_on_gpu0": peptide_on_gpu0,
        "selected_gpu_is_not_peptide_gpu": not (peptide_on_gpu0 and selected and gpu0 and selected["uuid"] == gpu0["uuid"]),
        "selected_gpu_no_compute_apps": not selected_apps,
        "selected_gpu_low_memory": selected_memory_mib is not None and selected_memory_mib < 512,
        "selected_gpu_idle": selected_util is not None and selected_util <= 5,
        "selected_gpu_free": selected_free,
        "our_round_not_already_running": not running_ours,
        "peptide_run_not_using_selected_gpu": not any(str(j.get("cuda_visible_devices")) == str(SELECTED_PHYSICAL_GPU) for j in running_peptides),
        "remote_gpu_check_ok": gpu["returncode"] == 0 and apps["returncode"] == 0 and ps["returncode"] == 0,
    }
    blocking = [k for k, v in checks.items() if v is not True]
    return {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "status": "PASS" if not blocking else "BLOCKED",
        "blocking_reasons": blocking,
        "checks": checks,
        "local": local,
        "remote": {
            "ssh_alias": "res",
            "shared_code": SHARED_REMOTE_CODE,
            "shared_commit": shared_commit,
            "our_code": OUR_REMOTE_CODE,
            "our_runs": OUR_REMOTE_RUNS,
            "our_code_exists": remote_code_exists,
            "our_code_commit": remote_commit,
            "our_code_clean": remote_clean,
            "selected_physical_gpu": SELECTED_PHYSICAL_GPU,
            "selected_gpu": selected,
            "selected_gpu_compute_apps": selected_apps,
            "selected_gpu_memory_used_mib": selected_memory_mib,
            "selected_gpu_utilization_pct": selected_util,
            "gpus": gpus,
            "compute_apps": compute_apps,
            "peptide_processes": peptide_pids,
            "peptide_on_gpu0": peptide_on_gpu0,
            "running_peptide_jobs": running_peptides,
            "running_our_jobs": running_ours,
            "disk": disk.get("stdout"),
            "rr_jobs_returncode": rr_jobs.get("returncode"),
        },
        "planned_execution": {
            "rr_config": "tracks/structural-sufficiency/lcrc_lipo_bond_object_pairing_v1/configs/rr_res_bond_pairing_v1.toml",
            "rr_host": "res-bond",
            "rr_pool": "res-bond-gpu1",
            "CUDA_VISIBLE_DEVICES": "1",
            "physical_gpu_index": SELECTED_PHYSICAL_GPU,
            "same_backend_for_all_six_runs": True,
        },
        "monitor_labels_read": False,
        "reserved_labels_read": False,
        "official_valid_test_read": False,
    }


__all__ = ["run_resource_audit", "SELECTED_PHYSICAL_GPU", "OUR_REMOTE_CODE"]
