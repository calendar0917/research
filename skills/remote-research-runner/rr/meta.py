"""Run identity + metadata.

Every run is bound to an experiment name, a unique run id, a host/backend, a
pool, a Git commit and a creation timestamp.  The authoritative copy of a run's
metadata lives on the remote filesystem as ``runs/<run-id>/meta.json`` — never
only in local process memory — so `rr jobs/status/logs` recover after the agent
or terminal disappears.
"""

from __future__ import annotations

from .util import make_run_id, now_iso

SCHEMA = "rr/run-meta/v1"

TERMINAL_STATES = {"completed", "failed", "cancelled", "timeout", "oom", "lost"}


def build_meta(
    *,
    experiment: str,
    run_id: str,
    host: str,
    backend: str,
    pool: str | None,
    git: dict,
    command: str,
    requested: dict,
    result_paths: list[str],
    torch_backend: str | None = None,
    slurm_job_id: str | None = None,
) -> dict:
    return {
        "schema": SCHEMA,
        "experiment": experiment,
        "run_id": run_id,
        "host": host,
        "backend": backend,
        "pool": pool,
        "torch_backend": torch_backend,
        "git_commit": git.get("commit"),
        "git_branch": git.get("branch"),
        "git_dirty": bool(git.get("dirty")),
        "git_diff_hash": git.get("diff_hash"),
        "command": command,
        "requested": requested,
        "result_paths": list(result_paths or []),
        "created_at": now_iso(),
        "state": "submitted",
        "slurm_job_id": slurm_job_id,
        "runtime": {},
    }


def new_run_id(experiment: str) -> str:
    return make_run_id(experiment)


def is_terminal(state: str | None) -> bool:
    return (state or "") in TERMINAL_STATES


def summarize(meta: dict) -> dict:
    """Flatten a meta.json into the stable shape used by `rr jobs`/`--json`."""
    runtime = meta.get("runtime") or {}
    return {
        "legacy": bool(meta.get("legacy")),
        "run_id": meta.get("run_id"),
        "experiment": meta.get("experiment"),
        "host": meta.get("host"),
        "backend": meta.get("backend"),
        "pool": meta.get("pool"),
        "state": meta.get("state"),
        "job": meta.get("slurm_job_id") or runtime.get("slurm_job_id") or runtime.get("pid"),
        "node": runtime.get("node") or runtime.get("hostname"),
        "gpu_count": runtime.get("gpu_count"),
        "gpu_names": (runtime.get("gpus") or {}).get("names"),
        "driver_version": runtime.get("driver_version"),
        "cuda_visible_devices": runtime.get("cuda_visible_devices"),
        "scheduler_state": runtime.get("scheduler_state"),
        "failure_reason": runtime.get("failure_reason"),
        "torch_version": runtime.get("torch_version"),
        "torch_cuda": runtime.get("torch_cuda"),
        "commit": (meta.get("git_commit") or "")[:12],
        "branch": meta.get("git_branch"),
        "dirty": meta.get("git_dirty"),
        "pool_or_host": meta.get("pool") or meta.get("host"),
        "created_at": meta.get("created_at"),
        "started_at": runtime.get("started_at"),
        "completed_at": runtime.get("completed_at"),
        "exit_code": runtime.get("exit_code"),
    }
