"""Unit tests for run identity + metadata."""

from __future__ import annotations

from rr.meta import build_meta, is_terminal, new_run_id, summarize


def test_build_and_summarize_meta():
    git = {"commit": "a" * 40, "branch": "main", "dirty": False, "diff_hash": "abcd1234"}
    meta = build_meta(
        experiment="zinc-seed0",
        run_id=new_run_id("zinc-seed0"),
        host="res-2",
        backend="slurm",
        pool="res2-cu124",
        git=git,
        command="python -m x",
        requested={"gpus": 2, "cpus": 4, "mem": "32G", "time": "24:00:00"},
        result_paths=["tracks/ksvd/runs/x"],
        torch_backend="cu124",
    )
    assert meta["git_commit"] == "a" * 40
    assert meta["state"] == "submitted"
    s = summarize(meta)
    assert s["run_id"] == meta["run_id"]
    assert s["commit"] == "a" * 12
    assert s["pool"] == "res2-cu124"


def test_terminal_states():
    assert is_terminal("completed")
    assert is_terminal("failed")
    assert is_terminal("cancelled")
    assert not is_terminal("running")
    assert not is_terminal("pending")
    assert not is_terminal("submitted")


def test_summarize_uses_runtime_provenance():
    meta = {
        "run_id": "r1",
        "experiment": "e",
        "host": "res-2",
        "backend": "slurm",
        "pool": "res2-cu124",
        "state": "completed",
        "git_commit": "b" * 40,
        "slurm_job_id": "12345",
        "runtime": {
            "node": "c05",
            "gpu_count": 1,
            "gpus": {"names": ["NVIDIA A100-SXM4-40GB"]},
            "driver_version": "525.85.12",
            "torch_version": "2.5.1+cu124",
            "torch_cuda": "12.4",
            "exit_code": 0,
        },
    }
    s = summarize(meta)
    assert s["job"] == "12345"
    assert s["node"] == "c05"
    assert s["driver_version"] == "525.85.12"
    assert s["gpu_names"] == ["NVIDIA A100-SXM4-40GB"]


def test_summarize_reports_cuda_visible_devices_and_scheduler_state():
    meta = {
        "run_id": "r1",
        "experiment": "e",
        "host": "res",
        "backend": "process",
        "pool": "res-gpu1",
        "state": "completed",
        "runtime": {
            "cuda_visible_devices": "1",
            "scheduler_state": "COMPLETED",
            "failure_reason": None,
            "exit_code": 0,
        },
    }
    s = summarize(meta)
    assert s["cuda_visible_devices"] == "1"
    assert s["scheduler_state"] == "COMPLETED"


def test_summarize_reports_failure_reason():
    meta = {"run_id": "r1", "state": "oom", "runtime": {"failure_reason": "out_of_memory", "exit_code": 137}}
    s = summarize(meta)
    assert s["failure_reason"] == "out_of_memory"
    assert s["exit_code"] == 137