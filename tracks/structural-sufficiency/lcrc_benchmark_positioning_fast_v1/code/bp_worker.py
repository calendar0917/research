"""One GPU worker: claim tasks from the frozen plan, run, record, release."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

import bp_common as C
import bp_budget as B


def _blocked_datasets() -> set[str]:
    out: set[str] = set()
    p = C.ANALYSIS / "blocked_datasets.json"
    if p.is_file():
        payload = C.read_json(p)
        if isinstance(payload, dict):
            out.update(str(x) for x in payload.get("datasets", []))
        elif isinstance(payload, list):
            out.update(str(x) for x in payload)
    for ds in list(C.TU_DATASETS) + list(C.MOL_DATASETS) + [C.PEPTIDE_DATASET]:
        short = ds.replace("ogbg-", "")
        for name in (ds, short):
            p = C.ANALYSIS / f"{name}_blocked.json"
            if p.is_file():
                try:
                    payload = C.read_json(p)
                except Exception:
                    out.add(ds)
                    continue
                if payload.get("status") in ("INPUT_ADAPTER_BLOCKED", "PREP_FAILED", "RESOURCE_BLOCKED"):
                    out.add(ds)
    return out


def _estimate_seconds(entry: dict[str, Any], max_task_seconds: float) -> float:
    # Conservative: reserve the full per-task cap so the 18-GPU-hour ledger is
    # never under-counted by a task that reaches its timeout.
    return float(max_task_seconds)


def _claim_path(task_id: str) -> Path:
    return C.RUNS / f"{task_id}.claim"


def _try_claim(entry: dict[str, Any], worker: str, max_task_seconds: float) -> tuple[bool, Path]:
    path = _claim_path(entry["task_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        age = time.time() - float(payload.get("created_epoch", path.stat().st_mtime))
        run_json = C.result_paths(entry["kind"], entry["dataset"], entry["unit"]) / "run.json"
        if age > float(max_task_seconds) + 300.0 and not run_json.is_file():
            path.unlink(missing_ok=True)
            return _try_claim(entry, worker, max_task_seconds)
        return False, path
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "task_id": entry["task_id"],
                "worker": worker,
                "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "created_epoch": time.time(),
                "created_utc": C.utcnow(),
            },
            fh,
            sort_keys=True,
        )
    return True, path


def _log_path(task_id: str) -> Path:
    p = C.ANALYSIS / "worker_logs" / f"{task_id}.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _record_skip(task_id: str, reason: str, details: dict[str, Any] | None = None) -> None:
    row = {"task_id": task_id, "status": "SKIPPED", "reason": reason, "utc": C.utcnow(), **(details or {})}
    p = C.ANALYSIS / "skipped_tasks.jsonl"
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default=str(C.ANALYSIS / "formal_plan.json"))
    parser.add_argument("--worker", required=True)
    parser.add_argument("--hard-deadline-epoch", type=float, required=True)
    parser.add_argument("--soft-deadline-epoch", type=float, required=True)
    parser.add_argument("--gpu-budget-hours", type=float, default=float(C.BUDGET["gpu_total_hours"]))
    parser.add_argument("--max-task-seconds", type=float, default=float(C.BUDGET["per_task_minutes"]) * 60.0)
    parser.add_argument("--max-tasks", type=int, default=None)
    args = parser.parse_args(argv)
    C.ANALYSIS.mkdir(parents=True, exist_ok=True)
    C.RUNS.mkdir(parents=True, exist_ok=True)
    plan = C.read_json(args.plan)["order"]
    blocked = _blocked_datasets()
    started = time.time()
    records: list[dict[str, Any]] = []
    n_started = 0
    gpu_name = ""
    try:
        import torch

        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        gpu_name = "torch-import-failed"
    C.log(f"worker {args.worker} start; gpu={gpu_name}; visible={os.environ.get('CUDA_VISIBLE_DEVICES','')}")
    for entry in plan:
        if args.max_tasks is not None and n_started >= int(args.max_tasks):
            break
        now = time.time()
        if now >= float(args.hard_deadline_epoch):
            C.log(f"worker {args.worker}: hard wall deadline reached")
            break
        task_id = entry["task_id"]
        run_dir = C.result_paths(entry["kind"], entry["dataset"], entry["unit"])
        run_json = run_dir / "run.json"
        if run_json.is_file():
            existing = C.read_json(run_json)
            if existing.get("status") == "complete":
                continue
        if entry["dataset"] in blocked:
            _record_skip(task_id, "DATASET_BLOCKED", {"dataset": entry["dataset"]})
            continue
        if now >= float(args.soft_deadline_epoch) and entry.get("cost_class") == "large":
            _record_skip(task_id, "SOFT_WALL_BUDGET_LARGE_TASK", {"dataset": entry["dataset"]})
            continue
        estimate = _estimate_seconds(entry, args.max_task_seconds)
        if now + estimate >= float(args.hard_deadline_epoch):
            _record_skip(task_id, "HARD_WALL_DEADLINE_WOULD_OVERRUN", {"estimate_seconds": estimate})
            break
        reservation = B.reserve(estimate, task_id, args.worker)
        if not reservation.get("ok"):
            _record_skip(task_id, str(reservation.get("reason")), reservation)
            break
        claimed, claim_path = _try_claim(entry, args.worker, args.max_task_seconds)
        if not claimed:
            B.finish(reservation["reservation"], 0.0, {"task_id": task_id, "status": "CLAIMED_BY_OTHER"})
            continue
        n_started += 1
        log_path = _log_path(task_id)
        cmd = [
            sys.executable,
            str(C.CODE_DIR / "bp_train.py"),
            "--kind",
            entry["kind"],
            "--dataset",
            entry["dataset"],
            "--unit",
            str(entry["unit"]),
            "--device",
            "cuda",
            "--max-seconds",
            str(args.max_task_seconds),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(C.CODE_DIR) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        t0 = time.time()
        status = "WORKER_EXIT_UNKNOWN"
        detail: dict[str, Any] = {}
        try:
            with open(log_path, "w", encoding="utf-8") as log:
                proc = subprocess.run(
                    cmd,
                    cwd=str(C.REPO),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=float(args.max_task_seconds) + 300.0,
                    env=env,
                    check=False,
                )
            status = f"EXIT_{proc.returncode}"
        except subprocess.TimeoutExpired:
            status = "WORKER_HARD_TIMEOUT"
            C.write_json(
                run_dir / "run.json",
                {
                    "status": "RESOURCE_BLOCKED",
                    "reason": "WORKER_HARD_TIMEOUT",
                    "kind": entry["kind"],
                    "dataset": entry["dataset"],
                    "unit": entry["unit"],
                    "max_seconds": float(args.max_task_seconds),
                },
            )
        except Exception as exc:  # noqa: BLE001
            status = "WORKER_LAUNCH_FAILED"
            detail = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        actual = time.time() - t0
        if run_json.is_file():
            try:
                result = C.read_json(run_json)
                status = str(result.get("status", status))
                actual = float(result.get("seconds_including_test", result.get("seconds", actual)))
                detail = {
                    "best_epoch": result.get("best_epoch"),
                    "epochs_run": result.get("epochs_run"),
                    "parameters": result.get("parameters"),
                    "checkpoint_sha256": result.get("checkpoint_sha256"),
                    "test_metrics": result.get("test_metrics"),
                }
            except Exception as exc:  # noqa: BLE001
                detail = {"run_json_error": f"{type(exc).__name__}: {exc}"}
        else:
            C.write_json(
                run_dir / "run.json",
                {
                    "status": "FAILED",
                    "reason": status,
                    "kind": entry["kind"],
                    "dataset": entry["dataset"],
                    "unit": entry["unit"],
                    "worker": args.worker,
                    "detail": detail,
                },
            )
        record = {
            "task_id": task_id,
            "kind": entry["kind"],
            "dataset": entry["dataset"],
            "unit": entry["unit"],
            "worker": args.worker,
            "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "gpu_name": gpu_name,
            "status": status,
            "seconds": float(actual),
            "log": C.track_rel(log_path),
            **detail,
        }
        records.append(record)
        try:
            B.finish(reservation["reservation"], actual, record)
        except Exception as exc:  # noqa: BLE001
            C.log(f"budget finish failed: {exc}")
        claim_path.unlink(missing_ok=True)
        C.log(f"worker {args.worker}: {task_id} -> {status} ({actual:.1f}s)")
    summary = {
        "worker": args.worker,
        "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_name": gpu_name,
        "started_epoch": started,
        "finished_epoch": time.time(),
        "wall_seconds": time.time() - started,
        "n_started": n_started,
        "records": records,
        "budget": B.snapshot(),
        "environment": C.environment(),
    }
    C.write_json(C.ANALYSIS / f"worker_{args.worker}.json", summary)
    with open(C.ANALYSIS / f"worker_{args.worker}.jsonl", "a", encoding="utf-8") as fh:
        for row in records:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    C.log(f"worker {args.worker} end; started {n_started} tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
