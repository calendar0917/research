"""Two persistent GPU workers (one per physical card) for the frozen task queue.

Pool -> owner mapping (task brief 1.2/1.4/8.3):

* stage 1 + stage 2 + stage 3: ``res-gpu0`` owns ``ogbg-molesol``,
  ``res-gpu1`` owns ``ogbg-mollipo``;
* oracle calibration: ``res-gpu1`` owns ``WC_LOGP``, ``res-gpu0`` owns ``MOLWT``.

Each worker runs at most one training/attribution unit at a time, honours the
dependency barrier (all 12 stage-1 cells before stage 2/3/oracle) and never
starts a unit after the wall-clock deadline.  Every unit is resumable and
idempotent: a completed signature is skipped, a matching partial resume file is
continued.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import gr_common as C
import gr_train as T

POOL_DATASET = {"res-gpu0": "ogbg-molesol", "res-gpu1": "ogbg-mollipo"}
POOL_ORACLE = {"res-gpu0": "MOLWT", "res-gpu1": "WC_LOGP"}


def plan(pool: str) -> list[dict[str, Any]]:
    dataset = POOL_DATASET[pool]
    prop = POOL_ORACLE[pool]
    out = []
    for task in C.stage1_tasks():
        if task["dataset"] == dataset:
            out.append(task)
    for task in C.stage2_tasks():
        if task["dataset"] == dataset:
            out.append(task)
    for task in C.stage3_tasks():
        if task["dataset"] == dataset:
            out.append(task)
    for task in C.oracle_tasks():
        if task["property"] == prop:
            out.append(task)
    return out


def _deps_ready(task: dict[str, Any]) -> tuple[bool, str]:
    stage1 = C.stage1_tasks()
    if not all(C.task_complete(t) for t in stage1):
        missing = [t["key"] for t in stage1 if not C.task_complete(t)]
        return False, f"stage1 pending: {len(missing)}"
    if task["stage"] == C.STAGE_ORACLE:
        stage2 = C.stage2_tasks()
        if not all(C.task_complete(t) for t in stage2):
            return False, "stage2_A pending"
    return True, "ready"


def _heartbeat(pool: str, payload: dict[str, Any]) -> None:
    C.write_json(C.ANALYSIS_DIR / f"worker_{pool}.json", payload)


def serve(pool: str, deadline: dt.datetime, *, force: bool = False, only: str | None = None) -> int:
    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        props = torch.cuda.get_device_properties(0)
        print(f"[worker] {pool} device={device} name={props.name} uuid={props.uuid}", flush=True)
    else:
        print(f"[worker] {pool} no CUDA, running on CPU", flush=True)
    tasks = plan(pool)
    if only:
        tasks = [t for t in tasks if only in t["key"]]
    started = time.time()
    _heartbeat(pool, {"pool": pool, "state": "started", "pid": os.getpid(), "device": device, "n_tasks": len(tasks)})
    done: list[str] = []
    for task in tasks:
        key = task["key"]
        if C.task_complete(task) and not force:
            print(f"[worker] skip completed {key}", flush=True)
            done.append(key)
            continue
        while True:
            ready, why = _deps_ready(task)
            now = dt.datetime.now(dt.timezone.utc)
            if ready:
                break
            if now >= deadline:
                print(f"[worker] deadline before {key} ({why})", flush=True)
                _heartbeat(pool, {"pool": pool, "state": "deadline", "pending": key, "why": why})
                return 0
            print(f"[worker] waiting for barrier before {key}: {why}", flush=True)
            _heartbeat(pool, {"pool": pool, "state": "barrier_wait", "pending": key, "why": why})
            time.sleep(20)
        now = dt.datetime.now(dt.timezone.utc)
        if now >= deadline:
            print(f"[worker] deadline reached, not starting {key}", flush=True)
            _heartbeat(pool, {"pool": pool, "state": "deadline", "pending": key})
            return 0
        print(f"[worker] START {key} at {now.isoformat()}", flush=True)
        _heartbeat(pool, {"pool": pool, "state": "running", "task": key, "started_at": now.isoformat()})
        t0 = time.time()
        try:
            if task["kind"] == "frozen_head":
                summary = T.run_stage3(task, device=device)
            else:
                summary = T.run_standard_task(task, device=device)
            seconds = time.time() - t0
            print(
                f"[worker] DONE {key} {seconds:.1f}s selected={summary.get('selected', {}).get('valid_rmse_raw')}",
                flush=True,
            )
            _append_cost(pool, key, seconds, summary, task)
            done.append(key)
        except Exception:
            traceback.print_exc()
            _heartbeat(pool, {"pool": pool, "state": "error", "task": key, "traceback": traceback.format_exc()})
            C.write_json(
                C.task_dir(task["stage"], task["dataset"], task["model"], task["recipe"]) / "status.json",
                {"state": "failed", "task": key, "traceback": traceback.format_exc()},
            )
            return 1
        finally:
            try:
                import gc

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
    _heartbeat(pool, {"pool": pool, "state": "complete", "tasks": done})
    print(f"[worker] queue complete in {time.time() - started:.1f}s", flush=True)
    return 0


def _append_cost(pool: str, key: str, seconds: float, summary: dict[str, Any], task: dict[str, Any]) -> None:
    path = C.ANALYSIS_DIR / "task_costs.json"
    payload = C.read_json(path) if path.is_file() else {"units": {}}
    payload["units"][key] = {
        "pool": pool,
        "seconds": seconds,
        "task": dict(task),
        "selected": summary.get("selected", {}),
        "n_parameters": summary.get("n_parameters"),
        "train_seconds": summary.get("train_seconds"),
        "eval_seconds": summary.get("eval_seconds"),
        "environment": summary.get("environment", {}),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    C.write_json(path, payload)


def _parse_deadline(text: str) -> dt.datetime:
    return dt.datetime.fromisoformat(text.replace("Z", "+00:00"))


def smoke(device: str = "cuda:0") -> int:
    """Short independent smoke: 3 epochs of the heaviest cell + additive head."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    out = C.ANALYSIS_DIR / "smoke.json"
    results = {}
    b = T.D.bundle("ogbg-mollipo")
    ms = T.M.ModelSet(b)
    for name, build in (
        ("G0/Rf", lambda: ms.dense("G0")),
        ("A/Gx-independent", lambda: ms.additive("Gx", active_indices=b.active_global["active_indices"], size_scaler=T.D.size_scaler(b))),
    ):
        model = build()
        task = {"key": f"smoke:{name}", "stage": "smoke", "dataset": "ogbg-mollipo", "model": name, "recipe": "Rf", "epochs": 3, "kind": "train"}
        t0 = time.time()
        summary = T.run_trajectory(
            task=task,
            b=b,
            model=model,
            init_audit={"kind": "smoke"},
            out_dir=C.ANALYSIS_DIR / "_smoke" / name.replace("/", "_"),
            device=device,
            epochs=3,
            recipe="Rf",
        )
        wall = time.time() - t0
        per_epoch = wall / 3.0
        results[name] = {
            "seconds_3_epochs": wall,
            "seconds_per_epoch": per_epoch,
            "predicted_360_seconds": per_epoch * 360,
            "selected": summary.get("selected"),
            "n_parameters": summary.get("n_parameters"),
        }
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    C.write_json(out, {"device": device, "results": results, "environment": C.environment()})
    print(json_dumps(results), flush=True)
    return 0


def json_dumps(payload) -> str:
    import json

    return json.dumps(payload, indent=1, default=str)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default=os.environ.get("RR_POOL", "res-gpu0"))
    parser.add_argument("--deadline", default="2026-11-17T00:20:00Z")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", default=None)
    args = parser.parse_args(argv)
    if args.plan:
        for task in plan(args.pool):
            print(f"{'DONE' if C.task_complete(task) else 'PEND'}  {task['key']}")
        return 0
    if args.smoke:
        return smoke()
    return serve(args.pool, _parse_deadline(args.deadline), force=args.force, only=args.only)


if __name__ == "__main__":
    sys.exit(main())
