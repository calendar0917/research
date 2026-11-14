"""Fail-closed launcher for the purchased arms (two-GPU ``rr`` or local CPU).

The ledger is the only source of truth about what has been submitted:

* a job whose ``train.json`` exists is ``COMPLETE`` and is never launched again;
* a job whose ``out_dir`` is non-empty but has no ``train.json`` is
  ``INCOMPLETE_UNKNOWN`` and is **not** relaunched automatically — an unknown or
  temporarily empty queue is never evidence that a job did not start;
* only ``PENDING`` jobs are launched, at most once per attempt number;
* every job gets its own ``out_dir`` and its own log file.

    # primary: one process per A100 pool, serial per pool
    uv run python code/sc_dispatch.py --backend rr --pools res-gpu0,res-gpu1
    # local fallback (documented execution-regime deviation)
    uv run python code/sc_dispatch.py --backend local --parallel 3
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import sc_common as C

LEDGER = C.ANALYSIS / "launch_ledger.json"
LOGS = C.ANALYSIS / "logs"


def job_list(*, dataset: str = C.PRIMARY_DATASET, fits: Sequence[str],
             arms: Sequence[str]) -> list[dict[str, Any]]:
    out = []
    for fit in fits:
        for arm in arms:
            name = "LIP" if dataset == C.PRIMARY_DATASET else "BACE"
            out.append({
                "job": f"{name}_F{fit}_{arm}", "dataset": dataset, "fit": fit, "arm": arm,
                "out_dir": C.track_rel(C.RUNS / f"{name}_F{fit}_{arm}"),
                "priority": C.ARM_PRIORITY[arm],
            })
    out.sort(key=lambda j: (j["fit"] != "100", j["priority"], j["job"]))
    return out


def ledger_load() -> dict[str, Any]:
    if LEDGER.exists():
        return C.read_json(LEDGER)
    return {"round_id": C.ROUND_ID, "jobs": {}, "log": []}


def ledger_save(led: dict[str, Any]) -> None:
    led["updated_utc"] = C.utcnow()
    C.write_json(LEDGER, led)


def job_state(job: dict[str, Any], led: dict[str, Any]) -> str:
    train_json = C.REPO / job["out_dir"] / "train.json"
    if train_json.exists():
        return "COMPLETE"
    rec = led["jobs"].get(job["job"])
    if rec is None:
        out = (C.REPO / job["out_dir"])
        if out.exists() and any(out.iterdir()):
            return "INCOMPLETE_UNKNOWN"
        return "PENDING"
    st = rec.get("status")
    if st in ("COMPLETE", "FAILED", "CANCELLED"):
        return st
    return "INCOMPLETE_UNKNOWN"         # RUNNING / LAUNCHED / unknown -> never relaunch


def local_cmd(job: dict[str, Any]) -> list[str]:
    return ["uv", "run", "python", C.track_rel(C.CODE_DIR / "sc_train.py"),
            "--dataset", job["dataset"], "--fit", job["fit"], "--arm", job["arm"],
            "--device", "cpu", "--out-dir", job["out_dir"]]


def rr_cmd(job: dict[str, Any], pool: str, attempt: int) -> list[str]:
    ident = job["job"].replace("_", "").lower()
    if attempt > 1:
        ident = f"{ident[:24]}r{attempt}"
    return ["rr", "run", "res", ident, "--pool", pool, "--allow-dirty",
            "--result", job["out_dir"], "--",
            "uv", "run", "python", C.track_rel(C.CODE_DIR / "sc_train.py"),
            "--dataset", job["dataset"], "--fit", job["fit"], "--arm", job["arm"],
            "--device", "cuda", "--out-dir", job["out_dir"]]


def launch_rr(job: dict[str, Any], pool: str, led: dict[str, Any]) -> None:
    rec = led["jobs"].get(job["job"], {"attempts": 0})
    attempt = int(rec.get("attempts", 0)) + 1
    cmd = rr_cmd(job, pool, attempt)
    C.log(f"[dispatch] rr launch {job['job']} on {pool} (attempt {attempt})")
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(C.REPO))
    rec.update({"attempts": attempt, "pool": pool, "out_dir": job["out_dir"],
                "started": C.utcnow(), "status": "LAUNCHED",
                "ident": cmd[3], "rc": proc.returncode})
    if proc.returncode != 0:
        rec["status"] = "FAILED"
        rec["stderr"] = proc.stderr.strip()[-1500:]
        C.log(f"[dispatch] launch failed: {proc.stderr.strip()[:200]}")
    led["jobs"][job["job"]] = rec
    led["log"].append({"utc": C.utcnow(), "event": rec["status"], "job": job["job"],
                       "pool": pool, "rc": proc.returncode})
    ledger_save(led)


def poll_rr(led: dict[str, Any]) -> None:
    for name, rec in led["jobs"].items():
        if rec.get("status") not in ("LAUNCHED", "RUNNING"):
            continue
        ident = rec.get("ident")
        proc = subprocess.run(["rr", "status", ident], capture_output=True, text=True)
        state = ""
        for line in proc.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "state":
                state = parts[-1]
        train_json = C.REPO / rec["out_dir"] / "train.json"
        if state == "complete" and train_json.exists():
            rec["status"] = "COMPLETE"
            rec["finished"] = C.utcnow()
            led["log"].append({"utc": C.utcnow(), "event": "COMPLETE", "job": name})
        elif state in ("failed", "cancelled"):
            rec["status"] = "FAILED" if state == "failed" else "CANCELLED"
            rec["finished"] = C.utcnow()
            led["log"].append({"utc": C.utcnow(), "event": rec["status"], "job": name})
        elif state:
            rec["status"] = "RUNNING"
    ledger_save(led)


def run_local(jobs: list[dict[str, Any]], led: dict[str, Any], parallel: int) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    running: dict[str, subprocess.Popen] = {}
    queue = list(jobs)
    while queue or running:
        while queue and len(running) < parallel:
            job = queue.pop(0)
            st = job_state(job, led)
            if st in ("COMPLETE", "INCOMPLETE_UNKNOWN", "FAILED", "CANCELLED"):
                C.log(f"[dispatch] skip {job['job']} ({st})")
                if st == "FAILED":
                    led["jobs"].setdefault(job["job"], {})["status"] = "FAILED"
                continue
            log_path = LOGS / f"{job['job']}.log"
            fh = open(log_path, "w", encoding="utf-8")
            env = {**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
            proc = subprocess.Popen(local_cmd(job), cwd=str(C.REPO), stdout=fh,
                                    stderr=subprocess.STDOUT, env=env)
            fh.close()
            running[job["job"]] = proc
            led["jobs"][job["job"]] = {"status": "RUNNING", "attempts": 1, "pool": "local",
                                       "out_dir": job["out_dir"], "started": C.utcnow(),
                                       "log": C.track_rel(log_path)}
            led["log"].append({"utc": C.utcnow(), "event": "LAUNCHED", "job": job["job"],
                               "pool": "local"})
            ledger_save(led)
            C.log(f"[dispatch] local launch {job['job']} (log {log_path.name})")
        time.sleep(3)
        for name, proc in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            running.pop(name)
            rec = led["jobs"][name]
            rec.update({"rc": rc, "finished": C.utcnow(),
                        "status": "COMPLETE" if rc == 0 else "FAILED"})
            led["log"].append({"utc": C.utcnow(), "event": rec["status"], "job": name,
                               "rc": rc})
            ledger_save(led)
            C.log(f"[dispatch] {name} -> {rec['status']} (rc={rc})")
    ledger_save(led)


def run_rr(jobs: list[dict[str, Any]], led: dict[str, Any], pools: Sequence[str],
           poll_seconds: float, stop_at_epoch: float | None) -> None:
    pools = list(pools)
    slot: dict[str, str | None] = {p: None for p in pools}
    queue = list(jobs)
    while True:
        poll_rr(led)
        for p in pools:
            name = slot[p]
            if name and led["jobs"][name]["status"] in ("COMPLETE", "FAILED", "CANCELLED"):
                slot[p] = None
        for p in pools:
            if slot[p] is not None:
                continue
            while queue:
                job = queue.pop(0)
                st = job_state(job, led)
                if st != "PENDING":
                    C.log(f"[dispatch] skip {job['job']} ({st})")
                    continue
                launch_rr(job, p, led)
                if led["jobs"][job["job"]]["status"] == "LAUNCHED":
                    slot[p] = job["job"]
                break
        if not queue and all(v is None for v in slot.values()):
            C.log("[dispatch] queue exhausted")
            break
        if stop_at_epoch and time.time() >= stop_at_epoch:
            C.log("[dispatch] wall-clock stop reached; leaving the rest PENDING")
            break
        time.sleep(poll_seconds)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="launch the purchased arms (fail-closed ledger)")
    ap.add_argument("--backend", default="rr", choices=["rr", "local"])
    ap.add_argument("--dataset", default=C.PRIMARY_DATASET)
    ap.add_argument("--fits", default="100,25")
    ap.add_argument("--arms", default=",".join(C.ARMS))
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--pools", default=",".join(C.GPU_POOLS))
    ap.add_argument("--poll-seconds", type=float, default=45.0)
    ap.add_argument("--max-wall-minutes", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    led = ledger_load()
    jobs = job_list(dataset=args.dataset, fits=[f for f in args.fits.split(",") if f],
                    arms=[a for a in args.arms.split(",") if a])
    C.log(f"[dispatch] {len(jobs)} jobs, backend={args.backend}")
    for job in jobs:
        C.log(f"  {job['job']:<24} state={job_state(job, led)}")
    if args.dry_run:
        return 0
    stop_at = (time.time() + args.max_wall_minutes * 60) if args.max_wall_minutes else None
    if args.backend == "local":
        run_local(jobs, led, max(1, args.parallel))
    else:
        run_rr(jobs, led, [p for p in args.pools.split(",") if p], args.poll_seconds, stop_at)
    final = {job["job"]: job_state(job, led) for job in jobs}
    C.write_json(C.ANALYSIS / "dispatch_summary.json", final)
    C.log(f"[dispatch] final states: {json.dumps(final, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
