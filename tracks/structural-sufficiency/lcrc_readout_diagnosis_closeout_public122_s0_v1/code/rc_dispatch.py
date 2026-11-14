"""Fail-closed dispatcher for the closeout round (``rr`` only; no raw ssh/nohup).

Jobs (exactly one per pool at a time, each into its own out_dir):

1. ``rcsrc``     source audit + identity-keyed clean-h cache (gpu0)
2. ``rcpwarm``   P_WARM probe (gpu0)
3. ``rcppost``   P_POST probe (gpu1, concurrent with p_warm)
4. ``rcppre``    P_PRE probe  (gpu0, after p_warm)

The dispatcher never treats an empty/unknown listing as "no job": every job must
reach an explicit terminal state (``complete``/``failed``/``cancelled``).  A job
is launched at most once; technical failures are reported, not silently retried.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import rc_common as R

LEDGER = R.ANALYSIS / "launch_ledger.json"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Job:
    name: str
    pool: str
    out_dir: Path
    result_path: Path
    args: list[str]
    requires: list[str] = field(default_factory=list)
    status: str = "PENDING"       # PENDING -> RUNNING -> COMPLETE/FAILED/CANCELLED
    ident: str | None = None
    attempts: int = 0
    started: str | None = None
    finished: str | None = None
    exit_code: str | None = None


def build_jobs() -> list[Job]:
    code = R.CODE_DIR
    src_rel = R.track_rel(code / "rc_source_audit.py")
    probe_rel = R.track_rel(code / "rc_probe.py")
    jobs: list[Job] = [
        Job("rcsrc", "res-gpu0", R.ANALYSIS, R.ANALYSIS / "source_audit.json",
            [src_rel, "--device", "cuda", "--out", R.track_rel(R.ANALYSIS / "source_audit.json")]),
        Job("rcpwarm", "res-gpu0", R.RUNS / "P_WARM", R.RUNS / "P_WARM",
            [probe_rel, "--rule", "P_WARM", "--device", "cuda", "--out-dir", R.track_rel(R.RUNS / "P_WARM")],
            requires=["rcsrc"]),
        Job("rcppost", "res-gpu1", R.RUNS / "P_POST", R.RUNS / "P_POST",
            [probe_rel, "--rule", "P_POST", "--device", "cuda", "--out-dir", R.track_rel(R.RUNS / "P_POST")],
            requires=["rcsrc"]),
        Job("rcppre", "res-gpu0", R.RUNS / "P_PRE", R.RUNS / "P_PRE",
            [probe_rel, "--rule", "P_PRE", "--device", "cuda", "--out-dir", R.track_rel(R.RUNS / "P_PRE")],
            requires=["rcpwarm", "rcppost"]),
    ]
    return jobs


def launch(job: Job, attempt: int = 1) -> bool:
    ident = job.name if attempt == 1 else f"{job.name}r{attempt}"
    cmd = ["rr", "run", "res", ident, "--pool", job.pool,
           "--result", R.track_rel(job.result_path),
           "--", "uv", "run", "python", *job.args]
    R.log(f"[dispatch] launch {ident} pool={job.pool} -> {cmd[-4:]}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        R.log(f"[dispatch] launch FAILED {ident}: {(proc.stderr or proc.stdout).strip()[:500]}")
        return False
    job.ident = ident
    job.status = "RUNNING"
    job.attempts = attempt
    job.started = _now()
    return True


def poll(job: Job) -> tuple[str, str]:
    if not job.ident:
        return "unknown", ""
    proc = subprocess.run(["rr", "status", job.ident, "--json"], capture_output=True, text=True)
    if proc.returncode != 0:
        return "unknown", ""
    try:
        import json
        data = json.loads(proc.stdout)
        state = str(data.get("state", "unknown"))
        code = str(data.get("exit_code", ""))
        return state, code
    except Exception:
        state = code = ""
        for line in proc.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "state":
                state = parts[-1]
            elif len(parts) >= 2 and parts[0] == "exit_code":
                code = parts[-1]
        return state or "unknown", code


TERMINAL = {"complete", "completed", "failed", "cancelled", "error"}
_STATE_MAP = {"complete": "COMPLETE", "completed": "COMPLETE", "failed": "FAILED",
              "cancelled": "CANCELLED", "error": "FAILED"}


def run_all(*, dry_run: bool = False, poll_seconds: float = 15.0) -> dict[str, Any]:
    jobs = build_jobs()
    by_name = {j.name: j for j in jobs}
    ledger: dict[str, Any] = {"round_id": R.ROUND_ID, "jobs": {}, "log": []}
    if LEDGER.exists():
        ledger = R.read_json(LEDGER)

    def record() -> None:
        ledger["jobs"] = {j.name: {"pool": j.pool, "ident": j.ident, "status": j.status,
                                   "attempts": j.attempts, "started": j.started,
                                   "finished": j.finished, "exit_code": j.exit_code,
                                   "out_dir": R.track_rel(j.out_dir)} for j in jobs}
        ledger["updated_utc"] = _now()
        R.write_json(LEDGER, ledger)

    if dry_run:
        for j in jobs:
            R.log(f"[dispatch][dry-run] {j.name} pool={j.pool} requires={j.requires}")
        return ledger

    R.log("[dispatch] ensuring remote deployment")
    dep = subprocess.run(["rr", "deploy", "res"], capture_output=True, text=True)
    if dep.returncode != 0:
        R.log(f"[dispatch] deploy FAILED: {(dep.stderr or dep.stdout).strip()[:500]}")
        ledger["log"].append({"utc": _now(), "event": "deploy_failed"})
        record()
        return ledger
    R.log("[dispatch] deploy OK")

    deadline = time.time() + R.WALL_HARD_HOURS * 3600
    while True:
        if time.time() > deadline:
            R.log("[dispatch] hard wall reached; stopping launches")
            break
        progressed = False
        for job in jobs:
            if job.status == "RUNNING":
                state, code = poll(job)
                if state in TERMINAL:
                    job.status = _STATE_MAP[state]
                    job.exit_code = code
                    job.finished = _now()
                    ledger["log"].append({"utc": _now(), "event": job.status, "job": job.name,
                                          "ident": job.ident, "exit": code})
                    R.log(f"[dispatch] {job.name} -> {job.status} exit={code}")
                    record()
                    progressed = True
        # launch PENDING whose requirements are COMPLETE
        busy_pools = {j.pool for j in jobs if j.status == "RUNNING"}
        for job in jobs:
            if job.status != "PENDING" or job.pool in busy_pools:
                continue
            if not all(by_name[r].status == "COMPLETE" for r in job.requires):
                if any(by_name[r].status in ("FAILED", "CANCELLED") for r in job.requires):
                    job.status = "SKIPPED"
                    R.log(f"[dispatch] {job.name} skipped: prerequisite failed")
                    record()
                continue
            if launch(job):
                busy_pools.add(job.pool)
                ledger["log"].append({"utc": _now(), "event": "LAUNCHED", "job": job.name,
                                      "ident": job.ident})
                record()
                progressed = True
            else:
                job.status = "FAILED"
                record()
                progressed = True
        if all(j.status in ("COMPLETE", "FAILED", "CANCELLED", "SKIPPED") for j in jobs):
            R.log("[dispatch] all jobs terminal")
            break
        if not progressed:
            time.sleep(poll_seconds)
    record()
    statuses = {j.name: j.status for j in jobs}
    R.log(f"[dispatch] final statuses: {statuses}")
    return ledger


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="closeout dispatcher")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--poll-seconds", type=float, default=15.0)
    args = ap.parse_args(argv)
    ledger = run_all(dry_run=args.dry_run, poll_seconds=args.poll_seconds)
    ok = all(v["status"] == "COMPLETE" for v in ledger.get("jobs", {}).values())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
