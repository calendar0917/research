"""Dispatch the frozen arm matrix on the two A100 GPUs (``res`` pools).

Every launch goes through the ``rr`` CLI (never manual ssh / nohup).  At most one
training process per pool.  No science decision, no re-ordering inside a
priority class, no retries beyond one infrastructure retry per arm.

    uv run python code/lb_dispatch.py --dry-run
    uv run python code/lb_dispatch.py --stop-at 2026-11-14T09:00:00Z
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import lb_common as C

STATE = Path(os.environ.get("LB_DISPATCH_STATE", str(C.ANALYSIS / "dispatch_state.json")))


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_utc(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


#: priority: (model, dataset, fit).  1 = start first (full-N C1 to get a prefix fast)
PRIORITY = {
    ("C1_R1", "Lipophilicity", "100"): 1,
    ("DMPNN_D2F", "Lipophilicity", "100"): 1,
    ("C1_R1", "Lipophilicity", "25"): 2,
    ("DMPNN_D2F", "Lipophilicity", "25"): 2,
    ("C1_R1", "Lipophilicity", "50"): 2,
    ("DMPNN_D2F", "Lipophilicity", "50"): 2,
    ("C1_R3", "Lipophilicity", "100"): 3,
    ("CHEM8_R1", "Lipophilicity", "100"): 3,
    ("K_R1", "Lipophilicity", "100"): 3,
    ("C1_R1", "BACE", "100"): 3,
    ("DMPNN_D2F", "BACE", "100"): 3,
    ("C1_R1", "BACE", "50"): 4,
    ("DMPNN_D2F", "BACE", "50"): 4,
}


def matrix() -> list[dict[str, Any]]:
    arms = C.all_arms()
    for a in arms:
        a["priority"] = PRIORITY.get((a["model"], a["dataset"], a["fit"]), 4)
    arms.sort(key=lambda a: (a["priority"], a["arm_id"]))
    return arms


@dataclass
class Slot:
    pool: str
    ident: str | None = None
    arm_id: str | None = None
    started: datetime | None = None
    status: str = "idle"


def _dirty_ok() -> list[str]:
    proc = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=str(C.REPO))
    return [ln[3:].strip() for ln in proc.stdout.splitlines() if len(ln) > 3]


def launch(pool: str, arm: dict[str, Any], attempt: int = 1) -> Slot:
    arm_id = arm["arm_id"]
    ident = f"bn{arm_id}".replace("_", "").lower()[:28]
    if attempt > 1:
        ident = f"{ident[:26]}r{attempt}"
    cmd = ["rr", "run", "res", ident, "--pool", pool, "--allow-dirty",
           "--result", C.track_rel(C.RUNS / arm_id),
           "--", "uv", "run", "python", C.track_rel(Path(__file__).resolve().parent / "lb_train.py"),
           "--model", arm["model"], "--dataset", arm["dataset"], "--fit", arm["fit"],
           "--device", "cuda", "--out-dir", C.track_rel(C.RUNS / arm_id)]
    C.log(f"[dispatch] launch {arm_id} on {pool}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        C.log(f"[dispatch] launch failed {arm_id}: {proc.stderr.strip()[:400]}")
        return Slot(pool=pool, ident=None, arm_id=None, status="failed")
    return Slot(pool=pool, ident=ident, arm_id=arm_id, started=_now(), status="running")


def poll(slot: Slot) -> tuple[str, str]:
    if not slot.ident:
        return "failed", ""
    proc = subprocess.run(["rr", "status", slot.ident], capture_output=True, text=True)
    state = code = ""
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "state":
            state = parts[-1]
        elif len(parts) >= 2 and parts[0] == "exit_code":
            code = parts[-1]
    return state or "unknown", code


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="dispatch the bottleneck-diagnosis arms")
    ap.add_argument("--stop-at", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-runs", type=int, default=None)
    ap.add_argument("--poll-seconds", type=float, default=30.0)
    ap.add_argument("--keep-pools", default=",".join(C.GPU_POOLS))
    args = ap.parse_args(argv)
    stop_at = _parse_utc(args.stop_at) if args.stop_at else None
    pools = [p for p in args.keep_pools.split(",") if p]
    state: dict[str, Any] = C.read_json(STATE) if STATE.exists() else {"runs": {}, "log": []}
    arms = matrix()
    for a in arms:
        d = C.RUNS / a["arm_id"] / "train.json"
        a["status"] = "COMPLETE" if d.exists() else "PENDING"
    slots = [Slot(pool=p) for p in pools]
    launched = 0
    while True:
        pending = [a for a in arms if a["status"] == "PENDING"]
        for i, slot in enumerate(slots):
            if slot.ident and slot.status == "running":
                st, code = poll(slot)
                if st in ("complete", "failed", "cancelled", "unknown") and st != "unknown":
                    arm = next(a for a in arms if a["arm_id"] == slot.arm_id)
                    ok = st == "complete" and code in ("0", "")
                    arm["status"] = "COMPLETE" if ok else "FAILED"
                    state["runs"][slot.arm_id] = {"pool": slot.pool, "ident": slot.ident, "state": st,
                                                  "exit": code, "started": slot.started.isoformat(),
                                                  "finished": _now().isoformat()}
                    state["log"].append({"utc": C.utcnow(), "event": arm["status"], "arm_id": slot.arm_id,
                                         "pool": slot.pool, "exit": code})
                    C.write_json(STATE, state)
                    C.log(f"[dispatch] {slot.arm_id} -> {arm['status']} exit={code}")
                    slots[i] = Slot(pool=slot.pool)
        pending = [a for a in arms if a["status"] == "PENDING"]
        for i, slot in enumerate(slots):
            if slot.status != "idle" or not pending:
                continue
            if stop_at and _now() >= stop_at:
                break
            if args.max_runs is not None and launched >= args.max_runs:
                break
            arm = pending.pop(0)
            if args.dry_run:
                C.log(f"[dispatch][dry-run] would launch {arm['arm_id']} on {slot.pool}")
                arm["status"] = "COMPLETE"
                launched += 1
                continue
            new = launch(slot.pool, arm)
            launched += 1
            if new.ident:
                slots[i] = new
                arm["status"] = "RUNNING"
            else:
                arm["status"] = "FAILED"
                slots[i] = Slot(pool=slot.pool)
        if args.dry_run:
            if all(a["status"] != "PENDING" for a in arms):
                break
            continue
        pending = [a for a in arms if a["status"] == "PENDING"]
        all_idle = all(s.status == "idle" for s in slots)
        if all_idle and not pending:
            C.log("[dispatch] matrix exhausted")
            break
        if all_idle and ((stop_at and _now() >= stop_at) or
                         (args.max_runs is not None and launched >= args.max_runs)):
            C.log("[dispatch] stopping: budget reached with pending arms")
            break
        time.sleep(args.poll_seconds)
    C.log(f"[dispatch] finished; launched={launched}; "
          f"pending={sum(1 for a in arms if a['status'] == 'PENDING')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
