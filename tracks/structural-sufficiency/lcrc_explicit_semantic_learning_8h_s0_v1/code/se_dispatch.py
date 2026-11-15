"""Fixed-manifest dispatcher: consume ``configs/run_matrix.csv`` on two A100s.

Every launch goes through ``rr`` (no manual ssh/nohup).  The dispatcher decides
*which* frozen row to launch next, keeps at most one training process per GPU,
records operator-side state (RUNNING/COMPLETE/FAILED + exit code + wall) and
enforces the T0 schedule (soft stop for new big jobs at T0+6h45, lock at
T0+7h15).

It is intentionally dumb: no science decisions, no re-ordering inside a
priority class, at most one infrastructure retry per row.

    uv run python code/se_dispatch.py --dry-run
    uv run python code/se_dispatch.py --stop-at 2026-11-15T06:20:00Z
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

import se_common as C

GPU_POOLS = ("res-gpu0", "res-gpu1")
STATE = C.ANALYSIS / "dispatch_state.json"
MATRIX = C.CONFIGS / "run_matrix.csv"
ALLOWED_DIRTY = (C.track_rel(MATRIX),)
EXPERIMENT_PREFIX = "selsem"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_utc(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def read_matrix() -> list[dict[str, Any]]:
    with open(MATRIX, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_matrix(rows: Sequence[dict[str, Any]]) -> None:
    with open(MATRIX, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def dirty_paths() -> list[str]:
    proc = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=str(C.REPO))
    out = []
    for line in proc.stdout.splitlines():
        if len(line) > 3:
            out.append(line[3:].strip())
    return out


@dataclass
class Slot:
    pool: str
    ident: str | None = None
    run_id: str | None = None
    started: datetime | None = None
    status: str = "idle"


def launch(pool: str, row: dict[str, Any], *, extra_args: Sequence[str] = (), attempt: int = 1) -> Slot:
    run_id = row["run_id"]
    ident = f"{EXPERIMENT_PREFIX}{run_id}".replace("_", "").lower()[:30]
    if attempt > 1:
        ident = f"{ident[:28]}r{attempt}"
    bad = [p for p in dirty_paths() if p not in ALLOWED_DIRTY]
    if bad:
        raise SystemExit(f"refusing to launch with dirty code paths: {bad}")
    cmd = [
        "rr", "run", "res", ident, "--pool", pool, "--allow-dirty",
        "--result", C.track_rel(C.RUNS),
        "--", "uv", "run", "python",
        C.track_rel(Path(__file__).resolve().parent / "se_train.py"),
        "--family", row["family"], "--recipe", row["recipe"],
        "--dataset", row["dataset"], "--budget", row["budget"],
        "--device", "cuda", *extra_args,
    ]
    C.log(f"[dispatch] launch {run_id} on {pool} ({row['family']}-{row['recipe']} {row['dataset']}/{row['budget']})")
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(C.REPO))
    if proc.returncode != 0:
        C.log(f"[dispatch] launch failed {run_id}: {proc.stderr.strip()[:400]}")
        return Slot(pool=pool, status="failed")
    return Slot(pool=pool, ident=ident, run_id=run_id, started=_now(), status="running")


def poll(slot: Slot) -> tuple[str, str]:
    if not slot.ident:
        return "failed", ""
    proc = subprocess.run(["rr", "status", slot.ident, "--json"], capture_output=True, text=True, cwd=str(C.REPO))
    import json

    try:
        data = json.loads(proc.stdout)
        state = str(data.get("state") or "unknown")
        code = str(data.get("exit_code") if data.get("exit_code") is not None else "")
        return state, code
    except Exception:
        state, code = "", ""
        for line in proc.stdout.splitlines():
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "state":
                state = parts[-1]
            elif parts[0] == "exit_code":
                code = parts[-1]
        return state or "unknown", code


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="dispatch the frozen semantic-sharing run matrix")
    ap.add_argument("--stop-at", default=None, help="UTC timestamp; no new launch after it")
    ap.add_argument("--soft-stop", default=None, help="UTC; no new BIG job after it")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-runs", type=int, default=None)
    ap.add_argument("--poll-seconds", type=float, default=20.0)
    ap.add_argument("--keep-pools", default=",".join(GPU_POOLS))
    ap.add_argument("--priorities", default=None)
    ap.add_argument("--once", action="store_true", help="one scheduling pass then exit (launch-only)")
    args = ap.parse_args(argv)
    stop_at = _parse_utc(args.stop_at) if args.stop_at else None
    soft_stop = _parse_utc(args.soft_stop) if args.soft_stop else None
    pools = [p for p in args.keep_pools.split(",") if p]
    prio = {int(p) for p in args.priorities.split(",")} if args.priorities else None

    state = C.read_json(STATE) if STATE.exists() else {"runs": {}, "log": []}
    rows = read_matrix()
    slots = [Slot(pool=p) for p in pools]
    launched = 0
    while True:
        pending = [r for r in rows if r["status"] == "PENDING" and (prio is None or int(r["priority"]) in prio)]
        pending.sort(key=lambda r: (int(r["priority"]), r["run_id"]))
        for slot in slots:
            if slot.ident and slot.status == "running":
                st, code = poll(slot)
                if st in ("complete", "failed", "cancelled"):
                    row = next(r for r in rows if r["run_id"] == slot.run_id)
                    ok = st == "complete" and code in ("0", "")
                    row["status"] = "COMPLETE" if ok else "FAILED"
                    row["note"] = f"exit={code} wall={(_now() - slot.started).total_seconds():.0f}s"
                    state["runs"][slot.run_id] = {"pool": slot.pool, "ident": slot.ident, "state": st, "exit": code,
                                                  "started": slot.started.isoformat(), "finished": _now().isoformat()}
                    state["log"].append({"utc": C.utcnow(), "event": row["status"], "run_id": slot.run_id,
                                         "pool": slot.pool, "exit": code})
                    C.log(f"[dispatch] {slot.run_id} -> {row['status']} exit={code}")
                    slot.ident = None
                    slot.run_id = None
                    slot.status = "idle"
                    write_matrix(rows)
                    C.write_json(STATE, state)
            if slot.status == "idle" and pending:
                if stop_at and _now() >= stop_at:
                    continue
                if args.max_runs is not None and launched >= args.max_runs:
                    continue
                row = pending.pop(0)
                if args.dry_run:
                    C.log(f"[dispatch][dry-run] would launch {row['run_id']} on {slot.pool}")
                    row["status"] = "COMPLETE"
                    launched += 1
                    continue
                new = launch(slot.pool, row)
                launched += 1
                if new.ident:
                    slots[slots.index(slot)] = new
                    row["status"] = "RUNNING"
                    write_matrix(rows)
        if args.dry_run:
            if all(r["status"] != "PENDING" for r in rows if prio is None or int(r["priority"]) in prio):
                break
            continue
        if args.once:
            break
        if all(s.status == "idle" for s in slots) and not pending:
            C.log("[dispatch] matrix exhausted for the selected priorities")
            break
        if all(s.status == "idle" for s in slots) and stop_at and _now() >= stop_at:
            C.log("[dispatch] stop-at reached with idle GPUs; exiting")
            break
        if all(s.status == "idle" for s in slots) and soft_stop and _now() >= soft_stop:
            C.log("[dispatch] soft-stop reached with idle GPUs; exiting")
            break
        time.sleep(args.poll_seconds)
    left = sum(1 for r in rows if r["status"] == "PENDING")
    C.log(f"[dispatch] finished; launched={launched} pending_left={left}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
