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
ALLOWED_DIRTY = ()
EXPERIMENT_PREFIX = "selsem"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_utc(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def read_matrix() -> list[dict[str, Any]]:
    with open(MATRIX, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_state(state: dict[str, Any]) -> None:
    """Status/log live only in the gitignored state file; the frozen matrix is
    never rewritten (so redeploys are never blocked by queue bookkeeping)."""
    C.write_json(STATE, state)


def dirty_paths() -> list[str]:
    """Tracked modifications only (untracked user files are not our edits)."""
    proc = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                          capture_output=True, text=True, cwd=str(C.REPO))
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


def ensure_deployed(rr_config: str | None) -> bool:
    """Deploy local HEAD to the host if the remote checkout is behind.

    The dispatcher may be restarted after an analysis commit moved local HEAD
    ahead of the deployed revision; rather than stalling every launch we sync
    the host ourselves (still exclusively through ``rr``)."""
    cmd = ["rr"]
    if rr_config:
        cmd += ["--config", rr_config]
    cmd += ["deploy", "res"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(C.REPO))
    if proc.returncode != 0:
        C.log(f"[dispatch] deploy failed: {proc.stderr.strip()[:300]}")
        return False
    return True


def launch(pool: str, row: dict[str, Any], *, extra_args: Sequence[str] = (), rr_config: str | None = None, attempt: int = 1) -> Slot:
    run_id = row["run_id"]
    ident = f"{EXPERIMENT_PREFIX}{run_id}".replace("_", "").lower()[:30]
    if attempt > 1:
        ident = f"{ident[:28]}r{attempt}"
    out_name = C.arm_name(row["family"], row["recipe"], row["dataset"], row["budget"])
    out_dir = C.RUNS / f"arm_{out_name}"
    bad = [p for p in dirty_paths() if p not in ALLOWED_DIRTY]
    if bad:
        raise SystemExit(f"refusing to launch with dirty code paths: {bad}")
    cmd = ["rr"]
    if rr_config:
        cmd += ["--config", rr_config]
    cmd += [
        "run", "res", ident, "--pool", pool, "--allow-dirty",
        "--result", C.track_rel(out_dir),
        "--", "uv", "run", "python",
        C.track_rel(Path(__file__).resolve().parent / "se_train.py"),
        "--family", row["family"], "--recipe", row["recipe"],
        "--dataset", row["dataset"], "--budget", row["budget"],
        "--device", "cuda", "--out-dir", C.track_rel(out_dir), *extra_args,
    ]
    C.log(f"[dispatch] launch {run_id} on {pool} ({row['family']}-{row['recipe']} {row['dataset']}/{row['budget']})")
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(C.REPO))
    if proc.returncode != 0 and "remote checkout is not at the local commit" in proc.stderr:
        C.log("[dispatch] remote behind local HEAD; deploying and retrying once")
        if ensure_deployed(rr_config):
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(C.REPO))
    if proc.returncode != 0:
        C.log(f"[dispatch] launch failed {run_id}: {proc.stderr.strip()[:400]}")
        return Slot(pool=pool, status="failed")
    return Slot(pool=pool, ident=ident, run_id=run_id, started=_now(), status="running")


def poll(slot: Slot, rr_config: str | None = None) -> tuple[str, str]:
    if not slot.ident:
        return "failed", ""
    cmd = ["rr"]
    if rr_config:
        cmd += ["--config", rr_config]
    cmd += ["status", slot.ident, "--json"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(C.REPO))
    import json

    try:
        data = json.loads(proc.stdout)
        state = _norm_state(str(data.get("state") or "unknown"))
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


def _norm_state(st: str) -> str:
    st = (st or "").strip().lower()
    if st.startswith("complete"):
        return "complete"
    if st.startswith("fail") or st.startswith("error"):
        return "failed"
    if st.startswith("cancel"):
        return "cancelled"
    return st


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="dispatch the frozen semantic-sharing run matrix")
    ap.add_argument("--stop-at", default=None, help="UTC timestamp; no new launch after it")
    ap.add_argument("--soft-stop", default=None, help="UTC; no new BIG job after it")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-runs", type=int, default=None)
    ap.add_argument("--poll-seconds", type=float, default=20.0)
    ap.add_argument("--keep-pools", default=",".join(GPU_POOLS))
    ap.add_argument("--rr-config", default=None, help="path to an rr config override")
    ap.add_argument("--priorities", default=None)
    ap.add_argument("--once", action="store_true", help="one scheduling pass then exit (launch-only)")
    args = ap.parse_args(argv)
    stop_at = _parse_utc(args.stop_at) if args.stop_at else None
    soft_stop = _parse_utc(args.soft_stop) if args.soft_stop else None
    pools = [p for p in args.keep_pools.split(",") if p]
    prio = {int(p) for p in args.priorities.split(",")} if args.priorities else None

    state = C.read_json(STATE) if STATE.exists() else {"runs": {}, "status": {}, "log": []}
    state.setdefault("status", {})
    state.setdefault("runs", {})
    state.setdefault("log", [])
    rows = read_matrix()
    for r in rows:
        # the frozen matrix default is only a starting point; the live status
        # lives in the gitignored state file so redeploys are never blocked
        if r["run_id"] in state["status"]:
            r["status"] = state["status"][r["run_id"]]
    slots = [Slot(pool=p) for p in pools]
    # resume: re-attach slots to any run the previous session left RUNNING
    for run_id, st in list(state["status"].items()):
        if st != "RUNNING":
            continue
        rec = state["runs"].get(run_id) or {}
        ident = rec.get("ident")
        pool = rec.get("pool")
        if not ident:
            state["status"][run_id] = "PENDING"
            continue
        attached = False
        for s in slots:
            if s.pool == pool and s.status == "idle":
                s.ident = ident
                s.run_id = run_id
                s.status = "running"
                s.started = _now()
                attached = True
                break
        if not attached:
            C.log(f"[dispatch] cannot re-attach {run_id} ({ident}); left RUNNING for inspection")
    launched = 0
    while True:
        pending = [r for r in rows if r["status"] == "PENDING" and (prio is None or int(r["priority"]) in prio)]
        pending.sort(key=lambda r: (int(r["priority"]), r["run_id"]))
        for slot in slots:
            if slot.ident and slot.status == "running":
                st, code = poll(slot, args.rr_config)
                if st in ("complete", "failed", "cancelled"):
                    ok = st == "complete" and code in ("0", "")
                    row["status"] = "COMPLETE" if ok else "FAILED"
                    row["note"] = f"exit={code} wall={(_now() - slot.started).total_seconds():.0f}s"
                    state["status"][slot.run_id] = row["status"]
                    state["runs"][slot.run_id] = {"pool": slot.pool, "ident": slot.ident, "state": st, "exit": code,
                                                  "started": slot.started.isoformat(), "finished": _now().isoformat()}
                    state["log"].append({"utc": C.utcnow(), "event": row["status"], "run_id": slot.run_id,
                                         "pool": slot.pool, "exit": code})
                    C.log(f"[dispatch] {slot.run_id} -> {row['status']} exit={code}")
                    slot.ident = None
                    slot.run_id = None
                    slot.status = "idle"
                    write_state(state)
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
                new = launch(slot.pool, row, rr_config=args.rr_config)
                launched += 1
                if new.ident:
                    slots[slots.index(slot)] = new
                    row["status"] = "RUNNING"
                    state["status"][row["run_id"]] = "RUNNING"
                    write_state(state)
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
