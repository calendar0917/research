"""Local scheduler: keep both ``res`` GPU pools busy, pull results as they land.

Runs *locally* and only shells out to the ``rr`` CLI.  It is deliberately
conservative, because remote job state can transiently fail to load and a
double launch would corrupt a shared ``--out-dir``:

1. **Launch ledger.** Every launch is recorded locally (``analysis/queue_ledger.json``)
   with the ident and a timestamp.  An arm is only relaunched when its recorded
   ident is *observed* in a terminal state, or when it has been silent for longer
   than ``--settle-minutes`` (default 60) — never merely because a job listing
   did not mention it.
2. **Fail-closed reconciliation.** If ``rr jobs --json`` cannot be parsed, the
   iteration does nothing but sleep (no launches, no pulls).
3. **One live job per pool.** A pool is busy while *any* live ``bn*`` job (this
   round's ident prefix) is in it.

    uv run python code/lb_queue.py --deadline-minutes 90
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lb_common as C
import lb_dispatch as DISPATCH

LIVE = {"running", "submitted", "queued", "pending"}
DONE = {"complete", "completed"}
BAD = {"failed", "cancelled", "canceled", "error", "unknown"}
LEDGER = C.ANALYSIS / "queue_ledger.json"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def ident_of(arm_id: str, attempt: int = 1) -> str:
    base = ("bn" + arm_id.replace("_", "").lower())[:27]
    return base if attempt == 1 else f"{base[:25]}r{attempt}"


def jobs() -> dict[str, list[dict[str, Any]]] | None:
    """ident -> [job dicts]; ``None`` if the listing could not be read."""
    out = subprocess.run(["rr", "jobs", "--json"], capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout.strip():
        return None
    try:
        payload = json.loads(out.stdout)
    except json.JSONDecodeError:
        return None
    grouped: dict[str, list[dict[str, Any]]] = {}
    for j in payload.get("jobs", []):
        grouped.setdefault(j["experiment"], []).append(j)
    return grouped


def pull(ident: str, arm_id: str) -> bool:
    res = subprocess.run(["rr", "pull", ident, "--light"], capture_output=True, text=True)
    if res.returncode != 0:
        C.log(f"[queue] pull failed {ident}: {res.stderr.strip()[:200]}")
        return False
    hits = sorted((C.REPO / ".rr" / "pulled").glob(f"*/result/{C.track_rel(C.RUNS)}/{arm_id}"))
    if not hits:
        return False
    dst = C.RUNS / arm_id
    dst.mkdir(parents=True, exist_ok=True)
    for f in hits[-1].iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)
    return (dst / "train.json").exists()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deadline-minutes", type=float, default=90.0)
    ap.add_argument("--poll-seconds", type=float, default=45.0)
    ap.add_argument("--settle-minutes", type=float, default=60.0)
    ap.add_argument("--pools", default=",".join(C.GPU_POOLS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    pools = [p for p in args.pools.split(",") if p]
    arms = DISPATCH.matrix()
    ledger: dict[str, Any] = C.read_json(LEDGER) if LEDGER.exists() else {"arms": {}}
    t_end = time.time() + args.deadline_minutes * 60.0

    def done(arm) -> bool:
        return (C.RUNS / arm["arm_id"] / "train.json").exists()

    def record(a: str, ident: str, pool: str) -> None:
        e = ledger["arms"].setdefault(a, {"idents": [], "attempts": 0})
        e["idents"].append(ident)
        e["attempts"] = len(e["idents"])
        e["last_ident"] = ident
        e["last_pool"] = pool
        e["launched_utc"] = _now().isoformat()
        C.write_json(LEDGER, ledger)

    while time.time() < t_end:
        jb = jobs()
        if jb is None:  # fail closed
            time.sleep(args.poll_seconds)
            continue
        busy = {j.get("pool") for name, js in jb.items() if name.startswith("bn")
                for j in js if j.get("state") in LIVE}
        busy = {b for b in busy if b}

        for arm in arms:
            a = arm["arm_id"]
            if done(arm):
                continue
            e = ledger["arms"].get(a)
            if not e:
                continue
            ident = e.get("last_ident")
            runs = jb.get(ident, []) if ident else []
            states = {r.get("state") for r in runs}
            if states & LIVE:
                continue
            if states & DONE:
                if pull(ident, a):
                    C.log(f"[queue] pulled {a}")
                    e["pulled_utc"] = _now().isoformat()
                    C.write_json(LEDGER, ledger)
                continue
            if states & BAD and not (states & LIVE):
                if e["attempts"] >= 2:
                    e["failed_final"] = True
                    C.write_json(LEDGER, ledger)
                    C.log(f"[queue] {a} failed twice; leaving as FAILED")
                continue
            if states & LIVE:
                continue
            # silent: only relaunch after the settle window
            launched = _parse(e.get("launched_utc"))
            if launched and (_now() - launched).total_seconds() < args.settle_minutes * 60:
                continue

        pending = []
        for arm in arms:
            a = arm["arm_id"]
            if done(arm):
                continue
            e = ledger["arms"].get(a)
            if e is None:
                pending.append(arm); continue
            if e.get("failed_final"):
                continue
            ident = e.get("last_ident")
            states = {r.get("state") for r in jb.get(ident, [])} if ident else set()
            if states & (LIVE | DONE):
                continue
            if states & BAD:
                if e["attempts"] >= 2:
                    continue
                pending.append(arm); continue
            launched = _parse(e.get("launched_utc"))
            if launched and (_now() - launched).total_seconds() < args.settle_minutes * 60:
                continue
            pending.append(arm)

        free = [p for p in pools if p not in busy]
        for pool in free:
            if not pending:
                break
            arm = pending.pop(0)
            a = arm["arm_id"]
            attempts = len(ledger["arms"].get(a, {}).get("idents", [])) + 1
            ident = ident_of(a, attempts)
            cmd = ["rr", "run", "res", ident, "--pool", pool, "--allow-dirty",
                   "--result", C.track_rel(C.RUNS / a),
                   "--", "uv", "run", "python", C.track_rel(C.TRACK / "code" / "lb_train.py"),
                   "--model", arm["model"], "--dataset", arm["dataset"], "--fit", arm["fit"],
                   "--device", "cuda", "--out-dir", C.track_rel(C.RUNS / a)]
            if args.dry_run:
                C.log(f"[queue][dry-run] would launch {a} as {ident} on {pool}")
                continue
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                record(a, ident, pool)
                C.log(f"[queue] launched {a} as {ident} on {pool}")
            else:
                C.log(f"[queue] launch failed {a}: {res.stderr.strip()[:300]}")

        settled = all(done(x) or ledger["arms"].get(x["arm_id"], {}).get("failed_final") for x in arms)
        if settled:
            C.log("[queue] all arms settled")
            break
        time.sleep(args.poll_seconds)

    missing = [a["arm_id"] for a in arms if not done(a)]
    C.write_json(C.ANALYSIS / "queue_history.json",
                 {"utc": C.utcnow(), "missing": missing,
                  "done": [a["arm_id"] for a in arms if done(a)], "ledger": ledger})
    C.log(f"[queue] missing={missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
