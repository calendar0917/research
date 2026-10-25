"""Build analysis/gpu_ledger.json from `rr jobs --json` (occupancy accounting).

Occupancy is wall-clock (completed_at - started_at) per job, summed over both
A100s, exactly as the round brief defines the budget.  Official test is not
involved.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime

import scm_common as S


def parse(ts: str | None):
    return datetime.fromisoformat(ts) if ts else None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prefix", default="scm-")
    p.add_argument("--budget", type=float, default=2.5)
    args = p.parse_args(argv)
    raw = subprocess.check_output(["rr", "jobs", "--json"], text=True)
    payload = json.loads(raw)
    jobs = []
    now = datetime.now().astimezone()
    for j in payload.get("jobs", []):
        if not str(j.get("experiment", "")).startswith(args.prefix):
            continue
        st, en = parse(j.get("started_at")), parse(j.get("completed_at"))
        hours = ((en or now) - st).total_seconds() / 3600.0 if st else 0.0
        jobs.append({
            "experiment": j["experiment"], "run_id": j["run_id"], "state": j["state"],
            "pool": j.get("pool"), "gpu": j.get("cuda_visible_devices"),
            "node": j.get("node"), "driver": j.get("driver_version"),
            "torch": j.get("torch_version"), "commit": j.get("commit"),
            "started_at": j.get("started_at"), "completed_at": j.get("completed_at"),
            "exit_code": j.get("exit_code"), "occupancy_hours": hours,
        })
    total = sum(j["occupancy_hours"] for j in jobs)
    out = {
        "protocol_id": S.TRACK_ID,
        "definition": "sum over jobs of (completed-started) wall clock on both A100s",
        "jobs": jobs,
        "n_jobs": len(jobs),
        "total_occupancy_hours": total,
        "budget_hours": args.budget,
        "fraction_of_budget": total / args.budget if args.budget else None,
        "remaining_hours": args.budget - total,
        "official_test_loaded": False,
    }
    S.write_json(S.ANALYSIS / "gpu_ledger.json", out)
    print(json.dumps({k: out[k] for k in ("n_jobs", "total_occupancy_hours",
                                         "budget_hours", "remaining_hours")}, indent=1))
    for j in jobs:
        print(f"  {j['experiment']:<18} {j['state']:<10} {j['pool']:<10} "
              f"gpu={j['gpu']} {j['occupancy_hours'] * 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
