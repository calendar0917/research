"""Capture the `rr` execution provenance for this round into a committed JSON.

The capture is a point-in-time record of the harness state (commit, pool, GPU
visibility, driver/torch, timings); it is not used to compute any metric.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

EXPERIMENTS = {
    "prep": "ep-prep-caches",
    "prep_branch": "ep-prep-branch",
    "preflight": "ep-preflight",
    "smoke": "ep-smoke",
    "A0": "ep-a0",
    "A1": "ep-a1",
    "A2": "ep-a2",
    "A3": "ep-a3",
    "COMBO": "ep-combo",
}

FIELDS = (
    "run_id",
    "state",
    "pool",
    "node",
    "gpu_count",
    "cuda_visible_devices",
    "driver_version",
    "torch_version",
    "torch_cuda",
    "commit",
    "branch",
    "dirty",
    "created_at",
    "started_at",
    "completed_at",
    "exit_code",
    "failure_reason",
)


def capture(experiments: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {"schema": "lcrc-pcba-explicit-performance-rr-status-v1", "runs": {}}
    for arm, experiment in experiments.items():
        proc = subprocess.run(
            ["rr", "--json", "status", experiment], capture_output=True, text=True
        )
        if proc.returncode != 0:
            out["runs"][arm] = {
                "experiment": experiment,
                "state": "not_found",
                "detail": (proc.stderr or proc.stdout).strip()[-400:],
            }
            continue
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError:
            out["runs"][arm] = {"experiment": experiment, "state": "unparsed"}
            continue
        entry = {"experiment": experiment}
        entry.update({k: payload.get(k) for k in FIELDS})
        meta = payload.get("meta") or {}
        runtime = meta.get("runtime") or {}
        entry["git_commit"] = meta.get("git_commit")
        entry["command"] = meta.get("command")
        entry["result_paths"] = meta.get("result_paths")
        entry["python_version"] = runtime.get("python_version")
        entry["requested"] = meta.get("requested")
        out["runs"][arm] = entry
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    payload = capture(EXPERIMENTS)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"captured {len(payload['runs'])} run statuses -> {out}")
    for arm, entry in payload["runs"].items():
        print(f"  {arm:10s} {str(entry.get('state')):18s} {entry.get('run_id') or entry.get('experiment')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
