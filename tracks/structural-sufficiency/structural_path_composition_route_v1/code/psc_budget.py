"""Compute-budget ledger: GPU/CPU time actually consumed, per run and total."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import csv

import numpy as np

import psc_common as C
import psc_analysis as A


def pulled_meta(ident: str) -> dict[str, Any] | None:
    hits = sorted(Path(C.REPO_ROOT, ".rr", "pulled").glob(f"{ident}-*"))
    if not hits:
        return None
    # prefer the completed attempt with the largest runtime
    best = None
    for d in hits:
        m = d / "meta.json"
        if not m.is_file():
            continue
        data = json.loads(m.read_text(encoding="utf-8"))
        if best is None or (data.get("state") == "completed"
                            and best.get("state") != "completed"):
            best = data
        elif best.get("state") != "completed" and data.get("state") == "completed":
            best = data
    return best


def loop_seconds(rid: str) -> float | None:
    path = A.RESULTS / rid / "curves.csv"
    if not path.is_file():
        return None
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    vals = [float(r["epoch_seconds"]) for r in rows if r.get("epoch_seconds")]
    return float(sum(vals)) if vals else None


def cmd_budget() -> dict[str, Any]:
    runs = []
    for man in sorted(A.RESULTS.glob("*/manifest.json")):
        rid = man.parent.name
        data = json.loads(man.read_text(encoding="utf-8"))
        runs.append({"run_id": rid, "kind": "gpu", "wall_seconds": data.get("wall_seconds"),
                     "gpu_hours": data.get("gpu_hours"),
                     "train_loop_seconds": loop_seconds(rid),
                     "peak_vram_bytes": data.get("max_vram_bytes"),
                     "commit": data.get("provenance", {}).get("git_commit"),
                     "meta": pulled_meta(f"psc-{rid}")})
    for man in sorted(A.CPU_OUT.glob("*/manifest.json")):
        rid = man.parent.name
        data = json.loads(man.read_text(encoding="utf-8"))
        runs.append({"run_id": rid, "kind": "cpu", "wall_seconds": data.get("wall_seconds"),
                     "threads": data.get("threads"), "gpu_hours": 0.0,
                     "commit": data.get("provenance", {}).get("git_commit"), "meta": None})
    attempts = []
    for d in sorted(Path(C.REPO_ROOT, ".rr", "pulled").glob("psc-*")):
        m = d / "meta.json"
        if not m.is_file():
            continue
        data = json.loads(m.read_text(encoding="utf-8"))
        rt = data.get("runtime") or {}
        start, end = rt.get("started_at"), rt.get("completed_at")
        secs = None
        if start and end:
            secs = (np.datetime64(end) - np.datetime64(start)) / np.timedelta64(1, "s")
        attempts.append({"pull_dir": d.name, "ident": data.get("ident"),
                         "state": data.get("state"), "pool": data.get("pool"),
                         "started_at": start, "completed_at": end,
                         "seconds": None if secs is None else float(secs),
                         "commit": (data.get("provenance") or {}).get("git_commit")})
    # compute totals
    gpu_hours = 0.0
    for r in runs:
        if r["kind"] == "gpu" and r["wall_seconds"]:
            gpu_hours += float(r["wall_seconds"]) / 3600.0
    # include non-completed attempts (cancelled/failed smoke + duplicates) at
    # their measured wall time
    extra = defaultdict(float)
    for a in attempts:
        if a["state"] in ("completed",) or a["seconds"] is None:
            continue
        extra[a["state"]] += a["seconds"] / 3600.0
    total_gpu_hours = gpu_hours + sum(extra.values())
    cpu_hours = sum(float(r["wall_seconds"] or 0) for r in runs if r["kind"] == "cpu") / 3600.0
    train_loop = sum(float(r["train_loop_seconds"] or 0) for r in runs
                     if r["kind"] == "gpu") / 3600.0
    out = {
        "generated_utc": C.now_utc(), "git_commit": C.git_commit(),
        "definition": ("GPU-hours = sum of per-run wall-clock of every launched job "
                       "(including cancelled duplicates and smoke tests) x 1 GPU; "
                       "train_loop_seconds is the sum of per-epoch epoch_seconds from curves.csv."),
        "counts": {"gpu_runs_completed": sum(1 for r in runs if r["kind"] == "gpu"),
                   "cpu_runs_completed": sum(1 for r in runs if r["kind"] == "cpu"),
                   "launch_attempts": len(attempts),
                   "noncompleted_attempts": sum(1 for a in attempts
                                                if a["state"] != "completed")},
        "totals": {"gpu_hours_including_attempts": total_gpu_hours,
                   "gpu_hours_completed_runs": gpu_hours,
                   "gpu_hours_noncompleted": dict(extra),
                   "train_loop_hours": train_loop,
                   "train_loop_fraction": (train_loop / gpu_hours) if gpu_hours else None,
                   "cpu_hours_closed_form": cpu_hours,
                   "budget_gpu_hours": 16.0,
                   "within_budget": total_gpu_hours <= 16.0},
        "runs": [{k: v for k, v in r.items() if k != "meta"} | {
            "host_node": ((r["meta"] or {}).get("runtime") or {}).get("node"),
            "started_at": ((r["meta"] or {}).get("runtime") or {}).get("started_at"),
            "completed_at": ((r["meta"] or {}).get("runtime") or {}).get("completed_at"),
            "state": (r["meta"] or {}).get("state")} for r in runs],
        "attempts": attempts,
    }
    C.write_json(A.ANALYSIS / "compute_budget.json", out)
    print(json.dumps(out["totals"], indent=1))
    print(json.dumps(out["counts"], indent=1))
    return out


if __name__ == "__main__":
    cmd_budget()