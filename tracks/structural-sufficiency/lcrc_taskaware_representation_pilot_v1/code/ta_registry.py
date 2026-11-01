"""Snapshot the round's remote run attempts into ``analysis/rr_run_registry.json``.

``rr`` keeps no local database, so the ledger is derived from ``rr jobs --json``
and from each reported run's own ``train.json``.  Superseded attempts are kept
with the reason they are not reported — the accounting is meant to be auditable,
not tidy.

Usage
-----
    rr jobs --json > /tmp/rr_jobs.json   # or let this script shell out with --live
    uv run python code/ta_registry.py --jobs /tmp/rr_jobs.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import ta_common as C  # noqa: I001

OUT = C.RUN_REGISTRY

REPORTED = {
    "lipo-r123-20261101-225118-03302b1c": "lipo_r123",
    "lipo-r124-20261101-225128-cba2a23a": "lipo_r124",
    "bbbp-cv-f0-20261101-230423-299af43a": "bbbp_cv_f0",
    "bbbp-cv-f1-20261101-230433-9abba0da": "bbbp_cv_f1",
    "bbbp-cv-f2-20261101-230711-b35cec86": "bbbp_cv_f2",
    "tox21-equal-task-20261101-231108-da2c7322": "tox21_equal_task",
    "hiv-sqrt-pos-20261101-231234-8865ea86": "hiv_sqrt_pos",
}

NOT_REPORTED = {
    "lipo-r123-20261101-224809-5386467d": "argparse: --dataset omitted; exited in 5 s, no artifact written",
    "lipo-r124-20261101-224825-5d020074": "argparse: --dataset omitted; exited in 4 s, no artifact written",
    "bbbp-cv-f0-20261101-225824-ae145e59": (
        "superseded: training completed but the post-training recheck aborted at the flat 1e-5 tolerance; "
        "the reloaded state differed by exactly one ROC-AUC discordant pair (2.30e-5) on this fold. "
        "Relaunched at de483098 after the audit fix."
    ),
    "bbbp-cv-f1-20261101-225834-7471d6c7": (
        "superseded: same recipe, ran at f49b5025; relaunched at de483098 so that all three folds share one commit"
    ),
}

PREFIXES = ("lipo-r", "bbbp-cv", "tox21-equal", "hiv-sqrt")
COMMITS = ("f49b5025", "de483098")


def _seconds(job: Mapping[str, Any]) -> float | None:
    start, end = job.get("started_at"), job.get("completed_at")
    if not start or not end:
        return None
    return (dt.datetime.fromisoformat(end) - dt.datetime.fromisoformat(start)).total_seconds()


def build(jobs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    runs = []
    for job in sorted(jobs, key=lambda j: (j.get("started_at") or "", j["run_id"])):
        if not (job["experiment"].startswith(PREFIXES) and job["commit"].startswith(COMMITS)):
            continue
        key = REPORTED.get(job["run_id"])
        wall = _seconds(job)
        entry: dict[str, Any] = {
            "experiment": job["experiment"],
            "run_id": job["run_id"],
            "state": job["state"],
            "exit_code": job.get("exit_code"),
            "git_commit": job["commit"],
            "pool": job["pool"],
            "node": job.get("node"),
            "gpu_names": job.get("gpu_names"),
            "driver_version": job.get("driver_version"),
            "torch_version": job.get("torch_version"),
            "started_at": job["started_at"],
            "completed_at": job.get("completed_at"),
            "wall_seconds": wall,
            "run_key": key,
            "reporting": "reported" if key else "not_reported",
            "note": None if key else NOT_REPORTED.get(job["run_id"], ""),
        }
        if key:
            train_json = C.run_dir(key) / "train.json"
            if train_json.exists():
                payload = C.read_json(train_json)
                entry["gpu_hours"] = round(float(payload["timing"]["train_wall_seconds"]) / 3600.0, 5)
                entry["epochs"] = payload["checkpoint"]["epochs_run"]
                entry["best_epoch"] = payload["checkpoint"]["best_epoch"]
        else:
            entry["gpu_hours_estimate"] = round((wall or 0.0) / 3600.0, 5)
        runs.append(entry)
    return {
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "generated_by": (
            "rr jobs --json on host res (process backend). wall_seconds is process wall time; "
            "gpu_hours for a reported run is train_wall_seconds from its train.json, i.e. forward+backward "
            "only, excluding structure-cache construction and validation passes."
        ),
        "host": "res",
        "node": "a100-2",
        "driver": "550.163.01",
        "runs": runs,
        "n_attempts": len(runs),
        "n_reported": sum(1 for r in runs if r["reporting"] == "reported"),
        "n_not_reported": sum(1 for r in runs if r["reporting"] == "not_reported"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="snapshot the remote run attempts")
    parser.add_argument("--jobs", default=None, help="path to `rr jobs --json` output")
    parser.add_argument("--live", action="store_true", help="call rr jobs --json itself")
    args = parser.parse_args(argv)
    if args.live or not args.jobs:
        raw = subprocess.run(["rr", "jobs", "--json"], capture_output=True, text=True, cwd=str(C.REPO))
        if raw.returncode != 0:
            raise SystemExit(f"rr jobs failed: {raw.stderr[-400:]}")
        jobs = json.loads(raw.stdout)["jobs"]
    else:
        jobs = json.loads(Path(args.jobs).read_text())["jobs"]
    payload = build(jobs)
    C.write_json(OUT, payload)
    C.log(f"wrote {C.track_rel(OUT)} ({payload['n_reported']} reported / {payload['n_attempts']} attempts)")
    for entry in payload["runs"]:
        print(
            f"{entry['experiment']:16s} {entry['state']:10s} {str(entry['run_key']):16s} "
            f"{(entry['wall_seconds'] or 0):8.1f}s {entry['reporting']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
