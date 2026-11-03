"""Launch, poll and account for the fresh LCRC-P1 training units through ``rr``.

Every remote action goes through the ``rr`` CLI (see
``skills/remote-research-runner/SKILL.md``); this module never calls ``ssh`` or
``sbatch`` itself.  It adds three things on top of ``rr``:

* a deterministic wave plan (at most one unit per A100, two GPUs on ``res``);
* an append-only launch/ledger record under ``analysis/``;
* a GPU-hour projection check against the 2.0 A100-hour cap before any launch.

Usage::

    python .../code/dispatch.py plan
    python .../code/dispatch.py launch --wave 0
    python .../code/dispatch.py wait --wave 0
    python .../code/dispatch.py ledger
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

from p1_common import (
    ANALYSIS,
    REPO as C_REPO,
    FRESH_UNITS,
    ROUND,
    RUNS,
    SPLITS,
    fresh_unit_key,
    track_rel,
    utcnow,
    write_json,
)

HOST = "res"
POOLS = ("res-gpu0", "res-gpu1")
GPU_HOURS_CAP = 2.0
#: measured on the same host/regime for a 360-epoch Lipophilicity run
PROJECTED_HOURS_PER_RUN = 0.12
DISPATCH_LOG = ANALYSIS / "dispatch.log"
LAUNCHES = ANALYSIS / "rr_launches.json"
PLAN_FILE = ANALYSIS / "dispatch_plan.json"
UNIT_WRAPPER = ROUND / "code/run_unit.py"


def _log(message: str) -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    line = f"[{utcnow()}] {message}"
    print(line, flush=True)
    with DISPATCH_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def experiment_name(key: str) -> str:
    return key.replace("_", "-")


def pending_units() -> list[dict[str, Any]]:
    out = []
    for spec in FRESH_UNITS:
        key = fresh_unit_key(spec["split"], spec["arm"], spec["seed"])
        if (RUNS / key / "train.json").exists():
            continue
        out.append(
            {
                "unit_key": key,
                "split": spec["split"],
                "arm": spec["arm"],
                "seed": int(spec["seed"]),
                "longrun_split": SPLITS[spec["split"]]["longrun_split"],
            }
        )
    return out


def build_command(unit: dict[str, Any]) -> str:
    return (
        f"uv run python {track_rel(UNIT_WRAPPER)}"
        f" --split {unit['split']} --arm {unit['arm']} --seed {unit['seed']} --device cuda"
    )


def build_waves(units: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    waves = []
    for i in range(0, len(units), len(POOLS)):
        chunk = list(units[i : i + len(POOLS)])
        for j, unit in enumerate(chunk):
            unit = dict(unit)
            unit["pool"] = POOLS[j]
            chunk[j] = unit
        waves.append(chunk)
    return waves


def cmd_plan(args: argparse.Namespace) -> int:
    units = pending_units()
    waves = build_waves(units)
    plan = {
        "created_at": utcnow(),
        "host": HOST,
        "pools": list(POOLS),
        "pending_units": units,
        "waves": [
            [
                {
                    **unit,
                    "command": f"rr run {HOST} {experiment_name(unit['unit_key'])} --pool {unit['pool']} "
                    f"--result {track_rel(RUNS / unit['unit_key'])} -- {build_command(unit)}",
                }
                for unit in wave
            ]
            for wave in waves
        ],
        "projected_gpu_hours": round(len(units) * PROJECTED_HOURS_PER_RUN, 4),
        "gpu_hours_cap": GPU_HOURS_CAP,
        "budget_ok": bool(len(units) * PROJECTED_HOURS_PER_RUN <= GPU_HOURS_CAP),
        "projection_basis": (
            "measured wall time of the ten frozen Lipophilicity runs on res/a100-2 "
            "(mean 0.097 A100-h, max 0.117 A100-h) rounded up to 0.12 A100-h per run"
        ),
    }
    write_json(PLAN_FILE, plan)
    _log(f"plan: {len(units)} pending units, {len(waves)} waves, projected {plan['projected_gpu_hours']} A100-h")
    for wave_index, wave in enumerate(plan["waves"]):
        for unit in wave:
            _log(f"  wave {wave_index} {unit['pool']}: {unit['command']}")
    if not plan["budget_ok"]:
        _log("BUDGET REFUSED: projected GPU hours exceed the cap")
        return 2
    return 0


def _read_launches() -> dict[str, Any]:
    if LAUNCHES.exists():
        return json.loads(LAUNCHES.read_text(encoding="utf-8"))
    return {"schema": "lcrc-p1-rr-launches/v1", "launches": []}


def _write_launches(payload: dict[str, Any]) -> None:
    write_json(LAUNCHES, payload)


def cmd_launch(args: argparse.Namespace) -> int:
    plan = json.loads(PLAN_FILE.read_text(encoding="utf-8"))
    wave = plan["waves"][int(args.wave)]
    payload = _read_launches()
    for unit in wave:
        exp = experiment_name(unit["unit_key"])
        cmd = [
            "rr",
            "--json",
            "run",
            HOST,
            exp,
            "--pool",
            unit["pool"],
            "--result",
            track_rel(RUNS / unit["unit_key"]),
            "--",
            *shlex.split(build_command(unit)),
        ]
        _log(f"launch wave {args.wave} {unit['pool']}: {' '.join(shlex.quote(c) for c in cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            _log(f"LAUNCH-FAIL {exp}: {proc.stdout.strip()} {proc.stderr.strip()}")
            payload["launches"].append(
                {
                    "unit_key": unit["unit_key"],
                    "experiment": exp,
                    "pool": unit["pool"],
                    "state": "launch_failed",
                    "stdout": proc.stdout[-4000:],
                    "stderr": proc.stderr[-4000:],
                    "at": utcnow(),
                }
            )
            _write_launches(payload)
            return 1
        info: Any
        try:
            info = json.loads(proc.stdout)
        except json.JSONDecodeError:
            info = {"raw": proc.stdout.strip()}
        run_id = info.get("run_id") if isinstance(info, dict) else None
        payload["launches"].append(
            {
                "unit_key": unit["unit_key"],
                "experiment": exp,
                "pool": unit["pool"],
                "run_id": run_id,
                "state": "launched",
                "launch_payload": info,
                "at": utcnow(),
            }
        )
        _write_launches(payload)
        _log(f"launched {exp} run_id={run_id}")
    return 0


def _status(exp: str) -> dict[str, Any]:
    proc = subprocess.run(["rr", "--json", "status", exp], capture_output=True, text=True)
    if proc.returncode != 0:
        return {"ok": False, "error": (proc.stdout + proc.stderr).strip()[-2000:], "experiment": exp}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:  # pragma: no cover
        return {"ok": False, "error": proc.stdout[-2000:], "experiment": exp}


def cmd_wait(args: argparse.Namespace) -> int:
    plan = json.loads(PLAN_FILE.read_text(encoding="utf-8"))
    wave = plan["waves"][int(args.wave)]
    names = [experiment_name(unit["unit_key"]) for unit in wave]
    deadline = time.time() + float(args.timeout)
    terminal = {"completed", "failed", "cancelled", "canceled"}
    last: dict[str, str] = {}
    while time.time() < deadline:
        states = {}
        for name in names:
            status = _status(name)
            states[name] = str(status.get("state", "unknown"))
            if last.get(name) != states[name]:
                _log(f"{name} state={states[name]}")
                last[name] = states[name]
        if all(state in terminal for state in states.values()):
            _log(f"wave {args.wave} terminal: {json.dumps(states)}")
            return 0 if all(state == "completed" for state in states.values()) else 1
        time.sleep(float(args.interval))
    _log(f"wave {args.wave} TIMEOUT after {args.timeout}s: {json.dumps(last)}")
    return 3


def cmd_pull(args: argparse.Namespace) -> int:
    plan = json.loads(PLAN_FILE.read_text(encoding="utf-8"))
    wave = plan["waves"][int(args.wave)]
    rc = 0
    for unit in wave:
        exp = experiment_name(unit["unit_key"])
        _log(f"pull {exp}")
        proc = subprocess.run(["rr", "pull", exp], capture_output=True, text=True)
        if proc.returncode != 0:
            _log(f"PULL-FAIL {exp}: {proc.stdout.strip()} {proc.stderr.strip()}")
            rc = 1
    return rc


def cmd_sync(args: argparse.Namespace) -> int:
    """Copy pulled run artefacts from ``.rr/pulled`` into the round's runs tree.

    ``rr pull`` always unpacks a run bundle (meta/logs + ``result/<declared
    path>``) into ``.rr/pulled/<run_id>/``; this step moves the declared result
    subtree to ``analysis/runs/<unit_key>`` so the analysis sees real unit
    directories.
    """
    import shutil

    payload = _read_launches()
    targets = []
    seen: set[str] = set()
    for launch in payload["launches"]:
        if launch.get("superseded") or launch.get("state") == "launch_failed":
            continue
        if launch["experiment"] in seen:
            continue
        seen.add(launch["experiment"])
        if args.wave is not None:
            plan = json.loads(PLAN_FILE.read_text(encoding="utf-8"))
            wave_names = {experiment_name(u["unit_key"]) for u in plan["waves"][int(args.wave)]}
            if launch["experiment"] not in wave_names:
                continue
        targets.append(launch)
    rc = 0
    for launch in targets:
        exp = launch["experiment"]
        status = _status(exp)
        run_id = status.get("run_id") or launch.get("run_id")
        if not run_id:
            _log(f"SYNC-SKIP {exp}: no run id")
            rc = 1
            continue
        pulled = C_REPO / ".rr/pulled" / run_id
        if not pulled.exists():
            _log(f"sync {exp}: pulling {run_id}")
            proc = subprocess.run(["rr", "pull", exp], capture_output=True, text=True)
            if proc.returncode != 0:
                _log(f"PULL-FAIL {exp}: {proc.stdout.strip()} {proc.stderr.strip()}")
                rc = 1
                continue
        src = pulled / "result" / track_rel(RUNS / launch["unit_key"])
        if not src.is_dir():
            _log(f"SYNC-SKIP {exp}: {track_rel(src)} not present after pull")
            rc = 1
            continue
        dest = RUNS / launch["unit_key"]
        dest.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            if item.is_file():
                shutil.copy2(item, dest / item.name)
        _log(f"synced {exp} -> {track_rel(dest)} ({len(list(dest.iterdir()))} files)")
    return rc


def cmd_ledger(args: argparse.Namespace) -> int:
    payload = _read_launches()
    entries = []
    total_hours = 0.0
    for launch in payload["launches"]:
        exp = launch["experiment"]
        status = _status(exp)
        started = status.get("started_at")
        completed = status.get("completed_at")
        seconds = None
        if started and completed:
            seconds = _iso_seconds(started, completed)
        hours = (seconds / 3600.0) if seconds else None
        if hours:
            total_hours += hours
        train_json = RUNS / launch["unit_key"] / "train.json"
        measured = None
        if train_json.exists():
            measured = json.loads(train_json.read_text(encoding="utf-8"))["timing"]["process_wall_seconds"]
        entries.append(
            {
                "superseded": bool(launch.get("superseded")),
                "note": launch.get("note"),
                "unit_key": launch["unit_key"],
                "experiment": exp,
                "pool": launch.get("pool"),
                "run_id": status.get("run_id") or launch.get("run_id"),
                "state": status.get("state"),
                "exit_code": status.get("exit_code"),
                "commit": status.get("commit"),
                "hostname": (status.get("meta", {}).get("runtime", {}) or {}).get("hostname")
                or status.get("node"),
                "node": status.get("node"),
                "driver_version": status.get("driver_version"),
                "torch_version": status.get("torch_version"),
                "torch_cuda": status.get("torch_cuda"),
                "python_version": (status.get("meta", {}).get("runtime", {}) or {}).get("python_version"),
                "gpu_names": status.get("gpu_names"),
                "started_at": started,
                "completed_at": completed,
                "job_wall_seconds": seconds,
                "job_wall_gpu_hours": hours,
                "trainer_measured_wall_seconds": measured,
                "trainer_measured_gpu_hours": (measured / 3600.0) if measured else None,
                "result_path": track_rel(RUNS / launch["unit_key"]),
            }
        )
    ledger = {
        "schema": "lcrc-p1-gpu-ledger/v1",
        "created_at": utcnow(),
        "host": HOST,
        "pools": list(POOLS),
        "gpu_hours_cap": GPU_HOURS_CAP,
        "charge_unit": "A100 GPU-hours = job wall time on a single pinned A100 (one unit per GPU)",
        "entries": entries,
        "totals": {
            "n_units": len(entries),
            "n_completed": int(sum(1 for e in entries if e["state"] == "completed" and not e["superseded"])),
            "n_superseded": int(sum(1 for e in entries if e["superseded"])),
            "job_wall_gpu_hours": round(total_hours, 4),
            "charged_gpu_hours_including_superseded": round(total_hours, 4),
            "trainer_measured_gpu_hours": round(
                sum(e["trainer_measured_gpu_hours"] or 0.0 for e in entries), 4
            ),
        },
        "budget": {
            "cap": GPU_HOURS_CAP,
            "used_job_wall_gpu_hours": round(total_hours, 4),
            "remaining_gpu_hours": round(GPU_HOURS_CAP - total_hours, 4),
            "exceeded": bool(total_hours > GPU_HOURS_CAP),
        },
        "reused_units": {
            "note": (
                "The ten reused units were trained in lcrc-performance-longrun-v1 on the same host, "
                "driver and torch; their GPU time is charged to that round, not to this cap."
            )
        },
    }
    write_json(ANALYSIS / "gpu_ledger.json", ledger)
    _log(json.dumps(ledger["totals"]) + " " + json.dumps(ledger["budget"]))
    return 0


def _iso_seconds(start: str, end: str) -> float:
    from datetime import datetime

    def parse(value: str) -> datetime:
        return datetime.fromisoformat(value)

    return float((parse(end) - parse(start)).total_seconds())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LCRC-P1 rr dispatcher")
    sub = parser.add_subparsers(dest="command", required=True)
    p_plan = sub.add_parser("plan")
    p_plan.set_defaults(func=cmd_plan)
    p_launch = sub.add_parser("launch")
    p_launch.add_argument("--wave", type=int, required=True)
    p_launch.set_defaults(func=cmd_launch)
    p_wait = sub.add_parser("wait")
    p_wait.add_argument("--wave", type=int, required=True)
    p_wait.add_argument("--timeout", type=float, default=1800.0)
    p_wait.add_argument("--interval", type=float, default=15.0)
    p_wait.set_defaults(func=cmd_wait)
    p_pull = sub.add_parser("pull")
    p_pull.add_argument("--wave", type=int, required=True)
    p_pull.set_defaults(func=cmd_pull)
    p_sync = sub.add_parser("sync")
    p_sync.add_argument("--wave", type=int, default=None)
    p_sync.set_defaults(func=cmd_sync)
    p_ledger = sub.add_parser("ledger")
    p_ledger.set_defaults(func=cmd_ledger)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
