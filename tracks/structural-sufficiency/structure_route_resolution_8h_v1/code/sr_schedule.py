"""Small deterministic scheduler for the two-GPU ``res`` queue.

Formal runs are always launched through ``rr`` (never raw SSH/sbatch).  This
script only decides *when* a declared run starts, records the allocation, and
pulls completed results.  The scientific queue itself is frozen in
``configs/sprint.json`` / ``PROTOCOL.md``.
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

ROUND = Path(__file__).resolve().parent.parent
REPO = ROUND.parents[2]
RESULTS = ROUND / "results"
STATE = ROUND / "analysis" / "queue_state.json"
T0 = datetime(2026, 11, 23, 15, 23, 1, tzinfo=timezone.utc)

EF_G = {"E_RAW", "E_MOB", "F_RAW", "F_MOB", "E_MOB_BIG", "F_MOB_BIG",
        "G_FULL5", "G_EGO2_5", "G_EGO3_5"}

# name, arm, split, seed, family, pool_hint
SPECS: dict[str, list[dict[str, Any]]] = {
    "stage1": [
        {"name": "R2_REF_D0_S0", "arm": "R2_REF", "split": "D0", "seed": 0},
        {"name": "E_MOB_D0_S0", "arm": "E_MOB", "split": "D0", "seed": 0},
        {"name": "E_RAW_D0_S0", "arm": "E_RAW", "split": "D0", "seed": 0},
        {"name": "F_MOB_D0_S0", "arm": "F_MOB", "split": "D0", "seed": 0},
        {"name": "F_RAW_D0_S0", "arm": "F_RAW", "split": "D0", "seed": 0},
        {"name": "G_FULL5_D0_S0", "arm": "G_FULL5", "split": "D0", "seed": 0},
        {"name": "G_EGO2_5_D0_S0", "arm": "G_EGO2_5", "split": "D0", "seed": 0},
    ],
    "baseline": [
        {"name": "R2_REF_D0_S1", "arm": "R2_REF", "split": "D0", "seed": 1},
        {"name": "E_MOB_D0_S1", "arm": "E_MOB", "split": "D0", "seed": 1},
        {"name": "F_MOB_D0_S1", "arm": "F_MOB", "split": "D0", "seed": 1},
    ],
    "big": [
        {"name": "E_MOB_BIG_D0_S0", "arm": "E_MOB_BIG", "split": "D0", "seed": 0},
        {"name": "F_MOB_BIG_D0_S0", "arm": "F_MOB_BIG", "split": "D0", "seed": 0},
    ],
    "p_scope": [
        {"name": "G_EGO2_5_D0_S1", "arm": "G_EGO2_5", "split": "D0", "seed": 1},
        {"name": "G_FULL5_D0_S1", "arm": "G_FULL5", "split": "D0", "seed": 1},
    ],
    "p_basis": [
        {"name": "F_RAW_D0_S1", "arm": "F_RAW", "split": "D0", "seed": 1},
    ],
    "ego3": [
        {"name": "G_EGO3_5_D0_S0", "arm": "G_EGO3_5", "split": "D0", "seed": 0},
    ],
}

ORDER = ["stage1", "baseline", "big", "p_scope", "p_basis", "ego3"]


def now() -> datetime:
    return datetime.now(timezone.utc)


def utc_stamp() -> str:
    return now().strftime("%Y-%m-%dT%H:%M:%SZ")


def load_state() -> dict[str, Any]:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"runs": {}, "log": [], "gpu_hours": 0.0}


def save_state(state: dict[str, Any]) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n",
                   encoding="utf-8")
    tmp.replace(STATE)


def command(spec: dict[str, Any]) -> str:
    arm = str(spec["arm"])
    run = str(spec["name"])
    if arm == "R2_REF":
        return ("python tracks/structural-sufficiency/r2_sentence_replication_v1/"
                f"code/r2_train.py --split {spec['split']} --seed {spec['seed']} "
                f"--run-id {run} --out {result_path(spec)}")
    return ("python tracks/structural-sufficiency/structure_route_resolution_8h_v1/"
            f"code/sr_train.py --arm {arm} --split {spec['split']} "
            f"--seed {spec['seed']} --run-id {run}")


def result_path(spec: dict[str, Any]) -> str:
    return f"tracks/structural-sufficiency/structure_route_resolution_8h_v1/results/{spec['name']}"


def rr(*args: str, check: bool = False) -> subprocess.CompletedProcess:
    cp = subprocess.run(["rr", *args], cwd=REPO, text=True,
                        capture_output=True)
    if check and cp.returncode != 0:
        raise RuntimeError(f"rr {' '.join(args)} failed:\n{cp.stdout}\n{cp.stderr}")
    return cp


def status(name: str) -> dict[str, Any] | None:
    cp = rr("status", name, "--json")
    if cp.returncode != 0:
        return None
    try:
        return json.loads(cp.stdout)
    except json.JSONDecodeError:
        return None


def refresh_active(state: dict[str, Any]) -> bool:
    changed = False
    for name, row in state["runs"].items():
        if row["state"] not in ("launched", "submitted", "running"):
            continue
        info = status(name)
        if info is None:
            continue
        remote_state = str(info.get("state"))
        if remote_state in ("completed", "failed", "cancelled"):
            row["state"] = remote_state
            row["exit_code"] = info.get("exit_code")
            row["finished_utc"] = utc_stamp()
            row["remote_run_id"] = info.get("run_id")
            try:
                started = datetime.fromisoformat(info["started_at"])
                finished = datetime.fromisoformat(info["completed_at"])
                row["wall_seconds"] = (finished - started).total_seconds()
            except Exception:
                row["wall_seconds"] = None
            if remote_state == "completed":
                cp = rr("pull", name)
                row["pull_ok"] = cp.returncode == 0
                row["pulled_dir"] = _copy_pulled(info, name)
            changed = True
        elif remote_state != row["state"]:
            row["state"] = remote_state
            changed = True
    return changed


def _copy_pulled(info: dict[str, Any], name: str) -> str | None:
    run_id = str(info.get("run_id") or "")
    if not run_id:
        return None
    src = REPO / ".rr" / "pulled" / run_id / "result" / result_path({"name": name})
    dst = RESULTS / name
    if not src.is_dir():
        return None
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return str(dst)


def active_pools(state: dict[str, Any]) -> set[str]:
    out = set()
    for row in state["runs"].values():
        if row["state"] in ("launched", "submitted", "running"):
            out.add(str(row.get("pool")))
    return out


def launch(spec: dict[str, Any], pool: str, state: dict[str, Any],
           *, dry: bool) -> None:
    name = str(spec["name"])
    cmd = command(spec)
    if dry:
        print(f"[dry] {pool} {name}: {cmd}")
        return
    cp = rr("run", "res", name, "--pool", pool,
            "--result", result_path(spec), "--", *cmd.split())
    row = {"state": "launched", "pool": pool, "command": cmd,
           "started_utc": utc_stamp(), "exit_code": None,
           "launch_stdout": cp.stdout[-2000:], "launch_stderr": cp.stderr[-2000:]}
    row["attempt"] = int(state["runs"].get(name, {}).get("attempt", 0)) + 1
    state["runs"][name] = row
    state["log"].append({"utc": utc_stamp(), "event": "launch", "name": name,
                         "pool": pool, "returncode": cp.returncode})
    save_state(state)
    print(f"launched {name} on {pool} (rr rc={cp.returncode})")
    if cp.returncode != 0 and "exited immediately" not in (cp.stdout + cp.stderr):
        print(cp.stdout[-1000:])
        print(cp.stderr[-1000:])


def run_queue(stage_names: list[str], *, dry: bool = False,
              max_minutes: float = 300.0) -> int:
    state = load_state()
    pending: list[dict[str, Any]] = []
    for stage in stage_names:
        for spec in SPECS[stage]:
            row = state["runs"].get(spec["name"], {})
            if row.get("state") in ("completed",):
                continue
            if row.get("state") in ("launched", "submitted", "running"):
                continue
            pending.append(spec)
    if dry:
        for i, spec in enumerate(pending):
            launch(spec, "res-gpu%d" % (i % 2), state, dry=True)
        return 0
    stop = time.time() + max_minutes * 60
    while pending or any(r["state"] in ("launched", "submitted", "running")
                         for r in state["runs"].values()):
        if refresh_active(state):
            save_state(state)
        busy = active_pools(state)
        for pool in ("res-gpu0", "res-gpu1"):
            if pool in busy:
                continue
            if not pending:
                break
            # keep the two longest E/F runs apart; any order is acceptable
            spec = pending.pop(0)
            launch(spec, pool, state, dry=False)
            busy.add(pool)
        if time.time() > stop:
            print("scheduler wall guard reached; leaving launched jobs running")
            return 2
        if pending or any(r["state"] in ("launched", "submitted", "running")
                          for r in state["runs"].values()):
            time.sleep(15)
    return 0


def budget(state: dict[str, Any]) -> dict[str, Any]:
    total = 0.0
    for row in state["runs"].values():
        if row.get("wall_seconds"):
            total += float(row["wall_seconds"]) / 3600.0
    return {"gpu_hours_wall": total, "n_runs": len(state["runs"]),
            "t0": T0.isoformat(), "now": utc_stamp()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", action="append", default=None,
                        choices=ORDER + ["all"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-minutes", type=float, default=300)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--budget", action="store_true")
    args = parser.parse_args()
    state = load_state()
    if args.status:
        refresh_active(state)
        save_state(state)
        print(json.dumps({"runs": {k: {kk: vv for kk, vv in v.items()
                                      if kk in ("state", "pool", "exit_code",
                                                "wall_seconds")}
                                  for k, v in state["runs"].items()},
                          "budget": budget(state)}, indent=1))
        return 0
    if args.budget:
        print(json.dumps(budget(state), indent=1))
        return 0
    stages = args.stage or ["stage1"]
    if "all" in stages:
        stages = ORDER
    return run_queue(list(dict.fromkeys(stages)), dry=args.dry_run,
                     max_minutes=args.max_minutes)


if __name__ == "__main__":
    raise SystemExit(main())