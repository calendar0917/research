"""Local dispatcher: one formal `rr` job at a time per GPU pool.

`rr` on the `res` host uses a process backend, so it does not queue.  This
script keeps two independent queues (one per GPU pool), launches the next run
only after the previous one on that pool has finished, pulls the declared
result path, and stops a queue on the first engineering failure.

It is an execution helper only; it never edits a tracked file and never runs
remotely itself.

Usage
-----
    uv run python code/cl_dispatch.py --plan p1 --dry-run
    uv run python code/cl_dispatch.py --plan p1
    uv run python code/cl_dispatch.py --plan p1 --only p1_chem8_lipo
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Sequence

import cl_common as C

LOG = C.ANALYSIS / "dispatch.log"
_LOCK = threading.Lock()


def _log(line: str) -> None:
    with _LOCK:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        text = f"[{stamp}] {line}"
        print(text, flush=True)
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as handle:
            handle.write(text + "\n")


def _rr(args: Sequence[str], *, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(["rr", *args], cwd=str(C.REPO), capture_output=True, text=True, check=check)


def _status(experiment: str) -> dict[str, Any] | None:
    proc = _rr(["status", experiment, "--json"])
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


def _launch(spec: dict[str, Any], *, dry_run: bool) -> str:
    key = spec["key"]
    experiment = key.replace("_", "-")
    result = f"tracks/structural-sufficiency/lcrc_performance_longrun_v1/analysis/runs/{key}"
    cmd = [
        "run",
        "res",
        experiment,
        "--pool",
        spec["pool"],
        "--result",
        result,
        "--",
        "uv",
        "run",
        "python",
        "tracks/structural-sufficiency/lcrc_performance_longrun_v1/code/cl_train.py",
        "--dataset",
        spec["dataset"],
        "--split",
        spec.get("split", "published"),
        "--arm",
        spec["arm"],
        "--seed",
        str(spec.get("seed", 0)),
        "--device",
        "cuda",
        "--key",
        key,
    ]
    if spec.get("protocol", "default") != "default":
        cmd += ["--protocol", spec["protocol"]]
    if spec.get("scale", "base") != "base":
        cmd += ["--scale", spec["scale"]]
    if dry_run:
        _log(f"DRY-RUN {spec['pool']}: {' '.join(cmd)}")
        return experiment
    proc = _rr(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"rr run failed for {key}: {proc.stdout}\n{proc.stderr}")
    _log(f"launched {key} on {spec['pool']}: {proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ''}")
    return experiment


def _worker(pool: str, queue: list[dict[str, Any]], *, dry_run: bool, stop: threading.Event) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for spec in queue:
        if stop.is_set():
            results.append({**spec, "state": "skipped_after_failure"})
            continue
        try:
            experiment = _launch(spec, dry_run=dry_run)
        except Exception as exc:  # noqa: BLE001
            _log(f"LAUNCH-FAIL {spec['key']}: {exc}")
            results.append({**spec, "state": "launch_failed", "error": str(exc)})
            stop.set()
            break
        if dry_run:
            results.append({**spec, "state": "dry_run"})
            continue
        started = time.time()
        last_state = None
        while True:
            if stop.is_set() and time.time() - started > 10:
                _log(f"stop requested; cancelling {spec['key']}")
                _rr(["cancel", experiment])
                results.append({**spec, "state": "cancelled"})
                break
            status = _status(experiment)
            if status is None:
                time.sleep(10)
                continue
            state = status.get("state")
            if state != last_state:
                _log(f"{spec['key']} state={state} elapsed={time.time() - started:.0f}s")
                last_state = state
            if state in ("completed", "failed", "cancelled"):
                pull = _rr(["pull", experiment])
                _log(f"{spec['key']} pull rc={pull.returncode}")
                results.append(
                    {
                        **spec,
                        "experiment": experiment,
                        "run_id": status.get("run_id"),
                        "state": state,
                        "exit_code": status.get("exit_code"),
                        "elapsed_seconds": round(time.time() - started, 1),
                    }
                )
                if state != "completed" or status.get("exit_code") not in (0, None):
                    stop.set()
                break
            time.sleep(10)
        if stop.is_set():
            break
    return results


PLANS: dict[str, list[dict[str, Any]]] = {
    "p5b": [
        {"key": "p5b_combo_lipo_s1", "dataset": "Lipophilicity", "arm": "combo_chem_contrib", "seed": 1, "pool": "res-gpu0"},
        {"key": "p4b_combo_lipo123", "dataset": "Lipophilicity", "split": "lipo-repeat:123", "arm": "combo_chem_contrib", "pool": "res-gpu0"},
        {"key": "p5b_combo_lipo_s2", "dataset": "Lipophilicity", "arm": "combo_chem_contrib", "seed": 2, "pool": "res-gpu1"},
        {"key": "p4b_combo_lipo124", "dataset": "Lipophilicity", "split": "lipo-repeat:124", "arm": "combo_chem_contrib", "pool": "res-gpu1"},
    ],
    "p2b": [
        {"key": "p2b_combo_chem_contrib_lipo", "dataset": "Lipophilicity", "arm": "combo_chem_contrib", "pool": "res-gpu0"},
        {"key": "p2b_combo_chem_contrib_esol", "dataset": "ESOL", "arm": "combo_chem_contrib", "pool": "res-gpu1"},
        {"key": "p2b_combo_chem_contrib_bace", "dataset": "BACE", "arm": "combo_chem_contrib", "pool": "res-gpu1"},
    ],
    "p5": [
        {"key": "p5_base_lipo_s1", "dataset": "Lipophilicity", "arm": "base", "seed": 1, "pool": "res-gpu0"},
        {"key": "p5_rho_lipo_s1", "dataset": "Lipophilicity", "arm": "rhoshape", "seed": 1, "pool": "res-gpu0"},
        {"key": "p5_base_esol_s1", "dataset": "ESOL", "arm": "base", "seed": 1, "pool": "res-gpu0"},
        {"key": "p5_rho_esol_s1", "dataset": "ESOL", "arm": "rhoshape", "seed": 1, "pool": "res-gpu0"},
        {"key": "p5_base_freesolv_s1", "dataset": "FreeSolv", "arm": "base", "seed": 1, "pool": "res-gpu0"},
        {"key": "p5_rho_freesolv_s1", "dataset": "FreeSolv", "arm": "rhoshape", "seed": 1, "pool": "res-gpu0"},
        {"key": "p5_base_lipo_s2", "dataset": "Lipophilicity", "arm": "base", "seed": 2, "pool": "res-gpu1"},
        {"key": "p5_rho_lipo_s2", "dataset": "Lipophilicity", "arm": "rhoshape", "seed": 2, "pool": "res-gpu1"},
        {"key": "p5_base_esol_s2", "dataset": "ESOL", "arm": "base", "seed": 2, "pool": "res-gpu1"},
        {"key": "p5_rho_esol_s2", "dataset": "ESOL", "arm": "rhoshape", "seed": 2, "pool": "res-gpu1"},
        {"key": "p5_base_freesolv_s2", "dataset": "FreeSolv", "arm": "base", "seed": 2, "pool": "res-gpu1"},
        {"key": "p5_rho_freesolv_s2", "dataset": "FreeSolv", "arm": "rhoshape", "seed": 2, "pool": "res-gpu1"},
    ],
    "p3": [
        {"key": "p3_base_scalem_lipo", "dataset": "Lipophilicity", "arm": "base", "pool": "res-gpu0", "scale": "M"},
        {"key": "p3_rho_scalem_lipo", "dataset": "Lipophilicity", "arm": "rhoshape", "pool": "res-gpu1", "scale": "M"},
        {"key": "p3_base_scalem_esol", "dataset": "ESOL", "arm": "base", "pool": "res-gpu0", "scale": "M"},
        {"key": "p3_rho_scalem_esol", "dataset": "ESOL", "arm": "rhoshape", "pool": "res-gpu1", "scale": "M"},
    ],
    "p4": [
        {"key": "p4_rho_lipo123", "dataset": "Lipophilicity", "split": "lipo-repeat:123", "arm": "rhoshape", "pool": "res-gpu0"},
        {"key": "p4_rho_lipo124", "dataset": "Lipophilicity", "split": "lipo-repeat:124", "arm": "rhoshape", "pool": "res-gpu0"},
        {"key": "p4_rho_hiv", "dataset": "ogbg-molhiv", "arm": "rhoshape", "pool": "res-gpu0", "protocol": "hiv_short"},
        {"key": "p4_rho_freesolv", "dataset": "FreeSolv", "arm": "rhoshape", "pool": "res-gpu1"},
        {"key": "p4_rho_bbbp", "dataset": "BBBP", "arm": "rhoshape", "pool": "res-gpu1"},
        {"key": "p4_rho_tox21", "dataset": "Tox21", "arm": "rhoshape", "pool": "res-gpu1"},
        {"key": "p4_base_hiv", "dataset": "ogbg-molhiv", "arm": "base", "pool": "res-gpu1", "protocol": "hiv_short"},
    ],
    "p2": [
        {"key": "p2_combo_rho_contrib_lipo", "dataset": "Lipophilicity", "arm": "combo_rho_contrib", "pool": "res-gpu0"},
        {"key": "p2_combo_rho_contrib_esol", "dataset": "ESOL", "arm": "combo_rho_contrib", "pool": "res-gpu1"},
        {"key": "p2_combo_rho_contrib_bace", "dataset": "BACE", "arm": "combo_rho_contrib", "pool": "res-gpu1"},
    ],
    "p1": [
        {"key": "p1_chem8_lipo", "dataset": "Lipophilicity", "arm": "chem8", "pool": "res-gpu0"},
        {"key": "p1_rho_lipo", "dataset": "Lipophilicity", "arm": "rhoshape", "pool": "res-gpu0"},
        {"key": "p1_contrib_bace", "dataset": "BACE", "arm": "multicontrib", "pool": "res-gpu0"},
        {"key": "p1_contrib_lipo", "dataset": "Lipophilicity", "arm": "multicontrib", "pool": "res-gpu1"},
        {"key": "p1_chem8_esol", "dataset": "ESOL", "arm": "chem8", "pool": "res-gpu1"},
        {"key": "p1_rho_esol", "dataset": "ESOL", "arm": "rhoshape", "pool": "res-gpu1"},
        {"key": "p1_contrib_esol", "dataset": "ESOL", "arm": "multicontrib", "pool": "res-gpu1"},
        {"key": "p1_chem8_bace", "dataset": "BACE", "arm": "chem8", "pool": "res-gpu1"},
        {"key": "p1_rho_bace", "dataset": "BACE", "arm": "rhoshape", "pool": "res-gpu1"},
    ]
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="dispatch formal runs across the two res GPU pools")
    parser.add_argument("--plan", required=True, choices=sorted(PLANS))
    parser.add_argument("--only", default=None, help="run only this key (still uses its pool)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    plan = [dict(x) for x in PLANS[args.plan]]
    if args.only:
        plan = [x for x in plan if x["key"] == args.only]
        if not plan:
            raise SystemExit(f"{args.only} is not in plan {args.plan}")
    queues: dict[str, list[dict[str, Any]]] = {}
    for spec in plan:
        queues.setdefault(spec["pool"], []).append(spec)
    _log(f"plan={args.plan} runs={len(plan)} queues={ {k: [x['key'] for x in v] for k, v in queues.items()} }")
    stop = threading.Event()
    all_results: list[dict[str, Any]] = []
    threads = []
    for pool, queue in queues.items():
        thread = threading.Thread(target=lambda p=pool, q=queue: all_results.extend(_worker(p, q, dry_run=args.dry_run, stop=stop)))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    _log("SUMMARY " + json.dumps(all_results, sort_keys=True))
    failed = [r for r in all_results if r.get("state") not in ("completed", "dry_run")]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
