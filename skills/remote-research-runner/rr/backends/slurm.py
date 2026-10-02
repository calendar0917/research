"""Slurm backend — sbatch / squeue / sacct / scancel (res-2).

Jobs are always submitted through Slurm.  rr never SSHes to a compute node to
run a formal task, and never re-implements the scheduler: concurrency and
queueing are Slurm's job, rr only provides observability.
"""

from __future__ import annotations

import json
import re
import shlex

from ..errors import JobError
from ..util import SLURM_TERMINAL, b64, normalize_slurm_state, now_iso
from .base import Backend

_STATE_MAP = {
    "PENDING": "pending",
    "CONFIGURING": "pending",
    "SUSPENDED": "pending",
    "STOPPED": "pending",
    "REQUEUED": "pending",
    "RUNNING": "running",
    "COMPLETING": "running",
    "COMPLETED": "completed",
    "FAILED": "failed",
    "CANCELLED": "cancelled",
    "TIMEOUT": "timeout",
    "OUT_OF_MEMORY": "oom",
    "NODE_FAIL": "failed",
    "PREEMPTED": "failed",
    "BOOT_FAIL": "failed",
    "DEADLINE": "timeout",
    "REVOKED": "failed",
}

# Scheduler states that are a trustworthy terminal fact about a job.
_TERMINAL_SCHEDULER = SLURM_TERMINAL


def _parse_exit(raw: str | None) -> int | None:
    """Parse a sacct ExitCode ('0:0', '137:0') into an int, else None."""
    if raw is None:
        return None
    first = str(raw).split(":")[0].strip()
    return int(first) if re.fullmatch(r"-?\d+", first) else None


def _reason_for(scheduler: str) -> str:
    return {
        "OUT_OF_MEMORY": "out_of_memory",
        "TIMEOUT": "timeout",
        "NODE_FAIL": "node_fail",
        "PREEMPTED": "preempted",
        "BOOT_FAIL": "boot_fail",
        "DEADLINE": "deadline",
        "REVOKED": "revoked",
        "FAILED": "failed",
        "CANCELLED": "cancelled",
    }.get(scheduler, "failed")


class SlurmBackend(Backend):
    name = "slurm"

    def _login_line(self) -> str:
        setup = self.host.slurm.get("login_setup", "")
        if not setup:
            return ""
        return f"eval {shlex.quote(setup)} >/dev/null 2>&1 || true"

    def _prelude(self) -> str:
        line = self._login_line()
        return line + "\n" if line else ""

    # -- launch -----------------------------------------------------------
    def launch(self, *, pool, meta, cmd_str, run_dir):  # noqa: ANN001
        tools = self.ensure_remote_tools()
        env = self.base_runner_env(
            pool, tools, b64(json.dumps(meta, ensure_ascii=False)), b64(cmd_str)
        )
        env["RR_RUN_DIR"] = run_dir
        if pool is not None and pool.min_driver:
            env["RR_MIN_DRIVER"] = pool.min_driver
        export_block = self.runner_export_block(env)

        self.write_meta(run_dir, meta)
        script_body = self._sbatch_script(meta, run_dir, export_block, pool)
        self.remote.put_text(f"{run_dir}/job.slurm", script_body)

        submit = self._prelude() + """
set -euo pipefail
run_dir="$1"
cd "$run_dir"
sbatch --parsable job.slurm
"""
        cp = self.remote.run(submit, [run_dir])

        job_id = ""
        for line in cp.stdout.splitlines():
            if line.strip():
                job_id = line.strip().split(";")[0].strip()
                break
        if not job_id or not job_id.isdigit():
            failed = dict(meta)
            failed["state"] = "failed"
            failed["runtime"] = {"failure_reason": "submit_failed", "exit_code": None}
            try:
                self.write_meta(run_dir, failed)
            except Exception:  # noqa: BLE001 - best effort annotation
                pass
            raise JobError(
                "sbatch did not return a job id",
                code="rr.job.submit_failed",
                host=self.host.name,
                pool=(pool.name if pool else None),
                output=cp.stdout.strip()[-1000:],
                stderr=cp.stderr.strip()[-2000:],
            )

        meta = dict(meta)
        meta["slurm_job_id"] = job_id
        meta["state"] = "submitted"
        self.write_meta(run_dir, meta)
        self.remote.put_text(f"{run_dir}/jobid", job_id + "\n")
        return {"run_dir": run_dir, "slurm_job_id": job_id, "state": "submitted"}

    def _sbatch_script(self, meta, run_dir, export_block, pool=None) -> str:  # noqa: ANN001
        host = self.host
        requested = meta.get("requested") or {}
        defaults = host.slurm
        partition = requested.get("partition") or defaults.get("partition", "gpu")
        cpus = requested.get("cpus", defaults.get("cpus", 4))
        mem = requested.get("mem", defaults.get("mem", "32G"))
        gpus = requested.get("gpus", 1)
        time_limit = requested.get("time", defaults.get("time", "24:00:00"))
        # The node constraint always comes from the pool (single source of
        # truth); requested may override it explicitly.
        nodes = requested.get("eligible_nodes") or (list(pool.eligible_nodes) if pool else [])
        nodelist = ",".join(nodes) if nodes else None
        job_name = f"rr-{meta.get('experiment', 'run')}"

        lines = [
            "#!/usr/bin/env bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --partition={partition}",
            # NOTE: do not emit --nodes here. This cluster's Slurm (21.08) rejects
            # `--nodes=1` together with a multi-node `--nodelist` (it computes an
            # invalid `-N 2-1`). --ntasks=1 already implies a single node.
            "#SBATCH --ntasks=1",
            f"#SBATCH --cpus-per-task={cpus}",
            f"#SBATCH --mem={mem}",
        ]
        if gpus and int(gpus) > 0:
            lines.append(f"#SBATCH --gres=gpu:{gpus}")
        lines += [
            f"#SBATCH --time={time_limit}",
            f"#SBATCH --output={run_dir}/slurm-%j.out",
            f"#SBATCH --error={run_dir}/slurm-%j.err",
        ]
        if nodelist:
            lines.append(f"#SBATCH --nodelist={nodelist}")
        lines += [
            "",
            "set -uo pipefail",
            export_block,
            'exec bash "$RR_RUNNER"',
            "",
        ]
        return "\n".join(lines)

    # -- state ------------------------------------------------------------
    def _job_id(self, meta: dict, run_dir: str) -> str:
        job_id = meta.get("slurm_job_id")
        if job_id:
            return str(job_id)
        return self.remote.get_text(f"{run_dir}/jobid").strip()

    def _query(self, job_id: str) -> dict:
        script = self._prelude() + """
set -uo pipefail
job="$1"
q="$(squeue -j "$job" -h -o '%T|%R|%M|%N' 2>/dev/null | head -1)"
if [ -n "$q" ]; then printf 'QUEUE|%s\\n' "$q"; exit 0; fi
a="$(sacct -j "$job" -X -n -P -o State,ExitCode,Elapsed,NodeList 2>/dev/null | head -1)"
if [ -n "$a" ]; then printf 'ACCT|%s\\n' "$a"; exit 0; fi
printf 'NONE|\\n'
"""
        out = self.remote.out(script, [job_id]).strip()
        kind, _, payload = out.partition("|")
        result: dict = {"kind": kind, "raw": payload.strip()}
        parts = payload.split("|")
        if kind == "QUEUE":
            result["scheduler_state"] = parts[0].strip() if parts else ""
            result["reason"] = parts[1].strip() if len(parts) > 1 else ""
            result["elapsed"] = parts[2].strip() if len(parts) > 2 else ""
            result["nodes"] = parts[3].strip() if len(parts) > 3 else ""
        elif kind == "ACCT":
            result["scheduler_state"] = parts[0].strip() if parts else ""
            result["exit_code"] = parts[1].strip() if len(parts) > 1 else ""
            result["elapsed"] = parts[2].strip() if len(parts) > 2 else ""
            result["nodes"] = parts[3].strip() if len(parts) > 3 else ""
        return result

    def live_state(self, meta: dict) -> dict:
        from ..meta import is_terminal

        run_dir = self.run_dir_of(meta)
        if is_terminal(meta.get("state")):
            return {"state": meta.get("state"), "live": False}
        job_id = self._job_id(meta, run_dir)
        if not job_id:
            return {"state": meta.get("state") or "unknown", "live": False}
        query = self._query(job_id)
        raw_scheduler = query.get("scheduler_state", "")
        scheduler = normalize_slurm_state(raw_scheduler)
        mapped = _STATE_MAP.get(scheduler, meta.get("state") or "unknown")
        live = scheduler in {"RUNNING", "COMPLETING", "CONFIGURING"}
        return {
            "state": mapped,
            "live": live,
            "job_id": job_id,
            "scheduler_state": scheduler or None,
            "scheduler_state_raw": raw_scheduler or None,
            "reason": query.get("reason"),
            "elapsed": query.get("elapsed"),
            "node": query.get("nodes") or (meta.get("runtime") or {}).get("node"),
            "acct_exit": query.get("exit_code"),
            "query_kind": query.get("kind"),
        }

    def reconcile(self, meta: dict, live: dict) -> dict | None:  # noqa: ANN001
        """Write a trustworthy terminal scheduler state back into meta.json.

        Covers OOM, TIMEOUT, NODE_FAIL, PREEMPTED, admin/user scancel and any
        death of the runner before it could record completion.  Never rewrites
        an already-terminal run and never fabricates a success exit code.
        """
        from ..meta import is_terminal

        if is_terminal(meta.get("state")):
            return None
        scheduler = normalize_slurm_state(live.get("scheduler_state"))
        if scheduler not in _TERMINAL_SCHEDULER:
            return None
        run_dir = self.run_dir_of(meta)
        fresh = self.read_meta(run_dir) or dict(meta)
        if is_terminal(fresh.get("state")):
            return None

        state = _STATE_MAP.get(scheduler, "failed")
        runtime = dict(fresh.get("runtime") or {})
        runtime["scheduler_state"] = scheduler
        runtime["slurm_exit_code"] = live.get("acct_exit")
        if live.get("node"):
            runtime["node"] = live["node"]
        runtime["completed_at"] = runtime.get("completed_at") or now_iso()
        code = _parse_exit(live.get("acct_exit"))
        if scheduler == "CANCELLED":
            runtime["failure_reason"] = runtime.get("failure_reason") or "cancelled"
            # A cancelled job has no success code: keep None unless sacct has a
            # real nonzero code to report.
            runtime["exit_code"] = code if code not in (None, 0) else None
        elif state == "completed":
            runtime["exit_code"] = 0
            runtime.pop("failure_reason", None)
        else:
            runtime["exit_code"] = code if code not in (None, 0) else 1
            runtime["failure_reason"] = runtime.get("failure_reason") or _reason_for(scheduler)

        updated = dict(fresh)
        updated["state"] = state
        updated["runtime"] = runtime
        self.write_meta_atomic(run_dir, updated)
        return updated

    def cancel(self, meta: dict) -> dict:
        run_dir = self.run_dir_of(meta)
        job_id = self._job_id(meta, run_dir)
        if not job_id:
            raise JobError(
                "no Slurm job id recorded for this run",
                code="rr.job.no_job_id",
                run_id=meta.get("run_id"),
            )
        script = self._prelude() + """
set -uo pipefail
job="$1"
# Only ever cancel a job owned by the current user.
if ! squeue -u "$USER" -h -o '%i' 2>/dev/null | grep -qx "$job"; then
  printf 'not-owner-or-not-queued\\n'
  exit 3
fi
scancel "$job"
printf 'cancelled %s\\n' "$job"
"""
        cp = self.remote.run(script, [job_id], check=False)
        if cp.returncode != 0:
            raise JobError(
                "refusing to cancel: job is not queued for the current user",
                code="rr.job.cancel_refused",
                run_id=meta.get("run_id"),
                job_id=job_id,
                output=cp.stdout.strip(),
            )
        meta = dict(meta)
        meta["state"] = "cancelled"
        runtime = dict(meta.get("runtime") or {})
        runtime["completed_at"] = now_iso()
        # A cancelled job was not a success: never fabricate exit_code = 0.
        runtime["failure_reason"] = runtime.get("failure_reason") or "cancelled"
        runtime["cancelled_by"] = "rr"
        meta["runtime"] = runtime
        self.write_meta(run_dir, meta)
        return {"run_id": meta.get("run_id"), "state": "cancelled", "job_id": job_id}

    # -- scheduler view ---------------------------------------------------
    def squeue(self) -> list[dict]:
        script = self._prelude() + """
set -uo pipefail
squeue -u "$USER" -h -o '%i|%j|%T|%R|%M|%N|%b' 2>/dev/null || true
"""
        out = self.remote.out(script)
        jobs = []
        for line in out.splitlines():
            parts = line.split("|")
            if len(parts) < 7:
                continue
            jobs.append(
                {
                    "job_id": parts[0].strip(),
                    "name": parts[1].strip(),
                    "scheduler_state": parts[2].strip(),
                    "reason": parts[3].strip(),
                    "elapsed": parts[4].strip(),
                    "nodes": parts[5].strip(),
                    "tres": parts[6].strip(),
                }
            )
        return jobs