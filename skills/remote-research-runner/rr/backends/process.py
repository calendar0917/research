"""Process backend — detached setsid/nohup (res and other non-Slurm hosts).

Reuses the proven keep-alive pattern: the job is a new session leader with
stdin detached, so an SSH drop cannot kill it.  Completion is recorded in
meta.json on the remote filesystem (authoritative), with the pid used only for
liveness checks.
"""

from __future__ import annotations

import json

from ..errors import JobError
from ..util import b64
from .base import Backend


class ProcessBackend(Backend):
    name = "process"

    def launch(self, *, pool, meta, cmd_str, run_dir):  # noqa: ANN001
        tools = self.ensure_remote_tools()
        env = self.base_runner_env(
            pool, tools, b64(json.dumps(meta, ensure_ascii=False)), b64(cmd_str)
        )
        env["RR_RUN_DIR"] = run_dir
        env["RR_PIDFILE"] = f"{run_dir}/job.pid"
        export_block = self.runner_export_block(env)

        script = f"""
set -euo pipefail
run_dir="$1"
mkdir -p "$run_dir"
{export_block}

setsid bash -c 'echo $$ > "$RR_PIDFILE"; exec bash "$RR_RUNNER"' \
  </dev/null >/dev/null 2>&1 &
disown 2>/dev/null || true

for _ in $(seq 1 100); do
  [ -s "$run_dir/job.pid" ] && break
  sleep 0.1
done
if [ ! -s "$run_dir/job.pid" ]; then
  echo "rr: launcher failed to start" >&2
  exit 4
fi
pid="$(cat "$run_dir/job.pid")"
# Give the runner a moment to publish meta.json / fail fast.
for _ in $(seq 1 200); do
  [ -s "$run_dir/meta.json" ] && break
  kill -0 "$pid" 2>/dev/null || break
  sleep 0.1
done
printf 'pid=%s\\n' "$pid"
"""
        cp = self.remote.run(script, [run_dir])
        pid = ""
        for line in cp.stdout.splitlines():
            if line.startswith("pid="):
                pid = line.split("=", 1)[1].strip()
        remote_meta = self.read_meta(run_dir)
        state = remote_meta.get("state")
        if pid and not self._pid_alive(pid):
            stderr = self.tail_logs(run_dir, "stderr", 30, meta)
            raise JobError(
                "process exited immediately after launch",
                code="rr.job.exited_immediately",
                host=self.host.name,
                run_id=meta.get("run_id"),
                exit_code=(remote_meta.get("runtime") or {}).get("exit_code"),
                state=state,
                stderr=stderr.strip()[-2000:],
            )
        return {"run_dir": run_dir, "pid": pid, "state": state or "running"}

    # -- state ------------------------------------------------------------
    def _pid_alive(self, pid: str) -> bool:
        return self.remote.ok('kill -0 "$1" 2>/dev/null', [pid])

    def _read_pid(self, run_dir: str) -> str:
        return self.remote.get_text(f"{run_dir}/job.pid").strip()

    # -- legacy (pre-rr) runs ---------------------------------------------
    def list_legacy_metas(self) -> list[dict]:
        root = self.legacy_root()
        if not root:
            return []
        script = r"""
set -uo pipefail
root="$1"
[ -d "$root" ] || exit 0
shopt -s nullglob
for meta in "$root"/*.meta; do
  tag="$(basename "$meta" .meta)"
  printf '###LEGACY###%s\n' "$tag"
  cat "$meta"
  if [ -f "$root/$tag.exit" ]; then printf 'state=completed exit=%s\n' "$(cat "$root/$tag.exit")";
  elif [ -f "$root/$tag.pid" ] && kill -0 "$(cat "$root/$tag.pid")" 2>/dev/null; then printf 'state=running pid=%s\n' "$(cat "$root/$tag.pid")";
  else printf 'state=unknown\n'; fi
  printf '###END###\n'
done
"""
        out = self.remote.out(script, [root])
        metas: list[dict] = []
        tag = None
        fields: dict = {}
        for line in out.splitlines():
            if line.startswith("###LEGACY###"):
                tag = line[len("###LEGACY###"):].strip()
                fields = {}
            elif line.startswith("###END###"):
                if tag:
                    metas.append(self._map_legacy(tag, fields, root))
                tag = None
            elif "=" in line:
                k, v = line.split("=", 1)
                fields[k.strip()] = v.strip()
        return metas

    def _map_legacy(self, tag: str, fields: dict, root: str) -> dict:
        state = fields.get("state", "unknown")
        exit_raw = fields.get("exit")
        exit_code = int(exit_raw) if exit_raw and exit_raw.lstrip("-").isdigit() else None
        return {
            "schema": "rr/legacy-meta/v1",
            "legacy": True,
            "legacy_tag": tag,
            "legacy_root": root,
            "run_id": tag,
            "experiment": tag,
            "host": self.host.name,
            "backend": self.host.backend,
            "pool": None,
            "state": state,
            "git_commit": fields.get("commit"),
            "command": fields.get("cmd"),
            "created_at": fields.get("started"),
            "slurm_job_id": None,
            "runtime": {"exit_code": exit_code, "node": self.host.env.get("legacy_node")},
        }

    def live_state(self, meta: dict) -> dict:
        from ..meta import is_terminal

        if meta.get("legacy"):
            return {"state": meta.get("state"), "live": meta.get("state") == "running"}

        run_dir = self.run_dir_of(meta)
        state = meta.get("state")
        if is_terminal(state):
            return {"state": state, "live": False}
        pid = self._read_pid(run_dir)
        alive = bool(pid) and self._pid_alive(pid)
        return {"state": state or "unknown", "live": alive, "pid": pid or None}

    def reconcile(self, meta: dict, live: dict) -> dict | None:  # noqa: ANN001
        """A dead PID with non-terminal metadata becomes an explicit failure.

        Otherwise `rr jobs` would report ``running live=false`` forever when the
        process was killed (OOM, reboot, `kill -9`) before the runner could
        record completion.
        """
        from ..meta import is_terminal
        from ..util import now_iso

        if meta.get("legacy") or is_terminal(meta.get("state")):
            return None
        if live.get("live"):
            return None
        run_dir = self.run_dir_of(meta)
        fresh = self.read_meta(run_dir) or dict(meta)
        if is_terminal(fresh.get("state")):
            return None
        updated = dict(fresh)
        updated["state"] = "lost"
        runtime = dict(fresh.get("runtime") or {})
        runtime["failure_reason"] = runtime.get("failure_reason") or "process_died_without_completion"
        runtime["completed_at"] = runtime.get("completed_at") or now_iso()
        updated["runtime"] = runtime
        self.write_meta_atomic(run_dir, updated)
        return updated

    def cancel(self, meta: dict) -> dict:
        from ..util import now_iso

        if meta.get("legacy"):
            raise JobError(
                "legacy run (pre-rr launches) is not managed by rr; use the old scripts"
                " (scripts/tail_remote.sh, scripts/status.sh) to inspect it",
                code="rr.job.legacy_not_managed",
                run_id=meta.get("run_id"),
            )
        run_dir = self.run_dir_of(meta)
        pid = self._read_pid(run_dir)
        if not pid or not self._pid_alive(pid):
            raise JobError(
                "run is not currently running; nothing to cancel",
                code="rr.job.not_running",
                run_id=meta.get("run_id"),
                state=meta.get("state"),
            )
        # setsid makes the child a process-group leader: signal the whole group.
        script = r"""
set -uo pipefail
pid="$1"; run_dir="$2"
kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
for _ in $(seq 1 30); do
  kill -0 "$pid" 2>/dev/null || break
  sleep 0.5
done
if kill -0 "$pid" 2>/dev/null; then
  kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
fi
printf 'cancelled pid=%s\n' "$pid"
"""
        self.remote.run(script, [pid, run_dir], check=False)
        meta = dict(meta)
        meta["state"] = "cancelled"
        runtime = dict(meta.get("runtime") or {})
        runtime["completed_at"] = now_iso()
        runtime["failure_reason"] = runtime.get("failure_reason") or "cancelled"
        runtime["exit_code"] = runtime.get("exit_code", 143)
        meta["runtime"] = runtime
        self.write_meta(run_dir, meta)
        return {"run_id": meta.get("run_id"), "state": "cancelled", "pid": pid}