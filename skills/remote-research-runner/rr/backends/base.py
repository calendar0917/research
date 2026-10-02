"""Backend abstraction.

A backend knows *how* a job is launched/monitored on one host:

  process backend  – detached setsid/nohup process (res)
  slurm backend    – sbatch/squeue/sacct/scancel (res-2)

Everything else (host/pool/regime resolution, deploy, provenance schema,
metadata layout, JSON output) is shared and lives outside the backends.
"""

from __future__ import annotations

import posixpath
from pathlib import Path

from ..config import Host, Pool
from ..errors import JobError
from ..remote import Remote
from ..util import b64, sha256_text, shell_export_lines

RUNNER_DIR = Path(__file__).resolve().parent.parent / "runner"


class Backend:
    name = "base"

    def __init__(self, host: Host, remote: Remote):
        self.host = host
        self.remote = remote

    # -- shared helpers ---------------------------------------------------
    def remote_env_export(self, pool: Pool | None) -> str:
        """Build the `export ...` block for a run (PATH, UV_*, venv)."""
        lines: list[str] = []
        uv_bin = self.host.env.get("uv_bin", "")
        if uv_bin:
            bindir = posixpath.dirname(uv_bin)
            lines.append(f'export PATH="{bindir}:$PATH"')
        uv_cache = self.host.env.get("uv_cache_dir")
        if uv_cache:
            lines.append(f'export UV_CACHE_DIR="{uv_cache}"')
        uv_python = self.host.env.get("uv_python_install_dir")
        if uv_python:
            lines.append(f'export UV_PYTHON_INSTALL_DIR="{uv_python}"')
        if self.host.env.get("offline"):
            lines.append("export UV_OFFLINE=1")
        if pool is not None:
            lines.append(f'export UV_PROJECT_ENVIRONMENT="{pool.venv_path(self.host)}"')
        env = dict(self.host.env.get("env", {}))
        if pool is not None:
            env.update(pool.env)
        lines.extend(shell_export_lines(env).splitlines())
        return "\n".join(line for line in lines if line.strip())

    def venv_python(self, pool: Pool | None) -> str:
        if pool is not None:
            return self.remote.expand(f"{pool.venv_path(self.host)}/bin/python")
        return self.remote.expand(f"{self.host.code.rstrip('/')}/.venv/bin/python")

    def run_prefix(self) -> str:
        return self.host.env.get("run_prefix", "uv run --no-sync")

    def runner_dir(self) -> str:
        return self.remote.expand(f"{self.host.tool_dir.rstrip('/')}/bin")

    def ensure_remote_tools(self) -> dict:
        """Push rr_runner.sh and rr_meta.py to the remote, content-addressed.

        Idempotent: unchanged tools are not re-uploaded.
        """
        runner = (RUNNER_DIR / "rr_runner.sh").read_text()
        meta_py = (RUNNER_DIR / "rr_meta.py").read_text()
        tools = {"runner": runner, "meta": meta_py}
        digest = sha256_text(runner + "\n--8<--\n" + meta_py)[:16]
        base = self.runner_dir()
        runner_path = f"{base}/rr_runner.sh"
        meta_path = f"{base}/rr_meta.py"
        stamp = f"{base}/.stamp-{digest}"

        script = r"""
set -euo pipefail
base="$1"; stamp="$2"
mkdir -p "$base"
if [ -f "$stamp" ]; then exit 0; fi
exit 1
"""
        if not self.remote.ok(script, [base, stamp]):
            self.remote.put_text(runner_path, runner)
            self.remote.put_text(meta_path, meta_py)
            self.remote.run(
                'chmod +x "$1"; : > "$2"',
                [runner_path, stamp],
            )
        return {
            "runner": runner_path,
            "meta": meta_path,
            "digest": digest,
        }

    def base_runner_env(self, pool: Pool | None, tools: dict, meta_b64: str, cmd_b64: str) -> dict:
        login_setup = self.host.slurm.get("login_setup", "")
        return {
            "RR_META_PY": tools["meta"],
            "RR_RUNNER": tools["runner"],
            "RR_WORKDIR": self.host.code,
            "RR_PYTHON": self.venv_python(pool),
            "RR_RUN_PREFIX": self.run_prefix(),
            "RR_META_B64": meta_b64,
            "RR_CMD_B64": cmd_b64,
            "RR_ENV_B64": b64(self.remote_env_export(pool)),
            "RR_LOGIN_SETUP_B64": b64(login_setup) if login_setup else "",
        }

    def runner_export_block(self, env: dict) -> str:
        lines = []
        for key, value in env.items():
            if value is None or value == "":
                continue
            lines.append(f"export {key}={_shq(str(value))}")
        return "\n".join(lines)

    # -- interface (overridden by subclasses) -----------------------------
    def launch(self, *, pool, meta, cmd_str, run_dir):  # noqa: ANN001
        raise NotImplementedError

    def list_jobs(self) -> list[dict]:
        return self.list_remote_metas()

    def live_state(self, meta: dict) -> dict:
        return {"state": meta.get("state"), "live": False}

    def cancel(self, meta: dict) -> dict:
        raise JobError("cancel is not supported by this backend", backend=self.name)

    # -- metadata listing -------------------------------------------------
    def runs_root(self) -> str:
        return self.remote.expand(self.host.runs)

    def legacy_root(self) -> str | None:
        if not self.host.legacy_runs:
            return None
        return self.remote.expand(self.host.legacy_runs)

    def run_dir_of(self, meta: dict) -> str:
        if meta.get("legacy_root"):
            return meta["legacy_root"]
        return f"{self.runs_root().rstrip('/')}/{meta.get('run_id')}"

    def list_legacy_metas(self) -> list[dict]:
        return []

    def list_remote_metas(self) -> list[dict]:
        """Read every runs/<id>/meta.json as structured dicts (one ssh call)."""
        import json

        script = r"""
set -uo pipefail
runs="$1"
[ -d "$runs" ] || exit 0
for m in "$runs"/*/meta.json; do
  [ -e "$m" ] || continue
  printf '###META###%s\n' "$m"
  cat "$m"
  printf '\n'
done
"""
        out = self.remote.out(script, [self.runs_root()])
        metas: list[dict] = []
        current: list[str] = []
        for line in out.splitlines():
            if line.startswith("###META###"):
                if current:
                    _flush(metas, "\n".join(current), json)
                current = []
            elif current or line.startswith("{"):
                current.append(line)
        if current:
            _flush(metas, "\n".join(current), json)
        return metas

    def read_meta(self, run_dir: str) -> dict:
        import json

        text = self.remote.get_text(f"{run_dir}/meta.json")
        try:
            return json.loads(text) if text.strip() else {}
        except json.JSONDecodeError:
            return {}

    def write_meta(self, run_dir: str, meta: dict) -> None:
        import json

        self.remote.put_text(f"{run_dir}/meta.json", json.dumps(meta, indent=2, ensure_ascii=False) + "\n")

    def write_meta_atomic(self, run_dir: str, meta: dict) -> None:
        """Atomically publish meta.json (write temp + rename on the remote)."""
        import json

        target = f"{run_dir}/meta.json"
        tmp = f"{target}.tmp"
        self.remote.put_text(tmp, json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
        self.remote.run('mv -f "$1" "$2"', [tmp, target])

    def reconcile(self, meta: dict, live: dict) -> dict | None:  # noqa: ANN001
        """Best-effort terminal write-back of a *trustworthy* live state.

        Returns the updated meta dict when the metadata was advanced to a
        terminal state, else ``None``.  Never rewrites a run that is already
        terminal and never invents a state the backend cannot justify.
        """
        return None

    def tail_logs(self, run_dir: str, which: str, lines: int, meta: dict | None = None) -> str:  # noqa: D401
        if meta is not None and meta.get("legacy"):
            tag = meta.get("legacy_tag") or meta.get("experiment")
            log = f"{run_dir.rstrip('/')}/{tag}.log"
            script = 'f="$1"; if [ -f "$f" ]; then tail -n "$2" "$f"; else printf "(no log at %s)\\n" "$f"; fi'
            return self.remote.out(script, [log, str(lines)])
        script = r"""
set -uo pipefail
d="$1"; which="$2"; n="$3"
case "$which" in
  stdout) f="$d/stdout.log" ;;
  stderr) f="$d/stderr.log" ;;
  slurm)  f="$d/slurm.out" ;;
  *) f="$d/stdout.log" ;;
esac
if [ -f "$f" ]; then tail -n "$n" "$f"; else printf '(no %s yet)\n' "$f"; fi
"""
        return self.remote.out(script, [run_dir, which, str(lines)])


def _flush(metas: list, text: str, json) -> None:
    text = text.strip()
    if not text:
        return
    # meta.json is a single JSON object; tolerate a leading marker residue.
    start = text.find("{")
    if start < 0:
        return
    try:
        metas.append(json.loads(text[start:]))
    except json.JSONDecodeError:
        return


def _shq(value: str) -> str:
    import shlex

    return shlex.quote(value)