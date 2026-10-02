"""High-level operations behind the `rr` commands.

Everything here is stateless and recoverable: remote filesystem meta.json plus
Slurm/Git are the sources of truth.  Backends only differ in how a job is
launched/monitored; host/pool/regime plumbing is shared.
"""

from __future__ import annotations

import contextlib
import json
import re
import shlex
import signal
import tempfile
import threading
import time
from pathlib import Path

from .backends import get_backend
from .config import Config, Host, Pool
from .errors import DeployError, JobError, PoolError, RRException
from .meta import build_meta, is_terminal, new_run_id, summarize
from .remote import Remote
from .util import (
    driver_at_least,
    expected_cuda_for_backend,
    git_info,
    git_repo_root,
    now_iso,
    run_local,
    sha256_file,
    short_hex,
)

DRIVER_RE = re.compile(r"Kernel Module\s+([0-9][0-9A-Za-z.\-+]*)")

DEPLOY_SCHEMA = "rr/deployment/v1"


# ---------------------------------------------------------------------------
# hosts
# ---------------------------------------------------------------------------

def list_hosts(config: Config) -> list[dict]:
    out = []
    for host in config.hosts.values():
        out.append(
            {
                "host": host.name,
                "ssh_alias": host.ssh_alias,
                "backend": host.backend,
                "code": host.code,
                "runs": host.runs,
                "default_pool": host.default_pool,
                "pools": [p.name for p in config.pools_for_host(host.name)],
                "offline": bool(host.env.get("offline")),
            }
        )
    return out


def _slurm_prelude(host: Host) -> str:
    setup = host.slurm.get("login_setup", "")
    if not setup:
        return ""
    return f"eval {shlex.quote(setup)} >/dev/null 2>&1 || true\n"


def _uv_prefix(host: Host) -> str:
    uv_bin = host.env.get("uv_bin", "uv")
    return f'export PATH="{Path(uv_bin).parent if uv_bin != "uv" else "$HOME/.local/bin"}:$PATH"'


# ---------------------------------------------------------------------------
# deployment markers
# ---------------------------------------------------------------------------
#
# A Git checkout at the right commit is NOT proof that the environment was
# synced successfully: `deploy` updates the remote checkout first and then runs
# `uv sync`.  If the sync fails, the commit already moved.  The deployment
# marker is the atomic proof that Git + env sync + validation all succeeded for
# (host, pool, commit, environment identity).  `rr run` refuses without it.


def pool_lock_rel(pool: Pool) -> str:
    return pool.lock_file or "uv.lock"


def env_identity(root: str | Path, pool: Pool) -> dict:
    """Identity of the environment a pool needs (lockfiles that pin it)."""
    root = Path(root)
    lock_rel = pool_lock_rel(pool)
    lock_abs = root / lock_rel
    return {
        "lock_file": lock_rel,
        "lock_exists": lock_abs.is_file(),
        "lock_hash": sha256_file(lock_abs),
        "pyproject_hash": sha256_file(root / "pyproject.toml"),
        "python_version_hash": sha256_file(root / ".python-version"),
    }


def marker_path(host: Host, pool: Pool) -> str:
    return f"{host.marker_dir().rstrip('/')}/{pool.name}.json"


def read_marker(remote: Remote, host: Host, pool: Pool) -> dict | None:
    try:
        text = remote.get_text(marker_path(host, pool))
    except RRException:
        return None
    if not text.strip():
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def write_marker(remote: Remote, host: Host, pool: Pool, marker: dict) -> str:
    """Atomically publish a marker (write temp + rename on the remote)."""
    path = marker_path(host, pool)
    tmp = f"{path}.tmp-{short_hex(8)}"
    remote.put_text(tmp, json.dumps(marker, indent=2, ensure_ascii=False) + "\n")
    remote.run('mv -f "$1" "$2"', [tmp, path])
    return path


def invalidate_markers(remote: Remote, paths: list[str]) -> None:
    if not paths:
        return
    remote.run('for p in "$@"; do rm -f "$p"; done', paths, check=False)


def pools_sharing_env(config: Config, host: Host, base: Pool | None) -> list[Pool]:
    """Pools whose venv+lock are the same environment as ``base``.

    One `uv sync` provisions all of them; `rr deploy` therefore publishes a
    marker for each so `rr run --pool X` works for any of the aliases.
    """
    if base is None:
        return []
    out = []
    for pool in config.pools_for_host(host.name):
        if pool.backend != base.backend:
            continue
        if pool.venv_path(host) != base.venv_path(host):
            continue
        if pool_lock_rel(pool) != pool_lock_rel(base):
            continue
        out.append(pool)
    return out or [base]


def deployment_status(
    config: Config,
    host: Host,
    pool: Pool,
    info: dict,
    root: Path,
    remote: Remote | None = None,
) -> dict:
    """Compare a pool's deployment marker with the local commit + env identity."""
    remote = remote or Remote(host.ssh_alias, config.ssh_opts)
    identity = env_identity(root, pool)
    marker = read_marker(remote, host, pool)
    reasons: list[str] = []
    if marker is None:
        reasons.append("missing")
    else:
        if marker.get("commit") != info.get("commit"):
            reasons.append("commit_mismatch")
        if marker.get("lock_hash") != identity.get("lock_hash"):
            reasons.append("lock_mismatch")
        if marker.get("venv") != pool.venv_path(host):
            reasons.append("venv_mismatch")
        if not marker.get("validated"):
            reasons.append("not_validated")
    return {
        "ok": not reasons,
        "reasons": reasons,
        "marker": marker,
        "identity": identity,
        "path": marker_path(host, pool),
    }


def require_deployment(
    config: Config,
    host: Host,
    pool: Pool | None,
    info: dict,
    root: Path,
    remote: Remote,
) -> dict | None:
    """Refuse a run unless (host, pool, commit, env) has a validated marker."""
    if pool is None:
        return None
    status = deployment_status(config, host, pool, info, root, remote=remote)
    if status["ok"]:
        return status
    raise JobError(
        f"no valid deployment for pool '{pool.name}' at commit "
        f"{(info.get('commit') or '')[:12]}; run `rr deploy {host.name} --pool {pool.name}`",
        code="rr.job.not_deployed",
        host=host.name,
        pool=pool.name,
        commit=(info.get("commit") or "")[:12],
        reasons=status["reasons"],
        marker_path=status["path"],
    )


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------

def doctor(config: Config, host_name: str, *, probe: bool = True, refresh: bool = False) -> dict:
    host = config.get_host(host_name)
    remote = Remote(host.ssh_alias, config.ssh_opts)
    checks: list[dict] = []

    def add(name: str, status: str, detail: str = "", **extra):
        checks.append({"name": name, "status": status, "detail": detail, **extra})

    result: dict = {
        "host": host.name,
        "ssh_alias": host.ssh_alias,
        "backend": host.backend,
        "checks": checks,
    }

    # --- ssh -------------------------------------------------------------
    try:
        info = remote.out('printf "%s|%s|%s\\n" "$(hostname)" "$(whoami)" "$(pwd)"').strip()
        hname, user, _ = (info.split("|") + ["", "", ""])[:3]
        add("ssh", "ok", f"{user}@{hname}")
        result["hostname"] = hname
    except RRException as exc:
        add("ssh", "fail", exc.message)
        result["ok"] = False
        result["summary"] = _summary(checks)
        return result

    # --- checkout --------------------------------------------------------
    code_info = remote.out(
        r"""
set -uo pipefail
code="$1"
if [ ! -d "$code/.git" ]; then echo "MISSING"; exit 0; fi
commit="$(git -C "$code" rev-parse HEAD 2>/dev/null)"
dirty="$(git -C "$code" status --porcelain --untracked-files=no 2>/dev/null | head -1)"
branch="$(git -C "$code" symbolic-ref --short HEAD 2>/dev/null || echo DETACHED)"
printf '%s|%s|%s|%s\n' "$commit" "$branch" "${dirty:+dirty}" "$code"
""",
        [host.code],
    ).strip()
    parts = code_info.split("|")
    if parts and parts[0] == "MISSING":
        add("checkout", "warn", f"no git checkout at {host.code} (run `rr deploy {host.name}`)")
        result["code"] = {"exists": False, "path": host.code}
    else:
        commit, branch, dirty_flag, path = (parts + ["", "", "", ""])[:4]
        status = "warn" if dirty_flag else "ok"
        add("checkout", status, f"{commit[:12]} ({branch}){' DIRTY' if dirty_flag else ''}")
        result["code"] = {
            "exists": True,
            "path": path or host.code,
            "commit": commit,
            "branch": branch,
            "dirty": bool(dirty_flag),
        }

    # --- GPUs (host-level; Slurm compute GPUs are validated per node) ----
    gpu_raw = remote.out(
        'nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader 2>&1 | head -20'
    ).strip()
    no_tool = ("command not found" in gpu_raw) or ("not found" in gpu_raw)
    if host.is_slurm and (no_tool or not gpu_raw):
        add("gpu", "ok", "login node has no GPU (compute GPUs are validated per node)")
        result["gpus"] = []
    elif (not gpu_raw) or ("Unable" in gpu_raw) or ("error" in gpu_raw.lower()) or ("No devices" in gpu_raw):
        add("gpu", "warn", (gpu_raw.replace("\n", "; ") or "nvidia-smi returned nothing")[:160])
        result["gpus"] = []
    else:
        rows = [line for line in gpu_raw.splitlines() if line.strip()]
        result["gpus"] = rows
        add("gpu", "ok", "; ".join(rows)[:160])

    # --- uv / offline env ------------------------------------------------
    offline = bool(host.env.get("offline"))
    uv_bin = host.env.get("uv_bin", "uv")
    uv_out = remote.out(
        r"""
set -uo pipefail
uv_bin="$1"
export PATH="$(dirname "$uv_bin"):$PATH"
if ! command -v "$uv_bin" >/dev/null 2>&1 && [ ! -x "$uv_bin" ]; then echo "MISSING"; exit 0; fi
"$uv_bin" --version 2>/dev/null || echo "BROKEN"
""",
        [uv_bin],
    ).strip()
    if uv_out == "MISSING":
        add("uv", "fail", f"uv not found at {uv_bin}")
    elif uv_out == "BROKEN":
        add("uv", "fail", f"uv at {uv_bin} is not runnable")
    else:
        add("uv", "ok", uv_out)

    if offline:
        cache = host.env.get("uv_cache_dir", "")
        py_dir = host.env.get("uv_python_install_dir", "")
        env_detail = remote.out(
            r"""
set -uo pipefail
cache="$1"; py="$2"
cc=missing; [ -d "$cache" ] && [ -n "$(ls -A "$cache" 2>/dev/null)" ] && cc=ok
pp=missing; [ -d "$py" ] && [ -n "$(ls -A "$py" 2>/dev/null)" ] && pp=ok
printf 'cache=%s|python=%s\n' "$cc" "$pp"
""",
            [cache, py_dir],
        ).strip()
        fields = dict(x.split("=", 1) for x in env_detail.split("|") if "=" in x)
        if fields.get("cache") == "ok" and fields.get("python") == "ok":
            add("offline-env", "ok", "uv cache + python present")
        else:
            add(
                "offline-env",
                "fail",
                f"uv cache/{fields.get('cache')} python/{fields.get('python')} "
                "(run references/res2-bootstrap.md)",
            )

    # --- venvs / torch ---------------------------------------------------
    pool_list = config.pools_for_host(host.name)
    default_pool = None
    if host.default_pool and host.default_pool in config.pools:
        default_pool = config.get_pool(host.default_pool)
    env_pools = pool_list or ([default_pool] if default_pool else [])
    pool_reports = []
    for pool in env_pools:
        venv = pool.venv_path(host)
        probe_script = (
            _uv_prefix(host)
            + "\n"
            + _slurm_prelude(host)
            + r"""
set -uo pipefail
venv="$1"
py="$venv/bin/python"
if [ ! -x "$py" ]; then printf 'venv=missing\n'; exit 0; fi
ver="$("$py" -c 'import sys;print(sys.version.split()[0])' 2>/dev/null || echo '?')"
printf 'venv=ok\npython=%s\n' "$ver"
"$py" - <<'PY' 2>/dev/null || printf 'torch_error=import_failed\n'
try:
    import torch
    print("torch_version=%s" % torch.__version__)
    print("torch_cuda=%s" % (torch.version.cuda or ""))
    print("cuda_available=%s" % bool(torch.cuda.is_available()))
    print("cuda_count=%s" % (torch.cuda.device_count() if torch.cuda.is_available() else 0))
except Exception as e:
    print("torch_error=%s" % type(e).__name__)
PY
"""
        )
        detail_raw = remote.out(probe_script, [venv]).strip()
        fields: dict[str, str] = {}
        for chunk in detail_raw.splitlines():
            if "=" in chunk:
                k, v = chunk.split("=", 1)
                fields[k.strip()] = v.strip()

        declared = pool.torch_backend
        expected_cuda = expected_cuda_for_backend(declared)
        actual_cuda = fields.get("torch_cuda") or None
        torch_version = fields.get("torch_version") or None
        venv_state = fields.get("venv", "?")
        cuda_available = fields.get("cuda_available")
        backend_ok: bool | None = None
        if venv_state == "ok" and expected_cuda and actual_cuda:
            backend_ok = actual_cuda.startswith(expected_cuda)

        env_status = "ok"
        if venv_state != "ok":
            env_status = "warn" if not pool.provisioned else "fail"
            detail = (
                f"not provisioned ({venv})" if not pool.provisioned else f"missing venv {venv}"
            )
        else:
            detail = f"{venv} → python {fields.get('python', '?')} torch {torch_version or '?'} cuda {actual_cuda or '-'}"
            if cuda_available is not None:
                detail += f" avail={cuda_available}"
            if backend_ok is False:
                env_status = "fail" if pool.provisioned else "warn"
                detail += f" [torch backend mismatch: {declared} expects cuda {expected_cuda}, got {actual_cuda}]"

        pool_reports.append(
            {
                "pool": pool.name,
                "venv": venv,
                "torch_backend": declared,
                "expected_cuda": expected_cuda,
                "torch_cuda": actual_cuda,
                "torch_version": torch_version,
                "cuda_available": cuda_available,
                "backend_ok": backend_ok,
                "provisioned": pool.provisioned,
                "venv_state": venv_state,
            }
        )
        add(f"env:{pool.name}", env_status, detail)
    result["pool_reports"] = pool_reports

    # --- slurm -----------------------------------------------------------
    node_drivers: dict[str, str] = {}
    if host.is_slurm:
        slurm_info = remote.out(
            _slurm_prelude(host)
            + r"""
set -uo pipefail
ver="$(sbatch --version 2>/dev/null || echo MISSING)"
q="$(squeue -u "$USER" -h -o '%i|%j|%T|%R' 2>/dev/null || true)"
quota="$(sacctmgr -n show assoc user="$USER" format=GrpTRES%40 2>/dev/null | head -1 || true)"
printf 'version=%s\n' "$ver"
printf 'quota=%s\n' "$quota"
printf 'queue=%s\n' "$(printf '%s' "$q" | tr '\n' ';')"
"""
        )
        sd = {}
        for line in slurm_info.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                sd[k.strip()] = v.strip()
        result["slurm"] = sd
        if "slurm 21" in sd.get("version", "") or sd.get("version", "").startswith("slurm"):
            add("slurm", "ok", f"{sd.get('version')} quota={sd.get('quota')}")
        else:
            add("slurm", "fail", f"sbatch unavailable ({sd.get('version')})")

        all_nodes = sorted({n for p in pool_list for n in p.eligible_nodes})
        node_states = probe_node_states(remote, host)
        if node_states:
            result["node_states"] = node_states
            idle = sum(1 for s in node_states.values() if s.get("state", "").startswith("idle"))
            add("nodes", "ok" if idle else "warn", f"{len(node_states)} nodes, {idle} idle (sinfo)")
        if all_nodes and probe:
            node_drivers = probe_node_drivers(config, host, all_nodes, refresh=refresh)
            result["node_drivers"] = node_drivers
            if node_drivers:
                source = "probe" if refresh else "cache"
                result["node_driver_source"] = source
                add(
                    "node-drivers",
                    "ok",
                    f"({source}) " + ", ".join(f"{n}={d or '?'}" for n, d in sorted(node_drivers.items())),
                )
            elif refresh:
                add("node-drivers", "warn", "probe submitted but no driver read back")
            else:
                add(
                    "node-drivers",
                    "warn",
                    "unverified; run `rr doctor HOST --refresh` (submits short probe jobs)",
                )

    # --- pool compatibility ---------------------------------------------
    pool_compat = []
    for pool in pool_list:
        compat = pool.eligible_nodes
        if pool.min_driver and node_drivers:
            compat = [n for n in pool.eligible_nodes if driver_at_least(node_drivers.get(n), pool.min_driver)]
        rep = {
            "pool": pool.name,
            "backend": pool.backend,
            "torch_backend": pool.torch_backend,
            "min_driver": pool.min_driver,
            "provisioned": pool.provisioned,
            "eligible_nodes": pool.eligible_nodes,
            "compatible_nodes": compat,
            "incompatible_nodes": [n for n in pool.eligible_nodes if n not in compat],
        }
        pool_compat.append(rep)
        if not pool.provisioned:
            add(f"pool:{pool.name}", "warn", "configured but environment not provisioned")
        elif pool.min_driver and not node_drivers:
            add(f"pool:{pool.name}", "warn", "driver unverified (run `rr doctor --refresh`)")
        elif not pool.eligible_nodes and not pool.min_driver:
            add(f"pool:{pool.name}", "ok", "unconstrained (host-level pool)")
        elif compat:
            add(f"pool:{pool.name}", "ok", f"nodes={','.join(compat)}")
        else:
            add(f"pool:{pool.name}", "fail", "no eligible node satisfies min_driver")
    result["pools"] = pool_compat

    result["summary"] = _summary(checks)
    result["ok"] = result["summary"]["fail"] == 0
    return result


def _summary(checks: list[dict]) -> dict:
    return {
        "ok": sum(1 for c in checks if c["status"] == "ok"),
        "warn": sum(1 for c in checks if c["status"] == "warn"),
        "fail": sum(1 for c in checks if c["status"] == "fail"),
    }


@contextlib.contextmanager
def _sigterm_as_interrupt():
    """Turn SIGTERM into KeyboardInterrupt so ``finally`` cleanup always runs."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGTERM)

    def _handler(signum, frame):  # noqa: ANN001
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, _handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def node_probe_cache_file(host: Host) -> Path:
    return Path.home() / ".cache" / "rr" / f"nodes-{host.name}.json"


def load_node_driver_cache(host: Host, nodes: list[str], *, ttl: float = 6 * 3600) -> dict[str, str]:
    cache_file = node_probe_cache_file(host)
    if not cache_file.exists():
        return {}
    try:
        data = json.loads(cache_file.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if time.time() - data.get("at", 0) >= ttl:
        return {}
    drivers = data.get("drivers", {})
    if not isinstance(drivers, dict) or not set(nodes) <= set(drivers):
        return {}
    return drivers


def save_node_driver_cache(host: Host, drivers: dict[str, str], job_ids: list[str]) -> None:
    cache_file = node_probe_cache_file(host)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps({"at": time.time(), "drivers": drivers, "job_ids": job_ids}, indent=2) + "\n"
    )


def node_probe_token() -> str:
    return short_hex(8)


def node_probe_submit_script(host: Host) -> str:
    """Submit the node-driver probe jobs and print their ids immediately.

    Every invocation gets a unique token (job name + scratch dir) and the script
    only references the ids *it* submitted.  A guarded trap scancels jobs
    started so far if the submit phase itself is interrupted before it finishes
    printing ids; on normal completion the caller owns the job lifecycle.
    """
    return _slurm_prelude(host) + r"""
set -uo pipefail
dir="$1"; part="$2"; token="$3"; shift 3
mkdir -p "$dir"
jobids=""
done=0
cleanup() { [ "$done" = "1" ] && return 0; for j in $jobids; do scancel "$j" >/dev/null 2>&1 || true; done; }
trap cleanup EXIT HUP INT TERM
for n in "$@"; do
  rm -f "$dir/$n.out"
  jid="$(sbatch --parsable -J "rrnode-$token" -p "$part" -w "$n" -c 1 --mem=200M -t 3 \
      -o "$dir/$n.out" --wrap='cat /proc/driver/nvidia/version 2>/dev/null | head -1' 2>/dev/null)"
  jid="${jid%%;*}"
  case "$jid" in
    ''|*[!0-9]*) ;;
    *) jobids="$jobids $jid"; printf 'RRPROBEJOB|%s\n' "$jid" ;;
  esac
done
done=1
"""


def node_probe_status_script(host: Host) -> str:
    """Print the given job ids that are still known to the scheduler."""
    return _slurm_prelude(host) + r"""
set -uo pipefail
for j in "$@"; do
  st="$(squeue -h -j "$j" -o '%T' 2>/dev/null | head -1)"
  [ -n "$st" ] && printf 'RRPROBEALIVE|%s|%s\n' "$j" "$st"
done
"""


def node_probe_collect_script(host: Host) -> str:
    """Read the driver line out of each node's probe output."""
    return _slurm_prelude(host) + r"""
set -uo pipefail
dir="$1"; shift
for n in "$@"; do
  if [ -f "$dir/$n.out" ]; then
    printf 'RRPROBEDRV|%s|%s\n' "$n" "$(grep -o 'Kernel Module[[:space:]]*[0-9][^ ]*' "$dir/$n.out" | head -1)"
  fi
done
"""


def node_probe_scancel_script(host: Host) -> str:
    """Cancel exactly the given job ids (best effort)."""
    return _slurm_prelude(host) + r"""
set -uo pipefail
for j in "$@"; do scancel "$j" >/dev/null 2>&1 || true; done
"""


def node_probe_script(host: Host) -> str:
    """Back-compat alias for the submit script."""
    return node_probe_submit_script(host)


def parse_probe_job_ids(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        if line.startswith("RRPROBEJOB|"):
            jid = line.split("|", 1)[1].strip()
            if jid:
                out.append(jid)
    return out


def parse_probe_alive(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        if line.startswith("RRPROBEALIVE|"):
            parts = line.split("|")
            if len(parts) >= 2 and parts[1].strip():
                out.append(parts[1].strip())
    return out


def parse_probe_drivers(text: str) -> dict[str, str]:
    drivers: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith("RRPROBEDRV|"):
            continue
        parts = line.split("|", 2)
        if len(parts) < 3:
            continue
        node, raw = parts[1].strip(), parts[2]
        m = DRIVER_RE.search(raw) or re.search(r"([0-9]{3}\.[0-9]+\.[0-9]+)", raw)
        drivers[node] = m.group(1) if m else ""
    return drivers


def parse_node_probe_output(text: str) -> tuple[dict[str, str], list[str]]:
    return parse_probe_drivers(text), parse_probe_job_ids(text)


def probe_node_states(remote: Remote, host: Host) -> dict[str, dict]:
    """Non-invasive live node view (sinfo).  Never submits a job."""
    if not host.is_slurm:
        return {}
    script = _slurm_prelude(host) + r"""
set -uo pipefail
sinfo -h -N -o '%N|%T|%G' 2>/dev/null | sort -u || true
"""
    try:
        cp = remote.run(script, check=False)
    except RRException:
        return {}
    nodes: dict[str, dict] = {}
    for line in cp.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 3 or not parts[0].strip():
            continue
        name = parts[0].strip()
        nxt = {"state": parts[1].strip(), "gres": parts[2].strip()}
        prev = nodes.get(name)
        # A node may appear per partition; keep the most "available" state.
        if prev is None or "idle" in nxt["state"] or "idle" not in prev["state"]:
            nodes[name] = nxt
    return nodes


def probe_node_drivers(
    config: Config,
    host: Host,
    nodes: list[str],
    *,
    refresh: bool = False,
    ttl: float = 6 * 3600,
    max_wait: float = 180.0,
    poll_interval: float = 3.0,
) -> dict[str, str]:
    """Return per-node drivers.

    ``refresh=False`` (the default for `rr doctor`) is strictly non-invasive:
    it reads the local cache and returns ``{}`` when the cache is missing or
    stale.  Jobs are only ever submitted with ``refresh=True``.

    With ``refresh=True`` the caller owns the probe-job lifecycle: the ids are
    recorded, only those ids are polled, and a ``finally`` block best-effort
    scancels any still-live probe (on timeout, Ctrl-C or SIGTERM).  Probe jobs
    also carry a short ``-t 3`` time limit, so even a hard kill cannot leave a
    long-lived orphan.
    """
    if not refresh:
        return load_node_driver_cache(host, nodes, ttl=ttl)

    remote = Remote(host.ssh_alias, config.ssh_opts)
    partition = host.slurm.get("partition", "gpu")
    token = node_probe_token()
    probe_dir = f"{host.tool_dir.rstrip('/')}/node-probe/{token}"
    drivers: dict[str, str] = {}
    job_ids: list[str] = []
    try:
        with _sigterm_as_interrupt():
            cp = remote.run(
                node_probe_submit_script(host),
                [probe_dir, partition, token, *nodes],
                check=False,
                timeout=90,
            )
            job_ids = parse_probe_job_ids(cp.stdout)
            deadline = time.time() + max_wait
            alive = list(job_ids)
            while alive and time.time() < deadline:
                time.sleep(poll_interval)
                status = remote.run(node_probe_status_script(host), alive, check=False, timeout=60)
                alive = parse_probe_alive(status.stdout)
            collected = remote.run(
                node_probe_collect_script(host), [probe_dir, *nodes], check=False, timeout=60
            ).stdout
            drivers = parse_probe_drivers(collected)
    except RRException:
        drivers = {}
    finally:
        # Never leave orphan probe jobs: cancel exactly the ids we submitted.
        if job_ids:
            try:
                remote.run(node_probe_scancel_script(host), job_ids, check=False)
            except RRException:
                pass
        try:
            remote.run('rm -rf "$1"', [probe_dir], check=False)
        except RRException:
            pass

    if drivers:
        save_node_driver_cache(host, drivers, job_ids)
    return drivers


# ---------------------------------------------------------------------------
# deploy
# ---------------------------------------------------------------------------

def deploy(
    config: Config,
    host_name: str,
    *,
    pool_name: str | None = None,
    allow_dirty: bool = False,
    dry_run: bool = False,
) -> dict:
    host = config.get_host(host_name)
    root = git_repo_root()
    if root is None:
        raise DeployError("not inside a local git repository", code="rr.deploy.no_repo")
    info = git_info(root)
    if info["dirty"] and not allow_dirty:
        raise DeployError(
            "local tracked files are modified; commit before deploying "
            "(or pass --allow-dirty to deploy a recorded dirty revision)",
            code="rr.deploy.dirty_local",
            commit=info["commit"],
            branch=info["branch"],
            diff_hash=info["diff_hash"],
            untracked=info["untracked_count"],
        )
    if info["branch"] == "DETACHED":
        raise DeployError("detached HEAD; check out a branch before deploy", code="rr.deploy.detached")

    pool = config.resolve_pool(host, pool_name)
    remote = Remote(host.ssh_alias, config.ssh_opts)

    # Per-pool environment strategy: a pool that declares its own lockfile must
    # have that lockfile committed, so its environment can never be silently
    # built from a different dependency set (e.g. cu124 reused as cu118).
    identity = env_identity(root, pool) if pool else None
    if pool is not None and not identity["lock_exists"]:
        raise DeployError(
            f"pool '{pool.name}' requires lockfile '{identity['lock_file']}', which is not committed",
            code="rr.deploy.lock_missing",
            host=host.name,
            pool=pool.name,
            lock_file=identity["lock_file"],
        )
    target_pools = pools_sharing_env(config, host, pool) if pool else []
    marker_paths = [marker_path(host, p) for p in target_pools]

    remote_commit = remote.out(
        'if [ -d "$1/.git" ]; then git -C "$1" rev-parse HEAD 2>/dev/null; fi',
        [host.code],
    ).strip()

    deploy_dir = f"{host.tool_dir.rstrip('/')}/deploy"
    remote.run('mkdir -p "$1"', [deploy_dir])
    bundle_remote = f"{deploy_dir}/{info['branch'].replace('/', '_')}.bundle"

    with tempfile.TemporaryDirectory() as tmp:
        bundle_local = str(Path(tmp) / "rr.bundle")
        incremental = False
        up_to_date = bool(remote_commit) and remote_commit == info["commit"]
        if remote_commit and not up_to_date:
            anc = run_local(["git", "merge-base", "--is-ancestor", remote_commit, info["commit"]], cwd=root)
            incremental = anc.returncode == 0
        if dry_run:
            return {
                "host": host.name,
                "commit": info["commit"],
                "branch": info["branch"],
                "remote_commit": remote_commit or None,
                "incremental": incremental,
                "up_to_date": up_to_date,
                "dry_run": True,
                "pool": pool.name if pool else None,
                "lock_file": identity["lock_file"] if identity else None,
                "markers": marker_paths,
            }
        if not up_to_date:
            if incremental:
                cmd = ["git", "bundle", "create", bundle_local, info["branch"], "--not", remote_commit]
            else:
                cmd = ["git", "bundle", "create", bundle_local, info["branch"]]
            cp = run_local(cmd, cwd=root)
            if cp.returncode != 0:
                # An incremental bundle is empty when the remote already has
                # everything reachable; that is the up-to-date case.
                if incremental and "empty bundle" in (cp.stderr or ""):
                    up_to_date = True
                else:
                    raise DeployError(
                        "git bundle creation failed",
                        code="rr.deploy.bundle_failed",
                        stderr=(cp.stderr or "").strip()[-2000:],
                    )
            else:
                remote.rsync_up(bundle_local, bundle_remote)

    # Invalidate previous markers *before* mutating the remote environment: if
    # the git+sync+validation pipeline fails, `rr run` refuses (no marker for
    # this commit) instead of trusting a half-applied deployment.
    invalidate_markers(remote, marker_paths)

    venv = pool.venv_path(host) if pool else f"{host.code.rstrip('/')}/.venv"
    offline = "1" if host.env.get("offline") else "0"
    uv_bin = host.env.get("uv_bin", "uv")
    cache = host.env.get("uv_cache_dir", "")
    py_dir = host.env.get("uv_python_install_dir", "")
    sync_mode = pool.sync_mode if pool else "frozen"

    apply = r"""
set -euo pipefail
code="$1"; bundle="$2"; branch="$3"; commit="$4"; uv_bin="$5"; offline="$6"; venv="$7"; cache="$8"; pydir="$9"; need_git="${10}"; syncmode="${11}"
export PATH="$(dirname "$uv_bin"):$PATH"
[ -n "$cache" ] && export UV_CACHE_DIR="$cache"
[ -n "$pydir" ] && export UV_PYTHON_INSTALL_DIR="$pydir"
[ "$offline" = "1" ] && export UV_OFFLINE=1
export UV_PROJECT_ENVIRONMENT="$venv"

if [ "$need_git" = "1" ]; then
  if [ ! -d "$code/.git" ]; then
    mkdir -p "$(dirname "$code")"
    git clone -q -b "$branch" "$bundle" "$code"
  else
    if [ -n "$(git -C "$code" status --porcelain --untracked-files=no)" ]; then
      echo "error: remote tracked worktree is dirty; refusing to overwrite" >&2
      git -C "$code" status --short --untracked-files=no >&2
      exit 3
    fi
    git -C "$code" fetch -q "$bundle" "+refs/heads/$branch:refs/rr/deploy/$branch"
    if git -C "$code" show-ref --verify --quiet "refs/heads/$branch"; then
      git -C "$code" checkout -q "$branch"
    else
      git -C "$code" checkout -q -b "$branch" "refs/rr/deploy/$branch"
    fi
    git -C "$code" merge -q --ff-only "refs/rr/deploy/$branch"
  fi
fi

actual="$(git -C "$code" rev-parse HEAD)"
if [ "$actual" != "$commit" ]; then
  echo "error: remote HEAD $actual != expected $commit" >&2
  exit 4
fi
cd "$code"
"$uv_bin" sync --"$syncmode" >/dev/null

# --- validation: the environment must actually import the framework ---------
vpy="$venv/bin/python"
if [ ! -x "$vpy" ]; then
  echo "error: no interpreter at $vpy" >&2
  exit 5
fi
val="$("$vpy" - <<'PY' 2>&1
import json, sys
try:
    import torch
    print(json.dumps({"python": sys.version.split()[0], "torch_version": torch.__version__, "torch_cuda": torch.version.cuda}))
except Exception as e:
    print("VALIDATION_ERROR: %s: %s" % (type(e).__name__, e))
    raise SystemExit(5)
PY
)" || { echo "error: environment validation failed: $val" >&2; exit 5; }
printf 'UVVERSION|%s\n' "$("$uv_bin" --version 2>/dev/null || true)"
printf 'VALIDATED|%s\n' "$val"
printf 'deployed %s\n' "$actual"
"""
    res = remote.run(
        apply,
        [
            host.code,
            bundle_remote,
            info["branch"],
            info["commit"],
            uv_bin,
            offline,
            venv,
            cache,
            py_dir,
            "0" if up_to_date else "1",
            sync_mode,
        ],
        check=False,
    )
    if res.returncode != 0:
        raise DeployError(
            "deployment failed; no marker written, `rr run` will refuse this pool",
            code="rr.deploy.failed",
            host=host.name,
            commit=info["commit"],
            pool=pool.name if pool else None,
            exit_code=res.returncode,
            reason=(res.stderr or res.stdout or "").strip()[-2000:],
        )
    deployed = ""
    uv_version = None
    validated: dict = {}
    for line in res.stdout.splitlines():
        if line.startswith("deployed "):
            deployed = line.split(" ", 1)[1].strip()
        elif line.startswith("UVVERSION|"):
            uv_version = line.split("|", 1)[1].strip() or None
        elif line.startswith("VALIDATED|"):
            try:
                validated = json.loads(line.split("|", 1)[1])
            except json.JSONDecodeError:
                validated = {}
    if not validated.get("torch_version"):
        raise DeployError(
            "environment validation produced no torch version; refusing to mark deployed",
            code="rr.deploy.validation_missing",
            host=host.name,
            pool=pool.name if pool else None,
        )

    commit_now = deployed or info["commit"]
    written = []
    for target in target_pools:
        marker = {
            "schema": DEPLOY_SCHEMA,
            "host": host.name,
            "pool": target.name,
            "commit": commit_now,
            "branch": info["branch"],
            "venv": venv,
            "lock_file": identity["lock_file"],
            "lock_hash": identity["lock_hash"],
            "pyproject_hash": identity["pyproject_hash"],
            "torch_backend": target.torch_backend,
            "python": validated.get("python"),
            "torch_version": validated.get("torch_version"),
            "torch_cuda": validated.get("torch_cuda"),
            "uv_version": uv_version,
            "deployed_at": now_iso(),
            "validated": True,
        }
        written.append(write_marker(remote, host, target, marker))

    return {
        "host": host.name,
        "commit": commit_now,
        "branch": info["branch"],
        "backend": host.backend,
        "pool": pool.name if pool else None,
        "pools": [p.name for p in target_pools],
        "venv": venv,
        "incremental": incremental,
        "up_to_date": up_to_date,
        "remote_commit_was": remote_commit or None,
        "lock_file": identity["lock_file"] if identity else None,
        "validated": validated,
        "markers": written,
    }


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def _shell_join(argv: list[str]) -> str:
    return " ".join(shlex.quote(a) for a in argv)


def run(
    config: Config,
    host_name: str,
    experiment: str,
    argv: list[str],
    *,
    pool_name: str | None = None,
    gpus: int | None = None,
    cpus: int | None = None,
    mem: str | None = None,
    time_limit: str | None = None,
    result_paths: list[str] | None = None,
    allow_dirty: bool = False,
    allow_stale: bool = False,
) -> dict:
    if not experiment or not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*$", experiment):
        raise JobError(
            "experiment name must match [A-Za-z0-9][A-Za-z0-9._-]*",
            code="rr.job.bad_experiment",
            experiment=experiment,
        )
    if not argv:
        raise JobError("no command given (use `rr run HOST EXP -- <cmd>`)", code="rr.job.no_command")

    host = config.get_host(host_name)
    pool = config.resolve_pool(host, pool_name)
    if pool is not None and not pool.provisioned:
        raise PoolError(
            f"pool '{pool.name}' is configured but its environment is not provisioned",
            code="rr.pool.not_provisioned",
            pool=pool.name,
            hint=f"run `rr doctor {host.name}` and see references/res2-bootstrap.md",
        )

    root = git_repo_root()
    info = git_info(root) if root else {"commit": None, "branch": None, "dirty": False, "diff_hash": None}
    if info.get("dirty") and not allow_dirty:
        raise JobError(
            "local tracked files are modified; commit before a formal run "
            "(or pass --allow-dirty to run a recorded dirty revision)",
            code="rr.job.dirty_local",
            commit=info.get("commit"),
            branch=info.get("branch"),
            diff_hash=info.get("diff_hash"),
        )

    backend = get_backend(host, config.ssh_opts)

    # The remote checkout must be at the local commit: no unrecorded code.
    remote_commit = backend.remote.out(
        'if [ -d "$1/.git" ]; then git -C "$1" rev-parse HEAD 2>/dev/null; fi',
        [host.code],
    ).strip()
    if remote_commit != info.get("commit") and not allow_stale:
        raise JobError(
            f"remote checkout is not at the local commit; run `rr deploy {host.name}` first",
            code="rr.job.stale_remote",
            host=host.name,
            local_commit=(info.get("commit") or "")[:12],
            remote_commit=(remote_commit or "none")[:12],
        )

    # A matching Git commit is not proof the environment synced successfully:
    # require a validated deployment marker for this (host, pool, commit, env).
    if pool is not None and root is not None and not allow_stale:
        require_deployment(config, host, pool, info, root, backend.remote)

    requested = {
        "gpus": gpus if gpus is not None else (pool.gpus if pool else 1),
        "cpus": cpus if cpus is not None else host.slurm.get("cpus", 4),
        "mem": mem or host.slurm.get("mem", "32G"),
        "time": time_limit or host.slurm.get("time", "24:00:00"),
        "partition": host.slurm.get("partition", "gpu") if host.is_slurm else None,
        "eligible_nodes": list(pool.eligible_nodes) if pool else [],
    }
    run_id = new_run_id(experiment)
    run_dir = f"{host.runs.rstrip('/')}/{run_id}"
    meta = build_meta(
        experiment=experiment,
        run_id=run_id,
        host=host.name,
        backend=host.backend,
        pool=pool.name if pool else None,
        git=info,
        command=_shell_join(argv),
        requested=requested,
        result_paths=result_paths or [],
        torch_backend=pool.torch_backend if pool else None,
    )
    cmd_str = _shell_join(argv)
    launch = backend.launch(pool=pool, meta=meta, cmd_str=cmd_str, run_dir=run_dir)
    meta.update({k: v for k, v in launch.items() if k in {"slurm_job_id", "state"}})
    return {"meta": meta, "launch": launch, "run_dir": run_dir}


# ---------------------------------------------------------------------------
# jobs / status
# ---------------------------------------------------------------------------

def _all_runs_with_errors(
    config: Config, host_names: list[str] | None = None
) -> tuple[list[tuple[Host, dict]], list[dict]]:
    """Collect runs across hosts, reporting unreachable hosts instead of hiding them."""
    runs: list[tuple[Host, dict]] = []
    errors: list[dict] = []
    names = host_names or list(config.hosts)
    for name in names:
        host = config.get_host(name)
        backend = get_backend(host, config.ssh_opts)
        try:
            metas = backend.list_remote_metas()
        except RRException as exc:
            metas = []
            errors.append({"host": host.name, "code": exc.code, "message": exc.message})
        for meta in metas:
            meta.setdefault("host", host.name)
            runs.append((host, meta))
        try:
            legacy = backend.list_legacy_metas()
        except RRException as exc:
            if not any(e["host"] == host.name for e in errors):
                errors.append({"host": host.name, "code": exc.code, "message": exc.message})
            legacy = []
        for meta in legacy:
            meta.setdefault("host", host.name)
            runs.append((host, meta))
    return runs, errors


def _all_runs(config: Config, host_names: list[str] | None = None) -> list[tuple[Host, dict]]:
    runs, _ = _all_runs_with_errors(config, host_names)
    return runs


def _summarize_with_live(backend, meta: dict) -> tuple[dict, dict]:
    """Summarize a run, reconciling trusted terminal state back into meta."""
    summary = summarize(meta)
    if is_terminal(meta.get("state")):
        return summary, meta
    try:
        live = backend.live_state(meta)
    except RRException:
        summary["live"] = None
        return summary, meta
    try:
        updated = backend.reconcile(meta, live)
    except RRException:
        updated = None
    if updated is not None:
        meta = updated
        summary = summarize(meta)
    elif live.get("state") and not is_terminal(meta.get("state")):
        summary["state"] = live["state"]
    summary["live"] = live.get("live")
    summary["job"] = live.get("job_id") or summary.get("job")
    summary["node"] = summary.get("node") or live.get("node")
    summary["queue_reason"] = live.get("reason")
    summary["elapsed"] = live.get("elapsed")
    if live.get("scheduler_state"):
        summary["scheduler_state"] = live.get("scheduler_state")
    if live.get("cuda_visible_devices") is not None:
        summary["cuda_visible_devices"] = live.get("cuda_visible_devices")
    return summary, meta


def list_jobs_report(config: Config, host_names: list[str] | None = None) -> dict:
    runs, errors = _all_runs_with_errors(config, host_names)
    backends: dict[str, object] = {}
    rows = []
    for host, meta in runs:
        backend = backends.get(host.name)
        if backend is None:
            backend = get_backend(host, config.ssh_opts)
            backends[host.name] = backend
        summary, _ = _summarize_with_live(backend, meta)
        rows.append(summary)
    rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return {"jobs": rows, "errors": errors, "partial": bool(errors)}


def list_jobs(config: Config, host_names: list[str] | None = None) -> list[dict]:
    return list_jobs_report(config, host_names)["jobs"]


def _candidates(
    config: Config,
    ident: str,
    host_name: str | None,
    run_id: str | None,
) -> list[tuple[Host, dict]]:
    runs = _all_runs(config, [host_name] if host_name else None)
    out = []
    for host, meta in runs:
        if run_id is not None:
            if meta.get("run_id") == run_id:
                out.append((host, meta))
        elif meta.get("run_id") == ident or meta.get("experiment") == ident:
            out.append((host, meta))
    out.sort(key=lambda hm: hm[1].get("created_at") or "", reverse=True)
    return out


def resolve_run(
    config: Config,
    ident: str,
    *,
    host_name: str | None = None,
    run_id: str | None = None,
) -> tuple[Host, dict]:
    candidates = _candidates(config, ident, host_name, run_id)
    if not candidates:
        raise JobError(
            f"no run found for '{ident}'",
            code="rr.job.not_found",
            ident=ident,
            host=host_name,
        )
    return candidates[0]


def resolve_run_unique(
    config: Config,
    ident: str,
    *,
    host_name: str | None = None,
    run_id: str | None = None,
) -> tuple[Host, dict]:
    """Like ``resolve_run`` but refuses when an experiment name is ambiguous.

    Mutating commands (cancel) must never guess which of several runs was
    meant; the caller has to pass ``--run-id``.
    """
    candidates = _candidates(config, ident, host_name, run_id)
    if not candidates:
        raise JobError(
            f"no run found for '{ident}'",
            code="rr.job.not_found",
            ident=ident,
            host=host_name,
        )
    if run_id is None:
        exact = [hm for hm in candidates if hm[1].get("run_id") == ident]
        if exact:
            return exact[0]
        if len(candidates) > 1:
            raise JobError(
                f"'{ident}' matches {len(candidates)} runs; pass --run-id",
                code="rr.job.ambiguous",
                ident=ident,
                candidates=[hm[1].get("run_id") for hm in candidates],
            )
    return candidates[0]


def status(config: Config, ident: str, *, host_name=None, run_id=None) -> dict:
    host, meta = resolve_run(config, ident, host_name=host_name, run_id=run_id)
    backend = get_backend(host, config.ssh_opts)
    summary, meta = _summarize_with_live(backend, meta)
    summary["meta"] = meta
    return summary


def logs(
    config: Config,
    ident: str,
    *,
    host_name=None,
    run_id=None,
    lines: int = 40,
    which: str = "stdout",
) -> dict:
    host, meta = resolve_run(config, ident, host_name=host_name, run_id=run_id)
    backend = get_backend(host, config.ssh_opts)
    run_dir = backend.run_dir_of(meta)
    text = backend.tail_logs(run_dir, which, lines, meta)
    return {"run_id": meta.get("run_id"), "experiment": meta.get("experiment"), "which": which, "text": text}


def cancel(config: Config, ident: str, *, host_name=None, run_id=None) -> dict:
    host, meta = resolve_run_unique(config, ident, host_name=host_name, run_id=run_id)
    backend = get_backend(host, config.ssh_opts)
    return backend.cancel(meta)


def pull(
    config: Config,
    ident: str,
    *,
    host_name=None,
    run_id=None,
    dest: str | None = None,
    paths: list[str] | None = None,
    light: bool = False,
) -> dict:
    host, meta = resolve_run(config, ident, host_name=host_name, run_id=run_id)
    if meta.get("legacy"):
        raise JobError(
            "legacy run (pre-rr) has no rr-managed result paths; use the old scripts to fetch it",
            code="rr.job.legacy_not_managed",
            run_id=meta.get("run_id"),
        )
    backend = get_backend(host, config.ssh_opts)
    run_dir = backend.run_dir_of(meta)

    if dest:
        dest_root = Path(dest).expanduser().resolve()
    else:
        root = git_repo_root()
        base = root if root else Path.cwd()
        dest_root = base / ".rr" / "pulled" / str(meta.get("run_id"))
    dest_root.mkdir(parents=True, exist_ok=True)

    light_flags = ["--exclude=*.pt", "--exclude=*.pth", "--exclude=*.ckpt", "--exclude=*.bin"]
    # Always pull the run's own metadata + logs (small, authoritative).
    backend.remote.rsync_down(f"{run_dir}/", str(dest_root) + "/", extra=light_flags)

    requested = paths if paths is not None else list(meta.get("result_paths") or [])
    pulled: list[str] = []
    missing: list[str] = []
    for rel in requested:
        remote_path = rel if rel.startswith("/") else f"{host.code.rstrip('/')}/{rel.lstrip('/')}"
        kind = backend.remote.out(
            'p="$1"; if [ -d "$p" ]; then echo dir; elif [ -f "$p" ]; then echo file; else echo missing; fi',
            [remote_path],
        ).strip()
        if kind not in {"dir", "file"}:
            missing.append(rel)
            continue
        local_target = dest_root / "result" / rel.lstrip("/")
        if kind == "dir":
            local_target.mkdir(parents=True, exist_ok=True)
            backend.remote.rsync_down(
                remote_path.rstrip("/") + "/",
                str(local_target) + "/",
                extra=light_flags if light else None,
            )
        else:
            # A file result must land as a file, not a directory named after it.
            local_target.parent.mkdir(parents=True, exist_ok=True)
            backend.remote.rsync_down(
                remote_path,
                str(local_target),
                extra=light_flags if light else None,
            )
        pulled.append(rel)

    return {
        "run_id": meta.get("run_id"),
        "experiment": meta.get("experiment"),
        "dest": str(dest_root),
        "pulled": pulled,
        "missing": missing,
        "declared": requested,
    }
