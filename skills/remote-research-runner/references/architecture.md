# Architecture

```
Agent
  │
  ▼
Skill (this bundle)           policy / workflow / research SOP
  │
  ▼
rr CLI (bin/rr → rr/cli.py)   deterministic, stateless execution interface
  │
  ▼
services.py                   host/pool/regime resolution, deploy, run, pull
  │
  ├── backends/process.py     detached setsid process (res)
  └── backends/slurm.py       sbatch / squeue / sacct / scancel (res-2)
  │
  ▼
remote.py                     SSH / rsync transport (alias-based)
  │
  ▼
OpenSSH · Git bundle · Slurm · uv
```

## Design constraints (deliberate)

No daemon, no database, no Redis/Celery, no custom remote agent, no Kubernetes,
no Tailscale, no SSHFS as an execution filesystem. SSH topology (IPs, ports,
ProxyJump) stays in `~/.ssh/config`; `rr` only ever names an ssh alias.
State that must survive is written to the remote filesystem and to Slurm/Git.

## Concepts

- **host** — access entry point + remote checkout + backend. Config: `[hosts.*]`.
- **backend** — how a job is launched/monitored on that host (process | slurm).
- **pool** — *where* a job may be scheduled. Config: `[pools.*]`.
- **regime** — *what it actually ran on*, recorded in `meta.json`.

`--nodelist`, driver minimums and torch-backend selection come from the pool,
never from scripts or the Skill.

## Run directory layout (remote, human-readable)

```
<runs>/<run-id>/
  meta.json      # identity + requested + runtime provenance (authoritative)
  job.pid        # process backend: session-leader pid (liveness only)
  jobid          # slurm backend: job id
  job.slurm      # slurm backend: generated batch script (auditable)
  stdout.log
  stderr.log
  slurm-<jobid>.out / .err
```

`<runs>` is `hosts.<h>.runs` (e.g. `/home/hxy/cy/rr-runs`,
`/share/home/snsun/rr/runs`). It is outside the Git checkout so logs/results
never dirty tracked files.

## Backend interface

```python
class Backend:
    def launch(self, *, pool, meta, cmd_str, run_dir) -> dict
    def list_remote_metas(self) -> list[dict]
    def list_legacy_metas(self) -> list[dict]      # pre-rr compatibility
    def live_state(self, meta) -> dict             # ps / squeue+sacct
    def reconcile(self, meta, live) -> dict | None # terminal write-back
    def cancel(self, meta) -> dict
    def run_dir_of(self, meta) -> str
    def tail_logs(self, run_dir, which, lines, meta=None) -> str
```

Shared code (env export, tool upload, metadata, paths) lives in
`backends/base.py`; a new backend implements only the differences.

## Remote runner (backend-agnostic)

`runner/rr_runner.sh` runs the experiment and calls `runner/rr_meta.py` to
record provenance (hostname/node, driver, GPUs, torch/cuda, python, Slurm vars,
timings, exit code). Both are content-addressed and pushed to
`hosts.<h>.tool_dir/bin`; unchanged tools are not re-uploaded. The runner
enforces the pool's `min_driver` at job start (fails with exit 78 /
`driver_too_old` rather than silently producing bad CUDA results).

## Deploy (atomic via deployment markers)

Local commit → `git bundle` (incremental when the remote commit is an ancestor)
→ rsync → remote `git fetch <bundle>` + `merge --ff-only` → `uv sync --frozen`
(with `UV_OFFLINE=1` and the offline cache on the cluster) → **validation**
(`venv/bin/python -c 'import torch'`) → atomic marker publish. No GitHub access
is required on either server. Dirty local or dirty remote checkouts are refused.

A Git commit at the right SHA is *not* proof that the environment synced, so
`rr run` requires a **deployment marker** (see
`references/configuration.md` → deployment markers) matching
`(host, pool, commit, lock, venv)`. Markers for the target pools are invalidated
*before* the remote is mutated, so a failed sync leaves `rr run` refusing
(`rr.job.not_deployed`) rather than running a half-applied environment.

## State reconciliation

`meta.json` is authoritative but not the only evidence. On every
`rr status`/`rr jobs` (and during `logs --follow`) rr reconciles a *trustworthy*
live state back into metadata, atomically (temp file + rename):

- **Slurm**: a canonical terminal scheduler state (`COMPLETED`, `FAILED`,
  `CANCELLED`, `TIMEOUT`, `OUT_OF_MEMORY`, `NODE_FAIL`, `PREEMPTED`, …) is
  written back with `scheduler_state`, `node`, `completed_at`, `exit_code`,
  `failure_reason`. A cancelled job is never given a fabricated exit code 0.
- **process**: a dead PID with non-terminal metadata becomes `lost` with
  `failure_reason = process_died_without_completion` — never a permanent
  `running live=false`.

Raw States from `sacct` are normalized first (`CANCELLED+`, `CANCELLED by
<uid>`, `OUT_OF_MEMORY`, …) before mapping, so decorations never leak into
metadata.

## Doctor node probing

`rr doctor HOST` is non-invasive by default: it reads the local driver cache
and live `sinfo` node states and **never submits jobs**. Only `rr doctor HOST
--refresh` submits short `rrnode-<token>` probes. Each refresh uses a unique
token (job name + scratch dir), tracks the exact Slurm job IDs it submitted,
and an `EXIT/INT/TERM/HUP` trap best-effort `scancel`s any still-live probe, so
timeouts/Ctrl-C never leave orphans and concurrent doctors do not interfere.

## GPU isolation (process backend)

The process backend has no scheduler; GPU isolation is by explicit pool
(`res-gpu0`/`res-gpu1`) exporting `CUDA_VISIBLE_DEVICES`, exactly like the old
`launch_remote.sh`. The value is captured as provenance.

## Extending

- **New host / cluster / pool / regime**: edit `rr/default_config.toml` or
  `~/.config/rr/rr.toml`. No Skill or code change.
- **New backend type**: implement the interface in `rr/backends/` and register
  it in `rr/backends/__init__.py`.
- **New scheduler attribute**: add it to the sbatch generator and `requested`.

## Machine-readable interface (future MCP)

`rr hosts|doctor|deploy|run|jobs|status|logs|cancel|pull` all accept `--json`
with stable keys and stable error objects (`ok`, `code`, `message`, `details`).
That maps directly to MCP tools (`list_hosts`, `doctor`, `deploy`, `run`,
`list_jobs`, `get_status`, `get_logs`, `cancel`, `pull`) without redesign.