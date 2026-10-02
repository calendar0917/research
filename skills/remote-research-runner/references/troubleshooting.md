# Troubleshooting

`rr` errors carry a stable `code`, a `message`, and `details`. Use `--json` to
consume them programmatically.

| symptom / code | meaning | action |
|---|---|---|
| `rr.host.unknown` | host not in config | `rr hosts`; add it to config (see `configuration.md`) |
| `rr.remote.timeout` / `rr.remote.failed` | SSH to the alias failed | check `ssh <alias>`; topology belongs in `~/.ssh/config` |
| `rr.pool.unknown` / `rr.pool.host_mismatch` | pool name/host wrong | `rr hosts`; pick a pool for that host |
| `rr.pool.not_provisioned` | pool declared but env missing (e.g. cu118) | build the env, or use a provisioned pool |
| `rr.deploy.dirty_local` | local tracked files modified | commit (smoke only: `--allow-dirty`, which does **not** deploy uncommitted edits) |
| `rr.deploy.failed` | git moved but `uv sync`/validation failed | marker invalidated; re-run `rr deploy`; no run will be allowed in between |
| `rr.deploy.lock_missing` | pool declares a lockfile that is not committed | add the committed lockfile (e.g. `uv.cu118.lock`) or fix `lock_file` |
| `rr.deploy.validation_missing` | env did not import after sync | inspect the deploy output; the venv is not usable |
| `rr.deploy.bundle_failed` | `git bundle` failed | check local git state |
| `rr.job.not_deployed` | no valid deployment marker for (host,pool,commit,env) | `rr deploy <host> --pool <pool>` |
| `rr.job.ambiguous` | an experiment name matched several runs | pass `--run-id` |
| `offline environment incomplete` | cluster uv cache missing packages | see `res2-bootstrap.md`; never retry online |
| `rr.job.dirty_local` | same guard on `run` | commit, or `--allow-dirty` for smoke (uncommitted edits are not run) |
| `rr.job.stale_remote` | remote checkout != local commit | `rr deploy <host>` |
| `rr.job.submit_failed` | `sbatch` returned no job id | check `rr doctor res-2`; partition/account |
| `rr.job.exited_immediately` | job died right after launch | read the `stderr` in the error and `rr logs <exp> --stderr` |
| `rr.job.cancel_refused` | job not queued for your user | confirm the job id / ownership |
| `rr.job.not_running` | process run already finished | `rr status <exp>` |
| `rr.job.not_found` | no run matches the ident | `rr jobs` |
| `rr.job.legacy_not_managed` | pre-rr run | use the old `scripts/tail_remote.sh`/`status.sh` |
| `rr.job.bad_experiment` | experiment name has illegal chars | match `[A-Za-z0-9][A-Za-z0-9._-]*` |

## Cluster-specific

- **Job `pending` forever**: normal queueing under quota `cpu<=16, gpu<=2`.
  `rr status <exp>` shows the scheduler reason. Do not build a local scheduler.
- **`driver_too_old` / job exit 78**: the job landed on a node older than the
  pool requires. Confirm with `rr doctor res-2 --refresh`.
- **`CUDA driver version is insufficient for CUDA runtime version`**: same class
  — cu124 code on a 510 node. Use the cu124 pool (c05/c06) until cu118 exists.
- **GPU work on an unconstrained pool silently gets no GPU**: `res2-cpu` declares
  `gpus = 0` but has no `eligible_nodes` and no `min_driver`, so a CPU-requested
  job can land on a 510-driver node (c01-c04, c07, c08). There the cu124 env
  imports fine but `torch.cuda.is_available()` returns `False` — no error, no
  `driver_too_old`, just CPU execution. Requesting GPUs (`--gpus 1`) makes Slurm
  schedule onto a GPU node, but the pool does not constrain the driver; for real
  GPU work always use `res2-cu124`. Verified 2026-10-02: `res2-cpu` smoke landed
  on c02 (driver 510.47.03, `cuda_avail False`), `res2-cu124` on c05
  (driver 525.85.12, A100, `cuda_avail True`). If you need GPUs on a
  non-cu124 node, build the cu118 env first — do not treat a `res2-cpu` run as
  GPU provenance.
- **`uv` not found over ad-hoc ssh**: non-interactive shells miss `~/.local/bin`
  (res) or `~/opt/uv/bin` (res-2). `rr` sets these itself; only hand-run ssh
  needs the export.
- **`Unable to determine the device handle for GPU...` on `res`**: current
  external NVML/hardware fault on that box. CPU jobs and deploy are unaffected;
  GPU runs on `res` may fail until it is serviced. Use `res-2` for GPU work.
- **`rr doctor` shows `node-drivers ... unverified`**: expected by default (no
  jobs submitted). Run `rr doctor res-2 --refresh` to probe; the probe jobs are
  cleaned up automatically, including on timeout/Ctrl-C.

## Reconciliation / stuck states

`rr status`/`rr jobs` reconcile a trustworthy terminal scheduler state (OOM,
TIMEOUT, NODE_FAIL, CANCELLED, …) back into `meta.json`. If you still see a
non-terminal state, the scheduler has not yet reported one (e.g. still queued);
`rr jobs` shows the live scheduler state. A process run whose PID died without
recording completion becomes `lost` (`failure_reason =
process_died_without_completion`) rather than staying `running` forever.

## Recovery

`rr` has no local state. After any interruption:

```bash
rr jobs
rr status <exp>
rr logs <exp>
```

If a long run died (not just the SSH connection), relaunch the same experiment
name (a new run id is allocated; history is preserved) and make stages
idempotent/resumable so the relaunch skips completed work.