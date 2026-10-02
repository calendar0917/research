# Execution regimes & provenance

A **regime** is the tuple of facts that determines whether two numbers are
comparable. A **pool** is only a scheduler hint about *where* a job may land;
the regime is what it *actually* ran on and is recorded by `rr` after the fact.

## Provenance recorded per run

`rr` writes `runs/<run-id>/meta.json` on the remote. It contains (among others):

| field | meaning |
|---|---|
| `experiment`, `run_id` | identity (unique run id; never silently reused) |
| `host`, `backend`, `pool` | access + scheduler + scheduler hint |
| `git_commit`, `git_branch`, `git_dirty`, `git_diff_hash` | source revision |
| `command` | exact command string |
| `requested` | gpus/cpus/mem/time/eligible_nodes |
| `slurm_job_id` | Slurm job id (cluster) |
| `runtime.hostname`, `runtime.node` | login host and allocated node |
| `runtime.driver_version` | NVIDIA driver on the node that ran |
| `runtime.gpus.{count,names}` | actual GPU model/count |
| `runtime.torch_version`, `runtime.torch_cuda`, `runtime.cuda_available` | torch stack |
| `runtime.python_version` | interpreter |
| `runtime.started_at`, `runtime.completed_at`, `runtime.exit_code` | timing/outcome |
| `runtime.cuda_visible_devices` | explicit GPU pin (process pools `res-gpu0/1`) |
| `runtime.scheduler_state`, `runtime.failure_reason` | reconciled terminal cause (OOM/TIMEOUT/NODE_FAIL/CANCELLED/…) |
| `runtime.slurm_*` | job id / nodelist / partition / gres / cpus |

`rr status <exp>` and `rr jobs --json` surface these. A result without this
provenance is not reportable.

## Known regimes (as observed 2026-10-02)

| regime | host | backend | nodes | driver | torch | status |
|---|---|---|---|---|---|---|
| `res-cu124` | `res` | process | a100-2 (2x A100-SXM4-40GB) | 550.163.01 | 2.5.1+cu124 | provisioned (GPU HW fault, external) |
| `res2-cu124-525` | `res-2` | slurm | c05, c06 | 525.85.12 | 2.5.1+cu124 | provisioned (pool `res2-cu124`) |
| `res2-cu118-510` | `res-2` | slurm | c01-c08 (510 on c01-c04,c07,c08; 525 on c05,c06) | >= 510.108.03 | +cu118 | **not provisioned** (pool `res2-cu118-all`) |

Notes:
- The cluster is **heterogeneous**: c05/c06 run driver 525, the other six run
  510.x. `gres.conf` comments in `/share` were found stale; `rr doctor` probes
  drivers at runtime instead of trusting static config.
- Even inside `res2-cu118-all`, a run may land on a 510 node or a 525 node.
  These are not automatically interchangeable for fine-grained claims; record
  the node/driver and compare like with like.

## Per-pool environments (how cu118 stays honest)

One environment cannot serve two torch backends. Each pool therefore declares
its environment identity explicitly:

- `torch_backend` — expected stack (`cu124`, `cu118`, …); selects the venv
  (`UV_PROJECT_ENVIRONMENT`) and is checked against the *actual*
  `torch.version.cuda` by `rr doctor`.
- `lock_file` — the committed lockfile that pins the environment (default
  `uv.lock`). `rr deploy` **refuses** a pool whose declared lockfile is absent
  (`rr.deploy.lock_missing`), so the main `uv.lock` can never be silently used
  to build a cu118 environment.
- `venv` — the per-stack venv path (e.g. `.venv-cu118`).

`rr doctor` performs the backend consistency check: for each pool it compares
`torch.version.cuda` with the CUDA version implied by `torch_backend` and marks
a mismatch as a failure for a provisioned pool. **`res2-cu118-all` stays
`provisioned = false`** and is refused by `rr run`
(`rr.pool.not_provisioned`) until a committed cu118 lockfile + venv exist and
that check passes; only then may `provisioned` be flipped to `true`.

## Comparison rules

1. Only compare runs with the same `protocol_id` / dataset+split fingerprint
   (the research control plane already enforces `INCOMPARABLE` otherwise).
2. Only compare within one regime unless you explicitly document the regime
   change as a separate infrastructure commit.
3. Small deltas require a matched baseline in the same regime, paired seeds,
   and a fixed Top-5 soup — not a historical checkpoint from another regime.
4. Do not mix `res` and `res-2` numbers in one table.

## `rr doctor` vs the regime record

`rr doctor <host>` gives the *expected/available* picture (pool nodes and their
current drivers from the local cache, refreshable with `--refresh`; live node
states from `sinfo`). It submits **no jobs** unless `--refresh` is passed. The
per-run `meta.json` gives the *actual* regime. Trust the per-run record for
results; use `doctor` for planning.