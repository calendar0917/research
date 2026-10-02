---
name: remote-research-runner
description: Run the calendar0917/research workflow with local-first development and remote A100 compute through the `rr` CLI. Use when ChatGPT needs to inspect the local research repo, deploy a committed revision to an SSH host (`res` = single box, `res-2` = Slurm cluster), launch/monitor/cancel/cancel experiments across process and Slurm backends, record execution-regime provenance (driver/torch/node), and pull results back for local analysis. Also covers preflight (`rr doctor`), offline uv on the air-gapped cluster, pool/regime selection, and recovery after an interrupted agent session.
---

# Remote Research Runner

This Skill is **workflow + policy**. The execution mechanism is the `rr` CLI.
Do not hand-roll SSH, `sbatch`, `nohup`, driver checks, or node selection; call
`rr`. Server/network/Slurm/CUDA specifics live in `rr` config and `references/`,
not here.

```
Agent -> Skill (policy) -> rr (deterministic execution) -> SSH / Git / Slurm / uv
```

Paths like `rr/...`, `scripts/...`, `references/...` are relative to this skill
bundle. The research repository is `cd`-determined: run `rr` from inside the
local checkout.

## Core rules

1. **Local Git is the source of truth.** Edit and commit locally. Remote
   checkouts are compute copies; never treat them as authoritative and never
   edit tracked files remotely.
2. **Formal experiments must be traceable to a commit.** `rr run` refuses a
   dirty local worktree and refuses a remote checkout that is not at the local
   commit. `--allow-dirty` only bypasses the *guard* — it never deploys or runs
   uncommitted changes, because the remote still executes its committed
   revision. `--allow-stale` runs whatever commit is already on the remote
   (use only for infrastructure smoke tests). Neither flag is for results you
   will report.
3. **All remote execution goes through `rr`.** Do not `ssh res-2 sbatch ...`,
   do not SSH to Slurm compute nodes, do not launch long jobs with an ad-hoc
   `nohup`. Literal bypass loses provenance and recovery.
4. **Never use official test data for architecture selection.** The repository's
   `test_policy: terminal` discipline is unchanged by remote execution.
5. **Two different machines are two execution regimes.** A number from `res`
   is not comparable to a number from `res-2`, and a `res-2` number from a
   525-driver node is not automatically comparable to one from a 510-driver
   node. `rr` records the actual regime; you must respect it. See
   `references/execution-regimes.md`.
6. **Do not manage concurrency yourself.** On the cluster, submit all seeds and
   let Slurm queue them; your quota (`cpu<=16, gpu<=2`) is enforced by Slurm.
   `rr` only observes.

## Quickstart

```bash
rr doctor res-2
rr deploy res-2
rr run res-2 zinc-seed0 -- python -m tracks.ksvd.experiments.foo --seed 0
rr jobs
rr status zinc-seed0
rr logs zinc-seed0 --follow
rr pull zinc-seed0
```

Add `--json` to any command for stable machine-readable output.

## Workflow

1. **Inspect locally** — `git status --short`, `uv run research context`, read
   the relevant `TRACK.md`/`STATE.yaml`/protocol before editing.
2. **Edit + test locally** — focused tests first
   (`uv run pytest -q -m "not slow" tracks/ksvd/tests`), then a GPU smoke.
3. **Commit** the change you intend to deploy.
4. **Preflight** — `rr doctor <host>`. It checks SSH, checkout, uv (offline
   cache on the cluster), pool environments + torch-backend consistency, and
   node-driver compatibility, and reports an actionable summary. It is
   **non-invasive by default** (no jobs): driver data comes from a local cache.
   Run `rr doctor <host> --refresh` only when you need a fresh probe — that
   submits short, self-cleaning probe jobs. Fix failures before running.
5. **Deploy** — `rr deploy <host>` ships the committed revision (Git bundle over
   SSH; no GitHub needed), syncs the environment (`uv sync`, offline on the
   cluster) and only then writes an atomic **deployment marker** binding
   host + pool + commit + lock + venv. It refuses a dirty local tree and a
   dirty remote tree. A marker is the proof the environment actually synced;
   `rr run` requires one and will refuse (`rr.job.not_deployed`) after a failed
   deploy, even if the Git commit already moved.
6. **Run** — `rr run <host> <experiment> -- <command>`. `rr` picks the backend
   (process on `res`, Slurm on `res-2`) and a pool, generates the Slurm script,
   and records provenance.
   - Declare result paths so `rr pull` knows what to fetch:
     `rr run res-2 zinc-seed0 --result tracks/ksvd/runs/<id> -- python -m ...`
   - Resource overrides when needed: `--gpus`, `--cpus`, `--mem`, `--time`.
7. **Monitor** — `rr jobs`, `rr status <exp>`, `rr logs <exp> [--follow|--stderr]`.
   Every call is a fresh, stateless query against remote state, so it survives a
   dropped SSH session, a closed terminal, or a restarted agent.
8. **Cancel** — `rr cancel <exp>` (process group kill on `res`, `scancel` on the
   cluster; only your own jobs). If an experiment name matches several runs, rr
   refuses and asks for `--run-id` — it never guesses.
9. **Pull + analyze** — `rr pull <exp> [--light]`. Compare only within the same
   regime and against a matched baseline.

## Host, pool, regime

- **Host** = an access entry point (`res`, `res-2`) with a backend.
- **Pool** = *where* a job may be scheduled (`res2-cu124` -> c05,c06;
  `res2-cu118-all` -> c01-c08). Selected with `--pool` or the host default.
  On the non-Slurm host `res`, pools also carry explicit GPU selection
  (`res-gpu0`, `res-gpu1` -> `CUDA_VISIBLE_DEVICES`), which is how you run two
  parallel jobs on different GPUs.
  `res2-cpu` is a host-level pool with **no driver constraint**: it can land on
  a 510 node where `torch.cuda.is_available()` is silently `False`. Use it for
  CPU work only; GPU work goes to `res2-cu124` (see `references/troubleshooting.md`).
- **Regime** = *what it actually ran on*, recorded per run: hostname, node,
  driver, GPU model/count, torch version + CUDA, python, Git commit, exit code.

Current bootstrap constraint (not permanent): `torch 2.5.1+cu124` needs driver
>= 525.60.13, so only `c05`/`c06` qualify today. The `res2-cu118-all` pool
(c01-c08) exists in config but is **not provisioned**; `rr run --pool
res2-cu118-all` will refuse until the cu118 environment is built. When an admin
upgrades the drivers, or when cu118 is built, the change is **config-only** —
the Skill does not change.

## Recovery after an interruption

`rr` keeps no local daemon or database. Because everything is driven from
remote `meta.json` + logs + Slurm state, after any crash a new session can:

```bash
rr jobs                 # what exists, across hosts, newest first
rr status <exp>         # current state + provenance
rr logs <exp>           # running or completed, including after an agent restart
```

Legacy runs launched by the pre-`rr` scripts still appear in `rr jobs` marked
`[legacy]`, and their logs are readable with `rr logs <tag>`.

## Failure handling

- A failed `rr` command prints a `reason`/`details` block and a stable `code`
  (e.g. `rr.job.stale_remote`, `rr.deploy.failed`, `rr.pool.not_provisioned`).
  Read it — it is written to be actionable. With `--json` the same fields are
  returned.
- `rr.job.stale_remote`: run `rr deploy <host>` first.
- `rr.job.not_deployed`: the pool has no valid deployment marker for this commit
  (e.g. the last `rr deploy` failed while syncing). Run
  `rr deploy <host> --pool <pool>` and fix whatever it reports.
- `rr.job.ambiguous`: an experiment name matches several runs; pass `--run-id`.
- `rr.job.dirty_local` / `rr.deploy.dirty_local`: commit, or use `--allow-dirty`
  only for smoke tests (it does **not** deploy or run uncommitted edits).
- `rr.deploy.lock_missing`: the pool needs a committed lockfile that is absent
  (the cu118 case). Add the lockfile; never build it from the main `uv.lock`.
- `rr.deploy.failed` with `offline environment incomplete`: the cluster uv cache
  is missing packages; see `references/res2-bootstrap.md`. Never retry online —
  the cluster has no internet.
- Slurm job `pending` is normal (quota/queue), not an error. Inspect
  `rr status <exp>` for the scheduler reason.
- Job failed with `driver_too_old`: it landed on a node older than the pool
  requires; check `rr doctor res-2` and the pool definition.
- `res` GPUs currently report an NVML/device-handle fault (external). CPU jobs
  and deploy still work; GPU runs on `res` may fail until it is serviced.
- `CUDA driver version is insufficient` is the same class of problem: wrong
  pool for the allocated node.

## Bundled assets

- `bin/rr` — the CLI (also symlinked to `~/.local/bin/rr`).
- `rr/` — implementation: `config.py`, `services.py`, `backends/{process,slurm}.py`,
  `runner/{rr_runner.sh,rr_meta.py}`, `default_config.toml`, `tests/`.
- `scripts/bootstrap_offline.sh` — one-time offline uv runtime transfer to `res-2`.
- `scripts/*.sh` — the legacy wrappers (kept for backward compatibility; `rr`
  supersedes them).
- `references/execution-regimes.md` — regime/provenance and comparison rules.
- `references/res2-bootstrap.md` — offline cluster bootstrap + env.
- `references/architecture.md` — layering and extension points.
- `references/troubleshooting.md` — symptoms -> action.
- `references/servers-survey.md` — raw server facts (historical; `rr doctor`
  is the live source).
- `references/gpu-migration.md` — converting CPU code to explicit devices.