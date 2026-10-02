# Migration & backward compatibility

The refactor was done incrementally and the old `res` workflow was not deleted.

## What changed

| old | new |
|---|---|
| `scripts/launch_remote.sh` + `wait_remote.sh` + `tail_remote.sh` | `rr run` / `rr status` / `rr logs` (same setsid keep-alive underneath) |
| `scripts/status.sh` + `scripts/preflight.sh` | `rr jobs` / `rr doctor` |
| `scripts/run_remote.sh` | `rr run <host> <exp> -- ...` |
| `scripts/pull_results.sh` | `rr pull <exp>` |
| `scripts/deploy.sh` (git push + fetch) | `rr deploy` (git bundle over SSH; no GitHub) |
| ad-hoc `ssh res-2 sbatch ...` | `rr run res-2 <exp> -- ...` |

## What was preserved

- **All legacy scripts remain** in `scripts/` unchanged. They still work.
- The proven `setsid`/`nohup` keep-alive, detached-stdin, and `.exit`/`.pid`
  semantics are reused by the process backend (with completion moved into
  `meta.json`, which is authoritative; the pid is only for liveness).
- **Legacy runs are visible**: `rr jobs` reads `~/.research_runs/*.meta` on
  `res` read-only and marks them `[legacy]`. `rr status <tag>` and
  `rr logs <tag>` work for them. `rr cancel`/`rr pull` on a legacy run return a
  clear `rr.job.legacy_not_managed` error pointing at the old scripts.
- The research control plane (`uv run research ...`, `runs/`, `records/`) is
  untouched. `rr` only transports/executes.

## Behaviour differences to expect

- `rr run` refuses a **dirty tracked local worktree** and a **remote checkout
  not at the local commit**. The old `launch_remote.sh` did not. For deliberate
  smoke tests use `--allow-dirty` and/or `--allow-stale`.
- New runs live in `<runs>/<run-id>/` under `hosts.<h>.runs`, not
  `~/.research_runs`. The two layouts coexist.
- `rr deploy` uses a Git bundle (works without GitHub) instead of
  `git push`+`fetch`. Same guarantee: local commit == remote commit.

## Rollback

Nothing destructive was performed. To go back to the old flow, use the existing
`scripts/*.sh` directly; `rr` does not remove or rewrite them. The only changes
made on servers are additive: the `~/opt/uv` offline runtime and `~/.rr` tools
on `res-2`, and a dedicated `res -> res-2` channel key (removable from
`res-2:~/.ssh/authorized_keys`).