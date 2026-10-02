# Configuration reference

`rr` loads bundled `rr/default_config.toml`, deep-merged with a user override at
`~/.config/rr/rr.toml` (or `$RR_CONFIG`, or `rr --config PATH`). Adding a host,
pool or regime is a **config-only** change.

## `[ssh]`

```toml
[ssh]
opts = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
        "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3"]
```

`rr` does not manage `ControlMaster`; if you want connection reuse add to
`~/.ssh/config` yourself:

```sshconfig
Host *
    ControlMaster auto
    ControlPersist 10m
    ControlPath ~/.ssh/cm-%C
    ServerAliveInterval 30
    ServerAliveCountMax 3
```

## `[hosts.<name>]`

| key | required | meaning |
|---|---|---|
| `ssh_alias` | yes | name in `~/.ssh/config` |
| `backend` | yes | `process` (setsid) or `slurm` |
| `code` | yes | remote checkout path |
| `runs` | yes | remote run-metadata root (outside the checkout) |
| `default_pool` | no | pool used when `--pool` is omitted |
| `deploy_method` | no | `bundle` (default) |
| `tool_dir` | no | where the runner tools live (default `$HOME/.rr`) |
| `legacy_runs` | no | pre-rr run dir to read read-only (e.g. `$HOME/.research_runs`) |
| `deployments_dir` | no | where per-pool deployment markers live (default `<tool_dir>/deployments`) |

`[hosts.<name>.env]`: `uv_bin`, `uv_cache_dir`, `uv_python_install_dir`,
`offline` (bool), `run_prefix`, plus arbitrary extra `K=V` exports.

`[hosts.<name>.slurm]`: `partition`, `login_setup`, `cpus`, `mem`, `time`.

## `[pools.<name>]`

| key | meaning |
|---|---|
| `host` | owning host (required) |
| `backend` | usually inherited from the host |
| `gpus` | default GPUs per run for this pool |
| `eligible_nodes` | node whitelist -> `#SBATCH --nodelist=...` |
| `min_driver` | minimum NVIDIA driver; enforced at job start and in `doctor` |
| `torch_backend` | `cu124` / `cu118`; selects the venv (`UV_PROJECT_ENVIRONMENT`) |
| `provisioned` | `false` => `rr run` refuses until the env is built |
| `venv` | explicit venv path when it is not `code/.venv` |
| `lock_file` | repo-relative lockfile that pins this pool's environment (default `uv.lock`) |
| `sync_mode` | `frozen` (default) or `locked` — passed to `uv sync --<mode>` |
| `env` | extra environment exported into the run (e.g. `CUDA_VISIBLE_DEVICES`) |

### Explicit GPU selection on a non-Slurm host

The process backend has no scheduler, so isolation is done the same way
`launch_remote.sh` always did it: pin the process with `CUDA_VISIBLE_DEVICES`
via an explicit pool.

```toml
[pools.res-gpu0]
host = "res"
backend = "process"
gpus = 1
torch_backend = "cu124"
env = { CUDA_VISIBLE_DEVICES = "0" }

[pools.res-gpu1]
host = "res"
backend = "process"
gpus = 1
torch_backend = "cu124"
env = { CUDA_VISIBLE_DEVICES = "1" }
```

Two parallel runs then use different GPUs unambiguously
(`rr run res exp0 --pool res-gpu0 ...`, `rr run res exp1 --pool res-gpu1 ...`).
The pinned value is recorded in `meta.json` as `runtime.cuda_visible_devices`.

### Per-pool environments

A pool that needs a *different* dependency set must declare its own committed
lockfile. `rr deploy` refuses to build a pool whose `lock_file` is missing, so a
cu124 environment can never be silently reused as if it were cu118:

```toml
[pools.res2-cu118-all]
host = "res-2"
torch_backend = "cu118"
venv = "/share/home/snsun/rr/research/.venv-cu118"
lock_file = "uv.cu118.lock"
provisioned = false        # flip only after the env exists and doctor passes
```

## Examples

Add a new Slurm pool on an existing host:

```toml
[pools.res2-cu124-all]
host = "res-2"
backend = "slurm"
eligible_nodes = ["c01","c02","c03","c04","c05","c06","c07","c08"]
min_driver = "525.60.13"
torch_backend = "cu124"
provisioned = true
```

After an admin upgrades all drivers, flip `res2-cu124`'s `eligible_nodes` to all
nodes and `res2-cu124` becomes the all-node pool — no Skill change.

Add a new offline host:

```toml
[hosts.res-3]
ssh_alias = "res-3"
backend = "slurm"
code = "/share/home/snsun/rr3/research"
runs = "/share/home/snsun/rr3/runs"

[hosts.res-3.env]
offline = true
uv_bin = "$HOME/opt/uv/bin/uv"
uv_cache_dir = "$HOME/opt/uv/cache"
uv_python_install_dir = "$HOME/opt/uv/python"

[hosts.res-3.slurm]
partition = "gpu"
login_setup = "source /share/config/zz-hpc-env.sh"

[pools.res3-a100]
host = "res-3"
backend = "slurm"
eligible_nodes = ["g01","g02"]
min_driver = "525.60.13"
```

## Deployment markers

`rr deploy` writes an atomic **deployment marker** per provisioned pool at
`<deployments_dir>/<pool>.json` (default `$HOME/.rr/deployments/`). A marker is
written only after Git checkout + `uv sync` + validation all succeed, and binds:

`host`, `pool`, `commit`, `branch`, `venv`, `lock_file`, `lock_hash`,
`pyproject_hash`, `torch_backend`, observed `python`/`torch_version`/`torch_cuda`,
`uv_version`, `deployed_at`, `validated`.

`rr run` refuses with `rr.job.not_deployed` unless a marker exists for the target
pool **and** matches the current commit, lock hash and venv. When a deploy fails,
markers for the target pools are invalidated first, so a half-applied deployment
can never be run. `rr deploy HOST` publishes markers for every pool that shares
the same venv + lockfile (e.g. `res-gpu0`, `res-gpu1`, `res-local`).

## Environment overrides

- `RR_CONFIG` — alternate user config path.
- `rr --config PATH` — explicit config for one invocation.