# res-2 offline bootstrap

`res-2` (uestc_hpc `mgt01`) has **no public internet and no DNS**. Nothing may
be installed from the network there. The uv runtime, Python and package cache
are assembled once from `res` (which does have internet and a warm uv cache).

`rr` never installs anything online on `res-2`; it only consumes this offline
runtime and refuses to degrade into online retries.

## What gets placed where (on res-2)

```
~/opt/uv/
  bin/uv                      # copied from res ~/.local/bin/uv
  python/cpython-3.12.*/      # copied from res ~/.local/share/uv/python/
  cache/                      # copied from res ~/.cache/uv  (archive-v0, ...)
```

`rr` sets these explicitly for every remote command (see `hosts.res-2.env` in
`rr/default_config.toml`), so **no `.bashrc` edit is required**:

```
PATH=$HOME/opt/uv/bin:$PATH
UV_CACHE_DIR=$HOME/opt/uv/cache
UV_PYTHON_INSTALL_DIR=$HOME/opt/uv/python
UV_OFFLINE=1
```

## Running the bootstrap

```bash
bash scripts/bootstrap_offline.sh            # idempotent; skips if present
bash scripts/bootstrap_offline.sh --force    # re-sync
bash scripts/bootstrap_offline.sh --verify-only
```

The transfer is **server-to-server**: the laptop runs the bootstrap (and
installs the channel key), but the bulk data travels `res -> res-2` directly.
A dedicated channel key `res:~/.ssh/id_ed25519_res2` is authorised in
`res-2:~/.ssh/authorized_keys` and is the key the res-side transfer uses; the
res-2 host/port/user are resolved from the laptop's `~/.ssh/config`
(`ssh -G res-2`) and passed explicitly to `res`, so the same endpoint is used
for every transfer. The laptop is never in the data path.

The cache is streamed compressed (`tar | ssh res-2 tar`). Expect ~10-25 min at
the observed ~4-5 MB/s cluster link.

> The channel key is not a general-purpose credential: it exists only for this
> bootstrap path, and is only kept because the data path actively uses it.
> Removing it would break `--force` re-syncs.

## Verifying

```bash
rr doctor res-2
```

`offline-env` must be `ok` (cache + python present) and the `env:res2-cu124`
pool environment must show a working torch. If `uv sync` reports
`offline environment incomplete: missing package ...`, the cache snapshot is
incomplete — re-run the bootstrap with `--force` from a host with the packages
cached, then re-deploy.

## Why the cache (not a copied venv)

A copied venv contains absolute paths and hardlinks to a machine-specific
layout; rebuilding with `uv sync --frozen --offline` from the cache is clean and
reproducible. `uv.lock` is the contract: unchanged lock => no expensive work;
changed lock => sync (offline), and a clear error if a package is missing.

## cu118 (all-node) environment — not provisioned

`torch 2.5.1+cu124` cannot run on driver 510 nodes. The `res2-cu118-all` pool is
declared in config but `provisioned = false`.

Provisioning requires a **committed, separate lockfile** for the cu118 stack
(the pool declares `lock_file = "uv.cu118.lock"`); `rr deploy` refuses to build
that pool's venv from the main `uv.lock` (`rr.deploy.lock_missing`), so the
cu124 environment can never be silently reused as cu118. After the lockfile and
`.venv-cu118` exist, `rr doctor res-2` verifies that the venv's actual
`torch.version.cuda` matches the declared `torch_backend` (`cu118` → `11.8`);
only when that check passes may `provisioned` be flipped to `true`. Until then
`rr run --pool res2-cu118-all` refuses with `rr.pool.not_provisioned`.

## Cluster facts (runtime-confirmed 2026-10-02)

- Login node `mgt01`; partition `gpu` -> c01-c08; quota `cpu=16, gres/gpu=2`.
- Non-interactive SSH does not load Slurm; `rr` sources
  `/share/config/zz-hpc-env.sh` before any `sbatch`/`squeue`/`sacct`.
- Drivers: c01/c02 = 510.47.03; c03/c04/c07/c08 = 510.108.03; c05/c06 = 525.85.12.
- `pam_slurm_adopt` blocks direct SSH to compute nodes; all compute goes through
  Slurm.