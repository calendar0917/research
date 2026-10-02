# remote-research-runner

Policy + `rr` execution layer for local-first research with remote A100 compute.

```bash
rr doctor res-2                       # non-invasive: no probe jobs by default
rr doctor res-2 --refresh             # opt in to short driver-probe jobs (self-cleaning)
rr deploy res-2                       # writes an atomic deployment marker for the pool(s)
rr run res-2 zinc-seed0 --result tracks/ksvd/runs/<id> -- python -m <module> --seed 0
rr jobs
rr status zinc-seed0
rr logs zinc-seed0 --follow           # exits on terminal state (incl. OOM/TIMEOUT/CANCELLED)
rr pull zinc-seed0
```

Explicit GPU selection on the non-Slurm host uses pools:

```bash
rr run res exp-a --pool res-gpu0 -- python -m <module> --seed 0
rr run res exp-b --pool res-gpu1 -- python -m <module> --seed 0
```

`rr run` requires a **deployment marker** for `(host, pool, commit, lock, venv)`,
so a git checkout at the right SHA is never mistaken for a synced environment.

Add `--json` for machine-readable output. See `SKILL.md` for the workflow/policy
and `references/` for configuration, regimes/provenance, the offline cluster
bootstrap, architecture, troubleshooting, and backward compatibility.

Install / refresh the CLI:

```bash
ln -sf "$(pwd)/bin/rr" ~/.local/bin/rr
rr --version
```

Run the unit tests (no network):

```bash
PYTHONPATH=. pytest rr/tests
```