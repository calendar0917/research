# LCRC PCBA explicit-performance sprint v1

A pre-registered, single-seed screen of three candidate gain sources on top of
the frozen no-message-passing LCRC-A model for `ogbg-molpcba`. All design
decisions are frozen in [`PROTOCOL.md`](PROTOCOL.md) before the formal runs.

| arm | addition | new params | total |
|---|---|---:|---:|
| A0 | none (exact historical A) | 0 | 287,644 |
| A1 | task-conditioned attention residual | 41,344 | 328,988 |
| A2 | 10 physicochemical descriptors + missing flag | 9,088 | 296,732 |
| A3 | explicit rooted 2-hop branch tokens | 266,304 | 553,948 |
| COMBO | the two eligible modules, fixed order | — | — |

Fixed recipe: seed 0, 40 epochs, batch 128, AdamW `3e-4`, 1-epoch warmup then
per-step cosine to `1e-5`, dropout 0.10, clip 5.0, FP32, observed-entry masked
BCE, official Validation macro AP at epochs 10/20/30/40 (epoch 40 is primary).
No Test label is ever read.

## Code map

| file | role |
|---|---|
| `code/ep_common.py` | frozen round constants, arms, recipe, decision rules |
| `code/ep_features.py` | A2 RDKit descriptor cache + A3 branch-token signature/cache |
| `code/ep_model.py` | A1/A2/A3 modules and the composed `EPModel` (A0 = frozen A) |
| `code/ep_data.py` | `augmented_collate`, unified predict/evaluate path |
| `code/ep_train.py` | the single unified 40-epoch trainer |
| `code/ep_checkpoint.py` | complete checkpoint payload / resume restore |
| `code/ep_build_features.py` | offline cache builder CLI (remote CPU) |
| `code/ep_preflight.py` | identity, geometry, gradient, isolation and smoke gates |
| `code/ep_analyze.py` | run registry, GPU ledger, comparison CSV/JSON, decision |
| `code/ep_task_summary.py` | per-task concentration/stratum summary |
| `code/ep_capture_rr.py` | point-in-time `rr` provenance capture |
| `tests/` | signature algebra, descriptor cache, model geometry, data alignment |

## Workflow

```bash
# CPU tests (needs the frozen PCBA cache; ~15 s)
uv run --no-sync pytest -q tracks/structural-sufficiency/lcrc_pcba_explicit_performance_sprint_v1/tests

# Remote CPU: build both label-free caches (run once, writes to ignored cache/)
rr run res ep-prep --pool res-local \
  --result tracks/structural-sufficiency/lcrc_pcba_explicit_performance_sprint_v1/runs/prep \
  -- bash -c 'mkdir -p .../runs/prep && uv run --no-sync --with rdkit==2026.9.1 python .../code/ep_build_features.py --which all --workers 12 --out .../runs/prep/build.json'

# Remote GPU preflight (GPU1 only while GPU0 is externally busy)
rr run res ep-preflight --pool res-gpu1 \
  --result tracks/structural-sufficiency/lcrc_pcba_explicit_performance_sprint_v1/runs/preflight \
  -- uv run --no-sync python .../code/ep_preflight.py --device cuda --out .../runs/preflight/preflight.json

# Formal runs (one at a time on GPU1)
rr run res ep-a0 --pool res-gpu1 --result .../runs/A0 \
  -- uv run --no-sync python .../code/ep_train.py --arm A0 --seed 0 --device cuda \
     --out-dir .../runs/A0 --soft-seconds 20000 --hard-seconds 21600

# Pull, then build the committed analysis
rr pull ep-a0
```

`uv run` here only means "use the deployed environment"; the remote job command
is executed with `uv run --no-sync` by `rr` itself. `--with rdkit` is used only
by the offline feature build and never by a training job.

## Outcome (this round)

A1 −0.000137, A2 −0.003193, A3 +0.001894 versus the historical A0 (0.246356);
no arm reaches the 0.003 screen, so no COMBO was run. A0 was cancelled by
principal instruction at epoch 29/40 and the historical phase-C run is the
reference — see `REPORT.md` and `PROTOCOL.md` §10 for the consequences.

## Analysis artifacts

After all runs are pulled into `runs/`, `code/ep_analyze.py` writes
`analysis/architecture_audit.json`, `feature_audit.json`, `preflight.json`,
`run_registry.json`, `gpu_ledger.json`, `epoch40_comparison.csv`,
`per_task_comparison.csv` and `decision.json`. `REPORT.md` summarises the
pre-registered outcome.
