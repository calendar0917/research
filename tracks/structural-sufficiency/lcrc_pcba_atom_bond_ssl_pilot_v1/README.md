# `lcrc-pcba-atom-bond-ssl-pilot-v1`

Single-seed pilot asking whether a **masked atom/bond reconstruction pretext** on
official-Train structures of `ogbg-molpcba` yields an `atom_encoder` /
`bond_encoder` pair that is a better **initialisation** for the original
no-message-passing LCRC-A than the historical seed-0 scratch init.

* Round: `lcrc-pcba-atom-bond-ssl-pilot-v1`
* Model: `A_SSL_INIT_S0` (original LCRC-A, C1 + 4-head graph-attention pooling)
* Seed: `0`; one pretrain (10,000 updates) + one 40-epoch fine-tune
* Frozen definitions: [`PROTOCOL.md`](PROTOCOL.md)
* Historical reference (not a same-round control): Validation macro AP
  `0.2463560956`, params `287644`, from
  `tracks/structural-sufficiency/lcrc_pcba_no_mp_sprint_v1/runs/phaseC_seed0_A`

This is an **initialisation** experiment: no SSL module is added to the
downstream graph, the inference path and the `287,644` parameters are unchanged,
and the pretext heads are discarded before fine-tuning.

## Status

| stage | state |
|-------|-------|
| code, tests, pre-GPU audit | complete (31 tests, 12/12 audit checks pass on CPU) |
| SSL pretrain (10,000 updates) | see `analysis/pretrain_run.json` |
| 40-epoch fine-tune | see `analysis/supervised_epoch40.json` |
| decision | see `analysis/decision.json` and [`REPORT.md`](REPORT.md) |

## Layout

```text
PROTOCOL.md                 frozen definitions, gate, decision bands, budget
README.md                   this file
code/
  ssl_common.py             frozen constants, hashes, historical parity, helpers
  ssl_data.py               structure-only cache loader, probe split, masks, batches
  ssl_model.py              encoders + temporary heads, static incidence contexts, transfer
  ssl_pretrain.py           the single 10,000-update masked pretraining
  ssl_train.py              the single 40-epoch fine-tune of A_SSL_INIT_S0
  ssl_audit.py              pre-GPU anti-leakage / correctness / no-MP / GPU-smoke audit
  ssl_analyze.py            assembles analysis/*.json from the two runs
  ssl_report.py             renders REPORT.md from analysis/*
tests/
  test_ssl_mask.py          mask rule, probe rule, bands, geometry constants
  test_ssl_contexts.py      static contexts: widths, symmetry, renaming, transfer
  test_ssl_cache.py         real cache: partitions, masked views, seed-0 parity
analysis/                   committed deliverables (see below)
runs/                       git-ignored run artifacts
```

## Frozen design in one paragraph

15 % of atoms and 15 % of bonds are masked per molecule (independent SHA256
streams keyed by molecule id and visit index); a masked atom's full 51-d feature
is zeroed and a masked bond additionally zeroes the 51-d features of both
endpoints. Two CE heads reconstruct `atom_type` (16 classes) and physical
`bond_type` (5 classes) from `[a_v, mean_b(v), mean_a(v)]` and
`[b_e, ctx_u+ctx_v, |ctx_u−ctx_v|]` contexts — one static encode per view, no
message passing. Only the two encoders are exported; they overwrite exactly
`atom_encoder.*` and `bond_encoder.*` of a freshly built historical seed-0
LCRC-A, everything else stays at the historical scratch init, and the frozen
40-epoch recipe (batch 128, AdamW, 1-epoch warmup `3e-5→3e-4`, per-step cosine to
`1e-5`, dropout 0.10, historical batch-order plan `06f43729…`) is run once.

## Reproduce

Local (CPU, no GPU needed):

```bash
uv run pytest -q tracks/structural-sufficiency/lcrc_pcba_atom_bond_ssl_pilot_v1/tests
uv run python tracks/structural-sufficiency/lcrc_pcba_atom_bond_ssl_pilot_v1/code/ssl_audit.py \
  --out tracks/structural-sufficiency/lcrc_pcba_atom_bond_ssl_pilot_v1/analysis/mask_leakage_audit.json \
  --device cpu
```

Remote (A100 pool `res-gpu1`, all through `rr`; commit first):

```bash
rr deploy res --pool res-gpu1
rr run res --pool res-gpu1 --name ssl-audit -- \
  uv run python tracks/structural-sufficiency/lcrc_pcba_atom_bond_ssl_pilot_v1/code/ssl_audit.py \
  --out tracks/structural-sufficiency/lcrc_pcba_atom_bond_ssl_pilot_v1/analysis/mask_leakage_audit.json \
  --device cuda --gpu-smoke-steps 20
rr run res --pool res-gpu1 --name ssl-pretrain -- \
  uv run python tracks/structural-sufficiency/lcrc_pcba_atom_bond_ssl_pilot_v1/code/ssl_pretrain.py \
  --out-dir tracks/structural-sufficiency/lcrc_pcba_atom_bond_ssl_pilot_v1/runs/pretrain \
  --soft-seconds 7200 --hard-seconds 12600
rr run res --pool res-gpu1 --name ssl-finetune -- \
  uv run python tracks/structural-sufficiency/lcrc_pcba_atom_bond_ssl_pilot_v1/code/ssl_train.py \
  --out-dir tracks/structural-sufficiency/lcrc_pcba_atom_bond_ssl_pilot_v1/runs/supervised \
  --ssl-checkpoint tracks/structural-sufficiency/lcrc_pcba_atom_bond_ssl_pilot_v1/runs/pretrain/checkpoint_final.pt \
  --soft-seconds 10800 --hard-seconds 18000
rr pull res --path tracks/structural-sufficiency/lcrc_pcba_atom_bond_ssl_pilot_v1/runs
```

Analysis (after pulling the runs):

```bash
uv run python tracks/structural-sufficiency/lcrc_pcba_atom_bond_ssl_pilot_v1/code/ssl_analyze.py
uv run python tracks/structural-sufficiency/lcrc_pcba_atom_bond_ssl_pilot_v1/code/ssl_report.py
```

## Committed analysis artifacts

| file | contents |
|------|----------|
| `analysis/data_audit.json` | cache/spec/split/probe fingerprints, loader file-access ledger |
| `analysis/mask_leakage_audit.json` | 12-13 anti-leakage / correctness / no-MP / GPU-smoke checks |
| `analysis/pretrain_probe.json` | probe CE vs class-marginal prior CE per eval point + gate |
| `analysis/pretrain_run.json` | pretrain status, update count, encoder change, checkpoint hash |
| `analysis/encoder_transfer_audit.json` | overwritten keys + SHA256, frozen-module equality, params |
| `analysis/supervised_epoch40.json` | epochs 10/20/30/40 AP, Train/Valid BCE and AP gap, buckets |
| `analysis/per_task_ap.csv` | per-task AP, historical AP, delta, positives, stratum |
| `analysis/run_registry.json` | the two runs, status, commits, devices, checkpoints |
| `analysis/gpu_ledger.json` | GPU hours vs the 5 h target / 7 h hard cap |
| `analysis/decision.json` | Delta AP40, pre-registered label, caveats |

## Non-goals

No multi-seed confirmation, no mask-ratio / loss-weight / LR search, no early
stopping or best-epoch selection, no A1/A2/A3, G or V arms, no retraining of the
historical A0, no message passing, and no access to the official Test labels.
