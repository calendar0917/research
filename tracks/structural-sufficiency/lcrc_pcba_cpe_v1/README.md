# LCRC PCBA Cross-atom Predictive Evidence (CPE) v1 — how to run

Round id `lcrc-pcba-cpe-v1`. Read [`PROTOCOL.md`](PROTOCOL.md) first: it is
frozen before the first formal step. Results live in [`REPORT.md`](REPORT.md).

Two seed-0 from-scratch 40-epoch runs on top of the historical **W2** backbone
(994,324 parameters, epoch-40 Validation AP 0.2618669181), which is **not**
retrained:

| arm | third statistic | new params | total params |
|---|---|---:|---:|
| `P` | **cross-atom** `(sum_A·sum_B − sum_AB) / (n(n−1))` | 33,408 | **1,027,732** |
| `S` | **same-atom** `sum_AB / n` | 33,408 | **1,027,732** |

Both arms share every parameter name/shape, the same bitwise-seeded initial
state, the same batch order (`06f43729…`) and the inherited recipe. The only
difference is `PredictiveEvidence.pair_mode` (`cross` vs `self`), a
non-parameter attribute. No message passing, no cross-atom attention, no atom
re-encoding, one dropout call. Official Test is never touched.

## Code map

| file | role |
|---|---|
| `code/cpe_common.py` | frozen constants: W2 identity, arm metadata, exact parameter targets, recipe, decision thresholds, the fixed 256 diagnostic Validation rows, IO helpers |
| `code/cpe_model.py` | `pair_statistics` (the pure second-order reduction), `PredictiveEvidence`, `CPEModel`, the W2-preserving builder, evidence diagnostics, the diagnostic `alpha_pair=0` ablation helper |
| `code/cpe_train.py` | the 40-epoch loop, per-epoch Validation, per-epoch evidence diagnostics, resumable + primary checkpoints, `run.json` |
| `code/cpe_audit.py` | `gpu-preflight` / `gpu-sampling` / `init` / `architecture` / `smoke` / `mechanism` audit modes |
| `code/cpe_compare.py` | epoch-40 records, per-task deltas, training parity, decision, GPU ledger |

Inherited **read-only**: `bs_common`, `bs_model` (the historical W2 builder,
`state_cpu`, `make_optimizer`, `permute_atoms`), `tu_model.TULCRC`,
`nmp_model.AttentionPoolReadout`, `sg_common` recipe, `sg_data`
data/plan/official evaluator, `pc_metrics`, `pc_model`, `pf_pipeline`.

## Commands

```bash
# 0) local tests (CPU, ~11 s)
uv run --no-sync pytest -q tracks/structural-sufficiency/lcrc_pcba_cpe_v1/tests

# 1) preflight + pre-training audit + 1-epoch infrastructure smoke on the A100
rr doctor res
rr deploy res
rr run res cpe-audit --pool res-gpu1 \
  --result tracks/structural-sufficiency/lcrc_pcba_cpe_v1/runs/audit -- \
  bash -c "uv run --no-sync python tracks/structural-sufficiency/lcrc_pcba_cpe_v1/code/cpe_audit.py \
  --modes gpu-preflight,gpu-sampling,init,architecture,smoke \
  --out-dir tracks/structural-sufficiency/lcrc_pcba_cpe_v1/runs/audit"

# 2) formal runs (see PROTOCOL.md §11 for the GPU0 co-tenancy rule)
rr run res cpe-p --pool res-gpu1 \
  --result tracks/structural-sufficiency/lcrc_pcba_cpe_v1/runs/P -- \
  bash -c "uv run --no-sync python tracks/structural-sufficiency/lcrc_pcba_cpe_v1/code/cpe_train.py \
  --model P --seed 0 --device cuda --eval-every 1 \
  --out-dir tracks/structural-sufficiency/lcrc_pcba_cpe_v1/runs/P \
  --soft-seconds 14400 --hard-seconds 18000"

rr run res cpe-s --pool res-gpu1 \
  --result tracks/structural-sufficiency/lcrc_pcba_cpe_v1/runs/S -- \
  bash -c "uv run --no-sync python tracks/structural-sufficiency/lcrc_pcba_cpe_v1/code/cpe_train.py \
  --model S --seed 0 --device cuda --eval-every 1 \
  --out-dir tracks/structural-sufficiency/lcrc_pcba_cpe_v1/runs/S \
  --soft-seconds 14400 --hard-seconds 18000"

# 3) pull, post-training mechanism audit, comparison
rr pull cpe-p ; rr pull cpe-s ; rr pull cpe-audit
cp tracks/structural-sufficiency/lcrc_pcba_cpe_v1/runs/audit/{gpu_preflight,gpu_sampling,initialization_parity,architecture_audit,smoke}.json \
   tracks/structural-sufficiency/lcrc_pcba_cpe_v1/analysis/
cp tracks/structural-sufficiency/lcrc_pcba_cpe_v1/runs/audit/gpu_occupancy_samples.csv \
   tracks/structural-sufficiency/lcrc_pcba_cpe_v1/analysis/

uv run --no-sync python tracks/structural-sufficiency/lcrc_pcba_cpe_v1/code/cpe_audit.py \
  --modes mechanism --out-dir tracks/structural-sufficiency/lcrc_pcba_cpe_v1/runs/mechanism

rr jobs --json > /tmp/rr_jobs_all.json   # keep cpe-* records only
uv run --no-sync python - <<'PY'
import json
d = json.load(open('/tmp/rr_jobs_all.json'))
json.dump({'ok': d['ok'], 'jobs': [j for j in d['jobs'] if j['experiment'].startswith('cpe-')],
           'filtered': 'cpe-* only'},
          open('tracks/structural-sufficiency/lcrc_pcba_cpe_v1/analysis/rr_jobs.json', 'w'), indent=2)
PY

uv run --no-sync python tracks/structural-sufficiency/lcrc_pcba_cpe_v1/code/cpe_compare.py \
  --rr-jobs tracks/structural-sufficiency/lcrc_pcba_cpe_v1/analysis/rr_jobs.json
```

`runs/` is git-ignored. Final checkpoints, resumable state, curves, per-task AP,
evidence diagnostics and the order plan stay there; the report records their
SHA-256.

## Artifacts

```
analysis/gpu_preflight.json            host/driver/torch/pool facts before the audit
analysis/gpu_sampling.json             six fresh occupancy samples (GPU0 co-tenancy decision)
analysis/gpu_occupancy_samples.csv     raw nvidia-smi samples
analysis/initialization_parity.json    W2 init identity, params, initial logits/AP, gradients, resume, Test isolation
analysis/architecture_audit.json       the 34 frozen architecture checks
analysis/smoke.json                    1-epoch infrastructure smoke (not a result)
analysis/paired_training_audit.json    same plan/recipe/init/RNG, epoch-0 parity, arm pairing
analysis/p_epoch40.json                P epoch-40 record (AP/BCE/curve/best/checkpoint/resources)
analysis/s_epoch40.json                S epoch-40 record
analysis/paired_comparison.json        D_base / D_mech / per-task strata / cost
analysis/per_task_deltas.csv           128 rows, 127 scorable, three deltas
analysis/evidence_mechanism_audit.json alpha norms, residual magnitudes, CROSS/SELF on the fixed 256 molecules, alpha_pair=0 ablation
analysis/gpu_ledger.json               every cpe-* job, wall clock, budget
analysis/rr_jobs.json                  raw cpe-* job records (filtered)
analysis/decision.json                 pre-registered label + sub-flags
```
