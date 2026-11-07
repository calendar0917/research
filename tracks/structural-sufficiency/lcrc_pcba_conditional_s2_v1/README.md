# LCRC PCBA Conditional S2 v1 — how to run this round

Round id `lcrc-pcba-conditional-s2-v1`. Read [`PROTOCOL.md`](PROTOCOL.md)
first: it is frozen before the first formal step.

Two seed-0 from-scratch 40-epoch runs on top of the historical W2 backbone:

| arm | what is added | new params | total params |
|---|---|---:|---:|
| `C2` | Conditional S2: `(E_c(path_cat) ⊙ E_e(path_end63)) V` added to the scalar path gate | 10,752 | 1,005,076 |
| `F` | near-matched local residual FFN `128→40→128` after `LocalContent`, before dropout | 10,664 | 1,004,988 |

Both start from the same untrained W2 seed-0 state, use the same batch order
(`06f43729…`) and the same inherited recipe. W2 is **not** retrained. Official
Test is never touched.

## Code map

| file | role |
|---|---|
| `code/cs2_common.py` | frozen constants: W2 identity, arm metadata, exact parameter targets, recipe, decision thresholds, IO helpers |
| `code/cs2_model.py` | `CS2LocalContent`/`CS2Model` (vector gate) and `LocalResidualFFN`/`FModel`, the W2-preserving builder, gate/FFN diagnostics |
| `code/cs2_train.py` | the 40-epoch loop, per-epoch Validation, new-module diagnostics, resumable + primary checkpoints, `run.json` |
| `code/cs2_audit.py` | `gpu-preflight` / `gpu-sampling` / `init` / `architecture` audit modes |
| `code/cs2_compare.py` | epoch-40 records, per-task deltas, training parity, decision, GPU ledger |

Inherited **read-only**: `bs_common`, `bs_model`, `bs_train.train_step`,
`tu_model.TULCRC`/`TULocalContent`, `nmp_model.AttentionPoolReadout`,
`sg_common` recipe, `sg_data` data/plan/official evaluator, `pc_metrics`,
`pc_model`.

## Commands

```bash
# 0) local tests
uv run --no-sync pytest -q tracks/structural-sufficiency/lcrc_pcba_conditional_s2_v1/tests

# 1) preflight + pre-training audit on the A100 (audit includes 6x10 s GPU sampling)
rr doctor res
rr deploy res
rr run res cs2-audit --pool res-gpu1 --result tracks/structural-sufficiency/lcrc_pcba_conditional_s2_v1/runs/audit -- \
  bash -c "uv run --no-sync python tracks/structural-sufficiency/lcrc_pcba_conditional_s2_v1/code/cs2_audit.py \
  --modes gpu-preflight,gpu-sampling,init,architecture \
  --out-dir tracks/structural-sufficiency/lcrc_pcba_conditional_s2_v1/runs/audit"

# 2) formal runs. Parallel on GPU0/GPU1 only if the fresh sample shows GPU0 is safe;
#    otherwise both serially on res-gpu1.
rr run res cs2-c2 --pool res-gpu1 --result tracks/structural-sufficiency/lcrc_pcba_conditional_s2_v1/runs/C2 -- \
  bash -c "uv run --no-sync python tracks/structural-sufficiency/lcrc_pcba_conditional_s2_v1/code/cs2_train.py \
  --model C2 --seed 0 --device cuda --eval-every 1 \
  --out-dir tracks/structural-sufficiency/lcrc_pcba_conditional_s2_v1/runs/C2 \
  --soft-seconds 14400 --hard-seconds 18000"

rr run res cs2-f --pool res-gpu0 --result tracks/structural-sufficiency/lcrc_pcba_conditional_s2_v1/runs/F -- \
  bash -c "uv run --no-sync python tracks/structural-sufficiency/lcrc_pcba_conditional_s2_v1/code/cs2_train.py \
  --model F --seed 0 --device cuda --eval-every 1 \
  --out-dir tracks/structural-sufficiency/lcrc_pcba_conditional_s2_v1/runs/F \
  --soft-seconds 14400 --hard-seconds 18000"

# 3) pull, copy audit JSONs into analysis/, then compare
rr pull cs2-c2 ; rr pull cs2-f ; rr pull cs2-audit
rr jobs --json > tracks/structural-sufficiency/lcrc_pcba_conditional_s2_v1/analysis/rr_jobs.json
cp tracks/structural-sufficiency/lcrc_pcba_conditional_s2_v1/runs/audit/{initialization_audit,architecture_audit}.json \
   tracks/structural-sufficiency/lcrc_pcba_conditional_s2_v1/analysis/
uv run --no-sync python tracks/structural-sufficiency/lcrc_pcba_conditional_s2_v1/code/cs2_compare.py \
  --rr-jobs tracks/structural-sufficiency/lcrc_pcba_conditional_s2_v1/analysis/rr_jobs.json
```

`runs/` is git-ignored. Final checkpoints, resumable state, curves, per-task AP
and the order plan stay there; the report records their SHA-256.

## Artifacts

```
analysis/initialization_audit.json   W2 init identity, params, initial logits, gradients, resume, Test isolation
analysis/architecture_audit.json     path ranges, gate shape, call counts, no propagation, mask, renumbering
analysis/training_parity.json        same plan/recipe/init/RNG, epoch-0 parity
analysis/c2_epoch40.json             C2 epoch-40 record (AP/BCE/curve/best/checkpoint/resources)
analysis/ffn_epoch40.json            F epoch-40 record
analysis/paired_comparison.json      D_C2 / D_F / D_CF + per-task strata
analysis/per_task_deltas.csv         128 rows, 127 scorable
analysis/gpu_ledger.json             every cs2-* job, wall clock, budget
analysis/decision.json               pre-registered label + sub-flags
```
