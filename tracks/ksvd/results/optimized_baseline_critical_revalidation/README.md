# Optimized-Baseline Critical Revalidation Audit

**Verdict.** Exactly **2 new full-training runs** (one per branch, seed 0)
against the optimized compact-v4-hinge baseline. **Official test was never
loaded.**

* **Part A — TOPOLOGY STRONGLY SURVIVES SEED0.** Optimized compact-v2 valid
  **0.166266** vs optimized compact-v4-hinge **0.146420**, Δtopo =
  `MAE(v2) − MAE(v4)` = **+0.019846** ≥ +0.006 (A-STRONG), and *larger* than
  the historical 60-epoch gap +0.014093. Keep topology as core architecture;
  seed1 not purchased.
* **Part B — CORRECTED TOKENIZER CLEAR NO-GO.** Optimized corrected-token v4
  valid **0.150849** vs optimized historical-token v4 **0.146420**, corrected
  still worse by **+0.004429** ≥ +0.004. Longer training relieved the
  historical degradation (+0.008093 → +0.004429) but did **not** rescue it.
  Performance/correctness tradeoff remains; seed1 not purchased.

Frozen optimized protocol (inherited, not re-tuned): Adam, lr 1e-3, wd 1e-5,
batch 128, max_epochs 240, patience 40, no scheduler, L1, best official-valid
checkpoint, single stage. Serial deterministic CPU execution
(`torch.set_num_threads(4)`, fixed global RNG).

## Files

| file | content |
|---|---|
| `protocol_lock.json` | frozen protocol, config SHA-256s, tokenizer fingerprints, seed/compute policy |
| `checkpoint_inventory.json` | reused optimized-v4 reference checkpoints (verified) + new checkpoints |
| `part_a_v2_seed0.json` | Part A seed0 result and Δtopo |
| `part_a_decision.json` | Part A seed0 decision gate |
| `part_b_corrected_seed0.json` | Part B seed0 result and Δcorrect |
| `part_b_decision.json` | Part B seed0 decision gate |
| `final_decision.json` | combined verdict + estimated new-run count |
| `compute_accounting.csv` | wall time, steps, params, threads per new run |
| `curves/` | per-run learning curves (epoch, train MAE, valid MAE, step, best flag) |
| `states/` | selection-checkpoint state dicts for the two new runs |
| `cache/` | tokenizer-pinned train/valid record caches + metadata |

Conditional files (`part_a_v2_seed1.json`, `part_a_two_seed_summary.csv`,
`part_b_corrected_seed1.json`, `part_b_two_seed_summary.csv`) are intentionally
absent: neither seed0 gate required a seed1 replication.

## Part A table (optimized)

| seed | optimized v2 valid | optimized v4 valid | Δtopo = v2 − v4 |
| ---: | -----------------: | -----------------: | --------------: |
| 0 | 0.166266 | 0.146420 | **+0.019846** |

## Part B table (optimized)

| seed | historical-token v4 | corrected-token v4 | Δcorrect = hist − corrected |
| ---: | ------------------: | -----------------: | --------------------------: |
| 0 | 0.146420 | 0.150849 | **−0.004429** |

## Reproduce

```bash
uv run python -m tracks.ksvd.experiments.luyin16.zinc_optimized_baseline_critical_revalidation all
uv run pytest tracks/ksvd/tests/test_optimized_baseline_critical_revalidation.py -q
```

Note: `notes/optimized_baseline_critical_revalidation.md`;
claims: `records/claims/claim-optimized-topology-survives-20260918.yaml`,
`records/claims/claim-optimized-corrected-tokenizer-nogo-20260918.yaml`;
decision: `records/decisions/decision-optimized-baseline-critical-revalidation-20260918.yaml`.
