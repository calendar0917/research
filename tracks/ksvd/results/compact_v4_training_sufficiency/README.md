# Compact-v4 Training Sufficiency & Protocol Calibration

**Verdict: HISTORICAL UNDERTRAINING CONFIRMED.** Extending the *unchanged*
canonical optimisation regime (Adam, lr 1e-3, batch 128, wd 1e-5) from 60 to
240 epochs materially improves compact-v4-hinge. Batch 512 and weight decay 0
do **not** provide an independent gain; the canonical late-readout
*continuation* signal was a horizon / early-stopping-rule effect, not an
optimisation-regime switch. The new baseline is **longer single-stage
compact-v4-hinge training**.

Leakage safety: official `test` was **never loaded** during any search stage; it
was accessed once, only after `final_training_protocol_lock.json` existed and
`final_decision.json` authorised it. No test-based selection/tuning/refit.

## Historical protocol (reconstructed)

Adam; lr 1e-3; wd 1e-5; batch 128; max 60 epochs; patience 12; no scheduler;
L1; best-valid checkpoint; 99,613 parameters. Seed-0 valid **0.170066** (best
epoch 53); seed-1 valid **0.163167**. A0 re-ran the exact protocol and
reproduced the historical valid curve **bit-exactly** (max diff 0.0).

## Stage 1A — horizon ladder (seed 0)

| protocol | max ep | patience | best epoch | train@best | valid@best |
|---|---:|---:|---:|---:|---:|
| A0 historical | 60 | 12 | 53 | 0.150202 | 0.170066 |
| A1 moderate | 120 | 24 | 74 | 0.132462 | 0.161252 |
| A2 long | 240 | 40 | 169 | 0.084949 | **0.146420** |

`A0 − A2 = +0.02365`, best epoch 169 > 60 ⇒ H1 undertraining; A1 already
matches the continuation anchor (0.160976) and A2 goes far below it.

## Stage 1B — batch × weight-decay (seed 0, 240 epochs)

| protocol | batch | wd | best valid | best epoch |
|---|---:|---:|---:|---:|
| B0 | 128 | 1e-5 | **0.146420** | 169 |
| B1 | 512 | 1e-5 | 0.153920 | 204 |
| B2 | 128 | 0 | 0.159668 | 79 |
| B3 | 512 | 0 | 0.151145 | 201 |

`B0→B1 = −0.00750`, `B0→B2 = −0.01325`, `B0→B3 = −0.00473`: larger batch and
wd removal both hurt; the continuation-regime change gives no independent gain.

## Stage 2 — learning rate (seed 0, winning regime B0)

| protocol | lr | best valid | best epoch |
|---|---:|---:|---:|
| S_lr1e-3 | 1e-3 | **0.146420** | 169 |
| S_lr3e-4 | 3e-4 | 0.146615 | 232 (horizon-pinned) |

lr 1e-3 retained; no scheduler needed.

## Candidate P* and seed replication

P* = **A2/B0**: Adam, lr 1e-3, batch 128, wd 1e-5, max_epochs 240,
patience 40, no scheduler, L1.

| seed | historical valid | optimized valid | valid gain | optimized test | historical test | test gain |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.170066 | 0.146420 | +0.023645 | 0.126512 | 0.133901 | +0.007389 |
| 1 | 0.163167 | 0.149332 | +0.013834 | 0.127506 | 0.132006 | +0.004500 |
| 2 | 0.174149 | 0.151455 | +0.022694 | 0.127327 | 0.144614 | +0.017287 |
| 3 | 0.170421 | 0.141595 | +0.028826 | 0.119371 | 0.137019 | +0.017648 |

- 4-seed valid: historical **0.169451 ± 0.003965** → optimized
  **0.147201 ± 0.003697**, paired mean gain **+0.022250** (4/4 positive).
- Frozen test: historical **0.136885** → optimized **0.125179 ± 0.003374**,
  paired mean gain **+0.011706** (4/4 positive).
- Honest note: the valid-selected gain (+0.022) is ≈2× the test gain (+0.0117);
  part of the valid gain is validation-selection bias, but a real, substantial
  official-test improvement remains.

Representation drift (seed 0, 1000 probes): `normalized_L2_drift = 0.314`,
`cosine = 0.950` — the optimized latent manifold has moved materially; prior
NO-GO diagnostics describe the historical, not-yet-optimised manifold.

## Files

| file | content |
|---|---|
| `historical_protocol.json` | reconstructed canonical protocol + historical runs |
| `historical_learning_curve.csv` | stored historical valid curve |
| `search_preregistration.json` | pre-registered stages / thresholds |
| `stage1a_horizon_results.csv` | horizon ladder |
| `stage1b_regime_results.csv` | batch × wd 2×2 |
| `stage2_lr_results.csv` | learning-rate calibration |
| `candidate_protocol.json` | frozen seed-0 candidate P* |
| `candidate_seed0_curve.csv` | candidate train/valid curve |
| `seed1_replication.json` | seed-1 replication + gates |
| `seed2_results.json`, `seed3_results.json` | extra confirmation seeds |
| `valid_summary.csv` | paired historical-vs-optimized valid |
| `final_training_protocol_lock.json` | frozen protocol lock (pre-test) |
| `final_decision.json` | decision + verdict |
| `test_results.csv`, `test_summary.json` | frozen one-shot official test |
| `representation_drift.json` | descriptive R drift |
| `compute_accounting.csv` | wall-clock / epochs / optimizer steps |
| `curves/`, `figures/`, `runs/`, `states/` | per-run curves, 4 figures, summaries, frozen checkpoints |

## Reproduce

```bash
uv run python -m tracks.ksvd.experiments.luyin16.zinc_compact_v4_training_sufficiency <stage>
# stages: stage0 lock stage1a stage1b stage2 candidate seed1 seed2 seed3
#         freeze --allow-test decision drift compute figures test all
uv run pytest tracks/ksvd/tests/test_compact_v4_training_sufficiency.py
```

Runs must be serial (`torch_threads=4`). See
`tracks/ksvd/notes/compact_v4_training_sufficiency.md` for the full note.
