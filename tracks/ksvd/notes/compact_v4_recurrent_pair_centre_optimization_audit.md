# Compact-v4 T=2 recurrent pair--centre — optimization-regime audit

Date: 2026-09-13
Protocol: `compact_v4_recurrent_pair_centre_optimization_audit_v1`
Reference: `compact_v4_recurrent_pair_centre` (T=2, h=48, q=16, 82,115 params,
seed0 valid 0.138376 @ ep199, seed1 valid 0.133440 @ ep234)
Official test: **never loaded**.
No architecture / weight-decay / batch / activation / relation-descriptor / T
change; no LR or scheduler grid.

## Question

Is the current Q16 model mainly limited by the training horizon /
early-stopping patience or by the learning-rate schedule, and was the earlier
Q24 negative result merely an under-optimised wider model?

## Phase A — zero-cost 2-seed ensemble diagnostic

Reused the existing official-valid predictions of `recurrent_seed0/1.json`.
No training, no ensemble-weight search.

| quantity | value |
|---|---|
| seed0 valid MAE | 0.138376 |
| seed1 valid MAE | 0.133440 |
| mean single-model MAE | 0.135908 |
| **2-seed equal-weight ensemble MAE** | **0.126227** |
| **ensemble improvement over mean single** | **+0.009681** |
| mean abs seed disagreement | 0.093219 |
| prediction correlation | 0.996511 |
| residual correlation | 0.975661 |

The ensemble gain is **+0.00968 >= +0.003**, i.e. a **strong seed /
optimization-variance signal**.  Single-model best-valid MAE is therefore
noisy at the same order as the optimization-regime effects measured below.

## Phase B — Q16 long-horizon control (seed0, validation only)

Identical architecture / optimizer / data; only `max_epochs 240 -> 500`,
`patience 40 -> 80`, scheduler still none.

| quantity | long-constant |
|---|---|
| best valid MAE | 0.139688 |
| best epoch | 244 |
| train MAE @ best-valid | 0.060817 |
| epochs run | 324 (early stop) |
| horizon boundary warning | false |
| improvement vs historical Q16 0.138376 | **-0.001313** |

A 500-epoch / patience-80 horizon does **not** recover the historical
reference; its best checkpoint is ~0.0013 *worse*.  Extending the horizon is
not the missing factor.

## Phase C — Q16 plateau scheduler (seed0, validation only)

Same as Phase B plus
`ReduceLROnPlateau(factor=0.5, patience=20, min_lr=1e-5)` driven by the
existing per-epoch validation MAE.

| quantity | plateau |
|---|---|
| best valid MAE | 0.138201 |
| best epoch | 231 |
| train MAE @ best-valid | 0.045961 |
| epochs run | 311 (early stop) |
| horizon boundary warning | false |
| final / minimum LR | 3.125e-5 / 3.125e-5 |
| improvement vs historical Q16 0.138376 | **+0.000174** |

LR trajectory (also written to `q16_plateau_seed0.json:scheduler_events`):

| epoch | LR after step | valid MAE at trigger |
|---|---|---|
| 188 | 1e-3 -> 5e-4 | 0.145404 |
| 209 | 5e-4 -> 2.5e-4 | 0.147640 |
| 252 | 2.5e-4 -> 1.25e-4 | 0.150954 |
| 273 | 1.25e-4 -> 6.25e-5 | 0.147396 |
| 294 | 6.25e-5 -> 3.125e-5 | 0.142399 |

The best checkpoint (0.138201 @ 231) occurs while the LR is 2.5e-4; the three
later decays do not improve valid MAE despite driving train MAE from 0.061 to
0.040.  This is a **generalization-limited**, not an optimization-limited,
signature.

## Phase D — current / long / plateau comparison

All rows share the exact same data split (official train 10,000 / valid 1,000,
targets hash-identical across seeds), architecture and parameter count
(82,115); only the optimization protocol differs.

| run | best valid | best ep | epochs run | train@best | vs hist. Q16 |
|---|---|---|---|---|---|
| current Q16 (historical, 240/40) | 0.138376 | 199 | 239 | 0.064573 | 0.000000 |
| current protocol re-derived, current code (240/40) | 0.140609 | 167 | 240 | 0.074466 | -0.002234 |
| long constant (500/80) | 0.139688 | 244 | 324 | 0.060817 | -0.001313 |
| plateau (500/80 + ReduceLROnPlateau) | 0.138201 | 231 | 311 | 0.045961 | +0.000174 |

Paired effects (same code branch unless noted):

| comparison | effect |
|---|---|
| horizon/patience: long - current-code-current | +0.000921 |
| **scheduler: long - plateau** (paired until ep188) | **+0.001487** |
| combined: current-code-current - plateau | +0.002408 |
| reproduction gap: current-code-current - historical | +0.002234 |

Against the **frozen historical reference 0.138376** neither long
(-0.00131) nor plateau (+0.00017) reaches the +0.002 continuation gate.
Against a **same-code re-baseline** (0.140609) the scheduler effect is
+0.00149 and the combined horizon+scheduler effect is +0.00241, but the
reproduction gap between the historical and current-code baselines is itself
+0.00223 — i.e. as large as the effect being measured.

### Reproduction control (critical caveat)

Because long did not reproduce epoch-level history, a 4-thread reproduction
control was run (61 epochs, current 240/40 protocol, same seed):

* `rA` and `rA2` (`real_batch_identity=True`) are **bit-identical to each
  other and to the long run** (max train/valid diff 0.0 for 61 epochs).
* `rA` is **bit-identical to the historical `recurrent_seed0` through
  ~epoch 40**, then diverges (max valid diff 0.0225 over 61 epochs;
  epoch-60 train 0.112866 vs historical 0.118542).
* `rB` (`real_batch_identity=False`) differs from epoch 1, confirming the
  identity-block RNG state is part of the trajectory.

Conclusion: the current code + 4-thread environment is *deterministic*, but it
reproduces the new `q16_long` branch, **not** the 2026-09-12
`recurrent_seed0` branch.  The two branches agree for the first ~40 epochs and
then diverge by implementation/run-level float effects.  At the selected
checkpoint the branch gap is **~+0.0022 valid MAE**, i.e. at or above the
+0.002 decision gate.  This is an optimization/reproduction noise floor and
must be carried with any sub-0.002 conclusion.

## Phase E — Q24 retry

**Not triggered.**  The task gate requires at least one of Phase B/C to beat
the reference Q16 seed0 (0.138376) by >= +0.002.  Best new improvement vs the
frozen reference is +0.000174 (plateau) -> `q24_retry_triggered = false`;
Q24 was not retrained (so its q-sparsity / dead-dim / effective-rank
comparison under a new protocol is not available).

For reference, the previous Q24 result under the old protocol was
0.138746 @ ep120 (train@best 0.086021), 8/24 dead q dimensions, effective
rank 5.59 vs Q16 7.12.

## Verdict

**No meaningful optimization signal** against the frozen reference:

* horizon / patience alone does not help (long is *worse* than the reference);
* LR decay gives only a weak, sub-threshold paired effect (+0.00149 vs long,
  +0.00175 vs the historical reference);
* the plateau's extra train fit (0.061 -> 0.040) does not convert into valid
  MAE, consistent with a generalization-limited regime;
* the 2-seed ensemble (+0.0097) shows that seed / optimization variance is
  much larger than the horizon/scheduler effects;
* a ~0.0022 reproduction gap between the historical and current-code branches
  is as large as the effect being tested.

**Do not trigger Q24 / Q32 / balanced and do not start an LR/scheduler/WD
sweep.**  The relation-width scaling question stays closed for this encoder
family under the current evidence.

## Sanity / fairness

* `fairness.json`: same official train/valid split (10,000 / 1,000), valid
  targets identical across seeds, Q16 capacity builder bit-identical to the
  frozen recurrent builder, Q24 = 91,211 params; only
  `max_epochs`/`patience`/`scheduler` differ.
* `sanity` is inherited from the frozen recurrent/ capacity modules; the only
  change to the canonical loop is an optional, default-off scheduler.
* `official_test_loaded = false` everywhere; the official test is never
  loaded by this module.

## Files

- `experiments/luyin16/zinc_compact_v4_recurrent_pair_centre_optimization_audit.py`
- `tests/test_compact_v4_recurrent_pair_centre_optimization_audit.py` (8 pass)
- `results/compact_v4_recurrent_pair_centre_optimization_audit/`:
  `phase_a_seed_ensemble.json`, `fairness.json`, `q16_long_seed0.json`,
  `q16_plateau_seed0.json`, `decision.json`, `q24_retry.json`, `report.json`,
  `reproduction_control/` (rA/rA2/rB curves, log, `reproduction_control.json`),
  `curves/`, `states/`, `runs/`, `logs/`.
- Canonical loop change: `zinc_compact_v4_smallhead_e2e.train_model` gained an
  optional `scheduler="reduce_on_plateau"` path; default remains `"none"` and
  is bit-inert for every existing protocol.
