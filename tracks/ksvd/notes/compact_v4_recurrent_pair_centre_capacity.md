# Compact-v4 T=2 recurrent pair--centre — capacity scaling audit

Date: 2026-09-13
Protocol: `compact_v4_recurrent_pair_centre_capacity_v1`
Reference: `compact_v4_recurrent_pair_centre` (T=2, h=48, q=16, 82,115 params)
Official test: **never loaded**.

## Question

Is the current best model limited mainly by the relation width `q_dim=16`
and/or the centre width `h_dim=48`?  Small, interpretable capacity audit;
no structural primitives, no hyperparameter search.

## Phase A — Q16 capacity audit (existing logs)

| quantity | recurrent seed0 | recurrent seed1 | 1-round baseline seed0 |
|---|---|---|---|
| best valid MAE | 0.138376 @ ep199 | 0.133440 @ ep234 | 0.145334 @ ep176 |
| epochs run | 239 | 240 (horizon warning) | 216 |
| train MAE at best-valid | 0.064573 | 0.075917 | 0.073274 |
| best train MAE | 0.057973 | 0.064274 | 0.065794 |
| final train MAE | 0.067652 | 0.068838 | 0.066886 |
| train--valid gap at best | 0.073803 | 0.057523 | 0.072059 |
| tail (10%) train delta | +0.0017 | -0.0012 | -0.0060 |
| tail (10%) valid delta | -0.0166 | +0.0011 | -0.0006 |

Read: train MAE is far below valid MAE (gap ~0.06--0.07) and valid is noisy
with its best epoch late in the run.  Train error is not at an interpolation
floor either, but the dominant gap is train -> valid.  This is a
**generalization / finite-data-limited regime**, not a clean
under-capacity signature.

## Phase B — relation width scaling (h=48 fixed)

Only `pair_hidden` (q) changed.  T=2 recurrence, tokenizer, relation
descriptor, ReLU, readout, small head (13,13), optimizer/scheduler/batch,
seed, protocol all inherited.  Seed0, validation only.

| name | h | q | total params | head params | Δ vs Q16 | R width |
|---|---|---|---|---|---|---|
| Q16 (reference) | 48 | 16 | 82,115 | 4,135 | 0 | 302 |
| Q24 | 48 | 24 | 91,211 | 5,175 | +9,096 | 382 |
| Q32 | 48 | 32 | 100,307 | 6,215 | +18,192 | 462 |
| BALANCED | 64 | 32 | 104,211 | 6,631 | +22,096 | 494 |

Q24 seed0 result:

| quantity | Q24 seed0 | Q16 seed0 |
|---|---|---|
| best valid MAE | 0.138746 @ ep120 | 0.138376 @ ep199 |
| epochs run | 160 (early stop) | 239 |
| train MAE at best-valid | 0.086021 | 0.064573 |
| best train MAE | 0.072673 | 0.057973 |
| epoch time | 9.03 s | 7.44 s |
| peak RSS | 2,390,004 KB (~2.3 GB) | not recorded |

`improvement = Q16 - Q24 = -0.000371` → **NO_MEANINGFUL_SIGNAL**
(pre-registered stop threshold: ≤ +0.001).  Q24 does not help; it is tied
within validation noise and slightly worse at its selected checkpoint.
Notably Q24 reaches its optimum much earlier (ep120 vs ep199) and fits the
training set *worse* (best train 0.0727 vs 0.0580), i.e. extra relation
capacity did not increase usable train fit.

### q utilisation diagnostics (trained seed0, first 128 valid molecules)

| state | zero frac | dead dims | never-active dims | eff. rank (part.) | mean | std |
|---|---|---|---|---|---|---|
| Q16 q1 | 0.860 | 1/16 = 0.0625 | 0.0625 | 7.12 | 0.042 | 0.130 |
| Q24 q1 | 0.940 | 8/24 = 0.333 | 0.333 | 5.59 | 0.012 | 0.062 |

Widening q from 16 to 24 leaves **8 of 24 dimensions completely dead**
(never positive, zero variance) and lowers the effective rank from 7.1 to
5.6.  The extra width is largely inactive — the relation channel is already
wider than the ReLU-activated subspace the model uses.

## Phase C — balanced ~100k config

**Not run.** Authorised only if q-width shows a positive signal (or Q32 a
clear trend).  Q24 = -0.000371 → not authorised.  `BALANCED (h=64, q=32,
104,211 params)` was constructed and sanity-checked (finite fwd/bwd, weight
tying) but not trained.

## Phase D — replication gate

**Not triggered.** Gate requires ≥ +0.003 improvement; best new improvement
was -0.000371.  seed1 not purchased.

## Sanity / regression

`results/.../sanity.json`: Q16 capacity builder is bit-identical to the
frozen recurrent builder (state hash + output max diff 0.0); all variants
finite forward/backward; weight tying 2x per variant.  10/10 unit tests pass.

## Verdict

**没有明显 capacity scaling（relation channel 不是主要瓶颈）.**

- Widening q alone (Q24, +9,096 params) gives no validation gain and mostly
  dead dimensions; q=16 is not shown to be too narrow.
- Both train and valid remain far from zero with a large train--valid gap,
  so the binding constraint is generalization / finite data, not relation or
  centre hidden capacity.
- Caveat: the valid curve is noisy and Q24's optimum/stopping differ from
  Q16's; a pure capacity conclusion is partly confounded by
  optimization/generalization.  This does **not** justify an LR/scheduler
  sweep or a wider q sweep now.

Next authorised action: stop.  Do not run Q32 / balanced / seed1 / official
test and do not start a hyperparameter search without a new pre-registered
hypothesis.

## Files

- `experiments/luyin16/zinc_compact_v4_recurrent_pair_centre_capacity.py`
- `tests/.../test_compact_v4_recurrent_pair_centre_capacity.py` (10 pass)
- `results/compact_v4_recurrent_pair_centre_capacity/`:
  `phase_a_capacity_audit.json`, `parameter_accounting.json`, `sanity.json`,
  `q24_seed0.json`, `diagnostics_q16_seed0.json`,
  `diagnostics_q24_seed0.json`, `decision_q24_seed0.json`,
  `decision_summary_seed0.json`, `report_seed0.json`, `curves/`, `states/`,
  `logs/`.
