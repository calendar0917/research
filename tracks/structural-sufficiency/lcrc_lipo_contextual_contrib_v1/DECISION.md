# DECISION — `lcrc-lipo-contextual-contrib-v1`

## Decision

**`NO_CLEAR_GAIN`** (tail label: **`NOT_TAIL_CONCENTRATED`**).

## Fixed rule and observed values

The rule was frozen before training in `PROTOCOL.md` / `configs/contextual_contrib_v1.yaml`:

* `PROMISING_EXPLORATORY` iff
  `gain = RMSE(BASE) − RMSE(CONTEXT) >= 0.020` **and**
  `MAE(CONTEXT) − MAE(BASE) <= 0.010`;
* otherwise `NO_CLEAR_GAIN`.

Observed on Validation420 (raw logD, last-5 soup, seed 0):

| quantity | observed | threshold | pass |
| --- | ---: | ---: | :---: |
| Validation RMSE BASE | 0.580416 | — | — |
| Validation RMSE CONTEXT | 0.568587 | — | — |
| **gain** | **+0.011829** | ≥ 0.020 | **no** |
| Validation MAE BASE | 0.427854 | — | — |
| Validation MAE CONTEXT | 0.411573 | — | — |
| **MAE change** | **−0.016282** | ≤ +0.010 | yes |

The MAE guard passes, but the primary RMSE threshold is not met. Because the
decision is conjunctive, the outcome is `NO_CLEAR_GAIN`.

## Tail rule

The five row IDs `{1630, 2924, 822, 1153, 3634}` were fixed by the prior gap
audit; no new tail search was performed.

* net SSE improvement (BASE − CONTEXT) = +5.708358 overall;
* tail net SSE improvement = **−0.786659** (CONTEXT is worse on the five rows
  in net; 2 improve, 3 worsen);
* tail share of the overall improvement = **−0.1378**;
* gain after removing the five rows = **+0.014550**.

The fixed `KNOWN_TAIL_CONCENTRATED` trigger requires a positive net SSE
improvement and either a tail share ≥ 0.50 or a non-positive gain without the
tail. Neither holds, so the label is **`NOT_TAIL_CONCENTRATED`**. The modest
gain is not carried by the known tail molecules — and it is also too small to
pass the primary threshold.

## Was the mechanism trained and used?

Yes: all four new layers moved from initialisation and received non-zero
gradients across all 9,720 steps; the correction is non-zero on all 2,612
hetero atoms in Validation420 (mean L2 2.0931 per atom); `mean |f(h_new) −
f(h_original)| = 0.12216` per hetero atom; and zeroing the trained `W_out`
changes predictions by 0.7992 on average (RMSE 0.568587 → 1.112137 on its own
trained base path). This establishes mechanical use, not generalisation.

## Scope and follow-up

One seed, one frozen split, Validation420 only, no Test access, one
pre-registered pair of runs, no early stopping and no Validation-selected
checkpoint. Total GPU spend 0.2172 / 0.45 A100 GPU·h. No follow-up experiment,
no structure or learning-rate change, and no new model is proposed. The result
is exploratory and must not be reported as evidence of independent
generalisation.
