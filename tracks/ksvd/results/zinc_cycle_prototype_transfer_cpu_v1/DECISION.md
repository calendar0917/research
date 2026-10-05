# DECISION — `zinc_cycle_prototype_transfer_cpu_v1`

**Verdict: `TARGETED_REPAIR_ONLY` + `SINGLE_ROW_DOMINATED`.** No `PROTOTYPE_DEPLOY_SUPPORT`, no
`NO_CANDIDATE` (the point values are genuinely positive), no automatic next round.

## Decision

STOP this exact-lookup, train-consistent prototype line **as a broad transfer mechanism**. Keep
`H` only as a documented single-row tail repair on this fixed exposed valid. Do **not**:

* change the coverage gate, the key, the consistent-class rule or any support threshold;
* add a nearest-neighbour / rounding grid / smoothing rescue;
* add seeds, folds, coefficients, shrinkage or a severity condition;
* open official-test or start a new training run;
* re-tune `b_H` or select between `b_H` and `b_B`.

Do **not** re-confirm COMP again; COMP stays the single-seed working reference.

## Why

| quantity | value |
|---|---|
| valid B `y_cal` MAE | 0.1174061835 |
| valid H `y_cal` MAE | 0.0966943750 |
| overall cal gain (point) | +0.0207118085 |
| overall cal 95% CI | [−0.0002259826, +0.0623398197] |
| overall raw gain | +0.0207312735 |
| G0 cal worsening | +0.0000116891 |
| max positive row | `valid:0172` (k=−6), gain +20.74 |
| max positive row / positive sum | 0.980 |
| gain excluding B's worst row | −0.0000288437 |
| H `y_cal` (`0.09` marker) | not observed (0.09669) |

The apparent +0.0207 gain is `valid:0172` alone: a singleton train class (train:3776, k=−6) with
the **exact same T25** as the valid row, where the frozen `Q` predicted ≈ 0 instead of
`c = −20.80`. The prototype replaces it with `c`, cutting that row's error from 20.81 to 0.074.
Every other route behaves as follows:

* `CONSISTENT_HIT` (excluding 0172, n=987): `q` error drops 0.001139 → 8.6e−10 but `y_cal` MAE
  moves 0.087806 → 0.087835 (slightly **worse**) because the small favourable cancellation of the
  body error is removed.
* `UNSEEN_FALLBACK` (n=12): no change by construction; `q` error 0.802 stands.

So the mechanism is: the body error `e_g = h − g` (MAE ≈ 0.087) dominates the complete `y`; `Q`'s
cycle error is already small (≈ 0.0011) on covered rows; forcing `q = c` removes a tiny
cancellation and does not help.

## Answers to the five questions

1. **q more accurate?** Yes, dramatically on train-consistent exact-hit rows (`q` MAE 0.0011 →
   ~0 on the bulk; one severe row fixed).
2. **complete y improved?** Only on one row. On 999/1000 rows it is flat-to-slightly-worse.

## Next responsibility (one question, not executed)

The covered classes are not where the `y` budget is; the residual is the **chemistry body error
`h − g`** and its cancellation with the cycle term. The next legitimate question is whether
`h − g` can be reduced without breaking the cancellation — a chemistry-side question, not a
cycle-head or lookup question. A learnable cross-class cycle regularity would need a separate
design (to test whether T25 can transfer across classes), and a train-conflict interface would
first have to prove it adds distinguishable information in train.

## Revisit if

A new preregistration mechanistically explains why a train-consistent exact T25 class can
generalize to held-out rows (i.e. describes the alias structure), or establishes a reduced
`h − g` body error while preserving the cycle cancellation, and carries the replayed COMP+Q as
the matched reference.
