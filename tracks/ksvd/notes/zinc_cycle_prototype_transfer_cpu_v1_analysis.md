# zinc_cycle_prototype_transfer_cpu_v1 — analysis note

**Question.** Among the exact T25 classes the frozen `Q` sees, which were already
train-consistent but mislearned, and does a train-only per-class median-`c` prototype improve
the complete deployable `y`? Reference `B` = frozen COMP+Q (`CHEM_PASS_DEPLOY_PASS`). One fixed
candidate `H`. CPU only, no training, official-test never read. Reused exposed official-valid.

## Headline

| | B (COMP+Q) | H (prototype) |
|---|---|---|
| valid `y_cal` MAE | 0.1174061835 | 0.0966943750 |
| valid `y_raw` MAE | 0.1166162887 | 0.0958850152 |
| overall cal gain | — | **+0.0207118085**, CI [−0.0002260, +0.0623398] |
| overall raw gain | — | +0.0207312735 |
| G0 cal worsening | — | +0.0000116891 |

Gate conditions 1/3/4 pass, CI condition 2 fails → `TARGETED_REPAIR_ONLY`. The gain is
`SINGLE_ROW_DOMINATED`: `valid:0172` is +20.74 of the +21.17 positive sum (share 0.980);
excluding the reference's worst row the gain is −0.0000288. `y_cal < 0.09` not observed.

## Why the point value is large but not broad

1. Train has 355 exact T25 classes (350 consistent / 5 conflict, 151 singletons). Within a
   consistent class `c = (k − mu_cycle)/sigma_cycle` exactly, so the prototype value is exactly
   the class cycle label.
2. Trained `Q` misses this on train (`q` MAE vs `c` 0.011506; prototype 0.005207) and the head is
   known to be weak on extreme cycles.
3. On the reused valid, 988/1000 rows are `CONSISTENT_HIT` (0 new conflicts, k match rate 1.0),
   12 are `UNSEEN_FALLBACK`, 0 conflict-fallback.
4. `valid:0172` (k=−6) matches a singleton train class (train:3776, same T25, k=−6): `Q`
   predicted ≈0 instead of `c = −20.80`. The prototype sets `q_H = c` and cuts a 20.81 error to
   0.074. This one row is essentially the entire gain.
5. On the other 987 `CONSISTENT_HIT` rows the `q` error collapses 0.001139 → 8.6e−10 **but**
   `y_cal` MAE moves 0.087806 → 0.087835 (slightly worse): the body error `e_g = h − g`
   (MAE 0.087004) dominates and `Q` was already providing a small favourable cancellation
   (triangle gap −0.001149), which setting `q = c` removes. `Σ(|e_g|+|e_c|) = 0.088143` vs the
   true `|e_g + e_c| = 0.086994`.

So the answer splits cleanly: **yes**, `q` is more accurate (the original `Q` missed a
train-present class signal on covered rows — a function-fit miss, not an input information
deficit); **no**, the complete `y` does not improve beyond the single repaired tail row, because
the residual budget is the chemistry body error `h − g` and its cancellation with the cycle term.

## Conflicts, coverage and the three failure modes

* **Function-fit miss (covered resolved rows):** prototype drives `q` error to ~0 while `Q` errs
  0.0011; a small head was underfit. But the body dominates, so no deployable gain.
* **Input conflict (train):** 5 classes have non-unanimous `k` at the same T25 (e.g. class 270
  k∈{−2,0}; class 280 k∈{−6,0}); same-input capacity cannot remove their class-constant L1 floor.
  No valid row routes here (0), so it does not move the valid result.
* **Uncovered:** 12 valid rows (including the large-error `valid:0935` k=−2, |err| = 5.27) are
  `UNSEEN_FALLBACK`; the rule is inactive and changes nothing. This closes only the exact-lookup
  rule's action there; it does not falsify a learnable cross-class cycle regularity, and it does
  not show those inputs lack information.

## Next responsibility (written, not executed)

Most relevant classes are covered, cycle prediction is already accurate, and `y` is still poor
(0.0967; `h − g` dominates). The next responsibility returns to the **chemistry body error and
its cancellation**, not the cycle head or the lookup. Explicitly stopped: touching the
gate/key/rule/threshold, nearest-neighbour/smoothing/shrinkage rescue, new seeds/folds, scanning
coefficients, re-confirming COMP, reopening dictionary/binding/reader branches, opening
official-test, and any new training run.

## Boundaries

Single seed; reused exposed valid; extra train-side label `c` (as in source `Q`); one rule, one
bias; not a dictionary proof; not `luyin19` fulfillment; no test; the oracle (`h + c + b_B`,
MAE 0.089325) is a label-reading diagnostic only.
