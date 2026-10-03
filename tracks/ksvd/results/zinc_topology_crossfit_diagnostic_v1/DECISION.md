# DECISION — zinc-topology-crossfit-diagnostic-v1

Step start `2026-10-03 03:15 UTC` → decision `2026-10-03 03:57 UTC`.  All compute
stopped after the four fixed CPU readout fits; no conditional work was bought.

## Decision

**Close the fixed ExtraTrees prompt-value residual readout.**  The pre-registered
gate FAILS in both folds:

```
gain_base = MAE(base) - MAE(TP) = -0.026359   (required >= +0.003)   FAIL
gain_topo = MAE(P)    - MAE(TP) = +0.019743   (required >= +0.003)   pass
TP vs base G0 worsening          = +0.031459  (required <= +0.001)   FAIL
per-fold TP-base gain            = A -0.045248, B -0.039601          FAIL
contribution identity            = true                              pass
```

Main dev MAE: base `0.150606`, P `0.196708`, TP `0.176965` (const control
`0.150507`, i.e. a pure offset buys nothing).  The corrected historical 8000-row
F is `0.116365` (reference only).

## Which configuration is closed, and what is NOT closed

* **Closed:** the `ExtraTreesRegressor(256, leaf=1, mf=1.0, rs=0)` readout of the
  per-row residual `r = y - p_base_cal` on top of a 4000-row Full base, at
  amplitude 1 with no regularisation.  Its off-fold variance dominates: even the
  1-feature `P` arm loses 0.046 MAE to the untouched base, so the negative
  `gain_base` is a readout property, not a property of T25.
* **NOT closed:** the topology25 channel, the structural/task dictionary route,
  the topology-as-internal-representation question, or any larger-dictionary /
  capacity direction.  A negative on this readout is not a negative on T25.

## Why this is not interpretation #1 but is #4-shaped

The topology channel does show a held-out conditional increment (`TP > P` in
both folds, +0.0280 / +0.0162), and it partially improves the ring/severe pools
(severe contribution 0.025252 → 0.021436).  But its G0 regression (+0.031459)
is ~8× the severe gain, so it cannot be sold as a general improvement.  The
bootstrap intervals for both gains cover zero, and the frozen base-max-error-row
deletion does not change the sign.  The direction is consistent (both folds
negative), so this is not the “contradictory folds” case either — it is a fixed
readout with too much variance to be useful.

## The single next action worth buying

**A regularised topology-class residual correction under the identical
cross-fit protocol.**  Replace the leaf-1 per-row memoriser with a fixed
shrinkage/aggregation estimator: group meta-fold rows by their exact T25 class,
shrink each class residual mean toward the global meta median with a fixed prior
strength, and apply the resulting class-level correction to dev.  Optionally a
single small ridge/linear map on the standardised T25 columns as a second fixed
arm.  No leaf/depth/tree-count scan, no new topology features, no dictionary
change, no third base.

*Why this is the action tied to the result:* the only positive signal this round
is `TP > P` (topology groups residuals better than the prediction value), and
the only diagnosed failure is readout variance.  A class-mean/shrinkage
estimator directly attacks the variance while keeping the same information
channel and the same cross-fit discipline.

*Evidence required before anything else is bought:* in the same 2×4000 cross-fit
on the frozen 2000-row outer dev, **both** `gain_base ≥ 0.003` **and**
`gain_topo ≥ 0.003`, both per-fold `TP−base` gains non-negative, TP vs base G0
worsening ≤ 0.001, and the contribution identity holding.  If a regularised
readout still fails `gain_base`, the honest conclusion is that this external
residual-correction route does not pay on top of a 4000-row Full, and the next
question becomes whether T25 is useful *inside* the trained representation
rather than as an external correction.

## Competing explanations still open (explicit)

1. **Readout variance** (favoured): leaf-1 on a single heavy-tailed scalar.
2. **Residual is genuinely not a function of T25**: exact-class coverage is
   ~98 %, yet within-class residuals vary by ≈0.10–0.14 — the residual carries
   chemistry/base error T25 does not encode.
3. **Base residual too heteroscedastic**: 4000-row bases have larger, more
   skewed residuals than the 8000-row Full, making external residual learning
   harder.
4. **T25 as internal representation**: the Full shares a topology encoder, but a
   4000-row budget may not shape it; an external probe cannot decide this.

## Guardrails honoured

* Official test never instantiated / loaded / evaluated.
* Official-valid not newly evaluated (historical predictions read-only, used
  only for the errata).
* No D-wide, no capacity/λ/s sweep, no WD/loss/optimizer search, no node rescue,
  no extra topology features, no third base, no extra seed, no full-data
  confirmation.
* Two bases ran concurrently on one regime (res-2 / res2-cu124 / c05 / driver
  525.85.12 / torch 2.5.1+cu124 / FP32, no AMP/DDP), same deployed commit
  `240bd2b`; no push, no merge, no history rewrite.