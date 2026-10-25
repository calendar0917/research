# NEXT DECISION — what is worth buying after this round

**Round verdict: `NO_USEFUL_GAIN`.** Q-fixed (0.110569) and F-content (0.101948)
both fit the 10k train set as hard as DCC-L and still lose +0.032 / +0.023 on
valid; both curves are flat from epoch 300 and W20/T100 agree with the best
epoch. Nothing in this round justifies another explicit-statistics *body* at
official 10k.

## 1. Do not buy

* **Q-adaptive / F-fixed (declared controls).** They were conditional on a
  competitive first seed and would now only decorate an eliminated arm.
* **A third explicit-statistics architecture at 10k.** The repo now has nine
  negative structural-statistics rounds (R0/R1, B, T, MFIC, SFC, RCC, SCM, GOS,
  QF/FC) against six positive ones of size ≤0.005. Another body is a lottery
  ticket, not a research step.
* **A seed-1 confirmation of an arm that is 12–17 seed-sd behind.** It cannot
  change the verdict.
* **A rescue of Q by readout surgery.** The rebinding probe shows the root
  binding is worth 0.0049 MAE in the trained model; the missing 0.032 is not
  hiding in the root pairing.
* **Anything that reads official test.** Unchanged from the task.

## 2. The one purchase that can still change the direction's fate

**Label-scale point for the stronger body: F-content on the frozen nested Fit4k
split (and Fit8k if the 4k point is not decisive), matched protocol.**

* Why this and nothing else: the only A-grade lever ever measured in this repo is
  label count (DCC-L 4k→8k = **−0.037** locked / −0.040 valid; 8k→10k ≈ −0.017).
  If an explicit-statistics body is *relatively* better when labels are scarce,
  then its deficit at 10k is a variance/asymptote problem and the direction has a
  real niche. If it is uniformly worse, the direction is closed for good at this
  capacity — and that is worth knowing for 1 GPU·h rather than nine more bodies.
* Exact arms: `F-content` (205,506 params, unchanged forward, unchanged code at
  this commit) trained on the frozen `bjoc_predictive_bottleneck_v1` Fit4k rows,
  batch 128, **47,400 updates** (1,482 epochs), same recipe and early-stop rules,
  monitor read only, **no test**; 1 seed first, second seed only if the first
  point is within ±0.005 of the DCC-L 4k band (0.121–0.136 valid).
* Cost: **≈0.95 GPU·h** for the 4k point (measured: 2.3 s/epoch × 1,482 epochs on
  one A100), 0.7 GPU·h for Fit8k. Both fit easily in the remaining 14.86 GPU·h.
* Decision rule (frozen before running):
  - F_Fit4k valid **≤ 0.135** (DCC-L 4k band) → the explicit-statistics family is
    relatively better in the label-scarce regime: next round should attack its
    *asymptote* (organisation/capacity), not its variance, and the 10k deficit is
    not evidence against the direction.
  - F_Fit4k valid **≥ 0.150** → uniformly worse than the DCC-L 4k arm: close the
    explicit-statistics direction at ≤500k / 10k, keep this round's mechanism
    findings (bag collapse of rooted measures; active-but-non-beneficial forest
    marginals), and move the main line to the data side (12k labels, unchanged
    DCC-L forward) rather than to a tenth structure module.

## 3. Documented single-cause candidate if F is ever reopened

The learned edge weights saturate **exactly** at both ends of the declared
interval (`w_min = 0.1`, `w_max = 2.0` on real valid bonds). The bound was fixed
by design and never tuned, so "the model wants weights outside `[0.1, 2]`" is an
identified, untested parameterisation issue — one clean correction (positive
unbounded, e.g. `w = 0.1 + softplus(·)`, or a wider declared interval) would be
the allowed single fix. It was **not** bought this round because the arm was not
competitive; it should not be bundled with the label-scale probe above, or the
two effects become unidentifiable.

## 4. Mechanism facts to carry forward

1. **Rooted measures collapse to a bag under sum pooling** (root rebinding: mean
   |Δpred| 0.0049, corr 0.99999; root-pool survival 0.43). Any future
   root-conditioned head must use a readout that cannot cancel the root identity
   (e.g. root-pair interactions or set attention), and must be *checked* with this
   probe at the selected checkpoint before being believed.
2. **Forest marginals are genuinely active and genuinely unhelpful** at 10k
   (w→1 costs +0.030 MAE, w→molecule mean +0.010), i.e. the statistic is read but
   does not buy generalisation. Do not re-derive this with another forest
   variant.
3. **Matched-fit comparisons are the honest discriminator here.** All three
   bodies reach online train MAE ≈0.004–0.005 at epoch 500 while valid is
   0.112 / 0.103 / 0.079. Report future candidates at matched train fit, not at
   matched epoch.
