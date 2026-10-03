# DECISION — zinc-g0-transfer-audit-v1

Start `2026-10-03` (local CPU).  All compute stopped after the Part A/B/C
scripts; no new training, no remote job, no optimizer step.  Official test never
touched; official-valid not re-evaluated.

## Part A decision

**Keep the previous round's FAIL, with a corrected magnitude.**  The paired
bootstrap bug inflated the intervals; after fixing it the failure is
statistically sharper, not reversed (`gain_base = −0.026359`,
CI [−0.033471, −0.019069]; `gain_topo = +0.019743`, CI [+0.007380, +0.031683]).
The fixed ExtraTrees prompt-value readout stays **closed**.  The `const` control
is now correctly bounded by its shift (`±0.00234`).

## Part B decision — **FAIL (close this exact rule)**

The class-median shrinkage toward zero with fixed prior strength 5:

* protects G0 (G0 worsening **−0.000236**, i.e. a small improvement);
* is direction-consistent and non-negative in both folds (A +0.001238,
  B +0.000592);
* has a bootstrap interval excluding zero vs both base and const;
* **but** gains only **+0.001195** vs base and **+0.001096** vs const, both below
  the pre-registered **≥0.003** purchase threshold.

Per the frozen rule this scheme is **closed**.  Do **not** sweep the prior
strength, do not switch to class means / ridge / tree coefficients, and do not
attach this CPU correction to the historical 8000/10000-row models (changing the
base training size changes the residual distribution).  Severe cycles that have
no exact meta T25 class (435, 464, 874, 1004, 1499) remain uncovered; a class
correction cannot reach them, and their label-formula/definition difference is
recorded, not solved.

## Part C decision — **coverage proxy did not locate the bottleneck**

The pre-registered main exposure `joint_rare_marginals_seen` is essentially
unpopulated: its molecule mean is 0.0008–0.0046 and the fixed `≥0.25` high group
contains **0 molecules** in every fold/view/key.  Matching returns 0 pairs; no
matched contrast can be estimated, and thresholds are not changed.  The
pre-registered full-descriptor key agrees (also low-cardinality), so the two keys
do not conflict — they both say the current structural code is collapsed
(1182 unique `phi65` rows, 202–212 supports, 317 edge joint keys).  Mechanism
evidence additionally shows the node channel is dead in both folds and F_B's
edge channel has collapsed.

Therefore:

* **This coverage proxy is closed as a bottleneck locator** for the current
  representation.  It does **not** justify buying a structure-semantics fusion
  change, and it does **not** clear one either.
* We do **not** conclude that "regularisation / readout is the only problem" and
  we do **not** conclude that structure-semantics fusion is useless.  The
  correct statement is: *the current structural key is too coarse for
  coverage-based rarity analysis; the bottleneck is not located by this proxy.*
* The one hypothesis genuinely **closed** this round is the specific
  "combination-rarity explains G0 error" hypothesis under this exact
  representation/key/threshold; and Part B's exact class-median shrinkage rule.
* The hypothesis worth future confirmation is the **representation** question:
  the shared structural dictionary code has collapsed to a tiny discrete set and
  the structure channel is not stably carried across folds.  That is a property
  of the current承重 interface, not of fusion per se.

## Single next action (diagnostic, no training auto-purchase)

A cheap, decisive discrimination of "collapsed structural code" vs "structure
channel not trained": freeze the two soups and measure, on the same unseen G0,
how much of the model's prediction is attributable to the residual structural
coordinate **when it is actually load-bearing** — i.e. re-encode G0 with the
F_A soup (alive edge channel) while intervening only on `coord` (residual
support) and compare the induced prediction change against the base residual.
This is a forward-only probe on existing checkpoints; it requires no training and
would tell us whether the coarse structural key still carries usable signal
before any fusion change is bought.  If it is flat, the next purchase is a
representation/regularisation question, not a fusion change.

## Guardrails honoured

* No new base/warm/GPU/dictionary-prep/widening/λ-s-WD sweep/node rescue/ridge/
  tree fit.  Only the frozen class-median correction was fit (Part B), as
  allowed.
* No `rr run/deploy`, no manual SSH; only existing local checkpoints were read.
* Official test never instantiated/loaded/evaluated; official-valid not
  re-evaluated.
* No push, no merge, no history rewrite; the new results live on the isolated
  branch `zinc-g0-transfer-audit-v1`.  Other untracked work was left untouched.