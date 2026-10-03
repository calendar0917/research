# DECISION — zinc-joint-dictionary-decision-v1

Step `2026-10-03 00:48:19 UTC` → decision `2026-10-03 02:1x UTC`.  Compute
stopped at the C3 gate; no conditional work was submitted after the gate failed.

## Decision

**Stop.  Do not buy seed 1.  Do not run Stage D.**

The fixed joint-dictionary prototype (D) does **not** improve unseen-graph
generalisation.  The pre-registered C3 gate fails at the total-cal-gain clause
by a large margin:

```
D − F  total calibrated dev gain = −0.006487   (required ≥ +0.003)   FAIL
D − F  G0 calibrated dev gain    = −0.006099   (required ≥ +0.002)   FAIL
D − F  severe-contribution worsen = −0.000181  (required ≤ +0.001)   pass
D − B  severe-contribution worsen = −0.001788  (required ≤ +0.001)   pass
mechanism available / no failure  = true                             pass
```

Calibrated dev MAE: `F 0.118018`, `B 0.128883`, `D 0.124505`.

## Answers to the two questions

### Q1 — are the severe-cycle failures explained by existing topology25 confusion?

**No.**

* The raw and standardised topology25 keys induce the *identical* partition of
  the 10000 train rows (355 classes, affine residual 6.5e-7), so the normalised
  key adds no ambiguity.
* Train penalties are pure in 350/355 classes; no query row falls in a mixed
  class.
* The failing valid ring rows (214, 229, 249, 403, 478, 804, 935) have **no**
  exact train topology match; three of the severe five are among them.
* `train:3776` and `valid:0172` have bit-identical keys and penalty −6, yet are
  predicted 21 MAE apart.
* A train-only probe transfers penalty 0/−1 strongly (valid exact ≈ 0.99) but
  cannot recover the severe ≤ −2 rows (ExtraTrees valid MAE 0.885).

The severe rows are a small set whose target is not a function of the local
topology key; ambiguity in topology25 is not the failure mechanism.

### Q2 — does moving the dictionary to a fixed local structure/semantics description with a real sparse-reconstruction constraint improve generalisation?

**No — not to the required level, and not reproducibly over the canonical Full.**

* The reconstruction constraint is *satisfied* (X175 recovered at 1.26 %
  relative error, held-out = fit), so the sparse encoder works.
* The dictionary beats an equal-capacity plain MLP on the same input
  numerically (`D − B = +0.0044`), so the dictionary mechanism is a good
  decoder — but that edge is not significant (G0 bootstrap 95 % CI
  `[−0.0085, +0.0009]`).
* D is significantly *worse* than F overall and in the bulk G0 group
  (G0 bootstrap 95 % CI `[+0.0015, +0.0105]`), and the deficit is in the bulk
  rows (0.10048 F vs 0.10658 D on penalty 0).

**Failure localisation.**  The bottleneck is **not** the encoding (reconstruction
is near-exact), **not** the sparse mechanism, and **not** a train/test
distribution shift.  It is that the fixed X175 + an 82 k-parameter decoder is a
weaker function class than F's 311 k-parameter learned local environment, which
also consumes channels X175 does not carry (atom/bond category identity and the
remaining patch/anchor blocks).  The loss lands in the bulk G0 group; the
severe rows move within noise.

## The single next experiment worth buying

**A capacity-matched fixed-description control.**  Keep X175 exactly as is, keep
the shared sparse dictionary (K=256, s=64, 10 IHT) and the reconstruction
constraint, and widen **only** the local decoder so that D's local parameter
count matches F's 311,338 (e.g. `α(256) → 1024 → SiLU → 144`, and/or a wider
shared dictionary), then run **D-wide at seed 0** against the already-computed
`F_seed0`.

* If D-wide closes the ~0.0065 gap to F, the deficit was **capacity**, the
  fixed-description + dictionary route is viable, and the next step is a
  proper capacity sweep.
* If D-wide does not close the gap, the deficit is **information** — the fixed
  description lacks the category-correspondence channels F binds — and the
  single justified change is to extend X175 with the raw atom/bond category
  correspondence blocks, not to touch the dictionary.

This is one trajectory (≈ 941 s of A100), directly tests the one diagnosed
ambiguity (capacity vs information), and reuses the exact fold-internal prep
blob and the F/D seed-0 baselines already on disk.  No other direction is worth
buying: node amplitude/WD, reader WD, ISTA-step count, K/s/λ, warm-tail soup and
topology widening are closed.

## Guardrails honoured

* Official test never instantiated / loaded / evaluated.
* No message passing, no transformer/attention, no pair→centre update.
* All train-fit objects fit on the 8000 fit molecules only; the remote X175 was
  verified bit-identical to the local prep (`zjd-verify` MATCH = True).
* Conditional work (seed 1, Stage D) was **not** submitted before the gate.
* No push / merge; history was not rewritten.