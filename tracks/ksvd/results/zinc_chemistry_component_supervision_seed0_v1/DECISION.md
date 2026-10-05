# DECISION — `zinc_chemistry_component_supervision_seed0_v1`

## Decision

**Performance gate passed → freeze a confirmation design; do not extend this round.**

* Classification: `COMPONENT_SUPERVISION_CANDIDATE`.
* COMP vs SUM calibrated `g`-MAE gain: G0 `+0.009974` CI `[0.0065, +0.0135]`, overall `+0.009873`
  CI `[0.0064, +0.0136]`; 3/3 frozen gate conditions satisfied. Raw gains point the same way.

## What is authorised next

1. **Confirmation (frozen).** On the same fresh fold and the same fresh-fold M skeleton, the only
   legitimate next step is a single confirmation of the passed gate with a new seed (seed 1) to
   test training-randomness uncertainty. This is **not** executed here; it requires its own
   preregistration committing to (a) the same fold, (b) the same fresh init, (c) two arms SUM/COMP,
   (d) the same 0.5 component weight, (e) a fixed stop before a minute-90-equivalent cutoff, and
   (f) the physical-GPU UUID capture rule that was violated last year in
   `zinc_direct_bond_relation_seed0_v1`. Until that preregistration lands, **no seed-1 / no second
   confirmation is bought from this point estimate**.

2. **Component diagnosis → one directed design.** Both components are `FIT_ADEQUATE`, but the
   residual chemistry component `s` is the dev-G0 gap leader (dev-G0 MAE `0.083` vs `ell` `0.054`,
   ratio ≥ 1.5) and departs from SA exactly where the cycle-snap residual is large (`epsilon` grows
   with cycle rarity). One new design is motivated: a targeted study of why the residual chemistry
   term is harder to generalise than the logP term (e.g. feature ablation of the cycle-snap inputs
   feeding `s`), again as a new preregistration — not a continuation of the current loss
   configuration.

## What is closed

* This round stops with the current artifacts. No second seed, no third arm, no new fold, no
  parameter search, no loss-weight tuning, no dictionary/path resurrection, no official-valid/test
  confirmation.
* No official-test is opened. The result is an internal-dev signal only.
