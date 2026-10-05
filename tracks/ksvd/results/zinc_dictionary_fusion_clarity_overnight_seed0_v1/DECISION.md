# DECISION — `zinc_dictionary_fusion_clarity_overnight_seed0_v1`

**Decision date**: 2026-10-06 ~04:10 CST.

## What was decided (all by gates frozen at Stage 0, never by dev peeking)

1. **Stage-4 purchase: REJECTED.** Neither Stage-3 candidate passed the frozen
   purchase minimum on split B (G0 cal ≥ .002, overall cal ≥ .001, both raw
   gains > 0): B_F_D_RAND reversed on all four metrics; B_F_M_I kept its
   calibrated gains (+0.0051 G0 cal, CI > 0) but its raw gains are
   significantly negative (−0.0176, CI [−0.0214, −0.0137]) — the advantage is a
   per-arm median-bias (calibration-offset) effect, not per-molecule accuracy.
   No full-10k arm was trained; official-valid was never read.
2. **The separate-structure×semantics construction (F) is closed at this
   scale/interface** (matched bodies, COMP supervision, seed 0, this fold
   family): do not re-purchase without a new preregistration that changes the
   *scale* (e.g. matched-parameter families at several widths) or the
   *supervision*, not the split/seed.
3. **The J_D-vs-J_M (dictionary-vs-MLP) comparison is downgraded to
   noise-scale**: the source round's and this round's matched runs give
   opposite sign verdicts. Any future claim about this pair must either
   (a) be prefaced with the noise bracket ±0.005 cal / ±0.012 raw from two
   matched runs, or (b) come from ≥3-seed matched ensembles under a fresh
   preregistration.
4. **Do not quote the historical held-out numbers of
   `zinc_component_supervision_fulltrain_confirmation_seed0_v1` (valid) or
   `zinc_cycle_level_transfer_terminal_test_seed0_v1` (valid replay + terminal
   test body inference) until they are re-scored once under the corrected
   held-out incidence loader** (ERRATA; `valid_structure_check.py` in this
   round's directory; this round's own module contains the corrected reader,
   unused because Stage 4 was not purchased).

## Follow-up questions (at most two)

1. **One re-scoring pass for the historical held-out artifacts** (CPU-only,
   no training, no new labels): re-evaluate the frozen fulltrain-round
   bodies on valid **and** the cycle round's frozen roster on valid+test with
   each held-out graph's own incidence structure, and re-issue their
   REPORT/DECISION deltas. The measured magnitude on valid is ~0.005 y-MAE;
   the cycle round's C-vs-H terminal test gap (~0.015 in y units) is large
   enough to likely survive, but its test numbers must be re-quoted.
2. **If the dictionary direction is ever re-opened**: does a frozen/random
   local dictionary help only in *larger* matched-parameter regimes (the
   signal, if any, may be width-dependent), and does the F_M_I calibration
   offset (large fit-side median bias) indicate a semantic-branch scale
   mis-match that a *single* fit-side affine (not per-arm) would remove? Both
   require a fresh preregistration; neither is licensed by this round.

## Stop conditions honored

No new message passing/Transformers; no hyperparameter/sweep/width scans; no
extra seeds; no dev-score-driven epoch/model selection; no node revival,
ring-tail weighting, bridge/reader capacity changes; official-test never
opened; warm starts never used; failed runs never converted into parameter
search; GPU allocation stayed inside ≤8 GPU-h (≈2.7 h actually used, 12 formal
arm trajectories ≤ 14); every remote job through `rr` (res-2, pool
res2-cu124, 1 GPU/4 CPU each); git commits local only (no push/merge).
