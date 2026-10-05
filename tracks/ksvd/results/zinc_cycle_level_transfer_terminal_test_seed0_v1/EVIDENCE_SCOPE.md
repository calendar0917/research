# EVIDENCE_SCOPE — zinc_cycle_level_transfer_terminal_test_seed0_v1

## What this round's evidence supports

1. **A discrete train-vocabulary k-level head (D) learns cross-exact-class structure that a
   continuous c regression misses** — established TRAIN-ONLY by 3-fold class-grouped OOF on the
   10000 train rows (held classes never in fit): pooled OOF c-MAE gain +0.004677 overall,
   +0.7345 on k=−2, +0.0626 on k=−1; held-class level accuracy 0.92–1.00 for k∈{−2,−1}.
   Scope: the fixed vocabulary K=7 from train; the fixed seed-0 recipe (300 epochs, batch 128,
   Adam 1e-3/1e-5, soup 296–300); the frozen T25 representation and prep. Single seed, single
   decode rule (median-level ≥0.5 cumulative), no class weights.

2. **C = H with D_full on the UNSEEN branch improves over H on the official test**
   (1000 rows, frozen-manifest terminal read): y_cal 0.081174 vs 0.084719, gain +0.003545,
   paired 95% CI [+0.000153, +0.009389]. Scope: this official split; the gain concentrates on
   the 20/1000 unseen rows (all 20 levels exactly right); the k ≤ −3 singleton levels are a
   known extrapolation failure (n=7 pooled OOF, all negative) and are not claimed. On the
   exposed valid the same comparison was single-row dominated (valid:0935, 95.1% share) with CI
   crossing zero — the valid endpoint alone would NOT have supported the mechanism.

3. **The cycle channel is exhausted after C**: test e_c MAE 1.02e-05 vs body e_g 0.0809
   (99.98% of C's remaining y error); triangle_gap ≥ 0 on both splits (C valid 6.2e-05, test
   2.0e-05).

4. **B − SUM_Q (component supervision) is CI-supported positive out-of-sample** on both
   splits (valid +0.005247 [+0.000133, +0.009954]; test +0.007512 [+0.001810, +0.012664]).
   Scope: the single frozen seed-0 pair; no ensemble claim.

5. **H − B does not generalize broadly** (test +0.000388, CI crosses 0; the valid +0.0207 was
   the single valid:0172 k=−6 row, a regime absent from test).

## What it does NOT support

- No claim about k ≤ −3 extreme levels (12 train rows total, 7 in pooled OOF, all failed OOF
  extrapolation; test has none).
- No claim beyond the single seed-0 recipe and the single median-level decoder; no ensemble,
  no class-weighting, no temperature variants (never run).
- No claim that C < 0.09 on any future split; the valid endpoint of the same system is 0.096150.
- No selection claim from the test read: the roster and gate were frozen before it; the test
  numbers are terminal-only reporting.
- No body/chemistry claims: COMP/SUM bodies, prep, prototype table, biases are reused frozen
  from prior rounds and only replayed here (bit-exact).

## Frozen reuse (read-only, SHA-verified)

SUM/COMP soup states, Q soup state, full-train prep/targets, calibration biases, prototype
table + b_H, T25 caches — zinc_component_supervision_fulltrain_confirmation_seed0_v1 (commit
947d2837…) and zinc_cycle_prototype_transfer_cpu_v1 (commit 5a510d9). Official split mapping
from zinc_full_decomposition_valid_test_confirmation_v1 (engineering only; no old test values
used for any decision).
