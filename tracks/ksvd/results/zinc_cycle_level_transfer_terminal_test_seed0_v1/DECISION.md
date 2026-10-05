# DECISION — zinc_cycle_level_transfer_terminal_test_seed0_v1

**Verdict: `CYCLE_LEVEL_TRANSFER_BOUGHT__TEST_SUPPORTED__STOP_CYCLE_CHANNEL`.**

1. **BOUGHT (train-only gate PASS, all 4 conditions).** The discrete k-level head D carries
   genuine cross-exact-class signal the continuous regression R misses: pooled OOF overall
   c-MAE gain +0.004677, k=−2 gain +0.7345, k=−1 improvement +0.0626, k=0 improvement +0.0025;
   held-class level accuracy 0.92–1.00 on k∈{−2,−1}. The honest failure boundary is the extreme
   singleton levels k ≤ −3 (pooled n=7, all negative OOF) — these do not extrapolate and are
   not claimed.

2. **TEST-SUPPORTED (unconditional terminal report, post-freeze read).** C (H + D_full on the
   UNSEEN branch only) beats H on the official test: y_cal 0.081174 vs 0.084719, gain +0.003545,
   paired 95% CI [+0.000153, +0.009389] (lower > 0). On the 20 unseen test rows D_full decoded
   the exact correct level 20/20 (q c-MAE 0.1965 → 0.0000; k=−2 group y_cal 0.4324 → 0.0742).
   On valid the same comparison was single-row dominated and CI-crossing (valid:0935, 95.1%
   share) — the test result is the broad confirmation, reported unconditionally as authorized.

3. **STOP the cycle channel.** After C, the test cycle error e_c is 1.02e-05 (vs body e_g
   0.0809). The cycle channel is exhausted; every further y gain must come from the chemistry
   body error h−g and its cancellation, not from any cycle head (continuous, prototype, or
   level).

4. **Fixed neighbors, closed.** H−B on test is +0.000388, CI [−0.000262, +0.001205] — the
   prototype transfer's valid gain (+0.0207, single-row valid:0172 k=−6) does not generalize
   broadly; H stays a documented tail repair. B−SUM_Q is CI-supported on both splits
   (test +0.007512 [+0.001810, +0.012664]) — component supervision is confirmed out-of-sample.
   SUM_Q stays the reference; B stays the working reference for the body line.

## What NOT to do next

- No further cycle-head variants (ensembles, class weights, temperature, extra decoders,
  re-tuned routing/thresholds) — the channel is at 1e-05 error on test.
- No reopening of official test for selection; this round's test read was terminal-only under
  the frozen manifest. Any next test read requires a new preregistration.
- No changes to the frozen gate/key/rules to rescue the k ≤ −3 singleton levels; their failure
  is an honest extrapolation boundary (7 train rows total).
- No re-confirmation of COMP/SUM/Q or new seeds; no body retraining from this endpoint.

## Revisit if

A new preregistration attacks the chemistry body error h−g (test MAE 0.0809, 99.98% of C's
remaining y error) while preserving the now-exact cycle channel, and carries C as the matched
reference with paired seeds and a train-only purchase gate of the same strength.
