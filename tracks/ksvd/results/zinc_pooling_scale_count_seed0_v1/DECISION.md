# DECISION — `zinc_pooling_scale_count_seed0_v1`

**Frozen classification: `NO_CANDIDATE` (rule 5).** The round closes the fixed
pooling-scale/count-accessibility variant. No confirmation round, no official-valid/test read.

## What the result is

1. **`N` (mean/count) is rejected.** Against `C` (sum/count) it loses on all four endpoints with
   CI-separated negative point estimates (G0 cal `−0.011631`, CI `[−0.015971,−0.007671]`;
   overall cal `−0.011797`; both raw endpoints also negative with CIs below zero). Against the
   read-only source `M` it loses likewise (`−0.0079…−0.0092`). Frozen gate 0/5 for both
   comparisons. Because C and N carry the same information and identical init/RNG/schedule, this
   is a scale/function-accessibility verdict for this skeleton and recipe, not an information
   claim: dividing both moments by the actual per-graph/per-bucket `n` (block RMS ~22× smaller at
   init) produced better fit (`0.025735` vs `0.028201`) but worse dev (`0.107541` vs `0.095743`).
2. **Count visibility alone (`C` vs `M`) is not purchased.** It passes 3/5 frozen conditions:
   calibrated G0 `+0.003722` CI `[+0.000460,+0.007317]`, overall cal `+0.003560` CI
   `[+0.000292,+0.006764]`; but raw is negative (G0 `−0.001053`, overall `−0.001525`, CIs
   crossing zero), i.e. the calibrated edge is a fit-median-bias shift (`C` bias `−0.031017` vs
   `M` `+0.013944`), not better raw prediction. In addition the `C`/`M` comparison is
   **cross-regime** (local CPU vs source A100) because no GPU was available, so it could not have
   supported a clean count-accessibility claim even at 5/5.
3. **Identities hold.** Both arms: source-identical init (max abs 0.0), frozen RNG and schedule,
   actual gid stream `69187f13…` equal to the source round, 15,120/15,120 steps; source `M`
   replay `9.5e-7` and all four source main metrics reproduced to 0.0; C/N single-file CPU
   replay 0.0. The mechanism evidence (block RMS, count/moment witnesses, scale bins, reader
   gradients) is complete and consistent with the frozen protocol.

## Action

1. **Close** the sum-vs-mean pooling-scale variant and the count-visibility interface change as
   specified: do not resubmit `N`, do not scan scales/coefficients, do not run a confirmation on
   this result, do not open official-valid/test.
2. Keep the source fresh-fold `M` as the working reference; `C`'s calibrated numbers
   (`G0 cal 0.093661`, overall cal `0.095743`) are recorded but are **not** a performance
   candidate because raw did not improve and the comparison is regime-confounded.
3. Any future buy of this direction requires a **regime-matched** run (same host/device as the
   reference, ideally GPU `res-2`) showing `C` raw > `M` raw under a pre-registered gate, or a
   new discriminative hypothesis. Merely re-running with a different scale/threshold does not
   qualify.

## Uncertainty statement

The `N`-vs-`C` degradation is CI-separated in this frozen comparison and is the only clean
same-regime result, but it is one seed, one soup, no training-randomness coverage; it does not
show that mean pooling is universally worse, that the model lacks input information, or where
the residual bottleneck lies. The `C`-vs-`M` calibrated signal is not confirmed and may be
calibration/regime-driven. No 0.09 or deployable-`y` claim is made.

## Process note (budget / regime)

The res-2 A100 jobs (55976/55977) never started (pending `(Resources)`) and were cancelled; `res`
GPUs are unavailable (NVML error, `cuda_available=false`). With explicit user authorization the
two frozen trajectories were executed on local CPU (16 cores, 2×8 threads, FP32, no AMP/DDP) at
commit `ddd5bd394fe2`. This exceeded the pre-registered 90-minute compute-stop and 120-minute
wall budget; both facts are recorded in `EXECUTION.md`. The CPU/GPU replay-equality item could
not be performed (no GPU); CPU single-file replay is exact.
