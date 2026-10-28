# DECISION — LCRC-v1

**Verdict:** `STOP_NO_STABLE_GAIN`  
**Test:** never loaded in this round.  
**Date:** 2026-10-28

## Decision

Stop LCRC-v1 after the pre-registered 6-run program (A/R/N × seed 0/1). Do not read the official test, do not add seeds, and do not launch patch experiments (head scaling, weight-decay removal, re-initialization, category attention, etc.).

## Why

1. **No stable arm advantage.** Two-seed means: A 0.078543, R 0.079032, N 0.080075; historical DCC-L four-seed mean 0.075990 (not retrained). A_s0 is nominally better than DCC-L seed0, but A_s1 is worse than DCC-L seed1; the two-seed mean does not reach DCC-L.
2. **R vs N flips.** R−N = −0.003950 at seed0 and +0.001866 at seed1. The mean (−0.001042) is not a consistent gain, and the seed0 confidence interval includes zero.
3. **The relation channel is functionally off.** R's `Δ_rel` is nearly constant across molecules (range ~2×10⁻⁵); at fixed weights, replacing true correspondence by the mean-field content changes valid MAE by ≤1×10⁻⁸. The zero-init head collapsed to a constant bias. A fresh ridge head on hidden or raw relation content also fails to produce a meaningful valid gain.
4. **Test gate fails.** The pre-registered rule requires consistent R<N across seeds, mean R−N ≤ −0.001, and R valid mean ≤0.079 (or a clear win over DCC-L). Consistency is false and R mean is 0.079032 (>0.079); even rounding would not fix the consistency failure.
5. **No <0.060 signal.** Best model is 0.076582; even with the fixed worst-10 set to zero the overall would be 0.061892. Tail/row172 remains the dominant error.

## What would justify a new round

A new pre-registered round (not a patch) would need a different relation-readout hypothesis — for example distributional/quantile summaries, learned-query pooling, or pair-level auxiliary supervision — evaluated with at least 3 seeds, tail-stratified reporting, and a clearly specified mechanism probe. The local A body could separately be confirmed with 3–4 seeds if the goal is a no-junction baseline; no test should be read until the pre-registered valid gate is met.

## Artifacts

- `REPORT.md` — full results and five-question analysis
- `analysis/seed_summary.{md,json}` — test gate
- `analysis/mechanism.json` — head-zero / mode-swap / scale probes
- `analysis/relation_probe.json` — post-hoc relation-content ridge diagnostics
- `analysis/main_table.{md,json}`, `analysis/contrasts.{md,json}`, `analysis/groups.{md,json}`
