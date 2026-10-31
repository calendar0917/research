# Decision — LCRC Lipophilicity v2

**Date:** 2026-10-31  
**Protocol:** `lcrc-lipo-mechanism-generalization-ssl-v2`  
**Primary evidence:** Fit2352 3-fold scaffold CV, seeds 0/1; Monitor504 external check on frozen predictions (not an independent blind test).

## Verdicts

| Hypothesis | Verdict | Decision |
|---|---|---|
| **H1** A vs capacity-matched E | **PROMISING** | Keep A as the mechanism candidate. Capacity matching does **not** explain A's advantage: CV +0.04510 RMSE, all folds favor A; Monitor +0.23219. E overfits and is a failed control. |
| **H2** fixed supervised recipes | **PROMISING** | Adopt **R3** (peak LR 3e-4, wd 1e-3, dropout 0.10) for A. A's gain over R0 is +0.07578 mean CV, positive on all 3 folds and all 8 seed-fold pairs. C's +0.02836 is Fold-0-only and weaker (seed-1 gain +0.00810); do not over-claim C. |
| **H3** Fit-only SSL | **NOT_PROMISING** | Drop the SSL branch for this round. The pretext gate passed (heldout CE 0.0308/0.0273 vs prior 1.5377), but downstream SSL is worse by −0.03147 CV and −0.04898 Monitor. A learned pretext is not a useful representation here. |
| **H4** LCRC-A vs strong CPU chemistry | **DESCRIPTIVE TIE** | No material gap. A seed-averaged Monitor 0.76865 vs XGB-P3 0.77878 (delta −0.01013, 95% CI [−0.11159, +0.11794]); per-seed A average 0.79162 is 0.01284 worse. Keep the CPU XGB-P3 baseline as a mandatory reference. |

## Bottom line

1. **A's advantage survives effective-parameter capacity matching.** The matched E has 247,049 effective parameters vs A's 247,106, no message passing, and a dead S2 slot; it is substantially worse on every fold and much worse on the Monitor. The result supports a mechanism claim about A's S2 joint binding, not merely parameter count.
2. **Supervised regularization is the practical win.** R3 improves A by ~0.063–0.088 RMSE per fold and is the recipe to carry forward. The gain is larger than the capacity effect and much larger than anything SSL produced.
3. **SSL is a negative result.** The masked-root pretext is easy to learn but does not transfer to the supervised local representation. Do not add SSL epochs, larger heads or more pretext data in the next round without a different pretext/transfer design.
4. **The CPU baseline is essentially tied with A on the external Monitor.** A is not a decisive chemistry-ML advance over a well-tuned XGB-P3 baseline. Any claim of practical superiority needs a new external test and a pre-registered materiality margin.
5. **Mechanism diagnostics are consistent with A using joint S2 pairings.** A is sensitive to an independent S2 swap (0.296 mean absolute change) but nearly invariant to a full path swap (1.8e-7); C/E are invariant to all S2 swaps. A also depends on global features (0.423 mean absolute change when replaced by the train mean). These are mechanism clues, not proof of causal sufficiency.

## What to do next

- **Run a new external-generalization round** with a pre-registered blind test, not another Monitor re-use. The current Monitor is a single 504-molecule check and was already used.
- **Keep R3 and the A architecture fixed** for that round; do not retune recipes on the new test.
- **Test the S2 mechanism directly**: a targeted S2 joint-binding ablation or a synthetic pairing stress test, with a pre-registered effect size. The current diagnostics are single-seed and descriptive.
- **Retain the CPU XGB-P3 baseline** and report the per-seed and seed-averaged gaps side by side. The H4 result is a tie, not a win.
- **Do not pursue capacity-matched E or SSL fine-tuning** as the next main line; both are negative controls in this round.

## Caveats

- Two primary seeds (AMENDMENT-001), not three. Fold-0 R0 has a 3-seed sensitivity check (A 0.80110, E 0.88799, delta +0.08689), but folds 1/2 and the full retrain do not.
- E is an effective-parameter construction (dead S2 slot + fixed 70,535-parameter replacement), not a literal 529-wide LocalMLP.
- C-R0 was not run on folds 1/2, so H2's C arm is Fold-0-only. H2's robustness comes from A.
- The first Monitor execution was invalid due to a prediction denormalization bug; it is quarantined under `analysis/audit/monitor_invalid_normalization/` and not used. The corrected predictions were generated without labels, but the invalid execution had already computed aggregate Monitor metrics, so the valid evaluation is a corrected external check, not a pristine first look. This is disclosed in AMENDMENT-003.
- Monitor H1 magnitude (+0.232) is much larger than CV (+0.045); the Monitor contrast is A-R3 vs E-R0 while the CV test is A-R0 vs E-R0, so the magnitude includes the R3 recipe advantage. Direction agrees; magnitude should not be over-interpreted.
