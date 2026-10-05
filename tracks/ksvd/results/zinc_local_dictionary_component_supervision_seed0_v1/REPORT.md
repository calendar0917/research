# REPORT — zinc_local_dictionary_component_supervision_seed0_v1

**Round**: shared sparse IHT local tuple dictionary (`D_COMP`) vs
parameter-matched SiLU local MLP (`M_COMP`), both seed 0, 240 epochs, 15,120
steps, COMP supervision (`L = MAE(hat_g,g) + 0.5·(MAE(hat_ell,ell)+MAE(hat_s,s))`),
new internal fold `rng(20261006)` 8000/2000, all fitted statistics refit on the
new fit rows. Two formal arms only, executed in parallel on two allocated A100s
(res-2, res2-cu124, distinct GPU UUIDs `ba05543b…` / `afab53a3…`).

## Direct answers

1. **Best internal dev g-MAE**: `M_COMP` dev G0(k=0) **calibrated 0.088062**
   (raw 0.090340; overall cal 0.089313). `D_COMP`: G0 cal 0.090108 (raw 0.099279;
   overall cal 0.090451). D−M gains (gain = MAE(M_COMP) − MAE(D_COMP), positive
   = dictionary improves; paired 1000×, seed 20261006, same per-graph indices):
   **G0 cal −0.002045 CI [−0.005525, +0.001502]; overall cal −0.001138 CI
   [−0.004850, +0.002301]; G0 raw −0.008939 CI [−0.012561, −0.005426];
   overall raw −0.007946 CI [−0.011362, −0.004266]** — the dictionary is
   *worse*, significantly on raw, directionally on calibrated.

2. **Did the dictionary meet the gate under COMP supervision? No — the result
   is negative, not equivalence and not mere imprecision.** All five
   dictionary-candidate conditions fail (points negative; the two raw CIs lie
   entirely below zero). The swapped-role M gate passes only its two raw
   conditions, failing the +0.003 cal thresholds and the CI condition — so M
   does not symmetrically win either. Classification: **`DIRECTIONAL_NOT_
   CONFIRMED`** (same-direction point values favouring the MLP, no gate
   passed). Not `LOCAL_EQUIVALENCE`: the raw CIs lie far outside [−0.003,
   +0.003] and the G0-cal CI lower bound (−0.005525) is below the band. Sensitivity
   (drop M's worst dev row, once): G0 cal gain −0.001411, overall −0.000526 —
   the direction survives; not a one-row artifact. Fit-side agrees (fit cal
   D 0.031079 vs M 0.027889).

3. **The dictionary genuinely participated; the difference is a performance
   difference, not a mechanism or bias artifact.** D mechanism evidence: per-tuple
   IHT codes exactly 8-sparse (8.0 nnz of 64), 61–62/64 atoms live, dictionary
   drift 71.6 (init L2 ≈ 89.4), `W_loc` norm 8.76, injection RMS 0.415, nonzero
   task gradients into `D_loc_raw` from epoch 40 on (epoch 1 = 0 is the expected
   W_loc=0 null). Zeroing the local injection collapses D's G0 cal 0.090108 →
   0.493916 (M: 0.088062 → 0.548597) — the channel carries real signal in both
   arms. The performance gap tracks the **s component**: dev raw MAE s − M better
   by 0.006832 (G0: 0.007253); ell — D marginally better by 0.000811. Biases
   (b_g D −0.042809 vs M −0.022461) and cancellation are comparable (triangle
   gap raw dev G0: D 0.031947 vs M 0.034231; opposite-sign 48.9% vs 51.1%), and
   calibrating moves both arms the same direction, so the gap is not a
   bias/cancellation artifact. Still not attributable: single seed, single fold,
   and the raw→cal gap shrink (0.0079→0.0011 overall) means part of the raw
   deficit is a larger fit-side median offset, not pure signal quality.

4. **Support for luyin19**: none for the dictionary mechanism. In this fixed
   interface (same J incidence, same skeleton, same COMP supervision, matched
   parameters), the shared joint tuple dictionary shows **no** transferable
   chemical advantage over a matched MLP — this *weakens* the "shared joint
   tuple dictionary" line as the current lever. Strictly bounded: both arms use
   the J incidence, so this says nothing about J vs marginal aggregation; and
   it does not test the original "structure-attribute cross/shared-structure-
   dictionary" claim (luyin19's own direction) — only this local-encoder-vs-MLP
   comparison under component supervision.

5. **Next investment**: stop the fixed local IHT dictionary recipe in this
   interface (no rescue, no seeds, no sparsity/step/width tuning — forbidden
   and unpurchased). Keep the COMP+matched-local-MLP recipe as the working
   internal reference (best internal dev G0 cal 0.088062). The next legitimate
   question is luyin19's actual subject — how the structure/semantics fusion is
   built (structure-attribute crossing), not which local encoder sits under
   the current fusion; and the body error channel, not this local branch.
   No full-train/official-valid confirmation is purchased (that requires D to
   win, which it did not).

## Main table (internal dev, 2000 rows; k counts 1931/60/7/2)

| arm | b_g | fit cal | dev raw | dev cal | dev G0 raw | dev G0 cal | gap (dev−fit cal) |
|---|---|---|---|---|---|---|---|
| D_COMP | −0.042809 | 0.031079 | 0.099704 | 0.090451 | 0.099279 | 0.090108 | +0.059372 |
| M_COMP | −0.022461 | 0.027889 | 0.091759 | 0.089313 | 0.090340 | **0.088062** | +0.061424 |

Per-k dev cal MAE (n; Σ|err|/N contribution): D k=0 0.090108 (1931; 0.086999),
k=−1 0.097386 (60; 0.002922), k=−2 0.119754 (7; 0.000419), k≤−3 0.110984 (2;
0.000111); M k=0 0.088062 (0.085024), k=−1 0.115872 (0.003476), k=−2 0.220087
(0.000770), k≤−3 0.041917 (0.000042). Contributions sum to the overall MAE
exactly (identity ≤1.5e-17). Note M is better on k=0 but worse on the rare
k=−1/k=−2 rows — the overall gap is dominated by the k=0 bulk, not rare rows.
Both arms' internal dev g-MAE (0.088–0.090) is an **internal fold
chemistry-component score** — not official-valid y-MAE, not a deployment
threshold, not comparable to historical valid/test y-MAE (different split,
different target, no cycle channel).

## Component diagnostics (raw, never separately calibrated)

| arm | ell fit/dev | s fit/dev | ell G0 dev | s G0 dev |
|---|---|---|---|---|
| D_COMP | 0.025963 / 0.049163 | 0.040232 / 0.082931 | 0.048979 | 0.082247 |
| M_COMP | 0.024887 / 0.049974 | 0.028106 / 0.076099 | 0.049578 | 0.074993 |

D−M: ell dev raw delta +0.000811 (D better), s dev raw delta −0.006832 (M
better; G0 −0.007253). Cancellation dev G0: triangle_gap raw D 0.031947 /
M 0.034231 (both ≥ 0; favourable cancellation present in both, similar size);
opposite-sign 48.9% / 51.1%. `e_g_raw = e_ell + e_s` identity exact. Component
MAEs never sum into a cal-g budget; one bias for the total g only.

## Frozen interventions (dev, native fit bias; sensitivity-only)

| arm | baseline G0 cal | zero injection | mean root code | J→I incidence |
|---|---|---|---|---|
| D_COMP | 0.090108 | 0.493916 (mean\|Δpred\| 0.485) | 0.437877 (0.430) | 0.090115 (0.0031) |
| M_COMP | 0.088062 | 0.548597 (0.540) | 0.541796 (0.532) | 0.088046 (0.0003) |

Zeroing the local injection destroys both arms (synergy destruction included —
not an information share). The J→I switch is nearly inert in both converged
arms (mean |Δpred| ≤ 0.003) — at the converged state this interface does not
depend on the joint-vs-marginal incidence distinction; this is a frozen-model
sensitivity statement, not evidence about training-time inductive bias.

## Verification summary (all pass)

Fold hashes frozen (`734d27f2…` / `cb5f49dc…`); dev-label shuffle leaves fit
constants exactly unchanged; shared initial tensors identical (max abs 0.0);
`A = D_init.T`; `W_loc = 0` both; reader 39→2 split-sum equals the original
39→1 head (3e-8 float rounding); forward label independence 0.0; root codes vs
production reference operators 0.0 (both arms); IHT per-tuple nnz ≤ 8; bridge
is the matched MLP in both arms; schedule hash `7b11a529…` with identical
position + global gid streams across arms; build RNG states equal; spectral
estimate and eval loader RNG-neutral; smoke (1 run, 4 steps/arm, discarded):
W_loc task gradient at step 1, local task gradients live by step 4;
dev-eval fit replay vs training-time fit predictions ≤1e-5 (both arms);
bootstrap witnesses same→0, swap→mirror; contribution add-back exact.

## Deviations (engineering; no scientific definition changed)

1. First training pair (jobs 56046/56047, commit `af6368f`): both arms finished
   all 15,120 epochs and saved states + fit predictions, then the delivery
   layer crashed writing the curve JSON (`dict(list)` in `write_json`). No dev
   score existed or was read. Fixed (one line), redeployed, and both arms were
   re-run from scratch once each (deterministic recipe unchanged; fragments
   discarded; the re-runs reproduced the same trajectories — the formal pair
   is jobs 56048/56049, commit `d1fe445`).
2. First dev-eval crashed on a missing argument in the cancellation table
   *before any metric was computed, printed or written* (raw dev predictions
   had been saved only). Fixed deterministically and re-run; roster was frozen
   and committed before both attempts.
3. GPU budget: 4 jobs total ≈ 0.78 allocation-GPU-hours (two failed-fragment
   jobs ~0.35 h + two formal jobs 0.230 + 0.192 h) ≤ 1.0 h limit. The failed
   fragments are counted, disclosed, and excluded from the comparison.
