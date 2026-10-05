# REPORT — `zinc_chemistry_component_supervision_seed0_v1`

## Directly answering the five headline questions

1. **Component formula / identity.** Yes — the formula is identical in spirit and bytes to the old
   `g`. `ell = (logP − MU_LOGP)/sigma_logP` with `MU_LOGP = 2.4570953396190123` and the frozen
   fit-only `sigma_logP = 1.434428173759835`; `s = g − ell`; `s_SA = (SA − mu_SA)/sigma_SA`
   (reference only). The `g = ell + s` identity holds to `4.4e-16` in float64 and `g = y − c` to
   `0.0`. So `s` is **not** named "pure SA": on fit-G0 `epsilon = s − s_SA` has MAE `0.001512` and
   max `0.008953`, both above the pre-frozen interpretation threshold (MAE ≤ 1e-3 and max ≤ 1e-2);
   the SA-interpretation flag `approx_SA_component_allowed = False`. The gap widens with cycle
   rarity (k=−1 MAE 0.0083, k=−2 0.0168, k≤−3 0.1074), so the residual term departs from SA
   exactly where the cycle-snap component is large.

2. **COMP vs SUM calibrated g-MAE.** Component supervision is a clear internal-dev candidate:

   | arm | dev G0 cal | dev overall cal | fit overall cal |
   |---|---:|---:|---:|
   | SUM (baseline) | 0.097945 | 0.099852 | 0.030805 |
   | COMP | 0.087972 | 0.089979 | 0.030259 |

   Paired bootstrap (1000 draws, seed 20261008, shared indices): G0 cal gain `+0.009974`
   CI `[0.006459, +0.013493]`; overall cal gain `+0.009873` CI `[0.006440, +0.013551]`. All
   three frozen gate conditions pass → **`COMPONENT_SUPERVISION_CANDIDATE`**. Raw gains point the
   same way (G0 `+0.009573`, overall `+0.009413`), so the result is not calibration-only, though the
   gate itself is calibration-based.

3. **Component fit/dev.** Both components are fit-adequate (`FIT_ADEQUATE`, ratio
   `0.044` / `0.047` vs the `0.20` constant-predictor reference, far below noise). On dev-G0 the
   **`s` (residual chemistry) component is the worse, gap-leading term**: `s` dev-G0 MAE `0.083`
   vs `ell` dev-G0 MAE `0.054` (ratio 1.54 ≥ 1.5), and `s` has the larger fit→dev gap (0.038→0.083
   vs 0.034→0.054). Conclusion: `SINGLE_COMPONENT_TARGET`, the residual chemistry component is the
   next research object. (This is evidence in the *new COMP model*, not a causal decomposition of
   the original g-only model.)

4. **Error cancellation.** Substantial. On dev-overall, `mean|e_ell| = 0.0552`, `mean|e_s| = 0.0850`,
   but `mean|e_g| = 0.0902` — the two component MAEs sum to `0.1401` yet the total `g` error is
   only `0.0902`. Opposite-sign errors on `61.9%` of dev rows; `triangle_gap = 0.0499 > 0`
   (favourable cancellation). Fit the gap is similar (`0.0726` summed vs `0.0305` actual). The
   global `g` bias is reported separately (`bias = −0.00455`) and is never decomposed into a
   component. Because cancellation inflates the summed component MAEs, you cannot add `ell`-MAE and
   `s`-MAE and treat the sum as the `g` error budget.

5. **Buy status.** Performance candidate **and** a directed component diagnosis. The gain is real
   and the gate is passed, so a frozen confirmation design is the only authorized next step; the
   component result also points one new design: a targeted study of the residual chemistry term
   (`s`), which is the dev-gap leader. Nothing is auto-queued; this round is closed.

## Scope / limitation (not a deployed y result)

`g`-MAE is the internal target; the 2000 dev rows were trained on by the old fold and the new fit
overlaps the old fit (partition-stability replication). No official-valid/test data were loaded or
scored.
