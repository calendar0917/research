# REPORT — `zinc_local_tuple_dictionary_vs_mlp_seed0_v1`

**Question.** Keeping the real local tuple structure `(phi_v, a_v, a, t)`, the real correspondence
aggregation `e(v) = Σ w_J(v,t,a) f(x)`, the injection location and the original skeleton unchanged:
does the current local IHT dictionary `f_D` bring a generalisation gain on the chemical target
`g = y − c` relative to a same-parameter-count local nonlinear projection `f_M`
(row-normalised 125→64 Linear + SiLU)?

**One-line answer.** `INCONCLUSIVE`, with a directional M_J advantage: M_J is better than D_J on
all four dev endpoints in point estimate (calibrated G0 `+0.002200`, overall `+0.002526`, raw G0
`+0.002245`, overall `+0.002684`) but every paired CI crosses zero and the frozen
`MLP_LOCAL_SUPPORT` thresholds are not met; vs the historical B the calibrated point gains exceed
`+0.003` but the G0 calibrated CI still crosses zero, so no performance candidate is bought.

## Opening table (the five closing questions)

| # | question | answer |
|---|---|---|
| 1 | Same tuple input: does the local dictionary win, lose, tie or stay unresolved vs the fixed nonlinear encoder? | **Unresolved (INCONCLUSIVE), directionally M_J.** M_J better in point estimate on all 4 endpoints (G0 cal `+0.002200` CI `[−0.001564,+0.006029]`; overall cal `+0.002526` CI `[−0.001649,+0.006467]`; raw G0 `+0.002245`; raw overall `+0.002684`) but frozen gate demands G0 cal ≥ +0.003 with CI>0 and equivalence is not shown (CIs wider than ±0.003). |
| 2 | Is there a frozen-gate performance candidate vs the old B? raw vs cal? Mostly from bias? | **No performance candidate.** M_J dev cal overall `0.099381`, G0 `0.098216`; raw overall `0.101414`, G0 `0.100434`. B→M_J: overall cal `+0.004336` CI `[+0.000264,+0.008757]`, G0 cal `+0.003659` CI `[−0.000438,+0.008602]`, G0 raw `+0.006817` CI `[+0.002339,+0.011765]`, overall raw `+0.007542` CI `[+0.003279,+0.012083]`. Not mainly a bias artifact: raw gains are larger than cal gains, and the reported fit-median bias (`−0.015250` vs B `−0.031378`) reduces rather than creates the calibrated gain; but the frozen G0 cal CI still includes zero. |
| 3 | Are both encoders active, and how sensitive are they to the real pairing right away? Which interventions prove only dependence? | **Both active.** D_J: exactly 8/64 sparse per tuple, `D_loc` drift 0.756, `W_loc` norm 9.18. M_J: `A_raw` drift 0.721, `W_loc` norm 7.62, 0 dead code dims, effective rank 4.53. Immediate real-vs-marginal switch: D_J mean\|Δpred\| `0.003382` (dev raw MAE `+1.5e-4`, cal `+8.4e-5`), M_J `0.000260` (dev raw MAE `+2.8e-6`, cal `+7.0e-6`) — ~13× less sensitive for M_J. Zero-ablation (D `0.1019→0.5485`; M `0.0994→0.6229`) and fit-mean replacement (D `0.1019→0.4423`; M `0.0994→0.4845`) prove **dependence only**, and include a large systematic shift plus co-adaptation breakage; they do not estimate an information share. |
| 4 | Which small part of `luyin19` does this support, and what is not tested? | Supports: the real root-local structure–atom–bond tuple interface carries a load-bearing signal, and a same-capacity per-tuple nonlinear encoder **at least matches (directionally beats) the IHT dictionary** at this budget, so the dictionary form is not required for this interface here. Not tested: the J>I correspondence claim (still INCONCLUSIVE), ring/cycle relief, message passing/global structure, multi-seed variance, official-valid/test, and the pre-registered incidence-permuted marginal control. |
| 5 | What is closed, what is kept, why will the next spend not sweep this encoder family on this dev again? | Closed: the fixed dictionary-vs-SiLU same-capacity local-encoder comparison on this interface/fold — no practically separable encoding gain; this config is frozen. Kept: the local tuple interface as a candidate carrier and the D_J/B references. Next evidence must come from **independent folds/seeds or the permuted-marginal control**, not another encoder on the same repeated dev; the remaining uncertainty is seed/optimisation noise vs a genuine G0-specific effect. |

## 1. Frozen setup and contract checks

Protocol, method contract, evidence scope and errata were committed before the formal run
(`PROTOCOL.md`, `METHOD_CONTRACT.md`, `EVIDENCE_SCOPE.md`, `ERRATA.md`). The only changed block is
the local encoder: `A_raw[64,125]` (8,000), `A_bar = row_l2_normalise`, `f_M = SiLU(x@A_bar.T)`
per tuple, then real `pair_wJ` aggregation, then the unchanged zero-init `W_loc[342,64]` injection.
Parameter audit by name: body 184,667 + bridge 82,944 + A 8,000 + W_loc 21,888 = **297,499**; no
idle `D_loc_raw` is kept. The shared init, schedule and dropout stream are the historical ones
(all checks in `EXECUTION.md` §1); `kappa_M = 2.016961` (label-free fit-only, `kappa_M.json`).

## 2. Performance

Dev calibrated g-MAE (one fit-median bias per arm, `b = median(g_fit − p_fit_raw)`):

| arm | fit overall cal | dev overall raw | dev overall cal | dev G0 raw | dev G0 cal | bias |
|---|---:|---:|---:|---:|---:|---:|
| B | 0.029280 | 0.108956 | 0.103717 | 0.107251 | 0.101875 | −0.031378 |
| D_J | 0.028155 | 0.104099 | 0.101907 | 0.102679 | 0.100416 | −0.016432 |
| **M_J** | 0.030235 | **0.101414** | **0.099381** | **0.100434** | **0.098216** | −0.015250 |

Paired bootstrap (1000 draws, seed 20261004, shared per-metric indices):

| gain | G0 cal | overall cal | G0 raw | overall raw |
|---|---:|---:|---:|---:|
| D_J→M_J (positive = M_J better) | +0.002200 `[−0.001564,+0.006029]` | +0.002526 `[−0.001649,+0.006467]` | +0.002245 `[−0.001542,+0.006115]` | +0.002684 `[−0.001277,+0.006758]` |
| B→M_J | +0.003659 `[−0.000438,+0.008602]` | +0.004336 `[+0.000264,+0.008757]` | +0.006817 `[+0.002339,+0.011765]` | +0.007542 `[+0.003279,+0.012083]` |
| B→D_J (recomputed) | +0.001459 `[−0.002886,+0.006012]` | +0.001810 `[−0.002073,+0.005961]` | +0.004572 `[+0.000205,+0.009014]` | +0.004857 `[+0.000804,+0.009270]` |

Bootstrap witnesses pass exactly: identical predictions → gain `0` with CI `[0,0]`; swapping arms
mirrors the CI; constant shift moves the point by exactly the shift within the absolute bound.

Per-`k` calibrated MAE (contribution = Σ|err|/N, N=2000; group contributions sum exactly to the
overall MAE and group gains sum exactly to the total gain, `+0.002526`):

| arm | k=0 (1915) | k=−1 (73) | k=−2 (10) | k≤−3 (2) |
|---|---:|---:|---:|---:|
| B | 0.101875 (.09755) | 0.126388 (.00461) | 0.182842 (.00091) | 0.644664 (.00064) |
| D_J | 0.100416 (.09615) | 0.122661 (.00448) | 0.187662 (.00094) | 0.343133 (.00034) |
| M_J | **0.098216 (.09404)** | **0.110771 (.00404)** | **0.147104 (.00074)** | 0.560122 (.00056) |

The M_J advantage is concentrated in G0 and the small −1/−2 groups; on the 2 rows of k≤−3 M_J is
worse (0.560 vs 0.343, 2 rows — descriptive only, no ring claim). Fit side: M_J fit cal
`0.030235` is **worse** than D_J `0.028155`, so the dev advantage is not a fit-only artifact; the
fit→dev gap shrinks from 3.62× (D_J) to 3.29× (M_J).

**PERFORMANCE_SIGNAL vs B**: `G0_cal ≥ 0.003` true (`+0.003659`); `overall_cal ≥ 0.003` true
(`+0.004336`); `G0_raw > 0` true (`+0.006817`); `overall_raw > 0` true (`+0.007542`);
`G0 cal CI_lower > 0` **false** (`−0.000438`). Not passed.

## 3. Frozen classification

`gain_MJ = MAE(D_J) − MAE(M_J)`; frozen rules in `PROTOCOL.md`. Fired: **INCONCLUSIVE**.

* `MLP_LOCAL_SUPPORT` false: G0 cal point `+0.002200 < +0.003` **and** CI lower `−0.001564 ≤ 0`.
* `DICT_LOCAL_SUPPORT` false: direction is positive.
* `LOCAL_EQUIVALENCE` false: G0 cal CI upper `+0.006029 > +0.003` and overall cal upper
  `+0.006467 > +0.003` (the CIs do not fit inside ±0.003).
* `MLP_MECHANISM_FAILED` false: the local path executes to the final state (A drift 0.721,
  `W_loc` norm 7.62, zeroing it changes dev cal MAE 0.0994 → 0.6229).
* `INVALID/INCOMPLETE` false: every input/lineage/init/stream/replay check passed and the fixed
  trajectory completed 15,120/15,120 steps.

Pre-fixed sensitivity: dropping the unique dev row with the largest mean calibrated error across
the two arms (dev pos 1632, gid 8049, in G0) leaves G0 cal gain `+0.001915` and overall cal gain
`+0.002253` — directionally unchanged, still below the frozen gate.

## 4. Mechanism

* **Activity.** M_J `A_raw` relative drift 0.721 (soup) with `W_loc` norm 7.62; task gradients
  reach `A_raw` from epoch 40 (`0.0405`; zero at step 1 by construction because `W_loc=0`).
  Fit root code (scaled) RMS `0.664`, 0 dead dimensions of 64, effective rank `4.53`
  (a very low-rank learned local code). D_J for comparison: `D_loc` drift 0.756, `W_loc` 9.18,
  8/64 active per tuple.
* **Sensitivity to the real pairing.** Frozen J→I operator switch on the soups (same procedure for
  both arms; the D_J switch recomputed here reproduces the previous round to `5.1e-9`): D_J
  mean\|Δpred\| `0.003382` dev / `0.003297` fit, dev MAE change raw `+1.50e-4` / cal `+8.43e-5`
  (G0 raw `+1.52e-4`); M_J mean\|Δpred\| `0.000260` dev / `0.000251` fit, dev MAE change raw
  `+2.77e-6` / cal `+6.99e-6` (G0 raw `−1.87e-6`). After training, the SiLU projection is ~13×
  less sensitive to replacing the real incidence by the marginal product than the IHT dictionary.
* **Dependence interventions (not information shares).** Zero-ablation dev cal MAE:
  D_J `0.1019 → 0.5485`, M_J `0.0994 → 0.6229` (M raw change `+0.5085`). Fit-mean replacement
  (μ of `kappa·e(v)` over fit roots; d=0 roots stay zero; bias untouched) dev cal MAE: D_J
  `0.1019 → 0.4423` (mean\|Δ\| `0.4266`, raw change `+0.3276`), M_J `0.0994 → 0.4845`
  (mean\|Δ\| `0.4754`, raw change `+0.3725`). Both prove the local channel is load-bearing; both
  include large systematic offsets and co-adaptation breakage, so they do not attribute an
  information share.
* **Four-grid decomposition (previous round, re-read read-only).** Calibrated J/I grid:

  | state/operator | overall cal | G0 cal |
  |---|---:|---:|
  | J/J | 0.1019071074 | 0.1004162377 |
  | J/I | 0.1019913913 | 0.1005100907 |
  | I/J | 0.1045416120 | 0.1038224926 |
  | I/I | 0.1046819255 | 0.1039915517 |

  `G = O + W` exactly: overall `G=0.002774818`, `O=0.000112299`, `W=0.002662519` (W/G 96.0%);
  G0 `G=0.003575314`, `O=0.000131456`, `W=0.003443858` (W/G 96.3%). Descriptive split only — it
  neither proves the correspondence is worth only ~0.0001 nor that the training-time inductive
  bias held.
* **Raw vs cal ranking.** M_J is best in every raw and calibrated endpoint; the ranking is stable.
  The fit side reverses (D_J better), so the effect is held-out-directional and still compatible
  with one-seed optimisation noise.

## 5. Boundaries

One new trajectory, one seed, one repeatedly used dev fold, one soup; no seed/fold/config search.
The comparison fixes capacity, input and injection location but not expressive power per se,
computation order or optimisation geometry. A positive `dev g-MAE < 0.09` was not reached and no
official-valid/test or 10k confirmation is bought. The next evidence is independent seed/fold
replication or the pre-registered permuted-marginal control, not another encoder family on this dev.

## 6. Direct answers

* **Best score this round**: M_J dev calibrated overall `0.099381`, dev calibrated G0 `0.098216`
  (raw: G0 `0.100434`, overall `0.101414`).
* **Gap to 0.09**: `+0.008216` on dev G0 cal (k=0, 1915 rows) and `+0.009381` on dev overall cal.
  No threshold crossing; this is a chemical-component (`g`) diagnostic on a repeatedly used dev
  fold, not an official-valid y-MAE.
* **Performance progress**: yes, directionally — M_J beats D_J on all four dev endpoints
  (`+0.0022` to `+0.0027`) and B on all four (`+0.0037` to `+0.0075`, raw significant), but the
  frozen calibrated G0-CI condition fails, so it is not a performance candidate.
* **New discriminative mechanism finding**: both local encoders are load-bearing, but after
  training the same-capacity SiLU projection is ~13× less sensitive to the real-vs-marginal
  incidence substitution than the IHT dictionary (`0.000260` vs `0.003382` mean\|Δpred\|), and its
  local code is near four-dimensional (effective rank 4.53, 0 dead dims). At this budget the
  dictionary form is not required by/for the interface, and the remaining difference cannot be
  separated from optimisation noise.
