# REPORT — zinc_cycle_level_transfer_terminal_test_seed0_v1

**Round**: cycle-level transfer (discrete `k`-level head D replacing continuous `c` regression R
on unseen T25 classes), train-only 3-fold class-grouped purchase gate, then UNCONDITIONAL
terminal official-test report.
**Machine**: local CPU-only, ≤8 threads, FP32, no GPU, no remote job.
**Roster (frozen before any test prediction)**: `SUM_Q`, `B` (frozen COMP+Q), `H` (prototype
transfer, TARGETED_REPAIR_ONLY), `C` (H with the UNSEEN_FALLBACK branch served by `D_full`).
**Purchase gate**: **PASS** (all 4 conditions) → `D_full` trained (3975 params) → `C` deployed.

## Direct answers (the round's questions)

1. **Does a discrete k-level classification head carry cross-exact-class transferable signal
   that a continuous c regression misses?** **Yes — decisively, at the level-decoding stage.**
   Pooled OOF (3 class-grouped folds, held classes never in fit): D overall c-MAE 0.019068 vs
   R 0.023745 (gain +0.004677); k=−2: 0.1733 vs 0.9078 (gain +0.7345); k=−1: 0.0533 vs 0.1160;
   k=0: 0.0043 vs 0.0069. D's median-level decode predicted the exact correct integer `k` on
   held k=−2/k=−1 classes at accuracy 0.92–1.00 per fold. The continuous head cannot land on
   the exact snapped level (R OOF k=−2 error 0.9078 is dominated by near-miss level errors).
   Honest counter-result: extreme singleton levels k ≤ −3 (pooled n=7) FAIL to extrapolate
   (D OOF worse than R at every one of −12, −6, −5, −4); these are held-out singleton classes
   whose T25 signature D generalizes to level 0/−1 instead.

2. **Does C (D_full on the unseen branch) improve H on the exposed valid?** Only marginally and
   single-row dominated: C−H valid cal gain **+0.000544**, CI [−0.014359, +0.015802] (crosses 0);
   POINT_EFFECT_PASS false (<0.003); SINGLE_ROW_DOMINATED (valid:0935, k=−2, is 95.1% of the
   positive sum; excluding H's worst row the gain is −0.00469); y_cal 0.096150, **y<0.09 not
   observed** on valid. On the 12 unseen valid rows D_full hit the exact level on 11/12 (the
   miss: valid:0214, k=−2, decoded level 0, y err +6.89 vs B's +2.00).

3. **What does the official test (1000 rows, post-freeze, unconditional) say?**

   | system | test y_raw | test y_cal | test g_cal | valid y_cal | valid−test |
   |---|---|---|---|---|---|
   | SUM_Q | 0.091876 | 0.092619 | 0.089133 | 0.122653 | −0.030034 |
   | B | 0.084922 | 0.085107 | 0.081184 | 0.117406 | −0.032299 |
   | H | 0.084458 | 0.084719 | 0.081184 | 0.096694 | −0.011975 |
   | **C** | **0.080918** | **0.081174** | 0.081184 | 0.096150 | −0.014976 |

   Fixed comparisons (gain = ref MAE − cand MAE, paired row bootstrap 1000×, seed 20261012):

   | comparison | test cal gain | test 95% CI | valid cal gain | valid 95% CI |
   |---|---|---|---|---|
   | B − SUM_Q | **+0.007512** | **[+0.001810, +0.012664]** | +0.005247 | [+0.000133, +0.009954] |
   | H − B | +0.000388 | [−0.000262, +0.001205] | +0.020712 | [−0.000233, +0.062382] |
   | C − H | **+0.003545** | **[+0.000153, +0.009389]** | +0.000544 | [−0.014359, +0.015802] |

   **On test the cycle channel is essentially solved by C**: overall q c-MAE 0.000011 (the only
   nonzero cycle error is the single TRAIN_CONFLICT row, correctly routed to frozen Q, err
   0.0102); on the 20 UNSEEN_FALLBACK rows D_full decoded the **exact correct k level on all
   20/20 rows** (q c-MAE 0.1965 → 0.0000; k=−2 test group y_cal 0.4324 → 0.0742). Test e_c MAE
   1.02e-05 (B: 0.0060, H: 0.0039); remaining error is the body e_g (0.0809) plus tiny
   cancellation (triangle_gap 2.0e-05 ≥ 0). **C test y_cal 0.081174 < 0.09.**

4. **Is the C−H gain robust or another single-row artifact?** On test it is broad within its
   channel: the gain comes from 20 unseen rows (test routing: 979 CONSISTENT_HIT / 1
   TRAIN_CONFLICT / 20 UNSEEN), of which all 20 level predictions are exactly right; positive
   gains spread over the k=−1 (44 rows) and k=−2 (8 rows) groups rather than one row. On valid
   it WAS single-row dominated (Q2). Note both valid and test are 1000-row splits; the C−H test
   CI lower bound is +0.000153 > 0 but the gain (+0.0035) rests on 20/1000 unseen rows.

5. **valid−test gap**: every system is substantially better on test than on valid (−0.0120 to
   −0.0323). Driver: the valid split contains extreme-cycle rows the test split does not
   (valid: 1× k=−6 [the valid:0172 20.8-error row], 4× k=−2, 1× k=−4; test: 8× k=−2, 44× k=−1,
   948× k=0, none ≤ −3), plus a slightly easier body regime (test g_cal 0.0812 vs valid 0.0893).

## Purchase gate (train-only, pooled OOF)

| condition | required | observed | pass |
|---|---|---|---|
| overall c gain | ≥ 0.003 | +0.004677 | ✓ |
| k=−2 group gain | ≥ 0.25 | +0.734451 | ✓ |
| k=0 group worsening | ≤ 0.001 | −0.002545 (improves) | ✓ |
| k=−1 group worsening | ≤ 0.05 | −0.062642 (improves) | ✓ |

Descriptive class-resampled 95% CI on the overall gain (1000 draws, seed 20261011, same draws
shared R/D): [−0.004195, +0.012589] (crosses 0 — descriptive only, not a gate condition).
Fold-level: R/D OOF c-MAE 0.035316/0.035356 (fold 0), 0.012281/0.000000 (fold 1, D exact),
0.023635/0.021844 (fold 2); held k accuracy 0.9961/1.0000/0.9985. Honest rare-level OOF table
(pooled): k=−12 (n=1) D 38.14 vs R 13.70; k=−6 (n=2) 20.80 vs 19.92; k=−5 (n=3) 10.40 vs 10.25;
k=−4 (n=1) 13.87 vs 13.05 — all negative (singleton-class extrapolation failure, disclosed).
Fit-vocab gaps: fold 0 fit lacked level −4, fold 2 fit lacked −12 (marked in folds_summary.json).
Max positive OOF row share 2.8% (not single-row dominated). Identities: R/D hidden init equal
and equal to Q's untrained seed-0 hidden init (all 3 folds); OOF reload bit-identical.

## Test engineering (frozen, no refits)

Official ZINC subset test 1000 rows via the zinc_full_decomposition_valid_test_confirmation_v1
mapping (load_test_data → zinc_compact_v4 _load_zinc/extract/build_encoded → p1run attach env →
frozen full-train prep). Test labels: y = g + c exact (max |y−(g+c)| 1.7e-18); c from the frozen
full-train constants (mu_cycle −0.00014482659782137728, sigma_cycle 0.28844036744667567) and
k = round(snapped c); NOT the CSV's own label column (old 8k constants). logP/SA joined via
smi_line from gvae_full_properties.npz. Test k counts: {0: 948, −1: 44, −2: 8}.

## Replay / identity checks (all pass)

Bodies replayed in float64 from frozen soup states on both splits (identical to the valid
float32 cache within 1e-5; replayed H/C raw predictions vs saved rows: max abs 0.0); H valid
y_cal 0.09669437497661369 and B valid 0.11740618350630393 reproduce the published values
exactly. Wrapper replay from saved D_full/prototype/Q states: bit-identical (max abs 0.0,
routes identical). Row-order shuffle→unshuffle: 0.0 (the wrapper reads only T — label
independent). C ≡ H on shared branches (valid 988/1000, test 980/1000): max |Δpred| 0.0.
Routing-table contributions add back to the overall y_cal MAE exactly (0.0). Bootstrap
witnesses: identical→0.0, swap→mirror, +0.25 constant shift→bounded change. D_full train
decoded c-MAE 0.005200, k accuracy 0.9995 (gen seed 20261003, 23700 steps, soup 296–300).

## ERRATA (prior-round text corrections, no artifacts changed)

1. The previous round's REPORT said the oracle `h+c+b_B` valid y_cal 0.089325 "remains above
   0.09". This is a text error: 0.089325 < 0.09 (the oracle was BELOW 0.09). The numerical
   artifact was always correct.
2. The previous round reported a negative `triangle_gap`. That round used the opposite sign
   convention |e_g|+|e_c|−|e_g+e_c| with the sum-minus-parts orientation reversed. Under this
   round's frozen convention (triangle_gap = mean(|e_g|+|e_c|−|e_g+e_c|) ≥ 0) all values are
   non-negative, as observed here on both splits for B/H/C.

## Conclusions

- The discrete-level head is a real mechanism: on exact-T25-unseen rows it predicts the integer
  cycle level essentially perfectly (20/20 on test, 11/12 on valid), eliminating the cycle
  channel error (test e_c 1.0e-05) where the continuous Q cannot (unseen q c-MAE 0.197 on test,
  0.802 on valid).
- C is the best deployed system on test (y_cal 0.081174, CI-supported gain over H), and the
  first frozen system below 0.09 on the official test in this line. But the remaining error is
  now ~entirely the chemistry body error e_g (test MAE 0.0809): the cycle channel is exhausted.
- B − SUM_Q is CI-supported positive on BOTH valid and test (+0.0052 / +0.0075): the
  component-supervision benefit is confirmed out-of-sample, not a valid artifact.
- H − B on test is only +0.000388 (CI crosses 0): the prototype transfer's large valid gain
  (+0.0207) was driven by valid:0172 (k=−6), a regime absent from test.
