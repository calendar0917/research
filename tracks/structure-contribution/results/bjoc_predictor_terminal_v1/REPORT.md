# REPORT — cscl-bjoc-predictor-terminal-evaluation-v1

Frozen-version terminal evaluation of the BJOC-Null predictor against the historical Full-Y arm on
the official ZINC valid/test splits.  Four pre-fixed models, one new training trajectory
(BJOC-Null full-10k seed 1), label-blind prediction + freeze, one unified scoring invocation.

**Verdict (one line):** the official-test primary endpoint moved in the pre-declared favourable
direction, `Δ_test = MAE_N_mean − MAE_Y_mean = −0.003383` raw-y MAE, but the paired 95% CI
**[−0.009859, +0.003465]** includes zero, one of the two seeds shows no difference at all, the
within-BJOC seed spread (0.004660) exceeds the between-system difference, and the previously
reported *validation* advantage is shown here to be 91.27% a single-molecule effect.  Branch:
`point_estimate_negative_ci_crosses_zero`.  The official test was historically exposed; this is not
a blind confirmation.

---

## Table 1 — Primary endpoint (official test, raw-y MAE, two-seed mean)

| quantity | Full-Y (historical) | BJOC-Null | Δ (N − Y) |
| --- | --- | --- | --- |
| seed 0 MAE | 0.094588971 | 0.094563116 | **−0.0000259** |
| seed 1 MAE | 0.096643341 | 0.089903184 | **−0.0067402** |
| **two-seed MAE mean (primary)** | **0.095616156** | **0.092233150** | **−0.0033830** |
| paired bootstrap CI95 (5000 draws, seed 0) | — | — | **[−0.0098593, +0.0034652]** |
| molecules better / worse | — | — | 535 / 465 (53.5%) |
| equal-weight seed ensemble MAE | 0.085954605 | 0.085232507 | −0.0007221 (CI [−0.007532, +0.006439]) |

Reading aids, not endpoints: the BJOC two-seed MAE spread is 0.004660 and the Full-Y spread is
0.002054 — both **larger than the between-system difference**; the ensembles differ by 0.00072.
The primary metric is the mean of two per-seed MAEs and is *not* the MAE of an averaged prediction;
both are reported.

## Table 2 — Full metric surface of the four frozen models

| split | model | MAE | RMSE | median | p90 | p95 | max \|e\| |
| --- | --- | --- | --- | --- | --- | --- | --- |
| valid | Y0 | 0.118617547 | 0.653580 | 0.060427 | 0.224687 | 0.308601 | 19.9057 |
| valid | Y1 | 0.117694799 | 0.664394 | 0.061345 | 0.202542 | 0.311629 | 20.3835 |
| valid | N0 | 0.103423211 | 0.255893 | 0.060960 | 0.197989 | 0.307891 | 6.0375 |
| valid | N1 | 0.098429780 | 0.297117 | 0.056017 | 0.178818 | 0.301118 | 7.7751 |
| valid | Y ensemble | 0.109857258 | 0.655798 | 0.054459 | 0.192535 | 0.280951 | 20.1446 |
| valid | N ensemble | 0.094635781 | 0.271470 | 0.053876 | 0.171761 | 0.279092 | 6.9063 |
| test | Y0 | 0.094588971 | 0.170265 | 0.061878 | 0.185808 | 0.258918 | 2.4922 |
| test | Y1 | 0.096643341 | 0.169100 | 0.064813 | 0.195144 | 0.261845 | 2.4484 |
| test | N0 | 0.094563116 | 0.170447 | 0.062463 | 0.182981 | 0.240465 | 2.4356 |
| test | N1 | 0.089903184 | 0.176831 | 0.053285 | 0.182415 | 0.259901 | 2.6595 |
| test | Y ensemble | 0.085954605 | 0.158386 | 0.054949 | 0.169414 | 0.237993 | 1.9523 |
| test | N ensemble | 0.085232507 | 0.163428 | 0.053304 | 0.168814 | 0.227065 | 2.5475 |

Seed-mean RMSE on test: Y 0.169682, N 0.173639 → **BJOC-Null is worse in RMSE** while better in MAE
(N1's tail is heavier: max 2.66, tail error-mass share 14.7% vs Y1 11.5%).

Pre-declared secondary subgroups (official test):

| subgroup | n | Δ_mean | CI95 | frac N better |
| --- | --- | --- | --- | --- |
| ordinary `y > −3` | 923 | −0.004316 | [−0.010862, +0.002687] | 54.4% |
| tail `y ≤ −3` | 77 | +0.007807 | [−0.016966, +0.031803] | 42.9% |
| all rows minus single worst row | 999 | −0.005266 | — | — |

## Table 3 — Freeze integrity and provenance

| item | value |
| --- | --- |
| protocol / driver / tests | `configs/bjoc_predictor_terminal_v1.yaml`, `code/run_bjoc_predictor_terminal_v1.py`, `tests/test_bjoc_predictor_terminal_v1.py` (17 tests pass) |
| commit | `7a6baec5f974` (protocol frozen before training; remote executed this revision) |
| Y0 checkpoint | `Y_seed0_raw_soup_state.pt` sha `7792c5604545…`, state `6c1e81ce…`, 408 651 params, soup 236–240 |
| Y1 checkpoint | `Y_seed1_raw_soup_state.pt` sha `f40c2f5b62ee…`, state `6b7ab1b3…`, 408 651 params, soup 236–240 |
| N0 checkpoint | `bjoc_null_full10k_s0_soup_state.pt` sha `e5a2619e35fe…`, 413 505 params (exact) |
| N1 checkpoint | `seed1_last5_soup_state.pt` sha `745870714997…`, 413 505 params, soup 236–240, wall 482.6 s |
| frozen predictions | valid `cd7b323117bf…`, test `82ba72c0bfad…` (per-model files also hashed) |
| split fingerprints | valid: processed-`.pt` `3f343e47281b…`, structure `fbbe58011b01…`, 1000 rows; test: `75668e7838f6…`, 1000 rows |
| label-blind guarantee | all four models ran on CPU behind a `LabelGuard`; label tensors zeroed; 0 vs 999 relabel changes predictions by exactly 0.0; `official_test_labels_read = false` during `predict` |
| replay fidelity | Y0/Y1 CPU replay of historical exports max \|Δ\| = 0.0 (bit-identical); N0 vs frozen A100 export 5.7e-06; N1 replay fit MAE \|Δ\| = 8.4e-10 |
| label reads | exactly 2 (valid, test) inside one `score` invocation, `post_score_prediction_changes = 0` |
| GPU budget | 0.134 GPU·h of the 0.5 GPU·h cap; exactly one new trajectory |
| exposure | official test read previously by the KSVD track (2026-10-03), `project_wide_pristine = false` |

## Table 4 — Key-question answers

| # | question | answer | evidence |
| --- | --- | --- | --- |
| Q1 | does the BJOC-Null advantage reproduce at full-10k seed 1? | **yes in direction, on both splits** — N1 0.098430 (valid) / 0.089903 (test) vs Y1 0.117695 / 0.096643; N1 is also the stronger of the two BJOC trajectories, so the reproduction is conflated with seed quality | `official_{valid,test}_metrics.json` |
| Q2 | how much of the valid gain 0.118618 → 0.103423 is single row `valid:0172`? | **91.27%** (0.013868 of 0.015194); the residual 999-row gain is 0.001326 and is *not* broad — on the 935 ordinary molecules N0 is **worse** (+0.00074) and N0 is better on only 47.9% of molecules; the residual is itself tail-concentrated (`y ≤ −3` without 0172: −0.03153) | `valid_error_decomposition.json` |
| Q3 | BJOC-Null test raw-y MAE vs historical Full-Y two-seed system | **0.092233 vs 0.095616, Δ = −0.003383 (3.5% relative), CI crosses zero**; within-BJOC seed spread 0.004660 > |Δ|; ensembles differ by 0.000722 ⇒ numerically lower, statistically unresolved | Table 1 |
| Q4 | is the test point estimate row-driven? | **no** — removing the single worst molecule (702) leaves −0.005266, and the 923 ordinary molecules give −0.004316 | subgroup table |
| Q5 | is the test advantage the same phenomenon as the validation advantage? | **no** — validation advantage is one extreme row; test advantage is ordinary-molecule-based with a **tail penalty** (+0.007807).  Opposite structure, neither stable across splits | §Q2 vs §test subgroups |
| Q6 | ensemble or single model? | ensembles are near-tied (−0.000722); the frozen equal-weight ensemble is reported but never used to select a seed or weight | Table 1 |
| Q7 | does the trajectory variation dominate? | **yes** — soup fit MAE 0.0395 (N1) vs 0.0521 (N0); test MAE 0.0899 vs 0.0946; seed spread > system difference | `seed1_train_meta.json` |
| Q8 | any label leakage or post-test rescue? | none — label-blind CPU prediction, frozen hashes verified before/after scoring, one read, no calibrator, no re-selection | `prediction_access_log.json`, `scoring_record.json` |
| Q9 | supervision contract respected? | yes — raw-y L1 only; traced loss calls and `y_source_lines`; no `g/ell/s/c/k` term; global stats fit-only | `seed1_pretrain_evidence.json` |
| Q10 | decision branch | **`point_estimate_negative_ci_crosses_zero`** | `scientific_decision.json` |

---

## 1. What was run

1. `reuse-audit` (CPU, 27 s): hash and replay the three reusable models; `hard_checks_pass = true`.
2. `valid-audit` (CPU, 24 s, zero training): per-molecule decomposition of the official-valid
   BJOC-Null vs historical-Full-Y difference (the round that answers Q2).
3. `train-n1` (remote `res`, pool `res-gpu1`, A100-2, 491 s): the single authorised trajectory
   through the frozen `train_bjoc_arm("null", 1, full10k=True)` entry point, checkpoint copied to
   the round output with a byte-identity requirement and an immediate replay check.
4. `predict` (local CPU, 15 s): all four models, both splits, labels never read.
5. `freeze` (21 s): manifest with all hashes, fingerprints, rules and the exposure declaration.
6. `score` (5.5 s): the one unified terminal scoring, reading valid and test labels once.

## 2. Interpretation

* **The BJOC-Null predictor is a working one-shot structural-object composer.**  On the official
  test split its two-seed mean MAE is lower than the historical Full-Y reference and it is lower on
  ordinary molecules, with a 54% per-molecule better-fraction.  That is the strongest positive
  signal this architecture family has produced on an official split.
* **The signal is not statistically resolvable.**  Two seeds per system cannot separate a
  −0.0034 difference when the seeds themselves disagree (−0.00003 vs −0.00674) and the
  within-system spread is 0.0047.  RMSE moves the other way, so the gain is a body-of-distribution
  shift, not a tail improvement.
* **The validation split has been over-interpreted in the previous round.**  The 0.118618 → 0.103423
  improvement is 91% one molecule, and the same architecture is *worse* than Full-Y on the 935
  ordinary validation molecules.  The official-validation number must no longer be quoted as
  evidence of a general accuracy advantage.
* **Nothing here isolates the ownership (junction→bond) representation.**  The historical Full-Y
  reference is a different preprocessing and recipe; a matched full-10k Full-Y arm on this evaluation
  path remains the missing control before any architecture attribution.
* **No causal/mechanism claim and no SOTA claim is made.**  The audited statement is limited to: the
  parameter set contains no GNN/attention/message-passing module, receives raw-y L1 supervision
  only, and produced these frozen predictions.

## 3. Discipline statement

* No test-driven selection: the four models, the primary endpoint, the secondary endpoints, the
  ensemble rule and the bootstrap procedure were frozen in the protocol commit `7a6baec5` before the
  seed-1 run and before any label read.
* No calibrator was fit after test, no molecule was deleted from the official metric, no prediction
  file was regenerated after scoring, and no training was performed after the test read.
* The official test split was exposed historically (`project_wide_pristine = false`); this round is
  a terminal descriptive evaluation of a frozen version and may never be cited as independent blind
  evidence.

## 4. Artifact map

`reuse_audit.json`, `checkpoint_inventory.json`, `valid_error_decomposition.json`,
`seed1_pretrain_evidence.json`, `seed1_train_meta.json`, `seed1_last5_soup_state.pt`,
`seed1_replay.json`, `prediction_access_log.json`, `{valid,test}_predictions.npz`,
`{valid,test}_{Y0,Y1,N0,N1}_preds.npz`, `terminal_freeze_manifest.json`,
`official_valid_metrics.json`, `official_test_metrics.json`, `paired_bootstrap.json`,
`subgroup_analysis.json`, `scoring_record.json`, `scientific_decision.json`
(all under `results/bjoc_predictor_terminal_v1/`, git-ignored).
