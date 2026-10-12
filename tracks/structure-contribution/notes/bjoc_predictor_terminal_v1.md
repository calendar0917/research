# cscl-bjoc-predictor-terminal-evaluation-v1 — notes

* Round type: **frozen-version terminal evaluation** (zero new architecture work, exactly one new
  training trajectory, one unified scoring invocation).
* Protocol: `configs/bjoc_predictor_terminal_v1.yaml` (written and committed **before** the training
  and before any label read).
* Driver: `code/run_bjoc_predictor_terminal_v1.py` (stages `reuse-audit`, `valid-audit`,
  `train-n1`, `predict`, `freeze`, `score`).
* Tests: `tests/test_bjoc_predictor_terminal_v1.py` (14 fast + 3 slow, all passing).
* Artifacts: `results/bjoc_predictor_terminal_v1/` (git-ignored).
* Report: `results/bjoc_predictor_terminal_v1/REPORT.md`.
* Status: **COMPLETED 2026-10-12; round stopped (no further seeds, no tuning, no post-test rescue
  training).**

---

## 1. What was authorised and what was executed

| authorisation | execution |
| --- | --- |
| one and only one new formal training trajectory (`BJOC-Null full10k seed1`, ≤0.5 GPU·h) | exactly one: `bjoc_null_full10k_s1`, 240 epochs, last-5 soup, **482.6 s** train wall (491.2 s including replay), A100-2 GPU1 via `rr` (pool `res-gpu1`), commit `7a6baec5` |
| zero-training reuse of Y0, Y1, N0 | reused read-only; all four checkpoints and their preprocessing hash-verified before and after every stage |
| freeze four models + label-blind predictions, then ONE terminal scoring | `predict` on CPU ran with official labels never read (access log), predictions hashed, `terminal_freeze_manifest.json` written, then **one** `score` invocation read the two label vectors (recorded in `scoring_record.json`) |
| no seed 2, no organized arm, no Full-Y retraining, no architecture/hyper-parameter change, no post-test rescue | honoured; the only changed quantity vs the frozen recipe is the trajectory seed (0 → 1) |

Budget: 1 GPU trajectory = 482.6 s ≈ 0.134 GPU·h (cap 0.5 GPU·h).  Total round walls: reuse-audit
27 s (CPU), valid-audit ~24 s (CPU), train-n1 491 s, predict 15 s (CPU), freeze 21 s, score 5.5 s.

## 2. Reuse audit (zero training)

`reuse_audit.json`, `hard_checks_pass = true`, decision `REUSE_FROZEN_MODELS`:

* Y0 / Y1 are the two historical Full-Y raw-y soups (`Y_seed0_raw_soup_state.pt` sha
  `7792c560…`, `Y_seed1_raw_soup_state.pt` sha `f40c2f5b…`), both last-5 soups 236–240, 408,651
  params, `fit_target_is_y = true`, `h1_lambda = 33.9587`, recipe pairs identical except the
  trajectory seed; the two state hashes differ (`6c1e81ce…` vs `6b7ab1b3…`).
* CPU replay of the *training* objective on all 10 000 train rows: Y0 max|Δpred| 5.72e-06,
  Y1 2.38e-06 vs the stored exports; replay fit MAE reproduces the recorded value to 5.7e-09.
* N0 = `bjoc_null_full10k_s0_soup_state.pt` sha `e5a2619e…`, 413,505 params exactly
  (atom 26 336 / bond 62 816 / junction 112 896 / composer 62 976 / reader 148 481), forbidden
  module scan empty (no GNN / attention / message passing), supervision `raw-y L1 only`, global
  normaliser fit on 10 000 train rows only, train json sha recorded.
* `BJOC_N1_present_before_training = false` → the round started with exactly three frozen models.
* `official_valid_loaded = false`, `official_test_loaded = false` during the audit.

## 3. The single new trajectory (N1)

`seed1_pretrain_evidence.json` + `seed1_train_meta.json` + `seed1_replay.json`:

| field | value |
| --- | --- |
| command | `run_structural_relation_composer_v1.py --device cuda:0 train-bjoc --arm null --full10k`, driven by `run_bjoc_predictor_terminal_v1.py train-n1` |
| host / GPU | `res`, a100-2, `NVIDIA A100-SXM4-40GB`, torch 2.5.1+cu124 (CUDA_VISIBLE_DEVICES=1) |
| recipe | Adam lr 1e-3, wd 1e-5, batch 128, clip 5, 240 epochs, shuffle seed 102 = 1+101, last-5 soup 236–240, raw-y L1 only |
| init hash | `c79e6b02521c224e79790382438ca72c08f9e22e691f66fb5ab950dea4b49c0b` |
| steps | 79/epoch, 18 960 total |
| wall | 482.6 s (train) / 491.2 s (command) |
| soup fit MAE (10 000 train rows) | **0.039523751913** (N0: 0.052095845) |
| replay | `replay_fit_mae` 0.039523752751, |Δ| = 8.4e-10 ≤ 5e-5, released copy byte-identical (sha `74587071499735ed…`) |
| label discipline | `official_valid_loaded = false`, `official_test_loaded = false`, global statistics on fit rows only, `locked_labels_read = false` |
| supervision audit | traced `y_source_lines` + the only loss calls are `F.l1_loss(pred.view(-1), y…)`; no `g/ell/s/c/k` term anywhere in the fit path |

**Deviation worth recording (not a protocol breach):** the two trajectories are much further apart
in fit quality than expected — N1's last-5 soup fits the training set at 0.0395 vs N0's 0.0521
(−24%).  Combined with the test spread (N1 0.0899 vs N0 0.0946) this is the clearest quantitative
statement in the round: **trajectory-to-trajectory variation is larger than the system-level
difference being measured.**

## 4. Label-blind prediction and freeze

`prediction_access_log.json`, `terminal_freeze_manifest.json`:

* All four models were run on **CPU** so that neither CUDA nor a data loader could touch labels.
  The official-validation and official-test `Data` objects passed through a `LabelGuard` wrapper and
  every label tensor (`y`, `g`, `c`, `k`, `ell`, `s`) was replaced by zeros before any forward pass
  (`label_values_read = false`).
* y-independence gate: re-running 32 molecules with all label tensors set to 0 vs 999 gives
  **exactly identical** predictions for all four models on both splits (max |Δpred| = 0.0).
* Alignment gates: row order, atom/bond/junction counts, canonical-SMILES ordering and the
  structure signature verified per split (valid 1000 rows, 23 083 atoms / 24 846 bonds / 34 520
  junctions; test 1000 rows, 23 117 / 24 860 / 34 595).
* Engineering cross-checks (prediction-only, no labels): the CPU replay of Y0/Y1 reproduces the
  historical GPU exports **bit-identically** on valid (max |Δ| = 0.0) and the historical valid MAE
  to 0.0; N0's CPU export differs from the frozen A100 export by 5.7e-06 (float32).
* `terminal_freeze_manifest.json` records the four checkpoint hashes, four prediction-file hashes
  per split, the per-model files, the split fingerprints (processed-`.pt` sha, 1000-row
  structure sha), the inference revision, the ensemble rule, the primary metric, the pre-declared
  secondaries, the bootstrap seed/count, the evaluation rules and the historical-exposure
  declaration.  `frozen_before_any_official_label_read = true`.
* `stage_score` refuses to run if any frozen hash changed; `stage_train_n1` refuses to start if the
  seed-1 checkpoint or its released copy already exists (tested).

## 5. Terminal scoring — the one label read

`scoring_record.json`: two label reads (valid, test), one invocation, `n_label_reads = 2`,
`single_invocation = true`, `post_score_prediction_changes = 0`.  No calibrator, no re-selection, no
molecule deletion, no re-run of `predict`.

### Primary endpoint (official test, raw-y MAE, two-seed mean)

* `MAE_Y_mean = 0.0956161563`, `MAE_N_mean = 0.0922331502`,
  **`Δ_test = −0.0033830061`** (negative ⇒ BJOC-Null lower).
* Paired molecule bootstrap (5000 resamples, seed 0) CI95 = **[−0.0098593, +0.0034652]** →
  includes zero.  Branch: `point_estimate_negative_ci_crosses_zero`.
* Per-seed: `Δ0 = MAE_N0 − MAE_Y0 = −0.0000259` (i.e. *no* difference), `Δ1 = −0.0067402`.
  Both negative, but one of the two seeds is a tie.
* Molecules: 53.5% better / 46.5% worse under the mean statistic.
* Ensemble (equal-weight two-seed average, reported separately): Y 0.0859546 vs N 0.0852325,
  Δ = −0.0007221 (paired CI [−0.0075317, +0.0064393]).
* RMSE (seed MAE-mean is *not* an RMSE): N mean 0.173639 is **worse** than Y mean 0.169682 — the
  MAE gain comes with a heavier tail (N1 test max |e| 2.66 vs Y0 2.49; N1 tail error-mass share
  14.7% vs Y1 11.5%).

### Secondary subgroups (official test)

| subgroup | n | Δ_mean | CI95 | frac N better |
| --- | --- | --- | --- | --- |
| ordinary `y > −3` | 923 | **−0.004316** | [−0.010862, +0.002687] | 54.4% |
| tail `y ≤ −3` | 77 | **+0.007807** | [−0.016966, +0.031803] | 42.9% |
| all rows minus the single worst row (702) | 999 | −0.005266 | — | — |

So the test point estimate is **broad-based** (it survives deletion of the worst molecule and is
positive on ordinary molecules) and its tail behaviour is the *opposite* of the validation-split
behaviour (see §6).  Median/p90 also favour N (0.0625/0.1830 vs 0.0633/0.1905), p95 is roughly tied.

## 6. Q1/Q2/Q3

**Q1 — does the BJOC-Null advantage reproduce at seed 1?**
Yes in direction, on both splits: seed-1 MAE N1 0.098430 (valid) / 0.089903 (test) vs historical
Full-Y Y1 0.117695 / 0.096643.  The seed-1 trajectory is also the *stronger* of the two BJOC
trajectories on both splits, so the "reproduction" is real but conflated with seed quality.

**Q2 — how much of the validation gain 0.118618 → 0.103423 comes from `valid:0172`?**
**91.27%.**  Row 0172 (y = −20.3405): |e_Y0| 19.9057 → |e_N0| 6.0375.  Total gain 0.015194, of
which 0.013868 is that one row (1/1000 of the split, 16.8% of Y0's total error mass).  The residual
999-row gain is 0.001326 — and it is *not* a broad improvement: on ordinary molecules (y > −3,
n = 935) N0 is slightly **worse** (+0.00074), and N0 is better on only 47.9% of all molecules.  The
residual 999-row gain is itself concentrated in the deep tail (`y ≤ −3` without 0172, n = 64:
−0.03153; complexity strata: cyclomatic ≥ 4 +0.00583, above-median diameter +0.01399).  The
organized arm O0 shows the same pathology more strongly (gain on the other 999 rows is *negative*,
−0.00568).
⇒ The earlier public statement "BJOC-Null is 0.010–0.015 MAE better on official validation" must be
**re-scoped**: the validation number is a single-extreme-row effect, not a broad accuracy gain.

**Q3 — BJOC-Null test raw-y MAE vs the historical Full-Y two-seed system.**
0.092233 vs 0.095616, Δ = −0.003383, CI crosses zero, 3.5% relative.  With only two seeds per
system, and with a within-BJOC seed spread (0.004660) larger than the entire between-system
difference, this is a **numerically lower but statistically unresolved** comparison.  The
equal-weight ensembles differ by only −0.000722, i.e. 3.5% of a 1.4% relative gap — the two systems
are effectively indistinguishable on this split, and the honest headline is
*"point estimate favours BJOC-Null; not supported at this sample size; validation-split advantage was
row-driven."*

## 7. Decision and claims

`scientific_decision.json` branch: **`point_estimate_negative_ci_crosses_zero`** (G3-like:
underpowered / non-separated point estimate).

Allowed claims: a frozen-version descriptive comparison on an already-exposed official test split;
which pre-declared secondary endpoints moved and in which direction; disclosed historical-exposure
language.

Forbidden: independent blind-test confirmation, SOTA, causal structural-mechanism claims from a test
improvement, and any reuse of this split for further model selection.  The single extreme row must
never be deleted from the official metric; this round reports it as a decomposition only.

## 8. What a next round should buy (unchanged from the BJOC round's own reading)

* The predictive constraint is **ordinary-molecule** accuracy, not seed count: on valid the
  advantage vanishes on the 935 ordinary molecules; on test the advantage is ordinary-molecule-based
  while the tail degrades.  Neither pattern is stable across splits.
* A **matched full-10k Full-Y control on this same evaluation path** is still the missing piece
  before any architecture attribution: the 0.095616 reference is another trajectory (different
  preprocessing, different recipe), so the −0.003383 is not a controlled contrast.
* Mechanism/attribution work needs the additive-object route described in
  `scientific_decision.json.next_steps`, not more MAE seeds.

## 9. Artifact index (`results/bjoc_predictor_terminal_v1/`)

| artifact | content |
| --- | --- |
| `reuse_audit.json` | three-model reuse audit, hard checks, replay evidence |
| `checkpoint_inventory.json` | per-checkpoint sha/state-hash vs recorded values |
| `valid_error_decomposition.json` | C1 row-0172 table, C2 strata, C3 input-content audit, C4 naming caveat |
| `seed1_pretrain_evidence.json` | pre-training contract: rows, seed stream, init hash, params, loss code |
| `seed1_train_meta.json` | full v1 train json (curve, hyper, config, walls, GPU, discipline flags) |
| `seed1_last5_soup_state.pt` | released seed-1 checkpoint (byte-identical to the v1 artifact) |
| `seed1_replay.json` | byte-identical copy proof + replay fit MAE |
| `prediction_access_log.json` | label-blind evidence, alignment gates, y-independence, cross-checks |
| `{valid,test}_{Y0,Y1,N0,N1}_preds.npz` | per-model frozen predictions |
| `{valid,test}_predictions.npz` | combined frozen predictions + meta (`y_in_file: false`) |
| `terminal_freeze_manifest.json` | freeze record (hashes, fingerprints, rules, exposure) |
| `official_{valid,test}_metrics.json` | full metric surface, subgroups, paired stats |
| `paired_bootstrap.json` | valid/test/per-seed/ensemble paired bootstraps |
| `subgroup_analysis.json` | all structural strata for both splits |
| `scoring_record.json` | label-read log and post-score immutability statement |
| `scientific_decision.json` | branch, inputs, claims allowed/forbidden, 10 answers, next steps |

## 10. Disclosures

1. **Test exposure.** Official test was read by this repository before (KSVD track, 2026-10-03);
   `project_wide_pristine = false`.  This is a terminal descriptive evaluation of an exposed
   instrument, **not** a blind confirmation.
2. **Historical reference is not matched.** Y0/Y1 come from the KSVD full-decomposition round with
   different preprocessing and recipe; there is no matched Full-Y arm on this evaluation path.
3. **Seed-1 trajectory is the only new GPU work** and it is well inside budget; no second seed was
   trained after seeing any result.
4. **Local artifact hygiene:** the local `prep_fit7200.npz`/`_meta.json` copy (BJOC round's fit-8000
   prep) was stale relative to the canonical remote version; it was replaced by the remote copy
   (sha `50219e07…`, commit `290423e5`) during the pull.  Neither file participates in this round's
   inference, which uses `all_train_prep.npz` (KSVD) for the Y models and
   `bjoc_global_norm_full10k.json` + `bjoc_objects_v1.pt` for the BJOC models.
5. **Training-stability limits.** Two seeds per system cannot estimate across-trajectory variance,
   and a molecule bootstrap does not cover it either; the seed spread reported above is a *lower
   bound* on that noise.  No claim in this note treats a seed pair as a variance estimate.
6. **Method naming.** BJOC is a strictly non-iterative (one-shot) structural object composer with
   junction→bond ownership features and a plain MLP reader; the words "GNN-free architecture
   improvement" and "not a GNN" are avoided as unqualified claims — the audited statement is
   "no GNN/attention/message-passing module is present in the audited parameter set".
7. **Scoring is terminal.** After `score` no prediction was regenerated, no model was selected among
   {N0, N1, ensemble}, and no further training was performed.  The result belongs to this frozen
   version only.
