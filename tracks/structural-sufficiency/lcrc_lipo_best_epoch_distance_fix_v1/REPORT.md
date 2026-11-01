# LCRC Lipophilicity — best single Validation epoch + corrected distance aggregation

`lcrc_lipo_best_epoch_distance_fix_v1` · protocol `lcrc-lipo-best-epoch-distance-fix-v1`
· plan commit `ac3a6e65` → frozen protocol `71e8195c5a52` · results commit (this file) is the
final SHA in §13.

Scope tag for every Test420 number in this report: **`EXPOSED_TEST_EXPLORATORY`**. The
random42 split has been used in earlier rounds, so Test420 here is a *consistency* read on an
already-exposed partition, evaluated once, after all 18 trainings and after the checkpoint
manifest was frozen. It is not a new blind test and it was not used for any selection.

---

## 0. Short answers

| question | answer |
| --- | --- |
| **Q1** does removing the last-5 soup help? | Yes, on Validation, for every matched comparator. Best single epoch beats the historical soup by −0.0042 (ROLEBLIND s0, inside noise) to −0.0381 (ROLEBLIND s1), and beats this round's own last-5-epoch proxy by 0.0119–0.0154 for all six models. ROLEBLIND's 3-seed Validation mean (0.5411) is 0.0194 better than the mean of its two historical soup values (0.5605), and its between-seed spread shrinks from 0.0525 (soup) to 0.0098. |
| **Q2** is ROLEBLIND reliably better than BASE? | On the random42 **Validation, yes**: 3/3 paired wins, mean gain +0.0209 (5.8× the measured run-to-run noise floor of 0.0036); on **Test only 2/3** (mean +0.0101, seed 1 reverses by −0.0157). It does **not** replicate under scaffold shift (§11: +0.0081 on Validation, 2/3 seeds, −0.0008 on the scaffold Test). So: a real effect on the split that was used for selection, not a general structural-sufficiency result. |
| **Q3** after the aggregation fix, is DIST-TRUE a useful model? | **No.** Aggregation is now provably correct (hand-computed 6-atom chain, gradient locality, batch/permutation invariance: 20/20 gates), yet `DIST-TRUE-FIXED` is **worse than its matched blind twin on 3/3 seeds and both splits** (Validation mean 0.5469 vs 0.5395; Test 0.5541 vs 0.5348). The failure is in the *content* of the D34/D5 buckets, not in the code path. |
| **Q4** how do the two G2L variants do? | `G2L-ALL` has the best 3-seed mean Validation RMSE (0.5369) and wins the paired test vs ROLEBLIND 3/3, but two of the three margins (+0.0011, +0.0014) are **below** the noise floor and its Test mean is only 3rd (0.5413, seed 0 reverses by +0.0334). `G2L-LOO` is worse than ROLEBLIND on Validation in 2/3 seeds and on Test in 3/3 (mean +0.0041 / +0.0072 worse). Neither is evidence of a structure-aware gain. |
| **Q5** next direction | Two things only: (a) settle the context effect with **more seeds on both splits** before adding any mechanism (the between-seed spread, 0.003–0.011, is larger than every difference among the context-conditioned models), and (b) **close** the D34/D5 distance-bucket line — it is content-harmful, not code-broken. Scaffold shift (§11) says no structural route may be claimed from random42 Validation alone. Details in `DECISION.md`. |

---

## 1. What this round changed

1. **`DIST-TRUE` aggregation bug fixed** (§2). The historical `DistanceCorrection.forward()` wrote
   every bucket sum to the **target** atom index instead of the **root** index. Because the flat
   bucket arrays are target-indexed, row `i` of the resulting context was
   `h_i · (#roots whose bucket contains i) / |D*_i|` — a rescaled copy of the atom's own
   representation, not a neighbourhood mean. Measured correlation between the historical context
   and the atom's own `h`: **1.000**; for the fixed version: **0.0017**.
2. **Last-five-epoch soup replaced by the single best-Validation epoch** (§3), for all six models
   and all three seeds: full Validation420 evaluation after *every* epoch, state dict saved only on
   strict improvement (ties keep the earlier epoch), max 360 epochs, no early stopping, no
   soup/SWA/EMA, no Test-based selection. The best checkpoint is reloaded, re-verified and then
   used for Test.
3. **18 full trainings** = 6 models × 3 seeds, with paired controls inside each seed (§3.2).
4. **One Test420 pass**, after the checkpoint manifest was frozen (§9).

Nothing else was changed: same frozen random42 split, same structure cache, same recipe, same
initialisation scheme, same batch order per seed.

---

## 2. The distance-context bug and the fix

Definition used from here on (per molecule, per root atom `i`, with `h_j` the encoder
representation of atom `j`):

```
D34_i = { j : 3 <= dist(i, j) <= 4 }        c34_i = mean_{j in D34_i} h_j   (0 if empty)
D5_i  = { j : dist(i, j) >= 5, finite }     c5_i  = mean_{j in D5_i}  h_j   (0 if empty)
```

`d34_idx` / `d5_idx` are *target* atom indices; `d34_counts` / `d5_counts` are per-root bucket
sizes, so the root index of every flat entry is `repeat_interleave(arange(N), counts)`. The fix is
exactly that: aggregate with the root index, then write to the root row.

Gate evidence (`analysis/cpu_gates.json`, 20/20):

| gate | result |
| --- | --- |
| `distance_chain_hand_computed` | 6-atom chain, `h = [0,1,4,9,16,25]` → expected `c34_0 = 12.5`, `c5_0 = 25`; got `c34 = [12.5, 20.5, 25.0, 0, 0.5, 2.5]`, `c5 = [25, 0, 0, 0, 0, 0]` — exact |
| `historical_bug_degenerates_to_self` | historical context vs own `h`: correlation **1.0**, max abs diff to `h_i·k/|D_i|` 8.9e-16; fixed vs historical max abs diff 4.49 |
| `distance_aggregation_matches_reference` | PyTorch path vs an independent Python loop over 423 atoms / 2 456 D34 + 6 652 D5 entries: max abs diff **0.0** |
| `distance_empty_bucket_zero` | empty buckets give exactly zero context and zero correction contribution |
| `distance_no_cross_molecule_mixing` | single-molecule vs batch: max diff 0.0; perturbing one molecule leaves the others at 0.0 |
| `distance_atom_permutation_and_order_invariance` | atom renumbering 1.4e-6, batch reorder 4.8e-7 |
| `distance_gradient_reaches_true_target_atoms` | gradient support equals bucket membership on all 110 atoms |
| `distance_target_perturbation_locality` | 0 mismatches on 24 tested atoms |
| `shared_init_and_distance_twin` | `DIST-TRUE-FIXED` and `DIST-BLIND` state dicts bitwise identical for every seed; different seeds differ |
| `permutation_utility_is_consistent` | the frozen `dc_data.permute_batch_atoms` re-draws its permutation (generator advances between the two draws); this round uses a single-draw local version. Training is unaffected (helper is audit-only) — recorded, not silently patched |

Content tests live in `tests/test_distance_fix.py` (18 pytest tests: chain values, empty buckets,
multi-molecule batch, renumbering, reorder, gradient, twin-init).

`DIST-BLIND` — the matched control — keeps the *same parameters, the same seed, the same gates*
(`has34`/`has5`, i.e. true bucket non-emptiness) but replaces the bucket content with the
per-molecule leave-one-out mean `m_except`, i.e. a molecule-level pooled context of the same shape
and scale. The contrast `DIST-TRUE-FIXED − DIST-BLIND` therefore isolates *whether the
distance-restricted identity of the neighbours matters*, not whether "more context" helps.

---

## 3. Protocol

### 3.1 Best-epoch rule (one rule, all 18 runs)

* Full Validation420 RMSE + MAE after every epoch (1…360).
* Save `state_dict` on **strict** improvement of Validation RMSE; ties keep the earlier epoch.
* No early stopping, no LR schedule responder, no soup/SWA/EMA, no Test signal.
* After training: reload the saved checkpoint, recompute Validation RMSE (must match the recorded
  value within 1e-5 fp32), compute Train RMSE from the reloaded model, write `curve.csv` (360 rows:
  loss, Train RMSE, Validation RMSE/MAE/R², `is_best`) and `train.json` (`best_epoch`,
  `epochs_run`, hash, git/config/seed provenance).
* `analysis/checkpoint_verify.json`: all 18 checkpoints are **not** the final-epoch state, their
  `best_epoch` equals the curve argmin, their reloaded Validation RMSE reproduces, and the saved
  per-molecule Validation predictions reproduce to ≤1.8e-6.

### 3.2 Pairing

Seeds `[0,1,2]`; inside a seed, all six models share the BASE initialisation stream, the shuffling
order, the dropout RNG stream, the recipe and the data order. Verified globally: the per-seed batch
order hash and dropout-mask hash are **identical across all six models**
(`checkpoint_verify.json → global_checks.batch_order_identical_within_seed`,
`dropout_rng_identical_within_seed`). Paired contrasts are therefore within-seed differences.

---

## 4. Main result — 3-seed means, ranked by Validation

`analysis/table_summary.csv`, `analysis/table_main.csv`. RMSE in logD units; "wins vs BASE" counts
seeds with a strictly lower RMSE. Test column is `EXPOSED_TEST_EXPLORATORY`.

| # (valid) | model | Validation RMSE | Validation MAE | Validation R² | Test RMSE | Test MAE | Test R² | wins vs BASE (valid / test) |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | **G2L-ALL** | 0.536916 ± 0.009393 | 0.39943 | 0.80922 | 0.541258 ± 0.024051 | 0.38681 | 0.81624 | 3 / 1 |
| 2 | **DIST-BLIND** | 0.539459 ± 0.003069 | 0.39872 | 0.80744 | **0.534773 ± 0.014981** | 0.37422 | **0.82076** | 3 / 3 |
| 3 | **ROLEBLIND** | 0.541109 ± 0.009805 | 0.40539 | 0.80622 | 0.537053 ± 0.022765 | 0.38288 | 0.81910 | 3 / 2 |
| 4 | **G2L-LOO** | 0.545251 ± 0.010796 | 0.40520 | 0.80324 | 0.544277 ± 0.018186 | 0.38785 | 0.81429 | 3 / 2 |
| 5 | **DIST-TRUE-FIXED** | 0.546892 ± 0.007182 | 0.40670 | 0.80208 | 0.554116 ± 0.024781 | 0.39022 | 0.80740 | 3 / 1 |
| 6 | **BASE** | 0.561962 ± 0.001350 | 0.41943 | 0.79105 | 0.547195 ± 0.006463 | 0.39511 | 0.81241 | — |

Per-run detail (best epoch, Train / Validation / Test RMSE) — `analysis/table_seed.csv`:

| model | s0 | s1 | s2 |
| --- | --- | --- | --- |
| BASE | 192 · 0.31760 / 0.562861 / 0.553415 | 176 · 0.33061 / 0.560410 / 0.547655 | 281 · 0.25105 / 0.562616 / 0.540514 |
| ROLEBLIND | 214 · 0.27738 / 0.530030 / 0.522817 | 115 · 0.39776 / 0.548668 / 0.563309 | 262 · 0.24371 / 0.544629 / 0.525035 |
| G2L-LOO | 282 · 0.23684 / 0.536620 / 0.529702 | 128 · 0.36999 / 0.557356 / 0.564657 | 275 · 0.23449 / 0.541776 / 0.538473 |
| G2L-ALL | 158 · 0.31993 / 0.528899 / 0.556188 | 176 · 0.30278 / 0.547252 / 0.554072 | 271 · 0.22498 / 0.534598 / 0.513513 |
| DIST-TRUE-FIXED | 291 · 0.19544 / 0.552642 / 0.563856 | 128 · 0.32936 / 0.549192 / 0.572548 | 215 · 0.23358 / 0.538842 / 0.525945 |
| DIST-BLIND | 181 · 0.28935 / 0.540002 / 0.545667 | 277 · 0.22253 / 0.542221 / 0.540962 | 262 · 0.22459 / 0.536156 / 0.517689 |

---

## 5. Paired within-seed contrasts, read against the noise floor

`analysis/table_pairs.csv`. Gain is `ref − candidate`; positive = candidate better.

**The noise floor.** Two literal repeat runs of an already-trained arm (same seed, same code, same
recipe; `analysis/nondeterminism.json`): `dist_blind_s0` best epoch 181 → 174 with Validation RMSE
0.540002 → 0.537033 (|Δ| = 0.00297); `dist_blind_s1` best epoch 277 → 214 with 0.542221 → 0.546457
(|Δ| = 0.00424). Mean |Δ| = **0.0036**, and the curves are never bitwise identical (CUDA
`index_add_` reduction order). **Any Validation difference below ≈0.004 is indistinguishable from
re-running the same model.** All statements below respect that.

| pair | seed 0 | seed 1 | seed 2 | wins | read |
| --- | --- | --- | --- | --- | --- |
| ROLEBLIND − BASE (valid) | +0.03283 | +0.01174 | +0.01799 | **3/3** | real (5.8× noise floor on the mean) |
| ROLEBLIND − BASE (test) | +0.03060 | −0.01565 | +0.01548 | 2/3 | one Test reversal |
| G2L-LOO − ROLEBLIND (valid) | −0.00659 | −0.00869 | +0.00285 | 1/3 | LOO gating hurts |
| G2L-LOO − ROLEBLIND (test) | −0.00689 | −0.00135 | −0.01344 | 0/3 | LOO gating hurts |
| G2L-ALL − ROLEBLIND (valid) | +0.00113 | +0.00142 | +0.01003 | 3/3 | mean +0.0042, 2/3 below noise |
| G2L-ALL − ROLEBLIND (test) | −0.03337 | +0.00924 | +0.01152 | 2/3 | no stable Test gain |
| DIST-TRUE-FIXED − DIST-BLIND (valid) | −0.01264 | −0.00697 | −0.00269 | 0/3 | real content is harmful |
| DIST-TRUE-FIXED − DIST-BLIND (test) | −0.01819 | −0.03159 | −0.00826 | 0/3 | same on Test |
| DIST-TRUE-FIXED − ROLEBLIND (valid) | −0.02261 | −0.00052 | +0.00579 | 2/3 | worse than plain role conditioning |
| DIST-TRUE-FIXED − ROLEBLIND (test) | −0.04104 | −0.00924 | −0.00091 | 0/3 | — |

Reading: **one** effect is larger than the noise floor and consistent on both splits — adding
molecule-level context on top of BASE (ROLEBLIND, and its distance-gated twin DIST-BLIND). Every
*structural* refinement on top of that (leave-one-out role gating, global-graph gating, true
distance-bucket content) moves the mean by ≤0.004 in either direction, i.e. inside noise, and its
per-seed signs disagree on Test.

---

## 6. Stability

`analysis/table_stability.csv` (Validation420 only).

* **Did best-epoch selection reduce ROLEBLIND seed variance?** Against the historical soup: yes —
  historical ROLEBLIND soup values 0.534232 (s0) and 0.586765 (s1), spread 0.0525; now
  0.530030 / 0.548668 / 0.544629, sd 0.0098. Against BASE it is still the least stable family
  (BASE sd 0.00135, DIST-BLIND 0.00307, ROLEBLIND 0.00980, G2L-ALL 0.00939, G2L-LOO 0.01080,
  DIST-TRUE 0.00718). Selection removed the catastrophic seed, not the family's sensitivity.
* **Are best epochs systematically early?** No. Best epochs span 115–291 of 360, per-model medians
  176 (BASE), 214 (ROLEBLIND), 275 (G2L-LOO), 176 (G2L-ALL), 215 (DIST-TRUE), 262 (DIST-BLIND).
  Four of eighteen runs peak before epoch 160, so "always early" is wrong; "never at 360" is right.
* **Is G2L-LOO more stable?** No — largest Validation sd (0.01080) and largest best-epoch range
  (128–282).
* **Structure vs seed outlier.** The seed-1 stream is the worst seed for 4 of 6 models
  (ROLEBLIND, G2L-LOO, G2L-ALL, DIST-TRUE) while BASE is flat across seeds (sd 0.0014). The
  outlier is therefore a *seed × model interaction* amplified by context conditioning (its
  trajectory is visibly noisier: ROLEBLIND s1 peaks earliest of all 18 runs, epoch 115, with the
  highest Train RMSE at its best epoch, 0.398, and the largest late Validation rise, +0.164), not a
  data-order or split artefact — the batch order is identical across models for that seed.
* **Is BASE still the most stable?** Yes, by an order of magnitude on Validation (0.00135) and on
  Test (0.00646). Stability and level trade off: BASE is the most reproducible and the worst model.
* **Does Validation rise late while Train falls?** Yes, in **18/18** runs epoch 360 is worse than
  the best epoch (mean degradation 0.0128–0.0164) while the raw Train RMSE at epoch 360 is
  0.20–0.24 versus 0.20–0.40 at the best epoch: the late epochs keep fitting the training set and
  lose Validation. This is exactly the regime where a best-epoch rule (or soup) matters, and it is
  why the rule is applied identically to all 18 runs.

---

## 7. Soup → best epoch (Q1)

Two independent comparisons, both Validation420 only.

**(a) Same-round proxy.** Mean of the last five epoch-end Validation RMSEs of each run (what a
last-5 soup is selected to approximate) versus the best epoch:

| model | last-5 mean (proxy) | best epoch | best − last-5 |
| --- | --- | --- | --- |
| BASE | 0.576072 | 0.561962 | −0.014109 |
| ROLEBLIND | 0.556231 | 0.541109 | −0.015122 |
| G2L-LOO | 0.558167 | 0.545251 | −0.012917 |
| G2L-ALL | 0.548814 | 0.536916 | −0.011898 |
| DIST-TRUE-FIXED | 0.562323 | 0.546892 | −0.015431 |
| DIST-BLIND | 0.552124 | 0.539459 | −0.012664 |

**(b) Historical soups (read-only reference, same recipe and seed).**

| arm | historical soup | this round, best epoch | Δ (best − soup) | historical best-curve epoch (not saved) |
| --- | --- | --- | --- | --- |
| `BASE_s0` | 0.580416 | 0.562861 | −0.01756 | 0.564350 @181 |
| `BASE_s1` | 0.576780 | 0.560410 | −0.01637 | 0.563294 @176 |
| `ROLEBLIND_s0` | 0.534232 | 0.530030 | −0.00420 | 0.527955 @282 |
| `ROLEBLIND_s1` | 0.586765 | 0.548668 | −0.03810 | 0.564655 @133 |

Best-epoch helps in all four, by 0.004 (inside the 0.0036 noise floor) to 0.038 (10×). The large
gain is on the seed where the soup was pathological — 0.586765 was worse than BASE_s1, which is
what motivated dropping the soup. ROLEBLIND's 3-seed mean 0.541109 is 0.0194 better than the mean
of its two soup values (0.5605).

*Protocol reproduction caveat.* The re-implemented loop reproduces the historical curves to
epoch-level equality of the *first* epoch (train-loss relative difference 2e-8…2e-6 for the arms
that used the same `dc_train` loop; 3e-8…1.3e-4 for the arms from the earlier round loop) but not
bitwise later, because of the GPU reduction-order non-determinism measured in §5. Curve pairs and
per-arm offsets are in `analysis/history_curve_check.json`. Differences of |Δ| ≲ 0.007 between this
round and a historical single-seed curve are therefore expected and are not protocol drift.

---

## 8. Best-epoch distribution

`analysis/answers.json → best_epoch_distribution`:

```
BASE              192 / 176 / 281
ROLEBLIND         214 / 115 / 262
G2L-LOO           282 / 128 / 275
G2L-ALL           158 / 176 / 271
DIST-TRUE-FIXED   291 / 128 / 215
DIST-BLIND        181 / 277 / 262
```

Median 214.5, range 115–291, no run selects epoch 360, median fraction of the budget used 0.60. The
distribution is model-dependent (G2L-LOO and DIST-BLIND peak late, ROLEBLIND and DIST-TRUE span the
whole range), which is another argument against a fixed epoch budget or a fixed soup window.

---

## 9. Test420 — `EXPOSED_TEST_EXPLORATORY`, one pass

Order of operations, all recorded: 18 trainings → `run_be.py verify` (`all_passed=True`,
`n_arms=18`, `test_read=False`) → `run_be.py manifest` → freeze
(`analysis/MANIFEST_FROZEN`, manifest sha `4747f9f649b36621`, `any_smoke=False`,
`test_read_at_freeze=False`) → `run_be.py eval-test`. The evaluation code refuses to run if any
manifest entry changed, if a checkpoint hash differs, or if the Validation RMSE does not reproduce
(≤1e-5) before the Test read.

Headline (3-seed means): best Test RMSE is **DIST-BLIND 0.534773**, then ROLEBLIND 0.537053,
G2L-ALL 0.541258, G2L-LOO 0.544277, BASE 0.547195, DIST-TRUE-FIXED 0.554116. Ranking inversions
against Validation (`analysis/answers.json → ranking.rank_inversions`):

| model | Validation rank | Test rank |
| --- | ---: | ---: |
| G2L-ALL | 1 | 3 |
| DIST-BLIND | 2 | 1 |
| ROLEBLIND | 3 | 2 |
| G2L-LOO | 4 | 4 |
| DIST-TRUE-FIXED | 5 | 6 |
| BASE | 6 | 5 |

Sign reversals between splits: `G2L-ALL − ROLEBLIND` (valid 3/3 wins, test 2/3 with a −0.0334 loss
on seed 0) and `ROLEBLIND − BASE` seed 1. Everything else keeps its sign.

**Historical reference, not a matched control:** PNA reached Validation 0.554034 / Test 0.528874 in
an earlier round with a different architecture *and* a different training protocol (native
training, no best-epoch rule). Our numbers are not comparable to it term by term; the only fair
statement is that the context-conditioned LCRC models now sit in the same Validation band
(0.537–0.547) as that historical PNA reference, with a Test read that is 0.006–0.025 worse.

**Single-seed vs 3-seed optimism** (`answers.json → single_seed_versus_3seed`): picking the seed
with the best Validation value and reporting its Test number is optimistic by up to +0.0282
(DIST-TRUE-FIXED) and can even hide a reversal (G2L-ALL: best-valid seed s0 has Test 0.5562 while
the 3-seed Test mean is 0.5413 — *pessimistic* for that model). One seed is not a result.

---

## 10. Gates

**20/20 passed**, CPU-only. Three artifacts, deliberately kept distinct:
`analysis/cpu_gates_frozen.json` is the artifact certified by `protocol_freeze.json` (its SHA256
equals the freeze's `cpu_gates_sha256`, its `code_hashes` equal the freeze's, 20/20);
`analysis/cpu_gates.json` is a re-run with the final code (20/20, same config hash, `code_hashes`
differ only in the two reporting-only modules added after the runs); and
`analysis/cpu_gates_final_code.json` records that comparison explicitly. Groups: data
identity (split/labels/cache hashes), row & label alignment, target normalisation fitted on Train
only, distance correctness (chain, reference loop, empty buckets, no cross-molecule mixing,
renumbering/reorder invariance, gradient locality, perturbation locality, bucket coverage),
twin-initialisation equality, zero-correction ≡ BASE, ρ/dropout semantics, metric arithmetic,
best-epoch tracker semantics (tie keeps earlier epoch), best-epoch end-to-end, and
"no Test access in the training path" (no Test key in the training views, no Test token in
`be_train.py`/`be_models.py`, target scaler `valid_or_test_used=False`).

---

## 11. Stage D — Bemis–Murcko scaffold shift (optional stage, development only)

**Scope.** This is a *different partition of the same 4,200 Lipophilicity molecules* whose labels
were already used, so it is a **distribution-shift development evaluation**, not a new independent
dataset and not a blind confirmation. It ran only after §9/§10 were signed off. Protocol:
`run_be.py scaffold-preflight` → 9 trainings → `scaffold-manifest` (frozen before the scaffold Test
read) → `scaffold-eval`.

**The split, verified not assumed** (`analysis/scaffold/split_audit.json`). Source:
`tracks/structure-contribution/results/bjoc_cross_task_lipophilicity_v1/split_lipo_v1.json`
(`deepchem==2.8.0` `ScaffoldSplitter`, verbatim algorithm; 3360 / 420 / 420; 2,408 scaffold groups,
0 groups crossing splits; `official_valid_loaded = false`, `official_test_loaded = false`).
Independent re-derivation from the frozen CSV: the three row lists are disjoint, their union is
exactly rows 0…4199, the split's embedded `train_y` equals the CSV `exp` column at those rows to the
last bit, and all three declared row hashes reproduce under this round's own int64 row-hash
serialisation. Normalisation is refitted on scaffold Train only (structure stats and target
scaler: mean 2.162905, scale 1.210993, `valid_or_test_used=false`), and every run restarts from
scratch with the same recipe and the same best-epoch rule.

**Models** were pre-registered by the rule "BASE, ROLEBLIND and the strongest other model by the
random42 3-seed mean **Validation** RMSE" → **G2L-ALL** (`analysis/scaffold/preflight.json`). No
Test420 information entered that choice. Gate: `analysis/scaffold/preflight_amendment.json` records
that all 9 runs trained under preflight v1, that the only files changed afterwards were the two
reporting-only modules (`be_analysis.py`, `be_ledger.py`), and that every training-path module hash
is identical in v1 and v2.

| model | best epochs | Validation RMSE | Validation MAE | scaffold-Test RMSE | scaffold-Test MAE |
| --- | --- | --- | --- | --- | --- |
| G2L-ALL | 240 / 230 / 194 | **0.722564 ± 0.004882** | 0.497878 | 0.714686 ± 0.005381 | 0.524912 |
| ROLEBLIND | 149 / 174 / 140 | 0.750177 ± 0.023095 | 0.507352 | 0.710395 ± 0.015781 | 0.528723 |
| BASE | 184 / 135 / 126 | 0.758291 ± 0.008879 | 0.514442 | 0.709644 ± 0.011837 | 0.539780 |

Per-seed Validation / scaffold-Test RMSE (`analysis/scaffold/table_scaffold.csv`):

| model | s0 | s1 | s2 |
| --- | --- | --- | --- |
| BASE | 0.750184 / 0.723304 | 0.767780 / 0.702379 | 0.756908 / 0.703249 |
| ROLEBLIND | 0.734535 / 0.710392 | 0.776702 / 0.694615 | 0.739294 / 0.726177 |
| G2L-ALL | 0.725627 / 0.712777 | 0.716934 / 0.720761 | 0.725130 / 0.710519 |

Paired, within seed (gain = reference − candidate, positive = candidate better):

| pair | Validation, s0 / s1 / s2 | wins | scaffold Test, s0 / s1 / s2 | wins |
| --- | --- | --- | --- | --- |
| ROLEBLIND − BASE | +0.015649 / −0.008922 / +0.017614 | 2/3 | +0.012912 / +0.007764 / −0.022928 | 2/3 |
| G2L-ALL − BASE | +0.024557 / +0.050846 / +0.031778 | **3/3** | +0.010527 / −0.018382 / −0.007270 | 1/3 |
| G2L-ALL − ROLEBLIND | +0.008908 / +0.059768 / +0.014164 | **3/3** | −0.002385 / −0.026146 / +0.015658 | 1/3 |

Reading (no scaffold-specific noise floor was measured — 3 seeds and 420 rows, so only differences
of ≥0.01 should be read as signal):

* **Absolute error roughly doubles** under scaffold shift (0.72–0.76 Validation versus 0.537–0.562
  under random42). The shift is real, so no random42 conclusion transfers for free.
* **The ROLEBLIND − BASE gain does not replicate.** It shrinks from +0.0209 (random42, 3/3 seeds) to
  +0.0081 (scaffold, 2/3, with one −0.0089 reversal) on Validation and to −0.0008 on scaffold Test.
  The molecule-level context effect of §5 is therefore **split-dependent**, and it is not a general
  structural-sufficiency result.
* **G2L-ALL is the only model that beats both others on scaffold Validation**, 3/3 with margins
  0.009–0.060 — larger than on random42 — and it also has the smallest Validation sd (0.0049). But
  on the scaffold Test split it is the *worst* of the three (1/3 wins vs BASE). This is the same
  pattern as §5 and §9: global-graph conditioning improves what Validation sees, and the improvement
  does not survive the held-out scaffolds. One plausible reading is that the graph-level summary
  helps the model interpolate within the training scaffolds' chemistry rather than transfer to new
  scaffolds — the data here cannot separate that from a 420-row Test-noise effect.
* Best epochs stay mid-training (126–240), again never at 360.

---

## 12. GPU ledger

`analysis/gpu_ledger.json` (`run_be.py ledger`), built from one `rr status --json` record per run
(`analysis/rr_status/`, 45 records including failures and cancellations) plus in-process timings.

| group | runs | GPU-hours |
| --- | ---: | ---: |
| formal arms (6 models × 3 seeds) | 18 | 3.209 |
| Stage D scaffold (incl. 2 retried duplicates and pre-CSV failures) | 19 | 2.402 |
| non-determinism repeats (same arm twice + one aborted launch) | 3 | 0.359 |
| smoke (CUDA, 6 families × 4 epochs on 64/32 rows; 3 records incl. 2 rejected launches) | 5 | 0.014 |
| **total** | **45** | **5.983 / 14.000** |

All runs on `NVIDIA A100-SXM4-40GB` (driver 550.163.01, torch 2.5.1+cu124), pools `res-gpu0` /
`res-gpu1` (one GPU per run, `cuda:0` in-process). Formal per-run wall clock: mean 628 s, range
580–663 s; scaffold mean 712 s. Charging rule: full rr-measured lifetime of every CUDA process of
this round, both pools summed (not wall-clocked). Executing under half of the budget is itself part
of the result: the failure mode of this round was not GPU scarcity but (i) an aggregation bug that
made a mechanism look active, and (ii) single-seed noise, which the repeats and 3 seeds now
quantify.

---

## 13. Reproduce

```bash
# CPU gates (no GPU, ~2 min)
python tracks/structural-sufficiency/lcrc_lipo_best_epoch_distance_fix_v1/code/run_be.py gates

# one arm on a GPU pool (rr; 360 epochs, ~10.5 min)
rr run res be-base-s0 --pool res-gpu0 \
  --result tracks/structural-sufficiency/lcrc_lipo_best_epoch_distance_fix_v1/analysis/runs/base_s0 \
  -- python tracks/structural-sufficiency/lcrc_lipo_best_epoch_distance_fix_v1/code/run_be.py train \
     --arm base_s0 --device cuda:0 --pool res-gpu0

# verification -> manifest freeze -> single Test420 pass -> tables (CPU)
python .../run_be.py verify
python .../run_be.py manifest
python .../run_be.py eval-test --label EXPOSED_TEST_EXPLORATORY
python .../run_be.py analysis
```

The CUDA smoke artifact (6 families, 4 epochs, `smoke: true`, excluded from every table by the
smoke guard) is in `analysis/smoke_cuda/`; `analysis/smoke_cpu/` holds the CPU audit run.

Artifacts: `analysis/runs/<arm>/{checkpoint.pt,curve.csv,train.json,diagnostics.json,train_predictions.npz,valid_predictions.npz}`,
`analysis/checkpoint_manifest.json`, `analysis/MANIFEST_FROZEN`, `analysis/eval_test/*` (per-molecule
Validation and Test predictions + `metrics.json`), `analysis/table_*.csv`, `analysis/answers.json`,
`analysis/nondeterminism/` + `nondeterminism.json`, `analysis/history_curve_check.json`,
`analysis/cpu_gates.json`, `analysis/data_audit.json`, `analysis/protocol_freeze.json`.

## 14. Limitations

* Three seeds. Model differences of ≤0.004 Validation RMSE cannot be resolved: the measured
  run-to-run noise floor is 0.0036 for a single arm, and the between-seed sd of the
  context-conditioned families is 0.003–0.011.
* Test420 is an exposed partition; it is reported once, after freezing, and is not used for
  selection. It is a consistency check, not confirmation.
* The historical comparisons in §7(b) are read-only references from earlier rounds whose loops and
  checkpoints differ in detail (soup instead of best epoch).
* Lipophilicity-only, logD-only, one split (random42) plus the scaffold split in §11; no claim
  about other tasks or about scaffold-generalisation beyond what §11 measures.
