# LCRC failure-mechanism P0 — report

**Round ID:** `lcrc-failure-mechanism-p0-v1`  
**Track:** `tracks/structural-sufficiency/`  
**Status:** completed, CPU-only, read-only with respect to all historical artefacts  
**Run timestamp (UTC):** 2026-11-03T13:15:58Z  
**Code commit at run:** `b54d3cfb` (working tree contains only this new track)

This round executes the pre-specified P0 diagnostic only. It does **not** train, fine-tune, re-infer, select checkpoints, search the literature, introduce new models/hypotheses, or change any threshold. All writes are confined to this track's `analysis/` directory.

---

## 0. Scope and pre-declared analysis

Two frozen evaluation groups are kept strictly separate:

| group | dataset / split | role | exposure |
|---|---|---|---|
| **A** `A_random42` | Lipophilicity, random42 (`torch random_split`, seed 42), Validation420 | LCRC-A-R3 vs PNA-native, paired per molecule | `FROZEN_VALIDATION_ONLY` |
| **B** `B_lipo_random122` | Lipophilicity, published random122 split, Validation420 + Test420 | BASE / CHEM8 / MULTICONTRIB / COMBO, seeds 0/1/2 where available | Validation `FROZEN_VALIDATION_ONLY`; Test `POST_HOC_EXPOSED_DISCOVERY` |

The two splits are different partitions of the same 4,200-molecule table. They are never pooled, never treated as the same evaluation set, and no per-molecule pairing is made across them. Group A Test predictions and labels are not read; Group B Test is read only under the post-hoc label and never selects a checkpoint.

Fixed constants, frozen before inspecting results:

- support signature: rooted radius-2 labelled ball, `SIGNATURE_VERSION = rooted-radius2-labelled-ball-v1`
- rare support: `1 <= document frequency <= 4`
- similarity: Morgan radius 2, 2048 bits
- metric reproduction tolerance: `1e-5`
- tail: top 10% of **BASE** absolute error, fixed per seed
- bootstrap: 2,000 replicates, Bemis–Murcko scaffold clusters, seed `20261103`
- conditional model: 5 whole-scaffold folds, Ridge `alpha = 10`, fold-local standardisation, no search
- `LOW_SUPPORT`: group `n < 30`

---

## 1. Frozen-input alignment and integrity (Task A)

### 1.1 What was loaded

**Group A** (`lcrc_lipo_esa_random42_v1`):

| item | value |
|---|---|
| split file | `data/split_esa_random42_v1.json`, SHA-256 `1834414a…c233b4` |
| sizes | train 3,360 / valid 420 / test 420 |
| row hashes | train `0df18c72…`, valid `8f9c3517…`, test `4bc44910…` |
| labels | `data/labels_train_valid_v1.json`, train+valid only, `test_labels_included: false` |
| LCRC-A-R3 predictions | `analysis/runs/lcrc_a_r3_s0/valid_predictions.npz`, SHA-256 `7f9ad391…d88e894` |
| PNA-native predictions | `analysis/runs/pna_native_s0/valid_predictions.npz`, SHA-256 `735df749…1ac905` |
| historical metric file | `analysis/metrics_validation.json`, SHA-256 `c540f9ac…` |

`n = 420` for both prediction files; strict row order equals the frozen Validation420 order; train/valid/test row sets are disjoint and cover all 4,200 molecules.

**Group B** (`lcrc_performance_longrun_v1` + published ChemBench split):

| item | value |
|---|---|
| split file | `Lipop.random122.ind.pkl`, SHA-256 `bf7acb17…24a7d` |
| labels | `Lipop.csv.gz`, SHA-256 `34280f0a…cd758` |
| sizes | train 3,360 / valid 420 / test 420 |
| row hashes | train `4e8d42a9…`, valid `31e3f28f…`, test `a4d92ad9…` |
| runs used | `p0_base_lipo_parity`, `p1_chem8_lipo`, `p1_contrib_lipo`, `p2b_combo_chem_contrib_lipo`, `p5_base_lipo_s1/s2`, `p5b_combo_lipo_s1/s2` |
| target normalisation | train-only mean `2.176985119047619`, scale `1.2091154725538142`, `n = 3360`; identical across runs |

Valid predictions are de-standardised with each run's own frozen `train.json:data.target_norm`; the Test npz files already contain de-standardised predictions. For every run, the given Test prediction equals `raw * scale + mean` within `1e-12`.

The published split is named random122; the frozen `train.json` records no split seed, so the audit records `split_seed = null` rather than asserting one.

Data gaps are recorded, not filled:

- seeds 1 and 2 have BASE and COMBO only; CHEM8 and MULTICONTRIB have no seed-1/seed-2 per-molecule files, so the 2×2 interaction is computed at seed 0 only.
- `p4_rho_lipo123/124` and `p4b_combo_lipo123/124` exist on random123/124 but have no matching BASE per-molecule file, so those splits are excluded.
- `p1_rho_lipo`, `p2_combo_rho_contrib_lipo`, `p3_base_scalem_lipo`, `p3_rho_scalem_lipo` are outside the pre-declared four-arm set and are excluded.

### 1.2 Metric reproduction

Every recomputed RMSE, MAE and SSE matches the frozen history within the `1e-5` gate; the maximum observed absolute difference is `2.84e-14` (floating-point round-off only).

Group A:

| model | recomputed RMSE | recomputed MAE | recomputed SSE |
|---|---:|---:|---:|
| LCRC-A-R3 (`lcrc_a_r3_s0`) | 0.5890072081930064 | 0.4208143415034744 | 145.71038634739426 |
| PNA-native (`pna_native_s0`) | 0.5540342318996931 | 0.4007184723844865 | 128.92065064900686 |

Group B, seed 0:

| arm | valid RMSE | valid MAE | test RMSE | test MAE |
|---|---:|---:|---:|---:|
| BASE | 0.7157746995984355 | 0.4147544542658695 | 0.5704458064646916 | 0.42411895467959168 |
| CHEM8 | 0.7027949346110950 | 0.4040006880337306 | 0.5659573475226861 | 0.4245368359742913 |
| MULTICONTRIB | 0.6003849081524373 | 0.3992705605877558 | 0.5739434185606259 | 0.4260087963513577 |
| COMBO | 0.7011459121417385 | 0.4036864031751311 | 0.5510418889639674 | 0.4120265041770213 |

Cross-checks against the long-run stage tables (`p1_table.csv`, `p2b_table.csv`, `p5_table.csv`, `p5b_table.csv`) also pass at `1e-5`; every Test row hash and every prediction-file hash matches the frozen record. All Group B checkpoints report `selection_uses_test: false`.

### 1.3 Group separation

`random42` and `random122` share **zero** validation-row order, and only 40 of 420 validation molecules overlap by identity (9.5%)—the expected overlap of two independent 10% draws. Train and Test row sets are also different. The audit stores `treated_as_same_evaluation_set: false`, `pooled_across_groups: false`, and `per_molecule_pairing_across_groups: false`.

Integrity result: **210 checks, 0 failed, 0 skipped** (`analysis/integrity_checks.json`).

---

## 2. Train-only structural-support dictionary (Task B)

### 2.1 Signature construction

For each heavy atom (root), the signature is the exact labelled ball of graph radius 2:

- node label: atomic number, formal charge, total H count, aromatic flag, in-ring flag, and node kind (`root` / `ball atom` / boundary stub);
- edge label: bond type (single, double, triple, aromatic, dative);
- boundary stubs carry the external atom's label and the bond that leaves the ball, so the root position and the first shell outside radius 2 are encoded;
- canonical serialisation uses RDKit `CanonicalRankAtoms(breakTies=True)` on a dummy-atom graph with isotope-encoded labels.

The signature is invariant to atom renumbering by construction. It contains no global atom index, no row id, no label, no split, and no dataset statistic.

### 2.2 Dictionary statistics

Document frequency is the number of **training molecules** containing a signature at least once (one count per molecule, not per atom occurrence). Query molecules never enter the dictionary.

| statistic | random42 (A train) | random122 (B train) |
|---|---:|---:|
| training molecules | 3,360 | 3,360 |
| training atom tokens | 91,083 | 90,863 |
| distinct signatures | 28,921 | 29,024 |
| `df = 1` | 19,391 | 19,335 |
| `df = 2–4` | 6,273 | 6,511 |
| `df = 5–9` | 1,753 | 1,710 |
| `df >= 10` | 1,504 | 1,468 |
| max `df` | 733 | 727 |
| dictionary SHA-256 | `fb844f53…` | `eacbc955…` |

### 2.3 Per-molecule features

Every molecule in `per_molecule.csv` and `support_diagnostics.csv` carries:

- `unseen_atom_frac`, `rare_atom_frac` (`1 <= df <= 4`), `median_log1p_df`, `p10_log1p_df` (df = 0 → `log1p(0) = 0`);
- `max_train_tanimoto` (Morgan r=2, 2048 bits, maximum over training molecules);
- Bemis–Murcko `scaffold` and `scaffold_seen`;
- `heavy_atoms`, `tpsa`, `mol_logp`, `hbd`, `hba`, `formal_charge`, `ring_count`;
- atom counts and explicit flag columns for handling (`parse_ok`, `n_components`, `multi_component`, `scaffold_empty`, `scaffold_fallback_used`).

Group B query summary:

| split | mean unseen frac | mean rare frac | mean median log1p df | molecules with ≥1 unseen atom |
|---|---:|---:|---:|---:|
| Validation420 | 0.2392 | 0.2329 | 1.7517 | 81.4% |
| Test420 | 0.2581 | 0.2427 | 1.6159 | 80.2% |
| train LOO (3,360) | 0.2479 | 0.2381 | — | 78.9% |

Group A Validation420: mean unseen frac `0.2517`, mean rare frac `0.2432`, mean median log1p df `1.6363`, mean max Tanimoto `0.644`.

The query distributions are close to the Train leave-one-molecule-out distribution; this split does not show a large aggregate support shift.

### 2.4 Support audits and special cases

- **Permutation invariance:** 25 molecules / 601 atoms renumbered at random; 0 signature mismatches.
- **Exactness vs VF2:** 60 molecules, 1,580 rooted balls, 1,291 distinct signatures. All 464 same-signature pairs are graph-isomorphic under node-label/bond-type VF2; none of 1,200 different-signature pairs is isomorphic. No collision was found.
- **LOO consistency:** all 3,360 Group-B training molecules were recomputed with the molecule removed from its own dictionary; four support fields match the stored LOO row within `1e-12`.
- **Handling (4,200 unique SMILES):** 0 RDKit parse failures; 1 multi-component record (row 1646, three water fragments plus the main component) kept as one disconnected graph with no salt stripping; 7 charged molecules; 5 acyclic molecules with an empty Murcko scaffold, keyed by canonical non-isomeric SMILES and flagged `scaffold_fallback_used = true`; no explicit-H molecules. Nothing is silently dropped.
- **Leakage guard:** the dictionary built from Train rows equals a manual document-frequency count over Train only, and differs from a dictionary that would include Validation/Test molecules. Perturbing the historical Test labels by `+1000` in memory leaves every Test support feature unchanged.

A known degeneracy: `p10_log1p_df` has a point mass at zero (in many molecules at least 10% of atoms have `df = 0`, so `log1p(df) = 0`), so its four quantile boundaries collapse to two populated bins (`Q1/Q2` empty). The empty bins are recorded in `combo_attribution.json:degenerate_bins` and are not merged post hoc.

---

## 3. Seed stability and tail error, random122 (Task C)

### 3.1 Per-seed overall metrics

Tail is always the top 10% of **same-seed BASE** absolute error, fixed before looking at the candidate.

**Validation420**

| seed | BASE RMSE | BASE MAE | BASE P90 AE | COMBO RMSE | COMBO MAE | COMBO P90 AE | ΔRMSE | ΔMAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.7158 | 0.4148 | 0.8554 | 0.7011 | 0.4037 | 0.8656 | −0.0146 | −0.0111 |
| 1 | 0.6767 | 0.4084 | 0.8113 | 0.6080 | 0.4026 | 0.8120 | −0.0688 | −0.0057 |
| 2 | 0.7330 | 0.4019 | 0.8171 | 0.5729 | 0.3717 | 0.8061 | −0.1602 | −0.0302 |
| **seed mean** | **0.7085** | **0.4083** | **0.8280** | **0.6273** | **0.3927** | **0.8279** | **−0.0812** | **−0.0157** |

**Test420** (`POST_HOC_EXPOSED_DISCOVERY`)

| seed | BASE RMSE | BASE MAE | BASE P90 AE | COMBO RMSE | COMBO MAE | COMBO P90 AE | ΔRMSE | ΔMAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.5704 | 0.4241 | 0.8787 | 0.5510 | 0.4120 | 0.8905 | −0.0194 | −0.0121 |
| 1 | 0.5634 | 0.4243 | 0.8996 | 0.5472 | 0.4101 | 0.8576 | −0.0161 | −0.0142 |
| 2 | 0.5779 | 0.4277 | 0.9556 | 0.5567 | 0.4112 | 0.8782 | −0.0212 | −0.0165 |
| **seed mean** | **0.5706** | **0.4254** | **0.9113** | **0.5517** | **0.4111** | **0.8755** | **−0.0189** | **−0.0142** |

All three seeds favour COMBO on RMSE and MAE; there is no direction reversal. The *size* of the Validation gain varies about tenfold across seeds (ΔRMSE SD `0.0601`), whereas the Test gain is stable (ΔRMSE SD `0.0021`). The Validation seed-mean ΔRMSE (−0.0812) is much larger than the Test seed-mean ΔRMSE (−0.0189), consistent with the already-documented Validation/Test divergence of this round.

Seed-mean P90 absolute error is essentially unchanged on Validation (0.8280 → 0.8279) and improves on Test (0.9113 → 0.8755), but the Test improvement is carried by seeds 1 and 2; seed 0 is slightly worse (0.8787 → 0.8905).

### 3.2 Tail results

| split / seed | n tail | BASE RMSE | COMBO RMSE | ΔRMSE | BASE MAE | COMBO MAE | ΔMAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| valid / 0 | 42 | 1.9724 | 1.8756 | −0.0969 | 1.4276 | 1.2953 | −0.1323 |
| valid / 1 | 42 | 1.8356 | 1.5219 | −0.3137 | 1.3320 | 1.2105 | −0.1215 |
| valid / 2 | 42 | 2.0474 | 1.4540 | −0.5934 | 1.3738 | 1.1702 | −0.2036 |
| test / 0 | 42 | 1.3469 | 1.2363 | −0.1106 | 1.3101 | 1.1771 | −0.1330 |
| test / 1 | 42 | 1.3050 | 1.1905 | −0.1145 | 1.2621 | 1.1157 | −0.1464 |
| test / 2 | 42 | 1.3448 | 1.2171 | −0.1277 | 1.3015 | 1.1184 | −0.1831 |

Tail membership is only moderately seed-stable:

- Validation: 17 of 42 molecules are in all three BASE tails; 40 are in at least two; 29 in exactly one.
- Test: 24 of 42 are in all three; 38 in at least two; 26 in exactly one.

The “stable tail” (in ≥2 of 3 seeds) has `n = 40` on Validation and `n = 38` on Test; its per-seed COMBO-vs-BASE RMSE improvements are the same direction as the full-tail results but are not independent of the tail selection.

### 3.3 Scaffold-cluster bootstrap

2,000 whole-scaffold resamples; no molecule-independent interval is used to narrow anything.

- Validation: 357 clusters, 317 singletons, largest cluster 7 molecules (1.7%).
- Test: 357 clusters, 314 singletons, largest cluster 12 molecules (2.9%).
- `low_cluster_support = false` by the pre-declared rule (`<30` clusters or largest cluster `>10%`).

| contrast | point | 95% interval |
|---|---:|---:|
| valid, seed-mean ΔRMSE | −0.0812 | [−0.1906, 0.0126] |
| valid, seed-mean ΔMAE | −0.0157 | [−0.0363, 0.0009] |
| test, seed-mean ΔRMSE | −0.0189 | [−0.0367, −0.0014] |
| test, seed-mean ΔMAE | −0.0142 | [−0.0278, −0.0003] |
| test, seed 0 tail ΔRMSE | −0.1106 | [−0.2080, −0.0249] |
| test, seed 1 tail ΔRMSE | −0.1145 | [−0.2010, −0.0182] |
| test, seed 2 tail ΔRMSE | −0.1277 | [−0.2127, −0.0409] |
| valid, seed 0 tail ΔRMSE | −0.0969 | [−0.1604, 0.0069] |
| valid, seed 1 tail ΔRMSE | −0.3137 | [−0.6612, 0.0255] |
| valid, seed 2 tail ΔRMSE | −0.5934 | [−1.2687, 0.0495] |

Per-seed overall ΔRMSE intervals on Test all cross zero; the seed-mean interval just excludes zero. Tail intervals exclude zero on Test for all three seeds, but not on Validation.

Prediction-sign stability across seeds (BASE): mean sign agreement `0.918` (Validation) / `0.918` (Test); 75.5% / 75.2% of molecules keep the same residual sign in all three seeds. Per-molecule mean squared error is lower for COMBO on 219/420 Validation molecules and 225/420 Test molecules.

---

## 4. Controlled conditional model, random42 Validation420 (Task D)

### 4.1 Target and design

`d_i = SE_LCRC - SE_PNA` where `SE = (prediction - y)^2`. Positive `d_i` means PNA has the smaller squared error.

- `n = 420`; mean `d_i = 0.03998`, median `−0.00038`, SD `0.74540`; 49.76% positive, 50.24% negative.
- The aggregate RMSE gap (LCRC 0.5890 vs PNA 0.5540) is therefore not a uniform per-molecule shift; it is driven by a relatively small set of large paired errors (range −4.805 to +6.968).
- M0: `heavy_atoms, tpsa, mol_logp, hbd, hba, formal_charge, ring_count, max_train_tanimoto, scaffold_seen`.
- M1: M0 plus `unseen_atom_frac, rare_atom_frac, median_log1p_df, p10_log1p_df`.
- Same deterministic whole-scaffold 5-fold assignment for both models (84 molecules / 74 scaffolds per fold); fold-local standardisation; fixed Ridge `alpha=10`; OOF predictions kept.

### 4.2 OOF predictive comparison

| model | OOF MSE | OOF MAE | OOF RMSE |
|---|---:|---:|---:|
| M0 | 0.5561095856590219 | 0.3778530371407109 | 0.7457275545794334 |
| M1 | 0.5670097961863908 | 0.3858746743381504 | 0.7530005286760367 |
| **M1 − M0** | **+0.0109002105273689** | **+0.0080216371974395** | **+0.0072729740966033** |

The four support variables do **not** improve held-out prediction of `d_i`; M1 is slightly worse on both MSE and MAE. The scaffold-cluster bootstrap for the difference:

- ΔMSE `+0.01090`, 95% interval `[−0.00123, 0.02402]`; 96.0% of resamples positive.
- ΔMAE `+0.00802`, 95% interval `[−0.00033, 0.01590]`; 97.2% positive.

Four of five folds have a positive ΔMSE; fold 1 is slightly negative (−0.00387). No fold shows a material M1 gain.

### 4.3 Descriptive coefficients (standardised, full sample; not causal)

M1 support coefficients: `unseen_atom_frac −0.1299`, `rare_atom_frac +0.0025`, `median_log1p_df −0.0764`, `p10_log1p_df −0.0464`. These are predictive associations under collinearity with size, polarity and similarity, not effects.

The support variables are themselves strongly related to the base covariates: Spearman `unseen_atom_frac` vs `max_train_tanimoto = −0.79`, vs `median_log1p_df = −0.69`, vs `p10_log1p_df = −0.84`; `p10_log1p_df` vs Tanimoto `+0.57`. M1 therefore tests incremental value over an already similarity-aware base model, not the total explanatory power of “support”.

### 4.4 Stratified Spearman correlations with `d_i`

Overall correlations are essentially zero:

| support variable | Spearman rho | p |
|---|---:|---:|
| `unseen_atom_frac` | −0.0037 | 0.940 |
| `rare_atom_frac` | +0.0030 | 0.950 |
| `median_log1p_df` | −0.0056 | 0.909 |
| `p10_log1p_df` | −0.0275 | 0.574 |

Within heavy-atom / TPSA / Tanimoto quartiles and seen/unseen scaffolds, some stratum correlations reach |rho| ≈ 0.18–0.23 (e.g. `p10_log1p_df` vs `d_i` in TPSA Q2, rho −0.23), but they are not consistent in sign across strata and are not selected as findings. Every populated group had n ≥ 73; the two empty `p10_log1p_df` quartile bins are recorded as degenerate rather than treated as `LOW_SUPPORT` evidence.

### 4.5 Support-quartile trends

Mean `d_i` by quartile (positive = PNA better):

| variable | Q1 | Q2 | Q3 | Q4 |
|---|---:|---:|---:|---:|
| `unseen_atom_frac` | +0.0180 | +0.0201 | +0.1816 | −0.0644 |
| `rare_atom_frac` | +0.0106 | −0.0677 | +0.1443 | +0.0740 |
| `median_log1p_df` | −0.0754 | +0.1601 | +0.0254 | −0.0199 |
| `p10_log1p_df` | — | — | +0.0564 (n=284) | +0.0056 (n=136) |

Medians are near zero in every quartile. There is no monotone “lower support → larger LCRC penalty” pattern. The `p10_log1p_df` quartiles again collapse to two populated bins because of the point mass at zero.

**Task D answer:** on random42 Validation420, the exact-radius-2 support features do not explain the LCRC−PNA gap in either OOF prediction or rank association. This is a null result for this operationalisation of support, not a proof that no structural-distribution effect exists.

---

## 5. COMBO benefit attribution, random122 (Task E)

### 5.1 Overall and per-seed four-arm table

Seed 0, all four arms (overall RMSE / MAE / P90 absolute error):

| split | arm | RMSE | MAE | P90 AE |
|---|---|---:|---:|---:|
| valid | BASE | 0.7158 | 0.4148 | 0.8554 |
| valid | CHEM8 | 0.7028 | 0.4040 | 0.8200 |
| valid | MULTICONTRIB | **0.6004** | **0.3993** | 0.8577 |
| valid | COMBO | 0.7011 | 0.4037 | 0.8656 |
| test | BASE | 0.5704 | 0.4241 | 0.8787 |
| test | CHEM8 | 0.5660 | 0.4245 | 0.8575 |
| test | MULTICONTRIB | 0.5739 | 0.4260 | 0.9122 |
| test | COMBO | **0.5510** | **0.4120** | 0.8905 |

COMBO vs BASE per seed:

| split | seed | ΔRMSE | ΔMAE | ΔSSE |
|---|---:|---:|---:|---:|
| valid | 0 | −0.0146 | −0.0111 | −8.71 |
| valid | 1 | −0.0688 | −0.0057 | −37.10 |
| valid | 2 | −0.1602 | −0.0302 | −87.84 |
| test | 0 | −0.0194 | −0.0121 | −9.14 |
| test | 1 | −0.0161 | −0.0142 | −7.53 |
| test | 2 | −0.0212 | −0.0165 | −10.10 |

The sign is stable (3/3 seeds better on both RMSE and MAE; no direction reversal), but the Validation magnitude is seed-dependent and much larger than the Test magnitude.

### 5.2 Scaffold-seen vs scaffold-unseen

`scaffold_seen` means the Bemis–Murcko scaffold occurs in the training split after the fallback policy.

| split | group | n | BASE RMSE | CHEM8 RMSE | MULTICONTRIB RMSE | COMBO RMSE | COMBO ΔRMSE |
|---|---|---:|---:|---:|---:|---:|---:|
| valid | unseen | 201 | 0.9015 | 0.8871 | 0.7351 | 0.9024 | +0.0009 |
| valid | seen | 219 | 0.4864 | 0.4743 | 0.4420 | 0.4420 | −0.0444 |
| test | unseen | 201 | 0.6756 | 0.6648 | 0.6750 | 0.6571 | −0.0185 |
| test | seen | 219 | 0.4530 | 0.4568 | 0.4621 | 0.4314 | −0.0216 |

MULTICONTRIB's large Validation gain is concentrated on unseen scaffolds (0.7351 vs 0.9015, Δ−0.1664) and vanishes on Test (0.6750 vs 0.6756, Δ−0.0005). COMBO's Test gain is similar for seen and unseen scaffolds.

### 5.3 Support, size and TPSA quartiles (seed 0, COMBO ΔRMSE vs BASE)

Quartile edges are fitted on Validation420 and applied unchanged to Test420.

| grouping | Q1 | Q2 | Q3 | Q4 |
|---|---:|---:|---:|---:|
| valid, `unseen_atom_frac` | −0.034 | −0.056 | −0.013 | +0.001 |
| valid, `rare_atom_frac` | −0.044 | −0.017 | +0.040 | −0.012 |
| valid, `median_log1p_df` | −0.008 | −0.012 | −0.026 | −0.033 |
| valid, `heavy_atoms` | −0.047 | 0.000 | −0.015 | +0.026 |
| valid, `tpsa` | −0.006 | −0.028 | +0.037 | −0.038 |
| test, `unseen_atom_frac` | −0.009 | −0.028 | −0.020 | −0.021 |
| test, `rare_atom_frac` | −0.001 | −0.039 | −0.007 | −0.020 |
| test, `median_log1p_df` | −0.021 | −0.016 | −0.025 | −0.017 |
| test, `heavy_atoms` | −0.056 | −0.025 | +0.025 | −0.026 |
| test, `tpsa` | −0.015 | −0.017 | −0.057 | +0.009 |

On Test, COMBO improves most quartiles by small amounts, but there is no monotone support gradient; one size quartile (Q3) and the top TPSA quartile are slightly worse. The support quartiles do not isolate the benefit.

### 5.4 BASE-defined tail

| split / seed | tail n | BASE RMSE | COMBO RMSE | ΔRMSE | BASE MAE | COMBO MAE | ΔMAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| valid / 0 | 42 | 1.9724 | 1.8756 | −0.0969 | 1.4276 | 1.2953 | −0.1323 |
| valid / 1 | 42 | 1.8356 | 1.5219 | −0.3137 | 1.3320 | 1.2105 | −0.1215 |
| valid / 2 | 42 | 2.0474 | 1.4540 | −0.5934 | 1.3738 | 1.1702 | −0.2036 |
| test / 0 | 42 | 1.3469 | 1.2363 | −0.1106 | 1.3101 | 1.1771 | −0.1330 |
| test / 1 | 42 | 1.3050 | 1.1905 | −0.1145 | 1.2621 | 1.1157 | −0.1464 |
| test / 2 | 42 | 1.3448 | 1.2171 | −0.1277 | 1.3015 | 1.1184 | −0.1831 |

The tail definition is reproducible from same-seed BASE predictions only (integrity check `base_tail_indices_reproduced_from_base_predictions_only`).

### 5.5 Fraction improved and SSE concentration

| split / seed | fraction of molecules with SE improved | net SSE reduction | top-5 positive SSE | top-10 positive SSE | gross positive | gross negative |
|---|---:|---:|---:|---:|---:|---:|
| valid / 0 | 0.5333 | 8.71 | 11.98 | 15.48 | 38.64 | −29.93 |
| valid / 1 | 0.4952 | 37.10 | 46.76 | 49.51 | 69.76 | −32.66 |
| valid / 2 | 0.5143 | 87.84 | 90.99 | 94.34 | 118.21 | −30.36 |
| test / 0 | 0.5286 | 9.14 | 8.32 | 11.97 | 34.41 | −25.27 |
| test / 1 | 0.5286 | 7.53 | 6.61 | 11.13 | 39.13 | −31.60 |
| test / 2 | 0.5214 | 10.10 | 5.34 | 9.80 | 44.95 | −34.85 |

The net Test SSE reduction is 7.5–10.1, against gross positive contributions of 34–45 and gross negative contributions of −25 to −35. Top-10 negative contributions range from −8.70 to −13.51 (Validation) and −8.45 to −11.61 (Test). A handful of molecules dominate the gain: top-5 positive contributions are 53–91% of the net Test reduction (they can exceed 100% when other molecules regress). On Validation seed 2, top-5 positive contributions (90.99) exceed the net reduction (87.84). These top-k tables are post-hoc selected extremes and are not independent effect estimates.

### 5.6 2×2 interaction (all four arms, seed 0 only)

`MSE(COMBO) − MSE(CHEM8) − MSE(MULTICONTRIB) + MSE(BASE)`; positive means the joint change is worse than the sum of the two single changes.

| split | sample | n | interaction MSE | 95% interval |
|---|---|---:|---:|---:|
| valid | overall | 420 | +0.1496 | [−0.0083, 0.4432] |
| valid | BASE tail | 42 | +1.4674 | [−0.0694, 1.4111] |
| valid | scaffold seen | 219 | +0.0116 | — |
| valid | scaffold unseen | 201 | +0.2998 | — |
| test | overall | 420 | −0.0207 | [−0.0525, 0.0098] |
| test | BASE tail | 42 | −0.0152 | [−0.1396, 0.0744] |
| test | scaffold seen | 219 | −0.0309 | — |
| test | scaffold unseen | 201 | −0.0095 | — |

On Validation the interaction is positive and large relative to the single-arm differences: MULTICONTRIB's gain does not add to CHEM8, and COMBO ends up close to BASE. On exposed Test the interaction is small and slightly negative. All computed intervals (overall and BASE-tail) cross zero; the scaffold cells are descriptive point estimates only. Seeds 1 and 2 have no CHEM8/MULTICONTRIB files, so no interaction is computed there.

---

## 6. What the evidence supports

### 6.1 Supported by these measurements

1. **The frozen artefacts are alignable and reproducible.** All RMSE/MAE/SSE values reproduce within `2.84e-14`; row order, row hashes, label hashes, prediction hashes, split membership and model seeds are consistent.
2. **The support dictionary is exact and Train-only.** The rooted radius-2 signature passes renumbering and VF2 exactness checks; the dictionary equals the manual Train document frequencies; Validation/Test molecules and Test labels cannot change it.
3. **On random42, the LCRC−PNA gap is not explained by the four support features.** M1 does not improve OOF prediction of `d_i` (ΔMSE `+0.0109`, ΔMAE `+0.0080`), overall rank correlations are ~0, and the quartile trends are non-monotone. The paired gap is sign-balanced across molecules and heavy-tailed.
4. **On random122, COMBO's exposed-Test gain is small, sign-stable, and concentrated.** Seed-mean ΔRMSE is `−0.0189` (95% CI `[−0.0367, −0.0014]`), per-seed intervals include zero, and top-5 molecules account for 53–91% of the net SSE reduction. The BASE tail improves consistently on Test (ΔRMSE −0.11 to −0.13, intervals exclude zero).
5. **MULTICONTRIB's Validation gain does not transfer to Test.** Validation RMSE is 0.6004 vs BASE 0.7158 (Δ−0.1154), concentrated on unseen scaffolds (0.7351 vs 0.9015); Test RMSE is 0.5739 vs 0.5704 (Δ+0.0035). This is a Validation-specific association in the frozen runs, not a Test effect.
6. **The three seeds do not agree on the size of the Validation gain** (ΔRMSE −0.0146, −0.0688, −0.1602). Test gains are much more stable in magnitude (ΔRMSE −0.0194, −0.0161, −0.0212).

### 6.2 Not supported / not shown

- **Not supported:** a claim that low exact-radius-2 support is a dominant cause of the LCRC−PNA gap on random42. The incremental predictive test is null-to-slightly-negative, and the support distributions of Validation/Test are close to the Train LOO distribution.
- **Not shown:** a causal effect of support, chemistry features, or COMBO. All conditional quantities are predictive associations under collinearity; no intervention or randomisation was performed.
- **Not shown:** a mechanism for the MULTICONTRIB Validation gain. The data show where it appears (unseen scaffolds) and that it does not transfer to Test; they do not identify why.
- **Not shown:** a uniform COMBO improvement. The gain is a small net difference of larger positive and negative per-molecule contributions, concentrated in a few molecules, and P90 absolute error does not improve at seed 0.
- **Not shown:** any random123/random124 or other-split result. Those runs lack a matching BASE per-molecule file and were excluded rather than paired across splits.

### 6.3 Interpretation boundary

Test420 is `POST_HOC_EXPOSED_DISCOVERY`: it was already visible in the historical round and cannot serve as blind confirmation. The Test intervals in this report describe the frozen predictions on the exposed split; they are not a prospective estimate. No checkpoint was selected or reselected from Test.

---

## 7. Uncertainty and limitations

- **Sample size:** 420 molecules per evaluation split. The conditional model has 420 rows; the Test tail has 42 molecules per seed.
- **Bootstrap unit:** Bemis–Murcko scaffold. With 357 clusters and 314–317 singletons, many resamples are effectively molecule-level for singleton scaffolds; no molecule-independent interval is used to narrow any interval. Conditional-model clusters: 370, with 340 singletons.
- **Seed coverage is asymmetric:** Group A has one seed per model; Group B has BASE/COMBO at seeds 0/1/2 but CHEM8/MULTICONTRIB only at seed 0. Seed variability is not a variance decomposition of the training pipeline.
- **Support operationalisation is one choice:** exact rooted radius-2 labelled balls with Train document frequency. The null result applies to this operationalisation; a different radius, label set, or reference distribution could behave differently.
- **Collinearity:** support features correlate strongly with Tanimoto and size (`unseen_atom_frac` vs Tanimoto rho −0.79), so M1 has limited independent variation. The coefficient signs are not interpretable as effects.
- **Degenerate bins:** `p10_log1p_df` has only two populated quartile bins because of a point mass at zero; empty bins are recorded and not merged.
- **Group B split seed:** the published random122 file records no split seed; only the split file and row hashes are asserted.
- **Test exposure:** all Test summaries are post-hoc and exposed; they can support description, not confirmation.
- **No cross-split pairing:** random42 and random122 are different partitions; all per-molecule comparisons are within one split and one seed.

---

## 8. Reproducibility and outputs

### Commands

```bash
uv run python tracks/structural-sufficiency/lcrc_failure_mechanism_p0_v1/code/run_p0.py
uv run pytest -q tracks/structural-sufficiency/lcrc_failure_mechanism_p0_v1/tests
```

The full pipeline runs in about 40 seconds on CPU (no GPU). The 24 tests cover row-shuffle invariance, strict alignment, duplicate-row rejection, hand-computed SSE/RMSE/MAE, `d_i` sign, signature renumbering and label encoding, document-frequency/LOO, Train-only dictionary guards, Test-label independence, scaffold folds, frozen Group A/B loading, group separation, and output self-consistency.

### Deliverables

| path | contents |
|---|---|
| `analysis/manifest.json` | round id, git/environment, input audit, flags, constants, output hashes, repro commands |
| `analysis/per_molecule.csv` | 1,260 rows: 420 Group A valid + 420 Group B valid + 420 Group B Test, with predictions, errors, `d_i`, OOF, support features, descriptors, scaffolds, tail flags |
| `analysis/support_diagnostics.csv` | 7,980 rows: Group A/B query + Train LOO support rows |
| `analysis/conditional_model_results.json` | M0/M1 folds, OOF metrics, coefficients, correlations, quartile trends |
| `analysis/combo_attribution.json` | per-seed/split overall, subgroup, tail, contribution, interaction and bootstrap tables |
| `analysis/seed_stability.json` | per-seed metrics, tail overlap, stable tail, scaffold bootstrap |
| `analysis/integrity_checks.json` | 210 checks with pass/fail detail and blocking policy |
| `analysis/combo_group_table.csv` | long-form group attribution table used by the report |

`per_molecule.csv` has NaN only in columns that are structurally unavailable for a group (Group A has no BASE/CHEM8/MULTICONTRIB/COMBO predictions; Group B has no LCRC/PNA predictions or `d_i`). No molecule was dropped.

No historical file was modified. The only working-tree change is the new `tracks/structural-sufficiency/lcrc_failure_mechanism_p0_v1/` directory.

---

## Appendix A — integrity check inventory (210 checks)

| category | examples | result |
|---|---|---|
| Group A split/labels/predictions | SHA-256, row order, row isolation, n, seed, metric reproduction (RMSE/MAE/SSE) | all pass |
| Group B split/CSV/runs | split and CSV SHA-256, row hashes, run identity, train-only target norm, Test-prediction reconstruction, metric reproduction (RMSE/MAE/SSE), stage-table cross-checks | all pass |
| Group separation | different train/valid/test row sets, no pooling, no cross-group pairing | all pass |
| Signature | permutation invariance (0/601 mismatches), VF2 exactness (0 collisions), root/bond/charge/ring encoding | all pass |
| Support | Train-only dictionary equals manual document frequency, query rows absent from dictionary, LOO reproduction on 3,360 molecules, Test-label perturbation independence | all pass |
| Alignment | row-shuffle invariance, strict-order rejection, duplicate-row rejection | all pass |
| Conditional | `d_i` recomputation, OOF MSE/MAE recomputation, same folds for M0/M1, whole scaffolds, fold sizes 84×5 | all pass |
| Attribution | BASE-tail indices reproduced from BASE predictions only | all pass |
| Per-molecule table | 1,260 rows with frozen split order and no support-feature NaN | all pass |
| Synthetic | hand-computed SSE/RMSE/MAE; `d_i` sign | all pass |

Full detail is in `analysis/integrity_checks.json`.

## Appendix B — key frozen values used in this report

- Group A split seed: `42`; sizes `3360/420/420`; split SHA-256 `1834414a…`; valid rows SHA-256 `8f9c3517…`.
- Group B split: published `random122`; sizes `3360/420/420`; split SHA-256 `bf7acb17…`; valid rows `31e3f28f…`; Test rows `a4d92ad9…`.
- Group A metric file SHA-256 `c540f9ac…`; Group B CSV SHA-256 `34280f0a…`.
- Bootstrap seed `20261103`; 2,000 replicates; 2.5/97.5 percentile interval.
- Ridge `alpha = 10`; 5 folds; 84 molecules and 74 scaffolds per Group A fold.
- Tail fraction `0.10`; `LOW_SUPPORT_N = 30`; `TOL = 1e-5`.
