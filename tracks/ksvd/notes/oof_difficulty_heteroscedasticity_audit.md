# OOF Difficulty / Heteroscedasticity Confirmation (ZINC-12k official TRAIN, frozen compact-v4-hinge)

**Status:** FINAL — 2026-09-10
**Phase:** diagnostic gate (follow-up to the Post-v4 Residual Audit); **no model development**
**Questions answered:** Q1–Q14 · Tables A–E · Figures 1–6
**Decision:** **GO (difficulty)** — cross-fitted OOF difficulty Spearman 0.374 ≥ 0.25 with 5/5 folds same sign; **GO (epistemic)** — disagreement (seed-spread) vs per-molecule error Spearman 0.393 ≥ 0.30 with 5/5 folds same sign.
**Mode:** the only new computation is K-fold (K=5) **out-of-fold training** on official **train** with the frozen compact-v4-hinge protocol. Official **test and validation splits were never loaded**; architecture/loss/features untouched; only `|residual|` (difficulty) is studied — signed residuals appear only as context, never in verdicts.

---

## 0. Executive summary

| Fact | Value | Status |
|---|---|---|
| Frozen baseline (context, recorded prior) | compact-v4-hinge, official valid MAE 0.1700656 (seed 0 run 20260909-194445, 99,613 params); official test MAE 0.136885±0.005552 (primary protocol, seeds 0–3) | never re-loaded in this audit |
| OOF protocol | K=5, fold_seed=0; per outer fold nested inner 7200-train / 800-inner-valid; frozen v4-hinge training; best-epoch checkpoint on inner-valid; held-out fold = true OOF | 10000 train molecules × 2 model seeds |
| Determinism gate | fold 0 seed 0 rerun bit-identical (trace, npz SHA, state SHA, predictions); states forward-gate max diff 0.0 on all 5 folds | VERIFIED |
| Pooled OOF MAE (per-molecule, train) | seed 0: 0.1758 · seed 1: 0.1743 · mean-of-seeds 0.1750 · 2-seed ensemble (mean of preds) 0.1604 | — |
| Ensemble gain vs mean per-seed | ΔMAE −0.0146 (−8.3% relative) | VERIFIED |
| Difficulty signal (cross-fitted Ridge, model-visible features) | Spearman 0.3736 (5/5 folds same sign); hardest/easiest predicted-quintile MAE ratio 3.67 (Q1 0.095 → Q5 0.351) | **GO** (gate 0.25, ≥4/5) |
| Difficulty signal (all features incl. disagreement) | Spearman 0.490, R²(log) 0.264 (5/5) | — |
| Rarity vs difficulty | monotone: MAE 0.133 (rare≤5 = 0, n=4316) → 0.417 (rare≤5 > 0.2, n=424); Spearman(mean-inverse-1+freq) 0.339, (rare≤5) 0.309, (OOV) 0.277 — all 5/5 same sign | VERIFIED (variance side only) |
| Epistemic (disagreement) signal | Spearman(disagreement, mean-abs-error) 0.3933 overall; per-fold 0.359–0.420, 5/5 same sign | **GO** (gate 0.30, ≥4/5) |
| Long-cycle / topology (context) | cycle-length-7/8 and hinge-6/7/10 bands positively associated with difficulty (Spearman ≈ 0.10–0.14, 5/5) — consistent with the already-accounted long-cycle mechanism, no new signal | context |
| Signed side (context only) | pooled mean signed residual −0.040 (std 0.62); leading features' signed correlations ≪ their abs correlations (e.g. rare≤5 signed −0.08 vs abs +0.31) | no signed claim |
| y-level control | difficulty is *not* a pure y-level artifact: within both halves of target y the model-visible difficulty predictor still separates MAE (low-y half 0.136 vs 0.280; high-y half 0.102 vs 0.151) | VERIFIED |

**Bottom line:** on the official **train** split, per-molecule OOF error of the frozen v4-hinge protocol is (i) *heteroscedastic difficulty*, strongly and monotonically predicted by token rarity and by seed-disagreement, with a cross-fitted Spearman of 0.374 (visible features) / 0.490 (incl. disagreement) and 5/5 fold consistency — well past the pre-registered difficulty GO gate — and (ii) substantially *seed-epistemic* (disagreement–error Spearman 0.393, 5/5; 2-seed ensembling recovers 8.3%). This is the train-side confirmation of the validation-side conclusion ("NO CLEAR SECONDARY REPRESENTATION SIGNAL"): remaining error is **predictive difficulty / heteroscedastic conditional variance**, not a hidden signed structure the model is missing on molecules it has seen.

---

## 1. Scope, frozen objects, protocol

* Module: `tracks/ksvd/experiments/luyin16/zinc_oof_difficulty_audit.py` (stages `splits → fold → states → features → table → analysis → predictor → figures → decision`).
* Artifacts: `tracks/ksvd/results/oof_difficulty_audit/` (stage markers, `fold_assignments.csv`, `inner_splits.csv`, `oof_per_molecule.csv`, Tables A–E, `difficulty_predictions.npz`, `decision_record.json`, figures, per-fold npz/state/meta in `cache/folds/`).
* Tracked note: this file. Claims/decisions: `records/claims/claim-oof-difficulty-confirmed-20260910.yaml`, `records/decisions/decision-oof-difficulty-heteroscedasticity-go-20260910.yaml`.
* Freeze honored: the exact frozen config `configs/luyin16/zinc_compact_v4_topology_hinge.yaml` is read and only the hybrid-embedding full-table counts are clamped to the per-fold fitted vocabulary (identical practice to the v2 OOF audit; see §2). Loss (L1), optimizer, LR, epochs, topology hinge 25-D features, standardizer fit, head, patch-path code: untouched. No quantile/Gaussian-NLL/weighting/v5 code anywhere in the module.
* Test/valid: never loaded. Record/target sources are the train-only caches:
  * `results/post_v4_residual_audit/cache/v4_records_train.pkl.gz` (10 000 `GraphRecord`s with hinge topology, rebuilt under the frozen mirror),
  * train labels = `results/zinc_long_cycle_audit/label_effective_cycle.csv` rows with `split == train` (bit-verified: `oof_per_molecule.target == y_stored` for all 10 000; label source is the frozen benchmark label generator, y = normalized logP, pool mean 0.015, range −42.0 … +3.8),
  * sidecars / structural-context caches of the information-gap audit (read-only).

### 1.1 OOF scheme (leakage-free)

* `fold_assignments.csv`: deterministic 5-fold split of official train (fold_seed=0; every train molecule in exactly one held-out fold; fold sizes 2000/2000/2000/2000/2000). Inner train/valid splits per outer fold are saved in `inner_splits.csv` with SHA-256 fingerprints (`inner_split_seed = 12345 + fold_seed*10000 + fold*7`; 7200 inner-train / 800 inner-valid / 2000 held-out).
* Per outer fold: model fit on the 7200 inner-train molecules only; inner-valid (800) used solely for best-epoch checkpoint selection / early stop; predictions on the 2000 held-out molecules are the molecule's **OOF prediction**. Rarity features are computed against a counter of the molecule's own outer-fold inner-train only — a held-out molecule's fold never contributes to any counter or to the topology standardizer of its own model.
* Two model seeds per fold: seed 0 (first) then seed 1 (second, for disagreement/ensemble only). Every train molecule has exactly one OOF prediction per model seed.

### 1.2 Determinism and gates

* CPU determinism policy (`notes/reproducibility_cpu_determinism.md`): runs were executed 2-way (bit-identical regime; 4-way diverges at epoch 3). `torch_threads=4` as in the frozen runs.
* Determinism gate: fold 0 / seed 0 rerun reproduced the first run **bit-exactly** (training trace identical, selected epoch 58, npz SHA and state SHA equal, OOF predictions array-equal).
* Checkpoint-reproduction gate (states stage): each fold's saved checkpoint was reloaded under the frozen mirror, the held-out fold re-encoded deterministically, and predictions compared to the fold npz: **max diff 0.0 on all 5 folds** before any internal-state row was emitted.
* Vocabulary/parameter gates per fold: re-derived `_phase_data` audit must match the fold-run meta (typed vocab, topology width); parameter counts must match (they do, 96 033–96 557).

---

## 2. Fold inventory

| fold | vocab typed/parent (incl OOV) | params | seed 0: epoch / inner-valid MAE / OOF MAE | seed 1: epoch / inner-valid MAE / OOF MAE |
|---|---|---|---|---|
| 0 | 5923/29 | 96 141 | 58 / 0.2109 / **0.1780** | 57 / 0.2084 / **0.1782** |
| 1 | 6023/31 | 96 557 | 54 / 0.1576 / **0.1930** | 47 / 0.1550 / **0.1923** |
| 2 | 5896/29 | 96 033 | 59 / 0.1937 / **0.1781** | 54 / 0.1961 / **0.1716** |
| 3 | 5981/29 | 96 373 | 59 / 0.1698 / **0.1699** | 52 / 0.1656 / **0.1693** |
| 4 | 5896/31 | 96 049 | 56 / 0.2350 / **0.1600** | 52 / 0.2290 / **0.1598** |

* Pooled per-seed OOF MAE: seed 0 0.1758, seed 1 0.1743 (mean-of-seeds per molecule 0.1750).
* Per-fold difficulty is **very stable across model seeds** (largest per-fold ΔMAE between seeds: fold 2, 0.0064; four folds within 0.0007) → fold populations have systematic difficulty that the training noise barely moves.
* Inner-valid MAE varies more across folds (0.155–0.235) than OOF MAE (0.160–0.193): fold membership (which molecules land in the 800) matters more than seed for the *selection* signal; OOF difficulty itself is seed-stable.

### Embedding-clamp note (auditable deviation, parameterization only)

Frozen config fixes hybrid full-table boundaries at 768 typed / 32 parent rows with `full_count ≤ vocabulary_size` enforced by the frozen `_HybridEmbedding`. Fold vocabularies fitted on 7200 molecules are 5896–6023 typed (always > 768, no typed clamp) and 29–31 parent incl. OOV, so the parent full-table count is clamped 32 → 29/31 (all parent rows in the full table; no rare module appears). This is the exact row-count dependence the frozen runner already has across selection (10k) vs refit (11k) vocabularies and is identical in kind to the v2 OOF audit; the clamp text is recorded per fold in each `cache/folds/fold{f}_seed{s}_meta.json`. Parameter totals 96 033–96 557 vs 99 613 frozen (Δ ≤ 3.6%: only embedding-table rows, head/encoder identical).

---

## 3. Results

### 3.1 OOF error distribution (per-molecule, mean over seeds)

| stat | value |
|---|---|
| mean (MAE) / median | 0.1750 / 0.1139 |
| p90 / p99 / max | 0.310 / 0.903 / 38.0 |
| share of molecules with error > 0.5 | 3.3 % |
| top-1% (100 molecules) share of total MAE mass | 17.0 % |

The distribution is heavy-tailed: the worst molecules (e.g. `train:2210`, target −42.0, per-seed errors 38.11/37.89) are the extreme normalized-logP outliers; their errors agree across seeds almost exactly (systematic, not seed luck).

### 3.2 Table A — per-fold OOF MAE (see §2; full table incl. signed residuals: `table_A_fold_results.csv`)

### 3.3 Table B — per-molecule feature correlations with difficulty (`table_B_difficulty_correlations.csv`)

| family | feature | Spearman (overall) | fold consistency |
|---|---|---|---|
| disagreement | disagreement (seed-spread of OOF preds) | **0.393** | 5/5 same sign |
| rarity | mean_inverse_of_1p_frequency | **0.339** | 5/5 |
| rarity | rare_le10_ratio | 0.310 | 5/5 |
| rarity | rare_le5_ratio | 0.309 | 5/5 |
| rarity | min_train_frequency | −0.307 | 5/5 |
| rarity | oov_patch_ratio | 0.277 | 5/5 |
| states (seed 0) | state_patch_norm_std | 0.182 | 5/5 |
| structure | diameter / max-shortest-path | −0.175 | 5/5 |
| topology | n_cycle_len_7 / n_cycle_len_8 | 0.141 / 0.129 | 5/5 |
| topology | hinge 6,7,10 (long-cycle bands) | 0.099–0.119 | 5/5 |

* The **rarity family is the leading visible-at-fit-time axis** on official train, 5/5 same sign on every member (cf. validation audit: rare≤5 abs Spearman 0.28–0.32/seed 4/4, ens 0.41). Fold spread is small (rare≤5 fold Spearman 0.293–0.326).
* Internal-state statistics are only weakly informative (best: spread of patch-state norms 0.18); head-penult / global-encoder norms are unstable (≤3/5).
* Structure (diameter −0.17) and long-cycle topology (+0.10–0.14) contribute small orthogonal components only; the long-cycle B/C mechanism is already inside v4-hinge (context).
* Signed side is weak everywhere (context; see §5 Q13).

### 3.4 Table C — cross-fitted difficulty predictor (`table_C_difficulty_predictor.json`; nested outer-fold CV StandardScaler+Ridge on log(error+1e-3); every molecule's prediction comes from a model fit on its own OOF folds only)

| variant | n features | Spearman | hardest/easiest quintile MAE ratio | R²(log) | fold consistency |
|---|---|---|---|---|---|
| **all** (visible + disagreement) | 70 | **0.490** | 4.14 | 0.264 | 5/5 |
| disagreement_only | 1 | 0.392 | 2.85 | 0.172 | 5/5 |
| **model_visible_single** (rarity+structure+topology+states) | 69 | **0.374** | 3.67 | 0.168 | 5/5 |
| rarity_only | 12 | 0.350 | 2.24 | 0.132 | 5/5 |
| structure_only | 44 | 0.304 | 3.12 | 0.128 | 5/5 |
| states_only | 13 | 0.174 | 1.99 | 0.046 | 5/5 |

Per-fold Spearman of the visible predictor: 0.306–0.355 (mean 0.334). Disagreement adds orthogonal signal over visible features (+0.116 Spearman, +0.096 R²(log) → all-variant 0.490).

### 3.5 Table D — predicted-difficulty quintiles (`table_D_difficulty_quintiles.csv`; model_visible_single)

| quintile | n | actual MAE | median abs err |
|---|---|---|---|
| Q1 (easiest) | 2000 | 0.0951 | 0.0763 |
| Q2 | 2000 | 0.1235 | 0.0975 |
| Q3 | 2000 | 0.1463 | 0.1048 |
| Q4 | 2000 | 0.1632 | 0.1249 |
| Q5 (hardest) | 2000 | 0.3510 | 0.1863 |

Monotone ladder; hardest quintile MAE 3.67× the easiest.

### 3.6 Table E — epistemic signal / ensemble gain (`table_E_epistemic_signal.json`, 2 seeds)

| quantity | value |
|---|---|
| Spearman(disagreement, mean-abs-error), overall | **0.3933** |
| per-fold (0–4) | 0.397 / 0.420 / 0.393 / 0.385 / 0.359 (mean 0.391, std 0.022) |
| individual MAE seed 0 / seed 1 / mean | 0.1758 / 0.1743 / 0.1750 |
| 2-seed ensemble MAE (mean prediction) | 0.1604 |
| ensemble gain vs mean-individual | −0.0146 (−8.3%) |
| per-molecule cross-seed Spearman(err0, err1) | 0.384 |

Ensemble gain grows monotonically with observed error quintile (Q1 +0.001 … Q5 +0.045; `ensemble_gain_by_error_bin.csv`): error-reduction by ensembling is concentrated exactly on the hardest molecules.

### 3.7 Rarity bins (`rarity_bin_table.csv`, fixed half-open bins on rare≤5 ratio)

| bin | n | OOF MAE | median abs | signed residual mean | mean rare≤5 |
|---|---|---|---|---|---|
| 0 | 4316 | 0.133 | 0.092 | −0.019 | 0.000 |
| (0.01, 0.05] | 2000 | 0.148 | 0.110 | −0.031 | 0.041 |
| (0.05, 0.1] | 1786 | 0.180 | 0.131 | −0.030 | 0.076 |
| (0.1, 0.2] | 1474 | 0.258 | 0.162 | −0.087 | 0.143 |
| > 0.2 | 424 | 0.417 | 0.253 | −0.171 | 0.270 |

Monotone 3.1× MAE ladder from "no rare token" to "> 20% rare tokens". ((0, 0.01] is empty — a molecule's rare-token fraction is an integer-count ratio with minimum positive value ≈ 1/n_patches; boundary masses at exact 0.05/0.10/0.20 are placed in the bin whose printed label contains them.)

### 3.8 y-level control (difficulty ≠ level artifact)

* Spearman(mean-abs-error, y) = −0.31: low normalized-logP molecules are harder (y is the stored normalized logP, mean 0.015, min −42).
* The rarity gradient is not merely y: rarity bins carry decreasing mean y (−0.33 … −1.5 as rarity grows) but MAE rises within every rarity bin, and in the 2×2 median grid (rows = y half, columns = predicted difficulty half) MAE is separated in **both** halves — low-y half 0.136 (easy) vs 0.280 (hard); high-y half 0.102 vs 0.151. Within-y-quintile pooled Spearman(predicted difficulty, error) = 0.284 (per-quintile 0.229–0.344).

### 3.9 Figures (`figures/`, matplotlib)

| file | content |
|---|---|
| `fig1_error_vs_rare_ratio.png` | per-molecule error vs rare≤5 ratio (binned means + q-ribbon) |
| `fig2_error_vs_disagreement.png` | per-molecule error vs seed disagreement (binned means) |
| `fig3_pred_vs_actual_model_visible_single.png` | predicted difficulty vs observed mean-abs-error (binned + identity) |
| `fig4_quintile_mae_model_visible_single.png` | actual MAE by predicted-difficulty quintile (bar + median) |
| `fig5_rare_vs_disagreement.png` | disagreement vs rarity (partial-overlap view) |
| `fig6_ensemble_gain.png` | ensemble gain by observed-error quintile |

---

## 4. Questions and answers (Q1–Q14)

**Q1. What is the pooled per-molecule OOF MAE of the frozen v4-hinge protocol on official train?**
A1. 0.1758 (seed 0) / 0.1743 (seed 1); per-molecule mean over seeds 0.1750 (n = 10 000, one OOF prediction per molecule per seed). Per-fold: 0.1600–0.1930 (seed 0), 0.1598–0.1923 (seed 1). For reference (recorded, not recomputed here): frozen baseline official-valid MAE 0.1701 (seed 0) and official-test MAE 0.1369±0.0055; the OOF protocol fits 7 200 molecules per fold and evaluates the same pool it samples from, which is the intended difficulty measurement, not a model-quality comparison.

**Q2. Are fold-level difficulties stable across model seeds?**
A2. Yes — per-fold MAE between seeds differs by ≤ 0.0064 (fold 2) and ≤ 0.0007 on the other four folds. Fold populations have intrinsic difficulty ordering 1 > 0 ≈ 2 > 3 > 4 that survives retraining noise.

**Q3. Is per-molecule difficulty consistent across seeds (systematic vs seed-luck)?**
A3. Moderately: cross-seed Spearman of per-molecule absolute error = 0.384; the extreme tail is essentially seed-independent (worst-molecule errors agree to ~0.2%). Two seeds put a floor under molecule-level consistency estimates; the strong disagreement–error Spearman (0.393) shows the seed-spread itself is the best single difficulty feature.

**Q4. Is remaining train error concentrated where tokens are rare?**
A4. Yes, monotonically and 5/5-folds-stable: MAE 0.133 (no rare≤5 token, n=4316) → 0.417 (>20% rare≤5 tokens, n=424); Spearman(rare≤5) = 0.309, (OOV ratio) = 0.277, (mean inverse 1+freq) = 0.339, all 5/5 same sign with small fold spread (0.293–0.326). This is the train-side confirmation of the valid-side rarity finding (valid ens rare≤5 0.41).

**Q5. Does rarity remain informative after controlling for target level and structure?**
A5. Yes. The 2×2 y/prediction grid and rarity-bin mean-y drift show the rarity ladder is not a y-level echo; within-y-quintile Spearman of the full visible difficulty predictor is 0.284. Rarity is the leading *single* visible feature; structure-only and states-only variants reach 0.304 and 0.174 respectively.

**Q6. How hard is the hardest tail, and is it systematic?**
A6. p90 0.310 / p99 0.903 / max 38.0; 3.3% of molecules exceed 0.5; top-1% carries 17% of total MAE mass. Worst molecules are extreme-negative normalized-logP outliers with cross-seed agreeing errors (e.g. 38.11/37.89) — systematic conditional variance at the extremes.

**Q7. Do internal (hidden-state) statistics of the fold model predict difficulty?**
A7. Weakly: best is the spread of patch-state norms (0.182, 5/5); patch-norm means/dim-variance 0.13–0.16; head-penult and global-encoder norms are unstable across folds (≤3/5) and uninformative. States add little beyond rarity (states-only 0.174 vs rarity-only 0.350).

**Q8. What does a cross-fitted difficulty predictor achieve (the pre-registered GO metric)?**
A8. Nested-OOF StandardScaler+Ridge on log(error): **model-visible** (rarity+structure+topology+states) Spearman **0.374** with **5/5** folds same sign (per-fold 0.306–0.355) — above the difficulty GO gate (≥ 0.25 with ≥ 4/5). Including seed disagreement: Spearman 0.490, R²(log) 0.264, 5/5.

**Q9. Does predicted difficulty sort actual MAE monotonically?**
A9. Yes (Table D): actual MAE by predicted quintile 0.095 → 0.124 → 0.146 → 0.163 → 0.351 (ratio 3.67).

**Q10. Is the remaining error epistemic (model-uncertainty-like)?**
A10. Yes at the level of a 2-seed diagnostic: Spearman(disagreement, error) = 0.393 overall, per fold 0.359–0.420, 5/5 same sign — above the epistemic GO gate (≥ 0.30 with ≥ 4/5). Mean per-seed MAE 0.1750 vs 2-seed ensemble 0.1604 → gain −0.0146 (−8.3%), concentrated in the hardest error quintile (+0.045 there).

**Q11. Does ensembling gain grow with difficulty?**
A11. Yes, monotonically across observed-error quintiles (Q1 +0.001 → Q5 +0.045; `ensemble_gain_by_error_bin.csv`) — the seed-spread is exactly where the errors are.

**Q12. Is "difficulty" just a label/noise artifact of extreme target values?**
A12. No. Difficulty is not a signed-bias artifact (mean signed residual −0.040, |signed| ≪ |abs| for every leading feature), and it survives y-level controls (Q5). y-level itself correlates with error (Spearman −0.31), which is itself a legitimate heteroscedasticity axis; the documented axes are conditional-variance axes, matching the valid-side audit conclusion.

**Q13. Is there any signed (directional) structure in the OOF residuals on train?**
A13. Only weak and unused for verdicts (as mandated). Pooled mean signed residual (target − mean prediction) −0.040 (median −0.008, std 0.62); per fold −0.065…−0.013. Rarity bins drift slightly negative with rarity (−0.019 → −0.171: rare-token molecules over-predicted on average), but leading signed correlations are small (rare≤5 −0.08 vs abs +0.31) and this audit never gates on direction.

**Q14. Verdicts.**
A14. **Difficulty: GO** — cross-fitted model-visible Spearman 0.374 ≥ 0.25 with 5/5 folds same sign (already GO on seed 0 alone: 0.300, 5/5). **Epistemic: GO** — disagreement–error Spearman 0.393 ≥ 0.30 with 5/5 folds same sign. The official-train OOF residual of the frozen v4-hinge protocol is confirmed as predictive difficulty / heteroscedastic conditional variance, concentrated on rare-token molecules, low-y extremes, and seed-disagreement — with no sign of a second-layer signed structural signal. This closes the diagnostic loop opened by the validation-only Post-v4 Residual Audit ("NO CLEAR SECONDARY SIGNAL") under the pre-registered confirmation protocol.

---

## 5. Decision, claims, and revisit conditions

* **Decision record:** `results/oof_difficulty_audit/decision_record.json` + `records/decisions/decision-oof-difficulty-heteroscedasticity-go-20260910.yaml`.
* **Claim record:** `records/claims/claim-oof-difficulty-confirmed-20260910.yaml` (status: supported).
* Revisit conditions recorded in the decision YAML (calibration/objective direction re-opens only under the frozen protocol's own rules, and any future model change must be measured against this OOF-train residual set).

## 6. Limitations

* 2 model seeds (per the pre-registered plan: seed 0 + 1 disagreement seed). Molecule-level cross-seed consistency (0.384) is therefore a 2-sample estimate.
* Internal-state statistics captured from seed-0 fold models only (single deterministic seed as designed).
* y is the stored normalized-logP label of the frozen generator; "difficulty" here means absolute error on that normalized scale.
* Rarity counters are per-molecule outer-fold inner-train (7 200); a molecule's own fold is excluded (no leakage) but counts differ slightly from a full-train counter (population-context comparison of train vs valid rarity used a full-train counter: rare≤5 means 0.039 (train) vs 0.045 (valid) — similar mix, so the OOF-vs-valid MAE gap is not a rarity-mix artifact).
* OOF MAE (0.175, fit 7.2k) vs frozen valid MAE (0.170, full-protocol fit): reflects fit-size and population, and is not itself a claim input.
* Signed-residual numbers appear only as context; no directional claim is made from them.
