# Frozen function-basis accessibility audit — output manifest

**Representation-frozen, function-basis diagnostic** on the frozen compact-v4-hinge OOF
representation (official TRAIN molecules only; official valid/test never loaded). It asks a
single question:

> On the *same* frozen graph representation `R` (302D) and with the *same* L1/MAE
> objective, does changing only the final function parameterisation — from a generic ReLU
> MLP to an explicit target-independent piecewise-linear (quantile-hinge) basis — make
> residual regression more stable and more sample-efficient?

No backbone is trained; patch representation, tokenizer, pair encoder, centre update,
pooling, topology branch and loss are all unchanged. No new structural feature is added.

Full scientific note: `tracks/ksvd/notes/function_basis_accessibility_audit.md`.
Module: `tracks/ksvd/experiments/luyin16/zinc_function_basis_accessibility.py`
(stages `inventory → spec → cache → integrity → splits → preprocessing → convergence →
stage1 → stage1b → stage2 → mechanism → fixed_knot → figures → decision → all`).
Tests: `tracks/ksvd/tests/test_function_basis_accessibility.py` (13 tests, all pass).

CSV/JSON/NPZ/PNG artifacts below are generated outputs (git-ignored by policy); this
README, the note, and the claim/decision YAML records are the tracked scientific record.

## Representation and readers

The frozen export is the corrected `frozen_state_export_v4_centre_incidence` family
(`R` = true pre-head 302D input, `yhat_0` = true frozen forward prediction). Every reader
predicts `yhat = yhat_0 + delta(z)` where `z` is the fit-only coordinate standardisation of
`R` (`z_j = (R_j - mean_j)/max(std_j, 1e-6)`, `z_j = 0` when `std_j < 1e-6`; statistics
from the 1200 adapter-fit molecules only).

| reader | function | params |
|---|---|---|
| **B0** | frozen baseline prediction, no training | 0 |
| **B1** | generic ReLU MLP `302 → 4 → 2 → 1` | **1225** |
| **B2** | historical-strength generic ReLU MLP `302 → 13 → 13 → 1` | 4135 |
| **E** | additive linear reader on the explicit hinge basis `[z_j, ReLU(z_j-t25), ReLU(z_j-t50), ReLU(z_j-t75)]` (1208D) | **1209** |
| **B-linear** | descriptive `302 → 1`, not in the gate | 303 |

B1/E parameter mismatch is **+1.32%** (≤ pre-registered 3%). B2 is deliberately *not*
parameter-matched: it is the stronger generic reference (`~3.4×` E).

## Files

| File | Contents |
|---|---|
| `checkpoint_inventory.json` | frozen OOF checkpoint inventory (fold × seed, state `.pt` SHA-256, params, oof_mae); seeds [0,1] complete |
| `representation_integrity.json` | integrity/reconstruction gates G1–G6 on 5/5 folds; `all_passed: true` |
| `fold_split_manifest.json` | per-fold target-independent molecule-ID-hash split (1200 fit / 400 selection / 400 evaluation), cross-checked identical to the frozen-readout / pair / centre manifest; common-input bulk threshold |
| `preprocessing_stats.json` | per-fold fit-only standardisation + knot hashes and degenerate-coordinate counts |
| `basis_spec.json` | representation, preprocessing, basis, reader and training specification + parameter accounting |
| `convergence_trace.csv` / `.json` | fold0/seed0 selection MAE at 100/200/400/800 epochs for B1/B2/E and the fixed-horizon rule |
| `stage1_fold_results.csv` | per-fold `B0 / B1 / B2 / B-linear / E / E-no-hinge` MAE, `Δsmall`, `Δstrong`, bulk safety, params, selection MAE, basis sensitivity |
| `stage1_bootstrap.json` | stratified molecule-level paired bootstrap for `Δsmall`, `Δstrong`, and the hinge-vs-linear-only ablation + aggregates |
| `final_decision.json` | final verdict + criteria + case + summary + scope caveat |
| `state_exports/function_basis_R_cache_v1_fold{0-4}_seed0.npz` | compact per-molecule cache `R / yhat_0 / target / subset_index / fingerprint` |
| `figures/figure1_stage1.png` | per-fold `Δsmall`/`Δstrong` and reader MAE |

`stage1b_init_results.csv`, `stage2_fold_results.csv`, `pooled_backbone_results.csv`,
`final_bootstrap.json`, `coefficient_summary.csv`, `hinge_ablation_results.csv` and
`fixed_knot_control.csv` are **not** produced: Stage 1 was a pre-registered CLEAR NO-GO, so
the second init, the second backbone seed, the mechanism analysis and the fixed-knot
control were **not** spent.

## Key numbers (frozen OOF backbone seed 0, 5 folds, 1 init, horizon 400)

Adapter-evaluation MAE mean over folds:

| B0 | B1 (1225) | B2 (4135) | B-linear (303) | E (1209) | E no-hinge |
|---:|---:|---:|---:|---:|---:|
| 0.193160 | **0.187815** | 0.190508 | 0.203257 | 0.203204 | 0.250995 |

Per-fold basis gains:

| fold | B0 | B1 | B2 | E | `Δsmall` = B1−E | `Δstrong` = B2−E |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.224365 | 0.216585 | 0.217165 | 0.234422 | −0.017836 | −0.017257 |
| 1 | 0.159687 | 0.145999 | 0.151477 | 0.157910 | −0.011911 | −0.006433 |
| 2 | 0.263058 | 0.262321 | 0.267341 | 0.285166 | −0.022845 | −0.017825 |
| 3 | 0.153514 | 0.149360 | 0.153222 | 0.164665 | −0.015305 | −0.011444 |
| 4 | 0.165176 | 0.164809 | 0.163337 | 0.173859 | −0.009050 | −0.010523 |
| **mean** | **0.193160** | **0.187815** | **0.190508** | **0.203204** | **−0.015389** | **−0.012696** |

- **`mean Δsmall = MAE(B1) − MAE(E) = −0.015389`** (median `−0.015305`), **0/5** folds
  positive; stratified paired bootstrap 95% CI `[−0.01979, −0.01093]`, `P(>0) = 0`.
- **`mean Δstrong = MAE(B2) − MAE(E) = −0.012696`** (median `−0.011444`), **0/5** folds
  positive; i.e. `mean(MAE_E − MAE_B2) = +0.012696`.
- `mean(MAE_E − MAE_B0) = +0.010044`: the additive hinge reader is **worse than the frozen
  baseline** it is supposed to correct.
- B1 is better than B0 on all 5 folds (`mean Δ = +0.005345`); B2 is better on all 5
  (`mean Δ = +0.002651`).
- Bulk safety (common-input rare-patch subgroup, `rare_le5 < 80th percentile of fit`):
  `mean(MAE_E − MAE_B1) = +0.018408`, max `+0.021165` — far above the pre-registered
  `+0.002` gate (FAIL).
- The hinge terms are *not* dead: within E, removing them at inference (`β_hinge = 0`)
  raises evaluation MAE by `mean +0.047791` (95% CI `[+0.03975, +0.05587]`, 5/5 folds).
  So the explicit basis does express L1-relevant shape — it is simply not competitive with
  the small generic MLP at the same budget.
- Decision: **CLEAR NO-GO / Case C** — `mean Δsmall ≤ +0.0005` and `0/5` folds
  E better than B1 (`records/decisions/decision-function-basis-accessibility-nogo-20260912.yaml`).
  No second backbone, no more knots, no learned knots, no B-splines / polynomial / RBF /
  Fourier / KSVD response dictionary, no XGBoost comparison.

## Gate 0 detail (5/5 folds, seed 0)

`G1` stored `yhat_0` == frozen OOF fold prediction (max diff `0.0`); `G2` stored target ==
official-train label `y_stored` (max diff `0.0`); `G3` `subset_index` == outer holdout
(pass); `G4` offline reconstructed `R` == exported `R` (max `1.1e-5`…`9.2e-5`);
`G5` re-feeding stored `R` through the frozen head == stored `yhat_0` (max `0.0`);
`G6` re-feeding reconstructed `R` through the frozen head == stored `yhat_0`
(max `9.5e-7`…`1.9e-6`). Degenerate coordinates (fit-constant) range 18–94 of 302 per
fold and are set to `z = 0`.
