# LCRC Lipophilicity v2 — mechanism generalization and SSL

**Round:** `lcrc_lipo_mechanism_generalization_ssl_v2`  
**Date:** 2026-10-31  
**Data:** Fit2352 only; internal 3-fold scaffold CV; Monitor504 used once after a label-blind prediction freeze. Reserved and official valid/test labels remained sealed.  
**Primary seeds:** 0 and 1 (AMENDMENT-001). Fold-0 seed-2 R0 runs were completed before that amendment and are reported only as a sensitivity check.

## 1. Verdicts

| Hypothesis | Verdict | Primary evidence |
|---|---|---|
| **H1** A beats capacity-matched E | **PROMISING** | CV delta E−A = **+0.04510** RMSE (all 3 folds favor A; Fold-0 CI [+0.0364, +0.1387]); Monitor delta = **+0.23219** (CI [+0.1234, +0.3333]) |
| **H2** fixed supervised recipes improve RMSE | **PROMISING** | A: R3 vs R0 = **+0.07578** mean CV (folds +0.06288/+0.07655/+0.08791, all 8 seed-fold pairs positive). C: +0.02836 on Fold 0 only (no C-R0 on folds 1/2) |
| **H3** Fit-only SSL improves A | **NOT_PROMISING** | Pretext gate passed strongly (heldout CE 0.0308/0.0273 vs prior 1.5377), but SSL CV delta = **−0.03147** (folds −0.0360/+0.0111/−0.0695); Monitor delta = **−0.04898** |
| **H4** LCRC-A vs strong CPU chemistry baselines | **DESCRIPTIVE TIE** | A seed-averaged Monitor RMSE **0.76865** vs XGB-P3 **0.77878** (A nominal −0.01013, CI [−0.1116, +0.1179]); per-seed A mean 0.79162 is 0.01284 *worse* than CPU |

The Monitor is an external check on frozen predictions, **not** an independent blind test; all selection and verdict rules use internal CV.

## 2. Protocol, freeze and amendments

- Protocol ID: `lcrc-lipo-mechanism-generalization-ssl-v2`.
- Final protocol freeze: `analysis/protocol_freeze_v2.json`, SHA-256 `76524f0e9dfc…` (full hash in `summary_v2.json`).
- GPU authorization: `analysis/gpu_authorization_v2.json`, SHA-256 `965080b2fe15…`; it is pinned to the earlier two-seed freeze `d78d71cfac2d…` and was consumed before the post-run analysis fixes. No GPU job was launched after the authorization was superseded by the final analysis freeze; the ledger still enforces the 3.5 h cap.
- Prediction manifest: `analysis/predictions/MANIFEST.json`, SHA-256 `1bc705295b4e…`, 9 predictions, `label_blind=true`.
- Hard cap: 3.5 A100 GPU-hours. Charged: **2.4647 h** (61 jobs); remaining **1.0353 h**. v1's 0.5539 h remains separate historical context.
- v1 immutability: `assert_v1_unchanged` = **51/51 checks ok** (v1 freeze `7fffba42…`, manifest `ba8ad0be…`, formal checkpoints included).

### Amendments (all disclosed)

1. **AMENDMENT-001** — researcher instructed: use `res`, run two seeds instead of three. Primary seeds became 0/1. Three Fold-0 seed-2 R0 runs had already completed; they are retained as an exploratory sensitivity check and excluded from primary verdicts. Registry: 56 runs (46 supervised + 10 SSL/pretrain); five full-retrain seed-2 jobs and the Fold-0 SSL seed-2 jobs were dropped.
2. **AMENDMENT-002** — post-run analysis/inference key-resolution fixes: folds 1/2 use the registered `SEL` alias; prediction and diagnostics read the string recipe through `C.RECIPES`; the gated Monitor artifact is opened exactly once; the single-pass scan tuple is unpacked. These are implementation fixes, not model or decision changes.
3. **AMENDMENT-003** — the first Monitor execution was **invalid**: `predict_nn` froze standardized outputs without denormalizing, producing NN RMSE ≈2.3 and R²≈−3. Its complete artifacts are quarantined at `analysis/audit/monitor_invalid_normalization/` and are not used. After fixing the denormalization and adding a fail-closed mean check, corrected predictions were frozen (`1bc705…`) and a single valid Monitor evaluation was performed. The invalid execution is not counted as a scientific evaluation.

## 3. Data and folds

Fit2352: 2,352 molecules, 729 Murcko-scaffold groups. Groups are ordered by size descending, ties by `sha256("lcrc-lipo-v2|" + group_key)` ascending, and greedily assigned to the smallest fold. Normalization is recomputed per training fold; the full-Fit target normalization is mean **2.157857**, scale **1.224198**.

**Table 1 — folds**

| Fold | Molecules | Scaffold groups | Row SHA-256 (text) |
|---|---:|---:|---|
| 0 | 784 | 241 | `7b2e86d8a381…` |
| 1 | 784 | 244 | `4d7a715dfe44…` |
| 2 | 784 | 244 | `f458ad003701…` |

Fit row SHA-256 (text) `c474e38cdecf…`; int64 `4ecba42cc028…`. Full hashes are in `analysis/folds_v2.json`, `analysis/table1_folds.csv` and `analysis/summary_v2.json`.

## 4. Capacity-matched E (H1 manipulation check)

E is not a literal 529-wide LocalMLP. The original S2 slot is zeroed at runtime (all 38,400 columns dead, as in the old B control), and a parameter-free fixed residual injects a `Linear(320,135) → SiLU → Linear(135,200)` replacement (70,535 parameters) into the 200-wide local representation (200→192; the last 8 are folded into the first 8). This avoids reviving dead columns and keeps E free of message passing/attention.

| Quantity | A | E |
|---|---:|---:|
| Nominal parameters | 247,106 | 285,449 |
| Effective (live) parameters | 247,106 | **247,049** |
| Delta vs A | — | **−57 (−0.023%)** |
| Decomposition | — | B effective 176,514 + replacement 70,535 |

The 27 shared state entries are bitwise identical to A at the same seed; verified S2 tensor swaps and `RestrictedBatch` leave E bit-identical; replacement gradients are nonzero, dead-column gradients are zero. E's Fold-0 train RMSE is 0.2246 vs A's 0.0923, and its dev RMSE is 0.86–0.90 vs A's 0.79–0.81: capacity matching did not reproduce A's behavior, and E overfits more.

## 5. Recipes (H2)

Fold-0 recipe screen, seeds 0/1. Selection rule: lowest mean dev RMSE; keep R0 if improvement < 0.02; ties R0 > R1 > R2 > R3.

**Table 3 — recipe screen (Fold 0, mean dev RMSE)**

| Arm | R0 | R1 | R2 | R3 | R3−R0 gain | Selected |
|---|---:|---:|---:|---:|---:|---|
| A | 0.79759 | 0.73521 | 0.73544 | **0.73471** | **+0.06288** | R3 |
| C | 0.78896 | 0.77187 | 0.77452 | **0.76060** | **+0.02836** | R3 |

R3 = peak LR 3e-4, weight decay 1e-3, dropout 0.10 at the final atom-level local representation. A's R3 gain replicates on every fold: +0.06288 (F0), +0.07655 (F1), +0.08791 (F2), mean **+0.07578**, 8/8 seed-fold pairs positive. The C result is Fold-0 only because C-R0 was not registered on folds 1/2; C's seed-1 Fold-0 gain is only +0.00810, so the C arm is weaker evidence. H2 is promising primarily because of A.

## 6. Cross-fold CV (H1–H3)

**Table 2 — internal scaffold CV dev RMSE (mean of seeds 0/1; lower is better)**

| Fold | A-R0 | A-R3 | E-R0 | C-R0 | C-R3 |
|---|---:|---:|---:|---:|---:|
| 0 | 0.79759 | 0.73471 | 0.88093 | 0.78896 | 0.76060 |
| 1 | 0.81806 | 0.74151 | 0.84984 | — | 0.71762 |
| 2 | 0.81420 | 0.72629 | 0.83437 | — | 0.71181 |

### H1 — A vs capacity-matched E

A-R0 beats E-R0 on every fold: deltas (E−A) **+0.08334 / +0.03178 / +0.02018**, mean **+0.04510**. Fold-0 paired molecule bootstrap 95% CI [+0.03641, +0.13872]; folds 1/2 CIs include zero, so the effect is consistent in direction but not individually significant on the smaller folds. The 3-seed Fold-0 sensitivity gives A 0.80110 vs E 0.88799, delta **+0.08689** — same conclusion, not used for the primary verdict.

Monitor: A 0.76865 vs E 1.00084, delta **+0.23219** (CI [+0.12336, +0.33331]). This Monitor contrast uses the final **A-R3** retrain (the selected recipe) against **E-R0**, so it is not the same contrast as the CV A-R0 vs E-R0 capacity test; the CV result is the pre-registered H1 test and the Monitor is directional only. Even so, E's external degradation is much larger than its CV gap; E's much lower train RMSE and high generalization gap point to overfitting. H1 is supported by internal CV and by the external direction, with the recipe difference disclosed.

### H2 — recipes

Covered in §5. H2 is PROMISING for A; the C arm is a Fold-0-only observation. The Monitor C result (0.83696 seed-averaged) is worse than A (0.76865); this is a C-vs-A comparator, not an R3-vs-R0 test.

### H3 — Fit-only SSL

The masked root atom-type pretext passed the pre-registered gate by a wide margin:

| Seed | Heldout CE (fixed mask) | Marginal-prior CE | Delta (nat) |
|---:|---:|---:|---:|
| 0 | 0.03082 | 1.53772 | 1.50690 |
| 1 | 0.02725 | 1.53772 | 1.51046 |

This means the pretext task was learned, not that the representation transfers. Downstream A-R3 with SSL was worse on Fold 0 (−0.03598) and Fold 2 (−0.06952); Fold 1 was nominally better (+0.01109) but seed 1 was slightly negative. Mean CV **−0.03147**; Monitor **−0.04898** (SSL 0.81763 vs scratch 0.76865). H3 is **NOT_PROMISING**. The pretext may be too easy (atom-type reconstruction from local environment) and the 120-epoch encoder perturbation does not help the supervised readout; no iterative message passing was used.

## 7. H4 — CPU chemistry baselines

Workstream B ran 30 candidates (6 Ridge + 24 XGB) on P1 Morgan/ECFP4 (2,048 bits), P2 RDKit descriptors (fixed 217 names) and P3 = P1+P2, all fold-train-only preprocessing. Selection used mean CV RMSE with a <0.01 complexity tie-break. Chosen: **`xgb_P3_n800_d5_w1`** (P3, 800 trees, depth 5, min_child_weight 1; fixed lr 0.03, subsample/colsample 0.8, λ=5, hist, rs=0, n_jobs=4).

| Model | CV mean RMSE | CV folds | Monitor RMSE |
|---|---:|---|---:|
| XGB-P3 (chosen) | **0.72357** | 0.77626 / 0.67746 / 0.71699 | **0.77878** |
| A-R3 (CV mean) | 0.73417 (3 folds) | 0.73471 / 0.74151 / 0.72629 | **0.76865** |
| A-R3 per-seed | — | — | 0.77318 / 0.81005 (mean 0.79162) |

**Table 4 — Monitor504 (504 molecules; per-seed means and seed-averaged RMSE)**

| Arm | Recipe | Seeds | Per-seed RMSE mean | Seed-avg RMSE | MAE mean | R² mean | Bias mean |
|---|---|---|---:|---:|---:|---:|---:|
| A scratch | R3 | 0,1 | 0.79162 | **0.76865** | 0.57420 | 0.5452 | −0.0677 |
| A SSL | R3 | 0,1 | 0.85037 | 0.81763 | 0.61124 | 0.4747 | −0.1039 |
| C tuned | R3 | 0,1 | 0.85048 | 0.83696 | 0.57843 | 0.4741 | +0.0068 |
| E R0 | R0 | 0,1 | 1.05099 | 1.00084 | 0.70765 | 0.1987 | −0.0119 |
| CPU XGB-P3 | — | — | 0.77878 | 0.77878 | 0.58852 | 0.5601 | +0.0132 |

The seed-averaged A−CPU difference is **−0.01013** (A nominally better), bootstrap 95% CI **[−0.11159, +0.11794]**; the per-seed A average is **+0.01284** worse than CPU. The honest reading is a **tie**: no material H4 gap, and the CPU baseline remains a strong, much cheaper reference. The optional PhysChem-augmented A/C diagnostic was **not run**. Its trigger was met on the A-R0 comparison (CPU 0.72357 vs A-R0 0.79759 on Fold 0, gap 0.074; 0.086 on the 3-fold mean), but the diagnostic was outside the frozen 56-run registry and no GPU authorization remained for new arms. It is a missed optional follow-up, not evidence that PhysChem augmentation would fail. The H4 table above compares CPU with the selected A-R3, where the CV gap is ~0.01.

## 8. Diagnostics (Fit dev, label-blind)

A seed-0 baselines across folds: A-R3 **0.72532**, C-R3 **0.72801**, E-R0 **0.83958** (diagnostics are single-seed).

- **S2 joint binding.** A's mean absolute prediction change under a joint S2 swap is **0.01637**, under an independent swap **0.29630**, and under a full path swap **1.76e-7**. C and E are exactly 0 under all three. This is consistent with A using the joint S2 pairing while C/E do not; the joint swap is not perfectly invariant (small residual), so the report does not claim exact invariance, only strong joint-binding sensitivity.
- **Global features.** Replacing the global vector by the training-fold mean changes A by 0.42347, C by 0.29575, E by 0.46350 (mean absolute prediction change). A is not a local-only model.
- **Error structure (A-R3, Fold 0).** RMSE 0.72279, MAE 0.52628, bias −0.0119. Atom-count quartiles: 0.79935 (9–20 atoms), 0.76496 (21–27), 0.62025 (28–32), 0.68947 (33–45). Worst scaffold groups have mean absolute errors 0.99, 0.93, 0.85, 0.83; absolute error correlations with ring atom fraction are weak (−0.11), i.e. no simple size/ring explanation dominates.

Full details: `analysis/diagnostics_v2.json`.

## 9. Integrity, label discipline and audit trail

- All model selection, recipe selection, SSL gating and fold decisions use Fit2352 CV only; the Monitor was never used for early stopping, seeds or selection.
- Monitor labels are opened only through `v2_monitor_labels.load_monitor_labels`, gated by `verify_manifest`; the artifact is read once as bytes through the v1 one-shot access guard, hashed, and scanned in memory. Exactly 504 values are constructed; the 2,352 Fit values are skipped.
- Predictions were frozen before labels: 9 npz files + sidecars, manifest hash `1bc705…`. `freeze_predictions` refuses overwrite; `evaluate` refuses a second evaluation if `monitor_metrics_v2.json` exists.
- **Invalid first Monitor execution.** As disclosed in AMENDMENT-003, the first execution froze unnormalized NN outputs and produced RMSE ~2.3. It is quarantined under `analysis/audit/monitor_invalid_normalization/` (old manifest `ec043a7f…`, metrics, decision, tables). It is not used. Two earlier launch attempts failed before metrics (one-shot guard denial; tuple unpack). The valid evaluation is the one in the main `analysis/` paths. The corrected predictions were regenerated without reading labels (their sidecars certify `monitor_labels_read=false`), but because the invalid execution had already computed aggregate Monitor metrics, the corrected evaluation is not a pristine first look. It is a disclosed corrected external check, not a blind test.
- No Reserved rows or official valid/test labels were read. v1 was unchanged before and after v2 tests (51 checks).
- All write-behavior tests use temporary directories; no run artifact is overwritten in place.

## 10. GPU ledger

| Item | Value |
|---|---:|
| Hard cap | 3.5 A100 h |
| Registered v2 runs charged | 56 |
| Extra charged runs | 3 pre-amendment seed-2 R0 + 2 duplicate successful A-R0-s1 |
| Total charged | **2.4647 h** |
| Remaining | **1.0353 h** |
| Status | WITHIN_CAP |
| v1 historical | 0.5539 h (separate) |

Failed duplicate jobs and the cancelled `res-2` Slurm pending job consumed no GPU allocation and are listed at zero in `analysis/gpu_ledger_v2.json`. The two duplicate A-R0-s1 executions and the three seed-2 runs are charged even though their results are not used.

## 11. Limitations

1. Two primary seeds, not three (AMENDMENT-001); Fold-0 R0 has a 3-seed sensitivity check, but folds 1/2 and the full retrain do not.
2. E's capacity matching is an effective-parameter construction with a dead S2 slot and a fixed residual, not a literal 529-input LocalMLP. It is the pre-registered operationalization, but it is not the only possible capacity match.
3. C-R0 was not run on folds 1/2, so H2's C arm is Fold-0-only; H2's robustness comes from A.
4. The Monitor is a single 504-molecule external check, not an independent blind test. Its H1 contrast is A-R3 vs E-R0, whereas the CV H1 test is A-R0 vs E-R0; the Monitor magnitude (0.232) therefore includes the R3 recipe advantage and should not be read as a pure capacity effect. The direction agrees with CV, but the magnitude should not be over-interpreted.
5. The first Monitor execution was invalid and is preserved only for audit. The valid evaluation is the one used; this is a disclosed deviation from the literal one-evaluation guard.
6. Diagnostics are single-seed (seed 0) and use Fit dev folds; they are mechanism evidence, not a substitute for a new external test.

## 12. Decision

See `DECISION.md`. In brief: keep LCRC-A with the R3 regularization recipe as the mechanism candidate; drop the SSL branch for this round; treat capacity-matched E as a failed control; treat the CPU XGB-P3 baseline as a near-tie and a mandatory reference for the next round. The next experiment should test external generalization and the S2 joint-binding mechanism, not add capacity or SSL epochs.
