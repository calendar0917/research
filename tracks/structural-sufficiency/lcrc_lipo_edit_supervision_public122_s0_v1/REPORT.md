# REPORT — LCRC Lipophilicity real-edit differential supervision

Round / protocol ID: `lcrc-lipo-edit-supervision-public122-s0-v1`
Training revision: `f0363e634caf`
Host/regime: `res`, process backend, one NVIDIA A100-SXM4-40GB per process
(CUDA_VISIBLE_DEVICES 0 or 1), torch 2.5.1+cu124, driver 550.163.01.
Seed: 0. No seed expansion, no hyperparameter search, no DDP, no `res-2`.

---

## 1. Bottom line

**The pre-registered common-benefit code is `EDIT_PERFORMANCE_TRADEOFF`, with a
strong caveat: this is not a robust edit benefit.** R's E_T component-equal
pair-difference MAE is lower than A by only 0.000751 (0.305260 vs 0.306011;
gain A−R = `+0.000751`; 95 % paired-component bootstrap interval
`[-0.027790, +0.027102]`). At the same time R's Test RMSE is higher by 0.016222
(gain A−R = `-0.016222`; `[−0.041010, +0.010198]`) and Valid RMSE is higher by
0.024694. The E_T gain is not supported by its interval, it
disappears under pair-equal weighting (R 0.314020 vs A 0.306786), and it reverses
when A's single worst E_T component is removed (gain `−0.000895`). Therefore the
honest conclusion is **no joint benefit in this configuration**, not “edit
supervision works but hurts performance”.

**The real-relation increment is `RELATION_INCREMENT_UNRESOLVED`.** R has the
best E_T point estimate among the four arms (0.305260 vs N 0.318295, W 0.320247,
A 0.306011), but its interval overlaps W and N, R's Test RMSE is worse than
both, and the fixed influence checks are not stable. This run cannot claim R is
better than paired-endpoint reweighting (W) or generic random-pair differential
training (N); nor can it claim they are equivalent.

No two-endpoints-untrained generalization claim is made: E_HH has only 31 pairs
/ 24 components and is `LOW_SUPPORT`; E_HH unseen-core has 6 pairs. E_T
Train–Test cores are seen in Train, so E_T measures edit response **involving**
a Test molecule, not a new core.

---

## 2. Data audit (P0, before formal training)

Frozen public ChemBench Lipophilicity `random122`; files verified by SHA256 and
the published split row-index bytes (Train/Valid/Test = 3360/420/420). All
labels come from the published `exp` column; no molecule/row filtering.

### 2.1 Pair construction

All 4200 structures were scanned label-blind:

* 3014 eligible molecules (single component, no explicit H, no stereo tokens);
* 11802 legal non-ring single-bond cuts;
* 9884 core classes; 988 shared-core candidate mappings checked, 0 failures;
* 823 unique unordered row pairs after protocol dedup; 165 superseded
  definitions.

The historical Train table had 509 rows but only **501 unique unordered row
pairs**: 8 unordered pairs had two directed definitions under the old
direction-keyed dedup. This round deduplicates the unordered row pair first and
reports 501 P_train pairs with the same 502 participating Train molecules. This
is a dedup correction, not a relaxed filter. The check is stored in
`analysis/data_audit.json::historical_regression_check` and is re-run by the
tests.

### 2.2 Support

| set | pairs | molecules | components | note |
|---|---:|---:|---:|---|
| P_train | 501 | 502 | 175 | 14.94 % of Train molecules are endpoints |
| E_T | 178 | 199 | 59 | at least one Test endpoint |
| — Train–Test | 147 | 177 | 55 | core necessarily seen in Train |
| — Valid–Test | 19 | 35 | 16 | |
| — Test–Test | 12 | 21 | 10 | |
| E_HH | 31 | 51 | 24 | **LOW_SUPPORT** |
| — seen core | 25 | 39 | 18 | **LOW_SUPPORT** |
| — unseen core | 6 | 12 | 6 | **LOW_SUPPORT** |
| Train–Valid (descriptive) | 139 | 196 | 59 | |
| Valid–Valid (descriptive) | 5 | 10 | 5 | |

All P0 gates pass: P_train 501 ≥ 100 and 175 ≥ 30 components; E_T 178 ≥ 40 and
59 ≥ 15 components. Data audit status `OK`. E_HH is below the fixed low-support
threshold and cannot support a structural-generalization claim.

### 2.3 Frozen schedule

9720 optimizer steps; 32 real pairs per step; 0 step redraws; maximum shuffle
attempts 14 (cap 256); 0 final N edges equal to a P_train real edit edge, 0 same
row, 0 same canonical molecule. Endpoint multiset equality holds for all 9720
steps. See `analysis/schedule_manifest.json`.

---

## 3. Four-arm results

All arms use C1 `NO_BOND_MEAN`, 228,674 parameters, one shared initial state
hash `aac1e061…`, one shared ordinary batch order `b10278f9…`, and identical
first-epoch RNG hashes at all 27 steps. Formal recipe: 360 epochs, 27 steps per
epoch, batch 128, AdamW, per-step cosine 3e-4 → 1e-5, wd 1e-3, dropout 0.10,
clip 5.0, FP32, no AMP/early stopping/soup/EMA. Checkpoint selection is the
lowest Valid RMSE (raw logD), strict improvement, earlier epoch on ties.

| arm | best epoch | Train RMSE | Valid RMSE | Test RMSE | E_T CE-MAE | E_T pair-MAE | E_T zero-change | E_HH CE-MAE | E_HH pair-MAE | E_HH zero-change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A (molecule MSE) | 244 | 0.288252 | 0.702536 | **0.576167** | 0.306011 | **0.306786** | 0.607809 | 0.355794 | 0.340428 | 0.509032 |
| R (real edit) | 117 | 0.403925 | 0.727230 | 0.592390 | **0.305260** | 0.314020 | 0.607809 | 0.359800 | 0.351731 | 0.509032 |
| W (endpoint reweight) | 140 | 0.404822 | 0.702066 | 0.576643 | 0.320247 | 0.324795 | 0.607809 | **0.341947** | **0.338203** | 0.509032 |
| N (random pairs) | 146 | 0.390147 | **0.695048** | 0.590433 | 0.318295 | 0.326225 | 0.607809 | 0.363623 | 0.353678 | 0.509032 |

CE-MAE is component-equal (primary edit metric); pair-MAE gives every pair equal
weight; zero-change is the reference `mean |y_b-y_a|` (same weights). All arms
are far better than the zero-change reference, but the differences between arms
are small relative to the pair-level noise.

Test subsets (component-equal MAE, no bootstrap interval):

| arm | Train–Test | Valid–Test | Test–Test | E_HH seen | E_HH unseen |
|---|---:|---:|---:|---:|---:|
| A | 0.318137 | 0.329421 | 0.377229 | 0.385648 | **0.266233** |
| R | **0.317543** | 0.348776 | **0.363306** | 0.369496 | 0.330715 |
| W | 0.336565 | 0.342982 | 0.325647 | **0.360248** | 0.287046 |
| N | 0.331049 | 0.362181 | 0.346989 | 0.380759 | 0.312215 |

R's apparent E_T advantage is concentrated in Test–Test (+0.0139 vs A) and is
slightly positive on Train–Test (+0.0006); it is negative on Valid–Test
(−0.0194). E_HH unseen-core is worse for R (−0.0645 vs A) on only 6 pairs.

---

## 4. Main comparison: R vs A

Gain is `control − R` (protocol section 8): positive means R better. Bootstrap:
1000 paired structural-component replicates, seed 20261114.

| metric | A | R | gain (A−R) | 95 % CI | reading |
|---|---:|---:|---:|---|---|
| Test RMSE | 0.576167 | 0.592390 | **−0.016222** | [−0.041010, +0.010198] | R worse on point estimate; CI crosses |
| Valid RMSE | 0.702536 | 0.727230 | **−0.024694** | point only | R worse |
| E_T CE-MAE | 0.306011 | 0.305260 | **+0.000751** | [−0.027790, +0.027102] | R barely better; not supported |
| E_T pair-MAE | 0.306786 | 0.314020 | **−0.007234** | point only | R worse under pair weighting |
| E_HH CE-MAE | 0.355794 | 0.359800 | −0.004006 | [−0.062585, +0.047539] | R slightly worse; low support |

Pair-level sign check on E_T: R is better on 95/178 pairs and worse on 83/178,
but the mean paired absolute-error gain is `−0.00723` (R worse) and the median
is `+0.00602`. The component-equal metric gives R a tiny positive point
estimate; pair-equal weighting reverses it. This metric sensitivity is part of
the result, not a reason to choose a friendlier metric.

### 4.1 Fixed influence checks

* A's worst E_T component is id 2548: 1 pair, A mean absolute difference error
  2.1047. Removing that **single pair** changes R vs A E_T gain from `+0.000751`
  to `−0.000895` (A better), with 177 pairs / 58 components left. The tiny edit
  gain is therefore not robust.
* A's worst Test molecule is row 3986 (A squared error 4.7354). Removing it
  leaves 419 Test rows and all 178 E_T pairs (it has no E_T pair). R vs A Test
  RMSE gain remains negative (`−0.017666`); E_T gain remains `+0.000751`.
  The performance deficit is not caused by one Test molecule.

### 4.2 Fixed cases (illustrative only)

Cases are selected by R vs A on E_T; they are not effect-size estimates.

| case | pair | split | true Δy | A Δpred / abs err | R Δpred / abs err |
|---|---|---|---:|---:|---:|
| max improvement | p02605_03352 | Train–Test | +0.69 | −0.2431 / 0.9331 | +0.1997 / 0.4903 |
| max worsening | p01511_01863 | Train–Test | −0.99 | −0.6012 / 0.3888 | +0.0744 / 1.0644 |
| median improvement | p03384_04082 | Train–Test | +0.70 | +0.4807 / 0.2193 | +0.4858 / 0.2142 |

The largest improvement and largest worsening are both Train–Test pairs with
Train-seen cores. This is exactly why the round does not claim new-core
generalization.

---

## 5. Mechanism comparisons: R vs W and R vs N

| comparison | metric | control | R | gain (control−R) | 95 % CI | reading |
|---|---|---:|---:|---:|---|---|
| R vs W | E_T CE-MAE | 0.320247 | 0.305260 | **+0.014987** | [−0.010062, +0.041242] | R better point; not supported |
| R vs W | Test RMSE | 0.576643 | 0.592390 | **−0.015747** | [−0.040072, +0.008043] | R worse point; not supported |
| R vs W | Valid RMSE | 0.702066 | 0.727230 | −0.025164 | point only | R worse |
| R vs N | E_T CE-MAE | 0.318295 | 0.305260 | **+0.013036** | [−0.007465, +0.032598] | R better point; not supported |
| R vs N | Test RMSE | 0.590433 | 0.592390 | **−0.001956** | [−0.033195, +0.031172] | R slightly worse; not supported |
| R vs N | Valid RMSE | 0.695048 | 0.727230 | −0.032182 | point only | R worse |

Both E_T intervals include zero, and R's Test RMSE point estimate is worse than
both controls. R's E_T point estimate is nevertheless the best of the four arms
(0.3053), while A is second (0.3060); W and N are 0.3202 and 0.3183. This is a
small, non-significant ordering, not an increment. The pre-registered code is
`RELATION_INCREMENT_UNRESOLVED`, not `RELATION_INCREMENT_THIS_RUN`; the controls
are also not significantly better, so `CONTROL_BETTER_THIS_RUN` is not claimed.

Subset directions are mixed: R vs W is better on Train–Test (+0.0190) and worse
on Valid–Test (−0.0058) and Test–Test (−0.0377); R vs N is better on Train–Test
(+0.0135) and Valid–Test (+0.0134) but worse on Test–Test (−0.0163). No
consistent relation-specific advantage.

### 5.1 Training diagnostics (corrected units)

`train_diagnostics.json` initially mixed standardized predictions with raw
targets for the endpoint/nonpaired split. Training, checkpoint selection and all
evaluation metrics were unaffected. The code is fixed and the affected fields
were recomputed from `predictions_best.npz` / `predictions_initial.npz`; the
correction is recorded in `analysis/diagnostics_correction.json`. Corrected
Train diagnostics at each arm's best checkpoint:

| arm | Train endpoint RMSE (502) | Train nonpaired RMSE (2858) | P_train pair CE-MAE (501) |
|---|---:|---:|---:|
| A | **0.2895** | **0.2880** | 0.32561 |
| R | 0.3198 | 0.4169 | **0.22734** |
| W | 0.2545 | 0.4258 | 0.27007 |
| N | 0.2404 | 0.4108 | 0.26072 |

This is the clearest mechanistic signal in the run: R fits **Train pair
differences** much better than the other arms, but pays for it on both endpoint
and nonpaired Train levels, and that pair-difference fit does not transfer to
E_T. W and N fit Train endpoint levels best while also fitting nonpaired
molecules worse than A. These diagnostics are post-hoc and do not change the
pre-registered decision.

Auxiliary-gradient diagnostics (steps 1, 28, 271, 2701) show that R's loss is a
much weaker update than W/N: R's aux/molecule gradient L2 ratio is 0.0028, 0.0556,
0.1860, 0.0293; W's is 0.8983, 0.6919, 0.7444, 0.4606; N's is 0.0818, 0.1013,
0.8652, 0.4210. R's auxiliary direction is also less aligned with the molecule
gradient (cosine −0.63 to +0.55) than W's (0.976–0.996). This is consistent with
a weak residual-coupling objective, but it is a design property, not proof that
a larger λ would help; λ was fixed at 1.0 and was not searched.

---

## 6. Test descriptive groups and endpoint

Test molecules with a Train real-edit neighbour (66 rows) have lower error than
those without (354 rows) for every arm; this is descriptive and is not attributed
to edit training (the groups differ in structure and label range):

| arm | with neighbour RMSE (n=66) | without neighbour RMSE (n=354) |
|---|---:|---:|
| A | 0.440695 | 0.598041 |
| R | 0.404796 | 0.621130 |
| W | 0.419999 | 0.601351 |
| N | 0.482888 | 0.608385 |

Last-epoch values are reported beside best-epoch values but never used for
re-selection. For A, last Test RMSE is 0.571863 (best 0.576167) while last E_T
CE-MAE is 0.309762 (best 0.306011); for R, last Test 0.585589 and last E_T
0.300719. Last-epoch improvements do not rescue the common-benefit result:
they are a single terminal point, not a selected checkpoint, and R's last Test
RMSE is still worse than A's best and last.

---

## 7. Correctness, provenance and budget

* Source bytes, split sizes/row hashes, Train-only target normalization,
  feature-spec hash, 228,674 parameters, and shared initial state: pass.
* Loss identity `(e_b−e_a)^2/2 = (e_a^2+e_b^2)/2 − e_a e_b`, analytic
  gradients, N label-difference recomputation, reverse-direction sign, unordered
  dedup, core isomorphism/attachment on all pairs: pass.
* All 9720 N/R endpoint multisets equal; no forbidden N edge; shared ordinary
  batch order; auxiliary eval forward has nonzero gradients and restores train
  mode without RNG/buffer side effects: pass.
* 20 pre-training gates and 9 post-training gates (checkpoint reload, independent
  NumPy metrics, last-epoch record, shared RNG stream): **29/29 pass**.
* 12 focused tests pass.
* All four arms share initial-state hash `aac1e061…`, batch-order hash
  `b10278f9…`, and all 27 first-epoch RNG hashes.
* Formal A100 GPU hours: A 0.0866, R 0.1744, W 0.1747, N 0.1445;
  **total 0.5802 A100 GPU·h**. GPU smoke adds <0.01 h. Budget `BUDGET_OK`
  (soft 2.0 h, hard 4.0 h); no retries of scientific configurations.
* Nonfinite events: 0 in all arms. Gradient clipping occurred frequently
  (A 51.1 %, R 52.6 %, W 65.9 %, N 58.3 % of steps), as expected at the frozen
  peak LR; this was not tuned after seeing results.
* CUDA bitwise determinism is not claimed; the shared RNG stream and checkpoint
  reload checks are the reproducibility evidence.

---

## 8. Deviations and corrections

1. **Historical 509 vs 501.** The earlier Train table counted 8 unordered pairs
   twice under a directed dedup key. This protocol dedups unordered row pairs
   first; the unique set and 502 molecules are identical. Documented and tested.
2. **Evaluation direction.** The first evaluation pass had `reference`/`candidate`
   reversed relative to the protocol's `gain = reference − candidate`. It was
   corrected before the final tables: for `R_vs_A`, reference is A and candidate
   R, so positive gain means R better. Formal training was unaffected. A
   regression test now checks the identity for all three comparisons.
3. **Diagnostics units.** The endpoint/nonpaired training diagnostics initially
   compared standardized predictions with raw targets. Fixed and recomputed from
   saved predictions; see section 5.1 and `analysis/diagnostics_correction.json`.
   No checkpoint, prediction, metric, or decision depends on these fields.
4. **Remote schedule cache.** `prepare.py --cache-only` rebuilds the ignored
   schedule cache from committed P_train pairs and verifies all four array
   hashes against the committed manifest. This avoids committing a 4.2 MB
   binary while keeping the schedule frozen by hash.

No chemical features, context modules, propagation layers, attention,
explanation heads, residual correctors, external pretraining, seed expansion, or
hyperparameter search were added.

---

## 9. Limitations

* One training seed (0). Bootstrap intervals condition on the four trained
  models; they are not training-seed uncertainty.
* One target, one split (`random122`), one model (C1, 228,674 parameters), one
  λ (1.0), one pair budget (32/step), one pair rule. The result is local to this
  configuration.
* E_T has 178 pairs / 59 components; E_HH has 31 / 24. Power for small edit
  effects is limited. E_HH is low-support and cannot support a
  two-endpoints-untrained structural claim.
* Test labels were already public and were not used for model/checkpoint
  selection, but this is not a pristine test set.
* Bootstrap intervals and influence checks are sensitivity analyses, not
  additional training runs.
* No claim about unique atom contributions, chemical causality, cross-seed
  stability, all-split validity, or superiority over other GNNs.
* R's auxiliary objective is weak relative to W/N; the experiment does not
  distinguish “real relations do not help” from “this λ/pair budget makes the
  real-relation update too weak”. That distinction would require a new,
  pre-registered strength sweep, which this round intentionally does not do.

---

## 10. Verdict

In this configuration, real-edit differential supervision **does not produce a
robust joint benefit**. It improves the component-equal E_T point estimate by
0.00075 and fits Train pair differences much better than the controls, but it
worsens Test/Valid molecule prediction, loses under pair-equal weighting, and
its tiny E_T gain reverses when one E_T pair is removed. The relation increment
over paired-endpoint reweighting and generic random-pair training is unresolved;
R's E_T point estimate is nominally best, but the intervals overlap and R's
molecule performance is worse.

The correct next step is not to tune λ on these results. If the question is
pursued, freeze a new protocol with a pre-registered strength/pair-budget
comparison and a larger E_T/E_HH support target, and evaluate on a genuinely
held-out test regime. For this round, the negative/inconclusive result stands.

---

## 11. Artifacts

* `PROTOCOL.md` — frozen rules.
* `configs/experiment.yaml` — machine-readable configuration.
* `analysis/data_audit.json` — P0 audit, historical dedup check, support.
* `analysis/pairs.csv`, `pair_mappings.json`, `core_classes.json` — pair/core
  evidence (823 pairs; 501 P_train).
* `analysis/schedule_manifest.json` — 9720-step schedule hashes/counts.
* `analysis/correctness_gates.json` — 29/29 gates.
* `analysis/results.csv` — four-arm metrics.
* `analysis/comparisons.json` — gains, CIs, impact checks, decisions.
* `analysis/cases.json` — three fixed cases.
* `analysis/per_molecule.csv`, `per_pair.csv` — row/pair predictions and errors.
* `analysis/diagnostics_correction.json` — diagnostics-unit correction.
* `runs/{A,R,W,N}_s0/` — local copies of checkpoints, curves, predictions,
  `train.json`, diagnostics (git-ignored; remote paths declared to `rr`).
* `tests/test_edit_supervision.py` — 12 focused checks.
