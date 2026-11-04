# REPORT — LCRC Lipophilicity bond-object pairings pilot v1

- **Round:** `tracks/structural-sufficiency/lcrc_lipo_bond_object_pairing_v1/`
- **Protocol id:** `lcrc-lipo-bond-object-pairing-v1`
- **Final status:** `COMPLETE_MONITOR_EVALUATED`
- **Final label:** **`EXPOSED_MONITOR_EXPLORATORY_ONLY`**
- **Pre-registered decision case:** **`NO_CLEAR_VALIDATION_SIGNAL`**
- **Frozen code commit:** `9887fc380ab081ccec054a8aef3470f0877c87ad`
  (branch `exp/lcrc-lipo-bond-object-pairing-v1`)
- **Prediction freeze:** `analysis/prediction_manifest.json`
  SHA-256 `8c1a234ff8c1b9854b8e1d09914aa78b8bd98e36432897774f1d45dabfcfcc8a`
- **Monitor evaluation:** one label read at `2026-11-04T03:41:27Z`, after the freeze
- **Execution device:** local CPU, 4 threads; **0 A100 GPU-hours used**
- **Reserved504 / official validation / official test:** never read or scored

This report applies only the pre-registered rules in `PROTOCOL.md`. It does not
reinterpret the result, authorise a follow-up round, or treat Monitor504 as a
blind test. Monitor504 was already exposed in SSRC Phase 1a.

---

## 1. One-page answer

The six formal runs (`R_s0, N_s0, R_s1, N_s1, R_s2, N_s2`) completed with the
frozen recipe and produced frozen, label-blind Monitor504 predictions. The
pre-registered decision rule returns **`NO_CLEAR_VALIDATION_SIGNAL`**.

| quantity | value |
|---|---:|
| A historical mean Monitor RMSE | **0.9100237932** |
| R (real endpoint pairing) mean Monitor RMSE | **0.9010772297** |
| N (type-conditioned mean-field null) mean Monitor RMSE | **0.9222639020** |
| **mean R − A RMSE** | **−0.0089465635** |
| **mean R − N RMSE** | **−0.0211866723** |
| mean N − A RMSE | +0.0122401088 |
| mean R − A MAE | +0.0006610942 |
| seeds with R better than both A and N | 2 / 3 |
| materiality threshold | 0.03 raw RMSE |

Both primary mean differences are in the expected direction but **neither
reaches the 0.03 materiality threshold**:

- R − A = −0.00895, which is greater than −0.03 (not material);
- R − N = −0.02119, which is greater than −0.03 (not material).

The secondary MAE condition is satisfied (R − A = +0.00066, not worse by more
than 0.01), and R beats A on all three seeds, but the frozen decision rule
requires both RMSE thresholds. It therefore returns
**`NO_CLEAR_VALIDATION_SIGNAL`**, not `BOTH_SIGNALS`, not
`RELATION_SIGNAL_NOT_COMPETITIVE`, and not
`BOND_BRANCH_GAIN_NOT_ATTRIBUTABLE_TO_PAIRING`.

The mechanism diagnostic is descriptive only: on the best R seed, swapping the
real endpoint pairing for the mean-field null at inference changes Monitor RMSE
by **+2.28 × 10⁻⁷** (R 0.8725162194 → 0.8725164472). The learned R residual is
essentially insensitive to the pairing term. That does not change the decision
case, because R − A itself is below materiality.

---

## 2. Question and frozen design

Question: with the original LCRC-A trunk unchanged as a computation, does a
trainable per-physical-bond residual that uses the **real endpoint pairing**
`z_u ⊙ z_v` produce a material Monitor504 gain beyond (a) the historical A
baseline and (b) an identically parameterised **type-conditioned mean-field
null** that preserves the per-(molecule, bond-type) stub multiset but destroys
which atoms are actually bonded?

The arms are:

- **A** — historical frozen LCRC-A predictions, reused read-only, not retrained;
- **R** — `REAL_PAIRING`, `j_e = z_u ⊙ z_v` for physical bond `e=(u,v)`;
- **N** — `TYPE_CONDITIONED_MEANFIELD_NULL`, the exact stub-pair mean
  `j_null = (sum_z² − sum_z2) / (S(S−1))` per `(molecule, bond_type)` group,
  with `S = 2M` stubs and denominator clamped only for unused empty groups.

Frozen constraints that were not changed: no trunk edits, no SSL, no early
stopping, no monitor-based model selection, no hyperparameter/structure search,
no threshold changes. The six runs use one backend. Seeds are `0,1,2`.

---

## 3. Data boundary and label access

| role | n | use |
|---|---:|---|
| Fit | 2352 | only supervised training data; Fit-only target standardisation |
| Monitor | 504 | label-blind prediction, then one frozen evaluation |
| Reserved | 504 | never accessed |
| official valid/test | — | never accessed |

Verified frozen inputs:

| artifact | SHA-256 / fingerprint |
|---|---|
| phase1a split | `9fa96d49efd94cd2ff381922f83a07f963d832064c75469657b4ce5f68c332b6` |
| `labels_fit_monitor.json` | `0260a3909eec30900df646bc760e65360315f4a1398c7092c7c72d04a4e404f7` |
| `feature_spec_lipo.json` | `a65c67f9595d9190e7c913d2594ba984a07b51300d63916de865dccca434206c` |
| `norm_lipo.json` | `084c129c1731ea92a67d5e9bdce0d8d3bb335cf75ef45595ee690da288394d77` |
| Fit cache content | `0a73407fe3b20bb5158b521bcd8dd94bac69d0857ec5c5bd31314c378e9990d9` |
| Monitor cache content | `db48a9c237d628025185f76497a024dac35537c77413c511b1d76fb3721919b8` |

Split sizes and row hashes:

- Fit 2352, row hash `c474e38cdecf…`; Monitor 504, row hash `fe81cc79…`;
  Reserved 504, row hash `bc28c5d3…`; pairwise overlap 0.
- Target normalisation: mean `2.157857142857143`, population scale
  `1.2241982660328445`, certified Fit-only.
- Feature dimensions: atom 51, bond 13, global 42, scalars 9,
  `n_bond_types=5`.

Cache caveat: the torch serialized **file** bytes differ from the historical
serialization, but all array contents match the frozen content fingerprints.
`analysis/data_audit.json` records both facts; the content fingerprint is the
authoritative consistency check.

Label-access audit from the one-shot evaluation:

- `label_values_constructed = 504`
- `label_values_skipped = 2352`
- `unrequested_values_constructed = 0`
- `reserved_rows_in_artifact = 0`
- `official_valid_test_rows_in_artifact = 0`
- `monitor_label_vector_sha256 = 769b33398d739774bb7f24b908c58a2bebc143e7f74a823b70de41f87d4661fd`
- prediction manifest verified before the read:
  SHA-256 `8c1a234f…`

Training and prediction artifacts all record `monitor_labels_read=false`,
`reserved_labels_read=false`, `official_valid_test_read=false`.

---

## 4. Model, parameter and initialisation audit

The original LCRC-A trunk is fully trainable and unchanged as a computation:

- trunk parameters: **247,106**
- new branch parameters: **12,449**

  | component | parameters |
  |---|---:|
  | `Linear(64,32)` atom context | 2,080 |
  | `Linear(128,64)` edge hidden | 8,256 |
  | `Linear(64,32)` edge hidden | 2,080 |
  | `Linear(32,1)` score, zero-init | 33 |

- total, both R and N: **259,555**

Initialisation protocol, checked for all three seeds:

1. `torch.manual_seed(seed)` and build the original trunk;
2. trunk state hash matches the historical A init hash exactly;
3. build the new modules;
4. zero the final `Linear(32,1)` weight and bias;
5. R and N have the same initial state hash for a given seed;
6. at initialisation, `prediction_R == prediction_N == prediction_A`
   (max absolute difference **0.0**).

| seed | trunk init hash | R/N init state hash |
|---:|---|---|
| 0 | `e7a688ff91f29489…` | `f7e6aa36c1b6c9c4…` |
| 1 | `de28d5f425a2f7f8…` | `3a5fde7b2fd41180…` |
| 2 | `08ed4e6dcc895af0…` | `183d2f280fb69de7…` |

Parameter/init evidence: `analysis/parameter_audit.json` (all checks pass).

---

## 5. Correctness gates

`analysis/gates.json`: **G1–G7 all PASS**, `all_passed=true`, at code commit
`9887fc38…`. The pytest suite `tests/test_bond_pairing.py` passed **8/8**.

| gate | evidence summary |
|---|---|
| G1 data/permissions | split/cache/fingerprint checks; Fit label audit 2352 constructed, 504 skipped, 0 unrequested |
| G2 trunk preservation | historical A init hashes match; R/N parameter counts 259,555; R/N initial state equal |
| G3 bond-object invariance | atom-renumbering max |Δ| 4.77e-7; bond reorder 0; batch split/merge 0; edge cases finite |
| G4 mean-field formula | exact stub-pair mean vs brute force 5.96e-8; N differs from real pairing (1.97 and 0.51 on the test cases) |
| G5 relation sensitivity | R nonlinear score changes under pairing swap; N group statistics equal; R pairing statistics differ |
| G6 optimisation path | same init/config/parameter count; both branches update after 3 steps; no parameter frozen by accident |
| G7 resource/conflict | independent worktree/branch, clean tracked tree, no Peptides process on the selected GPU, GPU1 free; CPU used |

The smoke test retained in `analysis/smoke.json` is 512 Fit rows × 12 epochs:
R and N both converge to Fit soup RMSE **1.05242**; no non-finite value.
An earlier 128-row × 2-epoch smoke had only one update per epoch and was not
informative; it was discarded and is not part of the evidence.

---

## 6. Execution backend and resource isolation

The CPU benchmark measured one complete Fit2352 epoch per arm:

| arm | seconds / epoch | updates |
|---|---:|---:|
| R | 0.7501 | 19 |
| N | 0.7915 | 19 |

Estimated six-run CPU time: **1709.7 s**, below the pre-declared 2700 s CPU
eligibility threshold. Backend was therefore fixed to **CPU, 4 threads** for all
six formal runs (`analysis/backend.json`). No A100 was used.

Resource facts:

- total formal training wall time: **1732.9 s = 28.88 min**
- per-run wall: 286.1–291.3 s
- peak RSS: 631.7–643.7 MB per run
- GPU-hours: **0.0000**
- no `CUDA_VISIBLE_DEVICES` set; local `torch.cuda.is_available()=false`
- `analysis/resource_isolation.json`: **PASS**, blocking list empty

The parallel Peptides run was observed on remote GPU0 during preflight and
completed at `2026-11-04T10:57:45+08:00`, before the formal runs. Its checkout
`/home/hxy/cy/research`, GPU0, checkpoints and deploy directory were not touched.
GPU1 remained free throughout. This round used no remote checkout and no GPU.

---

## 7. Formal training facts

All six runs used the frozen recipe: AdamW, LR 1e-3 → 1e-5 per-step cosine,
weight decay 1e-5, batch 128, 360 epochs, 19 updates/epoch, 6840 updates,
grad clip 5.0, dropout 0, MSE on the Fit-standardised target, soup = mean of the
last 5 epoch-end state dicts. No early stopping, no monitor selection.

| run | wall s | Fit soup RMSE | Fit soup MAE | grad-norm max | clip events | non-finite |
|---|---:|---:|---:|---:|---:|---:|
| R_s0 | 286.9 | 0.090998 | 0.052158 | 5260.4 | 1790 | 0 |
| N_s0 | 290.3 | 0.095988 | 0.055056 | 5260.4 | 1850 | 0 |
| R_s1 | 287.4 | 0.098614 | 0.056097 | 5630.6 | 1835 | 0 |
| N_s1 | 291.3 | 0.092746 | 0.052944 | 5630.6 | 1754 | 0 |
| R_s2 | 286.1 | 0.097713 | 0.055822 | 5345.4 | 1719 | 0 |
| N_s2 | 290.7 | 0.096082 | 0.055316 | 5345.4 | 1661 | 0 |

For each run:

- 360/360 epochs and 6840/6840 updates completed;
- `batch_order.matches=true`; independent recomputation reproduced the batch
  hash `1aeefe0f9276b256…`;
- all 2352 Fit cache positions were covered exactly once per epoch;
- losses, gradient norms and optimizer steps were finite;
- `soup.pt` hash equals the recorded `soup_state_hash`;
- all flags `monitor_labels_read=false`, `reserved_labels_read=false`,
  `official_valid_test_read=false`.

The large pre-clip gradient norms and frequent clipping are part of the frozen
recipe (clip 5.0), not a deviation.

---

## 8. Prediction freeze

After all training, label-blind Monitor inference produced:

- six primary prediction files: `R_s0, N_s0, R_s1, N_s1, R_s2, N_s2`;
- three frozen-weight mechanism counterfactuals: `R_s0_nullswap`,
  `R_s1_nullswap`, `R_s2_nullswap` (R weights, N operator at inference only);
- three historical A prediction files reused read-only.

All nine R/N/counterfactual files plus the three A files have 504 rows in the
frozen Monitor order. `rows_sha256 =
a31ab4d07d4f81e5c25bffa71e6a696fb620d0c1a46cf17a4ea1b255b1539c24`.

| key | parameters | prediction SHA-256 |
|---|---:|---|
| R_s0 | 259,555 | `3bec2140be28…` |
| R_s1 | 259,555 | `5e4f48d8db53…` |
| R_s2 | 259,555 | `b3274bd75e9f…` |
| N_s0 | 259,555 | `5c0c92946769…` |
| N_s1 | 259,555 | `3f102c80ecd3…` |
| N_s2 | 259,555 | `83ac53861ab3…` |
| R_s0_nullswap | 259,555 | `1d223b42b397…` |
| R_s1_nullswap | 259,555 | `8759e085f064…` |
| R_s2_nullswap | 259,555 | `e8a9c4b1792c…` |
| A_s0 | 247,106 | `3b40979b0686…` |
| A_s1 | 247,106 | `49ca4b5b0011…` |
| A_s2 | 247,106 | `6c1360979cee…` |

The manifest records the rows, config hash
`2005d1f12630b71ec96231e9b442183e78e648a6b79f0e349f4cbabd98cc8d90`,
16 code hashes, recipe, soup/init hashes and all prediction hashes. The lock file
`analysis/prediction_manifest.sha256` pins the manifest SHA-256
`8c1a234f…`. Evaluation refuses to start without both and re-verifies every
file; this check passed before the single label read.

---

## 9. Monitor504 absolute metrics

Units are raw logD/exp. A metrics recomputed from the frozen historical
predictions match the historical report **exactly** (`abs_diff = 0.0` for all
three seeds).

| arm | seed | RMSE | MAE | R² | bias |
|---|---:|---:|---:|---:|---:|
| A | 0 | 0.934324 | 0.657335 | 0.36683 | −0.00655 |
| A | 1 | 0.880661 | 0.630620 | 0.43747 | +0.01276 |
| A | 2 | 0.915086 | 0.653724 | 0.39263 | −0.03222 |
| R | 0 | 0.931178 | 0.658718 | 0.37108 | −0.05727 |
| R | 1 | 0.872516 | 0.625364 | 0.44783 | +0.00399 |
| R | 2 | 0.899537 | 0.659578 | 0.41310 | −0.02868 |
| N | 0 | 0.996108 | 0.677017 | 0.28032 | −0.05809 |
| N | 1 | 0.878087 | 0.640046 | 0.44075 | −0.01788 |
| N | 2 | 0.892596 | 0.657235 | 0.42212 | −0.04893 |

Seed summary:

| arm | RMSE mean | RMSE SD | RMSE min | RMSE max | MAE mean | seed-mean-pred RMSE |
|---|---:|---:|---:|---:|---:|---:|
| A | **0.910024** | 0.022198 | 0.880661 | 0.934324 | 0.647226 | 0.853221 |
| R | **0.901077** | 0.023973 | 0.872516 | 0.931178 | 0.647887 | 0.844368 |
| N | **0.922264** | 0.052551 | 0.878087 | 0.996108 | 0.658099 | 0.862312 |

The seed-mean-prediction row is a secondary ensemble description, not the
decision metric. The decision uses the mean of per-seed RMSEs.

Fit diagnostics (training-only, not generalisation):

| arm | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| A | RMSE 0.095827 / MAE 0.055125 | 0.097433 / 0.056082 | 0.098562 / 0.056903 |
| R | 0.090998 / 0.052158 | 0.098614 / 0.056097 | 0.097713 / 0.055822 |
| N | 0.095988 / 0.055056 | 0.092746 / 0.052944 | 0.096082 / 0.055316 |

A Fit values are a read-only copy of the historical A run diagnostics
(`analysis/historical_A_fit.json`).

---

## 10. Paired comparisons and pre-registered decision

Definition used here and in the tables:
`delta = metric(first arm) − metric(second arm)`. A negative RMSE delta means
the first arm has lower RMSE and is better. The generic `definition` strings
inside the JSON comparison objects were generated by a template and should be
read with this first-arm/second-arm convention.

| comparison | mean ΔRMSE | SD across seeds | per-seed ΔRMSE (s0, s1, s2) |
|---|---:|---:|---|
| R − A | **−0.008947** | 0.005095 | −0.003146, −0.008145, −0.015549 |
| R − N | **−0.021187** | 0.031350 | −0.064930, −0.005571, +0.006941 |
| N − A | +0.012240 | 0.035964 | +0.061784, −0.002574, −0.022490 |

Direction counts:

- R better than A: **3/3** seeds;
- R better than N: **2/3** seeds;
- R better than both A and N: **2/3** seeds (seeds 0 and 1; seed 2 is worse
  than N by +0.00694).

Paired molecule bootstrap (2,000 resamples, seed 0, molecule unit; seeds are not
treated as independent resamples):

| comparison | seed-mean RMSE A | seed-mean RMSE B | ΔRMSE | 95% CI |
|---|---:|---:|---:|---|
| R − A | 0.853221 (A) | 0.844368 (R) | −0.008853 | [−0.035711, +0.016173] |
| R − N | 0.844368 (R) | 0.862312 (N) | −0.017944 | [−0.039869, +0.006511] |
| N − A | 0.862312 (N) | 0.853221 (A) | +0.009091 | [−0.025078, +0.035335] |

All three seed-mean intervals cross zero. Per-seed intervals also cross zero
(see `analysis/paired_metrics.json`).

Decision-rule check:

| condition | observed | required | pass? |
|---|---:|---:|---|
| mean R − A RMSE | −0.008947 | ≤ −0.03 | **no** |
| mean R − N RMSE | −0.021187 | ≤ −0.03 | **no** |
| seeds R better than both | 2 / 3 | ≥ 2 / 3 | yes |
| mean R − A MAE | +0.000661 | ≤ +0.01 | yes |

Because the first two materiality conditions fail, the frozen rule returns
**`NO_CLEAR_VALIDATION_SIGNAL`**. It does not return `BOTH_SIGNALS`
(thresholds fail), `RELATION_SIGNAL_NOT_COMPETITIVE` (R − N not ≤ −0.03), or
`BOND_BRANCH_GAIN_NOT_ATTRIBUTABLE_TO_PAIRING` (R − A not ≤ −0.03). No new
interpretation is applied.

Final exposure label: **`EXPOSED_MONITOR_EXPLORATORY_ONLY`**. Monitor504 is a
development split already used before this round; this is not an independent
blind result and does not authorise a follow-up.

---

## 11. Mechanism and robustness diagnostics (descriptive only)

### 11.1 Delta residual magnitudes

| arm | seed | delta std (raw) | delta abs max | delta mean |
|---|---:|---:|---:|---:|
| R | 0 | 0.001971 | 0.025651 | +0.012584 |
| R | 1 | 0.002182 | 0.028377 | −0.013927 |
| R | 2 | 0.001125 | 0.014626 | −0.007182 |
| N | 0 | 0.001906 | 0.024831 | +0.012157 |
| N | 1 | 0.103297 | 0.609402 | +0.006102 |
| N | 2 | 0.001146 | 0.014854 | −0.007306 |

N has two separate outliers: N_s1 has the much larger residual magnitude, while
N_s0 has the worst Monitor RMSE (0.9961). N's cross-seed RMSE spread (0.0526)
exceeds the materiality threshold; the N mean is therefore seed-sensitive. R's
residual is small and consistent in magnitude across seeds.

### 11.2 Frozen-weight R → N operator swap

On the best R seed by Monitor RMSE (seed 1):

| quantity | value |
|---|---:|
| R Monitor RMSE | 0.8725162194 |
| R-nullswap Monitor RMSE | 0.8725164472 |
| ΔRMSE (nullswap − R) | **+2.278 × 10⁻⁷** |
| ΔMAE (nullswap − R) | +6.13 × 10⁻⁸ |
| mean absolute prediction movement | 2.53 × 10⁻⁶ |
| median absolute movement | 1.81 × 10⁻⁶ |
| fraction moved > 1e−6 | 0.728 |

The swap changes almost nothing. The learned R edge score is effectively
independent of the pairing term `j`; the small R residual behaves like a
bond-feature-only correction. This is consistent with the observed R − A gain
being too small to matter, and it is **not** used to change the pre-registered
decision. It is a mechanism fact, not a causal estimate.

### 11.3 Tail concentration and remove-top-5 sensitivity

- Top-10 A-error molecules account for **85.6%** of the total R − A MSE
  improvement; top-20 accounts for 132.6% (other molecules contribute negative
  improvement).
- Removing the fixed top-5 influential molecules changes mean R − A ΔRMSE from
  **−0.008947** to **−0.001327**; the sign is unchanged but the already-small
  gain nearly vanishes.
- For R − N, removing the same five changes mean ΔRMSE from −0.021187 to
  −0.025970; the sign is unchanged.
- For N − A, removing the same five changes +0.012240 to +0.024643.

The R gain is therefore concentrated in a small number of high-error
molecules, which is a robustness caveat, not a new decision rule.

### 11.4 Bond-count groups

Monitor bond counts are overwhelmingly in the `13+` bin (499/504 molecules);
there are 4 molecules in `9–12`, 1 in `5–8`, and none in `0` or `1–4`. The
`13+` group reproduces the overall direction: R − A = −0.00933, R − N =
−0.01823, N − A = +0.00890. The tiny lower-count bins are descriptive only and
cannot support subgroup claims.

### 11.5 Seed vs molecule variance

Error variance decomposition (not a confidence interval):

| arm | within-molecule seed share | between-molecule share |
|---|---:|---:|
| A | 0.0866 | 0.9134 |
| R | 0.0600 | 0.9400 |
| N | 0.2015 | 0.7985 |

Molecule-to-molecule variation dominates; N has the largest seed share, again
driven by N_s1. Seed uncertainty and molecule-resampling uncertainty are
reported separately and never combined.

---

## 12. Engineering and resource issues

1. **Backend:** CPU was eligible from the measured 0.75/0.79 s per full Fit
   epoch and was used for all six runs. No GPU-hour was consumed. This is
   within the 1.5 A100 GPU-hour cap and the 1200 s per-run wall cap (each run
   was 286–291 s).
2. **Parallel agent isolation:** the Peptides job was on remote GPU0 and
   completed at 10:57:45 before formal training. We never touched
   `/home/hxy/cy/research`, GPU0, its checkpoints, or its deploy directory.
   Our worktree/branch and rr host config are separate; the remote code
   directory for this round was never needed because CPU was chosen.
3. **Cache serialization:** historical torch cache **file** bytes differ from a
   fresh serialization, while all array contents match the frozen content
   fingerprints. The audit records both; no cache was overwritten.
4. **Smoke instability at one update/epoch:** the initial 128-row × 2-epoch
   smoke had one optimizer update per epoch and was not informative; the
   retained 512-row × 12-epoch smoke converges to Fit RMSE 1.05242. Full-Fit
   training converges normally from epoch 1 (MSE 31.87) to epoch 360
   (MSE ~0.0057–0.0063).
5. **Gradient clipping:** 1,661–1,850 clip events per run and pre-clip gradient
   norms up to 5,631 are observed under the frozen clip 5.0 recipe. No
   non-finite event occurred. This is reported, not tuned.
6. **Prediction-writer bug:** an early predict attempt passed the cache payload
   instead of the cache metadata and failed on missing `rows`; it was fixed
   before any prediction was frozen. The fix is in the frozen code commit
   `9887fc38…`. No Monitor label was involved.
7. **Monitor access discipline:** the manifest and lock were written before the
   evaluation process started; the evaluation audit reports
   `label_values_constructed=504`, `skipped=2352`, `unrequested=0`.
8. **Disk:** remote `/home` was at 100% used with 177 GB available; local
   `.venv` is 5.8 GB and this round's cache is ~33 MB. Not a blocker; no remote
   data was written.

No run was retried, resumed, or selected by Monitor performance. No
hyperparameter, architecture, threshold, or recipe change was made after the
freeze.

---

## 13. Artifacts, reproduction, and limitations

### Deliverables

| path | content |
|---|---|
| `PROTOCOL.md` | frozen protocol |
| `configs/protocol.json` | frozen config (hash `2005d1f1…`) |
| `code/` | model, data, train, predict, evaluate, gates, runner |
| `tests/` | 8 pytest checks; all passed |
| `analysis/preflight.json` | protocol freeze, 16 code hashes, parameter audit |
| `analysis/data_audit.json` | split, rows, caches, labels, Fit audit |
| `analysis/gates.json` | G1–G7 all PASS |
| `analysis/resource_isolation.json` | workspace/GPU/Peptides isolation PASS |
| `analysis/parameter_audit.json` | 247,106 + 12,449 = 259,555 per arm/seed |
| `analysis/cpu_timing.json`, `analysis/backend.json` | CPU eligibility and fixed backend |
| `analysis/smoke.json` | 512-row × 12-epoch smoke |
| `analysis/prediction_manifest.json` + `.sha256` | frozen predictions, rows, hashes |
| `analysis/paired_metrics.json` | absolute and paired Monitor metrics, decision |
| `analysis/mechanism_diagnostics.json` | delta/nullswap/tail/bond-count diagnostics |
| `runs/R_s0 … N_s2/` | `train.json`, `curve.csv`, `soup.pt` |
| `results/monitor_predictions/` | six primary + three counterfactual + sidecars |
| `results/monitor_comparison.csv` | arm and comparison table |
| `results/per_molecule_comparison.csv` | 504 rows × predictions/errors/deltas |

Key artifact hashes:

| file | SHA-256 |
|---|---|
| `analysis/preflight.json` | `22af220c57589a62068546e961c10c4ac176faac3f068ef3716b8009b309d378` |
| `analysis/prediction_manifest.json` | `8c1a234ff8c1b9854b8e1d09914aa78b8bd98e36432897774f1d45dabfcfcc8a` |
| `analysis/paired_metrics.json` | `fddd6522c2ce515fcb2ede790743f9e2eaf66835afb275c34c27212825f83ee2` |
| `analysis/mechanism_diagnostics.json` | `80cb244e7d5d24beebcf70b7b506352812efea714d8419f2b6e676e436b93d4e` |
| `results/monitor_comparison.csv` | `1b84179f556d3edc611b8a0e9afb8930451add8914ef0a18e78d286be31c0f27` |
| `results/per_molecule_comparison.csv` | `ea45ef98e50b04f3a9eac225566635b764784e97025f2462810079a9b82a0d64` |

### Reproduction

From the repository root:

```bash
git checkout exp/lcrc-lipo-bond-object-pairing-v1
cd tracks/structural-sufficiency/lcrc_lipo_bond_object_pairing_v1
uv run python code/run.py verify-data
uv run python code/run.py preflight
uv run python code/run.py gates
uv run python code/run.py predict --device cpu   # uses frozen soups
uv run python code/run.py freeze
uv run python code/run.py evaluate               # one label read; do not rerun casually
```

The formal runs can be reproduced with `train --arm R|N --seed 0|1|2 --device
cpu --threads 4` in the fixed order. The frozen soup files are included so
predictions can be checked without retraining.

### Limitations

- Monitor504 is a development split already used by SSRC Phase 1a; it is
  **not** an independent blind test. The label `EXPOSED_MONITOR_EXPLORATORY_ONLY`
  is deliberate.
- Reserved504 and official validation/test labels and structures were never
  read or scored.
- Three seeds quantify training-seed variation only. The molecule bootstrap
  intervals quantify molecule-resampling uncertainty only; the two are not
  combined.
- The R → N swap is a frozen-weight mechanism diagnostic, not a causal estimate
  of the pairing benefit.
- Bond-count subgroup results are dominated by the `13+` bin; lower bins are too
  small for claims.
- The small R − A gain is concentrated in a few high-error molecules and does
  not survive materiality. No follow-up, architecture search, or
  hyperparameter search is authorised by this round.
