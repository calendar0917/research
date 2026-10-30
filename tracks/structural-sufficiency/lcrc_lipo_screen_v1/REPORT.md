# REPORT — LCRC-A Lipophilicity screen v1 (amendment-001, executed)

- **Round:** `tracks/structural-sufficiency/lcrc_lipo_screen_v1/`
- **Final status:** `FORMAL_GPU_COMPLETE_MONITOR_EVALUATED`
- **Scientific result:** **`CASE_1_S2_INCREMENT_SUPPORTED`** under the frozen rules
  (`S2 = CLEAR_INCREMENT`; A learned the task; A not materially worse than C or D).
- **Protocol freeze:** `analysis/protocol_freeze.json`
  sha256 `7fffba423df449fae09688f29e1d4d33cdd5df2e51c4a15ae5986031470818b8`
- **Authorization:** `analysis/gpu_authorization.json` sha256
  `a2183bdf62f61041b844791f97f1be3d848e0d3c88ad9d27c0c99b820c291dcc`
  (stage cap 1.4861 A100 GPU-h; nine runs only).
- **Execution regime:** `res`, node `a100-2`, NVIDIA A100-SXM4-40GB,
  driver 550.163.01, torch 2.5.1+cu124, Python 3.12.14, commit `a871657`.
- **GPU ledger:** historical 0.0139 h + formal 0.5400 h = **0.5539 h** of the 1.5 h
  round cap (0.9461 h remaining). Detail: `analysis/gpu_ledger.json`.
- **Prediction freeze:** `analysis/predictions/MANIFEST.json` sha256
  `ba8ad0be8c1e01395aa03e619f9fc4d0552d33ea8dabfc8370cc43c4a79ee161`.
- **Monitor evaluation:** ran **exactly once** after the freeze
  (`2026-10-30T16:54:57Z`).
- **Reserved / official valid+test labels:** never read, never scored.
- **Amendment record:** [`AMENDMENT-001-fit-batch-index.md`](AMENDMENT-001-fit-batch-index.md)
- **CPU gates at re-authorization:** G0–G9 all PASS (`analysis/cpu_gates.json`
  sha256 `98d8e27c…`); 59 pytest checks pass.

This report supersedes the pre-execution "no performance result" version. The halted
v1 record and the superseded v1 authorization remain preserved (§9).

---

## 1. Execution order actually followed

| step | action | result |
|---|---|---|
| 1 | preservation of the old authorization bytes | archived byte-identically, sha256 unchanged (`48cb9a28…`) before the live path was rewritten |
| 2 | new authorization pinned to `7fffba42…` | accepted; all nine run keys accepted, `D_r*`/unknown keys rejected |
| 3 | CPU gates G0–G9 under the new authorization | 10/10 PASS; `real_authorization_matches_current_freeze=true` |
| 4 | `rr deploy res` (res-local, res-gpu0, res-gpu1) | deployed commit `a871657` |
| 5 | remote preflight (CPU) | freeze, authorization, Fit/Monitor/Reserved split and caches verified; no formal run present; CUDA available; reserved cache not loaded |
| 6 | stage 1: `A_s0` | 360/360 epochs, 6840/6840 updates, verified, then stage 2 allowed |
| 7 | stage 2: remaining eight runs | all completed with the frozen recipe |
| 8 | nine label-blind Monitor predictions (local CPU) | written and checked against `D_r0..D_r2` |
| 9 | `freeze-predictions` | 12 files hashed and locked |
| 10 | one `evaluate` | Monitor504 labels read once; frozen metrics and case written |

No formal run was retried, cancelled or resumed. No code/config changed after the
amended freeze. The nine runs all executed the committed revision `a871657`, which has
the frozen code hashes.

## 2. Authorization provenance (old record preserved)

The old v1 authorization was **not** overwritten in place. Its original bytes were
copied to

```
analysis/gpu_authorization_v1_superseded_pre_amendment.json
sha256 48cb9a28ff36ec6765bcbc05d72391f5018e123a1ae8cef11d89196e44a3f30f  (byte-identical)
```

before the live path `analysis/gpu_authorization.json` was replaced. The historical
bytes and hash are also retrievable from Git commit `4d8f136`
(`git show 4d8f136:…/analysis/gpu_authorization.json`). A provenance index records both
records, hashes, commits and the supersession reason:
`analysis/gpu_authorization_history.json` (sha256 `7ff9dd44…`).

| record | freeze pinned | cap | runs | sha256 | Git |
|---|---|---:|---|---|---|
| v1 (superseded, preserved) | `87ec4751…` | 1.5 | 9 | `48cb9a28…` | `4d8f136` |
| amendment-001 (active) | `7fffba42…` | 1.4861 | 9 | `a2183bdf…` | `854e36c` |

The new record explicitly states the 0.0139 h already consumed, the 1.4861 h stage cap
and the 1.5 h cumulative cap. The frozen code/config/data hashes did not change during
the stage; every formal run re-verified them with `require_clean=True`.

## 3. Engineering correctness evidence

### 3.1 Stage-1 gate (`A_s0`)

`A_s0` completed **360 epochs / 19 batches / 6840 updates** in 191.3 s of training
(209 s rr wall-clock). Independent local replay of the shuffle stream (`fit_epoch_plans`)
reproduced all **360 per-epoch** position/global-id/target fingerprints with **0
mismatches**, 2352 unique in-range cache positions and 19 batches per epoch, last batch
48. `batch_order.actual_hash == independent_recomputation ==
1aeefe0f9276b256…`. Loss and gradient norms were finite at every step
(`nonfinite_events=0`), the last-5 soup (epochs 356–360) was saved and its file hash
equals the recorded `soup_state_hash`, and Fit-only diagnostics were produced. The run
was then accepted as the stage-1 gate; training performance was **not** used to alter
the design.

### 3.2 All nine runs

Every run was verified locally after `rr pull`:

- 360 epochs, 19 steps/epoch, 6840 updates;
- `batch_order.matches=true`, and every one of the 360 epochs was independently
  recomputed from the frozen shuffle offset: **0 fingerprint mismatches in all 3240
  epochs** (9 runs × 360);
- every epoch covers all 2352 Fit cache positions exactly once, all in `0..2351`;
- soup = elementwise mean of epochs 356–360; `soup.pt` hash matches `soup_state_hash`;
- all losses, gradient norms and optimizer steps finite (`nonfinite_events=0`);
- `monitor_labels_read=false`, `reserved_labels_read=false`,
  `official_valid_test_read=false`; the runtime deny-list recorded **no denied
  attempts**;
- A and B share the identical init state hash for each seed (seeds 0/1/2).

| run | GPU | wall s | fit soup RMSE (Fit, raw) | grad-norm max | clip events | soup hash |
|---|---|---:|---:|---:|---:|---|
| A_s0 | gpu0 | 209 | 0.09583 | 5253.3 | 1869 | `aefd6d54…` |
| A_s1 | gpu0 | 210 | 0.09743 | 5628.0 | 1675 | `da4895c7…` |
| A_s2 | gpu1 | 281 | 0.09856 | 5338.6 | 1665 | `adf0d95b…` |
| B_s0 | gpu0 | 215 | 0.21321 | 4495.3 | 2612 | `d23f87e5…` |
| B_s1 | gpu1 | 268 | 0.21975 | 3732.9 | 1915 | `ff0f4d02…` |
| B_s2 | gpu0 | 219 | 0.22263 | 4440.8 | 2311 | `30a63f06…` |
| C_s0 | gpu1 | 188 | 0.11951 | 11075.9 | 3634 | `17389b8c…` |
| C_s1 | gpu0 | 165 | 0.11793 | 10657.5 | 3924 | `9ae21e09…` |
| C_s2 | gpu1 | 189 | 0.11448 | 12542.4 | 3679 | `69cb5d4e…` |

The large pre-clip gradient norms and frequent clipping are part of the frozen recipe
(clip 5.0); they are reported as observed, not tuned. `A_s2` and `B_s1` ran slower
because they overlapped other jobs on the host; this does not change training semantics.

### 3.3 Frozen split and caches

The remote preflight recomputed the frozen artifacts: Fit/Monitor/Reserved sizes
2352/504/504, row hashes `c474e38c…` / `fe81cc79…` / `bc28c5d3…`, cache content
fingerprints `0a73407f…` (Fit) and `db48a9c2…` (Monitor), `labels_present=false`,
Reserved cache never loaded. The full-Fit alignment gate G9 is part of the passing
G0–G9 record.

### 3.4 Post-evaluation test-suite caveat (no effect on the frozen result)

Before deployment the full suite passed **59/59** at commit `a871657`, including the
six checks in `tests/test_fit_alignment.py`; that file also passes when run alone after
the stage. Four checks in the frozen suite assert the *pre-execution* state
(`MANIFEST` absent, no training artifacts), so they fail once the stage exists and must
not be re-run after the freeze:

- `test_evaluate_is_blocked_before_prediction_freeze`
- `test_monitor_access_blocked_before_freeze`
- `test_predict_nn_refuses_without_a_training_run`
- `test_freeze_refuses_an_incomplete_registry`

Running that suite once post-evaluation exercised `predict_nn("A", 0)`, which rewrote
`analysis/predictions/A_s0.{npz,json}`. The frozen bytes were restored from commit
`46d0ab1` (`git checkout --`), and all 12 files were re-verified against the manifest
(`A_s0.npz` `3b40979b0686…`, sidecar `a1355f8d…`). The single Monitor evaluation and
all reported metrics had already completed before this happened, so nothing in §6
changed. The tests were **not** modified: they are part of the frozen code hashes. The
correct post-execution integrity check is the manifest verification, not this
pre-execution test suite.

## 4. GPU ledger

GPU seconds are rr wall-clock seconds of GPU jobs, rounded up to whole seconds; they
include interpreter start and cache build and are an upper bound on kernel time.
Per-run hashes, timestamps and provenance are in `analysis/gpu_ledger.json`
(sha256 `2eca7b14…`).

| item | GPU seconds | GPU hours |
|---|---:|---:|
| historical v1 halt (3 jobs on `res`) | 50 | 0.0139 |
| this stage, nine formal runs | 1944 | 0.5400 |
| preflight jobs (CPU pool / 0 GPU) | 0 | 0.0000 |
| nine predictions, freeze, evaluation (local CPU) | 0 | 0.0000 |
| **cumulative** | **1994** | **0.5539** |
| round cap | — | 1.5 |
| remaining | — | **0.9461** |
| stage cap / stage used | — | 1.4861 / 0.5400 |

The stage stayed well inside the authorized 1.4861 h and the cumulative 1.5 h caps.

## 5. Prediction freeze (12 files, label-blind, shared row order)

All nine neural predictions were produced from the pulled soups **after** all training,
without opening any Monitor label; the three `D_r*` files were produced before the
freeze and are unchanged. Every file has 504 rows in the frozen Monitor order
(`rows_sha256 = a31ab4d07d4f81e5c25bffa71e6a696fb620d0c1a46cf17a4ea1b255b1539c24`),
the same config hash `b724a521…`, model identity and `monitor_labels_read=false`.

| key | model | parameters | prediction sha256 |
|---|---|---:|---|
| A_s0 | lcrc_a_lipo_v1 | 247,106 | `3b40979b06861cde7644481d0818326190a8ff9fd66bee43c7d44544365a5e03` |
| A_s1 | lcrc_a_lipo_v1 | 247,106 | `49ca4b5b00115b4daef42a126558842bc45583d4f9403b432e427440bbf20d92` |
| A_s2 | lcrc_a_lipo_v1 | 247,106 | `6c1360979ceecd3f023f1a2362b75bf858210d7feb889484d93984e6346859bb` |
| B_s0 | lcrc_a_lipo_v1 | 247,106 | `8b00bb16924821fd1096bab206a5537e999e7624d40a9ed9ef76dfdd02c14246` |
| B_s1 | lcrc_a_lipo_v1 | 247,106 | `84fe4dfdd8332beb66d2d230c8f56865a95ea1ad397b4a0ac4725668a3e91930` |
| B_s2 | lcrc_a_lipo_v1 | 247,106 | `c6f05298996966c9e22f9d7aacec5567ca1512e1cb7d2e888225173a50ca2c00` |
| C_s0 | gine_lipo_v1 | 245,305 | `fbbda6786abdcf2a4ddab6283429da253ba93f8d52cb11c33f113485211c44be` |
| C_s1 | gine_lipo_v1 | 245,305 | `003407c8c5510eba306e0ce49f11a5415af0cafbd30c350c0a52f62369a2aa45` |
| C_s2 | gine_lipo_v1 | 245,305 | `c5d4b15b5e3cd2942afe4b83629a51243e3c2b933d36021feea94da11fac4b5e` |
| D_r0 | ecfp4_xgb | — | `e2a18b9be03ed9e1da96614e9337abf2da577e14079d1a43958d295f736696b2` |
| D_r1 | ecfp4_xgb | — | `d2b42ebcc4a024d7e0dafa12e2c7672eb2650797361ba19a4e1704dc2c9be6c1` |
| D_r2 | ecfp4_xgb | — | `af4831f7dada243cde31f098e23dcfda4edc02e4582929dbd36ced9e49f46b78` |

The freeze manifest (`ba8ad0be…`) records the freeze sha, config hash, code hashes,
recipe, row order and every prediction hash; the lock file is
`analysis/predictions/MANIFEST.sha256` (`f083600e…`). Evaluation refuses to start
without a valid manifest + lock and verifies every hash again.

## 6. Monitor504 results (frozen rules, raw logD units)

Monitor504 is a **development set that SSRC used before**; it is not a fresh blind test.
Only the 504 Monitor labels were constructed by the evaluation audit
(`label_values_constructed=504`, `label_values_skipped=2352`,
`unrequested_values_constructed=0`, Reserved/official rows in artifact = 0).

### 6.1 Absolute metrics per seed

| arm | seed | RMSE | MAE | R² | bias |
|---|---:|---:|---:|---:|---:|
| A | 0 | 0.93432 | 0.65733 | 0.3668 | −0.0065 |
| A | 1 | 0.88066 | 0.63062 | 0.4375 | +0.0128 |
| A | 2 | 0.91509 | 0.65372 | 0.3926 | −0.0322 |
| B | 0 | 1.07046 | 0.70750 | 0.1689 | −0.0521 |
| B | 1 | 1.02273 | 0.68433 | 0.2413 | −0.0568 |
| B | 2 | 1.05418 | 0.70522 | 0.1940 | −0.0228 |
| C | 0 | 0.91542 | 0.63738 | 0.3922 | +0.0279 |
| C | 1 | 0.93372 | 0.60806 | 0.3676 | −0.0374 |
| C | 2 | 0.91938 | 0.61690 | 0.3869 | −0.0419 |
| D | r0 | 0.97439 | 0.77071 | 0.3114 | −0.0099 |
| D | r1 | 0.98537 | 0.77621 | 0.2958 | −0.0144 |
| D | r2 | 0.97217 | 0.76765 | 0.3145 | −0.0174 |

Reference: the Fit-mean constant predictor has RMSE **1.17645**, MAE 0.96520,
R² −0.0039. All three A seeds are below it, so A learned the task by the frozen
definition.

### 6.2 Seed summary (training-seed uncertainty)

| arm | RMSE mean | SD | min | max | spread | MAE mean | R² mean | seed-mean-pred RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0.91002 | 0.02220 | 0.88066 | 0.93432 | 0.05366 | 0.64723 | 0.3990 | **0.85322** |
| B | 1.04912 | 0.01981 | 1.02273 | 1.07046 | 0.04773 | 0.69902 | 0.2014 | 0.99633 |
| C | 0.92284 | 0.00786 | 0.91542 | 0.93372 | 0.01831 | 0.62078 | 0.3823 | 0.87832 |
| D | 0.97731 (3 states) | 0.00577 | 0.97217 | 0.98537 | 0.01320 | 0.77152 | 0.3072 | — |

A's seed spread (0.0537) exceeds the 0.03 materiality threshold, so the report flags
`seed_sensitive=true`; its best seed (0.88066) is not substituted for the mean. Averaging
the three A seeds improves RMSE to 0.85322, i.e. seed noise partly cancels.

### 6.3 Paired comparisons (molecule bootstrap, 2000 resamples, seed 0)

`Δ = metric(A) − metric(baseline)`; negative favours A.

| comparison | ΔRMSE | 95% CI | ΔMAE | 95% CI | frozen state |
|---|---:|---|---:|---|---|
| A vs B (seed mean) | **−0.14311** | [−0.26299, −0.02038] | −0.05114 | [−0.09446, −0.00990] | S2 rule below |
| A vs C (seed mean) | −0.02510 | [−0.13114, +0.07588] | +0.02109 | [−0.01621, +0.05791] | `INCONCLUSIVE` |
| A vs D (`D_r0`) | −0.12117 | [−0.25361, +0.04189] | **−0.16692** | [−0.22570, −0.10732] | `INCONCLUSIVE` |
| A vs constant | −0.32323 | [−0.44643, −0.17267] | −0.36140 | [−0.42999, −0.29576] | learned |
| B vs C | +0.11801 | [+0.05069, +0.20554] | +0.07224 | [+0.03596, +0.11375] | B higher, CI excludes zero |

Interpretation of the primary metric (RMSE): A's point estimate is best among the four
arms, but only the A-vs-B difference has a CI excluding zero. A vs C and A vs D are
**inconclusive** at the frozen 0.03 materiality rule because their CIs cross zero. A's
MAE versus D is clearly lower (CI excludes zero), but MAE is secondary; the primary
decision therefore does **not** claim A beats D.

### 6.4 S2 increment (A vs B)

| seed | ΔRMSE | 95% CI | ΔMAE |
|---|---:|---|---:|
| 0 | −0.13614 | [−0.23361, −0.02892] | −0.05017 |
| 1 | −0.14206 | [−0.24080, −0.04182] | −0.05371 |
| 2 | −0.13909 | [−0.32386, +0.03150] | −0.05150 |
| per-seed mean | **−0.13910** | — | −0.05179 |
| seed-mean prediction | −0.14311 | [−0.26299, −0.02038] | −0.05114 |

The seed-mean-prediction Δ (−0.14311) is not identical to the mean of the per-seed
RMSE differences (−0.13910) because averaging three predictions before scoring is not
the same operation as averaging three RMSEs. The frozen `S2` rule uses the three
per-seed signs and the seed-mean-prediction CI.

Frozen `S2` rule: all three per-seed Δ < 0 (yes), seed-mean CI_high < 0 (yes),
mean Δ ≤ −0.03 (yes) → **`CLEAR_INCREMENT`**. The seed-2 CI alone crosses zero; the
frozen rule uses the three signs plus the seed-mean CI, and both hold.

**Confound that must travel with this result:** B has the same 247,106 nominal
parameters but only **176,514 effective parameters** (it zeroes the 200-d S2 block
`[320:520)` in train and eval). A-vs-B is therefore an S2-channel ablation with unequal
effective capacity, not an information-matched contrast. B's much worse Fit RMSE
(0.213–0.223 vs 0.096–0.099) shows the ablation also changes optimisation/fit capacity.
The measured increment is real for this pair of models; it cannot be attributed solely
to "the S2 information".

### 6.5 Round case

Frozen rule application:

- A learned: mean RMSE 0.91002 < constant 1.17645 and each seed below it → yes.
- S2: `CLEAR_INCREMENT` → yes.
- Not materially worse than C or D: A-vs-C and A-vs-D are `INCONCLUSIVE`, not
  `A_MATERIALLY_WORSE` → yes.

→ **`CASE_1_S2_INCREMENT_SUPPORTED`**.

This is the rule's label, not a claim that A dominates C or D. The case explicitly
allows A to be statistically indistinguishable from C/D.

### 6.6 Seed vs molecule variation and tails

| arm | seed share of squared error | molecule share | mean difficulty rank corr. |
|---|---:|---:|---:|
| A | 0.0866 | 0.9134 | 0.531 |
| B | 0.0261 | 0.9739 | 0.558 |
| C | 0.0604 | 0.9396 | 0.608 |

Most error variation is between molecules, not between seeds. Tail diagnostics:

| arm | top 1% error share | top 2% | top 5% | RMSE excl. top 2% | RMSE excl. top 5% |
|---|---:|---:|---:|---:|---:|
| A | 0.240 | 0.297 | 0.417 | 0.77067 | 0.71280 |
| B | 0.365 | 0.417 | 0.517 | 0.80953 | 0.74780 |
| C | 0.333 | 0.393 | 0.497 | 0.72601 | 0.67133 |
| D_r0 | 0.108 | 0.165 | 0.301 | 0.89909 | 0.83537 |

A has the least tail concentration among the neural arms, just under the frozen
`tail_sensitive` threshold (top-2% share 0.297 ≤ 0.30); no A-vs-B or A-vs-C sign flip
appears after dropping the worst 2% or 5%. C's advantage after removing the largest
errors is descriptive only and does not change the RMSE decision rule.

### 6.7 `contrib` / `rho` diagnostics (arm A, descriptive)

| seed | `contrib` mean | `rho` mean | max decomposition error | median \|rho\|/\|contrib\| | rho dominates | variance share contrib / rho |
|---|---:|---:|---:|---:|---:|---|
| 0 | 0.1739 | 2.0504 | 5.8e−07 | 3.33 | 95.0% | 0.671 / 0.329 |
| 1 | 0.3983 | 1.8453 | 2.9e−07 | 2.51 | 91.1% | 0.742 / 0.258 |
| 2 | 0.2097 | 1.9890 | 2.9e−07 | 3.13 | 93.7% | 0.614 / 0.386 |

`contrib + rho = prediction` holds to ≤1e−06 for every seed. `rho` carries the level
(near-constant positive offset), while `contrib` carries most of the prediction
variance. These are magnitudes of a frozen additive decomposition, **not** chemical
causal attribution or a faithfulness claim.

### 6.8 Train/Monitor gap

Using the Fit-only soup RMSE (in-sample training rows) and the Monitor per-seed RMSE:

| arm | Fit soup RMSE (mean of seeds) | Monitor RMSE (mean) | gap |
|---|---:|---:|---:|
| A | 0.09727 | 0.91002 | **0.81275** |
| B | 0.21853 | 1.04912 | 0.83059 |
| C | 0.11731 | 0.92284 | 0.80553 |

Per-seed gaps are 0.783–0.839 (A), 0.803–0.857 (B), 0.796–0.816 (C). The near-zero Fit
RMSE confirms heavy in-sample fitting; the Monitor gap mixes overfitting with the
Fit→Monitor distribution shift on only 504 molecules and is not a clean
generalisation estimate.

## 7. What the round does and does not establish

**Established (frozen rules, Monitor504):**

1. Scratch LCRC-A learns Lipophilicity better than the Fit-mean constant on all three
   seeds (mean 0.91002 vs 1.17645; R² ≈ 0.40).
2. A-vs-B S2 ablation shows a consistent A advantage (lower RMSE) under the frozen rule
   (`CLEAR_INCREMENT`), with all three per-seed ΔRMSE negative and the seed-mean CI
   excluding zero.
3. A is not materially worse than GINE or ECFP4+XGBoost on the primary metric, and its
   cross-seed mean prediction is the best point estimate (0.85322).
4. Seed averaging helps A (0.91002 per-seed mean → 0.85322 seed-mean prediction).
5. `contrib` and `rho` are both active with a machine-precision additive identity.

**Not established:**

- A is **not** shown to be better than GINE or ECFP4+XGBoost: both primary RMSE CIs
  cross zero (`INCONCLUSIVE`).
- The S2 increment is **not** information-matched: B has lower effective capacity, so
  capacity and channel are entangled.
- No mechanism claim follows from `contrib`/`rho` magnitudes, and no chemical
  interpretation is licensed.
- Monitor504 is a reused development set, not a fresh blind test; the result is a
  development-set result.
- Reserved504 and official valid/test remain unread; nothing here is a confirmation on
  unseen data.
- No SSL, AQSOL, extra ablation, tuning, seed selection or checkpoint selection was run
  or is implied.

## 8. Failed / non-training jobs

| job | status | GPU | record |
|---|---|---:|---|
| v1 `lcrc-timing-a` | completed, timing only | 17 s | `analysis/gpu_stage_halt.json` |
| v1 `lcrc-formal-A-s0` | failed, `ValueError: row index outside cache` | 16 s | preserved, commit `e09c627` |
| v1 `lcrc-formal-A-s1` | failed, same defect | 17 s | preserved |
| v1 `res-2` submissions (3) | cancelled pending, 0 GPU-s | 0 | preserved |
| `lcrc-preflight-amd001` | failed exit 2, local launcher quoting mistake, no training code ran | 0 | recorded in `gpu_ledger.json` |
| `lcrc-preflight-amd001b` | completed, all preflight checks passed | 0 | recorded |

No job in the amended formal stage failed, was cancelled or was retried; 9/9 completed.

## 9. Historical records preserved

- v1 failure ledger: `analysis/gpu_stage_halt.json` (sha256 `111275dc…`) unchanged.
- v1 authorization: archived byte-identically (`gpu_authorization_v1_superseded_pre_amendment.json`,
  `48cb9a28…`), provenance in `analysis/gpu_authorization_history.json`, Git commit
  `4d8f136`.
- Pre-amendment smoke: `analysis/smoke_invalidated_pre_amendment.json` unchanged,
  still marked invalid.
- The amended freeze `7fffba42…` was not edited; actual GPU usage is recorded in the
  separate `analysis/gpu_ledger.json` because the freeze's `gpu_hours_used` is a
  freeze-time value.

## 10. Artifact index (this stage)

| artifact | sha256 |
|---|---|
| `analysis/protocol_freeze.json` | `7fffba423df449fae09688f29e1d4d33cdd5df2e51c4a15ae5986031470818b8` |
| `analysis/gpu_authorization.json` (active) | `a2183bdf62f61041b844791f97f1be3d848e0d3c88ad9d27c0c99b820c291dcc` |
| `analysis/gpu_authorization_v1_superseded_pre_amendment.json` | `48cb9a28ff36ec6765bcbc05d72391f5018e123a1ae8cef11d89196e44a3f30f` |
| `analysis/gpu_authorization_history.json` | `7ff9dd444e596c41d7115e8bc6fa61b7789996eb0bbc2104d02783e3d7b89023` |
| `analysis/cpu_gates.json` | `98d8e27cd547f0a91c0a83e82854506a69c8e2162df7cd4d81c7de29da3a176c` |
| `analysis/gpu_ledger.json` | `2eca7b148e90795e74432116742abd05618e93ed5f6191aac0495348caa57268` |
| `analysis/predictions/MANIFEST.json` | `ba8ad0be8c1e01395aa03e619f9fc4d0552d33ea8dabfc8370cc43c4a79ee161` |
| `analysis/predictions/MANIFEST.sha256` | `f083600e7b69b29a41208e9e3f9a18f117fe45a05ec36d0c582e62dd8ae757fd` |
| `analysis/monitor_metrics.json` | `f9df66abe7c93cdb01b65d9601f2362f01a37763f0f0bf1aa515ec600fa65779` |
| `analysis/monitor_paired.json` | `9c0e4631df52b3d133654678ed41b7edfbf15eff5d118fd2dd78bfe4157ff528` |
| `analysis/monitor_diagnostics.json` | `41d6b860aa64f853fd9c2ffd23d7399da910c95214aeb373896141c42765e90f` |
| `analysis/smoke.json` (pre-execution re-run) | `b845637b…` |
| `analysis/gpu_stage_halt.json` | `111275dcc2037f19526fcb2c041a3e2e2868955e66755c065d52301fad72b113` |

Training artifacts (`analysis/runs/<key>/train.json`, `soup.pt`) are git-ignored by the
round's `.gitignore`; their per-run hashes, soup hashes and rr provenance are durably
recorded in `analysis/gpu_ledger.json`, and the prediction manifest binds every
prediction to its soup and training manifest hash.

## 11. Reproduction

```bash
cd tracks/structural-sufficiency/lcrc_lipo_screen_v1

# pre-execution checks (CPU)
uv run --project /home/calendar/code/research --group lipo python code/run_screen.py cpu-gates
uv run --project /home/calendar/code/research --group lipo python -m pytest tests -q
uv run --project /home/calendar/code/research --group lipo python code/run_screen.py status

# remote formal runs (from the repo root, one example)
rr run res lcrc-formal-A-s0 --pool res-gpu0 --time 0:15:00 \
  --result tracks/structural-sufficiency/lcrc_lipo_screen_v1/analysis/runs/A_s0 \
  -- python tracks/structural-sufficiency/lcrc_lipo_screen_v1/code/run_screen.py train --arm A --seed 0 --device cuda

# label-blind predictions, freeze, one evaluation
uv run --project /home/calendar/code/research --group lipo python code/run_screen.py predict-nn --arm A --seed 0 --device cpu
uv run --project /home/calendar/code/research --group lipo python code/run_screen.py freeze-predictions
uv run --project /home/calendar/code/research --group lipo python code/run_screen.py evaluate
```

Expected frozen result: `CASE_1_S2_INCREMENT_SUPPORTED`, `S2=CLEAR_INCREMENT`, manifest
sha `ba8ad0be…`.

## 12. Invalidation and stop conditions

This report is invalidated if any of the following is changed/claimed without a new
recorded amendment and a new authorization: recipe, seeds, split, FeatureSpec, caches,
arms, metrics, materiality, bootstrap, decision rules, prediction hashes, or the
interpretation boundaries in §6–§7. Do not cite the invalidated pre-amendment smoke as
training evidence, do not reuse the v1 authorization, and do not present the A-vs-B
result as an information-matched S2 effect.

After this report the round **stops**. SSL, AQSOL, extra ablations, Reserved evaluation,
official valid/test evaluation and any further training are not authorized.
