# DECISION — LCRC-A Lipophilicity screen v1 (amendment-001, completed)

- **Date:** 2026-10-30
- **Round status:** `FORMAL_GPU_COMPLETE_MONITOR_EVALUATED`
- **Scientific decision:** **`CASE_1_S2_INCREMENT_SUPPORTED`** under the frozen rules.
- **S2 channel:** `CLEAR_INCREMENT` (A vs B, all three seed deltas negative, seed-mean
  bootstrap CI_high < 0, mean ΔRMSE ≤ −0.03).
- **GPU:** 0.5400 h in this stage; 0.0139 h historical; **0.5539 h cumulative** of the
  1.5 h round cap.
- **Monitor/Reserved/official labels:** Monitor504 read exactly once, after the
  prediction freeze. Reserved504 and official valid/test were never read.

## 1. Decision

1. **Execute the frozen protocol under the new authorization.** The amended freeze
   `7fffba42…` was authorized for 1.4861 A100 GPU-hours, deployed, and executed without
   changing the model, data, split, recipe, seeds, metrics or decision rules.
2. **Apply the stage-1 gate before the rest.** `A_s0` completed 360 epochs / 6840
   updates with every per-epoch index fingerprint independently reproduced; only then
   were the other eight runs launched. No performance number was used to change the
   design.
3. **Freeze all 12 predictions before reading any Monitor label.** Nine label-blind
   neural predictions plus the existing three `D_r*` files were hashed and locked
   (`MANIFEST.json` `ba8ad0be…`).
4. **Run exactly one Monitor evaluation** and apply the pre-registered rules. This
   decision is the rule output, not a post-hoc choice.
5. **Stop the round after reporting.** No SSL, AQSOL, extra ablation, Reserved
   evaluation, official valid/test evaluation, retry, tuning or next-round work is
   started.

## 2. Frozen-rule derivation

Monitor504, raw logD units, seed-mean and per-seed values:

| quantity | value |
|---|---:|
| A RMSE mean / seed range | 0.91002 / 0.88066–0.93432 |
| B RMSE mean | 1.04912 |
| C RMSE mean | 0.92284 |
| D (`D_r0`, primary) RMSE | 0.97439 |
| Fit-mean constant RMSE | 1.17645 |
| A vs B mean ΔRMSE | −0.13910 (mean of the three per-seed deltas) |
| A vs B seed-mean-prediction ΔRMSE | −0.14311 |
| A vs B per-seed ΔRMSE | −0.13614, −0.14206, −0.13909 (all < 0) |
| A vs B seed-mean bootstrap CI | [−0.26299, −0.02038], CI_high < 0 |
| A vs C seed-mean ΔRMSE / CI | −0.02510 / [−0.13114, +0.07588] → `INCONCLUSIVE` |
| A vs D seed-mean ΔRMSE / CI | −0.12117 / [−0.25361, +0.04189] → `INCONCLUSIVE` |
| A learned (mean and every seed below constant) | yes |

Rule application:

- `S2`: all three per-seed deltas < 0, seed-mean CI_high < 0, mean Δ ≤ −0.03 →
  **`CLEAR_INCREMENT`**. (The rule uses the mean of the three per-seed deltas for
  materiality and the seed-mean-prediction bootstrap for the CI; both are negative and
  exclude zero at the upper end.)
- `learned`: A mean 0.91002 < 1.17645 and all three seeds below the constant → **yes**.
- `competitive`: neither A-vs-C nor A-vs-D is `A_MATERIALLY_WORSE` (both
  `INCONCLUSIVE`) → **yes**.
- Case: learned ∧ `CLEAR_INCREMENT` ∧ competitive → **`CASE_1_S2_INCREMENT_SUPPORTED`**.

The case is the frozen label. It does **not** mean A beats C or D: both primary RMSE
CIs cross zero. It means A learned, the S2 ablation difference is clear in the
pre-registered direction, and A is not materially worse than the two references.

## 3. Evidence used

| evidence | result |
|---|---|
| nine formal runs | 9/9 completed, 360 epochs / 6840 updates each, 3240/3240 replayed epoch fingerprints matched |
| training sanity | finite loss/gradients; `nonfinite_events=0`; last-5 soup saved and hash-verified |
| label discipline | training/report flags `monitor/reserved/official=false`; access guard recorded no denied attempts |
| prediction freeze | 12/12 files, common 504-row order, hashes locked before label access |
| Monitor audit | 504 labels constructed, 2352 skipped, 0 unrequested; Reserved/official rows 0 |
| `contrib + rho = prediction` | max error ≤ 1e−06 per seed |
| GPU ledger | 1944 s formal, 0.5539 h cumulative (cap 1.5 h) |

## 4. Interpretation boundaries

**Supported under the frozen rules**

- Scratch LCRC-A learns Lipophilicity on Monitor504 (R² ≈ 0.40; all seeds below the
  Fit-mean constant).
- Adding the S2 channel (A vs the S2-off B) is associated with a consistent RMSE
  improvement under the frozen ablation and rule.
- A is competitive with GINE and ECFP4+XGBoost on the primary metric in the weak sense
  of "not materially worse"; its point estimate is best, but the CIs are inconclusive.
- The seed-mean prediction of A (0.85322) is the best point estimate among the four
  arms.

**Not supported / must not be claimed**

- That A is proven better than GINE or ECFP4+XGBoost — the primary RMSE CIs cross zero.
- That the S2 difference is an isolated information effect — B has only 176,514
  effective parameters versus A's 247,106, and its Fit error is much larger
  (0.213–0.223 vs 0.096–0.099). Capacity and channel are confounded.
- Any causal or chemical reading of `contrib`/`rho`; the decomposition is descriptive
  only, and `rho` merely carries most of the level while `contrib` carries most of the
  variance.
- A fresh-blind-test result: Monitor504 is a development set previously used by SSRC.
- Anything about Reserved504 or official valid/test: those labels were never read.

## 5. Budget

| item | GPU hours |
|---|---:|
| historical v1 halt | 0.0139 |
| this stage (nine runs) | 0.5400 |
| predictions / freeze / evaluation (local CPU) | 0.0000 |
| **cumulative** | **0.5539** |
| remaining of 1.5 h round cap | 0.9461 |
| authorized stage cap | 1.4861 |
| stage headroom unused | 0.9461 |

No budget shortfall occurred; no job was launched beyond the authorization.

## 6. Provenance and preservation

- Freeze: `analysis/protocol_freeze.json` `7fffba42…` (unchanged by this stage).
- Active authorization: `analysis/gpu_authorization.json` `a2183bdf…`; it pins the
  freeze, allows only the nine runs and records the 0.0139 h prior use.
- Superseded v1 authorization: `analysis/gpu_authorization_v1_superseded_pre_amendment.json`
  `48cb9a28…`, byte-identical to the v1 record at commit `4d8f136`; hashes/commits are in
  `analysis/gpu_authorization_history.json`.
- v1 halt record `analysis/gpu_stage_halt.json` `111275dc…` is preserved and still
  describes why v1 stopped.
- Prediction freeze `analysis/predictions/MANIFEST.json` `ba8ad0be…` + lock
  `f083600e…`; evaluation `analysis/monitor_metrics.json` `f9df66ab…`,
  `monitor_paired.json` `9c0e4631…`, `monitor_diagnostics.json` `41d6b860…`.
- GPU and per-run provenance: `analysis/gpu_ledger.json` `2eca7b14…` (per-run rr ids,
  wall seconds, train/soup hashes, batch-order hashes, fit diagnostics).

## 7. Invalidation conditions

This decision is invalidated if any of the following is changed or reinterpreted
without a new recorded amendment, a regenerated freeze and a new authorization:

- recipe, seeds, epochs/updates, optimizer, checkpoint rule, split, FeatureSpec, caches,
  target standardisation, arms, metrics, bootstrap, materiality or decision rules;
- prediction files or manifest hashes;
- the interpretation boundaries in §4 (especially treating A-vs-B as an
  information-matched S2 effect, or treating an `INCONCLUSIVE` comparison as a win).

Also invalid: citing the pre-amendment smoke as training evidence, reusing the v1
authorization, or reading Reserved/official labels to "confirm" this result.

## 8. Stop

The authorized scope is complete. The round stops here. Do **not** start SSL, AQSOL,
another dataset, extra ablations, hyperparameter search, seed/checkpoint selection,
Reserved evaluation, official valid/test evaluation, or a follow-up round from this
decision. Any future question needs a new pre-registration and a new authorization.
