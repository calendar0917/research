# Compact-v5: Uncertainty-Aware Multi-Quantile Regression (ZINC-12k, frozen compact-v4-hinge)

**Status:** FINAL — 2026-09-10
**Phase:** objective-only stage (representation frozen at compact-v4-hinge)
**Protocol:** `zinc-context-gap`; primary benchmark protocol = train → validation
selection → frozen checkpoint → single test evaluation
**Decision:** **NO-GO for a confirmed multi-quantile benchmark improvement.**
Stage-1 seed-0 reached Case A (Strong-GO point + GO uncertainty), but the frozen
4-seed confirmation landed in the **Mild/Mild** band (valid q50 mean +0.0037,
3/4 seeds; width-error Spearman mean 0.211, 4/4 seeds) and the one-time
benchmark test did **not** transfer (test mean 0.139403 ± 0.002510 vs v4
0.136885 ± 0.005552; paired −0.0025, 1/4 seeds). The remaining test is a frozen
observation only — never used for tuning.
**Controls:** `quantile_mode=none` is bit-identical to the canonical
compact-v4-hinge seed-0 run (60/60 epochs, 0.17006561887910357 @53); the
`median_only` control (scalar head + `2*pinball_0.5`) is bit-identical to the L1
run. The auxiliary-quantile weight is pre-registered to `{0.10, 0.25, 0.50}`.

---

## 1. Motivation (from post-v4 + OOF difficulty audit)

The representation direction was closed by two diagnostic stages:

* **Post-v4 residual audit** (validation-only): *NO CLEAR SECONDARY STRUCTURAL
  SIGNAL* — no fitted representation probe cleared the Weak 0.003 ΔMAE band
  (best positive ΔMAE ≈ +0.00007). Residual structure is variance-side only:
  low-`z_SA` molecules and rare-patch molecules are harder (heteroscedastic
  difficulty), with no correctable signed bias.
* **OOF difficulty / heteroscedasticity confirmation** (official train, K=5 OOF,
  frozen v4-hinge): difficulty **GO** — cross-fitted model-visible Ridge
  Spearman **0.374** (5/5 folds) with a 3.67× hardest/easiest predicted-quintile
  MAE ratio and a monotone rarity ladder (0.133 → 0.417 MAE); epistemic-like
  **GO** — 2-seed disagreement vs |error| Spearman **0.393** (5/5), ensemble gain
  ≈ 8.3 %. Signed residuals stayed weak.

Together: remaining error contains stable, input-dependent **predictive
difficulty / conditional variance** and a substantial **epistemic-like**
component, but **no new representation channel** is warranted. This stage
therefore changes only the **regression output / objective** and asks:

> (i) does auxiliary conditional-quantile supervision improve the q50 point
> prediction?  (ii) does the learned interval width internalize the previously
> confirmed OOF difficulty signal in a *single* model?

## 2. Frozen v4 representation (proof only output/objective changed)

Untouched: radius-2 patch construction, exact patch tokenization, hybrid
embeddings, shell descriptor, patch encoder, pair/relation encoder, pooling,
existing global features, topology hinge features, topology encoder, local–global
fusion, hidden dimensions, message passing. The only model change is the **final
head `Linear` width** (`1 → 3` raw outputs) in the three-quantile mode; the
`none` / `median_only` modes keep the byte-identical v4 head.

Invariants held (checked in `tests/test_zinc_patch_path_quantile_v5.py`): no
capacity change beyond the `1→3` final Linear (+66); **no uncertainty weighting**
(the loss is a plain sum of pinball terms, no learned/derived uncertainty is used
to reweight samples or terms); and **the auxiliary quantiles q10/q90 are never fed
back as model inputs** — they exist only in the loss and in the readout.

| object | change |
|---|---|
| `PatchPathModel` head (mq) | `Linear(head_hidden_1, 1) → Linear(head_hidden_1, 3)`; `+2*32+2 = +66` params |
| everything else | unchanged |
| v4 total params | 99,613 |
| v5 mq total params | 99,679 (Δ +66, still ≈ 0.07 % of the ~100k budget) |
| v5 median-only / none params | 99,613 (unchanged) |

## 3. Quantile formulation (non-crossing)

From the last unified graph representation `z_G` the head predicts three raw
outputs `[m, d_low_raw, d_high_raw]` and forward decodes

```
d_low  = softplus(d_low_raw)
d_high = softplus(d_high_raw)
q50 = m
q10 = m - d_low
q90 = m + d_high
```

Because `softplus(·) > 0`, `q10 <= q50 <= q90` holds by construction (no crossing
penalty). `q50` is the **only** point prediction used for benchmark MAE and for
validation selection.

## 4. Loss scaling (`2*pinball_0.5 == L1`)

With `e = y - q_tau`, `L_tau(e) = max(tau*e, (tau-1)*e)`. At `tau = 0.5` the two
branches are `±0.5*e`, so `2 * L_0.5(e) = |e|` **elementwise and exactly** (both
halves are exact power-of-two scalings). The main loss therefore uses

```
L = 2 * L_0.5(q50) + lambda_q * (L_0.1(q10) + L_0.9(q90))
```

so the q50 main task keeps the original v4 L1 gradient scale while q10/q90 are
auxiliary. `tau in {0.1, 0.5, 0.9}` fixed.

## 5. Controls

* **A. Exact v4 guard (`quantile_mode=none`)** — seed-0 run
  `20260910-081449-790d891a` reproduces the canonical compact-v4-hinge run
  `20260909-194445-182c7021` **bit-exactly** (60/60 epochs, best
  0.17006561887910357 @ epoch 53, params 99,613).
* **B. Median-only (`quantile_mode=median_only`, scalar head + `2*pinball_0.5`)**
  — run `20260910-082335-4160bc7f` is **bit-identical** to the L1 run
  (60/60 epochs, 0.17006561887910357 @53). Replacing the L1 implementation by the
  pinball implementation therefore produces **zero fake gain**.

## 6. Lambda selection (pre-registered grid, Stage 1, seed 0)

Only `lambda_q ∈ {0.10, 0.25, 0.50}` was run. Selection metric = **validation q50
MAE** (never width/coverage).

| variant | params | lambda | valid q50 MAE | Δ vs v4 | width–|error| Spearman | 80 % coverage |
|---|---:|---:|---:|---:|---:|---:|
| v4 L1 (canonical + guard) | 99,613 | — | 0.170066 | — | — | — |
| median-only | 99,613 | 0 | 0.170066 | 0.000000 (bit-identical) | — | — |
| MQ-0.10 | 99,679 | 0.10 | 0.186205 | **−0.016139** | 0.1996 | 0.836 |
| MQ-0.25 | 99,679 | 0.25 | 0.158643 | **+0.011423** | **0.2543** | 0.798 |
| MQ-0.50 | 99,679 | 0.50 | 0.158424 | **+0.011642** | **0.2630** | 0.783 |

Selection rule: λ=0.50 has the lowest valid q50 MAE, but by only 0.000219
(< the 0.001 tie band), so the smaller **λ = 0.25** is selected (§35: prefer the
smaller auxiliary weight when MAE is within the tie band; uncertainty
correlation is *not* used as a tie-breaker).

Notes: λ=0.10 is a clear regression (best epoch 32/44; the auxiliary term at
0.10 destabilises early training). λ=0.25/0.50 both clear the "Strong GO"
point-prediction band at seed 0.

**Stage-1 case: A — STRONG MECHANISTIC GO** (point improvement +0.0114 ≥ 0.008;
width-error 0.254 ≥ 0.25) → proceed to frozen multi-seed confirmation.

## 7. Point-prediction results (multi-seed, λ=0.25, validation only)

Serial isolated runs (CPU policy), same protocol, no test access.

| seed | v4 valid | v5 q50 valid | Δ (v4−v5) | width–|error| Spearman | central-80 coverage |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.170066 | 0.158643 | +0.011423 | 0.2543 | 0.798 |
| 1 | 0.163167 | 0.165556 | −0.002390 | 0.1646 | 0.744 |
| 2 | 0.174149 | 0.171256 | +0.002893 | 0.2060 | 0.790 |
| 3 | 0.170421 | 0.167601 | +0.002820 | 0.2196 | 0.789 |

| stat | paired Δ |
|---|---:|
| mean | **+0.003686** |
| std | 0.005720 |
| median | +0.002856 |
| min / max | −0.002390 / +0.011423 |
| v5 better | **3/4 seeds** |

**Pre-registered multi-seed point gate:** mean ≥ 0.005 & ≥3/4 seeds → **NOT met**
(mean +0.0037). Band = **Mild** (0.003–0.005, majority seeds improve). The seed-0
effect does not reproduce; the gain is fragile (seed 1 is negative).

## 8. Uncertainty diagnostics (λ=0.25, seed 0)

* `width80 = q90 − q10`; lower/upper widths `q50−q10` / `q90−q50`.
* **width vs |error| Spearman = 0.2543** (GO band ≥ 0.25 at seed 0); robust to the
  outlier tail — 0.2489 after removing |error| ≥ 1 (n=995), 0.2493 after removing
  the top 1 % errors.
* **width vs signed residual** Spearman = −0.095, Pearson = −0.049 → width
  predicts *magnitude*, not direction (consistent with the prior audits).
* Width distribution: mean 0.402, median 0.339, std 0.203, p10/p50/p90 =
  0.297/0.339/0.540, min 0.253, max 3.040. No collapse (no constant width), no
  explosion beyond a few `C`-group molecules.
* Asymmetry: mean upper−lower = −0.019 (near symmetric; quantiles not forced).
* Coverage (nominal 0.10 / 0.90 / 0.80): **0.121 / 0.919 / 0.798** — near-calibrated
  at the 80 % level, slightly over-covered at q10 (raw learned quantiles; no
  post-hoc calibration applied).
* **width vs v4 4-seed disagreement = 0.408** (seed 0) and 0.39–0.41 all seeds —
  larger than width vs |error| itself. The width internalizes the *epistemic-like*
  axis more strongly than the observed residual magnitude.
* **width vs frozen OOF difficulty predictor**: a static model-visible
  StandardScaler+Ridge difficulty predictor (31 shared rarity/structure/topology
  features) fitted on the OOF-train table reaches OOF-train Spearman 0.3735 and
  its validation predictions correlate with v5 width at **0.376** (λ=0.25) /
  0.375 (λ=0.50). So the single-model interval width absorbs essentially the full
  difficulty axis the independent OOF audit could extract (0.374), and it
  correlates with the OOF *epistemic* axis (disagreement) at ≈0.40.

### Multi-seed uncertainty gate

Per-seed width–|error| Spearman: 0.254 / 0.165 / 0.206 / 0.220 → **mean 0.211**
(std 0.037), **4/4 seeds positive**. Pre-registered gate (mean ≥ 0.25 & ≥3/4) →
**NOT met**; band = **Mild** (0.15–0.25). So the width signal is real, consistently
direction-correct, but weaker than the seed-0 case suggested.

## 9. Rarity → width mechanism (seed 0, λ=0.25)

Fixed rarity bins on `rare<=5` ratio (train-derived frequencies):

| rarity bin | n | q50 MAE | mean width80 | mean rare≤5 |
|---|---:|---:|---:|---:|
| 0 | 502 | 0.100 | 0.381 | 0.000 |
| (0.01, 0.05] | 191 | 0.107 | 0.376 | 0.041 |
| (0.05, 0.1] | 153 | 0.185 | 0.443 | 0.075 |
| (0.1, 0.2] | 117 | 0.408 | 0.461 | 0.136 |
| > 0.2 | 37 | 0.319 | 0.455 | 0.273 |

The MAE ladder is monotone through the (0.1,0.2] bin; width increases with rarity
but modestly (Spearman `rare_le5 ↔ width80` = 0.225; `oov ↔ width` = 0.104).
Width also tracks the target level/SA axis strongly (`width ↔ z_SA` = −0.766,
`width ↔ z_logP` = −0.766): low-`z_SA` (and large-magnitude target) molecules get
wide intervals — the same axis the post-v4 audit flagged as difficulty.

### Width quintile ladder (seed 0, λ=0.25)

| width quintile | n | actual MAE | median | rare≤5 mean |
|---|---:|---:|---:|---:|
| Q1 (narrowest) | 200 | 0.0958 | 0.0713 | 0.024 |
| Q2 | 200 | 0.1105 | 0.0772 | 0.036 |
| Q3 | 200 | 0.1183 | 0.0794 | 0.045 |
| Q4 | 200 | 0.2493 | 0.1136 | 0.056 |
| Q5 (widest) | 200 | 0.2193 | 0.1458 | 0.066 |

Monotone through Q4; Q5 dips below Q4 on mean MAE (but not on median, 0.146 >
0.114) because the very widest intervals include a few heavy-tail `C`/long-cycle
molecules whose individual errors are order-artifact dominated. Ratio
Q5/Q1 = 2.29 (median ratio 2.04).

## 10. Subgroup / rarity-group analysis (seed 0, λ=0.25, Δ = v4 − v5)

| topology group | n | v4 MAE | v5 q50 MAE | Δ | mean width |
|---|---:|---:|---:|---:|---:|
| A ordinary | 965 | 0.1378 | 0.1287 | **+0.0091** | 0.377 |
| B mild long-cycle | 30 | 0.2262 | 0.2320 | −0.0058 | 0.942 |
| C extreme long-cycle | 5 | 6.068 | 5.499 | +0.569 | 1.923 |

| rarity group | n | v4 MAE | v5 q50 MAE | Δ | mean width |
|---|---:|---:|---:|---:|---:|
| easy/common | 502 | 0.1011 | 0.1004 | +0.0007 | 0.381 |
| medium | 344 | 0.1567 | 0.1417 | +0.0150 | 0.406 |
| rare | 154 | 0.4249 | 0.3864 | +0.0385 | 0.460 |

The seed-0 gain concentrates in **hard/rare** molecules (rare +0.0385, medium
+0.0150) while **common molecules are flat** (+0.0007) — the bulk-safety
guideline is respected (no degradation of the common bulk). The B-group
mild-long-cycle mean regresses slightly (30 molecules); the C-group is dominated
by the known order artifact.

## 11. OOF difficulty comparison (§48 closure)

| signal | value |
|---|---:|
| OOF audit difficulty predictor (model-visible) | 0.374 |
| OOF audit disagreement–error | 0.393 |
| v5 width–|error| (seed 0 / mean 4 seeds) | 0.254 / 0.211 |
| v5 width–v4-disagreement (seed 0 / range) | 0.408 / 0.39–0.41 |
| v5 width–frozen-OOF-difficulty predictor | 0.376 |

A single model's quantile width correlates with the previously confirmed
difficulty axis at ≈0.376 (almost the full OOF ceiling) and with the
epistemic-like disagreement at ≈0.40, while its correlation with its *own*
observed |error| is only ≈0.21–0.25. The width therefore internalises the
OOF-difficulty structure, but that internalisation does **not** translate into a
reliable median improvement or into a stronger width–error coupling.

## 12. Multi-seed confirmation and final benchmark test (frozen, one-time)

Selected/frozen: `quantile_mode=q10_q50_q90`, **λ=0.25**, all other variables from
the frozen v4 config. Primary benchmark protocol (frozen validation-selected
checkpoint, train-only fits, single test evaluation), seeds 0–3, serial:

| seed | v4 test | v5 test | Δ (v4−v5) |
|---:|---:|---:|---:|
| 0 | 0.133901 | 0.136037 | −0.002136 |
| 1 | 0.132006 | 0.141317 | −0.009311 |
| 2 | 0.144614 | 0.141324 | +0.003290 |
| 3 | 0.137019 | 0.138932 | −0.001913 |
| **mean** | **0.136885** | **0.139403** | **−0.002518** |
| std | 0.005552 | 0.002510 | 0.005177 |

**v5 is not better on the benchmark test** (1/4 seeds; paired mean −0.0025,
within the seed spread). The terminal runs' selection-phase validation traces are
bit-identical to the corresponding scratch runs (measurement-transparent), so the
test evaluation introduced no training change. This is a frozen final
observation; no tuning followed.

## 13. Decision

Stage 1 seed 0: **Case A** (Strong-GO point + GO uncertainty) → multi-seed
confirmation was entered.
Stage 2 (4 seeds): point **Mild** (+0.0037, 3/4), uncertainty **Mild** (0.211,
4/4) — the strict pre-registered multi-seed gates (≥0.005 point, ≥0.25 width) are
**not met**, and the one-time benchmark test **regresses** by −0.0025.

Because the point signal is Mild (not GO) rather than NO-GO, this is not a clean
pre-registered Case B/C/D. The honest verdict:

> **NO-GO for a confirmed multi-quantile benchmark improvement.** The
> auxiliary-quantile objective does not reliably improve the conditional median,
> and the seed-0 gain does not survive multi-seed or test. Interval width is
> consistently (4/4 seeds) but weakly input-dependent, and it absorbs the
> previously confirmed OOF difficulty/epistemic axis; it is not a signed-residual
> predictor.

**Next step (single):** do **not** open uncertainty-guided weighting /
resampling / distillation yet — its pre-registered precondition (mean width–error
GO ≥ 0.25) was not met. The route should be re-registered or closed on the
OOF-train set before any uncertainty-guided objective is attempted.

## 14. Limitations

* Only 4 seeds (n=4); the paired point estimate has wide uncertainty (±0.006).
  The λ grid is pre-registered to three values and was not widened.
* `q90−q10` is a **model-implied predictive interval width**, not aleatoric
  uncertainty: the ZINC target is deterministic (normalized logP).
* Width correlates strongly with the target/SA level (−0.77); it reflects target
  magnitude as well as difficulty, so `width ↔ |error|` is not a pure difficulty
  measure.
* The frozen difficulty predictor used for §10 is a 31-feature **static**
  re-fit of the OOF "model-visible" predictor (states excluded because they are
  per-fold artifacts); the OOF-train number 0.3735 reproduces the audit's 0.374.
* Validation subgroup B/C estimates are small-n (30/5) and single-seed.
* No calibration (temperature/conformal/isotonic) was applied — raw learned
  quantiles only.

## 15. Answers Q1–Q15

1. **Q1 – code paths changed:** only `zinc_patch_path_pooling.py`
   (`pinball_loss`, `quantile_regression_loss`, `_validate_quantile_config`,
   non-crossing decode in `PatchPathModel.forward`, head width `1→3` in the mq
   mode, loss/eval/selection plumbing in `_train_phase`/`_evaluate`/`run`).
   Representation code untouched.
2. **Q2 – parameter delta:** +66 (99,613 → 99,679); median-only/none = 99,613.
3. **Q3 – median-only reproduces L1:** yes, **bit-identical** (60/60 epochs,
   same best MAE/epoch as the L1 run).
4. **Q4 – best λ (validation q50 MAE):** λ=0.50 (0.158424) by 0.000219; λ=0.25
   selected by the smaller-λ tie rule.
5. **Q5 – best q50 MAE improvement vs v4:** seed 0: **+0.011423** (0.170066 →
   0.158643) for λ=0.25.
6. **Q6 – width–error Spearman:** seed 0 = **0.2543** (λ=0.25), 0.2630 (λ=0.50);
   4-seed mean = **0.2111**.
7. **Q7 – width quintiles vs MAE monotone:** yes through Q4
   (0.096→0.111→0.118→0.249), Q5 mean dips (0.219) but its median keeps rising.
8. **Q8 – rare-patch ratio vs width:** weakly positive (Spearman rare≤5↔width
   0.225; OOV 0.104); width is more strongly tied to the SA/target axis.
9. **Q9 – width vs v4 seed disagreement:** **0.408** (seed 0), 0.39–0.41 across
   seeds — stronger than width–|error|.
10. **Q10 – q10/q90 coverage (λ=0.25 seed 0):** 0.121 / 0.919; central-80 0.798.
11. **Q11 – A/B/C subgroups keep v4 gain:** A ordinary +0.0091 (improves);
    B mild −0.0058 (slight regression, n=30); C +0.569 (artifact-dominated).
12. **Q12 – hard/rare benefit more than common:** yes — rare +0.0385, medium
    +0.0150, common +0.0007.
13. **Q13 – multi-seed stability of q50 improvement:** **not stable** — mean
    +0.0037, 3/4 seeds, one negative; Mild, below the 0.005 GO gate, and the test
    regresses.
14. **Q14 – case:** not a pre-registered Case A (Stage-1 seed 0 was A, but the
    frozen multi-seed gates are not met); outcome = **Mild/Mild ⇒ no confirmed
    GO** (closest to, but not equal to, Case D; explicitly neither a confirmed GO
    nor a clean NO-GO).
15. **Q15 – single recommended next stage:** re-register or close the
    multi-quantile route on the OOF-train set; do **not** open
    uncertainty-guided weighting/distillation (width GO precondition not met).

## 16. Evidence index

* Runs (local, git-ignored): `runs/2026/09/10/`
  * none guard `20260910-081449-790d891a`, median-only `20260910-082335-4160bc7f`,
    MQ-0.10 `20260910-082959-706bb17a`, MQ-0.25 `20260910-083648-62c77ebd`,
    MQ-0.50 `20260910-084328-b0817b52`
  * multiseed λ=0.25 s1 `20260910-085142-c49d0536`, s2 `20260910-085811-e7e99190`,
    s3 `20260910-090440-ba4ca9ae`
  * test (terminal) s0 `20260910-091252-5e114a07`, s1 `20260910-091947-500a2536`,
    s2 `20260910-092640-8ef02123`, s3 `20260910-093339-417cbcbd`
* Analysis artifacts (git-ignored except summaries):
  `results/compact_v5_multi_quantile/` — `run_map.json`,
  `stage1_seed0_summary.json`, `decision_record.json`, `seed0/*.csv`, `figures/`.
* Configs: `configs/luyin16/zinc_compact_v5_quantile_{none,median_only,lambda010,lambda025,lambda050}.yaml`.
* Tests: `tests/test_zinc_patch_path_quantile_v5.py` (13 tests).
* Guard tool: `experiments/luyin16/zinc_compact_v5_trace_guard.py`.
* Analysis tool: `experiments/luyin16/zinc_compact_v5_multi_quantile_analysis.py`.

Figures (seed 0, λ=0.25; also produced for λ=0.10/0.50):
`fig1_width_vs_error_*`, `fig2_rarity_vs_width_*`, `fig3_width_quintile_mae_*`,
`fig4_easy_hard_quantiles_*`, `fig5_width_vs_disagreement_*`,
`fig6_width_by_rarity_group_*`.
