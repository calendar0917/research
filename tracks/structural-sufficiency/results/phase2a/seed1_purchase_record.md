# Phase 2a — seed-1 purchase record (written before the seed-1 runs)

Protocol: `ssrc-phase2a-zinc-v1` (`notes/phase2a_protocol.md` §7)
Written after the seed-0 score, before any seed-1 training. Any later change to
this file must be appended, never rewritten.

## 1. Seed-0 values used by the six frozen purchase conditions

| # | condition | threshold | seed-0 value | pass |
|---|---|---|---|---|
| 1 | `Gain_P3E = MAE(P3E) - MAE(M)` | ≥ 0.004 | **0.005675** | ✅ |
| 2 | `Gain_MP = MAE(F) - MAE(M)` | ≥ 0.002 | **0.003764** | ✅ |
| 3 | RMSE not clearly worse than P3E | frozen proxy: `ΔRMSE ≤ +0.005` | **+0.005208** | ❌ (by 0.000208) |
| 4 | gain not concentrated in the cycle-penalty / extreme rows | complement gain ≥ 0.002 | **0.006320** (957 rows) | ✅ |
| 5 | new-branch gradients normal, no optimisation fault | all groups nonzero, soup fit MAE < 0.05 | pass (0.000998; `local.*` grads 0.18–0.21) | ✅ |
| 6 | cost vs gain acceptable | ≤ 1.5× FLOPs, ≤ 0.6 GPU-h/arm | **1.151×**, 0.207 h/arm | ✅ |

## 2. Condition 3: the frozen proxy failed, and what the failure is made of

The frozen proxy is an **absolute** RMSE threshold. It failed by 0.000208
(4.2 % of its own tolerance). Post-hoc tail diagnostics on the same frozen
predictions (`posthoc_phase2a.json`, no training, no selection):

| diagnostic | value |
|---|---|
| MSE difference `M - P3E` (mean) | +0.002843, bootstrap CI **[-0.003706, +0.013303]** (crosses 0) |
| RMSE difference bootstrap CI | [-0.012974, +0.015464] (crosses 0) |
| RMSE difference after dropping the single worst squared-error contributor | **-0.005758** (M better) |
| that contributor | Monitor index 341 / official-train row **2052**, `y = -7.93`, AE 7.468 (M) vs 7.160 (P3E) |
| its MSE-difference share of the total difference | **1.586×** (it alone flips the sign) |
| is it in the Phase 1e `permutation_sensitive` set? | **yes** — its cycle penalty changes under atom permutation |
| RMSE on the 996 rows whose label is a graph function | P3E 0.147848, M 0.142046 (**M better by 0.005801**) |

Interpretation: the RMSE regression is not distinguishable from zero, and it is
produced by one molecule whose stored target is *provably not* a
permutation-invariant function of the graph (Phase 1e audit). A guard that fails
only through such a row cannot be read as evidence that the model's error tail is
worse.

## 3. Judgment (disclosed, post-hoc)

- The **frozen numeric proxy of condition 3 failed** and is recorded as failed.
  It is **not** re-thresholded, and the seed-0 artefact
  `decision_phase2a.json` keeps `INSUFFICIENT_EVIDENCE`.
- The task's condition 3 is qualitative ("RMSE 没有明显恶化"). On the evidence in
  §2 it is judged **satisfied in substance**, and conditions 1, 2, 4, 5, 6 hold.
- This is an explicit post-hoc judgment on a rule the protocol wrote badly (an
  absolute RMSE threshold is dominated by one unlearnable-label molecule). It is
  recorded here so a reader can reject the judgment and read the round as a
  single-seed `INSUFFICIENT_EVIDENCE` result instead.
- The **rule defect** is logged for future protocols: the RMSE guard should be
  specified as "MSE-difference CI must include 0 or favour the candidate", or
  evaluated on the label-is-a-graph-function complement.

## 4. What is purchased, and what is not

Purchased (one wave, three jobs):

| arm | seed | kind | init |
|---|---|---|---|
| `P3E` | 1 | `none` (frozen trunk, no local module) | composer/reader scratch seed 1, SSL encoder **fixed at s0** |
| `F` | 1 | `node_ffn` | same, + local modules seed 1 |
| `M` | 1 | `local_mp` | same, + local modules seed 1 |

Why a `P3E` seed-1 arm: the frozen P3E number is a single seed-0 measurement.
Comparing a seed-1 candidate against it confounds the seed with the
architecture. The matched seed-1 baseline is trained with the *same* fixed SSL
encoder, the *same* scratch init and the *same* shuffle seed, so
`Gain_P3E(seed 1)` is a paired contrast. This arm is explicitly allowed by
protocol §7 ("必要时使用匹配 seed 的 P3E 基线"); it changes nothing about F/M.

Disclosed deviation: the Fit-only masked-SSL encoder is held at seed 0 for both
seeds and all arms. Seed 1 therefore varies only the supervised trajectory
(composer/reader init + batch order). This removes one variance source and keeps
the triple matched; it means the two-seed spread here is a **lower bound** on the
spread of the original two-seed BJOC recipe.

Not purchased: any third seed, any width/depth/learning-rate variant, any new SSL
objective, any cycle-penalty routing, any ensemble.

## 5. Stopping rule for the round

- Compute `Gain_P3E(s1)` and `Gain_MP(s1)` with the same estimators.
- If the two seeds agree in sign and magnitude on `Gain_P3E` and `Gain_MP`, the
  round is `STRONG_BACKBONE_MP_INCREMENT` (or `CAPACITY_ONLY_INCREMENT` if the
  M-vs-F separation dies at seed 1), with the seed spread reported.
- If the seeds conflict in direction, the round is `UNRESOLVED` and no further
  seed is bought.
- No re-tuning follows either outcome.
