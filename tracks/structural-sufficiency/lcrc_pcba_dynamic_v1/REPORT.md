# REPORT — LCRC-W2-DYNAMIC v1

Round: `lcrc-pcba-dynamic-v1`
Host: `res` (`a100-2`), single idle A100-SXM4-40GB (40 GB), driver 550.163.01,
torch 2.5.1+cu124, CUDA 12.4, Python 3.12.14, FP32, TF32 off
Formal run: 2026-11-09 13:05:05–14:49:08 UTC (commit `ea7eec6b`)
Test role: **`EXPOSED_TEST_ENGINEERING_COMPARISON_SET`**
Single-seed diagnostic boundary experiment: **`W2-DYNAMIC`, seed 0, 40 epochs**

---

## Result in one paragraph

Adding two layers of real chemical-bond message passing that read the
neighbours' *learned hidden states* on top of the frozen LCRC-W2 one-shot S1/S2
encoder raises official MolPCBA scaffold-split performance from **0.265273 →
0.275067 Validation AP** and **0.253617 → 0.262226 Test AP** at the
best-Validation checkpoint (Δ = **+0.009794** Valid, **+0.008609** Test), and
also beats the historical RC-REAL reference (Valid 0.267718, Test 0.255941) by
**+0.007349 / +0.006285**. The gain is **broad**: 76 of the 108 high-support Test
tasks and 75 of the 106 common high-support tasks improve, with mean per-task
Δ ≈ **+0.0106…+0.0109** on those subsets, and the single <5-positive Test task
actually *reduces* the macro number by −0.0012. The dynamic branch is
unambiguously used (R1 = 0.65, R2 = 0.53, Rtotal = 0.94 on the full Validation
split; zeroing the message term after training drops Validation AP to 0.1149 and
within-molecule source shuffling drops it to 0.1758). Because the *holistic*
ΔTest = +0.0086 sits just below the pre-registered +0.010 STRONG threshold (only
that one STRONG condition is missed — the ≥70/106 improvement count, the
high-support mean ≥ +0.006, the positive Validation direction and the
branch-usage condition are all met), the frozen decision rule labels the round
**`POSSIBLE_DYNAMIC_SIGNAL`**, not `STRONG_DYNAMIC_SIGNAL`.

---

## OBSERVED

### 1. Headline comparison

Official `ogb.graphproppred.Evaluator("ogbg-molpcba")` AP; the independent
per-task sklearn recomputation agrees to **≤1.1e-16** on every model (5.6e-17 for
W2), so the aggregation path is verified, not assumed. W2 / RC-REAL numbers are
recomputed here from their already frozen BOC-round Test logits through the
identical code and are **bit-identical on Test AP** (absolute difference 0.0) and
within 8.1e-7 on Validation AP to the BOC-round recorded values — a metric-code
identity check, not a new measurement. The rounded figures in the brief
(0.253617 / 0.255941) are those same numbers to 6 dp.

| Model | Params | Best Valid AP | Test AP best-Valid | Valid@40 | Test@40 | GPU-hour |
|---|---:|---:|---:|---:|---:|---:|
| W2 historical | 994,324 | 0.265273 | 0.253617 | 0.261867 | 0.250986 | 1.3949 |
| RC-REAL historical | 1,051,022 | 0.267718 | 0.255941 | 0.266950 | 0.256167 | 1.9186 |
| **W2-DYNAMIC** | **1,128,468** | **0.275067** | **0.262226** | **0.268791** | **0.260136** | **1.7340** |

Δ (W2-DYNAMIC − reference):

| Quantity | vs W2 | vs RC-REAL |
|---|---:|---:|
| Test AP, best-Valid | **+0.008609** | **+0.006285** |
| Validation AP, best-Valid | +0.009794 | +0.007349 |
| Test AP, fixed epoch 40 | +0.009150 | +0.003969 |
| Validation AP, fixed epoch 40 | +0.006924 | +0.001841 |

Best epoch: W2-DYNAMIC 25 (0.275067), W2 24 (0.265273), RC-REAL 33 (0.267718).

**The two checkpoints disagree in the useful direction.** At the fixed epoch-40
endpoint — which carries no Validation-selection advantage at all — W2-DYNAMIC is
+0.009150 Test / +0.006924 Valid above W2@40, i.e. *larger* than the
best-Valid gap. The headline is therefore not a best-epoch cherry-pick: the
40-epoch curve is above W2 at 39 of 40 epochs (only epoch 2 is below), and the
late-epoch advantage (+0.004…+0.007) persists to the frozen endpoint.

### 2. Is the gain broad, or a few sparse tasks?

Test positive-count strata (best-Valid checkpoints, 127 scored tasks; 108 tasks
have ≥20 Test positives, 18 have 5–19, exactly **1** has <5). Task id 45 is
excluded: it has **zero** labelled positives in Valid and Test, so no AP exists for
any model and it never enters any macro average.

| Stratum | n tasks | mean Δ | median Δ | improved | degraded | macro contribution |
|---|---:|---:|---:|---:|---:|---:|
| Test positives ≥ 20 | 108 | **+0.010866** | +0.011586 | **76** | 32 | +0.009240 |
| Test positives 5–19 | 18 | +0.004212 | +0.000598 | 11 | 7 | +0.000597 |
| Test positives < 5 | 1 | −0.155968 | −0.155968 | 0 | 1 | **−0.001228** |
| all scored | 127 | +0.008609 | +0.009379 | 87 | 40 | +0.008609 |
| Validation & Test both ≥ 20 | **106** | **+0.010624** | +0.011586 | **75** | 31 | +0.008867 |
| Validation ≥ 20 (Validation AP) | 107 | +0.011262 | +0.007760 | 75 | 32 | +0.009488 |
| all scored, **vs RC-REAL** | 127 | +0.006285 | +0.006336 | 79 | 48 | +0.006285 |

Answer: **broad, and low-support tasks do not create the gain — they slightly
suppress it.** The one <5-positive task (task 57, 4 Test positives, Δ = −0.156)
contributes −0.00123 to the macro; without it the macro would be larger. The
≥20 stratum mean (+0.0109) is higher than the all-task macro (+0.0086), so the
effect is strongest exactly where the estimate is most stable. Against RC-REAL
the same shape holds (79/127 improve, mean +0.0063, median +0.0063), so the
result is not an artefact of which reference is used.

The low-support dominance check is negative: `analysis/comparison.json` →
`low_support_dominance` shows the sub-20-support tasks carrying only
+0.000597 − 0.001228 = **−0.00063** of the +0.008609 macro, while the ≥20 stratum
carries +0.00924 (**107%** of the total). The single low-support task alone is
−14.3% of the summed per-task delta and 4.7% of the summed |delta|.

Distribution inside the high-support stratum: 57/108 tasks improve by > +0.01 and
36/108 by > +0.02, against 14/108 degrading by > 0.01 and 9/108 by > 0.02. It is
a genuine two-sided redistribution, not a uniform lift.

Descriptive sensitivity to the extremes (official macro AP remains the reported
number; this decomposition only shows concentration):

| Removed | Tasks removed | Mean Δ of removed | Macro Δ remaining |
|---|---|---:|---:|
| abs-Δ rank 1 | 77 | +0.2676 | +0.006553 |
| abs-Δ ranks 1–3 | 77, 108, 57 | −0.0488 | **+0.009999** |
| abs-Δ ranks 1–5 | above + 31, 2 | −0.0177 | +0.009686 |
| abs-Δ ranks 1–10 | above + 62, 117, 110, 120, 9 | −0.0021 | +0.009527 |

Removing the three largest-magnitude tasks *raises* the macro delta (two of them
are losses). No single task, and no group of tasks, carries the result.

Largest positive and negative per-task changes (task id is the official PCBA
task index; both splits shown):

| Rank | Task | Test pos | Valid pos | Δ Test vs W2 | Δ Test vs RC-REAL |
|---:|---:|---:|---:|---:|---:|
| +1 | 77 | 10 | 7 | **+0.2676** | +0.2296 |
| +2 | 31 | 6 | 6 | +0.1489 | +0.0417 |
| +3 | 62 | 11 | 10 | +0.0838 | +0.0530 |
| +4 | 110 | 267 | 161 | +0.0745 | +0.0548 |
| +5 | 9 | 32 | 14 | +0.0564 | +0.0678 |
| −1 | 108 | 7 | 1 | **−0.2582** | −0.1744 |
| −2 | 57 | 4 | 7 | −0.1560 | −0.2810 |
| −3 | 2 | 19 | 16 | −0.0907 | −0.0239 |
| −4 | 117 | 7 | 9 | −0.0818 | −0.1281 |
| −5 | 120 | 11 | 13 | −0.0658 | +0.1652 |

The extremes are mostly sparse tasks (4–19 Test positives) on both sides; the
exceptions are 110 and 9 on the positive side, where the improvement is
+0.056…+0.075 on 267 and 32 positives. Per the brief, no architecture claim is
made from any single one of these tasks.

### 3. Validation / Test agreement

| Quantity | Value |
|---|---:|
| Validation macro Δ (127 tasks) | +0.009794 |
| Test macro Δ (127 tasks) | +0.008609 |
| Validation improved tasks | 85 / 127 |
| Test improved tasks | 87 / 127 |
| sign agreement of per-task Δ | 65.4% |
| Pearson / Spearman of per-task Δ | 0.203 / 0.256 |
| Validation ≥20-pos subset mean Δ (107 tasks) | +0.011262 (75 improved) |

Both splits point the same way in level and in breadth; task-level correlation is
modest, as expected between two different scaffold partitions.

### 4. Was the dynamic branch actually used?

Full official Validation split, eval mode, best-Valid checkpoint:

| Quantity | best-Valid (ep 25) | epoch-40 |
|---|---:|---:|
| RMS h0 | 0.973763 | 0.987573 |
| RMS (h1 − h0) | 0.628428 | 0.710811 |
| RMS (h2 − h1) | 0.617609 | 0.730953 |
| RMS (h2 − h0) | 0.914377 | 1.050539 |
| **R1** | **0.645361** | 0.719755 |
| **R2** | **0.529090** | 0.591138 |
| **Rtotal** | **0.939014** | 1.063759 |
| finite fraction of h2 | 1.000000 | 1.000000 |

Per-epoch trajectory on the fixed 32-row diagnostic batch: the update magnitude
grows steadily over training (with small epoch-to-epoch dips, e.g. R1 0.6167 →
0.6071 at epochs 21→22) from R1 0.0935 / R2 0.0885 / Rtotal 0.1788 at epoch 1 to
R1 0.7122 / R2 0.6107 / Rtotal 1.0820 at epoch 40; `fc2` weight norm 0.85 → 11.71
for block 0, and `bond_proj` gradient norm stays ≈ 1.1e-3 → 2.9e-3 over all 40
epochs. No epoch produced a non-finite logit or a non-finite h2; grad-norm
clipping fired 13 times in 109,520 steps.

Gradient audit (real training steps, `analysis/preflight.json`):

| Step | New tensors receiving non-zero gradient | Structurally zero |
|---:|---|---|
| 1 | `fc2.weight`, `fc2.bias` of both blocks (4 of 14) | `bond_proj.*`, both `fc1.*`, both `norm.*` (10 of 14) |
| 3 | **all 14** new tensors | none |
| 10 | all 14 | none |
| 100 | all 14 | none |

This is exactly the predicted consequence of `fc2 = 0` initialisation: at step 1
only the (zero) output layer can receive gradient, and from step 3 onward the
gradient reaches `bond_proj`, both `fc1` and both `LayerNorm`s of both layers.
Total new-tensor gradient norm falls 4.52 (step 1) → 4.03 (step 3) → 1.67
(step 10) → 0.140 (step 100) against a W2-trunk norm of 16.66 → 15.12 → 6.76 →
0.357, i.e. the new path holds a stable **20–28%** share of the total gradient as
the loss drops from 1.881 to 0.063. `bond_proj.weight` itself carries 7.5e-3 at
step 10 and 2.9e-3 at epoch 40 — small in absolute terms, but non-zero and
stable, so the shared projection never falls into a dead state.
The branch is therefore **not** `DYNAMIC_BRANCH_SUPPRESSED` in any sense the
protocol allows: the updates are two orders of magnitude away from "numerically
~zero" (Rtotal 0.94, and R1/R2 alone are 0.65/0.53), every new tensor receives
gradient, and Rtotal keeps *growing* through epoch 40 — training moves toward
more, not less, message exchange. Per the protocol, Rtotal is a usage indicator
only and is never cited as evidence of predictive ability.

### 5. Post-training Validation-only interventions (frozen best-Valid checkpoint)

No retraining, no checkpoint reselection, no tuning. Both interventions are
out-of-distribution inputs.

| Mode | Validation AP | Δ vs real | Δ Valid BCE | rel. RMS logits change | tasks improved vs real |
|---|---:|---:|---:|---:|---:|
| real | 0.275064 | — | — | — | — |
| `message_off` (message term = 0) | **0.114931** | **−0.160133** | +0.0299 | 0.475 | 4 / 127 |
| `message_shuffle` (within-molecule `src` permutation; `dst` multiplicity, `h0`, bond features and every W2 static input unchanged) | **0.175790** | **−0.099274** | +0.0066 | 0.232 | 7 / 127 |

At epoch 40 the same pattern holds (0.268791 → 0.104043 off / 0.161251 shuffle).
Two facts follow: (i) the trained model depends on the message path on
essentially every task (123/127 and 120/127 tasks get worse under the two
interventions), and (ii) the *correspondence* matters, not merely the presence of
aggregated neighbour content — randomising which neighbour contributes to which
target inside the same molecule recovers only ~62% of the lost AP.

These numbers bound sensitivity only. A static-architecture model given the same
out-of-distribution input perturbation would also lose AP; this is **not** a
substitute for the missing capacity-matched STATIC control, and it is **not** a
causal proof of an independent message-passing generalisation gain.

### 6. Train / Valid curves

Train BCE improves strictly and is better than W2 at every one of the 40 epochs
(epoch 1: 0.060564 vs 0.061794; epoch 40: **0.023577** vs 0.026289), and train
macro AP at epoch 40 is **0.654260** vs W2's 0.581875. Validation BCE, however,
is slightly *worse* at epoch 40 (**0.048017** vs 0.046538). The gain is therefore
purely a ranking gain, not a calibration or regularisation artefact — the same
signature the historical structural-gain round reported (AP up strongly, BCE
nearly flat). It also means W2-DYNAMIC is better at ordering positives without
being better calibrated, so downstream use would need recalibration.

The same holds on Test (BCE 0.049109 for W2-DYNAMIC vs 0.048584 for W2, 0.049581
for RC-REAL, while Test AP is +0.0086 / +0.0063). AP and BCE therefore move in
opposite directions across the two architectures, which is a useful sanity check
that the reported AP delta is a reordering of positives rather than a
probability-level shift.

Validation AP per epoch (W2-DYNAMIC vs the historical W2 curve at the same epoch;
both runs share the identical batch-order plan `06f43729…`):

| Epoch | 1 | 4 | 8 | 12 | 16 | 20 | 24 | 25 | 30 | 35 | 40 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| W2-DYNAMIC | .0995 | .2066 | .2552 | .2627 | .2722 | .2693 | .2688 | **.2751** | .2677 | .2686 | .2688 |
| W2 | .0958 | .1997 | .2335 | .2462 | .2561 | .2586 | .2653 | .2631 | .2620 | .2641 | .2619 |
| Δ | +.0037 | +.0069 | +.0217 | +.0165 | +.0161 | +.0107 | +.0036 | +.0120 | +.0057 | +.0045 | +.0069 |

The advantage is largest in the middle of training (epochs 8–17, +0.010…+0.022)
and settles at a smaller but persistent +0.004…+0.007 band once W2's own curve
saturates.

---

## INTERPRETATION

### Which explanations fit

1. **Learned neighbour-state exchange carries usable information beyond the
   one-shot S1/S2 encoding.** In W2 the only route from a bond to a prediction is
   the bond encoder feeding the *local* block of the two atoms it touches; the
   readout sees `h0` only. Here the second layer demonstrably moves atom C's state
   because atom A's state changed two hops away (chain test: h1_C changes by 0.0,
   h2_C changes by 0.022), and the trained model loses 0.10–0.16 Validation AP
   when that correspondence is destroyed. The measured effect is thus located in
   the mechanism the experiment was designed to probe.
2. **The effect is not sparse-task noise.** 76/108 and 75/106 high-support tasks
   improve; removing the three largest-magnitude tasks *raises* the macro delta;
   the only sub-5-support Test task hurts rather than helps.
3. **The effect is not a best-epoch artefact.** The fixed epoch-40 endpoint
   reproduces it with a slightly larger Test delta.

### What cannot be excluded

1. **Added capacity.** W2-DYNAMIC has +134,144 parameters (+13.5%) and two extra
   non-linear layers. The closest capacity-controlled precedent in this repo is the
   structural-gain sprint on the much smaller C1 base (245,184 params, warm-started
   continuation to epoch 12): arm **F** = the same two pre-norm residual FFN blocks
   per atom *without* messages (+33,408 params) gave Δ = **−0.000934** (51/127 tasks
   improved, i.e. capacity and non-linearity alone bought nothing, slightly
   negative), while arm **M** = F + directed physical-bond messages (+896 params,
   capacity-matched to within 2.7%) gave Δ = **+0.003785** (81/127). That is real
   evidence that message content, not block capacity, drove the gain there — but it
   is a different base, a warm start and an epoch-12 comparison, so it does not
   transfer automatically to a from-scratch 40-epoch width-128 model. No
   capacity-matched STATIC arm was trained **in this round**, so "the gain comes
   from the message mechanism" is supported, not established.
2. **Longer-range reachability.** Two MP layers extend the effective receptive
   field of W2's local representation from 1 to 2 hops. This round cannot
   separate "more distant information becomes reachable" from "learned
   neighbour-context interaction" from "added capacity".
3. **Joint adaptation.** The dynamic layers were trained with the whole W2 trunk
   from step 0, sharing one LR and one parameter group. Co-adaptation of the
   trunk to the dynamic path is inside the measured effect and cannot be
   decomposed with one arm.
4. **Selection.** Validation AP was used to pick epoch 25 out of 40. The fixed
   epoch-40 numbers mitigate but do not remove this.
5. **Single seed, exposed Test.** No seed variance is available, and the Test
   split has been opened by earlier rounds, so the Test number is an engineering
   comparison, not a blind estimate.

### What this round does *not* say

It does not say "dynamic message passing is proven better than static LCRC". It
does not say "iterative MP is required for MolPCBA". It does not produce a
leaderboard claim. It does not licence any conclusion about the historical V arm
(graph-context feedback is absent here and its historical gain is not attributed
to message passing).

---

## DECISION

**Pre-registered label: `POSSIBLE_DYNAMIC_SIGNAL`.**

| STRONG condition (frozen in `PROTOCOL.md` §8) | Measured | Met |
|---|---|---:|:---:|
| ΔTest ≥ +0.010 | +0.008609 | **no** |
| ≥ 70 of 106 common high-support tasks improved | 75 / 106 | yes |
| high-support mean Δ clearly positive (ref +0.006) | +0.010624 (106) / +0.010866 (108) | yes |
| Validation direction positive | +0.009794 | yes |
| dynamic branch confirmed used | R1 .65 / R2 .53 / Rtotal .94, all gradients non-zero | yes |

Four of five STRONG conditions hold; the holistic ΔTest misses the +0.010 line by
0.0014. Per the frozen rule ("if the result falls between the bands, state the
measured evidence rather than forcing a label") this is reported as
`POSSIBLE_DYNAMIC_SIGNAL` — a real, broad, mechanism-consistent gain that is
*not yet* large enough, on the pre-registered holistic metric, to be called
strong. The subset evidence (high-support +0.0106…+0.0109, common-106 +0.0106)
means the honest summary is "at the boundary of the STRONG band, evaluated on the
holistic macro metric it falls just short".

**Is it worth continuing?** Yes — W2-DYNAMIC is the highest Test AP recorded so
far on this comparison set (0.262226 vs 0.255941 for RC-REAL and 0.254142 for
BOC-FULL), the Validation and Test directions agree, and a dedicated usage audit
(gradient reach, layer-input capture, chain dependency, post-training
interventions) confirms the added component is the one being exercised. That is
more evidence than any previous LCRC-PCBA round carried for its positive arm. But
the effect is currently not separated from added capacity or from extended
receptive range, so continuing is only justified **if the next step is a
mechanism-isolating one**, not a bigger version of the same arm.

**What a confirmation round would have to include** (none of it started here):

1. A capacity-matched STATIC control: same +134,144 parameters, same two extra
   non-linear layers, identical training, with the message term replaced by a
   parameter-matched non-message path (e.g. the same `MessageFFN` with `msg ≡ 0`
   plus a same-size per-atom projection). Without it, no causal claim is possible.
2. A one-layer MP arm, to separate "one hop of learned neighbour state" from "two
   hops" — this is the direct test of whether *iteration* matters, which is the
   actual research question.
3. At least one additional seed, to bound run-to-run variance against a measured
   +0.0086.
4. Ideally a Train-only ablation within the same run (e.g. a message-dropout
   schedule), so the comparison is not confounded with checkpoint selection.

Not to be done as a follow-up: larger widths, more layers, more capacity, other
GNNs, or a fourth round of the same design.

---

## Mandatory questions

**1. Does adding dynamic neighbour-state updates to W2 clearly outperform W2 and
RC?** Yes on both splits at both comparison endpoints — Test +0.008609 /
+0.006285 (best-Valid) and +0.009150 / +0.003969 (@40); Validation +0.009794 /
+0.007349 and +0.006924 / +0.001841. "Clearly" is bounded: the holistic Test
delta is 0.086 AP points, below the pre-registered +0.010 strong threshold, and
single-seed with an exposed Test split.

**2. Do the two new layers really perform consecutive hidden-state exchange?**
Yes, verified five independent ways: (a) forward hooks show block 0 receives
exactly `h0` and block 1 receives exactly `h1` (max abs diff 0.0, and block 1's
input differs from `h0` by 1.82); (b) on an `A—B—C` chain with non-degenerate
deterministic weights, perturbing `h0_A` leaves `h1_C` unchanged (0.0) but moves
`h2_C` by 0.022 and `h2_A` by 1.223 — the information reaches C *only* by going
through the second layer; (c) the aggregation matches an independent float64
numpy implementation to 4.5e-7 for both layers; (d) after training, within-molecule
`src` shuffling (which keeps message counts, `h0`, bond features and every W2
static input fixed) costs −0.099 Validation AP; (e) R1 = 0.65, R2 = 0.53 and
Rtotal = 0.94 on the full Validation split, still growing at epoch 40.

**3. Is the gain broad rather than a few sparse tasks driving macro AP?** Broad.
87/127 Test tasks improve; 76/108 and 75/106 high-support tasks improve with
mean +0.0106…+0.0109; the ≥20 stratum contributes +0.00924 of the +0.00861 macro;
the single <5-positive task contributes −0.00123 (i.e. it works *against* the
headline); removing the three largest-magnitude tasks raises the macro to
+0.0100. The 14/108 high-support regressions > 0.01 are real but outweighed.

**4. If it succeeds, what does it mean for the "no iterative MP" research
direction?** It means the no-iteration constraint was a *real* constraint, not a
free design choice: 0.86 Test AP points and a broad task-level improvement are
available from two layers of genuine neighbour-state exchange on top of an
otherwise unchanged W2, at +13.5% parameters and +24% training time
(1.734 vs 1.395 GPU-hours). It does **not** mean LCRC should be replaced by a GNN.
It means the question "does *iteration* buy anything once the one-shot S1/S2
chemistry is already encoded?" now has a positive, usage-verified answer inside
the 40-epoch MolPCBA budget — with the caveat that two hops also widen the
receptive field and add parameters, so "iteration" is not yet isolated from
"one extra hop". The follow-up should therefore be a *mechanism isolation*
(capacity-matched STATIC + one-layer arm), not a capacity expansion.

**5. If it fails, what would that refute, and what would stay open?** The round
did not fail, so this branch is counterfactual; recorded for the other outcome.
Had the delta stayed inside ±0.003 with a used dynamic branch, it would have
refuted only the specific claim "two hops of learned neighbour state improve
MolPCBA macro AP under this 40-epoch from-scratch recipe at width 128". It would
*not* have refuted (i) that iterative MP helps other datasets or budgets,
(ii) that MP helps under a longer schedule where W2 is undertrained,
(iii) that a differently-placed MP block (e.g. inside `local`, or with
down-weighting / degree normalisation) helps, or (iv) that the one-shot S1/S2
encoding is sufficient *for the specific target of macro AP on sparse tasks*. In
this round the message path itself was never the failure mode: gates passed,
gradients flowed, Rtotal grew, and the post-training interventions moved AP by
0.10–0.16, so a null result would have been a statement about the objective and
budget, not about whether the mechanism was active.

---

## Incidents, deviations and provenance

1. **Isolation tolerance calibrated before any formal step** (commit `ea7eec6b`).
   The first GPU gate attempt (`dyn-audit`, 32 s wall) failed the cross-molecule
   isolation check at 1.9e-6 against a 1e-6 threshold. The measured effect is
   float32 reduction-order noise, not a real dependency: with **0** cross-molecule
   edges found, the untouched molecule's `h0` and `h2` move by 4.8e-7 and its
   logits by 1.9e-6, versus 4.64 for the perturbed molecule's logits (a separation
   of ~2.4e6), and the same check is exactly 0.0 on CPU. The tolerance was set to
   1e-5, *justified by the measurement and recorded before any training step*; all
   nine gates then passed (`dyn-audit2`) with the identical measured noise. No
   model, data, LR, depth or decision rule changed; the failed 32 s allocation is
   charged in `analysis/gpu_ledger.json`.
2. **No post-hoc change** to the model, widths, LR, dropout, weight decay, epoch
   count, batch plan, thresholds or checkpoint rule after any Validation or Test
   number was seen. Three code changes after the run were analysis-only and touch
   no model, metric or stored number: a wrong dict key in the extreme-task helper,
   the addition of the robustness/decomposition block, and a task-id mapping fix
   in that block (array positions were printed where official task ids were meant;
   the corrected ids are 77/108/57/31/2/62/117/110/120/9 — see
   `analysis/comparison.json` → `robustness`).
3. **Provenance.** `run.json` records `git_commit = ea7eec6bdac3`, and
   `git_state.dirty_files` lists only **untracked output** — this round's
   `analysis/` directory and two pre-existing `lcrc_pcba_s3_v1/analysis/*failed.json`
   leftovers. No tracked source file differed from the deployed revision.
   Deterministic identities in `run.json`: W2 trunk fingerprint
   `5f7101a8…` (= the historical W2 seed-0 trunk), full plan
   `06f43729…` (= the historical W2 40-epoch plan), consumed-40 prefix
   `1a24dfca…`, feature-spec / norm / split / label hashes = the frozen sources.
4. **CPU tests:** `24 passed` (`tests/test_dy_model.py`), covering the exact
   parameter budget, the bitwise W2 part, the zero-residual identity, atom
   renumbering, bond-direction flip, molecule isolation, layer-input capture, the
   chain dependency, the independent numpy aggregation, the zero-bond graph, the
   shuffle invariants and the inherited recipe/plan/split identities.
5. **Test discipline.** No Test label existed in the training artifact (the
   loader refuses finite Test labels; the check is in `run.json`'s
   `train/label` hashes and in `tests/`). Test logits were written and sealed
   (`seal 5441073e…`, `test_labels_read: false`) *before* `dy_test.py score`
   opened `raw/graph-label.csv.gz` through the inherited
   `cc_data.read_official_test_labels` gate. The official Test row-id hash
   `9171aa54…` is identical to the frozen BOC artifact, proving the same rows
   were scored. Nothing was retrained, reselected or tuned afterwards.
6. **Budget.** Training 1.7340 GPU-hours (target ≤ 2.5, hard cap 3.0); all five
   allocations including the failed gate attempt total **1.7771 GPU-hours**
   (`analysis/gpu_ledger.json`, with each entry's wall duration). Peak allocated
   memory 235.6 MiB. One job at a time; GPU0 was occupied by another user's
   process throughout and was never touched (`res-gpu1` =
   `CUDA_VISIBLE_DEVICES=1` in every run).
7. **Profile fidelity — the projection was optimistic by 44%.** The 100-batch
   profile measured 37.0 ms/step (mean after 10 warmup steps, p50 35.4 ms) and
   projected 108.7 s/epoch → **1.2079 GPU-hours**. The realised run averaged
   **53.5 ms/step** (5,857.8 s train / 109,520 steps) and 1.7340 GPU-hours. The
   44% gap is not explained by the validation pass (7.3 s/epoch, which the
   projection did include); it is a step-time underestimation whose cause is not
   established with one run — plausibly the first 100 batches of epoch 1 are not
   representative of the epoch's size distribution, and/or co-tenant load on the
   host changed between the profile and the run. The outcome is unaffected (the
   projection bounded the budget by a wide margin and no optimisation was
   needed), but a future preflight should profile batches sampled across an epoch
   or measure one full epoch, and treat the 100-batch figure as a lower bound.

---

## Artifacts

| Path | Contents |
|---|---|
| `PROTOCOL.md` | pre-registration (frozen before the first step) |
| `README.md` | file map and the exact execution order |
| `code/dy_common.py` | frozen constants, inherited references, freeze seal |
| `code/dy_model.py` | `W2DynamicModel`, private init, parameter audit, usage metrics |
| `code/dy_audit.py` | nine pre-training gates + gradient audit + cost profile |
| `code/dy_train.py` | the single 40-epoch formal run |
| `code/dy_mechanism.py` | Validation-only `message_off` / `message_shuffle` |
| `code/dy_test.py` | Test-logit freeze and post-freeze official scoring |
| `code/dy_analyze.py` | comparison table, strata, robustness, decision label |
| `tests/test_dy_model.py` | 24 CPU correctness tests |
| `analysis/preflight.json` | gate evidence, gradient audit at steps 1/3/10/100, profile |
| `analysis/mechanism.json` | the three intervention modes per checkpoint |
| `analysis/selected_checkpoint.json` | the unique best-Valid selection (no Test input) |
| `analysis/test_prediction_manifest.json` | sealed Test-logit freeze (`5441073e…`) |
| `analysis/test_scores.json` | official + independent Test AP, comparisons, strata |
| `analysis/per_task_valid_test.csv` | per-task Validation and Test AP for all six predictions |
| `analysis/per_task_deltas.csv` | per-task Δ vs W2 / RC-REAL with support counts |
| `analysis/comparison.json` | full comparison + robustness + decision record |
| `analysis/report_tables.md` | the generated tables used in this report |
| `analysis/gpu_ledger.json` | all GPU stages including the failed gate attempt |
| `runs/W2_DYNAMIC/` (git-ignored) | checkpoints, curve, per-task AP, gradient history, RNG state |

Reproduction (after `rr pull`):

```bash
uv run pytest -q tracks/structural-sufficiency/lcrc_pcba_dynamic_v1/tests
uv run python tracks/structural-sufficiency/lcrc_pcba_dynamic_v1/code/dy_audit.py --device cuda
uv run python tracks/structural-sufficiency/lcrc_pcba_dynamic_v1/code/dy_train.py \
  --out-dir tracks/structural-sufficiency/lcrc_pcba_dynamic_v1/runs/W2_DYNAMIC \
  --soft-seconds 9000 --hard-seconds 10000
uv run python tracks/structural-sufficiency/lcrc_pcba_dynamic_v1/code/dy_mechanism.py --device cuda
uv run python tracks/structural-sufficiency/lcrc_pcba_dynamic_v1/code/dy_test.py freeze --device cuda
uv run python tracks/structural-sufficiency/lcrc_pcba_dynamic_v1/code/dy_test.py score
uv run python tracks/structural-sufficiency/lcrc_pcba_dynamic_v1/code/dy_analyze.py
```

## Stop

The single formal training run, the post-training Validation-only mechanism pass,
the Test freeze/score and this report are complete. **No STATIC control, no
one-layer or four-layer arm, no additional seed, no re-tuning and no next-round
experiment has been started, and none should be started without a new
pre-registration.**
