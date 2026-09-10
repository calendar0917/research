# Fast baseline-difficulty / typed-refinement redistribution audit

**Track:** ksvd · **Study:** zinc-context-gap · **Stage:** decision audit (no training)
**Module:** `experiments/luyin16/zinc_baseline_difficulty_typed_refinement_audit.py`
**Outputs:** `results/baseline_difficulty_typed_refinement_audit/`
**Claim/decision:** `records/claims/claim-baseline-difficulty-typed-refinement-20260910.yaml`,
`records/decisions/decision-baseline-difficulty-typed-refinement-20260910.yaml`

**Verdict: `D_UNEXPLAINED_REDISTRIBUTION` → NO architecture GO.**

The corrected exact typed tokenizer hurts historical *easy* molecules and helps
historical *hard* molecules, but the redistribution is **not** explained by either
of the two pre-registered mechanisms: it is not a variance-reducing-regularisation
effect (the *ensemble centre* moves, not just the seed spread), and typed-refinement
informativeness carries **no** independent signal for where the corrected tokenizer
helps (partial ρ ≈ −0.04; all seven informativeness metrics null-to-anti-predictive,
0–2/4 seeds). A coarse-to-fine architecture is therefore not authorised.

---

## 0. Executive summary

| test | question | result |
|---|---|---|
| **H1** | is the degradation locked to historical difficulty? | **yes** — quintile degradation `+0.031, +0.018, +0.012, +0.002, −0.040`; per-seed Spearman(baseline, degradation) `−0.22/−0.17/−0.27/−0.17` (4/4 negative). |
| **H1** | centre (bias) or spread (variance)? | **centre-dominated** — easy Δcentre-MAE `+0.0213` vs Δspread-MAE `+0.0125`; the 4-seed ensemble itself degrades on easy Q1 (`0.0245→0.0489`). Easy moves are systematic (wrong-direction 36.9 % + overshoot 28.6 %), not noise (magnitude-only 4.8 %). |
| **H2** | do corrected typed children of a historical token carry distinguishable train target/residual context? | **yes, statistically** (707 eligible parents of 2 363 split parents), **but it does not matter** — informativeness vs corrected gain ensemble ρ `−0.019`, partial given difficulty `−0.036`, 2/4 seeds; across all 7 metric variants partial ρ ∈ `[−0.069, −0.025]`. |
| **H3** | does the 2×2 isolate an informativeness effect? | **no** — hard+**low**-info gains most `+0.0252`; hard+high `+0.0129`; easy+high hurts most `−0.0268`. Interaction `−0.0080` (high-informativeness is weakly *anti*-predictive). |

→ **Case D**: the redistribution is a stable, difficulty-locked, **centre(bias)-dominated**
error shift that is not attributable to coarse variance-regularisation (A) nor to
selectively-useful fine typed information (B/C). No coarse-to-fine / coarse-prior
architecture is opened.

---

## 1. Data (all reused, nothing retrained, test never loaded)

* **Frozen paired validation predictions** — compact-v4-hinge historical / corrected,
  seeds 0–3 (same runs as `notes/corrected_token_fragmentation_rarity_audit.md`).
* **Historical → corrected train token split map** — from
  `results/typed_patch_tokenizer_correctness/occurrences.npz` (shared id space, refinement asserted).
* **Official-train molecule targets + historical compact-v4-hinge train OOF residuals**
  — `results/oof_difficulty_audit/oof_per_molecule.csv` (10 000 train molecules;
  `mean_signed_residual = target − prediction`).
* Official **test never loaded**; nothing trained; only load → join → compute → plot → decide.

---

## 2. H1 — difficulty quintiles, movement classes, centre/spread decomposition

**Table 1 — historical difficulty redistribution** (quintiles fixed once from
`mean historical |error|` across seeds):

| quintile | n | hist MAE | corr MAE | degradation | Δ disagreement | ensemble deg | Δcentre (MAE) | Δspread (MAE) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q1 (easiest) | 200 | 0.0407 | 0.0718 | **+0.0312** | +0.0217 | +0.0243 | +0.0243 | +0.0182 |
| Q2 | 200 | 0.0685 | 0.0865 | +0.0180 | +0.0064 | +0.0182 | +0.0182 | +0.0067 |
| Q3 | 200 | 0.0974 | 0.1090 | +0.0116 | −0.0018 | +0.0150 | +0.0150 | −0.0017 |
| Q4 | 200 | 0.1439 | 0.1459 | +0.0020 | +0.0005 | −0.0000 | −0.0000 | +0.0008 |
| Q5 (hardest) | 200 | 0.4969 | 0.4572 | **−0.0397** | −0.0088 | −0.0454 | −0.0454 | −0.0069 |

`degradation = corrected − historical |error|` (positive = corrected worse);
`Δcentre = |ensemble − y|` change; `Δspread = mean_s |pred_s − ensemble|` change.
The L1 centre/spread split is robust; the exact L2 telescoping split is reported in
the CSV but is outlier-dominated on ZINC's heavy target tail (Q5 Δcentre-MSE `+0.133`
even though the ensemble MAE *improves* — this is why the decision uses L1).

**Table 2 — movement classification** (per molecule × seed; % of pairs):

| quintile | helpful % | harmful % | wrong-direction % | overshoot % | magnitude-only % |
|---:|---:|---:|---:|---:|---:|
| Q1 | 34.5 | 65.5 | 36.9 | 28.6 | 4.8 |
| Q2 | 46.5 | 53.5 | 33.5 | 20.0 | 3.5 |
| Q3 | 47.8 | 52.3 | 36.6 | 15.6 | 5.0 |
| Q4 | 55.0 | 45.0 | 33.9 | 11.1 | 1.4 |
| Q5 | 61.3 | 38.8 | 35.6 | 3.1 | 2.3 |

Easy degradation is **systematic** (wrong-direction + overshoot ≈ 65 % of Q1 pairs),
not a small-magnitude disturbance; the harmful share falls monotonically with difficulty.

**Centre vs spread (H1 resolution).** On easy Q1/Q2 the ensemble (centre) component
dominates the seed-spread component, and the **ensemble itself degrades** (Q1 ensemble
MAE 0.0245 → 0.0489). The individual→ensemble shrink is only ~13 % on easy
(`+0.0213` vs `+0.0246`), not the "essentially unchanged ensemble" that a pure
variance/regularisation loss would give. On hard Q5 both components improve and the
centre improvement dominates. So the redistribution is a **prediction-centre (bias)
shift by difficulty**, with a secondary variance increase concentrated in Q1.

---

## 3. H2 — train-derived typed-refinement informativeness

Construction (train only): every historical r2 token `H` is split into corrected
children; for each child we take its **molecule-level** train support and the mean
train target / mean historical-OOF residual over the training molecules that contain it.
A parent is *eligible* when it has ≥2 children and ≥2 children with molecule support ≥ 5.
`child dispersion` = support-weighted std of child means (L2); a residual version
(historical OOF signed residual) is the pre-registered primary. Per validation molecule
we aggregate the dispersion of its patches (mean over split patches, `PRIMARY =
info_resid_mean_split`).

Coverage: 2 363 split parents, **707 eligible parents**; all 1 000 validation molecules
carry ≥1 split patch. High-informativeness threshold (train-derived) = 0.0476.

**Table 3 — typed informativeness groups** (validation tertiles of the primary metric):

| info group | n | hist MAE | corr MAE | corrected gain | Δ disagreement |
|---|---:|---:|---:|---:|---:|
| low | 333 | 0.1446 | 0.1477 | −0.0031 | +0.0066 |
| mid | 333 | 0.1315 | 0.1348 | −0.0033 | +0.0034 |
| high | 334 | 0.2321 | 0.2395 | **−0.0074** | +0.0008 |

Corrected tokens lose in **every** informativeness group, and lose *more* in the
high-informativeness group — the opposite of "fine typed distinctions are selectively useful".

**Table 4 — independent association** (corrected gain = hist |err| − corr |err|, positive = corrected better):

| metric | value |
|---|---:|
| corr(informativeness, gain) — ensemble | −0.019 |
| corr(informativeness, gain) — seed 0 / 1 / 2 / 3 | +0.002 / +0.018 / −0.019 / −0.004 |
| partial corr given baseline difficulty | **−0.036** |
| partial corr given frequency | −0.018 |
| seeds same-sign (positive) | 2 / 4 |

Robustness across the whole metric family (`informativeness_metric_robustness.csv`):
`info_resid_mean / _max / _mean_split`, `info_target_mean / _max / _mean_split`,
`frac_high_info_patches` → partial-given-difficulty ∈ `[−0.069, −0.025]`,
seeds-positive 0–2 / 4. **The H2 null is not a metric choice.**

---

## 4. H3 — 2×2 difficulty × typed informativeness

**Table 5 — mechanism 2×2** (easy = Q1∪Q2, hard = Q4∪Q5; low/high = median split of the primary metric):

| difficulty | typed info | n | hist MAE | corr MAE | gain | Δ disagreement |
|---|---|---:|---:|---:|---:|---:|
| easy | low | 209 | 0.0554 | 0.0780 | −0.0226 | +0.0111 |
| easy | high | 191 | 0.0537 | 0.0804 | **−0.0268** | +0.0173 |
| hard | low | 193 | 0.2541 | 0.2290 | **+0.0252** | +0.0064 |
| hard | high | 207 | 0.3821 | 0.3692 | +0.0129 | −0.0139 |

The gain is ordered by **difficulty**, not informativeness: the corrected tokenizer
helps hard molecules (most when informativeness is *low*) and hurts easy molecules
(most when informativeness is *high*). Interaction `(hard,high − hard,low) −
(easy,high − easy,low) = −0.0080`. The expected "hard + high-info is the strongest
candidate for exact typed benefit" pattern is **absent**.

---

## 5. Figures

| file | content |
|---|---|
| `figures/fig1_difficulty_degradation.png` | degradation vs difficulty quintile (individual vs 4-seed ensemble) |
| `figures/fig2_difficulty_disagreement.png` | historical vs corrected seed disagreement by quintile |
| `figures/fig3_informativeness_gain.png` | typed informativeness quintile vs corrected gain |
| `figures/fig4_mechanism_2x2.png` | 2×2 easy/hard × low/high-info gain |

---

## 6. Interpretation

1. **Difficulty-locked, stable.** The sign of the degradation ladder is identical for
   all 4 seeds; the corrected representation systematically shifts the prediction
   *centre* down on easy molecules and up on hard molecules.
2. **Centre (bias), not variance.** Easy degradation is ~63 % centre of the L1 decomposition; the ensemble
   itself degrades on the easiest bin. This rejects the operational "variance-reducing
   regularisation" test for Case A (ensemble degradation is ~87 % of the individual
   degradation on easy, not ≈0).
3. **No fine-information signal.** The corrected exact children of a historical token
   *do* carry heterogeneous train target/residual context, but that heterogeneity has
   no independent relation to where the corrected tokenizer gains (partial ρ ≈ −0.04,
   all variants null/negative, ≤2/4 seeds). Fine typed refinement is therefore not
   *selectively* useful at the molecule level here.
4. **Frequency/topology/tier are not the axis** — frequency control leaves partial ρ
   unchanged (`−0.018`); the prior topology control (`|ρ| ≤ 0.034`) and tier
   non-explanatoriness are inherited unchanged.

Because the redistribution is a *centre* effect that none of the three pre-registered
mechanisms captures, the branch is closed: no coarse-to-fine, no coarse shared prior.

---

## 7. Answers to Q1–Q10

**Q1. Is the corrected degradation concentrated in historical easiest molecules?**
Yes. Quintile degradation `+0.0312 (Q1), +0.0180, +0.0116, +0.0020, −0.0397 (Q5)`;
per-seed Spearman(baseline error, degradation) is negative 4/4
(`−0.218, −0.167, −0.268, −0.172`).

**Q2. Is easy-group degradation from a worse prediction centre or from seed variance?**
Predominantly the **centre** (bias): easy Δcentre-MAE `+0.0213` vs Δspread-MAE `+0.0125`,
and the 4-seed ensemble MAE itself degrades on Q1 (`0.0245→0.0489`). There is a
secondary variance increase concentrated in Q1 (spread `+0.0182`), and the movements
are systematic (wrong-direction 36.9 % + overshoot 28.6 %), not noise (magnitude-only 4.8 %).

**Q3. Is the hard-group improvement from bias reduction or variance reduction?**
**Bias (centre) reduction**: hard Δcentre-MAE `−0.0227` vs Δspread-MAE `−0.0030`;
the ensemble error falls on hard quintiles.

**Q4. Do the corrected typed children inside a historical token carry stable train-derived target heterogeneity?**
Yes, statistically: 707 eligible parents (of 2 363 split parents) show support-weighted
child target/residual dispersion above a train-derived threshold. The heterogeneity is
real but (see Q5) uninformative for the corrected gain.

**Q5. Does typed-refinement informativeness predict the corrected tokenizer's gain?**
No. Ensemble ρ `−0.019` (seeds `+0.002/+0.018/−0.019/−0.004`); all seven informativeness
metrics are null-to-weakly-negative.

**Q6. Does the relation survive controlling historical baseline difficulty?**
No. Partial ρ given baseline difficulty `−0.036` (range `[−0.069, −0.025]` across
metrics); given frequency `−0.018`.

**Q7. Is hard + high-informativeness the largest corrected-benefit group?**
No. Hard + **low**-info gains `+0.0252` > hard + high-info `+0.0129`. High
informativeness is weakly *anti*-predictive.

**Q8. Is easy + low-informativeness the largest historical-benefit group?**
It is the least-affected easy cell (easy+low gain `−0.0226` vs easy+high `−0.0268`),
consistent with the historical coarse token being relatively better there — but the
driver is difficulty, not informativeness (Q7).

**Q9. Which does the evidence support?**
**D — unexplained redistribution** (with the precise form: stable, difficulty-locked,
centre/bias-dominated). B and C are ruled out by the informativeness null; A is ruled
out because the ensemble itself degrades on easy molecules (a variance-reducing
regularisation loss would leave the ensemble ~unchanged).

**Q10. Single recommended next stage?**
**Stop the tokenizer-derived architecture branch.** Do not design coarse-to-fine or a
coarse shared prior on this evidence. The only residual question (why the exact typed
representation shifts the prediction centre by difficulty) is a representation-geometry
diagnostic, not a tokenizer-architecture direction, and is *not* opened here.

---

## 8. Decision gates (pre-registered, applied literally)

```
hurts_easy                      True   (easy deg +0.0246 > hard deg -0.0188)
easy_disagreement_increases     True   (Q1 spread +0.0217 on 0.0385)
ensemble_degradation_smaller_easy False (0.0213 vs 0.0246 = 87%, not ~0)   <-- A fails
easy_variance_dominated         False  (centre > spread)
typed_info_weak_after_control   True   (partial -0.036)
typed_info_strong_seed_consistent False (partial < 0.20)                    <-- B fails
typed_info_moderate_seed_consistent False (partial < 0.15)                  <-- C fails
high_info_group_benefits        False  (gain high -0.0074 < low -0.0031)
low_info_group_no_gain          True
hard_high_largest_gain          False  (hard+low > hard+high)
clear_2x2_interaction           False  (interaction -0.0080)
=> Case D: NO ARCHITECTURE GO.
```

Tracked records: `records/claims/claim-baseline-difficulty-typed-refinement-20260910.yaml`,
`records/decisions/decision-baseline-difficulty-typed-refinement-20260910.yaml`.
