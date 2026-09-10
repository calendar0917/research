# Corrected Token Fragmentation & Rarity Audit (ZINC, official train/valid)

**Status:** FINAL — 2026-09-10
**Phase:** mechanism audit of a one-variable tokenizer intervention (diagnosis only; **no training**, **no architecture**)
**Questions answered:** Q1–Q16 · Tables A–F · Figures 1–8
**Module:** `experiments/luyin16/zinc_corrected_token_fragmentation_audit.py`
**Artifacts:** `results/corrected_token_fragmentation_audit/` (`audit_summary.json`, `historical_to_corrected_token_map.csv`, `r2_token_fragmentation.csv`, `parent_token_fragmentation.csv`, `validation_occurrence_transitions.csv`, `validation_molecule_fragmentation.csv`, `per_seed_degradation.csv`, `rarity_comparison.csv`, `disagreement_comparison.csv`, `group_transition_results.csv`, `partial_analysis.json`, `decision_record.json`, `figures/`)
**Decision:** **D — MIXED MECHANISM.** The corrected-vs-historical degradation is **not directed by token fragmentation** (all unconditional molecule-level Spearman |ρ| ≤ 0.06; frequency loss is even mildly wrong-signed). It is dominated by a **baseline-difficulty-dependent error redistribution** (corrected worse on the easy bulk, better on the hard tail; Spearman(baseline error, degradation) = **−0.352**). A **weak, largely bulk-localised, non-specific** frequency-loss / newly-rare signal remains after conditioning on baseline difficulty (partial ρ ≈ 0.10; G3 matched +0.014). Newly-OOV molecules do **not** degrade, and fragmentation does **not** raise cross-seed disagreement. **No architecture GO.**

---

## 0. Executive summary

| Fact | Value |
|---|---|
| Frozen paired models | compact-v4-hinge, historical vs corrected typed tokenizer, seeds 0–3 (8 promoted runs, bit-verified) |
| Test split | **never loaded** (train + validation only) |
| Sign convention | `delta_error = err_corrected − err_historical`; **positive = corrected worse** |
| Overall valid | 0.169451 → 0.174062, mean paired degradation **+0.00461** (per seed +0.0081/+0.0111/−0.0052/+0.0045) |
| r2 fragmentation | 6,784 historical → 15,218 corrected types; **2,363 (34.8%) split**, mean multiplicity **2.24**, max **104**, mean per-child dilution **11.5×** |
| parent fragmentation | 31 → 512 types; **23 (74%) split**, mean multiplicity **16.5**, max **98**, mean dilution **1927×** |
| Validation transitions (occurrences) | stable-common 20,761 · newly-rare **1,065** · newly-OOV **396** · already-rare 583 · hist-OOV 278 |
| Molecules affected | newly-rare 565 · newly-OOV 240 · either **631 / 1,000** |
| `frequency_loss` vs degradation | per-seed ρ +0.01/+0.05/−0.07/−0.03 (2/4 positive); pooled **−0.025** |
| `fraction_newly_rare` vs degradation | per-seed −0.04/+0.06/−0.03/−0.05 (1/4); pooled **−0.023** |
| `fraction_newly_oov` vs degradation | per-seed −0.07/+0.01/−0.06/−0.01 (1/4); pooled **−0.056** |
| Difficulty-conditioned partial ρ (best of the three) | **+0.10** (newly-rare; 4/4 seeds positive) |
| Baseline difficulty vs degradation | **−0.352** (robust 4/4 seeds; not a within-model seed artefact) |
| Group degradation G1/G2/G3/G4 | **+0.0016 / +0.0042 / +0.0141 / −0.0083** |
| Matched (difficulty-stratified) group effect | G3 **+0.0143** · G4 **−0.0089** · G2 −0.017 · G1 −0.008 |
| Rarity ↔ \|error\| Spearman | historical rare≤5 **0.305** / OOV **0.349** → corrected rare≤5 **0.322** / OOV **0.372** (marginally *stronger*, +0.02) |
| Cross-seed disagreement | hist 0.0809 → corr 0.0845 (Δ **+0.0036**); vs degradation **+0.31**; **but not correlated with any fragmentation metric** (\|ρ\| ≤ 0.07) |
| HybridEmbedding tier reallocation | 16.6% of validation occurrences full→low-rank; collinear with frequency loss (ρ **0.52**); vs degradation ρ **0.025** → no independent signal |

**Bottom line.** The corrected tokenizer fragments the vocabulary massively and dilutes exact-token statistics exactly as the hypothesis says. But the *paired molecule-level degradation* is essentially uncorrelated with every fragmentation metric, and what signal exists is a **baseline-difficulty redistribution** (the corrected model trades bulk accuracy for tail accuracy), not a fragmentation dose-response. The statistical-sharing-loss explanation is therefore **not supported at the level required**; the hypothesis survives only as a weak, difficulty-conditional, non-specific residual.

---

## 1. Motivation

The typed-patch-tokenizer correctness repair (`notes/typed_patch_tokenizer_correctness_repair.md`) showed that the historical `pynauty.certificate` token was a coarse rooted-topology token, that the corrected `typedpatchkey/v2` key is a complete invariant (0 gate failures), and that the correction **slightly regresses** compact-v4-hinge (mean valid +0.0046 worse; test 0.136885 → 0.140895). Correctness cannot be rolled back. The open, pre-registered hypothesis is:

> The historical collision accidentally created hard parameter sharing. Correct tokenization removes that sharing, fragments the vocabulary, reduces observations per exact token, and raises estimation variance — which is the observed regression.

This stage tests **only** that mechanism. It reuses the frozen 4-seed historical and corrected compact-v4-hinge runs and the frozen tokenizer occurrence tables. It trains nothing, adds no loss/optimizer/embedding, and never loads the official test split. All frequency statistics are computed from **official train only**; validation never contributes to any frequency definition.

## 2. Frozen paired models

Promoted, reproducible v4-hinge runs (seed 3 corrected uses the reproducible terminal trajectory; the discarded scratch run is never referenced):

| family | seed | run id | valid MAE | params | best epoch | predictions sha256 (16) | state sha256 (16) |
|---|---:|---|---:|---:|---:|---|---|
| historical | 0 | `20260909-194445-182c7021` | 0.170066 | 99,613 | 53 | `7204577303d634fb` | `fe208ec6544527a0` |
| historical | 1 | `20260909-200320-34b347bf` | 0.163167 | 99,613 | 48 | `b34c1e41e38e238d` | `fa444f5e2ed3e7d9` |
| historical | 2 | `20260909-201918-45fbe48d` | 0.174149 | 99,613 | 56 | `e62549b796555626` | `88d163a97d1277f7` |
| historical | 3 | `20260909-203516-04e62a28` | 0.170421 | 99,613 | 60 | `9ff0d54a44c106c4` | `173da2afab6355a6` |
| corrected | 0 | `20260910-111954-64845bd9` | 0.178158 | 107,201 | 44 | `1f92c27951aff9b5` | `7ac10f9439a45548` |
| corrected | 1 | `20260910-112530-c02ac7f0` | 0.174228 | 107,201 | 48 | `9db352cabb165102` | `02c66e898a1286e6` |
| corrected | 2 | `20260910-113058-a40ee3e1` | 0.168984 | 107,201 | 51 | `8695f761d2b24221` | `45ec67971cfa4549` |
| corrected | 3 | `20260910-123241-6067c9ac` | 0.174880 | 107,201 | 54 | `f374933266db86dc` | `e523941eb2e346ce` |

Per-run `predictions_sha256` and checkpoint `state_sha256` are recorded in `audit_summary.json → frozen_runs`. Validation targets are identical across all eight runs (max abs diff 0.0) and match the frozen post-v4 table.

## 3. Token split map

For every historical token we record its corrected child classes from **official-train occurrences** (`historical_to_corrected_token_map.csv`, one row per historical token per level). The corrected key is a refinement of the historical key (checked: every corrected class maps to exactly one historical class, 0 violations).

**Table A — token fragmentation**

| metric | r2 (radius-2 patch) | parent (radius-1) |
|---|---:|---:|
| historical tokens | 6,784 | 31 |
| corrected tokens | 15,218 | 512 |
| split historical tokens (k>1) | 2,363 (34.8%) | 23 (74.2%) |
| mean multiplicity k | **2.24** | **16.52** |
| max multiplicity k | 104 | 98 |
| mean per-child frequency dilution | **11.5×** | **1927×** |

Example row: historical token `h…` had old_count 24 and split into children 9/6/5/4 → k = 4. The parent tokenizer is *more* fragmented per token, but it has only 31 historical types, so the radius-2 vocabulary is where the 2.24× type explosion happens.

## 4. Frequency dilution definition

For a corrected child occurrence `C` of historical token `H`:

```
dilution_ratio            = historical_count(H) / corrected_count(C)
log_dilution              = log(1 + historical_count(H)) − log(1 + corrected_count(C))
child_frequency_fraction  = corrected_count(C) / historical_count(H)
split_entropy(H)          = − Σ_j p_j log p_j ,  p_j = child_count_j / old_count(H)
```

A large dilution means the historical bucket *looked* data-rich but the true typed identity is data-poor. Per-child dilution is stored in `r2_token_fragmentation.csv` / `parent_token_fragmentation.csv`; molecule-level means/maxes in `validation_molecule_fragmentation.csv`. (Note: the *mean* dilution over children of a token equals its split multiplicity only when children are equal-sized; we report the true per-child means.)

## 5. Newly rare / newly OOV definitions

Validation occurrence transition (hist freq / corr freq, both from official train):

| class | rule | occurrences | molecules |
|---|---|---:|---:|
| stable common | hist > 5 and corr > 5 | 20,761 | 1,000 |
| newly rare | hist > 5 and 1 ≤ corr ≤ 5 | **1,065** | **565** |
| newly OOV | hist > 0 and corr = 0 | **396** | **240** |
| already rare | 1 ≤ hist ≤ 5 and 1 ≤ corr ≤ 5 | 583 | 374 |
| historical OOV | hist = 0 | 278 | 162 |

Union of newly-rare / newly-OOV molecules: **631 / 1,000**. This is the natural experiment: patches the historical tokenizer showed as non-OOV are exposed as genuinely unseen exact typed keys.

## 6. r2 vs parent fragmentation

Both levels fragment, but neither explains degradation:

| metric | r2 pooled ρ | r2 partial\|difficulty | parent pooled ρ | parent partial\|difficulty |
|---|---:|---:|---:|---:|
| frequency loss | −0.025 | +0.045 | −0.023 | +0.075 |
| newly rare fraction | −0.023 | +0.080 | −0.046 | +0.009 |
| newly OOV fraction | −0.056 | +0.051 | −0.018 | +0.034 |
| split multiplicity | +0.002 | −0.056 | +0.044 | +0.010 |

The parent tokenizer is not the driver: its conditional partials are ≤ 0.08 with inconsistent per-seed signs. The radius-2 signal is the (still weak) larger one. **There is no "parent-only" effect and no meaningful r2+parent increment** — the two levels measure the same underlying fragmentation and both are uncorrelated with the paired degradation.

## 7. Paired degradation analysis

`per_seed_degradation.csv` holds per-molecule `err_hist`, `err_corr`, `delta = err_corr − err_hist` for all four seeds.

**Table B — fragmentation ↔ degradation (Spearman, `delta_error`)**

| metric | seed0 | seed1 | seed2 | seed3 | mean | pooled (seed-avg) | condition. partial |
|---|---:|---:|---:|---:|---:|---:|---:|
| frequency loss | +0.009 | +0.052 | −0.073 | −0.028 | −0.010 | **−0.025** | +0.061 |
| mean log dilution | +0.009 | +0.052 | −0.073 | −0.028 | −0.010 | −0.025 | +0.061 |
| newly-rare ratio | −0.036 | +0.057 | −0.029 | −0.048 | −0.014 | −0.023 | +0.096 |
| newly-OOV ratio | −0.072 | +0.005 | −0.060 | −0.014 | −0.035 | −0.056 | +0.084 |
| split multiplicity | −0.005 | −0.022 | +0.049 | +0.008 | +0.007 | +0.002 | −0.057 |
| split-common fraction | +0.043 | −0.055 | +0.084 | +0.038 | +0.028 | +0.038 | **−0.138** |

Unconditional molecule-level fragmentation does **not** predict degradation (|ρ| ≤ 0.06, no sign consistency). Conditioning on the molecule's *baseline (historical) error* turns the frequency-loss/newly-rare/newly-OOV partials mildly positive and sign-consistent (4/4 seeds) but still **≤ 0.10**, while the fragmentation-*severity* metrics (split-common, split≥3) become negative. This is the whole story in one table: the apparent sign of the fragmentation signal depends entirely on whether baseline difficulty is controlled, and even then it is small.

**Quintile ladder (frequency loss, Figure 3)** — no monotone gradient: Q1→Q5 degradation = +0.0022, +0.0099, +0.0044, +0.0198, −0.0132.

## 8. Rarity–difficulty comparison

Train-only rarity (Figure 6, Table below): the corrected tokenizer makes the tail heavier (rare≤1 2,658 → 6,598; mean log-freq 1.65 → 1.48) and roughly triples validation OOV occurrences (278 → 674 by the shared-id count, 1,163 in the earlier audit).

| family | rare≤1 | rare≤2 | rare≤5 | rare≤10 | OOV | min | mean log-freq |
|---|---:|---:|---:|---:|---:|---:|---:|
| historical | 2,658 | 3,652 | 4,745 | 5,429 | — | 1 | 1.649 |
| corrected | 6,598 | 8,905 | 11,417 | 12,794 | — | 1 | 1.479 |

Rarity ↔ \|error\| (seed-averaged |residual|):

| family | rare≤5 ρ | OOV ρ |
|---|---:|---:|
| historical | 0.305 | 0.349 |
| corrected | **0.322** | **0.372** |

The rarity–difficulty relation is **marginally stronger** under the corrected tokenizer (+0.017 / +0.023) but the effect is tiny — not the "clearly steeper ladder / larger OOV penalty" that §18 pre-registered as support.

## 9. Cross-seed disagreement

"Disagreement" = std over seeds of the per-molecule prediction.

| scope | historical | corrected | Δ |
|---|---:|---:|---:|
| overall | 0.08086 | 0.08447 | **+0.00361** |
| G1 stable | 0.07018 | 0.07835 | +0.00817 |
| G2 split-common | 0.05133 | 0.05800 | +0.00666 |
| G3 newly rare | 0.08257 | 0.08444 | +0.00187 |
| G4 newly OOV | 0.10910 | 0.10970 | +0.00060 |

The corrected tokenizer raises disagreement slightly, but **not where fragmentation is worst** — Δ is *largest* for the least-fragmented G1 and smallest for G4. Correlating Δdisagreement with fragmentation metrics gives |ρ| ≤ 0.07 for everything (frequency loss −0.057, newly-rare −0.017, newly-OOV −0.040). Δdisagreement *does* correlate with degradation (+0.307; partial +0.263), so estimation variance is real — but it is **not produced by tokenization fragmentation**.

## 10. Group A ordinary control

Repeating the core analysis inside the frozen Group A (no long-cycle, n=965):

| metric | all valid | Group A |
|---|---:|---:|
| frequency-loss ρ | −0.025 | −0.025 |
| newly-rare ρ | −0.023 | −0.017 |
| newly-OOV ρ | −0.056 | −0.043 |
| G3 degradation | +0.0141 | **+0.0054** |
| G4 degradation | −0.0083 | −0.0082 |

The G1<G2<G3 ordering is only weakly present in Group A and the already-weak G3 effect roughly *halves*. The relations do **not** cleanly survive the topology control — further evidence the signal is not a robust independent local-representation effect.

### 10b. Is fragmentation just a proxy for long-cycle topology? (spec §32)

Direct check: Spearman of each fragmentation feature with topology severity (subgroup order A<B<C) and with the continuous `label_excess`, plus the hard-topology (B/C) share in the lowest vs. highest fragmentation quintile:

| feature | ρ vs subgroup severity | ρ vs `label_excess` | B/C share, lowest Q | B/C share, highest Q |
|---|---:|---:|---:|---:|
| frequency loss | −0.034 | −0.034 | 3.5% | 2.5% |
| mean log dilution | −0.034 | −0.034 | 3.5% | 2.5% |
| newly-OOV | +0.005 | +0.005 | 5.5% | 3.5% |
| newly-rare | +0.007 | +0.007 | 4.5% | 3.5% |
| split burden | −0.016 | −0.016 | 4.5% | 3.5% |
| split-common | +0.011 | +0.011 | 2.5% | 4.5% |

All |ρ| ≤ 0.034 and the hard-topology share is flat/near-flat across fragmentation quintiles, so fragmentation is **not** a proxy for long-cycle topology; fragmentation and difficulty are separate axes (the difficulty ladder of §13 is therefore *not* just the B/C long-cycle tail).

## 11. HybridEmbedding tier analysis

With `hybrid_full_typed_tokens = 768`, a token id < 768 uses a full row, else a rank-4→16 factorised row. Under the corrected vocabulary the 768-row boundary reallocates tiers:

* 16.6% of validation occurrences move full → low-rank; 2.6% move low-rank → full.
* Mean per-molecule downgrade fraction 0.173.
* Downgrade fraction vs degradation: ρ = **+0.025** (no signal); partial given frequency loss +0.044.
* Downgrade fraction vs frequency loss: ρ = **+0.52** (collinear — the two are the same phenomenon).

Tier reallocation therefore carries **no independent explanatory signal**: whatever it captures is already captured by frequency loss, and neither explains degradation. It is **not** the mechanism (fails the §24 gate).

## 12. Partial / incremental analysis

Standardised OLS on the seed-averaged paired degradation (`partial_analysis.json`):

| model | R² | key coefficient(s) |
|---|---:|---|
| A: degradation ~ alias(split) | 0.003 | +0.058 |
| B: degradation ~ frequency_loss | 0.007 | −0.083 |
| C: ~ alias + frequency_loss | 0.011 | +0.067 / −0.090 |
| D: ~ frequency_loss + graph_size | 0.009 | −0.060 / +0.049 |
| E: ~ frequency_loss + Δdisagreement | 0.053 | −0.097 / −0.215 |
| F: ~ frequency_loss + baseline difficulty | 0.022 | −0.087 / +0.121 |
| G: ~ newly_rare + baseline difficulty | 0.015 | −0.030 / +0.121 |
| H: ~ newly_OOV + baseline difficulty | 0.046 | −0.179 / +0.135 |

* **Alias burden does not survive frequency loss**: controlling for frequency loss leaves the alias coefficient essentially unchanged and tiny (both ≈ 0), i.e. neither the "wrongness of the old tokenizer" nor "the statistical support lost" explains degradation at the required level. The §21 prediction that alias signal would collapse while frequency loss survives is **not** observed because *both* are weak.
* Graph size and delta-disagreement do not rescue the frequency-loss term.
* The **baseline-difficulty** term dominates every model it enters (standardised +0.12–0.14), and `Spearman(baseline error, degradation) = −0.352` even after partialling frequency loss (−0.354).

## 13. Interpretation — two candidate explanations

**Explanation A — statistical sharing loss.** Expected: fragmentation positively correlated with degradation, newly-rare/OOV clearly worse, disagreement rising with fragmentation, high-frequency-loss group worst. Observed: **mostly absent**. Correlations ≈ 0 (often wrong sign), newly-OOV molecules are *better* under the corrected tokenizer, disagreement rises most for the *least* fragmented molecules. Only a weak difficulty-conditional, non-specific newly-rare signal (partial ρ ≈ 0.10; G3 matched +0.014) survives.

**Explanation B — unrelated model-level redistribution.** Expected: degradation not explained by fragmentation, driven by a global shift. Observed: **strongly supported**. `Spearman(baseline error, degradation) = −0.352`; degradation by baseline-error quintile is perfectly monotone (+0.031, +0.018, +0.012, +0.002, −0.040). The corrected tokenizer is *worse on the easy/common bulk* and *better on the hard/rare tail*, 4/4 seeds in each extreme quintile. Within-model seed-pair deltas do not reproduce this (ρ ≈ −0.01…−0.11), so it is a genuine, systematic historical-vs-corrected difference — not sampling noise and not a pure change-score artefact.

## 14. Architecture implication

The pre-registered next architecture (**hierarchical coarse-shared + corrected-exact-residual**) was gated on "frequency dilution, newly-rare/OOV transitions, or coarse-abstraction loss quantitatively explain the degradation". None clears its bar:

* frequency dilution / newly-rare / newly-OOV: pooled |ρ| ≤ 0.06 (conditionally ≤ 0.10) — far below the 0.25 strong / 0.15 moderate gates;
* split-but-still-common does **not** degrade more than the less-fragmented common control after difficulty matching (matched Δ = −0.0027) → the "coarse abstraction is intrinsically useful" reading is not established either;
* tier reallocation is collinear with frequency loss and carries no independent signal.

What *is* established is that correctness redistributes error from the hard tail to the easy bulk. That is closer to "the coarse historical token behaved as a bulk-favouring regulariser" than to "correctness destroyed useful statistical sharing", but it is a **global, difficulty-dependent effect, not a fragmentation dose-response**, and it is not specific to the tokens that actually fragmented.

## 15. Final decision

**D — MIXED MECHANISM**, ranked by effect size:

1. **Baseline-difficulty-dependent error redistribution** (dominant; *not* a fragmentation mechanism): ρ = −0.352, robust 4/4 seeds.
2. **Weak, largely bulk-localised, non-specific frequency-loss / newly-rare sharing signal**: conditional partial ρ ≈ 0.10; G3 matched +0.014 with no clean counterpart in newly-OOV.
3. **Newly-OOV sharing loss**: absent / wrong direction (G4 matched −0.009).
4. **HybridEmbedding tier reallocation**: no independent signal (collinear with frequency loss).

The strong and moderate sharing-loss gates are **not met**. **No architecture GO:** do not build coarse-shared + exact-residual on this evidence. The single recommended next stage is to *quantify and separate* the baseline-difficulty redistribution from the weak fragmentation residual (a diagnostic, no new training), rather than to open a hierarchical embedding.

---

## Tables D–F (cross-seed views)

**Table D — fragmentation ↔ degradation, per seed** (positive = corrected worse as token fragmentation rises)

| metric | seed0 | seed1 | seed2 | seed3 | mean | pooled |
|---|---:|---:|---:|---:|---:|---:|
| frequency loss | +0.009 | +0.052 | −0.073 | −0.028 | −0.010 | −0.025 |
| newly-rare ratio | −0.036 | +0.057 | −0.029 | −0.048 | −0.014 | −0.023 |
| newly-OOV ratio | −0.072 | +0.005 | −0.060 | −0.014 | −0.035 | −0.056 |
| split multiplicity | −0.005 | −0.022 | +0.049 | +0.008 | +0.007 | +0.002 |

**Table E — disagreement by group** (Δ = corrected − historical)

| group | hist | corr | Δ |
|---|---:|---:|---:|
| G1 | 0.07018 | 0.07835 | +0.00817 |
| G2 | 0.05133 | 0.05800 | +0.00666 |
| G3 | 0.08257 | 0.08444 | +0.00187 |
| G4 | 0.10910 | 0.10970 | +0.00060 |

**Table F — Group A ordinary vs all valid**

| metric | all valid | Group A |
|---|---:|---:|
| freq-loss vs degradation | −0.025 | −0.025 |
| newly-rare vs degradation | −0.023 | −0.017 |
| newly-OOV vs degradation | −0.056 | −0.043 |
| G3 degradation | +0.0141 | +0.0054 |
| G4 degradation | −0.0083 | −0.0082 |

## Figures

| file | content |
|---|---|
| `fig1_frequency_split.png` | historical token frequency → corrected child frequency (red = largest child) |
| `fig2_split_multiplicity.png` | r2 and parent split-multiplicity distributions |
| `fig3_freq_loss_quintile.png` | frequency-loss quintile vs corrected degradation (no monotone gradient) |
| `fig4_group_mae.png` | G1/G2/G3/G4 historical vs corrected MAE |
| `fig5_fragmentation_disagreement.png` | fragmentation quintile vs Δdisagreement |
| `fig6_rarity_ladder.png` | historical vs corrected rarity–MAE ladder |
| `fig7_groupA_freq_loss.png` | Group A frequency loss vs degradation |
| `fig8_difficulty_redistribution.png` | degradation by baseline-difficulty quintile (the dominant effect) |

## Case inspection

Selected purely by paired delta (`case_inspection` in `audit_summary.json`); thresholds never re-tuned from cases. Both the top-10 regressions and the top-10 improvements carry high frequency loss and high split multiplicity (e.g. top regression `valid:0214`, `fl=1.66`, `k=24.9`; top improvement `valid:0249`, `fl=2.93`, `k=14.9`). The extremes are *not* separated by fragmentation — consistent with the aggregate finding that severe cases are not cleanly fragmentation-driven.

---

## Q1–Q16

**Q1. On average, into how many true typed classes does the corrected tokenizer split a historical token?**
Radius-2: mean split multiplicity **2.24** (max 104) across the 6,784 historical tokens; the 2,363 aliased tokens split into 2–104 classes. The parent token splits much more per token (mean 16.5, max 98) but from only 31 historical types.

**Q2. Where does the main fragmentation occur — radius-2 or parent?**
Both; the radius-2 level is where the *vocabulary* explodes (6,784→15,218; 34.8% of tokens split), the parent level is where the *per-token* multiplicity is extreme (16.5, dilution ~1,927×). For explaining degradation neither helps (see Q6/Q12); the parent partials are the weaker.

**Q3. How many validation patches go from historical-common to corrected-rare?**
**1,065** occurrences (5.0% of 23,083) across **565** molecules.

**Q4. How many become newly OOV?**
**396** occurrences (1.9%) across **240** molecules.

**Q5. How many molecules are affected by newly-rare / newly-OOV?**
newly-rare **565**, newly-OOV **240**, union **631 / 1,000**.

**Q6. Spearman between frequency loss and corrected degradation?**
Per-seed +0.009 / +0.052 / −0.073 / −0.028 (mean −0.010); pooled **−0.025**. All fragmentation metrics pooled |ρ| ≤ 0.06.

**Q7. Is the relation ≥3/4 seeds same direction?**
**No** for the pre-registered unconditional relation (2/4 for frequency loss, 1/4 for newly-rare/OOV). Conditioning on baseline difficulty makes it 4/4 but the magnitude stays ≤ 0.10.

**Q8. Do G1/G2/G3/G4 show a degradation gradient?**
Partially: +0.0016 / +0.0042 / +0.0141 / **−0.0083**. G1<G2<G3 holds, but **G4 reverses** (newly-OOV molecules improve). Difficulty-matched, G3 = +0.0143 but G4 = −0.0089.

**Q9. Do split-but-still-common molecules also clearly degrade?**
No. G2 degradation +0.0042 (vs G1 +0.0016); after difficulty matching the split-common-vs-less-fragmented-common difference is −0.0027 (essentially zero, slightly negative). The coarse-abstraction ("Case B") reading is **not** supported.

**Q10. Is the corrected rarity↔\|error\| relationship stronger or weaker than historical?**
Marginally **stronger**: rare≤5 0.305→0.322, OOV 0.349→0.372 (≈ +0.02). Too small to count as support.

**Q11. Does fragmentation correspond to rising cross-seed disagreement?**
No. Corrected disagreement is slightly higher overall (Δ +0.0036) but is **largest on the least-fragmented G1** and uncorrelated with every fragmentation metric (|ρ| ≤ 0.07). Δdisagreement itself correlates with degradation (+0.31), but it is not fragmentation-driven.

**Q12. Do these relations still hold in Group A ordinary molecules?**
No stronger — the frequency-loss/newly-rare correlations are the same (~0) and the G3 effect roughly halves (+0.0141 → +0.0054). Not a robust independent local-representation effect.

**Q13. How much regression does HybridEmbedding tier reallocation explain?**
Essentially none. 16.6% of occurrences downgrade full→low-rank; the per-molecule downgrade fraction correlates with degradation at ρ = +0.025 and is collinear with frequency loss (ρ = 0.52). No independent signal.

**Q14. Which explanation does the data support?**
**D — mixed**, with the effect-size ordering (1) baseline-difficulty-dependent redistribution (ρ = −0.352), (2) weak non-specific frequency-loss/newly-rare signal (partial ρ ≈ 0.10), (3) newly-OOV absent, (4) tier reallocation non-explanatory. The pure "statistical sharing loss" (A) and "coarse structural inductive bias" (B) readings each capture at most a small fraction; the dominant pattern is not fragmentation-directed.

**Q15. Is there enough evidence to open "coarse shared + exact residual"?**
**No.** No fragmentation metric clears even the moderate gate; split-but-still-common does not degrade; newly-OOV molecules improve. Do not build it on this evidence.

**Q16. Single recommended next stage?**
**Diagnose the baseline-difficulty-dependent error redistribution** — why does semantic correctness trade easy-bulk accuracy for hard-tail accuracy? (A frozen, no-training analysis of *which* molecules/patch types move, and whether the coarse historical identity acts as a bulk-favouring regulariser.) Do **not** open a hierarchical embedding; close the simple-frequency-fragmentation route.
