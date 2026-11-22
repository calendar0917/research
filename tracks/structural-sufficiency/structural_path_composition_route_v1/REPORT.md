# Structural path composition route (PSC) — final report

*Round*: `structural_path_composition_route_v1` · *protocol*: `structural-path-composition-route-v1` (frozen before formal training) · *analysis-code commit*: `03dd1b6ef57a7ea23395117595956bbe36917634` · *generated*: 2026-11-22T20:33:51Z · *frozen artefacts*: `analysis/*.json`, `analysis/figures/`, `analysis/tables.md`, `REPORT.md` in the same commit series

Dataset: `ogbg-mollipo` (official split untouched; Test never read). All numbers below are whole-molecule RMSE / MAE in **original label units** (the training target is the standardised label, all reported metrics are inverted back). Higher-is-better quantities are marked.

## TL;DR

* The frozen champion of the eight PSC arms is **`PSC_POOL_BAG_PAIR`** (3-seed D0 RMSE 0.687 ± 0.014).
* It is **0.107 RMSE worse than `ACRA_REF`** (0.580) and 0.161 worse than the GINE reference. Paired-molecule bootstrap (champion − reference, positive = champion better): -0.107 [-0.151, -0.066] vs `ACRA_REF`, -0.161 [-0.203, -0.118] vs GINE.
* A closed-form **path count ridge** (`PATH_COUNT_RIDGE`, D0-selected α) reaches 0.617, i.e. **better than every PSC arm** (-0.070 vs the champion).
* The additive `SUM` readout — the property the route was designed around — is consistently beaten by the non-additive `POOL` readout (+0.058 [+0.024, +0.093] RMSE, positive = POOL better).
* `ORDER` (path identity) vs `BAG` (atom/bond multiset) shows no consistent benefit for ORDER: the four matched contrasts are +0.041, +0.019, +0.058, +0.030 (BAG − ORDER, positive = BAG better).
* PAIR coupling is a small, consistent-in-sign effect: 4/5 contrasts positive (0 of 5 with a CI excluding 0), maximum +0.021 RMSE — below the pre-registered +0.02 practical-gain threshold.
* **Official Valid** (frozen full-Train refit, single pass): champion 0.779 vs `ACRA_REF` 0.786 (Δ +0.006 [-0.091, +0.129]; a tie), with `GINE_REF` 0.622 and the path-count KRR 0.724 still ahead. The D0 deficit versus ACRA does not replicate at full-Train scale, but the branch still does not beat the baselines.
* **Verdict**: do not continue this branch as specified. The compositional machinery is implementable and exactly verifiable, but on Lipo the strict path-sum parameterisation is dominated on the selection split by a linear path-count ridge and by ACRA, ties ACRA at best on the official Valid refit while staying behind GINE and both kernel baselines, and the phenomenon it targets (unseen path words) carries only 4.1% of the held-out path mass.

## 0. Pre-registered route classification (§12 of the brief)

Δ is always *candidate − reference* in RMSE/MAE, **positive = the candidate is better**; the brief's directional labels use the same sign convention.

| field | value |
|---|---|
| **D0_PRACTICAL_GAIN** | **false** — ΔRMSE -0.107 (0/3 seeds in C*'s favour), ΔMAE -0.086 (rule: ≥ +0.02 RMSE, ≥2/3 seeds, MAE no more than 0.01 worse than the reference) |
| **D1_DIRECTION** | **REVERSED** — mean ΔRMSE -0.068, paired CI [-0.118, -0.019] (REVERSED = candidate significantly worse; the D0 deficit replicates) |
| **VALID_DIRECTION** | **CONSISTENT** (literal rule) — mean ΔRMSE +0.006, CI [-0.091, +0.129]. |Δ| < 0.02 and the CI straddles 0, so this is a tie in substance: the D0/D1 deficit does **not** replicate on Valid, and neither does it become a gain |
| **COMPETITIVE_GAP** (D0) | vs `GINE_REF` -0.161 [-0.203, -0.118]; vs `MORGAN_COUNT_KRR` -0.028 [-0.072, +0.015]; vs `PATH_COUNT_KRR` -0.032 [-0.078, +0.010] — behind all three in point estimate; the two kernel intervals straddle 0 |
| **COMPETITIVE_GAP** (Valid) | vs `GINE_REF` -0.158; vs `PATH_COUNT_RIDGE` +0.004; vs `MORGAN_COUNT_KRR` -0.035; vs `PATH_COUNT_KRR` -0.055 (point estimates only; the frozen scoring pass computed an interval for the ACRA contrast alone) |
| **M_order** | no gain: every BAG−ORDER contrast is ≥ 0 (+0.041, +0.019, +0.058, +0.030), and 2/4 CIs exclude 0 — connection order is not a source of benefit |
| **M_readout** | POOL−SUM = +0.058, +0.037, +0.064, +0.036 (3/4 CIs exclude 0), replicated on Valid (+0.065): strict additivity is the current limit |
| **M_pair** | PAIR−DIAG +0.035 [+0.008, +0.061] (the signed difference carries real signal); PAIR−MOLECULE +0.020 [-0.006, +0.044] (absolute-prediction gain below the +0.02 threshold) — keep the mechanism claim small |
| **M_share** | in the T-unseen / fragments-seen cell (156 pairs) the shared model is 0.468 vs 0.459 (path-count ridge), 0.467 (path-count KRR) and 0.429 (ACRA) — no extrapolation advantage over exact-ID baselines; in the one-sided cell (100 pairs) it is 0.480 vs 0.535/0.527 (better) but 0.463 (ACRA, worse) |

Classification summary: the route does not reach the pre-registered practical gain on D0, is directionally reversed on the independent D1 split, and is a tie at best on Valid; within the mechanism fields, only `M_readout` (POOL ≫ SUM) is a clean, replicating effect. Per the brief's rules this is a **stop** for the strict path-sum branch, not an invitation to re-tune it.

## 1. Question and pre-registered decision rule

The round asks whether sharing one structural score along **concrete connection paths** can carry seen atom/bond regularities to unseen path words, unseen exact substitutions and unseen molecules, *while also* improving whole-molecule prediction. The frozen protocol (`PROTOCOL.md`) fixed in advance: the 15 arms, the 160-epoch recipe, the 7.5 h/16 GPU-hour budget, the champion rule (min 3-seed mean D0 RMSE; within 0.005 s.t. priority SUM>POOL, MOLECULE>PAIR, ORDER>BAG), the `E_m`-epoch D1 replication, the full-Train refit with a single frozen Valid pass, and the degradation order. No Test label was ever read, and official Valid labels were opened exactly once, after the final checkpoint list was frozen (`analysis/final_eval_lock.json`).

## 2. What was built

* **Path objects.** Simple undirected paths of length `l = 0..5` (L = 5) over the raw OGB graph, canonicalised by their raw atom/bond token sequence read from both ends (order-preserving canonical orientation; `BAG` variants collapse the orientation and keep the multiset). Multiplicity per molecule is counted; per-length scale `c_l = 1/max(1, mean |P_l|)` keeps molecule totals comparable.
* **Cache**: 65159 distinct path keys, 739417 (molecule, path) entries, 15.1 MB, sha `436156363fcac411…`.
* **Model** (`psc_model.py`). Shared bias-free encoder `E(token_0..token_l)` (embeddings 32/16 → 256 → 128 → 64), scalar head `h`, readout `F = β + Σ_l c_l Σ_{p ∈ P_l} h(E(p))` (**SUM**: head before the sum, i.e. *one* score per path shape shared by every molecule) or `F = β + h(Σ_l c_l Σ_p E(p))` (**POOL**: head after the sum). `BAG` means the encoder sees the orientation-invariant multiset; `ORDER` sees the sequence and averages the two orientations.
* **Supervision**: `MOLECULE` (whole-molecule label), `PAIR` (`L_mol + 0.25·Σ_e w_e (f_b − f_a)²` over matched pairs, component weights `w = 1/(K·m_c)`), `DIAG` (the same formula against a fixed zero pair target). λ_aux = 0.25, 32 auxiliary pairs per batch.
* **Baselines in the same harness**: ACRA-v2 (frozen parent modules and cache; `ACRA_REF` = parent objective, `ACRA_PAIR` = + pair regulariser), GINE reference recipe, and four closed-form CPU baselines (path-count ridge/KRR, Morgan-count KRR r = 2/3, count-only ridge) with α locked on D0.

## 3. Data discipline

* Labels are loaded only on official Train rows; the training object's label vector is NaN outside Train. Train-only label file sha `a44d29deba1fd36e…`.
* `D0` = the frozen parent split (Train-sub 2856 / Dev 504, **scaffold-disjoint**, 0 shared Bemis–Murcko scaffolds). `D1` = an independent scaffold-disjoint split of the same 3360 Train rows (seed 20261122, 504 held out). `FULL` = all 3360 Train rows, evaluated once on the official Valid 420 rows.
* Selection used D0 only. D1 (fixed `E_m` epochs, final checkpoint) and Valid cannot change the champion.

## 4. Engineering validation (before any formal number)

`code/psc_checks.py` produced `analysis/checks.json`; all checks must pass and do:

| check | evidence |
|---|---|
| `1_enumeration` (PASS) | identical_atom_pair_same_key=True; row0=24, 27, 38, 51; row1=33, 35, 50, 66; row11=26, 28, 39, 49; row2=21, 23, 32, 44; row7=34, 36, 49, 61 |
| `2_renumbering` (PASS) | keys=178; rows=row2 + 2 random rows |
| `3_padding_reversal` (PASS) | input_dim=289; plen=4, 4, 5, 4 |
| `4_order_bag` (PASS) | bag_perm_max_abs=0.0; bag_single_max_abs=1.4901161193847656e-08; order_rev_distinguishable=True |
| `3b_batch_vs_single` (PASS) | max_abs_batch_single=0.0; max_len=5 |
| `6_pair_algebra` (PASS) | exact_pair=1.845284651962062; masked_acra_equals_parent=True; n_components=398; n_unique_pairs=2456; sampler_mean=1.839020657741019; sum_w=1.0 |
| `7_fit_only` (PASS) | heldout_only_atom_categories=0; masked=0; n_dev=504; n_fit=2856; scaler=mean=2.185815826311436, n=2856, std=1.2139664740467941 |
| `8_init_grad` (PASS) | n_grad_tensors=21; params=beta=1, encoder=95056, head=4160, total=99217; sum_pool_init_equal=True |
| `8b_overfit32` (PASS) | mse_end=0.11202660948038101; mse_start=1.7159264087677002; n_molecules=32; steps=300 |

All checks pass; the raw output is `analysis/checks.json`. The 32-molecule overfit matters for the interpretation below: the optimiser, the path cache and the readout are not broken, so a plateau on D0 is a statement about capacity and data, not about a bug.

## 5. Q1 — does the route close the gap to ACRA / fingerprints / GINE?

**No.** On D0 the ordering is stable and monotone (3 seeds, mean ± sd):

| model | D0 RMSE | D0 MAE | Δ vs ACRA_REF [95% CI] |
|---|---|---|---|
| `PSC_POOL_BAG_PAIR` | 0.687 ± 0.014 | 0.530 | +0.107 |
| `PSC_POOL_PAIR` | 0.717 ± 0.057 | 0.564 | +0.137 |
| `PSC_SUM_BAG_PAIR` | 0.723 ± 0.027 | 0.556 | +0.143 |
| `ACRA_REF` | 0.580 ± 0.002 | 0.445 | — |
| `ACRA_PAIR` | 0.585 ± 0.022 | 0.450 | -0.102 [-0.141, -0.065] |
| `GINE_REF` | 0.527 ± 0.006 | 0.398 | -0.161 [-0.203, -0.118] |
| `PATH_COUNT_RIDGE` | 0.617 ± 0.000 | 0.465 | -0.070 [-0.115, -0.028] |
| `PATH_COUNT_KRR` | 0.655 ± 0.000 | 0.495 | -0.032 [-0.078, +0.010] |
| `MORGAN_COUNT_KRR` | 0.659 ± 0.000 | 0.520 | -0.028 [-0.072, +0.015] |
| `COUNT_ONLY_RIDGE` | 1.161 ± 0.000 | 0.902 | +0.474 [+0.405, +0.544] |

Paired-molecule bootstrap (2000 resamples, shared index across models and seeds): the champion is significantly worse than `ACRA_REF`, `ACRA_PAIR`, `GINE_REF`, `PATH_COUNT_RIDGE` and statistically indistinguishable (CI straddles 0) from `PATH_COUNT_KRR`, `MORGAN_COUNT_KRR`. The per-molecule view agrees: mean per-molecule |error| change vs `ACRA_REF` is -0.085 [-0.114, -0.056], and the champion is closer to truth on 40.9% of molecules.

The gap is **not** an optimisation artefact: at the same 160-epoch horizon the ACRA and GINE reference runs converge to lower held-out error while PSC training curves plateau (Figure 4), and the 32-molecule overfit check shows the same code drives the loss down when the task is small.

## 6. Q2 — ORDER vs BAG: is path identity worth anything?

Matched one-factor contrasts (same readout, same target), D0:

| contrast | Δ RMSE [95% CI] | p |
|---|---|---|
| PSC_SUM_BAG vs PSC_SUM | +0.041 [+0.012, +0.072] | 0.006 |
| PSC_POOL_BAG vs PSC_POOL | +0.019 [-0.026, +0.062] | 0.379 |
| PSC_SUM_BAG_PAIR vs PSC_SUM_PAIR | +0.058 [+0.029, +0.086] | 0.000 |
| PSC_POOL_BAG_PAIR vs PSC_POOL_PAIR | +0.030 [-0.011, +0.068] | 0.147 |

The four contrasts are all ≥ 0 for BAG, i.e. **ORDER never wins**: in the SUM family the orientation-invariant multiset is significantly better (+0.041 and +0.058), and in the POOL family the two are indistinguishable (CIs straddle 0; scaffold-cluster bootstrap: `PSC_POOL_BAG_PAIR_vs_PSC_POOL_PAIR` +0.030 [-0.019, +0.078], `PSC_POOL_BAG_vs_PSC_POOL` +0.019 [-0.035, +0.072], `PSC_SUM_BAG_PAIR_vs_PSC_SUM_PAIR` +0.058 [+0.019, +0.095], `PSC_SUM_BAG_vs_PSC_SUM` +0.041 [+0.002, +0.076]). **Conclusion**: for Lipo, the *order* in which the atom/bond word is read along a path adds nothing — path identity beyond the multiset is not the useful signal, and the ORDER-specific machinery (sequence encoding, reversal averaging) is not earning its keep.

## 7. Q3 — SUM vs POOL: is strict local additivity the limiting factor?

**Yes, and it is the largest single effect in the whole design.**

| contrast | Δ RMSE [95% CI] | p |
|---|---|---|
| PSC_POOL vs PSC_SUM | +0.058 [+0.024, +0.093] | 0.001 |
| PSC_POOL_BAG vs PSC_SUM_BAG | +0.037 [-0.005, +0.077] | 0.080 |
| PSC_POOL_PAIR vs PSC_SUM_PAIR | +0.064 [+0.030, +0.097] | 0.000 |
| PSC_POOL_BAG_PAIR vs PSC_SUM_BAG_PAIR | +0.036 [+0.008, +0.065] | 0.012 |

All four POOL contrasts are positive (POOL better) and three of four CIs exclude 0. Under SUM the only way two molecules can differ is through the multiset of per-path scores, and the scorer must express each path's contribution as a smooth function of a 64-d embedding of its token sequence. Under POOL the summed representation is read by a 2-layer head, which restores *cross-path* interactions at the cost of the ledger property. The accuracy ordering (POOL ≫ SUM) is therefore direct evidence that **the strict additive constraint, not the path vocabulary, is what caps the route**.

Consistent with this, the pure count-only ridge (6 features) is far worse (1.161) than the 65k-feature path-count ridge (0.617), so the *path* vocabulary does carry signal; the neural shared scorer simply cannot extract it as efficiently as a closed-form linear map on the same counts.

## 8. Q4 — does PAIR supervision help, and does it depend on the readout?

| contrast | Δ RMSE [95% CI] | p |
|---|---|---|
| PSC_SUM_PAIR vs PSC_SUM | +0.003 [-0.015, +0.021] | 0.717 |
| PSC_POOL_PAIR vs PSC_POOL | +0.009 [-0.014, +0.033] | 0.445 |
| PSC_SUM_BAG_PAIR vs PSC_SUM_BAG | +0.021 [-0.002, +0.042] | 0.070 |
| PSC_POOL_BAG_PAIR vs PSC_POOL_BAG | +0.020 [-0.006, +0.044] | 0.113 |
| ACRA_PAIR vs ACRA_REF | -0.004 [-0.026, +0.016] | 0.636 |

The coupling is **real but small**: every PSC PAIR contrast is ≥ 0 (SUM +0.003, POOL +0.009, SUM+BAG +0.021, POOL+BAG +0.020), but 0 of 5 CIs exclude 0 and **no contrast reaches the pre-registered +0.02 practical gain threshold**. The ACRA regulariser is a wash (-0.004). The dependence on readout is weak: the pair term gives roughly +0.01 to +0.02 in both families, consistent with a data-efficiency device rather than a substitute for capacity.

The pair algebra is exact and verified: `PAIR = DIAG − 2·cross` (deviation < 1e-12), `Σ w_e = 1` per component, and the sampler matches the component-weighted distribution.

Because the champion carries the PAIR objective, the protocol required a **DIAG twin** (`PSC_POOL_BAG_DIAG`, 3 seeds): same input/readout, same pair sampling, but the pair target is a fixed zero vector, so it can only anchor levels. The champion beats it by +0.035 [+0.008, +0.061] RMSE (positive = PAIR better), i.e. the *difference* signal — not merely spending compute on pairs — is what the PAIR objective contributes.

Note the two controls answer different questions. Against the molecule-only twin `PSC_POOL_BAG` the PAIR term buys little (+0.020 RMSE, ns); against the DIAG twin, which samples the same pairs but never sees the signed difference, it is clearly better. The DIAG twin is in turn +0.015 worse than the molecule-only twin, so pair *sampling* per se is a small cost — the benefit comes from supervising the difference.

That contrast is the cleanest available test of the brief's "pair coupling vs pair weighting" question: sharing weights across the pair term alone does not help; supervising the signed difference does.

## 9. Q5 — where would unseen-substitution gains have to come from?

The novelty accounting first, because it bounds what any transfer mechanism can deliver:

* 65159 path keys; 53763 appear in the D0 fit set, 11396 never do.
* The median fit-set key appears in **1.0** molecule(s) (mean 10.31).
* Only **4.1%** of held-out path mass falls on keys unseen in fit; 8.2% falls on keys seen ≤5 times.
* Raw categorical UNK on held-out molecules is 0.0% (atoms and bonds), so the `zero_unseen` embedding masking is inert on this dataset — the novelty that matters is at the path-word level, not the raw feature level.

Matched-pair support classes (held-out pairs, cut-row weighted, pair Δ):

| support class | pairs | weight | mean \|Δtrue\| | PSC_POOL_BAG_PAIR | ACRA_REF | PATH_COUNT_RIDGE | PATH_COUNT_KRR | MORGAN_COUNT_KRR |
|---|---|---|---|---|---|---|---|---|
| T_seen | 195 | 173 | 0.693 | 0.443 (rel 0.64) | 0.392 (rel 0.57) | 0.412 (rel 0.60) | 0.431 (rel 0.62) | 0.465 (rel 0.67) |
| T_unseen_both_frags_unseen | 24 | 19 | 0.417 | 0.316 (rel 0.76) | 0.478 (rel 1.14) | 0.438 (rel 1.05) | 0.426 (rel 1.02) | 0.407 (rel 0.97) |
| T_unseen_frag_both_seen | 156 | 136 | 0.839 | 0.468 (rel 0.56) | 0.429 (rel 0.51) | 0.459 (rel 0.55) | 0.467 (rel 0.56) | 0.492 (rel 0.59) |
| T_unseen_one_frag_unseen | 100 | 85 | 0.673 | 0.480 (rel 0.71) | 0.463 (rel 0.69) | 0.535 (rel 0.79) | 0.527 (rel 0.78) | 0.489 (rel 0.73) |

Read this table with the `mean |Δtrue|` column: the *rel* column is MAE divided by that scale, so it is comparable across classes.

Molecule-level unseen-path-mass strata (D0, seed 0):

| bucket | n | PSC_POOL_BAG_PAIR | ACRA_REF | PATH_COUNT_RIDGE | MORGAN_COUNT_KRR |
|---|---|---|---|---|---|
| (0.0,0.01] | 80 | 0.604 | 0.599 | 0.578 | 0.590 |
| (0.01,0.05] | 159 | 0.698 | 0.583 | 0.620 | 0.659 |
| >0.05 | 120 | 0.825 | 0.672 | 0.725 | 0.750 |
| eta_mol=0 | 145 | 0.610 | 0.478 | 0.531 | 0.613 |

Mean held-out unseen-path mass is 0.0434 (median 0.0140, max 0.503); 23.8% of molecules exceed 5%.

From the lowest to the highest unseen-mass bucket the error grows by +0.214 (champion), +0.194 (`ACRA_REF`), +0.194 (path-count ridge) and +0.133 (GINE). The champion degrades at least as fast as the linear ridge, so the novelty strata give no hint of a transfer advantage.

**Pair-level vs molecule-level.** On held-out × held-out pairs the champion's Δ MAE is 0.453, versus 0.422 for `ACRA_REF` and 0.462 for the path-count KRR — i.e. at the level of *differences* the champion is close to (and slightly better than) the kernel baselines, while at the level of whole molecules it is 0.687 vs 0.655. The pair Δ regression slope is 0.514 for the champion vs 0.589 for `ACRA_REF`: the additive model attenuates contrasts more, even though their ordering is mostly right. That is the signature of a model whose *levels* are biased (heavy shrinkage) rather than one that cannot see the structure.

**Answer to Q5.** Transfer does not show up where it would have to. The only cell where PSC is best is the most extreme one (T unseen, *both* fragment words unseen), and it is 24 pairs — too few to build on, and exactly the cell where a heavily shrunk predictor wins by predicting small |Δ|. In the cell where a real transfer mechanism should shine (T unseen, both fragments **seen**, 156 pairs) PSC trails `ACRA_REF` (0.468 vs 0.429 MAE), is level with the exact-ID path-count KRR (0.467) and behind the path-count ridge (0.459), and is only marginally ahead of the raw Morgan-count kernel (0.492); the same ordering holds in the scale-free *rel* column (0.56 vs 0.51 vs 0.59). In other words, the route shows no advantage in the cell where unseen *path words* meet seen *fragments* — the very transfer the design targets. Combined with the 4.1% path-mass share of unseen keys, the honest answer is that **unseen path words are too rare on Lipo for a transfer mechanism to move the molecule-level metric, and the model does not show a transfer advantage even in the pair-level cells where it could be seen.**

## 10. Mechanistic checks

### 10.1 Pair-residual decomposition (§9.5)

For D0 held-out × held-out pairs with component weights, `E[(e_b−e_a)²] = E[e_a²] + E[e_b²] − 2E[e_a e_b]`:

| model | E[(e_b−e_a)²] | E[e_a²] | E[e_b²] | 2·cross | corr(e_a,e_b) |
|---|---|---|---|---|---|
| ACRA_REF | 0.268 | 0.375 | 0.272 | 0.379 | 0.440 |
| GINE_REF | 0.283 | 0.362 | 0.222 | 0.301 | 0.495 |
| PSC_POOL_BAG_DIAG | 0.400 | 0.583 | 0.456 | 0.639 | 0.589 |
| PSC_POOL_BAG_PAIR | 0.328 | 0.580 | 0.432 | 0.684 | 0.467 |
| PSC_SUM_BAG_PAIR | 0.308 | 0.622 | 0.504 | 0.819 | 0.553 |

The identity closes to machine precision in every run (max |error| 1.7e-16). A large `corr(e_a,e_b)` means the pair Δ is dominated by endpoint-level errors that could in principle cancel; the table shows how much of the pair error is *shared* between the two endpoints versus pair-specific.

### 10.2 SUM edit ledger (§9.6)

Ledger model: `PSC_SUM_BAG_PAIR` (champion `PSC_POOL_BAG_PAIR`). The champion uses POOL, so this ledger is reported for its matched SUM twin and does not explain the champion's own predictions. Identity `F = β + Σ c_l h(E)` holds to 1.7e-06 (standardised units) over 12 held-out molecules; 204 of 413 held-out pairs admit an unambiguous S/C split, of which 54 have both sides ≥3 atoms.

| rule | a | b | true Δ | ACRA_B* Δ | eta | ΔF_S | ΔF_C | ΔF_X | ΔF pred |
|---|---|---|---|---|---|---|---|---|---|
| T unseen / fragments seen · largest ACRA error | 114 | 3789 | +0.200 | -0.921 | 0.00 | -0.084 | -0.506 | +0.612 | +0.023 |
| T unseen / fragments seen · largest ACRA error | 570 | 3184 | +1.300 | +0.727 | 0.01 | -0.057 | +0.000 | +0.000 | -0.057 |
| T unseen / fragments seen · smallest ACRA error | 1594 | 2299 | -0.630 | -0.620 | 0.20 | +0.530 | +0.321 | -0.496 | +0.354 |
| T unseen / fragments seen · smallest ACRA error | 2501 | 3184 | -0.400 | -0.380 | 0.00 | +0.065 | +0.000 | +0.000 | +0.065 |
| high eta · largest ACRA error | 181 | 1944 | -0.760 | +0.805 | 0.40 | +0.493 | +0.000 | +0.000 | +0.493 |
| high eta · smallest ACRA error | 1560 | 4135 | +0.470 | +0.531 | 0.28 | +0.588 | +0.000 | +0.000 | +0.588 |
| seen T · largest positive Δ | 53 | 3051 | +3.610 | +2.756 | 0.02 | +0.789 | +0.000 | +0.000 | +0.789 |
| seen T · most negative Δ | 815 | 3159 | -1.020 | -0.479 | 0.00 | -0.034 | +0.000 | +0.000 | -0.034 |

In 6 of 8 selected examples the C-side and crossing ledger terms are **exactly 0**: those pairs share an identical context subgraph and their C/X path-token multisets are identical, so 100% of the predicted Δ comes from paths inside the edited fragment side (checked directly on the token multisets). In the remaining examples the two sides differ beyond the cut bond and the C/X terms contribute. That exact cancellation is the property the route wanted — and it also shows why the property alone is not enough: being exactly additive does not make the *magnitude* right, and the predicted Δ is systematically smaller than the true one in the high-|Δ| cases.

### 10.3 Size strata and worst-error sensitivity (§9.7)

| model | overall | Q1 | Q2 | Q3 | Q4 | top-5 removed |
|---|---|---|---|---|---|---|
| ACRA_REF | 0.581 | 0.668 | 0.593 | 0.557 | 0.536 | 0.546 |
| COUNT_ONLY_RIDGE | 1.161 | 1.185 | 1.198 | 0.966 | 1.287 | 1.117 |
| GINE_REF | 0.522 | 0.559 | 0.535 | 0.509 | 0.501 | 0.490 |
| PSC_POOL_BAG_PAIR | 0.693 | 0.753 | 0.750 | 0.635 | 0.667 | 0.644 |

Quartile edges on fit-set atom counts: [21.0, 27.0, 32.0]. Removing the union of each model's 5 worst molecules moves RMSE by a few percent at most, so no conclusion rests on a handful of outliers. Selection never used this analysis.

## 11. D1 replication (independent scaffold-disjoint split of the same Train rows)

| model | D1 RMSE (2 seeds) | D0 RMSE | Δ(D1−D0) | D1 Δ vs ACRA_REF (CI where bootstrapped) |
|---|---|---|---|---|
| `PSC_POOL_BAG_PAIR` | 0.706 ± 0.004 | 0.687 | +0.018 | -0.068 [-0.118, -0.019] |
| `PSC_POOL_PAIR` | 0.755 ± 0.010 | 0.717 | +0.037 | +0.117 |
| `PSC_SUM_BAG_PAIR` | 0.808 ± 0.052 | 0.723 | +0.085 | +0.171 |
| `ACRA_REF` | 0.638 ± 0.008 | 0.580 | +0.057 | — |
| `GINE_REF` | 0.553 ± 0.003 | 0.527 | +0.027 | -0.152 [-0.199, -0.104] |

D1 runs used the frozen `E_m` epochs (median D0 best epoch per arm) and the final checkpoint, with no selection.

The champion's deficit versus `ACRA_REF` is +0.068 RMSE on D1 versus +0.107 on D0: the sign and the order of magnitude replicate, and no arm moves across the ACRA boundary.

## 12. Official Valid (full-Train refit, single frozen pass)

Valid labels were opened once, after `analysis/final_eval_lock.json` fixed the checkpoint list (420 rows; members read: row_id, label; label sha `dae6a22f70a9e533…`). No Valid number influenced training or selection.

| model | seeds | Valid RMSE | Valid MAE | ACRA_B* − model [95% CI] |
|---|---|---|---|---|
| `PSC_POOL_BAG_PAIR` | 3 | 0.779 ± 0.013 | 0.604 | +0.006 [-0.091, +0.129] |
| `ACRA_REF` | 3 | 0.786 ± 0.018 | 0.555 | — |
| `PSC_SUM_BAG_PAIR` | 3 | 0.845 ± 0.014 | 0.639 | -0.059 [-0.123, +0.011] |
| `GINE_REF` | 3 | 0.622 ± 0.012 | 0.454 | +0.164 [+0.094, +0.246] |
| `PATH_COUNT_RIDGE` | 1 | 0.784 ± 0.000 | 0.560 | +0.002 [-0.053, +0.051] |
| `PATH_COUNT_KRR` | 1 | 0.724 ± 0.000 | 0.541 | +0.062 [-0.028, +0.169] |
| `MORGAN_COUNT_KRR` | 1 | 0.745 ± 0.000 | 0.563 | +0.041 [-0.056, +0.153] |
| `COUNT_ONLY_RIDGE` | 1 | 1.169 ± 0.000 | 0.944 | -0.383 [-0.505, -0.244] |

**What Valid says.** The champion's frozen Valid RMSE is 0.779 (MAE 0.604), versus 0.786 (MAE 0.555) for `ACRA_REF` and 0.622 for `GINE_REF`.

On Valid the champion and `ACRA_REF` are statistically indistinguishable: Δ = +0.006 [-0.091, +0.129] RMSE (positive = champion better). The MAE ordering is the other way round (0.604 vs 0.555), so the RMSE tie reflects compensating errors (the champion has a heavier tail and a larger positive bias on average), not equal accuracy.

This does **not** overturn the D0 result, and the D0-versus-Valid gap is not a uniform shift: every model is worse on Valid than on D0, but by very different amounts — `PSC_POOL_BAG_PAIR` +0.092, `ACRA_REF` +0.206, `GINE_REF` +0.095, `PATH_COUNT_RIDGE` +0.167. D0 dev is a scaffold-disjoint subset of Train-sub; official Valid is a different 420-row set, and the models that fit D0 dev best lose the most. The pre-registered rule selects on D0 and the champion stands.

Caveat in the other direction: each full-Train refit ran exactly its `E_m` epochs with its own cosine schedule, so `ACRA_REF` received 87 epochs against 160 in its D0 run (on 18% more rows), while the champion received 127. The frozen rule therefore handicaps the early-stopping arms on Valid; the near-tie should be read with that in mind.

The POOL ≫ SUM ordering does replicate on Valid: champion 0.779 versus its SUM twin 0.845 (+0.065 in favour of POOL), the same sign as the D0 contrast and a similar magnitude. That is the one design conclusion of the round that survives the distribution shift.

Valid was scored once, from the frozen lock, after every full-Train prediction existed; no Valid number changed any model, epoch, seed or arm.


## 13. Budget and artefacts

* GPU-hours consumed (every launched job, 1 GPU each, including smoke runs and cancelled duplicates): **3.48** of the 16 h ceiling (within budget, 21.7% of it).
* Train-loop hours: 3.18 (of the 3.48 GPU-hours: 91.5% in the training loop, the rest in evaluation/checkpoint I/O).
* Closed-form CPU hours: 0.14.
* Completed GPU runs: 58; jobs launched and recorded by the budget ledger: 61, of which 0 did not complete. The 44 pre-flight rejections in §14 never started a job and are not in this ledger.

Artefacts (all reproducible from the committed tree):

```
PROTOCOL.md                      frozen pre-training protocol
code/psc_paths.py                path enumeration / canonical keys / cache / gather
code/psc_model.py                path encoder + SUM/POOL readouts
code/psc_train.py                all three families (PSC / ACRA / GINE)
code/psc_analysis.py             tables, selection, bootstrap, pairs, strata
code/psc_ledger.py               exact SUM edit ledger (geometric S/C split)
code/psc_cpu.py                  closed-form path-count / Morgan baselines
code/psc_final_eval.py           lock + single frozen Valid pass
code/psc_budget.py               GPU/CPU ledger
analysis/checks.json             engineering checks (all pass)
analysis/prepare.json            data prep + cache hashes
analysis/selection_lock.json     champion + twins + E_m, frozen on D0
analysis/bootstrap{,_D1}.json    paired molecule / scaffold-cluster bootstrap
analysis/unique_pair_metrics.json, molecule_novelty.json, size_strata.json
analysis/pair_residual_decomposition.json, sum_edit_ledger.json
analysis/final_eval_lock.json, final_valid.json
analysis/compute_budget.json, path_sparsity.json, tables.md, README.md, figures/
results/main_table.csv, results/summary.json
```

## 14. Deviations, incidents and honest limits

* **Pre-flight launch failures (reruns only).** The queue driver launches through `rr`, which refuses to start a formal run from a dirty tracked tree or a stale remote checkout. Analysis edits made while a queue was running produced 44 rejected launch attempts across the round (`LAUNCH_FAILED`, all visible in `analysis/logs/`); each affected run was relaunched from a clean commit and **no job that actually started ever failed** (0 non-completed runs in `analysis/compute_budget.json`). Rejections consumed no GPU time. No run was selected or excluded on the basis of its result.
* **Horizon.** 160 epochs is the frozen budget. The PSC arms' median best epoch is late (151, 132, 127, 132, 124, 143, 122, 127 of 160), i.e. they are still improving when the schedule ends. The comparison to a closed-form ridge on the same features is therefore a *lower bound* on the additive restriction's cost; but the 32-molecule overfit shows the gap is not an implementation failure, and the POOL>SUM contrast is measured at the same horizon for both.
* **Baseline reproduction.** In this harness `ACRA_REF` reaches D0 dev RMSE 0.580 ± 0.002 (3 seeds), whereas the parent round's own reference run reported 0.555 for the same nominal recipe on the same split. The ~0.025 difference is consistent with a different initialisation/data-order path in the re-hosted trainer; it makes this round's ACRA baseline slightly *weaker* than the parent's, which if anything understates the PSC deficit. The strongest non-PSC baseline here is GINE (0.527), so no conclusion rests on the ACRA reproduction alone.
* **Recipe fairness.** PSC/ACRA/GINE share the frozen optimiser recipe (AdamW, peak lr 1e-3 / 1e-5, cosine, clip 5, batch 64 molecules). The ACRA and GINE baselines are the parent rounds' own recipes, not tuned for this round; the CPU baselines lock α on D0. No Valid-based tuning anywhere.
* **Path-length ceiling.** L = 5 (the pre-registered fallback L = 4 was not needed). Longer paths would raise per-molecule coverage but also the key count; the sparsity table shows the key space is already the binding constraint.
* **Ledger scope.** The ledger covers the SUM readout only, and its S/C split is geometric (component sizes) rather than the TRC word identity, because the parent cache's per-parse atom sets are stored in fragment-local numbering. The exact-cancellation result above holds under either convention for same-context pairs.
* **DIAG twin** was run because the champion carries the PAIR objective: three D0 seeds at the same `E_m`, exact contrast in §8. The *optional* extensions (ESOL transfer, size curve, M* full-train beyond the reference arms) were not started: the required matrix, the D1 replication and the full-Train refits consumed the budget, and the protocol's extension gate requires the required queue to be complete first.

## 15. What this means for the branch

1. **Do not continue the strict path-sum branch as specified.** A shared scalar score per path shape is dominated by a linear model on the same path counts (-0.070 RMSE means the ridge is *better*) and by ACRA on the D0 selection split; on the official Valid refit it ties ACRA but stays behind GINE and the kernel baselines. More capacity in the encoder is not the fix — the readout is the prime suspect.
2. **If the branch is revived, replace the shared scorer with a hybrid** `F = β + Σ_p [w_key(p) + h(E(p))]`, i.e. a free coefficient for keys with sufficient support and a shared, token-conditioned residual for the tail. That keeps a ledger (per-key plus per-shape terms) while removing the capacity cliff; the ridge/path-sum gap measures the headroom.
3. **Keep the pair algebra and the diagnostic harness.** The PAIR objective is exactly verifiable, gives a small positive effect in every PSC family and a clear gain over its DIAG twin, while the ACRA variant is a wash (-0.004); the matched-pair ledger (exact C/X cancellation) is a reusable instrument even if the model class changes.
4. **Do not expect unseen-path-word transfer to move Lipo numbers.** It is 4.1% of held-out path mass. A route that targets it should be evaluated on a dataset or split where that slice is large, otherwise the metric cannot see it.

## 16. Non-claims

* No Test label was read; no Test metric is reported.
* No Valid label was read before `final_eval_lock.json`; no Valid number was used for training, early stopping, arm selection or hyper-parameter choice.
* The negative result is about *this* dataset, vocabulary (raw atom/bond tokens, L=5), recipe (160 epochs, shared MLP scorer) and budget. It is not a claim that path-based composition cannot work with a different parameterisation; §15.2 states the specific change the evidence supports.

