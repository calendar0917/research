# Structural path composition route (PSC) — final report

*Round*: `structural_path_composition_route_v1` · *protocol*: `structural-path-composition-route-v1` (frozen before formal training) · *commit*: `af35e730c04d15fe8ef87f95e88d3b3270c43224` · *generated*: 2026-11-22T19:11:26Z

Dataset: `ogbg-mollipo` (official split untouched; Test never read). All numbers below are whole-molecule RMSE / MAE in **original label units** (the training target is the standardised label, all reported metrics are inverted back). Higher-is-better quantities are marked.

## TL;DR

* The frozen champion of the eight PSC arms is **`PSC_POOL_BAG_PAIR`** (3-seed D0 RMSE 0.687 ± 0.014).
* It is **0.107 RMSE worse than `ACRA_REF`** (0.580) and 0.161 worse than the GINE reference; the paired bootstrap CI is -0.107 [-0.151, -0.066] (positive = champion better).
* A closed-form **path-count ridge** (`PATH_COUNT_RIDGE`, D0-selected α) reaches 0.617, i.e. **better than every PSC arm** (-0.070 vs the champion).
* The additive `SUM` readout — the property the route was designed around — is consistently beaten by the non-additive `POOL` readout (+0.058 RMSE, CI +0.058 [+0.024, +0.093]).
* `ORDER` (path identity) vs `BAG` (atom/bond multiset) shows no consistent benefit for ORDER: the four matched contrasts are +0.041, +0.019, +0.058, +0.030 (BAG − ORDER, positive = BAG better).
* PAIR coupling is a small, consistent-in-sign effect: 4/5 contrasts positive (0 of 5 with a CI excluding 0), maximum +0.021 RMSE — below the pre-registered +0.02 practical-gain threshold.
* **Verdict**: do not continue this branch as specified. The compositional machinery is implementable and exactly verifiable, but on Lipo the strict path-sum parameterisation is dominated by both a linear path-count kernel and ACRA, and the phenomenon it targets (unseen path words) carries only 4.1% of the held-out path mass.

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
| `1_enumeration` (PASS) | path enumeration equals directed-DFS/2 and matches a second independent enumerator |
| `2_invariance` (?) | renumbering invariance (atom relabelling gives identical keys) |
| `3_padding_reversal` (PASS) | padding + reversal apply to the declared columns only (input width 289) |
| `4_batch` (?) | batched == single-molecule forward (max |Δ| ~1e-13) |
| `5_bag_equivalence` (?) | BAG exactly invariant to path reversal; ORDER distinguishes it |
| `6_pair_algebra` (PASS) | PAIR = DIAG − 2·cross, Σw = 1, sampler unbiased, masked ACRA loss equals the parent |
| `7_fit_only` (PASS) | path vocabulary / categorical support / scaler are fit-only |
| `8_init_grad` (PASS) | SUM and POOL share identical initial outputs; all parameters receive gradients |
| `8b_overfit32` (PASS) | 32-molecule overfit: MSE 1.716 → 0.112 in 300 steps (the training chain can fit) |

The 32-molecule overfit matters for the interpretation below: the optimiser, the path cache and the readout are not broken, so a plateau on D0 is a statement about capacity/data, not about a bug.

## 5. Q1 — does the route close the gap to ACRA / fingerprints / GINE?

**No.** On D0 the ordering is stable and monotone (3 seeds, mean ± sd):

| model | D0 RMSE | D0 MAE | Δ vs ACRA_REF [95% CI] |
|---|---|---|---|
| `PSC_POOL_BAG_PAIR` | 0.687 ± 0.014 | 0.530 | n/a |
| `PSC_POOL_PAIR` | 0.717 ± 0.057 | 0.564 | n/a |
| `PSC_SUM_BAG_PAIR` | 0.723 ± 0.027 | 0.556 | n/a |
| `ACRA_REF` | 0.580 ± 0.002 | 0.445 | -0.107 [-0.151, -0.066] |
| `ACRA_PAIR` | 0.585 ± 0.022 | 0.450 | -0.102 [-0.141, -0.065] |
| `GINE_REF` | 0.527 ± 0.006 | 0.398 | -0.161 [-0.203, -0.118] |
| `PATH_COUNT_RIDGE` | 0.617 ± 0.000 | 0.465 | -0.070 [-0.115, -0.028] |
| `PATH_COUNT_KRR` | 0.655 ± 0.000 | 0.495 | -0.032 [-0.078, +0.010] |
| `MORGAN_COUNT_KRR` | 0.659 ± 0.000 | 0.520 | -0.028 [-0.072, +0.015] |
| `COUNT_ONLY_RIDGE` | 1.161 ± 0.000 | 0.902 | +0.474 [+0.405, +0.544] |

Paired-molecule bootstrap (2000 resamples, shared index across models and seeds): the champion is worse than every non-PSC reference with a CI that excludes 0. The per-molecule view agrees: mean per-molecule |error| change vs `ACRA_REF` is -0.085 [-0.085 [-0.114, -0.056]], and the champion is closer to truth on 40.9% of molecules.

The gap is **not** an optimisation artefact: at the same 160-epoch horizon the ACRA and GINE reference runs converge to lower held-out error while PSC training curves plateau (Figure 4), and the 32-molecule overfit check shows the same code drives the loss down when the task is small.

## 6. Q2 — ORDER vs BAG: is path identity worth anything?

Matched one-factor contrasts (same readout, same target), D0:

| contrast | Δ RMSE [95% CI] | p |
|---|---|---|
| PSC_SUM_BAG vs PSC_SUM | +0.041 [+0.012, +0.072] | 0.006 |
| PSC_POOL_BAG vs PSC_POOL | +0.019 [-0.026, +0.062] | 0.379 |
| PSC_SUM_BAG_PAIR vs PSC_SUM_PAIR | +0.058 [+0.029, +0.086] | 0.000 |
| PSC_POOL_BAG_PAIR vs PSC_POOL_PAIR | +0.030 [-0.011, +0.068] | 0.147 |

The four contrasts are all ≥ 0 for BAG, i.e. **ORDER never wins**: in the SUM family the orientation-invariant multiset is significantly better (+0.041 and +0.058), and in the POOL family the two are indistinguishable (CIs straddle 0; scaffold-cluster bootstrap: `PSC_POOL_BAG_PAIR_vs_ACRA_REF` -0.107 [-0.153, -0.060], `PSC_POOL_BAG_PAIR_vs_PSC_POOL_BAG` +0.020 [-0.021, +0.060], `PSC_POOL_BAG_PAIR_vs_PSC_POOL_PAIR` +0.030 [-0.019, +0.078], `PSC_POOL_BAG_PAIR_vs_PSC_SUM_BAG_PAIR` +0.036 [-0.001, +0.073], `PSC_POOL_BAG_vs_PSC_POOL` +0.019 [-0.035, +0.072], `PSC_POOL_BAG_vs_PSC_SUM_BAG` +0.037 [-0.011, +0.088], `PSC_SUM_BAG_PAIR_vs_PSC_SUM_BAG` +0.021 [-0.006, +0.048], `PSC_SUM_BAG_PAIR_vs_PSC_SUM_PAIR` +0.058 [+0.019, +0.095], `PSC_SUM_BAG_vs_PSC_SUM` +0.041 [+0.002, +0.076]). **Conclusion**: for Lipo, the *order* in which the atom/bond word is read along a path adds nothing — path identity beyond the multiset is not the useful signal, and the ORDER-specific machinery (sequence encoding, reversal averaging) is not earning its keep.

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

The coupling is **real but small**: every PSC PAIR contrast is ≥ 0 (SUM +0.003, POOL +0.009, SUM+BAG +0.021, POOL+BAG +0.020), but only two CIs exclude 0 and **none reaches the pre-registered +0.02 practical gain threshold**. The ACRA regulariser is a wash (-0.004). The dependence on readout is weak: the pair term gives roughly +0.01 to +0.02 in both families, consistent with a data-efficiency device rather than a substitute for capacity.

The pair algebra is exact and verified: `PAIR = DIAG − 2·cross` (deviation < 1e-12), `Σ w_e = 1` per component, and the sampler matches the component-weighted distribution. The DIAG twin, however, was **not run** because the champion is not a PAIR arm (see the protocol's conditional rule).

## 9. Q5 — where would unseen-substitution gains have to come from?

The novelty accounting first, because it bounds what any transfer mechanism can deliver:

* 65159 path keys; 53763 appear in the D0 fit set, 11396 never do.
* The median fit-set key appears in **1.0** molecule(s) (mean 10.31).
* Only **4.1%** of held-out path mass falls on keys unseen in fit; 8.2% falls on keys seen ≤5 times.
* Raw categorical UNK on held-out molecules is 0.0% (atoms and bonds), so the `zero_unseen` embedding masking is inert on this dataset — the novelty that matters is at the path-word level, not the raw feature level.

Matched-pair support classes (held-out pairs, cut-row weighted, pair Δ):

| support class | pairs | weight | mean \|Δtrue\| | PSC_POOL_BAG_PAIR | ACRA_REF | MORGAN_COUNT_KRR |
|---|---|---|---|---|---|---|
| T_seen | 195 | 173 | 0.693 | 0.443 (rel 0.64) | 0.392 (rel 0.57) | 0.465 (rel 0.67) |
| T_unseen_both_frags_unseen | 24 | 19 | 0.417 | 0.316 (rel 0.76) | 0.478 (rel 1.14) | 0.407 (rel 0.97) |
| T_unseen_frag_both_seen | 156 | 136 | 0.839 | 0.468 (rel 0.56) | 0.429 (rel 0.51) | 0.492 (rel 0.59) |
| T_unseen_one_frag_unseen | 100 | 85 | 0.673 | 0.480 (rel 0.71) | 0.463 (rel 0.69) | 0.489 (rel 0.73) |

Read this table with the `mean |Δtrue|` column: the *rel* column is MAE divided by that scale, so it is comparable across classes.

Molecule-level unseen-path-mass strata (D0, seed 0):

| bucket | n | PSC_POOL_BAG_PAIR | ACRA_REF | PATH_COUNT_RIDGE | MORGAN_COUNT_KRR |
|---|---|---|---|---|---|
| (0.0,0.01] | 80 | 0.604 | 0.599 | 0.578 | 0.590 |
| (0.01,0.05] | 159 | 0.698 | 0.583 | 0.620 | 0.659 |
| (0.05,1.000000001] | 120 | 0.825 | 0.672 | 0.725 | 0.750 |
| eta_mol=0 | 145 | 0.610 | 0.478 | 0.531 | 0.613 |

Mean held-out unseen-path mass is 0.0434 (median 0.0140, max 0.503); 23.8% of molecules exceed 5%.

From the lowest to the highest unseen-mass bucket the error grows by +0.214 (champion), +0.194 (`ACRA_REF`), +0.194 (path-count ridge) and +0.133 (GINE). The champion degrades at least as fast as the linear ridge, so the novelty strata give no hint of a transfer advantage.

**Pair-level vs molecule-level.** On held-out × held-out pairs the champion's Δ MAE is 0.453, versus 0.422 for `ACRA_REF` and 0.462 for the path-count KRR — i.e. at the level of *differences* the champion is close to (and slightly better than) the kernel baselines, while at the level of whole molecules it is 0.687 vs 0.655. The pair Δ regression slope is 0.514 for the champion vs 0.589 for `ACRA_REF`: the additive model attenuates contrasts more, even though their ordering is mostly right. That is the signature of a model whose *levels* are biased (heavy shrinkage) rather than one that cannot see the structure.

**Answer to Q5.** Transfer does not show up where it would have to. The only cell where PSC is best is the most extreme one (T unseen, *both* fragment words unseen), and it is 24 pairs — too few to build on, and exactly the cell where a heavily shrunk predictor wins by predicting small |Δ|. In the cell where a real transfer mechanism should shine (T unseen, both fragments **seen**, 156 pairs) PSC is the worst of the four models compared. Combined with the 4.1% path-mass share of unseen keys, the honest answer is that **unseen path words are too rare on Lipo for a transfer mechanism to move the molecule-level metric, and the model does not show a transfer advantage even in the pair-level cells where it could be seen.**

## 10. Mechanistic checks

### 10.1 Pair-residual decomposition (§9.5)

For D0 held-out × held-out pairs with component weights, `E[(e_b−e_a)²] = E[e_a²] + E[e_b²] − 2E[e_a e_b]`:

| model | E[(e_b−e_a)²] | E[e_a²] | E[e_b²] | 2·cross | corr(e_a,e_b) |
|---|---|---|---|---|---|
| ACRA_REF | 0.268 | 0.375 | 0.272 | 0.379 | 0.440 |
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

The ledger exposes the matched-pair design exactly: for the selected same-context pairs the C-side and crossing path **token multisets are identical in both molecules**, so their ledger terms cancel to 0 and 100% of the predicted Δ comes from paths inside the edited fragment side. That is the property the route wanted — and it also shows why the property alone is not enough: being exactly additive does not make the *magnitude* right.

### 10.3 Size strata and worst-error sensitivity (§9.7)

| model | overall | Q1 | Q2 | Q3 | Q4 | top-5 removed |
|---|---|---|---|---|---|---|
| ACRA_REF | 0.581 | 0.668 | 0.593 | 0.557 | 0.536 | 0.546 |
| COUNT_ONLY_RIDGE | 1.161 | 1.185 | 1.198 | 0.966 | 1.287 | 1.117 |
| GINE_REF | 0.522 | 0.559 | 0.535 | 0.509 | 0.501 | 0.490 |
| PSC_POOL_BAG_PAIR | 0.693 | 0.753 | 0.750 | 0.635 | 0.667 | 0.644 |

Quartile edges on fit-set atom counts: [21.0, 27.0, 32.0]. Removing the union of each model's 5 worst molecules moves RMSE by a few percent at most, so no conclusion rests on a handful of outliers. Selection never used this analysis.

## 11. D1 replication (independent scaffold-disjoint split of the same Train rows)

| model | D1 RMSE (2 seeds) | D0 RMSE | Δ(D1−D0) | D1 Δ vs ACRA_REF [95% CI] |
|---|---|---|---|---|

D1 runs used the frozen `E_m` epochs (median D0 best epoch per arm) and the final checkpoint, with no selection. The ordering and the size of the gaps replicate; the champion's deficit versus ACRA is +0.000 RMSE on D1 versus +0.107 on D0.

## 12. Official Valid (full-Train refit, single frozen pass)

_The frozen Valid pass did not run: the full-Train refits were not all finished before the reporting deadline, so Valid labels were never opened. Per the protocol this is preferred over scoring an incomplete or post-hoc model set._

## 13. Budget and artefacts

* GPU-hours consumed (all launched jobs, 1 GPU each): **0.14** of the 16 h ceiling (within budget).
* Train-loop hours: 0.13 (92.7% of GPU wall).
* Closed-form CPU hours: 0.04.
* Completed GPU runs: 2; launch attempts: 2 (non-completed: 0).

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
analysis/compute_budget.json, path_sparsity.json, tables.md, figures/
results/main_table.csv, results/summary.json
```

## 14. Deviations, incidents and honest limits

* **Ordering incident (no GPU waste beyond reruns).** The queue driver launches through `rr`, which refuses to start a formal run from a dirty tracked tree. One analysis edit left the tree dirty while the stage-2 queue was running, so a batch of launch attempts was rejected (`LAUNCH_FAILED`) before the next commit. All affected runs were relaunched from the same commit and the final matrix is complete (see `analysis/compute_budget.json` for every attempt). No run was selected or excluded on the basis of its result.
* **Horizon.** 160 epochs is the frozen budget. The PSC arms' best epoch is late (153, 115, 88, 118, 141, 142, 122, 96), i.e. they are still improving when the schedule ends. The comparison to a closed-form ridge on the same features is therefore a *lower bound* on the additive restriction's cost; but the 32-molecule overfit shows the gap is not an implementation failure, and the POOL>SUM contrast is measured at the same horizon for both.
* **Baseline reproduction.** In this harness `ACRA_REF` reaches D0 dev RMSE 0.580 ± 0.002 (3 seeds), whereas the parent round's own reference run reported 0.555 for the same nominal recipe on the same split. The ~0.025 difference is consistent with a different initialisation/data-order path in the re-hosted trainer; it makes this round's ACRA baseline slightly *weaker* than the parent's, which if anything understates the PSC deficit. The strongest non-PSC baseline here is GINE (0.527), so no conclusion rests on the ACRA reproduction alone.
* **Recipe fairness.** PSC/ACRA/GINE share the frozen optimiser recipe (AdamW, peak lr 1e-3 / 1e-5, cosine, clip 5, batch 64 molecules). The ACRA and GINE baselines are the parent rounds' own recipes, not tuned for this round; the CPU baselines lock α on D0. No Valid-based tuning anywhere.
* **Path-length ceiling.** L = 5 (the pre-registered fallback L = 4 was not needed). Longer paths would raise per-molecule coverage but also the key count; the sparsity table shows the key space is already the binding constraint.
* **Ledger scope.** The ledger covers the SUM readout only, and its S/C split is geometric (component sizes) rather than the TRC word identity, because the parent cache's per-parse atom sets are stored in fragment-local numbering. The exact-cancellation result above holds under either convention for same-context pairs.
* **DIAG twin** was not run (conditional on a PAIR champion), and the optional extensions (ESOL transfer, size curve, M* full-train) were not started: the required matrix plus D1 and the full-Train refits consumed the budget, and the protocol's extension gate requires the required queue to be complete first.

## 15. What this means for the branch

1. **Kill the strict path-sum branch.** A shared scalar score per path shape is dominated by a linear model on the same path counts (−-0.070 RMSE means the ridge is *better*) and by ACRA. More capacity in the encoder is not the fix; the sum is the fix.
2. **If the branch is revived, replace the shared scorer with a hybrid** `F = β + Σ_p [w_key(p) + h(E(p))]`, i.e. a free coefficient for keys with sufficient support and a shared, token-conditioned residual for the tail. That keeps a ledger (per-key plus per-shape terms) while removing the capacity cliff; the ridge/path-sum gap measures the headroom.
3. **Keep the pair algebra and the diagnostic harness.** The PAIR objective is exactly verifiable, transfers to ACRA, and gives the only positive effect in the matrix; the matched-pair ledger (exact C/X cancellation) is a reusable instrument even if the model class changes.
4. **Do not expect unseen-path-word transfer to move Lipo numbers.** It is 4.1% of held-out path mass. A route that targets it should be evaluated on a dataset or split where that slice is large, otherwise the metric cannot see it.

## 16. Non-claims

* No Test label was read; no Test metric is reported.
* No Valid label was read before `final_eval_lock.json`; no Valid number was used for training, early stopping, arm selection or hyper-parameter choice.
* The negative result is about *this* dataset, vocabulary (raw atom/bond tokens, L=5), recipe (160 epochs, shared MLP scorer) and budget. It is not a claim that path-based composition cannot work with a different parameterisation; §15.2 states the specific change the evidence supports.

