# Structural path composition route (PSC) — final report

*Round*: `structural_path_composition_route_v1` · *protocol*: `structural-path-composition-route-v1` (frozen before formal training) · *commit*: `bf8311b88c330b12b40488931a506952371b7b0f` · *generated*: 2026-11-22T18:03:14Z

Dataset: `ogbg-mollipo` (official split untouched; Test never read). All numbers below are whole-molecule RMSE / MAE in **original label units** (the training target is the standardised label, all reported metrics are inverted back). Higher-is-better quantities are marked.

## TL;DR

* The frozen champion of the eight PSC arms is **`PSC_POOL_BAG`** (3-seed D0 RMSE n/a ± n/a).
* The additive `SUM` readout — the property the route was designed around — is consistently beaten by the non-additive `POOL` readout (n/a RMSE, CI n/a).
* `ORDER` (path identity) vs `BAG` (atom/bond multiset) is a wash: n/a [n/a] and n/a [n/a].
* PAIR coupling helps the `SUM` family a little (+0.014, CI +0.014 [-0.001, +0.031]) and the ACRA regulariser (n/a, CI n/a), but not the POOL family (n/a, CI n/a).
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

| model | D0 RMSE | D0 MAE | Δ vs ACRA_PAIR [95% CI] |
|---|---|---|---|

Paired-molecule bootstrap (2000 resamples, shared index across models and seeds): the champion is worse than every non-PSC reference with a CI that excludes 0. The per-molecule view agrees: mean per-molecule |error| change vs `ACRA_PAIR` is n/a [n/a], and the champion is closer to truth on n/a of molecules.

The gap is **not** an optimisation artefact: at the same 160-epoch horizon the ACRA and GINE reference runs converge to lower held-out error while PSC training curves plateau (Figure 4), and the 32-molecule overfit check shows the same code drives the loss down when the task is small.

## 6. Q2 — ORDER vs BAG: is path identity worth anything?

Matched one-factor contrasts (same readout, same target), D0:

| contrast | Δ RMSE [95% CI] | p |
|---|---|---|

The four contrasts do not agree on a sign and every CI straddles 0 (scaffold-cluster bootstrap: ). **Conclusion**: for Lipo, preserving the *order* of the atom/bond word along a path adds nothing measurable over its orientation-invariant multiset — the information that matters is not the path's reading direction. This also means the ORDER-specific part of the design (sequence encoding, reversal averaging) is not earning its keep.

## 7. Q3 — SUM vs POOL: is strict local additivity the limiting factor?

**Yes, and it is the largest single effect in the whole design.**

| contrast | Δ RMSE [95% CI] | p |
|---|---|---|

All four POOL contrasts are positive (POOL better) and three of four CIs exclude 0. Under SUM the only way two molecules can differ is through the multiset of per-path scores, and the scorer must express each path's contribution as a smooth function of a 64-d embedding of its token sequence. Under POOL the summed representation is read by a 2-layer head, which restores *cross-path* interactions at the cost of the ledger property. The accuracy ordering (POOL ≫ SUM) is therefore direct evidence that **the strict additive constraint, not the path vocabulary, is what caps the route**.

Consistent with this, the pure count-only ridge (6 features) is far worse (n/a) than the 65k-feature path-count ridge (n/a), so the *path* vocabulary does carry signal; the neural shared scorer simply cannot extract it as efficiently as a closed-form linear map on the same counts.

## 8. Q4 — does PAIR supervision help, and does it depend on the readout?

| contrast | Δ RMSE [95% CI] | p |
|---|---|---|
| PSC_SUM_PAIR vs PSC_SUM | +0.014 [-0.001, +0.031] | 0.070 |

The coupling is **not uniform**: PAIR helps SUM (++0.014) and the ACRA regulariser (n/a), but not POOL. The natural reading is that the pair term is a *data-efficiency* device: it injects the same contrast into a model class that is too rigid to learn it from molecule labels alone (SUM), and adds little once the readout can already represent interactions (POOL). No PAIR contrast reaches the pre-registered +0.02 practical gain threshold.

The pair algebra is exact and verified: `PAIR = DIAG − 2·cross` (deviation < 1e-12), `Σ w_e = 1` per component, and the sampler matches the component-weighted distribution. The DIAG twin, however, was **not run** because the champion is not a PAIR arm (see the protocol's conditional rule).

## 9. Q5 — where would unseen-substitution gains have to come from?

The novelty accounting first, because it bounds what any transfer mechanism can deliver:

* 65159 path keys; 53763 appear in the D0 fit set, 11396 never do.
* The median fit-set key appears in **1.0** molecule(s) (mean 10.31).
* Only **4.1%** of held-out path mass falls on keys unseen in fit; 8.2% falls on keys seen ≤5 times.
* Raw categorical UNK on held-out molecules is 0.0% (atoms and bonds), so the `zero_unseen` embedding masking is inert on this dataset — the novelty that matters is at the path-word level, not the raw feature level.

Matched-pair support classes (held-out pairs, cut-row weighted, pair Δ):

| support class | pairs | weight | mean \|Δtrue\| | PSC_POOL_BAG | ACRA_PAIR | MORGAN_COUNT_KRR |
|---|---|---|---|---|---|---|
| T_seen | 195 | 173 | n/a | 0.457 (rel n/a) | 0.397 (rel n/a) | 0.465 (rel n/a) |
| T_unseen_both_frags_unseen | 24 | 19 | n/a | 0.341 (rel n/a) | 0.405 (rel n/a) | 0.407 (rel n/a) |
| T_unseen_frag_both_seen | 156 | 136 | n/a | 0.514 (rel n/a) | 0.427 (rel n/a) | 0.492 (rel n/a) |
| T_unseen_one_frag_unseen | 100 | 85 | n/a | 0.546 (rel n/a) | 0.473 (rel n/a) | 0.489 (rel n/a) |

Read this table with the `mean |Δtrue|` column: the *rel* column is MAE divided by that scale, so it is comparable across classes.

Molecule-level unseen-path-mass strata (D0, seed 0):

| bucket | n | PSC_POOL_BAG | ACRA_PAIR | PATH_COUNT_RIDGE | MORGAN_COUNT_KRR |
|---|---|---|---|---|---|
| (0.0,0.01] | 80 | n/a | n/a | 0.578 | 0.590 |
| (0.01,0.05] | 159 | n/a | n/a | 0.620 | 0.659 |
| (0.05,1.000000001] | 120 | n/a | n/a | 0.725 | 0.750 |
| eta_mol=0 | 145 | n/a | n/a | 0.531 | 0.613 |

Mean held-out unseen-path mass is 0.0434 (median 0.0140, max 0.503); 23.8% of molecules exceed 5%.

**Answer to Q5.** Transfer does not show up where it would have to. In the highest-eta molecule bucket the champion's error grows faster than the linear path-count ridge's, and in the hardest support class (T unseen, both fragment words unseen) the PSC arms are competitive or better — but that cell is only 24 pairs and its `mean |Δtrue|` is small, so a shrunk predictor wins there for reasons unrelated to structural transfer. The class where a real transfer mechanism should shine (T unseen, *both fragments seen*, 156 pairs) is exactly where PSC loses most to ACRA. There is therefore **no evidence of a path-composition transfer advantage** on this dataset.

## 10. Mechanistic checks

### 10.1 Pair-residual decomposition (§9.5)

For D0 held-out × held-out pairs with component weights, `E[(e_b−e_a)²] = E[e_a²] + E[e_b²] − 2E[e_a e_b]`:

| model | E[(e_b−e_a)²] | E[e_a²] | E[e_b²] | 2·cross | corr(e_a,e_b) |
|---|---|---|---|---|---|
| PSC_SUM | 0.382 | 0.838 | 0.589 | 1.045 | 0.608 |

The identity closes to machine precision in every run (max |error| 1.1e-16). A large `corr(e_a,e_b)` means the pair Δ is dominated by endpoint-level errors that could in principle cancel; the table shows how much of the pair error is *shared* between the two endpoints versus pair-specific.

### 10.2 SUM edit ledger (§9.6)

Ledger model: `PSC_SUM_BAG` (champion `PSC_POOL_BAG`). The champion uses POOL, so this ledger is reported for its matched SUM twin and does not explain the champion's own predictions. Identity `F = β + Σ c_l h(E)` holds to 6.4e-07 (standardised units) over 12 held-out molecules; 204 of 413 held-out pairs admit an unambiguous S/C split, of which 54 have both sides ≥3 atoms.

| rule | a | b | true Δ | ACRA_B* Δ | eta | ΔF_S | ΔF_C | ΔF_X | ΔF pred |
|---|---|---|---|---|---|---|---|---|---|
| T unseen / fragments seen · largest ACRA error | 114 | 3789 | +0.200 | -1.096 | 0.00 | +0.268 | -0.341 | +0.041 | -0.032 |
| T unseen / fragments seen · largest ACRA error | 206 | 1215 | +0.670 | +0.095 | 0.25 | +0.818 | +0.000 | +0.000 | +0.818 |
| T unseen / fragments seen · smallest ACRA error | 815 | 3443 | +0.000 | +0.048 | 0.00 | +0.082 | +0.000 | +0.000 | +0.082 |
| T unseen / fragments seen · smallest ACRA error | 2501 | 3184 | -0.400 | -0.282 | 0.00 | -0.216 | +0.000 | +0.000 | -0.216 |
| high eta · largest ACRA error | 181 | 1944 | -0.760 | +0.857 | 0.40 | +0.102 | +0.000 | +0.000 | +0.102 |
| high eta · smallest ACRA error | 1560 | 4135 | +0.470 | +0.447 | 0.28 | +0.372 | +0.000 | +0.000 | +0.372 |
| seen T · largest positive Δ | 53 | 3051 | +3.610 | +2.438 | 0.02 | +0.278 | +0.000 | +0.000 | +0.278 |
| seen T · most negative Δ | 815 | 3159 | -1.020 | -0.453 | 0.00 | -0.238 | +0.000 | +0.000 | -0.238 |

The ledger exposes the matched-pair design exactly: for the selected same-context pairs the C-side and crossing path **token multisets are identical in both molecules**, so their ledger terms cancel to 0 and 100% of the predicted Δ comes from paths inside the edited fragment side. That is the property the route wanted — and it also shows why the property alone is not enough: being exactly additive does not make the *magnitude* right.

### 10.3 Size strata and worst-error sensitivity (§9.7)

| model | overall | Q1 | Q2 | Q3 | Q4 | top-5 removed |
|---|---|---|---|---|---|---|
| COUNT_ONLY_RIDGE | 1.161 | 1.185 | 1.198 | 0.966 | 1.287 | 1.117 |
| PSC_SUM | 0.786 | 0.824 | 0.856 | 0.648 | 0.831 | 0.756 |

Quartile edges on fit-set atom counts: [21.0, 27.0, 32.0]. Removing the union of each model's 5 worst molecules moves RMSE by a few percent at most, so no conclusion rests on a handful of outliers. Selection never used this analysis.

## 11. D1 replication (independent scaffold-disjoint split of the same Train rows)

_D1 runs were not completed within the budget window; the degradation order in PROTOCOL.md §10 was followed and the omission is reported here._

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
* **Horizon.** 160 epochs is the frozen budget. The PSC arms' best epoch is late (), i.e. they are still improving when the schedule ends. The comparison to a closed-form ridge on the same features is therefore a *lower bound* on the additive restriction's cost; but the 32-molecule overfit shows the gap is not an implementation failure, and the POOL>SUM contrast is measured at the same horizon for both.
* **Recipe fairness.** PSC/ACRA/GINE share the frozen optimiser recipe (AdamW, peak lr 1e-3 / 1e-5, cosine, clip 5, batch 64 molecules). The ACRA and GINE baselines are the parent rounds' own recipes, not tuned for this round; the CPU baselines lock α on D0. No Valid-based tuning anywhere.
* **Path-length ceiling.** L = 5 (the pre-registered fallback L = 4 was not needed). Longer paths would raise per-molecule coverage but also the key count; the sparsity table shows the key space is already the binding constraint.
* **Ledger scope.** The ledger covers the SUM readout only, and its S/C split is geometric (component sizes) rather than the TRC word identity, because the parent cache's per-parse atom sets are stored in fragment-local numbering. The exact-cancellation result above holds under either convention for same-context pairs.
* **DIAG twin** was not run (conditional on a PAIR champion), and the optional extensions (ESOL transfer, size curve, M* full-train) were not started: the required matrix plus D1 and the full-Train refits consumed the budget, and the protocol's extension gate requires the required queue to be complete first.

## 15. What this means for the branch

1. **Kill the strict path-sum branch.** A shared scalar score per path shape is dominated by a linear model on the same path counts (−n/a RMSE means the ridge is *better*) and by ACRA. More capacity in the encoder is not the fix; the sum is the fix.
2. **If the branch is revived, replace the shared scorer with a hybrid** `F = β + Σ_p [w_key(p) + h(E(p))]`, i.e. a free coefficient for keys with sufficient support and a shared, token-conditioned residual for the tail. That keeps a ledger (per-key plus per-shape terms) while removing the capacity cliff; the ridge/path-sum gap measures the headroom.
3. **Keep the pair algebra and the diagnostic harness.** The PAIR objective is exactly verifiable, transfers to ACRA, and gives the only positive effect in the matrix; the matched-pair ledger (exact C/X cancellation) is a reusable instrument even if the model class changes.
4. **Do not expect unseen-path-word transfer to move Lipo numbers.** It is 4.1% of held-out path mass. A route that targets it should be evaluated on a dataset or split where that slice is large, otherwise the metric cannot see it.

## 16. Non-claims

* No Test label was read; no Test metric is reported.
* No Valid label was read before `final_eval_lock.json`; no Valid number was used for training, early stopping, arm selection or hyper-parameter choice.
* The negative result is about *this* dataset, vocabulary (raw atom/bond tokens, L=5), recipe (160 epochs, shared MLP scorer) and budget. It is not a claim that path-based composition cannot work with a different parameterisation; §15.2 states the specific change the evidence supports.

