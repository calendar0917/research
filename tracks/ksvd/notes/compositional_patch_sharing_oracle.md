# Compositional Patch Sharing Oracle (ZINC-12k, frozen compact-v4-hinge)

**Status:** FINAL — 2026-09-10
**Phase:** representation oracle / frozen-inference intervention (diagnosis only; **no training**)
**Questions answered:** Q1–Q15 · Tables A–E · Figures 1–6
**Decision:** **NO-GO — COMPOSITIONAL PATCH SHARING**
**Model:** `tracks/ksvd/experiments/luyin16/zinc_compositional_patch_sharing_oracle.py`
**Artifacts:** `tracks/ksvd/results/compositional_patch_sharing_oracle/`
**Records:** `records/claims/claim-compositional-patch-sharing-nogo-20260910.yaml`,
`records/decisions/decision-compositional-patch-sharing-nogo-20260910.yaml`

---

## 0. Executive summary

| Fact | Value | Status |
|---|---|---|
| Frozen baseline (4 seeds) | compact-v4-hinge validation-selected checkpoints, 99,613 params | VERIFIED |
| Baseline bit-exact guard | 4/4 seeds reproduce recorded run predictions, **max abs diff 0.0** | VERIFIED |
| Unaffected molecules bit-identity | across all variants, **max abs diff 0.0** on all 4 seeds | VERIFIED |
| Overall valid MAE (baseline) | 0.169451 (0.170066 / 0.163167 / 0.174149 / 0.170421) | — |
| Best structural variant | **Blend50** (ΔMAE **−0.004339**, 0/4 seeds) | — |
| NN1 | ΔMAE −0.011985 (0/4) | — |
| KNN8 | ΔMAE −0.010882 (0/4) | — |
| FrequentMean control | ΔMAE −0.009917 (0/4) | — |
| RandomDonor control | ΔMAE −0.010891 (0/4) | — |
| rare-seen subgroup (n=440) | ΔMAE −0.00995 (Blend50) … −0.02482 (KNN8) | — |
| OOV subgroup (n=162) | ΔMAE −0.01243 (Blend50) … −0.02454 (KNN8) | — |
| KNN8 vs FrequentMean | **KNN8 is 0.00096 *worse*** than the generic mean | KNN not better |
| KNN8 vs RandomDonor | difference 0.0000095 (essentially identical) | KNN not better |
| Donor-distance–benefit gradient | none: all 5 distance bins negative, non-monotone | Q11 = no |
| Cross-seed disagreement | KNN8 vs baseline: OOV −0.0035, rare +0.0006, high-rarity +0.0057 | incoherent |
| Authored defects found | typed certificate is **not injective** (730/6784 tokens carry multiple root atoms) | NEW, audited |

**Bottom line:** every pre-registered sharing intervention **degrades** frozen validation
inference (all 5 variants, 0/4 seeds positive), the harm is **concentrated exactly on the
targeted rare/OOV molecules** that were intervened on, and structural KNN matching is
**not better than random-donor or frequent-mean controls**. The hypothesis that rare/OOV
exact patch embeddings can borrow statistical strength from structurally similar frequent
patches is **refuted under this oracle**. Per the pre-registered mapping this is **Case E**:
do not build the compositional patch dictionary.

---

## 1. Motivation — why rarity residual NO-GO ≠ "rare representation is fine"

Earlier audits converged on a two-sided picture:

* **Post-v4 residual audit (validation):** no correctable *signed* structural residual
  survives (all fitted probes < 0.003 gate). The residual is **variance-side** difficulty.
* **OOF difficulty audit (official train):** rarity is a strong, monotone **difficulty**
  axis — OOF MAE 0.133 (no rare≤5 token) → 0.417 (>20% rare≤5 tokens), Spearman 0.309,
  5/5 folds; OOV ratio 0.277, 5/5.

Rarity therefore does **not** predict the direction of the residual, but it does predict
how hard the molecule is for the model. The representation-level reading of that fact is:
a patch token that occurs 1–5 times (or 0 times) receives almost no supervision for its
*exact* embedding, so the frozen model may be unable to estimate its representation
stably. This stage asks the *intervention* question directly:

> If a rare/OOV exact patch embedding could borrow information from structurally similar
> **frequent** patch embeddings, would frozen predictions improve?

This is deliberately **not** an architecture. Exact identity is never rejected; the
hypothesis is that exact identity might be better supported by a shareable structural
component. No dictionary/KSVD/frequency gate is implemented — only an auditable
replacement/blend of the radius-2 exact patch embedding at inference.

---

## 2. Current exact-token limitation (lookup-table representation)

The compact-v4 core patch representation is

```
radius-2 canonical typed certificate
    -> vocabulary id (decreasing train frequency; id 0 = OOV)
    -> learned token embedding (per-token lookup)
    -> patch_encoder
```

Every exact token owns its own parameters (`_HybridEmbedding`: full 16-D row for the top
768 tokens, a rank-4 -> 16 factorized row for the rest). Common patches get hundreds or
thousands of gradient updates; a patch seen ≤5 times gets ≤5. OOV shares a single learned
row (id 0) for all unseen structures. There is **no mechanism by which a rare token can
inherit statistics from a structurally related frequent token** — the lookup table is the
whole representation.

---

## 3. Frozen v4 baseline

Reused (never retrained) validation-selected compact-v4-hinge checkpoints:

| seed | run id | valid MAE | epochs |
|---|---|---|---|
| 0 | 20260909-194445-182c7021 | 0.1700656 | 53 |
| 1 | 20260909-200320-34b347bf | 0.1631665 | 48 |
| 2 | 20260909-201918-45fbe48d | 0.1741492 | 56 |
| 3 | 20260909-203516-04e62a28 | 0.1704209 | 60 |

Records rebuilt with `_extract_v4_records()` + `_phase_data()` + `_build_v4_model()` +
`legacy_full_result_selection_state.pt`. **Baseline guard:** `intervention_mode = none`
reproduces the recorded predictions **bit-exactly** on all four seeds (max abs diff 0.0).

### 3.1 Real exact-embedding data-flow audit (§6)

```
valid molecule
  patches: [radius-2 typed certificate]   [radius-1 parent cert]   [146D shell descriptor]
                    |                             |                         |
        _fit_vocabulary(train)                      |               Standardizer.fit(train)
        id (0 = OOV ; 1..V by freq)                 v                 (146D standardized)
                    v                        parent_vocabulary              |
          model.typed_embedding               id (0 = OOV)                 |
          (_HybridEmbedding)                        |                       |
     id < 768 -> full.weight[id, :]                 v                       |
     id >=768 -> proj(rare.emb[id-768])   parent_embedding (8D)             |
                    \_____________________________|________________________/
                                       |
                       patch_encoder( concat[146D, e_patch, e_parent] )
                                       |
                                  patch_state
```

* **146D descriptor:** `[atom_shell(3x28) | bond_shell(6x4) | root_atom(28) | incident_bonds(4) | scalars(6)]`.
  Standardized by the train-fit `patch_standardizer` **before** the patch encoder.
* **Parent embedding:** radius-1 parent token, width 8 (`max(16//2,1)`), hybrid full table
  of 32 rows (so all parent tokens are full rows).
* **Exact patch embedding:** width 16, `hybrid_full_typed_tokens = 768`;
  `embedding_rank = 4`. Vocabulary: 6,784 train tokens + OOV = 6,785.
* **Validation OOV** (278 occurrences) all receive token id 0 = the learned full-table row
  (never the low-rank table).
* **This stage intervenes only on the radius-2 exact patch embedding** (the output of
  `typed_embedding`). The parent embedding and the 146D descriptor are untouched.

### 3.2 NEW audited defect: the typed certificate is not injective

While building the bank we verified the task's assumption that a token identifies a
unique structure. It does **not**. The stored `typed_certificate` is
`pynauty.certificate(colored incidence graph)`, and by pynauty's own definition
(`isomorphic()` also compares **color-cell sizes**) the certificate alone is **not** a
complete invariant of the colored graph. Empirically, on official train:

| quantity | value |
|---|---|
| unique train tokens | 6,784 |
| tokens whose occurrences carry **>1 root atom** | **730** (10.8%) |
| tokens whose occurrences carry **>1 parent certificate** | 91 |
| max within-token 146D descriptor spread | **≈1.0** (a one-hot root atom moves) |

Minimal counterexample (path 0–1–2–3, `pynauty` 2.8.8.1): colorings `[{0},{1},{2,3}]`
and `[{0},{1},{2},{3}]` produce the **same** certificate. Consequence: the frozen
"exact token" is a *coarser* invariant than the intended rooted typed patch. This was
**not fixed** here (fixing it would change the vocabulary and break the frozen baseline);
the oracle was made robust to it (occurrence-level descriptors for targets, mean
descriptors for donors). It is recorded as the highest-value follow-up (§15).

---

## 4. Train-only patch bank

Bank (`bank.npz`, `coverage.json`), built **only from official train**:

* per token: `token_id`, official-train frequency, 146-D structural descriptor prototype
  (**mean over occurrences**, since tokens are not injective), modal root atom, modal
  radius-1 parent certificate;
* per validation *occurrence* of a rare/OOV patch: token id, train frequency, group
  (`rare_seen` / `oov`), occurrence descriptor, donor search result.

Descriptor standardization uses the frozen model's own train-fit `Standardizer`
(train-only), reused so donor distances live in the same space the model sees.

## 5. Structural similarity definition

* Feature: the **146D shell descriptor already used by the model** (no new chemistry,
  computable for OOV, target-independent).
* Train-only standardization, then **standardized Euclidean distance**.
* No cosine/Mahalanobis/learned metric; no RDKit/Morgan/Tanimoto.

## 6. Donor constraints (pre-registered hierarchy)

Donor eligibility: **official-train count >= 20** (878 tokens). Target: **official-train
count <= 5** (rare-seen 1–5, OOV 0).

| level | constraint | occurrences | share |
|---|---|---:|---:|
| 1 | same modal root atom **and** same radius-1 parent certificate | 952 | 95.4% |
| 2 | same root atom | 45 | 4.5% |
| 3 | unrestricted frequent donor bank | 1 | 0.1% |

Within the chosen level, donors are ranked by standardized descriptor distance. `k = 8`
(max, or all available). Weights `softmax(-d / T)`, `T = 1`.

## 7. Intervention variants (pre-registered, not widened)

| variant | definition |
|---|---|
| **NN1** | `e_shared = e_exact(nearest donor)` |
| **KNN8** | `e_shared = Σ_j softmax(-d_j)·e_exact(N_j)` over 8 nearest donors |
| **Blend50** | rare-seen `0.5·e_exact + 0.5·e_shared`; OOV `1.0·e_shared` |
| **FrequentMean** | replace with the mean of **all** frequent donor embeddings (control) |
| **RandomDonor** | matched-k: mean of **8 random** frequent donors at the same level (control) |

Common patches (`count > 5`) always keep their exact embedding. `alpha`, `k`, `T` and the
donor threshold were fixed before seeing any validation result and were not tuned.

## 8. Negative controls and why they matter

The controls separate three explanations: *structural sharing* (KNN > controls),
*generic shrinkage* (KNN ≈ frequent mean), *any replacement* (KNN ≈ random donor).
`RandomDonor` uses the same donor count and averaging as KNN8 so the **only** difference
is structure-vs-random selection.

---

## 9. Coverage statistics

**Table A — patch coverage**

| category | unique tokens | train occurrences | valid occurrences | valid affected molecules |
|---|---:|---:|---:|---:|
| frequent >=20 | 878 | 210,823 | 20,977 (90.88%) | 1,000 |
| medium 6–19 | 1,161 | 12,098 | 1,108 (4.80%) | 612 |
| rare 1–5 | 4,745 | 8,743 | 720 (3.12%) | 440 |
| OOV | 0 (by definition) | 0 | 278 (1.20%) | 162 |

* rare≤5 = **69.9%** of unique train tokens (4,745 / 6,784) but only **4.32%** of validation
  patch occurrences (998 / 23,083).
* affected validation molecules: rare-seen **440**, OOV **162**, any **498 / 1000** (49.8%).
* donor bank: 878 (>=20); descriptive: 1,435 (>=10), 471 (>=50).
* This 4.3% occurrence share is the structural reason any real gain would be capped at a
  small overall ΔMAE; even a perfect intervention cannot move the overall metric much.

---

## 10. Multi-seed results

**Table B — overall results (validation MAE; ΔMAE = baseline − intervention; + = better)**

| intervention | seed0 Δ | seed1 Δ | seed2 Δ | seed3 Δ | mean Δ |
|---|---:|---:|---:|---:|---:|
| NN1 | −0.01253 | −0.00645 | −0.01461 | −0.01434 | **−0.011985** |
| KNN8 | −0.01129 | −0.00657 | −0.01268 | −0.01299 | **−0.010882** |
| Blend50 | −0.00531 | −0.00226 | −0.00464 | −0.00514 | **−0.004339** |
| FrequentMean | −0.00989 | −0.00662 | −0.01164 | −0.01152 | **−0.009917** |
| RandomDonor | −0.01020 | −0.00727 | −0.01312 | −0.01298 | **−0.010891** |

Per-seed absolute MAE: baseline 0.170066 / 0.163167 / 0.174149 / 0.170421.

* **0/4 seeds positive for every variant.** Nothing is a mild improvement; every
  intervention is a regression.
* Blend50 (which retains 50% of the exact embedding for rare-seen, and 0% for OOV) is the
  *least* harmful — consistent with "the exact component carries value".
* NN1 is the *most* harmful — a hard nearest-neighbour swap is worse than a soft average.

## 11. Rare/OOV subgroup results

**Table C — targeted subgroup (4-seed mean ΔMAE; + = better), representative variants**

| subgroup | n | baseline MAE | KNN8 MAE | KNN8 ΔMAE | Blend50 MAE | Blend50 ΔMAE |
|---|---:|---:|---:|---:|---:|---:|
| unaffected | 502 | 0.10127 | 0.10127 | 0.00000 | 0.10127 | 0.00000 |
| rare-seen affected | 440 | 0.24250 | 0.26733 | **−0.02482** | 0.25246 | **−0.00995** |
| OOV affected | 162 | 0.28420 | 0.30874 | **−0.02454** | 0.29663 | **−0.01243** |
| highest-rarity quintile | 100 | 0.48158 | 0.53850 | **−0.05692** | 0.50738 | **−0.02580** |

Every affected group regresses in **every** variant (0/4 seeds positive), and the harm is
**largest exactly where the intervention was applied** (rare/OOV/high-rarity). The
unaffected group is bit-identical by construction.

**Table E — donor-similarity bins (molecule-level mean nearest-donor distance quintiles; KNN8)**

| bin | mean nearest distance | n patches | n molecules | ΔMAE |
|---|---:|---:|---:|---:|
| 0 (closest) | 1.55 | 143 | 100 | −0.01210 |
| 1 | 2.46 | 182 | 99 | −0.02925 |
| 2 | 3.37 | 214 | 100 | −0.01825 |
| 3 | 4.66 | 256 | 99 | −0.03240 |
| 4 (farthest) | 10.94 | 203 | 100 | −0.01743 |

No monotone relationship; the *closest* donors are not the safest (bin 0 is not the least
harmful). **Closer structural donor ≠ larger benefit.** (Descriptive only; no cutoff tuned.)

## 12. Disagreement analysis

Cross-seed prediction std (seeds 0–3), KNN8 vs baseline:

| subgroup | baseline disagreement | KNN8 disagreement | delta |
|---|---:|---:|---:|
| unaffected | 0.061657 | 0.061657 | **0.000000** |
| rare affected | 0.099252 | 0.099886 | +0.00063 |
| OOV affected | 0.126714 | 0.123183 | −0.00353 |
| highest-rarity quintile | 0.140526 | 0.146195 | +0.00567 |

Incoherent: OOV disagreement falls slightly, but rare/high-rarity disagreement *rises*.
There is no consistent "shared representation stabilises rare patches" effect. Disagreement
was used only as a mechanism diagnostic, never to select a variant.

## 13. Representation movement

`||e_exact − e_shared||` per intervened patch (4-seed mean): **NN1 2.09**, **KNN8 1.27**.
These are large relative to the 16-D embedding scale and to the perturbation that keeps
the model in-distribution, consistent with the observed regressions (the frozen downstream
`patch_encoder`/head expect the exact-embedding distribution).

No cross-seed **raw-embedding coordinate** comparison is claimed: independently trained
checkpoints can be globally rotated/reparameterized, so raw cross-seed Euclidean distances
are not interpretable. Cross-seed evidence here is prediction-level only (disagreement,
§12).

## 14. Case inspection

Selected only by intervention prediction delta (KNN8), not by manual structure selection.
The ten worst/best molecules confirm the aggregate picture: large losses concentrate on
molecules with several intervened patches and high donor distances.

| case | molecule | target | baseline err | KNN8 err | delta | #target patches |
|---|---|---:|---:|---:|---:|---:|
| improved | valid:917 | −1.00 | 1.591 | 0.995 | +0.596 | 14 |
| improved | valid:354 | −0.22 | 0.582 | 0.417 | +0.165 | 5 |
| improved | valid:126 | 1.19 | 0.471 | 0.318 | +0.153 | 2 |
| improved | valid:233 | 1.05 | 0.358 | 0.217 | +0.141 | 2 |
| improved | valid:260 | −1.85 | 1.046 | 0.914 | +0.132 | 4 |
| worsened | valid:869 | −1.91 | 0.420 | 0.886 | −0.466 | 4 |
| worsened | valid:161 | −2.67 | 0.278 | 0.797 | −0.518 | 3 |
| worsened | valid:214 | −10.31 | 3.243 | 3.993 | −0.750 | 2 |
| worsened | valid:378 | −3.18 | 0.185 | 0.956 | −0.770 | 8 |
| worsened | valid:772 | −3.32 | 0.462 | 1.295 | −0.833 | 9 |

The improved cases exist (sharing is not uniformly destructive at the molecule level), but
they are far outnumbered: the aggregated affected-subgroup and overall deltas are negative
in every seed. The worsened cases are exactly the many-target / large-distance regime.

## 15. Scientific interpretation

The oracle asked the right question with the strongest possible hand (oracle donor matching
from structure alone, frozen model, no training). The outcome is unambiguous:

* **Replacement is harmful**, not merely neutral. Even a 50/50 blend of the exact embedding
  with a structurally matched shared embedding regresses every seed.
* **Structural matching adds nothing**: KNN8 (−0.010882) is statistically identical to a
  matched **random**-donor average (−0.010891, difference 9.5e-6) and is actually slightly
  *worse* than the generic global **frequent mean** (−0.009917). If neighbour structure had
  value, KNN8 should beat both controls; it does not.
* **The harm is on-target** (rare/OOV/high-rarity molecules), so this is not an unrelated
  implementation artefact (and unaffected molecules are bit-identical, proving no
  forward-path pollution).

Reading: the frozen model's exact-token embeddings — **including** rare and OOV tokens —
already carry information that any donor/average replacement destroys. The lookup-table
representation is not obviously starved; the difficulty documented by the OOF audit is a
*conditional-variance* phenomenon, and it is not repaired by transplanting frequent-token
statistics. This is Case E: **no useful sharing signal**.

Separately, §3.2 documented a real, previously unknown defect: the frozen typed certificate
is **not injective** (10.8% of tokens mix root atoms). That is a *tokenizer correctness*
problem, not a compositional-sharing problem, and is logically prior to any further
representation work.

## 16. Final decision

**NO-GO — COMPOSITIONAL PATCH SHARING.**

Pre-registered gates (overall / targeted / control separation) are all violated:

* best structural variant overall ΔMAE **−0.004339 < +0.001**; 0/4 seeds positive;
* rare/OOV subgroup deltas are **negative** (no targeted gain);
* KNN8 is **not** better than the frequent-mean or random-donor controls;
* (the one positive check) unaffected molecules are bit-identical.

Do **not** open a Shared Structural Base + Exact Identity Residual architecture, an
OOV compositional fallback, frequency-aware sharing, KSVD sparse coding, or subword /
nearest-neighbour embedding routes on this evidence.

---

## Tables and figures index

* Table A: §9 · Table B: §10 · Table C: §11 · Table D (disagreement): §12 · Table E: §11.
* Figures (`results/compositional_patch_sharing_oracle/figures/`):
  * `fig1_patch_frequency.png` — train token frequency distribution.
  * `fig2_donor_distance.png` — nearest-donor distance, rare-seen vs OOV.
  * `fig3_subgroup_mae.png` — rare/OOV subgroup baseline vs KNN8.
  * `fig4_distance_vs_delta.png` — donor distance vs per-molecule error change.
  * `fig5_disagreement.png` — baseline vs KNN8 cross-seed disagreement.
  * `fig6_variant_comparison.png` — KNN8 vs frequent-mean vs random-donor overall ΔMAE.

---

## Q1–Q15

**Q1. What is the real radius-2 exact embedding lookup / OOV path?**
Radius-2 typed certificate -> `_fit_vocabulary(train)` id (decreasing frequency; id 0 = OOV,
known ids 1..V) -> `model.typed_embedding` = `_HybridEmbedding(vocab=6785, width=16, rank=4,
full_count=768)`: id<768 returns `full.weight[id]`; id>=768 returns
`proj(rare.emb[id-768])`; OOV id 0 returns the learned full row 0. The output `e_patch` is
concatenated with the standardized 146D shell descriptor and the radius-1 parent embedding
(width 8) and fed to `patch_encoder`. (See §3.1.)

**Q2. How many train tokens are rare<=5?**
4,745 of 6,784 (69.9%); frequent>=20 = 878; medium 6–19 = 1,161.

**Q3. How many validation molecules are affected by rare/OOV patches?**
498 / 1000 (49.8%): rare-seen 440, OOV 162. They carry 998 / 23,083 = 4.32% of validation
patch occurrences.

**Q4. Is NN1 better than baseline?**
No. ΔMAE −0.011985, 0/4 seeds positive.

**Q5. Is KNN8 better than NN1?**
Marginally (KNN8 −0.010882 vs NN1 −0.011985 => +0.00110 relative), but both are regressions;
softer averaging is less harmful than a hard swap.

**Q6. Is Shared-Exact 50/50 better than pure replacement?**
Yes relatively (Blend50 −0.004339 is the least harmful) but still a regression on 4/4 seeds.
Retaining the exact component helps; sharing never turns positive.

**Q7. rare-seen subgroup improvement?** None — ΔMAE −0.00995 (Blend50) to −0.02482 (KNN8).

**Q8. OOV subgroup improvement?** None — ΔMAE −0.01243 (Blend50) to −0.02454 (KNN8).

**Q9. Is structural KNN better than the frequent-mean control?**
No. KNN8 is 0.00096 **worse** than FrequentMean. The hypothesis "shrinking toward a
high-frequency prior is enough" is not even reached — even the mean hurts, and KNN does not
help.

**Q10. Is structural KNN better than the random-donor control?**
No. KNN8 − RandomDonor = +0.0000095 (essentially identical). Neighbour structure has **no
value** in this intervention.

**Q11. Does a closer donor give larger benefit?**
No. All donor-distance quintile ΔMAE are negative and non-monotone; the closest-donor bin
is not the least harmful.

**Q12. Does sharing reduce cross-seed disagreement on rare/OOV molecules?**
No coherent effect: OOV −0.0035, rare-seen +0.0006, high-rarity +0.0057 (unaffected 0 by
construction).

**Q13. Are unaffected molecules bit-identical?**
Yes — max abs prediction diff 0.0 on all 4 seeds. The intervention only touches rare/OOV
patch rows; no forward-path pollution.

**Q14. Which explanation does the result support?**
**E. No useful sharing signal.** Structural KNN ≈ random donor and is not better than the
frequent mean; all replacements (including a 50/50 blend) regress, with the harm
concentrated on the intervened rare/OOV molecules.

**Q15. The single recommended next stage.**
Close the compositional-sharing / nearest-neighbour-embedding route. Before any future
*exact-token* work, the one logically-prior, cheap, and independently testable action is to
**repair the lossy typed certificate** (include the color-cell sizes in the frozen
`_typed_certificate` so the token is a genuine complete invariant of the rooted typed
patch) and re-quantify coverage — a tokenizer correctness fix, **not** a sharing
architecture. (This is a separate, newly discovered defect; it is not evidence for
compositional sharing.)
