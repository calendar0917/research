# Typed Patch Tokenizer Correctness Repair + Re-baseline (ZINC, official train/valid)

**Status:** FINAL — 2026-09-10
**Phase:** representation-correctness repair + single-variable re-baseline (v2/v4)
**Questions answered:** Q1–Q16 · Tables A–G · Figures 1–6
**Decision:** **Case C — CORRECTNESS FIX REVEALS ACCIDENTAL SHARING BENEFIT** (primary v4:
mean valid degradation −0.0046 ≥ 0.003; v2 −0.0067). The corrected tokenizer is promoted as
the *semantically correct* method; the historical certificate-derived tokenizer is retained
as the *historical benchmark baseline*. No rollback of correctness.
**Code:**
`experiments/luyin16/typed_patch_tokenizer.py` (versioned keys + oracle),
`experiments/luyin16/zinc_typed_tokenizer_audit.py` (audit),
`experiments/luyin16/zinc_typed_tokenizer_model_analysis.py` (model-side analysis),
`tests/test_typed_patch_tokenizer.py` (regression suite).
**Artifacts:** `results/typed_patch_tokenizer_correctness/` (`audit.json`, `occurrences.npz`,
`keys.json`, `alias_molecules.json`, `figures/`, `model_analysis/`).
**Records:** `records/claims/claim-typed-tokenizer-correctness-*`, `records/decisions/decision-typed-tokenizer-*`.

---

## 0. Executive summary

| Fact | Value | Status |
|---|---|---|
| Historical r2 token = ? | `pynauty.certificate` of the colored incidence graph | the certificate is the **canonical adjacency only**; it does **not** encode the vertex coloring |
| What it actually identifies | the rooted **uncolored** incidence topology | consequence: atom/bond/root colors are dropped |
| Historical r2 tokens (train) | 6,784 | — |
| Aliased r2 tokens (>1 true typed class) | **2,363** (34.8%) | of which root-only 730 (10.8%), non-root 1,633 |
| Aliased r2 occurrences | **218,718 / 231,664** (94.4%) | affected molecules 10,000/10,000 |
| Max true classes per historical token | **104** | mean split multiplicity 4.57 |
| Historical parent (r1) tokens | 31 | **23 aliased (74%)**, 99.99% of occurrences, max 98 classes |
| Corrected key | certificate + canonical semantic color sequence | complete invariant of the colored incidence graph |
| Correctness gates (permutation / soundness / completeness / determinism / adversarial) | **0 failures** | 16,000 permutation cases + 489k soundness oracle calls |
| Corrected r2 vocabulary | 15,218 unique types (capped at 8,192) vs 6,784 | valid OOV occurrences 278 → 1,163 (4.2×) |
| Parameters | 99,613 → 107,201 (+7.6%, v4) · 98,549 → 106,137 (+7.7%, v2) | natural vocab-row growth |
| v2 seed0 valid | 0.184158 → **0.208384** (Δ −0.0242) | historical guard bit-identical |
| v4 seed0 valid | 0.170066 → **0.178158** (Δ −0.0081) | historical guard bit-identical |
| v4 4-seed valid | mean 0.169451 → **0.174062** (paired Δ **−0.0046**, 1/4 seeds positive) | Case C |
| v2 4-seed valid | mean 0.182844 → **0.189548** (paired Δ −0.0067, 2/4) | mild regression |
| v4 4-seed benchmark test | 0.136885 → **0.140895** (Δ −0.0040, 1/4) | same direction |
| Alias-burden gradient | higher burden → larger regression (q1 −0.0020 → q3 −0.0082) | correction hurts where alias was heaviest |
| Cross-seed disagreement | corrected **higher** (Δ −0.0036 overall; −0.0057 high-burden) | hypothesis "correction reduces disagreement" **refuted** |

**Bottom line.** The historical *exact typed patch token* was in fact a **rooted uncolored
topology token**; the intended typed identity was recovered only by the 146D continuous
descriptor and the (also-aliased) parent token. The corrected tokenizer is provably correct
(0 gate failures) but **does not improve and slightly degrades** the current compact family:
v4 −0.0046 valid / −0.0040 test; v2 −0.0067 valid. The harm grows with alias burden and is
worst on the *most* aliased tokens, and the fixed/full vocabulary is not the cause (an
uncapped corrected diagnostic is even worse, 0.1993). This is consistent with the historical
collision acting as **accidental parameter sharing / regularisation** — the coarse topology
token was a shared structural base and the continuous descriptor carried the chemistry. Per
the pre-registered mapping this is **Case C**.

---

## 1. Discovery

The defect was found while building the frozen compositional patch sharing oracle
(`notes/compositional_patch_sharing_oracle.md` §3.2): the donor bank audit assumed one token
= one structure, and a check of "root atom diversity inside a token" returned **730 / 6,784**
radius-2 tokens whose occurrences carry more than one modal root atom. That is a *representation
correctness* defect, logically prior to any compositional-sharing question, and it was recorded
there as the single highest-value follow-up. This stage executes that follow-up.

## 2. Historical semantics — why `pynauty.certificate` is insufficient

`certificate(g)` (pynauty 2.8.8.1, `nautywrap.c::graph_cert`) returns `g->cmatrix`, the
**canonical adjacency matrix** produced by nauty under the initial color partition. The
partition is used to constrain the canonical labelling, but the partition itself is **not part
of the returned bytes**. pynauty's own `isomorphic(a,b)` compensates with an extra
`len(cell)`-sequence comparison, but the bare certificate does not.

Minimal counterexample (path `0–1–2–3`): with the incidence construction used here, a patch
rooted at atom 0 with atom types `[0,1,1,1]` and a patch with types `[0,1,2,1]` produce the
**same certificate bytes** (verified in `test_5_known_historical_collision_separated`). More
generally the certificate is an invariant of the **uncolored** incidence topology: the
atom type, bond type, root designation and root-distance class are all invisible to it.

## 3. Intended equivalence relation (recovered from the code, not assumed)

Two patches `P, Q` are equivalent iff there is a graph isomorphism of their incidence graphs
preserving every semantic color label. The incidence construction (`build_colored_incidence`,
identical to the historical one) attaches to each vertex:

* **atom vertex**: `("node", is_root, distance_from_root, atom_type)`
* **bond/incidence vertex**: `("edge", bond_type)`

So the equivalence relation is exactly *isomorphism of the vertex-colored incidence graph*,
preserving atom type, bond type, root designation, root-distance class and the node-vs-edge
incidence type. No other semantic color is active on this path.

## 4. Ground-truth isomorphism oracle (independent of the key)

`typed_isomorphic_vf2` uses **networkx VF2** with a node matcher on the full semantic color
tuple — a completely different implementation from nauty. It was validated against
brute-force labelled color-preserving isomorphism on random small colored graphs (2,000 cases,
`test_9`). A pynauty-based oracle (`typed_isomorphic`, with an explicit ordered-label check
that bare `isomorphic` misses for color-label identity) is retained as a cross-check. The
oracle is **never** on the training path; it only validates keys.

## 5. Corrected canonical key design

```
typed_tokenizer_v1_historical : bytes(pynauty.certificate(incidence))
typed_tokenizer_v2_corrected  : b"typedpatchkey/v2;" + certificate + b"|"
                                + repr(tuple(semantic_color[lab[i]] for i in range(n)))
```

where `lab = pynauty.canon_label(incidence)` is the canonical labelling and the color cells
are ordered by `sorted(color_groups, key=repr)` (a total, process-independent order). The
canonical **semantic color sequence** carries color-label identity, so two patches with the
same topology but different atom/bond/root colors cannot collide. This is candidate 2 from the
brief (canonical serialization of adjacency + semantic labels); candidate 1 (certificate +
color-cell-size signature) was rejected because it does not carry *which* label lives in each
cell. Deterministic encoding, no `hash()`, no unordered iteration.

## 6. Correctness gates (Table B)

| test | cases | failures |
|---|---:|---:|
| permutation invariance (r2 / r1) | 16,000 / 16,000 | 0 / 0 |
| same-key soundness (train r2 / r1, valid r2 / r1) | 216,446 / 231,152 / 18,362 / 22,783 | 0 / 0 / 0 / 0 |
| true-isomorphic completeness (no over-split) | all aliased buckets | 0 |
| deterministic rerun (r2 / r1) | 300 / 300 | 0 / 0 |
| adversarial separation | 5 pre-built cases | 0 |
| **total** | — | **0** |

Adversarial cases (all `key_left != key_right` and `oracle == False`): different root atom,
different atom type at same root, different bond type, same raw adjacency with a different
root position, star-vs-path at same root. The first two collide under the historical
certificate (documented in the audit JSON), the latter three do not.

## 7. Historical collision statistics (Table A, official train)

**radius-2 patch tokens**

| metric | value |
|---|---:|
| historical r2 tokens | 6,784 |
| aliased r2 tokens | **2,363** (34.8%) |
| aliased occurrences | **218,718 / 231,664** (94.4%) |
| affected molecules | 10,000 / 10,000 |
| max true classes per token | **104** |
| mean split multiplicity | 4.57 |
| root-only collisions (differing root atom) | **730** |
| non-root typed collisions (same root, different typed class) | **1,633** |

The `730` reproduces the compositional-oracle number exactly; the fuller audit shows the defect
is far larger than root-atom mixing alone: **1,633** tokens collide with an identical root atom,
and **94.4%** of patch occurrences sit in an aliased bucket. Root-diversity distribution (number
of distinct root atoms within an aliased token): 1 → 1,633, 2 → 480, 3 → 150, 4 → 73, 5 → 21,
7+ → 5.

## 8. Parent-token audit (radius-1)

The **same `_typed_certificate`** produces the parent token, and it is aliased *more* than the
radius-2 token:

| metric | parent (r1) |
|---|---:|
| historical parent tokens | 31 |
| aliased parent tokens | **23** (74.2%) |
| aliased occurrences | 231,651 / 231,664 (99.99%) |
| max / mean split multiplicity | 98 / 21.9 |
| root-only / non-root collisions | 15 / 8 |

Because a radius-1 patch is a star around the root, its uncolored topology is essentially the
root **degree**; the historical parent token therefore collapsed all radius-1 neighbourhoods
with the same degree regardless of neighbour atoms and bond types. This is why there were only
31 historical parent tokens. The corrected parent vocabulary has 512 types.

## 9. Vocabulary shift (Table C, official train → validation)

| category | historical | corrected | delta |
|---|---:|---:|---:|
| r2 unique train types | 6,784 | **15,218** | +8,434 (2.24×) |
| r2 vocab (known, capped at 8,192) | 6,784 | 8,192 (truncated) | +1,408 |
| parent unique train types | 31 | 512 | +481 |
| rare ≤5 (in-vocab) | 4,745 | 4,391 | −354 |
| frequent ≥20 | 878 | 1,458 | +580 |
| valid OOV types | 266 | 1,087 | +821 |
| valid OOV occurrences | 278 | **1,163** | +885 (4.2×) |

The corrected identity fragments the token distribution heavily. `max_typed_tokens=8192` is
**binding** for the corrected tokenizer (historical 6,784 < 8,192), so the corrected vocabulary
is a top-8,192 frequency truncation and the ~7,026 rarer corrected types share the OOV row.

## 10. Parameter shift (Table D)

| model | historical params | corrected params | Δ | Δ% |
|---|---:|---:|---:|---:|
| v2 | 98,549 | 106,137 | +7,588 | +7.70% |
| v4-hinge | 99,613 | 107,201 | +7,588 | +7.62% |

The increase is entirely natural (vocabulary rows: typed 6,785→8,193, parent 32→513); no head
dimension was changed. The parameter-budget guard was raised to 120,000 for the corrected
configs (the historical 100k/105k guard would otherwise reject the legitimate row growth).

## 11. v2 seed0 results (Stage A)

| model | tokenizer | params | valid MAE | best epoch | Δ |
|---|---|---:|---:|---:|---:|
| v2 | historical (guard) | 98,549 | 0.184158 | 56 | — |
| v2 | corrected | 106,137 | 0.208384 | 27 | **−0.0242** |

The historical guard reproduces the canonical record **bit-identically**
(`0.18415821571176638`). The corrected seed-0 v2 run early-stopped at epoch 39 with best epoch
27 and reproduced exactly on a rerun (so it is a genuine trajectory, not noise).

## 12. v4 seed0 / multi-seed (Stage B)

**Table D — seed 0**

| model | tokenizer | params | valid MAE | best epoch | Δ |
|---|---|---:|---:|---:|---:|
| v4-hinge | historical (guard) | 99,613 | 0.170066 | 53 | — |
| v4-hinge | corrected | 107,201 | 0.178158 | 44 | **−0.0081** |

The historical guard reproduces `0.17006561887910357` **bit-identically**.

**Table E — 4 seeds (validation MAE; Δ = historical − corrected; + = corrected better)**

| seed | historical v4 | corrected v4 | Δ |
|---:|---:|---:|---:|
| 0 | 0.170066 | 0.178158 | −0.00809 |
| 1 | 0.163167 | 0.174228 | −0.01106 |
| 2 | 0.174149 | 0.168984 | +0.00517 |
| 3 | 0.170421 | 0.174880 | −0.00446 |
| **mean** | **0.169451** | **0.174062** | **−0.00461** |
| std | — | — | 0.00706 |
| median | — | — | −0.00628 |
| sign | — | — | 1/4 positive |

v2 (4 seeds): mean 0.182844 → 0.189548, **Δ −0.00670**, 2/4 positive, median −0.00409.

**Reproducibility incident (must be recorded).** The first corrected-v4 seed-3 scratch run
(`20260910-113625-e5d57ba6`) produced valid 0.168201 at epoch 60; a terminal companion run
(`20260910-122548-86896214`) diverged from it at epoch 55–60 and produced 0.174880. A scratch
rerun (`20260910-123241-6067c9ac`) reproduced the terminal value 0.174880 exactly. The
trajectories are bit-identical through epoch 54 and then diverge — the late-training numerical
instability already documented in `notes/reproducibility_cpu_determinism.md`. The reproducible
value (0.174880) is used in Table E; the 0.168201 trajectory is discarded as
non-canonical. (Seeds 0–2 corrected and the historical runs were stable across scratch/terminal
pairs.) This episode is itself a warning: single-run late-epoch numbers in this family carry a
small nondeterministic risk even under serial execution.

## 13. Alias subgroup analysis (Table F)

All 1,000 validation molecules are alias-affected (median alias burden 0.96), so the
pre-registered binary *unaffected vs alias-affected* split is **degenerate** (`n_unaffected=0`).
An audit-time (not outcome-time) refinement by alias-burden quartile is used instead.
Subgroup MAE is the mean over seeds of the per-seed subgroup MAE.

**Pre-registered groups (v4)**

| subgroup | n | historical MAE | corrected MAE | Δ |
|---|---:|---:|---:|---:|
| unaffected | 0 | — | — | — |
| alias-affected | 1,000 | 0.169451 | 0.174062 | −0.00461 |
| root-collision | 1,000 | 0.169451 | 0.174062 | −0.00461 |
| non-root collision | 992 | 0.168840 | 0.174135 | −0.00529 |
| high alias burden | 349 | 0.107651 | 0.113783 | −0.00613 |

**Alias-burden quartiles (v4)** (q1 = lowest burden, q4 = highest)

| subgroup | n | historical MAE | corrected MAE | Δ |
|---|---:|---:|---:|---:|
| q1 | 250 | 0.251195 | 0.253152 | −0.00196 |
| q2 | 250 | 0.136822 | 0.140372 | −0.00355 |
| q3 | 250 | 0.179288 | 0.187480 | −0.00819 |
| q4 | 250 | 0.110498 | 0.115246 | −0.00475 |

**Mechanism reading.** Every group regresses; the regression is *larger* where the historical
alias was heaviest (high-burden −0.0061; q3 −0.0082 vs q1 −0.0020). The data therefore support
the opposite of the pre-registered "alias-affected improves more" expectation: removing the
accidental sharing **hurts most on the most-shared tokens**, i.e. the coarse topology token was
providing a useful shared base. Note the burden/MAE confound (low-burden molecules are the
hardest), so this is a mechanism indication, not a clean causal partition.

## 14. Seed disagreement (Table G)

Mean cross-seed prediction std over seeds 0–3 (v4):

| subgroup | historical disagreement | corrected disagreement | Δ (hist−corr) |
|---|---:|---:|---:|
| all / alias-affected | 0.08086 | 0.08447 | −0.00361 |
| high alias burden | 0.06425 | 0.06999 | −0.00574 |
| q1 | 0.11829 | 0.11775 | +0.00054 |
| q2 | 0.07626 | 0.08176 | −0.00550 |
| q3 | 0.06408 | 0.06947 | −0.00540 |
| q4 | 0.06480 | 0.06889 | −0.00409 |

The corrected tokenizer does **not** reduce cross-seed disagreement; it increases it slightly,
and most on the high-alias-burden molecules. The hypothesis that conflicting supervision from
aliased tokens destabilises training is **not supported** here.

## 15. Rarity / OOV reinterpretation

Train-only r2 frequency statistics:

| statistic | historical | corrected |
|---|---:|---:|
| rare ≤1 | 2,658 | 6,598 |
| rare ≤2 | 3,652 | 8,905 |
| rare ≤5 (in-vocab) | 4,745 | 4,391 |
| rare ≤10 (in-vocab) | 5,429 | 12,794 |
| valid OOV occurrences | 278 | 1,163 |
| min frequency | 1 | 1 |
| mean log frequency | 1.649 | 1.479 |

Validation rarity statistics use **official train only** (unchanged rule). The rarity ↔ |residual|
relationship is essentially unchanged (Spearman rare-fraction 0.305 historical / 0.322 corrected;
OOV-fraction 0.349 / 0.372, seed-mean residuals). The corrected tokenizer makes the tail heavier
but does not remove or reshape the rarity-difficulty axis.

## 16. Impact on previous claims

| prior result | status | note |
|---|---|---|
| compositional patch sharing **NO-GO** | **requires re-check** | donor/recipient banks were built on the historical (aliased) tokens; the frozen oracle matched occurrence descriptors, so the NO-GO stands as a frozen-oracle result but must be re-measured on corrected tokens if reopened |
| "exact typed patch token" wording | **invalidated** | historical notes with that wording get the addendum below; historical numbers remain valid as the historical baseline |
| compact-v2 / v4-hinge performance | **unaffected** (as historical baselines) | numbers are correct for the historical tokenizer; reused bit-identically here |
| v4-hinge global topology GO, benchmark-comparable 4-seed | **unaffected** | topology channel is certificate-free |
| protocol correction + 4-seed confirmation | **unaffected** | — |
| compact-v3 ring/context NO-GO | **unaffected** | context channel uses cycle enumeration, not certificates |
| ZINC long-cycle target audit | **unaffected** | cycle/spectral features |
| Information Gap Audit | **narrowed / requires re-check** | any token-based coverage claim changes with the tokenizer |
| post-v4 residual audit | **requires re-check** | rarity/token-frequency axis changes |
| OOF difficulty audit | **requires re-check** | same rarity/token axis |
| compact-v5 multi-quantile NO-GO | **unaffected** (objective-only) but re-check gated | residual definition changes with tokenization |
| seed-disagreement / instability narratives | **unaffected** | the alias does not explain them (correction does not reduce disagreement) |

**Addendum for historical notes (not a batch rewrite):**
> Historical tokenizer used a pynauty-certificate-derived identity later found not to be a
> complete key for the intended typed/color-preserving isomorphism relation; it was in fact an
> uncolored rooted-topology token. The `typed_tokenizer_v2_corrected` key is the verified typed
> canonical key.

## 17. Final decision

**Case C — CORRECTNESS FIX REVEALS ACCIDENTAL SHARING BENEFIT.**

* Primary (v4): mean valid degradation **−0.00461 ≥ 0.003**; 1/4 seeds positive; the one-time
  frozen benchmark test agrees (**0.136885 → 0.140895**, Δ −0.0040).
* v2: mean degradation −0.00670.
* Not a truncation artefact: an uncapped corrected diagnostic (vocab 15,219, 135,305 params)
  is **worse still** (valid 0.1993).

Per the pre-registered rules the corrected tokenizer is **not** reverted: it is the
semantically correct method and becomes the default for future exact-token work, while the
historical tokenizer is retained as the historical benchmark baseline. The result is read as
evidence that the historical collision supplied a *principled-looking shared structural base*
(shared topology) while the 146D descriptor supplied the chemistry. The pre-registered next
research direction is therefore **a principled shared base + exact identity residual**, which
is **not implemented in this stage**.

**Token versioning & cache invalidation.** `representation.typed_tokenizer_version` is recorded
in every resolved config, run result (`representation.typed_tokenizer_version` /
`typed_tokenizer_fingerprint`) and extraction metadata. `typed_patch_tokenizer.py` exposes
`typed_tokenizer_fingerprint(version, radius)` and `assert_cache_tokenizer_version(metadata,
expected)`; caches without a version tag are treated as historical v1 and **rejected** by any
corrected run. The historical tokenizer is unchanged and still reproduces all old records
(Tests 8/`test_8_historical_tokenizer_reproduces_legacy_certificate`).

---

## tokenizer_dependency_map (active certificate paths)

| path | symbol | uses certificate? | audited | action |
|---|---|---|---|---|
| radius-2 patch token | `zinc_patch_path_pooling._typed_certificate(radius=2)` via `_graph_record` | **yes** | yes — heavily aliased (Table A) | version-dispatched to corrected key |
| radius-1 parent token | `zinc_patch_path_pooling._typed_certificate(radius=1)` | **yes** | yes — most aliased (§8) | version-dispatched to corrected key |
| global/topology channel (v4) | `zinc_topology_features` (cycle spectrum / MCB / hinge) | **no** | confirmed certificate-free | unchanged |
| structural / ring context (v3) | `structural_context` (cycle enumeration) | **no** | confirmed certificate-free | unchanged (historical v3 not reopened) |
| relation / path code | `_pair_relation`, `_shortest_path_summary` | **no** (uses patch node sets + distances) | confirmed | unchanged |
| legacy `zinc_exact_patch_relation._canonical_typed_patch` | separate experiment (not on the compact path) | yes (same certificate) | recorded, **not active** | not changed here (out of scope) |
| `zinc_ksvd_patch_path_pooling`, `molhiv_patch_path_pooling` | separate tracks/experiments | yes | recorded | not changed here |

## Tables / figures

* Tables A §7, B §6, C §9, D §11–12, E §12, F §13, G §14.
* Figures (`results/typed_patch_tokenizer_correctness/figures/` and `model_analysis/`):
  `fig1_split_multiplicity.png`, `fig2_frequency_distribution.png`,
  `fig3_alias_burden_vs_error.png`, `fig4_subgroup_mae.png`, `fig5_disagreement.png`,
  `fig6_rarity_residual.png`.

---

## Q1–Q16

**Q1. Why was `pynauty.certificate` insufficient?**
It returns only the canonical adjacency matrix; the vertex coloring is used to constrain the
canonical labelling but is not part of the bytes. It is therefore an invariant of the
*uncolored* incidence topology, dropping atom type, bond type, root designation and
root-distance class.

**Q2. Which semantic colors must isomorphism preserve?**
Atom vertices `("node", is_root, distance_from_root, atom_type)`; incidence vertices
`("edge", bond_type)`. These are the only active semantic colors on the compact path.

**Q3. How is the corrected key constructed?**
`b"typedpatchkey/v2;" + pynauty.certificate(incidence) + b"|" + repr(canonical semantic color
sequence)` with color cells ordered by `sorted(..., key=repr)`.

**Q4. Does it pass all permutation-invariance tests?**
Yes — 16,000/16,000 for r2 and 16,000/16,000 for r1, 0 failures.

**Q5. Any same-key-but-non-isomorphic collision?**
**NO** — 489k+ oracle checks over corrected-key buckets (train+valid, r2+r1) and the random
small-graph soundness test found 0.

**Q6. True alias fraction of the historical r2 vocabulary?**
2,363 / 6,784 tokens (34.8%); 94.4% of occurrences; 730 root-only collisions and 1,633
non-root typed collisions; max 104 classes per token.

**Q7. Is the parent vocabulary also affected?**
Yes, worse: 23/31 tokens aliased, 99.99% of occurrences, max 98 classes.

**Q8. Vocabulary growth?**
r2: 6,784 → 15,218 unique types (2.24×), capped at 8,192; parent: 31 → 512.

**Q9. How do rare/OOV statistics change?**
Heavier tail (rare ≤1 2,658→6,598); validation OOV 278→1,163 occurrences (266→1,087 types);
mean log frequency 1.65→1.48.

**Q10. Parameter increase?**
+7,588 params (+7.6–7.7%): v2 98,549→106,137; v4 99,613→107,201.

**Q11. Corrected v2 validation?**
seed0 0.184158 → 0.208384 (Δ −0.0242); 4-seed mean Δ −0.0067.

**Q12. Corrected v4 validation / multi-seed?**
seed0 0.170066 → 0.178158; 4-seed mean 0.169451 → 0.174062 (Δ −0.0046, 1/4 positive);
benchmark test 0.136885 → 0.140895.

**Q13. Do alias-affected molecules benefit more?**
No. Every molecule is alias-affected; the regression is *largest* on the highest alias-burden
molecules (Δ −0.0061) — the opposite direction, consistent with accidental-sharing benefit.

**Q14. Does correction reduce cross-seed disagreement?**
No — it slightly increases it (Δ −0.0036 overall, −0.0057 high-burden).

**Q15. Impact on historical scientific claims?**
Invalidated: the "exact typed token" wording. Requires re-check: compositional-sharing NO-GO,
post-v4 residual audit, OOF difficulty audit, information-gap token coverage, compact-v5
residual definition. Unaffected: compact-v2/v4 performance as historical baselines, v4 topology
GO, protocol 4-seed confirmation, v3 ring NO-GO, long-cycle audit.

**Q16. Single recommended next stage?**
Register **Corrected-tokenizer residual / rarity re-audit** on the frozen corrected-v4
representation (the previous bottleneck ranking may change), and treat the accidental-sharing
finding as a *hypothesis* for a future principled *shared base + exact identity residual* — do
not implement it in this stage. The corrected tokenizer is the default exact-token method;
the historical tokenizer stays as the benchmark baseline.
