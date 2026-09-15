# Identity incremental-information audit (ZINC + OGBG-MolHIV)

**Branch:** `audit/identity-incremental-information`
**Protocol:** `identity_incremental_information_audit_v1`
**Module:** `tracks/ksvd/experiments/luyin16/identity_incremental_information_audit.py`
**Tests:** `tracks/ksvd/tests/test_identity_incremental_information_audit.py` (10 pass)
**Results:** `tracks/ksvd/results/identity_incremental_information/`

**Test policy:** ZINC official test is *never* loaded. The MolHIV official test
was already opened once for the frozen recurrent checkpoint; here it is only
re-read read-only from the existing `molhiv_parameter_attribution` artifacts. No
backbone was trained. No architecture selection was performed.

**Core question.** Once a model already receives every deterministic *non-ID*
structural descriptor of a rooted patch, what does the discrete typed identity
token still contribute — real structural information, categorical
distinguishability, task-specific memory, or a mixture?

---

## 0. Headline

> Identity is a **genuine per-patch discrete separator** that the deterministic
> descriptor does not supply (Case C at the per-patch level), but it is
> **redundant for molecule-level distinguishability** (Case A) and its **learned
> geometry is not organised by structural similarity** (Case B). It is best read
> as *distinguishability + task-specific arbitrary categorical memory*, **not**
> as missing raw structural information, and there is no evidence for unique
> task semantics (Case D unsupported).

Final labels: `A_mostly_redundant_at_molecule_level`,
`B_distinguishability_matters`, `C_per_patch_descriptor_ambiguity`.

A second, previously-unrecorded finding: **the MolHIV "exact rooted typed
certificate" is the same coarse uncoloured rooted-topology invariant as the ZINC
*historical* token** — `molhiv_patch_path_pooling._canonical_typed_patch` returns
`bytes(pynauty.certificate(vertex-coloured incidence))` and discards the semantic
colour sequence. Its 26,232-row vocabulary is therefore *not* a complete coloured
key; the corrected coloured key has 87,039 distinct classes (3.3x). All
descriptor/identity statements below are reported for both.

---

## 1. Real data flow

### 1.1 ZINC (`zinc_compact_v4_topology_hinge.yaml`, historical path)

| item | value |
|---|---|
| certificate semantics (historical) | `bytes(pynauty.certificate(coloured incidence graph))`; the returned canonical adjacency does **not** carry the vertex-colour sequence, so atom type / bond type / root designation / root-distance class are invisible. |
| certificate semantics (corrected) | certificate **+ canonical semantic colour sequence** — a complete invariant of the coloured incidence graph. |
| patch_cont dim | 146 (radius-2 shell descriptor) |
| typed vocabulary | 6,784 known + id 0 OOV = 6,785 rows |
| parent vocabulary | 31 known + OOV = 32 rows |
| token width | typed 16, parent 8 |
| embedding policy | hybrid: 768 frequent typed rows full 16D; 6,017 rare rows rank-4 factorised + 16x4 projection; 32 parent rows full 8D |
| typed embedding params | 12,288 full + 24,132 factorised = **36,420** |
| parent embedding params | 256 |
| total model params | **99,613** (typed share 36.6%) |
| OOV policy | id 0 learned OOV row; known tokens start at 1 |

Corrected vocabulary size (for reference, not used by the model): uncapped
15,218; capped-8192 8,192. `patch_cont` (146D) is the X_nonID feature; it is the
descriptor used throughout this audit.

### 1.2 OGBG-MolHIV (`molhiv_recurrent_pair_centre`, frozen checkpoint)

| item | value |
|---|---|
| certificate semantics (used) | `molhiv_patch_path_pooling._canonical_typed_patch` = `bytes(pynauty.certificate(vertex-coloured incidence graph))` — **same coarse defect as ZINC historical**: colour sequence discarded. |
| certificate semantics (corrected) | certificate + canonical colour sequence, computed by this audit. |
| patch descriptor dim | 793 (`patch_cont`); relation dim 42 |
| typed vocabulary | **26,232** known + OOV = 26,233 rows |
| parent vocabulary | 102 known + OOV = 103 rows |
| token width | typed 32, parent 16 |
| embedding policy | dense `nn.Embedding` (no hybrid / no factorisation) |
| typed embedding params | **839,456** |
| parent embedding params | 1,648 |
| total frozen params | **1,076,589** (identity share **78.0%**) |
| OOV policy | id 0 learned OOV row; known tokens start at 1 |
| valid OOV occurrence fraction | 7.37% (historical); 20.8% (corrected) |
| valid molecules with >=1 OOV | 59.2% |

**MolHIV extraction (this audit, light re-extraction):** train 32,901 molecules
/ 830,936 patches in 122 s; valid 4,113 / 114,300 in 18 s. Historical unique
fingerprints 26,232 — exactly the model vocabulary size; **corrected unique
87,039** (~3.3x fragmentation).

---

## 2. Descriptor -> identity uniqueness

`X_nonID` = exact-byte patch descriptor signature (no rounding). For each
signature, count the distinct typed identities that occur.

### ZINC (train, 231,664 occurrences)

| metric | historical coarse token | corrected exact key |
|---|---|---|
| unique X_nonID signatures | 19,701 | 19,701 |
| distinct identities | 6,784 | 15,218 |
| collision mass | 106,401 (45.9%) | 106,401 (45.9%) |
| mean identity multiplicity | 1.264 | 1.264 |
| max multiplicity | 9 | 9 |
| H(identity \| descriptor) | **0.271 nats** | **0.271 nats** |
| signatures with multiplicity 1 | 16,095 | 16,095 |

Identical for historical and corrected because the descriptor is coarser than
both keys — the collision is descriptor-side, not identity-side. Valid split
mirrors this (H = 0.204, collision 32.3%, max 5).

### MolHIV (train, 830,936 occurrences)

| metric | historical certificate | corrected coloured key |
|---|---|---|
| unique X_nonID signatures | 79,844 | 79,844 |
| distinct identities | 26,232 | 87,039 |
| collision mass | 194,922 (23.5%) | 216,760 (26.1%) |
| max multiplicity | 11 | 11 |
| H(identity \| descriptor) | **0.114 nats** | **0.134 nats** |

**Reading.** The descriptor does not determine identity. A large minority of
patch occurrences share a deterministic descriptor while carrying different
identities. So identity is *not* a redundant re-encoding of the non-ID
descriptor — it separates cases the descriptor merges. But the merged cases are
mostly *within* the descriptor's own resolution limit, so this is per-patch
ambiguity, not molecule-level missing information (see Section 7).

---

## 3. Identity -> descriptor variability

Reverse direction.

### ZINC

| metric | historical coarse token | corrected exact key |
|---|---|---|
| identities | 6,784 | 15,218 |
| mean unique descriptors per identity | 3.670 | 1.636 |
| max unique descriptors per identity | 198 | 9 |
| fraction with a single descriptor | 0.520 | 0.642 |
| within-identity mean dim variance | 3.0e-4 | 1.8e-6 |
| within-identity mean max spread | 0.155 | 0.025 |

Historical radius-2 -> corrected split: **34.8%** of historical tokens split
(mean 2.24 corrected classes, max 104). The radius-1 *parent* token splits far
more (74.2%, mean 16.5, max 98), consistent with it acting as a coarse
parameter-sharing bucket.

### MolHIV

| metric | historical certificate | corrected coloured key |
|---|---|---|
| identities | 26,232 | 87,039 |
| mean unique descriptors per identity | 3.278 | **1.000** |
| max unique descriptors per identity | 987 | **1** |
| fraction with a single descriptor | 0.637 | **1.000** |
| within-identity mean dim variance | 1.9e-4 | 1.3e-14 |
| historical -> corrected split | 36.4% split (mean 3.32, max 1,026) | — |

**Nuance (ZINC only).** The corrected MolHIV descriptor is an exact function of
the corrected key (variance ~0). ZINC is *almost* the same (mean 1.64, max 9) but
not exact. The residual is explained by a **context leak in one descriptor
coordinate**: `_shell_descriptor` computes its final scalar
(`degrees.mean()/4`, coordinate 145) from the **full-molecule** degree
(`graph.neighbors(node)`), while every other coordinate uses the induced patch
subgraph. Full-molecule degree of a patch node depends on context outside the
patch, so a fixed rooted typed patch can still produce more than one descriptor.
This is a descriptor-design fact, not an identity fact.

---

## 4. kNN identity purity (label-free, k = 1, 8, no sweep)

Train-fit standardized descriptor space, exact brute-force neighbours, query
itself excluded, 20,000-query sample (seed 20260927).

### ZINC (reference 231,664)

| identity | top-1 | k=8 | random top-1 |
|---|---|---|---|
| historical coarse token | 0.7996 | 0.7664 | 0.0167 |
| corrected exact key | 0.7988 | 0.7639 | 0.0057 |

Strata (historical, k=8): frequent (>20) 0.809 (n=18,147), medium (6–20) 0.501
(n=1,114), rare (2–5) 0.177 (n=513), singleton 0.000 (n=226).

### MolHIV (reference 830,936)

| identity | top-1 | k=8 |
|---|---|---|
| historical certificate | 0.893 | 0.846 |
| corrected coloured key | 0.869 | 0.801 |

**Reading.** For frequent tokens identity is largely (not entirely) predictable
from the descriptor neighbourhood; for rare/singleton tokens it is essentially
orthogonal. So identity carries real information *precisely where the descriptor
is data-poor*, and mostly duplicates the descriptor where data is plentiful.
This is the signature of categorical memory doing work on the tail.

---

## 5. Learned embedding geometry vs structural geometry (frozen checkpoints)

No retraining. Structural geometry = token-mean standardized descriptor.
Memory-safe: top-k neighbours computed over row blocks; Mantel-style rank
correlation on 200,000 sampled off-diagonal pairs.

### ZINC (`Pstar_A2_long_seed0_selection_state.pt`, 6,784 rows, 16D)

| metric | value |
|---|---|
| participation ratio (all) | 3.08 |
| participation ratio (frequency-weighted) | 9.82 |
| participation ratio (frequent only) | 13.98 |
| stable rank (all) | 1.99 |
| norm vs log-frequency Spearman | +0.365 |
| NN overlap@10 vs structural (all) | 0.46% (random ~0.15%) |
| sampled-pair Spearman vs structural (all) | 0.004 |

### MolHIV (`raw_state_seed0.pt`, 26,232 rows, 32D)

| metric | value |
|---|---|
| participation ratio (all) | 18.99 |
| participation ratio (frequency-weighted) | 8.31 |
| participation ratio (frequent only) | 21.45 |
| stable rank (all) | 6.27 |
| norm vs log-frequency Spearman | +0.391 |
| NN overlap@10 vs structural (all) | 0.087% (random ~0.038%) |
| sampled-pair Spearman vs structural (all) | 0.021 |

**Reading.** In both models the identity matrix is *not* organised by structural
similarity (near-zero NN overlap and rank correlation). MolHIV uses much more
of its 32 dimensions (PR ~19) than ZINC uses of its 16 (PR ~3) — MolHIV's wider
identity block is genuinely higher-rank — yet even there the axes are not
structural. Norm grows with frequency, i.e. frequent categorical memory is
simply better fit, not more structural. The learned semantics are **task-specific
arbitrary categorical coordinates**, not an emergent structural embedding.

---

## 6. Frequency / OOV / embedding parameter share

| metric | ZINC historical | ZINC corrected (uncapped) | MolHIV historical | MolHIV corrected |
|---|---|---|---|---|
| vocabulary types | 6,784 | 15,218 | 26,232 | 87,039 |
| occurrences | 231,664 | 231,664 | 830,936 | 830,936 |
| singleton type fraction | 0.392 | 0.434 | 0.393 | 0.469 |
| types with freq <= 5 | 4,745 (69.9%) | 11,417 (75.0%) | 19,547 (74.5%) | 71,540 (82.2%) |
| top-100 occurrence coverage | 0.664 | 0.429 | 0.603 | 0.279 |
| valid OOV occurrence frac | 0.0120 | 0.0292 | 0.0737 | 0.208 |
| embedding param fraction | 36.6% | (99,613 -> 107,201) | **78.0%** | — |

Parent (radius-1) vocabularies: ZINC 31 types (historical) / 512 (corrected),
valid OOV 0 / 0.07%; MolHIV 102 types, parent params 1,648.

**Reading.** The identity vocabulary is long-tailed in every setting and the
storage cost tracks it (MolHIV identity alone is 78% of the model). OOV is not
rare: 59% of MolHIV valid molecules contain at least one OOV patch. This is the
parameter-allocation pressure that motivates Agent A/B, and it is why a fixed
deterministic code is attractive.

---

## 7. with-ID vs without-ID expressivity (official ZINC train, target-lock-then-read)

Protocol: lock the four molecule signatures first (SHA
`dba5e3e6c90d45c18a11ffd136beb39897e69735df9e38bd6e5580157da077e9`), then read
official-train targets only.

| signature system | unique | collision classes | collision mass | non-raw-isomorphic classes |
|---|---|---|---|---|
| descriptor_only | 9,997 | 3 | 6 (0.06%) | 0 |
| descriptor + identity | 9,997 | 3 | 6 | 0 |
| full system with identity | 9,997 | 3 | 6 | 0 |
| full system without identity | 9,997 | 3 | 6 | 0 |

All four systems are **identical**: identity adds **zero** molecule-level
distinguishing power once the deterministic descriptor system is present. Target
empirical lower bound MAE = 0.0 for all four.

**Reading.** Section 2's per-patch ambiguity does **not** aggregate into any
molecule-level collision. Identity is redundant at the molecule level (Case A),
even though it is a genuine per-patch separator (Case C). The ambiguity identity
resolves is real but *not necessary* for distinguishing molecules.

---

## 8. MolHIV generalization diagnostics (existing predictions only; descriptive)

Re-read frozen valid logits/targets from `molhiv_recurrent_pair_centre`
(seeds 0/1, raw + soup); test bucket AUCs re-read read-only from
`molhiv_parameter_attribution/rarity_prediction_diagnostic.json`.

Overall valid AUC: seed0 raw 0.814 / soup 0.810; seed1 raw 0.850 / soup 0.851.
Test overall 0.780.

| subgroup (valid) | n | prevalence | mean error | raw AUC | soup AUC |
|---|---|---|---|---|---|
| zero OOV | 1,679 | 0.0232 | 0.0288 | 0.917 | 0.920 |
| any OOV | 2,434 | 0.0173 | 0.0245 | 0.744 | 0.734 |

Controls:

* **Size.** zero-OOV mean size 29.8 vs any-OOV 26.4. Size-decile-stratified AUC:
  zero 0.90–0.96 vs any 0.72–0.76 — **the gap persists**.
* **Molecule-key frequency proxy** (document frequency of the sorted typed-token
  multiset; no RDKit available). Stratified AUC essentially unchanged
  (zero 0.917 / any 0.744) — **not explained** by the frequency proxy.
* **Prevalence.** The two subgroups have different prevalence (2.32% vs 1.73%),
  and the *mean-error* difference (0.0288 vs 0.0245) is in the opposite
  direction from the AUC gap, i.e. the mean-error comparison is
  prevalence-confounded. Partial Spearman(OOV fraction, error | size)
  = -0.32 to -0.37, and adding the key-frequency control barely changes it
  (-0.32 to -0.37) — also prevalence-driven.
* **Subgroup n.** 39 vs 42 positives; adequate for a rough AUC, not for a precise
  one.

Prior phenomenon check: **replicates**. Test zero-OOV AUC 0.837 vs OOV-fraction
(0.1,0.3] 0.654; by min-train-frequency, >5 0.894 vs (1,5] 0.670 vs 0 0.691.
Valid shows the same direction and a larger gap (0.917 vs 0.744).

**Reading.** The OOV-difficulty signal is real and survives size and
molecule-frequency stratification, so it is *not* merely a size or scaffold
artifact. But (a) the mean-error comparison is prevalence-confounded, (b) the
remaining "difficulty" is exactly what OOV means (novel chemistry), (c) scaffold
frequency is approximated, not measured, and (d) n is small. This is a
**descriptive difficulty statement, not a causal claim** and not an architecture
decision.

---

## 9. Historical experiment reconciliation

| experiment | changed about identity | retrained | lookup kept | tested axis | conclusion | constraint on future routes |
|---|---|---|---|---|---|---|
| typed tokenizer correctness repair | token = coarse uncoloured topology -> complete coloured key | yes | yes | information correctness + statistical sharing | corrected token is correct but slightly worse; the coarse token acted as accidental parameter sharing | do not claim the coarse token is "exact identity"; correctness alone is not a gain |
| corrected fragmentation / rarity audit | none (measures the corrected distribution) | no | yes | sharing / statistical | degradation is baseline-difficulty redistribution, not directed by fragmentation | fragmentation is not the lever; stop chasing token merging |
| compositional patch sharing oracle | frozen embedding transplant for rare/OOV | no | replaced for rare/OOV | sharing | NO-GO: structural KNN donors not better than frequent-mean/random | structural similarity does not transfer identity semantics (matches Section 5) |
| compact-v6 topology-attribute factorization | exact token -> shared (type, role) attribute encoder | yes | no | information / function class | NO-GO: branch collapses, does not replicate | additive attribute factorisation cannot replace the exact key |
| SBCI shared-basis compositional interaction | identity kept; free relation/centre family replaced | yes | yes | function class / sample efficiency | NO-GO: sample-efficiency frontier crossing | compositional relation bases do not buy sample efficiency here |
| current recurrent Cell A | identity lookup retained (~36.6% params) | yes | yes | capacity / architecture | parameters dominated by identity storage; width gains do not transfer | capacity and identity storage are entangled; must be unconfounded (Agent A) |
| MolHIV parameter attribution | none (accounting only) | no | yes | parameter accounting | identity storage = 78.1% of the ~1.08M model; no cutoff chosen | large-vocabulary identity cost is the real MolHIV pressure (Agent B) |

**Logical thread.** The repair established that the coarse token was an
accidental sharing device; the fragmentation audit showed sharing is not the
lever; the sharing oracle and v6/SBCI showed that *structural* replacements fail
to recover whatever the lookup provides. Sections 5 and 7 of this audit explain
why: the lookup is not structural (geometry) and not molecule-level-necessary
(expressivity), so it is storing *distinguishable task-specific categorical
memory*. That is exactly what Agent A (capacity-controlled A1 fixed code vs A2
no identity) and Agent B (shared connectivity encoder) are designed to arbitrate.

---

## 10. Final decision map (Case A/B/C/D)

### Case A — Mostly redundant: **SUPPORTED at molecule level**
Descriptor-only molecule signatures already achieve 0 non-raw-isomorphic
collision classes and identical collision mass to the full system (Section 7).
Identity adds nothing to molecule-level distinguishability.

### Case B — Distinguishability matters, learned semantics unclear: **SUPPORTED**
Per-patch collision/entropy (Section 2) and kNN purity (Section 4) show identity
provides genuine discrete separation, strongest on the rare tail. The learned
geometry is not organised by structure (Section 5): NN overlap ~0.5%/0.09%,
sampled-pair Spearman ~0.004/0.021. This is the "fixed/shared-code worth testing"
case.

### Case C — Genuine missing structural information: **SUPPORTED at per-patch level only**
Identity resolves substantial *per-patch* descriptor ambiguity (45.9% ZINC /
23.5–26.1% MolHIV collision mass; H = 0.271 / 0.114 nats). But the ambiguity does
not aggregate to molecule-level collisions (Section 7), so a replacement need not
reconstruct it for discriminability.

### Case D — Learned identity semantics carry unique task-relevant value: **UNSUPPORTED**
No existing experiment isolates task-relevant semantics of the learned rows from
the structural information already present. The compositional-sharing transplant
was a NO-GO and the embedding geometry is not structural. Proposing Case D would
require a *new* evidence source, not lookup performance.

**Answer to the core question:** identity is a **mixture dominated by
distinguishability + task-specific memory**. It supplies categorical separation
(real, tail-heavy) and task-specific arbitrary memory (high-rank, non-structural,
frequency-scaled norm), while being redundant for molecule-level structural
information. It is *not* primarily "missing raw structural information."

---

## 11. Pre-registered guidance for Agent A and Agent B

This audit was run before Agents A/B finished; the interpretation below is fixed
in advance so their results cannot be re-narrated post hoc.

### Agent A (`exp/identity-capacity-control`: A0 learned lookup / A1 fixed code + shared capacity / A2 no identity + shared capacity)

Meaningful threshold 0.002 on the 2-seed Top-5 soup valid mean; reference
A0 = 0.126368. Anchor predictions from this audit:

* **Case I (`A0 ~= A1 << A2`)** — "distinguishability matters, per-ID memory not
  necessary." This is the *expected* outcome given Section 2 (identity separates
  descriptor collisions) and Section 5 (no structural semantics). If observed,
  it is a strong result: replace the lookup with a deterministic identity code +
  shared adapter.
* **Case II (`A0 << A1 ~= A2`)** — "the gain is task-specific learned categorical
  memory." This would revise the headline toward memory-dominance: the value is
  not merely knowing tokens differ. Requires the memory to not be recoverable by
  a shared network.
* **Case III (`A0 ~= A1 ~= A2`)** — "typed lookup largely dispensable." This would
  *strengthen* Case A/B (redundant at the level that matters) and weaken the
  memory claim; do not then claim identity is necessary.
* **Case IV (`A0 < A1 < A2`)** — mixed; require a quantitative decomposition
  before adopting any label.

Audit-derived expectations: A1 should land closest to A0 if distinguishability
is the operative ingredient (Sections 2/4 support this); A2 should degrade most
on the *rare/OOV* tail (Section 4: singleton purity 0.0; Section 6: long tail),
so report A2 error by token-frequency stratum as a secondary check. A1/A2 must
be reported with exact parameter parity (85,763 +/- 1%) or the comparison is
void. Do **not** open ZINC test.

### Agent B (`exp/shared-structural-patch-encoder`: remove vocab-sized typed lookup, shared connectivity-aware encoder)

Decision logic as pre-registered by the task: strong success if improvement
>= 0.002 on 2-seed soup and same sign; neutral success if |delta| < 0.002 with no
vocab-sized lookup and vocabulary-independent parameters; failure if regression
> 0.003 on both seeds.

Audit-derived guidance:

* Section 5 predicts the learned rows are **not** structural, so a shared
  connectivity encoder is *not* trying to reproduce an existing structural
  embedding — it is trying to *replace* arbitrary memory with a principled
  function. A neutral result is therefore already meaningful (principled,
  OOV-native, vocabulary-independent), exactly as the task states.
* Section 7 predicts no molecule-level collision is lost, so any large
  regression points to **optimisation / inductive bias**, not to missing
  structural information.
* Section 8 predicts difficulty concentrates on OOV/rare patches; a shared
  encoder that is OOV-native should help *there* first. Report per-OOV-stratum
  valid error, not just the mean.
* Section 6 (MolHIV identity = 78% of parameters, 59% of molecules with OOV) is
  the strongest motivation for Agent B; the ZINC-only run understates the
  benefit. But do not train MolHIV in Agent B (task stop rule).
* If Agent B fails on both seeds, the correct conclusion is "this categorical
  lookup supplies an inductive bias this encoder cannot recover," **not** "the
  lookup carries unique task semantics" (Case D stays unsupported).

### Joint reading

* A observing Case I **and** B neutral => strong support for
  **distinguishability + generic shared capacity**, with per-ID memory
  dispensable. Headline unchanged.
* A observing Case II **and** B failing => memory-dominance; Case B strengthens
  toward "task-specific memory," Case D still not established.
* A observing Case III **and** B succeeding => identity largely dispensable;
  Case A strengthens and the whole lookup line can be retired.

---

## 12. Artifacts, reproducibility, no-go compliance

Artifacts in `tracks/ksvd/results/identity_incremental_information/`:
`dataflow.json`, `zinc_descriptor_identity.json`, `zinc_knn_purity.json`,
`zinc_frequency.json`, `zinc_expressivity.json`,
`zinc_expressivity_signature_lock.json`, `zinc_embedding_geometry.json`,
`molhiv_descriptor_identity.json`, `molhiv_embedding_geometry.json`,
`molhiv_generalization.json`, `final_decision.json`, `answers_q1_q20.json`,
`figures/fig1_identity_multiplicity.png`,
`figures/fig2_knn_purity.png`, `figures/fig3_embedding_geometry.png`, plus the
MolHIV per-split descriptor/identity arrays.

Reuse (verified from code, not copied from notes):
`results/post_v4_residual_audit/cache/v4_records_{train,valid}.pkl.gz` and
`results/typed_patch_tokenizer_correctness/occurrences.npz` (exact alignment:
231,664 patches, 100% historical-certificate match);
`results/corrected_token_fragmentation_audit`;
`results/raw_graph_patch_system_sufficiency`;
`results/molhiv_parameter_attribution`.

Memory safety: descriptor operations are chunked (no full float64 copies of the
2.6 GB MolHIV matrix), top-k neighbour computation never materialises an
`n x n` cosine matrix, and the MolHIV vocabulary row map is cached to JSON. The
first `molhiv-embedding` attempt OOM-killed on a full-matrix copy; the committed
code path is chunked and succeeded.

No-go compliance: no backbone trained; identity not deleted; no new embedding
designed; no top-K cutoff chosen; no hash compression; no compositional encoder;
no optimizer sweep; no architecture selection; ZINC official test never opened;
MolHIV test only re-read read-only from existing artifacts.