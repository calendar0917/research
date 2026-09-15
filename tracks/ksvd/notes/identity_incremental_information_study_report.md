# Identity incremental-information study — consolidated report

**Date:** 2026-09-15
**Study:** `zinc-context-gap` — "once a model already has every deterministic
non-ID structural descriptor of a rooted patch, what does the discrete typed
identity token still contribute?"
**Consolidates three lines, all now closed:**

| line | branch | verdict |
|---|---|---|
| training-free audit (ZINC + MolHIV) | `audit/identity-incremental-information` | Case A + B + C; D unsupported |
| identity / capacity control | `exp/identity-capacity-control` | **Case III — typed lookup dispensable** |
| shared structural patch encoder (vocab-free) | `exp/shared-structural-patch-encoder` | **STRONG_SUCCESS — vocab-free replacement works** |

**Official test:** ZINC official test was never loaded by any of the three
lines. The MolHIV official test was already opened once for the frozen recurrent
checkpoint; the audit only re-read existing artifacts read-only. No ZINC test
read is authorized by this report.

**Headline (post-resolution).**

> The discrete typed identity token supplies real **per-patch** categorical
> separation that the deterministic descriptor does not, but it carries **no
> molecule-level structural information**, its learned geometry is **not
> structural** (task-specific categorical memory), and on the frozen ZINC cell A
> it is **dispensable**: replacing the 36,420-param learned lookup with either
> token-shared capacity (A1/A2 match or beat A0) or a **vocab-free shared
> connectivity encoder** (soup 0.118972 vs 0.126368) loses nothing and slightly
> helps. Identity is best read as **replaceable task-specific categorical
> memory**, not as missing raw structure, necessary distinguishability, or
> unique task semantics.

This is a strengthening of the audit's own pre-resolution headline
("distinguishability + task-specific memory"): Agent A showed that even the
*distinguishability* component is not required at this scale, and Agent B showed
the memory itself is replaceable by a small shared function.

---

## 1. ZINC / MolHIV identity data flow

### ZINC (compact-v4 / cell A family)

| item | value |
|---|---|
| historical certificate | `bytes(pynauty.certificate(coloured incidence graph))`; the returned bytes do **not** carry the semantic colour sequence → rooted **uncoloured** topology (atom/bond type, root flag, root-distance class invisible) |
| corrected key | certificate + canonical semantic colour sequence = complete coloured-incidence invariant |
| patch descriptor | 146-D `patch_cont` (radius-2 shell) |
| typed vocabulary | 6,784 historical (+OOV 6,785); corrected 15,218 uncapped / 8,192 capped |
| parent vocabulary | 31 historical / 512 corrected |
| token / parent width | 16 / 8 |
| embedding policy | hybrid: 768 frequent rows full 16-D, 6,017 rare rows rank-4 factorised + projection |
| typed embedding params | 36,420 (cell A total 85,763; identity share 42.4%) |
| OOV | id 0 learned row; known tokens start at 1 |

### OGBG-MolHIV (recurrent H96 port)

| item | value |
|---|---|
| certificate semantics | `_canonical_typed_patch` returns `bytes(pynauty.certificate(...))` — **the same coarse defect as the ZINC historical token**; colour sequence discarded |
| corrected key | certificate + colour sequence; **87,039** distinct classes (3.3× the vocabulary) |
| patch / relation descriptor | 793-D / 42-D |
| typed vocabulary | 26,232 (+OOV 26,233) |
| parent vocabulary | 102 (+OOV 103) |
| token / parent width | 32 / 16 |
| embedding policy | dense `nn.Embedding` |
| typed embedding params | **839,456** (78.0% of 1,076,589 total) |
| OOV | valid occurrence 7.37% historical / 20.8% corrected; 59.2% of valid molecules contain ≥1 OOV |

**Key correction to the record:** MolHIV's "exact rooted typed certificate" is
**not** exact — it is the same coarse pynauty certificate as the ZINC
*historical* token. All audit statements are reported for both historical and
corrected keys.

---

## 2. Descriptor → identity collision (exact-byte signatures)

| dataset | key | unique sigs | collision mass | H(identity \| descriptor) | max mult |
|---|---|---|---|---|---|
| ZINC train (231,664) | historical | 19,701 | 45.9% | 0.271 nats | 9 |
| ZINC train | corrected | 19,701 | 45.9% | 0.271 nats | 9 |
| MolHIV train (830,936) | historical | 79,844 | 23.5% | 0.114 nats | 11 |
| MolHIV train | corrected | 79,844 | 26.1% | 0.134 nats | 11 |

The deterministic descriptor does **not** determine identity; identity
separates patch occurrences the descriptor merges. (ZINC historical = corrected
because the descriptor is coarser than both keys.)

## 3. Identity → descriptor variability

* **MolHIV corrected key → exactly 1 descriptor** (variance 1.3e-14): the
  descriptor is a pure function of the corrected key.
* ZINC corrected identity → mean 1.64 descriptors (max 9). The residual is a
  **descriptor design fact**: coordinate 145 of `_shell_descriptor`
  (`degrees.mean()/4`) uses the **full-molecule** degree while all other
  coordinates use the induced patch, so it leaks context outside the patch.
* Historical → corrected split: ZINC radius-2 34.8% (mean 2.24, max 104);
  MolHIV 36.4% (mean 3.32, max 1,026). The radius-1 **parent** token splits far
  more (ZINC 74.2%, mean 16.5) → the parent token is the real coarse
  parameter-sharing bucket.

## 4. kNN identity purity (train-fit standardized descriptor, k = 1, 8, no sweep)

| dataset | top-1 | k=8 | random top-1 |
|---|---|---|---|
| ZINC historical | 0.7996 | 0.7664 | 0.0167 |
| ZINC corrected | 0.7988 | 0.7639 | 0.0057 |
| MolHIV historical | 0.893 | 0.846 | — |

ZINC frequency strata (historical, k=8): frequent (>20) **0.809** (n=18,147),
medium (6–20) 0.501, rare (2–5) 0.177, singleton **0.000**. Identity is largely
predictable from the descriptor on the head and essentially orthogonal on the
tail — the signature of categorical memory working where the descriptor is
data-poor.

## 5. Learned embedding geometry vs structural geometry (frozen checkpoints)

| metric | ZINC (6,784×16) | MolHIV (26,232×32) |
|---|---|---|
| participation ratio (all) | 3.08 | 18.99 |
| participation ratio (freq-weighted) | 9.82 | 8.31 |
| participation ratio (frequent) | 13.98 | 21.45 |
| stable rank (all) | 1.99 | 6.27 |
| norm vs log-frequency Spearman | +0.365 | +0.391 |
| NN overlap@10 vs structural | 0.46% (random ~0.15%) | 0.087% (random ~0.038%) |
| sampled-pair Spearman vs structural | 0.004 | 0.021 |

The learned identity matrices are **not** organised by structural similarity,
in either dataset; norms grow with frequency. The semantics are task-specific
arbitrary categorical coordinates.

## 6. Frequency / OOV / parameter share

| metric | ZINC hist. | ZINC corr. (uncapped) | MolHIV hist. | MolHIV corr. |
|---|---|---|---|---|
| vocabulary types | 6,784 | 15,218 | 26,232 | 87,039 |
| singleton type fraction | 0.392 | 0.434 | 0.393 | 0.469 |
| types freq ≤ 5 | 69.9% | 75.0% | 74.5% | 82.2% |
| top-100 coverage | 0.664 | 0.429 | 0.603 | 0.279 |
| valid OOV occurrence | 1.2% | 2.9% | 7.37% | 20.8% |
| embedding param share | 36.6% | — | **78.0%** | — |

Long-tailed vocabulary; storage tracks it. MolHIV's identity table alone is 78%
of the model and 59% of valid molecules contain an OOV patch — the parameter
pressure that motivates replacement.

## 7. with-ID vs without-ID expressivity (official ZINC train, target-lock-then-read)

Signature lock SHA-256 `dba5e3e6c90d45c18a11ffd136beb39897e69735df9e38bd6e5580157da077e9`
written **before** any official-train target read.

| system | unique | collision classes | collision mass | non-raw-isomorphic | target LB MAE |
|---|---|---|---|---|---|
| descriptor_only | 9,997 | 3 | 6 | 0 | 0.0 |
| descriptor + identity | 9,997 | 3 | 6 | 0 | 0.0 |
| full with identity | 9,997 | 3 | 6 | 0 | 0.0 |
| full without identity | 9,997 | 3 | 6 | 0 | 0.0 |

All four systems are identical: identity adds **zero** molecule-level
distinguishing power. The per-patch ambiguity of §2 does not aggregate into any
molecule-level collision.

---

## 8. MolHIV generalization (descriptive, read-only)

Overall valid AUC: seed0 raw 0.814 / soup 0.810; seed1 raw 0.850 / soup 0.851.
Test overall 0.780.

| subgroup (valid) | n | prevalence | mean error | raw AUC |
|---|---|---|---|---|
| zero OOV | 1,679 | 2.32% | 0.0288 | 0.917 |
| any OOV | 2,434 | 1.73% | 0.0245 | 0.744 |

* The AUC gap **persists** after size-decile stratification (zero 0.90–0.96 vs
  any 0.72–0.76) and after a molecule-frequency proxy control (essentially
  unchanged).
* The mean-error difference is **prevalence-confounded** (zero-OOV has higher
  prevalence yet higher mean error).
* Prior test phenomenon replicates read-only: zero-OOV test AUC 0.837 vs
  (0.1,0.3] 0.654; by min-train-frequency >5 0.894 vs (1,5] 0.670 vs 0 0.691.
* Scaffold frequency is approximated (no RDKit); subgroup n is small.

**Reading:** the OOV-difficulty signal is real and not merely size/scaffold, but
it is a **descriptive novelty/difficulty statement, not causal** and did not
inform any architecture choice.

---

## 9. Agent A — identity / capacity control (ZINC cell A)

**Question:** does cell A's 0.126368 depend on the learned per-identity lookup,
or only on identity distinguishability / generic shared capacity? (Naive
deletion is confounded by the released parameters.)

| condition | typed-token channel | params | soup seed0 | soup seed1 | **2-seed mean** |
|---|---|---|---|---|---|
| A0 frozen learned lookup | learned hybrid 36,420 | 85,763 | 0.124704 | 0.128032 | **0.126368** |
| A1 fixed code + shared adapter | fixed 16-D code, 36,397 adapter | 85,740 | 0.121245 | 0.126014 | **0.123629** |
| A2 zero slot + shared adapter | identity slot = 0 | 85,740 | 0.121694 | 0.122134 | **0.121914** |

Deltas vs A0 (negative = better): A1 −0.002739, A2 −0.004454, A2−A1 −0.001715;
all four per-seed deltas negative. Raw 2-seed means: A0 0.130562, A1 0.129283,
A2 0.124667.

* **Fairness:** released 36,420 lookup params reinvested as a 36,397-param
  **token-shared** adapter (input = [16-D identity slot ; 146-D shell]);
  A1/A2 are 23 params (0.027%) below A0. 43 shared tensors bit-identical to A0
  (max abs diff 0.0); graph head untouched.
* **Single variable verified:** A1 is permutation-sensitive (7.8e-04), A2 is
  invariant (0.0).
* **Sanity:** 25/25; baseline guard reproduces the frozen A0 forward
  (2.94e-08 best / 2.68e-09 soup); deterministic repro bit-identical.
* **Bootstrap (paired per-molecule, B=2000):** all 95% CIs include 0 → "not
  worse", not proven better.

**Verdict: Case III — typed lookup largely dispensable; shared capacity
suffices; identity distinguishability adds nothing (A1 ≈ A2).**

---

## 10. Agent B — shared structural patch encoder (vocab-free)

**Question:** can the vocab-sized typed certificate→ID→learned-embedding lookup
be removed entirely and replaced by a shared encoder reading each rooted typed
radius-2 patch's internal connectivity + semantics?

Pre-registered architecture (exactly one): node primitive = atom(28,48) +
root(2,48) + distance(3,48); edge primitive = bond(4,24); **2 untied edge-aware
message-passing rounds**; permutation-invariant `[root; mean; std]` pooling →
fusion MLP → `e_struct ∈ R^16`; **35,152** params replacing the 36,420 lookup.

| | seed0 | seed1 | 2-seed |
|---|---|---|---|
| raw best valid | 0.125324 (ep231) | 0.124394 (ep197) | 0.124859 |
| Top-5 soup valid | 0.119818 | 0.118126 | **0.118972** |
| candidate params | | | **84,495** |

Δ vs Cell A 0.126368 = **−0.007396**, per-seed −0.004886 / −0.009906 (same
direction) → **STRONG_SUCCESS**. Deterministic GPU repro bit-identical across
both A100s; 14/14 sanity + 72 repo tests; no `typed_embedding.weight`; OOV
naturally encodable; parent/q16/h64/T2/tying preserved.

**Binding caveat (honest mechanism):** the trained encoder output is **nearly
rank-1 across patches** (effective rank 1.107 / 1.202; top singular fraction
0.979 / 0.955) while patch `h0` keeps rank ~18–19. So the gain must **not** be
attributed to rich structural message passing; it is a small shared
connectivity-derived (near-bias) function replacing the lookup.

MolHIV parameter projection (accounting only, no training): 1,076,589 − 839,456
+ 35,936 = **273,069**.

---

## 11. Historical experiment reconciliation (updated)

| experiment | changed about identity | retrained | lookup kept | axis | conclusion | constraint |
|---|---|---|---|---|---|---|
| typed tokenizer correctness repair | coarse topology → complete coloured key | yes | yes | information / sharing | corrected key is correct but slightly worse; coarse token was accidental sharing | do not call the coarse token "exact"; correctness ≠ gain |
| corrected fragmentation / rarity audit | none (measurement) | no | yes | sharing | degradation is baseline-difficulty redistribution | fragmentation is not the lever |
| compositional patch sharing oracle | frozen embedding transplant for rare/OOV | no | replaced for rare/OOV | sharing | NO-GO | structural KNN does not transfer identity semantics (matches §5) |
| compact-v6 topology-attribute factorization | exact token → shared (type, role) encoder | yes | no | information / function class | NO-GO | additive attribute factorisation cannot replace the key |
| SBCI shared-basis compositional interaction | identity kept; relation/centre family replaced | yes | yes | function class / sample efficiency | NO-GO | compositional relation bases buy no sample efficiency |
| current recurrent Cell A | identity lookup retained (~36.6%) | yes | yes | capacity / architecture | params dominated by identity storage | capacity and identity entangled |
| MolHIV parameter attribution | none (accounting) | no | yes | parameter accounting | identity = 78.1% of ~1.08M | large-vocab cost is the real pressure |
| **Agent A capacity control** | **learned lookup → shared adapter; fixed code vs zeros** | **yes** | **no (A1/A2)** | **capacity / distinguishability** | **Case III: lookup dispensable; distinguishability adds nothing** | **do not treat the lookup as necessary; do not claim superiority (2 seeds)** |
| **Agent B shared encoder** | **lookup → vocab-free connectivity encoder** | **yes** | **no** | **information / function class** | **STRONG_SUCCESS (Δ −0.0074) but encoder near rank-1** | **vocab-removal result, not structural-bias evidence** |

**Thread.** Correctness repair showed the coarse token was an accidental sharing
device; fragmentation showed sharing is not the lever; the sharing oracle, v6
and SBCI showed structural replacements fail to recover whatever the lookup
provides; the audit explained why (lookup is non-structural and
molecule-level-redundant); Agent A showed the lookup is not needed at all
(shared capacity suffices); Agent B showed a vocab-free shared function replaces
it with a small gain.

---

## 12. Final decision map (Case A/B/C/D)

* **Case A — mostly redundant: SUPPORTED at molecule level.** Descriptor-only
  signatures already achieve 0 non-raw-isomorphic collision classes; identity
  adds nothing (§7).
* **Case B — distinguishability matters, semantics unclear: SUPPORTED
  descriptively, but NOT functionally.** Per-patch separation is real (§2, §4),
  and the learned geometry is non-structural (§5). However Agent A's A2 (no
  identity at all) matches A1 (fixed code) and beats A0 — so the
  distinguishability is not required for the task. Case B's "distinguishability
  matters" half is therefore **not** confirmed at cell A scale.
* **Case C — genuine missing structural information: SUPPORTED per-patch only.**
  Identity resolves substantial per-patch ambiguity (45.9% ZINC / 23.5–26.1%
  MolHIV collision mass) but it never aggregates into molecule-level collisions
  (§7) and the descriptor gap is not required by the model (Agent A/B).
* **Case D — unique learned task semantics: UNSUPPORTED.** No isolating
  experiment; the geometry is not structural; and the lookup is replaceable
  (Agent A Case III, Agent B strong success).

**Answer to the core question.** Identity is **replaceable task-specific
categorical memory**: it does provide genuine per-patch categorical separation
and stores a long-tail of task-specific arbitrary coordinates, but it supplies
**no** necessary molecule-level structural information, its geometry encodes
**no** structural organisation, and on the frozen ZINC cell A it can be removed
in favour of shared capacity or a vocab-free shared function without loss. It is
**not** "missing raw structural information", **not** required
distinguishability, and **not** demonstrated unique task semantics.

---

## 13. Pre-registered guidance → observed resolution

The audit fixed its interpretation before Agents A/B finished. Joint reading
table from the audit:

| observed | pre-registered meaning | actual |
|---|---|---|
| A Case I (A0≈A1≪A2) + B neutral | distinguishability matters, per-ID memory not necessary | — |
| A Case II (A0≪A1≈A2) + B failure | memory-dominance | — |
| **A Case III (all equal) + B success** | **identity largely dispensable; retire the lookup line** | **← this happened** |

Agent A = Case III; Agent B = STRONG_SUCCESS (vocab-free). The pre-registered
rule therefore fires unambiguously: **the typed-lookup line is retired as a
necessity**. The audit's Case B ("distinguishability matters") is **superseded**:
Agent A shows even the fixed-code distinguishability is unnecessary, and Agent B
shows the memory is replaceable.

Additional pre-registered points that held:
* Agent A parameter parity was honoured (85,740 vs 85,763, 0.027%).
* Agent B reported the near-rank-1 caveat and did **not** claim a structural
  inductive bias.
* No ZINC official test read; no MolHIV training.

---

## 14. Limits, no-go compliance, artifacts

**Limits.** 2 seeds for both training experiments; per-seed spreads comparable
to the deltas and per-molecule bootstrap CIs include 0 (Agent A). Agent B ran
one pre-registered architecture, no test confirmation, and its encoder is nearly
rank-1, so "vocab-removal" is the defensible claim, not "structure is used".
ZINC target-dependent statements are limited to the locked-signature empirical
LB on official train. MolHIV §8 is descriptive, prevalence-confounded in its
mean-error form, and uses a scaffold proxy. ZINC corrected identity → descriptor
variability is explained by a single context-leaking descriptor coordinate.

**No-go compliance (all three lines):** no ZINC official test; no architecture
selection beyond the single pre-registered candidate; no identity deletion
except as an experimental arm; no new embedding design outside the registered
adapters/encoder; no top-K cutoff chosen; no hash compression; no optimizer
sweep; Agent B did not train MolHIV.

**Artifacts.**
* Audit: `results/identity_incremental_information/` (dataflow, collision, kNN,
  frequency, expressivity + signature lock, embedding geometry for ZINC and
  MolHIV, generalization, final_decision, `answers_q1_q20.json`, 3 figures) +
  `notes/identity_incremental_information_audit.md` +
  `tests/test_identity_incremental_information_audit.py` (10 pass) +
  `records/claims|decisions/...identity-incremental-information-audit-20260915.yaml`.
* Agent A: `results/compact_v4_identity_capacity_control/` (parameter_accounting,
  sanity 25/25, baseline_guard, diagnostics, decision, report, runs/, soups,
  states/, curves/) + `notes/compact_v4_identity_capacity_control.md` +
  `tests/test_compact_v4_identity_capacity_control.py` (5 pass) +
  `records/claims|decisions/...zinc-identity-capacity-control...20260915.yaml`.
* Agent B: `results/shared_structural_patch_encoder/` (runs, soups, states,
  diagnostics, parameter_accounting, molhiv_projection, sanity, decision,
  report, repro) + `notes/shared_structural_patch_encoder.md` +
  `tests/test_shared_structural_patch_encoder.py` (11 pass) +
  `records/claims|decisions/...shared-structural-patch-encoder-vocab-free-20260915.yaml`.
* Code modules: `identity_incremental_information_audit.py`,
  `zinc_compact_v4_identity_capacity_control.py`,
  `structural_patch_encoder.py`, `zinc_shared_structural_patch_encoder.py`.

**What would reopen this.** (a) A new pre-registered mechanism witness showing
the lookup is necessary (more seeds, pre-declared threshold), or (b) an
authorized, pre-registered official-test read for a frozen vocab-free variant.
Not by sweeping encoder width/rounds/pooling, not by reopening the closed
identity lines, not by MolHIV training.