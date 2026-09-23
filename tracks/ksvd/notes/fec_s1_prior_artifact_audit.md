# FEC-S1 — prior-artifact audit

Round: **FEC-S1** (`fec_s1`, study `zinc-context-gap`).
Pre-registration: [`fec_s1_preregistration.md`](fec_s1_preregistration.md).
Lineage HEAD at audit time: `068cb06` (`record(fec-s0) ... STOP LOCAL_FACTORIZATION_BLOCKED`),
worktree clean.

This audit is written **before any code change, deploy or GPU run**. Its only
job is to prove that FEC-S1 is not a re-run of a closed experiment and to fix
exactly what is new.

Official ZINC **test is never loaded** by this round. No baseline is retrained
by this audit.

---

## 0. What FEC-S1 is

> In historical **strict-static S0**, delete the two per-key learned lookup
> memories (`typed_embedding`, `parent_embedding`) *completely*, and replace
> them with one parameter-matched, vocabulary-independent shared function
> `A(x_i): R^146 -> R^24` that reads only the already-factorized, standardized
> local environment descriptor `patch_cont_i`. Keep every other S0 computation
> frozen. Does the strict-static S0 performance band survive?

Success would give the first object of this line that is simultaneously:

```
shared local environment formation  ->  read-only static composition  ->  y
no vocabulary-sized local memory
no message passing / no recurrence / no pair->centre / no context writeback
```

---

## 1. FEC-S0 (the object we build on)

| field | value |
|---|---|
| protocol / revision | `fec_s0` / `d4de88e` |
| verdict | `FEC_S0_LOCAL_FACTORIZATION_BLOCKED` (Case C) |
| result | `patch_cont` 146, `pair_relation` 23, `global_context` 62, `topology_features` 25 all rebuilt from raw primitives **bit-identically**; `h/q/R/pred` bit-identical; fixed-checkpoint valid MAE `0.1456743378872634` |
| active local blocker | `typed_token` (16-D, `_HybridEmbedding`, 6,785 ids, **36,420 params**) + `parent_token` (8-D, **256 params**) |
| historical S0 object | `StrictStaticPairModel`, `center_context=False`, `center_update=None`, `residual_mode="none"`, total **66,228 params** |
| historical S0 seed0 | best valid `0.14567435123870381` @ epoch 164; recorded Top-5 soup `0.140794` (members `[125,142,159,164,167]`) — **member states not persisted** |
| historical S0 seed1 | soup `0.136423` |

FEC-S0 stopped *before* any attempt to remove the blocker: it is an audit, not a
performance run. FEC-S1 is the natural next step, and it is a **performance
round** (one full seed-0 training run) with a hard correctness gate suite.

**Consequence for FEC-S1 provenance**: the S0 Top-5 soup `0.140794` cannot be
replayed (no member states). The matched reference this round may only use the
persisted **selection checkpoint** `0.14567435...` (read-only valid replay) plus
the FEC-S1 run's **own** Top-5 soup.

---

## 2. Identity incremental-information audit

| field | value |
|---|---|
| branch | `audit/identity-incremental-information` |
| module | `identity_incremental_information_audit.py` |
| training | **none** (audit) |
| verdict | Case A (molecule-level redundant) + Case B (distinguishability matters) + Case C (per-patch descriptor ambiguity); Case D (unique task semantics) **unsupported** |

Finding: the discrete typed identity is a genuine **per-patch separator** that
the deterministic 146-D descriptor does not fully supply, but it is redundant
for molecule-level distinguishability, and its learned geometry is **not**
organised by structural similarity. It is best read as *distinguishability +
arbitrary task-specific categorical memory*, **not** missing raw structural
information. It also re-identified the MolHIV typed certificate as the same
coarse uncoloured rooted-topology invariant.

Relevance to FEC-S1: this is the **theoretical licence** for replacing a
per-key table with a shared function — identity is not missing structure. It is
**not** an experimental result in the strict-static S0 computation class and is
not counted as FEC-S1 evidence.

---

## 3. Identity / capacity control A0 / A1 / A2

| field | value |
|---|---|
| protocol | `compact_v4_identity_capacity_control_v1` |
| base | frozen ZINC **cell A**: 85,763 params, `T=2` weight-tied **recurrent pair–centre** |
| A0 | frozen learned hybrid typed lookup (36,420 params) |
| A1 | fixed deterministic 16-D code + token-shared adapter (36,397 params) |
| A2 | same adapter, identity slot forced to exact zeros |
| result (2-seed Top-5 soup) | A0 `0.126368`, A1 `0.123629`, A2 `0.121914` |
| verdict | Case III — typed lookup largely dispensable; A1 ≈ A2; *not worse*, not proven better |

## Q1 — How is FEC-S1 different from identity-capacity-control A2?

* **Computation family.** A2 is a **recurrent** cell-A model (weight-tied
  `T=2` pair–centre with `center_update`). FEC-S1 is the **strict-static** S0
  model (`center_context=False`, `center_update=None`, no recurrence). The A2
  result does not transfer automatically, because recurrent pair→centre
  computation can absorb identity differences that a static composition cannot.
* **Which memory is retired.** A2 isolates **only** the typed lookup and leaves
  the parent embedding intact. FEC-S1 must retire **both** `typed_embedding`
  and `parent_embedding` — both are FEC-S0 blockers.
* **Adapter input.** A2's adapter reads `[16-D identity slot ; 146-D shell]`,
  i.e. it still receives an identity code. FEC-S1's adapter reads **only** the
  146-D factorized descriptor; there is no identity slot at all.
* **Downstream.** FEC-S1 must inherit the strict-static S0 composition
  bit-for-bit except for the two deleted modules.

**Rule adopted:** the A2 positive result is recorded as *prior motivation only*.
It is **never** cited as a FEC-S1 result, and it does not relax any FEC-S1 gate.

---

## 4. Shared structural patch encoder (vocab-free)

| field | value |
|---|---|
| protocol | `shared_structural_patch_encoder` |
| architecture | 2 untied edge-aware **message-passing** rounds over the rooted typed patch → `[root; mean; std]` → 16-D, 35,152 params |
| result | 2-seed soup `0.118972` vs cell A `0.126368` (Δ −0.0074) |
| caveat | encoder is **near rank-1** (effective rank ≈ 1.1–1.2); read as a *vocab-removal* result, not structural-bias evidence |

Relevance: a shared, vocab-free replacement can match/beat a learned table in
that family, but that result is confounded in two ways FEC-S1 deliberately
avoids: (i) it used **message passing** between patch nodes; (ii) it read
**atom/bond primitives** rather than the already-formed environment. FEC-S1
reads only the explicit 146-D environment and performs **no** MP.

---

## 5. Local-token-null

| field | value |
|---|---|
| protocol | `local_token_null_v1` |
| intervention | eval-only knockout of `e_patch` (zero / mean / within-molecule permutation) on frozen soups |
| A0 (learned lookup) | mean token norm 0.44/0.36; **high-rank** categorical memory; zero-16 Δ = **+0.196** |
| B-Bag (shared bag) | mean token norm 0.016/0.002; zero-16 Δ = **+0.0003** (inert) |
| B-Full (shared structural) | near rank-1; zero-16 Δ = **+0.0025** (nearly inert) |

## Q4 — Why not start from a naive token-null?

Because naive deletion of `typed_token` + `parent_token` removes ≈ 36,676
trainable parameters (≈ 55 % of S0) at the same time as it removes the
identity information. The observed change would mix two causes:

1. loss of per-key identity information;
2. loss of generic trainable capacity.

FEC-S1 therefore **must** reinvest the released budget into the shared adapter
to within **≤ 1 %**, so that the only experimental variable is the *form* of
the local channel (per-key memory vs shared function), not its size.
`local_token_null` is cited only as the evidence that the historical lookup
token is a genuinely high-rank memory in the *mixed/recurrent* backbone.

---

## 6. Dictionary lines (SRDA-v0, SDPK-v0, DTX-v0)

| line | what it changed | result (seed 0) |
|---|---|---|
| **SRDA-v0** | deleted typed+parent identity outright **and** replaced the raw pair interaction with a dictionary prototype algebra; standalone no-MP model | best `0.155449`, soup `0.149873` (S0 soup `0.140794`, Δ ≈ **+0.0091**) |
| **SDPK-v0** | promoted the dictionary to the pair-kernel coordinate system; kept the inherited S0 channels | best `0.142193`, soup `0.139735`; gate FAIL (`SDPK_V0_NO_STRONG_PERFORMANCE_SIGNAL`) |
| **DTX-v0** | dictionary environment × generic topology role cross | soup `0.144330` vs aligned arm `0.150357`; verdict `NO_ALIGNED_CROSS_SIGNAL` |

## Q2 — How is FEC-S1 different from SRDA?

SRDA changed **two** variables at once: it deleted identity memory **and**
replaced the raw pair interaction with dictionary prototype algebra. Its
`≈ +0.009` regression therefore **cannot be attributed to identity deletion**.

FEC-S1 is a **single-variable isolation**:

> replace the identity-memory *path only*; keep the raw S0 pair interaction,
> relation encoder, pair encoder, graph composition and head exactly as S0.

## Q3 — Why no dictionary this round?

The only question is whether a **shared environment function** can substitute
for **per-key memory**. Introducing a dictionary would add a second, structurally
different object (learned prototype vocabulary + sparse codes) and reintroduce
the confound that already closed SRDA/SDPK/DTX. Therefore this round forbids:

```
NO K-SVD   NO sparse code   NO learned dictionary   NO prototype vocabulary
```

A dictionary line (`FEC-D1`) is contemplated **only if** FEC-S1 first reaches a
viable band, and even then only under its own pre-registration with
`g = 0 => FEC-D1 == FEC-S1`.

---

## 7. PEC-C1 / PEC-I1

| round | verdict | result |
|---|---|---|
| **PEC-C1** | `PURE_ENV_COMPOSITION_ABSOLUTE_WEAK` (Case A) | best no-MP arm `M_CD` soup `0.151767`; band `> 0.145`; pure architecture has insufficient absolute capacity |
| **PEC-I1** | `STATIC_POOLING_NOT_PRIMARY_GAP` (Case C) | internal 2k screen gain `+0.041287` → full-data gain `+0.000110`; pooling hypothesis closed |

Relevance to FEC-S1 (two rules, both adopted verbatim):

* PEC-C1 shows the *pure* PEC architecture is weak in absolute terms. FEC-S1
  does **not** rebuild a PEC model; it edits the proven-strong historical S0
  object minimally, so it is a *replacement* test, not a new architecture.
* PEC-I1 shows that a 2k small-data screen has **no** predictive value for this
  family's full-data effect size (`+0.041` → `+0.0001`). FEC-S1 therefore
  **does not run a small-data performance screen** to choose an architecture.
  After the correctness/parameter gates pass it buys exactly **one full seed-0**
  run.

---

## 8. Audit conclusion

1. **No prior round is equivalent to FEC-S1.** Every prior identity-removal
   experiment is either (a) a different computation family (A2 recurrent),
   (b) confounded with architecture deletion (SRDA), (c) a dictionary line
   (SRDA/SDPK/DTX), (d) a full read-only audit with zero training (FEC-S0), or
   (e) a different backbone and message-passing replacement (shared structural
   encoder).
2. **The FEC-S1 object is uniquely defined**: historical strict-static S0 with
   `typed_embedding` + `parent_embedding` deleted and replaced by one
   parameter-matched shared function of the 146-D factorized standardized
   `patch_cont`, all other S0 computation frozen.
3. **All hard requirements are declared in the pre-registration**:
   parameter matching, vocabulary-independence (token poisoning), bit-identity
   of the adapter input with the FEC-S0 factorized descriptor, strict-static
   purity, downstream architecture identity, and the frozen decision bands.
4. **No official test, no baseline retrain** (only a read-only checkpoint
   replay), **one full seed-0 run**, no rescue.

Evidence list (read-only): `notes/fec_s0_*.md`,
`notes/identity_incremental_information_audit.md`,
`notes/identity_incremental_information_study_report.md`,
`notes/compact_v4_identity_capacity_control.md`,
`notes/shared_structural_patch_encoder.md`, `notes/local_token_null.md`,
`notes/zinc_static_relational_dictionary_algebra_v0_analysis.md`,
`notes/zinc_static_dictionary_pair_kernel_v0_analysis.md`,
`notes/zinc_no_ring_dictionary_topology_cross_v0_analysis.md`,
`notes/pec_c1_analysis.md`, `notes/pec_i1_analysis.md`,
`results/zinc_static_dictionary_pair/runs/s0_seed0.json`,
`results/zinc_static_dictionary_pair/states/s0_seed0_selection_state.pt`.

`official_test_loaded = false` on every artifact used above.
