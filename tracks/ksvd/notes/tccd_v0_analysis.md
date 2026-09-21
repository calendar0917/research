# TCCD-v0 — analysis (Task-Coupled Compositional Dictionary, ZINC)

Pre-registration: [tccd_v0_preregistration.md](tccd_v0_preregistration.md)
(frozen before any run). Code `code/run_tccd_v0.py` + `code/tccd_v0.py`; tests
`tests/test_tccd_v0.py` (9 pass, data-free, CPU). Regime: deterministic remote
A100 host `res`, **GPU1**, canonical PyG ZINC `subset=True` official **train
10 000**; **official valid and official test were never loaded**
(`official_valid_used: false`, `official_test_loaded: false`, `y_used: false` in
Gate 1). Commit `008b5e3`.

**Frozen verdict (first line):**

> **A fixed-coordinate canonical raw local patch is not a suitable K-SVD /
> dictionary-learning domain: structurally near patches are as far apart in the
> frozen coordinate/code space as size/root-matched random patches.**

**Next decision (frozen, per the pre-registration):**
`STOP TCCD-v0 at Gate 1. Gates 2/3/4 are NOT RUN.`

---

## 0. What was asked

Verify the round hypothesis: *a molecule can be represented as reusable local
chemical environments learned from data, plus explicit observed composition
relations, without learned message passing.* The frozen candidate **TCCD-v0**
builds, for every atom/root, a radius-2 attributed rooted patch placed in a
fixed coordinate vector via the repository's exact rooted colored-incidence
canonicalization, learns a shared `K=64`, `s=8` dictionary, forms one-shot
composition contractions `Cᵀ R C` over a frozen 5-relation set, and uses a
single linear reader. The round executes Gate -1 → Gate 0 → Gate 1, and only
continues to Gates 2/3/4 if Gate 1 passes.

Official-test freeze was respected throughout (the loader refuses `test`, and
Gate 1 never uses `y`).

---

## 1. Gate -1 — canonical GPU1 baseline reference

*PASS*. A same-era, same-protocol, same-A100-regime canonical strong ZINC
baseline already exists and is documented in a Git-tracked claim: **B-Full /
shared-structural Top-5 soup valid MAE = 0.119818**, 84 495 params,
deterministic A100, seed 0, official test never loaded
(`records/claims/claim-local-token-null-20260919.yaml`; also
`STATE.yaml: development_references`). No fresh same-revision GPU1 re-run was
performed because no small architecture delta is interpreted at Gate 1 (Gate 1
is a label-free domain audit) and Gate 4 was not reached. The same-commit-era
GPU1 attempt `sc-bfull-s0` (commit `eb6668a`) reached valid 0.136360 at epoch
119/240 before its SSH session died, consistent with the ~0.12 soup.

---

## 2. Gate 0 — correctness (implementation)

*PASS.* Local data-free tests (9) pass, and the real-data GPU1 smoke
(`gate0`, commit `008b5e3`, device `cuda`, 48 molecules, `M=11`, `A=12`,
`F=396`) passes **all** pre-registered checks:

| check | value | threshold | result |
|---|---|---|---|
| permutation `x_v` | max Δ = 0.0 | ≤ 1e-6 | PASS |
| permutation code support Jaccard | mean 1.0 | ≈ 1.0 | PASS |
| permutation `CᵀRC` (float64) | max Δ = 5.7e-14 | ≤ 1e-6 | PASS |
| permutation prediction | max Δ = 0.0 | ≤ 1e-5 | PASS |
| batching invariance | max Δ = 0.0 | ≤ 1e-5 | PASS |
| relation correctness (joint perm) | ≤ 1e-6 (float64) | ≤ 1e-6 | PASS |
| sensitivity (permute `C` rows only) | rel. change 0.313 | ≥ 0.05 | PASS |
| exact sparsity under IHT | `nnz = 8` | = 8 | PASS |
| task + reconstruction grads reach `D`; `D` changes; no norm collapse | yes / `col_norm∈[1-6e-7, 1+2e-7]` | — | PASS |

**Implementation finding (fixed during the round, before any formal run).**
The first coordinate implementation used `canonical_atom_order`, whose
`pynauty.canon_label` is **not** permutation-invariant as a vertex *ordering*
when the colored graph has automorphisms (verified directly: path-3 and ring-6
canonical labelings differ across relabelings, while the certificate/key is
invariant). This produced `max Δx_v = 1.0`. The fix uses the repository's exact
**rooted** colored-incidence construction
(`experiments/luyin16/typed_patch_tokenizer.build_colored_incidence`), whose
vertex colours include `(is_root, distance_from_root, atom_type)`, so the
canonical labelling is root/shell-preserving and `x_v` is exactly invariant
(48 200 relabeling checks, max Δ = 0.0). A second finding: the tied IHT step
size `η = 1/σ_max(D)²` originally used a **random** power-iteration start
vector, making `η` non-deterministic and breaking invariance; it now uses a
frozen deterministic start vector. Both were bugfixes to the frozen
construction, not architecture changes; the pre-registration was amended
before any formal run.

---

## 3. Gate 1 — local dictionary domain (the decisive gate)

Official train only, no `y`. Deterministic internal split `seed 20260922`:
**8 000 internal-train / 2 000 internal-dev**. Capacity
`M = max train radius-2 patch size = 14` (no truncation anywhere:
`truncations = 0`), `A = 21` atom categories, `B = 3` bond categories,
**`F = 714`**. Fit set `X_fit = 185 538 × 714`; holdout `X_dev = 46 126 × 714`.
Detached OMP + mini-batch K-SVD, `K = 64`, `s = 8`, 12 chunk epochs
(`chunk 16 384`, frozen `seed 20260922`); wall `3 145 s`.

### 3.1 Pre-registered checks

| # | check | value | threshold | result |
|---|---|---|---|---|
| 1 | permutation `x_v` (100 dev mol × 20 relabels) | max Δ = 0.0 | ≤ 1e-6 | PASS |
| 1 | OMP support Jaccard mean / frac=1 | **1.0 / 1.0** (48 200 checks) | 1.0 / ≥0.99 | PASS |
| 2 | held-out reconstruction ratio `E_learned/E_random` | **0.0491** | ≤ 0.70 | PASS |
| 2 | `E_learned` (dev) / `E_random` (dev) | 0.0470 / 0.9568 | — | — |
| 2 | `E_learned` (fit) / `E_random` (fit) | 0.0466 / 0.9568 | — | — |
| 3 | dead atoms (support < 5) | **0** | ≤ 10 % | PASS |
| 3 | reused atoms (support ≥ 20, ≥ 5 mols) | **64 / 64** | ≥ 60 % | PASS |
| 3 | top-1 / top-8 atom mass share | 0.0515 / 0.2879 | ≤ 0.20 / ≤ 0.80 | PASS |
| 3 | mean coefficient entropy | 1.542 bits | report | — |
| 4 | **local-coordinate continuity, code-space AUC** | **0.4675** | **≥ 0.70** | **FAIL** |
| 4 | local-coordinate continuity, `x`-space AUC | 0.5019 | report | — |
| 4 | graded tail stratum (WL rank 801–1600) code AUC | 0.5494 | report | — |
| 5 | atom semantics (report only) | mean top-50 canonical-key concentration **0.610** vs random baseline 0.075 (8.1×); max 1.0 | report | — |

`gate1_pass = False`; single stop reason `continuity_auc`.

### 3.2 The continuity audit in detail

1 500 random (molecule, root) patches from the 2 000 internal-dev molecules
were fingerprinted with a typed WL subtree kernel (rounds 0–3) on the **raw
attributed rooted patch**. *Near* pairs = the 800 highest-cosine pairs that are
**not isomorphic** (different complete corrected canonical key); *random* pairs
= matched on (patch size, root atom category). Distance in `x`-space and in OMP
code space; AUC = P(near pair is closer than its matched random pair).

* near-pair WL cosine mean = **0.99999999999** (the near pairs are essentially
  1-WL-equivalent but non-isomorphic — the hardest genuine "structurally near"
  pairs);
* `x`-space: near mean 3.164 vs random 3.193 → AUC **0.502**;
* code-space: near mean 5.799 vs random 5.710 → AUC **0.467** (slightly *below*
  chance);
* graded tail (structurally less-close pairs): code AUC 0.549, `x` AUC 0.528;
* the same result held on the 400-graph development smoke (code AUC 0.54,
  `x` AUC 0.54), i.e. it is not a full-data artefact.

**Interpretation.** The exact canonical coordinate construction is perfectly
permutation-invariant and the K-SVD dictionary *does* learn the coordinate space
extremely well (holdout relative error 4.7 % vs 95.7 % for a matched random
dictionary; 64/64 atoms reused, no dead atoms). But the coordinate space itself
does **not** carry the raw structural nearness relation: two non-isomorphic
patches with near-identical rooted WL signatures are scattered to (and coded at)
essentially random distances, because the canonical slot arrangement of a patch
changes discontinuously under small structural changes. This is the exact
failure the round brief anticipated ("canonicalization correct ≠ Euclidean
coordinate space suitable for dictionary learning"; "if canonical slot
rearrangement makes structurally near patches as far apart as random patches,
FAIL").

The atom-semantics result is consistent and not contradictory: mean top-50
canonical-key concentration 0.61 vs 0.075 random (8.1×) means atoms do latch
onto **repeated exact** rooted motifs (isomorphic patches share the same
coordinate vector), but that exact-motif repetition is orthogonal to metric
smoothness — the continuity audit explicitly excludes isomorphic pairs.

### 3.3 Why this is not a tuning problem

* `K`, radius, sparsity, the relation set and the reader are frozen; §5 of the
  pre-registration and §11 of the round brief forbid rescuing a failed gate by
  changing them.
* The reconstruction/reuse/permuation checks all pass **comfortably**, so the
  failure is not "the dictionary is bad" or "the code is broken"; it is
  specifically that the frozen coordinate domain does not preserve structural
  proximity.
* The matching smoke at `M=11, F=396, 400 graphs` shows the identical failure
  (AUC 0.54), so it is not an artefact of capacity or data scale.

---

## 4. Gates 2 / 3 / 4 — not run

Gate 1 FAIL stops the round. No composition (`BAG`/`REL`/`REL-SHUFFLE`), no task
coupling (`FROZEN-D`/`TASK-D`/`DENSE`), and no full-data absolute comparison was
run. No official test was opened; no test read is spent by this round.

---

## 5. Answers

* **Q1 — Does exact rooted canonicalization give a permutation-invariant
  coordinate vector?** **Yes**, once the colouring includes root identity and
  shell (the repository's `typed_patch_canonicalization`), and the tied IHT step
  size is deterministic: `x_v`, codes, `CᵀRC` and predictions are exactly
  invariant and batching-invariant (Gate 0 PASS).
* **Q2 — Is the fixed coordinate vector a good K-SVD domain?** For
  *reconstruction* and *reuse*, yes (holdout ratio 0.049; 64/64 reused atoms).
  For **local-coordinate continuity**, **no** (code AUC 0.467, `x` AUC 0.502
  against a pre-registered PASS threshold of 0.70).
* **Q3 — Does the dictionary find structure at all?** Yes for repeated exact
  motifs (atom-semantics enrichment 8.1×) and for reuse, but the
  dictionary's success is confined to the slot space; it does not make the
  representation metric-smooth in raw structural terms.

---

## 6. Verdict

> **A fixed-coordinate canonical raw local patch is not a suitable K-SVD /
> dictionary-learning domain for ZINC: structurally near patches are as far
> apart in the frozen coordinate/code space as size/root-matched random
> patches.**

## 7. Decision

**STOP TCCD-v0 at Gate 1.** Write the negative result, do **not** start Gate 2/3
and do **not** run the full-data TCCD-v0 model. No rescue is permitted in this
round by changing `K`, radius, slot ordering, adding a learned local encoder, or
strengthening the reader. If the local-dictionary hypothesis is pursued again,
it must be a **new, explicitly different pre-registration** — e.g.
*permutation-invariant learned local encoder → dictionary* (a different object
class that does not rely on canonical slot coordinates), or a
permutation-invariant reconstruction objective (bag/multiset of canonical
sub-structures) rather than slot-indexed coordinates, as the whole-graph
canonical-registration audit already recommended
(`notes/wholegraph_canonical_registration_audit.md` §9).

## 8. Scope / what cannot be said

* This is **ZINC** (radius-2 patches up to 14 atoms), seed 0, one internal
  8 000/2 000 split, `K=64`, `s=8`, one frozen coordinate construction and one
  WL-based nearness operationalization. It is **not** a statement that "dictionary
  learning on molecular patches is impossible".
* The continuity audit is an operationalization: nearness is a typed WL subtree
  kernel and the AUC is paired against (size, root-category)-matched random
  pairs; the failure is uniform across the primary and graded strata and across
  the smoke and full runs, but a different nearness notion could differ.
* Gates 2/3/4 were not run, so nothing can be said about composition relations,
  task coupling, or absolute predictive performance of TCCD-v0.
* Official valid and test were never read.

## Artefacts

`tracks/ksvd/results/tccd_v0/`: `gate0.json` (GPU1 smoke, device `cuda`),
`gate1.json` (full audit), `dictionary_gate1.pkl`,
`cache/tccd_v0_train_r2_M14_A21_B3_n10000.pkl` (git-ignored). Remote runs:
`tccd-gate1` (GPU1, commit `008b5e3`, exit 1).
