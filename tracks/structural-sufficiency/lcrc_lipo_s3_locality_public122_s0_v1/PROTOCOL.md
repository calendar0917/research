# PROTOCOL — LCRC-C1 structural-confusion audit and a minimal 3-hop static addition

Round / protocol ID: `lcrc-lipo-s3-locality-public122-s0-v1`
Directory: `tracks/structural-sufficiency/lcrc_lipo_s3_locality_public122_s0_v1/`
Design date: 2026-11-13 (Asia/Shanghai); design snapshot `f0814e859d9`.

Frozen before any formal GPU training of this round. The scientific
definitions below are not changed after the first formal run. Only §13
(P0 observed support) is appended after freezing (still before any formal
GPU training) and §1's hash list is copied verbatim from the frozen sources.

---

## 0. Question, two propositions, boundary

The object of study is always LCRC-C1 / `lipo_v1` with the frozen 51/13/9/42
input semantics. Two separate questions are asked and answered separately:

* **H_rep** — does the current C1 input summarise *distinct* connected
  structures into one input, and does one per-root typed 3-hop statistic
  separate a specific, already-located subset of those cases at the input
  level?
* **H_pred** — does adding it give a *practical* out-of-sample predictive
  gain over (a) the reused ordinary C1 baseline and (b) a same-capacity
  control that only re-reads information C1 already contains?

H_rep is a statement about the representation. H_pred is a statement about
generalisation on 4200 public molecules. H_rep is *not* evidence for H_pred,
and five pairs cannot explain the residual error, nor can label differences
alone establish chemical causality.

**Boundary of the round.** Lipo only, published `random122` split, formal
training seed 0 only. No seed replication, no width/weight/radius sweep, no
new split, no other dataset, no fourth hop, no distance bucketing, no external
model (PNA/GIN/…), no pair loss, no attention, no pretraining, no iterative
state propagation, no auxiliary loss, no correction-strength tuning. Maximum
three formal trainings this round (LOCAL3, DUP2, and NULL3 only if the fixed
Stage-II purchase rule of §9 passes).

Data policy: Test is exposed and readable for evaluation, but Test labels
never enter gradients and never select a checkpoint (Valid RMSE does). Every
selection and conclusion in this round is exploratory (seed 0).

---

## 1. Data and split (frozen)

ChemBench upstream commit `0b10d4c1ac72b6929503694a5f60446824736391`.

| file | SHA256 |
|---|---|
| `Lipop.csv.gz` | `34280f0adde234bcf477bd98878c9918c9ea51ed859c9efdc31917c6e5ecd758` |
| `Lipop.random122.ind.pkl` | `bf7acb177e080caf2be9ee220f479f8ea91e0039ff4867a7281753a2fce24a7d` |
| `raw_attributes_v1.json.gz` | `1127607de45e341c89ffc67db85714117831ec84b0845c75aaa11005489c51be` |

Split sizes Train/Valid/Test = 3360/420/420 (4200 molecules). Published
row-index bytes, int64, original order:

* Train `4e8d42a95c0b705f43f932b3bbb265277c76a7aa65684b0e5e6ca028580aaf96`
* Valid `31e3f28f31da69249f70922126f19795920bc1e32cc05d5e29e27507bd5aa365`
* Test  `a4d92ad939bf268c00ff69afe77aa5d58c19a6f809b6a09f6dadc51cd648ec93`

Feature-spec hash `7270f7677b349a46bb193174381abcb84f14d1b4f83bd1362d4aa20389a515ee`
(`lipo_v1`, atom_feat 51, bond_feat 13, atom_scalar 9, global 42, atom_type
16 classes including UNK, S1 80 codes, S2 400 codes). Source files are located
**by content hash**; a migrated path does not change identity. All 4200
molecules and their original atom order are kept: no desalting, no
deduplication, no stereo filtering, no row reordering.

Target `exp` (raw logD). `z = (y - mu) / sigma` with `mu`, `sigma` fitted on
Train only (ddof = 0); evaluation always converts back to raw logD units
exactly once. Structure normalisation is likewise Train-only.

---

## 2. The five frozen confusion cases (design-time candidates)

0-based ChemBench row ids. All ten rows are already-exposed mechanism cases;
they are **not** an independent test set. Design-time A predictions and
labels:

| row A | row B | split A/B | y_A | y_B | pred_A | pred_B | U3 expected |
|---|---|---|---|---|---|---|---|
| 294 | 1665 | train/train | 1.74 | 2.20 | 1.696551646 | 1.696551574 | separates |
| 748 | 913 | train/train | 2.92 | 2.92 | 2.821447260 | 2.821446972 | separates |
| 963 | 4186 | train/test | -0.03 | 0.83 | 0.371185149 | 0.371185581 | separates |
| 1467 | 2674 | train/train | 3.90 | 3.80 | 3.597154216 | 3.597153784 | **not** separated |
| 2102 | 3823 | train/train | 0.98 | 1.06 | 1.047859854 | 1.047859781 | separates |

Only 4186 lies in Test. 294/1665, 963/4186 and 2102/3823 are related
heteroatom-position variations along a chain and are *not* three independent
chemical regularities. 748/913 is kept and reported although `y_A = y_B`
(Δlabel = 0), so the case list is not filtered by label difference.

---

## 3. Sufficient per-root signature (the auditable object)

For a root atom `v` the signature `S_v` is the following tuple of multisets,
built **only** from fields the frozen forward actually reads, with all
identity fields excluded:

| block | content |
|---|---|
| A | `atom_feat[v]` (51) + `atom_scalar[v]` (9) |
| B | multiset of `(atom_feat[u], bond_feat[b])` over incident bonds |
| C | multiset of S1 `nbr_code` |
| D | multiset of `(path_cat, path_end63)` |
| E | multiset of `(path_cat, atom_feat[path_end])` |
| F | multiset of `path_closure` |

Repeated occurrences are preserved (no set-ification). Floats are serialised
as canonical finite float32 bytes with signed zeros folded (`-0.0 -> +0.0`);
integers keep their original type semantics. A stable serialisation +
SHA256 is used only to *bucket* candidates; a hit is confirmed by comparing
the full block contents field by field. The signature contains no atom id,
no `path_end` index, no molecule id, no SMILES, no split, no label.

**Why `S_v` is a sufficient description of the local input.** Reading the
frozen `DissectLCRC` local path (`dissect_model.py`): the 433 C1 columns are
`a`(96), `b_sum`(96), `s1_joint`(32), `s2_type`(32), `s2_end`(32),
`s2_joint`(32), `s2_closure`(8), `s2_content`(96), `atom_scalar`(9).

* `a = atom_encoder(atom_feat[v])` and `atom_scalar[v]` — fixed by A.
* `b_sum` is an `index_add` over incident bonds of the encoded
  `(bond_feat, atom_feat[neighbour])` pair — fixed by B (the multiset, hence
  also the degree).
* `s1_joint` is an `index_add` over incident S1 codes — fixed by C.
* `s2_type`, `s2_end` are per-occurrence sums of `path_emb(path_cat)` and
  `end_emb(path_end63)`, and `s2_joint` is a per-occurrence sum of
  `joint_proj(path_u(path_cat) * path_v(path_end63))` — all three are sums of
  a function of the *pair* `(path_cat, path_end63)`, hence fixed by D.
* `s2_closure` sums `close_emb(path_closure)` — fixed by F.
* `s2_content` sums `path_gate(path_cat) * a[path_end]` — a per-occurrence
  function of the pair `(path_cat, atom_feat[path_end])`, hence fixed by E.
  (`path_closure` is *not* read by the content block; E must keep the full
  end-atom feature vector, not only its element.)

Therefore `S_v` equal ⇒ identical local input row and identical `h_v =
LayerNorm(MLP(x))` under **any** shared weights. This is what makes "input
aliasing" a statement about the model's own readout rather than about a
hand-written descriptor. `S_v` is *a* sufficient description, not claimed to
be minimal; a signature mismatch does not prove the model separates anything
numerically.

At molecule level: equal multiset of `S_v` **and** equal `global42` ⇒ equal
`mean h`, `mean h²`, `n_atoms` and `sum f(h)` ⇒ equal eval prediction, up to
floating-point reduction order. Hence the audit compares the local-signature
multiset and `global42` separately, and `global42` is never folded silently
into the local signature.

## 4. Pre-screen re-verification (no lightweight substitutes)

The design-time pre-screen (4200 molecules, 113568 atoms, 18122 root
signatures, 87 local groups → 86 after `global42`) was pure NumPy and used a
temporary biconnected-component adapter where a dependency was missing. Those
counts are **clues only**. This round rebuilds every structure cache through
the repository's own builder and re-runs the enumeration with the repo's
NetworkX labelled-graph isomorphism helpers (`edit_graph.py`), and checks the
resulting content fingerprints against the reused baseline cache. The
temporary adapter is not part of this round's implementation.

Explicit labelled-graph isomorphism (node labels `atom_class/charge/H`, edge
labels `edge_class/edge_stereo`) separates two very different situations and
they are never merged into one mechanism:

* `RAW_ATTRIBUTE_GRAPH_ISOMORPHIC` — the same graph written differently
  (typically stereochemistry annotations); not a new structure.
* `CONNECTIVITY_OR_ATTRIBUTE_NONISOMORPHIC` — genuinely different connected
  structures summarised into the same C1 input.

## 5. P0 (CPU) and its gates

P0 runs before any GPU work, on CPU, with the reused checkpoint only:

1. full data / source-hash / split / row-id / Train-only-norm / feature-spec
   identity, plus the structure-cache content fingerprints and row hashes;
2. enumerate identical local-signature multisets over all 4200 molecules,
   then add `global42` to get full-input groups; classify each member against
   the group representative by labelled-graph isomorphism;
3. real `eval` forward of the reused A checkpoint on the five pairs (no proxy
   model): per-pair local input and `h` multisets, `global42`, raw-unit
   prediction difference, plus a row-id join against the existing
   `per_molecule.csv` and the design-note predictions (±2e-5);
4. atom-renumbering and single-vs-batched inference invariance; groups whose
   signature is identical must show identical local input and `h` multisets;
5. feature separation: for every full-input group, compare the multiset of
   `(signature digest, structural vector)` pairs; `u3_separated` and
   `u2_separated` are recorded per group and per pair;
6. baseline identity: 228674 parameters, `best_epoch = 244`,
   Valid RMSE ≈ 0.7025358558, Test RMSE ≈ 0.5761674772, init state hash
   prefix `aac1e061`, and the same public122/seed0/plain-MSE A — the old FULL
   variant, pair-supervised R, or any other split is rejected.

**P0 pass condition.** All five pairs reproduce as raw-attribute-non-isomorphic
with equal local-signature multiset, equal `global42` and equal A prediction;
the new vector separates exactly four of the five (1467/2674 stays identical);
DUP2 separates none. If this cannot be reproduced, report
`PHENOMENON_NOT_REPRODUCED` and stop before formal GPU training. The feature
definitions are **not** adjusted to make the count come out.

Failures are fixed in the implementation, never by changing a scientific
definition.

---

## 6. The two structural vectors and the three conditions

**U3 (LOCAL3's input).** For each root `v` and atom type `t ∈ 0..15`:

```
U3[v,t] = #{ u : shortest_path_distance(v,u) = 3 and atom_type(u) = t }
```

Distinct end *atoms*, not 3-step walks. Endpoints at distance 1 or 2 are
excluded, atoms in another connected component (distance = -1 / unreachable)
are excluded, and `v` itself is never counted. Fixed structural
pre-processing; no learned state is propagated.

**U2 (DUP2's input).**

```
U2[v,t] = #{ S2 path occurrences rooted at v with path_end63 // 5 == t }
```

i.e. the histogram of the existing S2 *end* marginal, keeping the S2
duplicate-endpoint and triangle semantics (a root with two paths to the same
end atom counts 2, a triangle contributes its closure occurrences). It is a
deterministic function of the original S2 end contingency table: an
existing-information re-read with the same capacity. It is *not* a clean
"capacity-only" control and is never described as one.

Both vectors: `log1p` of the raw counts, then per-column standardisation
using the 3360 Train molecules only (all their atoms), `ddof = 0`,
zero-variance columns get scale 1. Dimension fixed at 16. NULL3 uses exactly
LOCAL3's U3 normalisation parameters. Normalisation never reads labels;
structure may be pre-computed for all 4200 molecules, fitting uses Train only.

**Correction operator.** All conditions share one parameterisation:

```
psi(h,s) = W2 * SiLU(W_h h + W_s s + b1) + b2
h 64 → hidden 32 → 64,  s = 16      (W_h 64×32, W_s 16×32, b1 32, W2 32×64, b2 64)
```

4,704 parameters. `W_h/W_s/b1` come from one fixed initialisation (seed 1) and
are copied tensor-by-tensor to all conditions; `W2` and `b2` are exactly zero,
so every condition starts as an exact copy of C1. Insertion point: right
after the original local `LayerNorm` produces `h`, and **before** the original
single dropout. `h' = h + psi(h, s3_v)` (LOCAL3) or `psi(h, s2_v)` (DUP2). The
original `f` and `rho` read the same `h'` after the original dropout; `g` and
all original statistics are unchanged. Base and corrector train end-to-end
(no freezing, no extra loss). Total parameters must be
228674 + 4704 = **233378**.

| arm | correction |
|---|---|
| LOCAL3 | `h' = h + psi(h, s3_v)` — per-root typed 3-hop counts |
| DUP2 | `h' = h + psi(h, s2_v)` — S2 end-marginal counts (existing information) |
| NULL3 | `h' = h + mean_{u ∈ C(v)} psi(h_v, s3_u)` — see below (Stage II only) |

**NULL3 (only bought if §9 passes).** `C(v)` = all atoms of the *same
molecule* with `atom_type` equal to `v`'s, including `v` itself. The
correction keeps the full multiset of `s3` vectors inside each
(molecule × atom_type) cell and removes *which* `s3` belongs to which root. It
uses fixed structural statistics only; it never reads other atoms' learned
`h_u`. It must **not** be replaced by the mean-vector shortcut
`psi(h, mean_u s3_u)`; SiLU-then-average ≠ average-then-SiLU. The exact
implementation is

```
W2 * mean_{u ∈ C(v)} SiLU(W_h h_v + W_s s3_u + b1) + b2
```

grouped by (molecule, atom_type), computed in chunks, no random sampling, no
numbering-dependent permutation, no cross-molecule mixing. Degenerate cells
(|C(v)| = 1, or all `s3` identical) make NULL3 identical to LOCAL3; the
fraction of such roots is reported. NULL3 keeps the 3-hop *distribution* and
may therefore separate some of the five pairs itself — this is not forced to
be otherwise. A naive double loop over the same cells is kept as a reference
implementation and both forward and gradient must agree.

Interpretation of the three comparisons:
LOCAL3 vs A = is the new input useful at all; LOCAL3 vs DUP2 = does it exceed
the extra read/parameterisation of information already present;
LOCAL3 vs NULL3 = is the per-root correspondence needed, or is a typed 3-hop
distribution sufficient.

---

## 7. Initialisation, batch order, recipe (frozen)

* A is **reused**, not retrained: `lcrc_lipo_edit_supervision_public122_s0_v1/runs/A_s0/best.pt`
  (228674 params, `best_epoch = 244`, Valid 0.7025358558, Test 0.5761674772).
* New conditions start from the frozen
  `build_full_state(spec, train_norm, seed=0)` → `remap_state("C1")`; the
  initial base state hash must again be `aac1e061…`. Never initialised from
  the *trained* A weights.
* One corrector initialisation, copied tensor-by-tensor to all conditions;
  after module creation the training CPU/CUDA RNG is reset to 0.
* Zero correction must reproduce C1 bit-identically.
* Molecule shuffle generator seed 917000, identical ordinary batch sequence
  for all conditions (verified against the baseline `schedule_manifest.json`,
  first epoch = 27 steps, sha256 `b10278f960300fef…`); epoch-1 dropout RNG
  compared against the baseline trace. The new branch has no dropout; the
  original dropout is still called exactly once per forward.

Fixed recipe R3 (unchanged): Train 3360, batch 128, `drop_last = false`,
360 full epochs, 27 steps/epoch, 9720 updates; AdamW β = (0.9, 0.999),
eps 1e-8, wd 1e-3; per-step cosine LR 3e-4 → 1e-5, `T_max = 9720`, same call
order as before; FP32, no AMP, grad clip 5.0, plain standardised molecule MSE;
no early stop, no soup/EMA/SWA, no pair loss, no case reweighting.

Per epoch, full Valid 420 raw-unit RMSE must *strictly* improve to save the
best checkpoint; ties keep the earlier epoch; Test never selects an epoch.
Best and last are saved. Reloading best and recomputing Valid must agree with
the curve's argmin and the recorded value to 1e-5.

Numerical non-determinism still exists; synchronised RNG is not bit-level
training determinism and is not reported as such.

---

## 8. Evaluation (fixed before running)

Primary performance metric: full Test 420 molecule RMSE in raw logD. Train,
Valid and Test RMSE/MAE, `best_epoch`, parameter count and GPU·h are reported
alongside. `gain = metric(reference) - metric(candidate)`, positive = candidate
better.

**Paired uncertainty.** Label-free evaluation *components* are built by
union-find from the previous edit-supervision round's per-molecule
`eval_component` merged with this round's "identical full C1 input" molecule
groups. Components are used for resampling only, never for training sampling.
Per split, 1000 paired bootstrap replicates over components containing that
split's rows (`numpy.random.default_rng(20261114)`), keeping all rows inside a
component and re-weighting to the original molecule weights, shared across
models and comparisons. Intervals are conditional on the trained models and
say nothing about training-seed repeatability; crossing zero is not
equivalence.

**Fixed sensitivities.**
* remove A's single largest Test squared-error row, recompute the three Test
  comparisons;
* remove A's single largest Valid squared-error row, recompute the Valid
  comparison;
* re-evaluate the primary metric at the shared last epoch 360 (training
  progress / checkpoint-choice check only, never to re-select a checkpoint).

**Mechanism cases.** All five pairs, per model: both endpoints' predictions,
prediction difference, label difference, difference error, split. For each
identical-signature root class, show how the U3 vector multiset changes and
give the real 3-bond shortest paths / atom labels responsible; when repeated
roots or automorphisms make the mapping ambiguous, aggregate at the signature
group and do not fabricate a unique atom correspondence. Empirical minimum
SSE of an identical-input group, `Σ_g Σ_i (y_i - mean_g y)²`, may be reported
as a description of the fitting constraint this sample imposes — not as
irreducible noise.

**Computational interventions** on the trained LOCAL3 weights only:
(a) switch the correction branch off; (b) replace the LOCAL3 operator with the
exact NULL3 mean-field operator while keeping LOCAL3's own `s3` normalisation
and all its weights. Report prediction movement, full Valid/Test metrics and
the five pairs' responses. These are *model-computation* interventions, not
chemical edits or causal effects of the trained model; the primary mechanism
comparison remains the independently trained matched control. If LOCAL3's
input separates a case but training produces almost no response, record
`FEATURE_AVAILABLE_NOT_USED` and do not claim a repair.

Also recorded: correction-vector norms, gradient norms of the output/feature
projections, and the fraction of non-degenerate NULL3 roots. No automatic
strengthening of the correction, no auxiliary loss, no wider hidden layer.

---

## 9. Stage II purchase rule (fixed budget filter)

After Stage I, evaluated in raw units:

1. LOCAL3 Test RMSE point estimate < A **and** < DUP2;
2. LOCAL3 Valid RMSE point estimate < A;
3. after removing A's largest Test squared-error molecule, LOCAL3's Test RMSE
   gain stays positive versus **both** A and DUP2.

All three must hold to buy NULL3; otherwise record
`NOT_PURCHASED_BY_FIXED_RULE` and do not present the unrun condition as a zero
effect. This is a budget filter, not a significance test: no effect-size
threshold, and no configuration is changed in response to Test.

## 10. Pre-registered interpretation labels

| condition | label |
|---|---|
| five pairs reproduce (non-isomorphic, same input, same A prediction) | `C1_INPUT_ALIAS_VERIFIED` |
| U3 separates exactly four of five | `S3_INPUT_SEPARATION_VERIFIED` |
| Valid/Test disagree, CI crosses zero, or the fixed influence check flips sign | `PERFORMANCE_UNRESOLVED_THIS_RUN` |
| LOCAL3 not better than A | `NO_PREDICTIVE_GAIN_THIS_CONFIG` (stop; no S4 attempt) |
| LOCAL3 better than A but not than DUP2 | `NEW_INFORMATION_INCREMENT_UNRESOLVED` |
| LOCAL3 gains positive vs A and DUP2, both CI lower bounds > 0, Valid same direction, influence checks not flipped | `S3_PREDICTIVE_SIGNAL_THIS_RUN` |
| NULL3 trained, LOCAL3 gain vs NULL3 positive with CI lower bound > 0, Valid same direction, influence not flipped | `+ ROOT_ALIGNMENT_SIGNAL_THIS_RUN` |
| NULL3 trained, otherwise | `ROOT_ALIGNMENT_UNRESOLVED` |
| data/implementation/budget failure | the matching execution status; no validity claim |

`C1_INPUT_ALIAS_VERIFIED`/`S3_INPUT_SEPARATION_VERIFIED` support only the
representation boundary of this C1/`lipo_v1` configuration and do not
generalise to all LCRC/PCQM models. `ROOT_ALIGNMENT_SIGNAL_THIS_RUN` supports
only "better than this mean-field control here"; it neither proves per-root
binding is generally necessary nor fully excludes optimisation/regularisation
differences in the averaging operator. If NULL3 is close to or better than
LOCAL3, report "the 3-hop distribution may be sufficient; a per-root increment
is not established" and do **not** call the two equivalent.

---

## 11. Execution and budget

Local: P0, features, tests, then a 2-epoch GPU smoke for LOCAL3 and DUP2 (plus
a small-batch NULL3 optimised-vs-naive forward/gradient check) to measure
throughput. Smoke models never enter the formal results. Shared caches are
built once before the parallel trainings.

Remote via the `rr` CLI only (`rr doctor / deploy / run / jobs / status /
logs / pull`), host `res`, process backend, `res-gpu0` for LOCAL3 and
`res-gpu1` for DUP2, at most one round job per card, one parallel round. No
manual ssh+nohup, no `res-2`. Formal runs must map to a committed revision and
a valid deployment (no `--allow-dirty`, no stale deployment). Stage II, if
bought, runs alone on a free `res-gpu0`.

Budget: target ≤ 1.0 A100 GPU·h, hard cap 2.0 GPU·h including smoke,
inference, failures and retries, summed over all CUDA processes' wall time. If
the smoke shows the fixed comparison cannot finish inside the cap, output
`BUDGET_LIMITED` rather than shrinking epochs, dropping controls or changing
the recipe. One same-configuration infrastructure retry is allowed; a
scientific code bug requires rerunning and re-charging every affected
condition.

## 12. What this round cannot support

Even with a positive performance result: no unique atom contributions, no real
chemical causality, no cross-seed or cross-dataset confirmation, no claim that
5 pairs represent the population, no claim that the correction is the only
way to use 3-hop information, and no claim of superiority over all GNNs. With
only `C1_INPUT_ALIAS_VERIFIED` + `S3_INPUT_SEPARATION_VERIFIED` and no
predictive gain, the boundary result is kept as the round's output and this
configuration ends.

---

## 13. P0 observed support (appended 2026-11-13, before any formal GPU run)

CPU only; `code/prepare.py`, `code/p0.py`, `tests/`. Every number below is
produced by the committed code from the frozen sources.

**Data/cache identity.** Rebuilt caches match the reused baseline's content
fingerprints exactly: Train `d9a41ef2…`, Valid `f1ec6415…`, Test `f8bdb280…`;
Train-only norm fingerprint `4e62dae7…`; atom counts 90863/11465/11240;
batch-order sha256 `b10278f960300fef…` equal to the baseline manifest.

**Baseline A identity and replication.** `best_epoch = 244`, Valid
0.702535894, Test 0.576167455 on CPU; per-molecule row-id join max |Δ| 2.6e-06
and design-note prediction max |Δ| 8.6e-07 (tolerance 2e-5). 228674
parameters; init state hash prefix `aac1e061`.

**Signature and group enumeration.** 4200 molecules, 113568 atoms, 18122
distinct root signatures, 4097 distinct per-molecule signature multisets.
87 local-signature groups (72×2, 14×3, 1×4 atoms sets) → 86 full-input groups
after `global42`; 188 molecules in full-input groups. Of the 102 group members
compared against their representatives, 97 are raw-attribute-graph isomorphic
(stereo writing) and 5 are connectivity/attribute non-isomorphic. One extra
local-only group (837, 1200) is separated by `global42` and is therefore *not*
a full-input alias; the number of additional full-input alias pairs beyond the
five frozen ones is 0. Empirical minimum SSE summed over the 86 full-input
groups: 3.304825.

**The five pairs.** 5/5 raw-attribute-graph non-isomorphic, 5/5 equal local
signature multiset (block-by-block root correspondence verified on all roots:
32/25/31/37/31 per pair), 5/5 equal `global42`, 5/5 A predictions equal within
2e-5. U3 separates exactly
4/5 with 1467/2674 still identical; U2 separates 0/5. Forward identity: roots
with identical signatures have identical local-input and `h` multisets (max
|Δ| ≤ 1.2e-06 from float32 reduction order).

**Model plumbing.** 233378 parameters (228674 base + 4704 corrector); shared
base init hash `aac1e061…`; zero correction bit-identical to C1 (max |Δ| = 0);
NULL3 optimised vs naive double loop: forward |Δ| 3e-07, gradient |Δ| ~1e-05
(values scale ~50), and the mean-then-SiLU shortcut differs by 0.288, so the
operator is not the shortcut.

**Status.** P0 gates pass; no formal training had been run when this section
was written. Formal results are recorded in `REPORT.md` only.

---

## 14. Implementation mapping

| file | role |
|---|---|
| `code/common.py` | frozen constants, source verification by hash, provenance/IO |
| `code/prepare.py` | structure caches, U3/U2 features, batch order, manifests |
| `code/signature.py` | six-block per-root signature, canonical bytes, equality, audit |
| `code/features.py` | U3/U2 counts, Train-only normalisation, NULL3 degeneracy stats |
| `code/p0.py` | alias audit, group enumeration, A replication, forward identity |
| `code/data.py` | dataset/collate/prediction helpers with s3/s2/u3/u2 tensors |
| `code/model.py` | `S3Corrector`, hook insertion, NULL3 mean field, arm construction |
| `code/schedule.py` | ordinary shuffle order, baseline manifest equality |
| `code/train.py` | one arm per job: frozen init, R3 recipe, best-Valid checkpointing |
| `code/metrics.py` | NumPy metrics, components, paired component bootstrap |
| `code/evaluate.py` | `results.csv`, `comparisons.json`, sensitivities, labels, cases |
| `code/intervene.py` | branch-off / NULL3-operator interventions on trained LOCAL3 |
| `tests/test_s3_locality.py` | the §"key checks" list of the round |

Config: `configs/experiment.yaml` (mirrors `code/common.py`, which is
authoritative).

Reproduce: `uv run --group lipo python code/prepare.py`,
`uv run --group lipo python code/p0.py`,
`uv run python -m pytest -q tests/`, then `rr` for the GPU stages.
