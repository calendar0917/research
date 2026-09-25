# E2E-DictEnv-A2 — pre-registration (frozen before implementation)

Round **E2E-DictEnv-A2** · protocol `e2e_dictenv_a2` · subtitle **Attributed Code
Value & Compression Control** · study `zinc-context-gap` · branch
`exp/e2e-dictenv-a2-code-value-compression`.

Prior-artifact audit:
[`e2e_dictenv_a2_prior_artifact_audit.md`](e2e_dictenv_a2_prior_artifact_audit.md)
(committed immediately before this file).

This file is the frozen specification.  It is committed **before** any A2
implementation, cache access, dictionary refit, training or GPU use.  Any
deviation requires a numbered amendment written here before the affected run.

---

## 0. Round status, scope and relation to A1

A1 (`e2e_dictenv_a1`) is closed with the frozen verdict
`REPRESENTATION_NOT_QUALIFIED` because its pre-registered binary Gate 0.3
(`REAL` code-space continuity AUC ≥ 0.70) measured 0.534375.  A1's own
diagnostics show that the criterion, not the object, is degenerate on the frozen
tiny-patch population (the criterion also failed on the unrelated TCCD-v0 object
at 0.4675), and A1 therefore **never answered** its primary question:
Stage 1–4 were not run.

**A2 is not an A1 rescue.**  It does not re-run, reinterpret, relax or repair
A1's gate.  It re-asks A1's unanswered question under a new pre-registration, and
adds the control that separates two hypotheses that A1 could only confound:

> **Q1.** Does the real structure↔attribute pairing inside dictionary-code
> formation carry **task value** at all?
>
> **Q2.** If it does, does the frozen `K=32, s=8` sparse dictionary
> **preserve** it, or does the sparse bottleneck destroy it?

Order of operations is frozen and must be respected:
**task value (Stage 1) → compression sufficiency (Stage 2) → sparse specificity
(Stages 3–5)**, with continuity reported as a *diagnostic* rather than a
round-stopping prerequisite.

Not a SOTA attempt, not a capacity search, not a decoder/optimizer search.  The
only new objects in the round are (i) a rank-32 dense compression reference and
(ii) the pair population of the continuity diagnostic; the dictionary input
coordinates, dictionaries, scaler, coder family and downstream model are reused
bit-identically from A1.

**Device discipline (hard).**  GPU0 is foreign-occupied and is **never** touched,
never co-tenanted, never used for a smoke, diagnostic or formal run.  Every CUDA
operation in this round runs on **physical GPU1** through
`bash scripts/run_remote.sh 1 …` (equivalently `launch_remote.sh 1 …`), i.e. with
`CUDA_VISIBLE_DEVICES=1`, and the runner refuses any other value.  No DDP, no
multi-GPU, no GPU0/GPU1 parallelism, one CUDA process at a time.  If GPU1 is
unavailable, CUDA fails, the remote checkout is dirty, ZINC is missing or
`research doctor` fails: **stop and report**, never fall back to GPU0.

---

## 1. Hypotheses

* **H1 (primary).** Under a frozen dictionary and an exact `s = 8` OMP code, the
  `REAL` 433-D attributed object yields a lower official-valid MAE than its
  parameter-identical, assignment-independent `INDEP` analytic null, by at
  least the material threshold.
* **H2 (secondary).** The same `REAL` arm also beats the chemistry-blind `TOPO`
  arm.
* **H3 (compression).** If H1 fails, the same contrast under a *dense* rank-32
  projection of the same two objects either does or does not recover the
  material gain.  Recovery means the bottleneck is the sparse compression;
  non-recovery means there is no attributed-code-formation signal at this
  32-D/H1 resolution.
* **H4 (mechanism).** A trained `REAL` sparse arm degrades materially when the
  attribute→code pairing is removed at inference (node block `J^V → P^V`, edge
  block `J^E → P^E`, downstream chemistry untouched) while the same trained
  dictionary and all other inputs are held fixed.
* **H5 (specificity).** Conditional on H1-surrogate (E2E) and H4 passing, the
  sparse tied dictionary beats the parameter-matched dense tied coordinate on
  the same `REAL` object.

---

## 2. Frozen objects — no redesign

A2 reuses, bit-identically and without re-derivation:

```
phi65                      (FSAR-R2 explicit pure-topology coordinate)
chi_REAL  = [phi65 ; vec(JV) ; vec(JE)]          433-D
chi_INDEP = [phi65 ; vec(PV) ; vec(PE)]          433-D
train-only RMS scaler + zero-RMS masks + block-energy normalisation
A1 K-SVD dictionaries (TOPO 65x32 reuse, INDEP 433x32, REAL 433x32)
A1 exact-OMP code infrastructure
```

with `J^V_i = Σ_{v∈P_i} b^V_{iv} q_vᵀ` (`b^V ∈ R^11`, `q_v ∈ R^28`), `J^E_i =
Σ_{e∈P_i} b^E_{ie} r_eᵀ` (`b^E ∈ R^15`, `r_e ∈ R^4`), and
`P^V_i = (1/n_i)(Σ_v b^V_{iv})(Σ_v q_v)ᵀ`, `P^E_i = (1/m_i)(Σ_e b^E_{ie})(Σ_e r_e)ᵀ`
(exactly zero when `m_i = 0`), all defined and audited in A1.

**Frozen and forbidden to change**: primitives, radius (2), atom categories (28),
bond categories (4), presence/absence of rings or motifs, `patch_cont`,
learned embeddings, block weights, scaler procedure, atom/bond block content.

### 2.1 Artifact-identity gate (blocking)

Before any GPU work, A2 writes `artifact_identity.json` recording, per reused
artifact: `source_path`, `expected_value` (and where it is pinned),
`observed_sha256`, `identity PASS/FAIL`.  The expected values are exactly those
of the audit §4: the TOPO/INDEP/REAL dictionary digests (pinned in tracked
files), the per-tensor raw-cache digests recorded in A1's `cache_meta_*.json`
with the tracked prefixes from `notes/e2e_dictenv_a1_analysis.md`, and — for the
scaler and the six OMP code tensors, which have no recorded digest — an exact
deterministic recomputation (`a1.fit_object_scaler` on the raw train cache;
`sdb.omp_codes(normalize_columns(D_arm), X_arm, s=8)` on the full split) that
must reproduce the stored tensor bitwise.

Any FAIL stops the round with verdict `ARTIFACT_IDENTITY_FAILURE`; no
substitution, no refit, no partial reuse is permitted.  A deterministic rebuild
of a missing raw cache is allowed **only** if the rebuilt tensor matches the
expected digest; otherwise the round stops.

### 2.2 Parameter budget

Inherited unchanged: `K = 32`, `s = 8`, H1 decoder, `d_e = 48`, horizon 320,
`lambda_rec = 33.95873017865987`, budget 80 000–130 000.
`TOPO = 97 487`, `INDEP = REAL = 109 263` (exactly parameter-matched).

---

## 3. Continuity-v2 — a diagnostic, never a gate

Continuity-v2 is computed once, on CPU, from the identity-verified artifacts.
It is reported in `continuity_v2.json` and **cannot stop the round** and cannot
select anything.  Its only purpose is to describe the geometry of the three
coordinates on a population that is now correctly conditioned.

**Frozen definition.**

* **Pool**: 2 000 roots sampled uniformly from official-train molecules (one
  draw per distinct `(molecule, root)`, deterministic seed `20260930`); the
  patch's size is the cache's own `n_patch`.
* **WL similarity**: `attributed_wl_fingerprint(graph, node_types, edge_types,
  rounds=3)` with the root recoloured (`+1000`), L2-normalised histogram over the
  pool vocabulary, cosine similarity (the A1/TCCD definition, reused verbatim).
* **Pair population**: all pool pairs `i < j` filtered by
  (a) **equal patch size**,
  (b) **different molecule** (molecule-disjoint),
  (c) **different rooted typed-WL canonical key** (`key_i != key_j`, i.e. not
  isomorphic);
  then deterministically permuted (seed `20260930`) and capped at **4 000 pairs
  per stratum**; actual counts are reported.  A pooled row over all eligible
  equal-size molecule-disjoint pairs is reported as well.
* **Strata** (fixed bins from the official-train patch-size distribution,
  audit §7): `small = n_patch ≤ 5`, `medium = 6..8`, `large = ≥ 9`.
* **Spaces**: x-space (65-D for TOPO, 433-D for INDEP/REAL) and exact-OMP code
  space (32-D), per arm.
* **Primary diagnostic**: `rho_arm(space, stratum) =
  Spearman(attributed_WL_similarity, −angular_distance)` where
  `angular_distance(a,b) = arccos(clip(cos(a,b), −1, 1))` and Spearman is
  Pearson on ranks (`np.corrcoef(argsort(argsort(.)))`, the A1 implementation).
  The true Euclidean distance is reported as a secondary column.
* **Reported contrasts**: `rho_REAL − rho_INDEP`, `rho_REAL − rho_TOPO`,
  `rho_INDEP − rho_TOPO`, per space and stratum.
* **Reconciliation row** (descriptive only): the same statistic recomputed on
  A1's pool (1 500, seed `20260933`) with the audit §3 populations, so that the
  A1 numbers are explained rather than reused.

No A1 continuity number is used as a baseline, a threshold or a calibration
target.  There is **no** binary AUC gate in A2 and **no** `≥ 10`-atom tail
restriction.

---

## 4. Stage 1 — frozen exact-OMP attributed-code value screen (primary)

The first and most important experiment: are the attributes doing anything for
the *task* once they enter the code, with the dictionary and the code frozen so
that nothing about coding can confound the comparison?

```
T0:  phi65        -> frozen topology D32 -> exact OMP s=8 -> alpha32 -> H1
I0:  chi_INDEP433 -> frozen INDEP D32    -> exact OMP s=8 -> alpha32 -> H1
R0:  chi_REAL433  -> frozen REAL D32     -> exact OMP s=8 -> alpha32 -> H1
```

* Dictionary and codes are **frozen** (`freeze_dictionary = True`, precomputed
  exact-OMP codes attached).  Only the genuinely trainable downstream parameters
  (`W_A_*`, `W_E_*`, H1 slot encoders, fusion, pair/global/topology encoders,
  reader) receive gradients.  The reconstruction term is a constant in this
  stage and cannot influence training; it is still computed and reported.
* Downstream is the frozen P2-ABS **H1** winner, unchanged in every respect:
  decoder `h1`, `d_e = 48`, horizon 320, Adam `lr 1e-3`, `wd 1e-5`, batch 128,
  gradient clip 5.0, no scheduler, no early stop, deterministic algorithms,
  fixed equal-weight Top-5 official-valid soup, seed 0.  No sweep.
* Selection metric: official-valid Top-5 soup MAE.  Official train is used only
  for fitting the (already frozen) objects and codes; official valid is used for
  the pre-registered soup selection; the official test split is **never loaded**.

**Primary quantity.**

```
G_pair_OMP = MAE(I0) - MAE(R0)          (positive = REAL better)
gate       = 0.003                      (material)
```

**Secondary quantity.**

```
G_topo_OMP = MAE(T0) - MAE(R0)   >= 0.003
```

Precedence is frozen: `REAL` vs `INDEP` (a strictly matched assignment contrast:
same width, same parameter count, same interface) dominates `REAL` vs `TOPO`
(different width and parameter count, reported as context).  The historical
H1 anchor `0.12354862861608853` and any earlier TOPO number are context only and
are never used as a baseline: A2 re-establishes a matched TOPO arm in the same
execution regime.

Artifacts: `omp_screen_topo.json`, `omp_screen_indep.json`,
`omp_screen_real.json`, `omp_decision.json`.

---

## 5. Stage 2 — dense rank-32 compression control (conditional)

Runs **only if** `G_pair_OMP < 0.003` (Case O2).  It exists to separate
"pairing has no value" from "pairing has value that the `K=32`, `s=8` sparse
code fails to preserve", which A1's own reconstruction numbers make a live
possibility (REAL OMP holdout 1.97e-01 vs INDEP 2.11e-02 at identical settings).

For each of the two 433-D objects, on the **same** normalised object that Stage 1
uses:

* fit the already-audited train-only affine rank-32 PCA
  (`sdb_v0.fit_pca_rank`, `PCARank.dense_codes(X) = (X − mean) @ Vᵀ`), on
  official train only; official valid is transformed with the frozen components
  and never influences the fit; the official test split is never loaded;
* the projection is then **frozen** (the model's `D` is set to `Vᵀ`, whose
  columns are unit-norm, and the centred codes are attached as precomputed
  coordinates, exactly as stage 1 attaches frozen OMP codes).

Arms:

```
I-D = ATTR-INDEP-PCA32        R-D = ATTR-REAL-PCA32
```

Requirements: identical output width 32, identical H1, identical training
protocol, identical parameter count (109 263 total, identical visible trainable
set), identical seed, identical batch order, identical GPU1, same commit.
Stage 2 is *not* a second Stage 1 and *not* a trained dense variant: the
compression map is fixed and label-free, so the only variable is the object.

**Primary quantity.**

```
G_pair_PCA = MAE(I-D) - MAE(R-D)  >= 0.003
```

Artifacts: `pca_meta_indep.json`, `pca_meta_real.json`, `pca_screen_indep.json`,
`pca_screen_real.json`, `compression_decision.json`.

The 2×2 (rows = compression, columns = object) is reported explicitly:

| compression | INDEP | REAL |
|---|---|---|
| sparse `K32/s8` exact OMP (frozen) | `I0` | `R0` |
| dense rank-32 PCA (frozen) | `I-D` | `R-D` |

Case C1 (sparse positive, dense not triggered) → proceed on the sparse route.
Case C2 (`G_pair_PCA ≥ 0.003`, `G_pair_OMP < 0.003`) → **stop the round** with
`ATTRIBUTED_PAIRING_SUPPORTED_COMPRESSION_BOTTLENECK`.  Case C3 (both negative)
→ **stop** with `NO_ATTRIBUTED_CODE_FORMATION_SIGNAL`.  Case C4 (sparse
positive, dense negative) is structurally unreachable in this design because
Stage 2 is not triggered when Stage 1 passes; it is recorded as `NOT RUN` with
that reason, and the sparse route's specificity question is instead answered by
Stage 5, which is the sharper instrument (both arms trained, matched
parameterisation).

**Within A2 it is forbidden** to respond to a bottleneck verdict by trying
`K = 64`, `s = 12`, `K = 128`, another dictionary or another sparsity.  That is a
new dictionary-capacity pre-registration, not an A2 rescue.

---

## 6. Stage 3 — IHT coder qualification (conditional on the sparse route)

Runs only when Stage 1 passed (Case O1).  Candidate step counts, frozen:
`{10, 30, 100, 200}`.

For each of `TOPO`, `INDEP`, `REAL`, on a deterministic official-train row subset
(50 000 rows, seed 20260924), compare the tied-IHT code against the exact-OMP
code on the frozen dictionary and record: normalised reconstruction error,
support overlap, support Jaccard, exact sparsity (`mean l0`), relative code
error, runtime.

**Frozen selection rule**: choose the **smallest** step count that satisfies

```
normalised reconstruction error <= 0.002   for ALL THREE arms
```

If no candidate satisfies it, the round stops with `CODER_NOT_QUALIFIED`.  No
extension to 500/1000 steps, no LISTA, no ISTA variant, **no per-arm step
count**, no per-arm λ.

Artifacts: `iht_diagnostic.json`, `coder_decision.json`.

---

## 7. Stage 4 — matched end-to-end sparse attributed dictionary (conditional)

Runs only when Stage 1 passed **and** Stage 3 qualified.  All three arms use the
**same** selected IHT step count, the same `K = 32`, `s = 8`, the same H1, the
same horizon 320, the same optimizer/schedule, the same λ (below), the same
batch order and the same GPU1/commit, with the dictionary trainable (tied) as in
P1/P2-ABS:

```
E0 = TOPO-SPARSE        E1 = ATTR-INDEP-SPARSE        E2 = ATTR-REAL-SPARSE
```

**Primary.**

```
G_pair_E2E = MAE(E1) - MAE(E2)  >= 0.003
```

**Secondary.**

```
G_topo_E2E = MAE(E0) - MAE(E2)  >= 0.003
```

Artifacts: `formal_runs/` (per-arm JSON + curve + soup state),
`formal_decision.json`.

### 7.1 Reconstruction weight λ — frozen, shared, not recalibrated

```
lambda_rec = 33.95873017865987     (P2-ABS Stage-A winner, inherited by A1)
reconstruction loss = mean_i ||x_i - Dbar alpha_i||^2 / (||x_i||^2 + 1e-12)
```

The **same scalar** is used for every arm, every stage, unchanged.  There is
**no** arm-specific λ, **no** recalibration from `rec_init`, and **no** use of
the IHT initial reconstruction error to set a weight.  (A1/P2's lesson: the
`lambda_base` recalibration converted a coder deficit into a weighting change.)
The reported reconstruction error is normalised per row, so it is comparable
across arms even though the 433-D objects have roughly three times the energy of
`phi65`.

---

## 8. Mechanism intervention (inference only, on the trained REAL arm)

Required whenever the `E2` formal run completes, **independent of** whether
`G_pair_E2E` passed.

Normal: `chi_REAL → trained REAL dictionary → alpha_REAL → H1`.
Intervention: the same trained REAL dictionary, the same REAL scaler
(scale/mask/block weight) and the same `phi65`, but the node block is
`J^V → P^V` and the edge block is `J^E → P^E`, i.e. the pairing is removed from
the *dictionary input* while all downstream chemistry
(post-code atom chemistry, post-code bond chemistry, anchor, relation, reader)
stays **real and unchanged**.  `alpha` is recomputed from the intervened input
and the prediction is recomputed on official valid.

**Primary mechanism quantity.**

```
G_code_pairing = MAE(pairing_removed) - MAE(real)   >= 0.010   (material)
```

Reported additionally: MAE delta, mean / median / p90 / max absolute prediction
shift, and the fraction of molecules shifted by ≥ 0.01 / 0.05 / 0.10.

**Descriptive decompositions** (reported, never used to change the architecture
or to weaken the primary gate): node pairing removed only (`J^V → P^V`, edge
block real) and edge pairing removed only (`J^E → P^E`, node block real), written
to `mechanism_node_only.json` and `mechanism_edge_only.json`.

The inherited P1 interventions on the trained REAL arm (zero-code, node-
assignment shuffle, edge-shuffle, all-shuffle) are re-run for comparability and
recorded as descriptive mechanism evidence.

Artifact: `mechanism_code_pairing.json`.

---

## 9. Liveness / dictionary-health audit of every formal arm

For every formally trained arm (E0, E1, E2, and the Stage-5 arm if run) record:
`||dL/dD||` at a fixed probe batch, dictionary movement from the K-SVD init
(normalised Frobenius, relative), active atoms, effective atom count, usage
entropy, effective rank of `D`, code variance, code-space std, train↔valid usage
Spearman, and the per-slot decoder gradients (node / edge / anchor).

**Frozen liveness PASS for an arm** requires all of:

```
dictionary_grad_norm > 0
soup_D_vs_ksvd_init.relative > 0.01
active_atoms_train >= 24                       (of 32)
effective_atoms_train >= 8.0
code_variance_mean_train > 0
train_valid_usage_spearman >= 0.5
node / edge / anchor slot gradient norms all > 0
```

If any **formal arm** fails liveness, the round cannot report
`ATTRIBUTED_SPARSE_DICTIONARY_SUPPORTED` or a pairing-value verdict on that
arm's evidence; the verdict becomes `CODE_FORMATION_NOT_LIVE` with the full
numbers reported.  A "live but decorative" code is not a success, and an
attractive MAE produced by a dead branch is not a mechanism result.

Artifacts: `liveness.json`, `dictionary_health.json` (the Gate-0 health audit of
the reused dictionaries), `formal_runs/`.

---

## 10. Stage 5 — sparse specificity (conditional)

Runs **only if** `G_pair_E2E ≥ 0.003` **and** `G_code_pairing ≥ 0.010` **and**
all formal-arm liveness checks pass.

One new arm on the same `REAL` 433-D input: `ATTR-REAL-DENSE-TIED` — the same
H1, the same output width 32, the same horizon/optimizer/seed/batch order, the
same parameter count, the same λ, the only change being the coding operator
(dense tied `phi @ Dbar` instead of tied-IHT top-8), exactly as in the P1
specificity control.

```
G_sparse = MAE(ATTR-REAL-DENSE-TIED) - MAE(ATTR-REAL-SPARSE)  >= 0.003
```

`G_sparse ≥ 0.003` → `ATTRIBUTED_SPARSE_DICTIONARY_SUPPORTED`.
`G_sparse < 0.003` (with pairing supported) →
`ATTRIBUTED_PAIRING_SUPPORTED_DICTIONARY_NOT_SPECIFIC`.

Artifact: `specificity.json`.

---

## 11. Matched initialization and execution

* Same seed 0 for every arm; `torch.use_deterministic_algorithms(True)`;
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
* `a1.build_model` reseeds deterministically per arm, and every tensor whose
  shape matches across arms (all except `D`) receives the **same initial values**
  by construction; A2 additionally copies the arms' shared-shape tensors from a
  single reference state and asserts bit-identity in a test, so "same
  initialization where tensor shapes match" is enforced, not assumed.
* Same batch order: the loaders use the frozen `SEED + TRAIN_SHUFFLE_OFFSET`
  (91011) and `SEED + EVAL_SHUFFLE_OFFSET` (91012) with `num_workers=0`, so the
  epoch permutations are identical across arms.
* Same training schedule, same optimizer hyper-parameters, same Top-5 soup rule,
  same code commit, same physical GPU1, one process at a time.
* Stage 1 arms (frozen codes) and Stage 4 arms (trainable dictionary) are both
  run in the same regime so that "frozen OMP" and "trained sparse" are
  comparable; the historical `0.123548629` H1 soup is context only.

---

## 12. Seeds and budget

* **Seed 0 only** for every stage.  No seed hunting, no seed screening.
* Seed 1 is authorised **only if** `G_pair_E2E ≥ 0.003` at seed 0, and then only
  as a single pre-specified paired replication of `E0`, `E1`, `E2` with the same
  protocol on GPU1, sequential, reported as robustness evidence that cannot
  change the primary verdict.  No further seeds in any case.
* Budget: the round is one durable remote run on GPU1 walking the frozen order.
  There is no per-stage retry budget and no re-tuning.  Stage 2 is not run when
  Stage 1 passes; Stage 3–5 are not run when Stage 1 fails; Stage 5 is not run
  when its two conditions fail.

---

## 13. Official split discipline

```
official train : fitting / dictionary / scaler / PCA / training
official valid : pre-registered evaluation and Top-5 soup selection
official test  : NEVER LOADED
```

The shared P1 loader refuses `split == "test"`; A2 adds an explicit
`official_test_loaded = false` assertion in every artifact and a hard blocker
gate.  The round does not perform a terminal test read, and finishing this round
does not authorise one.

---

## 14. Decision tree (frozen, exhaustive)

Evaluated strictly in this order; the first matching row is the round verdict.

| # | condition | verdict | further action |
|---|---|---|---|
| 0 | any artifact-identity check FAILs | `ARTIFACT_IDENTITY_FAILURE` | stop |
| 1 | correctness / assignment semantics / dictionary health / official-test blocker FAIL; or any formal arm's curve is non-finite | `GATE0_NOT_QUALIFIED` | stop |
| 2 | `G_pair_OMP >= 0.003` (Case O1) | *(continue on the sparse route)* | Stage 2 `NOT RUN` (reason: `stage1_primary_pass`); Stage 3 |
| 2a | sparse route and no IHT step count qualifies | `CODER_NOT_QUALIFIED` | stop |
| 2b | sparse route, coder qualified, any formal arm liveness FAIL | `CODE_FORMATION_NOT_LIVE` | report, stop |
| 2c | sparse route, `G_pair_E2E < 0.003` | `PAIRING_SUPPORTED_AT_FROZEN_OMP` | mechanism + liveness still reported; stop |
| 2d | sparse route, `G_pair_E2E >= 0.003`, `G_code_pairing < 0.010` | `ATTRIBUTED_PAIRING_SUPPORTED_DICTIONARY_NOT_SPECIFIC` (reason `code_pairing_below_threshold`) | stop |
| 2e | sparse route, `G_code_pairing >= 0.010`, `G_sparse >= 0.003` | `ATTRIBUTED_SPARSE_DICTIONARY_SUPPORTED` | — |
| 2f | sparse route, `G_code_pairing >= 0.010`, `G_sparse < 0.003` | `ATTRIBUTED_PAIRING_SUPPORTED_DICTIONARY_NOT_SPECIFIC` (reason `dense_tied_not_worse`) | — |
| 3 | `G_pair_OMP < 0.003` and `G_pair_PCA >= 0.003` (Case C2) | `ATTRIBUTED_PAIRING_SUPPORTED_COMPRESSION_BOTTLENECK` | stop; no K/s experiment inside A2 |
| 4 | `G_pair_OMP < 0.003` and `G_pair_PCA < 0.003` (Case C3) | `NO_ATTRIBUTED_CODE_FORMATION_SIGNAL` | stop; do not enter IHT/E2E |

Labels 0–4 are the pre-registered set of this round; `GATE0_NOT_QUALIFIED`,
`CODE_FORMATION_NOT_LIVE` and `TRAINING_FAILURE` (subsumed by row 1) are the
A2-specific labels added for the qualification/liveness cases that the brief's
verdict list does not name.  No label may be invented after the data are seen,
and no threshold may be moved.

Every stage that did not run is written into `stage_status.json` as
`NOT RUN` with its frozen reason, so no stage is left ambiguous.

---

## 15. The seven questions the final report must answer

1. Were the A1 `REAL`/`INDEP` objects reused bit-identically? (identity table)
2. How does the corrected continuity-v2 behave in each patch-size stratum, in
   x-space and code space, for TOPO/INDEP/REAL? (diagnostic, not the verdict)
3. Frozen exact-OMP: who wins, `REAL` or `INDEP`, and what is `G_pair_OMP`?
4. If OMP did not support `REAL`: what does PCA32 say, and does the round show
   "pairing useful but sparse compression destroys it"?
5. If the sparse route entered E2E: is there a material `REAL` vs `INDEP` gain
   (`G_pair_E2E`)?
6. Does removing only the attribute→dictionary-code pairing materially degrade
   the trained `REAL` model (`G_code_pairing`)?
7. If authorised: does sparsity have independent value over the matched dense
   tied coordinate (`G_sparse`)?

Each answer is `RUN` with numbers or `NOT RUN` with the frozen reason.  The
report may not answer a `NOT RUN` question by inference.

---

## 16. Artifacts (all under `tracks/ksvd/results/e2e_dictenv_a2/`)

```
artifact_identity.json
stage_status.json
correctness.json
assignment_semantics.json
dictionary_health.json            (+ dictionary_health_{topo,indep,real}.json)
parameter_accounting.json
continuity_v2.json
omp_screen_topo.json  omp_screen_indep.json  omp_screen_real.json
omp_decision.json
pca_meta_indep.json  pca_meta_real.json
pca_screen_indep.json  pca_screen_real.json  compression_decision.json
iht_diagnostic.json  coder_decision.json
formal_runs/                      (per-arm run JSON, curve CSV, soup state)
formal_decision.json
mechanism_code_pairing.json  mechanism_node_only.json  mechanism_edge_only.json
liveness.json
specificity.json
decision.json
REPORT.md  DECISION.md
```

Tracked (not gitignored): `REPORT.md`, `DECISION.md`.

---

## 17. Explicit non-goals / forbidden rescues

```
K sweep, s sweep, radius sweep, 433-D feature redesign, new primitive
new/learned atom or bond embedding, ring features, motifs, patch_cont
attention, Transformer, message passing, pair->centre, recurrence
reader/decoder widening, LR / dropout / weight-decay / activation sweep
λ sweep or per-arm λ, per-arm IHT steps, K64/s12/K128 inside A2
dictionary refit, scaler refit, block-weight change
continuity-v2 as a gate, threshold or selection input
seed hunting; a second seed other than the pre-specified conditional pair
official-test access
GPU0 access, DDP, multi-GPU, GPU0/GPU1 parallelism
```

Seeing `REAL` OMP holdout reconstruction `1.97e-01` is **not** a licence to
reopen `K`/`s`; capacity is only discussable in a new pre-registration after a
`ATTRIBUTED_PAIRING_SUPPORTED_COMPRESSION_BOTTLENECK` verdict.

---

## 18. Run discipline

1. Local inspection → this pre-registration → commit → implementation → local
   targeted tests → commit → remote preflight (SSH, clean remote checkout,
   `uv sync --frozen`, `research doctor`, ZINC present, GPU1 visible **and
   usable**; GPU0's foreign processes are recorded but never touched) → deploy
   the committed revision → GPU1 run → pull → local analysis → claims /
   decisions / STATE → commit and push.
2. Every CUDA command is issued as `bash scripts/run_remote.sh 1 …` /
   `launch_remote.sh 1 …` (physical GPU1).  The runner asserts
   `CUDA_VISIBLE_DEVICES == "1"`.
3. The formal round is launched durably (`launch_remote.sh 1 a2-full …`,
   detached so an SSH drop cannot kill it) and then **awaited to completion**
   (`wait_remote.sh a2-full`).  Detaching is only for durability; the round is
   never abandoned, never run in parallel with another CUDA job, and every stage
   is resumable so a relaunch continues rather than restarts.
4. Provenance recorded in every formal artifact: round, protocol, git commit,
   pre-registration commit, branch, hostname, `physical_gpu_requested = 1`, GPU
   model, CUDA, PyTorch, Python, seed, start/end time, wall time, peak GPU
   memory, `official_test_loaded = false`.
5. Stopping conditions that require a report rather than a workaround: GPU1
   unavailable or CUDA failure (never switch to GPU0), dirty remote checkout,
   missing ZINC, failing `research doctor`, `uv sync --frozen` failure, artifact
   identity FAIL, non-finite training, OOM at the frozen configuration.
