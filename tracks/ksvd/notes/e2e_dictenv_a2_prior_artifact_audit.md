# E2E-DictEnv-A2 — prior-artifact audit

Round **E2E-DictEnv-A2** · protocol `e2e_dictenv_a2` · subtitle *Attributed Code
Value & Compression Control* · study `zinc-context-gap` · branch
`exp/e2e-dictenv-a2-code-value-compression`.

Written **before** any A2 code, cache, dictionary fit, training or GPU use, and
committed before the A2 pre-registration.  Its jobs are: (i) fix the A1 closure
facts A2 inherits, (ii) **resolve the provenance of every A1 post-hoc continuity
number** so that A2 never mixes two populations into one statement, (iii) record
the identifier chain that lets A2 prove it reused the A1 artifacts bit-for-bit,
and (iv) list what A2 is *not* free to do.

Audit revision: HEAD `db1aa48` (`docs(e2e-dictenv-a1): correct formal-run wall
clock…`), which is also `main` and `origin/main`.  A1's durable tip `db1aa48`
is an ancestor of this branch's base; `git log main..HEAD` is empty at audit
time.  Nothing below is from memory; every number cites a Git-tracked file or a
gitignored-but-present A1 result artifact whose own provenance is stated.

---

## 0. Method

1. Read `tracks/ksvd/STATE.yaml` (`e2e_dictenv_a1` block), the A1 analysis /
   implementation / pre-registration / prior-artifact-audit notes, A1's
   `DECISION.md` and `REPORT.md`, and the A1 decision record
   `tracks/ksvd/records/decisions/decision-e2e-dictenv-a1-stop-gate0-representation-not-qualified-20260924.yaml`.
2. Read the A1 result JSONs actually present in
   `tracks/ksvd/results/e2e_dictenv_a1/` (they are gitignored, but they are the
   A1 formal-run outputs pulled from the remote and are the direct subject of
   the identity audit in §5).
3. Read the prior rounds named by the brief: P2-ABS (H1), P1, SDB-v0, FEC-D1,
   FSAR-R2-AR0, FSAR-R2-AR0-EDGE (plus TCCD-v0 / FSAB facts already distilled in
   the A1 audit).
4. Read the code that actually produced the A1 numbers
   (`zinc_e2e_dictenv_a1.py`, `e2e_dictenv_a1.py`, `sdb_v0.py`,
   `code/analyze_e2e_dictenv_a1_gate0_stratum.py`) so that each statistic is
   attributed to the exact code path that computed it.
5. Independently recompute the official-train radius-2 patch-size distribution
   and cross-check it against the A1 cache's own `n_patch` field (§9).
6. Convert each fact into an explicit constraint/reuse decision.

---

## 1. Frozen A1 closure

| item | value | source |
|---|---|---|
| round | E2E-DictEnv-A1, protocol `e2e_dictenv_a1`, subtitle *Invariant Attributed Dictionary Core* | A1 prereg |
| branch | `exp/e2e-dictenv-a1-attributed-dictionary` | git |
| preregistration commit | `bd3700a` | A1 prereg / STATE |
| implementation commits | `1ac429f`, `37324eb`, `17017125` | STATE |
| formal-run commit | `37324eb` | STATE |
| formal run tag | `a1-full`, pid 3802491, 1 h 02 m 55 s | A1 DECISION |
| device | NVIDIA A100-SXM4-40GB, **GPU1 only** (`CUDA_VISIBLE_DEVICES=1`); GPU0 foreign-owned all round | A1 DECISION |
| **verdict** | **`REPRESENTATION_NOT_QUALIFIED`** (frozen, not reinterpretable) | A1 `DECISION.md` |
| reason | Gate 0.3 continuity `REAL` code-space AUC 0.534375 < 0.70 | `continuity_audit.json` |
| stages run | only Gate 0 (cache, scaler, dictionaries, OMP codes, correctness, assignment semantics, health, continuity, post-hoc continuity) | A1 DECISION |
| official test | **never loaded** (`official_test_loaded = false` everywhere) | A1 artifacts |

### 1.1 What A1 *did* prove (reusable)

* **Correctness**: 19/19 gates `G0a`…`G0s` PASS on official train/valid.
* **Assignment semantics** PASS: within-patch attribute permutation moves `REAL`
  blocks by up to `3.6243`, leaves `INDEP` blocks exactly invariant (`0.0`),
  leaves the marginals exactly preserved (`drift 0.0`) and `phi` untouched
  (`drift 0.0`).
* **Dictionary health** PASS for all three arms (official train, exact OMP
  `s=8`, 20 % held-out train rows): used atoms 32/32 (TOPO), 32/32 (INDEP),
  31/32 (REAL); effective atoms 4.03 / 18.59 / 19.99; holdout normalised
  reconstruction 1.41e-05 / 2.11e-02 / 1.97e-01; train↔valid usage Spearman
  0.999.
* **Equal construction quality**: `REAL` and `INDEP` are the same width (433),
  the same parameter count (109 263), consume the same scaler *procedure*, and
  present the same downstream interface to H1.  `TOPO` = 97 487.
* **Isomorphism invariance**: isomorphic-control coordinate distance
  4.66e-09 (x) / 1.24e-09 (REAL code), i.e. float32 level.

Therefore **A2 does not redesign the 433-D object set**.

### 1.2 What A1 did *not* prove

Never run: Stage 1 OMP screen, Stage 2 IHT qualification, Stage 3 E2E
`REAL` vs `INDEP`, Stage 4 dense-tied specificity, mechanism, liveness, seed 1.
Hence A1 supports **none** of: "REAL is worse than INDEP", "pairing has no task
value", "the attributed dictionary fails on ZINC".  The round's own analysis
note says the continuity failure is evidence about the *criterion* (it also
failed on the unrelated TCCD-v0 object at 0.4675), not about the object.

---

## 2. The A1 warning A2 must respect: REAL is 3× harder to sparse-compress

A1's own frozen OMP holdout reconstruction (official train, 20 % held-out,
exact top-8):

| arm | dim | OMP rec (holdout) | K-SVD final fit MSE |
|---|---:|---:|---:|
| TOPO | 65 | 1.41e-05 | (reuse of SDB-v0) |
| INDEP | 433 | **2.11e-02** | 0.17009 |
| REAL | 433 | **1.97e-01** | 0.51739 |

At identical `K=32`, `s=8`, identical fit budget (10 K-SVD epochs, seed
20260924, 231 664 train atoms) and identical scaler procedure, the real
structure↔attribute pairing object is ~9.3× harder to reconstruct sparsely than
its assignment-independent analytic null.  Independently, the binary
continuity criterion **rewards forgetting**: `INDEP` (which averages the
pairing away) scores 0.7056 while `REAL` scores 0.5344 and the chemistry-blind
`TOPO` scores 0.4325.

**Consequence for A2 (the round's reason to exist).** Two distinct scientific
questions must be answered separately and in this order:

* **Q1** — does the real structure↔attribute pairing itself carry task value?
* **Q2** — if it does, does the frozen `K=32, s=8` sparse dictionary preserve
  it?

A single "dictionary works / fails" statement is forbidden.

---

## 3. Continuity-statistic provenance (the A1 post-hoc reconciliation)

The brief requires that, before freezing A2, every A1 post-hoc Spearman number
be attributed to an exact population.  `tracks/ksvd/notes/e2e_dictenv_a1_analysis.md`
quoted `REAL ≈ 0.287`, `INDEP ≈ 0.281`, `TOPO ≈ 0.128`; `REPORT.md` quoted
`REAL x/code ≈ 0.317 / 0.259` and `TOPO x ≈ 0.518`.  They are **different
populations of the same computation**, as follows.

### 3.1 The two producing code paths

* **Path R** — `zinc_e2e_dictenv_a1.posthoc_continuity_stage()`
  (implemented in A1's own runner, run inside the A1 formal run at commit
  `37324eb`; output `results/e2e_dictenv_a1/continuity_posthoc.json`).
* **Path D** — `tracks/ksvd/code/analyze_e2e_dictenv_a1_gate0_stratum.py`
  (local post-hoc diagnostic script run *after* the A1 formal run on the pulled
  artifacts; output `results/e2e_dictenv_a1/gate0_stratum_diagnostics.json`,
  `kind = "posthoc (local analysis of pulled frozen artifacts; cannot change the
  verdict)"`).

Both share the *same pool and the same sampled pairs*:

| shared element | value |
|---|---|
| pool | 1500 rooted radius-2 patches sampled from **official train** (`n_patch` roots) |
| pool seed | `20260933` = `sdb.DICT_SEED + a1.CONTINUITY_SEED_OFFSET` (20260924 + 9) |
| attributed-WL fingerprint | `run_wholegraph_canonical_registration_audit.attributed_wl_fingerprint`, 3 rounds, root recoloured (`node_type + 1000`), L2-normalised histogram over the pool vocab; similarity = cosine |
| sampled pairs | `POSTHOC_PAIRS = 200000` i.i.d. index draws from seed `CONTINUITY_SEED + 1` = `20260934`, kept iff `i < j` **and** `keys[i] != keys[j]` (distinct rooted typed-WL canonical key) → **99 102 pairs** |
| same-molecule pairs | **not filtered** (the two paths differ here, see below) |
| equal-size subset of the sample | 19.0168 % of the 99 102 pairs |
| distances | plain Euclidean in x-space (65-D for TOPO, 433-D for INDEP/REAL) and in exact-OMP code space (32-D), per arm |
| Spearman | `np.corrcoef(argsort(argsort(.)))` rank correlation (Pearson on ranks), no tie correction |

### 3.2 The numbers, attributed

| statistic | population | REAL | INDEP | TOPO | appears in |
|---|---|---:|---:|---:|---|
| `spearman_wl_cosine_vs_neg_xdist` (Path R) | all 99 102 distinct-key pairs, **all sizes** | 0.31750 | 0.32730 | 0.51755 | A1 `REPORT.md` (REAL x 0.317, TOPO x 0.518) |
| `spearman_wl_cosine_vs_neg_codedist` (Path R) | all 99 102 pairs, all sizes | 0.25926 | 0.29617 | 0.53437 | A1 `REPORT.md` (REAL code 0.259) |
| `spearman_wl_cosine_vs_neg_xdist_equal_size` (Path R) | equal-size subset of the 99 102 pairs (18 843-like), **same-molecule pairs not filtered** | 0.28728 | 0.28123 | 0.12828 | *(close to, but not the same as, the analysis-note numbers)* |
| `spearman_wl_cosine_vs_neg_codedist_equal_size` (Path R) | same | 0.26019 | 0.35495 | 0.13322 | — |
| `graded/*/spearman_cosine_vs_neg_xdist_equal_size_mol_disjoint` (Path D) | **equal size AND molecule-disjoint**, n = 18 843 | **0.28699** | **0.28085** | **0.12823** | A1 `analysis.md` §3 (REAL 0.287, INDEP 0.281, TOPO 0.128) |
| `graded/*/spearman_cosine_vs_neg_codedist_equal_size_mol_disjoint` (Path D) | same | 0.26007 | 0.35470 | 0.13306 | A1 `analysis.md` §3 (partially) |
| `graded/*/spearman_*_all` (Path D) | all 99 102 pairs (Path D's own copy of the Path-R statistic) | 0.31749 / 0.25925 | 0.32729 / 0.29616 | 0.51749 / 0.53452 | reconciliation only |

**Resolution.** The analysis note's `0.287 / 0.281 / 0.128` are Path D's
*equal-size and molecule-disjoint* x-space Spearmans; the report's
`0.317 / 0.259 / 0.518` are Path R's *all-pairs, all-sizes* x- and code-space
Spearmans.  Both are correct for their own population.  The population
difference is decisive because `TOPO`'s code distance is dominated by patch size
(TOPO code rho 0.534 on all pairs vs 0.133 on equal-size molecule-disjoint
pairs), so mixing them would manufacture a spurious "TOPO tracks WL similarity"
effect.

The written wording in `analysis.md` §3 ("equal-size, molecule-disjoint sampled
pairs") is accurate for Path D and *imprecise* for Path R (Path R filtered only
equal size, not molecule disjointness).  No A1 verdict depends on this: the A1
verdict is the frozen binary Gate 0.3.  A2 therefore treats **all** A1
continuity numbers as *labeled descriptive provenance*, never as a calibration
target, and computes continuity-v2 fresh (§ of the pre-registration).

### 3.3 What A2 may and may not take from this

* **May** use it to design continuity-v2's population: equal-size pairs must
  also be molecule-disjoint, sizes must be stratified, and the graded statistic
  must never be pooled across sizes.
* **May not** reuse any A1 number as a threshold, a baseline, or a "REAL vs
  INDEP" contrast, because no A1 number was computed on A2's pre-registered
  population, and the A1 pool (1500, one draw per (molecule, root), no
  molecule-disjointness constraint) is not A2's pool.
* **May not** re-derive the failed binary criterion as a gate.

---

## 4. Frozen A1 objects A2 reuses (identity chain)

All paths are relative to the repo root.  `tracks/ksvd/results/**` is
gitignored, so the durable part of the chain is: (a) hashes pinned in
**tracked** files (A1 decision record, `STATE.yaml`, the A1 runner's code
constant), and (b) for artifacts with no tracked hash, an exact deterministic
recomputation performed by A2's identity stage.

| artifact | path | durable expected value | where pinned | A2 verification |
|---|---|---|---|---|
| TOPO dictionary `D` (65×32) | `results/sdb_v0/dictionary.pt` (`D_ksvd`) | `b0c5da98aee5795450927fcd76606281aca947448f77ab186e11de09b33dfecd` | tracked code constant `zinc_e2e_dictenv_a1.SDB_DICT_SHA256_F32`; A1 `dictionary_meta_topo.json` | recompute sha256(float32 bytes), compare |
| INDEP dictionary (433×32) | `results/e2e_dictenv_a1/dictionary_indep.pt` | `400821ee5105050eb34600a0fb4cb8040f983daa1e773738dc9e2774036ff32d` | tracked: A1 decision record + `STATE.yaml` | recompute sha, compare |
| REAL dictionary (433×32) | `results/e2e_dictenv_a1/dictionary_real.pt` | `c1cafb086662fb0753d52987fc321b1369d4f586164dde7960a3bed775d4b809` | tracked: A1 decision record + `STATE.yaml` | recompute sha, compare |
| raw train block cache | `results/e2e_dictenv_a1/cache/a1_raw_train.pt` | `phi f8641dab…`, `joint_v 206f4958…`, `joint_e f3559ca8…`, `marginal_v 40651b98…`, `marginal_e 67dee4ca…` (full digests in `cache_meta_train.json`) | A1 run metadata (gitignored) + **tracked** prefixes for `phi`/`joint_v` in `notes/e2e_dictenv_a1_analysis.md` §2 | recompute sha per tensor, compare to `cache_meta_train.json`, confirm tracked prefixes |
| raw valid block cache | `…/cache/a1_raw_valid.pt` | `phi 33db50b8…`, `joint_v 1599e147…`, `joint_e dd77a9ec…`, `marginal_v 95469216…`, `marginal_e 616372a9…` | same | same |
| train-only scalers (REAL, INDEP) | `results/e2e_dictenv_a1/scaler_meta.json` | `fit_split = official train`, `mean_subtracted = false`, per-block `weight` / `masked_coordinates` (REAL S 9, V 86, E 26; INDEP S 9, V 77, E 22) | A1 run metadata + **tracked** note `analysis.md` §2 | recompute every scale/mask/weight from the raw train cache with `a1.fit_object_scaler`, require bitwise equality; assert no valid row influences it |
| exact-OMP codes (6 tensors) | `…/cache/omp_{TOPO,INDEP,REAL}_{train,valid}.pt` | none pinned (file hashes not recorded in A1) | — | deterministic recomputation: `sdb.omp_codes(normalize_columns(D_arm), X_arm, s=8)` on the full split must reproduce the stored tensor exactly |
| A1 P1 environment cache (input to A1's cache build) | `results/e2e_dictenv_p1/cache/env_{train,valid}.pt` | — | — | only needed for the (unplanned) rebuild path |

**Rebuild policy.** A2 reuses the above; it never refits a scaler, never refits
a dictionary, and never re-runs K-SVD.  The deterministic-rebuild code path
(A1's `build_cache` / `fit_scalers` / `fit_dictionaries`) is retained only as a
documented fallback; if it is ever exercised, the rebuilt tensor must match the
same expected digests above, otherwise the round stops with
`ARTIFACT_IDENTITY_FAILURE`.

---

## 5. Constraints inherited from the prior rounds

| round | fact | constraint on A2 |
|---|---|---|
| **P2-ABS** (`notes/e2e_dictenv_p2_abs_analysis.md`) | H1 is the clean dictionary-core winner: official-valid Top-5 soup **0.12354862861608853**, members `[293,309,313,316,317]`, `d_e=48`, `K=32`, `s=8`, horizon 320, `lambda_rec = 33.95873017865987`, 97 487 params. `E64`, `H2`, `K64S8` all lost | reuse H1 unchanged; no decoder/`d_e`/K/s/horizon/λ axis exists in A2; `0.123549` is context, never a selection input |
| **P2-ABS §5** (coder diagnostic) | frozen 10-step tied IHT is ~700× worse than exact OMP on SDB-K32 (0.010084 vs 1.4518e-05) while still hitting exact top-8; the `lambda_base` recalibration then converts a coder deficit into a weighting change (K64 λ=12.43) | A2 screens with **exact OMP on frozen dictionaries** before any IHT training, and uses **one** IHT step count for all arms, selected label-free; **no per-arm λ**, no recalibration |
| **P1** | clean end-to-end core: zero-code `+0.347116`, node shuffle `+0.013124`, all-shuffle `+0.072999`; `G_specific = 0.002559 < 0.003`; official test reversal | mechanism interventions are standard equipment to be repeated on A2's trained arms; sub-threshold is FAIL, not "close" |
| **SDB-v0** | `K=32, s=8` sparse coordinate is a non-lossy, causally used replacement of `phi65` on a *weak* base (recovery 1.159, 3 seeds) but adds only `+0.002381 < 0.003` on the strong S0 backbone | A2 keeps the strong H1 backbone and still requires `≥ 0.003`; no K/s sweep |
| **FEC-D1** | `G_D = +0.005245` (localized binding helps) but `G_dict_specific = -0.001306` (train-fit affine **PCA32** beats the sparse SDB dictionary under a matched protocol) | "coordinate carries value" ≠ "sparsity is specific"; A2 separates the two questions by design and must carry a matched dense reference |
| **FSAR-R2-AR0 / -EDGE** | node `C` and edge `C_E` assignment residuals are real, seed-stable, and beat both the marginal baseline and a parameter-matched marginal-capacity control; attribute permutation leaves `S`, `A`, marginals exactly invariant | the A1 433-D objects built from these statistics remain the right attributed coordinate; A2 does not re-derive them |
| **FSAB** | a binding/attribute branch can die (near-zero gradient + Adam/L2 contraction) while the harness still prints a plausible MAE | A2 carries a preregistered liveness criterion per formal arm (§ pre-registration §10) and cannot declare success on a dead dictionary |
| **TCCD-v0** | canonical node-slot raw attributed patches fail the continuity gate (code AUC 0.4675) | no canonical-slot / node-ID-slot / `patch_cont` / typed-WL-ordering coordinate may enter A2 |

---

## 6. Reuse inventory (exact call targets)

| need | reuse target (no re-implementation) |
|---|---|
| `phi65`, `J^V`, `J^E`, `P^V`, `P^E`, `n_patch`, `m_patch`, `node_sizes`, `atom` | `zinc_e2e_dictenv_a1.load_raw(split)` over the identity-verified A1 cache |
| train-only scaler application (433-D REAL / INDEP) | `zinc_e2e_dictenv_a1.arm_coordinate(arm, split)` / `e2e_dictenv_a1.apply_block_scaler` |
| K-SVD dictionaries | `zinc_e2e_dictenv_a1.load_arm_dictionary(arm)` |
| exact-OMP codes | `zinc_e2e_dictenv_a1.load_or_build_codes(arm, split)` → `sdb_v0.omp_codes` |
| H1 model, coder, parameter accounting | `e2e_dictenv_a1.build_model / A1Model / total_parameter_count` (`decoder="h1"`, `d_e=48`, `K=32`, `s=8`) |
| training loop, Top-5 soup, curve, provenance | `zinc_e2e_dictenv_a1.train_arm` (with A2 tag/dir redirection) |
| Gate-0 correctness / assignment / health checks | `zinc_e2e_dictenv_a1.{correctness_stage,assignment_stage,health_stage}` executed with the A2 result directory substituted for the A1 one, so A1's own artifacts are never rewritten |
| official-test blocker | `zinc_e2e_dictenv_v0._g12_official_test_blocker` via the A1 gate stack |
| `P1` batching / collate / evaluation / shuffle interventions | `e2e_dictenv_p1.make_env_loader`, `env_collate`, `zinc_e2e_dictenv_p1.evaluate`, `_permute_node_shuffle`, `_permute_edge_shuffle` |
| attributed-WL fingerprint + histogram matrix + paired AUC | `code/run_wholegraph_canonical_registration_audit.attributed_wl_fingerprint`, `_histogram_matrix`; `code/run_tccd_v0._patch_graph_for_wl`, `_auc_paired`; `code/tccd_v0._adjacency` |
| tied IHT coder (for the qualification diagnostic) | `e2e_dictenv_v0.tied_iht_codes` |
| **rank-32 dense compression reference (new in A2)** | `sdb_v0.fit_pca_rank` / `sdb_v0.PCARank` — already-audited train-only affine rank-`r` PCA with `dense_codes(X) = (X - mean) @ Vᵀ`, used by SDB-v0 Stage 1/2 (there: `pca32` reference) and by FEC-D1 (`M_P`).  **A2 does not write a second PCA.** |
| code-pairing-removal intervention | `zinc_e2e_dictenv_a1.mechanism_stage`'s exact construction: REAL scaler, `phi` unchanged, node block `J^V → P^V`, edge block `J^E → P^E`, downstream chemistry untouched |

**Constraint A2-C1.** No second implementation of any scientific object, coder,
scaler, dictionary or metric.  If a shared helper must be extended, the
extension is backward-compatible and the owning round's tests still pass.

---

## 7. Official-train patch-size distribution → continuity-v2 strata

Independently recomputed here two ways, which agree exactly:

1. direct BFS radius-2 node counts over all 10 000 official-train molecules
   (`tccd_v0.bfs_distances`, 231 664 roots): mean 6.1231, median 6;
2. the A1 cache's own `n_patch` field (the object's own patch size), same
   counts element-for-element.

| size | count | share |
|---|---:|---:|
| 3 | 6 028 | 2.60 % |
| 4 | 32 067 | 13.84 % |
| 5 | 35 644 | 15.39 % |
| 6 | 70 325 | 30.36 % |
| 7 | 46 315 | 19.99 % |
| 8 | 26 875 | 11.60 % |
| 9 | 11 708 | 5.05 % |
| 10 | 2 386 | 1.03 % |
| 11 | 269 | 0.116 % |
| 12–14 | 47 | 0.020 % |

Quantiles: q05 4, q25 5, q50 6, q75 7, q90 8, q95 9, q99 10.
`≤ 5`: 31.83 %; `6–8`: 61.95 %; `≥ 9`: 6.22 %; `≥ 10`: 1.166 %.

**Frozen A2 strata** (fixed bins on the train quantiles, no per-pool tuning):
`small = n_patch ≤ 5`, `medium = 6 ≤ n_patch ≤ 8`, `large = n_patch ≥ 9`.
A2 additionally reports the pooled equal-size molecule-disjoint row, and
**never** restricts the primary diagnostic to a `≥ 10`-atom tail.

---

## 8. Forbidden list A2 inherits (any of these = a new round)

```
K sweep, s sweep, radius sweep
433-D feature redesign / new primitive / ring / motif / patch_cont / learned embedding
block-weight change, scaler refit, rescaled coordinates
changed atom/bond categories
K64 / s12 / K128 / alternative dictionary or sparsity
per-arm IHT step counts; per-arm or recalibrated lambda
LISTA / ISTA variant / new sparse solver / new optimizer
seed hunting (seed 1 only if the pre-registered condition fires, on GPU1, sequential)
LR / dropout / weight-decay / activation / reader / decoder / horizon sweep
validation-driven selection of anything other than the frozen Top-5 soup rule
official ZINC test access
GPU0 access, DDP, multi-GPU, cross-GPU parallelism
```

---

## 9. Result of the audit

1. A1 is closed with `REPRESENTATION_NOT_QUALIFIED` and its Stage 1–4 budget
   unspent; the A2 round is the *named* successor path from the A1 analysis
   note's §5 proposal, not a rescue of A1's frozen gate.
2. The 433-D `REAL`/`INDEP` objects, the scaler, the K-SVD dictionaries and the
   exact-OMP codes are all present, all pinned by an explicit expected value
   (tracked for the dictionaries, A1-run metadata plus a recomputation for the
   rest), and are therefore reusable without redesign.
3. The A1 continuity numbers are now fully attributed to their populations; A2
   will not reuse any of them as a gate or a baseline, and will compute
   continuity-v2 fresh on a pre-registered equal-size, molecule-disjoint,
   size-stratified population.
4. The scientific order A2 must follow is fixed by the ledger: **task value
   (Stage 1, frozen exact OMP) → compression sufficiency (Stage 2, train-only
   PCA32) → sparse specificity (Stages 3–5)**.
5. The pre-registration freezes all of the above plus the gates, the decision
   tree and the verdict labels at the commit following this file.
