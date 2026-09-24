# E2E-DictEnv-A1 — pre-registration (frozen before implementation)

Round **E2E-DictEnv-A1** · protocol `e2e_dictenv_a1` · subtitle **Invariant
Attributed Dictionary Core** · study `zinc-context-gap` · branch
`exp/e2e-dictenv-a1-attributed-dictionary`.

Prior-artifact audit:
[`e2e_dictenv_a1_prior_artifact_audit.md`](e2e_dictenv_a1_prior_artifact_audit.md).

This file is the frozen specification.  It is committed **before** any A1
implementation, cache build, dictionary fit, training or GPU use.  Any deviation
requires a numbered amendment written here before the affected run.

---

## 0. Round status and scope

One mechanism question, one round:

> In the current, already strong H1 strict-static / no-message-passing ZINC
> framework, if atom/bond attributes participate in the formation of the sparse
> dictionary code `alpha`, does the **real** structure↔attribute pairing provide
> a stable, attributable prediction increment over (i) a **topology-only**
> dictionary and (ii) a **parameter-matched, assignment-independent** attributed
> dictionary?

Not a SOTA attempt.  Not a decoder/optimizer search.  Not a dictionary-capacity
search.  The only new object in the whole round is the dictionary-input
coordinate and therefore the code `alpha` that H1 consumes.

Frozen device discipline: **all CUDA work on physical GPU1 only**; GPU0 is
foreign-occupied and is never touched, never co-tenanted, never used for a
diagnostic.  No DDP, no multi-GPU, no cross-GPU parallelism, one process per
formal run.

---

## 1. Hypotheses

* **H-A1 (primary).** The real structure↔attribute pairing inside dictionary
  formation carries information that the *same* structural marginals and the
  *same* attribute marginals cannot express, and this is visible as a
  counterfactual: `REAL` beats `INDEP` under an exact parameter-matched,
  architecture-matched protocol, using the same `K=32, s=8` sparsity and the
  same downstream H1.
* **H-A1-topo (secondary).** `REAL` also beats the topology-only dictionary.
* **H-A1-mech.** At inference time, replacing only the *attribute part of the
  dictionary input* (`J^V, J^E → P^V, P^E`, i.e. destroying the pairing that
  forms `alpha`) while leaving all downstream chemistry intact materially
  degrades the trained `REAL` model.
* **H-A1-sparse (conditional, Stage 4).** The attributed *sparse* dictionary is
  better than an equally-parameterized dense tied coordinate of the same 433-D
  object.

Pre-declared interpretation of a negative primary: if `REAL ≈ INDEP`, then
attribute pairing inside dictionary formation is not supported, and a good
absolute MAE for `REAL` may **not** be attributed to the attributed pairing.

---

## 2. Frozen objects

All objects are per-root (`i` = original node id); every statistic is invariant
to node relabeling, patch storage order and edge listing order, and is
assignment-sensitive by construction.

### 2.1 Reused pure-topology root coordinate

```
phi_i in R^65
```

bit-identical reuse of the audited builder `fsar_r2_ar0.build_phi` /
`fsar_v2._explicit_basis_for_patch` via the existing P1 environment cache
(`results/e2e_dictenv_p1/cache/env_{train,valid}.pt`, field `phi`).
Layout: `root_basis(11) | node_mean(11) | node_std(11) | edge_mean(15) |
edge_std(15) | log1p|V_i| | log1p|E_i|`.  Re-derivation of a "similar" 65-D
coordinate is forbidden.

### 2.2 Reused pure-topology occurrence bases

For the radius-2 induced patch `P_i` of root `i`:

```
b^V_{iv} in R^11   (node occurrence basis, pure topology, root-conditioned)
b^E_{ie} in R^15   (undirected-edge basis, endpoint-swap symmetric, pure topology)
```

Both come from `fsar_v2._explicit_basis_for_patch(graph, center, radius=2)`;
`b^V` = the node-basis row of `v` in the rooted frame of `i`; `b^E` = the
edge-basis row of the induced undirected edge `e ⊂ P_i`.  `b^V` contains only
root indicator, rooted shell one-hot, log degree, shell-resolved neighbour
counts, rooted walk counts `A, A^2, A^3`; `b^E` contains only shell-pair
one-hot, degree statistics, common-neighbour count, shell-resolved neighbour
statistics.  No chemistry, no ring/cycle feature, no learned parameter.

### 2.3 Node attributed joint block

```
q_v = onehot(atom_type_v) in R^28          (ZINC primitive atom categories)
J^V_i = sum_{v in P_i} b^V_{iv} q_v^T      in R^{11 x 28}  -> vec 308
```

Requirements (all tested): node-relabel invariant, storage-order invariant,
root-conditioned (because `b^V` is rooted), assignment-sensitive, no message
passing, no canonical node slot, storage `float32` computed in `float64`.

### 2.4 Edge attributed joint block

```
r_e = onehot(bond_type_e) in R^4           (ZINC primitive bond categories)
J^E_i = sum_{e in P_i} b^E_{ie} r_e^T      in R^{15 x 4}   -> vec 60
```

Each **undirected** induced edge contributes exactly once.  Requirements:
endpoint-swap invariant, edge-order invariant, graph-relabel invariant, bond-
assignment sensitive, no learned bond embedding.

### 2.5 The two 433-D dictionary objects

```
chi_REAL_i  = [ phi_i (65) ; vec(J^V_i) (308) ; vec(J^E_i) (60) ]   in R^433
```

Assignment-independent analytic null (identical marginals, no pairing), with
`n_i = |P_i|`, `m_i = |E_i|`:

```
P^V_i = (1/n_i) (sum_{v in P_i} b^V_{iv}) (sum_{v in P_i} q_v)^T     in R^{11x28}
P^E_i = (1/m_i) (sum_{e in P_i} b^E_{ie}) (sum_{e in P_i} r_e)^T     in R^{15x4}
        = all zeros when m_i = 0   (mandatory zero-edge convention)
chi_INDEP_i = [ phi_i (65) ; vec(P^V_i) (308) ; vec(P^E_i) (60) ]    in R^433
```

`REAL` and `INDEP` have the same dimension, the same patch support, the same
marginals, the same `phi65`, the same downstream, and exactly the same
parameter count (the dictionary is `433 x 32` in both).

---

## 3. Train-only scaling protocol

Fit on **official train only**; identical *procedure* for both objects.  No
official-valid or official-test statistic may enter.

### 3.1 Layer 1 — per-coordinate RMS

For each object block, per coordinate `j`:

```
scale_j = sqrt(E_train[z_j^2] + eps)      eps = 1e-12
mask_j  = 1 if sqrt(E_train[z_j^2]) > 1e-9 else 0
z~_j    = (z_j / scale_j) * mask_j
```

Zero-RMS coordinates are masked to zero (never divided by tiny noise) and the
masked count per block is recorded.  **No dataset-mean subtraction** (inherited
FSAR/SDB convention).

### 3.2 Layer 2 — block-energy normalization

After layer 1, with the three blocks
`S = phi65`, `V = node joint 308`, `E = edge joint 60`:

```
e_b = E_train[ ||block_b||_2^2 ]        (b in {S, V, E})
w_b = 1 / sqrt(e_b + eps)
block_b <- w_b * block_b
```

so the three blocks carry equal mean squared L2 energy on official train.
Final coordinate:

```
x_i = [ w_S * S~_i ; w_V * V~_i ; w_E * E~_i ]   in R^433   for REAL and INDEP
```

Every `scale`, `mask`, `w_b`, the block energies, the masked-coordinate counts
and the SHA256 of the resulting scaler blob are persisted
(`results/e2e_dictenv_a1/scaler_meta.json`).

### 3.3 The topology-only arm's coordinate

`T0` uses the frozen SDB-v0 dictionary on **raw `phi65`**, exactly as
P1/P2-H1 do (single block, therefore no block-energy step).  This keeps `T0`
bit-comparable with the historic H1 pipeline and makes the secondary comparison
conservative (its OMP screen is stronger than the historic IHT-10).

**Forbidden:** fitting any scaler or block weight on official valid, sweeping
`w_S/w_V/w_E`, per-arm scaling protocols, dataset-mean subtraction.

---

## 4. Dictionary budget and fitting

```
K = 32, s = 8      (frozen; no sweep)
K-SVD: sdb_v0.fit_ksvd(X_train, atoms=32, sparsity=8, epochs=10, seed=20260924)
OMP:   sdb_v0.omp_codes(D, X, s=8)      (exact top-8)
IHT:   e2e_dictenv_v0.tied_iht_codes(Dbar, X, s=8, steps=selected)
```

* `D_TOPO  in R^{65 x 32}`  := the **existing frozen SDB-v0 dictionary**
  (`results/sdb_v0/dictionary.pt`, sha256(float32) `b0c5da98…`), which was fit on
  official-train `phi65` with exactly the procedure above.  No re-fit, no
  re-seed.
* `D_INDEP in R^{433 x 32}` := fresh fit on `x_INDEP` official-train rows.
* `D_REAL  in R^{433 x 32}` := fresh fit on `x_REAL` official-train rows.

All fits on official train only.  Valid/test never enter K-SVD, OMP, the scaler,
or dictionary selection.

---

## 5. Gate 0 — label-free representation qualification (before any predictor training)

Modeled on TCCD-v0 Gate 1.  Official ZINC **test is never loaded** anywhere in
A1.  Gate 0 is label-free (`y` never read).

### G0.1 correctness — all must pass

```
433-D exact layout and block boundaries (65 / 308 / 60)
node relabel invariance (phi, J^V, J^E, P^V, P^E, x)      tol 1e-6
edge listing order invariance and endpoint swap invariance
batching invariance (per-molecule vs batched aggregation)
deterministic rebuild (bit-identical on repeat build)
finite / no NaN / no Inf in every block and in x
train-only scaler (fit set = official train; changes if a valid row is added)
official-test blocker: loading "test" raises
```

plus hand-computed reference values for one toy patch (`J^V`, `J^E`, `P^V`,
`P^E`) and the zero-edge case.

### G0.2 assignment semantics

On a fixed topology with a fixed per-patch attribute multiset, permute the
attribute assignment inside each patch (per-root permutation of the occurrence
list):

```
REAL:  J^V and J^E must change (relative change >= 1e-6)
INDEP: P^V and P^E must be invariant (max abs diff <= 1e-6)
phi65: exactly unchanged (0.0)
marginals: sum_v q_v, sum_e r_e, sum_v b^V_iv, sum_e b^E_ie all preserved
```

### G0.3 TCCD-style continuity gate (primary qualification)

Adapted from `code/run_tccd_v0.py:continuity_audit`, run on official train only,
seed `20260924 + 9`:

1. pool of `1500` (molecule, root) patches from official train, deduplicated;
2. ground-truth similarity = typed WL subtree kernel (rounds 0-3) on the **raw
   attributed rooted patch** (`_patch_graph_for_wl` +
   `attributed_wl_fingerprint`);
3. *near* pairs = the `800` highest-cosine **non-isomorphic** pairs (complete
   canonical key differs); *graded tail* = the next `800`;
4. *random* pairs = matched on (patch size, root atom category), non-isomorphic;
5. report distance AUC in `x`-space and in exact-OMP code space
   (`_auc_paired`), for the primary and graded strata.

**Primary continuity gate (REAL only):**

```
code-space AUC(chi_REAL) >= 0.70
```

`x`-space AUC and the INDEP figures are reported descriptively.

**Stop rule.**  If `REAL` fails this gate the round stops at Gate 0:
no Stage 1, no H1 training, no canonicalization redesign, no descriptor swap, no
"temporary" rescue.  Verdict `REPRESENTATION_NOT_QUALIFIED`.

### G0.4 dictionary health

For each of `D_TOPO / D_INDEP / D_REAL`, label-free, on official train with a
deterministic held-out slice of official-train rows:

```
normalized reconstruction error (OMP, s=8) on fit rows and held-out rows
used atoms, effective atom count, dictionary column norms (min/mean/max)
top-1 and top-8 coefficient mass, support entropy
train<->heldout usage Spearman, dead-atom fraction
matched random-normalized-dictionary reconstruction control
```

Purpose: exclude collapse / useless dictionaries.  This gate is a
qualification, not a superiority test.

---

## 6. Stage 1 — frozen dictionary + exact-OMP representation screen

Prerequisite: Gate 0 PASS.

```
fit K-SVD on official train        (once per arm)
exact OMP encode train + valid      (s=8)
freeze D, freeze alpha
train H1 with the frozen code
```

Arms (only the `alpha` source differs):

| arm | coordinate | dictionary | code |
|---|---|---|---|
| `T0` TOPO-OMP | `phi65` | `D_TOPO` (frozen) | exact OMP s=8 |
| `A0` ATTR-INDEP-OMP | `x_INDEP` 433-D | `D_INDEP` | exact OMP s=8 |
| `A1` ATTR-REAL-OMP | `x_REAL` 433-D | `D_REAL` | exact OMP s=8 |

Frozen training protocol for all three arms (and for Stage 3): seed 0, official
train 10,000 / official valid 1,000, Adam `lr 1e-3`, `weight_decay 1e-5`,
batch 128, gradient clip 5.0, no scheduler, 320 epochs, fixed Top-5 soup by
official-valid MAE, `lambda_rec = 33.95873017865987` with the arm-independent
normalized reconstruction objective below, H1 decoder/backend unchanged.
`D` is frozen (`requires_grad = False`) in Stage 1, so the reconstruction term
is a reporting quantity with zero gradient; it is reported, never optimized.

Arm-independent normalized reconstruction objective (identical formula in every
arm, on the arm's own dictionary input `x`):

```
rec = mean_i  ||x_i - x_hat_i||^2 / (||x_i||^2 + 1e-12)
```

Same scalar λ for all arms; no per-arm recalibration.

**Primary Stage-1 quantity:**

```
G_attr_OMP = MAE(A0 ATTR-INDEP-OMP) - MAE(A1 ATTR-REAL-OMP)     gate >= 0.003
```

**Secondary:**

```
G_topo_OMP = MAE(T0 TOPO-OMP)      - MAE(A1 ATTR-REAL-OMP)      gate >= 0.003
```

**Stop rule.**  `REAL` vs `INDEP` decides.  If `G_attr_OMP < 0.003`, the
attributed pairing has insufficient representation-level increment:
STOP, do **not** enter the task-coupled E2E (Stage 3), verdict
`NO_ATTRIBUTED_DICTIONARY_FORMATION_SIGNAL`.  Official test is not read.

---

## 7. Stage 2 — label-free IHT coder qualification

Prerequisite: Stage 1 primary PASS.  No `y`, official train rows only.

Candidate tied-IHT step counts, fixed for **every** arm:

```
{10, 30, 100, 200}
```

On one deterministic official-train row subset (`n = 50,000`, seed `20260924`),
for each of the three dictionaries compare

```
exact OMP (reference)  |  IHT-10  |  IHT-30  |  IHT-100  |  IHT-200
```

and record: normalized reconstruction error, support size, support Jaccard with
the OMP support, relative code error vs the OMP code, runtime.

**Selection rule (frozen).**  Choose the **smallest** step count such that
`TOPO`, `INDEP` and `REAL` all reach normalized reconstruction error
`<= 0.002`.  One step count is used by all formal arms; per-arm step counts are
forbidden.

**Stop rule.**  If no candidate satisfies it: STOP with verdict
`ATTRIBUTED_REPRESENTATION_SUPPORTED_BUT_CODER_BLOCKED`.  Adding 500/1000
steps, LISTA, a new optimizer, or a different solver is forbidden.

---

## 8. Stage 3 — matched E2E attributed dictionary

Prerequisite: Stage 1 primary PASS **and** Stage 2 qualified (one shared step
count).  `D` is now **trainable** (task-coupled), as in P1/P2-H1.

Arms (identically configured except the coordinate/dictionary):

| arm | coordinate | dictionary | code |
|---|---|---|---|
| `E0` TOPO-SPARSE | `phi65` | `D_TOPO` | tied IHT, selected steps |
| `E1` ATTR-INDEP-SPARSE | `x_INDEP` 433-D | `D_INDEP` | tied IHT, selected steps |
| `E2` ATTR-REAL-SPARSE | `x_REAL` 433-D | `D_REAL` | tied IHT, selected steps |

Matched conditions, all arms: same physical GPU1, same seed, same data, same
batch size and batch order, same H1 decoder/anchor/post-code node binding/
post-code edge binding/15-D pure-topology relation/global path/reader, same
`K=32, s=8`, same selected IHT step count, same 320-epoch horizon, same
optimizer/Adam/lr/wd/clip, same Top-5 soup definition, same reconstruction
objective and the same scalar `lambda_rec = 33.95873017865987`.  `E1` and `E2`
have exactly identical parameter counts and identical input width 433.
No per-arm tuning of anything.

**Primary:**

```
G_attr = MAE(E1 ATTR-INDEP-SPARSE) - MAE(E2 ATTR-REAL-SPARSE)   gate >= 0.003
```

**Secondary:**

```
G_topo = MAE(E0 TOPO-SPARSE)       - MAE(E2 ATTR-REAL-SPARSE)   gate >= 0.003
```

The primary claim is decided **only** by `REAL vs INDEP`.  A low absolute MAE
for `E2` — even below the historic `0.123549` — is **not** evidence for the
attributed pairing.

**Stop rule.**  If `G_attr < 0.003`, verdict
`NO_ATTRIBUTED_DICTIONARY_FORMATION_SIGNAL`; Stage 4 is not run.

---

## 9. Stage 4 — sparse specificity (conditional)

Runs only if `G_attr >= 0.003` **and** the §10 mechanism intervention passes.

Arm:

```
S4 = ATTR-REAL-DENSE-TIED
```

Same 433-D input `x_REAL`, same dictionary parameterization (one learned
`433 x 32` matrix `P` initialized from `D_REAL`, column-normalized), same
downstream H1, same parameter count, same optimizer/protocol; the top-8
sparse/IHT operator is **removed** and replaced by the matched dense tied
coordinate `z = x @ Pbar`.  No extra MLP, no extra parameter.

```
G_sparse = MAE(S4 ATTR-REAL-DENSE-TIED) - MAE(E2 ATTR-REAL-SPARSE)   gate >= 0.003
```

Interpretation is fixed (see §12).

---

## 10. Mechanism interventions (inference-only, on trained E2)

### 10.1 Dictionary-input pairing intervention (primary mechanism test)

For the trained `E2` model:

```
normal:       chi_REAL -> D -> code -> H1 downstream
intervention: chi_INT = [phi ; vec(P^V) ; vec(P^E)] -> the SAME trained D
              -> alpha_int -> H1 downstream (REAL chemistry everywhere else)
```

Only *attribute → code formation* is destroyed; the post-code node/bond
chemistry binding, the anchor chemistry and every other downstream chemistry
path stay real.  Record `MAE_real`, `MAE_code_pairing_removed`, the delta, the
mean absolute prediction shift, and the per-molecule shift distribution.  Also
report the node-only and edge-only removals descriptively (they are **not** used
to re-select an architecture).

**Primary mechanism threshold:** `delta >= 0.010`.

### 10.2 Inherited P1 interventions

```
zero dictionary/code coordinate (coord = 0)
node assignment shuffle within (root, shell)   (existing helper)
edge (bond-type) assignment shuffle within (root, shellpair) (existing helper)
combined shuffle
```

Each is evaluated with the frozen soup state; a stale/decorative branch must
show up here.

### 10.3 Liveness

```
||dL/dD|| on a real batch; D movement from the K-SVD init (Frobenius, relative)
active atoms / effective atom count; effective rank of D and of the codes
train<->valid code-usage correlation; per-slot code variance
decoder dictionary-slot gradient statistics
```

FSAB-style annihilation or a constant attributed channel is an explicit
negative finding and may not be called a success.

---

## 11. Seeds, budget and run discipline

* Seed 0 only for the full gate sequence.  **No automatic multi-seed runs.**
* Seed 1 (paired, arms `{E0, E1, E2}`) is **authorized** only if the seed-0
  Stage-3 primary `G_attr >= 0.003`, and only if the remaining wall-clock budget
  allows a sequential execution; it is a stability check reported alongside the
  seed-0 verdict and never replaces it.  It is **not** run if `G_attr < 0.003`
  (no seed hunting).
* All formal arm runs are executed **strictly sequentially** on GPU1 so that the
  paired arms share identical execution conditions.  No concurrency, no DDP, no
  cross-GPU parallelism.
* GPU0 is never used, never co-tenanted.  `nvidia-smi` is recorded once for
  provenance but no A1 process may be scheduled on GPU0.
* Every formal artifact records: git commit, branch, hostname, requested
  physical GPU = 1, GPU model, CUDA, PyTorch, Python, seed, protocol id,
  `official_test_loaded = false`, wall time, peak GPU memory.
* The K-SVD/OMP/scaler work is CPU work and may run on the remote host CPU; it
  is never a reason to touch GPU0.

---

## 12. Decision table (frozen)

| condition | verdict |
|---|---|
| Gate 0 G0.3 continuity FAIL | `REPRESENTATION_NOT_QUALIFIED` |
| Stage 1 `G_attr_OMP < 0.003` | `NO_ATTRIBUTED_DICTIONARY_FORMATION_SIGNAL` |
| Stage 1 PASS but Stage 2 has no qualifying step count | `ATTRIBUTED_REPRESENTATION_SUPPORTED_BUT_CODER_BLOCKED` |
| Stage 3 `G_attr < 0.003` | `NO_ATTRIBUTED_DICTIONARY_FORMATION_SIGNAL` |
| Stage 3 `G_attr >= 0.003`, mechanism `< 0.010`, or Stage 4 skipped | `ATTRIBUTED_PAIRING_SUPPORTED_DICTIONARY_NOT_SPECIFIC` (mechanism reported separately) |
| Stage 3 `G_attr >= 0.003`, mechanism `>= 0.010`, Stage 4 `G_sparse >= 0.003` | `ATTRIBUTED_SPARSE_DICTIONARY_SUPPORTED` |
| Stage 3 `G_attr >= 0.003`, mechanism `>= 0.010`, Stage 4 `G_sparse < 0.003` | `ATTRIBUTED_PAIRING_SUPPORTED_DICTIONARY_NOT_SPECIFIC` |

Precedence: qualification gates first, then Stage 1, then Stage 2, then Stage 3,
then mechanism, then Stage 4.  Additional non-gating observations are reported
but never override a FAIL.

---

## 13. The six questions this round must answer

| # | question | evidence |
|---|---|---|
| Q1 | does the invariant 433-D attributed domain pass a TCCD-style continuity qualification? | G0.3 (`x`-space AUC, code-space AUC, graded tail) |
| Q2 | frozen exact-OMP screen: `REAL` vs `INDEP` material gain? | Stage 1 `G_attr_OMP` |
| Q3 | matched E2E H1: `REAL` vs `INDEP` material gain? | Stage 3 `G_attr` |
| Q4 | does `REAL` beat the re-established same-GPU1-regime TOPO H1 baseline? | Stage 3 `G_topo` |
| Q5 | does removing only "attribute → dictionary code formation" degrade the trained model? | §10.1 |
| Q6 | sparse attributed dictionary vs dense tied attributed coordinate — which supplies the increment? | Stage 4 `G_sparse` |

---

## 14. Claim boundary

If the round succeeds, the maximum claim is:

> In a permutation-invariant, strict-static, no-message-passing radius-2 local
> representation, letting the real node/edge structure↔attribute assignment
> participate in dictionary-code formation provides a matched-control prediction
> increment on ZINC official valid.

Without the Stage-4 gate, A1 may **not** claim that sparsity itself is
necessary, that K-SVD is uniquely necessary, or that discrete dictionary atoms
are uniquely necessary.  A1 does **not** reproduce the mentor pipeline: it
restores the mentor's scientific idea that the local object entering dictionary
formation is attributed, and replaces the mentor's / history's fixed-coordinate
slot representation with an invariant joint statistic (because TCCD-v0 measured
the slot variant's coordinate discontinuity).

Regardless of outcome, A1 may **not** claim anything about the official ZINC
test: the test set is never loaded in this round.

---

## 15. Artifacts (all under `tracks/ksvd/results/e2e_dictenv_a1/`, gitignored)

```
correctness.json
scaler_meta.json
dictionary_meta_{topo,indep,real}.json
continuity_audit.json
dictionary_health_{topo,indep,real}.json
assignment_semantics.json
omp_screen_{T0,A0,A1}.json          (Stage-1 runs incl. curves)
omp_screen_selection.json           (G_attr_OMP, G_topo_OMP)
iht_diagnostic.json                 (Stage 2)
formal_runs/E0.json E1.json E2.json (Stage 3) + curves/ states/
mechanism_code_pairing.json
mechanism_{zero,node_shuffle,edge_shuffle,all_shuffle}.json
liveness_E2.json
specificity_E2_vs_dense.json        (Stage 4, conditional)
parameter_accounting.json
decision.json
REPORT.md
DECISION.md
seed1/                              (only if authorized)
```

Code (single implementation, no duplicated logic):

```
tracks/ksvd/experiments/luyin16/e2e_dictenv_a1.py        (object + model + scaling)
tracks/ksvd/experiments/luyin16/zinc_e2e_dictenv_a1.py   (runner: gates, stages, reports)
tracks/ksvd/tests/test_e2e_dictenv_a1.py                 (targeted CPU tests)
```

Notes: `e2e_dictenv_a1_implementation.md`, `e2e_dictenv_a1_analysis.md`.

---

## 16. Explicit non-goals / forbidden rescues

Any of the following after a gate FAIL requires a **new** preregistration and is
forbidden inside A1:

```
K / s / radius / d_e / decoder / reader / activation / dropout / LR / wd /
horizon / lambda change; a new sparse solver; LISTA; an extra MLP or
nonlinearity after the code; attention; Transformer; GNN / message passing;
recurrence; explicit ring/cycle features; motif counts; patch_cont;
atom_shell/bond_shell histograms; a new chemistry embedding; a learned atom/bond
embedding in the dictionary input; per-arm IHT steps; per-arm lambda;
validation-driven block reweighting; seed hunting; official-test probing;
GPU0 usage; non-sequential paired arm execution for the primary comparison.
```

A failed gate is a result.  It is recorded, not rescued.
