# Pre-registration — FEC-D0: Within-Coarse-Role Residual Structural Dictionary Audit

Round: **FEC-D0**. Protocol id `fec_d0`. Study `zinc-context-gap`.
Prior-artifact audit: [`fec_d0_prior_artifact_audit.md`](fec_d0_prior_artifact_audit.md).

Written **before** any dictionary fit, sparse code, probe or diagnostic run.
Lineage HEAD: `ca2e947013263a2590c7ad4edffff03a46e6b830`.

This round is:

```
LABEL-FREE
NO PROPERTY TRAINING
NO OFFICIAL VALID
NO OFFICIAL TEST
NO FEC-S1 RETRAIN
NO DICTIONARY TASK COUPLING
```

Everything below is **frozen**. A gate FAIL ends the round at that gate.
There is **no** sweep of `K`, `s`, iterations, initialization, solver, `λ`, and
no rescue (no K=32, no s=8, no more iterations, no nonlinear dictionary, no new
basis, no edge dictionary, no second seed).

---

## 0. Scientific question

> After the coarse rooted structural role `shell(i,v) ∈ {0,1,2}` is *known*,
> does the remaining pure-topology variation of a rooted radius-2 node
> occurrence admit a **small, reusable, sparse, chemistry-independent,
> molecule- and root-independent** structural vocabulary?

Formally, with `φ^⊥_{iv}` the shell-residualized rooted occurrence coordinate,

```
φ^⊥_{iv} ≈ D α_{iv},   ‖α_{iv}‖_0 ≤ 4,   D ∈ R^{d_res × 16}
```

A YES authorizes only the *proposal* of a next round
(`FEC-D1 — Baseline-Preserving Sparse Structural Subrole Binding`); it does
**not** authorize any property experiment.

---

## 1. Purity contract (hard)

The dictionary fit, coding, probes and diagnostics may read only raw **untyped**
topology: adjacency, rooted BFS distance, induced degrees, rooted untyped walk
counts. They may **not** read:

```
atom category · bond category · aromatic/ring chemistry label · target y
learned hidden state · message-passing state · typed token · parent token
official valid · official test
```

Ring/cycle structure that is a deterministic function of the untyped topology
is not leakage, but **no** coordinate beyond the historically audited 11 is
added.

---

## 2. Frozen objects

### 2.1 Rooted structural basis (`φ_{iv}`)

`φ_{iv} = b^V_{iv} ∈ R^{11}` = the audited FSAR explicit rooted **node**
basis row of occurrence `(i,v)`, root `i`, radius-2 induced patch, produced by
`fsar_v2._explicit_basis_for_patch` (chemistry-free, root-conditioned,
occurrence-level). Coordinates (see audit §1.2):

```
[0]    root_indicator
[1:4]  shell_one_hot (distance from root ∈ {0,1,2})
[4]    log1p(within-patch degree)
[5:8]  log1p(neighbour_by_shell)
[8:11] log1p(A^k e_root), k = 1,2,3
```

No edge basis is used. `b^E` is explicitly **out of scope** (`edge dictionary`,
`bond-role dictionary`, `node+edge joint dictionary` forbidden).

### 2.2 Coarse role

`s_{iv} = shell(i,v) ∈ {0,1,2}` (BFS distance from the root of the occurrence's
patch, read off coordinate `[1:4]`).

### 2.3 Shell-coordinate deletion

Delete every **exact** shell coordinate from `φ_{iv}` before residualization:

* `[1:4]` — the shell one-hot;
* `[0]` — `root_indicator`, which is *exactly* `1[shell=0]`.

```
φ̃_{iv} ∈ R^{7}  =  [ log1p(deg), log1p(neigh_by_shell) x3, log1p(walk1..3) ]
```

`φ̃` is the *pre-residualization* coordinate. `d_raw = 7`.

### 2.4 Per-shell residualization (FIT-only statistics)

For each shell `s`, using **FIT occurrences only**:

```
μ_s = mean_{iv ∈ FIT, s_{iv}=s} φ̃_{iv}
σ_s = std_{iv ∈ FIT, s_{iv}=s} φ̃_{iv}          (per-coordinate, ddof=0)
φ^⊥_{iv} = ( φ̃_{iv} − μ_{s_{iv}} ) / max(σ_{s_{iv}}, 1e-6)
```

HOLDOUT applies the FIT `μ_s, σ_s` **unchanged**. Forbidden: HOLDOUT-fit
statistics, PCA-whitening-then-K-SVD, target-informed scaling,
chemistry-conditioned scaling. `d_res = 7`.

### 2.5 Dictionary (fixed, no sweep)

```
K = 16  atoms
s = 4   sparsity (exact top-s)
ONE shared dictionary D ∈ R^{7 × 16}  (not one dictionary per shell)
```

Fit: `sdb_v0.fit_ksvd` → `tccd_v0.ksvd_fit` (deterministic init from
`random_normalized_dictionary`, unit-normalized atoms, OMP exact top-s,
float64, `epochs = DICT_SEED`-seeded mini-batches, fixed iteration policy
reused from SDB/TCCD). Coding: `tccd_v0.omp_codes` (exact top-s). No
re-implementation of any sparse solver.

Fit data: all FIT-split residualized occurrences `φ^⊥`, pooled across the three
shells.

### 2.6 Controls (fixed)

* **D** — learned dictionary `D_learned` (FIT-only K-SVD).
* **R** — matched random dictionary: same `K=16`, `s=4`, column normalization,
  same sparse solver, `random_normalized_dictionary(F=7, K=16, seed=DICT_SEED)`.
* **P** — PCA reference, FIT-only, `rank = min(4, d_res) = 4`, dense low-rank
  reference only (not a baseline the dictionary must beat).

No k-means, no autoencoder, no LISTA, no dense learned projection, no nonlinear
encoder.

---

## 3. Data split (frozen)

Official ZINC **train** only (`data/ZINC/subset`, PyG `subset=True`). Canonical
repo train-internal split, `SPLIT_SEED = 20260922`:

```python
rng = np.random.RandomState(20260922)
perm = rng.permutation(10000)
HOLDOUT = sorted(perm[:2000])                      # 2000 official-train molecules
FIT     = sorted(set(range(10000)) - set(HOLDOUT)) # 8000 official-train molecules
```

Recorded per-split fingerprints (sorted molecule indices, `n_nodes`, `n_edges`)
must be written to `split_fingerprint.json`. Guards raise on any
`"val"`/`"test"` access and on any target `y` read.

---

## 4. Gate 0 — purity / correctness (all must pass before any fit)

| id | check |
|---|---|
| **G0** | Chemistry invariance: for a fixed topology, randomize atom category and bond category maps; `φ^⊥` and `α` must be bit-identical. |
| **G1** | Relabel invariance: node relabeling of the molecule; with the induced occurrence mapping, `φ^⊥` bit-identical and `α` bit-identical. |
| **G2** | Exact sparsity: every code `‖α‖_0 ≤ 4`; report the exact-top-4 fraction. |
| **G3** | FIT-only transforms: assert residual `μ_s, σ_s`, `D`, and PCA never saw HOLDOUT statistics (provenance flags + a HOLDOUT-poisoning test). |
| **G4** | No target: monkeypatch any target access to raise; whole D0 still succeeds. `targets_loaded = false`. |
| **G5** | Coarse anchor removed: deterministic probe that no exact-shell coordinate is retained in `φ^⊥`; report `E[φ^⊥ | shell]` (≈0 on FIT by construction). |
| **G6** | Numerical reference: independent float64 reference verifies `Dα` against the reported reconstruction for a small synthetic matrix. |

Any failure → STOP (`FEC_D0_CORRECTNESS_FAILURE`).

---

## 5. Gate 1 — residual structure exists (HOLDOUT)

```
E_D = ‖φ^⊥ − D_learned α_D‖² / ‖φ^⊥‖²
E_R = ‖φ^⊥ − D_random  α_R‖² / ‖φ^⊥‖²
R_rec = E_D / E_R
```

(per-molecule-normalized mean of squared residual / mean of squared input,
computed HOLDOUT-only).

```
R_rec ≤ 0.80  → PASS
R_rec > 0.80  → FEC_D0_NO_LEARNABLE_RESIDUAL_STRUCTURE (STOP)
```

Also report `E_D`, `E_R`, and the PCA-4 reference `E_P`.

---

## 6. Gate 2 — reusable vocabulary

Per atom, on FIT and on HOLDOUT separately:

```
support occurrence count · distinct molecule count · distinct root count
shell distribution · support frequency · coefficient magnitude distribution
```

atom `active` := `support_count > 0`.

```
≥ 12 / 16 atoms active on FIT
≥ 12 / 16 atoms active on HOLDOUT
Jaccard(active_fit, active_holdout) ≥ 0.75
≥ 12 FIT-active atoms appear on ≥ 20 distinct HOLDOUT molecules
Spearman(p_fit, p_holdout) ≥ 0.70   (atom support-frequency vectors)
```

Report also: entropy, effective atom count, top-1 atom mass, top-4 atom mass,
Gini, max single-molecule contribution per atom.

Failure → `FEC_D0_DICTIONARY_NOT_REUSABLE` (STOP).

---

## 7. Gate 2b — dense basis has held-out subrole signal

### 7.1 Diagnostic label (never read by the dictionary)

`T_iv` = exact **untyped marked-node rooted topology class** of occurrence
`(i,v)`:

1. radius-2 induced untyped patch of root `i`;
2. mark root `i`, marked node `v`, and root-distance shell;
3. delete atom category and bond category;
4. canonicalize into a complete structural invariant.

Canonicalization reuses the **repaired** machinery
(`typed_patch_tokenizer.corrected_canonical_key`, certificate **plus**
canonical semantic color sequence) with colors
`("node", is_root, is_marked, distance_from_root)` and `("edge",)`. The
historical aliased-certificate bug must not recur; unit tests assert
relabel-invariance and non-collision.

### 7.2 Class eligibility (rare-class control)

Keep marked-topology classes with `FIT count ≥ 20` **and**
`HOLDOUT count ≥ 5`. Probes are run **within each shell** so `shell` itself
contributes no gain.

### 7.3 Probe (fixed, no tuning)

Multinomial logistic regression, L2, `C = 1`, fixed `max_iter`, fit on FIT only,
evaluate on HOLDOUT only. Four inputs, identical protocol:

* **B** trivial baseline: within-patch degree only (shell already fixed by stratification);
* **R** random-dictionary code `α_R`;
* **D** learned-dictionary code `α_D`;
* **O** dense oracle: the full `φ^⊥`.

Primary metric macro-F1; report accuracy too.

```
G_O = F1_O − F1_B
G_O ≥ 0.03   else  FEC_D0_BASIS_HAS_NO_HELDOUT_SUBROLE_SIGNAL (STOP)
```

Report shell0 / shell1 / shell2 separately and the pre-registered
across-shell **weighted aggregate** (weight = eligible occurrence count per
shell). A shell without enough eligible classes is reported as coverage, not
forced to fail; in shells with enough data the learned dictionary may not be
worse than random by more than `0.02` macro-F1.

---

## 8. Dictionary semantic gate

```
G_D = F1_D − F1_B
G_R = F1_R − F1_B
```

Both required:

```
F1_D − F1_R ≥ 0.03
G_D / G_O ≥ 0.70
```

Failure → `FEC_D0_DICTIONARY_LOSES_SUBROLE_INFORMATION` (STOP).

---

## 9. Gate 3 — atom semantics (report-only)

Per atom: degree distribution, marked-topology distribution, shell
distribution; atom→degree mutual information, atom→marked-topology mutual
information, conditional marked-topology entropy given degree. Report-only, no
gate. If atoms are essentially degree buckets, the interpretation must state
`dictionary is primarily a discretized degree representation`.

---

## 10. Stability diagnostics (report-only)

* Input-noise stability: add `ε ~ N(0, 1e-4)` to standardized residual
  coordinates with a fixed seed; report support Jaccard, coefficient cosine,
  top-1 atom stability.
* Split stability: compare FIT vs HOLDOUT atom usage, support semantics,
  marked-topology concentration. **No second dictionary fit.**

---

## 11. Success gate (all required)

```
A. Gate 0 all pass
B. E_D / E_R ≤ 0.80
C. ≥12/16 active FIT, ≥12/16 active HOLDOUT, active Jaccard ≥ 0.75,
   ≥12 atoms used by ≥20 HOLDOUT molecules, usage Spearman ≥ 0.70
D. F1_O − F1_B ≥ 0.03
E. F1_D − F1_R ≥ 0.03  and  (F1_D − F1_B)/(F1_O − F1_B) ≥ 0.70
```

All → `FEC_D0_RESIDUAL_STRUCTURAL_DICTIONARY_QUALIFIED`.

Frozen negative verdicts (first decisive failure, in order):

```
FEC_D0_ROOTED_BASIS_UNAVAILABLE              (audit)
FEC_D0_CORRECTNESS_FAILURE                   (Gate 0)
FEC_D0_NO_LEARNABLE_RESIDUAL_STRUCTURE       (Gate 1)
FEC_D0_DICTIONARY_NOT_REUSABLE               (Gate 2)
FEC_D0_BASIS_HAS_NO_HELDOUT_SUBROLE_SIGNAL   (Gate 2b)
FEC_D0_DICTIONARY_LOSES_SUBROLE_INFORMATION  (Gate semantic)
```

No "close so continue" verdicts.

---

## 12. Out of scope / forbidden

Even on full PASS: **STOP.** No `y`, no MAE training, no official valid, no
official test, no FEC-S1 retrain, no dictionary residual gate, no task-coupled
`D`, no seed0 task run. A PASS authorizes only the *proposal* of
`FEC-D1 — Baseline-Preserving Sparse Structural Subrole Binding` under a new
pre-registration, in which `g = 0 ⇒ FEC-D1 ≡ FEC-S1`.

---

## 13. Artifacts

```
notes/fec_d0_prior_artifact_audit.md
notes/fec_d0_preregistration.md          (this file)
notes/fec_d0_analysis.md

results/fec_d0/
    basis_audit.json
    split_fingerprint.json
    correctness.json
    residualization.json
    reconstruction.json
    dictionary_usage.json
    reuse_stability.json
    marked_topology_classes.json
    marked_topology_probe.json
    atom_semantics.json
    DECISION.md  REPORT.md  decision.json
```

plus claim YAML, decision YAML and a `STATE.yaml` update, recording commit,
dirty flag, official-train molecules used, fit/holdout fingerprint,
`official_valid_loaded=false`, `official_test_loaded=false`,
`targets_loaded=false`, `K`, `s`, dictionary solver, active atoms,
reconstruction ratio, reuse metrics, probe metrics, final verdict.
