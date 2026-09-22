# TCCD-v2 GRAD-CONT — ordinal-only local geometry: graded manifold or relocated cliff?

Pre-registration: `tracks/ksvd/notes/tccd_v2_gradcont_preregistration.md`
(commit `423b703`). Shared chemistry/ordinal code:
`tracks/ksvd/code/tccd_v2_local_chem.py`. New-arm code:
`tracks/ksvd/code/tccd_v2_gradcont.py`, `tracks/ksvd/code/run_tccd_v2_gradcont.py`,
tests `tracks/ksvd/tests/test_tccd_v2_gradcont.py`. Read-only audit:
`run_tccd_v2_gradcont_continuity_audit.py` → `tccd_v2_gradcont_continuity_audit.json`.

One arm, seed 0, one λ_grad, no sweeps. BASE and CHEM-CONT were **not** rerun.
Official test and official valid were never loaded.

---

## 1. Executive conclusion

**No. An ordinal-only objective — with no absolute similarity value, no scalar
chemical distance and no shell weighting — does not produce a graded local
chemical geometry. It produces a relocation and sharpening of the same cliff,
now at the radius-2 boundary `d2=1 → d2=2`.**

Ordinal ranking accuracy rises a lot (Z dev ordinal accuracy `0.716 → 0.848`,
paired CI `[+0.123, +0.142]`), but every smoothness diagnostic gets **worse**:

* the Exact→(0,1) cliff shrinks (`0.444 → 0.259`), then a **new** cliff appears:
  the Z cosine drops `+0.741 → −0.038` across the single boundary
  `(d1=0,d2=1) → (d1=0,d2=2)`;
* the maximum adjacent-bin jump rises to `0.786`
  (BASE `0.405`, CHEM-CONT `0.594`);
* the maximum largest-boundary share rises to `1.075`
  (BASE `0.892`, CHEM-CONT `1.032`) because the curve is now V-shaped;
* only `2/6` fixed-axis slices are monotone non-increasing;
* the `d1=0` slice is explicitly non-monotone:
  `0.977 → 0.741 → −0.038 → 0.138 → 0.177 → 0.259 → 0.246`.

The model satisfied the dominant training comparison (`(0,1) ≺ (0,2)`, 80 % of
all triplets) by pushing the `d2=2` shell into its own region and overshooting,
instead of interpolating chemistry. **Pre-registered outcome: B — the ordering
was learned, the geometry is still a step function.** Dictionary coding is not
authorized.

---

## 2. Data support — Phase 0 ordinal coverage (zero training)

Phase 0 ran before training (`coverage_seed0.json`, same commit). Structure-only
chemistry, cross-molecule, training split for training triplets, held-out
internal dev for scoring; no label `y` read.

| statistic | train | held-out dev |
|---|---:|---:|
| anchors total | 185538 | 46126 |
| eligible anchors | 185488 | 46107 |
| eligible fraction | **0.99973** | **0.99959** |
| ordinal triplets | **741774** | 184332 |
| unmatched fraction | 0.00027 | 0.00041 |
| graph coverage | 1.000 | 1.000 |
| cross-molecule anchor↔closer / closer↔farther | 1.000 / 1.000 | 1.000 / 1.000 |
| comparison types same_d1 / same_d2 / both | 650603 / 82052 / 9119 | 159917 / 21049 / 3366 |

Most common train comparisons: `(0,1)≺(0,2)` 590168 (79.6 %),
`(0,1)≺(1,1)` 70172, `(0,2)≺(0,3)` 24011, `(1,0)≺(1,1)` 18652,
`(0,1)≺(1,2)` 7053. Every `d1∈{0..4}`, `d2∈{0..6}` candidate bin is populated.

Coverage stop rule (`eligible ≥ 0.50`, `triplets ≥ 10000`, `cross ≥ 0.99`):
**PASS.** The ordinal relation is data-supported; the round proceeds. The strong
dominance of a single comparison `(0,1)≺(0,2)` is itself recorded here and turns
out to be the mechanistic key to §5.

---

## 3. GRAD-CONT arm (seed 0)

Identical to frozen TCCD-v2 Prototype-REL except one added term on `z`:

    L_grad = mean softplus( cos(z_i, z_far) − cos(z_i, z_close) )

over Pareto triplets `close ≺ far` (`j ≺ k` iff `d1(i,j) ≤ d1(i,k)` and
`d2(i,j) ≤ d2(i,k)` with one strict; exact pairs excluded; cross-molecule; root
type+degree matched; size within ±2; at most 4 triplets/anchor; farthest bin
capped at `d1 ≤ 3, d2 ≤ 5`). No scalar distance, no shell weight, no margin.

λ calibrated once, detached, on the first 32 training graphs:

| quantity | value |
|---|---:|
| initial task loss | `1.5957586` (identical to frozen calibration) |
| initial ordinal loss | `0.6913669` |
| λ_grad | `0.1154061` |
| initial weighted contribution | `0.0797879` = 5.00 % of task |

Run: tag `gradcont-s0`, GPU1, commit `423b703`, seed 0, wall `1319 s`, early stop
epoch 215, best epoch 175. Checkpoint best MAE `0.299083`; soup `0.275521`.
Usage after training: 64/64 active, effective `61.06`, τ `0.09495`.
`gate0` passed on GPU1 (all 7 boolean checks).

---

## 4. Did Z become graded? Full geometry curve, not just Spearman

All numbers are on one shared, **uniform-over-pairs** internal-dev pair set
(1.7 M pairs) and one shared held-out triplet set (184 332), so the three
checkpoints are exactly paired. `n` per row is identical across checkpoints.

### 4.1 `(d1,d2)` curve — mean Z cosine (selected cells)

| d1,d2 | n | BASE | CHEM-CONT | GRAD-CONT |
|:--|--:|--:|--:|--:|
| 0,0 (exact) | 211686 | +0.961 | +0.989 | +0.977 |
| 0,1 | 112202 | +0.556 | +0.825 | **+0.741** |
| 0,2 | 133898 | +0.462 | +0.750 | **−0.038** |
| 0,3 | 65592 | +0.295 | +0.627 | +0.138 |
| 0,4 | 36605 | +0.220 | +0.630 | +0.177 |
| 0,5 | 11537 | +0.341 | +0.504 | +0.259 |
| 0,6 | 2748 | +0.303 | +0.466 | +0.246 |
| 1,1 | 36899 | +0.475 | +0.343 | +0.522 |
| 1,2 | 113359 | +0.380 | +0.267 | +0.308 |
| 1,3 | 170305 | +0.341 | +0.237 | +0.197 |
| 2,1 | 10354 | +0.312 | −0.246 | +0.302 |
| 2,2 | 53590 | +0.266 | −0.327 | +0.273 |
| 2,3 | 147528 | +0.194 | −0.422 | +0.122 |
| 3,4 | 21343 | +0.159 | −0.445 | +0.087 |

The `(0,2)` cell median for GRAD-CONT is `−0.232`: a substantial fraction of
`(d1=0,d2=2)` pairs are strongly negative.

### 4.2 Fixed-axis slices (Z mean cosine)

Fix `d1`, sweep `d2`:

| slice | BASE | CHEM-CONT | GRAD-CONT |
|:--|:--|:--|:--|
| `d1=0` | 0.961, 0.556, 0.462, 0.295, 0.220, 0.341, 0.303 | 0.989, 0.825, 0.750, 0.627, 0.630, 0.504, 0.466 | **0.977, 0.741, −0.038, 0.138, 0.177, 0.259, 0.246** |
| `d1=1` | 0.625, 0.475, 0.380, 0.341, 0.307, 0.275, 0.245 | 0.547, 0.343, 0.267, 0.237, 0.264, 0.252, 0.254 | 0.622, 0.522, 0.308, 0.197, 0.166, 0.160, 0.149 |
| `d1=2` | 0.461, 0.312, 0.266, 0.194, 0.196, 0.193, 0.185 | −0.198, −0.246, −0.327, −0.421, −0.314, −0.227, −0.068 | 0.439, 0.302, 0.273, 0.122, 0.061, 0.068, 0.089 |

Fix `d2`, sweep `d1`:

| slice | BASE | CHEM-CONT | GRAD-CONT |
|:--|:--|:--|:--|
| `d2=1` | 0.556, 0.475, 0.312 | 0.825, 0.343, −0.246 | 0.741, 0.522, 0.302 |
| `d2=2` | 0.462, 0.380, 0.266, 0.259 | 0.750, 0.267, −0.327, −0.214 | **−0.038, 0.308, 0.273, 0.340** |

Audit-comparable sub-curves:

| slice | BASE | CHEM-CONT | GRAD-CONT |
|:--|:--|:--|:--|
| l1-identical, `d2=0..6` | 0.961, 0.556, 0.462, 0.294, 0.214, 0.351, 0.304 | 0.989, 0.826, 0.753, 0.632, 0.638, 0.514, 0.472 | 0.977, 0.742, −0.044, 0.133, 0.170, 0.257, 0.246 |
| radius-1, `c1=0..4` | 0.347, 0.336, 0.205, 0.138, 0.200 | 0.477, 0.261, −0.306, −0.389, −0.073 | 0.338, 0.227, 0.116, 0.079, 0.207 |

GRAD-CONT makes the radius-1 (`c1`) axis smoother/monotone-ish and removes the
CHEM-CONT negative push there, but the radius-2 (`d2`) axis — the family the
ordinal loss actually optimized — becomes a sharp V with a new boundary.

### 4.3 Cliff / smoothness metrics (pre-registered)

| metric (Z, internal-dev) | BASE | CHEM-CONT | GRAD-CONT | bar for A |
|---|---:|---:|---:|---:|
| median adjacent jump | 0.079 | 0.108 | 0.101 | — |
| **max adjacent jump** | 0.405 | 0.594 | **0.786** | ≤ 0.445 |
| **max largest-boundary share** | 0.892 | 1.032 | **1.075** | ≤ 0.774 |
| monotone non-increasing slices | 3/6 | 1/6 | **2/6** | ≥ 3/6 |

Largest boundary moved: BASE `(0,0)→(0,1)`, CHEM-CONT `(1,1)→(2,1)`/
`(0,1)→(1,1)`, GRAD-CONT `(0,1)→(0,2)`.

### 4.4 Anomalous-negative guardrail

| metric (Z) | BASE | CHEM-CONT | GRAD-CONT |
|---|---:|---:|---:|
| minimum bin mean (N ≥ 200) | +0.093 | −0.445 | **−0.038** @ `0,2` |
| bins (N ≥ 200) with negative mean | 0 | 16 | **1** |
| far-bin mean | +0.208 | −0.052 | +0.113 |
| random-pair mean | +0.066 | +0.050 | +0.049 |

Unlike CHEM-CONT, GRAD-CONT does **not** push chemically different environments
into a globally negative region. The damage is localized: one valley at `d2=2`.
That is exactly why the largest-boundary share is inflated (the curve recovers
after the valley), and it is the signature of a rule boundary, not a manifold.

---

## 5. Ordinal accuracy (held-out dev triplets)

`P(cos(i,close) > cos(i,far))`, ties 0.5; molecule-cluster bootstrap CIs.

| stratum | n | BASE | CHEM-CONT | GRAD-CONT | Δ(G−best) |
|:--|--:|--:|--:|--:|--:|
| all | 184332 | 0.699 | 0.716 | **0.848** | **+0.132** [+0.123, +0.142] |
| same `d1` (sweep `d2`) | 159917 | 0.696 | 0.692 | 0.858 | +0.166 |
| same `d2` (sweep `d1`) | 21049 | 0.691 | 0.881 | 0.764 | **−0.117** |
| both coordinates better | 3366 | 0.892 | 0.843 | 0.910 | +0.067 |

GRAD-CONT improves the dominant `same_d1` stratum strongly but **regresses on the
`same_d2` (radius-1) stratum** relative to CHEM-CONT. The headline `+0.132` is
driven by the comparison family the loss saw 80 % of the time — the same family
whose farther bin (`0,2`) becomes the new cliff. This is the pre-registered
Outcome-B pattern: high ordering accuracy bought by one rule boundary.

---

## 6. Z vs C

| metric | BASE | CHEM-CONT | GRAD-CONT |
|---|---:|---:|---:|
| C ordinal accuracy (all) | 0.643 | 0.686 | 0.771 |
| C ordinal accuracy (same_d1) | 0.645 | 0.664 | 0.774 |
| `(0,1)` C cosine | 0.467 | 0.554 | 0.531 |
| `(0,1)` top-1 prototype agreement | 0.147 | 0.257 | **0.329** |
| active prototypes / effective | 64 / 62.58 | 64 / 62.39 | 64 / 61.06 |

C follows Z only partially: Z ordinal accuracy `+0.132` transmits to C `+0.085`,
and `(0,1)` top-1 agreement reaches 0.329 (still low). The Z/C decoupling
reported for CHEM-CONT persists. C was deliberately not trained or repaired.

---

## 7. Task guardrail (not a tuning signal)

| metric (internal-dev) | BASE | CHEM-CONT | GRAD-CONT |
|---|---:|---:|---:|
| best-checkpoint MAE | 0.286244 | 0.302059 | `0.299083` (**+0.012839**) |
| Top-5 soup MAE | 0.266352 | 0.275267 | `0.275521` (**+0.009169**) |

Best-checkpoint degradation is marginally above the repo's 0.010 materiality
floor; soup degradation is just below it. λ_grad was fixed before the run and was
**not** retuned after seeing MAE.

---

## 8. Outcome (pre-registered classification)

Machine-computed in `verdict` of the audit JSON:

| criterion | result |
|---|---|
| (a) ordinal accuracy ≥ best baseline + 0.03 with CI > 0 | **PASS** (+0.132, CI [+0.123, +0.142]) |
| (b) max jump ≤ 0.445 and max boundary share ≤ 0.774 | **FAIL** (0.786 / 1.075) |
| (c) ≥ 3/6 monotone fixed-axis slices | **FAIL** (2/6) |
| (d) no bin (N≥200) with mean Z cosine < −0.10 | PASS (−0.038) |
| `mae_ok` (best ≤ +0.010 or soup ≤ +0.010) | PASS (soup +0.0092) |

**Primary outcome: B — the ordinal ordering is learned, but the geometry is still
a step function with a moved boundary.** `graded_Z = False`, `(a) = True`.

This closes the reframing route opened by CHEM-CONT: re-specifying the local
similarity (binary → ordinal) changes *which* boundary absorbs the geometry but
does not produce a graded manifold. Together with CHEM-CONT it shows the effect
is not the choice of binary vs ordinal supervision, and CHEM-CONT already ruled
out the `714 → 64` encoder as the binding limit. The remaining explanation is
that a ranking/contrastive auxiliary objective on this frozen architecture
converges to piecewise-constant separation, over-satisfying the dominant
comparison (`(0,1)≺(0,2)`, 80 % of training mass) by pushing its farther bin into
a separate region.

---

## 9. One next step (Outcome B)

**Do not touch prototype or composition, and do not start `Z → C` dictionary
coding.** Before any further auxiliary latent loss, fix the *local environment
definition* itself under a new pre-registration: the current `(d1,d2)` shell-edit
notion is dominated by one comparison and is not the substrate the task wants —
redefine the local environment (e.g. matched composition/size local boundaries,
or a task-relevant local boundary) and re-measure whether a graded geometry is
even definable before adding any continuity term.

---

## 10. Provenance

* pre-registration commit `423b703`; training commit `423b703` (clean worktree).
* run tag `gradcont-s0`, GPU1 `NVIDIA A100-SXM4-40GB`, CUDA `12.4`, torch
  `2.5.1+cu124`, seed 0, `λ_grad = 0.1154061`, wall `1319 s`, peak `588 MB`.
* triplet cache fingerprint `ab06179048c6db44…` (train),
  `c88b37723966e467…` (dev); dev triplets reused unchanged by the audit.
* BASE / CHEM-CONT read from frozen checkpoints and frozen result JSONs; no
  frozen artifact was modified.
* official test loaded: **False**; official valid loaded: **False**.
