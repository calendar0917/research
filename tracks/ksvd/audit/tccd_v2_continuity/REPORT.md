# TCCD-v2 Representation Continuity Audit (read-only, diagnostic)

Central question: **does the TCCD-v2 representation have graded chemical continuity?**

Artifacts:
- code: `tracks/ksvd/audit/tccd_v2_continuity/run_tccd_v2_continuity_audit.py`
- raw results: `tracks/ksvd/audit/tccd_v2_continuity/tccd_v2_continuity_audit.json`
- catalog map: `tracks/ksvd/audit/tccd_v2_continuity/catalog.json`
- no training, no tuning, no official test, no writes to formal results / STATE / claims.

---

## 1. Executive conclusion

TCCD-v2 learned a **task-shaped, exact-anchored local motif vocabulary**, not a smooth
chemical manifold. On internal-dev patches it is strongly monotone across five
chemically ordered tiers (Z cosine `1.00 > 0.43 > 0.33 > 0.21 > 0.07`), so
chemistry is clearly not ignored. But the geometry is dominated by the
**exact-match cliff**: the first non-identical step (radius-1 identical, only
radius-2 differs) already consumes **61% of the total Z-space range** and
**75% of the total C-space range**. The discrete dictionary atom (`argmax`
prototype) is essentially exact-match-only: radius-1-identical pairs share the
same top-1 prototype only **9.0%** of the time (random 2.0%). A real but weak
graded residual exists (Spearman excluding Exact = 0.41 in Z, 0.37 in C;
within the same `(root, degree)` family, Spearman(radius-1 edits, Z cosine) =
-0.30 over 2.4M pairs), and it survives an element-level control. Aggregate task
organization is not supported: the prototype partition explains less target
variance than the L1 chemical family, and near-pairs that split across
prototypes show no target difference. There is a localized non-chemical
compression in a minority of prototypes (heavy-atom-different pairs sharing one
atom at Z cos 0.49–0.55), but it does not dominate the vocabulary.

**Primary verdict: B — mostly an exact/discrete motif vocabulary, with a
weak-but-real graded A component. C is ruled out as the primary organizing
principle (only a localized minority of buckets look non-chemical).** The
discrete assignment code is nearly discontinuous (Failure B), and a small
number of prototypes are genuine incoherent buckets (Failure A).

---

## 2. Audit setup

| item | value |
|---|---|
| local commit | `27937d9722f4ac596a66fa76d99397c0804e0a6d` |
| checkpoint | `tracks/ksvd/results/tccd_v2/prototype_rel_seed0_best.pt` (best, seed 0) |
| checkpoint tensors | `W (714,64)`, `P (64,64)`, `temp_logit ()`, `head.weight (1,10464)`, `head.bias (1,)` |
| records | `tracks/ksvd/results/tccd_v0/cache/tccd_v0_train_r2_M14_A21_B3_n10000.pkl` (radius-2 canonical patch records) |
| split | TCCD internal split seed 20260922: 8000 train / 2000 dev |
| graphs audited | 2000 internal-dev graphs |
| patches audited | 46126 internal-dev patches |
| unique canonical keys | 6929 |
| unique radius-1 keys | 346 |
| learned temperature | `tau = 0.094572` |
| official test | **never loaded** (and official valid never loaded) |
| training / tuning | none |
| runtime | 47 s on local CPU |

Validation against the frozen run:
`C` recomputed as `softmax(cos(z,P)/tau)` reproduces the frozen
`vocabulary_seed0.json` top-1 prototype counts **exactly** (max abs count
difference 0 / 46126); active prototypes = 64; effective count = 62.5829 vs the
frozen 62.5755 (float summation-order difference only). So the audited
representation is the formal one.

All tiers are sampled as **cross-molecule pairs** (`graph_i != graph_j`), so no
tier can be explained by two patches living in the same molecule.

Patch chemistry is decoded deterministically from the frozen 714-D coordinate
(canonical slot order: slot 0 = root, shells 0/1/2), and atom/ bond catalog
indices are mapped to the dataset's own labels via `data/ZINC/raw/atom_dict.pickle`
and `bond_dict.pickle` (e.g. `C`, `C H1`, `N H2 +`, `O -`, single/double/triple).

---

## 3. Similarity-tier definition

Let `L1` = the rooted **radius-1 labeled subgraph** (root + shell-1 atoms, atom
types, root–shell-1 bond types, shell-1↔shell-1 bonds), canonicalized by exact
brute-force permutation of the ≤4 shell-1 slots. Let `key` = the frozen
canonical radius-2 key. Chemical edit count `c1` = (# shell-1 atom-type
substitutions) + (# root-bond-type substitutions), i.e. radius-1 edits.

| tier | ordinal | deterministic definition |
|---|---:|---|
| **Exact** | 4 | same frozen canonical `key` (verified: the 714-D vectors are bit-identical) |
| **VeryNear** | 3 | `L1` identical, `key` different (difference is confined to radius-2) |
| **Moderate** | 2 | same root type and degree, `key` and `L1` different, exactly `c1 = 1` radius-1 edit |
| **HardNegative** | 1 | same root type and degree, `|n_atoms_i - n_atoms_j| <= 1`, `c1 >= 2` radius-1 edits |
| **Random** | 0 | uniform random cross-molecule patch pair, no matching |

`Exact` is the positive control. `HardNegative` matches the simple nuisance
variables root type, degree and patch size. Tier sample sizes: 120,000 pairs per
tier (pools: 5.97M Exact pairs, 60.9M VeryNear pairs, 0.94M Moderate, 0.73M
HardNegative before sub-sampling).

---

## 4. Main quantitative results

Mean (median; IQR) per tier. `Z` = normalized latent cosine; `C` = cosine of the
full 64-D soft assignment; `JS` = Jensen–Shannon **distance** (base-2); `top-1` =
argmax-prototype agreement rate.

| Pair tier | Z cosine | C cosine | C JS distance | same top-1 prototype |
|---|---:|---:|---:|---:|
| Exact | 1.000 (1.000; [1.000, 1.000]) | 1.000 (1.000; [1.000, 1.000]) | 0.000 | 1.000 |
| VeryNear | 0.428 (0.511; [0.104, 0.713]) | 0.301 (0.071; [0.009, 0.612]) | 0.720 | 0.090 |
| Moderate | 0.332 (0.339; [0.137, 0.542]) | 0.218 (0.097; [0.017, 0.345]) | 0.784 | 0.056 |
| HardNegative | 0.206 (0.182; [0.065, 0.354]) | 0.095 (0.027; [0.008, 0.097]) | 0.881 | 0.014 |
| Random | 0.066 (0.037; [-0.137, 0.235]) | 0.073 (0.006; [0.001, 0.037]) | 0.912 | 0.020 |

Reference raw canonical-X cosine (not a chemical metric, shown only as control):
`1.000 / 0.801 / 0.739 / 0.718 / 0.596` (Exact → Random).

### Monotonicity (ordinal level vs representation)

| pooled pairs | Spearman vs Z cos | vs C cos | vs C JS dist |
|---|---:|---:|---:|
| all 5 tiers | 0.695 [0.687, 0.701] | 0.678 [0.672, 0.685] | -0.684 [-0.691, -0.676] |
| excluding Exact | **0.406 [0.401, 0.411]** | **0.371 [0.365, 0.377]** | **-0.380 [-0.385, -0.373]** |

Brackets are 95% molecule-cluster bootstrap CIs (2000-graph resampling,
120 resamples). The trend is real and stable, but the rank association among
*near-but-not-identical* environments is only moderate.

### Adjacent-tier separation (how graded is it really?)

| comparison | ΔZ mean | ΔC mean | AUC(P(higher tier more similar)) |
|---|---:|---:|---:|
| Exact > VeryNear | +0.572 | +0.699 | 1.000 |
| VeryNear > Moderate | +0.096 | +0.084 | **0.588** |
| Moderate > HardNegative | +0.126 | +0.123 | **0.645** |
| HardNegative > Random | +0.140 | +0.022 | **0.669** |

Adjacent tiers overlap heavily (AUC 0.59–0.67 near chance = 0.5); only the
Exact boundary is perfectly separated. Note the C-space collapse between
HardNegative and Random: chemically different hard negatives are almost as far
apart in assignment space as random pairs (0.095 vs 0.073), even though Z-space
still separates them (0.206 vs 0.066).

### Continuous graded axes (independent of the tier construction)

- Within the same `(root, degree)` family, excluding exact pairs (2,435,802
  pairs): Spearman(radius-1 edits, Z cosine) = **-0.298**, Spearman(radius-1
  edits, C cosine) = **-0.230**. Binned Z cosine (full atom types): `c1=0:
  0.427 / c1=1: 0.333 / c1=2: 0.204 / c1>=3: 0.137`.
- **Element-level control** (collapse `C H1`, `N H2 +`, `O -`, … to base
  elements, same element-root and degree): Z cosine `0.413 / 0.304 / 0.180 /
  0.162` for `0 / 1 / 2 / >=3` heavy-atom radius-1 edits. The gradient is not an
  artifact of hydrogen/charge decorations.
- Within `L1`-identical pairs (VeryNear), radius-2 edits give Z cosine
  `d2=1: 0.563 / d2=2: 0.462 / d2=3-4: 0.260 / d2>=5: 0.344` (last bin n=5010
  and non-monotone). So even radius-2 does carry a gradient, but only the first
  one to two edits are cleanly ordered.

---

## 5. Exact vs near-nonidentical gap

| quantity | Z-space | C-space |
|---|---:|---:|
| Exact similarity | 1.000 | 1.000 |
| VeryNear (radius-1 identical) | 0.428 | 0.301 |
| Random | 0.066 | 0.073 |
| **gap Exact → VeryNear** | **0.572** | **0.699** |
| gap VeryNear → Moderate | 0.096 | 0.084 |
| gap Moderate → HardNegative | 0.126 | 0.123 |
| gap HardNegative → Random | 0.140 | 0.022 |
| fraction of the full Exact→Random range lost at the first non-identical step | **61.3%** | **75.3%** |
| argmax top-1 agreement Exact → VeryNear | 1.000 → 0.090 | — |

Interpretation: the representation does **not** collapse to random as soon as the
canonical key differs (Z cosine 0.43 vs random 0.07), so it is not a pure lookup
table. But the distance travelled at the exact/non-exact boundary is ~6x the
VeryNear→Moderate step, and the discrete atom identity is almost pure
exact-match. The graded component is real but secondary. This is the central
evidence for **B over A**.

---

## 6. Prototype chemical coherence

Argmax-assigned patches per prototype (64 prototypes). Baselines: matched random
= same `(root, degree)` draw per assigned patch; uniform random = 2000 random
patches; random-pair baseline from 200k cross-molecule random pairs.

| statistic | prototypes | matched random | uniform random |
|---|---:|---:|---:|
| modal exact-key share (mean over prototypes) | **0.266** | 0.102 | 0.047 |
| modal radius-1-family share | **0.518** | 0.377 | 0.200 |
| modal root-type share | 0.881 | 0.881 | 0.701 |
| modal degree share | 0.901 | 0.901 | 0.502 |
| within-prototype pair exact-key equality | 0.157 | — | 0.0055 |
| within-prototype pair radius-1 equality | 0.387 | — | 0.0573 |

The asymmetry is the key result: prototypes concentrate **exact canonical keys
~28x above random and ~2.6x above the matched baseline**, but radius-1 *families*
only **~6.8x above random and ~1.37x above matched**. In other words the
vocabulary is much better at remembering an exact motif than at covering a
chemical family. Mean within-prototype radius-1 edit count is 0.81; mean
within-prototype soft C cosine is 0.85.

**Failure A (same prototype, clearly different chemistry)** is real but
localized: the 64 prototypes split into a well-formed majority and a minority of
incoherent task buckets. Examples (see §7): proto 27 (`l1_conc = 0.09`,
mean radius-1 edits 2.3) mixes `C(F)(F)(F)` with `S(=O)(=O)(N)(N)`; proto 17
mixes 3-coordinate nitrile/amine carbon with 4-coordinate ether carbon;
proto 7 collapses `O-`–C (alkoxide-like) and `O=`S (sulfoxide/sulfone oxygen).

**Failure B (near-identical chemistry split across prototypes)** is the
dominant failure mode: only 9.0% of radius-1-identical pairs share the same
top-1 prototype; some of them have *negative* Z cosine (see §7a).

---

## 7. Representative examples

Notation: `single`/`double` bond types, `s1` = shell-1 neighbours,
`s2` = shell-2 atom types, `proto` = argmax prototype, `y` = molecule target.

### 7a. Chemically near, representation far (Failure B)

All four pairs have **identical radius-1 chemistry** (`root=C`, deg 2, two
`single-C` neighbours) and differ only at radius-2.

| A | B | Z cos | C cos |
|---|---:|---:|---:|
| g2831 y=-0.88 s2=[C,C,N] proto=30 | g3252 y=-0.97 s2=[N H2 +] proto=24 | **-0.318** | 0.000 |
| g617 y=-5.43 s2=[C,C,N] proto=30 | g8746 y=+2.22 s2=[N] proto=24 | **-0.303** | 0.000 |
| g7194 y=+2.49 s2=[C] proto=24 | g2999 y=-1.02 s2=[C,C,C,C H1] proto=3 | **-0.347** | 0.000 |
| g4816 y=-1.78 s2=[C,C,C,N H2 +] proto=3 | g3252 y=-0.97 s2=[N H2 +] proto=24 | **-0.365** | 0.000 |

Identical radius-1 chemistry, opposite latent directions, disjoint assignment.

### 7b. Chemically near, representation near (the weak A component)

| A | B | Z cos | C cos |
|---|---:|---:|---:|
| g1770 y=+1.01 root=C deg1 [single-N] s2=[C,C] proto=59 | g5549 y=+1.19 s2=[C,C H1] proto=59 | 0.953 | 0.999 |
| g4997 y=+1.00 root=C deg2 [single-C, single-C] s2=[C,N] proto=42 | g6062 y=-1.61 s2=[C,N H1 +] proto=42 | 0.941 | 1.000 |
| g7217 y=-0.20 root=C deg2 [single-C, single-N H1 +] s2=[C,C,C,C] proto=63 | g3021 y=-1.81 s2=[C,C,C,O] proto=63 | 0.935 | 1.000 |
| g1483 y=+2.53 root=C deg2 [single-C, double-C] s2=[C,C,O] proto=45 | g132 y=+0.64 s2=[C,C,N +] proto=45 | 0.933 | 0.999 |

Single radius-2 edits do produce close latent and identical prototype codes.

### 7c. Chemically different, representation close (compression / coherence loss)

Element-different pairs (`heavy_c1 = 2`) with the highest C cosine:

| A | B | Z cos | C cos | same proto |
|---|---:|---:|---:|---:|
| g7011 y=-4.53 root=`O -` deg1 `single-C` s2=[C H1,O] | g9506 y=-1.22 root=`O` deg1 `double-S` s2=[C,C H1] | 0.701 | 1.000 | yes (7) |
| g7014 y=-1.85 root=`O` deg1 `double-S` | g5541 y=-10.22 root=`O -` deg1 `single-C` | 0.797 | 1.000 | yes (7) |
| g4882 y=-3.97 root=C `[single-C, single-N H2 +]` proto=63 | g6251 y=-0.93 root=C `[single-Br, single-C H1]` proto=63 | 0.722 | 0.997 | yes |

The assignment code saturates: chemically distinct heavy-atom environments can
land on the same prototype with C cosine ≈ 1 even when Z cosine is only
0.55–0.80.

### 7d. Failure A prototypes (chemically incoherent buckets)

| prototype | size | L1 conc. | key conc. | worst pair | Z cos |
|---|---:|---:|---:|---|---:|
| 27 | 684 | 0.09 | 0.05 | g6134 `C(F)(F)(F)(C)` vs g8659 `S(=O)(=O)(N)(N)` (6 radius-1 edits) | 0.530 |
| 57 | 401 | 0.12 | 0.11 | g6851 `C(=C)(C)` vs g1657 `C(=O)(C H1)(O -)` (3 edits) | 0.615 |
| 17 | 883 | 0.21 | 0.03 | g763 `C(=N H1 +)(C)(N)` vs g3810 `C(C)(C)(C)(O)` (4 edits) | 0.528 |
| 63 | 695 | 0.22 | 0.09 | g8717 `C(C H1)(S)` vs g1095 `C(C)(F)(F)(F)` (4 edits) | 0.544 |
| 32 | 257 | 0.23 | 0.05 | g7165 `C(=C)(C)(N)` vs g7824 `C H1(C H1)(C H1)(O)` (4 edits) | 0.492 |

Note: these chemically-diverse pairs are *more* Z-similar than the population
mean for their edit level (`c1h>=3` population Z cos = 0.16, `c1h=2` = 0.18).
So a minority of prototypes does apply non-chemical compression, but it is
localized: population-wide, heavy-atom-different pairs share a prototype only
0.9–1.0% of the time.

### 7e. Task-organization check

| diagnostic | value |
|---|---:|
| R² of target `y` from argmax prototype (64 groups) | 0.075 |
| R² from radius-1 family (346 groups) | 0.114 |
| R² from canonical key (6929 groups) | 0.299 |
| R² from root+degree (28 groups) | 0.032 |
| R² prototype **increment** beyond L1 chemistry | 0.038 |
| mean `|Δy|` for VeryNear pairs that split prototypes | 1.8816 |
| mean `|Δy|` for VeryNear pairs that do not split | 1.8813 |

Properly pair-count-weighted same-`(root,degree)` sample (2.0M cross-molecule
pairs), split by the number of **element-level** radius-1 edits. Same-prototype
pairs are slightly closer in target at low edit counts, but the effect vanishes
for chemically diverse pairs:

| element edits | n | same-prototype rate | Z cos | mean `|Δy|` same proto | mean `|Δy|` diff proto |
|---|---:|---:|---:|---:|---:|
| 0 | 566,578 | 0.164 | 0.460 | 1.883 | 1.973 |
| 1 | 794,315 | 0.045 | 0.302 | 1.989 | 2.207 |
| 2 | 566,728 | 0.009 | 0.180 | 2.014 | 2.226 |
| >=3 | 63,591 | 0.010 | 0.162 | 2.170 | 2.051 |

Prototypes are *less* target-aligned than the chemical families they sit on
(R² 0.075 vs 0.114), the prototype increment beyond chemistry is only 0.038,
and splitting a near-pair across prototypes is uncorrelated with the target
difference (Δy 1.8816 vs 1.8813). Same-prototype pairs are weakly closer in
target only at low chemical edit counts, and the association disappears for
element-level edit >= 3. **No support for H3 as the organizing principle.**
(R² is descriptive; patches of one molecule share `y`, so group counts and
molecule clustering inflate all of these. The split/non-split comparison is the
clean part.)

---

## 8. Scientific verdict

**Main judgment: Verdict B (mostly exact/discrete motif vocabulary), with a
weak but genuine graded-coordinate component. Verdict C is ruled out.**

Evidence for B:
1. 61% (Z) / 75% (C) of the representation range is consumed by the exact →
   non-identical boundary.
2. Discrete prototype identity is essentially exact-match-only: 9.0% argmax
   agreement for radius-1-identical pairs, 1.4% for matched hard negatives
   (random 2.0%).
3. Adjacent non-exact tiers overlap heavily (AUC 0.588 / 0.645 / 0.669); many
   radius-1-identical pairs have negative Z cosine.
4. Prototypes remember exact keys far better than chemical families (2.6x vs
   1.37x over the matched baseline); a minority of prototypes are incoherent
   buckets.
5. C-space saturates between hard negative and random (0.095 vs 0.073), so the
   assignment code throws away most of the remaining graded latent geometry.
6. A localized minority of prototypes (e.g. 27, 17, 63, 32) compresses
   chemically very different environments into one atom at Z cos 0.49–0.55
   (population mean for that edit level 0.16), which looks non-chemical.

Evidence against pure B (the A component):
1. All tiers are strictly ordered in mean Z, C and JS, with cluster-bootstrap
   CIs that exclude flatness.
2. Excluding exact pairs, Spearman is 0.41 (Z) / 0.37 (C); within a fixed
   `(root, degree)` family, more radius-1 edits stably predict lower Z cosine
   (-0.30 over 2.4M pairs), and the effect survives an element-level control.
3. Single radius-2 edits within an identical radius-1 environment give
   Z cosine up to 0.95, so the latent space does interpolate locally.

**Which problem dominates?**

- **(1) local dictionary geometry is not good enough — directly evidenced
  here.** The learned dictionary atom behaves like an exact motif label, not a
  graded chemical family. This is a real representational defect for a
  "reusable local environment" program.
- **(2) composition loses information — not tested by this audit.** Existing
  TCCD evidence says composition is heavily used (assignment-shuffle control
  0.839 vs 0.286, `Δ_comp = 0.553`) and the prototype arm actually beat the
  matched dense arm on internal dev (`Δ_proto = -0.125`). Later rounds tried
  higher-order assembly (v4), nonlinearity placement (v5), post-gain
  decomposition (v6) and explicit normalized moments (v7); none recovered
  standalone strength, and v7 closed the TCCD standalone performance route.
- **Synthesis:** this audit shows the local dictionary is under-smooth, and an
  exact-anchored local code is a plausible upstream cause of a composition
  ceiling (assembly can only compose the discrete atoms it is given). It does
  **not** show that fixing the local dictionary will fix the absolute ZINC gap,
  and it adds no new evidence about the composition operator itself. Both
  concerns are live; (1) is now evidenced, (2) remains inferred from older
  rounds.

---

## 9. One next experiment only

**A single paired-arm test of whether the exact-anchored geometry is caused by
the input encoding or by the task-only objective.** No architecture redesign,
no new reader, one seed.

- **Frozen:** records, split, `714 → 64` linear encoder class, `K = 64`
  prototypes, cosine + trainable temperature softmax, task loss, entropy and
  balance regularizers, `C^T R C` reader, optimizer, stopping rule, seed 0.
- **Treatment (one added term, nothing else):** a label-free chemical-continuity
  regularizer on the latent, with a fixed precomputed kernel over training pairs
  `w_ij = 1` when radius-1 is identical (VeryNear), `0.5` when `c1 = 1`
  (Moderate), `0` otherwise; exact pairs excluded.
  `L_chem = mean_ij [ w_ij * (1 - cos(z_i, z_j)) ]`,
  `L = L_task + λ_local L_local + λ_balance L_balance + λ_chem L_chem`, with
  `λ_chem` calibrated once by the same detached-initial-magnitude rule already
  used for the other two terms (0.05 initial share).
- **Control:** identical code with `λ_chem = 0` under the same execution regime.
- **Primary readout (representation, not score):** Exact→VeryNear Z gap,
  Spearman(level, Z cos) excluding Exact, within-family Spearman(`c1`, Z cos),
  and VeryNear argmax agreement. **Guardrail:** internal-dev MAE must stay
  within the matched control's seed noise.
- **Decision rule:** if continuity improves materially at preserved MAE, the
  exact anchoring is objective-induced and a continuity term is a viable fix;
  if continuity barely moves, the binding cause is the 714-D canonical slot
  encoding itself, and the next round should replace the local input encoding
  rather than add another loss term.
