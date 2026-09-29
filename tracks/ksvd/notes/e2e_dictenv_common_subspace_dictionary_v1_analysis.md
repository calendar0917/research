# Analysis — `e2e_dictenv_common_subspace_dictionary_v1`

Common-Subspace-Separated Structural Dictionary (CSSD) on the closed ZINC
dictionary-environment line.  CPU only, no GPU, no SSH, official ZINC test
never loaded (`official_test_loaded = false` everywhere).

* Pre-registration: `notes/e2e_dictenv_common_subspace_dictionary_v1_preregistration.md`
  — frozen before Stage A, sha256 `8e722afbc757…`, implementation freeze
  commit `5f0f284` (the runner was committed at `ed8ba38` + `34fb311`; the
  logging-only fix `5f0f284` predates the training process and changes no
  gate, threshold or training math).
* Results: `results/e2e_dictenv_common_subspace_dictionary_v1/`
  (`preflight.json`, `stage_a.json`, `common_subspace.json`,
  `zero_training/`, `training/`, `reusable_structure/`, `analysis_tables.md`,
  `summary.json`, `REPORT.md`, `DECISION.md`).
* Frozen tests: `tests/test_e2e_dictenv_common_subspace_dictionary_v1.py`,
  24/24 passing, including the bit-equivalence of `train_cssd` with the frozen
  `audit.train_cpu` loop and the projected-dictionary hard constraint.
* Transparency: the first launch of `stage-train` was killed at ~epoch 20 to
  apply the logging-only fix (`5f0f284`); that attempt wrote no checkpoint and
  no artifact, and the formal trajectory is the restarted single process.

## 0. Question

The audit round showed that the frozen sparse dictionary spends a large part
of its code budget re-representing directions that are common to the whole
dataset: atoms 6/24/27 are active in 100 % of rows and 0.76–0.98 aligned with
the train-mean direction, and atom 23 is the PC1-like atom.  The question here
is purely structural: if that common subspace is modelled explicitly
(`c = U^T x`, dense, never dropped) and the sparse dictionary only ever sees
the residual `r = (I − U U^T) x`, does the dictionary become a cleaner,
cross-graph reusable factor vocabulary — *without* changing the task model,
the coder, K/s/λ, the features, the relation, the fusion or the split?

## 1. What stayed frozen

FINAL-CLEAN unchanged: C6 mask/path, paired node and edge structure-semantic
binding, sparse tied-IHT (K=32, s=8, IHT-10), full relation, H1 optimizer
schedule, epochs 1 → 40 → 320 as **one** run with one optimizer state.
Exactly one new training trajectory (CSSD q1 seed 0).  The three reused C6
soup dictionaries and the historical DenseTied seed-0 soup number
(0.125563, context only, not a matched control) were never retrained.

## 2. Stage A — feature space

The dictionary input is re-derived from the real pipeline:

```text
train 231664 x 65 float32   (10 000 molecules)
valid  23083 x 65 float32   (1 000 molecules)
pipeline cache slice == dataset dict_phi   bit-identical on the probed molecules
no normalization between cache and dictionary (P2Model.code reads dict_phi directly)
```

Nine descriptors are identically zero: `[2, 3, 5, 7, 8, 33, 35, 48, 50]`,
including the named `root_neighbour_shell1` (col 5) and `root_walk1`
(col 8).  They are **kept**; the 65-D layout is frozen and no descriptor was
removed.

## 3. Stage B — train-only common subspace

Train-only fit (`q1`: `u1 = μ/||μ||`; `q2`: + orthogonalised PC1 of `r1`):

| quantity | q1 | q2 |
|---|---|---|
| `E_common / ||x||²` (train / valid) | 0.9772 / 0.9769 | 0.9903 / 0.9900 |
| `E_residual / ||x||²` | 0.0228 / 0.0231 | 0.0097 / 0.0100 |
| `E_centered_common` | 0.2189 | 0.6664 |
| `E_centered_residual` | **0.7811** | 0.3336 |
| train RMS `s_k` | 5.0829 | 5.0829, 0.5876 |

`||μ|| = 5.0662`.  `|cos(u1, PC1)| = 0.524`: the mean direction is *not* PC1
(PC1 = 57.7 % of centred variance); it is the DC/patch-scale direction.  Raw
energy is DC-dominated (97.7 %), so condition C uses the centred residual
fraction: q1 keeps 78.1 % (train) / 78.5 % (valid) of the centred structural
variation, q2 only 33.4 % (train) / 34.0 % (valid).  This is exactly the
preregistered reason to prefer q1 when it passes.

## 4. Stage C — zero-training same-`D` probe and the frozen selection

Each of the three reused C6 soup dictionaries, one identical `Dbar` per seed,
four descriptor variants encoded with the frozen `iht10`, valid split shown
(train agrees; full table in `analysis_tables.md` §2):

| seed | variant | top-5 | `N_eff` | max rate | rate 6 | rate 24 | rate 27 | rate 23 |
|---|---|---|---|---|---|---|---|---|
| 0 | RAW | 4.796 | 14.52 | 1.000 | 1.000 | 1.000 | 1.000 | 0.977 |
| 0 | MC | 2.694 | 21.13 | — | 0.520 | 0.393 | 0.510 | 0.602 |
| 0 | Q1 | 2.922 | 19.89 | — | **0.032** | **0.128** | **0.069** | 0.608 |
| 0 | Q2 | 3.205 | 18.99 | — | 0.000 | 0.000 | 0.088 | 0.100 |
| 1 | Q1 | 3.167 | 19.22 | — | 0.052 | 0.258 | 0.402 | 0.811 |
| 2 | Q1 | 3.353 | 20.42 | — | 0.029 | 0.155 | 0.246 | 0.904 |

Frozen rule on valid: Q1 satisfies A (≥ 2 of {6,24,27} below 0.95 — measured
3/3 on every seed), B (top-5 drop ≥ 0.4 — measured 1.87 / 1.39 / 1.06) and C
(centred residual ≥ 0.5 — 0.7851) on **3/3** dictionaries; Q2 also passes A∧B
and D2 (atom 23 ratio 0.103 / 0.378 / 0.192) but keeps far less centred
variation.  Selection: **q1** (`train_valid_agree = true`).  q2 was stored as
a diagnostic only; no q2 trajectory exists.

Three facts worth recording because they shape the later reading:

1. **Mean-centering alone is not the mechanism.**  MC already lowers the
   triplet (1.000 → 0.39–0.56) and top-5 (4.80 → 2.69), i.e. part of the
   concentration is the non-zero mean magnitude along `u1`.
2. **The residual direction is what frees the atoms.**  Q1 collapses the
   triplet to 0.03–0.40 while atom 23 stays 0.60–0.90: atom 23's role is
   *not* the common direction, it is genuine residual structure (PC1-like).
3. **Q1 is not a re-code of the raw dictionary.**  Support Jaccard / code
   cosine vs RAW are in `zero_training/comparison.csv`; the codes change
   structure, not just amplitudes.

## 5. Training — epoch-40 gate and the single trajectory

`stage_train` ran CSSD q1 seed 0, 320 epochs, one optimizer state (39.5 min,
CPU, 4 threads).  The epoch-40 gate was evaluated inside the running process
on the live model/optimizer (eval-mode gradient probe, so the callback cannot
perturb the training RNG stream):

```text
condition A  dc_count_gt095 = 0        (reference 3)          PASS
condition B  top-5 2.746 vs 4.796 (drop 2.050 >= 0.4)         PASS
             N_eff 22.725 vs 14.520 (gain 8.205 >= 1.0)       PASS
             weighted Spec ratio 1.320 >= 1.2                  PASS
condition C  ||dL/dD|| = 6.647e-2 > 1e-8, column min 0.317    PASS
             max |U^T Dbar_perp| = 4.2e-16 (hard constraint)
catastrophic false (train MAE 0.1930, valid 0.2108)
verdict CSSD_CONTINUE
```

Final run: best valid `0.136913 @ 307`, soup (5 members, 282–313)
**valid MAE 0.130028**; 97 727 params (FINAL-CLEAN 97 487 + 240 = exactly the
preregistered q1 count), 0 dead-column fallbacks.  Reference FINAL-CLEAN
sparse seed-0 soup `0.128499`: `M_CSSD − REF = +0.001529`, inside the frozen
neutral band `|Δ| ≤ 0.003` → `CSSD_REPRESENTATION_SUPPORTED /
TASK_NEUTRAL_SINGLE_SEED`.  The "no representation gain" clause does not fire
(0 universal residual atoms; top-5 3.007 ≪ 4.796; weighted Spec +0.051).
The reported `train_rec` in `curve.csv` is the frozen-loop full-descriptor
diagnostic (`||x − r̂||²/||x||² ≈ 0.976`, i.e. ≈ the common fraction); the
optimised term is `train_rec_term` (4.9e-5 at epoch 320).

## 6. Stage G — reusable-structure audit (CSSD soup vs RAW seed-0)

### 6.1 Usage and health (valid)

| metric | RAW seed 0 | CSSD seed 0 |
|---|---|---|
| active atoms | 27 / 32 | **32 / 32** |
| `N_eff` | 14.52 | **21.39** |
| top-5 usage (of 8) | 4.796 | **3.007** |
| max activation rate | 1.000 | **0.796** (atom 19) |
| atoms rate > 0.95 | 4 | **0** |
| atoms rate > 0.50 | — | 3 |
| usage-weighted Spec | 0.2067 | **0.2580** |
| median Spec | 0.564 | 0.373 |
| dictionary coherence mean / max | 0.182 / 0.987 | 0.206 / **0.780** |

Every CSSD atom is alive: minimum graph coverage 0.0126 (126 / 10 000 train
graphs) against 5 never-active RAW atoms (coverage 0.000).  Coverage median
0.9895.  The dictionary is therefore usable as a vocabulary: no atom is dead,
none is universal, and the coherence maximum (near-duplicate atoms) drops
from 0.987 to 0.780.

### 6.2 Graph-level reuse (post-hoc descriptive, `|α|` mass)

| split | representation | effective atoms / graph (mean) | atoms ≥ 5 % mass | max atom share |
|---|---|---|---|---|
| train | RAW | 6.14 | 4.01 | 0.278 |
| train | CSSD | **16.69** | **7.67** | **0.133** |
| valid | RAW | 6.15 | 4.01 | 0.278 |
| valid | CSSD | **16.66** | **7.71** | **0.133** |

Train and valid agree to the second decimal (no overfitting of the usage
structure).  Each graph's activation budget is spread over ~2.7× more atoms
and no atom carries more than 13 % of a graph's mass.  Caveat in the same
table: the graph argmax is still often the same broad atom — RAW: atom 6 in
**100 %** of train graphs; CSSD: atom 19 in 75.5 %, atom 9 in 16.5 %, the
rest ≤ 2.6 %.  Breadth improved; a single generalist still tops most graphs.

### 6.3 Fate of the universal triplet and specialization

The near-universal RAW atoms were *exactly* the atoms pointing along the
common direction (`|cos(atom, u1)|`: atom 6 **0.965**, atom 24 **0.941**,
atom 27 **0.819**; only 3 atoms > 0.5, only 2 > 0.9; all-atom median 0.022).
After separation and training:

| atom | RAW rate | RAW `|cos(u1)|` | CSSD rate | CSSD Spec | cos(RAW, CSSD) |
|---|---|---|---|---|
| 6 | 1.000 | 0.965 | **0.010** | **0.965** | −0.10 |
| 24 | 1.000 | 0.941 | 0.495 | 0.180 | 0.21 |
| 27 | 1.000 | 0.819 | 0.366 | 0.336 | 0.44 |
| 23 | 0.977 | 0.324 | 0.409 | 0.356 | 0.58 |

The DC-triplet identity is broken, and it is broken *by construction*: the
CSSD dictionary has `max |cos(atom, u1)| = 8e-8`.  Atom 6 does not merely
drop — it becomes the **most specialized atom** (Spec 0.965 at rate 0.010),
while atom 19 becomes the broad generalist (rate 0.796, Spec 0.080).  New
rare specialists appear with extreme SMD (atoms 11/16 Spec 3.85/4.19 at rates
0.001–0.011, atom 12 4.19 at 0.011).  Note the honest counterpoint: the mean
pairwise Jaccard of the top-5 |SMD| feature sets is 0.136 over the 32 CSSD
atoms vs 0.125 over the 27 active RAW atoms — the atoms' *top-feature
profiles* are not more mutually distinct than before (28 vs 29 of 65
descriptors covered); what changed is use, not the response alphabet.

### 6.4 What the common coordinate is

`c1` is a patch-scale / root-degree coordinate (valid Spearman): `patch_nodes`
0.960, `patch_edges` 0.952, `root_induced_degree` / `root_neighbour_shell2` /
`root_walk2` 0.891, `shell_pop1` 0.882, `std_log1p_walk3` 0.829.  It is a
dense, interpretable, exactly-reconstructed coordinate — not a discarded
nuisance direction — and it is *not* PC1 (`|cos| 0.524`).  q2's second
coordinate (`s_2 = 0.588`) was never trained.

### 6.5 Sparse ↔ dense residual recoverability (train-only OLS, valid)

| direction | `R²` | per-dim median | mean cosine | CKA |
|---|---|---|---|---|
| dense residual → IHT-10 residual | 0.9709 | 0.9747 | 0.983 | 0.999 |
| IHT-10 residual → dense residual | 0.9993 | 1.0000 | 1.000 | 1.000 |
| RAW IHT-10 → CSSD IHT-10 | 0.9704 | 0.9941 | 0.988 | 0.999 |
| CSSD IHT-10 → RAW IHT-10 | 0.9813 | 0.9978 | 1.000 | 0.999 |
| [c1 ; CSSD IHT-10] → RAW IHT-10 | 0.9815 | 0.9978 | 1.000 | 0.999 |

Reference RAW seed 0 (audit round): dense→IHT-10 0.970825, IHT-10→dense
0.998845.  Separation neither destroys nor manufactures linear
recoverability.  The mutual RAW↔CSSD `R²` (0.97/0.98) is the main caveat to
any strong "new vocabulary" reading: the two sparse codes remain near-linear
re-parameterisations of the same patch, and `c1` adds nothing beyond the CSSD
code for predicting the RAW code (0.9813 → 0.9815).

## 7. Claim decision (frozen framework)

* **Claim C — "separation does not resolve concentration": refuted.**
  `N_eff` 14.52 → 21.39, top-5 4.80 → 3.01, max rate 1.000 → 0.796, atoms
  > 0.95: 4 → 0, dead atoms: 5 → 0, graph breadth ×2.7, coherence max 0.987
  → 0.780.
* **Claim A — "reusable cross-graph structural factors beyond the common
  background": supported at the usage/reuse level**, on one seed: all 32 atoms
  are used across graphs (min coverage 126 graphs, median 0.99), no universal
  atom survives, the old universal triplet is exactly the set of
  common-direction atoms and is replaced by one broad generalist plus many
  rare, sharply specialized factors.
* **Claim B — "shared coordinate, but no distinct reusable vocabulary":
  not excluded.**  The task MAE is unchanged (+0.0015), the top-5 SMD
  feature-profile diversity is unchanged (0.136 vs 0.125), and RAW↔CSSD
  codes are mutually ~0.97–0.98 linearly recoverable.  What the round shows
  cleanly is a change of *dictionary usage structure*, not a demonstrated new
  information channel.

So the round's honest headline is: **the concentration was the common
direction; removing it gives a structurally healthier, reusability-shaped
dictionary at equal task accuracy — but not (yet) a demonstrably distinct
representation.**  This is single-seed exploratory evidence only.

## 8. Limitations

* One new seed (0), one dataset, one architecture family, valid split only.
* The DenseTied `0.125563` is historical context, **not** a matched control;
  no matched dense run with the same extra parameters exists.
* No CSSD seed 1/2, no q2 training, no matched-capacity control.
* The graph-reuse, geometry, reverse-recoverability and diversity tables in
  the frozen result layout are **post-hoc descriptive**; the frozen Stage G
  items are usage, coverage, specialization, common-coordinate correlations
  and recoverability.  No frozen threshold was changed.
* `train_rec` in the curve is the full-descriptor diagnostic; use
  `train_rec_term` for the optimised residual term.
* phi65 is topology-only; no chemistry motif names are assigned to atoms.

## 9. Next-round shapes (not authorised here)

1. Matched capacity control: CSSD q1 **and** DenseTied at the same 97 727
   parameters, 3 seeds, same soup protocol — the only way to turn "neutral
   MAE" into a specificity statement.
2. A usage-structure metric as a *preregistered* gate (effective atoms per
   graph, min coverage, coherence max), since those moved by far the most.
3. Targeted test of the generalist atom (19): is a single broad atom needed,
   or does the model work with a cap on the top share?
4. Cross-seed vocabulary alignment of the CSSD atoms (Hungarian matching) to
   see whether the new rare specialists replicate.
5. If representation distinctness is the target: train the same CSSD model
   with the common coordinate *removed from the binding* (c1 as a pure
   reconstruction term) to test whether the extra width or the extra
   information carries the (unchanged) MAE.
