# Pre-registration — `e2e_dictenv_common_subspace_dictionary_v1`

Round: **Common-Subspace-Separated Structural Dictionary (CSSD)** on the
closed ZINC dictionary-environment line.  Protocol id:
`e2e_dictenv_common_subspace_dictionary_v1`.

**Frozen before any CSSD candidate is built or trained.**  The file hash and
the commit are recorded in
`results/e2e_dictenv_common_subspace_dictionary_v1/preregistration_snapshot.json`
and `preflight.json`.  No threshold, definition, split or gate rule in this
document may be edited after the first Stage-A process starts.

* **CPU only** (`CUDA_VISIBLE_DEVICES=""`, `device = torch.device("cpu")`).
* **Never touches SSH / the remote runner / GPU.**
* **Never loads the official ZINC test split** (`official_test_loaded = false`
  in every artifact).
* **At most one new full training trajectory** (seed 0 of the selected CSSD
  candidate, 1 → 40 → 320 as one run with one optimizer state).
* Existing baselines are **not retrained**: FINAL-CLEAN sparse seed-0 soup
  `0.12849851670576026` and historical DenseTied seed-0 soup
  `0.1255625270641758` are reused as frozen references only; the DenseTied
  number is **not** a matched control for the new descriptor.

---

## 0. Frozen context (not re-opened)

The closed rounds `e2e_dictenv_clean_mechanism_v1` and
`e2e_dictenv_dictionary_coder_audit_v1` froze:

```text
FINAL-CLEAN = C6
              + paired node structure-semantic binding
              + paired edge structure-semantic binding
              + sparse tied-IHT coding (K=32, s=8, IHT-10)
              + full distance+overlap+boundary relation
```

Closed findings reused here without re-testing:

* C6 is safe across 3 CPU seeds; the graph chemistry histogram shortcut is
  structurally unused; paired node and edge correspondence are kept;
  distance-only relation is rejected; moments / anchor chemistry are retained.
* The reused three C6 dictionaries (seeds 0/1/2, soup states) each contain a
  near-universal DC/scale triplet: atoms **6 / 24 / 27**, activation rate
  ≈ 1.000, aligned with the train phi mean with `|cos| = 0.76–0.98`, atoms 6
  and 24 0.88–0.99 collinear with each other.  A fourth near-universal atom
  **23** is the PC1-like atom (`|cos(atom 23, PC1)| = 0.71–0.80`).
* IHT-10 support is the IHT fixed point; IHT-30/100 change amplitudes only.
  OMP is not an oracle on the tied-trained dictionary.
* Sparse → dense valid `R² ≈ 0.9988–0.9989`, dense → sparse `R² ≈ 0.97–0.98`.
* Sparse dictionary specificity vs DenseTied is **not** established.

This round does **not** reopen the coder, the fusion, the relation, the
features or the training protocol.  It changes exactly one thing: it removes
the dataset-common structural subspace from the sparse dictionary input and
models it explicitly.

### 0.1 Frozen reference values (used by the epoch-40 gate)

Extracted from the frozen audit artifacts
(`results/e2e_dictenv_dictionary_coder_audit_v1/coder_geometry/seed0/summary.json`,
`atom_specialization/seed0.json`, valid split, IHT-10, seed-0 C6 soup
dictionary):

| reference | value |
|---|---|
| `REF_NEFF_VALID` (effective atoms) | `14.520461423605052` |
| `REF_TOP5_VALID` (top-5 activation share) | `4.795910410258632` |
| `REF_WEIGHTED_SPEC` (train-usage-weighted train Spec) | `0.20673863977279258` |
| `REF_DC_COUNT_GT095` (atoms 6/24/27 with rate > 0.95) | `3` |
| `REF_MAX_ACTIVATION_RATE` | `1.0` |
| `REF_SPARSE_SEED0_SOUP_MAE` | `0.12849851670576026` |
| `REF_DENSE_SEED0_SOUP_MAE` (historical, not matched) | `0.1255625270641758` |

`preflight` re-reads the stored audit JSON and aborts if these numbers differ
by more than `1e-9` (provenance check).

---

## 1. Stage A — feature-space audit (zero training)

The dictionary input space is re-derived from the real pipeline, not from
memory:

* `results/e2e_dictenv_p1/cache/env_{train,valid}.pt` — the exact tensor
  `p1run.load_split` attaches as `data.dict_phi` (verified on a small subset
  by comparing the cache slice against the dataset field).
* Expected: train `231664 × 65` (10 000 molecules), valid `23083 × 65`
  (1 000 molecules), float32, **no normalization** between the cache and the
  dictionary input (`P2Model.code` consumes `data.dict_phi` directly; the
  anchor standardization does not touch `dict_phi`).
* Identically-zero descriptors are **recorded but not removed**: the round
  keeps 65-D and the frozen channel layout.  The named pair
  `root_neighbour_shell1` (column 5) and `root_walk1` (column 8) must be
  identically zero on both splits; any *additional* zero columns found are
  also recorded (audit note only, no descriptor revision this round).

## 2. Stage B — train-only common subspace

All common directions are fit on the **official-train cache rows only**;
valid is transform-only.  Let `x_v = phi_v ∈ R^65`, `N = 231664`.

**q1 (one-dimensional).**

```text
mu      = E_train[x]
u1      = mu / ||mu||_2
c_{v,1} = u1^T x_v
r^{(1)}_v = x_v - c_{v,1} u1
```

**q2 (two-dimensional; diagnostic only unless q1 fails, see §5).**

```text
r1_train     = x_train - (x_train u1) u1
v2           = PC1 (top right singular vector) of the mean-centred r1_train
u2           = v2 - (u1^T v2) u1 ;  u2 = u2 / ||u2||_2
U            = [u1, u2]        (65 x 2, orthonormal by construction)
c_v          = U^T x_v
r^{(2)}_v    = x_v - U c_v
```

No other q is considered.  `q ∈ {1, 2}` only; q = 3+ / sweeps are forbidden.

**Scaling of common coordinates (frozen).**  For each common coordinate,
train-only RMS:

```text
s_k = sqrt( E_train[c_k^2] )
c~_k = c_k / max(s_k, 1e-12)
```

The absolute/common component is **not** removed: no mean subtraction and no
per-coordinate centring is applied to `c`.  Valid uses the frozen train `s_k`.

**Energy accounting (report-only, train and valid separately).**

```text
E_common   = ||U U^T x||_F^2 / ||x||_F^2
E_residual = ||r||_F^2     / ||x||_F^2
E_centered_common   = ||U U^T (x - mu_x)||_F^2 / ||x - mu_x||_F^2
E_centered_residual = ||r - mean(r)||_F^2      / ||x - mu_x||_F^2
```

## 3. Stage C — zero-training same-`D` structural probe

No new dictionary is trained here.  For each of the three reused C6 soup
dictionaries (seeds 0/1/2) and **the same frozen `Dbar`**, four descriptor
variants are encoded with the frozen coder `iht10` (K=32, s=8, 10 steps):

| variant | target space `x_v` given to the dictionary | common dim |
|---|---|---|
| `RAW` | `x_v` | 0 |
| `MC` (diagnostic only) | `x_v - mu` | 0 |
| `Q1` | `r^{(1)}_v` | 1 |
| `Q2` | `r^{(2)}_v` | 2 |

`MC` is a **diagnostic control**, never a candidate architecture
(`mean-centering only`).  Per (seed × variant × split) the following are
recorded:

```text
recon_frobenius        = ||x - A Dbar^T||_F / ||x||_F                (target space)
recon_mean_row_squared = mean_v ||x_v - a_v Dbar^T||^2 / (||x_v||^2 + EPS)
common_energy_fraction / residual_energy_fraction                     (same space)
active_atoms, effective_atoms
top1_usage, top3_usage, top5_usage, top8_usage
max_activation_rate
n_atoms_gt_099, n_atoms_gt_095, n_atoms_gt_090
atom6_rate, atom24_rate, atom27_rate, atom23_rate
usage_entropy, gini
support_jaccard_vs_raw        (per-row Jaccard of supports, mean)
code_cosine_vs_raw            (per-row cosine of the 32-D codes, mean)
```

Support/cosine comparisons are always **RAW vs variant with the same seed and
split** (same 32-D coordinate, so comparable).

## 4. Condition definitions (numerical, frozen)

* **A (universal triplet broken).**  Of the RAW near-universal triplet
  `{6, 24, 27}`, at least **two** atoms have
  `activation_rate(variant) < 0.95`.  RAW sanity precondition: all three have
  RAW activation rate `≥ 0.99` for that seed (must hold; if not, the seed is
  reported as a failed precondition).
* **B (usage mass spread).**  `top5_usage(RAW) − top5_usage(variant) ≥ 0.4`
  (absolute, same seed/split).
* **C (residual keeps variation).**  For Q1 only:
  `E_centered_residual / ||x − mu_x||² ≥ 0.5`, i.e. the residual keeps at
  least half of the mean-removed structural variation energy.  Raw-energy
  fractions are reported but do not enter C (raw phi65 is DC dominated:
  `E_common(u1) ≈ 0.977` of `||x||²`).
* **D2 (PC1-like atom demoted), Q2 fallback only.**
  `atom23_rate(Q2) ≤ 0.6 × atom23_rate(RAW)` for that seed.

## 5. Candidate selection rule (frozen; representation metrics only)

Target y and valid MAE are **never** used for selection.  Conditions are
evaluated on the **official-valid** split (primary, consistent with the audit
and the epoch-40 gate); the train-split values are reported alongside and a
train/valid disagreement is recorded but does not change the verdict.

1. **q1 is selected** iff the Q1 variant satisfies **A ∧ B ∧ C** on at least
   **2 of the 3** reused dictionaries.
2. Only if q1 fails: **q2 is selected** iff the Q2 variant satisfies **A ∧ B**
   (same definitions) on at least 2 of 3 dictionaries **and** **D2** on at
   least 2 of 3 dictionaries.  Q2 diagnostics are still stored if q1 wins.
3. If neither fires:

```text
COMMON_SUBSPACE_SEPARATION_NOT_SUPPORTED
NO NEW TRAINING
```

No q3, no penalty, no K/s/λ change, no rescue.

## 6. Stage D — CSSD architecture (selected q)

Let `q ∈ {1, 2}` be the selected common dimension and `U` the frozen
train-only component matrix (65 × q) from §2.

**Residual dictionary (hard constraint).**  The raw parameter stays
`D_raw ∈ R^{65×32}` (same initialization as the frozen SDB artifact).  Every
forward:

```text
D_perp = (I - U U^T) D_raw
Dbar_perp = column_normalize(D_perp)        # v0.normalized_dictionary
```

so `U^T Dbar_perp = 0` up to numerical precision.  A hard test asserts
`max |U^T Dbar_perp| ≤ 1e-6` and that every projected column norm is `> 1e-8`.
**Dead-column fallback (deterministic, initialization-time only):** if
`||(I-UU^T) d_j||_2 ≤ 1e-8 · ||d_j||_2` for a column `j`, that column is
replaced by `(I-UU^T) d_{j*}` with
`j* = argmax_k ||(I-UU^T) d_k||_2`; the fallback count is recorded.  No
runtime NaN and no atom-selection sweep.

**Sparse input.**  IHT-10 always receives the residual `r_v`, never raw phi.

**Structural coordinate.**

```text
z_v = [ c~_{v,1}, ..., c~_{v,q} ; alpha^{res}_v ]  ∈ R^{32+q}
alpha^{res}_v = IHT10( Dbar_perp, r_v )
```

The common coordinates are explicit, dense and never dropped.

**Node binding (form unchanged).**

```text
u_v = ( z_v W_A^S ) ⊙ ( q_v W_A^C ) / sqrt(D_A)
```

with `W_A^S ∈ R^{(32+q)×48}` (`D_A = 48`).  No new fusion.

**Edge binding (form unchanged).**

```text
g_{uv} = [ z_u + z_v ; |z_u − z_v| ; z_u ⊙ z_v ]
```

with `W_E^S ∈ R^{3(32+q)×48}`, followed by the frozen bond semantic
multiplicative binding (`D_E = 48`).

**Loss (frozen form).**  The reconstruction term keeps the frozen denominator
on the full descriptor:

```text
r_v        = x_v - U U^T x_v
r_hat_v    = alpha^{res}_v Dbar_perp^T
L_rec      = mean_v ||r_v - r_hat_v||^2 / (||x_v||^2 + EPS)
L          = L1(y_hat, y) + H1_LAMBDA * L_rec
```

The common component is reconstructed exactly (`U c`) and contributes zero
reconstruction error; the numerator is the residual error.  `H1_LAMBDA`,
optimizer, lr, weight decay, batch, clip and horizon are the frozen H1 ones.

**Initialization (frozen).**  The model is first built with the **frozen
factory** `cm.build_clean_mech_model(dictionary, 0, C6 sparse spec)` under
`torch.manual_seed(0)`, so every non-CSSD tensor keeps the exact FINAL-CLEAN
seed-0 initialization stream.  The two binding matrices are then **widened
in place**:

* new coordinate rows are inserted at the common-coordinate positions
  (row 0 of `W_A^S`; rows `0`, `32+q`, `2(32+q)` of `W_E^S` when the three
  concatenated blocks are written in `[sum; |diff|; prod]` order);
* old rows keep their original values in the new layout;
* **new rows are initialized to exactly zero**, so at init the common
  coordinate has no direct binding effect and training must learn to use it;
* the dictionary parameter is unchanged (the projection is applied at every
  forward and is not stored).

Recorded provenance: which tensors cannot be bit-matched with the FINAL-CLEAN
seed-0 run (the widened matrices and the projected dictionary operator).

**Parameter accounting (frozen expectation).**  FINAL-CLEAN = 97 487 params
(verified from the frozen C6 seed-0 checkpoint).  `D_A = 96`, `d_e = 48`, so
CSSD adds `q × 96` (node) + `3q × 48` (edge) = `240q` params; q1 → +240
(97 727), q2 → +480 (97 967).  These counts are recorded; no matched-capacity
control run is purchased this round.

## 7. Stage E — focused tests (must pass before training)

The focused test file must cover at least:

```text
train-only U construction; valid cannot influence U
q1 identity: x == U c + r (numerically)
q2 identity: x == U c + r (numerically)
U^T U == I ;  U^T r == 0
RMS scaling train-only; c~ keeps non-zero mean
projected dictionary: U^T Dbar_perp == 0
projected dictionary columns finite/non-zero
IHT exact top-s unchanged (frozen coder reused)
CSSD construction preserves all non-CSSD parameters bit-identically
RAW/C6 path unmodified (C6 mask equality + existing C6 checkpoint keys)
CSSD q1/q2 node shapes [n, 3, 48]
CSSD q1/q2 edge shapes [n, 6, 48]
training-loop bit-equivalence: train_cssd(callback=None) == audit.train_cpu
CPU-only guard; official-test blocker
epoch-40 gate logic (synthetic)
```

Run:

```bash
uv run pytest -q tracks/ksvd/tests/test_e2e_dictenv_common_subspace_dictionary_v1.py
```

If a shared helper is touched (it is not planned), the clean-mechanism /
dictionary-coder-audit / P2 focused tests are re-run as well.

## 8. Stage F — the single training trajectory

Only if §5 selects q1 or q2.  Exactly one run:

```text
CSSD-selected, seed 0, from scratch (same data order/optimizer/horizon as
the FINAL-CLEAN seed-0 run), C6 mask, paired/paired, sparse tied-IHT,
K=32, s=8, IHT-10, H1_LAMBDA, batch 128, Adam 1e-3 / wd 1e-5, clip 5.0,
320 epoch maximum, Top-5 soup.
```

The trajectory runs **in one process**; at epoch 40 the frozen gate of §9 is
evaluated on the live model/optimizer and decides:

* gate passes → the **same process** continues epochs 41…320 with the same
  optimizer state and RNG state (no restart, no reload);
* gate fails → the run stops at epoch 40 and `training/` records the failure.

The training loop is a faithful copy of the frozen `audit.train_cpu` body
(proven by a bit-equivalence test on a small run with `callback=None`);
it adds only an epoch callback and does not alter data order, RNG use,
loss, optimizer or checkpoints.

## 9. Epoch-40 gate (frozen; representation-first)

**Basic health (all required, else `CSSD_TRAINING_BROKEN`):**

```text
finite task loss, finite rec loss, finite dictionary and projections
finite valid predictions
dictionary movement after 40 epochs > 0
dictionary receives a non-zero finite task gradient (condition C below)
```

**Catastrophic task stop (only task criterion):**

```text
train_mae(40) > 2.0  OR  valid_mae(40) > 2.0  OR any non-finite value
  -> CSSD_CATASTROPHIC_STOP
```

**Continuation gate — continue to 320 iff all of:**

* **Condition A — redundancy reduced.**  At least one of:
  `DC_count_gt095(CSSD, valid) ≤ 1` (triplet {6,24,27} counted after
  projection)  OR  `REF_TOP5_VALID − top5_share(CSSD, valid) ≥ 0.4`.
* **Condition B — vocabulary health.**  At least one of:
  `effective_atoms(CSSD, valid) ≥ REF_NEFF_VALID + 1.0`  OR
  `usage_weighted_spec(CSSD) ≥ 1.2 × REF_WEIGHTED_SPEC`.
* **Condition C — load-bearing.**  On one frozen train batch (batch 0 of a
  deterministic, unshuffled train loader), the model is evaluated in `eval()`
  mode (dropout off, so the callback cannot perturb the training RNG stream)
  and `||∂L/∂D||_2` must be finite and `> 1e-8`; every projected dictionary
  column norm must be `> 1e-8`.

The epoch-40 metrics use the same definitions as the audit round (activation
frequency, `effective_atoms`, top-5 share, SMD-based
`atom_specialization_rows` with train-only mean/std).  Valid-split metrics are
the gate's primary input; train values are reported.

If A ∧ B ∧ C fails: `CSSD_REPRESENTATION_GATE_FAIL`, stop at epoch 40, no
continuation, no rescue, no second run.

## 10. Stage G — reusable-structure audit (only if the full run completes)

On the trained CSSD seed-0 **soup** state (and the frozen RAW C6 seed-0 soup
dictionary for comparison):

* **Atom usage**: activation rate, effective atoms, top-k usage, entropy,
  Gini, per-atom rates; compare RAW seed 0 vs CSSD seed 0.
* **Coherence (diagnostic only, never a loss)**:
  off-diagonal `|Dbar^T Dbar|` mean / median / p90 / max of the projected
  normalized dictionary (and RAW for reference).  Coherence is **not**
  optimised.
* **Structural specialization**: exactly the frozen
  `dca.atom_specialization_rows` definition (SMD profiles with train-only
  mean/std, `Spec_j`, top-response Spec, train→valid profile cosine).  No
  metric change.
* **Graph-level reuse / coverage** (new): with `graph_id(v)` from the cache
  `node_sizes`:
  `Coverage_j = #{graphs with ≥ 1 activation of j} / N_graphs` (train and
  valid); `mean`/`median` activations per active graph;
  `top-10% share` = fraction of atom `j`'s activations that fall in the
  top 10 % of its active graphs by activation count (ceil at ≥1).
  Coverage is reported as a distribution; no coverage threshold is a gate.
* **Reusable-specialization table**: per atom — node activation rate, train /
  valid coverage, `Spec_train`, `Spec_valid`, profile cosine, mean
  activations per active graph, top structural features; sorted once by
  activation rate and once by specialization.
* **Common coordinates**: Spearman correlation of `c1` (and `c2` if present)
  with the frozen named topology descriptors
  (`dca.structural_descriptors`), plus train/valid distribution summaries.
* **Sparse ↔ dense residual recoverability**: the frozen train-only OLS
  `fit_linear_recoverability`, in the residual space:
  `dense (= r @ Dbar_perp) → IHT10` and `IHT10 → dense`; valid `R²`,
  normalised error, mean cosine, CKA.  A remaining ≈ 0.999 is **allowed**
  and must not be engineered downward.

## 11. Task interpretation bands (frozen)

With `M_CSSD` = CSSD seed-0 Top-5 soup valid MAE, references
`REF_SPARSE_SEED0_SOUP_MAE` and historical `REF_DENSE_SEED0_SOUP_MAE`:

| band | condition | label |
|---|---|---|
| strong exploratory | `M_CSSD ≤ 0.1255` or `REF − M_CSSD ≥ 0.003` | `CSSD_SINGLE_SEED_PROMISING` |
| neutral | `|M_CSSD − REF| ≤ 0.003` | `CSSD_REPRESENTATION_SUPPORTED / TASK_NEUTRAL_SINGLE_SEED` |
| cost | `M_CSSD > REF + 0.010` | `COMMON_COMPONENT_SEPARATION_COSTS_MATERIAL_TASK_SIGNAL` (not adopted) |
| no representation gain | final dictionary recreates ≥ 2 universal residual atoms / top-5 ≥ RAW − 0.2 / no specialization gain | `CSSD_HYPOTHESIS_NOT_SUPPORTED` regardless of MAE |

Only single-seed exploratory evidence may ever be claimed from this round.

## 12. Forbidden this round

```text
IHT30/IHT100/OMP training; K/s/lambda sweeps
coherence / usage-balance / entropy / orthogonality penalties
new fusion, new relation, new graph or chemistry features
shell experiments, A1 attributed dictionary
width/depth/LR/dropout searches
CSSD DenseTied full run (explicitly deferred to a possible next round)
seed 1 / seed 2 / third architecture full run
official ZINC test access
modification of docs/luyin/luyin19.txt
```

## 13. Stop conditions

1. q1/q2 zero-training gates both fail → no training
   (`COMMON_SUBSPACE_SEPARATION_NOT_SUPPORTED`).
2. Epoch-40 gate fails → stop the trajectory at 40.
3. Epoch-40 catastrophic task failure → stop.
4. Otherwise: one CSSD seed-0 run to epoch 320 + Stage G audit → stop.

## 14. Result layout

```text
tracks/ksvd/results/e2e_dictenv_common_subspace_dictionary_v1/
  preflight.json
  preregistration_snapshot.json
  common_subspace.json
  zero_training/{raw,mean_center,q1,q2}.csv
  zero_training/selection.json
  zero_training/comparison.csv
  training/{curve.csv,epoch40_gate.json,final.json,checkpoints/}
  training/NOT_RUN.json          (if the selection gate fails)
  reusable_structure/{atom_profiles.csv,graph_coverage.csv,
                      common_coordinates.csv,sparse_dense_recoverability.csv}
  analysis_tables.md
  summary.json
  REPORT.md
  DECISION.md
```

## 15. Claim framework (must be stated in the report)

The report must decide between:

* **Claim A** — the sparse dictionary learns reusable, cross-graph
  structural factors beyond the common structural background;
* **Claim B** — the dictionary provides a shared structural coordinate, but
  sparsity does not yield a distinct reusable vocabulary;
* **Claim C** — common-subspace separation does not resolve the dictionary
  concentration problem.

No chemistry motif names are assigned to atoms; phi65 is topology-only.
