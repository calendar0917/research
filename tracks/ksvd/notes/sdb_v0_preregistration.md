# SDB-v0 pre-registration — Sparse Structural Dictionary Binding

Round: **sdb-v0** (`Sparse Structural Dictionary Binding`).
Dataset: ZINC strict-static, PyG `ZINC(subset=True)` canonical PyG split —
**official train 10000 / official valid 1000** (reused verbatim from the
FSAR-R2-AR0 cache; see §13).
Official ZINC **test is never loaded** at any stage (`official_test_loaded =
false`, `test_policy = no_test`).

This file is frozen **before** any formal run. After the formal run starts, no
definition, width, threshold, protocol or control below may be changed. Any
deviation must be written as an explicit Amendment with its own timestamp.

---

## 0. The one question

> The `pure structural role ↔ atom chemistry assignment` that FSAR-R2-AR0
> proved has real, multi-seed, control-beating task value on ZINC —
> can its **structural-role axis** be replaced by a **reusable, sparse,
> task-coupled dictionary coordinate** without losing its main task
> information?

Concretely: fit a sparse dictionary `D` on the frozen pure-topology coordinate
`φ_v ∈ R^65`, code each atom as `α_v` (`K=32`, `‖α_v‖₀ ≤ 8`), and ask whether
the dictionary-space assignment tensor `C_D = Σ_v (α_v − ᾱ)(q_v − q̄)ᵀ`
preserves (a) the label-free identity `C_φ ≈ D C_D` and (b) the *task* signal of
the historical `C_φ` assignment residual, under a matched dense-32 control.

### This round is NOT

* dictionary predicting all of ZINC standalone;
* a dictionary **pair** kernel;
* the `Cᵀ R C` relation algebra;
* graph message passing / centre update / recurrence;
* a ring/cycle feature rescue;
* a new reader / Transformer / attention;
* a new radius;
* a `K` / sparsity / width sweep.

---

## 1. Historical evidence this round is built on (and must not repeat)

### 1.1 FSAR-R2-AR0 — the positive evidence (the object we are compressing)

* `notes/fsar_r2_ar0.md`, `notes/fsar_r2_ar0_node_binding_mechanism.md`,
  `claim-fsar-r2-ar0-node-assignment-residual-20260917`,
  `decision-fsar-r2-ar0-assignment-supported-20260917`.
* Frozen pure-topology coordinate
  `φ_v = [root_basis(11), mean_node(11), std_node(11), mean_edge(15),
  std_edge(15), log1p|V_{B2}|, log1p|E_{B2}|] ∈ R^65`, chemistry-free, fixed
  radius 2.
* `q_v = onehot(x_v) ∈ R^28`; sum-centred
  `C_φ = Σ_v (φ_v − φ̄)(q_v − q̄)ᵀ`.
* `MB = F0([z_S, a]) + ⟨W_B, C̃_φ⟩` beat the strong marginal baseline `M0` and
  the matched marginal-capacity control `MM` on **all three seeds**:
  `M0` soup `0.553790 / 0.535570 / 0.546893`;
  `MB` soup `0.469285 / 0.458746 / 0.462365`;
  `M0 − MB = +0.084505 / +0.076824 / +0.084528`.
* Mechanism audit corrected the old rank-1 claim: raw `C` effective rank
  `≈ 56.6`; RMS-scaled model input `C̃` effective rank `≈ 116.3`; `PC1`
  explains `< 2 %` of the learned `B` branch; branch outputs agree across seeds
  at Pearson `≥ 0.988`; contribution-profile cosine `≥ 0.96`.
  → **the binding tensor is high-dimensional, not one direction.**
* Frozen-`M0` + a plain linear `C̃_φ` readout (`h5_frozen_m0.json`) already
  recovers `≈ 2/3` of the full `MB` gain, soup `0.494538 / 0.481283 /
  0.490486`.
* FSAR-C1 (incidence processor) and AR0-EDGE (`C_E` edge assignment) are
  **out of scope here**; SDB-v0 keeps only the first-order node `C_φ` object.

### 1.2 TCCD-v0…v7 — CLOSED (do not rebuild as TCCD-v8)

`notes/tccd_*.md`. A learned prototype/dictionary vocabulary can be healthy and
assignment-sensitive, but prototype/dictionary **as a standalone
representation** has a large absolute-performance gap; `Cᵀ R C` standalone is
CLOSED (`decision-tccd-v7-stop-standalone-normalized-moment-20260922`).

### 1.3 DTX-v0 — CLOSED

`decision-zinc-dtx-v0-stop-no-aligned-cross-signal-20260923`. DTX crossed a
ring-blind dictionary (over the **mixed** `patch_cont` descriptor) with a
generic topology role, then compressed the `64×8` joint to 32-D. It found the
aligned cross load-bearing but **worse** than a matched marginal control.

### 1.4 compact-v6 — CLOSED

`decision-compact-v6-topology-attribute-factorization-nogo-20260910`. Appended
zero-init attribute branch collapsed (shuffle-invariant, `e_attribute` norm std
`2.3e-10`); extra encoder capacity was not a reliable mechanism.

### 1.5 SDPK / SDPL / SRDA / GSCN / PSD / PSCD — CLOSED

Dictionary-as-pair-algebra (`SDPK`, `SDPL`, `SRDA`), task-driven dictionary as
the raw graph hidden transition (`GSCN`), raw pair-state encoder (`PSD`),
merge-only prototype compression (`PSCD`) are all closed; dictionary can be
load-bearing yet not improve absolute MAE.

---

## 2. Formal differences vs the closed routes (required explicitly)

| route | what it was | why SDB-v0 is different |
|---|---|---|
| **TCCD** | dictionary = whole local environment; relations = `Cᵀ R C`; standalone prediction | SDB-v0's dictionary input is **only** the 65-D pure-topology coordinate; chemistry is a separate one-hot; it uses the already-validated structure↔chemistry assignment; no `Cᵀ R C`. |
| **DTX** | dictionary over **mixed** `patch_cont`; cross = dictionary × generic topology role; joint compressed to low-D | SDB-v0's dictionary input is **chemistry-free pure topology**; cross = dictionary × **atom chemistry**; the `32×28` joint is **never** compressed. |
| **compact-v6** | appended zero-init learned attribute branch; collapsed | SDB-v0's chemistry is an explicit categorical assignment; the dictionary is fit/subsequently trained **directly** by reconstruction/task loss, not a zero-init side branch. |
| **SDPK/SDPL/SRDA/GSCN** | dictionary drives pair/state computation; dictionary pair algebra | SDB-v0 does **not** let the dictionary take over pair/state computation and builds no dictionary pair algebra; its entire output is a single linear residual on `C_D`. |

---

## 3. The frozen scientific object (chemistry purity)

`φ_v ∈ R^65` is exactly `fsar_r2_ar0.build_phi` — frozen and audited. The
dictionary path is allowed to read **only** `φ_v`.

**Forbidden dictionary input:** atom type; bond type; aromatic/ring label;
engineered chemistry descriptor; typed token; parent token; historical mixed
`patch_cont`; B-Full hidden state; any target-derived feature.

Topology naturally expressing degree / rooted walks / adjacency-derived
structure is allowed (it is not an explicit ring label). This is enforced by an
AST/behavioural purity test: `α_v` must be **bit-identical** under any
reassignment of `q_v` (atom chemistry), because `α` is a function of `φ` only.

---

## 4. Frozen dictionary configuration

```
K = 32           # dictionary atoms
s = 8            # exact sparsity, ‖α_v‖₀ ≤ 8
D ∈ R^{65×32}    # φ_v ≈ D α_v
```

Rationale for `K = 32` (frozen, one-shot, **not validation-selected**):

* the source structural coordinate is only `65`-D;
* the FSAR feature audit reports a low intrinsic/effective rank for the
  pure-topology `φ` (effective rank `9.11` across atoms with `11` exactly
  constant coordinates), so `32` atoms are already an over-complete basis for
  this object;
* `K = 64` is almost equal-width to `65` and is not a meaningful structural
  compression;
* `32` is a single, pre-declared capacity choice.

**No `K` sweep. No `s` sweep. No radius sweep. No width sweep.**

First dictionary-domain gate uses **detached K-SVD + OMP** (reusing the
correctness-tested `tracks/ksvd/code/tccd_v0.ksvd_fit` / `omp_codes`). If the
round later reaches end-to-end coupling it reuses TCCD's correctness-tested
**tied unrolled IHT** (`tccd_v0.iht_codes`), never a free LISTA encoder.

---

## 5. Core representation: dictionary structural binding

```
q_v ∈ {0,1}^28   original atom category one-hot
α_v ∈ R^32       sparse dictionary code of φ_v (‖α_v‖₀ ≤ 8)

C_D(G) = Σ_v (α_v − ᾱ)(q_v − q̄)ᵀ ∈ R^{32×28},  flattened 896-D
```

The `32×28` joint is **never** compressed to 16/32-D before the readout.
Train-only per-coordinate RMS scaling follows the AR0 logic (`C̃_D = C_D /
D_D`, `D_D = sqrt(E_train[C_D²] + ε)`, no dataset-mean subtraction,
zero-RMS coordinates masked).

---

## 6. The label-free sanity identity (most important cheap gate)

Because `φ_v ≈ D α_v`, the sum-centred statistics satisfy, exactly in the limit
of exact reconstruction,

```
C_φ = Σ_v (φ_v − φ̄)(q_v − q̄)ᵀ ≈ D · C_D .
```

Define the binding residual on a split

```
E_bind = ‖C_φ − D C_D‖_F² / (‖C_φ‖_F² + ε)
```

aggregated over the split's molecules (sum of squared Frobenius norms in
numerator and denominator); the per-molecule ratio median is reported as a
secondary statistic. **This gate uses no `y`.** It tests that the dictionary
preserves a *previously task-validated* assignment object, not merely that
reconstruction is good.

---

## 7. Stage 0 — dedup + implementation sanity (before any training)

Dedup (done, recorded in the analysis note): no existing repo implementation of
`FSAR pure-topology coordinate → sparse dictionary`, `dictionary-code × atom
category centered assignment`, or an `SDB`-style binding residual. The
`K-SVD`/`OMP`/`tied-IHT` machinery is reused, not reinvented.

Targeted tests (all required to pass):

1. node relabel invariance of `α` and `C_D`;
2. dictionary input chemistry purity (`α` bit-identical under `q` reassignment);
3. exact sparsity `‖α_v‖₀ ≤ 8`;
4. batching invariance of the batched `C_D`;
5. `C_D` assignment sensitivity (endpoint↔endpoint vs endpoint↔internal);
6. chemistry permutation: `α` unchanged; `q` marginal unchanged; structural
   marginal unchanged; `C_D` changes;
7. joint node permutation (topology + `q` together) leaves `C_D` invariant;
8. no official test access (AST/path scan);
9. no raw/mixed bypass (`q = 0 ⇒ C_D = 0`; `C_D` reachable only through `α`, `q`).

Synthetic controls (before any ZINC conclusion):

* **positive** — fixed topology/marginals, chemistry placement changed only;
  a linear readout on `C_D` must separate the cases (`M0` cannot);
* **negative** — label depends only on marginals, placement randomised; the
  `C_D` branch must show no stable advantage.

Any correctness failure is fixed as an implementation bug; it is never
explained by a formal ZINC number.

---

## 8. Stage 1 — label-free dictionary-domain gate

No existing baseline is re-run. Only a new `D_KSVD(K=32, s=8)` is fit on the
FSAR cache. Cheap references:

* `D_rand` — fixed-seed random normalised dictionary (`OMP` `s=8`);
* `D_pca` — rank-32 affine PCA projection of `φ` (dense reference, best linear
  rank-32 reconstruction).

Reported:

* **A. structural reconstruction** `E_φ = ‖φ − Dα‖² / ‖φ‖²` (train / dev);
* **B. dictionary health** used atoms, dead atoms, molecule coverage,
  top-1/top-8 coefficient mass, support entropy, atom support frequency;
* **C. binding reconstruction** `E_bind` primary for `D_KSVD`, plus `D_rand`
  and `D_pca` references (train / dev);
* **D. dictionary semantic audit** (report only, never input): top-activating
  real structural neighbourhoods, degree/shell/walk statistics; optional
  post-hoc ring/cycle enrichment is interpretation only.

### Stage 1 gate (frozen)

Let `*_dev` denote official valid (internal-dev).

* **PASS** iff
  * `E_bind_ksvd_dev ≤ 0.10`, **and**
  * `E_bind_ksvd_dev < E_bind_rand_dev`, **and**
  * `E_bind_ksvd_dev ≤ 3.0 × E_bind_pca_dev`, **and**
  * `used_atoms ≥ 24 / 32`.
* **STOP** iff any of
  * `E_φ_ksvd_dev ≥ E_φ_rand_dev` (no reconstruction advantage over random), or
  * `used_atoms < 16 / 32` (massive atom collapse), or
  * `E_bind_ksvd_dev > 0.50` (`C_D` cannot approximate `C_φ`).
* otherwise **INSPECT**.

On STOP: the round ends with a durable negative; **no** `K=64` rescue, `s`
sweep, radius sweep, or chemistry injection.

---

## 9. Stage 2 — FSAR mechanism-preservation gate (only if Stage 1 PASS)

Question: does the sparse structural dictionary preserve the **task** signal of
the historical `C_φ` node-binding residual? Not a SOTA attempt.

**Reused durable artifacts (not re-trained):** FSAR `M0` soup states
`r2ar0_m0_seed{0,1,2}_top5_soup.pt` (official valid MAE
`0.553789896648319 / 0.535570007022412 / 0.5468928938917234`), the FSAR cache
(`φ`, `q`, `C̃` scalers), and the durable frozen-`M0`+linear-`C̃_φ` oracle
`h5_frozen_m0.json` (`0.4945375594366235 / 0.48128274935426624 /
0.4904858892400294`). Historical FSAR seeds are **not** re-trained.

New models (frozen `M0` base + **one** linear assignment residual; identical
protocol; `M0` predictions precomputed because the frozen base has no
dropout/BN):

| arm | statistic | dim | trainable |
|---|---|---|---|
| `phi65` (oracle) | `C̃_φ` (full coordinate) | `65×28` | 1820 |
| `dict32` (**SDB**) | `C̃_D` from frozen `D_KSVD` | `32×28` | 896 |
| `dense32` (**matched dense control**) | `C̃_z`, `z_v = W φ_v`, `W ∈ R^{65×32}` **trained** | `32×28` | 2976 |
| `rand32` (extra control) | `C̃_D` from frozen random `D` | `32×28` | 896 |

`dict32` and `dense32` are **total-parameter-matched** (`65×32 + 32×28 = 2976`
each); `rand32` is trainable-parameter-matched to `dict32` and isolates sparse
dictionary *inductive bias* from a generic random projection.

Primary quantity:

```
Recovery_oracle = (MAE(M0) − MAE(M_dict32)) / (MAE(M0) − MAE(M_phi65))
```

with `M_phi65` the in-session re-run of the full 65-D linear branch (the durable
`h5` numbers are cross-checked for protocol drift).

### Stage 2 gate (frozen), evaluated per seed (0/1/2) and as a mean

* **PASS** (mechanism preserved → Stage 3 authorised) iff
  * `Recovery_oracle ≥ 0.75` (mean; and not contradicted on every seed), **and**
  * within-molecule atom-category assignment shuffle degrades MAE by `≥ 0.02`
    (dictionary branch uses the real `α_v ↔ q_v` alignment), **and**
  * `dict32` branch output std `> 1e-3` (alive, not collapsed).
* **STOP** iff `Recovery_oracle < 0.50`.
* otherwise **INSPECT** (default: do **not** buy the expensive Stage 4 run
  unless `dict32` is also clearly better than `dense32`).
* **Dictionary-advantage flag:** `M_dict32 ≤ M_dense32 + 0.005`. If this fails,
  no "dictionary inductive advantage" claim is permitted and Stage 4 is not
  bought even on PASS.

No marginal/capacity-only explanation is accepted: `rand32` and `dense32` are
the guards.

---

## 10. Stage 3 — task-coupled end-to-end dictionary (only if Stage 2 PASS)

Only then is `D` made task-coupled via TCCD's correctness-tested tied unrolled
IHT:

```
α^{t+1} = H_s[ α^t + η Dᵀ(φ − D α^t) ]
L = L_MAE + λ L_rec ,  L_rec = (1/N) Σ_v ‖φ_v − D α_v‖² / (‖φ_v‖² + ε)
```

`λ` by TCCD-style one-shot detached initial-magnitude calibration. Forbidden:
free LISTA / independent encoder, entropy/balance/orthogonality/temperature/
gamma terms, dictionary bypass, validation-tuned `λ`. Must report: task
gradient reaches `D`; reconstruction gradient reaches `D`; atom movement;
usage; exact sparsity; train→valid dictionary health; frozen-`D` vs task-`D`.
If task-coupling gives no material improvement, no full-backbone run is bought.

---

## 11. Stage 4 — strict-static strong-backbone validation (only if all gates pass)

Reuse the existing strict-static `S0` (`zinc_static_dictionary_pair`), **without
re-running the historical baseline**. Historical seed-0 soup
`0.14079417109390488`, seed-1 soup `0.13642283965486587`; durable states exist
(`s0_seed0_selection_state.pt`, `s0_seed1_selection_state.pt`).

Only new arms:

* `S0 + DictBinding`: `ŷ = ŷ_S0 + ⟨W_D, C̃_D⟩`;
* `S0 + DenseBinding`: a completely parameter-matched dense structural
  coordinate binding control.

Forbidden: `Cᵀ R C`, dictionary pair coordinates, topology cross, pair kernel,
centre update, recurrence, message passing, new global features, ring features,
MLP cross compressor. The SDB branch is exactly:

```
pure topology φ_v → sparse K=32 dictionary → α_v
[α_v , atom chemistry q_v] → centred 32×28 binding tensor
→ train-only RMS scaling → linear residual
```

### Stage 4 seed-0 gate (frozen)

* `M_Dict ≤ 0.1378` (material gain `≥ 0.003` over historical `S0` seed-0 soup
  `0.140794`), **and**
* `M_Dict ≤ M_Dense + 0.002`, **and**
* within-molecule assignment shuffle (preserving the dictionary-code multiset
  and the chemistry multiset, breaking only `α_v ↔ q_v`) causes clear MAE
  degradation.

Only if all three pass on seed 0 is **Ticket 2 (seed 1)** bought. No seed 2/3.
No HPO. No rescue run.

---

## 12. Reuse / no-rerun inventory

Reused, **not** re-run: historical `S0` seed 0/1; FSAR `M0`/`MB`/`MM`; TCCD
baselines; B-Full; B-Bag; DTX; SDPK; SRDA; all other historical controls. The
only new training in Stage 2 is four ~1–3k-parameter linear residual branches
(few seconds on CPU) on frozen bases. Which baselines were **not** re-run is
recorded in every results JSON.

## 13. Time budget / tickets

Formal GPU tickets (at most):

1. **Ticket 1** — Stage 4 `seed0`: `DictBinding` + `DenseBinding` (may use both
   idle GPUs, respecting remote skill occupancy rules).
2. **Ticket 2** — Stage 4 `seed1`: `DictBinding` + `DenseBinding`, only if
   Ticket 1 passes all seed-0 gates.

No seed 2/3, no HPO, no rescue run. Stages 1–3 are CPU-cheap and are not formal
GPU tickets.

## 14. Forbidden rescues (after any formal FAIL)

No `K` 32→64; no sparsity 8→4/16; no deeper IHT; no free LISTA; no attention;
no nonlinear `896→MLP`; no edge dictionary; no pair-level `B²`; no dictionary
atom relation; no ring feature; no radius-3; no extra topology primitive; no
deeper reader; no optimizer/regularisation sweep; no seed hunting. A durable
failure diagnosis is written.

## 15. Provenance recorded in every result

local commit; remote deployed commit; dirty flag; GPU; seed; split fingerprint;
wall time; reused artifact identity; baselines NOT re-run; `official_test_loaded
= false`; exact stop reason; `test_access` never opened.

## 16. Expected outcome

* Stage 1 is expected to PASS (the 65-D object is low-rank and `K=32` is
  over-complete) — its value is a positive, reproducible `C_φ ≈ D C_D`
  identity + dictionary health, not a surprise.
* Stage 2 is the decisive mechanism gate: does the compression retain the
  task-validated assignment signal under a matched dense-32 control?
* If Stage 2/4 fails or lands INSPECT, the round records a durable NO/INSPECT
  and the "dictionary as a ZINC core contribution" space is narrowed
  accordingly. A clean STOP is preferred to an unattributable large run.

---

# Amendment A1 (made before the formal Stage-1 fit)

Timestamp: 2026-09-24. Status: **pre-formal-run**; no formal Stage-1 result
had been produced when this amendment was written (only a reduced smoke with
`max-fit-atoms = 20000`, `epochs = 3`, used to validate the code path).

**Observed in the reduced smoke** (`φ` from the FSAR cache, `K=32`, `s=8`):
`E_φ_ksvd_dev = 0.0040`, `E_bind_ksvd_dev = 0.0589`,
`E_φ_pca_dev = 2.1e-8`, `E_bind_pca_dev = 6.9e-7`, `used_atoms = 32/32`,
exact sparsity `8`.

**Finding.** The rank-32 *affine* PCA reference is essentially lossless for
`φ` (`E_φ_pca ≈ 1e-8`), because the FSAR pure-topology coordinate cloud is
globally low-rank (audited effective rank `≈ 9.1`). Consequently the §8 clause
`E_bind_ksvd_dev ≤ 3.0 × E_bind_pca_dev` is not a discriminating criterion —
it compares a sparse `s=8` code to a near-zero dense-limit reference and would
force `INSPECT` by construction, which is not the pre-registered intent (the
brief asks only to *report* the PCA reference).

**Amendment.** The PCA reference is downgraded from a hard gate to a
**reported reference**. The hard Stage-1 gate becomes exactly:

* **PASS** iff `E_bind_ksvd_dev ≤ 0.10` **and**
  `E_bind_ksvd_dev < E_bind_rand_dev` **and** `used_atoms ≥ 24/32`;
* **STOP** iff `E_φ_ksvd_dev ≥ E_φ_rand_dev`, or `used_atoms < 16/32`, or
  `E_bind_ksvd_dev > 0.50`;
* otherwise **INSPECT**.

One additional **reported** (not gated) dictionary reference is added:
`E_bind^{s=K}` — the same learned `D_KSVD` coded **densely** (`s = K = 32`),
which isolates the *sparsity* cost from the *subspace* cost. All other
definitions, thresholds, arms and gates are unchanged.

---

# Amendment A2 (made before the formal Stage-4 run)

Timestamp: 2026-09-24. Status: **pre-formal-run**.

**Finding.** The historical strict-static S0 runs persisted only the single
best `s0_seed{0,1}_selection_state.pt` plus the fixed Top-5 **soup
predictions on the valid split** (`runs/s0_seed*.json`). They did **not**
persist the five per-member soup states, so the S0 soup cannot be reproduced
bit-exactly on the **train** split, which a frozen-base residual branch needs
for training.

**Amendment.** Stage 4 uses the durable S0 **selection state** (best
checkpoint) as the frozen base for *both* train and valid, self-consistently.
The historical S0 Top-5 soup (`0.140794` seed 0) is still reported as context
but the material-gain gate is measured against the *same frozen base* used to
train the branch, because a residual branch must be trained on the base it is
compared against. Concretely the Stage-4 seed-0 gate becomes:

* `DictBinding_soup ≤ base_valid_mae − 0.003` (material gain on the frozen
  selection-state base), and
* `DictBinding_soup ≤ DenseBinding_soup + 0.002`, and
* within-molecule assignment shuffle degrades MAE by `≥ 0.02`.

The SDB branch and all forbidden-rescue rules are unchanged. `D` is the frozen
Stage-1 `D_KSVD(K=32, s=8)`; the readout is the only trained statistical part
(the matched dense control additionally trains a `65×32` projection).
