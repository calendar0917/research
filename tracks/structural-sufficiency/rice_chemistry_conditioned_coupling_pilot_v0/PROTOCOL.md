# RICE-CC-v0 — atom-conditioned coupling on the rooted interaction cumulant encoder

Round: `rice_chemistry_conditioned_coupling_pilot_v0`
Version: 2026-11-25/v1 — **frozen before any formal training run**
Task: `ogbg-mollipo` whole-molecule lipophilicity regression
Primary metric: whole-molecule Dev RMSE in original label units (D0, seed 0)
Runs: exactly two (`CC_NULL_S0`, `CC_REAL_S0`). No third run, no retry, no
search, no second seed, no D1, no official Valid/Test.

This is a **mechanism contrast**, not an architecture search. It keeps the
RICE-v0 encoder unchanged and changes only the pair potential passed to
`rice_energy.compute_phi`. The round asks whether direct atom conditioning of a
physical bond's two-state interaction, with **no new learnable parameter**, makes
the same rooted Möbius sentence a better molecular regressor.

## 1. Question and two frozen arms

Both arms are the RICE-v0 `RICE_CUMULANT` encoder:

* 16 independent two-state channels, `logZ` by exact enumeration (≤16 states);
* the same real radius-2 induced subgraphs, connected supports of size 1–4,
  every internal induced bond, triangle / 4-clique closure;
* root-conditioned Möbius inversion `κ(S,v)` of size 1→4;
* per-order `[sum, mean, std]` pooling, independent per-order LayerNorm,
  `log1p(count)`, shared sentence MLP and whole-molecule nonlinear readout;
* the same 8 Fit-standardised graph scalars, the same atom/bond categorical
  fields, the same unary MLP and bond MLP;
* no GNN, no message passing, no teacher checkpoint, no distillation, no
  auxiliary loss, no pretraining.

The arms differ in exactly one tensor:

| arm | pair potential entering `compute_phi` |
|---|---|
| `CC_NULL` | `J_base` (original bond-only potential) |
| `CC_REAL` | `J_real` (atom-conditioned coupling, formula below) |

Both arms have the same module set, the same parameter count, the same random
initial state hash, the same Fit/Dev rows, sample order, optimizer, loss,
schedule, epoch budget, clipping, checkpoint rule and readout. No channel count,
latent state count, word size, sentence width or training recipe changes.

## 2. Chemistry-conditioned pair potential

The model produces `u = [n_atoms, 16, 2]` and
`J_base = [n_bonds, 16, 3]` with `J_base[...,0] = J00`,
`J_base[...,1] = J01 = J10`, `J_base[...,2] = J11`.

With the physical bond endpoints `i = batch["bond_index"][:,0]`,
`j = batch["bond_index"][:,1]`:

```python
h = torch.tanh((u[..., 1] - u[..., 0]) / 2.0)          # [n_atoms, 16]
k = (
    J_base[..., 0]
    + J_base[..., 2]
    - 2.0 * J_base[..., 1]
) / 4.0                                                 # [n_bonds, 16]

delta = k * h[i] * h[j]                                 # [n_bonds, 16]

J_real = torch.stack(
    [
        J_base[..., 0] + delta,
        J_base[..., 1] - delta,
        J_base[..., 2] + delta,
    ],
    dim=-1,
)
```

`CC_NULL` uses `J_base`; `CC_REAL` uses `J_real`. Nothing is detached
(`u`, `h`, `k`, `delta` all carry gradients). No parameter is added: `h` and
`k` are deterministic functions of the existing unary and bond potentials.

Because both base potentials are `2·tanh(·)`, the conservative bounds are
`|J_base| ≤ 2`, `|k| ≤ 2`, `|delta| = |k| |h_i| |h_j| ≤ 2`, and therefore
`|J_real| ≤ 4`. The relevant numerical checks use the new **4** bound; the
original ±2 check is not reused for `J_real`. The exact enumeration, `Phi`,
Möbius inversion and pooled sentence are mathematically unchanged.

## 3. Data

D0 exactly as frozen by `acra_relation_generalization_pilot_v1`:

| item | value |
|---|---|
| raw structure cache | `tracks/structural-query-learning/data/ogbg_mollipo_sq-pilot-cache-v1.npz`, sha256 `fe002287…23c7c` |
| reused label-safe loader | `graph_word_incidence_composition_pilot_v0/code/gwic_data.py`, sha256 `f02863be…0cef` |
| Train labels | `structural_path_composition_route_v1/data/labels_official_train.npz` (3360 rows) |
| D0 split | `acra_relation_generalization_pilot_v1/data/split_train_sub_dev.json`, split sha256 `60ac13c0…559e` |
| Fit / Dev | **2856 / 504**, scaffold-disjoint, exactly covering official Train |
| target scaler | Fit rows only, z-score (`mean 2.185815826311436`, `std 1.2139664740467941`) |
| graph scalars | 8 deterministic whole-graph scalars, standardised on Fit rows only |
| official Valid | structure only; labels are NaN in the training object |
| official Test | never touched: no structure, no label, no metric |

No official Valid or Test label is read. The Dev rows are used only for the
primary metric and checkpoint selection, exactly as in RICE-v0.

## 4. Training protocol

The recipe is copied unchanged from `configs/rice.json` of RICE-v0:

* AdamW, peak LR 3e-4, min LR 1e-5, weight decay 1e-4;
* batch 64 molecules, 5 warmup epochs, cosine schedule, max 240 epochs;
* gradient clip 5.0, minimum 80 epochs, patience 45;
* whole-molecule standardised MSE only;
* best Dev RMSE selects the checkpoint;
* Fit and Dev RMSE/MAE are evaluated every epoch in original label units;
* seed 0, D0 only, one process per arm.

`CC_NULL` must be mathematically and computationally equivalent to the historical
`RICE_CUMULANT`. Two read-only verifications are mandatory before training:

1. the first forward of the historical RICE-v0 source and the new `CC_NULL` path
   agree in init hash, `u`, `J_base`, `Phi`, `κ` and standardized prediction;
2. the historical `RICE_CUMULANT` best checkpoint replayed through the new
   `CC_NULL` path reproduces its recorded Dev predictions and Dev RMSE.

Historical checkpoints are read-only validation objects and are never used to
initialise a new run. The historical `0.5994` Dev RMSE is the reproducibility
coordinate, not a training target.

## 5. Mandatory pre-training checks

`code/rice_checks.py` writes `analysis/checks.json`. Any FAIL stops the formal
training. The checks are:

**A — exact algebra.** For every graph with ≤4 atoms, every root and every
connected root-conditioned support, an independent float64 brute-force reference
confirms `Phi(S,v) = Σ_{T⊆S, v∈T} κ(T,v)` and the vectorised production `Phi`/`κ`
for **both** `J_base` and `J_real`. Float32 deviation is reported.

**B — connectivity / zero coupling.** Disconnected root-anchored `S` with
`|S| > 1` has `κ ≈ 0`; with all `J_base = 0`, `delta = 0` and every `|S| ≥ 2`
`κ` is zero. Triangle closure reads all 3 bonds and a 4-clique all 6 bond slots.

**D — invariance.** For both arms: atom renumbering, physical bond direction
swap, bond storage order, batch composition and molecule order leave the
prediction (and, where applicable, per-certificate `κ`) unchanged.

**E — gradients / activity.** On the first real batch, unary, pairwise,
embedding, sentence and readout gradients are finite and non-zero for both arms;
the `CC_REAL` `delta` term is not detached and its gradient reaches both `u` and
`J_base`; order-3 and order-4 `κ` are non-zero on real cycles; 20 optimiser steps
reduce the MSE for both arms with no NaN/Inf.

**F — cost / vectorisation.** Word census, module forward-call counts per
64-molecule batch independent of word count, and measured CPU step cost.

**G — coupling-specific.**
1. `CC_NULL` first forward equals the historical RICE-v0 source (init hash, `u`,
   `J_base`, `Phi`, `κ`, prediction).
2. Historical best checkpoint replay reproduces the recorded Dev predictions/RMSE.
3. Both arms have identical modules, parameter ledger, state-dict keys and random
   initial state.
4. Two bonds with identical bond fields but different endpoint atom types have
   the same `J_base` but different `delta` / `J_real`.
5. Swapping a bond's two endpoints leaves `J_real` unchanged.
6. `J_base`, `delta`, `J_real` are finite and obey `2 / 2 / 4`.
7. A simultaneous 0/1 latent-state swap in every channel leaves `delta` invariant
   and makes `J_real` transform covariantly:
   `R J_real R = R J_base R + delta·[[1,-1],[-1,1]]`. This is a factor-algebra
   covariance check only; the downstream readout is not required to be invariant.
8. The vectorised formula agrees with an independent NumPy reference.

**H — data identity.** Fit 2856 / Dev 504, disjoint, exactly covering official
Train; row/split/data hashes match the historical D0 identity; Fit-only z-score
and graph normalisation; official Valid labels are NaN; mutating Dev labels does
not change the Fit scaler.

## 6. Resource budget and execution

All execution goes through `rr` on `res`; no manual SSH, `sbatch` or ad-hoc
remote launch. Formal training is at most **1.2 GPU·hours** for both arms; the
total including preflight/smoke is at most **1.5 GPU·hours**. A small forward /
backward / short preflight is run first to measure step cost. If the budget
cannot be met, only an equivalent vectorisation optimisation is allowed; model,
samples and candidate enumeration may not change, and the round stops rather
than swapping in a cheaper model.

The protocol, code and config are committed before the formal run. The run
manifest records the clean source commit, host, GPU, torch/CUDA, driver,
parameter ledger, init hashes, data hashes, seed, split and full curves.

## 7. Primary analysis and verdict rule

On the same 504 Dev molecules:

```
Delta = RMSE(CC_NULL, D0 Dev) - RMSE(CC_REAL, D0 Dev)
```

Positive means chemistry-conditioned coupling is better. A paired
molecule-level bootstrap with **5000** resamples reports Delta and its 95% CI.
The CI reflects Dev molecule sampling only, not training-seed variance.

Also reported:

* Fit/Dev RMSE and MAE, best epoch, both complete curves;
* the relation to historical `RICE_CUMULANT` Dev 0.5994;
* descriptive distances to R2-SENTENCE D0 ≈0.55 and GINE D0 ≈0.527;
* `delta` per bond type: absolute magnitude, variance, positive/negative
  fractions;
* within-bond-type coupling variation attributable to different endpoint
  combinations;
* a frozen `CC_REAL`-weights / `CC_NULL`-formula prediction switch, explicitly
  labelled an **off-manifold frozen intervention** and not a retrained control.

Frozen decision:

* **`CHEM_BINDING_PROMISING`** iff all of:
  `Delta ≥ 0.025`, paired bootstrap 95% CI lower bound `> 0`,
  `CC_REAL` Dev RMSE `≤ 0.575`, the coupling is non-degenerate on Dev, and all
  mathematical/gradient/equivalence checks pass.
* `Delta > 0` but not all conditions: **`WEAK_OR_INSUFFICIENT_SIGNAL`**.
* `Delta ≤ 0`: **`NO_GAIN`**.
* Formula, data or control consistency failure (including a `CC_NULL`
  reproduction deviation larger than the pre-registered 0.02 RMSE tolerance that
  cannot be explained as numerical/training fluctuation): **`INVALID_COMPARISON`**.

Non-degenerate coupling is frozen as: on Dev, `max |delta| > 1e-3`,
`std(delta) > 1e-4`, and the fraction of channel values with `|delta| > 1e-3`
is `> 1%`.

`CHEM_BINDING_PROMISING` is not final success. It only means an independent D1
split confirmation is worth running next. No D1 is run here.

## 8. Deliverables and stop condition

`PROTOCOL.md`, `configs/`, `code/`, `tests/`, `analysis/` (checks, summary,
coupling diagnostics, frozen intervention, verdict), `results/` with both
manifests, complete curves, best checkpoints and Fit/Dev predictions, and
`REPORT.md` answering:

1. Did atom-conditioned coupling really change the physical bond potential?
2. Did it improve held-out Dev rather than only Fit?
3. Did it narrow the gap to the strong R2 reference?
4. Should this route be continued or stopped?

The round stops after these two arms, one seed and one D0 split. No second
seed, no D1, no official Valid/Test, no RICE-v2, no hyper-parameter search, no
channel/latent-state/word-size change, and no next research round.