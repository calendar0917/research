# FEC-D1 — pre-registration (frozen)

Round: **FEC-D1** (*Localized Sparse Structural Dictionary Binding*).
Protocol id: `fec_d1`. Study: `zinc-context-gap`.
Prior-artifact audit: [`fec_d1_prior_artifact_audit.md`](fec_d1_prior_artifact_audit.md).

This pre-registration is written **before any code is committed** and freezes:
the single question, the frozen artifacts, the two (and only two) arms, the
statistic, the branch, the injection point, the training protocol, the
evaluation-only mechanism probe, the gates and the verdicts. Nothing below may
be changed after the first formal run; any change requires a new round id.

`official_test_loaded = false` is mandatory.

---

## 1. Single question

> An independently validated reusable sparse pure-topology dictionary
> coordinate `alpha_v` (SDB-v0: `phi^65 → D^{65×32}`, `s=8`, 32/32 atoms used,
> FSAR role recovery vs `phi65` = 1.159, assignment-shuffle degradation 0.245),
> if **localized to each rooted chemical environment** and **bound to primitive
> atom chemistry during environment formation**, does it give a strong frozen
> FEC-S1 a *dictionary-specific* predictive gain?

This is **not** a whole-graph residual (SDB Stage-4: `+0.002381`, below gate),
**not** a pair-kernel dictionary (SDPK/SRDA), **not** a mixed chemistry
dictionary (TCCD), **not** the R11 residual design (FEC-D0: STOP).

---

## 2. Frozen artifacts (read-only; no refit)

| object | definition | path |
|---|---|---|
| `D_SDB ∈ R^{65×32}` | SDB-v0 Stage-1 frozen K-SVD, `K=32`, `s=8`, `DICT_SEED=20260924` | `results/sdb_v0/dictionary.pt::D_ksvd` |
| `PCA32` | SDB-v0 train-fit affine PCA, rank 32 | `results/sdb_v0/dictionary.pt::{pca_mean,pca_components}` |
| `phi_v^65` | `fsar_r2_ar0.build_phi`, pure topology, node order = patch order | `results/fsar_r2_ar0/cache/*.pkl.gz` |
| `alpha_v^32` | `sdb_v0.omp_codes(D_SDB, phi_v, s=8)` | recomputed deterministically |
| FEC-S1 base | seed-0 selection checkpoint (`best_valid_mae = 0.1367825070246472 @238`) | `results/fec_s1/states/fec_s1_seed0_selection_state.pt` |

Forbidden: refit K-SVD, change `K`, change `s`, change the solver, change
normalization, task-couple `D`, backprop through the sparse solver, use R11,
add an edge dictionary. `D_SDB` stays chemistry-free, label-free, node-centric.

**G0 identity requirement.** On a fixed set of nodes:
`phi65` identical to the FSAR cache, `alpha` identical to a fresh
`omp_codes(D_SDB, phi65, s=8)`, exact `l0 = 8`.

---

## 3. Frozen dense control

`z_v^P = PCA32(phi_v) ∈ R^32`, the SDB-v0 train-fit affine PCA, frozen. It is
the dictionary-specific matched dense control: same `R^65` information source,
same 32 dimensions, same branch, same protocol. The two arms differ **only** in
`alpha_v` vs `z_v^P`.

---

## 4. Localized structural role

`alpha_v` is node-centric; the environment is root-conditioned. For root `i`:

```
s_iv = shell(i, v) ∈ {0,1,2}
```

is the BFS distance from the root inside the radius-2 patch (node order =
`graph.nodes`, patch order = root order). The FEC-D1 role is `(s_iv, alpha_v)`;
`shell` is **not** re-fed to the dictionary.

---

## 5. Localized assignment statistic

For root `i` and shell `k`:

```
S_ik = { v in P_i : shell(i,v) = k }
```

with `a_v = alpha_v ∈ R^32`, `q_v = onehot(atom_v) ∈ R^28`:

```
C^D_ik = Σ_{v∈S_ik} (a_v - ā_ik)(q_v - q̄_ik)^T ∈ R^{32×28}
C^D_ik = 0                       if |S_ik| < 2
C_i^D  = [ C^D_i0 ; C^D_i1 ; C^D_i2 ] ∈ R^{2688}   (shell-major flatten)
```

`q̄_ik = mean_{v∈S_ik} q_v`, `ā_ik = mean_{v∈S_ik} a_v`. Within-shell centring
is mandatory: FEC-S1 already carries the coarse shell × chemistry marginals,
so this statistic may only carry the **subrole ↔ chemistry assignment**.

### 5.1 PCA matched statistic

Identical definition with `z_v^P` in place of `alpha_v`:

```
C^P_ik = Σ_{v∈S_ik} (z_v^P - z̄^P_ik)(q_v - q̄_ik)^T ,  C_i^P ∈ R^{2688}
```

### 5.2 Train-only scaling

For each arm independently, a train-only per-coordinate RMS scaler:

```
C̃_j = (C_j / sqrt(E_train[C_j^2] + eps)) * mask_j
mask_j = 1[ sqrt(E_train[C_j^2]) > floor ]
```

No mean subtraction (within-shell centring already happened inside the
molecule). Official-valid is **not** used to fit the scaler. Effective/zero
coordinates and the RMS range are recorded.

---

## 6. Branch (small, matched, rank-8)

```
u_i  = W1 · C̃_i          W1 : R^2688 → R^8   (bias = false)
Δe_i = W2 · u_i          W2 : R^8 → R^24     (bias = false)
```

No nonlinearity. Effective linear rank ≤ 8. Dict and PCA have **identical**
parameter counts `8*2688 + 24*8 = 21696`.

Initialization (frozen): `W1` deterministic standard `nn.Linear` init
(`kaiming_uniform_(a=√5)` under `torch.manual_seed(0)`), `W2 = 0` exactly, so
step 0 gives `Δe_i = 0`.

Rank, width, bias and init are **not swept**.

---

## 7. Injection point

FEC-S1 computes `A(x_i^146) = e_i^shared ∈ R^24`. FEC-D1 defines

```
e_i^new = e_i^shared + Δe_i
```

and then reuses the frozen FEC-S1 local encoder / pair system / reader. The
dictionary statistic is **never** added to the graph prediction, never injected
into the pair kernel, never used for a centre update, recurrence or context
writeback. Formally:

```
dictionary → local structure × chemistry binding → local environment → static composition
```

---

## 8. Frozen FEC-S1 base

All `66170` FEC-S1 parameters have `requires_grad = false`. The only trainable
parameters are `W1` and `W2`. `D_SDB` and `PCA32` transforms are frozen.

Matched baseline: `M_B = ` exact replay of the seed-0 best checkpoint.
Recorded value `0.136783 @238`. Baseline guard tolerance `1e-5`; if the replay
does not match, **STOP**.

`g = 0 (W2 = 0) ⇒ exact FEC-S1` must hold (bit-identical preferred).

Historical FEC-S1 Top-5 soup `0.130422` is **external strong-anchor context
only**, not the matched baseline.

---

## 9. Hard correctness gates (all before formal training; any FAIL ⇒ STOP)

* **G0 SDB code identity** — fixed nodes: `phi65` identical, `alpha` identical,
  exact `l0 = 8`.
* **G1 dictionary chemistry purity** — changing atom/bond categories while
  keeping topology leaves `phi65` and `alpha` unchanged.
* **G2 root localization** — the same node `v` in different root patches has
  identical `alpha_v`; `shell(i,v)` may differ.
* **G3 binding reference** — a toy patch's `C_ik` matches an independent
  `float64` numpy hand computation.
* **G4 centring** — within each root/shell, permuting the `alpha`↔`q`
  correspondence keeps both multisets but changes `C_ik` (the statistic is
  genuinely an assignment statistic).
* **G5 exact baseline containment** — with `W2 = 0`: local 24-D, `h`, pair
  state, reader state and prediction all equal FEC-S1; prediction
  **bit-identical**.
* **G6 no direct readout bypass** — forward-hook/AST proof that `C_i` reaches
  the model only through `C_i → W1 → W2 →` the local 24-D slot.
* **G7 gradients** — on a real batch, `W2` grad finite and nonzero; after ≥1
  optimizer step, `W1` grad finite and nonzero.
* **G8 strict-static purity** — no MP, no recurrence, no pair→centre, no
  context writeback; mutating pair tensors leaves the dictionary local binding
  and the local environment pre-pair object unchanged.
* **G9 official-test blocker** — any `test` split access raises.

---

## 10. Formal arms (exactly two)

```
D — Dict32 localized binding   (alpha_v)
P — PCA32 localized binding    (z_v^P)
```

Not run: new FEC-S1, Random32, Dense65, task-coupled dictionary, edge
dictionary, R11 dictionary, graph residual, null-capacity arm. The frozen
baseline is provided by the `W2 = 0` exact containment of G5.

---

## 11. Training protocol (inherited from FEC-S1)

```
official train = 10000 ; official valid = 1000 ; official test = NEVER
seed = 0 ; Adam ; lr = 1e-3 ; weight_decay = 1e-5 ; batch = 128
loss = L1 / MAE ; grad clip = 5 ; no scheduler
240 epochs ; no early termination
fixed Top-5 soup over branch checkpoints
```

Base checkpoint frozen throughout. The two arms share the seed, optimizer,
branch architecture, batch order and protocol; they may run in parallel on
GPU0/GPU1 after a deterministic GPU smoke.

---

## 12. Primary metrics

```
M_D = Dict Top-5 soup official-valid MAE
M_P = PCA  Top-5 soup official-valid MAE
M_B = frozen FEC-S1 best-checkpoint replay MAE (0.136783...)
G_D             = M_B - M_D              (local dictionary gain)
G_dict-specific = M_P - M_D              (dictionary-specific advantage)
```

---

## 13. Evaluation-only assignment shuffle (Dict soup only)

After training, for each root `i` and shell `k` independently permute the `q_v`
against `alpha_v` correspondence, keeping node set, shell, `alpha` multiset,
`q` multiset and the FEC-S1 base input unchanged; recompute `C_i^D`. Five fixed
permutations. Report `M_shuffle`, `G_assign = M_shuffle - M_D`, and the
mean/max prediction shift. The FEC-S1 base chemistry is never modified.

---

## 14. Branch neutralization

Force `Δe_i = 0` in the trained Dict branch; the prediction must return to the
frozen FEC-S1 base. Report `max |pred_neutral - pred_base|` (target: floating
tolerance zero). This proves the candidate gain comes from the localized
branch, not from baseline drift.

---

## 15. Frozen success gates

* **Gate A — material local gain**: `G_D ≥ 0.003` ⇔ `M_D ≤ M_B - 0.003`
  (≈ `M_D ≤ 0.133783`).
* **Gate B — dictionary-specific advantage**: `M_P - M_D ≥ 0.002`.
* **Gate C — assignment mechanism**: `M_shuffle - M_D ≥ 0.010`.

## 16. Frozen verdicts (in order)

| condition | verdict |
|---|---|
| A ∧ B ∧ C | `FEC_D1_LOCALIZED_DICTIONARY_BINDING_SUPPORTED` |
| A ∧ ¬B | `FEC_D1_LOCAL_BINDING_SUPPORTED_DICTIONARY_NOT_SPECIFIC` (STOP dictionary performance route) |
| A ∧ B ∧ ¬C | `FEC_D1_DICTIONARY_GAIN_NOT_ASSIGNMENT_MEDIATED` (STOP) |
| ¬A | `FEC_D1_NO_MATERIAL_LOCAL_DICTIONARY_GAIN` (STOP) |

No rescue: no `K`/`s` sweep, no rank-16, no nonlinear branch, no LayerNorm, no
attention, no extra epochs, no task-coupled `D`, no edge `D`, no `R65` redesign,
no R11 revival, no pair-kernel injection, no graph residual, no reader widening,
no FEC-S1 finetuning.

---

## 17. Reporting

Always also report `M_D - 0.130422` against the historical FEC-S1 soup anchor,
honestly labelled as external context. If `M_D < 0.130422`, record
`new seed0 strict-static anchor` — but the official test stays closed.

Six mandatory answers: (Q1) exact SDB `R65→K32/s8` reuse; (Q2) dictionary reads
pure topology and meets chemistry only at environment formation; (Q3) localized
Dict gain ≥ 0.003 vs matched frozen base; (Q4) Dict better than PCA32 by
≥ 0.002; (Q5) assignment shuffle worsens MAE by ≥ 0.010; (Q6) one of the four
verdicts above.

---

## 18. Durable artifacts

```
notes/fec_d1_prior_artifact_audit.md
notes/fec_d1_preregistration.md
notes/fec_d1_implementation.md
notes/fec_d1_analysis.md

results/fec_d1/
    artifact_identity.json
    correctness.json
    parameter_accounting.json
    baseline_guard.json
    binding_scalers.json
    dict_seed0.json
    pca_seed0.json
    dict_curve.csv
    pca_curve.csv
    mechanism.json
    states/
    REPORT.md
    DECISION.md
    decision.json
```

plus claim YAML, decision YAML and a `STATE.yaml` update. JSON under `results/`
follows the repo `.gitignore` convention (not force-added).
