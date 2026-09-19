# Pre-registration — WG-ICSC-v0: Whole-Graph Incidence-Coupled Sparse Coding

Round name: **WG-ICSC-v0**.  Written **before** any formal ZINC run.

Local code → remote A100 compute → local analysis, per the
`remote-research-runner` skill.  Official ZINC `test` is **never** loaded,
instantiated or referenced by any stage of this round.  Architecture /
sparsity / stopping decisions use official `train` + `valid` only.

This note freezes the objective, the calibration rule and the decision gates
before the implementation is deployed.  It does **not** modify any historical
record.

---

## 0. The single question this round asks

> Can a variable-size attributed molecule be represented by a **sparse
> graph-level support over shared node/bond dictionary primitives**, where the
> sparse codes are inferred **jointly over the full true node–bond incidence
> structure** rather than independently from local attributes — and does that
> mechanism actually do the work?

This is a formal **architecture feasibility experiment**, not a leaderboard
attempt.  It deliberately uses no hand-crafted patch, no GNN encoder, no
canonical node slots, no graph matching, no positional encoding.

## 1. Scientific hypothesis (frozen before the run)

**Main hypothesis.**

> A variable-size attributed molecule can be represented by a sparse
> graph-level support over shared node/bond dictionary primitives, where the
> sparse codes are inferred jointly over the full true node–bond incidence
> structure rather than independently from local attributes.

**Mechanistic prediction.**  If the mechanism is real then:

1. `WG-ICSC` should measurably depend on the real bond↔endpoint assignment;
2. the learned graph code `α_G` should keep genuine sparsity;
3. setting `γ = 0` (no incidence coupling) should hurt the representation;
4. a dense control (`λ₁ = λ_g = 0`) should show whether the sparse constraint
   contributes at all;
5. dictionary / composition gradients must stay active over training.

No "must outperform model X" claim is pre-registered.

## 2. Confirmed graph semantics (from the current repo loader, not textbook ZINC)

Read directly from `torch_geometric.datasets.ZINC(subset=True, split=...)`
(official PyG splits) across train+val:

| quantity | value |
|---|---|
| `x` (atom type) | integer categories `{0,…,20}` → **d_V = 21** |
| `edge_attr` (bond type) | integer categories `{1,2,3}` → **d_E = 3** |
| `edge_index` | each undirected bond stored as **two** directed entries |
| self-loops | none |
| multi-edges | none (directed pairs unique) |
| directed-copy consistency | every bond's two copies have equal `edge_attr` |
| graph size (train+val) | min 9, median 23, p95 31, max 37 |

A bond object is formed by deduplicating the two directed entries and
validating the two `edge_attr` copies agree.  If they ever disagreed the run
would stop, not guess.  Bond type is re-indexed `{1,2,3} → {0,1,2}`.

## 3. Model — frozen architecture

Per molecule `G = (X_V, X_E, B)`:

* node one-hot `X_V ∈ R^{d_V × n}`, bond one-hot `X_E ∈ R^{d_E × m}`
  (raw categorical; **no** learnable encoder, MLP, GNN or descriptor).
* incidence `B ∈ {0,1}^{n×m}`, column-normalised `Q_{ve} = B_{ve}/Σ_u B_{ue}`;
  for a normal bond `Q_{ue} = Q_{ve} = 1/2`.  Not materialised as dense
  `B`/`Q`; endpoint-index / scatter arithmetic must be **exactly equivalent**.
* shared trainable objects: `D_V ∈ R^{d_V × K_V}`, `D_E ∈ R^{d_E × K_E}`,
  `W ∈ R^{K_E × K_V}`, with `K_V = 128`, `K_E = 64`.
  `D_V`, `D_E` are column-normalised every forward,
  `d_k ← d_k/(‖d_k‖₂+ε)`; `W` is not hard-normalised but is weight-decayed and
  its spectral/Frobenius norm is logged.
* per-graph latent `Z_V ∈ R_+^{K_V × n}`, `Z_E ∈ R_+^{K_E × m}`, initialised
  `Z_V⁰ = Z_E⁰ = 0`, re-inferred from zero on **every** forward by an unrolled
  solver.  Not trainable embeddings, not dataset-level parameters.

Lower-level whole-graph objective computed by the solver:

```
E_G(Z_V, Z_E) =
    1/(2n) ‖X_V − D_V Z_V‖_F²
  + 1/(2m) ‖X_E − D_E Z_E‖_F²
  + γ/(2m) ‖Z_E − W Z_V Q‖_F²
  + ρ/2 ( ‖Z_V‖_F²/n + ‖Z_E‖_F²/m )
  + λ₁ ( ‖Z_V‖₁/n + ‖Z_E‖₁/m )
  + λ_g ( ‖Z_V‖_{2,1}/√n + ‖Z_E‖_{2,1}/√m ),      Z_V, Z_E ≥ 0
```

with `γ = 1`, `ρ = 1e-3`; `λ₁, λ_g` calibrated once (Sec. 4) then frozen.

**Solver.** `T = 8` steps, parameters shared across steps; each step is a
block proximal-gradient update (update `Z_V`, then update `Z_E` with the new
`Z_V`).  Smooth gradients:

```
∇_{Z_V}F = 1/n D_Vᵀ(D_V Z_V − X_V)
         + γ/m Wᵀ(W Z_V Q − Z_E)Qᵀ
         + ρ/n Z_V
∇_{Z_E}F = 1/m D_Eᵀ(D_E Z_E − X_E)
         + γ/m (Z_E − W Z_V Q)
         + ρ/m Z_E
```

**Step size.**  Conservative graph-specific block-Lipschitz upper bound.
With detached `s_V=‖D_V‖₂`, `s_E=‖D_E‖₂`, `s_W=‖W‖₂` and
`q_bound = max_v Σ_e Q_ve`:

```
L_V = s_V²/n + γ s_W² q_bound/m + ρ/n
L_E = s_E²/m + γ/m + ρ/m
η_V = 0.9/(L_V+ε),   η_E = 0.9/(L_E+ε)
```

Spectral norms are used **only** for step-size control (detached).

**Proximal operator** (`node` shown; edge identical with `n↔m`):

```
U_V  = Z_V − η_V ∇_{Z_V}F
Ũ_V  = [U_V − η_V λ₁/n]_+
Z_V[k,:] = ( 1 − (η_V λ_g/√n)/(‖Ũ_V[k,:]‖₂+ε) )_+ · Ũ_V[k,:]
```

Exact zeros (no sigmoid / soft gate).

**Graph code.**  `α_G = [α_G^V ; α_G^E] ∈ R^{192}` with
`α_G^V[k] = ‖Z_V[k,:]‖₂`, `α_G^E[l] = ‖Z_E[l,:]‖₂`.  No canonical ordering, no
extra pooling, no hand-crafted graph statistics; row-L₂ amplitudes keep
graph-size information (not divided by √n).

**Head.**  `Linear(192,64) → SiLU → Linear(64,1)`.  No attention / residual /
normalisation / auxiliary branch.

**Outer loss.**  `L_task = MAE(ŷ,y)`, `L_fit =` graph-level mean of the
reconstruction + composition residual (using the **final** `T`-th codes, no
L₁/L_{2,1} re-added).  `L_train = L_task + μ L_fit`, `μ = 0.1`.

## 4. Sparsity calibration (train-only, unlabelled, once)

Fixed seed, ~256 **train-only** graphs, at model-initialisation state, no `y`.
Grid: `λ₁ ∈ {0.01, 0.03, 0.10}`, `λ_g ∈ {0.02, 0.05, 0.10}` (9 combinations).
Select the combination numerically stable and closest to:

* whole-graph active dictionary-row fraction **15 %–30 %**;
* element-wise nonzero fraction **within active rows** roughly **20 %–50 %**.

If all 9 combos are uniformly too dense or all collapse, multiply/divide the
whole grid by 3 **once** and repeat.  Then `λ₁, λ_g` are **frozen**.  All
calibration results are recorded.  Validation MAE is never used to choose λ.

## 5. Training protocol (canonical, reused)

`OPTIMIZED_PROTOCOL`: Adam, lr 1e-3, weight decay 1e-5, batch 128, L1/MAE,
grad-clip 5.0, no scheduler, max 240 epochs, patience 40, best official-valid
checkpoint, fixed equal-weight **Top-5 soup** (no K / weight search).  Data:
official PyG ZINC `subset=True` train (10 000) + valid (1 000).  Seed 0 only in
the first formal run.  CUDA/A100 is the execution regime.

## 6. Implementation order (non-negotiable)

1. per-graph **correctness-first reference solver** (dense small matrices);
2. **batched GPU solver**;
3. targeted test asserting `max|Z_reference − Z_batch|` within numerical
   tolerance;
4. targeted tests A–G (below);
5. A100 smoke (forward/backward/step) + 64-graph train overfit sanity;
6. formal seed-0, valid-only.

Targeted tests: (A) shape/finite/non-negativity; (B) exact row zeros from the
proximal operator; (C) node- and bond-object permutation equivariance of
`Z_V, Z_E`; (D) graph-code invariance `α'_G = α_G` (< 1e-5); (E) composition
endpoint test — same node/edge feature multiset, different `Q` ⇒ different
`α_G`; (F) solver descent diagnostic (lower-level objective non-increasing at
the safe step size); (G) gradient flow to `D_V, D_E, W`, head.

## 7. Recorded mechanism metrics (every epoch / final)

sparse support (node/edge active-row fraction; mean/median/quartiles);
element sparsity; dictionary utilisation (per-atom activation frequency, mean
amplitude, dead atoms); dictionary coherence / singular values / effective
rank; `W` singular values / spectral norm; inference dynamics per step
(`t=0…8`: node & edge reconstruction error, composition residual, total lower
energy, active-row fraction, relative code change `‖Z^{t+1}−Z^t‖/‖Z^t‖`).
Evaluation-only `T=16` re-run on the trained model (no T=16 retraining).

## 8. Mechanism-vitality interventions (evaluation-only)

* **Intervention 1 — bond-assignment permutation.**  Keep `X_V`, `X_E`,
  topology and the number of bond objects fixed; randomly permute the columns
  of `Q` re-assigning bond attributes/latents to different endpoint pairs
  (without permuting `X_E`).  Report relative change
  `‖α^orig − α^shuf‖₂ / (‖α^orig‖₂+ε)` and prediction change.
* **Intervention 2 — coupling-gradient vitality.**  Per inference step report
  `‖g_comp‖ / (‖g_attr‖+ε)` for the node side (and the edge-side analogue),
  where `g_attr = 1/n D_Vᵀ(D_V Z_V−X_V)`,
  `g_comp = γ/m Wᵀ(W Z_V Q−Z_E)Qᵀ`.

If assignment shuffling barely changes the code, or `‖g_comp‖` is two orders
of magnitude smaller than `‖g_attr‖`, the mechanism is judged **not used** even
if MAE looks reasonable.

## 9. Decision gates (seed 0, valid only)

* **Case A — clear failure** (stop, analyse only): NaN/instability; sparsity
  ≈ 0; active groups persistently > 90 %; broad dictionary collapse; `W` /
  composition gradient dead; train loss barely moves; valid Top-5 soup
  clearly bad (e.g. > 0.15).
* **Case B — mechanism healthy, soup 0.13–0.15**: borderline feasibility; no
  scale-up; run the two matched seed-0 controls to attribute the limit.
* **Case C — mechanism healthy, soup ≤ 0.13**: worth continuing; run seeds 1, 2
  and the two seed-0 controls.

No sweeps of `K`, `T`, head depth, hidden width, encoder or message passing.

## 10. The only permitted controls (if the main gate passes)

* **Control A — No-incidence** (`γ = 0`; `W` kept for identical parameter
  shape but unused): does the real node–bond incidence composition add value?
* **Control B — Dense** (`λ₁ = λ_g = 0`): does the sparse constraint add value
  over a plain continuous latent incidence model?

Everything else identical (params, dictionaries, T, head, protocol).

## 11. Baseline policy

Matched **A100** Top-5 soup references under the `zinc-context-gap` protocol
are cited for orientation only (`B-Full` seed0 valid soup 0.119818; `A0` seed0
0.124704; commit/protocol recorded from `zinc_local_token_null.REFERENCE_SOUP`).
No historical CPU micro-delta is used as an architecture claim.  Official
`test` is never read by any stage; on completion the official test read count
is unchanged.

## 12. Deliverables

implementation; targeted tests; this preregistration; seed-0 / full result
configs; machine-readable metrics; support / utilisation / solver-dynamics
plots; mechanism-intervention results; final analysis note (five questions);
commit hash, GPU, seed, exact config, data splits, unchanged test-read count.
