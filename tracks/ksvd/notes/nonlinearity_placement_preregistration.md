# Pre-registration — Nonlinearity Placement Audit (NPA)

Round name: **NPA**.  Written **before** any formal ZINC run.

Local code → remote A100 compute → local analysis, per the
`remote-research-runner` skill.  Official ZINC `test` is **never** loaded,
instantiated or referenced by any stage of this round.  Architecture /
bandwidth / stopping decisions use official `train` + `valid` only.

This note freezes the objective, the three lifts, the readout, the protocol and
the decision gates before the implementation is deployed.  It does **not**
modify any historical record.

---

## 0. The single question this round asks

> DOI/AIOM stack many propagation steps as `H_t = S^t X` with **no nonlinear
> discrimination between them**.  Is adding nonlinearity *between* propagation
> steps already enough — or must the nonlinear structural coordinates be
> **task-coupled**?

This is a diagnostic representation audit, not a leaderboard attempt.  It uses
no dictionary, ISTA, K-SVD, GNN backbone, attention, graph transformer, learned
pooling, graph token, canonical node IDs, patch descriptors, shortest paths or
hand-crafted graph statistics.

## 1. Confirmed graph semantics (from the current repo loader)

Read directly from `torch_geometric.datasets.ZINC(subset=True, split=...)`:

| quantity | value |
|---|---|
| `x` (atom type) | integer categories `{0,…,20}` → **C_V = 21** |
| `edge_attr` (bond type) | integer categories `{1,2,3}` → **C_E = 3** |
| incidence graph | `A = [[0,B],[Bᵀ,0]]`, `S = D^{-1/2} A D^{-1/2}`, `X` raw one-hot |
| bond object | one undirected chemical bond (two directed copies deduplicated) |
| self-loops / multi-edges | none (asserted on a 500-graph audit) |

Confirmed by the run; if `C_V/C_E` differ the run stops (full splits only).

## 2. The three lifts (frozen)

All three share the *same* graph semantics and the *same* downstream
distributional readout
`Phi = [mu_V; mu_E; mu_I; log(1+n); log(1+m)]` (DOI's object wherever
possible).

**Lift A — Frozen Linear DOI.**  `H_t = S^t X`, `t=0..4`, per-object responses,
RFF/KME.  Reused from `decision-4b8e2d1f`; **not retrained**.  Frozen probe:
train 0.3273 / valid **0.3865**.

**Lift B — Fixed Nonlinear Scattering DOI.**  No trainable structural
parameter.  `P = (I+S)/2`; fixed dyadic wavelets `Psi_0 = I-P`,
`Psi_1 = P-P^2`, `Psi_2 = P^2-P^4`; low-pass `Phi_lp = P^4`.  First order
`U_j = |Psi_j X|`, `j∈{0,1,2}`; second order for ordered increasing paths
`(0,1),(0,2),(1,2)`: `U_{j1,j2} = |Psi_{j2} U_{j1}|`.  Per-object response
`r_i = [X_i; U_0; U_1; U_2; U_{01}; U_{02}; U_{12}; L_i]` (8 blocks × 24 = 192D).
Then the exact DOI distribution construction (atom/bond objects + real
incidence pairs).

**Lift C — Tiny Task-Coupled Incidence Lift.**  `h = 16`, `L = 2` atom–bond
alternating `tanh` rounds; only local coordinates `{h_v}`, `{e_e}`; no pooling,
no graph token, no graph-level state.
`e_e <- tanh(W_EE e_e + W_VE(h_u+h_v) + b_E)`,
`h_v <- tanh(W_VV h_v + W_EV Σ_{e∋v} e_e + b_V)`.  Sum (not mean).  No
residual / LayerNorm / dropout.  Structural parameter budget **~2.5K** (input
tables 384 + 2 × 1056 = 2496); if it exceeds 4K the run reports a module-budget
violation.

## 3. Readout (frozen)

Scattering and learned lift use the fixed KME/RFF readout with
`D_RFF = 1024` **per distribution**; `Phi ∈ R^{3074}` (identical to DOI T=4).
Prediction head, identical for both:
`Linear(3074, 64) → SiLU → Linear(64, 1)`.

* Scattering: train-only RMS coordinate scaling, train-only median-heuristic
  bandwidths, fixed-seed RFF; **only the head is trained**.
* Learned: the lift + fixed KME + head are trained **end-to-end** with **MAE
  only** (no reconstruction / smoothness / contrastive / WL / auxiliary loss).
  Bandwidths are estimated once from the **seed-0 init** coordinates
  (train-only median pair distance) and the RFF frequencies are frozen.  The
  head sees **only** `Phi_theta(G)` (no raw `h_v`, no direct pooling).
* The learned RFF dimension is **fixed at 1024** (section 18 of the task).

## 4. Audits (frozen)

* **Correctness (data-free, CPU float64):** scattering permutation equivariance
  (<1e-10); scattering KME invariance; learned lift equivariance and KME
  invariance; incidence sensitivity (same atom/bond multisets + counts,
  different endpoint assignment → responses and `Phi` differ); finiteness.
* **RFF adequacy:** exact Gaussian KME inner product vs RFF on ≥300–500
  graph-pairs; require median<0.02 and p95<0.05 for all three kernels
  (scattering dimension rule).  The learned lift is audited **after training**;
  only `p95 > 0.10` marks it representation/RFF-confounded (no bandwidth
  re-estimation, no rerun).
* **Unified geometry (500 train graphs):** degree-preserving 2-switch,
  bond-type reassignment, atom substitution, pure permutation, random
  size-matched pairs; standardized `Δ_Φ`; report
  `R_topo = median d(random) / median d(2-switch)`.
* **WL geometry:** frozen 3-round attributed WL histogram; Φ↔WL distance
  Spearman, top-1/top-10 WL neighbour similarity, size-matched random control.
* **Learned mechanism:** within-graph 2-switch, layer-wise
  `Δ_h0, Δ_e0, Δ_e1, Δ_h1, Δ_e2, Δ_h2, Δ_Φ`.
* **64-graph smoke:** gradients flow, loss descends, no NaN (not an
  architecture-selection gate).

## 5. Frozen decision gates (declared before seeing the result)

Let `S` = scattering valid MAE, `L` = learned-lift valid MAE, DOI valid = 0.3865.

* **Gate A (fixed nonlinearity suffices):** `S ≤ 0.25` and train also clearly
  down → prefer the **fixed nonlinear operator lift**; do not adopt the learned
  lift as main line.
* **Gate B (task-coupled lift necessary):** `S > 0.30` but `L ≤ 0.20` →
  **task-coupled local coordinate formation is necessary**.
* **Gate C (learned lift in the middle):** `0.20 < L ≤ 0.25` with clear train
  drop and stronger rewiring sensitivity and ~2.5K structural params →
  *minimal task-coupled lift promising but not yet dictionary-ready*.
* **Gate D (even the tiny learned lift fails):** `L > 0.25` and train also high
  → a minimally parameterised pre-dictionary structural lift is insufficient
  for ZINC.

**Gated no-message control:** only if `L < DOI_valid − 0.10` (i.e. `L < 0.2865`)
is an extra seed-0 control run with `W_VE^l = W_EV^l = 0` (no atom↔bond
communication), everything else identical.

## 6. Out of scope (this round)

No dictionary, sparse coding, depth/width sweeps, residual, LayerNorm,
attention, edge gates, hidden 32/64, learned RFF/bandwidth, graph pooling,
`T>4`, higher scattering orders, or different wavelet families.  The learned
lift has exactly one architecture.

## 7. Deliverables

Machine-readable `results/nonlinearity_placement/`: scattering filter metadata,
RFF bandwidth/seed metadata, learned-lift config + parameter counts, kernel
approximation audits, perturbation + rewiring metrics, learned layer-wise
metrics, WL geometry, probe results, plots.  Plus a post-run analysis note and
a decision record.

## 8. Final report must answer

1. Does interleaved fixed nonlinearity significantly fix DOI?
2. If not, does the tiny task-coupled incidence lift fix it?
3. If the learned lift works, does the gain depend on real atom↔bond
   communication (gated control)?
4. Which hypothesis is true: (A) fixed linear diffusion was the problem;
   (B) task-coupled local coordinate formation is necessary; (C) even a minimal
   task-coupled lift lacks capacity.
