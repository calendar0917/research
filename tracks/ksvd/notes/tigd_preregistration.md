# Pre-registration — TIGD-v0: Typed Incidence Graph Dictionary unmixing

Round name: **TIGD-v0** (Typed Incidence Graph Dictionary). Written **before**
any formal ZINC run, following the `remote-research-runner` skill.

Local code → remote A100 compute → local analysis. Official ZINC `test` is
**never** loaded, instantiated or referenced by any stage. All decisions use
official `train` + `valid` only. This note does **not** modify any historical
record.

---

## 0. The single question this round asks

Previous rounds (canonical coordinates, WG-ICSC, I-CRATE, AIOM, DOI, NPA)
all tried to remove variable-size / permutation / graph structure *before* the
dictionary. TIGD asks the opposite:

> Can a shared dictionary whose atoms are **whole attributed incidence graphs**
> unmix real ZINC molecules through **graph-specific soft transport**, producing
> stable, discriminative, topology-sensitive whole-graph codes?

This round is an **object feasibility audit only**. No task-driven dictionary,
no backprop of ZINC labels into the dictionary, no GNN, no RFF, no AIOM/DOI, no
official test, and **no sparsity penalty** yet. If and only if this object is
viable may the next round add sparsity + downstream task coupling.

## 1. Confirmed graph semantics (from the current repo loader)

| quantity | value |
|---|---|
| `x` (atom type) | integer categories `{0..20}` → **C_V = 21** |
| `edge_attr` (bond type) | integer categories `{1,2,3}` → **C_E = 3** |
| bond object | one undirected chemical bond (directed copies deduplicated) |
| `n` range (train) | 9–37 |
| `m` range (train) | 8–41 |
| `y` (log solubility) | mean 0.015, std 2.01, range −42.0–3.80 |

`B ∈ {0,1}^{n×m}` has `Σ_v B_ve = 2` for every bond column. Inputs are the raw
one-hot `X_V ∈ {0,1}^{n×21}`, `X_E ∈ {0,1}^{m×3}`. **No** degree/ring/path/
hand-crafted/canonical-ID features.

## 2. Dictionary atom = whole attributed incidence graph (frozen sizes)

`q_V = q_E = 24`, `K = 16`. Atom `k`: `A_k^V ∈ [0,1]^{24×21}`,
`A_k^E ∈ [0,1]^{24×3}`, `R_k ∈ [0,2]^{24×24}`.

* `A_k^V[a,:] = softmax(Θ_k^V[a,:])`, `A_k^E[b,:] = softmax(Θ_k^E[b,:])`.
* `R_k[:,b] = 2 · softmax_a(Ξ_k[:,b])`, so `Σ_a R_k[a,b] = 2` exactly.

## 3. Graph code, mixture, transport (frozen)

`α_G ∈ Δ_16` (no sparsity penalty this round). Mixture:
`Ā^V = Σ_k α_k A_k^V`, `Ā^E = Σ_k α_k A_k^E`, `R̄ = Σ_k α_k R_k`
(so `Σ_a R̄[a,b] = 2`).

Transport (graph-specific soft alignment):
`P_V ∈ Π(1/n·1_n, 1/24·1_24)`, `P_E ∈ Π(1/m·1_m, 1/24·1_24)`.
Atoms match atom slots, bonds match bond slots; a single generic node transport
is not used.

## 4. Objective (fixed, λ=1, not tuned)

```
L_V = Σ_{v,a} P_V[v,a] ||X_V[v] − Ā^V[a]||²
L_E = Σ_{e,b} P_E[e,b] ||X_E[e] − Ā^E[b]||²
L_B = Σ_{v,e,a,b} P_V[v,a] P_E[e,b] (B_ve − R̄_ab)²
E_G = L_V + L_E + L_B
```

No categorical CE (quadratic / interpretable only). Three term magnitudes are
recorded; not auto-tuned.

## 5. Inner solver (frozen)

Entropic OT, `ε = 0.05`, log-domain Sinkhorn, **50 iterations** per transport
update. Alternating cycles `T_alt = 8`, each: update `P_V`, update `P_E`, then
`5` projected-gradient `α` steps. `α` step initial `η = 1`, deterministic
backtracking (`η ← η/2`, ≤ 12 halvings, accept iff
`E(α_new) ≤ E(α_old) + 1e-10`), exact Euclidean simplex projection.

Init: `α⁰ = 1/K`, `P_V⁰ = 1/(24n)`, `P_E⁰ = 1/(24m)`.

## 6. Dictionary initialisation (seed 0, frozen)

`Θ^V = log(train empirical atom freq) + 0.25·N(0,1)` (independent per slot),
`Θ^E = log(train empirical bond freq) + 0.25·N(0,1)`,
`Ξ ~ N(0, 0.5²)`. No canonical/K-means/validation/multi-seed init.

## 7. Dictionary learning (classical block coordinate, frozen)

For each train minibatch: (1) `no_grad` infer `α, P_V, P_E`; (2) detach them;
(3) recompute `E_G` with them fixed; (4) backprop into `Θ^V, Θ^E, Ξ` only.
Optimizer Adam, `lr = 1e-3`, `weight_decay = 0`. **No** unrolled backprop, no
bilevel/task gradient.

## 8. Stages (frozen)

* **Stage 1 pilot**: 2048 train graphs (fixed seed selection), 40 epochs. Log
  every 5 epochs: total energy; `L_V, L_E, L_B`; code entropy / effective
  support `exp(H)` / max coefficient; atom utilization; pairwise atom distance;
  transport entropy / max-entry concentration; Sinkhorn marginal errors; α
  backtracking failure rate.
* Pilot **collapse criteria** (stop, no Stage 2): dictionary collapse; code
  collapse; transport collapse (near-uniform and `L_B` flat); structure
  ignored (α with `λ_B=0` ≈ full); solver instability (NaN / non-descent /
  frequent backtracking failure).
* **Topology mechanism audit** on 256 train holdout graphs (never used for the
  dictionary update): Original / Permutation (IDs only) / degree-preserving
  2-switch rewiring / size-matched unrelated graph. Report
  `Δ_α = ||α_G − α_{G'}||_1` distributions (ideal
  `Δ_perm ≪ Δ_rewire < Δ_random`) and inferred `E*_G`, with emphasis on
  whether free transport "absorbs" identity (`E_random ≈ E_original`).
* **Structure-off control** (`λ_B = 0` at inference only): compare
  `α^full` vs `α^attr`, code geometry and rewiring sensitivity.
* **Stage 1 gate** (all must hold for Stage 2): no atom collapse; code
  diversity; permutation invariance; rewiring changes α; random change ≥
  rewiring; `L_B` truly decreases; structure-off changes codes; transports not
  uniform; original/random reconstruction separation.
* **Stage 2** (only if gate passes): all 10000 train graphs, same config, 60
  epochs. Freeze dictionary; re-infer all train + valid; full mechanism audit;
  dictionary diversity; visualisation of top atoms (`R_k` soft bipartite).
  No K/q/ε/weight sweep.

## 9. Frozen-code target probe (only use of `y`)

Dictionary and codes frozen. Train only `α_G → y`: (a) linear; (b) tiny MLP
`Linear(16,32) → SiLU → Linear(32,1)`. Head input is **only** `α_G`. Numeric
readout is raw-MAE, comparable to frozen DOI 0.3865 / scattering 0.3657 /
I-CRATE 0.399. Bands: ≤0.30 very promising; 0.30–0.38 promising mechanism;
>0.40 weak. The probe is **not** the sole success criterion.

## 10. Priorities and final answer

Priority order: (1) does transport fail to absorb topology; (2) is `α_G` a
genuinely graph-sensitive whole-graph code; (3) only then the `α_G → y` probe.
The final conclusion is one of the five frozen sentences (viable & active /
structurally viable but task-misaligned / transport absorbs identity / collapse /
inconclusive); the next decision is one of
`proceed to task-driven sparse graph dictionary learning` /
`revise the graph-space reconstruction / transport objective` /
`stop dictionary-as-primary-representation`.

## 11. Prohibited this round

sparse penalty; task-driven dictionary; unrolled backprop through inference;
`K > 16`; `q` sweep; `ε` tuning; GNN; RFF; learned graph lift; official test;
extra descriptors; multiple seeds; task-supervised dictionary init.

## 12. Solver correctness tests (data-free, CPU float64 reference)

A. transport marginals; B. energy descent; C. permutation invariance of `α`;
D. dictionary latent-slot global permutation invariance; E. topology
sensitivity (same feature multisets / same size, different incidence ⇒
different structure energy / transport / α). A single-graph CPU float64 numpy
reference is implemented and cross-checked against the batched torch solver.
