# TIGD-v0 — Typed Incidence Graph Dictionary unmixing: pilot failure analysis

Round **TIGD-v0** (pre-registration: `notes/tigd_preregistration.md`). Local
code → remote A100 compute → local analysis.

Formal run: commit `28bab97`, GPU 0 (A100-40GB), seed 0, wall 151.6 s,
evidence `results/tigd_graph_dictionary/` (`SUMMARY.json`, `pilot/history.json`,
`pilot/mechanism_audit.json`, `pilot/probe.json`, `pilot/pilot_gate.json`,
`pilot/dictionary.npz`, `pilot/codes*.npz`, `pilot/atoms.png`).

**Official ZINC `test` was never loaded, instantiated or referenced.** Only
official `train` (10000) / `valid` (1000) were used; `y` was touched only in the
frozen-code probe. No K/q/ε/weight sweep, no sparsity, no task-driven
dictionary, no unrolled backprop.

Frozen sizes: `K=16`, `q_V=q_E=24`, `ε=0.05`, 50 log-domain Sinkhorn
iterations, `T_alt=8`, 5 α steps, `λ_V=λ_E=λ_B=1`.

---

## 0. Verdict first

> **Soft transport absorbs too much graph identity; this graph-space dictionary
> formulation is not viable.**

The pre-registered pilot gate fails (5/10 checks). The dictionary learns
attribute prototypes but **not** incidence structure; the graph code collapses
to a near one-hot assignment over ~3 of 16 atoms; degree-preserving rewiring
changes the code by the same amount as a pure relabelling (≈ 0); and the
entropic transport never converges to a valid coupling once the atoms sharpen.
Per the pre-registration the full stage (10 000 graphs / 60 epochs) is **not**
run. Next decision: `revise the graph-space reconstruction / transport
objective` (see §9).

---

## 1. What the pilot did

2048 fixed-seed train graphs, 40 epochs, batch 256, Adam lr 1e-3, block
coordinate (no_grad inference → detached `α, P_V, P_E` → recompute `E_G` →
gradient only into `Θ^V, Θ^E, Ξ`). Dictionary init = train empirical category
frequencies + N(0,0.25²) logits, `Ξ ~ N(0,0.5²)`, seed 0.

Per epoch (train): `total`, `L_V = Σ P_V c_V`, `L_E = Σ P_E c_E`,
`L_B = Σ P_V P_E (B − R̄)²`; code entropy / effective support / max coef /
cross-graph pairwise L1; atom utilisation; atom pairwise distances; transport
entropy / max entry; Sinkhorn marginal error; α backtracking failure rate.

## 2. Q1 — does the dictionary lower L_V, L_E, L_B, or only attributes?

| epoch | total | L_V | L_E | L_B |
|--:|--:|--:|--:|--:|
| 1 | 0.8197 | 0.4546 | 0.2816 | **0.0835** |
| 10 | 0.7535 | 0.4258 | 0.2445 | 0.0832 |
| 20 | 0.6895 | 0.3979 | 0.2087 | 0.0829 |
| 30 | 0.6348 | 0.3736 | 0.1786 | 0.0827 |
| 40 | 0.5867 | 0.3412 | 0.1632 | **0.0823** |

L_V falls 25 % and L_E 42 %, but **L_B falls only 1.4 %** (0.0835 → 0.0823)
and is essentially flat after epoch 5. The objective's structure term is
`≈ 0.083 ≈ 1.68/n`, i.e. exactly the value of *fully unexplained* incidence —
the model never moves structures. **Answer: attributes only; L_B does not
descend.**

## 3. Q2 — are the transports meaningful or diffuse?

Transport is *not* uniform (mean max entry `P_V` = 0.125 vs uniform 1/24 =
0.042, ≈ 3×; entropy 2.59 vs log 24 = 3.18), but the non-uniformity is **not
topology-driven**. Rewiring does not move it, and the same transport statistics
appear on unrelated size-matched graphs. **Answer: mildly non-uniform but
structurally meaningless.**

## 4. Q3 — is the code sensitive to degree-preserving rewiring?

Holdout = 256 train graphs never used for the dictionary update. `Δ_α = ||α_G − α_{G'}||_1`:

| perturbation | mean Δ_α | p90 |
|--|--:|--:|
| permutation (IDs only) | 1.9e-08 | 0.0 |
| **degree-preserving 2-switch rewiring** (success 100 %) | **5.8e-08** | 0.0 |
| size-matched unrelated graph | 5.45e-01 | 2.0 |

`Δ_rewire` is **indistinguishable from `Δ_perm`**, so the whole-graph code is
provably blind to which atom connects to which bond. Inferred energies agree:
original `total` 0.58584, rewired 0.58585, random 0.58910; `L_B` 0.08287 /
0.08287 / 0.08210 — the unrelated random graph is fitted *slightly better* than
the original. **Answer: no; rewiring does not change `α_G` or structure
reconstruction.**

## 5. Q4 — does structure-off (`λ_B = 0`) change the codes?

Inference-only `λ_B = 0`: `Δ_α = 1.0e-04` (p90 = 0), energy drops from 0.58584
to 0.50304 (−L_B). Removing the whole structural term changes the code by
~1e-4, i.e. essentially nothing. **Answer: no — the code is decided entirely by
the attribute terms.**

## 6. Q5 — atoms / codes diverse or collapsed?

* Atom *parameters* are diverse: min pairwise atom distance 1.49, near-duplicate
  fraction 0.0 — atoms do not collapse in parameter space.
* **Usage collapses.** Dictionary-update code `α` (2048 graphs): entropy 0.0091,
  effective support 1.01, mean max coef 0.996, **97.8 % of codes one-hot
  (max coef > 0.99)**; argmax atom 7 = 1705 (83.3 %), atom 4 = 275 (13.4 %),
  atom 5 = 68 (3.3 %); **13 of 16 atoms are never selected**; `mean_alpha` =
  [0,…,0, 0.137 (atom 4), 0.031 (atom 5), 0, 0.831 (atom 7), 0,…].
* Latent structure templates are near-uniform/informative-free: `R_k` mean
  0.0833 (= 2/24), mean |dev| from uniform 0.028, max 0.363.
* The gate's `dictionary_no_collapse` fails (`top1_util_frac = 0.836 > 0.8`,
  13 dead atoms). Cross-graph `code_pairwise_mean` stays high (0.58) only
  because a minority of graphs pick the runner-up atoms; pairwise L1 *median*
  is 0 (most graph pairs have identical codes).

**Answer: effective collapse — parameter-diverse atoms, but a near one-hot
code concentrated on ~3 atoms.**

## 7. Q6 — does the frozen code carry task signal?

Frozen-code probe (only use of `y`), head input = `α_G ∈ Δ_16` only:

| probe | train MAE | valid MAE | params |
|--|--:|--:|--:|
| ridge | 1.309 | 1.343 | 17 |
| linear | 1.288 | 1.317 | 17 |
| MLP 16→32→1 | 1.288 | **1.318** | 577 |

Frozen references: DOI 0.3865, scattering 0.3657, I-CRATE 0.399. A
predict-the-mean baseline is ≈1.6 MAE. The TIGD code therefore carries only a
faint task signal, far below every prior fixed lift. **Answer: no meaningful
advantage.**

## 8. Solver stability (a second, independent failure)

The pre-registered Sinkhorn budget (`ε=0.05`, 50 iterations) is **not
converged** for a learned, sharpened dictionary: the transport marginal error
grows monotonically with training,

| epoch | 1 | 10 | 20 | 30 | 40 |
|--|--:|--:|--:|--:|--:|
| marginal err | 1.0e-04 | 1.6e-03 | 4.9e-03 | 8.9e-03 | **1.24e-02** |

i.e. at epoch 40 the returned `P_V` violates its row marginals (`1/n ≈ 0.043`)
by ~29 % relative. Because the dictionary sharpens, the entropic cost landscape
becomes stiffer and 50 log-domain iterations no longer suffice. So the
"transport update" is not even a valid transport at the end of training, which
further undermines `L_B` and the structure signal. Backtracking failure rate
stayed ≈ 0, and no NaN occurred; the failure is silent mal-convergence, not a
crash.

## 9. Pre-registered pilot gate

```json
{"dictionary_no_collapse": false, "code_diversity": true,
 "permutation_invariance": true, "rewiring_changes_code": false,
 "random_ge_rewiring": true, "structure_learned": true,
 "structure_off_changes_code": false, "transport_nonuniform": true,
 "recon_separation": false, "solver_stable": false, "all_pass": false}
```

Fails: physical dictionary collapse (usage), rewiring sensitivity,
structure-off sensitivity, original/random reconstruction separation, solver
stability. Positive controls that *do* pass: exact permutation invariance
(1.9e-08), relative ordering `Δ_perm ≪ Δ_rewire < Δ_random`, dictionary
parameter diversity, and partial (but topology-blind) transport concentration.
Per the pre-registration the full 10 000-graph stage is skipped and this failure
note is written.

## 10. Mechanism picture

The three terms of the objective are the same order of magnitude
(L_V≈0.34, L_E≈0.16, L_B≈0.08 at the end), so this is not simple
under-weighting of `L_B`. The failure is structural:

1. **Free transport decouples topology.** `P_V, P_E` are unconstrained
   graph-specific bi-stochastic-to-uniform couplings. Any graph's incidence can
   be reshuffled onto the atom's slot templates at no attribute cost, so the
   coupling never has to respect the real endpoint assignment. Rewiring is
   therefore absorbed exactly.
2. **The mixture code has no pressure to stay mixed.** `E_G(α)` with fixed
   transports is a convex quadratic whose interactions push interpolated
   attribute prototypes *away* from the one-hot inputs, so projected gradient +
   free transport drives `α` to a vertex within a few epochs; with `α`
   one-hot only ~3 atoms ever receive gradient, so the other 13 stay at their
   (near-uniform) initialisation.
3. **The structural symmetry is never broken.** With `α` one-hot and transport
   near-uniform, the gradient of `L_B` w.r.t. `R̄` is structureless
   (`Q_U, Q_W` ≈ uniform), so `R_k` stays ≈ 2/24 and `L_B` stays at its
   unexplained value.
4. **The solver is under-converged** exactly when the atoms sharpen (§8).

These are properties of the *frozen object / solver*, not of the data or the
attribute channel (which train cleanly).

## 11. Honest caveats

* Single seed (0), single instantiation; the pilot used 2048/10000 train graphs.
* The collapse/solver findings could in principle be mitigated by objective or
  solver changes this round was explicitly forbidden to make (no sparsity, no
  ε/iteration tuning, no weight tuning, no entropy regulariser on `α`,
  no transport anchoring).
* The 0.012 marginal error means part of the reported `L_B`/transport behaviour
  is contaminated by a non-converged coupling; a converged solver would be
  needed to cleanly separate "object fails" from "solver fails".
* No official-test read; nothing here is terminal evidence.

---

## Final answer to the six questions

| # | question | answer |
|--|--|--|
| Q1 | does `L_B` descend? | **No** (L_V −25 %, L_E −42 %, L_B −1.4 %). |
| Q2 | non-uniform transport? | Mildly (3× uniform), but **not topology-driven**. |
| Q3 | rewiring changes `α_G`? | **No** (5.8e-08 ≈ permutation). |
| Q4 | structure-off changes codes? | **No** (1.0e-04). |
| Q5 | atoms/codes diverse? | Parameter-diverse but **usage-collapsed** (13/16 dead, 98 % one-hot, 83 % atom 7). |
| Q6 | task signal? | **No** (valid MAE 1.32 ≫ 0.37–0.40 frozen lifts). |

## Final conclusion

> **Soft transport absorbs too much graph identity; this graph-space dictionary
> formulation is not viable.**

## Next decision

> `revise the graph-space reconstruction / transport objective`

A revision of the object (not a task-driven round yet) is the justified next
step, because the failures are localised and diagnosable: (a) anchor /
regularise the transport so it cannot reshuffle endpoints for free;
(b) break the transport↔structure symmetry (e.g. a stronger or coupled
structural term, or a transport prior tied to the real incidence); (c) prevent
the degenerate one-hot mixture without the forbidden ad-hoc sparsity penalty
(e.g. an entropic code regulariser); and (d) make the entropic solver converge
(more iterations and/or a larger `ε`, reported as a solver change, not a tuned
hyper-parameter). This is a **research bet**: if a revised object still shows
`Δ_rewire ≈ Δ_perm` and code collapse, the decision becomes
`stop dictionary-as-primary-representation`.
