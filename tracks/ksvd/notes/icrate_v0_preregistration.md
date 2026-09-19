# Pre-registration — I-CRATE-v0: Incidence-Structured White-Box Dictionary Transformer

Round name: **I-CRATE-v0**.  Written **before** any formal ZINC run and before
the implementation is deployed.  Local code → remote A100 compute → local
analysis.

Official ZINC `test` is **never** loaded, instantiated or referenced by any
stage.  Architecture / hyperparameter / stopping decisions use official
`train` + `valid` only.  This note freezes the architecture, the objective and
the decision gates.  It does **not** modify any historical record.

This is a **new mathematical architecture**, not a repair of WG-ICSC-v0.  Per
`decision-1379b7f2`, WG-ICSC-v0 failed because (a) the 192-D row-L2 graph code
was nearly assignment-invariant and (b) the outer objective could only shape
the dictionaries through a hard, un-converged unrolled solver.  I-CRATE
replaces both: the graph code is now a **global sparse code α_G** over a learned
dictionary after an incidence-constrained token transformer, and the sparse
solver is a **fixed short ISTA** inside the forward pass whose parameters
(`D_a, D_s, D_G, U`) are shaped by the task loss alone.

---

## 0. The single question this round asks

> Can a molecular graph be represented end-to-end by persistent atom and bond
> tokens on the true incidence graph, repeatedly applying incidence-constrained
> CRATE-style token compression and overcomplete sparse dictionary coding, then
> sparsely coding a whole-graph token over a learned global dictionary — and do
> the **incidence** and **sparse-dictionary** mechanisms actually do the work?

Formal **architecture feasibility experiment**, not a leaderboard attempt.  No
GNN encoder, no hand-crafted descriptors, no canonical node IDs, no positional
encoding, no message passing beyond the incidence mask.

## 1. Scientific hypotheses (frozen before the run)

* **H1 — Incidence.** Persistent atom/bond tokens with incidence-constrained
  MSSA allow exact bond↔endpoint assignments to materially affect the learned
  whole-graph representation.
* **H2 — Sparse dictionary.** Overcomplete nonnegative sparse coding provides a
  useful representation transformation beyond a matched-parameter ordinary FFN.
* **H3 — Whole-graph code.** A task-trained sparse code `α_G` of a graph-level
  representation can serve as the final molecule representation without
  canonical node coordinates or handcrafted structural statistics.
* **H4 — Capacity.** Layer-specific ODL with residual synthesis avoids the
  severe underfitting observed in WG-ICSC's shared fixed-objective solver.

No "must outperform B-Full" claim is pre-registered.

## 2. Confirmed graph semantics (from the current repo loader)

Read from `torch_geometric.datasets.ZINC(subset=True, split=...)`:

| quantity | value |
|---|---|
| `x` (atom type) | integer categories `{0,…,20}` → **d_V = 21** |
| `edge_attr` (bond type) | integer categories `{1,2,3}` → **d_E = 3** |
| `edge_index` | each undirected bond stored as **two** directed entries |
| self-loops / multi-edges | none |
| directed-copy consistency | both copies have equal `edge_attr` (checked; mismatch aborts) |
| graph size (train+val) | min 9, median 23, p95 31, max 37 |

One **bond token == one undirected chemical bond**.  The two directed PyG
entries are deduplicated; if their `edge_attr` disagree the run stops and
reports (never guesses).  Bond type is re-indexed `{1,2,3} → {0,1,2}`.  For a
molecule `n = |V|`, `m = |E|`, token count `N = n + m`, internal order
`[atoms ; bonds]` (the model is permutation equivariant/invariant and never
relies on this order).

## 3. Frozen architecture

* hidden dim **d = 48**; trainable categorical embeddings `E_V (21×48)`,
  `E_E (3×48)`; `Z⁰ ∈ R^{N×48}`.  No pre-GNN / patch / descriptor / PE.
* **Incidence mask** `M_G ∈ {0,−∞}^{N×N}`, `M_G[v,e] = M_G[e,v] = 0` iff
  `v ∈ e`; all other pairs (including self) `−∞`.  A token whose attention row
  has no legal key receives a **zero** MSSA update (no NaN).
* **Structural depth L = 4.**  Each layer owns its own `U^ℓ, D_a^ℓ, D_s^ℓ`
  (no cross-layer sharing).
* **Incidence-MSSA** (H = 4 heads, p = 12): columns of `U_h^ℓ ∈ R^{48×12}` are
  L2-normalised each forward; `Ȳ_h = LN_mssa(Z^ℓ) Ū_h`;
  `S_h = Y_h Y_hᵀ/√12 + M_G`; `A_h = softmax(S_h)`;
  `Δ_h = A_h Y_h Ū_hᵀ`; `Δ_MSSA = ¼ Σ_h Δ_h`;
  `Z^{ℓ+½} = Z^ℓ + Δ_MSSA`.  No Q/K/V, no FFN, no learnable bias.
* **Token ODL** (overcomplete C = 2, M = 96): `D_a^ℓ ∈ R^{48×96}`,
  columns L2-normalised.  `X = LN_odl(Z^{ℓ+½})ᵀ`;
  `A^ℓ ≈ argmin_{A≥0} ½‖X − D̄_a A‖_F² + λ_tok‖A‖₁` via **R_tok = 2** ISTA
  steps from `A₀ = 0`, step `η = 0.9/(‖D̄_a‖₂²+ε)` (spectral norm detached),
  `λ_tok = 0.10` fixed (not tuned on valid).
* **Synthesis residual** with an **independent** `D_s^ℓ ∈ R^{48×96}`
  (columns L2-normalised): `Z^{ℓ+1} = Z^{ℓ+½} + (D̄_s^ℓ A^ℓ)ᵀ`.  `D_s ≠ D_a`.
* **Graph seed** `g⁰ = N^{-1/2} Σ_i Z^L_i` (size-sensitive; no concat of n, m,
  degrees, raw features).
* **Cross-MSSA** (one-way, `U^G_h ∈ R^{48×12}`): `ḡ = LN_G(g⁰)`,
  `Z̄ = LN_tokens(Z^L)`, `q_h = ḡᵀ Ū^G_h`, `K_h = V_h = Z̄ Ū^G_h`,
  `a_h = softmax(q_h K_hᵀ/√12)`, `δ_h = a_h V_h Ū^{Gᵀ}_h`,
  `g^{½} = g⁰ + ¼ Σ_h δ_h`.  The graph token does not feed back into tokens.
* **No LayerNorm before the global dictionary** (keeps absolute magnitude).
* **Global dictionary** `D_G ∈ R^{48×96}` (columns L2-normalised);
  `α_G ≈ argmin_{α≥0} ½‖g^{½} − D_G α‖₂² + λ_G‖α‖₁` via **R_G = 4** ISTA
  steps, `η_G = 0.9/(‖D_G‖₂²+ε)` (detached), `λ_G = 0.10` fixed.  **α_G** is
  the final molecule representation (no concat of `g^{½}` or raw features).
* **Head**: `Linear(96,64) → SiLU → Linear(64,1)` on `α_G` only.
* **Outer loss**: `L_train = MAE(ŷ, y)`.  No reconstruction / sparsity /
  auxiliary losses (those live inside the forward operators).

Expected parameter count ≈ **61K** (tokenizer ≈ 1.2K, 4 structural layers
≈ 11.7K each, global MSSA ≈ 2.4K, global dictionary 4.6K, head ≈ 6.3K).  If the
actual count is substantially > 80K, check for an accidental extra QKV/FFN.

## 4. Why this should not repeat WG-ICSC's failure

* The graph code is no longer a permutation/assignment-invariant row-L2 norm;
  `α_G` is a learned sparse code that can depend on the full incidence-shaped
  token field.
* Dictionary parameters are shaped through a **fixed 2/4-step ISTA** whose
  Jacobian flows directly to `D_a, D_s, D_G` under the task loss; there is no
  requirement that the 4 layers converge to one shared objective.
* Residual synthesis (`Z^{ℓ+1} = Z^{ℓ+½} + D_s A`) keeps representation
  capacity instead of forcing a hard sparse bottleneck.

## 5. Training protocol (canonical, reused)

`OPTIMIZED_PROTOCOL`: Adam, lr 1e-3, weight decay 1e-5, batch 128, MAE,
grad-clip 5.0, no scheduler, max 240 epochs, patience 40, best official-valid
checkpoint, fixed equal-weight **Top-5 soup** (no K / weight search).  Data:
official PyG ZINC `subset=True` train (10 000) + valid (1 000).  Seed 0 only in
the first formal run.  A100 execution regime.  No target normalisation (matches
the current ZINC A100 protocol).

Matched A100 Top-5 soup references (valid-only orientation, protocol
`zinc-context-gap`): `B-Full` seed0 **0.119818** (84 495 params), `A0` seed0
**0.124704**.  No historical CPU micro-delta is used as a claim.

## 6. Implementation order (non-negotiable)

1. single-graph correctness-first reference forward;
2. padded/batched forward;
3. targeted tests A–J (below), incl. batch↔reference parity;
4. A100 smoke (forward/backward/step) + **64-train-graph overfit smoke**;
5. formal seed-0, valid-only.

## 7. Targeted tests (A–J)

* **A** undirected bond deduplication (one token per bond; equal `edge_attr`;
  correct endpoints).
* **B** incidence mask correctness on `atom0–bond0–atom1–bond1–atom2`.
* **C** permutation equivariance of every structural layer (`<1e-5`).
* **D** final invariance of `α_G` and `ŷ` under permutation.
* **E** incidence sensitivity (equal feature multisets, different endpoints ⇒
  different `Z^ℓ` and `α_G`).
* **F** no-self structural dependency.
* **G** ODL objective non-increasing (token 2-step and global 4-step).
* **H** exact zeros from the ReLU-threshold step.
* **I** batch↔reference equivalence for `Z^L`, `g`, `α`, prediction.
* **J** gradient reaches every parameter block.

## 8. Recorded mechanism metrics (every epoch / final)

* per-layer token ODL: nonzero fraction, active atoms per token, dictionary
  utilisation, dead atoms, coefficient concentration (top-1/top-5 mass,
  entropy);
* global dictionary: `α_G` nonzero fraction, active atoms, dead atoms,
  activation frequency, amplitude distribution;
* structural compression vitality `r^ℓ_mssa = ‖Δ_MSSA^ℓ‖_F/(‖Z^ℓ‖_F+ε)` and
  attention entropy / max weight per token type;
* dictionary audit per layer (`D_a, D_s`) and global `D_G`: effective rank,
  pairwise column coherence, near-collinear fraction, singular spectrum;
* gradient vitality per parameter block.

## 9. Mechanism interventions (evaluation-only)

* **Incidence assignment intervention (within-graph).**  Hold atom tokens, bond
  tokens, `n, m` and the bond feature multiset fixed; randomly reassign each
  bond token's endpoint pair.  On ≥ 500 valid graphs report
  `Δ_α = ‖α^orig − α^rewired‖₂/(‖α^orig‖₂+ε)`, `Δ_y = |ŷ^orig − ŷ^rewired|`,
  and layer-wise `‖Z^ℓ_orig − Z^ℓ_rewired‖_F/(‖Z^ℓ_orig‖_F+ε)` (token ordering
  preserved, so no matching needed).
* If `Δ_α` stays at the 1–2 % level, the incidence information still fails to
  survive into the graph representation.

## 10. Decision gates (seed 0, valid only)

* **Case A — clear failure**: valid Top-5 soup **> 0.20**, or train≈valid with
  a high train error (strong underfit).  Stop; do not sweep d / L / heads /
  overcomplete ratio / λ / readout.
* **Case B — borderline but mechanistically viable**: `0.13 < soup ≤ 0.20`
  with healthy incidence vitality, genuinely sparse ODL and no severe
  dictionary collapse.  Run only the two seed-0 controls (§11).
* **Case C — competitive**: `soup ≤ 0.13` with healthy mechanism.  Run main
  seeds 1/2 and the two seed-0 controls.  Still no test.
* **Case D — predictive but mechanism inactive**: good MAE but MSSA ≈ 0 /
  ODL dense-or-zero / α collapse / rewiring barely changes α_G ⇒ classify
  "predictive, intended mechanism inactive"; do not claim the mechanism works.

## 11. The only permitted controls (if seed-0 main passes Case B)

* **Control A — No-incidence**: identical parameters/count but
  `Δ_MSSA = 0` (skip Incidence-MSSA; ODL unchanged).  The mask is **not**
  replaced by global attention.
* **Control B — FFN replacement**: replace each structural `D_a + D_s` with
  matched-width `LN → Linear(48,96) → SiLU → Linear(96,48)` + residual, and
  `D_G + ISTA` with `Linear(48,96) → ReLU` (96-D graph code, same head).
  No third control.

## 12. Things this round must NOT claim

Even if results are good: no strict CRATE sparse-rate-reduction claim, no
WL-transcendence, no "dictionary atoms = chemical functional groups", no
"graph atoms are discrete subgraphs", no "attention proves a causal
mechanism".  Accurate description:

> CRATE-inspired / optimization-structured white-box architecture with
> incidence-constrained token compression and explicit sparse dictionary
> coefficients.

## 13. Deliverables

implementation; single-graph reference; batched implementation; targeted tests;
this pre-registration; formal config; seed-0 metrics; ODL sparsity metrics;
dictionary spectra/coherence; incidence-intervention results; plots; final
analysis note.  Recorded: git commit, exact GPU, seed, exact data splits,
official test untouched + read count unchanged, wall time, parameter count,
peak memory.
