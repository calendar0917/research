# Pre-registration — GSCN-v0: Task-Driven Sparse Dictionary-Core Graph Network

Round name: **GSCN-v0** (*Graph Sparse Coding Network*, feasibility audit).
Written **before** any Stage 0/1/2 run, following the `remote-research-runner`
skill (local code → remote A100 compute → local analysis). This note modifies no
historical record.

Protocol version: `gscn_v0`. Study: `zinc-context-gap`. Regime: deterministic
A100, canonical PyG ZINC `subset=True` official **train 10 000 / valid 1 000**.
**Official test is never loaded, instantiated or referenced.** Split sizes/hashes
are inherited from the canonical ZINC loader.

Branch / code revision that will run: recorded with every result
(`code_state_hash`, commit).

---

## 0. Mandatory clarification of the canonical "B-Null" / "B-Full"

This round is a **research reset** and deliberately does *not* inherit PSCD
(GraphBPE, motif vocabulary, ports, composition graph, MDL selection, frequency
motifs, whole-graph dictionary) or any handcrafted structural statistics as an
*input*. Before writing any model this section fixes the two canonical objects
by reading the repo (not memory).

### 0.1 What the repo actually defines

Source of truth read for this section:
`tracks/ksvd/notes/local_token_null_preregistration.md`,
`notes/local_token_null.md`, `records/claims/claim-local-token-null-20260919.yaml`,
`records/decisions/decision-local-token-null-test-read-20260918.yaml`,
`experiments/luyin16/zinc_patch_path_pooling.py`,
`experiments/luyin16/zinc_shared_structural_patch_encoder.py`,
`experiments/luyin16/zinc_shared_bag_patch_encoder.py`.

All canonical "B-*" references instantiate an **identical downstream patch–path
backbone (49,343 params)** and differ **only** in the local 16-D patch-token
generator `e_patch`:

| Reference | `patch_representation` | Local 16-D generator | Generator params | Total params |
|---|---|---|---|---|
| A0 | `typed` | `typed_embedding` (learned exact-certificate lookup) | 36,420 | 85,763 |
| B-Bag | `shared_bag` | `SharedBagPatchEncoder` (connectivity-free bag) | 35,168 | 84,511 |
| **B-Full** | `shared_structural` | `SharedStructuralPatchEncoder` (radius-2 edge-aware MP) | 35,152 | **84,495** |
| **B-Null** | `null` | *none* (`e_patch ≡ 0`) | 0 | **49,343** |
| Constant | `constant` | one trainable graph-wide `c ∈ R^16` | 16 | 49,359 |

Canonical seed-0 fixed Top-5 soup valid MAE (deterministic A100): A0 0.124704,
B-Bag 0.127382, **B-Full 0.119818**, **B-Null 0.123028**.

### 0.2 What "B-Null deleted" and what "B-Full adds"

* **B-Null deletes** exactly the molecule-dependent **local 16-D patch-token
  generator** (`typed_embedding` / bag / structural encoder). It does **not**
  delete handcrafted structural statistics. B-Null still consumes
  `patch_cont` (146-D shell-conditioned chemistry + size/cycle/degree scalars),
  `pair_relation` (23-D distance / overlap / boundary / path-bond / shortest-path
  count), `global_context` (62-D), `topology_features` (hinge), and the radius-1
  typed `parent_token` — all **handcrafted structural / attribute statistics**.
* **B-Full adds** back only a learned radius-2 edge-aware encoder producing that
  16-D token. It is therefore **not** a "handcrafted-statistic" model either; it
  is B-Null + one learned local structural encoder. Empirically the added token
  is near rank-1 (effective rank 1.107/1.202) and worth only ~0.0032 valid MAE.

$$
\boxed{\text{B-Full} = \text{B-Null} + \text{learned radius-2 16-D patch encoder}}
$$

### 0.3 Discrepancy with this round's premise → restated discipline

This round's premise was that **B-Null = a raw-graph learner** and
**B-Full = B-Null + handcrafted structural statistics**. **That is false in this
repo.** Both canonical objects are rooted-patch models whose inputs are
handcrafted statistics; they differ only by a learned local token, and the
canonical handcrafted gap `B-Null − B-Full = 0.0032 < 0.02`.

Per the round's Section 0 rule (restate the actual definition, then keep the
binding discipline) the following is frozen:

1. **GSCN input = raw atom category + raw bond category + adjacency only.**
   No `patch_cont`, no `pair_relation`, no `global_context`, no
   `topology_features`, no `parent_token`, no shell/distance/overlap/path-count
   fields, no rings/cycles/degrees/motifs (GSCN asserts this programmatically).
2. The **raw-graph primary baseline for GSCN is a raw-input graph model** under
   the canonical training protocol — i.e. the repo's own raw atom-graph reader
   primitive (`experiments/luyin16/zinc_patch_path_pooling.py` has no raw reader;
   the raw primitive is `code/run_pscd_compositional_dictionary_audit.py::
   make_module("raw", …)` / its collate, which uses *exactly* raw atom/bond/
   adjacency). We lift it to the canonical capacity `d=64, L=4` and call it
   **`B-Null-Raw`**; it is the operational `M_Null` of this round.
3. The canonical patch **B-Null** (49,343; 0.123028) and **B-Full** (84,495;
   0.119818) are reported as **handcrafted-statistics references**, with
   `M_Full := 0.119818` (canonical B-Full seed-0 soup).

$$
\boxed{\text{GSCN input feature set} \;=\; \text{raw atom/bond/adjacency set}
\;\subsetneq\; \text{handcrafted-statistic set of canonical B-Null/B-Full}}
$$

---

## 1. Reference chain

$$
\boxed{\text{B-Null-Raw (raw baseline)} \rightarrow \text{GSCN} \rightarrow
\text{B-Full (handcrafted reference)}}
$$

* `B-Null-Raw`: what a raw graph learner reaches with **no** handcrafted
  statistics.
* `GSCN`: does a task-driven **sparse dictionary code** act as the hidden-state
  transition and recover useful structure?
* `B-Full` (canonical, 0.119818): where handcrafted statistics + a learned
  local encoder land. **Reported as reference only; never a GSCN backbone.**

---

## 2. Core question

> Can a graph model learn task-relevant structural representations directly
> from **raw topology** by repeatedly **coding relational states over shared
> sparse dictionaries**, rather than relying on handcrafted structural
> statistics?

This is not "does ISTA improve ZINC". Compression is **not** a contribution of
this round and will not be claimed.

---

## 3. Architecture (autonomous; no side channel, no bypass)

For `L=4` layers and hidden width `d=64` (fixed; no depth/width sweep):

$$
H^{(\ell)} \xrightarrow{\text{relational mixing}} Y^{(\ell)}
\xrightarrow{\text{sparse dictionary coding}} H^{(\ell+1)}.
$$

`H^(0) = E_A(atom_types)`. The **sparse coding is the hidden-state transition**;
there is **no** `H_BNull + z_dict` side channel and **no** raw residual bypass.

### 3.1 Step A — raw relational mixing

The repo's most basic edge-aware primitive (linear, permutation-equivariant, no
nonlinear MLP), reused verbatim in functional form:

$$
Y_v^{(\ell)} = W_{\rm self}^{(\ell)} h_v^{(\ell)}
+ \sum_{u\in\mathcal N(v)} \big(W_{\rm msg}^{(\ell)} h_u^{(\ell)}
+ W_{\rm edge}^{(\ell)} e_{uv}\big), \qquad
Y^{(\ell)} \leftarrow \operatorname{LayerNorm}(Y^{(\ell)}).
$$

`e_uv` is the raw bond-category embedding; edges are directed both ways.

### 3.2 Step B — per-layer sparse dictionary code (the core)

`D^(ℓ) ∈ R^{K×d}`, `K = 2d = 128` (fixed, overcomplete; no K sweep), not shared
across layers, shared across all molecules/nodes. Normalized atoms
`d̄_k = d_k/(‖d_k‖₂+ε)`, and **all** coding uses `D̄` (no free atom scale).

$$
\alpha_v^* = \arg\min_{\alpha\ge0}\ \tfrac12\|y_v-\alpha\bar D\|_2^2 + \lambda^{(\ell)}\|\alpha\|_1 .
$$

Unrolled ISTA: `T=3` (fixed), `A^(0)=0`,

$$
A^{(t+1)} = \operatorname{ReLU}\!\big[A^{(t)} + \eta\,(Y\bar D^\top - A^{(t)}\bar D\bar D^\top) - \eta\lambda^{(\ell)}\big],
\qquad
\eta = \frac{0.9}{\sigma_{\max}(\bar D)^2+\epsilon}.
$$

`σ_max( D̄ )` is computed by deterministic power iteration and **detached**;
reconstruction / coding back-propagate fully through `D`. No learned LISTA
encoder. Then

$$
\boxed{H^{(\ell+1)} = \operatorname{LayerNorm}\big(A^{(\ell)}\bar D^{(\ell)}\big)}.
$$

LayerNorm is used only for numerical scale stabilization. No extra hidden MLP
(`Linear→SiLU→Linear`) inside any GSCN block. The prediction head and pooling
are inherited from the raw baseline (sum-pool → `Linear(d,d)→SiLU→Linear(d,1)`).

### 3.3 `λ` — one-shot label-free initialisation calibration (no sweep)

After seed-0 initialisation, using a **fixed** batch of 512 canonical **train**
graphs (deterministic order), forward to each layer's initial `Y^(ℓ)`, form
`C = Y D̄ᵀ`, and set

$$
\lambda^{(\ell)} = \operatorname{pctile}_{80}\big(\{C_{ij} : C_{ij} > 0\}\big),
$$

then **freeze** `λ^(ℓ)` for the whole experiment. Calibration never looks at
labels, valid or test. The values are written to the run artifact. (Intuition:
~20 % of atoms clear the threshold on the first ISTA step.)

---

## 4. Primary arms

| Arm | id | hidden-state update | notes |
|---|---|---|---|
| A | `bnull_raw` | `SiLU(W_self h + agg)` (raw baseline) | `M_Null` |
| B | `generic` | mixing + `Linear(d,m)→SiLU→Linear(m,d)` | `m` chosen once by parameter algebra (≤10 % of GSCN) |
| C | `nothresh` | full GSCN block with `λ=0` | same `D,K,T`; projected ISTA |
| D | `sparse` | full GSCN block with calibrated `λ` | `M_Sparse` |

Shared across **all** arms: raw atom/bond embeddings, directed edge batching,
sum pooling, prediction head, optimizer family/schedule, target handling,
checkpoint protocol, Top-5 soup, seed.

`M_Full` = canonical B-Full seed-0 soup (0.119818), reference only.

### 4.1 Generic control width

Per-layer trainable parameters (with 2 LayerNorms in both):

* GSCN: `mixing(3d²+3d) + D(2d²) + 2·LN(4d) = 5d² + 7d`
* Generic: `mixing(3d²+3d) + MLP(2dm+m+d) + 2·LN(4d) = 3d²+7d+m(2d+1)`

Setting them equal gives `m = 2d²/(2d+1)`. For `d=64`, `m ∈ {63,64}`; the best
integer is chosen once from the algebra (no valid-based choice) and the achieved
`|P_B−P_GSCN|/P_GSCN` is reported (target ≤ 10 %).

**Computed (frozen before run, `d=64, L=4, K=128`)**: `m=64`;
`P_Null=55,425`, `P_Generic=89,729`, `P_NoThresh=P_Sparse=89,217`,
so `|P_Generic−P_Sparse|/P_Sparse = 0.57 %` and `P_Sparse/P_Null = 1.61×`
(reported explicitly per §32).

---

## 5. Stage 0 — data-free implementation tests (must pass before any training)

* **A permutation equivariance**: relabel a molecule's atoms (and edge order);
  every layer `H(PG)=P·H(G)` and the graph prediction is invariant; max error
  `< 1e-5`.
* **B batching invariance**: single-graph vs batched forward prediction max
  error `< 1e-5`.
* **C dictionary gradient**: on a random train batch, `‖∇_{D^(ℓ)} L‖₂ > 0` for
  every layer.
* **D reconstruction dimensions**: `Y: N×d`, `A: N×K`, `A D̄: N×d`.
* **E code non-negativity**: `A ≥ 0`.
* **F no hidden bypass**: with `Y` fixed, perturbing `D` changes the block
  output materially (the block actually depends on the dictionary).

## 6. Stage 1 — train-only smoke + mechanism gate (no valid-based tuning)

Short train-only run of `sparse`. It must show: loss decreasing, no NaN/Inf,
non-zero `D` gradients at all layers, atoms not instantly all collapsed, code
neither 100 % active nor all zero. Mechanism gate:

* global active fraction `0.02 < ‖A‖₀/(N·K) < 0.50`;
* `≥ 50 %` of dictionary atoms activated at least once;
* no single atom `> 50 %` of all non-zero activations;
* dictionary effective rank not near 1.

If the gate fails → **stop**: "sparse coding mechanism collapsed before
predictive evaluation" (do **not** repair by sweeping `λ`).

## 7. Stage 2 — formal training

If Stage 1 passes: arms A–D, **seed 0 only**, canonical budget
(Adam lr 1e-3, wd 1e-5, batch 128, max 240 epochs, patience 40, L1, clip 5,
unscheduled, best-official-valid checkpoint, fixed equal-weight Top-5 soup;
`torch.use_deterministic_algorithms(True)`). Official test never loaded.
No GSCN-specific optimisation: no extra epochs/optimiser/LR/T/K/λ/residual/MLP.

## 8. Primary quantities and gates

`M_Null, M_Generic, M_NoThresh, M_Sparse, M_Full` (best-valid **and** Top-5
soup), `Δ_Null = M_Null−M_Sparse`, `Δ_Capacity = M_Generic−M_Sparse`,
`Δ_Sparsity = M_NoThresh−M_Sparse`.

* **strong-positive**: `M_Sparse ≤ M_Null−0.02` **and**
  `M_Sparse ≤ M_Generic−0.01` **and** `M_Sparse ≤ M_NoThresh−0.01`, with
  code genuinely sparse, dictionary non-collapsed, task gradients reaching `D`.
* **handcrafted-gap recovery**: if `M_Null−M_Full ≥ 0.02`,
  `ρ = (M_Null−M_Sparse)/(M_Null−M_Full)`; report `ρ ≥ 0.3` as substantial.
  Otherwise report absolute MAE deltas only.
* **capacity-only failure**: `M_Generic ≤ M_Sparse` or `|Δ_Capacity| < 0.01`.
* **sparsity-not-needed**: `M_NoThresh ≤ M_Sparse`, stable.
* **ambiguous positive** (`0.005 < M_Null−M_Sparse < 0.02`, or sparse vs control
  `< 0.01`): run seed 1 paired confirmation (no model change).

## 9. Mechanism / coupling / geometry audits

Per layer, train and valid: active fraction `‖A‖₀/(NK)`, median/p90 active
atoms per node, atom usage counts, dead atoms; dictionary effective rank,
coherence `max_{i≠j}|d_iᵀd_j|`, average pairwise cosine, post-normalisation
norms, `‖D_final−D_init‖`; `‖∇_{D^(ℓ)} L_task‖` during training (a genuine
`L_task → D` path). Primary Sparse-GSCN uses **no** reconstruction auxiliary
loss: `L = L_task` only.

## 10. Inference ablations (Sparse-GSCN, no retraining)

A atom-row permutation (sanity, prediction invariant); B random-normalised
dictionary destruction (valid MAE must degrade); C within-batch code shuffle
(degrade); D layer-wide mean-code replacement (degrade). These only test that the
model truly uses learned atom/code structure.

## 11. Post-hoc structural semantics (explanation only, never fed back)

For each layer/atom, top-50/100 activating nodes → rooted radius-1/2
neighbourhood canonical key (permutation-invariant WL hash); report top
concentration (most-common key fraction, entropy) vs random and
activation-matched-random baselines. No chemistry functional groups are defined
in advance. This is not a primary success gate; if nearly all atoms show no
structural concentration, the conclusion may only claim a *task-learned latent
sparse dictionary*, not an interpretable structural dictionary.

## 12. Reporting discipline

Per-arm parameter breakdown (embedding / mixing / dictionary / head / total);
if `P_Sparse > 1.5·P_Null` this is reported explicitly (K is **not** re-tuned).
Computational honesty only (graphs/s, peak GPU MB, wall/epoch, ISTA overhead);
no compression claim. Official test never opened.

## 13. Forbidden this round

PSCD; GraphBPE; motifs as input; ports; MDL; handcrafted structural statistics
as input; manual chemistry; K/λ/T/hidden-width/depth sweeps; attention; raw
hidden residual bypass; reconstruction auxiliary loss; official test;
running changes driven by valid.

## 14. Frozen verdict / next-decision vocabularies

Verdict first line ∈ { "Task-driven sparse dictionary coding is viable as the
core graph representation update." ; "The gain is explained by generic model
capacity rather than sparse dictionary learning." ; "Dictionary factorization
helps, but explicit sparsity is not supported." ; "The intended sparse
dictionary mechanism collapses despite predictive gains." ; "The
optimization-derived sparse graph update is not trainable under the canonical
regime." ; "Inconclusive due to baseline or implementation integrity failure." }.

Next decision ∈ { `formalize GSCN as the new primary method` ;
`investigate one optimization-level revision of the sparse coding block` ;
`drop sparsity and reconsider whether dictionary factorization alone is
scientifically sufficient` ; `reject dictionary-core architecture and reassess
the research premise` ; `fix baseline/implementation integrity before
interpretation` }.

---

## 15. Deliverables

Preregistration (this file); Stage 0 test log; Stage 1 smoke log + gate;
parameter accounting; Stage 2 histories + Top-5 soups for arms A–D; mechanism,
geometry and task-coupling audits; inference ablations; post-hoc semantics;
reference table incl. canonical B-Null/B-Full; analysis note; claim + decision;
`STATE.yaml` pointer. All under `results/gscn_v0/` with provenance
(commit, device, seed, protocol).
