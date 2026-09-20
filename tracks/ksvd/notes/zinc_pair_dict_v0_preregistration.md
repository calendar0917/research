# Pre-registration — PSD-v0: Pair-Structural Representation + Function-Preserving Dictionary

Round name: **PSD-v0** (*Pair-Structural + Dictionary*, feasibility → mechanism audit).
Written **before** any Stage 0 run. Protocol id: `zinc-pair-struct-dict-v0`.
Study: `zinc-context-gap`. Regime: deterministic A100, canonical PyG ZINC
`subset=True` official **train 10 000** and **valid 1 000** exist but valid is
**only** used for the later mechanism screen; **official test is never loaded,
instantiated or referenced**. This note modifies no historical record.

This round is a direct response to `decision-gscn-v0-reject-dictionary-core-20260920`:
GSCN-v0 showed that compressing `neighbors → sum → one node vector → dictionary`
destroys structure *before* the dictionary and that the resulting dictionary-core
is dominated by a plain raw node-MPNN and by a parameter-matched generic control.
The two questions here are therefore asked **in order**, and Q2 is not started
until Q1 passes.

---

## 0. Input discipline (frozen)

Input = **raw atom category + raw bond category + adjacency only.** No
`patch_cont`, `pair_relation`, `global_context`, `topology_features`,
`parent_token`, ring/cycle/degree/motif counts, WL certificates, or any
handcrafted statistic. This is asserted programmatically in the runner.

The primary raw baseline is the **same** object GSCN-v0 used: `B-Null-Raw`
(the raw edge-aware SiLU node-MPNN), `d=64, L=4`, **55 681 params** with the
real category counts (21 atom / 3 bond categories). It is imported verbatim from
`code/run_gscn_v0.py::RawBaseline` so the comparison is bit-comparable with the
GSCN-v0 raw baseline.

---

## 1. Questions

**Q1.** Can a *pair-state* (relation-preserving) structural encoder learn a
task-discriminative whole-graph representation `u_G ∈ R^d` from **raw** input,
at least matching a raw node-MPNN of comparable parameter budget?

Motivation: GSCN compressed node neighbourhoods into a single vector before any
dictionary. A pair state `H_ij` keeps structural relations alive until the graph
readout, so the representation is not pre-collapsed.

**Q2 (only if Q1 passes).** Given a frozen warm-start dense structural model
`u_G → f(u_G)`, does inserting a **function-preserving learned sparse
dictionary** (`z_G = α_T D`, head reads only `z_G`) provide a task benefit over
matched controls (dense-restart, fixed-identity dictionary, parameter-matched
generic Top-k)?

---

## 2. Pair-state structural encoder (frozen, no sweep)

Fixed capacity (no width/depth/K search): `d = 64`, `L = 3` relation layers.

Objects on one molecule with `n` atoms:

* atom embedding `E_A[c] ∈ R^d` (c = raw atom category);
* relation embedding `E_R[r] ∈ R^d`, `r ∈ {none (non-bond off-diagonal),
  bond category, self (i=j)}`;
* pair state `H_ij ∈ R^d`, initialised permutation-equivariantly as
  `H^(0)_ij = W_init [E_A[c_i] ; E_A[c_j] ; E_R[r_ij]]`, masked to valid pairs.

Relation layer (`ℓ = 1..L`):

```
φ, ψ : Linear(d,d)
M_ij = Σ_k φ(H_ik) ⊙ ψ(H_kj)          # contract over the middle index
H'_ij = LayerNorm( H_ij + η([H_ij ; M_ij]) ),   η = Linear(2d,d) + SiLU
```

Masking removes padding nodes in the `k` sum and zeroes invalid pairs.

Graph readout (permutation-invariant) →
`u_G = W_read [ mean_{ij valid} H_ij ; mean_i H_ii ] ∈ R^d`,
then the head `f(u_G) = Linear(d,d) → SiLU → Linear(d,1)`.

Exact parameter count is computed and reported; the target is ~80–90k so a
width-matched raw control exists. No architecture search of any kind is allowed
in this round: one formulation, one capacity.

## 3. Raw controls

* `raw` — canonical `B-Null-Raw`, `d=64, L=4` (55 681 params).
* `raw_wide` — node-MPNN `d=76, L=4` (~77 980 params), parameter-matched to the
  pair encoder. Used as the *primary* comparison.

Same raw input, same optimizer/epoch protocol, same seed, same pooling/head
family, same Top-5 soup.

---

## 4. Stage 0 — data-free correctness (must pass before any training)

Only targeted tests. Runner `stage0` + `tests/test_zinc_pair_dict_v0.py`:

1. permutation equivariance of every layer state `H(P G)=P H(G) Pᵀ`, and
   graph-prediction invariance, max err `< 1e-5`;
2. batching invariance (single vs batched), `< 1e-5`;
3. padding/mask correctness: adding/removing padded nodes does not change
   per-graph predictions (`< 1e-6`), and a padded batch equals the unbatched
   per-graph results;
4. graph-ordering invariance: reordering graphs inside a batch permutes
   predictions only;
5. gradient reaches `E_A, E_R, W_init, φ, ψ, η, W_read, head` at every layer
   (`‖∇‖ > 0`);
6. no node-id / graph-id leakage: predictions invariant to node relabelling and
   graph order (already 1/4), and the model contains no absolute position
   embedding (asserted structurally);
7. official test never constructed (`test_access=blocked`);
8. parameter count and peak-memory complexity consistent with ZINC-scale graphs.

If Stage 0 fails → **fix the implementation, do not train.**

## 4b. Synthetic mechanism sanity (mechanism-only, never a ZINC claim)

A cheap in-process check that the pair update actually builds relational
structure: on `C6` (cycle, diameter 3) vs `2·C3` (two triangles, diameter 2),
both 2-regular and 1-WL-equivalent, the pair-state multiset differs (mean
pairwise sorted-norm distance `> 1e-3`) while the raw node-MPNN node-state
multiset is (near) identical. This is a *mechanism sanity* test only; it can
never be reported as a ZINC performance claim.

---

## 5. Stage 1 — 128-graph overfit sanity (train only)

Fixed deterministic subset of 128 **official-train** graphs. Optimizer
canonical (Adam lr 1e-3, wd 1e-5, batch 64/128, ≤60 epochs, no scheduler, L1,
clip 5). Evaluate **train MAE only**; official valid never used.

Gate (frozen): overfit **GO** iff within budget the best train MAE is
`≤ 0.15` **and** `≤ 0.5 × (MAE at epoch 1)`; the same run on `raw` serves as a
sanity reference. On failure, allow **at most one** mechanism-driven revision
(bug, optimization, or under-capacity — diagnose before choosing), and if it
still fails: **STOP the pair formulation** (no width/depth/lr sweep).

## 6. Stage 2 — train-only structural screen (no official valid)

Deterministic split of official train: `2048 train-dev` + `512 train-monitor`
(disjoint, seeded). Compare `pair` vs `raw` vs `raw_wide`, seed 0, short budget
(40–80 epochs). Observe train and monitor curves.

Early-stop rule: stop a branch when both curves plateau and the pair encoder is
clearly (`> 0.005` MAE) worse than `raw_wide`, or clearly (`> 0.005`) better.
If `pair ≤ raw_wide` (within ±0.005) the structure question is answered: the
pair representation is at least as good as a node-MPNN of equal budget.

---

## 7. Q2 — function-preserving dictionary insertion (only if Q1 passes)

Warm start: freeze a `pair` checkpoint. Extract `u_G`; head `f`. Insert

```
α_T = UnrolledSparseCode(u_G, D, λ),   z_G = α_T D,   ŷ = f(z_G)
```

with **signed soft-threshold ISTA** (prox of L1) and `α` initialised at `u_G`,
`D = I`, `λ = 0`, so that at insertion `z_G = u_G` **exactly**. Measure

```
max | pred_before − pred_after |
```

over a fixed batch; it must be `< 1e-4` (near float precision). If not → fix
insertion, do **not** train.

`λ` schedule: 0 at insertion, then ramped to a train-only label-free value that
targets **~25 % active coefficients** (75th percentile of `|u_G|`-projections on
a fixed train batch); frozen thereafter. No sparsity sweep.

### Matched controls (all from the SAME warm-start checkpoint, optimizer restarted, equal extra epochs)

* **A `dense_restart`** — encoder+head, no dictionary. ("just more training")
* **B `learned_dict`** — learnable `D` (`K=d=64`, init `I`), `λ` schedule.
* **C `fixed_identity`** — `D=I` fixed, same `λ` schedule. (is a learned basis needed?)
* **D `generic_topk`** — tied learned basis `W` (`d×d`), `z = W · TopK(Wᵀu)`,
  `k` set to the same final active count. (dictionary vs generic sparsity)
* **E `learned_nosp`** — learnable `D`, `λ=0`. (coordinates vs sparsity; only if cheap)

Matched: encoder params, head params, extra params (B/C/D ≈ `d²`), final active
count, warm start, budget. Any mismatch is reported.

### Success criterion (frozen)

`learned_dict` must beat **all** of `dense_restart`, `fixed_identity`,
`generic_topk` by a *practically meaningful* margin (`≥ 0.005` MAE) on the
train-monitor set, with equal budgets/params/active-count, and the mechanism
must be alive (non-collapsed `D`, task gradients reach `D`, shuffle ablation
degrades). Otherwise the honest conclusion is one of H1/H2/H4 (see §10) and the
dictionary is **not** rescued by K/λ/solver/residual/bypass changes.

---

## 8. Representation-budget accounting (mandatory, no compression claim)

Report latent dim, number of nonzeros / active fraction, coefficient precision,
index cost, dictionary storage, amortisation assumption, encoder params, head
params, actual GPU memory, wall time. **Latent sparsity ≠ whole-model
compression ≠ speedup**; no compression claim without measurement.

## 9. Compute-cost honesty (pair encoder)

Report average/max graph size, batch size, peak GPU MB, epoch wall time, and the
`O(n²d)` state / `O(n³d)`-worst-case contraction cost. If the pair encoder works
but is slow, that is a later engineering issue; if it shows no task value, do not
optimise kernels first.

## 10. Frozen verdict vocabulary / hypotheses

The outcome maps to one of:

* **H1** raw relation-preserving representation solves the task and the
  dictionary adds nothing;
* **H2** sparse bottleneck helps but learned dictionary basis ≈ generic Top-k;
* **H3** learned task-driven dictionary gives a unique matched-budget benefit;
* **H4** the pair structural representation itself is the wrong direction.

Verdict first line ∈ { "Pair-state structural representation is viable and
matches/beats the raw node baseline." ; "Pair-state structural representation is
not viable (dominated by the raw node baseline)." ; "Learned dictionary provides
a task benefit over matched controls at equal budget." ; "Learned dictionary
provides no benefit over matched controls at equal budget." ; "Inconclusive due
to implementation/optimization integrity failure." }.

## 11. Discipline / forbidden

No official test; no official valid in Stages 1–2; no width/depth/K/λ sweep; no
architecture fishing; no residual/side-channel/bypass (`head([u_G, z_G])`
forbidden); no PSCD/motifs/ports/MDL/handcrafted statistics; no seed-1 unless a
`0.005–0.02` ambiguous band appears; one mechanism revision per failed branch.

## 12. Deliverables

Preregistration (this file); Stage 0 log; synthetic sanity log; Stage 1 overfit
curve; Stage 2 screen curves + params; (if reached) insertion-preservation log,
control table, mechanism/ablation audits; analysis note; claim + decision;
`STATE.yaml` pointer. Artefacts under `tracks/ksvd/results/zinc_pair_dict_v0/`.
