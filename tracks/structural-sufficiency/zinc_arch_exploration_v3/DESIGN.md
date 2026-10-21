# DESIGN — zinc-arch-exploration-v3

**Round:** `zinc-arch-exploration-v3`
**Track:** `tracks/structural-sufficiency/`
**Status:** declared before the first formal seed-0 run; timing/prep numbers are
filled in from measured runs (timing is engineering input, not a stop rule).
**Results:** see `REPORT.md` (all four arms finished, one follow-up used, one
frozen test read; `analysis/` holds the tables, curves, freeze file and test
reporting).
**Data:** official PyG ZINC `subset=True`, train 10,000 / valid 1,000 / test 1,000,
raw-`y` MAE. Valid drives every development decision; test is evaluated once at
the end on the frozen best-valid checkpoints of this round (exposed benchmark,
reported as such).

---

## 0. Boundary

* Predictor parameter cap 500,000 (trainable, including the P3 SSL encoders).
  Temporary A-SSL reconstruction heads are reported separately (6,176).
* No recursive local neighbour hidden-state message passing. Allowed and used:
  shared static encoders, one full-graph pair composition, global static
  operators, one explicit association contraction.
* Inputs are the raw atom/bond graph (`x`, `edge_index`, `edge_attr`) and
  deterministic label-free functions of it. No `y`, no SA/logP component, no
  target-derived feature, no old oracle cycle target.
* No new exact-pattern lookup table, no VQ/codebook, no explanatory loss.
* Only train molecules (and train labels) are used for training. Valid is used
  for per-epoch evaluation and best-valid selection; test is not read during
  training or selection.
* Reference numbers are reused, not retrained: P3B_s0 0.076223751 valid /
  0.069864841 test; L_s0 0.078537150 / 0.066266555; C_s0 0.082367432 /
  0.070014820 (authoritative source: `zinc_dual_channel_composer_v1/`).

## 1. What is new relative to prior implementations

| prior work | its cross-object computation | what it does not do |
|---|---|---|
| P3 Null composer | `c_b = MLP([b_e, n_b·mean_j w_j, n_b·mean_j w_j², log1p n_b, bond_feat])`, sum over bonds | pair objects never exist; the intermediate atom of a junction is averaged away before any nonlinearity |
| T (rank-16) | `Σ_incidence U a_v ⊗ V b_e`, `Σ U b_e ⊗ V w_j` | 2-point, linear in the two objects, no intermediate-atom conditioning, no all-pairs objects |
| J / Q induced paths | shared φ over full typed induced path content (≤5 edges), pooled per root | local exact subgraph enumeration; no all-pairs object and no composition through a *shared intermediate atom at arbitrary distance* |
| DLP | pair moments pooled by hard/soft distance buckets | pair content is a fixed pair state; no contraction over a third object |
| STA / ZSRT | attention bias from type/distance/overlap; RRWP | relation enters as a bias on atom attention, not as a joint higher-order invariant |
| M / S trunks | global moments or slot context re-mixed into each object | context is a per-graph summary; no explicit `i–k–j` association |
| L / C | parallel local composer over real junctions | parallel path, local only, plus a P3 Null base |

This round buys two computations that none of the above performs:

1. **A** — for *every* atom pair `(i,j)` a nonlinear pair object `X_ij`, followed
   by one contraction `U_ij = (1/n) Σ_k φ(X_ik)·ψ(X_kj)·χ(a_k)` (channel-wise
   product), then a per-pair nonlinear `H([X_ij,U_ij])` and only then graph
   pooling. The intermediate atom `k` keeps its identity through `χ(a_k)` and
   through the full pair descriptors `X_ik`, `X_kj`.
2. **B** — third-order invariants of static whole-graph operators,
   `T_(r,t,s,q) = tr(D_u_r K_t D_v_r K_s D_w_r K_q)`, where `K` is a library of
   full matrix functions (heat kernels of the symmetric normalized Laplacian)
   plus typed normalized adjacencies. This is a genuine three-index contraction
   of content and structure, not a renamed pair moment.
3. **A-SSL** — the *new* composition trunk (not just the local encoders) is
   pretrained with fresh-mask masked atom/bond reconstruction from rebuilt
   masked features, then fine-tuned with the identical supervised recipe as A.

The hypothesis is shared *composable* functions over static structure — not
per-pattern memorisation — can generalise to local environments unseen in
training. It is a working hypothesis, not a promise.

## 2. Arm A — full-graph pair composition (`PairComposition`)

Static inputs (per molecule, from the raw graph):

* P3 object table (`a_v`/`b_e`/`w_j` encoders, fine-tuned; seed-0 train-only
  masked-SSL initialisation, sha256 `006fabc1…`).
* New `pair_*` cache (`zae_data.build_pair_mol`), label-free, per unordered pair:
  shortest-path distance (0–8, 9 = no path), raw bond type for adjacent pairs
  (read from the possibly-masked bond table at collate time), common-neighbour
  count (0–5), ring-bond flag, same-biconnected-block flag (topology only).

Computation:

```
g_i   = LN(MLP([a_i, mean_{j∈Jedge(i)} w_j, mean_{j∈Jcentre(i)} w_j,
                log1p|Jedge(i)|, log1p|Jcentre(i)|]))            # 96, static
p_i   = W_p g_i                                                # 48
X_ij  = LN(MLP([p_i+p_j, |p_i-p_j|, p_i·p_j, 1[adj]·b_ij, 1[adj], R_ij]))   # 64
        R_ij = [dist emb 8, bond-type emb 8, common emb 8, ring emb 4, block emb 4]
U_ij  = (1/n) Σ_k φ(X_ik) ⊙ ψ(X_kj) ⊙ χ(a_k)                   # 32 channels
h_ij  = H([X_ij, U_ij])                                        # 96
z     = [ LN(Σ_{i≠j}h/√P), LN(mean_{i≠j}h), LN(mean_{i≠j}h²), LN(mean_i h_ii),
          LN(mean a), LN(mean b), LN(mean w), log1p counts, global33 ]
y     = Reader(z)      # 709 → 192 → 192 → 1
```

Declared conventions: ordered pairs including the diagonal; `X` is exactly
symmetric in `(i,j)`; `φ≠ψ` are different maps on the ordered pair, so
`U_ij ≠ U_ji` in general and the pooled readout sums over all ordered pairs
(invariant under relabelling, checked numerically); the contraction is divided
by `n` (mean over intermediate atoms), and the count `n` is kept as an explicit
`log1p` channel; the four pair statistics are per-sample LayerNorm'd separately
because their raw scales differ by ~150× (measured), while counts and `global33`
stay raw.

Parameters **488,601** (encoders 202,048; endpoint state 37,440; pair projection
4,656; pair descriptor embeddings 200; `X` MLP 43,328 + LN 128; φ/ψ/χ 7,264;
`H` 18,624; 7 readout block LNs 1,344; reader 173,569).

## 3. Arm B — multi-point invariants of static operators (`SpectralJoint`)

Operator library `K` (built per batch from the padded adjacency, `no_grad`):

* `exp(-t L)`, `L = I - D^{-1/2}A D^{-1/2}`, `t ∈ {0.5, 1, 2}` — full matrix
  functions via `Q exp(-tΛ) Qᵀ` (no eigenvector selection, no sign convention,
  stable under repeated eigenvalues);
* `D^{-1/2} A_b D^{-1/2}` for raw bond types `b ∈ {1,2,3}` (typed one-step
  operators; a masked bond contributes no typed edge);
* every operator is exactly zero on padded rows/columns (checked).

Invariants (per molecule):

```
s_i   = LN(MLP([a_i, junction means, log1p counts]))          # 96 (same static state as A)
u,v,w = tanh(W_u s), tanh(W_v s), tanh(W_w s)                  # 6 channels each
pair  P_(r,t,s)     = tr(D_u_r K_t D_v_r K_s)                  # 21 unordered kernel pairs
triple T_(r,t,s,q)  = tr(D_u_r K_t D_v_r K_s D_w_r K_q)        # 84 declared triples:
                       3³ HHH + 3·3·3 THH + 3·3·3 TTH + 3 TTT-same-type
first F_(r)         = Σ_i u_r(i)v_r(i)w_r(i),  Σ_i u, Σ_i v, Σ_i w
z     = [ LN(T), LN(P), LN(F), log1p counts / ratios, global33 ]
y     = Reader(z)      # 654+6+33 → 224 → 224 → 1
```

Each trace is divided by `n · ‖K_t‖_F ‖K_s‖_F ‖K_q‖_F` (deterministic scale from
the graph, no split statistic), and the three invariant families get separate
per-sample LayerNorms because their raw magnitudes differ by ~4 orders of
magnitude (measured; a shared norm suppresses the triple block).

Parameters **448,623**.

## 4. Arm A-SSL — native masked pretraining of the new trunk

* Mask: the frozen `bjoc_masked_pretraining_v1` sampler, 15 % of atom rows and
  15 % of physical bonds, `SeedSequence([0, mol_id, epoch])` → **a new mask on
  every visit**, at least one object of each kind.
* Rebuild: raw codes are replaced by mask tokens and *every* derived feature is
  rebuilt from the masked raw graph (`build_mol_objects`). Neighbour-type
  histograms, radius-2 histograms, junction bond-type descriptors, typed pair
  features (read from the masked bond table), typed kernel edges and `global33`
  therefore see the mask, not the answer. Topology, distance, common-neighbour,
  ring/block flags stay (they are functions of the untyped graph).
* Task: cross-entropy on the hidden raw atom class (28-way) from
  `[h_ii, mean_{j≠i} h_ij]`, and on the hidden raw bond class (4-way) from
  `[h_ij, h_ji]`. Both heads read the new composition state, so the pretext
  trains the new trunk; the final reader is not part of the pretext.
* No labels, no valid/test molecule, no target component. Reconstruction heads
  (6,176 parameters) are dropped before fine-tuning.
* Length: 120 epochs, declared in advance; the fixed last-500-train-molecules
  reconstruction curve decides the checkpoint (best atom+bond CE, ties → earlier
  epoch). Declared extension rule: if the holdout objective is still improving
  by > 0.01 over the last 30 epochs at epoch 120, extend once to 200 epochs.
* Supervised phase: identical architecture, initialisation source and recipe as
  A, all parameters fine-tuned, no reconstruction loss.

## 5. Recipe (declared before the first formal run)

| item | value |
|---|---|
| data | official PyG ZINC subset train 10,000 / valid 1,000 |
| loss | raw-`y` L1 |
| batch | 64 train (last partial batch kept, 16 rows), 128 valid |
| optimizer | AdamW, lr 1e-3 → 1e-5 per-step cosine, warmup 1,000 steps |
| weight decay | 1e-4 |
| grad clip | 5.0 |
| dropout | 0.0 |
| schedule | arm A and arm B: 1,200 epochs × 157 steps = 188,400 updates. `A_SSL` and `A_attn`: **900 epochs × 157 steps = 141,300 updates** (declared at launch, see §7 budget decision) |
| valid | full 1,000 every epoch; best-valid by strict `<`, ties → earlier epoch |
| eval | best and final both re-scored on full 10k train + 1k valid in eval mode |
| SSL (A-SSL only) | 120 epochs × 149 steps = 17,880 updates, AdamW 1e-3 → 1e-5, warmup 300, batch 64, CE(atom)+CE(bond) |
| seeds | seed 0 only for A, B, A-SSL in this round |

Measured timing (A100-SXM4-40GB, full 10k train + full valid per epoch):

| arm | s/epoch (steady) | planned epochs | projected GPU-h (with 15 % + startup) |
|---|---:|---:|---:|
| A | 7.76 | 1,200 | 3.0 |
| B | 9.73 | 1,200 | 3.7 |
| A-SSL pretrain | 11.0 (incl. fresh-mask rebuild) | 120 | 0.4 |
| A_SSL / A_attn supervised | 7.4 | 900 | 1.9 each |

Measured with 4 full epochs (`runs/timing_A_s0`, `runs/timing_B_s0`, 3 epochs for
SSL) on `res` GPU 0/1; the steady value is the mean of epochs 2..N including a
full 1,000-molecule valid pass. The first B measurement (16.8 s/epoch) used a
per-step eigendecomposition; the operator library is now precomputed at prep
time and verified against the on-the-fly path (max abs diff 6e-7), giving the
9.73 s/epoch above. Projected wall for the three formal trajectories ≈ 6 h with
A and B in parallel on the two GPUs.

(Each run's horizon is fixed at its own start and is never reset or compressed
mid-run. GPU time has no hard cap in this round; timing is engineering
information for scheduling, not a stop rule — but the horizon for a run that has
not started yet may be chosen from completed-run evidence, which is what
happened for the last two trajectories: see the budget decision in §7.)

## 6. Checks performed before formal runs (`analysis/cpu_checks.json`)

* parameter counts under the cap; optimizer covers every parameter (checked in
  the training loop);
* exact permutation invariance (rebuilt objects + pair cache from a permuted raw
  graph): max |Δŷ| 3e-8 (A), 2e-8 (B); sorted `U` rows agree to 7e-8;
* batch isolation and padding: alone vs batched 1.5e-8, and vs a batch with a
  larger `Nmax` 2.2e-8 → no cross-graph mixing, no padding leak;
* diagonal/off-diagonal bookkeeping (`|diag| = n`, `|offdiag| = n(n−1)`, unique
  dense slot per pair, adjacency count = 2·bonds);
* kernels: finite, symmetric (0 error), zero on padding (0 leak), heat
  eigenvalues in [0, 1]; masked bonds create no typed edge;
* A contraction is genuinely triadic: 71 % of `U` variance is not linearly
  explained by `X`; perturbing atom `k`'s content changes `U_ij` for pairs that
  do not touch `k` (3.5e-2); perturbing an unrelated `X` row changes `U_ij` by
  exactly 0;
* B triple block is not degenerate: 8.8 % residual variance beyond
  pair+first-order invariants (HHH 7.5 %, THH 8.2 %, TTH 11.0 %, TTT 10.7 %),
  non-constant, deterministic, and sensitive to bond-type changes;
* A-SSL masking: with the same mask but a different *hidden* atom/bond class the
  whole input table is byte-identical; masked inputs differ from unmasked;
  every neighbour-type histogram drops the hidden type by exactly 1; masked
  bonds remove a typed edge;
* gradient audit: all parameter tensors receive nonzero finite gradient under
  the supervised loss; φ/ψ/χ/H/X/endpoint/reader/encoder paths all active;
* recipe arithmetic: 157 steps/epoch, 16-row last batch, warmup 1,000 < horizon;
* 80-step 32-molecule overfit probe on train rows (execution only): L1
  1.06 → 0.034 (A), 1.06 → 0.039 (B).

## 7. Run matrix, follow-up rule, test plan

Formal seed-0 runs: **A**, **B**, **A-SSL** (A and B in parallel on the two
GPUs; A-SSL next; each on its own GPU).

After the first pass, at most **one** substantive follow-up is allowed, and only
with a written three-sentence justification (why, what, decision criterion).
That slot was used for **`A_attn`** (content-based pooling of the pair set);
the declaration is `analysis/followup_A_attn.md`, written before the run started
and based on seed-0 valid only. No LR/WD/rank/depth grids.

### Budget decision taken after arm A and arm B finished (or were stopped)

Evidence (`runs/A_s0/curve.csv`, `runs/B_s0/curve.csv`):

| run | best valid | best epoch | epochs declared | share of the horizon spent after the best | action |
|---|---:|---:|---:|---:|---|
| A | 0.082189 | 632 | 1,200 | 47 % | completed (horizon was already declared) |
| B | 0.101720 | 332 | 1,200 | 72 % | **cancelled at epoch 817** of 1,200 |

B was stopped because its best valid was set at epoch 332 and did not move for
485 further epochs even after entering the learning-rate band
(`lr` 2.4e-4) where P3B, L and A all set their bests, while it remained 0.0255
behind P3B — no remaining epoch could change its rank. The run is recorded as
`status: CANCELLED_BY_BUDGET` with `epochs_run: 817` (recovered from the epoch
log by `code/zae_recover.py`, which refuses to proceed unless `best.pt` matches
the best epoch line); its `best.pt` and curve are kept, nothing is deleted.

Because A's last 568 epochs bought 0.0016 valid at 47 % of the compute, the two
trajectories that had not started yet were **declared** at 900 epochs
(141,300 updates) instead of 1,200: that keeps the whole productive
`lr` 3e-4 → 1e-5 band inside the horizon, and every A-versus-`A_attn`
comparison is additionally reported in a matched learning-rate band
(`lr` ∈ (1e-4, 3e-4]) so the horizon difference cannot create a difference.

Test: after all best-valid checkpoints of this round are frozen (model choice,
epoch, recipe, no further tuning), evaluate exactly those checkpoints on the
official test split with the pair cache built from raw test graphs, and report
test MAE together with valid. P3B/L/C test numbers are quoted from the existing
`zinc_dual_channel_composer_v1/analysis/test_reporting.json`, not re-run. The
test split is an already-exposed benchmark; the new numbers are descriptive.

## 8. Reproduction

```bash
# data preparation (pair descriptors; label-free)
uv run python tracks/structural-sufficiency/zinc_arch_exploration_v3/code/zae_runner.py prep

# CPU checks
uv run python tracks/structural-sufficiency/zinc_arch_exploration_v3/code/zae_checks.py

# formal runs (remote A100 through rr)
rr run res zae-A-s0    --pool res-gpu0 --gpus 1 --time 3:00:00 --result <runs> -- \
  python tracks/structural-sufficiency/zinc_arch_exploration_v3/code/zae_runner.py train --arm A --seed 0
rr run res zae-B-s0    --pool res-gpu1 --gpus 1 --time 3:00:00 --result <runs> -- \
  python tracks/structural-sufficiency/zinc_arch_exploration_v3/code/zae_runner.py train --arm B --seed 0
rr run res zae-ASSL-pre --pool res-gpu0 --gpus 1 --time 2:00:00 --result <runs> -- \
  python tracks/structural-sufficiency/zinc_arch_exploration_v3/code/zae_runner.py ssl --seed 0
rr run res zae-ASSL-s0 --pool res-gpu0 --gpus 1 --time 3:00:00 --result <runs> -- \
  python tracks/structural-sufficiency/zinc_arch_exploration_v3/code/zae_runner.py train --arm A_SSL --seed 0 \
  --init-from tracks/structural-sufficiency/zinc_arch_exploration_v3/runs/ASSL_pretrain_s0/pretrain.pt
```
