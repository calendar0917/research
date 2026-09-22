# Preregistration — TCCD-v5: Occurrence-Preserving Pair Nonlinearity Audit

Round name: **TCCD-v5** (*Occurrence-Preserving Pair Nonlinearity Audit*).
Study: `zinc-context-gap`.
Status: frozen before formal experiments on 2026-09-22.

## 1. Question

TCCD-v2 established that a task-learned K=64 local prototype vocabulary and
assignment-sensitive relation composition are useful, but its absolute result
remains far above the canonical GPU1 reference. TCCD-v3b showed that a stronger
local state alone does not rescue the ceiling. TCCD-v4 showed that appending the
parameter-free `C^T S^2 C` moment adds no material frozen-screen signal.

This round tests exactly one causal placement variable:

```text
POST: phi(mean_{i<j} p_ij)
PRE:  mean_{i<j} phi(p_ij)
```

The scientific question is whether a shared nonlinear interpretation must occur
before occurrence interactions are globally pooled.

## 2. Frozen lineage and execution discipline

Base: exact TCCD-v2 PrototypeREL implementation and best checkpoint.

* TCCD-v2 checkpoint commit: `69a985a4ea3eb184c96c8ddad3857c4e87ee1dda`.
* Checkpoint: `tracks/ksvd/results/tccd_v2/prototype_rel_seed0_best.pt`.
* Checkpoint SHA-256: `093e0e46d4d5dab7e5558b807d993d1136df58d8f531d6b9b3621e8ab05c0f42`.
* Local encoder: exact `714 -> 64` linear encoder.
* Prototype vocabulary: K=64, latent width 64, cosine assignment.
* Temperature: `0.05 + 0.95*sigmoid(a)`, initialized at 0.20.
* Existing relations: exact TCCD-v2 five-relation set, unchanged.
* Internal split: seed `20260922`, 8000 train / 2000 dev.
* Primary seed: 0. Seed 1 is allowed only by the registered ambiguous gates.
* Formal compute: remote A100 GPU1 only. GPU0 is forbidden.
* Official ZINC test: blocked throughout and never loaded.
* No K-SVD, OMP/IHT, stronger local encoder, B-full local state, S^2/S^3,
  new relation, handcrafted topology, GNN, message passing, attention,
  Transformer, pair-width sweep, K sweep, temperature sweep, reader sweep,
  or pair-to-centre update.

TCCD-v2 internal references reused only for matched comparison:

* PrototypeREL best: `0.2862437069416046`.
* PrototypeREL Top-5 soup: `0.26635152101516724`.
* Corrected official-valid reference: best `0.287337`, soup `0.261988`.
* Canonical strong GPU1 reference: `0.119818`.

## 3. Pair set and relation values

For each graph with n centre/environment occurrences, use every unordered
pair `(i,j)` with `i<j`; no self-pairs and no sampling. For `n<2`, the pair
branch is a fixed zero vector.

The relation vector is read from the exact TCCD-v2 relation matrices:

```text
r_ij = [R_int(i,j), R_b1(i,j), R_b2(i,j), R_b3(i,j), R_geo(i,j)]
```

Thus the actual relation count is `m=5` and the pair descriptor width is
`3*64 + 5 = 197`.

For every unordered pair:

```text
p_ij = [c_i+c_j, |c_i-c_j|, c_i*c_j, r_ij]
```

The descriptor is symmetric under `i <-> j`. No ordered concatenation, node
identifier, canonical index, positional embedding, raw feature bypass, or new
graph statistic is permitted.

## 4. Matched nonlinear pair function

Both POST and PRE use the same two-layer pair function and output width:

```text
Linear(197, 64)
ReLU
Linear(64, 16)
```

No normalization, dropout, residual, attention, deeper MLP, or width sweep is
allowed. POST and PRE are independently trained, but use the same seed,
initialization procedure, optimizer, stopping rule, reader, and regularizers.
Initial state checksums are recorded to verify comparable initialization.

## 5. Graph representation

The frozen TCCD-v2 base representation is retained exactly:

```text
h_base = [sum_i c_i, {vec_sym(C^T R_r C)}_r]
```

The only new block is 16-dimensional:

```text
POST = [h_base, phi(mean p_ij)]
PRE  = [h_base, mean phi(p_ij)]
```

POST and PRE therefore have identical output width and identical final-reader
parameter count. BASE has no new 16D block and is a secondary control.

## 6. Assignment-shuffle intervention

For each graph, generate one deterministic fixed permutation using seed
`20260922` and the graph's original index. Form `C_pi=P_pi C` only inside the
new PRE pair branch. All base `C^T R C` features remain real and unshuffled.
No new model is trained for PRE-SHUFFLE; it is evaluation-only.

## 7. Gate 0 — correctness

Gate 0 must pass before formal work:

1. pair-swap descriptor invariance;
2. graph node-relabel invariance of pair multiset, POST branch, PRE branch,
   and prediction;
3. pair-enumeration-order invariance;
4. single-graph versus batched computation invariance;
5. linear-phi equivalence, `mean(Wp+b) == W mean(p)+b`;
6. nonlinear separation on a heterogeneous synthetic pair set;
7. Stage-A gradients reach pair MLP and reader only; Stage-B gradients reach
   pair MLP, reader, prototypes, local encoder, and temperature;
8. pair branch has no target/raw-feature/official-test access;
9. official test remains explicitly blocked.

Any Gate-0 failure stops the round. No formal training is allowed after a
failed Gate 0.

## 8. Stage A — frozen PrototypeREL screen

Reuse the exact TCCD-v2 best checkpoint. Freeze local encoder, prototypes, and
temperature. Cache only C assignments, exact pair indices, exact relation
values, base features, and metadata; construct p on GPU in vectorized batches.
Do not write all high-dimensional pair descriptors to disk.

Arms:

* A0 BASE: exact matched frozen TCCD-v2 reader result, reused from TCCD-v4
  when protocol and feature dimensions match; otherwise one seed-0 rerun.
* A1 POST: train only the 197->64->16 pair MLP and final linear reader on
  `phi(mean p_ij)`.
* A2 PRE: train only the same pair MLP and final reader on `mean phi(p_ij)`.
* A3 PRE-SHUFFLE: evaluation-only intervention on the trained PRE model.

Stage-A optimizer and stopping are the exact TCCD-v2 lightweight protocol:
Adam, batch 32, learning rate 1e-3, weight decay 1e-5, gradient clip 5,
maximum 240 epochs, patience 40, and Top-5 checkpoint soup. Because C is
frozen, the TCCD-v2 assignment regularizers are constants and do not affect
Stage-A gradients; only pair MLP and reader parameters are optimized.

Primary metric is Top-5 soup MAE:

```text
delta_place  = MAE_POST_soup - MAE_PRE_soup
delta_add    = MAE_BASE_soup - MAE_PRE_soup
delta_shuffle= MAE_PRE_SHUFFLE_soup - MAE_PRE_soup
```

Best checkpoints are reported but do not override the soup-first decision.

### Stage-A decision

**STRONG PASS:** all three deltas are at least `0.015`; authorize Stage B.

**FAIL:** any delta is below `0.005`; stop. Do not run Stage B.

**AMBIGUOUS:** no delta is below `0.005`, but at least one is below `0.015`;
run exactly one paired seed-1 screen. The two-seed means must satisfy all three
means `>=0.010` to authorize Stage B; otherwise stop.

## 9. Stage B — end-to-end co-adaptation

Only after Stage-A PASS. Run matched POST-E2E and PRE-E2E models with the exact
TCCD-v2 local encoder, K=64 prototypes, temperature, regularizers, optimizer,
reader, split, epochs, and seed. Only the new pair MLP and its placement differ.
BASE is reused from the exact TCCD-v2 internal reference unless protocol drift
makes it incomparable.

Primary metrics remain Top-5 soup:

```text
delta_place_e2e = MAE_POST_E2E_soup - MAE_PRE_E2E_soup
delta_base_e2e  = MAE_TCCD_V2_BASE_soup - MAE_PRE_E2E_soup
```

**STRONG PASS:** both deltas are at least `0.020`.

**FAIL:** either delta is below `0.005`; stop.

**AMBIGUOUS:** neither is below `0.005`, but one or both is below `0.020`;
run exactly one paired seed-1 comparison. Both two-seed means must be at least
`0.010` to pass.

On the final PRE-E2E model, report evaluation-only PRE-SHUFFLE. The intervention
must satisfy `MAE_shuffle - MAE_real >= 0.010` to claim that the gain uses
correct occurrence-to-relation alignment rather than generic capacity.

## 10. Vocabulary diagnostics

For every E2E model reaching diagnostics, report active/dead prototypes,
effective count, top-1 and top-8 usage, local entropy, global usage entropy,
learned temperature, and semantic coherence. Collapse invalidates the claim
of a healthy reusable prototype vocabulary even if prediction improves.

## 11. Official-valid authorization

Do not run official train/valid merely because Stage A or B shows a small gain.
Only after Stage B PASS and at least one route below is met may a single GPU1
official train 10000 / valid 1000 run be authorized:

* Route A: PRE-E2E internal soup `<=0.23`.
* Route B: improvement versus TCCD-v2 internal soup `>=0.030`.

Official test remains blocked even if official-valid is authorized.

## 12. Absolute interpretation

For any authorized official-valid result:

* `soup <=0.20`: major rescue;
* `0.20<soup<=0.24`: partial rescue;
* `0.24<soup<0.262`: small mechanism gain;
* `soup>=0.262`: no practical absolute improvement.

If PRE fails, the next authorized hypothesis becomes center-preserving
composition, e.g. per-centre `u_i = mean_j phi(c_i,c_j,r_ij)` followed by
centre-aware pooling. That architecture is explicitly forbidden in TCCD-v5.

## 13. Durable records

Required tracked records after local analysis:

* this preregistration;
* Gate-0 result and targeted tests;
* Stage-A result;
* Stage-B result only if authorized;
* official-valid result only if authorized;
* analysis note, claim, decision, and `tracks/ksvd/STATE.yaml` update.

Every formal result records preregistration commit, implementation commit,
GPU1, checkpoint SHA, split, pair-count statistics, relation count, descriptor
width, pair-MLP parameters, BASE/POST/PRE metrics, PRE-SHUFFLE, runtime,
vocabulary diagnostics, stop reason, and official-test status.
