# Preregistration — ZINC strict-static dictionary-pair v0

Round name: **ZINC-strict-static-dictionary-pair-v0**.
Study: `zinc-context-gap`.
Status: frozen before any formal (GPU) experiment.

User authorization for this round is exactly three seed-0 full-training runs:

```text
S0       strict-static baseline        seed 0
S-Dense  S0 + generic residual MLP     seed 0
S-Dict   S0 + residual dictionary      seed 0
```

No seed 1/2/3, no official test, no HPO, no rescue.

## 1. The one question

> On a modern, strongly-trained **strict-static** patch-pair baseline, is a
> residual *learnable dictionary* more valuable than a parameter-matched
> *generic dense* residual adapter?

The occurrence-level dictionary-conditioned pair kernel is **out of scope** in
this round.

## 2. Architecture contract (non-negotiable)

The forward pass may not contain any message-passing-style interaction::

    pair states -> centre/patch aggregation -> patch hidden update    # FORBIDDEN
    h_i' = Update(h_i, Aggregate_j(q_ij))                             # FORBIDDEN

and no relation/pair may be recomputed *after* such an update.  Allowed:
static graph / patch preprocessing, one shared per-patch local encoder,
unordered patch-pair direct computation, relation/path descriptors, invariant
graph-level pooling, graph-level topology features, end-to-end backprop.

The baseline therefore sets ``center_context = False`` (so ``center_update is
None``) and never calls ``_pool_pairs_to_centres``.  Every pair state is
computed exactly once and is only pooled at graph level.

## 3. S0 — strict-static compact-v4-smallhead

Inherited verbatim from the modern compact-v4-smallhead protocol except for the
single architecture change:

```text
patch radius            2
patch_hidden            48
pair_hidden             16
token_width             16
embedding_mode          hybrid
embedding_rank          4
historical aliased tokenizer (hybrid_full_typed_tokens 768,
                              hybrid_full_parent_tokens 32)
topology_mode           hinge (25D -> 16 -> 8)
small raw head          R -> 13 -> 13 -> 1   (GenericReader, 4,135 params)
dropout                 0.05
center_context          False     <-- THE strict-static change
```

Local state and pair computation::

    x_i -> patch_encoder -> h0_i in R^48
    u_i  = pair_projection(h_i) in R^16
    q_ij = pair_encoder([u_i + u_j, |u_i - u_j|,
                         (u_i * u_j) * distance_gate, relation_encoder(r_ij)])

The two `u_i` are the post-local-state projections.  The pair is evaluated
exactly once; nothing is written back.

**The graph representation width is measured at runtime, not assumed.**  For
this configuration the runtime width is 302D
(`[unary moments 97 | pair moments 5x33 | global 32 | topology 8]`), and the
small head is built from the *real* `unified_graph_width` via `GenericReader`.
S0 total = 66,228 trainable parameters (including the dormant residual-scale
scalar), S0 head = 4,135.

## 4. S-Dict — residual dictionary (fixed, no sweep)

Inserted directly on the local-encoder output `h0_i`::

    q_i = Wq(h0_i)                       # 48 -> 32
    q_i = l2_normalize(q_i)
    D_k = l2_normalize(dictionary_k)     # K x 32, forward normalization
    tau = 0.05 + 0.95 * sigmoid(tau_logit)
    alpha_i = softmax((q_i @ D^T) / tau)
    d_i = alpha_i @ D
    delta_i = Wv(d_i)                    # 32 -> 48
    h_i  = h0_i + gamma * delta_i

```text
K = 64
d = 32
tau init ~= 0.20     (tau_logit init = logit((0.20-0.05)/0.95))
gamma init = 0.1     (learnable)
Wq / Wv / dictionary: normal small random init
```

The final projection is **not** zero-initialised: zero-init has previously
truncated the upstream gradient and silently disabled a branch in this repo.
The residual path always keeps `h0_i`; `h_i = d_i` or any dictionary-only
bottleneck is forbidden.

S-Dict branch parameters = 5,201 (Wq 1,568 + atoms 2,048 + Wv 1,584 + tau 1).

## 5. S-Dense — strict parameter-matched control

```text
h0_i in R^48
delta_i = Linear(48, H) -> SiLU -> Linear(H, 48)
h_i = h0_i + gamma * delta_i          (same gamma init 0.1)
```

`H` is chosen programmatically as the integer minimising the added-parameter
mismatch against S-Dict.  Result: `H = 53`, dense branch parameters = 5,189,
relative mismatch **0.23%** (< 1%).  Dense and Dict share **bit-identical**
baseline tensors (max_abs_diff == 0), the same reader initialisation, the same
optimizer / data order / seed / training protocol; only the residual branch
differs.

## 6. No dictionary regularization

Training loss stays the inherited `L1 / MAE`.  Forbidden: reconstruction loss,
K-SVD objective, top-k, L1 sparse-code penalty, entropy/balance/orthogonality/
contrastive/auxiliary-prototype losses.  Prototype statistics are diagnostics
only and never enter selection.

Reported diagnostics (report-only): mean assignment entropy, effective
prototype count, active/argmax prototype counts, top-8 assignment mass, max/min
mean assignment mass, dictionary coherence (mean/max |cos|), mean residual norm,
mean `||h0||`, grad norms for dictionary / Wq / Wv / tau, and the inference
ablation "residual forced to zero" (mean/max prediction shift).

## 7. Programmatic hard tests

Local CPU tests (`tracks/ksvd/tests/test_zinc_static_dictionary_pair.py`) and a
remote `integrity` stage verify:

* **A.** `center_context == False`, `center_update is None`; with
  `_pool_pairs_to_centres` monkeypatched to raise, forward still succeeds.
* **B.** capturing `h_i`, then mutating pair relations, leaves `h_i`
  **bit-identical** (and the graph prediction changes, so the test is not
  vacuous) — the core no-pair-to-local-feedback gate.
* **C.** pair encoder and relation encoder are each called exactly once per
  forward.
* **D.** pair-order and node-relabel permutation invariance.
* **E.** on a real mini-batch, dictionary / Wq / Wv / both dense MLP layers get
  strictly positive task gradient and `gamma` is finite; abort (after one fresh
  branch re-init) if not.

## 8. Training protocol (inherited, not re-tuned)

```text
optimizer            Adam
lr                   1e-3
weight_decay         1e-5
batch_size           128
max_epochs           240
patience             40
scheduler            none
loss                 L1 / MAE
gradient_clip_norm   5.0
single stage
checkpoint selection best official-valid MAE
```

Official train (10,000) / official valid (1,000), seed 0.  A fixed Top-5
weight soup over the five best validation epochs is recorded as a corroborating
secondary metric (no soup framework is developed).  Official **test is never
loaded**.

## 9. Budget and stopping rule

* Phase 0 (local): implementation, targeted pytest, CPU tiny forward/backward,
  parameter audit.  No full test suite.
* Phase 1 (server): preflight, `research doctor`, ZINC availability, commit +
  deploy, tiny GPU smoke.
* Phase 2: exactly three full-training runs — `S0 seed0` first, then, after the
  GPU regime is confirmed, `S-Dense seed0` (GPU0) and `S-Dict seed0` (GPU1) in
  parallel.

Forbidden: seed1/2/3, official test, hyperparameter / tau / K / width /
sparsity / top-k sweeps, dictionary-pair experiment, additional rescue.  After
the three seed-0 runs the round stops.

## 10. Pre-registered interpretation

```text
M0 = best-checkpoint valid MAE(S0)
MD = best-checkpoint valid MAE(S-Dense)
MK = best-checkpoint valid MAE(S-Dict)

gain_vs_base_dict   = M0 - MK
gain_vs_base_dense  = M0 - MD
dict_specific_gain  = MD - MK
```

Primary metric is the best-checkpoint official-valid MAE (the inherited
`checkpoint_selection`); the fixed Top-5 soup is a corroborating secondary
metric, and both cases are reported.

* **Case A — `DICT_SPECIFIC_SIGNAL`**: `M0 - MK >= 0.004` and `MD - MK >= 0.002`.
  Still no seed 1 this round; only recommend a next stage that buys seed 1 and
  implements the occurrence-level dictionary-conditioned pair kernel.
* **Case B — `GENERIC_CAPACITY_SIGNAL`**: `M0 - MK >= 0.004` but
  `MD - MK < 0.002`.  Do not claim a dictionary-specific value.
* **Case C — `LOCAL_DICTIONARY_NO_CLEAR_SIGNAL`**: `M0 - MK < 0.004`.  Stop the
  local dictionary route; no sparsity/temperature/K rescue.

These are exploratory seed-0 gates; no statistical-significance claim is made.

## 11. Scientific controls and forbidden confusions

* The compact-v4 original baseline has `center_context = True` and therefore
  contains pair -> centre/patch feedback; it is **not** a strict-static
  baseline.  This round's S0 is the strict-static baseline and is trained here.
* TCCD ≈ 0.25 is not used as a non-MP ceiling (TCCD has an early-occurrence /
  prototype-statistic bottleneck).
* The tokenizer is unchanged and is described as the **historical aliased
  rooted-topology token** (not an "exact typed token").
* Official ZINC test is never used for any architecture decision.
