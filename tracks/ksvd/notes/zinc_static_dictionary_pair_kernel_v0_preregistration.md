# Preregistration — ZINC static dictionary-conditioned pair kernel (SDPK-v0)

Round name: **ZINC-static-dictionary-pair-kernel-v0** (SDPK-v0).
Study: `zinc-context-gap`.
Status: frozen **before** any formal (GPU) experiment.

## 0. Scope and budget

One performance-oriented question, one candidate architecture, no ablation grid.

```text
MAX FULL RUNS = 2
  Run 1  SDPK-v0 seed 0    (always)
  Run 2  SDPK-v0 seed 1    (only if the seed-0 GO gate fires)
official test  forbidden (never loaded)
matched dense control  not purchased in this round
```

No `K` sweep, no `rank` sweep, no `coord_dim` sweep, no `tau` sweep, no pair
width sweep, no sparsity / entropy / orthogonality / reconstruction objective,
no multi-codebook, no higher-order/triadic kernel, no message passing, no
centre update, no relation refresh, no attention, no new tokenizer / topology /
graph head / objective, no seed 2/3, no rescue.

## 1. The one question

> If the learnable dictionary is promoted from a **bypassable local residual
> adapter** to the **core coordinate system of the occurrence-level static
> pair / relation kernel**, does absolute strict-static ZINC performance open a
> new band (e.g. `0.12x`)?

This round is *not* a mechanism-control round.  It does not ask
"dictionary vs generic local capacity".  It asks only whether the
dictionary-conditioned static pair kernel improves absolute MAE against the
modern strict-static S0 baseline.  The previous round's residual dictionary
(`h_i = h0_i + gamma * delta_dict`) and its `gamma`-collapse question are
**removed entirely**: there is no local dictionary residual and no `gamma` in
this architecture.

## 2. Architecture contract (non-negotiable)

```text
NO message passing
NO pair -> centre/patch write-back
NO relation refresh
NO second pair evaluation
```

Full forward:

```text
h_i     = LocalEncoder(x_i)              # h_i = h0_i, no residual
alpha_i = Dictionary(h_i)                # pure function of the local state
c_i     = Coordinate(alpha_i)

q_ij    = PairKernel(h_i, h_j, c_i, c_j, r_ij)   # evaluated exactly once

graph_repr = Pool_i(h_i) + Pool_ij(q_ij) + global/topology features
y_hat      = Head(graph_repr)
```

Absolute prohibitions verified by hard tests:

```text
Aggregate_j(q_ij) -> h_i        FORBIDDEN
h_i' -> recompute q_ij          FORBIDDEN
pair -> centre -> pair          FORBIDDEN
```

## 3. Local patch state (inherited S0, unchanged)

```text
x_i -> patch_encoder -> h_i in R^48
h_i = h0_i
```

No dictionary residual, no trainable `gamma`, no pulse training.  The
dictionary cannot collapse the local-state contribution to zero through a
residual scale because it does not touch the local state at all.

## 4. Dictionary coordinates (fixed; end-to-end, no pretraining)

```text
K          = 64
dict_rank  = 32
coord_dim  = 16
tau_init   = 0.20

s_i = Wq(h_i)                              # 48 -> 32
s_i = l2_normalize(s_i)
D_k = l2_normalize(dictionary_k)           # 64 x 32 (forward normalization)
tau = 0.05 + 0.95 * sigmoid(tau_logit)
alpha_i = softmax((s_i @ D^T) / tau)       # 64D
c_i = U(alpha_i)                           # 64 -> 16
```

`D`, `Wq`, `U`, `tau` are learned **only** by the final ZINC L1/MAE gradient.

Forbidden: K-SVD pretraining, reconstruction loss, sparse-code supervision,
entropy loss, balance loss, top-k, orthogonality loss.

## 5. Occurrence-level pair kernel

Inherited raw interaction (unchanged):

```text
u_i = pair_projection(h_i)                 # 48 -> 16
raw_sum  = u_i + u_j
raw_diff = |u_i - u_j|
raw_prod = u_i * u_j
rel_ij   = relation_encoder(r_ij)          # existing static relation
gate_ij  = 1 + tanh(distance_gate(bucket)) # existing distance gate
```

Dictionary pair coordinates:

```text
dict_sum  = c_i + c_j                      # 16
dict_diff = |c_i - c_j|                    # 16
dict_prod = c_i * c_j                      # 16
```

Dictionary-conditioned multiplicative gate:

```text
gate_input = [dict_sum, dict_diff, dict_prod, rel_ij]   # 48 + 16 = 64
m_ij       = 1 + tanh(Wm(gate_input))                   # 64 -> 16
conditioned_prod = raw_prod * gate_ij * m_ij
```

Final pair input (frozen layout, exactly this order):

```text
pair_input = [
    raw_sum,            # 16
    raw_diff,           # 16
    conditioned_prod,   # 16
    rel_ij,             # 16
    dict_sum,           # 16
    dict_diff,          # 16
    dict_prod,          # 16
]                       # total 112
q_ij = pair_encoder(pair_input)          # 112 -> 64 -> 16, computed ONCE
```

`m_ij` is occurrence-specific and depends on both dictionary coordinates and
the static relation, but it never updates any local state.  No width search.

## 6. Graph readout (inherited S0)

```text
unary moments
distance-bucket pair moments
global encoder
topology hinge (25D -> 16 -> 8)
small raw head  R -> 13 -> 13 -> 1  (GenericReader over the runtime width)
```

Runtime `unified_graph_width` is measured, never assumed (302 for this config).
The tokenizer, patch radius, topology representation, objective, loss and
training split are unchanged.

## 7. Parameter budget

```text
S0 (strict-static reference)        66,228
SDPK-v0                            74,996
  dictionary (Wq + D + U + tau)     4,657
  dictionary gate Wm                1,040
  pair encoder 112 -> 64 -> 16      8,400
  head (inherited)                  4,135
```

Audited at runtime.  `<= 82,000` preferred, `< 90,000` hard.  No module is
artificially shrunk to hit a round number.

## 8. Training protocol (inherited optimized strict-static regime)

```text
optimizer            Adam
lr                   1e-3
weight_decay         1e-5
batch_size           128
max_epochs           240          # formal candidate trains ALL 240 epochs
loss                 L1 / MAE
gradient_clip_norm   5.0
scheduler            none
early termination    none
selection            best official-valid MAE over epochs 1..240
secondary             fixed Top-5 weight soup (5 lowest-valid epochs)
```

No shadow early-stop is used as an interpretation view.

## 9. Pre-registered gates

Reference (historical A100 strict-static S0, do **not** retrain):

```text
seed0  S0 best 0.145674   S0 Top-5 soup 0.140794
seed1  S0 best 0.139389   S0 Top-5 soup 0.136423
```

### Seed-0 GO gate (primary = Top-5 soup)

```text
SDPK seed0 Top-5 soup <= 0.1328            (absolute)
S0 soup - SDPK soup      >= 0.008          (delta)
SDPK seed0 best-checkpoint improvement over S0 best >= 0.006
```

Both soup conditions and the best-checkpoint condition must hold to purchase
seed 1.  Otherwise the round stops with
`SDPK_V0_NO_STRONG_PERFORMANCE_SIGNAL`.

### Seed-1 conditional gate (only if seed-0 GO)

```text
SDPK seed1 Top-5 soup <= 0.1300
   or  S0 seed1 soup - SDPK seed1 soup >= 0.006
```

with the best-checkpoint movement in the same direction.  If
`0 < soup improvement < 0.006` the result is recorded as `DIRECTIONAL_ONLY`.
No seed 2/3 is purchased.

## 10. Cheap diagnostics (not GO gates)

Inference-only, zero-training:

```text
assignment entropy, effective atom count, argmax-used atoms,
top-8 assignment mass, tau final, dictionary coherence,
normal vs graph-mean coordinates and normal vs neutral dictionary
prediction shift (mean / max |delta prediction|)
```

A pretty entropy value is **not** a success criterion.  Absolute MAE is the
first success criterion, mechanism diagnostics the second.

## 11. Strict-static hard tests (must pass before any formal run)

```text
A  center_context == False, center_update is None,
   forward succeeds with _pool_pairs_to_centres raising
B  h_i / alpha_i / c_i bit-identical under pair-relation mutation,
   prediction changes (non-vacuous)
C  pair_encoder calls == 1, relation_encoder calls == 1,
   dictionary calls == 1 per forward
D  pair-order permutation invariance (and node relabel where applicable)
E  dictionary atoms / Wq / U / Wm / pair encoder gradients > 0 and finite
```

## 12. Durable outputs

```text
tracks/ksvd/results/zinc_static_dictionary_pair_kernel_v0/
tracks/ksvd/notes/zinc_static_dictionary_pair_kernel_v0_analysis.md
records/claims/...  records/decisions/...  tracks/ksvd/STATE.yaml
```

Every result records implementation commit, remote commit, GPU, parameter
count, best MAE, best epoch, Top-5 soup, epochs run, wall clock, peak GPU
memory, dictionary diagnostics, strict-static integrity gates and
`official_test_loaded=false`.
