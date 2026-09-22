# Preregistration — SRDA-v0: Static Relational Dictionary Algebra (ZINC, strict-static)

Round name: **ZINC-static-relational-dictionary-algebra-v0** (SRDA-v0).
Study: `zinc-context-gap`.
Status: frozen **before** any formal (GPU) experiment.

## 0. Scope and budget

One performance-oriented question, one candidate architecture, no ablation grid.

```text
MAX FULL RUNS = 2
  Run 1  SRDA-v0 seed 0    (always)
  Run 2  SRDA-v0 seed 1    (only if the seed-0 GO gate fires)
official test            forbidden (never loaded)
matched control          not purchased in this round
K / R / residual / width / tau sweeps   forbidden
rescue after seed 0      forbidden
```

Explicitly **not** in scope: the closed SDPK-v0 line, the closed
residual-dictionary / `gamma`-collapse line, any message-passing variant, any
auxiliary objective, any second pair pass, any HPO.

## 1. The one question

> In a strictly static model, if the already-proven-unnecessary **typed-token**
> (16D) and **parent** (8D) identity channels are deleted outright, and the
> end-to-end dictionary assignment is made to **define the occurrence-level
> relation algebra directly** (instead of being compressed into a generic 16D
> coordinate and handed to a generic pair MLP), does absolute ZINC MAE enter a
> materially better band?

This is a performance-first question, not a mechanism-decomposition question.
It does **not** ask which component contributes how much; a matched control is
deliberately not purchased here.

## 2. Architecture contract (non-negotiable)

```text
NO message passing
NO pair/relation state -> aggregate to centre/patch -> update local state
NO relation refresh
NO second pair evaluation
NO attention / iterative refinement
```

Full forward (frozen):

```text
x_i     = [patch_cont | patch_context]                  # 146D static descriptor
z_i     = SiLU MLP(146 -> 96 -> 64)                     # independent per patch
q_i     = l2_normalize(Wq(z_i))                         # 64 -> 32
D_k     = l2_normalize(D)                               # K = 64, rank = 32
tau     = 0.05 + 0.95 * sigmoid(tau_logit)              # tau_init = 0.20
alpha_i = softmax((q_i @ D_norm^T) / tau)               # 64D
a_i     = alpha_i @ A                                   # 64 -> 32
b_i     = alpha_i @ B                                   # 64 -> 32
q_rec_i = alpha_i @ D_norm                              # 32
eps_i   = q_i - q_rec_i                                 # 32
e_i     = W_eps(eps_i)                                  # 32 -> 8

r_ij          -> Linear(23,32) -> SiLU -> rel_hidden_ij # 32
rel_gate_ij   = 1 + tanh(G(rel_hidden_ij) + bucket_embedding[bucket_ij])
rel_feat_ij   = Linear(32,16)(rel_hidden_ij)            # 16

proto_pair_ij = 0.5 * (a_i * b_j + a_j * b_i)           # 32
p_dict_ij     = proto_pair_ij * rel_gate_ij             # 32
p_eps_ij      = [e_i+e_j | |e_i-e_j| | e_i*e_j]         # 24
pair_input_ij = [p_dict_ij | p_eps_ij | rel_feat_ij]    # 72
q_ij          = SiLU MLP(72 -> 64 -> 32)                # ONCE per pair

unary  = [mean(z) | std(z) | log1p(count)]              # 129
pair   = 5 buckets x [mean(q_ij) | std(q_ij) | log1p(count)]   # 325
global = MLPBlock(62 -> 32 -> 32)                       # 32
topo   = Linear(25,16) -> ReLU -> Linear(16,8)          # 8
R      = [unary | pair | global | topo]                 # 494
y_hat  = GenericReader(R, (16, 16))                     # -> 1
```

The mathematical core is the low-rank contraction

```text
p_dict_ij[a] = sum_{k,l} alpha_i[k] alpha_j[l]
               0.5 * (A[k,a] B[l,a] + A[l,a] B[k,a]) * g_a(r_ij)
```

i.e. **dictionary prototypes themselves carry the relation-conditioned
interaction**, and the 8D residual is the only endpoint-continuous correction
path. `q_ij` is never written back into `z_i`, `alpha_i` or `eps_i`.

### 2.1 Identity channels deleted (this round's explicit design decision)

Deleted outright: typed token embedding (16D), parent embedding (8D).
**Not** replaced by corrected/historical/hash/fixed-code/parent-ID/certificate
lookups, and not replaced by an in-patch learned GNN. The local input is the
existing static continuous patch descriptor pipeline only.

Hard evidence required: `state_dict()` contains no identity key, the only
`nn.Embedding` in the model is the 5-row distance-bucket embedding, and
mutating `typed_token` / `parent_token` leaves the prediction **exactly**
unchanged.

## 3. Frozen constants (no sweep)

```text
local           146 -> 96 -> 64, SiLU, dropout 0.05
K = 64          dictionary rank = 32, tau_init = 0.20
A, B            R^(64 x 32), init std 2.0 (frozen; see below)
tensor rank R = 32
residual dim  = 8
relation trunk  Linear(23,32) + SiLU, bucket embedding 5 x 32
pair input 72   equals 32 + 24 + 16 (attested at runtime)
pair encoder    72 -> 64 -> 32, SiLU, dropout 0.05
pair pooling    5 distance buckets (1, 2, 3, 4, 5+), mean/std/log-count
unary pooling   mean/std/log-count over z
graph head      494 -> 16 -> 16 -> 1
```

`A`/`B` init standard deviation is fixed at **2.0**, chosen once by a
data-free scale argument (the prototype-pair block scales as `std^2`; 2.0 puts
it at the same magnitude as the narrow residual block at initialisation). The
measured initial block norms are recorded in `integrity_gates.json`
(`pair_block_scale_at_init`) and are not used to select anything.

### 3.1 Forbidden objectives / structures

K-SVD objective, reconstruction loss, entropy regularisation, sparsity penalty,
balance loss, top-k, orthogonality loss, contrastive loss, any raw
`P(z_i)+P(z_j) / |P(z_i)-P(z_j)| / P(z_i)*P(z_j)` endpoint bypass into the pair
kernel, any `alpha -> 16D generic coordinate` compression of the SDPK-v0 form.

## 4. Hard architecture tests (must all pass before the formal run)

```text
A  center_context is False, center_update is None; forward succeeds while the
   historical _pool_pairs_to_centres is monkeypatched to raise
B  z / q / alpha / eps bit-identical under pair-relation mutation; prediction
   changes (non-vacuous)
C  every pair is evaluated once: dictionary assignment = 1, relation trunk = 1,
   pair encoder = 1 per forward
D  no raw endpoint bypass: perturbing z inside ker(Wq) shifts z and the unary
   readout materially while the pair-kernel input stays at floating-point noise;
   the pair input is exactly the declared 72D block algebra reconstructed from
   alpha / eps / rel only
E  no identity channel: typed/parent mutation has zero effect; no identity
   parameter exists
F  invariance: pair-order permutation, endpoint swap, within-graph patch
   relabel, graph order
G  gradients: local encoder, Wq, D, A, B, relation trunk, W_eps, pair encoder
   all nonzero and finite on a real mini-batch backward
H  pooling semantics equal the inherited mean/std/log-count moments
```

## 5. Training protocol (inherited optimized A100 regime, unchanged)

```text
Adam, lr = 1e-3, weight_decay = 1e-5
batch_size = 128
max_epochs = 240, full horizon, NO early termination
scheduler = none
loss = L1 / MAE
gradient clip = 5.0
checkpoints: best official-valid MAE + fixed Top-5 weight soup
official test: never loaded
```

## 6. Gates

Reference points (already recorded; not retrained in this round):

```text
strict-static S0 seed0      best 0.145674   Top-5 soup 0.140794
strict-static S0 seed1      best 0.139389   Top-5 soup 0.136423
SDPK-v0 seed0               best 0.142193   Top-5 soup 0.139735
```

### 6.1 Seed-0 GO gate (both must hold)

```text
Top-5 soup <= 0.1340        (primary)
best valid <= 0.1385        (secondary)
```

If the gate fails:

```text
SRDA_V0_NO_STRONG_PERFORMANCE_SIGNAL
```

stop immediately. Forbidden then: `K` / `R` / residual-dim / width / `tau`
sweeps, relation redesign, rescue, seed 1, any objective term.

### 6.2 Conditional seed-1 gate (only after a seed-0 GO)

```text
strong        : SRDA seed1 soup <= 0.1335
or directional: S0_seed1_soup - SRDA_seed1_soup >= 0.0030
                AND best-checkpoint delta in the same direction
```

The round stops after seed 1 regardless of the outcome. No seed 2/3.

## 7. Cheap mechanism diagnostics (inference only, never a GO gate)

Recorded for the candidate: assignment entropy, effective atom count, argmax-used
atoms, top-8 assignment mass, `tau` final, dictionary coherence, mean
`||p_dict||` / `||p_eps||` / `||rel_feat||`, and the pair-encoder first-layer
weight norms per input block.

Three cheap inference interventions (no retraining):

```text
A  neutralise p_dict only
B  zero p_eps only
C  replace alpha_i by the graph-mean alpha
```

each reported as mean |prediction shift|, max |prediction shift| and valid MAE
after intervention. The explicit question they answer:

> does the model finally depend on the prototype algebra, or did the 8D
> residual path quietly take over the computation?

## 8. Honest limits

* Single seed (seed 1 only under the gate). Training noise on this protocol is
  of order 0.003–0.006 soup MAE, so a `+0.001` "win" would not be a signal.
* No matched control: this round asks only whether the new computational object
  moves the absolute band. Component attribution is explicitly deferred.
* SUCCESS IS DEFINED BY ABSOLUTE MAE FIRST. A healthy-looking dictionary is
  not a success criterion.

## 9. Artifacts

```text
tracks/ksvd/experiments/luyin16/zinc_static_relational_dictionary_algebra.py
tracks/ksvd/tests/test_zinc_static_relational_dictionary_algebra.py
tracks/ksvd/results/zinc_static_relational_dictionary_algebra_v0/
tracks/ksvd/notes/zinc_static_relational_dictionary_algebra_v0_analysis.md
records/claims/... , records/decisions/... , tracks/ksvd/STATE.yaml
```
