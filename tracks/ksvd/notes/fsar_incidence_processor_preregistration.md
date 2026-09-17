# Pre-registration — FSAR-C1: Parameter-Matched Incidence Processor

Round name: **FSAR-C1**.  Branch: `exp/fsar-incidence-processor-zinc`.
Written **before** any formal ZINC run.  Official ZINC test is never loaded.
Local code → remote compute → local analysis, per the `remote-research-runner`
skill.

---

## 1. Motivation and the single core question

The FSAR-R2-AR0 / AR0-EDGE diagnostics established, on a fixed radius-2
explicit pure-topology coordinate, that

* the node structural-role ↔ atom-type assignment `B_V` carries a stable,
  control-beating increment (3 paired seeds, `ASSIGNMENT_SUPPORTED`),
* the edge structural-role ↔ bond-type assignment `B_E` carries an additional
  stable increment beyond `B_V` (2 paired seeds, `EDGE_ASSIGNMENT_SUPPORTED`),

but both rounds used **tiny linear-residual readouts** (24,797 / 25,317 params)
whose absolute MAE (~0.44–0.47) is far from the strong backbones (B-Full
84,495 params, seed0 Top-5 soup valid 0.119818; B-Bag 0.127382).

Both assignments are currently **collapsed into a graph-level statistic before
any nonlinear processing** (`C_V`, `C_E`, `P_E` are `[65,28]`/`[130,4]`
matrices that are immediately flattened and linearly read).  The strong
backbones instead keep persistent per-node / per-patch objects and process them.

**Core question (the only question this round asks):**

> With the *same* S / A / B primitives, no new structural information, and a
> parameter budget matched to the ~80–100k strong reference, does keeping
> **persistent node objects and persistent edge objects** and running a real
> **node → incident-edge → node incidence processor** recover a substantial
> part of the gap between the current factorized diagnostic and the strong
> mixed backbone?

This is a **processor hypothesis**, not a feature-engineering experiment.

---

## 2. What is frozen / reused verbatim

* dataset, splits, and feature extraction: `data/ZINC`, official train/valid
  only, via `zinc_fsar_r2_ar0_edge.build_edge_datasets()` (already validated and
  cached as `fsar_r2_ar0_edge_features_v1`); the official test split is never
  loaded.
* node structural input: `phi_v ∈ R^65`, the frozen radius-2 pure-topology
  coordinate of `fsar_r2_ar0.py`.
* node attribute: `q_v = onehot(atom_type) ∈ R^28`.
* edge structural input: `psi_e = [phi_u + phi_v, |phi_u - phi_v|] ∈ R^130`
  (endpoint-swap invariant, chemistry-free), exactly the AR0-EDGE role.
* edge attribute: `r_e = onehot(bond_type) ∈ R^4`.
* whole-graph attribute marginal `A(G) ∈ R^64` (pure attribute multiset;
  supplied to the readout so that no primitive available to AR0 is removed).
* training protocol: `OPTIMIZED_PROTOCOL` — Adam, lr 1e-3, wd 1e-5, batch 128,
  L1, grad clip 5.0, no scheduler, max 240 epochs, patience 40, best
  official-valid checkpoint, fixed equal-weight Top-5 soup.

**Added structural information: none.**  The forbidden list is explicit:
no RRWP, no LapPE, no C4/C5 counts, no homomorphism counts, no attention
positional encoding, no radius > 2, no learned motif vocabulary, no historical
typed-token identity lookup, no pair-state / triangle machinery, no new
structural feature family.

---

## 3. Architecture C1

Hidden width `d = 48`; shared recurrent processor hidden `2d = 96`; 4 rounds;
scalar residual scales initialised to `0.05` (small but non-zero).

### 3.1 Persistent object initialisation

```
S_v = f_s(phi_v)            A_v = f_a(q_v)
B_v = W_b( (U_s S_v) ⊙ (U_a A_v) )
h_v^0 = S_v + A_v + B_v

S_e = f_se(psi_e)           A_e = f_ae(r_e)
B_e = W_be( (U_se S_e) ⊙ (U_ae A_e) )
g_e^0 = S_e + A_e + B_e
```

`f_s`, `f_a`, `f_se`, `f_ae` are biased `Linear` (`65→48`, `28→48`, `130→48`,
`4→48`).  `U_s`, `U_a`, `U_se`, `U_ae` are bias-free `48→48`; `W_b`, `W_be` are
biased `48→48`.  `S` and `A` sources are strictly separate; `B` depends on the
real role ↔ attribute pairing.  No graph-level sum happens here.

### 3.2 Shared recurrent incidence processor (one round, shared over 4 rounds)

Edge update, for `e = (u, v)` (symmetric in the endpoints):

```
x_e = [ h_u + h_v , |h_u - h_v| , g_e ]                # R^144
g_e ← g_e + α_e · W_e2( SiLU( W_e1( LN_3d(x_e) ) ) )   # pre-LN, SiLU, residual
```

Node update, for node `v` with incident edges `e=(v,w)` and neighbour `w`:

```
msg_{e→v} = W_m2( SiLU( W_m1( [ g_e , h_w ] ) ) )
m_v       = mean_{e incident to v} msg_{e→v}          # degree-normalised
h_v ← h_v + α_v · W_n2( SiLU( W_n1( LN_2d( [ h_v , m_v ] ) ) ) )
```

* `α_e`, `α_v` are scalar `nn.Parameter`s, init `0.05`, shared across rounds.
* incidence aggregation is **dense matmul** against the endpoint incidence
  matrices (no atomic `index_add_`), so it stays bit-reproducible under
  `torch.use_deterministic_algorithms(True)`.
* the edge update input is exactly symmetric under endpoint swap; the node
  update is standard permutation-equivariant message passing on the true graph
  incidence.  No all-pairs attention; no pair state; no triangles.

### 3.3 Readout

Population moments of the persistent objects (no global attention token, no
pair pooling):

```
read = [ mean_B(h), std_B(h), mean_B(g), std_B(g), A(G), log1p(n), log1p(m) ]
ŷ    = MLP( read )        # 258 → 64 → 1, SiLU
```

`mean_B` / `std_B` are per-graph, computed with the dense graph indicator.

### 3.4 Parameter accounting (computed by the `params` stage, not asserted here)

Planned module breakdown, to be printed and stored at
`results/fsar_incidence/parameter_accounting.json`:

| module | content |
|---|---|
| node initializer | `f_s`, `f_a` |
| edge initializer | `f_se`, `f_ae` |
| binding | `U_s`, `U_a`, `W_b`, `U_se`, `U_ae`, `W_be` |
| incidence processor | `W_e1/W_e2` + `W_m1/W_m2` + `W_n1/W_n2` + 2 LayerNorms + 2 scales |
| readout | `258→64→1` |

Target: **85k–95k total, exact count reported**.  No hidden padding to hit a
number; `d=48` / `2d=96` / 4 shared rounds are structurally chosen and the
resulting count is reported as-is and compared to B-Full 84,495 / B-Bag 84,511 /
cell A 85,763 / AR0-edge BVE 25,317.

---

## 4. Optimizer / anti-collapse protocol

* Everything uses the canonical `OPTIMIZED_PROTOCOL` except one documented,
  minimal deviation: parameters in the **binding modules** (`U_s`, `U_a`, `W_b`,
  `U_se`, `U_ae`, `W_be`), the two **LayerNorm** parameter pairs and the two
  **residual scalar scales** are placed in a `weight_decay = 0` group.  All
  other parameters keep `wd = 1e-5`.  Rationale: the repo has three independent
  replications of branch annihilation by Adam + L2 on parameters whose task
  gradient vanishes (Z1/ASB, BCE, EOR); this is the single repair already
  recorded in those decisions, applied here at registration time rather than
  after a failure.
* No branch / output projection is zero-initialised.  Residual scales are
  `0.05`, not `0`.  No narrow ReLU branch, no hard gate, no sigmoid gate.

---

## 5. Control C0 (parameter-exact incidence-free bag processor)

`C0` uses the **same object initialisation and the same module shapes** as
`C1` (hence an identical parameter count), but destroys the real node–edge
incidence:

```
node_ctx = segment_mean(h)                 # per-graph [B, d]
edge_ctx = node_ctx[graph(e)]              # every edge gets its graph's global node mean
x_e      = [ node_ctx_e , node_ctx_e , g_e ]
g_e     ← g_e + α_e · W_e2( SiLU( W_e1( LN(x_e) ) ) )

msg_e    = W_m2( SiLU( W_m1( [ g_e , edge_ctx ] ) ) )
m_v      = segment_mean_{edges of graph(v)}(msg_e)      # every node in a graph gets the same m
h_v     ← h_v + α_v · W_n2( SiLU( W_n1( LN( [h_v, m_v] ) ) ) )
```

This is permutation-invariant, performs no use of which node an edge connects,
and keeps every parameter alive, so it is a **parameter-exact, function-class
matched** control for "incidence composition vs premature global aggregation".
(Deviation from the suggested "independent per-object MLP" control is
deliberate and is a *tighter* match: an independently processed bag with a
different width would confound capacity with the control.  Documented here.)

---

## 6. Experimental economy — seed0 gate

Per the round instructions, do **not** run multiple seeds up front.

* **Stage A (local, mandatory):** targeted unit tests; tiny CPU/GPU
  forward-backward smoke; shape checks; endpoint-swap / permutation
  invariances; parameter count; and a check that after one optimizer step every
  intended branch has non-zero gradient and non-zero update.
* **Stage B (remote, mandatory):** shortest GPU smoke on the A100 — CUDA
  correctness, memory, no NaN, node/edge binding branch alive, incidence
  processor branch alive.  No performance conclusion from smoke.
* **Stage C:** exactly **one** formal run, `C1` seed 0, canonical protocol, no
  hyperparameter sweep, no official test.

### Pre-registered seed0 gate (valid Top-5 soup of C1 seed 0)

| C1 seed0 soup | call | action |
|---|---|---|
| `> 0.20` | **STOP** | no more seeds; analyse branch liveness / underfitting / bottleneck first |
| `0.14 – 0.20` | **INSPECT** | analyse mechanism + learning curve; report; decide on a 2nd seed later |
| `≤ 0.14` | **seed1-worthy** | report seed0; recommend (do not auto-run) seed 1 |
| `≤ 0.12` | **STRONG** | check reproducibility + mechanism first, then paired seed confirmation |

C0 is run at seed 0 **only if** C1 is not a clear STOP (`> 0.20`); otherwise the
budget is not spent.  C0 is a mechanism control, never a promotion candidate.

---

## 7. Mechanism diagnostics (recorded at early / middle / final checkpoints)

* node binding `B_v`: output std, parameter norm, gradient norm
* edge binding `B_e`: output std, parameter norm, gradient norm
* edge update: residual std, gradient norm
* node update: residual std, gradient norm
* node representation: mean per-dim std, effective rank
* edge representation: mean per-dim std, effective rank

If a branch is exactly zero / near-zero at the end, it is labelled
`mechanism/optimization failure` and **not** "incidence composition is useless".

Gradient norms are measured on a fixed diagnostic batch (first 4,096 valid
nodes / all valid edges of 32 batches) at epochs 1, 20, 40, … and the final
epoch, via one extra backward on the current (training-mode) model.  Cheap
stats are recorded every epoch.

---

## 8. Explicitly out of scope / forbidden

official ZINC test; 3+ seeds up front; hyperparameter / architecture sweep;
pair-state model; triangle updates; transformer / global attention; new
structural feature family; radius > 2; C4/C5/RRWP/homomorphism counts;
parameter budget > 200k; restoring any historical mixed-feature bypass;
editing tracked files directly on the remote checkout; ad-hoc hyperparameter
tuning after a bad result.

## 9. Decision rule for the durable note

The note reports the seed0 gate call with the numbers above and the most
conservative interpretation.  No SOTA, no general-binding, no
"incidence composition works" claim beyond what the seed0 numbers and the
C0 comparison license.  If a branch is dead, the honest label is
"not cleanly tested".

## 10. Revisit-if

* C1 seed0 falls in (0.14, 0.20) and the learning curve shows the processor is
  clearly alive but capacity-limited — a pre-registered depth/width round on
  the *same* primitives, not a feature change.
* C1 seed0 ≤ 0.14 — paired seed confirmation of the identical architecture.
* C1 seed0 > 0.20 — evidence that radius-2 pure-topology + incidence processing
  at ~90k is itself insufficient; the next move is a *different* processor
  hypothesis, not a patch of this one.
