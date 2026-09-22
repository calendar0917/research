# Analysis — ZINC Static Relational Dictionary Algebra (SRDA-v0)

Round: **ZINC-static-relational-dictionary-algebra-v0** (SRDA-v0).
Preregistration: `tracks/ksvd/notes/zinc_static_relational_dictionary_algebra_v0_preregistration.md`
frozen at commit `d4ef523`.
Formal run commit: `82adff72829fe2f1e0b9a8927ada8812bded2ad8`
(`d4ef523` + two audit-only bugfixes, see §1.1).
Device: remote `A100-SXM4-40GB` GPU 0 (`a100-2`).
Status: **one** seed-0 full run; seed-0 GO gate FAILS; round closed with
**1 of 2** allowed full runs and **no** seed 1.

Official ZINC **test was never loaded** in any stage
(`official_test_loaded=false` on every artifact).

## 0. What the round asked

> With the already-proven-unnecessary typed-token (16D) and parent (8D) identity
> channels deleted outright, and with the end-to-end dictionary assignment made
> to **define the occurrence-level relation algebra directly** (not compressed
> into a generic 16D coordinate feeding a generic pair MLP), does absolute
> strict-static ZINC MAE enter a materially better band?

No SDPK-v0 repair, no residual-dictionary/`gamma` question, no ablation grid, no
sweep, no rescue, no matched control.

## 1. Architecture actually implemented

Standalone module (not a `PatchPathModel` subclass), so the strict-static
contract is structural:

```text
x_i     = [patch_cont | patch_context]                  # 146D static descriptor
z_i     = SiLU MLP(146 -> 96 -> 64)                     # independent per patch
q_i     = l2_normalize(Wq(z_i))                         # 64 -> 32
D_k     = l2_normalize(D)                               # K = 64, rank = 32
tau     = 0.05 + 0.95 * sigmoid(tau_logit)              # init 0.20
alpha_i = softmax((q_i @ D_norm^T) / tau)               # 64D, end-to-end
a_i     = alpha_i @ A ;  b_i = alpha_i @ B              # A,B in R^{64x32}
q_rec_i = alpha_i @ D_norm ;  eps_i = q_i - q_rec_i     # 32D
e_i     = W_eps(eps_i)                                  # 32 -> 8

rel_hidden_ij = SiLU(Linear(23,32)(r_ij))               # 32
rel_gate_ij   = 1 + tanh(G(rel_hidden_ij) + bucket_embedding[bucket_ij])
rel_feat_ij   = Linear(32,16)(rel_hidden_ij)            # 16

proto_pair_ij = 0.5 * (a_i * b_j + a_j * b_i)           # 32
p_dict_ij     = proto_pair_ij * rel_gate_ij             # 32
p_eps_ij      = [e_i+e_j | |e_i-e_j| | e_i*e_j]         # 24
pair_input_ij = [p_dict | p_eps | rel_feat]             # 72
q_ij          = SiLU MLP(72 -> 64 -> 32)                # ONCE per pair

unary  129   pair 325 (5 buckets)   global 32   topo 8  ->  R in R^494
y_hat  = GenericReader(494 -> 16 -> 16 -> 1)
```

`p_dict_ij[a] = sum_{k,l} alpha_i[k] alpha_j[l] · 0.5·(A[k,a]B[l,a] +
A[l,a]B[k,a]) · g_a(r_ij)` — the dictionary prototypes themselves carry the
relation-conditioned interaction, and the 8D within-prototype residual is the
only endpoint-continuous correction path. There is no raw
`P(z_i)+P(z_j) / |P(z_i)-P(z_j)| / P(z_i)*P(z_j)` bypass.

### 1.1 Deleted identity channels

Deleted outright: `typed_embedding` (16D) and `parent_embedding` (8D). Not
replaced by any corrected/historical/hash/fixed-code/parent-ID/certificate
lookup, and not replaced by an in-patch learned GNN. No `typed_embedding` /
`parent_embedding` / `typed_token` / `parent_token` key exists in
`state_dict()`; the **only** `nn.Embedding` in the model is
the 5-row distance-bucket table; `identity_channel_params = 0`.

Local input is exactly the existing static descriptor pipeline:
`local_input_width = 146 = patch_cont(146) + patch_context(0)`, attested from a
real batch at runtime.

### 1.2 Preregistration vs. formal-run commit

`d4ef523` froze the architecture and gates. Two subsequent commits
(`ba38b0c`, `82adff7`) touched **only the audit code**:
the identity gate compared two independent forwards with bitwise equality and
CUDA `index_add_` scatter is not bit-deterministic (the smoke failed on the
noise floor, not on the tokens), and the within-graph patch-relabel audit built
a CPU index tensor for a CUDA batch. The gates now poison the typed/parent
indices out of range (any lookup would raise) and require the identity shift to
stay at execution-noise level. No frozen mathematical object, hyperparameter,
width, loss or protocol was changed; the CPU contract results before and after
are identical except for the tolerance definition.

### 1.3 Parameter / width audit (runtime)

| module | params |
|---|---|
| SRDA-v0 total | **49,970** |
| local encoder (146→96→64) | 20,320 |
| dictionary (`Wq` + `D` + `A` + `B` + `tau`) | 8,225 |
| residual `W_eps` (32→8) | 264 |
| relation trunk + gate + bucket + feature | 2,512 |
| pair encoder (72→64→32) | 6,752 |
| global / topology | 3,136 / 552 |
| graph head (494→16→16→1) | 8,209 |

References: strict-static S0 66,228 (with identity channels), SDPK-v0 74,996.
Deleting the identity lookups and the generic coordinate of SDPK-v0 drops the
model to 49,970 params, well inside the round's soft target (<100K).

Runtime widths (attested from real tensors, `widths_consistent=true`):
`x 146 → z 64 → alpha 64 → a 32 → eps 32 → e 8 → pair 72 → q_ij 32 →
unary 129 + pair 325 + global 32 + topo 8 = graph 494`, relation input 23.

Baseline guard: the inherited S0 seed-0 checkpoint was replayed to
`0.1456743331188918` vs recorded `0.145674` (abs diff 3.3e-07); the inherited
pipeline source hashes are unchanged
(`zinc_patch_path_pooling.py` `ee67a5f1…`, `zinc_static_dictionary_pair.py`
`9bfcbbe5…`).

## 2. Strict-static hard gates (all pass, CPU and A100)

| gate | result |
|---|---|
| A standalone model, `center_context is False`, `center_update is None` | true |
| A forward succeeds with `_pool_pairs_to_centres` raising | true |
| B `z` / `q` / `alpha` / `eps` bit-identical under pair-relation mutation | `0.0` / `0.0` / `0.0` / `0.0` |
| B prediction changes under the same mutation (non-vacuous) | `1.48e-3` |
| C dictionary / relation trunk / pair encoder calls per forward | `1 / 1 / 1` |
| D pair input is exactly the declared 72D block algebra rebuilt from `alpha`/`eps`/`rel` | exact (atol 1e-6) |
| D `ker(Wq)` perturbation: z relative shift / unary relative shift | `0.64` / `0.28` |
| D same perturbation: pair-input relative shift | `4.4e-07` (gate `< 1e-4`) |
| E typed/parent mutation: prediction shift | `4.5e-08` (relation shift `1.48e-3`) |
| E identity index poisoning, no lookup raised | true |
| E identity lookup parameters | `0` |
| F pair-order permutation / endpoint swap / patch relabel | `5.2e-08` / `0.0` / `1.5e-08` |
| G step-0 gradients (`local`, `Wq`, `D`, `A`, `B`, `rel trunk`, `W_eps`, pair encoder) | all `> 0`, all finite |
| H pooling equals the inherited mean/std/log-count moments | exact |

Step-0 gradient norms on a real 64-graph batch (CPU): local encoder 0.036,
`Wq` 0.0072, `D` 0.0023, `A` 2.5e-4, `B` 2.0e-4, `W_eps` 0.0043, relation trunk
0.0026, pair encoder 0.032, head 0.83.

Targeted CPU tests: `tracks/ksvd/tests/test_zinc_static_relational_dictionary_algebra.py`,
**16 passed** (plus the 15 SDPK tests unchanged).

## 3. Seed-0 full run (240 epochs, no early termination)

```text
device              A100-SXM4-40GB GPU 0
commit              82adff72829fe2f1e0b9a8927ada8812bded2ad8
epochs run          240 / 240   (no early stop)
best valid MAE      0.15544865403691074 @ epoch 196
Top-5 soup          0.14987310345092555
soup members        [196, 213, 223, 227, 231]
member valid MAE    [0.155449, 0.157117, 0.156549, 0.157005, 0.156342]
valid MAE @ 240     0.174678
valid MAE <= 0.15   never (0 epochs out of 240)
parameters          49,970
GPU wall clock      909.3 s
elapsed wall clock  ~923 s (22:29:06 -> 22:44:29 +08:00)
peak GPU memory     228.7 MB
official test       never loaded
```

The horizon is sufficient: the best checkpoint is epoch 196 and the last 20
epochs average `0.1727`, i.e. the run is not still descending when it stops.

## 4. Seed-0 gate — FAIL (negative direction)

```text
SRDA seed0      best 0.155449   soup 0.149873
S0 seed0        best 0.145674   soup 0.140794
SDPK-v0 seed0   best 0.142193   soup 0.139735
```

| comparison | best | soup |
|---|---|---|
| vs S0 seed0 | **−0.009775** (worse) | **−0.009079** (worse) |
| vs SDPK-v0 seed0 | **−0.013255** (worse) | **−0.010138** (worse) |

Registered gate:

```text
soup <= 0.1340  AND  best <= 0.1385
```

Neither holds — and the model never reached the baseline band at any epoch
(`valid <= 0.15` never occurred; S0/SDPK bests are 0.142–0.146). Verdict:
**`SRDA_V0_NO_STRONG_PERFORMANCE_SIGNAL`**. Per the preregistration the round
stops: no `K`/`R`/residual/width/`tau` sweep, no relation redesign, no rescue,
no seed 1, no control.

## 5. Cheap mechanism diagnostics (report only; never a GO gate)

Dictionary health at the seed-0 best checkpoint:

| diagnostic | value | (SDPK-v0 seed0 for reference) |
|---|---|---|
| mean assignment entropy (max `ln 64 = 4.159`) | 2.672 | 2.251 |
| effective atom count (`exp H`) | 14.47 | 9.49 |
| active atoms (`mean mass > 1e-3`) | 64 / 64 | 58 / 64 |
| argmax-used atoms | 51 | 43 |
| top-8 assignment mass | 0.750 | 0.824 |
| max / min mean assignment mass | 0.157 / 0.0031 | 0.086 / — |
| dictionary coherence mean / max abs | 0.172 / 0.781 | 0.229 / 0.998 |
| `tau` init → best-checkpoint / epoch-240 | 0.200 → 0.1718 / 0.1635 | 0.200 → 0.1498 |

The assignment is *less* sharp than SDPK-v0 but unambiguously alive
(never collapsed, no dead atoms).

### 5.1 Which pair block actually carries the computation?

| block | mean ‖·‖ at init | mean ‖·‖ at best ckpt | growth | pair-input energy share (init → final) | pair-encoder first-layer weight norm (init → final) |
|---|---|---|---|---|---|
| `p_dict` (prototype algebra) | 0.611 | **2.256** | ×3.69 | 25.2% → **81.8%** | 3.08 → **7.04** (×2.29) |
| `p_eps` (8D residual) | 0.621 | 0.659 | ×1.06 | 25.9% → 7.0% | 2.64 → 4.53 (×1.71) |
| `rel_feat` (static relation) | 0.852 | 0.835 | ×0.98 | 48.9% → 11.2% | 2.17 → 3.12 (×1.43) |

(init values are from the frozen `integrity_gates.json → pair_block_scale_at_init`
with the preregistered `std(A)=std(B)=2.0`.)

### 5.2 Cheap inference interventions (best checkpoint, full official valid)

| intervention | mean \|Δpred\| | max \|Δpred\| | valid MAE after |
|---|---|---|---|
| A neutralise `p_dict` only | 1.2018 | 2.8373 | 1.2319 |
| B zero `p_eps` only | 0.2462 | 0.5007 | 0.2861 |
| C replace `alpha_i` by graph-mean `alpha` | 0.6125 | 2.5659 | 0.6436 |
| (normal) | — | — | 0.155449 |

**Answer to the round's mechanism question:** the prototype algebra is the
primary driver, and the 8D residual did **not** take over. The `p_dict` block
grew 3.7× while `p_eps` stayed flat, it holds ~82% of the pair-input energy, the
pair encoder allocates it the largest first-layer weight norm, and neutralising
it destroys the prediction (MAE 0.155 → 1.232, i.e. the output collapses to the
target mean). The 8D residual is a genuine but minor correction
(`zero_eps` → MAE 0.286, `mean_alpha` → 0.644).

## 6. Interpretation

* **This round does not fail because the new object is bypassed.** Unlike the
  earlier residual-dictionary round (`gamma → 4e-4`, 1e-5 ablation) this model
  demonstrably computes with the prototype-pair algebra: it dominates the pair
  input energy, dominates the pair encoder's first-layer weights, and its
  neutralisation is catastrophic for the prediction.
* **It fails because that object does not carry enough task-relevant
  information to beat the strict-static baseline.** SRDA is `0.0091` soup MAE
  *worse* than S0 seed 0 and `0.0101` worse than SDPK-v0 seed 0, and it never
  entered the `0.14x` band at all.
* The most plausible reading (stated as a hypothesis, not a tested claim): the
  deleted 24D of identity/typed information plus the removed raw
  `z`-projection pair blocks (`u_i+u_j`, `|u_i-u_j|`, `u_i*u_j`) removed more
  usable pairwise information than the prototype algebra added. In SDPK-v0 the
  dictionary was *added on top of* the raw pair interaction without removing
  it; here it replaces it. The single-variable comparison SRDA-vs-SDPK
  therefore bundles two changes (delete identity channels AND replace the raw
  pair bypass by prototype algebra), and this round cannot attribute the loss
  between them — that attribution would require the matched control this round
  deliberately did not buy.
* A second honest observation: `p_dict`'s dominance over `p_eps` is partly a
  consequence of the preregistered factor-init scale (`std = 2.0`), which was
  chosen to keep the three blocks comparable at initialisation. It is *not*
  evidence that the residual would have taken over under a different init; it
  is the measured outcome of the frozen choice.
* No rescue was attempted and none is authorized. Absolute MAE is the first
  success criterion; a healthy, load-bearing dictionary is not a success.

## 7. Direct answers to the round's questions

1. **Is the typed 16D + parent 8D channel fully deleted?** Yes.
   `identity_channel_params = 0`, no `typed`/`parent` key exists in
   `state_dict()`, the only `nn.Embedding` is the 5-row bucket table, and
   poisoning both token tensors out of range raises nothing while the
   prediction shifts by `4.5e-08` (execution noise).
2. **Does the local state come only from the existing static 146D descriptor?**
   Yes: `x_i = [patch_cont | patch_context]`, runtime-attested width 146.
3. **Is the no-message-passing contract strictly satisfied?** Yes. Standalone
   module, `center_context=False`, `center_update=None`, forward succeeds with
   `_pool_pairs_to_centres` raising, and `z`/`q`/`alpha`/`eps` are bit-identical
   under pair-relation mutation.
4. **Is there any raw `z_i`/`z_j` bypass in the pair kernel?** No. A `ker(Wq)`
   perturbation moves `z` by 64% relative and the unary readout by 28% while
   the pair input moves by `4.4e-07` relative; the 72D pair input is rebuilt
   exactly from `alpha`/`eps`/`rel`.
5. **Does the dictionary assignment directly enter a prototype-pair tensor
   contraction?** Yes: `proto_pair = 0.5·(a_i·b_j + a_j·b_i)` with
   `a = alpha@A`, `b = alpha@B`, relation-gated into `p_dict`.
6. **Total parameters?** 49,970.
7. **Runtime unary / pair / graph widths?** 129 / 325 (5 buckets × 65) / 494;
   pair input 72; relation input 23.
8. **Seed-0 best / Top-5 soup?** best `0.15544865403691074 @196`; soup
   `0.14987310345092555`.
9. **Improvement vs S0 seed 0 / SDPK-v0 seed 0?** `−0.009775 / −0.013255`
   (best) and `−0.009079 / −0.010138` (soup): worse on both.
10. **Seed-0 GO gate?** Failed in the negative direction.
11. **Is seed 1 purchased?** No.
12. **Seed-1 numbers?** Not run (not authorized).
13. **Is `p_dict` really used?** Yes, and it is the dominant pair block:
    82% of pair-input energy, largest first-layer weight norm (7.04, ×2.29),
    neutralisation moves predictions by mean 1.202 / max 2.837 and raises valid
    MAE from 0.155 to 1.232.
14. **Did the 8D `p_eps` residual become the main bypass?** No. It stayed at
    0.66 norm (7% of pair-input energy); zeroing it costs 0.13 MAE.
15. **Are the dictionary assignments healthy?** Yes: 64/64 active atoms, 51
    argmax-used, effective count 14.5, top-8 mass 0.750, coherence mean 0.172,
    `tau` 0.200 → 0.1718, no collapse.
16. **How many full runs were used?** **1** of the allowed 2.
17. **GPU / elapsed wall clock?** GPU 909.3 s; elapsed ≈ 923 s; peak 228.7 MB.
18. **Was the official test ever loaded?** No, never, in any stage.
19. **Did the performance enter a genuinely new strict-static band?** No. It is
    `0.0091` soup MAE *above* the S0 seed-0 band and never reached `0.15` on
    valid, which is outside the `0.13x–0.14x` region in the wrong direction.

## 8. Verdict

```text
SRDA_V0_NO_STRONG_PERFORMANCE_SIGNAL
```

The prototype-algebra computation is real, dominant and load-bearing, and the
identity channels are provably gone — but the absolute MAE is ~0.009 worse than
the strict-static baseline, so the architecture is **not** promoted. STOP: one
full run, no seed 1, no sweep, no rescue, no control. Official test unread.

## 9. Artifacts

* `results/zinc_static_relational_dictionary_algebra_v0/parameter_audit.json`
* `results/zinc_static_relational_dictionary_algebra_v0/integrity_gates.json`
* `results/zinc_static_relational_dictionary_algebra_v0/baseline_guard.json`
* `results/zinc_static_relational_dictionary_algebra_v0/runs/srda_seed0.json`
* `results/zinc_static_relational_dictionary_algebra_v0/curves/srda_seed0_curve.csv`
* `results/zinc_static_relational_dictionary_algebra_v0/states/srda_seed0_best_state.pt`
* `results/zinc_static_relational_dictionary_algebra_v0/states/srda_seed0_soup_state.pt`
* `results/zinc_static_relational_dictionary_algebra_v0/analysis.json`
* `results/zinc_static_relational_dictionary_algebra_v0/RESULTS_SUMMARY.md`
