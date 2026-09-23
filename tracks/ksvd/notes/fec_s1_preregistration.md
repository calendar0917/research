# FEC-S1 — pre-registration (frozen before any code change / deploy / GPU run)

Round: **FEC-S1 — Shared Local Environment Replacement**.
Study: `zinc-context-gap`. Protocol: `fec_s1`.
Prior-artifact audit: [`fec_s1_prior_artifact_audit.md`](fec_s1_prior_artifact_audit.md).

**Status: frozen.** Any deviation requires a frozen amendment note before the
affected stage runs. Official ZINC **test is never loaded**.

---

## 0. Single question

> In historical strict-static S0, delete the two per-key learned lookup
> memories `typed_embedding` + `parent_embedding` **completely** and replace
> them with one parameter-matched, vocabulary-independent, shared function that
> reads only the already-factorized 146-D local environment descriptor — does
> the strict-static S0 performance band survive?

No other change. No dictionary. No message passing. No recurrence.

---

## 1. Frozen baseline object (from FEC-S0)

```
model               = StrictStaticPairModel
center_context      = False
center_update       = None
residual_mode       = "none"
historical S0 params = 66,228
historical seed0 best valid (persisted selection ckpt) = 0.14567435123870381 @ epoch 164
FEC-S0 re-evaluated checkpoint valid MAE = 0.1456743378872634
historical seed0 recorded soup = 0.140794   (members [125,142,159,164,167])  <-- NOT replayable
historical seed1 soup = 0.136423
```

Because the Top-5 soup member states were never persisted, `0.140794` is
**provenance/reference only** and must not be treated as a replay target. The
matched reference for this round is:

* the historical **seed-0 selection checkpoint** (`0.14567435...`), replayed
  read-only in the same execution regime as the candidate (GPU baseline guard,
  §13 of the task / §10 here); and
* FEC-S1's **own** fixed Top-5 soup.

Model object fixed by the FEC-S0 audit; the checkpoint is
`results/zinc_static_dictionary_pair/states/s0_seed0_selection_state.pt`.

---

## 2. Parameter accounting (computed, not assumed)

Before any training, a code-level audit from the real `state_dict` records:

```
typed_embedding params        P_typed
parent_embedding params       P_parent
P_lookup = P_typed + P_parent
full S0 params
remaining params after deletion = full_S0 - P_lookup
```

`P_lookup` is the shared adapter's parameter budget. (Historical hint:
`P_typed ≈ 36,420`, `P_parent = 256`, so `P_lookup ≈ 36,676`; the audit is
authoritative.)

---

## 3. Deleted modules (hard)

The candidate model must contain **no** trainable:

```
typed_token  -> nn.Embedding / hybrid / factorized lookup
parent_token -> nn.Embedding
certificate  -> trainable per-key table
hash         -> trainable per-key table
```

`typed_embedding is None` and `parent_embedding is None`; no vocabulary-sized
trainable table exists anywhere in the model.

---

## 4. Shared replacement (one frozen architecture)

```
x_i = factorized standardized patch_cont_i            # exact 146-D FEC-S0 descriptor
A(x_i) = Linear(146 -> H) -> SiLU -> Linear(H -> 24)
A(x_i) = [ e_i^shared (16) ; p_i^shared (8) ]
```

* `e_i^shared` replaces `typed_embedding[typed_token_i]`;
* `p_i^shared` replaces `parent_embedding[parent_token_i]`;
* concatenation order and all downstream widths are exactly S0's.

**Forbidden inside the adapter**: LayerNorm, BatchNorm, attention, dropout,
residual stack, second hidden layer, gating, dictionary, token code. Exactly
two `nn.Linear` (both with bias) and one `SiLU`.

`A` is an independent per-root shared function. It may read **only**
`patch_cont_i`. It may **not** read other roots, pair relations, pooled graph
state, targets, token ids or vocabulary frequency. It is therefore not message
passing.

### 4.1 Adapter input provenance

The theoretical input is the FEC-S0-verified explicit descriptor: raw topology
+ atom/bond primitives → role × primitive bindings → historical train-fit
scaler → 146-D `patch_cont`. Reusing the cached `data.patch_cont` for speed is
allowed, **but** correctness gate G0 must prove

```
shared_adapter_input == FEC-S0 factorized standardized patch_cont   (bit-identical)
```

---

## 5. H is set by parameter matching only

Both `Linear` layers carry bias:

```
P_A(H) = 146 H + H + 24 H + 24 = 171 H + 24
H*     = argmin_H | P_A(H) - P_lookup |
require |P_A(H*) - P_lookup| / P_lookup <= 0.01
```

`H` is **not** a hyperparameter and may **not** be tuned for performance.
(Historical budget suggests `H* ≈ 214`; the audit program computes it.)

Recorded: released lookup params, `H*`, actual adapter params, new total params,
relative parameter mismatch.

---

## 6. Everything else in S0 is frozen

The candidate reuses, unmodified:

### 6.1 Local continuous environment path
* `data.patch_cont` (146), historical `patch_cont` standardizer, `patch_encoder`.

### 6.2 Pair composition (verbatim)
```
u_i = pair_projection(h_i)
pair_input = [ u_i+u_j | |u_i-u_j| | (u_i*u_j)*distance_gate(bucket) | pair_relation_ij ]
pair_encoder once
```

### 6.3 Pair relation (23-D, verbatim)
topology relations, `path_bond_mean`, adjacent-bond one-hot, all retained and
now read as composition-level topology/chemistry relation primitives.

### 6.4 Graph composition (verbatim)
unary moments, 5-bucket pair moments, `global_encoder`, topology hinge,
small raw reader.

---

## 7. Purity contract (hard)

```
NO message passing
NO pair -> centre
NO recurrence
NO relation refresh
NO attention
NO context writeback
```

`center_context=False`, `center_update=None`. `A(x_i)` is a per-root shared
function and does not violate this.

---

## 8. Provenance / vocabulary-independence contract (hard)

The 24-D shared channel is one shared function applied to every patch; its
parameter count is `O(1)` in the vocabulary and does **not** grow with the
number of distinct tokens.

Automated tests must show: changing typed vocabulary size, parent vocabulary
size, or `typed_token`/`parent_token` values while keeping the raw graph fixed
leaves the prediction unchanged. Strongest form: poison token ids to
out-of-range values and require a successful forward with identical prediction.

---

## 9. Correctness gates G0–G10 (must all pass before any training)

| gate | requirement |
|---|---|
| **G0** | adapter input `x_i` bit-identical to FEC-S0 factorized standardized `patch_cont` |
| **G1** | `state_dict()` contains no trainable `typed_embedding` / `parent_embedding` / `token_table` / `certificate_embedding` |
| **G2** | typed/parent ids poisoned to illegal large values → forward succeeds, prediction unchanged (bit-identical) |
| **G3** | real mini-batch backward: both adapter `Linear` layers receive finite, strictly nonzero gradients (adapter is used, not bypassed) |
| **G4** | `_pool_pairs_to_centres` monkeypatched to raise → forward still succeeds |
| **G5** | mutating `pair_relation` leaves shared local state (`e_patch`, parent slice, `h0`, `h`) bit-identical |
| **G6** | `relation_encoder` and `pair_encoder` called exactly once per forward (no recurrence / refresh) |
| **G7** | invariance: pair-order permutation, endpoint swap (where defined), patch relabel, graph order — within historical tolerance |
| **G8** | downstream architecture identity: all shared modules (excluding the replaced lookup) have identical shape, parameter name, initialization semantics and forward route vs S0 |
| **G9** | parameter fairness: total parameter mismatch ≤ 1 % |
| **G10** | any `"test"` split access raises immediately |

If any gate fails: **STOP, no training.**

---

## 10. GPU baseline guard (before the candidate run)

On the same deployed revision / execution regime:

1. load the historical S0 seed-0 **selection checkpoint**;
2. perform a **read-only** valid replay (no optimizer, no training);
3. require `|M_replay - 0.14567435| <= 1e-5`.

This is not a retrain. If it fails, STOP and localise environment/protocol
drift; the candidate delta is not interpretable under baseline drift.

---

## 11. No small-data screen

PEC-I1 fixed the fact that a 2k internal screen (`+0.041287`) has no
full-data predictive value (`+0.000110`) for this family. FEC-S1 therefore
buys **no** small-data performance screen. After G0–G10 and the baseline guard
pass, it purchases exactly **one full seed-0** run at the inherited horizon.

---

## 12. Formal training protocol (inherited, unmodified)

```
official train = 10,000        official valid = 1,000        official test = NEVER LOAD
seed                       = 0
optimizer                  = Adam
lr                         = 1e-3
weight_decay               = 1e-5
batch_size                 = 128
loss                       = L1 / MAE
gradient clip              = 5.0
scheduler                  = none
max_epochs                 = 240
patience                   = 40 (inherited; no manual early stop for a "bad" curve)
selection                  = best official-valid MAE checkpoint
soup                       = fixed Top-5 weight soup (by epoch valid MAE)
```

No LR / weight-decay / hidden-width / activation / head tuning; no post-hoc
epoch extension; no early manual termination.

---

## 13. Budget: exactly one arm

```
FEC-S1 shared replacement, seed 0          <-- the only training run
```

Not run: S0 retrain, zero-token arm, fixed-code arm, dictionary arm,
structural-GNN arm, alternate adapter, seed 1.

---

## 14. Primary and secondary metrics

Report: best official-valid MAE, best epoch, Top-5 soup MAE, soup members,
train MAE at best, train minimum, full valid curve, wall time, peak GPU memory.

**Primary decision metric: fixed Top-5 soup official-valid MAE.**
Best checkpoint is secondary.

---

## 15. Frozen decision gates

| case | condition | verdict |
|---|---|---|
| **A** | `M_soup <= 0.1408` | `FEC_S1_SHARED_REPLACEMENT_STRONG` |
| **B** | `0.1408 < M_soup <= 0.145` | `FEC_S1_SHARED_REPLACEMENT_VIABLE` |
| **C** | `0.145 < M_soup <= 0.148` | `FEC_S1_SHARED_REPLACEMENT_BORDERLINE` |
| **D** | `M_soup > 0.148` | `FEC_S1_SHARED_REPLACEMENT_FAILED` |

* Case A: a vocabulary-sized identity memory can be replaced by a pure shared
  local environment function while at least retaining the historical seed-0
  soup band. Authorises paired seed 1.
* Case B: the strict-static performance band is retained with no per-key
  memory. Authorises paired seed 1.
* Case C: STOP, `seed1_authorized = false`; no `H` change, no activation swap,
  no added depth.
* Case D: STOP. Explanation allowed: under the strict-static S0 computation
  class the minimal shared replacement cannot recover the optimisation /
  inductive-bias role of the lookup memory. **Not** allowed: "identity contains
  indispensable chemistry" (historical evidence does not support it).

---

## 16. Secondary sanity (report only, never changes the verdict)

If best and soup strongly disagree (e.g. `best <= 0.145`, `soup > 0.148`),
report curve instability, soup-member spread and best-epoch location. The
frozen soup gate still fixes the verdict. No extra epochs.

---

## 17. Mechanism diagnostics (report-only)

* **A. Token poisoning after training** — prediction must remain invariant.
* **B. Adapter ablation** — zero the 24-D shared output; report mean/max
  `|Δpred|`, MAE after zeroing, and flag `shared channel inert` if ~0.
* **C. Effective rank** of the valid-patch 24-D shared output — participation
  ratio, top singular fraction, stable rank. Compared against the historical
  near-rank-1 shared structural encoder. `Not a gate`; no rescue for low rank.

---

## 18. Allowed / disallowed interpretations

If successful, the permitted statement is:

> A vocabulary-independent shared function of the explicit factorized local
> environment can replace S0's typed/parent categorical memory while retaining
> the strict-static performance band.

Not permitted: "the learned 24-D representation is physically correct
chemistry." If the adapter is near rank-1, only the lookup-removal /
shared-capacity statement holds; no rich environment geometry is claimed.

---

## 19. Seed 1 guard

Only `M_soup <= 0.145` sets `seed1_authorized = true`, and even then seed 1 is
**not** auto-executed: this round stops after seed 0 and records, then waits for
explicit authorisation. Seed 1 (if authorised) must use the identical
architecture and hyperparameters.

---

## 20. Dictionary remains forbidden

```
NO K-SVD   NO sparse code   NO learned dictionary   NO prototype vocabulary
```

`FEC-D1` (`r_new = r_coarse + g * r_dict`, degenerating to the verified FEC-S1
baseline at `g = 0`) is a future proposal only, considered **after** FEC-S1
reaches at least the viable band.

---

## 21. Forbidden rescues (on failure)

`H` sweep, 2→3 layers, activation swap, LayerNorm, dropout tuning, fixed
identity code, hashed/corrected token, parent-only or typed-only memory, local
GNN, dictionary, recurrence, MP, extra epochs, reader enlargement. Failure is
recorded as failure.

---

## 22. Durable artifacts

```
notes/fec_s1_prior_artifact_audit.md
notes/fec_s1_preregistration.md      (this file)
notes/fec_s1_implementation.md
notes/fec_s1_analysis.md

results/fec_s1/parameter_accounting.json
results/fec_s1/correctness.json
results/fec_s1/baseline_guard.json
results/fec_s1/seed0.json
results/fec_s1/seed0_curve.csv
results/fec_s1/mechanism.json
results/fec_s1/states/
results/fec_s1/REPORT.md
results/fec_s1/DECISION.md
results/fec_s1/decision.json
```

plus a claim YAML, a decision YAML and a `STATE.yaml` update. All record:
implementation commit, formal-run commit, remote commit, dirty flag, GPU id /
model, seed, train/valid sizes, `official_test_loaded=false`, wall time, peak
memory, best, soup, stop reason.

---

## 23. Questions the final report must answer

1. **Memory removal** — were `typed` lookup, `parent` lookup and every
   vocabulary-sized local trainable memory removed completely?
2. **Shared function** — is the 24-D channel strictly `A(x_i^146)` from one
   shared function?
3. **Purity** — no MP / recurrence / pair→centre / context writeback?
4. **Capacity fairness** — was the released lookup budget reinvested within
   ≤ 1 %?
5. **Performance** — which band (`<=0.1408` / `0.1408–0.145` / `0.145–0.148` /
   `>0.148`)?
6. **Mechanism** — is the shared channel actually used after training?
7. **Next** — the unique allowed verdict and `seed1_authorized`.

---

## 24. The scientific question this round actually asks

Not "how do we delete tokens while keeping MAE". The question is:

> Is the per-key categorical memory that currently blocks shared
> environment-factorization of historical strict-static S0 a **necessary local
> computation** of the task, or only a **capacity allocation** that a
> parameter-matched shared function can take over?

If `M_FEC-S1 <= 0.145`, the line obtains its most important baseline:

```
explicit shared local chemical environment -> read-only static composition -> y
no vocabulary memory / no MP / no recurrence / no context writeback
```

Only then is a sparse-dictionary structural-role refinement scientifically
worth discussing again. If `M_FEC-S1 > 0.148`, the shared-pure performance route
is stopped cleanly, with no architecture search to rescue it.
