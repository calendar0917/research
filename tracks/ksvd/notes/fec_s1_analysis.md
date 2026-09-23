# FEC-S1 — Shared Local Environment Replacement: analysis and frozen verdict

Round **FEC-S1** · study `zinc-context-gap` · protocol `fec_s1_v1`.
Pre-registration: [`fec_s1_preregistration.md`](fec_s1_preregistration.md).
Prior-artifact audit: [`fec_s1_prior_artifact_audit.md`](fec_s1_prior_artifact_audit.md).
Implementation: [`fec_s1_implementation.md`](fec_s1_implementation.md).
Results: `tracks/ksvd/results/fec_s1/` (git-ignored working evidence).

Implementation commit `341be8f`, diagnostic commit `e442c22`, remote deployed
revision `e442c22`, remote tracked worktree clean except the git-ignored
`results/fec_s1/` directory. GPU: `NVIDIA A100-SXM4-40GB` (39.4 GiB), GPU 0.
Official ZINC **test was never loaded**. Exactly **one** training run.

---

## 1. Frozen verdict

```
FEC_S1_SHARED_REPLACEMENT_STRONG
```

| quantity | value | reference | delta |
|---|---:|---|---:|
| **primary: fixed Top-5 soup official-valid MAE** | **0.130422** | S0 seed-0 soup `0.140794` (provenance only) | **−0.010372** |
| best-checkpoint official-valid MAE | 0.136783 @ epoch 238 | S0 seed-0 best `0.14567435` | −0.008892 |
| — | — | S0 seed-1 soup `0.136423` | **−0.006001** |
| — | — | SDPK-v0 soup `0.139735` | −0.009313 |

`soup = 0.130422 <= 0.1408` → **Case A**. `seed1_authorized = true`; seed 1 was
**not** executed (the pre-registration leaves it to explicit authorisation).

Frozen bands (pre-registered): A strong `<= 0.1408`, B viable `<= 0.145`,
C borderline `<= 0.148`, D failed `> 0.148`. The result is inside Case A with
margin `0.010378`.

---

## 2. Provenance

| field | value |
|---|---|
| implementation / formal-run commit | `341be8f` |
| diagnostic-revision commit | `e442c22` (mechanism stage only; no training) |
| remote commit at formal run | `341be8f` |
| remote dirty at formal run | `?? tracks/ksvd/results/fec_s1/` only (git-ignored) |
| GPU / model | `cuda` / NVIDIA A100-SXM4-40GB |
| seed | 0 |
| train / valid | 10,000 official-train / 1,000 official-valid |
| max_epochs / epochs run | 240 / 240 (no early stop) |
| best epoch | 238 |
| wall clock | 774.9 s |
| peak GPU memory | 139.7 MB |
| params | 66,170 |
| `official_test_loaded` | false |
| stop reason | fixed horizon reached; frozen soup gate evaluated |

Protocol (inherited, unmodified): Adam, lr 1e-3, wd 1e-5, batch 128, L1/MAE,
grad clip 5.0, no scheduler, patience 40, best-valid selection, fixed Top-5
weight soup over epochs `{218, 219, 223, 238, 239}`.

**Baseline guard** (same revision / execution regime, read-only, no training):
historical S0 seed-0 selection checkpoint replay valid MAE
`0.14567434728989612` vs recorded `0.14567435123870381`,
`|Δ| = 3.95e-09 <= 1e-5` → **passed**. The candidate delta is therefore not
confounded by environment/protocol drift.

---

## 3. Q1 — Memory removal

Complete. Code-level audit from the real model:

```
typed_embedding  params = 36,420    ->  attribute is None
parent_embedding params =    256    ->  attribute is None
P_lookup                = 36,676    (55.4 % of S0)
S0 total                = 66,228
remaining after deletion= 29,552
```

`state_dict()` contains **no** `typed_embedding`, `parent_embedding`,
`token_table` or `certificate_embedding` key (G1). No vocabulary-sized local
trainable memory exists anywhere in the model; no hash, no fixed code, no
dictionary, no prototype table.

## 4. Q2 — Shared function

Yes, strictly one shared function:

```
A(x_i) = Linear(146, 214) -> SiLU -> Linear(214, 24)
A(x_i) = [ e_i^shared (16) ; p_i^shared (8) ]
```

`e_i^shared` replaces `typed_embedding[token_i]` and `p_i^shared` replaces
`parent_embedding[parent_i]`, with the concatenation order and every downstream
width identical to S0. Exactly two `nn.Linear` and one `SiLU`; no LayerNorm, no
attention, no dropout, no gate, no second hidden layer.

* **G0**: the adapter input is **bit-identical** to the FEC-S0 factorized,
  train-fit-standardized 146-D `patch_cont` (rebuilt from raw primitives via
  `fec_s0_factorization.FactorizedFeatureTransform`, `max_abs = 0.0`), and the
  captured forward input equals `batch.patch_cont` bit-identically.
* **Vocabulary independence**: a second model with vocab widths `9000 / 64`
  loads the same state dict and, with `typed_token`/`parent_token` poisoned to
  `1e9`, produces a prediction **bit-identical** to the original model on the
  un-poisoned batch. Parameter count is `O(1)` in vocabulary size.

## 5. Q3 — Purity

Fully retained. `center_context=False`, `center_update=None`; forward succeeds
with `_pool_pairs_to_centres` monkeypatched to raise; `pair_encoder` and
`relation_encoder` are called **exactly once** per forward; the 24-D adapter
output is **bit-identical** under random `pair_relation` mutation
(`max_abs = 0.0`), i.e. the local environment is frozen before pair
composition: no message passing, no pair→centre, no recurrence, no relation
refresh, no attention, no context writeback.

## 6. Q4 — Capacity fairness

```
P_A(H) = 171 H + 24 ;  H* = argmin_H |P_A(H) − 36,676| = 214
adapter params      = 36,618      retention 99.84 %
relative mismatch   = 0.158 %     (gate: <= 1 %)
FEC-S1 total        = 66,170      Δ vs S0 = −58  (0.088 %)
```

`H` was computed by the audit program and never tuned. G9 passes.

**G8 — downstream architecture identity**: all 41 shared tensors are
bit-identical to S0 (`max_abs_shared_diff_vs_s0 = 0.0`) and
`unified_graph_width = 302` in both. Only the four lookup keys
(`typed_embedding.full.weight`, `typed_embedding.rare.embedding.weight`,
`typed_embedding.rare.projection.weight`, `parent_embedding.full.weight`) were
removed and replaced by the four adapter keys.

**G3 — adapter is really used**: a real mini-batch L1 backward gives finite,
strictly nonzero gradients on both adapter layers (weight-grad norms
`0.125 / 0.170`).

**G10 — official-test blocker**: a guarded `_load_zinc` raises on any `test`
split; the encoded cache reports `n_train=10000`, `n_valid=1000`,
`official_test_loaded=false`.

## 7. Q5 — Performance band

```
primary (Top-5 soup official-valid MAE) = 0.130422   ->  <= 0.1408   band A
best-checkpoint official-valid MAE      = 0.136783 @ epoch 238
soup members                            = [218, 219, 223, 238, 239]
soup member MAEs                        = 0.136943, 0.137355, 0.139387, 0.136783, 0.140019
soup member spread                      = 0.003236
train MAE at best                       = 0.098696
train minimum                           = 0.094507 @ epoch 239
valid MAE at epoch 240                  = 0.186434
```

The curve is a late-minimum, late-epoch regime: the best checkpoint arrives at
epoch 238 of 240 and the five soup members all sit in the last 22 epochs. No
manual early stop, no post-hoc extension.

Secondary sanity: `best (0.136783) > soup (0.130422)` — the normal soup
direction. There is no best/soup contradiction, so §16 of the round spec does
not engage.

## 8. Q6 — Mechanism

Report-only (`results/fec_s1/mechanism.json`).

* **Token invariance after training** (CPU, deterministic): poisoning all
  typed/parent ids to `1e9` leaves the prediction bit-identical
  (`max_abs_pred_diff = 0.0`, rerun noise floor `0.0`). The trained model is
  genuinely vocabulary-free.
* **Adapter ablation** (zero the 24-D shared output): mean `|Δpred| = 0.463155`,
  max `|Δpred| = 1.721934`, fraction shifted `> 1e-6` = **1.0**; official-valid
  MAE degrades from `0.136783` to `0.498511`. The shared channel is **not
  inert** — it is load-bearing for essentially the entire prediction.
* **Effective rank** of the 24-D shared output over 23,083 valid patches:
  participation ratio `6.204`, top singular fraction `0.312`, stable rank
  `3.208`. This is a **multi-dimensional** local channel (contrast the
  historical shared structural encoder, effective rank ≈ 1.1). Report-only; not
  a gate and not used for any rescue.

## 9. Q7 — Answer to the scientific question

The historical strict-static S0 object's dominant learned local channel was an
aliased per-key categorical memory (FEC-S0: 36,420 of 66,228 params, 55 %).
FEC-S1 shows that memory is **not a necessary local computation** of the task:
one parameter-matched, vocabulary-independent shared function of the explicit
factorized 146-D local environment reproduces the strict-static S0 performance
band and in fact improves it by `−0.010372` on the primary metric. The
per-key memory was a **capacity allocation**, not missing chemistry.

The line therefore now has its target object:

```
explicit shared local chemical environment  ->  read-only static composition  ->  y
no vocabulary memory  /  no MP  /  no recurrence  /  no context writeback
```

## 10. Allowed interpretation (and what is not claimed)

Permitted:

> A vocabulary-independent shared function of the explicit factorized local
> environment can replace S0's typed/parent categorical memory while retaining
> the strict-static performance band.

Not claimed:

> the learned 24-D representation is physically correct chemistry.

The adapter is not near rank-1 (participation ratio 6.2), so the richer
statement is also not claimed: the result is a lookup-removal /
shared-capacity result. No structural-chemistry semantics are asserted.

## 11. Scope caveats

* **One seed.** The primary gate is single-seed by pre-registration; the verdict
  is `STRONG` at the frozen band, not a claim about seed-averaged superiority.
  The margins (`−0.0104` on soup, `−0.0089` on best) exceed the historical
  single-seed soup spread between S0 seeds (`0.140794` vs `0.136423`), but the
  round deliberately does not extrapolate to a 2-seed mean.
* **Baseline provenance.** The historical S0 seed-0 soup `0.140794` has no
  persisted member states and is therefore used only as provenance; the
  in-regime read-only replay reference is the selection checkpoint
  `0.14567435`, guarded to `4e-9`.
* **No dictionary.** `FEC-D1` remains unimplemented and unauthorised this round.

## 12. Next

* `seed1_authorized = true` (`M_soup = 0.130422 <= 0.145`).
* Seed 1 is **not** auto-executed: seed 0 is recorded and the round stops. A
  paired seed 1 with the identical architecture and hyperparameters is the only
  authorised follow-up.
* A dictionary round (`FEC-D1 — Baseline-Preserving Sparse Structural Role
  Refinement`, `r_new = r_coarse + g·r_dict`, degenerating to FEC-S1 at `g = 0`)
  now has the required scientific prerequisite (FEC-S1 at least viable) and may
  be proposed under its own pre-registration.
* Frozen prior verdicts (PEC-v0 / PEC-C1 / PEC-I1 / FEC-S0) are untouched.
