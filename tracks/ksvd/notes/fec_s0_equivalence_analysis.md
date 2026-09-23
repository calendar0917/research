# FEC-S0 — equivalence analysis and frozen verdict

Round **FEC-S0** · study `zinc-context-gap` · protocol `fec_s0`.
Pre-registration: [`fec_s0_preregistration.md`](fec_s0_preregistration.md).
Access audit: [`fec_s0_prior_artifact_audit.md`](fec_s0_prior_artifact_audit.md).
Path factorization: [`fec_s0_path_factorization.md`](fec_s0_path_factorization.md).

Implementation: `tracks/ksvd/experiments/luyin16/fec_s0_factorization.py`.
Tests: `tracks/ksvd/tests/test_fec_s0_factorization.py` (7/7 pass).
Results: `tracks/ksvd/results/fec_s0/`.
Lineage HEAD `d4de88e`, clean worktree, CPU-only, **official test never
loaded, zero training / HPO / dictionary.**

---

## 0. What was done

The original `StrictStaticPairModel` (strict-static S0, `center_context=False`,
`residual_mode="none"`) was loaded unchanged from the historical checkpoint.
Every input tensor it reads was rebuilt from raw molecular primitives through
explicit role × primitive bindings (`FactorizedStaticEnvironmentCompositionS0`),
and the wrapped model was executed on both the historical and the factorized
inputs for the **same molecules** (official-valid 1 000, plus a 512-molecule
train sample for the descriptor stages).  Tolerances were frozen before the
run (§6 of the pre-registration).

## 1. Input identity (per-tensor, bit-level)

| tensor | width | train (512 mol) | official-valid (1 000 mol) | max abs | bit-equal |
|---|---:|---|---|---:|---:|
| `patch_cont` raw | 146 | exact | exact | 0.0 | 1.0 |
| `patch_cont` standardized | 146 | exact | exact | 0.0 | 1.0 |
| `pair_relation` (all 5 blocks) | 23 | exact | exact | 0.0 | 1.0 |
| `global_context` raw | 62 | exact | exact | 0.0 | 1.0 |
| `global_context` standardized | 62 | exact | exact | 0.0 | 1.0 |
| `topology_features` | 25 | — | exact | 0.0 | 1.0 |
| `typed_token` ids | — | — | 1000/1000 molecules | exact | 1.0 |
| `parent_token` ids | — | — | 1000/1000 molecules | exact | 1.0 |
| rebuilt typed/parent certificate bytes | — | — | 23 083/23 083 patches | exact | 1.0 |

Standardizer reproduction (full train): `patch` and `global` refit standardizers
reproduce the historical cached encodings with `max_abs = 0.0`,
`bit_equal_fraction = 1.0`; the refit vocabularies reproduce every cached token
id (`match_fraction = 1.0`).  No scaler was refit on valid.

## 2. Intermediate state identity (official-valid, 1 000 molecules)

| layer | max abs | mean abs | allclose fraction | bit-equal fraction |
|---|---:|---:|---:|---:|
| local environment `h` (`patch_encoder`) | 0.0 | 0.0 | 1.0 | 1.0 |
| pair state `q` (`pair_encoder`) | 0.0 | 0.0 | 1.0 | 1.0 |
| `relation_encoder` output | 0.0 | 0.0 | 1.0 | 1.0 |
| `global_encoder` output | 0.0 | 0.0 | 1.0 | 1.0 |
| reader representation `R` | 0.0 | 0.0 | 1.0 | 1.0 |
| prediction | 0.0 | 0.0 | 1.0 | 1.0 |

Equality is **bit-identical**, stricter than the pre-registered fallback
(`1e-6`).

## 3. Prediction inheritance (evaluation of a fixed historical checkpoint)

* historical input MAE (official-valid, CPU) = `0.1456743378872634`
* factorized input MAE (same molecules) = `0.1456743378872634`
* recorded historical best-checkpoint MAE = `0.14567435123870381`
* `|Δ|` vs recorded = `1.34e-8`

This is a **re-evaluation of an already-fixed checkpoint**, not model selection.

### Soup limitation (declared, not repaired)

The historical Top-5 soup (`0.140794`, members `[125,142,159,164,167]`) has no
persisted member states — the S0 round saved only the best selection state.
The round forbids retraining to reconstruct soup members, so the soup
equivalence is **not** performed and `0.140794` remains recorded provenance
only.  The factorized implementation is function-preserving on the available
state, which is the object the soup would also be built from.

## 4. Static purity (hard tests)

| test | result |
|---|---|
| `center_context=False`, `center_update=None` | pass |
| forward succeeds with `_pool_pairs_to_centres` monkeypatched to raise | pass |
| `h` and `h0` bit-identical under pair-relation mutation | pass (`max_abs = 0.0`) |
| pair state still changes under relation mutation (mechanism live) | pass |
| environment formation strictly before pair composition | pass (`patch` before `pair`) |
| pair/relation encoders exactly once per forward (no recurrence) | pass |
| pair-order permutation invariant | pass (`9.5e-7`) |
| relabel invariance of the factorized builder + model | pass (`2.4e-7`) |

No context writeback: pair state never enters the local environment; the only
thing pair-relation mutation changes is the graph readout, never `h`.

## 5. Verdict computation (pre-registered rule)

Classification of reachable chemistry-bearing paths:

```
FACTORIZED_SHARED : patch_cont, pair_relation, global_context
PURE_TOPOLOGY     : pair_bucket, topology_features, pair_index
EXPLICIT_BINDING_LOOKUP : typed_token, parent_token   <-- reachable local chemistry
```

Because a reachable **local** chemistry path cannot be written as a shared
role × primitive factorization (it is an aliased exact-token lookup), the
pre-registered rule selects Case C.

```
FEC_S0_LOCAL_FACTORIZATION_BLOCKED
```

Blocking tensors: `typed_token` (6785-row / 36 420-param per-key table over a
non-injective certificate) and `parent_token` (same family).

## 6. Required questions

**Q1 — What are S0's active predictive paths?**
`patch_cont` (146), `typed_token` (16-D embedding), `parent_token` (8-D),
`pair_relation` (23), `pair_bucket`, `global_context` (62), `topology_features`
(25), `pair_index`.  Everything else (`patch_context`, `structural_*`,
`attribute_*`, `center_update`, direct-token readout) is provably dormant.

**Q2 — Can every local chemistry-bearing path be written as
`structural role × chemical primitive → environment`?**
No.  `patch_cont`'s four chemistry blocks, `root_atom` and `incident` are exact
role × primitive bindings, but `typed_token`/`parent_token` are an **aliased,
non-injective exact-binding certificate** answered by a per-key learned table
(66.3 % of valid patch occurrences sit in tokens that mix distinct root atoms).
Reconstructible from primitives, but not a shared role × primitive
factorization.

**Q3 — Can pair/global chemistry be explained as explicit primitives?**
Yes.  `pair_relation`'s `path_bond_mean` and adjacent-bond one-hot are explicit
composition-level chemical relation primitives; `global_context` is
`[pure topology | atom marginal | bond marginal]` (zeroth-order composition).
Both reconstruct exactly, with no opaque bypass.

**Q4 — Is the factorized implementation numerically equivalent?**
Yes, **bit-identically** at every measured layer (inputs, standardized inputs,
`h`, `q`, relation encoder, global encoder, `R`, prediction), over the full
official-valid split.

**Q5 — Is it strictly static?**
Yes: no message passing, no recurrence, no pair→centre, no context writeback,
environment formed before pair composition, relabel-invariant.  All purity
tests pass.

**Q6 — Does it inherit historical performance under zero training?**
It reproduces the recorded best-checkpoint official-valid MAE to `1.34e-8`
(`0.14567434` vs `0.14567435`).  The Top-5 soup cannot be reproduced because
its member states were not persisted; the recorded value stays provenance only.

**Q7 — Next?** `FEC_S0_LOCAL_FACTORIZATION_BLOCKED` (Case C).

## 7. Interpretation (careful, and deliberately dual-labelled)

The **function** of strict-static S0 *is* reproducible by a wrapper that forms
every feature from raw primitives; nothing numerically changed.  What the audit
shows is narrower and more informative than "equivalence on/off":

* The mixed *handcrafted tensors* — `patch_cont`, `pair_relation`,
  `global_context` — factor **exactly** into `role × primitive` bindings plus
  explicit topology.  This confirms the PEC-I1 finding at the exact-component
  level: the coarse local chemistry was never the obstruction, and the local
  handcrafted descriptor is genuinely an environment-composition object.
* The one path that resists the factorization is a **learned categorical
  memory** over an *aliased* rooted-typed certificate.  It is the reason
  strict-static S0 is **not already** a pure
  environment → read-only-composition model: 36 420 of its 66 228 parameters
  (55 %) sit in a per-key table that is not a shared function of local roles
  and local chemistry.

Stated as a single sentence: *strict-static S0 is a static
environment→composition model in its handcrafted channels, but its dominant
learned local channel is an aliased per-configuration token memory, not a
factorized environment.*  Removing that memory is exactly what the PEC line
did; FEC-S0 now gives that removal a precise, quantitative justification.

**Conservative reading.**  Under the round's literally permissive clause
("reconstructible from structural primitive + chemical primitive + explicit
binding"), the token would count as factorized and the verdict would flip to
`FEC_S0_FUNCTIONALLY_EQUIVALENT`.  The frozen pre-registration deliberately
separates `FACTORIZED_SHARED` (shared operator) from `EXPLICIT_BINDING_LOOKUP`
(per-joint memory) and therefore returns Case C.  Both facts are recorded here
so a future round can adopt the permissive reading without re-running anything.

## 8. Provenance / reproducibility

* 7 targeted CPU tests pass (`tests/test_fec_s0_factorization.py`).
* No training, no optimizer, no HPO, no dictionary, no seed, no official test.
* One available historical checkpoint loaded; only local read-only artifacts
  consumed.
* Raw data / encodings are the historical S0 artifacts; no new cache or model
  state is produced.
