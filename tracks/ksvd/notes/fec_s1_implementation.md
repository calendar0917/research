# FEC-S1 — implementation

Round `fec_s1`. Preregistration: [`fec_s1_preregistration.md`](fec_s1_preregistration.md).
Prior-artifact audit: [`fec_s1_prior_artifact_audit.md`](fec_s1_prior_artifact_audit.md).

---

## 1. Files

| file | role |
|---|---|
| `tracks/ksvd/experiments/luyin16/zinc_patch_path_pooling.py` | new `LocalEnvironmentAdapter` + `patch_representation="shared_local_env"` (additive; all other modes untouched) |
| `tracks/ksvd/experiments/luyin16/fec_s1_shared_local_env.py` | FEC-S1 builder, correctness gates, baseline guard, single-arm run, mechanism diagnostics, frozen decision |
| `tracks/ksvd/tests/test_fec_s1_shared_local_env.py` | 9 targeted tests |

The historical S0 path is **not** modified: the new branch is only reachable
when `patch_representation="shared_local_env"`, and the existing FEC-S0
equivalence tests still pass (7/7).

---

## 2. Deleted modules

The FEC-S1 model contains no trainable `typed_embedding` and no
`parent_embedding` (`del`-ed and set to `None`). Code-level audit from the real
`state_dict`:

```
P_typed  = 36,420   (hybrid: 768 full 16-D rows + 6,017 rare rank-4 + 16x4 projection)
P_parent =    256   (32 full 8-D rows)
P_lookup = 36,676
S0 total = 66,228
remaining after deletion = 29,552
```

## 3. Shared replacement

```
A(x_i) = Linear(146, H) -> SiLU -> Linear(H, 24)
A(x_i) = [ e_i^shared (16) ; p_i^shared (8) ]
```

The two slices replace the two deleted lookups with **no change** to the patch
encoder input width (`146 + 16 + 8` unchanged), concatenation order, or any
downstream module.

## 4. Parameter matching

```
P_A(H) = 146 H + H + 24 H + 24 = 171 H + 24
H*     = argmin_H |P_A(H) - 36,676| = 214
P_A(214) = 36,618      abs error 58      relative error 0.158 %
FEC-S1 total = 66,170  (Δ −58 vs S0, 0.088 %)
```

Both are well inside the ±1 % fairness band. `H` is computed by the audit
program (`choose_local_env_hidden`), never hardcoded in the model, and is not a
performance hyperparameter.

## 5. Frozen reuse of the training protocol

`fec_s1_shared_local_env.seed0_stage` runs the **frozen**
`zinc_static_dictionary_pair.train_arm` implementation unchanged (same
`OPTIMIZED_PROTOCOL`: Adam 1e-3, wd 1e-5, batch 128, L1, grad clip 5.0, no
scheduler, 240 max epochs, patience 40, best-valid selection, fixed Top-5
soup), registering `ARMS["fec_s1"] = build_fec_s1` and temporarily redirecting
the module's output-path globals into `results/fec_s1/`. This guarantees the
formal protocol is the historical one, not a re-implementation.

## 6. Correctness gates (all required before training)

`correctness_stage` writes `results/fec_s1/correctness.json`:

| gate | implementation |
|---|---|
| G0 descriptor identity | `fec_s0_factorization.FactorizedFeatureTransform` (train-fit scaler) rebuilt from raw primitives vs cached `patch_cont`, **bit-identical** |
| G0 adapter input | forward pre-hook on the adapter captures `x_i`, compared bit-identical to `batch.patch_cont` |
| G1 no lookup params | no `typed_embedding` / `parent_embedding` / `token_table` / `certificate_embedding` keys; both attributes `None` |
| G2 token poisoning | ids set to `10**9` → prediction bit-identical |
| G2b vocabulary independence | second model with vocab `9000 / 64`, same state dict, poisoned ids → prediction bit-identical |
| G3 adapter gradient | real L1 backward → both adapter `Linear` layers finite, strictly nonzero grads |
| G4/G6 static contract | `_pool_pairs_to_centres` forced to raise; pair/relation encoders exactly once |
| G5 environment freeze | random `pair_relation` mutation leaves the 24-D adapter output bit-identical |
| G7 invariance | pair-order permutation + relabel of the raw graph through the factorized builder (≤ 1e-5) |
| G8 downstream identity | all shared tensors bit-identical to S0 (`max_abs = 0.0`); `unified_graph_width` equal |
| G9 parameter fairness | ≤ 1 % |
| G10 official test blocker | guarded `_load_zinc` raises on `test`; encoded cache audit `n_train=10000`, `n_valid=1000`, `official_test_loaded=false` |

## 7. Baseline guard

`baseline_guard_stage` loads the historical S0 seed-0 selection checkpoint and
performs a **read-only** full official-valid replay; requires
`|M_replay − 0.14567435| ≤ 1e-5`. No optimizer, no training.

## 8. Budget

Exactly one training run: FEC-S1 seed 0. No small-data screen, no seed 1, no
dictionary, no rescue.
