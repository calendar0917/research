# TCCD-v3b — Correct Pre-Pair Local-State Bridge Audit

**Date:** 2026-09-21  
**Study:** `zinc-context-gap`  
**Status:** preregistered diagnostic round; formal computation blocked until this revision is committed.

## 1. Corrective scope

TCCD-v3 used the isolated 16-D B-full `e_struct` output as the purported strong-local bridge. The model audit now confirms that `e_struct` is only one input block to the B-full patch encoder. The actual B-full local data flow is:

```text
patch_cont[146]
+ patch_context[0]
+ e_patch = e_struct[16]
+ parent_embedding[8]
+ other pure-local blocks (none in the source B-full configuration)
    -> patch_encoder
    -> pre-pair patch state[64]
    -> pair_projection / pair encoder
    -> centre update
    -> global / topology readout
```

Therefore TCCD-v3 tested:

```text
e_struct[16] -> TCCD
```

and **not**:

```text
full fused pre-pair patch state -> TCCD
```

TCCD-v3 remains historical and is not deleted or numerically changed. Its supported claim is narrowed to:

> the isolated 16-D B-full structural-token channel is not a sufficient standalone local representation for TCCD.

It must not be interpreted as evidence that stronger local representations cannot help TCCD.

## 2. Frozen scientific question

If TCCD-v2's simple `714 -> 64` local encoder is replaced by the **complete B-full patch encoder output before any pair, centre-context, global, or topology operation**, while prototype, composition, relation set, and reader remain unchanged, how much can TCCD's absolute valid MAE improve?

Only the bridge tensor changes. No other TCCD module is redesigned.

## 3. Exact source checkpoint and provenance

Reuse the already validated frozen B-full/shared-structural checkpoint; do not retrain it.

- checkpoint: `tracks/ksvd/results/shared_structural_patch_encoder/soup_states/sspe_seed0_top5_soup.pt`
- checkpoint SHA-256: `785dff866fa8d952d6e9764549761a7019a4e46ffd70d625110b618f51e6a2c4`
- source model builder: `tracks/ksvd/experiments/luyin16/zinc_shared_structural_patch_encoder.py::build_candidate`
- source training commit: `0aa71c81e845fd334c8ecd2bf7904c149277c289`
- source official-valid result: seed-0 Top-5 soup MAE `0.11981802638241788`
- two-seed B-full soup reference: `0.11897220489243046`
- source model selection: official-valid-selected Top-5 soup

Because the source checkpoint used official-valid selection, this round is a **representation-capacity diagnostic bridge**, not an unbiased final model comparison. Official test remains prohibited.

## 4. Required bridge tensor

The only acceptable source tensor is the output of the full B-full `patch_encoder` module:

```text
PatchPathRecurrentPairCentreModel._encode_core
    patch = self.patch_encoder(torch.cat([... pure-local blocks ...], dim=1))
    # SELECT THIS `patch`
    source = data.pair_index[0]
    relation = ...
    q = self._pair_value(...)
```

Confirmed source dimensions from the real model construction:

- patch-encoder input width: `170`
- fused pure-local inputs: `patch_cont[146] + patch_context[0] + e_struct[16] + parent_embedding[8]`
- selected pre-pair patch-state width: `64`
- next operation after selection: pair projection / pair encoder

The selected tensor is a `patch_encoder` **OUTPUT**, not an input block. It is not `e_struct`, `e_patch`, a typed-token embedding, a parent embedding, a shell descriptor, a structural encoder output, a post-pair state, a centre-updated state, or a graph-level representation.

## 5. Pure-local proof requirements

Formal computation is blocked unless all checks pass on the selected 64-D state:

1. **Exterior invariance:** two molecules with identical root-centred radius-2 attributed induced patch and different exterior graph structure produce identical pre-pair states within tolerance.
2. **Rooted node-relabel invariance:** random consistent node relabeling of the same local patch leaves the pre-pair state unchanged within tolerance.
3. **Batching invariance:** local state from a patch/graph alone matches its state in a batch within tolerance.
4. **Pair/global independence:** intervention on pair features, other-centre/pair state path, global context, or topology features with the root local patch unchanged leaves the captured pre-pair state unchanged. Static code-path plus intervention evidence is required.
5. **Deterministic reload:** loading the same frozen checkpoint twice produces identical pre-pair state.
6. **Cache freshness:** cached pre-pair state matches fresh source-model forward within tolerance.

Any failed item is a hard STOP. No later tensor may be substituted.

## 6. Frozen extraction and cache

- B-full/shared-structural source model is fully frozen.
- Extract only to the selected `patch_encoder` output using a forward hook.
- Cache once for official train `10000` and official valid `1000`.
- Cache metadata records split, graph id, centre id, patch count, source checkpoint, SHA-256, source commit, selected layer, raw width, relation metadata reference, and `official_test_loaded: false`.
- Bridge training must never rerun the frozen B-full encoder per epoch.
- Official test is never loaded.

## 7. Matched 64-D adapter

For the selected state `h_v in R^64`, both arms use the identical single trainable adapter:

```text
z_v = W h_v + b,  z_v in R^64
```

Only one `Linear(64,64)` is allowed. No hidden layer, nonlinearity, residual bypass, normalization sweep, or arm-specific adapter is allowed.

## 8. Frozen TCCD-v2 interface

All downstream TCCD-v2 mechanics remain config-equivalent and bit-for-bit matched:

- `K=64` prototypes;
- cosine-normalized assignments;
- `tau = 0.05 + 0.95 * sigmoid(a)` with `tau_init=0.20`;
- exact normalized-Gaussian prototype initialization seed;
- exact local-entropy and balance regularizers;
- exact relation set: overlap/incidence, native bond relations, frozen global relative-position relation;
- exact `C^T R C` composition and lightweight reader;
- no pair encoder, centre-context, B-full pair machinery, global context, topology channel, higher-order moments, deep MLP, GNN, or Transformer in the bridge reader.

## 9. Formal arms

### Arm A — CorrectStrongLocal-DenseREL

```text
h_v[64] -> Linear(64,64) -> z_v
[sum_v z_v, {Z^T R_r Z}_r] -> exact TCCD-v2 lightweight reader
```

### Arm B — CorrectStrongLocal-PrototypeREL

```text
h_v[64] -> Linear(64,64) -> z_v -> cosine prototype assignment c_v
[sum_v c_v, {C^T R_r C}_r] -> exact TCCD-v2 lightweight reader
```

### Evaluation-only control

After Arm B training, randomly permute rows of `C` within each graph while keeping every relation matrix `R_r` fixed. Do not retrain. Report:

```text
Delta_comp = MAE_shuffle - MAE_real
```

## 10. Gate 0

All of the following must pass before formal training:

- selected tensor is `patch_encoder` output;
- selected tensor is not a `patch_encoder` input block;
- pure-local exterior invariance;
- node-relabel invariance;
- batching invariance;
- deterministic checkpoint reload;
- pair/global independence;
- frozen B-full parameters receive no gradient;
- adapter, prototypes, temperature, and reader receive task gradient;
- PrototypeREL reader has no raw pre-pair or continuous-`z` bypass;
- official test not loaded;
- cached tensor equals fresh source forward within tolerance.

## 11. Execution and stop rules

- official train `10000`, official valid `1000`;
- GPU1 only for all formal remote computation;
- no official test;
- no source-encoder retraining;
- one registered seed `0`;
- no second seed, K sweep, temperature sweep, reader sweep, relation sweep, architecture sweep, or rescue round;
- commit before formal run;
- remote execution uses the repository's remote runner with GPU1 only.

## 12. References and decision thresholds

Frozen references:

- TCCD-v2 best official-valid PrototypeREL: `0.287337`
- TCCD-v2 PrototypeREL Top-5 soup: `0.261988`
- TCCD-v3 isolated `e_struct` PrototypeREL: `0.982502` (wrong/partial local-channel reference only)
- canonical B-full baseline: `0.119818`

Primary metric:

```text
Delta_local = 0.287337 - MAE_CorrectStrongProto
```

Decision tree:

- **L1 — STRONG LOCAL RESCUE:** `MAE_CorrectStrongProto <= 0.20` and `Delta_local >= 0.07`.
- **L2 — MIXED LOCAL + COMPOSITION BOTTLENECK:** `0.20 < MAE <= 0.24` and `Delta_local >= 0.04`.
- **L3 — CORRECT STRONG LOCAL STILL DOES NOT RESCUE:** `MAE > 0.24` or `Delta_local < 0.04`.

Prototype value:

```text
Delta_proto = MAE_CorrectStrongProto - MAE_CorrectStrongDense
```

- `<= 0.015`: prototype preserves dense information;
- `<= 0`: strong pass;
- `> 0.03`: prototype harmful; do not tune K/tau.

Composition sanity:

```text
Delta_comp = MAE_shuffle - MAE_real
```

Requirement: `Delta_comp > 0.01`. If not, report that explicit assignment-sensitive `C^T R C` no longer contributes materially; do not alter relations.

## 13. Durable artifacts

This round must produce:

- this preregistration;
- implementation and targeted tests;
- cached 64-D pre-pair state metadata for train/valid;
- Gate 0 JSON;
- DenseREL result JSON;
- PrototypeREL result JSON;
- shuffle / vocabulary diagnostics JSON;
- analysis note;
- claim;
- decision;
- v3 clarification/amendment record;
- `STATE.yaml` pointer update;
- final commit and push.

All formal artifacts must preserve `official_test_loaded: false`, GPU1/device provenance, source checkpoint SHA-256, source commit, cache schema, split sizes, and protocol details.
