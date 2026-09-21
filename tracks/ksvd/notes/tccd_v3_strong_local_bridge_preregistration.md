# TCCD-v3 — Strong-Local Bridge Audit

**Date:** 2026-09-21  
**Study:** `zinc-context-gap`  
**Status:** preregistered diagnostic round; not a final architecture round.

## 1. Frozen question

TCCD-v2 reached official-valid best MAE `0.287337` and Top-5 soup MAE
`0.261988` with the matched `714 -> 64` local encoder, while the canonical
GPU1 strong local/structural reference reached `0.119818`. Internal and
official-valid behaviour was close enough that the primary diagnosis is a
representation/capacity ceiling rather than an obvious generalization failure.

This round asks only:

> If the TCCD-v2 prototype vocabulary and exact `C^T R C` reader receive a
> stronger frozen connectivity-sensitive pure-local representation, how much
> of the absolute gap is attributable to the simple `714 -> 64` local encoder?

The only changed object is the local representation. Prototype vocabulary,
composition, relation operators, reader family, temperature parameterization,
regularization and data protocol remain frozen to TCCD-v2.

## 2. Chosen strong local representation

The bridge reuses the canonical B-full/shared-structural encoder, but only its
pure-local patch output:

- source module: `tracks/ksvd/experiments/luyin16/structural_patch_encoder.py`
- class: `SharedStructuralPatchEncoder`
- exact tensor path: `SharedStructuralPatchEncoder.forward(data)` return value
  `e_struct`
- computation: atom/root/distance embeddings plus bond embeddings, exactly two
  edge-aware message-passing rounds over the induced radius-2 patch, followed
  by root/mean/std permutation-invariant pooling and a fusion MLP
- output width: `16`
- patch radius: `2`
- no pair state, centre-context update, graph-global feature, topology channel,
  target, or other-centre aggregation enters this tensor

The downstream bridge adds the only new mapping permitted in this round:

```text
e_struct in R^16 -> z = W e_struct + b in R^64
```

The adapter is one trainable `Linear(16, 64)` layer, shared by Dense and
Prototype arms. No hidden layer, nonlinearity, or bypass is allowed.

## 3. Checkpoint provenance

The frozen encoder is the existing canonical GPU1 strong checkpoint; it is not
retrained in this round.

- source builder:
  `tracks/ksvd/experiments/luyin16/zinc_shared_structural_patch_encoder.py::build_candidate`
- checkpoint:
  `tracks/ksvd/results/shared_structural_patch_encoder/soup_states/sspe_seed0_top5_soup.pt`
- source run record:
  `tracks/ksvd/results/shared_structural_patch_encoder/runs/sspe_seed0.json`
- source training commit recorded in run metadata:
  `0aa71c81e845fd334c8ecd2bf7904c149277c289`
- source result: seed-0 Top-5 soup official-valid MAE `0.1198180264`
- aggregate B-full reference: 2-seed soup mean `0.1189722049`
- source protocol: official train `10000`, official valid `1000`, A100,
  shared-structural B-full, fixed Top-5 soup

Important caveat: the source checkpoint used official-valid selection. Therefore
this bridge is a **representation-capacity diagnostic / upper-bound bridge
 audit**, not an unbiased final model comparison. Official test remains frozen
and is never loaded.

## 4. Data and execution protocol

- official train: `10000`
- official valid: `1000`
- official test: never opened
- local strong embeddings: precomputed once and cached per graph/centre
- formal GPU: GPU1 only through `scripts/run_remote.sh 1 ...` or detached
  `launch_remote.sh 1 ...`
- no strong encoder retraining
- no K-SVD, OMP, IHT or reconstruction loss
- no temperature/K/radius/relation/reader/message-passing/depth sweep
- no global feature, pair encoder, centre-context, topology or raw local
  embedding bypass
- no seed-1 rescue; one frozen strong checkpoint and one registered training
  seed (`seed=0`) are used for this low-cost diagnostic

Trainable bridge parameters are only:

1. the matched `16 -> 64` linear adapter;
2. prototypes for the Prototype arm;
3. the lightweight reader.

## 5. Exact arms

### Arm A — StrongLocal-DenseREL

```text
e_struct -> Linear(16,64) -> Z
[sum_v z_v, {Z^T R_r Z}_r] -> TCCD-v2 Dense-REL reader
```

### Arm B — StrongLocal-PrototypeREL

```text
e_struct -> Linear(16,64) -> z
zbar/prototype cosine -> softmax assignment C, K=64
[sum_v c_v, {C^T R_r C}_r] -> exact TCCD-v2 Prototype-REL reader
```

Frozen prototype details:

- `K=64`, latent width `64`
- L2-normalized latent and prototypes
- `tau = 0.05 + 0.95*sigmoid(a)`
- `tau_init = 0.20`
- fixed TCCD-v2 normalized-Gaussian prototype initialization seed
- exact TCCD-v2 local entropy and global balance regularizers
- exact TCCD-v2 relation set: overlap, three native bond relations, global
  relative position
- exact TCCD-v2 lightweight linear reader and Top-5 soup procedure

### Evaluation-only control

After Arm B training, evaluate the best Prototype-REL checkpoint after
randomly permuting assignment rows within each graph while keeping all relation
matrices fixed. Do not retrain the shuffled arm.

## 6. Gate 0 hard requirements

Formal training is blocked unless all checks pass:

1. **Pure locality:** two graphs with identical radius-2 rooted patch and
   different exterior structure produce identical `e_struct` within tolerance.
2. **Patch relabel invariance:** random node relabeling of the same rooted patch
   leaves `e_struct` unchanged within tolerance.
3. **Batching invariance:** embedding a patch alone or in a batch gives the same
   result.
4. **Determinism:** loading the same checkpoint twice gives identical output.
5. **Frozen encoder:** all strong-encoder parameters have
   `requires_grad=False`; no strong parameter receives a gradient.
6. **Task gradient:** gradients reach adapter, prototypes, temperature and
   reader.
7. **No bypass:** Prototype-REL reader consumes only assignment aggregates and
   fixed relation contractions; it never reads `e_struct` or `z` directly.
8. **Relation correctness:** inherited TCCD-v2 permutation and shuffle tests
   remain valid.
9. **Official-test freeze:** no test data is loaded by any stage.

Any Gate 0 failure stops the round without formal bridge training.

## 7. Decision thresholds

Primary metric:

```text
Delta_local = 0.287337 - MAE_StrongProto
```

- **L1 — STRONG LOCAL BRIDGE PASS:**
  `MAE_StrongProto <= 0.20` and `Delta_local >= 0.07`
- **L2 — MIXED BOTTLENECK:**
  `0.20 < MAE_StrongProto <= 0.24` and `Delta_local >= 0.04`
- **L3 — LOCAL BRIDGE FAIL:**
  `MAE_StrongProto > 0.24` or `Delta_local < 0.04`

Prototype value under strong local representation:

```text
Delta_proto,strong = MAE_StrongProto - MAE_StrongDense
```

- `<= 0.015`: prototype vocabulary essentially preserves predictive ability
- `<= 0`: prototype has non-positive gap / positive inductive-bias evidence
- `> 0.03`: prototype is harmful; do not tune K or temperature

Composition sanity:

```text
Delta_comp,strong = MAE_shuffle - MAE_real
```

- `> 0.01`: composition remains materially assignment-sensitive
- approximately zero: do not claim explicit composition remains necessary

## 8. Stop rules

Regardless of the result, stop after Dense, Prototype, shuffle evaluation and
vocabulary diagnostics. Do not unfreeze or retrain the strong encoder. Do not
add message passing, global topology, pair machinery, higher-order moments,
reader capacity, new relations, K/temperature changes, or rescue seeds. Any
next architecture requires a new preregistration.

## 9. Durable artifacts

Required outputs:

- this preregistration
- implementation and targeted tests
- frozen strong-local embedding cache with graph/centre/split metadata
- Gate 0 JSON
- Dense bridge result JSON
- Prototype bridge result JSON
- shuffle result JSON
- vocabulary diagnostics JSON
- analysis note and decision record
- `STATE.yaml` pointer update

Every formal artifact must include `official_test_loaded: false`, GPU1/device,
commit, source checkpoint, cache schema, split sizes, protocol and runtime.
