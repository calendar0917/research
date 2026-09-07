# Compact-Hybrid-v1 (parameter-efficiency check)

Single-purpose question: does the exact-patch + hierarchical relation pipeline
retain its ZINC quality when the model is compressed from 277,053 to ~100k
trainable parameters?  Representation, features, split, protocol, optimizer
and loss are unchanged; only the parameterization changes.

- config: `tracks/ksvd/configs/luyin16/zinc_hierarchical_patch_relation_context_compact_hybrid_v1.yaml`
- protocol: `zinc-context-gap` (terminal, seed 0)
- run: `tracks/ksvd/runs/2026/09/07/20260907-190926-031c1448/` (terminal, 60 epochs)
- smoke: `20260907-190617-a06591de` (scratch, 2 epochs), `20260907-190745-7e73ad82` (terminal, 2 epochs)

## Architecture

- patch descriptor: 146D (radius 2, unchanged); relation descriptor: 23D
- typed token embedding: hybrid — full table `768 x 16` (OOV + top-767
  train-frequent tokens) + rank-2 factorized table for every remaining exact
  token (`(V-768) x 2` + shared `2 -> 16` projection); no merging/hashing
- parent token embedding: full `32 x 8` (31 known + OOV)
- patch hidden 48, pair hidden 16, center-context hidden 60, head 262 -> 96 -> 48 -> 1

## Parameter audit (validation-selection phase, train-only vocab 6785)

| block | params | share |
|---|---:|---:|
| typed_token_embedding | 24354 | 24.71% |
| parent_token_embedding | 256 | 0.26% |
| patch_encoder | 14192 | 14.40% |
| pair_projection | 768 | 0.78% |
| relation_encoder | 1360 | 1.38% |
| distance_gate | 80 | 0.08% |
| pair_encoder | 5328 | 5.40% |
| center_context | 15888 | 16.12% |
| global_encoder | 3136 | 3.18% |
| graph_head | 33217 | 33.70% |
| **total** | **98579** | 100.00% |

Refit phase (train+valid vocab 7051): 99,111.  Both phases pass the
`expected_max_trainable_params: 100000` budget assertion (fail-fast).

## Result (single seed, terminal protocol)

- valid MAE 0.188216 (best epoch 52)
- test MAE (train+valid refit) **0.143307**
- baseline (277,053 params): test MAE 0.134610
- parameters reduced 277,053 -> 98,579 (-64.4%)

Verdict per the compact road map: `test <= 0.145` => the compressed
parameterization substantially retains the 277k-quality result
(relative degradation +6.5% MAE).

## Deviation from the first-version spec

The first-version spec used `center_context_hidden: 64`.  With that value the
selection-phase model is 99,635 (<=100k) but the protocol-mandated
train+valid refit model is 100,167 (+167 over budget).  Per the priority list
(`slight center-context hidden reduction` before touching the 768 full-token
boundary) the delivered config uses `center_context_hidden: 60`, keeping both
phases strictly <=100k while preserving the full center-context feature
pipeline.  The 64-variant numbers are 99,635 / 100,167 within 0.17% of the
budget and remain a valid alternative if the refit-phase count is not counted
against the budget.
