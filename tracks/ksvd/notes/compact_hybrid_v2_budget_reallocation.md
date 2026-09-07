# Compact-Hybrid-v2 (fixed-budget capacity reallocation)

Single-purpose question: under the same ~100k trainable-parameter budget, is
it better to spend capacity on exact token representation (rare-token rank
2 -> 4) or on the final graph prediction head (96/48 -> 64/32)?  Compact-v1
is the baseline; representation, features, split, protocol, optimizer, loss
and seed are unchanged; only the two parameter counts move.

- config: `tracks/ksvd/configs/luyin16/zinc_hierarchical_patch_relation_context_compact_hybrid_v2_budget_reallocation.yaml`
- protocol: `zinc-context-gap` (seed 0)
- run: `tracks/ksvd/runs/2026/09/07/20260907-193612-46c1a12f/` (screen, 60 epochs, test_access=blocked)
- terminal run: `tracks/ksvd/runs/2026/09/07/20260907-194818-604fa0f5/` (terminal, 60 + 56 epochs, test_access=granted)
- smoke: `20260907-193437-848f66d5` (scratch, 2 epochs, test_access=blocked)

## Architecture

- patch descriptor: 146D (radius 2, unchanged); relation descriptor: 23D; 5 buckets
- typed token embedding: hybrid — full table `768 x 16` unchanged + rank-4
  factorized table for every remaining exact token (`(V-768) x 4` + shared
  `4 -> 16` projection); no merging/hashing; parent table `32 x 8` unchanged
- patch hidden 48, pair hidden 16, center-context hidden 60 (all unchanged)
- graph head: `294 -> 64 -> 32 -> 1` (was `294 -> 96 -> 48 -> 1`); ReLU /
  LayerNorm / dropout sequence unchanged, only hidden widths move

## Parameter audit (validation-selection phase, train-only vocab 6785)

| block | v1 | v2 | delta |
|---|---:|---:|---:|
| typed_token_embedding | 24354 | 36420 | +12066 |
| parent_token_embedding | 256 | 256 | 0 |
| patch_encoder | 14192 | 14192 | 0 |
| pair_projection | 768 | 768 | 0 |
| relation_encoder | 1360 | 1360 | 0 |
| distance_gate | 80 | 80 | 0 |
| pair_encoder | 5328 | 5328 | 0 |
| center_context | 15888 | 15888 | 0 |
| global_encoder | 3136 | 3136 | 0 |
| graph_head | 33217 | 21121 | -12096 |
| **total** | **98579** | **98549** | **-30** |

Refit phase (train+valid vocab 7051): v2 = 99,613 (v1 = 99,111), both
strictly <= 100,000.  The refit delta is +502 because the rank-4 rare table
grows faster with the bigger vocabulary than the shrunken head saves; the
selection phase is a near-exact transfer (-30).

## Result (single seed, selection protocol, test blocked)

- valid MAE **0.184158** (best epoch 56 of 60; no early stop triggered)
- Compact-v1 same-protocol valid MAE 0.188216 (best epoch 52)
- absolute improvement -0.004058 on valid; lands in the "success" band
  (0.183 < valid <= 0.185)
- PROMOTION_CANDIDATE = YES (valid <= 0.185); test remains blocked for this
  run and was not touched (no_test policy, test split never loaded)

## Terminal result (train+valid refit at selected epoch 56)

- test MAE after refit **0.135362** (refit params 99,613, budget PASS)
- Compact-v1: test 0.143307 (refit 99,111); 277,053-param baseline: test 0.134610
- v2 improves test over v1 by -0.007945 and is only +0.000752 above the
  277k baseline (+0.6% relative)
- refit-phase vocab/coverage identical to v1: 7051 vocab, test coverage
  0.9899208374789116, 2397 unique types, 7050 train+valid unique types

## Invariance evidence

- train/valid feature-build statistics bit-identical to the v1 run
  (mean centres 23.1664/23.117, mean pairs 266.83/265.443, patch radius 2)
- valid typed-token coverage identical to v1: 0.9879565047870728 known
  occurrence fraction; 2379 unique types; train-only vocab 6784 + OOV
- no change in `_fit_vocabulary` / `_encode_records` / patch extraction /
  standardizers / descriptor widths (git diff touches only the parameter
  audit reporting, the head construction, and config plumbing)

## Status

Fixed-budget reallocation toward token representation improved both the
single-seed validation result and the terminal test result relative to
Compact-v1 (same 100k budget), and the terminal test is within 0.6% of the
277k baseline.  Per the study rules this is not a claim that exact identity
is the dominant module nor that the graph head is unimportant; it is limited
to: "fixed-budget capacity reallocation toward token representation improved
the current single-seed result."  Next step (if requested): promote the
terminal run to a durable record and consider whether to run more seeds.
