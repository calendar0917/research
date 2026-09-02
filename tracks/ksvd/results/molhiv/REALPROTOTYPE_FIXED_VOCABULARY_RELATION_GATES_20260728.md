# Fixed-vocabulary task-aware relation gates (8k development)

Date: 2026-07-28

## Question

Can task matching be stabilized by keeping the broad deterministic real-patch vocabulary fixed and using nested outer-fit-only supervision only as a smooth prototype gate for exact-distance 1+2 relation composition?

## Protocol

- Data: 8,000-graph subset; 6,400 official-train graphs only.
- Validation: three outer Bemis-Murcko scaffold folds.
- Vocabulary: fixed 32-prototype `farthest` and 32-prototype `scaffold_facility` banks per outer fold.
- Gate: source-clean nested inner-scaffold OOF balanced-log-loss ablation rank, transformed as `(0.5 + importance) / mean(0.5 + importance)`.
- Base: frozen probability average of the two broad occurrence predictors.
- Relation: top-3 positive-cosine assignment, compact exact distance 1+2, both families concatenated (280 dimensions).
- Residual: zero-initialized bounded linear head, cap fixed at 0.3125.
- Controls: uniform gate, shuffled-label gate, and matched node-assignment shuffling.
- Official valid evaluations: **0**.
- Official test evaluations: **0**.

## Outer-fold AUC

| Gate / relation | Fold 0 | Fold 1 | Fold 2 | Mean | Gain over frozen base |
|---|---:|---:|---:|---:|---:|
| Frozen broad occurrence base | 0.766886 | 0.699399 | 0.800924 | **0.755736** | — |
| Uniform / real | 0.768618 | 0.703381 | 0.808833 | **0.760277** | +0.004541 |
| Uniform / assignment-shuffled | 0.768164 | 0.699660 | 0.804557 | 0.757461 | +0.001724 |
| Task gate / real | 0.767809 | 0.703119 | 0.808177 | **0.759702** | +0.003965 |
| Task gate / assignment-shuffled | 0.767489 | 0.700509 | 0.804914 | 0.757637 | +0.001901 |
| Shuffled-label gate / real | 0.769360 | 0.704414 | 0.808750 | **0.760842** | +0.005105 |
| Shuffled-label gate / assignment-shuffled | 0.768329 | 0.700573 | 0.804582 | 0.757828 | +0.002092 |

## Matched comparisons

| Comparison | Fold values | Mean | Wins |
|---|---|---:|---:|
| Task real − uniform real | -0.000809 / -0.000262 / -0.000656 | **-0.000576** | 0/3 |
| Task real − shuffled-label real | -0.001552 / -0.001294 / -0.000574 | **-0.001140** | 0/3 |
| Task real − task assignment-shuffled | 0.000320 / 0.002610 / 0.003263 | **+0.002064** | 3/3 |

## Decision

**Promotion gate: FAIL.**

The smooth task-aware gate is below the matched uniform gate in all three folds and below the shuffled-label gate in all three folds. It does preserve assignment-specific local signal—the real task-gated relation beats its own assignment-shuffled control in all three folds—but that signal is not specifically improved by the true-label gate.

Therefore:

1. Do **not** run official valid/test.
2. Stop treating prototype-wise scalar reweighting as the main stabilization mechanism.
3. Retain the stronger mechanism result: fixed broad prototypes plus exact local relation composition consistently helps the occurrence base; the supervised scalar gate does not explain that gain.
4. The next attempt should change the supervised object rather than gate strength or residual cap—for example, learn a low-rank **pairwise relation metric** under strict nested/OOF constraints, or move supervision to graph-level mixture calibration while keeping the local vocabulary and relation tensors label-free.
