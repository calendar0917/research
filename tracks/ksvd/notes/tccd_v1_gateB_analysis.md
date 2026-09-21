# TCCD-v1 Gate B — Task Coupling

Pre-registration: `notes/tccd_v1_preregistration.md`.
Result: `results/tccd_v1/gateB_decision.json`.
Commit: `0b6ebc4`.
Device: **local CPU**, after the original remote GPU1 path was stopped for excessive runtime. Official test: never loaded.
K-SVD refit: **NO**; reused TCCD-v0 `D0`.

## Execution note

The original graph-by-graph remote implementation was too slow. A padded-batch/vectorized implementation was added as a pure engineering optimization. Targeted tests confirmed exact slow-vs-fast agreement for relation contractions and reconstruction loss. Gate B used the same split, seed, initialization, loss, sparsity, relation set, early-stop rule, and threshold; only the execution regime changed to local CPU per user instruction.

## Results

| arm | best internal-dev MAE | Top-5 soup MAE |
|---|---:|---:|
| FROZEN-D | 1.087302 | 0.912628 |
| TASK-D | 0.938099 | 0.898649 |

`lambda_rec = 8.446242`.
`delta_task = 0.149203`.

Threshold: PASS at `delta_task >= 0.010`.

**Gate B: PASS.**

Conclusion: property supervision materially improves the frozen sparse dictionary, so task coupling is supported. This does not establish sparse-dictionary uniqueness; Gate C remained required.
