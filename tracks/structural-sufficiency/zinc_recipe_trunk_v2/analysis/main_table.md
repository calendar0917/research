# zinc-recipe-trunk-v2 — measured runs

P3 seed-0 reference (same valid split): 0.079105

## 2x2 architecture x recipe

| | recipe A (Adam, 360, no warmup) | recipe B (AdamW, 600, warmup 500) |
|---|---|---|
| **P3** | 0.079105 @272 (0.201 h) | 0.093734 @464 (0.343 h) |
| **T** | 0.088070 @289 (0.206 h) | 0.091375 @426 (0.355 h) |

## All runs

| run | recipe | optim | epochs | params | best-valid | best ep | train@best(eval) | Δ vs P3 s0 | GPU-h |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| P3 x A | A | Adam | 360 | 413,505 | 0.079105 | 272 | 0.011261 | -0.000001 | 0.201 |
| P3 x B | B | adamw | 600 | 413,505 | 0.093734 | 464 | 0.008248 | +0.014629 | 0.343 |
| T x A | A | adam | 360 | 462,081 | 0.088070 | 289 | 0.006312 | +0.008965 | 0.206 |
| T x B | B | adamw | 600 | 462,081 | 0.091375 | 426 | 0.009922 | +0.012270 | 0.355 |
| M supervised | C | adam | 360 | 435,681 | 0.087468 | 281 | 0.010860 | +0.008363 | 0.252 |
| S supervised | C | adam | 360 | 468,161 | 0.087267 | 302 | 0.008844 | +0.008162 | 0.381 |
