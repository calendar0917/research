# LCRC A_s0 post-hoc test comparison (descriptive only)

A_s0 was frozen before this read; DCC-L predictions are historical and not retrained. No model/seed/epoch choice uses these numbers.

| model | test MAE | ordinary | tail |
|---|---:|---:|---:|
| A_s0 | 0.066782 | 0.063410 | 0.107208 |
| DCC-L s0 | 0.066267 | 0.060392 | 0.136686 |
| DCC-L s1 | 0.067688 | 0.062325 | 0.131978 |
| DCC-L s2 | 0.067538 | 0.061729 | 0.137170 |
| DCC-L s3 | 0.067785 | 0.063093 | 0.124026 |

| comparison | Δ MAE (A_s0 − DCC-L) | 95% CI | A better rate |
|---|---:|---|---:|
| vs DCC-L s0 | +0.000516 | [-0.007302, +0.007178] | 0.502 |
| vs DCC-L s1 | -0.000906 | [-0.007180, +0.005060] | 0.513 |
| vs DCC-L s2 | -0.000756 | [-0.006870, +0.005134] | 0.500 |
| vs DCC-L s3 | -0.001002 | [-0.006396, +0.004330] | 0.503 |

DCC-L four-seed mean test MAE = 0.067319; A_s0 − mean = -0.000537. Single seed; CI spans zero against every seed.
