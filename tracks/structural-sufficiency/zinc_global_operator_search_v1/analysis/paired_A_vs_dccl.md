# Paired valid analysis: A vs DCC-L (descriptive, no test)

- n = 1000 official valid molecules; A = ['A_s0', 'A_s1']; DCC-L = ['B-R0 s0', 'B-R0 s1', 'B-R0 s2', 'B-R0 s3'].
- seed-average AE MAE: A 0.075248, DCC-L 0.075990, diff -0.000742; this equals the average of the 8 per-seed paired differences.
- pairwise signed-error corr range 0.794-0.956; absolute-error corr range 0.796-0.962.

## Same-seed pairs

| pair | MAE A | MAE D | diff | win A | signed corr | abs corr |
|---|---:|---:|---:|---:|---:|---:|
| A_s0_vs_DCCL_s0 | 0.075919 | 0.078537 | -0.002618 | 0.523 | 0.954 | 0.957 |
| A_s0_vs_DCCL_s1 | 0.075919 | 0.073837 | +0.002083 | 0.517 | 0.944 | 0.953 |
| A_s0_vs_DCCL_s2 | 0.075919 | 0.074677 | +0.001243 | 0.520 | 0.794 | 0.796 |
| A_s0_vs_DCCL_s3 | 0.075919 | 0.076911 | -0.000991 | 0.505 | 0.956 | 0.962 |
| A_s1_vs_DCCL_s0 | 0.074577 | 0.078537 | -0.003960 | 0.503 | 0.894 | 0.908 |
| A_s1_vs_DCCL_s1 | 0.074577 | 0.073837 | +0.000740 | 0.494 | 0.874 | 0.869 |
| A_s1_vs_DCCL_s2 | 0.074577 | 0.074677 | -0.000100 | 0.503 | 0.894 | 0.931 |
| A_s1_vs_DCCL_s3 | 0.074577 | 0.076911 | -0.002334 | 0.501 | 0.861 | 0.863 |

## Descriptions by fixed structural stratum

| stratum | n | MAE A | MAE D | diff A-D | 95% CI | win A | same-seed A0-D0 | A1-D1 |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| size_low_natoms | 266 | 0.080508 | 0.082259 | -0.001751 | [-0.006689, +0.003343] | 0.549 | -0.005526 | +0.005498 |
| size_mid_natoms | 353 | 0.070270 | 0.070512 | -0.000241 | [-0.003854, +0.003366] | 0.538 | +0.001063 | -0.006664 |
| size_high_natoms | 381 | 0.076188 | 0.076690 | -0.000502 | [-0.004535, +0.003666] | 0.533 | -0.003997 | +0.004278 |
| rings_0_1 | 76 | 0.083985 | 0.081456 | +0.002529 | [-0.004306, +0.009562] | 0.474 | -0.004307 | +0.010282 |
| rings_2 | 343 | 0.055618 | 0.058171 | -0.002553 | [-0.005949, +0.000899] | 0.569 | -0.001213 | -0.000870 |
| rings_3 | 368 | 0.073752 | 0.073257 | +0.000495 | [-0.003008, +0.004008] | 0.522 | +0.000080 | -0.005488 |
| rings_4plus | 213 | 0.106328 | 0.107457 | -0.001129 | [-0.008287, +0.006078] | 0.545 | -0.008938 | +0.010688 |
| r2_seen | 749 | 0.053439 | 0.054061 | -0.000621 | [-0.002568, +0.001422] | 0.543 | +0.002115 | -0.002696 |
| r2_unseen | 251 | 0.140328 | 0.141430 | -0.001102 | [-0.008362, +0.006456] | 0.526 | -0.016740 | +0.010993 |
| ordinary_y | 935 | 0.063266 | 0.064554 | -0.001288 | [-0.003401, +0.001033] | 0.545 | -0.002997 | +0.000908 |
| tail_y | 65 | 0.247605 | 0.240492 | +0.007113 | [-0.010789, +0.025901] | 0.446 | +0.002843 | -0.001671 |

## Fixed worst-set overlap

- ref_DCCL_s0_k50: overlap with A_s0 28/0.389 Jaccard; A_s1 28/0.389; DCCL_s0 50/1.000; share of A seed-average AE in this set 0.328, of DCC-L seed-average AE 0.329.
- ref_DCCL_s0_k100: overlap with A_s0 65/0.481 Jaccard; A_s1 65/0.481; DCCL_s0 100/1.000; share of A seed-average AE in this set 0.443, of DCC-L seed-average AE 0.444.
- ref_DCCL_s0_k200: overlap with A_s0 138/0.527 Jaccard; A_s1 135/0.509; DCCL_s0 200/1.000; share of A seed-average AE in this set 0.579, of DCC-L seed-average AE 0.587.
- ref_DCCL_seed_avg_AE_k50: overlap with A_s0 37/0.587 Jaccard; A_s1 37/0.587; DCCL_s0 35/0.538; share of A seed-average AE in this set 0.348, of DCC-L seed-average AE 0.350.
- ref_DCCL_seed_avg_AE_k100: overlap with A_s0 63/0.460 Jaccard; A_s1 70/0.538; DCCL_s0 78/0.639; share of A seed-average AE in this set 0.447, of DCC-L seed-average AE 0.460.
- ref_DCCL_seed_avg_AE_k200: overlap with A_s0 137/0.521 Jaccard; A_s1 149/0.594; DCCL_s0 158/0.653; share of A seed-average AE in this set 0.597, of DCC-L seed-average AE 0.612.
