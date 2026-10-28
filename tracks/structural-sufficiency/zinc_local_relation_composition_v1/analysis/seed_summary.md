# LCRC-v1 seed summary and test gate

| arm | seeds | best valid per seed | mean |
|---|---:|---|---:|
| A | 0,1 | 0.076582, 0.080504 | 0.078543 |
| N | 0,1 | 0.081541, 0.078608 | 0.080075 |
| R | 0,1 | 0.077591, 0.080474 | 0.079032 |

| seed | A | R | N | DCC-L | R-N | R-A | N-A | A-DCC-L | R-DCC-L |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.076582 | 0.077591 | 0.081541 | 0.078537 | -0.003950 | +0.001009 | +0.004959 | -0.001956 | -0.000946 |
| 1 | 0.080504 | 0.080474 | 0.078608 | 0.073837 | +0.001866 | -0.000030 | -0.001895 | +0.006667 | +0.006637 |

Pre-registered test gate: consistent R<N on every seed = **False**; mean R-N = -0.001042 (<= -0.001 required); R valid mean = 0.079032 (<= 0.079 required, or clearly beat DCC-L 0.075990). Decision: **do_not_read_test**.
