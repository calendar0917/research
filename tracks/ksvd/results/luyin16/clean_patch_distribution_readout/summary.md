# Clean local patch distribution readout probe

Protocol: `luyin16-clean-local-distribution-readout-probe-v1`

Official test was not encoded or evaluated.

| view | dim | valid ROC-AUC |
|---|---:|---:|
| `ta_mean` | 154 | 0.794468 ± 0.003303 |
| `ta_distribution` | 1793 | 0.817198 ± 0.002827 |
| `s_plus_ta_mean` | 359 | 0.802208 ± 0.002013 |
| `s_plus_ta_distribution` | 1998 | 0.827574 ± 0.004311 |

The distribution view changes only the graph-level readout of the same clean centre-level marginals.
The global `S` block is a control for information budget, not part of the invariant local object.
