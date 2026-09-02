# Mentor-shaped ring-context proxy

Protocol: `mentor-ring-context-proxy-k64-s8-official-valid-v1`

This is a shape-aligned proxy, not an exact reproduction of the unknown upstream feature builders.

| view | dim | valid ROC-AUC |
|---|---:|---:|
| s | 69 | 0.755822 ± 0.008162 |
| t | 624 | 0.668703 ± 0.007239 |
| a | 325 | 0.683039 ± 0.009598 |
| st | 693 | 0.742439 ± 0.010928 |
| sa | 394 | 0.755005 ± 0.010830 |
| ta | 949 | 0.683986 ± 0.005583 |
| sta | 1018 | 0.737784 ± 0.014598 |
| t_init | 624 | 0.639481 ± 0.004009 |
| a_init | 325 | 0.667264 ± 0.006793 |
| sta_init | 1018 | 0.729600 ± 0.006978 |

Primary attribution: `st-s`, `sa-s`, `sta-max(st,sa)`, and `sta-sta_init`.
Official test is not evaluated.
