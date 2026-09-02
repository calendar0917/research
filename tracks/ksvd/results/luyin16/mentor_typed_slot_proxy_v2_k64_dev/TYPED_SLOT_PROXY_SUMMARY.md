# Typed-slot reconstruction proxy v2

Protocol: `mentor-typed-slot-proxy-v2-k64-dev`

This is not the mentor's exact feature builder. It tests a natural 624-D slot-aligned object: 8×64 atom bins + 28×4 bond bins.

| view | dim | valid ROC-AUC |
|---|---:|---:|
| s | 205 | 0.750143 ± 0.008856 |
| r_raw | 624 | 0.581680 ± 0.016058 |
| r_init | 624 | 0.605261 ± 0.030623 |
| r_final | 624 | 0.639349 ± 0.022054 |
| s_r_raw | 1248 | 0.572162 ± 0.008800 |
| s_r_init | 1248 | 0.612101 ± 0.033861 |
| s_r_final | 1248 | 0.630756 ± 0.030207 |

Best by validation only: `s`.

Primary checks: `r_final-r_init`, `r_final-r_raw`, and `s_r_final-s`.
Test evaluation is disabled.
