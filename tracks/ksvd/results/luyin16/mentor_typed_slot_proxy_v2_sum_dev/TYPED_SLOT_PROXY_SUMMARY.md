# Typed-slot reconstruction proxy v2

Protocol: `mentor-typed-slot-proxy-v2-sum-dev`

This is not the mentor's exact feature builder. It tests a natural 624-D slot-aligned object: 8×64 atom bins + 28×4 bond bins.

| view | dim | valid ROC-AUC |
|---|---:|---:|
| s | 205 | 0.750143 ± 0.008856 |
| r_raw | 624 | 0.582079 ± 0.019937 |
| r_init | 624 | 0.530548 ± 0.020805 |
| r_final | 624 | 0.573246 ± 0.017081 |
| s_r_raw | 1248 | 0.606059 ± 0.021480 |
| s_r_init | 1248 | 0.592363 ± 0.011075 |
| s_r_final | 1248 | 0.584805 ± 0.043300 |

Best by validation only: `s`.

Primary checks: `r_final-r_init`, `r_final-r_raw`, and `s_r_final-s`.
Test evaluation is disabled.
