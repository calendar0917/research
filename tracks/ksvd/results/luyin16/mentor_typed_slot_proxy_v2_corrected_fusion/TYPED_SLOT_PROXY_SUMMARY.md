# Typed-slot reconstruction proxy v2

Protocol: `mentor-typed-slot-proxy-v2-official-valid`

This is not the mentor's exact feature builder. It tests a natural 624-D slot-aligned object: 8×64 atom bins + 28×4 bond bins.

| view | dim | valid ROC-AUC |
|---|---:|---:|
| s | 205 | 0.781696 ± 0.006559 |
| r_raw | 624 | 0.701410 ± 0.006237 |
| r_init | 624 | 0.653540 ± 0.007464 |
| r_final | 624 | 0.675165 ± 0.015278 |
| s_r_raw | 829 | 0.776296 ± 0.010209 |
| s_r_init | 829 | 0.742054 ± 0.009386 |
| s_r_final | 829 | 0.742925 ± 0.007230 |

Best by validation only: `s`.

Primary checks: `r_final-r_init`, `r_final-r_raw`, and `s_r_final-s`.
Test evaluation is disabled.
