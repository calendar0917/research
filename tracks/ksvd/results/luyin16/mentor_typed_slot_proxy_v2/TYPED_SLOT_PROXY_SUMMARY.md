# Typed-slot reconstruction proxy v2

Protocol: `mentor-typed-slot-proxy-v2-official-valid`

This is not the mentor's exact feature builder. It tests a natural 624-D slot-aligned object: 8×64 atom bins + 28×4 bond bins.

| view | dim | valid ROC-AUC |
|---|---:|---:|
| s | 205 | 0.781696 ± 0.006559 |
| r_raw | 624 | 0.701410 ± 0.006237 |
| r_init | 624 | 0.653540 ± 0.007464 |
| r_final | 624 | 0.675165 ± 0.015278 |
| s_r_raw | 1248 | 0.699441 ± 0.007993 |
| s_r_init | 1248 | 0.673885 ± 0.010291 |
| s_r_final | 1248 | 0.680060 ± 0.006694 |

Best by validation only: `s`.

Primary checks: `r_final-r_init`, `r_final-r_raw`, and `s_r_final-s`.
Test evaluation is disabled.
