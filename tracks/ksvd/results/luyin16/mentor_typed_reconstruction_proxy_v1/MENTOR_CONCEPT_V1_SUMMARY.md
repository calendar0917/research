# Mentor concept replication v1

Protocol: `mentor-typed-reconstruction-proxy-v1-official-valid`

This is an independent concept reproduction; it does not claim to reproduce the mentor's unknown 69/624-D features.

| view | dim | valid ROC-AUC | test ROC-AUC |
|---|---:|---:|---:|
| s | 205 | 0.781696 ± 0.006559 | not evaluated |
| t_init | 328 | 0.671811 ± 0.009430 | not evaluated |
| t_final | 328 | 0.652077 ± 0.014593 | not evaluated |
| r_raw | 624 | 0.664896 ± 0.005712 | not evaluated |
| r_init | 624 | 0.652111 ± 0.008763 | not evaluated |
| r_final | 624 | 0.654166 ± 0.006706 | not evaluated |
| s_r_raw | 829 | 0.751030 ± 0.012658 | not evaluated |
| s_r_init | 829 | 0.762756 ± 0.005680 | not evaluated |
| s_r_final | 829 | 0.747650 ± 0.011049 | not evaluated |

Best by validation only: `s`.

Primary attribution checks:

- `t_final - t_init`: contribution of K-SVD updates under matched initialization.
- `s_t_final - s`: structural dictionary features beyond explicit composition.
- Test evaluation is disabled unless explicitly requested after freezing the protocol.

Typed-reconstruction proxy checks:

- `r_final - r_init`: K-SVD update contribution in typed reconstruction coordinates.
- `r_final - r_raw`: information retained or lost through sparse reconstruction.
- `s_r_final - s`: typed reconstruction beyond explicit composition.
