# Mentor concept replication v1

Protocol: `mentor-r2-atom-chem-k64-s8-smoke`

This is an independent concept reproduction; it does not claim to reproduce the mentor's unknown 69/624-D features.

| view | dim | valid ROC-AUC | test ROC-AUC |
|---|---:|---:|---:|
| s | 205 | 0.448276 ± 0.000000 | not evaluated |
| r_raw | 624 | 0.965517 ± 0.000000 | not evaluated |
| r_init | 624 | 1.000000 ± 0.000000 | not evaluated |
| r_final | 624 | 0.551724 ± 0.000000 | not evaluated |
| s_r_raw | 829 | 0.931034 ± 0.000000 | not evaluated |
| s_r_init | 829 | 1.000000 ± 0.000000 | not evaluated |
| s_r_final | 829 | 0.517241 ± 0.000000 | not evaluated |

Best by validation only: `r_init`.

Primary attribution checks:

- `t_final - t_init`: contribution of K-SVD updates under matched initialization.
- `s_t_final - s`: structural dictionary features beyond explicit composition.
- Test evaluation is disabled unless explicitly requested after freezing the protocol.

Typed-reconstruction proxy checks:

- `r_final - r_init`: K-SVD update contribution in typed reconstruction coordinates.
- `r_final - r_raw`: information retained or lost through sparse reconstruction.
- `s_r_final - s`: typed reconstruction beyond explicit composition.
