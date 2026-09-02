# Mentor concept replication v1

Protocol: `mentor-r2-atom-chem-k64-s8-all-centers-official-valid`

This is an independent concept reproduction; it does not claim to reproduce the mentor's unknown 69/624-D features.

| view | dim | valid ROC-AUC | test ROC-AUC |
|---|---:|---:|---:|
| s | 205 | 0.781696 ± 0.006559 | not evaluated |
| r_raw | 624 | 0.798839 ± 0.005777 | not evaluated |
| r_init | 624 | 0.776830 ± 0.012689 | not evaluated |
| r_final | 624 | 0.793545 ± 0.011550 | not evaluated |
| s_r_raw | 829 | 0.812917 ± 0.005638 | not evaluated |
| s_r_init | 829 | 0.784054 ± 0.012141 | not evaluated |
| s_r_final | 829 | 0.809298 ± 0.003946 | not evaluated |

Best by validation only: `s_r_raw`.

Primary attribution checks:

- `t_final - t_init`: contribution of K-SVD updates under matched initialization.
- `s_t_final - s`: structural dictionary features beyond explicit composition.
- Test evaluation is disabled unless explicitly requested after freezing the protocol.

Typed-reconstruction proxy checks:

- `r_final - r_init`: K-SVD update contribution in typed reconstruction coordinates.
- `r_final - r_raw`: information retained or lost through sparse reconstruction.
- `s_r_final - s`: typed reconstruction beyond explicit composition.
