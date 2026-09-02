# Mentor concept replication v1

Protocol: `mentor-r2-atom-chem-k64-s8-official-valid`

This is an independent concept reproduction; it does not claim to reproduce the mentor's unknown 69/624-D features.

| view | dim | valid ROC-AUC | test ROC-AUC |
|---|---:|---:|---:|
| s | 205 | 0.781696 ± 0.006559 | not evaluated |
| r_raw | 624 | 0.679726 ± 0.008662 | not evaluated |
| r_init | 624 | 0.679066 ± 0.004191 | not evaluated |
| r_final | 624 | 0.706807 ± 0.004173 | not evaluated |
| s_r_raw | 829 | 0.781777 ± 0.003118 | not evaluated |
| s_r_init | 829 | 0.762917 ± 0.021469 | not evaluated |
| s_r_final | 829 | 0.773751 ± 0.005032 | not evaluated |

Best by validation only: `s_r_raw`.

Primary attribution checks:

- `t_final - t_init`: contribution of K-SVD updates under matched initialization.
- `s_t_final - s`: structural dictionary features beyond explicit composition.
- Test evaluation is disabled unless explicitly requested after freezing the protocol.

Typed-reconstruction proxy checks:

- `r_final - r_init`: K-SVD update contribution in typed reconstruction coordinates.
- `r_final - r_raw`: information retained or lost through sparse reconstruction.
- `s_r_final - s`: typed reconstruction beyond explicit composition.
