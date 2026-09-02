# Mentor concept replication v1

Protocol: `mentor-r2-atom-topo-k64-s8-all-centers-official-valid`

This is an independent concept reproduction; it does not claim to reproduce the mentor's unknown 69/624-D features.

| view | dim | valid ROC-AUC | test ROC-AUC |
|---|---:|---:|---:|
| s | 205 | 0.781696 ± 0.006559 | not evaluated |
| r_raw | 336 | 0.719520 ± 0.004030 | not evaluated |
| r_init | 336 | 0.744762 ± 0.012146 | not evaluated |
| r_final | 336 | 0.735570 ± 0.009767 | not evaluated |
| s_r_raw | 541 | 0.785364 ± 0.010729 | not evaluated |
| s_r_init | 541 | 0.796761 ± 0.006421 | not evaluated |
| s_r_final | 541 | 0.792413 ± 0.007670 | not evaluated |

Best by validation only: `s_r_init`.

Primary attribution checks:

- `t_final - t_init`: contribution of K-SVD updates under matched initialization.
- `s_t_final - s`: structural dictionary features beyond explicit composition.
- Test evaluation is disabled unless explicitly requested after freezing the protocol.

Typed-reconstruction proxy checks:

- `r_final - r_init`: K-SVD update contribution in typed reconstruction coordinates.
- `r_final - r_raw`: information retained or lost through sparse reconstruction.
- `s_r_final - s`: typed reconstruction beyond explicit composition.
