# Mentor concept replication v1

Protocol: `mentor-concept-replication-v1-k32-official-valid`

This is an independent concept reproduction; it does not claim to reproduce the mentor's unknown 69/624-D features.

| view | dim | valid ROC-AUC | test ROC-AUC |
|---|---:|---:|---:|
| s | 205 | 0.781696 ± 0.006559 | not evaluated |
| t_init | 328 | 0.671811 ± 0.009430 | not evaluated |
| t_final | 328 | 0.652077 ± 0.014593 | not evaluated |
| s_t_init | 533 | 0.771681 ± 0.010280 | not evaluated |
| s_t_final | 533 | 0.762350 ± 0.008964 | not evaluated |

Best by validation only: `s`.

Primary attribution checks:

- `t_final - t_init`: contribution of K-SVD updates under matched initialization.
- `s_t_final - s`: structural dictionary features beyond explicit composition.
- Test evaluation is disabled unless explicitly requested after freezing the protocol.
