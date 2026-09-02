# Mentor concept replication v1

Protocol: `mentor-concept-replication-v1-k32-topology-official-valid`

This is an independent concept reproduction; it does not claim to reproduce the mentor's unknown 69/624-D features.

| view | dim | valid ROC-AUC | test ROC-AUC |
|---|---:|---:|---:|
| s | 205 | 0.781696 ± 0.006559 | not evaluated |
| t_init | 288 | 0.615718 ± 0.005208 | not evaluated |
| t_final | 288 | 0.620059 ± 0.005797 | not evaluated |
| s_t_init | 493 | 0.771316 ± 0.007954 | not evaluated |
| s_t_final | 493 | 0.753879 ± 0.005598 | not evaluated |

Best by validation only: `s`.

Primary attribution checks:

- `t_final - t_init`: contribution of K-SVD updates under matched initialization.
- `s_t_final - s`: structural dictionary features beyond explicit composition.
- Test evaluation is disabled unless explicitly requested after freezing the protocol.
