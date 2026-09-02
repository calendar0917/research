# Mentor concept replication v1

Protocol: `mentor-concept-replication-v1-dev`

This is an independent concept reproduction; it does not claim to reproduce the mentor's unknown 69/624-D features.

| view | dim | valid ROC-AUC | test ROC-AUC |
|---|---:|---:|---:|
| s | 205 | 0.750143 ± 0.008856 | not evaluated |
| t_init | 328 | 0.520440 ± 0.032117 | not evaluated |
| t_final | 328 | 0.588441 ± 0.023081 | not evaluated |
| s_t_init | 533 | 0.659040 ± 0.006601 | not evaluated |
| s_t_final | 533 | 0.666948 ± 0.033331 | not evaluated |

Best by validation only: `s`.

Primary attribution checks:

- `t_final - t_init`: contribution of K-SVD updates under matched initialization.
- `s_t_final - s`: structural dictionary features beyond explicit composition.
- Test evaluation is disabled unless explicitly requested after freezing the protocol.
