# Center-token checkpoint fixed screen

Protocol: `luyin16-molhiv-center-token-route-checkpoint-v1`

All rows and transforms are official-train scaffold only.

| view | dim | fixed scaffold mean | fold std |
|---|---:|---:|---:|
| `s_marginal` | 508 | 0.775249 | 0.011936 |
| `s_distribution` | 2244 | 0.783741 | 0.011349 |
| `s_conditional` | 1110 | 0.782849 | 0.012731 |
| `s_conditional_pca` | 524 | 0.776180 | 0.011540 |
| `s_ksvd_init` | 4806 | 0.775952 | 0.023678 |
| `s_ksvd_final` | 4806 | 0.781050 | 0.022845 |
| `s_center_token_all` | 7144 | 0.783378 | 0.024038 |

## Null controls

- `conditional_shuffled`: 0.772606 (fold std 0.013582)
- `ksvd_final_shuffled`: 0.758547 (fold std 0.012190)
