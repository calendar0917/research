# Mentor-shaped ring-context proxy

Protocol: `mentor-t-readout-ablation-k64-s8-official-valid-v1`

This is a shape-aligned proxy, not an exact reproduction of the unknown upstream feature builders.

| view | dim | valid ROC-AUC |
|---|---:|---:|
| s | 69 | 0.755822 ± 0.008162 |
| t_init_raw | 624 | 0.701410 ± 0.006237 |
| t_final_raw | 624 | 0.701410 ± 0.006237 |
| t_init_reconstruction | 624 | 0.639481 ± 0.004009 |
| t_final_reconstruction | 624 | 0.668703 ± 0.007239 |
| t_init_atom_signed | 624 | 0.649124 ± 0.013791 |
| t_final_atom_signed | 624 | 0.667828 ± 0.016510 |
| t_init_code_mean | 64 | 0.660314 ± 0.011157 |
| t_final_code_mean | 64 | 0.662746 ± 0.011436 |
| t_init_abs_code_mean | 64 | 0.602538 ± 0.008599 |
| t_final_abs_code_mean | 64 | 0.673213 ± 0.013206 |
| t_init_rich_code | 640 | 0.652528 ± 0.007177 |
| t_final_rich_code | 640 | 0.691967 ± 0.008468 |
| t_init_residual_abs | 624 | 0.666689 ± 0.003344 |
| t_final_residual_abs | 624 | 0.711641 ± 0.017009 |
| s_t_init_raw | 693 | 0.759456 ± 0.006480 |
| s_t_final_raw | 693 | 0.759456 ± 0.006480 |
| s_t_init_reconstruction | 693 | 0.712583 ± 0.012531 |
| s_t_final_reconstruction | 693 | 0.742439 ± 0.010928 |
| s_t_init_rich_code | 709 | 0.722513 ± 0.013466 |
| s_t_final_rich_code | 709 | 0.744418 ± 0.011918 |
| s_t_init_residual_abs | 693 | 0.715858 ± 0.006139 |
| s_t_final_residual_abs | 693 | 0.753904 ± 0.009609 |

Primary attribution: `st-s`, `sa-s`, `sta-max(st,sa)`, and `sta-sta_init`.
Official test is not evaluated.
