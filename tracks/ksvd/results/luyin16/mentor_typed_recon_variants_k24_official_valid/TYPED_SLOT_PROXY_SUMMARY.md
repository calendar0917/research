# Typed-slot reconstruction proxy v2

Protocol: `mentor-typed-reconstruction-variants-k24-official-valid`

This is not the mentor's exact feature builder. It tests a natural 624-D slot-aligned object: 8×64 atom bins + 28×4 bond bins.

| view | dim | valid ROC-AUC |
|---|---:|---:|
| s | 205 | 0.781696 ± 0.006559 |
| r_init | 624 | 0.640179 ± 0.010237 |
| r_final | 624 | 0.633510 ± 0.007944 |
| r_init_atom_signed | 624 | 0.672507 ± 0.013106 |
| r_final_atom_signed | 624 | 0.627711 ± 0.004611 |
| r_init_atom_abs | 624 | 0.669406 ± 0.009749 |
| r_final_atom_abs | 624 | 0.632069 ± 0.010048 |
| r_init_abs_mean | 624 | 0.670102 ± 0.012411 |
| r_final_abs_mean | 624 | 0.650484 ± 0.012705 |
| s_r_final | 829 | 0.751336 ± 0.009094 |
| s_r_final_atom_signed | 829 | 0.761891 ± 0.005941 |
| s_r_final_atom_abs | 829 | 0.754835 ± 0.010361 |
| s_r_final_abs_mean | 829 | 0.751875 ± 0.008791 |

Best by validation only: `s`.

Primary checks: `r_final-r_init`, `r_final-r_raw`, and `s_r_final-s`.
Test evaluation is disabled.
