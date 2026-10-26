# BOSI v0 — frozen Locked tables

generated 2026-10-26T16:58:17+0800 · n_locked = 1000

## Runs (best-monitor checkpoint)

| run | arm | seed | Locked MAE | Monitor MAE |
|---|---|---:|---:|---:|
| R_s0/best_monitor | R | 0 | 0.080390 | 0.121329 |
| R_s0/final | R | 0 | 0.079717 | 0.121329 |
| B_s0/best_monitor | B | 0 | 0.082449 | 0.124505 |
| B_s0/final | B | 0 | 0.081969 | 0.124505 |
| R_s1/best_monitor | R | 1 | 0.083037 | 0.124166 |
| R_s1/final | R | 1 | 0.082794 | 0.124166 |
| B_s1/best_monitor | B | 1 | 0.081491 | 0.122400 |
| B_s1/final | B | 1 | 0.080795 | 0.122400 |

## Frozen groups

| group | n |
|---|---:|
| all | 1000 |
| ordinary | 932 |
| tail | 68 |
| low_support | 398 |
| any_unseen | 287 |
| all_supported | 315 |
| has_T_or_Q | 75 |
| alloc_defined_top_quartile | 252 |
| alloc_defined_zero | 3 |
| rings_ge4 | 212 |

## Per-group MAE (best-monitor)

| group | B_s0 | B_s1 | R_s0 | R_s1 |
|---|---:|---:|---:|---:|
| all | 0.082449 | 0.081491 | 0.080390 | 0.083037 |
| ordinary | 0.065825 | 0.064926 | 0.064903 | 0.066084 |
| tail | 0.310291 | 0.308520 | 0.292658 | 0.315394 |
| low_support | 0.061111 | 0.060220 | 0.058252 | 0.060161 |
| any_unseen | 0.158053 | 0.154992 | 0.156808 | 0.160127 |
| all_supported | 0.040525 | 0.041399 | 0.038737 | 0.041703 |
| has_T_or_Q | 0.075582 | 0.061191 | 0.084564 | 0.078990 |
| alloc_defined_top_quartile | 0.090129 | 0.087765 | 0.082218 | 0.091307 |
| alloc_defined_zero | 0.126862 | 0.138284 | 0.139223 | 0.130810 |
| rings_ge4 | 0.140503 | 0.137866 | 0.129064 | 0.140271 |

## Paired contrasts (Locked, negative = first is better)

| contrast | ΔMAE | 95% CI | better frac | top-10 dropped |
|---|---:|---|---:|---:|
| R_s0_minus_B_s0 | -0.002058 | [-0.005559, +0.001307] | 0.505 | -0.000942 |
| R_s0_minus_DCCL_s0 | -0.003183 | [-0.007361, +0.001072] | 0.548 | -0.002346 |
| B_s0_minus_DCCL_s0 | -0.001125 | [-0.005465, +0.003245] | 0.502 | -0.001433 |
| R_s1_minus_B_s1 | +0.001546 | [-0.001693, +0.004873] | 0.492 | +0.001690 |
| R_s1_minus_DCCL_s1 | -0.000386 | [-0.004016, +0.003185] | 0.504 | +0.000774 |
| B_s1_minus_DCCL_s1 | -0.001932 | [-0.005725, +0.001817] | 0.499 | -0.000987 |

## Frozen-weight interventions (best-monitor)

| run | swap-to-arm MAE | zero-stats MAE | side-zero MAE | rel-perm mean \|Δy\| |
|---|---:|---:|---:|---:|
| R_s0 | 0.111759 | 0.292847 | 0.317174 | 0.177518 |
| B_s0 | 0.092184 | 0.301454 | 0.336774 | 0.000000 |
| R_s1 | 0.109913 | 0.324171 | 0.306990 | 0.173682 |
| B_s1 | 0.091666 | 0.286573 | 0.328032 | 0.000000 |

**decision = `NO_GO`**
