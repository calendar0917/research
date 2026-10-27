# DCC-L frozen-transfer audit — machine-generated tables

Δ = MAE(probe on z_4k) − MAE(probe on z_8k); **positive = 8k representation easier to read**.

## 1. Primary cells (full 513, best checkpoint, Monitor1k)

| trunk seed | head init | MAE 4k | MAE 8k | Δ |
|---|---:|---:|---:|---:|
| 0 | 0 | 0.164350 | 0.140058 | +0.024292 |
| 0 | 1 | 0.165418 | 0.140322 | +0.025096 |
| 1 | 0 | 0.172839 | 0.138695 | +0.034143 |
| 1 | 1 | 0.177634 | 0.142373 | +0.035261 |

| block | Δ s0 | Δ s1 | Δ init0 | Δ init1 | Δ all | Δ all (final ckpt) |
|---|---:|---:|---:|---:|---:|---:|
| atom_global | -0.002901 | +0.008119 | +0.002699 | +0.002519 | +0.002609 | +0.009963 |
| local_global | +0.021561 | +0.050535 | +0.035991 | +0.036105 | +0.036048 | +0.037651 |
| full | +0.024694 | +0.034702 | +0.029218 | +0.030178 | +0.029698 | +0.025571 |

## 2. Group table (full 513, best checkpoint)

| group | n | MAE 4k | MAE 8k | Δ mean | Δ s0 | Δ s1 | rel. Δ | bootstrap 95% |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| overall | 1000 | 0.1649 | 0.1402 | +0.029698 | +0.024694 | +0.034702 | +17.5% | [+0.022574, +0.039011] |
| ordinary_nopenalty | 964 | 0.1095 | 0.0886 | +0.026253 | +0.020906 | +0.031600 | +22.9% | [+0.021520, +0.030926] |
| penalised_ring_gt6 | 36 | 1.6475 | 1.5214 | +0.121958 | +0.126141 | +0.117775 | +7.4% | [-0.005501, +0.325799] |
| any_unseen_fit4k | 423 | 0.1518 | 0.1214 | +0.032979 | +0.030389 | +0.035569 | +20.6% | [+0.023752, +0.041961] |
| any_unseen_fit8k | 301 | 0.1698 | 0.1403 | +0.032943 | +0.029503 | +0.036382 | +18.3% | [+0.021333, +0.044445] |
| low_support_fit4k | 405 | 0.1660 | 0.1459 | +0.028249 | +0.020054 | +0.036444 | +16.8% | [+0.015231, +0.049268] |
| low_support_fit8k | 359 | 0.0946 | 0.0756 | +0.025253 | +0.018962 | +0.031544 | +25.5% | [+0.017924, +0.032622] |
| full_support_fit4k | 155 | 0.0625 | 0.0528 | +0.021353 | +0.009753 | +0.032953 | +30.4% | [+0.012886, +0.031556] |
| full_support_fit8k | 324 | 0.1736 | 0.1525 | +0.029929 | +0.021120 | +0.038737 | +17.0% | [+0.014814, +0.054504] |
| newly_covered_4k_to_8k | 123 | 0.1082 | 0.0755 | +0.032617 | +0.032654 | +0.032581 | +29.4% | [+0.020192, +0.046489] |
| still_unseen_both | 317 | 0.2355 | 0.2007 | +0.034497 | +0.034839 | +0.034155 | +14.2% | [+0.023335, +0.046141] |
| extreme_tail | 74 | 0.9089 | 0.8407 | +0.074840 | +0.068232 | +0.081449 | +8.2% | [+0.007407, +0.180636] |
| non_tail | 926 | 0.1054 | 0.0842 | +0.026091 | +0.021215 | +0.030966 | +23.6% | [+0.021414, +0.030974] |
| drop_fixed_top10 | 990 | 0.1074 | 0.0872 | +0.026031 | +0.020172 | +0.031890 | +22.9% | [+0.021488, +0.030678] |

## 3. Block probes (same reader shape, unused columns zeroed)

| group | n | Δ atom_global | Δ local_global | Δ full |
|---|---:|---:|---:|---:|
| overall | 1000 | +0.002609 | +0.036048 | +0.029698 |
| ordinary_nopenalty | 964 | +0.000154 | +0.030916 | +0.026253 |
| penalised_ring_gt6 | 36 | +0.068357 | +0.173466 | +0.121958 |
| any_unseen_fit4k | 423 | -0.001219 | +0.035454 | +0.032979 |
| any_unseen_fit8k | 301 | -0.004655 | +0.053336 | +0.032943 |
| low_support_fit4k | 405 | +0.002137 | +0.029435 | +0.028249 |
| low_support_fit8k | 359 | +0.000844 | +0.022244 | +0.025253 |
| full_support_fit4k | 155 | +0.008902 | +0.036621 | +0.021353 |
| full_support_fit8k | 324 | +0.008096 | +0.024424 | +0.029929 |
| newly_covered_4k_to_8k | 123 | +0.005534 | -0.013811 | +0.032617 |
| still_unseen_both | 317 | -0.001001 | +0.063562 | +0.034497 |
| extreme_tail | 74 | +0.048854 | +0.145151 | +0.074840 |
| non_tail | 926 | -0.001087 | +0.027329 | +0.026091 |
| drop_fixed_top10 | 990 | +0.001401 | +0.033465 | +0.026031 |

### 3a. All block probe cells (best checkpoint, Monitor1k)

| block | trunk seed | head init | MAE 4k | MAE 8k | Δ |
|---|---:|---:|---:|---:|---:|
| atom_global | 0 | 0 | 0.296472 | 0.299939 | -0.003467 |
| atom_global | 0 | 1 | 0.302753 | 0.305088 | -0.002335 |
| atom_global | 1 | 0 | 0.310249 | 0.301384 | +0.008865 |
| atom_global | 1 | 1 | 0.315154 | 0.307781 | +0.007373 |
| local_global | 0 | 0 | 0.562432 | 0.540665 | +0.021767 |
| local_global | 0 | 1 | 0.562192 | 0.540837 | +0.021355 |
| local_global | 1 | 0 | 0.555702 | 0.505487 | +0.050215 |
| local_global | 1 | 1 | 0.553311 | 0.502456 | +0.050856 |
| full | 0 | 0 | 0.164350 | 0.140058 | +0.024292 |
| full | 0 | 1 | 0.165418 | 0.140322 | +0.025096 |
| full | 1 | 0 | 0.172839 | 0.138695 | +0.034143 |
| full | 1 | 1 | 0.177634 | 0.142373 | +0.035261 |

### 3b. Block levels (probe MAE on Monitor1k, best checkpoint)

| block | MAE 4k (2 seeds × 2 inits) | MAE 8k | Δ abs | Δ rel |
|---|---:|---:|---:|---:|
| atom_global | 0.3062 | 0.3035 | +0.002609 | +0.85% |
| local_global | 0.5584 | 0.5224 | +0.036048 | +6.46% |
| full | 0.1701 | 0.1404 | +0.029698 | +17.46% |

### 3c. Probe vs raw trunk (full 513, Monitor1k)

| condition | probe MAE (head-init mean) | raw trunk MAE (final ckpt) | probe − raw |
|---|---:|---:|---:|
| fit4k_s0 | 0.16488 | 0.16269 | +0.00220 |
| fit4k_s1 | 0.17524 | 0.16701 | +0.00823 |
| fit8k_s0 | 0.14019 | 0.12383 | +0.01636 |
| fit8k_s1 | 0.14053 | 0.12851 | +0.01202 |

Raw 4k→8k gain (Monitor, final ckpt) = **+0.03868**; fixed-head Δ = **+0.02970** (recovered fraction 76.8%; s0 63.8%, s1 89.7%).

## 4. Controls

| control | value |
|---|---:|
| recovery fit4k_s0: probe_sel L1 vs trunk pred | 0.05754 |
| recovery fit4k_s0: monitor MAE vs trunk pred | 0.04650 |
| recovery fit4k_s1: probe_sel L1 vs trunk pred | 0.05508 |
| recovery fit4k_s1: monitor MAE vs trunk pred | 0.05938 |
| recovery fit8k_s0: probe_sel L1 vs trunk pred | 0.06087 |
| recovery fit8k_s0: monitor MAE vs trunk pred | 0.06229 |
| recovery fit8k_s1: probe_sel L1 vs trunk pred | 0.04793 |
| recovery fit8k_s1: monitor MAE vs trunk pred | 0.05111 |
| null (permuted labels): monitor MAE vs y | 1.65058 |
| constant median predictor: monitor MAE | 1.53864 |

## 5. Gates and verdict

- `U1_pass` = True
- `U2_pass` = True
- `U3_pass` = True
- `U4_pass` = True
- `T1` = True
- `T2` = True
- `T3` = True
- raw_gain (Monitor, final ckpt) = +0.038676
- Δ_all (full) = +0.029698
- **verdict = `TRUNK_TRANSFER_SIGNAL`** (T1 & T2 & T3 all pass)
