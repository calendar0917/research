# zinc_shared_factor_supplement_v1 (T vs N, seed 1)

| arm | seed | params | best valid | best ep | train@best | final valid | best20 median | tail100 median | s/epoch | GPU-h |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SF-Dense | 0 | 449601 | 0.081252 | 339 | 0.019250 | 0.082611 | 0.082610 | 0.082645 | 2.37 | 0.411 |
| SF-SharedState | 0 | 451777 | 0.077045 | 384 | 0.016686 | 0.078846 | 0.078646 | 0.078800 | 3.58 | 0.620 |
| SF-SharedState | 1 | 451777 | 0.079088 | 363 | 0.019183 | 0.082169 | 0.082070 | 0.082243 | 3.13 | 0.540 |
| SF-IndependentCopies | 1 | 451777 | 0.079887 | 415 | 0.012027 | 0.081271 | 0.081238 | 0.081305 | 4.69 | 0.809 |

## paired per-molecule contrasts (best checkpoint)

| pair | ΔMAE | 95% CI (molecules only) | frac improved | median ΔAE |
| --- | --- | --- | --- | --- |
| T1_minus_N1 | -0.000799 | [-0.008986, +0.004826] | 0.459 | +0.002766 |
| T1_minus_T0 | +0.002042 | [-0.003023, +0.007677] | 0.486 | +0.000929 |
| N1_minus_T0 | +0.002841 | [-0.004991, +0.014750] | 0.533 | -0.001775 |
| T0_minus_D0 | -0.004207 | [-0.008290, -0.000270] | 0.526 | -0.002131 |
| N1_minus_D0 | -0.001365 | [-0.008526, +0.008733] | 0.556 | -0.004351 |

## bulk: ordinary molecules only (y_raw > -3, n=935)

| arm | MAE |
| --- | --- |
| D_s0 | 0.071447 |
| T_s0 | 0.069207 |
| T_s1 | 0.070187 |
| N_s1 | 0.067666 |

Δ and molecule-level CI: T1_minus_N1 +0.002521 [-0.000415, +0.005401], T1_minus_T0 +0.000980 [-0.003103, +0.004968], N1_minus_T0 -0.001541 [-0.004961, +0.001861], T0_minus_D0 -0.002240 [-0.005591, +0.001079], N1_minus_D0 -0.003781 [-0.007654, +0.000059]

## epoch-level direction

- epochs where T1 < N1: 210/600 (after ep100: 0.332)
- after ep100: T0 < D0 0.852, T1 < T0 0.206

## groups: radius-2 x ordinary (all 1000 molecules)

| cell | n | D_s0 | T_s0 | T_s1 | N_s1 | Δ(T1−N1) | Δ(T1−T0) | P3 (desc.) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seen_ordinary | 710 | 0.0510 | 0.0487 | 0.0503 | 0.0471 | +0.0033 | +0.0016 | 0.0475 |
| seen_tail | 39 | 0.1605 | 0.1472 | 0.1725 | 0.2570 | -0.0845 | +0.0253 | 0.1163 |
| unseen_ordinary | 225 | 0.1359 | 0.1339 | 0.1329 | 0.1327 | +0.0002 | -0.0010 | 0.1339 |
| unseen_tail | 26 | 0.3149 | 0.2537 | 0.2591 | 0.2538 | +0.0053 | +0.0054 | 0.3019 |

### contribution decomposition (n/1000 x cell ΔMAE)

| cell | n/1000 | Δ(T1−N1) | contributes | Δ(T1−T0) | contributes |
| --- | --- | --- | --- | --- | --- |
| seen_ordinary | 0.710 | +0.003270 | +0.002322 | +0.001609 | +0.001143 |
| seen_tail | 0.039 | -0.084466 | -0.003294 | +0.025297 | +0.000987 |
| unseen_ordinary | 0.225 | +0.000157 | +0.000035 | -0.001008 | -0.000227 |
| unseen_tail | 0.026 | +0.005306 | +0.000138 | +0.005366 | +0.000140 |
| **sum** | 1.000 | | -0.000799 | | +0.002042 |

decomposition check (T_s1): overall 0.079088 vs sum of cells 0.079088 (|diff| 1.39e-17)

## pre-registered Q1 (T vs N at seed 1)

```json
{
 "best_valid_T_lt_N": {
  "T": 0.07908751202275743,
  "N": 0.07988671014731517,
  "pass": true
 },
 "best20_median_T_lt_N": {
  "T": 0.08207038963300875,
  "N": 0.0812378194881021,
  "pass": false
 },
 "paired_delta_mae_T_lt_N": {
  "delta_mae": -0.0007991981245577336,
  "pass": true
 },
 "n_conditions_passed": 2,
 "supports_sharing": false
}
```

## Q2 (T absolute level across supervised seeds)

- T seed0 best 0.077045, T seed1 best 0.079088, reference bar 0.075: below at s0=False, below at s1=False

## Q3 (225 unseen+ordinary molecules)

- n=225; MAE: D_s0 0.135892, T_s0 0.133880, T_s1 0.132872, N_s1 0.132715; historical P3 0.133939597 (descriptive)
- Δ(T1−N1) +0.000157 (contributes +0.000035 of overall MAE), Δ(T1−T0) -0.001008 (contributes -0.000227)
