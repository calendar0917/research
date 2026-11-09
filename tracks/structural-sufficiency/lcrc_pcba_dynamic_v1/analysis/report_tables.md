### Headline comparison (official Test = EXPOSED_TEST_ENGINEERING_COMPARISON_SET)

| Model | Params | Best Valid AP | Test AP best-Valid | Valid@40 | Test@40 | GPU-hour |
|---|---:|---:|---:|---:|---:|---:|
| W2 historical | 994324 | 0.265273 | 0.253617 | 0.261867 | 0.250986 | 1.3949 |
| RC-REAL historical | 1051022 | 0.267718 | 0.255941 | 0.266950 | 0.256167 | 1.9186 |
| W2-DYNAMIC | 1128468 | 0.275067 | 0.262226 | 0.268791 | 0.260136 | 1.7340 |

### Deltas

| Quantity | Value |
|---|---:|
| Dynamic − W2, Test AP (best-Valid, 127 tasks) | 0.008609 |
| Dynamic − RC-REAL, Test AP (best-Valid, 127 tasks) | 0.006285 |
| Dynamic − W2, Validation AP (best-Valid) | 0.009794 |
| Dynamic − RC-REAL, Validation AP (best-Valid) | 0.007349 |
| Dynamic − W2, Test AP @40 | 0.009150 |
| Dynamic − RC-REAL, Test AP @40 | 0.003969 |
| Dynamic − W2, Validation AP @40 | 0.006924 |

### Support strata (Test positives, best-Valid checkpoints)

| Stratum | n tasks | mean Δ | median Δ | improved | degraded | macro contribution |
|---|---:|---:|---:|---:|---:|---:|
| Test positives ≥ 20 | 108 | 0.010866 | 0.011586 | 76 | 32 | 0.009240 |
| Test positives 5–19 | 18 | 0.004212 | 0.000598 | 11 | 7 | 0.000597 |
| Test positives < 5 | 1 | -0.155968 | -0.155968 | 0 | 1 | -0.001228 |
| all scored tasks | 127 | 0.008609 | 0.009379 | 87 | 40 | 0.008609 |
| Validation & Test ≥ 20 (common) | 106 | 0.010624 | 0.011586 | 75 | 31 | 0.008867 |
| Validation ≥ 20 (Validation AP) | 107 | 0.011262 | 0.007760 | 75 | 32 | 0.009488 |

Test positives ≥20 tasks: **108**; 5–19: 18; <5: 1; scored: 127.

### Dynamic usage on the full official Validation split

| Quantity | Value |
|---|---:|
| rms_h0 | 0.973763 |
| rms_h1_minus_h0 | 0.628428 |
| rms_h2_minus_h1 | 0.617609 |
| rms_h2_minus_h0 | 0.914377 |
| R1 | 0.645361 |
| R2 | 0.529090 |
| Rtotal | 0.939014 |
| finite_fraction | 1.000000 |

### Post-training Validation-only interventions (best-Valid checkpoint)

| Mode | Validation AP | Δ vs real | rel. RMS logits change |
|---|---:|---:|---:|
| real | 0.275064 | — | — |
| message_off | 0.114931 | -0.160133 | 0.474952 |
| message_shuffle | 0.175790 | -0.099274 | 0.232090 |

### Decision

**`POSSIBLE_DYNAMIC_SIGNAL`** — overall gain in the 0.005..0.010 band or offset by low-support tasks

| Condition | Value | Met |
|---|---:|:---:|
| delta_test_best | 0.008609 | — |
| delta_valid_best | 0.009794 | — |
| high_support_improved | 76 | — |
| high_support_tasks | 108 | — |
| high_support_mean_delta | 0.010866 | — |
| low_support_mean_delta | -0.155968 | — |
| dynamic_branch_used | True | — |
| Rtotal_valid | 0.939014 | — |
| delta_test_ge_0.010 | False | no |
| improved_high_support_ge_70 | True | yes |
| high_support_mean_ge_0.006 | True | yes |
| validation_direction_positive | True | yes |
| dynamic_branch_used | True | yes |
