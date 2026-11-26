# FSRP v0 tables

## Main table

| arm | best_epoch | best_valid_ap | best_valid_bce | endpoint_valid_ap | endpoint_valid_bce | last3_mean_valid_ap | d_best_valid_ap_vs_base | d_best_valid_ap_vs_start | train_ap_25 | train_ap_29 | train_ap_33 | train_bce_25 | train_bce_29 | train_bce_33 | best_train_ap | best_train_bce | train_valid_gap_mean_ap | high_support_valid_ap_best | high_support_valid_ap_endpoint | n_tasks_scored_valid | parameters | mask_dropped_fraction_mean | gpu_hours | wall_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BASE_CONT | 32 | 0.273380 | 0.047292 | 0.272753 | 0.047380 | 0.272875 | 0.000000 | -0.002467 | 0.599570 | 0.626365 | 0.642637 | 0.024222 | 0.023268 | 0.022632 | 0.639658 | 0.022779 | 0.366361 | 0.297022 | 0.297226 | 127 | 1128468 | - | 0.414936 | 1493.768941 |
| STRUCT_DROP | 32 | 0.277572 | 0.046734 | 0.276899 | 0.046744 | 0.277095 | 0.004192 | 0.001727 | 0.599579 | 0.618362 | 0.632375 | 0.024222 | 0.023568 | 0.023035 | 0.630122 | 0.023146 | 0.352634 | 0.301422 | 0.300830 | 127 | 1128468 | 0.099977 | 0.349512 | 1258.241605 |

## Curves

| epoch | BASE_CONT_lr | BASE_CONT_valid_ap | BASE_CONT_valid_bce | BASE_CONT_train_ap | BASE_CONT_train_bce | BASE_CONT_mask_dropped_fraction | STRUCT_DROP_lr | STRUCT_DROP_valid_ap | STRUCT_DROP_valid_bce | STRUCT_DROP_train_ap | STRUCT_DROP_train_bce | STRUCT_DROP_mask_dropped_fraction | valid_ap_delta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 25 | 0.000104 | 0.275848 | 0.046377 | 0.599570 | 0.024222 | - | 0.000104 | 0.275845 | 0.046377 | 0.599579 | 0.024222 | - | -0.000002 |
| 26 | 0.000093 | 0.272653 | 0.046557 | - | - | - | 0.000093 | 0.275235 | 0.046358 | - | - | 0.100027 | 0.002582 |
| 27 | 0.000083 | 0.273336 | 0.046537 | - | - | - | 0.000083 | 0.276506 | 0.046225 | - | - | 0.100141 | 0.003170 |
| 28 | 0.000073 | 0.272956 | 0.046738 | - | - | - | 0.000073 | 0.276348 | 0.046381 | - | - | 0.100046 | 0.003392 |
| 29 | 0.000063 | 0.272559 | 0.046857 | 0.626365 | 0.023268 | - | 0.000063 | 0.276227 | 0.046388 | 0.618362 | 0.023568 | 0.099968 | 0.003668 |
| 30 | 0.000055 | 0.273208 | 0.047042 | - | - | - | 0.000055 | 0.275845 | 0.046521 | - | - | 0.099777 | 0.002637 |
| 31 | 0.000046 | 0.272492 | 0.047088 | - | - | - | 0.000046 | 0.276808 | 0.046509 | - | - | 0.100108 | 0.004316 |
| 32 | 0.000039 | 0.273379 | 0.047292 | - | - | - | 0.000039 | 0.277578 | 0.046734 | - | - | 0.099911 | 0.004198 |
| 33 | 0.000032 | 0.272753 | 0.047380 | 0.642637 | 0.022632 | - | 0.000032 | 0.276899 | 0.046744 | 0.632375 | 0.023035 | 0.099841 | 0.004146 |

## Paired comparisons (tasks)

- **best**: mean delta 0.004192, 95% CI [0.001740, 0.006560], tasks +97/-30, bootstrap P(delta>0)=0.999500; drop-top3 mean 0.004155
- **endpoint**: mean delta 0.004146, 95% CI [0.002386, 0.006167], tasks +94/-33, bootstrap P(delta>0)=1.000000; drop-top3 mean 0.002961
- **last3**: mean delta 0.004220, 95% CI [0.002519, 0.006067], tasks +96/-31, bootstrap P(delta>0)=1.000000; drop-top3 mean 0.003878
- **STRUCT_DROP best vs epoch25 start**: mean delta 0.001725, 95% CI [-0.001138, 0.004547], tasks +75/-52, bootstrap P(delta>0)=0.879200; drop-top3 mean 0.001343
- **BASE_CONT best vs epoch25 start**: mean delta -0.002467, 95% CI [-0.005705, 0.000700], tasks +56/-71, bootstrap P(delta>0)=0.063200; drop-top3 mean -0.002008

## Verdict: `NO_CLEAR_GAIN`

the pre-registered +0.003 best-Valid-AP practical threshold is not met

Criteria:

- best_gain_vs_base_ge_0.003: True
- best_gain_vs_start_ge_0.003: False
- endpoint_not_worse: True
- last3_not_worse: True
- high_support_not_worse: True
- both_runs_completed: True

## Train-Valid gap (best checkpoint, common scorable tasks)

- BASE_CONT: train mean AP 0.639741, valid mean AP 0.273380, gap 0.366361 over 127 tasks
- STRUCT_DROP: train mean AP 0.630206, valid mean AP 0.277572, gap 0.352634 over 127 tasks
