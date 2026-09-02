# Train-only stitching reliability 与 unseen completion 审计

> 日期：2026-08-02  
> K24/T3/u25；3-fold graph isolation；test truth 不参与权重或 completion 拟合。

## 1. 判定

**TRAIN_ONLY_POSTPROCESSING_REDUCES_ERROR**

## 2. Observed stitching

| base | observed RMSE | observed F1 | zero-fill full recall |
|---|---:|---:|---:|
| raw_uniform | 0.0000 | 1.0000 | 0.6845 |
| ksvd_uniform | 0.3627 | 0.8411 | 0.5917 |
| ksvd_weighted | 0.3607 | 0.8410 | 0.5919 |

Slot weighting RMSE reduction：`0.0055`；checks：`{'observed_rmse_reduction_at_least_001': False, 'observed_f1_not_worse': False, 'full_recall_preserved': True}`。

## 3. Full graph completion

| stage | full RMSE | precision | recall | F1 |
|---|---:|---:|---:|---:|
| raw_uniform_zero | 0.3544 | 1.0000 | 0.6845 | 0.8121 |
| raw_uniform_density | 0.3079 | 1.0000 | 0.6845 | 0.8121 |
| raw_uniform_structural | 0.2900 | 0.9729 | 0.7232 | 0.8268 |
| raw_uniform_shuffled | 0.2951 | 0.9702 | 0.7206 | 0.8242 |
| ksvd_uniform_zero | 0.4358 | 0.8210 | 0.5917 | 0.6868 |
| ksvd_uniform_density | 0.3989 | 0.8210 | 0.5917 | 0.6868 |
| ksvd_uniform_structural | 0.3854 | 0.8051 | 0.6304 | 0.7046 |
| ksvd_uniform_shuffled | 0.3890 | 0.8024 | 0.6278 | 0.7019 |
| ksvd_weighted_zero | 0.4350 | 0.8206 | 0.5919 | 0.6868 |
| ksvd_weighted_density | 0.3980 | 0.8206 | 0.5919 | 0.6868 |
| ksvd_weighted_structural | 0.3845 | 0.8047 | 0.6306 | 0.7046 |
| ksvd_weighted_shuffled | 0.3881 | 0.8021 | 0.6280 | 0.7019 |

Structural vs density RMSE reduction：`0.0581`。

Structural vs shuffled RMSE reduction：`0.0174`；checks：`{'rmse_vs_density_at_least_001': True, 'rmse_vs_shuffled_at_least_001': True}`。

## 4. Error budget

`{'raw_zero_full_rmse': 0.35439427233501664, 'ksvd_uniform_zero_full_rmse': 0.4358181317416565, 'ksvd_weighted_zero_full_rmse': 0.4350071555556423, 'ksvd_weighted_structural_full_rmse': 0.3845058249969305, 'compression_rmse_increment': 0.08142385940663988, 'slot_weighting_rmse_change': -0.0008109761860142228, 'completion_rmse_change': -0.05050133055871181}`

## 5. 边界

- slot reliability 只改变 repeated-pair aggregation，不改变 patch reconstruction 本身。
- completion targets 来自 train graphs 的 unseen pairs；它是 graph completion，不是 patch compressor 的内生能力。
- shuffled structural predictions 保留每图预测分布，只破坏 pair binding。
