# Train-only stitching reliability 与 unseen completion 审计

> 日期：2026-08-02  
> K24/T3/u25；3-fold graph isolation；test truth 不参与权重或 completion 拟合。

## 1. 判定

**TRAIN_ONLY_POSTPROCESSING_REDUCES_ERROR**

## 2. Observed stitching

| base | observed RMSE | observed F1 | zero-fill full recall |
|---|---:|---:|---:|
| raw_uniform | 0.0000 | 1.0000 | 0.8740 |
| ksvd_uniform | 0.3489 | 0.8780 | 0.8137 |
| ksvd_weighted | 0.3489 | 0.8780 | 0.8138 |

Slot weighting RMSE reduction：`0.0003`；checks：`{'observed_rmse_reduction_at_least_001': False, 'observed_f1_not_worse': True, 'full_recall_preserved': True}`。

## 3. Full graph completion

| stage | full RMSE | precision | recall | F1 |
|---|---:|---:|---:|---:|
| raw_uniform_zero | 0.2200 | 1.0000 | 0.8740 | 0.9323 |
| raw_uniform_density | 0.2086 | 1.0000 | 0.8740 | 0.9323 |
| raw_uniform_structural | 0.2031 | 1.0000 | 0.8740 | 0.9323 |
| raw_uniform_shuffled | 0.2054 | 1.0000 | 0.8740 | 0.9323 |
| ksvd_uniform_zero | 0.3349 | 0.8322 | 0.8137 | 0.8224 |
| ksvd_uniform_density | 0.3275 | 0.8322 | 0.8137 | 0.8224 |
| ksvd_uniform_structural | 0.3241 | 0.8322 | 0.8137 | 0.8224 |
| ksvd_uniform_shuffled | 0.3255 | 0.8322 | 0.8137 | 0.8224 |
| ksvd_weighted_zero | 0.3349 | 0.8322 | 0.8138 | 0.8224 |
| ksvd_weighted_density | 0.3275 | 0.8322 | 0.8138 | 0.8224 |
| ksvd_weighted_structural | 0.3241 | 0.8322 | 0.8138 | 0.8224 |
| ksvd_weighted_shuffled | 0.3254 | 0.8322 | 0.8138 | 0.8224 |

Structural vs density RMSE reduction：`0.0266`。

Structural vs shuffled RMSE reduction：`0.0113`；checks：`{'rmse_vs_density_at_least_001': True, 'rmse_vs_shuffled_at_least_001': True}`。

## 4. Error budget

`{'raw_zero_full_rmse': 0.21998420447846848, 'ksvd_uniform_zero_full_rmse': 0.33494521952388084, 'ksvd_weighted_zero_full_rmse': 0.3348932691907274, 'ksvd_weighted_structural_full_rmse': 0.32405993984641607, 'compression_rmse_increment': 0.11496101504541237, 'slot_weighting_rmse_change': -5.195033315341657e-05, 'completion_rmse_change': -0.01083332934431136}`

## 5. 边界

- slot reliability 只改变 repeated-pair aggregation，不改变 patch reconstruction 本身。
- completion targets 来自 train graphs 的 unseen pairs；它是 graph completion，不是 patch compressor 的内生能力。
- shuffled structural predictions 保留每图预测分布，只破坏 pair binding。
