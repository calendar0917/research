# Explicit transition-aware decoder 审计

> 日期：2026-08-01  
> FINAL KSVD codes frozen；ridge 只用 train graphs；无 labels。

## 1. 判定

**REJECT_LINEAR_TRANSITION_CONTEXT**

## 2. 三折 held-out 结果

| fold | train/test graphs | stage | patch rel err | observed RMSE | observed F1 | full edge recall | overlap disagreement |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | 45/27 | raw | 0.0000 | 0.0000 | 1.0000 | 0.6876 | 0.0000 |
| 0 | 45/27 | base_final | 0.4865 | 0.3641 | 0.8401 | 0.5947 | 0.1395 |
| 0 | 45/27 | current_only | 0.4864 | 0.3639 | 0.8409 | 0.5948 | 0.1404 |
| 0 | 45/27 | true_transition | 0.4916 | 0.3691 | 0.8356 | 0.5912 | 0.1400 |
| 0 | 45/27 | shuffled_transition | 0.4990 | 0.3721 | 0.8309 | 0.5882 | 0.1486 |
| 1 | 45/27 | raw | 0.0000 | 0.0000 | 1.0000 | 0.6781 | 0.0000 |
| 1 | 45/27 | base_final | 0.4920 | 0.3642 | 0.8381 | 0.5829 | 0.1410 |
| 1 | 45/27 | current_only | 0.4919 | 0.3640 | 0.8393 | 0.5840 | 0.1409 |
| 1 | 45/27 | true_transition | 0.4967 | 0.3687 | 0.8345 | 0.5808 | 0.1394 |
| 1 | 45/27 | shuffled_transition | 0.5056 | 0.3723 | 0.8294 | 0.5760 | 0.1517 |
| 2 | 54/18 | raw | 0.0000 | 0.0000 | 1.0000 | 0.6878 | 0.0000 |
| 2 | 54/18 | base_final | 0.4867 | 0.3599 | 0.8449 | 0.5975 | 0.1381 |
| 2 | 54/18 | current_only | 0.4866 | 0.3597 | 0.8429 | 0.5941 | 0.1380 |
| 2 | 54/18 | true_transition | 0.4899 | 0.3631 | 0.8376 | 0.5889 | 0.1350 |
| 2 | 54/18 | shuffled_transition | 0.5000 | 0.3676 | 0.8358 | 0.5868 | 0.1490 |

## 3. Fold-balanced stage means

| stage | patch rel err | observed RMSE | observed F1 | full edge recall | overlap disagreement |
|---|---:|---:|---:|---:|---:|
| raw | 0.0000 | 0.0000 | 1.0000 | 0.6845 | 0.0000 |
| base_final | 0.4884 | 0.3627 | 0.8411 | 0.5917 | 0.1395 |
| current_only | 0.4883 | 0.3625 | 0.8410 | 0.5909 | 0.1398 |
| true_transition | 0.4927 | 0.3670 | 0.8359 | 0.5869 | 0.1381 |
| shuffled_transition | 0.5015 | 0.3707 | 0.8320 | 0.5837 | 0.1498 |

## 4. Registered gates

- RAW gate：`True`；decoder invariants：`True`。
- CURRENT vs BASE mean RMSE reduction：`0.0006`；2% fold support：`0/3`。
- TRUE vs CURRENT mean RMSE reduction：`-0.0123`。
- TRUE vs SHUFFLED mean RMSE reduction：`0.0100`。
- TRUE complete fold support：`0/3`；mean checks：`{'rmse_vs_current_at_least_002': False, 'rmse_vs_shuffled_at_least_002': False, 'disagreement_not_worse': True, 'observed_f1_not_worse': False, 'full_edge_recall_preserved': True}`。
- transition gate：`False`；current recalibration gate：`False`。

## 5. Feature isolation

- current decoder 输入只有当前 FINAL sparse code。
- transition decoder 的 previous degrees 只由 BASE_FINAL previous reconstruction 计算；不读取 previous raw adjacency。
- SHUFFLED 在每张测试图内部循环错位 context tail，current code 保持逐行不变。
- 每折 feature dimensions、row counts、finite values、train/test graph isolation 均写入 JSON invariants。

## 6. 解释边界

- 这是线性、低容量的机制探针；失败不等价于所有非线性 transition model 都失败。
- TRUE 若不优于 SHUFFLED，说明收益不能归因于正确 correspondence binding。
- 本轮只回答无标签 reconstruction，不回答分类或 Transformer 的有效性。
