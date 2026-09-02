# KSVD continuous-cover stitched reconstruction 审计

> 日期：2026-08-01  
> train/test graph isolation；无 labels、无 classification。

## 1. 判定

**FAIL_KSVD_COVER_COMPRESSION**

## 2. 三折 held-out 结果

| fold | train/test graphs | train/test patches | stage | patch rel err | observed RMSE | observed F1 | full edge recall/F1 | overlap disagreement | nondead/max share |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | 45/27 | 793/478 | raw | 0.0000 | 0.0000 | 1.0000 | 0.6876/0.8143 | 0.0000 | - |
| 0 | 45/27 | 793/478 | init | 0.5126 | 0.3854 | 0.8222 | 0.5816/0.6736 | 0.1437 | 24/0.0862 |
| 0 | 45/27 | 793/478 | final | 0.4865 | 0.3641 | 0.8401 | 0.5947/0.6880 | 0.1395 | 24/0.0551 |
| 0 | 45/27 | 793/478 | pca3 | 0.5538 | 0.4186 | 0.7694 | 0.5572/0.6324 | 0.1198 | - |
| 1 | 45/27 | 795/476 | raw | 0.0000 | 0.0000 | 1.0000 | 0.6781/0.8075 | 0.0000 | - |
| 1 | 45/27 | 795/476 | init | 0.5174 | 0.3848 | 0.8204 | 0.5714/0.6665 | 0.1442 | 24/0.0783 |
| 1 | 45/27 | 795/476 | final | 0.4920 | 0.3642 | 0.8381 | 0.5829/0.6803 | 0.1410 | 24/0.0537 |
| 1 | 45/27 | 795/476 | pca3 | 0.5616 | 0.4196 | 0.7686 | 0.5461/0.6258 | 0.1214 | - |
| 2 | 54/18 | 954/317 | raw | 0.0000 | 0.0000 | 1.0000 | 0.6878/0.8145 | 0.0000 | - |
| 2 | 54/18 | 954/317 | init | 0.5190 | 0.3872 | 0.8217 | 0.5820/0.6734 | 0.1403 | 24/0.0640 |
| 2 | 54/18 | 954/317 | final | 0.4867 | 0.3599 | 0.8449 | 0.5975/0.6919 | 0.1381 | 24/0.0521 |
| 2 | 54/18 | 954/317 | pca3 | 0.5560 | 0.4143 | 0.7705 | 0.5530/0.6322 | 0.1230 | - |

## 3. Fold-balanced stage means

| stage | patch rel err | observed RMSE | observed F1 | full edge recall | overlap disagreement |
|---|---:|---:|---:|---:|---:|
| raw | 0.0000 | 0.0000 | 1.0000 | 0.6845 | 0.0000 |
| init | 0.5163 | 0.3858 | 0.8214 | 0.5783 | 0.1427 |
| final | 0.4884 | 0.3627 | 0.8411 | 0.5917 | 0.1395 |
| pca3 | 0.5572 | 0.4175 | 0.7695 | 0.5521 | 0.1214 |

## 4. Registered gates

- raw gate：`True`；
- KSVD optimization gate：`False`；mean patch reduction `0.0540`；
- stitched gate：`True`；mean observed RMSE reduction `0.0597`；
- stitched checks：`{'final_better_all_folds': True, 'mean_rmse_reduction': True, 'disagreement_not_worse': True, 'full_edge_recall_preserved': True, 'observed_f1_not_worse': True, 'beats_pca3_rmse': True}`。

## 5. 解释边界

- RAW 是当前 sampler 的 ceiling，不是可部署压缩模型；未观察 pairs 仍固定预测为 0。
- PCA3 与 KSVD 都使用 3 个连续系数，但 PCA 没有 sparse atom identity；胜负只回答压缩误差，不回答 motif 语义。
- overlap disagreement 衡量同一 global pair 从不同 local slots 重构时是否一致，它不会被单纯 patch Frobenius error 自动保证。
- 本轮不评估下游任务；通过也只能把 KSVD 定位为 continuous-cover sparse compressor。
