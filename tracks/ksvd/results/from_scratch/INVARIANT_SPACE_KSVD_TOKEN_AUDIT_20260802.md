# Invariant-space KSVD patch token：优化结果

> 日期：2026-08-02  
> 协议：`tracks/ksvd/docs/KSVD_INVARIANT_SPACE_TOKEN_PROTOCOL_20260802.md`  
> 判定：`INVARIANT_KSVD_STABLE_BUT_TOO_LOSSY`

## 1. Held-out masked prediction

| branch | overall RMSE | degree | spectrum | density/triangle |
|---|---:|---:|---:|---:|
| INV_KSVD_BAG | 0.19613 | 0.24589 | 0.09113 | 0.27747 |
| INV_KSVD_TRUE_RELATION | 0.13625 | 0.17360 | 0.06623 | 0.17319 |
| INV_KSVD_SHUFFLED_RELATION | 0.15538 | 0.19551 | 0.07534 | 0.20941 |

- TRUE vs BAG reduction：`0.3053`；
- TRUE vs SHUFFLED reduction：`0.1231`；
- TRUE vs raw invariant TRUE change：`0.0923`；

## 2. Relabel + resampling stability

| branch | cosine | relative L2 |
|---|---:|---:|
| INV_KSVD_BAG | 0.8703 | 0.4921 |
| INV_KSVD_TRUE_RELATION | 0.9506 | 0.2983 |
| INV_KSVD_SHUFFLED_RELATION | 0.9502 | 0.2997 |

Registered checks：`{'true_vs_bag_gain_at_least_002': True, 'true_vs_shuffled_gain_at_least_002': True, 'true_better_both_all_folds': True, 'true_embedding_cosine_at_least_090': True, 'true_not_worse_than_raw_invariant_by_005': False, 'invariants': True}`。

## 3. 解释

该方案保留 KSVD sparse dictionary/code，但字典学习对象从 arbitrary adjacency slots 改为 ID-free local structural coordinates。它优化的是稳定 graph token，不再声称 atom 是固定位置的局部邻接图像。
