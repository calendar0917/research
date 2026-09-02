# Patch relations：masked structural representation 审计

> 日期：2026-08-02  
> 协议：`tracks/ksvd/docs/KSVD_PATCH_RELATION_REPRESENTATION_PROTOCOL_20260802.md`  
> 判定：`PATCH_RELATIONS_ADD_MASKED_VALUE`

## 1. Held-out masked-patch prediction

| branch | overall RMSE | degree | spectrum | density/triangle |
|---|---:|---:|---:|---:|
| RAW_BAG | 0.22443 | 0.27472 | 0.10659 | 0.34384 |
| KSVD_BAG | 0.18916 | 0.23416 | 0.08776 | 0.28318 |
| KSVD_TRUE_RELATION | 0.11869 | 0.14750 | 0.05941 | 0.16752 |
| KSVD_SHUFFLED_RELATION | 0.13980 | 0.17280 | 0.06822 | 0.20443 |

- TRUE vs KSVD_BAG RMSE reduction：`0.3725`；
- TRUE vs SHUFFLED RMSE reduction：`0.1510`；
- TRUE simultaneously better by fold：`[True, True, True]`；
- registered checks：`{'true_vs_bag_gain_at_least_002': True, 'true_vs_shuffled_gain_at_least_002': True, 'true_better_both_all_folds': True, 'invariants': True}`。

## 2. Node-relabel + resampling stability

| branch | cosine | relative L2 | cosine >= 0.90 |
|---|---:|---:|---:|
| RAW_BAG | 0.9908 | 0.1321 | True |
| KSVD_BAG | 0.6953 | 0.7831 | False |
| KSVD_TRUE_RELATION | 0.7221 | 0.7466 | False |
| KSVD_SHUFFLED_RELATION | 0.7214 | 0.7472 | False |

## 3. 解释

- RAW_BAG vs KSVD_BAG：回答 sparse token 相对 raw context 的信息损失；
- TRUE vs BAG：回答 relation-weighted context 是否增加信息；
- TRUE vs SHUFFLED：排除只靠 feature dimension、relation marginals 或 graph-level statistics；
- stability 同时包含 sampler、slot ordering 和 token 的变化，不要求 exact chain replay。

本轮是低容量机制审计，不是最终 Transformer，也不使用 graph labels。
