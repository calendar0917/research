# ID-free invariant patch token + relation：跟进结果

> 日期：2026-08-02  
> 协议：`tracks/ksvd/docs/KSVD_INVARIANT_PATCH_RELATION_FOLLOWUP_PROTOCOL_20260802.md`  
> 判定：`ID_FREE_PATCH_RELATION_SUBSTRATE_SUPPORTED`

## 1. Held-out masked prediction

| branch | overall RMSE | degree | spectrum | density/triangle |
|---|---:|---:|---:|---:|
| INVARIANT_BAG | 0.16831 | 0.20962 | 0.08555 | 0.22738 |
| INVARIANT_TRUE_RELATION | 0.12474 | 0.15678 | 0.06939 | 0.14748 |
| INVARIANT_SHUFFLED_RELATION | 0.14263 | 0.17783 | 0.07651 | 0.18319 |

- TRUE vs BAG reduction：`0.2589`；
- TRUE vs SHUFFLED reduction：`0.1255`；
- 3-fold simultaneous wins：`[True, True, True]`。

## 2. Relabel + resampling stability

| branch | cosine | relative L2 |
|---|---:|---:|
| INVARIANT_BAG | 0.9988 | 0.0488 |
| INVARIANT_TRUE_RELATION | 0.9983 | 0.0557 |
| INVARIANT_SHUFFLED_RELATION | 0.9982 | 0.0561 |

Registered checks：`{'true_vs_bag_gain_at_least_002': True, 'true_vs_shuffled_gain_at_least_002': True, 'true_better_both_all_folds': True, 'true_embedding_cosine_at_least_090': True, 'invariants': True}`。

## 3. 解释边界

该分支证明的是 ID-free local structural tokens 能否保留 relation signal。descriptor 是手工 control；后续 learned encoder 必须在相同 relabel 和 shuffled gates 下比较。
