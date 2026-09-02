# Rooted-canonical KSVD：直接统计 vs patch readout

> 日期：2026-08-02  
> 协议：`tracks/ksvd/docs/KSVD_ROOTED_CANONICAL_DOWNSTREAM_ATTRIBUTION_PROTOCOL_20260802.md`  
> 判定：`DIRECT_STATS_SUFFICIENT_FOR_SYNTHETIC_FACTORS`

所有 branches 均使用 train-only standardization + 12D PCA + fixed ridge。

| branch | joint-9 balanced acc | family acc | degree acc | relabel-resample cosine |
|---|---:|---:|---:|---:|
| GLOBAL_STATS | 0.8210 | 1.0000 | 0.8333 | 1.0000 |
| ROOTED_RAW_BAG | 0.4753 | 0.6790 | 0.7346 | 0.6534 |
| ROOTED_RAW_TRUE_RELATION | 0.6605 | 0.7222 | 0.8333 | 0.7111 |
| ROOTED_KSVD_BAG | 0.4259 | 0.5864 | 0.6481 | 0.3869 |
| ROOTED_KSVD_TRUE_RELATION | 0.4815 | 0.5556 | 0.7284 | 0.4270 |
| ROOTED_KSVD_SHUFFLED_RELATION | 0.4259 | 0.5802 | 0.7037 | 0.4330 |

Joint deltas：`{'ksvd_true_minus_global_stats': -0.3395061728395062, 'ksvd_true_minus_ksvd_bag': 0.05555555555555558, 'ksvd_true_minus_ksvd_shuffled': 0.05555555555555558, 'ksvd_true_minus_raw_true': -0.1790123456790123}`。  
TRUE relation fold wins：`[True, False, True]`。  
Registered checks：`{'ksvd_true_vs_stats_at_least_002': False, 'ksvd_true_vs_bag_at_least_002': True, 'ksvd_true_vs_shuffled_at_least_002': True, 'ksvd_true_better_bag_shuffled_all_folds': False, 'all_invariants': True}`。

解释：GLOBAL_STATS 回答这些 synthetic generation factors 是否已由宏观统计充分决定；RAW/KSVD BAG 比较局部token统计；TRUE/SHUFFLED 比较正确 patch binding；RAW TRUE/KSVD TRUE 比较 dictionary 是否在同一关系机制下增加可线性读取信号。
