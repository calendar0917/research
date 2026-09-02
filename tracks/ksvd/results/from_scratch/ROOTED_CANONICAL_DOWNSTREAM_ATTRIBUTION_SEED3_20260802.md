# Rooted-canonical KSVD：直接统计 vs patch readout

> 日期：2026-08-02  
> 协议：`tracks/ksvd/docs/KSVD_ROOTED_CANONICAL_DOWNSTREAM_ATTRIBUTION_PROTOCOL_20260802.md`  
> 判定：`DIRECT_STATS_SUFFICIENT_FOR_SYNTHETIC_FACTORS`

所有 branches 均使用 train-only standardization + 12D PCA + fixed ridge。

| branch | joint-9 balanced acc | family acc | degree acc | relabel-resample cosine |
|---|---:|---:|---:|---:|
| GLOBAL_STATS | 0.7901 | 1.0000 | 0.8642 | 1.0000 |
| ROOTED_RAW_BAG | 0.5494 | 0.6235 | 0.7654 | 0.6135 |
| ROOTED_RAW_TRUE_RELATION | 0.6049 | 0.7531 | 0.8210 | 0.6825 |
| ROOTED_KSVD_BAG | 0.4136 | 0.5988 | 0.5926 | 0.2592 |
| ROOTED_KSVD_TRUE_RELATION | 0.4568 | 0.5926 | 0.6852 | 0.3515 |
| ROOTED_KSVD_SHUFFLED_RELATION | 0.3951 | 0.5802 | 0.6728 | 0.3453 |

Joint deltas：`{'ksvd_true_minus_global_stats': -0.3333333333333333, 'ksvd_true_minus_ksvd_bag': 0.043209876543209846, 'ksvd_true_minus_ksvd_shuffled': 0.06172839506172839, 'ksvd_true_minus_raw_true': -0.14814814814814808}`。  
TRUE relation fold wins：`[False, True, True]`。  
Registered checks：`{'ksvd_true_vs_stats_at_least_002': False, 'ksvd_true_vs_bag_at_least_002': True, 'ksvd_true_vs_shuffled_at_least_002': True, 'ksvd_true_better_bag_shuffled_all_folds': False, 'all_invariants': True}`。

解释：GLOBAL_STATS 回答这些 synthetic generation factors 是否已由宏观统计充分决定；RAW/KSVD BAG 比较局部token统计；TRUE/SHUFFLED 比较正确 patch binding；RAW TRUE/KSVD TRUE 比较 dictionary 是否在同一关系机制下增加可线性读取信号。
