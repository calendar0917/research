# Rooted-canonical KSVD：直接统计 vs patch readout

> 日期：2026-08-02  
> 协议：`tracks/ksvd/docs/KSVD_ROOTED_CANONICAL_DOWNSTREAM_ATTRIBUTION_PROTOCOL_20260802.md`  
> 判定：`DIRECT_STATS_SUFFICIENT_FOR_SYNTHETIC_FACTORS`

所有 branches 均使用 train-only standardization + 12D PCA + fixed ridge。

| branch | joint-9 balanced acc | family acc | degree acc | relabel-resample cosine |
|---|---:|---:|---:|---:|
| GLOBAL_STATS | 0.7901 | 1.0000 | 0.8148 | 1.0000 |
| ROOTED_RAW_BAG | 0.4259 | 0.6543 | 0.8704 | 0.6363 |
| ROOTED_RAW_TRUE_RELATION | 0.4753 | 0.7099 | 0.8148 | 0.6987 |
| ROOTED_KSVD_BAG | 0.4012 | 0.6049 | 0.5988 | 0.3666 |
| ROOTED_KSVD_TRUE_RELATION | 0.4198 | 0.5988 | 0.6975 | 0.4238 |
| ROOTED_KSVD_SHUFFLED_RELATION | 0.4136 | 0.5679 | 0.7160 | 0.4246 |

Joint deltas：`{'ksvd_true_minus_global_stats': -0.3703703703703704, 'ksvd_true_minus_ksvd_bag': 0.018518518518518434, 'ksvd_true_minus_ksvd_shuffled': 0.006172839506172756, 'ksvd_true_minus_raw_true': -0.05555555555555558}`。  
TRUE relation fold wins：`[False, False, False]`。  
Registered checks：`{'ksvd_true_vs_stats_at_least_002': False, 'ksvd_true_vs_bag_at_least_002': False, 'ksvd_true_vs_shuffled_at_least_002': False, 'ksvd_true_better_bag_shuffled_all_folds': False, 'all_invariants': True}`。

解释：GLOBAL_STATS 回答这些 synthetic generation factors 是否已由宏观统计充分决定；RAW/KSVD BAG 比较局部token统计；TRUE/SHUFFLED 比较正确 patch binding；RAW TRUE/KSVD TRUE 比较 dictionary 是否在同一关系机制下增加可线性读取信号。
