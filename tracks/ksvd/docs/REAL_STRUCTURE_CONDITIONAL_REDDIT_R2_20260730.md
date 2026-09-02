# 控制 size/degree 后的 KSVD 条件增益审计（2026-07-30）

> 核心问题：patch/字典特征是否提供超出 graph size、density、degree histogram、triangle、component 等简单统计的信息。所有字典/PCA 仅在 outer-train 拟合；分类器 C 仅用 outer-train 的 inner CV 选择。

## 协议

- outer CV：5-fold × seeds [0, 1, 2]；inner CV：3-fold。
- atoms=16，T=2，KSVD iterations=10，C grid=[0.01, 0.1, 1.0, 10.0, 100.0]。
- residual 特征：先用 outer-train 上的 Ridge(alpha=1.0) 从 patch block 中线性回归掉 stats，再分类。
- `stats_plus_*` 是主要条件增益判断；`residual_*` 是辅助诊断，不应单独替代 paired delta。

## REDDIT-BINARY/raw/R2

图数：2000；类别：`{'0': 1000, '1': 1000}`。

stats+degree baseline balanced accuracy：**0.8432**。

| 新增表征 | stats+feature | paired Δ vs stats | fold W/T/L | seed wins | residual-only |
|---|---:|---:|---:|---:|---:|
| raw_patch_mean | 0.8212 | -0.0220 | 1/0/14 | 0/3 | 0.5592 |
| pca_content | 0.8432 | -0.0000 | 8/0/7 | 2/3 | 0.5538 |
| random_dictionary_content | 0.8438 | +0.0007 | 7/1/7 | 2/3 | 0.5582 |
| real_patch_dictionary_content | 0.8368 | -0.0063 | 6/1/8 | 0/3 | 0.5467 |
| ksvd_content | 0.8410 | -0.0022 | 8/0/7 | 0/3 | 0.5572 |
| relation_graph_only | 0.8610 | +0.0178 | 14/1/0 | 3/3 | 0.5707 |
| ksvd_content_graph | 0.8488 | +0.0057 | 9/1/5 | 3/3 | 0.5752 |
| ksvd_content_true_relation | 0.8137 | -0.0295 | 0/0/15 | 0/3 | 0.5675 |
| ksvd_content_shuffled_relation | 0.8058 | -0.0373 | 1/0/14 | 0/3 | 0.5565 |

- true relation − shuffled relation：+0.0078，fold wins=11/15。
- KSVD content − PCA：-0.0022，fold wins=6/15，seed means=[-0.0035, 0.0015, -0.0045]。
- KSVD content − real-patch dictionary：+0.0042，fold wins=7/15，seed means=[0.0015, 0.013, -0.002]。
- KSVD content+graph − raw patch：+0.0277，fold wins=15/15，seed means=[0.0325, 0.026, 0.0245]。
- KSVD content+graph − real-patch dictionary：+0.0120，fold wins=13/15，seed means=[0.009, 0.0145, 0.0125]。
- true atom-position relation − content+graph：-0.0352，fold wins=0/15，seed means=[-0.0435, -0.026, -0.036]。
- 解释时优先看 paired delta 是否跨 split seeds 同方向，并要求 KSVD 同时优于 raw/PCA/real-patch，而不是只优于 stats。

## 判断口径

1. `stats_plus_*` 相对 stats 的正增益，只能说明该 patch block 含额外信息；若 raw/PCA/real-patch 同样或更强，不能归因于 KSVD。
2. `true relation > shuffled relation` 只是必要条件；还必须比较 `true relation` 与较低容量的 `content+graph`，否则可能只是 shuffle 更坏，而不是 exact binding 真正有益。
3. residual-only 接近随机不否定联合条件信息，但若 stats+feature 也无稳定增益，则该表征没有通过当前 conditional test。
4. 重复 CV folds 不是独立样本；结论以 split-seed 方向一致性和跨数据集复现为主，不把 fold 数当显著性样本量。

