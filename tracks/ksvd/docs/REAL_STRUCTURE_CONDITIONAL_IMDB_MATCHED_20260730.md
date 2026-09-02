# 控制 size/degree 后的 KSVD 条件增益审计（2026-07-30）

> 核心问题：patch/字典特征是否提供超出 graph size、density、degree histogram、triangle、component 等简单统计的信息。所有字典/PCA 仅在 outer-train 拟合；分类器 C 仅用 outer-train 的 inner CV 选择。

## 协议

- outer CV：5-fold × seeds [0, 1, 2]；inner CV：3-fold。
- atoms=16，T=2，KSVD iterations=10，C grid=[0.01, 0.1, 1.0, 10.0, 100.0]。
- residual 特征：先用 outer-train 上的 Ridge(alpha=1.0) 从 patch block 中线性回归掉 stats，再分类。
- `stats_plus_*` 是主要条件增益判断；`residual_*` 是辅助诊断，不应单独替代 paired delta。

## IMDB-BINARY/cleaned/B0

图数：493；类别：`{'0': 261, '1': 232}`。

stats+degree baseline balanced accuracy：**0.7533**。

| 新增表征 | stats+feature | paired Δ vs stats | fold W/T/L | seed wins | residual-only |
|---|---:|---:|---:|---:|---:|
| raw_patch_mean | 0.7668 | +0.0135 | 8/0/7 | 3/3 | 0.5586 |
| pca_content | 0.7549 | +0.0016 | 6/1/8 | 2/3 | 0.5595 |
| random_dictionary_content | 0.7488 | -0.0045 | 8/0/7 | 1/3 | 0.5270 |
| real_patch_dictionary_content | 0.7544 | +0.0011 | 8/0/7 | 1/3 | 0.5393 |
| ksvd_content | 0.7479 | -0.0055 | 7/0/8 | 1/3 | 0.5480 |
| relation_graph_only | 0.7466 | -0.0067 | 5/0/10 | 1/3 | 0.4905 |
| raw_patch_mean_graph | 0.7654 | +0.0120 | 9/0/6 | 3/3 | 0.5507 |
| pca_content_graph | 0.7530 | -0.0003 | 9/0/6 | 1/3 | 0.5605 |
| random_dictionary_content_graph | 0.7520 | -0.0013 | 6/1/8 | 2/3 | 0.5270 |
| real_patch_dictionary_content_graph | 0.7590 | +0.0057 | 9/1/5 | 2/3 | 0.5474 |
| ksvd_content_graph | 0.7600 | +0.0067 | 9/0/6 | 3/3 | 0.5376 |
| ksvd_content_true_relation | 0.7413 | -0.0120 | 4/0/11 | 1/3 | 0.5248 |
| ksvd_content_shuffled_relation | 0.7298 | -0.0235 | 5/0/10 | 0/3 | 0.5470 |

- true relation − shuffled relation：+0.0115，fold wins=12/15。
- KSVD content − PCA：-0.0070，fold wins=7/15，seed means=[-0.0108, -0.0164, 0.0062]。
- KSVD content − real-patch dictionary：-0.0065，fold wins=5/15，seed means=[-0.0251, -0.0087, 0.0142]。
- KSVD content+graph − raw+same graph：-0.0053，fold wins=7/15，seed means=[0.0021, -0.0135, -0.0046]。
- KSVD content+graph − PCA+same graph：+0.0070，fold wins=7/15，seed means=[0.0195, -0.0083, 0.0097]。
- KSVD content+graph − real-patch+same graph：+0.0010，fold wins=7/15，seed means=[0.0147, -0.0106, -0.001]。
- true atom-position relation − content+graph：-0.0187，fold wins=2/15，seed means=[-0.0247, 0.0132, -0.0446]。
- 解释时优先看 paired delta 是否跨 split seeds 同方向，并要求 KSVD 同时优于 raw/PCA/real-patch，而不是只优于 stats。

## IMDB-MULTI/cleaned/B0

图数：321；类别：`{'0': 144, '1': 85, '2': 92}`。

stats+degree baseline balanced accuracy：**0.5009**。

| 新增表征 | stats+feature | paired Δ vs stats | fold W/T/L | seed wins | residual-only |
|---|---:|---:|---:|---:|---:|
| raw_patch_mean | 0.5463 | +0.0454 | 10/0/5 | 3/3 | 0.4249 |
| pca_content | 0.5164 | +0.0155 | 10/0/5 | 3/3 | 0.4419 |
| random_dictionary_content | 0.5420 | +0.0411 | 11/0/4 | 3/3 | 0.4402 |
| real_patch_dictionary_content | 0.5538 | +0.0529 | 10/0/5 | 3/3 | 0.4373 |
| ksvd_content | 0.5372 | +0.0362 | 9/0/6 | 3/3 | 0.4485 |
| relation_graph_only | 0.5055 | +0.0045 | 9/0/6 | 2/3 | 0.3425 |
| raw_patch_mean_graph | 0.5443 | +0.0433 | 9/0/6 | 3/3 | 0.4310 |
| pca_content_graph | 0.5405 | +0.0396 | 10/0/5 | 3/3 | 0.4367 |
| random_dictionary_content_graph | 0.5471 | +0.0462 | 12/0/3 | 3/3 | 0.4447 |
| real_patch_dictionary_content_graph | 0.5545 | +0.0536 | 10/0/5 | 3/3 | 0.4323 |
| ksvd_content_graph | 0.5570 | +0.0561 | 12/0/3 | 3/3 | 0.4549 |
| ksvd_content_true_relation | 0.5211 | +0.0201 | 7/0/8 | 3/3 | 0.4280 |
| ksvd_content_shuffled_relation | 0.4980 | -0.0029 | 7/0/8 | 1/3 | 0.4275 |

- true relation − shuffled relation：+0.0230，fold wins=11/15。
- KSVD content − PCA：+0.0207，fold wins=7/15，seed means=[0.0356, 0.0144, 0.0121]。
- KSVD content − real-patch dictionary：-0.0166，fold wins=6/15，seed means=[-0.0339, -0.0134, -0.0026]。
- KSVD content+graph − raw+same graph：+0.0127，fold wins=8/15，seed means=[0.0022, 0.0263, 0.0097]。
- KSVD content+graph − PCA+same graph：+0.0165，fold wins=9/15，seed means=[0.0092, 0.0337, 0.0066]。
- KSVD content+graph − real-patch+same graph：+0.0025，fold wins=7/15，seed means=[0.0129, -0.0133, 0.0079]。
- true atom-position relation − content+graph：-0.0360，fold wins=3/15，seed means=[-0.0519, -0.0362, -0.0197]。
- 解释时优先看 paired delta 是否跨 split seeds 同方向，并要求 KSVD 同时优于 raw/PCA/real-patch，而不是只优于 stats。

## 判断口径

1. `stats_plus_*` 相对 stats 的正增益，只能说明该 patch block 含额外信息；若 raw/PCA/real-patch 同样或更强，不能归因于 KSVD。
2. `true relation > shuffled relation` 只是必要条件；还必须比较 `true relation` 与较低容量的 `content+graph`，否则可能只是 shuffle 更坏，而不是 exact binding 真正有益。
3. residual-only 接近随机不否定联合条件信息，但若 stats+feature 也无稳定增益，则该表征没有通过当前 conditional test。
4. 重复 CV folds 不是独立样本；结论以 split-seed 方向一致性和跨数据集复现为主，不把 fold 数当显著性样本量。

