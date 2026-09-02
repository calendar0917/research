# 控制 size/degree 后的 KSVD 条件增益审计（2026-07-30）

> 核心问题：patch/字典特征是否提供超出 graph size、density、degree histogram、triangle、component 等简单统计的信息。所有字典/PCA 仅在 outer-train 拟合；分类器 C 仅用 outer-train 的 inner CV 选择。

## 协议

- outer CV：5-fold × seeds [0, 1, 2]；inner CV：3-fold。
- atoms=16，T=2，KSVD iterations=10，C grid=[0.01, 0.1, 1.0, 10.0, 100.0]。
- residual 特征：先用 outer-train 上的 Ridge(alpha=1.0) 从 patch block 中线性回归掉 stats，再分类。
- `stats_plus_*` 是主要条件增益判断；`residual_*` 是辅助诊断，不应单独替代 paired delta。

## REDDIT-BINARY/raw/B0

图数：2000；类别：`{'0': 1000, '1': 1000}`。

stats+degree baseline balanced accuracy：**0.8432**。

| 新增表征 | stats+feature | paired Δ vs stats | fold W/T/L | seed wins | residual-only |
|---|---:|---:|---:|---:|---:|
| raw_patch_mean | 0.8143 | -0.0288 | 0/0/15 | 0/3 | 0.6567 |
| pca_content | 0.8252 | -0.0180 | 1/1/13 | 0/3 | 0.5315 |
| random_dictionary_content | 0.8460 | +0.0028 | 9/2/4 | 3/3 | 0.5247 |
| real_patch_dictionary_content | 0.8377 | -0.0055 | 4/1/10 | 0/3 | 0.5292 |
| ksvd_content | 0.8332 | -0.0100 | 4/3/8 | 0/3 | 0.5247 |
| relation_graph_only | 0.8575 | +0.0143 | 12/2/1 | 3/3 | 0.5765 |
| ksvd_content_graph | 0.8375 | -0.0057 | 9/0/6 | 0/3 | 0.5550 |
| ksvd_content_true_relation | 0.7965 | -0.0467 | 0/0/15 | 0/3 | 0.6045 |
| ksvd_content_shuffled_relation | 0.7918 | -0.0513 | 0/0/15 | 0/3 | 0.6182 |

- true relation − shuffled relation：+0.0047，fold wins=10/15。
- KSVD content − PCA：+0.0080，fold wins=12/15，seed means=[0.0085, 0.007, 0.0085]。
- KSVD content − real-patch dictionary：-0.0045，fold wins=5/15，seed means=[-0.008, -0.0015, -0.004]。
- KSVD content+graph − raw patch：+0.0232，fold wins=13/15，seed means=[0.023, 0.024, 0.0225]。
- KSVD content+graph − real-patch dictionary：-0.0002，fold wins=8/15，seed means=[-0.003, 0.0005, 0.002]。
- true atom-position relation − content+graph：-0.0410，fold wins=1/15，seed means=[-0.037, -0.04, -0.046]。
- 解释时优先看 paired delta 是否跨 split seeds 同方向，并要求 KSVD 同时优于 raw/PCA/real-patch，而不是只优于 stats。

## 判断口径

1. `stats_plus_*` 相对 stats 的正增益，只能说明该 patch block 含额外信息；若 raw/PCA/real-patch 同样或更强，不能归因于 KSVD。
2. `true relation > shuffled relation` 只是必要条件；还必须比较 `true relation` 与较低容量的 `content+graph`，否则可能只是 shuffle 更坏，而不是 exact binding 真正有益。
3. residual-only 接近随机不否定联合条件信息，但若 stats+feature 也无稳定增益，则该表征没有通过当前 conditional test。
4. 重复 CV folds 不是独立样本；结论以 split-seed 方向一致性和跨数据集复现为主，不把 fold 数当显著性样本量。

