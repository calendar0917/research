# luyin14 后续：uncompressed RAW patch relation terminal screen

> 日期：2026-08-12  
> 状态：结果前冻结

## 1. 研究问题

已有 synthetic graph-isolation 审计显示，真实 patch relation 对 masked structure 有信号，但
K24/T3 压缩会削弱 relation 下游可读性。luyin14 的 TRUE/SHUFFLED 分类只绑定了 KSVD
codes，尚未隔离未压缩 rooted-canonical adjacency patch token。

本轮只回答：

> TRUE patch relation 在不经过 KSVD 压缩时，是否能在真实四数据集上稳定优于
> SHUFFLED binding，并在简单 graph statistics 外提供增量？

## 2. 冻结设置

- 完全复用 FAIR95 patches、rooted-canonical slots、四数据集与 3 seeds × 3 folds；
- local token 是 8-node padded adjacency upper triangle（28D），不拟合字典；
- RAW bag：每个坐标的 mean/std/max（84D）；
- relation graph：现有 chain/overlap 两通道的低容量图统计；
- aligned relation：两通道上加权 token dot/cosine/L1/same-winner；
- SHUFFLED 只打乱 token 到 patch position 的绑定；
- 分类器仍为 train-only scaler + 固定 logistic regression。

必报：`STATS / RAW_BAG / RAW_RELATION_GRAPH / RAW_TRUE_RELATION /
RAW_SHUFFLED_RELATION / STATS_RAW_TRUE_RELATION`。MUTAG/PTC_MR 额外报告
`FEATURE_STATS` 与 `FEATURE_STATS_RAW_TRUE_RELATION`。

## 3. 判定

单数据集稳定通过要求 paired mean `>= +0.01` balanced accuracy 且 wins `>= 6/9`。

1. `RAW_TRUE - RAW_SHUFFLED` 至少 2/4 数据集稳定通过；
2. `RAW_TRUE - RAW_BAG` 至少 2/4 数据集稳定通过；
3. `STATS+RAW_TRUE - STATS` 至少 2/4 数据集稳定通过。

三个 gate 同时通过，才允许立项 relation-aware patch encoder；首版仍应是低容量 patch-graph
模型，并保留 TRUE/SHUFFLED、RAW/KSVD 与 stats controls。若 binding 通过但 added-value 失败，
关系只是在重复全图统计。若 binding 失败，则停止 Transformer/patch-graph 路线。

