# 不使用 WL 的 exact canonical patch 审计（2026-07-30）

> 目的：区分 KSVD 的失败是否可能由当前 WL histogram 的信息损失导致。Canonical 表示用 nauty 规范化完整邻接矩阵，不依赖节点编号；rooted canonical 额外保留 sampler center。

## 表示定义

- `WL`：当前 degree histogram + degree-pair histogram + 3轮 hashed 1-WL histogram。
- `canonical`：完整 induced adjacency 的 exact canonical labeling + node mask；不使用 WL 作为最终特征。
- `rooted canonical`：canonical graph 同时把 patch center 作为 singleton color，保留中心在子图中的角色。
- 多指标排序不是严格替代：只要仍有 tie 并由 node id 决定，表示就不是置换不变；canonical labeling 才能在保留完整邻接的同时解决该问题。

分类协议：5-fold × seeds [0, 1, 2]，inner 3-fold；此处先比较 raw patch mean，不训练 KSVD。

## IMDB-BINARY/cleaned/B0/max_nodes=12

### 信息损失审计

| coarse → exact | coarse signatures | exact signatures | ambiguous occurrences | exact types affected | max exact/coarse |
|---|---:|---:|---:|---:|---:|
| WL → exact unrooted | 311 | 311 | 0.000 | 0.000 | 1 |
| unrooted → exact rooted | 311 | 311 | 0.000 | 0.000 | 1 |
| WL → exact rooted | 311 | 311 | 0.000 | 0.000 | 1 |

### 控制 stats 后的 raw patch mean

| 表示 | balanced accuracy | seed means |
|---|---:|---:|
| stats only | 0.7533 | [0.7642, 0.7438, 0.7521] |
| WL raw | 0.7668 | [0.7725, 0.7595, 0.7685] |
| canonical raw | 0.7440 | [0.759, 0.7229, 0.7501] |
| rooted canonical raw | 0.7502 | [0.7443, 0.744, 0.7623] |

- `canonical − WL`：-0.0228，fold W/T/L=4/1/10，seed means=[-0.0134, -0.0366, -0.0185]。
- `rooted_canonical − WL`：-0.0166，fold W/T/L=5/0/10，seed means=[-0.0281, -0.0156, -0.0062]。

## IMDB-MULTI/cleaned/B0/max_nodes=12

### 信息损失审计

| coarse → exact | coarse signatures | exact signatures | ambiguous occurrences | exact types affected | max exact/coarse |
|---|---:|---:|---:|---:|---:|
| WL → exact unrooted | 253 | 253 | 0.000 | 0.000 | 1 |
| unrooted → exact rooted | 253 | 253 | 0.000 | 0.000 | 1 |
| WL → exact rooted | 253 | 253 | 0.000 | 0.000 | 1 |

### 控制 stats 后的 raw patch mean

| 表示 | balanced accuracy | seed means |
|---|---:|---:|
| stats only | 0.5009 | [0.5188, 0.4973, 0.4867] |
| WL raw | 0.5463 | [0.5714, 0.5487, 0.5188] |
| canonical raw | 0.4978 | [0.5238, 0.4887, 0.4809] |
| rooted canonical raw | 0.5128 | [0.5444, 0.4919, 0.5022] |

- `canonical − WL`：-0.0485，fold W/T/L=4/0/11，seed means=[-0.0476, -0.0601, -0.038]。
- `rooted_canonical − WL`：-0.0335，fold W/T/L=4/0/11，seed means=[-0.027, -0.0569, -0.0166]。

## REDDIT-BINARY/raw/R2/max_nodes=24

### 信息损失审计

| coarse → exact | coarse signatures | exact signatures | ambiguous occurrences | exact types affected | max exact/coarse |
|---|---:|---:|---:|---:|---:|
| WL → exact unrooted | 8167 | 8167 | 0.000 | 0.000 | 1 |
| unrooted → exact rooted | 8167 | 8598 | 0.510 | 0.090 | 6 |
| WL → exact rooted | 8167 | 8598 | 0.510 | 0.090 | 6 |

### 控制 stats 后的 raw patch mean

| 表示 | balanced accuracy | seed means |
|---|---:|---:|
| stats only | 0.8432 | [0.845, 0.845, 0.8395] |
| WL raw | 0.8202 | [0.818, 0.82, 0.8225] |
| canonical raw | 0.8160 | [0.8235, 0.8125, 0.812] |
| rooted canonical raw | 0.8115 | [0.816, 0.8075, 0.811] |

- `canonical − WL`：-0.0042，fold W/T/L=5/1/9，seed means=[0.0055, -0.0075, -0.0105]。
- `rooted_canonical − WL`：-0.0087，fold W/T/L=5/0/10，seed means=[-0.002, -0.0125, -0.0115]。

## 决策规则

- 若 canonical 显著减少 collision，且 raw 表征跨 seeds 稳定优于 WL，则再运行完整 projected-KSVD。
- 若 exact canonical 仍不优于 WL，则不能继续把 KSVD 失败主要归因于 WL 信息损失。
- rooted 优于 unrooted 时，问题更可能是当前表示丢失 patch center，而不只是 1-WL 表达力不足。
