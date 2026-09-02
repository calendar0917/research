# MolHIV exact topology × conditional chemistry 快速机制筛选

日期：2026-09-01  
最终协议：`luyin16-molhiv-exact-topology-conditional-chemistry-normalized-v1`

## 问题

本轮不再问 K-SVD reconstruction，而是把“结构—属性融合”拆成三个可证伪层级：

1. 完整 rooted topology 是否比当前 rooted-WL 局部结构表示更有用；
2. 属性落在正确 structural automorphism orbit 上是否重要；
3. 若精确位置不重要，同一个 patch 内的 topology 与 attribute multiset 配对是否重要。

第三层对应一种更接近导师“先解耦、后融合”主张的对象：结构通道表示
`T`，属性通道表示 `A`，融合通道只建模同一 patch 内的条件关系
`P(A | T)`，而不是把两个全图统计向量直接拼接。

## 协议

- 数据：MolHIV official-train 内固定 3 个 scaffold folds；每折 `1200/600`。
- 分类器：固定 XGBoost；model seeds `0/1/2`。
- 控制：2 个 matched shuffle repeats。
- official-valid/test：均未编码、未评估。
- patch：每个节点的完整 radius-2 induced ego。
- structure：root-colored incidence graph 的 nauty exact certificate。
- position：rooted structural automorphism 的 node/edge orbits。
- attributes：40D strict atom semantics（排除 degree/ring）与 13D bond semantics。
- vocabulary：每个 outer-train fold 的 top-32 exact topology，train/valid patch mass
  覆盖均值分别为 `94.46%/93.95%`。
- prototype：每种 topology 最多 4 个 deterministic farthest empirical templates；
  不使用随机 INIT，不运行 K-SVD update。

两种 shuffle 的含义不同：

- `orbit shuffle`：在同一个 patch 内打乱 node/edge attributes，保留 topology 与
  patch attribute multiset，只破坏属性落在哪个 orbit；
- `patch-pair shuffle`：在同一张图内错配 topology patch 与 attribute patch，保留
  topology bag 和 attribute-patch bag，只破坏二者属于同一个 patch 的事实。

## 审计

- node relabel invariance：通过；exact patch multiset、两个 shuffle、WL readout 与
  context 均保持一致。
- 三折 valid patch 中，任一 exact type 在对应 outer-train 出现过的 patch mass 均值为
  `99.47%`；长尾主要是 type 数量，不是 patch 质量。
- 在本轮每折约 `294--328` 个观测局部类型上，rooted-WL full key 与实际 96D
  binned patch vector 对 exact topology 的碰撞均为 `0`。这不是一般性的 WL 完备性
  证明，但说明 radius-2 MolHIV 当前样本里没有观察到 WL 信息瓶颈。

## 主要结果

| view | train-fold validation AUC | 关键比较 |
|---|---:|---|
| `S + WL structure` | `0.6977` | — |
| `S + exact top-32 structure` | `0.6858` | exact − WL = `-1.19pt` |
| `S + WL + attribute` | **`0.7145`** | 当前最强未绑定基线 |
| `S + exact + attribute` | `0.6840` | exact-unbound − WL-unbound = `-3.06pt` |
| `S + exact + attribute + raw orbit` | `0.6821` | true − orbit-shuffle = `-0.90pt` |
| `S + WL + exact + attribute + orbit prototype` | `0.7113` | 相对同 base `+1.59pt`，但 true − orbit-shuffle = `-0.38pt` |
| `S + exact + attribute + conditional prototype` | `0.6891` | true − patch-pair-shuffle = **`+2.36pt`** |
| `S + WL + exact + attribute + conditional prototype` | `0.7021` | 相对同 base `+0.66pt`；true − patch-pair-shuffle = **`+1.35pt`** |
| `S + WL + attribute + conditional mean` | `0.6983` | 相对最强 unbound `-1.63pt` |

更细的 `seed × fold` 结果：

- exact conditional true − patch-pair shuffle：均值 `+2.36pt`，`8/9` wins，
  三个 fold 的均值都为正；
- hybrid conditional true − patch-pair shuffle：均值 `+1.35pt`，`6/9` wins，
  `2/3` folds 为正；
- hybrid conditional − hybrid unbound：均值 `+0.66pt`，`7/9` wins，
  但 fold 2 为负；
- topology-normalized `P(A|T)` true − shuffle 仍为正，但相对最强
  `S+WL+attribute` 基线三折均下降。

## 结论

### 1. WL 不是当前主要问题

rooted-WL 理论上会碰撞，但本轮观测的 radius-2 分子 patch 中没有检测到这种
碰撞；直接换 exact topology 没有提高分类。当前不能把后续缺口归因于“WL 丢了
大量局部拓扑”。exact certificate 仍适合做审计和条件分组，但不应替换 WL 主结构
通道或继续扩大 exact vocabulary。

### 2. 精确 orbit 位置绑定被否定

7.5k--7.7kD raw orbit 上界和 256D orbit prototype 都不能稳定胜 matched
within-patch shuffle。属性具体落在哪个 structural orbit 上，并不是当前 MolHIV
radius-2/XGBoost 路线的有效增量来源。停止 orbit 数、prototype 数、K-SVD 或
readout 的扫描。

### 3. patch 级 topology--attribute 配对是真实机制信号

只要把“哪个 attribute patch 属于哪个 topology patch”打乱，AUC 会稳定下降；
该结果在 8/9 个 seed×fold 上复现。因此有效融合层级不是 node-slot/orbit，
也不是两个全图 marginal 的简单拼接，而是：

```text
exact/local topology T
        +
local attribute multiset A
        ↓
topology-conditioned chemistry response  P(A | T)
```

这是本轮最重要的信息增量。

### 4. 机制存在，但当前 joint XGBoost 读出尚未转成最强预测增量

conditional branch 能区分 true 与 patch-pair shuffle，却仍低于最强
`S+WL+attribute=0.7145`。因此当前状态应表述为：

> **patch-level conditional fusion 有可复现机制证据，但直接特征拼接会与强边际
> 分支冲突或重复，尚未形成可晋级的预测方案。**

不能据此进入 official-valid/test，也不能开始 Optuna、topology-K、prototype 数或
XGBoost 深度搜索。

## 唯一建议的下一步

只做一次 cross-fitted conditional residual gate：

1. 冻结 base：`S + WL + attribute`；
2. 在 outer-train 内生成 base 的 OOF logits；
3. conditional expert 只读取 topology-conditioned chemistry response，拟合 base
   未解释的 residual；
4. outer-valid 使用 `base_logit + alpha * residual_logit`，`alpha` 只能在 inner-train
   确定，不能看 outer-valid；
5. matched control 使用 patch-pair-shuffled conditional response，保持完全相同容量。

晋级必须同时满足：

- true residual 相对 frozen base `>= +0.3pt` 且至少 `2/3` folds 为正；
- true residual 相对 shuffled residual `>= +0.3pt` 且至少 `2/3` folds 为正；
- 改善不能只由 fold 0 或单一 model seed 驱动。

若该 residual gate 失败，则应停止“统计特征 + XGBoost”的结构—属性融合路线；
后续若仍继续，只能明确改变研究命题，使用 node/edge-level 可学习局部 encoder，
而不是继续扩展统计量。

## 复现入口

- 实验：`tracks/ksvd/experiments/luyin16/exact_orbit_fusion_screen.py`
- 最终配置：`tracks/ksvd/configs/luyin16/exact_conditional_fusion_screen_normalized.yaml`
- 原始摘要：`tracks/ksvd/results/luyin16/exact_conditional_fusion_screen_normalized/summary.json`
- 自动 Markdown：`tracks/ksvd/results/luyin16/exact_conditional_fusion_screen_normalized/summary.md`
- 单测：`tracks/ksvd/tests/test_exact_orbit_fusion_screen.py`
