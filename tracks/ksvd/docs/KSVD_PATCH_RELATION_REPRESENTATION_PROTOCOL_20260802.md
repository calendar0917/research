# Patch relations 是否形成有效图表示：最小审计协议

> 日期：2026-08-02  
> 状态：结果前冻结  
> 范围：无 graph labels；只检验 patch token、关系 binding、masked structural prediction 与重编号稳定性。

## 1. 核心命题

Beam8 已经提供连续、重叠且覆盖较好的 patches，但尚未证明 overlap、center distance 和 transition substrate 真正进入了可用表示。

本轮检验：

> 遮住一个 target patch 后，正确绑定到该 target 的其他 patch tokens，是否比无关系 bag 和同容量 shuffled binding 更能预测 target patch 的置换不变结构。

若不能，当前 continuous-cover 仍只是 sampler；若能，才能把 patch relations 升格为 graph representation substrate。

## 2. 数据与冻结设置

- 72 个 50-node synthetic graphs；
- `regular/small_world/block × degree 15/20/25 × 8 replicates`；
- graph bank seed `810001`；
- Beam8/R1，`s=10,o=3,m=1.5`；cover seed `930101`；
- 3-fold graph isolation，沿用 `graph_index % 8 % 3`；
- KSVD `K=24,T=3,updates=25`，dictionary 只在 train graphs 学习；
- ridge `alpha=0.01`，所有 masked rows 只由 train graphs 拟合。

## 3. Masked target

每个 patch 轮流作为 masked target。模型不能读取 target 的 raw vector 或 KSVD code，只读取其他 patches。

预测目标是 patch 的 permutation-invariant 22D structural descriptor：

```text
10 sorted normalized degrees
10 sorted normalized adjacency eigenvalues
edge density
triangle density
```

因此 construction slots 不会直接进入监督 target。

## 4. 四个分支

### RAW_BAG

对其他 raw 45D patch vectors 做 `mean/std/max` pooling。

### KSVD_BAG

对其他 K24 sparse codes 做 `mean/std/max` pooling。

### KSVD_TRUE_RELATION

在 KSVD_BAG 基础上，加入相对于 masked target 的两个 relation-weighted context：

1. overlap fraction weighted mean code；
2. `exp(-center shortest-path distance)` weighted mean code；

并加入 overlap/distance 的四个 scalar summaries。

### KSVD_SHUFFLED_RELATION

保留 target 的 relation weights、relation marginals、feature shape 和 graph 内 context tokens，只把 context-token binding 做确定性循环错位。它是 TRUE 的同容量负对照。

## 5. 主要指标与 gate

对 held-out graphs 先按 graph 汇总 masked rows，再做 fold-balanced mean：

- invariant-target RMSE；
- degree-block RMSE；
- spectrum-block RMSE；
- density/triangle RMSE。

TRUE relation gate：

1. TRUE 相对 KSVD_BAG overall RMSE 改善至少 `2%`；
2. TRUE 相对 SHUFFLED overall RMSE 改善至少 `2%`；
3. 3/3 folds TRUE RMSE 同时低于 BAG 和 SHUFFLED；
4. 所有 train/test graph isolation、finite、row-count invariants 通过。

判定：

- 全部通过：`PATCH_RELATIONS_ADD_MASKED_VALUE`；
- 均值方向为正但未过强 gate：`PATCH_RELATION_SIGNAL_BELOW_GATE`；
- TRUE 不优于 BAG 或 SHUFFLED：`REJECT_CURRENT_RELATION_FEATURES`。

RAW_BAG 与 KSVD_BAG 只回答压缩 token 的信息损失，不进入 relation 主 gate。

## 6. Relabel-resampling stability

对每 fold 的 held-out graphs 做一次随机 node relabel，并用相同 scalar sampler seed 重新运行 Beam8。分别构造四分支的 full-graph embeddings，报告：

- cosine similarity；
- relative L2 difference。

该诊断同时包含 sampler 与 token instability，不要求 exact chain replay。注册稳定阈值为 mean cosine `>=0.90`，但不用于把 relation added-value failure 改判为 pass。

## 7. 解释边界

- 这是低容量机制审计，不是最终 Transformer；
- TRUE 通过只说明 overlap/distance binding 含有 held-out structural signal；
- TRUE 失败不排除更强 message passing，但会否定“当前关系特征直接可用”；
- 本轮不包含分类标签，因此不能单独证明下游效用。
