# Beam8/NCI1 patch-chain classification Stage A protocol

> 日期：2026-08-13  
> 状态：结果前冻结  
> 数据：完整 TU NCI1；不构造数据集。

## 1. 研究问题

在同一条 Beam8 cover、同一组 patch token 和同一个线性分类头下，保留连续
patch chain、overlap 与 coverage progression，是否比 bag-of-patches 和打乱
token-to-position binding 更能支持图分类？若成立，收益来自 RAW/INIT 已有内容，
还是来自普通无监督 KSVD updates？

本轮不扫描 sampler geometry、字典容量、稀疏度、网络深度或分类器超参数。

## 2. 数据与 Beam8 cover

- 数据：完整 raw TU `NCI1`（4,110 graphs，二分类，37 类 node labels）。
- geometry：`s=10, o=3, multiplier=1.5`。
- sampler：marginal candidate `Beam8/R1`；`retained_beam=8`，
  `candidate_restarts=1`。
- operating point：每个连通分量采用冻结的 BASE budget；未覆盖真实边只进入相同的
  residual sidecar，不增加 EDGE100 completion patch。
- 非连通图：每个连通分量独立形成一个 segment；禁止伪造跨分量 chain edge。
- 节点预序：分量内 GLOBAL-WL stable order；patch 内 rooted-canonical slots。
- cover 完全不使用 graph labels，每张图在所有 outer folds 与所有分支中共用一次 cover。

## 3. Patch token

每个 patch 的 node-label histogram 与以下结构 token 拼接：

- `RAW`：45 维 canonical upper-triangle adjacency；
- `INIT`：outer-train-only deterministic maximin dictionary 的绝对 sparse code；
- `FINAL`：从同一 INIT 做普通无监督 KSVD updates 后的绝对 sparse code。

字典固定为 `K=24, T=3, T_min=1, iterations=5`。每个 outer-train fold 最多使用
3,000 个训练 patch；test patch 不进入均值、初始化、字典或 token normalization。

## 4. Beam8 relation 与低容量 message passing

关系图固定包含：

- `PREVIOUS`：同 segment 的前一 patch 到当前 patch；
- `NEXT`：同 segment 的后一 patch到当前 patch；
- `OVERLAP`：任意共享节点的 patch pair，权重为 node-set Jaccard；
- `SLOT`：overlap 再乘 canonical slot persistence。

每个通道只做一次无参数 row-normalized message passing。图级表示包含原 token、
传播后 token、二者逐维乘积/距离的低容量 moments，以及 new-node、new-edge、
segment 内 prefix coverage、segment-start 等冻结位置元数据。最终分类器统一为 train-only
`StandardScaler + LogisticRegression(C=1)`。

这一步不使用 Transformer、attention 或可学习 patch GNN。只有 Stage A 的真实关系
绑定通过后，才允许进入 1--2 层可学习 patch-graph GNN。

## 5. 必报对照

- `FEATURE_ONLY`、`FEATURE_STATS`；
- `RAW_BAG`、`RAW_CHAIN_TRUE`、`RAW_CHAIN_SHUFFLED`；
- `INIT_BAG`、`INIT_CHAIN_TRUE`；
- `FINAL_BAG`、`FINAL_CHAIN_TRUE`、`FINAL_CHAIN_SHUFFLED`。

`TRUE` 与 `SHUFFLED` 使用完全相同的 token multiset、patch graph、relation weights、
position metadata 和 residual sidecar；`SHUFFLED` 只打乱 token 到 patch position 的
绑定。单 patch 图不做虚假的 shuffle。

## 6. Evaluation 与冻结 gate

- Stage A：split seed `0`，3-fold stratified CV；主指标 balanced accuracy。
- 所有分支共用完全相同的 folds。
- Beam relation gate：`FINAL_CHAIN_TRUE - FINAL_CHAIN_SHUFFLED >= 1 point`，
  且至少 2/3 folds 为正；同时 `FINAL_CHAIN_TRUE` 平均不低于 `FINAL_BAG`。
- RAW substrate gate：`RAW_CHAIN_TRUE - RAW_CHAIN_SHUFFLED >= 1 point`，
  且至少 2/3 folds 为正。
- KSVD update gate：`FINAL_CHAIN_TRUE - INIT_CHAIN_TRUE >= 1 point`，
  且至少 2/3 folds 为正。

解释必须拆开：relation/RAW 通过但 KSVD gate 失败，表示 Beam8 chain 是有效 substrate，
不能解释为普通无监督 KSVD updates 对分类有效；三项均失败才停止扩大 Beam8 分类模型。
