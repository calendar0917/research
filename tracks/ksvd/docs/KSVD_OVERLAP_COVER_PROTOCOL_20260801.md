# KSVD 连续重叠 patch cover：第一轮协议

> 日期：2026-08-01  
> 状态：v0.1，正式全量结果不可见前冻结  
> 本轮不训练 KSVD、不做分类；只检验 patch 构造本身能否形成连续、可拼接的整图表示。

## 1. 研究问题

对固定 `n=50`、平均度约为 `15/20/25` 的图，比较三种同预算 patch 构造：

1. `independent_walk`：当前独立 root 随机游走基线；
2. `sliding_walk`：一条长的 coverage-prioritized walk，其首次发现节点序列按滑动窗口切成 patch；
3. `frontier_cover`：相邻 patch 保留固定 overlap，从当前边界补入优先覆盖未观察节点对的新节点。

本轮回答：

> 让采样过程本身具有相邻和重叠约束，能否在相同 patch 数量与大小下，提高连续性和整图 adjacency 的可恢复性？

## 2. 数据

使用三种不带标签的连通图族，避免结论只属于单一随机图模型：

- `regular`：随机 `d`-regular 图；
- `small_world`：Watts-Strogatz 风格 ring lattice，固定 degree 后随机重连；
- `block`：两个等大社区的 stochastic block 图，并通过拒绝采样保证连通。

固定：

```text
n_nodes = 50
target average degree = 15, 20, 25
graphs/family/degree/seed = 8
audit seeds = 810101, 810102, 810103
```

实际边数和平均度必须报告，不能假设生成器恰好达到目标。

## 3. Patch 与预算

```text
patch_size s = 10
target consecutive overlap b = 5
```

每张图的 patch 数量由信息容量下界决定：

\[
m_{edge}=\left\lceil\frac{|E|}{\binom{s}{2}}\right\rceil,
\qquad
m_{node}=1+\left\lceil\frac{|V|-s}{s-b}\right\rceil.
\]

正式预算为：

\[
m=\max(m_{node},\lceil1.5m_{edge}\rceil).
\]

该预算只是让完整恢复在容量上不再明显不可能，不保证一定覆盖全部边。三种方法在同一张图上必须使用相同 `m`。

## 4. 方法定义

### 4.1 independent walk

无放回选择 roots；每个 root 独立随机游走，按首次发现顺序收集 10 个不同节点，取完整 induced adjacency。若 `m>n`，roots 在每轮随机排列后循环使用。

### 4.2 sliding walk

生成一条 graph-level coverage-prioritized walk：

- 优先走向尚未发现的邻居；
- 没有未发现邻居时，走向与未发现区域距离最短的邻居；
- 若所有节点已经发现，则继续选择能增加未观察 node-pair 的邻居。

第一个 patch 收集 10 个不同节点。之后每个 patch 严格保留前一个 patch 后 5 个节点，并沿同一条 walk 继续寻找 5 个不属于前一个 patch 的节点。walk 可以经过旧节点，但只有满足当前窗口约束的新节点才进入 patch，因此相邻窗口恰好重叠 5 个节点。

### 4.3 frontier cover

第一个 patch 用 connected frontier expansion 构造。之后：

1. 从当前 patch 中选择 5 个 retained nodes；
2. retained set 优先包含连接到 patch 外的 boundary nodes；
3. 逐个加入与 retained/current patch 相邻的候选节点；
4. 候选评分优先增加尚未观察的 node pairs，其次增加未覆盖节点和未覆盖真实边；
5. 若局部 frontier 不足，使用到当前 patch 距离最近的全图节点补足。

每次保存相邻 patch 的精确 local-slot correspondence，而不只保存 overlap 数量。

## 5. 指标

### 5.1 覆盖与拼接

- node coverage；
- true-edge coverage：至少在一个 patch 内共同出现的真实边比例；
- node-pair coverage：至少在一个 patch 内共同出现的无序节点对比例；
- stitched adjacency accuracy：只对 observed pairs 聚合原始 induced patches；
- full adjacency accuracy：未观察 pair 默认预测为无边；
- edge observation multiplicity mean/CV。

原始 induced patch 在 observed pair 上应当严格一致；若不为 `1.0`，属于实现错误。

### 5.2 连续性

- consecutive overlap size/Jaccard；
- nonconsecutive overlap size/Jaccard；
- consecutive-minus-nonconsecutive gap；
- consecutive patch center shortest-path distance；
- 每一步新节点和新观察 node-pair 数。

### 5.3 稳定性

- mapped replay relabel invariance：固定抽象 patch 序列并映射到重编号图后，局部 adjacency 与 transition maps 必须完全一致；
- fresh resampling relabel set similarity：用相同 RNG seed 在重编号图上重新采样，比较映回原编号后的 patch sets；
- seed stability：同一图不同 sampler seeds 的 node-pair coverage 与 edge coverage 波动。

严格 invariance 只对 mapped replay 声明。fresh resampling 会受到结构 ties 和随机候选顺序影响，只作稳健性指标。

## 6. 主判定

`frontier_cover` 进入下一轮 KSVD stitched-reconstruction 的条件：

1. observed-pair consistency 在所有图上为 `1.0`；
2. mapped replay relabel invariance 在所有方法上为 `1.0`；
3. 相对 `independent_walk`，mean true-edge coverage 提高至少 `0.03`；
4. 相对 `independent_walk`，mean node-pair coverage 降低不超过 `0.05`；
5. mean consecutive overlap 不低于 `4.5`，且 consecutive/nonconsecutive Jaccard gap 为正；
6. 三个 audit seeds 中至少两个同时满足 3–5。

可能判定：

- `PASS_FRONTIER_OVERLAP_COVER`：进入 KSVD 后的 patch/整图双重重构；
- `SELECT_SLIDING_WALK_COVER`：sliding walk 达标而 frontier 不达标；
- `FAIL_SMALL_PATCH_GRAPH_RECOVERY`：容量充足后仍不能稳定覆盖边/节点对，需要增大 patch、显式记录 boundary edges，或放弃线性序列；
- `FAIL_IMPLEMENTATION_INVARIANTS`：observed consistency 或 mapped replay 失败。

## 7. 解释边界

通过只说明获得了 traversal-relative、连续、可拼接的 patch cover，不说明：

- 得到了图像式唯一二维坐标；
- 不同图中的节点槽位天然对齐；
- KSVD 原子已经有稳定 motif 语义；
- 对分类或 Transformer 一定有帮助。

若后续 50-node 数据的节点身份跨图共享，应另加 aligned-matrix baseline；仅节点数量相同不等于节点语义已经对齐。

## 8. v0.0 到 v0.1 的正式运行前修订

最小 smoke 只用于检查实现和指标量纲，不进入正式结果。它暴露了两处协议问题：

1. v0.0 的 sliding budget 补足没有继续维持严格滑窗，v0.1 改成每步保留 5 个节点并沿同一 walk 补 5 个节点；
2. v0.0 要求 overlap 方法的 node-pair coverage 比 independent baseline 高 `0.10`。但每个 50% overlap transition 按构造至少重复观察 `C(5,2)=10` 个 pairs，这与同预算下提高 pair coverage 的 gate 冲突。v0.1 改成 pair coverage 损失不超过 `0.05`，并要求 true-edge coverage 至少提高 `0.03`，用来检验重复容量是否换来了更有效的真实边观察。

修订发生在完整 `3 seeds × 72 graphs × 3 methods` 正式运行前；正式结果不再修改 gate。
