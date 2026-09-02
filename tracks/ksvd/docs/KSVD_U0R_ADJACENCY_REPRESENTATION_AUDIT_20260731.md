# KSVD U0-R：邻接向量是否具有可学习的结构几何

> 日期：2026-07-31
>
> 状态：**冻结 v0；先于任何 U0 KSVD 训练**

## 1. 为什么 exact canonicalization 仍然不够

对于一个 6-node rooted patch，枚举固定 root 的 `5!=120` 个排列并选择 lexicographically minimum adjacency vector，可以保证：

> 两个 rooted-isomorphic graphs 映射到相同向量。

但 KSVD 使用的不是“是否相同”判断，而是 Euclidean reconstruction、inner product 和线性组合。因此还需要更强的性质：

1. 向量中同一 coordinate 在不同样本间有可比较语义；
2. 结构只变化一条边时，向量不应无故跳变很多 coordinates；
3. 向量距离应与允许节点重排后的结构距离大致一致；
4. learned continuous atom 即使不是合法 adjacency，也应是某种稳定结构方向，而不是 canonical relabeling artifact。

U0-R 的目的不是寻找“完美图表示”，而是避免在尚未确认线性几何时直接运行 KSVD。

## 2. 两种候选邻接表示

### R-CAN：rooted canonical adjacency

- patch 含 6 个不同节点；
- root 固定为 slot 0；
- 枚举其余 5 个节点的排列；
- 取 15 维 upper-triangle binary vector 的 lexicographic minimum。

优点：rooted-isomorphism invariant、同构类唯一。

风险：不同 patch 会独立选择 canonical permutation；一个很小的结构扰动可能改变最小排列，从而使很多坐标同时移动。

### R-WALK：walk first-discovery-order adjacency

- slot 0 是 root；
- slot 1..5 是随机游走首次发现新节点的顺序；
- 直接对该顺序下的 induced adjacency 取 15 维 upper triangle。

优点：coordinate 语义固定为“第 i 与第 j 个首次发现节点是否相连”；同一 ordering 下的一条 edge flip 严格只改变一个 coordinate。

风险：同一个 rooted induced graph 通过不同 walk trace 可能产生不同向量；它学习的是 walk-induced rooted patch distribution，而不是纯粹的 unlabeled subgraph distribution。

## 3. exact evaluator-only structural distance

对两个 6-node rooted adjacency matrices `A,B`，定义：

\[
d_{GED}(A,B)=\min_{\pi:\pi(0)=0}
\left\|u(A)-u(P_\pi^T B P_\pi)\right\|_0,
\]

其中 `u` 是邻接上三角向量。由于只有 5 个非 root 节点，可以穷举 120 个排列得到 exact distance。

该距离只用于评估，不提供给 KSVD。

## 4. 冻结数据

- graph generator：U0 的 60-node 4-regular ring lattice + connected degree-preserving swaps；
- LOW：20–40 accepted swaps；
- HIGH：60–80 accepted swaps；
- audit seeds：`732001, 732002, 732003`，不与正式 U0 seeds 重叠；
- 每个 seed：LOW/HIGH 各 40 图；
- 每图 12 个 root/walk patches；
- patch size 6；
- 每个 seed 共 960 patches；
- 每个 seed 固定抽取 2000 个不同 patch pairs 做距离审计。

U0-R 不使用 graph labels 选择表示；LOW/HIGH 只保证 audit patches 来自后续目标分布的两端。

## 5. 必须报告的检查

### 5.1 基本有效性

两种表示均检查：

- 6×6 adjacency 对称、binary、zero diagonal；
- upper vector 与 adjacency round trip；
- walk selected nodes 恰有 6 个且包含 root；
- induced adjacency 包含所有 selected-node edges，而非只有 traversed edges。

### 5.2 置换性质

R-CAN：

- 对每个 patch 随机做 20 次 root-preserving permutation；
- canonical vector 必须逐项完全相同。

R-WALK：

- 对完整图全局重编号；
- 同步映射 root、walk trace 和 first-discovery order；
- walk-order vector 必须逐项完全相同。

同时明确记录：R-WALK 对“任意重新排列同一 selected node set”不要求不变，因为 ordering 是 sampler 提供的可观察 side information。

### 5.3 one-edge perturbation continuity

对抽样 patch 的 15 个可能 edge positions 逐一 flip，在固定原始 node identities 下比较：

- exact rooted GED，理论上恒为 1；
- R-CAN vector Hamming distance；
- R-WALK vector Hamming distance，理论上恒为 1。

R-CAN 报告：

- mean/median/95th/max canonical Hamming distance；
- `distance > 1` 的 amplification rate；
- `distance >= 4` 的 severe-jump rate。

这直接回答：一条真实边变化是否会因为 canonical relabeling 看起来像多条坐标变化。

### 5.4 pairwise metric fidelity

对 2000 个 patch pairs 计算：

- exact rooted GED；
- R-CAN Hamming/Euclidean distance；
- R-WALK Hamming/Euclidean distance；
- Pearson 与 Spearman correlation；
- 表示距离减 exact GED 的 mean、95th 和 maximum excess。

这里不期待 R-WALK 等于 GED，因为两个 walk 的 discovery slots 不必是最佳图匹配；该指标用于量化 sampler-order variation。

### 5.5 canonical injectivity self-test

在 sampled pairs 上验证：

- `canonical vectors equal` 当且仅当 exact rooted GED 为 0。

该检查只验证 canonical implementation，没有证明其线性几何良好。

## 6. 决策规则

U0-R 不允许仅凭“canonical 是置换不变的”自动选择 R-CAN。

### 选择 R-CAN 作为 KSVD 主表示

需要同时满足：

- permutation invariance rate = 1；
- sampled injectivity disagreement = 0；
- one-edge amplification rate `<=0.20`；
- severe-jump rate `<=0.05`；
- pairwise Spearman with exact GED `>=0.80`。

### 选择 R-WALK 作为 KSVD 主表示

当 R-CAN 未通过，但 R-WALK 满足：

- mapped-trace global-relabel invariance rate = 1；
- one-edge distance 恒为 1；
- pairwise Spearman 不低于 R-CAN 超过 0.05；
- 后续 U0-P signal-exposure gate 通过。

此时方法的准确表述必须是：

> KSVD learns a sparse basis of rooted walk-induced patch signals.

不能声称学习的是唯一 unlabeled-subgraph vocabulary。

### 两者均不适合

若 R-CAN 几何失真严重，而 R-WALK 的 pairwise fidelity 又明显更差，则暂不运行 adjacency KSVD。下一步改为固定语义的 permutation-invariant structural statistics，例如 rooted distance counts、walk-return counts、degree/distance histograms，再重新审计。

## 7. 本审计不要求的内容

- 不要求 atom 是合法 adjacency；
- 不要求每个 coordinate 对应人类 motif role；
- 不训练 KSVD；
- 不做 LOW/HIGH 分类；
- 不根据下游准确率选择 R-CAN 或 R-WALK；
- 不使用 restart。
