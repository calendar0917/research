# KSVD G0B-R 结论：先停止 noisy adjacency-mask KSVD

> 日期：2026-07-31  
> 状态：**representation gate 已完成**

## 1. 直接结论

当前四种 motif（triangle、4-cycle、3-star、5-path）一旦允许最基础的 within-patch variation，`exact canonical adjacency + 固定 edge-mask atom` 这一定义就不再稳定。

因此现在**不应该**直接运行 noisy G0B KSVD，也不应该用更多 restart、iterations 或更复杂初始化修复。否则即使实验失败，也无法判断是 KSVD 失败还是 representation/target 本身不可辨识。

## 2. 已确认的三个层次

### 2.1 Clean G0

- 完整图、全局节点置换、oracle cell patch、exact canonicalization pipeline 正确；
- deterministic maximin 在 INIT 已枚举四种 clean prototypes；
- fixed-random 单次初始化 INIT 0/10、FINAL 10/10，说明 KSVD 在理想离散条件下有真实 refinement 能力。

### 2.2 Unrooted noisy representation

加入恰好一条 non-core edge：

- cross-family collision mass：0.4773；
- Bayes upper-bound accuracy：0.8920；
- worst-motif robust fixed-core F1：0.5909；
- nearest-medoid macro accuracy：0.5379。

其中存在两个明确的观测冲突：

- triangle family 和 3-star family 可以生成同一个无类型 canonical graph；
- 4-cycle family 和 5-path family 可以生成同一个无类型 canonical graph。

这不是学习算法问题，而是 latent “base motif + nuisance” 分解对观测图不唯一。

### 2.3 Root repair

尝试了：

1. graph-observable maximum-degree root；
2. generator node 0 的 oracle root，模拟 extractor 提供稳定 anchor。

二者均未通过完整 gate。oracle root 虽提升 Bayes accuracy 和部分 motif 的 core consistency，但 five-path 等 family 仍没有稳定 fixed support。因此单个 root 不足以普遍修复当前命题。

## 3. 当前不能再沿用的强命题

暂时否定的是：

> 对一般无类型 noisy patch，exact canonical adjacency 中存在一个固定 edge support，可由一个 KSVD atom 表示生成器定义的隐藏 motif core。

这不等于否定所有 KSVD 图路线。仍有三种不同路线：

### 路线 A：保留显式 motif atom，但增加可观测语义

使用：

- node/edge types；
- sampling root 与路径次序；
- 领域属性；
- 多个 anchors；
- typed/rooted canonicalization。

此时必须把“稳定 anchor/type 可获得”写成方法假设，不能暗中使用 oracle 信息。

### 路线 B：保留纯结构，但重新选择可辨识 motif family

寻找在给定 nuisance model 下满足以下条件的结构 family：

- perturbation closures 不发生 cross-family collision；
- canonical core coordinates 稳定；
- family prototypes 可分。

这可以作为新的 positive control，但必须承认 motif set 是按 representation robustness 选择的，不能外推到任意 motif。

初步穷举审计表明：在 6-node connected unlabeled graphs 中，常见稀疏 motifs 很难同时找到四个满足“一条新增边后仍 closure-disjoint 且 fixed-core F1>=0.9”的同密度 family；到约 9 edges 的较稠密区域才容易找到四个。说明当前 canonical edge-mask 表示对 motif 选择有明显偏置。

### 路线 C：弱化 atom 语义

不再要求 atom 是固定 motif edge mask，而解释为：

- latent structural basis；
- canonical-patch family direction；
- graph-invariant statistic basis；
- reconstruction/occurrence embedding component。

此时评价应转向跨数据稳定性、observed-patch coverage、graph-level usefulness 和相对简单 baseline，而不能继续报告“恢复 hidden core motif”。

## 4. 推荐的下一步顺序

为了继续从基础推进，推荐：

1. **先做路线 B 的 identifiable-family positive control**：证明在 representation assumptions 成立时，KSVD 能否从多个 variants 中提炼 family atom；
2. 然后做路线 A 的非 oracle anchor，例如随机游走起点/rooted path，检查真实采样器是否提供足够 side information；
3. 最后再决定真实路线要坚持“显式 motif atom”，还是转为路线 C 的 latent structural basis。

不建议现在进入：

- CIN/MolHIV；
- graph classification；
- 大量 hyperparameter/restart；
- 随机游走覆盖率；
- noisy KSVD on current four motifs。

## 5. 证据文件

- `tracks/ksvd/results/from_scratch/G0B_REPRESENTATION_GATE_20260731.md`
- `tracks/ksvd/results/from_scratch/g0b_representation_gate_20260731.json`
- `tracks/ksvd/results/from_scratch/G0B_ROOT_REPAIR_PROBE_20260731.md`
- `tracks/ksvd/results/from_scratch/g0b_root_repair_probe_20260731.json`
