# ENZYMES Beam8 canonical-slot attribute protocol

> 日期：2026-08-14  
> 状态：在 full-feature-canonical Beam8 split7/8 结果可见后冻结；split9/10 结果不可见。

## 1. 问题

上一轮每个 patch 只保留21维 node-feature mean。它知道“patch 里有哪些属性”，但不知道连续
属性属于 canonical 结构中的哪个节点槽位，因此可能在进入分类器前就丢失导师要求的
结构—属性对应关系。

本轮只改变 patch 属性表示，不扩大 frozen residual、GIN、GLOBAL_STATS 或 dictionary 容量。

## 2. 表示

每个 Beam8 patch 继续使用 full-feature-row canonical colors 生成合法 cover 与 slot order；
52维 structural vector 仍只编码3维离散 node labels 与单一边类型，并由 train-only 24-atom
INIT dictionary 得到3-sparse code。

属性分支只读取18维连续 node attributes：

- 按 patch 的 canonical `slot_nodes` 顺序放入8个 slots；
- 不足8个节点时补零；
- 追加8维 occupancy mask；
- 得到 `8×18+8=152` 维 slot tensor；
- FULL token 为24维 code与152维 slot tensor 拼接，共176维。

不使用标签监督选择 dictionary、cover、slot order、归一化或表示维度。

## 3. 必需对照

- `BASE_MULTIMODAL`：冻结 full-attribute GIN，再冻结 GLOBAL_STATS residual；
- `SLOT_FULL_TRUE`：真实 dictionary code + canonical-slot attributes + 真实 incidence；
- `SLOT_FULL_BAG`：同一 FULL node field 做图级均值广播；
- `SLOT_FULL_INCIDENCE_SHUFFLED`：保留同图 node-field multiset，打乱 node binding；
- `SLOT_FULL_WITHIN_PATCH_SHUFFLED`：每个 patch 内保留连续属性 row multiset，但打乱属性与
  canonical structural slots 的对应；
- `SLOT_CODE_ONLY_TRUE`：只保留24维真实 code；
- `MEAN_FULL_TRUE`：真实 code + 18维 patch attribute mean，复现上一轮的信息瓶颈；
- `SLOT_RANDOM_DICTIONARY_TRUE`：随机 patch dictionary code + 真实 canonical-slot attributes。

所有较短 token 在 node-incidence 后零 padding 到 FULL 的输入维度。训练、epoch selection、
rank-16 residual、fold dictionary pool 和 normalization 与上一轮完全相同。

## 4. 合法性 gate

分类前在固定64 graphs × 3 permutations 上审计 `SLOT_FULL_TRUE` 与
`SLOT_FULL_WITHIN_PATCH_SHUFFLED`：

- token rows = 100%；
- token multiset = 100%；
- graph readout = 100%；
- mapped node-incidence equivariance = 100%。

任一失败则不运行或不解释分类。

## 5. 新未见矩阵

- split seeds 9/10；
- model seeds 0/1/2；
- 3 outer folds；
- 共18个 split×model×fold units。

split0--8 不用于本轮阈值、表示或超参数选择。

## 6. 冻结 classification gate

全部满足才支持 canonical-slot Beam8 表示：

1. `SLOT_FULL_TRUE−BASE_MULTIMODAL >= +1pt`，至少12/18正；
2. `SLOT_FULL_TRUE−SLOT_FULL_BAG >= +0.5pt`，至少12/18正；
3. `SLOT_FULL_TRUE−SLOT_FULL_INCIDENCE_SHUFFLED >= +0.5pt`，至少12/18正；
4. `SLOT_FULL_TRUE−MEAN_FULL_TRUE >= +0.5pt`，至少12/18正；
5. `SLOT_FULL_TRUE−SLOT_FULL_WITHIN_PATCH_SHUFFLED >= +0.5pt`，至少12/18正；
6. `SLOT_FULL_TRUE−SLOT_RANDOM_DICTIONARY_TRUE >= +0.25pt`，至少11/18正；
7. increment、slot-vs-mean、within-patch binding 的 split9/10 means 全正；
8. 上述三项均至少2/3 model means 正；
9. 固定不变性 gate 全部通过。

全部满足时判定：

`ENZYMES_CANONICAL_SLOT_ATTRIBUTE_BEAM8_INCREMENT_SUPPORTED`

否则判定：

`ENZYMES_CANONICAL_SLOT_ATTRIBUTE_BEAM8_INCREMENT_NOT_ESTABLISHED`

失败时停止 ENZYMES Beam8 分类，不通过 attention、更大 head、监督 dictionary 或调整阈值补救。

