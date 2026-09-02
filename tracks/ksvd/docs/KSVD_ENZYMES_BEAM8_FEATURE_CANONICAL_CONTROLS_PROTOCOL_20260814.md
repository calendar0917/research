# ENZYMES full-feature-canonical Beam8 条件增量 controls

> 日期：2026-08-14  
> 状态：split5/6 离散-only canonical 结果因不变性失败作废后冻结；split7/8 结果不可见。

## 1. 修复

离散-only canonical 无法区分离散结构对称、但连续属性不同的节点，导致重标号后 patch content
变化。本轮将每个节点完整原始21维 feature row 映射为图内确定性 canonical color：

- unique feature rows 按数值字典序排序并编号；
- 编号只用于 component order、patch canonicalization 和 automorphism orbits；
- dictionary structural vector 仍只编码3维离散 node labels + 单一 edge type，维度保持52；
- patch histogram 仍读取完整21维内容。

分类前必须先通过64 graphs × 3 permutations 的 token/readout/incidence 100% 不变性 gate。

## 2. 新未见矩阵

- split seeds 7/8 × model seeds 0/1/2 × 3 folds，共18 units；
- base 仍为冻结 `GIN_FULL_ATTRIBUTES + GLOBAL_STATS residual`；
- 第二阶段 variants、容量、train-only dictionary/normalization 与作废协议完全相同；
- 比较 `FULL_TRUE / BAG / SHUFFLED / CODE_ONLY / HIST_ONLY / RANDOM_DICTIONARY`。

## 3. Gate

沿用作废协议中结果不可见前的阈值，不事后修改：

1. FULL_TRUE−BASE `>=+1pt` 且至少12/18正；
2. FULL_TRUE−BAG、FULL_TRUE−SHUFFLED 均 `>=+0.5pt` 且至少12/18正；
3. FULL_TRUE−HIST、FULL_TRUE−RANDOM 均 `>=+0.25pt` 且至少11/18正；
4. increment/binding 的 split7/8 means 均正且至少2/3 model means 正；
5. fixed feature-canonical invariance 全部为1.0。

全部满足才判定 `ENZYMES_FEATURE_CANONICAL_BEAM8_SPECIFIC_INCREMENT_SUPPORTED`；失败则停止
ENZYMES Beam8 分类路线。
