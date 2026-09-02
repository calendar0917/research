# ENZYMES full-feature-canonical Beam8 不变性确认

> 日期：2026-08-14  
> 前置：离散-only canonical 审计失败并作废 split5/6 分类结果。

- 64 graphs × 3 random node permutations；
- 完整21维 raw feature rows 经图内排序映射为 canonical colors；
- structural vector 仍使用3维离散 labels，patch content 使用21维完整属性；
- token rows、token multiset、compact readout 与映回原节点后的 incidence equivariance 必须全部
  为1.0；
- 只有全部通过才允许运行 split7/8 分类 controls。
