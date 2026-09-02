# luyin14：结构—属性共享稀疏码 strict screen

> 日期：2026-08-12  
> 状态：结果前冻结  
> 数据：MUTAG / PTC_MR；不构造数据集。

## 1. 问题

已有 concat、OOF late fusion、patch-local bilinear 和节点级 FiLM 都是在结构 KSVD 完成后再
融合属性。节点级审计还发现 rooted `v∪N(v)` 邻接 patch 在 MUTAG/PTC_MR 分别只有 4/6 种，
`K24/T3` 的训练重构误差约为 `1e-14`；因此 INIT 已足够表示结构 patch，FINAL 没有学习余量。

本轮不增加分类器容量，而把多模态融合前移到字典学习：

> 结构 view 与属性 view 使用同一个稀疏系数重构时，学到的联合局部 token 是否能稳定补充 GIN？

## 2. 联合节点 patch

每个节点一个样本：

- 结构 view：rooted-canonical `v∪N(v)` padded adjacency upper triangle，28D；
- 属性 view：中心节点属性与一跳邻居属性均值；
- 两个 view 分别使用 outer-train node patches 做中心化和 RMS block scaling，使两块平均能量相等；
- 拼接后拟合一个共享系数 K-SVD：`K24/T3/updates5`；字典只看 outer-train graphs。

这相当于最小的 coupled/multiview dictionary：一个 atom 同时有结构部分和属性部分，而同一
稀疏系数表示二者的共同出现。它与“分别编码后 concat”不同。

## 3. 变体与归因

- `GIN_ONLY`：原始节点属性；
- `JOINT_FINAL_TRUE`：FINAL 联合 code 通过 zero-init FiLM 注入每层 GIN；
- `JOINT_INIT_TRUE`：同一初始化字典、0 次 K-SVD update；
- `JOINT_FINAL_SHUFFLED`：每图内置换属性 view 后再学习/编码，保持结构和属性各自边际分布，
  只破坏二者在节点上的对应。

FiLM、GIN、优化器和 checkpoint 与节点级 Stage A 完全一致。INIT/FINAL 必须分别用各自
outer-train normalization；test 不进入字典、scaling 或 checkpoint 选择。

## 4. Stage A

- split seed 0，3 folds，model seed 0；
- outer-train 内固定 80/20 stratified validation 选 epoch；
- 完整 outer-train 重训固定轮数，outer-test 只评一次；
- 不根据结果扫描 K/T、block weight、hidden、FiLM strength 或 checkpoint 规则。

至少一个数据集同时满足以下条件、且另一个相对 GIN 不低于 `-0.01`，才扩展 split seeds 1/2：

1. `JOINT_FINAL_TRUE - GIN_ONLY >= +0.01`，至少 2/3 folds 正；
2. `JOINT_FINAL_TRUE - JOINT_FINAL_SHUFFLED >= +0.005`，至少 2/3 folds 正；
3. `JOINT_FINAL_TRUE - JOINT_INIT_TRUE > 0`，至少 2/3 folds 正；
4. FINAL 联合字典的训练重构相对 INIT 有下降，证明更新并非数值空操作。

若失败，不进入 cross-attention；结论限定为当前共享系数 joint patch 未形成可归因的分类增益。
