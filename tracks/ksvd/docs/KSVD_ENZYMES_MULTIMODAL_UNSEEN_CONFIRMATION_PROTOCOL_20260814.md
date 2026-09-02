# ENZYMES 完整属性多模态基线未见 split 确认

> 日期：2026-08-14  
> 状态：split0--2 预筛与 follow-up 结果可见后冻结；split3/4 结果不可见。

## 1. 目标

在未见划分上确认此前观察到的信号：完整连续节点属性的图内组织与宏观统计是否形成稳定互补，
而不是 split0--2 的结果驱动偶然性。

## 2. 冻结矩阵

- TU ENZYMES，18 continuous attributes + 3 discrete node labels；
- split seeds 3/4，model seeds 0/1/2，每 split 3 folds，共18 units；
- inner-validation 选择 GIN 与 residual epoch，outer-test 只评估一次；
- 连续属性只按相应训练节点拟合 mean/std；离散 labels 不标准化；
- 比较：
  - `GLOBAL_STATS_LINEAR`；
  - `GIN_FULL_ATTRIBUTES`；
  - `GIN_LABEL_ONLY_PLUS_GLOBAL`；
  - `GIN_FULL_PLUS_GLOBAL`。

两个 GIN 分支使用相同3-layer/hidden64配置；两个 global residual 使用相同 rank16、输入、优化器
与 checkpoint 流程。只有 GIN 是否读取18维连续属性不同。

## 3. 确认 gate

全部满足才判定 `ENZYMES_MULTIMODAL_BASE_CONFIRMED_ON_UNSEEN_SPLITS`：

1. FULL+GLOBAL−GLOBAL mean `>=+5pt`，至少12/18为正；
2. FULL+GLOBAL−LABEL+GLOBAL mean `>=+5pt`，至少12/18为正；
3. 上述两个比较的 split3/4 means 均为正；
4. 上述两个比较均至少2/3 model means 为正；
5. 数据维度精确为18 continuous + 3 discrete。

通过后才允许在新的 split seeds 5/6 上冻结 Beam8 条件增量 controls；失败则停止 ENZYMES
Beam8 分类路线。该确认不重新评价或修改 split0--2 的原预筛 gate。
