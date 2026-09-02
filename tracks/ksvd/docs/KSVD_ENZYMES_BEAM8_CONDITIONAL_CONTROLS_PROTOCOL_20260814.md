# ENZYMES Beam8 在强多模态基线后的条件增量 controls

> 日期：2026-08-14  
> 状态：ENZYMES split3/4 多模态基线确认通过后冻结；split5/6 结果不可见。

## 1. 研究问题

已确认 `GIN_FULL_ATTRIBUTES + GLOBAL_STATS` 在未见 split3/4 上稳定超过统计和 label-only
matched control。本轮在新的 split5/6 上回答：

> Beam8 连续 patch cover、正确 node binding 与确定性 INIT dictionary，是否还能在这个强基线
> 之上提供独立分类增量？

## 2. 属性与不变性边界

- GIN 与 patch content 读取 18 continuous attributes + 3 discrete labels；
- Beam8 canonicalization、component order、automorphism orbits 只读取3维离散 labels；
- 18维连续属性只进入每个 patch 的 node-feature mean，不参与 `argmax` 节点着色；
- ENZYMES 无 edge labels，所有真实边使用单一合法 edge type；
- dictionary、normalization、epoch selection 全部 train-fold only。

## 3. 顺序冻结模型

每个 outer unit：

1. 训练 full-attribute GIN；
2. 冻结 GIN，训练 GLOBAL_STATS rank-16 residual；
3. 冻结两者，将其 logits 作为 `BASE_MULTIMODAL`；
4. 只训练第二个同容量 rank-16 residual 读取 Beam8 control。

网格：split seeds 5/6 × model seeds 0/1/2 × 3 folds，共18 units。每一阶段都用 inner
validation 选择 epoch，outer-test 只评估一次。

## 4. Beam8 matched controls

- `BEAM8_FULL_TRUE`：deterministic INIT K24/T3 code + 21维 patch attribute mean，通过正确
  patch→node incidence 注入；
- `BEAM8_FULL_BAG`：同一 FULL row multiset 图级平均后广播；
- `BEAM8_FULL_SHUFFLED`：同图保持 row multiset/node heterogeneity，打乱 node binding；
- `BEAM8_CODE_ONLY_TRUE`：只保留 INIT code；
- `BEAM8_HIST_ONLY_TRUE`：只保留 patch attribute mean，不使用 dictionary；
- `BEAM8_RANDOM_DICTIONARY_TRUE`：相同 cover/K/T/histogram，dictionary atoms 从相同
  outer-train patch pool 随机抽取并归一化。

所有第二阶段输入 padding 到 FULL 相同维度，残差容量和 checkpoint 流程完全相同。

## 5. Beam8-specific gate

全部满足才判定 `ENZYMES_BEAM8_SPECIFIC_INCREMENT_SUPPORTED`：

1. FULL_TRUE−BASE mean `>=+1pt`，至少12/18为正；
2. FULL_TRUE−BAG mean `>=+0.5pt`，至少12/18为正；
3. FULL_TRUE−SHUFFLED mean `>=+0.5pt`，至少12/18为正；
4. FULL_TRUE−HIST_ONLY mean `>=+0.25pt`，至少11/18为正；
5. FULL_TRUE−RANDOM_DICTIONARY mean `>=+0.25pt`，至少11/18为正；
6. FULL_TRUE−BASE 与 FULL_TRUE−SHUFFLED 的 split5/6 means 均为正；
7. 上述两个比较均至少2/3 model means 为正。

失败时停止 ENZYMES Beam8 分类路线；不能用 cross-attention、监督 dictionary 或更大分类头
补救。通过时也只支持低容量条件增量，后续仍需独立数据集确认。
