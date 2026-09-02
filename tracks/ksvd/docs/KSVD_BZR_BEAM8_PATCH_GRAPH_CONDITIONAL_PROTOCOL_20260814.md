# BZR Beam8 patch-graph conditional increment protocol

> 日期：2026-08-14  
> 状态：BZR prescreen split0/1/2 结果可见后冻结；split3/4 结果不可见。

## 1. 依据与边界

BZR prescreen 中 RAW patch graph 的 TRUE−TOKEN_SHUFFLED 为 `+2.09pt`、7/9且三个 split
means 全正；TRUE−BAG 为 `+1.69pt`、7/9且三个 split means 全正。但 TRUE standalone 比
GLOBAL_STATS 低 `5.09pt`，因此原晋级 gate 失败，不能追溯修改。

本 follow-up 只回答：在已经提供 GLOBAL_STATS 后，正确 Beam8 patch relation 是否仍有独立
线性增量。它不是原 prescreen 的确认性通过，也不训练 patch-GNN。

## 2. 固定特征

Beam8 geometry、attributed canonicalization、RAW structural tokens、train-only normalization、
compact PREVIOUS/OVERLAP/SLOT pair statistics 与 prescreen 完全相同。

variants：

- `GLOBAL_ONLY`；
- `GLOBAL_PLUS_BAG`；
- `GLOBAL_PLUS_PATCH_GRAPH_TRUE`；
- `GLOBAL_PLUS_PATCH_GRAPH_TOKEN_SHUFFLED`。

BAG 在末尾补零到 TRUE relation feature 维度，使 GLOBAL+BAG 与 GLOBAL+TRUE 输入容量一致。
TRUE 与 TOKEN_SHUFFLED 完全同维、同 token multiset、同 patch graph，只改变 token-position
binding。

分类器统一为 `StandardScaler + class-weight-balanced LogisticRegression(C=1)`，不扫描 C。

## 3. 未见 evaluation

- split seeds `3/4`；
- 每个 split 3 folds，共6 units；
- balanced accuracy；所有 variants 共用 folds。

## 4. 冻结 gate

全部满足才授权一层 patch-GNN：

1. `GLOBAL+TRUE−GLOBAL_ONLY >= +0.5pt`，至少4/6正；
2. `GLOBAL+TRUE−GLOBAL+BAG >= +0.5pt`，至少4/6正；
3. `GLOBAL+TRUE−GLOBAL+TOKEN_SHUFFLED >= +1pt`，至少4/6正；
4. 上述三项的 split3/4 means 均正；
5. prescreen 的 BZR relabel audit required checks 保持100%。

全部通过：

`BZR_BEAM8_PATCH_GRAPH_CONDITIONAL_INCREMENT_SUPPORTED`

否则：

`BZR_BEAM8_PATCH_GRAPH_CONDITIONAL_INCREMENT_NOT_ESTABLISHED`

失败时不训练 patch-GNN。通过时也只能在新的 split5/6 上运行一层低容量 patch-GNN，并保留
GLOBAL、BAG、TOKEN_SHUFFLED 与 matched/random cover controls。

