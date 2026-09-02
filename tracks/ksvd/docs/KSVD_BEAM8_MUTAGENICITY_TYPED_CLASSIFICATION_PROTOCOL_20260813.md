# Beam8/Mutagenicity typed-anchor classification protocol

> 日期：2026-08-13  
> 状态：typed-anchor 无标签 gate 通过后冻结，分类结果不可见。

## 1. 固定表示

- 数据：完整 TU Mutagenicity，split seed 0，3-fold stratified CV；
- cover：`s8/o2/Beam8/R1/m1.5 BASE`；
- ANCHOR：通过无标签 gate 的最小 typed-anchor，每种缺失 bond type 最多一个独立 segment；
- patch payload：binary Beam8 rooted-canonical slots 上的 `28 slots × 3 bond types` one-hot，
  拼接 14 维 patch node-label mean；
- 字典：只在 outer-train 的 ANCHOR covers 上拟合，`K24/T3/5 updates/3000 patches`；
  BASE 与 ANCHOR 共用同一 INIT/FINAL dictionary、mean 和 token normalization；
- readout：此前冻结的 30 维 compact relation；linear head `StandardScaler + LR(C=1)`。

## 2. 必报分支

- `FEATURE_STATS`；
- BASE：`RAW_BAG/TRUE/SHUFFLED`、`INIT_TRUE`、`FINAL_TRUE`；
- ANCHOR：`RAW_BAG/TRUE/SHUFFLED`、`INIT_BAG/TRUE`、
  `FINAL_BAG/TRUE/SHUFFLED`。

TRUE/SHUFFLED 共用同一 cover、relation graph、token multiset、position metadata；只打乱
token-to-position binding。Anchor 是独立 segment，因此不会产生虚假的 chain edge，但可参与
all-overlap relation 和 bag readout。

## 3. 冻结 gate

- BASE relation：`BASE RAW TRUE−SHUFFLED ≥1 point`，至少 2/3 folds 正；
- typed-anchor increment：`ANCHOR INIT TRUE−BASE INIT TRUE ≥0.5 point`，至少 2/3 folds 正；
- ANCHOR relation：`ANCHOR FINAL TRUE−SHUFFLED ≥1 point`，至少 2/3 folds 正；
- ANCHOR over bag：`ANCHOR FINAL TRUE−BAG ≥0`，至少 2/3 folds 不为负；
- KSVD update：`ANCHOR FINAL TRUE−INIT TRUE ≥1 point`，至少 2/3 folds 正；
- attribute complement：`ANCHOR INIT TRUE−FEATURE_STATS ≥1 point`，至少 2/3 folds 正。

typed-anchor increment 通过只说明 edge-semantic completeness 有价值；BASE/ANCHOR relation gate
通过才支持 Beam8 relation；KSVD gate 通过才支持普通 updates。不得合并解释。

