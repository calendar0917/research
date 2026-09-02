# Attributed Beam8/Mutagenicity classification protocol

> 日期：2026-08-13  
> 状态：attributed invariance 与 typed-anchor geometry gate 通过后冻结，分类结果不可见。

## 1. 固定表示

- 数据：完整 TU Mutagenicity，split seed 0，3-fold stratified CV；
- BASE：`s8/o2/Beam8/R1/m1.5`，全图和 patch 均使用 atom-colored、bond-typed
  nauty canonical coordinates；
- ANCHOR：对 BASE 完全遗漏的实际 bond type，每种至多添加一个 canonical 两端点语义
  token，作为独立且 relation-isolated 的 segment；
- patch payload：8 个 canonical slots 的 atom-type one-hot，加 28 对 slots 的
  bond-type one-hot；另拼接 patch node-label mean；
- 字典：仅在 outer-train ANCHOR patches 上拟合，`K24/T3/5 updates/3000 patches`；
  BASE/ANCHOR 共用 INIT/FINAL dictionary、mean 与每个 token family 的 normalization；
- readout：冻结的 compact relation + `StandardScaler + LR(C=1)` 线性分类头。

## 2. 必报分支

- `FEATURE_STATS`；
- BASE：`RAW_BAG/TRUE/SHUFFLED`、`INIT_TRUE`、`FINAL_TRUE`；
- ANCHOR：`RAW_BAG/TRUE/SHUFFLED`、`INIT_BAG/TRUE`、
  `FINAL_BAG/TRUE/SHUFFLED`。

TRUE/SHUFFLED 共用 cover、relation graph、token multiset 和 position metadata，只打乱
token-to-position binding。Anchor 不产生连续 chain/overlap relation，但参与 bag 和 position
readout。

## 3. 冻结 gate

- BASE relation：`BASE RAW TRUE−SHUFFLED ≥1 point`，至少 2/3 folds 正；
- typed-anchor increment：`ANCHOR INIT TRUE−BASE INIT TRUE ≥0.5 point`，至少 2/3 folds 正；
- ANCHOR relation：`ANCHOR FINAL TRUE−SHUFFLED ≥1 point`，至少 2/3 folds 正；
- ANCHOR over bag：`ANCHOR FINAL TRUE−BAG ≥0`，至少 2/3 folds 不为负；
- KSVD update：`ANCHOR FINAL TRUE−INIT TRUE ≥1 point`，至少 2/3 folds 正；
- attribute complement：`ANCHOR INIT TRUE−FEATURE_STATS ≥1 point`，至少 2/3 folds 正。

typed-anchor、Beam8 relation、KSVD updates 必须分别归因。旧 binary-order typed
classification 已被 invariance audit 作废，不得作为本实验的支持证据。
