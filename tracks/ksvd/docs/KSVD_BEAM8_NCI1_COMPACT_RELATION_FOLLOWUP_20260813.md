# Beam8/NCI1 compact-relation follow-up protocol

> 日期：2026-08-13  
> 状态：看到 s8/o2 Stage A 后注册；属于机制诊断，不包装为独立确认实验。

## 1. 动机

冻结的 `s8/o2` Stage A 中：

- `FINAL TRUE - SHUFFLED = +1.51 points`，3/3 folds 为正；
- `FINAL TRUE - BAG = -0.49 points`。

真实 token-to-position binding 可检测，但现有 wide relation readout 每个关系通道输出
`2d+6` 维，四个通道合计数百维，并重复包含 message mean/std。它可能因共线与有限样本
惩罚而使 BAG+relation 低于 BAG。

## 2. 固定 compact readout

geometry、cover、folds、字典、token normalization 与 linear head 全部沿用 s8/o2 Stage A。
只把 wide relation block 替换成固定 compact block。

三个关系通道：

- forward chain；
- all-overlap Jaccard；
- overlap × canonical-slot persistence。

每个通道只统计 relation-edge pair 上的加权：cosine mean/std/max、L1 mean/std/max、
normalized dot mean、target/source norm-ratio mean/std、winner agreement，共 10 维；三个通道
合计 30 维。BAG common block完全相同。

必报：RAW/INIT/FINAL BAG、COMPACT_TRUE；RAW/FINAL COMPACT_SHUFFLED，以及 outer-train
与 outer-test balanced accuracy，用于判断是否为 generalization gap。

## 3. 诊断 gate

- compact binding：FINAL TRUE−SHUFFLED ≥1 point 且至少 2/3 folds 为正；
- compact incremental：FINAL TRUE−BAG ≥0 且至少 2/3 folds 不为负；
- capacity diagnosis：compact TRUE−BAG 比 wide TRUE−BAG 至少改善 0.5 point。

三项都通过，才认为此前主要是 readout capacity 问题，并允许进入 1 层可学习 patch-GNN。
binding 通过但 incremental 失败，则 Beam8 relation 可检测但对当前线性分类没有互补增量；
不得通过继续扫描手工统计量救结果。

