# Beam8/Mutagenicity typed-anchor protocol

> 日期：2026-08-13  
> 状态：binary MAXCHAIN 失败后注册；分类结果不可见。

## 1. 动机

`s8/o2` binary MAXCHAIN 平均达到 90.7 patches，第三类 bond recall 仍仅 61.8%。因此继续
延长 binary marginal chain 不是有效解法。采用最小 edge-aware 修复：

1. 固定原 `s8/o2 Beam8/R1/m1.5 BASE` chain；
2. 对每张图、每个实际存在但 BASE 完全没观察到的 bond type，选一个该类型未覆盖 edge；
3. 用 binary connected fill 构造一个包含该 edge 的 8-node canonical patch；
4. patch 标为独立 `typed_anchor` segment，不伪装成原 Beam8 chain transition。

edge type 是数据输入，不使用 graph label；每种类型最多增加一个 anchor。

## 2. 无标签 gate

- 各 bond type graph coverage = 1；
- 第三类 bond aggregate recall ≥0.80；
- mean extra anchors ≤0.25 patch/graph；
- 95% 图不超过一个 anchor；
- BASE chain 和所有原 patch 完全不变。

通过后才运行 typed classification，并必须同时报告 BASE 与 BASE+ANCHOR。ANCHOR 改善只能
解释为 edge-semantic completeness，不能解释为连续 Beam8 relation 的收益。

