# Beam8/Mutagenicity FAIR95 completion protocol

> 日期：2026-08-13  
> 状态：typed BASE feasibility 结果可见后注册；分类结果不可见。

## 1. 已定位问题

BASE cover 遗漏的第三类稀有 bond 中，所有遗漏都源于至少一个端点从未进入任何 patch；
不存在“两端都已观察但没有共同进入 patch”。因此第一瓶颈是 coverage，而不是 relation
readout。

## 2. 固定 completion

分别从 `s10/o3` 和 `s8/o2` 的原 BASE cover 出发：

- cover 与 completion 只读取 binary adjacency，不读取 node/bond labels 或 graph labels；
- 每次选择当前未覆盖 binary edge 中 endpoint deficit 最大者；
- 构造包含目标边的 connected patch，并优先加入可补最多未覆盖边的节点；
- 每个 completion patch 是显式新 segment，不建立虚假 chain transition；
- 首次达到 edge coverage≥0.95、incident recall p10≥0.90、node coverage=1 时停止，称 FAIR95。

不运行 EDGE100，不扫描 threshold 或 completion policy。

## 3. 必报

- BASE/FAIR95 patch count、额外 patch 数与额外比例；
- FAIR95 edge/node coverage 与 incident p10；
- 原始 Beam8 chain edges（completion 不增加 chain）；
- 三类 bond aggregate recall、mean graph recall；
- 第三类 bond graph coverage；
- completion patch 中各 bond type 的新增边数。

## 4. 分类准入 gate

一个 geometry 只有同时满足以下条件才可进入 typed classification：

1. 所有图达到 FAIR95；
2. mean edge coverage≥0.95，node coverage=1；
3. 第三类 bond aggregate recall≥0.80；
4. 第三类 bond graph coverage≥0.80；
5. 平均额外 patch 不超过 BASE patch 的 75%；
6. 至少 50% 图保留一条或以上原始 Beam8 chain edge。

若两种 geometry 均通过，选择平均额外 patch 更少者；若差值小于 0.25 patch，则选择原始
chain edge 更多者。分类必须并报 `BASE` 与 `FAIR95`，用来区分收益来自连续 Beam8 chain
还是 coverage completion。

