# Beam8/Mutagenicity MAXCHAIN audit protocol

> 日期：2026-08-13  
> 状态：FAIR95 cost gate 失败后注册；不使用 graph labels。

## 1. 目的

FAIR95 completion 能补回稀有 bond，但平均新增 patch 超过 BASE 的 100%，且 completion
patch 是独立 segment。现在检验另一种严格 Beam8 用法：复用同一条 binary Beam8/R1
候选链，不插入 edge-targeted completion，直接把 chain 从 BASE prefix 延长到已生成的
自然 MAXCHAIN（最多 128 patches，或 sampler 自然停止）。

这区分：

- 稀有 bond 是否只是 BASE operating point 太早停止；
- 连续 Beam8 本身是否能覆盖稀有 bond；
- 继续延长 chain 是否只是昂贵地重复观察常见边。

## 2. 固定指标

- BASE 与 MAXCHAIN patch count / ratio；
- chain edge count 与 segment count；
- overall、各 bond type edge recall；
- 含稀有 bond 图的 graph coverage；
- node coverage、incident recall p10；
- MAXCHAIN 新增边中各 bond type 比例。

## 3. 预注册判定

- `MAXCHAIN` 稀有 bond aggregate recall ≥0.80 且 graph coverage ≥0.80；
- 相对 BASE 新增 patch 不超过 BASE 的 100%；
- chain edge 图比例不下降；
- 若稀有 bond 通过但新增 patch 比例超过 100%，只能说明“连续链可补回但成本过高”，
  不进入分类；
- 只有满足全部条件，才允许用 MAXCHAIN 做一次 typed classification，且必须同时报告 BASE。

