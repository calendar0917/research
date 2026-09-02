# ZINC long-range local-object relation 快筛

日期：2026-08-30  
协议：`luyin16-zinc-long-range-local-object-relation-screen-v1`  
状态：small-slice no-go；未加载 test

## 问题

已有 factorial 表明 radius-3 优于 radius-1/2。本轮检查该提升能否解释为：

> radius-2 局部对象本身不变，但对象在图上的远距离排列含有额外信息。

baseline 为 `global+typed radius-2 raw`。新增视图统计中心最短路距离 `>=3` 的
无序对象对；距离截断为 7+。position-shuffle 在每个图内打乱对象对中心位置的
分配，保留图拓扑和对象 bag，并为每个 shuffle 拟合 matched train-only relation
vocabulary。固定 XGBoost 和 model seeds `0/1/2`，不加载 test。

## schema 覆盖快筛

先用 `200/50` smoke，只按 relation vocabulary 覆盖选择对象 schema：

| object schema | max tokens | train pair coverage | valid unknown rate | 判定 |
|---|---:|---:|---:|---|
| full structural signature + atom | 256 | 0.4822 | 0.5430 | 过度碎片化，淘汰 |
| atom + centre degree | 512 | 0.9906 | 0.0203 | 可用，但非最简 |
| atom | 512 | 1.0000 | 0.0023 | 进入正式快筛 |

smoke 标签分数不用于 schema 选择。局部结构已经由 typed raw baseline 提供；新增
atom relation 分支只回答远距离 chemical-object arrangement 是否有额外价值。

## 两个非重叠 2000/200 切片

| slice | global+raw | + long atom relation | relation 对 raw 增益 | mean true-shuffle gap | min true-shuffle gap |
|---|---:|---:|---:|---:|---:|
| A | 0.73813 | 0.73223 | +0.00591 | +0.01436 | -0.00528 |
| B | 0.56485 | 0.59294 | -0.02809 | +0.00264 | -0.00062 |

正式切片的 train pair coverage 均为 `100%`，valid unknown rate 为
`0.00017/0.00010`；每图平均保留约 `209/212` 个距离≥3 的 pair。因此失败不是
coverage、unknown token 或长程 pair 数量不足造成的。

## 判定

预注册 gate 要求两个切片均相对 raw 改善 `>=0.01 MAE`，且 true 相对 shuffle
平均间隔均 `>=0.01`。两项均未跨切片通过，因此不扩大到 `5000/500`，也不增加
relation vocabulary 容量或下游调参。

该结果否定的是简单解释：

```text
radius-3 gain = bag of long-distance atom-pair co-occurrences
```

它不否定长程信息本身。结合既有结果，更合理的当前解释是：扩大 radius 改善了
每个局部对象内部的上下文和结构--属性组合，而不是仅靠图级 pair histogram 恢复
对象间关系。若继续采样问题，应直接比较单尺度 radius-3 与 radius-2+3 多尺度
对象，不再扩张 pair-token 或 patch-relation summary。

原始结果：

- `ZINC_LONG_RANGE_ATOM_RELATION_SLICE_A_20260830.json`
- `ZINC_LONG_RANGE_ATOM_RELATION_SLICE_B_20260830.json`
