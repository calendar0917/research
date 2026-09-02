# luyin16 ZINC 机制路线阶段综合判断

日期：2026-08-30

本摘要整合 long-range factorial、joint/shuffle、conditional residual 和 long-range
object relation 快筛。所有机制选择只使用 official train/valid；test 保持冻结。

## 四个问题的当前答案

### 1. 统计什么

当前最稳定的是：全中心 radius patch 的 typed coordinate-wise distribution，再与
global structure/attribute statistics 拼接。结构--属性 joint、conditional 和
centered residual 都能通过 shuffle 检出 binding，但没有跨独立切片稳定超过 typed
marginal。因此 binding 统计暂时是诊断量，不是主预测表示。

### 2. 怎么采样

全原子中心优于少量随机中心；ZINC 上 radius-3 优于 radius-1/2，且 max_nodes
从 12 扩到 20 几乎不再提升，说明关键是感受野而不是 cap 截断。简单的远距离
atom-pair histogram 未复现增益，因此 radius-3 更可能改善单个对象的上下文。

最后一项有清晰归因价值的采样实验固定比较：

```text
global + radius-2 raw
global + radius-3 raw
global + radius-2 raw + radius-3 raw
```

该实验已完成：r2+r3 在四个切片均为最优，但相对最佳单尺度的增益仅
`0.01066/0.00549/0.01501/0.00471`，未稳定达到 `0.01`。因此多尺度记为弱互补，
不晋级 full；大样本单尺度固定为 radius-3 raw。不再新增随机游走、pair token
或 relation encoder。

### 3. K-SVD 怎么用、是否有用

当前所有稳定正增益都可以在 RAW/fixed statistics 中出现；K-SVD 虽降低重建误差，
FINAL 未稳定优于 INIT/RAW。binding residual 可以被 SVD16 压缩并保留约 71% 方差，
但仍无稳定标签增量。因此普通无监督 K-SVD 目前只保留 compressor/diagnostic 身份，
不进行 K/T、稀疏度或 objective 搜索。

只有未来某个 RAW 多尺度对象在独立切片稳定增益后，才值得问 K-SVD 能否在固定
维度下压缩它；不能反过来靠字典搜索寻找任务信号。

### 4. 属性与结构如何解耦、融合

实验已区分三层：

1. marginals：结构统计、属性统计和 typed local distribution；
2. binding：true vs within-graph attribute shuffle；
3. relation：local-object bag vs position shuffle。

当前稳定融合仍是低容量 late concat：`global + typed raw`。binding 明确存在，
但无稳定 target increment；relation pair 也未通过。因此暂不使用 cross-attention、
GNN/Transformer 或更高容量融合来补救上游对象定义。

## 证据表

| 路线 | 机制控制 | 是否跨切片有机制信号 | 是否稳定超过 typed raw | 决策 |
|---|---|---|---|---|
| joint v1 | attribute shuffle | 是 | 否 | 停止高维 joint v1 |
| conditional signature v2 | attribute shuffle | 是 | 否 | 不进 full/Optuna |
| centered residual + SVD16 | matched shuffled SVD | 是 | 否 | 不扫维度/K-SVD |
| long-distance atom relation | position shuffle | 否 | 否 | 停止 pair relation |
| larger radius | radius-1/2/3 matched | radius-3 正增益 | 是 | 保留 radius-3 |
| r2+r3 multiscale | matched single-scale views | 四片方向一致 | 未稳定达到 0.01 | 弱候选，不进 full |

## 当前停止项与阶段冻结

停止：局部 joint 展开、conditional histogram、SVD 维度扫描、pair token、粗
patch-relation、K/T/稀疏度扫描、复杂融合。

多尺度比较也已完成并在第二个大切片未通过 `0.01` 门槛。至此固定 radius-3 raw
为 ZINC 当前结构对象 baseline，r2+r3 仅作为 future candidate；本轮机制探索结束。
下一阶段不应继续同类特征扫描，而应等待导师 schema，或把问题转成明确的
task-aligned dictionary objective，并重新预注册独立协议。
