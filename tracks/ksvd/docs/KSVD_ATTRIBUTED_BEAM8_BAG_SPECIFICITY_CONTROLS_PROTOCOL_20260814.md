# Attributed Beam8 frozen BAG specificity matched controls

> 日期：2026-08-14  
> 状态：完整 BAG 双轴矩阵通过后冻结；matched-control 结果不可见。

## 1. 研究问题

Graph-conditioned BAG residual 在 45 folds 上稳定改善 frozen GINE，但导师明确指出：对字典表示
取平均可能退化为普通图统计。本轮回答：

> 当前校准收益是否确实需要 Beam8 patch 分解与确定性 INIT vocabulary，而不是任意图级统计
> 或随机特征？

## 2. 压力网格与冻结基线

- 数据：TU Mutagenicity；
- split seeds：3/4；model seeds：0/1/2；每 cell 3 folds，共 18 units；
- base epochs、GINE scores 与 FULL BAG scores 读取已冻结结果；
- 每 fold 重训同一 inner/full GINE，GINE score 与 FULL BAG repeat 必须逐 fold零误差复现；
- 所有 residual 共享 rank 16、zero-init、optimizer、loader seed、inner checkpoint protocol；
- 所有结构输入零填充到 FULL 相同维度，保持 residual 参数量一致。

## 3. Matched controls

- `BEAM8_FULL`：deterministic maximin INIT code + patch node-attribute histogram；
- `BEAM8_CODE_ONLY`：只保留 deterministic INIT code；
- `BEAM8_HIST_ONLY`：只保留 Beam8 patch 内节点属性均值，不使用 dictionary；
- `BEAM8_RANDOM_DICTIONARY`：相同 Beam8 patches、K24/T3 与属性 histogram，但 dictionary atoms
  从相同 outer-train patch pool 随机抽取并归一化；
- `GLOBAL_STATS`：不使用 Beam8 cover，直接使用节点属性 mean/max/sum 与图大小、度、三角形、
  连通分量等低容量统计。

每个变体均形成 graph-level BAG field 并广播给节点，与 frozen GINE states 做相同 conditional
residual。

## 4. Specificity gate

全部满足才认为当前收益具有 Beam8-specific 证据：

1. FULL−GINE mean `≥+0.5pt`，至少 12/18 units 正；
2. FULL−GLOBAL_STATS mean `≥+0.5pt`，至少 12/18 正；
3. FULL−HIST_ONLY mean `≥+0.25pt`，至少 11/18 正；
4. FULL−RANDOM_DICTIONARY mean `≥+0.25pt`，至少 11/18 正；
5. FULL−GLOBAL_STATS 的 split3/4 means 均为正；
6. FULL−GLOBAL_STATS 至少 2/3 model-seed means 为正；
7. 所有 GINE/FULL parity checks 精确通过。

失败时，BAG residual 仍可作为经验校准器，但不能声称收益来自 Beam8 dictionary/patch
机制；随后应优先简化为统计基线或重新定义 Beam8-specific 表示，而不是扩展多数据集。
