# Attributed Beam8 localized INIT multi-model-seed validation

> 日期：2026-08-14  
> 状态：content-relation Stage B1 后冻结；model seeds 1/2 结果不可见。

Stage B1 中 `INIT_LOCAL_CONTENT` 为 `74.90%`，GINE 为 `72.68%`，paired `+2.23pt`
且 2/3 folds 正；content-aware relation 明显失败。因此本轮停止 patch message，只验证最简单
的 localized INIT incidence 是否跨 neural initialization 稳定。

- 数据/split：Mutagenicity，固定 split seed 0、3 outer folds；
- model seeds：0/1/2；seed0 读取冻结 Stage-B1 JSON，seed1/2 新训练；
- dictionary：每 fold 同一 outer-train deterministic maximin INIT；不运行 KSVD updates；
- variants：`GINE_ONLY`、`INIT_LOCAL_CONTENT`；
- node feature：INIT code + patch atom histogram 的 incidence mean/max，旧 relation metadata 全零，
  attributed orbit-safe；
- GINE/checkpoint 与 Stage B1 完全相同。

在 9 个 fold×model-seed 单元上，只有同时满足才进入 split seeds 1/2：

1. LOCAL−GINE mean `≥+1pt`；
2. 至少 6/9 为正；
3. 三个 model seeds 的 seed-level mean delta 至少 2/3 为正；
4. 最差 model-seed mean delta不低于 `-0.5pt`。

未通过时，不能把 seed0 的 `+2.23pt` 当稳定分类收益。
