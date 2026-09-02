# Beam8/NCI1 s8/o2 fixed-classification addendum

> 日期：2026-08-13  
> 状态：分类结果前冻结。

完整无标签 geometry audit 已按
`KSVD_BEAM8_NCI1_GEOMETRY_FEASIBILITY_PROTOCOL_20260813.md` 运行，判定为
`S8_O2_READY_FOR_FIXED_CLASSIFICATION`，六项 gate 全部通过。因此允许运行一次
`s8/o2/Beam8/R1/m1.5` 分类。

除 `patch_size: 10 → 8`、`overlap: 3 → 2` 外，所有设置严格沿用
`KSVD_BEAM8_NCI1_CHAIN_CLASSIFICATION_PROTOCOL_20260813.md`：同一 NCI1 数据、split
seed 0 的 3-fold、K24/T3/5 iterations、3,000 train patches、相同 RAW/INIT/FINAL、
BAG/TRUE/SHUFFLED、固定 relation message passing、linear head 和冻结 gates。

不得根据本轮分类结果在 s8/o2 与 s10/o3 之间选择性汇报；两组结果必须并列解释。

