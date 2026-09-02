# Attributed Beam8 INIT-incidence attribution follow-up

> 日期：2026-08-14  
> 状态：FINAL node-incidence Stage A 结果可见后的容量/机制诊断，不是独立确认实验。

Stage A 显示：FINAL TRUE 相对严格 orbit-safe SHUFFLED 为 `+2.76pt`、3/3 folds 正，
但 FINAL−INIT 为 `-1.14pt`、0/3。INIT FULL TRUE 为 `73.54%`，相对 GINE `+0.86pt`。

本轮固定同一 split、dictionary initialization、GINE、checkpoint 与 seeds，只补：

- `INIT_FULL_SHUFFLED`：同图、同 orbit-size canonical orbit 间置换 node incidence；
- `INIT_BAG_BROADCAST`：同图 INIT incidence mean 广播全部节点；
- `INIT_NO_RELATION`：slot/center/position/relation-degree channels 置零。

`GINE_ONLY` 与 `INIT_FULL_TRUE` 直接读取冻结 Stage-A JSON，不重新选择结果。

诊断 gate：

1. INIT TRUE−GINE `≥+0.5pt`，至少 2/3 folds 正；
2. INIT TRUE−SHUFFLED `≥+0.5pt`，至少 2/3 folds 正；
3. INIT TRUE−BAG `≥+0.5pt`，至少 2/3 folds 正；
4. INIT TRUE−NO_RELATION `≥+0.5pt`，至少 2/3 folds 正。

若 1--3 通过，说明无需普通 KSVD updates，maximin INIT vocabulary 的 localized binding 已可
转化为分类收益；若只通过 2，说明 binding signal 存在但尚未超过强基线；若 4 失败，下一步
应学习 content-aware patch messages，而不是继续使用 relation degree metadata。
