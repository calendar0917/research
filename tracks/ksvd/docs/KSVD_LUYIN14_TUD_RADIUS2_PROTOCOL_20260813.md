# luyin14：TU typed radius-2 shared-code Stage A

> 日期：2026-08-13；真实 Mutagenicity/NCI1，不构造数据集。

## 固定变更

相对 `KSVD_LUYIN14_TUD_EXTERNAL_VALIDATION_PROTOCOL_20260813.md` 只改变 patch 采样：

- rooted radius-2 ego；
- 最多 8 个节点；先按到中心距离，再按 degree、component-stable rank、原始 ID 截断；
- canonicalization、K-SVD `K24/T3/updates5`、shared attribute context、GINE/GIN、FiLM、
  BASE/FINAL TRUE/SHUFFLED/INIT、strict checkpoint 均不变；
- split seed 0，3 folds，model seed 0；不扫描任何超参数。

## Gate

每个数据集要求 FINAL 相对 BASE、SHUFFLED、INIT 三个 paired gate 同时达到之前协议阈值，且
FINAL reconstruction 优于 INIT。任一数据集未通过则不扩展 split seeds；radius-2 只保留为
substrate 诊断，不再增大半径或 patch 容量。
