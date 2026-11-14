# TRACK — 支持度条件化的联合块收缩（`lcrc-support-conditioned-shrinkage-public122-s0-v1`）

一句话：把 C1 的**二跳联合块 `J`** 用一个**跨训练分子结构支持度得到的冻结标量 `a_v`**
在 `J_full` 与 `J_marg`（`(root,bt2)` 边际）之间插值，检验"低支持收缩更多"是否比
**两种均匀收缩**（0.70 常数与 Train-only OOF 剂量常数）更有预测价值。

- 科学常量与停止条件：`PROTOCOL.md`（正文）+ `configs/experiment.yaml` + `code/sc_common.py`
- 结构准备/门槛/冻结：`uv run --group lipo python code/run.py prepare`
- 入口：`prepare / preflight / smoke / status / dispatch / analyze / report / tests`
- 正式清单：Lipo fit100 五臂 + fit25 五臂（条件 BACE）；一次冻结 key（`K_FINE`）
- 结果：`REPORT.md`、`analysis/*.json`、`analysis/main_table.csv`

上游只读：`lcrc_performance_route_8h_public122_s0_v1`（C1、`build_support_v2`、ERRATA）、
`lcrc_bottleneck_diagnosis_public122_s0_v1`（update-driven trainer、嵌套子集、norm、50 点 grid）、
`lcrc_lipo_model_dissection_v1`（`LocalContent` 与 `marginal_joint_agg`）、
`lcrc_lipo_s3_locality_public122_s0_v1`（别名对）、`lcrc_readout_diagnosis_closeout_public122_s0_v1`（阴性边界）。
