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

## 结果（2026-11-14，单 seed 0，DEVELOPMENT_TEST）

- 冻结 `K_FINE`，`KEYS_PASS_BOTH_N`；正式训练 10 条（Lipo fit100 五臂 + fit25 五臂），
  **未采购 BACE**（Lipo 未过晋升门槛 → `NOT_PURCHASED_BY_BUDGET`）。
- 判定：**`SPLIT_OR_SELECTION_SENSITIVE`** —— fit100 上 SELECT 相对 BASE 在 Test 改善 2.76% 但在
  Valid 恶化 1.17%；fit25 上两 split 都是正（+4.47% / +3.25%）但 MAE 在 Test 恶化 1.2%；
  bootstrap（canonical-structure / Murcko 簇）区间除 fit100 Valid 的 SELECT-vs-UNIFORM 外全部跨 0。
- 门控确实被使用：`a=1` 反事实改变 90.2%（Test）/92.6%（Valid）分子的预测，p50|Δp|=0.025。
- 全部正式训练在本机 CPU regime 完成（`res` 共享 `/home` 写满，`rr deploy res` 失败），
  见 `REPORT.md` §7 与 `analysis/run_manifest.json`。
