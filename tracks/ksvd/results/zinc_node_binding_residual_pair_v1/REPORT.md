# ZINC 节点绑定残差配对训练 v1 — REPORT / DECISION

日期：2026-10-02 · 提交：task/zinc-nodebind-residual-v1 (e8c9315c621d) · 官方 test 从未实例化

实际总耗时：首次工具调用 14:41:54Z → 交付约 16:0xZ（≈80 min）；其中正式两 arm 并行 GPU 训练各 ≈42 min（10.5 s/epoch × 240）。

## 结论摘要

| 项 | control (product) | candidate (residual) |
|---|---:|---:|
| 末5soup calibrated valid MAE | 0.109042 | 0.114261 |
| raw valid MAE | 0.110221 | 0.114024 |
| train-fit bias | -0.019201 | -0.005941 |
| 240轮节点通道终态 | 塌缩 | 塌缩 |

- **calibrated gain (control − candidate) = -0.005219**；raw gain = -0.003804；gain_without172 = -0.003041。
- **购买决定：FAIL（本轮不购买后续 seed）**（门槛：cal gain ≥0.003，gain_without172 >0，G0 贡献恶化 ≤0.001）。
- **机制判定：MECHANISM_FAILED：candidate 节点通道仍在训练中塌缩，未达到稳定绑定目标。**

## 1. 父节点死亡是否在 fresh control 重现？

是。相同 canonical fresh 初始化下，product control 的节点槽在训练中自锁死亡：

| epoch | W_A_S rel_change | slot RMS | slot zero_frac | node_out const_frac | task grad W_A_S |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | 2.418e-04 | 0.000 | 0.000 | 2.961e-04 |
| 1 | 0.778 | 1.118e-04 | 0.000 | 0.000 | 8.113e-05 |
| 10 | 1.000 | 1.057e-05 | 0.009 | 0.264 | 2.233e-06 |
| 40 | 1.000 | 1.158e-10 | 0.890 | 1.000 | 1.205e-16 |
| 80 | 1.000 | 0.000e+00 | 1.000 | 1.000 | 0.000e+00 |
| 160 | 1.000 | 0.000e+00 | 1.000 | 1.000 | 0.000e+00 |
| 240 | 1.000 | 0.000e+00 | 1.000 | 1.000 | 0.000e+00 |

说明父 Full soup 的 denormal 死亡不是 soup/续训产物：从 fresh 初始化、无监督旧权重、240 轮完整协议下同样发生，
且在 epoch 40–80 之间完成（slot 先降 ~1e-10，epoch 80 起恒 0，W_A_S/W_A_C/node_encoder.0 的 task 梯度在 epoch 40 已降到 ≤1e-16）。

## 2. candidate 是否恢复通道及交叉项？

| epoch | slot RMS | slot zero_frac | node_out const_frac | task grad W_A_S | product RMS | struct-add RMS | atom-add RMS | cross-mix ΔRMS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.440e-04 | 0.000 | 0.000 | 2.824e-04 | 1.566e-04 | 1.568e-04 | 1.590e-04 | 1.641e-04 |
| 1 | 1.677e-04 | 0.000 | 0.000 | 3.456e-04 | 4.397e-05 | 7.128e-05 | 9.657e-05 | 3.781e-05 |
| 10 | 3.863e-05 | 0.000 | 0.049 | 2.733e-05 | 4.613e-06 | 2.079e-05 | 8.633e-06 | 3.540e-06 |
| 40 | 2.156e-07 | 0.565 | 1.000 | 1.568e-09 | 8.643e-11 | 1.797e-07 | 7.862e-09 | 6.049e-11 |
| 80 | 3.153e-19 | 0.948 | 1.000 | 1.476e-20 | 0.000e+00 | 2.731e-19 | 2.249e-20 | 0.000e+00 |
| 160 | 0.000e+00 | 1.000 | 1.000 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 240 | 0.000e+00 | 1.000 | 1.000 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |

加性项确实把梯度路径延长了：epoch 1 candidate 的 `grad_W_A_S`（3.46e-4）与 `grad_node_encoder_first`（2.12e-4）
分别是 control 的 ~4x / ~6.5x，epoch 10 时 slot RMS 仍为 3.86e-5（control 1.06e-5）。
但 **两 arm 的 `node_out` 都在 epoch 40 变成逐列常量（const_frac=1.0）**，而 candidate 的 slot 到 epoch 80 才实质消失：
即下游 fusion 的常量吸收早于节点槽死亡，加性项没能改变这条自锁路径。
**结论：残差只延后、未阻止节点路径自锁死亡；乘法交叉项也未保持可学习（epoch ≥80 三项 RMS 全为 0）。**

## 3. 性能收益是否超出 bias / id172？G0 如何？

| group | n | control MAE | candidate MAE | gain | control contrib | candidate contrib | Δcontrib |
|---|---:|---:|---:|---:|---:|---:|---:|
| G0 | 965 | 0.087976 | 0.091917 | -0.003941 | 0.084897 | 0.088700 | +0.003803 |
| G1 | 34 | 0.210942 | 0.188448 | +0.022493 | 0.007172 | 0.006407 | -0.000765 |
| G172 | 1 | 16.973145 | 19.154438 | -2.181293 | 0.016973 | 0.019154 | +0.002181 |

- 组贡献之和闭合：True（等于全体 MAE）。
- G0 贡献恶化 = +0.003803（门槛 ≤0.001，FAIL）。
- 唯一单项改善在 G1（34 行，+0.0225），被 G0 的 -0.0039 与 id172 的 -2.18（单行）抵消；id172 上 candidate 仍显著更差。
- signed error 约定 `pred−y`；bias 差 = -0.013260（candidate 的 train 中位残差只有 control 的约 1/3）。
- 参照：本轮 fresh control 的 calibrated valid 0.109042 好于历史父 soup 0.115066；后者是 top-5-by-valid 选模、
  本轮的固定末 5 轮是更公平的协议，这也是本轮 control 更适合作为对照的原因。

## 4. 假设关闭 / 未知

- **关闭**：'纯乘法绑定的塌缩可由固定幅值加性残差梯度路径修复'——candidate 同样塌缩，加性项只延后了死亡，未改变终态。
- **未关闭/未知**：塌缩是否由 weight decay + 下游常量吸收的优化几何主导；如果冻结 node_encoder 或从下游切断常量吸收，通道能否存活；性能差异在完整 240 轮下是否稳定。

## 5. 下一步（只一个动作）

**关闭本固定幅值残差绑定配置，停止；不再扫系数 / residual 形式。**
若要继续研究主张，需新 preregistration 针对塌缩机制（例如切断下游常量吸收或改变 node 路径正则），而不是继续在本绑定家族内变体。

## 6. 执行记录

- 服务器 skill：`/home/calendar/.pi/agent/skills/remote-research-runner/SKILL.md`
- 主机 `res-2`、pool `res2-cu124`；两 arm 各 1×A100-PCIE-40GB、8 CPU、独立 Slurm 作业并行。
- 实际 regime：node `c05`，driver `525.85.12`，torch `2.5.1+cu124`（CUDA 12.4），python 3.12.14；两 arm 同 regime、FP32、无 AMP/DDP。
- 固定系数（来自 canonical shared init `93c2f23c…`，1024 图 fixed seed-0 抽样）：
  `eta_s=0.0059866898`、`eta_a=0.0026590702`、`gamma=0.5436830673`；
  `rms_u0=1.5571e-03`、`rms_s=2.6009e-01`、`rms_a=5.8558e-01`；
  固定系数按公式构造，标定集（1024 图）上缩放后槽 RMS 匹配误差 = 0.0；
  occurrence 145,510，slot 条目 71,220；graph-id sha256 `6b884fdf…`。
- 初始节点槽 RMS：标定集 control 2.419e-03 vs candidate 缩放后 2.419e-03（精确匹配）；
  固定诊断 batch 上按图聚合 RMS：control mean 4.050e-03 / std 7.298e-04，candidate mean 4.122e-03 / std 4.529e-04
  （匹配整体 RMS 不保证分布等同，这正是干预的一部分）。
- 训练 commit `e8c9315c621d`（branch `task/zinc-nodebind-residual-v1`，未 push）；`protocol_hash=52b50e0bdd50`；
  split fingerprint `58c69506…f28a`（与参考一致）；`test_access=blocked`；`git.dirty=false`。
- rr 作业：`zinc-nodebind-control-s0-20261002-230823-fb319cba`（Slurm 55767）、
  `zinc-nodebind-residual-s0-20261002-230854-14cb88f8`（Slurm 55768）；smoke `zinc-nodebind-smoke-s0-…3a839273`（55766）。
- 控制面 run：`20261002-230635-289d3c4a`（control）、`20261002-230705-78710657`（residual），已 `research promote` → `records/runs/`。
- 本地部署提交：隔离分支 `task/zinc-nodebind-residual-v1` 上两个 commit（`15724a0` 协议/实现、`e8c9315c` smoke 导出），未 push/merge；
  原工作区既有未提交审计/结果保持原样。
- 官方 valid 仅记录；**官方 test 从未实例化/加载/评估**。
