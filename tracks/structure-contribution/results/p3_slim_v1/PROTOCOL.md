# PROTOCOL — cscl-p3-slim-v1 (P3 vs P3-Slim)

**状态**：预注册（在任何本轮 GPU 轨迹开始之前冻结）
**Track**：`structure-contribution`；**Protocol id**：`cscl-p3-slim-v1`；**Split tag**：`U`
**冻结方式**：`protocol_freeze.json` 绑定本文件、`configs/p3_slim_v1.yaml`、两份代码文件、
split / objects / prep / init audit / parameter inventory / CPU gates / input audit 的 sha256。
冻结后任何语义改动都会使 `verify_freeze` 失败。

---

## 0. 一句话问题

**在完全保留 P3 的结构编码、SSL、训练样本、优化配方、batch 顺序和 update 预算的前提下，
把图级 Reader 从 hidden 256 缩小到 128，能否稳定降低 held-out MAE？**

这是固定架构的假设检验，不是宽度/超参数搜索。主指标始终是 held-out MAE；Fit–Monitor gap
减小本身不算成功。

## 1. 两个模型臂

| 臂 | Reader | Reader 参数 | 总参数 | 唯一差异 |
|---|---|---:|---:|---|
| `P3` | `321 → 256 → 256 → 1` | 148,481 | **413,505** | 无 |
| `P3Slim` | `321 → 128 → 128 → 1` | 57,857 | **322,881** | Reader 宽度 |

- 两臂的 Atom / Bond / Junction encoder、Null composer、输入结构特征、321 维 Reader 输入构造、
  Reader 深度 2、SiLU、dropout 0、归一化统计全部相同。
- 共享模块参数总数 265,024；Reader 减少 90,624，总参数减少 90,624。
- 参数量必须由代码实际实例化和 `named_parameters()` 重算，不能依赖手填；`parameter_inventory.json`
  与 `initialization_audit.json` 记录实际值。
- 不使用 GNN / Transformer / Attention / 迭代消息传递 / ensemble / 外部数据 / 目标分量标签 /
  penalized-logP 公式特征。

## 2. 初始化与公平性

同 seed：

1. 先创建 canonical P3 初始化（reader 256）。
2. 用独立固定 RNG `seed + 10000` 创建 Slim（reader 128）。
3. 从 canonical P3 **逐位复制** atom / bond / junction / composer 四个公共模块及两个 buffer。
4. Slim Reader 保持自己的初始化；创建 Reader 消耗的随机数不会改变公共模块（复制在构造之后）。
5. 两臂装入同一个 Fit-only SSL encoder checkpoint。
6. 训练 minibatch 顺序由独立固定 RNG（seed + 101）控制；两臂同 seed 每一步索引相同。
7. 不要求完整模型 hash 相同（Reader 形状不同）；要求公共模块 init hash、encoder hash、
   batch-order hash 和 update budget 相同。

保存：shared-module init hashes、Reader init hashes、SSL encoder hashes、batch-order hashes、
parameter inventory。不得用历史其他 split 的 P3 checkpoint 作为本轮直接比较基线；本轮在同一
split U 上重新训练同轮 P3。

## 3. 数据与划分

- 只使用 PyG `ZINC(subset=True)` official **train** 10,000 行；不读取 official valid/test。
- 剔除历史 4 行输入核对异常行 `941, 2210, 6071, 9766`，9,996 行可用。
- 身份 = 由 PyG 图重建的 canonical constitution SMILES；与 committed canonical SMILES 的身份
  划分逐行相同。
- 新 grouped 划分 U：目标 `fit ≈ 7,996 / monitor = 1,000 / locked = 1,000`（±10，按身份组
  贪心填充）。实际大小记录在 `split_U.json`。
- **显式 salt** `cscl-p3-slim-v1`，不使用 builder 默认值。必须断言：
  1. 实际 salt == 保存 JSON salt == 硬编码常量；
  2. 用保存的 salt + identity 元数据独立重算，三区逐行相同；
  3. 保存的 `split_sha256` 与重算一致；
  4. 换 salt probe 会改变划分；
  5. U 的任一区行哈希不与历史 E/F/S/T 的任何区相同，且 U 不与历史任一轮逐行相同。
- `split_compare_U.json` 记录 U 与 E/F/S/T 的完整行重叠矩阵。**披露**：同一 official-train 10k
  总体已被历史多轮使用；U 是新的内部重采样评估，**不是独立盲测**。不得把新 salt 说成新独立数据。
- 当前实际 split hash：`9c303656720780eab8f41f5b48113fd7d538267c16d4d7806ee14e27fb324fe9`；
  尺寸 7,996 / 1,000 / 1,000；历史行覆盖比例 1.0。

## 4. SSL 与监督

- SSL：仅 Fit 结构，masked atom+bond，7,560 updates，batch 128，Adam 1e-3，wd 1e-5，clip 5；
  每 seed 一个 SSL checkpoint，两臂共享。不引入新 pretext；不用 Monitor/Locked 结构做 SSL。
- 监督：raw-y L1，batch 128，R3 = 360 epoch cosine，Adam 1e-3→1e-5，wd 1e-5，clip 5，
  last-5 parameter soup。`steps_per_epoch = ceil(7996/128) = 63`，总 update = **22,680**；
  实际 update 按真实 Fit 数量计算并在冻结协议中断言。
- 不改变学习率日程、epoch 数、loss、监督 update 预算或 soup 成员；两臂 update 数相同。
- 不做宽度/lr/dropout/epoch/seed 搜索；不追加 seed。

## 5. CPU 正确性门禁（GPU 前必须全部通过）

| gate | 内容 |
|---|---|
| G1 | P3 = 413,505；Slim = 322,881；Reader 148,481 vs 57,857；差异全部来自 Reader |
| G2 | Atom/Bond/Junction/Null composer 架构一致；公共模块 init 逐位相等；无公共通路失活 |
| G3 | 新 driver 的 P3 forward 与冻结 P3 实现一致；2-epoch soup / batch order / update 与冻结 trainer 等价 |
| G4 | 同 batch、同公共参数下，两臂 Reader 输入前 321 维逐位相同；不要求训练后相同 |
| G5 | 非退化 synthetic batch 上 Slim Reader 与三个 encoder/composer 均有有限非零梯度 |
| G6 | atom relabel、bond endpoint swap、batch composition、save/load replay 不变性 |
| G7 | 测试与正式 train 路径不能读取 official valid/test 或 Monitor/Locked 标签 |
| G8 | 两臂相同 update budget、相同 batch-order 记录、相同 last-5 soup 语义；Fit 规模下 update = 22,680 |

门禁使用合成分子与合成目标，不加载 ZINC、不读任何 split 标签、不用 GPU。结果写入
`cpu_gates_U.json`；`freeze` 要求 `all_passed`。

## 6. GPU 执行与预算

- `res` 单机：GPU0 跑 seed 0，GPU1 跑 seed 1；每卡串行 SSL → P3 → P3Slim → predict。
- 正式 GPU 预算上限 **2.0 GPU-hours**（smoke、SSL、训练、预测、失败重试均计入 ledger）。
- 先运行 smoke（20 update SSL + 1 epoch mini run），实测吞吐。若全轮投影超过 **1.8 GPU-hours**，
  在正式训练前停止并报告预算不足；不删 seed、不减 epoch、不改配方。若投影在 1.8–2.0 之间，
  仍可执行但必须如实记录；超过 2.0 则不得开始正式训练。
- 所有远端执行走 `rr`，不手工 SSH 启动后台训练。

## 7. 预测冻结与标签防火墙

1. 两臂 × 两区 × 两 seed 共 8 个预测文件写盘；同时记录 4 个 soup、2 个 SSL checkpoint、
   parameter audit、batch-order / split hashes 的 sha256。`freeze-predictions` 计算合并预测
   哈希并写 `complete=true`。
2. **先只评分 Monitor**：读取 Monitor `y`，创建 `holdout_read_monitor.lock`；不读取 Locked `y`。
3. 若 Monitor gate 失败：写 `decision.json = SLIM_NO_USEFUL_GAIN`，停止；Locked 保持未读，
   不购买 Locked 评价，不训练 full10k，不读 official valid/test，不跑 hidden=64/96/160/192
   等 rescue，不追加 seed/optimizer/loss/dropout 改动。
4. 只有 Monitor gate 通过，才读取 Locked `y`（单次，创建 `holdout_read_locked.lock`），按同一
   gate 评分。

主指标：raw-y MAE，逐分子配对差异 `Δ = |pred_Slim - y| - |pred_P3 - y|`（负 = Slim 更好）。
报告 seed 均值、逐 seed 方向、分子级 paired bootstrap 95% CI（5,000 次，分子为抽样单位，
固定 seed 0）、RMSE、median/p90/p99、ordinary (`y > -3`)、tail (`y <= -3`)、top-10-removed、
Fit MAE、Fit–Monitor gap、seed 间波动、参数量与计算成本。

- seed-mean 是分别计算两个单模型的 MAE 后求均值；禁止平均两个 seed 的预测。
- bootstrap 对每个分子的两 seed 平均误差差值重采样，不把两个 seed 的分子当成 2,000 个独立样本。
- top-10 的定义：按 **P3 在该评估区的两 seed 平均绝对误差**取最大的 10 个分子，两臂删除同样
  10 行后重算 Δ。

### 7.1 冻结 gate（Monitor 与 Locked 相同，全部满足才通过）

| # | 条件 |
|---|---|
| M1 | `Δ_seed_mean ≤ -0.004` |
| M2 | seed 0 与 seed 1 的 Δ 均 < 0 |
| M3 | 分子级 paired 95% CI 上界 < 0 |
| M4 | ordinary Δ < 0 |
| M5 | top-10-removed Δ < 0 |
| M6 | `RMSE_Slim - RMSE_P3 ≤ +0.005` |
| M7 | `tail Δ ≤ +0.005` |

### 7.2 判定

- Monitor 失败 → `SLIM_NO_USEFUL_GAIN`（stage=monitor，Locked 未读）。
- Monitor 通过但 Locked 失败 → `SLIM_GAIN_NOT_REPLICATED`。
- Monitor 与 Locked 均通过 → `SLIM_GENERALIZATION_GAIN_SUPPORTED`；**允许提出一次** full10k
  确认方案，但不自动执行，不授权 official test。
- 若 Fit MAE 上升而 Locked MAE 下降，只能报告“与容量约束改善泛化的假设一致”，不能声称
  Reader 过拟合的因果机制已经完全确定。

## 8. 禁止后验补救

无论成功与否，严禁：宽度网格搜索、增减 Reader 层数、结构模块添加、换 loss/SAM/ASAM、增加
weight decay/dropout、EMA/SWA/soup 窗口搜索、剔除高误差分子后重选模型、在 Locked 上挑 seed、
用 official valid/test 决定下一步、把更小的泛化 gap 本身称为成功。若无稳定净增益，Reader 缩窄
这一具体干预正式停止。

## 9. 已知限制（预注册即披露）

1. 同一 10k 总体已被历史多轮使用；split U 不是独立盲测。
2. 只有两个 seed；对很小的效应分辨力有限，0.004 阈值沿用历史冻结尺度。
3. 两臂参数量不同；本轮不声称等参数/等 FLOP 比较，Reader 宽度是唯一干预。
4. Fit MAE 的差异可能受 Reader 容量和末端 LR 共同影响；不能把 Fit–Monitor gap 的变化单独
   当作机制证据。
5. 本轮不读取 official valid/test；结论只针对内部 held-out 划分。

## 10. 交付

`results/p3_slim_v1/`：`PROTOCOL.md`、`protocol_freeze.json`、`split_U.json`、
`split_compare_U.json`、`pairing_verification.json`、`input_audit_U.json`、
`objects_manifest_U.json`、`prep_fit_U.json`、`parameter_inventory.json`、
`initialization_audit.json`、`cpu_gates_U.json`、`staged_inputs_U.json`、`smoke/smoke_U.json`、
`ssl_U_s*.json`、`train_{P3,P3Slim}_U_s*.json`、`train_round_U_s*.json`、
`prediction_manifest_U.json`、`monitor_score_U.json`、`monitor_gate_U.json`、`decision.json`、
`gpu_budget_ledger.json`、`evidence_manifest.json`、`predictions/*.npz`、`REPORT.md`（简洁
中文科研判断）。若 Monitor 通过，另有 `locked_score_U.json` / `locked_gate_U.json` 与
`holdout_read_locked.lock`。
