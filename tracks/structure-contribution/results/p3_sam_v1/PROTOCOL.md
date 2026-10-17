# PROTOCOL — cscl-p3-sam-v1 (P3 vs P3-SAM)

**状态**：预注册（在任何本轮 GPU 轨迹开始之前冻结）
**Track**：`structure-contribution`；**Protocol id**：`cscl-p3-sam-v1`；**Split tag**：`T`
**冻结方式**：`protocol_freeze.json` 绑定本文件、`configs/p3_sam_v1.yaml`、两份代码文件、
split / objects / prep / CPU gates / input audit 的 sha256。冻结后任何语义改动都会使
`verify_freeze` 失败。

---

## 0. 一句话问题

**当前 P3 的泛化缺口，能否只通过改变监督优化的平坦性偏置（标准两步 SAM），而不增加结构信息、
模型容量或新的监督信号来缩小？**

本轮不重开 SAC；不改变任何旧轮次冻结结果。A 臂是历史冻结的 P3-R3 配方，B 臂只把 Adam 外层换成
固定 `rho=0.05` 的标准两步 SAM。

## 1. 两个模型臂

| 臂 | 模型/初始化 | SSL | 监督样本 | Adam 更新次数 | 唯一差异 |
|---|---|---|---|---|---|
| `P3` | 413,505 参数 BJOC-Null，P3 scratch init | 同 seed Fit-only masked SSL | 同 split T Fit | 22,680 | 无 |
| `P3SAM` | 与 P3 逐位相同的 pre-SSL init + 同 SSL encoder | 同一 SSL checkpoint | 同 split T Fit | 22,680 | 标准两步 SAM |

同 seed 两臂必须通过代码断言验证：pre-SSL full/group/encoder 哈希相同、SSL encoder 哈希相同、
batch order 哈希相同、update budget 相同。

### 1.1 标准两步 SAM（唯一改动）

对每个 minibatch：

```
g_t   = grad L(w_t)                         # 第一次 backward
eps_t = rho * g_t / (||g_t||_2 + 1e-12)     # 全局 L2，所有有梯度的参数
w_t'  = w_t + eps_t
g_t'  = grad L(w_t')                        # 第二次 backward，同一 minibatch
w_t   = w_t                                 # 精确恢复（clone/copy）
clip(g_t', 5.0)                             # 只在第二次 backward 完成后
Adam.step(g_t')
```

* `rho=0.05` 固定；**不**搜索 rho，不用 ASAM，不换扰动范数，不加正则/dropout/loss 改动。
* 第一次梯度不裁剪，仅用于生成扰动；裁剪只发生一次，且发生在第二次 backward 之后。
* SAM 每 update 有 2 次 forward + 2 次 backward；成本必须如实记录，不宣称与 P3 等计算量。
* 训练 MAE 曲线统一取 clean forward（第一次 forward），使 A/B 拟合曲线定义一致；同时记录
  adversarial forward 的 MAE。

## 2. 数据与划分

* 只使用 PyG 官方 `ZINC(subset=True)` official **train** 10,000 行；不读取 official valid/test。
* 剔除历史 4 行 SMILES/PyG 输入核对异常行 `941, 2210, 6071, 9766`，9,996 行可用。
* 身份 = 由 PyG 图重建的 canonical constitution SMILES；与 committed canonical SMILES 的
  身份划分逐行相同。
* 新 grouped 划分 T：`fit ≈ 7,996 / monitor = 1,000 / locked = 1,000`（±10，按身份组贪心填充）。
* **显式 salt** `cscl-p3-sam-v1`，传给 builder 时显式写出，不使用默认值。必须断言：
  1. 实际使用的 salt == 保存 JSON 中的 salt == 硬编码常量；
  2. 用保存的 salt + identity 元数据独立重算，三区行完全一致；
  3. 保存的 `split_sha256`（salt + 三区 int64 行）与重算一致；
  4. 换 salt probe 会改变划分；
  5. T 的任一区行哈希不与历史 E/F/S 的任何区相同，且 T 不与历史划分逐行相同。
* **披露**：同一 official-train 10k 总体已被历史多轮使用；split T 不是独立盲测，只提供
  新的重采样稳健性证据。`split_compare_T.json` 记录与 E/F/S 的重叠矩阵和“曾被历史区间使用过”
  的行比例。

## 3. SSL 与监督

* SSL：仅 Fit 结构，masked atom+bond，7,560 updates，batch 128，Adam 1e-3，wd 1e-5，clip 5；
  每 seed 一个 SSL checkpoint，两臂共享。
* 监督：raw-y L1，batch 128，R3 = 360 epoch cosine，Adam 1e-3→1e-5，wd 1e-5，clip 5，
  last-5 parameter soup。`steps_per_epoch = ceil(7996/128) = 63`，总 update = 22,680。
* 不做宽度/lr/dropout/epoch/rho 搜索；不追加 seed。

## 4. CPU 正确性门禁（GPU 前必须全部通过）

| gate | 内容 |
|---|---|
| G1 | SAM 参数扰动后精确恢复：恢复后与原始 state 逐位相同（max abs error = 0） |
| G2 | 扰动全局 L2 范数 = `rho` |
| G3 | optimizer step 数 = 计划 update 数；scheduler step 数一致 |
| G4 | 裁剪只在第二次 backward 后调用一次；裁剪看到的是第二次梯度；第一次梯度不裁剪 |
| G5 | last-5 soup：成员为最后 5 个 epoch，soup 等于其参数均值 |
| G6 | `rho=0` 的 SAM 循环与冻结 P3 trainer 的 soup 逐位相同、batch order 相同、update 数相同 |
| G7 | 实际 Fit 规模下 update budget = 22,680，且 seed 0/1 batch order 哈希不同 |

门禁使用合成分子与合成目标，不加载 ZINC、不读任何 split 标签、不用 GPU。门禁结果写入
`cpu_gates_T.json`，`freeze` 要求 `all_passed`。

## 5. GPU 执行与预算

* `res` 单机：GPU0 跑 seed 0，GPU1 跑 seed 1；每卡串行 SSL → P3 → P3SAM → predict。
* 正式 GPU 预算上限 **3 GPU-hours**（smoke、SSL、训练、预测均计入 ledger；超限即停止）。
* 先运行 smoke（20 update SSL + 1 epoch mini run），实测吞吐并按每 seed 的 SSL + 360 epoch
  P3 + 360 epoch P3SAM + 预测外推；投影超过 2.7 h 则停止，不进入正式训练。

## 6. 预测冻结与评分顺序

1. 两臂 × 两区 × 两 seed 共 8 个 own-arm 预测文件写盘；同时记录 soup/SSL checkpoint 的
   sha256。`freeze-predictions` 计算合并预测哈希并写 `complete=true`。
2. **先只评分 Monitor**：读取 Monitor `y`，创建 `holdout_read_monitor.lock`；不读取 Locked `y`。
3. 若 Monitor gate 失败：写 `decision.json = SAM_NO_USEFUL_GAIN`，停止；Locked 标签保持未读，
   不购买 Locked 评价，不追加训练。
4. 只有 Monitor gate 通过，才读取 Locked `y`（单次，创建 `holdout_read_locked.lock`），按同一
   gate 评分。

主指标：raw-y MAE，逐分子配对差异 `Δ = |pred_B - y| - |pred_A - y|`（负 = B 更好）。
报告 seed 均值、逐 seed 方向、分子级 paired bootstrap 95% CI（5,000 次，分子为抽样单位）、
RMSE、median/p90/p99、ordinary (`y > -3`)、tail (`y <= -3`)、top-10-removed、seed 间波动。

### 6.1 冻结 gate（Monitor 与 Locked 相同）

| # | 条件 |
|---|---|
| M1 | `Δ_seed_mean ≤ -0.004` |
| M2 | 两个 seed 的 Δ 均 < 0（同向） |
| M3 | 分子级 paired 95% CI 上界 < 0 |
| M4 | ordinary Δ < 0 且 top-10-removed Δ < 0（方向一致） |
| M5 | `RMSE_B - RMSE_A ≤ +0.005` 且 `tail Δ ≤ +0.005`（无灾难性退化） |

top-10 的冻结定义：按 **P3 的 seed-mean 绝对误差**取最大的 10 个分子，两臂同时移除后重算 Δ。
ordinary/tail 使用同一批分子级 Δ 的掩码。

### 6.2 判定

* Monitor 失败 → `SAM_NO_USEFUL_GAIN`（stage=monitor，Locked 未读）。
* Monitor 通过但 Locked 失败 → `SAM_NO_USEFUL_GAIN`（stage=locked）。
* Monitor 与 Locked 均通过 → `GENERALIZATION_OPTIMIZATION_PROMISING`；**允许提出一次**
  full10k 确认，但不自动执行，不授权 official test。
* 无论结果如何：不重启 rho/seed/优化器变体搜索；不把 SAM 收益解释成新结构信息；不声称
  平坦极小值已被证明是原模型瓶颈。

## 7. 已知限制（预注册即披露）

1. 同一 10k 总体已被历史多轮使用；split T 不是独立盲测。
2. SAM 的额外前反向计算不计入“等计算量”比较；成本单独报告。
3. SAM 只改变优化轨迹的平坦性偏置；它不增加输入信息、参数或监督样本。
4. 两个 seed 对很小的效应分辨力有限；0.004 阈值沿用历史冻结尺度。
5. 本轮不读取 official valid/test；因此任何结论都只针对内部 held-out 划分。

## 8. 交付

`results/p3_sam_v1/`：`PROTOCOL.md`、`protocol_freeze.json`、`split_T.json`、
`split_compare_T.json`、`input_audit_T.json`、`objects_manifest_T.json`、`prep_fit_T.json`、
`cpu_gates_T.json`、`staged_inputs_T.json`、`smoke/smoke_T.json`、`ssl_T_s*.json`、
`train_{P3,P3SAM}_T_s*.json`、`train_round_T_s*.json`、`prediction_manifest_T.json`、
`monitor_score_T.json`、`monitor_gate_T.json`、`decision.json`、`gpu_budget_ledger.json`、
`evidence_manifest.json`、`predictions/*.npz`、`REPORT.md`（简洁中文科研判断）。
若 Monitor 通过，另有 `locked_score_T.json` / `locked_gate_T.json` 与两个 holdout lock。
