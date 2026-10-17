# REPORT — cscl-p3-slim-v1 (P3 vs P3-Slim)

**结论：`SLIM_NO_USEFUL_GAIN`。** P3-Slim 的 Monitor seed 均值 MAE 确实比同轮 P3 低
`0.004203`，达到预注册的 `≤ -0.004` 幅度；但两个 seed 方向相反、分子级 paired 95% CI 跨 0、
top-10-removed 反而略差，7 条冻结 gate 只满足 4 条。按预注册规则**停止**，Locked 标签保持未读，
不启动 full10k，不读取 official valid/test，也不做 hidden=64/96/160/192 等 rescue。

---

## 1. 主表（Monitor, n=1000, seed 均值；负 Δ = Slim 更好）

| 项目 | P3 | P3-Slim | 差异 |
|---|---:|---:|---:|
| 总参数 | 413,505 | **322,881** | −90,624（−21.9%） |
| Reader 参数 | 148,481 | **57,857** | −90,624（−61.0%） |
| Fit MAE s0 / s1 | 0.001264 / 0.001084 | 0.001303 / 0.001319 | 略差 |
| Fit MAE seed 均值 | 0.001174 | 0.001311 | +0.000137（+11.7%） |
| Monitor MAE s0 | 0.084178 | 0.085367 | **+0.001188**（s0 更差） |
| Monitor MAE s1 | 0.093493 | 0.083900 | **−0.009594**（s1 更好） |
| Monitor MAE seed 均值 | 0.088836 | **0.084633** | **−0.004203** |
| 分子级 paired 95% CI | — | — | **[−0.014906, +0.002712]**（跨 0） |
| ordinary MAE（y > −3） | 0.070833 | 0.069989 | −0.000844 |
| tail MAE（y ≤ −3, n=74） | 0.314111 | 0.267880 | −0.046231 |
| RMSE | 0.374088 | 0.229635 | −0.144453 |
| top-10-removed Δ | — | — | **+0.000771**（略差） |
| 分子更好比例 | — | — | 51.7% |
| seed MAE 极差 | 0.009315 | **0.001467** | −0.007848 |
| seed MAE 标准差 | 0.006586 | **0.001037** | −0.005549 |
| 训练 wall s0 / s1 | 536.3 / 653.1 s | 538.2 / 647.2 s | 基本相同 |
| 训练 wall 合计 | 1,189.3 s | 1,185.4 s | ~1.00× |
| 预测（两臂×两区/seed） | — | — | 约 9–10 s/seed |

**总 GPU ledger：1.6045 / 2.0 GPU-hours**（含 smoke、SSL、正式训练、预测，以及失败重试的
补记账；见 `DEVIATIONS.md`）。

## 2. 冻结 gate 逐条结果

| # | 条件 | 结果 |
|---|---|---|
| M1 | `Δ_seed_mean ≤ -0.004` | **PASS**（−0.004203） |
| M2 | 两个 seed 的 Δ 均 < 0 | **FAIL**（s0 +0.001188，s1 −0.009594） |
| M3 | paired 95% CI 上界 < 0 | **FAIL**（上界 +0.002712） |
| M4 | ordinary Δ < 0 | **PASS**（−0.000844） |
| M5 | top-10-removed Δ < 0 | **FAIL**（+0.000771） |
| M6 | RMSE 退化 ≤ +0.005 | **PASS**（−0.144453，明显改善） |
| M7 | tail Δ ≤ +0.005 | **PASS**（−0.046231） |

4/7 通过，但 M2/M3/M5 是稳定性和“非极端分子驱动”的核心条件，不能因为 M1 刚好过线而忽略。

## 3. 逐问回答

### 3.1 P3-Slim 是否在 Monitor 上稳定超过 P3？Locked 是否复现？

**没有稳定超过。** seed 均值改善 `−0.004203`，但 seed 0 是 **+0.001188**（Slim 更差），
seed 1 是 **−0.009594**（Slim 更好），方向相反；paired CI 从 −0.0149 到 +0.0027，跨 0。
按预注册规则 Monitor 已失败，因此 **Locked 没有读取**，不存在 Locked 复现结论。

### 3.2 训练拟合是否变差？是否换来更好的 held-out MAE？

**Fit 略差，held-out 均值略好，但不足以称为稳健增益。** Fit MAE seed 均值从 0.001174
升到 0.001311（+11.7%）；Monitor seed 均值从 0.088836 降到 0.084633。Fit–Monitor gap
从 0.087662 降到 0.083322，但**gap 减小本身不是成功**；本轮 gate 明确要求 held-out 的
方向、CI 和非极端分子一致性，而这些没有同时满足。

### 3.3 改善或退化是否覆盖普通分子，还是只被极端分子主导？

**主要是极端误差/尾部驱动，不是普通分子的广泛改善。** ordinary MAE 只改善 `0.000844`
（约 1.2%），分子更好比例仅 51.7%；top-10-removed 后 Δ 变为 **+0.000771**（Slim 略差）。
RMSE 从 0.3741 降到 0.2296，主要反映 P3 的单个极大误差（P3 max AE 10.19，Slim 5.79），
而不是普通分子整体变好。tail MAE 改善 −0.0462 也说明尾部重排贡献了大部分均值改善。
因此不能宣称“容量约束改善了普通分子的泛化”。

### 3.4 每 seed 的方向和结果是否一致？

**不一致。** P3 / Slim 的 Monitor MAE：

- seed 0：0.084178 → 0.085367（**变差 +0.001188**）
- seed 1：0.093493 → 0.083900（**变好 −0.009594**）

虽然 seed 间波动从 0.009315 降到 0.001467（这是一个值得记录的观察），但两个 seed 的相对
方向相反，不能当作稳定增益。seed 间方差下降也不能替代“同向 + CI 排除 0”的预注册条件。

### 3.5 是否值得进入 full10k？下一步应停止什么？

**不值得。** 冻结判定为 `SLIM_NO_USEFUL_GAIN`；按协议：

- 不读 Locked，不读 official valid/test；
- 不启动 full10k，也不提出“先跑 full10k 再看”的补救；
- 不跑 hidden=64/96/160/192、Reader 层数、dropout、loss、seed、优化器或 soup 窗口搜索；
- **Reader 缩窄这一具体干预停止**。若未来还要研究容量，需要新的预注册和新的独立问题，
  不能把本轮结果升级为“Reader 过拟合已被证明”。

## 4. 设计与执行摘要

| 项目 | 内容 |
|---|---|
| 协议 | `cscl-p3-slim-v1`；split U；显式 salt `cscl-p3-slim-v1` |
| 数据 | 官方 ZINC-12K train 10k，排除 `941, 2210, 6071, 9766`；9,996 行 |
| split U | 7,996 / 1,000 / 1,000；hash `9c303656720780ea…`；与 E/F/S/T 无 zone-hash 碰撞 |
| 暴露披露 | `historical_rows_seen_fraction = 1.0`；U 是同一 10k 池的新重采样，不是独立盲测 |
| 初始化 | canonical P3；复制四个公共模块；Slim Reader 用 `seed + 10000` 独立初始化 |
| 公平性 | 同 seed 公共模块 init hash、SSL encoder、batch order、22,680 updates 均断言相同 |
| 监督 | raw-y L1、Adam、cosine 1e-3→1e-5、wd 1e-5、clip 5、batch 128、360 epoch、last-5 soup |
| CPU 门禁 | G1–G8 全过；8 个 pytest 全过；P3 新 loop 与冻结 trainer 的 2-epoch soup 逐位相同 |
| 预测冻结 | 8 个预测 + 4 soup + 2 SSL，combined `7c407aeefc2a9ae8…`；之后才读 Monitor |
| 标签 | 只读 Monitor；Locked 未读；official valid/test 未读 |

## 5. 解释边界

1. 本轮只否证“把 P3 Reader 从 256 缩到 128 能稳定改善 held-out”这一具体干预，**不能**证明
   Reader 容量与泛化无关，也不能证明 Reader 过拟合是原模型的因果瓶颈。
2. P3-Slim 的平均 MAE 改善伴随明显的 seed 不稳定和 top-10-removed 退化；这不是“接近成功”，
   而是未通过预注册稳定性门槛。
3. 同一 10k 池已被历史多轮使用；split U 不是独立盲测，不能把本轮结论外推为官方 test 结论。
4. 两臂参数量不同，本轮不是等参数/等 FLOP 比较；Slim 训练 wall 与 P3 基本相同（Reader 只占
   总计算的一小部分），不能宣称显著的计算节省。
5. 两个 seed 对 0.004 量级效应的分辨力有限；但 M2/M3/M5 同时失败，停止决策不依赖微小效应。

## 6. 产物

`results/p3_slim_v1/`：`PROTOCOL.md`、`protocol_freeze.json`、`split_U.json`、
`split_compare_U.json`、`initialization_audit.json`、`parameter_inventory.json`、
`cpu_gates_U.json`、`prediction_manifest_U.json`、`monitor_score_U.json`、
`monitor_gate_U.json`、`decision.json`、`gpu_budget_ledger.json`、`evidence_manifest.json`、
`predictions/*.npz`、`REPORT.md`、`DEVIATIONS.md`、`AMENDMENTS.md`。

提交：实验冻结/两次预训练前 amendment 为 `8bada73`；最终报告与证据清单见后续 commit。
