# ZPR-v1 预注册协议（在任何 GPU 结果之前冻结）

**Round:** `zinc-perf-recipe-pilot-v1` · **Track:** `tracks/structural-sufficiency/zinc_perf_recipe_pilot_v1/`
**冻结代码:** `code/zpr_{common,train,checks,smoke,phase_a,analyze}.py` + `tests/test_zpr_v1.py`
**数据:** 官方 PyG `ZINC(subset=True)` train 10,000 / valid 1,000 / test 1,000，raw-`y` MAE
**seed:** **仅 0**。本轮不运行 seed 1/2/3，不建超参网格，不新增结构模块，不改 LCRC-A 推理架构。
**正式轨迹:** 2 条（GPU0 = L1 控制，GPU1 = SmoothL1 候选），每条 600 epoch = 47,400 updates。
**GPU 硬上限:** 1.5 A100·h（含 smoke / 失败 / 计时）；两条轨迹合计。

> 本文件在 Phase B 任何 GPU 训练之前冻结。Phase A（零训练 test 失败样本分析）不改动任何
> 历史实验文件，只读历史预测与原始 PyG 图；其结论不参与 Phase B 的选择或调参。

---

## 0. 一句话

在冻结的 LCRC-A（237,519 参数）上，用**同一 seed 0、同一初始 state、同一数据顺序、同一优化器与日程**，
只把训练损失从 `L1` 换成 `SmoothL1(β=0.05)`，判定这是否是同 seed 下值得另立一轮确认的性能提升；
同时用**被动 EMA（decay 0.999，25% updates 后开始，不参与梯度）**回答"checkpoint 平均是否还有免费收益"。

---

## 1. Phase A — 零训练 test 失败样本分析

复用（只读）：

| 用途 | 路径 |
|---|---|
| LCRC-A s0 test 逐分子 | `zinc_local_relation_composition_v1/analysis/test_per_molecule_lcrcA_s0.csv` |
| DCC-L s0–s3 test 逐分子 | `zinc_performance_search_v1/analysis/test_pred_B-R0-s{0..3}_s{0..3}.csv` |
| LCRC valid 逐分子 / 预测 | `.../analysis/per_molecule_valid.csv`, `.../runs/lcrcA_s0/predictions/valid_best.npz` |
| DCC-L valid 逐分子 | `zinc_performance_search_v1/analysis/valid_pred_B-R0-s{0..3}_s{0..3}.npz` |
| 冻结分组掩码 | `zinc_generalization_audit_v1/analysis/fixed_worst10.json`, `per_molecule_valid.csv:r2_unseen_atom_frac` |
| 标签分项 provenance | `tracks/ksvd/results/zinc_long_cycle_audit/{test_cycle_audit_label.csv,formula_verification_per_molecule.csv,cache/gvae_full_properties.npz,upstream/*}` |
| 真实图结构 | `data/ZINC`（PyG `subset=True`，`processed/{train,val,test}.pt`） |

### 1.1 先复核的已知数字（必须逐位复现）

`LCRC-A s0 test 0.066782`；`DCC-L s0 test 0.066267`；50/50 融合 test `0.058831` / valid `0.070932`；
LCRC-A 与 DCC-L 四 seed 均值 50/50 融合 test `0.057300` / valid `0.067534`。
全部标为 **test-as-dev 历史分析**，本轮不是新盲测。

### 1.2 结构诊断对象

- 指定 11 行：`702,298,159,510,648,796,634,822,527,622,402`
  （与 LCRC-A s0 test AE 排名 1–10 加 rank 19 的 402 一致，402 同时是 DCC-L s0 rank 1 —— 记录，不作为选择）。
- LCRC-A s0 test AE top-20；DCC-L s0 test AE top-20；二者交集（"共同大误差"）。

只从**原始图**与**train 侧统计**提取：原子数、键数、环数/环大小、组成；
radius-1 / radius-2 typed rooted 环境的 train document frequency（DF）；是否存在训练中未见的
typed rooted 环境；train 最近结构邻居相似度（**只作描述，不解释为误差成因**）；
标签分项（logP / SA / cycle）重构与是否一致。

### 1.3 事前定义的判定词（不是按误差大小推断机制）

| 字段 | 定义 |
|---|---|
| `shared_failure` | 该行同时落在 LCRC-A s0 与 DCC-L s0 的 test AE top-20 内 |
| `cycle_penalty` | GVAE 目标里的 `cycle = -max(0, max(nx.cycle_basis(G)) - 6)`；`penalty_rows` 由 `label_excess = max_basis-6 > 0` 判定 |
| `r2_support` | 分子内 radius-2 typed 环境在 train 的 DF：`min` / `median`；`any_unseen = (min == 0)`；沿用冻结列 `r2_unseen_atom_frac > 0` 的口径 |
| `identity` | 图→RDKit 分子的身份校验：tier1 = Kekulé 键序精确同构；tier2 = 芳香键放宽同构；tier3 = 元素拓扑同构；再叠加 `|y_rec - y|`（标签重构误差）。任一失败则该行化学特征标 `UNKNOWN` |
| `chemical_pattern` | 仅在 identity 通过时给出：环系（环数 / 环大小多重集 / 最大基环）、是否含 ≥7 元环、稠合/桥/螺、自动 SMARTS 官能团计数；否则 `UNKNOWN` |
| `possible_explanation` | 见下 |

`possible_explanation` 只允许四类，按**优先顺序**判定（机制结论必须落在同一行内可核对的结构/标签事实上）：

1. `LABEL_OR_DATA_ANOMALY` — identity 失败，或 `|y_rec - y| > 0.05`，或该行 penalty 方向与两模型 signed error 同号且量级与 penalty 相当。
2. `SHARED_FAILURE` — `shared_failure` 且（`any_unseen` 或 `cycle_penalty`），并记录主因子。
3. `MODEL_COMPLEMENTARY` — 未同时落入两模型 top-20，且两模型 signed error 反号，或一方 AE ≤ 0.5×另一方。
4. `UNRESOLVED` — 以上都不成立。

**明确禁止**：把"近邻距离大"直接写成误差成因；把"误差大"本身当作机制证据。

### 1.4 Phase A 产出

`analysis/phase_a_baselines.json`、`analysis/phase_a_rows.csv`（全部 1000 行诊断）、
`analysis/phase_a_table.md`（短表）、`analysis/phase_a_summary.json`（三类解释的计数与成立/未成立清单）。

---

## 2. Phase B — 两条 seed-0 训练轨迹

### 2.1 两臂

| 臂 | GPU | 损失 | EMA |
|---|---|---|---|
| `G0` (L1 控制) | `res-gpu0` | `F.l1_loss(pred, y)` | 被动 `decay=0.999`，从 25% updates 后开始 |
| `G1` (候选) | `res-gpu1` | `F.smooth_l1_loss(pred, y, beta=0.05)` | 同上 |

**唯一训练差异** = 损失函数。其余全部冻结并对两臂逐项断言相等：
seed `0`、`build_model("A", …)` 初始 state（`11c698ae…`）、SSL 编码器（`006fabc1…`）、
train/valid cache（file+content SHA）、train-only normalization（`8c3e8813…`）、
`torch.randperm(10000, Generator(seed+101))` 每 epoch 重排、AdamW(lr 1e-3→1e-5, wd 1e-5)、
warmup 500、batch 128、79 steps/epoch、47,400 updates、grad clip 5、dropout 0、参数 237,519。

### 2.2 EMA 语义（逐字冻结，不允许换实现）

```
ema_start_step = ceil(0.25 * total_updates) = 11850          # = end of epoch 150
for step in 1..total_updates:
    optimizer.step()
    if step == ema_start_step:   ema_param <- online_param.clone()      # 初始化 = 该步之后的在线参数
    elif step > ema_start_step:  ema_param <- 0.999*ema_param + 0.001*online_param
```
EMA 分支在 `torch.no_grad()` 下运行、不产生梯度、不消耗 RNG（gate 断言 RNG 状态与 `gen` 状态不变）、
不改变 optimizer trajectory。EMA 模型对象在训练开始前一次性构建（在 `isolated_rng()` 内），
每 epoch 把 EMA 参数/buffer 拷入该对象后评估，**拷贝方向 = `module_param.copy_(ema_slot)`**
（历史 zps 轮次的 bug 是把新构建模型写进 EMA buffer，此处必须由 gate 排除）。

### 2.3 评估与选择

每个 epoch 记录 `online` 与 `ema`（EMA 在 start step 之前不存在，记为 `null`）的 official valid-1000 MAE。
**每条分支各自**用同一条规则选 checkpoint：strict argmin official-valid MAE，平局取更早 epoch。
报告项（对选定 checkpoint 的预测）：

`full (1000) | ordinary (y>-3) | r2-unseen ordinary | seen ordinary | minus row172 | minus fixed worst10 | W20 | T100 | train@best`

- `minus fixed worst10` = 去掉 `zinc_generalization_audit_v1/analysis/fixed_worst10.json` 的 10 个 row_id
  （与 LCRC 轮 `all_minus_worst10` 完全同一口径，逐行复用，不重新选）。
- `seen ordinary` = `r2_unseen_atom_frac == 0 且 y > -3`。
- `W20` / `T100` = 该分支 valid 曲线的最小 20-epoch 窗口中位数 / 末 100 epoch 中位数。

**主要比较**：`SmoothL1 online` vs `L1 online`（各自最优 checkpoint）。
**次要比较**：每条轨迹的 `EMA` vs **自身** `online`。禁止把某条轨迹 EMA 的最优值与另一条轨迹的非最优值混成机制结论。

### 2.4 判决规则（事前固定）

设 `Δfull = MAE(SmoothL1 online) − MAE(L1 online)`（负 = 候选更好），
`Δord`、`Δm172` = 对应分组差。

- `Δfull ≤ −0.003` 且 `Δord < 0` 且 `Δm172 < 0` → **`SEED0_CANDIDATE_SMOOTHL1`**
  （仅 seed-0 candidate，**不自动启动 seed 1**）。
- 否则 → **`NO_COMPELLING_RECIPE_GAIN`**（直接判定，不做二次调 β、不换 β、不加 epoch）。

EMA 的判决独立于上面：只有当"某条轨迹 EMA 最优 < 自身 online 最优 且 `Δfull ≤ −0.003`"时才允许写
"EMA 有实际收益"；仅 `Δfull < 0` 的 0.001 量级差异一律写作"在单 seed 执行漂移带内"。

### 2.5 正确性门禁（全部通过才允许 600-epoch 正式 run）

| 门禁 | 判定 |
|---|---|
| C1 初始 state 一致 | 两臂 `initial_state_sha256 == 11c698ae…`（与历史 LCRC-A s0 相同）且彼此相同 |
| C2 冻结输入一致 | cache file+content SHA、norm fingerprint、SSL SHA 全等于 LCRC 轮冻结值 |
| C3 数据顺序一致 | `first_batch_rows` 两臂相等、`data_order_seed == 101`、79 steps/epoch、47,400 updates |
| C4 L1 控制 = 原 LCRC 路径 | 2 epoch：冻结 `lc_train.train` 与 `G0` 的 valid/online MAE 逐 epoch `|Δ| < 1e-5`，参数漂移 `< 1e-3`；正式 run 结束后**整条 600-epoch 在线曲线**与历史 `runs/lcrcA_s0/curve.csv` 逐 epoch `|Δ| < 1e-5` |
| C5 SmoothL1 参考实现 | `β=0.05` 的数值/梯度与手算公式在 `|d| = 0, 0.02, 0.05, 0.2` 上 `< 1e-12`；小网络有限差分梯度相对误差 `< 1e-6` |
| C6 EMA 公式与存取 | CPU 参考递推与训练器逐位一致；`step == 11850` 时 EMA == 在线；EMA `state_dict` 往返 + `load_model_from_state` 推理复算 valid MAE `< 1e-6` |
| C7 EMA 被动性 | EMA 块前后 `torch.get_rng_state()` 与 `gen.get_state()` 逐位不变；`G0` 在线曲线 = 历史无 EMA 曲线（C4 同时覆盖） |
| C8 部署形态 | 两臂 `parameters == 237519`；EMA checkpoint 与 online checkpoint 各自是**单个** `state_dict`，无 ensemble/soup |

任一门禁失败 → 停止，写 `CPU_STOP_CORRECTNESS` / `GATE_STOP`，不救、不改 β、不缩短 horizon。

### 2.6 预算与停止

- 硬上限 **1.5 A100·h**（两卡并行，按 wall-clock × 卡数计）。
- 先 CPU gates（0 GPU），再 2-epoch smoke（< 0.02 A100·h），按 smoke 实测外推正式成本。
- 若外推 > 1.5 A100·h → 停止并报告成本，**不临时缩短正式训练 horizon**。
- 最多两条正式轨迹。禁止第三个模型、第二个 seed、超参网格。

### 2.7 Test 策略

历史 test 已暴露。只有当**全部训练与 valid 选择完成、checkpoint 冻结并写盘（含 sha256）之后**，
才允许执行**一次**标记 `TEST_AS_DEV_DESCRIPTIVE` 的 test 分组分析（同一份 `zps_test` 口径的分组：
full / ordinary / tail / minus row172 / minus fixed worst10）。本轮不据其调整 β、EMA decay 或重训。
该数字**不得**被称为新的独立 benchmark。

---

## 3. 产出

| 文件 | 内容 |
|---|---|
| `PROTOCOL.md` | 本预注册 |
| `code/zpr_*.py` | Phase A 分析、训练器、CPU gates、smoke、valid 分析、test 一次性读取 |
| `tests/test_zpr_v1.py` | fresh-clone-safe 纯数学测试（SmoothL1 参考、EMA 递推、分组掩码） |
| `analysis/cpu_checks.json` | C1–C8 的机器可读结果 |
| `analysis/phase_a_*` | Phase A 全部产物 |
| `runs/G0_s0`, `runs/G1_s0` | 两条正式轨迹（本地拉回）/ 远端 checkpoints |
| `analysis/main_table.{md,json}` | online/EMA 全部分组指标 |
| `analysis/decision.json` | 判决与理由 |
| `REPORT.md` | 最终报告（只回答任务书要求的 5 个问题） |
