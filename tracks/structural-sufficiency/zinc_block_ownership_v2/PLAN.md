# PLAN — `zinc_block_ownership_v2`（训练前冻结，2026-10-27T12:35+08:00）

**性质**：新一轮预注册。**不修改** `zinc_block_ownership_v1` 的任何结果、协议、判决或
停止记录（v1 仍然是 `UNRESOLVED_DUE_TO_VARIANCE`，seed-0 主门仍然 FAIL）。
v2 只回答一个新问题，并**复用 v1 冻结代码与冻结缓存**，不改块划分、模型宽度、
学习率、归一化、训练长度、参数量（497,161）、数据划分。

## 0. 问题

v1 里 R（真实 biconnected 块归属）比 N（全图均场）好 **+0.004884**（best-valid），
但没过 0.005 主门、CI 跨零、且 S 从未运行。于是两个解释未被区分：

* **H1（单键内容）**：收益来自单键块上的**逐键真实内容 + 非线性 φ**；
  多键块的归属并不重要。
* **H2（多键归属）**：收益来自**多键块之间真实的归属关系**（哪个键属于哪个环系）。

v1 的 S 臂正是为此设计的拆分：**单键块保留真实内容，多键块在分子内共享守恒均场**。
因此

```
d_RS = MAE(S)_best − MAE(R)_best      （正 = 真实多键块归属更好）
```

在 seed 0 上即可把 H1/H2 分开，且不需要新 seed。这一轮先买 S_seed0（0.5 GPU·h），
只有它在**实用规模**上为正、且不是单行造成，才买跨 seed 确认。

## 1. Phase A（CPU，无 GPU 授权）

用**已冻结**的 `R_s0` / `N_s0` 预测与 v1 缓存做五件事：

1. **交叉覆盖**：support tier（all_supported / low_support / any_unseen）× ordinary/tail ×
   block-size combo novelty × block-signature combo novelty 的列联表；并单独报告
   row 172 是否属于各 novel 子集、属于哪些（不猜、不筛）。
2. **贡献分解**：对每个子群给出 `contribution = Δ_group × n_group / N`，并验证
   `Σ contribution == Δ_total`；同时报告**全部**剔除方案（整体 / 去 row172 /
   去固定 worst-10 / 两者都去 / 去最大单行贡献者 / 去 top-5 / 去 top-10），
   **不挑选最有利的一种**。另报单行贡献占比与 leave-one-out 的最小/最大值。
3. **`n_multi` 分层**：`n_multi = 1 / 2 / ≥3`（`n_multi=1` 是 R/S 的**输入差异负对照**：
   此类分子只有一个多键块，S 的块内容与 R 逐位相同）。
4. **S 的块数据检查**（与 v1 同级）：守恒（每分子一阶/二阶总量）、
   `n_multi ≤ 1` 时 S 与 R 逐位相同、原子重编号/边反向/边序/batch 序不变性、
   同 seed 初值哈希与 R/N 相同、梯度非零、以及"同一权重下 arm 切换的预测一致性"
   （只在 `n_multi ≥ 2` 的分子上允许出现非零差）。
5. **冻结权重干预**：对 R_s0 与 N_s0 checkpoint 做固定权重内容替换
   `R→S`、`R→N`、`S→R`，报告 `mean|Δpred|`、分组 ΔMAE。
   **声明**：这类干预只证明**模型读取了什么**，不能证明泛化因果效应。

Phase A 的唯一停止权：若第 4 项任一硬检查失败（守恒/不变性/初值/梯度/一致性），
**不授权 S 训练**。其余诊断只产出描述性事实，不改变 Phase B 的阈值。

## 2. Phase B（1 个正式 run：`S_s0`）

* 复用 v1 冻结代码（`zbo_train.py --arm S`），v1 config `151deb02…` 与 v1 gates
  `177271fb…` 必须在 v2 运行时**逐位未变**；
* 复用 v1 冻结缓存（relation/block，按 `prep.json` 的 sha256 校验）；
* 唯一变化：`BC.RUNS` 指向 v2 的 `runs/`；产物目录 `runs/S_s0/`；
* 497,161 参数、47,400 updates、seed 0、`pretrain_encoder_s0.pt`、R0 配方、官方
  train 10k / valid 1k、strict-min valid 选模；
* **不**调任何超参、宽度、层数、块类型、readout、阈值。

### 2.1 Phase B 判决（预注册）

设 `d_RS = MAE(S) − MAE(R)`（best-valid；正 = R 更好）。

```
PROCEED_TO_PHASE_C iff
  d_RS >= +0.003
  and d_RS_W20 >= -0.002 and d_RS_T100 >= -0.002
  and the signal is NOT single-row driven:
      single_row_max_contribution < 0.50 * d_RS
      and min(d_RS | drop row172, d_RS | drop fixed-worst-10,
              d_RS | drop top-1 contributor) >= 0.5 * d_RS
  and mechanism_active(S)
```

否则 **STOP_GPU_AFTER_B**，结论写为
**“多键块归属未取得足以继续购买确认实验的证据”**——
**不写**“已证明多键块归属无效”（v1 的 0.004884 与 v2 的 S 都是单 seed 量级证据）。

分标签：

| 标签 | 条件 |
|---|---|
| `MULTI_BLOCK_OWNERSHIP_SUPPORTED_SEED0` | PROCEED 条件全部满足 |
| `SINGLETON_CONTENT_EXPLAINS_THE_GAIN` | `d_RS < 0.003` 且 `d_RN` 复现 v1 量级（S 与 R 等价） |
| `SINGLE_ROW_DRIVEN` | `d_RS ≥ 0.003` 但单行依赖条件失败 |
| `MECHANISM_INACTIVE_S` | S 的 block 梯度 / reader_ext / 内容敏感度任一门失败 |

## 3. Phase C（条件性，≤3 个 run）

只在 Phase B `PROCEED_TO_PHASE_C` 时执行。

1. **并行** `R_s1` + `N_s1`（两卡），seed 1 的 SSL（`pretrain_encoder_s1.pt`）、
   seed 1 初值、seed 1 数据顺序，47,400 updates。**不得**把 R_s0 与 N_s1 直接配对。
2. seed-1 停止条件：`d_RN(s1) = N−R < 0`（反向）或 `R_s1 best > 0.082`（绝对退化）
   → 立即停，标签 `R_N_NOT_REPLICATED`。
3. 若 seed-1 `R−N` 与 seed 0 同向（≥0），再跑 `S_s1`；检验 `d_RS(s1)` 与 seed 0 同号、
   两 seed 均值 ≥ +0.003。
4. **联合检查**：两 seed 的 R/N/S、W20/T100、逐分子 bootstrap、执行方差
   （同代码重复运行的漂移量级）、ordinary 与 any_unseen 的方向。

标签：

| 标签 | 条件 |
|---|---|
| `MULTI_BLOCK_OWNERSHIP_CONFIRMED_TWO_SEEDS` | 两 seed `d_RS` 同号且均值 ≥0.003，非单行驱动，ordinary 不反向 |
| `MULTI_BLOCK_OWNERSHIP_PARTIAL` | 两 seed 同号但均值 <0.003，或 any_unseen 反向 |
| `SINGLETON_CONTENT_ONLY_CONFIRMED` | 两 seed `d_RS` 都 <0.003 且 `d_RN` 两 seed 同向 |
| `R_N_NOT_REPLICATED` | seed 1 反向或 R 绝对退化 |

### 3.1 追加的次级量（Phase B 训练期间加入，非门槛）

`n_multi=1` 负对照只能说明"输入相同"，不能说明"模型相同"。因此在对比里追加一个
**次级、非判决**量：

```
DiD = Δ(n_multi >= 2) − Δ(n_multi <= 1)
```

它把"内容差异带来的差"与"两个模型整体的训练路径差异"分开。因为对照集只有 111 行
（v1 上该子集的 CI 宽达 ±0.03），**DiD 只报告、不参与任何门**。
加入时间：2026-10-27T12:38+08:00，**在 S_s0 训练进行中、任何 S 预测产出之前**；
因此不是看到结果后追加的分析。

## 4. 阈值纪律

* **0.005 仍是重要的效应参考尺度**；v1 的 0.004884 **不重新记为 PASS**，本轮的
  `d_RN` 也不因为重复出现 0.0049 而升级。
* 本轮 `d_RS` 用 **0.003** 作为"机制规模"门槛（任务书给定），理由是 R−S 只隔离
  多键块归属、效应天然小于 R−N；这不是为让旧数字通过而设的阈值，**不改动** v1 的 0.005。
* 任何 <0.003 的两臂单次对比按 `ERRATA.md` E2 的漂移带（0.0015–0.0025）读作“不可分辨”。

## 5. 预算与执行

| 项 | 预计 |
|---|---|
| Phase A（CPU） | 0 GPU·h |
| Phase B `S_s0` | 0.50 A100·h |
| Phase C `R_s1`+`N_s1` 并行 | 1.05 A100·h |
| Phase C `S_s1` | 0.50 A100·h |
| 合计 | **2.05 / 软 3 / 硬 4** |

* 全部远端执行经 `rr`（`doctor`/`deploy`/`run`/`pull`），正式 run 绑定干净 commit，
  不用 `--allow-dirty`/`--allow-stale`，不手工 SSH 提交作业。
* 只用 official train/valid；**official test 全程不构造、不加载、不评分**。
* 不开超参搜索、不加新网络、不蒸馏、不 ensemble、不做 seed 平均推理。
* 同一 official valid 已被多轮适应性研究使用：**即使 v2 两 seed 重复成功，也只能作为
  机制继续研究的理由，不是独立数据上的最终确认**。独立确认需要结构组合留出
  （见 REPORT 最后一节），本轮不启动。
