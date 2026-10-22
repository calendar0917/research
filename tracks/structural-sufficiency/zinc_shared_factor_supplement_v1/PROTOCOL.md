# zinc_shared_factor_supplement_v1 — T vs N（seed 1）补充机制实验：预注册

**性质**：`zinc_shared_factor_v1` 的**补充 study**，只回答一个机制问题。它**不修改**上一轮
的 `A_FAIL` 判定，**不修改**任何门槛，**不**被当作原协议自动进入阶段 B/C，也不覆盖旧记录。
旧协议/代码/结果保持原样（`../zinc_shared_factor_v1/`，提交 `a9253ba`）。

注册时间：2026-10-22（先于任何本补充实验的训练）。所有内容在训练开始前冻结。

## 1. 唯一问题

> 在**相同参数（451,777）、相同初始化、相同 batch 顺序、相同 recipe** 下，共享非根原子状态的
> **T** 是否优于把非根原子状态独立复制的 **N**（supervised seed 1）？

附带观察：T 在**另一个监督训练 seed（seed 1）**下的绝对 valid MAE 水平是否保持。

明确不在问题内：D（上一轮已有两次 draw）、seed 方差估计、超参数搜索、结构家族/通道/状态数、
任何 test 集表现。

## 2. 复用与冻结项（不改动）

| 项 | 值 |
| --- | --- |
| 代码 | 上一轮 `sf_models.py` / `sf_core.py` / `sf_runner.py` **逐字节未改**（本次只新增文档与分析脚本） |
| 配置 | `../zinc_shared_factor_v1/configs/sf_atomic_v1.json`，sha256 `d1f6b67fcd3339bedf0e278fbe09a5e2d57ac842d9b969465574a65123cdabc8`（含 seed 1 的冻结 init hash） |
| 模型 | C=16 通道，R=4 状态，共享 `A[28,C,R]` 与 `K`（3 个键类型槽），133 维内容输入，`Linear(133,64)→SiLU→Linear(64,32)`；T/N 参数 451,777 |
| 结构家族 | P3 载体 + `bf_data` S3/S4/F4 模式（4/5 节点），不增删 |
| 接入口 | 198→128 zero-init `W_s`（`w_s_zero=True`，gate 逐一验证） |
| 初始化 | 只加载 `pretrain_encoder_s0.pt`（sha256 `006fabc1…bfa97`）；seed 1 的冻结值 T = N：init `db2792ca…`，content `7b023184…`，public carrier `21963b43…` |
| 优化器 / 调度 | AdamW 1e-3→1e-5（per-step cosine，warmup 500），wd 1e-5，clip 5，全参数训练 |
| 数据 / 顺序 | ZINC subset train 10,000 / valid 1,000；batch 128，79 steps/epoch，600 epochs = 47,400 updates；`shuffle_seed_offset=101`，两臂同 seed 的 batch 顺序必须逐 epoch 相同（runner 与 gate 检查） |
| 选择方式 | 单一 checkpoint，严格 `<` 取最低 valid MAE；无 EMA/soup/SWA/早停 |
| 确定性设置 | **不改**（不改 `torch.use_deterministic_algorithms`、不改内核/求和顺序、不改 seed 流），因此与旧结果同条件 |
| 测试数据 | **不访问 official test**（runner 无 test 命令） |

**唯一重复的随机性**：监督训练的 seed（0 → 1）。SSL 仍是同一个 train-only seed 0 checkpoint，
不重训 SSL、不做独立 SSL 重复。

**N 必须独立训练**：N 从自己的初值开始跑满 47,400 updates；**不允许**用训练好的 T 在推理时
切换成 N 来代替对照。

## 3. 运行

| 臂 | 含义 | 池 | 运行名 |
| --- | --- | --- | --- |
| T seed1 | 共享非根原子状态 | `res-gpu0` | `sf-supp-T-s1` |
| N seed1 | 非根边独立状态副本 | `res-gpu1` | `sf-supp-N-s1` |

两臂同提交、同环境，并行。N 的 gate 要求配对 T 的 `analysis/SF_T_s1/init.json` 存在且 hash 一致，
因此 T 先启动、N 在 T 通过 gate 后立即启动（串行时间 <1 分钟，不影响并行主体）。

**预算**：每臂 ≤1 GPU 小时（含 4-epoch timing、检查、打分），本补充任务合计 ≤2 GPU 小时；
重试与检查计入；不扩大预算；若某臂投影超过 3600 s，按其既有逻辑停止并如实报告，不缩短候选。

## 4. 预先声明的判定规则

**Q1（T vs N，seed 1）**：只有以下三条**同时**满足才记为「支持共享机制」：
1. `best_valid(T,s1) < best_valid(N,s1)`；
2. `best20_window_median(T,s1) < best20_window_median(N,s1)`（20-epoch 滑动中位数，避免孤立最低点）；
3. 逐分子配对 ΔMAE `= MAE(T) − MAE(N) < 0`（同一批 1,000 个分子，方向与 1、2 一致）。

只满足部分条件 → 记为「方向不一致 / 不成立」，不给部分学分。**不**用分子级 bootstrap
区间做 seed 级推断；区间只解释为固定模型下的分子抽样不确定性。

**Q2（T 的绝对水平在 seed 1）**：报告 T1 的 best/final/best20/tail100 与 T0 的同项数值和
0.075 参考线；不据此重设门槛、不宣布通过/不通过原门槛（原判定已固定为 A_FAIL）。

**Q3（225 个 unseen+ordinary 分子）**：报告 T0/T1/N1/D0 在该 cell 的 MAE、绝对变化量
（ΔMAE）与对总体 MAE 的净贡献 `(n_cell/1000)·ΔMAE_cell`；历史 P3 的 0.133939597 只作
**描述性参照**，不冒充同轮配对基线（P3 不是本轮训练出来的 arm）。

**Q4（停止 vs 再提一次有限确认）**：只按 Q1–Q3 的结果回答，预先声明：
- Q1 成立 → 可以提出**一次**有限确认（明确写清要买什么证据），但仍需新的预注册；
- Q1 不成立 → 停止该具体方案（shared-atomic-state 作为 P3/LF 线的改进），不再调超参数；
- Q1 方向不一致 → 明确记录不可判定，停止而不是加 seed 去凑。

## 5. 分析计划（全部实现于 `code/supplement_analysis.py`）

- 主指标：完整 valid MAE（全部 1,000 个分子保留），best / final / best20 中位数 / tail-100 中位数 /
  train@best / 每 epoch 秒数 / GPU 小时。
- 保留 **T0 与 T1 两行**（不只展示更好的 seed），并列出 D0（上一轮）与 N1 作上下文。
- 四个 radius-2 seen/unseen × ordinary/tail 分组（分组来自
  `zinc_restricted_adaptation_v1/analysis/reference_preds_valid.csv`，冻结）。
- 分组去分量：每 cell 的 `n`、MAE(各臂)、ΔMAE、`(n/1000)·ΔMAE` 净贡献，并核对
  `Σ_cell (n_cell/1000)·ΔMAE_cell == ΔMAE_overall`（避免混淆分母）。
- 极端样本诊断（仅诊断，不改主结果）。
- 类型覆盖（atom/bond 类型是否在 train 出现过；bond type 3 是否仍无数据）。

## 6. 预先声明的限制（写进报告的固定段落）

1. **单 seed 的 T/N 对照**只能说明「在 seed 1 这一次训练里，共享比独立复制更好/更差」，
   不是 seed 级推断；两个 seed 也不构成 seed 方差估计。
2. 没有 D seed1，因此 T1 的绝对水平**不能**与「seed 1 更难/更易」分离；T1 vs N1 因为同 seed、
   同 init、同 batch 顺序，才隔离出机制差异。
3. 上一轮 D 与更早 LF-D 的 0.0026 差值只是**一次观测到的差异**，不是噪声下界；
   CUDA 非确定性是合理解释之一，但现有证据不足以证明是唯一原因（见勘误）。
4. 本轮不改确定性设置；因此与旧结果仍是同条件比较，但**不是**逐位可复现。
5. 不做：门槛重设、seed 增加、超参数扫描、结构/通道/状态数变更、official test 访问。

## 7. 交付

`analysis/{summary.json, main_table.md, preds_valid_TN_s1.csv}`、`REPORT.md`（本补充研究的简短报告）、
以及追加到 `../zinc_shared_factor_v1/REPORT.md` 的**勘误**段（保留旧正文，不改历史结论）。
