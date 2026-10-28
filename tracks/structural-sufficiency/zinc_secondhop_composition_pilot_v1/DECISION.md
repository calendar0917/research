# DECISION — second-hop composition pilot v1

**日期:** 2026-10-28
**轮次:** `tracks/structural-sufficiency/zinc_secondhop_composition_pilot_v1/`
**判定（PROTOCOL §6 机械门槛）:** 阶段一 `STAGE1_GATE_NOT_MET`（条件 0 未过）→ **不购买阶段二**
**实质读法（primary 指标 = official-valid 全量 raw-y MAE）:** `NO_PREDICTIVE_GAIN_THIS_CONFIG`
**预注册:** commit `73ebbb2`（写代码/上 GPU 之前）；GPU 运行 commit `97b360f`

## 决定

1. **停止"固定投影 + 128 维第二跳组成注入"这一配置。** primary 指标
   Real_s0 = 0.078976、Null_s0 = 0.082897，冻结 A_s0 = 0.076582；**两个 arm 都不优于 A**
   （+0.00240 / +0.00632）。按预注册决策表，这对应"当前固定投影和接入方式没有预测收益"。
2. **不购买阶段二（seed 1）。** 条件 0（不崩）要求 `max(Real, Null) ≤ A_s0 + 0.005 = 0.081582`，
   实际 0.082897（Null）→ FAIL。虽然信号 B 单独通过（Real−Null full = −0.00392），
   但门槛是合取式，我**没有事后放宽**条件 0。无 seed 1、无阶段二、无第三阶段。
3. **不据此否定"所有可学习投影"，也不重启 DFPB。** 本轮只测了一个固定随机草图
   （`P_b/P_a`，seed 20261105，96→32）和一种接入方式（零初始化 128 列、原配方、600 epoch）。
   上游审计对 DFPB 的 `REDUNDANT / NO-GO` 不受影响；本轮也没有为 DFPB 提供任何支持证据。
4. **不把 Real 相对 A 的普通行改善（−0.00105）当作结论。** 它低于预注册的 0.003 购买门槛，
   也小于历史 A 的 seed 间差 0.00392；在单 seed 下连符号都未确立。

## 本轮确立了什么

- **第二跳统计被网络实际使用**：step-1 新列梯度 = 0.197× 旧列（128/128 列非零）、
  `W_new/W_old = 0.589`（best）、`z_new/z_old = 0.296`、冻结权重下 C 通道切换改变 100% 分子的预测
  （mean |Δpred| 0.0371，MAE 0.078976→0.091999）。所以"注入被忽略"不是解释。
- **真实配对相对均场配对有小而方向一致的优势**：Real−Null full = −0.00392，
  drop row172 = −0.00236，drop fixed-worst10 = −0.00273，r2-unseen ordinary = −0.00383，
  ordinary = −0.00131。**但** 95% CI [−0.0107, +0.0021] 跨零，full 效应集中在 5 个高 `|y|` 行
  （尾部 top-5 占 64%），ordinary 的中位 Δ 是 +0.0008。这是"配对通道非空"，不是"配对买到 MAE"。
- **"第二跳内容"本身在这个注入方式下是净损失**：Null−A = +0.00632（full），tail +0.0933；
  Real 只是把损失减小了约 2/3，并未翻正。所以本轮否掉的是**这一版组成注入**，
  不是"第二跳信息无用"这一强命题。
- 机制上：新块输入 RMS ≈ 旧 X（1.49 vs 1.52），`n_v=1` 的 3.8% 原子上 `C_real ≡ C_null`，
  `n_v≥2` 的 96.2% 原子上两者不同——Real/Null 的差异确实只来自多路径 root 的配对。

## 本轮没有确立什么

- **没有确立 Real−A 的符号**（单 seed；A 的 seed 间差 0.00392 > 观测差 0.00240）。
- **没有确立"真实配对无用"**。恰恰相反，方向证据一致（5/5 预定义组 + 2 个稳健变体）；
  只是效应量在噪声带内、且尾部集中，达不到本次预注册的购买门槛。
- **没有测 official test**，没有任何 test 结论。
- **没有测可学习投影**、没有测不同维度/接入方式/正则化，因此不能推广到"所有第二跳组成方案"。
- **没有测 DFPB**。本轮的注入是无参数固定投影，与 DFPB 的学习式 Q/K/V 不是同一对象。

## 为什么不是"再来一个变体"

用户规则：两个 arm 都不优于 A ⇒ 停止该配置，不因为一个预注册实验失败就盲目加局部组成算子。
本轮唯一的广而小改善（ordinary −0.0010、r2-unseen ordinary −0.0055）**看起来**像"配对在未见组成上
更有用"的线索，但它 (a) 未达预注册阈值、(b) 单 seed、(c) 伴随尾部 +0.052 的代价。
把它变成新实验需要**新的预注册**（新的假设、新的门槛、新的 seed 预算），而不是本轮门槛的重新解释。

## 允许的后续（需用户点头，且必须重新预注册）

1. **"配对 × 未见组成"定向检验**（如果用户认为 r2-unseen ordinary 的 −0.0055 值得追）：
   预注册单一假设 `Real < Null` 只在 `r2_unseen_ordinary` 上，固定种子集合（≥3 seeds）、
   在 600 epoch 冻结配方下直接检验，并把尾部（尤其 row172 与 `y≤−3` 组）作为**共同 primary
   安全性指标**（要求 tail 不恶化超过 +0.005）。本轮数据只够当作它的先验，不够当作它的证据。
2. **尾部机制诊断**（CPU 即可）：为什么额外的第二跳自由度会系统性恶化高 `|y|` 行——
   是 `|y|` 大的分子 path 更多（`n_v` 更大）导致新块方差更大，还是优化/容量分配问题。
   这可以在不训练的情况下用冻结 A 的 `h`/梯度做探针，成本 0 GPU。
3. **什么都不做**（默认）：把本轮记为"第二跳组成在当前接入方式下不构成 A 的 MAE 瓶颈"，
   回到其他方向。

## 复现与证据

- 报告：`REPORT.md`（含 Real−Null / Real−A / Null−A、ordinary / r2-unseen / tail、drop row172、
  drop fixed-worst10、best epoch / W20 / T100 / train MAE、C 切换、新列权重/梯度/激活、SHA、GPU 时间）。
- 机器可读：`analysis/summary.json`、`analysis/stage1_gate.json`、`analysis/provenance.json`、
  `analysis/provenance_gpu.json`、`analysis/per_molecule_valid.csv`、`analysis/tables.md`。
- CPU 门禁：`analysis/cpu_checks.json`（8/8 PASS，0 GPU）。
- 预算：**1.048 GPU·h**（1.046 两次正式运行 + 0.002 smoke），硬上限 3。
