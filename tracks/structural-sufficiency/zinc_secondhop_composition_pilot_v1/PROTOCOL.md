# PROTOCOL — Second-Hop Composition Pilot v1

**Round id:** `zinc-secondhop-composition-pilot-v1`
**Track:** `tracks/structural-sufficiency/zinc_secondhop_composition_pilot_v1/`
**状态:** 预注册（在任何 GPU 时刻之前冻结；本轮不读 official test）
**上游审计:** `tracks/structural-sufficiency/zinc_path_binding_sufficiency_audit_v1/REPORT.md`（结论：`D = T_real − T_null` 广泛存在、不在 X 里、不是类型级、不是图级坍缩；建议的唯一最小实验就是"第二跳组成"注入）
**复用主体:** `tracks/structural-sufficiency/zinc_local_relation_composition_v1/`（LCRC-A，只读：代码、cache、A_s0/A_s1 checkpoint 与 valid 预测）

---

## 0. 问题

**LCRC-A 的剩余误差，能否通过补充第二跳 learned bond 内容、以及它与终点内容的真实配对来降低？**

这**不是**继续训练 DFPB，**不是**重新引入 learned junction token，**不是**新的关系 reader。本轮只改
`LocalContent` 的输入列，其余一切冻结。

可判别的对照是"同一批第二跳内容、同一 path count、同一输入尺度，只换配对"：
`C_real`（真实配对）vs `C_null`（同根均场、同边际）。两者都由同一固定随机投影给出，
训练期间不更新，不由 valid 标签选择。

---

## 1. 设计

### 1.1 冻结主干

`y = Σ_v f(h_v) + ρ(global33, mean(h), mean(h²), size)`，其中

```
h_v = LN(MLP([a_v, Σ_e b_e, mean_e b_e, S1joint_v, S2_path, S2_content, scalars_v]))   (64-d)
b_e = bond_encoder([a_u+a_v, |a_u-a_v|, a_u⊙a_v, bond_feat_e])
a_v = atom_encoder(atom_feat_v)
```

frozen：atom/bond encoder（含历史 masked-SSL 初始化，正常微调）、S1/S2 typed 统计、
`f`、`ρ`、训练配方、数据顺序、checkpoint 选择规则。不新增 junction token、attention、
迭代消息传递、独立关系 reader、目标函数分项。

**历史初值一致性（硬门禁）**：新模型的主干构造顺序、随机种子与历史
`build_model("A", seed)` 完全相同；CPU 门禁要求新旧主干在 seed 0/1 上的
`state_hash` 逐位相等，并等于历史 `runs/lcrcA_s{0,1}/train.json` 记录的
`initial_state_sha256`（`11c698ae…` / `d9422859…`）。做不到就不与 A 比较。

### 1.2 第二跳内容（唯一的输入扩展）

从真实有根二跳路径 `p = (v, w, u)` 取第二跳键 `e2 = (w, u)`。`b_{e2}` 由**同一个**
`bond_encoder` 作用在 `(a_w, a_u, bond_feat_{e2})` 上得到（不新增 encoder）。
`n_v` = 该根的有序非回溯二跳路径数（与冻结 cache 的 `path_ptr` 逐原子一致）。

```
B_v        = (1/√n_v) · Σ_p b_{e2(p)}                            ∈ R^96
U_p        = P_b · b_{e2(p)}       (P_b: 96→32, 固定, 不训练)      ∈ R^32
V_p        = P_a · a_{u(p)}        (P_a: 96→32, 固定, 不训练)      ∈ R^32
C_real(v)  = (1/√n_v) · Σ_p (U_p ⊙ V_p)                          ∈ R^32
C_null(v)  = √n_v · (mean_p U_p) ⊙ (mean_p V_p)                  ∈ R^32
```

* `n_v = 0`：`B_v = C_real(v) = C_null(v) = 0`
* `n_v = 1`：`C_real(v) = C_null(v) = U_p ⊙ V_p`（恒等；逐位校验）
* `C_null` 的均值**只在同一根内部**取：不跨 root、不跨分子、不跨 batch。
* `n_v`、`{U_p}`、`{V_p}` 的边际在 Real/Null 之间**完全相同**，唯一差别是配对。

新输入列：`[B_v, C(v)]`，共 **128** 维，接在原 529 维 `X_v` 之后（657 维）。

### 1.3 固定随机投影

* 单一冻结种子 `20261105`：`torch.Generator(seed)` 先 `randn(96,32)/√96` 得 `P_b`，
  再同样得 `P_a`。
* 两个投影是 **buffer**（`requires_grad=False`），所有 arm / seed / 阶段共享同一组；
  训练期间不更新；**不使用任何 valid 标签或 MAE 选择投影种子**（不做投影种子搜索）。
* 投影的 sha256、RMS 记录进 `analysis/cpu_checks.json` 与 `analysis/provenance.json`。

### 1.4 两个 arm

| arm | 输入扩展 | 参数结构 | 其余 |
|---|---|---|---|
| **Real** | `[B_v, C_real(v)]` | 完全相同 | 同初始化、同数据顺序、同配方 |
| **Null** | `[B_v, C_null(v)]` | 完全相同 | 同上 |

* Real/Null 在同一 seed 下 full-state hash 必须相同（投影同、零列同、主干同）。
* **零初始化新列**：把历史 `Linear(529,192)` 原位扩宽为 `Linear(657,192)`，
  新 128 列的权重全零、bias 不变、其余张量逐位等于历史 A 初始化。因此
  **step-0 输出 = A**（三个 arm 在 step 0 上逐位相等，max|Δ|=0）。
* step-1 必须观测到新列梯度非零；随后新列权重必须离开零（"真正被使用"的判据之一）。

### 1.5 参数上限

A = 237,519；新模型 = 237,519 + 128×192 = **262,095**（投影是 buffer，不计参数）。
硬上限 500k。

### 1.6 明确禁止

attention / learned junction token / per-wedge token / 迭代 hidden-state 更新 /
独立关系 reader / 新 encoder / 新目标分项 / 目标加权 / EMA / soup / seed averaging /
集成 / 蒸馏 / 宽度或深度搜索 / 投影种子搜索 / 关系类别选择 / 第三阶段。

---

## 2. 数据与训练配方

* 官方 PyG ZINC `subset=True`：train 10,000 / valid 1,000。**不另划分数据，不读 test。**
* 复用冻结 LCRC cache（只读）：train `8777494f…4c3a6cb2`（content `ded004e8…`），
  valid `aae2caa5…b7a6dbfe`（content `f28a0cea…`），feature version
  `lcrc-v1-atom108-s1joint-s2typedpath`。
* 第二跳补充表（`path_mid`, `path_bond2` 等）由冻结 cache 的原始邻接重新枚举生成，
  并与 cache 的 `path_cat / path_end63 / path_end / path_closure` 逐条比对（§3 G1）；
  生成的补充表记录 sha256，训练进程启动时校验。
* 标准化复用冻结 `analysis/norm.json`（fingerprint `8c3e8813…`）。新列**不做**额外标准化
  （避免引入新的可搜索量）；其 RMS 作为诊断报告（§3 G6）。
* 配方（与 LCRC-v1 逐字一致）：AdamW，peak LR 1e-3，min LR 1e-5，500 update warmup，
  per-step cosine，weight decay 1e-5，batch 128，600 epochs = 79 steps/epoch =
  **47,400 updates**，raw-y L1，clip 5，dropout 0，每 epoch 全量 valid，
  严格最小 valid MAE 选一个 checkpoint（平局取更早 epoch）。
* 数据顺序：`torch.randperm(10000, Generator().manual_seed(seed+101))`，每 epoch 重打乱，
  最后一个不满 batch 保留；Real/Null 同 seed 的顺序逐 epoch 相同（门禁校验）。

---

## 3. 训练前 CPU 门禁（全部 PASS 才允许启动 GPU）

| gate | 内容 | PASS 条件 |
|---|---|---|
| **G1** 路径枚举 | 从冻结 cache 的原始邻接重新枚举有序非回溯二跳路径，逐条比对 cache 的 `path_cat/path_end63/path_end/path_closure`、每原子 path count、`nbr_code`；检查 `root≠end`、`e1≠e2`、`e1` 关联 `(v,w)`、`e2` 关联 `(w,u)`；与审计 `analysis/paths.json` 的路径数一致（train 692,716 / valid 69,040） | 全部 mismatch = 0 |
| **G2** 历史初值一致性 | 用新代码重建 base A（扩宽前），比较 seed 0/1 的 `state_hash` 与历史记录；比较全部张量逐位相等；扩宽后除 `local.mlp.lins.0.weight` 新 128 列（必须全 0）外所有参数逐位相等；参数 262,095 ≤ 500k；无 junction/attention 模块 | 逐位相等；step-0 三 arm `max|Δy| = 0.0`（CPU 同一批） |
| **G3** Real/Null 同构同初值 | full-state hash 相等；两 arm 的参数形状表相同；用同一 batch 跑 1 步、30 步：全部梯度有限、新列梯度非零、loss 下降、新列权重范数离开 0；checkpoint 保存→重载→预测逐位相等 | 全部成立 |
| **G4** 不变性（非零新权重） | 原子重编号、键方向反转、batch split/merge（128 vs 7 vs 1）、batch 顺序置换：`y`、`h`、`B`、`C_real`、`C_null` 跨设置最大差 | ≤ 1e-5 |
| **G5** 边际/配对语义 | 同一 root：`n_v`、`B_v`、`Σ_p U_p`、`Σ_p V_p` 在 Real/Null 上逐位相等；`n_v≤1` 时 `C_real == C_null`（逐位）；`n_v≥2` 的真实 root 上 `C_real ≠ C_null`；显式构造"同边际不同配对"反例（`B`、`C_null`、path count 全同，`C_real` 不同，模型输出不同）；in-root shuffle 反例（打乱 U/V 配对 → `C_real` 变、`C_null` 与 `B` 不变） | 全部成立 |
| **G6** 真实数据尺度/覆盖 | train/valid 上报告 `n_v` 分布、`RMS(B)`、`RMS(C_real)`、`RMS(C_null)`、原 529 维输入 RMS 参考、`C_real≠C_null` 的原子占比 | 报告数值；`RMS(C_real)/RMS(C_null) ∈ [1/2, 2]`（否则标记 SCALE_MISMATCH 并如实报告，不调参） |
| **G7** 训练/评估路径正确性 | 用历史 A_s0/A_s1 checkpoint 载入扩宽模型（新列补零）后 valid MAE 与冻结值差、预测最大差；冻结 npz 重算 MAE；test split 在本轮任何代码路径上不可达（拒绝构建/加载） | `|ΔMAE| ≤ 1e-6`（相对 0.0765816392 / 0.0805037804）；test 拒绝 |
| **G8** 标签/预算守卫 | 全部门禁不读 valid 标签做任何选择；`official_test_loaded=false`；参数上限、GPU 预算写在 manifest | 成立 |

门禁脚本：`uv run python code/sh_cpu_checks.py` → `analysis/cpu_checks.json`。
任何一项 FAIL：**不启动 GPU**，本轮以 CPU 负结果收尾。

---

## 4. GPU 阶段与购买门槛（数值预注册）

主机 `res`（2×A100-40GB），pool `res-gpu0` / `res-gpu1` 并行，commit + `rr deploy` 后启动。
证据等级：`res` 单机 = 一个 execution regime；数字只与同一 regime 上的历史 A 比较。
历史 A_s0/A_s1 的 valid 数字来自同一台 `res`（`lcrc-A-s0` / `lcrc-A-s1`，commit
42c01a3 / b936765），因此同 regime。

**阶段一（并行）：`Real_s0`（gpu0）、`Null_s0`（gpu1）**，配置冻结。预计每条约 0.4–0.5 GPU·h。

参考值：`A_s0` frozen valid MAE **0.0765816392**（epoch 476，预测文件在盘上）。

**购买门槛（"探索信号"，不是统计显著性）：**

先定义变体集 `V = {full, drop_row172, drop_fixed_worst10}`（同一 1,000 行、同一 per-molecule AE）。
`drop_fixed_worst10` 使用 `zinc_generalization_audit_v1/analysis/fixed_worst10.json` 的固定
10 行（`[917,753,172,659,260,678,31,224,447,229]`，在其它模型上预先固定）。

* **条件 0（不崩）**：`max(MAE_full(Real), MAE_full(Null)) ≤ MAE_full(A_s0) + 0.005`。
* **信号 A（相对 A）**：存在 `M ∈ {Real, Null}`、`v ∈ V` 使
  `MAE_v(M) − MAE_v(A_s0) ≤ −0.003`，且 `MAE_full(M) − MAE_full(A_s0) ≤ +0.002`。
* **信号 B（Real vs Null）**：存在 `v ∈ V` 使
  `MAE_v(Real) − MAE_v(Null) ≤ −0.003`，且 `MAE_full(Real) − MAE_full(Null) ≤ +0.002`。
* **信号 C（预定义组）**：`MAE_{r2_unseen_ordinary}(Real) − MAE(Null) ≤ −0.010`，且
  `MAE_{seen_ordinary}(Real) − MAE(Null) ≤ +0.002`。
  `r2_unseen` = 该分子至少一个原子的 radius-2 结构 key 在 train 中 support 为 0
  （与 `zinc_performance_search_v1/analysis/per_molecule_valid.csv` 的
  `r2_unseen_atom_frac > 0` 列逐分子核对）；`ordinary = y > −3`；`seen_ordinary =
  ¬r2_unseen ∧ ordinary`。

**阶段二启动条件：条件 0 成立 且（信号 A 或 B 或 C）。** 否则停止，不跑 seed 1。

**阶段二（并行）：`Real_s1`（gpu0）、`Null_s1`（gpu1）。** 不新增结构、不调参、不加 seed 2。

---

## 5. 必报指标（主表 + 诊断）

主评价：**官方 valid 全量 raw-y MAE**（1,000 行，best epoch）。

每个 run 报：best valid MAE / best epoch、W20、T100、final valid、eval-mode train MAE、
wall time、参数量、初始 state hash、checkpoint sha256、GPU 型号/driver/torch。

配对比较（逐 seed）：
* `Real − Null`、`Real − A`、`Null − A`（full、drop_row172、drop_fixed_worst10）；
* paired bootstrap 10,000 次（只描述 1,000 行抽样，不替代跨 seed）；
* 单行贡献诊断：`max_i |ΔAE_i| / |Σ_i ΔAE_i|`（是否由单分子驱动）；
* 组：ordinary（y>−3）、tail（y≤−3）、r2_unseen、r2_unseen_ordinary、seen_ordinary、
  ring_rich（cycle_rank≥4）、row172、fixed worst-10；
* 固定 worst-10 集合上的平均 |AE|（不只看整体）。

机制诊断（冻结权重）：
* **C 切换**：best checkpoint 固定权重，Real→Null（和 Null→Real）后 valid 预测平均绝对变化、
  MAE 变化、逐分子符号；
* 新增列：第一层新 128 列的权重范数（init / step1 / best）、step-1 梯度范数、
  best epoch 时新列输入的 RMS 与经过新列后的贡献 RMS（与旧 529 列贡献对比）；
* h 的 participation ratio（init / best），用于确认表示未坍缩。

**不允许**把单一 row172 带来的整体改善解释为广泛结构泛化（row172 必单列）。

---

## 6. 判定规则（报告级，不再买新实验）

两个 seed 完成后按 mean 与方向描述（不做显著性宣称）：

| 观测 | 结论标签 | 含义 |
|---|---|---|
| mean(Real−Null) ≤ −0.001 且 mean(Real−A) ≤ −0.001，两 seed Real−Null 同号 | `REAL_PAIRING_SUPPORTED_PILOT` | 有初步证据支持真实第二跳键—终点配对；后续才考虑学习式投影 |
| mean(Real−A) ≤ −0.001 且 mean(Null−A) ≤ −0.001，但 mean(Real−Null) > −0.001 或两 seed 异号 | `SECOND_HOP_CONTENT_ONLY` | 第二跳 learned 内容可能重要，真实联合配对没有额外收益 |
| mean(Real−A) > −0.001 且 mean(Null−A) > −0.001 | `NO_PREDICTIVE_GAIN_THIS_CONFIG` | 当前固定投影与接入方式没有预测收益；只否定这个配置，不否定所有可学习投影 |
| 结论由单行驱动（`max_i|ΔAE_i| / |Σ_i ΔAE_i| ≥ 0.5`）或两 seed 方向冲突 | `INSUFFICIENT_EVIDENCE` | 证据不足，不自动追加结构 |

阶段一单独结束（无信号）时标签为 `STAGE1_NO_SIGNAL`，并记录在案。

---

## 7. 预算

目标 **1 GPU·h**（阶段一 + smoke），条件性扩展后硬上限 **3 GPU·h**。
计入 prep/smoke/timing/失败作业。不做超参数网格、不做第三阶段。

---

## 8. 复现

```bash
cd /home/calendar/code/research
# 0) 预注册（本文件 + configs/shc_v1.json）先提交
# 1) 第二跳补充表 + CPU 门禁（约 3–6 min CPU）
uv run python tracks/structural-sufficiency/zinc_secondhop_composition_pilot_v1/code/sh_cpu_checks.py
# 2) 聚焦测试
uv run pytest tracks/structural-sufficiency/zinc_secondhop_composition_pilot_v1/tests -q
# 3) 部署与 smoke（远程）
rr doctor res && rr deploy res
rr run res shsmoke-real-s0 --pool res-gpu0 --gpus 1 --time 0:20:00 \
  --result tracks/structural-sufficiency/zinc_secondhop_composition_pilot_v1/runs/shsmoke_s0 -- \
  python tracks/structural-sufficiency/zinc_secondhop_composition_pilot_v1/code/sh_train.py \
    train --arm Real --seed 0 --device cuda:0 --epochs 2
# 4) 阶段一
rr run res sh-real-s0  --pool res-gpu0 --gpus 1 --time 2:00:00 \
  --result tracks/structural-sufficiency/zinc_secondhop_composition_pilot_v1/runs/shReal_s0 -- \
  python .../code/sh_train.py train --arm Real --seed 0 --device cuda:0
rr run res sh-null-s0  --pool res-gpu1 --gpus 1 --time 2:00:00 \
  --result tracks/structural-sufficiency/zinc_secondhop_composition_pilot_v1/runs/shNull_s0 -- \
  python .../code/sh_train.py train --arm Null --seed 0 --device cuda:0
# 5) 拉回 + 分析
rr pull sh-real-s0 && rr pull sh-null-s0
uv run python .../code/sh_analysis.py --stage 1
```

---

## 9. 冻结信息与不做的声称

* 本轮不读 official test（任何阶段、任何理由）；`official_test_loaded=false`。
* 不声称 `C_real` 与 MAE 有因果关系以外的任何东西；不声称"ZINC 上路径绑定无用"
  （只对这一个固定投影 + 这一个接入方式 + 这两个 seed 下结论）。
* 不声称 `D`（审计的绑定残差）与 property 无关；本轮只测这一个投影下的可观测预测收益。
* 不修改 LCRC-v1 的任何权重/cache/报告数字；旧代码只读导入。
* 若 `Real ≈ Null` 但两者优于 A：下一步优化更简洁的边际统计，不急于 DFPB。
* 若 Real、Null 都不优于 A：停止该配置，不据此否定所有可学习投影。
