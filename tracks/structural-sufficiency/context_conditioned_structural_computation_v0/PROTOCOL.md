# PROTOCOL — CSC v0：上下文条件结构计算与结构功能发现

- Round：`tracks/structural-sufficiency/context_conditioned_structural_computation_v0/`
- 协议 id：`context-conditioned-structural-computation-v0`（简称 `csc-v0`）
- 本文件在任何正式训练开始之前写定；正式运行后只允许修实现（并记录到 REPORT 的工程日志段），不允许改公式、对照、评价口径或判定阈值。
- 机器可读冻结件：`configs/experiment.json`（队列、配方、宽度、身份 hash 的来源与生成方式）。
- Test 角色：`DEVELOPMENT_TEST`。每 epoch 全量评价并充分分析，但**不**进入任何选择；不作盲测声明。

## 1. 问题

**P1（性能）** 在保留 LCRC 有效计算的前提下，把结构内容与环境调制分离、并保留结构内容与真实连接关系对应性，能否提升下游图级预测？

**P2（可解释性）** 能否从模型实际执行的结构计算中提取环境条件下的结构作用，并归纳成可迁移到其他分子的简单结构规律？

唯一训练目标是原始图级标签（`ogbg-mollipo`、`ogbg-molesol`，官方 scaffold split，RMSE 主指标）。不做化学因果声明。

## 2. 数据

- `ogbg-mollipo` Train 3360 / Valid 420 / Test 420；`ogbg-molesol` 902 / 113 / 113。
- 数据管线只读复用：`tracks/structural-query-learning`（官方 split、图缓存、`lipo_v1` 特征）、`lcrc_cross_task_preparation_v1`（`feature_spec`、`lcrc_data` 缓存/collate/norm）、`lcrc_readout_context_pilot_small_v1/code/cr_data.py`（Bundle 组装、目标 scaler）。四个冻结缓存按真实字节 SHA-256 复核（`analysis/checks.json` 检查 1）。
- 归一化参数只用 Train：特征 `fit_norm(caches['train'])`，目标 `mean/std`（ddof=0）取自 Train 标签。
- 不删分子、不重新划分、不做数据增广。

## 3. 共同训练配方（冻结）

seed 0；batch 128；360 epochs；AdamW peak 3e-4 → min 1e-5 每步 cosine、**无 warmup**；wd 1e-3；betas (0.9,0.999)；eps 1e-8；grad clip 5；FP32；dropout 0.10（在 `h0` 之后，由继承的 `BSModel.drop` 施加）；标准化目标上的 MSE；单参数组；无辅助损失、无预训练、无蒸馏、无 EMA/SWA、无早停。每 epoch 完整 eval-mode 评价 Train/Valid/Test，指标恢复原始标签单位。best checkpoint 按最小 Valid RMSE（严格更小才替换，平局取更早 epoch）。同时保存 final checkpoint 与最后 20 epochs 的平均指标。所有臂在每个数据集上使用**同一**冻结 batch 顺序计划（`np.random.default_rng(0)` 每 epoch 一个置换，`plan_sha256` 记录在 `analysis/checks.json`）。

## 4. 共享基础架构

`lcrc_pcba_backbone_scaling_v1/code/bs_model.py::BSModel` 的 **1×** 宽度配置（`config_for_scale(1)`）：`width=64`、`atom_out=96`、`bond_out=96`、`stat_rank=32`、`local_hidden=192`、`rho_hidden=64`、`scorer_hidden=32`、`pool_hidden=88`、`n_heads=4`、`n_out=1`。原子/键编码器、局部内容（S1/S2）、`f`、`rho`、四头 attention pooling residual 一律不改；适配小任务只改数据接口与维度。`h0 = Dropout(LocalContent(G))`。

第一阶段（四个臂完全相同，`e_ji = bond_proj(bond_feat_ji)`，每个真实无向键展开为两个有向连接）：

```
z1 = LN1(h0);  m1_ji = SiLU(z1_j + e_ji);  M1_i = Σ_j m1_ji
h1_i = h0_i + fc2_1(SiLU(fc1_1(z1_i + M1_i)))
```

第二阶段公共计算（`W_in`/`b_in` 为 stage-2 `fc1` 的权重与 bias，`b_in` 只加一次）：

```
z_i = LN2(h1_i);  s_j = LN2(h0_j);  m_ji = SiLU(s_j + e_ji);  M_i = Σ_j m_ji
a_i = W_in z_i + b_in;  p_ji = W_in m_ji;  B_i = Σ_j p_ji
h2_i = h1_i + fc2_2(SiLU(a_i + B_i) + Q_i);   y_hat = BSModel.post_local(h2, batch)
```

## 5. 关系分组与环境调制（C/D）

真实关系分组由 `lipo_v1` 键类型给出：SINGLE / DOUBLE / TRIPLE / AROMATIC / OTHER（R=5，OTHER 为兜底；映射逐条与原始 `edge_class` 核对，见检查 1）。`P_i,r = Σ_{j: rel(ji)=r} p_ji`，`Σ_r P_i,r = B_i`。工程上先在 d 维聚合再乘无 bias 的 `W_in`。

```
c_i = concat(z_i, M_i / max(n_i,1), log1p(n_i))          # 2d+1 = 129
δ_i,r = 0.5 * tanh( Gate(c_i) ),  Gate = Linear(129,16) → SiLU → Linear(16, R*q)   # 末层零初始化
Q_i = Σ_r δ_i,r ⊙ P_i,r
```

`δ=0` 时 C/D 在位级退回 A（检查 7）。C 把分组内容替换为 `P_null_i,r = n_i,r/max(n_i,1) × B_i`（保持计数、总和、环境、参数量、初始化），只去掉“具体结构内容—关系分组”的对应性。

## 6. 四个臂

| 臂 | 第二阶段读取 | 修正项 |
|---|---|---|
| A `FIXED` | `LN2(h0_j)` on real in-edges | 无 Gate，`Q=0` |
| B `ONE_READ` | `LN2(h0_i)` on real in-edges（同键属性与重数） | 无 Gate，`Q=0` |
| C `MIXED` | 同 A | Gate + 混合分组 `P_null` |
| D `CSC` | 同 A | Gate + 真实分组 `P` |

参数量：A/B = 294,135；C/D = 307,095（新增模块 A/B 34,304，C/D 47,264；基础 LCRC 259,831）。A/B/C/D 的共享部分初始化逐位一致（检查 4），C/D 全参数初始化逐位一致（检查 5）。

## 7. 队列与预算

Lipo：A / D → C / B；ESOL：A / C / D。共 7 个新训练单元（seed 0）。开工前核查是否有完全匹配历史资产：无（历史 C1/GINE 的读出与训练入口都不同，只作描述性参照）。

预算：GPU 总消耗目标 ≤ 2 GPU·h，硬上限 3 GPU·h；本轮整体墙钟硬上限 2 h。降级顺序：Lipo A/C/D 完整 → Lipo B → ESOL A/C/D 完整。不为预算富余增加 seed/宽度/Gate 容量。正式训练前用真实 batch 做吞吐预检（`analysis/preflight_timing.json`），只按计时决定调度，不看精度。

## 8. 正确性检查（正式训练前必须全通过）

官方 split 与缓存身份；原子重编号/键顺序/端点交换不变性；单图 vs batch 一致；四臂共享初始化一致；C/D 全参数初始化一致；真实与混合分组都能重构 `B_i`；`δ=0` 精确退回 A；第一阶段四臂一致；空边/单原子/单一关系邻域可处理；Gate 非零时异质邻域上 C/D 必须不同、同质邻域上必须相同；Gate 梯度可达（首步梯度为 0 是零初始化残差的已知后果，随后必须非零）。结果写入 `analysis/checks.json`。

## 9. 性能分析口径

主表：数据集、臂、参数量、best epoch、Train/Valid/Test RMSE+MAE（best）、final、last-20 均值、训练/评价时间、GPU 消耗、推理吞吐。

比较：A−B、C−A、D−C、D−A；每项用相同分子的配对比较，报告 RMSE/MAE 差值、逐分子改善/退化比例、2000 次 paired bootstrap 区间、最大误差影响样本敏感性、Train–Valid gap、best 与 late-horizon 方向是否一致；若可用则给 scaffold-cluster bootstrap。D 与同官方划分的 C1（`method_L`）/GINE（`method_G`）作**描述性**绝对竞争力参照，不冒充严格对照。

附加固定分层：每图“具有至少两类真实入边类型的节点比例”，阈值为 Train 中位数，报告 Valid/Test 中 D−C 的分层表现（不替代官方主指标）。

## 10. 机制检查

对 C/D best checkpoint 记录 Gate 输出均值/RMS/标准差、分关系调制差异、`Q_i` RMS、`Q_i` 经 `fc2` 投影后的状态变化、异质关系节点与分子覆盖率、`P_i,r` 与 `P_null_i,r` 实际差异比例。对 D 同一 checkpoint 做三项冻结推理：I0 `Q=0`；I1 真实分组→混合分组（其余不变）；I2 动态 δ → Train 关系出现次数加权的固定 `δ̄_r`。每项报告 Valid/Test RMSE、MAE 以及预测变化的 RMS/MAE。严格区分“模型是否使用该机制”与“独立训练是否因此泛化更好”。

## 11. 精确计算分解

Lipo 的 A 与 D。`γ_i = [SiLU(a_i+B_i) − SiLU(a_i)] / B_i` 逐通道（`B_i→0` 用 `SiLU'(a_i)`，除法在 float64 中做）。`v_ji = W_out[(γ_i+δ_i,r) ⊙ p_ji]`，`W_out` 为 stage-2 `fc2` 权重（无 bias）。`h2_base_i = h1_i + fc2_2(SiLU(a_i))`，`h2_i = h2_base_i + Σ_j v_ji`；新增修正的独立作用向量 `v_extra_ji = W_out(δ_i,r ⊙ p_ji)`。沿 `h2(t) = h2_base + tΔh` 用完整 `BSModel.post_local` 计算

```
A_ji = ∫_0^1 ∇_{h2_i} F(h2(t)) · v_ji dt        （F 为标准化预测，报告前乘 Train 目标 std）
```

16 点 Gauss–Legendre；闭合误差超容差（`1e-4 + 1e-3|Δ|`）再用 32 点重算；不做事后归一化。参照状态是模型内部计算参照，不是化学反事实。

## 12. 规律发现

每条真实有向消息记录分子 row id/split、src/dst、真实键类型与属性、来源与接收端的局部结构记录（精确结构签名，含 S1/S2 类型与重数）、`p_ji` 摘要、`γ_i`、`δ_i,r`、`v_ji`/`v_extra_ji`、标量任务作用 `A_ji`。只保留标量与少量摘要向量。

(1) 同一（来源结构签名 + 关系类型）在 Train 中的重复：报告类型数、跨分子/scaffold 出现次数、作用中位数/IQR、符号变化比例、Valid/Test 覆盖率；结构身份定义不放宽。(2) 轻量固定复杂度解释树（`max_depth=4`、`min_samples_leaf=100` 条消息、`random_state=0`、按分子等权赋权）分别用 S（来源结构）/C（环境）/SC（并集）拟合 `A_ji`，只用 Train 拟合、Valid/Test 评价；报告分子等权归因 MAE 与相对常数预测器的 skill。禁止把隐向量作为树特征。(3) 只在 Train 上从树与重复结构记录里选最多 5 条候选规律，在 Valid/Test 报告支持与方向；至少展示 3 个真实分子（成功案例/环境改变作用案例/反例）。允许规律发现失败。

## 13. 判定

性能轴（Lipo 为主）：以 2% 相对 RMSE 改善为初步实用性参考（非显著性阈值），看 D−A、D−C、Valid/Test 是否同向、late-horizon 是否反转、是否集中于极少数异常样本。可能结论：`RELATION_CONDITIONED_GAIN` / `GENERIC_CONDITIONING_GAIN` / `FIXED_SOURCE_ONLY` / `NO_MATERIAL_GAIN`。

可解释性轴分三维分别报告：`TRACE`（账目闭合、真实消息可追溯到预测变化）、`RULE_REUSE`（简单结构+环境规则在未参与拟合的 Valid/Test 上能否解释真实计算作用，正向参考：SC 相对 S/C 较强者归因 MAE 改善约 ≥10%，同时作用幅度与支持充分）、`STRUCTURE_DISCOVERY`（是否出现具体的、非平凡的、跨分子支持的功能结构候选）。不得互相替代。

联合归属（五选一）：性能改善+可归纳结构作用；性能改善但无可解释性额外证据；性能基本保持+更好可归纳结构作用；两者均无实质改善；证据不足或冲突。只有解释性提升而性能明显下降不称为联合突破。

## 14. 交付物

`PROTOCOL.md`、`configs/`、`code/`、`analysis/{checks,main_table,paired_comparisons,mechanism_diagnostics,structure_rules,explanation_cases}.json|csv`（另含 `preflight_timing.json`、`decomposition_trace.json`、`strata.json`、`historical_reference.json`）、`results/`（曲线、预测、checkpoint manifest + SHA256；大体积 checkpoint 按仓库规范留在 GPU 机器并记录路径与 SHA256）、`REPORT.md`（中文）。
