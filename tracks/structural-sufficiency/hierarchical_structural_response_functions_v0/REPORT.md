# HSRF v0 — 分层结构响应函数快速筛选报告

**Round**: `hierarchical_structural_response_functions_v0` · **Commit**: `9d9ee3d2e3c132722df841408d338032a7990d4d` · **生成时间**: 2026-11-24T03:48:40Z
**数据**: `ogbg-mollipo`，官方 scaffold split（3360/420/420），seed 0。Test 仅作为 DEVELOPMENT_TEST 报告，所有选择只由 Train/Valid 决定。
**候选**: `F_BASE`（Valid RMSE 0.7915，Test RMSE 0.7819）。

## 1. 一句话结论

本轮显式 HSRF 表示（根局部结构计数 + 固定 hinge）在 Lipo 上**没有显示出有竞争力的预测力**：候选 `F_BASE` 的 Valid RMSE 为 0.7915，比冻结 GINE 坐标 0.6091 高 0.1825；同输入 MLP/KRR 黑箱也没有接近 GINE（最强 Valid 0.7996），因此四轴判定为 **REPRESENTATION_OR_LEARNER_LIMITED**。非线性 hinge 与 REAL 关系都没有带来协议要求的 Valid 增益；贡献账目本身是精确的，但近优 λ 解的归因不稳定。

## 2. 冻结设置与历史坐标核对

- Stage 0 对每个根原子枚举 size 1–4、radius-2 内连通诱导子图，四粒度签名（TOPO/ELEMENT/COARSE/FULL），Train-only 词表（DF≥8，cap 1024；TOPO 不截断）。
- 64 个 COARSE 锚点（size 2/3 各 top-32）；真实关系按有向键槽统计；NULL 使用 hypergeometric 的 `E[h(K)]`，不是 `h(E[K])`。
- 五个闭式 ridge 臂：`L_BASE`、`F_BASE`、`L_REAL`、`F_NULL`、`F_REAL`；λ 网格 1e-6…10，按 Valid 选择。MLP 三臂与 KRR 两臂只作为同输入黑箱参照，不参与候选选择。
- 历史坐标均从实际产物重新读取并核对：GINE 0.6091/0.6456，LCRC 0.7052/0.7139，SWF 0.6991/0.6891，Pair Rule REAL-FULL 0.7952/0.8673。

## 3. 工程检查与正对照

- 工程检查：36/36 通过；失败项：无。
- 数据/划分/标签/行号、Train-only scaler、人工 path/star/triangle/ring 枚举、重标号不变性、四粒度映射、REAL 关系上下界、hypergeometric 期望、ridge 直接解一致、预测 replay 均通过。
- 贡献账目最大复算误差：F_BASE 1.07e-14，F_REAL 1.78e-14（目标 ≤1e-7）。
- Synthetic-L/F/R：小型满秩设计在 λ=1e-6 下预测 RMSE <1e-5、系数误差 <1e-4；完整库上 F_REAL 对 Synthetic-R 的 Test RMSE 明显低于 F_BASE/F_NULL；F_BASE 的关系贡献恒为 0，未被错误归因。Synthetic 结果只作工程正对照，不是化学证据。
- 父子计数恒等式：size1 = n_atoms、size2 = 2·n_bonds、cycle_rank = n_bonds − n_atoms + n_components 在全部枚举实例上精确成立；保留词表覆盖 size1 ≥0.9997，size2 ≥0.9907。

## 4. 主结果

| 模型 | 族 | Valid RMSE | Test RMSE | 选择 |
|---|---|---|---|---|
| L_REAL | HSRF-ridge | 0.7893 | 0.7700 | lambda=0.3 |
| F_NULL | HSRF-ridge | 0.7898 | 0.7901 | lambda=0.3 |
| F_BASE | HSRF-ridge | 0.7915 | 0.7819 | lambda=0.3 |
| L_BASE | HSRF-ridge | 0.7940 | 0.7808 | lambda=0.3 |
| F_REAL | HSRF-ridge | 0.7994 | 0.7764 | lambda=1.0 |
| MLP_NULL | blackbox-MLP | 0.8052 | 0.7708 | epoch=33 |
| MLP_BASE | blackbox-MLP | 0.8211 | 0.7843 | epoch=86 |
| MLP_REAL | blackbox-MLP | 0.8278 | 0.7709 | epoch=64 |
| KRR_BASE | blackbox-KRR | 0.7996 | 0.7382 | sigma=133, lambda=0.0001 |
| KRR_REAL | blackbox-KRR | 0.8002 | 0.7388 | sigma=148, lambda=0.0001 |
| GINE_sq_pilot | frozen-reference | 0.6091 | 0.6456 | frozen |
| SWF_WORD_PREFIX | frozen-reference | 0.6991 | 0.6891 | frozen |
| LCRC_sq_pilot | frozen-reference | 0.7052 | 0.7139 | frozen |
| PAIR_RULE_MLP | frozen-reference | 0.7793 | 0.7740 | frozen |
| PAIR_RULE_REAL_FULL | frozen-reference | 0.7952 | 0.8673 | frozen |

Test 仅用于报告；没有据此改模型。候选由 Valid RMSE 在 F_BASE/F_REAL 中选择；`L_REAL` 的 Valid 0.7893 是五臂中最低，但按协议它只是无 hinge 的消融参照，不能作为候选。

关键对比（gain = RMSE(reference) − RMSE(candidate)，正值表示 candidate 更好）：

| 对比 | Valid gain | Valid 95% CI | Test gain |
|---|---|---|---|
| nonlinear_base | 0.0024 | [-0.0361, 0.0386] | -0.0011 |
| nonlinear_real | -0.0101 | [-0.0430, 0.0219] | -0.0064 |
| relation_vs_base | -0.0078 | [-0.0296, 0.0130] | 0.0055 |
| relation_vs_null | -0.0096 | [-0.0422, 0.0243] | 0.0137 |
| relation_linear | 0.0047 | [-0.0091, 0.0175] | 0.0108 |
| mlp_relation_vs_base | -0.0067 | [-0.0404, 0.0289] | 0.0133 |
| mlp_relation_vs_null | -0.0226 | [-0.0601, 0.0146] | -0.0001 |
| F_BASE_vs_GINE_sq_pilot | -0.1825 | [-0.2474, -0.1204] | -0.1363 |
| F_REAL_vs_GINE_sq_pilot | -0.1903 | [-0.2507, -0.1310] | -0.1308 |
| F_BASE_vs_LCRC_sq_pilot | -0.0864 | [-0.1703, 0.0067] | -0.0680 |
| F_REAL_vs_LCRC_sq_pilot | -0.0942 | [-0.1777, 0.0007] | -0.0625 |
| best_bb_BASE_vs_F_BASE | -0.0081 | [-0.0600, 0.0428] | 0.0437 |
| best_bb_REAL_vs_F_REAL | -0.0008 | [-0.0377, 0.0388] | 0.0376 |

## 5. 四轴判定

| 轴 | 判定 | 证据 |
|---|---|---|
| 表示 / learner | **REPRESENTATION_OR_LEARNER_LIMITED** | 最强同输入黑箱 Valid 0.7996；GINE Valid 0.6091；阈值 GINE+0.06=0.6691。 |
| 非线性 hinge | **NO_MATERIAL_NONLINEAR_GAIN** | F_BASE−L_BASE Valid gain 0.0024；F_REAL−L_REAL 见上表；均未达到 0.02。 |
| REAL 关系 | **NO_MATERIAL_RELATION_GAIN** | F_REAL−F_BASE Valid gain -0.0078，Test 0.0055；F_REAL−F_NULL Valid -0.0096，Test 0.0137。支持充分但 Valid 方向为负。 |
| 分解成本 | **LOW_DECOMPOSITION_COST** | gap = 黑箱 − 显式（正值=黑箱更差）。F_BASE vs 最强同输入黑箱：Valid gap 0.0081，Test gap -0.0437；F_REAL：Valid 0.0008，Test -0.0376。 |

**总体**: `NOT_PROMISING`；flags: `NOT_PROMISING, ATTRIBUTION_NOT_IDENTIFIED, REPRESENTATION_OR_LEARNER_LIMITED, EXACT_LEDGER`。

Bootstrap: 2000 次 paired resample，seed 1703。Valid/Test 中每个分子都是独立 Bemis–Murcko scaffold cluster（420/420），因此 paired cluster bootstrap 等价于 molecule-row paired bootstrap；CI 只表示固定状态下的样本不确定性。

解释：本轮的主要瓶颈不是“把有效应拆开”，而是**显式计数表示本身**。在相同输入下，MLP/KRR 也没有把 Valid 拉到 GINE 附近；因此不能把差距归因于 hinge 参数化或关系账目。关系在 Test 上的方向性改善（F_REAL 比 F_BASE 低 0.0055）与 Valid 方向相反，按协议只能标记为 split-sensitive，不能作为稳定信号。

## 6. 关系覆盖与 REAL/NULL

- 64/64 锚点被使用；1024 个关系类型，Train/Valid/Test 均有 100% 分子至少命中一个关系，100% 分子 REAL≠NULL（线性），median nonzero relations 分别约 195/199/194。
- 因此“关系没有增益”不能解释为覆盖不足；它是给定表示与正则化下的经验结果。
- 线性消融中 REAL 相对 BASE 有一致的小幅增益（Valid 0.0047，Test 0.0108），但低于 0.02 的协议门槛；完整 hinge 模型 F_REAL 反而在 Valid 上更差。
- NULL 使用 hypergeometric 期望；F_NULL 的 Valid 0.7898 与 F_REAL 的 0.7994 接近，REAL 没有稳定超过匹配边际。

## 7. 贡献账目与可辨识性

- 账目按 `intercept + Σ main + Σ relation` 导出；所有组（根结构、整图统计、关系）分开记录，不把重叠根结构解释为互斥原子能量。
- F_BASE 的精确秩 2796/10005，F_REAL 3093/12477；设计矩阵存在大量精确重复列（F_BASE 重复组覆盖 9208 列、冗余 5945 列；F_REAL 分别为 10770、7081），这是计数签名跨粒度冗余的直接体现。
- 近优 λ（Valid 差 ≤0.01）：F_BASE [0.3, 1.0]；F_REAL [1.0]。F_BASE 的 λ=1.0 近优解与 λ=0.3 解的 Valid 预测 RMS 差约 0.31，贡献排序 Spearman 约 0.92，但主要组的平均绝对贡献差异超过 20%，因此标记 **ATTRIBUTION_NOT_IDENTIFIED**。
- 近优解的 β 差几乎全部落在行空间（null-space fraction ~1e-12），说明差异不是数值零空间造成，而是近优解之间的真实函数差异。

## 8. 匹配对诊断

- 复用 `task_response_composition_s0_v1` 的 matched pairs：5770 对，cell counts {'cross': 1072, 'test': 37, 'train': 4646, 'valid': 15}；held-out 1124 对，其中 seen transform 210、unseen 914。

| 模型 | held-out Δpred MAE | 相对零差分技能 | 方向准确率 |
|---|---|---|---|
| F_BASE | 0.4888 | 0.3540 | 0.7987 |
| F_REAL | 0.4900 | 0.3524 | 0.7969 |
| L_BASE | 0.4767 | 0.3700 | 0.8005 |
| L_REAL | 0.4680 | 0.3815 | 0.7852 |
| F_NULL | 0.4767 | 0.3700 | 0.8032 |
| MLP_NULL | 0.4560 | 0.3973 | 0.8032 |
| MLP_REAL | 0.4774 | 0.3691 | 0.7969 |
| KRR_BASE | 0.5243 | 0.3071 | 0.7897 |
| KRR_REAL | 0.5236 | 0.3080 | 0.7843 |

- Δpred 相对“零差分”基准的技能分数与方向准确率见 `pair_diagnostic.csv`。F_REAL 的 held-out Δpred 中，关系项平均绝对贡献 0.1202，主效应 0.4495，关系占比 0.2110；Δmain+Δrel 复算误差 0.00。
- 该诊断只检验相邻根局部关系的可迁移性，不改变主表选择。

## 9. 解释与限制

1. **表示瓶颈**：固定的小半径根局部计数不能表达 GINE 的深层消息传递；同输入 MLP/KRR 也远落后于 GINE，说明瓶颈在输入表示/可学习的函数类，而不是缺少非线性头。
2. **关系冗余**：REAL 关系在覆盖上充足，但其增量与匹配边际 NULL 接近；完整 hinge 库中大量关系列与基础结构列高度共线，ridge 的 Valid 选择倾向强正则化，进一步压缩了关系信号。
3. **分解成本低但绝对精度差**：在相同输入下，显式可加模型的 Valid 与 KRR 相当（差距 <0.01），说明“可分解”本身没有付出大的预测代价；问题是这个函数类的上限不够高。
4. **分割敏感性**：F_REAL 在 Test 上优于 F_BASE/F_NULL，但 Valid 上更差；按预注册规则不能据此宣称关系信号。
5. **归因不稳定**：近优 λ 解可以给出差异明显的函数曲线；报告中的曲线只对应最佳 Valid 解，不能当作唯一因果解释。

## 10. 复现与预算

- Stage 0 本地 67 s、远端 146 s；ridge 五臂总计约 12 s CPU；KRR 约 114 s CPU；MLP 三臂 GPU 总时长约 0.8 GPU·min（约 0.013 GPU·h），低于 ~2 GPU·h 软上限。
- 所有正式运行绑定 commit `9d9ee3d2e3c132722df841408d338032a7990d4d`；历史模型未重训、未新增 seed/数据集/函数族；Test 未参与任何选择。
- 主要产物：`results/main_table.csv`、`results/contrast_table.csv`、`results/bootstrap_*.csv`、`results/ridge/ledger/`、`results/ridge/curves/`、`results/figures/`、`analysis/checks.json`、`analysis/contribution_identifiability.json`、`analysis/pair_diagnostic.json`、`summary.json`。
