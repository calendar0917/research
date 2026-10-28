# PROTOCOL — LCRC-v1（Local Contribution + Relation-Aware Composition）

**Round id:** `zinc-local-relation-composition-v1`
**Track:** `tracks/structural-sufficiency/zinc_local_relation_composition_v1/`
**状态:** 训练前冻结（CPU gate 8/8 PASS；GPU 未启动）
**数据:** official PyG ZINC `subset=True`，train 10,000 / valid 1,000 / test 1,000。
**指标:** raw-y MAE（单模型）。
**参数上限:** 500k 部署参数。A = 237,519；R = N = 269,664。
**计算预算:** 目标 6 GPU·h，硬上限 8 GPU·h；最多同时两张 `res` A100。

---

## 1. 问题

不用 learned junction token，能否用

> 共享多尺度局部内容 `h_v` + 原子级标量贡献之和 + 小型全局修正（A）
> 以及在同一 `h_v` 上的一次性真实环境间关系组合（R）

在 ZINC-12k 上达到或超过 DCC-L 的竞争区间（valid 4-seed 均值 0.075990；
历史 test 4-seed 均值 0.067319）？

三个 arm：

| arm | forward |
|---|---|
| **A** | `Σ_v f(h_v) + ρ(global33, mean h, mean h², log(1+|V|))` |
| **R** | A + `Δ_rel`，其中每个关系类别的内容均值来自该类别**真实**的原子对 `z_ij` |
| **N** | 与 R 完全同结构/同参数/同初始化/同数据顺序；唯一区别是每个非空类别的内容均值替换为该分子**全部合格原子对**的 `z` 均值，保留真实 `n_r` 与 existence flag |

R−A 含容量差异；R−N 隔离“真实归属”；N−A 检验“额外 pair 内容 + 容量”是否已足够解释收益。

## 2. 与旧实验的实际 forward 差别

| 旧轮 | 实际 forward | 本轮区别 |
|---|---|---|
| R0（rooted statistics，valid≈0.140） | 每原子 radius-1/2/3 exact joint count + 3 个 scale 的 MLP + 大 reader | 本轮 `h_v` 用共享低秩 typed-path 因子（含乘法交互），且读出是**原子级标量和 + 小 ρ**，不是宽 reader |
| T（association tensors，0.0914） | P3 encoders + rank-16 atom-bond / bond-junction 外积张量 + 大 reader | 无 junction object；无外积关联张量；局部读出是加性的；关系只在远距原子对上 |
| DCC-L（0.0760） | learned junction encoder → 96-d junction 状态 → global/local composer → 513-d reader | **没有 junction encoder、没有 junction 状态、没有 per-wedge token**；局部只做一次 atom/bond + rooted S1/S2 聚合；读出是 `Σ f(h_v)` |
| AR（additive reader，Locked 0.0845） | DCC-L trunk 上把 reader 换成加性贡献 + 小 ψ | 本轮连 DCC-L trunk 也没有；局部内容来自 atom/bond + 显式 rooted 统计，且关系模块是新的 R/N 对照 |
| BOSI partner-content | 目标半键的伙伴 learned 状态进入矩块 | 本轮无 bond/junction token；S1 是邻接 (type,bond) 联合码的共享 embedding，S2 是 typed 2-path 的乘法因子组合 |
| RCC（关系二阶通道 99% 冗余，STOP） | bond state 上的固定二次型 `T=Zᵀ(K_R−K_D)Z` | 本轮是**原子对内容** `z_ij=φ(h_i+h_j,|h_i−h_j|,h_i⊙h_j)` 的**类别均值**，且用同参数 N 对照直接测“归属”；但仍报告 RCC 式线性冗余风险（见 §5） |

结论：计算图不是旧实现的改名/重参数化。最大风险不是等价，而是**关系统计与 A 的池化特征冗余**——已用 CPU ridge probe 量化（common categories R²≈0.70，即约 30% 方差不能由 A 的池化特征线性重建；h 的 participation ratio 在初始化时仅 1.19，属于已记录的退化风险）。

## 3. 数据与选择规则

- 只用官方 split；不创建 Fit/Monitor/Locked。
- train 10k 训练 + 拟合标准化；valid 1k 每 epoch 全量评估、选 checkpoint；test 只在最终冻结后一次性终端评估。
- 不使用 EMA、soup、seed averaging、ensemble、蒸馏、辅助标签、目标分项监督。
- 不使用 RDKit logP/SA、目标环惩罚公式值、历史模型预测/残差作为输入。
- 输入只含原始图的 label-free 统计 + 仓库已有 `global33` + 已有 train-only masked-SSL 的 atom/bond encoder 权重（正常微调；不加载任何监督 DCC-L 权重）。
- 标准化只由 train 拟合：global33 复用冻结 P3 train stats；9 个原子标量用 train mean/std。

## 4. 训练配方（与历史 R0 完全一致）

AdamW，peak LR 1e-3、min LR 1e-5；500 update warmup；之后 per-step cosine；
weight decay 1e-5；batch 128；600 epochs = 79 steps/epoch × 600 = 47,400 updates；
raw-y L1；clip 5；dropout 0；每 epoch 全量 valid；严格最小 valid MAE 选一个 checkpoint（平局取更早 epoch）。
数据顺序：`torch.randperm(10000, Generator().manual_seed(seed+101))`，每 epoch 重新打乱；最后一个不满 batch 保留。
A/R/N 同一 seed 使用完全相同的初始化（共享参数逐 tensor 相同；R/N 全部参数逐 tensor 相同）和数据顺序。

## 5. 训练前 CPU gate（`analysis/cpu_checks.json`，8/8 PASS）

| gate | 关键结果 |
|---|---|
| G1 参数/模块/SSL/no-junction | A 237,519；R/N 269,664；SSL 只加载 atom/bond encoder（sha `006fabc1…`）；无任何 junction module/state key；R/N 初始 full-state hash 相同；R/N 与 A 的 step-0 预测 max abs diff = **0.0** |
| G2 不变性 | 原子重编号 0.0；键方向反转 ≤1.7e-7；batch split ≤1.2e-7；batch order 0.0（A/R/N） |
| G3 统计语义/边界 | 200 个真实分子逐原子核对 S1/S2 与 pair 类别，mismatch 全 0；单原子、链、6 环、双连通分量边界均有限；6 环对位原子正确落入 `d3_ring`；不可达对映射到 ≥6；同步态 189 路表的 rank-1 能量 0.955（说明边际占主导，但乘法 joint 项能把相同边际、不同共现的路径区分开，max abs 1.80） |
| G4 R/N 初始化与数据顺序 | full-state hash 相同；600 epoch 的 permutation stream 逐 epoch 相同 |
| G5 合成归属 | 相同 z 多重集、相同类别计数（2,2,2），不同归属：R 内容 max abs diff 0.341、Δ_rel diff 0.00885；N 内容 diff = 0.0；pair 顺序不变 |
| G6 真实关系覆盖/冗余 | 512 train 分子：平均 207.5 合格对/分子（45–532）；类别非空率 [1.00, 0.945, 1.00, 0.324, 1.00, 0.021]；100% 分子有 ≥2 个非空类别；R vs N 内容 100% 不同（mean L2 0.454）；ridge 从 A 池化特征重建 R 内容，common categories (0,2,4) R²≈0.702，permutation control ≈−1.44 |
| G7 梯度/短训/checkpoint | 30 步 CPU loss 下降；全部梯度有限；R 的 head 第一步有非零梯度，φ 第二步非零；checkpoint reload max abs = 0.0 |
| G8 test 隔离 | prep 只接受 train/valid，test prep 被拒绝；test cache 不存在；所有 run 记录 `official_test_loaded=false` |

冗余风险已量化：common categories 的 R 内容约 70% 可由 A 已有池化特征线性重建，约 30% 不可。初始化时 h 的 participation ratio 1.19，说明局部表示的维度偏低；这会在训练后重新测量，不把它当作已排除的风险。

## 6. 阶段与停止规则

**阶段 1（并行，`res-gpu0`/`res-gpu1`）：** `A_s0`、`R_s0`。

- STOP：`A_s0 > 0.085` 且 `R_s0 > 0.085` 且 `R_s0 − A_s0 > −0.004`。
- CONTINUE：`R_s0 ≤ 0.085` 或 `R_s0 − A_s0 ≤ −0.004`。

**阶段 2（并行）：** `N_s0`、`R_s1`。N_s0 必须与 R_s0 同 seed、同初始化、同数据顺序。

**阶段 3 门禁：** 只有 `R_s0 − N_s0 ≤ −0.001`（R 更好至少 0.001）且 `R_s1 ≤ 0.088`（没有明显崩溃），才并行跑 `N_s1`、`A_s1`。

**提前停止：** 任一阶段门禁不满足即停止；保存已完成事实，不追加宽度/关系类别/LR/正则化网格。

## 7. 分析与 test 规则

- 主表至少包含：各 arm 实际完成 seed 的 best-valid MAE、参数量、训练步数、best epoch、末段 valid 曲线（W20/T100/final）、train eval-mode MAE。
- 同 seed 逐分子配对：R−N、R−A、N−A；paired bootstrap 10,000 次只描述 valid 分子抽样，不替代跨 seed 复现。
- 分组：ordinary (`y>-3`)、tail (`y≤-3`，环惩罚相关)、row172、radius-2 unseen（251）、unseen ordinary（225）、ring-rich (`cycle_rank≥4`)、固定 DCC-L/P3B worst-10、去掉 row172、去掉 worst-10。
- 机制：R/N 真实关系输入的类别计数/内容差、非空覆盖；冻结权重把 R/N 的 `rel.head` 置零后预测变化；A 的 `|Σf|` 与 `|ρ|` 尺度；best checkpoint 上 h 的 participation ratio。
- test 只在以下条件同时满足时读一次：R 在两个 seed 上相对 N 有一致收益、mean `R−N ≤ −0.001`，且 R 的 valid mean ≤0.079（DCC-L 4-seed 均值 +0.003，属于“接近”），或新主体明确超过 DCC-L。只对预选的一个单模型 checkpoint 做一次终端评估。
- test 已 spent，本轮 test 结果只能是描述性性能记录，不是新盲测；不得据此改模型/seed/epoch。

## 8. 预算

目标 6 GPU·h、硬上限 8 GPU·h（含 prep/smoke/timing/失败作业）。正式轨迹预计 ≤0.5 GPU·h/run；6 条正式轨迹 + smoke 预计 3–4 GPU·h。

## 9. 禁止

- 不改写旧实验的结果/checkpoint；不加载 DCC-L 监督权重。
- 不增加 junction token、per-wedge token、迭代 hidden-state MP、attention、EMA/soup。
- 不根据 test 调参；不把 R>A 直接解释为关系机制成功。
- 不在门禁失败后自动开启新一轮架构搜索。
