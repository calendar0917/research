# 单模型 LCRC 的支撑集联合结构函数 v0 — REPORT

- Round: `tracks/structural-sufficiency/lcrc_support_joint_function_v0/`（round id `lcrc-support-joint-function-v0`）
- 协议：`PROTOCOL.md`（在任何正式训练与任何 Valid/Test 比较之前写成并提交）
- 冻结配置：`configs/experiment.json`（sha256 `caf0dea7db8ec1ba8d83b9c8973d726ec4a5d168420ac1c0b2a25cb11738918d`）
- 训练代码 commit：`6352d528`；正确性检查 commit：`0d8079d4`（两者之间只有 `analysis/` 产物差异）
- 执行 regime：`res` / `a100-2` / process 后端 / pool `res-gpu1` / torch 2.5.1+cu124 / CUDA 12.4 / driver 550.163.01，单卡顺序执行（GPU0 全程被其他任务占用，未并行、未抢卡）
- 实际 GPU 消耗：**0.558 GPU·h**（预算目标 2，硬上限 3）；墙钟 71 min（上限 2 h）
- Test 角色：`DEVELOPMENT_TEST`（仅报告，不参与任何选择；A 臂沿用它原轮次的 `EXPOSED_TEST_EXPLORATORY` 标签）

---

## 1. 结论（第一页）

### 1.1 主性能（best-Valid checkpoint）

| task | 臂 | 参数 | joint 模块参数 | 有 Train 支持的 joint 参数 | best epoch | Valid 主 | Test 主 | Valid 辅 | Test 辅 |
|---|---|---|---|---|---|---|---|---|---|
| hiv | A 原始 | 247,106 | 16,416 | 4,064 | 53 | **0.831441** | **0.765314** | 0.313169 | 0.246938 |
| hiv | **B 联合表** | 239,970 | 9,280 | 9,280 | 43 | **0.838061** | **0.781388** | 0.357522 | 0.271563 |
| hiv | C 仅边际 | 233,506 | 2,816 | 2,816 | 13 | 0.816139 | 0.745225 | 0.302726 | 0.272072 |
| lipo | A 原始 | 228,674 | 16,416 | 2,784 | 132 | **0.716493** | **0.586386** | 0.428017 | 0.440167 |
| lipo | **B 联合表** | 215,906 | 3,648 | 3,648 | 189 | **0.711393** | **0.566599** | 0.425313 | 0.426921 |
| lipo | C 仅边际 | 213,858 | 1,600 | 1,600 | 229 | **0.674787** | 0.568534 | 0.410430 | 0.424321 |

hiv 为 ROC-AUC / AP（越大越好），lipo 为原始单位 RMSE / MAE（越小越好）。A 为完全匹配的历史基准，未重训。

### 1.2 P1–P5

| 问题 | 回答 | 依据 |
|---|---|---|
| **P1：B 是否在 Valid 和 Test 都优于 A？** | **两个任务都是「是」** | hiv +0.00662 Valid / +0.01607 Test ROC-AUC；lipo −0.00510 Valid / −0.01979 Test RMSE（相对 −0.71% / −3.37%，负=更好）。Valid 上的 hiv 增益刚过 +0.005 的路线参考；lipo Valid 相对改善 0.71% 低于 3% 参考，Test 3.37% 高于参考 |
| **P2：B 是否优于 C？** | **hiv 是，lipo 否（Valid 明显输，Test 打平）** | hiv B−C = +0.02192 Valid / +0.03616 Test ROC-AUC；lipo B−C = **+0.03661 Valid**（B 更差 5.4%）/ −0.00194 Test（B 好 0.34%，区间跨 0） |
| **P3：B 的优势是否伴随实际不可加函数的使用？** | **是（两任务）** | B 的冻结函数干预：ADD_ONLY 在 hiv 使 Valid −0.01343 / Test −0.02150 ROC-AUC，在 lipo 使 Valid +0.16065 / Test +0.17976 RMSE；HALF_INTERACTION 分别 −0.00271/−0.00483 与 +0.03708/+0.04066。分解显示 B 的非加性分量 RMS 是加性分量的 1.48×（hiv）/1.15×（lipo），与原始 A（1.59×/1.18×）同一量级 |
| **P4：能否提取有结构意义、支撑充分、可核验的计算差分？** | **是（两任务各 5 个候选）** | Omega 中真实存在的四边矩形 hiv 4,658 / lipo 488，其中四对各自 Train 独立分子支持 ≥20 的 hiv 1,726 / lipo 170。最优候选 H(IV)#1：把 `root→C` 键由 SINGLE 换成 DOUBLE、同时把 `C→N` 换成 `C→P`，差分模 11.22（`W_joint` 投影 13.41），四结构 Train 分子支持 17,916 / 704 / 13,063 / 204。**但同样的可读差分在原始 A 中同样存在（A 的 RMS 差分为 1.49，B 为 1.14），所以这不是 B 独有的新证据** |
| **P5：参数与计算效率是否改善？** | **参数量是，算力/时间是中性** | 总参数 hiv 247,106 → 239,970（B）→ 233,506（C），lipo 228,674 → 215,906 → 213,858（B −2.9%，C −5.6%）；**joint 模块**参数 16,416 → 9,280（B）→ 2,816（C）。「有 Train 支持」的 joint 参数反而升高（hiv 4,064 → 9,280、lipo 2,784 → 3,648），因为原始 A 的 embedding 表里绝大多数行从未被任何 Train 结构触发；C 最低（2,816 / 1,600）。推理成本 0.10–0.13 ms/分子，三臂无差别；每 epoch 时间 B/C 与 A 同一量级（hiv 11.2/12.4 s，但历史 A 的 8.6 s 只评估 Valid，口径不同，不做等量比较） |

### 1.3 判定与「是否获得足以推进的新证据」

按协议 §11 的读法：

- **hiv：B>A 且 B>C** → 符合「支持继续研究联合结构函数」的那一支。
- **lipo：B>A 但 B≈C，且 C 在 Valid 上明显更好** → 符合「优先研究更简单的边际函数」的那一支。
- 两个任务的读法不一致，属于协议要求明确记录的 **split/task 冲突**，本轮不强行合成单一结论。

**是否获得足以推进单模型 LCRC 的新证据：获得了一条「值得继续、但尚不足以归因」的正向证据。**
具体地：

1. 把原始 LCRC 的 `joint_proj(path_u(p) * path_v(e))` 换成按 Train 支撑集索引的等价表（B），**在两个任务、两个 split 上都优于原始 A**（hiv +0.0066/+0.0161 ROC-AUC；lipo −0.0051/−0.0198 RMSE）。这是本轮唯一跨任务一致的性能信号。
2. 该参数化在训练后确实运行在非加性区（P3），且其非加性部分无法用边际函数复现（C 在 hiv 上低 0.022–0.036 AUC）。
3. 但 **B 与 C 的自由度和优化性质不同**（B 290×32 表 vs C 88×32 的边际参数化），且 lipo 上 C 反而最好，因此**不能把 B 的增益归因于「真实结构的联合语义」**；它至多说明「按支撑集索引的联合表」是一个不劣、参数更省、且在两任务上优于原参数化的替换。
4. 证据强度限制：单 seed（协议规定），且 hiv 的 scaffold 分组在 Valid/Test 上几乎全部是单分子簇（4,112 簇 / 4,113 分子），配对 bootstrap 实际退化为分子级重抽，区间偏窄；hiv Valid ROC-AUC 的区间 [−0.019, +0.031] 跨 0。

因此：**方向可以继续，但「联合结构函数本身带来任务价值」这一层尚未被证明**；能站得住的说法是「Sup-set 索引的联合表替换（B）在两个任务的 Valid/Test 上都不劣于原始 LCRC，并在 hiv 上稳定更好，同时保留了不可加计算」。

---

## 2. 本轮范围与复用

- 只做：实现、跑实验、正确性检查、报告。
- 复用（只读，未修改）：`ptb_model.py` / `ptb_train_mol.py`（hiv 图与配方顺序）、`dissect_model.py` / `dissect_train.py` / `dissect_data.py`（lipo 图与配方）、`mb2_data.py` / `mb2_metrics.py` / `lcrc_data.py`（缓存、collate、指标）、`lcrc_sparse_identity_v1/runs/mol_ogbg-molhiv_s0_A0`（hiv A 臂）、`lcrc_lipo_model_dissection_v1/runs/r122_C1_s0`（lipo A 臂）。
- 未做：集成/蒸馏/多教师/MoE/辅助监督/动态 Gate/新结构词表/新增消息传递；未用任何历史训练权重初始化新模型（B/C 的初始化来自**同 seed 的原始 A 随机初始化状态**，逐位一致，见 gate 4/5）。
- 历史是否已有「完全相同的联合函数替换 + 从头训练」：没有（`lcrc_computational_commonality_v0` 是冻结推理替换，其他轮次只读取 `path_u/path_v/joint_proj`），故 B/C 必须从头训练。

## 3. Train-only 联合支撑集（`analysis/support_graph.json`，`analysis/support/{hiv,lipo}_support.npz`）

`p = path_cat`，`e = path_end63`；`bond_type_2` 在两者中重复编码，合法性约束 `p % 5 == e % 5`。

| task | Omega | p | e | 连通分量 | rank(X) | 联合子空间维 | 四边矩形 | Train 覆盖 | Valid 覆盖 | Test 覆盖 | 合法类型对观测率 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hiv | 290 | 59 | 35 | 6 | 88 | 202 | 4,658 | 1.000000 | 0.999670 | 0.999744 | 4.53% |
| lipo | 114 | 29 | 25 | 4 | 50 | 64 | 488 | 1.000000 | 1.000000 | 0.999538 | 1.78% |

- Omega 只由 Train 决定（gate 2 复算相等）；未出现组合一律不进 Omega，不构造笛卡尔积。
- 两任务的联合子空间维数都 > 0，联合机制比较继续进行。
- 未见组合比例：hiv Valid 121/366,462（0.034%，涉及 0.34% 分子）、Test 85/331,532（0.026%，0.49% 分子）；lipo Test 16 次（0.048%，0.48% 分子）。
- 回退分支计数（B/C 共用，固定）：hiv Omega 290 / 边际 LS 325 / 类型等权均值 31,385；lipo 114 / 140 / 31,746。
- 联合计算的**使用覆盖**：Omega 的 290 行中 hiv Valid 实际出现 227 行、Test 221 行（lipo：85 / 89），即训练得到的联合表现在只被真实结构用到 hiv 76–78% / lipo 75–78%，其余行在 Valid/Test 上从未被触发。

## 4. 训练前正确性检查（`analysis/checks.json`，`git_commit 0d8079d4`，device `cuda:0`）

12/12 通过，两任务 `failed = []`，且 `init_hash_matches_history = true`
（hiv A0 初始化 hash `849adda7fc6a4f60…` 与 `lcrc_sparse_identity_v1` 记录一致；lipo C1 `aac1e061176a93bd…` 与其历史记录一致）。

| gate | 关键读数（hiv / lipo） |
|---|---|
| 1 split 身份 | 官方 32901/4113/4113 与 3360/420/420，row ID 无交叠，FeatureSpec hash、train-only 归一化、标签有限均一致 |
| 2 Omega 仅 Train | 与 Train 唯一 `(p,e)` 集合逐项相等 |
| 3 类型编码 | `path_emb`/`end_emb` 行数 = `n_path_codes`/`n_nbr_codes`，Omega 全部合法（290/290、114/114） |
| 4 公共初始化逐位 | A/B/C 公共张量 `torch.equal` 全真；参数 A 247,106 / B 239,970 / C 233,506（lipo 228,674 / 215,906 / 213,858） |
| 5 B 复现 A 初始 | joint 3.6e-7 / 3.6e-7，h0 1.4e-6 / 1.9e-6，logit 1.9e-6 / 1.4e-6（C 按定义偏离：1.75/1.07） |
| 6 梯度 | 无 `None`、无零梯度；B 的 `J_table` 有梯度行 91/290、77/114，C 的 `Theta` 全行有梯度 |
| 7 不变性 | 原子重编号 / 路径枚举顺序 / batch 重组：A/B/C 均 ≤1e-6 |
| 8 无 S2 路径 | 部分与全部分子去路径均可运行，未受影响分子预测不变 |
| 9 投影器 | 对称 0.0，幂等 1.1e-15，`U` 正交 3.3e-15，`rank = |P|+|E|−分量` 成立 |
| 10 C 的可加约束 | float64 参数化残差 0.0；float32 模型路径 4.8e-8 / 4.1e-8 |
| 11 四边交互为零 | float64 0.0；float32 2.7e-7 / 2.4e-7（4,658 / 488 个矩形全查） |
| 12 未见回退 | 两分支均被触发、有限、不新增参数、平均行一致 |

训练后对训练完成的 C 再复查（`function_decomposition.json → post_training_identity_checks`）：`(I−P)J_table` 相对残差 0.0（两任务），矩形交互差分 float64 = 0.0、float32 ≤ 3.9e-7 → 恒等式在训练后依然成立，C 全程没有离开边际子空间。

## 5. 主表细节（`analysis/main_table.csv` / `.json`）

best checkpoint 重放校验（本次分析进程重新前向 vs 记录值）：hiv A/B/C 主指标 |Δ| = 1.5e-6 / 0 / 0；lipo 5.9e-9 / 2.1e-8 / 9.0e-8（均 ≤1e-5，A 的 1.5e-6 是 CPU 与 GPU 求值差异）。

- **final epoch 与 last-20**（hiv Valid ROC-AUC）：A 0.8006 / 0.8025，B 0.7943 / 0.7964，C 0.7593 / 0.7615 → hiv 上 B 的 best 优势不是末尾漂移造成的假象，但三者末尾都低于各自 best，A 的 last-20 略高于 B。
  （lipo Valid RMSE）：A 0.7247 / 0.7262，B 0.7278 / 0.7278，**C 0.6828 / 0.6830** → C 的优势在 last-20 上同样成立，不是单点噪声。
- **训练/测试缺口**：hiv A/B/C 的 Train ROC-AUC 0.988 / 0.977 / 0.857，C 明显欠拟合联合通道（其 best 出现在第 13 epoch，随后 Valid 单调变差）。
- **有 Train 支持的 joint 参数**：A 的 4,064（hiv）= 59×32 + 35×32 + 1056，说明原参数化里 16,416 个参数中有 12,352 个从未被任何 Train 结构触发；B 的 9,280 = 290×32 全部可达。**hiv 上 B 的「有支持参数」比 A 多**——这是「按支撑集重参数化」的直接后果，不是容量放大（总参数仍更少）。lipo：A 2,784（29×32+25×32+1056）→ B 3,648 → C 1,600。
- **GPU 时间**：hiv B 0.199 GPU·h、C 0.220（各 100 epoch，每 epoch 评估三个 split）；lipo B 0.068、C 0.071（360 epoch）。A 的历史值 0.238（hiv）/0.104（lipo）只评估 Valid，**口径不同，不用于等量比较**。
- **推理**：Valid/Test 每分子 0.10–0.13 ms（CPU 上的同进程实测），三臂无实质差别；B/C 没有增加前向开销。

## 6. 配对 scaffold-cluster bootstrap（`analysis/main_table.json → bootstrap`，2,000 次，seed 20261126）

Murcko scaffold（`includeChirality=False`）分组：hiv Valid 4,112 簇 / 4,113 分子、Test 4,112/4,113（**几乎每分子一簇**，故等价于分子级重抽，区间偏窄，需谨慎解读）；lipo Valid 356/420（最大簇 7）、Test 357/420（最大簇 12）。

| task | split | 对比 | 主指标 Δ (95% CI) | 辅指标 Δ (95% CI) |
|---|---|---|---|---|
| hiv | Valid | B−A | ROC-AUC **+0.00662** [−0.01939, +0.03096] | AP **+0.04435** [+0.00248, +0.09433] |
| hiv | Valid | B−C | ROC-AUC **+0.02192** [−0.01994, +0.06172] | AP +0.05480 [−0.01439, +0.12561] |
| hiv | Valid | C−A | ROC-AUC −0.01530 [−0.06042, +0.03148] | AP −0.01044 [−0.08210, +0.06499] |
| hiv | Test | B−A | ROC-AUC **+0.01607** [−0.00284, +0.03503] | AP +0.02463 [−0.00295, +0.05209] |
| hiv | Test | B−C | ROC-AUC **+0.03616** [+0.00172, +0.07028] | AP −0.00051 [−0.05387, +0.05343] |
| hiv | Test | C−A | ROC-AUC −0.02009 [−0.05798, +0.01777] | AP +0.02513 [−0.02561, +0.07619] |
| lipo | Valid | B−A | RMSE **−0.00510** [−0.02099, +0.01535] | MAE −0.00270 [−0.02106, +0.01502] |
| lipo | Valid | B−C | RMSE **+0.03661** [−0.00300, +0.07139] | MAE +0.01488 [−0.00335, +0.03309] |
| lipo | Valid | C−A | RMSE **−0.04171** [−0.08485, +0.00862] | MAE −0.01759 [−0.04230, +0.00611] |
| lipo | Test | B−A | RMSE **−0.01979** [−0.04414, +0.00438] | MAE −0.01325 [−0.03105, +0.00506] |
| lipo | Test | B−C | RMSE −0.00194 [−0.02481, +0.02224] | MAE +0.00260 [−0.01641, +0.02375] |
| lipo | Test | C−A | RMSE −0.01785 [−0.04740, +0.00981] | MAE −0.01585 [−0.04120, +0.00758] |

**去掉极端误差样本的敏感性（lipo，对称删除两模型各自 |error| 最大的 k=1/5/10 的并集）**：

- Valid B−A RMSE：全样本 −0.00510 → drop1 **+0.00286**、drop5 **+0.00185**、drop10 **+0.00175**（符号翻转：**B 在 lipo Valid 上的微弱优势完全由少数极端误差分子驱动**，去掉后 B 略差于 A）。
- Test B−A RMSE：−0.01979 → drop1 −0.02270、drop5 −0.02974、drop10 −0.02700（符号稳定）。
- Valid B−C：+0.03661 → drop1..10 +0.0152…+0.0169（C 的优势保持）；Test B−C：≈0（±0.003，符号随裁剪改变，等于没有差别）。
- Valid Test C−A：−0.0417/−0.0179 → 裁剪后 −0.0121…−0.0125（Valid）与 −0.0220…−0.0260（Test），符号稳定。

统计读法：hiv 的 B−A 区间跨 0（主指标），只有 AP 与 hiv Test 的 B−C 区间排除 0；lipo 全部主指标区间跨 0。**单 seed + 上述区间 → 只能说「方向一致的正向信号」，不能声称显著或跨 seed 稳定。**

## 7. 精确结构函数分解与冻结干预（`analysis/function_decomposition.json`）

同一 Train-only `P`，`J_add = P J`，`J_int = J − P J`（类型等权；`J_add ⟂ J_int` 由投影器保证，实测内积 2.5e-16 / −6.2e-17，投影闭包误差 ≤4.2e-15）。

| task | 臂 | RMS(J) | RMS(J_add) | RMS(J_int) | J_int/J_add（等权） | J_int/J_add（出现加权） | W_joint 投影后比值 |
|---|---|---|---|---|---|---|---|
| hiv | A | 0.7360 | 0.3920 | 0.6229 | 1.589 | 2.190 | 1.648 |
| hiv | B | 0.5829 | 0.3264 | 0.4829 | 1.479 | 1.844 | 1.548 |
| hiv | C | 0.3211 | 0.3211 | **0.0000** | 0 | 0 | 0 |
| lipo | A | 0.6187 | 0.3992 | 0.4727 | 1.184 | 1.578 | 1.231 |
| lipo | B | 0.5664 | 0.3715 | 0.4275 | 1.151 | 1.455 | 1.206 |
| lipo | C | 0.3716 | 0.3716 | **0.0000** | 0 | 0 | 0 |

- 原始 A 与 B 的「加性 / 非加性」比例几乎相同（1.59 vs 1.48；1.18 vs 1.15），说明 B 不是靠削弱非加性来拟合，而是保持了同类的分解结构，同时整体尺度更小。
- C 的非加性分量严格为 0（训练后复查同样为 0），这是函数空间约束的直接体现。

**冻结干预（B 的同一 best-Valid checkpoint，只替换 joint 表，其余全部不变）**

| task | split | ORIGINAL | ADD_ONLY | HALF_INTERACTION | Δ主(ADD) | Δ主(HALF) | 平均输出变化(ADD) |
|---|---|---|---|---|---|---|---|
| hiv | Valid | 0.83806 | 0.82463 | 0.83535 | **−0.01343** | −0.00271 | 0.87 logit |
| hiv | Test | 0.78139 | 0.75989 | 0.77656 | **−0.02150** | −0.00483 | 0.92 logit |
| lipo | Valid | 0.71139 | 0.87204 | 0.74847 | **+0.16065** | +0.03708 | 0.38 原始单位 |
| lipo | Test | 0.56660 | 0.74636 | 0.60726 | **+0.17976** | +0.04066 | 0.37 原始单位 |

按分子自身联合结构覆盖率分桶（`full_1.0` 等）：hiv Test 中 4,093/4,113 个分子覆盖率 1.0，其 ADD_ONLY 的 ROC-AUC 由 0.7858 降到 0.7615；覆盖率 0.9–1.0 的 17 个分子由 0.125 变 0.250（样本太小，不解读）。lipo Test 418/420 覆盖率 1.0，ADD_ONLY 的 RMSE 由 0.5657 升到 0.7474。
→ 训练后的 B 依赖不可加分量（尤其在真实结构几乎全覆盖的分子上），去掉它就明显变差。

## 8. 从真实结构组合提取的交互规律（`analysis/interaction_contrasts.json`）

差分定义 `D = J(p1,e1) + J(p2,e2) − J(p1,e2) − J(p2,e1)`，只对 Omega 中真实存在的四组合计算。候选规则（冻结）：四对各自 Train 独立分子支持 ≥20，按 B 的 `||W_joint D||` 降序取 5。

| task | 臂 | 矩形数 | RMS(D) | max |D| | RMS(W_joint D) |
|---|---|---|---|---|
| hiv | A | 4,658 | 1.493 | 7.77 | 0.581 |
| hiv | B | 4,658 | 1.140 | 6.13 | 0.414 |
| lipo | A | 488 | 1.263 | 5.19 | 0.279 |
| lipo | B | 488 | 1.136 | 4.97 | 0.260 |

**hiv 候选 1**（`||W_joint D|| = 13.41`，四结构 Train 分子支持 17,916 / 704 / 13,063 / 204）：

```
(p1,e1) root -[SINGLE]-> C  -[SINGLE]-> N      ||D|| = 11.22
(p1,e2) root -[SINGLE]-> C  -[SINGLE]-> P
(p2,e1) root -[DOUBLE]-> C  -[SINGLE]-> N
(p2,e2) root -[DOUBLE]-> C  -[SINGLE]-> P
```

即「根—中间原子键的级（SINGLE↔DOUBLE）」与「末端原子（N↔P）」的**二阶交叉**是模型内部一个真实存在的非加性规律：单看换键或单看换末端都有各自效应，但两者不可简单相加。Valid/Test 支撑：2,135/85/1,547/19 与 2,206/103/1,374/17 个分子（`(p2,e2)` 在 Test 只有 17 个分子，注意）。

**lipo 候选 1**（`||W_joint D|| = 7.34`，`||D|| = 11.92`）：`root -[AROMATIC]-> C_ar -[AROMATIC]-> C_ar` / `…-> N_ar` / `root -[SINGLE]-> N_ar -[AROMATIC]-> C_ar` / `…-> N_ar`，四结构 Train 支持 3,320 / 2,331 / 714 / 229；即「根键级（AROMATIC↔SINGLE）」与「末端 N 是否在芳香端」的交叉。

其余 4 个候选（两任务各 8 个结构描述、split 支持、反例）见 JSON：均是可解码的真实原子/键类型组合，且四对都在 Omega 中真实出现（没有构造非法组合；`counterexamples` 里记录了同类型但**不在** Omega 的成对组合，即「该规律在其它类型组合上并未被支撑」）。

**解读边界（协议要求）**：D 是模型内部联合计算的差分，不是分子性质之间的因果交互，也不是对最终预测的贡献分解；A 与 B 的隐坐标不对齐，不比较单通道语义。需要特别说明：**同样的可读差分在原始 A 中已存在且更强**（RMS 1.49 vs 1.14），所以 §8 支持「可提取、可核验」但**不支持**「B 学到了新的结构规律」。

## 9. 判定细节与冲突记录

1. **hiv B vs A**：Valid +0.00662（参考 +0.005，刚过）、Test +0.01607；方向在 Train-末尾（last-20）上反过来了（A 0.8025 > B 0.7964）。也就是说 B 的优势来自「更早出现更好的 checkpoint」（43 vs 53），而不是「收敛得更好」。协议规定按 Valid 选择，故主表以 best 为准，但这个末段反转必须记录。
2. **hiv B vs C**：C 的联合通道欠拟合（Train ROC-AUC 0.857 vs 0.977），best 出现在第 13 epoch。C 的失败不能直接读作「边际函数不足」，因为它同时是「容量更小 + 优化更差」。
3. **lipo B vs A**：Valid 的 −0.0051 在去掉 1 个极端误差分子后即翻正（+0.0029）；Test 的 −0.0198 在同样裁剪下保持（−0.023…−0.030）。**lipo Valid 上「B 优于 A」不稳健。**
4. **lipo C vs A**：Valid −0.0417（−5.8%）、Test −0.0179（−3.0%），裁剪与 last-20 都保持，是本轮最稳的单一改进——但它来自**更简单的边际函数**，与「联合结构有价值」相反。
5. **Test 与 Valid 方向不一致的实例**：hiv B−C 的 AP 在 Test 上为 −0.0005（打平）而 ROC-AUC +0.036；lipo B−C Valid 明显、Test 打平。任何「B 全面更好」的说法都不成立。
6. 未使用 Test 做任何选择；Test 仅报告，标签 `DEVELOPMENT_TEST`（A 臂沿用 `EXPOSED_TEST_EXPLORATORY`）。

## 10. 局限

- 单 seed（协议规定），只能作为架构筛选，不能声称跨 seed 稳定性；所有 bootstrap 区间都是同一 seed 下的重抽。
- hiv 的 Valid/Test Murcko scaffold 几乎每分子唯一（4,112 簇 / 4,113 分子），所谓「scaffold-cluster bootstrap」在该任务上近似分子级重抽，区间偏窄。
- B（290×32 表）与 C（88×32 边际参数化）自由度不等，C 不是等容量对照；B/C 的差异混合了「函数约束」与「优化难度」。
- Omega 是 Train-only 的经验支撑集；Valid/Test 有 0.03–0.05% 的出现落在未见组合上，走固定回退（边际 LS 或类型均值），回退值不被解释为「已学到该联合结构」。
- hiv/lipo 两任务的训练量差异很大（100 vs 360 epoch，32901 vs 3360 分子），跨任务比较只能看方向，不能比较增益大小。

## 11. 复现与工件

```bash
# 支撑集（CPU，缓存来自 lcrc_molnet_benchmark_pilot_v2）
# 重跑会得到与已提交 .npz 逐字节相同的文件（已实测 sha256 相同），图除 created_utc 外完全相同
python tracks/structural-sufficiency/lcrc_support_joint_function_v0/code/sj_support.py --tasks hiv lipo
# 12 项训练前检查（GPU）
python .../code/sj_gates.py --device cuda:0            # → analysis/checks.json
# 吞吐探测（GPU）
python .../code/sj_probe.py --device cuda:0            # → analysis/throughput_probe.json
# 四个正式单元（各一次）
python .../code/sj_train.py --task hiv  --arm B --seed 0 --device cuda:0
python .../code/sj_train.py --task hiv  --arm C --seed 0 --device cuda:0
python .../code/sj_train.py --task lipo --arm B --seed 0 --device cuda:0
python .../code/sj_train.py --task lipo --arm C --seed 0 --device cuda:0
# 分析（CPU）
python .../code/sj_analyze.py --device cpu             # → main_table / decomposition / contrasts
```

| 工件 | 内容 |
|---|---|
| `PROTOCOL.md` | 冻结协议（先于训练提交） |
| `configs/experiment.json` | 冻结配置（`sj_config.py` 校验） |
| `code/` | `sj_common/support/model/train/gates/probe/analyze/config` + `tests/test_sj.py`（6 项快速结构测试） |
| `analysis/support_graph.json` + `analysis/support/*.npz` | Train-only 支撑集、X/U/P/pinv、code 表、覆盖与合法性（加载时复算校验） |
| `analysis/checks.json` | 12 项检查（GPU，两任务全过） |
| `analysis/throughput_probe.json` | 训练前吞吐与预算（预测 0.93 GPU·h，实测 0.558） |
| `analysis/main_table.csv` / `main_table.json` | 主表 + 重放校验 + 配对 bootstrap + 裁剪敏感性 |
| `analysis/function_decomposition.json` | §7 分解、§9 干预（含覆盖率分桶）、训练后恒等式复查 |
| `analysis/interaction_contrasts.json` | §8 交互差分、候选结构、split 支撑、反例 |
| `runs/{task}_{arm}_s0/` | `run.json`、`curve.csv`（每 epoch 三 split 指标）、`checkpoint.pt`（best-Valid）、`{train,valid,test}_predictions.npz` |

**provenance 记录（不影响结论，但需如实登记）**：部署到 `res` 时，远端 checkout 在 commit `f6361d60` 上存在另一轮次（`context_conditioned_structural_computation_v0`）的 72 个未跟踪结果文件，其中 71 个与新 commit 将要写入的字节完全一致（逐文件 sha256 比对）、1 个（`analysis/queue_cudacuda_0.json`）本地无对应副本。为避免破坏该轮次的任何字节，这些文件被整体打包转存到
`res:/home/hxy/cy/research_deploy_backup_20261125T092300Z/csc_untracked.tar`（sha256 `71d0d9f825fa143b3ccd89837536559ebd207f18f9d67637c88b6fc55ece97ad`，路径清单在同目录 `manifest.sha256`）后才执行部署；本轮未删除、未修改、未清理任何其他轮次的文件。检查结果文件与吞吐探测结果在后续部署中被同样转存到 `…/research_deploy_backup_20261125T095142Z/`，并在本地 commit 后重新部署。
