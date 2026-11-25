# LCRC 支撑集联合结构函数实验 v0 — PROTOCOL（冻结）

- Round: `tracks/structural-sufficiency/lcrc_support_joint_function_v0/`
- Round id: `lcrc-support-joint-function-v0`
- 状态：本文件在**任何正式 GPU 训练、任何 Valid/Test 指标比较之前**写成并提交。
  之后只允许追加"实际发生了什么"的事实记录（运行记录、指标、ledger、判定），
  不允许修改臂、seed、epoch、配方、划分、支撑集定义、回退规则、候选规则、阈值与预算。
- 执行 regime：`res`（a100-2，process 后端，pool `res-gpu1`，torch 2.5.1+cu124），
  与两个被复用基准的训练 regime 相同。
- 冻结配置：`configs/experiment.json`（由 `code/sj_common.config_payload()` 生成，
  `code/sj_config.py` 校验一致）。本文件的每个数字都由该配置或 `sj_common.py` 唯一决定。

## 0. 研究目标（verbatim scope）

1. 改善 LCRC 单模型的结构函数学习能力。
2. 检验联合结构函数相对边际结构函数的任务价值。
3. 从真实结构组合中提取可精确验证的计算交互规律。

不做：集成、蒸馏、多教师、MoE、辅助监督、动态 Gate、新结构词表、新增消息传递。
不使用任何历史训练完成的模型权重初始化新模型。允许复用历史代码、数据、初始随机状态
和完全匹配的基准结果。

## 1. 历史资产核对（先做，再设计）

| 资产 | 用途 | 结论 |
|---|---|---|
| `lcrc_predictive_trace_benchmark_v1/code/ptb_model.py` | `PTBLCRC`（A0 图）+ 冻结同 seed 初始化 | 直接复用（只读） |
| `lcrc_predictive_trace_benchmark_v1/code/ptb_train_mol.py` | MolHIV 训练单元（配方顺序、shuffle 生成器、损失） | 逐项复刻（只读） |
| `lcrc_lipo_model_dissection_v1/code/dissect_model.py` | `DissectLCRC`（A0/C1 图）、`variant_cols`、`remap_state`、联合通道实现 | 直接复用（只读） |
| `lcrc_lipo_model_dissection_v1/code/dissect_train.py` | Lipo 360-epoch 配方顺序 | 逐项复刻（只读） |
| `lcrc_lipo_model_dissection_v1/code/dissect_data.py` | Lipophilicity public122 官方划分、结构缓存、train-only 目标归一化 | 直接复用（只读） |
| `lcrc_sparse_identity_v1/runs/mol_ogbg-molhiv_s0_A0` | **MolHIV A 臂**（完全匹配基准） | 复用，不重训 |
| `lcrc_lipo_model_dissection_v1/runs/r122_C1_s0` | **Lipo A 臂**（官方划分 C1 + 小任务配方） | 复用，不重训 |
| `lcrc_molnet_benchmark_pilot_v2` / `lcrc_cross_task_preparation_v1` | 结构缓存、`lipo_v1` FeatureSpec、collate、指标 | 直接复用（只读） |
| `lcrc_computational_commonality_v0` | 冻结推理下的 S2 结构函数替换（Part B） | 只做冻结推理替换，**不重训**；不可复用为本轮结果 |
| `lcrc_tu_quick_screen_v1`、`lcrc_explicit_semantic_learning_8h_s0_v1`、`lcrc_readout_context_pilot_small_v1` | 同族 A0/C1 图与 TU 适配 | 无本轮所需的联合通道替换实验 |

**是否已存在"完全相同的联合函数替换 + 从头训练"实验**：没有。
`lcrc_computational_commonality_v0` 在冻结模型上替换结构函数（无梯度、无重训）；
`lcrc_sparse_identity_v1` 增加的是词表式 S2 身份分支（不是替换 joint 通道）；
`lcrc_esol_physchem_s2_v1`、`lcrc_lipo_screen_v1`、`lcrc_taskaware_representation_pilot_v1`
只在审计/删除实验中读取 `path_u`/`path_v`/`joint_proj`，没有支撑集索引表的重训练实验。
因此 B/C 必须在本轮从头训练；A 完全匹配，直接复用。

**S2 联合函数核对**（来自 `dissect_model.LocalContent`）：

```
S2_joint_v = sum_{v->w->u} joint_proj( path_u(path_cat) * path_v(path_end63) )
joint_proj = Linear(rank -> rank)   # 含 bias，rank = 32
```

`path_cat = middle_type*25 + (bt1-1)*5 + (bt2-1)`，`path_end63 = end_type*5 + (bt2-1)`
（`lipo_v1` FeatureSpec）。因此 **bond_type_2 在 p 与 e 中重复编码**，合法性约束为
`p % 5 == e % 5`；`n_path_codes = 400`，`n_nbr_codes = 80`，`rank = 32`，`width = 64`。
模型列布局（A0，529 列）：`s2_joint` 占 `[384, 416)`；C1 去掉 `b_mean` 96 列后同一块位于
`[288, 320)`。本轮**不硬编码**该偏移，一律由 `model.local.local_cols` 反查。

## 2. 数据与配方（冻结）

主任务 `ogbg-molhiv`，官方 Train/Valid/Test = 32901/4113/4113（OGB scaffold split）。
补充任务 `Lipophilicity`，public122 官方 Train/Valid/Test = 3360/420/420（ChemBench repeat 122）。

MolHIV 配方（= `lcrc_sparse_identity_v1` A0 原始骨干与配方）：width 64、rank 32、100 epochs、
batch 128、AdamW（β=(0.9,0.999)，eps 1e-8）、peak LR 3e-4、per-step cosine 至 1e-5、
weight decay 1e-3、gradient clip 5、dropout 0.10（作用于 `h`，单次调用）、
原始 `BCEWithLogitsLoss`、per-step scheduler、shuffle 生成器 `seed + 917000`。
主指标 ROC-AUC，辅指标 AP。

Lipo 配方（= `lcrc_lipo_model_dissection_v1` C1 原始配方）：width 64、rank 32、360 epochs、
batch 128、AdamW 3e-4→1e-5（per-step cosine）、wd 1e-3、clip 5、dropout 0.10、
标准化目标上的 MSE、train-only 目标归一化。主指标原始单位 RMSE，辅指标 MAE。

seed 全部为 0。每 epoch 完整评价 Train/Valid/Test；checkpoint 只按 Valid 主指标选择
（binary：最高 Valid ROC-AUC，平局取更低 Valid BCE，再取更早 epoch；regression：最低 Valid RMSE，
严格改善，平局保留更早 epoch）。Test 标记 `DEVELOPMENT_TEST`，不参与任何选择。
同一 task 的各臂使用完全相同的配方、数据与训练顺序；RNG 播种点与被复用基准的 loop 一致
（MolHIV 在建模前、Lipo 在建模后，见 `SEED_TIMING`）。

历史完全匹配基准直接复用、不重训。历史曲线的逐 epoch Train/Test 指标缺失时如实记为缺失
（MolHIV A0 曲线只有 Valid ROC-AUC/AP/BCE 与耗时；Lipo C1 曲线只有 Valid 原始 RMSE/MAE/R² 与
标准化 Train MSE）。

## 3. Train-only 联合结构支撑集

使用实际生产 S2 记录：`p = path_cat`、`e = path_end63`。定义
`Omega = { (p,e) : 该组合在 Train 中真实出现 }`。不构造未出现的笛卡尔积，不增加根类型、
端点完整属性、闭环类型或其他结构字段。原始其他 S2 通道保持不变。两个任务各构造一次 Omega。

记录：真实键集合（升序 `(p,e)`）、出现次数、独立分子支持（不同 Train 分子数）、
各 split occurrence 覆盖率与含未见组合的分子比例、合法类型约束、p/e 中重复的键类型字段。

Train-only 二部支撑图：左=实际出现的 `p`，右=实际出现的 `e`，每个 `(p,e)` 一条边。

关联设计矩阵 `X`（`|Omega| x (|P|+|E|)`）：每行对应一个 Omega 元素，行内**恰有两个取值 1 的
列**——该元素 `p` 的列与该元素 `e` 的列。这就是二部关联矩阵：`span(X)`（以 Omega 元素为坐标、
类型等权内积）恰好是全部可加函数 `a_p + b_e`。以此求 float64 精简 SVD `X = U S V^T`，
保留 `s > 1e-10 s_max` 的列得 `U`，`P = U U^T`，`I − P` 为非加性联合计算子空间投影。
采用类型等权定义（每个 Omega 元素权重 1），不搜索权重。

记录连通分量数、`rank(X)`、联合子空间维数 `|Omega| − rank(X)`、实际存在的四边交互矩形数量
（`sum_{e1<e2} C(|N(e1) ∩ N(e2)|, 2)`）。若联合子空间维数为零，停止该任务的联合机制比较并报告原因。

**实测（`analysis/support_graph.json`，冻结）**

| task | `|Omega|` | `|P|` | `|E|` | 分量 | `rank(X)` | 联合维数 | 矩形 | Valid occ 覆盖 | Test occ 覆盖 | Test 含未见分子 |
|---|---|---|---|---|---|---|---|---|---|---|
| hiv | 290 | 59 | 35 | 6 | 88 | 202 | 4658 | 0.99967 | 0.99974 | 0.00486 |
| lipo | 114 | 29 | 25 | 4 | 50 | 64 | 488 | 1.00000 | 0.99954 | 0.00476 |

两个任务的联合子空间维数均非零，比较继续进行。

## 4. 模型

保留原 LCRC 原子编码器、键编码器、S1、`path_emb`、`end_emb`、`closure`、endpoint content、
`path_gate`、scalar、LocalMLP、LayerNorm 与整图读出。唯一修改是原始 joint 通道：

```
A（ORIGINAL）  J(p,e) = joint_proj( path_u(p) * path_v(e) )         # 保持全部原参数与公式
B（JOINT）     J(p,e) = J_table[index(p,e)]， J_table ∈ R^{|Omega| x r}
C（ADDITIVE）  J_table = U Theta，          Theta ∈ R^{rank(X) x r}
```

`*` 为逐元素乘积；输出维度 `r = 32` 不变；不增加新的 MLP。B 删除 `path_u`、`path_v`、
`joint_proj` 并以其等价的表替代；C 与 B 使用同一 Omega 与输出维度，但表被限制在边际子空间，
不添加任何联合剩余项。C 的有效自由度低于 B，这是本轮有意检验的函数空间约束，**不是等容量对照**。

每次真实路径出现仍然按原根节点、原方向和原重数聚合（沿用 `LocalContent` 的 `index_add`）。

初始化（数学等价重参数化，不加载任何训练权重、不蒸馏）：
使用同 seed 的原始 A 随机初始化状态，逐个 Omega 元素计算 `J_A(p,e)`（含原来 `joint_proj` 的 bias），
物化为 `J_table_0`；`Theta_0 = U^T J_table_0`。因此 C 的初始表是 B 初始表的正交边际投影。
A/B/C 的公共参数逐位一致（gate 4），并且 A/B 在 Train 已见类型上的初始前向相等（gate 5）。

**未见类型处理（B/C 同一固定 Train-only 规则）**

- 若 `p`、`e` 均在 Train 中出现且处于支撑图同一连通分量：使用当前 `J_table` 对 `X` 的最小二乘
  边际拟合，取 `Jhat(p,e) = Theta_hat[p] + Theta_hat[e]`（`Theta_hat = pinv(X) J_table`，
  预先算好伪逆，前向不做 SVD）。
- 否则使用当前 `J_table` 的类型等权均值（Omega 行等权）。

未见类型不增加可学习参数。记录未见 occurrence 比例与含未见类型的分子比例（§3 表）。
不得把回退解释为"已经学到该联合结构"。

## 5. 实验队列与预算

两个任务比较 A/B/C，最多六个正式训练单元；完全匹配的 A 复用，因此实际训练 4 个：
`hiv-B`、`hiv-C`、`lipo-B`、`lipo-C`。seed 0。

执行：`res` 上 GPU1 单卡顺序执行（GPU0 被他人任务占用，`nvidia-smi` 于冻结时实测
90% 利用率 / 20.4GB，故不并行、不占用）。一卡一训练进程。

预算：GPU 目标 2 GPU·h，硬上限 3 GPU·h；墙钟硬上限 2 小时。
正式训练前先测量包含三个 split 全量评价的实际吞吐（`--smoke` 计时探测，记录到
`analysis/throughput_probe.json`），确认预算能容纳完整对照组；若资源不足，优先完成 MolHIV 完整对照组，
不通过缩短某一臂来制造不匹配结果。部署/缓存/主机状态阻塞时最多一次正常恢复尝试；
达到硬上限立即停止并交付已完成结果。

## 6. 正确性检查（正式训练前必须通过，`analysis/checks.json`）

1. 官方 split、row ID、标签及缓存身份一致（尺寸、published rows、互不相交、FeatureSpec hash、
   train-only 归一化、标签有限）。
2. Omega 仅使用 Train（等于 Train 唯一 `(p,e)` 集合；Valid/Test 独有组合被排除）。
3. 实际 p/e 类型和原模型编码一致（`path_emb`/`end_emb` 行数 = `n_path_codes`/`n_nbr_codes`，
   collate 记录顺序与 code 一致，全部 Omega 组合合法）。
4. A/B 公共初始参数逐位一致（含 C）。
5. 对 Train 已见类型，物化后的 B 能重现 A 初始 joint、`h0`（LocalMLP 输入）和最终预测。
   C 是同一 draw 的正交边际投影，偏差按定义记录、不作为通过条件。
6. B/C 的参数均能正常获得梯度（无 `None`、无零梯度、联合张量非零且有限）。
7. 原子重编号、路径枚举顺序改变及 batch 重组不改变预测（≤1e-5）。
8. 无 S2 路径的图能够正确运行（部分分子无路径、全批无路径、未受影响分子预测不变）。
9. `P^2 = P`、`P^T = P`（且 `rank(X) = |P|+|E|−分量数`，`U` 正交）。
10. C 的 `J_table` 满足 `(I−P) J_table ≈ 0`（float64 参数化精确；float32 模型路径报数值）。
11. 对实际存在的四边组合，C 的交互差分为零（float64 精确；float32 路径报数值）。
12. B/C 未见类型回退规则一致、有限且不引入新参数。

只允许工程修复，不根据 smoke 性能调整模型公式。任一 gate 失败则停止正式训练并修复。

## 7. 主性能分析

对每个模型报告：总参数量、被替换模块参数量、实际有 Train 支持的参数数量、best epoch、
best checkpoint 的 Train/Valid/Test 主辅指标、final epoch 指标、last-20 epochs 平均指标、
GPU 时间与推理成本；以及 best checkpoint 重放与复用记录的差异（≤1e-5 视为一致）。

主要比较：B vs A（直接联合函数参数化是否改善原始 LCRC）、B vs C（不可加联合计算是否有训练后的
预测价值）、C vs A（更简单的计算函数是否已经足够）。注意 B/A 有效自由度不同，若 B 更强不能直接
归因于真实结构语义。

MolHIV：配对 scaffold-cluster bootstrap 的 ROC-AUC/AP 差异（Murcko scaffold 分组，2000 次重抽，
seed 20261126，Valid 与 Test 分别报告）。Lipo：配对 scaffold-cluster bootstrap 的 RMSE/MAE 差异
及去掉少量极端误差样本的敏感性（对称删除两模型各自 |error| 最大的 k=1/5/10 的并集）。

单 seed 只作为架构筛选，不声称跨 seed 稳定。

## 8. 可解释性：精确的结构函数分解

对 A、B 的 best-Valid checkpoint，在完整 Omega 上提取 `J`，用同一 Train-only `P` 分解
`J = J_add + J_int`（`J_add = P J`，`J_int = J − P J`）。报告两部分 RMS、类型等权与 occurrence
加权的相对幅度、投影闭包误差、联合计算在实际结构中的覆盖率、经 LocalMLP 第一层对应 `W_joint`
投影后的幅度。不同模型之间的隐空间坐标不一定对齐，不直接比较单个通道的语义。
类型等权投影下 `J_add ⟂ J_int`；其他加权口径不保证正交。

## 9. 冻结联合函数干预（B 的同一 best-Valid checkpoint）

`ORIGINAL = J_add + J_int`、`ADD_ONLY = J_add`、`HALF_INTERACTION = J_add + 0.5 J_int`；
其他模型参数、其他路径通道与读出完全不变；未见 code 在三种模式下都保留原表的回退值。
评价完整 Valid/Test，报告主辅指标、logits 变化（平均/最大）、按分子联合结构覆盖率分桶的性能变化。
该干预只用于判断训练后的模型是否依赖非加性联合函数；B 与 C 的独立训练比较才用于判断
联合函数的泛化价值。

## 10. 从真实结构组合中提取交互规律

只考虑 Omega 中实际存在的四组合 `(p1,e1),(p1,e2),(p2,e1),(p2,e2)`，计算
`D = J(p2,e2) − J(p2,e1) − J(p1,e2) + J(p1,e1)`。不构造非法组合。该差分消除任意纯路径项与
纯端点项。候选规则（冻结）：四个组合各自的独立 Train 分子支持均 ≥ 20，按 B 的
`||W_joint D||_2` 降序取最多 5 个。将 p/e 解码为真实原子与键类型，记录四种结构的可读描述、
Train/Valid/Test 出现支持、差分向量范数、`W_joint` 投影后范数、相关结构的真实分子案例、
反例与支持不足情况。不把 D 解释为真实分子属性的因果交互或最终预测贡献；
候选只能称为模型内部的联合结构计算规律。若无足够支持的可读交互，明确报告，不制造案例。

## 11. 判定

主性能参考（路线筛选，不是显著性阈值）：MolHIV ROC-AUC 增量 +0.005；Lipo RMSE 相对改善 3%。

- P1：B 是否在 Valid/Test 同时优于 A？
- P2：B 是否优于 C？
- P3：B 的优势是否伴随实际非加性函数的使用（§9 干预）？
- P4：能否从联合函数中提取有结构意义、支持充分且可核验的计算差分（§10）？
- P5：参数和计算效率是否改善（参数量、有支持参数量、推理成本、GPU 时间）？
- 读法：B>A 且 B>C → 支持继续发展联合结构函数；B>A 但 B≈C → 优先研究更简单的边际函数；
  B≤A 但 B>C → 联合计算有作用但直接参数化未带来性能增量；B≤A 且 C 无优势 → 停止该参数化方向；
  split 冲突明确记录，不强行宣布成功。可解释性单独报告。

## 12. 交付物

`PROTOCOL.md`、`code/`、`configs/experiment.json`、`analysis/support_graph.json`、
`analysis/support/`（支撑矩阵与表，重建校验用）、`analysis/checks.json`、`analysis/main_table.csv`、
`analysis/main_table.json`（§7 的配对统计与主表）、`analysis/function_decomposition.json`
（§8 分解 + §9 干预 + 训练后恒等式复查）、`analysis/interaction_contrasts.json`、
`analysis/throughput_probe.json`、`runs/`（曲线、预测、checkpoint、run.json）、`REPORT.md`。
额外只保留 `analysis/support/` 与 `analysis/throughput_probe.json` 两个必需工件（重建身份与预算证据）。
不做下一版方案，不自动追加实验，不做集成或蒸馏。
