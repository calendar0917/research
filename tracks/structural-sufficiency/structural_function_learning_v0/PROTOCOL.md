# PROTOCOL — 显式结构函数学习与跨任务计算机制验证 v0

Round ID: `structural-function-learning-v0`
Directory: `tracks/structural-sufficiency/structural_function_learning_v0/`
Seed: `0`（单 seed，不做多 seed）
Predecessors（只读，不修改、不重训）：
`interpretable_structural_rule_discovery_v1`（ISRD-v1，冻结 commit `16d7a002`）、
`structural_gain_attribution_v1`（SGA-v1，冻结 commit `a306a63c`）、
`molhiv_multigranular_structure_sharing_v0`（MSDS-v0）。
Compute: CPU（线性、树、分析）+ res GPU1/GPU0（四个新神经模型，经 `rr`）。
墙钟目标 60 分钟，硬上限 120 分钟；GPU 预算 ≤2 A100-hours。

本文件在**任何新模型拟合之前**写入并提交。C 网格、树配方、网络结构、初始化、
折点分位数、epoch、batch、选择规则、bootstrap 设置全部固定。Test 标记
`DEVELOPMENT_TEST`，只报告，不参与任何选择（C、epoch、树轮数、特征、函数展示）。

---

## 1. 问题与对照

* Q1 全量线性结构模型的过拟合能否被更合理的正则/缩放缓解？（Phase A）
* Q2 XGB-ALL 的树优势是否需要结构间交互，还是单结构非线性已足够？（Phase B）
* Q3 神经网络能否学习同样的加性结构函数；类别加权是否解释 HIV 读出差距？（Phase C）
* Q4 上述机制在 MolHIV / ZINC 上是否同向？（Phase D）
* Q5 能否从加性模型精确提取真实、可追踪的单结构函数？（Phase E）

## 2. 数据与冻结输入身份

### MolHIV（官方 scaffold 划分 32901 / 4113 / 4113）
* 特征：`X_H = [log1p(n_atoms), log1p(n_bonds), C(754), F(3652)]`，共 4408 列。
  结构值 = 冻结 `count / sqrt(max(n_atoms,1))`（`bundle/sav1_inputs.npz`）。
* `n_bonds` = tracked 原始图 `bond_ptr` 差 / 2，与历史 `MSDS-v0` 逐分子相等（本地核验）。
* `y`、`rows_*`、`C`/`F` 列顺序 = 冻结 bundle；不引入 Morgan、RDKit descriptors、GNN。
* scaffold 分组只用于 Phase D bootstrap（本地冻结 `MSDS-v0/data/molhiv_scaffolds.npz`）。
* 历史参照（只读）：COUNT-ALL Valid 0.7463 / Test 0.7495；R128 0.8046 / 0.7873；
  XGB-ALL Valid 0.8153 / Test 0.8174（AP Valid 0.3977 / Test 0.3603）。

### ZINC（PyG subset 官方划分 10000 / 1000 / 1000）
* 复用 SGA-v1 冻结 `Z + Z-CF`：`Z`(4642) + `C`(9) + `F`(1250) = 5901 列
  （`data/zinc_design.npz`，不重建词表、不改属性定义）。
* 历史参照（只读）：ZX1 Valid MAE 0.4691 / Test 0.4845；
  ZN1 Valid MAE 0.2841 / Test 0.2744（best epoch 46）。

## 3. Phase 0 — 重算历史 COUNT-ALL 贡献（不重训）

读取 ISRD-v1 冻结 `analysis/phaseC.json` 的 COUNT-ALL 参数（`coef_raw`、
`coef_standardised`、`mean`、`sd`、两个截距）与冻结设计。
对每个分子重算两种精确分解：

1. 零结构基准：`contrib_s = w_raw_s * x_s`；`z = b_raw + size_raw + Σ contrib_s`。
2. 均值中心化基准：`contrib_s = w_std_s * (x_s - mu_s) / sd_s`；
   `z = b_std + size_std + Σ contrib_s`。

两者都必须重建存储 logit（max|Δ| < 1e-8）。主要分析使用零结构基准。
重算：top-5/10/20 绝对贡献占比、结构项 |贡献| 总量、主要贡献之外的**有符号**
剩余项、按 COARSE 父结构汇总 FULL 子型后的组级 top-k 集中度。组级贡献 = 组内
实际模型项之和，不增删项。旧值（重复除以 sd）保留为 ERRATA 对照并明确其被替代。

## 4. Phase A — MolHIV 线性对照（CPU）

统一：`fit_logistic_sparse_standardised`（稀疏实现，不构造稠密 4408 列矩阵），
参数 `C` 只取固定网格，选择规则 = 最高 Valid AUC（平局取更小 C）。

| arm | 监督权重 | 特征缩放 | C | 截距 |
|---|---|---|---|---|
| `L-BAL-STD` | class-balanced（Train，均值 1） | 全部列 Train mean/std | 0.0003, 0.003（+历史 0.03 复用） | 沿用历史（惩罚） |
| `L-NAT-STD` | 普通（1） | 全部列 Train mean/std | 0.0003, 0.003, 0.03 | 不惩罚 |
| `L-NAT-RAW` | 普通（1） | 大小项 Train mean/std；结构计数保持冻结缩放 | 0.0003, 0.003, 0.03 | 不惩罚 |

历史 C=0.03 的 `L-BAL-STD` 直接读取冻结系数与预测，不重训；报告其 Train/Valid/Test
AUC/AP。若最优落在 C 网格边界，明确记录“正则搜索范围有限”。

## 5. Phase B — 加性树 `ADD-TREE`（CPU）

计算形式 `prediction = b + Σ_j f_j(x_j)`。用 XGBoost singleton interaction
constraints：`interaction_constraints=[[0],[1],...,[d-1]]`（以 feature name 形式，
DMatrix 列名固定为 `c0..c{d-1}`）。不把 `max_depth=1` 当作替代，允许单特征多阈值。

* `H-ADD-TREE`：除约束外完整沿用 XGB-ALL 配方
  （binary:logistic, 500 rounds, depth 5, eta 0.03, subsample 0.8,
  colsample 0.8, lambda 5, min_child_weight 1, hist, seed 0, 普通无加权监督）。
* `Z-ADD-TREE`：除约束外沿用 ZX1 配方（`reg:absoluteerror` + 同一组超参，500 rounds）。

核验：训练前合成数据检查约束被当前 xgboost 实现；训练后解析所有树，每棵树不同
特征数 ≤1，size 特征与结构特征、两个不同结构特征均不共树；手工重放树输出与
booster raw prediction 一致。若约束不能实现 → `ADDITIVE_TREE_BLOCKED`，不得替代。

固定报告 rounds 100/250/500 的 Train/Valid/Test。历史 XGB-ALL / ZX1 booster 可读，
则用 `iteration_range` 截取同一训练过程的 100/250 轮，不重训；否则只用冻结 500 轮。
`CAPACITY_CONFOUNDED` 预注册判据：加性树的 Train 指标劣于完整树的幅度
> 完整树自身的 Train−Valid 差距（分类用 AUC；回归用 MAE，方向相反）时标记，
表示不能把差距完全归因于缺少交互。

## 6. Phase C — 神经读出（GPU，seed 0）

统一配方（沿用 SGA-v1）：epochs 50, batch 256, AdamW lr 1e-3, weight decay 1e-4,
grad clip 5, 无 scheduler/early stopping, 每 epoch 完整评价三划分,
HIV 选最高 Valid AUC、ZINC 选最低 Valid MAE，平局取较早 epoch。
输入：HIV 大小项 Train-only 标准化（clip ±10），结构计数保持冻结缩放；
ZINC 严格沿用 ZN1 的实际预处理（`Z` 的 n_atoms/n_bonds Train-only 标准化、
整块 clip ±10；`C/F` 不标准化）。Test 只在评价中使用。

* `H-NN-JOINT`：`Linear(4408,128)→ReLU→Dropout(.1)→Linear(128,128)→ReLU→Dropout(.1)→Linear(128,1)`，普通 BCE。
* `H-NN-JOINT-BAL`：同结构、同初始化、同 batch 顺序，唯一变化是 Train-only class-balanced BCE。
  初始化 = `torch.manual_seed(0)` 后按同一构造顺序；两臂初始化逐参数哈希必须一致。
* `H-NN-ADD` / `Z-NN-ADD`：每特征独立
  `f_j(x) = a_j x + Σ_{k=1..4} v_jk (ReLU(x−t_jk) − ReLU(−t_jk))`，
  `prediction = b + Σ_j f_j(x_j)`；`f_j(0)=0` 恒成立。折点初值 = 该列 Train 非零值
  25/50/75/90 分位；无非零值的列 a=v=0、贡献恒 0；a、v、b 为确定性 N(0,0.01) 初始化。
  HIV 用普通 BCE（与 H-NN-JOINT 一致），ZINC 用 Train 标准化 target 的 L1、原始单位评价。
  训练时对每个单特征输出做 Dropout(0.1)，评价关闭（与普通 MLP 的 hidden dropout 位置不同）。
  折点参数不施加 weight decay，其余预测参数按统一 weight decay。
  ZINC 普通 MLP 直接复用 ZN1（不重训）。

参数量、初始化哈希、完整曲线、逐分子预测、checkpoint 全部保存。
不新增 GNN、不搜索宽度/深度/学习率、不增加 seed。

## 7. Phase D — 统一比较与统计

主比较（HIV）：正则化 vs 历史 COUNT-ALL；L-NAT-STD vs L-NAT-RAW；
XGB-ALL vs H-ADD-TREE；H-NN-JOINT vs H-NN-ADD；H-ADD-TREE vs H-NN-ADD；
XGB-ALL vs H-NN-JOINT；H-NN-JOINT vs H-NN-JOINT-BAL。
ZINC：ZX1 vs Z-ADD-TREE；ZN1 vs Z-NN-ADD；Z-ADD-TREE vs Z-NN-ADD。
指标：HIV ROC-AUC + AP；ZINC 原始单位 MAE。

配对 bootstrap 1000 次，seed 0：HIV 按 scaffold 重采样（Valid/Test 分别）；
ZINC 按分子重采样。统一正增益方向：HIV `AUC(A)−AUC(B)`；ZINC `MAE(B)−MAE(A)`。
AP 同时报告主要配对差值。所有指标由存储预测独立重算。
bootstrap 只描述固定预测的样本不确定性，不代表跨 seed 稳定性；不显著 ≠ 等价。

## 8. Phase E — 函数提取与局部解释（不新增训练）

ADD-TREE：每棵树只用一个特征，合并同特征树并做零基准
`f̃_j(x) = f_j(x) − f_j(0)`；大小项以 Train 中位数为基准。核验
`prediction = adjusted_bias + Σ_j f̃_j(x_j)` 与原始 raw prediction 一致。
ADD-NN：直接读取 `f_j` 并核验与模型前向一致。两者使用同一横坐标网格与参考点。
HIV 选 6–8 个结构：Train 高支持且函数变化大、含 COARSE/FULL 对应关系、
1–2 个两模型函数差异明显；选择只用 Train 支持与训练后函数，不用 Test 表现。
局部解释用较好的加性模型，贡献必须精确重建预测；展示 3 个 Valid 真实分子
（正确正例、正确负例、最大误差失败例）。ZINC 只展示少量特征；未持久化 canonical
key 的结构只报告可核验的列身份/族，并明确不可直接还原为真实子图。

## 9. 评价、核验与停止

必核验：历史/新增输入列顺序、graph id、split 一致；线性参数与贡献公式；
ADD-TREE 单特征；ADD-NN `f_j(0)=0` 与闭合；两个 HIV MLP 初始化/顺序一致；
指标从存储预测独立重算。失败项写入 `analysis/verification.json`。

执行优先级：Phase 0 → A → H-ADD-TREE → HIV 神经 → ZINC 树/神经 → E/报告。
预算不足时先完成 HIV 完整对照。禁止：重训历史模型、重挖子图、扩大词表、
扩大超参/seed/数据集、新 GNN、依据 Test 追加实验。
