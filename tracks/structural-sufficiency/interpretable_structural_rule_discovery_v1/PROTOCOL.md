# PROTOCOL — 显式结构预测充分性与规则语义修复 v1

Round: `interpretable-structural-rule-discovery-v1`
Directory: `tracks/structural-sufficiency/interpretable_structural_rule_discovery_v1/`
Seed: `0`（单 seed，不做多 seed）
Predecessors（只读）：`interpretable_structural_rule_discovery_v0`（ISRD-v0，冻结 commit `6cddf702`）、
`structural_gain_attribution_v1`（SGA-v1）、`molhiv_multigranular_structure_sharing_v0`（MSDS-v0）。
Compute: CPU-only。目标 60 分钟，硬上限 120 分钟。

本文件在任何 v1 规则模型拟合**之前**写入并提交。规则预算、特征、划分、评价、
选择方法全部固定。已知实现错误只能按 §3 修复并记入 `ERRATA.md` /
`analysis/provenance.json`；看到 Test 数字后不允许改设计。

---

## 1. 问题

* Q1（Phase A/B）：修正规则提取语义后，v0 关于阈值规则与联合规则的结论是否改变？
* Q2（Phase C1）：规则模型的性能损失主要来自阈值化，还是过早限制结构数量？
* Q3（Phase C2/C3）：完整 4 406 个显式结构在**线性**与**非线性**读出下分别能达到什么水平？
* Q4（Phase D）：芳香碳—硫家族（COARSE 419；FULL 2078/2525/1788/2275）的细粒度差异，
  在控制粗结构、分子大小与已有预测后是否仍有增量预测关联？

范围外：新 GNN / 新聚合模块、新数据集、新词表、全库子图重枚举、三元及以上规则、
>128 条规则的容量扩张、超参扫描、多 seed、重训任何历史 XGBoost / MLP / GINE、
追加对另外三个 FULL 子型的逐项模型扫描。

证据等级：`EXACT_MATCH`（逐位复用）、`SAME_SPLIT_REFERENCE`（同划分同指标，不同配方）、
`HISTORICAL_REFERENCE`（仅引用数字）。

## 2. 数据、身份与冻结资产

* `ogbg-molhiv` 官方划分：Train / Valid / Test = **32 901 / 4 113 / 4 113**；
  行号 = 全局 0 基 graph id。
* 冻结结构表示（`SGA-v1/bundle/sav1_inputs.npz`，sha256 记录在 provenance）：
  * `C` 754 个 COARSE 类型，`F` 3 652 个 FULL 类型，共 4 406 个显式结构列；
  * `x_s(G) = count_s(G) / sqrt(max(n_atoms(G), 1))`（float32 冻结值）；
  * 列顺序 = 冻结词表顺序（`SGA-v1/analysis/registry_v1/`）。
* 结构描述与属性使用 `registry_v1` 的 `structure_registry_{coarse,full}_v1.csv`
  （含 `occurrence_train`、`document_frequency_train`、节点/键属性序列）。
* **原始整数计数**优先取 `MSDS-v0/data/phaseA_instances.npz` 的实例缓存
  （`full_indices` / `coarse_indices` + `registry.discovered_type_id` 映射）；
  由缩放值还原时核验 `count == x_s * sqrt(n_atoms)`（按 float32 容差取整）。
* 历史预测（逐分子 logit）：`MSDS-v0/data/phaseB_predictions.npz`（B0/B2/B3），
  `SGA-v1/analysis/predictions/{X-C,X-F,X-CF}.npz`。
* X-CF booster **未保存**，按本轮规则不重训：它只作为性能位置参照。
* Test 政策：`DEVELOPMENT_TEST` — 全部划分都评价并报告，但 Test 从不用于选择规则、
  阈值、容量、C 或任何设计元素。

## 3. Phase A — 规则语义修复（针对 v0 的已知错误）

### 3.1 A1 arm-specific 特征映射

历史 booster 的真实输入布局由 booster 自身核验：

| arm | 输入 | 宽度 | 结构列映射 |
|---|---|---:|---|
| X-C | H + C | 3012 | 全局列 `2258+i` ↔ `C[i]` |
| X-F | H + F | 5910 | 全局列 `2258+i` ↔ `F[i]` |

`H_WIDTH = 2258 = 2048 ECFP4 + 210 RDKit 描述符`。实现为

```python
def decode_feature(arm, feature_idx):
    width = ARM_WIDTH[arm]                    # X-C: 3012, X-F: 5910
    if not (0 <= feature_idx < width): raise ValueError(...)
    if feature_idx < H_WIDTH: return ("H", feature_idx)
    if arm == "X-C": return ("C", feature_idx - H_WIDTH)
    if arm == "X-F": return ("F", feature_idx - H_WIDTH)
    raise ValueError(arm)
```

不允许按全局列号区间猜粒度。核验（必须全部通过）：

1. X-C 的全部结构分裂映射到 C；X-F 的全部结构分裂映射到 F；
2. 映射后的结构列值 == 原始 booster 实际输入列值（按 arm 重建 `H+block` 设计逐列比对）；
3. `X-F:f2258` ↔ `F[0]`；
4. 每个 booster 的 `num_features` == `H_WIDTH + block 宽度`；
5. 冻结设计无 NaN（缺失分支从未被走过）；仍记录每个节点的 missing 方向。

### 3.2 A2 真实路径条件

逐节点递归读取（不再用 `global_feature -> 单一阈值` 反查）：

* 来源 arm、tree id、node id、原始 feature index、映射后的 (granularity, column)、
  实际 split threshold（`float32` 语义）、实际经过分支方向、missing 分支语义；
* 非缺失值语义：`yes` 子节点 ⇒ `x < t`；`no` 子节点 ⇒ `x >= t`；
* 每个叶节点保存真实 root-to-leaf 条件列表。

条件身份 = `(granularity, column, float32(threshold), direction)`。同一结构不同阈值、
不同方向都是不同条件，分别保留。

### 3.3 A3 单规则候选、去重与联合规则来源

* 单规则候选 = 所有真实出现在树节点上的**带方向**条件（两个方向都作为候选）。
  双侧 Train 支持门槛不变：每侧 ≥30 个 Train 分子、≥5 个 Train scaffold，且非常数。
* 按 Train 激活向量去重是允许的，但必须保留全部来源（arm/tree/node、条件、被合并的条件）。
  去重只说明 Train 上激活相同，不称为逻辑等价。
* 联合规则 = 同一真实路径上两个**不同结构**的实际带方向条件，标记 `PROJECTED_PAIR`；
  保留两个条件的阈值与方向；不使用叶节点权重作为规则系数；忽略同路径的 H 条件
  （不可投影为结构规则），因此候选不是原树规则。
* 来源约束：X-C 只产生 C–C，X-F 只产生 F–F。**真实路径 C–F 组合数必须为 0**，
  否则视为实现错误并中止（不训练）。
* 单规则池 ≤512，真实联合规则池 ≤256（与 v0 同预算）。联合规则 2×2 四格门槛不变
  （每格 ≥20 分子、≥3 scaffold），排除常数/包含/等价组合，按激活向量去重。

### 3.4 A4 联合规则主效应资格与匹配随机对照

* 真实联合规则与随机联合规则的两个主效应，都必须以**精确条件身份**
  （granularity, column, threshold, direction）存在于共同 U64 单规则模型中；
  允许使用已选单条件的**精确补集**（基础模型含截距），但阈值必须完全一致。
  仅凭 structure column 判断资格不再允许。
* 随机对照池 = U64 已选单条件的**精确条件闭包**（64 条 + 其精确补集，共 128 条；
  之所以需要补集，是因为真实路径条件本身可能就是 U64 成员的补集）。
* 随机组合要求：不同结构；不属于任何真实共同路径组合；与真实规则的
  granularity 组合、结构族组合匹配；同样的四格支持过滤；边际激活频率绝对差 ≤0.05
  （两边际各自），联合频率绝对差 ≤0.03；一对一无放回匹配。
  容差是操作性筛选条件，不为凑数放宽。
* 用共同 U64 Train 残差评分；对每个匹配对取真实/随机两条规则评分的**均值**排序，
  选出最多 16 对。K = min(16, 匹配对数, 新单规则数)；K < 8 时跳过联合训练并标记
  `INSUFFICIENT_MATCHED_PAIR_SUPPORT`。
* 三组增量规则使用同一 K：新增单结构规则 / 真实联合规则 / 匹配随机联合规则。

### 3.5 A5 数值与知识卡片语义

* 规则卡片的 count 优先取原始整数实例计数；否则核验 `count ≈ x_s * sqrt(n_atoms)`。
* “阈值等价于 raw_count ≥ k / 出现”只在 Train 上核验
  `activation == (raw_count >= k)`；若仅在观察数据成立，表述为**经验等价**。
* 结构名称取自 `registry_v1` 的真实拓扑字段（节点/键/度/H/芳香性），不手工命名。
* 报告与卡片从实际结果文件生成。

## 4. Phase B — 重跑受影响的规则实验

模型定义、规则预算、候选筛选、训练方法、C 网格与 v0 完全一致：

```
z(G) = b + a1*log1p(n_atoms) + a2*log1p(n_bonds) + sum_k w_k r_k(G)
```

* 前向选择：Train 加权 logistic 梯度 `score_j = g_j^2 / h_j`，`sw` = class-balanced，
  选择阶段 `C = 0.03`，每批最多 16 条，同一结构最多一个阈值，Train-only 标准化；
* 容量 16 / 64 / 128；最终 `C ∈ {0.03, 0.3}` 由 Valid ROC-AUC 选择（并列取小 C）。

| 模型 | 内容 |
|---|---|
| R0 | 分子大小（若与 v0 配置/输入完全一致则复用并核验重放） |
| R16/R64/R128 | 大小 + 16/64/128 条单规则 |
| B2-L64 | 冻结 B2 + R64 的 64 个结构的连续缩放计数 |
| B2-U64 | 冻结 B2 + R64 的 64 条阈值谓词 |
| B2-U+S / B2-U+P / B2-U+N | B2-U64 + K 条新单规则 / 真实联合 / 匹配随机（条件性） |
| R64+P | 大小 + U64 + 同一 K 条真实联合规则（条件性） |

B2 冻结：使用 MSDS-v0 的 `B2|0.03` 系数与截距重放（离线核对 `max|Δlogit|`）。
所有增量模型只重拟合修正项（offset logistic，`b_delta` 不受惩罚，`C ∈ {0.03,0.3}`）。

v0 受影响的实验在 `analysis/metrics.csv` 中标记 `SUPERSEDED_BY_RULE_MAPPING_FIX`
并保留原记录，不再作为树规则机制的证据。

## 5. Phase C — 三个预测充分性对照

### C1 COUNT128

输入 = `log1p(n_atoms)`、`log1p(n_bonds)` + R128 实际使用的**同一批**结构身份的连续
缩放计数 `x_s(G)`（不是阈值谓词）。与 R128 完全相同的参数数量、Train-only 标准化、
class-balanced 权重、`fit_logistic` 实现、C 网格与 Valid 选择。

### C2 COUNT-ALL

输入 = 同样两个大小项 + 全部 754 个 C + 3 652 个 F 连续计数（无 Morgan / 描述符 /
GNN / latent）。模型是同一个线性 logistic：

```
z(G) = b + a1*z(size1) + a2*z(size2) + sum_s w_s * z(x_s(G))
```

使用稀疏实现，但与稠密标准化后的目标函数**严格等价**：中心化只改变截距，
`z = b + A u`，其中 `A = X/sd`，`b = b' + (mean/sd)·u`；罚项对 `b` 与 `u` 施加，
即 `(b' + m·u)^2/(2C) + ||u||^2/(2C)`。该等价性由单元测试与 `C1` 重放共同核验。
最终把系数/截距转换回原始单位，并验证预测精确闭合。

### C3 XGB-ALL

输入与 COUNT-ALL **相同**（两个大小项 + 全部 C/F 连续计数，dense float32，
零就是零，不当作 missing）。冻结历史 XGBoost 配方：

```
500 trees, max_depth=5, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
reg_lambda=5, min_child_weight=1, tree_method=hist, objective=binary:logistic,
random_state=0, device=cpu
```

不调参。保存 booster（`analysis/models/XGB-ALL.json`）供后续只读解释。

### C4 统一对照表

| 模型 | Morgan/描述符 | 结构数量 | 表示 | 读出 |
|---|---|---:|---|---|
| R128 | 无 | ≤128 | 阈值 | 线性 |
| COUNT128 | 无 | 相同 | 连续计数 | 线性 |
| COUNT-ALL | 无 | 4406 | 连续计数 | 线性 |
| XGB-ALL | 无 | 4406 | 连续计数 | 树模型 |
| B2 | 有 Morgan | 4406 | 连续计数 | 线性 |
| X-CF | 有 H | 4406 | 连续计数 | 树模型 |

后两行直接复用历史结果；它们与新 ALL 模型的基础输入/配方不同，只作性能位置参照。

## 6. Phase D — 芳香碳—硫家族（不搜索新家族）

* COARSE 419；FULL 2078 / 2525 / 1788 / 2275。
* D1：只在含 COARSE 419 的分子中，报告四个 FULL 子型的数量、标签分布、scaffold 支持、
  共现/Jaccard、分子大小、Train 原子数分位数分组内的活性关联、B0/B2 的预测表现。
  不假定子型互斥（实际核验）。
* D2（`CS-BASE` / `CS-FINE`）：拟合与评价都限制在 COARSE 419 阳性的分子子集。
  * `CS-BASE`: 冻结 B0 logit（`B0|0.03`，普通连续输入）、`log1p(n_atoms)`、
    COARSE 419 连续计数；
  * `CS-FINE`: CS-BASE + FULL 2078 是否出现（0/1）。
  相同 Train-only 标准化、class-balanced logistic、固定 `C = 0.03`；Train 子集拟合，
  Valid/Test 子集评价。报告样本/正负例、AUC/AP、2078 系数、配对预测差值与 bootstrap 区间、
  2078 出现/未出现的支持与表现。若子集支持不足或 B0 logit 不可得 → `CONDITIONAL_CHECK_BLOCKED`。
  不追加其他三个子型的逐项模型。
* D3：少量真实分子（2078 支持案例/反例、同 COARSE 不同 FULL、模型计算作用案例），
  核验子图与连接位置；区分「数据关联」「模型内计算作用」「增量预测价值」。

## 7. 指标、不确定性与解释

* 主指标 OGB 官方 ROC-AUC（sklearn 交叉核验）；辅助 Average Precision。
  Train/Valid/Test 全部报告，Test = `DEVELOPMENT_TEST`。
* 每个模型保存完整逐分子 logit、label、graph id、split。
* 核心对照执行 1 000 次配对 **scaffold** bootstrap（seed 0，95% percentile）：
  修正后规则模型主要对照、COUNT128−R128、COUNT-ALL−COUNT128、XGB-ALL−COUNT-ALL、
  B2−COUNT-ALL、X-CF−XGB-ALL、CS-FINE−CS-BASE（若执行）。Valid/Test 都报方向与区间。
  不显著 ≠ 等价；单 seed bootstrap 不等于跨 seed 稳定性。
* COUNT-ALL 局部计算解释：每分子实际结构项数、每项 logit 贡献、
  |贡献| 前 5/10/20 项占全部结构项 |贡献| 总量比例、剩余贡献与重放误差、
  重复结构与重叠结构的解释边界。完整预测必须由全部结构项 + 大小项精确重建；
  前 K 项只是局部摘要，必须显示未展示的剩余贡献。不把集中度当作主要预测指标。

## 8. 预算与停止规则

CPU-only，目标 60 分钟，硬上限 120 分钟。执行优先级：

1. A1/A2 映射与路径语义修复 + 针对错误的测试；
2. 重跑受影响单规则实验（R16/R64/R128、B2-L64/B2-U64）；
3. COUNT128 / COUNT-ALL / XGB-ALL；
4. 条件性联合对照（严格匹配才执行）；
5. C–S 家族分析 + 汇报。

资源不足时优先完成三个充分性对照，不为勉强执行联合规则牺牲匹配质量。

## 9. 必要检查（必须真正覆盖已知错误）

1. 按 arm 的结构列映射（含 X-F `f2258` 边界列）；
2. 映射后的结构输入值 == booster 原始特征列值（直接比列值，不只比预测/叶节点）；
3. 同一特征多个阈值不被覆盖（逐条件保留）；
4. 每条真实路径的阈值、分支方向、来源节点一致（按实际设计逐节点重放）；
5. 单粒度 booster 不产生 C–F 真实路径；
6. 真实与随机联合规则具有相同的精确主效应资格；
7. 原始计数与归一化语义正确（`count == x*sqrt(n_atoms)`，`activation == raw_count>=k` 经验核验）；
8. B2/B0 历史预测重放；
9. R128 与 COUNT128 结构身份一致；
10. COUNT-ALL 与 XGB-ALL 输入身份一致；
11. 可解释线性预测的原始单位加和闭合；
12. 标准评价指标独立重算；
13. 稀疏 offset/中心化 logistic 与稠密 `fit_logistic` 等价。

## 10. 偏离日志

执行中只允许修复真实工程错误，写入 `analysis/provenance.json` 的
`engineering_deviations`，并说明受影响的运行与是否改变已报告数字。
v0 已知错误的影响范围写入 `ERRATA.md`。
