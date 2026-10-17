# PROTOCOL — explicit-composition-signal-audit-v1

- protocol_id: `explicit-composition-signal-audit-v1`
- track: `structure-contribution`
- round: `explicit_composition_signal_audit_v1`
- freeze rule: **Stage A 的结构定义、匹配集合与分析代码必须先写成哈希清单
  `FREEZE.json`；Stage B 只有在逐项哈希核对通过后才允许读取 official-train 的 `y`。**
- 本协议在首次标签读取之前冻结；Stage B 阶段不得修改 B0/B1/C 定义、匹配层级、
  近似匹配名单、统计口径或 Stage C 门槛。

---

## 0. 研究问题与边界

**H**：与只用结构单元内容和整体统计的模型相比，显式保留结构之间的真实连接组织，
对 ZINC 预测包含可学习、可泛化的增量信息。

本协议只回答“ZINC-12k official train 是否具备识别该增量的数据条件与标签条件”，
不训练新架构、不调参、不读 official valid/test、不读旧 Locked 标签、不读任何模型预测。
它把四件事分开：

1. **信息存在**：C 是否在给定数据中与 B0/B1 可分离；
2. **标签相关**：同 bag 不同 C 的样本是否有不同的 `y`；
3. **可学习性**：只在 Stage C 门槛通过时才做一个冻结的 CPU 探针；
4. **泛化收益**：本轮不做，也不由前三项直接推导。

---

## 1. 数据与完整性

只使用官方 `ZINC(subset=True, split="train")` 的结构与 `y`。

| 输入 | 路径 | SHA-256（运行时核对） |
|---|---|---|
| canonical SMILES | `tracks/ksvd/results/zinc_cssd_basis_reuse_v1/train_canonical_smiles.npz` | `04a59035…dcaff` |
| official PyG train | `data/ZINC/subset/processed/train.pt` | `e0f1837b…6955b` |
| SA scorer | `tracks/ksvd/results/zinc_long_cycle_audit/upstream/sascorer_gvae.py` | 运行时记录 |
| SA 特征 | `tracks/ksvd/results/zinc_long_cycle_audit/upstream/fpscores.pkl.gz` | 运行时记录 |

- 冻结排除行：`941, 2210, 6071, 9766`。这 4 行的提交 SMILES 与官方 PyG 图只在
  构型/立体层面不一致（上游按拓扑匹配造成的 constitutional isomer）；在读取任何
  `y` 之前排除。Stage A 用全量 10,000 行重跑配对核对，必须逐位复现该清单。
- 结构分析在**去立体化学的 constitution** 层面进行；`C` 不区分立体异构体。
  这是本协议的明确边界，不能把结论外推到立体化学。
- Stage A 的 `y` 隔离：结构加载器在返回任何记录前 `pop('y')`；Stage A 不导入
  标签加载函数；Stage A summary 写入 `labels_loaded=false`。
- Stage B 只从 `train.pt` 的 `blob[0]['y']` 读取 official-train 标签，并与冻结的
  `encoded_train.pt` 逐行交叉核对（若缓存存在）；不打开 official valid/test。

---

## 2. 三类信息（Stage A，标签盲）

### 2.1 B0 — 粗结构边际信息

对每个分子（去立体化学）定义：

- `n_heavy`：重原子数；`n_bonds`：键数。
- **atom-class counts**：对每个重原子取
  `(元素符号, 形式电荷, 总氢数 GetTotalNumHs(), 是否芳香)`，统计多重集。
  这是 `structure_context_support_v1` 的 T1 原子属性。
- **bond-class counts**：对每根键取 `SINGLE/DOUBLE/TRIPLE/AROMATIC`
  （`bond_order_code`，芳香 = 12），统计多重集。
- **Bond line-graph degree distribution**：对每根键 `(u,v)`，
  `d_L = deg(u) + deg(v) - 2`（与该键共享端点的其它键数）；统计所有键的 `d_L`
  的多重集。
- **junction-count distribution**：junction atom 定义为重原子度 `>= 3`；对每根键
  `(u,v)`，`j = 1[deg(u)>=3] + 1[deg(v)>=3]`；统计所有键的 `j` 多重集。
- `n_junctions`：度 `>= 3` 的重原子数。

B0 签名是上述字段的元组。B0 **不包含**环数；环数作为单独的更严格控制层
（`M0_ring`）使用。B0 包含原子类别计数，因此已经控制重原子数、元素组成、
氢/电荷/芳香状态与键类别组成；这是刻意的强边际控制。

### 2.2 B1 — 结构单元多重集

- 对分子中每一根真实键 `(u,v)`，取**以该键为锚的半径 1 带属性 patch**：
  顶点 = `{u,v}` 及其所有直接邻居；边 = patch 内所有真实键；顶点属性 = T1
  原子属性；边属性 = 键类别；锚点 = `{u,v}`（无序）。
- 用 `attributed_canon.canonical_form` 计算该带锚点属性图的**精确规范键** `S`。
  不使用 WL hash 作为最终身份，避免把不同局部结构合并。
- B1 签名 = 该分子所有键的 `S` 的排序多重集（不是集合；重复出现保留重数）。
- B1 保留局部内容（原子/键属性、局部连接），但**不直接保留**哪个 S 与哪个 S
  在全局上相连、共享哪个端点。它是“结构单元内容”的强 bag。
- 同时计算 T2（元素 + 单/多键）和 T3（元素+电荷+芳香，无氢计数）的 B1，
  仅作为敏感性/近似匹配口径，不替代 T1 主口径。

### 2.3 C — 真实结构组合信息

- 结构单元 = B1 中的每个 `S` 出现（每根键一个单元）。
- 原子 = 每个重原子，带 T1 原子属性。
- 构造**单元–原子关联图**：
  - 单元节点颜色 `("U", S)`；
  - 原子节点颜色 `("A", atom_class)`；
  - 对每个单元 patch 中出现的每个原子，加入一条关联边；边通过一个细分的
    incidence 节点表示，颜色为 `("R", "anchor")`（该原子是锚键端点）或
    `("R", "neighbor")`（该原子是锚点邻居）。
- 用 `pynauty` 的 colored-graph certificate + canonical colour sequence 计算
  **精确规范键** `C_key`。该构造对原子重编号、键枚举顺序不变。
- 独立计算 `C_smiles` = RDKit 非立体 canonical SMILES（constitution）。两者
  诱导的等价类必须在全量 9,996 行上**逐行一致**：
  - `C_key` 同 → `C_smiles` 同（完备性）；
  - `C_smiles` 同 → `C_key` 同（不变性/无假分裂）。
  若不一致，Stage A 直接失败，不进入 Stage B。
- `C` 的同一性以 `C_smiles` 作为分组 id（精确、可读），`C_key` 作为构造性
  验证；两者在数据上等价。

---

## 3. 匹配层级（Stage A 冻结）

记“C-different”为两个分子的 `C_smiles` 不同（即非同构 constitution）。
一个 bag 等价类是一组共享指定签名的分子；**pair 不是独立样本**。

### 3.1 严格层级（主分析）

| 层级 | 定义 |
|---|---|
| `M01` | 同 B0 **且** 同 B1(T1)，不同 C —— 最严格的 bag |
| `M0` | 同 B0，不同 C —— 严格边际控制；B1 可不同 |
| `M0_ring` | 同 B0 且同环数，不同 C |
| `B1` | 同 B1(T1)，不同 C —— 只看局部 bag |

主结论以 `M01` 为首要；`M0` 是“严格边际、但局部内容可不同”的补充；
`B1` 回答“局部 bag 是否已经决定 C”。

### 3.2 预注册的近似匹配阶梯（探索性，明确分开）

当严格匹配稀少时，按下列**预先定义**的阶梯报告；所有层级都报告，不做事后挑选：

| 层级 | 放松了什么 | 仍控制什么 |
|---|---|---|
| `A1` | B0 的 line-graph degree 与 junction-count 分布 | 重原子数、键数、原子类别计数、键类别计数、环数 |
| `A3` | 原子类别中的 H/电荷/芳香细节 | 重原子数、键数、元素计数、键类别、环数、line-graph degree、junction counts |
| `A4` | 原子类别细节 **且** degree/junction 分布 | 重原子数、键数、元素计数、键类别、环数 |
| `L_old` | 历史的 bond-RW 粗边际签名 | 键数、键类别、line-graph degree、junction counts；**不控制元素/原子类别** |

- `A1` 是主近似层级（仍精确控制分子式/原子类别组成与环数，只放松两个边际分布）。
- `A3` 是组成细节放松的敏感性层级。
- `A4` 样本量更大，但明确是**混杂的**：它允许元素组成相同而 H/电荷/芳香与
  degree/junction 不同；只能用于估计公式主导程度，不能用于声称 C 的独立效应。
- `L_old` 复现历史 bond-RW 的粗边际签名，用于量化“旧 208 对”为什么不等于
  完整 10k 上的组成控制结论；它不控制元素/原子类别，不能作为 H 的证据。

### 3.3 计数单位与独立性

对每个层级报告：

- `n_groups`（bag 等价类数）、`n_molecules`、`n_scaffolds`（Murcko）、
  `n_topologies`（全碳骨架）；
- `n_diffC_groups`、`n_molecules_in_diffC_groups`、
  `n_scaffolds_in_diffC_groups`；
- `n_pairs_diffC`（组内跨 C 无序对的闭式计数
  `(n² − Σ_c n_c²)/2`）、`n_scaffold_disjoint_pairs_diffC`；
- `max_group_size`、`max_diffC_group_size`、组大小直方图。

**禁止**把所有 pair 当作独立样本。Stage B 的置信区间按 **group 聚类 bootstrap**
（主）报告；scaffold 只作描述性独立单位与 leave-one-scaffold-out 稳健性检查。
同一组内多个 pair 共享分子，不能视为独立复制。

### 3.4 置换不变性与索引对齐单测

Stage A 必须通过：

1. 至少 200 个分子的 3 次随机原子重编号：B0/B1(T1/T2/T3)/C_key/C_smiles 全部不变；
2. 全量 `C_key` 与 `C_smiles` 的等价类一一对应；
3. 每个分子的 S patch 并集覆盖全部真实键；
4. 用“访问 y 即抛异常”的假张量验证结构加载器不触碰 `y`；
5. 固定小分子单测：B0 边际定义、C 规范键的同构/非同构反例、
   `M0`/`M01` 对真实碰撞对的分组行为、pair 闭式计数不重复计同 C 对。

---

## 4. Stage B — 冻结后只读 official-train `y`

Stage B 入口先重算 `FREEZE.json` 中 protocol/code/input/Stage-A artifact 的
SHA-256；任何一项不匹配即中止。然后：

### 4.1 严格 bag 的标签差异

- 对 `M01`、`M0`、`M0_ring`、`B1` 的 C-different pairs，报告 `Δy = y_b − y_a`
  的均值/中位数/std/mean|Δy|/分位数/符号比例；
- 若 `M01` 为空，明确报告“不存在严格同 B0+同 B1 的 C-different 样本”，
  不把 `M0` 的结果冒充为 `M01`；
- 所有 CI 使用 group-cluster percentile bootstrap（B=5000，seed 20261016）；
  组数 <2 时报告无 CI。

### 4.2 最优常数预测器的经验 MAE 下界

对每个 bag 层级，组内用该组 `y` 的中位数预测，计算
`mean_i |y_i − median(y_group)|`。报告：

- 在 **C-different matched subset** 上（只含 `diffC_groups`）；
- 在 **duplicate subset** 上（同 C 的重复行；只对 B1/B0 等有重复组的层级）；
- 在 **all multi groups** 上；
- 在 **整个训练集** 上（singleton 组贡献 0，分母 9,996）。

同时报告“处于 multi group 的分子比例”。明确声明：这是**组内 in-sample**
下界；singleton bag 使其接近 0，**不是**泛化下界，也**不是** BJOC-P3 的误差下界。

### 4.3 penalized-logP 目标公式分解

沿用 `bjoc_mmp_feasibility_v1` 的只读实现与常数：

```
y = (logP - μ_logP)/σ_logP
  + (-SA - μ_SA)/σ_SA
  + (-cycle_penalty - μ_cycle)/σ_cycle
```

- 主常数：community JTNN-era 值；敏感性：dataset 值。
- 对每个 pair 计算 `ΔlogP_scaled`、`ΔSAneg_scaled`、`Δcyclepen_scaled`、
  `Δformula`、`residual = Δy − Δformula`。
- 报告 formula vs Δy 的 `R²`、corr、带截距 OLS 系数（intercept, logP, SA, cycle）、
  residual 的均值/中位数/mean|·|/分位数/max。
- 在全部 9,996 行上重算逐分子公式重构残差，定义
  `noise_floor = max(0.01, 2 × median|recon|)`；对每个 pair 再计算
  `pair_recon_noise = |recon_a| + |recon_b|`，标记
  `|residual| > max(noise_floor, pair_recon_noise)` 的 pair。
- 明确：公式审计是 near-oracle 诊断，不是学习结果；公式 R² 高不等于
  “结构组合无信息”，但把可测的 beyond-formula 信号上界压到 residual 量级。

### 4.4 环基不稳定性

对每个进入 pair 的分子：

- 用目标公式相同的 `networkx.cycle_basis` 计算默认 `cycle_penalty`；
- 在 16 个随机原子重编号下重算 `cycle_basis` 的 `cycle_penalty`；
- 另算 `minimum_cycle_basis` 与 RDKit `GetSymmSSSR` 的环大小惩罚；
- 若上述任一给出不同的 penalty，标记 `ring_basis_unstable=true`。
- 报告不稳定分子数/比例、不稳定 pair 数/比例，以及 stable/unstable 两组的
  residual 统计。明确：若目标公式的环惩罚依赖环基选择，`Δy` 分解本身有
  实现依赖成分，相关 pair 的结论必须降级。

### 4.5 主导性 / 异常值

对每个有 pair 的层级报告：

- leave-one-group-out 与 leave-one-scaffold-out 对 `mean|Δy|`、`mean|residual|`
  的最大改变量及对应 group/scaffold；
- 去掉 top-1 / top-5% / top-10% `|Δy|` 或 `|residual|` 后的剩余均值；
- `|y|>3` 的 tail pair 数量、其 `|Δy|`/`|residual|` 占总量的份额；
- `|Δy|>5` 的极端 pair 数量；
- 若 `n_pairs<20`，标记 `underpowered_n_lt_20=true`，只做枚举式描述，不做
  显著性宣称。

### 4.6 重复行标签噪声

`B1` 中同 C（同 constitution）的重复组只能来自表中重复分子；它们的 `Δy`
是**重复标签噪声**，不是组合信号。报告组数、分子数、pair 数、`Δy` 分布和
每组明细。该噪声是后续结构信号可识别性的经验下限之一。

---

## 5. Stage C 门槛（条件执行；本轮预先冻结）

只有以下 **G1 ∧ G2 ∧ G3** 全部为真，才允许运行 Stage C 的 CPU 学习探针；
否则写 `stage_c_gate.json` 并停止，不训练、不使用 GPU。

- **G1 严格样本量**：`M01` 至少
  `n_groups ≥ 30`、`n_molecules ≥ 60`、`n_scaffolds ≥ 20`、
  `n_scaffold_disjoint_pairs ≥ 30`。
- **G2 非公式信号**：`M01` 的 `median|residual| ≥ 0.02`，
  `formula R² ≤ 0.95`，且 `mean|residual|` 的 group-cluster 95% CI 下界 > 0。
- **G3 公平对照**：至少存在一个 `B1` 组含 ≥2 个不同 C（即 B1 不是 C 的
  完全函数），从而可以构造“Control 只有 B1 内容、Real 额外有 C”的对照，
  而不必退回到受边际混杂的 B0-only。若 B1 在所有不同 constitution 上
  injective（`B1` 的 C-different 组 = 0），则 Control 已隐含 C，G3 失败。

若 G1–G3 通过，Stage C 的冻结设计（仅作预先声明，不在本轮自动执行）：

- 数据：同一 9,996 行；Murcko scaffold-grouped 5 折，折分配由固定 salt 的
  SHA-256 排序决定；
- 特征：Control = B0 显式向量 + 训练折内选出的 B1 top-K（K=256）计数 +
  bag 大小/熵/最大重数；Real = Control + 预定义的单元图关系摘要（共享端点角色、
  单元度分布、连通分量、环秩等）；Null = Control + 在同分子内对关系标签做
  保计数置换（多个固定置换 seed），维度与 Real 相同；
- 学习器：标准化后的 ridge，固定 alpha=1.0；三臂同维度、同折、同预处理；
- 指标：OOF MAE/RMSE、ordinary（`|y|≤3`）/tail（`|y|>3`）、有效样本数、
  scaffold-cluster bootstrap 95% CI、Real−Control 与 Null−Control 配对差；
- 禁止：超参数搜索、事后改关系定义、用 official valid/test 选择。

Stage C 只是预测信息筛选，不是 BJOC 模型的等性能比较；它的失败不能证明
复杂组合机制普遍无效。

---

## 6. 决策映射（预注册）

- `EXPLORATORY_SIGNAL_PRESENT`：只有在 **M01** 达到 G1 且 G2 的非公式
  residual 信号成立，并且 G3 公平对照可构造时；才建议下一轮架构实验。
  近似层级（A1/A3/A4/L_old）永远不能单独触发此标签。
- `INSUFFICIENT_OR_CONFOUNDED`：严格匹配不足，或 B1 对 C injective 导致
  无法分离，或公式 R² ≥ 0.99 / residual 在重构噪声内，或主要信号由
  `L_old` 的组成混杂解释。建议换任务/数据，而不是继续在 ZINC 上堆组合机制。
- `INCONCLUSIVE`：仅当严格匹配不足，但近似层级既没有清楚的正信号，也没有
  清楚的公式主导/混杂解释；不得解释为整个非 GNN 方向失败。

报告必须同时给出“最强反对证据”，并明确区分：信息存在 ≠ 标签相关 ≠ 可学习 ≠
泛化收益。

---

## 7. 复现

```bash
# 1) 结构阶段（不读 y），写出 analysis/ 与 FREEZE.json
uv run python tracks/structure-contribution/results/explicit_composition_signal_audit_v1/code/run_explicit_composition_signal_audit_v1.py stage-a

# 2) 单测
uv run pytest -q tracks/structure-contribution/results/explicit_composition_signal_audit_v1/tests

# 3) 标签阶段（先 verify freeze，再读 official-train y）
uv run python tracks/structure-contribution/results/explicit_composition_signal_audit_v1/code/run_explicit_composition_signal_audit_v1.py stage-b

# 4) 只核 freeze
uv run python tracks/structure-contribution/results/explicit_composition_signal_audit_v1/code/run_explicit_composition_signal_audit_v1.py verify-freeze
```

Stage A/B 的产物路径、哈希和运行环境写入 `FREEZE.json` 与
`analysis/stage_a_summary.json` / `analysis/stage_b_labels.json`。
