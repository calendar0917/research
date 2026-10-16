# PROTOCOL — structure-context-support-v1

**轨道**: `tracks/structure-contribution/`
**阶段**: Stage A（数据可行性审计，不是建模轮）
**性质**: 预先注册（pre-registered）分析协议。所有结构身份口径、上下文口径、统计量、
判断阈值与判断规则在本文件中冻结；`REPORT.md` 只允许报告本协议声明的统计量并据其判断，
不允许事后更换定义以制造正结论。
**标签边界**: `labels_loaded=false` —— 本轮不读取任何性质标签 `y`、不训练/调用任何模型、
不使用 embedding/归因，不使用 GPU。

## 0. 本轮问题

同一种具有明确结构—属性身份（element/charge/aromaticity/bond order 等真实化学属性）的
**局部模式** `S`，能否

1. 在不同的真实分子中重复出现（跨图复用，而非同一分子内的重复计数）；
2. 在这些分子中拥有**足够多样的真实结构关系上下文** `C`（局部子图之外的结构信息），
   并且这种多样性不是由分子尺寸/元素组成/同系物重复造成的平凡差异？

本轮只回答"数据条件是否具备"，不回答"上下文是否改善预测"，不做 GNN/Transformer。

## 1. 数据与边界

| 角色 | 输入 | 哈希（运行前运行时校验） |
| --- | --- | --- |
| 主数据集（结构） | `tracks/ksvd/results/zinc_cssd_basis_reuse_v1/train_canonical_smiles.npz`（官方 ZINC `subset=True, split="train"` 的 10 000 条 canonical SMILES） | `sha256=04a590355fb7bba71e35e52bed60c974999b92af9e96e4ee99bed4afe42dcaff` |
| 主数据集（配对核对） | `data/ZINC/subset/processed/train.pt`（官方 PyG collated train blob） | `sha256=e0f1837b4a19714fcf3a5c6727a30c23f57c3398c8987cab2983214a0e76955b` |
| 辅助数据集 | `tracks/structure-contribution/results/bjoc_cross_task_lipophilicity_v1/objects_train.pt`（冻结的 Lipophilicity train-only 结构对象，3 360 条，无 `y` 字段） | `sha256=6d681cb0eaaab8c448564993c3740b24204e52e9e430f8c366189aa0061c7c86` |
| 辅助数据集划分 | `.../split_lipo_v1.json` | 仅用于确认 train-only 身份 |

**允许读取**：SMILES、原子/键的化学属性、分子行号。
**禁止读取**：任何性质标签（ZINC 的 `y`、Lipophilicity 的 `y`）、official valid/test、
Locked split、ZINC-Full、任何已有模型的预测/embedding/归因，任何训练与 GPU。
Lipophilicity 辅助集在 `bjoc_cross_task_lipophilicity_v1` 生成阶段已被冻结为
train-only 且剥离标签；本轮对其做**结构性**检查，不做任何跨任务性质分析。

### 1.1 排除行

沿用 `bjoc_mmp_feasibility_v1` 审计的 4 行排除清单
`EXCLUDED_ZINC_ROWS = (941, 2210, 6071, 9766)`：官方 PyG 图与所提交 SMILES 表在这些行上
互为构成异构（上游立体化学层面的差异，冻结于 MMP 审计）。
本轮运行时**独立复现**该清单：把 SMILES 用 RDKit 重建为 Kekulé 图，与官方
`train.pt` 逐行按 constitution 级别（`useChiral=False`）比较，要求 mismatch 行集合恰好
等于该清单（`analysis/data_integrity.json::pairing_verification`）。
配对所有 10 000 行、不随 `--limit` 缩减；不匹配行不参与任何结构统计。

### 1.2 允许的数据版本差异

ZINC 的 SMILES 表（KSVD 轮提交）与 PyG 图在构成层面一致（除 1.1 的 4 行），
但 **hydrogens / 形式电荷细节**以 SMILES 表的 RDKit 解析为准；因此 §3.1 的
"总氢数"属性在 ZINC 上可能与 PyG 原始 atom label 不完全同义。T3 口径提供了不含
氢计数的最小属性集作为敏感性检查（§3.1）。

## 2. 复现命令

```bash
uv run --group lipo python \
  tracks/structure-contribution/results/structure_context_support_v1/code/structure_context_support_v1.py
```

可选参数（仅用于工程调试，不改变已冻结定义）：
`--limit N`（前 N 行子集冒烟）、`--no-instances`、`--skip-spotcheck`、`--out-dir DIR`。

输出（全部落在本目录，保持精简）：

```text
results/structure_context_support_v1/
  REPORT.md                     # 中文结论 + 主要统计 + 少量化学案例
  PROTOCOL.md                   # 本文件
  summary.json                  # 机器可读的全部统计与判断
  analysis/data_integrity.json  # 哈希 + 配对核对（标签隔离证据）
  analysis/types_zinc_T1_primary.json 等  # 各（数据集×口径×tier）的类型级表
  analysis/examples.json        # §5.4 的 5 类化学案例
  analysis/instances.npz        # 压缩实例清单（S/CT/骨架/行号）
  code/                         # 分析代码
  code/attributed_canon.py      # 带属性锚定图的精确同构规范化
  tests/                        # 正确性测试
```

`summary.json` 顶层记录 `labels_loaded=false`、输入哈希、口径版本与运行时间。

## 3. 冻结定义

### 3.1 局部结构身份 `S`（主口径 T1）

对分子图中**每一条真实化学键** `{u, v}`：

- 取诱导子图 `N[u] ∪ N[v]`（两端锚点 `u, v` 及其一阶邻居，含所有它们之间的键）；
- 顶点属性 = （元素符号, 形式电荷, 总氢数, 是否芳香）；边属性 = （单/双/三/芳香）；
- **锚点无序**（`{u,v}` 与 `{v,u}` 同一条键，不重复计数）；
- 身份 `S` = 该带属性锚定图在**确定性精确规范化**下的字典序最小证书字符串。

`S` 必须是结构恒等的（同构 ⟺ 同一个 `S`）：实现为 1-WL 颜色细化 + 个体化-细化
（individualization-refinement）搜索树的完整枚举取最小证书，**不使用哈希指纹**，
因此不存在"哈希碰撞把不同结构合并"的风险（正确性由
`tests/test_attributed_canon.py` 用暴力同构交叉校验）。

敏感性口径（分开报告，不参与主判断）：

- **T2（粗口径）**：原子属性 = （元素, 是否芳香丢掉的粗键型：单键/多重键），
  即只保留元素与单/多键区分；
- **T3（去氢计数）**：T1 去掉"总氢数"，用于排除 1.2 的氢计数来源差异。

### 3.2 上下文 `C`

上下文必须描述局部子图 `S` **之外**的信息。本轮只用图内可确定性计算、标签无关的量，
不构造大量人工描述符：

**主口径 `CT`（精确规范化的 radius-2 邻域）**：以锚点为根的 radius-2 诱导子图的
精确规范化证书，顶点带角色标记（`anchor` / `patch`（S 内非锚点）/ `outer`（仅在外层
出现）），外层顶点额外带"其在 S+外层子图中的度数"，边带键型。`CT` 是精确身份：两个
实例的 `CT` 相同 ⟺ 存在保持半径 2 邻域、锚点集与键型结构的同构，即**局部子图之外的
一步结构环境完全相同**。

**粗口径（辅助，分开报告）**：`(outer_shell_multiset, ring_key, het_key, ring_dist)`
- `outer_shell_multiset`：S 之外直接相连原子的多重集（元素+电荷+氢数+芳香+键型）
  以及外部节点之间的外层键记录，锚点角色标注后整体排序；
- `ring_key`：与 S 的任一原子/键共享环的环大小多重集（`ring_dist` 为其计数），
  用于区分"同一 fragment 挂在脂肪族 vs 环上"；
- `het_key`：S 之外的外层原子中杂原子组成（元素→计数）。

只报告以下**少量轴**的存在性计数（每个 S 跨实例的去重值个数）：
`n_outer`（外层多重集不同值数）、`n_ring_key`、`n_het_key`、`n_ring_dist`。

**不用于判断的分子整体背景**（仅用于 §4.3 的混杂识别）：重原子数 `n_heavy`、
重原子分子式 `heavy_formula`、环数 `n_rings`、Murcko 骨架、拓扑家族。

### 3.3 独立单位（去重规则）

- **同一分子内的重复不算复用**：所有支持度统计以**不同分子**为单位（`mol_uid` = SMILES
  表行号；重复 SMILES 只算一次）；同一 `S` 在一个分子中出现 10 次只计 1 个分子支持。
  实例层面的重复程度以 `repeat_inflation` 单独报告（§5.1）。
- **骨架去重**：`n_scaf` = 拥有该 `S` 的不同 Murcko 骨架数（骨架 = 环系+连接体，
  去掉侧链；无环分子取规范化的重原子骨架字符串）。
- **拓扑家族**：`n_topo` = 拥有该 `S` 的不同拓扑家族数（§5.3 的"同系物"控制口径，
  由 `structural_relation_objects_v1` 的既有实现计算，仅作分层，不重定义 `S`）。
- **上下文支持度**：`n_ct_2mol` = 该 `S` 下被 **≥2 个不同分子**支持的上下文类型数
  （该 `CT` 的分子数 ≥2）。`n_ct_2scaf` 同理按骨架计。

### 3.4 混杂控制阶梯（composition control ladder）

对每个类型 `S`，把其实例按四个越来越紧的分子级分层键分层，在同一层内统计
"存在 ≥2 个 `CT`，且每个 `CT` 至少被 ≥2 个不同分子支持"的 `S` 数量。
**在层内计数时，`CT` 的分子支持度也只在层内计算**（即两侧支持都来自同类分子）：

| 层 | 分层键 | 控制的混杂 |
| --- | --- | --- |
| `size` | `n_heavy` | 分子尺寸 |
| `size_formula` | `n_heavy` + 重原子分子式 | 尺寸 + 元素组成 |
| `exact_stratum` | `n_heavy` + 分子式 + `n_rings` | 尺寸 + 组成 + 基本环结构 |
| `near_stratum` | `n_heavy // 2` + 杂原子组成 + `n_rings` | 尺寸差 ≤1 重原子时的同上 |

`near_stratum` **不用于判断**：尺寸窗口 ±1 会放行同系物（烷基链长差 1、杂原子组成不变），
与本轮要排除的"同一分子系列重复实例"重叠，故只作为"更松控制下的上界"报告。

另报告（描述性，用于 §5.3 解释）：
**匹配对占比** = 在 `(S, CT)` 不同（discordant）的实例对中，落在同一分层键内的对占比，
按 `size` / `size_formula` / `exact_stratum` / `near_stratum` 四档给出（配对数为闭式组合计数，
不做全量二元组枚举）。

### 3.5 化学性判别（"化学区分性"）

`S` 记为**有化学区分性**（`patch_has_hetero`）当且仅当其 `S` 子图（含锚点与一阶邻居）
中存在价态敏感的杂原子（`N/O/S/P/F/Cl/Br/I`）。`anchor_in_ring` = 锚点键在环内。
这是"不是纯碳链/纯烷基"的粗判别，用于 §6 的 C3。

## 4. 必做分析（与本协议一一对应）

### 4.1 局部结构跨分子复用（§5.1）
- 类型级支持度曲线：分子支持阈值 `1/2/3/5/10/20/50`、骨架支持阈值 `1/2/3/5/10`
  下的类型数、实例数、含杂原子类型数、环锚类型数、有 ≥2 独立上下文支持的类型数；
- 实例数直方图头部、`repeat_inflation`（实例/分子膨胀比、单分子最大出现次数）；
- 长尾/短尾形状用上述曲线描述，不做分布拟合。

### 4.2 同结构不同上下文的可比较性（§5.2）
- 在 `n_mols ≥ 5` 且 `n_scaf ≥ 2` 的类型（记为候选集）上，报告：
  `n_ct`、`n_ct_2mol`、`n_ct_2scaf`、`n_ct_2mol_2scaf`、
  `n_ct_pair_disjoint_scaf`（存在两个独立支持上下文且骨架集合不相交）、
  `n_outer` / `n_ring_key` / `n_het_key` / `n_ring_dist` 的均值与分布；
- **原始 vs 独立支持多样性分开报告**：`n_ct`（原始上下文数，含单分子支持的）
  与 `n_ct_2mol`（独立支持）并列给出。

### 4.3 排除平凡上下文差异（§5.3）
- 单骨架家族支配：候选集 `max_scaf_share` 直方图、`n_types_single_scaffold_family`
  （`max_scaf_share ≥ 0.999`，即同一骨架家族的重复实例支配该类型）；
- 组成/尺寸/环数：§3.4 的阶梯与匹配对占比；
- 化学组成结构：候选集里纯碳链、环内锚、含杂原子、二者兼有的类型数。

### 4.4 具体化学结构展示（§5.4）
固定五类案例（每类最多 3 个，全部来自主数据集/辅助数据集的实际实例）：

1. 同一 `S` 出现在**两个不同骨架**的分子中（骨架集合不相交的两个上下文）；
2. 同一 `S`、**不同的外层环境**（`outer_shell_multiset` 不同）；
3. 同一 `S`、**不同关系轴**（`ring_dist` ≥2 与 `het_key` ≥2 同时出现）；
4. 高频但**上下文单一**的反例（分子支持度高而 `n_ct_2mol ≤ 1`）；
5. 高频但**单一骨架家族**的反例（`max_scaf_share ≥ 0.999`）。

每个案例给出：数据集、行号（ZINC 行号 / Lipo 序号）、SMILES、锚点原子索引与键、
`S`/`CT` 证书、骨架字符串、该类型的支持度统计。**不标注**任何性质方向或正负贡献。

## 5. 辅助数据集与敏感性口径

- `lipo`（辅助数据集）：与主数据集完全相同的定义、阈值与统计，用来说明"更同质的
  实验测量数据集是否提供更好或互补的跨图支持"，以及按 §3.4 控制后还剩多少。
- T2/T3（结构口径敏感性）、`coarse`（上下文口径敏感性）、Lipophilicity 的
  train/test 划分**不参与**：只报告 ZINC×primary 与 Lipo×primary 的判断，
  其余为敏感性表格。

## 6. 判断规则（预先冻结）

设主口径（T1 + primary CT）下：

- **C1 支持广度**：`n_mols ≥ 5 且 n_scaf ≥ 2` 的类型数 ≥ `10`
- **C2 独立上下文多样性**：其中 `n_ct_2mol ≥ 2` 的类型数 ≥ `5`
- **C3 化学区分性**：C2 类型中含杂原子的类型数 ≥ `3`
- **C4 非平凡上下文差异**：C2 类型中，在 `exact_stratum`（同尺寸+同分子式+同环数）
  层内存在 ≥2 个各由 ≥2 分子支持的上下文的类型数 ≥ `5`

（阈值取在个位数～十位数的最保守量级：只要少于此，就说明样本量不足以支撑
"跨分子上下文支持"这一数据条件，而不是"差一点"。）

判断组合（`CONTEXT_SUPPORT_PRESENT` 要求**至少一个数据集**满足全部四项，
与本轮任务书写法一致）：

```text
若数据完整性检查失败                       -> INCONCLUSIVE
否则若 存在数据集满足 C1∧C2∧C3∧C4          -> CONTEXT_SUPPORT_PRESENT
否则若 不存在数据集满足 C1∧C2∧C3           -> INSUFFICIENT_CROSS_GRAPH_SUPPORT
否则                                       -> ONLY_TRIVIAL_CONTEXT_VARIATION
（即：某个数据集有支持广度与独立上下文，但在组成/尺寸/环数控制后消失）
```

判断始终归属于单一数据集（按上表逐数据集判定后再合并），不做“用 ZINC 的 C1 配
Lipophilicity 的 C2”这类跨数据集拼接。

报告必须同时给出：两条支持度曲线、`n_ct` 与 `n_ct_2mol` 的分布、控制阶梯的四档结果、
§5.4 的案例，以及**最重要的反对证据**。禁止只给结论标签。

## 7. 已知限制（写进报告，不隐藏）

1. ZINC 是合成/虚拟结构库：`S` 的跨分子复用反映的是组合枚举的覆盖度，而非实验观测；
   同结构在不同分子中的"上下文"仍都是计算构造的分子图。
2. PyG 官方图的氢/电荷编码与 SMILES 表的口径可能不完全一致（§1.2），故保留 T3。
3. `CT` 只覆盖 radius-2 邻域与少量环/杂原子轴：更远的结构关系（如全局折叠、长程电子
   效应）不在上下文定义内，不能被本轮"没有支持"的结论所否定。
4. Murcko 骨架去重是启发式的（去侧链），同一骨架可能包含化学上很不同的分子；
   因此 §5.4 同时给出具体 SMILES 供人工核对。
5. Lipophilicity 辅助集的分子高度同源（系列化合物多），其"上下文多样性"容易被
   同系物放大 —— 这由 §3.4 的阶梯与 `max_scaf_share` 诊断显式暴露。

## 8. 协议冻结顺序与披露

- 实现与协议同时定稿；在定稿前只做过**工程性冒烟测试**：用 `--limit 300/400`
  （即 10 000 行中前 300/400 行）验证代码可运行、哈希与配对核对通过、输出结构正确，
  并据此修掉实现 bug。
- 冒烟期间观察到的统计量（例如 `matched_share` 极低、`near_stratum` 在 Lipo 上宽松）
  导致**两处收紧**：C4 从"匹配对占比 ≥0.10"改为"`exact_stratum` 层内类型数 ≥5"
  （前者在 10k 级数据上几乎恒为假，会让结论由定义而非数据决定），并把
  `near_stratum`（尺寸 ±1，会放行同系物）从判断口径降级为描述性口径。
  两处修改都使判断**更难**给出正结论，不构成为了正结论而调参。
- 全量运行（10 000 + 3 360）从未在定稿前执行；本文件中所有阈值在首次全量运行之前
  已写入磁盘。
- 本轮结论不依赖单一事后门槛：所有阈值都配有完整曲线、分布与具体案例。
