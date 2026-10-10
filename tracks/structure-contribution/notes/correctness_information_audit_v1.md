# CSCL 正确性与信息充分性审计 v1（cscl-correctness-v1）

Track: `structure-contribution` · 日期: 2026-10-10（第二轮）
性质: 对 cscl-v0 的**正确性核查、修复与表示/信息流诊断**。不推翻也不改写
`notes/research_core.md`；对 v0 既有结论的修订以**勘误边界**形式记录（§7）。

---

## 0. 一句话

在进入任何新的表示设计之前，本轮确认并修复了 cscl-v0 的**两个已证实的实现
缺陷**（结构签名化学抹除 + 重标号不稳定；物理键双计），以真实化学图同构
验证了 SMILES↔PyG 对齐，量化了 v0 "跨分子共享"中的假共享比例（**按实例
权重 98.2% 的 v0 类型组是化学混合的**），并审计了 A/B/C/D 的真实信息流。
CPU 筛查与 seed-0 GPU 预测信息实验（O-unit / O-rich）结论见 §5/§6。

## 1. 已确认的 bug（先最小反例复现，后修复，均有测试钉住）

### BUG-1 结构签名（`cscl_units.unit_signature`，v0）——已证实，已修复

**反例（修复前实测复现）**：

1. 重标号不稳定：同一 N–C–N 链单元的两种原子编号得到两个不同签名
   （`d7d89f…` vs `740a93…`）。
2. 化学抹除：C–O / C–N / C–F / C–S 两原子链单元**同签名**；
   C–C / C=C / C#C **同签名**；C₄ 链 vs O₄ 链同签名。

**根因**：v0 在每轮 WL 后按"当前节点顺序首次出现"稠密重编号，最终签名只保留
稠密 id 的多重集。化学身份（原子/键类型）不进入 id 值，id 值又依赖扫描顺序。

**修复（v1）**：hash-based 1-WL——颜色 = 内容哈希（sha256 截断），每轮
`H(own_color # sorted(bond_type:neighbor_color))`；签名 = kind | n_atoms |
hash(sorted final colors)。原子/键类型进入每条颜色串；多重集对重标号不变；
跨进程确定。**已知理论限制**（如实记录）：1-WL 不能区分所有非同构带标签图
（测试 `test_signature_wl_limitation_documented_not_hidden` 用 K3,3 vs 三棱柱
钉住该限制；签名相等不声称同构）。

**修复过程的额外教训**：修复的第一版实现中邻接表元组 `(bt, j)` 与推导式解包
`for j, bt in` 顺序不匹配——被**既有测试** `test_signature_relabel_invariant`
当场抓住。测试在该轮确实起到了回归作用。

**修复后抽样健全性**：2000 对同 v1 签名跨分子单元对 100% 通过带属性 VF2
同构（`audit.json: fake_sharing_check.v1_same_type_iso_rate = 1.0`）。

### BUG-2 物理键双计（`cscl_features.molecule_units` / partition 输入）——已证实，已修复

**事实**：本地 `data/ZINC` 快照（PyG `ZINC(subset=True)`，官方 molecules.zip +
benchmarking-gnns split index）的 `edge_index = symmetric_adj.nonzero()`——
**每根物理键存储两次**（双向各一次）。实测 10000 分子：无自环、无方向键型
不一致、无平行多重边（每键恰 2 条有向边）。

**反例（修复前实测复现）**：

1. 六元单环：`intra_bonds=12`，v0 描述子 `cyclomatic = 12−6+1 = 7`（正确 = 1）。
2. 单根 inter 键的关系直方图 `[bt, bt]`（应为 `[bt]`），`log1p(count)` 用 2×count。
3. `attach_count` 翻倍 → `is_terminal`（attach≤1）对任何有连接的单元恒为 0。

**修复（v1）**：`unique_undirected_bonds()` 先规范化为唯一无向物理键
（u<v；方向键型不一致 / 自环 / 多重边 → `ValueError`），`build_partition`
在划分前统一调用；描述子、关系直方图、环秩全部恢复物理正确。划分算法
（原子归属）本身不变——v0 的**原子归属是对的**（networkx Graph 天然去重
双向边），错的只是键计数路径。

**测试**：`test_unique_bonds_*`、`test_six_ring_cyclomatic_is_one_with_pyg_input`、
`test_inter_relation_bond_counted_once_with_pyg_input`、
`test_molecule_units_descriptors_not_double_counted`、
`test_real_zinc_row_bond_dedup_and_coverage_sample`（真实 ZINC 行级：dedup 后
`2×phys = len(edge_index)`、覆盖与唯一归属成立）。

### BUG-3 SMILES↔PyG 对齐声称过强——已证实为"不充分声称"，已升级为真实化学图核对

v0 的 `crosscheck_atom_counts` 只做 SMILES 字符串的粗略原子计数（容差 2），
却产出了 `rows_checked=10000, max_abs_diff=0` 的表格，被后续决策文档引用为
"对齐校验 10000/10000"。**原子数一致 ≠ 图与属性一致**。

**升级（v1，无 RDKit 环境下的替代方案）**：

- 环境核查：`rdkit` 不在 `pyproject.toml`/`uv.lock`；不为本轮新增依赖，
  改用**手写严格 SMILES 解析器** + kekulization（容量规则 + 完美匹配）+
  **VF2 带属性图同构**（`cscl_smiles_graph.py`），不支持语法一律报错不猜。
- 原子类型语义核查：ZINC 28 类原子 = 元素 + **显式** H + 电荷（芳香 CH、
  酰胺 NH 都是普通 'C'/'N'；`[C@H]`→'C H1'、`[nH+]`→'N H1 +'）。数据字典
  无不带电的 'N H1'/'O H1'/'S H1' → 芳环 [nH] 在数据侧只能存成普通 'N'
  （数据集信息损失，对比时按规则归一化并在报告里声明）。
- 三层判定：tier1 = 键级+全类型精确同构；tier2 = 芳香松弛（kekulé 选择歧义）；
  tier3 = 仅元素拓扑。

**结果（全部 10000 行）**：tier1 **9885** + tier2 **111** = **10000/10000
同构**；原子/键计数不符 0；解析错误 0；kekulé 歧义 43（其中 111 行 tier1
失败但 tier2 通过——纯 kekulé 选择差异，非化学差异）。→ 行对齐成立，但
表述必须是"**10000/10000 通过带属性同构（111 行在芳香松弛层）**"，而非
"精确化学图同构 10000/10000"（111 行的精确键级归属存在 kekulé 表示歧义，
无法在本环境裁决）。

**限制声明**：非 RDKit 实现；立体/同位素两侧均忽略；kekulé 43 例歧义未逐个
人工裁决；不覆盖 28 类字典之外的元素。

## 2. 表示变化（任务 §6.2；`results/cscl_v1_units_audit/audit.json`）

同一划分（fit_inner 7200 / dev 2000，SMILES 分组，划分逐字节不变）、同一
55271 个单元实例（划分不变 ⇒ 单元原子归属逐分子相同）：

| 统计 | v0（缺陷签名+双计键） | v1（修复后） |
|---|---:|---:|
| fit_inner 不同类型数（原始签名） | 346 | 3169 |
| 全 10000 不同类型数 | 398 | 3939 |
| 词表（fit_inner，min_count≥3） | 252 known / 267 total | **746 known / 761 total** |
| top-10 类型单元覆盖率（fit_inner） | 77.0% | **50.5%** |
| singleton 类型实例占比 | 0.36% | 5.14% |
| dev UNK 单元率 | （v0 审计 ≈低） | 8.45% |
| max cyclomatic（fit_inner 单元） | **31**（不可能值） | **6** |
| 每分子单元数 / 关系数 | 5.51 / （双计） | 5.51 / 4.51 |

**假共享量化（核心问题："原先的跨分子共享有多少是真的？"）**：

- 抽样带属性同构（2000 对，组均匀抽样）：**v0 同类型对仅 21.35% 真实同构**；
  v1 同类型对 100% 同构（2000 对，抽样）。
- 精确（无抽样）纯度：v0 的 398 个类型组中 177 个"纯"（组内只对应 1 个 v1
  化学类型），但**纯组只覆盖 1.81% 的单元实例**；**98.19% 的单元实例位于
  化学混合的 v0 类型组**。
- 类型拆分/合并：51.34% 的单元实例因修复被拆出原 v0 类型（该 v0 组的
  非最大 v1 子组部分）；2.07% 因修复被合并（v0 重标号不稳定把同化学单元
  分散，v1 正确统一）。

**结论**：v0 报告中的"跨分子共享充分、top-10 覆盖 77%"主要由签名碰撞构成。
按实例权重，v0 词表的"共享"约 98% 是假共享。这一量化**修订**（而非推翻）了
`decision_cscl_v0_h1_refuted_20261010.md` 中"保留资产：RINGCHAIN-v0 划分
（…跨分子复用充分…）"的表述——划分（原子归属）确实可保留，但其类型词表
与共享性证据不成立。

## 3. A/B/C/D 信息流审计（任务 §4；只读代码 + 确定性检查，未重训）

### 3.1 B 的关系项实际读取了什么

`CSCLModel.relation_terms`：`γ = MLPγ([E_ru[t_lo]; E_ru[t_hi]; r_ij − ν])`。

- **看得见**：两端单元的**类型 id**（经独立关系 embedding 表）、关系键型
  直方图 + log1p 键数（物理正确性在 v0 被双计破坏）。
- **看不见**：两端的**实例级描述子** c_i/c_j（大小、环秩、附着数、终端性、
  原子/键直方图——这些只进入 unary 的 δ）、任何端点局部环境聚合、任何高阶
  组合。
- **D 看得见而 B 看不见**：D 的 pair 路径 `pair_mlp([h_i; h_j; r])` 的 h 来自
  `[E[t]; c]`——即**实例级描述子**直接进入关系读出；D 的图级头再做
  mean/sum 聚合（受 `log1p(count)` 标量调制）。因此 `D − B` **不能**全部
  解释为"贡献分解造成的损失"：输入信息（实例描述子进 pair 通道）与读出
  结构（非线性聚合 vs 逐项加和）同时不同。
- **C 的关系打乱**：`shuffle_relation_contents` 只置换 `(rel_type_lo/hi,
  rel_feat)` 三元组，保持 `rel_mol`（每分子槽数）不变——**跨分子混合**了
  关系内容（这正是设计意图：破坏真实对应、保留边际）。但注意：(a) 打乱在
  **batch 内**进行，eval 时用固定种子 generator——同一分子在不同 batch 组成
  下会得到不同的关系内容 → **评估依赖 batch 组成**，不是干净的逐分子预测器
  （v0 报告里 C 的 dev MAE 因此只能读作机制对照，不能读作化学预测器）；
  (b) 打乱同时改变了"关系类型与分子"的联合分布（例：含 rare 类型的关系被
  均匀撒到所有分子），**边际保留只在全局 pool 层面成立**；(c) 因此"B < C"
  的旧结论只能支持"真实对应在 v0 表示层级无增益"，不能支持更强命题。

### 3.2 容量/读出差异来源

| 差异 | 来源 |
|---|---|
| B vs A（+0.042 seed 均值） | 输入不变，读出加了一路加性 pair 项 |
| C vs B | 同容量；只破坏关系对应（但评估依赖 batch，见上） |
| D vs B（−0.063） | **输入+读出同时不同**：实例描述子进 pair 通道 + 非线性聚合头；参数 27k vs 20k |
| XGB vs B | 完全不同的模型类与特征展开（含类型计数向量+描述子聚合），不可参数匹配（协议已声明） |

**对旧结论强度的修订**：`decision_cscl_v0_h1_refuted` 的机制读法"D 好于 B
说明瓶颈在加性成对分解形态"需要降格——D 与 B 的差异混合了 (i) 贡献分解
约束、(ii) pair 通道的输入信息差、(iii) 全局非线性聚合。v0 数据不足以把
性能差归因于 (i) 单独。H1 "被反驳"在预注册判据意义上成立，但**归因解释**
（第 2 条"机制读法"）应视为未验证假设。

## 4. 修复版本的工程边界

- 代码就地修复（git 历史保留 v0 = commit 6549c04 的行为）；v0 的
  REPORT/protocol/decisions 原文未动。
- 新版本标识：`FEATURE_VERSION = "cscl-correctness-v1"`、
  `SIGNATURE_VERSION = "wl-hash-v1"`、`PROTOCOL_ID = "cscl-correctness-v1"`、
  代码指纹 `cscl_features.code_fingerprint()`（cscl_units+cscl_features 的
  sha256）。
- 缓存隔离：新缓存 `data/cache/cscl_v1_units.pt` 存版本字段；旧
  `cscl_v0_units.pt` 无版本字段 → `load_or_build_units_cache` **拒绝加载**
  （测试钉住）。v0 驱动的 `prepare_data` 同样走该 guard，并在 main() 打印
  特征代码版本横幅，防止把 v1 语义误标为 v0。
- v0 签名保留在 `cscl_units_v0_reference.py`（仅审计用，明确 legacy 标注）。
- 本轮新协议与实验见 §5/§6；协议细节：`run_cscl_v1.py` 模块 docstring +
  `configs/`（如后续正式化，按 research_core §10 新开 protocol id）。

## 5. CPU 筛查（任务 §6.3；`results/cscl_v1_screen/screen.json`）

固定配置（无超参搜索），同一 fit/monitor/dev 划分，dev raw-y MAE：

| 模型 | v0 特征 | v1 修复特征 | rich573 |
|---|---:|---:|---:|
| Ridge α=1 | 0.5026 | 0.4171 | **0.3370** |
| Ridge α=10 | 0.5257 | 0.4609 | 0.3392 |
| HistGB | 0.5361 | 0.5421 | 0.4136 |
| XGBoost | 0.4771 | 0.4532 | **0.3717** |

读法：(a) 修复一致提升单元特征的信息量（Ridge −0.086，XGB −0.024）；
(b) rich 静态特征明显强于两套单元特征 → 进入 GPU 阶段的依据；rich 的
Ridge 在 clip 修复前后几乎不变（0.3366→0.3370），XGB 用原始特征不受影响，
旧筛查结论不变。

## 6. GPU seed-0 预测信息实验（任务 §7；`results/cscl_v1_gpu/REPORT.md`）

同一 fit/dev 划分与标签（raw y）、同 seed 0、同训练预算与监控规则；
非 GNN/Transformer 头。**容量与输入不严格等价**（36.3k vs 30.0k 参数；
单元特征 vs 573 维全局特征），如实声明。

| 臂 | soup dev MAE | best-epoch dev | fit | params |
|---|---:|---:|---:|---:|
| O-unit（修复后单元表示） | 0.35708 | 0.36393 | 0.26291 | 36 289 |
| O-rich（base573） | **0.33206** | 0.34345 | 0.25340 | 29 953 |

- 配对差 O-unit − O-rich = **+0.02502**，分子级 bootstrap 95% CI
  **[+0.00124, +0.05078]**（同一 seed 内部 dev 重采样，非跨 seed 证据）。
- 子群：ring_units≥2 分子上两臂持平（−0.003）；差距几乎全部来自
  0–1 个环系的分子（+0.23）。
- 对照：v0 同族 opaque（缺陷特征）0.40005 → 修复后 0.35708
  （**仅修复 +0.043**）。
- **判断：情况一成立** —— 修复后单元表示仍缺少 rich 静态特征的部分
  信息（集中在少环分子），且 v0 差距的实质部分来自表示实现错误
  （H-R 获证实），“归因于贡献分解”仍无证据支持。
- 过程记录：orich 首跑因未 clip 标准化爆炸（1309）作废重跑（详见
  REPORT §6）；逐分子导出初版为标准化单位，已精确重建并修正导出路径。

## 7. 对既有记录的勘误边界（不删改原文）

1. `results/cscl_v0_gpu/REPORT.md` 与 decision 文档的**数字与预注册判定**
   保持有效（它们记录的是 commit 6549c04 行为）。
2. 勘误 a：v0 报告引用的"对齐校验 10000/10000"应降格为"原子数核对
   10000/10000"；化学图级核对见本审计 §1 BUG-3（本轮通过 10000/10000，
   其中 111 行芳香松弛层）。
3. 勘误 b："跨分子复用充分"的证据不成立（假共享 ≈98% 实例权重，见 §2）；
   RINGCHAIN **划分**（原子/键归属）本身经本轮测试仍成立。
4. 勘误 c："D 好于 B ⇒ 瓶颈在分解形态"降格为未验证解释（信息流混同，
   见 §3）。
5. "C 是干净机制对照"降格：C 的评估依赖 batch 组成，只能作机制对照，
   不能作化学预测器比较（见 §3.1）。

## 8. 本轮测试清单（`tests/test_cscl_correctness_v1.py`，67/67 通过）

签名（重标号不变性 ×19 结构 ×8 置换、原子/键类型保留、同构⇒同签名抽样、
K3,3/三棱柱 WL 限制钉住、确定性）；物理键（dedup、方向不一致 raise、自环
raise、多重边 raise、六元环环秩=1、链环秩=0、稠环/螺环=2、inter 键计一次、
真实 ZINC 行级）；描述子（双计修复、terminal 修复、关系端点索引正确）；管线
（全管线重标号不变、v1 同签名⇒同构 300 随机图抽样）；工程（stale cache
拒绝、版本 guard 接受、y 标签隔离、划分指纹不变）；模型（Opaque 单元置换
不变性）。
