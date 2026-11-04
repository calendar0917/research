# 协议：`tsb-lipo-feature-ladder-v1`

> 本文件是**预注册协议**。所有数字在正式训练开始前冻结；执行期间只允许修工程 bug
> 并记录重跑范围。第一轮只做 Lipophilicity，只做 XGBoost，不训练任何 GNN/MLP。

- Track：`tracks/tree-statistics-boundary/`
- 协议 ID：`tsb-lipo-feature-ladder-v1`
- 机器可读配置：`configs/tsb-lipo-feature-ladder-v1.yaml`
- 列定义冻结文件：`configs/feature_spec.json`（由 `features` stage 生成）
- 控制面 protocol：`protocols/tsb-lipo-feature-ladder-v1.yaml`
- 研究问题：不同图预测任务中，哪些显式结构统计足以支持树模型预测？性能受局部模式数量、结构范围、输入信息和训练样本量怎样影响？

---

## 1. 数据与划分

数据集：Lipophilicity（MoleculeNet），4,200 个分子，目标 `exp`（logD）。

来源固定为仓库内已 vendor 的 ChemBench：

| 项 | 值 |
|---|---|
| upstream | `https://github.com/shenwanxiang/ChemBench` |
| commit | `0b10d4c1ac72b6929503694a5f60446824736391` |
| CSV | `tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v1/analysis/source/chembench/Lipop/Lipop.csv.gz` |
| CSV sha256 | `34280f0adde234bcf477bd98878c9918c9ea51ed859c9efdc31917c6e5ecd758` |

三套已发布划分，**原样使用，不生成任何新划分**：

| 划分 | 文件 | 文件 sha256 | Train / Valid / Test |
|---|---|---|---|
| `Lipop.random122` | `Lipop.random122.ind.pkl` | `bf7acb17…a7d` | 3360 / 420 / 420 |
| `Lipop.random123` | `Lipop.random123.ind.pkl` | `afaed4c6…0e8` | 3360 / 420 / 420 |
| `Lipop.random124` | `Lipop.random124.ind.pkl` | `2b48a41c…ef3` | 3360 / 420 / 420 |

行号哈希（int64 row index，逐划分）：

| 划分 | Train | Valid | Test |
|---|---|---|---|
| 122 | `4e8d42a9…af96` | `31e3f28f…a365` | `a4d92ad9…c93` |
| 123 | `a3af8ce7…6db4` | `a79e192e…0982` | `6c335c84…6df2` |
| 124 | `dacdee79…d887` | `e2100b1b…ef94` | `c4a5b71d…427d` |

完整哈希写在 `code/tsb_common.py`，每次载入都会校验（`code/tsb_data.py`）；
与 `lcrc_molnet_benchmark_pilot_v1/analysis/split_manifest_Lipophilicity.json`
和 `lcrc_taskaware_representation_pilot_v1` 的 Lipophilicity 记录逐项一致。
三划分两两不重叠、并集恰为 4,200 行（gate 01）。

## 2. 五个预注册实验臂

| ID | 输入 | 检验的问题 |
|---|---|---|
| `G0` | 基础全局组成与拓扑统计（76 列） | 不看具体局部模式时能做到多少？ |
| `WL2_BIN` | `G0` + rooted Typed-WL round 0–2 的 0/1 特征 | 局部模式是否出现是否足够？ |
| `WL2_COUNT` | `G0` + 同一词表下 round 0–2 的原始计数 | 模式出现次数是否额外有用？ |
| `WL3_COUNT` | `G0` + rooted Typed-WL round 0–3 的原始计数 | 增加结构范围是否有用？ |
| `MORGAN_COUNT` | `G0` + RDKit Morgan radius=2、2048 桶计数 | 标准化学指纹与 Typed-WL 有什么区别？ |

特征块维度（每个公开划分各自拥有 Train-only 词表，因此 round 0 维度因划分而差 1 列）：

| 臂 | 122 | 123 | 124 |
|---|---:|---:|---:|
| `G0` | 76 | 76 | 76 |
| `WL2_BIN` | 76+50+513+513 = 1152 | 1151 | 1151 |
| `WL2_COUNT` | 1152 | 1151 | 1151 |
| `WL3_COUNT` | 1152+513 = 1665 | 1664 | 1664 |
| `MORGAN_COUNT` | 76+2048 = 2124 | 2124 | 2124 |

**同一划分内**所有臂与三档样本量共享完全相同的列；不同臂维度不同，
因此臂间性能差异**不能**直接等同于纯信息增益（见 §8 限制）。

### 2.1 `G0`（76 列，全部为标签无关的计数或计数比值）

```
n_atoms, n_bonds, n_components, cyclomatic_number, n_rings, n_ring_atoms, n_ring_bonds,
n_aromatic_atoms, n_aromatic_bonds, n_heteroatoms, n_charged_pos, n_charged_neg,
n_degree1_atoms, n_aromatic_rings, n_nonaromatic_rings,
el_{C,N,O,S,F,Cl,Br,I,P,B,Si,Se,other},
deg_{1,2,3,4,5,6,7plus},
bond_{SINGLE,DOUBLE,TRIPLE,AROMATIC,other},
charge_{m2,m1,0,p1,p2,other},
h_{0,1,2,3,4plus},
hyb_{S,SP,SP2,SP3,SP3D,SP3D2,other},
chir_{R,S,none,other},
ring_size_{3,4,5,6,7,8,9plus},
ratio_ring_atoms, ratio_ring_bonds, ratio_aromatic_atoms, ratio_aromatic_bonds,
ratio_heteroatoms, ratio_degree1_atoms, rings_per_atom
```

`chir_*` 使用 RDKit CIP 码（R/S/none），而不是 `CHI_TETRAHEDRAL_CW/CCW`
原始 tag——原始 tag 相对邻居bond顺序定义，不是原子编号不变量。

**明确排除**：Crippen logP/MR、TPSA、QED、合成可及性分数、BCUT/EState/VSA、
`fr_*` 官能团计数、以及完整 217 维 RDKit 描述符列表。这些属于单独的化学描述符参照。

### 2.2 Typed-WL（`WL2_*` / `WL3_*`）

- 与 `G0` 共用 `code/tsb_mol.py` 的同一份图视图（原子/键属性只在该处提取一次）。
- 原子属性（顺序固定）：`atomic_number, formal_charge, aromaticity, total_h, hybridization, chirality`。
- 键属性（顺序固定）：`bond_type, aromaticity, conjugation, stereo`。
- round 0 token：`A(atomic_number=…,formal_charge=…,…)`。
- round k+1 token：`R{k+1}(<当前token>)[<排序后的邻接描述串>]`，
  邻接描述串 = `E(<bond attrs>)><邻居 round-k token>`，全部按字符串升序排序后 `|` 连接。
- token **只由字符串拼接构造**，不使用 Python 内置 `hash()`；与原子编号、键端点方向无关。
- 每轮词表：只用**对应公开划分的完整官方 Train** 的原子 token 频数；
  取频数最高的 512 个；频数相同按 token 规范字符串升序；另设 1 个 OOV 桶。
  round 0 的 Train 只出现 48–49 种签名（< 512），因此保留全部，列数 49/50。
- 各轮词表独立；`WL2` 与 `WL3` 的 round 0–2 列名、列序、OOV 规则完全一致。
- `WL2_BIN` 与 `WL2_COUNT` 使用**完全相同**的词表与列序，仅把计数截为 0/1
  （OOV 桶同样二值化，表示"本分子本轮存在未见 token"）。
- Validation/Test 只应用词表，不更新词表。
- 缺失值策略：不使用 NaN；未见 token 计入 OOV 桶。SMILES 解析失败是硬错误
  （本数据集 0 例，已在 manifest 记录）。

特征缓存为 CSR（`scipy.sparse`），列名 `wl{r}_t{rank:03d}_{token_sha256[:12]}` + `wl{r}_oov`。

### 2.3 `MORGAN_COUNT`

RDKit `rdMolDescriptors.GetHashedMorganFingerprint(mol, radius=2, nBits=2048)`，
保存每个桶的出现次数（不是 0/1）；同样与 `G0` 拼接。
仓库既有的 `ECFP4 binary + 217 RDKit descriptors + XGBoost` 结果只作为
**只读外部参照**（`analysis/tables/external_reference.json`），不参与任何臂或对照。

## 3. 唯一学习器

```yaml
objective: reg:squarederror
tree_method: hist
device: cpu
n_estimators: 500
max_depth: 5
learning_rate: 0.03
subsample: 0.8
colsample_bytree: 0.8
reg_lambda: 5.0
min_child_weight: 1
random_state: 0
n_jobs: 4          # <= 8
```

五个臂、三个划分、三档样本量共用同一配方。不使用 Optuna / 网格 / 早停 / 学习率搜索。

## 4. 训练样本量

- N = 1,000 / 2,000 / 3,360（完整 Train）。
- 顺序：`sha256("20261104:<原始 row id>")` 升序；三档是**同一顺序的前缀**。
- 不按标签抽样、不做标签分层。
- 词表始终来自**完整官方 Train**（3,360 行）的标签无关图；因此该曲线考察的是
  **有标签训练样本量**，不是"只有 N 张图可用"。这一点在报告与表格注释中显式标注。

## 5. 指标

主指标：原始标签尺度 RMSE（越低越好）。
辅助：MAE、R²、Train/Validation RMSE 差距、模型训练耗时、特征构造/装配耗时。

## 6. Test 政策（本轮实际执行记录）

1. 15 个完整 Train 模型训练完成，`analysis/fits/*.json`、模型 `.ubj`、
   Train/Valid 预测 `.npz` 全部落盘并记录 SHA-256。
2. `freeze` stage 校验每个模型/预测文件哈希，写出
   `analysis/freeze/pre_test_freeze.json`（15 个模型、科学配置哈希、代码哈希、
   特征清单哈希、特征列定义哈希），并把该文件的 sha256 写入
   `analysis/freeze/freeze_pointer.json`。此时 `test_not_read = true`。
3. 只在 `--mode terminal`（控制面 `test_access=granted`）下运行 `test` stage：
   重新校验 freeze/pointer/科学配置/代码/特征清单/特征列定义，再逐个校验模型哈希，
   只对这 15 个模型评分。
4. 15 个模型全部报告 Test，不只报告最优模型；Test 不参与任何选择。
5. **Test 暴露**：Lipophilicity Test 标签在本仓库更早的 structural-sufficiency
   Sprint 01 与后续 repeat 中已被读取。本轮所有 Test 数字标记为
   `EXPOSED_PUBLIC_TEST`，**不**声称是首次盲测。

执行过程中的两次 freeze：
- freeze v1（run `20261104-194231-436224f9`，sha `9a372c15…`）在 Test 读取**之前**
  被 freeze v2（run `20261104-194308-864f304c`，sha `58afd451…`）取代，
  原因：v1 把运行阶段键 `stage` 也算进了"科学配置哈希"，terminal 运行因此无法
  与 freeze 对齐。v2 把 `stage`/`rebuild_features` 排除在科学配置之外。
  两次都在任何 Test 标签被读取之前，因此不构成 Test 泄漏。
- 一次 terminal 运行 `20261104-194252-514c5178` 在上述配置哈希检查处**失败**，
  未读取任何 Test 标签；随后以 v2 freeze 成功执行。

## 7. CPU 正确性门槛（正式训练前全部通过）

`code/tsb_gates.py`，P0 stage 与 pytest 共用同一实现：

| # | 门槛 | 结果 |
|---|---|---|
| 01 | 三套划分行号/数量/互斥/并集与历史一致 | PASS |
| 02 | 特征不读标签（把 `exp` 换成 N(0,1) 噪声后特征逐位不变；特征构造函数签名无标签参数） | PASS |
| 03 | 每轮词表只由对应官方 Train 重建 | PASS |
| 04 | `WL2_BIN` 与 `WL2_COUNT` 列定义/列序/稀疏模式一致，BIN 为 0/1 且与计数同支撑 | PASS |
| 05 | `WL3_COUNT` 的 round 0–2 与 `WL2_COUNT` 完全相同，仅多加一个 round-3 块 | PASS |
| 06 | 随机原子重编号 + 完全端点翻转后 G0 / 四轮 token 多重集 / Morgan 不变（200 分子） | PASS |
| 07 | 缓存重载与全新计算逐元素一致（G0 全体、WL counts/binary 全体、Morgan 200 行） | PASS |
| 08 | 15 个臂/划分矩阵全有限；OOV 列存在；解析失败为硬错误；0 例解析失败 | PASS |
| 09 | 45 个已记录 fit 的 Train 行号/标签哈希/预测行号对齐 | PASS |
| 10 | 模型存盘重载后预测逐位一致 | PASS |
| 11 | 合成边界例子（见下） | PASS |

**合成正确性/表达能力诊断（gate 11）**：两个 24 元单键环，环原子 0 = N、
原子 k = O，分别取 k = 8 与 k = 12。

- N–O 图距离：8 vs 12；
- rooted typed-WL round 0/1/2/3 的 token 多重集：两图**完全相同**；
- `G0` 向量：两图相同；
- Morgan radius-2 计数向量：两图相同。

含义：本轮五个臂对这对图给出**同一个特征向量**。"两个标记原子之间的距离"
是非局部统计，G0 组成计数、round 0–3 的 rooted typed-WL、radius-2 Morgan 都表达不了。
该例子只用于特征正确性与表达能力诊断，不引入新模型或新 benchmark。
（因为 d(N,O) > 6，没有任何原子能在半径 3 内同时看到两个标记；控制实验
12 元环 d=4 vs 6 则可以被区分，见 `tests/test_tsb_synthetic.py`。）

## 8. 预算与停止条件

- CPU-only，本机执行，不申请 GPU。
- 预算：正式任务累计 ≤ 90 分钟 CPU 墙钟。实际：见 REPORT §7。
- 接近预算时优先保留完整 Train 的 3 个划分 × 5 臂；不擅自放宽预算。
- 完成第一轮后立即停止，不启动第二轮；Test 不用于选择下一步。

## 9. 不可变项（本轮执行完毕后）

- 数据集、划分、行号顺序；
- 五个臂的定义、G0 列定义、WL 属性集合、词表大小与 OOV 规则；
- XGBoost 参数；
- 三档样本量及其嵌套顺序；
- Test 政策与 15 个 Test 评分模型的清单。

任何改动都需要新的 `protocol_id`，不得覆盖本轮记录。

## 10. 已知冻结文件缺陷（不修改，保持哈希）

`configs/feature_spec.json` 中每个臂的 `blocks[*].dim` 对 WL 块记录的是
`len(column_names([], r)) = 1`（空词表占位），这是打包时的字段错误。
**真实维度**记录在同一文件的 `arms[*].per_split[*].n_columns`、
`arms[*].n_columns_by_split` 与缓存词表 manifest 中（49/50 或 513）。
该字段不参与任何计算；为保持预注册文件哈希不变，本轮不修改，仅在此与
`analysis/post_freeze_edits.json` 记录。
