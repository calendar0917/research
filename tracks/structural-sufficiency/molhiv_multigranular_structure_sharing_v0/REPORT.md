# REPORT — MolHIV 多粒度显式结构发现与预测价值研究 v0

Round: `molhiv-multigranular-structure-sharing-v0`
Dataset: `ogbg-molhiv`（官方 scaffold split，41 127 分子 / Train 32 901 / Valid 4 113 / Test 4 113）
Seed: `0`（单 seed，by design）
协议：`PROTOCOL.md`（执行前冻结）；机器可读配置 `configs/protocol.json`、`analysis/*.json`
资源/环境/哈希总账：`analysis/provenance.json`

本轮的显式结构是**精确枚举出的属性诱导子图**（不是 token、不是切分、不是 motif 词典猜测）：
size-2/3/4 为全部连通诱导子图，cycle-5/6 为全部无弦（诱导）环。类型同一性 = 属性图同构类，
规范键为 1-WL 颜色细分下的全排列最小化（`code/msds_enum.py`）。

---

## 0. 一页结论（先回答六个问题）

**Q1 多粒度显式结构在固定表示下是否有增量价值？** 有，但幅度小且只在高容量模型上变大。
固定表示（Morgan r2 2048 二值 + 线性 logit）下，`B0 → B1`（加 COARSE）Valid ROC-AUC
0.8114 → 0.8188（+0.0074，scaffold bootstrap 95% CI −0.0054 ~ +0.0205），Test 0.7478 →
0.7579（+0.0101，CI −0.0016 ~ +0.0210）；`B1 → B2`（再加 FULL）Valid +0.0095
（CI −0.0026 ~ +0.0223），Test +0.0110（CI +0.0024 ~ +0.0198，唯一在 Test 上 CI 不跨 0 的相邻对比）。
即：**方向一致、Test 上更明显，但单 seed 下 Valid 区间仍跨 0**，不能称为稳健显著。

**Q2 粗粒度与细粒度，哪个更值钱？** 细粒度（FULL）叠在粗粒度之上仍带来增益，且这条增益比
"把 COARSE 换成 FULL"（`BF`）更值钱：`B2`（C+F）Valid 0.8283 / Test 0.7688 优于
`BF`（仅 F）0.8250 / 0.7618，配对 Δ Test +0.0071（CI −0.0013 ~ +0.0158）。粗粒度的作用更像
**支架**：单独给 FULL 而缺 COARSE 会掉，两者一起给最好。层级不是可有可无的装饰。

**Q3 "真实父子关系 / 统计支持"是否提供超出打乱与平均的额外价值？** **没有可分辨的证据。**
`B2-PNULL`（父子映射整体打乱）Valid 0.8340 反而高于 `B2` 0.8283（Δ −0.0057，CI −0.0138 ~
+0.0031），Test 低 0.0021（CI −0.0052 ~ +0.0095）；`B2-U`（所有 FULL 子结构共享同一权重）
0.8271 / 0.7663 与 `B2` 几乎同分（Valid Δ +0.0012）。`B3`（按支持度加权正则）比 `B2` 高
0.0013（Valid）/ 0.0035（Test），但比 `B2-U` 只高 0.0025 / 0.0061，比 `B3-NULL`（权重打乱）
只高 0.0006 / 0.0034 —— 全部 CI 跨 0。**结论：真实层级与支持度信息在这一表示下没有被利用，
增益来自"多了一组结构化特征"，而不是"层级被理解了"。**

**Q4 结构价值是否集中在高支持结构？支持度正则是否真的压缩低支持系数？** 是，而且压缩得很干净。
按 Train scaffold 频次四等分（897/919/919/917 个 FULL 类型），覆盖率极不均衡：最高一档占各划分
诱导实例的 63.0%（Valid）/ 64.5%（Test）/ 68.8%（Train），最低一档只占 0.5%。
B3 相对 B2 的有效系数 |v| 比值为 0.30 / 0.75 / 1.17 / 1.82（低→高支持），
低支持档 99.7% 的类型被压缩、高支持档只有 7.1% 被压缩；FULL 块的 L2 质量从 20.35（B2）变成
49.71（B3），其中最高支持档的份额从 56.0% 升到 80.0%。**支持度加权确实把容量从长尾搬到了头部，
但这套搬运只值 +0.0013 Valid。**

**Q5 结构价值能否迁移到"低支持 / 未见结构"占比高的分子？** 部分能，但主要在"中等"分子上，
且与分子复杂度混杂。以分子内"低支持或未在 Train 词表出现的 FULL 实例占比"分箱（边界由 Train
定义），Test 上 `B0 → B2 → B3` 的 ROC-AUC：占比 0–0.25 组 0.8334 → 0.8417 → 0.8419；
0.25–0.50 组 0.5583 → 0.6146 → 0.6228；0.50–0.75 组 0.5580 → 0.5967 → 0.6032（n=310，仅 5 个正例，
只能算方向性）；0.75–1.00 组 n=55、正例 0，判 `INSUFFICIENT_SUPPORT`，不报数字。
Test 中有 4 081/4 113 个分子至少含一个词表外 FULL 类型 —— **FULL 词表在 Test 上的实例覆盖率只有
76.8%，"未见结构"是常态而不是边角情况**。所以这部分结论只能写成"在最难的中间档上仍有同向增益"，
不能写成"结构信息解决了未见结构"。

**Q6 与传统强基线、普通 GINE 相比，显式结构的增量还成立吗？** 在 XGBoost 上成立且幅度最大，
在 GINE 上方向一致但不能判定。冻结的历史 XGBoost（ECFP4 + 217 RDKit 描述符，EXACT_MATCH 复现
Valid 0.836692 / Test 0.762440）加上 C+F 后变成 `XGB_STRUCT`：Valid 0.8329（AUC −0.0038，
CI −0.0268 ~ +0.0222；但 Valid AP +0.0436，CI +0.0005 ~ +0.0845）/ Test 0.8070（**+0.0445，CI +0.0193 ~ +0.0697**），结构列拿走 54.1% 的
总 gain、4 406 个结构列里 821 个被用到。普通 GINE 上：`G-M`（仅 Morgan 图级分支）Valid 0.8038 /
Test 0.7591，`G-STRUCT-U`（+C+F 零初始化线性分支）Valid 0.8142（+0.0104，CI −0.0160 ~ +0.0389）/
Test 0.7694（+0.0103，CI −0.0181 ~ +0.0362）。**同一批结构特征在树模型上换来 Test +0.0445，
在 3 层 GINE 上只有 +0.010 且不可分辨；这不是"结构没用"，而是"这套固定表示 + 线性注入的用法
在 GNN 里几乎被 backbone 吞掉"。**

> 一句话总括：**"多粒度显式结构有增量价值"这件事在本轮被重复观察到了（线性、XGBoost、GINE 三个
> 方向同号），但"层级/支持度这类关系信息被模型利用"没有证据；增益几乎全部来自"多一组高频粗结构
> 特征"，而 Test 与 Valid 的不一致（例如 `XGB_STRUCT` Valid −0.0038 / Test +0.0445）说明单 seed
> 的 Valid 排序不能当结论。**

---

## 1. 边界与身份声明

* 数据：官方 `ogbg-molhiv` 原始 `raw/*.csv.gz` + 官方 scaffold index。审计 `analysis/data_audit.json`
  全项通过（`all_checks_pass = true`）：三划分互斥且覆盖 41 127 图；物理键计数经 PyG 双写视图去重
  逐图核对一致；历史 XGBoost 的 Train/Valid 行 id 与存储的 Valid/Test 逐分子预测与本轮行 id 逐项相同。
* 环境：本地 `uv`（torch 2.5.1+cu124 / pyg 2.6.1 / xgboost 2.1.3 / sklearn 1.5.2 / rdkit 2026.09.1）；
  远程 `rr` → `res`（A100-SXM4-40GB，见 `analysis/provenance.json`）。
* Test 只以 `EXPOSED_DEVELOPMENT_TEST` 出现：不参与 C 选择、不参与 checkpoint 选择、不参与
  Phase-E 触发判定、不参与词表/统计/标准化拟合。
* 证据分级：历史 XGBoost = `EXACT_MATCH`（预处理与逐分子预测 bit 级复现）；`B*`、`XGB_STRUCT`、
  `G-*` 之间 = `SAME_SPLIT_REFERENCE`（同划分、同评估、不同配方）；历史 KSVD / GINE 数字 =
  `HISTORICAL_REFERENCE`，本轮只在结尾做定位对照，不做严格消融。
* 解释边界：逻辑回归系数是**所选参数化下的拟合量**，不是因果化学贡献；所有"这个修正项关于结构 X"
  的说法都要求在具体分子里检查共现（§4.2 的 12 个家族就是为此做的）。

---

## 2. Phase A — 精确结构发现、Train-only 词表

### 2.1 实例规模（全量 41 127 分子，非抽样）

| family | instances | share |
|---|---|---|
| size-2（全部连通诱导 2 点子图） | 1 129 688 | 20.84% |
| size-3 | 1 610 382 | 29.71% |
| size-4 | 2 559 304 | 47.22% |
| cycle-5（无弦 5 环） | 28 587 | 0.53% |
| cycle-6（无弦 6 环） | 92 524 | 1.71% |
| **合计** | **5 420 485** | 100% |

每分子实例数：中位 119，均值 131.8，最大 1 478。全量枚举 73.4 s（10 worker，`analysis/phaseA_run.json`）。

发现的不同类型数（属性图同构类，不设阈值）：**COARSE 13 915 / FULL 71 674**。

### 2.2 词表冻结（只用 Train 的分子频次与 scaffold 频次）

筛选规则（协议 §3 原文，未改）：COARSE ≥20 Train 分子、≥5 Train scaffold，每族 ≤256；
FULL ≥5 分子、≥3 scaffold、父结构必须保留，每族 ≤1024；按文档频次、scaffold 频次、规范键排序。

| granularity | size-2 | size-3 | size-4 | cycle-5 | cycle-6 | 合计 |
|---|---|---|---|---|---|---|
| COARSE columns | 109 | 256 | 256 | 69 | 64 | **754** |
| FULL columns | 683 | 1024 | 1024 | 438 | 483 | **3 652** |

FULL→COARSE 投影（属性投影后再规范化）检查：**0 个冲突、0 个缺父**。
覆盖（实例级，未缩放）：

| split | COARSE 覆盖 | FULL 覆盖 |
|---|---|---|
| Train | 94.72% | 81.69% |
| Valid | 90.85% | 75.16% |
| Test | 92.81% | 76.78% |

**FULL 词表在 Test 上只覆盖 76.8% 的实例，且 99.2% 的 Test 分子至少含一个词表外 FULL 类型**——
这条数字决定了本轮所有 FULL 相关结论的适用范围。

### 2.3 规范键正确性（本轮补做的关键验证）

早期一次"规范键 vs 全排列暴力"比较报告了 329/800 不一致，看起来像类型同一性出错。实际上两个键
是**不同规范形**（前者只在 1-WL 胞内置换，后者对全部排列取最小），元组值本来就可以不同；正确的
判据是"两种键诱导的划分相同"。`code/verify_canonical_partition.py` 在 160 个分子（n_atoms 6–70，
中位 23，每分子最多 140 个实例，共 17 737 个实例）上检验：

| granularity | 实例 | 规范键类型数 | 暴力键类型数 | 划分相同 | 合并/拆分错误 |
|---|---|---|---|---|---|
| COARSE | 17 737 | 996 | 996 | ✅ | 0 / 0 |
| FULL | 17 737 | 4 027 | 4 027 | ✅ | 0 / 0 |

元组完全相等率只有 0.44（COARSE）/ 0.72（FULL），这正是"不同规范形"的预期结果。
`analysis/canonical_partition_check.json`；同一判据已固化进 `tests/test_structure_core.py`。

### 2.4 设计矩阵不变量（机器校验，`analysis/vocab_verification.json`）

* 每分子 `sum_f F_f ≤ C_p`（子计数不超过父计数）：0 违反；
* `C_null`（父子映射打乱）保持每分子总量，最大偏差 4×10⁻¹⁴；列合计**不**保持（按设计）；
* 父子打乱改变 96.2% 的赋值；支持权重打乱改变 95.9%；
* 计数值缩放 `count / sqrt(max(n_atoms,1))`；支持权重均值 1.564（截断在 [0.25, 4.0]）。

---

## 3. Phase B — 固定表示下的 8 个线性臂

配方固定：`z = b + M·w + C·u + F·v`，BCE + L2，`class_weight='balanced'`，liblinear，`max_iter=2000`，
C ∈ {0.03, 0.3, 3.0}，按 Valid ROC-AUC 选 C（并列取小 C）。**8 个臂全部选中 C=0.03**（网格下界，
说明这个表示下正则越强越好，未出现上界饱和）。OGB 官方 evaluator 与 sklearn 逐臂一致（`abs_diff = 0`）。

| arm | 输入块 | Valid ROC-AUC | Valid AP | Test ROC-AUC | Test AP | Δ Valid vs B0 |
|---|---|---|---|---|---|---|
| `B0` | M | 0.8114 | 0.2610 | 0.7478 | 0.2801 | — |
| `B1` | M + C | 0.8188 | 0.3349 | 0.7579 | 0.2752 | +0.0074 |
| `BF` | M + F | 0.8250 | 0.3404 | 0.7618 | 0.2953 | +0.0136 |
| `B2` | M + C + F | **0.8283** | 0.3597 | 0.7688 | 0.2862 | +0.0169 |
| `B2-PNULL` | M + C + F，父子映射整体打乱 | 0.8340 | 0.3393 | 0.7667 | 0.2883 | +0.0226 |
| `B2-U` | M + C + F，FULL 共享单一权重 | 0.8271 | 0.3533 | 0.7663 | 0.2843 | +0.0157 |
| `B3` | M + C + F，支持度加权正则 `1/√a_f` | 0.8296 | **0.3644** | **0.7724** | 0.2857 | +0.0182 |
| `B3-NULL` | 同上，`a_f` 打乱 | 0.8290 | 0.3597 | 0.7690 | 0.2890 | +0.0176 |

配对 scaffold bootstrap（1000 次重抽 scaffold 组，seed 0，全部 1000 次有效；`analysis/bootstrap_contrasts.json`）：

| 对比 | Δ Valid [95% CI] | Δ Test [95% CI] |
|---|---|---|
| `B1 − B0` | +0.0073 [−0.0054, +0.0205] | +0.0101 [−0.0016, +0.0210] |
| `B2 − B1` | +0.0095 [−0.0026, +0.0223] | **+0.0110 [+0.0024, +0.0198]** |
| `B2 − BF` | +0.0032 [−0.0056, +0.0133] | +0.0071 [−0.0013, +0.0158] |
| `B2 − B2-PNULL` | −0.0057 [−0.0138, +0.0031] | +0.0021 [−0.0052, +0.0095] |
| `B3 − B2` | +0.0013 [−0.0047, +0.0081] | +0.0035 [−0.0011, +0.0086] |
| `B3 − B2-U` | +0.0025 [−0.0057, +0.0118] | +0.0061 [−0.0007, +0.0133] |
| `B3 − B3-NULL` | +0.0006 [−0.0049, +0.0067] | +0.0034 [−0.0008, +0.0078] |

读法：**"加结构"这件事（B0→B1→B2）方向稳定，但只有 `B2 − B1` 在 Test 上 CI 不跨 0；
"关系结构被利用"这件事（PNULL / U / NULL 三个对照）没有任何一条能分开。**

---

## 4. Phase C — 结构 ↔ 任务分析（不训练新模型）

### 4.1 支持度四等分（FULL 类型，边界 = Train scaffold 频次四分位）

| bin | m_f 范围 | 类型数 | a_f 均值 | Train 实例占比 | Valid 占比 | Test 占比 |
|---|---|---|---|---|---|---|
| 0（最低） | 3 – 14.5 | 897 | 3.55 | 0.37% | 0.48% | 0.55% |
| 1 | 15 – 86.5 | 919 | 1.47 | 3.27% | 2.86% | 3.07% |
| 2 | 87 – 202.5 | 919 | 0.82 | 9.29% | 8.83% | 8.69% |
| 3（最高） | 203 – 9 158 | 917 | 0.46 | 68.77% | 62.99% | 64.46% |

（占比 = 该档保留类型的实例数 / 该划分全部诱导实例数，所以四档之和 = FULL 覆盖率。）

有效系数（B2 的 `v`、B3 折算回原 F 列的 `w/√a`）：

| bin | mean\|v\| B2 | mean\|v\| B3 | B3/B2 | B3 压缩的类型比例 | L2 质量 B2 | L2 质量 B3 |
|---|---|---|---|---|---|---|
| 0 | 0.0132 | 0.0040 | **0.30** | 99.7% | 0.50 | 0.06 |
| 1 | 0.0391 | 0.0293 | 0.75 | 90.8% | 2.71 | 1.73 |
| 2 | 0.0615 | 0.0722 | 1.17 | 21.7% | 5.74 | 8.15 |
| 3 | 0.0863 | 0.1570 | **1.82** | 7.1% | 11.40 | 39.78 |

即：**支持度加权确实把系数质量从长尾搬到头部（最高档 L2 份额 56.0% → 80.0%，总量 20.35 → 49.71），
但 Valid 只换来 +0.0013。**"压缩低支持系数"这个机制成立，"压缩低支持系数有用"这个结论不成立。

### 4.2 十二个 COARSE 家族（纯 Train 判据挑选）

三类判据各 4 个：A = 多子结构中 Train 文档频次最高；B = 该家族 FULL 修正项在 Train 分子上标准差最大；
C = 子结构 Train scaffold 频次 max/min 差异最大（≥3 子）。全部 12 个家族的规范键、代表子类型、
Train 分子/scaffold/正负支持、B2/B3 系数、真实分子位置见 `analysis/case_families.json`，
平面表 `analysis/case_families.csv`，图 36 张（每个家族 2–3 个不同分子、真实原子位置高亮）
在 `analysis/figures/case*.png`。

| case | 类别 | family | 节点签名 | Train df | Train scaffolds | 子类型数 |
|---|---|---|---|---|---|---|
| 0 | A | size-2 | `aBar+5x2`（芳香 C–芳香 C） | 27 907 | 9 521 | 3 |
| 1 | A | size-3 | `aBar+5x3` | 27 455 | 9 366 | 4 |
| 2 | A | size-2 | `aB+5x2`（芳香 C–芳香 B） | 26 210 | 9 040 | 4 |
| 3 | A | size-4 | `aBar+5x4` | 26 028 | 8 988 | 4 |
| 4 | B | size-3 | `aB+5x1 bBar+5x2` | 21 679 | 7 811 | 4 |
| 5 | B | size-4 | `aB+5x4` | 13 416 | 5 213 | 4 |
| 6 | B | size-4 | `aB+5x1 bBar+5x3` | 20 544 | 7 450 | 4 |
| 7 | B | size-3 | `aBar+5x2 bCar+5x1` | 11 348 | 4 327 | 4 |
| 8 | C | size-2 | `aB+5x1 bN+5x1` | 19 593 | 6 752 | 4 |
| 9 | C | size-2 | `aB+5x1 bC+5x1` | 18 105 | 6 482 | 4 |
| 10 | C | size-2 | `aB+5x1 bN+5x1`（不同 bond 属性档） | 18 608 | 6 656 | 4 |
| 11 | C | size-2 | `aBar+5x1 bC+6x1` | 2 440 | 1 141 | 4 |

作图时顺带做了一个数据一致性检查：**RDKit 重新解析 SMILES 得到的原子序数序列与原始 OGB
node-feat 原子序数列逐位相同（36/36 张图全部为真）**，所以图里高亮的原子编号就是枚举用的节点编号
（`case_families.json → rdkit_atom_order_matches_raw_features = true`）。

### 4.3 子群泛化（占比边界由 Train 定义；Test 标 `EXPOSED_DEVELOPMENT_TEST`）

占比 = 分子内"低支持档 FULL 实例 + 词表外 FULL 实例"占全部诱导实例的比例。

| 占比区间 | n | 正例 | B0 | B1 | B2 | B3 | XGB_STRUCT | G-STRUCT-U |
|---|---|---|---|---|---|---|---|---|
| 0.00–0.25 | 2 327 | 89 | 0.8334 | 0.8352 | 0.8417 | 0.8419 | **0.8854** | 0.8314 |
| 0.25–0.50 | 1 421 | 36 | 0.5583 | 0.5958 | 0.6146 | 0.6228 | **0.6538** | 0.6431 |
| 0.50–0.75 | 310 | 5 | 0.5580 | 0.5547 | 0.5967 | 0.6032 | 0.5481 | 0.6177 |
| 0.75–1.00 | 55 | 0 | `INSUFFICIENT_SUPPORT` | — | — | — | — | — |
| 至少一个词表外 FULL | 4 081 | 130 | 0.7482 | 0.7577 | 0.7687 | 0.7723 | 0.8064 | 0.7698 |

三点必须一起读：(a) 难度差异主要由**分子复杂度**驱动（占比高的分子普遍更大），这不是一个
干净的难度控制；(b) 在 0.25–0.50 这一档，`G-STRUCT-U`（0.6431）明显好于 `G-M`（0.5580），
说明结构分支在"中等陌生度"上最有话说；(c) 0.50–0.75 只有 5 个正例，只当方向性证据。

### 4.4 传导到下游的检查（不做新模型）

* §4.3 的分组变量（低支持/未见实例占比）与分子规模强相关（大分子枚举出的实例更多、其中长尾类型也更多），
  所以组间 AUC 差异**不能**解释成纯结构效应；本轮没有做规模匹配的子群对照。
* case 4/5/6/7 是按"B2 修正项在 Train 上标准差最大"这条判据选出来的（§4.2 B 类），
  因此它们只能证明"FULL 修正项确实在分子间变化"，不能当作独立发现。

---

## 5. Phase D — 传统强基线 + 结构

历史 XGBoost（ECFP4 2048 二值 + 217 RDKit 描述符，500 树 / depth 5 / lr 0.03 / subsample 0.8 /
colsample 0.8 / λ=5 / hist / seed 0）**不重训**：本轮从冻结缓存重建特征矩阵，复现
`features_sha256 / keep_sha256 / median_sha256 / n_used_columns = 2258`
（`EXACT_MATCH`），一次验证性重训把 Valid/Test 逐分子预测复现到 **max|Δ| = 0.0**
（Valid 0.836692 / Test 0.762440）。

新模型只有 `XGB_STRUCT` = 上述 2 258 列（未改动）+ C(754) + F(3 652) = 6 664 列，同一配方、同一 seed、
dense float32（协议 §6 说明为什么不能用稀疏 + `missing=NaN`：未存储的 0 会被当成缺失，实测会改变
96% 的预测行）。

| 模型 | Valid ROC-AUC | Valid AP | Test ROC-AUC | Test AP |
|---|---|---|---|---|
| 历史 XGBoost（`EXACT_MATCH`） | **0.8367** | 0.3895 | 0.7624 | 0.3210 |
| `XGB_STRUCT` | 0.8329 | **0.4331** | **0.8070** | **0.3590** |
| Δ（配对 scaffold bootstrap） | −0.0038 [−0.0268, +0.0222] | **+0.0436 [+0.0005, +0.0845]** | **+0.0445 [+0.0193, +0.0697]** | +0.0380 [−0.0020, +0.0744] |

增益归属（`gain` 分解）：结构块 54.1%、历史块 45.9%；4 406 个结构列中 821 个被树用到。
单列 gain 前两名是 FULL（size-4 的 `aBarRs+5deg3x2 bBarRsH1+5deg3x1 cPsp+5deg4x1`，df 529；
cycle-5 的 `aBRspH1+5deg4x3 bBRspH2+5deg4x1 cNRsp+5deg2x1`，df 364），第三名是 COARSE
`aBar+5x3 bP+5x1`（df 3 213）—— 高频粗结构仍然在树干上，细结构提供的是局部分裂。

**Valid 与 Test 方向相反（−0.0038 / +0.0445）是本轮最强的"不要只看 Valid"的证据**：两个划分
差 0.04 AUC，而 Valid 侧 CI 宽达 0.05，说明 4 113 个分子 / 81 个正例的 Valid 不足以给这类对比排序。

---

## 6. Phase E — 条件性 plain-GINE 验证

触发（协议 §7 的资源触发器，非显著性检验）：`B1` 比 `B0` Valid +0.0074 ≥ 0.005 → 触发；
输入用 C+F（`B2` 优于 `B1`）；`G-STRUCT-W` 不成立（`B3` 对 `B2`/`B2-U`/`B3-NULL` 的最小 Valid 间隔
= min(0.0013, 0.0025, 0.0006) = 0.0006 < 0.005，跳过）。

两臂共用同一 backbone 初始化（`initial_backbone_param_sha256` 完全相同 `b638bfc3…`）、同一 batch 顺序、
seed 0、30 epoch，每 epoch 全量评 Train/Valid/Test，checkpoint 只按 Valid 选。

| arm | 结构输入 | 参数量 | 选中 epoch | Valid ROC-AUC | Test ROC-AUC | Valid AP | Test AP | 训练秒 |
|---|---|---|---|---|---|---|---|---|
| `G-M` | 仅 Morgan 图级分支 | 39 430 | 3 | 0.8038 | 0.7591 | 0.2858 | 0.3256 | 342 |
| `G-STRUCT-U` | Morgan + C+F（零初始化，均匀惩罚） | 43 836 | 7 | **0.8142** | **0.7694** | **0.3803** | 0.3198 | 915 |

配对 scaffold bootstrap：`G-STRUCT-U − G-M` Valid +0.0104 [−0.0160, +0.0389]，
Test +0.0103 [−0.0181, +0.0362]；`G-M − 历史 XGBoost` Valid −0.0329 [−0.0808, +0.0127]。

定位（`HISTORICAL_REFERENCE`，本轮未复跑）：历史 KSVD 5-seed Valid 0.7946 / Test 0.7519、
历史 GINE-h64 Valid 0.7700 / Test 0.7527。本轮 `G-STRUCT-U`（0.8142 / 0.7694）高于这两个历史参照，
但配方不同（不同的 backbone 细节与结构分支），**不能当作"结构分支优于 KSVD"的消融结论**。

另外：`G-M` 与 `G-STRUCT-U` 都在 epoch 3–7 附近取得 Valid 最优，Train AUC 已到 0.90–0.93 后继续爬升
到 0.97 而 Valid 下滑 —— 本任务上这个 3 层 GINE 的容量与正则并不匹配，Valid 选择的 epoch 处
Test 也随之偏早。这是既有限制，不是本轮引入的。

---

## 7. 认识更新表

| # | 之前认识（进入本轮时的判断） | 本轮证据 | 更新后认识 | 置信度 / 边界 |
|---|---|---|---|---|
| 1 | "显式结构特征在 MolHIV 这种小数据 + scaffold split 上最多给 1–2 个点。" | 线性 +0.017 Valid / +0.021 Test；XGBoost Test +0.0445；GINE +0.010 | **增量真实存在且方向一致，但量级强烈依赖读出头**：树模型能吃下 4 个点，线性只吃 2 个点，3 层 GINE 只吃 1 个点且不可分辨。 | 中高。三方向同号；但单 seed，且 XGB 的 Valid/Test 相反。 |
| 2 | "细粒度子结构比粗粒度更有信息量，粗粒度是冗余的中间层。" | `BF` < `B2`（Test +0.0071）；`B1→B2` 是唯一 Test CI 不跨 0 的相邻对比；`B2−BF` CI 跨 0 | **粗粒度不是冗余，它是细粒度的支架**：只给 FULL（`BF`）明显更差。但"支架"的增益本身也只在 Test 上勉强可分辨。 | 中。方向稳定，幅度小。 |
| 3 | "把父子关系/支持度当作先验能提升泛化（层级共享应该有用）。" | `B2-PNULL` ≈ `B2`（Valid 甚至更高）；`B2-U` ≈ `B2`；`B3 − B3-NULL` = +0.0006 Valid / +0.0034 Test，CI 跨 0 | **在这个表示下，真实层级与支持度没有被模型利用**：打乱父子、共享权重、打乱支持权重都不掉点，说明增益来自"多了一组结构化计数特征"，不是"层级被理解了"。 | 高（三个独立对照同向）。注意这是"这个用法下"的结论，不是"层级永远无用"。 |
| 4 | "低支持结构的系数应该被压缩，否则是过拟合噪声。" | B3 把低支持档 |v| 压到 0.30×、99.7% 的类型被压缩；高支持档放大到 1.82×；最高档的 L2 份额 56.0% → 80.0% | **机制层面成立（压缩确实发生了），价值层面不成立（Valid +0.0013）。** "压缩了噪声"与"压缩带来收益"是两件事，本轮只证明前者。 | 高（机制），低（价值）。 |
| 5 | "结构增益会迁移到含未见结构的分子上，是解决分布外的手段。" | Test 4 081/4 113 分子含词表外 FULL；FULL 覆盖率 76.8%；子群 0.25–0.50 档 B0→B3 0.5583→0.6228，但 0.50–0.75 档只有 5 个正例 | **结构特征主要改善"中等陌生度"分子，且在低覆盖分子上没有可判定证据**；未见结构是常态，不是边角。 | 中低。子群与分子复杂度混杂，且最难一档判 `INSUFFICIENT_SUPPORT`。 |
| 6 | "Valid 排序可以用来选结构方案（本轮协议也是这么做的）。" | `XGB_STRUCT` Valid −0.0038 / Test +0.0445；`B2-PNULL` Valid 最高但 Test 不是最高；`B3` Valid 只高 0.0013 而 Test 高 0.0035 | **在 4 113/81 的 Valid 上做 8 选 1 的排序基本是噪声**：Valid 的 CI 宽度（±0.02–0.03）大于被比较的差异（0.001–0.01）。后续同规模任务应预先声明"Valid 只用于排除崩溃，不用于细排序"，或直接多 seed。 | 高（本轮内部一致）。这是方法论更新，不是数据结论。 |

---

## 8. 资源、环境、提交、数据哈希

全部数字由 `code/record_provenance.py` 从运行产物里读出，汇总在 `analysis/provenance.json`。

| 项 | 预算 | 实际 |
|---|---|---|
| 墙钟 | 4 h 硬上限 | 约 1.5 h（首个产物 12:02 → 收尾 13:10 本地；含 3 次 GPU 冒烟/计时尝试） |
| A100 GPU | 4 GPU-hours | **0.349 h**（`G-M` 342 s + `G-STRUCT-U` 915 s，两臂并行在不同节点） |
| 新神经训练单元 | ≤3 | **2**（各 1 seed，30 epoch） |
| CPU 结构准备 | ≤60 min | 1.38 min（枚举 73.4 s + 词表 9.5 s） |
| 线性实验 | ≤45 min | 2.23 min（8 臂 × 3 C） |
| 新 XGBoost | ≤30 min | 4.14 min（含历史复现重训 107 s） |
| 额外 seed / 词表扩张 / 架构搜索 | 禁止 | 无 |

提交与环境：

* 代码提交：`2c1cb51cc2b1`（Phase E 部署修订）→ 本轮最终交付提交见 `git log`；每份 `analysis/*.json`
  里都记录了产生它的 commit 与 `git_state`。
* 远程执行：`res` / `a100-2`，`NVIDIA A100-SXM4-40GB`，CUDA 12.4，torch 2.5.1+cu124，
  python 3.12.14（`analysis/phaseE/*_seed0.json → provenance`）。
* 数据哈希：原始 `raw/*.csv.gz`、`split/scaffold/*`、`mapping/mol.csv.gz`、历史 manifest / 特征 /
  逐分子预测，以及本轮全部 `data/*.npz`、`analysis/*.json|csv|npz` 的 SHA-256 全部列在
  `analysis/provenance.json → inputs / artifacts`。
* 关键产物哈希：
  * `data/phaseA_instances.npz` = `4f6e703251eaa27abfec46fbe63062b827861b7de9f735fded0ae739aba6e4e3`
  * `data/design_CF.npz` = `a30e18ebffb3c2b1a06c50f18048a2614268d62d3552a39b7842bb8d3ca480b7`
  * Phase-E 冻结输入 `bundle/phaseE_inputs.npz` = `92ed8c3bc009803bd1213f5061f6ea42f231cb04f7512d04d6d097a1e4ad3841`
* 远程 provenance 里的 `dirty_tracked = true` 只对应"结果目录尚未提交"（`analysis/phaseE/` 为
  untracked），代码本身是干净提交；每次 `rr run` 都从已提交修订部署。

---

## 9. 复现

```bash
cd tracks/structural-sufficiency/molhiv_multigranular_structure_sharing_v0

PYTHONPATH=code uv run python -m msds_phaseA --workers 10 --chunk 250   # 5.42M 实例, ~73 s
PYTHONPATH=code uv run python -m run_phaseA_vocab                        # 词表冻结, ~10 s
PYTHONPATH=code uv run python -m verify_canonical_partition              # 规范键划分检验, ~5 s
PYTHONPATH=code uv run python -m run_phaseB_linear                       # 8 臂 × 3 C, ~134 s
PYTHONPATH=code uv run python -m run_phaseD_xgb                          # ~248 s
PYTHONPATH=code uv run python -m build_phaseE_bundle                     # ~10 s
PYTHONPATH=code uv run --group lipo python -m run_phaseC_analysis        # 表/图, ~89 s
PYTHONPATH=code uv run python -m record_provenance                       # ~2 s

# GPU（必须从已提交修订部署；rr 拒绝 dirty worktree）
R=tracks/structural-sufficiency/molhiv_multigranular_structure_sharing_v0
rr run res phaseE-gm --pool res-gpu1 --result $R/analysis/phaseE \
  -- env PYTHONPATH=$R/code python -m run_phaseE_gine --arm G-M --epochs 30 \
        --out $R/analysis/phaseE/G-M_seed0.json
rr run res phaseE-gstruct-u --pool res-gpu0 --result $R/analysis/phaseE \
  -- env PYTHONPATH=$R/code python -m run_phaseE_gine --arm G-STRUCT-U --epochs 30 \
        --out $R/analysis/phaseE/G-STRUCT-U_seed0.json
rr pull phaseE-gm --host res && rr pull phaseE-gstruct-u --host res

PYTHONPATH=code uv run python -m pytest tests -q                          # 9 passed
```

产物清单：`analysis/` 下 `support_bins.csv`、`coefficient_bins.csv`、`case_families.{json,csv}`、
`figures/case*.png`（36 张）、`subgroup_generalisation.csv`、`bootstrap_contrasts.json`、
`predictions/*.npz`（12 个模型逐分子 Valid/Test logit）、`provenance.json`、
`canonical_partition_check.json`。

---

## 10. 工程偏差与本轮限制

| # | 事件 | 处理 |
|---|---|---|
| 1 | 首轮暴力参考检验在 222 原子分子上跑 `C(222,6)` 被中止 | 参考检验限制在 `n_atoms ≤ 18`；穷举检验固化为有界单元测试 |
| 2 | 诱导环枚举器最初把"根不是闭合点"的环当作有弦而漏掉 | 统计产出前修复；`phaseA_instances.npz` 哈希为修复后产物，重跑逐字节一致 |
| 3 | 早期"规范键 vs 暴力键"报 329/800 不一致 | 是不同规范形的元组比较错误，非类型错误；改为划分等价检验（§2.3），17 737 实例上划分完全一致 |
| 4 | xgboost 稀疏输入把未存储的 0 当缺失 | 显式测量后改用 dense（协议 §6） |
| 5 | Phase E 首跑 `CUDA error: device-side assert` | 根因：PyG `Data.__inc__` 会把任何含 `index` 的属性在 batching 时加上累计节点数，`graph_index` 因此越界。改名 `gid` 并加注释；不影响任何已报告结果 |
| 6 | 本地 RDKit 绘图后端缺 `libXrender` | 改用 `rdDepictor` 坐标 + matplotlib 自绘（图上原子编号与枚举节点编号一致性已逐图校验） |
| 7 | 早期 Phase E 冒烟（2 次）与计时（1 次）失败/中止 | 未产生被引用的模型；GPU 消耗 < 0.1 A100-hour，不计入训练单元 |

限制（按重要性）：

1. **单 seed**。所有 CI 都是"固定预测的抽样不确定性"，不是训练方差；`B*` 之间 0.001–0.01 的差异
   在单 seed 下不可判定。
2. **Valid 太小**（4 113 分子 / 81 正例），8 选 1 的 Valid 排序接近噪声（§7 第 6 条）。
3. **FULL 词表 Train-only 导致 Test 覆盖率只有 76.8%，99.2% 的 Test 分子含未见 FULL 类型**，
   所以本轮"FULL 的价值"其实是"高频 FULL 子集的价值"。
4. **cycle-5/6 只覆盖无弦环**，不是全部 5/6 点连通诱导子图；结论不能外推到"5/6 点结构"。
5. **COARSE/FULL 列是计数而非类型存在性**，同一分子重复出现同一类型会累加，`1/√a` 的重加权
   与计数缩放会互相影响；本轮未做这种交互的分离。
6. **XGBoost 的历史对比是 `EXACT_MATCH` 但配方固定**（500 树），`XGB_STRUCT` 多出的 4 406 列
   在同等树数下必然分走分裂预算，"结构占 54.1% gain"不等于"结构值 54.1% 的 AUC"。
