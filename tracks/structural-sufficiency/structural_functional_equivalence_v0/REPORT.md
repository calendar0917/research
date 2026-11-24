# SFEC v0 — 结构功能等价与组合规则实验报告

轮次：`structural_functional_equivalence_v0`  
目录：`tracks/structural-sufficiency/structural_functional_equivalence_v0/`  
状态：**已完成**（CPU only；未使用 GNN / Transformer / MLP / embedding / teacher / distillation / 预训练模型；未使用模型预测作为标签）  
协议：`PROTOCOL.md`（在标签分析前冻结）；配置：`configs/sfec_v0.json`

---

## 首页：八个必需问题的直接回答

### Q1. 交叉结构支持是否足够？哪个数据集能进入 P1/P2/P3？
**PCQM4Mv2 1M 前缀足够；ZINC 不足。** PCQM 1M 有 4,824,824 个切割单元、3,166,455 个 C 上下文、987,596 个分子；primary 64-S 二部图有 3 个连通分量（最大 3,010,213 个顶点）、2-core 419,353、cycle rank 777,293；支持 1,602 对结构对，主评估集有 96,991 个分子不相交的 pair 观测。ZINC 有 66,494 个单元、51,574 个训练上下文，但 primary 二部图 2-core 只有 6、cycle rank 仅 1，支持对为 **0**，主评估 pair 观测为 **0**。

### Q2. P1：发现集的结构响应能否在评估上下文复用？
**PCQM：`PARTIAL_RESPONSE_REUSE`。** context-equal 的主评估（1,602 对支持结构）上，PAIR MAE **0.23657**，ZERO MAE **0.26554**，相对增益 **10.91%**；ZERO−PAIR 的 95% 上下文配对 bootstrap 区间 **[0.02725, 0.03071]**（不含 0）；符号一致率 **64.7%**。pair-equal 下为 0.26215 vs 0.29750，增益 **11.88%**。  
**ZINC：`INSUFFICIENT_CROSSED_SUPPORT`**，没有可评估的支持对。

### Q3. P1：功能等价候选是否稳定？覆盖多少？
**有信号但很稀疏。** 在发现集冻结的 `tau=0.10` 下只有 **3 个候选**，覆盖 1,602 个支持对的 **0.19%**；它们共 415 个评估观测，**77.8%** 的评估残差落在 `tau` 内。`tau=0.20` 下有 **151 个候选**，覆盖 **9.43%**，评估残差落在 `tau` 内的比例为 **83.1%**。`tau=0.05` 下无候选。功能等价不是一张宽泛的替换字典。

### Q4. P2：SHARED 加性组合是否支持未观测替换？
**主判定：`COMPOSITION_SCREEN_ONLY`（未达到 15% 预注册门槛）。** 自然未观测替换对为 **0**（primary S 在 1M 前缀中共同出现太充分；所有评估 pair 都有直接发现观测）。受控遮蔽有 **16 个适用 pair**、18,584 个评估观测：SHARED MAE **0.16924**，ZERO **0.17083**，相对增益仅 **0.93%**；bootstrap 区间 **[1.90e-5, 0.00321]**（点估计为正，但远低于 15%）；PAIR-FULL 参考 MAE **0.16209**，SHARED 仍差 **0.00715**。按 pair 等权的 macro 增益也只有 **2.31%**。  
补充（不改变 P2 判定）：遮蔽后拟合 LOCAL 条件修正，MAE **0.15423**，相对 SHARED 改善 **8.87%**，相对 ZERO 改善 **9.72%**，覆盖 99.29%。这说明全局加性项不是主要载体，条件规则才是。

### Q5. P3：LOCAL 上下文规则是否优于 SHARED 和匹配 NULL？
**方向一致，但总体未过 10% 门槛：`CONTEXT_RULE_SCREEN_ONLY`。** 837 个 q 组中 **114 个**有足够支持；修正覆盖率 **99.00%**（真实 q 与 NULL 都能启用的共同子集 98.67%）。全体评估上：SHARED MAE **0.27145**，LOCAL **0.24881**，相对改善 **8.34%**（SHARED−LOCAL 区间 **[0.02105, 0.02425]**）；LOCAL_NULL **0.27319**，LOCAL 相对 NULL 改善 **8.93%**（LOCAL−NULL 区间 **[-0.02596, -0.02279]**，负值表示 LOCAL 更好）。在发现集冻结的“高非可加性”四分位子集（442 对、12,223 观测）上，LOCAL 相对 SHARED 改善 **10.76%**（区间 **[0.04476, 0.05457]**），超过 10%，但这是子集结果，不能替代总体判定。匹配 NULL 没有带来收益，说明真实 q 分组本身有信息。

### Q6. ZINC 是复现、反驳，还是结构阴性对照？
**结构阴性对照。** ZINC 的 primary 二部图几乎是森林：44 个连通分量、最大 20,194、2-core 6、cycle rank 1；没有任何 pair 同时满足“≥10 个发现上下文”和“≥5 个主评估上下文”。P1/P2 为 `INSUFFICIENT_CROSSED_SUPPORT`，P3 为 `CONTEXT_SUPPORT_INSUFFICIENT`。因此 ZINC 不能用来检验交叉响应复用，也不构成对 PCQM 结论的反驳。

### Q7. 官方 PCQM Valid 是否支持外部迁移？是否泄漏 Test？
**支持，且未读 Test 标签。** 在 36,952 个有冻结发现中位数的 Valid pair 观测上：PAIR MAE **0.29795**，ZERO **0.34791**，skill **14.36%**，符号一致率 **67.4%**。其中“未见精确 C 上下文”6,765 个观测，skill **5.59%**；“已见精确 C 上下文”30,187 个观测，skill **15.53%**。7,356 个 Valid 单元（来自 1,819 个分子）的完整图与 Train 重复，已从主迁移统计中排除。官方 Test 标签从未读取，Test 结构也未构建。

### Q8. 身份、重组、标签、划分和工程对照是否足够可信？
**在已记录的限制内可信。** PCQM 图级重组检查 57,873 个抽样切割，**0 个图失败**；立体化学精确率 91.1%（5,128 个不匹配，单独报告）。ZINC 20,000 个切割的精确证书重组 **0 失败**。1,000/1,000 个抽样 PCQM SMILES 与标签和原始 CSV 一致；1M 行全部唯一且属于官方 Train；发现/评估分子重叠为 **0**；primary 评估上下文之间不共享分子。10,557 个重复 PCQM 分子携带的标签最大相差 **4.94 eV**，它们是同一个身份但构成标签噪声。五个 FP64 工程对照全部通过（`checks.json` 的 `all_ok=true`）。

---

## 1. 判定总表

| 数据集 | 支持门槛 | P1 | P2 | P3 |
|---|---|---|---|---|
| PCQM4Mv2 Train 1M 前缀 | `p1_pass=true`；1,602 支持对；96,991 评估 pair 观测 | `PARTIAL_RESPONSE_REUSE` | `COMPOSITION_SCREEN_ONLY` | `CONTEXT_RULE_SCREEN_ONLY` |
| ZINC 官方子集 | `p1_pass=false`；0 支持对；0 评估 pair 观测 | `INSUFFICIENT_CROSSED_SUPPORT` | `INSUFFICIENT_CROSSED_SUPPORT` | `CONTEXT_SUPPORT_INSUFFICIENT` |

一句话结论：**PCQM 1M 足以证明“响应方向可以部分复用、稳定等价候选存在但覆盖很窄”；不足以证明“全局加性 SHARED 可以广泛做未观测替换”；LOCAL 条件规则有真实信号，但总体未达到预注册的 10% 实用门槛，主要在高非可加性子集上过线。ZINC 因结构上几乎无环，不能进入交叉检验。**

---

## 2. 数据、切割、身份与支持矩阵

### 2.1 固定输入

| 项目 | 选择 |
|---|---|
| ZINC | 官方 PyG `ZINC(subset=True)`；train 10,000 / valid 1,000 / test 1,000；原始节点/键类型、原始 `logP_SA_cycle_normalized` |
| PCQM4Mv2 | 官方 `data/ogb/pcqm4m-v2`；官方 Train/Valid；原始 `homolumogap`（eV）；Test 标签不读 |
| PCQM 行计划 | 缓存顺序前 100,000 → 前 1,000,000；扩展只依据结构支持，不依据标签结果 |
| 身份 | ZINC：pynauty 彩色关联证书；PCQM：RDKit 规范异构片段 SMILES + 原始键位 dummy；**WL 哈希从不作为身份** |

### 2.2 切割规则

- 只切原始图中的单键桥；ZINC 单键代码 = 1（已通过价键求和验证）。
- 小侧 `1 <= |S| <= 8`，大侧 `|C| >= 6`，且 `|C| >= 2|S|`。
- 两侧等大时按规范 key 字典序确定小侧。
- 多组件分子排除并计数。
- PCQM 1M：4,824,824 个切割；解析失败 11；多组件 132；使用行 999,857（唯一分子 987,596）；图级重组失败 0；立体化学重组失败 5,128（抽样 57,873）。
- PCQM Valid：322,514 个切割；解析失败 0；多组件 109；使用行 73,436（唯一分子 72,181）；图级重组失败 0；立体化学失败 1,404。
- ZINC：66,494 个切割；20,000 个精确证书重组 0 失败。

### 2.3 支持矩阵

| 指标 | PCQM 100k | PCQM 1M | ZINC |
|---|---:|---:|---:|
| 切割单元 | 481,992 | 4,824,824 | 66,494 |
| C 上下文 | 403,413 | 3,166,455 | 61,796 |
| S 类型（全量） | 2,707 | 6,420 | 2,576 |
| 分子 | 99,849 | 987,596 | 11,994 |
| 单元覆盖率 | 88.94% | 88.996% | 78.14% |
| 分子覆盖率 | 99.13% | 99.165% | 99.21% |
| 上下文覆盖率 | 92.05% | 96.458% | 76.98% |
| primary 二部图最大分量 | 367,040 | 3,010,213 | 20,194 |
| primary 二部图分量数 | 4 | 3 | 44 |
| primary 2-core | 13,296 | 419,353 | 6 |
| primary cycle rank | 14,880 | 777,293 | 1 |
| 支持 pair（≥10 发现、≥5 评估） | 37 | 1,602 | 0 |
| 主评估 pair 观测 | 381 | 96,991 | 0 |

**规模决策**：100k 时只有 37 个支持对、381 个评估 pair 观测，支持偏薄；按预先登记的行计划扩展到 1M。扩展依据只有支持统计和吞吐量，没有使用任何标签结果。1M 的支持已经充足，因此没有继续扩展到完整 3,378,606 行 Train。

### 2.4 上下文划分

- `h(C) = SHA256("0|" + C_key)` 前 16 个十六进制字符 `% 100`；`<80` 为发现，`>=80` 为评估。
- 所有出现在任何评估上下文中的分子身份都从发现集中整体删除。
- 主评估上下文按 `h(C)` 升序贪心选择，要求分子身份互不相交，且上下文内至少有两个 primary S。
- PCQM 1M：发现上下文 2,531,352；评估上下文 635,103；主评估上下文 417,064；发现/评估分子重叠 **0**。
- ZINC：发现 41,191；评估 10,383；主评估 6,566；重叠 **0**。

---

## 3. P1：响应稳定性、迁移与功能等价

### 3.1 发现集的响应分布

PCQM 1M 的发现集 pair 差异：

- `delta` 中位数 **−0.01905**，MAD **0.16599**；
- `|D|`（同一 pair 的上下文间差异）中位数 **0.32654**，p90 **1.14832**，均值 **0.48241**；
- 抽样 431,780 个上下文对（来自 1,603 个 pair；全量可能 117,860,358 个）；
- 发现集 `mu` 的绝对值中位数 **0.09388**，方向一致率中位数 **0.444**。

解释：替换响应本身有较大的上下文波动；发现集中位效应普遍偏小。这为“方向可复用、但稳定等价候选少”提供了直接解释。

### 3.2 主评估迁移

主评估 = 1,602 个支持对，96,991 个 pair 观测（38,768 个评估上下文）。指标为 `delta` 的 MAE。

| 加权 | ZERO MAE（95% CI） | PAIR MAE（95% CI） | 相对增益 | ZERO−PAIR（95% CI） | 符号一致率 |
|---|---:|---:|---:|---:|---:|
| context-equal（主） | 0.26554 [0.26218, 0.26901] | **0.23657** [0.23382, 0.23925] | **10.91%** | **0.02897** [0.02725, 0.03071] | **64.7%** |
| pair-equal | 0.29750 [0.29182, 0.30334] | **0.26215** [0.25753, 0.26670] | **11.88%** | **0.03434** [0.03286, 0.03789] | 64.7% |

scaffold-cluster bootstrap（5,039 个 scaffold，按 scaffold 重采样）：ZERO−PAIR = **0.03534**，95% CI **[0.02836, 0.03828]**，方向与上下文 bootstrap 一致。

判定：**`PARTIAL_RESPONSE_REUSE`**。PAIR 明确优于 ZERO，但稳定候选覆盖不足，不能升级为 `REUSABLE_RESPONSE_SIGNAL`。

### 3.3 功能等价候选（发现集冻结，评估集只读）

`tau=0.10`，要求至少 10 个发现上下文且 `P80_C(|delta−mu|) <= tau`：

| 候选 | n_disc | mu | p80 | n_eval | 评估残差 ≤ tau | 评估残差中位数 |
|---|---:|---:|---:|---:|---:|---:|
| `*OC → *OCC` | 809 | 0 | 0.09524 | 389 | **78.7%** | 0.01905 |
| `*CCCCC → *CC1CC1` | 63 | 0.00272 | 0.07075 | 17 | **76.5%** | 0.07347 |
| `*C1CC1 → *CCCCC` | 42 | 0 | 0.08871 | 9 | 44.4% | 0.11701 |

覆盖率只有 **3/1,602 = 0.19%**。`tau=0.20` 时候选增至 151 个，覆盖 9.43%，评估残差在 tau 内 83.1%；`tau=0.05` 时 0 个。结论：**近似等价存在，但只在很窄的、近零响应的替换上稳定；不能假设传递性。**

---

## 4. P2：SHARED 加性组合与未观测替换

### 4.1 自然未观测替换

**0 对。** 原因不是代码失败，而是结构支持太充分：primary 64 个 S 在 1M 前缀中共同出现很密集，所有评估 pair 都能在发现集中找到直接共现。自然未观测替换在这套 1M 结构矩阵中不是可用的检验对象。

### 4.2 受控遮蔽

候选选择只依据发现集支持（降序）、评估支持（降序）、规范 key；最多 20 对。每对删除“同时包含 A 和 B 的发现上下文中 B 的那个 cell”，保留 B 在其他上下文、A 在所有上下文；验证删除后 A、B 不再直接共现，但仍通过其他 S 间接连通。最终 **16 个适用 pair**，4 个因遮蔽后不连通或仍有直接共现而不适用。

| 指标 | 数值 |
|---|---:|
| 适用 pair | 16 |
| 评估观测 | 18,584 |
| 评估上下文 | 15,440 |
| ZERO MAE | 0.17083 [0.16701, 0.17474] |
| SHARED MAE | 0.16924 [0.16587, 0.17243] |
| SHARED 相对 ZERO | **+0.93%** |
| ZERO−SHARED 95% CI | [1.90e-5, 0.00321] |
| 符号一致率 | 61.81% |
| PAIR-FULL 参考 MAE | 0.16209 |
| SHARED 与 PAIR-FULL 差距 | 0.00715 |
| macro pair-equal 增益 | 2.31% |

判定：**`COMPOSITION_SCREEN_ONLY`**。点估计为正，但 0.93% 远低于 15%，且 SHARED 与直接监督的 PAIR-FULL 仍有明显差距。

### 4.3 遮蔽后的 LOCAL 补充评分

按协议“P2 masked pairs 也用遮蔽后拟合的条件修正评分”执行；没有恢复被遮蔽的标签。

| 指标 | 数值 |
|---|---:|
| LOCAL MAE | **0.15423** [0.15134, 0.15723] |
| 相对 SHARED | **+8.87%** |
| SHARED−LOCAL 95% CI | [0.01345, 0.01657] |
| 相对 ZERO | +9.72% |
| LOCAL 覆盖率 | 99.29% |
| macro LOCAL 相对 SHARED | 10.89% |

这是本轮最有信息量的补充结果之一：**在全局加性项几乎无效的 16 个遮蔽 pair 上，条件修正把 MAE 从 0.16924 降到 0.15423，甚至低于 PAIR-FULL 的 0.16209。** 但它是补充分析，pair 选择基于支持而非随机抽样，不能据此把 P2 主判定升级。

---

## 5. P3：LOCAL 上下文规则与匹配 NULL

### 5.1 支持与覆盖

- q 组总数 837；有 ≥10 个发现上下文的 **114 个**；修正覆盖率 **99.00%**；真实 q 与 NULL 同时启用的共同子集 **98.67%**。
- 未见 q 的评估观测比例 **2.38%**；未见 q 回退到 SHARED。
- 修正幅度：`|d_B−d_A|` 中位数 **0.05968**，p90 **0.22738**。
- 支持高度集中：最上面的 q 组有 20,913 / 19,051 / 8,554 个 `k>=2` 发现上下文。

### 5.2 全体评估

| 模型 | MAE（95% CI） | 符号一致率 |
|---|---:|---:|
| SHARED | 0.27145 [0.26731, 0.27585] | 64.78% |
| **LOCAL** | **0.24881** [0.24512, 0.25299] | **67.29%** |
| LOCAL_NULL | 0.27319 [0.26909, 0.27759] | 64.14% |

- LOCAL 相对 SHARED：**8.34%**；SHARED−LOCAL 区间 **[0.02105, 0.02425]**。
- LOCAL 相对 LOCAL_NULL：**8.93%**；LOCAL−NULL 区间 **[-0.02596, -0.02279]**（负值 = LOCAL 更好）。
- 匹配 NULL 没有优于 SHARED，说明“按真实 q 分组”本身带来了信息，而不是任意分组都能改善。

### 5.3 高非可加性子集

高非可加性定义在发现集冻结：`|mu_AB − (b_B−b_A)|` 的最高四分位，阈值 **0.14388**；评估集中命中 442 对、12,223 个观测。

| 模型 | MAE（95% CI） |
|---|---:|
| SHARED | 0.46332 [0.45244, 0.47454] |
| **LOCAL** | **0.41346** [0.40322, 0.42465] |
| LOCAL_NULL | 0.46302 [0.45213, 0.47428] |

LOCAL 相对 SHARED **10.76%**，区间 **[0.04476, 0.05457]**；相对 NULL 10.70%。子集过线，但总体 8.34% 未过 10%。

判定：**`CONTEXT_RULE_SCREEN_ONLY`**。真实上下文规则有可检测的信号，且在高非可加性 pair 上达到实用门槛；但按预注册的总体 10% 规则，不能标记为 `TRANSFERABLE_CONTEXT_RULE`。

---

## 6. 官方 Valid 与 ZINC 外部检查

### 6.1 PCQM 官方 Valid

只使用官方 Valid 的原始标签；Test 标签未读。排除完整图与 Train 重复的 7,356 个单元 / 1,819 个分子后：

| 子集 | 观测 | 上下文 | pair | ZERO MAE | PAIR MAE | skill | 符号一致率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 全部有发现 mu | 36,952 | 15,731 | 1,677 | 0.34791 | **0.29795** | **14.36%** | 67.44% |
| 未见精确 C 上下文 | 6,765 | 4,911 | 798 | 0.22367 | **0.21117** | **5.59%** | 61.85% |
| 已见精确 C 上下文 | 30,187 | 10,820 | 1,673 | 0.37576 | **0.31739** | **15.53%** | 68.68% |

Valid 上仍有 4,525 个单元属于“未见 S 类型”（3,027 个 Valid S 中有一部分不在 Train 词表），这些不能由冻结词表预测；上表只统计有发现 mu 的 pair。整体 skill 与 P1 的 10.9% 同量级，支持“发现集响应可部分迁移”的结论。

### 6.2 ZINC

- 66,494 个单元，11,994 个分子，51,574 个训练上下文。
- primary 二部图：44 个分量、最大 20,194、2-core 6、cycle rank 1。
- 支持 pair 0；主评估 pair 观测 0；Valid/Test 的 pair 观测也为 0。
- P1/P2：`INSUFFICIENT_CROSSED_SUPPORT`；P3：`CONTEXT_SUPPORT_INSUFFICIENT`。

ZINC 的负结果不是“响应不存在”，而是**结构设计不可识别**：几乎无环的观测矩阵无法提供交叉 pair 上下文。报告不把它解释为对 PCQM 结论的反驳。

---

## 7. 示例对（正面与负面，≥5 个）

以下示例均来自 PCQM 1M；数值为原始 eV 单位。前三个是功能等价/近零响应示例，后四个是 SHARED 组合的成功与失败示例。

| # | 类型 | A → B | n_disc | mu / p80 | n_eval | ZERO MAE | SHARED MAE | LOCAL MAE | PAIR-FULL | 读法 |
|---:|---|---|---:|---|---:|---:|---:|---:|---:|---|
| 1 | 功能等价（正） | `*OC → *OCC` | 809 | 0 / 0.0952 | 389 | — | — | — | — | 78.7% 评估残差 ≤0.10；近零稳定替换 |
| 2 | 功能等价（正，小样本） | `*CCCCC → *CC1CC1` | 63 | 0.00272 / 0.0707 | 17 | — | — | — | — | 76.5% ≤0.10；只有 17 个评估观测 |
| 3 | 功能等价（混合） | `*C1CC1 → *CCCCC` | 42 | 0 / 0.0887 | 9 | — | — | — | — | 只有 44.4% ≤0.10；小样本下不稳定 |
| 4 | 组合成功（正） | `*C → *C#N` | 1,341 | −0.337 / — | 442 | 0.4154 | **0.3470** | **0.2922** | 0.3491 | SHARED 增益 16.45%；LOCAL 更好 |
| 5 | 组合成功（正，但高估） | `*C → *C(=O)O` | 1,686 | −0.288 / — | 635 | 0.4979 | **0.4367** | **0.4005** | 0.4295 | SHARED 增益 12.29%，但预测 −0.394 明显高估 |
| 6 | 组合失败（负） | `*N → *O` | 3,565 | 0.0789 / — | 1,452 | 0.1924 | **0.2166** | **0.1767** | 0.1817 | SHARED 比 ZERO 差 12.60%；条件修正可救回 |
| 7 | 组合失败（强负） | `*Cl → *F` | 1,471 | 0.0517 / — | 531 | 0.1282 | **0.1819** | **0.1141** | 0.1154 | SHARED 比 ZERO 差 41.90%；b_F−b_Cl 不迁移 |

`n_disc` 为 P2 记录的直接发现上下文数。示例 4/5 的 LOCAL 是遮蔽后条件修正的 per-pair 值；它们说明“组合可以成功，但成功并不总是来自全局加性项”。

补充负面示例：`*C → *F`（SHARED 0.1437 vs ZERO 0.1251，−14.83%）和 `*C → *O`（SHARED 0.2674 vs ZERO 0.2656，−0.70%）说明，强电负性/杂原子替换最容易破坏可加性。正面示例中 `*C → *N`（SHARED 0.2748 vs ZERO 0.3052，+9.96%）和 `*C → *Cl`（+6.98%）则显示部分成功。

---

## 8. 工程检查与可复现性

`checks.json` 的 `all_ok=true`，包含：

| 检查 | 结果 |
|---|---|
| PCQM 切割不变量（nS+nC=n_atoms、1≤nS≤8、nC≥6、nC≥2nS） | 4,824,824 / 4,824,824 |
| PCQM 抽样图重组 | 57,873 检查，**0 图失败**；立体化学失败 5,128（精确率 91.14%） |
| PCQM 标签/SMILES 对齐 | 1,000 / 1,000 |
| PCQM 行唯一性与官方 Train 归属 | 1,000,000 / 1,000,000，全部在 Train |
| PCQM Valid 不变量 | 322,514 / 322,514 |
| 重复分子标签差 | 10,557 个重复分子；最大差 4.94 eV（记录为噪声，不隐藏） |
| 上下文划分 | 发现/评估分子重叠 0；每个分子最多出现在 1 个主评估上下文 |
| ZINC 切割不变量 | 66,494 / 66,494 |
| ZINC 精确证书重组 | 20,000 检查，0 失败 |
| 合成对照 A–E | 全部通过 |

合成对照（`checks.json`）：

- A 加性：b 差异最大误差 **3.05e-15**；矩形双差 **2.22e-16**。
- B 交叉：LOCAL 相对 SHARED 的 SSE 改善 **12.37%**（固定 ridge λ=5，不要求精确 0）。
- C 无环：任意标签都能被自由行列偏移精确拟合（SSE ~1e-30），证明完美训练拟合不能证明可加性。
- D 未观测替换：删除直接共现但保留间接路径后，SHARED 恢复真值误差 **3.89e-16**。
- E 不可识别：A、B 在不同分量，不产生有意义的共享 b 差。

数据身份见 `data_identity.json`：包含原始文件 SHA256、split 计数、缓存 gids/SMILES 计数、协议/配置/代码哈希、构建产物哈希；`git_commit` 和 `git_dirty` 也记录在案。

---

## 9. 限制、偏差与停止条件

1. **PCQM 只用了 1M 前缀**，不是完整 3.38M Train。支持统计（1,602 对、96,991 观测）已经足够，继续扩展不会改变 P1/P2/P3 的判定框架；本轮按预算停止，没有用标签结果决定规模。
2. **身份是图级**。PCQM 立体化学重组精确率 91.1%，5,128 个抽样不匹配单独报告；没有把 stereo 失败混入图级成功。
3. **重复完整图可以标签冲突**，最大 4.94 eV。它们共享一个分子身份，不会进入不同的发现/评估侧，但会增加标签噪声。
4. **P2/P3 是观察性筛选**，区间是固定数据和固定分析下的上下文 bootstrap，不是总体因果定理。15%/10% 是预注册的实用门槛，不是理论阈值。
5. **自然未观测替换为 0**，所以 P2 的主要证据来自受控遮蔽；受控 pair 是按支持选择的，不是随机样本。
6. **ZINC 是结构阴性对照**；它不能检验交叉响应复用，不能据此声称“ZINC 上功能等价不存在”。
7. **P3 总体未过 10%**；高非可加性子集过线，但不能把子集结果当作总体结论。
8. **P2 遮蔽后的 LOCAL 评分是补充分析**；主判定仍由预注册的 SHARED 规则给出。
9. **执行时序**：P1 的 scaffold-cluster bootstrap 和 P2 的遮蔽后 LOCAL 评分是协议已规定的二级分析，在主 P1/P2 运行之后补跑；没有改变任何阈值、词表、划分或主判定。报告如实标注为二级证据。

没有修改冻结协议或配置来迁就标签结果。所有主判定使用主运行结果；二级分析只用于解释和敏感性检查。所有判定按预注册门槛给出；没有为通过门槛而更换词表、tau、λ 或划分。

---

## 10. 产物与复现

### 10.1 主要产物

| 文件 | 内容 |
|---|---|
| `PROTOCOL.md` | 冻结协议 |
| `configs/sfec_v0.json` | 冻结配置 |
| `data_identity.json` | 原始数据/缓存/代码/构建产物哈希与计数 |
| `structural_support.json` | PCQM 100k/1M、ZINC 的支持矩阵、覆盖、二部图统计、门槛 |
| `observed_cells.parquet` | 5,213,832 行观测单元（PCQM Train/Valid + ZINC Train/Valid/Test），含原始标签 |
| `context_split.json` | 上下文划分规则与各数据集计数 |
| `checks.json` | 重组、标签、划分、重复、合成对照 |
| `rectangle_statistics.csv` | 矩形/共同上下文统计 |
| `response_transfer.csv` | P1 每 pair 的发现中位数、MAD、p80、评估覆盖 |
| `functional_equivalence_candidates.csv` | 发现集冻结的 tau 候选 |
| `unseen_composition_results.csv` | 自然/受控未观测替换结果 |
| `context_rule_results.csv` | P3 各子集 SHARED/LOCAL/LOCAL_NULL 指标 + q 支持 |
| `summary.json` | 机器可读总结与八个回答 |
| `p1_pcqm.json`, `p2_pcqm.json`, `p3_pcqm.json` | 完整 PCQM 阶段结果 |
| `p1_zinc.json`, `p2_zinc.json`, `p3_zinc.json` | ZINC 结构阴性结果 |
| `valid_transfer_pcqm.json` | 官方 Valid 外部迁移 |
| `valid_transfer_zinc.json` | ZINC Valid/Test 描述 |

`observed_cells.parquet` 计数：PCQM Train 4,824,824；PCQM Valid 322,514；ZINC Train 55,400；ZINC Valid 5,528；ZINC Test 5,566；合计 **5,213,832** 行。

### 10.2 复现命令

在 `tracks/structural-sufficiency/structural_functional_equivalence_v0/code/` 下：

```bash
# 结构构建（已完成的缓存可直接复用）
PYTHONPATH=. uv run python sfec_build.py zinc-build --workers 8
PYTHONPATH=. uv run python sfec_build.py pcqm-build --dataset train --start 0 --end 1000000 --workers 8
PYTHONPATH=. uv run python sfec_build.py pcqm-build --dataset valid --start 0 --end 73545 --workers 8

# 支持矩阵
PYTHONPATH=. uv run python sfec_support.py --prefix ../data/cells_pcqm_train_1m --top-k 64
PYTHONPATH=. uv run python sfec_support.py --prefix ../data/cells_zinc --top-k 64

# P1 / P2 / P3
PYTHONPATH=. uv run python sfec_p1.py --dataset pcqm --prefix ../data/cells_pcqm_train_1m
PYTHONPATH=. uv run python sfec_p2.py --dataset pcqm --prefix ../data/cells_pcqm_train_1m
PYTHONPATH=. uv run python sfec_p3.py --dataset pcqm --prefix ../data/cells_pcqm_train_1m

# 外部 Valid / ZINC
PYTHONPATH=. uv run python sfec_valid.py --dataset pcqm \
  --train-prefix ../data/cells_pcqm_train_1m --valid-prefix ../data/cells_pcqm_valid
PYTHONPATH=. uv run python sfec_valid.py --dataset zinc --train-prefix ../data/cells_zinc

# 检查、汇总、聚合
PYTHONPATH=. uv run python sfec_checks.py
PYTHONPATH=. uv run python sfec_summary.py
PYTHONPATH=. uv run python sfec_aggregate.py
PYTHONPATH=. uv run python sfec_identity.py
```

资源约束：CPU only；本地 ≤8 workers、每个 worker 1 个 BLAS 线程；进程 RSS 远低于 64 GB 上限；分块构建/写出，未把全部单元作为 Python 对象加载。实际墙钟约 **1 小时**（12:54–13:45 本地时间，含重跑与检查），在 2 小时硬上限内。

### 10.3 停止声明

本轮按预注册范围完成：数据身份、结构支持、P1/P2/P3、官方 Valid 外部检查、工程对照、聚合产物与中文报告均已生成。**完成本轮后停止**；没有启动 GNN、没有训练、没有 Test 标签分析、没有根据标签结果重新设计词表或门槛。

---

## 11. 附：关键机器可读字段

- P1：`analysis/p1_pcqm.json` → `transfer.supported.context_equal`、`candidates`、`judgement`。
- P2：`analysis/p2_pcqm.json` → `controlled_masking.pool`、`natural_unobserved`、`judgement`。
- P3：`analysis/p3_pcqm.json` → `metrics.all`、`metrics.high_nonadditivity`、`q_support`、`judgement`。
- Valid：`analysis/valid_transfer_pcqm.json` → `transfer`。
- 检查：`analysis/checks.json` → `all_ok`、`controls`。
- 总结：`analysis/summary.json` → `answers[0..7]`。

所有原始标签只来自 ZINC 的 `logP_SA_cycle_normalized` 和 PCQM 的 `homolumogap`；没有使用任何模型预测、教师标签或蒸馏信号。
