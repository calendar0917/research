# SSTA v0 — Structural State and Transferable Actions

**只做实现、运行与报告。没有训练、微调、蒸馏、权重修改，没有生成新分子标签，官方 Test 标签从未读取。**

- 目录：`tracks/structural-sufficiency/structural_state_action_v0/`；代码 `code/`，输出 `analysis/`
- CPU-only；总墙钟 **35.8 s**，峰值 RSS **2.15 GB**（目标 300 s，硬上限 600 s）
- 冻结 W3：LCRC-W3，2,038,167 参数，checkpoint epoch 39，Valid MAE **0.127809174 eV**（checkpoint 0.127809167 eV，复现差 6.30e-09 eV）
- 数据身份：观测 36,998，可评价 36,952；冻结 SFEC 指标复现最大绝对差 0.0e+00

---

## 0. 五问速答

### 1. LCRC 能否准确预测 double difference（环境交互）？

对同一 canonical 替换 A→B，两个不同 exact context 的`D = Δ(C2) − Δ(C1)`：

| 权重 | ZERO | PAIR_Q | LOCAL | LCRC | LCRC 相关 | LCRC 斜率 | 方向准确率 \|D_true\|≥0.10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 观测等权 (n=59,214) | 0.4827 | 0.4687 | 0.4406 | **0.1798** | 0.926 | 0.879 | 0.892 (45,876) |
| pair-type macro (1,542 types) | 0.5033 | 0.4914 | 0.4547 | **0.1803** | — | — | 0.886 |

- LCRC − ZERO 的 pair-type macro ΔMAE = **-0.3230 eV** (95% CI [-0.3380, -0.3066])；LCRC − PAIR_Q = **-0.3111 eV** (CI [-0.3256, -0.2955])；LCRC − LOCAL = **-0.2744 eV** (CI [-0.2878, -0.2606])。

**答：能，且不是靠平均替换效应。** 相关 0.93、斜率 0.88、方向准确率 0.89；ZERO（环境无关固定响应）与两个冻结查表模型都远不能表达这种环境间差异。分同 q / 不同 q、以及两个 context 都在 1M 前缀未见子集上结论不变（见 `interaction_results.csv`）。但绝对误差仍约 0.18 eV，double difference 只是被部分解释。**口径提醒**：LCRC 读整张图，ZERO/PAIR_Q/LOCAL 只有固定查表或局部片段；因此这个差距说明“环境效应真实存在且能被结构级模型预测”，而不是说 LCRC 动用了查表模型之外的额外环境机制。

### 2. 完整 W3 计算能否等价写成可累加的局部状态及非线性读出？

能。`U(G)=Σ_v φ(h_v)`，φ 维数 **1158**；仅由 `(U(G), g(G))` 重建全部读出中间量：

| 重建量 | 最大绝对误差 | 参照尺度 |
|---|---:|---:|
| 最终预测 (eV) | 9.878e-07 | 10.169 |
| 标准化 logits | 8.510e-07 | 3.861 |
| Σ f(h) | 3.254e-07 | 1.572 |
| rho 输出 | 5.776e-07 | 2.636 |
| attention residual 输出 | 3.463e-07 | 1.859 |
| mean h | 3.445e-07 | 2.060 |
| mean h² | 7.273e-07 | 8.115 |
| 每个 head 的 pooled 向量 | 3.595e-06 | 4.394 |
| log1p(n) | 0.000e+00 | 3.258 |
| 节点数 | 0.000e+00 | 25.000 |

在 1024 个固定 hash 分子上，最大预测误差 **9.878e-07 eV**（容差 1e-05）。`F(B)−F(A)=H(U(A)+ΔU, g(A)+Δg)−H(U(A),g(A))` 中 ΔU **只由有符号结构记录多重集重建**，对真实分子 B 的冻结预测复现最大误差 **6.947e-07 eV**（512 对）。

这是**等价计算重写**，不是性能提升。残差仅来自 float32/float64 重结合。注意力指数用**固定 head-specific 平移** `c_k=[6.5353, 11.2321, 6.3425, 9.3617]`（全分子一致）；若改用每图 max 平移，两分子 pooled 向量会出现最大 1.689 的伪差（见 `state_replay.json::shift_counterexample`），因此禁止那样跨分子相减。

### 3. 相同结构编辑记录能否产生相同状态增量？

能，四项检查全部通过：

- 原子重编号：root record 多重集完全不变（True），状态最大变化 2.98e-08，预测变化 1.38e-07 eV；
- 相同记录 → 相同 h：3,136 个重复记录组（2,771 组跨分子），最大 h 偏差 1.19e-06，最大 φ 偏差 3.68e-06；
- 有符号记录增量复算 ΔU：29,313 对，通过率 **1.0000**（最大绝对误差 7.46e-06，最大相对误差 2.35e-06）；
- 记账：节点数变化核对 29,313 对全部一致，失败 0；Δn≠0 的比例 0.773；全图描述符变化在 512 对中 98% 非零（L2 中位数 1.493）。

### 4. 相同状态增量能否在不同环境中产生正确的不同任务响应？

只由结构记录筛选的 `E_AB(C1)=E_AB(C2)` 矩形：

| 子集 | 矩形 | pair type | \|D_true\| 均值 (eV) | ZERO | PAIR_Q | LOCAL | LCRC | LCRC 方向 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `same_edit` | 946 | 491 | 0.3214 | 0.3015 | 0.3015 | 0.3015 | **0.1431** | 0.8351 |
| `same_edit_same_q` | 933 | 491 | 0.3197 | 0.3017 | 0.3017 | 0.3017 | **0.1434** | 0.8349 |
| `same_edit_different_q` | 13 | 4 | 0.4454 | 0.3864 | 0.3819 | 0.3949 | **0.1247** | 0.9524 |
| `same_edit+same_dg` | 38 | 27 | 0.2059 | 0.2168 | 0.2164 | 0.2155 | **0.1343** | 0.8333 |

（pair-type macro MAE，eV；ZERO 即“环境无关固定响应”，恒零预测没有方向，其方向准确率按定义记 0。）

- 相同编辑下两个 context 的 ΔU 相对差最大仅 1.28e-06，即编辑状态增量在数值上等同；真实响应仍显著不同：|D_true| 均值 0.3214 eV，602/946 个矩形 |D_true| ≥ 0.10 eV；两个 context 的 ΔU 差最大仅 8.47e-06。
- LCRC 在这种条件下仍显著优于基线：LCRC−ZERO = **-0.1584 eV** (CI [-0.1882, -0.1303])，相关 0.887，斜率 0.849。
- 同 q 子集里 PAIR_Q 按定义只能给 0（查表在固定 q 内是常数），LCRC macro MAE 0.1434 vs ZERO 0.3017；不同 q 子集 13 个矩形，样本太小只作描述。

严格子集（**ΔU 与 Δg 都相同**，即输入增量完全相同、只剩初始状态不同）：

- 38 个矩形、27 个 pair type、56 个 context、106 个真实分子；|D_true| 均值 0.2059 eV，19 个矩形 |D_true| ≥ 0.10 eV。
- 严格子集本身也通过独立复算：用**只读 cache 数组**的第二套记录实现（不经 torch/collate）重算 29,313 个 clean 矩形，得到同样的 946 个 same-edit 与 38 个 strict 矩形；strict 子集的逐矩形审计表在 `action_support.json::strict_rectangles`。
- 模型在这些矩形上的**响应差仍非零**：|D_LCRC| 均值 0.1184 eV，最大 0.7636 eV，31/38 个 ≥ 0.01 eV；该差异**只能来自对不同初始状态 (U(A), g(A)) 的非线性读出**，与输入增量无关（strict 子集内 ΔU 差最大 2.98e-06，Δg 逐位相同；状态读回复现误差最大 4.26e-07 eV）。
- 这些响应差与真实响应差方向一致的证据：LCRC macro MAE 0.1343 vs ZERO 0.2168，方向准确率 0.8333。

**答：局部状态增量相同并不能消除真实的任务环境差异；LCRC 能部分预测这种差异，且在 ΔU、Δg 完全相同的严格子集上仍优于 ZERO/PAIR_Q/LOCAL。**但严格子集只有 38 个矩形，低于预先声明的 50 个阈值，因此按规则标记 **`INSUFFICIENT_EXACT_ACTION_SUPPORT`**；上面的严格子集数字只作描述性证据，不做显著性主张。

### 5. 这些规律具备多大的真实支持覆盖？

| 项 | 数量 |
|---|---:|
| 可评价 pair 观测 | 36,952 |
| 有观测的 canonical pair type | 1,677 |
| ≥2 个 exact context 的 pair type | 1,542 |
| 可能的 context pair（矩形总数） | 2,551,216 |
| 实际抽取矩形（每 pair type ≤64，固定 hash） | 59,214（2.32%） |
| 不共享原始分子身份的矩形子集 | 11,930 （1,542 pair type） |
| 进入状态/记录分析的唯一分子（预算 12,000） | 11,984 |
| 该样本覆盖的 clean 矩形 / pair type | 29,313 / 1,288 |
| 四个分子两两互异的矩形（进入 same-edit 筛选） | 29,309 |
| `E_AB` 相同的矩形 / pair type | 946 / 491 |
| `E_AB` 与 Δg 均相同 | 38 / 27 pair type |

矩形数从不作为独立样本数：所有区间都是 **pair-type bootstrap**（1,000 次，pair type 为独立单位），并另报 molecule-disjoint 子集与 context-equal 口径。

---

## 1. 真实例子（5 个，全部来自官方 Valid）

**strict: identical edit record and identical dg** — canonical pair A_rank=9 → B_rank=29；`|E_AB|`=12，`E_AB` 相同：True，Δg 相同：True，q1/q2=5/5（同 q）

| context | side | SMILES | label (eV) | W3 (eV) |
|---:|:--:|---|---:|---:|
| 20061 | A | `CCOC(=O)CCCCC(=O)OC` | 7.3852 | 7.6050 |
| 20061 | B | `OCCCCCCCC(=O)OCC` | 7.3988 | 7.5339 |
| 34021 | A | `COC(=O)CCCCCCC(=O)O` | 6.2287 | 7.3838 |
| 34021 | B | `OCCCCCCCCCC(=O)O` | 7.3906 | 7.4391 |

D_true = +1.1483；D_LCRC = +0.1264；D_PAIR_Q = +0.0000；D_LOCAL = +0.0000；D_ZERO = 0（eV）。

**strict: largest LCRC double difference inside the strict subset** — canonical pair A_rank=28 → B_rank=29；`|E_AB|`=6，`E_AB` 相同：True，Δg 相同：True，q1/q2=39/5（不同 q）

| context | side | SMILES | label (eV) | W3 (eV) |
|---:|:--:|---|---:|---:|
| 5199 | A | `NCCCCc1cccc(c1)OC` | 5.9266 | 5.9622 |
| 5199 | B | `OCCCCc1cccc(c1)OC` | 5.8858 | 5.9386 |
| 34021 | A | `NCCCCCCCCCC(=O)O` | 6.5607 | 6.6991 |
| 34021 | B | `OCCCCCCCCCC(=O)O` | 7.3906 | 7.4391 |

D_true = +0.8708；D_LCRC = +0.7636；D_PAIR_Q = +0.0109；D_LOCAL = +0.0348；D_ZERO = 0（eV）。

**non-empty identical edit: LCRC tracked the real change** — canonical pair A_rank=1 → B_rank=29；`|E_AB|`=9，`E_AB` 相同：True，Δg 相同：False，q1/q2=5/5（同 q）

| context | side | SMILES | label (eV) | W3 (eV) |
|---:|:--:|---|---:|---:|
| 18351 | A | `BrCCCCCc1ccccc1` | 6.3539 | 6.4165 |
| 18351 | B | `OCCCCCCCCc1ccccc1` | 6.4899 | 6.5314 |
| 31292 | A | `BrCCCCCC(C)C` | 7.3090 | 7.3571 |
| 31292 | B | `OCCCCCCCCC(C)C` | 9.1430 | 9.2349 |

D_true = +1.6980；D_LCRC = +1.7630；D_PAIR_Q = +0.0000；D_LOCAL = +0.0000；D_ZERO = 0（eV）。

**largest real double differences: LCRC accurate** — canonical pair A_rank=18 → B_rank=44；`|E_AB|`=9，`E_AB` 相同：False，Δg 相同：False，q1/q2=13/29（不同 q）

| context | side | SMILES | label (eV) | W3 (eV) |
|---:|:--:|---|---:|---:|
| 3090 | A | `CN(CN1C(=O)c2c(C1=O)cccc2)C` | 3.7633 | 3.8403 |
| 3090 | B | `O=CCN1C(=O)c2c(C1=O)cccc2` | 4.8382 | 4.8270 |
| 82399 | A | `CN(C(=O)c1ccccc1C)C` | 5.7960 | 5.8168 |
| 82399 | B | `O=CC(=O)c1ccccc1C` | 4.1960 | 4.1259 |

D_true = +2.6749；D_LCRC = +2.6777；D_PAIR_Q = +0.0000；D_LOCAL = +0.1933；D_ZERO = 0（eV）。

**largest real double differences: LCRC misses** — canonical pair A_rank=13 → B_rank=18；`|E_AB|`=9，`E_AB` 相同：False，Δg 相同：False，q1/q2=5/20（不同 q）

| context | side | SMILES | label (eV) | W3 (eV) |
|---:|:--:|---|---:|---:|
| 28087 | A | `CCCCCCC(C)C` | 10.3158 | 10.3971 |
| 28087 | B | `CCCCCCC=O` | 6.3375 | 6.3331 |
| 183889 | A | `CN(CCCNC(C)C)C` | 7.4369 | 7.9351 |
| 183889 | B | `O=CNCCCN(C)C` | 7.5022 | 6.3149 |

D_true = +4.0436；D_LCRC = +2.4438；D_PAIR_Q = -0.3701；D_LOCAL = +0.1940；D_ZERO = 0（eV）。

---

## 2. 口径与方法（关键点）

- **数据**：复用 `structural_effect_diagnosis_v0/code/pcqm_response.py` 的真实 cell 聚合、支持筛选与数据身份检查；ZERO/PAIR_Q/LOCAL 是 SFEC v1 冻结估计；LCRC 是冻结 W3 `pred(B) − pred(A)`。不重扫 Train、不重建匹配目录、不读 Test。
- **矩形**：每个 canonical pair type 内，对全部可评价 context pair 按固定 SplitMix64 hash 排序取前 64；与标签、预测、误差无关。
- **记录**：每个根节点的完整确定性记录 —— 根原始 `atom_feat`/`atom_scalar`/`atom_type`、入射边 `bond_feat` 与对端原始 `atom_feat`、S1 neighbour codes、S2 `path_cat`/`path_end63`/`path_closure` 与末端原始 `atom_feat`；无序项用**带重数的排序多重集**，以 BLAKE2b-128 规范字节摘要比较；不用学习 embedding 近似相等判定。记录字段范围由实际 `TULocalContent`（C1 变体）代码核对，不按名称假定。
- **状态**：`φ`/`U` 在 float64 累加；注意力指数固定 head-specific 平移；`H` 只读 `U` 与 `g`。
- **多分子 cell**：主分析只使用 `cell_mol_n == 1` 且无 stereo/重复身份的 clean cell，避免把一个聚合 cell 当成单个分子。

## 3. 限制

- 严格子集（ΔU 与 Δg 都相同）只有 38 个矩形 / 27 个 pair type，低于预先声明的 50 个矩形阈值 → `INSUFFICIENT_EXACT_ACTION_SUPPORT`；该子集的数字只作描述，不作显著性主张。
- 矩形共享分子，不是独立样本；主要不确定性口径是 pair-type macro + bootstrap。
- 精确状态账目说明的是**冻结计算实际读取的结构内容**，不构成唯一的化学因果归因，也不证明模型“理解”化学。
- 未构造任何合成/混合状态；所有状态属于真实官方 Valid 分子。
- 本轮未训练、未选择 checkpoint、未修改权重、未生成标签。

