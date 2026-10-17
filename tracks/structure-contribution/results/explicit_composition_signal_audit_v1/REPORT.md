# REPORT — explicit-composition-signal-audit-v1

- 日期：2026-10-17 · track：`structure-contribution`
- protocol：`explicit-composition-signal-audit-v1`（`PROTOCOL.md` 在首次读标签前冻结）
- 结构阶段 commit：`40bbfe96444f585a1f5d543f98f4aa8d451c3c67`
- 数据：官方 `ZINC(subset=True, split="train")`；9,996 条可核实结构；official valid/test 从未读取
- 计算：本地 CPU；无 GPU、无新训练、无 BJOC 架构改动
- 完整数值：`analysis/stage_a_summary.json`、`analysis/stage_b_labels.json`、
  `analysis/stage_c_gate.json`、`analysis/stage_a_tier_*.json`、`analysis/stage_b_pairs.json`

---

## 0. 直接回答

**没有。** 在 ZINC-12k official train 上，我们没有找到足够强、足够独立、非平凡的证据，
表明“真实结构组合规则”值得成为下一轮 ZINC 预测模型的**核心归纳偏置**。

**判定：`INSUFFICIENT_OR_CONFOUNDED`。证据强度：强（针对 ZINC-12k train 这一数据/目标），
不是对整个非 GNN / 组合机制方向的否证。**

理由可以压成三句话：

1. **结构侧不可分离**：`B1`（每根键的精确半径 1 带属性 patch 多重集）在 9,996 条结构上
   与完整 constitution `C` 诱导**完全相同**的等价类划分（9,993 组；仅 3 组是表中重复行）。
   也就是说，“哪些单元如何连接”已经隐含在局部 patch 多重集里；不存在“同 B1、不同 C”
   的不同分子。
2. **严格边际碰撞极少**：`B0`（规模、原子类别计数、键类别计数、Bond line-graph 度分布、
   junction-count 分布）在 9,996 条上只产生 **11 组 / 22 个分子 / 11 对** 同 B0 不同 C 的
   真实碰撞；同 B0 且同 B1 的不同 C 样本为 **0**。
3. **标签侧公式主导**：这 11 对的 `y` 差异并不小（mean|Δy| = 0.595），但
   penalized-logP 公式解释了 **R² = 0.999897**；beyond-formula residual 的
   mean|·| = 0.00557、max = 0.0275，而最大 residual 对应的 pair-specific
   公式重构噪声为 0.0314，因此 **0/11** 超过 pair-specific 噪声。在所有层级中，
   排除环基不稳定与公式重构误差后，**没有任何一对**的 residual 超过噪声下限
   （genuine beyond-noise pairs = **0**；最大层级 `L_old` 有 2,154 对）。

因此：本轮没有 Stage C 训练探针；没有 GPU 方案；没有“下一轮架构候选”被授权。

---

## 1. 证据链（按任务要求的四层区分）

| 层次 | 本轮结果 | 解释 |
|---|---|---|
| 信息存在 | B1 与 C 的划分完全相同；B0 只差 11 组 | C 的信息**存在于** B1 中，但不能作为独立因子被分离 |
| 标签相关 | 11 对严格 B0 碰撞的 mean\|Δy\|=0.595 | `y` 确实随 constitution 变，但主要由公式项解释 |
| 可学习性 | **未测**（Stage C 门槛失败） | 不能从“信息存在/标签相关”推出“可学习” |
| 泛化收益 | **未测**，也不由前三层推出 | 旧 Bond-RW/Organized 信号仍是探索性、受混杂 |

---

## 2. 数据完整性与冻结

- 全量 10,000 行重跑 SMILES ↔ 官方 PyG 图配对：mismatch = `[941, 2210, 6071, 9766]`，
  与冻结排除清单逐位一致；其余 9,996 行进入结构统计。
- 输入 SHA-256 在 Stage A 运行时核对：
  - `train_canonical_smiles.npz` = `04a59035…dcaff`
  - `data/ZINC/subset/processed/train.pt` = `e0f1837b…6955b`
- `FREEZE.json` 在 Stage A 结束时写入；`git_dirty=false`；Stage B 逐项核对
  protocol/code/input/12 个 Stage-A artifact 的 SHA-256 后才读取 `y`。
- Stage B 只读 official-train `y`（从 `train.pt` 的 `blob[0]['y']`），并与
  `encoded_train.pt` 的 `y` 全量交叉核对：**max abs diff = 0.0**。
  official valid/test、内部 Locked 标签、模型预测均未读取。

---

## 3. Stage A：结构可比较性（标签盲）

### 3.1 匹配阶梯

| 层级 | 定义 | 组数 | C-different 组 | 分子 | 骨架 | 跨 C 对 | 骨架不相交对 |
|---|---|---:|---:|---:|---:|---:|---:|
| `M01` | 同 B0 + 同 B1(T1)，不同 C | 9,993 | **0** | 0 | 0 | **0** | 0 |
| `M0` | 同 B0，不同 C | 9,982 | **11** | 22 | 21 | **11** | 10 |
| `M0_ring` | 同 B0 + 同环数，不同 C | 9,982 | 11 | 22 | 21 | 11 | 10 |
| `B1` | 同 B1(T1)，不同 C | 9,993 | **0** | 0 | 0 | **0** | 0 |
| `A1` | 同原子类别组成 + 键类别 + 环数，不同 C（放松 degree/junction） | 9,950 | 41 | 84 | 78 | 45 | 40 |
| `A3` | 同元素组成 + 键类别 + 环数 + degree/junction，不同 C（放松 H/电荷/芳香） | 9,932 | 61 | 122 | 120 | 61 | 59 |
| `A4` | 同元素组成 + 键类别 + 环数，不同 C（大样本、明确混杂） | 8,683 | 940 | 2,251 | 1,894 | 1,909 | 1,835 |
| `L_old` | 历史 bond-RW 粗边际签名，不同 C（不控制元素/原子类别） | 8,531 | 1,017 | 2,479 | 2,163 | 2,154 | 2,093 |

关键读数：

- **B1 与 C 的划分完全相同**：B1 组数 9,993 = C 组数 9,993；B1 的 C-different 组 = 0。
  3 个 B1 多分子组全部是表中重复行（同 constitution），其 `Δy` 全为 0。
- **B0 接近完整**：9,996 条中只有 11 组、22 个分子的同 B0 不同 C 碰撞；
  B0+B1 的不同 C 碰撞为 0。
- `M0_ring` 与 `M0` 相同：这 11 对也恰好同环数，因此“控制环数”没有进一步缩小集合。
- `A1/A3` 是预注册的近似层级；`A4/L_old` 只用于估计公式主导程度，**不能**用于声称 C 的独立效应。

### 3.2 旧 208 对 / 历史 bond-RW 签名的含义

`L_old` 是历史 `_collision_search` 的粗签名（键数、键类别、line-graph 度、junction-count），
不包含元素/原子类别组成。全量 10k 上：

- `L_old` 有 2,154 对同签名不同 C 的分子对；
- 其中只有 **11 对（0.511%）** 同时满足同 B0；
- 只有 **11 对** 同原子类别组成；**61 对（2.83%）** 同元素组成；
- **0 对** 同 B1。

所以旧的“208 个关系不同匹配对”来自一个**不控制化学组成**的粗边际签名；
在全量 10k 上，它的 99.49% 的 C-different 对伴随原子类别组成差异。它不能作为
“真实结构组合有独立增量”的证据；本轮也没有把它当作证据。

### 3.3 不变性与完整性

- **C 的精确性**：`C_key`（unit–atom incidence 的 pynauty 规范键）与
  `C_smiles`（RDKit constitution canonical SMILES）在 9,996 行上诱导的划分
  **一一对应**：0 个 key→多个 smiles，0 个 smiles→多个 key。
- **置换不变性**：200 个分子 × 3 次随机原子重编号 = 600 次检查；
  B0/B1(T1/T2/T3)/C_key/C_smiles 全部不变，失败 0。
- **patch 并集**：每个分子的所有 S patch 并集覆盖全部真实键，失败 0。
- **环基不稳定性**：在进入任何 pair 的 3,894 个分子中，7 个（0.18%）的
  `cycle_penalty` 会因 `nx.cycle_basis` / `minimum_cycle_basis` / RDKit `GetSymmSSSR`
  的环基选择而改变；这些分子全部落在 `A4/L_old`，不在 `M0/A1/A3`。

---

## 4. Stage B：冻结后读 official-train `y`

### 4.1 严格 B0 的 11 对：`y` 差异大，但公式解释几乎全部

| # | rows | y_a → y_b | Δy | Δformula | residual | Δcycle_scaled | 骨架不相交 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 36/7416 | -0.4045 → -0.2983 | 0.1062 | 0.1036 | 0.0026 | 0.000 | 是 |
| 2 | 102/7981 | 1.9000 → -1.0620 | **-2.9620** | -2.9896 | 0.0275 | **-3.497** | 是 |
| 3 | 352/5352 | -2.7570 → -2.7278 | 0.0292 | 0.0278 | 0.0014 | 0.000 | 是 |
| 4 | 465/8692 | 0.8177 → 0.8500 | 0.0323 | 0.0334 | -0.0012 | 0.000 | 是 |
| 5 | 1929/3623 | -1.4845 → -2.5141 | -1.0295 | -1.0192 | -0.0104 | 0.000 | 是 |
| 6 | 2490/3339 | 2.0503 → 2.1244 | 0.0740 | 0.0743 | -0.0003 | 0.000 | 否 |
| 7 | 2645/5869 | 0.5939 → 0.7547 | 0.1608 | 0.1593 | 0.0015 | 0.000 | 是 |
| 8 | 3248/5684 | -1.3072 → -1.2844 | 0.0229 | 0.0230 | -0.0001 | 0.000 | 是 |
| 9 | 3354/8953 | 0.8320 → -0.1769 | -1.0089 | -1.0008 | -0.0081 | 0.000 | 是 |
| 10 | 3676/5288 | 1.7925 → 2.2890 | 0.4965 | 0.4923 | 0.0042 | 0.000 | 是 |
| 11 | 4738/7410 | 1.1908 → 0.5653 | -0.6254 | -0.6214 | -0.0040 | 0.000 | 是 |

汇总（B=5000，按 group 聚类）：

- `Δy`：mean = -0.428，median = 0.029，mean|Δy| = **0.595**，
  median|Δy| = 0.161，p90|Δy| = 1.030，max|Δy| = 2.962；
  95% CI（mean）= [-1.064, +0.046]。
- 公式：`R² = 0.999897`，corr = 0.999965；OLS 系数
  `(logP 1.0115, SA 0.9963, cycle 0.9938)`，截距 ≈ 0。
- residual：mean = 0.00121，mean|res| = **0.00557**，median|res| = 0.00260，
  p90|res| = 0.01037，max|res| = 0.02754；95% CI（mean|res|）= [0.00201, 0.01069]。
- **beyond reconstruction noise：0/11**。最大的 residual（pair 2，0.0275）
  对应更大的逐分子公式重构噪声（0.0314），不是 beyond-formula 信号。
- 无 `|y|>3` 的 tail pair。

**H0c 的直接案例**：pair 2 的 `Δy = -2.962`，其中 `Δcycle_scaled = -3.497`
（一分子有 7 元环惩罚、另一分子没有），公式给出 -2.990；这是 ZINC 人工目标环惩罚
主导的差异，不是“单元组合规则”的独立效应。该对也是 11 对中 |Δy| 最大的一对，
去掉它后 mean|Δy| 从 0.595 降到 0.359，但 mean|residual| 仅从 0.00557 降到 0.00338。

### 4.2 各层级的公式分解

| 层级 | pairs | groups | 分子 | 骨架 | mean\|Δy\| | formula R² | mean\|res\| | median\|res\| | raw beyond-noise / genuine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `M01` | 0 | 0 | 0 | 0 | — | — | — | — | — |
| `M0` | 11 | 11 | 22 | 21 | 0.595 | **0.999897** | 0.00557 | 0.00260 | 0 / **0** |
| `A1` | 45 | 41 | 84 | 78 | 0.480 | 0.999884 | 0.00460 | 0.00307 | 0 / **0** |
| `A3` | 61 | 61 | 122 | 120 | 1.199 | 0.999968 | 0.00672 | 0.00421 | 1 / **0** |
| `A4` | 1,909 | 940 | 2,251 | 1,894 | 1.393 | 0.994967 | 0.01294 | 0.00538 | 71 / **0** |
| `L_old` | 2,154 | 1,017 | 2,479 | 2,163 | 1.429 | 0.979226 | 0.02544 | 0.00737 | 143 / **0** |

- “raw beyond-noise” = `|residual| > 0.013433`（community 常数下的公式重构噪声下限）。
- “genuine” = raw beyond-noise **且** 环基稳定 **且** `pair_recon_noise ≤ 0.013433`。
  **所有层级的 genuine = 0。**
- `A4/L_old` 的 raw beyond-noise 全部可由以下两类已知误差解释：
  - 环基不稳定：`A4` 4 对（residual mean|·| = 2.67）、`L_old` 9 对（3.91）；
  - 逐分子公式重构误差：少数分子的 `|recon|` 可达数个单位（community 常数下 max 7.12）。
  排除这两类后，`A4` 的 residual mean|·| 从 0.01294 降到 0.00735，
  `L_old` 从 0.02544 降到 0.00916，且不再有任何 pair 超过噪声下限。

### 4.3 最优常数预测器的经验 MAE 下界

组内用该组 `y` 中位数预测；**in-sample**，singleton 组贡献 0。

| 层级 | C-different 组/分子 | matched mean\|dev\| | whole-train normalized | multi-group 分子占比 |
|---|---:|---:|---:|---:|
| `M01` / `B1` | 0 / 0 | — | 0 | 0.06% |
| `M0` | 11 / 22 | 0.298 | 0.000655 | 0.28% |
| `A1` | 41 / 84 | 0.210 | 0.001762 | 0.90% |
| `A3` | 61 / 122 | 0.599 | 0.007314 | 1.28% |
| `A4` | 940 / 2,251 | 0.689 | 0.1552 | 22.56% |
| `L_old` | 1,017 / 2,479 | 0.732 | 0.1814 | 24.86% |

解释：

- `M01/B1` 的 matched lower bound 为空（没有不同 C 的同 bag 样本）；whole-train 为 0，
  因为 B1 bag 基本是唯一指纹，singleton 组没有可消除误差。
- `M0` 的 matched lower bound 0.298 是 22 个分子的组内中位数误差；摊到全训练集只有
  0.000655 y-units。**这不是 BJOC-P3 的误差下界**，也不是泛化下界。
- 从 `M0` 到 `A4/L_old`，lower bound 的增大是**放松边际控制**换来的；
  它衡量的是被放开的规模/组成/degree 信息，而不是 C 的独立增量。

### 4.4 重复行与噪声

- B1 的 3 个多分子组全部是表中重复 constitution（6 行）；3 对的 `Δy` **全部为 0**。
- 因此本数据中没有可测的“同结构重复标签噪声”；但也没有“同 B1、不同 C”的重复
  上下文可用于识别组合规则。

### 4.5 主导性 / 异常值

- `M0`：最大 `|Δy|` pair（102/7981，环惩罚）贡献了 mean|Δy| 的约 40%；
  去掉它后 mean|Δy| 0.359，但 mean|residual| 仅 0.00338。
  leave-one-group-out 对 mean|residual| 的最大改变 = 0.00220。
- `A1/A3`：top-1 移除后 mean|residual| 分别为 0.00395 / 0.00638；
  leave-one-group-out 最大改变 0.00125 / 0.00035。无单组主导。
- `A4/L_old`：tail pair（`|y|>3`）分别占 sum|Δy| 的 28.6% / 21.8%，
  占 sum|residual| 的 55.3% / 69.5%；但最大的 residual 集中在环基不稳定的
  少数 pair（见 §4.2），且 top-5% 移除后 mean|residual| 降到 0.00625 / 0.00810。
  这是“公式/环基尾部”，不是广布的 beyond-formula 组合信号。

---

## 5. Stage C：门槛失败，未训练

| 门槛 | 阈值 | 观测 | 通过 |
|---|---|---|---:|
| G1 严格样本量 | M01 ≥30 组、≥60 分子、≥20 骨架、≥30 骨架不相交对 | 0 / 0 / 0 / 0 | ❌ |
| G2 非公式信号 | median\|res\| ≥0.02、formula R² ≤0.95、mean\|res\| CI 下界 >0 | 无 M01 pair | ❌ |
| G3 公平对照 | B1 的 C-different 组 >0 | B1 的 C-different 组 = 0 | ❌ |

`run_stage_c = false`。**没有运行任何学习探针，没有使用 GPU。**

G3 的失败是结构性的：B1 已经与 C 同划分，任何“Control = B1 内容”的模型都已经
隐含 C；若把 Control 退回到 B0-only，又会重新引入本轮要控制的边际混杂。
因此在本数据上无法构造公平的 Control/Real/Null 对照。

---

## 6. 最强反对证据（必须保留）

1. **B1 与 C 同划分**：局部 patch 多重集已经决定 constitution；没有“同局部内容、
   不同全局连接”的不同分子。C 不能作为独立信息因子被识别。
2. **B0 接近完整**：仅 11 对同 B0 不同 C；同 B0+B1 的不同 C 为 0。
3. **严格 B0 的 11 对公式主导**：R²=0.999897；residual mean 0.00557，
   raw beyond-noise 0/11（最大 residual 0.0275 < 该对的 pair-specific 噪声 0.0314）；
   genuine beyond-noise = 0。
4. **最大严格 pair 是环惩罚驱动**：pair 102/7981 的 Δy=-2.962，
   Δcycle_scaled=-3.497；这是 H0c 的直接证据。
5. **旧粗签名 99.49% 不受组成控制**：L_old 2,154 对中只有 11 对同 B0；
   0 对同 B1。历史 208 对不能提升为“完整 10k 严格控制”的证据。
6. **大样本探索层级的 residual 尾部可由已知误差解释**：A4 4 对 / L_old 9 对
   环基不稳定，加上少数公式重构异常分子；排除后 genuine beyond-noise = 0。
7. **重复行 Δy=0**：3 组重复 constitution 的标签完全一致；没有可利用的重复上下文。

---

## 7. 科研判断与下一步

**判断**：ZINC-12k train 不能为“显式结构组合规则是核心归纳偏置”提供所需证据。
不是因为组合规则一定无效，而是因为：

- 结构侧：C 与 B1 不可分离，严格匹配样本几乎不存在；
- 标签侧：`y` 是 logP/SA/环惩罚的近似确定函数，strict residual 在噪声内；
- 设计侧：B1 已含 C，无法构造公平的 Control/Real/Null。

**建议**：

1. **不要把显式组合规则作为下一轮 ZINC 预测模型的核心归纳偏置。**
   旧 Bond-RW 的探索性正信号（一跳局部拓扑优于 uniform 对照）仍是探索性的，
   且其匹配签名受组成混杂；本轮不把它升级为“已证明的组合增量”。
2. **换任务/数据，而不是继续在 ZINC 上堆组合机制。** 需要：
   - 实验测量性质（例如 lipophilicity 类），且同一局部结构在不同真实上下文重复出现；
   - 或逐原子/逐键、可原子映射的目标，或多任务监督，使“组合”成为预测上必要；
   - 或已知组合规则的合成基准，用于可识别性/可学习性检验；
   - 任何新方向都需要新的预注册，先做结构可识别性审计，再谈架构。
3. **本轮不提出 GPU 架构候选。** Stage C 门槛失败，且任务要求“支持进入下一阶段才提出
   一个机制候选”；因此不启动 GPU0/1、不训练 BJOC 新架构、不运行官方 test/valid。

---

## 8. 代码、结果与复现

```text
tracks/structure-contribution/results/explicit_composition_signal_audit_v1/
  PROTOCOL.md                         冻结协议（定义、层级、Stage B/C 门槛）
  REPORT.md                           本报告
  FREEZE.json                         Stage-A 哈希清单（protocol/code/input/artifacts）
  code/ecsa_objects.py                B0/B1/C、精确规范键、分组/统计/bootstrap
  code/run_explicit_composition_signal_audit_v1.py   Stage A/B 驱动
  tests/test_ecsa_objects.py          22 个 CPU 单测
  analysis/stage_a_summary.json       完整性、层级计数、不变性、C 划分、环基统计
  analysis/stage_a_tier_M0.json       11 对严格 B0 碰撞的完整结构记录
  analysis/stage_a_legacy_nesting.json L_old 与 B0/B1 的嵌套计数
  analysis/stage_b_labels.json        标签阶段全部统计 + Stage C gate
  analysis/stage_c_gate.json          Stage C 门槛与跳过原因
  analysis/stage_b_pairs.json         所有层级的逐对 Δy/公式分解（本地，可重生成）
  analysis/stage_a_tier_*.json        各层级完整分组/配对（本地，可重生成）
  analysis/stage_a_molecules.jsonl    逐分子结构表（本地，可重生成）
  analysis/stage_a_ring_instability.json  环基稳定性逐分子表（本地，可重生成）
```

复现：

```bash
# Stage A（标签盲；~140 s CPU）
uv run python tracks/structure-contribution/results/explicit_composition_signal_audit_v1/code/run_explicit_composition_signal_audit_v1.py stage-a

# 单测（22 passed）
uv run pytest -q tracks/structure-contribution/results/explicit_composition_signal_audit_v1/tests

# Stage B（先 verify freeze；~16 s）
uv run python tracks/structure-contribution/results/explicit_composition_signal_audit_v1/code/run_explicit_composition_signal_audit_v1.py stage-b
```

已运行的验证：

- 全量 10,000 行配对核对，复现 4 行冻结排除清单；
- 9,996 行 C_key/C_smiles 划分一一对应；
- 600 次原子重编号不变性检查，0 失败；
- 22 个单测全绿；
- Stage B 前 `FREEZE.json` 逐项哈希核对通过；
- PyG `y` 与 `encoded_train.pt` `y` 全量交叉核对，max abs diff = 0.0；
- 未读 official valid/test、未读内部 Locked、未读模型预测、未用 GPU。

---

## 9. 尚未解决的问题

1. **可学习性未测**：本审计只说明“在 ZINC-12k train 上不能识别独立、非公式的
   组合信号”，没有证明组合机制不可学习。Stage C 被预注册门槛阻止，不是失败结果。
2. **B1 的强度依赖定义**：B1 使用每根键的完整半径 1 带属性 patch；更弱的 motif
   词汇表可能不 injective。但既有 PSCD 审计已报告 64-motif bag 在 11k 分子上仅
   1 个非同构碰撞，说明“bag 近 injective”不是本定义的孤例；本轮没有重算该词汇表。
3. **目标公式实现**：我们复用了历史审计的 SA/logP/cycle 实现与常数；环基选择
   会让少数分子的 `cycle_penalty` 改变（7/3,894），相关 pair 已在
   `A4/L_old` 中标记并排除。ZINC 标签生成代码未被逐字节复现，因此
   “beyond-formula residual”只能上界到重构噪声，不能声称严格为零。
4. **A4/L_old 的残余尾部**：在排除环基不稳定与重构异常后为 0；但若未来用更忠实的
   目标实现，仍可能有 ≤0.01 量级的残余。它不足以支持“核心归纳偏置”，但可以作为
   一个更小、单独预注册的 refinement 问题。
5. **立体化学未覆盖**：C 在 constitution 层面定义；立体异构体不被区分。
   若目标/数据依赖立体化学，本结论不适用。

---

## 10. 最终决策

**`INSUFFICIENT_OR_CONFOUNDED`（强证据，限定 ZINC-12k train）。**

- 不建议进入下一轮 ZINC 组合架构实验；
- 不建议把本轮结果解释为“非 GNN/组合机制整体失败”；
- 建议先换到可识别组合效应的任务/数据，重新预注册；
- 本轮停在 CPU 审计与报告，不启动 GPU 训练。
