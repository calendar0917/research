# PROTOCOL — LCRC 解释忠实性与外部结构相关性联合实验 v1

Round / protocol ID: `lcrc-explanation-fidelity-alignment-v1`
目录：`tracks/structural-sufficiency/lcrc_explanation_fidelity_alignment_v1/`

状态：**规则在任何正式 GPU 运行之前冻结**。冻结后只允许追加：观测到的支持计数、
工程修复记录、门禁结果。任何科学规则不得在看到结果之后修改。

本轮只做一轮：不训练新架构、不新增 seed、不重选 checkpoint、不扩大数据或候选族。

---

## 0. 研究命题与判断对象

在**固定的 LCRC/GINE 预测模型**上，不同的自动连通子图发现方法，能否同时取得

1. **模型预测忠实度**（同一冻结模型对所选结构是否存在超出匹配随机结构的预测依赖）；
2. **独立的化学结构贡献定位**（是否覆盖 WC_LOGP 的生成式原子贡献质量）。

两条主线：

* **主实验（Main）**：单分子 WC_LOGP oracle。官方 Valid 420 + Test 420 全量 + 固定
  128 个 Train 分子（仅作描述）。只用 **WC_LOGP checkpoint**。
* **辅助实验（Aux）**：上一轮冻结的 823 对真实 Lipophilicity 配对（1646 端点）。只用
  **真实 Lipophilicity checkpoint**。编辑区域是**已知的结构差别，不是性质因果真值**；
  单分子 Top-1 不要求命中编辑区域，因此 Aux 只是辅助诊断，不能单独把不命中判为
  “不忠实”。

WC_LOGP 的逐原子贡献是一个**指定的生成式分解**，不是实验 logD 的化学因果机制。
“高 capture”只表示覆盖了已知生成贡献，不表示发现了化学机制。

---

## 1. 冻结资产与身份

| 资产 | 来源（只读） | 身份 |
|---|---|---|
| LCRC `L_lipophilicity` | `lcrc_explanation_task_relevance_sprint_v1/results/checkpoints` | SHA-256 + state hash + epoch + 参数数 |
| GINE `G0_lipophilicity` | 同上 | 同上 |
| LCRC `L_WC_LOGP` | 同上 | 同上 |
| GINE `T_WC_LOGP` | 同上 | 同上 |
| 823 对配对 / 原子映射 | 同上轮 `analysis/pair_catalog.jsonl.gz`、`row_atom_mapping.jsonl.gz` | SHA-256 |
| 端点目录（1646） | 同上轮 `analysis/pair_endpoints.jsonl.gz` | SHA-256 |
| WC_LOGP 原子贡献 | 同上轮 `analysis/ogbg-mollipo.npz`（与 route 轮 `oracle_contrib` 字节一致） | SHA-256 |
| 数据 / split | 官方 `ogbg-mollipo` 冻结缓存 + OGB scaffold split CSV | cache SHA-256 + split 行号集合 |

checkpoint 身份检查覆盖：文件 SHA-256、`state_dict` 哈希、参数量、epoch、全量
Train/Valid/Test 预测重放（容差 `tol_raw(x) = 1e-5 + 1e-6·max(1,|x|)`）。
任一不匹配 → 受影响方法 `CHECKPOINT_UNAVAILABLE` 停止，不得替换、重训或用随机初始化冒充。

---

## 2. 统一的候选结构集合（共享候选池）

对每个分子，生成**无标签、无 oracle、无编辑区域、无配对信息**的连通物理键子图集合。

* 候选大小 `k = 2..6` 条物理键。
* `k=2`：枚举**全部**共享端点的物理键对。
* `k=3..6`：从上一层保留的 top-12 候选出发，每次加入一条相邻物理键；每层最多保留
  beam width 12。
* 排序键（所有生成器与所有选择器统一）：`(score 降序, 键数升序, 排序后的物理键 ID 元组升序)`。

四个生成器（只读取结构与冻结模型场）：

| 生成器 | 打分（只用于发现） |
|---|---|
| `LCRC_GRAD` | `A(S) = |Σ_{e∈S} g_e| / sqrt(k)`，`g_e = ∂pred/∂m_e`（全 1 门控，LCRC WC/lipo） |
| `LCRC_LOCAL` | `B(S) = Σ_{v∈atoms(S)} |f(h_v)| / sqrt(k)`（LCRC 冻结局部输出） |
| `GINE_GRAD` | 与 `LCRC_GRAD` 同式，梯度来自 GINE 物理键门控 |
| `TOPO_S2` | 仅 `n_s2_affected(S)`（图拓扑；不读模型/标签/oracle） |

**共享候选池** = 四个生成器在 `k=2..6` **每层保留的候选**的并集，按排序后的物理键 ID
元组去重（再按 `(键数, 元组)` 规范排序）。同一分子的所有方法只能从这个池中选结构。

记录：每分子池大小、各生成器贡献、按 k 的构成、生成器两两重叠、各生成器评估数量；
以及**每个方法是否命中它自己单独运行时的 Top-1**（`hit_own_top1`）。定义：

* `A/B/E/D` 的 own pool = 对应生成器自己的保留候选集合；
* `C` 的 own pool = `LCRC_GRAD ∪ LCRC_LOCAL` 的保留候选（LCRC 家族单独能产生的集合）；
* `F` 的 own pool = `GINE_GRAD` 的保留候选。

own pool 只用于审计，**不参与**从共享池中选结构。

---

## 3. 固定候选选择策略（同一池）

| 方法 | 模型 | 打分 `score(S)` | 选择剂量 |
|---|---|---|---|
| `A_LCRC_GRAD` | LCRC | `|Σ_{e∈S} g_e| / sqrt(k)`（全 1 梯度） | — |
| `B_LCRC_LOCAL` | LCRC | `Σ_{v∈atoms(S)} |f(h_v)| / sqrt(k)` | — |
| `C_LCRC_MASK50` | LCRC | `|y(1) − y(m_S=0.5)|` | `m_S=0.5` |
| `D_TOPOLOGY` | 无 | `n_s2_affected(S)` | — |
| `E_GINE_GRAD` | GINE | 同 `A`，GINE 梯度 | — |
| `F_GINE_MASK50` | GINE | 同 `C`，GINE 门控 | `m_S=0.5` |

同分按 `(键数升序, 键 ID 元组)`。每个方法只取 **Top-1**（主指标就是 Top-1）。

`C`/`F` 的选择剂量是 0.5，**真正的忠实性评价剂量是 0**。两者的同族指标关联必须在
报告中单列（见 §5.4），不得把 `C`/`F` 更高的内部屏蔽效应冒充为独立确认。

**随机对照**（主实验与辅助实验同一规则）：

* 同一分子、**键数相同、涉及原子数相同**、连通；
* 排除与 Top-1 完全相同的集合；
* 固定 seed `20261117`，按 `SeedSequence([20261117, split_code, row_id, selector_code])`；
* 最多 16 个，最多 512 次增长尝试，按生成顺序接受；
* **少于 8 个标记 `INSUFFICIENT_RANDOM`（缺失），不放宽匹配条件、不用 0 补齐、不丢弃分子**；
* 采样器允许多次闭合环（一个 k 键连通子图落在 a 个原子上的独立环数恰为 `k−a+1`），
  这只改变提案分布，不改变接受规则。

---

## 4. 主实验 — 单分子 WC_LOGP oracle

范围：官方 `valid` 全部 420、`test` 全部 420、`train` 固定 128（选择规则沿用
`sha256(20261117|ogbg-mollipo|train|row_id)` 升序前 128；**仅作描述，不做推断**）。

对分子 `M`，oracle 原子贡献 `c_v` 满足 `Σ_v c_v = true_WC_LOGP(M)`（容差
`1e-8 + 1e-8·max(1,|·|)` 的图级闭合另按上游 `1e-6` 复核）。

```
capture(S)        = Σ_{v∈atoms(S)} |c_v| / Σ_v |c_v|       分母为 0 → 缺失（不写 0）
oracle_enrichment = capture(Top1) − median_{controls} capture
```

报告（分 Valid / Test，Train-128 仅描述）：

* `capture(Top1)` 与随机中位数的均值、中位数、四分位；
* `oracle_enrichment` 的均值/中位数/四分位、CI、`>0` 比例、`=0` 比例、`<0` 比例；
* 按实际候选大小 `k=2..6` 分层；
* 正 enrichment 的集中度：最大 5% 分子贡献的正 enrichment 份额；
* 各方法之间的分子级 paired difference 及其 bootstrap CI；
* 缺失计数（oracle 分母为 0、`INSUFFICIENT_RANDOM`）。

**禁止只报告均值**；必须给出零膨胀与集中度。

---

## 5. 同一候选的模型忠实性

对每个方法选出的同一个 `S`，用**对应冻结模型**计算

```
delta0    = pred(m=1) − pred(m_S=0)
delta_half= pred(m=1) − pred(m_S=0.5)
```

* `LCRC`：`CONDITIONAL_S1_S2_EVIDENCE_MASK`（S1 键证据 + S2 路径证据 `m_e1·m_e2`）；
* `GINE`：每个物理键一个共享门控，乘在该键两个方向、所有层的 message 上；
* `D_TOPOLOGY` 是模型无关选择，故在 **LCRC 与 GINE 上各评一次**，并分别与其同模型随机对照比。

```
fidelity_advantage      = |delta0(Top1)| − median_{controls} |delta0|
fidelity_advantage_std  = fidelity_advantage / σ_train(task)
```

记录：

* `|delta0|`、`delta0`、`delta0/σ_train`、`|delta_half|`、`half/full`、符号一致性；
* 随机对照的 `median |delta0|`、advantage、CI；
* `LCRC` 的 `local_sum` 与 `rho` 响应分解及闭合误差 `delta0 − (delta_local + delta_rho)`；
* `n_s2_internal / n_s2_boundary / n_s2_affected`（Top-1）；
  `internal = 两条键都在 S 内`，`boundary = 恰好一条在 S 内`，`affected = internal+boundary`；
* 同一分子上 `oracle_enrichment` 与 `fidelity_advantage_std` 的配对关系
  （同时为正 / 只有 oracle / 只有 fidelity / 两者都非正）。

**门控语义不同**：LCRC 门控在显式 S1/S2 证据上，GINE 门控在 message 上；禁止直接比较
两者的原始 `|delta0|` 大小，跨模型比较只用外部 oracle capture。

### 5.4 MASK50 的同族关联

`C`/`F` 的选择分数与 `m=0` 评价同属“屏蔽效应”族。必须报告：

* `C`/`F` 的 `mask50_score(Top-1)` 与其 `|delta0|` 的关系（分子级相对位置/比例）；
* `C`/`F` 的 Top-1 与 `A`/`E` 的 Top-1 的重合率；
* 结论只能写成“有限干预评分在外部 oracle 定位上是否有独立改进”，不能写成
  “内部屏蔽效应更大 ⇒ 解释更忠实”。

---

## 6. 辅助实验 — 真实 Lipophilicity 分子配对

复用 823 对真实配对、1646 端点、已核验的原子映射与编辑区域；**不扩大配对范围、不按标签
挑配对**。候选池与选择器完全沿用 §2–§3（使用 Lipophilicity checkpoint）。

报告：

1. 每个端点 Top-1 与已知编辑区域的重合：atom Jaccard、atom recall、bond recall、
   attachment-boundary hit，及其相对同键数/同原子数随机连通子图的 advantage；
2. 各解释评分选中结构在**共同核心 / 变量区域 / 附件边界**上的分布
   （选中原子与键落入三类区域的比例；变量区域 = 编辑原子及其全部入射键，
   附件边界 = 附件核心原子 + 连接变量片段的键）；
3. 完整分子 Valid/Test RMSE 与配对预测变化误差 `|Δŷ − Δy|`、方向正确率；
4. 正确预测配对与错误预测配对中的解释差异（同一批固定阈值：`|Δŷ−Δy| ≤ 0.25`；
   `direction_correct` 需 `|Δy| ≥ 0.1`）；
5. 分层：`Train–Train`、`at least one endpoint non-Train`、`both non-Train`。
   `both non-Train` 在当前 823 对中为 0 → `NO_SUPPORT`，不得编造。

**统计单位**：不是 823 个独立样本，而是共享端点连通分量（component）的 cluster
bootstrap（2000 次，seed `20261117`），分层的 component 数 < 30 → `LOW_SUPPORT`。
Aux **不做** fidelity 剂量实验（本轮不要求），只报告结构定位与预测误差。

---

## 7. 统计

* Main：分子是独立单位（官方 split 内）。每个 split 单独 bootstrap（2000 次，seed
  `20261117`，percentile 2.5/97.5）。paired difference 在同一批分子上重采样。
* Aux：共享端点 component cluster bootstrap（同上参数）。
* 所有分母、缺失、零值都落盘；聚合脚本只读原始记录，不重算候选。
* Test 一律标注 `DEVELOPMENT_TEST`（源轮已观测），不是最终盲测。

---

## 8. 预注册决策规则（跑前冻结）

`σ_train` = 该任务官方 Train 目标的标准差。有效样本比例 = 可评估分子 / 范围内分子。

**支持条件（每个方法、每个 split）**

* `SUPPORT_OK`：可评估比例 ≥ 0.80；否则 `INSUFFICIENT_SUPPORT`。
* `oracle_supported`：enrichment 均值的 bootstrap 95% CI 下界 > 0，点估计 ≥ 0.02，
  且集中度 `top5% 份额 ≤ 0.50`（防止少数大效果样本主导）。
* `fidelity_supported`：`fidelity_advantage_std` 均值的 CI 下界 > 0，点估计 ≥ 0.01，
  且同一集中度条件成立。
* `paired_improvement(m1 − m2, oracle)`：同一分子上 enrichment 差的均值 CI 下界 > 0
  且点估计 ≥ 0.02。
* `lrc_specific_advantage`：`max{A,B,C}` 与 `max{E,F}` 的 enrichment 点估计之差
  （同分子配对）CI 下界 > 0 且点估计 ≥ 0.02。

**决策标签（可同时出现，但互相矛盾时不并列）**

* `LOCAL_FAITHFUL_AND_RELEVANT`：`B` 在 Valid 上同时 `oracle_supported` 与
  `fidelity_supported`，且 Test 上两项点估计同号。
* `LOCAL_RELEVANT_NOT_FAITHFUL`：`B` 在 Valid 上 `oracle_supported` 但
  `fidelity_supported` 不成立（Test 点估计仍同号）。
* `FINITE_MASK_IMPROVES_ALIGNMENT`：`paired_improvement(C − A, oracle)` 成立
  （这是**外部** oracle 定位上的改进，不是内部屏蔽效应）。
* `NO_LCRC_SPECIFIC_ADVANTAGE`：`lrc_specific_advantage` 不成立，且 LCRC 与 GINE
  两侧都 `SUPPORT_OK`。
* `UNRESOLVED`：上述任一条所需支持不足（缺失 > 20%、CI 全跨 0、集中度不达标、
  门禁失败、或 Valid 成立但 Test 点估计反号）。

决策主 split 固定为 **Valid**；Test 只作为开发期复现条件。**不得事后更换 split 或指标。**

---

## 9. 执行与算力

`res` 两张 A100 并行：

1. `lfa-fields-lcrc`（`res-gpu0`）与 `lfa-fields-gine`（`res-gpu1`）：分别计算 LCRC/GINE
   在 WC_LOGP 与 Lipophilicity 两个任务上的全 1 门控梯度、LCRC 局部输出与基准预测，
   逐分子追加落盘（可恢复）。
2. `lfa-pool`（CPU，`res-local`）：读取两份场文件，生成**唯一**共享候选池 + 各方法
   模型无关选择 + 匹配随机对照；写出 `pool_main.jsonl` / `pool_aux.jsonl` 与 SHA-256 清单。
3. `lfa-select-lcrc`（`res-gpu0`）与 `lfa-select-gine`（`res-gpu1`）：**读取同一池文件**
   （入口强制校验 SHA-256 与清单一致），执行 MASK50 扫描与 `m=0/0.5` 干预。
4. 本地聚合：oracle capture、fidelity、分层、bootstrap、决策、报告。

约束：不训练；每卡软预算 1 GPU·h、硬预算 2 GPU·h；逐 split/逐分子落盘；内存硬上限
（单批 atoms/bonds 上限与分块）保护宿主；超预算立即停止并保留已完成结果。

---

## 10. 正确性门禁（任一失败 → 停止受影响方法）

1. WC_LOGP 图级与配对级原子贡献闭合；原子重编号下 capture 不变。
2. 四个 checkpoint 的文件 SHA-256、state hash、参数量、epoch、全量预测重放。
3. 所有模型 `eval()`、参数冻结；场/选择阶段前后参数哈希不变。
4. 全 1 门控等价：LCRC（4 个 sample 分子/任务）与 GINE（同）在 `tol_raw` 内复现冻结前向；
   新增的 GINE 批量实现与已核验的单图实现逐值一致。
5. 单键中心差分 `eps=1e-3` 与门控梯度一致；原子 + 物理键重编号等价。
6. 候选池方法一致性：所有方法读取同一文件（SHA-256 校验）；`pool_main/pool_aux` 的
   生成器成员可重放。
7. 搜索标签盲：API 无标签/编辑区域/oracle 参数；shuffle 标签与 oracle 数组后候选池
   逐字节不变（单元测试）。
8. `m=0.5` 与 `m=0` 的真执行：沿 `m(t)=1−t` 的 64 步积分与 `pred(0)−pred(1)` 闭合，
   `pred(0.5)` 与积分中点一致；门控权重分布符合 `internal=0 / boundary=0.5 / 其它=1`
   （半剂量）与 `internal=boundary=0`（全剂量）。
9. split 身份与官方 CSV 一致；样本 row id 集合与规则可重放。
10. 所有统计分母、缺失、零 enrichment 计数可从原始记录重放。

---

## 11. 明确禁止

* 训练新模型/新解释器、新增 seed、重选 checkpoint、改动架构或门控语义；
* 按标签、编辑区域、oracle 贡献或真实屏蔽效果生成/筛选候选；
* 用 `C`/`F` 的内部屏蔽效应替代外部 oracle 验证；
* 直接比较 LCRC 与 GINE 的原始门控效应大小；
* 把编辑区域称为因果真值，把 WC_LOGP 分解称为化学机制；
* 放宽随机对照的匹配条件或丢弃缺失分子；
* 把 823 对当作 823 个独立样本；把 Test 称为未触碰的最终盲测；
* 看到结果后修改指标、候选范围、阈值或 split。
