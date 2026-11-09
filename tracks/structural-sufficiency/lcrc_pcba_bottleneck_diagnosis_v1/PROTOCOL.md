# PROTOCOL — LCRC MolPCBA 性能瓶颈诊断 v1

- **round_id**: `lcrc-pcba-bottleneck-diagnosis-v1`
- **result_label**: `POST_HOC_EXPOSED_TEST_DIAGNOSIS`
- **性质**: 只读、离线、描述性诊断。**不训练、不调参、不做 checkpoint 选择、不设计新架构。**
- **冻结时间**: 2026-11（本文件在第一次读取 Test 逐分子标签之前写定；此后不再改动分析口径）

本文件冻结的是「分析定义、数据身份、执行范围与允许偏差」。所有阈值在读到本轮
任何结果之前固定；如果数据不支持某个结论，输出 `UNRESOLVED`，不得事后改阈值。

---

## 1. 问题

解释为什么 W2、S3、RC-REAL、BOC-FULL 在引入不同结构计算后，官方 MolPCBA
Test AP 仍然相近（约 0.2506–0.2559）。三个竞争解释可以同时成立：

| 代号 | 解释 |
|---|---|
| H1 | 预测函数趋同：不同架构在重要分子上的排序/错误/证据高度重合 |
| H2 | 学到的结构信息缺乏迁移性：新增优势集中在低新颖性局部环境，跨 scaffold 减弱或反转 |
| H3 | 任务稀疏性与评价不稳定性：收益可重复但被少数任务/低支持任务抵消 |

**不做**：内部 embedding 分析、新 attention 可视化、模型内部机制归因。
**不做**：把 Test 当作独立确认集。Test 已暴露，本轮只做事后工程分析。

---

## 2. 数据身份（只读）

### 2.1 官方数据

| 项 | 路径 | 用途 |
|---|---|---|
| molecule mapping | `data/ogb/ogbg_molpcba/mapping/mol.csv.gz` | 只读 `smiles` 列 |
| official splits | `data/ogb/ogbg_molpcba/split/scaffold/{train,valid,test}.csv.gz` | 行 id |
| official raw labels | `data/ogb/ogbg_molpcba/raw/graph-label.csv.gz` | **只经** 历史 gate 读取 |

Test 标签只通过历史 `cc_data.read_official_test_labels(task, freeze_path=...)`
读取（它会先校验冻结清单、重新校验 Train/Valid 原始标签与冻结 artifact 一致，
再返回 Test 行标签）。Validation 标签用训练 artifact（Test 行全 NaN）。

### 2.2 模型（best-Validation checkpoint，单 seed）

| 逻辑名 | 模型 | epoch | Test logits 来源 |
|---|---|---|---|
| `W2` | W2 | 24 | `lcrc_pcba_boc_v1/runs/freeze/test_predictions/W2_best_valid.npy` |
| `S3` | S3 | 29 | `lcrc_pcba_s3_selective_v1/analysis/test_predictions/S3_best_valid_epoch29.npy` |
| `RC_REAL` | RC-REAL | 33 | `lcrc_pcba_boc_v1/runs/freeze/test_predictions/RC_REAL_best_valid.npy` |
| `BOC_FULL` | BOC-FULL | 29 | `lcrc_pcba_boc_v1/runs/freeze/test_predictions/BOC_FULL_best_valid.npy` |
| `BOC_BOND` | BOC-BOND | 30 | `lcrc_pcba_boc_v1/runs/freeze/test_predictions/BOC_BOND_best_valid.npy`（辅助） |

实际路径、SHA256、形状、行 id 一律从 manifest 读，并由
`analysis/prediction_inventory.json` 记录；上表只是预期。

### 2.3 Validation logits

历史轮次没有保存 Validation 逐分子 logits（只保存了 hash 与逐任务 AP）。
因此本轮只做一次**只读 Validation 前向**：加载上表同一冻结 checkpoint，
`model.eval()` + `torch.inference_mode()`，官方 Valid 行、batch=256、CPU。
产物写入 `_local/valid_predictions/`（`*.npy` 被 gitignore），SHA256 与重放 AP
写入 inventory。不允许 `backward()` / `optimizer.step()` / 任何梯度路径。

### 2.4 对齐要求

- Test/Valid 预测必须是 `[43793,128]`。
- 所有模型的行 id 必须与官方 split 文件逐元素相等；若历史文件来自不同轮次，
  先按官方 row id 重排，并记录重排映射与前后 hash。**不允许假设 `.npy` 行序天然一致。**
- 每个任务只用该 split 上有观测标签（非 NaN）的分子。
- 任一模型身份/行序/缓存版本无法确认 → 标 `UNVERIFIED_ARTIFACT` 并排除。

---

## 3. 评价定义

- **官方 AP**：`ogb.graphproppred.Evaluator("ogbg-molpcba")`，输入
  `sigmoid(logits)`（float64 → float32，与历史评分路径一致）。
- **独立 AP**：`sklearn.metrics.average_precision_score`，逐任务、NaN 掩码。
- 可评分任务（`ALL_split`）：该 split 上同时有 ≥1 正例和 ≥1 负例。
- 历史冻结值重放容差：`|ΔAP| ≤ 1e-4`。超过即停该模型的相关分析。
- PM 值（概率）只在 AP 里使用；排序统计一律直接使用 logits（单调等价，且不会
  因 float32 sigmoid 饱和引入并列）。

---

## 4. 任务集合（按标签定义，不按 AP 筛选）

| 集合 | 定义 |
|---|---|
| `ALL_valid` / `ALL_test` | 该 split 上可计算 AP 的 127 个任务 |
| `HIGH_SUPPORT` | Validation 正例 ≥20 **且** Test 正例 ≥20，且两个 split 都有负例 |
| `LOW_SUPPORT` | Test 正例 <20（只描述，不用于 H1 主结论） |

可靠性分层（用于 P3 分组内 AP）：

| 层 | 组内正例数 | 用法 |
|---|---|---|
| main | ≥20 | 进入主要诊断 |
| descriptive | 5–19 | 只描述 |
| record | 1–4 | 只记录 |
| `NA` | 0 或无负例 | 不可评分 |

---

## 5. P1 预测函数趋同

固定模型对（主）：`W2–S3`、`W2–RC_REAL`、`W2–BOC_FULL`；
辅助：`RC_REAL–BOC_FULL`、`BOC_FULL–BOC_BOND`。不做全组合矩阵。

**指标 A：Spearman**。每个 (pair, split, task) 在共同观测分子上计算
`rho = Spearman(logits_A, logits_B)`，平均秩处理并列。禁止跨任务混合。

**指标 B：Top-K 重合**。每个任务按 logits 降序，`K1 = ceil(0.01n)`、
`K5 = ceil(0.05n)`（n = 该任务观测分子数）；并列用 row id 升序打破。
报告 Jaccard overlap 和每个模型 Top-K 内的真实正例数。

**指标 C：互斥正例发现**（A=W2，B=新模型）。同一 K 下统计
`both / only_W2 / only_new / neither` 的正例数（K1 与 K5 各一套）。
另外对每个真实正例计算其归一化排名百分位
`pct = (低于该分子的观测分子数 + 0.5×并列数) / n`（1 = 最好），
并汇总 `Δpct = pct_new − pct_W2`（正值 = 新模型排名改善）的中位数。

三个指标在 Validation 与 Test 上各算一遍。

**预注册判读规则**（只看 HIGH_SUPPORT）：

- `RANKING_SIMILAR`：两个 split 的 rho 中位数都 ≥ 0.80。
- `DECISION_SIMILAR`：两个 split 的 Top-1% Jaccard 中位数都 ≥ 0.50，
  且 Top-5% Jaccard 中位数都 ≥ 0.60。
- 结论标签：
  - `H1_SUPPORTED_SIMILAR_RANKING_AND_DECISIONS`：两者都成立；
  - `GLOBAL_RANKING_SIMILAR_BUT_DECISION_REGION_DIFFERS`：RANKING_SIMILAR
    成立但 DECISION_SIMILAR 不成立；
  - `H1_MIXED`：单 split 成立或两个方向冲突；
  - `H1_NOT_SUPPORTED`：rho 中位数 < 0.80。

以上只是**输出标签**，不构成"内部表示相同"的证明。

---

## 6. P2 Train-only 局部结构支持度

**定义**：有根 radius-2、带原子/键属性的 WL 颜色签名（`WL2_ID`）。

- 原子初始签名：`atomic number, formal charge, aromaticity, total H count, total degree, in-ring`
- 键签名：`bond type, aromaticity, conjugation, in-ring`
- 两轮 refinement：每轮把 `(self_color, sorted multiset of (bond_sig, neighbor_color))`
  用 SHA256（稳定 JSON 序列化）哈希。
- 字典只在**官方 Train** 分子上建立：`train_molecule_frequency[WL2_ID]` =
  包含该 WL2 的**不同 Train 分子数**（同一分子内重复只记 1）。
- 每个 Valid/Test 分子（保留原子重数）：
  `fraction_unseen = #{atom: freq==0}/n_atoms`，
  `fraction_rare = #{atom: 0<freq<=4}/n_atoms`，
  `median_train_support`，`n_heavy_atoms`。
- 解析失败/零原子必须记录，不得静默删行或改变行顺序（这些分子的特征写 NaN，
  并进入 `unresolved` 组，从分组 AP 中排除并在 summary 计数）。

**分组（两个 split 完全相同）**：

| 组 | 定义 |
|---|---|
| `SEEN` | `fraction_unseen == 0` |
| `PARTIAL_NOVEL` | `0 < fraction_unseen <= 0.25` |
| `HIGH_NOVEL` | `fraction_unseen > 0.25` |
| `UNRESOLVED` | 解析失败/零原子 |

不做 Train×Test 全量 Tanimoto 搜索，不按 Test AP 重选阈值。

---

## 7. P3 结构支持度 vs 收益

对 `S3−W2`、`RC_REAL−W2`、`BOC_FULL−W2`：每个 (split, task, novelty group)
用组内观测分子重算 AP（需同时有正负例），`delta_AP = AP_new − AP_W2`。

- 只解释**同 split、同 task、同组**的模型差值，不比较不同组的绝对 AP。
- 组内正例 <20 的 cell 不进主要结论。
- **任务匹配**：两个 split 的 main cell 任务集合并不相同（例如 HIGH_NOVEL 在
  Validation 有 37 个、Test 有 50 个可诊断任务）。因此方向一致性判读使用
  **两个 split 都 main-eligible 的任务交集**；全部 cell 的版本同时写入
  summary（`trend_all_cells`），两个版本若结论不同必须一并报告。
- 判读（预注册）：
  - `GAIN_CONCENTRATED_ON_SEEN`：在 ≥20 个 shared main cell 可用的前提下，
    `mean delta_AP(SEEN) − mean delta_AP(HIGH_NOVEL) > 0.001`，
    且三个组按 SEEN→PARTIAL→HIGH 单调下降；
  - `GAIN_REVERSED_ON_NOVEL`：SEEN 上有正增益而 HIGH_NOVEL 上为负；
  - 两个 split 方向一致才可升级为 supported，否则 `NOVELTY_EFFECT_UNRESOLVED`；
  - 任一组 main cell <20 → `NOVELTY_EFFECT_UNRESOLVED`。
- 排名变化：HIGH_SUPPORT 任务内正例的 `Δpct`，按 novelty group 汇总
  （先在 task 内归一化，再跨 task 取中位数；同时报告 pooled 中位数）。
- 混杂检查：报告三组的 `n_heavy_atoms` 分布；再按 **Validation** 重原子数的
  1/3、2/3 分位切成三个固定区间做描述性分层。不搜索其它分层方案。

---

## 8. P4 任务支持度与宏 AP 抵消

用本轮重算的逐任务 AP（5 个模型、两个 split）。对每个模型相对 W2：

- 127 个可评分任务上的平均 `delta_AP`、正值和、负值和、净值；
- 最大 5 个正向/负向/绝对变化（含 task id、Test 正例数、贡献值）；
- 正例 ≥20 / 5–19 / 1–4 任务上的平均差与改善任务数；
- 去掉绝对变化最大的 1/3/5 个任务后的**描述性**平均差（不替代官方 macro AP）；
- Validation 与 Test 的正例数、正例率、AP 差；同任务 Valid 增益在 Test 反转的统计；
- 低支持任务（Test 正例 <20）对净差值的总贡献与其占绝对变动的比例。

判读（预注册）：若低支持任务对净值/绝对变动的贡献 ≥25% 且方向与高支持任务相反，
记 `H3_SUPPORTED_BY_SUPPORT_ASYMMETRY`；若贡献 <10%，记
`H3_NOT_SUPPORTED_BY_SUPPORT_ASYMMETRY`；中间记 `H3_MIXED`。

---

## 9. 正确性审计（运行前写定，运行后逐项勾选）

1. checkpoint / logits 版本身份可追溯（SHA256 链）。
2. Valid/Test 逐行严格对齐（与官方 split 文件逐元素相等）。
3. 每个任务的排序与 AP 只在其观测标签上计算。
4. 官方 OGB AP 与独立 sklearn AP 一致（同一模型同一 split，`|Δ| ≤ 1e-6`）。
5. 历史 AP 重放与冻结登记值一致（`|Δ| ≤ 1e-4`）。
6. Spearman / Top-K / Jaccard 用小型手写例子验证（`tests/`）。
7. Top-K 候选不跨 assay 混用（按任务独立取 Top-K，测试覆盖）。
8. WL2 字典只用 Train SMILES 构造（代码路径检查 + summary 记录）。
9. 同一 Train 分子内重复 WL2 不重复计入 molecule frequency（测试覆盖）。
10. Valid/Test 结构特征计算不读活性标签（代码路径 + summary 记录）。
11. 不同 split 使用相同的结构分组阈值（常量单点定义）。
12. 所有低支持/无正例/不可评分任务显式标记。
13. 不修改历史实验、权重、原预测文件或旧报告（`git status` 检查）。
14. 不对 Test 预测做重校准、融合或优化。
15. 新推理无 `optimizer.step()` / `backward()` / 训练数据梯度。

---

## 10. 允许偏差

- Validation logits 是由冻结 checkpoint 重新前向得到的，与历史 GPU 前向
  **bitwise 不可比**（CPU/GPU 浮点差异）。允许的偏差：
  - 聚合重放 Valid AP 与冻结记录 `|Δ| ≤ 1e-4`（硬门禁）；
  - 正例 ≥20 的任务的逐任务 AP 与历史记录 `|Δ| ≤ 1e-3`（硬门禁）；
  - 极小任务（正例 <20）允许个别近并列分子重排导致的逐任务 AP 漂移，
    只记录漂移分布，不参与任何结论。
- Test logits 直接用历史冻结文件，不重跑。
- 结构解析允许 RDKit `unsanitized_fallback`，必须记录模式与计数。

## 11. 停止条件

- P1、P4 优先完成；P2/P3 若 CPU 结构构建超时（>30 min）则只报告已完成部分，
  并在 REPORT 中写明缺失。
- 不生成图像、dashboard、或额外统计扫描。
- 本轮结束即停止，不进入下一轮架构开发。

---

## 12. 修订记录（事后声明，必须与结果一起阅读）

本节记录协议写定之后发生的、**不改变任何判读阈值**的澄清。每一条都说明触发原因
与对结论的影响。

1. **§10 逐任务 Valid AP 容差**（执行 Validation 重放时）
   - 触发：CPU 与历史 GPU 前向的浮点差异使 task 11（53 个正例）的逐任务 AP
     漂移 2.6e-4，而聚合 AP 只漂移 5.8e-6。
   - 处理：聚合 AP 仍用 1e-4 硬门禁；正例 ≥20 的任务用 1e-3 硬门禁；正例 <20 的
     任务只记录漂移。这不是结果导向的放宽（门禁仍然失败即中止），也不影响任何
     P1–P4 的判读（P1 的排序统计直接使用 logits，不使用重放的 AP）。
   - 实际漂移：聚合 ≤5.8e-6；正例 ≥20 的任务逐任务 AP ≤2.6e-4（W2 task 11，
     53 个正例），任务级中位数 ≤4e-9。该漂移只影响"重放一致性"记录，不影响
     P1 的排序统计（直接用 logits）与 P4 的逐任务差值（同一 split 内的模型差，
     共用同一份 logits 路径）。

2. **§7 方向一致性改用任务匹配（shared main cells）**
   - 触发：两个 split 的组内 main cell 任务集合不同（HIGH_NOVEL：Valid 37 个任务、
     Test 50 个任务），直接比较均值是不同任务集合的平均。
   - 处理：主判读在两个 split 都 main-eligible 的任务交集上做；全部 cell 的版本
     仍写入 `novelty_vs_gain_summary.json`（`trend_all_cells`）。
   - 影响检查：两种版本对三个 primary pair 给出**相同的定性结论**（没有任何
     pair 达到 supported；FULL/RC 为 unresolved，S3 为 no-clear-trend），
     `agree_all_cells_vs_shared` 字段记录了该核对。

3. **P4 中 `H3_*` 判读的两个口径**
   - `H3_SUPPORTED_BY_SUPPORT_ASYMMETRY` 只看"低支持任务占绝对变动 ≥25% 且符号与
     高支持相反"；同时报告低支持对**净值**的贡献份额，避免把 S3 这类
     "同向但由低支持主导"的情形误读。两个份额都写进
     `macro_ap_decomposition.json`，报告同时给出。

以上修订均未改变模型集合、任务集合、结构分组阈值、Top-K 定义或任何模型选择。

4. **P1 低支持汇总的任务集合**（写报告前核对产物时）
   - 触发：`lc_rank.py` 初版把"低支持"实现为"可评分任务中不属于 HIGH_SUPPORT 的部分"
     （21 个，含 2 个 Test 高/Valid 低支持任务），与 §4 冻结的 `LOW_SUPPORT`
     （Test 正例 <20，19 个）口径不一致。
   - 处理：`spearman_low_support` / `jaccard_*_low_support` 改为严格使用冻结的 19 个任务；
     那 2 个不属于任何冻结集合的任务单独记入 `spearman_not_in_frozen_sets` /
     `jaccard_*_not_in_frozen_sets`，不静默丢弃。重跑后高支持与全任务统计不变。
   - 影响检查：低支持 rho 中位数由 0.479–0.626（21 个）变为 0.483–0.633（19 个），
     "低支持任务排序相似度低于高支持任务"的定性描述不变；报告中的 0.48–0.63 使用新口径。
