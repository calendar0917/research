# PROTOCOL — LCRC 事后诊断 v1（P0 勘误 + P1-A/B/C）

Round ID: `lcrc-posthoc-diagnosis-v1`
Track: `tracks/structural-sufficiency/lcrc_posthoc_diagnosis_v1/`
基准提交（上一轮）：`6cf9ca4eb3ddd45701c73baea66e80642034b756`
本文件在计算任何新统计量之前写出。**本轮不是预注册实验，不产生新的独立验证。**

---

## 0. 性质与边界

* 本轮只读取上一轮已经保存的冻结预测、冻结 JSON 产物、冻结结构来源。
* **GPU 用量必须为 0**；不训练、不微调、不重新选择 checkpoint。
* 不调用 `ptb_score.py`、`edit_probe.py --stage audit` 或任何一次性 Test 评分入口。
* 不重新执行 Test 推理；所有指标只在这些已保存预测上离线重算。
* 不生成新的 Train/Valid/Test 划分；NCI1 的 fold 是上一轮冻结划分的**只读重建与哈希核验**。
* 不读取 Local-edit Audit 的原始标签；只使用 `probe_audit_scores.json` 已保存的
  `d_pred`、`e_delta`、`pair_id`、`component`、`class_id`、`same_var_heavy`。
* 所有使用已曝光 Test 或 Audit 的新分析标记
  **`POST_HOC_EXPOSED_EVALUATION`**，不得称为新的独立验证。
* 如果所需原始预测文件不存在，记录 `MISSING_ARTIFACT`；不重训、不重评分补齐。

---

## 1. P0：报告与统计口径勘误

### P0-1 FULL/C1 胜负计数

按数据集主指标方向（回归 RMSE 越低越好；分类 ROC-AUC / macro ROC-AUC 越高越好），
对 10 个数据集逐任务比较三 seed 均值：

* 分子 6 个：ESOL、FreeSolv、BACE、BBBP、Tox21、ogbg-molhiv。
* TU 4 个：NCI1、PROTEINS、IMDB-BINARY、IMDB-MULTI（5 folds × 2 seeds 的均值）。

只修正描述文本，不修改任何原始分数。

### P0-2 typed-S2 计数范围

从 `analysis/mechanism_audit.json` 的 13 条 `intervention.rows` 逐行重算：

* `typed_s2_off.mean_abs_delta` 与同 checkpoint 的 `global_g_off`、
  `rho_mean_h2_off` 比较。
* 同时报告两种固定判据：
  1. **严格最大**：typed-S2 的 `mean_abs_delta` 严格大于另外两个通道。
  2. **5% 边际**：typed-S2 至少比次大通道高 5%，否则记为 “comparable/exception”。
* 分别报告 12 个本轮新 checkpoint 与 13 个含 Lipophilicity 历史 checkpoint 的计数。

明确说明：这是**固定权重下不同宽度输入块置零的敏感度比较**，不是训练后
S2 必要性的证明，也不支持因果结论。

### P0-3 分子任务跨 seed 平均差的 CI

历史 `code/ptb_report.py::molecule_summary` 取三个 seed 的 `boot_lo` 最小值和
`boot_hi` 最大值。该区间是 **`SEED_WISE_CI_ENVELOPE`**（各 seed 行 bootstrap
区间的外包络），不是跨 seed 平均 C1−A0 效应的 95% CI。

保留历史 `statistics.json` 与每个 seed 的 bootstrap 数值不变。新定义：

* 目标量：`Δ_mean = mean_seed[metric(C1, seed) − metric(A0, seed)]`。
* 三个 seed 的 checkpoint 均视为固定。
* 每次 bootstrap 从同一 Test 行集合抽样；**同一数据集、同一 replicate 内，
  三个 seed 与 A0/C1 使用完全相同的行索引**。
* 每次分别重算每个 seed 的任务主指标差，再对三个差值取平均。
* 重复 **2000** 次，随机种子 **20261103**，取 2.5%/97.5% 分位数。
* 重抽规则：
  * 回归：Test 分子行有放回重抽。
  * 单任务二分类：正、负样本分别有放回重抽，保持原类别数量。
  * Tox21：按分子行重抽；只使用完整原始 Test 上已定义的 12 个任务集合。
    若某次重抽使其中任一任务缺少正类或负类，该次记为无效并重抽，
    直到 2000 次有效或达到 10000 次尝试。未取得足够有效重复则记 `UNAVAILABLE`。
    不可把无法计算的 AUC 填成 0。
* 同时报告：每 seed 实际主指标、每 seed 配对差、配对差的 seed 标准差、
  新条件 bootstrap CI、旧 seed-wise envelope。
* 结论只能条件于当前已训练 checkpoint；3 个模型 seed 的变异单独列出，
  不得声称行 bootstrap 已覆盖训练 seed 不确定性。

---

## 2. P1-A：MolHIV 结构支持度与排序错误

### 2.1 只读数据

* 6 个 `runs/mol_ogbg-molhiv_s<seed>_<model>/test_predictions.npz`
  （`rows`、`pred`、`y_true`）与对应 `test_eval.json`、`run.json`。
* 官方 SMILES 与 scaffold split 从冻结来源
  `lcrc_molnet_benchmark_pilot_v2/analysis/source/ogb/ogbg_molhiv` 读取；
  SMILES 只读 `mapping/mol.csv.gz` 的 `smiles` 列（`labels="none"` 等价）。
* 不重新解码 OGB Test 标签文件；`y_true` 只来自已保存预测 NPZ。

### 2.2 固定结构特征：rooted radius-2 WL 支持度

在 Train 分子上构建标签无关的根原子中心 WL 指纹：

* 根原子初始描述符：
  `atomic_number, formal_charge, is_aromatic, total_num_H, total_degree, is_in_ring`。
* 边描述符：`bond_type, is_aromatic, is_conjugated, is_in_ring`。
* 稳定规范化 JSON（键排序、紧凑分隔符）后 SHA-256。
* 连续两轮 WL 颜色更新；每轮把 `(边描述符, 邻居上一轮颜色)` 排序后与
  自身上一轮颜色一起编码。
* 最终颜色即 radius-2 根指纹。
* Train 支持度：该指纹出现在多少个**不同 Train 分子**中（不是原子出现次数）。

Test 每分子特征：

* `n_heavy_atoms`, `n_bonds`, `n_distinct_wl2`
* `fraction_wl2_unseen`（Train 频次 0 的根原子比例）
* `fraction_wl2_low_support`（Train 分子频次 ≤4 的根原子比例）
* `median_wl2_train_support`, `min_wl2_train_support`

边界：这是 rooted-WL 局部支持度代理，不是模型表示的精确等价类，
不证明 WL2 完整刻画化学结构。指纹不使用标签、预测、错误或 Test 分组信息。

### 2.3 固定结构特征：Train 相似度

* RDKit Morgan fingerprint：radius=2，2048 bits，固定本机 RDKit 版本。
* 每个 Test 分子到所有 Train 分子的最大 Tanimoto 相似度；分块计算，
  不建立常驻的完整 Train×Test 矩阵。
* Train/Test canonical SMILES 完全重复计数。
* Bemis–Murcko scaffold 的 Train/Test 交集计数。
* MolHIV 为 scaffold split；若 Test scaffold 全部未见，则报告
  “没有可比较的 seen-scaffold 组”，不人为构造分组。

### 2.4 固定分组（看错误分布之前确定）

按 `fraction_wl2_unseen`：
`0` / `(0,0.25]` / `(0.25,0.50]` / `(0.50,1]`。

按 `max_train_tanimoto`：
`[0,0.30)` / `[0.30,0.50)` / `[0.50,0.70)` / `[0.70,1]`。

按重原子数：
`≤15` / `16–25` / `>25`。

每组报告 n、正例数、负例数、正例比例。若某组正例 <10 或负例 <30，
不报告独立 ROC-AUC，只写 `LOW_SUPPORT`。不因结果修改阈值。

### 2.5 排序错误分解

MolHIV 主指标为 ROC-AUC。对正例 `p`：

`positive_rank_loss(p) = mean_over_negatives[1(s_neg > s_p) + 0.5·1(s_neg = s_p)]`

对负例 `n`，对称地计算正例排在它之上的比例（同样 0.5 计平局）。
校验 `1 − mean_positive(positive_rank_loss) = ROC-AUC`，并与 sklearn/OGB 一致。

每组报告：正例平均 rank loss、负例平均 rank loss、独立 ROC-AUC（样本充分时）、
PR-AUC（附正例比例）、A0/C1 同组配对差、三 seed 方向一致性、C1:s2 是否集中退化。
只描述“相关/描述性集中”，不声称低支持导致性能下降。

### 2.6 统计与不确定性

* 全体与三 seed 的 ROC-AUC / PR-AUC。
* 所有固定分组的 n 与正负样本数。
* A0/C1 配对差。
* 结构支持度与 per-molecule rank loss 的 Spearman 相关（全体及分正负类）。
* 相似度与 rank loss 的 Spearman 相关。
* 同一重原子大小分组内，固定 `fraction_wl2_unseen=0` 与
  `fraction_wl2_unseen∈(0.50,1]` 两组的描述性对照。
* 结构分组与标签比例的关系。
* 需要不确定性时：相同模型配对、正负样本分层行 bootstrap，
  2000 次，seed 20261103；区间条件于已有 checkpoint。
  过小分组标记 `LOW_SUPPORT` / 估计不稳定。

不得把正例、负例或正负样本对当作独立模型运行次数。

---

## 3. P1-B：真实局部编辑响应预测

### 3.1 数据

只使用 `lcrc_local_edit_error_diagnosis_v1/analysis/probe_audit_scores.json`
的 6 个 `units`。每个单元 74 对、26 个 component。

`Δy = d_pred − e_delta`；`Δpred = d_pred`。

核验六个模型/seed 对同一 `pair_id` 还原的 `Δy` 最大绝对差 ≤ `1e-5`。
不重新读取 Audit 标签，不重新配对。

### 3.2 主指标

* `MAE_MODEL = component_equal_MAE(Δpred − Δy)`
* `MAE_ZERO = component_equal_MAE(0 − Δy)`
* `Edit_Gain = MAE_ZERO − MAE_MODEL`（正值表示优于“性质不变”基线）

`component_equal_MAE`：每个 component 总权重 1，component 内各 pair 均分权重。

逐模型、逐 seed 以及三 seed 均值报告。

### 3.3 Component bootstrap

* 重抽单位：26 个 component，有放回。
* 2000 次，seed 20261103。
* A0/C1 三个 seed 共用相同 component 重抽序列。
* 报告条件于当前模型的 CI；训练 seed 波动另列。

### 3.4 辅助指标

1. `Spearman(Δpred, Δy)`（74 对，pair-level）。
2. `Pearson(Δpred, Δy)`。
3. `RMSE(Δpred, Δy)`。
4. `RMSE(0, Δy)`。
5. 方向一致率：只在 `|Δy| ≥ 0.10` raw logD 的配对上计算
   `sign(Δpred) == sign(Δy)`，报告剩余对数。

分层：`same_var_heavy=true/false`；`mapping_unique=true/false`
（由冻结 `core_classes.json` 的 `core_class` 连接提供）。
组内有效 component 数过少时只报告描述性均值和 `LOW_SUPPORT`。
不拟合新的线性回归、类别先验或残差预测器。

### 3.5 与旧 TRACE 并列

同时呈现：

* 旧结论：`TRACE` 不能稳定预测 `e_delta`，对零误差修正基线无可靠增益。
* 新结果：`MODEL` 是否优于 `ZERO_CHANGE`。

两者是不同问题。**`NATIVE`（旧探针中令 `e_delta=0`）不是 `ZERO_CHANGE`**。
若数据/组件数不支持判定，标记 `INCONCLUSIVE`。

---

## 4. P1-C：ESOL 与 NCI1 辅助诊断

### ESOL

只使用既有 6 个 A0/C1 Test 预测。固定重原子数组：
`≤15` / `16–25` / `>25`。报告 n、A0/C1 RMSE/MAE、三 seed 配对差。
另报告误差集中度：全体绝对误差最高的 10% 分子所占总平方误差比例，
以及 A0/C1 高误差集合的重叠。不删除样本重算主成绩，不调整模型。
Train-similarity 分组为可选项，低于必需任务优先级。

### NCI1

复用 `runs/tu_NCI1_f<fold>_s<seed>_<model>/test_predictions.npz`
（`gids`、`logits`、`y_true`）。固定范围 5 folds × 2 seeds × A0/C1。

* 用冻结 fold 定义（5-fold stratified，fold_seed=20261102）只读重建
  outer Train 索引，并用 `split_manifest.json` 的 Train/Test SHA-256 核验。
* 从每个 fold 的 **outer Train 图结构**计算节点数的 33.3%/66.7% 分位数，
  用该 fold 阈值把同 fold 的 outer Test 图分为小/中/大三组。
* 报告每组图数、A0/C1 accuracy、配对错误数量
  （A0 对/C1 错，A0 错/C1 对）、每 fold/seed 差异、较大图是否稳定退化。
* 不重训 TU 模型；不把这些组当成独立 benchmark。

---

## 5. 门禁

正式统计前必须通过：

1. 基准 commit 与源码路径核验。
2. 预测文件存在性与完整性。
3. A0/C1 预测按明确 row-id 或 pair-id 对齐。
4. 与冻结报告已保存主指标一致。
5. 没有调用 CUDA。
6. 没有运行任何 train/eval 脚本。
7. 未读取新的 Test 标签源文件。
8. Local-edit Audit 标签没有被再次读取。
9. 不修改历史 checkpoint 和冻结协议。
10. 所有重抽索引、随机种子固定并可复现。
11. 分组先由标签无关结构构建，后与保存预测/标签连接。
12. 缺失数据不会被默默剔除。
13. 结果可由本目录脚本重建。

运行环境设置 `CUDA_VISIBLE_DEVICES=""`；不使用任何会在导入时初始化 CUDA
或隐式加载 Test 标签的旧实验模块。

## 6. 固定随机种子

* mean-seed 行 bootstrap：`20261103`，2000 次。
* MolHIV 分层行 bootstrap：`20261103`，2000 次。
* local-edit component bootstrap：`20261103`，2000 次。
* 所有 bootstrap 在报告中注明“条件于已有 checkpoint”。
