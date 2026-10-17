# 预注册协议 — cscl-bond-relation-attribution-v1

- Track：`tracks/structure-contribution`
- 日期：2026-10-18（在任何本轮标签读取、任何本轮 GPU 训练之前冻结）
- 协议 ID：`cscl-bond-relation-attribution-v1`
- 结果目录：`tracks/structure-contribution/results/bond_relation_attribution_v1/`
- split tag：`E`（salt `cscl-bond-relation-attribution-v1`）
- 前置轮次（只读、不覆盖）：`results/explicit_composition_signal_audit_v1/`（结论
  `INSUFFICIENT_OR_CONFOUNDED`）、`results/bond_rw_local_vs_multiscale_v1/`、
  `results/bond_random_walk_organization_v1/`、`results/bjoc_architecture_breakthrough_v1/`、
  `results/bjoc_predictive_research_redirection_v1/`、`results/bjoc_r3_frozen_terminal_test_v1/`

---

## 0. 问题（只此一个，不再扩展）

**在相同图输入、相同参数量、相同监督样本、相同 SSL 初始化与相同训练配方下，
真实的 Bond–Bond 一跳连接关系（R）能否比“行和完全匹配但不保留真实连接归属”的
Null（D）带来可复现的 held-out 泛化收益；该收益是否也超过历史 Uniform 核（U）？
R 与同轮强参照 P3 的差距是否小到值得继续研究？**

前置审计已证明：ZINC train 上不存在足够的“同 bag 不同连接”自然对照，且标签近似由
penalized-logP 公式决定。**本轮不要求 beyond-formula residual，也不要求 patch-bag
collision。** 本轮评价对象是模型的学习与泛化，不是结构组合对化学性质的因果效应。

明确不声称：R−D 不是纯 topology-causal effect（见 §3.3 与 §7）。

## 1. 数据与划分（split E）

- 数据：官方 `ZINC(subset=True, split="train")`，10,000 行。**official valid 与
  official test 在本轮任何代码路径中都不加载。**
- **SMILES ↔ PyG constitution 不匹配处理（前置冻结）**：前轮审计发现 4 行
  `[941, 2210, 6071, 9766]` 的提交 SMILES 与官方 PyG 图不一致。本轮在建模前重新
  跑一遍 10,000 行配对核对，必须精确复现该 4 行清单，然后**一致地排除**这 4 行
  （所有对照同一步骤、同一集合）。排除后可用 9,996 行。
- **分子身份定义**：使用**官方 PyG 图重建的 canonical constitution SMILES**
  （`structure_context_support_v1._pyg_rebuilt_smiles`，去立体化学），而不是可能错误的
  提交 SMILES 字符串。并要求：在 9,996 行上，“图重建身份”与“提交 SMILES 身份”诱导的
  划分一一对应（0 个 key→多个 smiles，0 个 smiles→多个 key）；否则停止。
- **划分**：按身份 key 分组，组内按行号；组顺序 = `sha256(salt|key)` 字典序；贪心填充
  fit 至 ≥7,996 行、monitor 至 ≥1,000 行、其余为 locked。要求同组不跨区；
  目标实际落在 7,996 / 1,000 / 1,000。
  - 与历史 10k 配方的区别：历史 split A/C/D 使用 10,000 行（含 4 个坏 SMILES 身份）
    且 fit=8,000；本轮为 9,996 行、fit=7,996、monitor=1,000、locked=1,000。
    因此 **本轮绝对 MAE 不能与历史 split 的绝对 MAE 直接比较**；只做同轮配对对比，
    历史数字只作描述性参照。
- 记录：`split_E.json`（行号 + group_id + 指纹，提交仓库）、`split_audit.json`
  （配对核对、身份一致性、分组统计、salt、行哈希）。
- 标签纪律：monitor / locked 的 `.y` 在预处理、SSL、训练、预测阶段通过 `LabelGuard`
  不可达；只在全部预测冻结后读取一次。fit 的 raw `y` 仅用于监督。

## 2. 对照臂（每 seed 四个，全部预定义）

同一 seed 内：相同 fit 行、相同 raw-y、相同 fit-only 全局标准化、相同 batch 顺序
（`seed + 101`）、相同 R3 配方（Adam 1e-3 → 1e-5 cosine，wd 1e-5，clip 5，batch 128，
360 epochs / 22,680 updates，raw-y L1，末 5 轮参数平均 soup）。**不做任何超参搜索、
不做早停、不做按 held-out 选择。**

| 臂 | 关系核 | 参数量 | 初始化 | SSL 权重 |
|---|---|---|---|---|
| **R** | `K^R = sym(P)`，`P = (I + D^{-1} A_L)/2`（物理键线图 lazy 一跳；零度行 one-hot） | 349,985 | `bwo.build_model(seed)` | atom+bond encoder（本 seed 的 Fit-only masked SSL） |
| **D** | `s_i = Σ_j K^R_ij`，`K^D_ij = s_i s_j / Σ_k s_k` | 349,985（同一模型类、同一 init 哈希） | 同上（逐位相同） | 同上（逐位相同） |
| **U** | `K^U_ef = 1/N_bond`（历史定义，`bwo.uniform_triplets`） | 349,985 | 同上 | 同上 |
| **P3** | BJOC-Null（含 learned Junction encoder；不同架构） | 413,505 | `bp.build_bjoc_null(seed)`（P3 自身方法） | 全部三组 encoder（同一 SSL artifact） |

- R/D/U 复用**未修改**的 `bond_random_walk_organization_v1.BondRWComposer` /
  `build_model` / `train_supervised_rw` / `predict_model`；R1 核复用
  `bond_rw_local_vs_multiscale_v1.k1_matrix`。本轮只新增 D 的核。
- **三臂的 pair 列表布局完全相同**：`unordered_triplets` 对对称矩阵枚举全部 `e≤f`
  （含权重 0 项），因此 R/D/U 的 pair-MLP 计算量、参加数、batch 组成相同；差异只在
  **权重值**（支撑集与权重分布），不在张量稀疏存储或 FLOPs。
- SSL：每个 seed 在**本轮 fit 行**上重新训练一次 masked atom/bond SSL
  （`bp.pretrain_masked`，`mode="atom_bond"`，7,560 updates，无标签）。R/D/U 只加载
  atom+bond 组；P3 按其原方法加载 atom+bond+junction 全部三组。**禁止**复用历史
  split 的 SSL（会把本轮 monitor/locked 分子泄漏进预训练）。

## 3. Phase 0（CPU，标签盲）——必须全部通过才允许 GPU

### 3.1 数学定义与行和
对每个分子（fit 前 500 + monitor 前 200 + 若干合成图）：

1. `K^R` 对称、行和 = `s`、总权重 = `Σs`；`P` 每行和为 1。
2. `K^D` 对称、行和逐行等于 `s`（误差 ≤1e-10）、总权重等于 `Σs`（≤1e-10）。
3. `K^D = s sᵀ / Σs` 是**秩 1**（解析性质，数值秩须为 1，除零外）。
4. `K^D ≠ K^R`（对 `N_bond ≥ 2` 的分子，`max|K^D − K^R| > 1e-6`），并报告
   - R 的非零项比例（含对角线）；D 的密度（应 = 1）；
   - R 支撑集上的权重占比（D 在 R 的零支撑上的质量比例）；
   - 逐分子 `corr(vec(K^R), vec(K^D))` 分布；
   - `mean |K^D − K^U|`（D 是否退化为 U）；
   - 单键分子（`N_bond=1`）上 `K^D = K^R = K^U`，单独报告计数并说明不可避免。
5. `K^U` 行和 = 1、总权重 = `N_bond`、密度 = 1、秩 1。
6. 报告 `corr(line-graph degree, rowsum(K^R))`、`rowsum` CV、`TV(K^R, uniform)`。

### 3.2 参数量、初始化与输入
1. R/D/U 参数量 = 349,985；同 seed 的 `state_dict_hash` 三者完全相同；
   R/D/U 的 `encoder_used_sha256`（atom+bond）逐 seed 相同；P3 = 413,505 且其
   atom/bond encoder 与本 seed SSL 的 atom/bond 组一致。
2. 同 seed 下 R/D/U/P3 的 batch-order 哈希 = `bp.batch_order_hash(seed, n, 128, 360)`。
3. 全局标准化参数来自 fit-only（`bp.fit_standardiser`），四臂共用。
4. 更新数 = 22,680（`ceil(7996/128)=63`，63×360）。

### 3.3 真实关系被替换 / D 非退化
1. R/D/U 的 `(pair_src, pair_dst)` 布局逐分子相同，仅 `pair_w` 不同。
2. 在共享初始化下，R/D/U 的前向输出不全等（`max|Δpred| > 1e-4`）。
3. D 与 R 的关系矩阵不相同（见 3.1.4）；D 与 U 不相同（除单键分子）。

### 3.4 梯度、前向、重编号不变性
对 R/D/U 各核：
1. 1 个 batch 反传后 `pair_mlp`、atom encoder、bond encoder、composer、reader 的
   梯度范数均有限且非零。
2. 原子重编号：前向输出不变（≤1e-6）。
3. bond 槽位置换 + 核同步置换：前向输出不变（≤1e-6）。
4. batch 组成：单分子批量预测 == 混合批量中的对应预测（≤1e-6）。
5. 保存/加载 checkpoint 后前向一致。
6. 2 次相同参数的 1-epoch micro-smoke 训练 soup 一致：**逐位一致，或
   `max|Δsoup| ≤ 1e-6`（远端 CPU float32 归约 run-to-run 非确定，历史
   轮次已在 `bjoc_task_adaptive_ssl_v1` 记录为 ~1e-7；本机逐位一致）**。
   同时记录是否逐位一致。见 §9 amendment-1。

7. SSL 编码器哈希的**前缀约定**必须与历史轮次一致（`atom:`/`bond:`/
   `junc:`），并断言本脚本重算的整编码器哈希等于 `pretrain_masked` 返回的
   `encoder_final_sha256`（P3 的 `encoder_used_sha256` 与之相等）。见 §9
   amendment-2。

原子循环 step 1 与历史 CPU 训练一致性：以 32 个合成/真实图训练 1 epoch，
`soup` 与直接调用 `bwo.train_supervised_rw` 一致（同一函数，不重实现）。

### 3.5 历史实现复用（不因重构改变计算含义）
本轮**不修改**以下文件的字节内容，并在 `reuse_audit.json` 中比对其 SHA-256 与历史
`protocol_freeze.json` 记录值：

- `code/bond_random_walk_organization_v1.py` = `efda1dd7…`
- `code/bond_rw_local_vs_multiscale_v1.py` = `6638b238…`
- `code/structural_relation_composer_v1.py` = `1c83a97e…`
- `code/structural_relation_objects_v1.py` = `8f4ba44d…`
- `code/bjoc_perfmech_long_v1.py` = `cb9796cb…`
- `code/bjoc_arch_breakthrough_v1.py` = `31294de0…`
- `code/bjoc_masked_pretraining_v1.py` = `f27dbfe3…`

并断言：本轮 R 的 triplet 构造函数与
`bond_rw_local_vs_multiscale_v1.k1_triplets_from_objects` 在同一分子上逐位相等；
U 的 triplet 来自未改动的 `bwo.uniform_triplets`。若任一哈希不符，停止并解释。

### 3.6 标签隔离
`LabelGuard` 正控制通过；预处理/SSL/训练/预测期间 monitor+locked 的 `.y`
访问尝试数 = 0；`objects` 的对齐检查只用 fit 行（与 `encoded_train.pt` 的 y
最大差 ≤1e-9）。

**任一关键检查失败 → 修复工程缺陷后重测；若对照本身无法构造 → 停止并解释，
不得替换为未经冻结的新干预。**

## 4. GPU 执行与预算

- 仅使用 `rr`，host `res`，pool `res-gpu0`（seed 0）与 `res-gpu1`（seed 1）并行。
- 每 seed 顺序执行：SSL → R → D → U → P3 → 预测（monitor + locked）。
- 预算上限 **4.0 GPU-hours**；估算：SSL 2×~1,660 s + R/D/U 6×~455 s +
  P3 2×~560 s ≈ 7,300 s ≈ 2.03 GPU-h。若预计或实际超过 4.0 h，停止扩张，
  不增加 seed、epoch 或候选模型。
- 先跑一次 1-epoch / 20-update 的 GPU smoke（只用 fit 标签），通过后再冻结协议并正式运行。

## 5. 评分（全部预测冻结之后，单次读取）

- 先冻结：soup 哈希、monitor+locked 预测文件、预测 manifest、评分代码、协议/配置哈希。
- 一次评分步骤同时读取 monitor（n=1,000）与 locked（n=1,000）的 y，各一次；
  写入 `holdout_read.lock`（y 哈希 + manifest 哈希 + 时间）。
- **主终点：Locked MAE（seed-mean，2 seeds）**；**复现终点：Monitor MAE（同定义）**。
  两个区域都完整报告，不据结果改协议。
- 指标：MAE（主）、RMSE、median |err|、p90 |err|、ordinary（y > −3）/ tail（y ≤ −3）
  Δ、top-10-removed Δ、去掉单条最大误差行的 Δ、`frac_molecules_better`；
  每 seed 的 Δ 与 seed-mean Δ；配对分子 bootstrap 5,000（seed 0，按分子重采样）。
  另报告每 seed 的 bootstrap CI，以及 2-seed 的符号一致性（2 seed 无法估计 seed 方差，
  明确说明）。
- 预声明对比：
  - **P1 = R − D**（主）
  - **P2 = R − U**（历史探索性正结果复现）
  - **P3c = D − U**（历史收益中有多少可由行和/权重分布解释）
  - **P4 = R − P3**（同轮强参照；P3 架构/参数量不同，仅作竞争力参照，不作因果）
  - 参考：D − P3、U − P3（描述性）
- Δ = MAE(A) − MAE(B)；负值 = A 更好。

### 5.1 冻结的判定尺度与决策词表

- 有意义效应量 `MEANINGFUL_DELTA = 0.004`（与历史 promotion gate 同尺度；约为
  held-out MAE 的 4%）。
- 单个对比在某一区域：
  - `GAIN`：seed-mean Δ ≤ −0.004，**且两 seed 的 Δ 均 < 0**，且配对 bootstrap CI 上界 < 0；
  - `LOSS`：seed-mean Δ ≥ +0.004，**且两 seed 的 Δ 均 > 0**，且 CI 下界 > 0；
  - 其他（含 CI 跨 0、seed 符号冲突、|Δ| < 0.004）一律 `NEUTRAL`。
- `replicated`：该对比在 Monitor 上 seed-mean Δ < 0 且两 seed 均 < 0（不设幅度门槛）。

### 5.2 总体判定（Case A / B / C）

按顺序取第一个满足者；同时报告所有触发的子条件。

1. **CASE_A_RELATION_CANDIDATE**：
   P1 在 Locked 为 `GAIN` 且 Monitor `replicated`；
   P2 在 Locked 为 `GAIN` 且 Monitor `replicated`；
   且 `R − P3` 在 Locked 的 seed-mean ≤ +0.010。
2. **CASE_B_NO_SUPPORT**（任一触发即 B，并记录具体子标签）：
   - `B1_ROWSUNM_ABSORBS`：P1 在 **两个区域** 都是 `NEUTRAL` 且 `|seed-mean Δ| < 0.004`；
   - `B2_NO_GAIN_OVER_UNIFORM`：P2 在 Locked 的 seed-mean Δ ≥ 0（两区域同向时记录）；
   - `B3_FAR_BELOW_STRONG_REFERENCE`：`R − P3` 在 Locked 的 seed-mean > +0.020；
   - `B4_D_BETTER_THAN_R`：P1 在 Locked 为 `LOSS`。
3. **CASE_C_INCONCLUSIVE**：以上都不满足（例如 Locked 与 Monitor 方向冲突、
   seed 符号冲突、点估计有收益但未达 `GAIN`、或 R 略弱于 P3 但未到 0.020）。

该词表是**解释框架**，不是替代数据判断的机械阈值；但一旦冻结，报告不得事后修改。

### 5.3 停止规则

无论结果如何，本轮结束后停止：不追加 seed、不增加架构、不做超参搜索、不读
official valid/test、不据结果重新定义 split。只有在 **CASE_A** 下才允许（但不自动执行）
提出一个**新的、独立的**预注册，专门区分“权重支撑/局部稀疏性”与“真实一跳归属”。

## 6. 机制诊断（冻结模型，预测先冻结）

预声明、不新增训练：

1. **跨核前向交换**：对每个已训练模型 R/D/U，在 held-out 分子上分别用
   R/D/U 三种核前向（共 9 个预测文件），全部在读取标签前冻结。用于区分
   “核输入效应”与“训练轨迹效应”。
2. **权重分布统计**：R/D/U 的权重分位数、非零比例、R 零支撑上 D 的质量占比、
   D 相对 U 的偏离、`corr(degree, rowsum)`、秩 1 解析、TV 到 uniform。
3. **梯度活性**：训练后对 1 个 fit batch 统计各子模块梯度范数。
4. **误差区域**：ordinary / tail / top-10 / 单条最大误差；按 seed、按区域。
5. **逐分子 Δ 相关**：P1 与 P2 的逐分子 Δ 相关，判断 R−D 与 R−U 是否同一批分子。

不做贡献分解、不做化学因果归因。

## 7. 必须写进报告的混杂说明

1. **degree / row-sum**：`rowsum(sym(P))` 与线图度相关（历史值 corr≈0.84–0.89），
   Pair MLP 带偏置 ⇒ R 与 D 都含度/中心性相关的行和偏置；D 逐行精确匹配 R 的行和，
   因此 **R−D 不再含“行和差异”这一项**，但 **R−U / D−U 仍含**。
2. **局部稀疏性**：R 的权重支撑是真实一跳邻域 + 对角线（约 78% 的枚举 pair 权重为 0），
   D 在同样枚举下权重全为正（秩 1 稠密）。所以 **R−D 仍混合“真实连接归属”与
   “权重支撑/集中度（局部性）”**；即使 R 胜出，也不能单独宣称“真实连接归属”有效。
   R/D/U 的 pair 枚举与 FLOPs 相同，差异只在权重值，这一点必须写清（避免把它误读为
   “R 更稀疏所以更省算力”）。
3. **P3**：413,505 参数、不同架构、不同输入对象（含 Junction），只作同轮竞争力参照；
   历史 exposed-test 0.0698 绝不可与本轮内部 held-out 数值排名或比较。
4. **单/双 seed**：2 seeds 只能检查符号一致性，不能估计 seed 方差；bootstrap 只覆盖
   分子抽样不确定性。
5. **划分来源**：split E 是 official train 的内部重划分；这些分子在历史实验中已出现，
   不是独立外部测试集。绝对 MAE 与 official split 数值不可比。

## 8. 交付物

- `PROTOCOL.md`（本文件）、`configs/bond_relation_attribution_v1.yaml`
- `code/bond_relation_attribution_v1.py`、`code/run_bond_relation_attribution_v1.py`
- `tests/test_bond_relation_attribution_v1.py`
- `split_E.json`、`split_audit.json`、`input_audit.json`、`reuse_audit.json`
- `kernel_audit.json`、`cpu_gates.json`、`protocol_freeze.json`
- `train_{R,D,U,P3}_E_s{0,1}.json`、`predictions/*.npz`、`prediction_manifest_E.json`
- `holdout_read.lock`、`holdout_score_E.json`、`per_molecule_errors_{locked,monitor}.csv`
- `post_hoc_mechanism.json`、`gpu_budget_ledger.json`、`decision.json`
- `REPORT.md`（中文，含 R/D/U/P3 同轮主表、混杂说明、最终判断）

## 9. 预执行修正记录（均在任何本轮标签读取与 GPU 训练之前）

- **amendment-1 (2026-10-18, pre-GPU)**：CPU gate §3.4.6 的 micro-smoke
  确定性检查由“soup 哈希逐位一致”改为“逐位一致或 `max|Δsoup| ≤ 1e-6`”，
  并记录 `soup_identical_bitwise`。触发原因：`bra-cpu` 在 `res` 上运行同一份
  冻结代码时，D 臂两次 1-epoch micro-smoke 的 soup 出现远端 CPU float32 归约
  的非确定性（本机为 0.0）；此前轮次已记录同一现象。阈值/其它 gate 未变，
  无标签参与，无训练轨迹产生。
- **amendment-2 (2026-10-18, pre-GPU, post-smoke)**：smoke 门禁发现
  `full_encoder_group_sha256` 用了 `junction:` 前缀，而历史 P3 约定为
  `junc:`（`bjoc_perfmech_long_v1.encoder_group_hash`）；修正前缀，并在 SSL
  结束时断言 `full_encoder_group_sha256 == encoder_final_sha256`，在
  `freeze-predictions` 中断言 P3 的 `encoder_used_sha256` 等于该值。该 bug
  会使“P3 与 R/D/U 共用同一 SSL”的匹配审计在冻结预测阶段误报失败，不影响
  任何已产生结果；仍在任何正式 GPU 训练与标签读取之前，故重新冻结
  （freeze 重新生成，旧 freeze 仅用于被弃用的 smoke）。
- **amendment-3 (2026-10-18, pre-GPU)**：两个 seed 在两块 GPU 上并行，共享
  `prediction_manifest_E.json` 与 `budget_ledger.json`；为这两个
  read-modify-write 加上 `fcntl.flock` 文件锁（`*.lock` 文件随证据一并哈希），
  防止并行时丢失预测记录或预算条目。无协议语义变化；重新冻结。
