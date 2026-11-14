# PROTOCOL — LCRC 瓶颈诊断（冻结执行协议）

Track: `tracks/structural-sufficiency/lcrc_bottleneck_diagnosis_public122_s0_v1/`
Round / protocol ID: `lcrc-bottleneck-diagnosis-public122-s0-v1`
本文件在**任何正式训练结果之前**写出；科学常量以本文件 + `code/lb_common.py` +
`configs/diagnosis.yaml` 为准。之后只允许追加“实现映射/事后记录”。

---

## 0. 本轮命题（不默认“表达能力不足”）

区分并**允许同时成立**：

| 命題 | 含义 |
|---|---|
| **H_SEL** | 训练过较好的状态，但 Validation 选择/有限样本波动未能可靠选中 |
| **H_OPT** | 在固定信息与训练数据下，指定的额外优化能产生更好的泛化状态 |
| **H_DATA** | 当前学习器的表现明显受训练样本量/结构覆盖限制 |
| **H_USE** | 冻结同一表示后，仍有可兑现的读出利用余量 |
| **H_REP** | 原输入缺少任务相关信息——**本轮不直接检验**，不得靠排除前三项宣布成立 |

不同干预收益**不能相加**分摊总误差；训练误差低、删块敏感、oracle 最好 Test、单一 probe
失败，都不能单独定位瓶颈。本轮的成功结果是**有直接证据的路线诊断**，不要求新 SOTA。

## 1. 资源与授权（收窄预算）

- `res` 两张 A100、seed 0、官方既有划分、每 epoch 完整评估 Train_fit/Valid/Test；
  允许 Test 曝光并用于开发诊断（报告明确标注）。
- 只经既有 `rr` 流程，`res-gpu0` / `res-gpu1`，每 pool 最多一条本轮训练进程，
  进程内 `cuda:0`；**不用 res-2**，不动其他任务。
- 目标 3–5h 墙钟、硬 6h；GPU 软 4、硬 6 A100 GPU·h（失败/冒烟/评估/提取照计）。
- 新正式训练最多 20 条：**9 条 Lipo 主体 + 4 条 BACE 确认 + 3 条 Lipo 读出探针 +
  至多 1 条重启 + 至多 3 条 BACE 探针**。连续轨迹的 U0/2U0 是同一路径两个观察预算，
  不是两次独立重复。冒烟每类 ≤2 epoch，单独计费、不进结果。
- 不增 seed、不重划 Valid/Test、不加新性质/高阶统计/基团/SSL/预训练/超参网格，
  不重跑 S 家族，不恢复辅助损失，不写论文初稿。预算到即停，未跑标 `NOT_RUN`。

## 2. 必须复用的来源（只读）

前缀 `tracks/structural-sufficiency/`：

- `lcrc_performance_route_8h_public122_s0_v1/ERRATA.md` 与
  `analysis/{learning_curves,corrected_manifest,run_table_corrected,paired_corrected}.json`
  ——只使用修正口径；`code/{lcpr_common,lcpr_data,lcpr_model,lcpr_train,lcpr_dmpnn}.py`
  ——数据、C1、K、chem8、修正后 D-MPNN（含 ERRATA E4 的 `optimizer_base_lr`）、
  RNG 安全评估。
- `lcrc_lipo_model_dissection_v1/{MODEL.md,code/dissect_model.py,code/dissect_common.py}`
  ——C1←FULL 的 state remap 与真实 forward。
- `lcrc_molnet_benchmark_pilot_v2/code/`、`lcrc_cross_task_preparation_v1/code/`
  ——split、缓存、norm、collate。
- `lcrc_pcba_frozen_readout_probe_v1/REPORT.md`——已有 pooled-input 读出 probe 只有弱信号；
  “再接 MLP”不是已支持路线。本轮探针对照是**冻结逐根 h 下的非线性在池化前/后、参数精确匹配**。
- `lcrc_lipo_oracle_task_relevance_public122_s0_v1/analysis/synthesis/METHOD.md`
  ——追溯的计算边界。
- 旧协议、原预测、checkpoint、缓存和 run 状态只读；不审计整个历史仓库、不另开勘误。

## 3. P0：先用已有产物定义诊断基准

重放 5 条 Lipo 曲线（`C1_R1`、`C1_R3`、`C1_CHEM8`、`C1_CONTRIB`、`K_COMBO`）的
Valid 所选状态、Test 曲线最小值、固定窗口中位数、选择位置。右侧只命名
`observed_oracle_gap = R_Test(theta_Valid) - min_E R_Test(theta_e)`，含 Test 选择乐观偏差，
不是可实现收益/总体风险上界/正式成绩，**不据此改成 Test 选 epoch**。旧 `K_COMBO`
best.pt 已丢失不构成重训全项目理由；新 K 轨迹是 `NEW_REPLICATION`，不声称找回旧权重或强求复现 .5573。

## 4. 数据与嵌套训练子集

- 主诊断 Lipo public `random122`：4200，官方 Train 3360 / Valid 420 / Test 420；
  确认 BACE public `scaffold122`：Train 1210 / Valid 151 / Test 152。
- 官方 Valid/Test 逐行不动。仅在官方 Train 内按
  `sha256('lcrc-bottleneck-v1|<dataset>|<global_row_id>')` **升序**构造嵌套子集（tie 用 row id）；
  与标签无关。Lipo fit 25/50/100% = n 840/1680/3360；BACE 50/100% = 605/1210。
  记录 fit 位置与发布 global row id 两套坐标、哈希；已验证严格嵌套。
- **受控尺度**：结构 norm 只用全部官方 Train 的**无标签结构**拟合一次，各 N 共用；
  回归目标 norm 只用**最小 fit 子集（Lipo 840）**标签拟合一次，各 N/模型/阶段共用；
  BACE 仍用原始 0/1，无目标 norm。不用被排除的 Train 标签额外拟合 norm、选择或训练 probe。
- Train 评估名明确为 `train_fit_N`，不把 fit840 写成 Train3360；Valid/Test 始终全量。
- 缓存/norm 写本轮独立路径，不覆盖上游 `train_norm_*` 或旧 S 表；保留共同 norm 与最小子集哈希。

## 5. 共享训练实现：固定 updates、控制选择机会

- update-driven 新 trainer：保留原模型、损失、FP32、clip 5、初始化与 shuffle 规则；
  norm 用第 4 节诊断定义。C1 variants 从同 seed A0 初始 state 按原 remap 构造；
  新模块 private RNG；训练 RNG 重新设 seed 0；同模型同 N 的配方/分支共享批计划；
  跨 N 共享相同可训练初始参数。保留 epoch 内 randperm 与末尾 partial batch。
- 每个 N 按 Umax 生成足够长的批计划（不复用旧 720-epoch permutation 缓存）。
  U0 快照含当前 epoch、当前置换、下一 batch 位置与 generator state。
- 基准更新数与 fit 比例无关：
  C1 `U0 = 360*ceil(N_official_Train/128)`（Lipo 9720 / BACE 3600）；
  D-MPNN `U0 = 100*ceil(N_official_Train/64)`（Lipo 5300 / BACE 1900），
  warmup `= 2*ceil(N_official_Train/64)`（Lipo 106 / BACE 38），其后指数衰减到 U0 的 min_lr。
- C1 在 `0..U0` 固定 U0 horizon 余弦衰减；Lipo 继续到 2U0 时 U0 后固定 min_lr，保留
  Adam/数据流/RNG，不重置。BACE 主体只跑至 U0。
- 每完成一个 fit epoch 对完整 train_fit/Valid/Test 评估；另在固定 update grid 评估：
  `E1 = {round(j*U0/50), j=1..50}`；Lipo `E2 = {U0+round(j*U0/50), j=1..50}`。零步另存。
  主 checkpoint 只在共同 grid 上选，各 N 每阶段 50 个候选；epoch 指标照存。
  每次 eval 保存/恢复 Python/NumPy/Torch CPU/CUDA RNG、model mode，不推进训练 shuffle 流。
- 不把 U0 与 2U0 当两次独立训练；额外预算看 E1/E2 各 50 点与整 100 点。

## 6. 主体实验清单（9 Lipo + 4 BACE）

- Lipo（均至 2U0）：`C1_R1 × fit{25,50,100}`、`DMPNN_D2F × fit{25,50,100}`、
  `fit100` 的 `C1_R3`、`CHEM8_R1`、`K_R1`（K = chem8 + Q8 贡献残差，recipe R1）。
- BACE（至 U0）：`C1_R1`、`DMPNN_D2F` 各 `fit{50,100}`。分类主 ROC-AUC、辅 PR-AUC/BCE，
  不与回归绝对数字混平均。两学习器输入/recipe/更新数不同，只比较各自样本效率与优化响应形状。
- 所有为新轨迹，不要求逐位复现旧成绩；结论不从旧 R1 减新 N25。

## 7. 三条预定 checkpoint 规则

- `R_SEL`（主）：回归 Valid MSE 最小，分类 Valid ROC-AUC 最大；严格改善、同分取较早 update。
  主体只用 grid，step0 不进主候选池。
- `A_SEL`：回归 Valid MAE 最小，分类 Valid BCE 最小；同分较早。
- `W_SEL`：每 50 点 grid 内连续 5 点的主指标均值最小/最大，取窗口最后状态（非权重平均）；
  首次 4 点不构窗，故每阶段 46 个完整窗口。
- 三规则全部预设、看 Test 后不调；分别在 E1、E2、累计 E1+E2 给结果。

保存 grid 每 state 的逐样本输出（原始单位或分类 raw logits）与真实 row ids，覆盖
fit/Valid/Test。给 `observed_oracle_gap` 与好状态持续窗口（不当正式选择规则）。
2000 次 Validation paired bootstrap（主面板 fullN 的 U0/E1：Lipo 5 条 + BACE 2 条，同任务
共用重采样索引；Lipo 用 row，BACE 按正负分层 row）→ 每个 replicate 只重选 Valid 后记录该
state 在原 Test 的指标/update。同面板做逐 Validation row 删除的选择 jackknife，单列发布
row 2087。尾部只作影响度诊断。

## 8. 数据量 × 优化预算（怎样读）

每个模型分别画 N→误差（同 U0/2U0/同 50 候选机会）与固定 N 的 update→train_fit/Valid/Test
轨迹；报每阶段最后 10 个 grid 点的中位数与固定终点（非新 checkpoint）。检查数据效应在
R_SEL/A_SEL/W_SEL 下方向是否保持。仅 DMPNN 随 N 改善而 C1 不改善 → C1 数据利用/归纳偏置
疑点，不等于已定位缺失结构信息。N 与 update 都不清楚 → `INCONCLUSIVE`。各 N 不单独调正则，
所以这是固定学习过程的样本效率，不是调优后的样本复杂度。

## 9. 唯一可选重启（至多 1 条）

仅针对 `Lipo C1_R1 fit100`。若 U0→2U0 低 lr 延长的 Train RMSE 降幅 <5% 且主 Test 结果未改善 1%，
从同一 U0 checkpoint 恢复 model+Adam+数据迭代器+RNG，以原 peak 1e-3 重启一次 cosine 到 min_lr、
额外 U0 步、不 reset Adam；与低 lr 延长配对、同一续接批计划、同样 E2 50 点。失败如实记，
不降 lr 重试/加 warmup 救配置。资源不足不购买，未跑不称为无效。

## 10. 冻结表示的读出定位（3 条 Lipo；条件下再 3 条 BACE）

从 `LIP_C1_R1_F100` 的 U0 阶段 `R_SEL` checkpoint 提取 clean h（dropout 之前），
global42 标准化量、n、row ids 与目标 norm。冻结 `atom_encoder/bond_encoder/local`，cache 每根 h
与 atom_ptr。只用本轮 fit 行标签训练头。原 `f/rho` 从该 checkpoint warm start 且可训练；
encoder 绝不更新；无 dropout/增广；AdamW lr 3e-4、wd 1e-3、batch 128、1800 updates、无 scheduler，
50 个共同 grid + step0；R_SEL 为主；每 fit epoch 与 grid 完整评估。

- `P_WARM`：仅训练原 f/rho，`y = Σ f(h_v) + rho(z_pool)`。
- `P_POST`：加 `psi:64→32→16`、`delta:187→64→1`，输入 `[z_pool(171), psi(mean_v h_v)(16)]`。
- `P_PRE`：同 psi/delta/初始参数，改用 `[z_pool(171), mean_v psi(h_v)(16)]`。
- `z_pool=[g42, mean_h64, mean_h2_64, log1p n]`；新增参数都应为 **14705**；delta 末层全零，
  step0 与 P_WARM 一致（已用测试锁死）。psi/delta private seed 1101。

若任一新增 probe 相对 P_WARM 在 Lipo Valid/Test RMSE 都改善 ≥1%（本轮采购门槛），
在 BACE fit100、U0 的 C1 checkpoint 复做三个 probe；否则不做 BACE probe。
正结果支持“固定 h 仍有可兑现利用余量”；PRE 超过 POST 给非线性池化位置的条件证据。
负结果只排除这些探针/预算，不证明 h 中没有任务信息。

## 11. 记录、数值与保留

- 继承修正后的 D-MPNN schedule 入口，但固定 warmup/update horizon；用无数据 toy 检查实际
  lr 首值、warmup 峰值、U0 末值、继续阶段恒定值。
- norm 夹具：扰动最小 fit 子集之外的 Train 标签/全部 Val/Test，共同 norm 不变；结构 norm
  只消费官方 Train 结构；每 batch 只含当前 fit 位置；global IDs / cache 局部位置 / root 索引分清。
- 训练/评估 RNG 探针、目标只反标准化一次、每阶段 50 候选、连续 prefix hash/optimizer/RNG 一致，
  都是优先测试。metric replay 微小差异记录 warn，只有实质错配才阻塞结论。
- 每轨迹保留：初始/最终 state hash、U0 model+Adam+RNG+batch cursor 快照、2U0 last、
  各规则实际选中权重与对应预测、原始 norm/feature spec/fit 映射、每 epoch 与 grid 曲线、
  grid 逐样本预测。无需存 100 份全权重；流式保存各规则当前最佳即可。
- 最终选中 checkpoint、读出 probe、输入缓存清单至少在本地与 res 各有可恢复副本；
  不删 K 或其他选中权重。manifest 写恢复命令/hash/路径；大文件按 git-ignore。
- GPU 成本按 `rr` job 占用卡的 wall 计（含已占卡的 CPU 等待/同步），卡外 CPU 准备/分析单列。
  远端不依赖临时 RDKit 解析，结构特征本地 CPU 先 materialize。

## 12. 路线判定矩阵（不靠排除法）

所有 gain 正号表示更好：`RMSE_control - RMSE_candidate`，`AUC_candidate - AUC_control`。
主比较做 2000 次 paired 评估样本 bootstrap（固定网络/规则），单列第 7 节选择 bootstrap。

- `SELECTION_SENSITIVITY`：同 stateset 内 oracle 差距大、Validation bootstrap/jackknife 选择有实质波动。
- `SELECTOR_HAS_UTILITY`：某个事先固定替代规则在多模型/任务带来一致实际收益（三规则并列）。
- `OPTIMIZATION_HEADROOM`：固定 N、同 prefix 的额外优化改善留出状态及可实施选择结果；
  只有 Train 变好 → `FIT_ONLY`。
- `SAMPLE_EFFICIENCY_LIMIT`：固定标度/更新/候选机会下更多 N 带来明确收益且预算/selector 敏感兼容。
- `REPRESENTATION_USE_HEADROOM`：冻结 h 的 probe 出现可兑现收益；PRE vs POST 给条件证据。
- `REPRESENTATION_DEFICIT_UNIDENTIFIED`：以上都没解决且 C1 仍落后参照时，只能记“表示/归纳偏置仍可疑”，
  不判定表达力不足；本轮不给 H_REP 支持标签。
- `NO_CLEAR_AVOIDABLE_LIMIT` / `BUDGET_INCONCLUSIVE`：允许。模型共同失败 ≠ 标签噪声。

不得只因为某臂 Train 误差高于另臂就宣告欠训练；不得仅因累计 100 候选最小值低于 50 候选就
宣告优化有效。不得把 oracle gap、数据增益、probe 增益相加解释“总误差的百分之多少”。

## 13. 交付与执行节奏

0–45min：P0 复用、缓存/子集/norm、smoke、toy 与协议冻结；C1/DP 的 fullN 先上两卡。
随后并行六条主体 N 曲线；K/CHEM8/R3 填空；一张卡空闲时做头提取/probe，CPU 同时做选择分析。
BACE 四主体与必要对照优先于可选 restart。最后 45–60min 做 bootstrap、图、报告、双端权重保留。

交付本 track：`PROTOCOL.md`、`configs/diagnosis.yaml`、`code/`+`tests/`、`REPORT.md`（中文，
先答 H_SEL/H_OPT/H_DATA/H_USE）、`analysis/run_manifest.json`、结果表与
`analysis/synthesis/{selection,data_budget,readout}.json`、3 张核心图。入口支持
`prepare/smoke/train/analyze/report` 并能跳过已完成轨迹。不创建 PR、不开始下一轮。

---

## 附录 A — 实现映射与事后记录（训练开始后只追加，不改上方科学常量）

### A1（第 10 节的实现偏差，在探针开跑之前记录）

第 10 节写的是“U0 阶段 `R_SEL` checkpoint”，但流式保留只持久化了**全路径**各规则的
`best_{rule}.pt` 与 `u0_snapshot.pt`，没有单独持久化 E1 作用域内的 `R_SEL` 选中权重
（每阶段 50 候选 × 13 条臂全存全权重不划算）。因此本轮读出探针的源固定为
**`LIP_C1_R1_F100/u0_snapshot.pt`**，即 update = U0 = 9720 的终点状态：

- 这是主线的**训练预算终点**，不含任何按 Valid/Test 挑选 epoch 的步骤；
- 三条探针共用同一份 `h` 缓存（同一 encoder 状态），只比较读出；
- 偏差方向明确：它不比“E1 内 R_SEL 选中状态”更乐观，只是不再带选择步骤；
- 报告里以 `source: u0_snapshot` 标注，不与其它选自 Valid 的状态混称。

### A2 判读口径

`grid_predictions.npz` 的回归预测是**标准化空间**（`lcpr_train.split_metrics` 内部才反标准化），
分析侧用 `train.json.data.target_norm` 反标准化一次；BACE 存 raw logits。曲线的 `curve.csv`
本来就是 raw 单位，二者已交叉核对一致。

### A3 执行 regime

全部正式臂经 `rr` 在 `res`（a100-2，A100-SXM4-40GB ×2，driver 550.163.01，
torch 2.5.1+cu124）上以 `res-gpu0`/`res-gpu1` 两个 pool、每 pool 一条进程执行；
`cuda:0` 进程内可见。逐臂 commit 见 `analysis/run_manifest.json`。
