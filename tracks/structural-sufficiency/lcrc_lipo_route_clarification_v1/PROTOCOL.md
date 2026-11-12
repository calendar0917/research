# PROTOCOL — LCRC 局部编辑路线判定（冻结执行协议）

Track: `tracks/structural-sufficiency/lcrc_lipo_route_clarification_v1/`
Round / protocol ID: `lcrc-lipo-route-clarification-v1`（v2，2026-11-12，Asia/Shanghai）
设计锚点：`ecb28cf1789c4b7369ff1041ef6d2defe11865a7`
本文件在**任何正式训练结果之前**写出。之后只允许追加"实现映射"与"事后记录"两节；
所有科学常量以本文件 + `code/lr_common.py` + `configs/frozen_protocol.json` 为准。

---

## 0 研究问题

真实编辑对（A→B，一次小片段替换）：

```
d_pred = pred(B) - pred(A)        d_y = y(B) - y(A)        e_delta = d_pred - d_y
```

* **Q1** 真实编辑差分监督（R）能否在**同输入同容量**下同时改善分子预测与真实编辑响应？
  收益是否超过样本加权（W）与一般差分训练（N）？
* **Q2** 额外容量（S）、固定上下文（O）、动态邻居组合（D）各提供多少增量？
* **Q3** 相同差分目标是否也改善普通指纹模型（F）？LCRC 相对 F/G/KRR/RF 的位置？
* **Q4** 结果是否跨组件划分（S1/S2/SG）、seed、scaffold 分布稳定？Lipo 的独立编辑支持是否够？

所有机制指标都用**整个函数的有限差分**；即使成功也只是"本模型预测响应与机制证据"，
不是化学因果效应，也不是原子贡献真值。

## 1 数据与标签边界

| 项 | 值 |
|---|---|
| 数据 | ChemBench `Lipop`（logD pH7.4），CSV SHA256 `34280f0a…cd758` |
| 划分文件 | `Lipop.random122.ind.pkl`，SHA256 `bf7acb17…4a7d` |
| 母集 | 仅 random122 **Train 3360** 行，行序列 SHA256 `4e8d42a9…af96` |
| 结构来源 | `raw_attributes_v1.json.gz`，SHA256 `1127607d…51be` |
| 编辑对 | 上游冻结 509 对（`pairs.csv` SHA256 `98db43ec…e356`），不重定义 |

* 训练/分析接口**不读** public Test420 标签；全 Train 阶段只用 Train3360 训练、
  原 public Validation420 选 epoch。
* CSV 字节会被读取（同一文件同时是 SMILES 与 exp 的来源），但 **exp 只对当前许可
  行做数值转换**；每次访问写 `analysis/label_access_ledger.json`，记录
  `csv_label_bytes_read=true` 与 `labels_numerically_converted_only_for_permitted_rows=true`。
* 标签权限（代码强制，`lr_data.labels`）：`train→fit`、`select→fit,select`、
  `diagnose→fit,select,diagnose`、`audit_score→全部四区`；Audit 只有在
  `analysis/AUDIT_FREEZE.json` 存在后才可读，且只读一次。
* 结构/目标标准化**只用各自 Fit**（全 Train 阶段只用 Train3360）。
* 本轮不是盲测：旧 Audit 与全数据已有历史曝光，报告性质是
  `WITHIN_DATASET_PROSPECTIVE_AUDIT`，且是 **pre-registered within-dataset 开发复核**。

## 2 组件划分 S0 / S1 / S2 / SG / FT

S0 完全复用上游 `component_split.json`（Fit2016/Select336/Diagnose504/Audit504，
pair 301/50/84/74，配对组件 106/17/26/26）。

S1/S2 在同一 Train3360 上按冻结 salt 重新分组，salt：
`lipo-route-v1-split-20261112-1` / `-2`。

* 不可拆分单位 = `分子—配对`、`分子—共用匹配核心`、`分子—相同 canonical 分子`
  的并查集连通分量；
* 组件稳定 ID = 内部 global row_id 升序以 `,` 连接；
* 配对组件按 `(size 降序, sha256(salt+"|"+稳定ID) 升序)` 排序后，按**相对配额缺口**
  `(target-filled)/target` 最大者分配四区，tie `Fit>Select>Diagnose>Audit`；
  无配对 singleton 填足 2016/336/504/504；
* 不尝试更多 salt。

SG（scaffold 压力测试，只用 Train3360）：

1. 计算不含手性的 canonical Bemis–Murcko scaffold；空 scaffold 用 canonical SMILES
   回退（不把所有无环分子并成一个大组）；
2. **SG_JOINT_GROUPS**：把旧"分子—匹配核心"组件与"相同 scaffold"关系并查集合并，
   按 `(size 降序, sha256("lipo-route-v1-SG|"+组稳定ID))` 排序，60/10/15/15 分配，
   组不拆，允许人数偏离；要求 Fit≥1680、Select≥200、Diagnose≥300、Audit≥300；
3. 不满足则 **SG_SCAFFOLD_ONLY**（相同算法、仅用 scaffold 分组；相同 canonical 分子
   仍同组；跨区 MMP 边全部排除），仍不满足则 **BLOCKED_SPLIT_FEASIBILITY**。

各区都检查组件/核心/canonical 分子/配对端点交集；跨重复划分的角色重叠是设计的一
部分，不能当三个独立数据集。Diagnose/Audit 各 ≥40 对且 ≥15 个配对组件才用于编辑
确认，否则 LOW_SUPPORT（仍报分子指标，不换 split）；Fit <30 个配对组件则标监督
研究支持不足。

## 3 模型（严格定义，全部在正式训练前实现）

`T` = 真实无向物理键图的 row-normalised 邻居均值，无 self-loop，孤立点零向量，
不跨分子。残差块 t=0,1：

```
r_t   = Wout_t( SiLU( Win_t( concat( LN_root_t(h_t), LN_context_t(c_t) ) ) ) )
h_t+1 = h_t + r_t
```

`Win` 128→128（bias）、`Wout` 128→64（bias，weight/bias **全零初始化**）、
两份 LN 带 affine；块内无 dropout，保留 h0 原一次 dropout。每块 25024 参数。

| 模型 | 定义 | 参数 |
|---|---|---|
| **B** | 上游 C1（`NO_BOND_MEAN`）原样；`sum_i f(h_i) + rho(g42, mean h, mean h², log1p n)` | 228 674 |
| **S** | 静态容量：`c_0=h_0`, `c_1=h_1`（新增模块不读邻居状态） | 278 722 |
| **O** | 固定上下文：`c_0=T h_0`, `c_1=T(T h_0)`（只由 h0 构造，不 detach） | 278 722 |
| **D** | 动态组合：`c_0=T h_0`, `c_1=T h_1`（第二块真实读取更新状态） | 278 722 |
| **F** | Morgan r2 2048 bits（`useChirality=False, useBondTypes=True, useFeatures=False`）+ g42 → 2090→96→64→1，dropout 0.10 | 207 009 |
| **G** | GINE 风格参照（显式实现，无库默认 self-loop/BN/dropout） | 251 781 |

S/O/D 与同 seed 的 C1 共享**每一个**子状态（同一 FULL→C1 映射，逐张量相同），
块零初始化 ⇒ step0 精确还原 B。`T²` 是两步均值 walk（含回返与路径重数），**不得**
换成 unique radius-2/BFS 桶。O/D 新增最大感受野都是 h0 的两跳；D 优于 O 只说明
**这一版**动态组合优于**这一版**固定上下文。

G：`edge_t=Linear13→128`，`s_i=(1+eps_t)u_i+Σ_j ReLU(u_j+edge_t(e_ij))`，
`u_next=LN(u_i+MLP128→256→128(s_i))`，三层后 `Linear128→64`+Dropout0.10；
读出沿 C1 `f`/`rho`，同 seed 拷贝对应未训练 C1 初值。

## 4 训练配方与四种损失

AdamW，LR 3e-4→1e-5 per-step cosine，betas(0.9,0.999)，eps 1e-8，wd 1e-3，
batch 128，clip 5，FP32/TF32 关闭；Fit2016 每 epoch 16 步，**360 epoch = 5760
updates**；每 epoch 全量 Select，严格最小 raw RMSE，tie 取早。无 early stop/warmup/
EMA/soup/ensemble/SSL。seed ∈ {0,1,2}。保存 best/last + optimizer/scheduler/RNG/sampler。

每优化步额外取 **16 个真实 Fit pair**（先均匀选配对组件，再组件内均匀选 pair，可重复）；
普通 batch 与 pair stream **提前冻结**，同 split/seed 所有条件共享（含 A 的零系数项）。
辅助 endpoint 前向：eval 模式但保留 autograd，去重 endpoint 后再映射 pair。

```
A: Lmol
R: Lmol + λ_R * mean_true_pair((e_j - e_i)²)
W: Lmol + λ_W * mean_true_pair(e_i² + e_j²)
N: Lmol + λ_N * mean_random_pair((e_j - e_i)²)
```

* W 与 R 使用相同 endpoint 出现次数，去掉残差相互作用：
  `R_aux = W_aux - 2·mean(e_i·e_j)`（数值核验写入 gates）。
* N：同一 pair batch 的 endpoint occurrences 随机重配，保持每分子出/入次数，
  禁 self-pair，重新计算 `z_j - z_i`；预生成固定 plan，尽量 ≥80% 无序边不属于真实
  MMP 图；最多 128 次重配尝试，仍失败按固定 RNG 重抽该步真实 pair batch（R/W/N 共用）。
  实测：s0 平均非 MMP 比例 0.934，shortfall 步数 0，无需 redraw。
* 所有 pair 标签来自原标签，不新增实验信息，不给同类替换强制恒定作用。

**一次剂量校准**（正式结果前，只做一次）：S0 Fit、seed0 原始初始化、32 组固定普通
batch + pair batch，不更新优化器：

```
gA = 普通损失参数梯度 L2；gX = 单独原始辅助梯度 L2
λ_X = clip(0.2·median(gA)/max(median(gX),1e-12), 0.001, 10)
```

各架构独立记录 R/W/N，固定数值复用所有 seed、S1/S2、SG 与全 Train，**不按效果再调**。
训练记录 step 1/100/1000/last 的普通/辅助梯度、pair 训练误差与 loss。

## 5 正确性 gates（CPU）+ 两张 GPU smoke

`code/lr_gates.py`（结果 `analysis/gates_cpu.json`）覆盖 §8 的 13 项：数据/hash/行号/
norm/pair 隔离；B 与旧 C1 数值等价（CPU ≤1e-6）；B-A 新训练器与旧训练器 ≥30 次真实
更新逐步等价；S/O/D 参数与张量一致且 step0 还原 B；T/T² 对独立 loop 相等；抽象 h0
扰动下 S 不交换 / O 只读 h0 / D 第二层读 h1 / 新增范围 ≤2 跳且不跨图；新分支梯度
无死参数；原子重编号与拆合 batch 不改预测；pair 符号/常数偏移/标准化换算；R/W/N
有梯度且辅助前向不消耗 dropout RNG、N 逐行次数等于 R；F/G 维度与 G 小图 loop；
checkpoint 内容；raw 指标与独立 NumPy 一致且旧文件未改。两张 GPU 各做端到端 smoke
（含 Select、辅助前向、保存），准入投影 ×1.25。

## 6 S0 主矩阵（33）与预定选择

B/S/O/D × {A,R} × 3 seed = 24；F × {A,R} × 3 = 6；G-A × 3 = 3。

购买 W/N 的判据（**只用 S0 Diagnose，逐 seed 平均**），对 m∈{B,S,O,D,F}：

```
g_m  = mean(RMSE(m-A) - RMSE(m-R))
ge_m = mean(PairMAE(m-A) - PairMAE(m-R))
de_m = max(0.020, 0.05 * mean PairMAE(m-A))
门槛: g_m >= 0.010 且 ge_m >= de_m 且 >=2/3 seed 两指标均正向 且 配对支持合格
```

多个过门取 `g_m/0.010 + ge_m/de_m` 最大者为 `C_loss`（完全平局 B>S>O>F>D），
只为其追加 W、N 各三 seed（6 次）；未过门 `C_loss=NONE`，不买控制臂，
转入点损失计算/样本效率路线。不能因 W/N 结果不好改 λ 或换 winner。

## 7 S1/S2 复核矩阵（36 + 30 + 12）

无论 loss 如何：B/S/O/D/F/G × A × 3 seed × 两 split = 36。
若 `C_loss` 存在：B/S/O/D/F × R × 3 × 两 split = 30，`C_loss`-W/N × 3 × 两 split = 12。
完整上界 33+6+36+30+12 = 117（核心组件上界，不是预算保证）。
复核优先级：B/S/O/D 的 A → C_loss 的 A/R/W/N 完整包 → F-A/R 通用性 → G-A；
其他架构的 R 交互列后购入，缺项明确标记。

## 8 SG 压力测试与全 Train

**部署候选冻结**（S1/S2 训练后、任何 Audit 用于决策之前，只用其 Diagnose）：
仅 B/S/O 的 A 或完整复核 R 可为候选。相对 B-A 要求：平均 Diagnose RMSE 改善 ≥0.010、
6 个 split-seed 单元 ≥4 个正向、PairMAE 平均不劣于 B-A 超过 0.010、无阻断；符合者选
S1/S2 平均 RMSE 最小，0.002 内 tie 按 B>S>O，再 A 优先 R；否则 `C_deploy=B-A`。

SG：B/S/O/D 各 A 三 seed 优先，再 F/G 与（若候选用 R）C_deploy 的 R 三 seed，最多 21。
固定 5760 updates：循环新 Fit permutation，末 batch 可不足 128，每 16 步一个评价周期
（共 360 周期，**不是 360 个自然 epoch**），norm 只用 SG Fit。单列报告，不与 S1/S2 合并。
SG 结果不能返回改架构/λ/C_deploy。

全 Train（`ft`）：Fit=Train3360、Select=public Validation420，360 自然 epoch、batch128、
9720 updates；至少 0.75h wall；先 B-A 与 C_deploy 三 seed，再 D-A/G-A（重复条件去重），
最多 12 次主模型；F-A 与 CPU KRR 作轻量参照；不使用 public Test420。

## 9 评价、统计、判定

主指标：raw RMSE/MAE/bias/R²（各评价分区）；组件等权的真实 pair PairMAE/PairRMSE；
best/final、参数、updates、GPU-h、推理时间。pair 指标按组件 bootstrap 2000；
分子指标按分子—核心 group bootstrap；跨 S1/S2 汇总按**原始全局组件 ID** 一起抽，
保留其在不同 split/seed 的所有出现。先逐 seed 差再固定单元平均；不报 best seed。

固定对比：R-A；R-W、R-N；S-A vs B-A；O-A vs S-A；D-A vs O-A（并报 D-A vs S-A）；
loss 交互 `[m-R - m-A] - [B-R - B-A]`（不叫化学协同）；F/G/KRR/RF 性能位置，F-R vs F-A。

判定（本轮实用/购买标准，非领域统一阈值）：

* `EFFECT_REPLICATED`：6 个 split-seed 单元 ≥4 个 >0，S1/S2 各自均值 >0，
  跨 split 组件 bootstrap 95% 下界 >0。
* `MATERIAL_MOL`：+ RMSE 平均改善 ≥0.010；`MATERIAL_EDIT`：+ PairMAE 平均改善
  ≥max(0.020, 参照 5%)。
* `CONTROL_INCREMENT`：相对控制 RMSE ≥0.005、PairMAE ≥0.010 且两终点均
  EFFECT_REPLICATED。`PAIR_NONINFERIOR`：PairMAE improvement 区间下界 ≥ -0.010。
* 判定表 SUPERVISION_SUPPORTED / WEIGHTING_EXPLAINS_GAIN /
  WEIGHTING_VS_PAIRING_UNRESOLVED / GENERIC_DIFFERENCE_GAIN / EDIT_ONLY_TRADEOFF /
  STATIC_CONTEXT_SUPPORTED / DYNAMIC_COMPOSITION_ADVANTAGE /
  CAPACITY_OR_OPTIMIZATION_DOMINATES / CAPACITY_GAIN_WITH_UNRESOLVED_CONTEXT_VALUE /
  NO_LCRC_SPECIFIC_ADVANTAGE / LCRC_SPECIFIC_ADVANTAGE_NOT_ESTABLISHED /
  DATA_SENSITIVE / SPLIT_DEPENDENT / UNRESOLVED_UNDER_SCAFFOLD_SHIFT /
  CONTEXT_EFFECT_SAMPLE_LIMITED / INTERVENTION_WEAK / PARTIAL_REPLICATION /
  SUGGESTIVE_UNCERTAIN / UNRESOLVED。多项可同时成立，不强迫单一赢家。

上下文双差分（§12.4）：同变换类不同配对组件两对
`context_true = Δy1 - Δy2`、`context_pred = Δf1 - Δf2`；零参照为同变换处处相同。
独立主表用全局贪心组件互斥，按 `sha256("lipo-route-v1-context"|class|pair1|pair2)`
排序，每组件整表最多用一次；少于 20 个比较标 `CONTEXT_EFFECT_SAMPLE_LIMITED`。

## 10 预算、调度与降级

Wall 目标 7.5h / 硬 8h（从接受任务起）；GPU 软 14 / 硬 16 A100·h（含 smoke、校准、
失败与重跑）；T+6h45m 前训练完成，T+7h 后不再训练。持久化队列
（`configs/run_matrix.csv` + `analysis/queue.json`）与 `analysis/budget_ledger.json`，
状态 PENDING/RUNNING/COMPLETE/FAILED/INCOMPLETE/NOT_PURCHASED。每类按实测端到端×1.25
保守投影，只准入能完成的**完整比较包**。降级顺序见任务书 §14（1…8）。同配置基础设施
重试一次；改变科学计算的 bug 修复后旧 run 作废且成本不删；某架构持续工程阻断则
`BLOCKED_IMPLEMENTATION`，其余独立问题继续。不因曲线差提前停止正式 run。

## 11 实现映射

| 协议要素 | 代码 |
|---|---|
| 常量/哈希/判定阈值/预算 | `code/lr_common.py` |
| 五个划分与 SG 构造 | `code/lr_split.py`（输出 `analysis/split_*.json`） |
| 结构 cache / Morgan / Fit-only norm / 标签权限 | `code/lr_data.py` |
| 真实 pair 表、冻结 pair stream、N 重配 | `code/lr_pairs.py` |
| B/S/O/D/F/G 与 T 算子 | `code/lr_models.py` |
| 单训练单元（含辅助前向、checkpoint、预测） | `code/lr_train.py` |
| 一次剂量校准 | `code/lr_dose.py` |
| 正确性 gates | `code/lr_gates.py` |
| 队列/预算账本 | `code/lr_queue.py` |
| 准备（cache/fp/plan） | `code/lr_prepare.py` |
| 统计、对照、影响度、上下文双差分、机制诊断 | `code/lr_analysis.py` |
| 图表与报告 | `code/lr_report.py` |

## 12 不能声称

化学因果效应；原子/路径真实贡献或唯一误差分配；全新独立外部验证；通过 Audit 后
追加模型或调 λ；"已修复 LCRC 单分子预测器"；性能/榜单结论；把本轮 dynamic 结果
推广成"迭代普遍必要"；把 O 说成"完全没有广义 message aggregation"。
