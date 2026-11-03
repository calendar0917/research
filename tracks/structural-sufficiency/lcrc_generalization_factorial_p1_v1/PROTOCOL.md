# PROTOCOL — LCRC-P1：固定基准划分上的四臂机制与跨划分稳定性

Round: `lcrc-generalization-factorial-p1-v1`
Track: `tracks/structural-sufficiency/`
代码：`code/`（本轮只新增**分析**与**调度**代码；训练代码 0 行新增）
冻结文件：`analysis/protocol_freeze.json`（写于任何本轮的 GPU 训练启动之前）

本文件冻结本轮所有会受结果影响的量。**它不含结果**；结果只写在 `REPORT.md`、
`analysis/*.json`、`analysis/*.csv` 与 `analysis/runs/` 的逐 run 记录里。

---

## 0. 角色与边界

本轮是**执行轮**，不是设计轮。研究问题、实验设计、模型组合、划分选择、评价
指标与停止条件已经确定。本轮不做以下任何一件事：

- 检索论文、设计科研路线；
- 新增 LCRC 架构、损失、特征或数据划分；
- 改动已指定的模型、训练配方、评价口径；
- 因 Test 表现调整超参数或重挑 seed；
- 启动 P2 或后续模型搜索。

P0（`lcrc_failure_mechanism_p0_v1`）已确认：在 random42 上，exact rooted-radius-2
support 对 LCRC−PNA 的误差差距没有可检测的增量预测价值（OOF ΔMSE `+0.0109`，
CI `[−0.00123, +0.02402]`）。因此本轮**不**训练支持度加权、support backoff 或
任何相关新模型；rooted-S2 support 只作为**描述性协变量**保留。

---

## 1. 两个固定问题

**Q1（random122，四臂 × 三 seed）**
CHEM8 与 MULTICONTRIB8 的组合是否具有可重复的性能交互？

- 主效应：`I = MSE(COMBO) − MSE(CHEM8) − MSE(MULTICONTRIB8) + MSE(BASE)`，负值
  表示有利的 MSE 交互。
- 重点检验 seed 0 历史上看到的交互（Test `−0.0207`，CI `[−0.0525, +0.0098]`；
  Validation `+0.1496`，CI `[−0.0083, +0.4432]`）是否在 seeds 1、2 上同方向，
  **而不以某个 Validation 极端增益为证据**。

**Q2（random123 / random124，BASE vs COMBO × seeds 0/1）**
COMBO 相对 BASE 的增益是否跨已有数据划分稳定（总体误差与尾部风险）？

- 主对比：`COMBO − BASE` 的 Test RMSE（负值 = COMBO 更好）。
- 每个 split/seed 单独记录，并给 seed 均值。

---

## 2. 数据划分（冻结，禁止重新生成）

直接复用仓库现有的 ChemBench Lipophilicity 冻结划分：

| 本轮名称 | 文件 | 大小 (Train/Valid/Test) |
|---|---|---|
| `random122` | `Lipop.random122.ind.pkl` | 3360 / 420 / 420 |
| `random123` | `Lipop.random123.ind.pkl` | 3360 / 420 / 420 |
| `random124` | `Lipop.random124.ind.pkl` | 3360 / 420 / 420 |

标签表为三者共用的 `Lipop.csv.gz`（4,200 行 `smiles,exp`）。

**禁止**重新生成随机划分，**禁止**额外切分 Train。

审计项（任一失败即阻塞受影响比较）：

1. 三个 pickle 的 SHA-256；三个 part 的行 ID 数组 SHA-256 必须与冻结 run 记录
   一致；
2. 每个 split 的 Train/Valid/Test 行 ID 集合两两不相交、并集恰为 `{0..4199}`；
3. 4,200 行 SMILES 互不重复（保证 row ID ↔ 分子是双射，因此任何划分文件的
   索引空间差异都会暴露）；
4. 由 row ID 反推的整表标签向量在三个 split 上逐行相同；
5. 每个 unit 的 `y_true` 必须与对应 split 的冻结标签**逐位相等**；
6. 每个 unit 的 Train-only 标准化参数（`mean`, `scale`, `n=3360`）在同一 split
   内完全相同，且 `source = train_split_only`；
7. 输入缓存指纹：`base` / `multicontrib` 的 train/valid 指纹必须等于 chem8 臂
   记录的 `*_base` 指纹（同一份分子切分与同一份基础特征）。

---

## 3. 运行单元（unit）设计

一个 unit = 一个 (split, arm, seed) 的完整训练 + 一次 Test 评分。

### 3.1 Q1：random122 四臂 × 三 seed（12 格）

| arm | 定义（完全复用现有代码） | 参数 |
|---|---|---:|
| `BASE` | 冻结 LCRC-A | 247,106 |
| `CHEM8` | 既有原子级八维化学扩展（CIP_R/S/UNASSIGNED_TETRA, HBD, HBA, HYB_SP/SP2/SP3） | 248,130 |
| `MULTICONTRIB8` | 既有八通道贡献读出（`Q:64→64→8`，graph sum，`delta=MLP([Q_graph, log1p(n)])`，末层零初始化） | 252,139 |
| `COMBO` | CHEM8 + MULTICONTRIB8 | 253,163 |

**不新写模型设计。** 三个扩展臂都是 BASE 的单个零初始化扩展，共享张量逐位从
同 seed 的 BASE 模型拷贝，step 0 与 BASE 等价（≤1e-6）。

### 3.2 Q2：random123 / random124，BASE vs COMBO × seeds 0/1（8 格）

只比较 BASE 与 COMBO，seeds 固定为 0、1。**不追加“表现好的 seed”**，不因结果
反转自动加 seed 2。

### 3.3 复用清单与缺口

复用判定规则（必须**全部**成立，否则该 unit 不得复用）：

1. `dataset/split/arm/seed` 身份一致；
2. 训练代码 12 个文件的 SHA-256 **逐字节相同**（`cl_common.code_hashes()`）；
3. R3 配方 22 个冻结键相同，`shuffle_generator_seed = seed + 917000`；
4. Train-only 标准化参数相同；
5. 冻结 split 行 ID 哈希相同；
6. 逐分子预测文件的行顺序与冻结 split 顺序严格一致；
7. `y_true` 与冻结标签逐位一致；
8. 重算 RMSE/MAE/SSE 与记录值差 ≤ `1e-5`；
9. `selection_uses_test = false` 且 `test_used_for_any_selection = false`；
10. 执行体制一致（同 host / torch / CUDA / GPU 型号）。

预运行审计见 `analysis/run_reuse_manifest.json`。结论：**10 个 unit 可复用，
10 个 unit 必须新训练**：

| 复用（`lcrc_performance_longrun_v1`） | 新训练（本轮） |
|---|---|
| random122 BASE s0/s1/s2 | random122 CHEM8 s1, s2 |
| random122 CHEM8 s0 | random122 MULTICONTRIB8 s1, s2 |
| random122 MULTICONTRIB8 s0 | random123 BASE s0, s1 |
| random122 COMBO s0/s1/s2 | random123 COMBO s1 |
| random123 COMBO s0 | random124 BASE s0, s1 |
| random124 COMBO s0 | random124 COMBO s1 |

一个模型不得使用历史不同配方、另一个使用本轮配方，然后把差值称为公平对照；
因此新训练的 BASE/COMBO 与复用的对应单元必须共享同一套代码哈希（见 §3.3 第 2 条）。

---

## 4. 训练配方（完全复用已验证的 R3）

| 项 | 值 |
|---|---|
| epoch cap | 360 |
| optimizer | AdamW (0.9, 0.999), ε = 1e-8 |
| peak / min LR | 3e-4 → 1e-5，per-step cosine，horizon 360 epochs |
| weight decay | 1e-3 |
| batch size | 128（27 steps/epoch，9720 updates） |
| dropout | 0.10（最后一个 atom-level `h`，在 `f`/`rho` 之前） |
| gradient clip | 5.0 |
| loss | Train 标准化目标上的 MSE |
| 标准化 | Train-only（`n=3360`），Valid/Test 只做编码或评估 |
| checkpoint | Validation RMSE 最低；严格改善；tie 保留更早 epoch |
| soup / EMA / SWA / SSL / early stopping | 全部关闭 |
| 超参搜索 | 无 |

模型构造、初始化、训练、加载规则全部来自
`lcrc_performance_longrun_v1/code/cl_train.py`（本轮 `code/run_unit.py` 只把
输出目录改到本轮，不改任何训练逻辑，并在训练前重新校验冻结协议哈希）。

新增模块的零初始化、共享初值、有效梯度、前向等价与 checkpoint reload 由

- `pytest tracks/structural-sufficiency/lcrc_performance_longrun_v1/tests`（19 项，本地通过），
- 本轮 `tests/test_p1.py`（含 `verify_frozen_driver` 哈希校验），
- 每个 unit 的 `train.json:model.extension_zero_init.passed == true`

共同覆盖。

---

## 5. 评价（全部在原始 logD 单位）

### 5.1 逐 split / seed / arm 记录

`RMSE`、`MAE`、`P90 |AE|`、`CVaR90-AE`、`SSE`，以及 `COMBO − BASE` 的逐分子
squared-error difference 与改善分子比例。Validation 与 Test 都报告，并给出
方向是否一致的汇总。

精确定义（先冻结）：

- `P90 |AE|` = `|AE|` 的 90 百分位，numpy 线性插值；
- `CVaR90-AE` = **本模型自己**最差 `ceil(0.10 × n)` 个分子的 `|AE|` 均值
  （n = 420 → 42 个分子）；
- `改善分子比例` = `SE_combo < SE_base` 的 row ID 占比。

### 5.2 四臂交互（仅 random122）

`I = MSE(COMBO) − MSE(CHEM8) − MSE(MULTICONTRIB8) + MSE(BASE)`，per seed，
以及在 Validation 与 Test 上分别给出；同时给出三个 seed 的 `I`、均值、
scaffold-cluster bootstrap 区间与方向一致性计数。

**不把统计交互直接写成已证实的化学机制。**

### 5.3 难例分组（组边界不得由误差定义）

| 分组 | 边界来源 |
|---|---|
| scaffold seen / unseen | 同一 split 的 Train 中出现过的 Bemis–Murcko scaffold（无环分子用规范化 SMILES 作 key，`scaffold_fallback_used` 记录） |
| heavy-atom count 四分位 | 同一 split 的 **Train** 分子的四分位边界 |
| TPSA 四分位 | 同上 |
| 训练集最大 Tanimoto 四分位 | **Train 侧**：每个 Train 分子对其他 Train 分子的最大 Morgan(r=2,2048) Tanimoto（留一）；边界由这组 Train 值计算；query 侧用对全部 Train 的最大值 |

组边界一次算定后**原样**用于 Validation 与 Test，不按 Test 误差重定义，不合并
空箱。rooted-S2 support（`unseen_atom_frac`、`rare_atom_frac`、
`median_log1p_df`、`p10_log1p_df`）只作补充描述变量。

BASE 实际误差最大的 10% 样本可做事后诊断，但必须标记
`LABEL_SELECTED_POSTHOC_TAIL`；**不得**用 BASE-top-tail 的改善代替 P90 或
CVaR90，也**不得**因其 bootstrap CI 排除零就声称模型具备可预先识别的尾部修复
能力。

### 5.4 统计核验

- 所有配对比对使用相同 row ID；
- 每个 split 以 **Bemis–Murcko scaffold cluster** 为单位做 2000 次 bootstrap，
  百分位区间 `(2.5, 97.5)`，随机流种子 `20261104`；
- **seed 不当作独立分子样本**：seed 均值区间在同一重采样上对每个 seed 分别
  计算统计量后取平均；
- random122/123/124 来自同一 4,200 分子池，**不**视为独立外部 cohort；三者的
  结果只做“同一分子池的不同划分”这一层解释；
- 不删除任何极端分子重算主成绩；
- top-5 / top-10 SSE 正负贡献按 row ID 汇总，只作集中度诊断。

### 5.5 Test 口径

Test 本轮明确为 `DEVELOPMENT_TEST`：

- 可以正常评分、导出逐分子预测、开展误差诊断；
- 允许研究者后续依据 Test 汇总结果决定研究方向；
- 每个 run 只用 Validation 选 checkpoint；
- 不允许 Test 梯度训练，不允许在同一个 run 内用 Test 选 epoch；
- 报告不得宣称 Test 是独立、未曝光的最终确认集。

---

## 6. 预算与停止规则

- 硬上限 **2.0 A100 GPU-hours**（`res`，`res-gpu0` / `res-gpu1`，一次一个 unit
  占一张 A100）。
- GPU 预算的计费口径 = 每个新 unit 的 job wall time；复用 unit 的历史 GPU 时间
  计入 `lcrc_performance_longrun_v1`，不计入本轮上限（账本中单列）。
- 先审计已有运行、给出待运行单元清单与投影，再启动 GPU
  （`analysis/dispatch_plan.json` 与 `analysis/dispatch.log` 记录）。
- 若投影超过上限：不启动、不缩减训练周期，保留完整记录并停止。
- 若某 unit 失败：不静默重试掩盖，不以别的 seed 顶替；记录失败并停止受影响的
  比较。
- 不因结果好坏增删 unit。10 个新 unit 是固定的。

---

## 7. 产出

```
PROTOCOL.md
REPORT.md
analysis/source_audit.json
analysis/run_reuse_manifest.json
analysis/metrics_by_split_seed_arm.csv
analysis/per_molecule_predictions.csv
analysis/factorial_interaction.json
analysis/tail_risk.json
analysis/scaffold_subgroups.json
analysis/paired_uncertainty.json
analysis/gpu_ledger.json
analysis/integrity_checks.json
analysis/concentration_diagnostics.json      (附加：top-k SSE 集中度)
analysis/manifest.json                       (附加：产物清单与复现命令)
code/…                                       (分析、调度、unit wrapper)
tests/…                                      (自动化测试)
```

---

## 8. 报告必须明确回答的五个问题

1. random122 的四臂交互是否跨三个 seed 保持同方向？
2. COMBO 相对 BASE 的 RMSE/MAE 改善是否在 random122/123/124 均出现？
3. CVaR90-AE 是否改善，还是仅 BASE-defined tail 的事后结果好看？
4. 改善是否只来自少数大 recovery，同时伴随相当大的 damage？
5. 是否存在一致的 scaffold-unseen 特异性收益？

不回答除此之外的开放问题；不为下一版架构取名；不启动下一轮实验。

---

## 9. 冻结与偏差纪律

- `analysis/protocol_freeze.json` 在本轮第一个 GPU 作业之前写入，记录本文件与
  分析代码的 SHA-256。
- 分析阶段的最终产物必须由 `code/run_analysis.py`（无 `--no-freeze-check`）
  生成；`--no-freeze-check` 只用于冻结前的本地调试。
- 任何事后发现的代码缺陷，修正必须以 `revisions` 条目追加到
  `protocol_freeze.json`（原因、影响范围、是否发生在见到本轮新数值之前），
  并同步重建全部产物。不静默修改。

---

## 10. 已知偏差与非声明

- 本轮复用的 10 个 unit 的历史结果在 P0 已被阅读过（random122 seed 0 的四臂与
  seeds 1/2 的 BASE/COMBO、random123/124 seed 0 的 COMBO）。因此本轮**不是**
  对这 10 格的盲测；新增的 10 格（CHEM8/MULTICONTRIB seeds 1–2、random123/124
  的 BASE 与 COMBO seed 1）在两两对照上是新的观测。
- random122/123/124 是同一分子池的三个随机划分，分子身份与标签完全共享。
- 单次 bootstrap 区间描述冻结预测在这 420 个分子上的抽样不确定性，不包含训练
  噪声的完整分解。
- 不宣称因果关系、化学机制或“尾部可预先识别”。
