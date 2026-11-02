# REPORT — LCRC Predictive Strength & Traceability Benchmark v1

Round / Protocol: `lcrc-predictive-trace-benchmark-v1`
Track: `tracks/structural-sufficiency/lcrc_predictive_trace_benchmark_v1/`
参照 HEAD：`9628ab4ca901aeecd2957b9e2c172a9fab485f7d`
本文件只报告**本轮实际执行了什么、得到了什么**；设计定义见 `PROTOCOL.md`
（在其 SHA-256 于 `analysis/protocol_freeze.json` 固定之后未再修改）。

---

## 0. 一页结论

| 假设 | 结果 | 判定 |
|---|---|---|
| **H1** 预测强度：C1（228 674）能否替代 A0（247 106） | 见 §3/§4 | {{H1_VERDICT}} |
| **H2** 结构机制：typed-S2 输入通道被训练权重依赖的程度 | 零输入后 \|Δpred\| 见 §5 | {{H2_VERDICT}} |
| **H3** 局部编辑误差是否能被迹特征预测 | `Gain = +0.0015`，CI 跨 0 | `NO_INCREMENT_THIS_PROBE` |
| **H4** TRACE vs NATIVE/FAMILY/GLOBAL 的定位 | 与零预测器打平 | `NO_INCREMENT_THIS_PROBE` |

* TU 交叉验证方案由**唯一允许的依据**（实测硬件成本）选出：
  `A_10x2` 与 `B_5x3` 均超预算，冻结为 **`C_5x2`（RESOURCE_LIMITED）**，
  fold_seed 20261102，每折外层训练集内 10% 分层 Valid（seed = 20261102 + fold_id）。
* 所有新 Test 结果标记 **`EXPOSED_TEST_EXPLORATORY`**：这些 Test 标签在历史轮次
  已被读过，本轮不是盲测。
* `Lipophilicity` 与 `ZINC` **不重训**，只读冻结历史并单独标注来源等级。

---

## 1. 执行事实

| 项 | 值 |
|---|---|
| 开始 / 结束（UTC） | {{T_START}} → {{T_END}} |
| 墙钟 | {{WALL}} |
| GPU | 2 × A100-SXM4-40GB（`res-gpu0`、`res-gpu1`），每 pool 至多一个正式训练进程 |
| 正式训练单元 | 分子 {{N_MOL}} + TU {{N_TU}} + local-edit 6 = **{{N_UNITS}}** |
| GPU 时间（全部，含失败与探针） | **{{GPU_H}} GPU-h**（软上限 14，硬上限 16） |
| 失败/未跑单元 | {{FAILED}} |
| CPU 工作量 | 13 个正确性门禁、P1 回溯 3054 条迹、探针 CV/审计、聚合与统计 |

调度：`code/ptb_queue.py` 在每个单元前读 `analysis/compute_ledger.json`，
拒绝会突破硬上限或墙钟保留时间的单元；`runs/<unit>/run.json` 已完成的单元跳过。
上游 `res` 工作树的脏改动没有被丢弃：先复制到 `~/cy/rr-backups/20261102T163030Z/`
再 `git stash --include-untracked`（见 §9）。

---

## 2. 正确性门禁（`analysis/correctness_gates.json`，13/13 PASS）

| 门禁 | 关键数字 |
|---|---|
| 参数量与列映射 | A0 = **247 106**，C1 = **228 674**，差 **18 432**；单任务局部输入列 529 → 433 |
| Tox21 头加宽 | 248 536 / 230 104，差仍 18 432 |
| 共享初始化 | C1 由 A0 初始 state 按冻结列映射删除，共享张量逐位相等 |
| 冻结重放 | 本仓库 A0 重实现 vs 上游冻结 Lipophilicity s0 Valid 预测，max\|Δ\| = **1.43e-6**（tol 1e-5） |
| dropout 语义 | 只在共享 `h` 上、每 forward 一次 |
| 重编号不变性 | 预测对原子重编号不变 |
| 批量/隔离 | 同批不同批预测一致；样本间不串扰 |
| s1/s2 路径构造 | 路径特征按冻结定义重建 |
| 梯度可达性 | 每个可训练张量都收到非零梯度（含 Tox21） |
| 归一化 | 只用训练部分拟合；Test 不被读取 |
| checkpoint 重载 | SHA 校验后重载预测逐位一致 |

完整机器可读记录：`analysis/correctness_gates.json`；表格见
`analysis/report_tables.md` §E。

---

## 3. H1 — 分子任务（36 个正式单元）

{{MOLECULE_TABLE}}

* 每格 3 seeds；`Δ` 为 C1 − A0（回归为 RMSE，越小越好；分类为 ROC-AUC，越大越好）。
* 配对 bootstrap 在 **Test 分子行**上重抽 2000 次（seed 20261102），
  三个 seed 各做一次并给出各自 CI；seed 离散度单独报告，
  不把 seed 当独立样本做显著性检验。
* MolHIV 上限 100 epoch，**不自动扩展**；{{MOLHIV_BUDGET}} 
* 冻结历史 Lipophilicity（只读、不重训）见生成表 §A 下半部分。

{{MOLECULE_READING}}

---

## 4. H1 — TU 实验包（{{N_TU}} 个单元，`C_5x2`）

方案选择只看**实测成本**（`analysis/cost_measurement.json`，
`code/ptb_protocol.py` 逐条评估）：

| 方案 | TU 单元 | 预测 GPU-h（含 1.2 安全系数） | 预测墙钟 | 结论 |
|---|---|---|---|---|
| `A_10x2` | 160 | 13.32 | 13.32 h | 超预算，不可行 |
| `B_5x3` | 120 | 9.95 | 9.95 h | 超预算，不可行 |
| **`C_5x2`** | **80** | **6.66** | **6.66 h** | **选中（RESOURCE_LIMITED）** |

{{TU_TABLE}}

* 每折外层训练集内 10% 分层 Valid 用于 checkpoint 选择（acc → 更低 CE → 更早 epoch），
  外层 held-out fold 只在 checkpoint 哈希落盘后评估一次。
* 节点类型词表与输入标准化只从该 fold 的实际训练部分拟合（含 Train-only 均值审计）。
* 不把 fold×seed 全部分数当独立样本；配对 bootstrap 以 **outer fold** 为重抽单元。

{{TU_READING}}

---

## 5. H2 — 结构机制审计（`analysis/mechanism_audit.json`）

### 5.1 只读冻结证据

从五个冻结轮次重读了对比表（`lcrc_lipo_model_dissection_followup_v1` 的
follow-up 勘误表优先，legacy 表只作对照），共 {{N_EVIDENCE}} 条，每条带源文件
SHA-256 与 split 标记。**没有**重新训练、重新打分或重新读 Test。

### 5.2 `FROZEN_MODEL_INPUT_INTERVENTION`（不重训的输入通道干预）

把 typed-S2 输入列（`LOCAL_LAYOUT["s2_type"]…["s2_content"]`，即 529 维局部块中的
**320:520 共 200 列**）整体置零，固定权重前向：

{{INTERVENTION_TABLE}}

对照：同一 checkpoint 上把 `g`（全局 42 列）与 `mean_h2`（rho 64 列）分别置零。
typed-S2 的平均 \|Δpred\| 是对照通道的 {{S2_RATIO}} 倍。

**边界声明**：这是固定权重下的输入通道干预，
**不是**同容量重训消融，**不是**合法化学编辑，
也不能据此声称该通道"必要"或"充分"。

### 5.3 参数差分解（实测，不是假设）

A0 − C1 = **18 432** 参数，全部落在 `local.mlp.lins.0.weight`：
`192 × 529 = 101 568` → `192 × 433 = 83 136`，即 96 列 × 192 投影。
其余张量逐位相同。

---

## 6. 效率（同环境、同 batch、同数据；`analysis/efficiency_table.csv`）

{{EFFICIENCY_TABLE}}

推理吞吐测在 **Validation** 结构缓存上（不触碰 Test）。
`epoch s` 来自本轮实测成本运行；`best-before-best` 的累计 GPU 时间按
best_epoch/epochs 线性估计并在 `analysis/efficiency_measurements.json` 标注
`ESTIMATED_FROM_BEST_EPOCH`。

---

## 7. H3 / H4 — Local-edit 回溯（独立冻结轮次）

该部分在 `../lcrc_local_edit_error_diagnosis_v1/` 内完成，本轮**没有**修改其科学定义，
只补上缺失的 `code/edit_probe.py` 并跑完 6 个训练单元、P1 回溯、一次性 Audit。

| 项 | 值 |
|---|---|
| 训练单元 | 6（A0/C1 × seed 0,1,2，360 epoch，Fit2016/Select336） |
| GPU 时间 | 0.3992 GPU-h |
| 迹 | 3054 条；恒等式 max 残差 1.1e-6；反序 5.7e-7；重编号 8.3e-7 |
| Audit | 74 对 / 26 个配对组件 / 19 个共享类，**只读一次** |
| C1 `Gain` | **+0.0015**（+0.0030 / +0.0036 / −0.0022），CI [−0.0108, +0.0114] |
| A0 `Gain` | +0.0033（3/3 正号） |
| 预注册判定 | **`NO_INCREMENT_THIS_PROBE`**（阈值 0.03） |

关键读法：Audit 上每个单元的最佳方法基本都是 `NATIVE`（预测 e_delta = 0）
或与其相差 <0.003 的 `FAMILY`；`Gain > 0` 只因为 Diagnose 选出的 `BEST_BASE=GLOBAL`
在 Audit 上比零预测器差 0.004–0.006。详见该目录 `REPORT.md` / `DECISION.md`。

---

## 8. 与冻结协议的事后偏差与不可声明项

{{DEVIATIONS}}

**不能声明**：

1. 不能说 C1 与 A0 "统计等价"：CI 跨 0 只表示证据不足，不表示等价。
2. 不能把 MolHIV 100-epoch 结果与历史 360-epoch A0 做配对比较。
3. 不能把 TU 的 fold×seed 分数当独立样本；不能把 `C_5x2` 当作与 10 折等价的证据强度。
4. 不能把 `FROZEN_MODEL_INPUT_INTERVENTION` 当作消融或化学编辑结论。
5. 不能把 Lipophilicity/ZINC 的冻结历史结果当作本轮新证据。
6. Test 结果均为 `EXPOSED_TEST_EXPLORATORY`，不是盲测。

---

## 9. 交付与复现

| 文件 | 内容 |
|---|---|
| `PROTOCOL.md` | 冻结设计（本文件不改它） |
| `configs/protocol.json`、`configs/run_matrix.json` | 机器可读协议与运行矩阵 |
| `analysis/protocol_freeze.json` | 三份设计文件的 SHA-256 + 代码哈希 + 选中方案 |
| `analysis/source_manifest.json` | 数据/上游来源与哈希 |
| `analysis/split_manifest.json` | TU 折划分、指纹与每折哈希 |
| `analysis/correctness_gates.json` | 13 个门禁的机器可读结果 |
| `analysis/run_registry.json`、`analysis/compute_ledger.json` | 每个单元的调度与 GPU 记账 |
| `analysis/checkpoint_manifest.json` | 冻结 checkpoint（含 SHA，冻结后禁止重训） |
| `analysis/test_access_ledger.json` | 一次性 Test 访问审计 |
| `analysis/molecular_results.csv`、`analysis/tu_results.csv` | 逐单元结果 |
| `analysis/paired_differences.csv` | 配对差（分子按行、TU 按 outer fold）+ 冻结历史对比 |
| `analysis/statistics.json` | bootstrap 协议、seed 离散、失败计数 |
| `analysis/sample_diagnostics.json` | 逐样本改善/退化、极端误差集中度、top-k 移除敏感性、size bin |
| `analysis/efficiency_table.csv` | 效率表 |
| `analysis/mechanism_audit.json` | H2 证据 + 干预 + 参数分解 |
| `analysis/literature_reference.csv` | 固定外部参照（不做实时检索） |
| `analysis/report_tables.md` | 由 `code/ptb_report.py` 从上述 CSV 生成的表格 |

复现（在 `res` 上）：

```bash
uv run python code/ptb_gates.py                     # 13 个门禁（CPU）
uv run python code/ptb_protocol.py                  # 由实测成本冻结 TU 方案
uv run python code/ptb_queue.py --plan molecule --pool-label res-gpu0
uv run python code/ptb_queue.py --plan local_edit_then_tu --pool-label res-gpu1
uv run python code/ptb_freeze.py --molecule --tu     # 只在全部单元完成后
uv run python code/ptb_score.py --all                # 一次性 Test
uv run python code/ptb_aggregate.py
uv run python code/ptb_efficiency.py --device cuda
uv run python code/ptb_mechanism.py --device cuda
uv run python code/ptb_report.py
```
