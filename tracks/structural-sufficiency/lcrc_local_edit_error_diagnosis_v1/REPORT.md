# REPORT — `lcrc_local_edit_error_diagnosis_v1`（本轮完成记录）

Round: `lcrc-local-edit-error-diagnosis-v1`
参照 HEAD：`9628ab4ca901aeecd2957b9e2c172a9fab485f7d`
本轮工作：**只执行**该目录 `PROTOCOL.md` 已冻结的科学定义，补上缺失的
`code/edit_probe.py`，跑完 6 个训练单元、P1 完整回溯、一次性 Audit 与诊断。
**没有**修改 509 对 / 211 类变换 / 2016-336-504-504 划分 / R3 360 epoch /
L2 = 5 / 探针方法集合 / 预注册判定阈值中的任何一项。

---

## 0. 结论摘要

| 问题 | 结果 |
|---|---|
| 6 个训练单元 | 全部完成（A0/C1 × seed 0,1,2，360 epoch，Select 选择） |
| P1 回溯 | 3054 条配对迹（509 对 × 6 单元），恒等式与交换/批量/重编号检查全部通过 |
| 探针 | `code/edit_probe.py` 实现完成，冻结后只读 Audit 一次 |
| C1 主结果 `Gain` | **+0.0015 raw logD**（3 seed：+0.0030 / +0.0036 / −0.0022） |
| A0 参照 | +0.0033（+0.0039 / +0.0056 / +0.0003） |
| 预注册判定 | **`NO_INCREMENT_THIS_PROBE`**（阈值 0.03，未达标） |
| 与平凡零预测器比较 | TRACE −0.0023 / +0.0035 / −0.0069（C1），即**与"预测 0"打平** |
| 迹恒等式 | `d_pred = dL_var + dL_core + dR`，最大残差 1.1e-6 raw |
| 闭环残差 | 30 个三角，mean \|Σê\| = 0.0081，max 0.0560（非零） |
| gauge 稳定性 | 等价变换后 Audit MAE 变化 0.38–0.40 logD（描述性，主指标不变） |

一句话：**局部编辑的误差在这个 Audit 集上不可由这些分支归因特征预测**；
`Gain > 0` 只是因为 Diagnose 上选出的 `BEST_BASE=GLOBAL` 在 Audit 上比零预测器略差，
不是 TRACE 提供了信息。

---

## 1. 执行事实

| 项 | 值 |
|---|---|
| 训练 | 6 单元，每个 360 epoch，batch 128，AdamW 3e-4 → 1e-5，dropout 0.10 |
| GPU | A100-SXM4-40GB（`res-gpu1`），每 pool 一个正式训练进程 |
| 训练 GPU 时间 | **0.3992 GPU-h**（6 单元合计，单单元 222–250 s） |
| 回溯 + 探针 | CPU（`res-local` pool），不占 GPU |
| 单元 GPU-h 记录 | `../lcrc_predictive_trace_benchmark_v1/analysis/compute_ledger.json`（`kind=local_edit`） |

训练单元（Select RMSE 为 checkpoint 选择指标，原始 logD）：

| unit | best epoch | Select RMSE | 参数 |
|---|---|---|---|
| A0:s0 | 139 | 0.6830 | 247 106 |
| A0:s1 | 140 | 0.6805 | 247 106 |
| A0:s2 | 134 | 0.6882 | 247 106 |
| C1:s0 | 182 | 0.6846 | 228 674 |
| C1:s1 | 126 | 0.6796 | 228 674 |
| C1:s2 | 91 | 0.6813 | 228 674 |

A0 与 C1 在 Select 上差异 ≤ 0.0046，且方向在两两配对里不一致 —— 与"删掉
`b_mean` 这一块对 Lipophilicity 类任务几乎无影响"的既有冻结结论一致。

---

## 2. P1 完整回溯（`analysis/trace_verification.json`、`analysis/pair_traces.csv`）

* 509 对 / 211 个变换类 / 250 个 core 类；四划分 2016-336-504-504。
* 每对记录 `pred_a_raw, pred_b_raw, L_a_raw, L_b_raw, rho_a_raw, rho_b_raw`、
  `dL_var_raw, dL_core_raw, dR_raw`、映射统计（`n_core_atoms_changed`,
  `max_abs_dh_unchanged`, `mapping_unique` …）。
* **恒等式**：`d_pred_raw = dL_var_raw + dL_core_raw + dR_raw`，
  最大绝对残差 **1.1e-6**，`identity_ok` 全部为真。
* **反序**：509 对反向传播后 `(φ(B)−φ(A)) = −(φ(A)−φ(B))`，
  最大 |Δ| **5.72e-7**，失败 0 例。
* **批量拆分/合并**：最大 |Δ| **4.77e-7**。
* **原子重编号**：502 个分子，`y/rho/h/f` 最大 |Δ| ≤ 8.3e-7。
* 迹分解的规模（mean \|·\|，Diagnose）：`|dL_var|` 0.33–0.36，
  `|dL_core|` 0.12–0.15，`|dR|` 0.09–0.11 —— 变体支路项比 reduction 项大 2–3 倍。

---

## 3. 探针冻结（`analysis/probe_freeze.json`）

按 `PROTOCOL.md` §6：

* 目标：`e_delta = d_pred − d_y`（**raw logD**，逐对）。
* 方法：`NATIVE`(=0)、`FAMILY`(=b_t)、`GLOBAL`(b_t + 3 个全局差)、
  `TRACE`(b_t + `dL_var`,`dL_core`,`dR`)、`LOCAL_ONLY`(b_t + 前两项)。
* 权重：组件等权（每组件总权重相同、组件内均分），loss = 各组件 MSE 之和。
* 加权 ridge、**无截距**、固定 **L2 = 5**（缩放系数与类项同罚），无 λ 搜索。
* 连续列除以 **Diagnose 组件等权 RMS**；无均值平移（保持反序反对称）；
  RMS=0 的列置 0 并记录有效秩。
* `BEST_BASE` 在 Diagnose 上用**留一配对组件** CV 选出，并列次序 NATIVE > FAMILY > GLOBAL。

留一组件 CV（25 折，组件等权 MAE，raw logD）：

| unit | NATIVE | FAMILY | GLOBAL | TRACE | LOCAL_ONLY | BEST_BASE |
|---|---|---|---|---|---|---|
| A0:s0 | 0.3305 | 0.3226 | **0.3171** | 0.3149 | 0.3146 | GLOBAL |
| A0:s1 | 0.3648 | 0.3599 | **0.3607** | 0.3650 | 0.3638 | GLOBAL |
| A0:s2 | 0.3312 | 0.3254 | **0.3277** | 0.3361 | 0.3357 | GLOBAL |
| C1:s0 | 0.3305 | 0.3285 | **0.3285** | 0.3291 | 0.3272 | GLOBAL |
| C1:s1 | 0.3589 | **0.3557** | 0.3560 | 0.3599 | 0.3580 | FAMILY |
| C1:s2 | 0.3563 | 0.3527 | **0.3518** | 0.3601 | 0.3586 | GLOBAL |

Diagnose 上 TRACE 与最佳 baseline 的差 ≤ 0.01，已经低于 0.03 的实质阈值；
类别先验 `b_t` 在 L2 = 5 下被压到 ≈0（68–71 个系数中绝大部分是类项，
有效秩与列数接近，说明不是数值病态而是**强收缩**）。

---

## 4. 一次性 Audit（`analysis/probe_audit_scores.json`）

Audit 74 对 / 26 个配对组件 / 19 个共享类；冻结系数直接套用，**不重拟合**；
Diagnose 未见的类其 `b_t = 0`。组件等权 MAE（raw logD）：

| unit | NATIVE | FAMILY | GLOBAL | TRACE | LOCAL_ONLY | BEST_BASE | Gain |
|---|---|---|---|---|---|---|---|
| **C1:s0** | **0.4070** | 0.4071 | 0.4124 | 0.4093 | 0.4091 | GLOBAL | **+0.0030** |
| **C1:s1** | 0.4178 | 0.4179 | 0.4199 | **0.4144** | 0.4158 | FAMILY | **+0.0036** |
| **C1:s2** | **0.4149** | 0.4145 | 0.4196 | 0.4217 | 0.4217 | GLOBAL | **−0.0022** |
| A0:s0 | 0.4221 | **0.4221** | 0.4274 | 0.4234 | 0.4233 | GLOBAL | +0.0039 |
| A0:s1 | 0.3944 | 0.3944 | 0.3982 | 0.3927 | **0.3923** | GLOBAL | +0.0056 |
| A0:s2 | 0.3916 | **0.3913** | 0.3968 | 0.3965 | 0.3950 | GLOBAL | +0.0003 |

读法（关键）：**每个单元里最好的方法几乎总是 `NATIVE`/`FAMILY`**，
即"预测 e_delta = 0"。`GLOBAL` 在 Diagnose 上赢了 CV，但在 Audit 上比零预测器
**更差** 0.005–0.006；`Gain` 正是拿这个更差的 baseline 作参照算出来的。

---

## 5. 预注册判定

`PROTOCOL.md` §7 规则逐条：

| 条件 | 阈值 | C1 实测 | 通过 |
|---|---|---|---|
| mean `Gain` | ≥ 0.03 logD | **+0.0015** | ✗ |
| 正号 seed 数 | ≥ 2 / 3 | 2 / 3 | ✓ |
| 组件配对 bootstrap CI 排除 0 | lo > 0 | [−0.0108, +0.0114] | ✗ |
| 单一组件不决定符号 | 最大份额 < 0.5 | 0.17–0.22 | ✓ |

**判定：`NO_INCREMENT_THIS_PROBE`** —— 幅度低于预注册阈值，且 CI 跨 0。

补充（同为预注册输出，不改判定）：A0 参照的 mean Gain = +0.0033（3/3 正），
同样远低于 0.03；化简与否不改变"迹探针无增量"这一结论。

---

## 6. 描述性诊断（不改变主指标）

**bootstrap**：Audit 配对组件上有放回重抽 2000 次（seed 20261102），
五个方法、三个 seed 与两个模型共享同一次重抽；
合并 CI [−0.0108, +0.0114]。CI 条件于本轮 6 个已训练单元，seed 离散度单列。

**影响度**（C1）：去掉 native 误差最大的组件后 Gain = +0.0012 / +0.0041 / −0.0050；
去掉 native 误差最大的分子后 +0.0031 / +0.0043 / −0.0023。
符号与量级不因单点移除而改变 —— 结论不是某个离群点造成的。

**闭环**：Audit 配对图有 30 个三角，`mean |Σ ê| = 0.0081`，`max = 0.0560`（raw logD）。
非零 → **不能声称**"单分子预测器被固定"。

**gauge 稳定性**（描述性）：施加等价变换
`c'_v = c_v + k`、`rho' = rho − k·n_atoms`（k = `GAUGE_K`）后，
`dL_var/dR` 改变而 `d_pred` 逐位不变（差 0.0），Audit MAE 变化 0.038–0.40：
TRACE 的系数与分支归因强耦合，属预期；因此本探针只能作为**误差预测器**读，
不能作为"化学贡献分解"。

---

## 7. 能说与不能说

**能说**

1. 迹恒等式在 6 个冻结 checkpoint 上成立到 1e-6，反序/批量/重编号不破坏它。
2. 变体支路项（`dL_var`）是迹分解中数值最大的一项，约为 reduction 项的 2–3 倍。
3. 以这些分支归因特征为输入的线性探针，**未能**在 Audit 上超过平凡的零预测器
   （C1 mean Gain +0.0015，CI 跨 0，低于 0.03 阈值）。
4. 化简（A0 → C1）不改变上述结论（A0 mean Gain +0.0033，同样不达标）。
5. 闭环残差非零，故单分子预测器未闭合。

**不能说**

* 不能说"局部编辑误差由变体支路主导"——数值占比不等于误差可预测性。
* 不能说"TRACE 优于 baseline"——`Gain` 为正只因 `BEST_BASE` 选到了更差的 `GLOBAL`。
* 不能把 gauge 后的 MAE 变化当作模型改写后的性能（未重训）。
* 不能把 74 对 / 26 组件的 Audit 结论外推到 ZINC 或其它编辑类型。

---

## 8. 交付清单

| 文件 | 内容 |
|---|---|
| `code/edit_probe.py` | **本轮新增**：`--stage fit|audit|report`，组件等权 ridge、留一组件 CV、一次性 Audit、bootstrap/影响度/闭环/gauge |
| `analysis/probe_freeze.json` | 探针冻结（系数、缩放、BEST_BASE、类别策略、代码与 commit 哈希） |
| `analysis/probe_audit_scores.json` | 一次性 Audit 评分与逐对残差 |
| `analysis/probe_report.json` | verdict、bootstrap、影响度、闭环、gauge |
| `analysis/trace_verification.json` | P1 验证汇总 |
| `analysis/pair_traces.csv` | 3054 条配对迹 |
| `analysis/label_access_ledger.json` | 标签访问审计（含训练期事件） |
| `DECISION.md` | 本轮判定与后续建议 |

复现（离线，`RR` 主机）：

```bash
uv run python code/edit_trace.py
uv run python code/edit_probe.py --stage fit
uv run python code/edit_probe.py --stage audit
uv run python code/edit_probe.py --stage report
```

`--stage audit` 在 `analysis/probe_audit_scores.json` 已存在时会拒绝第二次运行；
`--stage fit` 在 Audit 已被读取后会拒绝重拟合。
