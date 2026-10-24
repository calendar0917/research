# SFC-v0 CORRECTED — 修复后的重放与判定

协议 `sfc-v0-cpu-audit`（不变）。目录
`tracks/structural-sufficiency/zinc_structural_factor_contraction_v0/`。
本文件取代 `REPORT.md` 的结论；`REPORT.md` 保留为错误版本，逐条差异见
`ERRATUM.md`。

**本轮是 bug-fix replay，不是新模型搜索。** 架构、超参、合成任务、数据、划分、
判定门槛均未改动：K=8、400 steps、batch 64、lr 0.003、seeds {0,1}、
C / P / S / Srand / SP 五臂。CPU only，未启动 A100，未加载官方 valid/test。

---

## 0. 判定

**`COMPUTABLE_BUT_NO_SAMPLE_EFFICIENCY_EVIDENCE`**

标签与旧报告相同，但**理由完全不同**：

1. 旧报告说 "SFC 连训练集都拟合不了" —— **这是错的**，是评估期 bug 造成的假象。
   修复后 SFC 训练集准确率 reach **0.593**、junction **0.775**（随机 0.5）。
2. 真正成立的理由是：**修复后 SFC 仍然劣于同信息、同参数预算的 pair/junction
   对照 P**，且劣势超过 seed 波动。unseen 上 S−P = **−0.090**（reach）、
   **−0.347**（junction），而最大 seed spread 只有 0.005 / 0.041。
3. `PROMISING_FOR_ZINC_PILOT` 需要两个任务、两个 seed 都超过 P 且超过 seed 波动。
   reach 两个 seed 都低于 P，junction 两个 seed 都低于 P。**不成立。**
4. `COMPUTATION_INFEASIBLE` 不成立：10k 全量处理完，最大诱导宽度 3，
   K=8 单 epoch 前向+反向 49.5 s，无近似。

**不自动启动 ZINC 训练。**

---

## 1. 修复了什么

两个已定位缺陷 + 一个修复过程中发现的缺陷。完整证据见 `ERRATUM.md`。

| Bug | 内容 | 后果 |
|---|---|---|
| **A** | `with_marginals=False` 时 `T = logZ`；训练路径算边缘、评估路径不算 | 训练/评估特征不同（最大差 9.757），旧"S 不拟合"主要是这个假象 |
| **B** | `_minus_bethe_free_energy` 里 `log_q.detach()` / `log_K.detach()` | `T` 数值对、**梯度错**：森林上 `\|∇T\|` 高达 3.24（应为 0），下游 BCE 梯度误差 0.83 |
| **C** | `torch.as_tensor` 别名 → `sfc_fit_diagnostic` 就地覆写共享 `sfc_plan`；`_PLAN_CACHE` 用 `id(g)`；Srand buffer float32；缓存特征与 buffer 来自两个不同 RNG | Srand 缓存失效、初始化不可复现，本轮实测触发过一次崩溃 |

修复要点：`T` 未算边缘时**显式抛错**而不是返回替代值；`no_grad` 下**局部**
`enable_grad` 求边缘再 `detach`；`create_graph` 仅在需要反向时建立；
去掉两处 `detach()`；冻结因子全部 `.clone()`；plan 缓存改用拓扑键。

**检查矩阵：旧版 0/6，修复版 6/6**（`analysis/t0_corrigendum_prefix.json`
vs `_postfix.json`）。每项都在旧版实际暴露错误、在修复版通过：

| 检查组 | 旧 | 新 |
|---|---|---|
| 训练/评估/缓存特征逐位一致 + `T` 必须被拒绝 + `no_grad` 边缘对稠密枚举 | FAIL | PASS |
| `H` / `T` / 下游 BCE / `T@K=8` 对 float64 中心差分 | FAIL | PASS |
| 森林：`T≈0` **且** `∇θT≈0`（孤立点/单边树/路径/随机树/双分量） | FAIL | PASS |
| 环：`T` 非零且梯度与有限差分一致（ring4/ring5/方形带悬边） | FAIL | PASS |
| Srand：缓存 / 训练 / 评估 / plan 参数逐项一致 | FAIL | PASS |
| 冻结因子不可被诊断代码改写 | FAIL | PASS |

T1 原有 12 项（`logZ` 对稠密枚举、重编号、marginals、树零、消元序、确定性、
子图、饱和）在修复版上仍 **12/12**（`analysis/t1_correctness_corrected.json`）。
它们当年也全绿，因为**从未检查 `T` 的梯度或评估期一致性**——这是旧门禁的盲区。

---

## 2. 修复后的重放结果

同数据、同划分（`combination_overlap = 0`，标签率 0.5/0.5，边际 TV 不变）。
全部数字来自 `analysis/t3_synth_corrected.json`；曲线
`analysis/t3_bce_curves.csv` / `.png`。参数预算 ±3% 内一致。

**reach**

| 臂 | train acc | unseen acc | train BCE | 参数 |
|---|---|---|---|---|
| C 组成统计 | 0.588 | 0.524 | 0.668 | 17,158 |
| P pair/junction | **0.681** | **0.602** | 0.599 | 19,996 |
| S SFC（可训练） | 0.593 | 0.512 | 0.665 | 20,015 |
| Srand SFC（冻结随机） | 0.505 | 0.506 | 1.007 | 20,002 |
| SP = P+SFC | 0.677 | 0.606 | 0.603 | 19,975 |

**junction**

| 臂 | train acc | unseen acc | train BCE | 参数 |
|---|---|---|---|---|
| C 组成统计 | 0.526 | 0.535 | 0.691 | 17,158 |
| P pair/junction | **0.980** | **0.968** | 0.059 | 19,996 |
| S SFC（可训练） | 0.775 | 0.621 | 0.484 | 20,015 |
| Srand SFC（冻结随机） | 0.500 | 0.500 | 2.468 | 20,002 |
| SP = P+SFC | 0.982 | **0.979** | 0.065 | 19,975 |

对比（unseen）：S−P **−0.090 / −0.347**；S−C **−0.012 / +0.086**；
SP−P **+0.004 / +0.011**；Srand−P **−0.095 / −0.468**。
seed spread：P 0.005/0.039，S 0.003/0.041，SP 0.000/0.022。

### 2.1 训练 BCE 曲线（旧报告缺失的部分）

`analysis/t3_bce_curves.png`。要点：

* **S 现在真的在学**：junction 上 batch BCE 从 1.19/0.79 降到 ~0.48–0.50，
  对应训练集 0.775；reach 从 1.20/0.72 降到 ~0.66（≈ 组成统计的水平）。
* **Srand 在 junction 上完全不学**：BCE 停在 2.47（远高于 ln2=0.693），
  训练集 0.500 —— 冻结随机因子的 `[H,T]` 对该任务没有可用信息，
  与"S 靠学习才起来"一致。
* **P 在 junction 上 40 步内就降到 0.07**，S 到 400 步仍停在 0.48。
  差距不是"训练不够"，而是**同样的 400 步/64 batch/0.003 lr 下 P 收敛快一个量级**。
* **SP ≈ P**：修复后 SFC 不再"干扰"读出（旧报告说 −0.42，那是 Bug A）。
  但也**没有增量**（+0.004 / +0.011，均在 seed spread 内）。

### 2.2 SFC 参数的线性探针诊断（200 图，60 步）

reach 0.555→0.560，junction 0.510→0.480（`fit_diagnostics`）。
**这项诊断本身在旧版被 Bug C 污染**（就地覆写冻结因子），旧值不可用。
修复后的数值仍然说明：只优化 SFC 参数 + 闭式线性读出，学不动组合谓词。

---

## 3. 研究判断（严格按预注册规则）

**若修复后 SFC 仍无法拟合训练集** → 可以确认拟合/优化困难。
→ **这个分支不成立**：S 训练集达到 0.593 / 0.775，明显高于随机。

**若能够拟合却在 unseen 组合上明显落后于 P** → 判定当前算子的样本效率证据不足，
停止这条配置。
→ **这个分支成立**。两个任务、两个 seed 全部落后；junction 差距 0.347，
是 seed spread 的 8 倍以上。

因此：

* **停止当前 SFC 配置**（K=8、tanh 参数化、400 步、组合 split）。
* **不得**扩展为"整个共享因子方法族无效"。本轮只关闭了这一个配置；
  未扫描 K、未扫描参数化、未扫描训练长度、未测其他合成任务。
* **不得**声称"ZINC 上会更好"或"ZINC 上会更差"。本轮没有、也不应该做这个外推；
  合成图上的可区分性不等于真实任务性能。
* 唯一有意义的后续问题（若有人要继续）是：**为什么精确 `logZ` 的 `T` 在
  pair/junction 已经能线性解出的任务上没有增量**——这是关于配分函数可学习性的
  问题，不是 ZINC 的问题。

---

## 4. 计算成本（CPU only）

| 阶段 | 实测 |
|---|---|
| T2 全量 10k（修复版） | 212 s；宽度中位 2、p99 2、**max 3**；K=8 前向+反向 49.5 s/epoch |
| T3 完整重放（5 臂×2 seed×2 任务） | 1740 s（预演投影 1346 s，同量级） |
| 检查矩阵 + T1 + pytest | < 5 s |
| 峰值 RSS | 1788 MB（含 339 MB pickle） |

修复使 `T` 必须走二阶路径，单图成本上升但仍同量级（T2：51.4 s → 49.5 s/epoch）。
**没有超过合理 CPU 预算，未自动扩大实验。**

---

## 5. 出处限制（必须与第 2 节数字一起引用）

`t3_synth_corrected.json` 由 `sfc_core.py` sha `91eb9cd…` + `sfc_synth.py` sha
`0f50054…` 生成。Bug C 的 `.clone()` 修复在其后加入，故：

* **C、P**：不涉及 SFC 参数，数字有效。
* **S、SP**：特征数值路径正确，但初始参数来自已被 `sfc_fit_diagnostic` 覆写的
  `sfc_plan`（仍是一个随机初始化，协议允许，但不可声称是可复现的固定初始化）。
* **Srand**：缓存特征与 buffer 来自两组不同参数，**该行数字不能作为有效结果引用**。

修复后的代码已通过 `shared_plan_immutability` 与 `srand_frozen_factors`
保证三者一致，但**没有重跑完整 T3**（结论已定，按要求停止投入算力）。
重跑命令与耗时见 `ERRATUM.md` §6。**这一步不改变判定标签**：
C/P 与 S/SP 的差距在 seed 波动之外，且 Srand 无论用哪组冻结参数都低于 P。

---

## 6. 产物

| 文件 | 内容 |
|---|---|
| `ERRATUM.md` | 错误、影响范围、修复、检查矩阵、出处限制 |
| `REPORT_CORRECTED.md` | 本文件 |
| `REPORT.md` | **旧版，保留不动** |
| `analysis/t0_corrigendum_prefix.json` | 旧代码（逐字节快照）上的 0/6 失败证据 |
| `analysis/t0_corrigendum_postfix.json` | 修复版 6/6 |
| `analysis/t0_cost_preflight.json` | 重放前的成本预演 |
| `analysis/t1_correctness_corrected.json` | T1 12/12（修复版） |
| `analysis/t2_cost_corrected.json` | 全量 10k 成本（修复版） |
| `analysis/t3_synth_corrected.json` | 完整重放（含逐 step BCE 曲线） |
| `analysis/t3_bce_curves.csv` / `.png` | 训练 BCE 曲线 |
| `code/legacy_pre_fix/` | `ed70416` 的逐字节旧代码 + sha256 |
| `code/sfc_corrigendum_checks.py` | 6 组检查，同一文件跑旧/新两版 |
| `code/tests/test_sfc_corrigendum.py` | 19 项 pytest 回归 |

**禁止事项遵守**：未加载官方 valid/test；未读 `y` 以外标签；未改历史
checkpoint/config/result/report；未启动 A100/GPU/Slurm；未把合成可区分性
等同于真实任务性能；未用"近似 = 精确"的措辞。
