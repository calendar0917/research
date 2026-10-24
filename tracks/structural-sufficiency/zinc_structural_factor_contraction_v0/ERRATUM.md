# ERRATUM — SFC-v0 (`zinc_structural_factor_contraction_v0`)

基准提交 `ed70416`。本文件记录**已发表结果中的错误**、影响范围与修复。
旧 `REPORT.md` 与 `analysis/t1_correctness.json`、`t2_cost.json`、`t3_synth.json`
**未被修改**，保留为错误版本。修正版见 `REPORT_CORRECTED.md` 与
`analysis/*_corrected.json`。

本轮是 CORRIGENDUM / BUG-FIX REPLAY。未改架构、超参、合成任务、判定门槛、
数据与划分；未启动 A100；未加载官方 valid/test；未读取 `y` 以外的任何标签列
（T2 的 label-blocking reader 不变）。

---

## 0. 一句话

旧报告的核心负面结论（"SFC 连训练集都拟合不了"）是**评估期 bug 造成的假象**；
修复后 SFC 确实能拟合（reach 0.523→0.593，junction 0.500→0.775 训练集准确率），
但**仍然劣于同信息 pair/junction 对照 P**。因此**判定标签不变**，但**理由必须重写**。

---

## 1. Bug A — 评估期的 `T` 实际是 `logZ`

### 现象

`sfc_core.sfc_forward(..., with_marginals=False)` 把 `node_p` / `edge_p` 置零，
`_minus_bethe_free_energy` 于是返回 0，`T = logZ - 0 = logZ`。
`sfc_synth` 的训练路径用 `with_marginals=need_grad`（可训练臂为 `True`），
评估路径用 `with_marginals=False`，于是**同一个模型参数下训练特征与评估特征不同**。

### 实测（`analysis/t0_corrigendum_prefix.json`）

| 项 | 值 |
|---|---|
| 训练特征 vs 评估特征最大绝对差（小图，K=2） | **9.757** |
| 同一参数下 `T` 与 `logZ` 的差（ring4） | `T=0.000445` vs `logZ=7.527` |
| 评估期请求 `T` 是否被拒绝 | **否**（静默返回 `logZ`） |
| Srand 缓存 vs 训练期特征最大绝对差 | **151.66** |
| `no_grad` 下求精确边缘 | **崩溃**（`element 0 of tensors does not require grad`） |

`@torch.no_grad()` 下 `torch.autograd.grad` 直接不可用，所以旧代码的评估路径
不只是"值不对"，它连精确边缘都算不出来。

### 修复

* `SFCOutput.T` / `.minus_FB` 改为 **property**：未计算边缘时**抛 `RuntimeError`**，
  绝不返回替代值。
* `sfc_forward` 在 `with_marginals=True` 时，若检测到处于 `no_grad`，
  **局部 `torch.enable_grad()`** 只包住 `d logZ / d log` 这一步，
  返回前统一 `detach()`（`attach=False` 分支），不泄漏私有计算图。
* `sfc_synth._sfc_batch` 训练/评估**共用同一特征定义**（始终
  `with_marginals=True`）；`need_grad` 只决定是否保留到参数的反向路径，不改数值。
* 评估函数改为调用同一个 `_sfc_batch`。

---

## 2. Bug B — `T` 的自动微分错误

### 现象

`_minus_bethe_free_energy` 里 `log_q.detach()` 与 `log_K.detach()` 切断了显式势
的梯度。`F_B` 在**精确边缘 b\*(θ)** 处求值，因此对 θ 有两条路径：显式
(`log φ`, `log K`) 与隐式 (经 b\*)。detach 丢掉显式项后，`T` 的**数值**仍正确
（森林上恒为 0），但其**导数**与自身定义不符。

### 实测（`analysis/t0_corrigendum_prefix.json`）

| 检查 | 旧版 |
|---|---|
| 孤立节点 `\|T\|` | 0.0 |
| 孤立节点 `\|∇θT\|`（autograd） | **0.751**（有限差分为 0） |
| 单边树 `\|∇θT\|` | **0.613** |
| 随机树 n=9 `\|∇θT\|` | **3.244** |
| ring4 `dT/dA` 最大绝对误差 | **1.279**（全局相对 1526） |
| ring4 `dT/dS` 最大绝对误差 | **0.471**（全局相对 197） |
| 下游 BCE 对 A 的梯度误差 | **0.831**（全局相对 4.45） |
| K=8 ring4 `dT/dA` | **1.921**（全局相对 525） |
| `H` 的梯度（对照组） | 2.9e-10，正确 |

**只检查 `T≈0` 会漏掉这个 bug**——数值为零、梯度不为零。

### 修复

* 去掉两处 `.detach()`。
* `torch.autograd.grad(..., create_graph=attach)`：`attach` 为真时才建二阶图
  （`T` 对 θ 的依赖包含 `d b*/dθ`）；纯取值路径不建图。
* 修复后（`analysis/t0_corrigendum_postfix.json`）：

| 检查 | 修复后 |
|---|---|
| 森林 `\|∇θT\|`（孤立/单边/路径/随机树/双分量） | ≤ **2.8e-15**（FD ≤ 4.0e-9） |
| ring4 / ring5 / 方形带悬边 `dT/dθ` 误差 | ≤ **3.3e-9** |
| K=8 ring4 `dT/dA` 误差 | **1.0e-8** |
| 下游 BCE 梯度误差 | ≤ **4.6e-10** |
| 环上 `T` 非退化 | 0.000353 / 4.0e-7 / 0.00386，均 > 1e-8 |

---

## 3. Bug C — 冻结因子被就地覆写（本轮修复过程中发现）

不是任务书点名的两个 bug，但它**污染了 Srand 臂和 S/SP 的初始化**，必须记录。

`torch.as_tensor(x)` 在 `x` 已是 float64 时返回**同一个对象**。`sfc_fit_diagnostic`
里 `as_tensor(plan["A"]).requires_grad_(True)` + Adam 步，于是**原地改写**
`split["sfc_plan"]`；`Arm` 的 `nn.Parameter(...)`/`register_buffer(...)` 同样别名。

叠加三个次要缺陷：

1. `_PLAN_CACHE` 以 `id(g)` 为键 —— CPython 会复用 id，释放过的图可能命中别的图的
   消元计划（本轮实际触发过一次崩溃：`permute(sparse_coo)`）。
2. `Srand` 的 buffer 存 float32，而缓存特征是 float64 算的 —— 数值必然不同。
3. `assemble()` 用 `Generator(4242)` 生成缓存特征，而臂的 buffer 来自
   `make_sfc_plan(seed=0)`（`Generator(4000)`）—— **两份不同的参数**。

实测：`sfc_fit_diagnostic` 跑完后 `sfc_plan` 指纹
`7ab31f5a…` → `3d7496db…`，臂的初始参数与特征均不再逐位一致。

### 修复

`plan["A"]/["S"]` 全部 `.clone()`；`_PLAN_CACHE` 改用拓扑键
`(n, 边的端点序列)`；`Srand` buffer 用 float64；`assemble()` 让缓存特征与臂
**共用同一组张量**（`sfc_plan` 单一来源）；新增 `plan_fingerprint()` 与
`audit_sfc_feature_consistency()`，`main()` 在诊断后与全部臂跑完后各审计一次，
不一致即 `SystemExit`。

---

## 4. 检查矩阵（旧版必须红、修复版必须绿）

`code/sfc_corrigendum_checks.py`，同一文件对两个代码目录运行；
旧代码逐字节快照在 `code/legacy_pre_fix/`（sha256 记录在 JSON 里）。

| 检查组 | 旧版 `ed70416` | 修复版 |
|---|---|---|
| `feature_consistency`（训练=评估=缓存；`T` 必须被拒绝；`no_grad` 下边缘正确） | **FAIL** | PASS |
| `gradients`（`H`/`T`/下游 BCE/`T@K=8` 对 float64 中心差分） | **FAIL** | PASS |
| `forest_zero_gradient`（孤立点/单边树/路径/随机树/双分量：`T=0` **且** `∇T=0`） | **FAIL** | PASS |
| `cycle_nondegenerate`（ring4/ring5/方形带悬边：`T≠0` 且梯度一致） | **FAIL** | PASS |
| `srand_frozen_factors`（缓存=训练=评估=plan 参数，逐位） | **FAIL** | PASS |
| `shared_plan_immutability`（诊断不得改动冻结因子） | **FAIL** | PASS |
| **合计** | **0 / 6** | **6 / 6** |

`analysis/t0_corrigendum_prefix.json`、`analysis/t0_corrigendum_postfix.json`。
另有 19 项 pytest 回归：`code/tests/test_sfc_corrigendum.py`（全绿）。

T1 原有 12 项正确性检查（`logZ` 对稠密枚举、重编号、`marginals`、树零、
消元序、确定性、子图、饱和边界）在修复版上**仍全部通过**
（`analysis/t1_correctness_corrected.json`，12/12）——这些检查从未覆盖
"评估期 `T`"和"`T` 的梯度"，所以旧版当时也是 12/12 全绿。**这是旧 T1 门禁的
盲区，不是它算错了。**

---

## 5. 影响范围（哪些旧数字作废）

| 旧产物 | 是否作废 | 原因 |
|---|---|---|
| `t1_correctness.json` 12/12 | **不作废**，但**不充分** | 检查本身没错；缺 `T` 的梯度与评估一致性 |
| `t2_cost.json` 的宽度/结构统计 | **不作废** | 只依赖拓扑，与 θ 无关 |
| `t2_cost.json` 的耗时 | 量级不变 | 修复后 `C=8` 前向+反向 51.4 s → 49.5 s（同量级） |
| `t3_synth.json` 的 **C / P 两臂** | **不作废** | 不用 SFC 特征 |
| `t3_synth.json` 的 **S / SP 两臂** | **作废** | 训练/评估特征不一致 + Bug B |
| `t3_synth.json` 的 **Srand 臂** | **本来就缺失** | 旧 `arms` 里只有 C/P/S/SP，报告却引用了 Srand |
| 旧报告 "S 连训练集都拟合不了" | **作废** | 主要是 Bug A 的假象 |
| 旧报告 "SP 低于 P，SFC 干扰读出" | **作废** | 同上；修复后 SP≈P |
| 旧 `fit_diagnostics`（40 步） | **作废** | 就地覆写冻结因子 |

---

## 6. 修复版重放的**出处限制（必须与数字一起引用）**

`analysis/t3_synth_corrected.json` 是用 `sfc_core.py` sha `91eb9cd…` +
`sfc_synth.py` sha `0f50054…` 跑出来的。Bug C 的 `.clone()` 修复是在这次重放
**之后**才加的，所以：

* **C、P 两臂**：不涉及 SFC 参数，数字有效。
* **S、SP 两臂**：特征由自身参数实时计算，数值路径正确；但初始参数来自被
  `sfc_fit_diagnostic` 覆写过的 `sfc_plan`，即"一个随机初始化"而不是
  `make_sfc_plan(seed=0)` 的那一个。仍在协议内（协议只规定随机初始化），
  但**不可声称是可复现的固定初始化**。
* **Srand 臂**：缓存特征（`assemble()` 时算，原始 plan）与 buffer
  （`Arm` 构造时，已被覆写的 plan）**不是同一组参数**，所以
  `train 0.505 / unseen 0.506`（reach）与 `0.500 / 0.500`（junction）
  **不能作为有效的 Srand 结果引用**。

修复后的代码已能保证上述三条一致（`shared_plan_immutability` 与
`srand_frozen_factors` 均绿），但**没有重跑完整 T3**——按本轮指示在结论明确后
停止投入算力。若需要一份完全干净的 T3，命令是：

```
uv run python tracks/structural-sufficiency/zinc_structural_factor_contraction_v0/code/sfc_synth.py
```

（约 29 min CPU，无需 GPU。）这一步**不影响**判定标签，见下。

---

## 7. 成本记录（全部 CPU，无 A100）

| 阶段 | 时间 |
|---|---|
| 检查矩阵（旧 + 新） | 0.6 s + 1.6 s |
| T1 12 项 | 0.6 s |
| T2 全量 10k（修复版） | 212 s |
| 成本预演 `t0_cost_preflight.json` | 11 s |
| T3 完整重放（5 臂 × 2 seed × 2 任务） | 1740 s |
| pytest 19 项 | 1.8 s |

预演投影 1346 s，实测 1740 s（含 collate/读出/评估），同量级，未超预算。
峰值 RSS 1788 MB（其中 339 MB 是 pickle）。未启动任何 GPU / Slurm 任务。
