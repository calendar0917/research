# Frozen function-basis accessibility audit

> 问题：在 compact-v4-hinge 已经产生的同一个 frozen graph representation
> `R ∈ R^302` 上，**只改变最终函数的参数化方式**，能否让 MAE/L1 regression 更稳定、
> 更 sample-efficient？
>
> 模块：`experiments/luyin16/zinc_function_basis_accessibility.py`
> 结果：`results/function_basis_accessibility/`
> 测试：`tests/test_function_basis_accessibility.py`（13 tests，全过）
> 结论：**CLEAR NO-GO**（Case C）。显式 target-independent quantile-hinge basis
> 没有显示比同预算 ReLU MLP 更好的 sample efficiency。

---

## 1. Motivation

连续三条「missing local statistic」路线都 NO-GO：

- frozen richer readout witness → NO-GO；
- pair endpoint association witness → NO-GO；
- centre-incidence covariance / co-occurrence witness → NO-GO。

这些结果共同削弱了假设：

> 「当前模型主要是因为某一层丢掉了一个明确的局部统计量。」

于是转向 **information accessibility / statistical learnability**：

> 信息可能已经存在，但 generic MLP 不一定能在 10k ZINC、小参数预算、L1 objective 下
> 高效地学出任务需要的函数形状。

本阶段只隔离这一个变量：**same representation + same L1 objective，仅改变 basis。**

## 2. Why missing-statistic hunting was stopped

三次 witness 诊断都没有找到「被丢弃且 task-relevant」的统计量：pair endpoint
association 的 pooled ΔR = −0.006994（0/5），centre covariance 的 ΔM = −0.001361
（2/5），并且都输给 matched marginal / R-only control。继续叠加更精细的局部统计量
（full covariance / attention / relation-set Transformer）在 evidence 上是递减收益，
且每次都要付出新 representation 与 target audit 的双重 confound。因此升级到
representation-holder 之外的问题：**函数族是否可学**。

## 3. Function basis vs representation vs loss

三条轴必须分开：

- **Representation** 回答「有哪些信息可用」（`R` 的坐标与通道）。
- **Function basis** 回答「哪些响应形状容易表达、容易估计」。
- **Loss** 回答「训练更愿意压低哪类误差」。

本实验只动第二条：representation 固定为紧凑 v4 已导出的 `R`；loss 固定为 L1/MAE；
optimizer、数据顺序、checkpoint selection 全部沿用 frozen adapter convention。
唯一变化是 `delta(R)` 的参数化：generic ReLU MLP vs 显式 piecewise-linear hinge basis。

## 4. Relation to compact-v4 hinge

从 repo 重新核实 v4 global topology 通道：

| mode | 内容 | dim | valid MAE (seed 0) | Δ vs v2 |
|---|---|---:|---:|---:|
| v2 / none | — | 0 | 0.184158 | — |
| longest | `[L, cycle_rank]` | 2 | 0.185681 | −0.001523 |
| spectrum | `[L, n3..n10, n_gt10, mcb_count, mcb_max, mcb_mean, mcb_total, cycle_rank]` | 15 | 0.187674 | −0.003516 |
| **hinge** | spectrum + `[L, L², ReLU(L−3)…ReLU(L−10)]` | 25 | **0.170066** | **+0.014093** |
| capacity_control | 15 个全 0（等容量） | 15 | 0.184253 | −0.000095 |

要点：

- **spectrum 已经包含 scalar `L`**（longest simple cycle）以及各长度精确计数；
  hinge 只是再加上 `L²` 与 `ReLU(L−t)` 阶梯。
- hinge 的这些 feature 没有提供新的 graph identity information（同一 molecule
  的 `L`、counts 已经在 spectrum 里），它们主要把「closure scale 到一定程度后
  才重要」的**单调饱和形状**变成显式可表达/可估计的函数基。
- capacity_control ≈ 0 排除「纯容量」解释；shuffle 全消证明增益依赖真实拓扑。
- longest / spectrum 单独都 NO-GO（raw counts 甚至有害），说明有效的是**形状**而非
  更多 raw 维度。

因此 v4-hinge 的成功**部分**可能是「把难学的函数变成更容易学的函数基」。
这只是本实验的 hypothesis 来源，**不是结论**：v4 topology pathway 仍有 feature set、
parameterization、target-audit 启发、preprocessing 等差异（见 §15）。

## 5. Relation to MAE and boosted trees

导师曾提出 MAE 型 regression 未必最适合 XGBoost。本阶段**不直接做
XGBoost vs neural network**，因为 XGBoost 同时改变 function family、optimization、
regularization、feature selection、partition structure 和 absolute-error training 的
实现；赢了或输了都难以归因。先隔离最基本的问题：

> same representation + same L1 objective，仅改变 basis，是否产生稳定差异？

MAE-compatible boosted-tree family 留给「先证明 function family 本身值得研究」之后的
专门实验（`MAE-Compatible Function Family Comparison`）。

## 6. Frozen OOF protocol

- 只使用 official train universe；official valid/test **从未加载**（Test 12 + gate）。
- 复用已有 5-fold OOF compact-v4-hinge checkpoints（backbone seeds 0、1 均已存在），
  本阶段**不训练 backbone**。
- frozen representation `R` 与 frozen baseline `yhat_0` 直接来自 corrected
  `frozen_state_export_v4_centre_incidence`（同一 frozen forward；`G1` 验证
  `yhat_0 == oof_prediction` max diff `0.0`）。
- split 直接复用 frozen-readout / pair-witness / centre-witness 的
  `fold_split_manifest.json`：outer-held-out 2000 → 1200 adapter-fit / 400
  adapter-selection / 400 adapter-evaluation（molecule-ID hash，seed
  `frozen-readout-sufficiency-v2-20260910`）。本阶段的 manifest 与其
  `assignment_sha256` 逐 fold 交叉核对一致。
- 预算采用 staged policy：Stage 0（integrity）→ Stage 1（seed0 × 5 folds × 1 init）
  → 只有 advance 才 Stage 1b（第二 init）/ Stage 2（第二 backbone）。

### Gate 0 / representation integrity（5/5 folds, seed 0）

| gate | 内容 | 结果 |
|---|---|---|
| G1 | stored `yhat_0` == frozen OOF fold prediction | max 0.0 |
| G2 | stored target == official-train label `y_stored` | max 0.0 |
| G3 | `subset_index` == outer holdout | pass |
| G4 | offline reconstructed `R` == exported `R` | max 1.1e-5 … 9.2e-5 |
| G5 | re-feed stored `R` through frozen head == `yhat_0` | max 0.0 |
| G6 | re-feed reconstructed `R` through frozen head == `yhat_0` | max 9.5e-7 … 1.9e-6 |

`R_dim = 302`，布局 `unary 97 + pair-bucket moments 165 + global 32 + topology 8`。
frozen head 由 checkpoint state dict 直接重建（`Linear(302,64) LayerNorm ReLU Dropout
Linear(64,32) ReLU Linear(32,1)`），state-dict load + 零误差 prediction gate 验证架构。

## 7. Standardization

所有 trainable readers 使用**完全相同**的 fit-only coordinate standardisation：

```
mean_j, std_j 只由该 fold 的 adapter-fit 1200 molecules 计算
z_j = (R_j - mean_j) / max(std_j, eps),  eps = 1e-6
若 std_j < eps: z_j = 0（degenerate coordinate）
```

selection / evaluation **只使用** fit statistics，禁止重新估计。degenerate
（fit-constant）坐标每 fold 18–94 个（302 中）。目的：避免 basis model 因尺度处理更好
而获得不公平优势。该预处理 deterministic、几乎可逆，不引入任何 target information。

## 8. Generic MLP controls

- **B0**：`yhat_B0 = yhat_0`（不训练）。
- **B1**：`302 → 4 → 2 → 1`，ReLU，**1225 params**，与 E 参数匹配。
- **B2**：`302 → 13 → 13 → 1`，ReLU，**4135 params**（historical-strength，非匹配）。
- **B-linear**：`302 → 1`，303 params，descriptive only，不进入 gate。
- 所有 reader 都做 residual correction：`yhat = yhat_0 + delta(z)`。
- 统一 L1/MAE objective、Adam(lr=1e-3)、full-batch、selection checkpoint、
  deterministic seed、固定 horizon。无 LR / weight decay / optimizer sweep。

## 9. Quantile-hinge basis

对每个标准化坐标 `z_j`，只用 adapter-fit 输入分布计算
`t_j,25 / t_j,50 / t_j,75`（不看 y / residual / baseline error / valid / evaluation）。
basis：

```
φ_j(z_j) = [ z_j, ReLU(z_j - t_j,25), ReLU(z_j - t_j,50), ReLU(z_j - t_j,75) ]
```

`302 × 4 = 1208D`。E 是 additive linear reader，`1208 → 1`，**1209 params**。

- 未使用任何 target-informed knot（无 cycle threshold 6、无 residual-selected knot、
  无 decision-tree split）。
- 重复 knot 保留重复 column，保持固定维度与确定参数数。
- degenerate coordinate：`z_j = 0`，所有 hinge 为 0。

## 10. Parameter accounting

| reader | architecture | params |
|---|---|---:|
| B1 | `302 → 4 → 2 → 1` | 1225 |
| E | `1208 → 1` | 1209 |
| B2 | `302 → 13 → 13 → 1` | 4135 |
| B-linear | `302 → 1` | 303 |

B1/E mismatch = `+1.32%` ≤ 3%。B2 刻意不做 parameter match（~3.4× E），回答
「更大的 generic MLP 能否明显超过显式 basis」。

## 11. Stage 1 results

backbone seed 0 × 5 folds × 1 init，固定 horizon 400。fold0/seed0 convergence trace：
selection MAE 在 100 epoch 附近最低，400→800 对 B1/B2/E 均**变差**（无 reader 仍在
400→800 改善 >1e-4），故 H=400，best-selection checkpoint。

Adapter-evaluation MAE（mean over folds）：

| B0 | B1 (1225) | B2 (4135) | B-linear (303) | E (1209) | E no-hinge |
|---:|---:|---:|---:|---:|---:|
| 0.193160 | **0.187815** | 0.190508 | 0.203257 | 0.203204 | 0.250995 |

| fold | B0 | B1 | B2 | E | Δsmall = B1−E | Δstrong = B2−E |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.224365 | 0.216585 | 0.217165 | 0.234422 | −0.017836 | −0.017257 |
| 1 | 0.159687 | 0.145999 | 0.151477 | 0.157910 | −0.011911 | −0.006433 |
| 2 | 0.263058 | 0.262321 | 0.267341 | 0.285166 | −0.022845 | −0.017825 |
| 3 | 0.153514 | 0.149360 | 0.153222 | 0.164665 | −0.015305 | −0.011444 |
| 4 | 0.165176 | 0.164809 | 0.163337 | 0.173859 | −0.009050 | −0.010523 |
| **mean** | 0.193160 | 0.187815 | 0.190508 | 0.203204 | **−0.015389** | **−0.012696** |

- `mean Δsmall = −0.015389`（median −0.015305），**0/5** folds positive；
  stratified paired bootstrap 95% CI `[−0.01979, −0.01093]`，`P(>0)=0`。
- `mean Δstrong = −0.012696`（median −0.011444），**0/5** positive；
  `mean(MAE_E − MAE_B2) = +0.012696`。
- `mean(MAE_E − MAE_B0) = +0.010044`：additive hinge reader 比它要修正的 frozen
  baseline 还差。
- B1 vs B0 `mean +0.005345`（5/5）；B2 vs B0 `mean +0.002651`（5/5）。
- **Bulk safety**：`mean(MAE_E − MAE_B1) = +0.018408`，max `+0.021165` > `+0.002`
  gate，FAIL —— E 在 common-input 子群上也退化，不是只牺牲少数 unusual molecules。
- **Hinge 并非失效**：E 内部把 hinge 系数置 0（frozen inference ablation）后
  evaluation MAE 上升 `mean +0.047791`（95% CI `[+0.03975, +0.05587]`，5/5）。
  显式 basis 确实在表达 L1-relevant shape，只是整体上与同预算 generic MLP 竞争失败。

预注册 Stage 1 判据：`mean Δsmall ≤ +0.0005` **或** `≤2/5 folds E 优于 B1`
→ **CLEAR NO-GO**。二者都满足（−0.0154；0/5）。

## 12. Replication decision

CLEAR NO-GO ⇒ **STOP**。不花：

- 第二 backbone seed；
- Stage 1b 第二 reader init；
- 更多 knot、learned knot、B-spline、polynomial、RBF、Fourier、KSVD response dictionary；
- XGBoost comparison。

阶段结论严格限定为：**generic coordinatewise quantile-hinge basis 没有显示比同预算
ReLU MLP 更好的 sample efficiency**。不写「function parameterization 不重要」。

## 13. Stage 2 if run

未运行（NO-GO）。若曾 advance，计划是：second existing OOF backbone seed × same 5
folds，完全冻结 standardization rule / quantile definition / knots / basis definition /
widths / loss / optimizer / horizon / split / thresholds；每 fold/backbone 的 knot 仍只从
该 fold 的 adapter-fit input 计算。Final GO 判据（未使用）：pooled mean Δsmall ≥
+0.0025、两 backbone mean Δsmall > 0、≥4/5 fold-averaged positive、paired bootstrap
95% CI lower > 0。

## 14. Mechanism analysis if GO

未运行（NO-GO）。计划中的 mechanism analysis（coefficient summary、hinge ablation、
fixed-knot control）只在 advance 后执行。本轮已顺带记录（训练循环内即得、非额外预算）
的唯一 descriptive 数字是 `Δhinge = +0.047791`（§11），显示 hinge terms 在 E 内部
有效，但不足以让 E 超过 B1。

## 15. What is and is not proven

**Proven（在本 protocol 范围内）：**

- 同一 frozen `R`、同一 L1 objective、同一训练框架下，参数匹配（1225 vs 1209）时
  `302→4→2→1` generic ReLU MLP **稳定优于** additive quantile-hinge basis reader，
  5/5 folds，bootstrap CI 完全为负。
- E 也劣于更强的 `302→13→13→1`（4135）和 frozen baseline B0。
- 显式 hinge basis 本身不是无效的：E 内部去掉 hinge 显著变差（+0.0478）。
- v4 spectrum 已包含 scalar `L`；hinge 额外加入的是 `L²` 与 `ReLU(L−t)` 阶梯，
  即主要是**函数形状/参数化**，而非新的 graph identity 信息。capacity_control
  排除纯容量解释。

**Not proven：**

- 不能说「function parameterisation 不重要」；本轮只关闭**简单 coordinatewise
  additive hinge basis expansion**。
- 不能倒推「compact-v4 backbone 已经最优」；本轮只是说在 frozen `R` 上，
  generic depth 仍能学到 additive basis 学不到的东西（很可能是 coordinate
  interaction）。
- 不能断言「v4 的全部收益来自 function basis」；v4 topology pathway 仍有 feature set、
  parameterization、target-audit 启发、preprocessing 差异。本实验只是提供了
  一个更干净的独立 test，且结果是**不支持**显式 additive basis 的。
- 不能据此直接跳到 XGBoost/KSVD/learned dictionary；这些属于另一个实验。

## 16. Final verdict

**CLEAR NO-GO — Case C（SIMPLE QUANTILE-HINGE BASIS NO-GO）。**

下一步（按预注册）：关闭 frozen-R basis engineering（固定 hinge dictionary、
learned knot、B-spline、polynomial、RBF、Fourier、KSVD response dictionary），
研究决策升级到：

> 是否需要新的 representation family / structured computation，
> 而不是继续对 frozen `R` 做函数变换。

记录：`records/decisions/decision-function-basis-accessibility-nogo-20260912.yaml`，
`records/claims/claim-function-basis-accessibility-nogo-20260912.yaml`。

---

## Q1–Q16

- **Q1** frozen `R` 真实维度与组成？`R ∈ R^302`，`unary 97 + pair-bucket moments 165 +
  global 32 + topology 8`。见 `representation_integrity.json`、`basis_spec.json`。
- **Q2** v4 spectrum vs hinge 是新信息还是参数化差异？spectrum 已含 scalar `L` 与各
  长度精确计数；hinge 只额外加入 `L²` 与 `ReLU(L−3..10)` 阶梯，没有新增 graph
  identity information，主要是 function basis / parameterization 差异。
  capacity_control ≈ 0、shuffle 全消支持该判断。
- **Q3** B1 参数量？**1225**。
- **Q4** B2 参数量？**4135**。
- **Q5** E 参数量？**1209**（basis 1208 + bias）。
- **Q6** B0 MAE？mean **0.193160**。
- **Q7** B1 MAE？mean **0.187815**。
- **Q8** B2 MAE？mean **0.190508**。
- **Q9** E MAE？mean **0.203204**。
- **Q10** mean Δsmall？**−0.015389**（0/5 positive，95% CI `[−0.01979, −0.01093]`）。
- **Q11** E 相对 B2？`mean Δstrong = −0.012696`（0/5），即 E 差 +0.012696。
- **Q12** fold 方向是否一致？一致：Δsmall 与 Δstrong 均 5/5 为负。
- **Q13** 是否需要第二 backbone？**否**（CLEAR NO-GO）。
- **Q14** second backbone 是否复现？未运行。
- **Q15** 收益若存在来自 linear 还是 hinge？本轮无整体收益；但在 E 内部，去掉 hinge 会
  显著变差（`Δhinge = +0.047791`，5/5），说明 hinge terms 承担了 E 的大部分表达。
- **Q16** 最终属于？**NO-GO**（STRONG GO / EFFICIENCY GO / NO-GO / INCONCLUSIVE 中）。
