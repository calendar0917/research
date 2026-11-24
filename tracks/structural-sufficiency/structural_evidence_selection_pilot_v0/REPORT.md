# SES v0 — 非 MP 结构证据选择 pilot 报告

- 轮次：`structural_evidence_selection_pilot_v0`（SES v0）
- 数据：`ogbg-mollipo`，官方 split 3360/420/420，seed 0；Test 仅作 `DEVELOPMENT_TEST`
- 状态：**完成**。三臂正式训练、17 项 correctness checks、分析均已完成；本轮到此为止，不自动扩展 seed / 比例 / selector
- 正式训练代码：commit `565ad1b93b62`（checks 运行于 `e52ee9349f13`）
- 执行环境：`res` / `a100-2`，2×A100-SXM4-40GB；torch 2.5.1+cu124，CUDA 12.4，driver 550.163.01，Python 3.12.14

---

## 0. 一页结论

### 主表（best-Valid checkpoint；RANDOM 主数为 uniform expected）

| Arm | Predictor params | Selector params | Best epoch | Train RMSE | Valid RMSE | Test RMSE | Wall (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DENSE（全图） | 416,177 | 0 | 39 | 0.2489 | 0.6485 | 0.6815 | 185.7 |
| RANDOM（uniform expected） | 416,177 | 0 | 22 | 0.7410 | 1.0398 | 0.9837 | 341.7 |
| RANDOM（fixed index） | 416,177 | 0 | 22 | 0.7459 | 1.0520 | 0.9911 | 341.7 |
| SELECT（hard argmax） | 416,177 | 28,289 | 19 | 0.6184 | 0.9098 | 0.9143 | 374.0 |

**路线判断：`SELECTION_EFFECTIVE_COMPRESSION_LIMITING`**。  
学到的选择确实优于随机选择（SELECT−RANDOM：Valid −0.1300，Test −0.0695，95% CI 均不含 0），但 65% 节点 / 约 60% 键的证据压缩代价很大（SELECT−DENSE：Valid +0.2614，Test +0.2328，远大于 0.03 路线参考）。因此不是“没有选择价值”，也不是“已接近全图”，而是**选择有效、但压缩本身限制了预测性能**。

### 关键数字

| 项目 | Valid | Test |
|---|---:|---:|
| 选中候选节点覆盖率 mean（目标 0.65） | 0.6296 | 0.6310 |
| 选中候选边覆盖率 mean | 0.5949 | 0.5960 |
| 选中候选连通比例 | 420/420 | 420/420 |
| SELECT − DENSE RMSE（95% CI） | +0.2614 [+0.1918, +0.3371] | +0.2328 [+0.1701, +0.3045] |
| SELECT − RANDOM expected RMSE（95% CI） | −0.1300 [−0.1894, −0.0723] | −0.0695 [−0.1249, −0.0120] |
| RANDOM expected − DENSE RMSE（95% CI） | +0.3913 [+0.3150, +0.4674] | +0.3022 [+0.2311, +0.3785] |
| 独立 RANDOM 预测器在 SELECT 选择上的 RMSE（vs uniform） | 0.9889（1.0398），Δ −0.0509 | 0.9768（0.9837），Δ −0.0070 |
| SELECT 预测器 candidate oracle RMSE | 0.6654 | 0.6125 |
| RANDOM 预测器 candidate oracle RMSE | 0.6953 | 0.6655 |
| 资源 | 正式三臂 wall **11.0 min**；GPU·h **0.2504**；prepare 25.2 s | 硬上限 15 min 满足；10 min 目标略超 |

### 五问简答

1. **DENSE 在全图上是否足够强？** 可用但不算强上界：Valid 0.6485、Test 0.6815；Train 0.2489，存在明显过拟合/泛化间隙。它未触发 `DENSE_INSUFFICIENT` 的 >0.75 阈值，但 Test 明显差于历史 LCRC C1 的 0.5839（不同 recipe/horizon，仅作坐标）。
2. **65% 证据能否接近全图？** 不能。SELECT 比 DENSE 高 0.26/0.23 RMSE；RANDOM 高 0.39/0.30。压缩代价是主导效应。
3. **学习选择是否优于随机？** 是。SELECT 在 Valid/Test 都优于 RANDOM expected；也优于同一 SELECT 预测器上的 uniform 和 min-score 冻结选择。
4. **选择出的证据是否被独立预测器认可？** 部分认可：独立 RANDOM 预测器在 SELECT 选择上 Valid 改善 −0.0509（CI 不含 0），Test 仅 −0.0070（CI 含 0）。证据优势在 Test 上不稳健。
5. **候选 oracle 是否显示未实现空间？** 是，而且很大。SELECT/RANDOM 的 oracle 分别为 0.665/0.612 与 0.695/0.666，接近甚至局部优于 DENSE；实际选择只有 0.910/0.914 与 1.040/0.984。说明候选集合中存在未被选择机制实现的价值；但不能仅凭 oracle 差距判定唯一瓶颈。

---

## 1. 协议与执行

### 1.1 数据与冻结件

- 原始图缓存：`tracks/structural-query-learning/data/ogbg_mollipo_sq-pilot-cache-v1.npz`
  - sha256 `fe002287fa9c581278bc5430f5b4a5b26b35d9d4d5207d138f15956d93123c7c`
  - 4200 个分子，含官方 `train_idx/valid_idx/test_idx`（3360/420/420，互斥且覆盖全部行）
- 标签与 split 均从该缓存读取；本地 vault 对照 max |diff| = 2.29e-7。远端 checks 中 vault 不可用，因此以缓存 hash、split identity hash 和标签 hash 为准。
- 冻结 GWIC：
  - `gwic_model.py` sha256 `66926fac…`
  - `gwic_common.py` sha256 `6d081769…`
  - `gwic.json` sha256 `9ed13ab7…`
  - `GwicModel(use_words=False)`，416,177 参数；共享初始化 state sha256 `5ae62d08…`
- 训练集标签标准化：mean 2.1621607144740187，std 1.211539840098397；只由官方 Train 拟合。原子/键/图 scalar 的 normaliser 与 category support 同样只由官方 Train 完整图拟合。

### 1.2 候选与证据

- 每图 `k = max(1, min(n−1, floor(0.65·n)))`。
- 从每个原子出发，在固定、与标签无关的 canonical 原子顺序上做 BFS；取前 4 个唯一连通 induced support；第一个 canonical，余下按 greedy max-min Jaccard 选多样性。
- 候选只使用真实 induced 节点和键；不发明边；不做 motif 搜索；不把断开分量静默替换为全图。
- 候选内部 S1/S2/R2 只由候选自身的节点、键、路径构成；原始 9 类原子属性保留（可能携带原分子局部环境信息，这是已知限制）。
- 候选图 scalar 仅使用候选局部统计（候选节点数/边数、分量数、cycle rank 等），不加入原图节点数/边数、全图 descriptor、原图 R2 表示、选择分数、候选编号或原始节点索引。

共 16,775 个候选：Train 13,418、Valid 1,678、Test 1,679。4,175 个分子有 4 个候选，25 个分子只有 3 个。候选中位覆盖率接近目标；实际选中节点覆盖均值 0.6296/0.6310，边覆盖 0.5949/0.5960。

**覆盖例外**：Valid 行 1646 是一个 61 原子、6 个连通分量的分子；最大分量 39 原子，其他小分量均小于 k。受“每图最多 4 个候选”的限制，该分子得到 1 个 39-atom 候选和 3 个真实的 1-atom 候选；SELECT 最终选了其中一个 1-atom 候选，使 Valid 的 min node fraction = 0.0164、min edge fraction = 0。这是 420 个 Valid 分子中的 1 个；没有用全图替代，符合协议，但应在覆盖率解释中保留。

### 1.3 Horizon 选择

正式训练前只跑 preflight；preflight 模型全部丢弃，正式三臂从同一冻结初始化重新开始。按实测吞吐（含候选编码 + 三 split 全量 eval）估计：

| Horizon | 估计三臂 wall | 是否满足 10 min 目标 | 是否满足 15 min 硬上限 |
|---:|---:|---|---|
| 120 | 873.2 s（14.55 min） | 否 | 是（很接近） |
| 90 | 654.9 s（10.92 min） | 否 | 是 |
| 60 | 436.6 s（7.28 min） | 是 | 是 |

因此选择共同 **60 epochs**。该选择只依据吞吐/预算，未依据任何中间性能。实际正式训练每 epoch 比 preflight 慢约 18–27%（SELECT 6.23 s、RANDOM 5.70 s、DENSE 3.10 s），加上 DENSE 在首块 GPU 空闲后 119 s 才启动，最终三臂 wall 为 **660 s = 11.0 min**：略超 10 min 目标，但满足 15 min 硬上限。三臂 GPU·h 合计 **0.2504**，加 preflight/prepare 仍低于 0.5 GPU·h。

### 1.4 Correctness checks

`analysis/checks.json`：**17/17 PASS**。覆盖任务要求的 10 类：

- 官方 split/label/scaler 身份（C1–C3）
- 候选连通、induced、去重、k 上限与 below-k 记账（C4–C5）
- S1/S2/R2 只含选中结构（C6）
- 单候选 vs 批处理等价（C7，max diff 1.5e-8）
- 候选外原子/键修改不影响预测（C8）
- 其他候选修改不影响当前候选预测（C9）
- 重标记不变性，包括候选生成（C10）
- 候选局部重编号不变性（C11）
- SELECT eval 确为 argmax 单候选（C12，max diff 2.3e-8）
- predictor 与 selector 梯度有限且非零（C13）
- 三臂共享 predictor 初始化逐位一致（C14）
- 参数账本（C15）
- DENSE 与冻结 GWIC full-graph 前向一致（C16，max diff 1.9e-8）
- 无禁用预测器输入（C17）

C13 的 `predictor_abs_grad_sum=110.30`、`selector_abs_grad_sum=0.3949`，两者均非零且有限。C12 的独立 argmax 对比误差 2.26e-8，属于 float32 噪声。

---

## 2. 主结果与配对 bootstrap

### 2.1 三臂

- **DENSE**：Train 0.2489 / Valid 0.6485 / Test 0.6815；best epoch 39；wall 185.7 s。
- **RANDOM**：Train 0.7410 / Valid 1.0398 / Test 0.9837（uniform expected）；fixed index 为 0.7459/1.0520/0.9911；best epoch 22；wall 341.7 s。
- **SELECT**：Train 0.6184 / Valid 0.9098 / Test 0.9143（hard argmax）；best epoch 19；wall 374.0 s；selector 28,289 参数。

RANDOM 的主数使用 uniform expected；fixed index 只作“单次固定抽样”的参考，不用于 checkpoint 选择。SELECT 的 checkpoint 选择同样只看 Valid 的 argmax RMSE。

### 2.2 配对 molecule-level bootstrap（1000 次）

| 对比 | Split | Δ RMSE (A−B) | 95% CI | 方向 |
|---|---|---:|---|---|
| SELECT − DENSE | Valid | +0.2614 | [+0.1918, +0.3371] | SELECT 更差 |
| SELECT − DENSE | Test | +0.2328 | [+0.1701, +0.3045] | SELECT 更差 |
| SELECT − RANDOM expected | Valid | −0.1300 | [−0.1894, −0.0723] | SELECT 更好 |
| SELECT − RANDOM expected | Test | −0.0695 | [−0.1249, −0.0120] | SELECT 更好 |
| SELECT − RANDOM fixed | Valid | −0.1421 | [−0.2231, −0.0649] | SELECT 更好 |
| SELECT − RANDOM fixed | Test | −0.0768 | [−0.1409, −0.0065] | SELECT 更好 |
| RANDOM expected − DENSE | Valid | +0.3913 | [+0.3150, +0.4674] | RANDOM 更差 |
| RANDOM expected − DENSE | Test | +0.3022 | [+0.2311, +0.3785] | RANDOM 更差 |

所有主要 CI 均不含 0。注意这些 CI 是分子级 bootstrap，不是训练 seed 方差；本轮只有一个 seed。

### 2.3 收敛/预算限制

| Arm | Best epoch | Best Valid | Last Valid | Last−Best | Horizon-limited? |
|---|---:|---:|---:|---:|---|
| DENSE | 39 | 0.6485 | 0.6742 | +0.0257 | 否 |
| RANDOM | 22 | 1.0398 | 1.1419 | +0.1021 | 否 |
| SELECT | 19 | 0.9098 | 1.0042 | +0.0943 | 否 |

三臂的 best-Valid 都早于 60 epoch；候选两臂在 best 后明显过拟合（Train 继续下降，Valid 上升）。因此 60 epoch 并非“候选臂未收敛”造成；DENSE 是过拟合/recipe 限制，而非明显欠收敛。DENSE 的 Train–Valid 间隙（0.249 vs 0.648）也提示本轮 DENSE 不是调参上界。

---

## 3. 证据覆盖率

| 指标 | Valid | Test |
|---|---:|---:|
| 分子数 | 420 | 420 |
| 候选数 | 1,678 | 1,679 |
| 每分子候选数 mean / min / max | 3.995 / 3 / 4 | 3.998 / 3 / 4 |
| 选中节点覆盖率 mean / median | 0.6296 / 0.6333 | 0.6310 / 0.6333 |
| 选中节点覆盖率 p10 / p90 | 0.6111 / 0.6471 | 0.6111 / 0.6471 |
| 选中节点覆盖率 min | 0.0164（行 1646） | 0.5833 |
| 选中边覆盖率 mean / median | 0.5949 / 0.6000 | 0.5960 / 0.6000 |
| 选中边覆盖率 p10 / p90 | 0.5552 / 0.6296 | 0.5600 / 0.6316 |
| 选中边覆盖率 min | 0.0（行 1646） | 0.4545 |
| 选中候选连通比例 | 1.000 | 1.000 |
| 候选并集节点覆盖 mean | 0.9928 | 0.9940 |
| 两两 Jaccard mean | 0.5437 | 0.5433 |
| 两两交集/较小集合 mean | 0.6880 | 0.6887 |

解释：

- 目标 65% 节点实际落到约 63%；边覆盖约 60%。这是 induced subgraph 的正常结果（边界节点的部分键被切掉）。
- 候选并集覆盖约 99.3%，说明“从每个原子出发”的候选池几乎覆盖整图；但单个候选只覆盖约 63%。
- 两两 Jaccard 约 0.54，候选之间有明显重叠但并非简单复制；max-min 多样性确实产生了不同 support。
- 除行 1646 的多分量特例外，候选均连通、induced、无重复。该特例没有被静默替换为全图，符合“报告 below-k 覆盖”的要求。

---

## 4. 选择机制：干预、交叉评价、oracle

### 4.1 冻结选择的干预

在 SELECT 预测器上，比较三种选择规则：

| 规则 | Valid RMSE | Test RMSE |
|---|---:|---:|
| A：selector argmax | 0.9098 | 0.9143 |
| B：uniform expected | 1.0398 | 0.9837 |
| C：min selector score | 1.2192 | 1.0869 |

- A−B：Valid −0.1300 [−0.1894, −0.0723]；Test −0.0695 [−0.1249, −0.0120]。
- A−C：Valid −0.3094 [−0.3917, −0.2172]；Test −0.1726 [−0.2586, −0.0920]。

selector 的选择方向是有效的：它明显优于 uniform，也远好于把 score 反过来取最小。C 规则很差，说明 selector score 不是无关噪声，而是学到了与候选质量相关的排序。但 A 仍远逊于 DENSE。

### 4.2 独立 RANDOM 预测器交叉评价

把 SELECT 选中的候选 support 交给独立训练的 RANDOM 预测器（该预测器没有见过 SELECT 的 selector），只做前向：

| 指标 | Valid | Test |
|---|---:|---:|
| RANDOM 预测器在 SELECT 选择上的 RMSE | 0.9889 | 0.9768 |
| RANDOM 预测器 uniform expected RMSE | 1.0398 | 0.9837 |
| Δ（选择 − uniform） | −0.0509 [−0.1032, −0.0015] | −0.0070 [−0.0515, +0.0382] |
| 分子比例：cross 优于 uniform | 0.6095 | 0.5643 |
| SELECT 预测器在自己选择上的 RMSE | 0.9098 | 0.9143 |

Valid 上独立预测器也认可 SELECT 的选择（CI 刚过 0），说明选中的 support 有一定内在证据优势；Test 上这一优势缩小到 0.007 且 CI 含 0，不能认为稳健。SELECT 相对 cross 的额外优势（Valid 0.9889→0.9098，Test 0.9768→0.9143）说明 SELECT 的收益中有相当一部分来自 selector 与预测器的联合适配，而不是单纯“选中了客观上更好的子图”。

### 4.3 候选 oracle（post-hoc，仅诊断）

oracle 使用标签，在每个分子的候选预测中取误差最小者；它不是可部署的选择器，只提供候选集合的未实现上限。

| 预测器 | Valid actual | Valid oracle | Valid MSE gap | Test actual | Test oracle | Test MSE gap |
|---|---:|---:|---:|---:|---:|---:|
| SELECT | 0.9098 | 0.6654 | 0.3850 | 0.9143 | 0.6125 | 0.4608 |
| RANDOM | 1.0398 | 0.6953 | 0.5978 | 0.9837 | 0.6655 | 0.5248 |

- SELECT oracle 在 Valid 上 RMSE 0.6654，仅比 DENSE 0.6485 高 0.017；Test oracle 0.6125，反而优于 DENSE 0.6815。
- RANDOM oracle 在 Valid/Test 为 0.6953/0.6655，也接近 DENSE。
- oracle 相对 actual 的 MSE 改善很大（SELECT 0.385/0.461，RANDOM 0.598/0.525），说明候选集合中存在大量未被当前 selector 实现的价值。
- 但 oracle 是标签后验选择，差距不能直接解释为“selector 可轻易修复”；它也可能反映候选预测方差大、selector 泛化不足、候选分布与全图分布差异，以及 selector/预测器联合过拟合。不能仅凭 oracle 判定唯一瓶颈。

---

## 5. 六个 Valid 例子

`analysis/examples.json` 固定给出 6 个 Valid 分子，每类 2 个：

1. **准确且选择有效**（行 989、1819）：选中候选预测接近标签，其他候选明显偏离。例如行 989：y=−0.84，选中候选 pred=−0.279，其余候选 pred=2.37/2.80/2.64；说明不同 support 的预测差异很大，selector 选到了更合理的一个。
2. **选择收益不明显**（行 1856、1558）：多个候选预测都接近标签，argmax 与 uniform 差异小。
3. **选择预测失败**（行 1601、1647）：所有候选都预测偏高（约 2.5–3.1），而标签为 −0.85/−0.51；selector 只能在坏候选中选“相对不那么坏”的一个，误差仍 >3。

这些例子用于定性检查，不作为额外性能估计。

---

## 6. 资源与执行 regime

| 项目 | 实测 |
|---|---:|
| 候选预计算（远端） | 25.17 s（<5 min 限制） |
| 预计算产物 | 16,775 候选；本地 193.6 MB（`results/prepared/`，未提交） |
| Preflight | 10.0 s |
| 正式三臂 GPU·h 合计 | 0.2504 GPU·h |
| 正式三臂 wall（首启→末完成） | 660 s = 11.0 min |
| 单臂 wall | DENSE 185.7 s；RANDOM 341.7 s；SELECT 374.0 s |
| 硬上限 15 min | 满足 |
| 目标 8–10 min | 略超（11.0 min） |

目标略超的原因是可解释的工程因素：实际 epoch 时间比 preflight 高约 18–27%，且 DENSE 在首块 GPU 空闲后 119 s 才启动（agent 轮询延迟），不是模型规模或候选编码超预算。GPU·h 仍远低于 0.5 上限。

执行 regime（`analysis/runs.json`）：

- `ses-v0-select-20261124-175654-c9060aaa`，`res-gpu0`，process backend，commit `565ad1b93b62`
- `ses-v0-random-20261124-175654-16f02f72`，`res-gpu1`，process backend，commit `565ad1b93b62`
- `ses-v0-dense-20261124-180440-b71be877`，`res-gpu1`（RANDOM 释放后），process backend，commit `565ad1b93b62`
- 节点 `a100-2`；driver 550.163.01；torch 2.5.1+cu124；CUDA 12.4；Python 3.12.14

---

## 7. 路线判断与解释边界

### 7.1 判断

**`SELECTION_EFFECTIVE_COMPRESSION_LIMITING`**

- DENSE 未触发“自身失效”阈值（Valid/Test 均 <0.75），但绝对性能偏弱，且 Test 0.6815 比历史 LCRC C1 0.5839 差；本轮 DENSE 不是调参上界。
- SELECT 相对 RANDOM expected 在 Valid/Test 均改善，CI 不含 0；selector 也优于 uniform 和 min-score。因此“学习选择”不是无效的。
- 但 SELECT 相对 DENSE 的差距为 +0.2614/+0.2328，是 0.03 路线参考的约 8–9 倍。65% 节点、60% 键的压缩代价占主导。
- 独立 RANDOM 预测器交叉评价只在 Valid 上给出弱正证据，Test 不显著；选择收益有预测器联合适配成分。
- candidate oracle 显示候选空间还有很大未实现价值，说明“压缩上限”不等于“候选集合没有信息”；但 oracle 不是可部署结果，不能据此宣称只差一个更好的 selector。

### 7.2 与历史坐标的关系

仅作坐标，不重训历史模型：

- LCRC 官方 C1：Valid 0.6526 / Test 0.5839（官方 Test；不同 recipe/horizon）
- GINE D0：0.5266（内部 dev，非官方 Test）
- R2 D0：0.5506（内部 dev，非官方 Test）

本轮 DENSE 0.6485/0.6815：Valid 与 LCRC C1 接近，Test 明显更差；候选两臂更差。由于 DENSE 本身不是强上界，本报告的核心结论是**同一 recipe 下的相对比较**：即使给候选臂一个不算强的全图坐标，65% 证据仍然差 0.23–0.26 RMSE；若换成更强的 DENSE，压缩差距未必会缩小。绝对数值不应直接与历史调参结果做优劣判断。

### 7.3 限制

- 单 seed；bootstrap 只反映分子抽样，不反映训练 seed 方差。
- DENSE 的 Train–Valid 间隙大，说明本轮 recipe 下全图模型也过拟合；DENSE 绝对坐标偏弱。
- 65% 不是极端稀疏；结论不能外推到更低/更高覆盖率。
- 原始 9 类原子属性可能携带原分子局部环境信息；本轮的“结构证据”不是纯净的因果关系。
- 预测不变性 checks 只证明预测合同，不证明选择行为不受未选信息影响；不做因果声明。
- oracle 使用标签后验，不能作为部署性能或唯一瓶颈证明。
- 未做多 seed、未扫比例、未改 selector 结构，也未做 motif 搜索、梯度 explainer 或真实分子编辑；本轮按要求停止。

---

## 8. 产物

- `configs/ses.json`：冻结协议
- `PROTOCOL.md`：协议与检查说明
- `code/`：候选生成、张量化、训练、preflight、checks、分析
- `analysis/checks.json`：17/17 PASS
- `analysis/preflight.json`：horizon 决策与吞吐
- `analysis/runs.json`：三臂运行元数据/regime
- `analysis/candidate_stats.json`、`analysis/prepare_manifest.json`
- `analysis/evidence_analysis.json`：主表、bootstrap、覆盖、干预、交叉、oracle、资源、路线
- `analysis/examples.json`：6 个 Valid 例子
- `results/{dense,random,select}/`：`best.pt`、`curve.csv`、`manifest.json`、`predictions.npz`、`train.log`
- `results/main_table.csv`、`results/main_table_full.csv`
- `results/per_molecule_candidates.npz`：逐分子/逐候选预测、score、support 节点/键、fixed index、expected/fixed/select 预测
- `results/prepared/ses_prepared.pkl`：本地 193.6 MB，未提交；可用 `code/ses_prepare.py` 重建

复现顺序（远端）：

```bash
python code/ses_prepare.py
python code/ses_preflight.py
python code/ses_checks.py
python code/ses_train.py --arm SELECT --epochs 60 --out-dir results/select
python code/ses_train.py --arm RANDOM --epochs 60 --out-dir results/random
python code/ses_train.py --arm DENSE  --epochs 60 --out-dir results/dense
python code/ses_analyze.py
```

本轮结束。
