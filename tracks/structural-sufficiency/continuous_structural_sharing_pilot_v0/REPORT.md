# CSS v0 — Continuous Structural Sharing pilot 报告

- 轮次：`structural-sufficiency/continuous_structural_sharing_pilot_v0`（CSS v0）
- 数据：`ogbg-mollipo`，官方 split 3360/420/420，seed 0；Test 仅作 `DEVELOPMENT_TEST`
- 状态：**完成**。4 个正式训练单元、8 项 correctness/degeneracy checks、主表、bootstrap、机制分析均已完成；本轮到此为止，不扩展 seed / K / tau / SVD 维度 / 数据比例
- 冻结协议：`PROTOCOL.md`，protocol id `css-v0-continuous-structural-sharing-pilot`
- 执行：`res` / `a100-2`，2×A100-SXM4-40GB；torch 2.5.1+cu124，CUDA 12.4，driver 550.163.01
- 代码 commit：实现 + checks `00ab1d35`；远端 prep + checks `2b8c5865`；horizon 与四个正式 run `05e4c0c8`；结果快照 `7499c3c6`（run manifest 记录训练 commit）

---

## 0. 一页结论

### 四个问题的直接回答

1. **全量数据上有提升吗？没有。** `SHARE_100 − DIRECT_100`：Valid **−0.0266**（SHARE 更差），
   Test **+0.0136**（SHARE 更好，95% bootstrap CI `[−0.027, +0.054]`）。方向相反，幅度都远小于
   0.02 的路线参考值。全量数据上唯一稳定的现象是 **Valid/Test 符号冲突**：从 epoch ≈50 起
   SHARE 的 Test 一直低约 0.02，但 Valid 一直高约 0.03。
2. **半量数据上有提升吗？没有可用提升。** `SHARE_50 − DIRECT_50`：Valid **+0.0052**
   （CI `[−0.027, +0.036]`），Test **+0.0128**（CI `[−0.028, +0.050]`）。方向一致但都低于 0.02；
   且这个微弱优势依赖“best-Valid”选点：SHARE_50 在 epoch 32 取得 Valid 最低点后明显变差，
   而 DIRECT_50 到 epoch 48 才达到最低点（同 epoch 比较时 SHARE 并不占优）。
   `gain_50 − gain_100`：Valid **+0.0317**，Test **−0.0008**。
3. **有实际竞争力吗？没有达到本 track 的最好水平。** 本轮两个 100% 模型
   （Valid/Test `0.607/0.667` 与 `0.633/0.653`）略好于历史 SES DENSE（`0.649/0.682`，
   416,177 参数，read-only 参考），但明显差于历史 LCRC-C1（Test ≈`0.584`）和 PNA-native
   （`0.554/0.529`，7.77M 参数）。这些是不同 recipe / horizon 的历史参考，不能当作匹配对照。
4. **值得继续吗？不建议按当前机制直接扩展。** 现象更像“固定参考相似度分支是一个正则化瓶颈”，
   而不是“真实结构共享带来泛化收益”。若要继续，最小的下一步不是加 seed 或调 tau，而是加一个
   **同维度、同参数量的控制分支**（例如固定随机投影或对 `z` 的可学习 64→32 映射），
   先证明收益来自“真实参考结构”而不是“额外的 32 维非线性特征”。

### 决策标签

**`D SPLIT_CONFLICT_OR_BUDGET_LIMITED`** —— 触发原因是 **split conflict，不是 budget**：
100% 单元 Valid 与 Test 方向相反；50% 单元方向一致但幅度不足 0.02。预算侧完全达标
（120/120 epoch，四 run 合计 0.369 GPU·h，单 run 最长 355 s，均 < 10 min 目标）。
实质结论与 C 相同：**没有达到 0.02 参考值的有用共享增益**；标 D 是为了保留“全量 Test
小幅更好、Valid 明显更差”这一方向冲突，而不是宣称存在未决的“promising”信号。

### 主表（best-Valid checkpoint；原始标签单位）

| Unit | Params | Fit n | Best epoch | Fit RMSE | Heldout RMSE | Valid RMSE | Valid MAE | Test RMSE | Test MAE | Wall (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DIRECT_100 | 424,401 | 3360 | 82 | 0.1076 | — | **0.6065** | 0.4554 | 0.6670 | 0.4944 | 354.9 |
| SHARE_100 | 424,401 | 3360 | 58 | 0.1562 | — | 0.6331 | 0.4723 | **0.6535** | 0.4916 | 345.1 |
| DIRECT_50 | 424,401 | 1680 | 48 | 0.2432 | 0.6238 | 0.7050 | 0.5350 | 0.7386 | 0.5636 | 280.9 |
| SHARE_50 | 424,401 | 1680 | 32 | 0.3769 | 0.6335 | **0.6998** | 0.5325 | **0.7259** | 0.5771 | 339.1 |

`gain = RMSE(DIRECT) − RMSE(SHARE)`，正数表示 SHARE 更好：

| Comparison | Valid gain (95% paired bootstrap CI) | Test gain (95% CI) | frac. molecules > 0 (Valid / Test) |
|---|---:|---:|---:|
| 100% | **−0.0266** `[−0.0707, +0.0186]` | **+0.0136** `[−0.0272, +0.0543]` | 0.135 / 0.715 |
| 50% | **+0.0052** `[−0.0268, +0.0362]` | **+0.0128** `[−0.0282, +0.0503]` | 0.638 / 0.747 |

1000× molecule-level paired bootstrap，seed 0，420 个分子；所有区间都跨 0。

---

## 1. 设计与执行（frozen）

- **结构描述子**：每个真实 root 的完整 radius-2 ego graph，5 个稀疏 block
  （ROOT、S1、S2、EGO_ATOM、EGO_BOND），真实重数，`sqrt(count)` → block L2 → concat / `sqrt(5)`。
  每个训练规模用 **Fit only** 建词表 + `TruncatedSVD(64, n_iter=5, random_state=0)`，再逐 root
  L2 归一化得到 `z_v ∈ R^64`。100%：319 维词表、EVR 0.9853；50%：292 维、EVR 0.9855。
  无 padding 维度（64/64），Fit 外 OOV：100% Valid `3.6e-5`、Test `2.4e-5`；
  50% Valid `7.7e-5`、Test `6.5e-5`。
- **真实参考结构**：每个规模仅从 Fit roots 中用 seed-0 k-means++ D² 采样 64 个真实 root，
  不做 Lloyd、不合成中心、不重复投影向量。`tau = max(1e-6, median_Fit(d_8²))`
  （100%：0.40834；50%：0.40423），`q_v = softmax(−d²/tau)`。参考、温度、度量都不学习。
- **两个严格匹配的分支**：共享冻结的 SENTENCE_BASE trunk；`r_v = Linear(64,32)(normalize(t_v))`
  注入 `h_v = BackboneNorm(BackboneMLP(concat(x_v, r_v)))`；`DIRECT: t=z`，`SHARE: t=q`。
  两个分支的归一化统计量都是 Fit-only。参数量完全相同：
  416,177（冻结 trunk）+ 2,080（`branch_proj`）+ 6,144（backbone 首层 64→192 的替换）= **424,401**。
  C6 验证了同规模两臂的初始状态逐 bit 相同、batch plan 相同；C2 验证了 `z/q/参考/tau`
  全部可由对应 Fit 重算。
- **50% 单元**：1680 个官方 Train 行，按 `sha256("css-v0-50pct/<row_id>")` 最小选取（label-free），
  另外 1680 行只作为 held-out 诊断，不参与 Fit、归一化、词表、参考或选点。
  100% 与 50% 的官方 Valid/Test 行完全相同（C1 行哈希一致）。
- **recipe**：AdamW，peak 1e-3，min 1e-5，wd 1e-3，batch 64，warmup 5 epoch，cosine，
  dropout 0.1，clip 5，标准化标签 MSE；每 epoch 全量评估 Train/Valid/Test（原始单位）；
  best checkpoint 只按官方 Valid RMSE；Test 不参与任何选择或配置。
- **为什么是 120 epoch**：吞吐测量（3 epoch，含三 split 评估）得到 epoch 2–3 均值
  100%：2.66 s，50%：2.79 s；启动开销 3.8/4.5 s。按冻结规则和 25% 余量：
  H=120 投影单 run 323/339 s，合计 0.368 GPU·h（含余量 0.46），是 `{120, 90, 60}` 中
  仍满足 10-min 目标与 0.5 GPU·h 的最大共同 horizon，因此取 120。实际四 run 为
  354.9 / 345.1 / 280.9 / 339.1 s，合计 **0.3688 GPU·h**（加 timing run 共 0.376），
  峰值显存 66.6 MB。未发生预算阻塞。

## 2. Correctness checks（远程 prep 上执行，8/8 PASS）

`analysis/checks.json`（host `a100-2`，commit `00ab1d35`，29.5 s）：

- **C1** 官方行/split 固定；50% Fit 是 100% Fit 的真子集（1680），held-out 1680，
  Valid/Test 行哈希跨规模一致。
- **C2** 所有拟合统计量只来自 Fit：词表、SVD components、`z`、`q`、参考 roots、`tau`、
  branch 表全部可重算且与 manifest 哈希一致；参考 root 全在 Fit、无 held-out 泄漏。
- **C3** 原子重编号 + 键顺序扰动的 6 个分子，`z`、`q`、预测最大绝对差均为 **0.0**。
- **C4** `q` 有限、非负、逐行和为 1。
- **C5** 相同描述子 → 相同 `z/q`：5000 组、其中 4739 组跨分子，最大差 0.0；
  113,568 个 roots 只有 16,281 个不同描述子（大量真实重复结构）。
- **C6** 两臂参数量、初始状态、batch plan 完全一致；branch 输入不同。
- **C7** branch 与 trunk 梯度有限非零，step 后参数确实更新。
- **D** 参考权重非退化：Fit 平均有效参考数 40.03 / 41.14（K=64），平均最大权重
  0.091 / 0.087，Valid 上 40.19 / 41.32；最大单权重 0.32；无 NaN/Inf。

跨主机（本地 vs A100）prep 是可复现的：64 个参考的 `global_root` 与顺序完全一致，
8 个固定 Valid 示例完全一致，`z` 最大差 `4.3e-5`、`q` 最大差 `6.0e-7`、`tau` 差 `1e-8`
（BLAS/float32 级别）。远程缺 `data/lipophilicity/labels.npy`，因此 `label_vault`
在该主机不可用（`available: false`）；官方 split 和标签来自冻结 raw cache，本地 vault
交叉校验通过（`max_abs_diff 2.3e-7`）。

## 3. 主结果

### 3.1 同 epoch 曲线（Valid / Test，原始单位）

每 epoch 都评估了 Fit/Valid/Test；完整数值在 `results/curve.csv`（4 arms × 120 epochs）。
两条 100% 曲线：

| epoch | D100 Valid | S100 Valid | D100 Test | S100 Test |
|---:|---:|---:|---:|---:|
| 20 | 0.6697 | 0.6750 | 0.6895 | 0.6915 |
| 40 | 0.6305 | 0.6540 | 0.6639 | 0.6789 |
| 60 | 0.6238 | 0.6576 | 0.6693 | 0.6422 |
| 80 | 0.6117 | 0.6501 | 0.6633 | 0.6429 |
| 100 | 0.6166 | 0.6488 | 0.6626 | 0.6407 |
| 120 | 0.6144 | 0.6469 | 0.6625 | 0.6419 |

两条 50% 曲线：

| epoch | D50 Valid | S50 Valid | D50 Test | S50 Test |
|---:|---:|---:|---:|---:|
| 20 | 0.7490 | 0.7883 | 0.7770 | 0.8071 |
| 40 | 0.7052 | 0.7298 | 0.7222 | 0.7204 |
| 60 | 0.7426 | 0.7368 | 0.7576 | 0.7796 |
| 80 | 0.7296 | 0.7457 | 0.7449 | 0.7711 |
| 100 | 0.7321 | 0.7460 | 0.7411 | 0.7735 |
| 120 | 0.7350 | 0.7478 | 0.7412 | 0.7755 |

曲线读数：

- 100%：SHARE 的 Test 从 epoch ≈50 起稳定低约 0.02，但 Valid 从 epoch ≈40 起稳定高约
  0.02–0.04；best-Valid 选点没有改变这个冲突（同 epoch 82：D `0.6065/0.6670`，
  S `0.6429/0.6400`）。
- 50%：SHARE 只在早期（epoch 32 附近）拿到更好的 Valid；之后 Valid/Test 都变差。
  因此 `+0.005/+0.013` 更像是“early-stopping 时机”而不是稳定的低数据优势。
- 所有 run 在 120 epoch 内都进入平台期（Fit 仍缓慢下降，Valid 已平坦或轻微上升），
  不是欠训练；120 是由吞吐规则选的，不构成 budget-limited。

### 3.2 50% → 100% 的性能变化

| Arm | Fit RMSE 100→50 | Valid RMSE 100→50 | Test RMSE 100→50 | Heldout (50%) |
|---|---:|---:|---:|---:|
| DIRECT | 0.1076 → 0.2432 | 0.6065 → 0.7050 | 0.6670 → 0.7386 | 0.6238 |
| SHARE | 0.1562 → 0.3769 | 0.6331 → 0.6998 | 0.6535 → 0.7259 | 0.6335 |

数据减半让两个 arm 的 Valid 都上升约 0.10，Test 上升约 0.07；SHARE 的 Fit 误差始终明显更高
（100%：+0.049；50%：+0.134），说明这个固定参考相似度分支首先是一个**容量/正则化瓶颈**，
而不是一个更强的结构表示。50% 的 held-out 随机半量上，DIRECT（0.6238）反而略好于
SHARE（0.6335），与官方 Valid 的 +0.005 方向不一致，进一步说明低数据“增益”很弱。

## 4. 机制分析（best-Valid SHARE checkpoint）

完整 JSON：`analysis/reference_analysis.json`（同时复制到 `results/reference_analysis.json`）。
参考权重和 `q` 是**冻结的、不学习的**，因此 usage / top-3 / 最近距离与 checkpoint 无关；
只有逐分子预测和误差来自对应 best-Valid checkpoint。以下是描述性分析，**不能**解释为
原子/键贡献或化学因果重要性。

### 4.1 参考使用不是退化的

- Valid 上平均有效参考数 40.2 / 41.3（K=64），平均最大权重 0.089 / 0.085，
  单 root 最大权重 0.32 / 0.34；行和严格为 1，无 NaN/Inf。
- 64 个参考的平均使用权重范围：100% `0.0047–0.0303`，50% `0.0047–0.0267`
  （均匀值 0.0156），没有“某个参考吃掉一切”的塌缩。
- 最近参考平方距离：Valid 平均 0.118 / 0.123，分布连续；`tau≈0.40` 使 `q` 保持平滑，
  这也是为什么 top-3 权重和只有 0.14–0.21。

### 4.2 八个固定 Valid 局部结构（hash 选 row，非 cherry-pick）

同一组 8 个 row 在两个规模下都使用；下表给出真实 R2 ego 大小、最近参考距离、top-3 权重和、
标签与两臂预测（100% / 50%）。完整 atom/bond 支持与 top-3 参考在 JSON 中。

| row | R2 atoms | nearest d² (100/50) | top-3 w (100/50) | label | D100 | S100 | D50 | S50 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1603 | 4 | 0.136 / 0.003 | 0.171 / 0.188 | 2.74 | 3.106 | 2.988 | 2.824 | 2.722 |
| 1430 | 7 | 0.157 / 0.168 | 0.150 / 0.147 | 0.80 | 1.595 | 1.113 | 1.872 | 1.828 |
| 1337 | 6 | 0.004 / 0.000 | 0.203 / 0.187 | 3.37 | 2.650 | 1.934 | 2.605 | 2.689 |
| 1334 | 7 | 0.401 / 0.506 | 0.168 / 0.143 | 3.15 | 2.935 | 2.655 | 3.162 | 3.261 |
| 1126 | 4 | 0.300 / 0.219 | 0.205 / 0.204 | 3.76 | 3.494 | 4.105 | 3.407 | 3.409 |
| 1384 | 4 | 0.105 / 0.173 | 0.183 / 0.157 | 2.90 | 2.931 | 2.817 | 2.753 | 3.003 |
| 1066 | 10 | 0.203 / 0.090 | 0.138 / 0.137 | 0.90 | 0.888 | 1.022 | 1.504 | 1.286 |
| 1545 | 9 | 0.000 / 0.004 | 0.206 / 0.186 | 2.00 | 1.743 | 2.002 | 1.872 | 1.650 |

两臂在 8 个例子上各有胜负（100%：S 更好 4/8；50%：S 更好 4/8），没有“近距离结构一定更好”
的模式；例如 1337 的最近参考距离几乎为 0，S100 仍大幅低估（1.93 vs 3.37）。

### 4.3 near / mid / far 分子组（按 Valid 平均最近参考距离三分位，无标签/误差参与）

| size | group | mean d² range | n | DIRECT RMSE | SHARE RMSE | gain (D−S) |
|---|---|---:|---:|---:|---:|---:|
| 100 | near | 0.032–0.099 | 140 | 0.5248 | 0.5489 | −0.0241 |
| 100 | mid | 0.100–0.133 | 140 | 0.6169 | 0.6389 | −0.0219 |
| 100 | far | 0.133–0.270 | 140 | 0.6689 | 0.7020 | −0.0331 |
| 50 | near | 0.040–0.105 | 140 | 0.6297 | 0.6416 | −0.0119 |
| 50 | mid | 0.105–0.132 | 140 | 0.6133 | 0.6625 | −0.0492 |
| 50 | far | 0.133–0.470 | 140 | 0.8475 | 0.7865 | +0.0610 |

100% 下三个组 SHARE 都更差；50% 下只有 far 组 SHARE 更好（+0.061），mid 组更差（−0.049）。
这与“连续共享应在离参考结构远的地方更有用”的直觉不一致，也没有形成稳定模式。
分组样本只有 140 个分子/组，不作显著性声明。

## 5. 预算、可复现性与产物

- CPU prep：33 s（两规模合计，含 structs/descriptors/SVD/参考/表；job 端到端 41 s），
  < 3 min 限制。
- Correctness checks：29.5 s（CPU pool）。
- 吞吐 timing：12.2 s / 13.5 s；正式四 run：354.9 / 345.1 / 280.9 / 339.1 s；
  每 run 均 < 6 min，远低于 10 min 目标，更低于 15 min 硬限。
- GPU 合计：0.3688 GPU·h（正式）+ 0.0071（timing）= **0.376 GPU·h**，< 0.5。
- 峰值显存 66.6 MB/run；batch 64，单卡 A100-40GB。
- 数据/代码/配置哈希、Fit/held-out 行哈希、branch 表哈希、batch plan 哈希、初始参数哈希、
  梯度探针均在 `results/formal/*/manifest.json`；manifest 与逐分子预测一致性在分析脚本中断言。
- 产物：
  - `analysis/checks.json`（8/8 PASS，远程）；
  - `results/main_table.csv`、`results/curve.csv`（= `curves_all.csv`，4 arms × 120 epochs）、
    `results/gain_bootstrap.json`、`results/size_sensitivity.csv`、`results/horizon.json`；
  - `results/formal/{DIRECT,SHARE}_{100,50}/`：`state_best.pt`、`curves.csv`、`manifest.json`、
    逐分子 `predictions_best_{fit,train_full,valid,test,heldout}.{npz,csv}`；
  - `analysis/reference_analysis.json`、`results/reference_analysis.json`；
  - `results/timing/`（3-epoch 吞吐 run）与 `results/prep/`（远端 prep 产物）。

## 6. 解释与判断

**为什么可能没有增益。** 三个一致观察：

1. SHARE 的 Fit RMSE 始终高得多（100%：0.156 vs 0.108；50%：0.377 vs 0.243），
   说明 `q` 分支主要是在限制有效容量；在 100% 数据上，这种正则化以 Valid 变差为代价，
   只换来 Test 的小幅改善。
2. `q` 是固定的、平滑的全局相似度（有效参考数 ~40/64，平均最大权重 ~0.09），
   而 `z` 保留了精确的局部计数；把 `q` 线性压到 32 维后，可能丢失了区分相近局部环境
   所需的信息。参考集和温度都不学习，无法针对 lipophilicity 调整“共享尺度”。
3. 50% 的微弱优势对选点敏感（SHARE 在 epoch 32 之后变差，DIRECT 在 48 才达到最好），
   held-out 随机半量上 SHARE 还更差；因此不能把 +0.005/+0.013 解读为稳定的低数据收益。

**能与不能声称的。** 能声称：在这个冻结协议、这个 split、seed 0、这个 recipe 下，
用真实参考结构的连续共享没有带来 ≥0.02 的 Valid 增益，100% 上还出现方向冲突；
机制没有塌缩，且可复现。不能声称：连续共享在别的 K/tau/维度/数据集上一定无效；
也不能把 Test 的 +0.014 当作真实泛化收益（CI 跨 0、与 Valid 冲突、单 seed）。
本轮没有做 structure-replacement 响应、贡献分解或 state algebra，故不解释“哪些原子/键导致预测”。

**若继续，最小而有信息量的实验**（本轮不执行）：保持完全相同的模型、数据和评估，
把 SHARE 分支的 `t` 换成同维度的控制量（固定随机投影的 `z`，或对 `z` 的可学习 64→32 映射），
用 ≥3 个 seed 检查 `q` 是否真的优于“任意固定 64→32 非线性特征”；只有在控制之上仍有
≥0.02 的同方向 Valid/Test 增益，才值得扩大 K/tau/数据比例。

---

*Test 使用声明：`DEVELOPMENT_TEST` 只用于最终报告，未参与 checkpoint 选择、超参数、horizon、
参考集、温度、SVD 维度或任何配置决策。*
