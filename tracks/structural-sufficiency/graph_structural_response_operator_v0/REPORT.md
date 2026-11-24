# GSRO v0 — 图结构响应算子快速探索报告

**Round**: `graph-structural-response-operator-v0` · **Protocol**: `gsro-v0-lipo-s0` · **Seed**: 0
**代码版本**: `027b6354cec6b66961e474f7c14e7b0eb6f52f99`（训练与解释代码；`e4f15d7c` 是仅解释代码追加前的同训练代码版本）
**数据**: `ogbg-mollipo` 官方 scaffold split，Train/Valid/Test = 3360/420/420；原始图缓存
`ogbg_mollipo_sq-pilot-cache-v1.npz` SHA-256 `fe002287…`（复用 `structural-query-learning` 的 loader/split）。
**执行 regime**: `res` / `a100-2` / 2×A100-SXM4-40GB / driver 550.163.01 / torch 2.5.1+cu124 / process backend。
**Test 身份**: `DEVELOPMENT_TEST`；所有 checkpoint 与 horizon 选择只用 Train/Valid。

---

## 0. 首页：七个问题的直接回答

1. **FULL 与 LOCAL2 的预测表现**
   FULL：Train 0.6489 / Valid **0.8224** / Test **0.8424**（best-Valid epoch 89，0-based；120 epochs）。
   LOCAL2：Train 0.6092 / Valid **0.7868** / Test **0.8013**（best-Valid epoch 100）。
   两臂参数完全一致：**72,881**，初始 state bitwise 相同。Formal wall time：FULL 153.3 s，LOCAL2 93.6 s。

2. **全局响应相对局部二阶计算是否有收益**
   **没有。** 主执行 FULL − LOCAL2 的 Valid RMSE 差 **+0.0356**（95% paired bootstrap CI −0.0029…+0.0714），
   Test **+0.0411**（−0.0069…+0.0903）；MAE 差的 95% CI 偏向 LOCAL2（Valid +0.0332 [0.0021, 0.0643]，
   Test +0.0411 [0.0071, 0.0743]）。同配置重复执行的差为 +0.0060 / +0.0128，方向一致但幅度在运行间波动内。
   即：没有证据表明全局求解带来更好的预测或可学习性；方向上是 LOCAL2 至少不差。

3. **与历史模型的绝对性能差距**
   FULL 比 LCRC 差 **+0.1172 / +0.1285**（Valid/Test），比 HSRF 差 **+0.0309 / +0.0605**，
   比 SWF 差 +0.1233 / +0.1533，比 GINE 差 +0.2133 / +0.1968。
   LOCAL2 比 LCRC 差 **+0.0816 / +0.0874**，比 HSRF 为 **−0.0047 / +0.0194**。
   两臂都远未接近 LCRC/GINE；LOCAL2 只在 Valid 上略好于 HSRF 的坐标。

4. **预测贡献账目能否精确复算**
   **能。** 64 个固定 hash Valid 分子上，`C_bias + Σ_v C_v + Σ_e C_e` 与模型预测的最大绝对误差
   **1.55e-15**（平均 3.49e-16；全部 < 1e-9）；FP64 重算与保存的 FP32 预测最大差 6.7e-7。
   3 个分子的完整节点/边账目（含原子序数、键型、每条贡献）见 `results/explanation.json`。

5. **耦合敏感度是否正确**
   **正确。** 对 192 条边 × 16 通道 = 3072 个 `dQ_k/dw_ek`，解析式 `−(x_uk−x_vk)^2` 与 FP64 autograd
   最大绝对误差 **2.13e-14**；有限差分 spot check 最大误差 **9.36e-9**；最终预测对单通道边耦合的导数
   最大误差 **3.3e-16**。全通道 w=0 直接求解干预（192 条边）平均 |Δpred| 0.0442、最大 0.598。
   **注意**：耦合干预不是化学删键，能量贡献也不等于删边后的预测变化（实测两者相关 0.845 但比值不为 1）。

6. **计算时间与实际成本**
   吞吐量预检预计 120 epochs ≈ **142.7 s**（目标 10–15 min，硬上限 25 min），因此固定 horizon = **120**；
   预检后重新初始化。实际并发执行：FULL 153.3 s、LOCAL2 93.6 s，峰值显存 287/148 MB。
   主执行 GPU 成本 **0.069 GPU·h**；含预检、同配置重复与一次启动失败，合计约 **0.134 GPU·h**，远低于 0.85 上限。

7. **是否存在值得继续研究的明确现象**
   有一个明确的**机制性负结果**：冻结参数替换显示训练后的 FULL 强烈依赖全局耦合
   （0.8224 → 用 LOCAL2 替换 0.9854，→ w=0 节点响应 1.4294），但**独立训练的 LOCAL2 更好（0.7868）**。
   这正是任务 §11 所述“只在冻结替换中表现出依赖、却没有独立训练优势”的情形，不能宣称全局求解提高了学习能力。
   两个模型 Valid 均 > 0.75，按本轮参考规则**不继续该配置**。没有观察到值得追加实验的预测信号。

---

## 1. 设置与共享参数生成器

- **节点输入**：OGB 原始 9 个 atom categorical（每列 embedding 16，Train-only 词表 + unseen=0），
  加上一跳邻居的 `atomic number × bond type` 联合计数（22 个 Train 组合 + 1 个 unseen 桶，
  `log1p`，Train 节点上的 z-score）。节点 MLP：`167 → 128 → SiLU → 64 → SiLU`。
- **边输入**：原始 3 个 bond categorical（每列 embedding 16）；`[h_u+h_v, |h_u−h_v|, h_u*h_v, bond]`
  （192+48=240）→ `128 → SiLU → 64 → SiLU`；每条物理无向边只算一次，端点交换不变。
- **参数生成**：`s = 2 tanh(source_head(h_v))`，`r = 0.5 + 1.5 σ(self_head(h_v))`，
  `w = 2 σ(edge_head(h_e))`，固定 K=16。初始化：source 小非零随机（权重 N(0,0.02)、bias U(−0.05,0.05)），
  self bias 使 r≈1，edge bias 使 w≈0.2；共享生成器参数 72,881，FULL 与 LOCAL2 的初值 bitwise 相同。
- **FULL**：`K = diag(r + Σ_incident w) − A_w`，用 `torch.linalg.cholesky_ex` + `cholesky_solve`
  批量直接求解；不显式求逆、不迭代。同一 batch 内按分子大小排序分桶，padding 节点 `s=0`、对角=1、
  无跨图耦合，`n` 与所有归一化只统计真实节点（最大分子 115 原子，未截断）。
- **LOCAL2**：`D_v = r_v + Σ w`，`t=s/D`，`p_v = Σ_(u,v) w_uv t_u`，
  `Q0=Σ s²/D`，`Q1=2Σ_e w t_u t_v`，`Q2=Σ p²/D`，`Q=Q0+Q1+Q2`；不做矩阵求解。
- **读出**：`y_std = b + Σ_k (α_k/n̄ + β_k/n) Q_k`；α,β ~ N(0,0.01)，b=0；标签用 Train 均值/方差标准化，
  报告恢复到原始单位。没有额外 MLP readout、全局描述符旁路或隐表示旁路。
- **训练**：AdamW，peak LR 1e-3、min 1e-5、weight decay 1e-4、batch 128、warmup 5 epochs + cosine、
  MSE（标准化目标）、grad clip 5、seed 0、FP32；每 epoch 全量 eval-mode Train/Valid/Test；
  Valid RMSE 选 checkpoint；两臂相同 120 epochs。

---

## 2. 工程检查

`results/checks.json`（在正式训练同一 commit、同一 GPU regime 下运行）共 8 项必需检查 + 共享初值检查，
全部通过：

| 检查 | 结果 |
|---|---|
| 原子重编号 + 边顺序不变性（FULL/LOCAL2） | 最大相对差 2.78e-17，通过 |
| batch vs 单图一致性（含 padding） | Q 最大相对差 FULL 2.78e-17 / LOCAL2 0.0，预测差 0，通过 |
| K 对称正定、线性系统残差 | FP64 最小特征值 0.992、条件数 2.01、相对残差 5.1e-16；FP32 相对残差 2.7e-7；Cholesky 无失败 |
| FULL vs 独立 FP64 NumPy 直接解 | 最大相对误差 5.55e-17，通过 |
| LOCAL2 vs 显式矩阵多项式 `Q0+Q1+Q2` | 最大相对误差 2.78e-17，通过 |
| w=0 时 FULL 与 LOCAL2 一致 | 最大相对误差 8.33e-17，通过 |
| 所有有效参数梯度有限、各 head 梯度非零 | 通过（response 输出 s/r/w 均有非零 spread） |
| 精确响应分解复算预测 | 最大绝对误差 4.44e-16，通过 |
| 参数量 / 共享初值一致 | 均 72,881；state dict bitwise 相同；生成器 s/r/w bitwise 相同 |

---

## 3. 主结果

主执行：FULL 与 LOCAL2 于 2026-11-24 16:50:09 / 16:50:14 并发启动（`res-gpu0` / `res-gpu1`）。
`best epoch` 为 0-based（与 `curve.csv` 的 `epoch` 列一致）。

| 模型 | 参数量 | best epoch | Train RMSE | Valid RMSE | Test RMSE | wall time |
|---|---:|---:|---:|---:|---:|---:|
| FULL | 72,881 | 89 | 0.6489 | **0.8224** | **0.8424** | 153.3 s（2.55 min） |
| LOCAL2 | 72,881 | 100 | 0.6092 | **0.7868** | **0.8013** | 93.6 s（1.56 min） |

- 两臂 Train/Valid/Test 均在同一 epoch 网格上评估；Test 只在两臂完成后汇总，不参与任何选择。
- 主执行 run id：`gsro-v0-full-final-20261124-165008-8609a87a`、
  `gsro-v0-local2-final-20261124-165012-731941b5`；解释：`gsro-v0-explain3-20261124-165529-dfbb4b94`。

**同配置重复（透明度说明）。** 首次 formal FULL 启动因命令行引号错误在 6 s 内退出（未训练），修复后执行了
上面的并发主执行。另有一对更早完成的同配置、同 seed、同 120 epochs 运行（未并发；训练代码相同，commit
`e4f15d7c`）：FULL Valid/Test 0.8075/0.8215，LOCAL2 0.8015/0.8086。两臂同配置重跑的 Valid 波动约 0.015
（实现中的 `scatter_add` 在 CUDA 上对重复索引的累加顺序不确定，训练轨迹不是逐位可复现）。
因此**不把两臂的小差距解释为稳定的跨 seed 结果**；但两次执行中 FULL 都不优于 LOCAL2，且都 > 0.75。

---

## 4. FULL vs LOCAL2

### 4.1 best-Valid 与同 epoch 曲线

| epoch（0-based） | FULL Valid | LOCAL2 Valid | FULL−LOCAL2 | FULL Train | LOCAL2 Train |
|---:|---:|---:|---:|---:|---:|
| 0 | 1.2179 | 1.2179 | 0.0000 | 1.2114 | 1.2114 |
| 20 | 0.9567 | 0.9347 | +0.0220 | 0.8887 | 0.8805 |
| 40 | 0.8474 | 0.8502 | −0.0028 | 0.7380 | 0.7476 |
| 60 | 0.8421 | 0.8161 | +0.0260 | 0.6815 | 0.6603 |
| 80 | 0.8384 | 0.7952 | +0.0432 | 0.6551 | 0.6232 |
| 89（FULL best） | **0.8224** | 0.7908 | +0.0316 | 0.6489 | 0.6198 |
| 100（LOCAL2 best） | 0.8259 | **0.7868** | +0.0390 | 0.6376 | 0.6092 |
| 119 | 0.8253 | 0.7905 | +0.0347 | 0.6349 | 0.6068 |

- best-Valid 差距 **+0.0356**（FULL 更差）。
- 同 epoch 平均差 +0.0172，median +0.0214；FULL 仅在 **21.7%** 的 epoch 上 Valid 更好。
- FULL 的 Valid 在 epoch 89 后不再改善（最后 5 epoch 变化为 −0.0002），LOCAL2 在 epoch 100 后基本平台
  （最后 5 epoch 改善约 0.001）。**不是“horizon 过短、曲线仍快速下降”的情形**。

### 4.2 Train 拟合差异

在各自 best-Valid checkpoint 上，FULL Train RMSE **0.6489**，LOCAL2 **0.6092**：FULL 的 Train 拟合**更差 +0.0396**。
同 epoch 的 Train 差平均 +0.0158，最后 epoch +0.0281。也就是说 FULL 在本配置下不是“拟合更好但泛化更差”，
而是**训练目标本身就更难优化/拟合**。

### 4.3 逐分子配对差与 1000 次 paired bootstrap

定义 Δ = FULL − LOCAL2（正值 = FULL 更差）。对 best-Valid checkpoint 的逐分子预测，在同一组分子上配对重采样
1000 次（seed 0）：

| split | ΔRMSE（95% CI） | ΔMAE（95% CI） | FULL 更好的分子比例 |
|---|---:|---:|---:|
| Valid | +0.0356（−0.0029 … +0.0714） | +0.0332（+0.0021 … +0.0643） | 46.2% |
| Test | +0.0411（−0.0069 … +0.0903） | +0.0411（+0.0071 … +0.0743） | 46.0% |

逐分子平方误差差（FULL−LOCAL2）的均值：Valid +0.0572、Test +0.0675；中位数：+0.0115 / +0.0089。
RMSE 差的 CI 跨零；MAE 差的 CI 偏向 LOCAL2。结合同配置重复的波动（约 0.015 Valid），只能得到
“**没有 FULL 优势的证据**”，不能得到“LOCAL2 稳定更优”的强结论。

---

## 5. 历史坐标（只读、描述性）

历史协议与训练长度不同，以下仅作绝对位置参照，不与本轮构成同协议比较。

| 模型 | Valid RMSE | Test RMSE | FULL 差（Valid/Test） | LOCAL2 差（Valid/Test） |
|---|---:|---:|---:|---:|
| LCRC | 0.7052 | 0.7139 | +0.1172 / +0.1285 | +0.0816 / +0.0874 |
| SWF WORD_PREFIX | 0.6991 | 0.6891 | +0.1233 / +0.1533 | +0.0877 / +0.1122 |
| HSRF | 0.7915 | 0.7819 | +0.0309 / +0.0605 | −0.0047 / +0.0194 |
| GINE | 0.6091 | 0.6456 | +0.2133 / +0.1968 | +0.1777 / +0.1557 |

结论：两臂都显著落后于 LCRC 与 GINE；LOCAL2 的 Valid 仅与 HSRF 坐标相当，Test 仍差 +0.0194。

---

## 6. 解释性验证（只用 best-Valid FULL checkpoint）

### 6.1 精确预测分解

对固定 hash 选出的 64 个 Valid 分子，在 FP64 下计算：

`Q_k = Σ_v r_vk x_vk² + Σ_e w_ek (x_uk−x_vk)²`，
`C_v = σ_y Σ_k c_k(n) r_vk x_vk²`，`C_e = σ_y Σ_k c_k(n) w_ek (x_uk−x_vk)²`，
`C_bias = μ_y + σ_y b`。

- 64/64 分子满足 `prediction = C_bias + Σ_v C_v + Σ_e C_e`；最大绝对误差 **1.55e-15**，平均 **3.49e-16**。
- 本 checkpoint：`bias` = **2.13437**（原始单位）；FP64 预测与保存的 FP32 预测最大差 6.7e-7。
- 3 个分子的完整账目（每个原子的 atomic number、每条边的端点/键型/贡献）保存在
  `results/explanation.json → prediction_decomposition.three_molecule_ledgers`。其汇总：

| row_id | n / E | bias | Σ C_v | Σ C_e | 预测 | 复算误差 |
|---|---:|---:|---:|---:|---:|---:|
| 1508 | 29 / 32 | 2.13437 | +0.92795 | −0.33930 | 2.72302 | 4.4e-16 |
| 1342 | 26 / 29 | 2.13437 | +0.48438 | −0.50881 | 2.10995 | 4.4e-16 |
| 1362 | 25 / 27 | 2.13437 | −2.77480 | −0.65627 | −1.29669 | 1.6e-15 |

- 3 个分子的节点贡献范围约 [−0.89, +0.29]，边贡献范围约 [−0.23, +0.08]；边贡献绝对值明显小于节点项，
  且大多为负。该分解是当前计算图的精确能量分解，**不是唯一的化学归因**。

### 6.2 实际耦合敏感度

解析式：`dQ_k/dw_ek = −(x_uk−x_vk)²`；预测对单通道边耦合的导数为
`σ_y c_k(n) · [−(x_uk−x_vk)²]`。

- 64 个分子、每个至多 3 条固定 hash 边（共 192 条边、3072 个通道项）：
  - 解析式 vs FP64 autograd：最大绝对误差 **2.13e-14**（最大相对误差 7.0e-7）。
  - 解析式 vs FP64 中心差分（ε=1e-6，8 个 spot check）：最大绝对误差 **9.36e-9**。
  - 最终预测对单通道耦合的导数 vs autograd：最大绝对误差 **3.3e-16**。
- **全通道 w=0 直接求解干预**（192 条边）：平均 |Δpred| **0.0442**，最大 **0.598**；
  同一条边的能量贡献 `C_e` 平均绝对值 0.0149，二者相关系数 0.845，但逐条比值明显不为 1。
  因此“把耦合置零”不是“删除化学键”，“能量贡献”也不是“删边后的预测变化”；本节不宣称化学因果贡献。

### 6.3 全局求解是否被模型使用（冻结替换）

冻结参数生成器与读出，只替换 Valid 上的响应计算：

| 响应计算（FULL checkpoint） | Valid RMSE | 相对 FULL 的 mean / max |Δpred| |
|---|---:|---:|
| FULL（基线，FP64） | **0.8224** | — |
| 替换为 LOCAL2 | **0.9854** | 0.360 / 2.686 |
| 替换为 w=0 独立节点响应 | **1.4294** | 1.002 / 4.020 |

这说明：在该 FULL checkpoint 的参数下，模型**确实依赖**全局耦合（替换后显著变差）。但独立训练的 LOCAL2
（同生成器、同数据、同优化）达到 0.7868，优于 FULL。按任务 §11，这**不能**被解释为“全局求解提高了学习能力”；
它只说明 FULL 的参数已经适配了全局算子，而局部二阶参数化独立训练时能达到至少一样好的预测。

### 6.4 可选真实结构变化评价

未执行。仓库中未找到可直接复用的 **Lipo 官方 Valid/Test matched-pair 目录**（现有配对多为内部 split 或
针对特定酰胺改线自构造的目录，需要新的结构构造与预算）。按 §10.4 的可选条件跳过，未构造新目录、未训练新模型。

---

## 7. 计算、成本与重复性

- **吞吐量预检**（正式训练前，`res-gpu0`，只看速度）：按 120 epochs 预计 **142.7 s**（FULL 1.19 s/epoch，
  LOCAL2 0.72 s/epoch；pilot 于同 commit 运行）。目标 10–15 min → 选择 **120 epochs**；预检后重新以 seed 0
  初始化。另一次预检（`e4f15d7c`）预计 101.4 s，同样给出 120 epochs。horizon 决定与任何预测指标无关。
- **正式执行**：FULL 153.3 s（含每 epoch 全量 Train/Valid/Test 评估）；LOCAL2 93.6 s；并发启动相差 5 s。
  峰值显存 287.3 MB / 148.1 MB。解释性验证为 CPU FP64，约 10 s。
- **GPU 成本**：主执行 (153.3+93.6)/3600 = **0.069 GPU·h**；含预检、同配置重复、启动失败总计约
  **0.134 GPU·h**，低于 0.85 GPU·h 上限。
- **重复性**：同 seed 同配置重跑的 Valid 约有 0.015 的差异（两臂均如此）。实现中的 `scatter_add`
  在 CUDA 上对重复索引的累加顺序不确定（PyTorch 文档），所以训练轨迹不是逐位可复现。本文所有
  FULL−LOCAL2 小差距都用 paired bootstrap 与重复运行波动一起读，不把它当作稳定机制。

---

## 8. 决策

**A. FULL 的绝对性能是否值得继续发展？**
不。FULL Valid 0.8224（重复 0.8075），明显差于 LCRC 0.7052 与 GINE 0.6091；也差于 HSRF 0.7915。
按本轮参考规则（两臂 Valid > 0.75）**不继续该配置**。

**B. 全局求解是否比参数匹配的 LOCAL2 有明显收益？**
没有。主执行 FULL 比 LOCAL2 Valid 差 +0.0356、Test 差 +0.0411；重复执行为 +0.0060 / +0.0128；方向一致
但幅度在运行波动内，RMSE 的 95% CI 跨零。FULL 的 Train 拟合也更差（0.6489 vs 0.6092），说明本配置下
全局求解没有把优化/样本效率变得更好。
冻结替换实验显示 FULL 依赖全局耦合，但独立训练的 LOCAL2 至少一样好——按 §11 明确**不宣称**全局求解
提高了学习能力。

**结论**：本配置（K=16、共享局部生成器、显式 `c_k(n)` 读出、120 epochs、单 seed）在 `ogbg-mollipo`
官方 split 上没有显示全局结构响应算子的预测优势；属于**明确的快速筛选负结果**。不追加 seed、不调参、
不增加实验。精确分解与耦合敏感度机制是正确且可复用的工程资产，但本轮没有值得继续追的预测信号。

---

## 9. 产物与局限

**产物**（均在本目录）：

- `configs/gsro_v0.json`、`PROTOCOL.md`、`code/`、`tests/`
- `results/checks.json`、`results/pilot.json`
- `results/curve_full.csv`、`results/curve_local2.csv`
- `results/main_table.csv`、`results/analysis.json`
- `results/full_best.pt`、`results/local2_best.pt`
- `results/predictions_full.npz`、`results/predictions_local2.npz`
- `results/explanation.json`（含 64 分子分解、3 分子完整账目、耦合敏感度、冻结替换）
- `REPORT.md`（本文件）

**局限**：

- 单 seed、单数据集（Lipo）、单 horizon；不构成跨 seed 稳定性或跨任务结论。
- 约 0.015 的 GPU 运行间波动与 FULL−LOCAL2 的小差距同量级；因此只报告“无 FULL 优势证据”，
  不作 LOCAL2 稳定更优的强声明。
- Test 是 DEVELOPMENT_TEST；没有再为机制解释追加训练或调参。
- 历史坐标来自不同协议与训练长度，仅作描述性参照。
