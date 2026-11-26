# SBL v0 — FIXED1 骨架均衡学习短程验证（`fixed-source-scaffold-balanced-learning-v0`）

- Round：`tracks/structural-sufficiency/fixed_source_scaffold_balanced_learning_v0/`
- 数据：官方 `ogbg-mollipo` scaffold split，Train 3360 / Valid 420，seed 0。**Test 未读取、未评分、未进入任何选择**
  （数据层与所有 batch/eval 入口带 `train|valid` 卫兵；bundle 载入后立即丢弃 Test 数组）。
- 唯一训练臂：`SCAFFOLD_BALANCED`（模型/初始化/优化器/顺序/配方完全复用 FIXED1，只换损失权重）。
- 历史对比：FIXED1 = `context_conditioned_structural_computation_v0/runs/mollipo_A_seed0`（只读）。

## 首页结论

**让每个训练骨架家族的总优化权重相等，没有改善官方 Valid 预测。判定 `NO_CLEAR_GAIN`。**

| 指标 | SCAFFOLD_BALANCED | FIXED1（历史） | Δ（新 − 旧） |
|---|---:|---:|---:|
| best Valid RMSE | **0.7274001** | 0.7266780 | **+0.000722**（变差 +0.10%） |
| best Valid MAE | 0.5189704 | 0.4984493 | +0.020521（+4.1%） |
| endpoint（epoch 360）Valid RMSE | 0.7463312 | 0.7415200 | +0.004811 |
| endpoint Valid MAE | 0.5277198 | 0.5060956 | +0.021624 |
| last-20 Valid RMSE | 0.7471151 | 0.7406963 | +0.006419 |
| last-20 Valid MAE | 0.5267916 | 0.5061565 | +0.020635 |
| best epoch | 136 | 140 | — |
| 参数量 | 294,135 | 294,135 | 0 |
| 墙钟 / GPU·h | 565.1 s / **0** | 599.9 s / 0.167 | — |

- 三条主指标（best / endpoint / last-20）**同方向变差**，MAE 明确退化 → 不满足预注册的
  `STRUCTURAL_DISTRIBUTION_PROMISING`（需 Δbest ≤ −0.03、endpoint 与 last-20 同向改善、ΔMAE ≤ +0.005）。
- **没有超过历史 C1**（0.7086243）：本轮 best Valid RMSE 高 **+0.018776**。也没有接近历史 GINE（0.6780750）。
- 干预**确实发生了**：逐分子预测相关性 0.952、RMS 变化 0.348、中位绝对变化 0.216；权重对比从 76× 压到 1×。
  也就是说，这不是"权重没改到"的空操作，而是"改了、方向不对/无效"。
- **不是优化异常**：梯度裁剪频率 47.7%（历史 43.3%）、加权与普通 Train loss 都单调下降、无 NaN/Inf、
  LR 与冻结 schedule 逐位一致（首 step 3e-4、末 step 1e-5）。
- **低支持分子没有获益**（−0.0028 RMSE 但 MAE +0.024，bootstrap 跨零），而高支持组反而退化 +0.0236。
- 按预注册停止规则：不搜索新的权重公式、不追加 seed、不进入下一阶段。

## 1. 数据准备：骨架分布与权重对比

只用官方 Train SMILES，逐字复用上一轮 `sfl_common.scaffold_key`
（`MolFromSmiles → MurckoScaffold.GetScaffoldForMol → MolToSmiles`）：

| 规则 | 处理 |
|---|---|
| chirality | `MolToSmiles` 默认 `isomericSmiles=True`，立体标识保留在 scaffold 串里；**与上一轮记录的 1604/420 一致**（`includeChirality=False` 会给 1581，是不同定义，本报告如实标注） |
| 空 scaffold | 无环分子返回 `""`，5 个分子（row 1573/1701/1948/2087/4180）**共用一个 `""` 组**，不拆成单例 |
| 多组分 | 不剥离盐、不取最大片段，按解析后的整个分子计算；Train 中多组分 SMILES = 0 个（Valid 1 个，只影响诊断） |
| 标签 | 权重函数不接收标签；只用 scaffold 分组 + N_train + K_train |

**核对历史记录**：Train **K = 1604** 个不同 Bemis–Murcko scaffold（上一轮记录 1604 ✔）；
Valid 420 个分子 = **420 个不同 scaffold**，且与 Train scaffold **零重叠**（上一轮记录 420 ✔，
同时确认官方 scaffold split 的边界）。

| 项目 | 值 |
|---|---:|
| N_train | 3360 |
| K（不同 scaffold） | 1604 |
| 平均 scaffold 大小 | 2.095 |
| singleton scaffold | **1065**（占 scaffold 66.4%，占分子 31.7%） |
| 最大 scaffold | **76** 个分子（2.26% of Train） |
| ≥10 个分子的 scaffold | 37 个，覆盖 674 个分子（20.1%） |
| 大小分布（前段） | 1:1065, 2:258, 3:100, 4:43, 5:37, 6:22, 7:24, 8:14, 9:4, 10:5, …, 36:1, 76:1 |

权重 `w_i = N/(K·n_s(i))` 的对比：

| 口径 | 原始分子等权目标 | 骨架均衡目标 |
|---|---|---|
| 每分子权重 | 1（全 Train 归一后 1/N） | 0.02756 – **2.0948**（**76×**），均值 1.0，p25 0.299 / p50 0.698 / p75 2.095 |
| 单个 scaffold 总权重（权重单位） | 1 – 76（**76×**） | 全部 = N/K = **2.0948**（**1×**） |
| 单个 scaffold 总权重（÷N） | 2.98e-4 – 2.26e-2 | 全部 = 1/K = 6.234e-4 |
| 有效样本量 ESS = (Σw)²/Σw² | 3360 | **2046.5（60.9%）** |

预注册门禁：`max/min weight ≥ 1.5` 或 `ESS/N ≤ 0.9` 才继续 → 实际 76× 与 0.609，
**未触发 `INSUFFICIENT_WEIGHT_CONTRAST`**，训练继续。

## 2. 训练臂与正式训练前检查

`code/sb_checks.py` → `analysis/checks.json`：**10/10 通过**（含 0 号冻结配置检查）。

| 检查 | 结果 |
|---|---|
| 0 冻结配置/身份/顺序 | K、权重 sha256、plan sha256 与 `configs/experiment.json` 一致；plan = `f9c1398c…` 与历史 FIXED1 完全同一；缓存 SMILES 与 `mol.csv.gz` 规范形逐行一致；Train-only 目标标准化参数（mean 2.1621607144740187 / std 1.211539840098397 / n 3360 / ddof 0）与历史逐位相同；Test 数组已丢弃 |
| 1 初始化 | `initial_state_sha256 = 8b2df74e…`、`base_state_sha256 = 539e0dbe…`，与历史 FIXED1 **逐位相同** |
| 2 结构/推理不变 | 294,135 参数（base 259,831 + 34,304），state_dict 键集合与形状与历史 checkpoint 完全一致，无 Gate；历史 checkpoint 在本机 CPU 重载后 Valid RMSE `0.7266780031`（记录值 0.7266780070，Δ 3.9e-9，设备级浮点差） |
| 3 全部训练分子参与 | 360 个 epoch 每个都是 0..3359 的完整置换，累计行次 1,209,600 |
| 4 每个 scaffold 总权重相等 | 最大绝对偏差 8.9e-16（相对 4.2e-16） |
| 5 平均权重 / 分位数 / ESS | 均值 0.9999999999999999（float64 精确为 1）；min/max/分位/ESS 见 §1 |
| 6 权重与标签无关 | 200 次 Train 标签置换后权重向量 sha256 全部相同；函数签名不含标签参数 |
| 7 权重全 1 ⇒ 原始 MSE | 真实 batch 上损失**逐位相等**（35.29475021362305），65 个参数梯度**逐位相等**（max diff 0.0） |
| 8 验证指标不使用训练权重 | `evaluate` 不接受权重参数；手工重算 Valid RMSE/MAE 与报告值差 < 1e-12 |
| 9 吞吐预检 | 投影 360 epoch 611.95 s（实测 565.06 s），远低于 2100 s 硬墙钟 |

正式训练：360 epoch / 9720 step，AdamW peak 3e-4 → 1e-5 每步 cosine、wd 1e-3、clip 5、FP32、
dropout 0.10、batch 128、Train-only 标准化目标、与历史同一 batch 顺序。**唯一改动**是
`loss = mean_batch(w_i · per-molecule MSE)`。Valid RMSE/MAE 一律普通分子等权。

## 3. 结果

`analysis/main_table.csv`、`analysis/valid_curve.csv`（360 行，本轮与 FIXED1 逐 epoch 对齐）、
`analysis/curve_statistics.json`。

| 曲线统计（Valid RMSE） | SCAFFOLD_BALANCED | FIXED1 |
|---|---:|---:|
| best（epoch） | 0.7274001（136） | 0.7266780（140） |
| epoch 100–360 均值 ± std | 0.751387 ± 0.01046 | 0.744677 ± 0.01080 |
| epoch 100–360 中位数 | 0.747696 | 0.741070 |
| last-20 均值 ± std | 0.747115 ± 0.000998 | 0.740696 ± 0.001457 |
| 低于 0.73 的 epoch 数 | 1 | 3 |

两条曲线的逐 epoch 相关 0.835；late-horizon（100–360）均值差 **+0.006710**，与 last-20 差 (+0.006419)
一致，说明"best 差 0.0007"不是唯一读数：整条后段曲线都比 FIXED1 略差。
同时要注意 best 值是"单 epoch 顺序统计量"——两条曲线在 0.73–0.75 之间逐 epoch 抖动（std ≈ 0.010），
best 的 0.0007 差异远小于这一抖动。

**同 epoch 的普通（非加权）Train 指标**（`analysis/epoch_aligned_train.json`）：

| epoch | 本轮 Train RMSE / MAE | 同 epoch FIXED1 | ΔRMSE |
|---|---:|---:|---:|
| 136（本轮 best） | 0.391283 / 0.285880 | 0.335739 / 0.253643 | +0.055544 |
| 140（历史 best） | 0.377841 / 0.271812 | 0.307986 / 0.233503 | +0.069854 |

即：在**普通等权**口径下本轮拟合更差（被降权的大 scaffold 通常是"好拟合"的部分），
但被加权的小 scaffold 的拟合更好（这是权重公式的机械后果，不是泛化证据）。

## 4. 逐分子比较（本轮 best checkpoint vs FIXED1 best checkpoint，420 个 Valid 分子）

`analysis/per_molecule_valid.csv`、`analysis/paired_comparisons.json`。

| 统计 | 值 |
|---|---:|
| ΔMSE（新 − 旧） | +0.001050 |
| ΔRMSE | +0.000722 |
| ΔMAE | +0.020521 |
| 改善 / 退化分子数 | **204 / 216**（改善比 48.6%） |
| 逐分子增益（旧 SE − 新 SE）之和 | 正 64.310，负 −64.751，净 **−0.441** |
| paired bootstrap（2000 次，seed 20261126）ΔRMSE | 均值 +0.001590，95% 区间 **[−0.042839, +0.047972]**，47.15% 采样为负 |
| paired bootstrap ΔMAE | 均值 +0.020593，95% 区间 [−0.008580, +0.050518]，9.2% 采样为负 |
| 去掉双方共同平方误差最大的 5%（21 个分子）后 | RMSE 0.565750 vs 0.528994（Δ **+0.036755**）；MAE 0.447580 vs 0.419712（Δ +0.027868） |
| 预测变化 | 相关 0.952；RMS 变化 0.3479；MAE 变化 0.2644；最大 1.985；平均符号差 −0.0789 |

读法：整体判据（bootstrap 跨零、改善/退化几乎各半、去尾后差距更大）说明这不是"少数尾部案例"造成的，
也不是"整体小幅改善"。增益并没有集中在少数分子上：净增益为负，正向贡献最大的 5% 分子
（1425、1818、1397、1601、1539、1552、1660、1816、…）只占全部正向贡献的 58.3%。

## 5. 训练支持度诊断（唯一一个，预注册）

Morgan r=2、2048 bit；Valid 每个分子的最大 Train Tanimoto；阈值 = **Train leave-one-out 最大相似度的中位数
= 0.732143**（不搜索）。Train LOO 最大值中位 0.732、Valid 最大相似度中位 0.523（Valid 明显更低，
与官方 scaffold split 的外推性质一致）。

| 组 | n | RMSE 新 / 旧 | ΔRMSE | MAE 新 / 旧 | ΔMAE | ΔMSE | 改善/退化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 相对低支持（< 0.7321） | 326 | 0.790933 / 0.793760 | **−0.002827** | 0.569992 / 0.545756 | +0.024235 | −0.004480 | 154 / 172 |
| 相对高支持（≥ 0.7321） | 94 | 0.441096 / 0.417538 | **+0.023558** | 0.342025 / 0.334385 | +0.007640 | +0.020227 | 50 / 44 |

低支持组的 ΔRMSE 为 −0.0028（bootstrap 均值 −0.001279，95% 区间 [−0.052220, +0.051892]，
52.85% 采样为负），同时 MAE 退化 +0.024。**不能称为"低支持分子受益"**：这只是一条统计上不可区分、
方向不一致的线索。

## 6. 上一轮 11 个固定案例（描述性）

`analysis/fixed_cases.csv`（来源 `lcrc_structural_failure_localization_v0/analysis/cases.csv`，只读）：

- 平方误差改善 **7** 个（1397 −3.539、1425 −7.066、1601 −2.912、1647 −0.622、1653 −0.492、1816 −1.386、1818 −6.297），
  退化 **4** 个（1029 +0.982、1188 +0.448、1213 +0.415、1810 +0.320）。
- 这 11 个分子上的 RMSE 从 2.695 降到 2.330，平均绝对预测变化 0.616。
- 但这是**极端误差子集**：整体指标与"去掉最大共同误差 5%"的敏感性都不支持由此推广。
  上一轮这 11 个案例里有 5 个属于 GAIN5（当时相对 GINE 的受益组），本轮它们同样受益——这更多说明
  "这些分子的预测对训练权重敏感"，而不是"低支持机制被修复"。

## 7. 优化诊断（防止把优化异常解释成结构迁移）

`analysis/training_diagnostics.json`

| 项目 | SCAFFOLD_BALANCED | FIXED1 |
|---|---:|---:|
| 梯度裁剪事件 / 总 step | **4640 / 9720 = 47.7%** | 4210 / 9720 = 43.3% |
| 出现裁剪的 epoch | 342 / 360 | 328 / 360 |
| 单 epoch 最大裁剪次数 | 27（= 全部 step） | 27 |
| grad-norm 最大 | 1999.3 | 2198.2 |
| 加权 Train loss（batch 加权均值）第 1 / 360 epoch | 2.9802 → 0.03231 | —（原始 MSE：3.0776 → —） |
| 普通 Train MSE（train mode，分子等权）第 1 / 360 epoch | 3.1346 → 0.06184 | — |
| 普通 Train MSE（eval mode）最终 | 0.26398（RMSE） | 0.18316 |
| NaN/Inf | 无 | 无 |
| LR 首/末 step | 2.99995e-4 / 1.0000008e-5 | 同 |

裁剪频率与历史几乎同量级（+4.4 个百分点），说明"梯度裁剪"是本轮冻结配方（LR 3e-4、batch 128、clip 5）
的既有性质，不是加权损失造成的异常。加权 loss 与普通 Train MSE 都单调下降、无发散。

## 8. 判读

按 `analysis/verdict.json` 的预注册规则：

- Δbest RMSE = +0.000722 > −0.03；endpoint +0.004811 > 0；last-20 +0.006419 > 0；ΔMAE +0.020521 > +0.005
  → 不满足 `STRUCTURAL_DISTRIBUTION_PROMISING`。
- 也没有"少量收益"或"尾部集中收益"（净增益为负）→ 不是 `WEAK_OR_TAIL_ONLY_GAIN`。
- 无数据/实现/预算异常（`checks.json` 10/10 通过；`post_checks.json` 的 `all_ok = 1`，其中 `test_scored = 0` 表示 Test 未被评分）→ 不是 `INCONCLUSIVE`。

**判定：`NO_CLEAR_GAIN`。**

### 回答本轮的 5 个问题

1. **训练骨架权重实际上改变了多少？**
   很大。每个 scaffold 的总优化权重从 1–76（76×）压成全部 2.095（1×）；逐分子权重从全 1 变成
   0.0276–2.095（76×），1581 个分子被加权（>1）、1779 个被降权（<1），有效样本量 3360 → 2046.5。
   单例 scaffold 的分子权重 ×2.09，最大 scaffold 的分子权重 ×0.0276（降 36 倍）。
2. **完整官方 Valid 预测是否改善？**
   没有。best Valid RMSE +0.000722（变差）、endpoint +0.004811、last-20 +0.006419、MAE +0.020521；
   改善/退化分子 204/216，bootstrap 区间跨零，去尾敏感性反而更差。预测分布确实变了（相关 0.952），
   但没有变成"更准"。
3. **是否超过历史 C1？**
   没有。0.7274001 vs C1 的 0.7086243，差 **+0.018776**。也没有接近 GINE 的 0.6780750。
4. **低训练支持分子是否受益？**
   没有可确认的受益。低支持组 ΔRMSE −0.0028（bootstrap 跨零、MAE 反而 +0.024），高支持组 +0.0236。
   唯一的读法是"低支持组没有像高支持组那样退化"，不足以支持机制结论。
5. **能支持什么解释，仍不能支持什么解释？**
   - 能支持：在**这个冻结架构 + 冻结配方 + 这一个 scaffold split** 下，把训练骨架家族的总权重拉平
     不足以改善官方 Valid；这个干预是可以被检出的（预测/权重都实际改变了），且不是优化异常导致；
     权重换来的拟合改善集中在被加权的小 scaffold 上（普通等权 Train RMSE 反而差 +0.056），
     即"训练目标改变"没有转化为"scaffold split 外推更好"。
   - 不能支持：不能证明"低训练支持是上一轮预测失败的因果机制"（低支持组没有系统性改善）；
     不能证明"骨架均衡学习无效"（只有 1 个 seed、1 个权重公式、1 个架构、1 个划分、40 min 预算、
     无超参搜索；0.03 是资源参考不是显著性阈值）；不能把 0.0007 的差异读成"结构分布干预有害"；
     不能对 Test 作任何陈述（未读取、未评分）。

## 9. 预算与交付

| 项目 | 值 |
|---|---|
| 执行位置 | 本地 CPU（`res`/`res-2` 均被占用；本机无 CUDA 设备） |
| GPU·h | **0**（硬上限 0.25 未使用） |
| 排队 | 0 min（上限 5 min）；无远端队列 |
| 正式训练墙钟 | 565.06 s（9.4 min）；train 387.2 s + eval 175.0 s |
| 全流程墙钟（含检查/分析/排错） | ≈ 19 min（上限 40 min） |
| 峰值内存 | 预检实测 ≈ 681 MB RSS（同一 bundle + 模型 + 真实 batch），无 OOM 风险 |
| 新增训练单元 | 1（`SCAFFOLD_BALANCED`，seed 0）；无其他数据集、无超参搜索、无第二 seed |
| Test | 未读取、未评分、未进入选择 |
| 代码基座 | `d2d9c04d574c600af9bd73c6a783416f5bb71dbc`（冻结与训练时仓库 HEAD；本轮所有文件当时未提交，`git_dirty=1` 已记录在 `run.json`） |
| 环境 | Python 3.12.15、torch 2.5.1+cu124（CPU）、numpy 2.1.3、8 线程 |

交付物：

- `PROTOCOL.md`（训练前冻结）、`configs/experiment.json`（机器可读冻结件）
- `code/`：`sb_common.py`、`sb_scaffold.py`、`sb_data.py`、`sb_freeze.py`、`sb_checks.py`、
  `sb_train.py`、`sb_analyze.py`、`run_all.py`
- `runs/scaffold_balanced_seed0/`：`train_curve.csv`、`checkpoint_best.pt`、`checkpoint_final.pt`、
  `predictions_{best,final,last20}.npz`、`best_checkpoint_eval.json`、`training_order.json`、`run.json`
- `analysis/`：`checks.json`、`preflight_timing.json`、`post_checks.json`、`scaffold_distribution.json`、
  `scaffold_weights.csv`、`valid_scaffolds.json`、`weight_summary.json`、`weight_conventions.json`、
  `main_table.csv`、`valid_curve.csv`、`curve_statistics.json`、`epoch_aligned_train.json`、
  `per_molecule_valid.csv`、`paired_comparisons.json`、`support_strata.{json,csv}`、
  `fixed_cases.{json,csv}`、`training_diagnostics.json`、`historical_reference.json`、
  `verdict.json`、`freeze_log.json`、`engineering_log.md`
- `results/manifest.json`（checkpoint 与预测的逐文件 SHA256）

已知局限（如实记录）：

- 本轮与历史 FIXED1 的训练设备不同（CPU vs A100）。同一历史 checkpoint 在本机重载复现其记录的
  Valid RMSE 到 3.9e-9，说明数据/模型/评价链路等价；但逐位相同的训练轨迹不可能（浮点归约顺序不同）。
- best 值是 420 个分子上单一 epoch 的顺序统计量，曲线抖动（std ≈ 0.010）远大于 best 差异（0.0007）；
  因此结论依靠三条指标（best/endpoint/last-20）的**同方向**与 late-horizon 均值，而不是单个 best 值。
- Valid 在本仓库被反复使用；单 seed；0.03 是资源分配参考，不是统计显著性标准。

本轮到此停止。
