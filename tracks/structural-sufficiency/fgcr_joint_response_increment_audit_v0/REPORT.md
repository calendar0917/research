# FGCR 联合结构响应增量验证 — REPORT

Round: `fgcr-joint-response-increment-audit-v0`  
Status: **PASS**（17/17 必需核验通过；CPU；墙钟约 7 s；峰值 RSS 0.63 GB）  
Decision: **`JOINT_DEPENDENCE_SIGNAL`（依赖集中在 eta，H 接近局部充分）**

本报告只陈述冻结 checkpoint 的重放、精确响应分解和冻结 head 替换结果。没有训练、
微调、优化器 step、超参数搜索或 Test 评分。所有 `LOCAL_*` 数字都是**冻结替换**，
不是重新训练后的模型性能。

---

## 0. 首页：四个问题

### 1. 历史 FGCR 是否实际依赖全局联合增量？

**是，但分通道。** 冻结 head 对 `eta` 的联合增量依赖很强，对 `H` 的联合增量
依赖很小：

| Valid 条件 | H 输入 | eta 输入 | RMSE | 相对 FULL |
|---|---|---|---:|---:|
| FULL（历史重放） | H_FULL | ETA_FULL | **0.712894** | — |
| LOCAL_H | H_IND | ETA_FULL | 0.719405 | +0.006511（+0.91%） |
| LOCAL_ETA | H_FULL | ETA_STAR | 0.938596 | +0.225702（+31.66%） |
| LOCAL_BOTH | H_IND | ETA_STAR | 0.952930 | +0.240036（+33.67%） |

移除 H 的联合增量只损失 0.9% 的 Valid RMSE；移除 eta 的联合增量损失 31.7%。
`LOCAL_BOTH` 的退化不是单个困难分子造成的（见 §3）：62.1% 的 Valid 分子变差，
去掉双方共同平方误差最大的 5% 后 ΔRMSE 仍为 +0.2041，2000 次配对 bootstrap 的
95% 区间为 [+0.1691, +0.3080]。

### 2. 这种增量主要存在于 H 还是 eta？

**主要存在于 eta。**

| Valid 量 | FULL RMS | LOCAL RMS | INCREMENT RMS | INC/FULL |
|---|---:|---:|---:|---:|
| H | 6.0613 | 5.8559 | 1.0380 | **0.1712** |
| ETA（raw） | 1.1632 | 1.0988 | 0.5947 | **0.5112** |
| ETA（过历史 eta_norm 后） | 1.1178 | 1.1248 | 0.6457 | **0.5777** |

`H_IND` 与 `H_FULL` 的分子级余弦中位数为 0.988；`ETA_STAR` 与 `ETA_FULL` 的
原子级余弦中位数只有 0.874。`eta_norm` 还把这个相对增量从 0.511 放大到 0.578。
因此历史模型实际使用的独特联合信息主要在 eta 通道，而不是 H 通道。

### 3. 局部替换对官方预测造成什么影响？

- **H 的局部边和参考基本保持性能**：Valid RMSE 0.719405 vs 0.712894，Δ=+0.006511
  （+0.91%），MAE +0.004250；bootstrap 95% ΔRMSE 区间 [+0.000568, +0.012973]。
  小但非零；Train 上 Δ=+0.012827（+2.72%），说明 H 增量并非完全无用。
- **eta 的局部星形参考明显破坏性能**：Valid RMSE 0.938596，Δ=+0.225702
  （+31.66%），MAE +0.178648；bootstrap 区间 [+0.159218, +0.294025]；
  62.4% 分子变差；去掉最难 5% 后 Δ=+0.197775。
- **两者同时替换最差**：Valid 0.952930，Δ=+0.240036（+33.67%）。
- 没有任何冻结替换降低误差；`LOCAL_REPLACEMENT_IMPROVES` 不成立。

逐分子预测变化 RMS（raw 单位）：LOCAL_H 0.0734，LOCAL_ETA 0.6287，
LOCAL_BOTH 0.6513；最大绝对变化分别为 0.369 / 2.513 / 2.531。

### 4. 有无值得继续研究的证据？

有一个**具体的机制性证据**：历史冻结 head 并非把 eta 当作可由独立邻居边缘化
复现的局部量；它实际使用了 eta 的联合增量。可是这**不证明** JOINT 架构优于
重新训练的 LOCAL 模型，也不证明泛化收益：历史 FULL 本身 Valid 0.712894，仍
高于描述性 GINE 参考 0.61–0.64，且 Train–Valid gap 很大。若继续，唯一有价值的
问题是“重新训练的 LOCAL/星形模型能否补齐这一 eta 信息”，而不是直接宣称当前
JOINT 架构更好。本轮不设计、不启动下一代模型。

---

## 1. 合规、资产与重放

| 项 | 值 |
|---|---|
| 数据 | 官方 Lipo Train 3360 / Valid 420；**未计算任何 Test 指标** |
| Test 使用 | 无 Test 行进入任何 forward、响应、预测、metric、归一化、bootstrap 或判定；plan 只在 Train+Valid 的 3780 张图上构建 |
| 数据加载说明 | 历史 `sq_data` 缓存是单体 npz；历史 loader 会为文件级审计读取整个缓存，但本审计只索引 Train/Valid 行 |
| checkpoint | `results/seed0/best.pt`，epoch 96，SHA-256 `b62d9cd6…` |
| 历史 Valid RMSE | 0.7128937193209034 |
| 重放 Valid RMSE | **0.7128936104001214**（绝对差 1.09e-7，容差 1e-5） |
| 重放 Train RMSE / MAE | 0.4723507802 / 0.3566524196 |
| 重放 Valid MAE | 0.5387833003 |
| 与历史 `predictions_best.npz` 最大绝对差 | Valid 1.23e-5，Train 8.09e-6（容差 1e-4） |
| 模式 | `model.eval()`，无 dropout；CPU，4 threads，batch 256 |
| 训练 | 无；optimizer steps = 0；历史文件未修改 |

---

## 2. 分解、实现与核验

定义（逐通道；`log p_i = log_softmax(u_i)`）：

```
H_FULL      = logZ - sum_i logsumexp(u_i)
H_IND       = sum_e logsumexp_{s,t}(log p_i(s)+log p_j(t)+psi_ij(s,t))
H_INCREMENT = H_FULL - H_IND

ETA_FULL_i      = posterior_logodds_i - unary_logodds_i
ETA_STAR_i      = sum_{j in N(i)} [m_{j->i}(1)-m_{j->i}(0)]
m_{j->i}(s)     = logsumexp_t(log p_j(t)+psi_ij(s,t))
ETA_INCREMENT_i = ETA_FULL_i - ETA_STAR_i
```

实现直接复用历史 `potentials(batch)`、`fgcr_inference.exact_responses`、
sentence MLP、`eta_norm` LayerNorm、readout 和标量 bias。替换只改
`forward` 内部从一次精确推断得到的响应张量；其余代码和全部参数不动。

| 核验 | 结果 |
|---|---|
| 单边图、两个不相连单边图 FULL = 局部参考 | 通过；float64 差 0 |
| 零边势：H、eta、增量全零 | 通过；8 个零势例差 0 |
| 链、星、4-环、多组分 + 孤立原子 vs 穷举 | 通过；`logZ`、H、eta 最大差 0（float64） |
| 原子重编号 / 边方向反转 / 边存储顺序 | 通过；float64 最大差 8.9e-16 / 0 / 1.7e-16 |
| 真实模型（3 个 Valid 分子，同时重编号+反转+乱序） | 通过；float32 最大差 2.29e-5（容差 1e-4） |
| 每条物理边只计一次 | 通过；Valid 11,467 atoms / 12,585 bonds，Train 90,514 / 98,561，全部唯一 canonical 对；独立 Python 循环 vs 向量化 H_IND 最大差 2.26e-6 |
| `FULL = LOCAL + INCREMENT` 重建 | 通过；真实数据最大残差 0 |
| 替换确实注入目标 H/eta | 通过；最大差 3.8e-6（float32 重建） |
| LOCAL_BOTH 不执行完整 belief pass | 通过；belief calls = 0 |
| 原始 forward 预测复现 | 通过；见 §1 |

特别注意：**一般树形图不保证 H_INCREMENT = 0**。随机势的 chain4 上
`max|H_INCREMENT| = 0.1907`、`max|ETA_INCREMENT| = 0.5150`；本审计没有把树
当作零增量情形。

---

## 3. 四种冻结替换的完整结果

`main_table.csv` 给出全部列（含 bootstrap、trim、top-5% 份额）。核心结果：

| Split | 条件 | RMSE | MAE | ΔRMSE | Δ% | ΔMAE | pred-change RMS | 分子变差比例 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Valid | FULL | 0.712894 | 0.538783 | — | — | — | — | — |
| Valid | LOCAL_H | 0.719405 | 0.543033 | +0.006511 | +0.91% | +0.004250 | 0.073419 | 216/420 = 51.4% |
| Valid | LOCAL_ETA | 0.938596 | 0.717431 | +0.225702 | +31.66% | +0.178648 | 0.628667 | 262/420 = 62.4% |
| Valid | LOCAL_BOTH | 0.952930 | 0.727745 | +0.240036 | +33.67% | +0.188962 | 0.651293 | 261/420 = 62.1% |
| Train | FULL | 0.472351 | 0.356652 | — | — | — | — | — |
| Train | LOCAL_H | 0.485177 | 0.368403 | +0.012827 | +2.72% | +0.011751 | 0.072863 | 1881/3360 = 56.0% |
| Train | LOCAL_ETA | 0.814846 | 0.617332 | +0.342495 | +72.51% | +0.260679 | 0.593991 | 2364/3360 = 70.4% |
| Train | LOCAL_BOTH | 0.835501 | 0.633258 | +0.363150 | +76.88% | +0.276606 | 0.618008 | 2347/3360 = 69.9% |

2000 次配对分子 bootstrap（2.5–97.5 百分位）：

| Split | 条件 | mean ΔSE（平方误差差，cond−FULL） | ΔRMSE 区间 | resamples 中变差比例 |
|---|---|---:|---|---:|
| Valid | LOCAL_H | +0.00933 [0.00083, 0.01888] | [+0.00057, +0.01297] | 98.3% |
| Valid | LOCAL_ETA | +0.37274 [0.25892, 0.49467] | [+0.15922, +0.29403] | 100% |
| Valid | LOCAL_BOTH | +0.39986 [0.28385, 0.52314] | [+0.16913, +0.30803] | 100% |
| Train | LOCAL_H | +0.01228 [0.00984, 0.01459] | [+0.01032, +0.01518] | 100% |
| Train | LOCAL_ETA | +0.44086 [0.40855, 0.47707] | [+0.32135, +0.36402] | 100% |
| Train | LOCAL_BOTH | +0.47495 [0.43828, 0.51263] | [+0.34061, +0.38618] | 100% |

去除双方共同平方误差最大的 5%（按 `(SE_FULL+SE_cond)/2` 排序；Valid 去掉 21 个，
Train 去掉 168 个）后：

| Split | 条件 | ΔRMSE（trim-5%） | ΔRMSE（按 max(SE) trim-5%） |
|---|---|---:|---:|
| Valid | LOCAL_H | +0.008976 | +0.008555 |
| Valid | LOCAL_ETA | +0.197775 | +0.171200 |
| Valid | LOCAL_BOTH | +0.204083 | +0.186548 |
| Train | LOCAL_H | +0.013353 | +0.012953 |
| Train | LOCAL_ETA | +0.262575 | +0.243610 |
| Train | LOCAL_BOTH | +0.278973 | +0.257562 |

对 LOCAL_ETA / LOCAL_BOTH，正 ΔSE 最大的 5% 分子只占总正 ΔSE 的约 40%，其余
60% 来自其他 95% 分子；退化是广泛的，不是单个困难分子。LOCAL_H 的总效应小，
但其方向在 Valid 上仍有 98.3% 的 bootstrap resamples 为正。

---

## 4. 真实联合增量诊断

### 4.1 H 与 eta 的尺度

| Split | 量 | FULL RMS | LOCAL RMS | INC RMS | INC/FULL | 增量通道比范围 |
|---|---|---:|---:|---:|---:|---|
| Valid | H | 6.0613 | 5.8559 | 1.0380 | 0.1712 | 0.021–0.509 |
| Valid | ETA raw | 1.1632 | 1.0988 | 0.5947 | 0.5112 | 0.072–0.709 |
| Valid | ETA norm | 1.1178 | 1.1248 | 0.6457 | 0.5777 | 0.397–0.754 |
| Train | H | 5.7602 | 5.5327 | 1.0253 | 0.1780 | 0.024–0.556 |
| Train | ETA raw | 1.1663 | 1.0978 | 0.5964 | 0.5113 | 0.074–0.709 |
| Train | ETA norm | 1.1178 | 1.1246 | 0.6487 | 0.5803 | 0.402–0.739 |

`eta_norm` 把增量的相对尺度从 0.511 放大到 0.578（norm/raw = 1.086）。它不是
把增量压小，而是略微放大。

### 4.2 分子间变化与常数偏移

- **H_INCREMENT**：分子级 RMS 中位数 0.849（Valid q05 0.397，q95 1.618，
  max 4.197，min 0.258），CV = 0.464。通道均值向量 RMS 0.691，通道跨分子
  标准差向量 RMS 0.774；即有可观的通道常数偏移，也有跨分子变化。
- **ETA_INCREMENT**：分子级 RMS 中位数 0.578（q05 0.446，q95 0.745，
  max 0.954，min 0.380），CV = 0.146。通道均值向量 RMS 0.217，通道跨分子
  标准差向量 RMS 0.554；以变化为主，常数偏移较小。
- **ETA_norm_INCREMENT**：通道均值向量 RMS 0.222，跨分子标准差向量 RMS 0.607；
  LayerNorm 后仍是变化为主。
- 近零比例：H 与 eta 的增量分子级 RMS 都 **没有** 近零分子（阈值 1e-6）。
  唯一零增量的原子是 Valid 中分子内没有键的 3 个孤立原子，这符合定义。

### 4.3 增量与大小、局部响应的关系（Valid；Train 同量级）

| 目标 | 与 log(1+n_atoms) | 与 log(1+n_bonds) | 与局部响应 | 与 ETA_FULL RMS |
|---|---:|---:|---:|---:|
| H_INC RMS | 0.681 | 0.668 | H_IND 0.522 | — |
| H_INC RMS / (1+n_bonds) | 0.026 | 0.012 | H_IND −0.045 | 0.752 |
| ETA_INC RMS | 0.245 | 0.255 | ETA_STAR 0.526 | **0.814** |

解释（描述性，不是因果）：

- H_INCREMENT 的**总量**随分子大小增长（与键数相关 0.668），但按物理键归一化后
  与大小几乎无关（Valid: 0.01–0.03），而与 eta 全响应强度相关 0.752。描述性地说，
  H 增量的 extensive 部分主要由键数/分子大小解释，不是单纯原子数效应；按键的
  闭合修正仍与相互作用强度相关，但这不是因果分解。
- ETA_INCREMENT 与分子大小只有弱相关（0.25），与 ETA_STAR 相关 0.526，与
  ETA_FULL 相关 0.814。它更像是对总场强度的修正，而不是大小或普通局部响应的
  简单复制。
- 这些都是相关性；不能据此声称“增量只由局部量决定”或“增量包含独特信息”。

### 4.4 eta_norm 前后的分布

Valid：

- eta_norm 前后 pooled RMS：1.1178（full）、1.1248（star）、0.6457（increment）；
- 每原子范数中位数 4.469（full）vs 4.492（star），没有尺度爆炸或退化；
- 每原子余弦（full vs star）中位数 0.874，q05 0.549，min −0.413；
- 通道均值平移最大 0.394；通道标准差比（star/full）范围 0.745–1.149；
- `rms(eta_norm(x)-x)/rms(x)`：full 0.424，star 0.411。

因此 LOCAL_ETA 的替换不是“把输入放大/缩小一个数量级”那种明显的工程性分布
破坏；它是一个同尺度、同范数、通道结构部分对齐的局部场。但它确实改变了输入
分布：star 与 full 的 eta_norm 差异是 full RMS 的 **0.578**。这个差异正是被
检验的联合增量本身，而不是重新拟合的归一化参数。这一点在判读时保留。

---

## 5. 判读与边界

**`JOINT_DEPENDENCE_SIGNAL`，且依赖集中在 eta。**

支持：

1. 移除 eta 联合增量后 Valid RMSE +31.7%，Train +72.5%，bootstrap 区间远离 0；
2. 退化广泛（Valid 62.4% 分子变差；去掉最难 5% 后仍有 +0.198）；
3. eta_norm 替换前后尺度相同（范数 4.47 vs 4.49），不是数值退化；
4. LOCAL_H 是同一替换管线的内部对照：它只造成 +0.9% Valid 变化，说明替换机制
   本身不会自动摧毁预测；模型对 eta 内容的敏感性是特异且更强的。

不支持的类别：

- **不是 `LOCAL_SUFFICIENT_FOR_FROZEN_MODEL`**：H 通道近似充分，但 eta 通道
  不充分；整体不能用“局部足够”概括。
- **不是 `LOCAL_REPLACEMENT_IMPROVES`**：四个条件中没有任何一个冻结替换降低
  Valid 或 Train 误差；LOCAL_H 也略差。
- **不是 `INCONCLUSIVE`**：资产、哈希、重放、穷举、零势、不变性、边计数、
  重建、注入目标全部通过；局部参考与完整响应同尺度；退化不是由单个分子或
  单个通道的数值异常造成。唯一保留的保留意见是：冻结 head 只在 ETA_FULL 上
  训练过，因此 0.9386 同时包含“依赖联合信息”和“对替换输入的敏感性”；本轮
  不能把两者完全分离。

需要明确的矛盾/限制：

- H 的联合增量很小（17% RMS），但并非严格零：LOCAL_H Valid Δ=+0.0065，
  bootstrap 区间 [+0.0006, +0.0130]，Train Δ=+0.0128。因此“H 局部充分”是
  近似，不是精确等式。
- eta 替换的输入分布变化较大（eta_norm 差异 58% of full RMS）。这不是一个
  “微小扰动”实验；它证明冻结模型依赖 eta 的联合增量，但不能证明重新训练的
  LOCAL 模型一定差。
- 本轮没有 Test 指标、没有重新训练、没有第二 seed、没有架构搜索；不能把
  LOCAL_* 的差异解释为泛化收益或架构优劣。
- 较小的增量尺度不等于不重要：eta 增量只有 51% of full RMS，却造成 31.7%
  Valid RMSE 退化；H 增量 17%，只造成 0.9%。较大的预测变化也不等于泛化收益：
  没有重新训练，且历史 FULL 本身仍不在 GINE 竞争区间。

---

## 6. 复现与资源

```bash
cd /home/calendar/code/research
uv run python tracks/structural-sufficiency/fgcr_joint_response_increment_audit_v0/code/run_audit.py
```

输出：

| 文件 | 内容 |
|---|---|
| `checks.json` | 17 项核验、重放、穷举/不变性/边计数/注入目标、预算与合规 |
| `main_table.csv` | 4 条件 × Valid/Train：RMSE/MAE、Δ、bootstrap、trim-5%、逐分子变化 |
| `increment_statistics.json` | H/ETA/eta_norm 的通道均值/标准差/RMS、尺度、分子间分布、相关性、输入漂移 |
| `predictions_by_condition.csv` | 3,780 行逐分子真值、四条件预测、SE/ΔSE、H/ETA 增量 RMS |
| `provenance.json` | commit、配置 SHA、资产 SHA、环境、墙钟、峰值 RSS |
| `PROTOCOL.md` | 运行前冻结的分析定义 |

资源：CPU，4 threads，batch 256；墙钟 **约 7 s**（硬上限 1,200 s），峰值 RSS
**0.63 GB**；无 GPU、无 OOM 风险。实现核验的 float64 穷举在 0.04 s 内完成。
历史 checkpoint、配置、数据和结果均只读；无 Test 评分；无训练。

**停止。** 不启动新训练，不设计下一代架构。
