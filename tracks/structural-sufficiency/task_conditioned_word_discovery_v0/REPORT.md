# Task-conditioned structural vocabulary discovery v0 — REPORT

**Round**: `tracks/structural-sufficiency/task_conditioned_word_discovery_v0/`  
**Protocol**: `task-conditioned-word-discovery-v0` (`configs/tc.json`, frozen before formal GPU runs)  
**Execution commit**: `ae32cc85b356ab0e298c53082a768d409f795037`  
**Data**: `ogbg-mollipo` official Train only (3360 molecules), frozen D0/D1 Fit/Dev splits. Official Valid labels and Test rows were never used.  
**Date**: 2026-11-24 (UTC+08:00)

## 0. 结论

**Verdict: `NO_SUPPORTED_DISCOVERY`.**

在 D0/s0 的冻结比较中，任务条件词表没有优于同规模、同 size/granularity/DF-quartile 分布的频率词表：

| D0/s0 arm | Dev RMSE | Dev MAE | 相对 R2_REF |
|---|---:|---:|---:|
| BASE（无词） | 0.777429 | 0.612365 | +0.229750 |
| **D_TASK**（任务词表） | **0.734091** | **0.563584** | **+0.186412** |
| **D_FREQ**（频率词表） | **0.703492** | **0.548521** | **+0.155813** |
| R2_REF（描述性参照） | 0.547679 | 0.417100 | — |

- 点估计上，`D_TASK` 比 BASE 好 **0.043338 RMSE**（95% CI 跨 0），提示选出的词并非完全无信息；
- 但 `D_FREQ` 比 BASE 好 **0.073937 RMSE**（95% CI 不跨 0），且比 `D_TASK` 好 **0.030598 RMSE**（95% CI 跨 0）；
- `D_TASK` 距 R2_REF 仍有 **0.186412 RMSE**，远未达到“保留强 R2 预测能力”的 0.025 邻近标准；
- R2_REF 取自 `structure_route_resolution_8h_v1/results/R2_REF_D0_S0`，其 Dev rows hash 与本轮相同（`eacce8ac…87193`）；这里只作性能坐标，不作架构归因。
- 冻结晋级规则因此没有触发 Stage 2，也没有执行 D0/s1、D1 神经验证或 Stage 3。  
  这是一个**按协议停止**的负面结果，不是“所有任务条件词表都不可能有效”的普遍结论。

主要解释不是“任务词没有信号”，而是：**选出的任务词在根级覆盖上极度稀疏**。D0 的 `D_TASK` 只覆盖 **22.5%** 的 Fit 根，频率词表覆盖 **63.5%**；每个根平均有效词数 **0.356 vs 3.148**。在 K=128、单种子、504 个 Dev 分子的条件下，任务词的弱条件相关信号被频率词表的覆盖优势压过。

---

## 1. 问题与冻结判定

本轮要回答四个问题：

1. 从真实 R2 连通诱导子图里，能否学出紧凑、共享、任务条件的结构词表？
2. 它是否比同规模频率词表更有预测力？
3. 它能否保留强 R2-SENTENCE 的预测能力？
4. 选出的词是否只是低阶结构的重复？

冻结判定规则（`configs/tc.json`）：

| 判定 | 条件 |
|---|---|
| `VOCABULARY_PROMISING` | D0 两个 seed 的 TASK-vs-FREQ 方向都为正、均值 ≥0.01，且 D0/D1 的 TASK 模型都在各自 R2_REF 的 0.025 RMSE 内 |
| `DISCOVERY_ONLY` | 任务词表优于频率词表，但没有保留强 R2 性能 |
| `COMPRESSION_ONLY` | 紧凑词表接近 R2 性能，但任务选择没有优于频率选择 |
| `NO_SUPPORTED_DISCOVERY` | 以上信号都没有观察到 |
| `INSUFFICIENT_OR_INVALID` | 支撑不足、正对照失败或实现不一致 |

本轮支撑充分（K=128、4683 个去重候选、正对照通过、39/39 检查通过），因此不适用 `INSUFFICIENT_OR_INVALID`。

---

## 2. 协议执行摘要

### 2.1 数据与切分

| 项 | D0 | D1 |
|---|---:|---:|
| official Train | 3360 | 3360 |
| Fit | 2856 | 2856 |
| Dev | 504 | 504 |
| Fit scaffolds | 1354 | 1318 |
| Dev scaffolds | 231 | 231 |
| split identity sha256 | `60ac13c0…2559e` | `c94e2d4b…b6bee` |
| Dev rows sha256 | `eacce8ac…87193` | `342c1de0…380a6` |

标签只用当前 split 的 Fit 部分做 z-score；Dev 指标在原始标签单位报告。官方 Valid 标签保持 NaN，Test 行从未进入枚举、选择或训练。

### 2.2 候选枚举

- 枚举 R2(root) 内所有 **size 1–4** 的 rooted connected induced subgraph；
- 根固定在 slot 0，包含全部真实内部键；
- 用 `rice_structure.canonical_signature` 做精确的 root-marked attributed-graph 规范化；
- size 3/4 同时有 **FULL**（9 个 OGB atom 字段 / 3 个 bond 字段）和 **COARSE**（atomic_num、formal_charge、is_aromatic / bond_type）两种粒度；
- 只保留 Fit 中 ≥20 个分子且 ≥5 个 scaffold 的类型；
- 对完全相同的 Fit 根级计数列去重；代表元优先更小的 Fit document frequency，再 COARSE，再 canonical key 升序；
- size 1/2 FULL 类型只进入低阶控制 `X0`，不进入任务/频率词表。

### 2.3 CPU 选择

- 3 折 Bemis-Murcko scaffold 分组，按 scaffold 整组分配；
- ridge `alpha=10`；
- 每个内折：`y ~ X0`，`X_words ~ X0`，在 held-out 部分计算残差 Pearson 相关；
- 选择分数 = 三个内折 `|corr|` 的最小值；要求三折符号严格一致；
- 按分数降序、canonical key 升序取 K=128；
- 频率控制：在 `(size, granularity, DF-quartile)` 的 12 个联合单元格中按相同数量抽取，单元格内按 Fit document frequency 降序、key 升序；
- 任务/频率词表允许相交，实际交集如实报告。

### 2.4 神经验证

所有 arm 共用同一个非迭代、根级 readout：

- 根原子 embedding 48 维；
- 一阶真实邻居原子 + incident bond 经共享 pair encoder `[72]→128→64` 求和；
- 词向量 `z_v[t]=log1p(c(v,t))` 经 `K→128→64`；
- 拼接 `48+64+64=176`，经 `176→192→96`（SiLU，dropout 0.1）；
- 每分子对 96 维根表示取 sum/mean/std，拼接 `log1p(n_atoms)` 和 Fit 标准化的 8 个图标量，得到 `297`，经 `297→192→96→1`；
- 无 GNN、无预训练、无 teacher、无蒸馏、无 R2 trunk bypass、无 S2 分支、无额外 R2 统计。

冻结 recipe：AdamW，peak LR 3e-4，min LR 1e-5，cosine，wd 1e-4，clip 5，batch 64，warmup 5，最多 240 epochs，patience 45，min epochs 80，按 Dev RMSE 选 best checkpoint。没有超参数搜索。

### 2.5 实际运行矩阵

| Run | Split | Seed | Arm | 状态 |
|---|---|---|---:|---|
| `D0_S0_BASE` | D0 | 0 | BASE | 完成 |
| `D0_S0_TASK` | D0 | 0 | D_TASK | 完成 |
| `D0_S0_FREQ` | D0 | 0 | D_FREQ | 完成 |
| `D0_S1_*` | D0 | 1 | TASK/FREQ | **未执行**（Stage 2 未触发） |
| `D1_S0_*` | D1 | 0 | BASE/TASK/FREQ | **未执行**（Stage 3 未触发） |

D1 的 CPU 候选、选择和 ridge 已预先计算，仅作为跨 split 稳定性/方向检查，不参与神经判定。

---

## 3. 结果

### 3.1 支撑、完整性与实现检查

`analysis/checks.json`：**39/39 PASS**。`tests/test_tc.py`：**6 passed**。

覆盖内容包括：

- 7 个 frozen parent source 的 sha256 pin；
- D0/D1 split identity、Fit/Dev 行哈希；
- Valid 标签 NaN、无 Test 行；
- 候选计数、阈值、size、去重成员、列哈希、折分配；
- K、频率联合分布、任务方向一致性、正对照、ridge arms；
- 神经侧：参数数相等、初始化相等、维度、数据 K、词表 key、BASE 忽略词、TASK 词宽、R2 recipe、合成标签不进入真实训练。

候选与支撑：

| 项 | D0 | D1 |
|---|---:|---:|
| 枚举类型总数 | 32,966 | 32,440 |
| FULL_1 / FULL_2 | 74 / 1,014 | 73 / 987 |
| FULL_3 / COARSE_3 | 7,251 / 620 | 7,161 / 595 |
| FULL_4 / COARSE_4 | 21,772 / 2,235 | 21,496 / 2,128 |
| 达到 df/scaffold 阈值 | 5,035 | 5,101 |
| 完全相同列去重后 | 4,683 | 4,711 |
| 被去重吸收的类型 | 352 | 390 |
| `X0` 列数 | 1,098 | 1,070 |
| 3 折大小 | [949, 965, 942] | [951, 951, 954] |
| K task / freq | 128 / 128 | 128 / 128 |
| task∩freq | 8 | 3 |

### 3.2 选出的任务词

D0 的 128 个任务词 size/gran 分布：

| 粒度 | size 3 | size 4 |
|---|---:|---:|
| FULL | 42 | 64 |
| COARSE | 4 | 18 |

D1 为 41 / 3 / 62 / 22（FULL_3 / COARSE_3 / FULL_4 / COARSE_4）。频率词表在每个 split 上按构造具有完全相同的 size/gran/DF-quartile 联合分布。

选择分数是**残差相关的最小绝对值**，不是因果贡献。D0 最高分 0.1692、最低 0.0837；D1 最高 0.1536、最低 0.0792。大多数词的三折相关在 0.08–0.4 之间，方向一致但幅度弱。

D0 前几个词（完整 20 个见 `analysis/top_words.md`）：

| Rank | Size/gran | df | scaffold df | score | 三折相关 | 解码模式（`*` = root） |
|---:|---|---:|---:|---:|---|---|
| 0 | 3 FULL | 28 | 15 | 0.1692 | +0.447, +0.205, +0.169 | `*C(deg4,H2,sp3,ring) C(deg4,H1,sp3,ring) N(deg3,H0,sp2,ring)`；两条 single bond |
| 1 | 4 FULL | 88 | 32 | 0.1626 | +0.265, +0.165, +0.163 | `*C(arom) C(arom,H1) N(arom) N(sp2,ring)`；芳香 C–N / C–N 单键 |
| 2 | 3 FULL | 28 | 15 | 0.1604 | +0.439, +0.199, +0.160 | `*N(sp2,ring) C(deg4,H2,sp3,ring) C(deg4,H1,sp3,ring)`；N–CH2–CH |
| 5 | 3 FULL | 26 | 15 | 0.1464 | −0.408, −0.159, −0.146 | `*C(deg4,H2,sp3,ring) C(deg4,H1,sp3,ring) C(deg4,H1,sp3,ring)`；饱和三碳片段 |
| 6 | 3 FULL | 96 | 54 | 0.1400 | +0.146, +0.150, +0.140 | `*C(arom,H1) C(sp2,ring) C(arom)`；芳香/非芳香连接 |
| 9 | 4 FULL | 41 | 34 | 0.1381 | −0.138, −0.216, −0.147 | `*C(arom) C(sp2) C(arom) S(sp3,deg1)`；芳基–S 片段 |

这些模式在化学上可读（哌嗪/哌啶型 N–CH2–CH、氨基吡啶型芳香 C–N–N、芳基硫/磺酰基邻域、饱和碳环片段等），但**不能**把选择分数解释为“该片段导致亲脂性”。它只是 Fit 内折残差相关；D0/D1 的具体词差异很大。

### 3.3 Ridge arms（CPU 线性检查）

Ridge 的 Dev RMSE（原始标签单位）：

| Split | LOW (X0) | ALL | TASK | FREQ | TASK−FREQ |
|---|---:|---:|---:|---:|---:|
| D0 | 0.862685 | 0.897710 | **0.833173** | 0.864183 | **−0.031010**（TASK 更好） |
| D1 | 0.972795 | 0.903233 | 0.911526 | **0.901093** | **+0.010433**（TASK 更差） |

- D0 上 TASK ridge 比 FREQ 好 0.0310，和神经结果方向相反；
- D1 上 TASK ridge 比 FREQ 差 0.0104，和神经 D0 方向一致；
- 两个 split 的方向不一致，说明任务选择信号本身不稳定，不能把 D0 ridge 的局部优势当作可复制发现；
- ALL ridge 在 D0 明显过拟合（Fit RMSE 0.247，Dev 0.898），D1 则较好（Fit 0.250，Dev 0.903），进一步说明 504 个 Dev 分子的估计噪声不小。

### 3.4 正对照

植入一个不看真实标签选出的 size-3/4 词，用其 Fit 标准化计数生成合成标签，走完全相同的选择流程：

| Split | 植入词进入 TASK 词表 | 合成 TASK ridge Dev RMSE | 合成 FREQ ridge Dev RMSE | 判定 |
|---|---|---:|---:|---|
| D0 | 是 | **0.04918** | 1.02582 | PASS |
| D1 | 是 | **0.02286** | 1.24929 | PASS |

合成标签只用于这个对照，**从未进入真实 lipo 训练**（check D9）。因此，真实标签上的阴性结果不是“选择器连植入信号都找不到”。

### 3.5 神经 Stage 1 主结果

所有 arm 参数数 **179,385**，初始化哈希相同，batch 顺序和优化 recipe 相同。峰值显存约 **52.8–56.5 MB**。

| Run | Arm | Fit RMSE | Dev RMSE | Dev MAE | Best epoch | Epochs run | Wall (s) | GPU h |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `D0_S0_BASE` | BASE | 0.554007 | 0.777429 | 0.612365 | 187 | 232 | 223.7 | 0.06213 |
| `D0_S0_TASK` | D_TASK | 0.476196 | 0.734091 | 0.563584 | 207 | 240 | 251.6 | 0.06988 |
| `D0_S0_FREQ` | D_FREQ | 0.513543 | 0.703492 | 0.548521 | 153 | 198 | 208.8 | 0.05801 |

三个 run 的 `wall_seconds` 合计 **684.1 s**；正式阶段（两卡并行 + 第三个 run）约 **11.5 min**；GPU **0.1900 h**。

逐 epoch 的 Fit/Dev RMSE/MAE、学习率和 early-stop 记录全部在 `results/<run>/curves.csv`；没有平滑、截断或只保留最好点。

关键差值（负值表示前者更好）：

| 比较 | ΔRMSE | 95% scaffold bootstrap CI | ΔMAE | 95% CI | 前者绝对误差更小的分子比例 |
|---|---:|---|---:|---|---:|
| TASK − FREQ | **+0.030598** | [−0.045577, +0.100169] | +0.015063 | [−0.049442, +0.076754] | 48.6% |
| TASK − BASE | **−0.043338** | [−0.106367, +0.020118] | −0.048781 | [−0.103081, +0.006803] | 53.2% |
| FREQ − BASE | **−0.073937** | [−0.135528, −0.010756] | −0.063844 | [−0.120221, −0.008421] | 56.3%（FREQ 更好） |

Bootstrap 以 **Bemis-Murcko scaffold** 为重采样单位，5000 次，seed 1703；同一 draw 对两个 arm 使用同一批分子。504 个 Dev 分子只有 231 个 scaffold，区间较宽。

解读：

- TASK 相对 BASE 的点估计是 +0.043 RMSE，但 95% CI 跨 0；
- FREQ 相对 BASE 的点估计是 +0.074 RMSE，95% CI 不跨 0（FREQ 优势 [0.0108, 0.1355]），MAE 优势的 CI 也不跨 0；
- TASK 相对 FREQ 的点估计是 **−0.0306**（TASK 更差），CI 跨 0。也就是说，数据既兼容“频率略好”，也兼容“没有差异”；**没有**任务词表更好的证据。

作为次要描述，用同一 Dev 标签方差 1.41334 换算：

| 模型 | Dev R² |
|---|---:|
| R2_REF | 0.7878 |
| BASE | 0.5724 |
| D_TASK | 0.6187 |
| D_FREQ | 0.6498 |

任务模型保留了 R2_REF 解释方差的大约 **78.5%**（0.6187 / 0.7878），但 RMSE 仍高 **34.0%**，绝对差距 0.1864。两种表述都成立；主指标按协议用 RMSE。

### 3.6 冻结晋级判断

| 条件 | 数值 | 是否满足 |
|---|---:|---|
| Stage 2-A：`FREQ_RMSE − TASK_RMSE ≥ 0.015` | **−0.030598** | 否 |
| Stage 2-B：`BASE_RMSE − TASK_RMSE ≥ 0.03` | +0.043338 | 是 |
| Stage 2-B：`TASK_RMSE − R2_REF ≤ 0.025` | +0.186412 | 否 |
| Stage 2 总条件（A 或 B） | — | **否** |
| Stage 3：两个 D0 seed 均为正且均值 ≥0.01 | 未执行 seed 1 | **否** |

因此：

- 不执行 `D0_S1_TASK/FREQ`；
- 不执行 `D1_S0_BASE/TASK/FREQ` 神经验证；
- 不执行第三 seed；
- 不做任何事后超参数搜索或词表规则修改。

### 3.7 词表覆盖、稳定性与信息重复

根级覆盖（Fit）：

| Split | Arm | 根覆盖 | 分子覆盖 | 每根有效词均值 | 每根有效词中位数 | 每根总出现均值 |
|---|---|---:|---:|---:|---:|---:|
| D0 | D_TASK | **22.52%** | 90.16% | **0.356** | 0 | **0.442** |
| D0 | D_FREQ | **63.46%** | 99.58% | **3.148** | 1 | **4.047** |
| D1 | D_TASK | 41.43% | 97.76% | 0.585 | 0 | 0.708 |
| D1 | D_FREQ | 71.11% | 99.89% | 3.216 | 2 | 4.622 |

逐词根覆盖的差异更大：D0 任务词平均 **0.278%**、最大 **5.18%**；频率词平均 **2.46%**、最大 **38.09%**。任务词中只有 **5.5%** 的词根覆盖 >1%，频率词是 **30.5%**。

词表稳定性：

- D0/D1 任务词交集 **8 / 128**；Jaccard **0.0323**，overlap coefficient **0.0625**；
- 交集词多为较高 df、中低排名的词；两个 split 的头部词基本不同；
- D0/D1 的 Fit 集交集 **2415 / 2856**，Dev 交集 **63 / 504**，所以这不是独立重复实验，只是同一个 official Train 内的两个切分。

信息重复（Fit，描述性）：

| 指标 | D0 | D1 |
|---|---:|---:|
| 任务∪频率词表内的词对 `\|corr\| ≥ 0.95` | 220 | 226 |
| 被选词（task+freq）中带有被去重等价成员的代表 | 20 | 7 |
| 选中词与低阶特征的最大 `\|corr\|` | 0.9905 | 0.9879 |

其中任务词表本身带有等价成员的代表为 D0 8 个、D1 2 个。

这说明：

1. 部分任务词几乎是低阶 `X0` 的线性代理；残差化降低了但并未消除这种重复；
2. 任务词之间存在大量近重复（同一局部结构的不同根/不同枚举表示）；
3. 去重只处理“完全相同计数列”，不处理高相关列，因此 128 个词并不是 128 个独立信息方向。

### 3.8 D1 CPU 选择（探索性，不参与神经判定）

D1 独立重跑候选/选择/ridge：

- 候选 32,440 类型；qualified 5,101；dedup 4,711；`X0` 1,070；
- 选 128 个任务词，size/gran 分布见 3.2；
- D1 `TASK_RIDGE` 0.911526，`FREQ_RIDGE` 0.901093，TASK 更差 0.010433；
- 和 D0 ridge 的 TASK 优势方向相反，和 D0 神经的 TASK 劣势方向一致。

因为没有执行 D1 神经运行，这一节只能说明“选择信号在 D1 上也不稳定”，不能给出 D1 的神经泛化估计。

---

## 4. 解释：为什么任务词没有赢

### 4.1 选择目标优化的是“弱条件相关”，不是“覆盖”

任务词按三折最小残差相关选，最高分只有 0.169。这个目标会偏好：

- 在 Fit 中与残差有稳定小相关的局部模式；
- 但不必在多数根上出现；
- 也不必与其它入选词互补。

结果就是一组“局部有弱信号、全局覆盖稀疏”的词。频率控制虽然也在同样的 size/gran/DF 单元格内选，但它在每个单元格内选 **最高 document frequency** 的词，天然获得更广的根覆盖。D0 的根覆盖 22.5% vs 63.5%、每根有效词 0.36 vs 3.15，就是最直接的机制。

在共享根 readout 中，`z_v` 对大多数根是 0；词表再“任务条件”，也只能影响少数根。BASE 已经能利用一阶邻居和原子 embedding，任务词带来的增量因此有限。

### 4.2 频率基线不是“随便的频率词”

频率控制不是随机词表，而是：

- 精确匹配 size / granularity / DF-quartile 联合分布；
- 在每个单元格内取最高 Fit document frequency；
- 因而在公平的分布约束下最大化了覆盖。

所以“频率词表更强”本身是一个有信息的结果：在固定 K 和固定粒度分布下，**覆盖**比“弱任务条件相关”更重要。这不等于频率词在因果上更重要。

### 4.3 任务信号在 split 间不稳定

- D0 ridge：TASK 比 FREQ 好 0.031；
- D1 ridge：TASK 比 FREQ 差 0.010；
- D0 神经：TASK 比 FREQ 差 0.031；
- D0/D1 任务词交集只有 8/128。

这三个事实一致地说明：当前选择规则抓到的不是稳定的、可迁移的任务结构，而更像是 Fit 内折的弱相关噪声/局部代理。正对照能选中强植入信号，说明流程本身没有坏；但真实标签上的信号远弱于植入信号。

### 4.4 为什么没有保留强 R2 性能

本轮神经模型是**故意剥离**的：没有 R2 trunk、没有 S2 分支、没有额外 R2 统计，只有一阶邻居 + 根原子 + 选中词 + 图标量。它的 Dev RMSE 0.703–0.777，而完整 R2_REF 是 0.548。差距不能直接归因于“任务词没用”，因为模型本身也缺少 R2 的强结构通路；这正是本轮要隔离的问题。

在这个受控模型里，结论是：

- 加入任务词能把 BASE 从 0.777 改善到 0.734；
- 加入频率词能改善到 0.703；
- 两者都远未达到 R2_REF 的 0.548。

因此，**不能**声称“紧凑词表可以替代 R2-SENTENCE”，也**不能**声称“任务词完全没有信息”。能声称的是：在本协议、本模型、本 K、本 seed 下，任务词表没有表现出优于频率词表的预测力，也没有保留强 R2 性能。

### 4.5 统计功效的限制

- Dev 只有 504 个分子、231 个 scaffold；
- 只执行了 D0/s0 一个神经 seed；
- TASK−FREQ 的 95% CI 为 [−0.0456, +0.1002]，不能排除“无差异”；
- 冻结规则用点估计做晋级，因此停止是协议决定，不是显著性检验的证明。

如果未来要区分“频率覆盖优势”与“任务条件信息”，需要在**新轮次**中预注册：

- 覆盖/暴露匹配的频率控制（不仅匹配 df，也匹配根覆盖或出现次数）；
- 多 seed、多个 scaffold 划分；
- 对入选词做覆盖约束或多样性约束；
- 用消融/置换检验估计词表增量，而不是只看一个 Dev 点估计。

这些都没有在本轮执行，不能用来修改本轮 verdict。

---

## 5. 最终判定

**`NO_SUPPORTED_DISCOVERY`。**

不是 `DISCOVERY_ONLY`：任务词表没有优于频率词表（D0 神经点估计方向为负）。  
不是 `COMPRESSION_ONLY`：紧凑词表没有接近 R2 性能（TASK 距 R2_REF 0.186 RMSE，FREQ 距 0.156；本轮复用晋级规则的 0.025 邻近阈值作为“接近”的操作定义，两者都远大于它）。  
不是 `INSUFFICIENT_OR_INVALID`：K=128、4683 个去重候选、正对照通过、39/39 检查通过、6 个单测通过。  

**边界**：这是“按冻结规则在 D0/s0 停止”的轮次级判断；它不证明任务条件结构词在更大模型、更多 seed、覆盖匹配的控制或不同 K 下一定无效。

---

## 6. 预算与 provenance

| 项 | 值 |
|---|---:|
| 三个 run 的 wall 合计 / 正式阶段 | 684.1 s / 约 11.5 min |
| 正式神经 GPU | 0.1900 h / 3 h |
| 峰值显存 | ≤56.5 MB（无 OOM） |
| 正式运行数 | 3 |
| CPU 候选/选择 | D0、D1 各一次，已冻结 |
| 检查 | 39/39 PASS |
| 单测 | 6 passed |

运行 provenance：

- 执行 commit：`ae32cc85b356ab0e298c53082a768d409f795037`（`main`）；
- 部署主机：`res`，2×A100-40GB，`res-gpu0` / `res-gpu1` 各跑一个进程；
- 三个 run manifest 均记录同一 `git_commit`、`code_sha256=d7458308…`、`config_sha256=02e93ff1…`；
- 远程结果经 `rr pull` 拉回，包含 `manifest.json`、`curves.csv`、`predictions_best_{fit,dev}.{csv,npz}`、`state_{best,final}.pt`、`train.log`；
- `analysis/postrun_manifest.json` 记录所有结果/分析文件的 sha256、总墙钟/GPU 和 verdict。

关键数据哈希：

| 对象 | sha256 |
|---|---|
| raw cache | `fe002287…123c7c` |
| train labels | `a44d29de…1114c2` |
| D0 split file | `675289f6…94e845` |
| D1 split file | `bdcd3f58…e61fcc` |
| D0 Dev rows | `eacce8ac…87193` |
| D1 Dev rows | `342c1de0…380a6` |

### 偏差与说明

1. **只执行 Stage 1**：Stage 2/3 未触发，这是冻结晋级规则的结果，不是遗漏。  
2. **没有 D1 神经结果**：D1 只有 CPU 选择/ridge；不能把它当作 D1 神经泛化的替代。  
3. **正对照是 CPU ridge**：用于证明“植入词会被选进任务词表”，合成标签从未进入真实训练。  
4. **D_TASK 与 D_FREQ 有交集**：D0 8 个、D1 3 个；协议不强制互斥，已如实报告。  
5. **R² 是派生量**：主指标是 RMSE；R² 仅用 Dev 标签方差换算，作为辅助阅读。  
6. **没有超参数搜索**：K、alpha、模型宽度、词 size、粒度、评分函数均按冻结配置。  
7. **没有第三 seed**，没有修改选择规则，没有读 D1 Dev 后改规则。

---

## 7. 文件索引

```
task_conditioned_word_discovery_v0/
├── PROTOCOL.md
├── REPORT.md
├── configs/tc.json
├── code/
│   ├── tc_common.py        # pins, IO, config, split helpers
│   ├── tc_vocab.py         # enumeration, canonical types, dedupe, matrices
│   ├── tc_cpu.py           # folds, ridge selection, frequency control, positive control
│   ├── tc_data.py          # label-safe tensors / vocabularies
│   ├── tc_neural.py        # shared root readout
│   ├── tc_train.py         # training / checkpoint / manifests
│   ├── tc_checks.py        # 39 implementation checks
│   ├── tc_analysis.py      # vocabulary statistics / top-word extraction
│   ├── tc_eval.py          # main table, bootstrap, gates, verdict
│   ├── tc_report.py        # decoded top-word Markdown
│   └── tc_finalize.py      # post-run provenance manifest
├── analysis/
│   ├── checks.json
│   ├── candidates_D{0,1}.json
│   ├── vocab_D{0,1}_{task,freq}.json
│   ├── selected_words_D{0,1}.csv
│   ├── selection_scores_D{0,1}.csv
│   ├── ridge_arms_D{0,1}.{csv,json}
│   ├── positive_control_D{0,1}.json
│   ├── vocabulary_stats.json
│   ├── structure_top20_D{0,1}.json
│   ├── top_words.md
│   ├── main_table.csv
│   ├── bootstrap_d0_s0.{json,npz}
│   ├── evaluation.json
│   ├── frozen_manifest.json
│   └── postrun_manifest.json
├── results/
│   ├── D0_S0_BASE/
│   ├── D0_S0_TASK/
│   └── D0_S0_FREQ/
└── tests/test_tc.py
```

### 复现命令（摘要）

```bash
# 本地检查
uv run python code/tc_checks.py
uv run pytest -q tests/test_tc.py

# 远程（已部署 commit ae32cc85）
rr run res D0_S0_BASE --pool res-gpu0 --gpus 1 \
  --result tracks/structural-sufficiency/task_conditioned_word_discovery_v0/results/D0_S0_BASE \
  -- python tracks/structural-sufficiency/task_conditioned_word_discovery_v0/code/tc_train.py \
  --run-id D0_S0_BASE --arm BASE --split D0 --seed 0 --device cuda \
  --out tracks/structural-sufficiency/task_conditioned_word_discovery_v0/results/D0_S0_BASE
# TASK / FREQ 同理；然后
rr pull D0_S0_BASE
rr pull D0_S0_TASK
rr pull D0_S0_FREQ
uv run python code/tc_eval.py --runs D0_S0_BASE D0_S0_TASK D0_S0_FREQ --split D0
uv run python code/tc_report.py --limit 20
uv run python code/tc_finalize.py
```

---

**最终一句话**：在本轮冻结协议下，任务条件结构词表能小幅超过无词 BASE，但没有超过同规模频率词表，也远未保留 R2-SENTENCE 的强预测性能；因此判定为 `NO_SUPPORTED_DISCOVERY`，并按要求在 Stage 1 停止。