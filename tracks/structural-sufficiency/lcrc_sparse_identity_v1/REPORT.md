# REPORT — LCRC 稀疏局部模式身份保留 v1

Round: `lcrc-sparse-identity-v1`
Track: `tracks/structural-sufficiency/lcrc_sparse_identity_v1/`
Pre-registration: `PROTOCOL.md`（正式训练前冻结），决策：`DECISION.md`
全部工件：`analysis/*.json`、`analysis/predictions/*.npz`、`analysis/report_tables.md`
Result label for every Test number: **`EXPOSED_TEST_EXPLORATORY`**

**一句话结论：** 512 维冻结 S2 模式本身确实带有可测的预测信息（SPARSE_ONLY 官方
Validation ROC-AUC **0.6638**、Test **0.6734**），但把它作为零初始化、无 bias 的线性残差
叠加到同轮重训的 LCRC-A 上之后，**没有**在官方 Validation 或 Test 上得到正增量：
三 seed 平均 `WIDE_TRUE − A0` 在 Validation 为 **−0.004436**（0/3 seed 为正）、在 Test 为
**−0.000752**（2/3 seed 为正）；对严格匹配的 `WIDE_NULL` 的优势方向两 split 不一致
（Validation **−0.002169**，Test **+0.006190**）。14 项门禁全部通过，Test 仅一次性暴露。
预定判定：**`NO_PERFORMANCE_SIGNAL`**。

本报告把「观察」「解释边界」「不能成立的结论」分开书写。

---

## 1. 关键数字一览

| arm | 参数 | valid ROC-AUC（3 seed 均值±sd） | valid AP | test ROC-AUC | test AP | 正式 GPU·h |
|---|---|---|---|---|---|---|
| A0 | 247,106 | **0.834969** ± 0.003156 | 0.372993 ± 0.051794 | **0.771597** ± 0.005534 | 0.285930 ± 0.034463 | 0.7207 |
| WIDE_TRUE | 247,618 | 0.830532 ± 0.001790 | 0.388977 ± 0.024787 | 0.770845 ± 0.007958 | 0.293951 ± 0.007894 | 0.7375 |
| WIDE_NULL | 247,618 | 0.832702 ± 0.003663 | 0.368517 ± 0.051547 | 0.764655 ± 0.005031 | 0.274745 ± 0.024293 | 0.8700 |
| FOLD64 | 247,170 | 0.833031 ± 0.004124 | 0.382343 ± 0.043333 | 0.765002 ± 0.006816 | 0.292682 ± 0.007214 | 0.8583 |
| SPARSE_ONLY | 513（CPU） | 0.663770 | 0.116390 | 0.673443 | 0.228847 | 0（CPU） |

`SPARSE_ONLY` 无 seed 维度（单次 liblinear 拟合），sd 因此不适用。
A0 的 3 seed 结果与历史 PTB A0 几乎逐位复现（见 §9），因此配对基线是同一配方同轮重训，
不是旧 checkpoint。

---

## 2. 报告必须回答的 8 个固定问题

### Q1 稀疏 S2 模式本身的预测能力如何？

弱但非零，并且与神经主干差距很大。

`SPARSE_ONLY`（仅 512 维归一化二值 x(G)，Train-only L2 逻辑回归，C=0.1，liblinear）：

| split | ROC-AUC | AP | BCE | n_iter |
|---|---|---|---|---|
| Validation | 0.663770 | 0.116390 | 0.093967 | 8 |
| Test | 0.673443 | 0.228847 | 0.126844 | 8 |

- 相对同轮 A0 的 3 seed 均值：Validation −0.1712 ROC-AUC、−0.2566 AP；Test −0.0982 ROC-AUC、
  −0.0571 AP。
- Train ROC-AUC 0.7525，Train−Validation 差距 0.0887，比所有神经模型（0.128–0.131）都小：
  该线性模型几乎不拟合，属于「欠拟合的低容量线性读取」。
- 结论：这 512 个模式的选择是有效的（远高于随机），但**这些模式携带的可加信息远不足以替代
  LCRC-A 主干已经聚合的信息**。

### Q2 WIDE_TRUE 是否在官方 Validation 和 Test 上稳定胜过 A0？

否。

| split | A0 均值 | WIDE_TRUE 均值 | 差值 | 逐 seed 差值 | TRUE 更高的 seed 数 |
|---|---|---|---|---|---|
| Validation | 0.834969 | 0.830532 | **−0.004436** | −0.002027, −0.003190, −0.008088 | 0 / 3 |
| Test | 0.771597 | 0.770845 | **−0.000752** | +0.010873, +0.001067, −0.014197 | 2 / 3 |

两个 split 的 3-seed 均值都不为正，Validation 0/3、Test 2/3，方向随 split 变化。
预定条件 1（两个 split 均 ≥ +0.010）与条件 2（≥2 个 seed 在两个 split 同时优于配对 A0，
实际 0 个）均不通过。

### Q3 如果有增益，是否胜过严格匹配的 WIDE_NULL？

没有稳定优势；方向在两个 split 上相反。

| split | TRUE − NULL | 配对 bootstrap 95% 区间 | 差值 > 0 的重采样比例 | 逐 seed 差值 |
|---|---|---|---|---|
| Validation | **−0.002169** | [−0.011544, +0.006114] | 0.306 | −0.001230, +0.002090, −0.007367 |
| Test | **+0.006190** | [−0.002842, +0.015657] | 0.903 | +0.011460, +0.005085, +0.002024 |

Test 上 +0.0062 略高于预定阈值 +0.005，但 Validation 上为 −0.0022，且 Test 区间包含 0
（90.3% 的重采样为正，不到 95%）。预定条件 3 要求两个 split 同时 ≥ +0.005 → 不通过。

需要强调：`WIDE_NULL` 的随机矩阵只匹配行和、列和与总非零数，不匹配任何高阶结构
（Train 上非零位置仅 **17.4%** 与 TRUE 重合，Jaccard 0.174），所以它同时是
「结构信息被破坏」和「输入被重排」的复合对照，不能把 `TRUE − NULL` 直接读成
「可加身份信息的量」。

### Q4 固定模式折叠 FOLD64 与 WIDE_TRUE 有何差异？

几乎没有差异；FOLD64 用 1/8 的参数达到同等甚至略优的 Test ROC-AUC 均值。

| split | FOLD64 | TRUE − FOLD64 | 95% 区间 | 逐 seed |
|---|---|---|---|---|
| Validation | 0.833031 | −0.002499 | [−0.007515, +0.001912] | −0.000027, +0.000341, −0.007811 |
| Test | 0.765002 | +0.005843 | [−0.000806, +0.012484] | +0.014869, +0.001885, +0.000775 |

- FOLD64 的分支权重 L2 范数只有 0.53/0.57/0.53（TRUE 为 5.25/5.16/4.55），分支对 logit 的
  平均绝对贡献 0.026–0.028（TRUE 为 0.171–0.176）：64 维压缩后可行的残差幅度小一个量级。
- FOLD64 相对 A0 也是负的（Valid −0.001937，Test −0.006595），并不比 TRUE 更差。
- 按协议，FOLD64 参数量更少（247,170 vs 247,618），因此**不能把差异归因于压缩时机**；
  本轮只能说「在固定折叠下，残差容量被压到 64 维后表现与 512 维无实质差别」。

### Q5 增益是否主要体现为 ROC-AUC，还是也改善了 AP？

没有任何 ROC-AUC 增益；AP 上有小幅、方向为正但不稳定的差异。

| 指标 | A0 | WIDE_TRUE | TRUE − A0 | 逐 seed | TRUE 更高的 seed |
|---|---|---|---|---|---|
| valid AP | 0.372993 ± 0.0518 | 0.388977 ± 0.0248 | +0.015984 | +0.050260, −0.011318, +0.009009 | 2 / 3 |
| test AP | 0.285930 ± 0.0345 | 0.293951 ± 0.0079 | +0.008021 | +0.039279, −0.010304, −0.004911 | 1 / 3 |

- AP 的 seed 间波动（A0：sd 0.052 Valid / 0.035 Test）大于均值差本身，且 Test 上只有 1/3
  seed 为正。
- 预定条件 4（`WIDE_TRUE` 平均 Test AP 不低于 A0）**通过**（0.293951 > 0.285930），但这是
  五个条件中唯一与「有信号」同向的一条，且它只在 ROC-AUC 条件全部失败的前提下才有意义。
- 值得记录的旁证：`WIDE_TRUE` 的 Test AP 的 seed 间方差（sd 0.0079）明显小于 A0（0.0345），
  但这是 3 个点的描述，不作为结论。

### Q6 Test 和 Validation 结论是否一致？

均值层面一致（都略为负），逐 seed 与对照层面不一致。

- 均值：两个 split 的 `TRUE − A0` 均为负（−0.004436 / −0.000752）→ 一致。
- seed 排名：Validation 0/3、Test 2/3 为正；s0 在 Validation 是 −0.002027，在 Test 是
  **+0.010873**（本轮最大的单 seed 正差），s2 在 Validation −0.008088、Test −0.014197。
  同一 seed 的符号在两个 split 上不一致 → 不一致。
- 身份特异性：`TRUE − NULL` 在 Validation 为负、在 Test 为正（−0.002169 / +0.006190）
  → 不一致。
- 排序层面：Validation 排序 A0 > FOLD64 > NULL > TRUE；Test 排序 A0 > TRUE > FOLD64 > NULL。
  TRUE 在 Validation 最差、在 Test 次好 → 只改 split 就能改变 arm 的顺序，说明这一量级的
  差异（<0.01）被 scaffold 划分的具体组成主导，而不是被机制主导。

### Q7 是否足以支持继续研究「保留局部模式身份」？

作为**性能**方向：不足以支持。预定判定为 `NO_PERFORMANCE_SIGNAL`（见 §10）。

可继续使用的部分（事实，不是提案）：`SPARSE_ONLY` 显示这套 512 维冻结模式本身有
0.66–0.67 ROC-AUC 的信息量，且分支在训练中确实被激活（TRUE 分支 L2 范数 4.6–5.3，
平均绝对 logit 贡献 0.17，见 §7）。也就是说，本轮否定的是「在当前 A0 主干上挂一个
零初始化线性稀疏残差、用 R3 配方训 100 epoch 能带来增量」，而不是「这些模式没有信息」。

### Q8 哪些机制性结论仍然不能成立？

明确**不能**从本轮数据得出：

1. **不能**说「512 个 S2 模式无信息」——SPARSE_ONLY 与分支权重的非零范数都反驳这一点。
2. **不能**说「压缩是唯一瓶颈」或「恒等保留是必要瓶颈」——两者都没有产生正增量。
3. **不能**把 FOLD64 与 TRUE 的任何差异归因于「压缩时机」——参数量与容量不同（协议已声明）。
4. **不能**把 `TRUE − NULL` 读作「可加身份信息的量」——NULL 破坏了结构也重排了输入。
5. **不能**说「Validation 与 Test 都证明无效应」——Test 只暴露一次，n=4113，3 seed；
   配对区间半宽约 ±0.011，无法排除 ±0.01 量级的真实效应。
6. **不能**说「A0 已经完美利用了这些模式」——A0 的输入是 S1/S2 的 embedding 求和，
   但本轮没有做「移除模式信息后 A0 掉多少」的消融（协议未包含），所以信息是否冗余未测。
7. **不能**给出关于远距离表达性的任何结论——18 环分子对在 512 维表示下仍然碰撞
   （Jaccard = 1.0，见 §4），但本轮不研究方向性/远距离问题。

---

## 3. 数据、特征与实现（可核对事实）

### 3.1 划分

| split | n | row-id sha256 |
|---|---|---|
| Train | 32,901 | `4e77289653e41be5d2267f54d9f33fda4b70fd622c19b93a282db710a9318b0a` |
| Validation | 4,113 | `ab298cdbd817e15ad190c7a2a73dbb59babbfb471f947b58e9a9761452ce983a` |
| Test | 4,113 | `0006fe444e4d331312162f111f24c1f2b735d225f7290238a15527da9673c804` |

- 官方 OGB scaffold split 文件 sha256 `9f1e46e77897b3de3d194…`（见 `analysis/provenance.json`），
  与冻结 split manifest（`lcrc_molnet_benchmark_pilot_v2/analysis/split_manifest_ogbg-molhiv.json`，
  sha256 `107269d45ed817b2797e92620037fe5f8db09395e9ec87e38aee5a48688e4b33`）逐行一致。
- Test row-id sha256 与 v2 基线 Test 文件里的 `test_rows_sha256` 相同 → 与历史轮次是同一批 Test 分子。
- 冻结 FeatureSpec sha256 `7270f7677b349a46bb193174381abcb84f14d1b4f83bd1362d4aa20389a515ee`。
- 标签一致性：源 CSV 41,127 行与 OGB 标签逐行相同（0 mismatch，`runs/*/run.json:data.source.label_agreement`）。

### 3.2 稀疏模式词表

- key = `(root_atom_type, path_cat, path_end63, path_closure)`，直接取生产 S2 缓存的整数编码
  （编码为 `((root*n_path_codes+path_cat)*n_nbr_codes+path_end63)*2+path_closure`）。
- Train 上唯一 key 数 **1,197** ≥ 512 → 无 `FEATURE_INFEASIBLE`。
- 取 document frequency 最高的 512 个，df 范围 **[23, 27451]**，其中 df=1 的 key 有 237 个（未被选中）。
- 词表哈希 `ac090b0e5186c478a5c6d2a02b8b95b2c38acf91bb069817ddedd41d94e61abe`
  （顺序即冻结顺序，`analysis/feature_vocabulary.json`）。

### 3.3 三部分向量统计（同一冻结词表）

| split | 模式覆盖率 | 无模式分子 | 词表外 S2 路径比例 | 活跃模式数 均值 / 中位 / 最大 | nnz | 二值矩阵 sha256 | 归一化向量 sha256 |
|---|---|---|---|---|---|---|---|
| Train | 0.999726 | 9 | 0.003113 | 21.18 / 20 / 63 | 696,925 | `cef0d483…5eb4df` | `1bf7d576…ed8daee` |
| Validation | 1.000000 | 0 | 0.014122 | 21.10 / 20 / 57 | 86,789 | `cd29d837…a329185` | `a81a5509…ec6072` |
| Test | 0.999757 | 1 | 0.007577 | 19.05 / 18 / 54 | 78,340 | `adc6ee54…a138f77` | `b79fdc2a…83206fb` |

活跃模式数分位数（Train）：p5=8, p25=15, p50=20, p75=26, p95=37, p99=43。
向量定义 `x = c / sqrt(max(1, ||c||_0))`，float32。

### 3.4 实验臂

- **A0**：原 `PTBLCRC`（`ptb_model.build_model`，`dissect_common.RECIPE`），247,106 参数，
  同轮重训 3 seed。
- **WIDE_TRUE**：A0 + 512 维 `nn.Linear(512, 1, bias=False)`，零初始化，端到端联训，247,618 参数。
- **WIDE_NULL**：同结构同初始化同配方，输入换成 label-blind 二分图 double-edge swap 矩阵。
  每 split 独立随机化，`8 × nnz` 次交换尝试，**严格守恒**行和、列和与总非零数。

  | split | 尝试 | 接受 | 接受率 | 非零位置被改变比例 | Jaccard(TRUE, NULL) | 行和/列和守恒 |
  |---|---|---|---|---|---|---|
  | Train | 5,575,400 | 2,741,731 | 0.4918 | 0.7033 | 0.1742 | 均相等 |
  | Validation | 694,312 | 336,083 | 0.4841 | 0.7005 | 0.1761 | 均相等 |
  | Test（只在 Test 评分内部构建，seed 20261105） | 626,720 | 332,547 | 0.5306 | 0.7298 | 0.1562 | 均相等 |

  同一 seed 重算得到同一矩阵（`recompute_deterministic=True`）；`labels_used=False`。
- **FOLD64**：512 key → 64 桶（带盐 `lcrc-sparse-identity-v1` 的 SHA-256，按摘要升序
  `rank % 64`），恰好每桶 8 个 key，`z_j = (1/sqrt8)·Σx`，64 维零初始化无 bias 残差，
  247,170 参数，分配哈希 `54706fa513509fb0621a1fa5919765c6ea8ab6a8a335dbceffab26a9dab8e555`。
- **SPARSE_ONLY**：`LogisticRegression(penalty="l2", C=0.1, solver="liblinear",
  class_weight=None, fit_intercept=True, tol=1e-4, max_iter=4000, random_state=20261103)`，
  仅 Train 拟合，512 个系数全部非零（L1 范数 73.98，L2 5.70），n_iter=8。

### 3.5 训练配方（与历史 PTB MolHIV A0 完全一致）

AdamW，peak LR 3e-4，per-step cosine 到 1e-5，wd 1e-3，batch 128，dropout 0.10
（位置：`h` 之后、`f`/`rho` 之前），global clip 5.0，unweighted BCEWithLogits，100 epochs，
无 EMA/SWA/soup/预训练/类别加权/集成，无超参搜索，无早停。
Checkpoint 选择用 `ptb_train_mol.Tracker` 规则（最高 Validation ROC-AUC → 更低 Validation BCE
→ 更早 epoch），AP 只报告不参与选择。

同 seed 初始化一致性（逐参数逐位，`run.json:model.shared_state_equal_to_full`）：

| seed | A0 init 指纹 | WIDE_TRUE | WIDE_NULL | FOLD64 |
|---|---|---|---|---|
| 0 | `849adda7fc6a` | 相同，全参数相等 | 相同，全参数相等 | 相同，全参数相等 |
| 1 | `2e292ef24b02` | 相同，全参数相等 | 相同，全参数相等 | 相同，全参数相等 |
| 2 | `f621a7425213` | 相同，全参数相等 | 相同，全参数相等 | 相同，全参数相等 |

（分支权重在 `torch.random.fork_rng` 内构造并置零，未消耗 A0 的初始化随机流。）

---

## 4. 正确性门禁（14 项，全部 PASS）

`analysis/correctness_gates.json`（sha256 `457e35d1…37dda5d`），证据文件
`feature_checks.json` / `null_checks.json` / `initialization_checks.json`。

| # | 门禁 | 结果 | 关键证据 |
|---|---|---|---|
| 1 | 官方 Train/Validation 索引、标签与模型输入对齐 | PASS | 行数与 row-id 哈希匹配冻结 manifest；FeatureSpec 哈希匹配 |
| 2 | 词表只由 Train 结构产生 | PASS | `labels_used=false`；200 分子 df 重算 0 mismatch；重建哈希一致 |
| 3 | key 与真实 S2 记录严格对应 | PASS | 120 分子 / 9,602 条 path 记录，向量化构造 vs 独立逐条构造 0 mismatch |
| 4 | 对原子重编号、路径记录顺序不敏感 | PASS | 6 分子 RDKit `RenumberAtoms` + path 块倒序，向量逐位相同 |
| 5 | 批处理与单分子一致 | PASS | 64 行批次 = 逐行；多个单分子切片与整批逐位相同 |
| 6 | WIDE_TRUE 与 WIDE_NULL 参数量相同 | PASS | 均 247,618 = 247,106 + 512；FOLD64 247,170；三 seed 均成立 |
| 7 | NULL 行列边际与总非零守恒 | PASS | 三 split 行和直方图逐点相等、列和全等、nnz 相等 |
| 8 | NULL 与 TRUE 确实不同 | PASS | 变化比例 0.70 / 0.58（Hamming），Jaccard 0.17 |
| 9 | FOLD64 恰好 64 桶 × 8 key | PASS | 桶大小全为 8，512 key 各分配一次，重算一致 |
| 10 | 零初始化时三扩展模型与同 seed A0 预测一致（≤1e-6） | PASS | 32 分子 CPU float32 max\|Δ\| = **0.0**（三 seed × 三 arm） |
| 11 | 新增分支获得非零梯度、进入优化器、一次更新后改变 | PASS | 分支梯度 L1 0.1086，max\|g\| 3.88e-3；权重更新后非零 |
| 12 | 主干参数、数据缓存、训练规范与历史 PTB A0 一致 | PASS | 除 PTB recipe 额外记录的 `epochs_by_dataset` 字段外，Recipe 与 `dissect_common.RECIPE` 逐键相等；A0 init 指纹等于 `ptb_model.full_state` |
| 13 | checkpoint 保存/重载复现 Validation | PASS（冻结时补完） | 12 unit 重载 logit max\|Δ\| ≤ **5.72e-6** |
| 14 | 训练期不访问 Test 标签/指标 | PASS（冻结时补完） | 训练代码 AST 扫描无 `molecule_test`/`load_test_cache`；run 记录 `test_labels_read=false`；Test ledger 在冻结前 0 事件 |

非门禁记录（按协议要求的「额外记录」）：

- **18 环分子对仍然碰撞**：`C1CCCCCNCCCCCOCCCCC1`（O 与 N 环上距离 6）vs
  `C1CCCCNCCCCCCCOCCCC1`（距离 8），在 512 维冻结模式下二值向量完全相同
  （Jaccard = 1.0，各自 7 个活跃模式）。这与预期一致，本轮不研究方向性问题，
  因此**不是**失败。

---

## 5. Validation 结果（官方 scaffold Validation）

### 5.1 逐 seed 绝对值

| arm | seed | ROC-AUC | AP | BCE | best epoch | train ROC-AUC | train−valid |
|---|---|---|---|---|---|---|---|
| A0 | 0 | 0.831441 | 0.313169 | 0.0903 | 53 | 0.988063 | 0.156622 |
| A0 | 1 | 0.835754 | 0.401942 | 0.0717 | 32 | 0.951600 | 0.115846 |
| A0 | 2 | 0.837712 | 0.403926 | 0.0709 | 29 | 0.948640 | 0.110928 |
| WIDE_TRUE | 0 | 0.829409 | 0.363430 | 0.0901 | 43 | 0.978218 | 0.148809 |
| WIDE_TRUE | 1 | 0.832563 | 0.390624 | 0.0765 | 37 | 0.963976 | 0.131413 |
| WIDE_TRUE | 2 | 0.829624 | 0.412935 | 0.0711 | 29 | 0.943516 | 0.113892 |
| WIDE_NULL | 0 | 0.830639 | 0.315139 | 0.0897 | 53 | 0.988750 | 0.158111 |
| WIDE_NULL | 1 | 0.830473 | 0.372611 | 0.0732 | 32 | 0.952659 | 0.122186 |
| WIDE_NULL | 2 | 0.836992 | 0.417861 | 0.0697 | 29 | 0.948520 | 0.111528 |
| FOLD64 | 0 | 0.829436 | 0.336381 | 0.0871 | 40 | 0.975069 | 0.145633 |
| FOLD64 | 1 | 0.832222 | 0.388362 | 0.0758 | 37 | 0.965071 | 0.132849 |
| FOLD64 | 2 | 0.837436 | 0.422391 | 0.0695 | 29 | 0.946137 | 0.108701 |
| SPARSE_ONLY | – | 0.663770 | 0.116390 | 0.0940 | – | 0.752511 | 0.088741 |

### 5.2 三 seed 配对差值（ROC-AUC）

| 对照 | seed 0 | seed 1 | seed 2 | 均值 | 95% 配对 bootstrap（scaffold 分组，2000 次） | >0 比例 |
|---|---|---|---|---|---|---|
| TRUE − A0 | −0.002027 | −0.003190 | −0.008088 | **−0.004436** | [−0.016671, +0.006648] | 0.200 |
| TRUE − NULL | −0.001230 | +0.002090 | −0.007367 | −0.002169 | [−0.011544, +0.006114] | 0.306 |
| TRUE − FOLD64 | −0.000027 | +0.000341 | −0.007811 | −0.002499 | [−0.007515, +0.001912] | 0.153 |
| NULL − A0 | −0.000802 | −0.005281 | −0.000721 | −0.002267 | [−0.009139, +0.004837] | 0.226 |
| FOLD64 − A0 | −0.002005 | −0.003532 | −0.000277 | −0.001938 | [−0.012639, +0.008065] | 0.335 |

AP 侧的配对差值（描述性，不参与判定）：TRUE − A0 均值 **+0.015984**
（+0.050260, −0.011318, +0.009009），TRUE − NULL **+0.020459**，TRUE − FOLD64 +0.006618。

### 5.3 关于「scaffold bootstrap」（重要限制，必须如实记录）

协议规定以 Bemis–Murcko scaffold group 为重抽样单位。实测：

- Validation：4,113 个分子 → **4,113 个原子类型化 Bemis–Murcko scaffold，每个组大小 1**
  （最大组大小 1，单例组占比 1.0）。因此本 split 上的 scaffold bootstrap **退化为分子级
  （i.i.d.）bootstrap**，不能反映 scaffold 层聚类。
- 这是官方 scaffold split 的性质（重复的大 scaffold 全部在 Train 内：Train 32,901 分子 →
  10,863 个 scaffold，最大组 `c1ccccc1` 2,090 个分子、无环骨架 1,582 个），不是实现错误。
- Test 同样：4,113 分子 → 4,113 个 scaffold（全部单例）。
- 作为事后、明确标注的描述性补充（**不参与判定、取代不了预定分析**），额外用
  「擦除原子类型的通用骨架」（`MurckoScaffold.MakeScaffoldGeneric`）重做同一 bootstrap：
  Validation 2,291 组（最大 64），Test 2,017 组（最大 121）。结果几乎不变：
  Validation `TRUE − A0` −0.004436 [−0.016203, +0.006626]，Test `TRUE − A0` −0.000752
  [−0.011332, +0.010626]；未见结论翻转。见 `analysis/paired_bootstrap_extra.json`。
- 另有 4 个分子（Validation 2：含 `[AlH3-3]`/`[OH+]` 的金属配合物；Test 2：含 `[Al]`/`[B-2]`）
  无法被 RDKit 解析成可算 scaffold 的分子，各自获得独立组编号；由于所有分子本来就各自成组，
  不影响任何数值。

---

## 6. Test 结果（一次性，`EXPOSED_TEST_EXPLORATORY`）

### 6.1 执行纪律

- 冻结顺序：12 个神经 checkpoint + SPARSE_ONLY 全部完成后，先写
  `analysis/checkpoint_manifest.json`（含每 checkpoint 的 sha256、每个 unit 的
  best epoch / Valid 预测哈希 / 配方 / 参数量、code 哈希、Test 行 id 哈希、评分脚本哈希，
  manifest 自哈希 `604e25b028597b52…a37fa4`，`frozen_before_test_access=true`），
  **之后**才存在任何 Test 访问。
- 一次性评分：`code/si_score.py`（sha256 `eca81151…cf71043`）在一次进程内对 13 个模型
  各评分一次；Test randomisation（seed 20261105）在评分进程内部构建。
- `analysis/test_access_ledger.json`：13 条事件（12 unit + SPARSE_ONLY），全部发生在
  manifest 冻结之后；冻结前事件数 0。
- 评分后动作：`none: no re-selection, no re-ranking, no retraining, no new seed`
  （`analysis/test_scores.json:post_test_actions`）。
- 官方 `ogb.graphproppred.Evaluator(name='ogbg-molhiv')` 与 sklearn `roc_auc_score` 在
  13 个模型上 **max|Δ| = 0.0**。
- Test 前向在 CPU float32 上执行（checkpoint 以 SHA-256 绑定，记录于
  `test_scores.json:execution_regime`）。

### 6.2 逐 seed 绝对值

| arm | seed | ROC-AUC | OGB ROC-AUC | AP | best epoch |
|---|---|---|---|---|---|
| A0 | 0 | 0.765314 | 0.765314 | 0.246938 | 53 |
| A0 | 1 | 0.773651 | 0.773651 | 0.312296 | 32 |
| A0 | 2 | 0.775826 | 0.775826 | 0.298555 | 29 |
| WIDE_TRUE | 0 | 0.776187 | 0.776187 | 0.286217 | 43 |
| WIDE_TRUE | 1 | 0.774718 | 0.774718 | 0.301992 | 37 |
| WIDE_TRUE | 2 | 0.761629 | 0.761629 | 0.293645 | 29 |
| WIDE_NULL | 0 | 0.764727 | 0.764727 | 0.248669 | 53 |
| WIDE_NULL | 1 | 0.769632 | 0.769632 | 0.278912 | 32 |
| WIDE_NULL | 2 | 0.759605 | 0.759605 | 0.296653 | 29 |
| FOLD64 | 0 | 0.761318 | 0.761318 | 0.284568 | 40 |
| FOLD64 | 1 | 0.772833 | 0.772833 | 0.298390 | 37 |
| FOLD64 | 2 | 0.760855 | 0.760855 | 0.295117 | 29 |
| SPARSE_ONLY | – | 0.673443 | 0.673443 | 0.228847 | – |

### 6.3 Test 配对差值（ROC-AUC）

| 对照 | seed 0 | seed 1 | seed 2 | 均值 | 95% 配对 bootstrap | >0 比例 |
|---|---|---|---|---|---|---|
| TRUE − A0 | +0.010873 | +0.001067 | −0.014197 | **−0.000752** | [−0.011238, +0.010982] | 0.431 |
| TRUE − NULL | +0.011460 | +0.005085 | +0.002024 | **+0.006190** | [−0.002842, +0.015657] | 0.903 |
| TRUE − FOLD64 | +0.014869 | +0.001885 | +0.000775 | **+0.005843** | [−0.000806, +0.012484] | 0.958 |
| NULL − A0 | −0.000587 | −0.004018 | −0.016221 | −0.006942 | [−0.016456, +0.002605] | 0.077 |
| FOLD64 − A0 | −0.003996 | −0.000818 | −0.014971 | −0.006595 | [−0.016808, +0.004253] | 0.102 |

Test AP 配对差值：TRUE − A0 **+0.008021**（+0.039279, −0.010304, −0.004911，1/3 为正）；
TRUE − NULL **+0.019207**（2/3 为正）；TRUE − FOLD64 +0.001259。

### 6.4 Test 侧的 NULL / 特征完整性

- Test NULL：626,720 次尝试、332,547 次接受（0.5306），非零位置变化比例 0.7298，
  Jaccard 0.1562，行和/列和/总非零数与 TRUE 完全相等。
- Test 特征向量哈希 `b79fdc2a1d75f498c1dd91fca179ba8393ea3680dc96dcac65a45121883206fb`，
  与 manifest 中冻结前记录的 Test 行 id 哈希一致（行 id 未变，标签/向量未被用于选模）。

---

## 7. 分支行为诊断（机制事实，不是效应证据）

| arm | seed | 分支 L1 | 分支 L2 | 分支 L∞ | |contribution| 均值（valid） | |contribution| 均值（test） |
|---|---|---|---|---|---|---|
| WIDE_TRUE | 0 | 78.37 | 5.253 | 0.894 | 0.1756 | 0.1701 |
| WIDE_TRUE | 1 | 76.04 | 5.163 | 0.858 | 0.1718 | 0.1697 |
| WIDE_TRUE | 2 | 69.15 | 4.555 | 0.770 | 0.1741 | 0.1721 |
| WIDE_NULL | 0 | 82.07 | 4.972 | 0.819 | 0.1359 | 0.1320 |
| WIDE_NULL | 1 | 59.43 | 3.680 | 0.676 | 0.1100 | 0.1052 |
| WIDE_NULL | 2 | 54.92 | 3.416 | 0.626 | 0.0998 | 0.0958 |
| FOLD64 | 0 | 3.31 | 0.531 | 0.215 | 0.0263 | 0.0257 |
| FOLD64 | 1 | 3.73 | 0.570 | 0.195 | 0.0265 | 0.0267 |
| FOLD64 | 2 | 3.35 | 0.529 | 0.192 | 0.0277 | 0.0277 |

- 512 维分支训练结束时全部 512 个权重非零（`n_nonzero=512`），L∞ 0.63–0.89，
  平均把 logit 推向负方向（mean signed contribution −0.157/−0.147/−0.152 for TRUE，
  valid），与 MolHIV 正例率 2%（Train）/1.97%（Valid）/3.16%（Test）一致。
- 分支梯度非零（门禁 11），说明「零初始化 + 稀疏输入」没有造成梯度消失。
- WIDE_TRUE 的 Train ROC-AUC（0.9435–0.9782）低于 A0（0.9486–0.9881），
  Train−Valid 差距 0.114–0.149（A0 0.111–0.157）：加入 512 维稀疏残差没有让模型更过拟合，
  也没有带来更好的 Valid 泛化。
- gradient clipping 事件数：A0 32–45，TRUE 36–40，NULL 35–42，FOLD 36–43；
  无 non-finite 事件。

---

## 8. SPARSE_ONLY（CPU 诊断基线）

| split | ROC-AUC | AP | BCE | accuracy@0.5 | pred positive rate@0.5 |
|---|---|---|---|---|---|
| Validation | 0.663770 | 0.116390 | 0.093967 | 0.980306 | 0.0000 |
| Test | 0.673443 | 0.228847 | 0.126844 | 0.968393 | 0.0000 |

- 512 个系数全部非零，L1 73.98 / L2 5.70，n_iter 8（liblinear 快速收敛）。
- 概率在 0.5 阈值下几乎不产生正预测（Valid 0/4113，Test 0/4113），但这与评测指标无关
  （ROC-AUC / AP 都是排序指标）；此处只报告事实。
- 定位：这是「这些固定 S2 模式本身有多少可加预测能力」的下界估计。它说明模式有真实信号，
  但远不足以替代 LCRC-A。

---

## 9. 与历史已报告结果的描述性对照（不同来源，非配对）

| 来源 | 模型 | valid ROC-AUC | valid AP | test ROC-AUC | test AP |
|---|---|---|---|---|---|
| 本轮 | A0（同轮重训 3 seed 均值） | 0.834969 | 0.372993 | 0.771597 | 0.285930 |
| 本轮 | WIDE_TRUE | 0.830532 | 0.388977 | 0.770845 | 0.293951 |
| 本轮 | SPARSE_ONLY | 0.663770 | 0.116390 | 0.673443 | 0.228847 |
| 历史 PTB A0（3 seed 均值） | LCRC-A | 0.834908 | 0.372482 | 0.771606 | 0.285849 |
| v2 基线（ECFP4 + RDKit 2D，XGBoost） | 强指纹基线 | 0.836692 | 0.389505 | 0.762440 | 0.320992 |

- 本轮 A0 与历史 PTB A0 的 3 seed 均值差：Valid ROC-AUC **6.1e-5**、Test ROC-AUC **9.0e-6**、
  Valid AP 5.1e-4、Test AP 8.1e-5（见下表与 `validation_metrics.json:a0_history_replication`）；
  逐 seed 的 Valid ROC-AUC 差 **≤ 7.96e-5**，Valid AP 差 ≤ 1.47e-3，best epoch 完全一致
  （53/32/29）：说明配方与数据完全一致，剩余差异来自 GPU 上 scatter-add 的浮点非确定性。

  | seed | Valid ROC-AUC（本轮） | Valid ROC-AUC（历史） | Δ | best epoch（本轮/历史） |
  |---|---|---|---|---|
  | 0 | 0.831441 | 0.831409 | +3.22e-5 | 53 / 53 |
  | 1 | 0.835754 | 0.835674 | +7.96e-5 | 32 / 32 |
  | 2 | 0.837712 | 0.837641 | +7.04e-5 | 29 / 29 |

- 与 ECFP4+描述符 XGBoost 相比：本轮的 A0 / WIDE_TRUE 在 Valid ROC-AUC 上低约 0.006，
  在 Test ROC-AUC 上高约 0.008–0.009；Test AP 低约 0.027–0.035。
  该基线使用不同的特征族和模型族，**不是**本轮的严格配对对照，不参与任何判定。

---

## 10. 预定判定

逐条核对 `PROTOCOL.md §7` 的五个条件（阈值一律未改动）：

| 条件 | 阈值 | 实测 | 通过 |
|---|---|---|---|
| 1a `TRUE − A0`（Valid，3 seed 均值） | ≥ +0.010 | **−0.004436** | ✗ |
| 1b `TRUE − A0`（Test，3 seed 均值） | ≥ +0.010 | **−0.000752** | ✗ |
| 2 两个 split 上都优于配对 A0 的 seed 数 | ≥ 2 | **0**（Valid 0，Test 2，交集 0） | ✗ |
| 3a `TRUE − NULL`（Valid） | ≥ +0.005 | **−0.002169** | ✗ |
| 3b `TRUE − NULL`（Test） | ≥ +0.005 | +0.006190 | ✓ |
| 4 Test AP 不低于 A0 | ≥ A0 AP | 0.293951 ≥ 0.285930 | ✓ |
| 5 所有正确性门禁通过 | 全 PASS | 14/14 PASS | ✓ |

`PREDICTIVE_IDENTITY_SIGNAL` 需要 1–5 全部满足 → 不满足。
按协议的降级顺序，命中 **`NO_PERFORMANCE_SIGNAL`**：`WIDE_TRUE − A0` 的三 seed 平均
ROC-AUC 在 Validation **和** Test 上都不大于零。因此不再考察
`NO_IDENTITY_SPECIFIC_SIGNAL` / `MIXED_OR_WEAK_SIGNAL`（它们要求先有正向增量）。
详见 `analysis/decision.json` 与 `DECISION.md`。

---

## 11. 执行与资源

### 11.1 计费

| 项 | GPU·h |
|---|---|
| 性能预检（4 × 2 epoch 全 Train smoke，A100） | 0.022356 |
| 正式训练 12 个 unit | 3.186494 |
| **合计** | **3.208851** |

- 软预算 4.0，硬上限 5.0 → 未超预算，也未触发「停止并报告」分支。
- 预检投影 3.500 GPU·h > 实测 3.186 GPU·h：预检是 2 epoch（含首 epoch 的惰性分配/预热），
  其 epoch 均值高于 100 epoch 的平均值，因此投影偏保守。预检只用于「是否在预算内」的判断，
  正式计费一律用实测 wall time。

### 11.2 逐 unit 计时与显存

| arm | seed | mean epoch (s) | wall (min) | GPU·h | peak CUDA alloc (MiB) |
|---|---|---|---|---|---|
| A0 | 0 | 8.58 | 14.3 | 0.2385 | 91 |
| A0 | 1 | 8.67 | 14.5 | 0.2409 | 91 |
| A0 | 2 | 8.69 | 14.5 | 0.2413 | 91 |
| WIDE_TRUE | 0 | 8.87 | 14.8 | 0.2464 | 92 |
| WIDE_TRUE | 1 | 8.80 | 14.7 | 0.2444 | 92 |
| WIDE_TRUE | 2 | 8.88 | 14.8 | 0.2467 | 92 |
| WIDE_NULL | 0 | 10.37 | 17.3 | 0.2881 | 92 |
| WIDE_NULL | 1 | 10.51 | 17.5 | 0.2919 | 92 |
| WIDE_NULL | 2 | 10.44 | 17.4 | 0.2900 | 92 |
| FOLD64 | 0 | 10.32 | 17.2 | 0.2867 | 91 |
| FOLD64 | 1 | 10.44 | 17.4 | 0.2901 | 91 |
| FOLD64 | 2 | 10.13 | 16.9 | 0.2815 | 91 |

（峰值显存取 `torch.cuda.max_memory_allocated`；模型仅 0.25 M 参数、图小，91–92 MiB 属实。）

关于 per-epoch 时间的读法：四个 arm 的 epoch 循环是同一份代码（`si_train.train_unit`，
唯一的 arm 差异是每 batch 追加的 `si_feat` 行块）。A0/WIDE_TRUE（gpu0）为 8.6–8.9 s，
WIDE_NULL/FOLD64（gpu1）为 10.1–10.5 s；两张卡全程并发，且预检里 WIDE_TRUE 反而比 A0 慢
2.4 s/epoch（11.41 vs 9.04，随后正式运行只差 0.3 s/epoch）。因此这些 15% 量级的 arm 间差异
不能归因于超参或架构（协议完全一致），只能作为「实测值」记录，不能作为机制证据；
`gpu_hours` 全部按各 unit 自身 wall time 计算，未做任何归一化。

### 11.3 执行环境与提交

- 主机 `res`，进程后端，`res-gpu0` / `res-gpu1` 各一个正式进程（`a100-2`，
  2×A100-SXM4-40GB，driver 550.163.01，torch 2.5.1+cu124，CUDA 12.4）。
- 训练提交：`3d5230190484`。冻结提交：`6ad1fa177fa9`（manifest 写入时）。
- `analysis/provenance.json:training_code_identity.all_identical = true`：
  训练路径 6 个文件（`si_common/si_features/si_model/si_train/si_sweep/si_gates`）在训练提交
  与当前工作树中 **逐字节相同**（用 `git show <commit>:<path>` 的 sha256 对比）。
  训练之后只改过分析/评分侧的 `si_analysis.py` / `si_score.py`（在冻结之前），且它们的哈希
  已写入 manifest，评分脚本哈希在 Test 前就固定。
- 无正式训练失败、无重跑、无异常中断；`git_dirty_tracked=false`。

---

## 12. 复现

```bash
# 1) 特征、词表、NULL、FOLD64、14 项门禁（CPU，数分钟）
uv run python tracks/structural-sufficiency/lcrc_sparse_identity_v1/code/si_gates.py
# 2) 单元测试（15 项，CPU）
uv run pytest tracks/structural-sufficiency/lcrc_sparse_identity_v1/tests -q
# 3) 预算预检（A100）
rr run res --pool res-gpu0 --exp si-preflight ... code/si_preflight.py
# 4) 12 次正式训练（每卡一个顺序 sweep）
rr run res --pool res-gpu0 --exp si-gpu0-a0-true ... code/si_sweep.py --units "A0:0,A0:1,A0:2,WIDE_TRUE:0,WIDE_TRUE:1,WIDE_TRUE:2"
rr run res --pool res-gpu1 --exp si-gpu1-null-fold ... code/si_sweep.py --units "WIDE_NULL:0,WIDE_NULL:1,WIDE_NULL:2,FOLD64:0,FOLD64:1,FOLD64:2"
# 5) 冻结 → 一次性 Test → 分析
uv run python .../code/si_freeze.py
uv run python .../code/si_score.py
uv run python .../code/si_analysis.py
uv run python .../code/si_robustness.py   # 事后描述性补充，不参与判定
```

再次运行 `si_score.py` 会被 manifest 的 `test_access_ledger.json` 拒绝（一次性）。

---

## 13. 工件清单

| 路径 | 内容 |
|---|---|
| `PROTOCOL.md` | 冻结协议（运行前） |
| `REPORT.md` / `DECISION.md` | 本报告 / 预定判定落地 |
| `code/si_common.py` | 常量、哈希、run 目录、复用 PTB 助手 |
| `code/si_features.py` | key 编解码、词表、编码、double-edge swap、FOLD64、ring18 复现 |
| `code/si_model.py` | `WideLCRC`（冻结 A0 主干 + 无 bias 线性残差） |
| `code/si_train.py` | 与 `ptb_train_mol.train_unit` 同步的训练循环 |
| `code/si_gates.py` | 门禁 1–14 |
| `code/si_sparse.py` | SPARSE_ONLY |
| `code/si_preflight.py` | 预算投影 |
| `code/si_freeze.py` | checkpoint manifest + 门禁 13/14 |
| `code/si_score.py` | 一次性 Test 评分 |
| `code/si_analysis.py` | Validation/Test 指标、配对 bootstrap、GPU ledger、provenance、decision |
| `code/si_robustness.py` | 事后 scaffold 分组诊断（不参与判定） |
| `tests/test_sparse_identity.py` | 15 项 CPU 测试 |
| `analysis/feature_vocabulary.json` | 512 key、df、排序规则、词表哈希 |
| `analysis/feature_checks.json` | 门禁 1–5、9 + 三部分统计 + ring18 记录 |
| `analysis/null_checks.json` | 门禁 7–8 + 三 split 守恒/变化统计 |
| `analysis/initialization_checks.json` | 门禁 6、10、11 |
| `analysis/correctness_gates.json` | 14 项汇总 |
| `analysis/validation_metrics.json` | 逐 seed 指标、配对差、AP 差、train−valid、A0 复现对照 |
| `analysis/test_metrics.json` | Test 同结构 + OGB/sklearn 交叉核验 + Test NULL 统计 |
| `analysis/paired_bootstrap.json` | 预定 scaffold bootstrap（5 个对照 × 2 split） |
| `analysis/paired_bootstrap_extra.json` | 事后通用骨架 bootstrap（描述性） |
| `analysis/checkpoint_manifest.json` | 冻结清单（含自哈希） |
| `analysis/test_scores.json` | 一次性 Test 原始输出 |
| `analysis/test_access_ledger.json` | 13 条 Test 访问事件 |
| `analysis/predictions/*.npz` | 12 unit × {valid,test} + SPARSE_ONLY + 逻辑回归模型 |
| `analysis/gpu_ledger.json` / `analysis/provenance.json` | 计费与环境/哈希溯源 |
| `analysis/report_tables.md` | 本报告全部表格的机器生成版 |
| `runs/mol_ogbg-molhiv_s*_{A0,WIDE_TRUE,WIDE_NULL,FOLD64}/` | 每个 unit 的 `checkpoint.pt`（12 个，已入库，manifest 以此为 freeze 对象）、`run.json`、`selection.json`、`curve.csv` |
| `runs/smoke/costprobe_*` | 预算预检的 4 个 2-epoch probe 记录（GPU ledger 的一部分） |

`.gitignore` 只排除可重建的大文件（`train_predictions.npz`、`valid_predictions.npz`、smoke 的
checkpoint）；12 个正式 checkpoint 与全部 run 记录都已提交，因此在 clone 之后可以用
`analysis/checkpoint_manifest.json` 里的 sha256 逐个校验冻结对象。

所有历史目录只在读取路径上出现过；本轮没有修改任何既有文件
（`uv run research verify` 与 `git status` 均确认）。

---

## 14. 限制（如实记录）

1. **统计功效**：3 seeds、Valid/Test 各 4,113 分子，scaffold bootstrap（退化后等价分子级）
   的 95% 区间半宽约 ±0.008–0.011。因此无法排除 ±0.01 量级的真实效应，也无法证明
   「效应恰为 0」；本轮只能报告点估计与区间。
2. **scaffold bootstrap 退化**：见 §5.3，预定重抽样单位在此 split 上等价于分子。
3. **Test 非盲**：该 Test 在历史轮次已被多次读取，本轮所有 Test 结论均为
   `EXPOSED_TEST_EXPLORATORY`，不作为确认性证据。
4. **FOLD64 不可归因**：参数量不同（协议已声明）。
5. **WIDE_NULL 是复合对照**：破坏结构的同时重排输入，不能单独归因。
6. **A0 与历史不是逐位相同**：GPU scatter-add 浮点非确定性导致 ≤ 8e-5 的 Valid ROC-AUC 差，
   best epoch 与配方完全一致。
7. **Test 前向在 CPU**：与训练期 GPU 前向存在 ~1e-6 量级差异（重载校验已量化）。
8. **单一任务、单一数据集**：结论只对 `ogbg-molhiv` 官方 scaffold split 成立。
