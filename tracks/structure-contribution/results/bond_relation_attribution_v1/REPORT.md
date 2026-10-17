# cscl-bond-relation-attribution-v1 — 真实 Bond–Bond 一跳关系 vs 行和匹配 Null D

**协议**：`cscl-bond-relation-attribution-v1`（split E，salt `cscl-bond-relation-attribution-v1`）
**冻结 commit**：`42949d20ac61`；**代码哈希**：`bond_relation_attribution_v1.py 22c24392…`，
`run_bond_relation_attribution_v1.py 59725464…`；**配置** `f41507e2…`
**GPU**：`res` 2×A100-40GB（driver 550.163.01，torch 2.5.1+cu124），seed 0→GPU0，seed 1→GPU1
**标签纪律**：official valid / test 从未加载；Monitor+Locked 在本轮只读取一次（预测先冻结）；
所有 Monitor/Locked `.y` 在预处理/SSL/训练/预测阶段经 LabelGuard 不可达（0 次访问尝试）。

---

## 0. 结论

**冻结判定：`CASE_C_INCONCLUSIVE`**（预注册规则，未做任何事后改判）。原因是三条并列条件中
前两条成立、第三条不成立：

| 预注册条件 | 结果 | 数值（Locked，n=1000） |
|---|---|---|
| P1 R−D 有 GAIN 且在 Monitor 复现 | **成立** | −0.017537，CI95 [−0.028753, −0.010111]，两 seed 同号 |
| P2 R−U 有 GAIN 且在 Monitor 复现 | **成立** | −0.019151，CI95 [−0.032227, −0.010933]，两 seed 同号 |
| R 与 P3 差距 ≤ +0.010（实用显著性门槛） | **不成立** | R−P3 = **+0.015207**，CI95 [+0.008281, +0.025570] |

按预注册语义，Case C 不代表“没有效应”，而是：**真实一跳关系带来的收益是真实、可复现的，
但没有达到本轮预先声明的“值得继续投入”的实用显著性门槛**。因此本协议下停止，不追加 seed、
不追加架构、不改变判定。

科学问题的直接回答：

1. **R 比“行和完全匹配但不含真实连接归属”的 Null D 更好**：Locked MAE −0.0175（相对 −15.2%），
   Monitor −0.0122，两 seed、两区域同号，配对 bootstrap CI 上界 < 0。历史 R1 正结果
   （R1−U≈−0.0125）此前无法排除行和/度混杂；本轮 D 把行和与总权重逐分子精确对齐后，
   R−D 仍然为正，说明**收益不是行和匹配本身给出的**。
2. **行和/度结构本身几乎不提供信息**：D−U = −0.001615（CI 含 0），且 D 训练模型与 U 训练模型
   互换核后预测几乎不变（Locked 平均 |Δpred| = 0.0038），而 R↔D 互换核改变 0.19–0.27。
   D 与 U 在函数层面近似等价，R 与它们不等价。
3. **但仍不如同轮强参照 P3**：R−P3 = +0.0152（R 更差 18.4%），按预注册门槛“不值得继续”。
   R 只弥合了 D→P3 MAE 差距的 53.6%，尾部的弥合更少（29.5%）。

一句话：**真实一跳关系是一个真实、可复现、且不可由行和 Null 解释的归纳偏置，但它的绝对
预测力仍明显低于同一预算下的 BJOC-Null（P3）路线；按预注册标准应停止，而不是继续调 BondRW。**

---

## 1. 预注册与执行纪律

- 协议文档 `PROTOCOL.md`（哈希 `e3b048cd…`）在任何本轮 GPU 轨迹之前冻结；`protocol_freeze.json`
  绑定代码/配置/输入/核缓存/objects/初始化/预算上限。
- 预执行修正（均在任何正式 GPU 训练与标签读取之前）：amendment-1 micro-smoke 确定性门禁改为
  float32 噪声地板（远端 CPU 归约非确定，~1e-7，本机逐位一致）；amendment-2 整编码器哈希统一
  到历史 `junc:` 前缀并断言等于 `encoder_final_sha256`；amendment-3 两 seed 并行时对共享
  manifest / budget ledger 加 `fcntl.flock`。三次均无协议语义变化，重新冻结后执行。
- 预测冻结先于评分：40 个预测文件（4 臂×2 seed×2 区 + 6 组交叉核×2 seed×2 区）合并哈希
  `d3b23d1cc756cc8833122946950a95965b53c80c9bcbbd8c6386b3d9d9dd7354`；
  之后才写 `holdout_read.lock`（monitor y `531953fd…`，locked y `e0e798a8…`，单次读取两区）。
- 未使用 official valid/test；未用历史 exposed test 做任何选择；历史结果仅作背景引用，
  不与本轮内部 dev 数值排名。

## 2. Split E 与标签隔离

- 官方 train 10,000 行中 4 行 committed SMILES 与 PyG 图重建不一致（`941, 2210, 6071, 9766`），
  全部排除 ⇒ 9,996 行。**身份 = 由 PyG 图重建的 canonical constitution SMILES**（去立体化学），
  不是 committed 字符串；已验证两种身份定义给出完全相同的划分（0 行差异）。
- 按 `sha256(salt|identity)` 排序贪心分配，组不跨区：**fit 7,996 / monitor 1,000 / locked 1,000**，
  9,993 个身份组（3 个 2-member 重复组），`duplicate_groups_cross_sides = 0`。
- 行哈希：fit `b73ed7f7…`，monitor `77e79463…`，locked `7015fb01…`；身份摘要 `db1ecdf3…`。

## 3. 四个臂与“完全匹配”的证据

| 臂 | 关系/输入 | 参数 | 初始化 | SSL | batch order | updates |
|---|---|---|---|---|---|---|
| R | `K = sym(P)`，P 为物理 bond 线图的一步 lazy 随机游走 | 349,985 | `90bf6322…`/`9f656c4d…` | atom+bond，7,560 步 | `05acc80a…`/`2efdfa33…` | 22,680 |
| D | `s_i=Σ_j K_ij`，`K^D_ij = s_i s_j / Σ_k s_k` | 349,985 | 同上 | 同一 SSL | 同上 | 22,680 |
| U | `1/N_bond`（全枚举 pair） | 349,985 | 同上 | 同一 SSL | 同上 | 22,680 |
| P3 | BJOC-Null junction objects（无 bond 关系核） | 413,505 | `d06d5e16…`/`c79e6b02…` | atom+bond+junction 同一 SSL | 同上 | 22,680 |

`freeze-predictions` 的冻结匹配审计（两 seed 全为 True）：R/D/U 使用的编码器哈希 = SSL
`atom_bond_group_sha256`（`805ac9d8…`/`54284a16…`）；P3 使用 = `encoder_final_sha256`
（`08e5f613…`/`5289b180…`）；R/D/U 初始化逐位相同；四臂 batch order 相同；updates 相同。
R/D/U 枚举**同一份** pair 列表（每分子 `n(n+1)/2` 个三元组，仅权重不同），因此比较不是
“稀疏存储/计算量”差异造成的。R/D/U 的 soup Fit MAE 0.0017–0.0026，P3 0.0010（同轮拟合都接近饱和）。

## 4. 核的数学与混杂审计（1,200 个分子，fit/monitor/locked 各 400）

D 的定义保证逐分子**行和精确相等**（最大误差 6.7e-16）且总权重相等，`K^D` 秩为 1
（`rank_one_formula_err = 0`），bond 槽位置换不变（6.9e-17）。真实数据上：

| 统计量 | R | D | U |
|---|---|---|---|
| 非零质量占比 | **0.157** | 1.000 | 1.000 |
| 对角线质量占比 | 0.500 | 0.042 | 1/n |
| 行和 CV | 0.0710 | 0.0710 | 0 |
| `corr(线图度, 行和)` | 0.836 | 0.836 | — |
| top 特征值占比 | 0.123 | **1.000** | 1.000 |
| 到均匀核的 TV | 0.843 | 0.040 | 0 |

- `corr(K_R, K_D)` 仅 **0.043**；D 有 **83.9%** 的质量落在 R 的零支撑上。
- 本轮 1,200 个分子中 `D == U` 的个数为 **0**（仅在规则线图、即所有 bond 行和相同时才会退化；
  审计集里没有这种分子）。因此 D 在本轮是一个真正的“行和匹配 dense 插值”，不是 U 的换名。
- **必须声明的混杂**：R 稀疏（15.7% 非零、对角线质量 0.5）而 D 稠密（100% 非零、对角线 0.042），
  且 D 把 84% 的质量放到 R 的零支撑上。因此 **R−D 混合了“真实一跳 incidence”与“局部稀疏性/
  支撑选择（含自环质量）”两种效应**，本轮不能把 R−D 全部归因于“真实连接归属”。
  但 D−U≈0 与 D/U 换核等价这两点，反过来排除了“行和/度匹配”这一历史混杂。

## 5. CPU 门禁（正式 GPU 之前全部通过）

- 32 个快速单测（核数学、三元组重建、置换/重编号不变性、梯度活性、split builder、判定词汇表、
  评分数学、并发锁、无 official 路径扫描）全部通过；`ruff` clean。
- 参数审计：R/D/U 349,985，P3 413,505，逐模块精确；初始化哈希与历史 R3/P3 一致
  （`90bf6322…`/`9f656c4d…`，P3 `d06d5e16…`/`c79e6b02…`）。
- 前向不变性：重编号/槽位置换/batch 组成最大偏差 4.5e-8；保存-加载 0.0；五个模块梯度全部非零。
- 核被真实消费：init 时预测差 R−D 4.4e-5、D−U 6.6e-6；pair 聚合 `r` 的差为 0.0056 / 0.0181
  （均值 |r|≈0.043），即核差异进入 pair 表示而不只是浮点噪声。
- micro-smoke（同参两次 1-epoch）：soup 最大差 R 1.9e-9 / D 1.5e-8 / U 2.2e-8，均在 1e-6 噪声地板内
  （逐位一致为 False，已按 amendment-1 记录）。
- 端到端 smoke（20 步 SSL + 每臂 1 epoch + 两区预测）通过；正式训练前另有一次被弃用的 smoke，
  其暴露的哈希前缀 bug 已由 amendment-2 修正并重新冻结。

## 6. GPU 运行与预算

| 阶段 | seed 0 (s) | seed 1 (s) |
|---|---|---|
| SSL（7,560 步，无标签） | 1,634 | 1,661 |
| R（360 ep / 22,680 upd） | 431 | 450 |
| D | 447 | 478 |
| U | 444 | 479 |
| P3 | 510 | 616 |
| **合计（ledger）** | | **1.9859 GPU-h / 4.0 cap** |

两 seed 并行（GPU0/GPU1），墙钟约 63 min（12:04→13:07）。ledger 只计 SSL+训练；预测前向
（每 seed 8 次 ×1,000 分子）约 1 GPU-min，未入账，总量仍远低于 4.0 h 上限。
运行：`bra-s0-20261017-120438-69f7c209`、`bra-s1-20261017-120446-f8a346ae`（另 CPU：
`bra-cpu5`、`bra-cross`、`bra-freeze-pred`、`bra-score`、`bra-posthoc`）。

## 7. 主结果（Locked，n=1,000，seed-mean）

| 臂 | MAE | RMSE | median AE | p95 AE | top-10 SSE share | seed 0 / seed 1 MAE |
|---|---|---|---|---|---|---|
| **R** | **0.097797** | 0.352268 | 0.059641 | 0.2655 | 0.889 | 0.099443 / 0.096151 |
| D | 0.115334 | 0.492583 | 0.072032 | 0.2906 | 0.931 | 0.117784 / 0.112884 |
| U | 0.116949 | 0.503530 | 0.071212 | 0.2967 | 0.934 | 0.117883 / 0.116014 |
| P3 | 0.082591 | 0.223794 | 0.049305 | 0.2414 | 0.786 | 0.084380 / 0.080801 |

配对对比（负 = 前者更好；bootstrap 5,000 次分子重采样；tail = y ≤ −3，n=58）：

| 对比 | Δ MAE | CI95 | 判定 | per-seed Δ | top-10-removed | tail Δ | R 更好的分子比例 |
|---|---|---|---|---|---|---|---|
| **R−D** | **−0.017537** | [−0.028753, −0.010111] | **GAIN** | −0.018341 / −0.016733 | −0.012470 | −0.1077 | 0.615 |
| **R−U** | **−0.019151** | [−0.032227, −0.010933] | **GAIN** | −0.018440 / −0.019862 | −0.013160 | −0.1239 | 0.624 |
| D−U | −0.001615 | [−0.004897, +0.001499] | NEUTRAL | −0.000100 / −0.003129 | −0.000840 | −0.0163 | 0.503 |
| R−P3 | +0.015207 | [+0.008281, +0.025570] | LOSS | +0.015063 / +0.015350 | +0.012000 | +0.0747 | 0.407 |
| D−P3 | +0.032743 | [+0.020616, +0.052926] | LOSS | +0.033403 / +0.032083 | +0.024840 | +0.1824 | 0.325 |
| U−P3 | +0.034358 | [+0.021571, +0.056178] | LOSS | +0.033503 / +0.035213 | +0.024530 | +0.1987 | 0.325 |

R−D 的最大单分子收益占比 13.8%（top-10 移除后仍 −0.0125）；R−U 为 14.9%。
RMSE：R 0.3523 vs D 0.4926 vs U 0.5035 vs P3 0.2238 —— R 也大幅降低了重尾（RMSE 主要受尾部驱动）。

**Monitor 复现（n=1,000）**：R 0.101878，D 0.114121，U 0.115647，P3 0.090922；
R−D −0.012243 [−0.016668, −0.008023] GAIN（两 seed −0.012960/−0.011525）；
R−U −0.013768 [−0.018786, −0.009170] GAIN；D−U −0.001526（NEUTRAL）；
R−P3 +0.010957 [+0.005518, +0.016211] LOSS。**同一方向、同一量级，P1/P2 复现成立。**

## 8. 机制诊断（评分后的描述性分析，不改判定）

**(a) 冻结模型 × 交换核（Locked，seed-mean MAE）**

| 模型 | 自己的核 | 换成 R | 换成 D | 换成 U |
|---|---|---|---|---|
| R | 0.0978 | — | 0.3046 | 0.3082 |
| D | 0.1153 | 0.2317 | — | 0.1155 |
| U | 0.1169 | 0.2352 | 0.1170 | — |

D↔U 互换只改变 0.0038（Locked 平均 |Δpred|），R↔D 互换改变 0.19–0.27。即：
**模型确实消费了核；D 与 U 把模型带到几乎相同的函数解；R 不是。**

**(b) 误差区域**：R 的收益集中在重尾。Locked 普通区（n=942）MAE R 0.0843 / D 0.0963 / P3 0.0728；
尾部（n=58）R 0.3165 / D 0.4241 / U 0.4404 / P3 0.2417。R 把尾部误差相对 D 降低 25.4%，
但 P3 的尾部仍比 R 低 23.6%。RMSE 的差距（0.352 vs 0.224）也主要来自尾部。
逐分子收益相关性 `corr(Δ_RD, Δ_RU) = 0.956`（Locked），说明 R 相对 D 与相对 U 的收益几乎是
同一批分子——与 D≈U 一致。

**(c) 训练后梯度**：五个模块（atom/bond encoder、pair MLP、composer、reader）在两 seed 三臂
全部非零，模型没有退化成“忽略关系”的解。

**(d) 解释**：R 的核是稀疏、局部、带自环质量的一跳邻接；它提供的不是“度/行和”信息（D 已匹配
且 D≈U），而是**具体哪些 bond 对相邻**这一支撑结构。这个信号在低溶解度尾部尤其有用。但它
仍不足以弥补 P3 的 junction 对象 + 全局组合归纳偏置的差距。

## 9. 按预注册规则的解释与停止

- P1/P2：Locked GAIN，Monitor 两 seed 同号且 |Δ| > 0.004 ⇒ **复现成立**。
- R−P3：+0.0152 > +0.010 ⇒ 不满足 Case A。
- Case B 的四个条件（B1 D 吸收 R 的收益、B2 对 U 无增益、B3 落后强参照 > 0.02、
  B4 D 优于 R）**均不成立**（`reasons = ["no_condition_met"]`）⇒ 冻结判定 **Case C**。
- 停止规则：不再增加 seed、不换架构、不因结果改协议。**BondRW + 真实关系线不作为竞争性
  预测器继续投入。**

## 10. 混杂、限制与不可声称的结论

1. **R−D 仍混有稀疏性/支撑效应**：R 稀疏、D 稠密，D 的 84% 质量在 R 的零支撑上，二者对角线
   质量差 0.5 vs 0.042。本轮只能声称“真实一跳支撑 + 局部稀疏性”整体优于行和匹配的稠密 Null，
   不能把 0.0175 全部归因于“真实连接归属”。
2. **只有 2 个 seed、1 个内部划分、1 个数据集/端点**；CI 是分子级配对 bootstrap，不含
   split/超参不确定性。
3. **P3 不是等参数量对照**（413,505 vs 349,985），且使用不同归纳偏置（junction 对象）；
   R−P3 是“与同轮强参照的差距”，不是“关系的因果价值”。
4. 与历史轮次的数值不可直接排名（不同 split、预算、轨迹）；本文只引用方向性结论。
5. 未做：更多 seed、交叉验证、架构/超参搜索、official valid/test、把关系核接到 P3 上。
6. `D == U` 的退化只在规则线图上发生；本轮审计集没有这种分子，但在别的数据集上必须重新检查。

## 11. 交付物与复现

结果目录 `tracks/structure-contribution/results/bond_relation_attribution_v1/`：

| 文件 | 内容 |
|---|---|
| `PROTOCOL.md` / `protocol_freeze.json` | 预注册协议 + 冻结（代码/配置/输入/核/预算） |
| `split_E.json` / `split_audit.json` / `pairing_verification.json` | Split E、4 行 mismatch、身份一致性 |
| `objects_manifest_E.json` / `kernels_manifest_E.json` / `kernel_audit.json` | 对象与 R/D/U 核、1,200 分子审计 |
| `cpu_gates.json` / `reuse_audit.json` / `input_audit.json` | CPU 门禁、历史复用哈希、输入哈希 |
| `ssl_E_s{0,1}.json` / `train_{R,D,U,P3}_E_s{0,1}.json` | SSL 与训练审计（init/encoder/batch/参数/曲线） |
| `prediction_manifest_E.json` | 40 个预测文件的哈希与匹配审计（combined `d3b23d1c…`） |
| `holdout_read.lock` / `holdout_labels_E.npz` | 单次标签读取（y 哈希、行哈希、预测哈希） |
| `holdout_score_E.json` / `decision.json` | 主表、全部对比、冻结判定 |
| `per_molecule_errors_{monitor,locked}.csv` | 逐分子预测与绝对误差 |
| `post_hoc_mechanism.json` | 交换核、尾部/普通区、梯度、核统计 |
| `gpu_budget_ledger.json` / `evidence_manifest.json` | 预算账本（1.9859/4.0 h）、证据哈希清单 |

复现命令（冻结 commit 上）：

```bash
# 本地（需 RDKit，仅标签无关的 split 构建）
python tracks/structure-contribution/code/run_bond_relation_attribution_v1.py split
# 远端 CPU（res-local）
S=tracks/structure-contribution/code/run_bond_relation_attribution_v1.py
python $S verify-inputs && python $S objects && python $S kernels && python $S cpu-gates && python $S freeze
python $S smoke --seed 1 --device cuda:0
# 远端 GPU（GPU0/GPU1 各一 seed，并行）
python $S ssl --seed $s --device cuda:0
for a in R D U P3; do python $S train --arm $a --seed $s --device cuda:0; \
                       python $S predict --arm $a --seed $s --device cuda:0; done
# 远端 CPU
for s in 0 1; do for m in R D U; do for k in R D U; do [ $m != $k ] && \
  python $S predict-cross --model $m --kernel $k --seed $s --device cpu; done; done; done
python $S freeze-predictions && python $S score-holdout && python $S post-hoc && python $S finalize
```

## 12. 最终判断

1. **真实的一跳 Bond–Bond 连接关系确实携带可泛化的信号**：在两个独立 held-out 区域、两个 seed、
   以及与“行和/总权重精确匹配”的 Null D 的对照下，R 的 MAE 优势为 −0.0175 / −0.0122，
   配对 CI 不含 0；D≈U 与 D/U 换核等价排除了行和混杂；收益不能由少数分子解释（top-10 移除后仍为正）。
2. **该信号不足以让 BondRW 路线成为竞争性预测器**：R 仍比同轮 P3 差 0.0152（18.4%），
   超过预注册的 +0.010 实用门槛；尾部差距更大（0.317 vs 0.242）。
3. **按预注册规则**：`CASE_C_INCONCLUSIVE`，本轮停止，不追加 seed/架构/超参。
4. 若未来继续，方向不应是继续调 BondRW，而是（在新的预注册下）把“真实一跳关系 vs 行和 Null”
   这一**对照本身**迁移到更强的 backbone（例如作为 P3 的附加关系通道），并优先在尾部分子上
   验证；这需要新的授权，本轮未购买。
