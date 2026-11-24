# SFEC v1 后续实验报告

**轮次**: `structural_functional_equivalence_followup_v1`  
**状态**: 已完成，等待研究者判断；本轮到此停止。  
**约束遵守**: CPU only；单进程 `OMP_NUM_THREADS=1`；未训练新网络；未重扫完整 PCQM；未扩大 Train 前缀；只读复用父轮 1M/Valid 缓存；未读取官方 Test labels；未为标签修改阈值、数据或公式。  
**预注册协议**: `PROTOCOL.md`（分析前冻结）  
**配置**: `configs/sfec_v1.json`（τ=0.10，敏感性 0.05/0.20；2,000 次 context-level paired bootstrap，seed 0；λ=5；min discovery contexts=10）  
**工程检查**: `analysis/checks.json` 全部通过；`pytest` 5 passed。

---

## 0. 一页结论（直接回答六个问题）

### Q1. 原 v0 的 pair 方向错误和统计口径错误，实际影响多大？

**结论：对 P1/P3 的聚合 MAE 几乎为零，但对“方向/符号”类结论和 P2 的可用覆盖是实质性的；此外 v0 的 `pair_equal` 不是真正的 pair-type macro。**

- v0 discovery 有 **136,741 / 297,996 = 45.89%** 的 pair rows 存成了反序（B→A）；evaluation 有 **43,176 / 97,551 = 44.26%**。按无序对去重后，**846 个 discovery pair 的 median 符号被翻转**（846/1,831）。
- P1 的 MAE/RMSE/sign accuracy 在 canonical 修正前后完全相同：MAE `0.2621535875 → 0.2621535875`，差 `0.0`；RMSE 差 `0.0`；sign accuracy 差 `0.0`。原因是 delta 与 prediction 同时翻转，配对误差不变。
- P3 的 SHARED/LOCAL/LOCAL_NULL 也几乎逐位不变：`b` 最大差 `6.3e-14`，三个 MAE 差均 `<1e-14`。
- **P2 的 v0-style 受控遮蔽**受影响：v0 只有 16/20 对可评估、18,584 条观测；canonical 修正后 20/20 对可评估、21,970 条观测。LOCAL 相对 SHARED 的增益从 **8.87%** 变为 **7.82%**，方向不变但覆盖和数值都变了。
- 官方 Valid 的可见 `mu` 观测数 v0/canonical 都是 **36,952**、pair type 都是 **1,677**，所以 Valid 的聚合 MAE 表不受覆盖损失；受影响的是 pair 中位数的方向和符号报告。
- 统计口径：v0 的 `pair_equal` 实际是 **observation-equal**。真正的 pair-type macro 在 P1 上是 PAIR `0.3191122`、ZERO `0.3643207`（相对下降约 12.4%）；v0 的 `0.2621536` 不能当作 pair-type macro。
- 2×2 rectangle 按本轮定义 `Σ_AB k_AB(k_AB−1)/2` 为 **117,860,358**；它不是 pair-observation 数 297,996。

**判断**：v0 的 P1/P3 聚合数值可以沿用；但 v0 的 pair 方向、pair 符号、P2 可用性和 “pair_equal = pair-type macro” 的表述必须修正。方向错误不是“无影响”，而是“被聚合 MAE 掩盖了”。

### Q2. 真实接口条件 q 是否能提升直接替换预测？

**结论：能，而且这是本轮最稳健的收益来源。**

同一 evaluation observation 集合（main：96,991 obs / 38,768 contexts / 1,602 pairs）：

| 模型 | observation MAE | context-equal MAE | pair-type MAE |
|---|---:|---:|---:|
| PAIR | 0.26215 | 0.23657 | 0.31911 |
| PAIR_Q | **0.25112** | **0.22476** | **0.31378** |
| PAIR_Q_NULL | 0.26501 | 0.23919 | 0.32165 |

- `PAIR_Q − PAIR = −0.01103`，95% CI `[−0.01208, −0.00989]`；相对 PAIR 降低 **4.21%**。
- `PAIR_Q − PAIR_Q_NULL = −0.01389`，CI `[−0.01503, −0.01273]`；相对 NULL 降低 **5.24%**。
- `PAIR_Q_MEAN − PAIR_MEAN = −0.01242`，CI `[−0.01352, −0.01130]`；相对降低 **4.56%**。
- context-equal 上 `PAIR_Q − PAIR = −0.01181`，CI `[−0.01292, −0.01072]`；pair-type macro 上 `−0.00533`，CI `[−0.00695, −0.00376]`。三个口径方向一致。
- PAIR_Q 在 main 上的 coverage 是 **77.56%**；未支持时回退到 PAIR。因此上述 4.21% 是**被回退稀释后的保守值**；在 75,229 条真正启用 q 的观测上，PAIR_Q 相对 PAIR 的增益是 **5.71%**。
- 在 all-conditional-enabled 子集（64,321 obs）上收益更大：`PAIR_Q − PAIR = −0.01314`（相对 5.36%），`PAIR_Q − PAIR_Q_NULL = −0.01705`（相对 6.85%）。

**判断**：真实 q 不是噪声；它提供了直接 pair 查表之外的增量条件信息。`PAIR_Q_NULL` 明显更差，说明收益不是“任意条件键”带来的。

### Q3. LOCAL 是否具有超越直接条件查表的价值？

**结论：没有稳定、跨口径的超越；只有小幅且情境依赖的增益。LOCAL 的优势主要出现在“已见 context / 已支持 q / 直接监督被移除”的场景。**

| 比较集 | LOCAL | PAIR_Q | LOCAL − PAIR_Q | 读法 |
|---|---:|---:|---:|---|
| main, observation | 0.24844 | 0.25112 | **−0.00268** CI `[−0.00416, −0.00121]` | LOCAL 小胜 1.07% |
| main, context-equal | 0.22538 | 0.22476 | **+0.00062** CI `[−0.00037, +0.00161]` | 统计上打平 |
| main, pair-type | 0.30166 | 0.31378 | **−0.01212** CI `[−0.01583, −0.00865]` | LOCAL 胜 3.86% |
| all-conditional-enabled, obs | 0.23779 | 0.23183 | **+0.00596** CI `[+0.00480, +0.00716]` | PAIR_Q 更好 2.51% |
| Valid unseen exact context, obs | 0.21225 | 0.20257 | **+0.00968** CI `[+0.00614, +0.01302]` | PAIR_Q 更好 4.56% |
| Valid seen exact context, obs | 0.28657 | 0.29980 | **−0.01323** CI `[−0.01608, −0.01036]` | LOCAL 更好 4.41% |

- 在 main 上，LOCAL 相对 SHARED 降低 **8.34%**（CI 排除 0），相对 LOCAL_NULL 降低 **8.91%**；这证明“条件化的共享结构修正”不是全局可加模型能替代的。
- 但在所有条件模型都启用的子集上，LOCAL 反而比 PAIR_Q 差 2.51%；在官方 Valid 未见 exact context 上，LOCAL 比 PAIR_Q 差 4.56%。LOCAL 的“胜”主要来自它记住了已见 context 的局部结构，而不是可迁移到新 context 的共享功能。
- 严格遮蔽下（见 Q4），`LOCAL_MASKED` 比未遮蔽的 `PAIR_FULL` 低 0.00697（相对 `PAIR_FULL` 4.28%），但比未遮蔽的 `PAIR_Q_FULL` 高 0.00572（相对 `LOCAL_MASKED` 3.67%；相对 `PAIR_Q_FULL` 3.81%）。后者有额外的直接条件监督，**不是公平对照**，只能说明“直接条件查表仍是更强的实用预测器”。

**判断**：不能把 LOCAL 的增益宣称为“超越直接条件查表的共享结构功能”。更准确的表述是：**共享结构修正在已见/受支持 context 上有小幅增量；一旦要求跨 context 迁移或与直接 q 查表正面比较，这个增量就不稳定。**

### Q4. 严格删除直接分子监督后，条件组合信号是否仍然存在？

**结论：存在，统计上稳健，但幅度只有 7.76%（相对 SHARED_MASKED），低于预先声明的 10% 实用门槛。**

严格遮蔽：按结构支持排序取 20 对；20/20 可评估；21,970 条 evaluation observations、17,801 个 contexts。每对删除 discovery 中所有 A、B 共现 context 的 B cell，再删除这些 B 分子身份在所有切口的全部 cell；`SHARED_MASKED / LOCAL_MASKED / LOCAL_NULL_MASKED` 在完全相同的遮蔽数据上重拟合，NULL 在遮蔽后 context 集合内重建。checks 验证：直接共现为 0、被删分子 cell 残留为 0、A/B 仍连通。

| 模型 | observation MAE |
|---|---:|
| ZERO | 0.17013 |
| SHARED_MASKED | 0.16877 |
| LOCAL_MASKED | **0.15568** |
| LOCAL_NULL_MASKED | 0.17150 |
| PAIR_FULL（参考） | 0.16265 |
| PAIR_Q_FULL（参考） | 0.14996 |

- `LOCAL_MASKED − SHARED_MASKED = −0.01309`，CI `[−0.01459, −0.01167]`；相对 SHARED 降低 **7.76%**。
- `LOCAL_MASKED − LOCAL_NULL_MASKED = −0.01581`，CI `[−0.01735, −0.01438]`；相对 NULL 降低 **9.22%**。
- `LOCAL_MASKED − ZERO = −0.01445`，CI `[−0.01645, −0.01257]`；相对 ZERO 降低 **8.49%**。
- 对照：`SHARED_MASKED − ZERO = −0.00136`，CI `[−0.00270, +0.000056]`，只低 0.80% 且 CI 跨 0；LOCAL 的增益不是由 SHARED 基线带来的。
- context-equal 方向一致（相对 SHARED 7.36%）；pair-type macro 方向一致（相对 SHARED 9.72%）。
- leave-one-pair-out 最小增益 **6.75%**；最大单对贡献份额 **21.85%**；12/20 对为正、8/20 为负，per-pair 增益中位数 7.36%，范围 −6.64% 到 +37.26%。
- 四项统计条件（方向、CI、至少 10 对且每对至少 5 个 eval contexts、非单对驱动）均成立；**未过 10% 实用门槛**。

**判断**：严格遮蔽后条件组合信号没有消失，因此不能判 `NO_SUPPORTED_CONDITIONAL_RULE`；但它也没有强到满足 10% 实用门槛。按本轮保守操作化，判 **`CONDITIONAL_COMPOSITION_SCREEN_ONLY`**：真实但小，只够进入 screen，不够进入“已支持”的下一轮建模。

### Q5. 这种信号能否迁移到官方 Valid 未见 exact contexts？

**结论：相对 SHARED/NULL/ZERO 能迁移；相对直接条件查表 PAIR_Q 不能。未见 context 上 PAIR_Q 最好，LOCAL 不再占优。**

官方 Valid：36,998 条 pair observations，36,952 条可评估，1,677 个 pair types；剔除与 1M Train 前缀重复的完整分子。

| 子集 | n obs | ZERO | PAIR | PAIR_Q | SHARED | LOCAL | LOCAL_NULL |
|---|---:|---:|---:|---:|---:|---:|---:|
| all evaluable | 36,952 | 0.34791 | 0.29795 | 0.28200 | 0.30853 | **0.27296** | 0.30953 |
| unseen exact context | 6,765 | 0.22367 | 0.21117 | **0.20257** | 0.22927 | 0.21225 | 0.23025 |
| unseen context, q supported | 6,689 | 0.22395 | 0.21140 | **0.20270** | 0.22978 | 0.21256 | 0.23076 |
| seen exact context | 30,187 | 0.37576 | 0.31739 | 0.29980 | 0.32629 | **0.28657** | 0.32730 |
| clean subset | 24,381 | 0.35360 | 0.30158 | 0.28499 | 0.31436 | **0.27677** | 0.31494 |

未见 exact context 上：

- `LOCAL − SHARED = −0.01703`，CI `[−0.02084, −0.01311]`；LOCAL 比 SHARED 好 **7.43%**。
- `LOCAL − LOCAL_NULL = −0.01800`，CI `[−0.02188, −0.01404]`；好 **7.82%**。
- `LOCAL − ZERO = −0.01142`，CI `[−0.02009, −0.00296]`；好 **5.12%**。
- `LOCAL − PAIR_Q = +0.00968`，CI `[+0.00614, +0.01302]`；LOCAL **差 4.56%**。
- `PAIR_Q − PAIR = −0.00860`，CI `[−0.01194, −0.00539]`；PAIR_Q 好 **4.07%**。
- `SHARED − ZERO = +0.00560`，CI `[−0.00206, +0.01298]`；全局可加模型在未见 context 上没有可靠迁移收益。

在 seen exact context 上 LOCAL 仍比 PAIR_Q 好 4.41%（CI 排除 0），说明 LOCAL 的优势主要是“已见 context 的局部修正”，不是“未见 context 的共享功能泛化”。

**判断**：条件组合信号能迁移到未见 context（相对 SHARED/NULL/ZERO），但迁移后的最佳实用模型是 **PAIR_Q**。不能把 Valid 结果读成“LOCAL 的共享结构功能优于直接条件查表”。

### Q6. 是否发现具有非零、稳定且可复用的条件结构作用？

**结论：发现了真实的非零条件作用，但大部分条件候选是 near-zero；可复用的非零规则稀疏、支持小。**

- 条件残差离散度（221,334 条 supported rows，1,563 个 pairs）：pooled P80 |residual| 从 global **0.44082** 降到 conditional **0.39457**，降低 **10.49%**；median |residual| 从 0.15783 降到 0.13334，降低 **15.52%**。
- 但 per-pair 很不均匀：per-pair P80 reduction 中位数只有 **4.53%**，IQR `[−0.78%, +12.55%]`，只有 **69.1%** 的 pair 改善。
- 符号反转（阈值 0.05）：**271** 个反转 pairs、1,257 个 groups、**765** 个可验证组、**677** 个复现（**88.5%**）；**113** 个 pair 的两个方向都复现。阈值 0.10：172 个 pairs、460 个可验证组、418 个复现（90.9%）、75 个 pair 双复现。
- 条件候选（unique triples；τ=0.10）：**428** 个 triples、**259** 个可验证、**183** 个 pairs；覆盖 4,389 条 eval obs（main 的 **4.50%**），其中可验证候选覆盖 3,981 条（**4.10%**）。全局 pair 候选只有 **3** 个（415 obs）。τ=0.05 为 104 triples/64 可验证/861 obs；τ=0.20 为 1,516 triples/959 可验证（其中 932 个有 eval 覆盖）/22,491 obs。
- 候选的 eval 观测中 **78.92%** 落在 τ 内（按观测加权）；但 **near-zero 占绝对多数**：τ=0.10 的 259 个可验证候选中 252 个是 near-zero，非零只有 **7** 个（68 条 eval obs，约 0.07% main）。τ=0.20 的非零也只有 27 个（406 条 obs）。
- 可读的非零稳定例子：
  - `*C(=C)C → *C=C`，q=`p6,0,0|8,0,0,1;8,0,0,2`：discovery median −0.1565，eval median −0.1333，68.4% eval 在 τ=0.10 内，MAE 0.1085（38 discovery contexts / 19 eval）。
  - `*N → *NCC`，q=`p6,0,1|6,0,1,4;6,0,1,4`：discovery −0.1442，eval −0.1415，85.7% 在 τ 内，MAE 0.0676（25/7）。
  - `*C → *Cl`，q=`p6,0,1|6,0,1,4;16,0,1,4`：discovery −0.1034，eval −0.1388，77.8% 在 τ 内，MAE 0.0584（37/18）。
- 明确的符号反转例子：`*C → *C(C)=O`
  - q=`p8,0,0|6,0,1,1`：discovery +0.2136，eval +0.1742；
  - q=`p6,0,1|6,0,1,4;6,0,1,4`：discovery −0.7565，eval −0.6490；两个方向均复现。
- 失败例子：`*O → *OC`，q=`p6,0,0|6,0,1,1`：discovery +0.0218，eval +0.0068，只有 42.9% 在 τ 内，MAE 0.1436。

**判断**：条件结构作用不是零；符号反转和若干非零候选是真实的。但“非零 + 稳定 + 可复用”的规则很少，绝大多数可验证候选是 near-zero 的小修正，且 per-pair 异质性大。因此它更像“条件查表/局部修正”，还不是一个广泛可复用的共享结构功能规律。

---

## 路线标签

**主标签：`CONDITIONAL_COMPOSITION_SCREEN_ONLY`**

理由：严格遮蔽下 `LOCAL_MASKED` 稳定优于 `SHARED_MASKED`、`LOCAL_NULL_MASKED`、`ZERO`（7.76% / 9.22% / 8.49%，CI 均排除 0），方向在 context-equal 和 pair-type 上一致，LOO 最小增益 6.75%，不是单对驱动；但 observation 主口径只有 7.76%，未过预先声明的 10% 实用门槛。统计上真实，实用上只到 screen 级别。

**迁移限定标签：`CONDITIONAL_LOOKUP_ONLY`（作为 transfer qualifier，不是对严格遮蔽信号的否定）**

理由：真实 q 对直接 pair 查表有稳健增益（PAIR_Q < PAIR 且 < PAIR_Q_NULL），而在 all-conditional-enabled 与官方 Valid 未见 context 上 LOCAL 都不如 PAIR_Q；跨 context 最稳健的收益来自条件查表。严格遮蔽下仍有小幅组合信号，所以不能把整轮简单判成“只有 lookup、没有 composition”。

**未选择的标签及原因**：

- `CONDITIONAL_COMPOSITION_SUPPORTED`：未过 10% 实用门槛，保留 SCREEN_ONLY。
- `NO_SUPPORTED_CONDITIONAL_RULE`：符号反转 88.5% 复现，259 个条件候选可验证；信号不为零。
- `INSUFFICIENT_SUPPORT`：20/20 对可评估，21,970 条观测，支持充足。
- `RESULT_SENSITIVE_TO_IDENTITY_OR_MASKING`：clean subset 与 strict mask 方向一致；strict mask 相对 SHARED 仍有 7.76%，LOO 最小 6.75%，没有因身份清洗或遮蔽而反转。

---

## 1. 本轮做了什么

本轮只做协议内的 A–E，不设计新网络，不扩大数据：

- **A**：修正 canonical pair orientation（A=min rank，B=max rank，`delta=y(B)−y(A)`；原顺序相反时翻转 delta）；统一 observation-equal / context-equal / pair-type-equal 三个口径；修正 2×2 rectangle 计数；建立立体化学与重复标签 clean subset。
- **B**：在同一 evaluation observation 集合上比较 ZERO / PAIR / PAIR_MEAN / SHARED / PAIR_Q / PAIR_Q_MEAN / PAIR_Q_NULL / LOCAL / LOCAL_NULL，重复父轮 64 个结构、split、q、λ=5、seed=0 置换。
- **C**：严格遮蔽 ≤20 个结构对，重拟合 SHARED_MASKED / LOCAL_MASKED / LOCAL_NULL_MASKED；报告 10% 实用参考、LOO 和逐对结果。
- **D**：冻结 1M 模型迁移到官方 Valid；剔除与 1M 前缀重复的完整分子；分全体、未见 exact context、未见 context 且 q 受支持、已见 context、clean subset 报告。
- **E**：条件功能规律发现：global vs conditional residual 离散度、符号反转复现、条件候选与可读正反例。

### 数据规模

| 项目 | 数值 |
|---|---:|
| discovery pair observations | 297,996 |
| evaluation pair observations | 97,551 |
| canonical discovery pair types | 1,831 |
| P1 supported pairs | 1,602 |
| main evaluation observations / contexts | 96,991 / 38,768 |
| all-conditional-enabled observations / contexts / pairs | 64,321 / 30,304 / 1,267 |
| clean-subset observations / contexts / pairs | 49,135 / 18,004 / 1,412 |
| q keys | 1,056 |
| LOCAL 支持的 q / LOCAL_NULL 支持的 q | 114 / 114 |
| discovery unique (A,B,q) triples | 37,832 |

### 模型口径

- `ZERO`: 预测 0。
- `PAIR`: discovery 上 `(A,B)` 的 delta median。
- `PAIR_MEAN`: 同上，用 mean。
- `SHARED`: 上下文中心化加性拟合 `b_B − b_A`，λ=5。
- `PAIR_Q`: 对 `(A,B,q)` 有至少 10 个 discovery exact contexts 时取条件 median；否则回退 PAIR。
- `PAIR_Q_MEAN`: 条件 mean，同样回退。
- `PAIR_Q_NULL`: 在 discovery 内按 `(bond type, context row degree)` 分层、seed 0 置换 q，再按同样规则取条件 median。
- `LOCAL`: `b_B − b_A + d_(B,q) − d_(A,q)`；q 至少 10 个 discovery contexts，A/B 必须同组件，否则回退 SHARED。
- `LOCAL_NULL`: 同样的 LOCAL 估计，但用 seed 0 置换后的 q。

所有条件模型在条件不支持时回退到无条件 base；coverage 是“真正启用了条件修正”的比例。main 上 PAIR_Q coverage 77.56%，PAIR_Q_NULL 68.84%，LOCAL 99.02%，LOCAL_NULL 98.91%。

### 统计口径

- **observation-equal**：每条 pair observation 等权。
- **context-equal**：先算每个 exact context 的平均绝对误差，再对 context 等权平均。
- **pair-type-equal**：先算每个无序 pair type 的平均绝对误差，再对 pair type 等权平均。
- 所有关键差值用 **2,000 次 context-level paired bootstrap，seed 0**；同时报告 observation、context-equal、pair-type macro。
- Valid 上同样做 2,000 次 context-level paired bootstrap（observation/context/pair-type 均有 CI）；pair-type 点估计与 CI 均写入 `valid_transfer.csv`。

---

## 2. Phase A：方向、口径与数据修正

### 2.1 canonical orientation

修正规则：

```text
A_rank = min(rank_A, rank_B)
B_rank = max(rank_A, rank_B)
delta  = y(B_rank) - y(A_rank)
```

原存储顺序如果是 B→A，则 delta 取反。修正后：

- discovery 反序 rows：136,741 / 297,996 = **45.89%** → 0；
- evaluation 反序 rows：43,176 / 97,551 = **44.26%** → 0；
- discovery pair 中 median 符号被翻转：**846** 个。

`correction_table.csv` 与 `correction_details.json` 给出 v0/canonical 并列值。关键点：

| 项目 | v0 | 修正后 | 说明 |
|---|---:|---:|---|
| P1 MAE | 0.2621536 | 0.2621536 | 配对误差不变 |
| P1 RMSE | 0.4160078 | 0.4160078 | 不变 |
| P1 sign accuracy | 0.6471864 | 0.6471864 | 不变 |
| P3 SHARED MAE | 0.2714537 | 0.2714537 | 不变 |
| P3 LOCAL MAE | 0.2488080 | 0.2488080 | 不变 |
| P3 LOCAL_NULL MAE | 0.2731904 | 0.2731904 | 不变 |
| P2 可评估 pairs | 16/20 | 20/20 | 4 对恢复 |
| P2 observations | 18,584 | 21,970 | +3,386 |
| P2 LOCAL−SHARED 相对增益 | 8.87% | 7.82% | 方向不变 |
| Valid 可见 mu observations | 36,952 | 36,952 | 无覆盖损失 |

**为什么 P1/P3 不变**：pair 的 delta 和 PAIR 的 median 同时翻转，`|(-d) - (-p)| = |d-p|`；SHARED/LOCAL 的 A/B 索引在父轮拟合与评估中一致使用存储方向，所以 b 向量和差值不变。方向错误真正影响的是**逐对中位数的符号解释**、P2 的直接共现可见性、以及任何按 pair 方向阅读的规则。

### 2.2 统计口径修正

v0 的 `pair_equal` 不是 pair-type macro。以 P1 supported set 为例：

| 口径 | PAIR | ZERO | 相对下降 |
|---|---:|---:|---:|
| observation-equal（v0 `pair_equal`） | 0.2621536 | 0.2974965 | 11.88% |
| pair-type macro（真正按 pair 等权） | 0.3191122 | 0.3643207 | 12.41% |

两者数值不同，不能互换。本轮所有主表同时给出三个口径；`conditional_transfer.csv` 和 `strict_mask_results.csv` 均含 context-equal 与 pair-type-equal 行。

2×2 rectangle 的定义修正为：

```text
rectangles = Σ_{canonical (A,B)} k_AB * (k_AB - 1) / 2
```

其中 `k_AB` 是 discovery 中包含 A、B 的 exact context 数。本数据上为 **117,860,358** 个无序 context-pair（“rectangle”），不是 297,996 个 pair observations。

### 2.3 clean subset

- 立体化学标记：molecule key 中含 `@`、`/` 或 `\` 的分子剔除。
- 重复标签：同一 molecule identity 对应多于 1 个不同标签行的分子剔除；无 cut 的分子保留为 clean，但不会进入 pair 子集。
- clean molecules：418,603（其中 401,047 有至少一个 cut）；stereo molecules：567,621；duplicate-identity molecules：10,557。
- clean evaluation pair rows：49,384；main clean subset：49,135 obs / 18,004 contexts / 1,412 pairs。
- checks 验证：clean keys 不含 stereo 标记；clean 分子至多一行；clean pair rows 一致。

clean subset 的方向与 main 一致，没有出现“清洗后反转”：

- `PAIR_Q − PAIR = −0.01077`，CI `[−0.01239, −0.00925]`；
- `LOCAL − SHARED = −0.02575`，CI `[−0.02796, −0.02361]`；
- `LOCAL − PAIR_Q = −0.00201`，CI `[−0.00403, −0.000035]`（LOCAL 略好，但幅度很小）。

---

## 3. Phase B：统一 conditional transfer 比较

### 3.1 main set（PAIR 与 SHARED 均受支持）

96,991 obs / 38,768 contexts / 1,602 pairs。完整表：

| 模型 | observation MAE | context-equal MAE | pair-type MAE |
|---|---:|---:|---:|
| ZERO | 0.29750 | 0.26554 | 0.36432 |
| PAIR | 0.26215 | 0.23657 | 0.31911 |
| PAIR_MEAN | 0.27229 | 0.24587 | 0.33389 |
| SHARED | 0.27103 | 0.24575 | 0.33051 |
| PAIR_Q | 0.25112 | 0.22476 | 0.31378 |
| PAIR_Q_MEAN | 0.25987 | 0.23313 | 0.32509 |
| PAIR_Q_NULL | 0.26501 | 0.23919 | 0.32165 |
| LOCAL | **0.24844** | 0.22538 | **0.30166** |
| LOCAL_NULL | 0.27275 | 0.24746 | 0.33191 |

关键配对差（observation；括号为 95% CI）：

- `PAIR_Q − PAIR = −0.01103` `[−0.01208, −0.00989]`
- `PAIR_Q − PAIR_Q_NULL = −0.01389` `[−0.01503, −0.01273]`
- `LOCAL − SHARED = −0.02259` `[−0.02418, −0.02107]`
- `LOCAL − LOCAL_NULL = −0.02431` `[−0.02587, −0.02272]`
- `LOCAL − PAIR_Q = −0.00268` `[−0.00416, −0.00121]`
- `LOCAL − PAIR_Q_MEAN = −0.01143`（context-equal 与 pair-type 方向也一致）

### 3.2 all-conditional-enabled 子集

64,321 obs / 30,304 contexts / 1,267 pairs：这是 PAIR_Q、PAIR_Q_NULL、LOCAL、LOCAL_NULL 全部真正启用的“最有利条件”子集。

| 模型 | observation MAE |
|---|---:|
| PAIR | 0.24497 |
| SHARED | 0.25346 |
| PAIR_Q | **0.23183** |
| PAIR_Q_NULL | 0.24888 |
| LOCAL | 0.23779 |
| LOCAL_NULL | 0.25432 |

关键差：

- `PAIR_Q − PAIR = −0.01314` `[−0.01455, −0.01172]`
- `PAIR_Q − PAIR_Q_NULL = −0.01705` `[−0.01846, −0.01564]`
- `LOCAL − SHARED = −0.01566` `[−0.01698, −0.01436]`
- `LOCAL − PAIR_Q = +0.00596` `[+0.00480, +0.00716]`（**LOCAL 更差**）

这个子集是最重要的“反证”：当所有条件模型都有支持时，直接条件查表 PAIR_Q 比 LOCAL 更好；LOCAL 的优势不能简单归因于“共享结构功能”。

### 3.3 覆盖与回退

| 模型 | main coverage |
|---|---:|
| PAIR_Q | 77.56% |
| PAIR_Q_NULL | 68.84% |
| LOCAL | 99.02% |
| LOCAL_NULL | 98.91% |

PAIR_Q 的 4.21% 是“启用 q + 回退 PAIR”的混合结果；在启用 q 的 75,229 条观测上，PAIR_Q 相对 PAIR 的 observation MAE 降低 **5.71%**。LOCAL 的 coverage 很高，但其中大量 context 的 q 修正可能为零或很小，所以高 coverage 不等于高增量。

---

## 4. Phase C：严格遮蔽直接监督

### 4.1 选择与遮蔽

- 从结构支持排序选 20 对：direct discovery co-occurrence、至少 5 个 evaluation exact contexts；选择不看标签结果。
- 20/20 对可评估；21,970 条 eval obs、17,801 个 contexts。
- 每对删除 discovery 中所有同时含 A、B 的 context 里的 B cell；再删除这些 B 分子身份在所有切口的全部 cell。
- 删除的分子数 1,333–7,520；删除的 discovery cut rows 4,494–37,748，占 discovery rows 的 **0.42%–3.47%**（中位数 0.78%）。
- checks 验证：`direct_cooccurrence_after_mask=0`、`removed_molecule_cells_left_in_fit=0`、`connected_after_mask=1`；每对至少 5 个 eval contexts。
- `SHARED_MASKED / LOCAL_MASKED / LOCAL_NULL_MASKED` 在完全相同的遮蔽 discovery 数据上重拟合；NULL 的 q 在遮蔽后的 context 集合内按 `(bond type, masked context degree)` 分层、seed 0 重新置换。

### 4.2 结果

| 模型 | observation MAE | context-equal | pair-type |
|---|---:|---:|---:|
| ZERO | 0.17013 | 0.16874 | 0.20325 |
| SHARED_MASKED | 0.16877 | 0.16852 | 0.19941 |
| LOCAL_MASKED | **0.15568** | **0.15612** | **0.18002** |
| LOCAL_NULL_MASKED | 0.17150 | 0.17131 | 0.20156 |
| PAIR_FULL（参考） | 0.16265 | 0.16199 | 0.19145 |
| PAIR_Q_FULL（参考） | 0.14996 | 0.14984 | 0.17474 |

相对增益（observation）：

- vs SHARED_MASKED：**7.76%**，CI `[−0.01459, −0.01167]`（差值）。
- vs LOCAL_NULL_MASKED：**9.22%**，CI `[−0.01735, −0.01438]`。
- vs ZERO：**8.49%**，CI `[−0.01645, −0.01257]`。
- context-equal vs SHARED：7.36%；pair-type vs SHARED：9.72%。三个口径方向一致。
- LOO 最小增益：**6.75%**；最大单对贡献份额：**21.85%**；12/20 对为正、8/20 为负；per-pair 增益中位数 7.36%，范围 −6.64% 到 +37.26%。
- `PAIR_FULL` 的 MAE 比 `LOCAL_MASKED` 高 0.00697（相对 `PAIR_FULL` 4.28%）；`PAIR_Q_FULL` 的 MAE 比 `LOCAL_MASKED` 低 0.00572（相对 `LOCAL_MASKED` 3.67%，相对 `PAIR_Q_FULL` 3.81%）。后者有未遮蔽的直接条件监督，仅作参考，不作公平比较。

### 4.3 判定

所有统计条件均成立：

1. LOCAL_MASKED < SHARED_MASKED；
2. LOCAL_MASKED < LOCAL_NULL_MASKED；
3. LOCAL_MASKED < ZERO；
4. context-equal 与 pair-type 方向一致；
5. 20 对可评估，每对至少 5 个 eval contexts；
6. LOO 最小增益 > 0，不是单对驱动。

但 observation 主口径的 7.76% 低于 10% 实用门槛，因此判 **`CONDITIONAL_COMPOSITION_SCREEN_ONLY`**，而不是 `CONDITIONAL_COMPOSITION_SUPPORTED`。这不是“统计不显著”，而是“统计显著但实用幅度不足”。

---

## 5. Phase D：官方 Valid 迁移

### 5.1 数据处理

- 36,998 条 pair observations；36,952 条可评估；1,677 个 pair types。
- 剔除与 1M Train 前缀完整分子图重复的 Valid 分子（7,356 个 cells）。
- Valid cells 322,514；S 未映射 cells 4,525；重复分子 cells 7,356；使用 cells 273,379；contexts 211,356。
- 42 个 Valid q 未映射到训练 q 词汇；这些观测的 PAIR_Q 回退到 PAIR。
- 未使用任何 Valid label 做拟合或选择；Test labels 未读取。

### 5.2 结果

| 子集 | n obs | n contexts | n pairs | ZERO | PAIR | PAIR_Q | SHARED | LOCAL | LOCAL_NULL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all evaluable | 36,952 | 15,731 | 1,677 | 0.34791 | 0.29795 | 0.28200 | 0.30853 | **0.27296** | 0.30953 |
| unseen exact context | 6,765 | 4,911 | 798 | 0.22367 | 0.21117 | **0.20257** | 0.22927 | 0.21225 | 0.23025 |
| unseen context, q supported | 6,689 | 4,841 | 796 | 0.22395 | 0.21140 | **0.20270** | 0.22978 | 0.21256 | 0.23076 |
| seen exact context | 30,187 | 10,820 | 1,673 | 0.37576 | 0.31739 | 0.29980 | 0.32629 | **0.28657** | 0.32730 |
| clean subset | 24,381 | 9,585 | 1,440 | 0.35360 | 0.30158 | 0.28499 | 0.31436 | **0.27677** | 0.31494 |

未见 exact context 的关键差（observation）：

- `PAIR_Q − PAIR = −0.00860`，CI `[−0.01194, −0.00539]`；相对 PAIR 降低 4.07%。
- `LOCAL − SHARED = −0.01703`，CI `[−0.02084, −0.01311]`；相对 SHARED 降低 7.43%。
- `LOCAL − LOCAL_NULL = −0.01800`，CI `[−0.02188, −0.01404]`；相对 NULL 降低 7.82%。
- `LOCAL − ZERO = −0.01142`，CI `[−0.02009, −0.00296]`；相对 ZERO 降低 5.12%。
- `LOCAL − PAIR_Q = +0.00968`，CI `[+0.00614, +0.01302]`；LOCAL 的 MAE 比 PAIR_Q 高 4.56%（以 LOCAL 为分母；以 PAIR_Q 为分母为 4.78%）。
- `SHARED − ZERO = +0.00560`，CI `[−0.00206, +0.01298]`；全局可加模型没有可靠迁移收益。

### 5.3 解读

- 未见 context 上，LOCAL 相对 SHARED/NULL/ZERO 的改善说明“条件化结构修正”确实捕捉到了一部分可迁移信号。
- 但 PAIR_Q 在同一子集上最好，且 LOCAL 明显更差；因此这个信号不足以支撑“共享结构功能优于直接条件查表”。
- seen exact context 上 LOCAL 仍胜 PAIR_Q（4.41%），说明 LOCAL 的额外价值主要是对已见 context 的局部记忆/修正。
- clean subset 方向与全体一致：PAIR_Q 0.28499，LOCAL 0.27677，LOCAL 相对 PAIR_Q 好 2.89%（CI 排除 0）；但未见 context 上仍会反转。

---

## 6. Phase E：条件功能规律

### 6.1 离散度

221,334 条 supported rows、1,563 个 pairs：

| 指标 | global | conditional | 相对变化 |
|---|---:|---:|---:|
| pooled P80 \|residual\| | 0.44082 | 0.39457 | **−10.49%** |
| pooled median \|residual\| | 0.15783 | 0.13334 | **−15.52%** |
| per-pair P80 reduction 中位数 | — | — | **+4.53%** |
| per-pair P80 reduction IQR | — | — | `[−0.78%, +12.55%]` |
| 改善 pair 比例 | — | — | **69.1%** |

条件化确实降低了残差离散度，但 per-pair 中位数只有 4.53%，且约 31% 的 pair 没有改善。条件作用不是均匀的“全局规律”。

### 6.2 符号反转

| 阈值 | 反转 pairs | groups | 可验证 groups | 复现 | 复现率 | 双方向都复现的 pairs |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 271 | 1,257 | 765 | 677 | **88.5%** | 113 |
| 0.10 | 172 | — | 460 | 418 | **90.9%** | 75 |

符号反转是存在的：同一结构对在不同 q 下可以有相反符号，而且大部分在 evaluation 上复现。但注意：反转 pair 的绝对效应通常不大，且只占全部 pair 的一部分。

### 6.3 条件候选

| τ | triples | 可验证 | pairs | 覆盖 eval obs（全部/可验证） | 非零可验证 |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 104 | 64 | 47 | 957 / 861 | 0 |
| 0.10 | 428 | 259 | 183 | 4,389 / 3,981 | 7 |
| 0.20 | 1,516 | 959 | 548 | 23,795 / 22,491 | 27 |

τ=0.10 的 4,389 条覆盖是 main 的 **4.50%**；可验证候选覆盖 3,981 条（**4.10%**）。全局 pair 候选只有 **3** 个（415 obs），条件候选数量远多于全局候选，说明“条件化”确实能找到全局 pair 中位数看不到的结构。

但候选结构严重偏向 near-zero：

- τ=0.10 的 259 个可验证候选中，252 个是 near-zero；非零只有 7 个，覆盖 68 条 eval obs（约 main 的 0.07%）。
- τ=0.20 的非零也只有 27 个候选、406 条 obs。
- 这说明“非零、稳定、可复用”的规则存在，但很稀疏；大多数条件候选只是小修正。

### 6.4 可读例子

**非零稳定（τ=0.10）**

1. `*C(=C)C → *C=C`，q=`p6,0,0|8,0,0,1;8,0,0,2`
   - discovery median −0.1565，eval median −0.1333；
   - 68.4% eval 在 τ 内，MAE 0.1085；
   - 38 discovery contexts / 19 eval。
2. `*N → *NCC`，q=`p6,0,1|6,0,1,4;6,0,1,4`
   - discovery −0.1442，eval −0.1415；
   - 85.7% 在 τ 内，MAE 0.0676；
   - 25 / 7。
3. `*C → *Cl`，q=`p6,0,1|6,0,1,4;16,0,1,4`
   - discovery −0.1034，eval −0.1388；
   - 77.8% 在 τ 内，MAE 0.0584；
   - 37 / 18。

**符号反转**

`*C → *C(C)=O`：

- q=`p8,0,0|6,0,1,1`：discovery +0.2136，eval +0.1742；
- q=`p6,0,1|6,0,1,4;6,0,1,4`：discovery −0.7565，eval −0.6490；
- 两个方向均在 evaluation 上复现。

**失败例**

`*O → *OC`，q=`p6,0,0|6,0,1,1`：discovery +0.0218，eval +0.0068，只有 42.9% 在 τ 内，MAE 0.1436。说明候选并非都稳定。

---

## 7. 解释与边界

### 7.1 直接回答“查表 vs 共享结构”

- **直接条件查表（PAIR_Q）**：稳健、可迁移到未见 context，且明显优于 PAIR 和 PAIR_Q_NULL。它是本轮最可靠的 q 收益来源。
- **共享结构修正（SHARED/LOCAL）**：在已见/受支持 context 上有增量；严格遮蔽后仍比 SHARED/NULL/ZERO 好 7.76%/9.22%/8.49%，说明条件组合信号不是零。但这个增量在 all-conditional-enabled 和未见 Valid context 上都不如 PAIR_Q，且只有 7.76%，低于 10% 实用门槛。
- 因此最准确的结论是：**q 的主要可迁移收益来自条件查表；共享结构功能有真实但较小的增量，尚不足以判为 SUPPORTED。**

### 7.2 为什么不能把 LOCAL 的优势读成“泛化”

LOCAL 的公式是：

```text
LOCAL = SHARED + d_(B,q) - d_(A,q)
```

它需要 q 在 discovery 中有足够支持，并且 A/B 在同一结构组件。它在 seen context 上能记住 `d`，所以在 Valid seen subset 上仍胜 PAIR_Q；但在 unseen exact context 上，PAIR_Q 反而胜出。这个模式符合“局部条件修正”，不符合“可迁移的共享功能规律”。

### 7.3 为什么 10% 不是科学定理

10% 只是本轮预注册的实用门槛，用来防止把统计显著但幅度很小的信号推进到下一轮建模。严格遮蔽的 7.76% 在统计上稳健（CI 排除 0、LOO 最小 6.75%），但在实用上仍小；研究者可以选择接受或拒绝这个门槛，本轮不修改它。

### 7.4 身份/遮蔽敏感性

- clean subset 与 main 方向一致，没有反转。
- strict mask 与 full-data 的方向一致：全体 main 上 LOCAL 相对 SHARED 为 8.34%，严格遮蔽的 20 对子集为 7.76%；幅度略降但没有消失。
- 因此不选 `RESULT_SENSITIVE_TO_IDENTITY_OR_MASKING`。

---

## 8. 工程检查与可复现性

### 8.1 checks

`analysis/checks.json` 全部通过：

- canonical orientation：所有 A<B；delta 重算误差 0；500 个 context 的 cell 顺序随机置换后 canonical pairs/deltas 不变。
- parent fit equality：`shared_b` 最大差 `6.3e-14`；SHARED/LOCAL/LOCAL_NULL MAE 差 `<1e-14`。
- correction：反序行数、P1 MAE 不变、P1 观测数不变。
- strict mask：20/20 可评估；直接共现为 0；被删分子 cell 残留为 0；A/B 仍连通；每对至少 5 个 eval contexts。
- clean subset：clean keys 无 stereo 标记；clean 分子至多一行；clean pair rows 一致。
- identity row：200/200 抽样与官方 CSV/RDKit canonical isomeric SMILES 一致。
- synthetic controls A–D 全部通过（加性恢复、crossed local、strict-mask 间接路径、不可识别组件）。

`pytest`：`5 passed`（canonical orientation/delta、shuffle invariance、无反向重复、加性恢复、strict-mask 间接路径）。

### 8.2 运行预算

- CPU only，单进程，`OMP_NUM_THREADS=1`；未启动 GPU/Slurm。
- 计算阶段：prepare ~53 s；unified ~10 s；strict mask ~89 s；Valid ~6 s；candidates ~5 s；checks ~8 s；合计约 3 分钟。
- 从开始到报告完成约 35 分钟，低于 60 分钟硬上限；单进程未出现 OOM。峰值 RSS 未单独记录，但主要对象是 1M cells/100k pair 级别，远低于 64 GB 上限。
- 共享内存/磁盘：`prepared_pcqm.npz` 约 40 MB；本轮目录约 69 MB。

复现说明：`data/pcqm_labels.npy` 与 `data/prepared_pcqm.npz` 是父轮缓存派生的本地大文件，未提交到 Git；它们由 `code/sfec1_prepare.py` 在只读父轮缓存上重建。父轮目录未被覆盖或复制。

### 8.3 复现命令

```bash
cd tracks/structural-sufficiency/structural_functional_equivalence_followup_v1/code
uv run --project /home/calendar/code/research python sfec1_prepare.py
uv run --project /home/calendar/code/research python sfec1_unified.py
uv run --project /home/calendar/code/research python sfec1_mask.py
uv run --project /home/calendar/code/research python sfec1_valid.py
uv run --project /home/calendar/code/research python sfec1_candidates.py
uv run --project /home/calendar/code/research python sfec1_checks.py
uv run --project /home/calendar/code/research python sfec1_summary.py
uv run --project /home/calendar/code/research pytest -q test_sfec1.py
```

### 8.4 输出文件

| 文件 | 内容 |
|---|---|
| `PROTOCOL.md` | 预注册协议 |
| `configs/sfec_v1.json` | 配置 |
| `code/sfec1_*.py` | A–E、checks、summary |
| `code/test_sfec1.py` | 单元测试 |
| `analysis/correction_table.csv` | v0/canonical 修正并列表 |
| `analysis/correction_details.json` | 方向/覆盖/P2/Valid 细节 |
| `analysis/conditional_transfer.csv` | Phase B 主表、逐对、逐权重 |
| `analysis/conditional_transfer_by_pair.csv` | Phase B 逐对 |
| `analysis/strict_mask_results.csv` | Phase C 逐对与 pooled |
| `analysis/valid_transfer.csv` | Phase D Valid 迁移 |
| `analysis/conditional_function_candidates.csv` | Phase E 候选与例子 |
| `analysis/conditional_examples.csv` | 可读正反例 |
| `analysis/conditional_dispersion_by_pair.csv` | 逐对离散度 |
| `analysis/checks.json` | 工程检查 |
| `analysis/summary.json` | 机器可读汇总与标签 |
| `REPORT.md` | 本报告 |

---

## 9. 停止点

本轮已完成 A–E、checks、summary 和报告。没有启动下一轮模型实验，没有改阈值/数据/公式，没有读取 Test labels。

请研究者判断：

1. 是否接受 7.76% 的严格遮蔽增益作为“真实但只到 screen”的证据；
2. 是否把 `CONDITIONAL_LOOKUP_ONLY` 作为迁移层面的限定标签，还是只保留主标签 `CONDITIONAL_COMPOSITION_SCREEN_ONLY`；
3. 是否值得在下一轮专门检验“未见 context 上 PAIR_Q 优于 LOCAL”的边界，而不是继续扩大 LOCAL 模型。

在本轮约束下，建议的正式路线标签为：

```text
primary:  CONDITIONAL_COMPOSITION_SCREEN_ONLY
transfer: CONDITIONAL_LOOKUP_ONLY
```

本轮到此停止，等待研究者判断。
