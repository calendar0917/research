# ESOC-v1 — ZINC 显式键对象组合实验报告

**问题**：在完全相同的主体（body）与训练配方下，显式保留键中心（bond-centred）诱导局部图中真实的**成对边对应关系**（pair-edge correspondence），能否提升性能与泛化？

**结论先行**：**能，而且提升显著、方向一致**；但**绝对水平仍未达到 DCC-L**，因此本轮判定为 `STRUCTURE_SIGNAL_ONLY`。

---

## 1. 执行状态

| 项目 | 状态 |
|---|---|
| 训练是否跑完 | **是**。REAL、NULL 两臂各 600/600 epoch 完整跑完，无截断、无中途放弃 |
| 官方 test 是否加载 | **从未加载**。`esoc_data.py` 的构建器对 `split="test"` 直接抛异常，并有 gate 验证 |
| 参数预算 | 498,845 / 500,000（两臂完全相同，逐张量 bitwise 同初始化） |
| 总 GPU 时间 | **5.07 h / 48 h**（正式跑 2.07 h + 弃跑 2.65 h + profiling/gates 0.35 h） |
| 协议修订 | 有 1 次，**在任何 verdict 之前**：2500 → 600 epoch（理由见 §6） |

---

## 2. 主结果（official valid，1000 分子，raw-y MAE）

| run | best-valid | @epoch | W20 | T100 | final-valid | train@best (eval) | 距 DCC-L |
|---|---|---|---|---|---|---|---|
| **REAL** | **0.12801** | 481 | **0.12978** | **0.13017** | 0.13031 | 0.02043 | +0.05202 |
| NULL | 0.15617 | 432 | 0.15792 | 0.15827 | 0.15784 | 0.02728 | +0.08018 |
| **REAL − NULL** | **−0.02816** | | −0.02814 | −0.02810 | −0.02753 | | |

配对 bootstrap（10,000 次，固定重采样，行级配对 AE 差）：

* mean Δ(REAL − NULL) = **−0.02816**，95% CI **[−0.03448, −0.02197]**（不跨 0）
* 60.5% 的分子 REAL 更优
* W20 与 T100 与 best-valid **三者同向**，不是单点幸运

**这是受控对比**：两臂只有 `edge_ij` 的构造不同（真实类型 one-hot vs 对象级 4 类频率向量广播），参数量、初始化（逐张量 bitwise 相同）、数据顺序、优化器、lr 轨迹、dropout 流全部一致。

---

## 3. 冻结分组（分组在结果之前固定，从不参与 loss/路由/选择）

| 分组 | n | REAL | NULL | Δ(REAL−NULL) | REAL 更优比例 |
|---|---|---|---|---|---|
| 签名未见（r2 unseen） | 251 | 0.19758 | 0.23284 | **−0.03526** | 0.574 |
| 签名已见 | 749 | 0.10470 | 0.13047 | −0.02578 | 0.615 |
| 普通分子（y > −3） | 935 | 0.08844 | 0.11419 | −0.02576 | 0.612 |
| 尾部（y ≤ −3） | 65 | 0.69724 | 0.75992 | −0.06268 | 0.508 |
| 环丰富（n_rings ≥ 4） | 213 | 0.15201 | 0.18750 | −0.03549 | 0.629 |
| 环丰富 ∧ 签名未见 | 47 | 0.39057 | 0.42053 | −0.02996 | 0.553 |
| cycle_rank ≥ 3 | 581 | 0.15275 | 0.18394 | −0.03119 | 0.635 |
| DCC-L worst-10 | 10 | 3.27445 | 3.33611 | −0.06165 | 0.600 |
| DCC-L worst-100 | 100 | 0.57500 | 0.60546 | −0.03046 | 0.550 |
| row 172 | 1 | 20.66586 | 20.73934 | −0.07348 | 1.0 |

**unseen − covered 的差之差 = −0.00948**（负号 = REAL 在未见签名上的优势更大）。

要点：
* REAL 在**全部**分组上都不差于 NULL，没有"某一类变差"的抵消。
* 优势在**未见签名**上更大（−0.0353 vs −0.0258），这是泛化证据而非记忆证据。
* 尾部（65 个分子）绝对误差极大（0.70 / 0.76），但 REAL 仍更好；这批分子是本任务的已知难点，不是本实验引入的。
* row 172 是极端离群点（AE ≈ 20.7），两臂都失败，Δ 不具解释力，仅按协议报告。

---

## 4. 结构敏感性诊断（冻结权重，只干预"配对"）

在 best checkpoint 上冻结权重，对 256 个 valid 分子做**保持类别直方图**的成对标签置换（每个对象的 4 类边计数完全不变，只有"哪个标签落在哪一对原子上"改变）：

| arm | 置换后平均 \|Δpred\| | 破坏对应后 \|Δpred\| | 预测改变的行比例 |
|---|---|---|---|
| REAL | **0.3554** | 0.2538 | 1.000 |
| NULL | **0.0000**（精确） | 0.0000（精确） | 0.000 |

* 置换**保持每个对象的类别直方图**（`permutation_preserves_class_histogram = true`），所以 NULL 臂（只吃直方图）必须且确实**精确不变**。
* REAL 对置换高度敏感 → 模型确实在使用**对应关系**，而不只是边类型的边际分布。
* 敏感性 vs 分子级 REAL 优势的相关只有 0.141：说明 correspondence 的收益**不是**靠"越敏感越好"的单一维度，而是分布式地体现在预测上。

---

## 5. 训练健康度（是否 underfit / 不稳定）

| 指标 | REAL | NULL |
|---|---|---|
| epochs 完成 | 600 / 600 | 600 / 600 |
| 最终 online train MAE | ~0.021 | ~0.028 |
| train@best（eval mode） | 0.02043 | 0.02728 |
| valid 曲线形态 | 单调下降至 ep500 后平台 | 同 |
| 是否出现 NaN / 发散 / 回退 | 否 | 否 |
| best epoch 位置 | 481 / 600（退火末期） | 432 / 600 |

train ≪ valid（0.020 vs 0.128）说明**模型容量足以拟合训练集**，valid 的差距是泛化差距而非欠拟合。best 落在退火末期而非早期尖峰，W20/T100/best 三者一致，排除"靠某个幸运 epoch"。

---

## 6. 协议修订（必须记录）

冻结的 2500 epoch 相当于 **780,000 updates**，是 DCC-L 参考配方（600 ep × 78 upd = 46,800 updates）的 **16.7 倍**。这是把"预算对齐"错做成了"epoch 对齐"。

更严重的是它让绝对数字不可比：2500-epoch cosine 在 ep300 时 lr 仍 ≈1.9e-3，而 DCC-L 此时已退到 ≈5e-4。所以中途读到的 0.158 测的是"还没退火"，不是"模型更差"。

**修订**：600 epoch（187,200 updates，仍是参考的 4 倍），warmup 12 epoch（保持 2% 的 warmup 比例不变）。模型、数据、loss、gates、指标、判定规则**全部不变**。修订发生在任何 verdict 之前；旧调度下的两次运行已停止、列在 `configs/esoc_v1.json` 与 `analysis/ledger.json`，其数字不作为结果报告。

同轮还修正了两处工程问题（不影响数学）：
* `compile_mode` 由 `default` 改为 `reduce-overhead`（19 s → 6.1 s/epoch），并有 `analysis/compile_equivalence.json` 的等价性证明：单步 loss 差 1.2e-7、220 个梯度张量最大绝对差 1.2e-7（相对梯度尺度）、更新后参数差 2.2e-8、REAL/NULL 初始状态差 0.0。
* 日志改为**每 epoch 一行**并每 epoch 增量重写 `curve.csv`，此前每 100 epoch 才有一行，`rr logs` 在中间十几分钟完全空白，无法区分"健康运行"和"卡死"。

---

## 7. Observed / Interpretation / Decision

### Observed（可直接核对的事实）

1. REAL best-valid **0.12801**，NULL **0.15617**，差 **−0.02816**，配对 95% CI [−0.03448, −0.02197]。
2. W20、T100、final、best 四个指标方向完全一致；60.5% 分子 REAL 更优。
3. 14 个冻结分组中 REAL 全部不劣于 NULL；未见签名上的优势（−0.0353）大于已见签名（−0.0258）。
4. 冻结权重下，保持直方图的配对置换使 REAL 预测平均移动 0.3554、100% 的行都动；NULL **精确移动 0.0**。
5. 两臂参数量 498,845 完全相同，初始状态逐张量 bitwise 相同。
6. 600/600 epoch 跑完，train@best 0.0204（REAL）/ 0.0273（NULL），无发散。
7. 总 GPU 5.07 h。

### Interpretation（我的判断，含不确定性）

* **对应关系本身是有效信号**：置换实验是最硬的证据——在类别直方图完全不变的前提下，REAL 的预测大幅移动而 NULL 精确不动。这排除了"REAL 只是多了边类型边际信息"的解释，因为两臂的边际信息在对象层面是同一个量。
* **优势是泛化的，不只是拟合**：未见签名上优势更大，且 valid 曲线在退火末期才到 best，不符合"记住训练集"的模式。
* **但绝对水平不够**：REAL 0.128 距 DCC-L 的 0.07599 还差 **+0.052**，是 DCC-L 的 1.68 倍误差。这不能归因于本轮的 correspondence 假设，而更可能来自**主体/读出设计**：本模型把每个键压成 64 维单 token、无递归、读出只用 sum/sum-of-squares，而 DCC-L 是 494k 参数的完整组合器。**这两个模型不是同一个东西**，Δ 不能解释为"显式对象组合比 DCC-L 差"。
* **不能声称的部分**：本轮**没有**复现 PPGT 的 0.0566（URPE 未实现，离线无法核实论文对齐），也没有做任何 test 评估。任何与 0.0566 的比较都不成立。
* **不确定性**：(a) 只有 **1 个 seed**，seed-1 未触发（见下），所以 −0.028 的**幅度**没有跨种子误差棒，只有行级配对区间；(b) valid 是暴露的开发集，行级 bootstrap 不是跨种子确认；(c) 600 epoch 的绝对水平可能仍受 horizon 限制——REAL 的 best 在 ep481、之后基本平台，但换更长 horizon 是否继续下降未知；(d) 尾部 65 个分子两臂都极差，本实验没有解决该问题。

### Decision

**verdict = `STRUCTURE_SIGNAL_ONLY`**

判定依据（对照协议第 6 节的词表）：
* REAL 明确且稳定优于 NULL（跨 W20/T100/final/14 个分组/置换诊断）→ 不是 `NO_USEFUL_GAIN`；
* 绝对性能未达 DCC-L 量级（0.128 vs 0.076，超出 0.079 阈值）→ 不是 `STRUCTURE_GAIN_AND_COMPETITIVE`；
* 训练正常完成、无 underfit/unstable → 不是 `UNDERFIT_OR_UNSTABLE`；
* 成本可负担（5.07/48 h）→ 不是 `COST_INFEASIBLE`。

**seed-1 配对未启动**，因为两条触发条件都不满足：
* (a) REAL best-valid ≤ 0.075 → 0.12801，**否**；
* (b) REAL ≤ 0.079 **且** REAL−NULL ≥ 0.004 **且** W20/T100 同向 **且** 尾部无灾难性退化 → 第二项与方向项满足，但 **0.12801 > 0.079，否**。

按协议"否则停止：不做 LR / depth / patch-range / readout 扫描"，本轮到此为止，不追加种子。

---

## 8. 与 DCC-L 的距离（绝对参考，非受控对比）

| | params | valid MAE | test MAE |
|---|---|---|---|
| DCC-L B-R0（4-seed 均值） | 494,193 | **0.075990** | **0.067319** |
| ESOC REAL s0 | 498,845 | 0.128010 | 未评估（禁止） |
| 差距 | +4,652 | **+0.052020** | — |

DCC-L 的数字是历史绝对值；两者在 token 集、局部编码、PE、读出、horizon 上都不同，**差值不是架构归因**。

---

## 9. 不确定性与未做的事

1. **单种子**。−0.028 的幅度没有跨种子误差棒。
2. **horizon 未做敏感性**。600 epoch 是按 updates 对齐后的选择；未测更长 horizon（协议禁止扫描）。
3. **URPE 未实现**，因此不是 PPGT 复现。
4. **NULL 不是"无结构"模型**：它仍保留对象级 4 类频率、完整原子/角色/节点特征与同一个 body，所以 Δ 测的是"对应关系"这一项，不是"结构 vs 无结构"。
5. **test 完全未触碰**，因此没有任何 test 数字。
6. **尾部 65 个分子**（y ≤ −3）两臂 AE 都在 0.70 以上，本实验未改善。

---

## 10. 交付物清单

| 文件 | 内容 |
|---|---|
| `PROTOCOL.md` | 冻结协议 + 编译等价性证据 + 2500→600 修订记录 |
| `configs/esoc_v1.json` | 最终配方（含 amendment 字段） |
| `code/esoc_data.py` | 静态对象缓存、`pairing_permutation` |
| `code/esoc_models.py` | LocalEncoder（REAL/NULL）、RRWP、AdaRMSNorm、sL2、ESOC |
| `code/esoc_train.py` | 训练器（每 epoch 日志 + 增量 curve） |
| `code/esoc_checks.py` | 12 组 gate（全 PASS） |
| `code/esoc_equiv.py` | 编译等价性检查 |
| `code/esoc_analyze.py` | 冻结分析（主表/配对/分组/敏感性） |
| `code/esoc_eval_ckpt.py` | 任意 checkpoint 的 valid 评估 |
| `analysis/cpu_checks.json` | 12/12 gate 通过 |
| `analysis/compile_equivalence.json` | 编译等价性（pass=true） |
| `analysis/main_table.json` | 主表 |
| `analysis/paired_contrasts.json` | 配对 bootstrap |
| `analysis/groups.json` | 14 个冻结分组 |
| `analysis/structure_sensitivity.json` | 置换诊断 |
| `analysis/per_molecule_valid.csv` | 逐分子预测与误差 |
| `analysis/ledger.json` | GPU 预算账本 |
