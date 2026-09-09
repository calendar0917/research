# Compact-v2 ZINC 信息缺口审计（Information Gap Audit）

> 诊断对象：`compact-hybrid-v2-budget-reallocation`（98,549 params，valid MAE **0.1841582** @ epoch 56，test MAE **0.1353615**）
> 预注册问题：**99k 参数下，模型系统性无法解释的信息是什么？**
> 本审计**只做诊断**：不训练任何新网络、不改 compact-v2 架构/损失/超参。唯一的新训练产物是 OOF 交叉拟合的基线协议复现（用于得到 out-of-sample 训练残差），以及线性残差探针（StandardScaler + Ridge）。
> 代码：`experiments/luyin16/zinc_information_gap_audit.py`；产物：`results/information_gap_audit/`。

---

## 1. 结论速览（TL;DR）

| 探针（信息类别） | dim | 残差 R² | 基线 MAE | 修正 MAE | ΔMAE | 判定 |
|---|---|---|---|---|---|---|
| null_control（管道校验） | 0 | −0.007 | 0.18416 | 0.18416 | 0.00000 | 管道正确 |
| rarity_stats（OOV/罕见度） | 14 | +0.0004 | 0.18416 | 0.18493 | −0.0008 | NO-GO |
| size_long_range（规模/长距离） | 20 | −0.0010 | 0.18416 | 0.18585 | −0.0017 | NO-GO |
| cycle_stats（环/环系） | 16 | +0.0002 | 0.18416 | 0.18580 | −0.0016 | NO-GO |
| global_counts（全局化学计数） | 66 | −0.0010 | 0.18416 | 0.18586 | −0.0017 | NO-GO |
| baseline_global_all_control（模型自带 62D） | 62 | −0.0010 | 0.18416 | 0.18580 | −0.0016 | NO-GO（头部已用但没利用好——余量也不在） |
| unary_token_hash_log1p（单侧 token 精确计数） | 2048 | +0.0007 | 0.18416 | 0.18541 | −0.0013 | NO-GO |
| pair_hash_presence（精确 patch 对出现与否） | 2048 | +0.0001 | 0.18416 | 0.18571 | −0.0016 | NO-GO |
| pair_hash_log1p（精确 patch 对计数） | 2048 | +0.0006 | 0.18416 | 0.18586 | −0.0017 | NO-GO |
| pair_relation_hash_presence（pair+距离桶） | 2048 | −0.0001 | 0.18416 | 0.18588 | −0.0017 | NO-GO |
| pair_relation_hash_log1p（pair+距离桶计数） | 2048 | −0.0001 | 0.18416 | 0.18609 | −0.0019 | NO-GO |
| pair_oov_exact_hash_log1p（OOV 全同一性 oracle） | 2048 | +0.0006 | 0.18416 | 0.18592 | −0.0018 | NO-GO |
| continuous_pair_interaction（h_i⊙h_j 桶均值） | 240 | −0.0013 | 0.18416 | 0.19341 | −0.0092 | NO-GO（过拟合） |
| pair_log+rel_log+continuous 组合 | 4336 | +0.0017 | 0.18416 | 0.18615 | −0.0020 | NO-GO |

**13 个探针全部 NO-GO；没有任何一个探针能把 valid MAE 降低 0.003 以上。**

在所有 13 个探针中最好的一个 `pair_log_rel_log_continuous_combined` 的残差 R² 也只有 **0.0017**——即使用 2048 维精确 patch 对哈希计数 + 距离桶 + 连续交互特征，valid 残差的方差解释率也趋近于零。这与「探针特征太弱」无关：连 baseline 已经喂给 head 的 62 维 global features 控件（作为残差线性回归的输入）都是零信号，说明**头部已有这些信息但残余误差并不是这些信息的线性函数**。

**解释：这些信息类别中不存在可提取的残差信号。模型剩余误差不是「缺失的稀有度/大小/环/全局化学/精确共现信息」导致的——残差对所有这些可观测结构特征都不敏感。**

---

## 2. 基线复现（bit-identity 门）✅

- 选择阶段（`validation-selection`）：best valid MAE **0.18415821571176638 @ epoch 56**，与晋升运行 `runs/2026/09/07/20260907-193612-46c1a12f/` 的 trace **逐 epoch 完全一致**（60 个 epoch 全部比对，0 不一致；epoch 60 = 0.196959 也一致）。
- 参数 98,549（与记录一致）；训练集中 vocab 6785（含 OOV id 0），valid typed 覆盖 **0.98796**（按 occurrence）。
- 训练内样本 MAE（in-sample，**LEAKY**，仅作参考）：**0.1091116**（10k 训练图）。
- OOF（5 折）训练 MAE：**0.1778873**（5 折均按同一基线协议训练：64k 训练图去掉 1/5 → 8k；验证集仅用于 early stopping，与基线一致）。

in-sample 0.109 vs OOF 0.178 的差距 = 训练集内插记忆空间，不是我们的诊断目标（我们只关心 OOF/valid 这类 out-of-sample 残差）。

---

## 3. 残差定义与泄漏控制

- **残差**：`r = y − ŷ`（signed residual，未截断）。
- **探针训练目标**：OOF 训练残差（每折模型用 8k 训练图、同协议、同 seed 训练，预测该折被扣留的 2k 图）。**5 折均为 out-of-sample**。
- **alpha 选择**：train-internal 5-fold CV（MAE scoring）+ SVD 形式 ridge，直接算出每个 alpha 的折外 MAE；**验证集只做 transform + evaluate**。
- **validation 探针评估**：修正预测 `y_hat + r_hat`。alpha 选择过程的全部信息来自 OOF 训练残差。
- **test**：仅在 `split` stage 作为描述性特征分布审计 + 事后打标签的 target 描述统计；**从未用于任何拟合**（test 没有预测任务，无 refit）。
- **fold 模型的近似（已记录的偏差）**：基线 selection phase 用 64k 图；fold 模型用 8k 图（训练数据量不同），因此 OOF 残差携带「少数据训练的噪声」。这是我们为获得无泄漏训练残差必须付的代价，方向保守（更弱的折叠模型 → 折叠残差更大 → 探针前景更乐观，而不是悲观——但即便如此所有探针仍为 NO-GO，说明真信号确实不存在）。
- hash 碰撞审计：2048 维下，pair/pair_relation/unary/pair_oov_exact 的近似碰撞率 ≈ 5.4e-8（每对 distinct key 的估计碰撞率），可忽略。

---

## 4. 诊断：残差到底来自哪里？

### 4.1 决定性发现：残差是「重尾 + 非对称」，几乎被极少数极端 target 分子主导

validation 残差统计：

| 指标 | 值 |
|---|---|
| 残差均值 | −0.0648 |
| 残差中位数 | −0.0132 |
| 残差 std | 0.7652 |
| corr(r, \|r\|) | **−0.943**（绝对值误差与 signed residual 几乎完全线性） |
| corr(r, ŷ) | +0.0385（**× 模型自身输出与残差无关**） |
| corr(r, y) | +0.4205（残差几乎只与真实 target 相关） |

**误差质量按 target 分位**（valid，1000 分子）：

| target 分位 | n | MAE | 占 MAE 质量 | 平均残差 |
|---|---|---|---|---|
| <1%（y<−6.11） | 10 | **3.752** | 20.4% | −3.713 |
| 1–5%（−6.11..−3.36） | 40 | 0.368 | 8.0% | −0.133 |
| 5–20% | 150 | 0.162 | 13.2% | −0.034 |
| 20–80% | 600 | 0.147 | 47.8% | −0.048 |
| 80–95% | 150 | 0.093 | 7.6% | +0.043 |
| 95–99% | 40 | 0.100 | 2.2% | +0.084 |
| >99% | 10 | 0.163 | 0.9% | +0.163 |

**y<−4 的 33 个分子（3.3%）占平方误差的 91.7%；y<−6 的 11 个分子（1.1%）占 88.6%**。y<−10 的 2 个分子（0.2%）单独贡献了 ~20.4% 的 MAE 质量。

单看质量最差的一分子 valid:0172：y=−20.34，ŷ=+0.45，|误差|=20.79——这一个分子的绝对误差就占全 valid MAE 的 ~11%。

### 4.2 回归到均值（regression-to-mean）是主导机制

- `ŷ = 0.070 + 0.838·y`（拟合全 valid）：**斜率 0.838 = 16% 收缩**。
- 对极端负 target，模型系统性地**高估**（向 0 收缩）：y<−6 的分子预测均值 −5.52（目标实际 −6.7 以下），y<−10 的分子预测均值 −4.43。
- 预测值范围 [−9.31, 3.40]，而训练 target 范围 [−42.0, 3.80]。**头部永远不会输出低于 −9.3 的预测**，即便训练数据里有 −42 的样本。在 L1 目标下，少量极端负样本被「平均掉」——模型用最小化中位误差的方式处理它们（对 99% 的常规分子最优）。

### 4.3 在常规区间内：残差 ≈ 不可约噪声 + 轻微稀有度异方差

排除极端 target（|y|≤2，n=751）后 MAE=0.1421，此时：

- 残差 std/MAE ≈ 1.34（高斯噪声应为 1.25）：形状接近对称重尾噪声（skew −1.30）。
- **稀有度异方差真实存在但很小**：在 |y|≤2 的常规区间内，rare5>0.048 的分子 MAE 0.228 vs rare5<0.048 的 MAE 0.108（比 2×）；OOV 比例与误差的 Pearson 相关 +0.38。
- 但注意两点：(a) 这是在**排除 73% 的误差质量后**的次生效应；(b) 稀有度与误差的相关**在 OOF 训练集内部同样存在**（|r| 与 rare5 的 Pearson +0.33），说明这是模型在稀有 patch 上的拟合不足（训练时 rare token 的梯度信号弱），**不是待补的信息缺口**——模型已经能看到这些 token id（它们都在 vocab 里），只是学得不够好。
- **任何以稀有度为输入的中位数纠正（binned-median probe）在 OOF 训练集内部最多只能带来 +0.0003 的 MAE 改进**——细粒度单调结构不存在，异方差只是方差而非可预测的偏差。

### 4.4 split 审计：特征与 target 分布无漂移

| split | n | 节点数中位数 | diameter 中位数 | OOV ratio 均值 | rare5 ratio 均值 | target 中位数 |
|---|---|---|---|---|---|---|
| train | 10000 | 23.0 | 12.46 | 0.000 | 0.039 | 0.441 |
| valid | 1000 | 23.0 | 12.50 | 0.013 | 0.045 | 0.423 |
| test | 1000 | 23.0 | 12.51 | 0.012 | 0.044 | 0.425 |

- 特征分布三者几乎一致（节点/边/直径/SP 距离/环秩/原子-键组成/稀有度，全部在 ~2% 内）；没有 split 漂移。valid/test 的 MAE 差（0.184 vs 0.135）**不能用特征分布差解释**。
- target 分布也基本一致（中位数/分位），**只是 valid 恰好有 2 个 y<−10 极端分子（min −20.34）而 test 没有（min −9.68）**：
  - 删掉这两个极端分子后 valid MAE ≈ 0.1478（y≥−6 时）；删到 y≥−4 时 valid MAE = 0.1396，已经非常接近 test 的 0.1354。
  - **valid MAE 相对 test 偏高的主要来源 = 极端负 target 分子的存在，而非模型在 test 上更差。**
  - 也就是说：0.184 valid / 0.135 test 的差距中，相当一部分不是「泛化差距」，而是**两个 split 的 target 尾部运气差**。

### 4.5 被否决的假设（皆 NO-GO）

| 假设 | 证据 |
|---|---|
| 补精确 patch 对共现信息能改善 | 2048D pair hash 探针 R²=0.0006/0.0001，ΔMAE −0.0017 |
| 距离桶条件 pair 信息 | 2048D pair_relation hash R²=−0.0001 |
| 环/环系统计缺失 | cycle_stats 16D R²=+0.0002 |
| 全局化学组成（原子/键计数）缺失 | global_counts 66D R²=−0.0010；且 baseline 已用其分数形式（global_all） |
| 规模/长距离拓扑缺失 | size_long_range 20D R²=−0.0010 |
| OOV 全同一性可救（oracle 上限） | pair_oov_exact R²=+0.0006 —— 即使把 OOV patch 的真实 id 全部揭开，也一样零信号 |
| 连续 pair 交互（h_i⊙h_j） | R²=−0.0013，修正 MAE 反而恶化 −0.0092（过拟合） |

---

## 5. Q1–Q10 回答

1. **Q1：模型剩余误差的主要来源是什么？** → 重尾回归坍塌。极少数极端负 target 分子（y<−4，3.3% 的 valid 分子）占平方误差的 91.7%、MAE 质量的 26.7%；模型对它们系统高估（向 0 收缩 16%，预测下限 −9.3 vs 训练 target 下限 −42）。
2. **Q2：OOV/罕见 patch 信息是不是缺口？** → 不是。稀有度与误差的关联真实存在（|r|–rare5 Pearson ≈ +0.33~0.38）但在**训练集内部同样存在**（模型已经见过这些 token，只是梯度信号弱、没学好）；它表现为**方差增大**而非可预测的偏差——binned-median 纠正最多 +0.0003 MAE。OOV ratio 在 valid/test 只有 ~1.2%（几乎全部 patch 都在 vocab 里）。
3. **Q3：全局化学/结构信息（尺寸、直径、原子-键组成）？** → 不是。62D global features 已经是模型输入；把这 62 维作为残差线性探针的输入（控件）信号为零，说明「信息在但头部没提取」的解释不成立——残余误差不是这些信息的函数。raw counts 探针（66D）同样为零。
4. **Q4：环/环系信息？** → 不是。cycle rank 与误差有弱相关（Spearman 0.15）但被极端 target 混淆；以环统计为输入的残差探针 R²=+0.0002。
5. **Q5：精确 patch-pair 共现（cosmic 压缩）？** → 不是。baseline pair 路径把 exact pair 压缩为 16D 投影 + 每桶矩；但即使把精确 pair 计数（2048D hash，碰撞率 5e-8）作为探针输入，残差信号依然为零。**精确 pair 共现信息与残差无关**，所以 head 的压缩没有信息损失。
6. **Q6：pair+距离桶条件信息？** → 不是（同上，pair_relation hash R²≈0）。
7. **Q7：连续 pair 交互（模型内部状态 h_i⊙h_j）？** → 不是且过度拟合：240D 探针在 OOF 上最好的 alpha CV MAE 0.1768（略优于 null 0.1779），但 valid R²=−0.0013，修正后 MAE 恶化 0.0092。这不是稳定信号。
8. **Q8：baseline head 已经用了这些信息吗？** → 部分。global 62D 是显式输入；pair 矩池化是 16D 嵌入的每桶 sum/sum²/log-count。探针显示这些维度的「残差可解释量」都为零，所以无论是「没输入」还是「输入了没用好」——都不存在可提取的残差（对线性探针而言）。
9. **Q9：train/valid/test split 之间有没有分布差异导致 valid 高？** → 特征无漂移；差异来自 target 尾部运气（valid 有 2 个 y<−10 分子，test 没有）。排除极端 target 后 valid MAE 0.1396，接近 test 0.1354。
10. **Q10：如果这些都不是残缺信息，那模型差在哪？** → 差在**容量/目标权衡**而不是信息：模型用 99k 参数把 99% 分子拟合到 ~0.14 MAE（|y|≤2 区间内已是 Gaussian 形状的不可约噪声），但在重尾负端选择「放弃」——L1 损失 + 有限的非线性容量，对每类只有 10–100 个样本的极端区域，最优解就是把它们拉向中位数。这是**任务定义/损失形状**问题，不是信息缺口问题。

---

## 6. 推荐的下一步实验（唯一）

**「重尾感知训练」消融（tail-aware training）**：在**不改结构、不改参数预算**的前提下，修复 L1 损失对重尾的坍塌——例如：
- **目标截断/缩放的 studentized L1**（对 |y| 大的样本降权或对 y 做对称化变换 w=softplus），或
- **分位数回归头（quantile head）**：只把 head 的最后一层换成 3 个固定分位输出（q10/q50/q90），保持其余不变，向 L1 目标加入区间惩罚。
- 判据（沿用本审计的协议）：valid/test MAE；**附带**：y<−4 子集 MAE 需从 ~1.49 显著下降，且 |y|≤2 子集 MAE 不恶化超过 0.005。

为什么是这个（而不是「加 pair 特征」「换矩阵分解」）：

1. 探针证明所有「补信息」路径不存在残差信号——增加输入信息没有可提取的收益；
2. 误差结构是**单侧重尾**（corr(r,|r|)=−0.94，y<−4 占 91.7% 平方误差），这是损失函数对尾部样本的容忍，不是表示能力不足；
3. 唯一自洽的解释是对称的：模型已经把所有常规信息用尽（|y|≤2 内残差是 Gaussian 噪声形状），却在尾部做均值回归。**直接针对损失形状**的改动可以同时保持常规区间表现，并解锁尾部。
4. 已有先例可以对照：compact-v2 是 L1；若 tail-aware 训练在 valid 上从 0.184 降到 ~0.15 以下（主要是尾部贡献），就能回答「差距来自优化目标而非信息」——而本审计的全部证据表明后者。

**注意**：这个实验属于模型开发，不是诊断——本审计到此结束，不包含该实验的执行。

---

## 7. 产物清单

| 文件 | 内容 |
|---|---|
| `results/information_gap_audit/compact_v2_validation_information_gap_audit.csv` | 1000 行 × 60+ 列的逐分子审计表（molecule_id、target、pred、signed_residual、absolute_error、全部统计列） |
| `probe_results.csv` | 探针汇总表（dim/alpha/CV MAE/R²/ΔMAE） |
| `probe_A_B_correlations.csv` | 逐统计量的 Pearson/Spearman 相关 |
| `bin_mae_*.csv` | 固定/分位数 binned MAE 表（oov、rare5、min_freq、log_freq、diameter、num_nodes、SP distance、cycle_rank） |
| `split_audit_summary.csv` / `split_audit_test_rows.csv` / `split_audit_target_descriptive.csv` | 三 split 描述审计（test 仅描述） |
| `baseline_valid_predictions.npz` / `oof_train_predictions.npz` | 逐分子预测（valid 为 selection 模型；train 为 OOF） |
| `baseline_train_in_sample_predictions.npz` | **LEAKY** in-sample 训练预测（明确标记，未用于探针） |
| `figures/fig1–fig5` | 5 张诊断图 |
| `selection_model.pt` | 复现的 selection 模型权重（bit-identical） |

## 8. 复现

```bash
uv run python -m tracks.ksvd.experiments.luyin16.zinc_information_gap_audit features
uv run python -m tracks.ksvd.experiments.luyin16.zinc_information_gap_audit baseline   # bit-identity 门
uv run python -m tracks.ksvd.experiments.luyin16.zinc_information_gap_audit oof       # ~18 min
uv run python -m tracks.ksvd.experiments.luyin16.zinc_information_gap_audit probes
uv run python -m tracks.ksvd.experiments.luyin16.zinc_information_gap_audit split
uv run python -m tracks.ksvd.experiments.luyin16.zinc_information_gap_audit figures
```

全部 stage 幂等（产物存在即跳过；`--force` 重跑）。环境：torch 2.5.1+cu124 / pyg 2.6.1 / sklearn 1.5.2 / CPU，约 40 分钟端到端。

---

## 9. Addendum (2026-09-08): Long-cycle / target-decomposition audit

正文中的 "heavy-tail regression collapse" 解读已被后续机制审计取代/精确化（完整报告：
`notes/zinc_long_cycle_audit.md`；产物：`results/zinc_long_cycle_audit/`）。

**核心更正：**

1. **极端尾部不是 generic heavy-tail regression 失败，而是 benchmark target 定义中的
   long-cycle penalty（全局拓扑项）**，目标公式源自 GVAE/Kusner et al. 2017
   （`y = z(logP) + z(SA) + z(cycle)`，`cycle = −max(0, max(nx.cycle_basis)−6)`，全池
   mean/std 归一化）——已逐分子复现：11,981/12,000（99.84%）精确一致。
   - valid:0172 y=−20.34 = 化学部分 +0.47 + cycle 项 −20.81；train:2210 y=−42.04 = −0.42 − 41.62。
   - y<−10：12/12 分子全部带负 cycle 项；y<−6：110/132（83.3%）；y<−4：222/461（48.2%）。
   - 移除 cycle 项后 valid kurtosis 12.62 → 0.24（y<−10: 2→0；y<−6: 11→2）。

2. **收缩关系 ŷ≈0.07+0.838y 的机制**：非整体 L1 均值收缩，而是 3.5% 长环样本导致的
   "学习饱和"（λ 阶梯 0.99/0.89/0.46/0.001；无环组斜率 0.989 ≈ 无收缩）。L1 是条件中位数
   回归（"mean regression" 表述不准确）；饱和头 + 罕见极端 target 下 MAE 最优解低估严重度
   ——与预测下限 ≈−9.3 一致。

3. **旧 cycle 探针保持 NO-GO 但需限定**：线性/16-D 探针（R²=+0.0002）仍 NO-GO；但
   本审计证明正确形状是**单调饱和阶跃**而非线性——isotonic 在 invariant 统计量上
   ΔMAE=+0.0117、在精确（非不变）cycle 项上 +0.0247。16-D 探针未测单调形状且用的是
   局部环统计，不是全局环尺度。

4. **新发现的 benchmark artifact**：cycle 项的值取 `nx.cycle_basis` 在**特定节点顺序**
   下的 max——顺序依赖：34/655 分子基础统计不稳（8.8% 极端尾部）、19/12,000 分子 label
   与 GVAE 顺序复现不一致；train:3776（存储序 basis 6→无罚）实际 label −6。任意
   permutation-equivariant 模型最多拿到可恢复信号中 **47%**（0.0117/0.0247），其余 53%
   是不变量模型原则上无法表达的 artifact。

5. **下一步（唯一）**：compact permutation-invariant 全局环尺度通道（exact
   longest-simple-cycle/环尺寸谱 + 单调 readout），预期上限 +0.0117；若 NO-GO，则
   benchmark target 重定义（不变 cycle penalty）可再解锁 +0.013。**不推荐** weighted L1 /
   quantile / 尾部重加权（尾部不是采样不均衡）。

---

## Addendum (2026-09-09): protocol wording

本文档首行的 "test MAE 0.1353615" 是 internal **train+valid refit** 口径（现命名：
secondary refit-robustness protocol）。Benchmark audit（Dwivedi et al./CIN）确认
literature ZINC 主协议 = train-only + validation-selected checkpoint + 单次 test
（selection-checkpoint test；v2 seed-0 = 0.154284，seeds 0–3 mean 0.146069 ±
0.006177）。详见 `compact_v4_multiseed_protocol_confirmation.md`。本审计的所有
机制结论（残差 = long-cycle target 项、探针全 NO-GO、重尾分解）不受命名影响。
