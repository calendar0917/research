# cscl-v3b — 无损贡献规范化与可辨识性惩罚诊断（fit-only gauge normalization）

Track: `structure-contribution` · 日期: 2026-10-11 · 协议:
`configs/cscl_v3b_gauge.yaml`（冻结于正式 GPU 训练之前）
代码: `code/run_cscl_v3b.py`（唯一训练干预 + 审计 + 分析）、`code/contribution_gauge.py`（gauge 核心）
测试: `tests/test_cscl_v3b.py`（14 项）+ 既有 v3 14 项全部保持通过

> 研究命题：CSCL-v3 中逐类型 minibatch 均值惩罚是否人为压制了上下文修正 δ 的
> 有效变化，导致结构化贡献模型出现额外预测损失？能否通过**训练后保持预测不变**
> 的 fit-only 规范化，实现类型基础贡献与上下文贡献的分工，而不承担训练期惩罚？
>
> 唯一新模型 **S-gauge-seed0**；唯一机制变化 = 从训练目标中移除 δ 的 minibatch
> 逐类型均值惩罚（γ 项逐字保留，系数 1.0）。本轮不重训 F/O/S(v3)，不搜索任何
> 超参数，不访问 official valid/test。

正式 run：`cscl-v3b-s0-20261011-085706-7ae31a06`，res / a100-2 / GPU1（pool
res-gpu1，CUDA_VISIBLE_DEVICES=1），commit `a01968886525`，A100-SXM4-40GB，
driver 550.163.01，torch 2.5.1+cu124，exit 0，wall 272s（训练 240 epochs），
118,168 参数（与 v3 S 精确一致）。

---

## 0. 答案概要

1. **是**——v3 的逐类型 minibatch 惩罚期望值的 **93.6% 来自类型内方差泄漏项**
   `Σ_t w_t·σ_t²/n_{t,B}`，只有 6.4% 约束类型偏置（`Σ_t w_t·μ_t²`）。
2. **高频发生**——每个 batch 平均出现 58.3 个类型，但 **69.6% 的 (batch,类型)
   格子只有 1 个实例**（84.2% ≤2）；**32.4% 的单元实例**所在类型在其 batch 中
   只出现一次，这些实例的 δ² 以 1/N 的权重被**实例级**直接惩罚。
3. **是**——训练目标只差一项（测试钉住：split δ 惩罚 + split γ 惩罚 ≡ v3 组合
   惩罚，逐位相等；γ 项与"δ 置零后的 v3 惩罚"逐位相等）。
4. **是**——fit/monitor/dev 上 `pred_raw == pred_canonical` 最大差
   9.5e-7（门 1e-5）；规范化前后 MAE 在 1e-8 内一致。
5. **是**——fit 逐类型 canonical δ 均值最大 1.19e-7（门 1e-6）；类型内方差与
   同类型两两差逐位保持（实数恒等式，数值容差仅覆盖 float32 舍入 ≤1.4e-8）。
6. **dev MAE 0.17317**：S_gauge − S_v3 = **−0.02234** CI [−0.03092, −0.01435]；
   S_gauge − O = +0.03188 [0.02389, 0.04057]；S_gauge − F = +0.05675
   [0.04769, 0.06645]。
7. **部分是**——δ 恢复上下文变化（dev 类型内 std 0.0625 → 0.2423），γ 不再
   补偿类型信号（方差占比 40.2% → 2.6%，γ–unit 相关 +0.595 → −0.324）；但
   α 的语义由规范化约定承载（|m_t| 均值 0.657），其跨类型准确性未经真值证明
   （合成上 α Spearman 0.914，低于 v3 受约束版的 0.997）。
8. **支持**——修正后的合成任务：δ 恢复 raw 0.848 → **canonical 0.998**，
   δ̂–α* 混淆 0.432 → −0.013，真交互 +0.875/−0.704/+0.500（真值
   +0.9/−0.7/+0.5），混淆对 0.0045≈0，规范化前后预测一致 9.5e-7。
9. **是**——Full-Y 距离分桶 pair 一/二阶矩的压缩仍未解决（§6），O−F 与剩余
   S_gauge−O 差距的解释边界不变。
10. **下一轮单一问题**：完整保留 Full-Y 每距离桶 pair 矩的可分解读出设计
    （情况 B 剩余差距 S_gauge−O=+0.032 的首要候选解释）。

预冻结判断带读数：0.16 < 0.17317 ≤ 0.18 → **情况 B：移除 δ 惩罚只能部分恢复
能力**。不搜索惩罚系数；不追加训练。

---

## 1. Phase A.1 — v3 minibatch 惩罚的实际形态（无训练，冻结 v3-S + 精确重放）

`penalty_audit_s0.json`：重放正式 v3 训练的精确 batch 序列（seed 0，shuffle
generator 101，240 epochs × 57 batches = 13,680 个 batch；`torch.randperm` 的
CPU generator 与设备无关，重放忠实），配合冻结 v3-S 在 fit_inner 39,689 个
单元上的 δ 输出。

数学分解（对每个 batch 逐精确成立）：

```
E[Σ_t w_t·mean_{t,B}²] = Σ_t w_t·(μ_t² + σ_t²/n_{t,B})
                          └ 类型偏置约束 ┘   └ 类型内方差泄漏 ┘
```

| 量 | 值 |
|---|---:|
| 每 batch 出现类型数 | mean 58.3（min 13 / max 77 / p10 53 / p90 65） |
| (batch,类型) 格子中 n=1 占比 | **69.6%**（≤2 占比 84.2%） |
| 实例所在类型在其 batch 中单例的实例占比 | **32.4%** |
| 实际惩罚均值/batch | 0.02034 |
| 类型偏置部分 μ² | 0.00131（6.4%） |
| 方差泄漏部分 σ²/n | 0.01901（**93.6%**） |
| 实际惩罚 p90 | 0.03600 |

负担分布（epoch 级近似，组间相对比较有效）：top-10 高频类型承担 7.9e-5，
rank 11–50 承担 5.8e-4，rank 51+ 的 710 个稀有类型承担 9.9e-4——**稀有类型
因 n_{t,B} 小而单位实例负担远高于高频类型**。

单例格子的语义：当类型 t 在 batch 中只出现一次时，其惩罚项 = `(1/N)·δ_i²`，
即**实例级 L2 惩罚**，与"类型平均贡献归 α"的声明无关。

冻结 v3-S 在 fit 上的逐类型 δ（top-20 高频，`penalty_audit_s0.json`）：类型
均值 |μ_t| ≤ 0.125（多数 <0.01——类型均值确被约束），但类型内 std 高达
0.015–0.70；收敛时全 fit 惩罚 0.0071 vs fit L1 0.0883（惩罚量级约为损失的
8%，每步持续施压）。

**结论**：该惩罚名义上是"类型均值归 α"的约定，实现上主要在压制合法的类型内
上下文变化——训练期用强约束实现可辨识性的代价是真实的、可量化的。

## 2. 单变量训练干预（Phase B）

`train_gauge` 与 v3 `train_arm("S")` 逐行同构：相同 seed/init 流程、相同
collate、相同 batch 序列（generator 101）、Adam lr 1e-3 / wd 1e-5 / clip 5 /
batch 128 / 240 epochs、相同 checkpoint 规则（primary best-monitor 单检查点；
secondary top-5 monitor soup）。唯一差异：

```
v3:   loss = L1(pred, y_raw) + 1.0·(L_delta_batch_type + L_gamma_batch)
v3b:  loss = L1(pred, y_raw) + 1.0·L_gamma_batch
```

- γ 项实现逐字保留（`gamma_identifiability_penalty` ≡ `gamma.mean()**2`）；
- δ 项拆出为 `delta_identifiability_penalty`（逐字 = v3 的 δ 部分），训练中
  **不调用**（AST + 数值测试钉住，`tests/test_cscl_v3b.py`）；
- 监督仍是 raw y；冻结 Full-Y 预测从不进入输入或目标；
- architecture/参数量 118,168 与 v3 逐位一致（v3 checkpoint strict 加载测试）。

## 3. 正式结果（seed 0，primary = best-monitor 单检查点）

| 臂 | fit | monitor | **dev MAE** | best ep | params | wall |
|---|---:|---:|---:|---:|---:|---:|
| F（冻结，不重训） | — | — | **0.11641** | — | 408,651 | — |
| O（不透明读出） | 0.04755 | 0.14381 | **0.14129** | 236 | 138,753 | 342s |
| S v3（δ+γ 惩罚） | 0.08829 | 0.18686 | **0.19551** | 172 | 118,168 | 362s |
| **S-gauge（仅 γ 惩罚）** | 0.08023 | 0.15712 | **0.17317** | 188 | 118,168 | 272s |

配对差（分子级有符号 bootstrap，2000 次，seed 0）：

| 对比 | mean | 95% CI | S_gauge 更好比例 |
|---|---:|---|---:|
| S_gauge − S_v3 | **−0.02234** | [−0.03092, −0.01435] | 56.2% |
| S_gauge − O | +0.03188 | [+0.02389, +0.04057] | 43.8% |
| S_gauge − F | +0.05675 | [+0.04769, +0.06645] | 37.7% |

副指标（预注册）：soup dev S_gauge **0.16709**（v3 soup 0.19128）。

子群（`analysis_s0.json: subgroups`）：S_gauge 在**所有** y 带与拓扑子群上
优于 S_v3；最大恢复在 y≤−3 尾部（0.470 → 0.373）。与 v3 相同的薄弱点保留：
0–1 环小分子（n=12，S_gauge 0.597 vs O 0.292）与 y≤−3 尾部（0.373 vs O/F
0.316/0.315）——这些子群的剩余差距指向全局信息进入方式，不是 δ 惩罚。

**效应分离声明**：MAE 改善（−0.0223）来自训练期移除惩罚；训练后规范化本身
预测无损（§4，所有划分 pred 差 ≤9.5e-7、MAE 差 ≤3.8e-9）。两个效应分开报告，
未混淆。

## 4. fit-only gauge normalization（Phase C）

### 4.1 定义与执行

对每种类型 t，在 fit_inner 全部该类型单元实例上（eval 模式，float64 累积）：

```
m_t = mean(delta_raw | fit, type t);   alpha_canon(t) = alpha_raw(t) + m_t;
delta_canon(i) = delta_raw(i) − m_{t(i)}   ⇒   alpha_canon + delta_canon ≡ alpha_raw + delta_raw
```

- `m_t` 只用 fit_inner 单元与模型输出；不用 monitor/dev 标签或结构统计量；
- 覆盖：39,689 个 fit 单元；**760/761 个类型有 fit 实例**；唯一 fallback 类型
  是 id 760（GLOBAL_UNK 桶，m_t=0 确定性 fallback，1 个类型，fit/monitor/dev
  均无实例落入——`gauge_verification_s0.json: gauge_coverage`）；
- 类型频率按单元实例计数（非分子计数）；UNK 桶（fit 2,807 / monitor 393 /
  dev 941 个未知签名单元）与 fit-known 类型分别报告；
- 保存 `gauge_offsets_s0.pt`（offsets + 逐类型计数/哈希 + head state）+
  `gauge_meta_s0.json`（fit 行哈希、类型词表哈希、counts/offsets sha256），
  随 checkpoint 保存/加载（roundtrip 逐位一致测试）；
- 执行方式：`GaugedHead` 冻结包装（`alpha_embedding[type] + offset[type]`、
  `delta_mlp(...) − offset[type]`，offset 为 frozen buffer），**不修改已训练
  参数**；所有导出显式标注 raw / canonical，不混用。

### 4.2 验证门（全部通过，`gauge_verification_s0.json`）

| 门 | 结果 |
|---|---|
| pred_raw == pred_canonical（fit/monitor/dev） | max 9.5e-7（门 1e-5） |
| alpha+delta 逐单元恒等 | max 4.8e-7 |
| fit 逐类型 canonical δ 均值 = 0 | max **1.19e-7**（门 1e-6） |
| 类型内方差不变 / 同类型两两差不变 | ≤1.4e-8（实数恒等；容差仅覆盖 float32 舍入） |
| γ、g_ctx、g_topo、bias | **逐位不变**（diff = 0.0） |
| α 仍是共享类型标量（α_canon = α_raw + offset[t]） | 6.0e-8 |
| batch 组成不变性（dev 对半分 vs 全量） | 通过（≤1e-5） |
| save/load roundtrip | 逐位一致 |
| 导出严格加和（独立 numpy 重建） | canonical 1.00e-6 / raw 1.03e-6 |

### 4.3 不夸大的解释

该规范化只固定 **α 与 δ 之间按类型的常数平移自由度**。它不解决：α/δ 与 γ
之间的贡献转移（本轮由训练机制变化解决——见 §5 的证据）、关系项与全局项的
统计相关性、结构重叠上下文的归属歧义、真实化学因果、未见类型的稳定归因。
`mean_fit(delta_canonical|t)=0` 不等于整个分解唯一可辨识。

## 5. 贡献统计（raw 与 canonical 分别报告，`analysis_s0.json: S_gauge_audit`）

### 5.1 δ 恢复上下文变化、γ 停止补偿（对 v3 的机制对比）

| 量 | v3 S | v3b S-gauge |
|---|---:|---:|
| δ dev 类型内 std（实例加权） | 0.0625 | **0.2423** |
| δ dev pooled std | 0.191 | 0.359 (canonical) / 0.757 (raw) |
| γ dev std | 0.246 | **0.056** |
| γ 方差占比（Var/Var(pred)，描述性） | 40.2% | **2.6%** |
| γ–unit 通道相关 | +0.595 | **−0.324** |
| unit 通道方差占比 | 13.2% | 90.6% |
| g_ctx / g_topo 占比 | 0.9% / 13.1% | 0.6% / 12.7% |

v3 报告中"关系项方差占比最高（40%）、部分来自 δ 受约束后的贡献转移"的推断，
本轮得到直接验证：移除 δ 惩罚后，γ 收缩到 2.6%，类型级信号回到 unit 通道。
**注意**：通道占比是 Var(channel)/Var(pred) 的描述值，不是独立解释率、不总和
为 100%；canonical 下 unit–rel 相关 −0.324，通道间仍有统计相关。

### 5.2 m_t 与 α 的行为（诚实披露）

- **m_t 很大**：|m_t| 均值 0.657 / p50 0.463 / p90 1.60 / max 3.34。无训练
  约束时类型均值自由漂移；规范化把它显式吸收进 α（canonical α std 0.926、
  值域 [−5.34, +1.84]；raw α std 0.210、值域 [−2.09, +0.83]）。
- 这意味着"α=类型基础贡献"的语义在 v3b 中由**约定**（fit-only 均值）承载，
  而非训练目标强制。α 的跨类型排序因此未受任何真值约束；合成任务上 α
  Spearman 0.914（v3 受约束版 0.997）——**预测改善不自动等于 α 更准确**。
- 稀有类型规范化行为：所有 UNK 桶类型均有 fit 实例、获得正规 m_t；无实例的
  fallback 类型（GLOBAL_UNK）本轮在三个划分中均未出现（0 单元）。

## 6. Phase A.3 — 两个既有报告边界（只记录，不改特征/不重训）

1. **Full-Y reader 输入压缩（`phase_a_boundaries.json`）**：`MolFrozen` 只保存
   距离桶边缘化的矩。缺失信息清单：
   - 每个单元 × 每个距离桶的单元内 pair 一/二阶矩（extract 时的
     `intra_b_counts/sum/sq`，未存储）；
   - 每个单元对 × 每个距离桶的跨单元 pair 一/二阶矩（只存桶边缘化
     `rel_sum_pv/rel_sum_pv2` + 每对桶计数直方图）；
   - 因此冻结 reader 实际消费的"每桶图级矩池"无法从 S/O 输入在桶粒度重建：
     pair 值的桶条件幅度信息丢失（短距集中 vs 均匀分布在同 mean/rms 下不可分）；
   - 每对原始 E 行 / pair_value 行与 relation-encoder 输出属模型内部，本就不在
     S/O 输入契约内；图级 log1p count 块对 F 也被 C6 置零——非差异。
   本轮不补特征；这是 O−F=+0.025 与剩余 S_gauge−O=+0.032 的候选解释边界。
2. **v3 子群 `ring_units` 命名勘误**：`run_cscl_v3.subgroup_table` 的
   `ring_units` 行由 `nx.cycle_basis()` 计算（基环数），不是 RINGCHAIN 环系
   单元数。抽样 40 个 dev 分子中 11 个两种计数不同。本轮 v3b 子群表改用
   `KIND_RING` 单元计数，并保留 legacy 计数列显式标注；历史 v3 的
   "ring_units" 子群解读应读作"环复杂度代理"。

## 7. 修正后的合成已知真值任务（Phase D，CPU，seed 0）

**勘误**：v3 `synthetic_task` 用 `unit_desc[k, 4]` 当 attachment count——
实际读的是原子直方图 bin 4；正确槽位为 `D_HIST+4 = 36`
（`cscl_features` 布局：0–27 原子 / 28–31 键 / 32–39 标量，标量[4]=边界附着
数）。回归测试用 `build_partition().inter_relations()` 重建真附着数逐位钉住。
v3 合成结果保留，但其"attachment context"结论强度降级；修正后任务真值定义
已变，**不与 v3 合成数字做性能比较**。

修正任务（真实冻结特征，y_syn = Σα* + 0.8·(附着数 − fit 类型均值) + Σγ* +
0.05 噪声，γ*-only 惩罚目标，240 epochs）：

- **δ 恢复：raw r=0.848 → canonical r=0.998**；δ̂–α* 混淆 raw 0.432 →
  canonical **−0.013**。即：训练期不约束 δ 时，类型均值确实混入 δ（混淆
  0.43）；一次预测无损的规范化把混淆完全归位——**规范选择可以替代训练约束
  承担可辨识性约定**，这正是本轮的核心机制证据。
- α 符号恢复 0.938 / Spearman 0.914；γ̂ 真交互 +0.875 / −0.704 / +0.500
  （真值 +0.9/−0.7/+0.5，方向全对）；共现混淆对 γ̂=0.0045≈0；fit 未见单元对
  |γ̂|=0.013 vs 可见对 0.083（保守）。
- 规范化前后合成预测一致（gap 9.5e-7）；规范化后 α↔γ 混淆相关 0.022≈0；
  δ 幅值比（raw std/真值 std）1.29（无大幅压缩或膨胀）。
- 声明：合成结果是机制证据，不是 ZINC 化学因果证据；未据此调整任何真实任务
  的训练配置。

## 8. 科研判断

1. **训练期惩罚的代价被证实且可解释**：93.6% 的惩罚期望是方差泄漏；移除后
   dev MAE −0.0223（CI 不含 0），fit MAE 0.0883→0.0802，且 monitor 选择曲线
   全程更优（0.157 vs 0.187）。
2. **规范化成立**：所有 frozen 门通过，数学恒等式 + CPU 测试 + GPU 正式 run
   三层验证；checkpoint 可随 offsets 一起保存/加载。
3. **分工确实改变**：δ 恢复上下文变化、γ 的 40% 方差占比收缩到 2.6%、
   γ–unit 补偿相关从 +0.60 翻转为 −0.32。v3 报告第 6 条判断（"γ 的贡献部分
   来自 δ 约束的转移"）被本轮直接证实。
4. **但只是部分恢复（情况 B）**：S_gauge−O 仍有 +0.032 [0.024, 0.041]。剩余
   差距的可疑来源按本轮证据排序：(a) Full-Y 距离桶 pair 矩压缩（§6.1，S/O
   均受影响，解释 O−F=+0.025 的主体）；(b) 加法读出 vs O 的池化+MLP 容量差
   （0–1 环小分子 S_gauge 仍 0.60 vs O 0.29，n=12）；(c) 关系/单元交互表达
   限制。
5. **语义改善是部分的**：可辨识性约定从"训练期强约束"移到"训练后显式规范
   选择"是可行的（合成 δ 0.998、混淆 ≈0）；代价是 α 失去训练期锚定
   （m_t 大，α Spearman 0.914 < v3 0.997）。**本轮宣布：预测改善 + γ 补偿
   消除 + δ 恢复；不宣布 α/δ/γ 分解整体唯一可辨识**。
6. **下一轮单一问题**（不自动实施）：完整保留 Full-Y 每距离桶 pair 矩的
   可分解读出设计——这是情况 B 剩余差距的首要以输入信息为对象的解释，且与
   v3/v3b 两轮的子群证据（0–1 环、低 y 尾部）一致。

## 9. 边界与声明

- 单 seed 0；bootstrap 是固定训练模型在 dev 分子上的重采样区间，不是跨 seed
  稳定性证据。
- dev 已被多轮研究反复使用；本轮所有数字是开发证据，非假设检验。
- 判断带（0.16/0.18）是预冻结的工程尺度，不是显著性阈值。
- official valid/test 全程未加载（AST 测试钉住）；F 的预测仅用于 analyze 的
  配对比较，从不进入输入或监督。
- v3 的 checkpoint、预测、报告未做任何修改；勘误以追加形式记录（本文 §6/§7、
  `results/cscl_v3b/phase_a_boundaries.json`）。
