# CSC v0 执行报告 — 上下文条件结构计算与结构功能发现

- Round：`tracks/structural-sufficiency/context_conditioned_structural_computation_v0/`
- 协议：`PROTOCOL.md`（在任何正式训练之前冻结，commit `f6361d60`）；协议 id `context-conditioned-structural-computation-v0`
- 冻结件：`configs/experiment.json`（配方、架构、参数量、队列、数据身份、batch 计划 hash）
- 数据：`ogbg-mollipo`（Train 3360 / Valid 420 / Test 420）、`ogbg-molesol`（902 / 113 / 113），官方 scaffold split，未删分子、未重划分、未增广；归一化只用 Train
- Test 角色：`DEVELOPMENT_TEST`。每 epoch 全量评价、进入全部诊断，但**不参与**任何选择（checkpoint 只按最小 Valid RMSE，平局取更早 epoch）
- 主指标 RMSE（原始标签单位），MAE 为次要；11 项正确性检查在两个数据集上全部通过（`analysis/checks.json`，`all_passed=1`）
- 实际训练：7 个单元 × 360 epochs，全部 `COMPLETED`，无早停/无降级/无中断；**新训练 GPU·h ≈ 1.107**（目标 ≤2、硬上限 3）；训练墙钟 04:21–04:57 UTC（≈36 min），远小于 2 h 上限
- 独立复算：从提交的 `checkpoint_best.pt` 重新加载并全量评分，`mollipo A`（ep140）、`mollipo D`（ep148）、`molesol A`（ep265）的 Valid/Test RMSE 与 `run.json` 记录**逐位相同**（0.7267/0.7166、0.7389/0.7300、0.8601/0.8302）；21 个结果文件的 SHA-256 与 `run.json` 记录全部一致

---

## 第一页：八个问题与判定

**Q1｜把结构内容与环境调制分离（D：真实关系分组 + Gate）是否提升 Lipo 的图级预测？**
否。D−A 的 Valid RMSE 差 **+0.0123（+1.69%，D 更差）**，Test **+0.0134（+1.87%，D 更差）**；两个 split 的分子 bootstrap 95% CI 都含 0。方向上不但没有改善，而且两个 split 同号（都更差）→ 不满足协议要求的"改善 ≥2% 且两 split 同向"。

**Q2｜保留"结构内容↔真实关系"对应性（D）相对去掉该对应性但保持计数/环境/参数量（C）是否有价值？**
否，且信号自相矛盾。D−C：Valid **+0.0041（+0.56%，D 更差）**、Test **−0.0143（−1.92%，D 更好）**，CI 均含 0，符号在 split 之间翻转。分层看同样翻转：Valid 上 D−C 在"多关系邻居多"的分子上更好（−1.34%）、在其余分子上更差（+4.19%），Test 上恰好相反（+2.63% / −5.16%）。→ 真实关系分组的对应性**没有**可复现的预测价值。

**Q3｜第二次真实邻居读取本身是否有价值（A vs B），是否应把结论记为 `FIXED_SOURCE_ONLY`？**
A（FIXED1，Q=0）在两个 split 上都优于 B（B 是线性读取真实邻居 `LN2(h0_i)` 的第二阶段）：Valid **−0.75%**、Test **−4.93%（Test CI 不含 0，55% 分子改善）**。方向一致为正，但**主决策 split（Valid）未达到 2% 参考**，因此按预注册口径不判 `FIXED_SOURCE_ONLY`，只记录"第二轮真实邻居读取在 Test 上明确有害、在 Valid 上无收益"。

**Q4｜辅数据集 ESOL 是否复现同一图景？**
是"无差别"的复现：七对比较里最大的 |Δ| 只有 0.82% 相对 RMSE，全部 CI 含 0。D 在 Valid 略好（−0.46% vs A）、Test 略差（+0.55%）；C 与 A 实质相同（+0.13% / −0.26%）。ESOL 既不支持也不反驳 D。

**Q5｜若存在增益，是否广泛（不是被少数分子/异常样本驱动）？**
没有需要解释的增益。反向检查同样通过：D−A 的 Test 差在去掉 |Δ| 最大的 1/3/5 个分子后仍为 +0.0205/+0.0186/+0.0190（方向不变），best 与最后 20 epochs 均值方向一致，Train–Valid gap 四臂相近（0.42–0.46 Lipo），说明 D 的劣势不是个别异常分子或 horizon 选择造成的。反过来，D−C 在 Test 上 −1.92% 的微弱优势**不稳定**：去掉 3 / 5 个最大 |Δ| 分子后翻成 +1.13% / +2.27%（`direction_preserved=0`），即该优势由极少数分子支撑；C−A 的 Test +3.86% 同理（去掉 3 个后仅 +0.30%）。

**Q6｜模型真的"使用"了上下文条件调制吗（冻结干预 I0/I1/I2）？**
没有，而且干预方向与性能结论自洽：在 Lipo D 上把 `Q_i` 置零（I0）反而更好（Valid 0.7343 vs 0.7389、Test 0.7264 vs 0.7300）；把 Gate 输出冻结为 Train 关系加权均值 `δ̄_r`（I2）拿到最好的 Test（0.7217）；把分组内容换成 `P_null`（I1）影响很小（|Δpred| rms 0.020）。ESOL D 同向（I0 Valid 0.8547 < 0.8561）。→ 学到的逐节点环境调制是噪声，按关系类型的"平均调制"已足够，甚至更好。

**Q7｜条件化结构贡献能否被精确追踪到预测变化（TRACE）？**
能，且在两个数据集上都闭合。逐消息精确分解 `h2 = h2_base + Σ_j v_ji` 与路径积分归因（16 点 Gauss–Legendre，`A_ji = ∫₀¹ ∇F·v_ji dt`）的闭包误差最大 **1.2×10⁻⁶**（容差 `1e-4+1e-3|Δ|`，超容差条数 0），状态重建误差 ≤ 1×10⁻⁶；新增修正项 `A_extra` 从原始 FIXED1 状态出发同样闭合（Lipo D Valid：|Δpred| rms 0.064，闭包误差 1.3×10⁻⁶）。作用高度分散：top 1% 消息只占总 |A| 的 9%，top 10% 占 41%，正负各半（47% 为负）。

**Q8｜能否从中归纳出可迁移到其他分子的简单结构规律或结构功能候选（RULE_REUSE / STRUCTURE_DISCOVERY）？**
否。固定复杂度解释树（depth 4、min_leaf 100、分子等权）在 Lipo 上的分子等权归因 MAE 相对常数预测器的 skill 为 **−0.008（A，S）/ +0.001（D，SC）**，即在未参与拟合的 Valid/Test 上**没有**可解释的方差；SC 相对更强单特征组的改善只有 0.05%（A）/0.07%（D），远低于协议 10% 的参考。5 条 Train 候选规律在 Valid 上的覆盖率为 0.0–0.2%，同号率不稳定（0.00/0.20/0.50…）。重复结构（精确签名 + 真实键类型）中位数符号变化比例 0.80（1.0 为完全正负各半），作用幅度与族内 IQR 之比 0.16–0.17。→ 结构规律发现**失败**，这是本轮的一个明确负结果（协议允许），未做任何放宽。

### 判定（按 `analysis/verdicts.json`，报告不平均化冲突）

| 轴 | 判定 | 依据 |
| --- | --- | --- |
| 性能 | **`NO_MATERIAL_GAIN`** | D−A、D−C 均未达到 2% 相对改善；方向多为更差；ESOL 全部 \|Δ\|<1% |
| 可解释性 | `TRACE` = 是；`RULE_REUSE` = 否；`STRUCTURE_DISCOVERY` = 否 | 闭包精确；树 skill≈0；重复签名符号不稳定 |
| 联合归属 | **两者均无实质改善**（五选一中的第 4 项） | 性能无增益且无可归纳的结构作用 |

补充（不改变判定）：唯一跨 split 稳定的现象是"**第二轮真实邻居读取（B）比只读固定来源（A）更差**"，Test 上达到 −4.93%（CI 不含 0）；它说明在本配方下把 `h0` 再送进第二阶段会引入噪声，但不足以按预注册阈值宣布 `FIXED_SOURCE_ONLY`。

---

## 1. 本轮做了什么

四臂共享完全相同的骨架与初始化来源（`BSModel` 1× 宽度，d=64、q=128；`h0 = Dropout(LocalContent(G))`），第一阶段（MessageFFN 在真实有向键上的消息传递）四臂逐位相同：

| 臂 | 第二阶段读取 | 关系分组 | 环境调制 | 总参数 |
| --- | --- | --- | --- | --- |
| **A** FIXED | `LN2(h0_j)`（固定来源一跳，等价原 FIXED1） | — | 无（Q=0） | 294,135 |
| **B** ONE_READ | `LN2(h0_i)`（接收端自身） | — | 无 | 294,135 |
| **C** MIXED | `LN2(h0_j)` | `P_null_{i,r} = n_{i,r}/n_i · B_i`（保计数、保总和） | Gate | 307,095 |
| **D** CSC | `LN2(h0_j)` | `P_{i,r} = Σ_{j:rel=r} p_{ji}`（真实键类型） | Gate | 307,095 |

`δ_{i,r} = 0.5·tanh(Gate(c_i))`，`Q_i = Σ_r δ_{i,r} ⊙ P_{i,r}`，`h2 = h1 + fc2_2(SiLU(a+B) + Q)`；C/D 参数、初始化、输入、Gate 结构完全一致，只差分组内容的定义。检查 5/7/10 分别验证了 C/D 全初始化一致、`δ=0` 时逐位退回 A、以及 C/D 只在"邻居关系异质"的原子上产生不同输出。

训练配方对四臂逐项冻结（seed 0、batch 128、360 epochs、AdamW 3e-4→1e-5 per-step cosine 无 warmup、wd 1e-3、clip 5、FP32、dropout 0.10、标准化目标 MSE、每 epoch 全量三 split 评价）；同一数据集各臂共用同一 batch 顺序计划（Lipo `f9c1398c…`、ESOL `7ea312ba…`，见 `analysis/checks.json`）。

## 2. 主表（best = 最小 Valid RMSE；`analysis/main_table.csv`）

**ogbg-mollipo**

| 臂 | best ep | Valid RMSE | Valid MAE | Test RMSE | Test MAE | Train RMSE (best) | last20 Valid | last20 Test | 参数 | GPU·h |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A FIXED | 140 | **0.7267** | **0.4984** | **0.7166** | **0.5323** | 0.3080 | 0.7407 | 0.7317 | 294,135 | 0.167 |
| B ONE_READ | 154 | 0.7322 | 0.5089 | 0.7539 | 0.5602 | 0.2907 | 0.7546 | 0.7340 | 294,135 | 0.167 |
| C MIXED | 156 | 0.7348 | 0.5212 | 0.7443 | 0.5491 | 0.2790 | 0.7658 | 0.7413 | 307,095 | 0.173 |
| D CSC | 148 | 0.7389 | 0.5182 | 0.7300 | 0.5591 | 0.3111 | 0.7510 | 0.7355 | 307,095 | 0.331 |

**ogbg-molesol**

| 臂 | best ep | Valid RMSE | Valid MAE | Test RMSE | Test MAE | Train RMSE (best) | last20 Valid | last20 Test | 参数 | GPU·h |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A FIXED | 265 | 0.8601 | 0.6561 | **0.8302** | 0.6601 | 0.3227 | 0.8805 | 0.8505 | 294,135 | 0.040 |
| C MIXED | 265 | 0.8612 | **0.6558** | 0.8280 | 0.6625 | 0.3177 | 0.8817 | 0.8468 | 307,095 | 0.113 |
| D CSC | 200 | **0.8561** | 0.6635 | 0.8348 | 0.6617 | 0.3303 | 0.8744 | 0.8459 | 307,095 | 0.116 |

Lipo 上 A 在两个 split 都最好；ESOL 上 D 的 Valid 好 0.46%、Test 差 0.55%（互相抵消）。四臂 Train–Valid gap 相近（Lipo 0.419 / 0.442 / 0.456 / 0.428），说明差异不是"拟合得更狠"造成的。

## 3. 配对比较与稳健性（`analysis/paired_comparisons.json`）

2000 次分子重抽样 bootstrap（scaffold 聚类 bootstrap 并列，结论一致）：

| 对比 | split | ΔRMSE | 相对 | 95% CI（分子） | 改善/退化分子比例 | 去 top-1 dse 后 |
| --- | --- | --- | --- | --- | --- | --- |
| A−B | Valid | −0.0055 | −0.75% | [−0.0297, +0.0225] | 0.54 / 0.46 | −0.0109 |
| A−B | Test | −0.0372 | **−4.93%** | [−0.0648, **−0.0097**] | 0.55 / 0.45 | −0.0325 |
| C−A | Valid | +0.0082 | +1.12% | [−0.0193, +0.0358] | 0.44 / 0.56 | +0.0151 |
| C−A | Test | +0.0277 | +3.86% | [−0.0106, +0.0660] | 0.46 / 0.54 | +0.0185 |
| D−C | Valid | +0.0041 | +0.56% | [−0.0237, +0.0283] | 0.50 / 0.50 | −0.0011 |
| D−C | Test | −0.0143 | −1.92% | [−0.0516, +0.0222] | 0.49 / 0.51 | −0.0043 |
| D−A | Valid | +0.0123 | +1.69% | [−0.0253, +0.0432] | 0.46 / 0.54 | +0.0232 |
| D−A | Test | +0.0134 | +1.87% | [−0.0174, +0.0440] | 0.43 / 0.57 | +0.0205 |

ESOL 全部 4 个对比、2 个 split 的相对差 |Δ| ≤ 0.82%，CI 全含 0（D−A：−0.46% / +0.55%；D−C：−0.58% / +0.82%；C−A：+0.13% / −0.26%）。

其它稳健性：best 与 last-20 均值方向一致（`best_vs_last20_direction` 全部 `same_direction=1`）；去掉 top-1/3/5 个 |Δ| 最大的分子后所有对比方向不变；按 Train 中位"关系异质原子比例"分层的结论在 Valid/Test 之间符号翻转（见 Q2），因此不作为机制证据。

## 4. 机制诊断与冻结干预（`analysis/mechanism_diagnostics.json`）

Gate 在 Lipo D 上确实在学、也确实"用力"：`δ` 的 |·| 均值 0.169、RMS 0.205、tanh 饱和率 0.9%，`Q` RMS 0.156（相对 `h2−h1` 的 RMS 0.194），逐关系 Train 加权均值 `δ̄_r` 的 RMS：AROMATIC 0.193、DOUBLE 0.218、SINGLE 0.158、TRIPLE 0.243。异质邻居覆盖：Valid 上 21.7% 的原子 / 85.8% 的分子至少有两种关系类型；"真实分组 vs 混合分组"内容差异在 35.7% 的 (原子, 关系) 组上非零（ESOL D 35.7%，作为对照 C 恒为 0，说明对照组实现了设计意图）。

冻结干预（D、best checkpoint、全部参数与 FIXED1 主干不变）：

| 干预 | 含义 | Valid RMSE | Test RMSE | MAE Valid | \|Δpred\| rms (Valid) |
| --- | --- | --- | --- | --- | --- |
| baseline | 完整 D | 0.7389 | 0.7300 | 0.5182 | — |
| I0 | `Q_i = 0` | **0.7343** | **0.7264** | 0.5143 | 0.0772 |
| I1 | `P_{i,r} → P_null` | 0.7391 | 0.7285 | 0.5174 | 0.0197 |
| I2 | `δ_{i,r} → δ̄_r`（Train） | **0.7341** | **0.7217** | 0.5124 | 0.0502 |

ESOL D：baseline 0.8561 / I0 0.8547 / I1 0.8562 / I2 0.8563（Valid），同向。

读法：把条件修正**去掉**或**冻结成按关系类型的常数**都会让模型略好，而保留逐节点的环境调制则更差 → 该调制在预测上没有被有效利用。这解释了为什么 D 不优于 A/C：额外自由度只带来了方差，没有带来可用的条件信息。

## 5. 精确分解与路径积分归因（`analysis/decomposition_trace.json`，npz 在 `results/decomposition/`）

逐消息精确分解（Lipo best checkpoint，三个 split 全部计算）：

| 项 | A FIXED | D CSC |
| --- | --- | --- |
| 消息数（train/valid/test） | 197,122 / 25,170 / 25,506 | 同 |
| `h2 = h2_base + Σ_j v_ji` 重建误差（max） | 7.2×10⁻⁷ | 9.5×10⁻⁷ |
| `h2 = h2_fixed1 + Σ_j v_extra_ji` 重建误差（max） | 0（无修正项） | 2.4×10⁻⁷ |
| 路径积分闭包误差（max，Valid） | 1.1×10⁻⁶ | 1.2×10⁻⁶ |
| 超容差分子数 | 0 | 0 |
| 新增修正 `A_extra` 闭包误差（max，Valid） | — | 1.3×10⁻⁶ |
| 归因集中度（top1% / top5% / top10% 消息占 \|A\|） | 8% / 26% / 40% | 9% / 27% / 41% |
| 负号比例 | 0.47 | 0.45 |
| 平均 \|A\|（原始单位） | 0.0103 | 0.0134 |
| 修正项占 \|A\| 总量 | 0 | 31%（Valid） |

按关系类型的 |A| 份额（Valid）：AROMATIC 60%（D）/ 58%（A）、SINGLE 30%/32%、DOUBLE 8%/9%、TRIPLE 1%/0.7%。归因在分子内高度分散，不存在"少数关键消息决定预测"的结构。

## 6. 结构规律与结构功能发现（`analysis/structure_rules.json`、`analysis/explanation_cases.json`）

精确结构签名（原子特征 + S1 邻居码多重集 + S2 路径码/端点/闭环多重集）× 真实键类型：

| 指标 | Lipo A | Lipo D |
| --- | --- | --- |
| 重复消息比例 | 0.994 | 0.994 |
| 重复类型数（≥3 条支持） | 3,962 | 3,962 |
| 符号变化比例（中位数，1=完全各半） | 0.80 | 0.80 |
| 作用幅度 / 族内 IQR（中位数） | 0.17 | 0.16 |

解释树（depth 4、min_leaf 100、random_state 0、分子等权，仅 Train 拟合）：

| 输入 | A Valid MAE / skill | D Valid MAE / skill | A Test skill | D Test skill |
| --- | --- | --- | --- | --- |
| S（来源结构） | 0.01060 / **−0.008** | 0.01349 / **+0.001** | −0.007 | 0.000 |
| C（环境） | 0.01057 / −0.005 | 0.01348 / +0.002 | −0.005 | −0.000 |
| SC（并集） | 0.01056 / −0.004 | 0.01347 / +0.002 | −0.005 | 0.000 |

SC 相对更强单组的相对 MAE 改善 0.05%（A）/ 0.07%（D），远低于协议 10% 参考；因此"简单结构+环境特征"无法解释模型实际计算出的消息作用。

候选规律（≤5 条，只在 Train 选，Valid/Test 只做评估）：A 的 5 条候选在 Valid 覆盖率 0.000–0.001、同号率 0.00/0.20/0.50/0.75/0.80；D 的 5 条覆盖率 0.000–0.002、同号率 0.50–0.92。没有一条同时满足"支持充分 + 方向稳定"。

真实分子案例（每个臂 4 例，含完整向量）：

- **成功案例**（A：Test 分子 102；D：Valid 分子 10）：候选叶子内 8 条消息的作用中位数与 Train 中位数同号，且该分子整体预测方向一致；D 的这些消息 `A_extra` 占总作用的 32%。
- **环境改变作用**（A：Valid 分子 24，D：Valid 分子 98）：**完全相同**的来源结构记录（degree 2、成环、最小环 4、全环键）与相同键类型，在接收端关系构成不同时作用反号（A：+0.0833 vs −0.1559；D：+0.0935 vs −0.1842）。这是"环境调制"最直接的可追踪证据，但它不可被简单规则覆盖（见上表 skill）。
- **反例**（A：Valid 分子 212，D：Valid 分子 146）：Valid 上 |A| 最大的单条消息并不落在任何 Train 候选叶子内（A：−0.1738，SINGLE；D：+0.3951，SINGLE，其中 `A_extra` 占 19%）。
- **最大作用消息**：与反例同一分子，附 `p_ji`/`γ_i`/`δ_{i,r}`/`v_ji`/`v_extra_ji` 的完整 128 维向量（唯一保存的完整向量，其余只存标量与摘要）。

## 7. 与历史参照的描述性对照（非匹配对照，不可作为结论）

同一仓库、不同配方（无 seed 配对、无同参数量、无同 horizon）的历史数字，仅用于定位量级：

| 模型 | Lipo Valid RMSE | Lipo Test RMSE | 来源 |
| --- | --- | --- | --- |
| GINE（method_G） | 0.6326 | 0.6695 | `tracks/structural-query-learning/results/sq-pilot-small-v1/ogbg-mollipo/` |
| C1（method_L，LCRC 读出） | 0.7132 | 0.7312 | 同上 |
| 本轮 A FIXED（FIXED1） | 0.7267 | 0.7166 | 本报告 |
| 本轮 D CSC | 0.7389 | 0.7300 | 本报告 |

本轮四臂都明显弱于历史 GINE（约 −0.05～0.10 RMSE），与历史 C1 同量级。因为配方/宽度/预算不同，这些数字**不能**用来判断"条件化结构计算"与 GINE 的优劣。

## 8. 预算与实际执行

| 单元 | GPU | 实际 GPU·h | 每 epoch | 结束（UTC） |
| --- | --- | --- | --- | --- |
| Lipo A | gpu1 | 0.167 | 1.6 s | 04:32 |
| Lipo C | gpu1 | 0.173 | 1.7 s | 04:42 |
| ESOL A | gpu1 | 0.040 | 0.4 s | 04:45 |
| Lipo D | gpu0 | 0.331 | 3.3 s | 04:42 |
| ESOL C | gpu0 | 0.113 | 1.1 s | 04:49 |
| ESOL D | gpu0 | 0.116 | 1.1 s | 04:56 |
| Lipo B | gpu1 | 0.167 | 1.7 s | 04:57 |
| **合计** | | **1.107** | | |

预检（`analysis/preflight_timing.json`）预测 7 单元约 0.305 GPU·h（未含更慢的 D 单元与 gpu0 上的竞态），实际 1.107，仍在 2 h 目标内。峰值显存 70–175 MB，无 OOM 风险。推理吞吐（含全部 eval）约 4.8k–8.7k 分子/秒，参数量差异不构成吞吐量结论。

事件：gpu0 队列的墙钟自保护在剩余训练预算 <900 s 时跳过了 Lipo B 并继续执行 ESOL（`NOT_STARTED` 记录在日志中）；随即在 gpu1 上单独提交 `csc-b-lipo` 完成 B，因此**最终结果集是 7/7 完整单元**，协议优先级（B 先于 ESOL）在结果层面得到满足，没有单元缺失或降级。

## 9. 工程日志（只修实现，未改公式/对照/评价口径/阈值）

1. 原始边记录（含两个方向）与缓存物理键的一一对应、`bond_type = edge_class+1`（UNK→5）的映射校正，并把端点校验写进检查 1。
2. `log1p(n_i)` 拼接缺 `unsqueeze(1)`；`np.abs` 与 numpy 标量混用；`n_i`/`n_r` 索引错位。
3. `gate_statistics` 曾用 q 维 `fc2` 权重投影 d 维 `dh`（形状不匹配），改为只投影 `Q`。
4. 逐位比较前 `forward_states` 未强制 `eval()`（dropout 会破坏位级一致性），已修；检查 10 改为比较 `Q`/`h2`（分组只进入 `Q`，`v` 与分组无关）。
5. 依赖检查前对四臂统一 `randomize_new_modules`，避免"零初始化导致检测不到依赖"的假通过。
6. `cs_rules.py`：`mol_row` 需经 `msg_mol` 间接索引；上下文特征用接收端度；`gamma_rms` 是逐节点量（按接收节点索引）；`readable_condition` 在回溯时未清理失败分支的路径，导致不同叶子输出相同条件（已改为返回路径、并断言叶子可达）；单分子 batch 的原子索引是批内局部，取完整向量时需加 split 内偏移；新增 `cs_manifest.py` 生成结果清单。
7. `cs_decompose.py`：新增"从原始 FIXED1 状态到 CSC 状态"的新增修正路径积分（原先从 `h2_base` 出发），并新增两项状态重建误差与 `A_extra` 闭包的显式记录。
8. `cs_queue.py` 的队列汇总文件名用 `device.type + str(device)` 拼接，得到 `queue_cudacuda_0.json`，且两个 GPU 队列（同为 `cuda:0`）会互相覆盖——只有最后一个作业的文件留存。改为 `queue_{type}{index}_{utc}.json`，并把留存的旧文件重命名为 `queue_cuda0_last_job.json`；每个单元的状态以 `runs/*/run.json` 与 `results/manifest.json` 为准（两者都完整）。
9. 结果同步事故：把较早的 pull 快照 rsync 回工作区，覆盖了两个单元已完成的 `train_curve.csv` / `predictions_best.npz`（`molesol_D` 内部 epoch 13、`mollipo_B` 内部 epoch 59）。由 `run.json` 里的 `curve_sha256` 与文件实际 SHA 不一致发现；改为直接从 `res` 主机权威同步后，**21 个文件的 SHA 与 `run.json` 记录全部一致**（`results/manifest.json`），`mollipo_D`、`molesol_D` 的 `best_epoch`/`predictions_best` 内部 epoch 均与 `run.json` 对齐。教训与修复方式已写进本轮流程说明。

## 10. 交付物与复现

- `PROTOCOL.md`（冻结）、`configs/experiment.json`（冻结件）
- `code/`：`cs_common.py`、`cs_data.py`、`cs_model.py`、`cs_train.py`、`cs_checks.py`、`cs_queue.py`、`cs_freeze.py`、`cs_analyze.py`、`cs_mechanism.py`、`cs_decompose.py`、`cs_rules.py`、`cs_manifest.py`
- `analysis/`：`checks.json`、`preflight_timing.json`、`main_table.csv`、`paired_comparisons.json`、`strata.json`、`verdicts.json`（含性能轴、`interpretability_axis` 三维与联合归属）、`mechanism_diagnostics.json`、`decomposition_trace.json`、`structure_rules.json`、`explanation_cases.json`、`queue_cuda0_last_job.json`
- `runs/`：7 个单元的 `run.json`、`train_curve.csv`、`best_checkpoint_eval.json`、`gate_history.json`、`training_order.json`、`predictions_{best,final,last20}.npz`，以及 `checkpoint_{best,final}.pt`（同步回本地，`results/manifest.json` 记录路径与 SHA-256）
- `results/decomposition/*.npz`（Lipo A/D × train/valid/test）、`results/manifest.json`
- 复现命令（远端 `res`，GPU 队列）：
  - `uv run python code/cs_checks.py --dataset ogbg-mollipo --dataset ogbg-molesol`
  - `uv run python code/cs_queue.py --units "ogbg-mollipo:A,ogbg-mollipo:C,ogbg-molesol:A" --device cuda:0`
  - `uv run python code/cs_mechanism.py`
  - `uv run python code/cs_decompose.py --datasets ogbg-mollipo --arms A,D`
  - `uv run python code/cs_rules.py --datasets ogbg-mollipo --arms A,D`
  - `uv run python code/cs_analyze.py`（`verdicts.json` 会读取上面三份产物以填 `interpretability_axis`，因此放在最后）
  - `uv run python code/cs_manifest.py`

  产物均落在 `analysis/` 与 `results/`；`analysis/structure_rules.json` 约 4.4 MB（含逐消息冗余的安全校验字段）。

## 11. 本轮结论与下一步（不新增训练）

1. **在 Lipo/ESOL、1× 宽度、单 seed、360 epochs 的冻结配方下，"结构内容与环境调制分离 + 保留真实关系对应性"没有带来图级预测收益**；相反，A（FIXED1，Q=0）在两个数据集上都是最好的或并列最好的。
2. **可解释性上，"可追踪"成立，"可归纳"不成立**：完整计算可精确分解到每条真实消息并闭合到预测变化，但由此得到的简单结构规律在未参与拟合的 Valid/Test 上没有迁移能力，重复结构签名的作用符号本身就不稳定。
3. 机制诊断给出了与性能一致的因果性读数：把条件修正去掉或冻结为按关系类型的常数会让模型变好——本轮学到的逐节点环境调制是纯噪声。若要继续该方向，必须先解决"逐节点调制带来的方差"问题（例如：只在异质邻居上开启调制、对 `δ` 加结构化的低秩/分关系约束、或用 I2 那种常数化调制作为归纳偏置），而不是继续加容量；本轮不新增训练/seed/数据集/宽度。
