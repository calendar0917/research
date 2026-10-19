# SSRC Phase 1e protocol — ZINC 标签机制核验与性能路线决策

Protocol ID: `ssrc-phase1e-zinc-v1`
Date: 2026-10-20
Status: **FROZEN_PRE_GPU**（Phase A 为纯 CPU 描述性审计；Phase B 的 GPU 采购
规则与 Phase A4 的判定规则在写任何 Phase 1e 结果文件之前固定）
Upstream: `results/phase1d/REPORT.md`, `notes/phase1d_protocol.md`,
`notes/prior_work_boundary.md`, `results/phase1b/REPORT.md`,
`tracks/structure-contribution/results/explicit_composition_signal_audit_v1/`,
`tracks/ksvd/results/zinc_long_cycle_audit/`
Frozen code: `code/phase1e/`（只读已有 Phase 1b/1c/1d 产物；不改写、不重训任何
Phase 1b–1d 轨迹）

> **这是对已被多轮使用的内部 Monitor 的事后审计，不是独立确认集。**
> 本轮的任何一个数都不能被称为 "held-out confirmation"。
> 官方 ZINC valid/test 与内部 Reserved 仍然不读。
>
> 本轮的成功标准是**终止无价值的局部模型迭代**，不是再拿到一个小数点。

---

## 0. 已知事实（本轮不得推翻，只能检验）

1. `R_incidence`（加性 node–incidence）在 96% 无大环分子上优于非加性
   `R1_size`（−0.0281 MAE，CI −0.0460…−0.0117）。
2. 非加性 `R1_size` 在 4%（40 个）大环分子上明显更好（+1.751 MAE）。
3. `R1_size_cap` Monitor MAE 0.1935；同协议 `G_ref` 0.1653。参数 453K
   （1.031× `G_ref`），前向 FLOPs 2.10×，峰值显存 1.60×，实际 step 时间反而更低。
4. 旧 BJOC-P3 在已暴露的官方 test 上为 0.069822（seed-mean，两 seed），
   该数字**不能**与内部 Monitor 直接比较。
5. 旧轨 `explicit_composition_signal_audit_v1` 已证明 penalized-logP 公式主导
   （strict 11 对：公式 R²=0.999897，genuine beyond-noise = 0），并报告
   环基定义不一致。
6. 当前 Monitor 已被 Phase 1b/1c/1d 多轮使用。本轮任何"子群效应"都只能是
   描述性的、后验的。

## 1. 目标标量的确切来源（A1 的出发点）

ZINC 目标由 Kusner et al. 2017 (GVAE)
`molecule_optimization/latent_features_and_targets_grammar/generate_latent_features_and_targets.py`
生成，经 graphdeeplearning/benchmarking-gnns 的 `molecules.zip` 打包，PyG
`ZINC.process()` 原样读取 `logP_SA_cycle_normalized`：

```python
cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(mol)))
cycle_length = max(len(c) for c in cycle_list) if cycle_list else 0
cycle_length = 0 if cycle_length <= 6 else cycle_length - 6
cycle_scores.append(-cycle_length)
y = (logP-μ)/σ + (SA-μ)/σ + (cycle_scores-μ)/σ     # μ,σ over the 249,456 pool
```

关键点，必须被如实描述：

- cycle 项用的是 **NetworkX `cycle_basis`（Paton 基本环基）**，不是 RDKit
  **`GetSymmSSSR`**，也不是 `minimum_cycle_basis`，也不是最长简单环。
- `cycle_basis` 的结果依赖节点迭代顺序；顺序来自
  `MolToSmiles(MolFromSmiles(line))` 的 RDKit 规范 SMILES 原子序。
  因此 cycle 项是**图 + 规范化原子序**的函数，不是图的置换不变函数。
- 归一化常数存在 dataset 与 community 两套（相对差 ≤1.5%）。本轮以冻结的
  `zinc_long_cycle_audit` 逐分子标签分解为准（`label_effective_cycle_snapped`
  / `label_cycle_component` / `y_without_cycle_label`），并核对其与
  `train_cycle_audit_label.csv` 的 SHA-256。

## 2. Phase A 判定规则（在任何 Phase 1e 结果文件写出前固定）

符号约定：Δ > 0 表示候选模型更差。

记号：

- `P` = Monitor 中 **标签 cycle 惩罚非零** 的分子集合（冻结 KSVD 分解给出）。
- `R` = Monitor 中 **RDKit `GetSymmSSSR` 最大环 > 6** 的分子集合（Phase 1d
  的 `large_cycles_gt6 ≥ 1`；n=40）。
- `P̄ = Monitor \ P`（无标签 cycle 惩罚的分子）。
- 对模型 m：`MAE_all(m)`, `MAE_P(m)`, `MAE_Pbar(m)`。
- 全部区间为 molecule-level paired bootstrap（5000 次，95% percentile）。

### A1 `LABEL_MECHANISM` 读数（不单独触发方向，但逐条报告）

- `A1a` 混淆矩阵 `label_penalty × rdkit_large_ring`（Monitor n=1000，并给出
  10k official train 作对照）。
- `A1b` 每个 `P` 分子的置换稳定性：对 PyG 结构做 30 次随机节点重标号后重算
  `nx.cycle_basis` 的 excess；记录 span 与是否跨越惩罚等级。
- `A1c` 标签重构误差：`|y_recon - y_stored|`（KSVD `formula_verification`），
  对 `P` 与 `P̄` 分别报告。
- `A1d` 用 RDKit SSSR 作为路由代理时漏掉多少个 `P` 分子，以及这些漏掉分子
  占全部绝对误差的比例。

### A2 `CYCLE_ATTRIBUTION`（描述性，不选模型）

- 对每个冻结模型（G_ref、R1_size s0/s1、R1_size_cap、R_incidence、S_cap_size、
  R1_marginal_size、M_keep_edge）报告：`MAE_all`、`MAE_P`、`MAE_Pbar`、
  带符号残差、p90、以及 `P` 对总绝对误差的贡献份额。
- 对 `P` 报告 counterfactual 量 `MAE(|pred - y_without_cycle|)`。该量只用来
  判断"模型是否已经在预测惩罚项"，**绝不**作为模型成绩，也绝不作为新模型输入。
- 报告 `P` 组内 cycle offset 捕获率 `mean(pred - y_without_cycle) / mean(c)`。

### A3 零训练条件路由（事后诊断，不是新模型成绩）

路由规则固定为：

```
if rdkit_large_cycles_gt6 >= 1: R1_size_cap   else: R_incidence
```

报告：实际逐行 MAE、与 `G_ref` 的差、与单一 `R1_size_cap` / `R_incidence` 的差、
以及条件执行的理论 FLOPs 下界（按分子比例加权的前向 FLOPs；明确说明真实
batched 执行下不能低于各子模型的最大值，因此这是下界而非可达成本）。

同时给出**仅用于量化代理损失**的 oracle 路由（用标签惩罚集合 `P` 而非 RDKit
分组）作为对照，并说明它不可部署。

### A4 结论分类（三选一，允许同时触发但必须标注主因）

- `BENCHMARK_CYCLE_ARTIFACT_DOMINATES` — **要求同时满足**：
  1. `|R \ P| / |R| ≤ 0.05`（大环分组不被未惩罚分子稀释）；
  2. `MAE_Pbar(R1_size_cap) - MAE_Pbar(G_ref) < 0.02`（Phase 1d 的
     negligible band）；即去掉标签惩罚集合后，候选已进入 `G_ref` 的
     negligible band；
  3. `P` 对至少一个候选模型的绝对误差质量贡献 ≥ 25%。
- `STRUCTURAL_SIGNAL_BEYOND_CYCLE_REMAINS` — **任一满足**：
  - (a) 在 `P̄` 内，某候选比 `G_ref` 好 ≥ 0.02 且 CI 上界 < 0；
  - (b) 在 `P` 内，某候选比 `G_ref` 好 ≥ 0.02 且 CI 上界 < 0。
  （`R1_size_cap` 的 `P` 组 CI 已知很宽；若只有单 seed 且 CI 跨零，本规则不触发。）
- `UNRESOLVED_LABEL_MECHANISM` — **任一满足**：
  - (a) `|P|` 中 ≥5% 的分子其 cycle 惩罚在随机节点序下跨越惩罚等级
    （span ≥ 1 个 label unit）；
  - (b) `|P|` 中 ≥5% 的分子标签重构误差 > 1.0 y-unit；
  - (c) RDKit SSSR 代理漏掉 ≥5% 的 `P`，且这些分子携带 ≥5% 的总绝对误差。

不允许"弱相关"单独触发任何一个结论。若三条都不触发，主结论记为
`INSUFFICIENT_EVIDENCE`。

## 3. Phase B 规则（性能竞争力审计）

### B1 统一参照（不做跨协议排名）

三类数字**分开列表，永不合并排序**：

1. 内部 Fit 8,000 / Monitor 1,000 协议：`R1_size_cap` 0.1935、`G_ref` 0.1653
   （以及本轮 B2 若执行得到的强基线）。
2. 旧 BJOC-P3 的已暴露官方 test 0.069822（seed-mean of 2 seeds）。
3. 文献官方 ZINC-12k：ESA+ 0.052±0.001、GRIT 0.059±0.002、CSA-rings
   0.056±0.002、MoSE+GRIT 0.056±0.001、SPSE+GRIT 0.059±0.001、
   GSM/GPM 0.064±0.004、GraphGPS ≈0.070、GSN ≈0.101。

**污染检查（必须执行）**：对每个历史强模型 checkpoint，计算其**监督训练行集合**
与 SSRC Monitor 1,000 的重叠。重叠 > 0 的 checkpoint 不得用于内部 Monitor 评价。

### B2 条件性单 GPU 基线（默认不执行）

只有同时满足以下全部条件才执行：

1. 存在已冻结、已双 seed 复现、单次 ≤1 GPU-h 的强模型配方；
2. 能在 SSRC Fit 8,000 上**严格隔离**重建（Fit-only 预处理、Fit-only SSL、
   Fit-only 监督），Monitor 1,000 只在预测哈希冻结后读一次；
3. 不改变预训练数据权限（只用官方 train 的行，且只用 Fit 行）；
4. 单 seed、单配置、不调参、不早停选择。

预算上限 **2.0 A100 GPU-hours**。若预估超预算或不能保持协议一致，**不执行**，
只报告所需资源。

本轮**不**新增 R1 宽度/深度/Attention/ring-aware token/SSL 实验。

### B3 `R1_IS_NOT_COMPETITIVE` 规则

若强基线（B2 产出）的 Monitor MAE 比所有 R1 臂中最好的低 ≥0.02，且
molecule-bootstrap 95% CI 不重叠，则记 `R1_IS_NOT_COMPETITIVE`。
若 B2 未执行，本条记为 `NOT_EVALUATED`，不得用跨协议数字替代。

## 4. 最终路线选择规则（唯一优先方向）

输入：`A4` 的三个标签、`B3` 的标签。

| 条件 | 方向 | 理由 |
|---|---|---|
| `artifact_dominates` 且 `beyond_cycle` 不成立 | **A** | ZINC 的大环机制主要是人工 cycle penalty；停止 ZINC 专属 ring-aware R1 优化，把证据迁到无同类目标捷径的任务 |
| `R1_IS_NOT_COMPETITIVE` 且 `beyond_cycle` 成立 | **B** | 关系信息仍有价值，但当前 backbone 无竞争力；改为在强 backbone 上检验显式结构/关系计算 |
| `beyond_cycle` 成立且 `R1_IS_NOT_COMPETITIVE` 不成立 | **C** | 保留一次性关系方法并继续投入 |
| `artifact_dominates` 与 `beyond_cycle` 都不成立，且 `R1_IS_NOT_COMPETITIVE` | **D** | 暂停 R1 架构分支（不外推为 SSRC 整体失败） |
| 只有 `UNRESOLVED_LABEL_MECHANISM` | **D** | 无可辩护的机制主张 |

方向 A 与 B 若同时被支持，**优先 A**，并在 A 的任务设计里继承 B 的约束
（下一轮的对照 backbone 必须是有竞争力的强基线，而不是 `G_ref`）。

## 5. Phase C 规则（下一任务选择）

只选**一个**任务，且必须满足：

1. 目标不是人工评分规则的合成物（无 cycle-penalty 类捷径）；
2. 仓库已有可复现的数据/划分基础设施（不新下载多数据集）；
3. 有成熟文献基线可供**协议内**比较；
4. 能区分三个假设：H1 局部结构统计足够 / H2 一次性关系组合有增益 /
   H3 需要更长程或迭代结构计算。

本轮只产出设计与预注册草案，不下载、不训练。

## 6. 明确的非目标

- 不重开 R1 架构搜索（宽度、深度、attention、ring-aware token、SSL）。
- 不把 cycle penalty 真值当模型输入，也不做标签补偿后冒充图学习性能。
- 不用官方 valid/test 或 Reserved 做任何选择。
- 不合并跨协议数字做排名。
- 不为"延续研究路线"而重新解释 Phase 1c/1d 的既有限定。

## 7. 纪律声明（事后审计的性质）

本文件写于 Phase 1d 结论之后、Phase 1e 结果文件生成之前。Phase A 的审计对象
是已经被 Phase 1b/1c/1d 使用过的 Monitor，因此：

- 本轮所有 Phase A 数字都是 **post-hoc / descriptive**；
- 本轮任何"大环子群效应"都不是独立确认；
- Phase 1d 已披露：大环分组是在 Phase A 探索中发现的。本轮不把它升级为
  预注册假设；
- 如果 Phase A 与 Phase 1d 的解释冲突，以本轮**新增的标签侧证据**为准，并在
  REPORT 中显式记录冲突。
