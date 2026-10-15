# BJOC 冻结模型来源级结构归因可行性 v1 — PROTOCOL（冻结）

* 命名空间：`bjoc_source_attribution_feasibility_v1`
* 轨道：`tracks/structure-contribution`
* 前置审计：`results/bjoc_structure_explanation_audit_v1/`（commit `410aea6`）。本文件在任何新数值产生之前写入。
* 状态：冻结协议。CPU-only、label-blind、no-grad 审计（唯一的梯度使用是归因方法本身的输入梯度，不更新权重、不训练）。

## 1. 问题与非目标

**问题**：在完全冻结 BJOC-Null（R3 full-10k, seed 0/1）的情况下，能否从真实 Junction 编码来源中识别
稳定、具有可验证预测作用的局部结构？

回答四个子问题：

1. `w_j` 的来源级归因在数学与实现上是否可信（可加性/完整性、排列等变性、对称性、与精确遮蔽排序一致）？
2. 高排名 Junction（及其相邻组合）是否在**独立于排序计算**的干预下显示超出随机对照的预测响应？
3. 两个 seed、不同参照编码、不同干预强度下解释是否稳定？
4. 该方法能解释模型的哪一部分？与 Atom / Bond / Global 通路的未解释部分如何划分？

**非目标**：不设计/不训练新预测器；不报告监督 MAE/RMSE；不做真实分子编辑；不声称化学因果；
不做 Motif 搜索/可学习分割；不读取 official valid/test 或 Monitor/Locked 标签。

## 2. 冻结模型与来源复现

* 模型：`_make_downstream_model(seed, prep_train10k.json)` + `load_state_dict(soup_R3_seed{seed}.pt, strict=True)`；
  `eval()`；权重冻结；参数必须为 413,505；reader_in 必须为 321 维。
* checkpoint/缓存 SHA-256 与记录不一致即停止（阻塞），不允许替换模型。
* 原始输出复现门（先于任何归因）：
  * `forward(batch)` 与 `reader(forward_features(batch)["reader_in"])` 最大差 ≤ 1e-6；
  * 原子重标号预测差 ≤ 1e-5；单分子 vs 批 ≤ 1e-5；
  * 对固定 smoke 分子批，两 seed 的预测必须与前一审计
    `results/bjoc_structure_explanation_audit_v1/phase_b_baseline.json` 中记录的同一协议结果相容
    （本审计重新计算并记录 max 差）。

## 3. 数据与样本（预先固定，label-blind）

* 只使用官方 train 结构缓存 `objects_train10k.pt`（无标签字段）。
* 复用 `bjoc_structure_explanation_audit_v1/sample_manifest.json` 的哈希规则与行索引：
  诊断集 rank 0–255（smoke = rank 0–15），参照集 rank 256–1279（1024 个，用于匹配替换）。
* 报告与历史结果对比时明确：样本/模型/分组（Null）与前审计一致。

## 4. 归因方法（排序计算）

**干预单位**：编码后的 Junction 表示 `w_j`（真实 junction `(i,j,k)` 的 96 维编码），
在模型编码之后、Null 分组之前。

**积分梯度（Integrated Gradients, IG）**：对分子 `g` 的预测 `pred = Reader(Null(w))`，
第 `j` 个 junction 的输入为 `w_j`，参照（baseline）编码为 `b ∈ R^96`：

```
IG_j(b) = (w_j − b) ⊙ ∫_0^1 (∂pred/∂w_j)(b + t(w_j − b)) dt
```

* **主参照（primary）**：`b = μ_G`，即该分子所有 junction 编码的逐维均值（分布内、“中性图级”语义）。
* **次参照（stability）**：`b = 0`（前一审计的置零语义；逐维 OOD 明显，仅作稳定性对照）。
* 数值实现：K = 32 个中点步；在每一步对批量化的 `(∂pred/∂μ, ∂pred/∂ν)` 做一次 forward+backward；
  `μ,ν` 是 `mean(w)`、`mean(w²)`（Null 分组实际使用的两个统计量），因此该实现是对模型真实计算路径
  的精确路径积分（不是对代理模型的近似）。
* **符号约定**：`IG_j` 为 signed 标量；排序使用 `|IG_j|`。所有比较同时报告 signed 版本。
* **完整性（completeness）诊断**：`Σ_j IG_j` 与 `pred(w) − pred(所有 junction 均为 b)` 的差
  （相对量）必须报告；这是实现正确性的核心校验之一。

**参照语义与局限（必须随结果引用）**：
* `μ_G` 参照表示“把该 junction 换成该分子平均 junction 内容”；它不是删除原子/键，不对应任何化学操作。
* 零参照逐维 OOD（前一审计：`sum_atom_squared` 置零 z≈2.3；`w` 的零参照未单独测量），只用于稳定性。
* Null 分组下 `w_j` 经 `mean_G(w)` 影响**所有** Bond，因此 `IG_j` 不是“该 junction 对某条键的局部贡献”。
* IG 对梯度路径的弯曲敏感；Hessian/交互项会进入 `Σ_j IG_j` 与“全体同时移动”路径的差。

## 5. 验证干预（不同于排序计算，预先固定）

**匹配自然编码替换（validation）**：从参照集取与诊断分子 `(n_atoms, n_bonds, n_junctions)` L1 最近邻
（并列取行号小者）的分子；第 `j` 个诊断 junction 的替换向量取参照分子**第 `j mod n_ref_junc` 个**
junction 的编码 `w^ref_j`（位置对应，不声称化学对应）。干预强度 `α ∈ {0.5, 1.0}`：

```
w'_j = w_j + α (w^ref_j − w_j)
```

随后**完整重算** Null 分组（`W,V,n_b`）、Composer、`reader_in` 与 Reader。
响应 `Δpred_j = pred(w') − pred(w)`。该干预与 IG 的排序计算完全独立。

**精确遮蔽对照（已知真值，用于实现校验）**：`w_j ← b`（主参照），完整重算，得到 `occ_j`。
在 smoke 集上比较 `|IG_j|` 与 `|occ_j|` 的排序（Spearman）与 top-k 重合。

## 6. 预注册比较与统计

Junction 邻接定义（用于“相邻组合”）：两个 junction 相邻当且仅当它们的 `junc_bond` 集合相交
（共享至少一条物理键）。`k = min(4, n_junctions)`。

| 比较 | 统计量 | 预注册判据（全部 seed、全部 α） |
|---|---|---|
| top-k vs 同分子随机 junction | 每分子 `mean_{j∈topk} \|Δpred_j\|` 与从补集随机抽 k 个的均值（64 次抽样平均）配对差 | `mean\|Δ\|` 的 median 差 > 0 且符号检验 p < 0.01（两侧） |
| top-k 相邻性 vs 随机 k-子集 | top-k 内部邻接边数 vs 随机 k-子集（64 次） | top-k 边数 > 随机 95 分位；两 seed 一致 |
| 跨 seed | 池化 per-junction `IG` 的 Spearman；每分子 top-k Jaccard vs 随机 k-子集 Jaccard 分布 | Spearman ≥ 0.5；Jaccard > 随机 95 分位 |
| 参照稳定性 | 主参照 vs 零参照的池化 Spearman；top-k Jaccard | Spearman ≥ 0.7 |
| 强度稳定性 | α=0.5 vs α=1.0 的每-junction 响应排序相关 | Spearman ≥ 0.7 |
| 完整性 | `Σ_j IG_j` vs `pred(nat) − pred(all-b)` | 相对差 ≤ 0.10（smoke 集） |

**通道覆盖**（回答“方法能解释哪一部分”）：
* Junction 通路的**总**效应：`W=V=0` 反事实（复现前审计）的 `mean|Δpred|`；
* Junction 通路的**组内可归因**部分：`Σ_j IG_j`（主参照）的分布，与总效应比较，划分为
  “组间均值水平”（不可归因到单个 junction）+“组内偏差”（可归因）；
* 其它通路：四块置零（复现）与 `z_bond` 的匹配替换（复现）作为未解释部分参照。

**禁止**：用同一种干预的响应大小来选择结构、再用该响应大小证明解释成功。排序只用 IG；
验证干预与判据在上表冻结；不得在看到结果后改判据。

## 7. 已知真值/合成校验（实现正确性）

在 smoke 集（真实结构、冻结模型）与构造的合成对象上执行：

1. **排列等变性**：置换 junction 行（`w` 与 `junc_*` 同步置换），IG 分数按同一置换对应；max 差 ≤ 1e-5。
2. **对称性**：两个 `w` 行完全相同的 junction 必须得到相同的 IG（≤1e-6）。
3. **IG vs 精确遮蔽**：排序 Spearman ≥ 0.8、top-k Jaccard ≥ 0.5。
4. **完整性**：见 §6。
5. **合成已知真值（可选加强）**：构造含“一个特殊 junction”的小图（特殊 junction 的编码与其他不同），
   检验 top-1 命中特殊 junction；以及 `Σ_j IG_j` 与真值差的一致性。

任一校验失败 → 停止归因阶段，只报告失败与阻塞，不增加搜索轮次。

## 8. 停止规则

* 来源哈希/参数/复现门失败 → 停止。
* §7 任一实现校验失败 → 停止 attribution，报告阻塞。
* §6 判据未达成 → 对应结论记为 `INCONCLUSIVE` 或 `NOT_SUPPORTED`，不追加抽样或换判据。
* 结果不稳定或参照分布外问题严重 → 结论限制为方法可行性，不上升为结构解释。

## 9. 交付物

* `PROTOCOL.md`（本文件）
* `ERRATUM`（任务一）：`results/bjoc_structure_explanation_audit_v1/ERRATUM.md` +
  独立重算 `results/bjoc_source_attribution_feasibility_v1/erratum_recalcs.json`（不覆盖历史）
* `source_manifest.json`、`sample_manifest.json`
* `synthetic_checks.json`（§7）
* `attribution.json` + `attribution_arrays.npz`（§6 统计与 per-junction 分数）
* `metrics.json`（汇总 + 决策标签）
* `REPORT.md`（中文；按证据强度组织；回答 4 个问题 + 下一步建议）
* 代码 `code/bjoc_source_attribution_feasibility_v1.py`、测试
  `tests/test_bjoc_source_attribution_feasibility_v1.py`

## 10. 证据分级

1. **公式/实现必然**：IG 路径积分定义、Nil 分组线性统计结构、排列等变性、完整性恒等式。
2. **冻结模型数值干预**：IG 排名、匹配替换响应、跨 seed/参照/强度稳定性。
3. **无证据/未决**：化学因果、真实分子编辑语义、未重训练下的忠实子结构解释。

## 11. 修订记录（amendments to the frozen protocol）

**R1 — 完整性恒等式的参照修正（记录于任何主体归因结果产生之前）**

发现时机：实现校验阶段（smoke 集），尚未运行 256 分子主体归因。

问题：§6 表格与 §7 第 4 条把完整性校验写成
`Σ_j IG_j ≈ pred(natural) − pred(所有 junction 均为参照 b)`。该参照对**逐 junction 路径积分**
不成立：本方法的 `IG_j` 沿“仅第 j 个 junction 从 b 移动到 w_j、其余落在 natural”的路径积分，
由微积分基本定理，精确等于
`IG_j = pred(natural) − pred(w_j := b) = −occ_j`，
而与“联合把全部 junction 重置为 b”的预测差是两个不同的量（后者包含均值/二阶矩的不可加交互项）。
把两者相减得到的 ≈100% 的相对差不是实现错误，而是“组间均值水平重置”与“逐 junction 可归因偏差”
之间的不可加性。

修正：
1. 实现正确性以**逐 junction 恒等式**校验：`IG_j ≈ −occ_j`（点wise 相对误差与 Spearman/top-k 排序），
   预注册阈值：`max_j |IG_j + occ_j| / mean_j|occ_j| ≤ 0.05` 且逐分子 `Spearman(IG, −occ) ≥ 0.8`、
   top-k Jaccard ≥ 0.5（smoke 集）。
2. 原“完整性”量保留为**描述性**统计，更名为
   `joint_reset_nonadditivity_gap = Σ_j IG_j − (pred(natural) − pred(所有 junction = b))`，
   用于量化“联合重置效应”与“逐 junction 效应之和”的不可加部分；不再作为 pass/fail 门槛。
3. 其余预注册内容（样本、参照、干预、top-vs-random、跨 seed/参照/强度、通道覆盖）不变。

R1 在主体结果之前生效；驱动 `code/bjoc_source_attribution_feasibility_v1.py` 中
`COMPLETENESS_REL_THRESHOLD` 仅为记录保留，不再使用。

**R2 — 补充分析 S1：组合层比较（主体结果之后追加；探索性，不改主决策）**

任务清单要求的“排名靠前的**相邻 Junction 组合** vs 同规模/拓扑的随机连通组合”未在 §6 预先固定。
现以完全相同的方法与干预补做，仅改变聚合单位，并明确标注为**探索性**：

* pair 排序分 = `(|IG_i| + |IG_j|)/2`；top pair = 同分子相邻 pair 中排序分最高者
  （相邻 = `junc_bond` 集合相交）；
* joint 响应 `Δ_joint`：同时替换两个 Junction 的编码（匹配自然编码，α=1.0），完整重算 Null；
* 对照：同分子随机相邻 pair（64 draws；RNG `11000000 + seed*100000 + row`）与随机非相邻 pair（64 draws）；
* 报告：配对符号检验、`Δ_joint − (Δ_i + Δ_j)` 交互项、|Δ| 均值。
* S1 若为正，仅作为下一轮预注册实验的动机；**不回溯性修改 §6 判据、R1 校验或 decisions**。
