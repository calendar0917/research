# SSRC Phase 0 — Structural Sufficiency & Relational Computation

日期: 2026-10-18  
Track: `tracks/structural-sufficiency`  
父 commit: `666083e82a3a2482bd55c925fb3398553be7b756`  
本轮性质: **CPU-only、无训练、无 GPU、无官方 valid/test 标签读取、无 ZINC 排行榜搜索**

> **CORRIGENDUM (2026-10-19):** 本报告保留 Phase 0 原始证据与历史表述，
> 但其中 §2.1 的 Eijkelboom train/test 口径、§4.2 的风险分解符号、
> §7.7–7.8 的 pilot 方向/停止规则、§7.3 的 \(S(G)\) oracle 称呼，
> 以及“94.5% 单例 ⇒ 结构充分性”的解释，均已被
> [`notes/phase0_corrigendum.md`](../notes/phase0_corrigendum.md) 纠正。
> 凡与纠错文件冲突处，以纠错文件为准；本报告的 CPU 数字与证据不删除、
> 不改写。Phase 1a 的设计见
> [`notes/phase1a_protocol.md`](../notes/phase1a_protocol.md)。

---

## 0. 结论摘要

**Decision: `REFRAME_NEEDED`（针对原表述）；条件性 GO（针对重述后的受控问题）。**

- **原表述**“给定显式 \(E_r\)，消息传递/关系计算什么时候必要？”作为一个
  独立的新颖性主张，**不应 GO**：
  1. “强结构编码可以减少/替代 MP”已被 Eijkelboom et al. 2023 直接研究；
     GRIT/CSA/EGT/MoSE/SPSE/GPM 已展示 no-MP 在 ZINC 上竞争或 SOTA。
  2. CPU toy 证明：\(E_2\) 的碰撞是**信息缺失**；但一个**非迭代**的全图
     距离 oracle 也能解决同一任务。因此 toy 不支持“迭代必要”，只支持
     “信息访问决定下界”。
  3. 真实 Lipophilicity 上，精确 \(E_2/E_3\) 几乎每个训练分子一个组
     （94.5% 单例），LOO 只比全局均值好 0.052 MAE；粗 Murcko scaffold
     分组好 0.249 MAE。当前证据更像**统计可学习性/归纳偏置**问题，
     不是“局部信息缺失”。直接比较“静态 \(E_r\)”与“MPNN on \(G\)”会把
     信息差异与估计差异混在一起，得到不可解释的结果。
- **重述后的可检验问题**：
  > 精确结构充分性（\(E_r\) 足以标识分子）为什么不等于统计充分性
  > （能从有限样本泛化）？迭代关系计算的价值是补信息，还是改善
  > 样本效率/归纳偏置？
  
  这个问题有明确的 2×2+oracle 设计、可预注册的主端点和停止规则；
  若用户接受重述，**条件性 GO 到最小 GPU pilot**（§7）。
- **若用户不接受重述**：按 `STOP_THIS_FORMULATION` 处理原表述；
  不要用一个新的 no-MP/MPNN 模型把已经覆盖的主张重新包装。

**一句话证据**：toy 证明了“局部编码看不到远处”这个数学边界；
真实数据进一步显示“精确局部编码太细，反而没有统计共享”。
两者共同把问题从“MP 是否必要”推向“**精确结构充分 ≠ 统计充分**”。

---

## 1. 范围、证据分类与非声明

本报告严格区分五类陈述：

| 类别 | 含义 | 本报告对应 |
|---|---|---|
| **A. 文献已覆盖** | 已有论文直接研究/报告过 | §2；`notes/prior_work_boundary.md` |
| **B. CPU 已证明** | 本轮代码实际跑出的精确结果 | §3–§4；`analysis/phase0_results.json` |
| **C. 仅在该 \(E_r\) 定义下的数学限制** | 对“无序 rooted-patch 多重集”成立；不自动外推到其它编码/真实分子 | §4 |
| **D. 真实分子仍未知** | 只有数据/协议/成本审计，或 train-only 探索性诊断；不是确证 | §5 |
| **E. GPU pilot 要区分的两种解释** | 信息补充 vs 可学习性；需要 2×2+oracle | §6–§7 |

**非声明（必须保留）**：

1. 不声称“强结构编码减少 MP 需求”是新贡献（Eijkelboom 2023）。
2. 不声称“MP 一般必要/不必要”；只研究特定任务、信息访问、预算下的分解。
3. 不声称 \(E_r\) 是新表示；它是 rooted induced-subgraph isomorphism type
   多重集，接近 WL subtree / graphlet / GSN / ECFP 谱系。
4. 不把 toy 的 \(E_2\) 碰撞外推为真实分子上的结论。
5. 不把“训练集指纹唯一”当作“统计充分/可学习/可泛化”。
6. 不把 Dynamic(\(G\)) 访问 `edge_index` 而 Static(\(E_r\)) 只看无序 patch
   多重集的差异，解释为“同信息下动态更强”。
7. 不把 toy 的深度传播 oracle 当作“迭代必要性”证据。
8. 不声称本轮做了 GPU 训练、ZINC 排行榜搜索、新架构设计、外部无标签
   预训练，或读取了官方 valid/test 标签。

---

## 2. 文献已覆盖（A）

### 2.1 最接近的三篇

1. **Eijkelboom et al. 2023**, “Can strong structural encoding reduce the
   importance of Message Passing?”, PMLR v221:278–288.  
   - 直接研究 RW-Concat / RW-Tensor 结构编码下减少/移除 MP。
   - Peptides-struct：no-MP RW-Tensor 0.249 vs full 0.212（差距很小）；
     ZINC：no-MP 0.324–0.466 vs full 0.020–0.076（差距很大）。
   - Table 3 有参数量控制的 no-MP 行（433–57133）与 full MP 行
     （689–162109）；3 seeds；无 information-matched static/dynamic。
   - **结论**：“强编码可降低 MP 重要性”已被直接研究；但该文没有用
     精确 \(E_r\)，没有分离信息与可学习性，没有 Lipophilicity 小样本。

2. **GRIT** (ICML 2023, PMLR v202:23321–23337), “Graph Inductive Biases in
   Transformers without Message Passing”.  
   - RRWP + global self-attention，**无 MP**；ZINC 0.059±0.002，~500K 参数。
   - RRWP→RWSE 消融 0.081±0.010；4 seeds。
   - **结论**：no-MP 可以 SOTA；“no-MP 可行”不是缺口。RRWP 不是精确
     rooted-subgraph 多重集；没有信息/可学习性分解。

3. **MoSE（同态计数）** (ICML 2025, arXiv 2410.18676).  
   - homomorphism counts 作为结构编码；GRIT+MoSE **no MP** 0.056±0.001，
     GRIT+RRWP 0.059±0.002；GPS+MoSE 0.062±0.002。
   - 同一 backbone 替换编码；4 seeds；无 information-matched。
   - **结论**：固定模式计数 + no-MP 已覆盖；\(\,E_r\) 若只是“另一种编码”，
     只报性能提升会重复。

> 另外：CSA (0.070 / 0.056 with rings)、EGT (0.108 at 500K)、
> SPSE (GRIT-SPSE 0.059)、GPM (ZINC 0.064, ZINC-full 0.021) 都是
> no-MP/强结构编码的强先例。完整表见 `notes/prior_work_boundary.md` §1。

### 2.2 已被完全覆盖的主张

- “强结构编码可以减少消息传递的必要性” → Eijkelboom 2023。
- “no-MP 可以在 ZINC 上竞争/SOTA” → GRIT/CSA/MoSE/SPSE/GPM。
- “结构编码（RW / homomorphism / path / ring / pattern）改善 no-MP”
  → 上述多篇。
- “参数控制下 no-MP 仍可竞争” → Eijkelboom Table 3。
- “全局注意力/一次性读出可以替代迭代 MP” → EGT/GRIT/CSA；
  toy 的一次性距离 oracle 也支持。

### 2.3 仍可能的缺口（但不是“表示新”）

- 用**精确带属性 rooted-radius-\(r\) 诱导子图同构类型多重集**做
  **任务相对的碰撞证书**，并把碰撞与 label 冲突/不可约误差联系起来。
- 把 **information value** 与 **estimation/learnability gap** 分开的
  受控实验（\(S(E_r), D(E_r), S(G), D(G)\)）。
- 在小样本 Lipophilicity 上检验“精确结构充分 ≠ 统计充分”。
- 这些缺口单独都可能被视为增量；组合才是本路线的赌注。

**ESA+**：截至 2026-10-18，OpenAlex/arXiv/S2 检索无法唯一识别该名称；
不伪造引用。若用户给出准确标题/作者/venue，需补审。

---

## 3. CPU 已证明（B）：toy 设置与精确性

### 3.1 代码与数据

- 实现：`results/phase0/code/ssrc_toy.py`,
  `results/phase0/code/run_phase0.py`
- 测试：`results/phase0/tests/test_phase0.py` → **17 passed（约 8 s）**
- 运行：`uv run python .../run_phase0.py` → **约 22 s**
- 输出：`results/phase0/analysis/phase0_results.json`
- 三个独立结构家族：
  - `cycle_graph`: C24 环；
  - `caterpillar_graph`: 40 spine + leaves；
  - `grid_graph`: 20×9 网格。
- 每个家族：A/B 距离 6 vs 9；节点类型 C/A/B；边类型 ring/spine/leaf/grid。
- \(E_r\)：每个根的精确诱导 radius-\(r\) 子图，排序后的同构类型多重集；
  保留节点/边属性、根标记、到根距离；不保留根身份/patch 间邻接。

### 3.2 精确性交叉验证

| 检查 | 规模 | 结果 |
|---|---|---|
| pynauty canonical key vs 全排列 canonical vs networkx rooted isomorphism | C24 r=0..3：每 r 852 对（276 图内 + 576 跨 d6/d9） | 0 mismatch |
| 大 patch 抽样 | caterpillar r3, grid r2/r3：各 200 对 | 0 mismatch |
| 节点重标号不变性 | 6 个图 × 50 次随机重标号 × r=0..4 | 0 failure |
| 边顺序不变性 | 三个家族 d6，r=2/3 | true |
| 属性进入 key | 改节点类型/边类型 | key 改变，networkx 也不同构 |

> 注意：pynauty 的 certificate 单独不是完整的顶点着色规范形；实现里
> 把 canonical label 对应的颜色序列追加到 key 后面（与仓库旧
> `colored_graph_key` 先例一致），并用全排列/networkx 交叉验证。

### 3.3 碰撞结果

对三个家族，\(E_r(d6)=E_r(d9)\) 在 \(r=0,1,2\) 成立，\(r=3\) 不成立：

| r | 可区分的整体编码数（距离 1..12/1..9） | label-mixed 组数（label \(d\le4\)） | same d6/d9 |
|---|---|---|---|
| 0 | 1 | 1 | true |
| 1 | 3 | 1 | true |
| 2 | 5 | 0 | true |
| 3 | 7 | 0 | false |
| 4 | 9 | 0 | false |

- **\(r=2\) 静态不可区分**：\(E_2(d6)=E_2(d9)\)，但任务若要求区分 d6/d9，
  任何只看到 \(E_2\) 的预测器（静态或在 \(E_2\) 上迭代）都不可能同时正确。
- 对 label \(d\le4\)，\(E_2\) 没有 label-mixed 组（正对照完美）；碰撞 d6/d9
  都是负类，所以该阈值任务不受影响。**碰撞是否致命取决于任务**。
- 原因：一个 rooted radius-\(r\) patch 同时看到 A、B 当且仅当
  \[
  d(A,B)\le 2r.
  \]
  \(r=3\) 时 d6 的 multiset 中出现 1 个双标记 patch，d9 中 0 个；
  \(r=4\) 时 d6 为 3 个，d9 仍 0。

### 3.4 正对照：\(E_r\) 不是无用的

label \(=1[d\le4]\)，静态规则“存在一个 \(r\)-ball 同时含 A、B”：

| r | C24 | caterpillar | grid |
|---|---|---|---|
| 0 | 0.667 | 0.667 | 0.556 |
| 1 | 0.833 | 0.833 | 0.778 |
| 2 | **1.000** | **1.000** | **1.000** |
| 3 | 0.833 | 0.833 | 0.778 |
| 4 | 0.667 | 0.667 | 0.556 |

\(r=2\) 恰好是 \(2r=4\)；\(r<2\) 漏检，\(r>2\) 过检。这说明：
**信息是精确局部时，静态读出可以完美；限制是信息半径，不是“静态一定弱”。**

### 3.5 非迭代与迭代的全图 oracle

- **一次性距离 oracle**：直接读全图 A–B 距离，d6/d9 都正确。
  它是非迭代的，但访问了 \(E_r\) 没有的全局信息。→ **迭代不是必要的**，
  只要信息够，一层读出就够。
- **深度 \(L\) 标记传播 oracle**：在完整图上传播 A/B 指示；节点同时收到
  两个标记 iff \(d\le2L\)。d6 在 L=3 检出，d9 在 L=5 检出；
  L=3 时 d6=true, d9=false，因此可区分。→ 这是另一种全图计算，
  **不是**“迭代必要性”的证据。
- 两者都说明：toy 里“Dynamic > Static”的差异来自**信息访问**，
  不是“同信息下迭代更强”。

---

## 4. 仅在该 \(E_r\) 定义下的数学限制（C）

### 4.1 信息论下界

令 \(Y\) 为任务标签，\(R^\star(E_r)=\inf_h \mathbb E[\ell(h(E_r),Y)]\)。
若存在 \(G,G'\) 使 \(E_r(G)=E_r(G')\) 但 \(Y(G)\ne Y(G')\)，则
\(R^\star(E_r)>0\)，且**任何**只以 \(E_r\) 为输入的预测器——无论静态、
单层关系读出、还是 \(L\) 层迭代——都不能把这两个样本同时判对。
这是 Bayes 下界，不是优化/容量问题。

形式化地，若 Dynamic(\(E_r,L\)) 只允许在 \(E_r\) 的表示上做关系计算，
则它仍然是 \(E_r\) 的函数：
\[
\mathcal H_{\rm dyn}(E_r,L)\subseteq\{h(E_r)\},
\]
所以它的 Bayes 风险不可能低于 \(R^\star(E_r)\)。  
只有 Dynamic(\(G,L\))（访问完整图/节点身份/`edge_index`）才可能降低下界；
此时
\[
R^\star(G)\le R^\star(E_r),
\]
差值 \(R^\star(E_r)-R^\star(G)\) 是**信息价值**，不是“迭代价值”。

### 4.2 观测差距的分解

\[
\underbrace{R(\hat h_{\rm dyn}(G))-R(\hat h_{\rm stat}(E_r))}_{\text{observed gap}}
=
\underbrace{[R^\star(E_r)-R^\star(G)]}_{\text{information}}
+
\underbrace{[R(\hat h_{\rm dyn}(G))-R^\star(G)]}_{\text{estimation(dyn)}}
-
\underbrace{[R(\hat h_{\rm stat}(E_r))-R^\star(E_r)]}_{\text{estimation(stat)}}.
\]

只比较“静态 \(E_r\)”与“MPNN on \(G\)”会把三项混在一起。要分离：
至少需要 \(S(E_r)\), \(D(E_r)\), \(S(G)\), \(D(G)\) 四臂；其中 \(S(G)\)
是信息充分/无损的静态 oracle。若 \(S(G)\approx D(G)\)，则迭代不必要；
若 \(S(G)\ll D(G)\)，则即使信息充分，迭代仍有估计/归纳偏置价值。

### 4.3 这个限制**不**自动适用于真实分子

- 真实分子上 \(E_r\) 可能几乎唯一（见 §5），此时碰撞很少；
  问题从“信息不足”变成“太细、没有统计共享”。
- 真实任务可能只需要 \(2r\) 内的局部信息；此时 \(E_2\) 可能已经足够。
- 真实任务也可能依赖 \(>2r\) 的全局关系；此时 \(E_2\) 可能不足。
  **这必须用真实 label 的碰撞/学习曲线来回答，不能从 toy 外推。**

---

## 5. 真实分子可行性（D）

### 5.1 数据与协议审计

| 项 | 结果 |
|---|---|
| Lipophilicity CSV | 4200 行；列 `CMPD_CHEMBLID,exp,smiles`；sha256 `aed41590…` |
| 冻结 split | DeepChem 2.8.0 scaffold：3360/420/420；2408 scaffold 组；`reference_match=True` |
| 跨 split 泄漏 | scaffold 0；canonical SMILES 0；canonical SMILES 重复 0 |
| row manifest | label-free；canonical/scaffold hash 与重算 0 mismatch；n_atoms 0 mismatch |
| `objects_train.pt` | 3360 个 `MolObjects`；无 `y` 属性；与源 CSV 逐行 canonical 0 mismatch；n_atoms 0 mismatch |
| 官方 valid/test | `labels.npy`、`test_metrics.json`、`validation_metrics.json` 存在但**未打开**；`test_read.lock`/`validation_read.lock` 存在 |
| ESOL | 仓库内未找到 ESOL/Delaney 数据；本轮未下载 |
| ZINC | `data/ZINC/subset/processed/train.pt` 存在；本轮未使用 |

**候选 train-only 内部分割（未冻结，不是协议）**：

- scaffold-grouped greedy 70/15/15；
- fit/valid/locked = **2352/504/504**；
- scaffold 组数 725/421/422；组重叠 0；
- 行号 hash 存在 `analysis/lipo_feasibility_audit.json`；
- **status: CANDIDATE_ONLY_NOT_A_PROTOCOL**；pilot 前必须冻结。

### 5.2 精确 \(E_r\) 成本

200 个确定性 train 分子，pynauty 精确 canonical key：

| r | 秒/分子 | 3360 投影 | patch 数 | 唯一 patch key | 最大 patch 节点 |
|---|---|---|---|---|---|
| 2 | 0.00089 | 3.0 s | 5081 | 885 | 11 |
| 3 | 0.00117 | 3.9 s | 5081 | 1487 | 17 |

- 节点重标号不变性：20 分子 × r=2,3，0 failure。
- 结论：**预处理不是瓶颈**；瓶颈在统计/学习。

### 5.3 train-only LOO 诊断（探索性，非确证）

只读官方 train 3360 行与 `train_y`；官方 valid/test 标签/指标未打开。
LOO 组均值读出 + 500 次 label permutation：

| 分组 | 组数 | 单例分子 | 最大组 | LOO MAE | 全局均值 | null MAE | in-sample \(R^2\) | null \(R^2\) |
|---|---|---|---|---|---|---|---|---|
| \(E_2\) | 3259 | 94.5% | 4 | 0.9266 | 0.9786 | 0.9971 | 0.9994 | 0.970 |
| \(E_3\) | 3262 | 94.6% | 4 | 0.9276 | 0.9786 | 0.9968 | 0.9994 | 0.971 |
| Murcko scaffold | 1568 | 30.3% | 76 | **0.7300** | 0.9786 | 1.0859 | 0.7822 | 0.466 |

解释：

- \(E_2/E_3\) 在 3360 个训练分子上几乎一分子一组。它不是“统计充分”，
  而是“身份级精细”：能几乎唯一标识训练分子，却没有共享统计强度。
- in-sample \(R^2\approx0.999\) 是**记忆**；置换 null 也有 0.97，因为组太小。
  这直接验证了“训练集指纹唯一 ≠ 统计充分/可学习/可泛化”。
- \(r=2\to3\) 没有改善 LOO（0.9266→0.9276）；多加一圈局部结构没有带来
  可测的 train-label 信息。
- 粗 scaffold 分组（组大、单例少）LOO 0.730，显著优于 \(E_r\)。
  这是偏差-方差权衡：精细指纹低偏差高方差；粗分组高偏差低方差。
- 非单例组内标签：\(E_2\) 85 组/118 对，平均绝对差 0.132，仅 3.4% 对 >0.5；
  scaffold 551 组/10904 对，平均 0.962，60.9% 对 >0.5。  
  → \(E_r\) 的碰撞组“很准但覆盖太少”；scaffold 组“很糙但数据多”。

**与旧轨的一致性**：旧轨 `explicit_composition_signal_audit_v1` 已发现
B1 半径-1 patch 多重集几乎全唯一（9993 组），严格 B0 碰撞仅 11 对且
0 对超噪声，结论 `INSUFFICIENT_OR_CONFOUNDED`。本轮的精确 \(E_r\)
近唯一性与之同向，但增加了 LOO/置换/ scaffold 对照，说明**唯一性本身
不是充分性**。

### 5.4 真实分子仍未知（D）

- 公平的 \(S_{\rm patch}(E_r)\)（patch 内共享 GNN + 池化）能否匹配
  \(D(G)\)？
- \(D(G)\) 的收益是否随 n 缩小（样本效率）还是持续（信息）？
- Lipophilicity label 是否依赖 \(>2r\) 的全局关系？当前 LOO 只说明
  精确 \(E_r\) 分组不携带多少**可共享**信息，不能直接回答长程依赖。
- 内部 locked 504 行是否有足够 power 检测 0.05 MAE 效应？
- 单 scaffold split 的方差；2408 组、组大小不均衡。
- 官方 test 已被旧轨消耗；本路线不能用它做新选择。

---

## 6. GPU pilot 要区分的两种解释（E）

### 6.1 解释 A：信息补充

- \(E_r\) 缺少任务所需的全局关系；\(D(G)\) 的收益来自额外信息。
- 可检验预测：
  - \(D(G)\) 在 n=2352 仍显著优于 \(S_{\rm patch}(E_r)\)；
  - 信息充分/无损的静态 oracle \(S(G)\) 接近 \(D(G)\)；
  - 增大 \(r\)（或使用 \(E_{r_{\max}}\)）会缩小 \(S(E_r)\) 与 \(D(G)\) 的差距。
- 若成立：关系计算的价值是“补信息”；但这不是“迭代必要”，
  因为 \(S(G)\) 也能做到。论文主张应写成“信息访问”，不是“MP 必要”。

### 6.2 解释 B：可学习性/归纳偏置

- \(E_r\) 信息上足够（或近乎足够），但精确指纹太细、样本太少；
  关系计算通过共享局部规则/聚合全局特征改善样本效率。
- 可检验预测：
  - \(D(G)-S_{\rm patch}(E_r)\) 在 n=250 大于 n=2352（负交互）；
  - 在**相同信息**下，\(D(E_r)\) 优于 \(S_{\rm patch}(E_r)\)；
  - 静态 bag-of-exact-keys \(S_{\rm bag}\) 过拟合最严重。
- 若成立：这是本路线最有价值的结论；它把“结构编码”从表示问题
  推进到“结构充分性如何转化为统计效率”的问题。

### 6.3 为什么当前证据更偏向 B（但仍未证明）

- toy 的碰撞是信息问题，但真实 Lipophilicity 上 \(E_2/E_3\) 已经近唯一；
  再加 \(r\) 没有 LOO 改善。  
- 真正的瓶颈更像“精确指纹没有统计共享”（\(E_2\) LOO 0.927 vs
  scaffold 0.730），而不是“缺少局部信息”。
- 但一个公平的 patch encoder 也可能提供共享；所以必须用
  \(S_{\rm patch}\) 而不是弱 bag-of-keys 作为静态对照。

---

## 7. 条件性最小 GPU pilot 预注册草案

> **状态：DRAFT / CONDITIONAL / NOT AUTHORIZED / NOT FROZEN。**  
> 只有用户接受 §0 的重述后才可冻结；冻结前不得训练、不得看 locked 标签。

### 7.1 标题与问题

**SSRC-LIPO-PILOT-v1: Information vs Learnability on Lipophilicity**

> 在 Lipophilicity 小样本回归上，迭代关系计算相对**同信息**的非迭代
> 结构读出的收益，是来自额外信息，还是来自可学习性/样本效率？

### 7.2 数据与冻结

- 官方 train 3360 行；使用 §5.1 的候选内部 scaffold split
  （fit 2352 / valid 504 / locked 504）。
- **冻结内容**：row indices、row hash、scaffold 组、canonical SMILES hash、
  配置文件 hash、代码 commit。冻结后不再改 split。
- 官方 valid/test 不参与任何选择；`labels.npy` 不打开。
- 主评估：内部 locked 504 行，且只在所有模型/超参冻结后读一次。
- 调参：只用 fit 2352 训练、valid 504 早停/选模；locked 只做最终评估。

### 7.3 臂（5 + 1 baseline）

| 臂 | 输入 | 计算 | 目的 |
|---|---|---|---|
| \(S_{\rm bag}(E_r)\) | 精确 patch key 的 count/bag | 线性/DeepSets，无迭代 | 身份级指纹诊断；预计高方差 |
| \(S_{\rm patch}(E_r)\) | 每个 patch 的节点/边属性 | patch 内共享 GNN（2 层）+ sum/mean 池化 + DeepSets；**无跨 patch 全图迭代** | 公平静态关系对照 |
| \(D(E_r)\) | 与 \(S_{\rm patch}\) 相同 | patch token 上 L 层 self-attention（L=2/4）；信息相同 | 分离“迭代”与“关系” |
| \(D(G)\) | 完整图 \(G\) | L 层 GINE/MPNN；同宽/同深 | 应用动态基线；信息更多 |
| \(S(G)\) | 信息充分/无损的静态摘要（如全 pairs 距离直方图 + 原子/键属性，或 \(r_{\max}\) 的 \(E_{r_{\max}}\)） | 非迭代 set/DeepSets 读出 | 检验“信息够时迭代是否仍必要” |
| Scaffold baseline | Murcko scaffold one-hot / scaffold-level mean | 线性/ridge；CPU | 上下文基线，不是主 GPU 臂 |

**注意**：\(S_{\rm bag}\) 是诊断臂，不是主静态基线；主静态基线是
\(S_{\rm patch}\)，否则会沦为 straw man。

### 7.4 匹配规则（公平性）

- **参数量**：所有主臂（\(S_{\rm patch}, D(E_r), D(G)\)）参数在 ±5% 内；
  若差异超出，给较小臂加参数匹配 adapter（同读出）。
- **读出**：同一 pooling + 同一 MLP head；只改 encoder/迭代。
- **预算**：同一 optimizer（AdamW）、LR、weight decay、batch、epoch 上限、
  早停 patience；同一数据顺序/种子。
- **FLOPs**：用 `torch.utils.flop_counter.FlopCounterMode` 在固定 batch=64
  上测 forward FLOPs；报告 forward/epoch 与总训练 FLOPs。若无法精确匹配，
  预注册“参数 ±5%、FLOPs ±10%”，并报告实际差异；不把 FLOPs 不匹配的
  结果解释为机制差异。
- **预处理**：\(E_r\) 缓存只算一次；报告秒/分子（当前 r=2 约 0.00089，
  r=3 约 0.00117）。\(S_{\rm bag}\) 的 vocabulary/hashing 规则固定。

### 7.5 数据规模与种子

- 嵌套 scaffold-grouped 子集：n = 250, 500, 1000, 2352。
  按 scaffold 组抽样，不按分子抽样；同一 n 的所有臂用同一子集。
- 5 seeds（0–4）；所有臂配对。
- 若 seed 间 MAE 的 SD > 0.03，则 pilot 对 0.05 效应欠 power；
  此时不把 null 解释为“无效应”，只报告 CI。

### 7.6 训练协议

- 300 epochs 上限；early stopping patience 30，监控内部 valid MAE；
- batch 64；AdamW；LR 1e-3；weight decay 1e-5；cosine 或 plateau（固定一种）；
- 不做 test-time augmentation；不做外部预训练；不集成（除非预注册）。
- 所有 run 记录：commit、config hash、split hash、seed、n、arm、参数量、
  FLOPs、wall-clock、best epoch、valid/locked 指标。

### 7.7 主端点与统计

**主端点**：
\[
\Delta(n)=\mathrm{MAE}_{S_{\rm patch}(E_r)}(n)-\mathrm{MAE}_{D(G)}(n)
\]
在 locked 504 上，n=250 的配对差（5 seeds），以及交互
\[
\Delta(250)-\Delta(2352).
\]
正值 = 动态全图更好。

- 主检验：配对 t / bootstrap 95% CI；5 seeds；单侧（预注册方向：
  \(D(G)\) 更好）。
- 多重比较：3 个主对照（\(D(G)-S_{\rm patch}\) at n=250；交互；
  \(D(E_r)-S_{\rm patch}\) at n=250）用 Holm 校正。
- 报告 seed 级散点，不只报均值。

### 7.8 判定规则

- **H1（可学习性）支持**：
  - \(\Delta(250)\ge0.05\) 且 95% CI 排除 0；
  - \(\Delta(2352)\le0.5\Delta(250)\)，或交互显著为负；
  - \(D(E_r)-S_{\rm patch}\ge0.03\) at n=250，CI 排除 0。
- **H2（信息补充）支持**：
  - \(\Delta(2352)\ge0.05\) 且 CI 排除 0；
  - \(S(G)\) 与 \(D(G)\) 在 n=2352 的差 \(\le0.02\)（信息充分时迭代不必要）；
  - 或 \(S(G)\ll D(G)\) 且 \(D(G)\) 在 n=2352 仍大幅优于 \(S_{\rm patch}\)
    （信息充分时迭代仍有估计价值）。
- **H0 / 停止**：
  - 所有主对照 |Δ| < 0.02 且 CI 窄；
  - 或 \(S_{\rm patch}\ge D(G)\) 在所有 n；
  - 或 \(S(G)\) 完全解释 \(D(G)\) 且 \(S_{\rm patch}\) 也匹配 \(S(G)\)。
- **混合/不确定**：报告，不选边；下一轮只做一个诊断实验，不扩架构搜索。

### 7.9 资源

- 1× A100-40GB；≤12 GPU-hours；5 seeds × 4 n × 5 臂 = 100 runs；
  单 run 预计 1–5 min（Lipophilicity 小图），加早停/评估余量。
- CPU：≤16 GB RAM；\(E_r\) 预处理 <1 min（当前 3–4 s/3360，r=2/3）。
- 不下载外部数据；ESOL 若未来加入，需单独预注册。

### 7.10 停止/排除规则

- 若 >20% runs 出现 NaN/发散，停止并修实现；不把失败 run 当结果。
- 若 \(S_{\rm patch}\) 与 \(D(G)\) 的 valid 曲线都平坦（<全局均值），
  说明任务/特征有问题；停止，不调架构。
- 若 locked 评估前发现 split/config 变更，作废并重新冻结；不“补跑”。
- 任何 official test 读取都不在本 pilot 范围内。

### 7.11 预注册的“反重复”检查

在跑 GPU 前，必须回答：

1. 我们的主对照是否已经存在于 Eijkelboom/GRIT/CSA/MoSE/SPSE/GPM？
   → 没有“精确 \(E_r\) + information-matched \(D(E_r)\) + \(S(G)\) oracle +
   Lipophilicity 样本效率”的组合。
2. 我们是否只是把“no-MP 可行”再演示一遍？
   → 不是；主端点是**同信息下迭代 vs 非迭代**与**信息 oracle**，
   不是 no-MP vs MP 的平均性能。
3. 如果 \(S_{\rm patch}\) 赢了，我们是否愿意发表负结果？
   → 是；负结果会写进 REPORT，不包装成新方法。

---

## 8. 判断与理由

### 8.1 为什么是 `REFRAME_NEEDED` 而不是直接 GO

1. **文献已覆盖**：原表述的核心主张已被 Eijkelboom 2023 直接研究，
   并被多篇 no-MP SOTA 强烈反证。直接 GO 会重复。
2. **toy 不支持“迭代必要”**：\(E_2\) 碰撞是信息问题；一次性全图
   oracle 已解决同一任务。toy 只证明“信息访问决定下界”。
3. **真实数据把问题推向可学习性**：精确 \(E_2/E_3\) 近唯一、
   LOO 只比全局均值好 0.052；scaffold 好 0.249。当前证据更像
   “统计效率/归纳偏置”而不是“局部信息缺失”。
4. **直接 static-vs-MPNN 会混淆**：两者信息访问不同，不能把差异
   归因于“关系计算”。必须重述为 information-matched 设计。

### 8.2 为什么不是 `STOP_THIS_FORMULATION`

- 重述后的问题有明确的、可反驳的、未被文献覆盖的组合：
  精确 \(E_r\) 的任务相对充分性 + information-matched 分解 +
  Lipophilicity 小样本。
- 真实数据已经给出一个非平凡动机（精确指纹的统计低效），
  不是“再试一个模型”。
- 负结果也有价值：如果 \(S_{\rm patch}\) 匹配 \(D(G)\)，就能有边界地
  说“在这个任务上，迭代关系计算不改善统计效率”；这比又一个
  no-MP 模型更有信息量。

### 8.3 若用户坚持原表述

→ **`STOP_THIS_FORMULATION`**。理由：原表述的新颖性已被覆盖，
且 toy/真实数据都表明“MP 是否必要”不是可解释的单一问题；
继续做只会得到混淆的 static-vs-MPNN 数字。

---

## 9. 复现与产物

### 9.1 命令

```bash
# CPU toy + collision + oracle + positive control
uv run python tracks/structural-sufficiency/results/phase0/code/run_phase0.py

# 17 个 CPU tests
uv run pytest -q tracks/structural-sufficiency/results/phase0/tests

# 真实数据/协议/成本审计（不读标签）
uv run python tracks/structural-sufficiency/results/phase0/code/lipo_feasibility_audit.py

# train-only LOO/置换诊断（只读 train labels）
uv run python tracks/structural-sufficiency/results/phase0/code/train_only_er_information_audit.py
```

### 9.2 产物

| 路径 | 内容 |
|---|---|
| `analysis/phase0_results.json` | toy 全部结果、code sha256、git commit、scope |
| `analysis/lipo_feasibility_audit.json` | split/manifest/objects/候选内部 split/成本/标签隔离 |
| `analysis/train_only_er_information_audit.json` | LOO、置换、scaffold 基线、冲突统计 |
| `code/ssrc_toy.py` | 精确 \(E_r\)、pynauty/exhaustive/networkx、oracle |
| `code/run_phase0.py` | CPU 驱动 |
| `code/lipo_feasibility_audit.py` | 真实数据审计 |
| `code/train_only_er_information_audit.py` | train-only 信息诊断 |
| `tests/test_phase0.py` | 17 tests |

> 依仓库约定，`analysis/*.json` 是本地证据（`tracks/*/results/**/*.json`
> 被 `.gitignore` 忽略），fresh clone 不包含；用 §9.1 的命令可重生成。
> 本报告保留关键数字与判据，不依赖 JSON 文件存在。

### 9.3 环境

- Python 3.12.14（`uv run`）；torch 2.5.1+cu124；numpy 2.1.3；
  networkx 3.4.2；rdkit 2026.09.1；pynauty 2.8.8.1；pandas。
- `FlopCounterMode` 可用（`resources_and_reuse.flop_counter_available=True`）。
- 本轮未使用 GPU；scope 字段记录 `cpu_only=True, training=False,
  labels_read=False, official_valid_or_test_read=False`。

---

## 10. 限制与什么会改变判断

### 10.1 限制

- Toy 是手工结构，不是真实分子分布；不能外推。
- train-only LOO 是探索性统计，不是模型；不能替代 pilot。
- \(E_r\) 的精确 key 对化学属性很敏感；不同属性粒度会改变唯一性。
- 候选内部 split 未冻结；当前数字只用于可行性，不是最终协议。
- 官方 test 已被旧轨消耗；本路线不能再用它做无偏确认。
- 文献检索对 ESA+ 不完整；若 ESA+ 是已发表的精确 \(E_r\)+信息分解工作，
  新颖性判断需重做。
- 没有做 power analysis 的真实 seed 方差；pilot 的 0.05 阈值是暂定。

### 10.2 什么会改变判断

- 若找到一篇已做“精确 \(E_r\) 碰撞 + information-matched static/dynamic +
  Lipophilicity 样本效率”的论文 → 直接 `STOP_THIS_FORMULATION`。
- 若用户给出 ESA+ 的准确论文且它已覆盖本路线 → 重做边界。
- 若 pilot 显示 \(S_{\rm patch}(E_r)\) 在所有 n 匹配 \(D(G)\) →
  可学习性解释不成立；停止关系计算主张。
- 若 pilot 显示 \(S(G)\approx D(G)\) 且 \(S_{\rm patch}\ll D(G)\) →
  信息解释成立；主张应改为“信息访问”，不是“MP 必要”。
- 若 pilot 显示 \(D(E_r)>S_{\rm patch}\) 且交互为负 →
  可学习性解释成立；这是最有价值的路径，可进入更大规模验证。

---

## 11. 参考文献（审计用）

1. F. Eijkelboom, E. J. Bekkers, M. M. Bronstein, F. Di Giovanni.
   “Can strong structural encoding reduce the importance of Message Passing?”
   PMLR v221:278–288, 2023.  
   https://proceedings.mlr.press/v221/eijkelboom23a.html
2. Z. Wang, Z. Zhang, T. Ma, N. V. Chawla, C. Zhang, Y. Ye.
   “Beyond Message Passing: Neural Graph Pattern Machine.”
   ICML 2025, PMLR v267:65496–65517; arXiv 2501.18739.  
   https://github.com/Zehong-Wang/GPM
3. L. Ma, C. Lin, Z. Lim, et al. “Graph Inductive Biases in Transformers
   without Message Passing.” ICML 2023, PMLR v202:23321–23337; arXiv 2305.17589.  
   https://github.com/LiamMa/GRIT
4. L. Airale, et al. “Self-Attention in Colors: Another Take on Encoding
   Graph Structure in Transformers.” arXiv 2304.10933, 2023.  
   https://github.com/inria-thoth/csa
5. S. Hussain, et al. “Global Self-Attention as a Replacement for Graph
   Convolution.” KDD 2022:655–665; arXiv 2108.03348.  
   https://github.com/shamim-hussain/egt
6. L. Bao, et al. “Homomorphism Counts as Structural Encodings for Graph
   Learning.” ICML 2025; arXiv 2410.18676.  
   https://github.com/linusbao/MoSE
7. L. Airale, A. Longa, M. Rigon, A. Passerini, R. Passerone.
   “Simple Path Structural Encoding for Graph Transformers.” arXiv 2502.09365, 2025.
8. G. Bouritsas, et al. “Improving Graph Neural Network Expressivity via
   Subgraph Isomorphism Counting.” TPAMI 2022; arXiv 2006.09252.  
9. N. Shervashidze, et al. “Weisfeiler-Lehman Graph Kernels.” JMLR 2011.  
10. N. Shervashidze, et al. “Efficient Graphlet Kernels for Large Graph
    Comparison.” AISTATS 2009.  
11. D. Rogers, M. Hahn. “Extended-Connectivity Fingerprints.” J. Chem.
    Inf. Model. 2010.  

> 注：MoSE 名称有冲突；本文只指同态计数 MoSE（arXiv 2410.18676）。
> ESA+ 未唯一识别，未列入；需用户补充准确信息。
