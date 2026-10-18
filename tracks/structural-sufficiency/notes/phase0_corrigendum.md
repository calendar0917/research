# SSRC Phase 0 corrigendum — definitions, risk decomposition, literature, pilot rules

日期: 2026-10-19  
Track: `tracks/structural-sufficiency`  
性质: **Phase 0 的纠错与增补，不重做 Phase 0，不修改旧证据文件**  
适用: `notes/research_core.md`, `notes/prior_work_boundary.md`,
`results/phase0/REPORT.md` 中与本文件冲突的表述；本文件优先。

> Phase 0 的 CPU 证据（toy 碰撞、正对照、Lipophilicity 可行性/LOO）保留原样。
> 本文件只修正**解释、符号、方向、停止规则和文献口径**，并把新的 Phase 1a
> 设计限制写清楚。旧报告顶部已加入指向本文件的纠错链接。

---

## 1. 三层概念必须分开

Phase 0 把三件事混在同一句“结构是否充分”里。Phase 1a 起严格区分：

### A. 有限样本的身份可辨识性（identifiability）

训练集中的图能否被 \(E_r\) 区分？即对两个训练图 \(G,G'\)，
\(E_r(G)=E_r(G')\) 是否发生？这是**表示碰撞**问题，可以用精确同构/规范键
在 CPU 上回答。它不等于任务信息充分，也不等于可学习。

### B. 预测任务的信息充分性（information sufficiency）

是否存在
\[
P(Y\mid G)=P(Y\mid E_r(G))
\]
或对选定损失 \(\ell\) 有相同 Bayes 风险：
\[
R^\star(E_r):=\inf_h \mathbb E[\ell(h(E_r),Y)]
      =\inf_h \mathbb E[\ell(h(G),Y)]=:R^\star(G)?
\]
这是**任务相对**的问题。\(E_r\) 可能唯一标识训练分子（A 很强），却仍然丢失
任务需要的长程关系（B 不成立）；也可能 \(E_r\) 有碰撞但碰撞对之间标签相同
（A 有碰撞，B 仍充分）。A 和 B 都不能由“训练集 94.5% 单例”推出。

### C. 有限样本可学习性（learnability / sample efficiency）

在**相同输入表示** \(E_r\) 下，不同假设类（无跨 patch 交互的 S、跨 patch
attention 的 X）的泛化误差、样本效率和计算代价。C 是 Phase 1a 的主问题；
它必须在 B 的前提下解释，不能反过来证明 B。

### 1.1 Phase 0 LOO 的正确解释

Phase 0 的“按完整 \(E_r\) 类别分组的 LOO 均值预测器”只是**一个特定的
记忆型估计器**：

- 它的分组几乎都是单例（\(E_2\) 94.5% singleton），所以 LOO 实际上接近
  “留一后回退到全局均值”；
- 它没有共享局部编码器，没有跨分子权重共享，也没有利用 patch 内部结构；
- 它表现差（LOO MAE 0.9266 vs 全局均值 0.9786）**不能**证明“在 \(E_r\)
  上做共享 patch 编码的模型也表现差”。

因此 Phase 0 的 LOO 只能支持一个弱结论：**精确 \(E_r\) 的分组本身没有提供
很强的可共享统计强度**；它不能支持“结构已经充分/不充分”的二分判断，也
不能替代 Phase 1a 的 S/X 对照。

### 1.2 Scaffold LOO 与 scaffold-disjoint 不是同一个指标

Phase 0 的 Murcko scaffold LOO 使用**训练集中其他同骨架样本**的标签。
真正的 scaffold-disjoint 评估要求测试骨架**未出现在训练集中**。两者回答的
问题不同，不能直接互相替代，也不能把 scaffold LOO 的 0.730 当作
“scaffold-disjoint 泛化”的证据。Phase 1a 的 Monitor/Reserved 是
scaffold-disjoint 的；任何跨 scaffold 的结论必须用该口径报告。

---

## 2. 风险分解：符号纠正

设
\[
I := R^\star(E_r)-R^\star(G)\ge 0
\]
为 \(E_r\) 相对完整图 \(G\) 的**信息价值**（\(G\) 信息更多，所以风险不高于
\(E_r\)）。对训练得到的静态/动态模型定义
\[
\mathrm{excess}_{\rm dyn}:=R(h_{\rm dyn})-R^\star(G),\qquad
\mathrm{excess}_{\rm stat}:=R(h_{\rm stat})-R^\star(E_r).
\]
则正确的恒等式是
\[
\boxed{
R(h_{\rm dyn})-R(h_{\rm stat})
= -I+\mathrm{excess}_{\rm dyn}-\mathrm{excess}_{\rm stat}
}
\]
因为
\[
R(h_{\rm dyn})-R(h_{\rm stat})
=[R^\star(G)+\mathrm{excess}_{\rm dyn}]
-[R^\star(E_r)+\mathrm{excess}_{\rm stat}]
=-I+\mathrm{excess}_{\rm dyn}-\mathrm{excess}_{\rm stat}.
\]

**Phase 0 的旧文档把信息项写成了 \(+I\)，这是符号错误。** 旧式
\[
R(h_{\rm dyn})-R(h_{\rm stat})
=+[R^\star(E_r)-R^\star(G)]+\cdots
\]
必须废弃。直观检查：如果 \(h_{\rm dyn},h_{\rm stat}\) 都达到各自 Bayes 风险，
则 \(R(h_{\rm dyn})-R(h_{\rm stat})=R^\star(G)-R^\star(E_r)=-I\le 0\)，
即信息更多的动态模型不应更差；旧式的 \(+I\) 会给出相反符号。

### 2.1 两个特例

- 若 \(h_{\rm stat}\) 与 \(h_{\rm dyn}\) 都只以 \(E_r\) 为输入（S vs X），则
  \(R^\star(E_r)\) 相同，信息项为 0：
  \[
  R(h_X)-R(h_S)=\mathrm{excess}_{X}-\mathrm{excess}_{S}.
  \]
  这正是 Phase 1a 的 S/X 对照：**同信息，只比较估计/归纳偏置**。
- 若比较 \(X(E_r)\) 与 \(G(G)\)，则
  \[
  R(h_G)-R(h_X)=-I+\mathrm{excess}_{G}-\mathrm{excess}_{X}.
  \]
  观测到的 \(G-X\) 差距**不能**直接解释为信息增益；它同时包含 \(G\) 的
  估计误差和 \(X\) 的估计误差。Phase 1a 只把 \(G\) 当作应用参照，不做
  因果信息分解。

### 2.2 不能声称的事

Phase 1a 只有有限容量的 S/X/G，没有 Bayes 风险估计器。因此：

- 不能声称“识别了真实 Bayes 风险”；
- 不能把 \(I\)、\(\mathrm{excess}_{\rm dyn}\)、\(\mathrm{excess}_{\rm stat}\)
  做精确数值分解；
- 不能把 \(G-S\) 或 \(G-X\) 的观测差直接称为“信息价值”；
- 只能报告**有限容量、固定配方、固定 split** 下的预测差异和不确定性。

---

## 3. Pilot 方向与停止规则纠正

### 3.1 统一定义

\[
\boxed{\mathrm{Adv}(n):=\mathrm{MAE}_{\rm static}(n)
-\mathrm{MAE}_{\rm dynamic}(n)}
\]
**正值 = 动态模型更好**。Phase 1a 中 static = S（无跨 patch 交互），
dynamic = X（同信息、跨 patch attention）。

研究小样本优势时，检验
\[
\boxed{\mathrm{Adv}(n_{\rm small})-\mathrm{Adv}(n_{\rm large})>0}
\]
即动态模型的相对优势是否在小样本更大。不要再写“负向交互”，也不要把它
与“动态模型绝对 MAE 更小”混为一谈。

### 3.2 旧草案中必须废弃的规则

- **废弃** `D(E_r)-S_patch >= 0.03` 这种未定义方向的阈值；
- **废弃**“负向交互”作为 H1 支持条件；
- **废弃**“静态模型 MAE 大于动态模型时触发停止”——这恰好是动态模型更好的
  方向，不能当失败信号；
- **废弃**“差异不显著 ⇒ 等价”的推断。

### 3.3 不确定性表述

- “差异不显著”只表示**当前数据/精度下不能区分**，不表示等价。
- 除非预先定义等效性边界并做等效性检验（且精度足够），否则只报告
  **不确定/未定**。
- Phase 1a 是探索性 screening，不是最终显著性检验；任何 CI 都必须说明：
  单 scaffold split、3 seeds、seed 不是独立数据复制、Monitor 被用于早停。
- 分子级 bootstrap 必须按 scaffold 分组（cluster bootstrap），不能把
  seed × molecule 当作完全独立样本。

---

## 4. 文献增补与口径纠正

### 4.1 ESA+ / Buterez et al. 2025（已识别）

D. Buterez, J. P. Janet, D. Oglic, P. Liò.
“An end-to-end attention-based approach for learning on graphs.”
*Nature Communications* 16 (2025). DOI:
<https://doi.org/10.1038/s41467-025-60252-z>.

审计要点：

- 模型把图视为**边集合**，编码器**纵向交错** masked self-attention（MAB）
  与 vanilla self-attention（SAB）；MAB 的 mask 只允许共享一个节点的边对
  互相注意（连通性约束），SAB 是边 token 之间的全局交互；读出为 PMA。
- ZINC-12k：ESA+ **0.052 ± 0.001 MAE**（Table 6）。ESA+ 使用 RWSE 结构编码
  和 ZINC 的 ring-membership 节点特征；不是“无结构编码”的裸 ESA。
- **它如何区分“利用 connectivity 的交互”与“无 mask 的全局交互”？**
  它没有做这个区分。Table 5 只消融层数/层序（例如
  `M M M M S P S`、`M M M M M P`、`S M S M S P`），包含 all-masked 和
  混合配置，但**没有 all-S（unmasked-only）匹配深度/参数的控制行**。
  MAB 与 SAB 始终交错，因此论文的性能数字不能把收益归因于“连通性约束
  交互”本身；它证明的是“masked+global 交错的 edge-set attention 很强”。
- 与 SSRC 的关系：ESA+ 是“edge-set attention + 强结构编码”的强先例，
  说明 no-MP 的全局交互可以很强；但它**不**测试同信息 \(E_2\) 下的
  S vs X，也不提供 \(E_r\) 碰撞证书或信息/估计分解。Phase 1a 必须在
  prior-work 表中把它列为“应用性能先例”，不能列为“已覆盖同信息交互问题”。

### 4.2 Eijkelboom et al. 2023 Table 2 的 train/test 口径

Table 2 每个数据格是 **`train/test`**，不是单一测试 MAE。Phase 0 报告
引用的 “ZINC no-MP 0.324–0.466 vs full 0.020–0.076” 是**训练 MAE**；
必须改为测试 MAE。按原文 Table 2（3 seeds 均值）：

| 模型 / 编码 | full test | no-MP test |
|---|---:|---:|
| GCN / RW-Concat | 0.207 | 0.607 |
| GCN / RW-Tensor | 0.209 | 0.527 |
| GIN / RW-Concat | 0.263 | 0.609 |
| GIN / RW-Tensor | 0.242 | 0.538 |
| SAGE / RW-Concat | 0.192 | 0.614 |
| SAGE / RW-Tensor | 0.201 | 0.552 |

正确结论：ZINC 上 no-MP 的**测试** MAE 仍明显差于 full MP（约 0.33–0.42
的差距），所以“ZINC 上去掉 MP 损失很大”这一方向仍成立，但引用数字必须是
test 列，不能拿 train 列冒充 test。Peptides-struct Table 1 同样是
`train/test` 对；例如 GCN/RW-Tensor full test 0.252 vs no-MP test 0.268，
GIN/RW-Tensor 0.257 vs 0.266，SAGE/RW-Tensor 0.253 vs 0.267。也就是说
“Peptides 上 tensor 编码的 no-MP 与 full 接近”应引用 test 列（差距
约 0.01–0.016），不能引用 0.249 vs 0.212 这种 train 列。

### 4.3 Shen & Dong 2026 的术语重叠

C. Shen, Y. Dong. “A Graph Sufficiency Perspective for Neural Networks.”
*ACM Transactions on Probabilistic Machine Learning* 2(3), 2026.
DOI: <https://doi.org/10.1145/3816434>; arXiv:2507.10215.

审计要点：

- 该文把神经网络层解释为**图变量/成对函数**（神经元是输入与学习锚点之间的
  成对函数），研究层输出对层输入是否为**统计充分**（保持
  \(P(Y\mid \text{input})\)），并给出无限宽度/有限宽度、区域分离分布下的
  充分性条件和最优损失误差界；覆盖全连接、一般成对函数、ReLU/Sigmoid、
  CNN。
- 它的“graph”指层/神经元/锚点构成的计算图，不是分子图；它的“sufficiency”
  是统计充分性（层输出保留条件分布），不是本项目 \(E_r\) 的
  rooted-patch 碰撞/任务信息充分性。
- 因此：**不能凭标题判定重复，也不能把它当作已覆盖 SSRC**。它是
  “sufficiency”术语和统计视角的相关工作，应在论文中引用并明确对象差异：
  我们问的是图表示 \(E_r\) 对任务标签的信息充分性，以及同信息下不同
  假设类的可学习性；该文问的是网络层变换的统计充分性。
- 若未来要把 SSRC 的“信息充分性”形式化为层/表示充分性，该文是有用的
  理论工具，但不是本轮 pilot 的直接竞争者。

### 4.4 文献边界更新

- `prior_work_boundary.md` 中“ESA+ 未能唯一识别”的条目应改为已识别，
  并补上 §4.1 的边界。
- “强结构编码减少 MP 需求”仍由 Eijkelboom 2023 直接研究；ESA+/GRIT/
  CSA/MoSE/SPSE/GPM 仍是 no-MP 强先例。SSRC 的候选贡献仍然是
  **同信息 S/X 对照 + 精确 \(E_r\) 碰撞证书 + 小样本可学习性**，
  不是“no-MP 可行”或“新编码”。

---

## 5. 降级旧 S(G) “oracle”

Phase 0 草案把“全节点对距离直方图 + 原子/键属性”的非迭代模型称为
\(S(G)\) oracle。这个称呼必须降级：

1. **全节点对距离直方图不保证无损**：它只保留距离的多重集，丢失了
   “哪一对节点具有什么属性/键型”的配对信息，也丢失了距离以外的路径/
   环/连通性结构；存在不同的 \(G,G'\) 具有相同距离直方图。
2. **任何训练后的 \(S(G)\) 都不是 Bayes oracle**：它是一个有限容量、
   有限样本、固定优化的估计器；它的风险不是 \(R^\star(G)\)。
3. 因此本轮**不实施**旧的五臂 oracle 分解，也不训练 \(S_{\rm bag}\)、
   \(S(G)\) 或新的结构编码家族。Phase 1a 只有 S/X/G 三个模型。
4. 未来若需要“完整图信息的非迭代模型”，必须另外**明确定义**：
   - 输入是什么（全邻接矩阵？全对属性？距离矩阵？边列表？）；
   - 表达能力/是否无损（给出证明或反例）；
   - 参数/FLOPs/预处理成本；
   - 与 G 的训练预算如何匹配。
   不能用名字 `oracle` 代替证明，也不能把它的表现当作 Bayes 风险。

---

## 6. Phase 1a 的边界（从本纠错直接推出）

- Phase 1a 的唯一研究问题是：
  > 在输入精确相同的 \(E_2\) rooted-patch 多重集时，跨 patch 的学习式
  > 交互是否优于仅局部共享编码 + 图级池化？这种优势是否在小样本下更明显？
- 这不是“传统消息传递是否必要”的最终回答；G 只作应用参照。
- S 的 patch 内局部 GNN **允许**存在，所以 S 不是“完全没有消息传递”；
  它只是**没有跨 patch 交互**。
- X 仍然只是 \(E_2\) 多重集的函数，不能突破真正的 \(E_2\) 碰撞。
- S/X 不接收 root ID、原图 adjacency、patch 间 incidence、相对位置、
  原子编号或标签相关信息；X 的 attention 不使用位置编码。
- 任何 \(G-S\) 或 \(G-X\) 的观测差都不能直接解释为纯粹信息增益。
- 旧草案的 \(S_{\rm bag}\)、\(S(G)\)、五臂 oracle 分解、结构编码家族搜索
  本轮全部不执行。

---

## 7. 与旧文档的对应关系

| 旧位置 | 旧内容 | 本文件处理 |
|---|---|---|
| `research_core.md` §2.5 | 信息项写成 \(+[R^\star(E_r)-R^\star(G)]\) | §2 纠正为 \(-I\)；旧式保留在历史报告中，但标记废弃 |
| `results/phase0/REPORT.md` §4.2 | 同一符号错误 | 同上 |
| `results/phase0/REPORT.md` §7.7–7.8 | `D(E_r)-S_patch >= 0.03`、负向交互、静态 MAE 更大时停止 | §3 统一 Adv 方向与停止规则 |
| `results/phase0/REPORT.md` §2.1 | Eijkelboom 引用 train MAE | §4.2 改为 test MAE |
| `prior_work_boundary.md` §0/§1 | ESA+ 未识别 | §4.1 已识别并补边界 |
| `results/phase0/REPORT.md` §7.3 | \(S(G)\) 全对距离 oracle | §5 降级；本轮不实施 |
| `research_core.md` §4.3/§5 | 把 LOO 单例解释为“结构充分性”风险 | §1 三层区分 + LOO 估计器限定 |

---

## 8. 一句话总结

Phase 0 的 CPU 事实仍然有效；但“94.5% 单例 ⇒ 结构已经充分/不充分”的
解释、风险分解的符号、pilot 的方向/停止规则、Eijkelboom 的 train/test 口径、
ESA+ 的文献定位、以及 \(S(G)\) oracle 的称呼都必须按本文件纠正。Phase 1a
只问一个更窄、可反驳的问题：**同信息 \(E_2\) 下，跨 patch 交互是否改善
有限样本预测，尤其在小样本时。**
