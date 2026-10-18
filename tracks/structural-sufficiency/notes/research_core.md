# 研究核心 — SSRC: Structural Sufficiency & Relational Computation

Track: `structural-sufficiency`  
状态: Phase 0 权威概念文档（2026-10-18）；**2026-10-19 已加纠错**  
上游语境: `docs/luyin/luyin20.txt`, `docs/luyin/luyin21.txt`; 旧轨 `structure-contribution` 的关闭决定仅作边界，不继承其模型约束。

> **CORRIGENDUM (2026-10-19):** 本文的 §2.5 风险分解符号、§4.3/§5 对
> LOO 单例的解释、以及 pilot 的方向/停止规则，已由
> [`phase0_corrigendum.md`](./phase0_corrigendum.md) 纠正。本文保留 Phase 0
> 原始表述以保持历史可审计；凡与纠错文件冲突处，以纠错文件为准。
> 三层概念（A 身份可辨识性 / B 任务信息充分性 / C 有限样本可学习性）
> 的定义见纠错文件 §1。

> 本文只记录**可检验命题、定义、反例与未定假设**。所有数字的出处是
> `results/phase0/`；文献边界见 `prior_work_boundary.md`。  
> 标记：【已确定】定义/约束；【已证明-在本定义下】CPU 数学事实；
> 【暂定】待 pilot 检验；【未知】真实分子上未验证。

---

## 1. 一句话问题【已确定】

> 给定一个显式的结构表示 \(E_r\)，学习式关系计算（消息传递、显式对象间交互、
> 迭代注意力等）在什么时候是**必要的**？它的作用是
> (A) 补充 \(E_r\) 中缺失的信息，还是
> (B) 在信息不变的前提下改善可学习性、样本效率或泛化？

“必要”必须相对于一个**任务分布**和**信息访问权限**定义；脱离这两者，
“消息传递是否必要”没有可反驳的含义。

---

## 2. 定义

### 2.1 带属性图

令 \(G=(V,E,\alpha,\beta)\)，其中 \(\alpha:V\to\mathcal A\) 是节点属性，
\(\beta:E\to\mathcal B\) 是边属性。允许 \(G\) 有多个连通分量；允许查询标记
\(L\subseteq V\)（本文的 toy 中 \(L=\{A,B\}\)）。

### 2.2 精确 rooted-radius-\(r\) 编码 \(E_r\)

对每个根 \(v\in V\)，令
\[
N_r(v)=\{u\in V:\operatorname{dist}_G(u,v)\le r\},
\]
取**诱导子图** \(G[N_r(v)]\)，保留全部节点/边属性，并把 \(v\) 标为根、
把 \(L\cap N_r(v)\) 标为查询标记。令
\[
E_r(G,L)=\{\![(G[N_r(v)],v,L\cap N_r(v))]\!] : v\in V\}
\]
为这些带根、带属性、带标记的诱导子图的**同构类型多重集**（multiset，
保留重复次数；不保留根的身份、不保留 patch 之间的邻接关系）。

本文的 \(E_r\) 是**精确**的：同构判定用 canonical labeling / 全排列交叉验证 /
独立同构检验，不用普通哈希代替。节点重标号不改变 \(E_r\)。

> 重要：\(E_r\) 是经典对象。它接近 WL subtree kernel 的 rooted subtree 多重集、
> graphlet/local fingerprint、GSN 的子结构计数。**本路线不声称 \(E_r\) 本身新颖**；
> 新颖性候选只在“信息充分性 vs 可学习性”的受控分解。

### 2.3 三个读出类

- **Static(\(E_r\))**：只以 \(E_r\) 为输入、非迭代的预测器。形式上是某个
  \(h(E_r)\)，没有节点身份、没有 \(G\) 的 `edge_index`、没有跨 patch 的迭代。
  其 Bayes 风险为
  \[
  R^\star(E_r)=\inf_h \mathbb E[\ell(h(E_r),Y)].
  \]
- **Dynamic(\(E_r,L\))**：只以 \(E_r\) 为输入，但在 \(E_r\) 的表示上做 \(L\) 轮
  关系计算（例如把 patch occurrence 作为 token 做 \(L\) 层 self-attention）。
  **信息访问与 Static 完全相同**。因此它的 Bayes 风险仍是 \(R^\star(E_r)\)；
  它只能缩小估计误差，不能突破 \(E_r\) 的碰撞。  
  **边界**：若给 dynamic 提供 root identity、patch 间邻接/重叠 incidence，
  它已经不是在 \(E_r\) 上计算，而是在一个比 \(E_r\) 更强的表示上计算；
  必须把这类信息增量单独归入 Dynamic(\(G\)) 一侧，不能算作
  “同信息下迭代更强”。
- **Dynamic(\(G,L\))**：以完整图 \(G\)（含 `edge_index`、节点身份）和标记 \(L\)
  为输入，做 \(L\) 轮消息传递/注意力。它的 Bayes 风险是 \(R^\star(G)\)，一般
  \[
  R^\star(G)\le R^\star(E_r).
  \]
  两者的差是**信息价值**，不是“迭代的价值”。

- **Non-iterative relational control \(R_1(E_r)\)**：在 \(E_r\) 的 token 上做
  **一层**注意力/关系读出，不迭代。它仍是 \(E_r\) 的函数，Bayes 风险为
  \(R^\star(E_r)\)；它用于区分“关系”与“迭代”。

### 2.4 两个问题

**问题 A（信息可辨识性 / identifiability）**  
\(E_r\) 是否把任务相关的图分开？若存在 \(G,G'\) 使
\(E_r(G)=E_r(G')\) 但 \(Y(G)\ne Y(G')\)，则任何只看到 \(E_r\) 的预测器
（无论静态还是“在 \(E_r\) 上迭代”）都不可能同时拟合两者；这是
**信息论下界**，不是优化失败。CPU 可以证明碰撞；真实分子上是否有
label-conflicting 碰撞是数据问题。

**问题 B（可学习性 / sample efficiency / generalization）**  
即使 \(E_r\) 信息充分（\(R^\star(E_r)=R^\star(G)\)），从有限样本学习
\(h(E_r)\) 仍可能很难；关系计算可能通过权重共享、组合复用、局部归纳偏置
改善样本效率。反之，若 \(E_r\) 信息不足，Dynamic(\(G,L\)) 的收益首先是
**补信息**，不能归因于“迭代更好”。

### 2.5 观测差距的分解【已确定；2026-10-19 符号纠正】

> **原 Phase 0 版本把信息项写成 \(+[R^\star(E_r)-R^\star(G)]\)，这是
> 符号错误。正确式子如下；旧式已废弃，详见
> [`phase0_corrigendum.md`](./phase0_corrigendum.md) §2。**

令
\[
I:=R^\star(E_r)-R^\star(G)\ge 0,\qquad
\mathrm{excess}_{\rm dyn}:=R(\hat h_{\rm dyn})-R^\star(G),\qquad
\mathrm{excess}_{\rm stat}:=R(\hat h_{\rm stat})-R^\star(E_r).
\]
则
\[
\boxed{
R(\hat h_{\rm dyn})-R(\hat h_{\rm stat})
= -I+\mathrm{excess}_{\rm dyn}-\mathrm{excess}_{\rm stat}
}
\]
这个恒等式是 pilot 设计的基础：只比较“静态 \(E_r\)”与“MPNN on \(G\)”会把
information 与 estimation 混在一起。若两个模型都只以 \(E_r\) 为输入
（S vs X），则信息项为 0，只剩
\(\mathrm{excess}_{X}-\mathrm{excess}_{S}\)。
Phase 1a 不训练旧草案的 \(S(G)\) oracle，也不声称四/三个有限容量模型能
识别真实 Bayes 风险；详见纠错文件 §2.2 与 §5。

---

## 3. Toy：C24 上的精确碰撞与 \(2r\) 边界【已证明-在本定义下】

### 3.1 构造

- \(G_{24,d}\)：24 个 C 节点的环；节点 0 标为 A，节点 \(d\) 标为 B；
  边类型为 ring；A/B 是查询标记。
- 三个独立家族重复同一实验：
  `cycle_graph`（C24）、`caterpillar_graph`（40 spine + leaves）、
  `grid_graph`（20×9）。
- 每个家族比较 \(d=6\) 与 \(d=9\)。

### 3.2 结果

对三个家族，\(r=0,1,2\) 时
\[
E_r(G_{24,6})=E_r(G_{24,9}),
\]
\(r=3\) 时二者不同。精确结果（`analysis/phase0_results.json`）：

| r | 可区分的整体编码数（距离 1..12/1..9） | label-mixed 组数（label \(d\le4\)） | \(E_r(d6)=E_r(d9)\) |
|---|---|---|---|
| 0 | 1 | 1 | true |
| 1 | 3 | 1 | true |
| 2 | 5 | 0 | true |
| 3 | 7 | 0 | false |
| 4 | 9 | 0 | false |

- \(r=2\) 时 \(E_2(d6)=E_2(d9)\)：**任何只看到 \(E_2\) 的静态预测器都无法区分
  d6 与 d9**。这不是模型容量问题；是 \(E_2\) 丢失了全局距离信息。
- \(r=3\) 时 d6 的 multiset 中出现 **1 个同时含 A、B 的 patch**；d9 中为 **0**。
  \(r=4\) 时 d6 为 3 个，d9 仍为 0。
- 原因：一个 rooted radius-\(r\) patch 能同时看到 A、B，当且仅当
  \[
  d(A,B)\le 2r.
  \]
  所以 \(E_r\) 能回答的“两点关系”有一个精确的 \(2r\) 边界。

### 3.3 正对照：\(E_r\) 不是无用的

任务 label \(=1[d\le4]\)。确定性静态规则
“存在一个 \(r\)-ball 同时含 A、B” 的准确率：

| r | C24 | caterpillar | grid |
|---|---|---|---|
| 0 | 0.667 | 0.667 | 0.556 |
| 1 | 0.833 | 0.833 | 0.778 |
| 2 | **1.000** | **1.000** | **1.000** |
| 3 | 0.833 | 0.833 | 0.778 |
| 4 | 0.667 | 0.667 | 0.556 |

\(r=2\) 恰好是 \(2r=4\)，规则完美；\(r<2\) 漏检（看不到距离 3–4 的两点），
\(r>2\) 过检（把 d=5/6 也判正）。这证明：\(E_r\) 的信息是**精确局部**的，
不是“静态模型一定不行”。任务若只需要 \(2r\) 内的关系，静态读出可以完美。

### 3.4 非迭代与迭代的全图 oracle

- **一次性距离 oracle**：直接读全图 A–B 距离，d6/d9 都正确。它是
  **非迭代**的，但访问了 \(E_r\) 没有的全局信息。它证明：对这个任务，
  **迭代不是必要的**；只要信息够，一层读出就够。
- **深度 \(L\) 标记传播 oracle**：在完整图上传播 A/B 指示，节点同时收到
  两个标记 iff \(d\le2L\)。d6 在 \(L=3\) 检出，d9 在 \(L=5\) 检出；
  \(L=3\) 时 d6=true, d9=false，因此可区分。这只是**另一种全图计算**，
  不是“迭代必要性”的证据。

### 3.5 结论【已证明-在本定义下】

1. \(E_2\) 的碰撞是**信息缺失**，静态与“在 \(E_2\) 上迭代”都无法突破。
2. 全图一次性读出可以解决 toy；所以 toy 不支持“必须迭代”。
3. 任何“Dynamic > Static”的观察，若 Dynamic 访问了 `edge_index`/节点身份
   而 Static 只看到无序 patch 多重集，都是**信息访问差异**，不能解释为
   “同信息下动态更强”。pilot 必须把这一点做成显式对照。

---

## 4. 真实分子：Lipophilicity 的 Phase 0 诊断【已观察；非确证】

### 4.1 数据与协议

- `data/lipophilicity/Lipophilicity.csv`：4200 行；只用 structures。
- 旧轨冻结的 DeepChem 2.8.0 scaffold split：train/valid/test =
  3360/420/420；2408 个 scaffold 组；跨 split scaffold/SMILES 泄漏 0。
- 官方 valid/test 标签**未读**；`labels.npy` 未打开；旧轨的
  `test_metrics.json`/`validation_metrics.json` 未打开。
- `objects_train.pt` 是 label-free 的 `MolObjects`（3360 个），
  与 RDKit 源结构逐行一致；无 `y` 属性。
- 候选 train-only 内部分割（**未冻结，不是协议**）：
  scaffold-grouped 70/15/15 = 2352/504/504，组重叠 0。

### 4.2 精确 \(E_r\) 的成本

在 200 个确定性 train 分子上，用 pynauty 精确 canonical key：

| r | 秒/分子 | 3360 分子投影 | patch 数 | 唯一 patch key 数 | 最大 patch 节点 |
|---|---|---|---|---|---|
| 2 | 0.00089 | 3.0 s | 5081 | 885 | 11 |
| 3 | 0.00117 | 3.9 s | 5081 | 1487 | 17 |

节点重标号不变性：20 个分子 × r=2,3，0 失败。  
→ 在 Lipophilicity 上，精确 \(E_r\) 的**预处理成本可以忽略**；瓶颈不在算 \(E_r\)。

### 4.3 关键诊断：精确 \(E_r\) 是“精确但稀疏”的指纹

只用 train labels 的 leave-one-out（LOO）组均值读出 + 500 次 label permutation：

| 分组 | 组数 | 单例分子比例 | 最大组 | LOO MAE | 全局均值 MAE | 置换 null MAE | in-sample \(R^2\) | null \(R^2\) |
|---|---|---|---|---|---|---|---|---|
| \(E_2\) | 3259 | 94.5% | 4 | 0.9266 | 0.9786 | 0.9971 | 0.9994 | 0.970 |
| \(E_3\) | 3262 | 94.6% | 4 | 0.9276 | 0.9786 | 0.9968 | 0.9994 | 0.971 |
| Murcko scaffold | 1568 | 30.3% | 76 | **0.7300** | 0.9786 | 1.0859 | 0.7822 | 0.466 |

解释：

- \(E_2/E_3\) 在 3360 个训练分子上几乎**每个分子一个组**（94.5% 单例）。
  它不是“统计充分”，而是“身份级精细”：足以几乎唯一地标识训练分子，
  却没有共享统计强度。in-sample \(R^2\approx0.999\) 是**记忆**，不是泛化；
  置换 null 也有 0.97，正是因为组太小。
- \(r=2\to3\) **没有**改善 LOO MAE（0.9266 → 0.9276）。多加一圈局部结构
  没有带来可测的 train-label 信息。
- 粗粒度的 Murcko scaffold 分组（组更大、单例少）LOO MAE 0.730，
  显著优于 \(E_r\)。这不是说 scaffold 是“正确表示”；它说明
  **偏差-方差权衡**：精细指纹低偏差、高方差；粗分组高偏差、低方差。
- \(E_r\) 的非单例组内标签很一致（\(E_2\)：85 组、118 对，平均绝对差 0.132，
  仅 3.4% 对 >0.5；scaffold：551 组、10904 对，平均 0.962，60.9% 对 >0.5）。
  即：\(E_r\) 的碰撞组“很准”，但覆盖太少；scaffold 组“很糙”，但数据多。
- 这与旧轨 `explicit_composition_signal_audit_v1` 的“patch 多重集几乎全唯一”
  一致；再次说明 **训练集指纹唯一 ≠ 统计充分/可学习/可泛化**。

### 4.4 这对两个解释的含义【暂定】

在 Lipophilicity 上，最可能的瓶颈**不是**“\(E_r\) 缺少局部信息”，而是
“精确局部指纹没有统计共享”。因此：

- **信息补充解释**（A）在 toy 的全局距离任务上成立；在 Lipophilicity 上
  尚未被支持。\(E_3\) 没有比 \(E_2\) 多出可测的 label 信息。
- **可学习性/归纳偏置解释**（B）更符合当前证据：动态关系计算的价值可能在于
  跨分子共享局部规则、把细粒度环境聚合成 scaffold/global 特征，从而改善
  小样本泛化；而不是“补上 \(E_r\) 缺的局部结构”。
- 但这是**假设**。一个公平的静态 patch encoder（在 patch 内做共享 GNN，
  再池化）也可能获得同样的共享；pilot 必须把“权重共享”与“全图迭代”分开。

### 4.5 仍然未知【未知】

- 公平静态 patch encoder 能否匹配动态全图模型？
- 动态全图模型的收益是否在 n=2352 时消失（样本效率）还是持续（信息）？
- Lipophilicity 的 label 是否依赖 \(>2r\) 的全局关系？当前 LOO 诊断只说明
  精确 \(E_r\) 分组不携带多少可共享信息，不能直接回答“长程依赖”。
- 官方 test 已被旧轨消耗；本路线不能用它做新选择。内部 locked 504 行
  是否足够检测 0.05 MAE 的效应，需要 power analysis。
- 单 scaffold split 的方差；2408 个 scaffold 组、3360 个训练分子，
  scaffold 组大小分布很不均衡。

---

## 5. Pilot 的可检验假设【暂定】

数据：Lipophilicity 官方 train 3360 行；冻结候选内部 scaffold split
（fit 2352 / valid 504 / locked 504）；官方 valid/test 不参与选择。

**Phase 1a 只实施三个臂（2026-10-19 修正；旧五臂草案不执行）：**

1. **S（无跨 patch 交互）**：每个 patch 用共享的局部 GINE 编码成 patch
token；逐 token 的残差 FFN（无跨 patch 交互）；sum/mean pooling +统一回归头。patch 内局部 GNN 允许，因此 S 不是“完全没有消息传递”。
2. **X（同信息、跨 patch 交互）**：与 S 相同的 \(E_2\)、patch 构造、局部共享编码器与回归头；在 patch token 之间加固定深度的全局 self-attention（例如 2 层），不使用位置编码、root ID、原图 adjacency、patch 间 incidence 或原子编号。
3. **G（原始图 GINE 参照）**：原始分子图 + `edge_index` 的常规 GINE/MPNN；信息访问与 S/X 不同，**只作应用表现参照**，\(G-S\)/\(G-X\) 不得解释为纯信息增益。

旧草案的 \(S_{\rm bag}\)、\(S(G)\)（全对距离 oracle）、五臂分解、结构编码家族搜索
本轮**不执行**。\(S(G)\) 的“oracle”称呼已降级：全对距离直方图不保证无损，
训练后的模型也不是 Bayes oracle；详见
[`phase0_corrigendum.md`](./phase0_corrigendum.md) §5。

**主端点（方向修正）**
\[
\mathrm{Adv}_{\rm set}(n):=\mathrm{MAE}_S(n)-\mathrm{MAE}_X(n)
\]
正值表示同信息下 X 更好。小样本优势检验：
\[
\mathrm{Adv}_{\rm set}(n_{\rm small})-\mathrm{Adv}_{\rm set}(n_{\rm large})>0.
\]
辅助比较 \(\mathrm{MAE}_S-\mathrm{MAE}_G\) 与 \(\mathrm{MAE}_X-\mathrm{MAE}_G\)
只用于应用性能差距，不作因果信息分解。

**H1（同信息交互/样本效率）**  
\(\mathrm{Adv}_{\rm set}(n_{\rm small})>0\)，且
\(\mathrm{Adv}_{\rm set}(n_{\rm small})-\mathrm{Adv}_{\rm set}(n_{\rm large})>0\)，
并在 3 个种子上方向稳定。→ 跨 patch 交互改善有限样本可学习性；
不能据此声称突破 \(E_2\) 碰撞或识别 Bayes 风险。

**H0（本 pilot 无清晰交互价值）**  
S≈X≈G，或效应小于预设的实际意义阈值（0.02 MAE），或在种子间不稳定。
→ 报告为不确定/未定，不自动扩大模型或数据集；
“差异不显著”不等于“等价”。

**停止/解释规则**  
- S/X 都未学到基本有效预测 → 先排查实现/优化；正确仍不行则停止该 pilot，
  不推断机制不存在。
- X 仅在更大 n 上更好，或结果与种子强相关 → 如实报告，不硬解释成样本效率收益。
- 一轮结果不能直接证明或否定整个 SSRC 方向。

---

## 6. 非声明（写进论文/汇报时必须保留）

1. **不**声称“强结构编码减少消息传递需求”是新的；Eijkelboom et al. 2023
   已直接研究，GRIT/CSA/MoSE/SPSE/GPM 已展示 no-MP 的竞争力/SOTA。
2. **不**声称“消息传递一般必要/不必要”；本路线只研究特定任务、特定
   信息访问、特定预算下的分解。
3. **不**声称 \(E_r\) 表示本身新颖；它是 rooted induced-subgraph isomorphism
   type 多重集，接近 WL subtree / graphlet / GSN 子结构计数。
4. **不**把 toy 的 \(E_2\) 碰撞外推到真实分子；真实分子上是否有
   label-conflicting 碰撞是数据问题，当前只做了 train-only LOO 诊断。
5. **不**把“训练集指纹唯一”当作“统计充分/可学习/可泛化”。
6. **不**把 Dynamic(\(G\)) 访问 `edge_index` 而 Static(\(E_r\)) 只看到无序
   patch 多重集的差异，解释为“同信息下动态更强”。
7. **不**把 toy 的深度传播 oracle 当作“迭代必要性”证据；一次性全图距离
   oracle 已反驳该解释。
8. **不**声称本轮做了 GPU 训练、ZINC 搜索、新架构设计、外部无标签预训练，
   或读取了官方 valid/test 标签。

---

## 7. 术语表

| 术语 | 含义 |
|---|---|
| \(E_r\) | 精确带属性 rooted-radius-\(r\) 诱导子图同构类型多重集 |
| Static(\(E_r\)) | 只以 \(E_r\) 为输入、非迭代的预测器 |
| Dynamic(\(E_r,L\)) | 在 \(E_r\) 表示上做 \(L\) 轮关系计算；信息与 Static 相同 |
| Dynamic(\(G,L\)) | 在完整图上做 \(L\) 轮消息传递/注意力；信息更多 |
| \(R_1(E_r)\) | 在 \(E_r\) 上做一层关系读出的非迭代对照 |
| 问题 A | \(E_r\) 是否碰撞到任务不可辨识 |
| 问题 B | 信息充分时，有限样本/预算下是否可学 |
| \(2r\) 边界 | 一个 \(r\)-ball 能同时看到两标记 iff 距离 \(\le2r\) |
| 信息价值 | \(R^\star(E_r)-R^\star(G)\) |
| 估计差距 | \(R(\hat h)-R^\star(\cdot)\) |
