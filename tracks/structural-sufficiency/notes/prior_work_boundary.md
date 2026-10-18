# 先验工作边界 — SSRC Phase 0 文献审计

审计日期: 2026-10-18  
审计人: Agent（本地 PDF/HTML 精读 + OpenAlex/arXiv/Crossref/Semantic Scholar 检索）  
范围: 用户指定的 Eijkelboom 2023、Wang ICML 2025 NGPM/GPM、CSA、GRIT、ESA+、
Edge Transformer/EGT、MoSE、SPSE；另加 GSN 作为“子结构计数 + MP”上下文。

> 结论先行：**“强结构编码可以减少/替代消息传递”本身不是新贡献。**  
> Eijkelboom et al. 2023 已直接研究；GRIT、CSA、EGT、MoSE、SPSE、GPM
> 已展示 no-MP 模型在 ZINC 等 benchmark 上竞争甚至 SOTA。  
> SSRC 唯一可能站得住的候选贡献不是“发现 no-MP 可行”，而是：
> (i) 对精确 \(E_r\) 做**任务相对的信息充分性证书**；
> (ii) 把 **information value** 与 **estimation/learnability gap** 分开的
> 受控实验设计；
> (iii) 在小样本真实分子任务上检验“精确结构充分 ≠ 统计充分”。
> 这三条中任何一条单独都可能被认为增量；组合才是赌注。

---

## 0. 检索与代码可用性（先记不确定性）

| 项 | 结果 |
|---|---|
| Eijkelboom 2023 代码 | PMLR 论文未给代码链接；截至 2026-10-18 未找到公开仓库（OpenReview 论坛存在，未发现附件代码）。复现需重实现。 |
| GPM / NGPM 代码 | `https://github.com/Zehong-Wang/GPM`（HTTP 200，2026-10-18） |
| GRIT 代码 | `https://github.com/LiamMa/GRIT`（HTTP 200） |
| CSA 代码 | `https://github.com/inria-thoth/csa`（HTTP 200） |
| EGT 代码 | `https://github.com/shamim-hussain/egt`（HTTP 200） |
| MoSE（同态计数）代码 | `https://github.com/linusbao/MoSE`（HTTP 200） |
| SPSE 代码 | 论文所称 GraphPKU 仓库 404；仅找到 `https://github.com/LouisBearing/Graph-SPSE-Encoding` 的最小路径计数实现。 |
| ESA+ | **未能唯一识别**。OpenAlex/arXiv/S2 检索 “ESA+ graph / ESA plus structural encoding” 未命中一个明确的图学习论文；不排除是用户口径中的别名/未发表工作。本文不伪造引用；若用户给出准确标题/作者，需补审。 |
| MoSE 名称冲突 | 至少有三篇不同 “MoSE”：① 同态计数结构编码（arXiv 2410.18676，ICML 2025，本文审计对象）；② 多模态知识图谱补全 MoSE（EMNLP 2022）；③ 子图专家 MoSE（arXiv 2509.09337，2025）。本文只审计①。 |

检索限制：OpenAlex/arXiv/S2 对 CSA/SPSE 的会议版本与 “ESA+” 有漏检；
Semantic Scholar API 在审计时返回 429；因此“未找到”只表示**本次检索未找到**，
不表示不存在。引用时以 arXiv/PMLR 页面为准。

---

## 1. 逐篇审计表

| # | 工作 /  venue / 代码 | 输入的结构信息 | 学习式交互对象 | “去 MP”消融移除了什么 | 参数量/读出/预算控制 | 已覆盖的子问题 | 什么结果会让我们沦为重复 |
|---|---|---|---|---|---|---|---|
| 1 | **Eijkelboom et al. 2023**, “Can strong structural encoding reduce the importance of Message Passing?”, PMLR v221:278–288. 无公开代码 | Random-walk 结构编码：RW-Concat（拼接 RW 返回概率/转移矩阵）与 RW-Tensor（与节点特征做张量积；严格更强） | GCN/GIN/SAGE 的 MP 层；稀疏 K=10/K=1 或 no MP；joint encoder/decoder | 去掉全部 MP 层（L=0），保留 RW 编码 + 同一 decoder | **有参数量控制**（Peptides Table 3：no-MP 433–57133；full MP L=1 689–162109，L=4 最高 477037）；同一读出；3 seeds；**无**显式 FLOP/训练预算匹配；**无** information-matched static/dynamic | 强编码能否降低 MP 重要性；Peptides-struct/func、ZINC；参数消融 | 若 SSRC 只说“强编码下 no-MP 可竞争/MP 重要性下降” → 完全重复。若只说“no-MP 在 ZINC 上差很多” → 也是该文已报告的 negative。 |
| 2 | **Wang et al. ICML 2025**, “Beyond Message Passing: Neural Graph Pattern Machine”, PMLR v267:65496–65517; arXiv 2501.18739; `Zehong-Wang/GPM` | 从随机游走采样 semantic/anonymous paths，形成 pattern tokens；可选位置编码 | 对 pattern tokens 做 self-attention；**无 MP** | 模型本身无 MP；与 MPNN/GT 基线比较，不是同一 backbone 的“去 MP”消融 | 标准 benchmark 协议；附录参数量；**无** information-matched static/dynamic；**无** 小样本 Lipophilicity | no-MP pattern learning；ZINC/ZINC-full；OOD；可解释性 | 若 SSRC 只说“pattern/path token + attention 可以不用 MP” → 重复。若把 GPM 当“关系计算必要性”的反例 → 它是强反例，不是缺口。 |
| 3 | **GRIT** ICML 2023, PMLR v202:23321–23337; arXiv 2305.17589; `LiamMa/GRIT` | RRWP（relative random walk positional encoding）+ degree scaler；从邻接矩阵导出 | 节点 token 的 global self-attention；**无 MP** | 模型本身无 MP；与 MPNN/GT 比较 | ZINC ~500K 参数；4 seeds；RRWP→RWSE/SPDPE 消融；**无** information-matched static/dynamic | no-MP SOTA；长程；位置编码设计 | 若 SSRC 只说“no-MP + 结构位置编码可以 SOTA” → 重复（GRIT ZINC 0.059）。若 SSRC 的“静态”只是 GRIT 的一个弱化版 → 会被审稿人要求直接比 GRIT。 |
| 4 | **CSA** arXiv 2304.10933 (2023); `inria-thoth/csa` | chromatic attention（边颜色/键型）、长程节点注意力、ring encoding | 全局 self-attention + 颜色/环偏置；**无 MP** | 模型本身无 MP；CSA vs CSA-rings；与 MPNN/GT 比较 | ZINC CSA 0.070±0.003，CSA-rings 0.056±0.002；标准协议；消融 color/ring；**无** information-matched | no-MP SOTA；环/边颜色；ZINC | 若 SSRC 的“结构编码”只是 ring/edge color 的变体 → 重复。 |
| 5 | **EGT** KDD 2022:655–665; arXiv 2108.03348; `shamim-hussain/egt` | 节点/边特征 + 截断到 16 hops 的距离矩阵；residual edge channels | 全局 self-attention；**无局部卷积/MP** | EGT 本身无 MP；EGT-Constrained/Simple 消融 | 100K/500K 参数列；ZINC 0.108±0.009 (500K), 0.143±0.011 (100K)；标准协议；**无** information-matched | 全局注意力替代图卷积；边通道；距离编码 | 若 SSRC 只说“全局注意力可替代 MP” → 重复。 |
| 6 | **MoSE（同态计数）** arXiv 2410.18676, ICML 2025; `linusbao/MoSE` | 小模式图的 homomorphism counts 作为节点结构编码；接入 GPS/GRIT | 结构编码是预计算特征；交互仍由 GPS（MP+attention）或 GRIT（no MP）完成 | GRIT+MoSE 是 no-MP 行；GRIT+RRWP 0.059±0.002 vs GRIT+MoSE 0.056±0.001 | 同一 backbone 替换编码；4 seeds；参数量；**无** information-matched static/dynamic | 结构编码设计；同态计数；no-MP SOTA；ZINC | 若 SSRC 的 \(E_r\) 只是“另一种结构编码”，且只报“+编码更好” → 重复。MoSE 已覆盖“固定模式计数 + no-MP”。 |
| 7 | **SPSE** arXiv 2502.09365 (2025); 官方代码未见 | simple path counts（按长度）作为节点/边结构编码，替代 RWSE；接入 GRIT/CSA/GPS | 编码是预计算路径计数；交互由 backbone 完成 | GRIT-SPSE 是 no-MP 行；GRIT-RWSE 0.065±0.005 vs GRIT-SPSE 0.059±0.001；CSA-SPSE 0.061±0.003 | 同一 backbone 替换编码；10 seeds 重训；参数量；**无** information-matched | 路径计数编码；no-MP SOTA；ZINC/Peptides | 若 SSRC 只说“path/subgraph 计数是强结构编码” → 重复。 |
| 8 | **GSN** arXiv 2006.09252 (TPAMI 2022) | 子结构计数（graphlet 等）+ 节点特征 | 子结构计数向量 + MPNN | 没有 no-MP 消融（计数 + MP） | ZINC 0.101±0.010（~500K）；标准协议 | 子结构计数提升表达力；与 MP 结合 | 若 SSRC 把“子结构计数 + MP”当新方法 → 重复；若 SSRC 的 \(E_r\) 只是 GSN 的计数特征 → 表示层面不新。 |

---

## 2. 已被完全覆盖的主张（不能声称新颖）

1. **“强结构编码可以减少消息传递的必要性。”**  
   Eijkelboom et al. 2023 的标题就是这个问题；Peptides-struct 上 no-MP
   RW-Tensor 0.249 vs full 0.212（差距很小），参数控制表也存在。  
   → SSRC 不能把这句话当作贡献。

2. **“no-MP 模型可以在 ZINC 上竞争/SOTA。”**  
   GRIT 0.059±0.002；CSA-rings 0.056±0.002；MoSE+GRIT 0.056±0.001；
   SPSE+GRIT 0.059±0.001；GPM 0.064±0.004。  
   → “MP 是必要的”在 ZINC 上已被强反例否定；“no-MP 可行”不是缺口。

3. **“结构编码（RW / homomorphism / path / ring / pattern）可以改善 no-MP。”**  
   GRIT (RRWP)、MoSE (homomorphism counts)、SPSE (simple paths)、
   CSA (rings/colors)、GPM (paths/patterns) 已覆盖多种编码家族。  
   → 只换一种编码并报性能提升，是增量工程，不足以支撑新主线。

4. **“参数控制下 no-MP 仍可竞争。”**  
   Eijkelboom Table 3 已在 Peptides 上给出 433–57133 参数的 no-MP 行，
   并与 689–162109 参数的 full MP 行对照。  
   → “没人控制参数量”不是事实；SSRC 不能这样立论。

5. **“全局注意力/一次性读出可以替代迭代 MP。”**  
   EGT、GRIT、CSA 都是全局注意力/无 MP；toy 的一次性距离 oracle 也说明
   对这个 toy 任务迭代不必要。  
   → SSRC 不能把“一次性全图读出可行”当作新发现。

---

## 3. 部分覆盖 / 仍可能的缺口

### 3.1 精确 \(E_r\) 的信息充分性证书

上述工作使用的是：
- RW 返回概率/转移矩阵（Eijkelboom、GRIT 的 RRWP）；
- homomorphism counts（MoSE）；
- simple path counts（SPSE）；
- rings/edge colors（CSA）；
- sampled paths/patterns（GPM）；
- substructure counts + MP（GSN）。

**没有一篇**使用“精确带属性 rooted-radius-\(r\) 诱导子图同构类型多重集”
做**碰撞证书**，并把碰撞与任务 label 冲突联系起来。  
但注意：\(E_r\) 本身是经典表示（WL subtree / graphlet / local fingerprint /
GSN 计数），所以“表示新”不成立；新的是**问法**：在什么 \(r\)、什么任务上，
这个精确表示已经信息充分/不充分？碰撞是否导致不可约误差？

**会被视为重复的风险**：如果审稿人认为 \(E_r\) 只是 WL subtree kernel 的
变体，且碰撞分析是显然的（局部编码看不到远处），那么这条也不够。必须把
“碰撞 → 任务相对下界 → 与可学习性分离”做成完整论证，而不是只报碰撞。

### 3.2 information value vs estimation gap 的受控分解

上述工作都是**端到端性能比较**：
- Eijkelboom 比较 full/sparse/no MP；
- GRIT/CSA/EGT 比较 no-MP 与 MPNN/GT；
- MoSE/SPSE 比较“换编码”；
- GPM 比较 no-MP 与基线。

**没有一篇**同时设置：
- \(S(E_r)\)：静态、非迭代、只看 \(E_r\)；
- \(D(E_r)\)：在 \(E_r\) 上迭代，信息相同；
- \(S(G)\)：信息充分/无损的静态 oracle；
- \(D(G)\)：完整图上的动态模型；

并用参数/FLOP/读出匹配来分解
\(R^\star(E_r)-R^\star(G)\) 与 estimation gap。  
这是 SSRC 最像“方法学贡献”的部分，但也是最容易被说“显然”的部分：
信息少当然可能更差；真正的科学问题在**有限样本下估计 gap 是否可测、
是否随 n 缩小**。pilot 必须把样本效率作为主端点，而不是只报一个平均 MAE。

### 3.3 小样本真实分子任务（Lipophilicity 3360）

上述工作的主战场是 ZINC-12K/Full、Peptides、PCQM4Mv2、OGB 等；
没有一篇在 Lipophilicity 的 3360/420/420 scaffold split 上做
“精确结构 vs 关系计算”的样本效率研究。  
但“没人做过 Lipophilicity”本身不是贡献；它只有在能回答一个**可推广的问题**
时才有价值。当前 train-only 诊断给出一个具体动机：精确 \(E_2/E_3\) 在
Lipophilicity 上 94.5% 单例，LOO 只比全局均值好 0.052 MAE，而粗 scaffold
分组好 0.249 MAE。这把问题从“信息是否缺失”推向“为什么精确结构充分
不能转化为统计充分”。这是一个比“MP 是否必要”更窄、更可测的问题。

### 3.4 “静态但关系”的对照

Eijkelboom 的 no-MP 仍有 joint encoder/decoder；GRIT/CSA/EGT 仍有全局
self-attention（本身就是关系读出）。它们不是“无序 bag-of-patches”的静态
基线。SSRC 若把 Static 做成一个弱 bag-of-keys，然后赢它，是 straw man。  
必须加入 \(S_{\rm patch}(E_r)\)：在 patch 内做共享 GNN，再池化；它有权重共享、
有局部关系，但没有跨 patch 的全图迭代。只有赢过这个公平对照，才能说
“迭代/全图关系”有额外价值。

---

## 4. 表示层面的先验艺术（必须引用，避免“新表示”误判）

| 先验 | 与 \(E_r\) 的关系 |
|---|---|
| WL subtree kernel（Shervashidze et al. 2011） | 多尺度 rooted subtree 多重集；\(E_r\) 是带属性、诱导子图、精确同构的局部版本。 |
| Graphlet kernel（Shervashidze et al. 2009） | 全局小子图计数；\(E_r\) 是 rooted/local 版本。 |
| GSN（Bouritsas et al. 2022） | 子结构计数向量 + MPNN；\(E_r\) 可视为一种精确 local substructure 特征。 |
| ECFP / Morgan fingerprints（Rogers & Hahn 2010） | 半径 \(r\) 的 circular atom environment 计数；与 \(E_r\) 的“局部环境多重集”最接近的化学先验。 |
| Local relational fingerprints / WL-style molecular features | 工业界常用；\(E_r\) 不是新表示。 |

因此：**SSRC 不能声称 \(E_r\) 是新表示**。如果论文的卖点是“精确 rooted patch
多重集”，会被直接归入 WL/graphlet/ECFP 谱系。卖点必须是“用这个表示做
信息充分性 vs 可学习性的受控分解”，并且要证明这个分解在真实任务上有
非平凡的结论。

---

## 5. 什么结果会让我们沦为重复 / 应该停止

### 5.1 直接重复（出现任一即不能作为新贡献）

- 只报告“强结构编码下 no-MP 与 MP 可比/更好” → Eijkelboom 已做。
- 只报告“no-MP 在 ZINC 上 SOTA/接近 SOTA” → GRIT/CSA/MoSE/SPSE/GPM 已做。
- 只报告“换一种结构编码（RW→path/homomorphism/ring）提升性能” →
  MoSE/SPSE/CSA 已做。
- 只报告“参数控制下 no-MP 仍可竞争” → Eijkelboom Table 3 已做。
- 只报告“精确局部编码在某个 toy 上看不到远处” → 这是显然的 \(2r\)
  边界；必须加上任务相对下界、真实数据碰撞/统计充分性、或受控 pilot。

### 5.2 会让 pilot 失去意义的可能结果

- \(S_{\rm patch}(E_r)\) 在所有 n 上都匹配 \(D(G)\)：则“关系计算改善
  可学习性”不成立；最多是一个 negative result。
- \(D(G)\) 的收益完全由 \(S(G)\)（无损静态 oracle）复现：则收益是信息，
  不是迭代；SSRC 的“关系计算”主张退化为“信息更多更好”，没有新意。
- \(D(E_r)\) 与 \(S_{\rm patch}(E_r)\) 无差异：则“在相同信息上迭代”无价值；
  只剩“全图信息”的价值，而这已被 Eijkelboom/GRIT 类工作覆盖。
- Lipophilicity 的效应小于 0.02 MAE 且 CI 跨 0：停止该任务上的表述。

### 5.3 仍然安全的负结果

如果 pilot 设计正确但结果是 null，仍然可以贡献：
- “在 Lipophilicity 3360 上，精确 \(E_r\) 的统计低效不能靠迭代关系计算
  修复；scaffold 级粗结构才是主要信号。”
- 这是一个**有边界的负结果**，比“又一个 no-MP 模型”更有信息量。
但它需要预注册、足够 power、以及公平的静态对照；否则只是“我们的模型没训好”。

---

## 6. 安全的新颖性声明（可直接放进 REPORT/汇报）

> 本路线**不**声称“强结构编码减少消息传递需求”是新的；Eijkelboom et al.
> (2023) 已直接研究，GRIT/CSA/EGT/MoSE/SPSE/GPM 已展示 no-MP 的竞争力。
> 本路线的候选贡献是：在一个**信息访问被显式控制**的设置中，把
> “\(E_r\) 是否信息充分（问题 A）”与“在有限样本/预算下是否可学（问题 B）”
> 分开；对精确 \(E_r\) 给出任务相对的碰撞证书；并在小样本 Lipophilicity 上
> 检验“精确结构充分”是否等于“统计充分”。若 pilot 显示动态全图模型
> 的收益完全由额外信息解释、或静态 patch 编码即可匹配，则本路线应停止，
> 不把 negative 结果包装成新方法。

---

## 7. 待用户澄清

1. **ESA+ 的准确标题/作者/venue**：当前检索无法唯一识别；若它是某篇
   具体论文，需补审后再判断是否已被覆盖。
2. 是否接受把主问题从“MP 是否必要”**重述**为“精确结构充分性 vs
   统计可学习性”；若不接受，Phase 0 的证据更支持 `STOP_THIS_FORMULATION`
   而不是继续 pilot。
3. 是否允许在 pilot 中使用 Lipophilicity 官方 train 的内部 locked 504 行
   作为主评估；官方 valid/test 已被旧轨消耗，不应再用于新选择。
