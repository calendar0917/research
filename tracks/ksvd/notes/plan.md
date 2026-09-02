# KSVD 轨 · 执行计划（冻结 v0.1 · 2026-07-24）

## 已定决策

| 项 | 决定 |
|----|------|
| 结构管道 | R2：偏置 RW → **诱导子图** → 邻接 pad→\(m\) →（后）KSVD |
| 采样旋钮 | node2vec \(p,q\) + 长度/\(\|S\|\) cap + 边降权 \(\gamma\)（RW-C0）；**不永久删边** |
| 下游对标 | 主：**ogbg-molhiv**（OGB scaffold，test AUC@best val）；辅 ZINC；TUD 仅文献对照 |
| CIN TUD 数字 | 与 GIN 同属 **Xu 乐观 10-fold**；不与 strict 混表 |
| 评估 | 先 L1–L4 过程指标，后 L5 下游；分 `protocol_id` |

## 阶段

| 阶段 | 内容 | 状态 |
|------|------|------|
| **0** | 采样器 + 覆盖/重复/规模指标；B0 vs B1 vs M0 | **骨架可跑**；合成图已出 JSON；注意 B0/M0 的 \|S\| 不同，冗余比用 cnt/sg + 同族 B1 vs M0 |
| **1** | 诱导邻接向量化 + KSVD 烟测；重构/原子使用 | 未开始 |
| **2a** | 小分子图级探路（可选 MUTAG 等，自有 protocol） | 部分（MUTAG 融合历史） |
| **2b** | molhiv 主对标（协议 `ogb-molhiv-v0`） | **骨架**；见 `notes/molhiv_phase.md` |
| **2c** | 池化消融 mean/max/attn | 脚本就绪 `run_pool_ablation` |
| **2d** | 双通道 GINE ‖ \(s_G\) | 骨架 `run_molhiv_dual` |
| **3** | 消融 \(p,q,\gamma\)、诱导开/关、字典大小 | 未开始 |

## 协议 ID（只增不改语义）

见 [protocol.md](protocol.md)。

## 入口命令（阶段 0）

```bash
cd tracks/ksvd
python -m code.run_stage0 --config configs/stage0.yaml
```

结果 JSON → `results/stage0/`。

---

# 路线重构草案 v0.2（2026-07-30）

> 本节是当前执行优先级；上面的 v0.1 保留为历史决策记录。  
> 核心调整：**先在纯结构任务上把 KSVD 的能力、失败边界和研究命题做清楚，再进入 CIN/MolHIV 性能主线。**

## 1. 当前研究目标

我们暂时不把问题定义成“如何继续给 MolHIV 上的 KSVD token 调参”，而拆成两个层次：

1. **KSVD 科学问题（组内前置主线）**  
   KSVD 能否从图数据中学习一个显式、共享、稳定、可解释/可还原的结构词汇？它真正保留的是局部内容、原型出现，还是原型之间的组合关系？
2. **分子性能问题（后续外部对标）**  
   在 KSVD 的能力边界明确后，能否把“学习结构词汇”与精确 higher-order lifting 结合，并在严格协议下接近或达到 CIN 在 ogbg-molhiv 上约 0.80 的量级？

现阶段优先回答第 1 个问题。MolHIV/CIN 是后续目标和压力测试，不再作为发现 KSVD 机制的主要试验场。

## 2. 暂停项

在纯结构结论出来前，暂停以下低信息增量：

- 继续扫随机游走、字典大小、top-k activation 等超参；
- 继续做模糊 prototype activation occurrence graph；
- 继续以 official test 结果选择方法；
- 直接做“KSVD + CIN”或给 CIN cycle cell 加一个简单 gate。

原因：现有实验已表明随机游走不是必要条件，模糊 occurrence 的增益弱且常不优于 shuffled control；这些修改不足以回答 KSVD 本身是否学到了可迁移的结构词汇。

## 3. Phase A：纯结构任务上的 KSVD 澄清

### 3.1 必须区分的四个假设

| 编号 | 可证伪假设 | 需要回答的问题 |
|---|---|---|
| H1 | **共享词汇可恢复** | 数据由有限个真实 motif 生成时，KSVD 是否能跨图恢复这些 motif，而不只是拟合方差最大的向量方向？ |
| H2 | **重构与任务相关** | 更低的 patch/graph 重构误差是否稳定对应更好的结构判别与 OOD 泛化？ |
| H3 | **原子语义稳定** | 不同 seed、训练子集和 graph-family split 下，匹配后的字典原子是否保持一致语义？ |
| H4 | **组合可表达** | 当两类图具有相同 motif multiset、只改变 motif 的位置和连接关系时，KSVD 表征能否区分？若不能，是否必须显式恢复 occurrence/incidence？ |

这里要避免把“分类准确率高”直接解释成“学到了结构词汇”。每个任务都应有已知生成因子或 ground-truth motif，才能直接评价词汇恢复。

### 3.2 数据集阶梯

#### S0：最小机制与感受野探针

目的：只验证实现和基本结构敏感性，不作为方法有效性的主要证据。

- 已有 C4/C8、distant-triangle、cube-vs-Möbius 等探针；
- 增加节点置换重复实验，确认同一图重编号后编码与预测不变；
- 控制节点数、边数、度序列等简单统计，避免分类器走捷径；
- 用 B0/R1/R2/确定性 path/RW 分离“感受野扩大”和“随机性”的作用。

S0 的结论只允许是：某种 patch 定义能否看到目标结构，不能据此声称学到了通用 vocabulary。

#### S1：已知词汇的生成式 motif-recovery 数据

构造一个我们知道真实答案的数据生成器：

1. 预先定义一组合法结构原型 \(\mathcal V^*=\{P_1,\ldots,P_K\}\)，例如 cycle、path、branch、clique-like block、带 attachment slots 的 junction；
2. 从相同或匹配的 backbone 出发，按控制分布插入这些原型；
3. 控制图规模、边数、度分布和 motif 频率，防止简单统计泄漏标签；
4. 生成多种任务：presence、count、rare-but-discriminative motif，以及 motif-free nuisance variation；
5. 保存每个原型在每张图中的精确 node/edge mapping，作为恢复评价真值。

核心不是只做分类，而是比较学得字典与 \(\mathcal V^*\) 的匹配质量。

#### S2：相同内容、不同组合的 composition 数据

构造两类图，使其具有：

- 相同的 motif 类型与数量；
- 尽量匹配的节点数、边数、度序列和局部 patch 分布；
- 不同的 motif–motif 距离、相对位置、连接顺序或共享边界。

依次比较：

1. bag of sparse codes；
2. 原子出现次数/能量；
3. 带位置但不带 identity 的 occurrence；
4. 带 prototype identity 的精确 occurrence；
5. 完整 incidence/boundary lifting。

这一层直接检验当前路线的核心瓶颈是否是“patch 内容存在，但 composition 丢失”。

#### S3：结构 OOD 与词汇稳定性

训练和测试不再 IID 随机切分，而使用结构环境切分：

- held-out backbone family；
- held-out graph size/density；
- held-out motif combination，但单个 motif 在训练中出现过；
- motif frequency shift；
- nuisance substructure shift。

目标是区分：

- dataset memorization；
- 单图 instance-adaptive 编码；
- 真正可跨环境复用的 dataset-level shared vocabulary。

#### S4：标准纯结构 benchmark（外部有效性）

在 S0–S3 的机制结论成立后，再选择少量公认的结构表达力/图分类 benchmark 做外部检查。标准 benchmark 只回答“结果能否迁移”，不能替代带 ground truth 的词汇恢复实验。

候选集需要另行核查数据协议、是否含节点属性、是否存在泄漏以及合适的 baseline；不先因为某个 benchmark 容易取得高分就将其定为主证据。

### 3.3 必须比较的 matched controls

| 类别 | 对照 |
|---|---|
| 简单结构统计 | node/edge count、degree histogram、component/cycle-rank 等 + 同容量 MLP |
| 固定显式特征 | WL histogram、预定义 graphlet/motif count |
| 非字典降维 | PCA/NMF 或同维随机投影 |
| 字典对照 | random dictionary、shuffled atoms、per-graph dictionary、shared dictionary |
| 隐式模型 | GIN/GCN（匹配容量和训练协议） |
| oracle | 使用真实 motif identity/count；使用真实 occurrence/incidence |
| 关系归因 | real boundary/connection 对比保数量与大小分布的 shuffled boundary/connection |

只有 shared KSVD 稳定优于简单统计、非字典降维和 shuffled controls，且接近 oracle，才说明“学习结构词汇”这一解释得到支持。

### 3.4 评价指标

#### 表征与恢复

- patch reconstruction error 与 graph-level reconstruction error 分开报告；
- ground-truth prototype matching（在允许置换/同构后匹配）；
- prototype coverage、purity、duplicate rate；
- occurrence/support recovery precision、recall、F1；
- atom 是否能投影/解码成合法图结构；
- 节点重编号前后的 representation consistency。

#### 稳定性

- dictionary seed stability；
- train subsample stability；
- environment/scaffold-like split stability；
- 匹配原子后的 activation/semantic consistency。

#### 下游

- IID 与 OOD accuracy/AUC；
- sample efficiency；
- reconstruction error 与 task error 的相关性；
- 性能—词汇大小—编码成本曲线。

### 3.5 Go / no-go 判据

| 观察 | 解释与下一步 |
|---|---|
| 重构好，但 prototype recovery 和分类都差 | 欧氏 KSVD objective 与结构语义错位；停止把 reconstruction 当核心论据 |
| prototype 可恢复，presence/count 任务好，但 composition 任务失败 | KSVD 可作为 vocabulary learner，但必须增加精确 occurrence/incidence；bag-of-patches 路线终止 |
| 不同 seed/environment 下原子无法匹配 | shared vocabulary/stability 命题不成立；普通 KSVD 不宜作为 headline |
| 只有加入监督后才恢复稀有判别 motif | 将方法准确定位为 task-adaptive dictionary，而不是纯无监督发现 |
| S1–S3 均通过，且明显优于 PCA/NMF/random dictionary | 才进入分子结构词汇与 higher-order lifting 阶段 |

### 3.6 Phase A 的最小交付物

1. 一个可控的 motif-composition 数据生成器，输出 graph、label、ground-truth prototypes 和 exact occurrence mappings；
2. 一个统一 evaluator，同时评价 reconstruction、prototype recovery、occurrence recovery、stability 和 classification；
3. 一张二维结论表：`是否恢复词汇 × 是否恢复组合关系`；
4. 一份明确结论：KSVD 最适合作为 vocabulary learner、初始化器、压缩器，还是应终止为核心方法。

## 4. Phase B：CIN 复现与收益拆解

Phase A 之后再开展，目的不是创新，而是建立可信上界和确定 MolHIV 的结构收益来源。

### 4.1 先完整复现

- 固定官方实现/commit 和隔离环境；
- 复现 CIN、CIN-small 和 matched GINE；
- 使用同一数据切分、特征、训练轮数和模型选择协议；
- 报告独立 runs 的 mean/std、参数量、训练时间和 cell 数；
- official test 不再用于方法选择，只用于最终冻结评估。

### 4.2 机制拆解

至少比较：

1. GINE；
2. ring flags/count only；
3. virtual ring nodes；
4. full CIN；
5. CIN 去 2-cell readout；
6. 保持 cycle 数量/大小分布、但打乱 cycle–bond boundary。

这一步回答 CIN 收益来自环先验、精确 incidence、新通信路径，还是容量/协议差异。

## 5. Phase C：共享、合法、可还原的高阶结构词汇

若 Phase A 支持“词汇可以学习”，且 Phase B 证明 exact lifting 有必要，则方法命题升级为：

> 学习一个 dataset-level shared、合法、可解码、跨结构环境稳定的高阶结构词汇，并恢复其在每张图中的 exact occurrence/incidence。

与 CIN 的区别不应只是“给人工 cycle 加 gate”，而应同时具有：

1. **shared discrete vocabulary**：跨图、跨 split 可匹配；
2. **legal and decodable prototypes**：原型带节点、边、attachment slots 和 boundary；
3. **exact lifting**：每个 occurrence 有明确 node/edge mapping；
4. **environment stability**：词汇在 scaffold-like environments 中语义一致；
5. **可证伪压缩目标**：用 20%–50% cells 保持接近 CIN 的性能，并优于 random/frequency pruning。

候选空间逐步减少人工先验：

1. cycles（先固定 CIN operator，只研究 selection/vocabulary）；
2. cycles + biconnected/fused-ring/junction fragments；
3. generic connected candidate subgraphs；
4. learned candidate generation。

KSVD 在这一阶段可保留为 prototype initialization、nearest-real-patch projection、压缩/残差度量和 matched baseline，但不预设它必须是最终 headline。

## 6. 总体成功标准与风险

### 6.1 MolHIV 目标

- 最终目标是严格协议下接近或达到约 0.80 test ROC-AUC 的量级；
- 这不是 Phase A 的晋级条件，也不能由当前约 0.002–0.003 的 occurrence 增益外推；
- 方法选择首先看 internal/scaffold validation 和多 seed 稳定性，最后只做冻结 test 评估。

### 6.2 创新性停止条件

出现以下情况时，不将其包装为新方法主线：

- learned selector 与 cycle size/frequency heuristic 等价；
- real boundary 不优于 shuffled boundary；
- 不同结构环境选择出完全不同、不可匹配的词汇；
- 只有自由 latent cell gate 有效，而合法离散 prototype 无效；
- 只在 MolHIV 上有很小提升，没有词汇恢复、稳定性、效率或机制贡献；
- 方法本质只是 CIN 的轻量复刻。

## 7. 当前最近一步

暂不写新的 MolHIV 模型。下一轮先完成 Phase A 的实验设计锁定：

1. 定义 ground-truth motif vocabulary 和 graph generator；
2. 定义 `content-only`、`composition-only`、`OOD` 三组任务；
3. 明确 KSVD 输入表示是否满足置换不变，以及“可还原”具体还原到什么对象；
4. 预注册 matched controls、指标和 go/no-go threshold；
5. 只在设计无明显捷径后开始实现。


### 7.1 执行状态（2026-07-30）

已启动真实纯结构数据路线，首轮使用不含节点/边属性的 `IMDB-BINARY` 与 `IMDB-MULTI`，同时比较 raw/cleaned 数据版本、B0/R2 patch；协议为 3 个 split seeds × 5-fold，字典与 PCA 均仅在训练折拟合。代码与结果：

- runner：`code/run_real_structure_ksvd.py`；
- report：`docs/REAL_STRUCTURE_KSVD_IMDB_3SEED_20260730.md`；
- raw result：`results/real_structure_ksvd_imdb_3seed_20260730.json`。

首轮观察：

1. **KSVD 尚未显示稳定独立优势。** 在 8 个 `dataset variant × patch` 设置中，KSVD content 相对 PCA/clustered-real-patch 的差值大多接近 0，且正负不一致。
2. **简单结构统计是必须保留的强对照。** IMDB-BINARY 上 degree/global-stat baseline 明显高于 KSVD；说明仅报告 KSVD 分类准确率会把数据集的规模/度分布信号误当作词汇学习能力。
3. **B0 普遍比 R2 更可靠。** 扩大到 R2 并未稳定提高结果，尤其在 IMDB-MULTI 上下降；再次不支持“更大感受野自然更好”。
4. **真实 relation 没有稳定优于 shuffled relation。** 个别设置为正，但跨 raw/cleaned、B0/R2 后方向不一致；当前 atom identity–position binding 仍不能作为正结论。
5. **重构误差不支持 KSVD 特殊性。** 当前 PCA 与 clustered real-patch dictionary 的 test reconstruction 普遍低于 3-iteration KSVD；分类排序也与重构排序不一致。
6. **字典有中等稳定性，但不等于语义稳定。** Hungarian-matched cosine 约处于中等偏高区间，真实 patch 字典多数更稳定；仍需检查 atom 对应的真实 patch/graphlet 语义。

因此下一步不是立刻扩大 KSVD 超参搜索，而是：

1. 加入 matched GIN/1-WL 与更严格的 degree-preserving controls；
2. 在真实数据上设计 structure-family/OOD split，而不仅是随机 CV；
3. 导出每个 KSVD atom 的 nearest real patches，检查是否对应稳定合法 graphlet；
4. 再扩展一个图规模更大的真实结构数据集，检验结论是否只属于 IMDB ego-network；
5. 只有出现“KSVD > raw/PCA/real-patch 且跨 split 稳定”的数据集后，才研究更复杂的词汇—occurrence 机制。

### 7.2 Phase A 真实数据阶段结论（2026-07-30）

已完成此前 7.1 中的 atom semantic audit、conditional gain、REDDIT-BINARY 跨规模复核，并补齐所有 content 表征共享同一 relation-graph block 的 matched controls。总报告：

- `docs/REAL_STRUCTURE_KSVD_GO_NO_GO_20260730.md`；
- conditional runner：`code/run_real_structure_conditional.py`；
- patch audit：`code/run_real_structure_patch_audit.py`；
- atom audit：`code/run_real_structure_atom_audit.py`。

核心结论：

1. **普通无约束 KSVD 不再作为当前 headline。** 在控制 size/degree 后，KSVD content 没有跨 IMDB/REDDIT 稳定优于 raw、PCA、clustered real-patch；唯一小幅领先 real+graph 的设置也不稳定或仍低于 relation-graph-only。
2. **自由 atom 的合法性问题在更丰富数据上更严重。** KSVD projectable proxy 在 IMDB-BINARY/MULTI 约为 0.55/0.50，在 REDDIT R2/max_nodes=24 仅为 0.158；real-patch dictionary 为 1.0，语义稳定性也不弱于 KSVD。
3. **sampler substrate 决定了“词汇”可能学到什么。** IMDB B0 patches 约 82% 是 clique；REDDIT B0 约 62% 是单边；REDDIT R2 在 max_nodes=12/24 时分别约 62%/48% 触顶。扩大尺度没有救回 KSVD-specific gain。
4. **true > shuffle 不能单独证明 composition。** true relation 虽常高于 shuffled relation，但相对 `content+graph` 在 IMDB-BINARY、IMDB-MULTI、REDDIT R2/24 分别下降约 0.019、0.036、0.041；REDDIT 为 0/15 folds 获胜。
5. **保留的正信号是 relation topology，不是 KSVD。** REDDIT 上 relation-graph-only 跨 B0、R2/12、R2/24 均稳定优于 stats，约 +0.012 到 +0.018；加入 KSVD content 后反而下降。

由此更新 go/no-go：

- no-go：继续扫描 unconstrained KSVD 的 atoms、T、RW、top-k，或直接将其 atom 称为 motif/graphlet/cell；
- go：KSVD 保留为 compression/initialization/baseline，以及合法 prototype 方法的候选生成器；
- 若组内必须对 KSVD 做最后一次机制测试，只允许一个有界实验：`clustered real-patch` vs `final-projected KSVD` vs `iterative projected KSVD`，全部 atom 投影为 distinct real training patches；若不能在至少两个结构差异明显的数据集上跨 3 seeds 稳定胜出，则 KSVD 正式降为 baseline。

在该 projected-KSVD 门槛通过前，仍不进入 MolHIV/CIN-style learned lifting。更大的主命题保留为：学习 shared、legal、decodable、cross-environment stable vocabulary，并恢复 exact occurrence/incidence。


### 7.3 Projected-KSVD 最终门槛结果（2026-07-30）

已完成预注册的最后一个 KSVD 机制实验：

- 实现：`code/projected_ksvd.py`；
- runner：`code/run_real_structure_projected_ksvd.py`；
- self-test：`code/test_projected_ksvd.py`；
- 总报告：`docs/REAL_STRUCTURE_PROJECTED_KSVD_20260730.md`。

方法对照为 clustered real-patch、final-projected KSVD、iterative-projected KSVD；projection 以 Hungarian matching 保证每个 atom 都是 outer-train 中真实、非零、signature-distinct patch。三个主设置均使用 5-fold × 3 split seeds、16 atoms、T=2、10 iterations 和 nested classifier selection。

结论：

1. 两种 projected KSVD 均达到 projectability=1，但没有在任何数据集上同时以 3/3 seeds 稳定优于 raw、PCA、clustered-real 三个 baseline。
2. IMDB-MULTI 上 iterative projected 有平均正信号（相对 real +0.0135、相对 raw +0.0210），但 seed=1 方向反转，因此只记为弱现象，不晋级。
3. IMDB-BINARY 上 projected content 低于 raw；REDDIT 上与 PCA 基本持平且低于 stats，relation-graph-only 继续占优。
4. projection 修复 legality 后仍未产生稳定 KSVD-specific task gain，说明失败不只是“自由 atom 不合法”。

**Phase A 最终决定：KSVD 主线关闭。** 普通、final-projected、iterative-projected KSVD 后续只作为 baseline/初始化器/候选生成器；不再扫描 atoms、T、RW、top-k 或 projection 变体。

下一阶段不应直接把 KSVD occurrence 包装成 CIN 替代品。研究主命题改为：从合法候选结构中学习 shared vocabulary/selector，恢复 exact occurrence/incidence，并以 cross-environment stability 与 matched boundary controls 证明其不同于人工 CIN 和普通 GNN。

### 7.4 不使用 WL 的 exact canonical 复核（2026-07-30）

针对“WL histogram 是否有损、是否应增加多维排序依据”的疑问，已完成一个不以 WL 作为最终结构特征的严格复核：

- exact canonical 实现：`code/vectorize.py::canonical_adjacency_features`；
- 置换不变与 WL 反例自测试：`code/test_canonical_vectorize.py`；
- 真实 patch collision/raw 审计：`docs/REAL_STRUCTURE_NONWL_AUDIT_20260730.md`；
- WL/canonical/rooted projected-KSVD 总结：`docs/NONWL_PROJECTED_KSVD_FINAL_20260730.md`。

表示对照：

1. `WL`：度、度对和 3 轮 1-WL histogram；
2. `canonical`：nauty 规范标号后的完整邻接上三角 + node mask，对无根 induced patch 在图同构意义下无损；
3. `rooted canonical`：把 sampler center 作为 singleton color，额外保留中心角色。

关键事实：

1. **WL 理论上确实有损。** triangular prism 与 `K3,3` 在当前 WL 表示下完全相同，而 exact canonical adjacency 可区分。
2. **但在本次三个真实 patch populations 中，WL 没有合并不同的 unrooted exact graph types。** IMDB-BINARY、IMDB-MULTI、REDDIT-BINARY 的 WL→unrooted canonical collision occurrence fraction 都为 0。因此此前结果不能主要归因于实际发生的 WL type collision。
3. **Reddit 存在 center-role 丢失。** 约 51% occurrences 落在包含多个 rooted types 的 unrooted bucket 中；但 rooted 表示并未带来稳定任务收益，说明“中心信息存在”不等于“中心信息对当前任务与欧氏 KSVD 有用”。
4. **完整信息不自动产生合适的 KSVD 几何。** canonical raw 在 IMDB-BINARY、IMDB-MULTI、REDDIT 上相对 WL raw 分别为 -0.0228、-0.0485、-0.0042。canonical adjacency 虽无损，但规范标号后的坐标欧氏距离未必反映小的结构编辑或任务相似性。
5. canonical `final-projected` 有两个局部信号：IMDB-MULTI 对自身 raw/PCA/real-patch 为 3/3 seeds 正向；REDDIT 相对 WL 对应 final-projected 为 3/3 seeds 正向、平均 +0.0100。但前者只发生在一个数据集且仍低于 WL iterative，后者未稳定超过自身 PCA/real-patch，并低于 relation-only=0.8550。
6. rooted canonical 没有通过任何数据集：IMDB-MULTI 的 canonical final-projected 信号降至 0.5095；Reddit rooted iterative=0.8417，仍低于 relation-only，且未对自身所有 baseline 达到 3/3 seeds 正向。

最终门槛仍为 **NO-GO**：WL、canonical、rooted canonical 的 final/iterative projected KSVD，没有任何同一方法在至少两个数据集上稳定胜过其自身 raw、PCA、clustered-real 三个 baseline。

由此冻结以下决定：

- 不再把“增加几维排序依据”作为主实验。多指标排序仍可能有 tie；若最终用 node id 解 tie，便失去置换不变性。exact canonical labeling 已经给出了比启发式多排序更严格的检验。
- 不把失败归因于 WL 表达力或 sampler center 丢失；主要矛盾更可能是 **Euclidean reconstruction geometry 与 task-relevant structural similarity 不对齐**。
- 若未来仍研究结构字典，下一步应改变相似度/对象，而不是继续改排序：对合法离散候选做 graph-kernel/edit-distance/learned invariant metric 下的 selection，或直接学习 candidate selector + exact occurrence/incidence。该路线已超出普通 KSVD，必须与 KSVD baseline 清楚区分。
- KSVD 主线关闭的结论保持不变；保留的可复用成果是 exact canonical 表示、合法 real-patch projection、collision audit，以及 Reddit 上稳定的 relation topology 信号。

