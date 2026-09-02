# KSVD 连续重叠 patch cover：完整探索结论（更新至 v2）

> 初版日期：2026-08-01；v2 更新：2026-08-02  
> 起点：`docs/luyin/luyin12.txt` 里提出的“patch 应当相邻、重叠，并能像图像 patch 一样连续拼接”问题。  
> 范围：只研究图本身、sampling/cover、stitching 与无标签压缩；不做分类。

## 1. 最终结论

前六轮实验建立了 continuous patch chain 的可行性与 ordering/decoder 边界；随后针对参数与 heuristic 的“任意性”做了 cover Pareto、局部 exhaustive oracle、scalable candidate beam、KSVD capacity、stitching/completion 和三 seed robustness 审计。最终答案是：

1. **连续 patch cover 可以构造。** 相邻 patch 能严格重叠固定数量节点，并保存精确 slot-to-slot correspondence；给定同一抽象 cover 后，节点重编号不改变 patch adjacency 或 transition maps。
2. **简单 sliding walk 不够。** 它保证连续性，却稳定降低 edge coverage。
3. **局部 frontier 也不够。** 它会在 modular/block graph 中滞留于单一社区。
4. **target-edge bridge 是有效起点，但不是最终 sampler。** 它显著优于局部 frontier；小图 exhaustive oracle 随后证明它仍有约 `0.138` edge-coverage gap。
5. **KSVD 有稳定但中等的压缩收益。** FINAL 相对 INIT 改善 held-out patch 与 stitched reconstruction，并优于同为 3 个系数的 PCA3；但 patch error reduction 只有 `5.4%`，未达到冻结的 `10%` 强 gate。
6. **不能把图像式 slot continuity 简单硬编码。** 强制共享节点保持同 slot 虽降低 overlap disagreement，却显著恶化 reconstruction、F1 和 edge recall。
7. **显式 transition map 在线性 decoder 中有弱信号，但没有 added value。** 正确 context 比图内错位 context 稍好，然而仍比只读当前 code 更差；CURRENT 对 BASE 的改善也近乎为零。
8. **Fiedler coordinate 是图内谱位置，不是可直接拿来排序 KSVD patch 的跨图绝对坐标。** 它降低重复 pair disagreement，却没有改善 reconstruction；少数高密度 block 图还出现谱坐标 ties，使 total rank 依赖 tie-breaking。
9. **旧 `s10/o5/m1.5` 配置确实任意且被支配。** 同观察成本下，约三分之一 overlap 明显优于一半 overlap。
10. **轻量 marginal candidate Beam8 修复了主要瓶颈。** 它保持单链、固定 overlap 和显式 correspondence，并把“下一 patch”从单目标边 heuristic 改为有限候选上的边际覆盖选择。
11. **主要收益来自 sampler，而不是更复杂 decoder。** Beam8 + `o3` 相对 target + `o3` 的 KSVD full RMSE 改善 `19.52%`；slot weighting 几乎无效，train-only structural completion 只有较小但可重复的附加收益。

因此当前推荐的 v2 方法对象是：

> **target-seeded marginal candidate Beam8 + `s=10/o=3/m=1.5` + construction ordering + explicit overlap correspondence + optional `K24/T3` sparse KSVD + optional train-only structural completion**。

其中 correspondence 仍是应保留的数据结构。v2 可以作为当前 synthetic cover/sampler 的冻结方案；identity/rate 审计证明 ordered KSVD chain 不能直接作为编号无关表示或 bit-level codec。后续 masked-relation 审计则进一步证明：真实 overlap/distance binding 有强 held-out signal，ID-free local token 可以同时保留 relation 增益和 relabel stability。它仍不是已经成立的分类模型。

## 2. 第一轮：连续性本身

设置：50 nodes、平均度约 15/20/25，regular/small-world/block 三个图族，`s=10`、overlap `=5`，动态 matched budget。

| method | edge cover | pair cover | overlap | Jaccard gap |
|---|---:|---:|---:|---:|
| independent walk | 0.5879 | 0.4652 | 1.99 | -0.0015 |
| sliding walk | 0.5176 | 0.4317 | 5.00 | 0.2504 |
| local frontier | 0.5977 | 0.4289 | 5.00 | 0.1363 |

判定：`FAIL_SMALL_PATCH_GRAPH_RECOVERY`。

关键不是 overlap 失败，而是 edge coverage 没有获得足够收益。分图族后发现 local frontier 在 regular/small-world 为正，在 block 图上随密度增加而恶化。

## 3. 第二轮：跨区域 target-edge bridge

保持数据、预算、patch size 和 overlap 不变，只修改调度。

| method | edge cover | block edge cover | pair cover | continuous fraction | connected |
|---|---:|---:|---:|---:|---:|
| frontier | 0.5979 | 0.5838 | 0.4273 | 1.000 | 0.862 |
| single target bridge | 0.6889 | 0.6846 | 0.4928 | 1.000 | 1.000 |
| multi-chain target | 0.7152 | 0.7073 | 0.5142 | 0.762 | 1.000 |

单链相对 frontier：

- overall edge `+0.0910`；
- block edge `+0.1008`；
- pair coverage `+0.0655`；
- 3/3 seeds 全部通过 frozen gate。

判定：`PASS_SINGLE_CHAIN_TARGET_BRIDGE`。

multi-chain 上限更高，但单链已经通过，因此该轮当时保留更简单、与导师问题更贴近的 single chain。后续 v2 仍保持单链，只替换 next-patch 选择规则。

## 4. 第三轮：KSVD stitched reconstruction

3-fold graph isolation；`K=24,T=3,updates=25`；比较 RAW、INIT、FINAL、PCA3。

| stage | patch error | observed RMSE | observed F1 | full edge recall | disagreement |
|---|---:|---:|---:|---:|---:|
| RAW | 0.0000 | 0.0000 | 1.0000 | 0.6845 | 0.0000 |
| INIT | 0.5163 | 0.3858 | 0.8214 | 0.5783 | 0.1427 |
| FINAL | 0.4884 | 0.3627 | 0.8411 | 0.5917 | 0.1395 |
| PCA3 | 0.5572 | 0.4175 | 0.7695 | 0.5521 | 0.1214 |

证据：

- FINAL patch error 3/3 folds 低于 INIT；mean reduction `5.4%`；
- stitched observed RMSE 3/3 folds 降低；mean reduction `6.0%`；
- FINAL F1、full edge recall、disagreement 全部优于 INIT；
- FINAL observed RMSE 明显低于 PCA3；
- 24/24 nondead，maximum activation share约 0.05。

正式判定仍为 `FAIL_KSVD_COVER_COMPRESSION`，因为预注册要求 patch reduction >=10%。不能事后放宽 gate 改判 PASS。

准确解释是：

> KSVD 是健康、优于 PCA3、能改善 stitching 的 sparse compressor，但更新幅度没有达到本轮预定义的强效果量。

## 5. 第四轮：slot-persistent ordering

只重排同一批 node sets，让 shared nodes 在相邻 patch 中保持相同 slots。所有 coverage、RAW 和 mapped replay invariants 完全一致。

| ordering | FINAL patch error | observed RMSE | observed F1 | edge recall | disagreement |
|---|---:|---:|---:|---:|---:|
| construction | 0.4884 | 0.3627 | 0.8411 | 0.5917 | 0.1395 |
| persistent slot | 0.5411 | 0.4053 | 0.7956 | 0.5757 | 0.1243 |

persistent ordering：

- disagreement 改善 `10.9%`；
- observed RMSE 恶化 `11.7%`；
- patch INIT→FINAL reduction 只剩 `2.1%`；
- F1 和 full edge recall 下降。

判定：`REJECT_SLOT_PERSISTENT_ORDERING`。

这说明同一 slot 保持节点身份连续，会把不同时间步中的不同结构角色混入相同线性坐标。图像式“像素位置连续”不能直接照搬到一般图。

## 6. 第五轮：explicit transition-aware decoder

冻结 construction-order FINAL KSVD codes，比较标准 dictionary reconstruction、只读当前 code 的 ridge、读取正确 previous context 的 ridge，以及单图内部循环错位 context 的同容量对照。

| stage | patch error | observed RMSE | observed F1 | full edge recall | disagreement |
|---|---:|---:|---:|---:|---:|
| BASE FINAL | 0.4884 | 0.3627 | 0.8411 | 0.5917 | 0.1395 |
| CURRENT ONLY | 0.4883 | 0.3625 | 0.8410 | 0.5909 | 0.1398 |
| TRUE TRANSITION | 0.4927 | 0.3670 | 0.8359 | 0.5869 | 0.1381 |
| SHUFFLED TRANSITION | 0.5015 | 0.3707 | 0.8320 | 0.5837 | 0.1498 |

结果：

- CURRENT 相对 BASE observed RMSE 只改善 `0.06%`，0/3 folds 达到 2%；
- TRUE 相对 CURRENT 恶化 `1.23%`，observed F1 与 recall 也下降；
- TRUE 相对 SHUFFLED 改善 `1.00%`，说明正确 binding 有弱方向性信号，但低于冻结的 2% gate；
- TRUE 的 disagreement 略低于 CURRENT，仍不足以抵消绝对 reconstruction 退化；
- RAW、train/test isolation、78D feature shape、row count、finite value 和图内 shuffle invariants 全部通过。

判定：`REJECT_LINEAR_TRANSITION_CONTEXT`。

这排除了一个具体命题：把 previous code、transported degree 和 slot map 一次性拼接进线性 ridge，并不能把 correspondence 转化为 held-out reconstruction 增益。它不排除带共享参数、残差约束或非线性聚合的 message passing，但继续扩模型需要比“transition map 存在”更强的机制假设。

## 7. 第六轮：Laplacian graph-global ordering

为正面回答“Laplacian 是否能给节点绝对位置”，对每张完整图计算 symmetric normalized Laplacian 的 Fiedler vector，做 sign canonicalization 和全图 node rank，再用该 rank 纯重排同一批 patch node sets。

首先需要限定“位置”的含义：Fiedler coordinate 是**单张图内部**的全局谱坐标。不同图的谱轴没有天然共同方向；符号可以翻转，重复/近重复 eigenvalue 下 eigenspace 可以旋转，结构对称节点还可能得到相同坐标。

坐标审计：

- 72 图一次 connected degree-preserving edge swap 后，canonical pair-order agreement 为 `0.8593`；
- 忽略谱轴正负翻转后 agreement 为 `0.9599`；
- 但 5 张高密度 block 图仅做 node relabeling 后，Fiedler total rank 就不能完全 replay；
- post-result diagnostic 确认这些失败图存在 exact/near coordinate ties：谱坐标能区分大区域，却不能无歧义地区分所有区域内节点。

Matched KSVD 结果：

| ordering | FINAL patch error | observed RMSE | observed F1 | edge recall | disagreement |
|---|---:|---:|---:|---:|---:|
| construction | 0.4884 | 0.3627 | 0.8411 | 0.5917 | 0.1395 |
| Fiedler global rank | 0.4924 | 0.3651 | 0.8380 | 0.6012 | 0.1181 |

Fiedler ordering 降低 disagreement、提高一点 full edge recall，但 observed RMSE 恶化 `0.64%`，F1 和 patch error 也变差。正式判定优先停在 `FAIL_LAPLACIAN_ORDERING_INVARIANTS`；即使忽略 tie failure，comparative reconstruction gate 仍不通过。

因此本轮否定的是：

> 把一个 sign-canonicalized Fiedler rank 当作 total node order，不能为当前 KSVD patch 提供有效的“绝对坐标”。

它不否定现代 GNN/Transformer 中的 LapPE。后者通常把若干 eigenvectors 当作节点特征，并用 random sign、SignNet、eigenspace-aware 或 invariant spectral features 处理歧义，不要求把所有节点强行排成唯一顺序。

## 8. 第七轮：任意性、error budget 与 marginal Beam8 v2

前六轮回答了“连续链能否成立”以及 ordering/transition/Laplacian 是否能修复坐标问题，但还没有回答旧配置本身是否只是手工选择。2026-08-02 的审计把 cover、compression、stitching 和 unseen completion 分开冻结比较。

### 8.1 Cover 与 KSVD capacity

27 个 `s/o/m` cells 全部 feasible。旧 `s10/o5/m1.5` 在相同 patch 数和近似 raw pair slots 下，被 `s10/o3/m1.5` 同时改善：

| cell | edge cover | pair cover | RAW full RMSE |
|---|---:|---:|---:|
| target `s10/o5/m1.5` | 0.6870 | 0.4925 | 0.3530 |
| target `s10/o3/m1.5` | 0.7629 | 0.5599 | 0.3064 |

所以 50% overlap 不再保留。`K24/T3/updates25` 则没有被更低或相同 dictionary/code 成本的配置支配；提高到 `T4` 或 `K32/T4` 可以继续降误差，但明确增加容量。25→50 updates 只有约 `0.07%` observed-RMSE 收益。

### 8.2 Target heuristic 的 oracle gap

18-node 小图上逐步 exhaustive 枚举所有 exact-overlap、connected next patches：

| sampler | edge cover | pair cover |
|---|---:|---:|
| target-edge | 0.7358 | 0.4420 |
| exhaustive one-step marginal | 0.8737 | 0.4458 |

24/24 graphs 的 exhaustive edge coverage 更高，平均 gap `+0.1379`。这证明 target-edge bridge 虽然是正确的跨区域机制，却没有高效使用整张 next patch 的容量。

### 8.3 Scalable marginal candidate Beam8

v2 保留 target-edge 作为第一 patch seed。之后枚举 previous patch 的 connected retained subsets，以通向未覆盖边的 boundary potential 筛出 8 个候选；每个候选做一次 greedy frontier fill，再按以下 lexicographic objective 选择完整 next patch：

```text
new edges → new pairs → new nodes → induced edges
```

固定 `s10/o3/m1.5` 的 sensitivity：

| cell | sec/graph | edge cover | pair cover | RAW full RMSE |
|---|---:|---:|---:|---:|
| TARGET | 0.025 | 0.7650 | 0.5605 | 0.3054 |
| B8_R1 | 0.279 | 0.8734 | 0.5288 | 0.2205 |
| B16_R1 | 0.505 | 0.8803 | 0.5298 | 0.2139 |
| B32_R2 | 1.849 | 0.8845 | 0.5309 | 0.2101 |

预注册 knee rule 选择 `B8_R1`：它取得大部分可得 edge/RMSE 收益，同时比 B32/R2 快约 6.6 倍。三个 cover seeds 的 edge gain 为 `+0.1083/+0.1085/+0.1092`，RAW RMSE reduction 为 `27.81%/28.03%/28.17%`，所有 exact-overlap、connected、single-chain invariants 均通过。

Beam8 的明确代价是 pair coverage 下降约 `0.032`；它更偏向观察真实边，而不是广泛观察非边 pairs。

### 8.4 Compression 与 end-to-end error budget

相同 patch count、相同 `K24/T3/u25`：

| branch | patch error | observed RMSE | full RMSE | observed F1 | full recall/F1 |
|---|---:|---:|---:|---:|---:|
| old target o5 | 0.4928 | 0.3653 | 0.4367 | 0.8387 | 0.5909/0.6854 |
| target o3 | 0.5021 | 0.3760 | 0.4162 | 0.8211 | 0.6442/0.7127 |
| Beam8 o3 | **0.4319** | **0.3489** | **0.3349** | **0.8780** | **0.8137/0.8224** |

Beam8 相对 target o3 的 observed RMSE 改善 `7.20%`、full RMSE 改善 `19.52%`。判定为 `ADOPT_MARGINAL_BEAM_COVER_V2`。

slot reliability weighting 在 Beam8 上只改善约 `0.03%`，不保留。可选 train-only structural ridge completion 把 Beam8 + KSVD 的 full RMSE 从 `0.3349` 降至 `0.3241`；最终 recall/F1 为 `0.8138/0.8224`。相对旧 target-chain + KSVD + completion 的 `0.3845`，end-to-end full RMSE 改善约 `15.7%`。

因此 error budget 的主结论是：

> 先让连续 sampler 选择信息密度更高的 patches，比在旧 cover 上继续增加 ordering、decoder 或 stitching heuristic 更有效。

## 9. 对录音问题的直接回答

导师提出的两个要求需要分开：

### 9.1 patch 是否能连续、相邻、重叠

可以。v1 的 target-edge bridge 已经证明可执行，v2 的 Beam8 在保留相同结构约束的同时进一步提高覆盖效率：

- 前后 patch 固定 overlap；
- transition map 精确记录共享节点对应；
- candidate beam 使链主动进入低覆盖区域，并按边际覆盖选择下一 patch；
- 不是随机散落的 patch bag。

### 9.2 是否能像图像一样共享同一套局部坐标

当前证据是否定的。强制 slot persistence 提高一致性，却损害线性重构几何。应该保留 transition map，让后续模型显式使用对应关系，而不是把它压扁成唯一排序。

第五轮进一步限定了这个结论：**保留 transition map 是必要的数据表达选择，但简单线性读取它还不够。**

第六轮又排除了另一种捷径：**单个 Fiedler vector 可以描述图内大尺度方向，但不能自动产生跨图稳定、无 tie 的唯一节点坐标。**

## 10. 下一步

### 收到导师 50-node 数据后优先做

不看 labels，直接复用冻结审计：

1. 数据是否连通、是否共享 node identity；
2. 冻结 Beam8 v2 的 node/edge/pair coverage，并以 target sampler 为历史 baseline；
3. candidate-search time、patch count、overlap 和 patch connectivity；
4. RAW stitching ceiling；
5. construction-order INIT/FINAL/PCA3 held-out compression。

如果 50 个节点在不同图中具有真实共享身份，必须额外加入 whole aligned-adjacency baseline；节点数相同本身不等于身份对齐。

### 在合成路线继续时

当前 synthetic compression/reconstruction 探索已经可以冻结，不再尝试更多手工 ordering，也不继续堆叠线性 context feature。若确有新的下游目标，下一命题必须更窄：

> 固定 BASE_FINAL 为主预测，只学习一个受限 residual；该 residual 必须利用 shared-pair consistency，并以 identity/zero-residual 初始化，避免重写当前 patch 已经可靠的部分。

这已经不再是 vanilla KSVD。KSVD 应定位为 patch compressor/initializer；只有 residual 在 held-out 图上同时优于 CURRENT 与 shuffled context，才能把 transition model 升格为有效组合机制。考虑到本轮线性结果为负，优先级仍低于收到真实 50-node 数据后的 frozen audit。

若专门继续 Laplacian 路线，则不应再把 eigenvector 变成 total ordering；应把 sign/eigenspace-invariant spectral quantity 作为额外节点或 pair feature，并与不含 spectral feature 的同容量模型比较。这回答的是“谱结构特征是否有用”，不再声称它提供唯一绝对坐标。

## 11. 主要产物

- 第一轮协议/结果：`KSVD_OVERLAP_COVER_PROTOCOL_20260801.md`、`OVERLAP_COVER_AUDIT_20260801.md`；
- 第二轮协议/结果：`KSVD_TARGET_EDGE_BRIDGE_PROTOCOL_20260801.md`、`TARGET_EDGE_BRIDGE_AUDIT_20260801.md`；
- 第三轮协议/结果：`KSVD_STITCHED_RECONSTRUCTION_PROTOCOL_20260801.md`、`KSVD_STITCHED_RECONSTRUCTION_AUDIT_20260801.md`；
- 第四轮协议/结果：`KSVD_SLOT_PERSISTENT_ORDERING_PROTOCOL_20260801.md`、`SLOT_PERSISTENT_ORDERING_AUDIT_20260801.md`；
- 第五轮协议/结果：`KSVD_TRANSITION_DECODER_PROTOCOL_20260801.md`、`TRANSITION_DECODER_AUDIT_20260801.md`；
- 第六轮协议/结果：`KSVD_LAPLACIAN_GLOBAL_ORDERING_PROTOCOL_20260801.md`、`LAPLACIAN_GLOBAL_ORDERING_AUDIT_20260801.md`；
- 核心实现：`overlap_cover.py`、`overlap_stitching.py`；
- transition 实现：`transition_decoder.py`、`run_transition_decoder_audit.py`；
- spectral 实现：`laplacian_ordering.py`、`run_laplacian_global_ordering_audit.py`；
- v2 总协议/结果：`KSVD_PATCH_CHAIN_ARBITRARINESS_ERROR_BUDGET_PROTOCOL_20260802.md`、`PATCH_CHAIN_ARBITRARINESS_ERROR_BUDGET_AUDIT_20260802.md`；
- v2 sampler：`marginal_candidate_cover.py`、`run_marginal_candidate_beam_audit.py`、`run_marginal_beam_sensitivity_audit.py`、`run_beam8_multiseed_audit.py`；
- v2 compression/completion：`run_marginal_beam_compression_audit.py`、`run_stitch_completion_audit.py`；
- runners/tests：`run_*cover*audit.py`、`run_*stitched*.py`、`test_overlap_cover.py`、`test_overlap_stitching.py`、`test_transition_decoder.py`、`test_laplacian_ordering.py`、`test_marginal_coverage_cover.py`、`test_marginal_candidate_cover.py`。

## 12. 第八轮：节点编号与完整 payload 反向审计

此前的 relabel test 是 mapped replay：把已经生成的 cover 随 node permutation 一起搬运。它能验证 patch adjacency 和 transition map 的映射实现，但不能证明对重编号后的图重新运行 sampler 会产生同一个 substrate。

18 graphs × 3 cover seeds 的正式 relabel-resampling：

| metric | mean |
|---|---:|
| mapped replay patch/transition match | 1.0000 / 1.0000 |
| resampled exact ordered-chain match | 0.0000 |
| resampled patch-set Jaccard | 0.1916 |
| local-vector row match | 0.0345 |
| transition-map match | 0.0303 |
| absolute edge/pair coverage delta | 0.0070 / 0.0058 |
| absolute RAW full-RMSE delta | 0.0065 |

因此 Beam8 的 reconstruction quality 对重编号扰动稳定，但具体 chain、slots 和 transition substrate 不是严格 relabel-equivariant。

把恢复 labeled adjacency 所需的 global identity sidecar 计入后：

| payload | mean bits / graph |
|---|---:|
| direct upper-triangle bitset | 1225.0 |
| direct enumerative exact code | 1177.2 |
| optimistic patch identity estimate | 812.3 |
| RAW unique pairs + identity estimate | 1464.4 |

在先支付 identity estimate 和 5-bit atom indices 后，KSVD 若要不超过 1225 bits，每 coefficient 平均只剩 `3.43` bits；这还没有计算 dictionary、quantizer、completion model 或 framing。

密度拆分很重要：degree 15 的 RAW estimate 为 `1187.6` bits，接近 bitset break-even；degree 20/25 则上升到 `1483.2/1722.4` bits。当前 matched budget 随 edge density 增长，identity sidecar 也随 patch count 快速增长。

判定：`REVISE_PATCH_CHAIN_REPRESENTATION_CLAIM`。

所以最终对象必须拆开：

1. Beam8 保留为完整已知图上的结构引导 patch extractor；
2. labeled graph compression 必须做真实 quantized codec 与 whole-adjacency rate-distortion baseline；
3. ID-free structural representation 必须重新设计 sampler/patch token/pooling，并使用重新采样后的 permutation consistency，而不是 mapped replay。

本轮协议/结果：`KSVD_PATCH_CHAIN_IDENTITY_RATE_PROTOCOL_20260802.md`、`PATCH_CHAIN_IDENTITY_RATE_AUDIT_20260802.md`。

## 13. 第九轮：patch relations 到稳定 graph representation

为回答“当前采样得到的 patches 如何变成稳定有效表示”，冻结 Beam8 covers、3-fold graph isolation 和低容量 ridge，做 masked-patch prediction。target patch 的 raw/code 完全隐藏，监督目标为 22D permutation-invariant local structure。

第一轮使用 raw/KSVD tokens：

| branch | masked RMSE | relabel+resampling cosine |
|---|---:|---:|
| RAW_BAG | 0.22443 | 0.9908 |
| KSVD_BAG | 0.18916 | 0.6953 |
| KSVD_TRUE_RELATION | **0.11869** | 0.7221 |
| KSVD_SHUFFLED_RELATION | 0.13980 | 0.7214 |

TRUE relation 相对 KSVD bag 改善 `37.25%`，相对 shuffled binding 改善 `15.10%`，3/3 folds 同时通过。判定：`PATCH_RELATIONS_ADD_MASKED_VALUE`。

这证明 overlap/distance binding 不是装饰性 side information；正确 patch-to-patch binding 含有可泛化结构信号。但 KSVD graph embedding 的 relabel stability 明显不够。

跟进把 local token 换成不读取 global IDs 或 construction slots 的 invariant descriptor：

| branch | masked RMSE | relabel+resampling cosine |
|---|---:|---:|
| INVARIANT_BAG | 0.16831 | 0.9988 |
| INVARIANT_TRUE_RELATION | **0.12474** | **0.9983** |
| INVARIANT_SHUFFLED_RELATION | 0.14263 | 0.9982 |

TRUE relation 相对 invariant bag 改善 `25.89%`，相对 shuffled 改善 `12.55%`，3/3 folds 与 stability gate 全部通过。判定：`ID_FREE_PATCH_RELATION_SUBSTRATE_SUPPORTED`。

因此当前最准确的 representation 结论是：

> Beam8 负责产生局部重叠 patches；ID-free local token 负责消除 numeric-ID/slot 语义；overlap 与 center-distance 负责 patch-level relative position。三者组合可以作为 relation-aware Transformer/GNN 的输入 substrate。

KSVD_TRUE 的 masked RMSE仍比 invariant TRUE 低约 `4.85%`，说明它保留额外信息；但 stability 明显更差。因此 KSVD 暂时降为需要 consistency/invariant projection 的 auxiliary token，而不是唯一主 token。

本轮协议/结果：`KSVD_PATCH_RELATION_REPRESENTATION_PROTOCOL_20260802.md`、`PATCH_RELATION_REPRESENTATION_AUDIT_20260802.md`、`KSVD_INVARIANT_PATCH_RELATION_FOLLOWUP_PROTOCOL_20260802.md`、`INVARIANT_PATCH_RELATION_FOLLOWUP_20260802.md`。

## 14. 第十轮：KSVD 的 node-ID / local-slot 分解与修正

用户提出“KSVD 问题是否来自采样时用 node ID 排序”。matched decomposition 给出的答案是：**方向正确，但主因应表述为 arbitrary local-slot coordinates，而不是 global identity 本身。**

| intervention | graph cosine | patch-vector match | patch-code cosine | support Jaccard |
|---|---:|---:|---:|---:|
| mapped global relabel | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| frozen patch-set slot shuffle | 0.6359 | 0.0372 | 0.1225 | 0.1253 |
| frozen patch-set ID sort | 0.6446 | 0.0353 | 0.1449 | 0.1390 |
| patch sequence shuffle | 1.0000 | — | — | — |
| relabel + resample | 0.6718 | — | — | — |

只要重编号时把原 local order 一起搬运，KSVD 完全 replay；而同一 patch node set 只改变局部排列就会使45D adjacency vector 和 sparse code 大幅改变。按 numeric ID 重排并没有改善，因为 numeric ID 并不提供跨图稳定的结构坐标。patch sequence 对 bag pooling 则没有影响。

据此不再在 ordered adjacency slots 上继续优化 KSVD，而是先映射到22D ID-free patch space（sorted normalized degrees、sorted normalized adjacency eigenvalues、density、triangle density），再学习 sparse dictionary。

容量网格结果：

| cell | TRUE RMSE | vs raw invariant | vs BAG | vs SHUFFLED | relabel cosine | decision |
|---|---:|---:|---:|---:|---:|---|
| K16/T3 | **0.09945** | **-20.27%** | 29.63% | 13.14% | 0.9673 | adopt |
| K16/T4 | 0.10644 | -14.67% | 35.55% | 13.38% | 0.9722 | adopt |
| K24/T3 | 0.13625 | +9.23% | 30.53% | 12.31% | 0.9506 | too lossy |
| K24/T4 | 0.14758 | +18.32% | 32.16% | 14.36% | 0.9628 | too lossy |
| K32/T3 | 0.16162 | +29.57% | 29.96% | 13.07% | 0.9462 | too lossy |
| K32/T4 | 0.15776 | +26.48% | 29.81% | 13.47% | 0.9534 | too lossy |

冻结选择为 `K16/T3`：字典352 scalars，平均每图约52.96个非零 code scalars，16/16 atoms 有效，3/3 folds 同时通过。值得注意的是，更大 K/T 的 patch reconstruction error 更低，但 masked downstream RMSE 更高；因此当前 KSVD 的有效作用是低容量结构去噪/正则化，不是最大化局部重构保真度。

剩余边界：`0.9673` 仍不是严格 permutation invariance。token 本身已经不读取 numeric ID，剩余差异主要来自 Beam8 在 relabel 后可能重新选出不同 patch sets。若要求严格等变，下一步应优化 sampler canonicalization 或用多次重采样的一致性训练，而不是重新引入 node-ID ordering。

本轮协议/结果：`KSVD_ID_SLOT_INSTABILITY_DECOMPOSITION_PROTOCOL_20260802.md`、`KSVD_ID_SLOT_DECOMPOSITION_AUDIT_20260802.md`、`KSVD_INVARIANT_SPACE_TOKEN_PROTOCOL_20260802.md`、`INVARIANT_SPACE_KSVD_TOKEN_AUDIT_20260802.md`、`KSVD_INVARIANT_SPACE_CAPACITY_PROTOCOL_20260802.md`、`INVARIANT_SPACE_KSVD_CAPACITY_AUDIT_20260802.md`。

## 15. 第十一轮：保留完整45D邻接的 structural canonical slots

第十轮的 invariant-space 修正绕开了 local slots，但用户原始问题是：能否保留完整 patch adjacency，只把 slot ordering 从 numeric ID 换成结构规则。本轮直接回答该问题。

冻结相同 Beam8 patch node sets、coverage、folds 与 `K24/T3/u25`，比较 construction、ID sort、center/distance/degree/triangle/WL signature、exact canonical、rooted canonical 与 overlap-anchored canonical。

三组 graph/cover seeds 均选择 `ROOTED_CANONICAL`：patch center 是 singleton color，其余节点按完整 induced adjacency 做 exact canonical labeling。

| branch | frozen vector/code/pooled | mean observed RMSE | mean full RMSE |
|---|---:|---:|---:|
| construction | mapped replay 1/1/1 | 0.34842 | 0.33422 |
| ID sort | 约0.031/0.121/0.645 | 0.36992 | 0.34594 |
| signature | 约0.914/0.950/0.979 | 0.31661 | 0.31696 |
| canonical | 1/1/1 | 0.31682 | 0.31694 |
| rooted canonical | **1/1/1** | **0.31248** | **0.31477** |
| overlap canonical | 约0.996/0.997/0.999 | 0.33505 | 0.32680 |

rooted canonical 相对 construction 的三种子 observed-RMSE 改善为 `9.91%/10.90%/10.14%`，full-RMSE 改善为 `5.57%/6.24%/5.66%`。首个 seed 的 patch error `0.4305 → 0.3882`，observed F1 `0.8803 → 0.9030`，full F1 `0.8250 → 0.8455`。

这证明 numeric-ID ordering 确实是旧 KSVD 的一个可修复问题，而且不需要牺牲完整 adjacency。signature control 也说明结构排序本身改善了 KSVD geometry，但约50% patches 仍有 tie；exact canonicalization 才能严格消除 vector/code 的 ID dependence。

automorphism 边界仍然存在：rooted canonical adjacency vector/code 为1.0，但 node-order equivariance约0.941、transition-slot match约0.896。结构等价节点没有纯结构唯一身份。因此 canonical topology 与 canonical node naming 必须分开表述。

重编号后重新运行 Beam8 时，rooted canonical pooled-code cosine约0.7045，说明 sampler patch-set 仍不同；但 full RMSE 三种子均值 `0.31477 → 0.31482`，重构质量不漂移。下一步若追求严格 representation consistency，应优化 sampler/set aggregation，而不是继续修改 local slots。

更新后的 reconstruction 主链：

```text
Beam8 node sets
→ rooted exact canonical slots
→ complete 45D adjacency
→ shared K24/T3 KSVD
→ slot-to-node sidecar + overlap stitching
```

本轮协议/结果：`KSVD_STRUCTURAL_CANONICAL_SLOT_PROTOCOL_20260802.md`、`STRUCTURAL_CANONICAL_SLOT_AUDIT_20260802.md`、`STRUCTURAL_CANONICAL_SLOT_AUDIT_SEED2_20260802.md`、`STRUCTURAL_CANONICAL_SLOT_AUDIT_SEED3_20260802.md`、`STRUCTURAL_CANONICAL_SLOT_MULTISEED_CONCLUSION_20260802.md`。

## 16. 第十二轮：下游读出是否只是直接统计

rooted-canonical KSVD 已证明适合 adjacency reconstruction，但 graph-level prediction 最终需要 permutation-invariant readout。为判断 code mean/std/max 是否只是换一种统计，本轮使用三组 graph/cover seeds，对9-class `family × degree` 做 matched attribution。

各 branch 全部使用 train-only standardization、相同12D PCA和fixed ridge：

| branch | joint-9 acc | family | degree | relabel-resample cosine |
|---|---:|---:|---:|---:|
| global stats | **0.8004** | **1.0000** | **0.8374** | **1.0000** |
| rooted raw bag | 0.4835 | 0.6523 | 0.7901 | 0.6344 |
| rooted raw true relation | **0.5802** | 0.7284 | 0.8230 | 0.6974 |
| rooted KSVD bag | 0.4136 | 0.5967 | 0.6132 | 0.3376 |
| rooted KSVD true relation | 0.4527 | 0.5823 | 0.7037 | 0.4008 |
| rooted KSVD shuffled relation | 0.4115 | 0.5761 | 0.6975 | 0.4010 |

三次独立审计均判定 `DIRECT_STATS_SUFFICIENT_FOR_SYNTHETIC_FACTORS`。当前 targets 是生成器的宏观 family 与 degree；sorted degree、whole spectrum、density 和triangle density 与其直接对齐，因此 direct stats显著胜出并不意外。

relations 仍有部分信号：RAW TRUE相对RAW BAG平均 `+0.0967`，KSVD TRUE相对KSVD BAG平均 `+0.0391`。但 K24/T3 compression 把相同relation机制从 `0.5802` 降至 `0.4527`，所以 KSVD不是下游relation增益的来源。

更新后的任务拆分：

```text
adjacency reconstruction    rooted canonical 45D + KSVD + stitching
synthetic macro downstream  direct whole-graph statistics
relation research           raw canonical patch tokens + true relations
```

不再声称对 KSVD codes 求 mean/std/max 会天然优于直接统计。真实50-node labels到位后，必须保留 `GLOBAL_STATS / RAW_RELATION / KSVD_RELATION` 三路 matched baseline。

本轮协议/结果：`KSVD_ROOTED_CANONICAL_DOWNSTREAM_ATTRIBUTION_PROTOCOL_20260802.md`、`ROOTED_CANONICAL_DOWNSTREAM_ATTRIBUTION_20260802.md`、`ROOTED_CANONICAL_DOWNSTREAM_ATTRIBUTION_SEED2_20260802.md`、`ROOTED_CANONICAL_DOWNSTREAM_ATTRIBUTION_SEED3_20260802.md`、`ROOTED_CANONICAL_DOWNSTREAM_ATTRIBUTION_MULTISEED_CONCLUSION_20260802.md`。
