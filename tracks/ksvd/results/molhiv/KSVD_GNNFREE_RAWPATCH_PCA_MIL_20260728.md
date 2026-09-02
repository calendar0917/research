# KSVD 路线：无 GNN raw-patch PCA prototype MIL 与匹配 GINE 审计

日期：2026-07-28

## 1. 这轮要回答的问题

本轮直接回答三个架构问题：

1. 当前 MolHIV 节点 patch 路线是否真的受 `max_nodes=8` 限制；
2. 字典是否应当与下游人物/任务匹配；
3. 三层原始节点 GINE 是否应继续作为 KSVD 的基础架构，还是存在更好的无 GNN 架构。

所有新实验都只使用 8,000 图开发子集中的 6,400 个 official-train 图：

- 三个固定 Bemis--Murcko scaffold folds；
- 每个模型固定训练 30 epochs；
- 只在训练结束后评估一次 outer held-out fold；
- official-valid/test 不编码、不评估；
- 主要确认采用 3 folds x 3 seeds，共 9 个配对单元。

## 2. `max_nodes=8` 在当前节点 patch 路线中不是节点截断

当前代码 `code/molhiv_node_tokens.py::centered_ego_vector()` 首先通过

```python
nodes, dist = ego_nodes(g, center, radius=radius)
```

取得完整 radius-2 ego。这里没有把节点数截到 8。随后完整 `nodes` 被用于：

- permutation-invariant WL/chemical/ring summary；
- 各距离壳层的原子直方图；
- induced ego 内各边的键直方图；
- 每层节点数与边数统计。

当前 `max_nodes=8` 的可见作用是：

1. 传给统计型 `labeled_wl_ring_patch_features()`，用于固定尺度/维度的 descriptor 设计；
2. 在 `count_scale=max_nodes` 中缩放节点数和边数坐标。

因此，本轮所用 **centered radius-2 node-patch 路线不存在“最多保留 8 个节点”的硬 patch cap**。早期 adjacency-padding/random-walk 路线中 `max_nodes` 的确可能表示节点集合截断，但不能把那个含义直接套到当前 848-d permutation-invariant descriptor 上。

## 3. GNN-free local metric

为了排除“冻结的 SSL-GINE 仍然偷偷提供了 GNN 基础”的混淆，本轮构造了完全无 GNN 的 local metric：

1. 完整 radius-2 permutation-invariant raw descriptor；
2. 每个 outer fold 只用 fit fold 拟合 PCA64/scale；
3. 每个节点向量 L2 normalization；
4. supervised GNN layers = 0；
5. unsupervised GNN layers = 0。

对应代码：

- `code/extract_molhiv_patch_metric_latents.py`
- `code/run_molhiv_prototype_occurrence_graph.py`

对应 latent archives：

- `rawpatch_pca64_latents_n8000_scaffoldfit_fold0.npz`
- `rawpatch_pca64_latents_n8000_scaffoldfit_fold1.npz`
- `rawpatch_pca64_latents_n8000_scaffoldfit_fold2.npz`

三个 archive 均只对 official-train 节点编码，official-valid/test 行严格为零。

## 4. 3 folds x 3 seeds：无 GNN node MIL

矩阵行是 fold 0/1/2，列是 seed 0/1/2。

### 4.1 Random real-prototype node MIL

| Fold | Seed 0 | Seed 1 | Seed 2 | Fold mean |
|---|---:|---:|---:|---:|
| 0 | 0.755089 | 0.748288 | 0.750804 | 0.751394 |
| 1 | 0.659546 | 0.676204 | 0.690189 | 0.675313 |
| 2 | 0.803926 | 0.760168 | 0.774673 | 0.779589 |

Grand mean：**0.735432**。

### 4.2 KSVD-direction node MIL

| Fold | Seed 0 | Seed 1 | Seed 2 | Fold mean |
|---|---:|---:|---:|---:|
| 0 | 0.721740 | 0.734171 | 0.760467 | 0.738793 |
| 1 | 0.644062 | 0.654842 | 0.660034 | 0.652979 |
| 2 | 0.782404 | 0.765515 | 0.797553 | 0.781824 |

Grand mean：**0.724532**。

### 4.3 Random real prototypes 对 KSVD

Random node MIL - KSVD node MIL：

- mean delta：**+0.010900**；
- 9 个单元中 random 赢 **6**、输 3；
- fold mean deltas：`+0.012601, +0.022334, -0.002235`。

因此，当前最强的无 GNN 结果来自“真实观察到的局部 prototype + node-level MIL”，而不是自由的 KSVD reconstruction directions。

这与此前 task-adapted dictionary 审计一致：KSVD 更擅长全局重建覆盖，但实际观察 prototype 的离散身份更适合图标签 MIL。

## 5. 协议匹配的 fixed-epoch original-node GINE

旧 GINE 参考是在每个 held-out fold 上选择最佳 epoch，和本轮固定 epoch 30 不完全公平。因此新增了严格匹配 runner：

- `code/run_molhiv_fixed_epoch_gine.py`

架构与历史强基线一致：

- OGB `AtomEncoder` / `BondEncoder`；
- 3 x GINEConv；
- 每层 `Linear-ReLU-Linear + BatchNorm + ReLU`；
- zero-initialized gated JK sum；
- mean graph pooling；
- 无 KSVD 特征；
- 固定 epoch 30 后只评估一次 held-out；
- 只构造 6,400 个 official-train PyG graphs。

结果：

| Fold | Seed 0 | Seed 1 | Seed 2 | Fold mean |
|---|---:|---:|---:|---:|
| 0 | 0.726927 | 0.725798 | 0.729449 | 0.727391 |
| 1 | 0.632355 | 0.626158 | 0.641338 | 0.633284 |
| 2 | 0.721477 | 0.748977 | 0.725448 | 0.731967 |

Grand mean：**0.697548**。

配对结果：

- random GNN-free MIL - fixed GINE：**+0.037884**，**9/9 wins**；
- KSVD GNN-free MIL - fixed GINE：**+0.026984**，**8/9 wins**。

为了防止结论只来自固定 epoch 对 GINE 不利，还比较了旧的 best-heldout-epoch GINE：

- 旧 GINE grand mean：0.717420；
- random GNN-free MIL - 旧 selected GINE：**+0.018012**，**8/9 wins**；
- KSVD GNN-free MIL - 旧 selected GINE：**+0.007112**，**8/9 wins**。

旧参考对 GINE 是乐观估计，因为它直接按 held-out AUC 选 epoch；即使如此，无 GNN random prototype MIL 仍稳定更强。

## 6. occurrence GINE 为什么不应继续加层/加 gate

此前 occurrence-graph screen 已比较：

- node MIL；
- occurrence MIL；
- 一层 occurrence GINE；
- zero-initialized gated occurrence GINE；
- KSVD persistent/shuffled/no identity。

关键结论：

- random occurrence GINE - occurrence MIL：-0.00372 mean；
- random gated GINE - occurrence MIL：-0.03551 mean，0/3 wins；
- KSVD occurrence GINE - occurrence MIL：-0.00865 mean；
- KSVD gated GINE - occurrence MIL：-0.00508 mean；
- KSVD gate 最终几乎停在零。

persistent KSVD ID 对 shuffled/no-ID 有约 +0.032 AUC 的稳定贡献，说明字典 atom identity 是真实信号；但 occurrence message passing 没有把这个信号变成 matched random 之上的优势。

因此不再运行多 seed occurrence-GINE，也不建议继续堆 GINE 层、gate 或复杂 fusion。

## 7. KSVD-guided real medoid：保留覆盖排序、恢复真实 prototype 身份

为了直接测试“KSVD coverage + real prototype identity”的折中，本轮新增：

- 对每个 KSVD atom，在对应 fit fold 的所有真实节点 patch 中寻找最近 cosine patch；
- 施加 32 个 prototype 互不重复的约束；
- prototype identity/order 仍由 KSVD atom 决定；
- prototype 本身变成真实观察到的 patch；
- 不使用图标签；
- 仍使用完全无 GNN node MIL。

代码控制：`ksvd_medoid_node_mil`。

Seed-0 结果：

| Control | Fold 0 | Fold 1 | Fold 2 | Mean |
|---|---:|---:|---:|---:|
| KSVD direction node MIL | 0.721740 | 0.644062 | 0.782404 | 0.716069 |
| KSVD-guided real medoid MIL | 0.750402 | 0.644684 | 0.777656 | **0.724247** |
| random real-prototype MIL | 0.755089 | 0.659546 | 0.803926 | **0.739520** |

Medoid - KSVD：

- mean：**+0.008179**；
- 2/3 nominal wins。

Medoid - random：

- mean：**-0.015273**；
- 0/3 wins。

所以 medoid 结果支持“真实 prototype 身份能够修复一部分 KSVD 损失”，但尚不足以击败更简单的随机真实 prototype。按照 matched-random gate，本分支不值得直接做多 seed promotion。

## 8. 对“字典是否应该和人物/任务匹配”的回答

答案需要拆成两层。

### 8.1 字典 atom 本体不应直接由图标签自由塑形

已有实验中：

- naive graph-label-based atom selection 明显失败；
- label-shuffled selector 反而可能更好；
- task-adapted KSVD 在 3 x 3 中没有稳健超过 frozen KSVD；
- random real prototypes 持续强于 KSVD directions。

这说明把 molecule-level label 粗暴分配到局部 patch，会制造严重的 credit-assignment 错误。一个阳性分子中绝大多数节点并不一定是致因局部结构。

因此不建议把 dictionary identity 变成完全监督、随人物/任务自由漂移的列向量。

### 8.2 任务匹配应发生在 occurrence weighting/readout，而非 prototype identity

更合理的职责分工是：

1. **局部 vocabulary**：由真实、无标签、fold-fit patch 构成，保证 prototype 可解释且确实出现过；
2. **任务层**：学习每个 prototype 在什么节点、什么图上下文中重要；
3. **graph-level MIL**：只用图标签训练 attention/readout，不把标签硬贴给每个节点；
4. **可选 coverage regularizer**：KSVD 只作为覆盖/重建约束或 residual audit，不主导 prototype identity；
5. **交互层**：只有在稳定的无 GNN 基线之上，再加入非常稀疏、受约束的 pair/transition interaction，而不是恢复三层原始图 GINE。

也就是说，字典应当“与任务配合”，但不是“字典列本身被图标签直接监督”。

## 9. 当前架构决策

### 9.1 当前应该保留什么

保留三层 original-node GINE 作为标准强基线和审稿对照，但不再把它作为 KSVD 方法的中心。

### 9.2 当前主架构

目前最有希望的基础架构是：

```text
完整 radius-2 permutation-invariant raw patch
    -> fold-fit PCA64
    -> 32 个真实 local prototypes
    -> sparse node-to-prototype affinities
    -> prototype identity-aware node representation
    -> attention + mean + max graph MIL
    -> graph prediction
```

特点：

- 原始节点图 supervised message passing = 0；
- upstream unsupervised GNN = 0；
- 不存在 GINE 绕过字典的路径；
- local prototype 是真实 patch；
- 任务信号只进入 graph MIL readout。

### 9.3 KSVD 在其中的合理位置

当前证据不支持“KSVD directions 是最佳分类 vocabulary”。更合理的位置有：

- coverage/reconstruction regularizer；
- 对真实 prototype bank 的覆盖诊断；
- 初始化或约束 medoid/candidate selection；
- 提供 reconstruction residual/novelty，而不是直接替代真实 prototype identity。

KSVD-guided medoid 的 +0.00818 改善表明这个方向有机制依据，但它还没有跨过 random-real matched control。

## 10. 最终结论

1. 当前 node-patch 路线没有 8 节点硬截断；`max_nodes=8` 主要是 descriptor scaling/固定统计设计参数。
2. 三层 GINE 是合理基线，但不是合理的 KSVD 中心架构；它与 radius-2 patch 的信息范围重叠，并允许模型绕过字典。
3. 一层 occurrence GINE 和 gated GINE 均未改善 occurrence MIL，应停止继续堆 message passing。
4. 完全无 GNN 的 raw-PCA real-prototype node MIL 已得到 3 folds x 3 seeds 确认：0.735432。
5. 它以 +0.037884、9/9 wins 超过协议匹配 fixed-epoch GINE，也超过对 GINE 有利的旧 selected-epoch 参考。
6. KSVD identity 有用，但自由 KSVD directions 仍弱于真实 prototype；因此目前不能宣称 KSVD-specific superiority。
7. 字典应与任务配合，但任务信号应主要进入 prototype occurrence weighting 和 graph-level MIL，而不是用 graph label 直接监督局部 dictionary atom。
8. 下一阶段不应再搜 GINE gate/layer；应以稳定的 GNN-free real-prototype MIL 为主线，只加入受约束、可归因的 KSVD coverage/residual 或稀疏 prototype interaction。

## 11. 主要 artifacts

代码：

- `code/extract_molhiv_patch_metric_latents.py`
- `code/run_molhiv_prototype_occurrence_graph.py`
- `code/run_molhiv_fixed_epoch_gine.py`
- `code/summarize_molhiv_gnnfree_patch_mil.py`

结果：

- `rawpatch_pca64_node_mil_3fold3seed_summary.json`
- `fixed_epoch30_gine_fold{0,1,2}_seed{0,1,2}.json`
- `rawpatch_pca64_ksvd_medoid_node_mil_fold{0,1,2}_seed0.json`
- `KSVD_PROTOTYPE_OCCURRENCE_GRAPH_20260728.md`
- 本报告 `KSVD_GNNFREE_RAWPATCH_PCA_MIL_20260728.md`
