# 导师真实子图 relation-token masked audit 协议

> 日期：2026-08-06  
> 状态：在 `KSVD_GROUPED_RECONSTRUCTION_READY_FOR_RELATION_TOKEN_ABLATION` 之后、真实数据结果之前冻结。  
> 性质：无标签、低容量 mechanism audit；不是 downstream/Transformer 实验。

## 1. 研究问题与边界

导师子图库没有 graph/node/patch/downstream labels。因此本轮不构造伪标签，也不声称分类、回归或 Transformer 有效，只回答三个较窄的问题：

1. 其他 patch 的 token bag 能否预测一个被遮蔽 patch 的 permutation-invariant 局部结构；
2. 将相同 context tokens 与真实 patch overlap / center-distance 关系绑定后，是否优于无关系 bag 和等容量 shuffled-binding control；
3. FINAL KSVD relation token 是否优于同一起点、尚未更新的 INIT token，从而把 relation signal 与 dictionary-learning value 分开。

平均度 strata 只用于样本平衡和条件报告，不作为输入或目标。NetworkX first-insertion node 仍只称 `root_candidate`。source node IDs 只允许用于 split、映射与 exposure/integrity audit，禁止进入 patch vectors、dictionary、tokens、relation features 或 ridge targets。

本轮不加入 residual-edge token。精确 residual edges 来自完整真实图；若在 masked target 预测时直接可见，可能泄漏被遮蔽 patch 的结构。除非后续另行冻结严格的 visible-only 构造，否则 residual channel 不属于本实验。

## 2. 冻结数据与 split

首轮 pilot 使用：

- 数据：`data/subgraphs_50_20_10000_batch_0.npz` 安全缓存；原始 pickle 只读；
- 选择：`100` graphs，`select_pilot_indices(..., seed=20260806)`，五个平均度 strata 各 `20`；
- primary split：`root_candidate_grouped`，`3` folds，split seed `20260807`；
- root-candidate group 在 train/test 间必须完整隔离；
- 每个图恰好作为 test 一次；
- source-node/source-edge overlap 只报告，不伪装成 source-disjoint split。

本轮不使用 `random_reference` 做主判定，避免在一个小 pilot 中产生第二套选择自由度。若首轮通过，500-graph confirmatory run 仍使用同一 root-grouped split 定义。

## 3. 冻结 cover 与 local representation

只运行已通过 grouped reconstruction gate 的质量端点：

- geometry：`s8_o2`（patch size `8`，target overlap `2`）；
- checkpoint：`BASE`；
- cover seed：`970201`；
- Beam8 / candidate restarts `1`；
- maximum patches `60`，允许 partial cover；
- stable preprocessing：`GLOBAL_WL preorder → marginal candidate cover → rooted-canonical local slots`；
- checkpoint 通过现有 prefix trajectory 与 `select_operating_checkpoints` 重建，不读取先前结果中的 learned artifacts。

source IDs 不进入上述 local representation。

## 4. Masked target

对每个 graph 的每个 cover patch 轮流遮蔽。目标是 patch adjacency 的 permutation-invariant descriptor：

```text
s 个 sorted normalized degrees
s 个 sorted normalized adjacency eigenvalues
edge density
triangle density
```

对 `s=8`，目标维度为 `18`。实现必须由 patch size 动态推导维度，禁止保留旧 `s=10` 实验的硬编码 `22D` 假设。

被遮蔽 patch 的 token 必须从所有 context pooling、weighted means 和 relation rows 中排除。test target descriptor 只能用于最终评估，不能进入 feature 标准化、ridge fitting、dictionary fitting 或 checkpoint 选择。

## 5. Token families 与 matched branches

每个 token family 均运行 `BAG / TRUE_RELATION / SHUFFLED_RELATION` 三个等 decoder-capacity 分支，共九个分支。

### 5.1 `INVARIANT`

context token 就是第 4 节 descriptor。这是手工、ID-free 的信息上界/control，不是 learned encoder。

### 5.2 `INIT`

- train patches 上计算 train-only mean；
- deterministic maximin 初始化 `K=24`；
- 不做 dictionary update；
- train/test patch 均以 `T=3, T_min=1` 编码。

### 5.3 `FINAL`

- 与 INIT 使用完全相同的初始化；
- 普通 KSVD 更新 `25` 次，seed `0`；
- `K=24, T=3, T_min=1`；
- train/test patch 使用 FINAL dictionary 编码。

不扫描 K、T、updates、restart、ridge alpha 或 cover seeds 来挽救失败结果。

## 6. Feature construction

对目标 patch `i`，context 是同图中除 `i` 外的所有 patch tokens。

- `BAG`：context tokens 的 mean / std / max；
- `TRUE_RELATION`：BAG，加 overlap-fraction-weighted token mean、`exp(-center shortest-path distance)` weighted token mean，以及 overlap/distance summary；
- `SHUFFLED_RELATION`：保留同图、同目标、同 context token multiset、同 relation marginals 和相同 feature dimension，只确定性地错配 context tokens 与 relation weights。

relation 只由 stable-local patch node overlap 和图内 center shortest-path distance计算。禁止使用 global source-node ID 的数值、频率或 embedding。

## 7. Decoder 与指标

每个 fold、每个 branch 独立训练 ridge decoder：

- decoder：train-only feature mean/std；
- ridge `alpha=0.01`；
- 不共享 test statistics；
- 不调参。

主指标是 test graph-balanced RMSE：先对每个图的 masked rows 计算 RMSE，再对图等权平均。报告：

- overall RMSE；
- degree block RMSE；
- spectrum block RMSE；
- density/triangle block RMSE；
- fold-level、per-graph、五个 density-stratum 条件结果。

由于不同 token family 的 feature dimension 不同，结论只能在预注册 matched controls 下解释；INVARIANT 是手工 control，不能被描述为与 24D sparse code 完全同参数量的 encoder。

## 8. 冻结 gates

定义 reduction：

```text
reduction(A → B) = (RMSE_A - RMSE_B) / RMSE_A
```

正式通过需要同时满足：

1. `FINAL_TRUE_RELATION` 相对 `FINAL_BAG` 的 graph-balanced mean overall RMSE 改善 `>= 2%`；
2. `FINAL_TRUE_RELATION` 相对 `FINAL_SHUFFLED_RELATION` 改善 `>= 2%`；
3. 在 `3/3` folds 中，FINAL TRUE 同时优于 FINAL BAG 与 FINAL SHUFFLED；
4. 在五个 density strata 中至少 `4/5`，FINAL TRUE 同时优于 FINAL BAG 与 FINAL SHUFFLED；
5. `FINAL_TRUE_RELATION` 优于 `INIT_TRUE_RELATION`；
6. `INVARIANT_TRUE_RELATION` 相对 `INVARIANT_SHUFFLED_RELATION` 改善 `>= 2%`，证明真实 relation binding 不只依赖 KSVD；
7. cover reachability、finite matrices、dynamic target dimensions、masked-target exclusion、train/test graph isolation、root-group integrity、每图恰好 test 一次等 invariants 全部通过。

判定标签：

- 全部通过：`MENTOR_RELATION_TOKEN_PILOT_SUPPORTED`；
- relation gates 通过，但 `FINAL_TRUE < INIT_TRUE` 失败：`RELATION_SIGNAL_PRESENT_BUT_NO_FINAL_KSVD_VALUE`；
- TRUE 有正增益但不足 2% 或不稳定：`MENTOR_RELATION_SIGNAL_BELOW_GATE`；
- TRUE 不优于 matched shuffled/bag control，或数据/invariant gate 失败：`REJECT_CURRENT_MENTOR_RELATION_TOKEN`。

## 9. Pilot 后决策

- 若 `MENTOR_RELATION_TOKEN_PILOT_SUPPORTED`：原样扩展至 frozen 500-graph selection；不改变超参数或 gates。500-graph confirmation 通过后，才考虑额外比较 `s8_o2 FAIR95` 与 `s12_o4 BASE`。
- 若 `RELATION_SIGNAL_PRESENT_BUT_NO_FINAL_KSVD_VALUE`：保留 relation mechanism 结论，但不得声称 learned KSVD dictionary 提供额外价值；优先研究 token stability/encoder，而不是启动 Transformer。
- 若 below-gate 或 reject：停止扩大和 Transformer 路线；先检查关系定义是否缺少 segment-transition 等结构，但新定义必须另行预注册，不能回改本轮。

即使通过，本轮也只支持“真实无标签子图中存在可被低容量 decoder 利用的 relation-bound masked structural signal”，不支持 motif 语义、监督 downstream 增益或最终模型有效性。
