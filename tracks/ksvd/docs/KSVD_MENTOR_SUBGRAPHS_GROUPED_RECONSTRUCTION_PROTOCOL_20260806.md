# 导师 50-node 真实子图：Grouped-split KSVD reconstruction 协议

> 日期：2026-08-06  
> 状态：结果前冻结  
> 前置结果：`tracks/ksvd/results/mentor_subgraphs/BEAM8_PILOT_20260806.md`  
> 前置判定：`READY_FOR_GROUPED_KSVD_RECONSTRUCTION_FOLLOWUP`

## 1. 本轮问题

本轮只回答：在导师真实 50-node 子图的 held-out graphs 上，固定 `K=24/T=3` 的普通 KSVD 是否比同一 train fold 的 deterministic INIT 稳定改善 rooted-canonical patch reconstruction；这种改善在 source/root-overlap-aware split 下是否仍存在；BASE 与 FAIR95、三种 geometry 的 rate–distortion 关系是什么。

本轮不训练 Transformer，不使用 graph labels，不把 source node ID、root candidate 或 density stratum 输入 patch vector，也不重新声称 KSVD atoms 是语义 motif。Source identity 只用于 split 和 leakage audit。

## 2. 冻结数据与 branches

- 数据缓存：`data/subgraphs_50_20_10000_batch_0.npz`；source SHA-256 必须与 pilot 一致。
- 图集合：沿用 pilot seed `20260806` 的同一 500 图、五个 average-degree strata 各 100 图。
- preprocessing：`GLOBAL_WL preorder → Beam8/R1 → rooted-canonical local slots`。
- cover seed：`970201`；Beam8 设置与 pilot 完全一致。
- geometry：`s8/o2`、`s10/o3`、`s12/o4`。
- checkpoints：只比较 `BASE` 与 `FAIR95`，且都来自同一 graph/geometry 的最长 60-patch chain 前缀。
- 共 6 个 reconstruction branches；不得根据本轮结果回头扫描 cover seed、K、T、updates 或 restarts。

## 3. 两个 split views

### 3.1 Random-reference

建立 deterministic、density-balanced 的 3-fold graph split，seed=`20260807`。它只作为常规随机图隔离参考，不声称 source identity 隔离。

### 3.2 Root-grouped 主视图

以 `root_candidate` 的 global vocabulary index 为不可拆分 group，建立 deterministic 3-fold split：

1. 同一 root group 的所有图必须进入同一 test fold；
2. 贪心分配 root groups，使每折 graph count 与五个 density strata counts 接近全局三等分；
3. group 顺序固定为：group size 降序、density concentration 降序、seeded tie-break、root ID；
4. 每一步选择使加权平方偏差最小的 fold，精确 tie 用 fold index；
5. 每折必须非空，每个 root group 恰好出现于一个 test fold。

`root_candidate` 尚未由生成方确认是真实 sampling root，因此该 view 的准确名称是 **root-candidate-grouped**，不能写成已验证的 generation-root split。

### 3.3 为什么不要求 source-node-disjoint split

pilot 的 source-node overlap graph 在“共享至少一个 source node”或“共享至少一条 source edge”时都是单一 500-graph connected component。强制 node-disjoint 或 edge-disjoint grouping 会塌缩成一个 giant component，无法形成有意义的 3-fold evaluation。因此：

- root group integrity 是主 split gate；
- source node/edge/patch overlap 是必须报告的 exposure audit，而不是伪造为 0 的 hard constraint；
- random-reference 与 root-grouped 的差异用于衡量 root-level memorization sensitivity，不解释为完全 inductive source-graph generalization。

## 4. Split leakage / exposure audit

每个 view × fold 必报：

- train/test graph count 与 density-stratum counts；
- train/test distinct roots、root intersection count；root-grouped 必须为 0；
- test source nodes 中在 train 出现过的比例，以及 train/test source-node Jaccard；
- test source edges中在 train 出现过的比例，以及 train/test source-edge Jaccard；
- 每张 test graph 的 source-node exposure mean/p10/min；
- 每张 test graph 的 source-edge exposure mean/p10/min；
- 每个 geometry/checkpoint 下，test rooted-canonical patch adjacency vectors 在 train 中 exact-seen 的 occurrence fraction 与 unique-vector overlap。

Patch-vector overlap按 rooted-canonical adjacency bit vector 比较，不包含 source IDs；它衡量结构模板重复，不衡量 source identity 泄漏。

## 5. Train-only reconstruction protocol

对每个 split view × branch × fold：

1. 仅堆叠 train graphs 的 patch vectors；
2. 仅用 train patches 计算 coordinate-wise mean，并对 train/test 使用同一 mean；
3. 仅用 centered train patches构造所有 basis/dictionaries；
4. 测试只做固定 basis/dictionary 编码与 reconstruction。

冻结参数：

```text
K = 24
T = 3
T_min = 1
KSVD updates = 25
KSVD internal seed = 0
PCA rank = 3
random dictionary seed = 970301 + deterministic branch/fold offset
per-patch normalization = false
centering = train coordinate mean
```

阶段：

- `RAW`：原始 patch vectors；是 cover/stitch ceiling，不是压缩模型。
- `PCA3`：train-only rank-3 PCA，三个 dense coefficients。
- `RANDOM`：24 个 train-centered nonzero patch columns均匀无放回抽取并归一化，测试用 T=3 OMP。
- `INIT`：train-only deterministic maximin 24 atoms，测试用 T=3 OMP。
- `FINAL`：从同一 INIT 出发执行 25 次普通 KSVD updates，测试用 T=3 OMP。

RANDOM 必须来自 train patches，而不是 Gaussian basis；这样它控制“任意真实 patch dictionary”而不是改变 dictionary support/domain。

## 6. Reconstruction 报告

每个 stage 同时报告：

- graph-balanced patch relative Frobenius error；
- observed-pair RMSE、accuracy、edge precision/recall/F1；
- full-adjacency RMSE、accuracy、edge precision/recall/F1；
- repeated-pair disagreement；
- dictionary nondead atoms、effective atom count、maximum activation share、coherence（适用 stage）；
- dictionary scalars 与 mean code scalars per graph。

BASE 与 FAIR95 都给两种 full-graph reconstruction：

1. `uncorrected`：未观察 pair zero-fill；
2. `exact-residual-corrected`：只将 cover 未观察到的真实 residual edges 作为显式 sidecar 置 1。

Residual-corrected 只隔离 patch compression error，不把 residual bit cost 算作免费 codec，也不允许 residual 信息参与 dictionary training。

## 7. 冻结 gates

### 7.1 Split/data gate

- 两个 views 都完整、互斥地覆盖 500 图；
- 每折 train/test 非空；
- root-grouped view 的 train/test root intersection严格为 0；
- 所有 source exposure 指标有限且在 `[0,1]`；
- RAW patch error、observed RMSE、observed disagreement严格为 0；
- squared-error decomposition audit 误差 `<=1e-10`。

### 7.2 KSVD optimization gate（逐 branch、以 root-grouped 为主）

- 3/3 folds `FINAL patch error < INIT patch error`；
- graph-count-weighted mean INIT→FINAL patch-error relative reduction `>=0.05`；
- 3/3 folds FINAL nondead atoms `>=20/24`；
- 3/3 folds FINAL maximum activation share `<0.50`；
- FINAL observed-pair RMSE 不得比 INIT 更差；
- root-grouped 的 improvement 不得与 random-reference 系统性反向（root-grouped 3 折至少 2 折为正，且 weighted mean 为正）。

这里使用 5% 而不是旧的 10% 强 gate，因为近期 Beam8 synthetic audit 已将普通 KSVD定位为 compressor/baseline；低于 5% 则不值得把 learned dictionary 当作后续默认 token source。

### 7.3 Control/Pareto gate

FINAL 若要进入后续 relation-token 比较，还必须：

- root-grouped weighted mean patch error优于 RANDOM；
- root-grouped weighted mean patch error优于 PCA3。PCA3 与 KSVD 都使用每 patch 3 个连续系数，而 PCA basis 更小；若 FINAL 误差不低于 PCA3，则 PCA3 在本轮 reconstruction rate–distortion 上支配 FINAL，只能保留 KSVD 作为 atom-identity research control，不能进入默认 relation-token branch；
- 对 BASE/FAIR95 的选择同时考虑 uncorrected RMSE、residual-corrected RMSE、dictionary scalars 与 code scalars，多个非支配 geometry 必须全部保留，不强行合成单一分数。

## 8. 结果分类

- split/data gate失败：`FAIL_MENTOR_GROUPED_RECONSTRUCTION_CONTRACT`；
- 无 branch通过 KSVD optimization gate：`KEEP_RAW_PATCH_OR_INIT_BASELINE_NO_KSVD_GAIN`；
- 有 branch通过 optimization，但没有合理 control/Pareto位置：`KSVD_OPTIMIZES_BUT_NO_RATE_DISTORTION_ADVANTAGE`；
- 至少一个 branch 在 root-grouped 下通过 optimization 与 control/Pareto：`KSVD_GROUPED_RECONSTRUCTION_READY_FOR_RELATION_TOKEN_ABLATION`。

即使最后一类通过，也只允许进入 `RAW/INIT/FINAL/residual-token` 的 matched downstream ablation；不自动证明语义 motif，不自动开始 Transformer 大规模调参。

## 9. 必须向数据生成方确认的外部问题

这些问题不阻塞本轮 reconstruction，但会限制结论措辞：

1. 文件名中的 `20` 表示什么；
2. NetworkX first insertion node 是否确为 sampling root；
3. 10000 图是否来自同一张 2805-node source graph；
4. 是否有其他 batch、生成 seed、graph labels 或 sampling metadata。
