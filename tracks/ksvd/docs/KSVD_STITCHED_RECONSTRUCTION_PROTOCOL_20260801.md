# Target-bridge cover：KSVD stitched reconstruction 协议

> 日期：2026-08-01  
> 状态：正式结果不可见前冻结  
> 前置：`PASS_SINGLE_CHAIN_TARGET_BRIDGE`；raw cover 的 edge coverage `0.6889`，连续 overlap、target hit、patch connectivity 和 mapped replay 均通过。

## 1. 研究问题

本轮不再修改 sampler，只回答：

> 对连续 target-bridge patches 做 sparse KSVD 压缩后，held-out patch reconstruction improvement 能否转化为更好的整图 stitching，并且不在 overlap 区域产生更大的预测冲突？

仍不使用 labels、不做分类。

## 2. 数据、cover 与 split

复用相同固定 graph bank：

```text
graph_bank_seed = 810001
families = regular, small_world, block
n_nodes = 50
target degree = 15, 20, 25
graphs/family/degree = 8
patch_size = 10
target_overlap = 5
cover = edge_target_bridge
cover_seed = 830101
```

按每个 `family × degree` cell 的 replicate index 做 deterministic 3-fold：每折分别持有 replicate `fold, fold+3, fold+6`（若存在）。每个 cell 都同时出现在 train/test，图实例不重叠。

每折：

- cover 可独立从每张图提取，不使用 labels；
- coordinate mean、INIT、KSVD 和 PCA 只 fit train graphs；
- test graphs 只编码和 stitching；
- 不根据 test 选择 `K/T/iterations/rank/threshold`。

## 3. 表示与模型

10-node patch 使用 local ordered adjacency upper triangle：

```text
dimension = C(10,2) = 45
```

训练折 coordinate centering 后比较：

1. `RAW`：不压缩的真实 induced patches，作为 sampling/stitching ceiling；
2. `INIT`：deterministic maximin 真实 train columns；
3. `FINAL`：从同一 INIT 做 25 次 KSVD updates；
4. `PCA3`：train-only rank-3 PCA，使用与 sparse code 相同的 3 个连续系数。

冻结 KSVD：

```text
K = 24
T = 3
T_min = 1
updates = 25
restarts = 0
```

PCA3 是同系数预算的线性基线。它没有稀疏 atom 使用语义，但能检验收益是否只是普通低秩投影。

## 4. Stitching

每个 reconstructed patch 的 45 个 local pair predictions 通过保存的 node ids 映回原图 pair。若一个 pair 出现在多个 patches：

```text
stitched prediction = 所有 occurrence predictions 的平均
```

未观察 pair 的 prediction 固定为 0，不做图补全模型。这样 full-graph 指标同时包含 sampling ceiling 和 compression error；observed-pair 指标只衡量 compression。

## 5. 指标

每张 held-out graph 先独立计算，再 graph-balanced 平均：

- raw patch relative Frobenius error；
- observed-pair RMSE；
- observed-pair edge precision/recall/F1，threshold 固定 `0.5`；
- full adjacency RMSE/accuracy；
- full true-edge recall/F1；
- repeated-pair disagreement：同一 global pair 在多个 reconstructed patches 中 prediction std 的均值；
- overlap repeated-pair count；
- FINAL dictionary nondead atoms、maximum activation share 和 coherence。

`RAW` 必须满足 observed RMSE/disagreement = 0、observed F1 = 1；否则是 stitching 实现错误。

## 6. Gates

### 6.1 KSVD optimization gate

FINAL 必须：

1. held-out patch relative error 在 3/3 folds 低于 INIT；
2. mean relative reduction >= `0.10`；
3. 每折至少 `20/24` nondead atoms；
4. 每折 maximum activation share < `0.50`。

### 6.2 Stitched reconstruction gate

FINAL 必须：

1. observed-pair RMSE 在 3/3 folds 低于 INIT；
2. mean relative reduction >= `0.05`；
3. repeated-pair disagreement 不高于 INIT；
4. full true-edge recall 不低于 INIT `-0.02`；
5. observed-pair F1 不低于 INIT；
6. mean observed-pair RMSE 低于 PCA3。

## 7. 判定

- `PASS_KSVD_STITCHED_COMPRESSOR`：两个 gates 全部通过；可进入真实导师数据的相同无标签结构审计；
- `KSVD_RECON_ONLY_NO_PCA_ADVANTAGE`：KSVD/stitched INIT→FINAL 通过，但不优于 PCA3；只保留一般压缩结论；
- `FAIL_KSVD_STITCH_CONSISTENCY`：patch error 改善但 disagreement 或 stitched error gate 失败；
- `FAIL_KSVD_COVER_COMPRESSION`：held-out patch optimization/health 失败；
- `FAIL_RAW_STITCH_INVARIANTS`：RAW ceiling 不精确。

## 8. 边界

通过只说明 KSVD 适合压缩当前 traversal-relative cover，不说明：

- atom 是合法或可命名 motif；
- KSVD 能补回 sampler 未观察的约 31% edges；
- 不同图的 patch index/segment 具有共享绝对位置；
- 下游分类或 Transformer 已经有效。
