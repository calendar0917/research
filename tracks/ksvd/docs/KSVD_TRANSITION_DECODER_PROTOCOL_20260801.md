# KSVD patch chain：transition-aware decoder 协议

> 日期：2026-08-01  
> 状态：正式结果不可见前冻结  
> 前置：construction ordering 优于 persistent ordering；transition correspondence 不应继续通过手工排序压入 vanilla KSVD。

## 1. 研究问题

> 在当前 patch 的 KSVD sparse code 之外，正确的 previous-patch code、overlap slot map 和 transported local degree 是否能无标签地改善 held-out patch/graph reconstruction？

本轮是低容量线性机制探针，不做 GNN、不使用 graph labels。

## 2. 冻结输入

完全复用 construction-order stitched audit：

```text
72 graphs, deterministic 3-fold
target-edge bridge cover
s=10, overlap=5
K=24, T=3, updates=25
cover_seed=830101
```

每折 FINAL dictionary 和 codes 只由 train graphs 拟合；test graphs 只编码。

## 3. Decoders

### BASE_FINAL

标准 `D_FINAL @ x_t + train_mean`。

### CURRENT_ONLY

train-only multivariate ridge：

```text
input = current sparse code x_t (24-D)
target = centered raw patch y_t (45-D)
```

它控制“重新校准 decoder”本身的收益。

### TRUE_TRANSITION

对每个 `t>0`：

```text
current code                         24
previous code                        24
previous reconstructed node degrees
  transported into current slots     10
source slot index / 9                 10
shared-slot mask                      10
total                                 78
```

previous degrees只能从 `BASE_FINAL` reconstructed previous patch 计算，不能读取 previous raw adjacency。第一个 patch 没有 previous context，使用 CURRENT_ONLY prediction。

### SHUFFLED_TRANSITION

保持每个 current code 不动，在每张 test graph 内把 previous-code/map/degree context 循环错位一格。它保留 context 分布和模型容量，只破坏正确 transition binding。

## 4. Ridge protocol

```text
alpha = 1e-2
feature standardization = train-only
intercept = unpenalized
```

不扫描 alpha，不使用 test 选择参数。

## 5. 指标与 gates

继续报告 patch error、observed RMSE/F1、full edge recall/F1、overlap disagreement。

`TRUE_TRANSITION` 必须在至少 2/3 folds 且 fold-balanced mean 同时满足：

1. observed RMSE 比 CURRENT_ONLY 至少低 `0.02` relative；
2. observed RMSE 比 SHUFFLED_TRANSITION 至少低 `0.02` relative；
3. overlap disagreement不高于 CURRENT_ONLY；
4. observed F1不低于 CURRENT_ONLY；
5. full edge recall不低于 CURRENT_ONLY `-0.01`；
6. RAW stitching invariants 全部通过。

## 6. 判定

- `PASS_EXPLICIT_TRANSITION_CONTEXT`：正确 correspondence 有稳定 added value，可进入低容量 patch-chain message passing；
- `CURRENT_DECODER_RECALIBRATION_ONLY`：CURRENT_ONLY 改善 BASE，但 TRUE 不优于 CURRENT/SHUFFLED；transition map 尚无 added value；
- `REJECT_LINEAR_TRANSITION_CONTEXT`：连 current recalibration 也无稳定收益，停止线性 relation decoder；
- `FAIL_TRANSITION_DECODER_INVARIANTS`：RAW、feature isolation 或 shuffle contract 失败。

## 7. 边界

通过不代表 Transformer 或分类有效；它只说明 patch-chain correspondence 对无标签结构重构具有可测增量。失败也不否定非线性 message passing，但会提高继续扩模型所需的证据门槛。
