# Patch-chain 节点身份与 bit-rate 审计协议

> 日期：2026-08-02  
> 状态：结果前冻结  
> 目标：检验 Beam8 v2 是否依赖全图节点编号，以及把节点身份 side information 计入后是否仍能主张图压缩。

> 术语说明（结果后勘误，不改变公式或 gate）：下文的 `chain-aware optimistic lower bound` 更准确地说是“对 patch chain 有利的 generic fixed-field estimate”。它利用组合数减少固定字段长度，但没有建立 sampler-induced entropy 的信息论下界；因此失败约束的是当前显式编码主张，不证明所有可能的联合熵编码都不可能更短。

## 1. 问题拆分

当前方法同时使用三种坐标：

1. 全图 node ID：定位邻接矩阵行列，并在 stitching 时合并同一 global pair；
2. patch slot：把 `10×10` 局部邻接矩阵向量化为 45D KSVD sample；
3. transition map：记录相邻 patch 的三个共享节点分别位于哪些 slots。

本审计不把“程序必须用数组索引”误判为缺陷。真正要检验的是：

- 对同一抽象图重新编号并重新运行 sampler，得到的 chain/code substrate 是否等变；
- 若要求恢复原来的 labeled adjacency，node IDs、transition 和 sparse supports 是否被完整计入码率；
- 当前 reconstruction 收益究竟是结构压缩，还是忽略 identity sidecar 后的局部矩阵分解。

## 2. 数据与冻结方法

- graph bank seed：`820201`；
- 18 张 50-node graphs：`regular/small_world/block × degree 15/20/25 × 2 replicates`；
- 每张图三个 cover seeds：`920101/920102/920103`；
- 每张图一个由 graph index 冻结生成的 node permutation；
- sampler：Beam8/R1、`s=10,o=3,m=1.5`；
- 不使用 graph labels、test truth completion 或下游任务。

## 3. Relabel-resampling 审计

对每个 graph/cover-seed：

1. 在原图运行 Beam8；
2. 用 `new_index -> old_index` permutation 重编号邻接矩阵；
3. 在重编号图上用相同 scalar RNG seed 重新运行 Beam8；
4. 把新 cover 映回旧 node IDs；
5. 比较：
   - exact ordered-chain match；
   - position-wise patch-set Jaccard；
   - local 45D adjacency-vector row match；
   - transition-slot-map match；
   - edge/pair coverage 和 RAW full RMSE 的绝对变化。

另外对已经生成的原 cover 做 mapped replay。它只验证 representation mapping 正确，不能替代 resampling equivariance。

### 判定

- `STRICT_RELABEL_EQUIVARIANCE_PASS`：exact chain、local-vector rows、transition maps 均达到 `0.99`；
- 否则为 `SAMPLER_NOT_STRICTLY_RELABEL_EQUIVARIANT`。

覆盖质量稳定单独报告，不用于把 strict failure 改判为 pass：

- mean absolute edge-coverage delta `<=0.02`；
- mean absolute pair-coverage delta `<=0.02`；
- mean absolute RAW-RMSE delta `<=0.02`。

## 4. 完整 bit accounting

### 4.1 直接 labeled adjacency

50-node undirected simple graph 的 bit-packed upper triangle：

```text
M = C(50,2) = 1225 bits / graph
```

同时报告已知 edge count `m` 时的 enumerative lower bound：

```text
ceil(log2 C(M,m)) + ceil(log2(M+1))
```

### 4.2 Patch identity sidecar

报告两个边界：

- fixed-width upper bound：每个 patch 的 10 个 ordered global IDs，各用 `ceil(log2 50)=6` bits；
- chain-aware optimistic fixed-field estimate：
  - first patch：`log2 P(50,10)`；
  - 每个后继 patch：选择 previous patch 中保留的 3 slots，成本 `log2 C(10,3)`；
  - 再编码其余 7 个 ordered global IDs，成本 `log2 P(47,7)`。

该 estimate 已利用 Beam8 当前“retained nodes 位于前 3 slots 且保留 previous order”的实现约束；忽略 framing、patch count、checksums 和模型开销，因此对 patch chain 有利。它不是 sampler distribution 下的严格 entropy lower bound。

### 4.3 RAW 与 KSVD payload

- RAW-naive：identity estimate + `45 × patch_count` local adjacency bits；
- RAW-unique：identity estimate + 实际 unique observed global-pair values；
- sparse KSVD support：每个 nonzero 至少需要 `ceil(log2 24)=5` atom-index bits；
- coefficient budget：计算在不超过 1225-bit adjacency baseline 时，每个约 `3 × patch_count` coefficient 还能获得多少 bits。

dictionary、coefficient quantizer、completion model 和 framing 均未计入上述 KSVD break-even，因此结果仍是对 patch chain 有利的显式预算估算。

### 判定

- 若 `RAW-unique lower bound >= 1225`，判为 `RAW_CHAIN_NOT_BIT_COMPETITIVE_WITH_ADJACENCY`；
- 若扣除 identity 与 atom indices 后，每 coefficient 剩余 `<8 bits`，判为 `KSVD_REQUIRES_UNVALIDATED_AGGRESSIVE_QUANTIZATION`；
- 只有 RAW lower bound 更小，且 KSVD 至少有 8 coefficient bits，才能保留“可能 bit-competitive”的表述。

## 5. 总分类

- strict relabel 通过且 bit-rate gate 通过：`PATCH_CHAIN_IDENTITY_RATE_SUPPORTED`；
- 任一基础 gate 失败：`REVISE_PATCH_CHAIN_REPRESENTATION_CLAIM`。

该分类不否定 Beam8 作为完整已知图上的 patch extractor，也不否定局部 dictionary 对下游任务可能有用；它只约束“编号无关图表示”和“优于直接邻接矩阵的压缩”这两个更强主张。
