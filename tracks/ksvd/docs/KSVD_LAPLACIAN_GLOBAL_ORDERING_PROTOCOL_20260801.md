# Continuous cover：Laplacian global ordering 审计协议

> 日期：2026-08-01  
> 状态：正式结果不可见前冻结  
> 前置：construction-order KSVD 有中等重构收益；slot persistence 与线性 transition decoder 均未通过。

## 1. 研究问题

Laplacian positional encoding 常被描述为给节点“位置”。本轮把这个说法拆成可审计命题：

> normalized-Laplacian 的 Fiedler coordinate 能否在单张图内部提供重编号无关、对一次小结构扰动足够稳定的全局 rank，并在 node sets、coverage、folds 和 KSVD 容量完全相同时，通过纯 local ordering 改善 held-out patch/stitch reconstruction？

本轮不声称不同图的谱轴天然对齐。没有跨图共享 node identity 时，“第 0.2 谱坐标”不等于共同绝对语义。

## 2. Frozen spectral coordinate

对每张完整图计算 symmetric normalized Laplacian：

```text
L = I - D^{-1/2} A D^{-1/2}
```

取第二小特征值对应的 Fiedler vector。符号按以下规则确定：绝对值最大的分量必须为正；随后用稳定排序得到全图 node rank。

不扫描 combinatorial/normalized Laplacian，不扫描 eigenvector 数，不使用 labels。

## 3. Matched branches

### construction_order

冻结 target-edge bridge cover 的原始 construction ordering。

### fiedler_global_order

保持每个 patch 的 node set、center、target edge、segment 和 patch count 不变，只按该节点在完整图中的 Fiedler rank 从小到大重排 local slots。

同一节点拥有一个 graph-global rank，但它在不同 patch 中的 local slot 仍可随 patch 子集改变。

## 4. 数据与 KSVD

完全复用 stitched reconstruction audit：

```text
graph_bank_seed = 810001
cover_seed = 830101
72 graphs, deterministic 3-fold
s=10, overlap=5
K=24, T=3, T_min=1, updates=25
PCA rank=3
threshold=0.5
```

每个 branch 独立在相同 train graph IDs 上拟合 coordinate mean、INIT、FINAL 和 PCA3；test graphs 只编码。

## 5. Coordinate audits

每张图额外检查：

1. 对随机 node relabeling 重新计算 Fiedler vector/rank，映回原节点后 rank 必须完全一致；
2. 做一次保持 degree sequence 和 connectivity 的 double-edge swap；
3. 报告扰动前后所有 node-pair 相对次序保持率；
4. 同时报告 sign-invariant pair-order agreement，区分谱轴本身变化与单纯方向翻转；
5. 报告 relative eigengap `(lambda_3-lambda_2)/lambda_2`。

冻结 coordinate stability gate：

```text
mapped relabel rank match = 1.0 for all graphs
mean canonical pair-order agreement after one swap >= 0.80
at least 2/3 graphs have canonical pair-order agreement >= 0.80
```

## 6. Representation invariants

- 两个 branches position-wise node sets 完全相同；
- node/edge/pair coverage 完全相同；
- RAW observed RMSE/disagreement = 0；
- spectral ordering 在 mapped replay 后 patch adjacency/transition maps = 1；
- train/test graph IDs 完全匹配。

## 7. Registered reconstruction gate

`fiedler_global_order` 的 comparative gate 必须同时满足：

1. FINAL observed RMSE 比 construction FINAL 至少低 `0.02` relative；
2. FINAL overlap disagreement不高于 construction FINAL；
3. FINAL observed F1 不低于 construction FINAL；
4. FINAL full edge recall 不低于 construction FINAL `-0.01`；
5. FINAL observed RMSE 低于自己的 PCA3；
6. 3/3 folds FINAL patch error < INIT。

强 PASS 仍额外要求 mean patch INIT→FINAL reduction >= `0.10`、nondead >=20/24、maximum activation share <0.50。

## 8. 判定

- `PASS_LAPLACIAN_GLOBAL_ORDERING`：coordinate stability、comparative 和强 KSVD gates 全部通过；
- `LAPLACIAN_ORDERING_HELPS_BELOW_KSVD_GATE`：坐标稳定且 comparative 通过，但 10% optimization gate 未通过；
- `STABLE_LAPLACIAN_COORDINATE_NO_RECON_GAIN`：坐标稳定，但重排没有 reconstruction added value；
- `LAPLACIAN_ORDERING_HELP_BUT_UNSTABLE`：comparative 通过，但扰动稳定性 gate 失败；
- `REJECT_UNSTABLE_LAPLACIAN_ORDERING`：坐标不稳定且 reconstruction 无增益；
- `FAIL_LAPLACIAN_ORDERING_INVARIANTS`：node sets、coverage、RAW、relabel 或 replay contract 失败。

## 9. 边界

通过只说明 Fiedler rank 是当前图族中的有用 graph-internal coordinate。它不证明跨图绝对位置存在，也不等价于 LapPE 输入 GNN/Transformer 后会改善分类。
