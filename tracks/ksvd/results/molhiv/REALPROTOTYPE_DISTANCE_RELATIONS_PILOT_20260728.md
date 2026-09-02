# Real-prototype exact-distance relations pilot（2026-07-28）

## 1. 数据与边界

- 使用 8,000-graph development subset 中的 6,400 个 official-train graphs。
- 三个 official-train-only Bemis–Murcko scaffold outer folds；固定 seed 0、30 epochs。
- official valid/test 编码与评估次数均为 **0**。
- 冻结 broad-pool `farthest` 与 `scaffold_facility` vocabulary，不重新选择 prototype。

## 2. 方法

- 对 node→prototype positive-cosine top-3 assignment 构造 `C[d] = Z^T M[d] Z`。
- 距离 bins：self、1、2、3+、disconnected；关系向量维度 2,650。
- 关系支路只作为 zero-initialized gated scalar residual，不含 supervised message passing。
- matched shuffled control 固定分子距离矩阵并置乱节点 assignment，保持 prototype marginals。

## 3. 三折结果

| Vocabulary | Model | Fold 0 | Fold 1 | Fold 2 | Mean |
|---|---|---:|---:|---:|---:|
| farthest | occurrence | 0.750784 | 0.703799 | 0.776604 | 0.743729 |
| farthest | exact relation | 0.740286 | 0.697899 | 0.798534 | 0.745573 |
| farthest | distance-shuffled | 0.701045 | 0.694631 | 0.771308 | 0.722328 |
| scaffold | occurrence | 0.750577 | 0.687013 | 0.793251 | 0.743614 |
| scaffold | exact relation | 0.741029 | 0.680017 | 0.789274 | 0.736773 |
| scaffold | distance-shuffled | 0.718095 | 0.657742 | 0.759901 | 0.711913 |

单 vocabulary promotion gate：

- farthest relation − occurrence：`+0.001844`，赢 `1/3`；relation − shuffled `+0.023245`；promotion = **False**。
- scaffold relation − occurrence：`-0.006840`，赢 `0/3`；relation − shuffled `+0.024861`；promotion = **False**。

## 4. Vocabulary-pair ensemble

| Ensemble | Fold 0 | Fold 1 | Fold 2 | Mean |
|---|---:|---:|---:|---:|
| occurrence pair | 0.766886 | 0.699399 | 0.800924 | 0.755736 |
| exact-relation pair | 0.742854 | 0.697015 | 0.794678 | 0.744849 |
| shuffled-relation pair | 0.708227 | 0.672363 | 0.767485 | 0.716025 |

Pair relation − occurrence：`-0.010887`，赢 `0/3`；relation − shuffled `+0.028824`；promotion = **False**。

## 5. 审计与判断

- occurrence baseline 最大复现误差：farthest `0.000e+00`，scaffold `0.000e+00`，pair `0.000e+00`。
- disconnected graphs by fold：`[503, 503, 503]`。
- exact shortest-path relation 总 promotion：**False**。

当前 exact shortest-path summary residual 未通过预注册 gate。它不能证明 relational composition
整体无效，但说明当前全量 upper-triangle + 单标量 residual 的参数化不是可靠增益。下一步应优先做
更强的低秩/分尺度 relation readout 或 diffusion operator，并保留相同 shuffled control。