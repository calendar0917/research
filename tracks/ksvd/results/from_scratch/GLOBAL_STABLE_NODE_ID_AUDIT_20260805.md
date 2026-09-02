# Graph-global stable node ID 与 Beam8 记录稳定性审计

> 日期：2026-08-05
> 协议：`tracks/ksvd/docs/KSVD_GLOBAL_STABLE_NODE_ID_PROTOCOL_20260805.md`
> 判定：`ADOPT_GLOBAL_STABLE_ID_PREORDER`

## 1. 判定

72 graphs × 3 relabel permutations。

- class invariant：`True`；
- practical stable-ID gate：`True`；checks：`{'singleton_fraction_at_least_090': True, 'singleton_id_match_at_least_0999': True, 'canonical_adjacency_match_at_least_099': True}`；
- stable-preorder sampler gate：`True`；checks：`{'all_invariants': True, 'edge_coverage_not_worse_by_001': True, 'pair_coverage_not_worse_by_001': True, 'exact_chain_gain_at_least_050': True, 'rooted_vector_gain_at_least_030': True}`；
- selected practical ID：`GLOBAL_WL`（GLOBAL/ROOTED order match=`1.0000`）。

限定：fully-singleton 图可获得具体节点身份的严格 replay；ambiguous 图只应主张 equivalence-class、patch vector 与 transition 稳定，不能把对称节点的 concrete ID 称为唯一。

## 2. Stable ID

| method | singleton nodes | fully-singleton graphs | largest class mean/max | class match | singleton ID match | all-node concrete match | canonical adjacency match | tied pairs | tied-pair swap-auto | sec/graph |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GLOBAL_WL | 0.9939 | 0.9028 | 1.0972/2 | 1.0000 | 1.0000 | 0.9972 | 1.0000 | 33 | 1.0000 | 0.0262 |
| ROOTED_WL | 0.9939 | 0.9028 | 1.0972/2 | 1.0000 | 1.0000 | 0.9972 | 1.0000 | 33 | 1.0000 | 0.0341 |

Singleton ID 才是可证明的 unique stable ID；non-singleton class 内的 concrete ID 只是 opaque handle。

## 3. Frozen-cover records

| record IDs | ordered membership | rooted membership | class membership | transition | residual concrete | residual class |
|---|---:|---:|---:|---:|---:|---:|
| GLOBAL_WL | 0.9790 | 0.9790 | 1.0000 | 0.9934 | 0.9537 | 1.0000 |
| INPUT_ID | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ROOTED_WL | 0.9790 | 0.9790 | 1.0000 | 0.9934 | 0.9537 | 1.0000 |

## 4. Relabel + Beam8 resampling

| branch | abstract exact chain | abstract patch Jaccard | class chain | direct-coordinate replay (control) | rooted vector rows | transition map | edge/pair cover | edge/pair delta | RAW RMSE delta | sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RAW_BEAM8 | 0.0093 | 0.2220 | 0.0093 | 0.0000 | 0.0875 | 0.0483 | 0.8747/0.5268 | 0.0069/0.0059 | 0.0064 | 0.2904 |
| STABLE_ID_PREORDER_BEAM8 | 0.9491 | 0.9973 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8752/0.5290 | 0.0000/0.0000 | 0.0000 | 0.2922 |

Abstract-node exact-chain gain：`0.9398`；rooted-vector-row gain：`0.9125`。

`direct-coordinate replay` 不参与上述 gain：stable-preorder 行表示 canonical preprocessing 后的确定性 consistency；RAW 行没有共同 canonical 坐标，只是未映射 input-index control，不应解释为稳定性指标。

## 5. 分层

| family | degree | stable-ID singleton | stable exact chain | stable vector rows | stable edge cover |
|---|---:|---:|---:|---:|---:|
| block | 15 | 1.0000 | 1.0000 | 1.0000 | 0.8458 |
| block | 20 | 1.0000 | 1.0000 | 1.0000 | 0.9060 |
| block | 25 | 0.9450 | 0.5417 | 1.0000 | 0.9474 |
| regular | 15 | 1.0000 | 1.0000 | 1.0000 | 0.7960 |
| regular | 20 | 1.0000 | 1.0000 | 1.0000 | 0.8560 |
| regular | 25 | 1.0000 | 1.0000 | 1.0000 | 0.8792 |
| small_world | 15 | 1.0000 | 1.0000 | 1.0000 | 0.8630 |
| small_world | 20 | 1.0000 | 1.0000 | 1.0000 | 0.8952 |
| small_world | 25 | 1.0000 | 1.0000 | 1.0000 | 0.8884 |

### 5.1 Fully-singleton 与 ambiguous graphs

| branch | stratum | trials | abstract exact chain | abstract Jaccard | class chain | canonical consistency | vectors | transition |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| RAW_BEAM8 | fully_singleton | 195 | 0.0103 | 0.2179 | 0.0103 | 0.0000 | 0.0667 | 0.0507 |
| RAW_BEAM8 | ambiguous | 21 | 0.0000 | 0.2598 | 0.0000 | 0.0000 | 0.2815 | 0.0260 |
| STABLE_ID_PREORDER_BEAM8 | fully_singleton | 195 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| STABLE_ID_PREORDER_BEAM8 | ambiguous | 21 | 0.4762 | 0.9723 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## 6. 解释边界

- global stable ID 用于 membership/transition/stitching/residual；45D KSVD local slots 仍使用 rooted canonical，不按 global ID 排序。
- 本轮检验的是同图 numeric relabel，不是加减边后的 perturbation stability。
- stable ID 只在单张图内部有意义，不赋予跨图相同 ID 共同语义。
- 对称节点只能获得 stable class；具体 labeled identity 仍需 opaque sidecar。要恢复输入的原始 labeled adjacency，仍需 canonical-ID→input-ID 映射；稳定 ID 不会免费消除该 sidecar。
- 使用 SHA-256 digest 承载离散 WL signature；重编号稳定性是实测严格匹配，理论上仍采用密码学碰撞可忽略假设，不把 digest 称为无条件数学证明。
- stable-preorder 不改变 Beam8 objective；收益若存在来自消除输入 numeric-order 与 tie-list ordering 的影响。
