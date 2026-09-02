# KSVD atom 的真实 patch 语义审计（2026-07-30）

> 分类增益不能证明 atom 是合法结构词汇。本报告直接把每个 atom 投影到训练折真实 patch，并检查可投影性、非负性、近邻结构纯度和跨折语义一致性。

## 协议

- 数据集：`IMDB-BINARY, IMDB-MULTI`，variant=`cleaned`，patch=`B0`。
- 5-fold × seeds [0, 1, 2]；每折 16 atoms；KSVD iterations=10。
- 每个 atom 检索 top-10 真实训练 patch。
- `projectable proxy`：符号对齐后 nearest cosine ≥0.95 且 negative mass ≤0.05；它只是向量层面的必要条件，不等价于严格图可解码。
- 结构 signature 包含节点/边数、度序列、三角形、连通分量和 WL hash；不是严格 canonical isomorphism code。

## IMDB-BINARY/cleaned/B0

| 字典 | nearest cosine | negative mass | top-k signature purity | projectable proxy | nearest-signature cross-run agreement |
|---|---:|---:|---:|---:|---:|
| ksvd | 0.9481 ± 0.0683 | 0.0452 | 0.9304 | 0.5542 | 0.7167 |
| clustered_real_patch | 1.0000 ± 0.0000 | 0.0000 | 0.9375 | 1.0000 | 0.7333 |

- KSVD matched-vector cosine：0.8374；top-k mode signature agreement：0.7226。
- Real-patch matched-vector cosine：0.8257；top-k mode signature agreement：0.7458。
- KSVD nearest signatures 数量：51；最常见：`[['n7|m21|d6,6,6,6,6,6,6|t35|c1|wle372ad80e7a606d55dadcc076a7748e5', 16], ['n12|m66|d11,11,11,11,11,11,11,11,11,11,11,11|t220|c1|wla242e3d918bccecfe9af99a03d43b8dc', 15], ['n9|m36|d8,8,8,8,8,8,8,8,8|t84|c1|wl513f813ae6b0c87ac12515e5c4ba99fb', 15], ['n10|m45|d9,9,9,9,9,9,9,9,9,9|t120|c1|wl55d19345812813a686d3896d994935f1', 15], ['n6|m15|d5,5,5,5,5,5|t20|c1|wle5be41ac873fd68e9803271420963b83', 15]]`。
- nearest patch 中 clique 占比：KSVD=0.5792，real-patch=0.5875；最大观测 patch size=12，KSVD 达到该上限的比例=0.2125。

## IMDB-MULTI/cleaned/B0

| 字典 | nearest cosine | negative mass | top-k signature purity | projectable proxy | nearest-signature cross-run agreement |
|---|---:|---:|---:|---:|---:|
| ksvd | 0.9376 ± 0.0817 | 0.0534 | 0.9117 | 0.4958 | 0.6702 |
| clustered_real_patch | 1.0000 ± 0.0000 | 0.0000 | 0.9383 | 1.0000 | 0.6923 |

- KSVD matched-vector cosine：0.8268；top-k mode signature agreement：0.6810。
- Real-patch matched-vector cosine：0.8006；top-k mode signature agreement：0.6964。
- KSVD nearest signatures 数量：52；最常见：`[['n5|m10|d4,4,4,4,4|t10|c1|wlbde135a811cfdf794cdaa83144f8362f', 15], ['n7|m21|d6,6,6,6,6,6,6|t35|c1|wle372ad80e7a606d55dadcc076a7748e5', 15], ['n12|m66|d11,11,11,11,11,11,11,11,11,11,11,11|t220|c1|wla242e3d918bccecfe9af99a03d43b8dc', 15], ['n11|m55|d10,10,10,10,10,10,10,10,10,10,10|t165|c1|wl214ca21f4f97992978d89529accb534c', 15], ['n6|m15|d5,5,5,5,5,5|t20|c1|wle5be41ac873fd68e9803271420963b83', 15]]`。
- nearest patch 中 clique 占比：KSVD=0.5833，real-patch=0.5875；最大观测 patch size=12，KSVD 达到该上限的比例=0.2458。

## 判断规则

1. vector cosine 高但 signature agreement 低，表示字典只在 WL 向量空间稳定，未形成稳定 graphlet 语义。
2. negative mass 高或 nearest cosine 低，说明普通欧氏 KSVD atom 很难直接解释为合法计数/图结构。
3. clustered real-patch 若语义稳定性不低于 KSVD，则继续使用自由向量 atom 的必要性不足。
4. 本审计不使用标签；它只能判断 vocabulary plausibility，不能替代下游任务评估。

## 当前结论

- `IMDB-BINARY/cleaned/B0`：KSVD projectable proxy=0.554，nearest-signature agreement=0.717；real-patch 对应值为 1.000/0.733。
  nearest prototypes 中 57.9% 是 clique，主要按 clique/patch size 区分，容易复述规模与度结构。
- `IMDB-MULTI/cleaned/B0`：KSVD projectable proxy=0.496，nearest-signature agreement=0.670；real-patch 对应值为 1.000/0.692。
  nearest prototypes 中 58.3% 是 clique，主要按 clique/patch size 区分，容易复述规模与度结构。
- clustered real-patch dictionary 天然合法、可回指，且语义稳定性未弱于 KSVD；普通欧氏 KSVD 暂无不可替代性证据。
- 因此当前最多支持“WL patch 空间中的 coding directions”，不支持“已学得跨折稳定、合法、可组合的 graphlet vocabulary”。
