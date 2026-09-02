# KSVD atom 的真实 patch 语义审计（2026-07-30）

> 分类增益不能证明 atom 是合法结构词汇。本报告直接把每个 atom 投影到训练折真实 patch，并检查可投影性、非负性、近邻结构纯度和跨折语义一致性。

## 协议

- 数据集：`REDDIT-BINARY`，variant=`raw`，patch=`R2`。
- 5-fold × seeds [0, 1, 2]；每折 16 atoms；KSVD iterations=10。
- 每个 atom 检索 top-10 真实训练 patch。
- `projectable proxy`：符号对齐后 nearest cosine ≥0.95 且 negative mass ≤0.05；它只是向量层面的必要条件，不等价于严格图可解码。
- 结构 signature 包含节点/边数、度序列、三角形、连通分量和 WL hash；不是严格 canonical isomorphism code。

## REDDIT-BINARY/raw/R2

| 字典 | nearest cosine | negative mass | top-k signature purity | projectable proxy | nearest-signature cross-run agreement |
|---|---:|---:|---:|---:|---:|
| ksvd | 0.8627 ± 0.1001 | 0.0769 | 0.8129 | 0.1583 | 0.4125 |
| clustered_real_patch | 1.0000 ± 0.0000 | 0.0000 | 0.7571 | 1.0000 | 0.4345 |

- KSVD matched-vector cosine：0.8126；top-k mode signature agreement：0.4446。
- Real-patch matched-vector cosine：0.7262；top-k mode signature agreement：0.4435。
- KSVD nearest signatures 数量：91；最常见：`[['n4|m3|d3,1,1,1|t0|c1|wl68f38cbd23a2edb067d0f8d42838916c', 15], ['n3|m2|d2,1,1|t0|c1|wl7027fd65cc86d3efaefa1911a201c652', 15], ['n6|m5|d5,1,1,1,1,1|t0|c1|wlcb5656bcce45f2fde06fbd6a0bd0e26c', 14], ['n5|m4|d4,1,1,1,1|t0|c1|wl2fb48ebbfad1620deb8ade80c84f5b90', 14], ['n7|m6|d6,1,1,1,1,1,1|t0|c1|wld8ba19bed642c83724f7258a50d5e1c4', 13]]`。
- nearest patch 中 clique 占比：KSVD=0.0083，real-patch=0.0042；最大观测 patch size=24，KSVD 达到该上限的比例=0.4375。

## 判断规则

1. vector cosine 高但 signature agreement 低，表示字典只在 WL 向量空间稳定，未形成稳定 graphlet 语义。
2. negative mass 高或 nearest cosine 低，说明普通欧氏 KSVD atom 很难直接解释为合法计数/图结构。
3. clustered real-patch 若语义稳定性不低于 KSVD，则继续使用自由向量 atom 的必要性不足。
4. 本审计不使用标签；它只能判断 vocabulary plausibility，不能替代下游任务评估。

## 当前结论

- `REDDIT-BINARY/raw/R2`：KSVD projectable proxy=0.158，nearest-signature agreement=0.412；real-patch 对应值为 1.000/0.435。
  clique 仅占 0.8%，但 43.8% 的 nearest prototypes 达到观测最大 size=24；常见原型仍受 sampler 尺度/星形结构主导。
- clustered real-patch dictionary 天然合法、可回指，且语义稳定性未弱于 KSVD；普通欧氏 KSVD 暂无不可替代性证据。
- 因此当前最多支持“WL patch 空间中的 coding directions”，不支持“已学得跨折稳定、合法、可组合的 graphlet vocabulary”。
