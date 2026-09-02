# Beam8/Mutagenicity typed-anchor audit

> 协议：`tracks/ksvd/docs/KSVD_BEAM8_MUTAGENICITY_TYPED_ANCHOR_PROTOCOL_20260813.md`  
> 判定：`TYPED_ANCHOR_READY_FOR_CLASSIFICATION`

- mean/p95/max anchors：`0.0530` / `1.0` / `2`；
- BASE unchanged：`True`；
- aggregate recall BASE：`[0.7384524886877828, 0.8801068441564128, 0.5454545454545454]`；
- aggregate recall ANCHOR：`[0.7445701357466064, 0.8894338135481893, 0.9090909090909091]`；
- graph coverage BASE：`[1.0, 0.9525583705911574, 0.5851063829787234]`；
- graph coverage ANCHOR：`[1.0, 1.0, 1.0]`；

## Checks

- all_type_graph_coverage：`True`；
- rare_aggregate_recall：`True`；
- mean_anchors：`True`；
- p95_anchors：`True`；
- base_unchanged：`True`；

## Boundary

- anchor 使用 bond type 输入，但不使用 graph label；原 Beam8 BASE chain 完全不变。
- anchor 是显式新 segment，不能解释为连续 chain relation 收益。
