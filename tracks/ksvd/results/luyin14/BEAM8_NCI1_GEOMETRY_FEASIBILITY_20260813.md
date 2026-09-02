# Beam8/NCI1 geometry feasibility audit

> 协议：`tracks/ksvd/docs/KSVD_BEAM8_NCI1_GEOMETRY_FEASIBILITY_PROTOCOL_20260813.md`  
> 判定：`S8_O2_READY_FOR_FIXED_CLASSIFICATION`

| metric | s10/o3 | s8/o2 |
|---|---:|---:|
| mean_patches | 3.9229 | 4.9182 |
| single_patch_fraction | 0.2217 | 0.1061 |
| graphs_with_chain_fraction | 0.7343 | 0.8623 |
| mean_directed_chain_edges | 2.7358 | 3.7311 |
| partial_component_fraction | 0.2388 | 0.1377 |
| graphs_with_partial_component_fraction | 0.2667 | 0.1367 |
| mean_edge_coverage | 0.7832 | 0.7898 |
| edge_coverage_p10 | 0.5000 | 0.6249 |
| mean_node_coverage | 0.8007 | 0.8112 |
| nonchain_overlap_graph_fraction | 0.7195 | 0.8603 |

## Attribute relation binding

- s10_o3：TRUE `0.9707`，SHUFFLED `0.9291`，margin `+0.0416`，usable graphs `3018`。
- s8_o2：TRUE `0.9604`，SHUFFLED `0.9272`，margin `+0.0333`，usable graphs `3544`。

## Frozen checks

- single_patch_relative_reduction_ge_20pct：`True`；
- chain_graph_fraction_noninferior：`True`；
- mean_chain_edges_relative_gain_ge_20pct：`True`；
- edge_coverage_noninferior_1pt：`True`；
- partial_component_fraction_noninferior：`True`；
- attribute_true_minus_shuffled_positive：`True`；

## Boundary

- 本审计没有使用 graph labels，也没有训练分类器或字典。
- 只有全部冻结 checks 通过才允许用 s8/o2 重跑分类；不得以分类结果反向选择 geometry。
