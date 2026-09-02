# Attributed Beam8/Mutagenicity geometry audit

> 协议：`tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_MUTAGENICITY_GEOMETRY_PROTOCOL_20260813.md`  
> 判定：`TYPED_GEOMETRY_GATE_FAILED`

| metric | s10/o3 | s8/o2 |
|---|---:|---:|
| mean_patches | 4.0844 | 5.0881 |
| single_patch_fraction | 0.3977 | 0.2910 |
| graphs_with_chain_fraction | 0.5799 | 0.6929 |
| mean_directed_chain_edges | 2.4757 | 3.4794 |
| partial_component_fraction | 0.2556 | 0.1853 |
| mean_edge_coverage | 0.7404 | 0.7334 |
| edge_coverage_p10 | 0.4583 | 0.4444 |
| mean_node_coverage | 0.7424 | 0.7376 |
| rare_type_graph_coverage | 0.5106 | 0.4787 |

## Per-bond recall

| geometry | type | recall | graph recall |
|---|---:|---:|---:|
| s10_o3 | 1 | 0.7330 | 0.7156 |
| s10_o3 | 2 | 0.8624 | 0.8334 |
| s10_o3 | 3 | 0.5182 | 0.5053 |
| s8_o2 | 1 | 0.7366 | 0.7063 |
| s8_o2 | 2 | 0.8793 | 0.8403 |
| s8_o2 | 3 | 0.4545 | 0.4574 |

## Binding

- s10_o3 node margin `+0.0695`；new-bond margin `+0.0548`。
- s8_o2 node margin `+0.0712`；new-bond margin `+0.0461`。

## Checks

- single_patch_relative_reduction_ge_20pct：`True`；
- chain_graph_fraction_noninferior：`True`；
- mean_chain_edges_relative_gain_ge_20pct：`True`；
- edge_coverage_noninferior_1pt：`True`；
- partial_component_fraction_noninferior：`True`；
- all_bond_types_noninferior_2pt：`False`；
- node_chain_margin_positive：`True`；
- new_bond_chain_margin_positive：`True`；
