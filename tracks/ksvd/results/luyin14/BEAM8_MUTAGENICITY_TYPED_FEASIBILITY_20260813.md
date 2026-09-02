# Beam8/Mutagenicity typed feasibility audit

> 协议：`tracks/ksvd/docs/KSVD_BEAM8_MUTAGENICITY_TYPED_FEASIBILITY_PROTOCOL_20260813.md`  
> 判定：`TYPED_GEOMETRY_GATE_FAILED`

| metric | s10/o3 | s8/o2 |
|---|---:|---:|
| mean_patches | 4.0890 | 5.1042 |
| single_patch_fraction | 0.4007 | 0.2850 |
| graphs_with_chain_fraction | 0.5771 | 0.6996 |
| mean_directed_chain_edges | 2.4803 | 3.4955 |
| partial_component_fraction | 0.2548 | 0.1810 |
| mean_edge_coverage | 0.7399 | 0.7364 |
| edge_coverage_p10 | 0.4545 | 0.4444 |
| mean_node_coverage | 0.7417 | 0.7402 |
| rare_type_graph_coverage | 0.6064 | 0.5851 |

## Per-bond coverage

| geometry | bond type | edges | aggregate recall | mean graph recall | graphs |
|---|---:|---:|---:|---:|---:|
| s10_o3 | 1 | 110500 | 0.7321 | 0.7142 | 4337 |
| s10_o3 | 2 | 22837 | 0.8681 | 0.8387 | 4026 |
| s10_o3 | 3 | 110 | 0.5909 | 0.5957 | 94 |
| s8_o2 | 1 | 110500 | 0.7385 | 0.7099 | 4337 |
| s8_o2 | 2 | 22837 | 0.8801 | 0.8427 | 4026 |
| s8_o2 | 3 | 110 | 0.5455 | 0.5691 | 94 |

## Binding margins

- s10_o3 node TRUE−SHUFFLED：`+0.0691`；new-bond TRUE−SHUFFLED：`+0.0553`。
- s8_o2 node TRUE−SHUFFLED：`+0.0700`；new-bond TRUE−SHUFFLED：`+0.0459`。

## Frozen checks

- single_patch_relative_reduction_ge_20pct：`True`；
- chain_graph_fraction_noninferior：`True`；
- mean_chain_edges_relative_gain_ge_20pct：`True`；
- edge_coverage_noninferior_1pt：`True`；
- partial_component_fraction_noninferior：`True`；
- all_bond_types_noninferior_2pt：`False`；
- node_chain_margin_positive：`True`；
- new_bond_chain_margin_positive：`True`；
- rare bond evidence sufficient：`False`；

## Boundary

- cover 不读取 bond type；bond labels 只用于无标签 coverage/binding audit。
- graph labels 未进入本审计。
