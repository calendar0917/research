# Beam8/Mutagenicity FAIR95 completion audit

> 协议：`tracks/ksvd/docs/KSVD_BEAM8_MUTAGENICITY_FAIR95_COMPLETION_PROTOCOL_20260813.md`  
> 判定：`FAIR95_COMPLETION_BELOW_GATE`  
> selected：`None`

| metric | s10/o3 | s8/o2 |
|---|---:|---:|
| mean_base_patches | 4.0890 | 5.1042 |
| mean_fair_patches | 6.4386 | 7.8206 |
| mean_extra_patches | 2.3496 | 2.7164 |
| mean_extra_fraction | 1.2129 | 1.0262 |
| mean_edge_coverage | 0.9997 | 0.9996 |
| mean_node_coverage | 1.0000 | 1.0000 |
| mean_incident_p10 | 1.0000 | 1.0000 |
| graphs_with_chain_fraction | 0.5771 | 0.6996 |
| mean_directed_chain_edges | 2.4803 | 3.4955 |
| rare_type_graph_coverage | 1.0000 | 1.0000 |

## Per-bond FAIR95 recall

| geometry | bond | recall | mean graph recall | completion new edges |
|---|---:|---:|---:|---:|
| s10_o3 | 1 | 0.9997 | 0.9997 | 29571 |
| s10_o3 | 2 | 0.9995 | 0.9995 | 3001 |
| s10_o3 | 3 | 1.0000 | 1.0000 | 45 |
| s8_o2 | 1 | 0.9996 | 0.9996 | 28859 |
| s8_o2 | 2 | 0.9991 | 0.9993 | 2718 |
| s8_o2 | 3 | 1.0000 | 1.0000 | 50 |

## Frozen checks

### s10_o3

- all_reached：`True`；
- edge_node_coverage：`True`；
- rare_aggregate_recall：`True`；
- rare_graph_coverage：`True`；
- extra_patch_fraction：`False`；
- chain_graph_fraction：`True`；
### s8_o2

- all_reached：`True`；
- edge_node_coverage：`True`；
- rare_aggregate_recall：`True`；
- rare_graph_coverage：`True`；
- extra_patch_fraction：`False`；
- chain_graph_fraction：`True`；

## Boundary

- completion 不读取 bond type；新增 patch 都是新 segment。
- 分类必须同时报告 BASE 与 FAIR95，不能把 completion 伪装成连续 Beam8 chain。
