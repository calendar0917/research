# Beam8/Mutagenicity MAXCHAIN audit

> 协议：`tracks/ksvd/docs/KSVD_BEAM8_MUTAGENICITY_MAXCHAIN_PROTOCOL_20260813.md`  
> 判定：`MAXCHAIN_COST_OR_COVERAGE_BELOW_GATE`

| metric | value |
|---|---:|
| mean_base_patches | 5.1042 |
| mean_max_patches | 90.7459 |
| mean_extra_patches | 85.6417 |
| mean_extra_fraction | 16.0166 |
| graphs_with_chain_base | 0.6996 |
| graphs_with_chain_max | 0.6996 |
| mean_chain_edges_base | 3.4955 |
| mean_chain_edges_max | 89.1372 |
| mean_edge_recall_base | 0.7364 |
| mean_edge_recall_max | 0.8615 |
| mean_node_coverage_base | 0.7402 |
| mean_node_coverage_max | 0.8660 |
| rare_graph_coverage_max | 0.6596 |

- type recall BASE：`[0.7384524886877828, 0.8801068441564128, 0.5454545454545454]`；
- type recall MAXCHAIN：`[0.9175927601809954, 0.9530586329202609, 0.6181818181818182]`；
- new type edges：`[19795, 1666, 8]`；

## Checks

- rare_recall_ge_80：`False`；
- rare_graph_coverage_ge_80：`False`；
- extra_fraction_le_100：`False`；
- chain_fraction_noninferior：`True`；
