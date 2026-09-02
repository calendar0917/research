# Attributed Beam8 node-incidence feasibility audit

> 协议：`tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_NODE_INCIDENCE_FEASIBILITY_PROTOCOL_20260814.md`  
> 判定：`BEAM8_NODE_INCIDENCE_READY_FOR_LOW_CAPACITY_CLASSIFICATION`

| metric | value |
|---|---:|
| feature_dim | 443 |
| mean_node_coverage | 0.737610 |
| full_coverage_graph_fraction | 0.017985 |
| mean_incidence_per_node | 1.103224 |
| mean_incidence_per_covered_node | 1.450040 |
| multi_patch_node_fraction | 0.257597 |
| chain_active_node_fraction | 0.561573 |
| same_atom_pair_disambiguation | 0.803371 |
| non_singleton_orbit_node_fraction | 0.365866 |

## Relabel audit

- direct_node_equivariance：`0.420573`；
- node_equivariance：`1.000000`；
- graph_readout：`1.000000`；

## Checks

- usable_node_coverage：`True`；
- nontrivial_multi_patch：`True`；
- nontrivial_chain_context：`True`；
- atom_complement：`True`；
- node_equivariance：`True`；
- graph_readout_invariance：`True`；

## Interpretation boundary

- 本轮未使用分类标签；通过只授权低容量 GINE fusion screen。
- graph readout invariant 但 node equivariance 不通过时，不能直接接节点级 GNN。
