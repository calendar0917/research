# PEC-v0 — Gate 0 (CPU correctness)

Verdict: **PASS (8/8 checks)**

Data-free plus toy-graph checks; no molecule labels; official valid/test not read.

| check | observation |
|---|---|
| purity | codes finite; node/edge basis unchanged under a chemistry swap |
| chemistry_isolation | atom_idx chemistry-only; node/edge basis topology-only |
| assignment_sensitivity | chemistry-placement shuffle changes E_i, max abs 0.0991 |
| environment_freeze | E_i bit-identical under pair mutation: True |
| static_contract | environment module called exactly once; BAG forward finite (True) |
| relabel_invariance | prediction max abs diff 0.0 |
| sparse_correctness | max l0 node 4, edge 4 (s=4) |
| gradients | d_edge 0.00708; d_node 0.00384; environment 1.3; pair 0.333; reader 5.72 |

| arm | parameters |
|---|---:|
| CK_sparse | 94049 |
| CD_dense | 94049 |
| C0_coarse | 94036 |

`official_test_loaded = false`
