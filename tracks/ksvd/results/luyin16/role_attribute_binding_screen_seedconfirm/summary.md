# MolHIV role × attribute binding screen

Protocol: `luyin16-molhiv-role-attribute-binding-screen-v1-seed-confirm`

The structural role side uses only radius-2 induced topology; the attribute side uses compact OGB atom/bond semantics. Each patch contributes an entity-normalized joint table and its centered residual.

## Feature schema

- node role dimension: 48
- edge role dimension: 12
- compact atom attribute dimension: 48
- compact bond attribute dimension: 13
- role = shell × induced-degree-bin × cycle-membership (nodes), or shell-pair × cycle-membership (edges)
- centered binding = joint − role marginal × attribute marginal, averaged over rooted patches
- shuffle = independent within-patch permutation of node and edge attribute rows

## Fold scores

| view | mean validation ROC-AUC | fold std |
|---|---:|---:|
| `s` | 0.698143 | 0.020161 |
| `s_structure` | 0.706475 | 0.011792 |
| `s_attribute` | 0.705051 | 0.032373 |
| `s_marginals` | 0.704788 | 0.020201 |
| `s_raw_node` | 0.719187 | 0.030657 |
| `s_raw_edge` | 0.716303 | 0.014811 |
| `s_raw` | 0.717031 | 0.029461 |
| `s_binding_node` | 0.740456 | 0.031201 |
| `s_binding_edge` | 0.715753 | 0.021159 |
| `s_binding` | 0.738023 | 0.030097 |
| `s_raw_shuffled_0` | 0.711672 | 0.015520 |
| `s_binding_shuffled_0` | 0.685749 | 0.007608 |
| `s_raw_shuffled_1` | 0.697832 | 0.020601 |
| `s_binding_shuffled_1` | 0.662313 | 0.009853 |

## Gates

- `binding_vs_shuffle`: mean delta +0.063992, wins 3/3 — **PASS**
- `binding_vs_marginals`: mean delta +0.033235, wins 3/3 — **PASS**
- `binding_vs_s`: mean delta +0.039880, wins 3/3 — **PASS**
- `binding_vs_raw`: mean delta +0.020992, wins 2/3 — **PASS**
- `raw_vs_shuffle`: mean delta +0.012279, wins 2/3 — **PASS**
- `raw_vs_marginals`: mean delta +0.012244, wins 2/3 — **PASS**
- `raw_vs_s`: mean delta +0.018888, wins 3/3 — **PASS**
- `binding_vs_shuffle_0`: mean delta +0.052274, wins 3/3 — **PASS**
- `raw_vs_shuffle_0`: mean delta +0.005359, wins 2/3 — **PASS**
- `binding_vs_shuffle_1`: mean delta +0.075710, wins 3/3 — **PASS**
- `raw_vs_shuffle_1`: mean delta +0.019199, wins 3/3 — **PASS**

Interpretation: true-vs-shuffle is evidence that the representation contains role–attribute dependence; only a positive binding-vs-marginals (and S+binding-vs-S) increment is evidence that this dependence helps the present MolHIV task.

Decision: **JOINT_AND_CENTERED_BINDING_TASK_INCREMENT_PASS**.

Official test was not encoded or evaluated, and no hyperparameter search was run.
