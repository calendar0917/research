# MolHIV role × attribute binding screen

Protocol: `luyin16-molhiv-role-attribute-binding-screen-v1-official-valid`

The structural role side uses only radius-2 induced topology; the attribute side uses compact OGB atom/bond semantics. Each patch contributes an entity-normalized joint table and its centered residual.

Object audit: **PASS** — 64 official-train graphs × 2 arbitrary relabelings; maximum true-block drift `2.4e-7` (tolerance `1e-6`).

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
| `s` | 0.777459 | 0.000000 |
| `s_structure` | 0.787548 | 0.000000 |
| `s_attribute` | 0.785509 | 0.000000 |
| `s_marginals` | 0.784555 | 0.000000 |
| `s_raw_node` | 0.795457 | 0.000000 |
| `s_raw_edge` | 0.780719 | 0.000000 |
| `s_raw` | 0.795256 | 0.000000 |
| `s_binding_node` | 0.799321 | 0.000000 |
| `s_binding_edge` | 0.791649 | 0.000000 |
| `s_binding` | 0.801747 | 0.000000 |
| `s_raw_shuffled_0` | 0.781062 | 0.000000 |
| `s_binding_shuffled_0` | 0.802066 | 0.000000 |
| `s_raw_shuffled_1` | 0.775540 | 0.000000 |
| `s_binding_shuffled_1` | 0.798629 | 0.000000 |

## Gates

- `binding_vs_shuffle`: mean delta +0.001399, wins 1/1 — **FAIL**
- `binding_vs_marginals`: mean delta +0.017192, wins 1/1 — **PASS**
- `binding_vs_s`: mean delta +0.024288, wins 1/1 — **PASS**
- `binding_vs_raw`: mean delta +0.006491, wins 1/1 — **PASS**
- `raw_vs_shuffle`: mean delta +0.016955, wins 1/1 — **PASS**
- `raw_vs_marginals`: mean delta +0.010701, wins 1/1 — **PASS**
- `raw_vs_s`: mean delta +0.017797, wins 1/1 — **PASS**
- `binding_vs_shuffle_0`: mean delta -0.000319, wins 0/1 — **FAIL**
- `raw_vs_shuffle_0`: mean delta +0.014194, wins 1/1 — **PASS**
- `binding_vs_shuffle_1`: mean delta +0.003118, wins 1/1 — **PASS**
- `raw_vs_shuffle_1`: mean delta +0.019717, wins 1/1 — **PASS**

Interpretation: true-vs-shuffle is evidence that the representation contains role–attribute dependence; only a positive binding-vs-marginals (and S+binding-vs-S) increment is evidence that this dependence helps the present MolHIV task.

Decision: **JOINT_TASK_INCREMENT_PASS_CENTERED_BINDING_UNCONFIRMED**.

Official test was not encoded or evaluated, and no hyperparameter search was run.
