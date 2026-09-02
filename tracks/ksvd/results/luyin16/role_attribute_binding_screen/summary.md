# MolHIV role × attribute binding screen

Protocol: `luyin16-molhiv-role-attribute-binding-screen-v1`

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
| `s` | 0.693857 | 0.017439 |
| `s_structure` | 0.706472 | 0.013629 |
| `s_attribute` | 0.707102 | 0.034100 |
| `s_marginals` | 0.707718 | 0.022715 |
| `s_raw_node` | 0.720490 | 0.034153 |
| `s_raw_edge` | 0.714249 | 0.018564 |
| `s_raw` | 0.713679 | 0.029931 |
| `s_binding_node` | 0.743746 | 0.036699 |
| `s_binding_edge` | 0.716561 | 0.020621 |
| `s_binding` | 0.739255 | 0.026482 |
| `s_raw_shuffled_0` | 0.710393 | 0.012877 |
| `s_binding_shuffled_0` | 0.679405 | 0.009808 |
| `s_raw_shuffled_1` | 0.699841 | 0.017835 |
| `s_binding_shuffled_1` | 0.658980 | 0.012497 |

## Gates

- `binding_vs_shuffle`: mean delta +0.070063, wins 3/3 — **PASS**
- `binding_vs_marginals`: mean delta +0.031537, wins 3/3 — **PASS**
- `binding_vs_s`: mean delta +0.045399, wins 3/3 — **PASS**
- `binding_vs_raw`: mean delta +0.025576, wins 2/3 — **PASS**
- `raw_vs_shuffle`: mean delta +0.008561, wins 2/3 — **PASS**
- `raw_vs_marginals`: mean delta +0.005961, wins 2/3 — **PASS**
- `raw_vs_s`: mean delta +0.019822, wins 3/3 — **PASS**
- `binding_vs_shuffle_0`: mean delta +0.059851, wins 3/3 — **PASS**
- `raw_vs_shuffle_0`: mean delta +0.003285, wins 2/3 — **PASS**
- `binding_vs_shuffle_1`: mean delta +0.080275, wins 3/3 — **PASS**
- `raw_vs_shuffle_1`: mean delta +0.013837, wins 2/3 — **PASS**

Interpretation: true-vs-shuffle is evidence that the representation contains role–attribute dependence; only a positive binding-vs-marginals (and S+binding-vs-S) increment is evidence that this dependence helps the present MolHIV task.

Decision: **JOINT_AND_CENTERED_BINDING_TASK_INCREMENT_PASS**.

Official test was not encoded or evaluated, and no hyperparameter search was run.
