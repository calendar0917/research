# MolHIV invariant patch mechanism screen

Protocol: `luyin16-molhiv-invariant-patch-mechanism-screen-v1`

## Object gate

- invariant object: **PASS**
- maximum topology drift: 0
- maximum attribute drift: 0
- maximum joint drift: 0
- maximum full ego size observed: 19
- atom/bond categories use direct indices; no modulo collisions

## RAW official-train scaffold screen

| view | mean fold AUC |
|---|---:|
| `s` | 0.693865 |
| `invariant_topology_raw` | 0.664597 |
| `invariant_attributes_raw` | 0.735942 |
| `invariant_joint_raw` | 0.712774 |
| `s_invariant_topology_raw` | 0.705055 |
| `s_invariant_attributes_raw` | 0.724819 |
| `s_invariant_joint_raw` | 0.720513 |

`S+joint RAW - S` mean delta: +0.026647; wins: 2/3.
`S+attributes RAW - S`: +0.030953; wins: 2/3.
`S+topology RAW - S`: +0.011189; wins: 2/3.
Standalone `joint RAW - attributes RAW`: -0.023169; wins: 0/3.
RAW promotion: **PASS**.

## Fold-specific K-SVD attribution

| view | mean fold AUC |
|---|---:|
| `invariant_joint_init` | 0.736775 |
| `invariant_joint_final` | 0.734187 |
| `s_invariant_joint_init` | 0.748289 |
| `s_invariant_joint_final` | 0.740832 |
| `s_invariant_joint_final_residual` | 0.744841 |

`S+FINAL - S+INIT` mean delta: -0.007457; wins: 1/3.
`S+INIT - S+RAW`: +0.027776; wins: 3/3.
`S+FINAL - S+RAW`: +0.020319; wins: 2/3.
Validation reconstruction-error reductions: 26.3%, 20.8%, 20.5%.
K-SVD task update: **FAIL**.

## Interpretation

The stable signal is an all-center, shell-conditioned local chemistry distribution. Pure local topology is weaker, and naively joint-normalizing topology with attributes can dilute the attribute signal.

Sparse reconstruction under the empirical INIT dictionary is useful, but optimizing the unsupervised reconstruction objective does not improve the task consistently. K-SVD therefore remains a compressor/diagnostic in this protocol, not the source of the MolHIV gain.

## Boundary

No official validation label was used for supervised scoring, official test was not encoded or evaluated, and no hyperparameter search was run.
