# luyin16 fixed-feature block ablation

Source protocol: `mentor-concept-replication-v1-k32-official-valid`

All models use the frozen classifier settings and official validation only; test is not evaluated.

| view | dim | train ROC-AUC | valid ROC-AUC |
|---|---:|---:|---:|
| topology | 18 | 0.877038 ± 0.001309 | 0.683113 ± 0.010974 |
| atom_composition | 174 | 0.963067 ± 0.001039 | 0.706600 ± 0.006600 |
| bond_composition | 13 | 0.885889 ± 0.001246 | 0.599344 ± 0.006668 |
| chemistry_composition | 187 | 0.966519 ± 0.001128 | 0.719637 ± 0.007759 |
| s | 205 | 0.974530 ± 0.000711 | 0.781696 ± 0.006559 |
| topology_t_init | 346 | 0.992749 ± 0.000433 | 0.720479 ± 0.008959 |
| topology_t_final | 346 | 0.991556 ± 0.000545 | 0.699555 ± 0.017278 |
| chemistry_t_init | 515 | 0.991038 ± 0.000605 | 0.729039 ± 0.005623 |
| chemistry_t_final | 515 | 0.989091 ± 0.000673 | 0.727078 ± 0.006669 |
| s_t_init | 533 | 0.992522 ± 0.000137 | 0.771681 ± 0.010280 |
| s_t_final | 533 | 0.991461 ± 0.000539 | 0.762350 ± 0.008964 |

Best by validation only: `s`.

Interpretation gates:

- `topology` measures explicit structure without atom/bond categories.
- `chemistry_composition` measures atom/bond categories without explicit topology statistics.
- `*_t_final - *_t_init` attributes K-SVD updates under the same downstream model.
- `chemistry_t_final - chemistry_composition` tests dictionary information beyond composition.
