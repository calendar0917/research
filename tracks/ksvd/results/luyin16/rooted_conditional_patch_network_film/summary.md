# MolHIV Rooted Conditional Patch Network

Protocol: `molhiv-rooted-conditional-patch-network-film-v1`

This is a learnable patch-token screen, not a K-SVD/XGBoost feature sweep.

| variant | mean scaffold-fold AUC | fold AUC |
|---|---:|---|
| `structure` | 0.617102 | 0.612953, 0.580000, 0.658353 |
| `attribute` | 0.668323 | 0.706804, 0.675000, 0.623163 |
| `concat` | 0.633016 | 0.616532, 0.608190, 0.674327 |
| `conditional_film` | 0.583671 | 0.488207, 0.524655, 0.738151 |
| `conditional_film_shuffled` | 0.597137 | 0.601839, 0.568103, 0.621468 |

## Mechanism deltas

- `conditional_film_minus_concat`: mean `-0.049345`, wins `1/3`; fold deltas `[-0.12832491899630777, -0.08353448275862074, 0.06382337427473428]`
- `conditional_film_minus_patch_pair_shuffle`: mean `-0.013466`, wins `1/3`; fold deltas `[-0.1136312259814633, -0.0434482758620689, 0.11668299299223872]`
- `attribute_minus_structure`: mean `+0.051221`, wins `2/3`; fold deltas `[0.09385125461532662, 0.09499999999999997, -0.035189510963755444]`

The conditional model routes a permutation-invariant attribute set through a topology-conditioned expert gate. The shuffled control keeps both patch bags but breaks the same-patch pairing.

Official validation and test were not encoded or evaluated.
