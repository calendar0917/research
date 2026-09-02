# MolHIV Rooted Conditional Patch Network

Protocol: `molhiv-rooted-conditional-patch-network-v1`

This is a learnable patch-token screen, not a K-SVD/XGBoost feature sweep.

| variant | mean scaffold-fold AUC | fold AUC |
|---|---:|---|
| `structure` | 0.617102 | 0.612953, 0.580000, 0.658353 |
| `attribute` | 0.668323 | 0.706804, 0.675000, 0.623163 |
| `concat` | 0.633016 | 0.616532, 0.608190, 0.674327 |
| `conditional` | 0.553425 | 0.551127, 0.500000, 0.609148 |
| `conditional_shuffled` | 0.608825 | 0.597619, 0.629655, 0.599201 |

## Mechanism deltas

- `conditional_minus_concat`: mean `-0.079592`, wins `0/3`; fold deltas `[-0.06540577198402531, -0.10818965517241386, -0.06517971516841237]`
- `true_minus_patch_pair_shuffle`: mean `-0.055400`, wins `1/3`; fold deltas `[-0.046492351744405025, -0.1296551724137931, 0.009946499886971716]`
- `attribute_minus_structure`: mean `+0.051221`, wins `2/3`; fold deltas `[0.09385125461532662, 0.09499999999999997, -0.035189510963755444]`

The conditional model routes a permutation-invariant attribute set through a topology-conditioned expert gate. The shuffled control keeps both patch bags but breaks the same-patch pairing.

Official validation and test were not encoded or evaluated.
